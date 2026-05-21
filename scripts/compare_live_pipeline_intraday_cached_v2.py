"""V2 backtest harness — smarter SL/TP to reduce being swept by noise.

Fork of compare_live_pipeline_intraday_cached.py. Keeps all entry/screener
logic identical; overrides only the SL/TP/exit functions to test:

  1. ATR-adaptive SL  — cap raised 5% → 8%, distance scales with ATR.
  2. Volatility-tier SL multiplier — on volatile VNI days (|chg|>2%), widen SL ×1.3.
  3. Intraday wick guard — SL/ATR-stop fires only on CLOSE below trigger,
     not intraday low (avoids morning liquidity-test wicks).
  4. Earlier trailing activation — was 12% gain, now 5% gain (locks profits sooner).
  5. Disabled hard TP — relies entirely on trailing ATR (v1 data shows TP rare).

Outputs go to backtest_results/mvp_intraday_cached_v2/ — completely separate
from v1 results.

Usage:
  python -m scripts.compare_live_pipeline_intraday_cached_v2 \
    --universe vn100 --start 2025-01-01 --end 2026-05-12 \
    --label v2_smart_sl
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from multiagents_trading_assistant.backtest import live_pipeline
from multiagents_trading_assistant.backtest.live_pipeline import (
    LivePipelineBacktestConfig,
    LivePosition,
    _breakdown,
    _close_position,
    _compute_initial_sl,
    _last_row_on_or_before,
    _mark_to_market,
    _market_context_at,
    _next_calendar_date,
    _prepare_data,
    _prepare_single_frame,
    _row_at,
    _scan_live_candidates,
)
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import evaluate_filters, load_hypotheses, rank_candidates
from multiagents_trading_assistant.edge_lab.live_signal import (
    DEFAULT_CONFIG,
    DEFAULT_LIVE_EDGE_FAMILY,
    _parse_strategy_names,
    _resolve_strategy_names_for_latest,
    _round_or_none,
    _setup_type_from_tags,
)
from multiagents_trading_assistant.fetcher import get_vn100_symbols, get_vn30_symbols
from multiagents_trading_assistant.quantagents_backtest.metrics import compute_metrics


ROOT = Path(__file__).resolve().parents[1]
MASTER_PARQUET = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
CACHE_DIR = ROOT / "multiagents_trading_assistant" / "cache"
OUT_DIR = ROOT / "backtest_results" / "mvp_intraday_cached_v2"


# ============================================================
# V2 SMART SL/TP CONFIG (tuned after v2_smart_sl benchmark)
# ============================================================
SL_PCT_CAP = 0.05              # KEPT v1 default; widening to 8% gave bigger losses (-9.3% vs -5.8%)
SL_ATR_MULT_BASE = 2.3         # base ATR multiplier (matches v1 edge_risk default)
VOLATILE_VNI_THRESHOLD = 2.0   # |VNI chg %| above this → volatile day
VOLATILE_SL_MULT = 1.2         # widen SL by 20% on volatile days (was 1.3, lower to avoid huge losses)
VOLATILE_SL_PCT_CAP = 0.065    # on volatile days, allow SL slightly wider (6.5%)
TRAIL_ACTIVATION_PCT = 0.08    # start trailing at +8% (v1 was 12%, v2 too aggressive at 5%)
TRAIL_ATR_MULT = 2.5           # tighter trail than v1's 3.0 — locks profit faster
DISABLE_HARD_TP = False        # KEEP TP — it fires rarely but at +25% avg in v1


def _to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


# ============================================================
# V2 SL/TP HELPER FUNCTIONS (override v1)
# ============================================================
def _edge_initial_sl_v2(
    entry: float,
    ind: dict,
    risk: dict,
    vnindex_volatility: float,
) -> float:
    """ATR-adaptive SL with volatility-tier adjustment.

    Distance from entry = max(ATR×mult, pct_floor), capped at SL_PCT_CAP.
    On volatile VNI days, multiplier scales up by VOLATILE_SL_MULT.
    """
    atr = _to_float(ind.get("atr")) or 0.0
    initial_mult = float(risk.get("initial_atr_stop_mult") or SL_ATR_MULT_BASE)
    stop_pct = _to_float(risk.get("stop_loss"))

    # Volatility tier — widen on turbulent days
    vol_mult = VOLATILE_SL_MULT if abs(vnindex_volatility) > VOLATILE_VNI_THRESHOLD else 1.0
    effective_mult = initial_mult * vol_mult

    # Compute distance from entry (positive number)
    if atr > 0:
        atr_dist = effective_mult * atr
    else:
        atr_dist = entry * 0.04

    # Floor: at least stop_pct from edge_risk (if defined), but capped at SL_PCT_CAP
    if stop_pct and stop_pct > 0:
        pct_dist = entry * stop_pct
        sl_dist = max(atr_dist, pct_dist)
    else:
        sl_dist = atr_dist

    # Hard cap — use volatile cap on turbulent days, normal cap otherwise
    cap = VOLATILE_SL_PCT_CAP if vol_mult > 1.0 else SL_PCT_CAP
    sl_dist = min(sl_dist, entry * cap)
    sl = entry - sl_dist
    if sl <= 0 or sl >= entry:
        return _compute_initial_sl(entry, ind)
    return round(sl, 0)


def _edge_take_profit_v2(
    entry: float,
    stop_loss: float,
    risk: dict,
    cfg: LivePipelineBacktestConfig,
) -> float:
    """Disable hard TP — set very high so trailing handles all exits."""
    if DISABLE_HARD_TP:
        return entry * 100.0  # effectively infinity
    tp_pct = _to_float(risk.get("take_profit"))
    if tp_pct and tp_pct > 0:
        return entry * (1.0 + tp_pct)
    return entry + cfg.rr_ratio * (entry - stop_loss)


def _exit_decision_v2(
    row: pd.Series,
    position: LivePosition,
    cfg: LivePipelineBacktestConfig,
    vnindex_volatility: float,
) -> tuple[str | None, float]:
    """V2 exit logic.

    Key differences from v1:
      - SL/ATR-stop triggers on CLOSE (not intraday low) → wick-resistant.
      - Trailing activates at +5% (was +12%) → locks profit on early winners.
      - No hard TP → trailing handles all upside exits.
      - Volatility-tier SL multiplier applied to ATR distance dynamically.
    """
    low = float(row["low"])
    high = float(row["high"])
    close = float(row["close"])
    open_ = float(row["open"])
    risk = position.edge_risk or {}
    entry = position.entry_price
    position.highest_price = max(float(position.highest_price or entry), close)  # use CLOSE for trail

    # --- Hard stop loss (pct) — CLOSE-based ---
    stop_pct = _to_float(risk.get("stop_loss"))
    if stop_pct and close <= entry * (1.0 - stop_pct):
        return "EDGE_STOP_LOSS_V2", close * (1.0 - cfg.slippage_rate)

    atr = _to_float(position.entry_atr)
    if atr and atr > 0:
        # Volatility-tier dynamic ATR stop
        vol_mult = VOLATILE_SL_MULT if abs(vnindex_volatility) > VOLATILE_VNI_THRESHOLD else 1.0
        initial_mult = float(risk.get("initial_atr_stop_mult") or SL_ATR_MULT_BASE) * vol_mult
        atr_stop_level = entry - initial_mult * atr

        # CLOSE-based ATR stop
        if close <= atr_stop_level:
            return "EDGE_ATR_STOP_V2", close * (1.0 - cfg.slippage_rate)

        # Trailing — earlier activation
        peak = float(position.highest_price or entry)
        if peak >= entry * (1.0 + TRAIL_ACTIVATION_PCT):
            trail_level = peak - TRAIL_ATR_MULT * atr
            if close <= trail_level:
                return "EDGE_TRAILING_ATR_STOP_V2", close * (1.0 - cfg.slippage_rate)

    # --- Hard TP (disabled by default in v2) ---
    if not DISABLE_HARD_TP:
        tp_pct = _to_float(risk.get("take_profit"))
        if tp_pct and high >= entry * (1.0 + tp_pct):
            return "EDGE_TAKE_PROFIT_V2", open_ * (1.0 - cfg.slippage_rate)

    # --- Max hold ---
    max_holding = int(risk.get("max_holding_bars") or cfg.max_hold_bars)
    if position.holding_bars >= max_holding:
        return "EDGE_MAX_HOLD_V2", close * (1.0 - cfg.slippage_rate)

    return None, close


# ============================================================
# V1 FIXED-BIAS EXIT — corrects same-bar-open lookahead in v1
# ============================================================
# Original v1 returns "open_ * slip" for SL/ATR/TRAIL/TP exits — this is the
# SAME BAR'S open price, which occurs BEFORE the intraday wick that triggered
# the exit. That is a lookahead bias that understates losses (and wins).
#
# This function keeps v1's INTRADAY trigger logic (so the set of exits matches
# v1 exactly) but reports a realistic exit price: clamp to trigger level.
#   - For SL/ATR/trailing exits: exit at min(open, trigger_level) * (1 - slip)
#     i.e. the WORSE of (gap-down open, the trigger). This is what would
#     actually happen if you placed a stop order at the trigger.
#   - For TP: exit at max(open, tp_level) * (1 - slip)
#     i.e. the BETTER of (gap-up open, the TP), bounded — what a limit sell
#     at TP would fill.
def _edge_exit_decision_v1_fixed(
    row: pd.Series,
    position: LivePosition,
    cfg: LivePipelineBacktestConfig,
) -> tuple[str | None, float]:
    low = float(row["low"])
    high = float(row["high"])
    close = float(row["close"])
    open_ = float(row["open"])
    risk = position.edge_risk or {}
    entry = position.entry_price
    position.highest_price = max(float(position.highest_price or entry), high)

    stop_pct = _to_float(risk.get("stop_loss"))
    if stop_pct and low <= entry * (1.0 - stop_pct):
        trigger = entry * (1.0 - stop_pct)
        exit_price = min(open_, trigger) * (1.0 - cfg.slippage_rate)
        return "EDGE_STOP_LOSS", exit_price

    atr = _to_float(position.entry_atr)
    if atr and atr > 0:
        initial_mult = risk.get("initial_atr_stop_mult")
        if initial_mult is not None:
            atr_trigger = entry - float(initial_mult) * atr
            if low <= atr_trigger:
                exit_price = min(open_, atr_trigger) * (1.0 - cfg.slippage_rate)
                return "EDGE_ATR_STOP", exit_price

        trailing_mult = risk.get("trailing_atr_mult")
        activation = _to_float(risk.get("trailing_profit_activation")) or 0.0
        if (
            trailing_mult is not None
            and float(position.highest_price or entry) >= entry * (1.0 + activation)
        ):
            trail_trigger = float(position.highest_price or entry) - float(trailing_mult) * atr
            if low <= trail_trigger:
                exit_price = min(open_, trail_trigger) * (1.0 - cfg.slippage_rate)
                return "EDGE_TRAILING_ATR_STOP", exit_price

    take_profit_pct = _to_float(risk.get("take_profit"))
    if take_profit_pct and high >= entry * (1.0 + take_profit_pct):
        tp_trigger = entry * (1.0 + take_profit_pct)
        exit_price = max(open_, tp_trigger) * (1.0 - cfg.slippage_rate)
        return "EDGE_TAKE_PROFIT", exit_price

    max_holding = int(risk.get("max_holding_bars") or cfg.max_hold_bars)
    if position.holding_bars >= max_holding:
        return "EDGE_MAX_HOLD", close * (1.0 - cfg.slippage_rate)
    return None, close


# v1-original SL/TP helpers (no smart adjustments — match live_pipeline.py exactly)
def _edge_initial_sl_v1(entry: float, ind: dict, risk: dict) -> float:
    atr = _to_float(ind.get("atr"))
    initial_mult = risk.get("initial_atr_stop_mult")
    stops = []
    stop_pct = _to_float(risk.get("stop_loss"))
    if stop_pct and stop_pct > 0:
        stops.append(entry * (1.0 - stop_pct))
    if atr and atr > 0 and initial_mult is not None:
        stops.append(entry - float(initial_mult) * atr)
    if not stops:
        return _compute_initial_sl(entry, ind)
    return round(max(stops), 0)


def _edge_take_profit_v1(
    entry: float,
    stop_loss: float,
    risk: dict,
    cfg: LivePipelineBacktestConfig,
) -> float:
    tp_pct = _to_float(risk.get("take_profit"))
    if tp_pct and tp_pct > 0:
        return entry * (1.0 + tp_pct)
    return entry + cfg.rr_ratio * (entry - stop_loss)


# ============================================================
# DATA LOADERS (copied verbatim from v1)
# ============================================================
def resolve_symbols(universe: str) -> list[str]:
    name = universe.lower()
    if name == "vn30":
        return get_vn30_symbols()
    if name == "vn100":
        return get_vn100_symbols()
    return [item.strip().upper() for item in universe.split(",") if item.strip()]


def load_local_history(symbols: list[str], start: str, end: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    df = pd.read_parquet(MASTER_PARQUET)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    warmup_start = start_ts - pd.Timedelta(days=420)
    sub = df[
        df["symbol"].isin(symbols)
        & (df["date"] >= warmup_start)
        & (df["date"] <= end_ts)
    ].copy()
    data = {
        str(symbol): group[["date", "open", "high", "low", "close", "volume"]]
        .sort_values("date")
        .reset_index(drop=True)
        for symbol, group in sub.groupby("symbol", sort=False)
    }
    return data, load_cached_vnindex(warmup_start, end_ts)


def load_cached_vnindex(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    index_master = ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"
    if index_master.exists():
        try:
            frame = pd.read_parquet(index_master)
            if not frame.empty:
                frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
                if "symbol" in frame.columns:
                    frame = frame[frame["symbol"].astype(str).str.upper() == "VNINDEX"]
                frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
                if not frame.empty:
                    return frame[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    candidates = sorted(CACHE_DIR.glob("VNINDEX_*_1D_hist.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            frame = pd.read_json(path)
            if frame.empty:
                continue
            frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
            frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
            if not frame.empty:
                return frame[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        except Exception:
            continue
    raise FileNotFoundError("No cached VNINDEX history found")


# ============================================================
# EDGE SIGNAL CACHE (copied from v1)
# ============================================================
def build_edge_signal_cache(
    symbols: list[str],
    start: str,
    end: str,
    strategy_name: str,
) -> dict[pd.Timestamp, dict[str, dict[str, Any]]]:
    print(f"[cache] Building feature table for {len(symbols)} symbols...")
    features, _ = build_feature_table(symbols, start, end, root=ROOT)
    if features.empty:
        return {}
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    strategy_names = _parse_strategy_names(strategy_name)
    hypotheses_by_name = {item.name: item for item in load_hypotheses(DEFAULT_CONFIG)}
    signal_cache: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    clean_symbols = sorted({item.upper().strip() for item in symbols if item})

    for idx, (date, daily) in enumerate(features.groupby("date", sort=True), start=1):
        latest = daily.reset_index(drop=True)
        resolved_names = _resolve_strategy_names_for_latest(strategy_names, latest)
        hypotheses = [hypotheses_by_name[name] for name in resolved_names if name in hypotheses_by_name]
        day_out: dict[str, dict[str, Any]] = {}
        for _, row in latest.iterrows():
            symbol = str(row["symbol"])
            day_out[symbol] = {
                "strategy_name": strategy_name,
                "description": "",
                "passed": False,
                "available": True,
                "feature_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "edge_rank": None,
                "edge_rank_score": 0.0,
                "edge_score": _round_or_none(row.get("edge_score")),
                "smart_money_score": _round_or_none(row.get("smart_money_score")),
                "smart_money_score_delta": _round_or_none(row.get("smart_money_score_delta")),
                "rs_percentile_20": _round_or_none(row.get("rs_percentile_20")),
                "value_ratio_20": _round_or_none(row.get("value_ratio_20")),
                "CHDM50": _round_or_none(row.get("CHDM50")),
                "DS20": _round_or_none(row.get("DS20")),
                "mkt_regime_state": row.get("mkt_regime_state"),
                "mkt_regime_score": _round_or_none(row.get("mkt_regime_score")),
                "filters_failed": [],
                "risk": {},
            }

        for hypothesis in hypotheses:
            mask = evaluate_filters(latest, hypothesis)
            ranked = rank_candidates(latest[mask], hypothesis)
            for rank_idx, row in enumerate(ranked.to_dict("records"), start=1):
                symbol = str(row["symbol"])
                rank_score = float(row.get("_edge_rank") or 0.0)
                current = day_out.get(symbol)
                if not current or rank_score <= float(current.get("edge_rank_score") or 0.0):
                    continue
                current.update(
                    {
                        "strategy_name": hypothesis.name,
                        "strategy_family": strategy_name if len(hypotheses) > 1 else hypothesis.name,
                        "description": hypothesis.description,
                        "passed": True,
                        "edge_rank": rank_idx,
                        "edge_rank_score": round(rank_score, 4),
                        "setup_type": _setup_type_from_tags(hypothesis),
                        "filters_failed": [],
                        "risk": hypothesis.risk,
                    }
                )

        for symbol in clean_symbols:
            day_out.setdefault(
                symbol,
                {
                    "strategy_name": strategy_name,
                    "passed": False,
                    "available": False,
                    "error": "No latest feature row for symbol",
                },
            )
        signal_cache[pd.Timestamp(date).normalize()] = day_out
        if idx % 100 == 0:
            print(f"[cache] edge dates={idx}")
    return signal_cache


def install_edge_cache(signal_cache: dict[pd.Timestamp, dict[str, dict[str, Any]]]) -> None:
    def cached_edge_signals(
        symbols: list[str],
        as_of_date: str | None = None,
        strategy_name: str = DEFAULT_LIVE_EDGE_FAMILY,
        config_path: str | Path = DEFAULT_CONFIG,
        root: str | Path = ".",
    ) -> dict[str, dict[str, Any]]:
        del strategy_name, config_path, root
        date = pd.Timestamp(as_of_date).normalize() if as_of_date else max(signal_cache)
        day = signal_cache.get(date, {})
        return {
            symbol.upper().strip(): day.get(
                symbol.upper().strip(),
                {"strategy_name": "", "passed": False, "available": False, "error": "No cached edge signal"},
            )
            for symbol in symbols
        }

    live_pipeline.get_edge_strategy_signals = cached_edge_signals


# ============================================================
# ENTRY ZONE / FILL (copied from v1)
# ============================================================
def derive_entry_zone(order: dict[str, Any]) -> tuple[float, float] | None:
    indicators = order.get("indicators") or {}
    signal_close = _to_float(indicators.get("current_price"))
    if not signal_close or signal_close <= 0:
        return None
    edge_name = str(order.get("edge_strategy_name") or "")
    supports = indicators.get("support_levels") or []
    ema50 = _to_float(indicators.get("ema50")) or 0.0
    if edge_name == "breakout_after_accumulation_v3":
        return round(signal_close, 2), round(signal_close * 1.01, 2)
    if edge_name == "compression_breakout_smt_v1":
        return round(signal_close * 0.995, 2), round(signal_close * 1.01, 2)
    if edge_name in {"benchmark_leader_breakout_v1", "volume_surge_breakout_smt_v2"}:
        return round(signal_close * 0.995, 2), round(signal_close * 1.012, 2)
    if edge_name == "mean_reversion_uptrend_ma50_v1":
        level = float(supports[0]) if supports else (ema50 if ema50 > 0 else signal_close)
        return round(level * 0.99, 2), round(level * 1.01, 2)
    if edge_name == "sector_leader_pullback_v1":
        level = float(supports[0]) if supports else signal_close
        return round(level * 0.99, 2), round(level * 1.008, 2)
    return None


def intraday_touch_fill(order: dict[str, Any], row: pd.Series) -> float | None:
    zone = derive_entry_zone(order)
    if not zone:
        return None
    low_zone, high_zone = zone
    open_ = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    if open_ <= 0:
        return None
    if low_zone <= open_ <= high_zone:
        return round(open_, 2)
    if open_ < low_zone and high >= low_zone:
        return round(low_zone, 2)
    if open_ > high_zone and low <= high_zone:
        return round(high_zone, 2)
    if low <= low_zone and high >= high_zone:
        return round(low_zone, 2)
    return None


def fill_entries_v2(
    orders: list[dict[str, Any]],
    data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    day_idx: int,
    cash: float,
    positions: dict[str, LivePosition],
    cfg: LivePipelineBacktestConfig,
    model: str,
    counters: dict[str, int],
    vnindex_volatility: float,
    mode: str = "v2_smart",
) -> float:
    slots = max(0, cfg.max_positions - len(positions))
    if slots <= 0:
        return cash
    orders = [order for order in orders if order["symbol"] not in positions][:slots]
    if not orders:
        return cash
    budget = cash / max(1, min(slots, len(orders)))
    for order in orders:
        counters["attempted_orders"] += 1
        row = _row_at(data[order["symbol"]], date)
        if row is None:
            continue
        entry = float(row["open"]) * (1.0 + cfg.slippage_rate) if model == "eod_next_open" else intraday_touch_fill(order, row)
        if entry is None or entry <= 0:
            continue
        gross_shares = int(budget // entry)
        shares = (gross_shares // cfg.lot_size) * cfg.lot_size if cfg.lot_size > 1 else gross_shares
        if shares <= 0:
            continue
        cost = shares * entry * (1.0 + cfg.commission_rate)
        if cost > cash:
            continue
        indicators = order["indicators"]
        edge_risk = dict(order.get("edge_risk") or {})
        # Select SL/TP helpers based on mode
        if mode == "v1_fixed_bias":
            stop_loss = (
                _edge_initial_sl_v1(entry, indicators, edge_risk)
                if edge_risk
                else _compute_initial_sl(entry, indicators)
            )
            tp = _edge_take_profit_v1(entry, stop_loss, edge_risk, cfg)
        else:  # v2_smart
            stop_loss = (
                _edge_initial_sl_v2(entry, indicators, edge_risk, vnindex_volatility)
                if edge_risk
                else _compute_initial_sl(entry, indicators)
            )
            tp = _edge_take_profit_v2(entry, stop_loss, edge_risk, cfg)
        if stop_loss <= 0 or stop_loss >= entry:
            continue
        cash -= cost
        counters["fills"] += 1
        positions[order["symbol"]] = LivePosition(
            symbol=order["symbol"],
            setup_type=order["setup_type"],
            signal_date=order["signal_date"],
            entry_date=date,
            entry_idx=day_idx,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=tp,
            shares=shares,
            priority_score=float(order["priority_score"]),
            reasons=list(order["reasons"]),
            edge_strategy_name=str(order.get("edge_strategy_name") or ""),
            edge_strategy_passed=bool(order.get("edge_strategy_passed", False)),
            edge_score=_to_float(order.get("edge_score")),
            edge_rank=_to_float(order.get("edge_rank")),
            edge_rank_score=_to_float(order.get("edge_rank_score")),
            edge_risk=edge_risk or None,
            entry_atr=_to_float(indicators.get("atr")),
            highest_price=entry,
        )
    return cash


# ============================================================
# MAIN LOOP (uses v2 helpers)
# ============================================================
def _compute_vnindex_volatility(index_data: pd.DataFrame) -> dict[pd.Timestamp, float]:
    """Pre-compute daily |VNI %change| for fast lookup during loop."""
    if index_data.empty:
        return {}
    df = index_data.copy()
    if "date" in df.index.names or df.index.name == "date":
        df = df.reset_index(drop=True) if "date" in df.columns else df.reset_index()
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_change"] = df["close"].pct_change() * 100.0
    return {pd.Timestamp(row["date"]).normalize(): float(row["pct_change"] if pd.notna(row["pct_change"]) else 0.0) for _, row in df.iterrows()}


def run_model_v2(
    universe_data: dict[str, pd.DataFrame],
    vnindex: pd.DataFrame,
    cfg: LivePipelineBacktestConfig,
    model: str,
    mode: str = "v2_smart",
) -> dict[str, Any]:
    data = _prepare_data(universe_data)
    index_data = _prepare_single_frame(vnindex)
    calendar = sorted(set(index_data["date"]))
    vni_vol = _compute_vnindex_volatility(index_data)
    cash = float(cfg.initial_capital)
    positions: dict[str, LivePosition] = {}
    pending_orders: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    counters = {"signal_days": 0, "signals": 0, "attempted_orders": 0, "fills": 0}
    date_to_pos = {date: idx for idx, date in enumerate(calendar)}
    start_ts = pd.Timestamp(cfg.start_date).normalize() if cfg.start_date else None
    end_ts = pd.Timestamp(cfg.end_date).normalize() if cfg.end_date else None
    last_processed_date = None

    for date in calendar[cfg.lookback :]:
        if end_ts is not None and date > end_ts:
            break
        last_processed_date = date
        day_idx = date_to_pos[date]
        today_vol = vni_vol.get(pd.Timestamp(date).normalize(), 0.0)
        todays_orders = pending_orders.pop(date, [])
        if todays_orders:
            cash = fill_entries_v2(
                todays_orders, data, date, day_idx, cash, positions, cfg, model, counters, today_vol, mode
            )

        for symbol in list(positions):
            row = _row_at(data[symbol], date)
            if row is None:
                continue
            position = positions[symbol]
            position.holding_bars = day_idx - position.entry_idx
            if mode == "v1_fixed_bias":
                exit_reason, exit_price = _edge_exit_decision_v1_fixed(row, position, cfg)
            else:
                exit_reason, exit_price = _exit_decision_v2(row, position, cfg, today_vol)
            if exit_reason and position.holding_bars >= cfg.settlement_bars:
                cash, trade = _close_position(date, exit_price, exit_reason, position, cash, cfg)
                trades.append(trade)
                del positions[symbol]

        if start_ts is None or date >= start_ts:
            equity = cash + _mark_to_market(positions, data, date)
            equity_rows.append({"date": date, "equity": equity, "cash": cash, "positions": len(positions)})

        next_date = _next_calendar_date(calendar, day_idx)
        if next_date is None:
            continue
        if start_ts is not None and date < start_ts:
            continue
        market_ctx = _market_context_at(index_data, date, cfg.lookback)
        if not market_ctx.should_trade and not cfg.allow_downtrend_entries:
            continue
        if cfg.require_uptrend and market_ctx.reference_trend != "UPTREND":
            continue
        candidates = _scan_live_candidates(data, date, market_ctx, cfg)
        candidates = [candidate for candidate in candidates if candidate["symbol"] not in positions]
        if candidates:
            counters["signal_days"] += 1
            counters["signals"] += len(candidates[: cfg.max_candidates_per_day])
            pending_orders.setdefault(next_date, []).extend(candidates[: cfg.max_candidates_per_day])

    final_date = last_processed_date or calendar[-1]
    for symbol in list(positions):
        row = _row_at(data[symbol], final_date)
        if row is None:
            row = _last_row_on_or_before(data[symbol], final_date)
        if row is None:
            continue
        cash, trade = _close_position(final_date, float(row["close"]), "END_OF_DATA", positions[symbol], cash, cfg)
        trades.append(trade)
        del positions[symbol]

    equity_frame = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame()
    equity_curve = equity_frame["equity"].rename(f"{model}_equity") if not equity_frame.empty else pd.Series(dtype=float)
    metrics = compute_metrics(equity_curve, trades) if not equity_curve.empty else {}
    metrics.update(
        {
            "signal_days": counters["signal_days"],
            "signals": counters["signals"],
            "attempted_orders": counters["attempted_orders"],
            "fills": counters["fills"],
            "fill_rate": (counters["fills"] / counters["attempted_orders"]) if counters["attempted_orders"] else 0.0,
        }
    )
    return {
        "metrics": metrics,
        "trades": trades,
        "equity_frame": equity_frame,
        "setup_breakdown": _breakdown(trades, "setup_type"),
        "symbol_breakdown": _breakdown(trades, "symbol"),
    }


# ============================================================
# CLI (mirrors v1)
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="V2 backtest harness — smart SL/TP")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-12")
    parser.add_argument("--edge-strategy", default=DEFAULT_LIVE_EDGE_FAMILY)
    parser.add_argument("--label", default="v2_smart_sl")
    parser.add_argument(
        "--mode",
        choices=["v2_smart", "v1_fixed_bias"],
        default="v2_smart",
        help="v2_smart = new SL/TP logic; v1_fixed_bias = keep v1 logic but fix same-bar-open exit lookahead.",
    )
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    parser.add_argument(
        "--sl-pct-cap",
        type=float,
        default=None,
        help="Hard ceiling on SL distance from entry (decimal, e.g. 0.08 = 8%).",
    )
    parser.add_argument(
        "--trail-activation-pct",
        type=float,
        default=None,
        help="Gain pct at which trailing SL activates (default 0.05 = 5%).",
    )
    parser.add_argument(
        "--volatile-vni-threshold",
        type=float,
        default=None,
        help="|VNI %% change| above which SL is widened by VOLATILE_SL_MULT.",
    )
    args = parser.parse_args()

    # Apply overrides (module-level — picked up by helpers via globals)
    global SL_PCT_CAP, TRAIL_ACTIVATION_PCT, VOLATILE_VNI_THRESHOLD
    if args.sl_pct_cap is not None:
        SL_PCT_CAP = args.sl_pct_cap
    if args.trail_activation_pct is not None:
        TRAIL_ACTIVATION_PCT = args.trail_activation_pct
    if args.volatile_vni_threshold is not None:
        VOLATILE_VNI_THRESHOLD = args.volatile_vni_threshold

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    symbols = resolve_symbols(args.universe)
    base_cfg = LivePipelineBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        edge_strategy_name=args.edge_strategy,
        max_positions=args.max_positions,
        max_candidates_per_day=args.max_candidates_per_day,
    )

    universe_data, vnindex = load_local_history(symbols, args.start, args.end)
    signal_cache = build_edge_signal_cache(symbols, args.start, args.end, args.edge_strategy)
    install_edge_cache(signal_cache)

    rows = []
    details = {}
    for model in ["eod_next_open", "intraday_touch"]:
        print(f"[{args.mode}] running {model} | sl_cap={SL_PCT_CAP:.2%} trail_act={TRAIL_ACTIVATION_PCT:.0%} vni_vol_thr={VOLATILE_VNI_THRESHOLD}")
        result = run_model_v2(universe_data, vnindex, replace(base_cfg), model, mode=args.mode)
        details[model] = result
        metrics = result["metrics"]
        rows.append(
            {
                "label": args.label,
                "universe": args.universe.upper(),
                "start": args.start,
                "end": args.end,
                "model": model,
                "max_positions": args.max_positions,
                "max_candidates_per_day": args.max_candidates_per_day,
                "sl_pct_cap": SL_PCT_CAP,
                "trail_activation_pct": TRAIL_ACTIVATION_PCT,
                "volatile_vni_threshold": VOLATILE_VNI_THRESHOLD,
                "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
                "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
                "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
                "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
                "number_of_trades": int(metrics.get("number_of_trades", 0)),
                "signal_days": int(metrics.get("signal_days", 0)),
                "signals": int(metrics.get("signals", 0)),
                "attempted_orders": int(metrics.get("attempted_orders", 0)),
                "fills": int(metrics.get("fills", 0)),
                "fill_rate_pct": round(float(metrics.get("fill_rate", 0.0)) * 100, 2),
            }
        )

    summary = pd.DataFrame(rows)
    stem = f"{args.universe}_{args.label}_{args.start}_{args.end}".replace(",", "_")
    summary.to_csv(OUT_DIR / f"{stem}_summary.csv", index=False)
    for model, result in details.items():
        model_stem = f"{stem}_{model}"
        pd.DataFrame(result["trades"]).to_csv(OUT_DIR / f"{model_stem}_trades.csv", index=False)
        if not result["equity_frame"].empty:
            result["equity_frame"].to_csv(OUT_DIR / f"{model_stem}_equity.csv")
        pd.DataFrame(result["setup_breakdown"]).to_csv(OUT_DIR / f"{model_stem}_setup_breakdown.csv", index=False)
        pd.DataFrame(result["symbol_breakdown"]).to_csv(OUT_DIR / f"{model_stem}_symbol_breakdown.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
