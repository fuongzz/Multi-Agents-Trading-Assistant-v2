"""V3 backtest harness — entry quality filter + price-action exit.

Uses v1's PATCHED exit logic (bias-fixed) as the base, then adds:

  ENTRY FILTERS (C):
    - Drop candidates whose 5-bar return > POST_EXTENSION_THRESHOLD (default 8%):
      these are already extended; entry tomorrow is poor R:R.
    - Drop candidates with edge_score below cohort median per signal day.
      Cohort = candidates surviving from same day.

  EXIT PRICE-ACTION (D):
    - After at least +5% gain, exit if close < MA20 for 2 consecutive bars
      AND red-bar volume > green-bar volume over last 3 bars.
    - Fires BEFORE v1 trailing trigger, so it can lock profit earlier on
      structural breakdown instead of waiting for ATR trail level.

V1 patched exit (SL/ATR/TRAIL/TP) still applies as fallback.

Outputs to backtest_results/mvp_intraday_cached_v3/.
"""

from __future__ import annotations

import argparse
import statistics
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
    _edge_exit_decision,
    _edge_initial_sl,
    _edge_take_profit,
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
OUT_DIR = ROOT / "backtest_results" / "mvp_intraday_cached_v3"


# ============================================================
# V3 KNOBS
# ============================================================
POST_EXTENSION_THRESHOLD = 0.08   # drop entries where 5-bar return > 8%
EDGE_SCORE_PERCENTILE_GATE = 0.50  # keep only above-median per cohort
PA_EXIT_MIN_PNL = 0.05            # only PA exit after +5% profit
PA_EXIT_MA20_BARS_BELOW = 2       # close < MA20 for N consecutive bars


def _to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


# ============================================================
# ENTRY QUALITY FILTER
# ============================================================
def filter_quality_candidates(
    candidates: list[dict[str, Any]],
    data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Apply post-extension drop + edge_score cohort median gate."""
    if not candidates:
        return candidates

    survived: list[dict[str, Any]] = []
    for c in candidates:
        symbol = c["symbol"]
        if symbol not in data:
            continue
        df = data[symbol]
        # 5-bar return: today's close vs close 5 bars ago
        sub = df[df["date"] <= date].tail(6)
        if len(sub) < 6:
            continue
        ret_5d = (float(sub.iloc[-1]["close"]) - float(sub.iloc[0]["close"])) / float(sub.iloc[0]["close"])
        if ret_5d > POST_EXTENSION_THRESHOLD:
            continue
        survived.append(c)

    if not survived:
        return survived

    # Cohort median gate on edge_score
    edge_scores = [_to_float(c.get("edge_score")) for c in survived]
    edge_scores = [s for s in edge_scores if s is not None]
    if len(edge_scores) >= 3:
        threshold = statistics.median(edge_scores)
        survived = [
            c for c in survived
            if (_to_float(c.get("edge_score")) or 0.0) >= threshold
        ]

    return survived


# ============================================================
# V3 EXIT — V1 patched + price-action overlay
# ============================================================
def _ma20_for(df: pd.DataFrame, date: pd.Timestamp, lookback: int = 20) -> float | None:
    sub = df[df["date"] <= date].tail(lookback)
    if len(sub) < lookback:
        return None
    return float(sub["close"].mean())


def _price_action_exit_check(
    df: pd.DataFrame,
    date: pd.Timestamp,
    entry_price: float,
) -> bool:
    """Return True if PA exit should fire:
       1. close < MA20 for PA_EXIT_MA20_BARS_BELOW consecutive bars
       2. red-bar volume > green-bar volume over last 3 bars
    """
    sub = df[df["date"] <= date].tail(max(PA_EXIT_MA20_BARS_BELOW, 3) + 25)
    if len(sub) < 25:
        return False

    # Check MA20 breach
    closes = sub["close"].astype(float).values
    for i in range(1, PA_EXIT_MA20_BARS_BELOW + 1):
        if i > len(sub):
            return False
        # MA20 at bar -i
        if len(sub) < 20 + i:
            return False
        ma20 = float(sub["close"].iloc[-(i + 20):-i].mean()) if i > 0 else float(sub["close"].iloc[-20:].mean())
        # Simpler: MA20 ending at bar -i is mean of close from -i-19 to -i
        end = len(sub) - i + 1  # exclusive end index
        start = end - 20
        if start < 0:
            return False
        ma20_at_bar = float(sub["close"].iloc[start:end].mean())
        if closes[-i] >= ma20_at_bar:
            return False

    # Red vs green volume on last 3 bars
    last3 = sub.tail(3)
    red_vol = 0.0
    green_vol = 0.0
    for _, r in last3.iterrows():
        if float(r["close"]) < float(r["open"]):
            red_vol += float(r["volume"])
        elif float(r["close"]) > float(r["open"]):
            green_vol += float(r["volume"])
    if red_vol <= green_vol:
        return False

    return True


def _exit_decision_v3(
    row: pd.Series,
    position: LivePosition,
    cfg: LivePipelineBacktestConfig,
    symbol_data: pd.DataFrame,
    date: pd.Timestamp,
) -> tuple[str | None, float]:
    # First check v1 patched logic (SL/ATR/TRAIL/TP/MAX_HOLD)
    reason, price = _edge_exit_decision(row, position, cfg)
    if reason:
        return reason, price

    # Then check price-action overlay — only with profit cushion
    close = float(row["close"])
    pnl = (close - position.entry_price) / position.entry_price
    if pnl < PA_EXIT_MIN_PNL:
        return None, close

    if _price_action_exit_check(symbol_data, date, position.entry_price):
        return "EDGE_PA_MA20_BREAK", close * (1.0 - cfg.slippage_rate)

    return None, close


# ============================================================
# DATA LOADERS (copied from v2)
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
# EDGE SIGNAL CACHE (copied from v1/v2)
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
# ENTRY ZONE / FILL (copied from v2)
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
    # Pullback-family strategies — entry near support, wider tolerance for fill rate
    if edge_name in {
        "leader_pullback_market_regime_v3",
        "leader_pullback_market_healthy_v2",
        "leader_pullback_market_healthy",
        "mean_reversion_uptrend_ma20_v1",
        "pullback_deep_quality_v1",
        "pullback_deep_quality_v2",
        "reclaim_ma20_quality_v1",
        "reclaim_ma20_quality_v2",
    }:
        level = float(supports[0]) if supports else (ema50 if ema50 > 0 else signal_close)
        # Wider zone (6%): support - 4% to support + 2%
        # Pullback may overshoot support before bouncing — allow it
        return round(level * 0.96, 2), round(level * 1.02, 2)
    # Generic breakout/momentum family — entry within 0.5% to 1% above signal close
    if any(k in edge_name for k in ("breakout", "momentum", "trend", "crossover", "spring", "engulfing", "hammer")):
        return round(signal_close * 0.995, 2), round(signal_close * 1.012, 2)
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


def fill_entries_v3(
    orders: list[dict[str, Any]],
    data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    day_idx: int,
    cash: float,
    positions: dict[str, LivePosition],
    cfg: LivePipelineBacktestConfig,
    model: str,
    counters: dict[str, int],
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
        stop_loss = _edge_initial_sl(entry, indicators, edge_risk) if edge_risk else _compute_initial_sl(entry, indicators)
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
            take_profit=_edge_take_profit(entry, stop_loss, edge_risk, cfg),
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
# MAIN LOOP — v3 (apply quality filter + PA-exit overlay)
# ============================================================
def run_model_v3(
    universe_data: dict[str, pd.DataFrame],
    vnindex: pd.DataFrame,
    cfg: LivePipelineBacktestConfig,
    model: str,
    *,
    enable_entry_filter: bool = True,
    enable_pa_exit: bool = True,
) -> dict[str, Any]:
    data = _prepare_data(universe_data)
    index_data = _prepare_single_frame(vnindex)
    calendar = sorted(set(index_data["date"]))
    cash = float(cfg.initial_capital)
    positions: dict[str, LivePosition] = {}
    pending_orders: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    counters = {"signal_days": 0, "signals": 0, "attempted_orders": 0, "fills": 0, "filtered_out": 0}
    date_to_pos = {date: idx for idx, date in enumerate(calendar)}
    start_ts = pd.Timestamp(cfg.start_date).normalize() if cfg.start_date else None
    end_ts = pd.Timestamp(cfg.end_date).normalize() if cfg.end_date else None
    last_processed_date = None

    for date in calendar[cfg.lookback :]:
        if end_ts is not None and date > end_ts:
            break
        last_processed_date = date
        day_idx = date_to_pos[date]
        todays_orders = pending_orders.pop(date, [])
        if todays_orders:
            cash = fill_entries_v3(todays_orders, data, date, day_idx, cash, positions, cfg, model, counters)

        for symbol in list(positions):
            row = _row_at(data[symbol], date)
            if row is None:
                continue
            position = positions[symbol]
            position.holding_bars = day_idx - position.entry_idx
            if enable_pa_exit:
                exit_reason, exit_price = _exit_decision_v3(row, position, cfg, data[symbol], date)
            else:
                exit_reason, exit_price = _edge_exit_decision(row, position, cfg)
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

        # V3 entry filter
        if enable_entry_filter:
            raw_count = len(candidates)
            candidates = filter_quality_candidates(candidates, data, date)
            counters["filtered_out"] += raw_count - len(candidates)

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
            "filtered_out": counters["filtered_out"],
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
# CLI
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="V3 backtest — entry filter + PA exit")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-12")
    parser.add_argument("--edge-strategy", default=DEFAULT_LIVE_EDGE_FAMILY)
    parser.add_argument("--label", default="v3_filter_pa")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    parser.add_argument(
        "--ablation",
        choices=["full", "entry_filter_only", "pa_exit_only", "baseline_patched"],
        default="full",
        help="Run ablation: full=both, entry_filter_only=skip PA exit, pa_exit_only=skip entry filter, baseline_patched=neither (v1-patched only).",
    )
    parser.add_argument("--post-extension-threshold", type=float, default=None)
    parser.add_argument("--pa-exit-min-pnl", type=float, default=None)
    args = parser.parse_args()

    global POST_EXTENSION_THRESHOLD, PA_EXIT_MIN_PNL
    if args.post_extension_threshold is not None:
        POST_EXTENSION_THRESHOLD = args.post_extension_threshold
    if args.pa_exit_min_pnl is not None:
        PA_EXIT_MIN_PNL = args.pa_exit_min_pnl

    enable_entry_filter = args.ablation in {"full", "entry_filter_only"}
    enable_pa_exit = args.ablation in {"full", "pa_exit_only"}

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
        print(f"[v3 {args.ablation}] running {model} | entry_filter={enable_entry_filter} pa_exit={enable_pa_exit}")
        result = run_model_v3(
            universe_data, vnindex, replace(base_cfg), model,
            enable_entry_filter=enable_entry_filter,
            enable_pa_exit=enable_pa_exit,
        )
        details[model] = result
        metrics = result["metrics"]
        rows.append(
            {
                "label": args.label,
                "ablation": args.ablation,
                "universe": args.universe.upper(),
                "start": args.start,
                "end": args.end,
                "model": model,
                "max_positions": args.max_positions,
                "max_candidates_per_day": args.max_candidates_per_day,
                "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
                "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
                "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
                "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
                "number_of_trades": int(metrics.get("number_of_trades", 0)),
                "signal_days": int(metrics.get("signal_days", 0)),
                "signals": int(metrics.get("signals", 0)),
                "filtered_out": int(metrics.get("filtered_out", 0)),
                "attempted_orders": int(metrics.get("attempted_orders", 0)),
                "fills": int(metrics.get("fills", 0)),
                "fill_rate_pct": round(float(metrics.get("fill_rate", 0.0)) * 100, 2),
            }
        )

    summary = pd.DataFrame(rows)
    stem = f"{args.universe}_{args.label}_{args.ablation}_{args.start}_{args.end}".replace(",", "_")
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
