"""Replay backtest for the live trade screener pipeline.

This module intentionally reuses the live screener detectors and scoring
functions. It does not call LLM agents; it backtests the deterministic
pre-LLM candidate pipeline plus a realistic portfolio execution layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import pandas_ta as ta

from multiagents_trading_assistant.backtest.engine import _compute_initial_sl, _prep_ta_money_flow_features
from multiagents_trading_assistant.edge_lab.live_signal import get_edge_strategy_signals
from multiagents_trading_assistant.fetcher import (
    get_ohlcv_history,
    get_vn30_symbols,
    get_vn100_symbols,
)
from multiagents_trading_assistant.indicators import (
    _bb_percent_series,
    _bb_width_series,
    _channel_width_series,
    _bollinger_position,
    _crossed_above_recent,
    _diff_series,
    _ema_series,
    _macd_signal,
    _ma_phase,
    _ma_trend,
    _rsi_signal,
    _volume_surge,
    compute_confluence_score,
)
from multiagents_trading_assistant.quantagents_backtest.metrics import compute_metrics
from multiagents_trading_assistant.screener.trade_screener import (
    MarketContext,
    _extract_trend,
    _is_setup_allowed_in_regime,
    compute_priority_score,
    detect_adx_trend,
    detect_aroon_trend_shift,
    detect_bb_squeeze,
    detect_breakout,
    detect_breakout_retest_entry,
    detect_bullish_engulfing,
    detect_double_bottom,
    detect_flag_pennant,
    detect_golden_cross,
    detect_hammer,
    detect_inside_bar,
    detect_keltner_squeeze_breakout,
    detect_kijun_bounce,
    detect_kumo_breakout,
    detect_kumo_twist_entry,
    detect_linreg_momentum,
    detect_ma_pullback,
    detect_macd_crossover,
    detect_momentum_surge,
    detect_nr7,
    detect_obv_accumulation,
    detect_oversold_mean_reversion,
    detect_pin_bar,
    detect_retest,
    detect_rsi_bounce,
    detect_spring,
    detect_supertrend_pullback,
    detect_tk_cross,
    detect_trend_pullback,
)


RESULTS_DIR = Path(__file__).resolve().parents[2] / "backtest_results"


LIVE_STRATEGIES = [
    ("BREAKOUT", detect_breakout),
    ("FLAG_PENNANT", detect_flag_pennant),
    ("BB_SQUEEZE", detect_bb_squeeze),
    ("RETEST", detect_retest),
    ("SPRING", detect_spring),
    ("GOLDEN_CROSS", detect_golden_cross),
    ("DOUBLE_BOTTOM", detect_double_bottom),
    ("MOMENTUM_SURGE", detect_momentum_surge),
    ("MACD_CROSSOVER", detect_macd_crossover),
    ("MA_PULLBACK", detect_ma_pullback),
    ("INSIDE_BAR", detect_inside_bar),
    ("NR7", detect_nr7),
    ("HAMMER", detect_hammer),
    ("RSI_BOUNCE", detect_rsi_bounce),
    ("TREND_PULLBACK", detect_trend_pullback),
    ("BREAKOUT_RETEST_ENTRY", detect_breakout_retest_entry),
    ("BULLISH_ENGULFING", detect_bullish_engulfing),
    ("PIN_BAR", detect_pin_bar),
    ("KUMO_BREAKOUT", detect_kumo_breakout),
    ("TK_CROSS", detect_tk_cross),
    ("KIJUN_BOUNCE", detect_kijun_bounce),
    ("KUMO_TWIST_ENTRY", detect_kumo_twist_entry),
    ("ADX_TREND", detect_adx_trend),
    ("SUPERTREND_PULLBACK", detect_supertrend_pullback),
    ("OBV_ACCUMULATION", detect_obv_accumulation),
    ("KELTNER_SQUEEZE", detect_keltner_squeeze_breakout),
    ("OVERSOLD_MEAN_REVERSION", detect_oversold_mean_reversion),
    ("AROON_TREND_SHIFT", detect_aroon_trend_shift),
    ("LINREG_MOMENTUM", detect_linreg_momentum),
]


LEGACY_SETUP_BLOCKLIST = {
    # Empirical quality gate from VN100 live-pipeline walk-forward 2025-01-01..2026-05-10.
    # These broad legacy detectors should be re-enabled only after setup-level retests
    # show positive expectancy under live execution rules.
    "AROON_TREND_SHIFT",
    "ADX_TREND",
    "BB_SQUEEZE",
    "BREAKOUT",
    "BREAKOUT_RETEST_ENTRY",
    "BULLISH_ENGULFING",
    "DOUBLE_BOTTOM",
    "FLAG_PENNANT",
    "GOLDEN_CROSS",
    "HAMMER",
    "INSIDE_BAR",
    "LINREG_MOMENTUM",
    "MACD_CROSSOVER",
    "MOMENTUM_SURGE",
    "NR7",
    "OVERSOLD_MEAN_REVERSION",
    "RETEST",
    "RSI_BOUNCE",
    "SPRING",
    "SUPERTREND_PULLBACK",
}


@dataclass(frozen=True)
class LivePipelineBacktestConfig:
    initial_capital: float = 100_000_000.0
    start_date: str | None = None
    end_date: str | None = None
    max_positions: int = 5
    max_candidates_per_day: int = 10
    lookback: int = 200
    min_avg_volume: float = 300_000
    commission_rate: float = 0.001
    sell_tax_rate: float = 0.001
    slippage_rate: float = 0.0005
    rr_ratio: float = 1.8
    max_hold_bars: int = 30
    settlement_bars: int = 2
    allow_downtrend_entries: bool = False
    require_uptrend: bool = False          # chỉ vào lệnh khi UPTREND (bỏ qua SIDEWAY)
    setup_whitelist: frozenset[str] | None = None  # None = tất cả setups
    lot_size: int = 100
    # Backtest mode:
    #   "raw_screener"      — screener tự trade, hard gate, top-N (hiện tại)
    #   "broad_pool"        — screener rộng, không execute, chỉ đo candidate flow
    #   "simulated_llm_gate"— screener rộng → proxy LLM filter → execute
    backtest_mode: str = "raw_screener"
    # Số candidate tối đa cho broad/llm-gate mode (không ảnh hưởng raw_screener)
    broad_max_candidates: int = 100
    # Ngưỡng proxy LLM: cắt candidate có score thấp hơn ngưỡng này
    llm_gate_min_score: float = 68.0
    # Proxy LLM: tỉ lệ tối đa candidate được giữ lại sau gate (60-80% bị loại)
    llm_gate_keep_ratio: float = 0.30
    edge_strategy_name: str | None = None
    # Closed-loop learning: skip candidates whose (edge_strategy_name, regime) is in this set.
    # Set is built by Bob's Friday meeting from real losses (or offline from prior backtest).
    deprecated_keys: frozenset[tuple[str, str]] | None = None


@dataclass
class LivePosition:
    symbol: str
    setup_type: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_idx: int
    entry_price: float
    stop_loss: float
    take_profit: float
    shares: int
    priority_score: float
    reasons: list[str]
    edge_strategy_name: str = ""
    edge_strategy_passed: bool = False
    edge_score: float | None = None
    edge_rank: float | None = None
    edge_rank_score: float | None = None
    edge_risk: dict | None = None
    entry_atr: float | None = None
    highest_price: float | None = None
    holding_bars: int = 0

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.shares


def run_live_pipeline_backtest(
    universe_data: dict[str, pd.DataFrame],
    vnindex: pd.DataFrame,
    *,
    config: LivePipelineBacktestConfig | None = None,
) -> dict:
    cfg = config or LivePipelineBacktestConfig()
    data = _prepare_data(universe_data)
    index_data = _prepare_single_frame(vnindex)
    if not data:
        raise ValueError("No usable OHLCV data for live pipeline backtest")
    if index_data.empty:
        raise ValueError("VNINDEX data is required for market gate")

    calendar = sorted(set(index_data["date"]))
    if len(calendar) <= cfg.lookback + 2:
        raise ValueError("Not enough overlapping history for requested lookback")

    cash = float(cfg.initial_capital)
    positions: dict[str, LivePosition] = {}
    pending_orders: dict[pd.Timestamp, list[dict]] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []
    # broad_pool mode: collect daily candidate snapshots instead of executing
    candidate_log: list[dict] = []

    date_to_pos = {date: idx for idx, date in enumerate(calendar)}

    start_ts = pd.Timestamp(cfg.start_date).normalize() if cfg.start_date else None
    end_ts = pd.Timestamp(cfg.end_date).normalize() if cfg.end_date else None

    last_processed_date = None
    for date in calendar[cfg.lookback :]:
        if end_ts is not None and date > end_ts:
            break
        last_processed_date = date
        day_idx = date_to_pos[date]

        # Fill yesterday's accepted screener candidates at today's open.
        todays_orders = pending_orders.pop(date, [])
        if todays_orders:
            cash = _fill_entries(todays_orders, data, date, day_idx, cash, positions, cfg)

        # Manage open positions using today's OHLC.
        for symbol in list(positions):
            if symbol not in data:
                continue
            row = _row_at(data[symbol], date)
            if row is None:
                continue
            position = positions[symbol]
            position.holding_bars = day_idx - position.entry_idx
            exit_reason, exit_price = _exit_decision(row, position, cfg)
            if exit_reason and position.holding_bars >= cfg.settlement_bars:
                cash, trade = _close_position(date, exit_price, exit_reason, position, cash, cfg)
                trades.append(trade)
                del positions[symbol]

        if start_ts is None or date >= start_ts:
            equity = cash + _mark_to_market(positions, data, date)
            equity_rows.append(
                {
                    "date": date,
                    "equity": equity,
                    "cash": cash,
                    "positions": len(positions),
                    "gross_exposure": _position_value(positions, data, date) / equity if equity else 0.0,
                }
            )

        # Generate signal after close for next session.
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
        candidates = [c for c in candidates if c["symbol"] not in positions]

        if cfg.backtest_mode == "broad_pool":
            # Log candidate pool stats; no execution
            for c in candidates:
                candidate_log.append({
                    "date": date,
                    "symbol": c["symbol"],
                    "setup_type": c["setup_type"],
                    "priority_score": c["priority_score"],
                    "risk_flags": "|".join(c.get("risk_flags", [])),
                    "money_flow_regime": c.get("money_flow_regime", ""),
                    "market_trend": c.get("market_trend", ""),
                })
            continue

        if cfg.backtest_mode == "simulated_llm_gate":
            candidates = _apply_simulated_llm_gate(candidates, cfg)
            # cap to max_positions after LLM gate
            candidates = candidates[: cfg.max_candidates_per_day]

        if candidates:
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
    if equity_rows:
        equity_rows[-1]["equity"] = cash + _mark_to_market(positions, data, final_date)
        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["positions"] = len(positions)
        equity_rows[-1]["gross_exposure"] = (
            _position_value(positions, data, final_date) / equity_rows[-1]["equity"]
            if equity_rows[-1]["equity"] else 0.0
        )

    if cfg.backtest_mode == "broad_pool":
        # No execution — return candidate flow analytics only
        cdf = pd.DataFrame(candidate_log) if candidate_log else pd.DataFrame()
        pool_summary: list[dict] = []
        if not cdf.empty:
            for setup, grp in cdf.groupby("setup_type"):
                flag_counts = grp["risk_flags"].str.split("|").explode()
                flag_counts = flag_counts[flag_counts != ""].value_counts().to_dict()
                pool_summary.append({
                    "setup_type": setup,
                    "appearances": len(grp),
                    "unique_symbols": grp["symbol"].nunique(),
                    "avg_score": round(grp["priority_score"].mean(), 1),
                    "top_risk_flag": max(flag_counts, key=flag_counts.get) if flag_counts else "",
                })
            pool_summary.sort(key=lambda r: r["appearances"], reverse=True)
        return {
            "backtest_mode": "broad_pool",
            "candidate_log": cdf,
            "pool_summary": pool_summary,
            "metrics": {},
            "trades": [],
            "setup_breakdown": [],
            "symbol_breakdown": [],
        }

    equity_frame = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame()
    equity_curve = equity_frame["equity"].rename("live_pipeline_equity") if not equity_frame.empty else pd.Series(dtype=float)
    metrics = compute_metrics(equity_curve, trades)
    return {
        "backtest_mode": cfg.backtest_mode,
        "equity_curve": equity_curve,
        "equity_frame": equity_frame,
        "trades": trades,
        "metrics": metrics,
        "setup_breakdown": _breakdown(trades, "setup_type"),
        "symbol_breakdown": _breakdown(trades, "symbol"),
    }


def fetch_universe_history(
    universe: str,
    start: str,
    end: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    symbols = _resolve_symbols(universe)
    warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
    data = {}
    for idx, symbol in enumerate(symbols, start=1):
        df = get_ohlcv_history(symbol, warmup_start, end)
        if not df.empty:
            data[symbol] = df
        if idx % 10 == 0:
            print(f"[live-backtest] fetched {idx}/{len(symbols)} symbols")
    vnindex = get_ohlcv_history("VNINDEX", warmup_start, end)
    return data, vnindex


def save_live_pipeline_results(result: dict, label: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outdir = RESULTS_DIR / f"live_pipeline_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)
    if result.get("backtest_mode") == "broad_pool":
        cl = result.get("candidate_log")
        if cl is not None and not cl.empty:
            cl.to_csv(outdir / "candidate_log.csv", index=False)
        pd.DataFrame(result.get("pool_summary", [])).to_csv(outdir / "pool_summary.csv", index=False)
    else:
        ef = result.get("equity_frame")
        if ef is not None and not ef.empty:
            ef.to_csv(outdir / "equity_curve.csv")
        pd.DataFrame(result.get("trades", [])).to_csv(outdir / "trades.csv", index=False)
        pd.DataFrame([result.get("metrics", {})]).to_csv(outdir / "metrics.csv", index=False)
        pd.DataFrame(result.get("setup_breakdown", [])).to_csv(outdir / "setup_breakdown.csv", index=False)
        pd.DataFrame(result.get("symbol_breakdown", [])).to_csv(outdir / "symbol_breakdown.csv", index=False)
    return outdir


def _scan_live_candidates(
    data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    market_ctx: MarketContext,
    cfg: LivePipelineBacktestConfig,
) -> list[dict]:
    broad = cfg.backtest_mode in ("broad_pool", "simulated_llm_gate")
    ref_trend = market_ctx.reference_trend
    strategies = LIVE_STRATEGIES
    if cfg.setup_whitelist:
        strategies = [(name, fn) for name, fn in strategies if name in cfg.setup_whitelist]
    # In raw mode, restrict reversal-only setups during downtrend.
    # In broad mode we keep all setups but flag the regime mismatch.
    if not broad and ref_trend == "DOWNTREND":
        allowed_names = {
            "DOUBLE_BOTTOM", "RSI_BOUNCE", "HAMMER",
            "BULLISH_ENGULFING", "PIN_BAR", "OVERSOLD_MEAN_REVERSION",
        }
        strategies = [(name, fn) for name, fn in strategies if name in allowed_names]

    edge_signals: dict[str, dict] = {}
    if cfg.edge_strategy_name:
        edge_signals = get_edge_strategy_signals(
            list(data.keys()),
            as_of_date=date.strftime("%Y-%m-%d"),
            strategy_name=cfg.edge_strategy_name,
        )

    candidates = []
    for symbol, df in data.items():
        edge_signal = edge_signals.get(symbol, {})
        if cfg.edge_strategy_name and not edge_signal.get("passed"):
            continue
        window = (
            df.loc[:date].tail(cfg.lookback)
            if isinstance(df.index, pd.DatetimeIndex)
            else df[df["date"] <= date].tail(cfg.lookback)
        )
        # Hard block: insufficient history or clearly illiquid
        if len(window) < max(80, cfg.lookback // 2):
            continue
        avg_vol = float(window["volume"].tail(20).mean())
        if avg_vol < cfg.min_avg_volume:
            continue

        ind = _indicator_snapshot(window)
        if not ind:
            continue
        money_flow = _money_flow_at(window, market_ctx)
        mf_regime = money_flow.get("regime", "")

        # In raw mode: hard gate on bad money flow
        if not broad and mf_regime in {"DISTRIBUTION", "EXHAUSTION_INFLOW"}:
            continue

        ind["money_flow_analysis"] = money_flow
        if cfg.edge_strategy_name:
            ind["edge_strategy_analysis"] = edge_signal
            score = _edge_priority_score(edge_signal)
            candidates.append(
                {
                    "symbol": symbol,
                    "setup_type": str(edge_signal.get("setup_type") or "EDGE"),
                    "signal_date": date,
                    "priority_score": min(100.0, score),
                    "indicators": ind,
                    "reasons": [
                        f"Edge strategy passed: {cfg.edge_strategy_name}",
                        f"edge_score={edge_signal.get('edge_score')}",
                    ],
                    "risk_flags": [],
                    "money_flow_regime": mf_regime,
                    "market_trend": ref_trend,
                    "edge_strategy_name": edge_signal.get("strategy_name") or cfg.edge_strategy_name,
                    "edge_strategy_passed": True,
                    "edge_score": edge_signal.get("edge_score"),
                    "edge_rank": edge_signal.get("edge_rank"),
                    "edge_rank_score": edge_signal.get("edge_rank_score"),
                    "edge_risk": dict(edge_signal.get("risk") or {}),
                }
            )
            continue

        for setup_type, detector in strategies:
            if setup_type in LEGACY_SETUP_BLOCKLIST:
                continue
            if not _is_setup_allowed_in_regime(setup_type, ref_trend):
                if not broad:
                    continue
            passed, reasons = detector(window, ind)
            if not passed:
                continue

            score = compute_priority_score(setup_type, ind, ref_trend, money_flow)
            if mf_regime in {"BREAKOUT_FLOW", "EARLY_MONEY_IN", "MONEY_IN", "ACCUMULATION"}:
                reasons = [*reasons, f"Money flow: {mf_regime} score={money_flow.get('score')}"]

            # Build risk flags (informational — LLM/risk layer should read these)
            risk_flags: list[str] = []
            if mf_regime in {"DISTRIBUTION", "EXHAUSTION_INFLOW"}:
                risk_flags.append(f"BAD_MONEY_FLOW:{mf_regime}")
            if ref_trend == "DOWNTREND":
                risk_flags.append("MARKET_DOWNTREND")
            elif ref_trend == "SIDEWAY" and setup_type not in (
                "DOUBLE_BOTTOM", "RSI_BOUNCE", "BB_SQUEEZE", "OVERSOLD_MEAN_REVERSION",
                "OBV_ACCUMULATION", "KELTNER_SQUEEZE", "AROON_TREND_SHIFT",
            ):
                risk_flags.append("REGIME_MISMATCH:SIDEWAY")
            rsi = ind.get("rsi")
            if rsi and rsi > 75:
                risk_flags.append(f"OVERBOUGHT:RSI={rsi:.0f}")
            if avg_vol < cfg.min_avg_volume * 1.5:
                risk_flags.append("LOW_LIQUIDITY")

            candidates.append(
                {
                    "symbol": symbol,
                    "setup_type": setup_type,
                    "signal_date": date,
                    "priority_score": score,
                    "indicators": ind,
                    "reasons": reasons,
                    "risk_flags": risk_flags,
                    "money_flow_regime": mf_regime,
                    "market_trend": ref_trend,
                }
            )
            break  # one setup per symbol per day

    candidates.sort(key=lambda c: c["priority_score"], reverse=True)
    limit = cfg.broad_max_candidates if broad else cfg.max_candidates_per_day
    return candidates[:limit]


def _apply_simulated_llm_gate(
    candidates: list[dict],
    cfg: "LivePipelineBacktestConfig",
) -> list[dict]:
    """Rule-based proxy for LLM/debate layer. Simulates cutting 60-80% of candidates.

    Keeps a candidate only if:
    - score >= llm_gate_min_score
    - no hard risk flags (bad money flow, overbought)
    - at most one soft flag (regime mismatch, low liquidity)

    Then caps at keep_ratio × total to enforce the 60-80% cut.
    """
    _BAD_FLAGS = {"BAD_MONEY_FLOW:DISTRIBUTION", "BAD_MONEY_FLOW:EXHAUSTION_INFLOW"}

    filtered = []
    for c in candidates:
        flags = set(c.get("risk_flags", []))
        if flags & _BAD_FLAGS:
            continue
        if c["priority_score"] < cfg.llm_gate_min_score:
            continue
        overbought = any(f.startswith("OVERBOUGHT") for f in flags)
        if overbought:
            continue
        soft_flags = [f for f in flags if f not in _BAD_FLAGS and not f.startswith("OVERBOUGHT")]
        if len(soft_flags) > 1:
            continue
        # Closed-loop deprecate gate: skip if (strategy, regime) flagged from past losses
        if cfg.deprecated_keys:
            strat = str(c.get("edge_strategy_name") or "")
            regime = str(c.get("mkt_regime_label") or c.get("regime") or "").upper()
            if (strat, regime) in cfg.deprecated_keys:
                continue
        filtered.append(c)

    # Enforce keep_ratio cap (simulate LLM culling down to top fraction)
    max_keep = max(1, int(len(candidates) * cfg.llm_gate_keep_ratio))
    return filtered[:max_keep]


def _fill_entries(
    orders: list[dict],
    data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    day_idx: int,
    cash: float,
    positions: dict[str, LivePosition],
    cfg: LivePipelineBacktestConfig,
) -> float:
    slots = max(0, cfg.max_positions - len(positions))
    if slots <= 0:
        return cash
    orders = [order for order in orders if order["symbol"] not in positions][:slots]
    if not orders:
        return cash
    budget = cash / max(1, min(slots, len(orders)))
    for order in orders:
        row = _row_at(data[order["symbol"]], date)
        if row is None:
            continue
        entry = float(row["open"]) * (1.0 + cfg.slippage_rate)
        if entry <= 0:
            continue
        gross_shares = int(budget // entry)
        shares = (gross_shares // cfg.lot_size) * cfg.lot_size if cfg.lot_size > 1 else gross_shares
        if shares <= 0:
            continue
        gross = shares * entry
        cost = gross * (1.0 + cfg.commission_rate)
        if cost > cash:
            continue
        ind = order["indicators"]
        edge_risk = dict(order.get("edge_risk") or {})
        stop_loss = _edge_initial_sl(entry, ind, edge_risk) if edge_risk else _compute_initial_sl(entry, ind)
        if stop_loss <= 0 or stop_loss >= entry:
            continue
        take_profit = _edge_take_profit(entry, stop_loss, edge_risk, cfg)
        cash -= cost
        raw_edge_score = order.get("edge_score")
        edge_score = float(raw_edge_score) if raw_edge_score is not None and pd.notna(raw_edge_score) else None
        raw_edge_rank = order.get("edge_rank")
        edge_rank = float(raw_edge_rank) if raw_edge_rank is not None and pd.notna(raw_edge_rank) else None
        raw_edge_rank_score = order.get("edge_rank_score")
        edge_rank_score = (
            float(raw_edge_rank_score)
            if raw_edge_rank_score is not None and pd.notna(raw_edge_rank_score)
            else None
        )
        positions[order["symbol"]] = LivePosition(
            symbol=order["symbol"],
            setup_type=order["setup_type"],
            signal_date=order["signal_date"],
            entry_date=date,
            entry_idx=day_idx,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            shares=shares,
            priority_score=float(order["priority_score"]),
            reasons=list(order["reasons"]),
            edge_strategy_name=str(order.get("edge_strategy_name") or ""),
            edge_strategy_passed=bool(order.get("edge_strategy_passed", False)),
            edge_score=edge_score,
            edge_rank=edge_rank,
            edge_rank_score=edge_rank_score,
            edge_risk=edge_risk or None,
            entry_atr=_safe_float(ind.get("atr")),
            highest_price=entry,
        )
    return cash


def _exit_decision(row: pd.Series, position: LivePosition, cfg: LivePipelineBacktestConfig) -> tuple[str | None, float]:
    if position.edge_risk:
        return _edge_exit_decision(row, position, cfg)

    low = float(row["low"])
    high = float(row["high"])
    close = float(row["close"])
    open_ = float(row["open"])
    if low <= position.stop_loss:
        return "SL", min(open_, position.stop_loss) * (1.0 - cfg.slippage_rate)
    if high >= position.take_profit:
        return "TP", max(open_, position.take_profit) * (1.0 - cfg.slippage_rate)
    if position.holding_bars >= cfg.max_hold_bars:
        return "MAX_HOLD", close * (1.0 - cfg.slippage_rate)
    return None, close


def _edge_initial_sl(entry: float, ind: dict, risk: dict) -> float:
    atr = _safe_float(ind.get("atr"))
    initial_mult = risk.get("initial_atr_stop_mult")
    stops = []
    stop_pct = _safe_float(risk.get("stop_loss"))
    if stop_pct and stop_pct > 0:
        stops.append(entry * (1.0 - stop_pct))
    if atr and atr > 0 and initial_mult is not None:
        stops.append(entry - float(initial_mult) * atr)
    if not stops:
        return _compute_initial_sl(entry, ind)
    return round(max(stops), 0)


def _edge_take_profit(
    entry: float,
    stop_loss: float,
    risk: dict,
    cfg: LivePipelineBacktestConfig,
) -> float:
    tp_pct = _safe_float(risk.get("take_profit"))
    if tp_pct and tp_pct > 0:
        return entry * (1.0 + tp_pct)
    return entry + cfg.rr_ratio * (entry - stop_loss)


def _edge_exit_decision(row: pd.Series, position: LivePosition, cfg: LivePipelineBacktestConfig) -> tuple[str | None, float]:
    # 2026-05-14 bias fix: previous code returned open_ * slip on intraday
    # triggers, under-counting losses (and wins) ~1.8x on full portfolios.
    # Now: SL/ATR/trailing exit at min(open, trigger); TP at max(open, trigger).
    # See MIGRATION_NOTES.md for details.
    low = float(row["low"])
    high = float(row["high"])
    close = float(row["close"])
    open_ = float(row["open"])
    risk = position.edge_risk or {}
    entry = position.entry_price
    position.highest_price = max(float(position.highest_price or entry), high)

    stop_pct = _safe_float(risk.get("stop_loss"))
    if stop_pct and low <= entry * (1.0 - stop_pct):
        trigger = entry * (1.0 - stop_pct)
        return "EDGE_STOP_LOSS", min(open_, trigger) * (1.0 - cfg.slippage_rate)

    atr = _safe_float(position.entry_atr)
    if atr and atr > 0:
        initial_mult = risk.get("initial_atr_stop_mult")
        if initial_mult is not None:
            atr_trigger = entry - float(initial_mult) * atr
            if low <= atr_trigger:
                return "EDGE_ATR_STOP", min(open_, atr_trigger) * (1.0 - cfg.slippage_rate)

        trailing_mult = risk.get("trailing_atr_mult")
        activation = _safe_float(risk.get("trailing_profit_activation")) or 0.0
        if (
            trailing_mult is not None
            and float(position.highest_price or entry) >= entry * (1.0 + activation)
        ):
            trail_trigger = float(position.highest_price or entry) - float(trailing_mult) * atr
            if low <= trail_trigger:
                return "EDGE_TRAILING_ATR_STOP", min(open_, trail_trigger) * (1.0 - cfg.slippage_rate)

    take_profit_pct = _safe_float(risk.get("take_profit"))
    if take_profit_pct and high >= entry * (1.0 + take_profit_pct):
        tp_trigger = entry * (1.0 + take_profit_pct)
        return "EDGE_TAKE_PROFIT", max(open_, tp_trigger) * (1.0 - cfg.slippage_rate)

    max_holding = int(risk.get("max_holding_bars") or cfg.max_hold_bars)
    if position.holding_bars >= max_holding:
        return "EDGE_MAX_HOLD", close * (1.0 - cfg.slippage_rate)
    return None, close


def _close_position(
    date: pd.Timestamp,
    price: float,
    reason: str,
    position: LivePosition,
    cash: float,
    cfg: LivePipelineBacktestConfig,
) -> tuple[float, dict]:
    gross = float(price) * position.shares
    costs = gross * (cfg.commission_rate + cfg.sell_tax_rate)
    net = gross - costs
    cash += net
    pnl = net - position.cost_basis * (1.0 + cfg.commission_rate)
    pnl_pct = pnl / (position.cost_basis * (1.0 + cfg.commission_rate)) if position.cost_basis else 0.0
    return cash, {
        "symbol": position.symbol,
        "setup_type": position.setup_type,
        "signal_date": position.signal_date.date().isoformat(),
        "entry_date": position.entry_date.date().isoformat(),
        "exit_date": date.date().isoformat(),
        "entry_price": position.entry_price,
        "exit_price": float(price),
        "shares": position.shares,
        "entry_value": position.cost_basis,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "holding_bars": position.holding_bars,
        "exit_reason": reason,
        "priority_score": position.priority_score,
        "edge_strategy_name": position.edge_strategy_name,
        "edge_strategy_passed": position.edge_strategy_passed,
        "edge_score": position.edge_score,
        "edge_rank": position.edge_rank,
        "edge_rank_score": position.edge_rank_score,
        "edge_risk": position.edge_risk,
        "reasons": " | ".join(position.reasons),
    }


def _safe_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _edge_priority_score(edge_signal: dict) -> float:
    rank_score = _safe_float(edge_signal.get("edge_rank_score"))
    if rank_score is not None:
        return rank_score
    rank = _safe_float(edge_signal.get("edge_rank"))
    if rank is not None:
        return max(0.0, 10_000.0 - rank)
    score = _safe_float(edge_signal.get("edge_score"))
    return score if score is not None else 0.0


def _market_context_at(index_data: pd.DataFrame, date: pd.Timestamp, lookback: int) -> MarketContext:
    window = index_data[index_data["date"] <= date].tail(lookback)
    trend, price, ma20, ma60, ma200, change = _extract_trend(window)
    close = window["close"]
    high_20 = float(close.rolling(20).max().iloc[-1])
    low_20 = float(close.rolling(20).min().iloc[-1])
    rng = high_20 - low_20
    pos = (price - low_20) / rng if rng > 0 else 0.5
    zone = "NEAR_SUPPORT" if pos <= 0.25 else "NEAR_RESISTANCE" if pos >= 0.75 else "MID_RANGE"
    return MarketContext(
        trend=trend,
        should_trade=trend != "DOWNTREND",
        ma20=ma20,
        ma60=ma60,
        ma200=ma200,
        sideway_zone=zone,
        position_in_range=round(pos, 3),
        vni_change_pct=change,
        current_price=price,
        vnmidcap_trend="UNKNOWN",
        vnmidcap_change_pct=0.0,
        is_index_distorted=False,
        reference_trend=trend,
        reference_index="VNINDEX",
        vingroup_contribution_pct=0.0,
        distortion_note="historical replay uses VNINDEX reference only",
    )


def _prepare_data(universe_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        symbol.upper(): frame
        for symbol, frame in ((s, _prepare_single_frame(df)) for s, df in universe_data.items())
        if not frame.empty
    }


def _prepare_single_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    frame = df.copy()
    if "date" not in frame.columns:
        if "time" in frame.columns:
            frame["date"] = frame["time"]
        else:
            frame["date"] = frame.index
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    return _add_feature_columns(frame)


def _add_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    ta_values = _compute_standard_indicators_fast(data)
    mapping = {
        "ma20": "sma20", "ma60": "sma60", "ma200": "sma200",
        "ema20": "ema20", "ema50": "ema50", "ema200": "ema200", "vwma20": "vwma20",
        "rsi": "rsi14", "macd": "macd", "macd_signal": "macd_signal", "macd_hist": "macd_hist",
        "bb_upper": "bb_upper", "bb_mid": "bb_mid", "bb_lower": "bb_lower", "atr": "atr14",
        "adx_14": "adx_14", "dmp_14": "dmp_14", "dmn_14": "dmn_14",
        "aroon_up_14": "aroon_up_14", "aroon_down_14": "aroon_down_14", "aroon_osc_14": "aroon_osc_14",
        "supertrend_10_3": "supertrend_10_3", "supertrend_dir": "supertrend_dir",
        "willr_14": "willr14", "cmo_9": "cmo9", "stoch_k": "stoch_k", "stoch_d": "stoch_d",
        "roc_9": "roc9", "mom_10": "mom10", "kc_lower": "kc_lower", "kc_mid": "kc_mid",
        "kc_upper": "kc_upper", "stdev_14": "stdev14", "linreg_14": "linreg14", "obv": "obv",
    }
    for target, source in mapping.items():
        data[target] = _align_series(ta_values.get(source), data.index)
    data["bb_width"] = _bb_width_series(data["bb_upper"], data["bb_mid"], data["bb_lower"])
    data["bb_width_min20"] = data["bb_width"].shift(1).rolling(20).min()
    data["bb_percent"] = _bb_percent_series(data["close"], data["bb_upper"], data["bb_lower"])
    data["kc_width"] = _channel_width_series(data["kc_upper"], data["kc_mid"], data["kc_lower"])
    data["linreg_slope_14"] = _diff_series(data["linreg_14"], periods=5)
    data["obv_ema20"] = _ema_series(data["obv"], span=20)
    data["volume_ma20"] = data["volume"].rolling(20).mean()
    data["volume_ratio_20"] = data["volume"] / data["volume_ma20"].replace(0, pd.NA)
    data["volume_current"] = data["volume"]
    data["current_price"] = data["close"]
    data["support_20"] = data["low"].shift(1).rolling(20).min()
    data["support_50"] = data["low"].shift(1).rolling(50).min()
    data["resistance_20"] = data["high"].shift(1).rolling(20).max()
    data["resistance_50"] = data["high"].shift(1).rolling(50).max()
    data = _add_ichimoku_columns(data)
    try:
        mf = _prep_ta_money_flow_features(data)
        for col in ["mf_score", "mf_regime", "liquidity_ok", "recent_distribution"]:
            data[col] = mf[col].values
    except Exception:
        data["mf_score"] = 0
        data["mf_regime"] = "NEUTRAL"
        data["liquidity_ok"] = True
        data["recent_distribution"] = False
    return data.set_index("date", drop=False)


def _compute_standard_indicators_fast(data: pd.DataFrame) -> dict:
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    bb = ta.bbands(close, length=20, std=2)
    adx = ta.adx(high, low, close, length=14)
    aroon = ta.aroon(high, low, length=14)
    supertrend = ta.supertrend(high, low, close, length=10, multiplier=3)
    stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    kc = ta.kc(high, low, close, length=20, scalar=2.0, mamode="ema")
    return {
        "sma20": close.rolling(20).mean(),
        "sma60": close.rolling(60).mean(),
        "sma200": close.rolling(200).mean(),
        "ema20": ta.ema(close, length=20),
        "ema50": ta.ema(close, length=50),
        "ema200": ta.ema(close, length=200),
        "vwma20": ta.vwma(close, volume, length=20),
        "rsi14": ta.rsi(close, length=14),
        "macd": _prefixed_col(macd, "MACD_"),
        "macd_signal": _prefixed_col(macd, "MACDs_"),
        "macd_hist": _prefixed_col(macd, "MACDh_"),
        "bb_lower": _prefixed_col(bb, "BBL_"),
        "bb_mid": _prefixed_col(bb, "BBM_"),
        "bb_upper": _prefixed_col(bb, "BBU_"),
        "atr14": ta.atr(high, low, close, length=14),
        "adx_14": _prefixed_col(adx, "ADX_"),
        "dmp_14": _prefixed_col(adx, "DMP_"),
        "dmn_14": _prefixed_col(adx, "DMN_"),
        "aroon_up_14": _prefixed_col(aroon, "AROONU_"),
        "aroon_down_14": _prefixed_col(aroon, "AROOND_"),
        "aroon_osc_14": _prefixed_col(aroon, "AROONOSC_"),
        "supertrend_10_3": _prefixed_col(supertrend, "SUPERT_"),
        "supertrend_dir": _prefixed_col(supertrend, "SUPERTd_"),
        "willr14": ta.willr(high, low, close, length=14),
        "cmo9": ta.cmo(close, length=9),
        "stoch_k": _prefixed_col(stoch, "STOCHk_"),
        "stoch_d": _prefixed_col(stoch, "STOCHd_"),
        "roc9": ta.roc(close, length=9),
        "mom10": ta.mom(close, length=10),
        "kc_lower": _prefixed_col(kc, "KCL"),
        "kc_mid": _prefixed_col(kc, "KCB"),
        "kc_upper": _prefixed_col(kc, "KCU"),
        "stdev14": ta.stdev(close, length=14, ddof=1),
        "linreg14": ta.linreg(close, length=14),
        "obv": ta.obv(close, volume),
    }


def _prefixed_col(frame: pd.DataFrame | None, prefix: str):
    if frame is None or frame.empty:
        return None
    col = next((c for c in frame.columns if str(c).startswith(prefix)), None)
    return frame[col] if col else None


def _add_ichimoku_columns(data: pd.DataFrame) -> pd.DataFrame:
    high = data["high"]
    low = data["low"]
    close = data["close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a_raw = (tenkan + kijun) / 2
    senkou_b_raw = (high.rolling(52).max() + low.rolling(52).min()) / 2
    senkou_a = senkou_a_raw.shift(26)
    senkou_b = senkou_b_raw.shift(26)
    data["ichimoku_tenkan"] = tenkan
    data["ichimoku_kijun"] = kijun
    data["ichimoku_senkou_a"] = senkou_a
    data["ichimoku_senkou_b"] = senkou_b
    data["ichimoku_future_senkou_a"] = senkou_a_raw
    data["ichimoku_future_senkou_b"] = senkou_b_raw
    data["ichimoku_cloud_top"] = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    data["ichimoku_cloud_bottom"] = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    data["ichimoku_future_cloud_green"] = senkou_a_raw >= senkou_b_raw
    data["ichimoku_chikou_confirm"] = close > close.shift(26)
    data["ichimoku_kijun_slope"] = kijun.diff(5)
    data["ichimoku_regime"] = "UNKNOWN"
    data.loc[close > data["ichimoku_cloud_top"], "ichimoku_regime"] = "BULLISH"
    data.loc[close < data["ichimoku_cloud_bottom"], "ichimoku_regime"] = "BEARISH"
    neutral = (close <= data["ichimoku_cloud_top"]) & (close >= data["ichimoku_cloud_bottom"])
    data.loc[neutral, "ichimoku_regime"] = "NEUTRAL"
    return data


def _align_series(series, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(pd.NA, index=index)
    values = pd.Series(series).reset_index(drop=True)
    values.index = index[: len(values)]
    return values.reindex(index)


def _indicator_snapshot(window: pd.DataFrame) -> dict:
    if window.empty:
        return {}
    row = window.iloc[-1]
    keys = [
        "ma20", "ma60", "ma200", "ema20", "ema50", "ema200", "vwma20",
        "current_price", "rsi", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_width_min20", "bb_percent",
        "atr", "adx_14", "dmp_14", "dmn_14", "aroon_up_14", "aroon_down_14",
        "aroon_osc_14", "supertrend_10_3", "supertrend_dir", "willr_14", "cmo_9",
        "stoch_k", "stoch_d", "roc_9", "mom_10", "kc_lower", "kc_mid", "kc_upper",
        "kc_width", "stdev_14", "linreg_14", "linreg_slope_14", "obv", "obv_ema20",
        "volume_current", "volume_ma20", "volume_ratio_20", "ichimoku_tenkan",
        "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b", "ichimoku_cloud_top",
        "ichimoku_cloud_bottom", "ichimoku_future_senkou_a", "ichimoku_future_senkou_b",
        "ichimoku_kijun_slope", "ichimoku_regime",
    ]
    result = {key: _clean_scalar(row.get(key)) for key in keys}
    result["ichimoku_future_cloud_green"] = bool(row.get("ichimoku_future_cloud_green", False))
    result["ichimoku_chikou_confirm"] = bool(row.get("ichimoku_chikou_confirm", False))
    result["macd_bullish_cross_recent"] = _crossed_above_recent(window["macd"], window["macd_signal"], 4)
    result["rsi_signal"] = _rsi_signal(result["rsi"])
    result["ma_trend"] = _ma_trend(result)
    result["ma_phase"] = _ma_phase(result)
    result["macd_signal_label"] = _macd_signal(result)
    result["bollinger_position"] = _bollinger_position(result)
    result["volume_surge"] = _volume_surge(result)
    result["support_levels"] = _levels_below(row, result.get("current_price"), ["support_20", "support_50"])
    result["resistance_levels"] = _levels_above(row, result.get("current_price"), ["resistance_20", "resistance_50"])
    result["confluence_score"] = compute_confluence_score(result)
    return result


def _money_flow_at(window: pd.DataFrame, market_ctx: MarketContext) -> dict:
    row = window.iloc[-1]
    return {
        "regime": str(row.get("mf_regime", "NEUTRAL")),
        "score": int(row.get("mf_score", 0) or 0),
        "liquidity_ok": bool(row.get("liquidity_ok", True)),
        "recent_distribution": bool(row.get("recent_distribution", False)),
        "market_trend": market_ctx.reference_trend,
    }


def _levels_below(row: pd.Series, price, keys: list[str]) -> list[float]:
    levels = []
    for key in keys:
        value = _clean_scalar(row.get(key))
        if value is not None and price and value < price:
            levels.append(round(float(value), 0))
    return sorted(set(levels), reverse=True)[:3]


def _levels_above(row: pd.Series, price, keys: list[str]) -> list[float]:
    levels = []
    for key in keys:
        value = _clean_scalar(row.get(key))
        if value is not None and price and value > price:
            levels.append(round(float(value), 0))
    return sorted(set(levels))[:3]


def _clean_scalar(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value.item() if hasattr(value, "item") else value


def _row_at(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            row = df.loc[date]
        except KeyError:
            return None
        if isinstance(row, pd.DataFrame):
            return row.iloc[0]
        return row
    rows = df.loc[df["date"] == date]
    return None if rows.empty else rows.iloc[0]


def _last_row_on_or_before(df: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    rows = df.loc[df["date"] <= date]
    return None if rows.empty else rows.iloc[-1]


def _next_calendar_date(calendar: list[pd.Timestamp], idx: int) -> pd.Timestamp | None:
    return calendar[idx + 1] if idx + 1 < len(calendar) else None


def _mark_to_market(positions: dict[str, LivePosition], data: dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
    value = 0.0
    for symbol, position in positions.items():
        row = _row_at(data[symbol], date)
        price = float(row["close"]) if row is not None else position.entry_price
        value += price * position.shares
    return value


def _position_value(positions: dict[str, LivePosition], data: dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
    return _mark_to_market(positions, data, date)


def _breakdown(trades: list[dict], key: str) -> list[dict]:
    if not trades:
        return []
    df = pd.DataFrame(trades)
    rows = []
    for value, group in df.groupby(key):
        wins = group[group["pnl_pct"] > 0]
        losses = group[group["pnl_pct"] <= 0]
        gross_win = float(wins["pnl"].sum()) if not wins.empty else 0.0
        gross_loss = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0
        rows.append(
            {
                key: value,
                "trades": int(len(group)),
                "win_rate": float(len(wins) / len(group)),
                "avg_pnl_pct": float(group["pnl_pct"].mean()),
                "total_pnl": float(group["pnl"].sum()),
                "profit_factor": gross_win / gross_loss if gross_loss else float("inf"),
            }
        )
    rows.sort(key=lambda row: (row["total_pnl"], row["trades"]), reverse=True)
    return rows


def _resolve_symbols(universe: str) -> list[str]:
    name = universe.lower()
    if name == "vn30":
        return get_vn30_symbols()
    if name == "vn100":
        return get_vn100_symbols()
    symbols = [item.strip().upper() for item in universe.split(",") if item.strip()]
    if not symbols:
        raise ValueError("Universe must be vn30, vn100, or comma-separated symbols")
    return symbols
