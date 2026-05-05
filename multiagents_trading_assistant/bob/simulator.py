"""simulator.py — Bob (Simulated Trading Analyst) from QuantAgents paper.

Paper: "Bob executes simulated trading backtests on all potential new
strategies (μ't) using historical data. Selects strategies with strongest
historical performance to form new strategy set (μ'). Forwards results for
risk and market analysis."

Weekly trigger: Sunday 20:00 (after last trading day of the week).
Called from pipeline_runner.py APScheduler job.
"""
from __future__ import annotations

import traceback
from datetime import date, timedelta

from multiagents_trading_assistant.backtest.engine import run_universe
from multiagents_trading_assistant.backtest.metrics import compute_metrics
from multiagents_trading_assistant.memory.strategy_memory import (
    StrategyMemory,
    StrategyRecord,
    build_market_snapshot,
)
from multiagents_trading_assistant.bob.reward_tracker import RewardTracker
from multiagents_trading_assistant.services.data_service import (
    get_liquid_symbols,
    get_ohlcv_batch,
    get_vnindex,
)
from multiagents_trading_assistant.screener.trade_screener import get_market_context

# All 22 setups the backtest engine supports.
# Must match engine._STRATEGIES names exactly.
_ALL_SETUPS: list[str] = [
    # Original 14
    "BREAKOUT",
    "FLAG_PENNANT",
    "BB_SQUEEZE",
    "RETEST",
    "SPRING",
    "GOLDEN_CROSS",
    "DOUBLE_BOTTOM",
    "MOMENTUM_SURGE",
    "MACD_CROSSOVER",
    "MA_PULLBACK",
    "INSIDE_BAR",
    "NR7",
    "HAMMER",
    "RSI_BOUNCE",
    # Price Action
    "TREND_PULLBACK",
    "BREAKOUT_RETEST_ENTRY",
    "BULLISH_ENGULFING",
    "PIN_BAR",
    # Ichimoku
    "KUMO_BREAKOUT",
    "TK_CROSS",
    "KIJUN_BOUNCE",
    "KUMO_TWIST_ENTRY",
]

# A strategy is "active" (eligible for Otto) if it clears both thresholds
_MIN_SAMPLE = 10
_MIN_PROFIT_FACTOR = 1.0

# Rolling simulation window in calendar days.
# Cần đủ dài để engine có warmup bars (60) + trade bars có ý nghĩa thống kê.
# 180 ngày ≈ 6 tháng ≈ ~125 trading bars → sau warmup còn ~65 bars để trade.
_LOOKBACK_DAYS = 180


def run_strategy_development_meeting(
    symbols: list[str] | None = None,
    to_date: str | None = None,
    lookback_days: int = _LOOKBACK_DAYS,
    memory: StrategyMemory | None = None,
    tracker: RewardTracker | None = None,
) -> list[StrategyRecord]:
    """Bob's weekly Strategy Development Meeting.

    For each setup in the full strategy pool:
      1. Run simulated trading (backtest) on the liquid universe over the
         rolling window — pure signal evaluation, no money-flow gate.
      2. Compute performance metrics (win_rate, profit_factor, avg_rr).
      3. Mark strategy active when profit_factor >= 1.0 AND sample >= 10.
      4. Write all records to Strategy Memory (ℳₛ).

    Returns: active strategy records — Otto's new strategy set μ'.
    """
    if memory is None:
        memory = StrategyMemory()
    if tracker is None:
        tracker = RewardTracker()

    end = date.fromisoformat(to_date) if to_date else date.today()
    start = end - timedelta(days=lookback_days)
    from_str = start.isoformat()
    to_str = end.isoformat()

    print(f"\n[Bob] Strategy Development Meeting — window: {from_str} → {to_str}")

    # ── Market context for snapshot ──────────────────────────────────────────
    regime = "SIDEWAY"
    vni_change = 0.0
    ma20 = ma60 = 1.0
    vol_ratio = 1.0

    try:
        market_ctx = get_market_context()
        raw = (market_ctx.reference_trend or "").upper()
        regime = raw if raw in {"UPTREND", "SIDEWAY", "DOWNTREND"} else "SIDEWAY"
        vni_change = market_ctx.vni_change_pct
        ma20 = market_ctx.ma20
        ma60 = market_ctx.ma60
    except Exception as e:
        print(f"[Bob] market_context failed ({e}) — using defaults")

    try:
        vni_df = get_vnindex(30)
        if not vni_df.empty and len(vni_df) >= 20:
            avg_vol = float(vni_df["volume"].tail(20).mean())
            if avg_vol > 0:
                vol_ratio = float(vni_df["volume"].iloc[-1]) / avg_vol
    except Exception:
        pass

    market_snapshot = build_market_snapshot(regime, vni_change, ma20, ma60, vol_ratio)
    print(f"[Bob] Market regime: {regime}  VNI {vni_change:+.2f}%  vol_ratio={vol_ratio:.2f}")

    # ── Universe & OHLCV ─────────────────────────────────────────────────────
    if symbols is None:
        symbols = get_liquid_symbols(min_avg_vol=300_000)
    print(f"[Bob] Universe: {len(symbols)} symbols — fetching OHLCV...")
    # lookback_days = simulation window + 60 warmup bars cho indicators
    # +30 buffer để đảm bảo đủ bars sau weekends/holidays
    ohlcv_map = get_ohlcv_batch(symbols, n_days=lookback_days + 90)

    # ── Simulated trading: 1 pass duy nhất, group kết quả theo setup_type ────
    # Thay vì gọi run_universe() 22 lần (mỗi setup 1 lần) →
    # gọi 1 lần với setups=None, engine xử lý tất cả trong cùng 1 walk-forward pass.
    # Giảm thời gian từ O(22 × N) xuống O(N).
    period_label = end.strftime("%Y-%m")
    records: list[StrategyRecord] = []

    print(f"[Bob] Running single-pass simulated trading ({len(symbols)} symbols)...")
    try:
        trades_by_symbol = run_universe(
            ohlcv_map=ohlcv_map,
            from_date=from_str,
            to_date=to_str,
            setups=None,           # tất cả 22 setups trong 1 pass
            rr_ratio=1.5,
            use_money_flow=False,  # pure signal — không để gate làm nhiễu
            max_workers=8,
        )
    except Exception:
        print("[Bob] run_universe FAILED")
        traceback.print_exc()
        return []

    all_trades = [t for ts in trades_by_symbol.values() for t in ts]
    print(f"[Bob] Total simulated trades: {len(all_trades)} — grouping by setup...")

    # Group theo setup_type
    by_setup: dict[str, list] = {s: [] for s in _ALL_SETUPS}
    for t in all_trades:
        if t.setup_type in by_setup:
            by_setup[t.setup_type].append(t)

    print(f"[Bob] Computing metrics for {len(_ALL_SETUPS)} setups...")
    for setup in _ALL_SETUPS:
        trades = by_setup[setup]
        m = compute_metrics(trades)

        n = m["total_trades"]
        sim_pf = float(m.get("profit_factor") or 0.0)
        wr = float(m.get("win_rate_pct") or 0.0) / 100.0
        avg_loss = float(m.get("avg_loss_pct") or 0.0)
        avg_win = float(m.get("avg_win_pct") or 0.0)
        avg_rr = (avg_win / abs(avg_loss)) if avg_loss != 0.0 else 0.0
        avg_pnl = float(m.get("avg_pnl_pct") or 0.0)

        # Dual-reward: kết hợp sim với live performance
        # paper: final_score = wₜˢⁱᵐ × rₜˢⁱᵐ + wₜʳᵉᵃˡ × rₜʳᵉᵃˡ
        live_m = tracker.get_live_metrics(setup=setup)
        live_pf = live_m["profit_factor"]
        w_sim, w_real = tracker.get_adaptive_weights(sim_profit_factor=sim_pf)
        final_pf = w_sim * sim_pf + w_real * live_pf

        is_active = n >= _MIN_SAMPLE and final_pf >= _MIN_PROFIT_FACTOR

        records.append(StrategyRecord(
            strategy_id=f"{setup}_{regime}_{period_label}",
            setup=setup,
            regime=regime,
            win_rate=round(wr, 4),
            profit_factor=round(final_pf, 4),
            avg_rr=round(avg_rr, 4),
            avg_pnl_pct=round(avg_pnl, 4),
            sample_size=n,
            is_active=is_active,
            updated_at=to_str,
            market_snapshot=market_snapshot,
        ))

        tag = "✓ ACTIVE" if is_active else "  —"
        live_note = f"live_n={live_m['sample_size']}" if live_m["sample_size"] > 0 else "live=—"
        print(
            f"  {setup:<26}  n={n:>3}  sim_PF={sim_pf:.2f}"
            f"  final_PF={final_pf:.2f} (w={w_sim:.2f}/{w_real:.2f})"
            f"  {live_note}  {tag}"
        )

    # ── Write to ℳₛ ──────────────────────────────────────────────────────────
    memory.update(records)

    active = [r for r in records if r.is_active]
    print(
        f"\n[Bob] Complete — {len(active)}/{len(records)} strategies active"
        f" → ℳₛ updated ({memory._path})"
    )
    return active
