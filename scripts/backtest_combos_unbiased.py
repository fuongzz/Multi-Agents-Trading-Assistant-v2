"""Backtest STRATEGY COMBOS using the patched (bias-fixed) live_pipeline harness.

Combos are dicts of {name, strategies: [list]}. Loads features once, then for
each combo builds a multi-hypothesis cache (any strategy passes → candidate
flagged) and runs the backtest.

Output: ranked summary across all combos.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from multiagents_trading_assistant.backtest import live_pipeline
from multiagents_trading_assistant.backtest.live_pipeline import (
    LivePipelineBacktestConfig,
    LivePosition,
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
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG, _round_or_none, _setup_type_from_tags
from multiagents_trading_assistant.edge_lab.strategy_sleeves import MVP_EDGE_STRATEGIES
from multiagents_trading_assistant.fetcher import get_liquid150_symbols, get_vn100_symbols, get_vn30_symbols
from multiagents_trading_assistant.quantagents_backtest.metrics import compute_metrics


ROOT = Path(__file__).resolve().parents[1]
MASTER_PARQUET = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"

# ============================================================
# CANDIDATE COMBOS
# ============================================================
# Logic: live baseline + top single + likely diversification combos
COMBOS = [
    {"name": "CORE_MVP9",
     "strategies": list(MVP_EDGE_STRATEGIES)},

    # === Live baseline & near-baseline ablations ===
    {"name": "LIVE_CURRENT_3core",
     "strategies": ["breakout_after_accumulation_v3", "compression_breakout_smt_v1", "mean_reversion_uptrend_ma50_v1"]},
    {"name": "LIVE_minus_compression",
     "strategies": ["breakout_after_accumulation_v3", "mean_reversion_uptrend_ma50_v1"]},
    {"name": "LIVE_minus_meanrev",
     "strategies": ["breakout_after_accumulation_v3", "compression_breakout_smt_v1"]},
    {"name": "LIVE_breakout_only",
     "strategies": ["breakout_after_accumulation_v3"]},

    # === Top single strategies (best on solo run) ===
    {"name": "TOP1_leader_pullback_regime",
     "strategies": ["leader_pullback_market_regime_v3"]},
    {"name": "TOP2_leader_pullback_healthy_v2",
     "strategies": ["leader_pullback_market_healthy_v2"]},
    {"name": "TOP3_breakout_55_smt",
     "strategies": ["breakout_55_smt_v1"]},

    # === "Top 3" candidate: pure top performers ===
    {"name": "PURE_TOP3",
     "strategies": ["leader_pullback_market_regime_v3", "leader_pullback_market_healthy_v2", "breakout_55_smt_v1"]},

    # === Top 3 + diversification (different setup types) ===
    {"name": "TOP3_DIVERSE",
     "strategies": ["leader_pullback_market_regime_v3", "breakout_55_smt_v1", "money_cycle_reset_smt_confirm"]},
    {"name": "TOP3_pullback_breakout_meanrev",
     "strategies": ["leader_pullback_market_regime_v3", "breakout_55_smt_v1", "mean_reversion_uptrend_ma20_v1"]},

    # === Pullback-only combos (3 best pullback strategies) ===
    {"name": "PULLBACK_TRIO",
     "strategies": ["leader_pullback_market_regime_v3", "leader_pullback_market_healthy_v2", "leader_pullback_market_healthy"]},

    # === High-quality, low-trade setups (Sharpe champions) ===
    {"name": "QUALITY_QUARTET",
     "strategies": ["spring_reclaim_smt_v1", "inside_bar_refined_v1", "hammer_refined_v1", "nr7_refined_v1"]},

    # === Big multi-strategy: 5 strats ===
    {"name": "TOP5",
     "strategies": ["leader_pullback_market_regime_v3", "leader_pullback_market_healthy_v2",
                    "breakout_55_smt_v1", "linreg_momentum_refined_v1", "money_cycle_reset_smt_confirm"]},

    # === Live current + top performer ===
    {"name": "LIVE_PLUS_TOP1",
     "strategies": ["breakout_after_accumulation_v3", "compression_breakout_smt_v1",
                    "mean_reversion_uptrend_ma50_v1", "leader_pullback_market_regime_v3"]},

    # === Breakout-leaning multi-strat ===
    {"name": "BREAKOUT_TRIO",
     "strategies": ["breakout_after_accumulation_v3", "breakout_55_smt_v1", "linreg_momentum_refined_v1"]},

    # === Mean-reversion-leaning multi-strat ===
    {"name": "MEANREV_TRIO",
     "strategies": ["mean_reversion_uptrend_ma20_v1", "mean_reversion_uptrend_ma50_v1", "reclaim_ma20_quality_v1"]},

    # === Top winner alone vs. paired ===
    {"name": "TOP1_plus_breakout55",
     "strategies": ["leader_pullback_market_regime_v3", "breakout_55_smt_v1"]},
    {"name": "TOP1_plus_meanrev_ma20",
     "strategies": ["leader_pullback_market_regime_v3", "mean_reversion_uptrend_ma20_v1"]},

    # === Smart-money combos ===
    {"name": "SMT_TRIO",
     "strategies": ["money_cycle_reset_smt_confirm", "smart_money_strong_market_healthy", "accumulation_breakout_smt_v1"]},
]


def _to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def resolve_symbols(universe: str) -> list[str]:
    name = universe.lower()
    if name == "vn30":
        return get_vn30_symbols()
    if name == "vn100":
        return get_vn100_symbols()
    if name in {"liquid150", "liquid_150", "top150"}:
        return get_liquid150_symbols()
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
    index_master = ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"
    frame = pd.read_parquet(index_master)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    if "symbol" in frame.columns:
        frame = frame[frame["symbol"].astype(str).str.upper() == "VNINDEX"]
    frame = frame[(frame["date"] >= warmup_start) & (frame["date"] <= end_ts)]
    return data, frame[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def build_signal_cache_for_combo(features, hypotheses_list, combo_label, clean_symbols):
    """Build cache for COMBO: pass=True if ANY hypothesis in list passes."""
    signal_cache = {}
    for date, daily in features.groupby("date", sort=True):
        latest = daily.reset_index(drop=True)
        day_out = {}
        for _, row in latest.iterrows():
            symbol = str(row["symbol"])
            day_out[symbol] = {
                "strategy_name": combo_label,
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
        # Run each hypothesis, merge best rank per symbol
        for hyp in hypotheses_list:
            mask = evaluate_filters(latest, hyp)
            ranked = rank_candidates(latest[mask], hyp)
            for rank_idx, row in enumerate(ranked.to_dict("records"), start=1):
                symbol = str(row["symbol"])
                current = day_out.get(symbol)
                if not current:
                    continue
                rank_score = float(row.get("_edge_rank") or 0.0)
                # Keep best-ranked across hypotheses
                if rank_score <= float(current.get("edge_rank_score") or 0.0):
                    continue
                current.update(
                    {
                        "strategy_name": hyp.name,
                        "strategy_family": combo_label,
                        "description": hyp.description,
                        "passed": True,
                        "edge_rank": rank_idx,
                        "edge_rank_score": round(rank_score, 4),
                        "setup_type": _setup_type_from_tags(hyp),
                        "filters_failed": [],
                        "risk": hyp.risk,
                    }
                )
        for symbol in clean_symbols:
            day_out.setdefault(
                symbol,
                {"strategy_name": combo_label, "passed": False, "available": False, "error": "No latest feature row"},
            )
        signal_cache[pd.Timestamp(date).normalize()] = day_out
    return signal_cache


def install_edge_cache(signal_cache):
    def cached_edge_signals(symbols, as_of_date=None, strategy_name=None, config_path=None, root=None):
        del strategy_name, config_path, root
        date = pd.Timestamp(as_of_date).normalize() if as_of_date else max(signal_cache)
        day = signal_cache.get(date, {})
        return {s.upper().strip(): day.get(s.upper().strip(),
                {"strategy_name": "", "passed": False, "available": False, "error": "No cached"})
                for s in symbols}
    live_pipeline.get_edge_strategy_signals = cached_edge_signals


def fill_entries_baseline(orders, data, date, day_idx, cash, positions, cfg, counters):
    slots = max(0, cfg.max_positions - len(positions))
    if slots <= 0:
        return cash
    orders = [o for o in orders if o["symbol"] not in positions][:slots]
    if not orders:
        return cash
    budget = cash / max(1, min(slots, len(orders)))
    for order in orders:
        counters["attempted_orders"] += 1
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


def _filter_max_edge_rank(candidates: list[dict], max_edge_rank: int | None) -> list[dict]:
    if not max_edge_rank:
        return candidates
    out = []
    for candidate in candidates:
        rank = _to_float(candidate.get("edge_rank"))
        if rank is not None and rank <= max_edge_rank:
            out.append(candidate)
    return out


def run_one(universe_data, vnindex, cfg, max_edge_rank: int | None = None):
    data = _prepare_data(universe_data)
    index_data = _prepare_single_frame(vnindex)
    calendar = sorted(set(index_data["date"]))
    cash = float(cfg.initial_capital)
    positions = {}
    pending_orders = {}
    trades = []
    equity_rows = []
    counters = {"signal_days": 0, "signals": 0, "attempted_orders": 0, "fills": 0}
    date_to_pos = {d: i for i, d in enumerate(calendar)}
    start_ts = pd.Timestamp(cfg.start_date).normalize() if cfg.start_date else None
    end_ts = pd.Timestamp(cfg.end_date).normalize() if cfg.end_date else None
    last_processed_date = None

    for date in calendar[cfg.lookback:]:
        if end_ts is not None and date > end_ts:
            break
        last_processed_date = date
        day_idx = date_to_pos[date]
        todays_orders = pending_orders.pop(date, [])
        if todays_orders:
            cash = fill_entries_baseline(todays_orders, data, date, day_idx, cash, positions, cfg, counters)
        for symbol in list(positions):
            row = _row_at(data[symbol], date)
            if row is None:
                continue
            position = positions[symbol]
            position.holding_bars = day_idx - position.entry_idx
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
        candidates = _scan_live_candidates(data, date, market_ctx, cfg)
        candidates = _filter_max_edge_rank(candidates, max_edge_rank)
        candidates = [c for c in candidates if c["symbol"] not in positions]
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

    if equity_rows:
        equity_rows[-1]["equity"] = cash + _mark_to_market(positions, data, final_date)
        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["positions"] = len(positions)

    equity_frame = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame()
    equity_curve = equity_frame["equity"].rename("equity") if not equity_frame.empty else pd.Series(dtype=float)
    metrics = compute_metrics(equity_curve, trades) if not equity_curve.empty else {}
    metrics.update({
        "signal_days": counters["signal_days"],
        "signals": counters["signals"],
        "attempted_orders": counters["attempted_orders"],
        "fills": counters["fills"],
    })
    return {"metrics": metrics, "trades": trades, "equity_frame": equity_frame}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-12")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    parser.add_argument("--max-edge-rank", type=int, default=0, help="Keep only candidates with edge_rank <= N; 0 disables")
    parser.add_argument("--combos", default="", help="Comma-separated subset of combo names; empty = all")
    parser.add_argument("--label", default="combos_2025ytd")
    args = parser.parse_args()

    out_dir = ROOT / "backtest_results" / f"combos_unbiased_{args.universe}_{args.label}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = resolve_symbols(args.universe)
    print(f"[combos] universe={args.universe} symbols={len(symbols)}")
    universe_data, vnindex = load_local_history(symbols, args.start, args.end)
    print(f"[combos] OHLCV loaded, vnindex bars={len(vnindex)}")

    t0 = time.time()
    print("[combos] building features once...")
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    print(f"[combos] features {len(features):,} rows, {time.time()-t0:.1f}s")

    all_hypotheses = {h.name: h for h in load_hypotheses(DEFAULT_CONFIG)}
    clean_symbols = sorted({s.upper().strip() for s in symbols})

    combos_to_run = COMBOS
    if args.combos.strip():
        wanted = {c.strip() for c in args.combos.split(",")}
        combos_to_run = [c for c in COMBOS if c["name"] in wanted]
    print(f"[combos] running {len(combos_to_run)} combos")

    base_cfg = LivePipelineBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        max_positions=args.max_positions,
        max_candidates_per_day=args.max_candidates_per_day,
    )

    rows = []
    for idx, combo in enumerate(combos_to_run, 1):
        t_combo = time.time()
        name = combo["name"]
        strats = combo["strategies"]
        print(f"\n[combos] [{idx}/{len(combos_to_run)}] {name} = {strats}")
        try:
            hyps = [all_hypotheses[s] for s in strats if s in all_hypotheses]
            missing = [s for s in strats if s not in all_hypotheses]
            if missing:
                print(f"  ⚠ missing hypotheses: {missing}")
            signal_cache = build_signal_cache_for_combo(features, hyps, name, clean_symbols)
            install_edge_cache(signal_cache)
            cfg = replace(base_cfg, edge_strategy_name=name)
            result = run_one(universe_data, vnindex, cfg, max_edge_rank=args.max_edge_rank or None)
            m = result["metrics"]
            ret_pct = float(m.get("total_return", 0.0)) * 100
            sharpe = float(m.get("sharpe_ratio", 0.0))
            mdd = float(m.get("max_drawdown", 0.0)) * 100
            wr = float(m.get("win_rate", 0.0)) * 100
            n_trades = int(m.get("number_of_trades", 0))
            rows.append({
                "combo": name,
                "n_strategies": len(strats),
                "strategies": " + ".join(strats),
                "total_return_pct": round(ret_pct, 2),
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown_pct": round(mdd, 2),
                "win_rate_pct": round(wr, 2),
                "number_of_trades": n_trades,
                "signal_days": int(m.get("signal_days", 0)),
                "fills": int(m.get("fills", 0)),
                "elapsed_sec": round(time.time() - t_combo, 1),
            })
            print(f"  → return={ret_pct:+.2f}% sharpe={sharpe:.2f} mdd={mdd:.2f}% wr={wr:.1f}% trades={n_trades}  ({time.time()-t_combo:.1f}s)")
            pd.DataFrame(result["trades"]).to_csv(out_dir / f"{name}_trades.csv", index=False)
            if not result["equity_frame"].empty:
                result["equity_frame"].to_csv(out_dir / f"{name}_equity.csv")
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")
            rows.append({"combo": name, "n_strategies": len(strats), "strategies": " + ".join(strats),
                         "total_return_pct": None, "sharpe_ratio": None, "max_drawdown_pct": None,
                         "win_rate_pct": None, "number_of_trades": 0, "signal_days": 0, "fills": 0,
                         "elapsed_sec": round(time.time() - t_combo, 1), "error": str(exc)})

    summary = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False, na_position="last")
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n[combos] saved summary to {out_dir / 'summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
