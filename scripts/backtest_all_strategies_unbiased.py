"""Backtest all 45 hypotheses from vn30_money_smt_hypotheses.json with the
PATCHED (bias-fixed) live_pipeline harness.

Approach:
  - Build feature table ONCE for the universe.
  - For each strategy, rebuild only the per-strategy signal cache (cheap).
  - Run baseline_patched ablation (no entry filter, no PA exit) so the result
    isolates strategy alpha after the same-bar-open bias was removed.
  - Output one consolidated summary CSV.

Outputs to backtest_results/all_strategies_unbiased_<universe>_<start>_<end>/.
"""

from __future__ import annotations

import argparse
import json
import time
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
    _round_or_none,
    _setup_type_from_tags,
)
from multiagents_trading_assistant.fetcher import get_vn100_symbols, get_vn30_symbols
from multiagents_trading_assistant.quantagents_backtest.metrics import compute_metrics


ROOT = Path(__file__).resolve().parents[1]
MASTER_PARQUET = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
HYPOTHESES_PATH = ROOT / "multiagents_trading_assistant" / "edge_lab" / "configs" / "vn30_money_smt_hypotheses.json"


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
        frame = pd.read_parquet(index_master)
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
            if "symbol" in frame.columns:
                frame = frame[frame["symbol"].astype(str).str.upper() == "VNINDEX"]
            frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
            if not frame.empty:
                return frame[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
    raise FileNotFoundError("VNINDEX cached history not found")


def build_signal_cache_for_strategy(
    features: pd.DataFrame,
    hypothesis,
    strategy_name: str,
    clean_symbols: list[str],
) -> dict[pd.Timestamp, dict[str, dict[str, Any]]]:
    """Filter pre-loaded features for ONE hypothesis. Cheap because features are reused."""
    signal_cache: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    for date, daily in features.groupby("date", sort=True):
        latest = daily.reset_index(drop=True)
        day_out: dict[str, dict[str, Any]] = {}
        for _, row in latest.iterrows():
            symbol = str(row["symbol"])
            day_out[symbol] = {
                "strategy_name": strategy_name,
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
        mask = evaluate_filters(latest, hypothesis)
        ranked = rank_candidates(latest[mask], hypothesis)
        for rank_idx, row in enumerate(ranked.to_dict("records"), start=1):
            symbol = str(row["symbol"])
            current = day_out.get(symbol)
            if not current:
                continue
            current.update(
                {
                    "strategy_name": hypothesis.name,
                    "description": hypothesis.description,
                    "passed": True,
                    "edge_rank": rank_idx,
                    "edge_rank_score": round(float(row.get("_edge_rank") or 0.0), 4),
                    "setup_type": _setup_type_from_tags(hypothesis),
                    "filters_failed": [],
                    "risk": hypothesis.risk,
                }
            )
        for symbol in clean_symbols:
            day_out.setdefault(
                symbol,
                {"strategy_name": strategy_name, "passed": False, "available": False, "error": "No latest feature row"},
            )
        signal_cache[pd.Timestamp(date).normalize()] = day_out
    return signal_cache


def install_edge_cache(signal_cache):
    def cached_edge_signals(symbols, as_of_date=None, strategy_name=None, config_path=None, root=None):
        del strategy_name, config_path, root
        date = pd.Timestamp(as_of_date).normalize() if as_of_date else max(signal_cache)
        day = signal_cache.get(date, {})
        return {
            s.upper().strip(): day.get(s.upper().strip(),
                {"strategy_name": "", "passed": False, "available": False, "error": "No cached"})
            for s in symbols
        }
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


def run_one_strategy(universe_data, vnindex, cfg) -> dict:
    data = _prepare_data(universe_data)
    index_data = _prepare_single_frame(vnindex)
    calendar = sorted(set(index_data["date"]))
    cash = float(cfg.initial_capital)
    positions: dict[str, LivePosition] = {}
    pending_orders: dict[pd.Timestamp, list[dict]] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []
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
    parser.add_argument("--strategies", default="", help="Comma-separated subset; empty = all 45 from config")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Hypothesis config JSON path")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    args = parser.parse_args()

    config_stem = Path(args.config).stem
    out_dir = (
        ROOT
        / "backtest_results"
        / (
            f"all_strategies_unbiased_{config_stem}_{args.universe}_"
            f"p{args.max_positions}_c{args.max_candidates_per_day}_{args.start}_{args.end}"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = resolve_symbols(args.universe)
    print(f"[meta] universe={args.universe} symbols={len(symbols)}")

    # Load history & features ONCE
    print("[meta] loading OHLCV history...")
    universe_data, vnindex = load_local_history(symbols, args.start, args.end)
    print(f"[meta] history loaded: {len(universe_data)} symbols, vnindex bars={len(vnindex)}")

    print("[meta] building feature table (once)...")
    t0 = time.time()
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    if features.empty:
        raise RuntimeError("Empty feature table")
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    print(f"[meta] features: {len(features):,} rows, dates={features['date'].nunique()}, took {time.time()-t0:.1f}s")

    # Load all hypotheses
    hypotheses = load_hypotheses(args.config)
    if args.strategies.strip():
        wanted = {s.strip() for s in args.strategies.split(",")}
        hypotheses = [h for h in hypotheses if h.name in wanted]
    print(f"[meta] running {len(hypotheses)} strategies")

    clean_symbols = sorted({s.upper().strip() for s in symbols})
    base_cfg = LivePipelineBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        max_positions=args.max_positions,
        max_candidates_per_day=args.max_candidates_per_day,
    )

    rows = []
    for idx, hyp in enumerate(hypotheses, 1):
        t_strat = time.time()
        print(f"\n[meta] [{idx}/{len(hypotheses)}] {hyp.name}")
        try:
            signal_cache = build_signal_cache_for_strategy(features, hyp, hyp.name, clean_symbols)
            install_edge_cache(signal_cache)
            cfg = replace(base_cfg, edge_strategy_name=hyp.name)
            result = run_one_strategy(universe_data, vnindex, cfg)
            m = result["metrics"]
            ret_pct = float(m.get("total_return", 0.0)) * 100
            sharpe = float(m.get("sharpe_ratio", 0.0))
            mdd = float(m.get("max_drawdown", 0.0)) * 100
            wr = float(m.get("win_rate", 0.0)) * 100
            n_trades = int(m.get("number_of_trades", 0))
            rows.append({
                "strategy": hyp.name,
                "total_return_pct": round(ret_pct, 2),
                "sharpe_ratio": round(sharpe, 3),
                "max_drawdown_pct": round(mdd, 2),
                "win_rate_pct": round(wr, 2),
                "number_of_trades": n_trades,
                "signal_days": int(m.get("signal_days", 0)),
                "fills": int(m.get("fills", 0)),
                "elapsed_sec": round(time.time() - t_strat, 1),
            })
            print(f"  → return={ret_pct:+.2f}% sharpe={sharpe:.2f} mdd={mdd:.2f}% wr={wr:.1f}% trades={n_trades}  ({time.time()-t_strat:.1f}s)")
            # Save per-strategy trades
            pd.DataFrame(result["trades"]).to_csv(out_dir / f"{hyp.name}_trades.csv", index=False)
            if not result["equity_frame"].empty:
                result["equity_frame"].to_csv(out_dir / f"{hyp.name}_equity.csv")
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")
            rows.append({"strategy": hyp.name, "total_return_pct": None, "sharpe_ratio": None,
                         "max_drawdown_pct": None, "win_rate_pct": None, "number_of_trades": 0,
                         "signal_days": 0, "fills": 0, "elapsed_sec": round(time.time() - t_strat, 1),
                         "error": str(exc)})

    summary = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False, na_position="last")
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n[meta] saved summary to {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
