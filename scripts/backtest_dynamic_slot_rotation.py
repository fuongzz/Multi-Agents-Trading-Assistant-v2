"""Backtest dynamic slot expansion and rotation exits for the Core MVP9 pool."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
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
from multiagents_trading_assistant.edge_lab.hypothesis import load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG
from multiagents_trading_assistant.edge_lab.strategy_sleeves import MVP_EDGE_STRATEGIES
from multiagents_trading_assistant.quantagents_backtest.metrics import compute_metrics


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "dynamic_slot_rotation"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _edge(candidate: dict[str, Any]) -> dict[str, Any]:
    return (candidate.get("indicators") or {}).get("edge_strategy_analysis") or {}


def _candidate_score(candidate: dict[str, Any]) -> float:
    return _to_float(candidate.get("edge_rank_score") or _edge(candidate).get("edge_rank_score"), 0.0)


def _position_score(position: LivePosition) -> float:
    return _to_float(position.edge_rank_score, 0.0)


def _effective_max_positions(candidates: list[dict[str, Any]], *, base: int, expanded: int) -> tuple[int, str]:
    if expanded <= base:
        return base, "BASE_MAX_POSITIONS"
    scores = sorted((_candidate_score(item) for item in candidates), reverse=True)
    top = scores[: min(8, len(scores))]
    avg_top = sum(top) / len(top) if top else 0.0
    regimes = [str(_edge(item).get("mkt_regime_state") or "").upper() for item in candidates]
    risk_off = sum(1 for regime in regimes if regime == "RISK_OFF")
    constructive = sum(
        1
        for regime in regimes
        if regime in {"RISK_ON_UPTREND", "RISK_ON_RECOVERY", "NEUTRAL_UPTREND", "NEUTRAL"}
    )
    if len(candidates) >= 8 and avg_top >= 2.5 and constructive >= max(4, len(candidates) // 2) and risk_off <= max(1, len(candidates) // 4):
        return expanded, "EXPAND_STRONG_BREADTH_AND_CANDIDATES"
    return base, "BASE_MAX_POSITIONS"


def _rotation_reason(position: LivePosition, candidate: dict[str, Any], price: float, day_idx: int) -> str | None:
    if position.entry_price <= 0:
        return None
    pnl_pct = (price / position.entry_price - 1.0) * 100.0
    peak = max(float(position.highest_price or position.entry_price), price)
    drawdown_from_peak = (price / peak - 1.0) * 100.0 if peak > 0 else 0.0
    holding_bars = max(0, day_idx - position.entry_idx)
    score_gap = _candidate_score(candidate) - _position_score(position)

    if pnl_pct <= -3.0 and score_gap >= 0.75:
        return "ROTATE_WEAK_LOSER_TO_STRONGER_CANDIDATE"
    if holding_bars >= 15 and abs(pnl_pct) <= 2.0 and score_gap >= 1.0:
        return "ROTATE_STALE_SIDEWAYS_TO_STRONGER_CANDIDATE"
    if pnl_pct >= 8.0 and drawdown_from_peak <= -3.0 and score_gap >= 0.5:
        return "ROTATE_WINNER_WEAKENING_TO_STRONGER_CANDIDATE"
    if _position_score(position) > 0 and score_gap >= 1.5 and pnl_pct <= 4.0:
        return "ROTATE_LOWER_SCORE_TO_STRONGER_CANDIDATE"
    return None


def _choose_rotation(
    *,
    candidate: dict[str, Any],
    data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    day_idx: int,
    positions: dict[str, LivePosition],
) -> tuple[str, LivePosition, float, str] | None:
    if _candidate_score(candidate) < 2.5:
        return None
    choices: list[tuple[float, str, LivePosition, float, str]] = []
    for symbol, position in positions.items():
        row = _row_at(data[symbol], date)
        if row is None:
            continue
        price = float(row["open"])
        reason = _rotation_reason(position, candidate, price, day_idx)
        if not reason:
            continue
        pnl_pct = (price / position.entry_price - 1.0) * 100.0 if position.entry_price > 0 else 0.0
        priority = _candidate_score(candidate) - _position_score(position) + max(0.0, -pnl_pct / 10.0)
        choices.append((priority, symbol, position, price, reason))
    if not choices:
        return None
    choices.sort(key=lambda item: item[0], reverse=True)
    _priority, symbol, position, price, reason = choices[0]
    return symbol, position, price, reason


def _fill_one(
    *,
    order: dict[str, Any],
    data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    day_idx: int,
    cash: float,
    positions: dict[str, LivePosition],
    cfg: LivePipelineBacktestConfig,
    budget: float,
    counters: dict[str, int],
) -> float:
    row = _row_at(data[order["symbol"]], date)
    if row is None:
        return cash
    entry = float(row["open"]) * (1.0 + cfg.slippage_rate)
    if entry <= 0:
        return cash
    gross_shares = int(budget // entry)
    shares = (gross_shares // cfg.lot_size) * cfg.lot_size if cfg.lot_size > 1 else gross_shares
    if shares <= 0:
        return cash
    cost = shares * entry * (1.0 + cfg.commission_rate)
    if cost > cash:
        return cash
    indicators = order["indicators"]
    edge_risk = dict(order.get("edge_risk") or {})
    stop_loss = _edge_initial_sl(entry, indicators, edge_risk) if edge_risk else _compute_initial_sl(entry, indicators)
    if stop_loss <= 0 or stop_loss >= entry:
        return cash

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
        edge_score=combo_harness._to_float(order.get("edge_score")),
        edge_rank=combo_harness._to_float(order.get("edge_rank")),
        edge_rank_score=combo_harness._to_float(order.get("edge_rank_score")),
        edge_risk=edge_risk or None,
        entry_atr=combo_harness._to_float(indicators.get("atr")),
        highest_price=entry,
    )
    return cash


def _fill_entries_dynamic_rotation(
    *,
    orders: list[dict[str, Any]],
    data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    day_idx: int,
    cash: float,
    positions: dict[str, LivePosition],
    cfg: LivePipelineBacktestConfig,
    trades: list[dict[str, Any]],
    counters: dict[str, int],
    base_max_positions: int,
    expanded_max_positions: int,
    allow_rotation: bool,
) -> float:
    if not orders:
        return cash
    effective_max, reason = _effective_max_positions(orders, base=base_max_positions, expanded=expanded_max_positions)
    counters[f"policy_{reason.lower()}"] = counters.get(f"policy_{reason.lower()}", 0) + 1
    orders = [order for order in orders if order["symbol"] not in positions]
    for order in orders:
        counters["attempted_orders"] += 1
        if len(positions) >= effective_max:
            if not allow_rotation:
                counters["rotation_no_slot"] += 1
                continue
            rotation = _choose_rotation(candidate=order, data=data, date=date, day_idx=day_idx, positions=positions)
            if rotation is None:
                counters["rotation_no_slot"] += 1
                continue
            symbol, position, exit_price, exit_reason = rotation
            cash, trade = _close_position(date, exit_price * (1.0 - cfg.slippage_rate), exit_reason, position, cash, cfg)
            trade["rotation_replacement"] = order["symbol"]
            trade["rotation_new_score"] = round(_candidate_score(order), 4)
            trades.append(trade)
            del positions[symbol]
            counters["rotation_exits"] += 1

        slots_left = max(1, effective_max - len(positions))
        budget = cash / slots_left
        before = cash
        cash = _fill_one(
            order=order,
            data=data,
            date=date,
            day_idx=day_idx,
            cash=cash,
            positions=positions,
            cfg=cfg,
            budget=budget,
            counters=counters,
        )
        if cash == before:
            counters["unfilled_orders"] += 1
    return cash


def run_dynamic_rotation(
    universe_data: dict[str, pd.DataFrame],
    vnindex: pd.DataFrame,
    cfg: LivePipelineBacktestConfig,
    *,
    expanded_max_positions: int,
    allow_rotation: bool = True,
) -> dict[str, Any]:
    data = _prepare_data(universe_data)
    index_data = _prepare_single_frame(vnindex)
    calendar = sorted(set(index_data["date"]))
    date_to_pos = {date: idx for idx, date in enumerate(calendar)}
    cash = float(cfg.initial_capital)
    positions: dict[str, LivePosition] = {}
    pending_orders: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    counters = {
        "signal_days": 0,
        "signals": 0,
        "attempted_orders": 0,
        "fills": 0,
        "rotation_exits": 0,
        "rotation_no_slot": 0,
        "unfilled_orders": 0,
    }
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
            cash = _fill_entries_dynamic_rotation(
                orders=todays_orders,
                data=data,
                date=date,
                day_idx=day_idx,
                cash=cash,
                positions=positions,
                cfg=cfg,
                trades=trades,
                counters=counters,
                base_max_positions=cfg.max_positions,
                expanded_max_positions=expanded_max_positions,
                allow_rotation=allow_rotation,
            )

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
        candidates = [candidate for candidate in candidates if candidate["symbol"] not in positions]
        if candidates:
            counters["signal_days"] += 1
            selected = candidates[: cfg.max_candidates_per_day]
            counters["signals"] += len(selected)
            pending_orders.setdefault(next_date, []).extend(selected)

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
    metrics.update(counters)
    return {"metrics": metrics, "trades": trades, "equity_frame": equity_frame}


def _row(label: str, metrics: dict[str, Any], *, start: str, end: str, max_positions: int, expanded: int) -> dict[str, Any]:
    trades = int(metrics.get("number_of_trades", 0))
    weeks = max(1.0, (pd.Timestamp(end) - pd.Timestamp(start)).days / 7.0)
    return {
        "variant": label,
        "start": start,
        "end": end,
        "base_max_positions": max_positions,
        "expanded_max_positions": expanded,
        "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
        "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
        "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
        "number_of_trades": trades,
        "trades_per_week": round(trades / weeks, 2),
        "fills": int(metrics.get("fills", 0)),
        "rotation_exits": int(metrics.get("rotation_exits", 0)),
        "rotation_no_slot": int(metrics.get("rotation_no_slot", 0)),
        "policy_expand_days": int(metrics.get("policy_expand_strong_breadth_and_candidates", 0)),
        "policy_base_days": int(metrics.get("policy_base_max_positions", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Core MVP9 dynamic slots and rotation exits")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--label", default="core_mvp9_dynamic_rotation")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--expanded-max-positions", type=int, default=8)
    parser.add_argument("--max-candidates-per-day", type=int, default=15)
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.label}_{args.universe}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(args.universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, args.start, args.end)
    print(f"[dyn-rot] universe={args.universe} symbols={len(symbols)}")
    print("[dyn-rot] building features once...")
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)

    hypotheses = {item.name: item for item in load_hypotheses(DEFAULT_CONFIG)}
    missing = [name for name in MVP_EDGE_STRATEGIES if name not in hypotheses]
    if missing:
        raise KeyError(f"Missing MVP hypotheses: {missing}")
    signal_cache = combo_harness.build_signal_cache_for_combo(
        features,
        [hypotheses[name] for name in MVP_EDGE_STRATEGIES],
        "PURE_TOP3_PLUS_SMT_LIVE",
        clean_symbols,
    )
    combo_harness.install_edge_cache(signal_cache)

    cfg = LivePipelineBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        max_positions=args.max_positions,
        max_candidates_per_day=args.max_candidates_per_day,
        edge_strategy_name="PURE_TOP3_PLUS_SMT_LIVE",
    )
    print("[dyn-rot] baseline...")
    baseline = combo_harness.run_one(universe_data, vnindex, cfg)
    print("[dyn-rot] dynamic slots only...")
    dynamic_slots = run_dynamic_rotation(
        universe_data,
        vnindex,
        replace(cfg),
        expanded_max_positions=args.expanded_max_positions,
        allow_rotation=False,
    )
    print("[dyn-rot] rotation only...")
    rotation_only = run_dynamic_rotation(
        universe_data,
        vnindex,
        replace(cfg),
        expanded_max_positions=args.max_positions,
        allow_rotation=True,
    )
    print("[dyn-rot] dynamic slots + rotation...")
    dynamic = run_dynamic_rotation(
        universe_data,
        vnindex,
        replace(cfg),
        expanded_max_positions=args.expanded_max_positions,
        allow_rotation=True,
    )

    rows = [
        _row("BASELINE_P5", baseline["metrics"], start=args.start, end=args.end, max_positions=args.max_positions, expanded=args.max_positions),
        _row("DYNAMIC_SLOTS_P5_TO_P8", dynamic_slots["metrics"], start=args.start, end=args.end, max_positions=args.max_positions, expanded=args.expanded_max_positions),
        _row("ROTATION_ONLY_P5", rotation_only["metrics"], start=args.start, end=args.end, max_positions=args.max_positions, expanded=args.max_positions),
        _row("DYNAMIC_ROTATION_P5_TO_P8", dynamic["metrics"], start=args.start, end=args.end, max_positions=args.max_positions, expanded=args.expanded_max_positions),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(baseline["trades"]).to_csv(out_dir / "baseline_trades.csv", index=False)
    pd.DataFrame(dynamic_slots["trades"]).to_csv(out_dir / "dynamic_slots_trades.csv", index=False)
    pd.DataFrame(rotation_only["trades"]).to_csv(out_dir / "rotation_only_trades.csv", index=False)
    pd.DataFrame(dynamic["trades"]).to_csv(out_dir / "dynamic_rotation_trades.csv", index=False)
    if not baseline["equity_frame"].empty:
        baseline["equity_frame"].to_csv(out_dir / "baseline_equity.csv")
    if not dynamic["equity_frame"].empty:
        dynamic["equity_frame"].to_csv(out_dir / "dynamic_rotation_equity.csv")
    if not dynamic_slots["equity_frame"].empty:
        dynamic_slots["equity_frame"].to_csv(out_dir / "dynamic_slots_equity.csv")
    if not rotation_only["equity_frame"].empty:
        rotation_only["equity_frame"].to_csv(out_dir / "rotation_only_equity.csv")
    print(f"[dyn-rot] saved {out_dir / 'summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
