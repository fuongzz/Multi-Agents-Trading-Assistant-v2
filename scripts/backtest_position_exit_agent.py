"""Research-only backtest for the PositionStateExitAgent.

This harness reuses the bias-fixed combo backtest but adds an EOD position
review step. Reviews at date T are applied only from the next trading date,
which keeps the exit logic causal.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from multiagents_trading_assistant.agentic.position_exit_agent import (
    PositionExitAgentConfig,
    PositionState,
    PositionStateExitAgent,
)
from multiagents_trading_assistant.agentic.market_panic_guard import MarketPanicGuard
from multiagents_trading_assistant.backtest import live_pipeline
from multiagents_trading_assistant.backtest.live_pipeline import (
    LivePipelineBacktestConfig,
    _close_position,
    _edge_exit_decision,
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
from multiagents_trading_assistant.quantagents_backtest.metrics import compute_metrics


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "position_exit_agent_research"


def _feature_history_map(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        symbol: group.sort_values("date").reset_index(drop=True)
        for symbol, group in features.groupby("symbol", sort=False)
    }


def _position_state(position) -> PositionState:
    return PositionState(
        symbol=position.symbol,
        entry_date=position.entry_date.date().isoformat(),
        entry_price=float(position.entry_price),
        stop_loss_price=float(position.stop_loss),
        take_profit_price=float(position.take_profit),
        shares=int(position.shares),
        highest_price=float(position.highest_price or position.entry_price),
        setup_type=str(position.setup_type),
        strategy_name=str(position.edge_strategy_name or ""),
        holding_bars=int(position.holding_bars),
    )


def _apply_pending_reviews(
    *,
    date: pd.Timestamp,
    day_idx: int,
    data: dict[str, pd.DataFrame],
    pending_reviews: dict[pd.Timestamp, dict[str, dict[str, Any]]],
    positions: dict,
    cash: float,
    trades: list[dict],
    cfg: LivePipelineBacktestConfig,
    counters: dict[str, int],
) -> float:
    reviews = pending_reviews.pop(date, {})
    if not reviews:
        return cash
    for symbol, review in list(reviews.items()):
        position = positions.get(symbol)
        if position is None:
            continue
        position.holding_bars = day_idx - position.entry_idx
        row = _row_at(data[symbol], date)
        if row is None:
            continue
        action = review["action"]
        if action == "TAKE_PROFIT_NEXT_OPEN":
            if position.holding_bars < cfg.settlement_bars:
                continue
            exit_price = float(row["open"]) * (1.0 - cfg.slippage_rate)
            cash, trade = _close_position(date, exit_price, "AGENT_TAKE_PROFIT_NEXT_OPEN", position, cash, cfg)
            trade["agent_review_as_of"] = review["as_of_date"]
            trade["agent_reason_codes"] = "|".join(review.get("reason_codes", []))
            trades.append(trade)
            del positions[symbol]
            counters["agent_tp_next_open"] += 1
            continue
        suggested_stop = review.get("suggested_stop_loss")
        if suggested_stop is not None and suggested_stop > position.stop_loss:
            position.stop_loss = float(suggested_stop)
            setattr(position, "agent_stop_active", True)
            counters["agent_raise_stop"] += 1
        if action == "HOLD_RUNNER":
            # From this next bar onward, let the winner run and manage risk via
            # the raised stop. This is causal because the decision was made EOD T.
            position.take_profit = math.inf
            if position.edge_risk:
                position.edge_risk = dict(position.edge_risk)
                position.edge_risk["take_profit"] = None
            setattr(position, "agent_runner_active", True)
            counters["agent_hold_runner"] += 1
    return cash


def _exit_decision_with_agent_stop(row: pd.Series, position, cfg: LivePipelineBacktestConfig) -> tuple[str | None, float]:
    if bool(getattr(position, "agent_stop_active", False)):
        low = float(row["low"])
        open_ = float(row["open"])
        if low <= float(position.stop_loss):
            return "AGENT_RAISED_STOP", min(open_, float(position.stop_loss)) * (1.0 - cfg.slippage_rate)
    return _edge_exit_decision(row, position, cfg)


def run_with_exit_agent(
    universe_data: dict[str, pd.DataFrame],
    vnindex: pd.DataFrame,
    features: pd.DataFrame,
    cfg: LivePipelineBacktestConfig,
    *,
    enabled: bool,
    max_edge_rank: int | None = None,
    agent_config: PositionExitAgentConfig | None = None,
    market_panic_guard_enabled: bool = False,
    entry_market_gate: str = "baseline",
) -> dict[str, Any]:
    data = _prepare_data(universe_data)
    index_data = _prepare_single_frame(vnindex)
    calendar = sorted(set(index_data["date"]))
    date_to_pos = {date: idx for idx, date in enumerate(calendar)}
    feature_by_symbol = _feature_history_map(features)
    agent = PositionStateExitAgent(agent_config)
    panic_guard = MarketPanicGuard() if market_panic_guard_enabled else None

    cash = float(cfg.initial_capital)
    positions: dict[str, Any] = {}
    pending_orders: dict[pd.Timestamp, list[dict]] = {}
    pending_reviews: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []
    counters = {
        "signal_days": 0,
        "signals": 0,
        "attempted_orders": 0,
        "fills": 0,
        "agent_reviews": 0,
        "agent_tp_next_open": 0,
        "agent_hold_runner": 0,
        "agent_raise_stop": 0,
        "panic_guard_days": 0,
        "panic_guard_deferred_exits": 0,
    }
    start_ts = pd.Timestamp(cfg.start_date).normalize() if cfg.start_date else None
    end_ts = pd.Timestamp(cfg.end_date).normalize() if cfg.end_date else None
    last_processed_date = None

    for date in calendar[cfg.lookback:]:
        if end_ts is not None and date > end_ts:
            break
        last_processed_date = date
        day_idx = date_to_pos[date]

        cash = _apply_pending_reviews(
            date=date,
            day_idx=day_idx,
            data=data,
            pending_reviews=pending_reviews,
            positions=positions,
            cash=cash,
            trades=trades,
            cfg=cfg,
            counters=counters,
        )

        todays_orders = pending_orders.pop(date, [])
        if todays_orders:
            cash = combo_harness.fill_entries_baseline(todays_orders, data, date, day_idx, cash, positions, cfg, counters)

        panic_state = panic_guard.state_for_date(data=data, index_data=index_data, date=date) if panic_guard else None
        if panic_state and panic_state.is_panic:
            counters["panic_guard_days"] += 1

        for symbol in list(positions):
            row = _row_at(data[symbol], date)
            if row is None:
                continue
            position = positions[symbol]
            position.holding_bars = day_idx - position.entry_idx
            exit_reason, exit_price = _exit_decision_with_agent_stop(row, position, cfg)
            if panic_guard and panic_guard.should_defer_exit(exit_reason, panic_state):
                counters["panic_guard_deferred_exits"] += 1
                setattr(position, "panic_guard_deferred", int(getattr(position, "panic_guard_deferred", 0) or 0) + 1)
                continue
            if exit_reason and position.holding_bars >= cfg.settlement_bars:
                cash, trade = _close_position(date, exit_price, exit_reason, position, cash, cfg)
                trade["panic_guard_deferred"] = int(getattr(position, "panic_guard_deferred", 0) or 0)
                trades.append(trade)
                del positions[symbol]

        if start_ts is None or date >= start_ts:
            equity = cash + _mark_to_market(positions, data, date)
            equity_rows.append({"date": date, "equity": equity, "cash": cash, "positions": len(positions)})

        next_date = _next_calendar_date(calendar, day_idx)
        if next_date is None:
            continue

        if enabled and (start_ts is None or date >= start_ts):
            day_reviews: dict[str, dict[str, Any]] = {}
            for symbol, position in positions.items():
                history = feature_by_symbol.get(symbol)
                if history is None or history.empty:
                    continue
                review = agent.review(_position_state(position), history, as_of_date=date)
                counters["agent_reviews"] += 1
                if review.action in {"TAKE_PROFIT_NEXT_OPEN", "HOLD_RUNNER", "RAISE_TRAILING_STOP"}:
                    day_reviews[symbol] = {
                        "action": review.action,
                        "as_of_date": review.as_of_date,
                        "reason_codes": list(review.reason_codes),
                        "suggested_stop_loss": review.suggested_stop_loss,
                    }
            if day_reviews:
                pending_reviews.setdefault(next_date, {}).update(day_reviews)

        if start_ts is not None and date < start_ts:
            continue
        market_ctx = _market_context_at(index_data, date, cfg.lookback)
        if not market_ctx.should_trade and not cfg.allow_downtrend_entries:
            continue
        candidates = _scan_live_candidates(data, date, market_ctx, cfg)
        candidates = combo_harness._filter_max_edge_rank(candidates, max_edge_rank)
        candidates = [candidate for candidate in candidates if candidate["symbol"] not in positions]
        candidates = _apply_entry_market_gate(candidates, mode=entry_market_gate)
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
        trade["panic_guard_deferred"] = int(getattr(positions[symbol], "panic_guard_deferred", 0) or 0)
        trades.append(trade)
        del positions[symbol]

    equity_frame = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame()
    equity_curve = equity_frame["equity"].rename("equity") if not equity_frame.empty else pd.Series(dtype=float)
    metrics = compute_metrics(equity_curve, trades) if not equity_curve.empty else {}
    metrics.update(counters)
    return {"metrics": metrics, "trades": trades, "equity_frame": equity_frame}


def _apply_entry_market_gate(candidates: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    """Optional research gate for market risk state at signal time.

    Modes:
    - baseline: keep current historical behavior.
    - riskoff_veto: block every candidate whose edge signal says RISK_OFF.
    - riskoff_recovery: block ordinary RISK_OFF entries, but keep recovery
      candidates with strong edge evidence. This is a research proxy for
      "panic/recovery opportunity" and is evaluated causally at signal date.
    - riskoff_money_reset: softer version; keep money-cycle reset signals in
      RISK_OFF if smart-money evidence is acceptable.
    """
    mode = mode.lower().strip()
    if mode in {"", "baseline", "none"}:
        return candidates
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        edge = (candidate.get("indicators") or {}).get("edge_strategy_analysis") or {}
        regime = str(edge.get("mkt_regime_state") or "").upper()
        if regime != "RISK_OFF":
            out.append(candidate)
            continue
        if mode == "riskoff_veto":
            continue
        if mode == "riskoff_recovery" and _is_riskoff_recovery_candidate(candidate):
            kept = dict(candidate)
            kept["market_gate_action"] = "RISKOFF_RECOVERY_ALLOW"
            out.append(kept)
            continue
        if mode == "riskoff_money_reset" and _is_riskoff_money_reset_candidate(candidate):
            kept = dict(candidate)
            kept["market_gate_action"] = "RISKOFF_MONEY_RESET_ALLOW"
            out.append(kept)
            continue
        if mode not in {"riskoff_veto", "riskoff_recovery", "riskoff_money_reset"}:
            raise ValueError(f"Unsupported entry_market_gate: {mode}")
    return out


def _is_riskoff_recovery_candidate(candidate: dict[str, Any]) -> bool:
    edge = (candidate.get("indicators") or {}).get("edge_strategy_analysis") or {}
    strategy = str(candidate.get("edge_strategy_name") or edge.get("strategy_name") or "")
    edge_rank_score = _safe_float(edge.get("edge_rank_score"), 0.0)
    edge_score = _safe_float(edge.get("edge_score"), 0.0)
    smart_money = _safe_float(edge.get("smart_money_score"), 0.0)
    smart_money_delta = _safe_float(edge.get("smart_money_score_delta"), 0.0)
    rs20 = _safe_float(edge.get("rs_percentile_20"), 0.0)
    ds20 = _safe_float(edge.get("DS20"), 1.0)
    chdm50 = _safe_float(edge.get("CHDM50"), 0.0)

    is_reset = strategy == "money_cycle_reset_smt_confirm"
    strong_flow = smart_money >= 60.0 and smart_money_delta >= 1.0
    strong_relative = rs20 >= 0.65
    not_distributed = ds20 <= 0.70
    cycle_ok = chdm50 >= 50.0
    strong_rank = edge_rank_score >= 2.5 or edge_score >= 70.0
    return is_reset and strong_rank and strong_flow and strong_relative and not_distributed and cycle_ok


def _is_riskoff_money_reset_candidate(candidate: dict[str, Any]) -> bool:
    edge = (candidate.get("indicators") or {}).get("edge_strategy_analysis") or {}
    strategy = str(candidate.get("edge_strategy_name") or edge.get("strategy_name") or "")
    edge_rank_score = _safe_float(edge.get("edge_rank_score"), 0.0)
    smart_money = _safe_float(edge.get("smart_money_score"), 0.0)
    smart_money_delta = _safe_float(edge.get("smart_money_score_delta"), 0.0)
    rs20 = _safe_float(edge.get("rs_percentile_20"), 0.0)
    return (
        strategy == "money_cycle_reset_smt_confirm"
        and edge_rank_score >= 1.25
        and smart_money >= 58.0
        and smart_money_delta >= 0.0
        and rs20 >= 0.60
    )


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Research backtest for position exit agent")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--label", default="pure_top3_exit_agent")
    parser.add_argument("--max-positions", default="3,4")
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    parser.add_argument(
        "--market-panic-guard",
        action="store_true",
        help="Defer hard-stop exits on market-wide panic days instead of fire-selling.",
    )
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.label}_{args.universe}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(args.universe)
    universe_data, vnindex = combo_harness.load_local_history(symbols, args.start, args.end)
    print(f"[exit-agent] universe={args.universe} symbols={len(symbols)}")
    print("[exit-agent] building features once...")
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)

    all_hypotheses = {hyp.name: hyp for hyp in load_hypotheses(DEFAULT_CONFIG)}
    pure_top3 = next(combo for combo in combo_harness.COMBOS if combo["name"] == "PURE_TOP3")
    hyps = [all_hypotheses[name] for name in pure_top3["strategies"]]
    signal_cache = combo_harness.build_signal_cache_for_combo(
        features,
        hyps,
        "PURE_TOP3",
        sorted({symbol.upper().strip() for symbol in symbols}),
    )
    combo_harness.install_edge_cache(signal_cache)

    rows: list[dict[str, Any]] = []
    max_positions_values = [int(item.strip()) for item in args.max_positions.split(",") if item.strip()]
    for max_positions in max_positions_values:
        for enabled in [False, True]:
            label = "EXIT_AGENT" if enabled else "BASELINE"
            cfg = LivePipelineBacktestConfig(
                start_date=args.start,
                end_date=args.end,
                max_positions=max_positions,
                max_candidates_per_day=args.max_candidates_per_day,
                edge_strategy_name="PURE_TOP3",
            )
            print(f"[exit-agent] {label} p{max_positions}")
            result = run_with_exit_agent(
                universe_data,
                vnindex,
                features,
                cfg,
                enabled=enabled,
                market_panic_guard_enabled=args.market_panic_guard,
            )
            metrics = result["metrics"]
            row = {
                "variant": label,
                "combo": "PURE_TOP3",
                "max_positions": max_positions,
                "max_candidates_per_day": args.max_candidates_per_day,
                "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
                "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
                "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
                "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
                "number_of_trades": int(metrics.get("number_of_trades", 0)),
                "agent_reviews": int(metrics.get("agent_reviews", 0)),
                "agent_tp_next_open": int(metrics.get("agent_tp_next_open", 0)),
                "agent_hold_runner": int(metrics.get("agent_hold_runner", 0)),
                "agent_raise_stop": int(metrics.get("agent_raise_stop", 0)),
                "fills": int(metrics.get("fills", 0)),
                "market_panic_guard": bool(args.market_panic_guard),
                "panic_guard_days": int(metrics.get("panic_guard_days", 0)),
                "panic_guard_deferred_exits": int(metrics.get("panic_guard_deferred_exits", 0)),
            }
            rows.append(row)
            guard_suffix = "_panic_guard" if args.market_panic_guard else ""
            stem = f"p{max_positions}_{label.lower()}{guard_suffix}"
            pd.DataFrame(result["trades"]).to_csv(out_dir / f"{stem}_trades.csv", index=False)
            if not result["equity_frame"].empty:
                result["equity_frame"].to_csv(out_dir / f"{stem}_equity.csv")
            print(
                f"  -> ret={row['total_return_pct']:+.2f}% sharpe={row['sharpe_ratio']:.2f} "
                f"mdd={row['max_drawdown_pct']:.2f}% trades={row['number_of_trades']} "
                f"reviews={row['agent_reviews']} panic_defer={row['panic_guard_deferred_exits']}"
            )

    summary = pd.DataFrame(rows).sort_values(["max_positions", "variant"])
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"[exit-agent] saved {out_dir / 'summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
