"""Research backtest for the PositionAddOnAgent.

This harness tests pyramiding only on existing winning positions. Add-on
reviews happen EOD T and fills happen no earlier than next open T+1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from scripts.backtest_position_exit_agent import (
    _apply_pending_reviews,
    _exit_decision_with_agent_stop,
    _position_state,
)
from multiagents_trading_assistant.agentic.position_add_on_agent import (
    PositionAddOnAgent,
    PositionAddOnAgentConfig,
    PositionAddOnState,
)
from multiagents_trading_assistant.agentic.market_panic_guard import MarketPanicGuard
from multiagents_trading_assistant.agentic.position_exit_agent import PositionStateExitAgent
from multiagents_trading_assistant.backtest.live_pipeline import (
    LivePipelineBacktestConfig,
    _close_position,
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
from multiagents_trading_assistant.research.macro_regime.macro_market_regime import (
    MacroRegimeConfig,
    build_macro_regime_frame,
    load_macro_csv,
    regime_allows_add_on,
)
from multiagents_trading_assistant.research.macro_regime.market_phase_quality import (
    MarketPhaseConfig,
    build_market_phase_frame,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "position_add_on_agent_research"


COMBO_SETS: list[dict[str, Any]] = [
    {
        "name": "PURE_TOP3",
        "strategies": [
            "leader_pullback_market_regime_v3",
            "leader_pullback_market_healthy_v2",
            "breakout_55_smt_v1",
        ],
    },
    {
        "name": "PURE_TOP3_PLUS_SMT_LIVE",
        "strategies": [
            "leader_pullback_market_regime_v3",
            "leader_pullback_market_healthy_v2",
            "breakout_55_smt_v1",
            "money_cycle_reset_smt_confirm",
            "smart_money_strong_market_healthy",
            "accumulation_breakout_smt_v1",
            "breakout_after_accumulation_v3",
            "compression_breakout_smt_v1",
            "mean_reversion_uptrend_ma50_v1",
        ],
    },
]


def _feature_history_map(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        symbol: group.sort_values("date").reset_index(drop=True)
        for symbol, group in features.groupby("symbol", sort=False)
    }


def _add_on_state(position) -> PositionAddOnState:
    return PositionAddOnState(
        symbol=position.symbol,
        entry_date=position.entry_date.date().isoformat(),
        entry_price=float(position.entry_price),
        stop_loss_price=float(position.stop_loss),
        take_profit_price=float(position.take_profit),
        shares=int(position.shares),
        add_on_count=int(getattr(position, "add_on_count", 0) or 0),
        highest_price=float(position.highest_price or position.entry_price),
        setup_type=str(position.setup_type),
        strategy_name=str(position.edge_strategy_name or ""),
        holding_bars=int(position.holding_bars),
    )


def _apply_pending_add_ons(
    *,
    date: pd.Timestamp,
    data: dict[str, pd.DataFrame],
    pending_add_ons: dict[pd.Timestamp, dict[str, dict[str, Any]]],
    positions: dict[str, Any],
    cash: float,
    equity: float,
    cfg: LivePipelineBacktestConfig,
    counters: dict[str, int],
) -> float:
    reviews = pending_add_ons.pop(date, {})
    if not reviews:
        return cash
    slot_budget = equity / max(1, cfg.max_positions)
    for symbol, review in list(reviews.items()):
        position = positions.get(symbol)
        if position is None:
            continue
        row = _row_at(data[symbol], date)
        if row is None:
            continue
        entry = float(row["open"]) * (1.0 + cfg.slippage_rate)
        if entry <= 0:
            continue
        budget = min(cash, slot_budget * float(review.get("budget_fraction_of_slot") or 0.0))
        gross_shares = int(budget // entry)
        shares = (gross_shares // cfg.lot_size) * cfg.lot_size if cfg.lot_size > 1 else gross_shares
        if shares <= 0:
            continue
        cost = shares * entry * (1.0 + cfg.commission_rate)
        if cost > cash:
            continue

        old_shares = int(position.shares)
        old_cost = float(position.entry_price) * old_shares
        new_cost = entry * shares
        position.shares = old_shares + shares
        position.entry_price = (old_cost + new_cost) / position.shares
        position.stop_loss = max(float(position.stop_loss), float(review.get("suggested_stop_loss") or 0.0))
        position.highest_price = max(float(position.highest_price or entry), entry)
        setattr(position, "add_on_count", int(getattr(position, "add_on_count", 0) or 0) + 1)
        add_on_dates = list(getattr(position, "add_on_dates", []) or [])
        add_on_dates.append(date.date().isoformat())
        setattr(position, "add_on_dates", add_on_dates)
        setattr(position, "add_on_active", True)
        cash -= cost
        counters["add_on_fills"] += 1
        counters["add_on_shares"] += shares
    return cash


def _regime_value(
    regime_frame: pd.DataFrame | None,
    date: pd.Timestamp,
    column: str,
    default: float,
) -> float:
    if regime_frame is None or regime_frame.empty or column not in regime_frame.columns:
        return default
    rows = regime_frame[regime_frame["date"] <= pd.Timestamp(date).normalize()]
    if rows.empty:
        return default
    try:
        value = rows.iloc[-1].get(column, default)
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def run_with_add_on_agent(
    universe_data: dict[str, pd.DataFrame],
    vnindex: pd.DataFrame,
    features: pd.DataFrame,
    cfg: LivePipelineBacktestConfig,
    *,
    exit_agent_enabled: bool,
    add_on_agent_enabled: bool,
    add_on_config: PositionAddOnAgentConfig | None = None,
    add_on_regime: pd.DataFrame | None = None,
    add_on_regime_column: str = "add_on_regime_v2",
    add_on_budget_multiplier_column: str | None = None,
    market_panic_guard_enabled: bool = False,
) -> dict[str, Any]:
    data = _prepare_data(universe_data)
    index_data = _prepare_single_frame(vnindex)
    calendar = sorted(set(index_data["date"]))
    date_to_pos = {date: idx for idx, date in enumerate(calendar)}
    feature_by_symbol = _feature_history_map(features)
    exit_agent = PositionStateExitAgent()
    add_on_agent = PositionAddOnAgent(add_on_config)
    panic_guard = MarketPanicGuard() if market_panic_guard_enabled else None

    cash = float(cfg.initial_capital)
    positions: dict[str, Any] = {}
    pending_orders: dict[pd.Timestamp, list[dict]] = {}
    pending_reviews: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    pending_add_ons: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
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
        "add_on_reviews": 0,
        "add_on_signals": 0,
        "add_on_fills": 0,
        "add_on_shares": 0,
        "panic_guard_days": 0,
        "panic_guard_deferred_exits": 0,
    }
    start_ts = pd.Timestamp(cfg.start_date).normalize() if cfg.start_date else None
    end_ts = pd.Timestamp(cfg.end_date).normalize() if cfg.end_date else None
    last_processed_date = None

    for date in calendar[cfg.lookback :]:
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

        equity_before_add = cash + _mark_to_market(positions, data, date)
        cash = _apply_pending_add_ons(
            date=date,
            data=data,
            pending_add_ons=pending_add_ons,
            positions=positions,
            cash=cash,
            equity=equity_before_add,
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
                trade["add_on_count"] = int(getattr(position, "add_on_count", 0) or 0)
                trade["add_on_dates"] = "|".join(getattr(position, "add_on_dates", []) or [])
                trade["panic_guard_deferred"] = int(getattr(position, "panic_guard_deferred", 0) or 0)
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

        if exit_agent_enabled:
            day_reviews: dict[str, dict[str, Any]] = {}
            for symbol, position in positions.items():
                history = feature_by_symbol.get(symbol)
                if history is None or history.empty:
                    continue
                review = exit_agent.review(_position_state(position), history, as_of_date=date)
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

        market_ctx = _market_context_at(index_data, date, cfg.lookback)
        market_uptrend = market_ctx.should_trade and market_ctx.reference_trend == "UPTREND"
        macro_regime_ok = (
            regime_allows_add_on(add_on_regime, date, column=add_on_regime_column)
            if add_on_regime is not None
            else True
        )

        if add_on_agent_enabled and market_uptrend and macro_regime_ok:
            budget_multiplier = _regime_value(
                add_on_regime,
                date,
                add_on_budget_multiplier_column,
                1.0,
            ) if add_on_budget_multiplier_column else 1.0
            if budget_multiplier <= 0:
                continue
            day_add_ons: dict[str, dict[str, Any]] = {}
            for symbol, position in positions.items():
                history = feature_by_symbol.get(symbol)
                if history is None or history.empty:
                    continue
                review = add_on_agent.review(_add_on_state(position), history, as_of_date=date)
                counters["add_on_reviews"] += 1
                if review.action == "ADD_ON_NEXT_OPEN":
                    day_add_ons[symbol] = {
                        "as_of_date": review.as_of_date,
                        "reason_codes": list(review.reason_codes),
                        "suggested_stop_loss": review.suggested_stop_loss,
                        "budget_fraction_of_slot": review.budget_fraction_of_slot * budget_multiplier,
                    }
                    counters["add_on_signals"] += 1
            if day_add_ons:
                pending_add_ons.setdefault(next_date, {}).update(day_add_ons)

        if not market_ctx.should_trade and not cfg.allow_downtrend_entries:
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
        trade["add_on_count"] = int(getattr(positions[symbol], "add_on_count", 0) or 0)
        trade["add_on_dates"] = "|".join(getattr(positions[symbol], "add_on_dates", []) or [])
        trade["panic_guard_deferred"] = int(getattr(positions[symbol], "panic_guard_deferred", 0) or 0)
        trades.append(trade)
        del positions[symbol]

    equity_frame = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame()
    equity_curve = equity_frame["equity"].rename("equity") if not equity_frame.empty else pd.Series(dtype=float)
    metrics = compute_metrics(equity_curve, trades) if not equity_curve.empty else {}
    metrics.update(counters)
    return {"metrics": metrics, "trades": trades, "equity_frame": equity_frame}


def main() -> None:
    parser = argparse.ArgumentParser(description="Research backtest for position add-on agent")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--label", default="add_on_agent_2025now")
    parser.add_argument("--combo-set", default="PURE_TOP3_PLUS_SMT_LIVE")
    parser.add_argument("--max-positions", default="5")
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    parser.add_argument(
        "--add-on-regime",
        choices=[
            "basic",
            "macro_v2",
            "macro_v3_goldilocks",
            "phase_v1",
            "phase_v2",
            "phase_throttle_v1",
            "phase_risk_throttle_v1",
        ],
        default="basic",
    )
    parser.add_argument("--cpi-csv", default="")
    parser.add_argument("--fx-csv", default="")
    parser.add_argument("--interest-csv", default="")
    parser.add_argument("--add-on-budget-fraction", type=float, default=0.50)
    parser.add_argument("--max-add-ons-per-position", type=int, default=1)
    parser.add_argument(
        "--market-panic-guard",
        action="store_true",
        help="Defer hard-stop exits on market-wide panic days instead of fire-selling.",
    )
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.label}_{args.universe}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    combo_set = next(item for item in COMBO_SETS if item["name"] == args.combo_set)
    symbols = combo_harness.resolve_symbols(args.universe)
    universe_data, vnindex = combo_harness.load_local_history(symbols, args.start, args.end)
    print(f"[add-on] universe={args.universe} symbols={len(symbols)} combo={combo_set['name']}")
    print("[add-on] building features once...")
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    add_on_regime = None
    add_on_regime_column = "add_on_regime_v2"
    add_on_budget_multiplier_column = None
    if args.add_on_regime in {"phase_v1", "phase_v2", "phase_throttle_v1", "phase_risk_throttle_v1"}:
        phase_config = MarketPhaseConfig()
        if args.add_on_regime in {"phase_v2", "phase_throttle_v1", "phase_risk_throttle_v1"}:
            phase_config = MarketPhaseConfig(
                min_add_on_score=55.0,
                allow_late_bull_thrust=True,
                max_late_extension_20d=0.14,
                max_late_extension_ma20=0.07,
                min_mid_breadth_ma50=0.58,
                min_early_breadth_thrust_20d=0.14,
                max_distribution_pressure=1.65,
            )
        add_on_regime = build_market_phase_frame(features, config=phase_config)
        add_on_regime.to_csv(out_dir / f"add_on_{args.add_on_regime}.csv", index=False)
        if args.add_on_regime in {"phase_throttle_v1", "phase_risk_throttle_v1"}:
            add_on_budget_multiplier_column = (
                "add_on_risk_throttle_multiplier"
                if args.add_on_regime == "phase_risk_throttle_v1"
                else "add_on_budget_multiplier"
            )
            add_on_regime_column = add_on_budget_multiplier_column
        else:
            add_on_regime_column = "allow_add_on_phase_v1"
    elif args.add_on_regime in {"macro_v2", "macro_v3_goldilocks"}:
        regime_config = MacroRegimeConfig()
        if args.add_on_regime == "macro_v3_goldilocks":
            regime_config = MacroRegimeConfig(
                min_market_score=70.0,
                min_chdm50=55.0,
                max_ds20=0.35,
                min_breadth_ma50=0.63,
                min_vni_ret_60d=0.0,
                max_cpi_yoy=4.0,
                max_cpi_yoy_3m_delta=0.45,
                max_fx_60d_change=0.02,
                max_interbank_overnight=7.0,
            )
        add_on_regime = build_macro_regime_frame(
            features,
            cpi=load_macro_csv(args.cpi_csv or None),
            fx=load_macro_csv(args.fx_csv or None),
            interest=load_macro_csv(args.interest_csv or None),
            config=regime_config,
        )
        add_on_regime.to_csv(out_dir / f"add_on_{args.add_on_regime}.csv", index=False)

    hypotheses = {hyp.name: hyp for hyp in load_hypotheses(DEFAULT_CONFIG)}
    signal_cache = combo_harness.build_signal_cache_for_combo(
        features,
        [hypotheses[name] for name in combo_set["strategies"]],
        combo_set["name"],
        sorted({symbol.upper().strip() for symbol in symbols}),
    )
    combo_harness.install_edge_cache(signal_cache)

    rows: list[dict[str, Any]] = []
    max_positions_values = [int(item.strip()) for item in args.max_positions.split(",") if item.strip()]
    variants = [
        ("BASELINE", False, False),
        ("EXIT_AGENT", True, False),
        ("ADD_ON", False, True),
        ("EXIT_PLUS_ADD_ON", True, True),
    ]
    for max_positions in max_positions_values:
        for variant, exit_enabled, add_on_enabled in variants:
            cfg = LivePipelineBacktestConfig(
                start_date=args.start,
                end_date=args.end,
                max_positions=max_positions,
                max_candidates_per_day=args.max_candidates_per_day,
                edge_strategy_name=combo_set["name"],
            )
            print(f"[add-on] {combo_set['name']} {variant} p{max_positions}")
            result = run_with_add_on_agent(
                universe_data,
                vnindex,
                features,
                cfg,
                exit_agent_enabled=exit_enabled,
                add_on_agent_enabled=add_on_enabled,
                add_on_config=PositionAddOnAgentConfig(
                    add_on_budget_fraction_of_slot=args.add_on_budget_fraction,
                    max_add_ons_per_position=args.max_add_ons_per_position,
                ),
                add_on_regime=add_on_regime,
                add_on_regime_column=add_on_regime_column,
                add_on_budget_multiplier_column=add_on_budget_multiplier_column,
                market_panic_guard_enabled=args.market_panic_guard,
            )
            metrics = result["metrics"]
            row = {
                "combo_set": combo_set["name"],
                "variant": variant,
                "max_positions": max_positions,
                "max_candidates_per_day": args.max_candidates_per_day,
                "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
                "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
                "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
                "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
                "number_of_trades": int(metrics.get("number_of_trades", 0)),
                "fills": int(metrics.get("fills", 0)),
                "add_on_reviews": int(metrics.get("add_on_reviews", 0)),
                "add_on_signals": int(metrics.get("add_on_signals", 0)),
                "add_on_fills": int(metrics.get("add_on_fills", 0)),
                "agent_reviews": int(metrics.get("agent_reviews", 0)),
                "agent_raise_stop": int(metrics.get("agent_raise_stop", 0)),
                "add_on_regime": args.add_on_regime,
                "add_on_budget_fraction": args.add_on_budget_fraction,
                "max_add_ons_per_position": args.max_add_ons_per_position,
                "market_panic_guard": bool(args.market_panic_guard),
                "panic_guard_days": int(metrics.get("panic_guard_days", 0)),
                "panic_guard_deferred_exits": int(metrics.get("panic_guard_deferred_exits", 0)),
            }
            rows.append(row)
            guard_suffix = "_panic_guard" if args.market_panic_guard else ""
            stem = f"{combo_set['name']}_p{max_positions}_{variant.lower()}{guard_suffix}"
            pd.DataFrame(result["trades"]).to_csv(out_dir / f"{stem}_trades.csv", index=False)
            if not result["equity_frame"].empty:
                result["equity_frame"].to_csv(out_dir / f"{stem}_equity.csv")
            pd.DataFrame(rows).sort_values("total_return_pct", ascending=False).to_csv(out_dir / "summary.csv", index=False)
            print(
                f"  -> ret={row['total_return_pct']:+.2f}% sharpe={row['sharpe_ratio']:.2f} "
                f"mdd={row['max_drawdown_pct']:.2f}% wr={row['win_rate_pct']:.2f}% "
                f"trades={row['number_of_trades']} add_fills={row['add_on_fills']} "
                f"panic_defer={row['panic_guard_deferred_exits']}"
            )

    summary = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"[add-on] saved {out_dir / 'summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
