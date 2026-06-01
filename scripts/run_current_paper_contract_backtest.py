"""Backtest active MVP sleeves with the current paper-ledger contract.

This runner is deliberately separate from ``live_pipeline`` because the active
paper ledger uses a different execution contract:

- signal close on T, fill next available open on T+1
- VND price multiplier for local OHLCV data
- no commission/slippage in the ledger
- exits reviewed causally with the PositionStateExitAgent
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from multiagents_trading_assistant.agentic.position_exit_agent import (
    PositionExitAgentConfig,
    PositionState,
    PositionStateExitAgent,
)
from multiagents_trading_assistant.backtest import live_pipeline
from multiagents_trading_assistant.backtest.live_pipeline import (
    LivePipelineBacktestConfig,
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
from scripts.backtest_combos_unbiased import (
    build_signal_cache_for_combo,
    load_local_history,
    resolve_symbols,
)
from scripts.build_combined_paper_pnl import _exit_price_for_bar, _holding_bars


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "reports" / "combined_paper_trading_demo" / "current_paper_contract_backtest"
CONTRACT_ID = "signal_close_T_fill_next_open_T_plus_1_no_fee_agent_exit_v1"


@dataclass
class PaperPosition:
    sleeve_id: str
    symbol: str
    strategy_name: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_idx: int
    entry_price: float
    shares: int
    cost_value: float
    stop_loss: float
    take_profit: float
    max_holding_bars: int
    highest_price: float
    edge_risk: dict[str, Any]
    last_agent_action: str = ""
    last_agent_reason: str = ""
    pending_exit_action: str = ""
    pending_stop_loss: float | None = None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _lot_shares(target_value: float, price: float, lot_size: int) -> int:
    if price <= 0 or target_value <= 0:
        return 0
    raw = int(target_value // price)
    return (raw // lot_size) * lot_size if lot_size > 1 else raw


def _install_edge_cache(signal_cache: dict[pd.Timestamp, dict[str, dict]]) -> None:
    def cached_edge_signals(symbols, as_of_date=None, strategy_name=None, config_path=None, root=None):
        del strategy_name, config_path, root
        date = pd.Timestamp(as_of_date).normalize() if as_of_date else max(signal_cache)
        day = signal_cache.get(date, {})
        return {
            str(symbol).upper().strip(): day.get(
                str(symbol).upper().strip(),
                {"strategy_name": "", "passed": False, "available": False, "error": "No cached"},
            )
            for symbol in symbols
        }

    live_pipeline.get_edge_strategy_signals = cached_edge_signals


def _feature_history(features: pd.DataFrame, price_unit_multiplier: float) -> dict[str, pd.DataFrame]:
    history = features.copy()
    history["symbol"] = history["symbol"].astype(str).str.upper()
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    for column in ["open", "high", "low", "close", "ma20", "ma50", "ma200", "atr14"]:
        if column in history:
            history[column] = pd.to_numeric(history[column], errors="coerce") * price_unit_multiplier
    return {
        symbol: frame.sort_values("date").reset_index(drop=True)
        for symbol, frame in history.groupby("symbol", sort=False)
    }


def _enter_positions(
    *,
    sleeve_id: str,
    orders: list[dict[str, Any]],
    data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    day_idx: int,
    cash: float,
    positions: dict[str, PaperPosition],
    initial_capital: float,
    portfolio_value: float,
    max_positions: int,
    lot_size: int,
    price_unit_multiplier: float,
    sizing_mode: str,
    counters: dict[str, int],
) -> float:
    slots = max(0, max_positions - len(positions))
    if slots <= 0:
        return cash
    for order in [o for o in orders if o["symbol"] not in positions][:slots]:
        counters["attempted_orders"] += 1
        row = _row_at(data[order["symbol"]], date)
        if row is None:
            continue
        entry_price = float(row["open"]) * price_unit_multiplier
        sizing_capital = portfolio_value if sizing_mode == "compound_equity" else initial_capital
        target_value = min(sizing_capital / max_positions, cash)
        shares = _lot_shares(target_value, entry_price, lot_size)
        if shares <= 0:
            continue
        cost_value = shares * entry_price
        if cost_value > cash:
            continue
        risk = dict(order.get("edge_risk") or {})
        stop_pct = _to_float(risk.get("stop_loss"), 0.08)
        take_profit_pct = _to_float(risk.get("take_profit"), 0.25)
        max_holding = int(_to_float(risk.get("max_holding_bars"), 30))
        cash -= cost_value
        counters["fills"] += 1
        positions[order["symbol"]] = PaperPosition(
            sleeve_id=sleeve_id,
            symbol=order["symbol"],
            strategy_name=str(order.get("edge_strategy_name") or order.get("strategy_name") or "CORE_MVP9"),
            signal_date=pd.Timestamp(order["signal_date"]).normalize(),
            entry_date=date,
            entry_idx=day_idx,
            entry_price=entry_price,
            shares=shares,
            cost_value=cost_value,
            stop_loss=entry_price * (1.0 - stop_pct),
            take_profit=entry_price * (1.0 + take_profit_pct),
            max_holding_bars=max_holding,
            highest_price=entry_price,
            edge_risk=risk,
        )
    return cash


def _close_position(position: PaperPosition, date: pd.Timestamp, exit_price: float, reason: str, cash: float) -> tuple[float, dict[str, Any]]:
    exit_value = position.shares * exit_price
    pnl = exit_value - position.cost_value
    trade = {
        "sleeve_id": position.sleeve_id,
        "symbol": position.symbol,
        "strategy_name": position.strategy_name,
        "signal_date": position.signal_date.date().isoformat(),
        "entry_date": position.entry_date.date().isoformat(),
        "exit_date": date.date().isoformat(),
        "entry_price": round(position.entry_price, 2),
        "exit_price": round(exit_price, 2),
        "shares": position.shares,
        "cost_value": round(position.cost_value, 2),
        "exit_value": round(exit_value, 2),
        "net_pnl": round(pnl, 2),
        "pnl_pct": round((exit_price / position.entry_price - 1.0) * 100.0, 2),
        "holding_bars": int((date - position.entry_date).days),
        "exit_reason": reason,
        "agent_action": position.last_agent_action,
        "agent_reason": position.last_agent_reason,
        "fill_model": CONTRACT_ID,
    }
    return cash + exit_value, trade


def _mark_to_market(positions: dict[str, PaperPosition], data: dict[str, pd.DataFrame], date: pd.Timestamp, price_unit_multiplier: float) -> float:
    total = 0.0
    for symbol, position in positions.items():
        row = _row_at(data[symbol], date)
        if row is None:
            frame = data[symbol]
            prior = frame[frame["date"] <= date].tail(1)
            if prior.empty:
                total += position.cost_value
                continue
            row = prior.iloc[0]
        total += position.shares * float(row["close"]) * price_unit_multiplier
    return total


def _review_position(
    *,
    position: PaperPosition,
    data: dict[str, pd.DataFrame],
    feature_by_symbol: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    price_unit_multiplier: float,
    use_exit_agent: bool,
    agent: PositionStateExitAgent,
) -> tuple[str | None, float | None]:
    row = _row_at(data[position.symbol], date)
    if row is None:
        return None, None
    symbol_prices = data[position.symbol].reset_index(drop=True)
    holding_bars = _holding_bars(symbol_prices, position.entry_date, date)
    high_v = float(row["high"]) * price_unit_multiplier
    close_v = float(row["close"]) * price_unit_multiplier
    position.highest_price = max(position.highest_price, high_v)

    if position.pending_exit_action == "TAKE_PROFIT_NEXT_OPEN" and holding_bars >= 2:
        return "AGENT_TAKE_PROFIT_NEXT_OPEN", float(row["open"]) * price_unit_multiplier

    if position.pending_stop_loss and position.pending_stop_loss > position.stop_loss:
        position.stop_loss = float(position.pending_stop_loss)
    position.pending_exit_action = ""
    position.pending_stop_loss = None

    reason, exit_price = _exit_price_for_bar(row, position.stop_loss, position.take_profit, price_unit_multiplier)
    if reason is None:
        activation = _to_float(position.edge_risk.get("trailing_profit_activation"), 0.09)
        atr_mult = _to_float(position.edge_risk.get("trailing_atr_mult"), 2.6)
        if close_v >= position.entry_price * (1.0 + activation):
            atr = _to_float(row.get("atr") if "atr" in row else row.get("atr14"), 0.0) * price_unit_multiplier
            trailing_stop = close_v - atr_mult * atr if atr > 0 else close_v * (1.0 - 0.08)
            if trailing_stop > position.stop_loss:
                position.stop_loss = trailing_stop
        if holding_bars >= position.max_holding_bars:
            reason, exit_price = "MAX_HOLDING_BARS", close_v

    if reason is not None and exit_price is not None:
        return reason, exit_price

    if use_exit_agent and position.symbol in feature_by_symbol:
        hist = feature_by_symbol[position.symbol]
        hist = hist[hist["date"] <= date]
        review = agent.review(
            PositionState(
                symbol=position.symbol,
                entry_date=position.entry_date.date().isoformat(),
                entry_price=position.entry_price,
                stop_loss_price=position.stop_loss,
                take_profit_price=position.take_profit,
                shares=position.shares,
                highest_price=position.highest_price,
                strategy_name=position.strategy_name,
                holding_bars=holding_bars,
            ),
            hist,
            as_of_date=date,
        )
        position.last_agent_action = review.action
        position.last_agent_reason = "|".join(review.reason_codes)
        if review.action == "TAKE_PROFIT_NEXT_OPEN":
            position.pending_exit_action = review.action
        if review.suggested_stop_loss and review.suggested_stop_loss > position.stop_loss:
            position.pending_stop_loss = float(review.suggested_stop_loss)
    return None, None


def _filter_candidates_by_signal(
    candidates: list[dict[str, Any]],
    signal_cache: dict[pd.Timestamp, dict[str, dict]],
    date: pd.Timestamp,
    *,
    max_edge_rank: int | None = None,
    min_edge_rank_score: float | None = None,
    min_mkt_regime_score: float | None = None,
    allowed_mkt_regime_states: set[str] | None = None,
    min_chdm50: float | None = None,
    max_ds20: float | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return candidates
    day = signal_cache.get(pd.Timestamp(date).normalize(), {})
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        symbol = str(candidate.get("symbol") or "").upper()
        signal = day.get(symbol, {})
        if max_edge_rank is not None:
            rank = _to_float(signal.get("edge_rank"), 999999.0)
            if rank > max_edge_rank:
                continue
        if min_edge_rank_score is not None:
            score = _to_float(signal.get("edge_rank_score"), 0.0)
            if score < min_edge_rank_score:
                continue
        if min_mkt_regime_score is not None:
            regime_score = _to_float(signal.get("mkt_regime_score"), 0.0)
            if regime_score < min_mkt_regime_score:
                continue
        if allowed_mkt_regime_states:
            state = str(signal.get("mkt_regime_state") or "").upper()
            if state not in allowed_mkt_regime_states:
                continue
        if min_chdm50 is not None:
            chdm50 = _to_float(signal.get("CHDM50"), 0.0)
            if chdm50 < min_chdm50:
                continue
        if max_ds20 is not None:
            ds20 = _to_float(signal.get("DS20"), 1.0)
            if ds20 > max_ds20:
                continue
        out.append(candidate)
    return out


def run_sleeve(
    *,
    sleeve_id: str,
    max_positions: int,
    universe_data: dict[str, pd.DataFrame],
    vnindex: pd.DataFrame,
    features: pd.DataFrame,
    signal_cache: dict[pd.Timestamp, dict[str, dict]],
    start: str,
    end: str,
    initial_capital: float,
    max_candidates_per_day: int,
    lot_size: int,
    price_unit_multiplier: float,
    use_exit_agent: bool,
    sizing_mode: str = "fixed_initial_capital",
    max_edge_rank: int | None = None,
    min_edge_rank_score: float | None = None,
    min_mkt_regime_score: float | None = None,
    allowed_mkt_regime_states: set[str] | None = None,
    min_chdm50: float | None = None,
    max_ds20: float | None = None,
) -> dict[str, Any]:
    _install_edge_cache(signal_cache)
    data = _prepare_data(universe_data)
    index_data = _prepare_single_frame(vnindex)
    feature_by_symbol = _feature_history(features, price_unit_multiplier)
    calendar = sorted(set(index_data["date"]))
    date_to_pos = {date: idx for idx, date in enumerate(calendar)}
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    cfg = LivePipelineBacktestConfig(
        initial_capital=initial_capital,
        start_date=start,
        end_date=end,
        max_positions=max_positions,
        max_candidates_per_day=max_candidates_per_day,
        commission_rate=0.0,
        sell_tax_rate=0.0,
        slippage_rate=0.0,
        edge_strategy_name="CORE_MVP9",
    )

    cash = float(initial_capital)
    positions: dict[str, PaperPosition] = {}
    pending_orders: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    counters = {"signal_days": 0, "signals": 0, "attempted_orders": 0, "fills": 0}
    agent = PositionStateExitAgent(PositionExitAgentConfig())

    for date in calendar[cfg.lookback :]:
        if date > end_ts:
            break
        day_idx = date_to_pos[date]
        orders = pending_orders.pop(date, [])
        if orders:
            portfolio_value = cash + _mark_to_market(positions, data, date, price_unit_multiplier)
            cash = _enter_positions(
                sleeve_id=sleeve_id,
                orders=orders,
                data=data,
                date=date,
                day_idx=day_idx,
                cash=cash,
                positions=positions,
                initial_capital=initial_capital,
                portfolio_value=portfolio_value,
                max_positions=max_positions,
                lot_size=lot_size,
                price_unit_multiplier=price_unit_multiplier,
                sizing_mode=sizing_mode,
                counters=counters,
            )

        for symbol in list(positions):
            position = positions[symbol]
            holding_bars = date_to_pos[date] - position.entry_idx
            reason, exit_price = _review_position(
                position=position,
                data=data,
                feature_by_symbol=feature_by_symbol,
                date=date,
                price_unit_multiplier=price_unit_multiplier,
                use_exit_agent=use_exit_agent,
                agent=agent,
            )
            if reason is not None and exit_price is not None and holding_bars >= 2:
                cash, trade = _close_position(position, date, exit_price, reason, cash)
                trade["holding_bars"] = holding_bars
                trades.append(trade)
                del positions[symbol]

        if date >= start_ts:
            equity_rows.append(
                {
                    "date": date,
                    "equity": cash + _mark_to_market(positions, data, date, price_unit_multiplier),
                    "cash": cash,
                    "positions": len(positions),
                }
            )

        next_date = _next_calendar_date(calendar, day_idx)
        if next_date is None or date < start_ts:
            continue
        market_ctx = _market_context_at(index_data, date, cfg.lookback)
        if not market_ctx.should_trade and not cfg.allow_downtrend_entries:
            continue
        candidates = [c for c in _scan_live_candidates(data, date, market_ctx, cfg) if c["symbol"] not in positions]
        candidates = _filter_candidates_by_signal(
            candidates,
            signal_cache,
            date,
            max_edge_rank=max_edge_rank,
            min_edge_rank_score=min_edge_rank_score,
            min_mkt_regime_score=min_mkt_regime_score,
            allowed_mkt_regime_states=allowed_mkt_regime_states,
            min_chdm50=min_chdm50,
            max_ds20=max_ds20,
        )
        if candidates:
            counters["signal_days"] += 1
            counters["signals"] += len(candidates[:max_candidates_per_day])
            pending_orders.setdefault(next_date, []).extend(candidates[:max_candidates_per_day])

    equity_frame = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame()
    equity_curve = equity_frame["equity"].rename("equity") if not equity_frame.empty else pd.Series(dtype=float)
    metrics = compute_metrics(equity_curve, trades) if not equity_curve.empty else {}
    metrics.update(counters)
    open_rows = [
        {
            "sleeve_id": position.sleeve_id,
            "symbol": position.symbol,
            "strategy_name": position.strategy_name,
            "signal_date": position.signal_date.date().isoformat(),
            "entry_date": position.entry_date.date().isoformat(),
            "entry_price": round(position.entry_price, 2),
            "shares": position.shares,
            "cost_value": round(position.cost_value, 2),
            "stop_loss": round(position.stop_loss, 2),
            "take_profit": round(position.take_profit, 2),
            "highest_price": round(position.highest_price, 2),
            "agent_action": position.last_agent_action,
            "agent_reason": position.last_agent_reason,
        }
        for position in positions.values()
    ]
    return {
        "metrics": metrics,
        "trades": trades,
        "open_positions": open_rows,
        "equity_frame": equity_frame,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact current-paper contract backtest for active MVP sleeves.")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-29")
    parser.add_argument("--capital", type=float, default=100_000_000.0)
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    parser.add_argument("--mvp-p5-max-positions", type=int, default=5)
    parser.add_argument("--mvp-p4-max-positions", type=int, default=4)
    parser.add_argument("--max-edge-rank", type=int, default=0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--price-unit-multiplier", type=float, default=1000.0)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--disable-exit-agent", action="store_true")
    parser.add_argument(
        "--sizing-mode",
        choices=["fixed_initial_capital", "compound_equity"],
        default="fixed_initial_capital",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = resolve_symbols(args.universe)
    started = time.time()
    print(f"[paper-contract] universe={args.universe} symbols={len(symbols)}")
    universe_data, vnindex = load_local_history(symbols, args.start, args.end)
    print("[paper-contract] building features...")
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    hypotheses_by_name = {h.name: h for h in load_hypotheses(DEFAULT_CONFIG)}
    hypotheses = [hypotheses_by_name[name] for name in MVP_EDGE_STRATEGIES if name in hypotheses_by_name]
    signal_cache = build_signal_cache_for_combo(features, hypotheses, "CORE_MVP9", sorted({s.upper() for s in symbols}))

    summary_rows: list[dict[str, Any]] = []
    max_edge_rank = args.max_edge_rank or None
    for sleeve_id, max_positions in [("mvp_p5", args.mvp_p5_max_positions), ("mvp_p4", args.mvp_p4_max_positions)]:
        print(f"[paper-contract] running {sleeve_id} max_positions={max_positions}")
        result = run_sleeve(
            sleeve_id=sleeve_id,
            max_positions=max_positions,
            universe_data=universe_data,
            vnindex=vnindex,
            features=features,
            signal_cache=signal_cache,
            start=args.start,
            end=args.end,
            initial_capital=args.capital,
            max_candidates_per_day=args.max_candidates_per_day,
            lot_size=args.lot_size,
            price_unit_multiplier=args.price_unit_multiplier,
            use_exit_agent=not args.disable_exit_agent,
            sizing_mode=args.sizing_mode,
            max_edge_rank=max_edge_rank,
        )
        metrics = result["metrics"]
        equity = result["equity_frame"]
        trades = pd.DataFrame(result["trades"])
        open_positions = pd.DataFrame(result["open_positions"])
        if not equity.empty:
            equity.to_csv(out_dir / f"{sleeve_id}_equity.csv", encoding="utf-8-sig")
        trades.to_csv(out_dir / f"{sleeve_id}_trades.csv", index=False, encoding="utf-8-sig")
        open_positions.to_csv(out_dir / f"{sleeve_id}_open_positions.csv", index=False, encoding="utf-8-sig")
        summary_rows.append(
            {
                "sleeve_id": sleeve_id,
                "status": "Verified",
                "source_note": "Exact current paper contract backtest",
                "contract_id": CONTRACT_ID,
                "universe": args.universe,
                "start": args.start,
                "end": args.end,
                "max_positions": max_positions,
                "max_edge_rank": max_edge_rank,
                "max_candidates_per_day": args.max_candidates_per_day,
                "initial_capital": args.capital,
                "ending_equity": round(float(equity["equity"].iloc[-1]), 2) if not equity.empty else None,
                "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100.0, 2),
                "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
                "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100.0, 2),
                "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100.0, 2),
                "number_of_trades": int(metrics.get("number_of_trades", 0)),
                "signal_days": int(metrics.get("signal_days", 0)),
                "signals": int(metrics.get("signals", 0)),
                "attempted_orders": int(metrics.get("attempted_orders", 0)),
                "fills": int(metrics.get("fills", 0)),
                "open_positions": int(len(open_positions)),
                "sizing_mode": args.sizing_mode,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    meta = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "elapsed_sec": round(time.time() - started, 1),
        "contract_id": CONTRACT_ID,
        "sizing_mode": args.sizing_mode,
        "max_edge_rank": max_edge_rank,
        "output_dir": str(out_dir),
        "read_only_report": True,
    }
    (out_dir / "contract_audit_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_dir / "summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
