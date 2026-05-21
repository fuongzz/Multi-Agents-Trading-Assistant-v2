"""Build persistent paper ledger and PnL tables for the combined dashboard.

Execution contract:
- signal after close T;
- paper buy at next available open T+1;
- exits are reviewed causally from OHLCV bars already in local parquet;
- no broker orders are sent.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from multiagents_trading_assistant.agentic.position_exit_agent import (
    PositionExitAgentConfig,
    PositionState,
    PositionStateExitAgent,
)
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG


ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
CONFIG_DIR = ROOT / "multiagents_trading_assistant" / "edge_lab" / "configs"
CONFIGS = [
    Path(DEFAULT_CONFIG),
    CONFIG_DIR / "global_market_hypotheses.json",
    CONFIG_DIR / "sector_rotation_hypotheses.json",
    CONFIG_DIR / "oil_gas_rotation_hypotheses.json",
    CONFIG_DIR / "theme_flow_hypotheses.json",
]

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

OPEN_COLUMNS = [
    "sleeve_id",
    "sleeve_label",
    "symbol",
    "strategy_name",
    "signal_date",
    "entry_date",
    "entry_price",
    "shares",
    "cost_value",
    "stop_loss",
    "take_profit",
    "max_holding_bars",
    "highest_price",
    "last_review_date",
    "pending_exit_action",
    "pending_stop_loss",
    "agent_last_action",
    "agent_last_reason",
    "status",
    "fill_model",
]

CLOSED_COLUMNS = [
    "sleeve_id",
    "sleeve_label",
    "symbol",
    "strategy_name",
    "signal_date",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "shares",
    "cost_value",
    "exit_value",
    "net_pnl",
    "pnl_pct",
    "holding_bars",
    "exit_reason",
    "agent_reason",
    "fill_model",
]

EVENT_COLUMNS = [
    "date",
    "sleeve_id",
    "symbol",
    "event",
    "price",
    "shares",
    "reason",
    "details",
]

LEDGER_META = "paper_ledger_meta.json"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def _price_frame() -> pd.DataFrame:
    frame = pd.read_parquet(PRICE_PATH)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def _symbol_rows(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    return prices[prices["symbol"] == symbol].sort_values("date").reset_index(drop=True)


def _next_row(prices: pd.DataFrame, symbol: str, signal_date: pd.Timestamp) -> pd.Series | None:
    rows = prices[(prices["symbol"] == symbol) & (prices["date"] > signal_date)].sort_values("date")
    return None if rows.empty else rows.iloc[0]


def _latest_row(prices: pd.DataFrame, symbol: str, as_of: pd.Timestamp) -> pd.Series | None:
    rows = prices[(prices["symbol"] == symbol) & (prices["date"] <= as_of)].sort_values("date")
    return None if rows.empty else rows.iloc[-1]


def _lot_shares(target_value: float, price_vnd: float, lot_size: int) -> int:
    if price_vnd <= 0:
        return 0
    return (int(target_value / price_vnd) // lot_size) * lot_size


def _risk_map() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for config in CONFIGS:
        if not config.exists():
            continue
        try:
            for hypothesis in load_hypotheses(config):
                out[hypothesis.name] = dict(hypothesis.risk or {})
        except Exception:
            continue
    return out


def _risk_for(strategy_name: str, risk_by_strategy: dict[str, dict[str, Any]], sleeve_id: str) -> dict[str, Any]:
    if sleeve_id == "flow_v2":
        return {"stop_loss": 0.08, "take_profit": 0.25, "max_holding_bars": 20, "trailing_atr_mult": 2.6, "trailing_profit_activation": 0.09}
    risk = dict(risk_by_strategy.get(strategy_name) or {})
    risk.setdefault("stop_loss", 0.08)
    risk.setdefault("take_profit", 0.25)
    risk.setdefault("max_holding_bars", 30)
    risk.setdefault("trailing_atr_mult", 2.6)
    risk.setdefault("trailing_profit_activation", 0.09)
    return risk


def _load_ledger(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    open_ = _read_csv(out_dir / "paper_ledger_open.csv")
    closed = _read_csv(out_dir / "paper_ledger_closed.csv")
    events = _read_csv(out_dir / "paper_ledger_events.csv")
    if open_.empty:
        open_ = _empty(OPEN_COLUMNS)
    if closed.empty:
        closed = _empty(CLOSED_COLUMNS)
    if events.empty:
        events = _empty(EVENT_COLUMNS)
    for column in OPEN_COLUMNS:
        if column not in open_:
            open_[column] = None
    for column in CLOSED_COLUMNS:
        if column not in closed:
            closed[column] = None
    return open_[OPEN_COLUMNS].copy(), closed[CLOSED_COLUMNS].copy(), events


def _existing_signal_keys(open_: pd.DataFrame, closed: pd.DataFrame) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for frame in [open_, closed]:
        if frame.empty:
            continue
        for row in frame.to_dict("records"):
            keys.add((str(row.get("sleeve_id")), str(row.get("symbol")).upper(), str(row.get("signal_date"))[:10]))
    return keys


def _cash_for_sleeve(sleeve_id: str, open_: pd.DataFrame, closed: pd.DataFrame, capital: float) -> float:
    open_cost = 0.0 if open_.empty else pd.to_numeric(open_.loc[open_["sleeve_id"] == sleeve_id, "cost_value"], errors="coerce").fillna(0).sum()
    realized = 0.0 if closed.empty else pd.to_numeric(closed.loc[closed["sleeve_id"] == sleeve_id, "net_pnl"], errors="coerce").fillna(0).sum()
    return float(capital + realized - open_cost)


def _mvp_candidates(source_dir: Path, sleeve_id: str, label: str, as_of: pd.Timestamp) -> list[dict[str, Any]]:
    actions = _read_csv(source_dir / "rolling_candidate_actions_14d.csv")
    if actions.empty or "action" not in actions:
        return []
    actions["date"] = pd.to_datetime(actions["date"]).dt.normalize()
    buys = actions[
        (actions["date"] < as_of)
        & actions["action"].astype(str).str.startswith("PAPER_BUY_CANDIDATE")
    ].sort_values(["date", "priority"])
    rows = []
    for item in buys.to_dict("records"):
        rows.append(
            {
                "sleeve_id": sleeve_id,
                "sleeve_label": label,
                "symbol": str(item.get("symbol") or "").upper(),
                "strategy_name": item.get("strategy_name"),
                "signal_date": pd.Timestamp(item["date"]).normalize(),
                "max_positions": _as_int(item.get("max_positions_context"), 5),
                "priority": _as_int(item.get("priority"), 999),
            }
        )
    return rows


def _flow_candidates(source_dir: Path, as_of: pd.Timestamp) -> list[dict[str, Any]]:
    targets = _read_csv(source_dir / "flow_v2_target_plan.csv")
    status = _read_json(source_dir / "flow_v2_status.json")
    if targets.empty:
        return []
    signal_date = pd.Timestamp(status.get("as_of_date") or as_of).normalize()
    if signal_date >= as_of:
        return []
    rows = []
    for item in targets.to_dict("records"):
        rows.append(
            {
                "sleeve_id": "flow_v2",
                "sleeve_label": "Flow V2 Rotation",
                "symbol": str(item.get("symbol") or "").upper(),
                "strategy_name": "flow_v2_rotation_clean_sector_heavy",
                "signal_date": signal_date,
                "max_positions": _as_int(status.get("positions"), 2),
                "priority": _as_int(item.get("rank"), 999),
                "target_value": _as_float(item.get("target_value"), 0.0),
            }
        )
    return rows


def _enter_positions(
    *,
    out_dir: Path,
    open_: pd.DataFrame,
    closed: pd.DataFrame,
    events: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    capital: float,
    lot_size: int,
    price_unit_multiplier: float,
    risk_by_strategy: dict[str, dict[str, Any]],
    start_date: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = [
        *_mvp_candidates(out_dir / "mvp_p5", "mvp_p5", "MVP Original p5/c10", as_of),
        *_mvp_candidates(out_dir / "mvp_p4", "mvp_p4", "MVP Improved p4/c10", as_of),
        *_flow_candidates(out_dir / "flow_v2", as_of),
    ]
    if not candidates:
        return open_, events

    keys = _existing_signal_keys(open_, closed)
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda x: (x["signal_date"], x["sleeve_id"], x["priority"])):
        symbol = item["symbol"]
        sleeve_id = item["sleeve_id"]
        signal_date = item["signal_date"]
        if start_date is not None and signal_date < start_date:
            continue
        if not symbol or (sleeve_id, symbol, signal_date.date().isoformat()) in keys:
            continue
        if not open_.empty and ((open_["sleeve_id"] == sleeve_id) & (open_["symbol"].astype(str).str.upper() == symbol)).any():
            continue
        max_positions = max(1, int(item["max_positions"]))
        current_positions = 0 if open_.empty else int((open_["sleeve_id"] == sleeve_id).sum())
        if current_positions >= max_positions:
            continue
        entry = _next_row(prices, symbol, signal_date)
        if entry is None or pd.Timestamp(entry["date"]).normalize() > as_of:
            continue
        entry_price = float(entry["open"]) * price_unit_multiplier
        target_value = _as_float(item.get("target_value"), 0.0) or capital / max_positions
        cash = _cash_for_sleeve(sleeve_id, open_, closed, capital)
        target_value = min(target_value, cash)
        shares = _lot_shares(target_value, entry_price, lot_size)
        if shares <= 0:
            continue
        risk = _risk_for(str(item["strategy_name"]), risk_by_strategy, sleeve_id)
        cost_value = shares * entry_price
        row = {
            "sleeve_id": sleeve_id,
            "sleeve_label": item["sleeve_label"],
            "symbol": symbol,
            "strategy_name": item["strategy_name"],
            "signal_date": signal_date.date().isoformat(),
            "entry_date": pd.Timestamp(entry["date"]).date().isoformat(),
            "entry_price": round(entry_price, 2),
            "shares": shares,
            "cost_value": round(cost_value, 2),
            "stop_loss": round(entry_price * (1.0 - float(risk["stop_loss"])), 2),
            "take_profit": round(entry_price * (1.0 + float(risk["take_profit"])), 2),
            "max_holding_bars": int(risk["max_holding_bars"]),
            "highest_price": round(entry_price, 2),
            "last_review_date": pd.Timestamp(entry["date"]).date().isoformat(),
            "pending_exit_action": "",
            "pending_stop_loss": "",
            "agent_last_action": "",
            "agent_last_reason": "",
            "status": "OPEN_LEDGER",
            "fill_model": "signal_close_T_fill_next_open_T_plus_1",
        }
        rows.append(row)
        event_rows.append(
            {
                "date": row["entry_date"],
                "sleeve_id": sleeve_id,
                "symbol": symbol,
                "event": "ENTRY",
                "price": row["entry_price"],
                "shares": shares,
                "reason": "PAPER_BUY_CANDIDATE",
                "details": f"signal_date={row['signal_date']}; strategy={row['strategy_name']}",
            }
        )
        open_ = pd.concat([open_, pd.DataFrame([row])], ignore_index=True)
        keys.add((sleeve_id, symbol, signal_date.date().isoformat()))
    if event_rows:
        events = pd.concat([events, pd.DataFrame(event_rows)], ignore_index=True)
    return open_[OPEN_COLUMNS].copy(), events


def _build_feature_history(open_: pd.DataFrame, as_of: pd.Timestamp, price_unit_multiplier: float) -> dict[str, pd.DataFrame]:
    if open_.empty:
        return {}
    symbols = sorted(set(open_["symbol"].astype(str).str.upper()))
    start = (as_of - pd.Timedelta(days=520)).date().isoformat()
    try:
        features, _ = build_feature_table(symbols, start=start, end=as_of.date().isoformat(), root=ROOT)
    except Exception:
        return {}
    for column in ["open", "high", "low", "close", "ma20", "ma50", "ma200", "atr14"]:
        if column in features:
            features[column] = pd.to_numeric(features[column], errors="coerce") * price_unit_multiplier
    return {symbol: frame.sort_values("date").reset_index(drop=True) for symbol, frame in features.groupby("symbol", sort=False)}


def _exit_price_for_bar(row: pd.Series, stop_loss: float, take_profit: float, price_unit_multiplier: float) -> tuple[str | None, float | None]:
    open_v = float(row["open"]) * price_unit_multiplier
    high_v = float(row["high"]) * price_unit_multiplier
    low_v = float(row["low"]) * price_unit_multiplier
    if open_v <= stop_loss:
        return "STOP_GAP_OPEN", open_v
    if open_v >= take_profit:
        return "TAKE_PROFIT_GAP_OPEN", open_v
    hit_stop = low_v <= stop_loss
    hit_tp = high_v >= take_profit
    if hit_stop and hit_tp:
        return "STOP_LOSS_SAME_BAR_CONSERVATIVE", stop_loss
    if hit_stop:
        return "STOP_LOSS", stop_loss
    if hit_tp:
        return "TAKE_PROFIT", take_profit
    return None, None


def _holding_bars(symbol_prices: pd.DataFrame, entry_date: pd.Timestamp, date: pd.Timestamp) -> int:
    return int(((symbol_prices["date"] >= entry_date) & (symbol_prices["date"] <= date)).sum() - 1)


def _review_exits(
    *,
    open_: pd.DataFrame,
    closed: pd.DataFrame,
    events: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    price_unit_multiplier: float,
    risk_by_strategy: dict[str, dict[str, Any]],
    use_exit_agent: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if open_.empty:
        return open_, closed, events

    feature_history = _build_feature_history(open_, as_of, price_unit_multiplier) if use_exit_agent else {}
    agent = PositionStateExitAgent(PositionExitAgentConfig())
    keep_rows: list[dict[str, Any]] = []
    closed_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for position in open_.to_dict("records"):
        symbol = str(position["symbol"]).upper()
        symbol_prices = _symbol_rows(prices, symbol)
        if symbol_prices.empty:
            keep_rows.append(position)
            continue
        entry_date = pd.Timestamp(position["entry_date"]).normalize()
        last_review = pd.Timestamp(position.get("last_review_date") or position["entry_date"]).normalize()
        rows = symbol_prices[(symbol_prices["date"] > last_review) & (symbol_prices["date"] <= as_of)].sort_values("date")
        is_closed = False
        agent_reason = str(position.get("agent_last_reason") or "")
        for _, bar in rows.iterrows():
            date = pd.Timestamp(bar["date"]).normalize()
            holding_bars = _holding_bars(symbol_prices, entry_date, date)
            entry_price = _as_float(position["entry_price"])
            shares = _as_int(position["shares"])
            stop_loss = _as_float(position["stop_loss"])
            take_profit = _as_float(position["take_profit"])
            max_holding = _as_int(position["max_holding_bars"], 30)
            high_v = float(bar["high"]) * price_unit_multiplier
            close_v = float(bar["close"]) * price_unit_multiplier
            position["highest_price"] = round(max(_as_float(position.get("highest_price"), entry_price), high_v), 2)

            pending_action = str(position.get("pending_exit_action") or "")
            if pending_action == "TAKE_PROFIT_NEXT_OPEN" and holding_bars >= 2:
                reason, exit_price = "AGENT_TAKE_PROFIT_NEXT_OPEN", float(bar["open"]) * price_unit_multiplier
            else:
                pending_stop = _as_float(position.get("pending_stop_loss"), 0.0)
                if pending_stop > stop_loss:
                    position["stop_loss"] = round(pending_stop, 2)
                    stop_loss = pending_stop
                    event_rows.append(
                        {
                            "date": date.date().isoformat(),
                            "sleeve_id": position["sleeve_id"],
                            "symbol": symbol,
                            "event": "RAISE_STOP",
                            "price": round(stop_loss, 2),
                            "shares": shares,
                            "reason": "AGENT_RAISE_TRAILING_STOP",
                            "details": agent_reason,
                        }
                    )
                position["pending_exit_action"] = ""
                position["pending_stop_loss"] = ""
                reason, exit_price = _exit_price_for_bar(bar, stop_loss, take_profit, price_unit_multiplier)

            if reason is None:
                risk = _risk_for(str(position.get("strategy_name")), risk_by_strategy, str(position.get("sleeve_id")))
                activation = float(risk.get("trailing_profit_activation", 0.09))
                atr_mult = float(risk.get("trailing_atr_mult", 2.6))
                if close_v >= entry_price * (1.0 + activation):
                    atr = _as_float(bar.get("atr14"), 0.0) * price_unit_multiplier
                    trailing_stop = close_v - atr_mult * atr if atr > 0 else close_v * (1.0 - 0.08)
                    if trailing_stop > _as_float(position["stop_loss"]):
                        position["stop_loss"] = round(trailing_stop, 2)
                        event_rows.append(
                            {
                                "date": date.date().isoformat(),
                                "sleeve_id": position["sleeve_id"],
                                "symbol": symbol,
                                "event": "RAISE_STOP",
                                "price": round(trailing_stop, 2),
                                "shares": shares,
                                "reason": "RULE_TRAILING_STOP",
                                "details": f"close={round(close_v, 2)}; highest={position['highest_price']}",
                            }
                        )
                if holding_bars >= max_holding:
                    reason, exit_price = "MAX_HOLDING_BARS", close_v

            if reason is not None and exit_price is not None and holding_bars >= 2:
                exit_value = shares * exit_price
                cost_value = _as_float(position["cost_value"])
                pnl = exit_value - cost_value
                closed_row = {
                    "sleeve_id": position["sleeve_id"],
                    "sleeve_label": position["sleeve_label"],
                    "symbol": symbol,
                    "strategy_name": position["strategy_name"],
                    "signal_date": position["signal_date"],
                    "entry_date": position["entry_date"],
                    "exit_date": date.date().isoformat(),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "shares": shares,
                    "cost_value": round(cost_value, 2),
                    "exit_value": round(exit_value, 2),
                    "net_pnl": round(pnl, 2),
                    "pnl_pct": round((exit_price / entry_price - 1.0) * 100.0, 2),
                    "holding_bars": holding_bars,
                    "exit_reason": reason,
                    "agent_reason": agent_reason,
                    "fill_model": "causal_daily_exit_review",
                }
                closed_rows.append(closed_row)
                event_rows.append(
                    {
                        "date": closed_row["exit_date"],
                        "sleeve_id": position["sleeve_id"],
                        "symbol": symbol,
                        "event": "EXIT",
                        "price": closed_row["exit_price"],
                        "shares": shares,
                        "reason": reason,
                        "details": f"pnl={closed_row['net_pnl']}; pnl_pct={closed_row['pnl_pct']}",
                    }
                )
                is_closed = True
                break

            if use_exit_agent and symbol in feature_history:
                hist = feature_history[symbol]
                hist = hist[pd.to_datetime(hist["date"]).dt.normalize() <= date]
                review = agent.review(
                    PositionState(
                        symbol=symbol,
                        entry_date=str(position["entry_date"]),
                        entry_price=entry_price,
                        stop_loss_price=_as_float(position["stop_loss"]),
                        take_profit_price=take_profit,
                        shares=shares,
                        highest_price=_as_float(position.get("highest_price"), entry_price),
                        strategy_name=str(position.get("strategy_name") or ""),
                        holding_bars=holding_bars,
                    ),
                    hist,
                    as_of_date=date,
                )
                position["agent_last_action"] = review.action
                position["agent_last_reason"] = "|".join(review.reason_codes)
                if review.action == "TAKE_PROFIT_NEXT_OPEN":
                    position["pending_exit_action"] = review.action
                if review.suggested_stop_loss and review.suggested_stop_loss > _as_float(position["stop_loss"]):
                    position["pending_stop_loss"] = round(float(review.suggested_stop_loss), 2)

            position["last_review_date"] = date.date().isoformat()

        if not is_closed:
            keep_rows.append(position)

    new_open = pd.DataFrame(keep_rows, columns=OPEN_COLUMNS) if keep_rows else _empty(OPEN_COLUMNS)
    if closed_rows:
        closed = pd.concat([closed, pd.DataFrame(closed_rows)], ignore_index=True)
    if event_rows:
        events = pd.concat([events, pd.DataFrame(event_rows)], ignore_index=True)
    return new_open[OPEN_COLUMNS].copy(), closed[CLOSED_COLUMNS].copy(), events


def _mark_open(open_: pd.DataFrame, prices: pd.DataFrame, as_of: pd.Timestamp, price_unit_multiplier: float) -> pd.DataFrame:
    if open_.empty:
        return _empty([
            "sleeve_id", "sleeve_label", "symbol", "strategy_name", "signal_date", "entry_date", "entry_price",
            "shares", "cost_value", "stop_loss", "take_profit", "market_date", "market_price", "market_value",
            "unrealized_pnl", "unrealized_pnl_pct", "days_held", "status", "agent_last_action", "agent_last_reason",
            "fill_model",
        ])
    rows = []
    for position in open_.to_dict("records"):
        latest = _latest_row(prices, str(position["symbol"]).upper(), as_of)
        if latest is None:
            continue
        market_price = float(latest["close"]) * price_unit_multiplier
        entry_price = _as_float(position["entry_price"])
        shares = _as_int(position["shares"])
        market_value = shares * market_price
        cost_value = _as_float(position["cost_value"])
        row = dict(position)
        row.update(
            {
                "market_date": pd.Timestamp(latest["date"]).date().isoformat(),
                "market_price": round(market_price, 2),
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(market_value - cost_value, 2),
                "unrealized_pnl_pct": round((market_price / entry_price - 1.0) * 100.0, 2) if entry_price else 0.0,
                "days_held": int((pd.Timestamp(latest["date"]).normalize() - pd.Timestamp(position["entry_date"]).normalize()).days),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _summaries(open_holdings: pd.DataFrame, closed: pd.DataFrame, capital: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if open_holdings.empty:
        strategy_pnl = _empty(["sleeve_id", "strategy_name", "open_positions", "cost_value", "market_value", "unrealized_pnl", "unrealized_pnl_pct"])
        open_by_sleeve = pd.DataFrame(columns=["sleeve_id", "open_positions", "open_cost_value", "open_market_value", "unrealized_pnl"])
    else:
        strategy_pnl = (
            open_holdings.groupby(["sleeve_id", "strategy_name"], dropna=False)
            .agg(open_positions=("symbol", "count"), cost_value=("cost_value", "sum"), market_value=("market_value", "sum"), unrealized_pnl=("unrealized_pnl", "sum"))
            .reset_index()
        )
        strategy_pnl["unrealized_pnl_pct"] = (strategy_pnl["unrealized_pnl"] / strategy_pnl["cost_value"] * 100.0).round(2)
        open_by_sleeve = (
            open_holdings.groupby("sleeve_id", dropna=False)
            .agg(open_positions=("symbol", "count"), open_cost_value=("cost_value", "sum"), open_market_value=("market_value", "sum"), unrealized_pnl=("unrealized_pnl", "sum"))
            .reset_index()
        )

    if closed.empty:
        closed_by_sleeve = pd.DataFrame(columns=["sleeve_id", "realized_pnl", "closed_trades"])
    else:
        closed_by_sleeve = closed.groupby("sleeve_id", dropna=False).agg(realized_pnl=("net_pnl", "sum"), closed_trades=("symbol", "count")).reset_index()

    sleeves = pd.DataFrame({"sleeve_id": ["mvp_p4", "mvp_p5", "flow_v2"]})
    overview = sleeves.merge(open_by_sleeve, on="sleeve_id", how="left").merge(closed_by_sleeve, on="sleeve_id", how="left").fillna(0)
    overview["paper_cash"] = capital + overview["realized_pnl"] - overview["open_cost_value"]
    overview["paper_equity"] = overview["paper_cash"] + overview["open_market_value"]
    overview["unrealized_pnl_pct"] = overview.apply(
        lambda row: (float(row["unrealized_pnl"]) / float(row["open_cost_value"]) * 100.0) if float(row["open_cost_value"]) else 0.0,
        axis=1,
    ).round(2)
    overview = overview[
        [
            "sleeve_id",
            "paper_equity",
            "paper_cash",
            "open_positions",
            "open_cost_value",
            "open_market_value",
            "unrealized_pnl",
            "unrealized_pnl_pct",
            "realized_pnl",
            "closed_trades",
        ]
    ]
    return strategy_pnl, overview


def build_pnl(out_dir: Path, capital: float, lot_size: int, price_unit_multiplier: float, reset_ledger: bool, use_exit_agent: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prices = _price_frame()
    as_of = pd.to_datetime(prices["date"]).max().normalize()
    if reset_ledger:
        for name in ["paper_ledger_open.csv", "paper_ledger_closed.csv", "paper_ledger_events.csv"]:
            path = out_dir / name
            if path.exists():
                path.unlink()
        for name in [
            "paper_realtime_account_pnl.csv",
            "paper_realtime_open_holdings.csv",
            "paper_realtime_strategy_pnl.csv",
            "paper_realtime_status.json",
        ]:
            path = out_dir / name
            if path.exists():
                path.unlink()
        _write_json(
            out_dir / LEDGER_META,
            {
                "started_at": as_of.date().isoformat(),
                "capital_per_sleeve": capital,
                "reset_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "note": "Ledger was reset; historical rolling signals before started_at are ignored.",
            },
        )

    meta = _read_json(out_dir / LEDGER_META)
    start_date = pd.Timestamp(meta["started_at"]).normalize() if meta.get("started_at") else None
    risk_by_strategy = _risk_map()
    open_, closed, events = _load_ledger(out_dir)
    open_, closed, events = _review_exits(
        open_=open_,
        closed=closed,
        events=events,
        prices=prices,
        as_of=as_of,
        price_unit_multiplier=price_unit_multiplier,
        risk_by_strategy=risk_by_strategy,
        use_exit_agent=use_exit_agent,
    )
    open_, events = _enter_positions(
        out_dir=out_dir,
        open_=open_,
        closed=closed,
        events=events,
        prices=prices,
        as_of=as_of,
        capital=capital,
        lot_size=lot_size,
        price_unit_multiplier=price_unit_multiplier,
        risk_by_strategy=risk_by_strategy,
        start_date=start_date,
    )
    # On first adoption the ledger may be seeded from historical paper signals.
    # Review those seeded positions through the current as-of date immediately;
    # same-day new entries still have no later bars and remain untouched.
    open_, closed, events = _review_exits(
        open_=open_,
        closed=closed,
        events=events,
        prices=prices,
        as_of=as_of,
        price_unit_multiplier=price_unit_multiplier,
        risk_by_strategy=risk_by_strategy,
        use_exit_agent=use_exit_agent,
    )

    open_holdings = _mark_open(open_, prices, as_of, price_unit_multiplier)
    strategy_pnl, overview = _summaries(open_holdings, closed, capital)

    open_.to_csv(out_dir / "paper_ledger_open.csv", index=False, encoding="utf-8-sig")
    closed.to_csv(out_dir / "paper_ledger_closed.csv", index=False, encoding="utf-8-sig")
    events.to_csv(out_dir / "paper_ledger_events.csv", index=False, encoding="utf-8-sig")
    open_holdings.to_csv(out_dir / "paper_open_holdings.csv", index=False, encoding="utf-8-sig")
    closed.to_csv(out_dir / "paper_closed_trades.csv", index=False, encoding="utf-8-sig")
    strategy_pnl.to_csv(out_dir / "paper_strategy_pnl.csv", index=False, encoding="utf-8-sig")
    overview.to_csv(out_dir / "paper_account_pnl.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined persistent paper ledger and PnL tables.")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "combined_paper_trading_demo"))
    parser.add_argument("--capital", type=float, default=1_000_000_000.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--price-unit-multiplier", type=float, default=1000.0)
    parser.add_argument("--reset-ledger", action="store_true")
    parser.add_argument("--no-exit-agent", action="store_true")
    args = parser.parse_args()
    build_pnl(
        Path(args.out_dir),
        args.capital,
        args.lot_size,
        args.price_unit_multiplier,
        reset_ledger=args.reset_ledger,
        use_exit_agent=not args.no_exit_agent,
    )
    print(Path(args.out_dir) / "paper_account_pnl.csv")


if __name__ == "__main__":
    main()
