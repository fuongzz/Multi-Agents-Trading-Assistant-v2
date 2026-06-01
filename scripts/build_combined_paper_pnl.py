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
from scripts.backtest_flow_v2_rotation_production_like import RotationConfig, _prepare_features, run_backtest


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
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    frame["ma20"] = frame.groupby("symbol", sort=False)["close"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(20).mean())
    return frame


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


FLOW_SLEEVES = ["flow_v2", "flow_v2_baseline", "flow_v2_tiered"]
HOSTILE_SLEEVES = ["hostile_combo_long"]
CORE_SLEEVES = ["core_mvp9_rank2"]
ALL_SLEEVES = [*FLOW_SLEEVES, *HOSTILE_SLEEVES, *CORE_SLEEVES]
COMPOUND_EQUITY_SLEEVES = {"hostile_combo_long", "core_mvp9_rank2"}


def _risk_for(strategy_name: str, risk_by_strategy: dict[str, dict[str, Any]], sleeve_id: str) -> dict[str, Any]:
    if sleeve_id == "hostile_combo_long":
        return {
            "stop_loss": 0.9999,
            "take_profit": 99.0,
            "max_holding_bars": 100000,
            "trailing_atr_mult": 0.0,
            "trailing_profit_activation": 99.0,
        }
    if sleeve_id in {"flow_v2_baseline", "flow_v2_tiered"}:
        # These sleeves are tracked against the audited Flow logic. Do not
        # introduce the legacy dashboard stop/take-profit overlay.
        return {
            "stop_loss": 0.9999,
            "take_profit": 99.0,
            "max_holding_bars": 100000,
            "trailing_atr_mult": 0.0,
            "trailing_profit_activation": 99.0,
        }
    if sleeve_id in FLOW_SLEEVES:
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


def _filter_by_sleeve_start(frame: pd.DataFrame, sleeve_starts: dict[str, pd.Timestamp]) -> pd.DataFrame:
    if frame.empty or "sleeve_id" not in frame or "signal_date" not in frame:
        return frame
    work = frame.copy()
    signal_dates = pd.to_datetime(work["signal_date"], errors="coerce").dt.normalize()
    keep = pd.Series(True, index=work.index)
    for sleeve_id, started_at in sleeve_starts.items():
        mask = work["sleeve_id"].astype(str).eq(sleeve_id)
        keep &= ~(mask & signal_dates.lt(started_at))
    return work[keep].copy()


def _filter_events_by_sleeve_start(frame: pd.DataFrame, sleeve_starts: dict[str, pd.Timestamp]) -> pd.DataFrame:
    if frame.empty or "sleeve_id" not in frame or "date" not in frame:
        return frame
    work = frame.copy()
    event_dates = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    keep = pd.Series(True, index=work.index)
    for sleeve_id, started_at in sleeve_starts.items():
        mask = work["sleeve_id"].astype(str).eq(sleeve_id)
        keep &= ~(mask & event_dates.lt(started_at))
    return work[keep].copy()


def _cash_for_sleeve(sleeve_id: str, open_: pd.DataFrame, closed: pd.DataFrame, capital: float) -> float:
    open_cost = 0.0 if open_.empty else pd.to_numeric(open_.loc[open_["sleeve_id"] == sleeve_id, "cost_value"], errors="coerce").fillna(0).sum()
    realized = 0.0 if closed.empty else pd.to_numeric(closed.loc[closed["sleeve_id"] == sleeve_id, "net_pnl"], errors="coerce").fillna(0).sum()
    return float(capital + realized - open_cost)


def _equity_for_sleeve(
    sleeve_id: str,
    open_: pd.DataFrame,
    closed: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    capital: float,
    price_unit_multiplier: float,
) -> float:
    cash = _cash_for_sleeve(sleeve_id, open_, closed, capital)
    if open_.empty:
        return cash
    market_value = 0.0
    for item in open_[open_["sleeve_id"] == sleeve_id].to_dict("records"):
        latest = _latest_row(prices, str(item.get("symbol") or "").upper(), as_of)
        if latest is None:
            market_value += _as_float(item.get("cost_value"), 0.0)
            continue
        market_value += _as_int(item.get("shares")) * float(latest["close"]) * price_unit_multiplier
    return float(cash + market_value)


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


def _flow_candidates(source_dir: Path, as_of: pd.Timestamp, fallback_sleeve_id: str, fallback_label: str) -> list[dict[str, Any]]:
    targets = _read_csv(source_dir / "flow_v2_target_plan.csv")
    status = _read_json(source_dir / "flow_v2_status.json")
    if targets.empty:
        return []
    signal_date = pd.Timestamp(status.get("as_of_date") or as_of).normalize()
    if signal_date >= as_of:
        return []
    sleeve_id = str(status.get("sleeve_id") or fallback_sleeve_id)
    sleeve_label = str(status.get("sleeve_label") or fallback_label)
    strategy_name = (
        "flow_v2_rotation_high_rs_flow_heavy_tiered_exit"
        if status.get("early_exit_mode") == "flow_momentum_tiered"
        else "flow_v2_rotation_high_rs_flow_heavy"
    )
    rows = []
    for item in targets.to_dict("records"):
        rows.append(
            {
                "sleeve_id": sleeve_id,
                "sleeve_label": sleeve_label,
                "symbol": str(item.get("symbol") or "").upper(),
                "strategy_name": strategy_name,
                "signal_date": signal_date,
                "max_positions": _as_int(status.get("positions"), 2),
                "priority": _as_int(item.get("rank"), 999),
                "target_value": _as_float(item.get("target_value"), 0.0),
            }
        )
    return rows


def _hostile_candidates(source_dir: Path, as_of: pd.Timestamp) -> list[dict[str, Any]]:
    targets = _read_csv(source_dir / "hostile_target_plan.csv")
    status = _read_json(source_dir / "hostile_status.json")
    if not bool(status.get("paper_trading_enabled")):
        return []
    if targets.empty:
        return []
    signal_date = pd.Timestamp(status.get("as_of_date") or as_of).normalize()
    if signal_date >= as_of:
        return []
    rows = []
    for item in targets.to_dict("records"):
        if str(item.get("action") or "").startswith("PAPER_BUY_CANDIDATE") is False:
            continue
        rows.append(
            {
                "sleeve_id": str(status.get("sleeve_id") or "hostile_combo_long"),
                "sleeve_label": str(status.get("sleeve_label") or "Hostile Combo Long p1"),
                "symbol": str(item.get("symbol") or "").upper(),
                "strategy_name": str(item.get("strategy_name") or "hostile_combo_long"),
                "signal_date": signal_date,
                "max_positions": 1,
                "priority": _as_int(item.get("rank"), 999),
                "target_value": _as_float(item.get("target_value"), 0.0),
                "target_exposure": _as_float(item.get("target_exposure"), _as_float(status.get("target_exposure"), 1.0)),
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
    sleeve_starts: dict[str, pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = [
        *_mvp_candidates(out_dir / "core_mvp9_rank2", "core_mvp9_rank2", "Core MVP9 p2 rank2 compound", as_of),
        *_flow_candidates(out_dir / "flow_v2", as_of, "flow_v2", "Flow V2 Rotation"),
        *_hostile_candidates(out_dir / "hostile_combo_long", as_of),
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
        if sleeve_starts and sleeve_id in sleeve_starts and signal_date < sleeve_starts[sleeve_id]:
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
        cash = _cash_for_sleeve(sleeve_id, open_, closed, capital)
        if sleeve_id in COMPOUND_EQUITY_SLEEVES:
            equity = _equity_for_sleeve(
                sleeve_id,
                open_,
                closed,
                prices,
                pd.Timestamp(entry["date"]).normalize(),
                capital,
                price_unit_multiplier,
            )
            target_exposure = _as_float(item.get("target_exposure"), 1.0)
            if target_exposure <= 0 or target_exposure > 1:
                target_exposure = 1.0
            target_value = equity * target_exposure / max_positions
        else:
            target_value = _as_float(item.get("target_value"), 0.0) or capital / max_positions
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
            "fill_model": (
                "signal_close_T_fill_next_open_T_plus_1_compound_equity"
                if sleeve_id in COMPOUND_EQUITY_SLEEVES
                else "signal_close_T_fill_next_open_T_plus_1"
            ),
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
            elif pending_action == "HOSTILE_MA20_NEXT_OPEN" and holding_bars >= 2:
                reason, exit_price = "HOSTILE_MA20_NEXT_OPEN", float(bar["open"]) * price_unit_multiplier
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

            if reason is None and str(position.get("sleeve_id")) == "hostile_combo_long":
                ma20 = _as_float(bar.get("ma20"), 0.0) * price_unit_multiplier
                if ma20 > 0 and close_v < ma20:
                    position["pending_exit_action"] = "HOSTILE_MA20_NEXT_OPEN"
                    position["agent_last_action"] = "HOSTILE_MA20_NEXT_OPEN"
                    position["agent_last_reason"] = "CLOSE_BELOW_MA20"
                    event_rows.append(
                        {
                            "date": date.date().isoformat(),
                            "sleeve_id": position["sleeve_id"],
                            "symbol": symbol,
                            "event": "EXIT_SIGNAL",
                            "price": round(close_v, 2),
                            "shares": shares,
                            "reason": "HOSTILE_CLOSE_BELOW_MA20",
                            "details": f"ma20={round(ma20, 2)}; execute_next_open=true",
                        }
                    )

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


def _remaining_lots(trades: pd.DataFrame) -> dict[str, list[list[float]]]:
    lots: dict[str, list[list[float]]] = {}
    if trades.empty:
        return lots
    for row in trades.to_dict("records"):
        symbol = str(row["symbol"])
        shares = int(row["shares"])
        if row["side"] == "BUY":
            lots.setdefault(symbol, []).append([float(shares), float(row["price"])])
            continue
        outstanding = shares
        for lot in lots.get(symbol, []):
            if outstanding <= 0:
                break
            consumed = min(outstanding, int(lot[0]))
            lot[0] -= consumed
            outstanding -= consumed
        lots[symbol] = [lot for lot in lots.get(symbol, []) if lot[0] > 0]
    return lots


def _audited_flow_forward_state(
    *,
    out_dir: Path,
    as_of: pd.Timestamp,
    capital: float,
    price_unit_multiplier: float,
    start_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sleeve_configs = [
        ("flow_v2_baseline", "Baseline fresh-signal top2", "none"),
        ("flow_v2_tiered", "Early-exit hai tầng top2", "flow_momentum_tiered"),
    ]
    account_rows: list[dict[str, Any]] = []
    open_frames: list[pd.DataFrame] = []
    strategy_rows: list[dict[str, Any]] = []
    if as_of < start_date:
        for sleeve_id, label, _ in sleeve_configs:
            account_rows.append(
                {
                    "sleeve_id": sleeve_id,
                    "paper_equity": capital,
                    "paper_cash": capital,
                    "open_positions": 0,
                    "open_cost_value": 0.0,
                    "open_market_value": 0.0,
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_pct": 0.0,
                    "realized_pnl": 0.0,
                    "closed_trades": 0,
                }
            )
        return pd.DataFrame(), pd.DataFrame(strategy_rows), pd.DataFrame(account_rows)

    features = _prepare_features("vn100", start_date.date().isoformat(), as_of.date().isoformat())
    for sleeve_id, label, exit_mode in sleeve_configs:
        cfg = RotationConfig(
            universe="vn100",
            start=start_date.date().isoformat(),
            end=as_of.date().isoformat(),
            positions=2,
            rebalance_days=10,
            initial_capital=capital,
            market_gate="risk_on_or_strong_neutral",
            pool_filter="high_rs",
            score_mode="flow_heavy",
            early_exit_mode=exit_mode,
            price_unit_multiplier=price_unit_multiplier,
            liquidate_at_end=False,
        )
        result = run_backtest(cfg, features=features)
        trades = result["trades"].copy()
        trades.to_csv(out_dir / sleeve_id / "paper_forward_orders.csv", index=False, encoding="utf-8-sig")
        result["equity"].to_csv(out_dir / sleeve_id / "paper_forward_equity.csv", index=False, encoding="utf-8-sig")
        lots = _remaining_lots(trades)
        latest_day = features.loc[features["date"] <= as_of].sort_values("date").groupby("symbol", as_index=False).tail(1)
        close_by_symbol = latest_day.set_index("symbol")["close"] * price_unit_multiplier if not latest_day.empty else pd.Series(dtype=float)
        open_rows: list[dict[str, Any]] = []
        for symbol, shares in result["open_positions"].items():
            symbol_lots = lots.get(symbol, [])
            lot_shares = sum(int(lot[0]) for lot in symbol_lots)
            entry_price = (
                sum(float(lot[0]) * float(lot[1]) for lot in symbol_lots) / lot_shares
                if lot_shares
                else 0.0
            )
            market_price = float(close_by_symbol.get(symbol, entry_price))
            cost_value = entry_price * int(shares)
            market_value = market_price * int(shares)
            open_rows.append(
                {
                    "sleeve_id": sleeve_id,
                    "sleeve_label": label,
                    "symbol": symbol,
                    "strategy_name": sleeve_id,
                    "signal_date": "",
                    "entry_date": "",
                    "entry_price": round(entry_price, 2),
                    "shares": int(shares),
                    "cost_value": round(cost_value, 2),
                    "stop_loss": "",
                    "take_profit": "",
                    "market_date": as_of.date().isoformat(),
                    "market_price": round(market_price, 2),
                    "market_value": round(market_value, 2),
                    "unrealized_pnl": round(market_value - cost_value, 2),
                    "unrealized_pnl_pct": round((market_price / entry_price - 1.0) * 100.0, 2) if entry_price else 0.0,
                    "days_held": "",
                    "status": "AUDITED_FORWARD_PAPER",
                    "agent_last_action": "",
                    "agent_last_reason": "",
                    "fill_model": "audited_flow_signal_close_T_fill_open_T_plus_1",
                }
            )
        open_frame = pd.DataFrame(open_rows)
        if not open_frame.empty:
            open_frames.append(open_frame)
        final_equity = float(result["summary"]["ending_equity"])
        cash = float(result["ending_cash"])
        market_value = final_equity - cash
        gross_open_cost = float(open_frame["cost_value"].sum()) if not open_frame.empty else 0.0
        unrealized = float(open_frame["unrealized_pnl"].sum()) if not open_frame.empty else 0.0
        realized = final_equity - capital - unrealized
        account_rows.append(
            {
                "sleeve_id": sleeve_id,
                "paper_equity": round(final_equity, 2),
                "paper_cash": round(cash, 2),
                "open_positions": int(len(result["open_positions"])),
                "open_cost_value": round(gross_open_cost, 2),
                "open_market_value": round(market_value, 2),
                "unrealized_pnl": round(unrealized, 2),
                "unrealized_pnl_pct": round((unrealized / gross_open_cost * 100.0), 2) if gross_open_cost else 0.0,
                "realized_pnl": round(realized, 2),
                "closed_trades": int((trades["side"] == "SELL").sum()) if not trades.empty else 0,
            }
        )
        strategy_rows.append(
            {
                "sleeve_id": sleeve_id,
                "strategy_name": sleeve_id,
                "open_positions": int(len(result["open_positions"])),
                "cost_value": round(gross_open_cost, 2),
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(unrealized, 2),
                "unrealized_pnl_pct": round((unrealized / gross_open_cost * 100.0), 2) if gross_open_cost else 0.0,
            }
        )
    return (
        pd.concat(open_frames, ignore_index=True, sort=False) if open_frames else pd.DataFrame(),
        pd.DataFrame(strategy_rows),
        pd.DataFrame(account_rows),
    )


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

    sleeves = pd.DataFrame({"sleeve_id": ALL_SLEEVES})
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


def build_pnl(
    out_dir: Path,
    capital: float,
    lot_size: int,
    price_unit_multiplier: float,
    reset_ledger: bool,
    use_exit_agent: bool,
    paper_start_date: str | None = None,
) -> None:
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
        reset_start = pd.Timestamp(paper_start_date).normalize() if paper_start_date else (as_of + pd.Timedelta(days=1))
        _write_json(
            out_dir / LEDGER_META,
            {
                "started_at": reset_start.date().isoformat(),
                "audited_flow_started_at": reset_start.date().isoformat(),
                "capital_per_sleeve": capital,
                "reset_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "note": "Fresh paper run; historical signals and positions before started_at are ignored.",
                "audited_flow_note": "Audited Flow sleeves accept only signals formed on or after the fresh paper start date.",
            },
        )

    meta = _read_json(out_dir / LEDGER_META)
    if not meta.get("audited_flow_started_at"):
        meta["audited_flow_started_at"] = (as_of + pd.Timedelta(days=1)).date().isoformat()
        meta["audited_flow_note"] = "flow_v2_baseline and flow_v2_tiered run forward only from the next session after adoption."
        _write_json(out_dir / LEDGER_META, meta)
    sleeve_started_at = dict(meta.get("sleeve_started_at") or {})
    if "core_mvp9_rank2" not in sleeve_started_at:
        sleeve_started_at["core_mvp9_rank2"] = (as_of + pd.Timedelta(days=1)).date().isoformat()
        meta["sleeve_started_at"] = sleeve_started_at
        _write_json(out_dir / LEDGER_META, meta)
    sleeve_starts = {
        sleeve_id: pd.Timestamp(value).normalize()
        for sleeve_id, value in sleeve_started_at.items()
        if value
    }
    start_date = pd.Timestamp(meta["started_at"]).normalize() if meta.get("started_at") else None
    risk_by_strategy = _risk_map()
    open_, closed, events = _load_ledger(out_dir)
    open_ = _filter_by_sleeve_start(open_, sleeve_starts)
    closed = _filter_by_sleeve_start(closed, sleeve_starts)
    events = _filter_events_by_sleeve_start(events, sleeve_starts)
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
        sleeve_starts=sleeve_starts,
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
    if not open_holdings.empty and "sleeve_id" in open_holdings:
        open_holdings = open_holdings[open_holdings["sleeve_id"].astype(str).isin(ALL_SLEEVES)].copy()
    if not closed.empty and "sleeve_id" in closed:
        closed_for_summary = closed[closed["sleeve_id"].astype(str).isin(ALL_SLEEVES)].copy()
    else:
        closed_for_summary = closed
    strategy_pnl, overview = _summaries(open_holdings, closed_for_summary, capital)
    audited_open, audited_strategy, audited_overview = _audited_flow_forward_state(
        out_dir=out_dir,
        as_of=as_of,
        capital=capital,
        price_unit_multiplier=price_unit_multiplier,
        start_date=pd.Timestamp(meta["audited_flow_started_at"]).normalize(),
    )
    if not audited_open.empty:
        open_holdings = pd.concat([open_holdings, audited_open], ignore_index=True, sort=False)
    if not audited_strategy.empty:
        strategy_pnl = pd.concat(
            [strategy_pnl.loc[~strategy_pnl["sleeve_id"].isin({"flow_v2_baseline", "flow_v2_tiered"})], audited_strategy],
            ignore_index=True,
            sort=False,
        )
    overview = pd.concat(
        [overview.loc[~overview["sleeve_id"].isin({"flow_v2_baseline", "flow_v2_tiered"})], audited_overview],
        ignore_index=True,
        sort=False,
    )

    open_.to_csv(out_dir / "paper_ledger_open.csv", index=False, encoding="utf-8-sig")
    closed.to_csv(out_dir / "paper_ledger_closed.csv", index=False, encoding="utf-8-sig")
    events.to_csv(out_dir / "paper_ledger_events.csv", index=False, encoding="utf-8-sig")
    open_holdings.to_csv(out_dir / "paper_open_holdings.csv", index=False, encoding="utf-8-sig")
    closed_for_summary.to_csv(out_dir / "paper_closed_trades.csv", index=False, encoding="utf-8-sig")
    strategy_pnl.to_csv(out_dir / "paper_strategy_pnl.csv", index=False, encoding="utf-8-sig")
    overview.to_csv(out_dir / "paper_account_pnl.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined persistent paper ledger and PnL tables.")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "combined_paper_trading_demo"))
    parser.add_argument("--capital", type=float, default=1_000_000_000.0)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--price-unit-multiplier", type=float, default=1000.0)
    parser.add_argument("--reset-ledger", action="store_true")
    parser.add_argument("--start-date", default="", help="Paper activation date used with --reset-ledger, YYYY-MM-DD.")
    parser.add_argument("--no-exit-agent", action="store_true")
    args = parser.parse_args()
    build_pnl(
        Path(args.out_dir),
        args.capital,
        args.lot_size,
        args.price_unit_multiplier,
        reset_ledger=args.reset_ledger,
        use_exit_agent=not args.no_exit_agent,
        paper_start_date=args.start_date or None,
    )
    print(Path(args.out_dir) / "paper_account_pnl.csv")


if __name__ == "__main__":
    main()
