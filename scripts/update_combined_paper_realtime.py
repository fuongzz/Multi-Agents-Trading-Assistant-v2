"""Refresh combined paper dashboard with live/snapshot prices.

This writes a realtime overlay only. The persistent paper ledger remains the
source of truth for entries, exits, realized PnL and stop state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, time as day_time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiagents_trading_assistant.fetcher import get_live_price
from scripts.build_combined_paper_pnl import (
    EVENT_COLUMNS,
    OPEN_COLUMNS,
    _cash_for_sleeve,
    _existing_signal_keys,
    _load_ledger,
    _lot_shares,
    _risk_for,
    _risk_map,
    _summaries as _ledger_summaries,
)
from scripts.export_combined_paper_dashboard import export_dashboard


MORNING_START = day_time(9, 0)
MORNING_END = day_time(11, 30)
AFTERNOON_START = day_time(13, 0)
AFTERNOON_END = day_time(15, 0)
FLOW_OVERLAY_REFRESH_SECONDS = 120
FLOW_OVERLAY_LIMIT = 10


def _export_dashboard(out_dir: Path) -> None:
    export_dashboard(
        out_dir,
        None,
        None,
        out_dir / "flow_v2",
        out_dir / "flow_v2_baseline",
        out_dir / "flow_v2_tiered",
        out_dir / "hostile_combo_long",
        out_dir / "core_mvp9_rank2",
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(str(value).replace(",", "").replace("%", "").strip())
    except Exception:
        return default


def _normalize_live_price(live_price: float | None, reference_price: float | None) -> float | None:
    if live_price is None or live_price <= 0:
        return None
    price = float(live_price)
    if reference_price is None or reference_price <= 0:
        return price * 1000.0 if price < 1000.0 else price
    ref = float(reference_price)
    if price / ref > 100.0:
        price /= 1000.0
    elif ref / price > 100.0:
        price *= 1000.0
    return price


def _seconds_until(target: datetime, *, minimum: int = 30, maximum: int = 900) -> int:
    seconds = int((target - datetime.now()).total_seconds())
    return max(minimum, min(maximum, seconds))


def _market_session_state(now: datetime | None = None) -> tuple[bool, str, int]:
    now = now or datetime.now()
    today = now.date()
    current = now.time()
    if now.weekday() >= 5:
        days_until_monday = 7 - now.weekday()
        return False, "weekend", _seconds_until(datetime.combine(today + timedelta(days=days_until_monday), MORNING_START))
    morning_open = datetime.combine(today, MORNING_START)
    morning_close = datetime.combine(today, MORNING_END)
    afternoon_open = datetime.combine(today, AFTERNOON_START)
    if MORNING_START <= current <= MORNING_END:
        return True, "morning", 0
    if AFTERNOON_START <= current <= AFTERNOON_END:
        return True, "afternoon", 0
    if now < morning_open:
        return False, "before_open", _seconds_until(morning_open)
    if morning_close < now < afternoon_open:
        return False, "lunch_break", _seconds_until(afternoon_open)
    next_day = today + timedelta(days=1)
    while datetime.combine(next_day, MORNING_START).weekday() >= 5:
        next_day += timedelta(days=1)
    return False, "after_close", _seconds_until(datetime.combine(next_day, MORNING_START))


def _exit_alert(row: pd.Series, price: float) -> str:
    stop = _as_float(row.get("stop_loss"))
    target = _as_float(row.get("take_profit"))
    if stop > 0 and price <= stop:
        return "LIVE_STOP_TOUCH"
    if target > 0 and price >= target:
        return "LIVE_TAKE_PROFIT_TOUCH"
    return ""


def _summaries(open_holdings: pd.DataFrame, base_account: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if open_holdings.empty:
        strategy = pd.DataFrame(columns=["sleeve_id", "strategy_name", "open_positions", "cost_value", "market_value", "unrealized_pnl", "unrealized_pnl_pct"])
        return strategy, base_account

    strategy = (
        open_holdings.groupby(["sleeve_id", "strategy_name"], dropna=False)
        .agg(
            open_positions=("symbol", "count"),
            cost_value=("cost_value", "sum"),
            market_value=("market_value", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
        )
        .reset_index()
    )
    strategy["unrealized_pnl_pct"] = (strategy["unrealized_pnl"] / strategy["cost_value"] * 100.0).round(2)

    live_by_sleeve = (
        open_holdings.groupby("sleeve_id", dropna=False)
        .agg(
            open_positions=("symbol", "count"),
            open_cost_value=("cost_value", "sum"),
            open_market_value=("market_value", "sum"),
            unrealized_pnl=("unrealized_pnl", "sum"),
        )
        .reset_index()
    )
    account = base_account.copy()
    if account.empty:
        account = pd.DataFrame({"sleeve_id": sorted(open_holdings["sleeve_id"].astype(str).unique())})
    keep = [column for column in account.columns if column not in {"open_positions", "open_cost_value", "open_market_value", "unrealized_pnl", "unrealized_pnl_pct", "paper_equity"}]
    account = account[keep].merge(live_by_sleeve, on="sleeve_id", how="left").fillna(0)
    if "paper_cash" not in account:
        account["paper_cash"] = 0.0
    account["paper_equity"] = account["paper_cash"] + account["open_market_value"]
    account["unrealized_pnl_pct"] = account.apply(
        lambda row: (float(row["unrealized_pnl"]) / float(row["open_cost_value"]) * 100.0) if float(row["open_cost_value"]) else 0.0,
        axis=1,
    ).round(2)
    ordered = [
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
    for column in ordered:
        if column not in account:
            account[column] = 0
    return strategy, account[ordered]


def _pending_paper_targets(out_dir: Path, trade_date: pd.Timestamp) -> pd.DataFrame:
    pending_frames: list[pd.DataFrame] = []
    signal_ledger = _pending_targets_from_signal_ledger(out_dir, trade_date)
    if not signal_ledger.empty:
        pending_frames.append(signal_ledger)

    plans = _read_csv(out_dir / "17_signal_plans.csv")
    actions = _read_csv(out_dir / "05_orders_targets.csv")
    if not plans.empty and not actions.empty:
        plans["signal_date"] = pd.to_datetime(plans["signal_date"], errors="coerce").dt.normalize()
        targets = plans[
            plans["decision_status"].astype(str).eq("PAPER_TARGET")
            & (plans["signal_date"] < trade_date)
        ].copy()
        if not targets.empty:
            detail_columns = ["sleeve_id", "symbol", "max_positions_context"]
            if "target_exposure" in actions:
                detail_columns.append("target_exposure")
            details = actions[
                actions["action"].astype(str).str.startswith("PAPER_BUY_CANDIDATE")
            ][detail_columns].drop_duplicates()
            targets = targets.merge(details, on=["sleeve_id", "symbol"], how="left")
            pending_frames.append(targets)

    rolling = _pending_mvp_targets_from_rolling_history(out_dir, trade_date)
    if not rolling.empty:
        pending_frames.append(rolling)

    if not pending_frames:
        return pd.DataFrame()
    targets = pd.concat(pending_frames, ignore_index=True, sort=False)
    targets["signal_date"] = pd.to_datetime(targets["signal_date"], errors="coerce").dt.normalize()
    targets["symbol"] = targets["symbol"].astype(str).str.upper()
    targets["max_positions_context"] = targets.get("max_positions_context", 2)
    targets["max_positions_context"] = pd.to_numeric(targets["max_positions_context"], errors="coerce").fillna(2)
    if "score" not in targets:
        targets["score"] = 0.0
    targets["score"] = pd.to_numeric(targets["score"], errors="coerce").fillna(0.0)
    return targets.sort_values(["signal_date", "sleeve_id", "score"], ascending=[True, True, False]).drop_duplicates(
        ["sleeve_id", "symbol", "signal_date"],
        keep="first",
    )


def _pending_targets_from_signal_ledger(out_dir: Path, trade_date: pd.Timestamp) -> pd.DataFrame:
    pending = _read_csv(out_dir / "paper_signal_pending.csv")
    if pending.empty:
        pending = _read_csv(out_dir / "18_paper_signal_pending.csv")
    if pending.empty:
        return pd.DataFrame()
    pending["signal_date"] = pd.to_datetime(pending["signal_date"], errors="coerce").dt.normalize()
    pending = pending[
        pending["decision_status"].astype(str).eq("PAPER_TARGET")
        & (pending["signal_date"] < trade_date)
    ].copy()
    if pending.empty:
        return pending
    for column in ["score", "reference_close_vnd", "max_positions_context", "target_exposure"]:
        if column not in pending:
            pending[column] = pd.NA
    pending["score"] = pd.to_numeric(pending["score"], errors="coerce").fillna(0.0)
    pending["reference_close_vnd"] = pd.to_numeric(pending["reference_close_vnd"], errors="coerce")
    pending["max_positions_context"] = pd.to_numeric(pending["max_positions_context"], errors="coerce").fillna(2)
    pending["target_exposure"] = pd.to_numeric(pending["target_exposure"], errors="coerce")
    pending["source"] = pending.get("source", "paper_signal_pending")
    return pending


def _previous_rolling_date(out_dir: Path, sleeve_id: str, trade_date: pd.Timestamp) -> pd.Timestamp | None:
    summary = _read_csv(out_dir / "09_rolling_summary.csv")
    if not summary.empty and {"sleeve_id", "date"}.issubset(summary.columns):
        summary = summary[summary["sleeve_id"].astype(str).eq(sleeve_id)].copy()
        summary["date"] = pd.to_datetime(summary["date"], errors="coerce").dt.normalize()
        previous = summary[summary["date"] < trade_date]["date"].dropna()
        if not previous.empty:
            return pd.Timestamp(previous.max()).normalize()
    actions = _read_csv(out_dir / sleeve_id / "rolling_candidate_actions_14d.csv")
    if actions.empty or "date" not in actions:
        return None
    actions["date"] = pd.to_datetime(actions["date"], errors="coerce").dt.normalize()
    previous = actions[actions["date"] < trade_date]["date"].dropna()
    return None if previous.empty else pd.Timestamp(previous.max()).normalize()


def _pending_mvp_targets_from_rolling_history(out_dir: Path, trade_date: pd.Timestamp) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sleeve_id in ["core_mvp9_rank2"]:
        signal_date = _previous_rolling_date(out_dir, sleeve_id, trade_date)
        if signal_date is None:
            continue
        actions = _read_csv(out_dir / sleeve_id / "rolling_candidate_actions_14d.csv")
        if actions.empty:
            continue
        actions["date"] = pd.to_datetime(actions["date"], errors="coerce").dt.normalize()
        targets = actions[
            (actions["date"] == signal_date)
            & actions.get("action", pd.Series(dtype=str)).astype(str).str.startswith("PAPER_BUY_CANDIDATE")
        ].copy()
        if targets.empty:
            continue
        targets["sleeve_id"] = sleeve_id
        targets["signal_date"] = signal_date
        targets["decision_status"] = "PAPER_TARGET"
        targets["score"] = pd.to_numeric(targets.get("edge_rank_score"), errors="coerce").fillna(0.0)
        targets["reference_close_vnd"] = pd.NA
        frames.append(
            targets[
                [
                    "sleeve_id",
                    "symbol",
                    "strategy_name",
                    "signal_date",
                    "decision_status",
                    "score",
                    "reference_close_vnd",
                    "max_positions_context",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _paper_holding_row(row: dict[str, Any], market_date: str) -> dict[str, Any]:
    cost = float(row["cost_value"])
    return {
        **row,
        "market_date": market_date,
        "market_price": row["entry_price"],
        "market_value": cost,
        "unrealized_pnl": 0.0,
        "unrealized_pnl_pct": 0.0,
        "days_held": 0,
    }


def _enter_realtime_paper_positions(
    out_dir: Path,
    live_prices: dict[str, float],
    trade_date: pd.Timestamp,
    *,
    capital: float = 100_000_000.0,
    lot_size: int = 100,
) -> list[dict[str, Any]]:
    targets = _pending_paper_targets(out_dir, trade_date)
    if targets.empty:
        return []
    open_, closed, events = _load_ledger(out_dir)
    keys = _existing_signal_keys(open_, closed)
    risk_by_strategy = _risk_map()
    entered: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for target in targets.sort_values(["signal_date", "sleeve_id", "score"], ascending=[True, True, False]).to_dict("records"):
        sleeve_id = str(target["sleeve_id"])
        symbol = str(target["symbol"]).upper()
        signal_date = pd.Timestamp(target["signal_date"]).normalize()
        key = (sleeve_id, symbol, signal_date.date().isoformat())
        if key in keys:
            continue
        if sleeve_id in {"flow_v2_baseline", "flow_v2_tiered"}:
            continue
        if not open_.empty and ((open_["sleeve_id"] == sleeve_id) & (open_["symbol"].astype(str).str.upper() == symbol)).any():
            continue
        max_positions = max(1, int(float(target.get("max_positions_context") or 2)))
        current_positions = 0 if open_.empty else int((open_["sleeve_id"] == sleeve_id).sum())
        if current_positions >= max_positions:
            continue
        raw_price = live_prices.get(symbol)
        entry_price = _normalize_live_price(raw_price, _as_float(target.get("reference_close_vnd")))
        if entry_price is None or entry_price <= 0:
            continue
        cash = _cash_for_sleeve(sleeve_id, open_, closed, capital)
        open_market_value = 0.0
        if not open_.empty:
            for existing in open_[open_["sleeve_id"] == sleeve_id].to_dict("records"):
                existing_symbol = str(existing.get("symbol") or "").upper()
                market_price = _normalize_live_price(live_prices.get(existing_symbol), _as_float(existing.get("entry_price")))
                open_market_value += int(existing.get("shares") or 0) * float(market_price or 0.0)
        target_equity = cash + open_market_value
        target_exposure = _as_float(target.get("target_exposure"), 1.0)
        if target_exposure <= 0 or target_exposure > 1:
            target_exposure = 1.0
        shares = _lot_shares(min(target_equity * target_exposure / max_positions, cash), entry_price, lot_size)
        if shares <= 0:
            continue
        strategy_name = str(target.get("strategy_name") or sleeve_id)
        risk = _risk_for(strategy_name, risk_by_strategy, sleeve_id)
        row = {
            "sleeve_id": sleeve_id,
            "sleeve_label": sleeve_id,
            "symbol": symbol,
            "strategy_name": strategy_name,
            "signal_date": signal_date.date().isoformat(),
            "entry_date": trade_date.date().isoformat(),
            "entry_price": round(entry_price, 2),
            "shares": shares,
            "cost_value": round(shares * entry_price, 2),
            "stop_loss": round(entry_price * (1.0 - float(risk["stop_loss"])), 2),
            "take_profit": round(entry_price * (1.0 + float(risk["take_profit"])), 2),
            "max_holding_bars": int(risk["max_holding_bars"]),
            "highest_price": round(entry_price, 2),
            "last_review_date": trade_date.date().isoformat(),
            "pending_exit_action": "",
            "pending_stop_loss": "",
            "agent_last_action": "",
            "agent_last_reason": "",
            "status": "OPEN_LEDGER",
            "fill_model": "live_quote_after_market_open_T_plus_1_compound_equity",
        }
        entered.append(row)
        event_rows.append(
            {
                "date": row["entry_date"],
                "sleeve_id": sleeve_id,
                "symbol": symbol,
                "event": "ENTRY",
                "price": row["entry_price"],
                "shares": shares,
                "reason": "REALTIME_PAPER_BUY_CANDIDATE",
                "details": f"signal_date={row['signal_date']}; fill_model={row['fill_model']}; no_broker_order=true",
            }
        )
        open_ = pd.concat([open_, pd.DataFrame([row])], ignore_index=True)
        keys.add(key)
    if not entered:
        return []
    events = pd.concat([events, pd.DataFrame(event_rows, columns=EVENT_COLUMNS)], ignore_index=True)
    open_[OPEN_COLUMNS].to_csv(out_dir / "paper_ledger_open.csv", index=False, encoding="utf-8-sig")
    events.to_csv(out_dir / "paper_ledger_events.csv", index=False, encoding="utf-8-sig")

    holdings = _read_csv(out_dir / "paper_open_holdings.csv")
    new_holdings = pd.DataFrame([_paper_holding_row(row, trade_date.date().isoformat()) for row in entered])
    holdings = pd.concat([holdings, new_holdings], ignore_index=True, sort=False)
    holdings.to_csv(out_dir / "paper_open_holdings.csv", index=False, encoding="utf-8-sig")
    strategy, account = _ledger_summaries(holdings, closed, capital)
    strategy.to_csv(out_dir / "paper_strategy_pnl.csv", index=False, encoding="utf-8-sig")
    account.to_csv(out_dir / "paper_account_pnl.csv", index=False, encoding="utf-8-sig")
    return entered


def _build_realtime_flow_rows(
    watchlist: pd.DataFrame,
    price_board: pd.DataFrame,
    *,
    priced_at: str,
) -> pd.DataFrame:
    if watchlist.empty:
        return pd.DataFrame()
    base_columns = [
        "rank",
        "symbol",
        "industry",
        "flow_signal",
        "flow_trend_5d",
        "inflow_days_5d",
        "flow_score_today",
    ]
    base = watchlist[[col for col in base_columns if col in watchlist.columns]].copy()
    base["symbol"] = base["symbol"].astype(str).str.upper()
    board = price_board.copy()
    if board.empty:
        board = pd.DataFrame(columns=["symbol", "reference_price", "close_price", "foreign_buy_volume", "foreign_sell_volume"])
    board["symbol"] = board.get("symbol", pd.Series(dtype=str)).astype(str).str.upper()
    keep_board = [
        col
        for col in ["symbol", "reference_price", "close_price", "foreign_buy_volume", "foreign_sell_volume"]
        if col in board.columns
    ]
    overlay = base.merge(board[keep_board].drop_duplicates("symbol", keep="last"), on="symbol", how="left")
    for column in ["reference_price", "close_price", "foreign_buy_volume", "foreign_sell_volume"]:
        overlay[column] = pd.to_numeric(overlay.get(column), errors="coerce")
    overlay["intraday_change_pct"] = (
        (overlay["close_price"] / overlay["reference_price"] - 1.0) * 100.0
    ).where(overlay["reference_price"] > 0).round(2)
    overlay["foreign_net_volume"] = (
        overlay["foreign_buy_volume"].fillna(0.0) - overlay["foreign_sell_volume"].fillna(0.0)
    )

    def confirmation(row: pd.Series) -> str:
        change = row.get("intraday_change_pct")
        foreign_net = row.get("foreign_net_volume")
        if pd.isna(row.get("close_price")):
            return "CHUA_CO_SNAPSHOT"
        if change > 0 and foreign_net > 0:
            return "XAC_NHAN_VAO_TRONG_PHIEN"
        if foreign_net > 0:
            return "KHOI_NGOAI_MUA_RONG"
        if change > 0:
            return "GIA_DANG_XAC_NHAN"
        if change < 0 and foreign_net < 0:
            return "SUY_YEU_TRONG_PHIEN"
        return "THEO_DOI_TRONG_PHIEN"

    overlay["realtime_confirmation"] = overlay.apply(confirmation, axis=1)
    overlay["priced_at"] = priced_at
    overlay["data_basis"] = "EOD Flow V2 candidate + intraday price board/foreign volume; display only"
    return overlay


def _fetch_realtime_flow_board(symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    from vnstock_data import Market

    market = Market()
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for symbol in symbols:
        try:
            board = market.equity(symbol).price_board()
            if board is not None and not board.empty:
                frames.append(board)
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
    if not frames:
        return pd.DataFrame(), errors
    return pd.concat(frames, ignore_index=True), errors


def _refresh_realtime_flow_overlay(
    out_dir: Path,
    now: datetime,
    *,
    minimum_seconds: int = FLOW_OVERLAY_REFRESH_SECONDS,
) -> dict[str, Any]:
    status_path = out_dir / "realtime_money_flow_status.json"
    csv_path = out_dir / "20_realtime_money_flow.csv"
    prior_status = {}
    if status_path.exists():
        prior_status = json.loads(status_path.read_text(encoding="utf-8"))
    watchlist = _read_csv(out_dir / "14_money_flow_today.csv")
    if watchlist.empty:
        watchlist = _read_csv(out_dir / "flow_v2" / "flow_v2_money_flow_today.csv")
    watchlist = watchlist.head(FLOW_OVERLAY_LIMIT).copy()
    in_session, session_label, _ = _market_session_state(now)
    status: dict[str, Any] = {
        "updated_at": now.isoformat(timespec="seconds"),
        "active": in_session,
        "market_session": session_label,
        "watchlist_symbols": int(len(watchlist)),
        "basis": "Flow V2 daily candidates plus intraday price-board and foreign-volume confirmation.",
        "display_only": True,
        "trading_effect": "No gate, target or paper order is changed by this overlay.",
        "last_live_update": prior_status.get("last_live_update", ""),
    }
    if not in_session:
        status["state"] = "WAITING_FOR_MARKET_SESSION"
        status["rows"] = int(len(_read_csv(csv_path)))
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status
    if watchlist.empty:
        status["state"] = "NO_EOD_FLOW_WATCHLIST"
        status["rows"] = 0
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status
    previous_live = pd.to_datetime(prior_status.get("last_live_update"), errors="coerce")
    if (
        pd.notna(previous_live)
        and (now - previous_live.to_pydatetime()).total_seconds() < minimum_seconds
        and csv_path.exists()
    ):
        status["state"] = "LIVE_CACHED"
        status["last_live_update"] = prior_status["last_live_update"]
        status["rows"] = int(len(_read_csv(csv_path)))
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status
    symbols = watchlist["symbol"].dropna().astype(str).str.upper().tolist()
    board, errors = _fetch_realtime_flow_board(symbols)
    priced_at = now.isoformat(timespec="seconds")
    overlay = _build_realtime_flow_rows(watchlist, board, priced_at=priced_at)
    overlay.to_csv(csv_path, index=False, encoding="utf-8-sig")
    status["state"] = "LIVE" if not overlay.empty else "LIVE_NO_QUOTES"
    status["rows"] = int(len(overlay))
    status["last_live_update"] = priced_at
    status["fetch_errors"] = errors[:5]
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def refresh_once(
    out_dir: Path,
    *,
    html_refresh_seconds: int = 0,
    realtime_paper_entries: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    open_path = out_dir / "paper_open_holdings.csv"
    if not open_path.exists():
        raise FileNotFoundError(f"Missing open holdings: {open_path}")
    now = now or datetime.now()
    holdings = _read_csv(open_path)
    base_account = _read_csv(out_dir / "paper_account_pnl.csv")
    flow_overlay_status = _refresh_realtime_flow_overlay(out_dir, now)
    in_session, _, _ = _market_session_state(now)
    targets = _pending_paper_targets(out_dir, pd.Timestamp(now.date())) if realtime_paper_entries and in_session else pd.DataFrame()
    symbols = set(holdings.get("symbol", pd.Series(dtype=str)).dropna().astype(str).str.upper())
    symbols.update(targets.get("symbol", pd.Series(dtype=str)).dropna().astype(str).str.upper())
    raw_prices = get_live_price(sorted(symbols)) if symbols else {}
    entered = (
        _enter_realtime_paper_positions(out_dir, raw_prices, pd.Timestamp(now.date()))
        if realtime_paper_entries and in_session and not targets.empty
        else []
    )
    if entered:
        holdings = _read_csv(open_path)
        base_account = _read_csv(out_dir / "paper_account_pnl.csv")
    priced_at = now.isoformat(timespec="seconds")
    if holdings.empty:
        status = {
            "priced_at": priced_at,
            "updated_holdings": 0,
            "updated": 0,
            "symbols": len(symbols),
            "realtime_paper_entries": realtime_paper_entries,
            "paper_entries": 0,
            "realtime_money_flow_state": flow_overlay_status.get("state"),
            "html_refresh_seconds": html_refresh_seconds,
        }
        (out_dir / "paper_realtime_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        _export_dashboard(out_dir)
        return status

    holdings["symbol"] = holdings["symbol"].astype(str).str.upper().str.strip()
    holding_symbols = holdings["symbol"].dropna().unique().tolist()
    missing_symbols = [symbol for symbol in holding_symbols if symbol not in raw_prices]
    if missing_symbols:
        raw_prices.update(get_live_price(missing_symbols))

    updated = 0
    for idx, row in holdings.iterrows():
        symbol = str(row["symbol"]).upper()
        entry_price = _as_float(row.get("entry_price"))
        shares = _as_float(row.get("shares"))
        raw_live = raw_prices.get(symbol)
        live = _normalize_live_price(raw_live, entry_price)
        fallback = _as_float(row.get("market_price"))
        price = live if live and live > 0 else fallback
        holdings.at[idx, "price_source"] = "live" if live and live > 0 else "latest_close"
        holdings.at[idx, "raw_live_price"] = round(float(raw_live), 4) if raw_live else None
        holdings.at[idx, "priced_at"] = priced_at if live and live > 0 else ""
        holdings.at[idx, "live_exit_alert"] = _exit_alert(row, price) if price else ""
        if not price or not entry_price or not shares:
            continue
        market_value = shares * price
        pnl = (price - entry_price) * shares
        holdings.at[idx, "market_price"] = round(price, 2)
        holdings.at[idx, "market_value"] = round(market_value, 2)
        holdings.at[idx, "unrealized_pnl"] = round(pnl, 2)
        holdings.at[idx, "unrealized_pnl_pct"] = round((price / entry_price - 1.0) * 100.0, 2)
        if live and live > 0:
            updated += 1

    strategy, account = _summaries(holdings, base_account)
    holdings.to_csv(out_dir / "paper_realtime_open_holdings.csv", index=False, encoding="utf-8-sig")
    strategy.to_csv(out_dir / "paper_realtime_strategy_pnl.csv", index=False, encoding="utf-8-sig")
    account.to_csv(out_dir / "paper_realtime_account_pnl.csv", index=False, encoding="utf-8-sig")
    status = {
        "priced_at": priced_at,
        "updated_holdings": updated,
        "updated": updated,
        "symbols": len(holding_symbols),
        "realtime_paper_entries": realtime_paper_entries,
        "paper_entries": len(entered),
        "entry_symbols": sorted({str(row["symbol"]) for row in entered}),
        "realtime_money_flow_state": flow_overlay_status.get("state"),
        "html_refresh_seconds": html_refresh_seconds,
    }
    (out_dir / "paper_realtime_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    _export_dashboard(out_dir)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh combined paper dashboard with realtime/snapshot prices.")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "combined_paper_trading_demo"))
    parser.add_argument("--interval-seconds", type=int, default=0)
    parser.add_argument("--html-refresh-seconds", type=int, default=0)
    parser.add_argument("--market-session-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--realtime-paper-entries", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    while True:
        if args.interval_seconds > 0 and args.market_session_only:
            in_session, label, sleep_seconds = _market_session_state()
            if not in_session:
                print(f"[combined_realtime] {datetime.now().isoformat(timespec='seconds')} paused={label} retry_in={sleep_seconds}s", flush=True)
                time.sleep(sleep_seconds)
                continue
        status = refresh_once(
            out_dir,
            html_refresh_seconds=args.html_refresh_seconds or args.interval_seconds,
            realtime_paper_entries=args.realtime_paper_entries,
        )
        print(
            f"[combined_realtime] {status['priced_at']} updated_holdings={status['updated_holdings']} "
            f"paper_entries={status.get('paper_entries', 0)} symbols={status['symbols']} out_dir={out_dir}",
            flush=True,
        )
        if args.interval_seconds <= 0:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
