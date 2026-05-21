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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiagents_trading_assistant.fetcher import get_live_price
from scripts.export_combined_paper_dashboard import export_dashboard


MORNING_START = day_time(9, 0)
MORNING_END = day_time(11, 30)
AFTERNOON_START = day_time(13, 0)
AFTERNOON_END = day_time(15, 0)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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
    if reference_price is None or reference_price <= 0:
        return float(live_price)
    price = float(live_price)
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


def refresh_once(out_dir: Path, *, html_refresh_seconds: int = 0) -> dict[str, Any]:
    open_path = out_dir / "paper_open_holdings.csv"
    if not open_path.exists():
        raise FileNotFoundError(f"Missing open holdings: {open_path}")
    holdings = _read_csv(open_path)
    base_account = _read_csv(out_dir / "paper_account_pnl.csv")
    if holdings.empty:
        status = {"priced_at": datetime.now().isoformat(timespec="seconds"), "updated_holdings": 0, "updated": 0, "symbols": 0, "html_refresh_seconds": html_refresh_seconds}
        (out_dir / "paper_realtime_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        export_dashboard(out_dir, out_dir / "mvp_p5", out_dir / "mvp_p4", out_dir / "flow_v2")
        return status

    holdings["symbol"] = holdings["symbol"].astype(str).str.upper().str.strip()
    symbols = holdings["symbol"].dropna().unique().tolist()
    raw_prices = get_live_price(symbols)
    priced_at = datetime.now().isoformat(timespec="seconds")

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
        "symbols": len(symbols),
        "html_refresh_seconds": html_refresh_seconds,
    }
    (out_dir / "paper_realtime_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    export_dashboard(out_dir, out_dir / "mvp_p5", out_dir / "mvp_p4", out_dir / "flow_v2")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh combined paper dashboard with realtime/snapshot prices.")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "combined_paper_trading_demo"))
    parser.add_argument("--interval-seconds", type=int, default=0)
    parser.add_argument("--html-refresh-seconds", type=int, default=0)
    parser.add_argument("--market-session-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    while True:
        if args.interval_seconds > 0 and args.market_session_only:
            in_session, label, sleep_seconds = _market_session_state()
            if not in_session:
                print(f"[combined_realtime] {datetime.now().isoformat(timespec='seconds')} paused={label} retry_in={sleep_seconds}s", flush=True)
                time.sleep(sleep_seconds)
                continue
        status = refresh_once(out_dir, html_refresh_seconds=args.html_refresh_seconds or args.interval_seconds)
        print(
            f"[combined_realtime] {status['priced_at']} updated_holdings={status['updated_holdings']} "
            f"symbols={status['symbols']} out_dir={out_dir}",
            flush=True,
        )
        if args.interval_seconds <= 0:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
