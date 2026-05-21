"""Refresh the demo open-holdings page with live or near-live prices.

The layered pipeline demo is a static HTML export. This updater re-marks the
open holdings to market and rewrites only the portfolio overview/open holdings
CSV + HTML files inside an existing demo directory.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, time as day_time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiagents_trading_assistant.fetcher import get_live_price
from scripts.export_pipeline_layer_demo import (
    BASELINE_BACKTEST_DIR,
    ROOT,
    _closed_trades_page,
    _open_holdings_page,
    _portfolio_overview_page,
)


DEFAULT_DEMO_DIR = ROOT / "reports" / "layered_pipeline_demo_2026-05-17"
MORNING_START = day_time(9, 0)
MORNING_END = day_time(11, 30)
AFTERNOON_START = day_time(13, 0)
AFTERNOON_END = day_time(15, 0)


def _as_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(str(value).replace(",", "").replace("%", "").strip())
    except Exception:
        return None


def _summary_value(summary: pd.DataFrame, metric: str) -> float | None:
    if summary.empty or "metric" not in summary.columns or "value" not in summary.columns:
        return None
    rows = summary[summary["metric"].astype(str) == metric]
    if rows.empty:
        return None
    return _as_float(rows.iloc[0]["value"])


def _normalize_live_price(live_price: float | None, reference_price: float | None) -> float | None:
    """Normalize broker live prices to the demo/backtest price unit.

    Historical VN OHLCV in this project is stored in thousand VND, while some
    live sources return raw VND. Use the position entry price as the unit anchor.
    """
    if live_price is None or live_price <= 0:
        return None
    if reference_price is None or reference_price <= 0:
        return float(live_price)

    price = float(live_price)
    ref = float(reference_price)
    if price / ref > 100.0:
        price = price / 1000.0
    elif ref / price > 100.0:
        price = price * 1000.0
    return price


def _set_summary_value(summary: pd.DataFrame, metric: str, value: Any) -> pd.DataFrame:
    mask = summary["metric"].astype(str) == metric
    if mask.any():
        summary.loc[mask, "value"] = value
        return summary
    return pd.concat([summary, pd.DataFrame([{"metric": metric, "value": value}])], ignore_index=True)


def _with_meta_refresh(html: str, seconds: int) -> str:
    if seconds <= 0 or "<head>" not in html:
        return html
    return html.replace("<head>", f'<head>\n  <meta http-equiv="refresh" content="{seconds}">', 1)


def _open_holdings_page_with_totals(frame: pd.DataFrame, html_refresh_seconds: int = 0) -> str:
    html = _open_holdings_page(frame)
    if frame.empty:
        return _with_meta_refresh(html, html_refresh_seconds)

    total_value = sum(_as_float(v) or 0.0 for v in frame.get("market_value", []))
    total_pnl = sum(_as_float(v) or 0.0 for v in frame.get("unrealized_pnl", []))
    entry_value = total_value - total_pnl
    total_pnl_pct = (total_pnl / entry_value * 100.0) if entry_value else 0.0
    klass = "win" if total_pnl > 0 else "loss" if total_pnl < 0 else "flat"
    priced_at = ""
    if "priced_at" in frame.columns:
        priced = frame["priced_at"].dropna().astype(str)
        priced_at = priced.iloc[0] if not priced.empty else ""

    totals_html = f"""
<div class="cards">
  <div class="metric"><span>Open holdings value</span><strong>{total_value:,.0f}</strong></div>
  <div class="metric"><span>Open unrealized PnL</span><strong class="{klass}">{total_pnl:,.0f}</strong></div>
  <div class="metric"><span>Open unrealized PnL %</span><strong class="{klass}">{total_pnl_pct:+.2f}%</strong></div>
  <div class="metric"><span>Price updated at</span><strong>{priced_at}</strong></div>
</div>
"""
    html = html.replace("<div class=\"wrap\"><table>", f"{totals_html}<div class=\"wrap\"><table>", 1)
    return _with_meta_refresh(html, html_refresh_seconds)


def _closed_trade_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty:
        return {
            "closed_trades": 0,
            "closed_realized_pnl": 0.0,
            "closed_win_rate": None,
            "closed_avg_pnl_pct": None,
            "closed_profit_factor": None,
            "closed_avg_holding_bars": None,
            "closed_winners": 0,
            "closed_losers": 0,
        }

    pnl = frame["pnl"].map(_as_float).fillna(0.0) if "pnl" in frame.columns else pd.Series(dtype=float)
    pnl_pct = frame["pnl_pct"].map(_as_float).dropna() if "pnl_pct" in frame.columns else pd.Series(dtype=float)
    holding = frame["holding_bars"].map(_as_float).dropna() if "holding_bars" in frame.columns else pd.Series(dtype=float)
    winners = int((pnl > 0).sum())
    losers = int((pnl < 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    trades = int(len(frame))
    return {
        "closed_trades": trades,
        "closed_realized_pnl": float(pnl.sum()),
        "closed_win_rate": winners / trades * 100.0 if trades else None,
        "closed_avg_pnl_pct": float(pnl_pct.mean()) if not pnl_pct.empty else None,
        "closed_profit_factor": gross_profit / gross_loss if gross_loss else None,
        "closed_avg_holding_bars": float(holding.mean()) if not holding.empty else None,
        "closed_winners": winners,
        "closed_losers": losers,
    }


def _closed_trades_page_with_totals(frame: pd.DataFrame, html_refresh_seconds: int = 0) -> str:
    html = _closed_trades_page(frame)
    metrics = _closed_trade_metrics(frame)
    pnl = float(metrics["closed_realized_pnl"] or 0.0)
    klass = "win" if pnl > 0 else "loss" if pnl < 0 else "flat"
    win_rate = metrics["closed_win_rate"]
    avg_pnl_pct = metrics["closed_avg_pnl_pct"]
    profit_factor = metrics["closed_profit_factor"]
    avg_holding = metrics["closed_avg_holding_bars"]
    totals_html = f"""
<div class="cards">
  <div class="metric"><span>Closed trades</span><strong>{metrics['closed_trades']}</strong></div>
  <div class="metric"><span>Closed realized PnL</span><strong class="{klass}">{pnl:,.0f}</strong></div>
  <div class="metric"><span>Closed win rate</span><strong>{win_rate:.2f}%</strong></div>
  <div class="metric"><span>Closed winners / losers</span><strong>{metrics['closed_winners']} / {metrics['closed_losers']}</strong></div>
  <div class="metric"><span>Average closed PnL %</span><strong class="{klass}">{avg_pnl_pct:+.2f}%</strong></div>
  <div class="metric"><span>Profit factor</span><strong>{profit_factor:.2f}</strong></div>
  <div class="metric"><span>Average holding bars</span><strong>{avg_holding:.1f}</strong></div>
</div>
"""
    html = html.replace("<div class=\"wrap\"><table>", f"{totals_html}<div class=\"wrap\"><table>", 1)
    return _with_meta_refresh(html, html_refresh_seconds)


def _seconds_until(target: datetime, *, minimum: int = 30, maximum: int = 900) -> int:
    seconds = int((target - datetime.now()).total_seconds())
    return max(minimum, min(maximum, seconds))


def _market_session_state(now: datetime | None = None) -> tuple[bool, str, int]:
    """Return whether VN continuous/ATC market session is active.

    The process intentionally stays alive outside session hours. During lunch,
    after close, weekends, or before open, callers should sleep and retry.
    """
    now = now or datetime.now()
    today = now.date()
    current = now.time()

    if now.weekday() >= 5:
        days_until_monday = 7 - now.weekday()
        next_open = datetime.combine(today + timedelta(days=days_until_monday), MORNING_START)
        return False, "weekend", _seconds_until(next_open)

    morning_open = datetime.combine(today, MORNING_START)
    morning_close = datetime.combine(today, MORNING_END)
    afternoon_open = datetime.combine(today, AFTERNOON_START)
    afternoon_close = datetime.combine(today, AFTERNOON_END)

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


def _refresh_once(demo_dir: Path, backtest_dir: Path, html_refresh_seconds: int = 0) -> dict[str, Any]:
    holdings_path = demo_dir / "06c_open_holdings.csv"
    summary_path = demo_dir / "06a_portfolio_overview.csv"
    closed_path = demo_dir / "06b_closed_trades.csv"
    if not holdings_path.exists():
        raise FileNotFoundError(f"Missing holdings file: {holdings_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing portfolio overview file: {summary_path}")

    holdings = pd.read_csv(holdings_path)
    summary = pd.read_csv(summary_path)
    closed = pd.read_csv(closed_path) if closed_path.exists() else pd.DataFrame()
    if holdings.empty:
        return {"updated": 0, "symbols": [], "demo_dir": str(demo_dir)}

    holdings["symbol"] = holdings["symbol"].astype(str).str.upper().str.strip()
    symbols = holdings["symbol"].dropna().unique().tolist()
    live_prices = get_live_price(symbols)
    priced_at = datetime.now().isoformat(timespec="seconds")

    updated = 0
    for idx, row in holdings.iterrows():
        symbol = str(row["symbol"]).upper()
        entry_price = _as_float(row.get("entry_price"))
        shares = _as_float(row.get("shares"))
        raw_live_price = live_prices.get(symbol)
        live_price = _normalize_live_price(raw_live_price, entry_price)
        fallback_price = _as_float(row.get("last_close"))
        price = live_price if live_price and live_price > 0 else fallback_price

        holdings.at[idx, "price_source"] = "live" if live_price and live_price > 0 else "snapshot"
        holdings.at[idx, "raw_live_price"] = round(float(raw_live_price), 4) if raw_live_price else None
        holdings.at[idx, "priced_at"] = priced_at if live_price and live_price > 0 else ""
        if price is None or entry_price is None or shares is None:
            continue

        market_value = shares * price
        unrealized_pnl = (price - entry_price) * shares
        unrealized_pnl_pct = (price / entry_price - 1.0) * 100.0 if entry_price else None

        holdings.at[idx, "last_close"] = round(price, 4)
        holdings.at[idx, "market_value"] = round(market_value, 0)
        holdings.at[idx, "unrealized_pnl"] = round(unrealized_pnl, 0)
        holdings.at[idx, "unrealized_pnl_pct"] = round(unrealized_pnl_pct, 2) if unrealized_pnl_pct is not None else None
        if live_price and live_price > 0:
            updated += 1

    cash = _summary_value(summary, "Cash") or 0.0
    current_nav = cash + sum(_as_float(v) or 0.0 for v in holdings.get("market_value", []))
    initial_nav = _summary_value(summary, "Initial NAV") or 100_000_000.0
    open_positions = len(holdings)
    max_positions = int(_summary_value(summary, "Max positions") or 5)

    summary = _set_summary_value(summary, "Current NAV", round(current_nav, 0))
    summary = _set_summary_value(summary, "Return", round((current_nav / initial_nav - 1.0) * 100.0, 2) if initial_nav else None)
    summary = _set_summary_value(summary, "Cash / NAV", round(cash / current_nav * 100.0, 2) if current_nav else None)
    summary = _set_summary_value(summary, "Open positions", open_positions)
    summary = _set_summary_value(summary, "Available slots", max(0, max_positions - open_positions))
    open_market_value = sum(_as_float(v) or 0.0 for v in holdings.get("market_value", []))
    open_unrealized_pnl = sum(_as_float(v) or 0.0 for v in holdings.get("unrealized_pnl", []))
    open_entry_value = open_market_value - open_unrealized_pnl
    closed_metrics = _closed_trade_metrics(closed)
    closed_realized_pnl = float(closed_metrics["closed_realized_pnl"] or 0.0)
    summary = _set_summary_value(summary, "Open holdings value", round(open_market_value, 0))
    summary = _set_summary_value(summary, "Open unrealized PnL", round(open_unrealized_pnl, 0))
    summary = _set_summary_value(
        summary,
        "Open unrealized PnL %",
        round(open_unrealized_pnl / open_entry_value * 100.0, 2) if open_entry_value else None,
    )
    summary = _set_summary_value(summary, "Closed trades", closed_metrics["closed_trades"])
    summary = _set_summary_value(summary, "Closed realized PnL", round(closed_realized_pnl, 0))
    summary = _set_summary_value(
        summary,
        "Closed win rate",
        round(float(closed_metrics["closed_win_rate"]), 2) if closed_metrics["closed_win_rate"] is not None else None,
    )
    summary = _set_summary_value(
        summary,
        "Closed average PnL %",
        round(float(closed_metrics["closed_avg_pnl_pct"]), 2) if closed_metrics["closed_avg_pnl_pct"] is not None else None,
    )
    summary = _set_summary_value(
        summary,
        "Closed profit factor",
        round(float(closed_metrics["closed_profit_factor"]), 2) if closed_metrics["closed_profit_factor"] is not None else None,
    )
    summary = _set_summary_value(
        summary,
        "Closed average holding bars",
        round(float(closed_metrics["closed_avg_holding_bars"]), 1) if closed_metrics["closed_avg_holding_bars"] is not None else None,
    )
    summary = _set_summary_value(summary, "Total realized + open PnL", round(closed_realized_pnl + open_unrealized_pnl, 0))
    summary = _set_summary_value(summary, "Price updated at", priced_at)
    summary = _set_summary_value(summary, "Live prices updated", updated)

    holdings = holdings.sort_values("market_value", ascending=False).reset_index(drop=True)
    holdings.to_csv(holdings_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    (demo_dir / "06c_open_holdings.html").write_text(
        _open_holdings_page_with_totals(holdings, html_refresh_seconds),
        encoding="utf-8",
    )
    (demo_dir / "06a_portfolio_overview.html").write_text(
        _with_meta_refresh(_portfolio_overview_page(summary, backtest_dir), html_refresh_seconds),
        encoding="utf-8",
    )
    if not closed.empty:
        (demo_dir / "06b_closed_trades.html").write_text(
            _closed_trades_page_with_totals(closed, html_refresh_seconds),
            encoding="utf-8",
        )
    try:
        from scripts.inject_sortable_tables import inject

        for page in ("06a_portfolio_overview.html", "06b_closed_trades.html", "06c_open_holdings.html"):
            path = demo_dir / page
            if path.exists():
                inject(path, demo_dir)
    except Exception as exc:
        print(f"[live-holdings] sortable table injection failed: {type(exc).__name__}: {exc}")
    return {"updated": updated, "symbols": symbols, "demo_dir": str(demo_dir), "priced_at": priced_at}


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh demo open holdings with live market prices.")
    parser.add_argument("--demo-dir", default=str(DEFAULT_DEMO_DIR))
    parser.add_argument("--backtest-dir", default=str(BASELINE_BACKTEST_DIR))
    parser.add_argument("--interval-seconds", type=int, default=0, help="Loop refresh interval. 0 runs once.")
    parser.add_argument(
        "--html-refresh-seconds",
        type=int,
        default=0,
        help="Add browser auto-refresh to rewritten HTML pages. 0 uses interval-seconds in loop mode.",
    )
    parser.add_argument(
        "--market-session-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In loop mode, update only during VN morning/afternoon sessions but keep the process alive.",
    )
    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    backtest_dir = Path(args.backtest_dir)

    try:
        while True:
            html_refresh_seconds = args.html_refresh_seconds
            if html_refresh_seconds <= 0 and args.interval_seconds > 0:
                html_refresh_seconds = args.interval_seconds

            if args.interval_seconds > 0 and args.market_session_only:
                in_session, session_label, session_sleep = _market_session_state()
                if not in_session:
                    print(
                        f"[open_holdings_live] {datetime.now().isoformat(timespec='seconds')} "
                        f"paused={session_label} retry_in={session_sleep}s"
                    )
                    time.sleep(session_sleep)
                    continue

            try:
                result = _refresh_once(demo_dir, backtest_dir, html_refresh_seconds=html_refresh_seconds)
                print(
                    f"[open_holdings_live] {result['priced_at']} updated={result['updated']}/"
                    f"{len(result['symbols'])} demo_dir={result['demo_dir']}"
                )
            except Exception as exc:
                print(
                    f"[open_holdings_live] {datetime.now().isoformat(timespec='seconds')} "
                    f"refresh_error={type(exc).__name__}: {exc}"
                )

            if args.interval_seconds <= 0:
                break
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        html_refresh_seconds = args.html_refresh_seconds
        print(f"[open_holdings_live] {datetime.now().isoformat(timespec='seconds')} stopped_by_user")


if __name__ == "__main__":
    main()
