"""Intraday execution monitor for the live MVP trade family.

Research-only intraday money-flow signals are intentionally excluded here.
This monitor keeps the existing MVP family candidate logic and turns the
entry execution into an in-session watcher:

1. Build today's watchlist from the live edge family.
2. Run the normal trade pipeline once per watchlist symbol to get entry zones.
3. During market hours, watch live prices.
4. When price enters the pipeline's entry zone, re-check trader + risk with
   the live price and open the buy immediately if it still passes.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, time as dt_time
import os
import time
from zoneinfo import ZoneInfo

import requests

from multiagents_trading_assistant import database as db
from multiagents_trading_assistant.edge_lab.live_signal import (
    DEFAULT_LIVE_EDGE_FAMILY,
    get_edge_strategy_signals,
)
from multiagents_trading_assistant.fetcher import get_live_price, get_ohlcv, get_vn30_symbols, get_vn100_symbols
from multiagents_trading_assistant.nodes import risk_trade, trader_trade
from multiagents_trading_assistant.orchestrator.trade_graph import run_pipeline as run_trade_pipeline
from multiagents_trading_assistant.services.memory_service import (
    retrieve_knowledge,
    retrieve_trade_context,
    save_trade_decision,
)
from multiagents_trading_assistant.services.portfolio_service import get_total_nav_vnd

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


@dataclass
class WatchCandidate:
    symbol: str
    setup_type: str
    edge: dict
    state: dict
    entry_zone: tuple[float, float]


class IntradayTradeMVPMonitor:
    def __init__(
        self,
        symbols: list[str],
        *,
        poll_seconds: float,
        send_discord: bool,
        market_hours_only: bool,
    ) -> None:
        self.symbols = sorted({s.upper().strip() for s in symbols if s.strip()})
        self.poll_seconds = poll_seconds
        self.send_discord = send_discord
        self.market_hours_only = market_hours_only
        self.watchlist: dict[str, WatchCandidate] = {}
        self.watchlist_date: str | None = None
        self.last_heartbeat_minute: str | None = None

    def run(self, *, once: bool = False) -> None:
        self._refresh_watchlist(force=True)
        print(
            f"[intraday-mvp] watchlist={len(self.watchlist)} | "
            f"poll={self.poll_seconds:g}s | family={DEFAULT_LIVE_EDGE_FAMILY}"
        )
        print("[intraday-mvp] Press Ctrl+C to stop.\n")

        while True:
            try:
                now = datetime.now(VN_TZ)
                self._refresh_watchlist(force=False)

                if self.market_hours_only and not _is_market_session(now):
                    self._heartbeat(now, note="outside market hours")
                    if once:
                        return
                    time.sleep(max(self.poll_seconds, 30))
                    continue

                self._scan_live_entries(now)
                if once:
                    return
                time.sleep(self.poll_seconds)
            except KeyboardInterrupt:
                print("\n[intraday-mvp] Stopped by user.")
                return
            except Exception as exc:
                print(f"[intraday-mvp] error: {exc}")
                if once:
                    raise
                time.sleep(max(self.poll_seconds, 5))

    def _refresh_watchlist(self, *, force: bool) -> None:
        today = datetime.now(VN_TZ).strftime("%Y-%m-%d")
        if not force and self.watchlist_date == today and self.watchlist:
            return

        print(f"[intraday-mvp] rebuilding watchlist for {today} ...")
        edge_signals = get_edge_strategy_signals(
            self.symbols,
            as_of_date=today,
            strategy_name=DEFAULT_LIVE_EDGE_FAMILY,
        )
        passed = {sym: edge for sym, edge in edge_signals.items() if edge.get("passed")}
        passed_symbols = sorted(passed.keys())
        print(f"[intraday-mvp] edge pass: {len(passed_symbols)}/{len(self.symbols)}")

        live_prices = get_live_price(passed_symbols) if passed_symbols else {}
        watchlist: dict[str, WatchCandidate] = {}
        for symbol in passed_symbols:
            if db.has_position(symbol):
                continue
            edge = passed[symbol]
            live_price = live_prices.get(symbol)
            state = _build_candidate_state(symbol, today, edge, live_price)
            trader = state.get("trader_decision", {}) or {}
            entry_zone = _normalize_entry_zone(trader.get("entry_zone"))
            if not entry_zone:
                continue
            watchlist[symbol] = WatchCandidate(
                symbol=symbol,
                setup_type=str(state.get("setup_type") or "BREAKOUT"),
                edge=edge,
                state=state,
                entry_zone=entry_zone,
            )
        self.watchlist = watchlist
        self.watchlist_date = today
        print(f"[intraday-mvp] ready: {len(self.watchlist)} symbols with entry zone")

    def _scan_live_entries(self, now: datetime) -> None:
        if not self.watchlist:
            self._heartbeat(now, note="no active watchlist")
            return

        symbols = [sym for sym in self.watchlist if not db.has_position(sym)]
        if not symbols:
            self._heartbeat(now, note="all watchlist symbols already filled")
            return

        live_prices = get_live_price(symbols)
        triggered = 0
        for symbol in symbols:
            candidate = self.watchlist[symbol]
            price = live_prices.get(symbol)
            if price is None or price <= 0:
                continue
            if not _price_in_zone(float(price), candidate.entry_zone):
                continue
            triggered += 1
            result = _execute_candidate(candidate, float(price))
            print(result["message"])
            if self.send_discord:
                _send_discord(result["message"])
            if result.get("opened"):
                self.watchlist.pop(symbol, None)

        if triggered == 0:
            self._heartbeat(now, note=f"watching {len(symbols)} symbols")

    def _heartbeat(self, now: datetime, *, note: str) -> None:
        minute_key = now.strftime("%Y-%m-%d %H:%M")
        if self.last_heartbeat_minute == minute_key:
            return
        self.last_heartbeat_minute = minute_key
        print(f"[{now.strftime('%H:%M:%S')}] intraday-mvp {note}")


def _build_candidate_state(symbol: str, date: str, edge: dict, live_price: float | None) -> dict:
    market_context = _build_market_context(symbol, edge, live_price)
    setup_type = str(edge.get("setup_type") or "BREAKOUT")
    state = run_trade_pipeline(
        symbol=symbol,
        setup_type=setup_type,
        market_context=market_context,
        date=date,
        backtest_mode=False,
    )
    return state


def _execute_candidate(candidate: WatchCandidate, live_price: float) -> dict:
    symbol = candidate.symbol
    if db.has_position(symbol):
        return {"opened": False, "message": f"[intraday-mvp] {symbol} skip: already has position"}

    state = deepcopy(candidate.state)
    state["market_context"] = _build_market_context(symbol, candidate.edge, live_price)
    tech = deepcopy(state.get("technical_analysis") or {})
    tech["current_price"] = live_price
    snap = deepcopy(tech.get("indicator_snapshot") or {})
    snap["current_price"] = live_price
    tech["indicator_snapshot"] = snap
    state["technical_analysis"] = tech

    state["memory_context"] = {
        "internal": retrieve_trade_context(
            symbol,
            str(state.get("setup_type") or candidate.setup_type),
            tech.get("ma_trend", "UNKNOWN"),
            float(state.get("synthesis", {}).get("confluence_score") or 0.0),
            as_of_date=None,
            backtest_mode=False,
        ),
        "external": retrieve_knowledge(symbol, state.get("date") or datetime.now(VN_TZ).strftime("%Y-%m-%d")),
    }

    state["trader_decision"] = trader_trade.decide(state).get("trader_decision", {})
    state["risk_output"] = risk_trade.check(state)
    try:
        save_trade_decision(state)
    except Exception as exc:
        print(f"[intraday-mvp] decision save fail {symbol}: {exc}")

    trader = state.get("trader_decision", {}) or {}
    risk = state.get("risk_output", {}) or {}
    final_action = str(risk.get("final_action") or "")
    if final_action != "MUA":
        return {
            "opened": False,
            "message": (
                f"[intraday-mvp] {symbol} trigger hit but skip | final_action={final_action} "
                f"| reason={risk.get('override_reason') or trader.get('primary_reason', '')}"
            ),
        }

    nav_pct = risk.get("adjusted_position_pct")
    if nav_pct is None:
        nav_pct = float(trader.get("position_pct") or 0.0)
    nav_pct = round(float(nav_pct or 0.0) * float(risk.get("sizing_modifier") or 1.0), 2)
    if nav_pct <= 0:
        return {"opened": False, "message": f"[intraday-mvp] {symbol} skip: nav_pct=0"}

    qty = _estimate_quantity(live_price, nav_pct)
    if qty <= 0:
        return {"opened": False, "message": f"[intraday-mvp] {symbol} skip: qty=0"}

    today = datetime.now(VN_TZ).strftime("%Y-%m-%d")
    stop_loss = trader.get("stop_loss")
    target = trader.get("initial_target") or trader.get("take_profit")
    db.add_position(
        symbol=symbol,
        exchange=str(state.get("market_context", {}).get("exchange") or "HOSE"),
        entry_price=live_price,
        quantity=qty,
        entry_date=today,
        strategy=str(state.get("setup_type") or candidate.setup_type),
        sl=stop_loss,
        tp=target,
        nav_pct=nav_pct,
    )
    db.record_trade(
        symbol=symbol,
        action="MUA",
        price=live_price,
        quantity=qty,
        trade_date=today,
        strategy=str(state.get("setup_type") or candidate.setup_type),
        note="intraday_mvp_trigger_fill",
    )
    return {
        "opened": True,
        "message": (
            f"[AUTO-BUY][MVP] {symbol} @ {live_price:,.2f} | "
            f"zone={candidate.entry_zone[0]:,.2f}-{candidate.entry_zone[1]:,.2f} | "
            f"nav={nav_pct:.2f}% qty={qty:,} | "
            f"SL={stop_loss if stop_loss is not None else 'N/A'} "
            f"TP={target if target is not None else 'N/A'}"
        ),
    }


def _build_market_context(symbol: str, edge: dict, live_price: float | None) -> dict:
    avg_vol_20d = None
    day_change_pct = 0.0
    try:
        df = get_ohlcv(symbol, n_days=25)
        if df is not None and not df.empty:
            avg_vol_20d = float(df["volume"].tail(20).mean()) if "volume" in df.columns else None
            if live_price and len(df) >= 1:
                prev_close = float(df["close"].iloc[-1])
                if prev_close > 0:
                    day_change_pct = (float(live_price) - prev_close) / prev_close * 100.0
    except Exception as exc:
        print(f"[intraday-mvp] market context {symbol}: {exc}")

    return {
        "_portfolio_checked": True,
        "exchange": "HOSE",
        "stock_current_price": live_price,
        "stock_day_change_pct": round(day_change_pct, 2),
        "avg_vol_20d": avg_vol_20d,
        "reference_trend": "SIDEWAY",
        "trend": "SIDEWAY",
        "vni_change_pct": 0.0,
        "edge_strategy_analysis": edge,
        "active_strategies": [],
    }


def _normalize_entry_zone(value) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low = float(value[0])
    high = float(value[1])
    if low <= 0 or high <= 0 or low >= high:
        return None
    return (low, high)


def _price_in_zone(price: float, entry_zone: tuple[float, float]) -> bool:
    return entry_zone[0] <= price <= entry_zone[1]


def _estimate_quantity(price: float, nav_pct: float) -> int:
    total_nav = float(get_total_nav_vnd() or 0.0)
    if total_nav <= 0 or price <= 0 or nav_pct <= 0:
        return 0
    allocated = total_nav * nav_pct / 100.0
    return max(1, int(allocated / price))


def _send_discord(message: str) -> None:
    webhook = (
        os.getenv("DISCORD_WEBHOOK_TRADE")
        or os.getenv("DISCORD_WEBHOOK_URL")
        or ""
    )
    if not webhook:
        return
    try:
        requests.post(webhook, json={"content": message}, timeout=5)
    except Exception as exc:
        print(f"[intraday-mvp] Discord fail: {exc}")


def _is_market_session(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    current = now.time()
    morning = dt_time(9, 0) <= current <= dt_time(11, 30)
    afternoon = dt_time(13, 0) <= current <= dt_time(14, 45)
    return morning or afternoon


def _resolve_universe(universe: str) -> list[str]:
    if universe == "vn100":
        return get_vn100_symbols()
    if universe == "vn30":
        return get_vn30_symbols()
    raise SystemExit(f"Cannot resolve universe: {universe}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intraday execution monitor for MVP edge family")
    parser.add_argument("symbol", nargs="?", help="Stock symbol, e.g. HPG. Omit when using --universe.")
    parser.add_argument("--universe", choices=["vn30", "vn100"], help="Monitor a stock basket")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--discord", action="store_true", help="Send fills and skips to Discord webhook")
    parser.add_argument("--all-hours", action="store_true", help="Run outside Vietnam market hours")
    parser.add_argument("--once", action="store_true", help="Run one monitoring cycle")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.universe:
        symbols = _resolve_universe(args.universe)
    elif args.symbol:
        symbols = [args.symbol.upper().strip()]
    else:
        raise SystemExit("symbol is required unless --universe is provided")

    monitor = IntradayTradeMVPMonitor(
        symbols,
        poll_seconds=args.poll_seconds,
        send_discord=args.discord,
        market_hours_only=not args.all_hours,
    )
    monitor.run(once=args.once)


if __name__ == "__main__":
    main()
