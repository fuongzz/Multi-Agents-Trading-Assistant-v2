"""Intraday realtime money-flow monitor for Vietnamese stocks.

Primary source: DNSE WebSocket tick feed.
Fallback: DNSE REST 1-minute OHLCV when WebSocket has no fresh tick.

The monitor estimates intraday inflow from matched ticks:
  - Uptick value is treated as aggressive buy/inflow.
  - Downtick value is treated as aggressive sell/outflow.
  - Flat ticks are tracked as neutral value.

This is a practical tape-reading approximation, not exchange-classified
buy/sell data.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from multiagents_trading_assistant import database as db
from multiagents_trading_assistant.agents.trade import (
    flow_agent,
    sentiment_agent,
    synthesis_agent,
    technical_agent,
)
from multiagents_trading_assistant.data import get_data_provider
from multiagents_trading_assistant.fetcher import (
    _get_dnse_client,
    get_ohlcv,
    get_vn30_symbols,
    get_vn100_symbols,
)
from multiagents_trading_assistant.nodes import risk_trade, trader_trade
from multiagents_trading_assistant.services.memory_service import (
    retrieve_knowledge,
    retrieve_trade_context,
    save_trade_decision,
)
from multiagents_trading_assistant.services.dnse_ws_price import (
    get_cache_stats,
    get_ws_tick,
    start_ws_price_feed,
)
from multiagents_trading_assistant.services.portfolio_service import get_total_nav_vnd

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "data" / "realtime_money_flow"


@dataclass
class FlowTotals:
    inflow_value: float = 0.0
    outflow_value: float = 0.0
    neutral_value: float = 0.0
    matched_volume: int = 0
    tick_count: int = 0

    @property
    def net_value(self) -> float:
        return self.inflow_value - self.outflow_value

    @property
    def gross_value(self) -> float:
        return self.inflow_value + self.outflow_value + self.neutral_value


class IntradayMoneyFlowMonitor:
    def __init__(
        self,
        symbol: str,
        *,
        poll_seconds: float,
        window_minutes: int,
        flow_threshold_b: float,
        net_threshold_b: float,
        volume_threshold: int,
        cooldown_seconds: int,
        log_dir: Path,
        use_ws: bool,
        send_discord: bool,
        market_hours_only: bool,
        price_multiplier: float,
        allow_rest_fallback: bool = True,
        auto_buy: bool = False,
    ) -> None:
        self.symbol = symbol.upper().strip()
        self.poll_seconds = poll_seconds
        self.window_seconds = window_minutes * 60
        self.flow_threshold = flow_threshold_b * 1_000_000_000
        self.net_threshold = net_threshold_b * 1_000_000_000
        self.volume_threshold = volume_threshold
        self.cooldown_seconds = cooldown_seconds
        self.log_dir = log_dir
        self.use_ws = use_ws
        self.send_discord = send_discord
        self.market_hours_only = market_hours_only
        self.price_multiplier = price_multiplier
        self.allow_rest_fallback = allow_rest_fallback
        self.auto_buy = auto_buy

        self.totals = FlowTotals()
        self.events: deque[dict] = deque()
        self.last_price: float | None = None
        self.last_tick_ts: float | None = None
        self.last_alert_ts: float = 0.0
        self.last_rest_minute: datetime | None = None
        self.last_rest_volume_by_minute: dict[datetime, int] = {}
        self.last_close_check_date: str | None = None
        self.last_flow_signal: dict | None = None
        self.last_signal_alert_key: str | None = None
        self.last_execution_signal_key: str | None = None

        today = datetime.now(VN_TZ).strftime("%Y-%m-%d")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{today}_{self.symbol}_money_flow.csv"

    def run(self, *, once: bool = False) -> None:
        if self.use_ws:
            start_ws_price_feed([self.symbol])
            print(f"[monitor] DNSE WebSocket starting for {self.symbol}...")

        print(
            f"[monitor] {self.symbol} | poll={self.poll_seconds:g}s | "
            f"window={self.window_seconds // 60}m | "
            f"alert flow>={self.flow_threshold / 1e9:.2f}B, "
            f"net>={self.net_threshold / 1e9:.2f}B | log={self.csv_path}"
        )
        print("[monitor] Press Ctrl+C to stop.\n")

        while True:
            try:
                now = datetime.now(VN_TZ)
                if self.market_hours_only and not _is_market_session(now):
                    self._maybe_run_close_check(now)
                    self._heartbeat(now, note="outside market hours")
                    if once:
                        return
                    time.sleep(max(self.poll_seconds, 30))
                    continue

                event = self._read_event(now)
                if event:
                    self._record_event(event)
                    self._refresh_flow_signal()
                    self._print_status(event)
                    self._maybe_alert(event)
                    _maybe_alert_signal(self)
                    _maybe_execute_signal(self)
                else:
                    self._refresh_flow_signal()
                    self._heartbeat(now, note="waiting for fresh tick")

                if once:
                    return
                time.sleep(self.poll_seconds)
            except KeyboardInterrupt:
                print("\n[monitor] Stopped by user.")
                return
            except Exception as exc:
                print(f"[monitor] error: {exc}")
                if once:
                    raise
                time.sleep(max(self.poll_seconds, 5))

    def _read_event(self, now: datetime) -> dict | None:
        if self.use_ws:
            tick = get_ws_tick(self.symbol, max_age_seconds=max(30, self.poll_seconds * 3))
            if tick and tick.get("ts") != self.last_tick_ts:
                self.last_tick_ts = float(tick["ts"])
                price = float(tick["price"])
                volume = int(tick.get("volume") or 0)
                if volume > 0:
                    return self._build_event(now, price, volume, "ws_tick")

        if self.allow_rest_fallback:
            return self._read_rest_1m_event(now)
        return None

    def _maybe_run_close_check(self, now: datetime) -> None:
        if not _is_after_close(now):
            return
        today = now.strftime("%Y-%m-%d")
        if self.last_close_check_date == today:
            return
        snapshot = _fetch_gold_close_snapshot(self.symbol, self.price_multiplier)
        self.last_close_check_date = today
        if snapshot is None:
            print(f"[close-check] {self.symbol} no vnstock_data close snapshot for {today}")
            return
        print(
            f"[close-check] {self.symbol} {today} | "
            f"close={snapshot['close']:,.2f} high={snapshot['high']:,.2f} low={snapshot['low']:,.2f} "
            f"vol={snapshot['volume']:,} value={snapshot['value'] / 1e9:.3f}B source=vnstock_data"
        )

    def _refresh_flow_signal(self) -> None:
        snapshot = _fetch_rest_1m_snapshot(self.symbol)
        if snapshot is None:
            return
        self.last_flow_signal = _analyze_large_money_in(snapshot, self.price_multiplier)

    def _read_rest_1m_event(self, now: datetime) -> dict | None:
        snapshot = _fetch_rest_1m_snapshot(self.symbol)
        if snapshot is None:
            return None

        minute = snapshot["minute"]
        price = float(snapshot["close"])
        volume = int(snapshot["volume"])
        previous_volume = self.last_rest_volume_by_minute.get(minute, 0)
        delta_volume = max(0, volume - previous_volume)
        self.last_rest_volume_by_minute[minute] = volume

        # If this is the first sample of a minute, use the current candle volume.
        # On later polls inside the same minute, only use incremental volume.
        if self.last_rest_minute != minute:
            self.last_rest_minute = minute
            delta_volume = volume

        if delta_volume <= 0:
            return None
        return self._build_event(now, price, delta_volume, "dnse_1m")

    def _build_event(self, now: datetime, price: float, volume: int, source: str) -> dict:
        prev = self.last_price
        if prev is None:
            direction = "neutral"
            signed_value = 0.0
        elif price > prev:
            direction = "inflow"
            signed_value = price * volume
        elif price < prev:
            direction = "outflow"
            signed_value = -price * volume
        else:
            direction = "neutral"
            signed_value = 0.0

        self.last_price = price
        value = price * self.price_multiplier * volume
        return {
            "ts": now,
            "symbol": self.symbol,
            "price": price,
            "volume": volume,
            "value": value,
            "signed_value": signed_value * self.price_multiplier,
            "direction": direction,
            "source": source,
        }

    def _record_event(self, event: dict) -> None:
        value = float(event["value"])
        volume = int(event["volume"])
        direction = event["direction"]

        if direction == "inflow":
            self.totals.inflow_value += value
        elif direction == "outflow":
            self.totals.outflow_value += value
        else:
            self.totals.neutral_value += value

        self.totals.matched_volume += volume
        self.totals.tick_count += 1
        self.events.append(event)
        self._trim_window(event["ts"])
        self._append_csv(event)

    def _trim_window(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.window_seconds)
        while self.events and self.events[0]["ts"] < cutoff:
            self.events.popleft()

    def _window_stats(self) -> dict:
        inflow = sum(float(e["value"]) for e in self.events if e["direction"] == "inflow")
        outflow = sum(float(e["value"]) for e in self.events if e["direction"] == "outflow")
        neutral = sum(float(e["value"]) for e in self.events if e["direction"] == "neutral")
        volume = sum(int(e["volume"]) for e in self.events)
        return {
            "inflow": inflow,
            "outflow": outflow,
            "neutral": neutral,
            "net": inflow - outflow,
            "gross": inflow + outflow + neutral,
            "volume": volume,
            "ticks": len(self.events),
        }

    def _print_status(self, event: dict) -> None:
        w = self._window_stats()
        ts = event["ts"].strftime("%H:%M:%S")
        direction = {"inflow": "IN ", "outflow": "OUT", "neutral": "FLAT"}[event["direction"]]
        print(
            f"[{ts}] {self.symbol} {event['price']:,.2f} | {direction} "
            f"{event['value'] / 1e9:6.3f}B | "
            f"net{self.window_seconds // 60}m={w['net'] / 1e9:+.3f}B "
            f"gross={w['gross'] / 1e9:.3f}B vol={w['volume']:,} | "
            f"day_net={self.totals.net_value / 1e9:+.3f}B "
            f"src={event['source']}"
        )

    def _heartbeat(self, now: datetime, *, note: str) -> None:
        if int(time.time()) % 60 < self.poll_seconds:
            stats = get_cache_stats()
            print(
                f"[{now.strftime('%H:%M:%S')}] {self.symbol} {note}; "
                f"ws_connected={stats.get('connected')} cached={stats.get('symbols_cached')}"
            )

    def _maybe_alert(self, event: dict) -> None:
        w = self._window_stats()
        now_ts = time.time()
        if now_ts - self.last_alert_ts < self.cooldown_seconds:
            return

        reasons = []
        if w["gross"] >= self.flow_threshold and w["net"] >= self.net_threshold:
            reasons.append(
                f"strong inflow window: gross {w['gross'] / 1e9:.2f}B, net {w['net'] / 1e9:+.2f}B"
            )
        if w["volume"] >= self.volume_threshold and w["net"] > 0:
            reasons.append(f"volume burst: {w['volume']:,} shares/{self.window_seconds // 60}m")
        if event["direction"] == "inflow" and event["value"] >= self.net_threshold:
            reasons.append(f"large uptick print: {event['value'] / 1e9:.2f}B")

        if not reasons:
            return

        self.last_alert_ts = now_ts
        message = (
            f"[ALERT] {self.symbol} money inflow | price {event['price']:,.2f} | "
            f"{'; '.join(reasons)} | day net {self.totals.net_value / 1e9:+.2f}B"
        )
        print("\a" + message)
        if self.send_discord:
            _send_discord(message)

    def _ensure_csv_header(self) -> None:
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "symbol",
                    "price",
                    "volume",
                    "value",
                    "signed_value",
                    "direction",
                    "source",
                    "day_inflow_value",
                    "day_outflow_value",
                    "day_net_value",
                ]
            )

    def _append_csv(self, event: dict) -> None:
        self._ensure_csv_header()
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    event["ts"].isoformat(),
                    event["symbol"],
                    event["price"],
                    event["volume"],
                    round(event["value"], 2),
                    round(event["signed_value"], 2),
                    event["direction"],
                    event["source"],
                    round(self.totals.inflow_value, 2),
                    round(self.totals.outflow_value, 2),
                    round(self.totals.net_value, 2),
                ]
            )


class UniverseMoneyFlowMonitor:
    def __init__(
        self,
        symbols: list[str],
        *,
        poll_seconds: float,
        window_minutes: int,
        flow_threshold_b: float,
        net_threshold_b: float,
        volume_threshold: int,
        cooldown_seconds: int,
        log_dir: Path,
        send_discord: bool,
        market_hours_only: bool,
        price_multiplier: float,
        summary_seconds: int,
        top_n: int,
        auto_buy: bool,
    ) -> None:
        self.symbols = sorted({s.upper().strip() for s in symbols if s.strip()})
        self.poll_seconds = poll_seconds
        self.market_hours_only = market_hours_only
        self.summary_seconds = summary_seconds
        self.top_n = top_n
        self.auto_buy = auto_buy
        self.rest_scan_interval = max(60.0, poll_seconds)
        self.monitors = {
            symbol: IntradayMoneyFlowMonitor(
                symbol,
                poll_seconds=poll_seconds,
                window_minutes=window_minutes,
                flow_threshold_b=flow_threshold_b,
                net_threshold_b=net_threshold_b,
                volume_threshold=volume_threshold,
                cooldown_seconds=cooldown_seconds,
                log_dir=log_dir,
                use_ws=False,
                send_discord=send_discord,
                market_hours_only=market_hours_only,
                price_multiplier=price_multiplier,
                allow_rest_fallback=False,
                auto_buy=auto_buy,
            )
            for symbol in self.symbols
        }
        self.last_summary_ts = 0.0
        self.last_rest_scan_ts = 0.0
        self.ws_empty_since_ts: float | None = None
        self.last_close_check_date: str | None = None

    def run(self, *, once: bool = False) -> None:
        start_ws_price_feed(self.symbols)
        print(f"[monitor] DNSE WebSocket starting for {len(self.symbols)} symbols...")
        print(
            f"[monitor] Universe mode | symbols={len(self.symbols)} | "
            f"poll={self.poll_seconds:g}s | top_n={self.top_n}"
        )
        print("[monitor] Press Ctrl+C to stop.\n")

        while True:
            try:
                now = datetime.now(VN_TZ)
                if self.market_hours_only and not _is_market_session(now):
                    self._maybe_run_close_check(now)
                    self._heartbeat(now, note="outside market hours")
                    if once:
                        return
                    time.sleep(max(self.poll_seconds, 30))
                    continue

                event_count = self._consume_ws_events(now)
                if event_count == 0:
                    event_count = self._consume_rest_events(now)

                if event_count:
                    self.ws_empty_since_ts = None
                    self._print_summary(now, force=False)
                else:
                    if self.ws_empty_since_ts is None:
                        self.ws_empty_since_ts = time.time()
                    self._heartbeat(now, note="waiting for fresh ticks")

                if once:
                    self._print_summary(now, force=True)
                    return
                time.sleep(self.poll_seconds)
            except KeyboardInterrupt:
                print("\n[monitor] Stopped by user.")
                return
            except Exception as exc:
                print(f"[monitor] universe error: {exc}")
                if once:
                    raise
                time.sleep(max(self.poll_seconds, 5))

    def _consume_ws_events(self, now: datetime) -> int:
        event_count = 0
        for monitor in self.monitors.values():
            tick = get_ws_tick(monitor.symbol, max_age_seconds=max(30, self.poll_seconds * 3))
            if not tick or tick.get("ts") == monitor.last_tick_ts:
                continue
            monitor.last_tick_ts = float(tick["ts"])
            price = float(tick["price"])
            volume = int(tick.get("volume") or 0)
            if volume <= 0:
                continue
            event = monitor._build_event(now, price, volume, "ws_tick")
            monitor._record_event(event)
            monitor._maybe_alert(event)
            _maybe_alert_signal(monitor)
            event_count += 1
        return event_count

    def _consume_rest_events(self, now: datetime) -> int:
        stats = get_cache_stats()
        ws_silent_too_long = (
            stats.get("connected")
            and self.ws_empty_since_ts is not None
            and time.time() - self.ws_empty_since_ts >= 30
        )
        should_scan = (
            not stats.get("connected")
            or stats.get("symbols_cached", 0) == 0
            or ws_silent_too_long
            or self.auto_buy
        )
        if not should_scan:
            return 0
        if time.time() - self.last_rest_scan_ts < self.rest_scan_interval:
            return 0
        self.last_rest_scan_ts = time.time()
        if ws_silent_too_long or stats.get("symbols_cached", 0) == 0:
            print(f"[monitor] Universe fallback -> DNSE REST 1m scan ({len(self.symbols)} symbols)")

        event_count = 0
        with ThreadPoolExecutor(max_workers=12) as ex:
            futures = {ex.submit(_fetch_rest_1m_snapshot, sym): sym for sym in self.symbols}
            for fut in as_completed(futures):
                symbol = futures[fut]
                snapshot = fut.result()
                if snapshot is None:
                    continue
                monitor = self.monitors[symbol]
                monitor.last_flow_signal = _analyze_large_money_in(snapshot, monitor.price_multiplier)
                minute = snapshot["minute"]
                price = float(snapshot["close"])
                volume = int(snapshot["volume"])
                previous_volume = monitor.last_rest_volume_by_minute.get(minute, 0)
                delta_volume = max(0, volume - previous_volume)
                monitor.last_rest_volume_by_minute[minute] = volume
                if monitor.last_rest_minute != minute:
                    monitor.last_rest_minute = minute
                    delta_volume = volume
                if delta_volume <= 0:
                    continue
                event = monitor._build_event(now, price, delta_volume, "dnse_1m")
                monitor._record_event(event)
                monitor._maybe_alert(event)
                _maybe_alert_signal(monitor)
                _maybe_execute_signal(monitor)
                event_count += 1
        return event_count

    def _print_summary(self, now: datetime, *, force: bool) -> None:
        now_ts = time.time()
        if not force and now_ts - self.last_summary_ts < self.summary_seconds:
            return
        self.last_summary_ts = now_ts

        rows = []
        for symbol, monitor in self.monitors.items():
            w = monitor._window_stats()
            signal = monitor.last_flow_signal or {}
            if w["ticks"] <= 0 and not signal:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "net": w["net"],
                    "gross": w["gross"],
                    "volume": w["volume"],
                    "day_net": monitor.totals.net_value,
                    "price": monitor.last_price,
                    "signal": signal,
                    "score": float(signal.get("score", 0.0) or 0.0),
                    "large_money_in": bool(signal.get("large_money_in")),
                }
            )
        rows.sort(
            key=lambda r: (
                1 if r["large_money_in"] else 0,
                r["score"],
                r["net"],
                r["gross"],
            ),
            reverse=True,
        )

        stats = get_cache_stats()
        print(
            f"[{now.strftime('%H:%M:%S')}] VN100 real-inflow scan | "
            f"active={len(rows)}/{len(self.symbols)} ws_connected={stats.get('connected')} "
            f"cached={stats.get('symbols_cached')}"
        )
        for row in rows[: self.top_n]:
            price = row["price"]
            price_text = f"{price:,.2f}" if price is not None else "N/A"
            signal = row["signal"] or {}
            tag = "YES" if row["large_money_in"] else " . "
            print(
                f"  {row['symbol']:<5} LMI={tag} score={row['score']:>5.1f} "
                f"price={price_text:>9} "
                f"net15={signal.get('net_15m_value', 0.0) / 1e9:+7.3f}B "
                f"spike={signal.get('value_spike_5m', 0.0):>4.2f}x "
                f"chg15={signal.get('price_change_15m_pct', 0.0):+5.2f}% "
                f"closePos={signal.get('close_position_15m', 0.0):>4.2f}"
            )

    def _heartbeat(self, now: datetime, *, note: str) -> None:
        if int(time.time()) % 60 < self.poll_seconds:
            stats = get_cache_stats()
            print(
                f"[{now.strftime('%H:%M:%S')}] universe {note}; "
                f"ws_connected={stats.get('connected')} cached={stats.get('symbols_cached')}"
            )

    def _maybe_run_close_check(self, now: datetime) -> None:
        if not _is_after_close(now):
            return
        today = now.strftime("%Y-%m-%d")
        if self.last_close_check_date == today:
            return
        self.last_close_check_date = today
        print(f"[close-check] VN100 vnstock_data verification for {today}...")

        rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(_fetch_gold_close_snapshot, sym, self.monitors[sym].price_multiplier): sym for sym in self.symbols}
            for fut in as_completed(futures):
                row = fut.result()
                if row is not None:
                    rows.append(row)

        if not rows:
            print(f"[close-check] VN100 no vnstock_data close data for {today}")
            return

        rows.sort(key=lambda r: r["value"], reverse=True)
        out_path = self.monitors[self.symbols[0]].log_dir / f"{today}_vn100_close_check.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["symbol", "date", "open", "high", "low", "close", "volume", "value", "source"],
            )
            writer.writeheader()
            writer.writerows(rows)

        total_value = sum(float(r["value"]) for r in rows)
        print(
            f"[close-check] VN100 done | symbols={len(rows)} total_value={total_value / 1e12:.3f}T "
            f"saved={out_path}"
        )
        for row in rows[: self.top_n]:
            print(
                f"  {row['symbol']:<5} close={row['close']:>8,.2f} "
                f"vol={row['volume']:>10,} value={row['value'] / 1e9:>7.3f}B"
            )

def _normalize_minute(value) -> datetime:
    ts = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=VN_TZ)
    return ts.astimezone(VN_TZ).replace(second=0, microsecond=0)


def _fetch_rest_1m_snapshot(symbol: str) -> dict | None:
    client = _get_dnse_client()
    if client is None:
        return None
    now = datetime.now(VN_TZ)
    end_ts = int(now.timestamp())
    start_ts = int((now - timedelta(hours=6)).timestamp())
    raw = client.get_ohlcv_raw(symbol.upper(), "1", start_ts, end_ts, asset_type="STOCK")
    if not raw or not raw.get("t"):
        return None
    timestamps = raw.get("t") or []
    closes = raw.get("c") or []
    volumes = raw.get("v") or []
    if not timestamps or not closes or not volumes:
        return None
    ts = datetime.fromtimestamp(int(timestamps[-1]), tz=VN_TZ).replace(second=0, microsecond=0)
    bars = []
    highs = raw.get("h") or []
    lows = raw.get("l") or []
    opens = raw.get("o") or []
    for ts_raw, open_, high_, low_, close_, volume_ in zip(timestamps, opens, highs, lows, closes, volumes):
        bars.append(
            {
                "minute": datetime.fromtimestamp(int(ts_raw), tz=VN_TZ).replace(second=0, microsecond=0),
                "open": float(open_),
                "high": float(high_),
                "low": float(low_),
                "close": float(close_),
                "volume": int(volume_),
            }
        )
    return {
        "symbol": symbol.upper(),
        "minute": ts,
        "close": float(closes[-1]),
        "volume": int(volumes[-1]),
        "bars": bars,
    }


def _analyze_large_money_in(snapshot: dict, price_multiplier: float) -> dict:
    bars = list(snapshot.get("bars") or [])
    if len(bars) < 10:
        return {
            "score": 0.0,
            "large_money_in": False,
            "reason": "not_enough_bars",
        }

    values = [float(bar["close"]) * int(bar["volume"]) * price_multiplier for bar in bars]
    last_15 = bars[-15:]
    last_5 = bars[-5:]
    prev_20_values = values[-25:-5] if len(values) >= 25 else values[:-5]
    last_5_values = values[-5:]

    prev_close = None
    up_value = 0.0
    down_value = 0.0
    flat_value = 0.0
    green_minutes = 0
    for bar in last_15:
        close = float(bar["close"])
        value = close * int(bar["volume"]) * price_multiplier
        if prev_close is not None:
            if close > prev_close:
                up_value += value
                green_minutes += 1
            elif close < prev_close:
                down_value += value
            else:
                flat_value += value
        prev_close = close

    total_directional = up_value + down_value + flat_value
    net_15m = up_value - down_value
    first_close = float(last_15[0]["close"])
    last_close = float(last_15[-1]["close"])
    price_change_15m_pct = ((last_close - first_close) / first_close * 100.0) if first_close > 0 else 0.0
    high_15m = max(float(bar["high"]) for bar in last_15)
    low_15m = min(float(bar["low"]) for bar in last_15)
    range_15m = max(high_15m - low_15m, 1e-9)
    close_position_15m = (last_close - low_15m) / range_15m
    total_15m_value = sum(float(bar["close"]) * int(bar["volume"]) * price_multiplier for bar in last_15)
    total_5m_value = sum(last_5_values)
    prev_20_avg_value = (sum(prev_20_values) / len(prev_20_values)) if prev_20_values else 0.0
    last_5_avg_value = (sum(last_5_values) / len(last_5_values)) if last_5_values else 0.0
    value_spike_5m = (last_5_avg_value / prev_20_avg_value) if prev_20_avg_value > 0 else 0.0
    green_ratio_15m = green_minutes / max(len(last_15) - 1, 1)

    score = 0.0
    if total_directional > 0:
        score += max(0.0, min(35.0, 35.0 * (net_15m / total_directional)))
    if price_change_15m_pct > 0:
        score += min(20.0, price_change_15m_pct * 10.0)
    if close_position_15m >= 0.8:
        score += 18.0
    elif close_position_15m >= 0.65:
        score += 10.0
    if value_spike_5m >= 1.2:
        score += min(17.0, (value_spike_5m - 1.0) * 20.0)
    if green_ratio_15m >= 0.55:
        score += min(10.0, green_ratio_15m * 15.0)

    if price_change_15m_pct < 0:
        score -= 20.0
    if close_position_15m < 0.45:
        score -= 15.0
    if down_value > up_value:
        score -= 15.0

    score = max(0.0, min(100.0, score))
    large_money_in = bool(
        score >= 65.0
        and net_15m >= 2_000_000_000
        and total_5m_value >= 3_000_000_000
        and price_change_15m_pct > 0.2
        and close_position_15m >= 0.65
        and value_spike_5m >= 1.2
    )

    return {
        "score": round(score, 1),
        "large_money_in": large_money_in,
        "net_15m_value": net_15m,
        "up_15m_value": up_value,
        "down_15m_value": down_value,
        "total_15m_value": total_15m_value,
        "total_5m_value": total_5m_value,
        "value_spike_5m": round(value_spike_5m, 2),
        "price_change_15m_pct": round(price_change_15m_pct, 2),
        "close_position_15m": round(close_position_15m, 2),
        "green_ratio_15m": round(green_ratio_15m, 2),
        "signal_key": f"{snapshot['symbol']}-{snapshot['minute'].strftime('%Y%m%d%H%M')}",
    }


def _is_market_session(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    current = now.time()
    morning = dt_time(9, 0) <= current <= dt_time(11, 30)
    afternoon = dt_time(13, 0) <= current <= dt_time(15, 0)
    ato_atc_buffer = dt_time(8, 55) <= current <= dt_time(15, 5)
    return morning or afternoon or ato_atc_buffer


def _is_after_close(now: datetime) -> bool:
    return now.weekday() < 5 and now.time() >= dt_time(15, 5)


def _send_discord(message: str) -> None:
    webhook = (
        os.getenv("DISCORD_WEBHOOK_TRADE")
        or os.getenv("DISCORD_WEBHOOK_URL")
        or os.getenv("DISCORD_WEBHOOK_MONEY_FLOW")
        or ""
    )
    if not webhook:
        return
    try:
        requests.post(webhook, json={"content": message}, timeout=5)
    except Exception as exc:
        print(f"[monitor] Discord send failed: {exc}")


def _maybe_alert_signal(monitor: IntradayMoneyFlowMonitor) -> None:
    signal = monitor.last_flow_signal or {}
    if not signal.get("large_money_in"):
        return
    alert_key = str(signal.get("signal_key") or "")
    if not alert_key or monitor.last_signal_alert_key == alert_key:
        return
    now_ts = time.time()
    if now_ts - monitor.last_alert_ts < monitor.cooldown_seconds:
        return
    monitor.last_alert_ts = now_ts
    monitor.last_signal_alert_key = alert_key
    message = (
        f"[ALERT] {monitor.symbol} large money in | "
        f"score {signal.get('score', 0.0):.1f} | "
        f"net15 {signal.get('net_15m_value', 0.0) / 1e9:+.2f}B | "
        f"spike {signal.get('value_spike_5m', 0.0):.2f}x | "
        f"chg15 {signal.get('price_change_15m_pct', 0.0):+.2f}% | "
        f"closePos {signal.get('close_position_15m', 0.0):.2f}"
    )
    print("\a" + message)
    if monitor.send_discord:
        _send_discord(message)


def _maybe_execute_signal(monitor: IntradayMoneyFlowMonitor) -> None:
    if not monitor.auto_buy:
        return
    signal = monitor.last_flow_signal or {}
    if not signal.get("large_money_in"):
        return
    signal_key = str(signal.get("signal_key") or "")
    if not signal_key or monitor.last_execution_signal_key == signal_key:
        return
    monitor.last_execution_signal_key = signal_key

    if db.has_position(monitor.symbol):
        print(f"[auto-buy] {monitor.symbol} skip: already has open position")
        return
    if monitor.last_price is None or monitor.last_price <= 0:
        print(f"[auto-buy] {monitor.symbol} skip: no trigger price")
        return

    result = _run_intraday_entry_pipeline(
        monitor.symbol,
        trigger_price=float(monitor.last_price),
        signal=signal,
    )
    message = result.get("message") or f"[auto-buy] {monitor.symbol} no action"
    print(message)
    if monitor.send_discord:
        _send_discord(message)


def _run_intraday_entry_pipeline(symbol: str, *, trigger_price: float, signal: dict) -> dict:
    symbol = symbol.upper().strip()
    date = datetime.now(VN_TZ).strftime("%Y-%m-%d")
    setup_type = "MOMENTUM_SURGE"

    market_context = _build_intraday_market_context(symbol, trigger_price, signal)
    technical = technical_agent.analyze(symbol, date)
    technical["current_price"] = trigger_price
    indicator_snapshot = technical.get("indicator_snapshot") or {}
    indicator_snapshot["current_price"] = trigger_price
    technical["indicator_snapshot"] = indicator_snapshot

    flow = flow_agent.analyze(symbol, date)
    sentiment = sentiment_agent.analyze(symbol, date)
    money_flow = _build_intraday_money_flow(signal)
    synthesis = synthesis_agent.run(
        technical_analysis=technical,
        foreign_flow_analysis=flow,
        sentiment_analysis=sentiment,
        setup_type=setup_type,
        money_flow_analysis=money_flow,
        market_context=market_context,
    )
    memory_context = {
        "internal": retrieve_trade_context(
            symbol,
            setup_type,
            technical.get("ma_trend", "UNKNOWN"),
            float(synthesis.get("confluence_score") or 0.0),
            as_of_date=None,
            backtest_mode=False,
        ),
        "external": retrieve_knowledge(symbol, date),
    }

    state = {
        "symbol": symbol,
        "date": date,
        "setup_type": setup_type,
        "market_context": market_context,
        "macro_context": {},
        "technical_analysis": technical,
        "foreign_flow_analysis": flow,
        "sentiment_analysis": sentiment,
        "money_flow_analysis": money_flow,
        "synthesis": synthesis,
        "memory_context": memory_context,
        "backtest_mode": False,
    }
    state.update(trader_trade.decide(state))
    state["risk_output"] = risk_trade.check(state)

    try:
        save_trade_decision(state)
    except Exception as exc:
        print(f"[auto-buy] decision save fail {symbol}: {exc}")

    final_action = state.get("risk_output", {}).get("final_action")
    if final_action != "MUA":
        return {
            "opened": False,
            "message": (
                f"[auto-buy] {symbol} skip | final_action={final_action} "
                f"| conf={state.get('synthesis', {}).get('confluence_score', 0)} "
                f"| reason={state.get('risk_output', {}).get('override_reason') or state.get('trader_decision', {}).get('primary_reason', '')}"
            ),
        }

    trader = state.get("trader_decision", {}) or {}
    risk = state.get("risk_output", {}) or {}
    base_nav_pct = risk.get("adjusted_position_pct")
    if base_nav_pct is None:
        base_nav_pct = float(trader.get("position_pct") or 0.0)
    effective_nav_pct = round(float(base_nav_pct or 0.0) * float(risk.get("sizing_modifier") or 1.0), 2)
    if effective_nav_pct <= 0:
        return {"opened": False, "message": f"[auto-buy] {symbol} skip | effective_nav_pct=0"}

    quantity = _estimate_quantity(trigger_price, effective_nav_pct)
    stop_loss = trader.get("stop_loss")
    initial_target = trader.get("initial_target") or trader.get("take_profit")
    exchange = str(market_context.get("exchange") or "HOSE")

    db.add_position(
        symbol=symbol,
        exchange=exchange,
        entry_price=trigger_price,
        quantity=quantity,
        entry_date=date,
        strategy=setup_type,
        sl=stop_loss,
        tp=initial_target,
        nav_pct=effective_nav_pct,
    )
    db.record_trade(
        symbol=symbol,
        action="MUA",
        price=trigger_price,
        quantity=quantity,
        trade_date=date,
        strategy=setup_type,
        note=f"intraday auto-buy {signal.get('signal_key')}",
    )
    return {
        "opened": True,
        "message": (
            f"[AUTO-BUY] {symbol} OPENED @ {trigger_price:,.2f} | "
            f"nav={effective_nav_pct:.2f}% qty={quantity:,} "
            f"SL={stop_loss if stop_loss is not None else 'N/A'} "
            f"TP={initial_target if initial_target is not None else 'N/A'} "
            f"score={signal.get('score', 0.0):.1f}"
        ),
    }


def _build_intraday_money_flow(signal: dict) -> dict:
    score = float(signal.get("score") or 0.0)
    regime = "EARLY_MONEY_IN" if score >= 75.0 else "MONEY_IN"
    if score >= 85.0:
        regime = "BREAKOUT_FLOW"
    return {
        "engine": "intraday_large_money_in_v1",
        "regime": regime,
        "action_bias": "BUY_CANDIDATE",
        "score": round(max(0.0, min(10.0, (score - 50.0) / 5.0)), 2),
        "confidence": "HIGH" if score >= 75.0 else "MEDIUM",
        "signals": {
            "liquidity_ok": True,
            "intraday_large_money_in": True,
        },
        "metrics": {
            "net_15m_value": signal.get("net_15m_value", 0.0),
            "value_spike_5m": signal.get("value_spike_5m", 0.0),
            "price_change_15m_pct": signal.get("price_change_15m_pct", 0.0),
            "close_position_15m": signal.get("close_position_15m", 0.0),
        },
        "reasons": [
            f"net15={signal.get('net_15m_value', 0.0) / 1e9:+.2f}B",
            f"spike5m={signal.get('value_spike_5m', 0.0):.2f}x",
            f"chg15={signal.get('price_change_15m_pct', 0.0):+.2f}%",
            f"closePos={signal.get('close_position_15m', 0.0):.2f}",
        ],
        "blockers": [],
        "data_quality": {
            "ohlcv_days": 0,
            "has_foreign_flow": False,
            "has_intraday_flow": True,
            "warnings": [],
        },
    }


def _build_intraday_market_context(symbol: str, trigger_price: float, signal: dict) -> dict:
    prev_close = None
    avg_vol_20d = None
    try:
        df = get_ohlcv(symbol, n_days=25)
        if df is not None and not df.empty:
            closes = df["close"].tolist()
            if closes:
                prev_close = float(closes[-2] if len(closes) >= 2 else closes[-1])
            avg_vol_20d = float(df["volume"].tail(20).mean()) if "volume" in df.columns else None
    except Exception as exc:
        print(f"[auto-buy] market context fallback {symbol}: {exc}")
    day_change_pct = ((trigger_price - prev_close) / prev_close * 100.0) if prev_close and prev_close > 0 else 0.0
    return {
        "exchange": "HOSE",
        "stock_current_price": trigger_price,
        "stock_day_change_pct": round(day_change_pct, 2),
        "avg_vol_20d": avg_vol_20d,
        "reference_trend": "SIDEWAY",
        "trend": "SIDEWAY",
        "vni_change_pct": 0.0,
        "edge_strategy_analysis": {
            "strategy_name": "intraday_large_money_in_v1",
            "strategy_family": "intraday_large_money_in_v1",
            "passed": True,
            "edge_score": round(float(signal.get("score") or 0.0), 1),
            "feature_date": datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M"),
            "filters_failed": [],
        },
    }


def _estimate_quantity(trigger_price: float, nav_pct: float) -> int:
    total_nav = float(get_total_nav_vnd() or 0.0)
    if total_nav <= 0 or trigger_price <= 0 or nav_pct <= 0:
        return 0
    allocated = total_nav * nav_pct / 100.0
    return max(1, int(allocated / trigger_price))


def _fetch_gold_close_snapshot(symbol: str, price_multiplier: float) -> dict | None:
    provider = get_data_provider()
    today = datetime.now(VN_TZ).strftime("%Y-%m-%d")
    try:
        df = provider.get_ohlcv(symbol.upper(), today, today, interval="1D")
    except Exception:
        return None
    if df is None or df.empty:
        return None
    row = df.iloc[-1]
    close = float(row["close"])
    volume = int(float(row["volume"]))
    return {
        "symbol": symbol.upper(),
        "date": today,
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": close,
        "volume": volume,
        "value": close * price_multiplier * volume,
        "source": "vnstock_data",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime intraday money-flow monitor")
    parser.add_argument("symbol", nargs="?", help="Stock symbol, e.g. HPG. Omit when using --universe.")
    parser.add_argument("--universe", choices=["vn30", "vn100"], help="Monitor a stock basket instead of one symbol")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--window-minutes", type=int, default=5)
    parser.add_argument("--flow-threshold-b", type=float, default=1.0, help="Gross window value threshold in billion VND")
    parser.add_argument("--net-threshold-b", type=float, default=0.5, help="Net inflow threshold in billion VND")
    parser.add_argument("--volume-threshold", type=int, default=300_000, help="Window volume alert threshold")
    parser.add_argument("--cooldown-seconds", type=int, default=300)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--no-ws", action="store_true", help="Disable WebSocket and use DNSE REST 1m only")
    parser.add_argument("--discord", action="store_true", help="Send alerts to Discord webhook from .env")
    parser.add_argument("--all-hours", action="store_true", help="Run outside Vietnam market hours")
    parser.add_argument("--summary-seconds", type=int, default=30, help="Universe-mode summary interval")
    parser.add_argument("--top-n", type=int, default=10, help="Universe-mode top inflow rows to print")
    parser.add_argument("--auto-buy", action="store_true", help="Open buy immediately when intraday pipeline passes")
    parser.add_argument(
        "--price-multiplier",
        type=float,
        default=1000.0,
        help="Multiply quoted price to VND. Vietnamese stock APIs usually quote in thousand VND.",
    )
    parser.add_argument("--once", action="store_true", help="Run one polling cycle for smoke testing")
    return parser


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _build_parser().parse_args(argv)
    if args.universe:
        symbols = _resolve_universe(args.universe)
        monitor_many = UniverseMoneyFlowMonitor(
            symbols,
            poll_seconds=args.poll_seconds,
            window_minutes=args.window_minutes,
            flow_threshold_b=args.flow_threshold_b,
            net_threshold_b=args.net_threshold_b,
            volume_threshold=args.volume_threshold,
            cooldown_seconds=args.cooldown_seconds,
            log_dir=args.log_dir,
            send_discord=args.discord,
            market_hours_only=not args.all_hours,
            price_multiplier=args.price_multiplier,
            summary_seconds=args.summary_seconds,
            top_n=args.top_n,
            auto_buy=args.auto_buy,
        )
        monitor_many.run(once=args.once)
        return

    if not args.symbol:
        raise SystemExit("symbol is required unless --universe is provided")

    monitor = IntradayMoneyFlowMonitor(
        args.symbol,
        poll_seconds=args.poll_seconds,
        window_minutes=args.window_minutes,
        flow_threshold_b=args.flow_threshold_b,
        net_threshold_b=args.net_threshold_b,
        volume_threshold=args.volume_threshold,
        cooldown_seconds=args.cooldown_seconds,
        log_dir=args.log_dir,
        use_ws=not args.no_ws,
        send_discord=args.discord,
        market_hours_only=not args.all_hours,
        price_multiplier=args.price_multiplier,
        auto_buy=args.auto_buy,
    )
    monitor.run(once=args.once)


def _resolve_universe(universe: str) -> list[str]:
    if universe == "vn100":
        symbols = get_vn100_symbols()
    elif universe == "vn30":
        symbols = get_vn30_symbols()
    else:
        symbols = []
    if not symbols:
        raise SystemExit(f"Cannot resolve universe: {universe}")
    return symbols


if __name__ == "__main__":
    main()
