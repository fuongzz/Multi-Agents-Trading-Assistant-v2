"""Intraday realtime money-flow monitor for one Vietnamese stock.

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
from dataclasses import dataclass
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from multiagents_trading_assistant.fetcher import _get_dnse_client
from multiagents_trading_assistant.services.dnse_ws_price import (
    get_cache_stats,
    get_ws_tick,
    start_ws_price_feed,
)

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

        self.totals = FlowTotals()
        self.events: deque[dict] = deque()
        self.last_price: float | None = None
        self.last_tick_ts: float | None = None
        self.last_alert_ts: float = 0.0
        self.last_rest_minute: datetime | None = None
        self.last_rest_volume_by_minute: dict[datetime, int] = {}

        today = datetime.now(VN_TZ).strftime("%Y-%m-%d")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{today}_{self.symbol}_money_flow.csv"
        self._ensure_csv_header()

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
                    self._heartbeat(now, note="outside market hours")
                    if once:
                        return
                    time.sleep(max(self.poll_seconds, 30))
                    continue

                event = self._read_event(now)
                if event:
                    self._record_event(event)
                    self._print_status(event)
                    self._maybe_alert(event)
                else:
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

        return self._read_rest_1m_event(now)

    def _read_rest_1m_event(self, now: datetime) -> dict | None:
        client = _get_dnse_client()
        if client is None:
            return None

        end_ts = int(time.time())
        start_ts = end_ts - 15 * 60
        df = client.get_ohlcv(self.symbol, start_ts, end_ts, resolution="1")
        if df is None or df.empty:
            return None

        row = df.iloc[-1]
        minute = _normalize_minute(row["date"])
        price = float(row["close"])
        volume = int(float(row.get("volume", 0) or 0))
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


def _normalize_minute(value) -> datetime:
    ts = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=VN_TZ)
    return ts.astimezone(VN_TZ).replace(second=0, microsecond=0)


def _is_market_session(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    current = now.time()
    morning = dt_time(9, 0) <= current <= dt_time(11, 30)
    afternoon = dt_time(13, 0) <= current <= dt_time(15, 0)
    ato_atc_buffer = dt_time(8, 55) <= current <= dt_time(15, 5)
    return morning or afternoon or ato_atc_buffer


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime intraday money-flow monitor")
    parser.add_argument("symbol", help="Stock symbol, e.g. HPG")
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
    )
    monitor.run(once=args.once)


if __name__ == "__main__":
    main()
