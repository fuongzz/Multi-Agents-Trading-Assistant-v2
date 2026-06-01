"""Wait for complete-enough VN100 daily bars before generating paper signals."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import date, datetime, time as day_time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
STATUS_PATH = ROOT / "data" / "runtime" / "combined_paper_eod_status.json"


def _parse_hhmm(raw: str) -> day_time:
    hour, minute = raw.split(":", 1)
    return day_time(int(hour), int(minute))


def _required_symbols() -> set[str]:
    from multiagents_trading_assistant.fetcher import get_vn100_symbols

    return {str(symbol).upper() for symbol in get_vn100_symbols()}


def eod_coverage(target: date, required_symbols: set[str], path: Path = MASTER) -> dict[str, Any]:
    if not path.exists():
        return {"available": 0, "required": len(required_symbols), "latest_date": None}
    frame = pd.read_parquet(path, columns=["date", "symbol"])
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    latest = frame["date"].max() if not frame.empty else None
    day_symbols = set(frame.loc[frame["date"] == target, "symbol"].astype(str).str.upper())
    return {
        "available": len(required_symbols.intersection(day_symbols)),
        "required": len(required_symbols),
        "latest_date": latest.isoformat() if latest else None,
    }


def _write_status(payload: dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for post-close VN100 data before screening.")
    parser.add_argument("--target-date", default="", help="YYYY-MM-DD; defaults to local date.")
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    parser.add_argument("--retry-minutes", type=int, default=10)
    parser.add_argument("--retry-until", default="18:30")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    target = date.fromisoformat(args.target_date) if args.target_date else datetime.now().date()
    required = _required_symbols()
    needed = math.ceil(len(required) * args.minimum_coverage)
    deadline = _parse_hhmm(args.retry_until)

    while True:
        if not args.check_only:
            result = subprocess.run([sys.executable, "scripts/update_ohlcv_daily.py"], cwd=ROOT, check=False)
            if result.returncode != 0:
                print(f"[combined_eod] update_ohlcv_daily failed code={result.returncode}", flush=True)
        coverage = eod_coverage(target, required)
        ready = coverage["available"] >= needed
        payload = {
            "checked_at": datetime.now().isoformat(timespec="seconds"),
            "target_date": target.isoformat(),
            "minimum_coverage": args.minimum_coverage,
            "minimum_symbols": needed,
            **coverage,
            "ready": ready,
        }
        _write_status(payload)
        print(
            f"[combined_eod] target={target} available={coverage['available']}/{coverage['required']} "
            f"needed={needed} latest={coverage['latest_date']} ready={ready}",
            flush=True,
        )
        if ready:
            return
        if args.check_only or datetime.now().time() >= deadline:
            raise SystemExit(2)
        time.sleep(max(60, args.retry_minutes * 60))


if __name__ == "__main__":
    main()
