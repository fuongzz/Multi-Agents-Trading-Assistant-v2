"""Repair OHLC rows where close/open sit outside the recorded high-low range."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
DEFAULT_REPORT = ROOT / "backtest_results" / "data_quality" / "ohlcv_bounds_repair_report.json"


def repair_bounds(store: Path, dry_run: bool = False, backup: bool = True) -> dict[str, object]:
    if not store.exists():
        raise FileNotFoundError(store)

    df = pd.read_parquet(store)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    price_cols = ["open", "high", "low", "close"]
    bad_mask = (
        (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
        | (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    )
    bad = df.loc[bad_mask, ["date", "symbol", *price_cols, "volume"]].copy()

    repaired = df.copy()
    repaired["high"] = repaired[price_cols].max(axis=1)
    repaired["low"] = repaired[price_cols].min(axis=1)
    repaired = repaired[["date", "symbol", "open", "high", "low", "close", "volume", "value", "exchange", "industry"]]
    repaired = repaired.drop_duplicates(["date", "symbol"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)

    remaining_bad = repaired[
        (repaired["open"] <= 0)
        | (repaired["high"] <= 0)
        | (repaired["low"] <= 0)
        | (repaired["close"] <= 0)
        | (repaired["high"] < repaired[["open", "close", "low"]].max(axis=1))
        | (repaired["low"] > repaired[["open", "close", "high"]].min(axis=1))
    ]

    report = {
        "store_path": str(store),
        "dry_run": dry_run,
        "rows": int(len(df)),
        "repaired_rows": int(len(bad)),
        "affected_symbols": sorted(bad["symbol"].astype(str).unique().tolist()),
        "remaining_bad_rows": int(len(remaining_bad)),
        "sample": [
            {
                **{k: (str(v.date()) if k == "date" else v) for k, v in row.items()},
            }
            for row in bad.head(30).to_dict("records")
        ],
    }

    if not dry_run and len(bad):
        if backup:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = store.with_name(f"{store.stem}.bounds_backup_{stamp}{store.suffix}")
            shutil.copy2(store, backup_path)
            report["backup_path"] = str(backup_path)
        repaired.to_parquet(store, index=False)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair OHLC high/low bounds in ohlcv_master.parquet")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    report = repair_bounds(args.store, dry_run=args.dry_run, backup=not args.no_backup)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["dry_run", "rows", "repaired_rows", "remaining_bad_rows"]}, ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")
    if "backup_path" in report:
        print(f"Backup: {report['backup_path']}")


if __name__ == "__main__":
    main()
