"""Validate local market data coverage before research/backtest runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OHLCV = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
DEFAULT_INDEX = ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"
DEFAULT_REPORT = ROOT / "backtest_results" / "data_quality" / "data_quality_report.json"


def _pct(value: float) -> float:
    return round(float(value) * 100.0, 4)


def validate_ohlcv(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "status": "FAIL", "path": str(path)}
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    duplicate_rows = int(df.duplicated(["date", "symbol"]).sum())
    unknown_industry = df["industry"].fillna("Unknown").astype(str).str.upper().isin({"UNKNOWN", "", "NAN", "NONE"})
    null_required = {
        col: int(df[col].isna().sum())
        for col in ["date", "symbol", "open", "high", "low", "close", "volume", "value", "exchange", "industry"]
        if col in df.columns
    }

    bad_ohlc = df[
        (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
        | (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    ]
    returns = df.sort_values(["symbol", "date"]).groupby("symbol", sort=False)["close"].pct_change()
    spike_rows = int((returns.abs() > 0.30).sum())
    symbol_dates = df.groupby("symbol")["date"].agg(["min", "max", "count"]).reset_index()
    stale_cutoff = df["date"].max() - pd.Timedelta(days=14)
    stale_symbols = symbol_dates.loc[symbol_dates["max"] < stale_cutoff, "symbol"].tolist()

    status = "PASS"
    issues: list[str] = []
    if duplicate_rows:
        status = "FAIL"
        issues.append(f"duplicate date-symbol rows={duplicate_rows}")
    if unknown_industry.any():
        status = "FAIL"
        issues.append(f"unknown industry rows={int(unknown_industry.sum())}")
    if bad_ohlc.shape[0]:
        status = "FAIL"
        issues.append(f"bad OHLC rows={int(bad_ohlc.shape[0])}")
    if stale_symbols:
        status = "WARN" if status == "PASS" else status
        issues.append(f"stale symbols>{len(stale_symbols)}")
    if spike_rows:
        status = "WARN" if status == "PASS" else status
        issues.append(f"abs daily close return >30% rows={spike_rows}")

    return {
        "exists": True,
        "status": status,
        "path": str(path),
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "duplicate_rows": duplicate_rows,
        "unknown_industry_rows": int(unknown_industry.sum()),
        "unknown_industry_pct": _pct(unknown_industry.mean()),
        "industry_groups": int(df["industry"].nunique(dropna=True)),
        "exchange_groups": sorted(df["exchange"].dropna().astype(str).unique().tolist()),
        "null_required": null_required,
        "bad_ohlc_rows": int(bad_ohlc.shape[0]),
        "large_close_return_rows": spike_rows,
        "stale_symbols": stale_symbols,
        "issues": issues,
    }


def validate_index(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "status": "FAIL", "path": str(path)}
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == "VNINDEX"]
    duplicate_rows = int(df.duplicated(["date"]).sum())
    null_close = int(df["close"].isna().sum())
    status = "PASS"
    issues: list[str] = []
    if df.empty:
        status = "FAIL"
        issues.append("VNINDEX frame is empty")
    if duplicate_rows:
        status = "FAIL"
        issues.append(f"duplicate VNINDEX dates={duplicate_rows}")
    if null_close:
        status = "FAIL"
        issues.append(f"null close rows={null_close}")
    return {
        "exists": True,
        "status": status,
        "path": str(path),
        "rows": int(len(df)),
        "date_min": str(df["date"].min().date()) if not df.empty else None,
        "date_max": str(df["date"].max().date()) if not df.empty else None,
        "duplicate_rows": duplicate_rows,
        "null_close_rows": null_close,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local OHLCV/index data quality")
    parser.add_argument("--ohlcv", type=Path, default=DEFAULT_OHLCV)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    report = {
        "ohlcv": validate_ohlcv(args.ohlcv),
        "index": validate_index(args.index),
    }
    report["status"] = "PASS" if all(item["status"] == "PASS" for item in report.values() if isinstance(item, dict)) else "WARN_OR_FAIL"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
