"""Enrich OHLCV master metadata from vnstock_data Reference layer.

This fixes stale/empty ``exchange`` and ``industry`` fields without refetching
historical OHLCV. It is intended to be run with the sponsored home venv:

    C:\\Users\\NC\\.venv\\Scripts\\python.exe scripts\\enrich_ohlcv_metadata.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiagents_trading_assistant.data import ohlcv_store


def _normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def fetch_reference_maps(industry_level: int = 2) -> tuple[dict[str, str], dict[str, str], pd.DataFrame, pd.DataFrame]:
    """Fetch exchange and industry mappings using vnstock_data Unified UI."""
    try:
        from vnstock_data import Reference
    except ImportError as exc:
        raise RuntimeError("vnstock_data is required. Run with the sponsored home venv at C:\\Users\\NC\\.venv.") from exc

    ref = Reference()

    exchange_df = ref.equity.list_by_exchange()
    if not isinstance(exchange_df, pd.DataFrame) or exchange_df.empty:
        raise RuntimeError("Reference.equity.list_by_exchange() returned no data")
    exchange_df = exchange_df.copy()
    exchange_df["symbol"] = exchange_df["symbol"].map(_normalize_symbol)
    exchange_map = (
        exchange_df.dropna(subset=["symbol", "exchange"])
        .drop_duplicates("symbol", keep="first")
        .set_index("symbol")["exchange"]
        .astype(str)
        .to_dict()
    )

    industry_df = ref.equity.list_by_industry()
    if not isinstance(industry_df, pd.DataFrame) or industry_df.empty:
        raise RuntimeError("Reference.equity.list_by_industry() returned no data")
    industry_df = industry_df.copy()
    industry_df["symbol"] = industry_df["symbol"].map(_normalize_symbol)
    industry_level_df = industry_df[industry_df["icb_level"].astype("Int64") == int(industry_level)].copy()
    if industry_level_df.empty:
        raise RuntimeError(f"No ICB level {industry_level} industry rows returned")
    industry_map = (
        industry_level_df.dropna(subset=["symbol", "icb_name"])
        .drop_duplicates("symbol", keep="first")
        .set_index("symbol")["icb_name"]
        .astype(str)
        .to_dict()
    )

    return exchange_map, industry_map, exchange_df, industry_df


def enrich_master(
    store_path: Path,
    exchange_map: dict[str, str],
    industry_map: dict[str, str],
    backup: bool = True,
    dry_run: bool = False,
) -> dict[str, object]:
    if not store_path.exists():
        raise FileNotFoundError(f"OHLCV store not found: {store_path}")

    df = pd.read_parquet(store_path)
    if df.empty:
        raise RuntimeError(f"OHLCV store is empty: {store_path}")

    before = {
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "unknown_industry_rows": int((df["industry"].fillna("Unknown").astype(str).str.upper() == "UNKNOWN").sum()),
        "industry_unique": int(df["industry"].nunique(dropna=True)),
        "exchange_unique": sorted(df["exchange"].dropna().astype(str).unique().tolist()) if "exchange" in df else [],
    }

    enriched = df.copy()
    enriched["symbol"] = enriched["symbol"].map(_normalize_symbol)
    enriched["exchange"] = enriched["symbol"].map(exchange_map).fillna(enriched.get("exchange", "UNKNOWN")).fillna("UNKNOWN")
    enriched["industry"] = enriched["symbol"].map(industry_map).fillna(enriched.get("industry", "Unknown")).fillna("Unknown")
    enriched = enriched[["date", "symbol", "open", "high", "low", "close", "volume", "value", "exchange", "industry"]]
    enriched["date"] = pd.to_datetime(enriched["date"]).dt.tz_localize(None)
    enriched["volume"] = pd.to_numeric(enriched["volume"], errors="coerce").fillna(0).astype("int64")
    for col in ["open", "high", "low", "close", "value"]:
        enriched[col] = pd.to_numeric(enriched[col], errors="coerce").astype("float64")
    enriched = enriched.drop_duplicates(["date", "symbol"], keep="last").sort_values(["symbol", "date"]).reset_index(drop=True)

    after = {
        "rows": int(len(enriched)),
        "symbols": int(enriched["symbol"].nunique()),
        "unknown_industry_rows": int((enriched["industry"].fillna("Unknown").astype(str).str.upper() == "UNKNOWN").sum()),
        "industry_unique": int(enriched["industry"].nunique(dropna=True)),
        "exchange_unique": sorted(enriched["exchange"].dropna().astype(str).unique().tolist()),
        "mapped_symbols": int(enriched[enriched["industry"].fillna("Unknown").astype(str).str.upper() != "UNKNOWN"]["symbol"].nunique()),
    }

    report = {
        "store_path": str(store_path),
        "dry_run": dry_run,
        "before": before,
        "after": after,
        "missing_industry_symbols": sorted(
            set(enriched.loc[enriched["industry"].fillna("Unknown").astype(str).str.upper() == "UNKNOWN", "symbol"])
        ),
    }

    if not dry_run:
        if backup:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = store_path.with_name(f"{store_path.stem}.backup_{stamp}{store_path.suffix}")
            shutil.copy2(store_path, backup_path)
            report["backup_path"] = str(backup_path)
        ohlcv_store.append(enriched)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich OHLCV master exchange/industry metadata from vnstock_data Gold")
    parser.add_argument("--store", type=Path, default=ohlcv_store.STORE_PATH)
    parser.add_argument("--industry-level", type=int, default=2, help="ICB level to use as sector/industry label")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "backtest_results" / "data_quality" / "ohlcv_metadata_enrichment_report.json")
    args = parser.parse_args()

    exchange_map, industry_map, exchange_df, industry_df = fetch_reference_maps(args.industry_level)
    print(f"Reference exchange symbols: {len(exchange_map)}")
    print(f"Reference industry symbols level {args.industry_level}: {len(industry_map)}")

    report = enrich_master(
        args.store,
        exchange_map=exchange_map,
        industry_map=industry_map,
        backup=not args.no_backup,
        dry_run=args.dry_run,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ["dry_run", "before", "after"]}, ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")
    if "backup_path" in report:
        print(f"Backup: {report['backup_path']}")


if __name__ == "__main__":
    main()
