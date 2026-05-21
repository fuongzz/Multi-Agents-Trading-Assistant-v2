"""
One-time backfill script: fetch full historical OHLCV for VN30 + VN100 (2020 onwards).
Writes to ohlcv_master.parquet with full schema (date, symbol, OHLCV, value, exchange, industry).

Usage:
    python scripts/build_ohlcv_history.py --start 2020-01-01 --workers 10
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import pandas as pd

# Add parent to path to import from multiagents_trading_assistant
sys.path.insert(0, str(Path(__file__).parent.parent))

from multiagents_trading_assistant.fetcher import (
    get_vn30_symbols,
    get_vn100_symbols,
    get_ohlcv_history,
)
from multiagents_trading_assistant.data import ohlcv_store


def get_exchange_map() -> dict[str, str]:
    """Fetch exchange mapping: {symbol: 'HOSE' | 'HNX'}"""
    try:
        from vnstock_data import Reference

        ref_df = Reference().equity.list_by_exchange()
        # Columns: symbol, exchange, ...
        ref_df["symbol"] = ref_df["symbol"].astype(str).str.upper().str.strip()
        exchange_map = dict(zip(ref_df["symbol"], ref_df["exchange"]))
        return exchange_map
    except Exception as e:
        print(f"⚠️  Failed to fetch exchange map: {e}")
        return {}


def get_industry_map() -> dict[str, str]:
    """Fetch industry mapping: {symbol: icb_name (L2)}"""
    try:
        from vnstock_data import Reference

        industry_df = Reference().equity.list_by_industry()
        industry_df = industry_df[industry_df["icb_level"].astype("Int64") == 2].copy()
        industry_df["symbol"] = industry_df["symbol"].astype(str).str.upper().str.strip()
        return (
            industry_df.dropna(subset=["symbol", "icb_name"])
            .drop_duplicates("symbol", keep="first")
            .set_index("symbol")["icb_name"]
            .astype(str)
            .to_dict()
        )
    except Exception as e:
        print(f"⚠️  Failed to fetch industry map: {e}")
        return {}


def enrich_df(
    df: pd.DataFrame,
    symbol: str,
    exchange_map: dict,
    industry_map: dict,
) -> pd.DataFrame:
    """
    Enrich OHLCV DataFrame with exchange, industry, and value.

    Args:
        df: Raw OHLCV from get_ohlcv_history() with columns [date, symbol, open, high, low, close, volume]
        symbol: Symbol being processed
        exchange_map: {symbol: exchange}
        industry_map: {symbol: industry}

    Returns:
        DataFrame with columns: date, symbol, open, high, low, close, volume, value, exchange, industry
    """
    if df.empty:
        return df

    symbol_upper = symbol.upper()
    df = df.copy()
    df["symbol"] = symbol_upper
    df["value"] = df["volume"] * df["close"]
    df["exchange"] = exchange_map.get(symbol_upper, "HOSE")  # default to HOSE
    df["industry"] = industry_map.get(symbol_upper, "Unknown")
    return df[["date", "symbol", "open", "high", "low", "close", "volume", "value", "exchange", "industry"]]


def fetch_symbol_history(
    symbol: str,
    start_date: str,
    end_date: str,
    exchange_map: dict,
    industry_map: dict,
) -> tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """
    Fetch and enrich historical data for one symbol.

    Returns:
        (symbol, enriched_df, error_msg)
    """
    try:
        df = get_ohlcv_history(symbol, start=start_date, end=end_date)
        if df.empty:
            return (symbol, None, f"No data returned")
        df_enriched = enrich_df(df, symbol, exchange_map, industry_map)
        return (symbol, df_enriched, None)
    except Exception as e:
        return (symbol, None, str(e))


def main():
    parser = argparse.ArgumentParser(description="Backfill OHLCV data from 2020 onwards")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), default today")
    parser.add_argument("--workers", type=int, default=10, help="Parallel fetch workers")
    parser.add_argument("--clear-first", action="store_true", help="Clear existing parquet before backfill")
    args = parser.parse_args()

    if args.clear_first:
        ohlcv_store.clear()
        print("✓ Cleared existing ohlcv_master.parquet")

    # Determine end date
    end_date = args.end or datetime.now().strftime("%Y-%m-%d")

    # Fetch symbols
    print("📥 Fetching VN30 + VN100 symbols...")
    vn30 = set(get_vn30_symbols())
    vn100 = set(get_vn100_symbols())
    symbols = sorted(list(vn30 | vn100))
    print(f"   Found {len(symbols)} unique symbols")

    # Fetch maps
    print("📥 Fetching exchange and industry maps...")
    exchange_map = get_exchange_map()
    industry_map = get_industry_map()
    print(f"   Exchange map: {len(exchange_map)} symbols")
    print(f"   Industry map: {len(industry_map)} symbols")

    # Parallel fetch
    print(f"\n🚀 Fetching OHLCV for {len(symbols)} symbols ({args.start} → {end_date}) with {args.workers} workers...")
    all_dfs = []
    errors = {}
    processed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch_symbol_history,
                sym,
                args.start,
                end_date,
                exchange_map,
                industry_map,
            ): sym
            for sym in symbols
        }

        for future in as_completed(futures):
            sym, df, error = future.result()
            processed += 1
            if error:
                errors[sym] = error
                print(f"  ❌ {sym:6} — {error}")
            else:
                all_dfs.append(df)
                rows = len(df)
                date_min = df["date"].min().date()
                date_max = df["date"].max().date()
                print(f"  ✓ {sym:6} — {rows:6} rows ({date_min} → {date_max})")

    print(f"\n✓ Processed {processed}/{len(symbols)} symbols")
    print(f"  Successful: {len(all_dfs)}")
    print(f"  Errors: {len(errors)}")

    if errors:
        print("\n⚠️  Errors:")
        for sym, err in errors.items():
            print(f"  {sym}: {err}")

    if not all_dfs:
        print("❌ No data fetched. Exiting.")
        return

    # Combine and write
    print(f"\n📊 Combining {len(all_dfs)} DataFrames...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["date", "symbol"], keep="last")
    combined_df = combined_df.sort_values(["symbol", "date"]).reset_index(drop=True)

    print(f"   Total rows: {len(combined_df)}")
    print(f"   Symbols: {combined_df['symbol'].nunique()}")
    print(f"   Date range: {combined_df['date'].min().date()} → {combined_df['date'].max().date()}")
    print(f"   Exchanges: {combined_df['exchange'].unique().tolist()}")
    print(f"   Industries: {combined_df['industry'].nunique()}")

    print(f"\n💾 Writing to {ohlcv_store.STORE_PATH}...")
    rows_added = ohlcv_store.append(combined_df)
    print(f"✓ Appended {rows_added} rows")

    # Summary
    print("\n📈 Store summary:")
    final_df = ohlcv_store.load()
    print(f"   Total rows: {len(final_df)}")
    print(f"   Symbols: {final_df['symbol'].nunique()}")
    print(f"   Date range: {final_df['date'].min().date()} → {final_df['date'].max().date()}")


if __name__ == "__main__":
    main()
