"""
Daily incremental OHLCV update script.
Runs after market close (15:35 VN time) to append new trading data.

Usage:
    python scripts/update_ohlcv_daily.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import pandas as pd

# Add parent to path
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
        ref_df["symbol"] = ref_df["symbol"].astype(str).str.upper().str.strip()
        return dict(zip(ref_df["symbol"], ref_df["exchange"]))
    except Exception as e:
        print(f"⚠️  Failed to fetch exchange map: {e}")
        return {}


def get_industry_map() -> dict[str, str]:
    """Fetch industry mapping: {symbol: icb_name}"""
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
    """Enrich OHLCV DataFrame with exchange, industry, and value."""
    if df.empty:
        return df

    symbol_upper = symbol.upper()
    df = df.copy()
    df["symbol"] = symbol_upper
    df["value"] = df["volume"] * df["close"]
    df["exchange"] = exchange_map.get(symbol_upper, "HOSE")
    df["industry"] = industry_map.get(symbol_upper, "Unknown")
    return df[["date", "symbol", "open", "high", "low", "close", "volume", "value", "exchange", "industry"]]


def fetch_symbol_history(
    symbol: str,
    start_date: str,
    end_date: str,
    exchange_map: dict,
    industry_map: dict,
) -> tuple[str, pd.DataFrame, Optional[str]]:
    """Fetch and enrich historical data for one symbol."""
    try:
        df = get_ohlcv_history(symbol, start=start_date, end=end_date)
        if df.empty:
            return (symbol, pd.DataFrame(), None)  # No data but not an error
        df_enriched = enrich_df(df, symbol, exchange_map, industry_map)
        return (symbol, df_enriched, None)
    except Exception as e:
        return (symbol, pd.DataFrame(), str(e))


def main():
    print("📅 Daily OHLCV update started")

    # Check store state
    last_date = ohlcv_store.get_last_date()
    today = datetime.now().date()

    if last_date == today:
        print(f"✓ Store already updated for {today}. Skipping.")
        return

    # Determine fetch range: yesterday to today
    if last_date is None:
        print("⚠️  No data in store. Run build_ohlcv_history.py first.")
        return

    fetch_start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end = today.strftime("%Y-%m-%d")

    print(f"📥 Fetching data for {fetch_start} → {fetch_end}")

    # Fetch symbols
    vn30 = set(get_vn30_symbols())
    vn100 = set(get_vn100_symbols())
    symbols = sorted(list(vn30 | vn100))
    print(f"   {len(symbols)} symbols")

    # Fetch maps
    print("📥 Fetching exchange and industry maps...")
    exchange_map = get_exchange_map()
    industry_map = get_industry_map()

    # Parallel fetch (fewer workers for daily update)
    print(f"🚀 Fetching OHLCV for {len(symbols)} symbols...")
    all_dfs = []
    errors = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                fetch_symbol_history,
                sym,
                fetch_start,
                fetch_end,
                exchange_map,
                industry_map,
            ): sym
            for sym in symbols
        }

        for future in as_completed(futures):
            sym, df, error = future.result()
            if error:
                print(f"  ❌ {sym}: {error}")
                errors += 1
            elif not df.empty:
                all_dfs.append(df)
                rows = len(df)
                print(f"  ✓ {sym} — {rows} rows")

    if errors:
        print(f"⚠️  {errors} symbols had errors")

    if not all_dfs:
        print("✓ No new data to append (market closed or holiday)")
        return

    # Combine and append
    print(f"\n📊 Combining {len(all_dfs)} DataFrames...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["date", "symbol"], keep="last")
    combined_df = combined_df.sort_values(["symbol", "date"]).reset_index(drop=True)

    print(f"   Total rows: {len(combined_df)}")

    print(f"💾 Appending to {ohlcv_store.STORE_PATH}...")
    rows_added = ohlcv_store.append(combined_df)
    print(f"✓ Added {rows_added} rows")

    # Final check
    last_date_new = ohlcv_store.get_last_date()
    print(f"\n✓ Update complete. Latest date in store: {last_date_new}")


if __name__ == "__main__":
    main()
