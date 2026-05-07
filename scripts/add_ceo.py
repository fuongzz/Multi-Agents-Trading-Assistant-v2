#!/usr/bin/env python3
"""Quick script to add CEO stock to ohlcv_master.parquet"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multiagents_trading_assistant.fetcher import get_ohlcv_history
from multiagents_trading_assistant.data import ohlcv_store
import pandas as pd

SYMBOL = "CEO"
START_DATE = "2020-01-01"

print(f"🔄 Adding {SYMBOL} to ohlcv_master.parquet...\n")

# Fetch data
print(f"📥 Fetching {SYMBOL} history from {START_DATE}...")
try:
    df = get_ohlcv_history(SYMBOL, start=START_DATE)

    if df.empty:
        print(f"❌ No data found for {SYMBOL}")
        sys.exit(1)

    # Enrich
    df = df.copy()
    df["symbol"] = SYMBOL
    df["value"] = df["volume"] * df["close"]
    df["exchange"] = "HOSE"
    df["industry"] = "Unknown"  # Can be enriched later

    # Ensure columns in right order
    df = df[["date", "symbol", "open", "high", "low", "close", "volume", "value", "exchange", "industry"]]

    # Append to store
    rows_added = ohlcv_store.append(df)

    print(f"\n✓ Successfully added {SYMBOL}")
    print(f"  Rows added: {rows_added}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Verify
    final_df = ohlcv_store.load(symbols=[SYMBOL])
    print(f"\n📊 Verification:")
    print(f"  Total rows for {SYMBOL}: {len(final_df)}")
    print(f"  Latest date: {final_df['date'].max().date()}")
    print(f"\n✓ Done!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
