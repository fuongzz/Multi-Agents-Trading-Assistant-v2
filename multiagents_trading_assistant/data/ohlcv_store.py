"""
Standardized OHLCV Parquet data store.

Schema:
  date (datetime64[ns])   — trading date (Asia/Ho_Chi_Minh, normalized)
  symbol (str)            — stock symbol (uppercase, 3 chars)
  open, high, low, close (float64)
  volume (int64)          — trading volume (shares)
  value (float64)         — trading value = volume * close (VND)
  exchange (str)          — "HOSE" or "HNX"
  industry (str)          — ICB L2 industry name
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import date
from typing import Optional

STORE_PATH = Path(__file__).parent / "ohlcv_master.parquet"

# PyArrow schema for type enforcement
OHLCV_SCHEMA = pa.schema([
    ("date", pa.timestamp("ns")),
    ("symbol", pa.string()),
    ("open", pa.float64()),
    ("high", pa.float64()),
    ("low", pa.float64()),
    ("close", pa.float64()),
    ("volume", pa.int64()),
    ("value", pa.float64()),
    ("exchange", pa.string()),
    ("industry", pa.string()),
])


def load(
    symbols: Optional[list[str]] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data from parquet store.

    Args:
        symbols: Filter by symbols (case-insensitive)
        start: Filter by date >= start
        end: Filter by date <= end

    Returns:
        DataFrame with standardized schema. Empty if file doesn't exist.
    """
    if not STORE_PATH.exists():
        return pd.DataFrame(columns=[f.name for f in OHLCV_SCHEMA])

    # Read entire parquet
    df = pd.read_parquet(STORE_PATH)

    # Filter by symbols
    if symbols:
        symbols_upper = [s.upper() for s in symbols]
        df = df[df["symbol"].isin(symbols_upper)]

    # Filter by date range
    if start:
        df = df[df["date"].dt.date >= start]
    if end:
        df = df[df["date"].dt.date <= end]

    return df.reset_index(drop=True)


def append(df: pd.DataFrame) -> int:
    """
    Append new rows to parquet store. Deduplicates by (date, symbol).

    Args:
        df: DataFrame with columns matching OHLCV_SCHEMA

    Returns:
        Number of new rows added (after deduplication)
    """
    # Validate schema
    df = df[["date", "symbol", "open", "high", "low", "close", "volume", "value", "exchange", "industry"]]
    df["symbol"] = df["symbol"].str.upper()

    # Ensure datetime
    if df["date"].dtype != "datetime64[ns]":
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # Load existing data
    if STORE_PATH.exists():
        existing_df = pd.read_parquet(STORE_PATH)
        # Concat
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        # Deduplicate: keep last (most recent)
        combined_df = combined_df.drop_duplicates(subset=["date", "symbol"], keep="last")
        combined_df = combined_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    else:
        combined_df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Write back
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(combined_df, schema=OHLCV_SCHEMA)
    pq.write_table(table, STORE_PATH, compression="snappy")

    # Return count of new rows (rows not in existing)
    if STORE_PATH.exists() and len(existing_df) > 0:
        new_count = len(combined_df) - len(existing_df)
        return max(0, new_count)
    return len(combined_df)


def get_last_date(symbol: Optional[str] = None) -> Optional[date]:
    """
    Get the most recent trading date in store.

    Args:
        symbol: If provided, get last date for this symbol only

    Returns:
        Last date as date object, or None if store is empty
    """
    if not STORE_PATH.exists():
        return None

    df = pd.read_parquet(STORE_PATH)

    if df.empty:
        return None

    if symbol:
        symbol_upper = symbol.upper()
        df = df[df["symbol"] == symbol_upper]
        if df.empty:
            return None

    return df["date"].max().date()


def clear() -> None:
    """Delete the entire parquet file (use with caution)."""
    if STORE_PATH.exists():
        STORE_PATH.unlink()
        print(f"Cleared {STORE_PATH}")
