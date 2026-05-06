"""
Money Cycle Research Module

Tính chỉ báo CHDM (vị trí giá trong biên dao động N phiên) và DS (tỷ lệ cổ phiếu đang bị bán)
từ dữ liệu OHLCV. Xuất kết quả ở hai level: từng mã (by_symbol) và thị trường toàn bộ (market median/mean).

Input: OHLCV parquet/csv từ ohlcv_store.
Output: 5 parquet files trong data/research/money_cycle/:
  - chdm_market.parquet
  - ds_market.parquet
  - money_cycle_market.parquet
  - chdm_by_symbol.parquet
  - ds_by_symbol.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

WINDOWS = [3, 5, 10, 20, 50, 100, 200]


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """
    Load OHLCV data from parquet or CSV file.

    Args:
        path: File path (supports .parquet and .csv)

    Returns:
        DataFrame with columns: date, symbol, open, high, low, close, volume, ...

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format not supported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV DataFrame: lowercase columns, rename ticker→symbol, validate required columns.

    Args:
        df: Raw OHLCV DataFrame

    Returns:
        Normalized DataFrame sorted by [symbol, date], required columns in correct types.

    Raises:
        ValueError: If required columns are missing.
    """
    # Lowercase columns
    df.columns = df.columns.str.lower().str.strip()

    # Rename ticker → symbol if needed
    if "ticker" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"ticker": "symbol"})

    # Check required columns
    required = ["date", "symbol", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"])

    # Drop rows with missing OHLC
    df = df.dropna(subset=["high", "low", "close"])

    # Ensure symbol is string
    df["symbol"] = df["symbol"].astype(str).str.upper()

    # Sort by symbol, date
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    logger.info(f"Normalized: {len(df)} rows, {df['symbol'].nunique()} symbols")
    return df


def compute_chdm_by_symbol(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Compute CHDM (money cycle position) by symbol for multiple windows.

    CHDM = 100 * (Close - LowestLow_N) / (HighestHigh_N - LowestLow_N)
    Clipped to [0, 100]. When denominator is 0, result is NaN.

    Args:
        df: Normalized OHLCV DataFrame (must have symbol, high, low, close)
        windows: List of periods [3, 5, 10, ...]

    Returns:
        DataFrame with columns [date, symbol, CHDM03, CHDM05, ...CHDM200]
    """
    result = df[["date", "symbol", "close"]].copy()

    for n in windows:
        col_name = f"CHDM{n:02d}"

        # Compute rolling low/high per symbol, no look-ahead
        rolling_low = df.groupby("symbol", sort=False)["low"].transform(
            lambda x, period=n: x.rolling(period, min_periods=period).min()
        )
        rolling_high = df.groupby("symbol", sort=False)["high"].transform(
            lambda x, period=n: x.rolling(period, min_periods=period).max()
        )

        # Compute CHDM
        denom = rolling_high - rolling_low
        # Replace 0 denominator with NaN to avoid division issues
        denom = denom.replace(0.0, np.nan)
        chdm = 100 * (df["close"] - rolling_low) / denom
        chdm = chdm.clip(0, 100)

        result[col_name] = chdm

    return result[[c for c in result.columns if c != "close"]]


def compute_chdm_market(chdm_by_symbol: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Aggregate CHDM by symbol to market level using median.

    Args:
        chdm_by_symbol: DataFrame with date, symbol, CHDM* columns
        windows: List of windows (for column selection)

    Returns:
        DataFrame indexed by date with columns [CHDM03, CHDM05, ...CHDM200]
    """
    chdm_cols = [f"CHDM{n:02d}" for n in windows]
    market = chdm_by_symbol.groupby("date")[chdm_cols].median()

    logger.info(f"CHDM market: {len(market)} dates, columns={chdm_cols}")
    return market.reset_index()


def compute_ds_by_symbol(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Compute DS (distribution share) by symbol: indicator if stock is in sell/downtrend state.

    A stock is in sell_state_N if:
      - Close < MA_N
      - MA_N today < MA_N yesterday

    DS_N = 1 if sell_state_N else 0. NaN if MA data is incomplete.

    Args:
        df: Normalized OHLCV DataFrame
        windows: List of periods

    Returns:
        DataFrame with columns [date, symbol, DS03, DS05, ...DS200]
    """
    result = df[["date", "symbol", "close"]].copy()

    for n in windows:
        col_name = f"DS{n:02d}"

        # Compute MA_N and MA_(N-1) per symbol
        ma = df.groupby("symbol", sort=False)["close"].transform(
            lambda x, period=n: x.rolling(period, min_periods=period).mean()
        )
        ma_prev = df.groupby("symbol", sort=False)["close"].transform(
            lambda x, period=n: x.rolling(period, min_periods=period).mean().shift(1)
        )

        # Sell state: close below MA AND MA declining
        sell_state = (df["close"] < ma) & (ma < ma_prev)

        # Set to NaN where MA data is not available (insufficient history)
        has_data = ma.notna() & ma_prev.notna()
        ds = sell_state.where(has_data).astype("float64")

        result[col_name] = ds

    return result[[c for c in result.columns if c != "close"]]


def compute_ds_market(ds_by_symbol: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Aggregate DS by symbol to market level using mean.

    Mean of binary 0/1 values = proportion of stocks in sell state.

    Args:
        ds_by_symbol: DataFrame with date, symbol, DS* columns
        windows: List of windows

    Returns:
        DataFrame indexed by date with columns [DS03, DS05, ...DS200]
    """
    ds_cols = [f"DS{n:02d}" for n in windows]
    market = ds_by_symbol.groupby("date")[ds_cols].mean()

    logger.info(f"DS market: {len(market)} dates, columns={ds_cols}")
    return market.reset_index()


def compute_money_cycle_market(
    chdm_market: pd.DataFrame,
    ds_market: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge CHDM and DS market indicators.

    Args:
        chdm_market: DataFrame with date, CHDM* columns
        ds_market: DataFrame with date, DS* columns

    Returns:
        DataFrame with all columns [date, CHDM03, ...CHDM200, DS03, ...DS200]
    """
    mc_market = pd.merge(chdm_market, ds_market, on="date", how="inner")
    logger.info(f"Money cycle market: {len(mc_market)} dates")
    return mc_market


def save_outputs(
    output_dir: str | Path,
    chdm_market: pd.DataFrame,
    ds_market: pd.DataFrame,
    mc_market: pd.DataFrame,
    chdm_by_symbol: pd.DataFrame,
    ds_by_symbol: pd.DataFrame,
) -> None:
    """
    Save all output DataFrames to parquet files.

    Args:
        output_dir: Directory to save files
        chdm_market, ds_market, mc_market, chdm_by_symbol, ds_by_symbol: Data to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "chdm_market.parquet": chdm_market,
        "ds_market.parquet": ds_market,
        "money_cycle_market.parquet": mc_market,
        "chdm_by_symbol.parquet": chdm_by_symbol,
        "ds_by_symbol.parquet": ds_by_symbol,
    }

    for fname, df in files.items():
        fpath = output_dir / fname
        df.to_parquet(fpath, index=False)
        logger.info(f"Saved {fname}: {len(df)} rows")


def run_money_cycle_research(
    input_path: str | Path,
    output_dir: str | Path,
    windows: Optional[list[int]] = None,
) -> dict:
    """
    Run full money cycle research pipeline.

    Args:
        input_path: Path to OHLCV parquet/csv
        output_dir: Output directory for results
        windows: List of windows (default: [3, 5, 10, 20, 50, 100, 200])

    Returns:
        Dictionary with summary stats (rows, symbols, date range, etc.)
    """
    if windows is None:
        windows = WINDOWS

    logger.info(f"Starting money cycle research...")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Windows: {windows}")

    # 1. Load and normalize
    df = load_ohlcv(input_path)
    df = normalize_ohlcv_columns(df)

    n_rows = len(df)
    n_symbols = df["symbol"].nunique()
    date_min = df["date"].min()
    date_max = df["date"].max()

    logger.info(f"Data summary:")
    logger.info(f"  Rows: {n_rows}")
    logger.info(f"  Symbols: {n_symbols}")
    logger.info(f"  Date range: {date_min.date()} to {date_max.date()}")

    # 2. Compute indicators
    logger.info(f"Computing CHDM by symbol...")
    chdm_by_sym = compute_chdm_by_symbol(df, windows)

    logger.info(f"Computing CHDM market...")
    chdm_mkt = compute_chdm_market(chdm_by_sym, windows)

    logger.info(f"Computing DS by symbol...")
    ds_by_sym = compute_ds_by_symbol(df, windows)

    logger.info(f"Computing DS market...")
    ds_mkt = compute_ds_market(ds_by_sym, windows)

    logger.info(f"Merging to money cycle market...")
    mc_mkt = compute_money_cycle_market(chdm_mkt, ds_mkt)

    # 3. Save outputs
    logger.info(f"Saving outputs...")
    save_outputs(output_dir, chdm_mkt, ds_mkt, mc_mkt, chdm_by_sym, ds_by_sym)

    stats = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "windows": windows,
        "n_rows_input": n_rows,
        "n_symbols": n_symbols,
        "date_min": date_min.date(),
        "date_max": date_max.date(),
        "n_rows_chdm_market": len(chdm_mkt),
        "n_rows_ds_market": len(ds_mkt),
        "n_rows_mc_market": len(mc_mkt),
    }

    logger.info(f"Money cycle research complete!")
    return stats
