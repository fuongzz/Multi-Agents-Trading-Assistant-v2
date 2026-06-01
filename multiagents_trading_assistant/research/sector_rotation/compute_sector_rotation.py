"""
Sector Rotation computation — Layer 2 of the cycle framework.

For each (date, sector_l1), aggregate:
  - sector_chdm_{20,50}: median CHDM across member symbols (position in range)
  - sector_ds_{20,50}:   mean DS across member symbols (proportion in sell state)
  - sector_ret_{5,10,20}d: equal-weight log return over horizon
  - sector_rs_{5,10,20}d:  sector return minus equal-weight market composite
  - sector_rs_rank_20d:    cross-sector percentile rank (0..1) of rs_20d that day
  - sector_state:          Wyckoff-style label
                           {ACCUMULATION, EARLY_MARKUP, MARKUP,
                            DISTRIBUTION, MARKDOWN, NEUTRAL}

Inputs:
  - ohlcv_master.parquet            (date, symbol, close, industry, exchange)
  - chdm_by_symbol.parquet          (date, symbol, CHDM03..CHDM200)
  - ds_by_symbol.parquet            (date, symbol, DS03..DS200)

Output:
  - data/research/sector_rotation/sector_rotation.parquet
"""

from __future__ import annotations

from collections.abc import Collection
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .icb_mapping import L1_SECTORS, L2_TO_L1

logger = logging.getLogger(__name__)

# Sector state thresholds — calibrated for VN market typical ranges.
# CHDM is 0-100 (position in 50d range); DS20 is 0-1 (frac of stocks in sell mode).
_TH = {
    "chdm_low": 35.0,
    "chdm_mid_lo": 30.0,
    "chdm_mid_hi": 60.0,
    "chdm_high": 75.0,
    "chdm_top": 85.0,
    "ds_low": 0.35,
    "ds_mid": 0.50,
    "ds_high": 0.55,
}


def _classify_state(row: pd.Series) -> str:
    chdm = row["sector_chdm_50"]
    ds = row["sector_ds_20"]
    ds_delta = row["sector_ds_20_delta_5d"]  # negative => improving breadth
    rs5 = row["sector_rs_5d"]
    rs20 = row["sector_rs_20d"]

    if pd.isna(chdm) or pd.isna(ds) or pd.isna(rs20):
        return "NEUTRAL"

    # Accumulation: low CHDM, high DS, but RS stabilizing
    if chdm < _TH["chdm_low"] and ds > _TH["ds_mid"] and rs20 > -0.02:
        return "ACCUMULATION"

    # Early markup: CHDM lifting off, DS dropping, RS5 turning up
    if (_TH["chdm_mid_lo"] <= chdm <= _TH["chdm_mid_hi"]
            and ds < _TH["ds_mid"] and rs5 > 0 and ds_delta < 0):
        return "EARLY_MARKUP"

    # Markup: mid-to-high CHDM, low DS, RS positive
    if (_TH["chdm_mid_hi"] - 10 <= chdm <= _TH["chdm_top"]
            and ds < _TH["ds_low"] and rs20 > 0):
        return "MARKUP"

    # Distribution: high CHDM but breadth deteriorating or short-term RS rolling over
    if chdm > _TH["chdm_high"] and (ds_delta > 0 or rs5 < rs20 - 0.01):
        return "DISTRIBUTION"

    # Markdown: low CHDM, high DS, negative RS
    if chdm < _TH["chdm_mid_hi"] and ds > _TH["ds_high"] and rs20 < 0:
        return "MARKDOWN"

    return "NEUTRAL"


def _equal_weight_market_return(prices: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    """Equal-weight market composite log return per horizon."""
    pivot = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    log_close = np.log(pivot)
    out = pd.DataFrame(index=pivot.index)
    for h in horizons:
        ret = log_close - log_close.shift(h)
        out[f"mkt_ret_{h}d"] = ret.mean(axis=1)
    return out.reset_index()


def _sector_returns(prices: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    """Equal-weight log return per sector per horizon."""
    pivot = prices.pivot_table(
        index="date", columns=["sector_l1", "symbol"], values="close"
    ).sort_index()
    log_close = np.log(pivot)
    parts = []
    for h in horizons:
        ret = log_close - log_close.shift(h)
        sector_ret = ret.T.groupby(level="sector_l1").mean().T
        sector_ret = sector_ret.stack().rename(f"sector_ret_{h}d").reset_index()
        parts.append(sector_ret)
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["date", "sector_l1"], how="outer")
    return out


def compute_sector_rotation(
    ohlcv_path: str | Path,
    chdm_path: str | Path,
    ds_path: str | Path,
    out_path: str | Path,
    min_symbols_per_sector: int = 2,
    symbols: Collection[str] | None = None,
) -> pd.DataFrame:
    """Build sector_rotation.parquet from inputs and persist."""
    logger.info("Loading OHLCV master...")
    ohlcv = pd.read_parquet(ohlcv_path)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.normalize()
    ohlcv["symbol"] = ohlcv["symbol"].astype(str).str.upper()
    symbol_set = {str(symbol).upper() for symbol in symbols} if symbols else None
    if symbol_set is not None:
        ohlcv = ohlcv[ohlcv["symbol"].isin(symbol_set)].copy()
        logger.info("Restricted sector rotation input to %s requested symbols", len(symbol_set))
    ohlcv["sector_l1"] = ohlcv["industry"].map(L2_TO_L1)
    unmapped = ohlcv.loc[ohlcv["sector_l1"].isna(), "industry"].unique()
    if len(unmapped):
        logger.warning("Unmapped L2 industries dropped: %s", list(unmapped))
    ohlcv = ohlcv.dropna(subset=["sector_l1"]).copy()

    prices = ohlcv[["date", "symbol", "close", "sector_l1"]]

    logger.info("Computing market composite returns...")
    mkt = _equal_weight_market_return(prices)

    logger.info("Computing sector returns...")
    sec_ret = _sector_returns(prices)

    logger.info("Loading CHDM / DS by symbol...")
    chdm = pd.read_parquet(chdm_path)
    ds = pd.read_parquet(ds_path)
    chdm["date"] = pd.to_datetime(chdm["date"]).dt.normalize()
    ds["date"] = pd.to_datetime(ds["date"]).dt.normalize()
    chdm["symbol"] = chdm["symbol"].astype(str).str.upper()
    ds["symbol"] = ds["symbol"].astype(str).str.upper()
    if symbol_set is not None:
        chdm = chdm[chdm["symbol"].isin(symbol_set)].copy()
        ds = ds[ds["symbol"].isin(symbol_set)].copy()

    sym_sector = prices[["symbol", "sector_l1"]].drop_duplicates()
    chdm = chdm.merge(sym_sector, on="symbol", how="inner")
    ds = ds.merge(sym_sector, on="symbol", how="inner")

    logger.info("Aggregating CHDM (median) / DS (mean) per sector-day...")
    sec_chdm = (
        chdm.groupby(["date", "sector_l1"])[["CHDM20", "CHDM50"]]
        .median()
        .rename(columns={"CHDM20": "sector_chdm_20", "CHDM50": "sector_chdm_50"})
        .reset_index()
    )
    sec_ds = (
        ds.groupby(["date", "sector_l1"])[["DS20", "DS50"]]
        .mean()
        .rename(columns={"DS20": "sector_ds_20", "DS50": "sector_ds_50"})
        .reset_index()
    )
    n_sym = (
        ohlcv.groupby(["date", "sector_l1"])["symbol"]
        .nunique()
        .rename("n_symbols")
        .reset_index()
    )

    logger.info("Merging...")
    out = (
        sec_ret.merge(sec_chdm, on=["date", "sector_l1"], how="outer")
        .merge(sec_ds, on=["date", "sector_l1"], how="outer")
        .merge(n_sym, on=["date", "sector_l1"], how="left")
        .merge(mkt, on="date", how="left")
    )

    for h in (5, 10, 20):
        out[f"sector_rs_{h}d"] = out[f"sector_ret_{h}d"] - out[f"mkt_ret_{h}d"]

    out = out.sort_values(["sector_l1", "date"])
    out["sector_ds_20_delta_5d"] = (
        out.groupby("sector_l1")["sector_ds_20"].diff(5)
    )

    # cross-sector rank of rs_20d per day
    out["sector_rs_rank_20d"] = (
        out.groupby("date")["sector_rs_20d"].rank(pct=True, method="average")
    )

    out = out[out["n_symbols"] >= min_symbols_per_sector].copy()
    out["sector_state"] = out.apply(_classify_state, axis=1)

    out = out.sort_values(["date", "sector_l1"]).reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info("Saved %s rows -> %s", len(out), out_path)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(__file__).resolve().parents[3]
    compute_sector_rotation(
        ohlcv_path=root / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet",
        chdm_path=root / "data" / "research" / "money_cycle" / "chdm_by_symbol.parquet",
        ds_path=root / "data" / "research" / "money_cycle" / "ds_by_symbol.parquet",
        out_path=root / "data" / "research" / "sector_rotation" / "sector_rotation.parquet",
    )
