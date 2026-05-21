"""
Enrich edge_lab feature table with Layer-1 (market regime) and Layer-2 (sector
rotation) signals so hypothesis JSONs can filter/rank on them.

Adds columns:
  - sector_l1                      (str) ICB super-sector
  - sl1_state                      (str) Wyckoff label for the sector
  - sl1_rs_rank_20d                (float) cross-sector RS percentile
  - sl1_rs_20d                     (float) sector ret − market composite
  - sl1_chdm_50                    (float) median CHDM50 of sector members
  - sl1_ds_20                      (float) mean DS20 of sector members
  - sl1_state_prev                 (str) previous-day state (for `up`/`down` ops)
  - mkt_regime_v2                  (str) market regime label
  - mkt_risk_state_v2              (str) RISK_ON / NEUTRAL / RISK_OFF
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .icb_mapping import L2_TO_L1

_SECTOR_PATH = "data/research/sector_rotation/sector_rotation.parquet"
_REGIME_PATH = "data/research/sector_rotation/market_regime.parquet"


def enrich_features_with_rotation(
    features: pd.DataFrame,
    root: str | Path = ".",
) -> pd.DataFrame:
    root = Path(root)
    sector = pd.read_parquet(root / _SECTOR_PATH)
    regime = pd.read_parquet(root / _REGIME_PATH)
    sector["date"] = pd.to_datetime(sector["date"]).dt.normalize()
    regime["date"] = pd.to_datetime(regime["date"]).dt.normalize()

    feat = features.copy()
    feat["date"] = pd.to_datetime(feat["date"]).dt.normalize()
    if "sector_l1" not in feat.columns:
        feat["sector_l1"] = feat["industry"].map(L2_TO_L1)

    sec_cols = {
        "sector_state": "sl1_state",
        "sector_rs_rank_20d": "sl1_rs_rank_20d",
        "sector_rs_20d": "sl1_rs_20d",
        "sector_chdm_50": "sl1_chdm_50",
        "sector_ds_20": "sl1_ds_20",
    }
    sec = sector[["date", "sector_l1", *sec_cols.keys()]].rename(columns=sec_cols)
    feat = feat.merge(sec, on=["date", "sector_l1"], how="left")

    sec_prev = sec.copy()
    sec_prev["date"] = sec_prev["date"] + pd.Timedelta(days=1)
    sec_prev = sec_prev[["date", "sector_l1", "sl1_state"]].rename(
        columns={"sl1_state": "sl1_state_prev"}
    )
    feat = feat.merge(sec_prev, on=["date", "sector_l1"], how="left")

    reg = regime[["date", "regime", "risk_state"]].rename(
        columns={"regime": "mkt_regime_v2", "risk_state": "mkt_risk_state_v2"}
    )
    feat = feat.merge(reg, on="date", how="left")

    return feat
