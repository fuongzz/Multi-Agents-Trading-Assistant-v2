"""
Market Regime classifier — Layer 1 of the cycle framework.

Inputs:
  - money_cycle_market.parquet       (CHDM50, DS20 = breadth/position)
  - smart_money_daily_summary.parquet (smart_money_score, accum/dist ratio)

Output per day:
  - regime: ACCUMULATION | MARKUP | DISTRIBUTION | MARKDOWN | NEUTRAL
  - risk_state: RISK_ON | NEUTRAL | RISK_OFF  (compat with existing hypotheses)

Output file: data/research/sector_rotation/market_regime.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _classify_regime(row: pd.Series) -> str:
    chdm = row["CHDM50"]
    ds = row["DS20"]
    sm = row["avg_smart_money_score"]
    ad = row["accumulation_distribution_ratio"]
    ds_d = row["ds20_delta_5d"]
    sm_d = row["sm_delta_5d"]

    if pd.isna(chdm) or pd.isna(ds) or pd.isna(sm):
        return "NEUTRAL"

    # ACCUMULATION: basing low, breadth poor but smart money buying
    if chdm < 35 and ds > 0.5 and sm_d > 1 and ad > 0.6:
        return "ACCUMULATION"

    # DISTRIBUTION: high CHDM but topping signals
    if chdm > 65 and (sm_d < -3 or (not pd.isna(ad) and ad < 0.5) or ds_d > 0.08):
        return "DISTRIBUTION"

    # MARKUP: mid-to-high CHDM, low DS, strong smart money
    if chdm >= 50 and ds < 0.45 and sm > 50:
        return "MARKUP"

    # MARKDOWN: low CHDM, broad selling, weak smart money
    if chdm < 45 and ds > 0.55 and sm < 50:
        return "MARKDOWN"

    return "NEUTRAL"


def _risk_state(row: pd.Series) -> str:
    chdm = row["CHDM50"]
    ds = row["DS20"]
    sm = row["avg_smart_money_score"]
    if pd.isna(chdm) or pd.isna(ds) or pd.isna(sm):
        return "NEUTRAL"
    if chdm > 45 and ds < 0.50 and sm > 50:
        return "RISK_ON"
    if chdm < 40 and ds > 0.55 and sm < 50:
        return "RISK_OFF"
    return "NEUTRAL"


def compute_market_regime(
    money_cycle_path: str | Path,
    smart_money_summary_path: str | Path,
    out_path: str | Path,
) -> pd.DataFrame:
    mc = pd.read_parquet(money_cycle_path)
    sm = pd.read_parquet(smart_money_summary_path)
    mc["date"] = pd.to_datetime(mc["date"]).dt.normalize()
    sm["date"] = pd.to_datetime(sm["date"]).dt.normalize()

    df = mc[["date", "CHDM20", "CHDM50", "DS20", "DS50"]].merge(
        sm[["date", "avg_smart_money_score", "median_smart_money_score",
            "accumulation_distribution_ratio",
            "accumulation_total", "distribution_total"]],
        on="date", how="inner",
    ).sort_values("date").reset_index(drop=True)

    df["ds20_delta_5d"] = df["DS20"].diff(5)
    df["sm_delta_5d"] = df["avg_smart_money_score"].diff(5)
    df["chdm50_delta_5d"] = df["CHDM50"].diff(5)

    df["regime"] = df.apply(_classify_regime, axis=1)
    df["risk_state"] = df.apply(_risk_state, axis=1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved %s rows -> %s", len(df), out_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(__file__).resolve().parents[3]
    compute_market_regime(
        money_cycle_path=root / "data" / "research" / "money_cycle" / "money_cycle_market.parquet",
        smart_money_summary_path=root / "data" / "research" / "smart_money_trace" / "smart_money_daily_summary.parquet",
        out_path=root / "data" / "research" / "sector_rotation" / "market_regime.parquet",
    )
