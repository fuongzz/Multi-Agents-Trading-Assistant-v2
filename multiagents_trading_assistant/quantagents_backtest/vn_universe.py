"""VN universe helpers with historical-constituent support hooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CURRENT_VN30 = [
    "ACB",
    "BCM",
    "BID",
    "BVH",
    "CTG",
    "FPT",
    "GAS",
    "GVR",
    "HDB",
    "HPG",
    "MBB",
    "MSN",
    "MWG",
    "PLX",
    "POW",
    "SAB",
    "SHB",
    "SSB",
    "SSI",
    "STB",
    "TCB",
    "TPB",
    "VCB",
    "VHM",
    "VIB",
    "VIC",
    "VJC",
    "VNM",
    "VPB",
    "VRE",
]


@dataclass(frozen=True)
class UniverseSnapshot:
    date: pd.Timestamp
    symbols: list[str]
    source: str
    is_historical: bool


def load_historical_constituents(path: str | Path) -> pd.DataFrame:
    """Load historical index constituents.

    Expected CSV columns:
        effective_date,index,symbol

    Example:
        2021-01-01,VN30,ACB
    """

    df = pd.read_csv(path)
    required = {"effective_date", "index", "symbol"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Constituent file is missing columns: {sorted(missing)}")
    df = df.copy()
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df["index"] = df["index"].astype(str).str.upper()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["effective_date", "symbol"])
    return df.sort_values(["index", "effective_date", "symbol"]).reset_index(drop=True)


def get_universe_snapshot(
    as_of: str | pd.Timestamp,
    index_name: str = "VN30",
    constituents: pd.DataFrame | None = None,
) -> UniverseSnapshot:
    """Return symbols valid at ``as_of``.

    If historical constituents are unavailable, this returns the current VN30
    fallback and marks ``is_historical=False`` so reports can flag survivorship
    risk explicitly.
    """

    date = pd.Timestamp(as_of)
    index_name = index_name.upper()
    if constituents is not None and not constituents.empty:
        data = constituents[constituents["index"].astype(str).str.upper() == index_name]
        effective_dates = data.loc[data["effective_date"] <= date, "effective_date"]
        if not effective_dates.empty:
            latest = effective_dates.max()
            symbols = sorted(data.loc[data["effective_date"] == latest, "symbol"].dropna().unique().tolist())
            return UniverseSnapshot(date, symbols, source=f"{index_name}_historical_constituents", is_historical=True)

    if index_name != "VN30":
        raise ValueError(f"No fallback universe configured for {index_name}")
    return UniverseSnapshot(date, CURRENT_VN30.copy(), source="current_vn30_fallback", is_historical=False)


def build_rebalance_calendar(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    frequency: str = "Q",
) -> pd.DatetimeIndex:
    """Create rebalance dates for universe snapshots."""

    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=frequency)
    if len(dates) == 0 or dates[0] > pd.Timestamp(start):
        dates = dates.insert(0, pd.Timestamp(start))
    return dates
