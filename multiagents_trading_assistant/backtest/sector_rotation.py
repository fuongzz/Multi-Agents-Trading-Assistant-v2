"""Causal sector rotation model for lifecycle backtests.

The VN market often rotates by sector. This module computes a daily sector
strength table from OHLCV only up to each date, then lets symbol-level signal
detectors decide whether a stock is allowed to open a large core position.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import pandas as pd


DEFAULT_SECTOR_MAP: dict[str, list[str]] = {
    "BANK": ["VCB", "BID", "CTG", "TCB", "VPB", "MBB", "ACB", "STB", "HDB", "LPB"],
    "SECURITIES": ["SSI", "VND", "HCM", "MBS", "VCI", "BSI", "AGR", "CTS", "SHS"],
    "STEEL": ["HPG", "HSG", "NKG", "TLH", "VGS", "SMC"],
    "REAL_ESTATE": ["VIC", "VHM", "NVL", "PDR", "DIG", "KDH", "NLG", "DXG", "BCM"],
    "OIL_GAS": ["PVD", "GAS", "PVS", "BSR", "OIL", "PVC"],
    "CONSUMER_RETAIL": ["VNM", "SAB", "MSN", "MWG", "FRT", "PNJ", "DGC"],
    "TECH": ["FPT", "VGI", "CMG", "ELC"],
    "POWER_INFRA": ["REE", "PC1", "GEG", "POW", "NT2", "VSH"],
}


class SectorRegime(str, Enum):
    LEADING = "LEADING"
    IMPROVING = "IMPROVING"
    NEUTRAL = "NEUTRAL"
    WEAKENING = "WEAKENING"
    LAGGING = "LAGGING"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SectorSnapshot:
    sector: str
    regime: SectorRegime
    rank: int
    score: float
    ret20: float
    ret60: float
    breadth20: float
    breadth50: float
    notes: list[str]

    @property
    def allows_core(self) -> bool:
        return self.regime in {SectorRegime.LEADING, SectorRegime.IMPROVING}


class SectorRotationModel:
    """Precomputed daily sector ranks from historical OHLCV maps."""

    def __init__(
        self,
        sector_map: Mapping[str, list[str]],
        ohlcv_map: Mapping[str, pd.DataFrame],
    ):
        self.sector_map = {k: [s.upper() for s in v] for k, v in sector_map.items()}
        self.symbol_to_sector = {
            symbol: sector
            for sector, symbols in self.sector_map.items()
            for symbol in symbols
        }
        self._sector_daily = self._build_sector_daily(ohlcv_map)

    def snapshot(self, symbol: str, date: str) -> SectorSnapshot:
        sector = self.symbol_to_sector.get(symbol.upper(), "")
        if not sector or self._sector_daily.empty:
            return _unknown_snapshot(sector or "UNKNOWN", "missing_sector_data")
        d = pd.Timestamp(str(date)[:10])
        data = self._sector_daily
        rows = data[(data["sector"] == sector) & (data["date"] <= d)]
        if rows.empty:
            return _unknown_snapshot(sector, "no_sector_row")
        row = rows.iloc[-1]
        return SectorSnapshot(
            sector=sector,
            regime=SectorRegime(str(row["regime"])),
            rank=int(row["rank"]),
            score=float(row["score"]),
            ret20=float(row["ret20"]),
            ret60=float(row["ret60"]),
            breadth20=float(row["breadth20"]),
            breadth50=float(row["breadth50"]),
            notes=[str(row["note"])],
        )

    def _build_sector_daily(self, ohlcv_map: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        symbol_features: dict[str, pd.DataFrame] = {}
        for symbol, df in ohlcv_map.items():
            feat = _symbol_features(symbol, df)
            if not feat.empty:
                symbol_features[symbol.upper()] = feat

        sector_frames: list[pd.DataFrame] = []
        for sector, symbols in self.sector_map.items():
            frames = [symbol_features[s] for s in symbols if s in symbol_features]
            if len(frames) < 2:
                continue
            merged = pd.concat(frames, ignore_index=True)
            grouped = (
                merged.groupby("date", as_index=False)
                .agg(
                    ret20=("ret20", "mean"),
                    ret60=("ret60", "mean"),
                    breadth20=("above_ma20", "mean"),
                    breadth50=("above_ma50", "mean"),
                    members=("symbol", "nunique"),
                )
                .sort_values("date")
            )
            grouped["sector"] = sector
            sector_frames.append(grouped)

        if not sector_frames:
            return pd.DataFrame()

        data = pd.concat(sector_frames, ignore_index=True).sort_values(["date", "sector"])
        data["score"] = (
            data["ret20"] * 1.35
            + data["ret60"] * 0.75
            + data["breadth20"] * 0.18
            + data["breadth50"] * 0.12
        )
        data["rank"] = data.groupby("date")["score"].rank(method="first", ascending=False).astype(int)
        data["regime"] = data.apply(_classify_sector_row, axis=1)
        data["note"] = data.apply(
            lambda r: (
                f"sector={r['sector']} rank={int(r['rank'])} score={r['score']:.3f} "
                f"ret20={r['ret20']:.1%} ret60={r['ret60']:.1%} "
                f"breadth20={r['breadth20']:.0%} breadth50={r['breadth50']:.0%}"
            ),
            axis=1,
        )
        return data


def _symbol_features(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or len(df) < 65:
        return pd.DataFrame()
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data = data.sort_values("date")
    close = data["close"].astype(float)
    out = pd.DataFrame(
        {
            "date": data["date"],
            "symbol": symbol.upper(),
            "ret20": close / close.shift(20) - 1.0,
            "ret60": close / close.shift(60) - 1.0,
            "above_ma20": (close > close.rolling(20).mean()).astype(float),
            "above_ma50": (close > close.rolling(50).mean()).astype(float),
        }
    )
    return out.dropna(subset=["ret20", "ret60"])


def _classify_sector_row(row) -> str:
    rank = int(row["rank"])
    ret20 = float(row["ret20"])
    ret60 = float(row["ret60"])
    breadth20 = float(row["breadth20"])
    breadth50 = float(row["breadth50"])
    if rank == 1 and ret20 > 0.02 and ret60 > 0.04 and breadth20 >= 0.55:
        return SectorRegime.LEADING.value
    if rank <= 2 and ret20 > 0.0 and ret60 > 0.0 and breadth20 >= 0.45:
        return SectorRegime.IMPROVING.value
    if ret20 < -0.03 and breadth20 < 0.35:
        return SectorRegime.LAGGING.value
    if ret20 < 0.0 or breadth20 < 0.45:
        return SectorRegime.WEAKENING.value
    return SectorRegime.NEUTRAL.value


def _unknown_snapshot(sector: str, reason: str) -> SectorSnapshot:
    return SectorSnapshot(
        sector=sector,
        regime=SectorRegime.UNKNOWN,
        rank=0,
        score=0.0,
        ret20=0.0,
        ret60=0.0,
        breadth20=0.0,
        breadth50=0.0,
        notes=[reason],
    )
