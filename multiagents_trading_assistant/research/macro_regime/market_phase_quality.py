"""Causal market phase classifier for add-on/pyramiding decisions.

The goal is not to predict tops/bottoms. It classifies the market state using
only information available at date T so that add-on decisions can be applied
from T+1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


MarketPhase = Literal[
    "BEAR",
    "REBOUND",
    "EARLY_BULL",
    "MID_BULL",
    "LATE_BULL",
    "DISTRIBUTION",
    "CHOPPY",
]


@dataclass(frozen=True)
class MarketPhaseConfig:
    min_add_on_score: float = 68.0
    allow_late_bull_thrust: bool = False
    max_late_extension_20d: float = 0.11
    max_late_extension_ma20: float = 0.07
    min_mid_breadth_ma50: float = 0.62
    min_early_breadth_thrust_20d: float = 0.18
    max_distribution_pressure: float = 1.7


def build_market_phase_frame(
    features: pd.DataFrame,
    *,
    config: MarketPhaseConfig | None = None,
) -> pd.DataFrame:
    cfg = config or MarketPhaseConfig()
    feat = features.copy()
    feat["date"] = pd.to_datetime(feat["date"]).dt.tz_localize(None).dt.normalize()

    daily = (
        feat.groupby("date", sort=True)
        .agg(
            mkt_regime_score=("mkt_regime_score", "median"),
            mkt_chdm20=("mkt_CHDM20", "median"),
            mkt_chdm50=("mkt_CHDM50", "median"),
            mkt_ds20=("mkt_DS20", "median"),
            mkt_ds50=("mkt_DS50", "median"),
            vni_close=("vni_close", "median"),
            vni_ret_20d=("vni_ret_20d", "median"),
            vni_ret_60d=("vni_ret_60d", "median"),
            vni_above_ma20=("vni_above_ma20", "median"),
            vni_above_ma50=("vni_above_ma50", "median"),
            breadth_ma20=("above_ma20", "mean"),
            breadth_ma50=("above_ma50", "mean"),
            avg_distribution_days_10=("distribution_days_10", "mean"),
            high20_count=("breakout_20", "sum"),
            high55_count=("breakout_55", "sum"),
        )
        .reset_index()
        .sort_values("date")
    )
    daily["breadth_ma20_delta_10d"] = daily["breadth_ma20"].diff(10)
    daily["breadth_ma50_delta_20d"] = daily["breadth_ma50"].diff(20)
    daily["mkt_score_delta_10d"] = daily["mkt_regime_score"].diff(10)
    daily["mkt_chdm50_delta_20d"] = daily["mkt_chdm50"].diff(20)
    daily["vni_close_ma20"] = daily["vni_close"].rolling(20, min_periods=5).mean()
    daily["vni_close_ma50"] = daily["vni_close"].rolling(50, min_periods=10).mean()
    daily["extension_ma20"] = daily["vni_close"] / daily["vni_close_ma20"] - 1.0
    daily["extension_ma50"] = daily["vni_close"] / daily["vni_close_ma50"] - 1.0
    daily["distribution_pressure"] = daily["avg_distribution_days_10"].rolling(5, min_periods=1).mean()

    daily["market_phase"] = daily.apply(_classify_phase, axis=1, cfg=cfg)
    daily["market_phase_score"] = daily.apply(_phase_score, axis=1, cfg=cfg).clip(0.0, 100.0)
    phase_ok = daily["market_phase"].isin(["EARLY_BULL", "MID_BULL"])
    if cfg.allow_late_bull_thrust:
        phase_ok |= (
            (daily["market_phase"] == "LATE_BULL")
            & (daily["breadth_ma20_delta_10d"] >= 0.12)
            & (daily["extension_ma20"] <= cfg.max_late_extension_ma20)
        )
    daily["allow_add_on_phase_v1"] = (
        phase_ok
        & (daily["market_phase_score"] >= cfg.min_add_on_score)
        & (daily["distribution_pressure"] <= cfg.max_distribution_pressure)
        & (daily["breadth_ma20_delta_10d"] >= -0.15)
    )
    daily["add_on_budget_multiplier"] = daily.apply(_budget_multiplier, axis=1, cfg=cfg)
    daily["add_on_risk_throttle_multiplier"] = daily.apply(_risk_throttle_multiplier, axis=1, cfg=cfg)
    daily["phase_reason"] = daily.apply(_phase_reason, axis=1)
    return daily


def _classify_phase(row: pd.Series, cfg: MarketPhaseConfig) -> MarketPhase:
    score = _f(row, "mkt_regime_score")
    chdm50 = _f(row, "mkt_chdm50")
    ds20 = _f(row, "mkt_ds20")
    ret20 = _f(row, "vni_ret_20d")
    ret60 = _f(row, "vni_ret_60d")
    breadth20 = _f(row, "breadth_ma20")
    breadth50 = _f(row, "breadth_ma50")
    breadth20_delta = _f(row, "breadth_ma20_delta_10d")
    breadth50_delta = _f(row, "breadth_ma50_delta_20d")
    extension20 = _f(row, "extension_ma20")
    distribution = _f(row, "distribution_pressure")

    if score < 42 or (chdm50 < 42 and ds20 > 0.52) or ret60 < -0.08:
        return "BEAR"
    if distribution > 2.4 and breadth20_delta < -0.08:
        return "DISTRIBUTION"
    if score >= 58 and ret20 > 0 and ret60 < 0.03 and breadth20_delta >= cfg.min_early_breadth_thrust_20d:
        return "EARLY_BULL"
    if score >= 68 and ret60 > 0.03 and breadth50 >= cfg.min_mid_breadth_ma50 and ds20 <= 0.35:
        if ret20 >= cfg.max_late_extension_20d or extension20 >= cfg.max_late_extension_ma20:
            return "LATE_BULL"
        return "MID_BULL"
    if score >= 52 and ret20 > 0 and breadth20 > 0.55:
        return "REBOUND"
    if distribution > 1.8 and breadth50_delta < -0.04:
        return "DISTRIBUTION"
    return "CHOPPY"


def _phase_score(row: pd.Series, cfg: MarketPhaseConfig) -> float:
    score = 0.0
    score += 0.30 * _clip(_f(row, "mkt_regime_score"), 0.0, 100.0)
    score += 20.0 * _clip(_f(row, "breadth_ma50"), 0.0, 1.0)
    score += 15.0 * _clip(_f(row, "breadth_ma20_delta_10d") + 0.10, 0.0, 0.35) / 0.35
    score += 15.0 * _clip(_f(row, "vni_ret_60d") + 0.02, 0.0, 0.18) / 0.18
    score += 10.0 * (1.0 - _clip(_f(row, "mkt_ds20"), 0.0, 0.65) / 0.65)
    score -= 10.0 * _clip((_f(row, "extension_ma20") - cfg.max_late_extension_ma20) / 0.08, 0.0, 1.0)
    score -= 10.0 * _clip((_f(row, "distribution_pressure") - cfg.max_distribution_pressure) / 2.0, 0.0, 1.0)
    return score


def _phase_reason(row: pd.Series) -> str:
    reasons: list[str] = [str(row.get("market_phase", ""))]
    if _f(row, "extension_ma20") > 0.07:
        reasons.append("EXTENDED")
    if _f(row, "distribution_pressure") > 1.7:
        reasons.append("DISTRIBUTION_PRESSURE")
    if _f(row, "breadth_ma20_delta_10d") > 0.18:
        reasons.append("BREADTH_THRUST")
    if _f(row, "breadth_ma50") > 0.62:
        reasons.append("BROAD_PARTICIPATION")
    return "|".join(reasons)


def _budget_multiplier(row: pd.Series, cfg: MarketPhaseConfig) -> float:
    phase = str(row.get("market_phase", ""))
    score = _f(row, "market_phase_score")
    distribution = _f(row, "distribution_pressure")
    breadth_delta = _f(row, "breadth_ma20_delta_10d")
    extension20 = _f(row, "extension_ma20")

    if distribution > cfg.max_distribution_pressure or breadth_delta < -0.20:
        return 0.0
    if phase == "EARLY_BULL":
        return 1.0 if score >= 55 else 0.5
    if phase == "MID_BULL":
        if score >= 70:
            return 1.0
        if score >= 55:
            return 0.75
        return 0.50
    if phase == "LATE_BULL":
        if extension20 <= cfg.max_late_extension_ma20 and breadth_delta >= 0.08:
            return 0.50
        return 0.25 if score >= 65 and distribution <= 1.2 else 0.0
    if phase == "REBOUND":
        return 0.25 if score >= 65 and breadth_delta > 0.10 else 0.0
    return 0.0


def _risk_throttle_multiplier(row: pd.Series, cfg: MarketPhaseConfig) -> float:
    phase = str(row.get("market_phase", ""))
    distribution = _f(row, "distribution_pressure")
    breadth_delta = _f(row, "breadth_ma20_delta_10d")
    extension20 = _f(row, "extension_ma20")
    score = _f(row, "market_phase_score")

    if phase == "BEAR":
        return 0.0
    if phase == "DISTRIBUTION" and (distribution > 1.8 or breadth_delta < -0.12):
        return 0.0
    if distribution > 2.4 or breadth_delta < -0.30:
        return 0.0
    if phase == "LATE_BULL" and (extension20 > cfg.max_late_extension_ma20 or breadth_delta < -0.12):
        return 0.50
    if phase == "REBOUND" and score < 55:
        return 0.50
    return 1.0


def _f(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), lo), hi)
