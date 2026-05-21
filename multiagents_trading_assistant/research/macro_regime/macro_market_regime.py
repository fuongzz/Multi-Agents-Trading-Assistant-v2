"""Macro-aware add-on regime filters.

This module is intentionally conservative. It combines internal market
structure with optional macro series. Macro rows are joined with backward
as-of semantics only, so a date can only use macro observations already known
on or before that date.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MacroRegimeConfig:
    min_market_score: float = 62.0
    min_chdm50: float = 52.0
    max_ds20: float = 0.48
    min_breadth_ma50: float = 0.48
    min_vni_ret_60d: float = 0.0
    max_cpi_yoy: float = 4.5
    max_cpi_yoy_3m_delta: float = 0.6
    max_fx_60d_change: float = 0.03
    max_interbank_overnight: float = 8.5
    require_macro_known: bool = False


def build_macro_regime_frame(
    features: pd.DataFrame,
    *,
    cpi: pd.DataFrame | None = None,
    fx: pd.DataFrame | None = None,
    interest: pd.DataFrame | None = None,
    config: MacroRegimeConfig | None = None,
) -> pd.DataFrame:
    cfg = config or MacroRegimeConfig()
    feat = features.copy()
    feat["date"] = pd.to_datetime(feat["date"]).dt.tz_localize(None).dt.normalize()

    daily = (
        feat.groupby("date", sort=True)
        .agg(
            mkt_regime_score=("mkt_regime_score", "median"),
            mkt_chdm50=("mkt_CHDM50", "median"),
            mkt_ds20=("mkt_DS20", "median"),
            vni_ret_20d=("vni_ret_20d", "median"),
            vni_ret_60d=("vni_ret_60d", "median"),
            breadth_ma20=("above_ma20", "mean"),
            breadth_ma50=("above_ma50", "mean"),
            distribution_days_10=("distribution_days_10", "mean"),
        )
        .reset_index()
    )
    daily["market_structure_ok"] = (
        (daily["mkt_regime_score"] >= cfg.min_market_score)
        & (daily["mkt_chdm50"] >= cfg.min_chdm50)
        & (daily["mkt_ds20"] <= cfg.max_ds20)
        & (daily["breadth_ma50"] >= cfg.min_breadth_ma50)
        & (daily["vni_ret_60d"] >= cfg.min_vni_ret_60d)
    )

    daily = _merge_cpi(daily, cpi)
    daily = _merge_fx(daily, fx)
    daily = _merge_interest(daily, interest)

    cpi_known = daily["cpi_yoy"].notna()
    fx_known = daily["usd_vnd"].notna()
    interest_known = daily["interbank_overnight"].notna()

    cpi_ok = (
        daily["cpi_yoy"].isna()
        | (
            (daily["cpi_yoy"] <= cfg.max_cpi_yoy)
            & (daily["cpi_yoy_3m_delta"].fillna(0.0) <= cfg.max_cpi_yoy_3m_delta)
        )
    )
    fx_ok = daily["usd_vnd_60d_change"].isna() | (daily["usd_vnd_60d_change"] <= cfg.max_fx_60d_change)
    interest_ok = daily["interbank_overnight"].isna() | (
        daily["interbank_overnight"] <= cfg.max_interbank_overnight
    )
    if cfg.require_macro_known:
        cpi_ok &= cpi_known
        fx_ok &= fx_known
        # Interest history may be sparse in the current API; do not require it
        # unless a caller supplies a complete series.

    daily["macro_known"] = cpi_known | fx_known | interest_known
    daily["macro_ok"] = cpi_ok & fx_ok & interest_ok
    daily["add_on_regime_v2"] = daily["market_structure_ok"] & daily["macro_ok"]

    daily["macro_reason"] = ""
    daily.loc[~daily["market_structure_ok"], "macro_reason"] += "MARKET_STRUCTURE_WEAK|"
    daily.loc[~cpi_ok, "macro_reason"] += "CPI_PRESSURE|"
    daily.loc[~fx_ok, "macro_reason"] += "FX_PRESSURE|"
    daily.loc[~interest_ok, "macro_reason"] += "RATE_PRESSURE|"
    daily["macro_reason"] = daily["macro_reason"].str.strip("|")
    return daily


def regime_allows_add_on(
    regime_frame: pd.DataFrame | None,
    date: pd.Timestamp,
    *,
    column: str = "add_on_regime_v2",
) -> bool:
    if regime_frame is None or regime_frame.empty:
        return True
    as_of = pd.Timestamp(date).normalize()
    rows = regime_frame[regime_frame["date"] <= as_of]
    if rows.empty:
        return False
    return bool(rows.iloc[-1].get(column, False))


def load_macro_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    elif "report_time" in df.columns:
        df["date"] = pd.to_datetime(df["report_time"]).dt.tz_localize(None).dt.normalize()
    else:
        df["date"] = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def _merge_cpi(daily: pd.DataFrame, cpi: pd.DataFrame | None) -> pd.DataFrame:
    out = daily.copy()
    if cpi is None or cpi.empty:
        out["cpi_yoy"] = pd.NA
        out["cpi_yoy_3m_delta"] = pd.NA
        return out
    c = cpi.copy()
    if "date" not in c.columns:
        c["date"] = pd.to_datetime(c.index).tz_localize(None).normalize()
    c["date"] = pd.to_datetime(c["date"]).dt.tz_localize(None).dt.normalize()
    if "name" in c.columns:
        mask = c["name"].astype(str).str.contains("cùng kỳ|yoy|year", case=False, regex=True)
        if mask.any():
            c = c[mask]
    value_col = "value" if "value" in c.columns else "cpi_yoy"
    c = c[["date", value_col]].rename(columns={value_col: "cpi_yoy"}).dropna()
    c = c.groupby("date", as_index=False)["cpi_yoy"].last().sort_values("date")
    c["cpi_yoy_3m_delta"] = c["cpi_yoy"].diff(3)
    return pd.merge_asof(out.sort_values("date"), c, on="date", direction="backward")


def _merge_fx(daily: pd.DataFrame, fx: pd.DataFrame | None) -> pd.DataFrame:
    out = daily.copy()
    if fx is None or fx.empty:
        out["usd_vnd"] = pd.NA
        out["usd_vnd_60d_change"] = pd.NA
        return out
    f = fx.copy()
    if "date" not in f.columns:
        f["date"] = pd.to_datetime(f.index).tz_localize(None).normalize()
    f["date"] = pd.to_datetime(f["date"]).dt.tz_localize(None).dt.normalize()
    if "name" in f.columns:
        mask = f["name"].astype(str).str.contains("trung tâm|central|USD", case=False, regex=True)
        if mask.any():
            f = f[mask]
    value_col = "value" if "value" in f.columns else "usd_vnd"
    f = f[["date", value_col]].rename(columns={value_col: "usd_vnd"}).dropna()
    f = f.groupby("date", as_index=False)["usd_vnd"].last().sort_values("date")
    f["usd_vnd_60d_change"] = f["usd_vnd"].pct_change(60)
    return pd.merge_asof(out.sort_values("date"), f, on="date", direction="backward")


def _merge_interest(daily: pd.DataFrame, interest: pd.DataFrame | None) -> pd.DataFrame:
    out = daily.copy()
    if interest is None or interest.empty:
        out["interbank_overnight"] = pd.NA
        return out
    r = interest.copy()
    if "date" not in r.columns:
        r["date"] = pd.to_datetime(r.index).tz_localize(None).normalize()
    r["date"] = pd.to_datetime(r["date"]).dt.tz_localize(None).dt.normalize()
    if {"group_name", "name"}.issubset(r.columns):
        mask = (
            r["group_name"].astype(str).str.contains("Lãi suất|interest", case=False, regex=True)
            & r["name"].astype(str).str.contains("Qua đêm|overnight", case=False, regex=True)
        )
        if mask.any():
            r = r[mask]
    value_col = "value" if "value" in r.columns else "interbank_overnight"
    r = r[["date", value_col]].rename(columns={value_col: "interbank_overnight"}).dropna()
    r = r.groupby("date", as_index=False)["interbank_overnight"].last().sort_values("date")
    return pd.merge_asof(out.sort_values("date"), r, on="date", direction="backward")
