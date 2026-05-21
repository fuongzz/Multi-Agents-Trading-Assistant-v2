"""Build daily research feature tables from existing project artifacts."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta


def build_feature_table(
    universe: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    root: str | Path = ".",
    warmup_days: int = 370,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load OHLCV, Money Cycle, and Smart Money Trace into one daily table.

    Returns:
        (features, price_map)
    """
    root = Path(root)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    symbols = [s.upper() for s in universe]

    ohlcv = pd.read_parquet(root / "multiagents_trading_assistant/data/ohlcv_master.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.normalize()
    ohlcv["symbol"] = ohlcv["symbol"].astype(str).str.upper()
    ohlcv = ohlcv[
        ohlcv["symbol"].isin(symbols)
        & (ohlcv["date"] >= start_ts - pd.Timedelta(days=warmup_days))
        & (ohlcv["date"] <= end_ts)
    ].sort_values(["symbol", "date"]).reset_index(drop=True)

    chdm = _load_symbol_frame(root / "data/research/money_cycle/chdm_by_symbol.parquet", symbols)
    ds = _load_symbol_frame(root / "data/research/money_cycle/ds_by_symbol.parquet", symbols)
    smt = _load_symbol_frame(root / "data/research/smart_money_trace/smart_money_by_symbol.parquet", symbols)
    market = pd.read_parquet(root / "data/research/money_cycle/money_cycle_market.parquet")
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()

    smt_cols = [
        "date", "symbol", "smart_money_score", "smart_money_state",
        "rs_percentile_20", "distribution_days_10", "accumulation_days_10",
        "rs_score", "value_flow_quality_score", "clv_score",
        "accumulation_distribution_score", "pullback_quality_score",
        "sector_leadership_score", "clv_ma5",
        "ma20", "ma50", "ma200", "atr14", "cmf20", "mfi14",
        "ret_5d", "ret_20d", "value_ma20", "value_ratio_20",
        "volume_ma20", "volume_ratio_20", "pullback_depth_20",
    ]
    smt_cols = [c for c in smt_cols if c in smt.columns]

    ohlcv_cols = ["date", "symbol", "open", "high", "low", "close", "volume", "value"]
    if "industry" in ohlcv.columns:
        ohlcv_cols.append("industry")

    features = (
        ohlcv[ohlcv_cols]
        .merge(chdm, on=["date", "symbol"], how="left")
        .merge(ds, on=["date", "symbol"], how="left")
        .merge(smt[smt_cols], on=["date", "symbol"], how="left")
        .merge(market.add_prefix("mkt_").rename(columns={"mkt_date": "date"}), on="date", how="left")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    _add_benchmark_columns(features, root, start_ts - pd.Timedelta(days=warmup_days), end_ts)
    _add_derived_columns(features)

    price_map = {
        symbol: frame.reset_index(drop=True)
        for symbol, frame in ohlcv.groupby("symbol", sort=False)
    }
    return features, price_map


def _load_symbol_frame(path: Path, symbols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df[df["symbol"].isin(symbols)].copy()


def _add_derived_columns(features: pd.DataFrame) -> None:
    _add_smart_money_no_sector(features)
    group = features.groupby("symbol", sort=False)
    for col in ["DS20", "DS50", "CHDM03", "CHDM20", "CHDM50", "smart_money_score", "smart_money_score_no_sector"]:
        if col in features.columns:
            features[f"{col}_prev"] = group[col].shift(1)
            features[f"{col}_delta"] = features[col] - features[f"{col}_prev"]
    features["high20_prev"] = group["high"].transform(lambda item: item.rolling(20, min_periods=10).max().shift(1))
    features["high55_prev"] = group["high"].transform(lambda item: item.rolling(55, min_periods=25).max().shift(1))
    features["low20_prev"] = group["low"].transform(lambda item: item.rolling(20, min_periods=10).min().shift(1))
    features["low120_prev"] = group["low"].transform(lambda item: item.rolling(120, min_periods=60).min().shift(1))
    features["low250_prev"] = group["low"].transform(lambda item: item.rolling(250, min_periods=120).min().shift(1))
    features["ma20_slope_5"] = group["ma20"].diff(5) / group["ma20"].shift(5) if "ma20" in features.columns else pd.NA
    features["ma50_slope_10"] = group["ma50"].diff(10) / group["ma50"].shift(10) if "ma50" in features.columns else pd.NA
    features["ma200_slope_20"] = group["ma200"].diff(20) / group["ma200"].shift(20) if "ma200" in features.columns else pd.NA
    _add_market_derived_columns(features)
    features["above_ma50"] = features["close"] > features.get("ma50")
    features["above_ma20"] = features["close"] > features.get("ma20")
    ma200 = features.get("ma200", pd.Series(np.nan, index=features.index))
    features["above_ma200"] = features["close"] > ma200
    features["above_ma20_prev"] = group["above_ma20"].shift(1)
    features["distance_ma20"] = features["close"] / features.get("ma20") - 1.0
    features["distance_ma50"] = features["close"] / features.get("ma50") - 1.0
    features["distance_ma200"] = features["close"] / ma200.replace(0, pd.NA) - 1.0
    features["distance_low120"] = features["close"] / features["low120_prev"].replace(0, pd.NA) - 1.0
    features["distance_low250"] = features["close"] / features["low250_prev"].replace(0, pd.NA) - 1.0
    features["near_ma20"] = features["distance_ma20"].between(-0.03, 0.03)
    features["near_ma50"] = features["distance_ma50"].between(-0.04, 0.04)
    features["near_ma200_support"] = features["distance_ma200"].between(-0.035, 0.055)
    features["near_low120_support"] = features["distance_low120"].between(0.0, 0.075)
    features["near_low250_support"] = features["distance_low250"].between(0.0, 0.09)
    features["near_long_support"] = (
        features["near_ma200_support"]
        | features["near_low120_support"]
        | features["near_low250_support"]
    )
    support_hits = (
        features["near_ma200_support"].astype(float)
        + features["near_low120_support"].astype(float)
        + features["near_low250_support"].astype(float)
    )
    support_distance = pd.concat(
        [
            features["distance_ma200"].abs(),
            features["distance_low120"].abs(),
            features["distance_low250"].abs(),
        ],
        axis=1,
    ).min(axis=1)
    features["support_confluence_score"] = (
        35.0
        + 18.0 * support_hits
        + 18.0 * (1.0 - (support_distance / 0.10).clip(0.0, 1.0))
        + 12.0 * features["above_ma200"].fillna(False).astype(float)
        + 10.0 * (_feature_col(features, "ma200_slope_20", 0.0) >= -0.02).astype(float)
    ).clip(0.0, 100.0)
    features["reclaim_ma20"] = features["above_ma20"] & (features["above_ma20_prev"] == False)
    features["reclaim_ma50"] = features["above_ma50"] & (group["above_ma50"].shift(1) == False)
    features["breakout_20"] = features["close"] > features["high20_prev"]
    features["breakout_55"] = features["close"] > features["high55_prev"]
    _add_ichimoku_columns(features)
    _add_ta_columns(features)
    features["ret_1d"] = group["close"].pct_change()
    features["ret_3d"] = group["close"].pct_change(3)
    features["large_return_1d"] = features["ret_1d"].abs() > 0.30
    features["large_return_3d"] = group["large_return_1d"].transform(lambda item: item.rolling(3, min_periods=1).max()).astype(bool)
    features["ret_20d_local"] = group["close"].pct_change(20)
    features["ret_60d_local"] = group["close"].pct_change(60)
    features["excess_ret_20d"] = features["ret_20d_local"] - _feature_col(features, "vni_ret_20d", 0.0)
    features["excess_ret_60d"] = features["ret_60d_local"] - _feature_col(features, "vni_ret_60d", 0.0)
    features["excess_ret_20d_pctile"] = features.groupby("date", sort=False)["excess_ret_20d"].rank(pct=True)
    features["excess_ret_60d_pctile"] = features.groupby("date", sort=False)["excess_ret_60d"].rank(pct=True)
    features["data_quality_ok"] = ~features["large_return_3d"].fillna(False)
    features["range_pct"] = (features["high"] - features["low"]) / features["close"].replace(0, pd.NA)
    features["body_pct"] = (features["close"] - features["open"]).abs() / features["open"].replace(0, pd.NA)
    intraday_range = (features["high"] - features["low"]).replace(0, pd.NA)
    features["close_location"] = ((features["close"] - features["low"]) / intraday_range).clip(0.0, 1.0)
    features["lower_wick_pct"] = (
        features[["open", "close"]].min(axis=1) - features["low"]
    ) / features["close"].replace(0, pd.NA)
    features["upper_wick_pct"] = (
        features["high"] - features[["open", "close"]].max(axis=1)
    ) / features["close"].replace(0, pd.NA)
    features["range_pct_min7"] = group["range_pct"].transform(lambda item: item.rolling(7, min_periods=5).min().shift(1))
    features["range_pct_q20"] = group["range_pct"].transform(lambda item: item.rolling(20, min_periods=10).quantile(0.25).shift(1))
    features["nr7"] = features["range_pct"] <= features["range_pct_min7"]
    features["tight_range_20"] = features["range_pct"] <= features["range_pct_q20"]
    recent_low_vs_ma20 = group["low"].transform(lambda item: item.rolling(5, min_periods=2).min()) < features.get("ma20")
    features["reclaim_ma20_after_pullback"] = features["reclaim_ma20"] & recent_low_vs_ma20
    features["spring_20"] = (
        (features["low"] <= features["low20_prev"] * 1.02)
        & (features["close"] > features["low20_prev"])
        & (features["close_location"] >= 0.60)
    )
    features["hammer_like"] = (
        (features["close_location"] >= 0.60)
        & (features["lower_wick_pct"] >= features["body_pct"] * 1.5)
        & (features["body_pct"] <= 0.035)
    )
    features["breakout_20_quality"] = (
        features["breakout_20"]
        & features["distance_ma20"].between(0.0, 0.12)
        & (_feature_col(features, "value_ratio_20", 1.0) >= 1.10)
    )
    features["inside_bar_breakout"] = (
        (group["high"].shift(1) <= group["high"].shift(2))
        & (group["low"].shift(1) >= group["low"].shift(2))
        & (features["close"] > group["high"].shift(1))
        & (_feature_col(features, "volume_ratio_20", 1.0) >= 1.05)
    )
    breakout_level = group["high20_prev"].shift(3)
    features["retest_breakout_level"] = (
        breakout_level.notna()
        & ((features["close"] / breakout_level - 1.0).between(-0.03, 0.05))
        & (features["close"] >= breakout_level * 0.97)
        & (_feature_col(features, "volume_ratio_20", 1.0) <= 1.10)
    )
    features["linreg_slope_14_pct"] = group["linreg_14"].diff(5) / features["close"].replace(0, pd.NA)
    features["adx_bullish"] = features["dmp_14"] > features["dmn_14"] * 1.05
    features["rsi_recovering"] = features["rsi14"] > group["rsi14"].shift(1)
    _add_pattern_columns(features)
    _add_sector_derived_columns(features)
    _add_flow_v2_columns(features)
    features["edge_score"] = _edge_score(features)
    features["edge_score_no_sector"] = _edge_score(features, smart_money_col="smart_money_score_no_sector")


def _add_ichimoku_columns(features: pd.DataFrame) -> None:
    """Add causal Ichimoku features.

    Senkou A/B are shifted forward on charts, so the cloud visible at bar T is
    the raw cloud value computed 26 bars earlier. The raw current cloud color
    is still causal and represents the cloud projected into the future.
    """
    group = features.groupby("symbol", sort=False)
    high = features["high"]
    low = features["low"]
    close = features["close"]

    tenkan = (
        group["high"].transform(lambda item: item.rolling(9, min_periods=9).max())
        + group["low"].transform(lambda item: item.rolling(9, min_periods=9).min())
    ) / 2.0
    kijun = (
        group["high"].transform(lambda item: item.rolling(26, min_periods=26).max())
        + group["low"].transform(lambda item: item.rolling(26, min_periods=26).min())
    ) / 2.0
    senkou_a_raw = (tenkan + kijun) / 2.0
    senkou_b_raw = (
        group["high"].transform(lambda item: item.rolling(52, min_periods=52).max())
        + group["low"].transform(lambda item: item.rolling(52, min_periods=52).min())
    ) / 2.0
    senkou_a = senkou_a_raw.groupby(features["symbol"], sort=False).shift(26)
    senkou_b = senkou_b_raw.groupby(features["symbol"], sort=False).shift(26)
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    cloud_width = (cloud_top - cloud_bot) / close.replace(0, pd.NA)

    features["ich_tenkan"] = tenkan
    features["ich_kijun"] = kijun
    features["ich_senkou_a"] = senkou_a
    features["ich_senkou_b"] = senkou_b
    features["ich_cloud_top"] = cloud_top
    features["ich_cloud_bot"] = cloud_bot
    features["ich_cloud_width"] = cloud_width
    features["ich_future_green"] = senkou_a_raw > senkou_b_raw
    features["ich_current_green"] = senkou_a > senkou_b
    features["ich_price_above_cloud"] = close > cloud_top
    features["ich_price_below_cloud"] = close < cloud_bot
    features["ich_tk_bull"] = tenkan > kijun
    features["ich_tk_cross_bull"] = features["ich_tk_bull"] & (group["ich_tk_bull"].shift(1) == False)
    features["ich_chikou_free"] = close > group["close"].shift(26)
    features["ich_kijun_distance"] = close / kijun.replace(0, pd.NA) - 1.0
    features["ich_tenkan_distance"] = close / tenkan.replace(0, pd.NA) - 1.0

    aligned = (
        features["ich_tk_bull"]
        & (close > tenkan)
        & features["ich_price_above_cloud"]
        & features["ich_current_green"]
        & features["ich_future_green"]
        & features["ich_chikou_free"]
    )
    features["ich_perfect_bullish"] = aligned
    features["ich_perfect_bullish_signal"] = aligned & (group["ich_perfect_bullish"].shift(1) == False)
    features["ich_kijun_bounce"] = (
        features["ich_price_above_cloud"]
        & (low <= kijun * 1.01)
        & (close > kijun)
        & (close > features["open"])
    )
    features["ich_no_chase"] = features["ich_kijun_distance"].between(-0.02, 0.07)
    features["ich_strong_alignment"] = (
        features["ich_perfect_bullish"]
        & features["ich_no_chase"]
        & features["ich_cloud_width"].between(0.005, 0.12)
    )


def _add_smart_money_no_sector(features: pd.DataFrame) -> None:
    """SMT score with sector leadership removed and remaining weights normalized."""
    component_cols = [
        "rs_score",
        "value_flow_quality_score",
        "clv_score",
        "accumulation_distribution_score",
        "pullback_quality_score",
    ]
    if all(col in features.columns for col in component_cols):
        features["smart_money_score_no_sector"] = (
            0.25 * _feature_col(features, "rs_score", 50.0)
            + 0.20 * _feature_col(features, "value_flow_quality_score", 50.0)
            + 0.20 * _feature_col(features, "clv_score", 50.0)
            + 0.15 * _feature_col(features, "accumulation_distribution_score", 50.0)
            + 0.10 * _feature_col(features, "pullback_quality_score", 50.0)
        ) / 0.90
    else:
        features["smart_money_score_no_sector"] = (
            _feature_col(features, "smart_money_score", 50.0)
            - 0.10 * _feature_col(features, "sector_leadership_score", 50.0)
        ) / 0.90
    features["smart_money_score_no_sector"] = features["smart_money_score_no_sector"].clip(0.0, 100.0)


def _add_flow_v2_columns(features: pd.DataFrame) -> None:
    """Add decomposed flow scores while preserving the legacy SMT score.

    The legacy ``smart_money_score`` intentionally blends RS, sector, pullback,
    and flow proxies. V2 columns separate these dimensions so strategy research
    can test pure flow and distribution pressure without mutating old configs.
    """
    value_ratio = _feature_col(features, "value_ratio_20", 1.0).fillna(1.0)
    volume_ratio = _feature_col(features, "volume_ratio_20", 1.0).fillna(1.0)
    clv_ma5 = _feature_col(features, "clv_ma5", 0.5).fillna(0.5)
    cmf20 = _feature_col(features, "cmf20", 0.0).fillna(0.0)
    mfi14 = _feature_col(features, "mfi14", 50.0).fillna(50.0)
    acc10 = _feature_col(features, "accumulation_days_10", 0.0).fillna(0.0)
    dist10 = _feature_col(features, "distribution_days_10", 0.0).fillna(0.0)
    ret1 = _feature_col(features, "ret_1d", 0.0).fillna(0.0)
    close_location = _feature_col(features, "close_location", 0.5).fillna(0.5)
    upper_wick = _feature_col(features, "upper_wick_pct", 0.0).fillna(0.0)
    pullback_depth = _feature_col(features, "pullback_depth_20", 0.0).fillna(0.0)
    ds05 = _feature_col(features, "DS05", 0.0).fillna(0.0)
    ds20 = _feature_col(features, "DS20", 0.0).fillna(0.0)
    ds50 = _feature_col(features, "DS50", 0.0).fillna(0.0)
    chdm20 = _feature_col(features, "CHDM20", 50.0).fillna(50.0)
    chdm50 = _feature_col(features, "CHDM50", 50.0).fillna(50.0)
    rs20 = _feature_col(features, "rs_percentile_20", 0.5).fillna(0.5)
    sector = _feature_col(features, "sector_leadership_score", 50.0).fillna(50.0)

    value_intensity = ((value_ratio.clip(0.5, 2.5) - 0.5) / 2.0 * 100.0).clip(0.0, 100.0)
    volume_intensity = ((volume_ratio.clip(0.5, 2.5) - 0.5) / 2.0 * 100.0).clip(0.0, 100.0)
    clv_strength = (clv_ma5.clip(0.0, 1.0) * 100.0).clip(0.0, 100.0)
    cmf_strength = ((cmf20.clip(-0.25, 0.25) + 0.25) / 0.50 * 100.0).clip(0.0, 100.0)
    mfi_strength = mfi14.clip(0.0, 100.0)
    ad_strength = (50.0 + 12.0 * (acc10 - dist10)).clip(0.0, 100.0)

    features["flow_pure_score"] = (
        0.25 * value_intensity
        + 0.20 * clv_strength
        + 0.20 * cmf_strength
        + 0.15 * mfi_strength
        + 0.20 * ad_strength
    ).clip(0.0, 100.0)

    persistent_value = (
        features.groupby("symbol", sort=False)["value_ratio_20"]
        .transform(lambda item: (item > 1.0).astype(float).rolling(5, min_periods=3).sum())
        if "value_ratio_20" in features.columns
        else pd.Series(0.0, index=features.index)
    ).fillna(0.0)
    features["flow_sponsorship_score"] = (
        0.30 * value_intensity
        + 0.20 * volume_intensity
        + 0.20 * clv_strength
        + 0.15 * (persistent_value / 5.0 * 100.0).clip(0.0, 100.0)
        + 0.15 * (rs20 * 100.0)
    ).clip(0.0, 100.0)

    absorption_context = (
        0.30 * value_intensity
        + 0.25 * clv_strength
        + 0.20 * cmf_strength
        + 0.15 * (100.0 - (ds05 * 100.0))
        + 0.10 * (100.0 - (ds20 * 100.0))
    )
    reset_bonus = 15.0 * pullback_depth.between(-0.15, -0.03).astype(float)
    features["flow_absorption_score"] = (absorption_context + reset_bonus).clip(0.0, 100.0)

    red_high_value = ((value_ratio >= 1.2) & (ret1 < 0)).astype(float)
    weak_close = (close_location <= 0.35).astype(float)
    supply_wick = (upper_wick >= 0.025).astype(float)
    features["distribution_pressure_score"] = (
        20.0
        + 18.0 * dist10.clip(0.0, 3.0)
        + 18.0 * red_high_value
        + 14.0 * weak_close
        + 10.0 * supply_wick
        + 10.0 * ds05
        + 10.0 * ds20
    ).clip(0.0, 100.0)

    # Long DS often behaves like a reset/lifecycle input, not a short-term veto.
    long_reset = (0.50 * ds50 + 0.50 * (chdm50.between(35, 70).astype(float))) * 100.0
    short_reset = (100.0 - (chdm20 - 45.0).abs().clip(0.0, 45.0) / 45.0 * 100.0)
    features["stock_lifecycle_score"] = (
        0.35 * chdm50.clip(0.0, 100.0)
        + 0.25 * short_reset.clip(0.0, 100.0)
        + 0.20 * long_reset.clip(0.0, 100.0)
        + 0.20 * (100.0 - features["distribution_pressure_score"])
    ).clip(0.0, 100.0)

    features["sector_cycle_score"] = sector.clip(0.0, 100.0)
    features["flow_quality_v2_score"] = (
        0.35 * features["flow_pure_score"]
        + 0.25 * features["flow_sponsorship_score"]
        + 0.20 * features["flow_absorption_score"]
        + 0.10 * features["stock_lifecycle_score"]
        + 0.10 * (100.0 - features["distribution_pressure_score"])
    ).clip(0.0, 100.0)


def _add_benchmark_columns(features: pd.DataFrame, root: Path, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Attach causal VNINDEX return context from canonical parquet or cache."""
    vnindex = _load_vnindex_history(root, start, end)
    if vnindex.empty:
        for col in ["vni_close", "vni_ret_20d", "vni_ret_60d", "vni_above_ma20", "vni_above_ma50"]:
            features[col] = np.nan
        return

    vnindex = vnindex.sort_values("date").copy()
    vnindex["vni_close"] = vnindex["close"]
    vnindex["vni_ret_20d"] = vnindex["close"].pct_change(20)
    vnindex["vni_ret_60d"] = vnindex["close"].pct_change(60)
    vnindex["vni_ma20"] = vnindex["close"].rolling(20, min_periods=10).mean()
    vnindex["vni_ma50"] = vnindex["close"].rolling(50, min_periods=25).mean()
    vnindex["vni_above_ma20"] = vnindex["close"] > vnindex["vni_ma20"]
    vnindex["vni_above_ma50"] = vnindex["close"] > vnindex["vni_ma50"]
    keep = ["date", "vni_close", "vni_ret_20d", "vni_ret_60d", "vni_above_ma20", "vni_above_ma50"]
    merged = features[["date"]].merge(vnindex[keep], on="date", how="left")
    for col in keep:
        if col != "date":
            features[col] = merged[col].to_numpy()


def _load_vnindex_history(root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    index_master = root / "multiagents_trading_assistant" / "data" / "index_master.parquet"
    if index_master.exists():
        try:
            frame = pd.read_parquet(index_master)
            if not frame.empty and "date" in frame.columns and "close" in frame.columns:
                frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
                if "symbol" in frame.columns:
                    frame = frame[frame["symbol"].astype(str).str.upper() == "VNINDEX"]
                frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
                if not frame.empty:
                    return frame[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    cache_dir = root / "multiagents_trading_assistant" / "cache"
    candidates = sorted(cache_dir.glob("VNINDEX_*_1D_hist.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            frame = pd.read_json(path)
            if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
                continue
            frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
            frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
            if not frame.empty:
                return frame[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        except Exception:
            continue
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])


def _add_sector_derived_columns(features: pd.DataFrame) -> None:
    if "industry" not in features.columns:
        _add_cross_sectional_leadership_fallback(features)
        return

    industry = features["industry"].fillna("Unknown").astype(str)
    valid_industry = ~industry.str.upper().isin({"", "UNKNOWN", "NAN", "NONE"})
    if valid_industry.mean() < 0.20:
        _add_cross_sectional_leadership_fallback(features)
        return

    sector_keys = [features["date"], industry]
    features["sector_excess_ret_20d"] = features.groupby(sector_keys, sort=False)["excess_ret_20d"].transform("median").where(valid_industry)
    features["sector_excess_ret_60d"] = features.groupby(sector_keys, sort=False)["excess_ret_60d"].transform("median").where(valid_industry)
    features["sector_value_ratio_20"] = features.groupby(sector_keys, sort=False)["value_ratio_20"].transform("median").where(valid_industry)
    above_ma20 = features.get("above_ma20", pd.Series(False, index=features.index)).astype(float)
    features["sector_breadth_ma20"] = above_ma20.groupby([features["date"], industry], sort=False).transform("mean").where(valid_industry)

    sector_daily = (
        features.loc[valid_industry, ["date", "industry", "sector_excess_ret_20d"]]
        .drop_duplicates(["date", "industry"])
        .copy()
    )
    if sector_daily.empty:
        features["sector_rs_rank_20d"] = np.nan
    else:
        sector_daily["sector_rs_rank_20d"] = sector_daily.groupby("date", sort=False)["sector_excess_ret_20d"].rank(pct=True)
        enriched = features[["date", "industry"]].merge(sector_daily[["date", "industry", "sector_rs_rank_20d"]], on=["date", "industry"], how="left")
        features["sector_rs_rank_20d"] = enriched["sector_rs_rank_20d"].to_numpy()

    sector_rs = 100.0 * _feature_col(features, "sector_rs_rank_20d", 0.5).clip(0.0, 1.0)
    breadth = 100.0 * _feature_col(features, "sector_breadth_ma20", 0.5).clip(0.0, 1.0)
    sector_flow = 50.0 + 25.0 * (_feature_col(features, "sector_value_ratio_20", 1.0).clip(0.0, 3.0) - 1.0)
    features["sector_leadership_score"] = (0.45 * sector_rs + 0.35 * breadth + 0.20 * sector_flow).clip(0.0, 100.0)


def _add_cross_sectional_leadership_fallback(features: pd.DataFrame) -> None:
    """Fallback when industry metadata is unavailable in local OHLCV cache."""
    features["sector_excess_ret_20d"] = features.groupby("date", sort=False)["excess_ret_20d"].transform("median")
    features["sector_excess_ret_60d"] = features.groupby("date", sort=False)["excess_ret_60d"].transform("median")
    features["sector_rs_rank_20d"] = _feature_col(features, "excess_ret_20d_pctile", 0.5)
    above_ma20 = features.get("above_ma20", pd.Series(False, index=features.index)).astype(float)
    features["sector_breadth_ma20"] = above_ma20.groupby(features["date"], sort=False).transform("mean")
    features["sector_value_ratio_20"] = features.groupby("date", sort=False)["value_ratio_20"].transform("median")
    rs = 100.0 * _feature_col(features, "excess_ret_20d_pctile", 0.5).clip(0.0, 1.0)
    rs60 = 100.0 * _feature_col(features, "excess_ret_60d_pctile", 0.5).clip(0.0, 1.0)
    breadth = 100.0 * _feature_col(features, "sector_breadth_ma20", 0.5).clip(0.0, 1.0)
    flow = 50.0 + 25.0 * (_feature_col(features, "sector_value_ratio_20", 1.0).clip(0.0, 3.0) - 1.0)
    features["sector_leadership_score"] = (0.45 * rs + 0.25 * rs60 + 0.20 * breadth + 0.10 * flow).clip(0.0, 100.0)


def _add_pattern_columns(features: pd.DataFrame) -> None:
    """Candlestick patterns, oscillator cross signals, and volatility regime features."""
    g = features.groupby("symbol", sort=False)
    close = features["close"]
    open_ = features["open"]

    # ── Stochastic cross signals ──────────────────────────────────────────────
    if "stoch_k" in features.columns:
        stoch_k = features["stoch_k"].fillna(50.0)
        stoch_d = features["stoch_d"].fillna(50.0)
        stoch_k_prev = g["stoch_k"].shift(1).fillna(50.0)
        stoch_d_prev = g["stoch_d"].shift(1).fillna(50.0)
        features["stoch_cross_up"] = (
            (stoch_k > stoch_d) & (stoch_k_prev <= stoch_d_prev) & (stoch_k < 80)
        ).fillna(False)
        features["stoch_oversold_cross"] = (
            features["stoch_cross_up"] & (stoch_k_prev < 30)
        ).fillna(False)
    else:
        features["stoch_cross_up"] = False
        features["stoch_oversold_cross"] = False

    # ── CCI momentum / oversold bounce ───────────────────────────────────────
    if "cci_14" in features.columns:
        cci = features["cci_14"].fillna(0.0)
        cci_prev = g["cci_14"].shift(1).fillna(0.0)
        features["cci_momentum"] = (cci > 100).fillna(False)
        features["cci_oversold_bounce"] = ((cci > -100) & (cci_prev <= -100)).fillna(False)
    else:
        features["cci_momentum"] = False
        features["cci_oversold_bounce"] = False

    # ── Parabolic SAR flip ───────────────────────────────────────────────────
    if "psar_bullish" in features.columns:
        psar_bull_prev = g["psar_bullish"].shift(1).fillna(True)
        features["psar_flip_bull"] = (
            features["psar_bullish"].fillna(False) & (~psar_bull_prev.astype(bool))
        ).fillna(False)
    else:
        features["psar_bullish"] = False
        features["psar_flip_bull"] = False

    # ── Three White Soldiers ──────────────────────────────────────────────────
    c1_open = g["open"].shift(2)
    c1_close = g["close"].shift(2)
    c2_open = g["open"].shift(1)
    c2_close = g["close"].shift(1)
    safe_open = open_.replace(0, pd.NA)
    features["three_white_soldiers"] = (
        (c1_close > c1_open)
        & (c2_close > c2_open)
        & (close > open_)
        & ((c1_close - c1_open).abs() / c1_open.replace(0, pd.NA) >= 0.01)
        & ((c2_close - c2_open).abs() / c2_open.replace(0, pd.NA) >= 0.01)
        & ((close - open_).abs() / safe_open >= 0.01)
        & (c2_close > c1_close)
        & (close > c2_close)
    ).fillna(False)

    # ── Morning Star ─────────────────────────────────────────────────────────
    c1_midpoint = (c1_open + c1_close) / 2.0
    features["morning_star"] = (
        (c1_close < c1_open)
        & ((c1_open - c1_close) / c1_open.replace(0, pd.NA) >= 0.01)
        & ((c2_close - c2_open).abs() / c2_open.replace(0, pd.NA) < 0.005)
        & ((close - open_) / safe_open >= 0.01)
        & (close > c1_midpoint)
    ).fillna(False)

    # ── Doji at support ──────────────────────────────────────────────────────
    features["doji_candle"] = (features["body_pct"] < 0.003).fillna(False)
    features["doji_at_support"] = (
        features["doji_candle"]
        & features["distance_ma20"].between(-0.08, 0.02)
        & (close <= features["low20_prev"] * 1.05)
    ).fillna(False)

    # ── Volume climax reversal ────────────────────────────────────────────────
    vol_ratio = _feature_col(features, "volume_ratio_20", 1.0)
    features["vol_climax"] = (vol_ratio >= 2.5).fillna(False)
    features["vol_climax_reversal"] = (
        features["vol_climax"]
        & (features["lower_wick_pct"] >= 0.015)
        & (features["close_location"] >= 0.50)
    ).fillna(False)

    # ── Triple MA alignment ──────────────────────────────────────────────────
    ma200 = features.get("ma200")
    above_ma200 = (close > ma200).fillna(False) if ma200 is not None else pd.Series(False, index=features.index)
    features["triple_ma_align"] = (
        features["above_ma20"].fillna(False)
        & features["above_ma50"].fillna(False)
        & above_ma200
    ).fillna(False)
    features["triple_ma_pullback"] = (
        features["triple_ma_align"]
        & features["distance_ma20"].between(-0.03, 0.05)
    ).fillna(False)

    # ── Volatility squeeze break ──────────────────────────────────────────────
    if "atr14" in features.columns:
        atr = features["atr14"].fillna(method="ffill")
        atr_rolling_min = g["atr14"].transform(lambda s: s.rolling(10, min_periods=5).min().shift(1))
        features["atr_ratio"] = (atr / atr_rolling_min.replace(0, pd.NA)).fillna(1.0).clip(0.5, 5.0)
        atr_ratio_lag3 = g["atr_ratio"].shift(3).fillna(1.0)
        features["volatility_squeeze_break"] = (
            (atr_ratio_lag3 <= 1.15)
            & (features["atr_ratio"] >= 1.30)
            & features["above_ma20"].fillna(False)
        ).fillna(False)
    else:
        features["atr_ratio"] = 1.0
        features["volatility_squeeze_break"] = False

    # ── Kalman-filtered strong uptrend ────────────────────────────────────────
    kalman_trend = _feature_col(features, "kalman_trend_5d", 0.0)
    kalman_resid = _feature_col(features, "kalman_residual_pct", 0.0)
    kalman_conf = _feature_col(features, "kalman_confidence", 0.5)
    features["kalman_strong_uptrend"] = (
        (kalman_trend >= 0.015)
        & kalman_resid.between(-0.04, 0.03)
        & (kalman_conf >= 0.55)
    ).fillna(False)


def _add_ta_columns(features: pd.DataFrame) -> None:
    for _, idx in features.groupby("symbol", sort=False).groups.items():
        frame = features.loc[idx]
        if len(frame) < 20:
            continue  # too few bars (new listing) — skip TA, leave NaN
        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        rsi = ta.rsi(close, length=14)
        if rsi is None:
            continue
        features.loc[idx, "rsi14"] = rsi.to_numpy()
        features.loc[idx, "linreg_14"] = ta.linreg(close, length=14).to_numpy()
        kalman = _kalman_price_trend(close)
        features.loc[idx, "kalman_close"] = kalman["kalman_close"].to_numpy()
        features.loc[idx, "kalman_trend_5d"] = kalman["kalman_trend_5d"].to_numpy()
        features.loc[idx, "kalman_residual_pct"] = kalman["kalman_residual_pct"].to_numpy()
        adaptive = _kalman_adaptive_price_gate(frame)
        features.loc[idx, "kalman_adaptive_close"] = adaptive["kalman_adaptive_close"].to_numpy()
        features.loc[idx, "kalman_adaptive_trend_5d"] = adaptive["kalman_adaptive_trend_5d"].to_numpy()
        features.loc[idx, "kalman_adaptive_residual_pct"] = adaptive["kalman_adaptive_residual_pct"].to_numpy()
        features.loc[idx, "kalman_confidence"] = adaptive["kalman_confidence"].to_numpy()
        features.loc[idx, "kalman_shock_score"] = adaptive["kalman_shock_score"].to_numpy()
        adx = ta.adx(high, low, close, length=14)
        if adx is not None and not adx.empty:
            features.loc[idx, "adx_14"] = _prefixed_col(adx, "ADX_").to_numpy()
            features.loc[idx, "dmp_14"] = _prefixed_col(adx, "DMP_").to_numpy()
            features.loc[idx, "dmn_14"] = _prefixed_col(adx, "DMN_").to_numpy()

        stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
        if stoch is not None and not stoch.empty:
            features.loc[idx, "stoch_k"] = _prefixed_col(stoch, "STOCHk_").to_numpy()
            features.loc[idx, "stoch_d"] = _prefixed_col(stoch, "STOCHd_").to_numpy()

        cci_series = ta.cci(high, low, close, length=14)
        if cci_series is not None:
            features.loc[idx, "cci_14"] = cci_series.to_numpy()

        psar_df = ta.psar(high, low, close)
        if psar_df is not None and not psar_df.empty:
            psar_long = _prefixed_col(psar_df, "PSARl_")
            features.loc[idx, "psar_long"] = psar_long.to_numpy()
            features.loc[idx, "psar_bullish"] = psar_long.notna().to_numpy()


def _kalman_price_trend(close: pd.Series) -> pd.DataFrame:
    """Causal local-linear Kalman estimate of price level and short trend.

    The state is log price plus one-bar log slope. This keeps the estimate
    positive after exponentiation and makes slope comparable across symbols.
    """
    values = pd.to_numeric(close, errors="coerce").astype(float)
    out = pd.DataFrame(index=values.index)
    if values.dropna().empty:
        out["kalman_close"] = np.nan
        out["kalman_trend_5d"] = np.nan
        out["kalman_residual_pct"] = np.nan
        return out

    log_price = np.log(values.where(values > 0))
    valid = log_price.dropna()
    returns = valid.diff().dropna()
    observation_var = float(returns.rolling(20, min_periods=5).var().median())
    if not np.isfinite(observation_var) or observation_var <= 0:
        observation_var = 1e-4
    process_level_var = observation_var * 0.02
    process_slope_var = observation_var * 0.002

    state = np.array([float(valid.iloc[0]), 0.0], dtype=float)
    covariance = np.eye(2) * observation_var
    transition = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
    observation = np.array([1.0, 0.0], dtype=float)
    process_noise = np.diag([process_level_var, process_slope_var])

    levels: list[float] = []
    slopes: list[float] = []
    residuals: list[float] = []
    for raw_price, z in zip(values.to_numpy(), log_price.to_numpy()):
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_noise

        if np.isfinite(z):
            innovation = z - float(observation @ state)
            innovation_var = float(observation @ covariance @ observation.T + observation_var)
            gain = covariance @ observation / innovation_var
            state = state + gain * innovation
            covariance = (np.eye(2) - np.outer(gain, observation)) @ covariance

        level_price = float(np.exp(state[0]))
        levels.append(level_price)
        slopes.append(float(np.expm1(state[1]) * 5.0))
        residuals.append(float(raw_price / level_price - 1.0) if np.isfinite(raw_price) and level_price > 0 else np.nan)

    out["kalman_close"] = levels
    out["kalman_trend_5d"] = slopes
    out["kalman_residual_pct"] = residuals
    return out


def _kalman_adaptive_price_gate(frame: pd.DataFrame) -> pd.DataFrame:
    """Adaptive robust Kalman features tuned for VN daily stock data.

    The filter trusts price less on noisy/illiquid/shock bars and lets the
    latent trend move faster only when market and symbol context are healthy.
    """
    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    out = pd.DataFrame(index=frame.index)
    empty_cols = [
        "kalman_adaptive_close",
        "kalman_adaptive_trend_5d",
        "kalman_adaptive_residual_pct",
        "kalman_confidence",
        "kalman_shock_score",
    ]
    if close.dropna().empty:
        for col in empty_cols:
            out[col] = np.nan
        return out

    log_price = np.log(close.where(close > 0))
    valid = log_price.dropna()
    returns = valid.diff().dropna()
    base_var = float(returns.rolling(20, min_periods=5).var().median())
    if not np.isfinite(base_var) or base_var <= 0:
        base_var = 1e-4

    high = pd.to_numeric(frame.get("high", close), errors="coerce").astype(float)
    low = pd.to_numeric(frame.get("low", close), errors="coerce").astype(float)
    open_ = pd.to_numeric(frame.get("open", close), errors="coerce").astype(float)
    volume = pd.to_numeric(frame.get("volume", pd.Series(np.nan, index=frame.index)), errors="coerce").astype(float)
    value_ratio = _series_or_default(frame, "value_ratio_20", 1.0).astype(float)
    volume_ratio = _series_or_default(frame, "volume_ratio_20", 1.0).astype(float)
    smart_delta = _series_or_default(frame, "smart_money_score_delta", 0.0).astype(float)
    chdm50 = _series_or_default(frame, "CHDM50", 50.0).astype(float)
    ds20 = _series_or_default(frame, "DS20", 0.5).astype(float)
    mkt_score = _series_or_default(frame, "mkt_regime_score", 50.0).astype(float)

    range_pct = ((high - low) / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    rolling_range = range_pct.rolling(20, min_periods=5).median().replace(0, np.nan)
    range_noise = (range_pct / rolling_range).clip(lower=0.5, upper=3.0).fillna(1.0)
    gap_noise = ((open_ / close.shift(1) - 1.0).abs() / 0.03).clip(lower=0.0, upper=2.0).fillna(0.0)
    liquidity_noise = (1.0 / volume_ratio.replace(0, np.nan)).clip(lower=0.7, upper=2.0).fillna(1.0)
    value_noise = (1.0 / value_ratio.replace(0, np.nan)).clip(lower=0.7, upper=2.0).fillna(1.0)

    context_strength = (
        0.35 * _clip01((mkt_score - 45.0) / 25.0)
        + 0.25 * _clip01((chdm50 - 40.0) / 35.0)
        + 0.20 * (1.0 - _clip01((ds20 - 0.30) / 0.35))
        + 0.20 * _clip01((smart_delta + 3.0) / 8.0)
    ).fillna(0.5)

    state = np.array([float(valid.iloc[0]), 0.0], dtype=float)
    covariance = np.eye(2) * base_var
    transition = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
    observation = np.array([1.0, 0.0], dtype=float)

    levels: list[float] = []
    slopes: list[float] = []
    residuals: list[float] = []
    confidences: list[float] = []
    shock_scores: list[float] = []

    for i, (raw_price, z) in enumerate(zip(close.to_numpy(), log_price.to_numpy())):
        strength = float(context_strength.iloc[i]) if i < len(context_strength) else 0.5
        q_scale = 0.35 + 1.35 * strength
        process_noise = np.diag([base_var * 0.015 * q_scale, base_var * 0.0015 * q_scale])

        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_noise

        obs_noise_scale = (
            float(range_noise.iloc[i])
            * float(liquidity_noise.iloc[i])
            * float(value_noise.iloc[i])
            * (1.0 + 0.35 * float(gap_noise.iloc[i]))
        )
        observation_var = base_var * obs_noise_scale
        shock_score = 0.0

        if np.isfinite(z):
            innovation = z - float(observation @ state)
            innovation_std = float(np.sqrt(max(observation @ covariance @ observation.T + observation_var, 1e-12)))
            shock_score = abs(innovation) / innovation_std if innovation_std > 0 else 0.0
            clipped_innovation = float(np.clip(innovation, -3.0 * innovation_std, 3.0 * innovation_std))
            innovation_var = float(observation @ covariance @ observation.T + observation_var)
            gain = covariance @ observation / innovation_var
            state = state + gain * clipped_innovation
            covariance = (np.eye(2) - np.outer(gain, observation)) @ covariance

        level_price = float(np.exp(state[0]))
        confidence = 1.0 / (1.0 + obs_noise_scale + max(0.0, shock_score - 1.0) * 0.5)
        confidence = float(np.clip(2.2 * confidence, 0.0, 1.0))
        levels.append(level_price)
        slopes.append(float(np.expm1(state[1]) * 5.0))
        residuals.append(float(raw_price / level_price - 1.0) if np.isfinite(raw_price) and level_price > 0 else np.nan)
        confidences.append(confidence)
        shock_scores.append(float(shock_score))

    out["kalman_adaptive_close"] = levels
    out["kalman_adaptive_trend_5d"] = slopes
    out["kalman_adaptive_residual_pct"] = residuals
    out["kalman_confidence"] = confidences
    out["kalman_shock_score"] = shock_scores
    return out


def _series_or_default(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index)


def _prefixed_col(frame: pd.DataFrame, prefix: str) -> pd.Series:
    col = next((item for item in frame.columns if item.startswith(prefix)), None)
    if col is None:
        return pd.Series(pd.NA, index=frame.index)
    return frame[col]


def _add_market_derived_columns(features: pd.DataFrame) -> None:
    market_cols = [col for col in ["mkt_DS20", "mkt_DS50", "mkt_CHDM20", "mkt_CHDM50"] if col in features.columns]
    if not market_cols:
        return

    market = features[["date", *market_cols]].drop_duplicates("date").sort_values("date").copy()
    for col in market_cols:
        market[f"{col}_prev"] = market[col].shift(1)
        market[f"{col}_delta"] = market[col] - market[f"{col}_prev"]

    market["mkt_regime_score"] = _market_regime_score(market)
    market["mkt_regime_state"] = pd.cut(
        market["mkt_regime_score"],
        bins=[-1, 40, 60, 101],
        labels=["RISK_OFF", "NEUTRAL", "RISK_ON"],
    ).astype(str)
    market["mkt_regime_exposure"] = market["mkt_regime_state"].map(
        {"RISK_ON": 1.0, "NEUTRAL": 0.7, "RISK_OFF": 0.25}
    ).astype(float)

    added_cols = [
        "date",
        *(f"{col}_prev" for col in market_cols),
        *(f"{col}_delta" for col in market_cols),
        "mkt_regime_score",
        "mkt_regime_state",
        "mkt_regime_exposure",
    ]
    features.drop(columns=[col for col in added_cols if col != "date" and col in features.columns], inplace=True)
    enriched = features[["date"]].merge(market[added_cols], on="date", how="left")
    for col in enriched.columns:
        if col != "date":
            features[col] = enriched[col].to_numpy()


def _market_regime_score(market: pd.DataFrame) -> pd.Series:
    chdm20 = _clip01((_market_col(market, "mkt_CHDM20", 0.0) - 35.0) / 30.0)
    ds20 = 1.0 - _clip01((_market_col(market, "mkt_DS20", 1.0) - 0.25) / 0.40)
    chdm50 = _clip01((_market_col(market, "mkt_CHDM50", 0.0) - 30.0) / 35.0)
    ds50 = 1.0 - _clip01((_market_col(market, "mkt_DS50", 1.0) - 0.30) / 0.35)
    momentum = _clip01((_market_col(market, "mkt_CHDM20_delta", 0.0) + 10.0) / 20.0)
    return 100.0 * (0.32 * chdm20 + 0.28 * ds20 + 0.20 * chdm50 + 0.12 * ds50 + 0.08 * momentum)


def _market_col(market: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in market.columns:
        return market[column]
    return pd.Series(default, index=market.index)


def _clip01(value) -> pd.Series:
    return pd.Series(value, copy=False).clip(0.0, 1.0)


def _edge_score(features: pd.DataFrame, smart_money_col: str = "smart_money_score") -> pd.Series:
    smart_money = _feature_col(features, smart_money_col, 50.0).clip(0.0, 100.0)
    rs = 100.0 * _feature_col(features, "rs_percentile_20", 0.5).clip(0.0, 1.0)
    value_flow = _feature_col(features, "value_flow_quality_score", 50.0).clip(0.0, 100.0)
    money_cycle = _feature_col(features, "CHDM50", 50.0).clip(0.0, 100.0)
    regime = _feature_col(features, "mkt_regime_score", 50.0).clip(0.0, 100.0)
    ds_penalty = 100.0 * _feature_col(features, "DS20", 0.5).clip(0.0, 1.0)
    dist_penalty = 12.0 * _feature_col(features, "distribution_days_10", 1.0).clip(0.0, 5.0)
    value_expansion = 20.0 * (_feature_col(features, "value_ratio_20", 1.0).clip(0.0, 3.0) - 1.0)
    trend_bonus = 8.0 * features.get("above_ma50", pd.Series(False, index=features.index)).astype(float)
    breakout_bonus = 8.0 * features.get("breakout_20", pd.Series(False, index=features.index)).astype(float)

    score = (
        0.25 * smart_money
        + 0.20 * rs
        + 0.15 * value_flow
        + 0.15 * money_cycle
        + 0.15 * regime
        + 0.10 * (100.0 - ds_penalty)
        + value_expansion
        + trend_bonus
        + breakout_bonus
        - dist_penalty
    )
    return score.clip(0.0, 100.0)


def _feature_col(features: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in features.columns:
        return features[column].fillna(default)
    return pd.Series(default, index=features.index)
