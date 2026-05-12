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
        "value_flow_quality_score", "pullback_quality_score", "clv_ma5",
        "ma20", "ma50", "ma200", "atr14", "cmf20", "mfi14",
        "ret_5d", "ret_20d", "value_ma20", "value_ratio_20",
        "volume_ma20", "volume_ratio_20", "pullback_depth_20",
    ]
    smt_cols = [c for c in smt_cols if c in smt.columns]

    features = (
        ohlcv[["date", "symbol", "open", "high", "low", "close", "volume", "value"]]
        .merge(chdm, on=["date", "symbol"], how="left")
        .merge(ds, on=["date", "symbol"], how="left")
        .merge(smt[smt_cols], on=["date", "symbol"], how="left")
        .merge(market.add_prefix("mkt_").rename(columns={"mkt_date": "date"}), on="date", how="left")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
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
    group = features.groupby("symbol", sort=False)
    for col in ["DS20", "DS50", "CHDM03", "CHDM20", "CHDM50", "smart_money_score"]:
        if col in features.columns:
            features[f"{col}_prev"] = group[col].shift(1)
            features[f"{col}_delta"] = features[col] - features[f"{col}_prev"]
    features["high20_prev"] = group["high"].transform(lambda item: item.rolling(20, min_periods=10).max().shift(1))
    features["high55_prev"] = group["high"].transform(lambda item: item.rolling(55, min_periods=25).max().shift(1))
    features["low20_prev"] = group["low"].transform(lambda item: item.rolling(20, min_periods=10).min().shift(1))
    features["ma20_slope_5"] = group["ma20"].diff(5) / group["ma20"].shift(5) if "ma20" in features.columns else pd.NA
    features["ma50_slope_10"] = group["ma50"].diff(10) / group["ma50"].shift(10) if "ma50" in features.columns else pd.NA
    _add_market_derived_columns(features)
    features["above_ma50"] = features["close"] > features.get("ma50")
    features["above_ma20"] = features["close"] > features.get("ma20")
    features["above_ma20_prev"] = group["above_ma20"].shift(1)
    features["distance_ma20"] = features["close"] / features.get("ma20") - 1.0
    features["distance_ma50"] = features["close"] / features.get("ma50") - 1.0
    features["near_ma20"] = features["distance_ma20"].between(-0.03, 0.03)
    features["near_ma50"] = features["distance_ma50"].between(-0.04, 0.04)
    features["reclaim_ma20"] = features["above_ma20"] & (features["above_ma20_prev"] == False)
    features["reclaim_ma50"] = features["above_ma50"] & (group["above_ma50"].shift(1) == False)
    features["breakout_20"] = features["close"] > features["high20_prev"]
    features["breakout_55"] = features["close"] > features["high55_prev"]
    _add_ta_columns(features)
    features["ret_1d"] = group["close"].pct_change()
    features["ret_3d"] = group["close"].pct_change(3)
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
    features["range_pct_min7"] = group["range_pct"].transform(lambda item: item.rolling(7, min_periods=5).min())
    features["range_pct_q20"] = group["range_pct"].transform(lambda item: item.rolling(20, min_periods=10).quantile(0.25))
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
    features["edge_score"] = _edge_score(features)


def _add_ta_columns(features: pd.DataFrame) -> None:
    for _, idx in features.groupby("symbol", sort=False).groups.items():
        frame = features.loc[idx]
        close = frame["close"]
        high = frame["high"]
        low = frame["low"]
        features.loc[idx, "rsi14"] = ta.rsi(close, length=14).to_numpy()
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


def _edge_score(features: pd.DataFrame) -> pd.Series:
    smart_money = _feature_col(features, "smart_money_score", 50.0).clip(0.0, 100.0)
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
