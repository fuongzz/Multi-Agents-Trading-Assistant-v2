"""Build daily research feature tables from existing project artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


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
        "ma20", "ma50", "ma200", "cmf20", "mfi14",
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
    _add_market_derived_columns(features)
    features["above_ma50"] = features["close"] > features.get("ma50")


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
