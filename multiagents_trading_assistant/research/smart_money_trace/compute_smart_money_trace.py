"""
Smart Money Trace Research Module

Định lượng hóa dấu vết dòng tiền thông minh (Wyckoff / VSA / A/D family) để phân biệt:
- HOT_BUT_STRONG: cycle cao, cá mập vẫn hấp thụ
- HOT_AND_EXHAUSTED: cycle cao, dòng tiền đang suy yếu
- HOT_AND_DISTRIBUTING: cycle cao, dấu hiệu phân phối
- RESET_IN_UPTREND: pull back khỏe trong uptrend lớn

Input:  OHLCV parquet/csv + tùy chọn chdm_by_symbol.parquet
Output: 3 parquet files trong data/research/smart_money_trace/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Load & normalize
# ---------------------------------------------------------------------------

def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data từ parquet hoặc CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {path.suffix}")


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa cột, kiểm tra required, tính value nếu thiếu."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    if "ticker" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"ticker": "symbol"})

    required = ["date", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df["symbol"] = df["symbol"].astype(str).str.upper()

    if "value" not in df.columns:
        df["value"] = df["close"] * df["volume"]

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info("Normalized: %d rows, %d symbols", len(df), df["symbol"].nunique())
    return df


# ---------------------------------------------------------------------------
# 2. Basic features
# ---------------------------------------------------------------------------

def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tính returns, MAs, value/volume ratios, pullback depth."""
    df = df.copy()
    g = df.groupby("symbol", sort=False)

    df["ret_1d"] = g["close"].transform(lambda x: x.pct_change())
    df["ret_5d"] = g["close"].transform(lambda x: x.pct_change(5))
    df["ret_20d"] = g["close"].transform(lambda x: x.pct_change(20))

    for w in [20, 50, 200]:
        df[f"ma{w}"] = g["close"].transform(
            lambda x, p=w: x.rolling(p, min_periods=p).mean()
        )

    df["value_ma20"] = g["value"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["value_ratio_20"] = (df["value"] / df["value_ma20"]).replace([np.inf, -np.inf], np.nan)

    df["volume_ma20"] = g["volume"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["volume_ratio_20"] = (df["volume"] / df["volume_ma20"]).replace([np.inf, -np.inf], np.nan)

    df["high_20"] = g["high"].transform(lambda x: x.rolling(20, min_periods=10).max())
    df["pullback_depth_20"] = df["close"] / df["high_20"] - 1

    return df


# ---------------------------------------------------------------------------
# 3. CLV
# ---------------------------------------------------------------------------

def compute_clv(df: pd.DataFrame) -> pd.DataFrame:
    """CLV = (close - low) / (high - low); clv_ma5 = rolling 5 phiên."""
    df = df.copy()
    hl = df["high"] - df["low"]
    df["clv"] = np.where(hl > 0, (df["close"] - df["low"]) / hl, 0.5)
    df["clv_ma5"] = df.groupby("symbol", sort=False)["clv"].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    return df


# ---------------------------------------------------------------------------
# 4. ATR
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """ATR14 = rolling mean của True Range."""
    df = df.copy()
    prev_close = df.groupby("symbol", sort=False)["close"].transform(lambda x: x.shift(1))
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["_tr"] = tr
    df["atr14"] = df.groupby("symbol", sort=False)["_tr"].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    return df.drop(columns=["_tr"])


# ---------------------------------------------------------------------------
# 5. Relative Strength
# ---------------------------------------------------------------------------

def compute_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    """rs_20 = ret_20d - market_median; rs_percentile_20 = rank ngày."""
    df = df.copy()
    mkt_ret = df.groupby("date")["ret_20d"].median().rename("_mkt_ret_20d")
    df = df.merge(mkt_ret, on="date", how="left")
    df["rs_20"] = df["ret_20d"] - df["_mkt_ret_20d"]
    df["rs_percentile_20"] = df.groupby("date")["ret_20d"].transform(
        lambda x: x.rank(pct=True, na_option="keep")
    )
    return df.drop(columns=["_mkt_ret_20d"])


# ---------------------------------------------------------------------------
# 6. Accumulation / Distribution Days
# ---------------------------------------------------------------------------

def compute_accumulation_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """A/D days và balance 10 phiên theo định nghĩa VSA/Wyckoff."""
    df = df.copy()

    df["accumulation_day"] = (
        (df["ret_1d"] > 0)
        & (df["value_ratio_20"] >= 1.2)
        & (df["clv"] >= 0.65)
        & (df["close"] > df["open"])
    ).astype(float)

    df["distribution_day"] = (
        (df["ret_1d"] < 0)
        & (df["value_ratio_20"] >= 1.2)
        & (df["clv"] <= 0.35)
    ).astype(float)

    g = df.groupby("symbol", sort=False)
    df["accumulation_days_10"] = g["accumulation_day"].transform(
        lambda x: x.rolling(10, min_periods=5).sum()
    )
    df["distribution_days_10"] = g["distribution_day"].transform(
        lambda x: x.rolling(10, min_periods=5).sum()
    )
    df["ad_balance_10"] = df["accumulation_days_10"] - df["distribution_days_10"]
    return df


# ---------------------------------------------------------------------------
# 7. Pullback Quality Score
# ---------------------------------------------------------------------------

def compute_pullback_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Pullback khỏe trong uptrend: close > ma50, vol khô khi giảm, ít D-days."""
    df = df.copy()
    score = pd.Series(50.0, index=df.index)

    score += 15 * (df["close"] > df["ma50"]).fillna(False)
    score += 15 * (df["ma20"] > df["ma50"]).fillna(False)
    score += 15 * (
        (df["pullback_depth_20"] >= -0.12) & (df["pullback_depth_20"] <= -0.03)
    ).fillna(False)
    score += 10 * ((df["volume_ratio_20"] < 1.0) & (df["ret_1d"] <= 0)).fillna(False)
    score += 10 * (df["distribution_days_10"] <= 1).fillna(False)

    score -= 20 * (df["close"] < df["ma50"]).fillna(False)
    score -= 20 * (df["distribution_days_10"] >= 2).fillna(False)
    score -= 15 * ((df["volume_ratio_20"] > 1.5) & (df["ret_1d"] < 0)).fillna(False)
    score -= 15 * (df["clv"] < 0.35).fillna(False)

    df["pullback_quality_score"] = score.clip(0, 100)
    return df


# ---------------------------------------------------------------------------
# 8. Value Flow Quality Score
# ---------------------------------------------------------------------------

def compute_value_flow_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Phân biệt tiền vào chất lượng vs phân phối dựa trên value_ratio + CLV + hướng giá."""
    df = df.copy()
    g = df.groupby("symbol", sort=False)

    # Value_ratio > 1.0 trong 3/5 phiên gần đây
    df["_vr_gt1"] = (df["value_ratio_20"] > 1.0).astype(float)
    vr_count5 = g["_vr_gt1"].transform(lambda x: x.rolling(5, min_periods=3).sum())

    # Value_ratio đang giảm dần
    vr_diff = g["value_ratio_20"].transform(lambda x: x.diff())
    df["_vr_declining"] = (vr_diff < 0).astype(float)
    vr_declining_count3 = g["_vr_declining"].transform(
        lambda x: x.rolling(3, min_periods=2).sum()
    )

    score = pd.Series(50.0, index=df.index)

    score += 20 * ((df["value_ratio_20"] >= 1.2) & (df["ret_1d"] > 0)).fillna(False)
    score += 15 * ((df["value_ratio_20"] >= 1.2) & (df["clv"] >= 0.65)).fillna(False)
    score += 10 * ((df["volume_ratio_20"] >= 1.2) & (df["close"] > df["open"])).fillna(False)
    score += 10 * (vr_count5 >= 3).fillna(False)

    score -= 20 * ((df["value_ratio_20"] >= 1.2) & (df["ret_1d"] < 0)).fillna(False)
    score -= 20 * ((df["value_ratio_20"] >= 1.2) & (df["clv"] <= 0.35)).fillna(False)
    # Phân kỳ bearish: value ratio giảm trong khi giá tăng
    score -= 10 * ((vr_declining_count3 >= 2) & (df["ret_1d"] > 0)).fillna(False)

    df["value_flow_quality_score"] = score.clip(0, 100)
    return df.drop(columns=["_vr_gt1", "_vr_declining"])


# ---------------------------------------------------------------------------
# 9. Sector Leadership
# ---------------------------------------------------------------------------

def compute_sector_leadership(df: pd.DataFrame) -> pd.DataFrame:
    """Sector return vs market, value ratio, advancing ratio, above MA20 ratio."""
    df = df.copy()

    if "industry" not in df.columns or df["industry"].isna().all():
        df["sector_leadership_score"] = 50.0
        return df

    mkt_ret = df.groupby("date")["ret_20d"].median().rename("_mkt_ret")
    df = df.merge(mkt_ret, on="date", how="left")

    df["_above_ma20"] = (df["close"] > df["ma20"]).astype(float)

    sector_agg = (
        df.groupby(["date", "industry"])
        .agg(
            _sec_ret_20d=("ret_20d", "median"),
            _sec_value_ratio=("value_ratio_20", "median"),
            _sec_adv_ratio=("ret_1d", lambda x: (x > 0).mean()),
            _sec_ma20_ratio=("_above_ma20", "mean"),
        )
        .reset_index()
    )

    df = df.merge(sector_agg, on=["date", "industry"], how="left")

    score = pd.Series(50.0, index=df.index)
    score += 20 * (df["_sec_ret_20d"] > df["_mkt_ret"]).fillna(False)
    score += 15 * (df["_sec_value_ratio"] > 1.1).fillna(False)
    score += 10 * (df["_sec_adv_ratio"] > 0.5).fillna(False)
    score += 10 * (df["_sec_ma20_ratio"] > 0.5).fillna(False)

    df["sector_leadership_score"] = score.clip(0, 100)

    drop_cols = [
        "_mkt_ret", "_above_ma20",
        "_sec_ret_20d", "_sec_value_ratio", "_sec_adv_ratio", "_sec_ma20_ratio",
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


# ---------------------------------------------------------------------------
# 10. Baseline indicators: ADL, CMF20, OBV, MFI14
# ---------------------------------------------------------------------------

def compute_baseline_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """ADL, CMF20, OBV, MFI14 — dùng để so sánh với SmartMoneyScore."""
    df = df.copy()
    g = df.groupby("symbol", sort=False)

    hl = (df["high"] - df["low"]).replace(0.0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl
    mfm = mfm.fillna(0.0)
    df["_mfv"] = mfm * df["volume"]

    # ADL (cumsum per symbol)
    df["adl"] = g["_mfv"].transform(lambda x: x.cumsum())

    # CMF20
    mfv_roll = g["_mfv"].transform(lambda x: x.rolling(20, min_periods=10).sum())
    vol_roll = g["volume"].transform(lambda x: x.rolling(20, min_periods=10).sum())
    df["cmf20"] = (mfv_roll / vol_roll.replace(0.0, np.nan)).clip(-1, 1)

    # OBV
    prev_close = g["close"].transform(lambda x: x.shift(1))
    df["_obv_flow"] = np.sign(df["close"] - prev_close).fillna(0) * df["volume"]
    df["obv"] = g["_obv_flow"].transform(lambda x: x.cumsum())

    # MFI14
    df["_tp"] = (df["high"] + df["low"] + df["close"]) / 3
    df["_tp_prev"] = g["_tp"].transform(lambda x: x.shift(1))
    tp_mf = df["_tp"] * df["volume"]
    df["_pos_mf"] = tp_mf.where(df["_tp"] > df["_tp_prev"], 0.0)
    df["_neg_mf"] = tp_mf.where(df["_tp"] < df["_tp_prev"], 0.0)
    pos_roll = g["_pos_mf"].transform(lambda x: x.rolling(14, min_periods=14).sum())
    neg_roll = g["_neg_mf"].transform(lambda x: x.rolling(14, min_periods=14).sum())
    mfr = pos_roll / neg_roll.replace(0.0, np.nan)
    df["mfi14"] = (100 - 100 / (1 + mfr)).clip(0, 100)

    temp = ["_mfv", "_obv_flow", "_tp", "_tp_prev", "_pos_mf", "_neg_mf"]
    return df.drop(columns=[c for c in temp if c in df.columns])


# ---------------------------------------------------------------------------
# 11. Smart Money Scores
# ---------------------------------------------------------------------------

def compute_smart_money_scores(df: pd.DataFrame) -> pd.DataFrame:
    """CLV score, RS score, AD score → smart_money_score tổng hợp."""
    df = df.copy()

    # CLV score
    clv5 = df["clv_ma5"].fillna(0.5)
    df["clv_score"] = np.select(
        [clv5 >= 0.7, clv5 >= 0.6, clv5 >= 0.5, clv5 >= 0.4],
        [90, 75, 60, 45],
        default=25,
    ).astype(float)

    # RS score
    df["rs_score"] = (df["rs_percentile_20"].fillna(0.5) * 100).clip(0, 100)

    # A/D balance score
    bal = df["ad_balance_10"].fillna(0)
    df["accumulation_distribution_score"] = np.select(
        [bal >= 3, bal == 2, bal == 1, bal == 0, bal == -1],
        [90, 80, 65, 50, 35],
        default=20,
    ).astype(float)

    df["smart_money_score"] = (
        0.25 * df["rs_score"]
        + 0.20 * df["value_flow_quality_score"]
        + 0.20 * df["clv_score"]
        + 0.15 * df["accumulation_distribution_score"]
        + 0.10 * df["pullback_quality_score"]
        + 0.10 * df["sector_leadership_score"]
    ).clip(0, 100)

    return df


# ---------------------------------------------------------------------------
# 12. CHDM helper (tự tính nếu chdm_by_symbol.parquet không có)
# ---------------------------------------------------------------------------

def _compute_chdm_column(df: pd.DataFrame, window: int) -> pd.Series:
    """CHDM = 100 × (close − LowestLow_N) / (HighestHigh_N − LowestLow_N)."""
    lo = df.groupby("symbol", sort=False)["low"].transform(
        lambda x, n=window: x.rolling(n, min_periods=n).min()
    )
    hi = df.groupby("symbol", sort=False)["high"].transform(
        lambda x, n=window: x.rolling(n, min_periods=n).max()
    )
    denom = (hi - lo).replace(0.0, np.nan)
    return (100 * (df["close"] - lo) / denom).clip(0, 100)


def ensure_chdm_columns(df: pd.DataFrame, windows: list[int] = (3, 5, 10, 20, 50)) -> pd.DataFrame:
    """Đảm bảo CHDM columns tồn tại — tự tính nếu chưa có."""
    df = df.copy()
    for w in windows:
        col = f"CHDM{w:02d}"
        if col not in df.columns:
            logger.debug("Computing %s internally", col)
            df[col] = _compute_chdm_column(df, w)
    return df


# ---------------------------------------------------------------------------
# 13. State Classification
# ---------------------------------------------------------------------------

_STATE_TO_ACTION = {
    "HOT_BUT_STRONG": "HOLD_OR_BUY_PULLBACK",
    "HOT_AND_EXHAUSTED": "HOLD_TIGHTEN_STOP",
    "HOT_AND_DISTRIBUTING": "REDUCE_OR_EXIT",
    "RESET_IN_UPTREND": "BUY_PULLBACK_WATCHLIST",
    "NEUTRAL": "NEUTRAL",
}

_STATE_TO_WARNING = {
    "HOT_BUT_STRONG": "Do not chase; leader hot but real money still strong",
    "HOT_AND_EXHAUSTED": "Hot cycle with weakening money flow. Avoid new chase.",
    "HOT_AND_DISTRIBUTING": "Potential distribution. Tighten trailing stop or reduce exposure.",
    "RESET_IN_UPTREND": "Short-cycle reset inside larger uptrend. Wait for price trigger.",
    "NEUTRAL": "No clear smart money edge",
}


def classify_smart_money_state(df: pd.DataFrame) -> pd.DataFrame:
    """Phân loại HOT_BUT_STRONG / HOT_AND_EXHAUSTED / HOT_AND_DISTRIBUTING / RESET_IN_UPTREND / NEUTRAL."""
    df = df.copy()

    # --- HOT_CLUSTER ---
    hot_cluster = (
        (df["CHDM03"] >= 80)
        & (df["CHDM05"] >= 80)
        & (df["CHDM10"] >= 80)
        & (df["CHDM20"] >= 75)
    ).fillna(False)

    # --- SHORT_RESET ---
    chdm03_rising = df.groupby("symbol", sort=False)["CHDM03"].transform(
        lambda x: x > x.shift(1)
    ).fillna(False)

    short_reset = (
        (df["CHDM50"] >= 50)
        & df["CHDM03"].between(20, 45)
        & df["CHDM05"].between(20, 45)
        & df["CHDM10"].between(20, 45)
        & chdm03_rising
    ).fillna(False)

    # --- HOT_BUT_STRONG ---
    hbs = (
        hot_cluster
        & (df["smart_money_score"] >= 75)
        & (df["rs_score"] >= 80)
        & (df["clv_ma5"] >= 0.60).fillna(False)
        & (df["accumulation_days_10"] >= df["distribution_days_10"]).fillna(False)
        & (df["close"] > df["ma20"]).fillna(False)
    )

    # --- HOT_AND_EXHAUSTED ---
    weak_signs = (
        (df["clv_ma5"] < 0.55).astype(int).fillna(0)
        + (df["value_ratio_20"] < 1.0).astype(int).fillna(0)
        + (df["rs_score"] < 70).astype(int).fillna(0)
        + (df["distribution_days_10"] >= 1).astype(int).fillna(0)
    )
    hae = (
        hot_cluster
        & (df["smart_money_score"] >= 50)
        & (df["smart_money_score"] < 75)
        & (weak_signs >= 2)
    )

    # --- HOT_AND_DISTRIBUTING ---
    dist_signs = (
        (df["distribution_days_10"] >= 2).astype(int).fillna(0)
        + (df["clv_ma5"] < 0.45).astype(int).fillna(0)
        + ((df["value_ratio_20"] >= 1.2) & (df["ret_1d"] < 0)).astype(int).fillna(0)
        + (df["close"] < df["ma20"]).astype(int).fillna(0)
        + (df["smart_money_score"] < 50).astype(int).fillna(0)
    )
    had = hot_cluster & (dist_signs >= 2)

    # --- RESET_IN_UPTREND ---
    rut = (
        short_reset
        & (df["smart_money_score"] >= 65)
        & (df["close"] > df["ma50"]).fillna(False)
        & (df["ma20"] >= df["ma50"]).fillna(False)
        & (df["distribution_days_10"] <= 1).fillna(False)
    )

    # Priority: HOT_BUT_STRONG > HOT_AND_DISTRIBUTING > HOT_AND_EXHAUSTED > RESET_IN_UPTREND > NEUTRAL
    state = pd.Series("NEUTRAL", index=df.index)
    state = state.where(~rut, "RESET_IN_UPTREND")
    state = state.where(~hae, "HOT_AND_EXHAUSTED")
    state = state.where(~had, "HOT_AND_DISTRIBUTING")
    state = state.where(~hbs, "HOT_BUT_STRONG")

    df["smart_money_state"] = state
    df["action_bias"] = state.map(_STATE_TO_ACTION)
    df["warnings"] = state.map(_STATE_TO_WARNING)

    return df


# ---------------------------------------------------------------------------
# 14. Daily Summary
# ---------------------------------------------------------------------------

def summarize_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp theo ngày: score trung bình, số lượng từng state, A/D ratio."""
    agg = (
        df.groupby("date")
        .agg(
            avg_smart_money_score=("smart_money_score", "mean"),
            median_smart_money_score=("smart_money_score", "median"),
            hot_but_strong_count=("smart_money_state", lambda x: (x == "HOT_BUT_STRONG").sum()),
            hot_and_exhausted_count=("smart_money_state", lambda x: (x == "HOT_AND_EXHAUSTED").sum()),
            hot_and_distributing_count=("smart_money_state", lambda x: (x == "HOT_AND_DISTRIBUTING").sum()),
            reset_in_uptrend_count=("smart_money_state", lambda x: (x == "RESET_IN_UPTREND").sum()),
            accumulation_total=("accumulation_day", "sum"),
            distribution_total=("distribution_day", "sum"),
        )
        .reset_index()
    )
    total = agg["accumulation_total"] + agg["distribution_total"]
    agg["accumulation_distribution_ratio"] = (
        agg["accumulation_total"] / total.replace(0, np.nan)
    )
    return agg


# ---------------------------------------------------------------------------
# 15. Save outputs
# ---------------------------------------------------------------------------

_BY_SYMBOL_COLS = [
    "date", "symbol", "close", "volume", "value",
    "value_ma20", "value_ratio_20", "volume_ma20", "volume_ratio_20",
    "clv", "clv_ma5",
    "ret_1d", "ret_5d", "ret_20d",
    "ma20", "ma50", "ma200", "atr14",
    "rs_20", "rs_percentile_20",
    "accumulation_day", "distribution_day",
    "accumulation_days_10", "distribution_days_10", "ad_balance_10",
    "pullback_depth_20",
    "pullback_quality_score", "value_flow_quality_score",
    "clv_score", "rs_score",
    "accumulation_distribution_score", "sector_leadership_score",
    "smart_money_score", "smart_money_state", "action_bias", "warnings",
    # Baseline
    "adl", "cmf20", "obv", "mfi14",
]


def save_outputs(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Lưu 3 file parquet: by_symbol, daily_summary, hot_states."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_to_parquet(frame: pd.DataFrame, path: Path) -> None:
        tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
        tmp_path.unlink(missing_ok=True)
        frame.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)

    # Include CHDM columns if present
    chdm_cols = sorted(c for c in df.columns if c.startswith("CHDM"))
    all_cols = _BY_SYMBOL_COLS + [c for c in chdm_cols if c not in _BY_SYMBOL_COLS]
    existing = [c for c in all_cols if c in df.columns]
    df_out = df[existing]

    p1 = output_dir / "smart_money_by_symbol.parquet"
    _atomic_to_parquet(df_out, p1)
    logger.info("Saved %s: %d rows", p1.name, len(df_out))

    p2 = output_dir / "smart_money_daily_summary.parquet"
    _atomic_to_parquet(summary, p2)
    logger.info("Saved %s: %d rows", p2.name, len(summary))

    hot_states = {"HOT_BUT_STRONG", "HOT_AND_EXHAUSTED", "HOT_AND_DISTRIBUTING", "RESET_IN_UPTREND"}
    hot_df = df_out[df_out["smart_money_state"].isin(hot_states)]
    p3 = output_dir / "smart_money_hot_states.parquet"
    _atomic_to_parquet(hot_df, p3)
    logger.info("Saved %s: %d rows", p3.name, len(hot_df))


# ---------------------------------------------------------------------------
# 16. Full pipeline entry point
# ---------------------------------------------------------------------------

def run_smart_money_trace(
    input_path: str | Path,
    output_dir: str | Path,
    chdm_path: Optional[str | Path] = None,
) -> dict:
    """Chạy full Smart Money Trace pipeline.

    Args:
        input_path: Đường dẫn đến OHLCV parquet/csv
        output_dir: Thư mục lưu kết quả
        chdm_path: Đường dẫn chdm_by_symbol.parquet (auto-detect nếu None)

    Returns:
        dict thống kê kết quả
    """
    logger.info("=== Smart Money Trace ===")
    logger.info("  Input:  %s", input_path)
    logger.info("  Output: %s", output_dir)

    df = load_ohlcv(input_path)
    df = normalize_ohlcv_columns(df)

    n_rows = len(df)
    n_symbols = df["symbol"].nunique()
    date_min = df["date"].min()
    date_max = df["date"].max()
    logger.info("  Rows: %d | Symbols: %d | Range: %s → %s",
                n_rows, n_symbols, date_min.date(), date_max.date())

    # --- Merge CHDM if available ---
    chdm_loaded = False
    if chdm_path is None:
        default = Path("data/research/money_cycle/chdm_by_symbol.parquet")
        if default.exists():
            chdm_path = default

    if chdm_path and Path(chdm_path).exists():
        chdm_df = pd.read_parquet(chdm_path)
        chdm_df["date"] = pd.to_datetime(chdm_df["date"])
        chdm_df["symbol"] = chdm_df["symbol"].astype(str).str.upper()
        chdm_cols = [c for c in chdm_df.columns if c.startswith("CHDM")]
        df = df.merge(chdm_df[["date", "symbol"] + chdm_cols], on=["date", "symbol"], how="left")
        chdm_loaded = True
        logger.info("  CHDM merged: %s columns from %s", len(chdm_cols), chdm_path)
    else:
        logger.info("  CHDM file not found → computing internally")

    # --- Pipeline ---
    logger.info("Computing basic features...")
    df = compute_basic_features(df)

    logger.info("Computing CLV...")
    df = compute_clv(df)

    logger.info("Computing ATR14...")
    df = compute_atr(df)

    logger.info("Computing relative strength...")
    df = compute_relative_strength(df)

    logger.info("Computing A/D days...")
    df = compute_accumulation_distribution(df)

    logger.info("Computing pullback quality...")
    df = compute_pullback_quality(df)

    logger.info("Computing value flow quality...")
    df = compute_value_flow_quality(df)

    logger.info("Computing sector leadership...")
    df = compute_sector_leadership(df)

    logger.info("Computing baseline indicators (ADL, CMF20, OBV, MFI14)...")
    df = compute_baseline_indicators(df)

    logger.info("Computing smart money scores...")
    df = compute_smart_money_scores(df)

    logger.info("Ensuring CHDM columns (CHDM03/05/10/20/50)...")
    df = ensure_chdm_columns(df)

    logger.info("Classifying smart money states...")
    df = classify_smart_money_state(df)

    logger.info("Summarizing daily...")
    summary = summarize_daily(df)

    logger.info("Saving outputs...")
    save_outputs(df, summary, output_dir)

    state_counts = df["smart_money_state"].value_counts().to_dict()

    stats: dict = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "n_rows_input": n_rows,
        "n_symbols": n_symbols,
        "date_min": str(date_min.date()),
        "date_max": str(date_max.date()),
        "chdm_loaded_from_file": chdm_loaded,
        "n_rows_output": len(df),
        "state_counts": state_counts,
    }
    logger.info("=== Smart Money Trace complete ===")
    return stats
