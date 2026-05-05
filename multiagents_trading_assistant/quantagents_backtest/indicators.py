"""Causal OHLCV indicator enrichment for permutation strategies.

The indicator names and parameters follow ``docs/vnstock_ta/02-indicators.md``.
The implementation prefers ``vnstock_ta.Indicator`` for the broad QuantAgents
style feature set and falls back to pandas/numpy formulas where practical so
the backtester remains portable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from vnstock_ta import Indicator as VnstockTAIndicator
except Exception:  # pragma: no cover - optional dependency
    VnstockTAIndicator = None


REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV data with all strategy features added.

    The function does not shift ordinary indicators because signals are shifted
    by the execution engine before trading. Structure levels that compare the
    close against a prior range are explicitly shifted to avoid using today's
    high/low in today's breakout signal.
    """

    df = _validate_ohlcv(data)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ta_values = _compute_vnstock_ta_values(df)

    for length in (20, 50, 200):
        df[f"sma{length}"] = _numeric_series(
            ta_values.get(f"sma{length}", _sma(close, length)),
            df.index,
        )

    for length in (20, 50, 200):
        df[f"ema{length}"] = _numeric_series(
            ta_values.get(f"ema{length}", _ema(close, length)),
            df.index,
        )

    for length in (7, 14, 21):
        df[f"rsi_{length}"] = _numeric_series(
            ta_values.get(f"rsi_{length}", _rsi(close, length)),
            df.index,
        )

    macd = ta_values.get("macd")
    if macd is None:
        macd_line, macd_signal, macd_hist = _macd(close)
    else:
        macd_line = _numeric_series(_col_prefix(macd, "MACD_"), df.index)
        macd_signal = _numeric_series(_col_prefix(macd, "MACDs_"), df.index)
        macd_hist = _numeric_series(_col_prefix(macd, "MACDh_"), df.index)
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    df["willr_14"] = _numeric_series(ta_values.get("willr_14", _willr(high, low, close, 14)), df.index)
    df["cmo_9"] = _numeric_series(ta_values.get("cmo_9", _cmo(close, 9)), df.index)
    stoch = ta_values.get("stoch")
    if stoch is None:
        stoch_k, stoch_d = _stoch(high, low, close, k=14, d=3, smooth_k=3)
    else:
        stoch_k = _numeric_series(_col_prefix(stoch, "STOCHk_"), df.index)
        stoch_d = _numeric_series(_col_prefix(stoch, "STOCHd_"), df.index)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d
    df["roc_9"] = _numeric_series(ta_values.get("roc_9", _roc(close, 9)), df.index)
    df["mom_10"] = _numeric_series(ta_values.get("mom_10", close.diff(10)), df.index)

    df["volume_ma20"] = volume.rolling(20, min_periods=20).mean()
    df["volume_ratio_20"] = volume / df["volume_ma20"]
    df["volume_spike"] = df["volume_ratio_20"] >= 1.5
    df["obv"] = _numeric_series(ta_values.get("obv", _obv(close, volume)), df.index)
    df["obv_ema20"] = _ema(df["obv"], 20)
    df["vwma20"] = _numeric_series(ta_values.get("vwma20", _vwma(close, volume, 20)), df.index)

    bb = ta_values.get("bbands")
    if bb is None:
        bb_lower, bb_mid, bb_upper = _bbands(close, length=20, std=2.0)
    else:
        bb_lower = _numeric_series(_col_prefix(bb, "BBL_"), df.index)
        bb_mid = _numeric_series(_col_prefix(bb, "BBM_"), df.index)
        bb_upper = _numeric_series(_col_prefix(bb, "BBU_"), df.index)
    df["bb_lower"] = bb_lower
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_percent"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["atr_14"] = _numeric_series(ta_values.get("atr_14", _atr(high, low, close, length=14)), df.index)
    df["atr_pct"] = df["atr_14"] / close
    df["stdev_14"] = _numeric_series(ta_values.get("stdev_14", close.rolling(14, min_periods=14).std()), df.index)
    df["linreg_14"] = _numeric_series(ta_values.get("linreg_14", _linreg(close, 14)), df.index)
    df["linreg_slope_14"] = df["linreg_14"].diff()

    adx = ta_values.get("adx")
    if adx is None:
        adx_line, dmp, dmn = _adx(high, low, close, length=14)
    else:
        adx_line = _numeric_series(_col_prefix(adx, "ADX_"), df.index)
        dmp = _numeric_series(_col_prefix(adx, "DMP_"), df.index)
        dmn = _numeric_series(_col_prefix(adx, "DMN_"), df.index)
    df["adx_14"] = adx_line
    df["dmp_14"] = dmp
    df["dmn_14"] = dmn

    aroon = ta_values.get("aroon")
    if aroon is None:
        aroon_up, aroon_down, aroon_osc = _aroon(high, low, length=14)
    else:
        aroon_up = _numeric_series(_col_prefix(aroon, "AROONU_"), df.index)
        aroon_down = _numeric_series(_col_prefix(aroon, "AROOND_"), df.index)
        aroon_osc = _numeric_series(_col_prefix(aroon, "AROONOSC_"), df.index)
    df["aroon_up_14"] = aroon_up
    df["aroon_down_14"] = aroon_down
    df["aroon_osc_14"] = aroon_osc

    supertrend = ta_values.get("supertrend")
    if supertrend is None:
        supertrend_value, supertrend_dir = _supertrend(high, low, close, length=10, multiplier=3.0)
    else:
        supertrend_value = _numeric_series(_col_prefix(supertrend, "SUPERT_"), df.index)
        supertrend_dir = _numeric_series(_col_prefix(supertrend, "SUPERTd_"), df.index)
    df["supertrend_10_3"] = supertrend_value
    df["supertrend_dir"] = supertrend_dir

    kc = ta_values.get("kc")
    if kc is None:
        kc_lower, kc_mid, kc_upper = _keltner(close, df["atr_14"], length=20, scalar=2.0)
    else:
        kc_lower = _numeric_series(_col_prefix(kc, "KCLe_"), df.index)
        kc_mid = _numeric_series(_col_prefix(kc, "KCBe_"), df.index)
        kc_upper = _numeric_series(_col_prefix(kc, "KCUe_"), df.index)
    df["kc_lower"] = kc_lower
    df["kc_mid"] = kc_mid
    df["kc_upper"] = kc_upper
    df["kc_width"] = (df["kc_upper"] - df["kc_lower"]) / df["kc_mid"]

    df["high_20_prev"] = high.shift(1).rolling(20, min_periods=20).max()
    df["low_20_prev"] = low.shift(1).rolling(20, min_periods=20).min()
    df["breakout_20_high"] = close > df["high_20_prev"]
    df["breakdown_20_low"] = close < df["low_20_prev"]

    df["close_return"] = close.pct_change()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def _validate_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {missing}")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("OHLCV data must use a DatetimeIndex")
    df = data.loc[:, REQUIRED_OHLCV_COLUMNS].copy()
    df.sort_index(inplace=True)
    return df.astype(float)


def _compute_vnstock_ta_values(df: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
    if VnstockTAIndicator is None:
        return {}
    try:
        indicator = VnstockTAIndicator(data=df)
        values: dict[str, pd.Series | pd.DataFrame] = {
            "macd": indicator.macd(fast=12, slow=26, signal=9),
            "bbands": indicator.bbands(length=20, std=2),
            "atr_14": indicator.atr(length=14),
            "adx": indicator.adx(length=14),
            "aroon": indicator.aroon(length=14),
            "willr_14": indicator.willr(length=14),
            "cmo_9": indicator.cmo(length=9),
            "stoch": indicator.stoch(k=14, d=3, smooth_k=3),
            "roc_9": indicator.roc(length=9),
            "mom_10": indicator.mom(length=10),
            "kc": indicator.kc(length=20, scalar=2.0, mamode="ema"),
            "stdev_14": indicator.stdev(length=14),
            "linreg_14": indicator.linreg(length=14),
            "obv": indicator.obv(),
            "supertrend": indicator.supertrend(length=10, multiplier=3),
            "vwma20": indicator.vwma(length=20),
        }
        for length in (20, 50, 200):
            values[f"sma{length}"] = indicator.sma(length=length)
        for length in (20, 50, 200):
            values[f"ema{length}"] = indicator.ema(length=length)
        for length in (7, 14, 21):
            values[f"rsi_{length}"] = indicator.rsi(length=length)
        return values
    except Exception:
        return {}


def _sma(close: pd.Series, length: int) -> pd.Series:
    return close.rolling(length, min_periods=length).mean()


def _ema(close: pd.Series, length: int) -> pd.Series:
    return close.ewm(span=length, adjust=False, min_periods=length).mean()


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = _ema(close, 12) - _ema(close, 26)
    signal = line.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = line - signal
    return line, signal, hist


def _bbands(close: pd.Series, length: int, std: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(length, min_periods=length).mean()
    dev = close.rolling(length, min_periods=length).std(ddof=0)
    upper = mid + std * dev
    lower = mid - std * dev
    return lower, mid, upper


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _willr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    highest = high.rolling(length, min_periods=length).max()
    lowest = low.rolling(length, min_periods=length).min()
    return -100 * (highest - close) / (highest - lowest)


def _cmo(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).rolling(length, min_periods=length).sum()
    down = (-delta.clip(upper=0)).rolling(length, min_periods=length).sum()
    return 100 * (up - down) / (up + down)


def _stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int,
    d: int,
    smooth_k: int,
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k, min_periods=k).min()
    highest = high.rolling(k, min_periods=k).max()
    raw_k = 100 * (close - lowest) / (highest - lowest)
    stoch_k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    stoch_d = stoch_k.rolling(d, min_periods=d).mean()
    return stoch_k, stoch_d


def _roc(close: pd.Series, length: int) -> pd.Series:
    return close.pct_change(length) * 100


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def _vwma(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    return (close * volume).rolling(length, min_periods=length).sum() / volume.rolling(
        length,
        min_periods=length,
    ).sum()


def _linreg(close: pd.Series, length: int) -> pd.Series:
    x = np.arange(length, dtype=float)

    def fit(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        slope, intercept = np.polyfit(x, values, 1)
        return float(intercept + slope * x[-1])

    return close.rolling(length, min_periods=length).apply(fit, raw=True)


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)
    atr = _atr(high, low, close, length)
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False, min_periods=length).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    return adx, plus_di, minus_di


def _aroon(high: pd.Series, low: pd.Series, length: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    def periods_since_high(values: np.ndarray) -> float:
        return float(length - 1 - np.argmax(values))

    def periods_since_low(values: np.ndarray) -> float:
        return float(length - 1 - np.argmin(values))

    since_high = high.rolling(length, min_periods=length).apply(periods_since_high, raw=True)
    since_low = low.rolling(length, min_periods=length).apply(periods_since_low, raw=True)
    up = 100 * (length - since_high) / length
    down = 100 * (length - since_low) / length
    return up, down, up - down


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int,
    multiplier: float,
) -> tuple[pd.Series, pd.Series]:
    atr = _atr(high, low, close, length)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    direction = pd.Series(np.nan, index=close.index)
    trend = pd.Series(np.nan, index=close.index)
    for i in range(len(close)):
        if i == 0 or pd.isna(atr.iloc[i]):
            continue
        prev_trend = trend.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]
        if pd.isna(prev_dir):
            direction.iloc[i] = 1.0 if close.iloc[i] >= hl2.iloc[i] else -1.0
        elif close.iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1.0
        elif close.iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1.0
        else:
            direction.iloc[i] = prev_dir
        trend.iloc[i] = lower.iloc[i] if direction.iloc[i] > 0 else upper.iloc[i]
        if direction.iloc[i] > 0 and not pd.isna(prev_trend):
            trend.iloc[i] = max(trend.iloc[i], prev_trend)
        elif direction.iloc[i] < 0 and not pd.isna(prev_trend):
            trend.iloc[i] = min(trend.iloc[i], prev_trend)
    return trend, direction


def _keltner(
    close: pd.Series,
    atr: pd.Series,
    length: int,
    scalar: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = _ema(close, length)
    upper = mid + scalar * atr
    lower = mid - scalar * atr
    return lower, mid, upper


def _col_prefix(frame: pd.DataFrame | None, prefix: str) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    col = next((name for name in frame.columns if str(name).startswith(prefix)), None)
    if col is None:
        return pd.Series(index=frame.index, dtype=float)
    return frame[col]


def _numeric_series(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(index=index, dtype=float)
    values = pd.to_numeric(pd.Series(series), errors="coerce")
    if len(values) == len(index) and not values.index.equals(index):
        values.index = index
    return values.reindex(index)
