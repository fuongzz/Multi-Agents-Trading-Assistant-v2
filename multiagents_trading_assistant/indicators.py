import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
import os

import pandas as pd
import pandas_ta as ta

from multiagents_trading_assistant.fetcher import get_ohlcv
from multiagents_trading_assistant.price_volume import analyze_price_volume

try:
    from vnstock_ta import Indicator as VnstockTAIndicator
except Exception:
    VnstockTAIndicator = None


def compute_indicators_from_symbol(symbol: str, n_days: int = 200) -> dict:
    """Convenience: lấy OHLCV rồi tính indicators trong 1 bước."""
    df = get_ohlcv(symbol, n_days)
    return compute_indicators(df)


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Tính toán toàn bộ chỉ báo kỹ thuật từ OHLCV DataFrame.
    Input:  df với columns [date, open, high, low, close, volume]
    Output: dict chứa tất cả chỉ báo, dùng cho ptkt_agent và screener.
    """
    if df.empty or len(df) < 20:
        return {}

    close  = df["close"]
    volume = df["volume"]

    result = {}
    ta_values = _compute_standard_indicators(df)

    # ── Moving Averages ──
    result["ma20"]  = _last(ta_values.get("sma20"))
    result["ma60"]  = _last(ta_values.get("sma60")) if len(df) >= 60  else None
    result["ma200"] = _last(ta_values.get("sma200")) if len(df) >= 200 else None
    result["ema20"] = _last(ta_values.get("ema20"))
    result["ema50"] = _last(ta_values.get("ema50")) if len(df) >= 50 else None
    result["ema200"] = _last(ta_values.get("ema200")) if len(df) >= 200 else None
    result["vwma20"] = _last(ta_values.get("vwma20"))
    result["current_price"] = _last(close)

    # ── RSI(14) ──
    result["rsi"] = _last(ta_values.get("rsi14"))

    # ── MACD(12,26,9) ──
    result["macd"]        = _last(ta_values.get("macd"))
    result["macd_signal"] = _last(ta_values.get("macd_signal"))
    result["macd_hist"]   = _last(ta_values.get("macd_hist"))
    result["macd_bullish_cross_recent"] = _crossed_above_recent(
        ta_values.get("macd"),
        ta_values.get("macd_signal"),
        lookback=4,
    )

    # ── Bollinger Bands(20,2) ──
    result["bb_upper"] = _last(ta_values.get("bb_upper"))
    result["bb_mid"]   = _last(ta_values.get("bb_mid"))
    result["bb_lower"] = _last(ta_values.get("bb_lower"))
    bb_width = _bb_width_series(
        ta_values.get("bb_upper"),
        ta_values.get("bb_mid"),
        ta_values.get("bb_lower"),
    )
    result["bb_width"] = _last(bb_width)
    result["bb_width_min20"] = _last(bb_width.dropna().rolling(20).min()) if bb_width is not None else None
    result["bb_percent"] = _last(_bb_percent_series(close, ta_values.get("bb_upper"), ta_values.get("bb_lower")))

    # ── ATR(14) ──
    result["atr"] = _last(ta_values.get("atr14"))
    result["adx_14"] = _last(ta_values.get("adx_14"))
    result["dmp_14"] = _last(ta_values.get("dmp_14"))
    result["dmn_14"] = _last(ta_values.get("dmn_14"))
    result["aroon_up_14"] = _last(ta_values.get("aroon_up_14"))
    result["aroon_down_14"] = _last(ta_values.get("aroon_down_14"))
    result["aroon_osc_14"] = _last(ta_values.get("aroon_osc_14"))
    result["supertrend_10_3"] = _last(ta_values.get("supertrend_10_3"))
    result["supertrend_dir"] = _last(ta_values.get("supertrend_dir"))
    result["willr_14"] = _last(ta_values.get("willr14"))
    result["cmo_9"] = _last(ta_values.get("cmo9"))
    result["stoch_k"] = _last(ta_values.get("stoch_k"))
    result["stoch_d"] = _last(ta_values.get("stoch_d"))
    result["roc_9"] = _last(ta_values.get("roc9"))
    result["mom_10"] = _last(ta_values.get("mom10"))
    result["kc_lower"] = _last(ta_values.get("kc_lower"))
    result["kc_mid"] = _last(ta_values.get("kc_mid"))
    result["kc_upper"] = _last(ta_values.get("kc_upper"))
    kc_width = _channel_width_series(ta_values.get("kc_upper"), ta_values.get("kc_mid"), ta_values.get("kc_lower"))
    result["kc_width"] = _last(kc_width)
    result["stdev_14"] = _last(ta_values.get("stdev14"))
    result["linreg_14"] = _last(ta_values.get("linreg14"))
    result["linreg_slope_14"] = _last(_diff_series(ta_values.get("linreg14"), periods=5))
    result["obv"] = _last(ta_values.get("obv"))
    result["obv_ema20"] = _last(_ema_series(ta_values.get("obv"), span=20))

    # ── Volume ──
    result["volume_current"] = _last(volume)
    result["volume_ma20"]    = _last(volume.rolling(20).mean())
    result["volume_ratio_20"] = _safe_ratio(result["volume_current"], result["volume_ma20"])

    # Ichimoku(9,26,52). Keep standard parameters used by most VN practitioners.
    result.update(compute_ichimoku(df))

    # ── Tín hiệu dẫn xuất ──
    result["rsi_signal"]          = _rsi_signal(result["rsi"])
    result["ma_trend"]            = _ma_trend(result)
    result["ma_phase"]            = _ma_phase(result)
    result["macd_signal_label"]   = _macd_signal(result)
    result["bollinger_position"]  = _bollinger_position(result)
    result["volume_surge"]        = _volume_surge(result)

    # ── Support / Resistance ──
    sr = compute_support_resistance(df)
    result["support_levels"]    = sr["support"]
    result["resistance_levels"] = sr["resistance"]

    # ── Confluence score ──
    result["confluence_score"] = compute_confluence_score(result)

    # ── Price-Volume Intelligence Layer ──
    pv = analyze_price_volume(df)
    result["price_volume"] = pv
    # Flatten key scalars for quick access without dict nesting
    result["price_volume_score"] = pv.get("price_volume_score", 0)
    result["pv_entry_bias"]      = pv.get("entry_bias", "neutral")
    result["pv_risk_flags"]      = pv.get("risk_flags", [])
    result["pv_setup_tags"]      = pv.get("setup_tags", [])

    return result


def _compute_standard_indicators(df: pd.DataFrame) -> dict:
    """Compute standard indicators, preferring vnstock_ta and falling back to pandas_ta."""
    if VnstockTAIndicator is not None:
        try:
            indicator = VnstockTAIndicator(data=_vnstock_ta_frame(df))
            macd = indicator.macd(fast=12, slow=26, signal=9)
            bb = indicator.bbands(length=20, std=2)
            adx = indicator.adx(length=14)
            aroon = indicator.aroon(length=14)
            supertrend = indicator.supertrend(length=10, multiplier=3)
            stoch = indicator.stoch(k=14, d=3, smooth_k=3)
            kc = indicator.kc(length=20, scalar=2.0, mamode="ema")
            return {
                "sma20": indicator.sma(length=20),
                "sma60": indicator.sma(length=60),
                "sma200": indicator.sma(length=200),
                "ema20": indicator.ema(length=20),
                "ema50": indicator.ema(length=50),
                "ema200": indicator.ema(length=200),
                "vwma20": indicator.vwma(length=20),
                "rsi14": indicator.rsi(length=14),
                "macd": _prefixed_col(macd, "MACD_"),
                "macd_signal": _prefixed_col(macd, "MACDs_"),
                "macd_hist": _prefixed_col(macd, "MACDh_"),
                "bb_lower": _prefixed_col(bb, "BBL_"),
                "bb_mid": _prefixed_col(bb, "BBM_"),
                "bb_upper": _prefixed_col(bb, "BBU_"),
                "atr14": indicator.atr(length=14),
                "adx_14": _prefixed_col(adx, "ADX_"),
                "dmp_14": _prefixed_col(adx, "DMP_"),
                "dmn_14": _prefixed_col(adx, "DMN_"),
                "aroon_up_14": _prefixed_col(aroon, "AROONU_"),
                "aroon_down_14": _prefixed_col(aroon, "AROOND_"),
                "aroon_osc_14": _prefixed_col(aroon, "AROONOSC_"),
                "supertrend_10_3": _prefixed_col(supertrend, "SUPERT_"),
                "supertrend_dir": _prefixed_col(supertrend, "SUPERTd_"),
                "willr14": indicator.willr(length=14),
                "cmo9": indicator.cmo(length=9),
                "stoch_k": _prefixed_col(stoch, "STOCHk_"),
                "stoch_d": _prefixed_col(stoch, "STOCHd_"),
                "roc9": indicator.roc(length=9),
                "mom10": indicator.mom(length=10),
                "kc_lower": _prefixed_col(kc, "KCL"),
                "kc_mid": _prefixed_col(kc, "KCB"),
                "kc_upper": _prefixed_col(kc, "KCU"),
                "stdev14": indicator.stdev(length=14, ddof=1),
                "linreg14": indicator.linreg(length=14),
                "obv": indicator.obv(),
            }
        except Exception:
            pass

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    bb = ta.bbands(close, length=20, std=2)
    adx = ta.adx(high, low, close, length=14)
    aroon = ta.aroon(high, low, length=14)
    supertrend = ta.supertrend(high, low, close, length=10, multiplier=3)
    stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    kc = ta.kc(high, low, close, length=20, scalar=2.0, mamode="ema")
    return {
        "sma20": close.rolling(20).mean(),
        "sma60": close.rolling(60).mean(),
        "sma200": close.rolling(200).mean(),
        "ema20": ta.ema(close, length=20),
        "ema50": ta.ema(close, length=50),
        "ema200": ta.ema(close, length=200),
        "vwma20": ta.vwma(close, volume, length=20),
        "rsi14": ta.rsi(close, length=14),
        "macd": _prefixed_col(macd, "MACD_"),
        "macd_signal": _prefixed_col(macd, "MACDs_"),
        "macd_hist": _prefixed_col(macd, "MACDh_"),
        "bb_lower": _prefixed_col(bb, "BBL_"),
        "bb_mid": _prefixed_col(bb, "BBM_"),
        "bb_upper": _prefixed_col(bb, "BBU_"),
        "atr14": ta.atr(high, low, close, length=14),
        "adx_14": _prefixed_col(adx, "ADX_"),
        "dmp_14": _prefixed_col(adx, "DMP_"),
        "dmn_14": _prefixed_col(adx, "DMN_"),
        "aroon_up_14": _prefixed_col(aroon, "AROONU_"),
        "aroon_down_14": _prefixed_col(aroon, "AROOND_"),
        "aroon_osc_14": _prefixed_col(aroon, "AROONOSC_"),
        "supertrend_10_3": _prefixed_col(supertrend, "SUPERT_"),
        "supertrend_dir": _prefixed_col(supertrend, "SUPERTd_"),
        "willr14": ta.willr(high, low, close, length=14),
        "cmo9": ta.cmo(close, length=9),
        "stoch_k": _prefixed_col(stoch, "STOCHk_"),
        "stoch_d": _prefixed_col(stoch, "STOCHd_"),
        "roc9": ta.roc(close, length=9),
        "mom10": ta.mom(close, length=10),
        "kc_lower": _prefixed_col(kc, "KCL"),
        "kc_mid": _prefixed_col(kc, "KCB"),
        "kc_upper": _prefixed_col(kc, "KCU"),
        "stdev14": ta.stdev(close, length=14, ddof=1),
        "linreg14": ta.linreg(close, length=14),
        "obv": ta.obv(close, volume),
    }


def _vnstock_ta_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV data in the indexed shape expected by vnstock_ta."""
    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        return data.set_index("date")
    if "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"])
        return data.set_index("time")
    return data


def _prefixed_col(frame: pd.DataFrame | None, prefix: str):
    if frame is None or frame.empty:
        return None
    col = next((c for c in frame.columns if str(c).startswith(prefix)), None)
    return frame[col] if col else None


def _safe_ratio(numerator, denominator) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    try:
        return float(numerator) / float(denominator)
    except Exception:
        return None


def _crossed_above_recent(left, right, lookback: int) -> bool:
    left_series = pd.Series(left).dropna() if left is not None else pd.Series(dtype=float)
    right_series = pd.Series(right).dropna() if right is not None else pd.Series(dtype=float)
    aligned = pd.concat([left_series, right_series], axis=1).dropna()
    if len(aligned) < lookback:
        return False
    a = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    return any(
        a.iloc[i - 1] < b.iloc[i - 1] and a.iloc[i] >= b.iloc[i]
        for i in range(-lookback, 0)
    )


def _bb_width_series(upper, mid, lower) -> pd.Series | None:
    if upper is None or mid is None or lower is None:
        return None
    data = pd.concat(
        [pd.Series(upper), pd.Series(mid), pd.Series(lower)],
        axis=1,
    ).dropna()
    if data.empty:
        return None
    width = (data.iloc[:, 0] - data.iloc[:, 2]) / data.iloc[:, 1]
    return width.replace([float("inf"), float("-inf")], pd.NA)


def _bb_percent_series(close, upper, lower) -> pd.Series | None:
    if close is None or upper is None or lower is None:
        return None
    data = pd.concat([pd.Series(close), pd.Series(upper), pd.Series(lower)], axis=1).dropna()
    if data.empty:
        return None
    denom = (data.iloc[:, 1] - data.iloc[:, 2]).replace(0, pd.NA)
    return ((data.iloc[:, 0] - data.iloc[:, 2]) / denom).replace([float("inf"), float("-inf")], pd.NA)


def _channel_width_series(upper, mid, lower) -> pd.Series | None:
    return _bb_width_series(upper, mid, lower)


def _diff_series(series, periods: int = 1) -> pd.Series | None:
    if series is None:
        return None
    return pd.Series(series).diff(periods)


def _ema_series(series, span: int) -> pd.Series | None:
    if series is None:
        return None
    return pd.Series(series).ewm(span=span, adjust=False, min_periods=span).mean()


def get_ichimoku_config() -> tuple[int, int, int, int, int]:
    """Return Ichimoku params: tenkan, kijun, senkou_b, displacement, chikou."""
    raw = os.environ.get("ICHIMOKU_PARAMS", "").strip()
    if not raw:
        return 9, 26, 52, 26, 26
    try:
        values = [int(part.strip()) for part in raw.replace("/", ",").split(",") if part.strip()]
    except ValueError:
        return 9, 26, 52, 26, 26
    if len(values) == 4:
        tenkan, kijun, senkou_b, displacement = values
        return tenkan, kijun, senkou_b, displacement, displacement
    if len(values) >= 5:
        return tuple(values[:5])  # type: ignore[return-value]
    return 9, 26, 52, 26, 26


def compute_ichimoku(
    df: pd.DataFrame,
    tenkan_period: int | None = None,
    kijun_period: int | None = None,
    senkou_b_period: int | None = None,
    displacement: int | None = None,
    chikou_lookback: int | None = None,
) -> dict:
    """Compute latest Ichimoku values without look-ahead."""
    if None in {tenkan_period, kijun_period, senkou_b_period, displacement, chikou_lookback}:
        tenkan_period, kijun_period, senkou_b_period, displacement, chikou_lookback = get_ichimoku_config()
    empty = {
        "ichimoku_tenkan": None,
        "ichimoku_kijun": None,
        "ichimoku_senkou_a": None,
        "ichimoku_senkou_b": None,
        "ichimoku_cloud_top": None,
        "ichimoku_cloud_bottom": None,
        "ichimoku_future_senkou_a": None,
        "ichimoku_future_senkou_b": None,
        "ichimoku_future_cloud_green": False,
        "ichimoku_chikou_confirm": False,
        "ichimoku_regime": "UNKNOWN",
        "ichimoku_kijun_slope": None,
    }
    min_len = max(tenkan_period, kijun_period, senkou_b_period) + max(displacement, chikou_lookback)
    if df.empty or len(df) < min_len:
        return empty

    high = df["high"]
    low = df["low"]
    close = df["close"]

    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2
    senkou_a_raw = (tenkan + kijun) / 2
    senkou_b_raw = (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2

    # Values visible at the current bar were projected `displacement` bars ago.
    senkou_a = senkou_a_raw.shift(displacement)
    senkou_b = senkou_b_raw.shift(displacement)

    cur_close = _last(close)
    cur_a = _last(senkou_a)
    cur_b = _last(senkou_b)
    future_a = _last(senkou_a_raw)
    future_b = _last(senkou_b_raw)
    cloud_top = max(cur_a, cur_b) if cur_a is not None and cur_b is not None else None
    cloud_bottom = min(cur_a, cur_b) if cur_a is not None and cur_b is not None else None

    if cur_close is None or cloud_top is None or cloud_bottom is None:
        regime = "UNKNOWN"
    elif cur_close > cloud_top:
        regime = "BULLISH"
    elif cur_close < cloud_bottom:
        regime = "BEARISH"
    else:
        regime = "NEUTRAL"

    chikou_confirm = False
    if len(close) > chikou_lookback:
        past_close = float(close.iloc[-1 - chikou_lookback])
        chikou_confirm = bool(cur_close is not None and cur_close > past_close)

    kijun_slope = None
    kijun_clean = kijun.dropna()
    if len(kijun_clean) >= 6:
        kijun_slope = float(kijun_clean.iloc[-1] - kijun_clean.iloc[-6])

    return {
        "ichimoku_tenkan": _last(tenkan),
        "ichimoku_kijun": _last(kijun),
        "ichimoku_senkou_a": cur_a,
        "ichimoku_senkou_b": cur_b,
        "ichimoku_cloud_top": cloud_top,
        "ichimoku_cloud_bottom": cloud_bottom,
        "ichimoku_future_senkou_a": future_a,
        "ichimoku_future_senkou_b": future_b,
        "ichimoku_future_cloud_green": bool(future_a is not None and future_b is not None and future_a >= future_b),
        "ichimoku_chikou_confirm": chikou_confirm,
        "ichimoku_regime": regime,
        "ichimoku_kijun_slope": kijun_slope,
    }


def compute_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Tính S/R đơn giản dựa trên local min/max trong cửa sổ n phiên.
    Trả về tối đa 3 mức support và 3 mức resistance gần nhất.
    """
    if df.empty or len(df) < window:
        return {"support": [], "resistance": []}

    close   = df["close"]
    current = close.iloc[-1]

    # Local max (resistance) và local min (support)
    supports    = []
    resistances = []

    for i in range(window, len(df) - 1):
        window_low  = df["low"].iloc[i - window: i + 1]
        window_high = df["high"].iloc[i - window: i + 1]

        if df["low"].iloc[i] == window_low.min():
            supports.append(round(float(df["low"].iloc[i]), 0))

        if df["high"].iloc[i] == window_high.max():
            resistances.append(round(float(df["high"].iloc[i]), 0))

    # Lọc: support < current, resistance > current
    supports    = sorted(set(s for s in supports    if s < current), reverse=True)[:3]
    resistances = sorted(set(r for r in resistances if r > current))[:3]

    return {"support": supports, "resistance": resistances}


def compute_confluence_score(ind: dict) -> int:
    """
    Điểm hội tụ kỹ thuật 0-10.
    Dùng trong ptkt_agent để đánh giá chất lượng setup.
    """
    score = 0
    rsi   = ind.get("rsi")
    price = ind.get("current_price")
    ma20  = ind.get("ma20")
    ma60  = ind.get("ma60")
    ma200 = ind.get("ma200")

    # +2: RSI 40–70 (vùng khỏe, không quá mua/quá bán)
    if rsi and 40 <= rsi <= 70:
        score += 2

    # +2: Giá trên MA20 > MA60 > MA200 (stack uptrend)
    if price and ma20 and ma60 and ma200:
        if price > ma20 > ma60 > ma200:
            score += 2

    # +2: MACD > Signal (momentum tăng)
    macd = ind.get("macd")
    sig  = ind.get("macd_signal")
    if macd is not None and sig is not None and macd > sig:
        score += 2

    # +2: Volume > TB20 (có thanh khoản)
    vol     = ind.get("volume_current")
    vol_ma  = ind.get("volume_ma20")
    if vol and vol_ma and vol > vol_ma:
        score += 2

    # +2: Gần support (cách support trong nhất < 3%)
    supports = ind.get("support_levels", [])
    if supports and price:
        nearest_support = supports[0]
        dist = (price - nearest_support) / price * 100
        if dist < 3:
            score += 2

    return min(score, 10)


# ──────────────────────────────────────────────
# Helpers nội bộ
# ──────────────────────────────────────────────

def _last(series) -> float | None:
    """Trả về giá trị cuối cùng không NaN."""
    if series is None:
        return None
    s = pd.Series(series).dropna()
    return float(s.iloc[-1]) if len(s) > 0 else None


def _rsi_signal(rsi: float | None) -> str:
    if rsi is None:
        return "UNKNOWN"
    if rsi >= 70:
        return "OVERBOUGHT"
    if rsi <= 30:
        return "OVERSOLD"
    if rsi >= 55:
        return "BULLISH"
    if rsi <= 45:
        return "BEARISH"
    return "NEUTRAL"


def _ma_trend(ind: dict) -> str:
    price = ind.get("current_price")
    ma20  = ind.get("ma20")
    ma60  = ind.get("ma60")
    ma200 = ind.get("ma200")

    if price and ma20 and ma60 and ma200:
        if price > ma20 > ma60 > ma200:
            return "UPTREND"
        if price < ma20 < ma60:
            return "DOWNTREND"
    return "SIDEWAY"


def _ma_phase(ind: dict) -> str:
    price = ind.get("current_price")
    ma20  = ind.get("ma20")
    rsi   = ind.get("rsi")

    if not price or not ma20:
        return "UNKNOWN"

    dist = (price - ma20) / ma20 * 100

    if (rsi and rsi > 75) or dist > 10:
        return "OVERBOUGHT"
    if price < ma20 * 1.02:
        return "PULLBACK"
    return "HEALTHY"


def _macd_signal(ind: dict) -> str:
    macd = ind.get("macd")
    sig  = ind.get("macd_signal")
    hist = ind.get("macd_hist")

    if macd is None or sig is None:
        return "UNKNOWN"
    if macd > sig and hist and hist > 0:
        return "BULLISH"
    if macd < sig and hist and hist < 0:
        return "BEARISH"
    return "NEUTRAL"


def _bollinger_position(ind: dict) -> str:
    price = ind.get("current_price")
    upper = ind.get("bb_upper")
    lower = ind.get("bb_lower")
    mid   = ind.get("bb_mid")

    if not all([price, upper, lower, mid]):
        return "UNKNOWN"

    if price >= upper:
        return "UPPER"
    if price <= lower:
        return "LOWER"
    if price >= mid:
        return "UPPER_HALF"
    return "LOWER_HALF"


def _volume_surge(ind: dict) -> bool:
    vol    = ind.get("volume_current")
    vol_ma = ind.get("volume_ma20")
    if vol and vol_ma and vol_ma > 0:
        return vol > vol_ma * 1.5
    return False
