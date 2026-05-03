import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
import os

import pandas as pd
import pandas_ta as ta

from multiagents_trading_assistant.fetcher import get_ohlcv


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
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    result = {}

    # ── Moving Averages ──
    result["ma20"]  = _last(close.rolling(20).mean())
    result["ma60"]  = _last(close.rolling(60).mean()) if len(df) >= 60  else None
    result["ma200"] = _last(close.rolling(200).mean()) if len(df) >= 200 else None
    result["current_price"] = _last(close)

    # ── RSI(14) ──
    rsi_series = ta.rsi(close, length=14)
    result["rsi"] = _last(rsi_series)

    # ── MACD(12,26,9) ──
    # pandas_ta columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        macd_cols = macd_df.columns.tolist()
        macd_line = next((c for c in macd_cols if c.startswith("MACD_")),  None)
        macd_sig  = next((c for c in macd_cols if c.startswith("MACDs_")), None)
        macd_hist = next((c for c in macd_cols if c.startswith("MACDh_")), None)
        result["macd"]        = _last(macd_df[macd_line])  if macd_line else None
        result["macd_signal"] = _last(macd_df[macd_sig])   if macd_sig  else None
        result["macd_hist"]   = _last(macd_df[macd_hist])  if macd_hist else None
    else:
        result["macd"] = result["macd_signal"] = result["macd_hist"] = None

    # ── Bollinger Bands(20,2) ──
    # pandas_ta columns: BBL_20_2.0 (lower), BBM_20_2.0 (mid), BBU_20_2.0 (upper)
    bb = ta.bbands(close, length=20, std=2)
    if bb is not None and not bb.empty:
        bb_cols = bb.columns.tolist()
        bbl = next((c for c in bb_cols if c.startswith("BBL_")), None)
        bbm = next((c for c in bb_cols if c.startswith("BBM_")), None)
        bbu = next((c for c in bb_cols if c.startswith("BBU_")), None)
        result["bb_upper"] = _last(bb[bbu]) if bbu else None
        result["bb_mid"]   = _last(bb[bbm]) if bbm else None
        result["bb_lower"] = _last(bb[bbl]) if bbl else None
    else:
        result["bb_upper"] = result["bb_mid"] = result["bb_lower"] = None

    # ── ATR(14) ──
    atr_series = ta.atr(high, low, close, length=14)
    result["atr"] = _last(atr_series)

    # ── Volume ──
    result["volume_current"] = _last(volume)
    result["volume_ma20"]    = _last(volume.rolling(20).mean())

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

    return result


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
