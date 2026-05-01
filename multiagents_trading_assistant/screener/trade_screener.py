"""trade_screener.py — TA screener cho Trade pipeline.

14 chiến lược: BREAKOUT / FLAG_PENNANT / BB_SQUEEZE / RETEST / SPRING / GOLDEN_CROSS /
DOUBLE_BOTTOM / MOMENTUM_SURGE / MACD_CROSSOVER / MA_PULLBACK / INSIDE_BAR / NR7 / HAMMER / RSI_BOUNCE
Giữ nguyên logic từ _legacy/screener.py — chỉ đổi tên class và path.

Output: top 10 TradeCandidate sort theo priority_score.

VinGroup distortion detection:
  VNI là chỉ số vốn hóa lớn — VIC/VHM/VRE chiếm ~11% trọng số.
  Khi VinGroup phân kỳ so với thị trường rộng, VNI gây hiểu nhầm xu hướng.
  get_market_context() so sánh VNI vs VNMidCap để phát hiện méo và đặt
  reference_trend là tín hiệu thực tế cho pipeline quyết định.
"""

import importlib.metadata  # noqa: F401 — pandas-ta-openbb Python 3.11 fix
from dataclasses import dataclass, field

import pandas as pd

from multiagents_trading_assistant.services.data_service import (
    get_ohlcv,
    get_ohlcv_batch,
    get_vnindex,
    get_vnmidcap,
    get_vn30_symbols,
    get_liquid_symbols,
)
from multiagents_trading_assistant.indicators import compute_indicators
from multiagents_trading_assistant.agents.trade.money_flow_agent import (
    add_money_flow_features,
    classify_money_flow,
)


# ──────────────────────────────────────────────
# VinGroup constants
# ──────────────────────────────────────────────

# Trọng số ước tính trong VN-Index (2024-2025, market cap weighted)
_VINGROUP_WEIGHTS: dict[str, float] = {
    "VHM": 0.055,   # Vinhomes — lớn nhất nhóm
    "VIC": 0.045,   # Vingroup holding
    "VRE": 0.012,   # Vincom Retail
    "VPL": 0.004,   # Vinpearl
}
_VINGROUP_TOTAL_WEIGHT = sum(_VINGROUP_WEIGHTS.values())  # ~11.6%

# Ngưỡng phân kỳ 1 ngày giữa VNI và VNMidCap để coi là "méo"
_DISTORTION_DAILY_THRESHOLD = 0.5   # 0.5% chênh lệch thay đổi trong ngày


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class MarketContext:
    """Bức tranh tổng quan thị trường — gate đầu tiên cho Trade pipeline.

    reference_trend là tín hiệu thực tế cần dùng để ra quyết định.
    Khi VNI bị méo bởi VinGroup, reference_trend = vnmidcap_trend.
    """
    # VN-Index raw
    trend: str            # UPTREND | SIDEWAY | DOWNTREND (VNI thuần)
    should_trade: bool    # Dựa theo reference_trend, không phải VNI raw
    ma20: float
    ma60: float
    ma200: float
    sideway_zone: str     # NEAR_SUPPORT | MID_RANGE | NEAR_RESISTANCE
    position_in_range: float
    vni_change_pct: float
    current_price: float

    # VNMidCap & VinGroup distortion
    vnmidcap_trend: str              # UPTREND | SIDEWAY | DOWNTREND | UNKNOWN
    vnmidcap_change_pct: float       # % thay đổi VNMidCap trong ngày
    is_index_distorted: bool         # True khi VNI ≠ VNMidCap đáng kể
    reference_trend: str             # Trend thực tế để pipeline dùng
    reference_index: str             # "VNINDEX" hoặc "VNMIDCAP"
    vingroup_contribution_pct: float # Ước tính đóng góp VinGroup vào VNI hôm nay (điểm %)
    distortion_note: str             # Mô tả ngắn gọn lý do phân kỳ


@dataclass
class TradeCandidate:
    """Một mã pass TA screener — truyền vào trade_graph."""
    symbol: str
    setup_type: str        # BREAKOUT | RETEST | SPRING | MA_PULLBACK | RSI_BOUNCE
    priority_score: float
    market_context: MarketContext
    indicators: dict
    reasons: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# Market context helpers
# ──────────────────────────────────────────────

def _extract_trend(df: pd.DataFrame) -> tuple[str, float, float, float, float, float]:
    """Tính trend + MA từ một DataFrame OHLCV.

    Returns:
        (trend, current_price, ma20, ma60, ma200, change_pct_today)
    """
    close   = df["close"]
    current = float(close.iloc[-1])
    prev    = float(close.iloc[-2]) if len(close) >= 2 else current

    ma20  = float(close.rolling(20).mean().iloc[-1])
    ma60  = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60  else ma20
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma60

    if current > ma20 > ma60 > ma200:
        trend = "UPTREND"
    elif current < ma20 < ma60:
        trend = "DOWNTREND"
    else:
        trend = "SIDEWAY"

    change_pct = round((current - prev) / prev * 100, 2) if prev else 0.0
    return trend, current, ma20, ma60, ma200, change_pct


def _estimate_vingroup_contribution(vingroup_changes: dict[str, float]) -> float:
    """Ước tính đóng góp của VinGroup vào VNI hôm nay (điểm %).

    contribution = sum(weight_i × change_i) — xấp xỉ tuyến tính.
    Kết quả tính bằng điểm phần trăm (không phải tỷ lệ).
    """
    total = 0.0
    for sym, weight in _VINGROUP_WEIGHTS.items():
        chg = vingroup_changes.get(sym)
        if chg is not None:
            total += weight * chg
    return round(total, 3)


def _detect_distortion(
    vni_trend: str,
    vni_change: float,
    vnmidcap_trend: str,
    vnmidcap_change: float,
    vingroup_contribution: float,
) -> tuple[bool, str, str, str]:
    """Phát hiện VNI bị méo bởi VinGroup.

    Returns:
        (is_distorted, reference_trend, reference_index, distortion_note)
    """
    trend_diverges  = vni_trend != vnmidcap_trend and vnmidcap_trend != "UNKNOWN"
    daily_divergence = abs(vni_change - vnmidcap_change)
    change_distorted = daily_divergence >= _DISTORTION_DAILY_THRESHOLD

    if not (trend_diverges or change_distorted):
        return False, vni_trend, "VNINDEX", ""

    # Xác định hướng méo
    if vni_change > vnmidcap_change:
        direction = f"VIC/VHM/VRE đang kéo VNI lên ({vingroup_contribution:+.2f}% đóng góp ước tính)"
    else:
        direction = f"VIC/VHM/VRE đang kéo VNI xuống ({vingroup_contribution:+.2f}% đóng góp ước tính)"

    note = (
        f"VNI {vni_trend} ({vni_change:+.1f}%) ≠ VNMidCap {vnmidcap_trend} ({vnmidcap_change:+.1f}%) — "
        f"{direction}. Dùng VNMidCap làm tham chiếu."
    )

    return True, vnmidcap_trend, "VNMIDCAP", note


# ──────────────────────────────────────────────
# Market context
# ──────────────────────────────────────────────

def get_market_context(vnindex_df: pd.DataFrame | None = None) -> MarketContext:
    """Phân tích trạng thái thị trường, có bổ sung phát hiện méo VinGroup.

    Flow:
      1. Lấy VNI trend + MA (như cũ)
      2. Lấy VNMidCap — 1 API call bổ sung, cache theo ngày
      3. Lấy OHLCV VinGroup (VIC/VHM/VRE/VPL) để tính contribution — cache
      4. Phát hiện phân kỳ VNI vs VNMidCap → đặt reference_trend
    """
    if vnindex_df is None or vnindex_df.empty:
        vnindex_df = get_vnindex(200)
    if vnindex_df.empty:
        raise RuntimeError("Không lấy được dữ liệu VN-Index — abort trade_screener")

    # ── VNI: trend, MA, change ──
    vni_trend, vni_price, ma20, ma60, ma200, vni_change = _extract_trend(vnindex_df)

    # ── VNI: sideway zone ──
    close   = vnindex_df["close"]
    high_20 = float(close.rolling(20).max().iloc[-1])
    low_20  = float(close.rolling(20).min().iloc[-1])
    rng     = high_20 - low_20
    pos     = (vni_price - low_20) / rng if rng > 0 else 0.5
    if pos <= 0.25:
        zone = "NEAR_SUPPORT"
    elif pos >= 0.75:
        zone = "NEAR_RESISTANCE"
    else:
        zone = "MID_RANGE"

    # ── VNMidCap: trend + daily change ──
    vnmidcap_trend  = "UNKNOWN"
    vnmidcap_change = 0.0
    try:
        vnmidcap_df = get_vnmidcap(200)
        if not vnmidcap_df.empty:
            vnmidcap_trend, _, _, _, _, vnmidcap_change = _extract_trend(vnmidcap_df)
            print(f"[market_ctx] VNMidCap: {vnmidcap_trend} ({vnmidcap_change:+.2f}%)")
    except Exception as e:
        print(f"[market_ctx] VNMidCap lỗi — bỏ qua: {e}")

    # ── VinGroup: ước tính đóng góp vào VNI ──
    vingroup_contribution = 0.0
    try:
        ving_changes: dict[str, float] = {}
        for sym in _VINGROUP_WEIGHTS:
            df_sym = get_ohlcv(sym, 5)   # chỉ cần 5 nến gần nhất, dùng cache
            if not df_sym.empty and len(df_sym) >= 2:
                c_cur  = float(df_sym["close"].iloc[-1])
                c_prev = float(df_sym["close"].iloc[-2])
                ving_changes[sym] = round((c_cur - c_prev) / c_prev * 100, 2) if c_prev else 0.0
        vingroup_contribution = _estimate_vingroup_contribution(ving_changes)
        ving_str = " | ".join(f"{s} {v:+.1f}%" for s, v in ving_changes.items())
        print(f"[market_ctx] VinGroup: [{ving_str}] → đóng góp VNI ≈ {vingroup_contribution:+.3f}%")
    except Exception as e:
        print(f"[market_ctx] VinGroup fetch lỗi — bỏ qua: {e}")

    # ── Phát hiện méo ──
    is_distorted, reference_trend, reference_index, distortion_note = _detect_distortion(
        vni_trend, vni_change, vnmidcap_trend, vnmidcap_change, vingroup_contribution,
    )
    if is_distorted:
        print(f"[market_ctx] ⚠ DISTORTION: {distortion_note}")
    else:
        print(f"[market_ctx] VNI {vni_trend} vs MidCap {vnmidcap_trend} — không phân kỳ")

    return MarketContext(
        trend                    = vni_trend,
        should_trade             = reference_trend != "DOWNTREND",
        ma20                     = ma20,
        ma60                     = ma60,
        ma200                    = ma200,
        sideway_zone             = zone,
        position_in_range        = round(pos, 3),
        vni_change_pct           = vni_change,
        current_price            = vni_price,
        vnmidcap_trend           = vnmidcap_trend,
        vnmidcap_change_pct      = vnmidcap_change,
        is_index_distorted       = is_distorted,
        reference_trend          = reference_trend,
        reference_index          = reference_index,
        vingroup_contribution_pct= vingroup_contribution,
        distortion_note          = distortion_note,
    )


# ──────────────────────────────────────────────
# 5 chiến lược TA
# ──────────────────────────────────────────────

def detect_breakout(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if df.empty or len(df) < 22:
        return False, []

    close  = df["close"]
    volume = df["volume"]
    cur    = float(close.iloc[-1])
    prev_h = float(close.iloc[-21:-1].max())
    op     = float(df["open"].iloc[-1])
    v_now  = float(volume.iloc[-1])
    v_ma20 = float(volume.rolling(20).mean().iloc[-1])
    rsi    = ind.get("rsi")

    is_breakout    = cur > prev_h
    is_vol_surge   = v_now > v_ma20 * 1.5
    candle_body    = abs(cur - op) / op * 100
    is_big_candle  = candle_body > 2.0
    not_overbought = rsi is None or rsi < 75

    if is_breakout:
        reasons.append(f"Giá {cur:.0f} vượt đỉnh 20 phiên ({prev_h:.0f})")
    if is_vol_surge:
        reasons.append(f"Volume {v_now/v_ma20:.1f}× TB20")
    if is_big_candle:
        reasons.append(f"Thân nến {candle_body:.1f}%")

    passed = is_breakout and is_vol_surge and is_big_candle and not_overbought
    return passed, reasons if passed else []


def detect_retest(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if df.empty or len(df) < 25:
        return False, []

    close  = df["close"]
    volume = df["volume"]
    cur    = float(close.iloc[-1])
    v_now  = float(volume.iloc[-1])
    v_ma20 = float(volume.rolling(20).mean().iloc[-1])
    bk_lvl = float(close.iloc[-20:-5].max())
    dist   = (cur - bk_lvl) / bk_lvl * 100

    is_near    = -3.0 <= dist <= 5.0
    is_dry     = v_now < v_ma20 * 0.8
    is_holding = cur >= bk_lvl * 0.97

    if is_near:
        reasons.append(f"Retest vùng breakout {bk_lvl:.0f} (cách {dist:+.1f}%)")
    if is_dry:
        reasons.append(f"Volume cạn {v_now/v_ma20:.2f}× TB20")
    if is_holding:
        reasons.append("Giá vẫn giữ trên vùng breakout")

    passed = is_near and is_dry and is_holding
    return passed, reasons if passed else []


def detect_spring(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if df.empty or len(df) < 30:
        return False, []

    close  = df["close"]
    volume = df["volume"]
    window = 20
    recent = df.iloc[-window:]

    local_highs: list[tuple[int, float]] = []
    local_lows:  list[tuple[int, float]] = []
    for i in range(2, len(recent) - 2):
        h = float(recent["high"].iloc[i])
        l = float(recent["low"].iloc[i])
        if h > float(recent["high"].iloc[i-1]) and h > float(recent["high"].iloc[i+1]):
            local_highs.append((i, h))
        if l < float(recent["low"].iloc[i-1]) and l < float(recent["low"].iloc[i+1]):
            local_lows.append((i, l))

    if len(local_highs) < 2 or len(local_lows) < 2:
        return False, []

    hh1, hh2 = local_highs[-2][1], local_highs[-1][1]
    hl1, hl2 = local_lows[-2][1],  local_lows[-1][1]
    is_hh = hh2 > hh1
    is_hl = hl2 > hl1

    up_vol   = volume[close > close.shift(1)].mean()
    down_vol = volume[close < close.shift(1)].mean()
    vol_ok   = (up_vol > down_vol) if (up_vol > 0 and down_vol > 0) else False

    ma20  = ind.get("ma20")
    price = ind.get("current_price")
    above_ma20 = (price > ma20) if (price and ma20) else False

    if is_hh:
        reasons.append(f"HH: đỉnh {hh2:.0f} > {hh1:.0f}")
    if is_hl:
        reasons.append(f"HL: đáy {hl2:.0f} > {hl1:.0f}")
    if vol_ok:
        reasons.append("Volume phiên tăng > giảm")

    passed = is_hh and is_hl and vol_ok and above_ma20
    return passed, reasons if passed else []


def detect_ma_pullback(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    price = ind.get("current_price")
    ma20  = ind.get("ma20")
    ma60  = ind.get("ma60")
    ma200 = ind.get("ma200")
    rsi   = ind.get("rsi")
    if not all([price, ma20, ma60, ma200, rsi]):
        return False, []

    is_uptrend = price > ma20 > ma60 > ma200
    dist       = (price - ma60) / ma60 * 100
    near_ma60  = 0 <= dist <= 8.0
    rsi_ok     = 30 <= rsi <= 60

    vol_declining = False
    if len(df) >= 4:
        v1 = float(df["volume"].iloc[-4])
        v2 = float(df["volume"].iloc[-3])
        v3 = float(df["volume"].iloc[-2])
        vol_declining = (v1 > v2) or (v2 > v3)

    if is_uptrend:
        reasons.append("UPTREND (giá > MA20 > MA60 > MA200)")
    if near_ma60:
        reasons.append(f"Pullback về MA60 (cách {dist:.1f}%)")
    if rsi_ok:
        reasons.append(f"RSI {rsi:.1f} trung tính")
    if vol_declining:
        reasons.append("Volume giảm — áp lực bán cạn")

    passed = is_uptrend and near_ma60 and rsi_ok and vol_declining
    return passed, reasons if passed else []


def detect_rsi_bounce(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    price = ind.get("current_price")
    ma200 = ind.get("ma200")
    rsi   = ind.get("rsi")
    if not all([price, rsi]):
        return False, []

    is_oversold  = rsi < 40
    above_ma200  = (price > ma200) if ma200 else False
    reversal     = False
    if len(df) >= 2:
        cc = float(df["close"].iloc[-1])
        co = float(df["open"].iloc[-1])
        pc = float(df["close"].iloc[-2])
        reversal = cc > co and cc > pc

    if is_oversold:
        reasons.append(f"RSI {rsi:.1f} oversold")
    if above_ma200:
        reasons.append(f"Trên MA200 ({ma200:.0f}) — long-term trend ok")
    if reversal:
        reasons.append("Nến đảo chiều tăng")

    passed = is_oversold and above_ma200 and reversal
    return passed, reasons if passed else []


def detect_flag_pennant(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Flag/Pennant: cột cờ tăng mạnh → nén volume thấp → giá áp đỉnh consolidation."""
    reasons: list[str] = []
    if df.empty or len(df) < 20:
        return False, []

    close  = df["close"]
    volume = df["volume"]

    # Cột cờ: 10 phiên trước vùng nén
    pole   = df.iloc[-15:-5]
    if len(pole) < 5:
        return False, []
    pole_return   = (float(pole["close"].iloc[-1]) - float(pole["close"].iloc[0])) / float(pole["close"].iloc[0]) * 100
    pole_vol_avg  = float(pole["volume"].mean())
    pole_range    = float(pole["high"].max()) - float(pole["low"].min())

    # Vùng nén (flag): 5 phiên gần nhất
    consol         = df.iloc[-5:]
    consol_high    = float(consol["high"].max())
    consol_range   = consol_high - float(consol["low"].min())
    consol_vol_avg = float(consol["volume"].mean())

    cur = float(close.iloc[-1])

    is_strong_pole    = pole_return > 5.0
    is_tight_consol   = pole_range > 0 and consol_range < pole_range * 0.5
    is_low_vol        = consol_vol_avg < pole_vol_avg * 0.8
    is_near_top       = consol_high > 0 and cur >= consol_high * 0.98

    if is_strong_pole:
        reasons.append(f"Cột cờ +{pole_return:.1f}% trong 10 phiên")
    if is_tight_consol:
        reasons.append(f"Nén {consol_range:.0f} điểm (< 50% cột cờ)")
    if is_low_vol:
        reasons.append(f"Volume consolidation {consol_vol_avg/pole_vol_avg:.2f}× phase tăng")
    if is_near_top:
        reasons.append("Giá áp sát đỉnh vùng nén — sắp bung")

    passed = is_strong_pole and is_tight_consol and is_low_vol and is_near_top
    return passed, reasons if passed else []


def detect_golden_cross(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """MA20 cắt lên MA50 trong 5 phiên gần nhất, volume xác nhận."""
    reasons: list[str] = []
    if df.empty or len(df) < 55:
        return False, []

    close = df["close"]
    ma20  = close.rolling(20).mean()
    ma50  = close.rolling(50).mean()

    crossed = any(
        ma20.iloc[i - 1] < ma50.iloc[i - 1] and ma20.iloc[i] >= ma50.iloc[i]
        for i in range(-5, 0)
    )

    cur_ma20 = float(ma20.iloc[-1])
    cur_ma50 = float(ma50.iloc[-1])
    vol_ok   = float(df["volume"].iloc[-1]) > float(df["volume"].rolling(20).mean().iloc[-1])
    rsi      = ind.get("rsi")
    rsi_ok   = rsi is None or rsi < 72

    if crossed:
        reasons.append(f"Golden Cross: MA20 ({cur_ma20:.0f}) cắt lên MA50 ({cur_ma50:.0f})")
    if vol_ok:
        reasons.append("Volume xác nhận cross")

    passed = crossed and vol_ok and rsi_ok
    return passed, reasons if passed else []


def detect_macd_crossover(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """MACD line cắt lên signal line trong 3 phiên gần nhất."""
    import pandas_ta as _ta  # local import — tránh circular nếu có

    reasons: list[str] = []
    if df.empty or len(df) < 35:
        return False, []

    macd_df = _ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd_df is None or macd_df.empty:
        return False, []

    cols   = macd_df.columns.tolist()
    ml_col = next((c for c in cols if c.startswith("MACD_")),  None)
    ms_col = next((c for c in cols if c.startswith("MACDs_")), None)
    mh_col = next((c for c in cols if c.startswith("MACDh_")), None)
    if not (ml_col and ms_col and mh_col):
        return False, []

    ml = macd_df[ml_col].dropna()
    ms = macd_df[ms_col].dropna()
    mh = macd_df[mh_col].dropna()
    if len(ml) < 4:
        return False, []

    crossed = any(
        ml.iloc[i - 1] < ms.iloc[i - 1] and ml.iloc[i] >= ms.iloc[i]
        for i in range(-4, 0)
    )

    hist_val      = float(mh.iloc[-1])
    hist_positive = hist_val > 0
    above_zero    = float(ml.iloc[-1]) > 0
    rsi           = ind.get("rsi")
    rsi_ok        = rsi is None or rsi < 72

    if crossed:
        reasons.append(f"MACD crossover bullish (hist={hist_val:.2f})")
    if above_zero:
        reasons.append("Cross xảy ra trên zero line — signal mạnh hơn")

    passed = crossed and hist_positive and rsi_ok
    return passed, reasons if passed else []


def detect_momentum_surge(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """3 phiên tăng liên tiếp với volume tăng dần — đà tăng tốc."""
    reasons: list[str] = []
    if df.empty or len(df) < 6:
        return False, []

    close  = df["close"]
    volume = df["volume"]

    up3 = (
        float(close.iloc[-1]) > float(close.iloc[-2]) and
        float(close.iloc[-2]) > float(close.iloc[-3]) and
        float(close.iloc[-3]) > float(close.iloc[-4])
    )
    vol_acc = (
        float(volume.iloc[-1]) > float(volume.iloc[-2]) or
        float(volume.iloc[-2]) > float(volume.iloc[-3])
    )

    gain_3d = (float(close.iloc[-1]) - float(close.iloc[-4])) / float(close.iloc[-4]) * 100
    rsi     = ind.get("rsi")
    not_overbought = rsi is None or rsi < 72

    if up3:
        reasons.append(f"3 phiên tăng liên tiếp (+{gain_3d:.1f}% tổng)")
    if vol_acc:
        reasons.append("Volume tăng theo đà giá")

    passed = up3 and vol_acc and not_overbought and gain_3d > 2.0
    return passed, reasons if passed else []


def detect_double_bottom(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Double Bottom (W): hai đáy tương đương, giá vượt neckline."""
    reasons: list[str] = []
    if df.empty or len(df) < 35:
        return False, []

    window = df.iloc[-30:-1]
    local_lows: list[tuple[int, float]] = []
    for i in range(2, len(window) - 2):
        l = float(window["low"].iloc[i])
        if l < float(window["low"].iloc[i - 1]) and l < float(window["low"].iloc[i + 1]):
            local_lows.append((i, l))

    if len(local_lows) < 2:
        return False, []

    # 2 đáy thấp nhất, cách nhau ≥ 5 phiên
    local_lows.sort(key=lambda x: x[1])
    b1_idx, b1 = local_lows[0]
    b2_idx, b2 = local_lows[1]
    if abs(b1_idx - b2_idx) < 5:
        return False, []

    similar  = b2 >= b1 * 0.97

    mid_start = min(b1_idx, b2_idx)
    mid_end   = max(b1_idx, b2_idx)
    neckline  = float(window["high"].iloc[mid_start: mid_end + 1].max())

    cur       = float(df["close"].iloc[-1])
    breakout  = cur > neckline
    not_extended = cur < neckline * 1.08  # chưa quá xa

    if similar:
        reasons.append(f"Hai đáy {b1:.0f} và {b2:.0f} (chênh {abs(b2 - b1) / b1 * 100:.1f}%)")
    if breakout:
        reasons.append(f"Vượt neckline {neckline:.0f}")
    if not_extended:
        reasons.append("Chưa quá xa neckline — vào hàng còn hợp lý")

    passed = similar and breakout and not_extended
    return passed, reasons if passed else []


def detect_hammer(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Hammer / Pin Bar: bóng dưới dài > 2× thân, đóng cửa gần đỉnh nến."""
    reasons: list[str] = []
    if df.empty or len(df) < 5:
        return False, []

    last = df.iloc[-1]
    op   = float(last["open"])
    cl   = float(last["close"])
    hi   = float(last["high"])
    lo   = float(last["low"])

    body         = abs(cl - op)
    lower_shadow = min(op, cl) - lo
    upper_shadow = hi - max(op, cl)
    total_range  = hi - lo

    if total_range == 0 or body == 0:
        return False, []

    is_hammer      = lower_shadow > 2 * body
    small_upper    = upper_shadow < body
    close_near_top = (hi - cl) / total_range < 0.3

    rsi          = ind.get("rsi")
    oversold_area = rsi is None or rsi < 50

    price    = ind.get("current_price")
    supports = ind.get("support_levels", [])
    near_support = bool(supports and price and (price - supports[0]) / price * 100 < 5)

    if is_hammer:
        reasons.append(f"Bóng dưới {lower_shadow:.0f} > 2× thân {body:.0f}")
    if close_near_top:
        reasons.append("Đóng cửa gần đỉnh nến — lực mua hấp thu")
    if near_support and supports:
        reasons.append(f"Gần support {supports[0]:.0f}")
    if rsi and rsi < 45:
        reasons.append(f"RSI {rsi:.1f} — vùng oversold")

    passed = is_hammer and small_upper and close_near_top and oversold_area
    return passed, reasons if passed else []


def detect_bb_squeeze(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Bollinger Band Squeeze: BB thu hẹp tối thiểu 20 phiên, giá áp BB upper."""
    import pandas_ta as _ta

    reasons: list[str] = []
    if df.empty or len(df) < 30:
        return False, []

    bb = _ta.bbands(df["close"], length=20, std=2)
    if bb is None or bb.empty:
        return False, []

    cols    = bb.columns.tolist()
    bbl_col = next((c for c in cols if c.startswith("BBL_")), None)
    bbu_col = next((c for c in cols if c.startswith("BBU_")), None)
    bbm_col = next((c for c in cols if c.startswith("BBM_")), None)
    if not (bbl_col and bbu_col and bbm_col):
        return False, []

    width = ((bb[bbu_col] - bb[bbl_col]) / bb[bbm_col]).dropna()
    if len(width) < 20:
        return False, []

    cur_width  = float(width.iloc[-1])
    min_20     = float(width.iloc[-20:].min())
    is_squeeze = cur_width <= min_20 * 1.05

    upper = ind.get("bb_upper")
    lower = ind.get("bb_lower")
    cur   = float(df["close"].iloc[-1])
    above_mid = upper is not None and lower is not None and cur > (upper + lower) / 2

    vol_rising = float(df["volume"].iloc[-1]) > float(df["volume"].iloc[-2])

    if is_squeeze:
        reasons.append(f"BB squeeze: width {cur_width:.3f} ≈ min 20 phiên ({min_20:.3f})")
    if above_mid:
        reasons.append("Giá trên BB mid — bias bullish")
    if vol_rising:
        reasons.append("Volume bắt đầu tăng — tín hiệu sắp bung")

    passed = is_squeeze and above_mid
    return passed, reasons if passed else []


def detect_inside_bar(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Inside Bar Breakout: nến hôm nay đóng cửa vượt đỉnh inside bar, volume tăng."""
    reasons: list[str] = []
    if df.empty or len(df) < 5:
        return False, []

    # mother bar [-3], inside bar [-2], breakout bar [-1]
    m_hi = float(df["high"].iloc[-3])
    m_lo = float(df["low"].iloc[-3])
    i_hi = float(df["high"].iloc[-2])
    i_lo = float(df["low"].iloc[-2])

    cur_close = float(df["close"].iloc[-1])
    cur_vol   = float(df["volume"].iloc[-1])
    vol_ma20  = float(df["volume"].rolling(20).mean().iloc[-1])

    is_inside   = i_hi <= m_hi and i_lo >= m_lo
    is_breakout = cur_close > i_hi
    vol_ok      = vol_ma20 > 0 and cur_vol > vol_ma20 * 1.2

    rsi    = ind.get("rsi")
    rsi_ok = rsi is None or rsi < 72

    if is_inside:
        reasons.append(f"Inside bar [{i_lo:.0f}–{i_hi:.0f}] trong mother [{m_lo:.0f}–{m_hi:.0f}]")
    if is_breakout:
        reasons.append(f"Đóng cửa {cur_close:.0f} vượt đỉnh IB {i_hi:.0f}")
    if vol_ok:
        reasons.append(f"Volume {cur_vol / vol_ma20:.1f}× TB20 xác nhận breakout")

    passed = is_inside and is_breakout and vol_ok and rsi_ok
    return passed, reasons if passed else []


def detect_pin_bar(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Bullish Pin Bar tại key level: wick dưới ≥ 2.5× thân, upper wick < 0.5× thân,
    đóng cửa trong top 25% range — bắt buộc phải ở gần S/R quan trọng.

    Khác HAMMER:
      - HAMMER không yêu cầu key level (chỉ RSI <50)
      - PIN_BAR yêu cầu tại support level hoặc MA động — context bắt buộc
      - Tỷ lệ wick chặt hơn (2.5× vs 2×)
    """
    reasons: list[str] = []
    if df.empty or len(df) < 5:
        return False, []

    last = df.iloc[-1]
    op   = float(last["open"])
    cl   = float(last["close"])
    hi   = float(last["high"])
    lo   = float(last["low"])

    body         = abs(cl - op)
    lower_shadow = min(op, cl) - lo
    upper_shadow = hi - max(op, cl)
    total_range  = hi - lo

    if total_range == 0 or body == 0:
        return False, []

    is_pin_bar = (
        lower_shadow >= 2.5 * body and       # wick dưới dài
        upper_shadow < 0.5 * body and         # upper wick nhỏ (chặt hơn HAMMER)
        (hi - cl) / total_range < 0.25        # đóng cửa trong top 25%
    )

    if not is_pin_bar:
        return False, []

    # Context bắt buộc: phải ở gần ít nhất 1 key level
    price    = ind.get("current_price")
    supports = ind.get("support_levels", [])
    ma60     = ind.get("ma60")
    ma200    = ind.get("ma200")

    at_support = False
    support_note = ""
    if supports and price:
        nearest = supports[0]
        dist = (price - nearest) / price * 100
        if dist < 3.0:
            at_support = True
            support_note = f"Tại support {nearest:.0f} (cách {dist:.1f}%)"

    at_ma = False
    ma_note = ""
    for ma_val, ma_name in [(ma60, "MA60"), (ma200, "MA200")]:
        if ma_val and price:
            dist_ma = (price - ma_val) / ma_val * 100
            if 0 <= dist_ma <= 3.5:
                at_ma = True
                ma_note = f"Tại {ma_name} ({ma_val:.0f})"
                break

    if not (at_support or at_ma):
        return False, []

    reasons.append(f"Pin Bar: wick dưới {lower_shadow:.0f} ({lower_shadow/body:.1f}× thân)")
    reasons.append(f"Đóng gần đỉnh nến ({(hi - cl) / total_range * 100:.0f}% từ đỉnh)")
    if at_support:
        reasons.append(support_note)
    if at_ma:
        reasons.append(ma_note)

    return True, reasons


def detect_bullish_engulfing(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Bullish Engulfing: nến xanh nuốt hoàn toàn body nến đỏ trước, volume tăng.

    Điều kiện:
      - Bar[-2]: đỏ (close < open)
      - Bar[-1]: xanh, open ≤ prev_close, close ≥ prev_open (bao trùm body)
      - Volume bar xanh > bar đỏ (smart money buying)
      - Body ≥ 1% (có ý nghĩa thống kê)
      - Context: gần support hoặc trong uptrend/sideway

    Khác MOMENTUM_SURGE: Surge cần 3 bars tăng — Engulfing chỉ cần 1 nến đảo chiều mạnh.
    """
    reasons: list[str] = []
    if df.empty or len(df) < 5:
        return False, []

    prev = df.iloc[-2]
    curr = df.iloc[-1]

    prev_op  = float(prev["open"])
    prev_cl  = float(prev["close"])
    curr_op  = float(curr["open"])
    curr_cl  = float(curr["close"])
    curr_vol = float(curr["volume"])
    prev_vol = float(prev["volume"])

    is_prev_bearish = prev_cl < prev_op
    is_engulfing    = (
        curr_cl > curr_op           and   # green candle
        curr_op  <= prev_cl         and   # open at/below prev close
        curr_cl  >= prev_op               # close at/above prev open
    )

    if not (is_prev_bearish and is_engulfing):
        return False, []

    # Body phải có ý nghĩa
    body_pct = abs(curr_cl - curr_op) / curr_op * 100 if curr_op > 0 else 0
    meaningful = body_pct >= 1.0

    # Volume: bar xanh > bar đỏ
    vol_ok = prev_vol > 0 and curr_vol > prev_vol

    # Context
    price    = ind.get("current_price")
    supports = ind.get("support_levels", [])
    ma_trend = ind.get("ma_trend")
    rsi      = ind.get("rsi")

    near_support = bool(supports and price and (price - supports[0]) / price * 100 < 4.0)
    context_ok   = near_support or ma_trend in ("UPTREND", "SIDEWAY")
    rsi_ok       = rsi is None or rsi < 72

    prev_body = abs(prev_op - prev_cl)
    reasons.append(
        f"Engulfing: [{curr_op:.0f}–{curr_cl:.0f}] nuốt [{prev_cl:.0f}–{prev_op:.0f}] "
        f"(body {body_pct:.1f}%)"
    )
    if vol_ok:
        reasons.append(f"Volume xanh {curr_vol / prev_vol:.1f}× bar đỏ")
    if near_support and supports:
        reasons.append(f"Tại support {supports[0]:.0f}")

    passed = meaningful and vol_ok and context_ok and rsi_ok
    return passed, reasons if passed else []


def detect_trend_pullback(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Trend Pullback về old breakout level (kháng cự cũ → hỗ trợ mới).

    Logic VCBS: uptrend → giá pullback về đỉnh cũ vừa vượt → vào lệnh.

    Khác MA_PULLBACK: MA_PULLBACK dùng khoảng cách MA60 làm proxy.
    Setup này xác định bằng price structure (HH+HL) và vùng breakout thực sự,
    không phụ thuộc MA — phản ánh đúng hơn hành động giá.
    """
    reasons: list[str] = []
    if df.empty or len(df) < 40:
        return False, []

    close  = df["close"]
    volume = df["volume"]

    price = ind.get("current_price")
    ma20  = ind.get("ma20")
    ma60  = ind.get("ma60")
    rsi   = ind.get("rsi")

    # Uptrend: MA stack (price > MA20 > MA60)
    is_uptrend = price and ma20 and ma60 and price > ma20 > ma60
    if not is_uptrend:
        return False, []

    # Old resistance = đỉnh vùng 35-15 bars trước
    # Recent high = đỉnh 15 bars gần nhất
    old_resistance = float(close.iloc[-35:-15].max())
    recent_high    = float(close.iloc[-15:].max())
    cur            = float(close.iloc[-1])

    # Phải có breakout thực sự: recent high vượt old resistance ≥ 2%
    was_breakout = recent_high > old_resistance * 1.02
    if not was_breakout:
        return False, []

    # Giá đang pullback về vùng old resistance (giờ là support)
    dist = (cur - old_resistance) / old_resistance * 100
    in_retest_zone = -2.0 <= dist <= 6.0

    # Đã pullback ít nhất 2% từ recent high (không mua đuổi)
    pullback_from_high = (recent_high - cur) / recent_high * 100
    is_pullback = pullback_from_high >= 2.0

    # Volume cạn (áp lực bán suy yếu trong pullback)
    vol_ma20 = float(volume.rolling(20).mean().iloc[-1])
    vol_dry  = vol_ma20 > 0 and float(volume.iloc[-1]) < vol_ma20 * 0.9

    # Nến xác nhận: đóng cửa xanh
    is_bullish_candle = float(df["close"].iloc[-1]) > float(df["open"].iloc[-1])

    rsi_ok = rsi is None or rsi < 65

    if is_uptrend:
        reasons.append(f"Uptrend: {cur:.0f} > MA20({ma20:.0f}) > MA60({ma60:.0f})")
    if was_breakout:
        reasons.append(f"Breakout vùng {old_resistance:.0f} → recent high {recent_high:.0f}")
    if in_retest_zone:
        reasons.append(f"Pullback về vùng breakout (cách {dist:+.1f}%)")
    if vol_dry:
        reasons.append("Volume cạn — áp lực bán suy yếu")

    passed = (
        in_retest_zone and is_pullback and
        (vol_dry or is_bullish_candle) and rsi_ok
    )
    return passed, reasons if passed else []


def detect_breakout_retest_entry(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Breakout đã xảy ra 3-15 bars trước, giờ giá retest level → vào lệnh an toàn hơn.

    Logic VCBS: chờ retest sau breakout — kháng cự cũ trở thành hỗ trợ mới.

    Khác BREAKOUT: BREAKOUT vào ngay khi vượt (aggressive, nhiều false break).
    Setup này chờ xác nhận retest → ít false signal hơn, entry quality tốt hơn.
    Khác RETEST hiện tại: RETEST không verify có breakout event thực sự trước đó.
    """
    reasons: list[str] = []
    if df.empty or len(df) < 35:
        return False, []

    close    = df["close"]
    volume   = df["volume"]
    vol_ma20 = volume.rolling(20).mean()

    # Tìm breakout event trong 3-15 bars trước
    breakout_level: float | None = None
    bars_ago: int = 0

    n = len(df)
    for k in range(3, min(16, n - 22)):
        idx      = n - k  # absolute index của bar breakout
        bk_close = float(close.iloc[-k])
        bk_vol   = float(volume.iloc[-k])
        bk_vol_ma = float(vol_ma20.iloc[-k]) if not pd.isna(vol_ma20.iloc[-k]) else 0

        # Đỉnh 20 bars trước bar breakout
        prior_high = float(close.iloc[max(0, idx - 20): idx].max())

        if prior_high > 0 and bk_close > prior_high and bk_vol_ma > 0 and bk_vol > bk_vol_ma * 1.5:
            breakout_level = prior_high
            bars_ago = k
            break

    if breakout_level is None:
        return False, []

    # Giá đang retest breakout level
    cur  = float(close.iloc[-1])
    dist = (cur - breakout_level) / breakout_level * 100
    is_retest = -2.0 <= dist <= 5.0

    # Volume cạn trong retest (không có lực bán lớn)
    vol_now    = float(volume.iloc[-1])
    vol_ma_now = float(vol_ma20.iloc[-1]) if not pd.isna(vol_ma20.iloc[-1]) else 0
    vol_dry    = vol_ma_now > 0 and vol_now < vol_ma_now * 0.8

    # Nến đóng xanh (xác nhận giữ support)
    is_bullish = float(df["close"].iloc[-1]) > float(df["open"].iloc[-1])

    rsi    = ind.get("rsi")
    rsi_ok = rsi is None or rsi < 70

    reasons.append(f"Breakout {bars_ago} bars trước tại level {breakout_level:.0f}")
    if is_retest:
        reasons.append(f"Retest thành công (cách {dist:+.1f}%)")
    if vol_dry:
        reasons.append(f"Volume cạn {vol_now / vol_ma_now:.2f}× TB20")
    if is_bullish:
        reasons.append("Nến đóng xanh — support giữ vững")

    passed = is_retest and (vol_dry or is_bullish) and rsi_ok
    return passed, reasons if passed else []


def detect_nr7(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """NR7: range hôm nay nhỏ nhất trong 7 phiên — volatility compression, SL hẹp."""
    reasons: list[str] = []
    if df.empty or len(df) < 10:
        return False, []

    ranges    = (df["high"] - df["low"]).tail(7)
    cur_range = float(ranges.iloc[-1])
    min_range = float(ranges.min())

    is_nr7   = cur_range <= min_range * 1.02
    ma_trend = ind.get("ma_trend")
    trend_ok = ma_trend in ("UPTREND", "SIDEWAY", None)

    rsi    = ind.get("rsi")
    rsi_ok = rsi is None or rsi < 65

    atr   = ind.get("atr")
    price = ind.get("current_price")

    if is_nr7:
        reasons.append(f"NR7: range {cur_range:.0f} nhỏ nhất 7 phiên — sắp bung")
    if atr and price:
        reasons.append(f"ATR {atr / price * 100:.1f}% — SL hẹp, risk thấp")

    passed = is_nr7 and trend_ok and rsi_ok
    return passed, reasons if passed else []


# ──────────────────────────────────────────────
# Priority score
# ──────────────────────────────────────────────

_STRATEGY_BASE_SCORE = {
    # ── Original 14 setups ──
    "BREAKOUT":               80,
    "RETEST":                 75,
    "BB_SQUEEZE":             75,
    "FLAG_PENNANT":           78,
    "SPRING":                 70,
    "GOLDEN_CROSS":           70,
    "DOUBLE_BOTTOM":          73,
    "MOMENTUM_SURGE":         72,
    "MACD_CROSSOVER":         68,
    "MA_PULLBACK":            65,
    "INSIDE_BAR":             65,
    "NR7":                    63,
    "HAMMER":                 62,
    "RSI_BOUNCE":             55,
    # ── Price Action setups ──
    "TREND_PULLBACK":         76,   # structure + level + volume — chất lượng cao
    "BREAKOUT_RETEST_ENTRY":  75,   # safer breakout play — chờ xác nhận
    "BULLISH_ENGULFING":      74,   # đảo chiều mạnh một nến
    "PIN_BAR":                71,   # pin bar tại key level
}


def compute_priority_score(
    setup_type: str,
    ind: dict,
    ref_trend: str = "UPTREND",
    money_flow: dict | None = None,
) -> float:
    score = _STRATEGY_BASE_SCORE.get(setup_type, 50)

    # Điều chỉnh base score theo thị trường thực (reference_trend)
    # UPTREND: momentum setups được ưu tiên, RSI_BOUNCE giảm (nhiều false signal)
    # SIDEWAY: range setups tốt hơn breakout
    if ref_trend == "UPTREND":
        if setup_type in ("BREAKOUT", "FLAG_PENNANT", "MOMENTUM_SURGE"):
            score += 5
        elif setup_type in ("TREND_PULLBACK", "BREAKOUT_RETEST_ENTRY"):
            score += 5   # PA setups phát huy tốt nhất trong uptrend rõ ràng
        elif setup_type in ("RSI_BOUNCE", "HAMMER"):
            score -= 5   # oversold signals ít đáng tin trong uptrend mạnh
    elif ref_trend == "SIDEWAY":
        if setup_type == "BREAKOUT":
            score -= 8   # false break cao trong sideway
        elif setup_type in ("RETEST", "SPRING", "BB_SQUEEZE", "NR7", "INSIDE_BAR"):
            score += 5   # range-bound & compression setups hiệu quả hơn
        elif setup_type in ("DOUBLE_BOTTOM", "HAMMER", "BULLISH_ENGULFING", "PIN_BAR"):
            score += 3   # reversal signals tốt hơn tại vùng sideway support

    confluence = ind.get("confluence_score", 0)
    if confluence >= 6:
        score += 5
    elif confluence >= 4:
        score += 2
    if ind.get("volume_surge"):
        score += 3
    if ind.get("macd_signal_label") == "BULLISH":
        score += 3
    rsi = ind.get("rsi")
    if rsi and rsi > 70:
        score -= 5
    if money_flow:
        mf_regime = money_flow.get("regime")
        mf_score = float(money_flow.get("score") or 0)
        if mf_regime == "BREAKOUT_FLOW":
            score += 10
        elif mf_regime == "EARLY_MONEY_IN":
            score += 8
        elif mf_regime == "MONEY_IN":
            score += 6
        elif mf_regime == "ACCUMULATION":
            score += 3
        elif mf_regime == "CHASE_MONEY_IN":
            score -= 8
        elif mf_regime == "EXHAUSTION_INFLOW":
            score -= 12
        elif mf_regime == "MONEY_OUT":
            score -= 8
        elif mf_regime == "DISTRIBUTION":
            score -= 20
        score += max(-6, min(8, mf_score))
    return min(round(score, 1), 100.0)


def _build_screener_money_flow(
    symbol: str,
    df: pd.DataFrame,
    market_ctx: MarketContext,
) -> dict:
    """Classify latest bar for screener ranking without extra API calls."""
    ctx = {
        "symbol": symbol,
        "market_context": {
            "trend": market_ctx.reference_trend,
            "vni_change_pct": market_ctx.vni_change_pct,
            "current_price": market_ctx.current_price,
        },
        "sector_context": {
            "relative_strength_20d": 0.0,
            "is_outperforming": False,
        },
    }
    feat = add_money_flow_features(df)
    return classify_money_flow(feat, ctx)


# ──────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────

def run_screener(
    symbols: list[str] | None = None,
    max_candidates: int = 20,
) -> tuple[MarketContext, list[TradeCandidate]]:
    if symbols is None:
        symbols = get_liquid_symbols(min_avg_vol=500_000)

    print(f"[trade_screener] Scan {len(symbols)} mã (vol≥500k)...")

    market_ctx = get_market_context()

    # Log: VNI raw + reference (có thể khác nhau khi VinGroup méo chỉ số)
    ref_label = (
        f"ref={market_ctx.reference_index}:{market_ctx.reference_trend}"
        if market_ctx.is_index_distorted
        else market_ctx.reference_trend
    )
    print(
        f"[trade_screener] VNI: {market_ctx.trend} ({market_ctx.vni_change_pct:+.2f}%) | "
        f"MidCap: {market_ctx.vnmidcap_trend} ({market_ctx.vnmidcap_change_pct:+.2f}%) | "
        f"→ {ref_label}"
    )
    if market_ctx.is_index_distorted:
        print(f"[trade_screener] ⚠ {market_ctx.distortion_note}")

    if not market_ctx.should_trade:
        # should_trade đã dựa trên reference_trend — DOWNTREND thực sự
        print(f"[trade_screener] {market_ctx.reference_index} DOWNTREND → dừng scan")
        return market_ctx, []

    print(f"[trade_screener] Fetch OHLCV batch ({len(symbols)} mã)...")
    ohlcv_map = get_ohlcv_batch(symbols, n_days=200)

    candidates: list[TradeCandidate] = []
    min_liquidity = 300_000
    skipped = 0

    # Dùng reference_trend (không phải VNI raw trend) để chọn chiến lược
    ref_trend = market_ctx.reference_trend

    for symbol, df in ohlcv_map.items():
        if df is None or df.empty or len(df) < 30:
            continue
        if float(df["volume"].tail(20).mean()) < min_liquidity:
            skipped += 1
            continue
        ind = compute_indicators(df)
        if not ind:
            continue
        try:
            money_flow = _build_screener_money_flow(symbol, df, market_ctx)
        except Exception as e:
            print(f"[trade_screener] money_flow {symbol} lỗi: {e}")
            money_flow = {}
        if money_flow.get("regime") in {"DISTRIBUTION", "EXHAUSTION_INFLOW"}:
            skipped += 1
            continue
        ind["money_flow_analysis"] = money_flow
        if len(df) >= 2:
            prev_close = float(df["close"].iloc[-2])
            cur_close = float(df["close"].iloc[-1])
            ind["stock_day_change_pct"] = round((cur_close - prev_close) / prev_close * 100, 2) if prev_close else 0.0

        detected: list[tuple[str, list[str]]] = []
        strategies = [
            # ── Original 14 ──
            ("BREAKOUT",               detect_breakout),
            ("FLAG_PENNANT",           detect_flag_pennant),
            ("BB_SQUEEZE",             detect_bb_squeeze),
            ("RETEST",                 detect_retest),
            ("SPRING",                 detect_spring),
            ("GOLDEN_CROSS",           detect_golden_cross),
            ("DOUBLE_BOTTOM",          detect_double_bottom),
            ("MOMENTUM_SURGE",         detect_momentum_surge),
            ("MACD_CROSSOVER",         detect_macd_crossover),
            ("MA_PULLBACK",            detect_ma_pullback),
            ("INSIDE_BAR",             detect_inside_bar),
            ("NR7",                    detect_nr7),
            ("HAMMER",                 detect_hammer),
            ("RSI_BOUNCE",             detect_rsi_bounce),
            # ── Price Action ──
            ("TREND_PULLBACK",         detect_trend_pullback),
            ("BREAKOUT_RETEST_ENTRY",  detect_breakout_retest_entry),
            ("BULLISH_ENGULFING",      detect_bullish_engulfing),
            ("PIN_BAR",                detect_pin_bar),
        ]

        # DOWNTREND thực: chỉ giữ reversal setups (bắt đáy ngược chiều)
        # SIDEWAY / UPTREND: giữ toàn bộ 18 chiến lược
        if ref_trend == "DOWNTREND":
            strategies = [
                ("DOUBLE_BOTTOM",     detect_double_bottom),
                ("RSI_BOUNCE",        detect_rsi_bounce),
                ("HAMMER",            detect_hammer),
                ("BULLISH_ENGULFING", detect_bullish_engulfing),
                ("PIN_BAR",           detect_pin_bar),
            ]

        for name, fn in strategies:
            passed, reasons = fn(df, ind)
            if passed:
                detected.append((name, reasons))

        for setup_type, reasons in detected:
            score = compute_priority_score(setup_type, ind, ref_trend, money_flow)
            mf_regime = money_flow.get("regime")
            if mf_regime in ("BREAKOUT_FLOW", "EARLY_MONEY_IN", "MONEY_IN", "ACCUMULATION"):
                reasons = [*reasons, f"Money flow: {mf_regime} score={money_flow.get('score')}"]
            candidates.append(TradeCandidate(
                symbol         = symbol,
                setup_type     = setup_type,
                priority_score = score,
                market_context = market_ctx,
                indicators     = ind,
                reasons        = reasons,
            ))
            break  # chỉ lấy strategy ưu tiên cao nhất

    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    candidates = candidates[:max_candidates]

    print(f"[trade_screener] Lọc thanh khoản: bỏ {skipped} mã")
    print(f"[trade_screener] {len(candidates)} candidates")
    for c in candidates:
        print(f"  {c.symbol:6s} | {c.setup_type:12s} | score={c.priority_score:.0f}")

    return market_ctx, candidates
