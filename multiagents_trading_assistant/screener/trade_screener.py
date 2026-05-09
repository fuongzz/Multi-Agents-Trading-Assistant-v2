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
from multiagents_trading_assistant.indicators import compute_indicators, get_ichimoku_config
from multiagents_trading_assistant.agents.trade.money_flow_agent import (
    add_money_flow_features,
    classify_money_flow,
)
from multiagents_trading_assistant.setup_scoring import score_setup


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

# VinGroup chỉ có tác động méo từ tháng 3/2025 (sau khi VIC/VHM tăng vốn hóa lớn)
_VINGROUP_DISTORTION_START = "2025-03-01"


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
    """Candidate từ screener — truyền vào trade_graph.

    setup_type rỗng ở Phase 1 (broad candidate pool).
    Bob (Phase 2) sẽ gán setup_type sau khi chạy simulated trading.
    """
    symbol: str
    priority_score: float
    market_context: MarketContext
    indicators: dict
    setup_type: str = ""
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

def get_market_context(
    vnindex_df: pd.DataFrame | None = None,
    as_of_date: str | None = None,
) -> MarketContext:
    """Phân tích trạng thái thị trường, có bổ sung phát hiện méo VinGroup.

    Flow:
      1. Lấy VNI trend + MA (như cũ)
      2. Lấy VNMidCap — 1 API call bổ sung, cache theo ngày
      3. Lấy OHLCV VinGroup (VIC/VHM/VRE/VPL) để tính contribution — cache
      4. Phát hiện phân kỳ VNI vs VNMidCap → đặt reference_trend
    """
    if vnindex_df is None or vnindex_df.empty:
        if as_of_date:
            # Backtest: cần fetch full history để có data tại as_of_date
            # Caller (llm_backtest) nên pre-fetch và pass vào để tránh gọi lại nhiều lần
            from multiagents_trading_assistant.fetcher import get_ohlcv_history
            from datetime import date as _d, timedelta as _td
            hist_start = (_d.fromisoformat(as_of_date) - _td(days=400)).strftime("%Y-%m-%d")
            vnindex_df = get_ohlcv_history("VNINDEX", hist_start, as_of_date)
        else:
            vnindex_df = get_vnindex(200)
    if vnindex_df is None or vnindex_df.empty:
        raise RuntimeError("Không lấy được dữ liệu VN-Index — abort trade_screener")

    # Trim theo as_of_date để tránh leak future data trong backtest
    if as_of_date:
        date_col = "date" if "date" in vnindex_df.columns else None
        if date_col:
            vnindex_df = vnindex_df[vnindex_df[date_col].astype(str) <= as_of_date]
        if vnindex_df.empty:
            raise RuntimeError(f"VN-Index không có data tại hoặc trước {as_of_date}")

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
        if as_of_date:
            # Backtest: fetch historical range, trim to as_of_date
            from multiagents_trading_assistant.fetcher import get_ohlcv_history as _get_hist
            from datetime import date as _d2, timedelta as _td2
            _mc_start = (_d2.fromisoformat(as_of_date) - _td2(days=400)).strftime("%Y-%m-%d")
            vnmidcap_df = _get_hist("VNMIDCAP", _mc_start, as_of_date)
        else:
            vnmidcap_df = get_vnmidcap(200)
        if not vnmidcap_df.empty:
            if as_of_date and "date" in vnmidcap_df.columns:
                vnmidcap_df = vnmidcap_df[vnmidcap_df["date"].astype(str) <= as_of_date]
            if not vnmidcap_df.empty:
                vnmidcap_trend, _, _, _, _, vnmidcap_change = _extract_trend(vnmidcap_df)
                print(f"[market_ctx] VNMidCap: {vnmidcap_trend} ({vnmidcap_change:+.2f}%)")
    except Exception as e:
        print(f"[market_ctx] VNMidCap lỗi — bỏ qua: {e}")

    # ── VinGroup: ước tính đóng góp vào VNI ──
    # Chỉ áp dụng từ 2025-03-01 — trước đó VinGroup không đủ tác động để méo VNI
    vingroup_contribution = 0.0
    _check_vingroup = as_of_date is None or as_of_date >= _VINGROUP_DISTORTION_START
    if _check_vingroup:
        try:
            ving_changes: dict[str, float] = {}
            from multiagents_trading_assistant.fetcher import get_ohlcv_history as _get_hist2
            from datetime import date as _d3, timedelta as _td3
            for sym in _VINGROUP_WEIGHTS:
                if as_of_date:
                    _vs = (_d3.fromisoformat(as_of_date) - _td3(days=10)).strftime("%Y-%m-%d")
                    df_sym = _get_hist2(sym, _vs, as_of_date)
                    if not df_sym.empty and "date" in df_sym.columns:
                        df_sym = df_sym[df_sym["date"].astype(str) <= as_of_date]
                else:
                    df_sym = get_ohlcv(sym, 5)
                if not df_sym.empty and len(df_sym) >= 2:
                    c_cur  = float(df_sym["close"].iloc[-1])
                    c_prev = float(df_sym["close"].iloc[-2])
                    ving_changes[sym] = round((c_cur - c_prev) / c_prev * 100, 2) if c_prev else 0.0
            vingroup_contribution = _estimate_vingroup_contribution(ving_changes)
            ving_str = " | ".join(f"{s} {v:+.1f}%" for s, v in ving_changes.items())
            print(f"[market_ctx] VinGroup: [{ving_str}] → đóng góp VNI ≈ {vingroup_contribution:+.3f}%")
        except Exception as e:
            print(f"[market_ctx] VinGroup fetch lỗi — bỏ qua: {e}")

    # ── Phát hiện méo ── (chỉ từ 2025-03-01 trở đi)
    _allow_distortion = (as_of_date is None) or (as_of_date >= _VINGROUP_DISTORTION_START)
    if _allow_distortion:
        is_distorted, reference_trend, reference_index, distortion_note = _detect_distortion(
            vni_trend, vni_change, vnmidcap_trend, vnmidcap_change, vingroup_contribution,
        )
    else:
        is_distorted, reference_trend, reference_index, distortion_note = False, vni_trend, "VNINDEX", ""
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


def _fallback_macd_crossover_values(df: pd.DataFrame) -> tuple[bool | None, float | None, float | None]:
    import pandas_ta as _ta

    macd_df = _ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd_df is None or macd_df.empty:
        return None, None, None

    cols = macd_df.columns.tolist()
    ml_col = next((c for c in cols if c.startswith("MACD_")), None)
    ms_col = next((c for c in cols if c.startswith("MACDs_")), None)
    mh_col = next((c for c in cols if c.startswith("MACDh_")), None)
    if not (ml_col and ms_col and mh_col):
        return None, None, None

    ml = macd_df[ml_col].dropna()
    ms = macd_df[ms_col].dropna()
    mh = macd_df[mh_col].dropna()
    if len(ml) < 4 or len(ms) < 4 or mh.empty:
        return None, None, None

    crossed = any(
        ml.iloc[i - 1] < ms.iloc[i - 1] and ml.iloc[i] >= ms.iloc[i]
        for i in range(-4, 0)
    )
    return crossed, float(ml.iloc[-1]), float(mh.iloc[-1])


def detect_macd_crossover(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """MACD line cắt lên signal line trong 3 phiên gần nhất."""
    reasons: list[str] = []
    if df.empty or len(df) < 35:
        return False, []

    crossed = ind.get("macd_bullish_cross_recent")
    macd_val = ind.get("macd")
    hist_val = ind.get("macd_hist")
    if crossed is None or macd_val is None or hist_val is None:
        crossed, macd_val, hist_val = _fallback_macd_crossover_values(df)
    if crossed is None or macd_val is None or hist_val is None:
        return False, []

    hist_val      = float(hist_val)
    hist_positive = hist_val > 0
    above_zero    = float(macd_val) > 0
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


def _fallback_bb_width_values(df: pd.DataFrame) -> tuple[float | None, float | None]:
    import pandas_ta as _ta

    bb = _ta.bbands(df["close"], length=20, std=2)
    if bb is None or bb.empty:
        return None, None

    cols = bb.columns.tolist()
    bbl_col = next((c for c in cols if c.startswith("BBL_")), None)
    bbu_col = next((c for c in cols if c.startswith("BBU_")), None)
    bbm_col = next((c for c in cols if c.startswith("BBM_")), None)
    if not (bbl_col and bbu_col and bbm_col):
        return None, None

    width = ((bb[bbu_col] - bb[bbl_col]) / bb[bbm_col]).dropna()
    if len(width) < 20:
        return None, None
    return float(width.iloc[-1]), float(width.iloc[-20:].min())


def detect_bb_squeeze(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Bollinger Band Squeeze: BB thu hẹp tối thiểu 20 phiên, giá áp BB upper."""
    reasons: list[str] = []
    if df.empty or len(df) < 30:
        return False, []

    cur_width = ind.get("bb_width")
    min_20 = ind.get("bb_width_min20")
    if cur_width is None or min_20 is None:
        cur_width, min_20 = _fallback_bb_width_values(df)
    if cur_width is None or min_20 is None:
        return False, []

    cur_width  = float(cur_width)
    min_20     = float(min_20)
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


def detect_kumo_breakout(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Close breaks above the current Kumo with strong volume confirmation."""
    reasons: list[str] = []
    if df.empty or len(df) < 80:
        return False, []

    close = df["close"]
    volume = df["volume"]
    cur = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    cloud_top = ind.get("ichimoku_cloud_top")
    cloud_bottom = ind.get("ichimoku_cloud_bottom")
    if cloud_top is None or cloud_bottom is None:
        return False, []

    v_now = float(volume.iloc[-1])
    v_ma20 = float(volume.rolling(20).mean().iloc[-1])
    cloud_thickness = (cloud_top - cloud_bottom) / cur * 100 if cur > 0 else 0.0

    breakout = cur > cloud_top and prev <= cloud_top
    vol_ok = v_ma20 > 0 and v_now >= v_ma20 * 1.5
    cloud_ok = cloud_thickness >= 0.8
    confirms, confirm_reasons = _ichimoku_confirmations(ind, v_now, v_ma20, min_volume_ratio=1.2)

    if breakout:
        reasons.append(f"Close {cur:.0f} breakout Kumo top {cloud_top:.0f}")
    if vol_ok:
        reasons.append(f"Volume {v_now / v_ma20:.1f}x MA20")
    if cloud_ok:
        reasons.append(f"Kumo thickness {cloud_thickness:.1f}%")
    reasons.extend(confirm_reasons)

    passed = breakout and vol_ok and cloud_ok and confirms >= 2
    return passed, reasons if passed else []


def detect_tk_cross(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Tenkan crosses above Kijun while price is above Kumo."""
    reasons: list[str] = []
    if df.empty or len(df) < 80:
        return False, []

    ichi = _ichimoku_series(df)
    tenkan = ichi["tenkan"].dropna()
    kijun = ichi["kijun"].dropna()
    if len(tenkan) < 2 or len(kijun) < 2:
        return False, []

    cur_close = float(df["close"].iloc[-1])
    cloud_top = ind.get("ichimoku_cloud_top")
    if cloud_top is None:
        return False, []

    crossed = tenkan.iloc[-2] <= kijun.iloc[-2] and tenkan.iloc[-1] > kijun.iloc[-1]
    above_kumo = cur_close > cloud_top
    kijun_slope_ok = (ind.get("ichimoku_kijun_slope") or 0) >= 0
    confirms, confirm_reasons = _ichimoku_confirmations(
        ind,
        float(df["volume"].iloc[-1]),
        float(df["volume"].rolling(20).mean().iloc[-1]),
        min_volume_ratio=1.2,
    )

    if crossed:
        reasons.append(f"Tenkan {tenkan.iloc[-1]:.0f} cross up Kijun {kijun.iloc[-1]:.0f}")
    if above_kumo:
        reasons.append(f"Close above Kumo top {cloud_top:.0f}")
    if kijun_slope_ok:
        reasons.append("Kijun slope non-negative")
    reasons.extend(confirm_reasons)

    passed = crossed and above_kumo and kijun_slope_ok and confirms >= 2
    return passed, reasons if passed else []


def detect_kijun_bounce(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Uptrend retest of Kijun with dry volume and bullish reversal candle."""
    reasons: list[str] = []
    if df.empty or len(df) < 80:
        return False, []

    kijun = ind.get("ichimoku_kijun")
    cloud_top = ind.get("ichimoku_cloud_top")
    if kijun is None or cloud_top is None:
        return False, []

    cur_open = float(df["open"].iloc[-1])
    cur_high = float(df["high"].iloc[-1])
    cur_low = float(df["low"].iloc[-1])
    cur_close = float(df["close"].iloc[-1])
    prev_close = float(df["close"].iloc[-2])
    volume = df["volume"]
    v_now = float(volume.iloc[-1])
    v_ma20 = float(volume.rolling(20).mean().iloc[-1])

    uptrend = ind.get("ichimoku_regime") == "BULLISH" and cur_close > cloud_top
    retest = cur_low <= kijun * 1.01 and cur_close > kijun
    dry_volume = v_ma20 > 0 and v_now <= v_ma20 * 1.0
    bullish_reversal = cur_close > cur_open and cur_close > prev_close and cur_close >= (cur_low + (cur_high - cur_low) * 0.55)
    confirms, confirm_reasons = _ichimoku_confirmations(ind, v_now, v_ma20, min_volume_ratio=0.7)

    if uptrend:
        reasons.append("Ichimoku bullish regime")
    if retest:
        reasons.append(f"Retest Kijun {kijun:.0f} and close back above")
    if dry_volume:
        reasons.append(f"Pullback volume dry {v_now / v_ma20:.2f}x MA20")
    if bullish_reversal:
        reasons.append("Bullish reversal candle at Kijun")
    reasons.extend(confirm_reasons)

    passed = uptrend and retest and dry_volume and bullish_reversal and confirms >= 2
    return passed, reasons if passed else []


def detect_kumo_twist_entry(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Future cloud turns bullish with a price trigger above Kijun/Tenkan."""
    reasons: list[str] = []
    if df.empty or len(df) < 80:
        return False, []

    ichi = _ichimoku_series(df)
    future_a = ichi["senkou_a_raw"].dropna()
    future_b = ichi["senkou_b_raw"].dropna()
    if len(future_a) < 2 or len(future_b) < 2:
        return False, []

    cur_close = float(df["close"].iloc[-1])
    tenkan = ind.get("ichimoku_tenkan")
    kijun = ind.get("ichimoku_kijun")
    cloud_top = ind.get("ichimoku_cloud_top")
    if tenkan is None or kijun is None or cloud_top is None:
        return False, []

    bullish_twist = future_a.iloc[-2] <= future_b.iloc[-2] and future_a.iloc[-1] > future_b.iloc[-1]
    price_trigger = cur_close > max(tenkan, kijun)
    not_bearish_zone = cur_close >= cloud_top or ind.get("ichimoku_regime") == "BULLISH"
    confirms, confirm_reasons = _ichimoku_confirmations(
        ind,
        float(df["volume"].iloc[-1]),
        float(df["volume"].rolling(20).mean().iloc[-1]),
        min_volume_ratio=1.0,
    )

    if bullish_twist:
        reasons.append("Future Kumo bullish twist")
    if price_trigger:
        reasons.append(f"Close reclaimed Tenkan/Kijun ({tenkan:.0f}/{kijun:.0f})")
    if not_bearish_zone:
        reasons.append("Price not below current Kumo")
    reasons.extend(confirm_reasons)

    passed = bullish_twist and price_trigger and not_bearish_zone and confirms >= 2
    return passed, reasons if passed else []


def _ichimoku_series(
    df: pd.DataFrame,
    tenkan_period: int | None = None,
    kijun_period: int | None = None,
    senkou_b_period: int | None = None,
) -> dict[str, pd.Series]:
    if None in {tenkan_period, kijun_period, senkou_b_period}:
        tenkan_period, kijun_period, senkou_b_period, _, _ = get_ichimoku_config()
    high = df["high"]
    low = df["low"]
    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2
    senkou_a_raw = (tenkan + kijun) / 2
    senkou_b_raw = (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2
    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a_raw": senkou_a_raw,
        "senkou_b_raw": senkou_b_raw,
    }


def _ichimoku_confirmations(
    ind: dict,
    volume_now: float,
    volume_ma20: float,
    *,
    min_volume_ratio: float,
) -> tuple[int, list[str]]:
    confirmations = 0
    reasons: list[str] = []

    if ind.get("ichimoku_chikou_confirm"):
        confirmations += 1
        reasons.append("Chikou confirms momentum")
    if ind.get("ichimoku_future_cloud_green"):
        confirmations += 1
        reasons.append("Future cloud green")
    if volume_ma20 > 0 and volume_now >= volume_ma20 * min_volume_ratio:
        confirmations += 1
        reasons.append(f"Volume confirm {volume_now / volume_ma20:.1f}x MA20")
    return confirmations, reasons


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


def detect_adx_trend(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Directional trend setup: only follow strength when VN cash flow confirms."""
    if df.empty or len(df) < 60:
        return False, []
    price = _f(ind.get("current_price"))
    ema20 = _f(ind.get("ema20") or ind.get("ma20"))
    ema50 = _f(ind.get("ema50") or ind.get("ma60"))
    adx = _f(ind.get("adx_14"))
    dmp = _f(ind.get("dmp_14"))
    dmn = _f(ind.get("dmn_14"))
    rsi = _f(ind.get("rsi"))
    volume_ratio = _f(ind.get("volume_ratio_20")) or 1.0
    dist_ema20 = (price - ema20) / ema20 * 100.0 if price and ema20 else None

    passed = bool(
        price and ema20 and ema50 and adx and dmp and dmn
        and price > ema20 > ema50
        and adx >= 20
        and dmp > dmn * 1.08
        and 48 <= (rsi or 55) <= 72
        and 0.85 <= volume_ratio <= 2.5
        and (dist_ema20 is None or dist_ema20 <= 8.0)
    )
    reasons = [
        f"ADX trend: ADX {adx:.1f}, +DI {dmp:.1f} > -DI {dmn:.1f}",
        f"Price above EMA20/EMA50, volume {_fmt(volume_ratio)}x MA20",
    ] if passed else []
    return passed, reasons


def detect_supertrend_pullback(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """VN-friendly trend pullback: buy near dynamic support, not far above it."""
    if df.empty or len(df) < 60:
        return False, []
    price = _f(ind.get("current_price"))
    supertrend = _f(ind.get("supertrend_10_3"))
    supertrend_dir = _f(ind.get("supertrend_dir"))
    ema20 = _f(ind.get("ema20") or ind.get("ma20"))
    vwma20 = _f(ind.get("vwma20"))
    rsi = _f(ind.get("rsi"))
    volume_ratio = _f(ind.get("volume_ratio_20")) or 1.0
    near_dynamic_support = bool(
        price and (
            (ema20 and abs(price - ema20) / ema20 <= 0.035)
            or (vwma20 and abs(price - vwma20) / vwma20 <= 0.035)
        )
    )
    passed = bool(
        price and supertrend and supertrend_dir
        and supertrend_dir > 0
        and price > supertrend
        and near_dynamic_support
        and 40 <= (rsi or 50) <= 66
        and 0.65 <= volume_ratio <= 1.8
    )
    reasons = [
        "Supertrend bullish, pullback near EMA/VWMA support",
        f"RSI {_fmt(rsi)}, volume {_fmt(volume_ratio)}x MA20",
    ] if passed else []
    return passed, reasons


def detect_obv_accumulation(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Accumulation setup: OBV leads price while price holds value-weighted support."""
    if df.empty or len(df) < 60:
        return False, []
    price = _f(ind.get("current_price"))
    ema50 = _f(ind.get("ema50") or ind.get("ma60"))
    vwma20 = _f(ind.get("vwma20"))
    obv = _f(ind.get("obv"))
    obv_ema20 = _f(ind.get("obv_ema20"))
    rsi = _f(ind.get("rsi"))
    volume_ratio = _f(ind.get("volume_ratio_20")) or 1.0
    passed = bool(
        price and ema50 and vwma20 and obv is not None and obv_ema20 is not None
        and price > ema50
        and price >= vwma20
        and obv > obv_ema20
        and 45 <= (rsi or 50) <= 68
        and 0.75 <= volume_ratio <= 2.2
    )
    reasons = [
        "OBV above OBV EMA20, price holds VWMA20",
        f"Accumulation volume {_fmt(volume_ratio)}x MA20",
    ] if passed else []
    return passed, reasons


def detect_keltner_squeeze_breakout(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Volatility expansion setup using Keltner/Bollinger compression."""
    if df.empty or len(df) < 80:
        return False, []
    price = _f(ind.get("current_price"))
    kc_upper = _f(ind.get("kc_upper"))
    kc_width = _f(ind.get("kc_width"))
    bb_width = _f(ind.get("bb_width"))
    bb_width_min20 = _f(ind.get("bb_width_min20"))
    rsi = _f(ind.get("rsi"))
    volume_ratio = _f(ind.get("volume_ratio_20")) or 1.0
    compressed = bool(
        kc_width is not None and kc_width <= 0.07
        and bb_width is not None
        and (bb_width_min20 is None or bb_width <= bb_width_min20 * 1.35)
    )
    passed = bool(
        price and kc_upper
        and compressed
        and price >= kc_upper * 0.995
        and 48 <= (rsi or 55) <= 72
        and volume_ratio >= 1.15
    )
    reasons = [
        f"Keltner squeeze breakout, KC width {kc_width:.2%}",
        f"Volume expansion {_fmt(volume_ratio)}x MA20",
    ] if passed else []
    return passed, reasons


def detect_oversold_mean_reversion(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Long-only VN mean reversion with anti-downtrend gates."""
    if df.empty or len(df) < 80:
        return False, []
    price = _f(ind.get("current_price"))
    ema200 = _f(ind.get("ema200") or ind.get("ma200"))
    bb_percent = _f(ind.get("bb_percent"))
    willr = _f(ind.get("willr_14"))
    stoch_k = _f(ind.get("stoch_k"))
    adx = _f(ind.get("adx_14")) or 0.0
    volume_ratio = _f(ind.get("volume_ratio_20")) or 1.0
    recent_low20 = float(df["low"].tail(20).min())
    not_breakdown = bool(price and price > recent_low20 * 1.01)
    not_structural_downtrend = bool(price and (ema200 is None or price >= ema200 * 0.92))
    oversold = bool(
        (bb_percent is not None and bb_percent <= 0.20)
        or (willr is not None and willr <= -80)
        or (stoch_k is not None and stoch_k <= 25)
    )
    passed = bool(
        oversold
        and not_breakdown
        and not_structural_downtrend
        and adx < 35
        and volume_ratio >= 0.75
    )
    reasons = [
        "Oversold mean reversion, no fresh 20-day breakdown",
        f"WILLR {_fmt(willr)}, StochK {_fmt(stoch_k)}, ADX {_fmt(adx)}",
    ] if passed else []
    return passed, reasons


def detect_aroon_trend_shift(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Early trend shift after base building."""
    if df.empty or len(df) < 60:
        return False, []
    price = _f(ind.get("current_price"))
    ema50 = _f(ind.get("ema50") or ind.get("ma60"))
    aroon_up = _f(ind.get("aroon_up_14"))
    aroon_down = _f(ind.get("aroon_down_14"))
    aroon_osc = _f(ind.get("aroon_osc_14"))
    rsi = _f(ind.get("rsi"))
    volume_ratio = _f(ind.get("volume_ratio_20")) or 1.0
    passed = bool(
        price and ema50 and aroon_up is not None and aroon_down is not None
        and price > ema50
        and aroon_up >= 70
        and aroon_down <= 35
        and (aroon_osc is None or aroon_osc > 35)
        and 48 <= (rsi or 55) <= 70
        and volume_ratio >= 0.9
    )
    reasons = [
        f"Aroon trend shift: up {_fmt(aroon_up, 0)}, down {_fmt(aroon_down, 0)}",
        "Price reclaimed EMA50 with acceptable volume",
    ] if passed else []
    return passed, reasons


def detect_linreg_momentum(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """Linear-regression slope confirms short momentum without late chase."""
    if df.empty or len(df) < 60:
        return False, []
    price = _f(ind.get("current_price"))
    ema20 = _f(ind.get("ema20") or ind.get("ma20"))
    vwma20 = _f(ind.get("vwma20"))
    slope = _f(ind.get("linreg_slope_14"))
    roc = _f(ind.get("roc_9"))
    mom = _f(ind.get("mom_10"))
    rsi = _f(ind.get("rsi"))
    volume_ratio = _f(ind.get("volume_ratio_20")) or 1.0
    dist_ema20 = (price - ema20) / ema20 * 100.0 if price and ema20 else None
    passed = bool(
        price and ema20 and vwma20 and slope is not None and roc is not None and mom is not None
        and price > ema20 and price > vwma20
        and slope > 0 and roc > 0 and mom > 0
        and 50 <= (rsi or 55) <= 70
        and volume_ratio >= 0.9
        and (dist_ema20 is None or dist_ema20 <= 8.0)
    )
    reasons = [
        f"Linear regression slope positive, ROC {_fmt(roc)}%",
        "Momentum confirmed while price is not overextended",
    ] if passed else []
    return passed, reasons


def _f(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _fmt(value, digits: int = 1) -> str:
    num = _f(value)
    return "n/a" if num is None else f"{num:.{digits}f}"


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
    # Ichimoku setups
    "KUMO_BREAKOUT":          82,
    "TK_CROSS":               76,
    "KIJUN_BOUNCE":           78,
    "KUMO_TWIST_ENTRY":       72,
    "ADX_TREND":              79,
    "SUPERTREND_PULLBACK":    77,
    "OBV_ACCUMULATION":       74,
    "KELTNER_SQUEEZE":        76,
    "OVERSOLD_MEAN_REVERSION":66,
    "AROON_TREND_SHIFT":      73,
    "LINREG_MOMENTUM":        72,
}

SETUP_REGIME_REQUIREMENT = {
    "BREAKOUT": ["UPTREND"],
    "INSIDE_BAR": ["UPTREND"],
    "MOMENTUM_SURGE": ["UPTREND"],
    "FLAG_PENNANT": ["UPTREND"],
    "GOLDEN_CROSS": ["UPTREND", "SIDEWAY"],
    "DOUBLE_BOTTOM": ["SIDEWAY", "DOWNTREND"],
    "RSI_BOUNCE": ["SIDEWAY", "DOWNTREND"],
    "BB_SQUEEZE": ["SIDEWAY"],
    "NR7": ["UPTREND", "SIDEWAY"],
    "BULLISH_ENGULFING": ["UPTREND", "SIDEWAY", "DOWNTREND"],
    "KUMO_BREAKOUT": ["UPTREND"],
    "TK_CROSS": ["UPTREND"],
    "KIJUN_BOUNCE": ["UPTREND"],
    "KUMO_TWIST_ENTRY": ["UPTREND", "SIDEWAY"],
    "ADX_TREND": ["UPTREND"],
    "SUPERTREND_PULLBACK": ["UPTREND"],
    "OBV_ACCUMULATION": ["UPTREND", "SIDEWAY"],
    "KELTNER_SQUEEZE": ["UPTREND", "SIDEWAY"],
    "OVERSOLD_MEAN_REVERSION": ["SIDEWAY", "DOWNTREND"],
    "AROON_TREND_SHIFT": ["UPTREND", "SIDEWAY"],
    "LINREG_MOMENTUM": ["UPTREND"],
}


def _is_setup_allowed_in_regime(setup_type: str, ref_trend: str) -> bool:
    ref_trend = _normalize_ref_trend(ref_trend)
    allowed = SETUP_REGIME_REQUIREMENT.get(setup_type)
    return allowed is None or ref_trend in allowed


def _normalize_ref_trend(ref_trend: str) -> str:
    trend = (ref_trend or "").upper()
    return trend if trend in {"UPTREND", "SIDEWAY", "DOWNTREND"} else "SIDEWAY"


def compute_priority_score(
    setup_type: str,
    ind: dict,
    ref_trend: str = "UPTREND",
    money_flow: dict | None = None,
) -> float:
    ref_trend = _normalize_ref_trend(ref_trend)
    normalized = score_setup(
        setup_type,
        ind,
        money_flow=money_flow or {},
        reference_trend=ref_trend,
    )
    score = 0.70 * float(normalized.get("score") or 0.0) + 0.30 * _STRATEGY_BASE_SCORE.get(setup_type, 50)

    # Điều chỉnh base score theo thị trường thực (reference_trend)
    # UPTREND: momentum setups được ưu tiên, RSI_BOUNCE giảm (nhiều false signal)
    # SIDEWAY: range setups tốt hơn breakout
    if ref_trend == "UPTREND":
        if setup_type in ("BREAKOUT", "FLAG_PENNANT", "MOMENTUM_SURGE", "ADX_TREND", "LINREG_MOMENTUM"):
            score += 5
        elif setup_type in ("SUPERTREND_PULLBACK", "OBV_ACCUMULATION", "AROON_TREND_SHIFT"):
            score += 4
        elif setup_type in ("KUMO_BREAKOUT", "TK_CROSS", "KIJUN_BOUNCE"):
            score += 6
        elif setup_type in ("TREND_PULLBACK", "BREAKOUT_RETEST_ENTRY"):
            score += 5   # PA setups phát huy tốt nhất trong uptrend rõ ràng
        elif setup_type in ("RSI_BOUNCE", "HAMMER"):
            score -= 5   # oversold signals ít đáng tin trong uptrend mạnh
    elif ref_trend == "SIDEWAY":
        if setup_type == "BREAKOUT":
            score -= 8   # false break cao trong sideway
        elif setup_type in ("RETEST", "SPRING", "BB_SQUEEZE", "NR7", "INSIDE_BAR", "KELTNER_SQUEEZE"):
            score += 5   # range-bound & compression setups hiệu quả hơn
        elif setup_type in ("DOUBLE_BOTTOM", "HAMMER", "BULLISH_ENGULFING", "PIN_BAR", "OVERSOLD_MEAN_REVERSION"):
            score += 3   # reversal signals tốt hơn tại vùng sideway support
        elif setup_type == "KUMO_TWIST_ENTRY":
            score += 2

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
    max_candidates: int = 50,
    as_of_date: str | None = None,
    vnindex_df: pd.DataFrame | None = None,
    ohlcv_map: dict[str, pd.DataFrame] | None = None,
) -> tuple[MarketContext, list[TradeCandidate]]:
    """Broad candidate pool cho LLM agents.

    Không detect setup, không hard-gate theo regime hay money flow.
    Output: top N mã đủ data quality + liquidity, kèm raw indicators.
    LLM (và Bob ở Phase 2) sẽ quyết định setup/strategy.
    """
    if symbols is None:
        symbols = get_liquid_symbols(min_avg_vol=300_000)

    print(f"[trade_screener] Scan {len(symbols)} mã (vol≥300k)...")

    market_ctx = get_market_context(vnindex_df=vnindex_df, as_of_date=as_of_date)

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

    if ohlcv_map is not None:
        # Pre-fetched map provided (backtest mode) — trim to as_of_date if needed
        print(f"[trade_screener] Using pre-fetched OHLCV map ({len(ohlcv_map)} mã)...")
        if as_of_date:
            trimmed: dict = {}
            for sym, df in ohlcv_map.items():
                if df is None or df.empty:
                    trimmed[sym] = df
                    continue
                if "date" in df.columns:
                    trimmed[sym] = df[df["date"].astype(str) <= as_of_date]
                else:
                    trimmed[sym] = df
            ohlcv_map = trimmed
        # Only keep symbols requested
        if symbols:
            ohlcv_map = {s: ohlcv_map[s] for s in symbols if s in ohlcv_map}
    else:
        print(f"[trade_screener] Fetch OHLCV batch ({len(symbols)} mã)...")
        ohlcv_map = get_ohlcv_batch(symbols, n_days=200)

        # Trim OHLCV theo as_of_date để tránh leak future data trong backtest
        if as_of_date:
            trimmed = {}
            for sym, df in ohlcv_map.items():
                if df is None or df.empty:
                    trimmed[sym] = df
                    continue
                if "date" in df.columns:
                    trimmed[sym] = df[df["date"].astype(str) <= as_of_date]
                else:
                    trimmed[sym] = df
            ohlcv_map = trimmed

    candidates: list[TradeCandidate] = []
    skipped = 0

    for symbol, df in ohlcv_map.items():
        if df is None or df.empty or len(df) < 60:
            skipped += 1
            continue

        avg_vol_20 = float(df["volume"].tail(20).mean())
        if avg_vol_20 < 300_000:
            skipped += 1
            continue

        ind = compute_indicators(df)
        if not ind:
            skipped += 1
            continue

        if len(df) >= 2:
            prev_close = float(df["close"].iloc[-2])
            cur_close = float(df["close"].iloc[-1])
            ind["stock_day_change_pct"] = (
                round((cur_close - prev_close) / prev_close * 100, 2) if prev_close else 0.0
            )
        ind["avg_volume_20d"] = round(avg_vol_20)

        # Rank by volume momentum + price-volume intelligence
        vol_ratio = float(df["volume"].iloc[-1]) / avg_vol_20 if avg_vol_20 > 0 else 1.0
        pv = ind.get("price_volume") or {}
        pv_score_raw  = int(pv.get("price_volume_score") or 0)   # -100..+100
        pv_entry_bias = pv.get("entry_bias", "neutral")
        pv_setup_tags = pv.get("setup_tags", [])
        # Map PV score to a signed contribution: -20..+20 points on priority
        pv_contribution = pv_score_raw * 0.20
        priority_score = min(round(vol_ratio * 32 + avg_vol_20 / 1_000_000 * 5 + pv_contribution, 1), 100.0)
        # Hard cap: "avoid" stocks cannot become top BUY candidates
        if pv_entry_bias == "avoid":
            priority_score = min(priority_score, 30.0)
        ind["pv_setup_tags_screener"] = pv_setup_tags

        candidates.append(TradeCandidate(
            symbol=symbol,
            priority_score=priority_score,
            market_context=market_ctx,
            indicators=ind,
            setup_type="UNKNOWN",
        ))

    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    candidates = candidates[:max_candidates]

    print(f"[trade_screener] Skipped (data quality): {skipped}")
    print(f"[trade_screener] {len(candidates)} candidates → LLM")
    for c in candidates[:10]:
        vol_ratio = c.indicators.get("volume_ratio_20") or 0
        print(f"  {c.symbol:6s} | vol_ratio={vol_ratio:.1f}× | score={c.priority_score:.0f}")

    return market_ctx, candidates
