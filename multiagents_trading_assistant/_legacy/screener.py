"""
screener.py — Signal-First Screener cho AI Trading Assistant.

Chạy TRƯỚC toàn bộ multi-agent pipeline để tiết kiệm API cost.
Không pass screener → không gọi LLM nào.

Flow:
  VN30 (30 mã)
    → [1] Market Context  — VN-Index đang UPTREND / SIDEWAY / DOWNTREND?
    → [2] Signal Scanner  — Chạy 5 chiến lược song song trên từng mã
    → [3] Rank & Filter   — Chấm priority_score, lấy top 10
    → candidates[]        → Truyền xuống orchestrator
"""

import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
from dataclasses import dataclass, field
from statistics import mean

import pandas as pd

from multiagents_trading_assistant.fetcher import (
    get_ohlcv,
    get_ohlcv_batch,
    get_vnindex,
    get_vn30_symbols,
)
from multiagents_trading_assistant.indicators import compute_indicators, compute_support_resistance


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class MarketContext:
    """Bức tranh tổng quan VN-Index — đầu vào cho mọi quyết định."""
    trend: str            # UPTREND | SIDEWAY | DOWNTREND
    should_trade: bool    # False khi DOWNTREND — dừng pipeline
    ma20: float
    ma60: float
    ma200: float
    sideway_zone: str     # NEAR_SUPPORT | MID_RANGE | NEAR_RESISTANCE
    position_in_range: float  # 0.0 → 1.0
    vni_change_pct: float     # % thay đổi so với hôm qua (cho Rule 1 Risk Manager)
    current_price: float


@dataclass
class CandidateStock:
    """1 cổ phiếu pass screener — truyền xuống orchestrator."""
    symbol: str
    setup_type: str        # BREAKOUT | RETEST | SPRING | MA_PULLBACK | RSI_BOUNCE
    priority_score: float  # 0–100, càng cao càng ưu tiên
    market_context: MarketContext
    indicators: dict       # full indicators dict — tái dùng, không tính lại
    reasons: list[str] = field(default_factory=list)  # lý do pass screener


# ──────────────────────────────────────────────
# Bước 1 — Market Context
# ──────────────────────────────────────────────

def get_market_context(vnindex_df: pd.DataFrame = None) -> MarketContext:
    """
    Phân tích VN-Index để xác định xu hướng thị trường.
    Đây là gate đầu tiên — DOWNTREND thì dừng toàn bộ.
    """
    if vnindex_df is None or vnindex_df.empty:
        vnindex_df = get_vnindex(200)

    if vnindex_df.empty:
        raise RuntimeError("Không lấy được dữ liệu VN-Index — abort screener")

    close   = vnindex_df["close"]
    current = float(close.iloc[-1])
    prev    = float(close.iloc[-2]) if len(close) >= 2 else current

    ma20  = float(close.rolling(20).mean().iloc[-1])
    ma60  = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60  else ma20
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else ma60

    # Xác định xu hướng
    if current > ma20 > ma60 > ma200:
        trend = "UPTREND"
    elif current < ma20 < ma60:
        trend = "DOWNTREND"
    else:
        trend = "SIDEWAY"

    # Vị trí trong range 20 phiên (dùng cho đồng pha sideway)
    high_20 = float(close.rolling(20).max().iloc[-1])
    low_20  = float(close.rolling(20).min().iloc[-1])
    rng = high_20 - low_20
    pos = (current - low_20) / rng if rng > 0 else 0.5

    if pos <= 0.25:
        sideway_zone = "NEAR_SUPPORT"
    elif pos >= 0.75:
        sideway_zone = "NEAR_RESISTANCE"
    else:
        sideway_zone = "MID_RANGE"

    vni_change_pct = round((current - prev) / prev * 100, 2)

    return MarketContext(
        trend             = trend,
        should_trade      = (trend != "DOWNTREND"),
        ma20              = ma20,
        ma60              = ma60,
        ma200             = ma200,
        sideway_zone      = sideway_zone,
        position_in_range = round(pos, 3),
        vni_change_pct    = vni_change_pct,
        current_price     = current,
    )


# ──────────────────────────────────────────────
# Bước 2 — 5 chiến lược phát hiện signal
# ──────────────────────────────────────────────

def detect_breakout(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """
    Chiến lược 1 — BREAKOUT (chỉ trong UPTREND)
    Điều kiện:
      - Nến hôm nay đóng cửa trên đỉnh 20 phiên trước (breakout kháng cự)
      - Volume > 1.5× TB20 (dòng tiền lớn xác nhận)
      - Thân nến > 2% (nến mạnh, không phải râu)
      - RSI < 75 (chưa overbought)
    """
    reasons = []
    if df.empty or len(df) < 22:
        return False, []

    close  = df["close"]
    volume = df["volume"]

    current_close  = float(close.iloc[-1])
    prev_high_20   = float(close.iloc[-21:-1].max())  # đỉnh 20 phiên trước (không tính hôm nay)
    current_open   = float(df["open"].iloc[-1])
    volume_current = float(volume.iloc[-1])
    volume_ma20    = float(volume.rolling(20).mean().iloc[-1])

    rsi = ind.get("rsi")

    # Điều kiện breakout
    is_breakout    = current_close > prev_high_20
    is_vol_surge   = volume_current > volume_ma20 * 1.5
    candle_body    = abs(current_close - current_open) / current_open * 100
    is_big_candle  = candle_body > 2.0
    not_overbought = rsi is None or rsi < 75

    if is_breakout:
        reasons.append(f"Giá đóng cửa ({current_close:.0f}) vượt đỉnh 20 phiên ({prev_high_20:.0f})")
    if is_vol_surge:
        ratio = volume_current / volume_ma20
        reasons.append(f"Volume đột biến {ratio:.1f}× TB20")
    if is_big_candle:
        reasons.append(f"Thân nến lớn {candle_body:.1f}%")

    passed = is_breakout and is_vol_surge and is_big_candle and not_overbought
    return passed, reasons if passed else []


def detect_retest(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """
    Chiến lược 2 — RETEST sau Breakout (chỉ trong UPTREND)
    Điều kiện:
      - Trong 5-15 phiên trước đã có breakout (đỉnh 20 phiên bị vượt)
      - Giá hiện tại pullback về vùng breakout đó (±3%)
      - Volume hôm nay cạn (< TB20) — không có áp lực bán
      - Giá không đóng dưới vùng breakout (vẫn giữ được)
    """
    reasons = []
    if df.empty or len(df) < 25:
        return False, []

    close  = df["close"]
    volume = df["volume"]

    current_close  = float(close.iloc[-1])
    volume_current = float(volume.iloc[-1])
    volume_ma20    = float(volume.rolling(20).mean().iloc[-1])

    # Tìm mức breakout: đỉnh của đoạn 5-20 phiên trước
    breakout_level = float(close.iloc[-20:-5].max())

    # Kiểm tra: giá hiện tại đang retest về vùng breakout
    dist_pct = (current_close - breakout_level) / breakout_level * 100
    is_near_breakout = -3.0 <= dist_pct <= 5.0  # trong vùng ±3% của breakout level

    # Volume cạn = không có áp lực bán
    is_vol_dry = volume_current < volume_ma20 * 0.8

    # Giá vẫn trên vùng breakout (không breakdown)
    is_holding = current_close >= breakout_level * 0.97

    if is_near_breakout:
        reasons.append(f"Giá đang retest vùng breakout {breakout_level:.0f} (cách {dist_pct:+.1f}%)")
    if is_vol_dry:
        ratio = volume_current / volume_ma20
        reasons.append(f"Volume cạn {ratio:.2f}× TB20 — không có áp lực bán")
    if is_holding:
        reasons.append("Giá vẫn giữ được trên vùng breakout")

    passed = is_near_breakout and is_vol_dry and is_holding
    return passed, reasons if passed else []


def detect_spring(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """
    Chiến lược 3 — SPRING / Higher High Higher Low (chỉ trong UPTREND)
    Điều kiện:
      - Đỉnh gần nhất cao hơn đỉnh trước (HH)
      - Đáy gần nhất cao hơn đáy trước (HL)
      - Volume phiên tăng cao hơn phiên giảm (dòng tiền vào)
      - Đang trong nhịp tăng mới (giá > MA20)
    """
    reasons = []
    if df.empty or len(df) < 30:
        return False, []

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # Tìm 2 đỉnh gần nhất và 2 đáy gần nhất (lookback 20 phiên)
    window = 20
    recent = df.iloc[-window:]

    # Tìm local highs và local lows trong 20 phiên
    local_highs = []
    local_lows  = []

    for i in range(2, len(recent) - 2):
        h = float(recent["high"].iloc[i])
        l = float(recent["low"].iloc[i])

        if h > float(recent["high"].iloc[i-1]) and h > float(recent["high"].iloc[i+1]):
            local_highs.append((i, h))
        if l < float(recent["low"].iloc[i-1]) and l < float(recent["low"].iloc[i+1]):
            local_lows.append((i, l))

    if len(local_highs) < 2 or len(local_lows) < 2:
        return False, []

    # Lấy 2 đỉnh / 2 đáy gần nhất
    hh1, hh2 = local_highs[-2][1], local_highs[-1][1]  # đỉnh cũ, đỉnh mới
    hl1, hl2 = local_lows[-2][1],  local_lows[-1][1]   # đáy cũ, đáy mới

    is_higher_high = hh2 > hh1
    is_higher_low  = hl2 > hl1

    # Volume xác nhận: phiên tăng có volume cao hơn phiên giảm
    up_vol   = volume[close > close.shift(1)].mean()
    down_vol = volume[close < close.shift(1)].mean()
    is_vol_confirm = (up_vol > down_vol) if (up_vol > 0 and down_vol > 0) else False

    # Giá trên MA20 (đang trong uptrend ngắn hạn)
    ma20  = ind.get("ma20")
    price = ind.get("current_price")
    above_ma20 = (price > ma20) if (price and ma20) else False

    if is_higher_high:
        reasons.append(f"Đỉnh sau ({hh2:.0f}) cao hơn đỉnh trước ({hh1:.0f}) — Higher High")
    if is_higher_low:
        reasons.append(f"Đáy sau ({hl2:.0f}) cao hơn đáy trước ({hl1:.0f}) — Higher Low")
    if is_vol_confirm:
        reasons.append("Volume phiên tăng > phiên giảm — dòng tiền vào")

    passed = is_higher_high and is_higher_low and is_vol_confirm and above_ma20
    return passed, reasons if passed else []


def detect_ma_pullback(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """
    Chiến lược 4 — MA PULLBACK (trong UPTREND)
    Điều kiện:
      - Giá đang trong UPTREND (trên MA20 > MA60 > MA200)
      - Giá pullback về gần MA60 (cách MA60 < 5%)
      - RSI về vùng 35-55 (không overbought)
      - Volume 3 phiên gần nhất giảm dần (cạn áp lực bán)
    """
    reasons = []

    price = ind.get("current_price")
    ma20  = ind.get("ma20")
    ma60  = ind.get("ma60")
    ma200 = ind.get("ma200")
    rsi   = ind.get("rsi")

    if not all([price, ma20, ma60, ma200, rsi]):
        return False, []

    # Phải đang UPTREND
    is_uptrend = price > ma20 > ma60 > ma200

    # Giá gần MA60 — nới từ 5% → 8%
    dist_ma60_pct = (price - ma60) / ma60 * 100
    near_ma60 = 0 <= dist_ma60_pct <= 8.0

    # RSI vùng trung tính — nới rộng từ 35-55 → 30-60
    rsi_ok = 30 <= rsi <= 60

    # Volume 3 phiên gần nhất giảm dần — nới: 2/3 phiên giảm là đủ
    if len(df) >= 4:
        v1 = float(df["volume"].iloc[-4])
        v2 = float(df["volume"].iloc[-3])
        v3 = float(df["volume"].iloc[-2])
        vol_declining = (v1 > v2) or (v2 > v3)  # nới: chỉ cần 1 trong 2 giảm
    else:
        vol_declining = False

    if is_uptrend:
        reasons.append("Xu hướng UPTREND còn nguyên (giá > MA20 > MA60 > MA200)")
    if near_ma60:
        reasons.append(f"Giá pullback về MA60 (cách {dist_ma60_pct:.1f}%)")
    if rsi_ok:
        reasons.append(f"RSI {rsi:.1f} — về vùng trung tính, chưa oversold")
    if vol_declining:
        reasons.append("Volume 3 phiên liên tiếp giảm — áp lực bán cạn dần")

    passed = is_uptrend and near_ma60 and rsi_ok and vol_declining
    return passed, reasons if passed else []


def detect_rsi_bounce(df: pd.DataFrame, ind: dict) -> tuple[bool, list[str]]:
    """
    Chiến lược 5 — RSI OVERSOLD BOUNCE
    Điều kiện:
      - RSI < 35 (oversold)
      - Giá vẫn trên MA200 (downtrend ngắn hạn nhưng uptrend dài hạn còn)
      - Nến hôm nay bắt đầu đảo chiều: đóng cửa > mở cửa (nến tăng sau chuỗi giảm)
      - KHÔNG áp dụng khi VN-Index DOWNTREND dài hạn
    """
    reasons = []

    price  = ind.get("current_price")
    ma200  = ind.get("ma200")
    rsi    = ind.get("rsi")

    if not all([price, rsi]):
        return False, []

    # RSI oversold (nới từ 35 → 40)
    is_oversold = rsi < 40

    # Còn trên MA200 (trend dài hạn vẫn tốt)
    above_ma200 = (price > ma200) if ma200 else False

    # Nến đảo chiều: đóng > mở (nến xanh sau chuỗi đỏ)
    if len(df) >= 2:
        current_close = float(df["close"].iloc[-1])
        current_open  = float(df["open"].iloc[-1])
        prev_close    = float(df["close"].iloc[-2])
        reversal_candle = current_close > current_open and current_close > prev_close
    else:
        reversal_candle = False

    if is_oversold:
        reasons.append(f"RSI {rsi:.1f} — vùng oversold, tiềm năng bật lên")
    if above_ma200:
        reasons.append(f"Giá ({price:.0f}) vẫn trên MA200 ({ma200:.0f}) — trend dài hạn tốt")
    if reversal_candle:
        reasons.append("Nến đảo chiều tăng sau chuỗi giảm")

    passed = is_oversold and above_ma200 and reversal_candle
    return passed, reasons if passed else []


# ──────────────────────────────────────────────
# Bước 3 — Tính priority score
# ──────────────────────────────────────────────

# Trọng số ưu tiên theo chiến lược
_STRATEGY_BASE_SCORE = {
    "BREAKOUT":    80,
    "RETEST":      75,
    "SPRING":      70,
    "MA_PULLBACK": 65,
    "RSI_BOUNCE":  55,
}


def compute_priority_score(setup_type: str, ind: dict) -> float:
    """
    Tính priority score 0–100 để sort candidates.
    Base score theo chiến lược + bonus từ indicators.
    """
    score = _STRATEGY_BASE_SCORE.get(setup_type, 50)

    # +5 nếu confluence score cao (≥ 6)
    confluence = ind.get("confluence_score", 0)
    if confluence >= 6:
        score += 5
    elif confluence >= 4:
        score += 2

    # +3 nếu volume đột biến
    if ind.get("volume_surge"):
        score += 3

    # +3 nếu MACD bullish
    if ind.get("macd_signal_label") == "BULLISH":
        score += 3

    # -5 nếu RSI overbought (gần kháng cự — quy tắc vàng)
    rsi = ind.get("rsi")
    if rsi and rsi > 70:
        score -= 5

    return min(round(score, 1), 100.0)


# ──────────────────────────────────────────────
# Hàm chính — run_screener()
# ──────────────────────────────────────────────

def run_screener(
    symbols: list[str] = None,
    max_candidates: int = 10,
) -> tuple[MarketContext, list[CandidateStock]]:
    """
    Chạy toàn bộ screener pipeline.

    Returns:
        (market_ctx, candidates)
        candidates = [] nếu market DOWNTREND hoặc không có setup
    """
    # Lấy danh sách mã nếu không truyền vào
    if symbols is None:
        symbols = get_vn30_symbols()

    print(f"[screener] Bắt đầu scan {len(symbols)} mã...")

    # ── Bước 1: Market Context ──
    print("[screener] Bước 1/3: Phân tích VN-Index...")
    market_ctx = get_market_context()

    print(f"[screener] VN-Index: {market_ctx.trend} | "
          f"Giá {market_ctx.current_price:.0f} | "
          f"Change {market_ctx.vni_change_pct:+.2f}%")

    if not market_ctx.should_trade:
        print("[screener] DOWNTREND — dừng scan, không tốn API cost")
        return market_ctx, []

    # ── Bước 2: Fetch OHLCV batch (có rate limit sleep) ──
    print(f"[screener] Bước 2/3: Fetch OHLCV {len(symbols)} mã (có thể mất vài phút)...")
    ohlcv_map = get_ohlcv_batch(symbols, n_days=200)

    # ── Bước 3: Chạy 5 chiến lược trên từng mã ──
    print("[screener] Bước 3/3: Scan 5 chiến lược...")
    candidates = []

    _MIN_LIQUIDITY = 300_000  # cổ phiếu/ngày TB20
    skipped_liquidity = 0

    for symbol, df in ohlcv_map.items():
        if df is None or df.empty or len(df) < 30:
            continue

        # Lọc thanh khoản — TB20 volume phải > 300k cp/ngày
        avg_vol_20 = float(df["volume"].tail(20).mean())
        if avg_vol_20 < _MIN_LIQUIDITY:
            skipped_liquidity += 1
            continue

        # Tính indicators một lần, dùng cho tất cả 5 chiến lược
        ind = compute_indicators(df)
        if not ind:
            continue

        # Chạy 5 chiến lược
        # Chiến lược 1-4 chỉ chạy khi UPTREND
        # Chiến lược 5 (RSI bounce) chạy cả SIDEWAY
        detected = []

        if market_ctx.trend == "UPTREND":
            for strategy_name, detect_fn in [
                ("BREAKOUT",    detect_breakout),
                ("RETEST",      detect_retest),
                ("SPRING",      detect_spring),
                ("MA_PULLBACK", detect_ma_pullback),
                ("RSI_BOUNCE",  detect_rsi_bounce),
            ]:
                passed, reasons = detect_fn(df, ind)
                if passed:
                    detected.append((strategy_name, reasons))

        # SIDEWAY: RETEST + SPRING vẫn có giá trị trên từng cổ phiếu
        if market_ctx.trend == "SIDEWAY":
            for strategy_name, detect_fn in [
                ("RETEST",      detect_retest),
                ("SPRING",      detect_spring),
                ("MA_PULLBACK", detect_ma_pullback),
                ("RSI_BOUNCE",  detect_rsi_bounce),
            ]:
                passed, reasons = detect_fn(df, ind)
                if passed:
                    detected.append((strategy_name, reasons))

        # Nếu 1 mã pass nhiều chiến lược → lấy chiến lược ưu tiên cao nhất
        for setup_type, reasons in detected:
            score = compute_priority_score(setup_type, ind)
            candidates.append(CandidateStock(
                symbol         = symbol,
                setup_type     = setup_type,
                priority_score = score,
                market_context = market_ctx,
                indicators     = ind,
                reasons        = reasons,
            ))
            break  # Chỉ lấy chiến lược đầu tiên (đã sort theo priority)

    # Sort và lấy top N
    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    candidates = candidates[:max_candidates]

    print(f"[screener] Lọc thanh khoản <300k: bỏ {skipped_liquidity} mã")
    print(f"[screener] Kết quả: {len(candidates)} candidates")
    for c in candidates:
        print(f"  {c.symbol:6s} | {c.setup_type:12s} | score={c.priority_score:.0f} | {c.reasons[0] if c.reasons else ''}")

    return market_ctx, candidates


# ──────────────────────────────────────────────
# Chạy trực tiếp để test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Test với subset nhỏ để không bị rate limit
    test_symbols = ["VNM", "VCB", "HPG", "MBB", "FPT", "TCB", "MSN", "VIC", "GAS", "STB"]

    market_ctx, candidates = run_screener(symbols=test_symbols, max_candidates=5)

    print(f"\n=== KẾT QUẢ SCREENER ===")
    print(f"VN-Index: {market_ctx.trend} | Should trade: {market_ctx.should_trade}")
    print(f"Tìm được {len(candidates)} candidates:\n")

    for i, c in enumerate(candidates, 1):
        print(f"{i}. {c.symbol} — {c.setup_type} (score: {c.priority_score})")
        for r in c.reasons:
            print(f"   • {r}")
        print()
