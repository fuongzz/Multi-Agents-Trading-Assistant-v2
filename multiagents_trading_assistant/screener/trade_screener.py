"""trade_screener.py — TA screener cho Trade pipeline.

5 chiến lược: BREAKOUT / RETEST / SPRING / MA_PULLBACK / RSI_BOUNCE
Giữ nguyên logic từ _legacy/screener.py — chỉ đổi tên class và path.

Output: top 10 TradeCandidate sort theo priority_score.
"""

import importlib.metadata  # noqa: F401 — pandas-ta-openbb Python 3.11 fix
from dataclasses import dataclass, field

import pandas as pd

from multiagents_trading_assistant.services.data_service import (
    get_ohlcv_batch,
    get_vnindex,
    get_vn30_symbols,
)
from multiagents_trading_assistant.indicators import compute_indicators


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class MarketContext:
    """Bức tranh tổng quan VN-Index — gate đầu tiên cho Trade pipeline."""
    trend: str            # UPTREND | SIDEWAY | DOWNTREND
    should_trade: bool    # False khi DOWNTREND
    ma20: float
    ma60: float
    ma200: float
    sideway_zone: str     # NEAR_SUPPORT | MID_RANGE | NEAR_RESISTANCE
    position_in_range: float
    vni_change_pct: float
    current_price: float


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
# Market context
# ──────────────────────────────────────────────

def get_market_context(vnindex_df: pd.DataFrame | None = None) -> MarketContext:
    if vnindex_df is None or vnindex_df.empty:
        vnindex_df = get_vnindex(200)
    if vnindex_df.empty:
        raise RuntimeError("Không lấy được dữ liệu VN-Index — abort trade_screener")

    close   = vnindex_df["close"]
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

    high_20 = float(close.rolling(20).max().iloc[-1])
    low_20  = float(close.rolling(20).min().iloc[-1])
    rng = high_20 - low_20
    pos = (current - low_20) / rng if rng > 0 else 0.5

    if pos <= 0.25:
        zone = "NEAR_SUPPORT"
    elif pos >= 0.75:
        zone = "NEAR_RESISTANCE"
    else:
        zone = "MID_RANGE"

    return MarketContext(
        trend             = trend,
        should_trade      = trend != "DOWNTREND",
        ma20              = ma20,
        ma60              = ma60,
        ma200             = ma200,
        sideway_zone      = zone,
        position_in_range = round(pos, 3),
        vni_change_pct    = round((current - prev) / prev * 100, 2),
        current_price     = current,
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


# ──────────────────────────────────────────────
# Priority score
# ──────────────────────────────────────────────

_STRATEGY_BASE_SCORE = {
    "BREAKOUT":    80,
    "RETEST":      75,
    "SPRING":      70,
    "MA_PULLBACK": 65,
    "RSI_BOUNCE":  55,
}


def compute_priority_score(setup_type: str, ind: dict) -> float:
    score = _STRATEGY_BASE_SCORE.get(setup_type, 50)
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
    return min(round(score, 1), 100.0)


# ──────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────

def run_screener(
    symbols: list[str] | None = None,
    max_candidates: int = 10,
) -> tuple[MarketContext, list[TradeCandidate]]:
    if symbols is None:
        symbols = get_vn30_symbols()

    print(f"[trade_screener] Scan {len(symbols)} mã...")

    market_ctx = get_market_context()
    print(f"[trade_screener] VN-Index: {market_ctx.trend} | Δ {market_ctx.vni_change_pct:+.2f}%")

    if not market_ctx.should_trade:
        print("[trade_screener] DOWNTREND → dừng scan")
        return market_ctx, []

    print(f"[trade_screener] Fetch OHLCV batch ({len(symbols)} mã)...")
    ohlcv_map = get_ohlcv_batch(symbols, n_days=200)

    candidates: list[TradeCandidate] = []
    min_liquidity = 300_000
    skipped = 0

    for symbol, df in ohlcv_map.items():
        if df is None or df.empty or len(df) < 30:
            continue
        if float(df["volume"].tail(20).mean()) < min_liquidity:
            skipped += 1
            continue
        ind = compute_indicators(df)
        if not ind:
            continue

        detected: list[tuple[str, list[str]]] = []
        strategies = [
            ("BREAKOUT",    detect_breakout),
            ("RETEST",      detect_retest),
            ("SPRING",      detect_spring),
            ("MA_PULLBACK", detect_ma_pullback),
            ("RSI_BOUNCE",  detect_rsi_bounce),
        ]
        if market_ctx.trend == "SIDEWAY":
            strategies = [s for s in strategies if s[0] != "BREAKOUT"]

        for name, fn in strategies:
            passed, reasons = fn(df, ind)
            if passed:
                detected.append((name, reasons))

        for setup_type, reasons in detected:
            candidates.append(TradeCandidate(
                symbol         = symbol,
                setup_type     = setup_type,
                priority_score = compute_priority_score(setup_type, ind),
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
