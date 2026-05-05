"""Plan Scorer — system filter tách LLM khỏi quyết định cuối.

TradingAgents propose → evaluate_plan() score → system decide.
LLM không phải trader chính. System filter mới quyết định có vào lệnh không.

Score range: 0.0–1.0
Threshold mặc định: SCORE_THRESHOLD = 0.60
"""

from __future__ import annotations

from multiagents_trading_assistant.tradingagents_vn.schema import TradePlan

SCORE_THRESHOLD = 0.60   # Chỉ vào lệnh nếu score >= threshold


def evaluate_plan(
    plan: TradePlan,
    features: dict,
    threshold: float = SCORE_THRESHOLD,
) -> tuple[float, bool]:
    """Score TradePlan và quyết định có vào lệnh không.

    Args:
        plan: TradePlan đã validate.
        features: indicators dict từ TradeCandidate (output của screener).
        threshold: Ngưỡng score tối thiểu để vào lệnh.

    Returns:
        (score: float 0.0–1.0, should_trade: bool)
    """
    if plan.action != "MUA":
        return 0.0, False

    score = 0.0

    # Component 1: LLM confidence (30%)
    score += plan.confidence * 0.30

    # Component 2: R:R quality (30%)
    # R:R 1.5 → 0.5 điểm; R:R 2.0 → 0.75; R:R 3.0+ → 1.0
    rr = plan.rr_ratio
    rr_score = min(rr / 3.0, 1.0)
    score += rr_score * 0.30

    # Component 3: Technical alignment — screener indicators xác nhận (20%)
    tech_score = _technical_alignment(features)
    score += tech_score * 0.20

    # Component 4: Market regime bonus (20%)
    regime_score = _market_regime_score(features)
    score += regime_score * 0.20

    score = round(min(score, 1.0), 4)
    return score, score >= threshold


def _technical_alignment(features: dict) -> float:
    """Kiểm tra indicators từ screener xác nhận plan hay không.

    Trả về 0.0–1.0. Không có features → 0.5 (neutral).
    """
    if not features:
        return 0.5

    checks = 0
    passed = 0

    # RSI không overbought
    rsi = features.get("rsi")
    if rsi is not None:
        checks += 1
        if 35 <= rsi <= 70:
            passed += 1

    # MACD histogram dương
    macd_hist = features.get("macd_hist")
    if macd_hist is not None:
        checks += 1
        if float(macd_hist) > 0:
            passed += 1

    # Giá trên MA20
    close = features.get("close") or features.get("current_price")
    ma20 = features.get("ma20")
    if close and ma20:
        checks += 1
        if float(close) > float(ma20):
            passed += 1

    # Volume ratio > 1.0 (vol hiện tại cao hơn trung bình)
    vol_ratio = features.get("volume_ratio_20")
    if vol_ratio is not None:
        checks += 1
        if float(vol_ratio) > 1.0:
            passed += 1

    # ADX > 20 (có trend)
    adx = features.get("adx")
    if adx is not None:
        checks += 1
        if float(adx) > 20:
            passed += 1

    if checks == 0:
        return 0.5
    return round(passed / checks, 3)


def _market_regime_score(features: dict) -> float:
    """Bonus điểm theo market regime.

    UPTREND → 1.0
    SIDEWAY → 0.6
    DOWNTREND → 0.2
    Unknown → 0.5
    """
    # market_context có thể được nhúng vào features hoặc không
    trend = features.get("market_trend") or features.get("reference_trend")
    if trend == "UPTREND":
        return 1.0
    if trend == "SIDEWAY":
        return 0.6
    if trend == "DOWNTREND":
        return 0.2
    return 0.5
