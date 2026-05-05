"""Common scoring language for technical setups.

Detectors answer one narrow question: "did this pattern appear?".
This module answers the comparable question: "how strong is this setup on
the same 0-100 evidence scale as every other setup?".
"""

from __future__ import annotations

from typing import Any


MOMENTUM_SETUPS = {
    "BREAKOUT",
    "MOMENTUM_SURGE",
    "MACD_CROSSOVER",
    "GOLDEN_CROSS",
    "FLAG_PENNANT",
    "KUMO_BREAKOUT",
    "TK_CROSS",
    "KUMO_TWIST_ENTRY",
    "ADX_TREND",
    "AROON_TREND_SHIFT",
    "LINREG_MOMENTUM",
}

PULLBACK_SETUPS = {
    "RETEST",
    "MA_PULLBACK",
    "TREND_PULLBACK",
    "BREAKOUT_RETEST_ENTRY",
    "KIJUN_BOUNCE",
    "SUPERTREND_PULLBACK",
    "OBV_ACCUMULATION",
}

COMPRESSION_SETUPS = {
    "BB_SQUEEZE",
    "INSIDE_BAR",
    "NR7",
    "KELTNER_SQUEEZE",
}

REVERSAL_SETUPS = {
    "SPRING",
    "DOUBLE_BOTTOM",
    "HAMMER",
    "RSI_BOUNCE",
    "BULLISH_ENGULFING",
    "PIN_BAR",
    "OVERSOLD_MEAN_REVERSION",
}

# Money-flow gating sets — shared by live pipeline and backtest engine
MF_STRICT_SETUPS: frozenset[str] = frozenset({
    "BREAKOUT", "MOMENTUM_SURGE", "MACD_CROSSOVER", "KUMO_BREAKOUT", "TK_CROSS",
    "ADX_TREND", "AROON_TREND_SHIFT", "LINREG_MOMENTUM",
})
MF_COMPRESSION_SETUPS: frozenset[str] = frozenset({
    "NR7", "BB_SQUEEZE", "INSIDE_BAR", "FLAG_PENNANT", "KELTNER_SQUEEZE",
})
MF_REVERSAL_SETUPS: frozenset[str] = frozenset({
    "RETEST", "SPRING", "HAMMER", "RSI_BOUNCE", "DOUBLE_BOTTOM",
    "BULLISH_ENGULFING", "PIN_BAR", "BREAKOUT_RETEST_ENTRY",
    "TREND_PULLBACK", "KIJUN_BOUNCE", "KUMO_TWIST_ENTRY",
    "SUPERTREND_PULLBACK", "OBV_ACCUMULATION", "OVERSOLD_MEAN_REVERSION",
})

# Minimum confluence threshold by market regime — shared by all gating layers
MIN_CONFLUENCE_BY_REGIME: dict[str, float] = {
    "UPTREND": 55.0,
    "SIDEWAY": 62.0,
    "DOWNTREND": 70.0,
}


def score_setup(
    setup_type: str,
    indicators: dict[str, Any],
    *,
    money_flow: dict[str, Any] | None = None,
    reference_trend: str | None = None,
) -> dict[str, Any]:
    """Return a comparable technical setup score on a 0-100 scale.

    Dimensions are intentionally stable across setup families:
    trend_alignment (0-25), momentum (0-20), volume_flow (0-20),
    location_structure (0-25), volatility_risk (0-10).
    """
    ind = _indicators_view(indicators)
    setup = (setup_type or "").upper()
    ref_trend = (reference_trend or ind.get("ma_trend") or "SIDEWAY").upper()

    dims = {
        "trend_alignment": _trend_points(setup, ind, ref_trend),
        "momentum": _momentum_points(setup, ind),
        "volume_flow": _volume_points(setup, ind, money_flow),
        "location_structure": _location_points(setup, ind),
        "volatility_risk": _volatility_points(setup, ind),
    }
    score = round(sum(dims.values()), 1)
    score = max(0.0, min(100.0, score))
    quality = "STRONG" if score >= 70 else "MEDIUM" if score >= 50 else "WEAK"

    return {
        "setup_type": setup,
        "setup_family": _setup_family(setup),
        "score": score,
        "quality": quality,
        "dimensions": {k: round(v, 1) for k, v in dims.items()},
        "reference_trend": ref_trend,
    }


def _indicators_view(indicators: dict[str, Any]) -> dict[str, Any]:
    snapshot = indicators.get("indicator_snapshot")
    if isinstance(snapshot, dict):
        merged = dict(snapshot)
        merged.update({k: v for k, v in indicators.items() if k not in {"indicator_snapshot"}})
        return merged
    return indicators


def _setup_family(setup: str) -> str:
    if setup in MOMENTUM_SETUPS:
        return "MOMENTUM"
    if setup in PULLBACK_SETUPS:
        return "PULLBACK"
    if setup in COMPRESSION_SETUPS:
        return "COMPRESSION"
    if setup in REVERSAL_SETUPS:
        return "REVERSAL"
    return "GENERAL"


def _trend_points(setup: str, ind: dict[str, Any], ref_trend: str) -> float:
    ma_trend = str(ind.get("ma_trend") or ref_trend or "SIDEWAY").upper()
    ma_phase = str(ind.get("ma_phase") or "").upper()
    ichimoku_regime = str(ind.get("ichimoku_regime") or "").upper()

    if setup in MOMENTUM_SETUPS:
        points = {"UPTREND": 25.0, "SIDEWAY": 10.0, "DOWNTREND": 0.0}.get(ref_trend, 10.0)
        if ma_trend == "UPTREND":
            points += 3.0
        if setup in {"KUMO_BREAKOUT", "TK_CROSS", "KUMO_TWIST_ENTRY"}:
            points += 5.0 if ichimoku_regime == "BULLISH" else -8.0 if ichimoku_regime == "BEARISH" else -2.0
        if ma_phase == "OVERBOUGHT":
            points -= 7.0
        return _clip(points, 0.0, 25.0)

    if setup in PULLBACK_SETUPS:
        points = 22.0 if ma_trend == "UPTREND" else 14.0 if ref_trend == "SIDEWAY" else 6.0
        if setup == "KIJUN_BOUNCE":
            points += 5.0 if ichimoku_regime == "BULLISH" else -8.0
        if ma_phase == "PULLBACK":
            points += 3.0
        return _clip(points, 0.0, 25.0)

    if setup in COMPRESSION_SETUPS:
        return {"SIDEWAY": 23.0, "UPTREND": 18.0, "DOWNTREND": 5.0}.get(ref_trend, 15.0)

    if setup in REVERSAL_SETUPS:
        if ref_trend == "DOWNTREND":
            return 18.0
        if ref_trend == "SIDEWAY":
            return 22.0
        return 12.0

    return {"UPTREND": 18.0, "SIDEWAY": 14.0, "DOWNTREND": 6.0}.get(ref_trend, 14.0)


def _momentum_points(setup: str, ind: dict[str, Any]) -> float:
    rsi = _num(ind.get("rsi"))
    macd_label = str(ind.get("macd_signal_label") or ind.get("macd_signal") or "").upper()
    points = 0.0

    if macd_label == "BULLISH":
        points += 8.0
    elif macd_label == "NEUTRAL":
        points += 4.0

    if setup.startswith("KUMO") or setup in {"TK_CROSS", "KIJUN_BOUNCE"}:
        if ind.get("ichimoku_chikou_confirm"):
            points += 5.0
        if ind.get("ichimoku_future_cloud_green"):
            points += 4.0

    if rsi is None:
        points += 4.0
    elif setup in REVERSAL_SETUPS:
        if 30 <= rsi <= 55:
            points += 10.0
        elif 55 < rsi <= 68:
            points += 6.0
        elif rsi < 30:
            points += 7.0
        else:
            points -= 4.0
    elif setup in MOMENTUM_SETUPS:
        if 50 <= rsi <= 70:
            points += 10.0
        elif 40 <= rsi < 50 or 70 < rsi <= 76:
            points += 5.0
        else:
            points -= 3.0
    else:
        if 38 <= rsi <= 68:
            points += 9.0
        elif 68 < rsi <= 74:
            points += 3.0

    return _clip(points, 0.0, 20.0)


def _volume_points(setup: str, ind: dict[str, Any], money_flow: dict[str, Any] | None) -> float:
    vol = _num(ind.get("volume_current"))
    vol_ma = _num(ind.get("volume_ma20"))
    ratio = vol / vol_ma if vol and vol_ma and vol_ma > 0 else 1.0

    if setup in MOMENTUM_SETUPS:
        points = 4.0 + min(max(ratio - 1.0, 0.0) * 14.0, 11.0)
    elif setup in PULLBACK_SETUPS | COMPRESSION_SETUPS:
        points = 12.0 if 0.55 <= ratio <= 1.6 else 7.0 if ratio < 2.2 else 3.0
    elif setup in REVERSAL_SETUPS:
        points = 9.0 if ratio >= 0.8 else 5.0
    else:
        points = 8.0

    if money_flow:
        regime = str(money_flow.get("regime") or "").upper()
        mf_score = _num(money_flow.get("score")) or 0.0
        if regime in {"BREAKOUT_FLOW", "EARLY_MONEY_IN", "MONEY_IN"}:
            points += 5.0
        elif regime == "ACCUMULATION":
            points += 3.0
        elif regime in {"MONEY_OUT", "DISTRIBUTION", "EXHAUSTION_INFLOW"}:
            points -= 7.0
        elif regime == "CHASE_MONEY_IN":
            points -= 4.0
        points += _clip(mf_score, -3.0, 4.0)

    return _clip(points, 0.0, 20.0)


def _location_points(setup: str, ind: dict[str, Any]) -> float:
    price = _num(ind.get("current_price"))
    supports = [_num(x) for x in (ind.get("support_levels") or [])]
    resistances = [_num(x) for x in (ind.get("resistance_levels") or [])]
    supports = [x for x in supports if x and price and x < price]
    resistances = [x for x in resistances if x and price and x > price]

    support_dist = ((price - max(supports)) / price * 100.0) if price and supports else None
    resistance_room = ((min(resistances) - price) / price * 100.0) if price and resistances else None

    if setup in MOMENTUM_SETUPS:
        if resistance_room is None:
            points = 18.0
        elif resistance_room >= 8.0:
            points = 23.0
        elif resistance_room >= 4.0:
            points = 17.0
        else:
            points = 8.0
    elif setup in PULLBACK_SETUPS | REVERSAL_SETUPS:
        if support_dist is None:
            points = 12.0
        elif support_dist <= 2.0:
            points = 24.0
        elif support_dist <= 5.0:
            points = 18.0
        else:
            points = 9.0
    elif setup in COMPRESSION_SETUPS:
        points = 18.0
        if support_dist is not None and support_dist <= 4.0:
            points += 3.0
        if resistance_room is not None and resistance_room < 2.0:
            points -= 5.0
    else:
        points = 14.0

    return _clip(points, 0.0, 25.0)


def _volatility_points(setup: str, ind: dict[str, Any]) -> float:
    price = _num(ind.get("current_price"))
    atr = _num(ind.get("atr"))
    atr_pct = atr / price * 100.0 if price and atr and price > 0 else None
    ma_phase = str(ind.get("ma_phase") or "").upper()

    if atr_pct is None:
        points = 6.0
    elif 1.0 <= atr_pct <= 4.0:
        points = 10.0
    elif 0.5 <= atr_pct < 1.0 or 4.0 < atr_pct <= 6.0:
        points = 6.0
    else:
        points = 3.0

    if ma_phase == "OVERBOUGHT" and setup not in {"BREAKOUT", "MOMENTUM_SURGE"}:
        points -= 4.0
    return _clip(points, 0.0, 10.0)


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
