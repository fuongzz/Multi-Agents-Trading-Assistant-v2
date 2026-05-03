"""Market/symbol regime classification for lifecycle backtests.

Regime comes before strategy selection:

    regime -> strategy family -> setup -> playbook -> risk/execution

The classifier is deterministic and causal. It only uses the OHLCV window up to
the current bar plus optional money-flow features already computed causally by
the backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from multiagents_trading_assistant.setup_scoring import (
    COMPRESSION_SETUPS,
    MOMENTUM_SETUPS,
    PULLBACK_SETUPS,
    REVERSAL_SETUPS,
)


class Regime(str, Enum):
    UPTREND = "UPTREND"
    SIDEWAY = "SIDEWAY"
    DOWNTREND = "DOWNTREND"
    ACCUMULATION = "ACCUMULATION"
    HIGH_VOL_DISTRIBUTION = "HIGH_VOL_DISTRIBUTION"


class StrategyFamily(str, Enum):
    TREND_FOLLOWING = "TREND_FOLLOWING"
    RANGE_REVERSAL = "RANGE_REVERSAL"
    PROBE_REVERSAL = "PROBE_REVERSAL"
    COMPRESSION_BREAKOUT = "COMPRESSION_BREAKOUT"
    DEFENSIVE = "DEFENSIVE"


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: Regime
    strategy_families: tuple[StrategyFamily, ...]
    risk_multiplier: float
    trend_score: float
    volatility_pct: float
    money_flow_regime: str = "NEUTRAL"
    notes: list[str] = field(default_factory=list)


def classify_regime(window: pd.DataFrame, mf_row: pd.Series | None = None) -> RegimeSnapshot:
    """Classify the current symbol regime from a causal OHLCV window."""
    if window.empty or len(window) < 30:
        return RegimeSnapshot(
            regime=Regime.SIDEWAY,
            strategy_families=(StrategyFamily.RANGE_REVERSAL,),
            risk_multiplier=0.5,
            trend_score=0.0,
            volatility_pct=0.0,
            notes=["insufficient_history"],
        )

    closes = window["close"].astype(float)
    highs = window["high"].astype(float)
    lows = window["low"].astype(float)
    close = float(closes.iloc[-1])
    ma20 = float(closes.tail(20).mean())
    ma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else ma20
    ma120 = float(closes.tail(120).mean()) if len(closes) >= 120 else ma50
    ma20_prev = float(closes.iloc[-25:-5].mean()) if len(closes) >= 25 else ma20

    atr_pct = _atr_pct(highs, lows, closes)
    trend_score = _trend_score(close, ma20, ma50, ma120, ma20_prev)

    mf_regime = str(mf_row.get("mf_regime", "NEUTRAL")) if mf_row is not None else "NEUTRAL"
    mf_score = float(mf_row.get("mf_score", 0) or 0) if mf_row is not None else 0.0
    recent_distribution = bool(mf_row.get("recent_distribution", False)) if mf_row is not None else False

    notes: list[str] = []
    if atr_pct >= 5.5:
        notes.append(f"high_vol_atr={atr_pct:.1f}%")
    if mf_regime in {"DISTRIBUTION", "MONEY_OUT", "EXHAUSTION_INFLOW"} or recent_distribution:
        notes.append(f"money_flow={mf_regime}")
        return RegimeSnapshot(
            regime=Regime.HIGH_VOL_DISTRIBUTION,
            strategy_families=(StrategyFamily.DEFENSIVE,),
            risk_multiplier=0.0,
            trend_score=trend_score,
            volatility_pct=atr_pct,
            money_flow_regime=mf_regime,
            notes=notes,
        )

    if trend_score >= 2.0 and atr_pct < 6.5:
        return RegimeSnapshot(
            regime=Regime.UPTREND,
            strategy_families=(
                StrategyFamily.TREND_FOLLOWING,
                StrategyFamily.COMPRESSION_BREAKOUT,
            ),
            risk_multiplier=1.0,
            trend_score=trend_score,
            volatility_pct=atr_pct,
            money_flow_regime=mf_regime,
            notes=notes,
        )

    if mf_regime in {"ACCUMULATION", "EARLY_MONEY_IN", "MONEY_IN", "BREAKOUT_FLOW"} and mf_score >= 2:
        return RegimeSnapshot(
            regime=Regime.ACCUMULATION,
            strategy_families=(
                StrategyFamily.PROBE_REVERSAL,
                StrategyFamily.RANGE_REVERSAL,
                StrategyFamily.COMPRESSION_BREAKOUT,
            ),
            risk_multiplier=0.6,
            trend_score=trend_score,
            volatility_pct=atr_pct,
            money_flow_regime=mf_regime,
            notes=[*notes, f"accumulation_flow={mf_regime}"],
        )

    if trend_score <= -2.0:
        return RegimeSnapshot(
            regime=Regime.DOWNTREND,
            strategy_families=(StrategyFamily.PROBE_REVERSAL,),
            risk_multiplier=0.33,
            trend_score=trend_score,
            volatility_pct=atr_pct,
            money_flow_regime=mf_regime,
            notes=notes,
        )

    return RegimeSnapshot(
        regime=Regime.SIDEWAY,
        strategy_families=(
            StrategyFamily.RANGE_REVERSAL,
            StrategyFamily.COMPRESSION_BREAKOUT,
        ),
        risk_multiplier=0.7,
        trend_score=trend_score,
        volatility_pct=atr_pct,
        money_flow_regime=mf_regime,
        notes=notes,
    )


def setup_family(setup_type: str) -> StrategyFamily:
    setup = (setup_type or "").upper()
    if setup in MOMENTUM_SETUPS or setup in PULLBACK_SETUPS:
        return StrategyFamily.TREND_FOLLOWING
    if setup in COMPRESSION_SETUPS:
        return StrategyFamily.COMPRESSION_BREAKOUT
    if setup in REVERSAL_SETUPS:
        return StrategyFamily.RANGE_REVERSAL
    return StrategyFamily.TREND_FOLLOWING


def is_setup_allowed_for_regime(setup_type: str, snapshot: RegimeSnapshot) -> bool:
    if StrategyFamily.DEFENSIVE in snapshot.strategy_families:
        return False

    setup = (setup_type or "").upper()
    family = setup_family(setup)
    if family in snapshot.strategy_families:
        return True

    # Uptrend shakeouts/pullbacks are valid entries, but later gates should size
    # them conservatively unless flow/location confirms.
    if snapshot.regime == Regime.UPTREND and setup in {
        "SPRING",
        "BULLISH_ENGULFING",
        "PIN_BAR",
        "RSI_BOUNCE",
        "HAMMER",
        "DOUBLE_BOTTOM",
        "MA_PULLBACK",
        "TREND_PULLBACK",
        "RETEST",
        "BREAKOUT_RETEST_ENTRY",
    }:
        return True

    # In downtrend, only small probe reversals are allowed.
    if snapshot.regime == Regime.DOWNTREND and setup in REVERSAL_SETUPS:
        return StrategyFamily.PROBE_REVERSAL in snapshot.strategy_families

    # Accumulation can use early trend-following only when flow is already strong.
    if snapshot.regime == Regime.ACCUMULATION and setup in {"RETEST", "BREAKOUT_RETEST_ENTRY"}:
        return True

    return False


def playbook_hint_for_signal(setup_type: str, snapshot: RegimeSnapshot) -> str:
    """Return a coarse playbook hint for routing and audit."""
    setup = (setup_type or "").upper()
    if snapshot.regime == Regime.HIGH_VOL_DISTRIBUTION:
        return "defensive_exit"
    if snapshot.regime == Regime.UPTREND:
        if setup == "CORE_TREND":
            return "core_trend"
        if setup in REVERSAL_SETUPS:
            return "range_reversal"
        return "trend_following"
    if snapshot.regime == Regime.SIDEWAY:
        return "range_reversal"
    if snapshot.regime == Regime.DOWNTREND:
        return "probe_only_bear"
    if snapshot.regime in {Regime.DOWNTREND, Regime.ACCUMULATION} and setup in REVERSAL_SETUPS:
        return "scale_in_reversal"
    if snapshot.regime == Regime.ACCUMULATION:
        return "scale_in_reversal" if setup in REVERSAL_SETUPS else "range_reversal"
    if setup in REVERSAL_SETUPS and StrategyFamily.RANGE_REVERSAL in snapshot.strategy_families:
        return "range_reversal"
    return "single_entry"


def _trend_score(close: float, ma20: float, ma50: float, ma120: float, ma20_prev: float) -> float:
    score = 0.0
    if close > ma20:
        score += 1.0
    else:
        score -= 1.0
    if ma20 > ma50:
        score += 1.0
    else:
        score -= 1.0
    if ma50 > ma120:
        score += 1.0
    else:
        score -= 1.0
    if ma20 > ma20_prev * 1.005:
        score += 1.0
    elif ma20 < ma20_prev * 0.995:
        score -= 1.0
    return score


def _atr_pct(highs: pd.Series, lows: pd.Series, closes: pd.Series) -> float:
    if len(closes) < 15:
        return 0.0
    h = highs.tail(14).to_numpy(dtype=float)
    l = lows.tail(14).to_numpy(dtype=float)
    prev_c = closes.shift(1).tail(14).to_numpy(dtype=float)
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    close = float(closes.iloc[-1])
    if close <= 0:
        return 0.0
    return round(float(np.nanmean(tr)) / close * 100.0, 2)
