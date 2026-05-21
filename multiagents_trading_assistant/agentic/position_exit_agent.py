"""Deterministic position-state exit reviewer.

This is the first safe version of a "should we take profit or let it run?"
agent. It is intentionally rule-based and bounded so it can be backtested.

Anti-lookahead contract:
- The reviewer receives only bars with ``date <= as_of_date``.
- Any EOD decision emitted at ``as_of_date`` is actionable from the next bar.
- Intraday TP/SL trigger prices are not modified using same-bar future data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


ExitAction = Literal["HOLD", "HOLD_RUNNER", "TAKE_PROFIT_NEXT_OPEN", "RAISE_TRAILING_STOP"]


@dataclass(frozen=True)
class PositionState:
    symbol: str
    entry_date: str
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    shares: int = 0
    highest_price: float | None = None
    setup_type: str = "EDGE"
    strategy_name: str = ""
    holding_bars: int = 0


@dataclass(frozen=True)
class PositionExitReview:
    symbol: str
    as_of_date: str
    action: ExitAction
    confidence: float
    reason_codes: list[str] = field(default_factory=list)
    suggested_stop_loss: float | None = None
    next_bar_actionable: bool = True
    rationale: str = ""


@dataclass(frozen=True)
class PositionExitAgentConfig:
    near_tp_ratio: float = 0.92
    min_regime_score_to_act: float = 55.0
    strong_regime_min: float = 55.0
    strong_rs_min: float = 0.60
    strong_volume_ratio_min: float = 1.05
    strong_close_location_min: float = 0.62
    weak_close_location_max: float = 0.45
    weak_upper_wick_min: float = 0.08
    trail_atr_mult: float = 2.6
    breakeven_profit_pct: float = 0.04
    min_profit_raise_stop_pct: float = 0.08
    min_profit_pullback_raise_stop_pct: float = 0.12
    runner_breakout_only: bool = True


class PositionStateExitAgent:
    """Bounded EOD review for profitable positions.

    The agent is not an LLM. It is a deterministic policy that can later be
    used as a tool or guardrail by an LLM reviewer. Keeping it deterministic is
    important because exit improvements must be backtestable.
    """

    def __init__(self, config: PositionExitAgentConfig | None = None) -> None:
        self.config = config or PositionExitAgentConfig()

    def review(
        self,
        position: PositionState,
        history: pd.DataFrame,
        *,
        as_of_date: str | pd.Timestamp,
    ) -> PositionExitReview:
        as_of = pd.Timestamp(as_of_date).normalize()
        frame = _prepare_history(history, as_of)
        if frame.empty:
            return PositionExitReview(
                symbol=position.symbol,
                as_of_date=as_of.date().isoformat(),
                action="HOLD",
                confidence=0.0,
                reason_codes=["NO_HISTORY"],
                rationale="No causal history available for review.",
            )

        row = frame.iloc[-1]
        close = _safe_float(row.get("close"))
        if close is None or close <= 0 or position.entry_price <= 0:
            return PositionExitReview(
                symbol=position.symbol,
                as_of_date=as_of.date().isoformat(),
                action="HOLD",
                confidence=0.0,
                reason_codes=["BAD_PRICE"],
                rationale="Invalid price data.",
            )

        mkt_score = _safe_float(row.get("mkt_regime_score"))
        if mkt_score is not None and mkt_score < self.config.min_regime_score_to_act:
            return PositionExitReview(
                symbol=position.symbol,
                as_of_date=as_of.date().isoformat(),
                action="HOLD",
                confidence=0.60,
                reason_codes=["REGIME_TOO_WEAK_FOR_EXIT_AGENT"],
                rationale="Exit agent is disabled in weak regimes to avoid over-protecting choppy recoveries.",
            )

        tp_distance = max(position.take_profit_price - position.entry_price, 0.0)
        progress_to_tp = (
            (close - position.entry_price) / tp_distance
            if tp_distance > 0
            else 0.0
        )
        unrealized_pct = close / position.entry_price - 1.0
        at_or_near_tp = close >= position.take_profit_price or progress_to_tp >= self.config.near_tp_ratio

        if not at_or_near_tp:
            return self._raise_stop_if_needed(position, row, as_of, unrealized_pct, ["NOT_NEAR_TP"])

        strength_score, strength_reasons = self._strength_score(row)
        weakness_score, weakness_reasons = self._weakness_score(row)

        can_run = (not self.config.runner_breakout_only) or _is_breakout_setup(position)
        if can_run and strength_score >= 3 and weakness_score == 0:
            stop = self._runner_stop(position, row, close)
            return PositionExitReview(
                symbol=position.symbol,
                as_of_date=as_of.date().isoformat(),
                action="HOLD_RUNNER",
                confidence=min(1.0, 0.55 + 0.1 * strength_score),
                reason_codes=["NEAR_TP", *strength_reasons],
                suggested_stop_loss=stop,
                rationale="Near target but trend/flow state is strong; hold runner from next bar with raised stop.",
            )

        if weakness_score >= 2:
            return PositionExitReview(
                symbol=position.symbol,
                as_of_date=as_of.date().isoformat(),
                action="TAKE_PROFIT_NEXT_OPEN",
                confidence=min(1.0, 0.55 + 0.15 * weakness_score),
                reason_codes=["NEAR_TP", *weakness_reasons],
                rationale="Near target and EOD state shows exhaustion; take profit at next open.",
            )

        stop = self._runner_stop(position, row, close)
        return PositionExitReview(
            symbol=position.symbol,
            as_of_date=as_of.date().isoformat(),
            action="RAISE_TRAILING_STOP",
            confidence=0.60,
            reason_codes=["NEAR_TP", "MIXED_STATE"],
            suggested_stop_loss=stop,
            rationale="Near target with mixed state; keep position but raise trailing stop from next bar.",
        )

    def _strength_score(self, row: pd.Series) -> tuple[int, list[str]]:
        cfg = self.config
        reasons: list[str] = []
        checks = [
            ("MARKET_SUPPORT", (_safe_float(row.get("mkt_regime_score")) or 0.0) >= cfg.strong_regime_min),
            ("RELATIVE_STRENGTH", (_safe_float(row.get("rs_percentile_20")) or 0.0) >= cfg.strong_rs_min),
            ("VALUE_EXPANSION", (_safe_float(row.get("value_ratio_20")) or 0.0) >= cfg.strong_volume_ratio_min),
            ("STRONG_CLOSE", (_safe_float(row.get("close_location")) or 0.0) >= cfg.strong_close_location_min),
            ("ABOVE_MA20", bool(row.get("above_ma20", False))),
            ("ABOVE_MA50", bool(row.get("above_ma50", False))),
        ]
        for code, passed in checks:
            if passed:
                reasons.append(code)
        return len(reasons), reasons

    def _weakness_score(self, row: pd.Series) -> tuple[int, list[str]]:
        cfg = self.config
        reasons: list[str] = []
        close_location = _safe_float(row.get("close_location"))
        upper_wick = _safe_float(row.get("upper_wick_pct"))
        if close_location is not None and close_location <= cfg.weak_close_location_max:
            reasons.append("WEAK_CLOSE")
        if upper_wick is not None and upper_wick >= cfg.weak_upper_wick_min:
            reasons.append("SUPPLY_WICK")
        if bool(row.get("above_ma20", True)) is False:
            reasons.append("LOST_MA20")
        if (_safe_float(row.get("mkt_regime_score")) or 100.0) < 45:
            reasons.append("MARKET_WEAK")
        if (_safe_float(row.get("distribution_days_10")) or 0.0) >= 2:
            reasons.append("DISTRIBUTION")
        return len(reasons), reasons

    def _raise_stop_if_needed(
        self,
        position: PositionState,
        row: pd.Series,
        as_of: pd.Timestamp,
        unrealized_pct: float,
        reasons: list[str],
    ) -> PositionExitReview:
        min_profit = (
            self.config.min_profit_raise_stop_pct
            if _is_breakout_setup(position)
            else self.config.min_profit_pullback_raise_stop_pct
        )
        if unrealized_pct < min_profit:
            return PositionExitReview(
                symbol=position.symbol,
                as_of_date=as_of.date().isoformat(),
                action="HOLD",
                confidence=0.55,
                reason_codes=reasons,
                rationale="Position is not near target and has insufficient profit for v2 stop protection.",
            )
        close = _safe_float(row.get("close")) or position.entry_price
        stop = self._runner_stop(position, row, close)
        if stop is None or stop <= position.stop_loss_price:
            return PositionExitReview(
                symbol=position.symbol,
                as_of_date=as_of.date().isoformat(),
                action="HOLD",
                confidence=0.55,
                reason_codes=reasons,
                rationale="Position is not near target; existing stop is adequate.",
            )
        return PositionExitReview(
            symbol=position.symbol,
            as_of_date=as_of.date().isoformat(),
            action="RAISE_TRAILING_STOP",
            confidence=0.65,
            reason_codes=[*reasons, "PROTECT_PROFIT"],
            suggested_stop_loss=stop,
            rationale="Position is profitable; raise stop from next bar without forcing TP.",
        )

    def _runner_stop(self, position: PositionState, row: pd.Series, close: float) -> float | None:
        atr = _safe_float(row.get("atr14")) or _safe_float(row.get("atr"))
        candidates = [position.stop_loss_price]
        if atr and atr > 0:
            candidates.append(close - self.config.trail_atr_mult * atr)
        if close >= position.entry_price * (1.0 + self.config.breakeven_profit_pct):
            candidates.append(position.entry_price)
        stop = max(candidates)
        if stop <= 0:
            return None
        return round(float(stop), 4)


def _prepare_history(history: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    frame = history.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        frame = frame[frame["date"] <= as_of].sort_values("date")
    elif isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.loc[frame.index.tz_localize(None).normalize() <= as_of].sort_index()
    else:
        raise ValueError("history must have a `date` column or DatetimeIndex")
    return frame


def _safe_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _is_breakout_setup(position: PositionState) -> bool:
    haystack = f"{position.setup_type} {position.strategy_name}".upper()
    return "BREAKOUT" in haystack or "55" in haystack
