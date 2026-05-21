"""Deterministic add-on reviewer for winning positions.

The agent decides whether an existing winning position deserves more capital.
It is deliberately rule-based so it can be backtested and audited before any
LLM layer is allowed to use it.

Anti-lookahead contract:
- The reviewer receives only bars with ``date <= as_of_date``.
- Any EOD decision at ``as_of_date`` can only be filled from the next bar.
- Breakout/base checks use prior-bar ranges where needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


AddOnAction = Literal["HOLD", "ADD_ON_NEXT_OPEN"]


@dataclass(frozen=True)
class PositionAddOnState:
    symbol: str
    entry_date: str
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    shares: int
    add_on_count: int = 0
    highest_price: float | None = None
    setup_type: str = "EDGE"
    strategy_name: str = ""
    holding_bars: int = 0


@dataclass(frozen=True)
class PositionAddOnReview:
    symbol: str
    as_of_date: str
    action: AddOnAction
    confidence: float
    reason_codes: list[str] = field(default_factory=list)
    suggested_stop_loss: float | None = None
    budget_fraction_of_slot: float = 0.0
    next_bar_actionable: bool = True
    rationale: str = ""


@dataclass(frozen=True)
class PositionAddOnAgentConfig:
    max_add_ons_per_position: int = 1
    min_holding_bars: int = 5
    min_unrealized_pct: float = 0.08
    min_regime_score: float = 58.0
    min_rs_percentile: float = 0.62
    min_value_ratio: float = 1.05
    min_close_location: float = 0.62
    max_extension_from_ma20: float = 0.14
    max_upper_wick_pct: float = 0.10
    base_lookback: int = 15
    min_base_tightness: float = 0.16
    breakout_buffer: float = 0.0
    trail_atr_mult: float = 2.5
    add_on_budget_fraction_of_slot: float = 0.50


class PositionAddOnAgent:
    """Conservative pyramiding policy for confirmed winners."""

    def __init__(self, config: PositionAddOnAgentConfig | None = None) -> None:
        self.config = config or PositionAddOnAgentConfig()

    def review(
        self,
        position: PositionAddOnState,
        history: pd.DataFrame,
        *,
        as_of_date: str | pd.Timestamp,
    ) -> PositionAddOnReview:
        cfg = self.config
        as_of = pd.Timestamp(as_of_date).normalize()
        frame = _prepare_history(history, as_of)
        if frame.empty:
            return _hold(position, as_of, ["NO_HISTORY"], "No causal history available.")

        row = frame.iloc[-1]
        close = _safe_float(row.get("close"))
        if close is None or close <= 0 or position.entry_price <= 0:
            return _hold(position, as_of, ["BAD_PRICE"], "Invalid price data.")

        if position.add_on_count >= cfg.max_add_ons_per_position:
            return _hold(position, as_of, ["ADD_ON_LIMIT_REACHED"], "Position already reached add-on limit.")
        if position.holding_bars < cfg.min_holding_bars:
            return _hold(position, as_of, ["TOO_EARLY"], "Position has not proven itself long enough.")

        unrealized_pct = close / position.entry_price - 1.0
        if unrealized_pct < cfg.min_unrealized_pct:
            return _hold(position, as_of, ["INSUFFICIENT_PROFIT"], "Only add to confirmed winners.")

        checks, failed = self._quality_checks(row, close)
        if failed:
            return _hold(position, as_of, failed, "Uptrend/add-on quality checks failed.")

        base_ok, base_reasons = self._base_or_rebreakout_check(frame)
        if not base_ok:
            return _hold(position, as_of, base_reasons, "No causal re-breakout/base trigger.")

        stop = self._suggested_stop(position, row, close)
        return PositionAddOnReview(
            symbol=position.symbol,
            as_of_date=as_of.date().isoformat(),
            action="ADD_ON_NEXT_OPEN",
            confidence=min(0.95, 0.55 + 0.05 * len(checks) + 0.05 * len(base_reasons)),
            reason_codes=[*checks, *base_reasons, "CONFIRMED_WINNER"],
            suggested_stop_loss=stop,
            budget_fraction_of_slot=cfg.add_on_budget_fraction_of_slot,
            rationale="Winning position remains strong in uptrend and triggered a causal re-breakout/base add-on.",
        )

    def _quality_checks(self, row: pd.Series, close: float) -> tuple[list[str], list[str]]:
        cfg = self.config
        passed: list[str] = []
        failed: list[str] = []

        mkt_score = _safe_float(row.get("mkt_regime_score"))
        if mkt_score is not None and mkt_score >= cfg.min_regime_score:
            passed.append("MARKET_REGIME_STRONG")
        else:
            failed.append("MARKET_NOT_UPTREND")

        if bool(row.get("above_ma20", False)) and bool(row.get("above_ma50", False)):
            passed.append("PRICE_ABOVE_MA20_MA50")
        else:
            failed.append("PRICE_NOT_ABOVE_KEY_MA")

        rs = _safe_float(row.get("rs_percentile_20"))
        if rs is not None and rs >= cfg.min_rs_percentile:
            passed.append("RELATIVE_STRENGTH")
        else:
            failed.append("WEAK_RELATIVE_STRENGTH")

        value_ratio = _safe_float(row.get("value_ratio_20"))
        if value_ratio is not None and value_ratio >= cfg.min_value_ratio:
            passed.append("VALUE_EXPANSION")
        else:
            failed.append("NO_VALUE_EXPANSION")

        close_location = _safe_float(row.get("close_location"))
        if close_location is not None and close_location >= cfg.min_close_location:
            passed.append("STRONG_CLOSE")
        else:
            failed.append("WEAK_CLOSE_LOCATION")

        upper_wick = _safe_float(row.get("upper_wick_pct"))
        if upper_wick is None or upper_wick <= cfg.max_upper_wick_pct:
            passed.append("NO_SUPPLY_WICK")
        else:
            failed.append("SUPPLY_WICK")

        ma20 = _safe_float(row.get("ma20"))
        if ma20 and ma20 > 0 and close / ma20 - 1.0 <= cfg.max_extension_from_ma20:
            passed.append("NOT_EXTENDED_FROM_MA20")
        elif ma20 is not None:
            failed.append("EXTENDED_FROM_MA20")

        return passed, failed

    def _base_or_rebreakout_check(self, frame: pd.DataFrame) -> tuple[bool, list[str]]:
        cfg = self.config
        if len(frame) < cfg.base_lookback + 2:
            return False, ["INSUFFICIENT_BASE_HISTORY"]

        current = frame.iloc[-1]
        prev = frame.iloc[-cfg.base_lookback - 1 : -1]
        close = _safe_float(current.get("close"))
        high = _safe_float(current.get("high")) or close
        if close is None or high is None:
            return False, ["BAD_BREAKOUT_PRICE"]

        prior_high = float(prev["high"].max())
        prior_low = float(prev["low"].min())
        tightness = (prior_high - prior_low) / prior_low if prior_low > 0 else 1.0
        reasons: list[str] = []
        if tightness <= cfg.min_base_tightness:
            reasons.append("TIGHT_BASE")
        else:
            return False, ["BASE_TOO_WIDE"]

        if close >= prior_high * (1.0 + cfg.breakout_buffer):
            reasons.append("RE_BREAKOUT_CLOSE")
            return True, reasons
        if high >= prior_high * (1.0 + cfg.breakout_buffer) and close >= prior_high * 0.99:
            reasons.append("RE_BREAKOUT_INTRADAY_HOLD")
            return True, reasons
        return False, ["NO_RE_BREAKOUT"]

    def _suggested_stop(self, position: PositionAddOnState, row: pd.Series, close: float) -> float | None:
        atr = _safe_float(row.get("atr14")) or _safe_float(row.get("atr"))
        candidates = [position.stop_loss_price, position.entry_price]
        if atr and atr > 0:
            candidates.append(close - self.config.trail_atr_mult * atr)
        ma20 = _safe_float(row.get("ma20"))
        if ma20 and ma20 > 0:
            candidates.append(ma20 * 0.98)
        stop = max(candidates)
        return round(float(stop), 4) if stop > 0 else None


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


def _hold(
    position: PositionAddOnState,
    as_of: pd.Timestamp,
    reasons: list[str],
    rationale: str,
) -> PositionAddOnReview:
    return PositionAddOnReview(
        symbol=position.symbol,
        as_of_date=as_of.date().isoformat(),
        action="HOLD",
        confidence=0.55,
        reason_codes=reasons,
        rationale=rationale,
    )
