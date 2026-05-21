"""Phase 3a triage gate — bounded decisions over a frozen StrategySignal.

The gate may only choose from 4 actions:
    APPROVE          — pass through unchanged
    REDUCE_SIZE      — multiply position_pct by size_multiplier ∈ [0.3, 1.0]
    SKIP             — do not enter this signal
    WAIT_FOR_ENTRY   — pass on this session, re-evaluate next day

Implementation philosophy:
    * The 8 ReasonCodes are enumerated. Each is implemented as a deterministic
      rule against EvidencePacket. The LLM wrapper is optional and falls back
      to APPROVE on any parse failure (fail-safe).
    * No new alpha. The gate can only DOWNGRADE — never APPROVE something
      core3 rejected, never increase size, never invent reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from multiagents_trading_assistant.agentic.schemas import (
    AgenticReview,
    EvidencePacket,
    ReasonCode,
    StrategySignal,
)


# ── Rule thresholds (tunable) ───────────────────────────────────────────────

@dataclass(frozen=True)
class GateThresholds:
    news_sentiment_min: float = 30.0          # NEWS_RISK if sentiment < this AND negative items exist
    volume_ratio_min: float = 0.5             # BAD_LIQUIDITY if volume_ratio_20 < this
    avg_value_min_bn: float = 10.0            # BAD_LIQUIDITY if avg_value_20 < 10B VND
    market_risk_off_vni_pct: float = -1.5     # MARKET_RISK_OFF if VNI down ≥ this %
    overextended_gap_pct: float = 5.0         # OVEREXTENDED_GAP if intraday gap ≥ this %
    sector_rs_min: float = 30.0               # SECTOR_WEAK if rs_percentile_20 < this
    sector_leadership_min: float = 30.0       # SECTOR_WEAK if sector_leadership_score < this
    missing_fields_max: int = 3               # DATA_QUALITY_ISSUE if ≥ this many fields missing
    recent_same_symbol_days: int = 2          # RECENT_SAME_SYMBOL window (T+2.5)


DEFAULT_THRESHOLDS = GateThresholds()


# ── Rule-based gate ─────────────────────────────────────────────────────────

class RuleBasedTriageGate:
    """Deterministic gate. The reference implementation for Phase 3a."""

    def __init__(self, thresholds: GateThresholds | None = None) -> None:
        self.t = thresholds or DEFAULT_THRESHOLDS

    def review(
        self,
        signal: StrategySignal,
        evidence: EvidencePacket,
    ) -> AgenticReview:
        reasons: list[ReasonCode] = []
        rationale_parts: list[str] = []

        for check, code, label in self._checks(evidence):
            triggered, note = check()
            if triggered:
                reasons.append(code)
                rationale_parts.append(f"{label}: {note}")

        action, size_mult, wait = self._decide(reasons)

        return AgenticReview(
            symbol=signal.symbol,
            as_of_date=evidence.as_of_date,
            action=action,
            reason_codes=reasons,
            size_multiplier=size_mult,
            wait_condition=wait,
            confidence=1.0,
            rationale=" | ".join(rationale_parts)[:200],
        )

    # ---------------------------------------------------------------- checks

    def _checks(self, ev: EvidencePacket) -> list[tuple[Any, ReasonCode, str]]:
        return [
            (lambda: self._check_news(ev), ReasonCode.NEWS_RISK, "news"),
            (lambda: self._check_liquidity(ev), ReasonCode.BAD_LIQUIDITY, "liq"),
            (lambda: self._check_market_risk(ev), ReasonCode.MARKET_RISK_OFF, "mkt"),
            (lambda: self._check_overextended(ev), ReasonCode.OVEREXTENDED_GAP, "gap"),
            (lambda: self._check_sector(ev), ReasonCode.SECTOR_WEAK, "sector"),
            (lambda: self._check_data_quality(ev), ReasonCode.DATA_QUALITY_ISSUE, "dq"),
            (lambda: self._check_portfolio(ev), ReasonCode.PORTFOLIO_CONCENTRATION, "port"),
            (lambda: self._check_recent_symbol(ev), ReasonCode.RECENT_SAME_SYMBOL, "t2"),
        ]

    def _check_news(self, ev: EvidencePacket) -> tuple[bool, str]:
        news = ev.news or {}
        if not news:
            return False, ""
        key_neg = news.get("key_negative") or []
        if not key_neg:
            return False, ""
        # Sentiment score might live under different keys depending on source
        score = self._first_num(news, "sentiment_score", "avg_sentiment")
        if score is not None and score < self.t.news_sentiment_min:
            return True, f"neg_news={len(key_neg)}, sent={score:.0f}"
        if len(key_neg) >= 2:
            return True, f"neg_news={len(key_neg)}"
        return False, ""

    def _check_liquidity(self, ev: EvidencePacket) -> tuple[bool, str]:
        tech = ev.technical or {}
        indicators = tech.get("indicators") if isinstance(tech, dict) else {}
        indicators = indicators or {}
        vol_ratio = self._first_num(indicators, "volume_ratio_20")
        avg_value = self._first_num(indicators, "avg_value_20", "value_ma20")
        if vol_ratio is not None and vol_ratio < self.t.volume_ratio_min:
            return True, f"vol_ratio={vol_ratio:.2f}"
        if avg_value is not None and avg_value < self.t.avg_value_min_bn * 1e9:
            return True, f"avg_value={avg_value / 1e9:.1f}B"
        return False, ""

    def _check_market_risk(self, ev: EvidencePacket) -> tuple[bool, str]:
        market = ev.market or {}
        ctx = market.get("context") or {}
        ref_trend = str(ctx.get("reference_trend") or ctx.get("trend") or "").upper()
        vni_chg = self._first_num(ctx, "vni_change_pct")
        if ref_trend == "DOWNTREND" and vni_chg is not None and vni_chg <= self.t.market_risk_off_vni_pct:
            return True, f"trend={ref_trend}, vni={vni_chg:+.2f}%"
        # Money cycle market regime
        mc = market.get("money_cycle_market") or {}
        regime = str(mc.get("mkt_regime") or "").upper()
        if regime == "RISK_OFF":
            return True, "mkt_regime=RISK_OFF"
        return False, ""

    def _check_overextended(self, ev: EvidencePacket) -> tuple[bool, str]:
        tech = ev.technical or {}
        indicators = tech.get("indicators") if isinstance(tech, dict) else {}
        indicators = indicators or {}
        # stock_day_change_pct is already in percent units (e.g. 5.0 = +5%).
        # ret_1d / ret_1d_pct (if present) is fractional.
        day_chg = self._first_num(indicators, "stock_day_change_pct")
        if day_chg is None:
            frac = self._first_num(indicators, "ret_1d_pct", "ret_1d")
            day_chg = frac * 100.0 if frac is not None else None
        if day_chg is None:
            return False, ""
        if day_chg >= self.t.overextended_gap_pct:
            return True, f"day_chg={day_chg:+.2f}%"
        return False, ""

    def _check_sector(self, ev: EvidencePacket) -> tuple[bool, str]:
        sector = ev.sector or {}
        rs = self._first_num(sector, "rs_percentile_20")
        # rs_percentile_20 may be on [0, 1] or [0, 100]
        if rs is not None and rs < 1.0:
            rs = rs * 100.0
        leadership = self._first_num(sector, "sector_leadership_score")
        if rs is not None and rs < self.t.sector_rs_min:
            return True, f"rs={rs:.0f}"
        if leadership is not None and leadership < self.t.sector_leadership_min:
            return True, f"sector_lead={leadership:.0f}"
        return False, ""

    def _check_data_quality(self, ev: EvidencePacket) -> tuple[bool, str]:
        dq = ev.data_quality or {}
        missing = dq.get("missing_fields") or []
        if len(missing) >= self.t.missing_fields_max:
            return True, f"missing={len(missing)}"
        if dq.get("data_quality_ok") is False:
            return True, "dq_flag"
        return False, ""

    def _check_portfolio(self, ev: EvidencePacket) -> tuple[bool, str]:
        port = ev.portfolio or {}
        if port.get("has_position") is True:
            return True, "already_holding"
        return False, ""

    def _check_recent_symbol(self, ev: EvidencePacket) -> tuple[bool, str]:
        port = ev.portfolio or {}
        # Phase 2 portfolio packet doesn't include recent_decisions yet — keep
        # this hook for when it's added. For now we rely on has_position.
        recent = port.get("recent_buy_within_days")
        if isinstance(recent, (int, float)) and recent <= self.t.recent_same_symbol_days:
            return True, f"t+{int(recent)}"
        return False, ""

    # --------------------------------------------------------------- decide

    @staticmethod
    def _decide(reasons: list[ReasonCode]) -> tuple[str, float | None, str | None]:
        """Map reason set → bounded action."""
        if not reasons:
            return "APPROVE", None, None

        # Hard SKIP — no amount of size reduction makes these acceptable
        hard_skip = {
            ReasonCode.NEWS_RISK,
            ReasonCode.MARKET_RISK_OFF,
            ReasonCode.DATA_QUALITY_ISSUE,
            ReasonCode.RECENT_SAME_SYMBOL,
        }
        if any(r in hard_skip for r in reasons):
            return "SKIP", None, None

        # WAIT_FOR_ENTRY — entry timing is the issue, signal itself is fine
        if ReasonCode.OVEREXTENDED_GAP in reasons:
            return "WAIT_FOR_ENTRY", None, "wait for pullback to setup level or next session"

        # REDUCE_SIZE — quality concerns but not deal-breakers
        # Stack reductions: each concern shaves 25% off, floor at 0.3
        mult = 1.0
        for r in reasons:
            if r in (ReasonCode.SECTOR_WEAK, ReasonCode.BAD_LIQUIDITY,
                     ReasonCode.PORTFOLIO_CONCENTRATION):
                mult *= 0.75
        mult = max(0.3, round(mult, 2))
        if mult >= 0.95:
            return "APPROVE", None, None
        return "REDUCE_SIZE", mult, None

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _first_num(d: dict, *keys: str) -> float | None:
        for k in keys:
            v = d.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv:  # NaN check
                continue
            return fv
        return None


# ── Application helper ─────────────────────────────────────────────────────

def apply_review_to_position_pct(
    base_position_pct: float,
    review: AgenticReview,
) -> float:
    """Return the post-gate position_pct. 0.0 means do not enter."""
    if review.action == "APPROVE":
        return base_position_pct
    if review.action == "REDUCE_SIZE" and review.size_multiplier:
        return base_position_pct * review.size_multiplier
    # SKIP or WAIT_FOR_ENTRY → don't enter this session
    return 0.0
