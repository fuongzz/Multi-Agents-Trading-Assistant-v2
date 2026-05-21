"""Tests for Phase 3a triage gate."""

from __future__ import annotations

import pytest

from multiagents_trading_assistant.agentic.schemas import (
    AgenticReview,
    EvidencePacket,
    ReasonCode,
    StrategySignal,
)
from multiagents_trading_assistant.agentic.triage_gate import (
    RuleBasedTriageGate,
    apply_review_to_position_pct,
)


def _signal(symbol: str = "VCB") -> StrategySignal:
    return StrategySignal(
        symbol=symbol,
        signal_date="2024-06-15",
        as_of_date="2024-06-15",
        strategy_name="breakout_after_accumulation_v3",
    )


def _evidence(**sections) -> EvidencePacket:
    base = dict(
        symbol="VCB",
        as_of_date="2024-06-15",
        market={"context": {"trend": "UPTREND", "reference_trend": "UPTREND", "vni_change_pct": 0.5}},
        sector={"rs_percentile_20": 75.0, "sector_leadership_score": 60.0},
        technical={"indicators": {"volume_ratio_20": 1.5, "avg_value_20": 50e9, "stock_day_change_pct": 1.0}},
        money_flow={},
        edge={"passed": True},
        fundamental={},
        news={},
        portfolio={"has_position": False},
        data_quality={"missing_fields": [], "data_quality_ok": True},
    )
    for k, v in sections.items():
        base[k] = v
    return EvidencePacket(**base)


def test_approve_clean_signal():
    review = RuleBasedTriageGate().review(_signal(), _evidence())
    assert review.action == "APPROVE"
    assert review.reason_codes == []
    assert review.size_multiplier is None


def test_news_risk_triggers_skip():
    news = {"key_negative": ["bad headline 1", "bad headline 2"], "sentiment_score": 20.0}
    review = RuleBasedTriageGate().review(_signal(), _evidence(news=news))
    assert review.action == "SKIP"
    assert ReasonCode.NEWS_RISK in review.reason_codes


def test_market_risk_off_skip():
    market = {"context": {"trend": "DOWNTREND", "reference_trend": "DOWNTREND", "vni_change_pct": -2.0}}
    review = RuleBasedTriageGate().review(_signal(), _evidence(market=market))
    assert review.action == "SKIP"
    assert ReasonCode.MARKET_RISK_OFF in review.reason_codes


def test_overextended_gap_waits():
    tech = {"indicators": {"volume_ratio_20": 1.5, "avg_value_20": 50e9, "stock_day_change_pct": 6.0}}
    review = RuleBasedTriageGate().review(_signal(), _evidence(technical=tech))
    assert review.action == "WAIT_FOR_ENTRY"
    assert ReasonCode.OVEREXTENDED_GAP in review.reason_codes
    assert review.wait_condition


def test_sector_weak_reduces_size():
    sector = {"rs_percentile_20": 15.0, "sector_leadership_score": 20.0}
    review = RuleBasedTriageGate().review(_signal(), _evidence(sector=sector))
    assert review.action == "REDUCE_SIZE"
    assert ReasonCode.SECTOR_WEAK in review.reason_codes
    assert review.size_multiplier == 0.75


def test_low_liquidity_reduces_size():
    tech = {"indicators": {"volume_ratio_20": 0.3, "avg_value_20": 50e9, "stock_day_change_pct": 0.5}}
    review = RuleBasedTriageGate().review(_signal(), _evidence(technical=tech))
    assert review.action == "REDUCE_SIZE"
    assert ReasonCode.BAD_LIQUIDITY in review.reason_codes


def test_portfolio_holds_already():
    port = {"has_position": True}
    review = RuleBasedTriageGate().review(_signal(), _evidence(portfolio=port))
    assert review.action == "REDUCE_SIZE"
    assert ReasonCode.PORTFOLIO_CONCENTRATION in review.reason_codes


def test_data_quality_skips():
    dq = {"missing_fields": ["news", "fundamental", "portfolio"], "data_quality_ok": True}
    review = RuleBasedTriageGate().review(_signal(), _evidence(data_quality=dq))
    assert review.action == "SKIP"
    assert ReasonCode.DATA_QUALITY_ISSUE in review.reason_codes


def test_stacked_quality_concerns_floor_at_30pct():
    tech = {"indicators": {"volume_ratio_20": 0.3, "avg_value_20": 50e9, "stock_day_change_pct": 0.5}}
    sector = {"rs_percentile_20": 10.0, "sector_leadership_score": 15.0}
    port = {"has_position": True}
    review = RuleBasedTriageGate().review(
        _signal(), _evidence(technical=tech, sector=sector, portfolio=port)
    )
    # 0.75^3 = 0.42, clamped within [0.3, 1.0]
    assert review.action == "REDUCE_SIZE"
    assert review.size_multiplier is not None
    assert 0.3 <= review.size_multiplier <= 0.5


def test_apply_review_to_position_pct():
    assert apply_review_to_position_pct(0.05, AgenticReview(
        symbol="X", as_of_date="2024-06-15", action="APPROVE",
    )) == 0.05
    skip = AgenticReview(
        symbol="X", as_of_date="2024-06-15", action="SKIP",
        reason_codes=[ReasonCode.NEWS_RISK],
    )
    assert apply_review_to_position_pct(0.05, skip) == 0.0
    reduce = AgenticReview(
        symbol="X", as_of_date="2024-06-15", action="REDUCE_SIZE",
        reason_codes=[ReasonCode.SECTOR_WEAK], size_multiplier=0.5,
    )
    assert apply_review_to_position_pct(0.05, reduce) == 0.025


def test_schema_invariants():
    with pytest.raises(Exception):
        AgenticReview(symbol="X", as_of_date="2024-06-15", action="REDUCE_SIZE",
                      reason_codes=[ReasonCode.SECTOR_WEAK])  # missing size_multiplier
    with pytest.raises(Exception):
        AgenticReview(symbol="X", as_of_date="2024-06-15", action="WAIT_FOR_ENTRY",
                      reason_codes=[ReasonCode.OVEREXTENDED_GAP])  # missing wait_condition
    with pytest.raises(Exception):
        AgenticReview(symbol="X", as_of_date="2024-06-15", action="SKIP")  # missing reason_codes


def test_playbook_lookup():
    from multiagents_trading_assistant.agentic.playbook import get_playbook, PLAYBOOKS
    pb = get_playbook("EDGE_BREAKOUT", "UPTREND")
    assert pb.trail_atr_mult == 2.8
    pb_sw = get_playbook("EDGE_BREAKOUT", "SIDEWAY")
    assert pb_sw.trail_atr_mult == 2.2
    # Fallback to default
    pb_unknown = get_playbook("UNKNOWN_SETUP", "UPTREND")
    assert pb_unknown.note.startswith("Fallback")
