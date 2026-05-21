from dataclasses import dataclass, field

from multiagents_trading_assistant.agentic import (
    build_evidence_packet_from_candidate,
    build_strategy_signal_from_candidate,
    format_agentic_context,
)


@dataclass
class _Market:
    reference_trend: str = "UPTREND"
    vni_change_pct: float = 0.5


@dataclass
class _Candidate:
    symbol: str
    indicators: dict
    priority_score: float = 88.0
    setup_type: str = "EDGE_BREAKOUT"
    market_context: _Market = field(default_factory=_Market)
    reasons: list[str] = field(default_factory=lambda: ["core3 passed"])


def test_strategy_signal_requires_edge_passed_by_default():
    candidate = _Candidate(
        symbol="VCB",
        indicators={"edge_strategy_analysis": {"passed": False}},
    )

    assert build_strategy_signal_from_candidate(candidate, "2026-05-13") is None


def test_strategy_signal_and_evidence_packet_from_core3_candidate():
    candidate = _Candidate(
        symbol="VCB",
        indicators={
            "current_price": 50.0,
            "edge_strategy_analysis": {
                "passed": True,
                "strategy_name": "breakout_after_accumulation_v3",
                "setup_type": "EDGE_BREAKOUT",
                "edge_score": 72.5,
                "risk": {"stop_loss": 0.05},
            },
            "money_flow_analysis": {"regime": "BREAKOUT_FLOW"},
        },
    )

    signal = build_strategy_signal_from_candidate(candidate, "2026-05-13")
    evidence = build_evidence_packet_from_candidate(candidate, "2026-05-13")
    context = format_agentic_context(signal, evidence)

    assert signal is not None
    assert signal.symbol == "VCB"
    assert signal.strategy_name == "breakout_after_accumulation_v3"
    assert signal.immutable is True
    assert evidence.edge_strategy["passed"] is True
    assert "Do not invent a different source strategy" in context
