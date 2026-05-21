"""Agentic architecture contracts.

Phase 1 keeps deterministic strategy signals separate from LLM review.
"""

from multiagents_trading_assistant.agentic.adapters import (
    build_evidence_packet_from_candidate,
    build_strategy_signal_from_candidate,
    format_agentic_context,
)
from multiagents_trading_assistant.agentic.schemas import (
    AgenticReview,
    EvidencePacket,
    ExecutionProposal,
    FinalDecision,
    StrategySignal,
)
from multiagents_trading_assistant.agentic.position_exit_agent import (
    PositionExitAgentConfig,
    PositionExitReview,
    PositionState,
    PositionStateExitAgent,
)

__all__ = [
    "AgenticReview",
    "EvidencePacket",
    "ExecutionProposal",
    "FinalDecision",
    "PositionExitAgentConfig",
    "PositionExitReview",
    "PositionState",
    "PositionStateExitAgent",
    "StrategySignal",
    "build_evidence_packet_from_candidate",
    "build_strategy_signal_from_candidate",
    "format_agentic_context",
]
