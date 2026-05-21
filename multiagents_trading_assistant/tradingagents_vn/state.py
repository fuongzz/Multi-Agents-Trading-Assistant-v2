"""State schemas for TradingAgents-VN graph."""

from __future__ import annotations

from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def _last(a, b):
    """Reducer cho parallel nodes: lấy giá trị cuối."""
    return b


class InvestDebateState(TypedDict):
    bull_history: str       # concatenated string như repo gốc
    bear_history: str
    history: str            # full combined history
    current_response: str
    judge_decision: str
    count: int


class RiskDebateState(TypedDict):
    aggressive_history: str
    conservative_history: str
    neutral_history: str
    history: str
    latest_speaker: str
    current_aggressive_response: str   # tên field giống repo gốc
    current_conservative_response: str
    current_neutral_response: str
    judge_decision: str
    count: int


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sender: Annotated[str, _last]
    # Context
    company_of_interest: str    # tên field giống repo gốc
    trade_date: str             # signal_date — ngày ra quyết định
    as_of_date: str             # dữ liệu được phép nhìn tới (inclusive); "" = today
    # Analyst reports
    market_report: str
    sentiment_report: str       # news/sentiment — giống repo gốc
    flow_report: str            # foreign flow — VN specific
    fundamentals_report: str
    # Research debate
    investment_debate_state: InvestDebateState
    investment_plan: str        # Research Manager output — giống repo gốc
    # Trader
    trader_investment_plan: str  # tên field giống repo gốc
    # Risk debate
    risk_debate_state: RiskDebateState
    # Final output
    final_trade_decision: str   # tên field giống repo gốc
    # Memory
    past_context: str           # injected từ memory lúc bắt đầu
    # Agentic Phase 1 context
    strategy_signal: dict       # immutable deterministic alpha signal
    evidence_packet: dict       # broker-grade evidence snapshot
    agentic_context: str        # compact prompt context for LLM review
