"""TradingAgents-VN: LangGraph StateGraph — chuẩn hóa theo repo gốc tauricresearch."""

from __future__ import annotations

import json
from datetime import date as _date
from pathlib import Path
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from multiagents_trading_assistant.tradingagents_vn.agents.flow_analyst import (
    create_flow_analyst, create_flow_analyst_clear,
)
from multiagents_trading_assistant.tradingagents_vn.agents.fundamental_analyst import (
    create_fundamental_analyst, create_fundamental_analyst_clear,
)
from multiagents_trading_assistant.tradingagents_vn.agents.market_analyst import (
    create_market_analyst, create_market_analyst_clear,
)
from multiagents_trading_assistant.tradingagents_vn.agents.news_analyst import (
    create_news_analyst, create_news_analyst_clear,
)
from multiagents_trading_assistant.tradingagents_vn.agents.researchers import (
    create_bear_researcher, create_bull_researcher, create_research_manager,
)
from multiagents_trading_assistant.tradingagents_vn.agents.risk_managers import (
    create_aggressive_analyst, create_conservative_analyst,
    create_neutral_analyst, create_portfolio_manager,
)
from multiagents_trading_assistant.tradingagents_vn.agents.trader import create_trader
from multiagents_trading_assistant.tradingagents_vn.state import AgentState
from multiagents_trading_assistant.tradingagents_vn.tools import (
    FLOW_TOOLS, FUNDAMENTAL_TOOLS, MARKET_TOOLS, NEWS_TOOLS,
    as_of_date_context,
)
from multiagents_trading_assistant.tradingagents_vn.schema import TradePlan
from multiagents_trading_assistant.agentic import format_agentic_context

_HAIKU = "claude-haiku-4-5-20251001"
_SONNET = "claude-sonnet-4-6"

_MEMORY_DIR = Path(__file__).parent / "memory"
_MEMORY_DIR.mkdir(exist_ok=True)

MAX_INVEST_ROUNDS = 2
MAX_RISK_ROUNDS = 1


def _with_progress(name: str, fn):
    _LABELS = {
        "MarketAnalyst":         "📊 [1/4] Phân tích kỹ thuật...",
        "tools_market":          "   🔧 Gọi data tools (kỹ thuật)...",
        "clear_market":          "   ✓ Market xong",
        "FundamentalAnalyst":    "📋 [2/4] Phân tích cơ bản...",
        "tools_fundamental":     "   🔧 Gọi data tools (cơ bản)...",
        "clear_fundamental":     "   ✓ Fundamental xong",
        "NewsAnalyst":           "📰 [3/4] Phân tích tin tức...",
        "tools_news":            "   🔧 Gọi data tools (tin tức)...",
        "clear_news":            "   ✓ News xong",
        "FlowAnalyst":           "💸 [4/4] Phân tích dòng tiền NĐTNN...",
        "tools_flow":            "   🔧 Gọi data tools (dòng tiền)...",
        "clear_flow":            "   ✓ Flow xong",
        "BullResearcher":        "🐂 Bull đang lập luận...",
        "BearResearcher":        "🐻 Bear đang phản bác...",
        "ResearchManager":       "🧑‍💼 Trưởng phòng tổng hợp nghiên cứu...",
        "Trader":                "📈 Trader xây dựng kế hoạch giao dịch...",
        "AggressiveAnalyst":     "⚡ Risk — quan điểm Aggressive...",
        "ConservativeAnalyst":   "🛡️ Risk — quan điểm Conservative...",
        "NeutralAnalyst":        "⚖️ Risk — quan điểm Neutral...",
        "PortfolioManager":      "🏦 Giám đốc danh mục ra quyết định cuối...",
    }
    def wrapped(state):
        label = _LABELS.get(name, f"⏳ {name}...")
        print(label, flush=True)
        # ToolNode là Runnable, dùng .invoke(); plain function thì gọi trực tiếp
        if hasattr(fn, "invoke"):
            return fn.invoke(state)
        return fn(state)
    return wrapped


# ── Memory helpers ────────────────────────────────────────────────────────────

def _load_memory(symbol: str) -> str:
    path = _MEMORY_DIR / f"{symbol}.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        if not entries:
            return ""
        # Lấy 3 lần phân tích gần nhất
        recent = entries[-3:]
        lines = ["=== LỊCH SỬ PHÂN TÍCH GẦN ĐÂY ==="]
        for e in recent:
            lines.append(
                f"[{e['date']}] Quyết định: {e['decision']} | "
                f"Tóm tắt: {e['summary']}"
            )
        return "\n".join(lines)
    except Exception:
        return ""


def _load_memory_before(symbol: str, before_date: str) -> str:
    """Load memory nhưng chỉ lấy entries có date < before_date (anti-lookahead)."""
    path = _MEMORY_DIR / f"{symbol}.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = [e for e in data.get("entries", []) if e.get("date", "") < before_date]
        if not entries:
            return ""
        recent = entries[-3:]
        lines = ["=== LỊCH SỬ PHÂN TÍCH GẦN ĐÂY ==="]
        for e in recent:
            lines.append(
                f"[{e['date']}] Quyết định: {e['decision']} | "
                f"Tóm tắt: {e['summary']}"
            )
        return "\n".join(lines)
    except Exception:
        return ""


def _parse_trade_plan(raw: str, setup_type: str, symbol: str) -> "TradePlan | None":
    """Parse final_trade_decision text thành TradePlan.

    Thử 2 lần: JSON block trong text → fallback regex.
    Trả None nếu không parse được.
    """
    import re
    from multiagents_trading_assistant.tradingagents_vn.schema import TradePlan

    # Tìm JSON block trong text — hỗ trợ nested array (entry_zone, reasons, risks)
    # Pattern: tìm block bắt đầu từ { chứa "action" đến } đóng tương ứng
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if not json_match:
        # Thử không có code fence
        json_match = re.search(r'(\{[^{}]*"action"[^{}]*\})', raw, re.DOTALL)
    if json_match:
        try:
            raw_json = json_match.group(1) if json_match.lastindex else json_match.group()
            data = json.loads(raw_json)
            if "setup_type" not in data or data["setup_type"] == "UNKNOWN":
                data["setup_type"] = setup_type
            # entry_zone có thể là list [low, high] → convert to tuple
            if isinstance(data.get("entry_zone"), list) and len(data["entry_zone"]) == 2:
                data["entry_zone"] = tuple(data["entry_zone"])
            return TradePlan.model_validate(data)
        except Exception:
            pass

    # Fallback: parse key-value từ text (LLM hay viết dạng này)
    def _extract_float(pattern: str) -> float | None:
        m = re.search(pattern, raw, re.IGNORECASE)
        return float(m.group(1).replace(",", ".")) if m else None

    action_m = re.search(
        r"\b(MUA|CHO|BÁN|BAN|TRANH)\b", raw, re.IGNORECASE
    )
    if not action_m:
        return None

    action = action_m.group(1).upper()
    if action == "BÁN":
        action = "BAN"

    entry_low = _extract_float(r"entry.*?(\d+[\.,]\d+).*?[-–]")
    entry_high = _extract_float(r"[-–]\s*(\d+[\.,]\d+).*?entry")
    sl = _extract_float(r"(?:stop[_\s]loss|SL|dừng lỗ)[:\s]+(\d+[\.,]\d+)")
    tp = _extract_float(r"(?:take[_\s]profit|TP|chốt lời)[:\s]+(\d+[\.,]\d+)")
    conf = _extract_float(r"(?:confidence|tự tin)[:\s]+(\d[\.,]\d+)")

    if not all([entry_low, entry_high, sl, tp]):
        return None

    try:
        return TradePlan(
            action=action,
            entry_zone=(entry_low, entry_high),
            stop_loss=sl,
            take_profit=tp,
            position_pct=0.03,
            confidence=conf or 0.5,
            setup_type=setup_type,
            reasons=[],
            risks=[],
        )
    except Exception:
        return None


def _save_memory(symbol: str, trade_date: str, state: dict) -> None:
    path = _MEMORY_DIR / f"{symbol}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"entries": []}
    except Exception:
        data = {"entries": []}

    decision = state.get("final_trade_decision", "")
    # Tóm tắt ngắn từ 2 dòng đầu của quyết định
    summary = " ".join(decision.split("\n")[:2])[:200]

    data["entries"].append({
        "date": trade_date,
        "decision": summary[:50],
        "summary": summary,
    })
    # Giữ tối đa 20 entries
    data["entries"] = data["entries"][-20:]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Conditional routing helpers ───────────────────────────────────────────────

def _has_tool_calls(state: dict) -> bool:
    messages = state.get("messages", [])
    if not messages:
        return False
    last = messages[-1]
    return bool(getattr(last, "tool_calls", None))


class TradingAgentsVN:
    """Multi-agent trading analysis system cho thị trường Việt Nam."""

    def __init__(
        self,
        quick_model: str = _HAIKU,
        deep_model: str = _SONNET,
        max_invest_rounds: int = MAX_INVEST_ROUNDS,
        max_risk_rounds: int = MAX_RISK_ROUNDS,
        cheap: bool = False,
    ):
        self.max_invest_rounds = max_invest_rounds
        self.max_risk_rounds = max_risk_rounds

        if cheap:
            deep_model = _HAIKU
        haiku = ChatAnthropic(model=quick_model, temperature=0.3)
        sonnet = ChatAnthropic(model=deep_model, temperature=0.5)

        # Analyst nodes
        self._market_node = _with_progress("MarketAnalyst", create_market_analyst(haiku))
        self._market_clear = _with_progress("clear_market", create_market_analyst_clear())
        self._market_tools = _with_progress("tools_market", ToolNode(MARKET_TOOLS))

        self._fundamental_node = _with_progress("FundamentalAnalyst", create_fundamental_analyst(haiku))
        self._fundamental_clear = _with_progress("clear_fundamental", create_fundamental_analyst_clear())
        self._fundamental_tools = _with_progress("tools_fundamental", ToolNode(FUNDAMENTAL_TOOLS))

        self._news_node = _with_progress("NewsAnalyst", create_news_analyst(haiku))
        self._news_clear = _with_progress("clear_news", create_news_analyst_clear())
        self._news_tools = _with_progress("tools_news", ToolNode(NEWS_TOOLS))

        self._flow_node = _with_progress("FlowAnalyst", create_flow_analyst(haiku))
        self._flow_clear = _with_progress("clear_flow", create_flow_analyst_clear())
        self._flow_tools = _with_progress("tools_flow", ToolNode(FLOW_TOOLS))

        # Debate nodes
        self._bull = _with_progress("BullResearcher", create_bull_researcher(sonnet))
        self._bear = _with_progress("BearResearcher", create_bear_researcher(sonnet))
        self._research_mgr = _with_progress("ResearchManager", create_research_manager(sonnet))
        self._trader = _with_progress("Trader", create_trader(sonnet))
        self._aggressive = _with_progress("AggressiveAnalyst", create_aggressive_analyst(haiku))
        self._conservative = _with_progress("ConservativeAnalyst", create_conservative_analyst(haiku))
        self._neutral = _with_progress("NeutralAnalyst", create_neutral_analyst(haiku))
        self._pm = _with_progress("PortfolioManager", create_portfolio_manager(sonnet))

        self._graph = self._build_graph()

    def _build_graph(self):
        b = StateGraph(AgentState)

        # Analyst nodes + tool nodes + clear nodes (giống repo gốc)
        b.add_node("MarketAnalyst", self._market_node)
        b.add_node("tools_market", self._market_tools)
        b.add_node("clear_market", self._market_clear)

        b.add_node("FundamentalAnalyst", self._fundamental_node)
        b.add_node("tools_fundamental", self._fundamental_tools)
        b.add_node("clear_fundamental", self._fundamental_clear)

        b.add_node("NewsAnalyst", self._news_node)
        b.add_node("tools_news", self._news_tools)
        b.add_node("clear_news", self._news_clear)

        b.add_node("FlowAnalyst", self._flow_node)
        b.add_node("tools_flow", self._flow_tools)
        b.add_node("clear_flow", self._flow_clear)

        b.add_node("BullResearcher", self._bull)
        b.add_node("BearResearcher", self._bear)
        b.add_node("ResearchManager", self._research_mgr)
        b.add_node("Trader", self._trader)
        b.add_node("AggressiveAnalyst", self._aggressive)
        b.add_node("ConservativeAnalyst", self._conservative)
        b.add_node("NeutralAnalyst", self._neutral)
        b.add_node("PortfolioManager", self._pm)

        # Phase 1: Analysts tuần tự với ToolNode loop (giống repo gốc)
        b.set_entry_point("MarketAnalyst")
        b.add_conditional_edges("MarketAnalyst", self._route_analyst("clear_market", "tools_market"))
        b.add_edge("tools_market", "MarketAnalyst")
        b.add_edge("clear_market", "FundamentalAnalyst")

        b.add_conditional_edges("FundamentalAnalyst", self._route_analyst("clear_fundamental", "tools_fundamental"))
        b.add_edge("tools_fundamental", "FundamentalAnalyst")
        b.add_edge("clear_fundamental", "NewsAnalyst")

        b.add_conditional_edges("NewsAnalyst", self._route_analyst("clear_news", "tools_news"))
        b.add_edge("tools_news", "NewsAnalyst")
        b.add_edge("clear_news", "FlowAnalyst")

        b.add_conditional_edges("FlowAnalyst", self._route_analyst("clear_flow", "tools_flow"))
        b.add_edge("tools_flow", "FlowAnalyst")
        b.add_edge("clear_flow", "BullResearcher")

        # Phase 2: Bull/Bear debate
        b.add_conditional_edges("BullResearcher", self._route_invest_debate)
        b.add_conditional_edges("BearResearcher", self._route_invest_debate)
        b.add_edge("ResearchManager", "Trader")

        # Phase 3: Risk debate
        b.add_edge("Trader", "AggressiveAnalyst")
        b.add_conditional_edges("AggressiveAnalyst", self._route_risk_debate)
        b.add_conditional_edges("ConservativeAnalyst", self._route_risk_debate)
        b.add_conditional_edges("NeutralAnalyst", self._route_risk_debate)
        b.add_edge("PortfolioManager", END)

        return b.compile()

    @staticmethod
    def _route_analyst(clear_node: str, tools_node: str):
        def route(state: dict) -> str:
            return tools_node if _has_tool_calls(state) else clear_node
        return route

    def _route_invest_debate(
        self, state: AgentState
    ) -> Literal["BullResearcher", "BearResearcher", "ResearchManager"]:
        debate = state.get("investment_debate_state", {})
        if debate.get("count", 0) >= 2 * self.max_invest_rounds:
            return "ResearchManager"
        return "BearResearcher" if state.get("sender") == "BullResearcher" else "BullResearcher"

    def _route_risk_debate(
        self, state: AgentState
    ) -> Literal["AggressiveAnalyst", "ConservativeAnalyst", "NeutralAnalyst", "PortfolioManager"]:
        risk = state.get("risk_debate_state", {})
        if risk.get("count", 0) >= 3 * self.max_risk_rounds:
            return "PortfolioManager"
        speaker = risk.get("latest_speaker", "")
        if speaker == "Aggressive":
            return "ConservativeAnalyst"
        if speaker == "Conservative":
            return "NeutralAnalyst"
        return "AggressiveAnalyst"

    def analyze(
        self,
        symbol: str,
        trade_date: str | None = None,
        as_of_date: str | None = None,
        strategy_signal: dict | None = None,
        evidence_packet: dict | None = None,
    ) -> dict:
        """Chạy toàn bộ pipeline phân tích đa tác nhân.

        Args:
            symbol: Mã cổ phiếu VN (VCB, HPG, FPT...)
            trade_date: Ngày phân tích yyyy-mm-dd (signal_date). Mặc định hôm nay.
            as_of_date: Dữ liệu được phép nhìn tới (inclusive). None = hôm nay.
                        Dùng khi backtest để tránh leak future data.

        Returns:
            Final state dict với tất cả reports và final_trade_decision.
        """
        if trade_date is None:
            trade_date = _date.today().strftime("%Y-%m-%d")
        symbol = symbol.upper()
        effective_as_of = as_of_date or trade_date
        agentic_context = format_agentic_context(strategy_signal, evidence_packet)

        past_context = _load_memory(symbol)
        if as_of_date:
            # Chỉ đọc memory entries trước as_of_date để tránh leak
            past_context = _load_memory_before(symbol, as_of_date)

        initial: AgentState = {
            "messages": [],
            "sender": "",
            "company_of_interest": symbol,
            "trade_date": trade_date,
            "as_of_date": effective_as_of,
            "market_report": "",
            "sentiment_report": "",
            "flow_report": "",
            "fundamentals_report": "",
            "investment_debate_state": {
                "bull_history": "",
                "bear_history": "",
                "history": "",
                "current_response": "",
                "judge_decision": "",
                "count": 0,
            },
            "investment_plan": "",
            "trader_investment_plan": "",
            "risk_debate_state": {
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "history": "",
                "latest_speaker": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "judge_decision": "",
                "count": 0,
            },
            "final_trade_decision": "",
            "past_context": past_context,
            "strategy_signal": strategy_signal or {},
            "evidence_packet": evidence_packet or {},
            "agentic_context": agentic_context,
        }

        # Inject as_of_date vào tools qua context manager — tất cả tool calls
        # trong graph.invoke() sẽ dùng end_date = as_of_date.
        with as_of_date_context(effective_as_of):
            final_state = self._graph.invoke(initial)

        _save_memory(symbol, trade_date, final_state)
        return final_state

    def propagate(
        self,
        symbol: str,
        signal_date: str | None = None,
        as_of_date: str | None = None,
        setup_type: str = "UNKNOWN",
        strategy_signal: dict | None = None,
        evidence_packet: dict | None = None,
    ) -> TradePlan | None:
        """Phân tích và trả về TradePlan đã parse + validate.

        Dùng cho backtest LLM và deep mode pipeline.
        Trả None nếu không parse được quyết định thành TradePlan hợp lệ.

        Args:
            symbol: Mã cổ phiếu
            signal_date: Ngày ra quyết định (yyyy-mm-dd). Mặc định hôm nay.
            as_of_date: Dữ liệu cutoff. Mặc định = signal_date.
            setup_type: Setup type từ screener để điền vào TradePlan.
        """
        state = self.analyze(
            symbol,
            signal_date,
            as_of_date,
            strategy_signal=strategy_signal,
            evidence_packet=evidence_packet,
        )
        raw_decision = state.get("final_trade_decision", "")
        return _parse_trade_plan(raw_decision, setup_type, symbol)
