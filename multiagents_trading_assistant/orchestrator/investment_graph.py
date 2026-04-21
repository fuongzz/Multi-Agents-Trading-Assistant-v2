"""investment_graph.py — LangGraph StateGraph cho Investment pipeline (dài hạn).

Flow:
  load_macro
    → [fundamental | valuation]  (parallel)
    → bull_debate → bear_debate → synthesize_debate
    → trader_invest
    → risk_invest
    → format_output → END

Early exit: macro_bias = BEARISH → END sớm (ghi log, không ra signal).
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from multiagents_trading_assistant.agents.invest import (
    fundamental_agent, valuation_agent, debate_agent,
)
from multiagents_trading_assistant.agents.invest.macro_agent import get_macro_context
from multiagents_trading_assistant.nodes import trader_invest, risk_invest
from multiagents_trading_assistant.formatters.invest_output import format_invest_signal
from multiagents_trading_assistant.services import output_service


_BASE_DIR = Path(__file__).resolve().parent.parent
_CACHE_DIR = _BASE_DIR / "cache"
_CACHE_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# State schema
# ──────────────────────────────────────────────

class InvestState(TypedDict, total=False):
    # Input
    symbol: str
    date: str
    market_context: dict
    macro_context: dict

    # Analysts (parallel)
    fundamental_analysis: dict
    valuation_analysis: dict

    # Debate
    bull_argument: str
    bear_argument: str
    debate_synthesis: dict

    # Decision
    trader_decision: dict
    risk_output: dict

    # Output
    formatted_text: str
    error: Optional[str]


# ──────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────

def load_macro(state: InvestState) -> dict:
    date = state.get("date", datetime.now().strftime("%Y-%m-%d"))
    macro = get_macro_context(date=date)
    bias = macro.get("macro_bias", "NEUTRAL")
    print(f"[investment_graph] macro bias={bias}")
    return {"macro_context": macro}


def run_analysts(state: InvestState) -> dict:
    """Chạy fundamental + valuation song song."""
    symbol = state["symbol"]
    date = state["date"]
    print(f"[investment_graph] === parallel analysts {symbol} ===")

    def _fa(): return _safe("fundamental", fundamental_agent.analyze, symbol, date)
    def _val(): return _safe("valuation", valuation_agent.analyze, symbol, date)

    async def _run():
        loop = asyncio.get_event_loop()
        return await asyncio.gather(
            loop.run_in_executor(None, _fa),
            loop.run_in_executor(None, _val),
        )

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        fa, val = loop.run_until_complete(_run())
        loop.close()
    except Exception as e:
        print(f"[investment_graph] asyncio fail → sequential: {e}")
        fa, val = _fa(), _val()

    return {
        "fundamental_analysis": fa or {},
        "valuation_analysis": val or {},
    }


def run_bull(state: InvestState) -> dict:
    return debate_agent.run_bull(state)


def run_bear(state: InvestState) -> dict:
    return debate_agent.run_bear(state)


def run_synthesize(state: InvestState) -> dict:
    return debate_agent.synthesize(state)


def run_trader(state: InvestState) -> dict:
    return trader_invest.decide(state)


def run_risk(state: InvestState) -> dict:
    result = risk_invest.check(state)
    return {"risk_output": result}


def run_format(state: InvestState) -> dict:
    text = format_invest_signal(state)
    output_service.write_invest_signal(state, text)
    return {"formatted_text": text}


# ──────────────────────────────────────────────
# Conditional edge: early exit nếu BEARISH
# ──────────────────────────────────────────────

def _after_macro(state: InvestState) -> str:
    bias = state.get("macro_context", {}).get("macro_bias", "NEUTRAL")
    if bias == "BEARISH":
        print(f"[investment_graph] Macro BEARISH → early exit cho {state.get('symbol')}")
        return "early_exit"
    return "run_analysts"


def _early_exit(state: InvestState) -> dict:
    symbol = state.get("symbol", "?")
    reason = state.get("macro_context", {}).get("key_risk", "Macro BEARISH")
    result = {
        "trader_decision": {
            "action": "CHỜ", "target_price": None, "position_pct": 0,
            "holding_horizon": "3-12 tháng", "confidence": "THẤP",
            "primary_reason": f"Macro BEARISH: {reason}",
            "exit_condition": "N/A", "max_drawdown_tolerance": 20, "risks": [],
        },
        "risk_output": {
            "final_action": "CHỜ", "override_reason": f"Macro BEARISH: {reason}",
            "warnings": [], "sizing_modifier": 0.0,
        },
    }
    text = format_invest_signal({**state, **result})
    output_service.write_invest_signal({**state, **result}, text)
    result["formatted_text"] = text
    return result


# ──────────────────────────────────────────────
# Build graph
# ──────────────────────────────────────────────

def build_graph():
    g = StateGraph(InvestState)
    g.add_node("load_macro",   load_macro)
    g.add_node("early_exit",   _early_exit)
    g.add_node("run_analysts", run_analysts)
    g.add_node("bull_debate",  run_bull)
    g.add_node("bear_debate",  run_bear)
    g.add_node("synthesize",   run_synthesize)
    g.add_node("trader_invest",run_trader)
    g.add_node("risk_invest",  run_risk)
    g.add_node("format_output",run_format)

    g.set_entry_point("load_macro")
    g.add_conditional_edges("load_macro", _after_macro, {
        "run_analysts": "run_analysts",
        "early_exit":   "early_exit",
    })
    g.add_edge("early_exit",   END)
    g.add_edge("run_analysts", "bull_debate")
    g.add_edge("bull_debate",  "bear_debate")
    g.add_edge("bear_debate",  "synthesize")
    g.add_edge("synthesize",   "trader_invest")
    g.add_edge("trader_invest","risk_invest")
    g.add_edge("risk_invest",  "format_output")
    g.add_edge("format_output",END)
    return g.compile()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _safe(name: str, fn, *args, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[investment_graph] agent {name} fail: {e}")
        return {}


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def run_pipeline(
    symbol: str,
    market_context: dict | None = None,
    date: str | None = None,
) -> InvestState:
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'=' * 60}\n[investment_graph] START: {symbol} | {date}\n{'=' * 60}")

    initial: InvestState = {
        "symbol": symbol, "date": date,
        "market_context": market_context or {}, "macro_context": {},
        "fundamental_analysis": {}, "valuation_analysis": {},
        "bull_argument": "", "bear_argument": "", "debate_synthesis": {},
        "trader_decision": {}, "risk_output": {},
        "formatted_text": "", "error": None,
    }

    app = build_graph()
    try:
        final = app.invoke(initial)
        print(f"[investment_graph] END: {symbol} → {final.get('risk_output', {}).get('final_action', '?')}")
        return final
    except Exception as e:
        print(f"[investment_graph] FAIL: {e}")
        initial["error"] = str(e)
        initial["risk_output"] = {
            "final_action": "CHỜ", "override_reason": f"Pipeline lỗi: {e}",
            "warnings": [], "sizing_modifier": 0.0,
        }
        return initial
