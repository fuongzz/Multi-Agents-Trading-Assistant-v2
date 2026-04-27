"""trade_graph.py — LangGraph StateGraph cho Trade pipeline (ngắn hạn).

Flow:
  load_macro
    → [technical | flow | sentiment]  (parallel)
    → synthesis
    → retrieve_context   ← Hybrid RAG: internal (L1+L2) + external (Vietstock KB) song song
    → trader_trade
    → risk_trade
    → format_output → END

Early exit: market_context.should_trade = False → END (skip pipeline).
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from multiagents_trading_assistant.agents.trade import (
    technical_agent, flow_agent, sentiment_agent, synthesis_agent,
)
from multiagents_trading_assistant.nodes import trader_trade, risk_trade
from multiagents_trading_assistant.formatters.trade_output import format_trade_signal
from multiagents_trading_assistant.services import output_service
from multiagents_trading_assistant.services.memory_service import (
    retrieve_trade_context,
    retrieve_knowledge,
)


_BASE_DIR = Path(__file__).resolve().parent.parent
_CACHE_DIR = _BASE_DIR / "cache"
_CACHE_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# State schema
# ──────────────────────────────────────────────

class TradeState(TypedDict, total=False):
    # Input
    symbol: str
    date: str
    setup_type: str
    market_context: dict
    macro_context: dict

    # Analyst outputs (parallel)
    technical_analysis: dict
    foreign_flow_analysis: dict
    sentiment_analysis: dict

    # Synthesis
    synthesis: dict

    # Hybrid RAG context (retrieved before trader_trade)
    # {
    #   "internal": {recent_decisions, similar_setups, has_position, t3_blocked, ...},
    #   "external": {fundamental_summary, historical_stats},
    # }
    memory_context: dict

    # Decision
    trader_decision: dict
    risk_output: dict

    # Output
    formatted_text: str
    error: Optional[str]


# ──────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────

def load_macro(state: TradeState) -> dict:
    date = state.get("date", datetime.now().strftime("%Y-%m-%d"))
    cache_path = _CACHE_DIR / f"macro_{date}.json"
    if cache_path.exists():
        try:
            macro = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"[trade_graph] macro cache: bias={macro.get('macro_bias', 'N/A')}")
            return {"macro_context": macro}
        except Exception as e:
            print(f"[trade_graph] macro cache lỗi: {e}")
    print("[trade_graph] không có macro cache.")
    return {"macro_context": {}}


def run_analysts(state: TradeState) -> dict:
    """Chạy 3 analysts song song qua thread pool."""
    symbol = state["symbol"]
    date = state["date"]
    print(f"[trade_graph] === parallel analysts {symbol} ===")

    def _tech(): return _safe("technical", technical_agent.analyze, symbol, date)
    def _flow(): return _safe("flow",      flow_agent.analyze,      symbol, date)
    def _sent(): return _safe("sentiment", sentiment_agent.analyze, symbol, date)

    async def _run():
        loop = asyncio.get_event_loop()
        return await asyncio.gather(
            loop.run_in_executor(None, _tech),
            loop.run_in_executor(None, _flow),
            loop.run_in_executor(None, _sent),
        )

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tech, flow, sent = loop.run_until_complete(_run())
        loop.close()
    except Exception as e:
        print(f"[trade_graph] asyncio fail → sequential: {e}")
        tech, flow, sent = _tech(), _flow(), _sent()

    return {
        "technical_analysis": tech or {},
        "foreign_flow_analysis": flow or {},
        "sentiment_analysis": sent or {},
    }


def run_synthesis(state: TradeState) -> dict:
    result = synthesis_agent.run(
        technical_analysis=state.get("technical_analysis", {}),
        foreign_flow_analysis=state.get("foreign_flow_analysis", {}),
        sentiment_analysis=state.get("sentiment_analysis", {}),
        setup_type=state.get("setup_type", ""),
    )
    print(f"[trade_graph] synthesis → conf={result.get('confluence_score')} ({result.get('setup_quality')})")
    return {"synthesis": result}


def run_retrieve_context(state: TradeState) -> dict:
    """Hybrid RAG: truy xuất internal (L1+L2) + external (Vietstock KB) song song."""
    symbol     = state.get("symbol", "")
    setup_type = state.get("setup_type", "")
    ma_trend   = state.get("technical_analysis", {}).get("ma_trend", "UNKNOWN")
    confluence = float(state.get("synthesis", {}).get("confluence_score") or 50.0)
    date       = state.get("date", "")

    internal = {}
    external = {"fundamental_summary": "", "historical_stats": ""}

    def _internal():
        return retrieve_trade_context(symbol, setup_type, ma_trend, confluence)

    def _external():
        return retrieve_knowledge(symbol, date)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_int = pool.submit(_internal)
        fut_ext = pool.submit(_external)
        try:
            internal = fut_int.result(timeout=15)
        except Exception as e:
            print(f"[trade_graph] internal memory fail: {e}")
        try:
            external = fut_ext.result(timeout=15)
        except Exception as e:
            print(f"[trade_graph] external knowledge fail: {e}")

    ctx = {"internal": internal, "external": external}
    print(
        f"[trade_graph] context: "
        f"{len(internal.get('recent_decisions', []))} recent, "
        f"{len(internal.get('similar_setups', []))} similar, "
        f"pos={internal.get('has_position')}, t3={internal.get('t3_blocked')}, "
        f"fund={'✓' if external.get('fundamental_summary') else '–'}, "
        f"stats={'✓' if external.get('historical_stats') else '–'}"
    )
    return {"memory_context": ctx}


def run_trader(state: TradeState) -> dict:
    return trader_trade.decide(state)


def run_risk(state: TradeState) -> dict:
    result = risk_trade.check(state)
    return {"risk_output": result}


def run_format(state: TradeState) -> dict:
    text = format_trade_signal(state)
    output_service.write_trade_signal(state, text)
    return {"formatted_text": text}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _safe(name: str, fn, *args, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[trade_graph] agent {name} fail: {e}")
        return {}


# ──────────────────────────────────────────────
# Build graph
# ──────────────────────────────────────────────

def build_graph():
    g = StateGraph(TradeState)
    g.add_node("load_macro",        load_macro)
    g.add_node("run_analysts",      run_analysts)
    g.add_node("synthesis",         run_synthesis)
    g.add_node("retrieve_context",  run_retrieve_context)
    g.add_node("trader_trade",      run_trader)
    g.add_node("risk_trade",        run_risk)
    g.add_node("format_output",     run_format)

    g.set_entry_point("load_macro")
    g.add_edge("load_macro",       "run_analysts")
    g.add_edge("run_analysts",     "synthesis")
    g.add_edge("synthesis",        "retrieve_context")
    g.add_edge("retrieve_context", "trader_trade")
    g.add_edge("trader_trade",     "risk_trade")
    g.add_edge("risk_trade",       "format_output")
    g.add_edge("format_output",    END)
    return g.compile()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def run_pipeline(
    symbol: str,
    setup_type: str = "BREAKOUT",
    market_context: dict | None = None,
    date: str | None = None,
) -> TradeState:
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'=' * 60}\n[trade_graph] START: {symbol} | {setup_type} | {date}\n{'=' * 60}")

    initial: TradeState = {
        "symbol": symbol, "date": date, "setup_type": setup_type,
        "market_context": market_context or {}, "macro_context": {},
        "technical_analysis": {}, "foreign_flow_analysis": {}, "sentiment_analysis": {},
        "synthesis": {}, "memory_context": {}, "trader_decision": {}, "risk_output": {},
        "formatted_text": "", "error": None,
    }

    app = build_graph()
    try:
        final = app.invoke(initial)
        print(f"[trade_graph] END: {symbol} → {final.get('risk_output', {}).get('final_action', '?')}")
        return final
    except Exception as e:
        print(f"[trade_graph] FAIL: {e}")
        initial["error"] = str(e)
        initial["risk_output"] = {
            "final_action": "CHỜ", "override_reason": f"Pipeline lỗi: {e}",
            "warnings": [], "sizing_modifier": 0.0,
        }
        return initial
