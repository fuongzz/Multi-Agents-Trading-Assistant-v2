"""
orchestrator.py — LangGraph pipeline 8 nodes cho AI Trading Assistant.

Flow:
  load_macro → run_analysts (song song) → aggregate
  → bull_debate → bear_debate → synthesize
  → trader → risk_check → format_output

Mỗi node nhận TradingState, trả về dict merge vào state.
Analyst nodes (1-5) chạy song song với asyncio.gather.
"""

import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

# ── Imports agents ──
# Phase 2 agents (đã có)
from multiagents_trading_assistant.agents import ptkt_agent, fa_agent, foreign_flow_agent
from multiagents_trading_assistant.research import sentiment_agent

# Phase 3 agents
from multiagents_trading_assistant.research import debate_agent
from multiagents_trading_assistant.trader import trader_agent
_HAS_DEBATE = True
_HAS_TRADER = True

# Phase 4
try:
    from multiagents_trading_assistant import risk_manager
    _HAS_RISK = True
except ImportError:
    _HAS_RISK = False

# Cache / output dirs
BASE_DIR   = Path(__file__).parent
CACHE_DIR  = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# State schema
# ──────────────────────────────────────────────

class TradingState(TypedDict):
    # Input
    symbol: str
    date: str
    setup_type: str           # SHORT_TERM | MID_TERM | SIDEWAY_BUY | SIDEWAY_SELL
    market_context: dict      # từ screener
    macro_context: dict       # từ macro_agent (cached)

    # Analyst outputs (Node 1-5)
    ptkt_analysis: dict
    fa_analysis: dict
    foreign_flow_analysis: dict
    sentiment_analysis: dict

    # Debate (Node 6)
    bull_argument: str
    bear_argument: str
    debate_synthesis: dict

    # Decision (Node 7)
    trader_decision: dict

    # Final (Node 8)
    risk_output: dict         # final_action, override_reason, sizing_modifier

    # Output
    discord_message: str
    error: Optional[str]


# ──────────────────────────────────────────────
# Node 1: Load macro context từ cache
# ──────────────────────────────────────────────

def load_macro_context(state: TradingState) -> dict:
    """Đọc macro context từ cache. Nếu không có → dùng dict rỗng, pipeline tiếp tục."""
    date = state.get("date", datetime.now().strftime("%Y-%m-%d"))
    cache_path = CACHE_DIR / f"macro_{date}.json"

    if cache_path.exists():
        try:
            macro = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"[orchestrator] Macro cache loaded: bias={macro.get('macro_bias', 'N/A')}")
            return {"macro_context": macro}
        except Exception as e:
            print(f"[orchestrator] Lỗi đọc macro cache: {e}")

    print("[orchestrator] Không có macro cache, dùng context trống.")
    return {"macro_context": {}}


# ──────────────────────────────────────────────
# Node 2: Run analysts song song (asyncio)
# ──────────────────────────────────────────────

def run_parallel_analysts(state: TradingState) -> dict:
    """
    Chạy 4 analyst agents song song: PTKT, FA, ForeignFlow, Sentiment.
    Mỗi agent fail độc lập — lỗi 1 agent không ảnh hưởng agent khác.
    """
    symbol = state["symbol"]
    date   = state["date"]
    print(f"\n[orchestrator] === Chạy 4 analysts song song cho {symbol} ===")

    async def _run_all():
        loop = asyncio.get_event_loop()

        # Wrap blocking calls thành coroutines chạy trong thread pool
        results = await asyncio.gather(
            loop.run_in_executor(None, _safe_call, ptkt_agent.analyze,       symbol, date, "ptkt"),
            loop.run_in_executor(None, _safe_call, fa_agent.analyze,         symbol, date, "fa"),
            loop.run_in_executor(None, _safe_call, foreign_flow_agent.analyze, symbol, date, "foreign_flow"),
            loop.run_in_executor(None, _safe_call, sentiment_agent.analyze,  symbol, date, "sentiment"),
        )
        return results

    try:
        # Chạy event loop (sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ptkt_res, fa_res, ff_res, sent_res = loop.run_until_complete(_run_all())
        loop.close()
    except Exception as e:
        print(f"[orchestrator] Lỗi asyncio.gather: {e} — fallback sequential")
        ptkt_res  = _safe_call(ptkt_agent.analyze,         symbol, date, "ptkt")
        fa_res    = _safe_call(fa_agent.analyze,           symbol, date, "fa")
        ff_res    = _safe_call(foreign_flow_agent.analyze, symbol, date, "foreign_flow")
        sent_res  = _safe_call(sentiment_agent.analyze,   symbol, date, "sentiment")

    print(f"[orchestrator] Analysts xong — ptkt={bool(ptkt_res)}, fa={bool(fa_res)}, "
          f"ff={bool(ff_res)}, sentiment={bool(sent_res)}")

    return {
        "ptkt_analysis":         ptkt_res  or {},
        "fa_analysis":           fa_res    or {},
        "foreign_flow_analysis": ff_res    or {},
        "sentiment_analysis":    sent_res  or {},
    }


def _safe_call(fn, symbol: str, date: str, name: str) -> dict:
    """Gọi agent với error handling — lỗi trả về dict rỗng."""
    try:
        return fn(symbol, date)
    except Exception as e:
        print(f"[orchestrator] Agent {name} lỗi: {e}")
        return {}


# ──────────────────────────────────────────────
# Node 3: Aggregate — log tóm tắt analysts
# ──────────────────────────────────────────────

def aggregate_analysts(state: TradingState) -> dict:
    """Không biến đổi state — chỉ log tóm tắt để debug."""
    symbol = state["symbol"]
    ptkt   = state.get("ptkt_analysis", {})
    fa     = state.get("fa_analysis", {})
    ff     = state.get("foreign_flow_analysis", {})
    sent   = state.get("sentiment_analysis", {})

    print(f"\n[orchestrator] === Tóm tắt analysts {symbol} ===")
    print(f"  PTKT: trend={ptkt.get('ma_trend')}, score={ptkt.get('confluence_score')}, "
          f"quality={ptkt.get('setup_quality')}")
    print(f"  FA:   valuation={fa.get('valuation')}, health={fa.get('financial_health')}")
    print(f"  FF:   room={ff.get('room_status')}, flow={ff.get('flow_trend')}, "
          f"sizing={ff.get('sizing_modifier')}")
    print(f"  Sent: score={sent.get('sentiment_score')}, label={sent.get('sentiment_label')}")

    return {}  # Không thay đổi state


# ──────────────────────────────────────────────
# Node 4-5-6: Debate (stub nếu chưa có debate_agent)
# ──────────────────────────────────────────────

def run_bull_argument(state: TradingState) -> dict:
    """Bull argument — gọi debate_agent nếu có, không thì stub."""
    if _HAS_DEBATE:
        return debate_agent.run_bull(state)

    # Stub: tổng hợp từ analysts
    ptkt = state.get("ptkt_analysis", {})
    fa   = state.get("fa_analysis", {})
    ff   = state.get("foreign_flow_analysis", {})
    sent = state.get("sentiment_analysis", {})

    bull = (
        f"[BULL - stub] {state['symbol']}: "
        f"PTKT={ptkt.get('ma_trend','?')}/score={ptkt.get('confluence_score','?')}, "
        f"FA={fa.get('valuation','?')}/{fa.get('financial_health','?')}, "
        f"FF={ff.get('flow_trend','?')}, "
        f"Sentiment={sent.get('sentiment_label','?')}({sent.get('sentiment_score','?')})"
    )
    print(f"[orchestrator] Bull arg (stub): {bull[:100]}...")
    return {"bull_argument": bull}


def run_bear_argument(state: TradingState) -> dict:
    """Bear argument — gọi debate_agent nếu có, không thì stub."""
    if _HAS_DEBATE:
        return debate_agent.run_bear(state)

    # Stub: liệt kê rủi ro
    ptkt = state.get("ptkt_analysis", {})
    fa   = state.get("fa_analysis", {})
    macro = state.get("macro_context", {})

    bear = (
        f"[BEAR - stub] {state['symbol']}: "
        f"RSI={ptkt.get('rsi','?')}, "
        f"FA={fa.get('valuation','?')}, "
        f"Macro={macro.get('macro_bias','?')}"
    )
    print(f"[orchestrator] Bear arg (stub): {bear[:100]}...")
    return {"bear_argument": bear}


def synthesize_debate(state: TradingState) -> dict:
    """Tổng hợp debate — gọi debate_agent nếu có, không thì stub."""
    if _HAS_DEBATE:
        return debate_agent.synthesize(state)

    # Stub synthesis
    synthesis = {
        "bull_key_points": [state.get("bull_argument", "")[:100]],
        "bear_key_points": [state.get("bear_argument", "")[:100]],
        "balance": "NEUTRAL",
        "key_risk": "Chưa có debate agent — cần implement Phase 3.",
        "debate_conclusion": "Stub — debate agent chưa được implement.",
    }
    return {"debate_synthesis": synthesis}


# ──────────────────────────────────────────────
# Node 7: Trader Agent
# ──────────────────────────────────────────────

def run_trader(state: TradingState) -> dict:
    """Gọi trader_agent — stub CHỜ nếu chưa có."""
    if _HAS_TRADER:
        return trader_agent.decide(state)

    # Stub: mặc định CHỜ
    decision = {
        "action": "CHỜ",
        "entry": None,
        "sl": None,
        "tp": None,
        "nav_pct": 0,
        "holding_period": "N/A",
        "confidence": "THẤP",
        "primary_reason": "Trader agent chưa được implement (Phase 3).",
        "risks": ["Chưa có trader agent"],
        "trader_note": "Stub — implement Phase 3.",
    }
    print(f"[orchestrator] Trader stub → CHỜ")
    return {"trader_decision": decision}


# ──────────────────────────────────────────────
# Node 8: Risk Manager
# ──────────────────────────────────────────────

def run_risk_manager(state: TradingState) -> dict:
    """Gọi risk_manager — stub pass-through nếu chưa có."""
    if _HAS_RISK:
        result = risk_manager.check(state)
        return {"risk_output": result}

    trader_dec = state.get("trader_decision", {})
    action = trader_dec.get("action", "CHỜ")

    risk_out = {
        "final_action":    action,
        "override_reason": None,
        "warnings":        ["Risk manager chưa implement (Phase 4)"],
        "sizing_modifier": 1.0,
    }
    print(f"[orchestrator] Risk stub → final_action={action}")
    return {"risk_output": risk_out}


# ──────────────────────────────────────────────
# Node 9: Format Discord message
# ──────────────────────────────────────────────

def format_discord_message(state: TradingState) -> dict:
    """Tạo mini analyst note để gửi Discord."""
    symbol   = state["symbol"]
    date     = state["date"]
    risk_out = state.get("risk_output", {})
    trader   = state.get("trader_decision", {})
    ptkt     = state.get("ptkt_analysis", {})
    fa       = state.get("fa_analysis", {})
    ff       = state.get("foreign_flow_analysis", {})
    sent     = state.get("sentiment_analysis", {})
    debate   = state.get("debate_synthesis", {})

    action       = risk_out.get("final_action", "CHỜ")
    override     = risk_out.get("override_reason")
    confidence   = trader.get("confidence", "?")
    entry        = trader.get("entry")
    sl           = trader.get("sl")
    tp           = trader.get("tp")
    nav_pct      = trader.get("nav_pct", 0)
    reason       = trader.get("primary_reason", "")
    risks        = trader.get("risks", [])

    # Emoji theo action
    action_emoji = {"MUA": "🟢", "BÁN": "🔴", "CHỜ": "🟡"}.get(action, "⚪")

    lines = [
        f"**{action_emoji} [{action}] {symbol}** — {date}",
        f"",
        f"**Kỹ thuật:** {ptkt.get('ma_trend','?')} | Score {ptkt.get('confluence_score','?')}/10 | {ptkt.get('setup_quality','?')}",
        f"**Cơ bản:** {fa.get('valuation','?')} | {fa.get('financial_health','?')}",
        f"**Khối ngoại:** {ff.get('flow_trend','?')} | Room {ff.get('room_status','?')} | Sizing ×{ff.get('sizing_modifier',1.0)}",
        f"**Sentiment:** {sent.get('sentiment_label','?')} ({sent.get('sentiment_score','?')}/100)",
        f"",
    ]

    if action in ("MUA", "BÁN") and entry:
        lines += [
            f"**Entry:** {entry:,.0f} | **SL:** {sl:,.0f} | **TP:** {tp:,.0f}" if sl and tp else f"**Entry:** {entry:,.0f}",
            f"**Tỷ lệ:** {nav_pct}% NAV | **Độ tin cậy:** {confidence}",
            f"",
        ]

    lines.append(f"**Lý do:** {reason}")

    if override:
        lines.append(f"⚠️ **Risk override:** {override}")

    if risks:
        lines.append(f"**Rủi ro:** {' | '.join(risks[:2])}")

    if debate.get("key_risk"):
        lines.append(f"**Key risk (debate):** {debate['key_risk']}")

    msg = "\n".join(lines)

    # Lưu output
    _save_output(symbol, date, state)

    print(f"\n[orchestrator] === Discord message for {symbol} ===")
    print(msg)
    return {"discord_message": msg}


def _save_output(symbol: str, date: str, state: TradingState) -> None:
    """Lưu full pipeline output vào output/{SYMBOL}_{date}_decision.json."""
    try:
        out_path = OUTPUT_DIR / f"{symbol}_{date}_decision.json"
        # Serialize state (bỏ qua các field không serializable)
        data = {k: v for k, v in state.items() if isinstance(v, (str, int, float, dict, list, bool, type(None)))}
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[orchestrator] Saved output → {out_path.name}")
    except Exception as e:
        print(f"[orchestrator] Lỗi lưu output: {e}")


# ──────────────────────────────────────────────
# Build LangGraph
# ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Tạo và compile LangGraph pipeline."""
    graph = StateGraph(TradingState)

    graph.add_node("load_macro",    load_macro_context)
    graph.add_node("run_analysts",  run_parallel_analysts)
    graph.add_node("aggregate",     aggregate_analysts)
    graph.add_node("bull_debate",   run_bull_argument)
    graph.add_node("bear_debate",   run_bear_argument)
    graph.add_node("synthesize",    synthesize_debate)
    graph.add_node("trader",        run_trader)
    graph.add_node("risk_check",    run_risk_manager)
    graph.add_node("format_output", format_discord_message)

    graph.set_entry_point("load_macro")
    graph.add_edge("load_macro",    "run_analysts")
    graph.add_edge("run_analysts",  "aggregate")
    graph.add_edge("aggregate",     "bull_debate")
    graph.add_edge("bull_debate",   "bear_debate")
    graph.add_edge("bear_debate",   "synthesize")
    graph.add_edge("synthesize",    "trader")
    graph.add_edge("trader",        "risk_check")
    graph.add_edge("risk_check",    "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def run_pipeline(
    symbol: str,
    setup_type: str = "SHORT_TERM",
    market_context: dict | None = None,
    date: str | None = None,
) -> TradingState:
    """
    Chạy toàn bộ pipeline cho một mã cổ phiếu.

    Args:
        symbol:         Mã cổ phiếu (VD: "VNM")
        setup_type:     Loại setup từ screener
        market_context: Context thị trường từ screener
        date:           Ngày phân tích (mặc định hôm nay)

    Returns:
        TradingState cuối cùng sau khi qua tất cả nodes
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"[orchestrator] BẮT ĐẦU PIPELINE: {symbol} | {setup_type} | {date}")
    print(f"{'='*60}")

    initial_state: TradingState = {
        "symbol":                symbol,
        "date":                  date,
        "setup_type":            setup_type,
        "market_context":        market_context or {},
        "macro_context":         {},
        "ptkt_analysis":         {},
        "fa_analysis":           {},
        "foreign_flow_analysis": {},
        "sentiment_analysis":    {},
        "bull_argument":         "",
        "bear_argument":         "",
        "debate_synthesis":      {},
        "trader_decision":       {},
        "risk_output":           {},
        "discord_message":       "",
        "error":                 None,
    }

    app = build_graph()

    try:
        final_state = app.invoke(initial_state)
        print(f"\n[orchestrator] PIPELINE XONG: {symbol} → "
              f"{final_state.get('risk_output', {}).get('final_action', '?')}")
        return final_state
    except Exception as e:
        print(f"[orchestrator] PIPELINE LỖI: {e}")
        initial_state["error"] = str(e)
        initial_state["risk_output"] = {
            "final_action": "CHỜ",
            "override_reason": f"Pipeline lỗi: {e}",
            "warnings": [],
            "sizing_modifier": 0.0,
        }
        return initial_state


def run_batch(
    candidates: list[dict],
    date: str | None = None,
) -> list[TradingState]:
    """
    Chạy pipeline cho danh sách mã từ screener.

    Args:
        candidates: List dict từ screener, mỗi phần tử có 'symbol', 'setup_type', 'market_context'
        date:       Ngày phân tích

    Returns:
        List TradingState kết quả
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    results = []
    for c in candidates:
        symbol       = c.get("symbol", "")
        setup_type   = c.get("setup_type", "SHORT_TERM")
        mkt_ctx      = c.get("market_context", {})

        if not symbol:
            continue

        state = run_pipeline(symbol, setup_type, mkt_ctx, date)
        results.append(state)

    print(f"\n[orchestrator] Batch xong: {len(results)} mã")
    return results


# ──────────────────────────────────────────────
# Test nhanh
# ──────────────────────────────────────────────

if __name__ == "__main__":
    result = run_pipeline("VNM", setup_type="MID_TERM")
    print("\n=== Final action:", result["risk_output"].get("final_action"))
    print("=== Discord message:\n", result["discord_message"])
