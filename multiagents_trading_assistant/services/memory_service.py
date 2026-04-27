"""memory_service.py — Facade cho memory read/write trong trade pipeline.

Public API:
  L3_VN_RULES                                             — re-export hằng số thị trường
  retrieve_trade_context(symbol, setup_type, ma_trend,
                         confluence_score)  → dict        — RAG: đọc L1+L2 internal
  retrieve_knowledge(symbol, date)          → dict        — RAG: đọc Vietstock KB external
  save_trade_decision(state)               → None         — ghi L1 + L2 sau mỗi run
"""

from multiagents_trading_assistant.memory.memory_system import get_memory, L3_VN_RULES

__all__ = [
    "L3_VN_RULES",
    "retrieve_trade_context",
    "retrieve_knowledge",
    "save_trade_decision",
]


# ──────────────────────────────────────────────
# RAG Internal: L1 (SQLite) + L2 (ChromaDB experience)
# ──────────────────────────────────────────────

def retrieve_trade_context(
    symbol: str,
    setup_type: str,
    ma_trend: str,
    confluence_score: float,
) -> dict:
    """Truy xuất ngữ cảnh lịch sử nội bộ trước khi trader_trade ra quyết định.

    Gồm 3 phần:
      1. recent_decisions  — L1 SQLite: 7 ngày gần nhất cho mã này
      2. similar_setups    — L2 ChromaDB: vector search setup tương tự
      3. position_status   — L1 SQLite: vị thế / T+2.5 block

    Hoàn toàn safe — không raise exception kể cả khi DB trống.
    """
    mem = get_memory()
    ctx: dict = {}

    # ── Part 1: L1 Recent decisions (7 ngày) ──
    try:
        recent = mem.get_decision_history(symbol, days=7)
        ctx["recent_decisions"] = [
            {
                "date":            d.get("date"),
                "action":          d.get("action"),
                "final_action":    d.get("final_action"),
                "confidence":      d.get("confidence"),
                "override_reason": d.get("override_reason"),
                "entry":           d.get("entry"),
                "sl":              d.get("sl"),
                "tp":              d.get("tp"),
            }
            for d in recent
        ]
        ctx["cho_streak"] = mem.get_streak(symbol)
    except Exception as e:
        print(f"[memory_service] L1 retrieve fail: {e}")
        ctx["recent_decisions"] = []
        ctx["cho_streak"] = 0

    # ── Part 2: L2 ChromaDB similar setups ──
    try:
        if confluence_score >= 70:
            conf_band = "HIGH confluence STRONG"
        elif confluence_score >= 50:
            conf_band = "MEDIUM confluence"
        else:
            conf_band = "LOW confluence WEAK"

        query = (
            f"Setup {setup_type} trend {ma_trend} {conf_band} "
            f"Score {int(confluence_score)}"
        )

        same_symbol  = mem.find_similar_setups(query, symbol=symbol, n=2)
        cross_market = mem.find_similar_setups(query, symbol=None,   n=3)

        seen_ids     = {r["id"] for r in same_symbol}
        cross_unique = [r for r in cross_market if r["id"] not in seen_ids]

        ctx["similar_setups"] = same_symbol + cross_unique[:3]
        ctx["l2_available"]   = mem.l2.available
    except Exception as e:
        print(f"[memory_service] L2 retrieve fail: {e}")
        ctx["similar_setups"] = []
        ctx["l2_available"]   = False

    # ── Part 3: Position / T+2.5 status ──
    try:
        ctx["has_position"] = mem.has_position(symbol)
        ctx["t3_blocked"]   = mem.is_t3_blocked(symbol)
        if ctx["has_position"]:
            pos = mem.l1.get_position(symbol)
            ctx["current_position"] = {
                "entry_price": pos.get("entry_price"),
                "entry_date":  pos.get("entry_date"),
                "strategy":    pos.get("strategy"),
                "sl":          pos.get("sl"),
                "tp":          pos.get("tp"),
                "nav_pct":     pos.get("nav_pct"),
            }
        else:
            ctx["current_position"] = None
    except Exception as e:
        print(f"[memory_service] position check fail: {e}")
        ctx["has_position"]     = False
        ctx["t3_blocked"]       = False
        ctx["current_position"] = None

    return ctx


# ──────────────────────────────────────────────
# RAG External: Vietstock Knowledge Base
# ──────────────────────────────────────────────

def retrieve_knowledge(symbol: str, date: str) -> dict:
    """Truy xuất external knowledge từ Vietstock Knowledge Base.

    Gồm 2 phần:
      1. fundamental_summary — vietstock_reports: BCTC, nghị quyết, giải trình
      2. historical_stats    — vietstock_stats: insight chu kỳ, xác suất

    Trả về "" cho từng phần nếu collection trống (cold start graceful).
    Không raise exception.
    """
    from multiagents_trading_assistant.memory.knowledge_base import get_knowledge_base

    kb = get_knowledge_base()
    result = {"fundamental_summary": "", "historical_stats": ""}

    # ── Báo cáo tài chính / nghị quyết ──
    try:
        docs = kb.search_reports(
            symbol=symbol,
            query=f"Sức khỏe tài chính lợi nhuận nợ xấu rủi ro doanh nghiệp {symbol}",
            n=3,
        )
        if docs:
            result["fundamental_summary"] = " | ".join(d.strip() for d in docs if d.strip())
    except Exception as e:
        print(f"[memory_service] knowledge reports fail ({symbol}): {e}")

    # ── Thống kê lịch sử / chu kỳ ──
    try:
        month = int(date[5:7]) if len(date) >= 7 else 0
        month_str = f"tháng {month}" if month else ""
        docs = kb.search_stats(
            symbol=symbol,
            query=f"Xác suất tăng giá {symbol} {month_str} chu kỳ seasonality lịch sử",
            n=2,
        )
        if docs:
            result["historical_stats"] = " | ".join(d.strip() for d in docs if d.strip())
    except Exception as e:
        print(f"[memory_service] knowledge stats fail ({symbol}): {e}")

    return result


# ──────────────────────────────────────────────
# Save: L1 + L2 sau mỗi pipeline run
# ──────────────────────────────────────────────

def save_trade_decision(state: dict) -> None:
    """Lưu kết quả trade pipeline vào L1 (SQLite) và L2 (ChromaDB).

    Gọi từ pipeline_runner sau mỗi run_trade(). Không raise exception.
    """
    from multiagents_trading_assistant import database as db

    sym  = state.get("symbol", "")
    date = state.get("date", "")
    if not sym or not date:
        return

    trader = state.get("trader_decision", {})
    risk   = state.get("risk_output", {})
    tech   = state.get("technical_analysis", {})
    synth  = state.get("synthesis", {})

    entry_mid = None
    ez = trader.get("entry_zone")
    if isinstance(ez, (list, tuple)) and len(ez) == 2:
        entry_mid = (ez[0] + ez[1]) / 2

    # ── L1: SQLite ──
    try:
        db.save_decision(
            symbol          = sym,
            date            = date,
            action          = trader.get("action", "CHỜ"),
            final_action    = risk.get("final_action", "CHỜ"),
            strategy        = state.get("setup_type"),
            quality_score   = None,
            confidence      = trader.get("confidence"),
            entry           = entry_mid,
            sl              = trader.get("stop_loss"),
            tp              = trader.get("take_profit"),
            nav_pct         = trader.get("position_pct"),
            override_reason = risk.get("override_reason"),
            full_output     = {
                k: v for k, v in state.items()
                if isinstance(v, (str, int, float, dict, list, bool, type(None)))
            },
        )
        print(f"[memory_service] L1 saved: {sym} {date} → {risk.get('final_action', '?')}")
    except Exception as e:
        print(f"[memory_service] L1 save fail ({sym}): {e}")

    # ── L2: ChromaDB ──
    try:
        mem = get_memory()
        if not mem.l2.available:
            return

        doc_text = (
            f"Symbol: {sym} Date: {date} "
            f"Action: {risk.get('final_action', '?')} "
            f"Setup: {state.get('setup_type', '?')} "
            f"Trend: {tech.get('ma_trend', '?')} "
            f"Score: {synth.get('confluence_score', '?')} "
            f"Quality: {synth.get('setup_quality', '?')} "
            f"Confidence: {trader.get('confidence', '?')} "
            f"Reason: {str(trader.get('primary_reason', ''))[:500]}"
        )
        metadata = {
            "symbol":     sym,
            "date":       date,
            "action":     risk.get("final_action", "CHỜ"),
            "setup_type": state.get("setup_type", ""),
            "ma_trend":   tech.get("ma_trend", ""),
            "confidence": trader.get("confidence", ""),
        }
        mem.l2._collection.upsert(
            ids       = [f"{sym}_{date}"],
            documents = [doc_text],
            metadatas = [metadata],
        )
        print(f"[memory_service] L2 saved: {sym}_{date}")
    except Exception as e:
        print(f"[memory_service] L2 save fail ({sym}): {e}")
