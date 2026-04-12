# Architecture — AI Trading Assistant

## Full Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  EXTERNAL SOURCES                                                │
│  vnstock (VCI) · yfinance · CafeF · VnExpress                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │  qua fetcher.py (DataService)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  SCHEDULER — scheduler.py (APScheduler, giờ VN)                 │
│  06:00 morning_fetch │ 08:45 morning_brief                      │
│  15:10 session_close │ 20:00 evening_fetch                      │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  MACRO AGENT — agents/macro_agent.py (Haiku, cache 1 lần/ngày) │
│  INPUT:  S&P500, DXY, dầu, vàng, Nikkei, KOSPI, HSI            │
│          USD/VND, lãi suất SBV + tin tức vĩ mô                  │
│  OUTPUT: macro_score(-2→+2), macro_bias, key_risks, summary     │
│                                                                  │
│  macro_bias = BEARISH  →  STOP ALL  →  Discord alert            │
└──────────────────────────┬──────────────────────────────────────┘
                           │ macro_context (cached dict)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  SCREENER GATE — screener.py                                     │
│  Tầng 1: VN-Index MA200/60/20 → UPTREND / SIDEWAY / DOWNTREND  │
│           DOWNTREND → should_trade=False → STOP                  │
│  Tầng 2: Sector relative strength (20 phiên vs VN-Index)        │
│           Chỉ giữ sector outperform (RS > 0)                     │
│  Tầng 3: Stock setup per symbol                                  │
│           UPTREND  → SHORT_TERM (breakout) / MID_TERM (pullback) │
│           SIDEWAY  → SIDEWAY_BUY / SIDEWAY_SELL / MID_RANGE      │
│  OUTPUT: List[CandidateStock], sorted by priority_score          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ candidates[] — chỉ mã pass gate
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR — orchestrator.py (LangGraph)                      │
│                                                                  │
│  Node 1-5 PARALLEL (asyncio.gather):                            │
│  ┌──────────┐ ┌────────┐ ┌─────────────┐ ┌─────────┐ ┌──────┐ │
│  │   PTKT   │ │   FA   │ │ForeignFlow  │ │Sentiment│ │Macro │ │
│  │  Haiku   │ │ Haiku  │ │   Haiku     │ │  Haiku  │ │cache │ │
│  └──────────┘ └────────┘ └─────────────┘ └─────────┘ └──────┘ │
│                      ↓ aggregate                                 │
│  Node 6: Bull vs Bear Debate                                     │
│    debate_agent.py — Sonnet × 2 vòng                            │
│    Synthesizer — Haiku tổng hợp                                  │
│                      ↓                                           │
│  Node 7: Trader Agent                                            │
│    trader_agent.py — Sonnet                                      │
│    → MUA/BÁN/CHỜ + entry + SL + TP + %NAV + confidence         │
│                      ↓                                           │
│  Node 8: Risk Manager                                            │
│    risk_manager.py — Rule-based, KHÔNG LLM                      │
│    → override nếu vi phạm hard rules                             │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                          │
│  Discord — mini analyst note (discord_bot.py)                    │
│  SQLite  — lưu decisions (database.py)                           │
│  Cache   — output/{SYMBOL}_{date}.json                           │
│  UI      — Streamlit dashboard (dashboard.py)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## LangGraph State Schema

```python
from typing import TypedDict, Optional

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
    error: Optional[str]      # None nếu pipeline thành công
```

---

## LangGraph Node Map

```python
# orchestrator.py
graph = StateGraph(TradingState)

graph.add_node("load_macro",      load_macro_context)   # từ cache
graph.add_node("run_analysts",    run_parallel_analysts) # asyncio.gather
graph.add_node("aggregate",       aggregate_analysts)
graph.add_node("bull_debate",     run_bull_argument)
graph.add_node("bear_debate",     run_bear_argument)
graph.add_node("synthesize",      synthesize_debate)
graph.add_node("trader",          run_trader)
graph.add_node("risk_check",      run_risk_manager)
graph.add_node("format_output",   format_discord_message)

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
```

---

## Cache File Structure

```
multiagents_trading_assistant/
├── cache/
│   ├── macro_2026-04-12.json          # 1 lần/ngày — dùng chung toàn batch
│   ├── VNM_2026-04-12_ohlcv.json
│   ├── VNM_2026-04-12_indicators.json
│   ├── VNM_2026-04-12_fundamentals.json
│   └── VNM_2026-04-12_foreign_flow.json
└── output/
    └── VNM_2026-04-12_decision.json   # full pipeline output
```

---

## Error Handling

| Lỗi | Xử lý |
|-----|-------|
| Analyst agent fail | Dùng empty dict, pipeline tiếp tục |
| Vnstock rate limit (429) | Retry 3 lần exponential backoff, rồi skip |
| LLM timeout | Fallback về CHỜ, reason = "LLM timeout" |
| Discord send fail | Log SQLite, retry sau 5 phút |
| Screener không fetch được VN-Index | Abort toàn bộ batch |
