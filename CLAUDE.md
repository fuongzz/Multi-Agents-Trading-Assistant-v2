# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# AI Trading Assistant — Claude Code Context

> **ĐỌC FILE NÀY TRƯỚC KHI LÀM BẤT CỨ THỨ GÌ.**
> Đây là source of truth. Mọi quyết định thiết kế đều có lý do — không tự ý thay đổi.

---

## 📚 TÀI LIỆU — ĐỌC THEO NHU CẦU

| File | Nội dung | Khi nào đọc |
|------|----------|-------------|
| `docs/architecture.md` | Full pipeline, LangGraph nodes, State schema | Trước khi sửa orchestrator / pipeline |
| `docs/agents.md` | Input/output schema từng agent | Trước khi code bất kỳ agent nào |
| `docs/screener.md` | Top-down gate, market states, setup classification | Trước khi sửa screener.py |
| `docs/risk_rules.md` 🚨 | 5 hard rules VN — KHÔNG BAO GIỜ bypass | Trước khi sửa risk_manager.py |
| `docs/data_service.md` | DataService API, cache, rate limit | Trước khi sửa fetcher.py |
| `docs/bugs_and_fixes.md` 🚨 | Lỗi đã gặp + fix — ĐỌC TRƯỚC KHI DEBUG | Khi gặp lỗi bất kỳ |
| `docs/mcp_setup.md` | Cài ChromaDB MCP server cho Claude Code | Khi setup máy mới |

> **QUY TẮC VÀNG:** Tạo doc mới quan trọng (bug fix, architecture decision)
> → **cập nhật bảng này ngay lập tức.**

---

## 1. Vision

Hệ thống Multi-Agent AI **tự động quét và alert qua Discord** — không phải chatbot thụ động.

- **Universe:** VN100 (HOSE + HNX)
- **Output:** Mini analyst note → Discord: bối cảnh thị trường + ngành + lý do + khuyến nghị
- **Action:** Chỉ 3 giá trị — `MUA` / `BÁN` / `CHỜ`. Tuyệt đối không có giá trị nào khác.

---

## 2. Core Philosophy — Signal-First, AI là Analyst

> **Quyết định thiết kế (2026-04-12):** Chuyển từ "gate nhiều tầng" sang "signal-first + AI analyst".
> Lý do: gate nhiều → backtest đẹp nhưng 1 năm vài signal → không phải trading thật.
> Sản phẩm hướng đến người dùng cần **đủ signal hàng ngày** + **lý do rõ ràng từ AI**.

### Output mỗi ngày gồm 2 phần:

**Phần 1 — Daily Brief** (luôn chạy, bất kể market trend)
> Bức tranh toàn cục: macro + VN-Index trend + sector heatmap + dòng tiền ngoại
> Kể cả ngày DOWNTREND vẫn chạy — để người dùng hiểu *tại sao* nên đứng ngoài

**Phần 2 — Trading Signals** (Top 3-5 mã mỗi ngày)
> Screener chạy 3 chiến lược song song → 7-20 raw signals → AI analyst chọn top 3-5

### Quy tắc vàng về giá vào lệnh 🚨
> **KHÔNG mua gần kháng cự** — entry phải có room lên đến TP
> **KHÔNG bán gần hỗ trợ** — nếu BÁN, phải có room xuống rõ ràng
> Risk Manager và Trader Agent phải kiểm tra khoảng cách entry → resistance trước khi ra lệnh MUA.

### 5 chiến lược screener (rule-based, nới lỏng):

| # | Chiến lược | Điều kiện | Signal/ngày |
|---|------------|-----------|-------------|
| 1 | **Breakout** | UPTREND + nến đóng trên kháng cự cứng + volume >1.5×TB20 | 2-5 mã |
| 2 | **Retest sau Breakout** | UPTREND + giá pullback về vùng kháng cự vừa break (nay thành hỗ trợ) + volume cạn | 2-4 mã |
| 3 | **Spring (HH/HL)** | UPTREND + đỉnh sau > đỉnh trước + đáy sau > đáy trước + volume xác nhận | 2-4 mã |
| 4 | **MA Pullback** | UPTREND + giá về MA60 (<5%) + RSI 35-55 + volume cạn | 3-8 mã |
| 5 | **RSI Oversold Bounce** | RSI <35 + giá trên MA200 + không DOWNTREND dài hạn | 2-6 mã |

> Chiến lược 1-2-3 là bộ 3 liên kết: cùng 1 cổ phiếu có thể cho signal ở cả 3 thời điểm khác nhau trong cùng 1 chu kỳ breakout. Screener cần nhận biết cổ phiếu đang ở điểm nào trong chu kỳ.

### AI đóng vai Analyst, không phải Gatekeeper:

```
Raw signals (7-20 mã)
    → PTKT agent   — confluence score, S/R
    → FA agent     — fundamentals có hỗ trợ không?
    → Sentiment    — tin tức gần đây?
    → ForeignFlow  — khối ngoại đang làm gì?
    → Debate       — Bull vs Bear tranh luận
    → Trader       — chấm Quality Score 1-10 + lý do
    → Risk Manager — hard rules VN (giữ nguyên)
    → Top 3-5 mã kèm entry/SL/TP/lý do
```

---

## 3. Market States

| State | Điều kiện | Chiến lược |
|-------|-----------|-----------|
| **UPTREND** | Giá > MA20 > MA60 > MA200 | SHORT_TERM (breakout) hoặc MID_TERM (pullback) |
| **SIDEWAY** | MA20 ≈ MA60 | Range trading, **bắt buộc đồng pha** với VN-Index |
| **DOWNTREND** | Giá < MA20 < MA60 | **KHÔNG LÀM GÌ** — dừng pipeline, giữ tiền mặt |

Chi tiết → `docs/screener.md`

---

## 4. Pipeline

```
[scheduler.py — 08:45 VN]
        ↓
[macro_agent.py — cache 1 lần/ngày]
    BEARISH → DỪNG + Discord alert
        ↓
[screener.py — Top-Down Gate]
    DOWNTREND → DỪNG
    → candidates[] (chỉ mã có setup)
        ↓
[orchestrator.py — LangGraph]
    Node 1-5 SONG SONG:
      agents/ptkt_agent.py        Haiku — RSI, MA, MACD, Bollinger, ATR
      agents/fa_agent.py          Haiku — P/E, P/B, ROE, EPS
      agents/foreign_flow_agent.py Haiku — room, net flow
      research/sentiment_agent.py  Haiku — CafeF + VnExpress → score
      [macro context từ cache]
    Node 6: research/debate_agent.py  Sonnet × 2 vòng Bull vs Bear
    Node 7: trader/trader_agent.py    Sonnet → MUA/BÁN/CHỜ + entry/SL/TP/%NAV
    Node 8: risk_manager.py           Rule-based — 5 hard rules, KHÔNG LLM
        ↓
[discord_bot.py — alert]
```

Chi tiết nodes + State schema → `docs/architecture.md`

---

## 5. Model Assignment

| Agent | Model | Lý do |
|-------|-------|-------|
| PTKT, FA, ForeignFlow, Sentiment, Debate Synthesis | `claude-haiku-4-5-20251001` | Nhanh, rẻ, cacheable |
| Bull Debate, Bear Debate, Trader | `claude-sonnet-4-5-20251022` | Cần reasoning sâu |

---

## 6. VN Market Hard Rules 🚨

> Hardcoded — KHÔNG BAO GIỜ thay đổi theo bất kỳ lý do gì.

- **Không short selling** — BÁN chỉ khi `db.has_position(symbol) == True`
- **T+3 settlement** — mua xong chờ 3 ngày mới bán được
- **Biên độ** ±7% HOSE / ±10% HNX
- **ATC** 14:25–14:30 — không đặt lệnh mới
- **Giờ GD** 09:00–11:30 và 13:00–14:30

5 hard rules đầy đủ → `docs/risk_rules.md` 🚨

---

## 7. Tech Stack

```
Python      3.11  ← KHÔNG dùng 3.12+, incompatible numpy/anthropic
AI          anthropic==0.84.0 · langgraph · langchain-anthropic
Data        vnstock==3.4.2 (VCI source) · yfinance
Crawl       beautifulsoup4 · requests
Indicators  pandas-ta-openbb
DB          SQLite — trading_assistant.db
Alert       discord.py
UI          streamlit + plotly
Scheduler   apscheduler
Memory      3 lớp: SQLite (L1) · ChromaDB (L2, Phase 4) · hardcoded (L3)
```

---

## 8. File Structure

```
AI-Trading-Assistant/
├── CLAUDE.md                          ← file này
├── requirements.txt
├── .gitignore
├── .env.example
├── docs/                              ← tài liệu thiết kế (xem bảng trên)
├── scripts/
│   └── index_docs.py                  ← ChromaDB indexer
└── multiagents_trading_assistant/
    ├── main.py                        # Entry point CLI
    ├── orchestrator.py                # LangGraph pipeline (8 nodes)
    ├── agent.py                       # run_agent() Sonnet / run_agent_lite() Haiku
    ├── screener.py                    # Top-down gate: macro→market→sector→stock
    ├── fetcher.py                     # DataService wrapper ← agents KHÔNG gọi vnstock
    ├── indicators.py                  # RSI, MA, MACD, Bollinger, ATR, S/R
    ├── tools.py                       # Claude tools + dispatcher
    ├── database.py                    # SQLite interface
    ├── risk_manager.py                # 5 hard rules — rule-based, KHÔNG LLM
    ├── news_fetcher.py                # Crawl CafeF + VnExpress
    ├── scheduler.py                   # Cron 06:00/08:45/15:10/20:00
    ├── discord_bot.py                 # Bot + webhook
    ├── dashboard.py                   # Streamlit 4 tab
    ├── agents/
    │   ├── __init__.py
    │   ├── macro_agent.py             # Global + VN macro (cache 1 lần/ngày)
    │   ├── ptkt_agent.py              # Phân tích kỹ thuật — Haiku
    │   ├── fa_agent.py                # Phân tích cơ bản — Haiku
    │   └── foreign_flow_agent.py      # Khối ngoại + room — Haiku
    ├── research/
    │   ├── __init__.py
    │   ├── debate_agent.py            # Bull vs Bear (Sonnet × 2 vòng)
    │   └── sentiment_agent.py         # Sentiment tin tức — Haiku
    ├── trader/
    │   ├── __init__.py
    │   └── trader_agent.py            # Quyết định MUA/BÁN/CHỜ — Sonnet
    └── memory/
        ├── __init__.py
        └── memory_system.py           # L1 SQLite / L2 ChromaDB / L3 hardcoded
```

---

## 9. Code Conventions

- **Comments, docstrings, print/log** → **tiếng Việt**
- **Biến, hàm, class** → tiếng Anh, snake_case
- **JSON output fields** → tiếng Anh
- `action` chỉ nhận: `"MUA"` / `"BÁN"` / `"CHỜ"`
- Cache key: `{SYMBOL}_{YYYY-MM-DD}` — ví dụ `VNM_2026-04-12`
- **Agents KHÔNG BAO GIỜ import vnstock trực tiếp** — tất cả qua `fetcher.py`

---

## 10. Known Fixes 🚨 — ÁP DỤNG NGAY, KHÔNG HỎI LẠI

```python
# Fix 1 — BẮT ĐẦU mọi file import pandas_ta
import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11

# Fix 2 — fetcher.py, pandas 3.x + vnstock compatibility
if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = pd.DataFrame.map

# Fix 3 — Vnstock rate limit (free tier ~15 req/phút)
if i > 0 and i % 15 == 0:
    time.sleep(60)
```

Gặp bug mới → ghi ngay vào `docs/bugs_and_fixes.md` + cập nhật bảng 📚.

---

## 11. Scheduler

| Giờ VN | Job | Mô tả |
|--------|-----|-------|
| 06:00 | morning_fetch | Macro + foreign flow + news |
| 08:45 | morning_brief | Scan VN100 → pipeline → Discord |
| 15:10 | session_close | Recap phiên |
| 20:00 | evening_fetch | Compress memory |

---

## 12. Roadmap

- [ ] **Phase 1** — Fetcher + Indicators + Screener (market gate)
- [ ] **Phase 2** — 5 Parallel Agents + Orchestrator (LangGraph)
- [ ] **Phase 3** — Bull/Bear Debate + Trader Agent
- [ ] **Phase 4** — Risk Manager + Discord Alert
- [ ] **Phase 5** — Macro Context Agent (global + VN macro)
- [ ] **Phase 6** — Scheduler + Dashboard + Memory Layer
- [ ] **Phase 7** — Backtest VN100 + báo cáo tuần

---

## 13. Setup

```powershell
# Cài dependencies
py -3.11 -m pip install -r requirements.txt

# Cấu hình env
copy .env.example .env    # rồi điền API keys

# Chạy
py -3.11 multiagents_trading_assistant/main.py          # CLI
py -3.11 multiagents_trading_assistant/scheduler.py     # Auto cron
py -3.11 multiagents_trading_assistant/discord_bot.py   # Discord
py -3.11 -m streamlit run multiagents_trading_assistant/dashboard.py

# ChromaDB (sau khi thêm doc mới)
py -3.11 scripts/index_docs.py
```

```
ANTHROPIC_API_KEY=sk-ant-...
DISCORD_BOT_TOKEN=...
DISCORD_CHANNEL_ID=...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```
