# AI Trading Assistant — Project Reference

**Dự án**: Multi-agent trading assistant cho thị trường chứng khoán Việt Nam  
**Stack**: Python 3.11, LangGraph, Anthropic Claude, DNSE LightSpeed API, vnstock_data (Golden), FiinQuantX  
**Phạm vi**: Toàn bộ HOSE (~400 mã, lọc thanh khoản ≥500k/ngày)

---

## Kiến trúc Tổng Quan — Dual Pipeline

Hệ thống chạy **hai pipeline song song, hoàn toàn độc lập**, được trigger bởi APScheduler. Output gửi lên hai Discord channel riêng biệt. Tất cả output là **khuyến nghị** (RECOMMENDATION_ONLY) — không auto-execute lệnh.

```
APScheduler (Asia/Ho_Chi_Minh)
  ├── Investment pipeline  [Thứ 2, 08:00]          ID: invest_weekly
  │     HOSE liquid (vol≥300k) → invest_screener
  │       → load_macro [early exit nếu macro BEARISH]
  │       → [fundamental_agent + valuation_agent] song song (Haiku)
  │       → bull_debate (Sonnet) → bear_debate (Sonnet, 2 rounds)
  │       → synthesize (Haiku)
  │       → trader_invest (Sonnet) → risk_invest → format_output
  │       → #invest-signal (Discord, màu xanh)
  │
  ├── Trade pipeline  [Hàng ngày, 08:30]            ID: trade_daily
  │     HOSE liquid (vol≥500k) → trade_screener
  │       → load_macro → portfolio_monitor (daily guard)
  │       → [technical + flow + sentiment + money_flow] song song
  │       → synthesis (rule-based) → retrieve_context (hybrid RAG)
  │       → trader_trade (Sonnet) → risk_trade → persist_trade → format_output
  │       → #trade-signal (Discord, màu cam)
  │
  ├── Session Monitor  [Thứ 2–6, 09:00–14:35, mỗi 5 phút]  ID: session_monitor
  │     [A] Hard exit check (positions table):
  │           SL/TP hit + T+2.5 available → alert "⚡ THOÁT NGAY" / "💰 CHỐT LỜI"
  │           SL hit + chưa T+2.5 → alert "🔴 BỊ KẸP" (lặp mỗi 5 phút)
  │           TP hit + chưa T+2.5 → update peak_price → trailing SL alert "📈 ĐẠT TARGET"
  │     [B] Re-analysis check (decisions table):
  │           Re-run technical_agent (Haiku ~2s) trên BUY signals hôm nay
  │           Alert nếu: confluence drop ≥3pt / setup flip / giá sát SL hoặc TP ±2%
  │     Live price: DNSE WebSocket (≤60s cache) → DNSE HTTP fallback
  │
  ├── Bob Strategy Meeting  [Thứ 6, 20:00]          ID: bob_strategy_meeting
  │     Review tuần → cập nhật strategy memory (ℳₛ)
  │
  ├── Code update check  [Mỗi giờ]                  ID: code_update_check
  │     git fetch → pull → restart nếu có commit mới
  │
  └── Cleanup  [Chủ nhật, 02:00]                    ID: cleanup_weekly
        Xoá cache/data cũ
```

---

## Nguyên Tắc Phân Biệt Hai Pipeline

| Chiều | Investment pipeline | Trade pipeline |
|---|---|---|
| Mục tiêu | Nắm giữ trung-dài hạn | Giao dịch ngắn hạn ATO |
| Khung thời gian | 3–12 tháng | Price-action based (không fixed) |
| Tín hiệu chính | BCTC, định giá nội tại, macro | Giá, khối lượng, dòng NN, money flow |
| Logic ra quyết định | Valuation gap (giá < nội tại) | Price action + confluence |
| Exit condition | Story thay đổi / định giá đủ | Price action signal (không exit theo thời gian) |
| Risk logic | Margin of safety, sector concentration | R:R ratio, portfolio cap, money flow gate |
| T+3 / biên độ ±7% | Ít ảnh hưởng | Ràng buộc cứng, tính vào SL/TP |
| Screener criteria | ROE, P/E, tăng trưởng EPS | 18 setups TA |
| Tần suất chạy | Hàng tuần (Thứ 2) | Hàng ngày (08:30 trước ATO) |
| Persistence | RECOMMENDATION_ONLY | RECOMMENDATION_ONLY |

---

## Cấu Trúc File

```
multiagents_trading_assistant/
  orchestrator/
    pipeline_runner.py        — APScheduler (5 jobs), DNSE WebSocket feed, startup Discord
    investment_graph.py       — LangGraph StateGraph cho Investment pipeline
    trade_graph.py            — LangGraph StateGraph cho Trade pipeline
    session_monitor.py        — 5-phút real-time monitor (SL/TP + re-analysis)
  screener/
    invest_screener.py        — Lọc FA (ROE, P/E, EPS growth)
    trade_screener.py         — Lọc TA (18 setups, 20 workers parallel)
  agents/
    invest/
      fundamental_agent.py   — ROE, EPS, financial_health (Haiku)
      valuation_agent.py     — Intrinsic value PE/PB projection, MoS (Haiku, không DCF)
      macro_agent.py         — Macro bias, cached daily (Haiku)
      debate_agent.py        — Bull/Bear thesis 2 rounds + synthesize (Sonnet + Haiku)
    trade/
      technical_agent.py     — MA, RSI, MACD, BB, S/R, confluence 0-10 (Haiku)
      flow_agent.py          — Foreign room, net flow, accumulation signal (Haiku)
      sentiment_agent.py     — News crawl CafeF/VnExpress, ingest SQLite+ChromaDB (Haiku)
      money_flow_agent.py    — Blackbox regime detection (Rule-based, NO LLM)
      synthesis_agent.py     — Weighted merge → confluence 0-100 (Rule-based)
  nodes/
    trader_invest.py         — MUA/CHỜ/TRÁNH, 3-12 tháng, position 3-5% NAV (Sonnet)
    trader_trade.py          — State machine 7 actions, price-action exit (Sonnet)
    risk_invest.py           — MoS gate, sector concentration, macro veto
    risk_trade.py            — 9 hard rules (circuit breaker, portfolio cap, R:R, v.v.)
    portfolio_monitor.py     — Daily guard: check exits, trailing SL suggestions
    persist_trade.py         — Mark RECOMMENDATION_ONLY (không auto-write position)
  services/
    llm_service.py           — Provider abstraction (Haiku / Sonnet)
    data_service.py          — vnstock_data Golden + FiinQuantX wrapper
    memory_service.py        — ChromaDB operations (dùng chung)
    output_service.py        — send_to_channel(), send_pipeline_alert(), Discord + JSON
    dnse_client.py           — DNSE LightSpeed REST + WebSocket client
  formatters/
    invest_embed.py          — Discord embed màu xanh, format weekly
    trade_embed.py           — Discord embed màu cam, format daily + SL/TP
  data/
    providers/
      base.py                — DataProvider Protocol (vendor-neutral interface)
      vnstock_provider.py    — vnstock_data Golden implementation
    repository.py            — get_data_provider() factory, env: MATA_DATA_PROVIDER
  fetcher.py                 — Primary: vnstock_data Golden; fallback: DNSE broker
  main.py                    — CLI entry point
  bob/                       — Bob's Strategy Development Meeting (Thứ 6)
  backtest/                  — Walk-forward backtester
  quantagents_backtest/      — QuantAgents backtester
docs/
  fiinquant.md               — FiinQuantX API reference
```

---

## State Schema

### InvestState (investment_graph.py)
```python
class InvestState(TypedDict):
    symbol: str
    date: str
    market_context: dict
    macro_context: dict
    fundamental_analysis: dict    # output fundamental_agent (ROE, EPS, financial_health)
    valuation_analysis: dict      # output valuation_agent (intrinsic, MoS, valuation label)
    bull_argument: str            # output bull_debate (Sonnet)
    bear_argument: str            # output bear_debate (Sonnet, 2 rounds)
    debate_synthesis: dict        # output synthesize (Haiku): balance STRONG_BULL/NEUTRAL/STRONG_BEAR
    trader_decision: dict         # action, target_price, position_pct, holding_horizon, exit_condition
    risk_output: dict             # vetoed, margin_of_safety, warnings
    formatted_text: str
    error: Optional[str]
```

### TradeState (trade_graph.py)
```python
class TradeState(TypedDict):
    symbol: str
    date: str
    setup_type: str               # 18 setups — xem trade_screener.py
    market_context: dict
    macro_context: dict
    backtest_mode: bool           # Anti-look-ahead: disable memory retrieval khi backtest
    technical_analysis: dict      # confluence 0-10, ma_trend, RSI, MACD, BB, S/R
    foreign_flow_analysis: dict   # room_status, flow_trend, accumulation_signal
    sentiment_analysis: dict      # sentiment_score 0-100, news_count, key_positive/negative
    money_flow_analysis: dict     # regime (Blackbox), score, action_bias (rule-based)
    synthesis: dict               # confluence_score 0-100, setup_quality, drivers, blockers
    memory_context: dict          # {internal: {recent_decisions, has_position, t3_blocked, ...},
                                  #  external: {fundamental_summary, historical_stats, news_summary}}
    trader_decision: dict         # action, entry_zone, SL, initial_target, R:R, trail_sl_guide
    risk_output: dict             # vetoed, warnings, final_action
    portfolio_summary: dict       # open positions, daily guard output
    execution_plan: dict          # RECOMMENDATION_ONLY flag
    formatted_text: str
    error: Optional[str]
```

---

## Models & Agents

| Agent | Model | Loại | Mục đích |
|---|---|---|---|
| fundamental_agent | `claude-haiku-4-5-20251001` | LLM | ROE, EPS, financial_health |
| valuation_agent | `claude-haiku-4-5-20251001` | LLM | Intrinsic value (PE/PB projection, không DCF), MoS |
| macro_agent | `claude-haiku-4-5-20251001` | LLM | Macro bias, cached daily |
| technical_agent | `claude-haiku-4-5-20251001` | LLM | MA, RSI, MACD, BB, S/R, confluence 0-10 |
| flow_agent | `claude-haiku-4-5-20251001` | LLM | Foreign room, net flow, accumulation |
| sentiment_agent | `claude-haiku-4-5-20251001` | LLM | News crawl + ingest SQLite/ChromaDB |
| debate_agent (synthesize) | `claude-haiku-4-5-20251001` | LLM | Merge bull/bear → consensus |
| money_flow_agent | — | Rule-based | Blackbox regime (BREAKOUT_FLOW / DISTRIBUTION / v.v.) |
| synthesis_agent | — | Rule-based | Weighted merge → confluence 0-100 |
| bull_debate / bear_debate | `claude-sonnet-4-6` | LLM | Thesis dài hạn (bear 2 rounds) |
| trader_invest | `claude-sonnet-4-6` | LLM | MUA/CHỜ/TRÁNH, 3-12 tháng |
| trader_trade | `claude-sonnet-4-6` | LLM | 7-action state machine, price-action exit |

### Synthesis Weights (trade pipeline)
```
technical_score  × 0.40   (confluence 0-10 → normalize 0-100)
flow_score       × 0.20   (room + trend + accumulation → 0-100)
money_flow_score × 0.25   (regime-based → 0-100)
sentiment_score  × 0.15   (sentiment_score 0-100)
────────────────────────
confluence_score 0-100
  STRONG ≥70 | MEDIUM 50-69 | WEAK <50
```

### Valuation Method (valuation_agent — không dùng DCF)
```
EPS_next      = EPS × (1 + growth_clamped[0%, 30%])
intrinsic_PE  = EPS_next × PE_median_industry
intrinsic_PB  = projected_BVPS × PB_median_industry  (retention = 70%)
intrinsic      = avg(intrinsic_PE, intrinsic_PB)
MoS            = (intrinsic − price) / intrinsic
label: RẺ (≥30%) | HỢP_LÝ (0-30%) | ĐẮT (<0%)
```

---

## Data Sources & Phân công

| Dữ liệu | Nguồn | Dùng ở pipeline |
|---|---|---|
| OHLCV lịch sử | **vnstock_data** `Market.equity().ohlcv()` (primary) → DNSE broker fallback | Cả hai |
| Danh sách mã HOSE | **vnstock_data** `Reference.equity.by_exchange()` | Cả hai |
| Danh sách theo nhóm (VN30/VN100) | **vnstock_data** `Reference.equity.by_group()` | Cả hai |
| Price board / foreign room | **vnstock_data** `Market.equity().price_board()` | Trade |
| PE/PB hiện tại | **vnstock_data** `Fundamental.equity().ratio()` | Investment |
| ROE/EPS/Growth | **vnstock_data** `Fundamental.equity().income_statement()` | Investment |
| Industry | **vnstock_data** `Reference.equity.list_by_industry()` → `icb_name` L2 | Investment |
| Foreign flow net5d/net20d | **vnstock_data** `Market.equity().foreign_flow()` | Trade |
| VN-Index / Macro | yfinance | Investment |
| News/Sentiment | CafeF crawl (Crawl4AI) | Trade |

---

## Screener

### invest_screener.py — FA criteria
- Universe: `get_liquid_symbols(min_avg_vol=300_000)` — HOSE liquid ~200+ mã
- Lọc theo ROE > 15%, tăng trưởng EPS dương 2 quý liên tiếp
- P/E thấp hơn median ngành
- Output: top 5–7 candidates với `valuation_score`

### trade_screener.py — TA criteria
- Universe: `get_liquid_symbols(min_avg_vol=500_000)` — HOSE liquid ~80-130 mã
- Fetch song song (20 workers ThreadPoolExecutor), cache theo ngày
- **VinGroup distortion detection**: so sánh VNI vs VNMidCap, nếu VIC/VHM/VRE méo VNI thì dùng VNMidCap làm `reference_trend`
- **DOWNTREND**: chỉ scan 5 reversal setups (DOUBLE_BOTTOM, RSI_BOUNCE, HAMMER, BULLISH_ENGULFING, PIN_BAR)
- **SIDEWAY / UPTREND**: scan toàn bộ 18 setups

| Setup | Nhóm | Logic tóm tắt |
|---|---|---|
| BREAKOUT | Continuation | Giá vượt đỉnh 20 phiên + volume ≥1.5× + thân nến >2% |
| FLAG_PENNANT | Continuation | Cột cờ +5% → nén volume thấp → giá áp đỉnh vùng nén |
| BB_SQUEEZE | Volatility | BB width thu hẹp 20 phiên → giá trên BB mid |
| RETEST | Continuation | Retest vùng breakout cũ, volume cạn |
| SPRING | Continuation | Higher High + Higher Low + volume tăng > giảm |
| GOLDEN_CROSS | Continuation | MA20 cắt lên MA50 trong 5 phiên + volume |
| DOUBLE_BOTTOM | Reversal | W pattern, hai đáy ≥97%, vượt neckline |
| MOMENTUM_SURGE | Continuation | 3 phiên tăng liên tiếp + volume tăng + gain >2% |
| MACD_CROSSOVER | Continuation | MACD cắt signal từ dưới lên trong 3 phiên |
| MA_PULLBACK | Continuation | Uptrend + pullback về MA60 (0-8%) + RSI 30-60 |
| INSIDE_BAR | Pattern | Inside bar + đóng cửa vượt đỉnh IB + volume |
| NR7 | Volatility | Range nhỏ nhất 7 phiên — SL hẹp, sắp bung |
| HAMMER | Reversal | Bóng dưới >2× thân + đóng gần đỉnh + RSI <50 |
| RSI_BOUNCE | Reversal | RSI <40 + trên MA200 + nến đảo chiều |
| **TREND_PULLBACK** | **Price Action** | Uptrend (HH+HL) + pullback về old breakout level + vol khô + nến xanh |
| **BREAKOUT_RETEST_ENTRY** | **Price Action** | Breakout 3-15 bars trước + giá retest level + vol < 0.8×MA + đóng xanh |
| **BULLISH_ENGULFING** | **Price Action** | Nến xanh nuốt hoàn toàn body nến đỏ + vol xanh > đỏ + body ≥1% + tại support |
| **PIN_BAR** | **Price Action** | Wick dưới ≥2.5× thân + upper wick <0.5× thân + đóng top 25% + tại S/R key level |

- Output: top **20** candidates với `priority_score`

---

## API Lỗi Đã Biết

- **vnstock Finance (KBS)**: 404 — endpoint thay đổi, không dùng (đã migrate sang vnstock_data)
- **vnstock `Vnstock(source='VCI')`**: `KeyError: 'data'` trong company init — không dùng
- **vnstock VN100 symbols API**: `KeyError: 'data'` — dùng `get_liquid_symbols()` thay thế
- **vnstock_data `price_board()` per-symbol loop**: gọi tuần tự từng mã — chậm với 80-130 mã screener; kiểm tra batch API nếu có
- **DNSE instruments `securityGroupId='EQ'`**: trả 400 BAD_REQUEST — filter bằng `len(sym)==3 and sym.isalpha()` ở Python
- **DNSE instruments `page=0`**: trả 400 — dùng `offset=0` thay thế
- **DNSE instruments** chỉ trả 252 mã 3-ký-tự trong khi HOSE có 402 — không dùng DNSE cho listing nữa, dùng vnstock_data `Reference`
- **Timestamp OHLCV từ DNSE**: trả UTC unix timestamp — cần convert sang Asia/Ho_Chi_Minh rồi `.normalize()` để ra date
- **vnstock_data `Fundamental.equity().financial_health()`**: có thể không tồn tại trên một số tier — có try/except bảo vệ, fallback sang `balance_sheet()` + `income_statement()`
- **ROE annualize**: tính `profit * 4 / equity` (giả định quarterly) — cần confirm data period trước khi dùng

---

## FiinQuantX

- **Credentials**: `FIINQUANT_USERNAME` / `FIINQUANT_PASSWORD` trong `.env`
- **Lazy singleton**: `_get_fiin_client()` — login 1 lần/session
- **Free tier**: tối đa 33 mã/lần, lịch sử tối đa 1 năm
- **`get_ratios()`** thiếu fields — dùng `get_financial_statement()` thay thế
- **`BasicInfor(tickers=[...])`** cần gọi `.get()` để ra DataFrame
- **Industry values** là ICB code tiếng Anh (`BANKS_L2`, ...) — valuation_agent cần map với `_INDUSTRY_PE_MEDIAN`
- **Docs đầy đủ**: `docs/fiinquant.md`

---

## Discord Output

| Channel | Pipeline | Màu embed | Nội dung |
|---|---|---|---|
| `#invest-signal` | Investment | Xanh teal | Mã, luận điểm dài hạn, target price, exit condition, margin of safety |
| `#trade-signal` | Trade | Cam coral | Mã, setup type, entry zone, SL, TP, R:R ratio, confluence score |

---

## APScheduler — pipeline_runner.py

```python
# Investment: Thứ 2 hàng tuần lúc 08:00
CronTrigger(day_of_week="mon", hour=8, minute=0, timezone=_VN_TZ)   # ID: invest_weekly

# Trade: Hàng ngày lúc 08:30 (trước ATO 09:00)
CronTrigger(hour=8, minute=30, timezone=_VN_TZ)                      # ID: trade_daily

# Session Monitor: Thứ 2-6, 09:00-14:35, mỗi 5 phút
CronTrigger(day_of_week="mon-fri", hour="9-11,13-14", minute="*/5")  # ID: session_monitor

# Bob Strategy Meeting: Thứ 6, 20:00
CronTrigger(day_of_week="fri", hour=20, minute=0, timezone=_VN_TZ)   # ID: bob_strategy_meeting

# Code update check: mỗi giờ (git fetch → pull → restart nếu có commit mới)
CronTrigger(minute=0, timezone=_VN_TZ)                               # ID: code_update_check

# Cleanup: Chủ nhật, 02:00
CronTrigger(day_of_week="sun", hour=2, minute=0, timezone=_VN_TZ)    # ID: cleanup_weekly
```

**Startup:** gửi Discord notification kèm git commit hash + khởi động DNSE WebSocket price feed (background thread).

---

## Cache

- Lưu tại `multiagents_trading_assistant/cache/`
- Key: `{symbol}_{date}_{pipeline_type}_{data_type}.json`
- Hết hạn sau mỗi ngày

---

## DNSE LightSpeed API

- **Credentials**: `DNSE_API_KEY` / `DNSE_API_SECRET` trong `.env`
- **Client**: `services/dnse_client.py` — `DNSERestClient`, lazy singleton `_get_dnse_client()`
- **Signature**: HMAC-SHA256 của `method + path + date + nonce`, header `X-API-Key` + `X-Signature` + `X-Aux-Date`
- **REST base URL**: `https://openapi.dnse.com.vn`
- **WebSocket base URL**: `wss://ws-openapi.dnse.com.vn` (chưa dùng — reserved)
- **OHLCV endpoint**: `GET /price/ohlc?symbol=&type=STOCK&resolution=1D&from=&to=` (unix timestamp)
- **Instruments endpoint**: `GET /instruments?marketId=STO&limit=200&offset=0`
- **Các channel WebSocket hỗ trợ**: Trade, Quote (order book), OHLC realtime, OHLC Closed, Expected Price (ATO/ATC), Market Index, Foreign Investor

---

## CLI

```bash
python -m multiagents_trading_assistant.main --pipeline trade     # trade pipeline
python -m multiagents_trading_assistant.main --pipeline invest    # investment pipeline
python -m multiagents_trading_assistant.main --pipeline all       # cả hai
python -m multiagents_trading_assistant.main --symbol VCB --pipeline invest
python -m multiagents_trading_assistant.main --schedule           # APScheduler 24/7
run.bat                                                           # 24/7 với auto-restart + git pull
```

---

## Ràng Buộc Thị Trường VN (áp dụng cả hai pipeline)

- **Biên độ giá**: ±7% (HOSE), ±10% (HNX) — tính vào SL/TP của trade pipeline
- **T+3 settlement**: mua hôm nay, nhận cổ phiếu sau 3 ngày — trade pipeline không exit ngay T+0
- **ATO session**: 09:00 HOSE — trade pipeline chạy lúc 08:30 để kịp đặt lệnh ATO
- **Foreign room**: check trước khi khuyến nghị cổ phiếu cho nhà đầu tư nước ngoài

## Risk Rules — Trade Pipeline (risk_trade.py)

| # | Rule | Ngưỡng | Kết quả |
|---|---|---|---|
| 1 | Circuit breaker | VNI < -3%/ngày | Override → CHỜ |
| 2 | Foreign room | >95% | Override → CHỜ |
| | | 90-95% | Sizing ×0.5 |
| | | 80-90% | Sizing ×0.8 |
| 3 | Portfolio cap | open ≥5 + confluence <75 | Override → CHỜ |
| | | open ≥4 + confluence <65 | Override → CHỜ |
| | | open ≥3 | Sizing ×0.5 |
| 4 | Regime confluence min | UPTREND: <62 / SIDEWAY: <65 / DOWNTREND: <70 | Override → CHỜ |
| 5 | T+2.5 | Đã mua mã này trong 2 ngày | Override → CHỜ |
| 6 | Không mua đuổi | Mã tăng ≥5% trong phiên | Override → CHỜ (trừ breakaway) |
| 7 | Thanh khoản | avg_vol_20d < 200k | Override → CHỜ |
| 8 | Money flow gate | DISTRIBUTION hoặc AVOID_OR_EXIT | Override → CHỜ |
| 9 | R:R tối thiểu | R:R < 1.5 | Override → CHỜ |
| 10 | Max loss | position_pct × (entry-SL)/entry > 2% | Giảm position_pct |

**Breakaway exception** (Rule 6): nhóm setup `_BREAKAWAY_SETUPS` (BREAKOUT, MOMENTUM_SURGE, BB_SQUEEZE, GOLDEN_CROSS, BREAKOUT_RETEST_ENTRY) + confluence ≥75 + large-cap → được mua dù tăng ≥5%, sizing ×0.5.

**Thiết kế**: max loss tính thuần relative — không cần NAV tuyệt đối, áp dụng đúng cho mọi user.

## Trader Trade — Quy tắc LLM (trader_trade.py)

- **SL**: ≤ 5% dưới entry_low (không phải 7%)
- **R:R**: ≥ 1.5 trong prompt (risk_trade enforce cứng)
- **Position theo confluence**: ≥70 → 5% NAV, 55-69 → 3%, <55 → 2% (LLM output: confidence CAO/TRUNG_BÌNH/THẤP)
- **Holding horizon**: **KHÔNG exit theo thời gian** — exit khi price action nói "dừng"
- **Entry guide**: mỗi trong 18 setups có hướng dẫn entry/SL/TP riêng trong `_build_prompt()`
- **GIA_TĂNG** chỉ hợp lệ khi: pnl ≥5% + SL hiện tại ≥99.5% entry + tổng NAV sau tăng <7%

### Triết lý giữ lệnh (price-action based)
- `initial_target` chỉ là mục tiêu **tham chiếu** — KHÔNG đặt lệnh bán tự động tại đó
- Khi giá đạt `initial_target` → nâng SL lên khóa lợi nhuận, tiếp tục giữ
- Chỉ thoát khi price action tín hiệu rõ:
  - SL trailing bị chạm (swing low dưới entry giảm sâu)
  - Double top xuất hiện (2 đỉnh ngang, pull back qua midpoint)
  - MA20 bị phá 2 nến liên tiếp + MA20 đang dốc xuống
  - Cấu trúc HH+HL bị phá (giá đóng dưới prior swing low 2 nến liên tiếp)

### Trailing SL guide (3 mức)
| Mức lãi | Hành động |
|---|---|
| +5% | Nâng SL lên entry (hòa vốn) hoặc swing low gần nhất − ATR×0.5 |
| +10% | SL = swing low ngay trước (7-10 bar) − ATR×0.5 |
| +20% | SL = swing low rộng hơn (15-20 bar) − ATR×0.5, cho trend thở |

### State machine trader_trade
- **CHƯA CÓ VỊ THẾ**: MUA / CHỜ / TRÁNH
- **ĐANG CÓ VỊ THẾ**: GIỮ / GIA_TĂNG / GIẢM / BÁN (KHÔNG phát MUA mới)
  - GIA_TĂNG: chỉ khi đang lãi + cấu trúc HH/HL còn nguyên + SL hiện tại đã gần hòa vốn

## Risk Rules — Investment Pipeline (risk_invest.py)

### Margin of Safety theo ngành
| Nhóm | Ngưỡng MoS tối thiểu |
|---|---|
| Bluechip (FPT, VCB, VHM, ...) | ≥ 20% |
| Banks / Tech / FMCG | ≥ 22-25% |
| Cyclical / Real Estate / Securities | ≥ 30-35% |
| Default | ≥ 30% |

### Sector Concentration
| Số vị thế cùng ngành | Kết quả |
|---|---|
| ≥ 3 | Override → CHỜ |
| ≥ 2 | Cảnh báo |

**Position sizing invest**: 3-5% NAV (MoS cao + debate STRONG_BULL → 5%, thấp → 3%)

---

## Backtest Module

```
multiagents_trading_assistant/backtest/
  engine.py    — Walk-forward backtester, zero look-ahead, trailing SL
  positions.py — Trade dataclass, exit reasons: SL | TSL | TP | TIMEOUT
  metrics.py   — compute_metrics(), setup_breakdown(), symbol_breakdown()
  report.py    — print_report() console, save_report() → backtest_results/*.csv
  cli.py       — CLI args, gọi từ main.py qua --backtest flag
```

**CLI:**
```bash
python -m multiagents_trading_assistant.main --backtest --symbol VCB --from 2024-01-01
python -m multiagents_trading_assistant.main --backtest --universe vn30 --from 2023-01-01
python -m multiagents_trading_assistant.main --backtest --setup BREAKOUT --universe liquid
```

**Thiết kế engine:**
- Entry: `open[i+1]` (ATO hôm sau) — không dùng close tín hiệu
- SL: `entry − max(1.5×ATR, 2%)`, capped 5% dưới entry
- TP: `entry + rr_ratio × (entry − SL)`, mặc định `rr_ratio=1.5`
- Trailing SL 3 bước: giữ nguyên (<3% lãi) → breakeven (≥3%) → trail ATR (≥5%)
- Dynamic `max_hold` theo setup (3/5/8 bars)

**max_hold theo nhóm:**
| Nhóm | Setups | Bars |
|---|---|---|
| Momentum ngắn | BREAKOUT, NR7, INSIDE_BAR, HAMMER, PIN_BAR, BULLISH_ENGULFING | 3 |
| Default | BB_SQUEEZE, SPRING, RETEST, MACD_CROSSOVER, FLAG_PENNANT, DOUBLE_BOTTOM, TREND_PULLBACK, BREAKOUT_RETEST_ENTRY | 5 |
| Trend / Reversal | GOLDEN_CROSS, RSI_BOUNCE, MA_PULLBACK | 8 |

**Backtest VN30 2023–2024 (kết quả tham khảo):**

| Setup | WR | Profit Factor | Ghi chú |
|---|---|---|---|
| NR7 | 42% | 0.98 | Gần breakeven, tốt nhất original |
| MOMENTUM_SURGE | 49% | 0.78 | WR cao nhưng avg win thấp |
| BULLISH_ENGULFING | 54% | 1.09 | PA setup tốt nhất — trên breakeven |
| DOUBLE_BOTTOM | 41% | 0.87 | Ổn định |
| BREAKOUT | 28% | 0.34 | Tệ nhất — nhiều false break |
| INSIDE_BAR | 25% | 0.32 | Tệ nhất — tránh dùng đơn lẻ |

**Chẩn đoán vấn đề đã biết (backtest 2023-2024):**
- SL hit 0-2 bars đầu là nguồn lỗ lớn nhất (avg -1.91%) — ATO gap phiên VN
- TIMEOUT avg -0.12% (gần hòa) — không phải vấn đề
- `BREAKOUT_RETEST_ENTRY` trigger quá nhiều false signal khi điều kiện chỉ OR (vol_dry OR bullish) — cần AND
- Cả 18 setups đều cần market context filter (chỉ trade UPTREND) để cải thiện WR

---

## ⚠️ Backtest Bias Patch #2 — 2026-05-18

Phát hiện 3 self-referential rolling bugs trong `features.py` (edge_lab) và 1 optimistic fill trong `llm_backtest.py`.

| Bug | File | Fix |
|---|---|---|
| `range_pct_min7` / `range_pct_q20` không shift → NR7 signal luôn true với chính nó | `edge_lab/features.py:134-135` | Thêm `.shift(1)` trước `rolling` |
| `atr_rolling_min` không shift → `volatility_squeeze_break` self-referential | `edge_lab/features.py:481` | Thêm `.shift(1)` trước `rolling` |
| `bb_width_min20` không shift → BB squeeze detection self-referential | `backtest/live_pipeline.py:874` | Thêm `.shift(1)` trước `rolling` |
| SL/TP fill dùng exact trigger price, bỏ qua gap open | `backtest/llm_backtest.py:188-193` | Dùng `min(open, sl)` và `max(open, tp)` |

**Impact ước tính**: NR7/BB squeeze WR sẽ giảm nhẹ, SL loss avg sẽ xấu hơn một chút (thực tế hơn). Chưa re-run full benchmark sau patch này.

---

## ⚠️ Backtest Bias Discovery — 2026-05-14

Phát hiện **same-bar-open exit lookahead bias** trong `_edge_exit_decision()`. **Số liệu backtest trước 2026-05-14 phóng đại ~1.8 lần**.

Đã patch `live_pipeline.py` để sửa: SL/ATR/trailing exit dùng `min(open, trigger) * (1 - slip)` thay vì `open * slip`. TP dùng `max(open, trigger) * (1 - slip)`.

### True benchmark — 3 core live strategies (VN100, 2025-01 → 2026-05, 16 tháng)

| Metric | Reported (biased) | **TRUE (sau fix)** |
|---|---|---|
| Total return | +101% | **+56%** |
| CAGR | (~76%) | **~40%** |
| Sharpe | 3.28 | **2.05** |
| MaxDD | -11% | **-17%** |
| Win rate | 63% | 62% (giống) |
| Avg SL loss | -5.8% | **-8.1%** |

Vẫn outperform VN-Index buy-hold cùng kỳ (~30%/16th, MDD -25%): +26pp return với MDD thấp hơn 8pp.

Chi tiết: `MIGRATION_NOTES.md`. Reproducible: `scripts/compare_live_pipeline_intraday_cached_v2.py --mode v1_fixed_bias`.

---

**Last Updated**: 2026-05-14 (session: backtest bias discovery + fix; baseline benchmark sửa từ +101% phóng đại xuống +56% thực; added MIGRATION_NOTES.md)