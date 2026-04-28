# AI Trading Assistant — Project Reference

**Dự án**: Multi-agent trading assistant cho thị trường chứng khoán Việt Nam  
**Stack**: Python 3.11, LangGraph, Anthropic Claude, vnstock, FiinQuantX  
**Phạm vi**: VN30 (30 mã)

---

## Kiến trúc Tổng Quan — Dual Pipeline

Hệ thống chạy **hai pipeline song song, hoàn toàn độc lập**, được trigger bởi APScheduler. Output gửi lên hai Discord channel riêng biệt.

```
APScheduler
  ├── Investment pipeline  [Thứ 2, 08:00]
  │     VN30 → Screener (FA criteria)
  │       → [Fundamental + Valuation + Macro] song song
  │       → Bull/Bear Debate (thesis dài hạn)
  │       → Trader Invest → Risk Manager Invest
  │       → #invest-signal (Discord)
  │
  └── Trade pipeline  [Hàng ngày, 08:30 trước ATO]
        VN30 → Screener (TA criteria)
          → [Technical + ForeignFlow + Sentiment] song song
          → Signal Synthesis (confluence scoring)
          → Trader Trade → Risk Manager Trade
          → #trade-signal (Discord)
```

---

## Nguyên Tắc Phân Biệt Hai Pipeline

| Chiều | Investment pipeline | Trade pipeline |
|---|---|---|
| Mục tiêu | Nắm giữ trung-dài hạn | Giao dịch ngắn hạn ATO |
| Khung thời gian | Tuần → Tháng → Quý | Phiên → Ngày → Tuần |
| Tín hiệu chính | BCTC, định giá nội tại, macro | Giá, khối lượng, dòng NN |
| Logic ra quyết định | Valuation gap (giá < nội tại) | Price action + confluence |
| Exit condition | Story thay đổi / định giá đủ | SL/TP hit / momentum mất |
| Risk logic | Margin of safety, concentration | R:R ratio, max loss/trade |
| T+3 / biên độ ±7% | Ít ảnh hưởng | Ràng buộc cứng, tính vào SL/TP |
| Screener criteria | ROE, P/E, tăng trưởng EPS | Breakout, volume spike, momentum |
| Tần suất chạy | Hàng tuần (Thứ 2) | Hàng ngày (trước ATO) |

---

## Cấu Trúc File

```
multiagents_trading_assistant/
  orchestrator/
    pipeline_runner.py        — APScheduler, chạy cả hai pipeline song song
    investment_graph.py       — StateGraph cho Investment pipeline
    trade_graph.py            — StateGraph cho Trade pipeline
  screener/
    invest_screener.py        — Lọc theo FA criteria (ROE, P/E, tăng trưởng)
    trade_screener.py         — Lọc theo TA criteria (breakout, volume, momentum)
  agents/
    invest/
      fundamental_agent.py   — BCTC, ROE, EPS growth (Haiku)
      valuation_agent.py     — Định giá nội tại, P/E so ngành (Haiku)
      macro_agent.py         — Vĩ mô, chu kỳ ngành (Haiku)
      debate_agent.py        — Bull/Bear thesis dài hạn (Sonnet)
    trade/
      technical_agent.py     — MA, RSI, MACD, pattern (Haiku)
      flow_agent.py          — Dòng tiền NN, net5d/net20d (Haiku)
      sentiment_agent.py     — Tin ngắn hạn CafeF (Haiku)
      synthesis_agent.py     — Confluence scoring (Haiku)
  nodes/
    trader_invest.py         — Exit: story thay đổi / định giá đủ (Sonnet)
    trader_trade.py          — Exit: SL/TP cụ thể theo giá (Sonnet)
    risk_invest.py           — Margin of safety, concentration check
    risk_trade.py            — R:R ratio >= 2, max loss/trade
  services/
    llm_service.py           — Provider abstraction (Haiku / Sonnet)
    data_service.py          — vnstock + FiinQuantX wrapper (dùng chung)
    memory_service.py        — ChromaDB operations (dùng chung)
    output_service.py        — send_to_channel(), send_pipeline_alert(), Discord + JSON output
  formatters/
    invest_embed.py          — Discord embed màu xanh, format weekly
    trade_embed.py           — Discord embed màu cam, format daily + SL/TP
  fetcher.py                 — Data layer gốc (giữ nguyên)
  main.py                    — CLI entry point
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
    fundamental_analysis: dict    # output fundamental_agent
    valuation_analysis: dict      # output valuation_agent
    bull_argument: str
    bear_argument: str
    debate_synthesis: str
    trader_decision: dict         # action, target_price, thesis, exit_condition
    risk_output: dict             # position_size, margin_of_safety
    discord_message: str
```

### TradeState (trade_graph.py)
```python
class TradeState(TypedDict):
    symbol: str
    date: str
    setup_type: str               # BREAKOUT | RETEST | SPRING | MA_PULLBACK | RSI_BOUNCE
    market_context: dict
    technical_analysis: dict      # output technical_agent
    foreign_flow_analysis: dict   # output flow_agent
    sentiment_analysis: dict      # output sentiment_agent
    confluence_score: float       # output synthesis_agent (0-100)
    trader_decision: dict         # action, entry_zone, stop_loss, take_profit
    risk_output: dict             # rr_ratio, max_loss_vnd, t3_constraint
    discord_message: str
```

---

## Models

| Agent | Model | Lý do |
|---|---|---|
| Fundamental, Valuation, Macro, Technical, Flow, Sentiment, Synthesis | `claude-haiku-4-5-20251001` | Nhanh + rẻ |
| Bull debate, Bear debate | `claude-sonnet-4-6` | Reasoning thesis sâu |
| Trader Invest, Trader Trade | `claude-sonnet-4-6` | Quyết định cuối |

---

## Data Sources & Phân công

| Dữ liệu | Nguồn | Dùng ở pipeline |
|---|---|---|
| OHLCV | vnstock VCI (primary), KBS (fallback) | Cả hai |
| PE/PB hiện tại | FiinQuantX `MarketDepth.get_stock_valuation()` | Investment |
| ROE/EPS/Growth | FiinQuantX `FundamentalAnalysis.get_financial_statement()` | Investment |
| Industry | FiinQuantX `BasicInfor.get()` → `icbNameL2` | Investment |
| Foreign flow net5d/net20d | FiinQuantX `Fetch_Trading_Data(fields=["fb","fs","fn"])` | Trade |
| Foreign room usage | vnstock `Trading.price_board()` | Trade |
| VN-Index / Macro | yfinance | Investment |
| News/Sentiment | CafeF crawl (Crawl4AI) | Trade |

---

## Screener

### invest_screener.py — FA criteria
- Lọc theo ROE > 15%, tăng trưởng EPS dương 2 quý liên tiếp
- P/E thấp hơn median ngành
- Loại cổ phiếu volume < 300k
- Output: top 5–7 candidates với `valuation_score`

### trade_screener.py — TA criteria (giữ nguyên 5 setup từ screener.py cũ)
| Setup | Logic |
|---|---|
| BREAKOUT | Giá vượt đỉnh 20 phiên + volume tăng |
| RETEST | Giá retest vùng breakout cũ (cách <3%) |
| SPRING | Higher High — đỉnh sau > đỉnh trước |
| MA_PULLBACK | Giá pullback về MA20/MA50 trong uptrend |
| RSI_BOUNCE | RSI oversold bounce từ vùng 30-40 |
- Output: top 10 candidates với `priority_score`

---

## API Lỗi Đã Biết

- **vnstock Finance (KBS)**: 404 — endpoint thay đổi, không dùng
- **vnstock `Vnstock(source='VCI')`**: `KeyError: 'data'` trong company init — không dùng
- **vnstock VN100 symbols API**: `KeyError: 'data'` — screener dùng danh sách cứng VN30

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
scheduler.add_job(run_investment_pipeline, CronTrigger(day_of_week='mon', hour=8, minute=0))

# Trade: Hàng ngày lúc 08:30 (trước ATO 09:00)
scheduler.add_job(run_trade_pipeline, CronTrigger(hour=8, minute=30))
```

---

## Cache

- Lưu tại `multiagents_trading_assistant/cache/`
- Key: `{symbol}_{date}_{pipeline_type}_{data_type}.json`
- Hết hạn sau mỗi ngày

---

## CLI

```bash
python -m multiagents_trading_assistant.main                      # cả hai pipeline
python -m multiagents_trading_assistant.main --pipeline invest    # chỉ investment
python -m multiagents_trading_assistant.main --pipeline trade     # chỉ trade
python -m multiagents_trading_assistant.main --symbol VCB --pipeline invest
python -m multiagents_trading_assistant.main --no-discord         # không gửi Discord
python test_vn30_data.py                                          # test data layer
python test_vn30_data.py --fresh                                  # xóa cache, lấy mới
```

---

## Ràng Buộc Thị Trường VN (áp dụng cả hai pipeline)

- **Biên độ giá**: ±7% (HOSE), ±10% (HNX) — tính vào SL/TP của trade pipeline
- **T+3 settlement**: mua hôm nay, nhận cổ phiếu sau 3 ngày — trade pipeline không exit ngay T+0
- **ATO session**: 09:00 HOSE — trade pipeline chạy lúc 08:30 để kịp đặt lệnh ATO
- **Foreign room**: check trước khi khuyến nghị cổ phiếu cho nhà đầu tư nước ngoài

---

**Last Updated**: 2026-04-19