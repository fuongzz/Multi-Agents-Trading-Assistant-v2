# AI Trading Assistant — Project Reference

**Dự án**: Multi-agent trading assistant cho thị trường chứng khoán Việt Nam  
**Stack**: Python 3.11, LangGraph, Anthropic Claude, DNSE LightSpeed API, vnstock, FiinQuantX  
**Phạm vi**: Toàn bộ HOSE (~400 mã, lọc thanh khoản ≥500k/ngày)

---

## Kiến trúc Tổng Quan — Dual Pipeline

Hệ thống chạy **hai pipeline song song, hoàn toàn độc lập**, được trigger bởi APScheduler. Output gửi lên hai Discord channel riêng biệt.

```
APScheduler
  ├── Investment pipeline  [Thứ 2, 08:00]
  │     HOSE liquid (vol≥300k) → Screener (FA criteria)
  │       → [Fundamental + Valuation + Macro] song song
  │       → Bull/Bear Debate (thesis dài hạn)
  │       → Trader Invest → Risk Manager Invest
  │       → #invest-signal (Discord)
  │
  ├── Trade pipeline  [Hàng ngày, 08:30 trước ATO]
  │     HOSE liquid (vol≥500k) → Screener (TA criteria)
  │       → [Technical + ForeignFlow + Sentiment] song song
  │       → Signal Synthesis (confluence scoring)
  │       → Trader Trade → Risk Manager Trade
  │       → #trade-signal (Discord)
  │
  └── Session Monitor  [09:00–14:35, mỗi 15 phút]
        Đọc MUA decisions hôm nay từ DB
          → Re-analyze từng mã (technical_agent Haiku)
          → Alert Discord nếu: confluence drop, setup flip, gần SL/TP
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
    risk_trade.py            — R:R ratio >= 1.5, max loss/trade (relative, không cần NAV tuyệt đối)
  services/
    llm_service.py           — Provider abstraction (Haiku / Sonnet)
    data_service.py          — vnstock + FiinQuantX wrapper (dùng chung)
    memory_service.py        — ChromaDB operations (dùng chung)
    output_service.py        — send_to_channel(), send_pipeline_alert(), Discord + JSON output
  formatters/
    invest_embed.py          — Discord embed màu xanh, format weekly
    trade_embed.py           — Discord embed màu cam, format daily + SL/TP
  services/
    dnse_client.py           — DNSE LightSpeed REST client (OHLCV history, instruments)
  fetcher.py                 — Data layer: DNSE primary, vnstock fallback
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
    setup_type: str               # 18 setups — xem trade_screener.py
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
| OHLCV lịch sử | **DNSE REST** `GET /price/ohlc` (primary) → vnstock VCI → KBS | Cả hai |
| Danh sách mã HOSE | vnstock `Listing().symbols_by_group("HOSE")` → DNSE instruments | Cả hai |
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

- **vnstock Finance (KBS)**: 404 — endpoint thay đổi, không dùng
- **vnstock `Vnstock(source='VCI')`**: `KeyError: 'data'` trong company init — không dùng
- **vnstock VN100 symbols API**: `KeyError: 'data'` — dùng `get_liquid_symbols()` thay thế
- **DNSE instruments `securityGroupId='EQ'`**: trả 400 BAD_REQUEST — filter bằng `len(sym)==3 and sym.isalpha()` ở Python
- **DNSE instruments `page=0`**: trả 400 — dùng `offset=0` thay thế
- **DNSE instruments** chỉ trả 252 mã 3-ký-tự trong khi HOSE có 402 — dùng vnstock Listing làm primary, DNSE làm fallback cho `get_all_symbols()`
- **Timestamp OHLCV từ DNSE**: trả UTC unix timestamp — cần convert sang Asia/Ho_Chi_Minh rồi `.normalize()` để ra date

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

| Rule | Ngưỡng | Kết quả |
|---|---|---|
| Circuit breaker | VNI < -3%/ngày | Override → CHỜ |
| Foreign room | >95% | Override → CHỜ |
| | 90-95% | Sizing ×0.5 |
| | 80-90% | Sizing ×0.8 |
| T+2.5 | Đã mua mã này trong 2 ngày | Override → CHỜ |
| Không mua đuổi | Mã tăng ≥5% trong phiên | Override → CHỜ |
| R:R tối thiểu | R:R < 1.5 | Override → CHỜ |
| Max loss | position_pct × (entry-SL)/entry > 2% | Giảm position_pct |

**Thiết kế sản phẩm**: max loss tính thuần relative (không cần NAV tuyệt đối) — áp dụng đúng cho mọi user.

## Trader Trade — Quy tắc LLM (trader_trade.py)

- **SL**: ≤ 5% dưới entry_low (không phải 7% — SL 7% → cần TP 14% mới đạt R:R 2, không thực tế VN)
- **R:R**: ≥ 1.5 trong prompt (risk_trade enforce cứng)
- **Position theo confluence**: ≥70 → 5% NAV, 55-69 → 3%, <55 → 2%
- **Holding horizon**: 1-5 phiên (không phải 1-4 tuần)
- **Entry guide**: mỗi trong 18 setups có hướng dẫn entry/SL/TP riêng trong `_build_prompt()`

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

**Last Updated**: 2026-04-30 (session: 4 PA setups, backtest module, backtest VN30 2023-2024)