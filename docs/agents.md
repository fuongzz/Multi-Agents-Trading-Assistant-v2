# Agents Specification

## Nguyên tắc chung

- Tất cả agents nhận `state: TradingState`, trả về `dict` merge vào state
- **Haiku** agents dùng `run_agent_lite()` — prompt caching enabled
- **Sonnet** agents dùng `run_agent()` — full reasoning
- **Agents KHÔNG gọi vnstock trực tiếp** — tất cả data qua `fetcher.py`
- Mọi output là JSON chuẩn — parse được ngay, không có prose thừa

---

## 1. Macro Agent (`agents/macro_agent.py`) — Haiku

**Chạy 1 lần/ngày, kết quả cache vào** `cache/macro_{date}.json`

**Input:** `fetcher.get_global_macro()` + `fetcher.get_vn_macro()` + headlines (optional)

**Output:**
```json
{
  "macro_score": 1,
  "macro_bias": "BULLISH",
  "key_risks": ["FED chưa cắt lãi suất", "DXY tăng mạnh"],
  "key_supports": ["SBV nới lỏng", "FDI tăng"],
  "global_summary": "S&P500 +0.8%, DXY 103.2, dầu ổn định...",
  "vn_summary": "USD/VND ổn định, lãi suất liên ngân hàng giảm nhẹ...",
  "overall_summary": "Môi trường vĩ mô trung tính nghiêng tích cực..."
}
```

**macro_bias:** `BULLISH` | `NEUTRAL` | `BEARISH`
→ `BEARISH`: toàn bộ pipeline dừng ngay

---

## 2. PTKT Agent (`agents/ptkt_agent.py`) — Haiku

**Input data từ fetcher + indicators:**
OHLCV 200 phiên, RSI(14), MA20/60/200, MACD(12,26,9), Bollinger(20,2), ATR(14), S/R levels

**Output:**
```json
{
  "rsi": 58.3,
  "rsi_signal": "NEUTRAL",
  "ma_trend": "UPTREND",
  "ma_phase": "HEALTHY",
  "macd_signal": "BULLISH",
  "bollinger_position": "MIDDLE",
  "atr": 1250,
  "support_levels": [71000, 69500],
  "resistance_levels": [75000, 78000],
  "confluence_score": 7,
  "setup_quality": "TỐT",
  "technical_summary": "Giá trên MA20/60/200, RSI trung tính, MACD cắt lên..."
}
```

**Confluence score (0–10):** +2 RSI 40-70 | +2 Giá > MA stack | +2 MACD tăng | +2 Volume > TB20 | +2 Gần support

---

## 3. FA Agent (`agents/fa_agent.py`) — Haiku

**Input:** P/E, P/B, ROE, EPS trailing, revenue/profit growth YoY, industry median

**Output:**
```json
{
  "pe_ratio": 12.5,
  "pb_ratio": 1.8,
  "roe": 18.2,
  "eps_growth_yoy": 15.3,
  "valuation": "HỢP_LÝ",
  "vs_industry": "TỐT_HƠN",
  "financial_health": "KHỎE",
  "fa_summary": "P/E 12.5 thấp hơn ngành 15.2, ROE 18% vượt trung bình..."
}
```

**valuation:** `RẺ` (P/E < 70% ngành) | `HỢP_LÝ` (70–130%) | `ĐẮT` (>130%)

---

## 4. Foreign Flow Agent (`agents/foreign_flow_agent.py`) — Haiku

**Input:** room_usage_pct, net_flow_5d, net_flow_20d, flow_history[]

**Output:**
```json
{
  "room_usage_pct": 87.3,
  "room_status": "HIGH",
  "net_flow_5d": -15200000000,
  "flow_trend": "SELLING",
  "accumulation_signal": false,
  "sizing_modifier": 0.8,
  "foreign_summary": "Room 87%, khối ngoại bán ròng 15.2 tỷ 5 phiên..."
}
```

**room_status + sizing_modifier:**
- `CRITICAL` >95% → Force CHỜ (Risk Manager xử lý)
- `HIGH` 90–95% → sizing × 0.5
- `MEDIUM` 80–90% → sizing × 0.8
- `NORMAL` <80% → sizing × 1.0

**accumulation_signal = True:** khối ngoại mua ròng khi VN-Index giảm

---

## 5. Sentiment Agent (`research/sentiment_agent.py`) — Haiku

**Input:** 10–20 bài CafeF + VnExpress (3 ngày gần nhất), từ `news_fetcher.py`

**Output:**
```json
{
  "sentiment_score": 72,
  "sentiment_label": "TÍCH_CỰC",
  "news_count": 14,
  "key_positive": ["Lợi nhuận Q1 tăng 20%", "Xuất khẩu kỷ lục"],
  "key_negative": ["Giá nguyên liệu tăng"],
  "sentiment_summary": "Tin tức tích cực chiếm ưu thế..."
}
```

**sentiment_label:** `TIÊU_CỰC` (0–30) | `TRUNG_TÍNH` (31–60) | `TÍCH_CỰC` (61–80) | `RẤT_TÍCH_CỰC` (81–100)

---

## 6. Debate Agent (`research/debate_agent.py`) — Sonnet

**Flow:**
```
Round 1: Bull argument (Sonnet) — dựa trên aggregate analyst output
Round 1: Bear counter (Sonnet)  — phản bác Bull
Round 2: Bull rebuttal
Round 2: Bear final position
→ Synthesizer (Haiku): tổng hợp điểm mạnh cả 2
```

**Synthesizer output:**
```json
{
  "bull_key_points": ["Kỹ thuật mạnh", "Khối ngoại tích lũy"],
  "bear_key_points": ["Định giá cao", "Macro bất lợi"],
  "balance": "BULL_SLIGHT_EDGE",
  "key_risk": "Lãi suất có thể ảnh hưởng định giá",
  "debate_conclusion": "Tiềm năng short-term nhưng cần theo dõi macro..."
}
```

**balance:** `STRONG_BULL` | `BULL_SLIGHT_EDGE` | `NEUTRAL` | `BEAR_SLIGHT_EDGE` | `STRONG_BEAR`

---

## 7. Trader Agent (`trader/trader_agent.py`) — Sonnet

**Input:** Toàn bộ TradingState (analysts + debate + market + macro)

**Output:**
```json
{
  "action": "MUA",
  "entry": 72500,
  "sl": 69000,
  "tp": 82000,
  "nav_pct": 5,
  "holding_period": "2–3 tuần",
  "confidence": "CAO",
  "primary_reason": "Pullback về MA60, volume cạn, setup MID_TERM",
  "risks": ["Nếu VN-Index break MA20 → thoát ngay"],
  "trader_note": "..."
}
```

**confidence → nav_pct tối đa:**
- `THẤP` → 3% NAV
- `TRUNG_BÌNH` → 5% NAV
- `CAO` → 8% NAV
- `RẤT_CAO` → 10% NAV

---

## 8. Risk Manager (`risk_manager.py`) — Rule-based, KHÔNG LLM

Xem đầy đủ → `docs/risk_rules.md`

Node cuối, có quyền override Trader Agent. Output:
```python
{
    "final_action":    "MUA" | "BÁN" | "CHỜ",
    "override_reason": str | None,
    "warnings":        list[str],
    "sizing_modifier": float   # 1.0 | 0.5 | 0.0
}
```
