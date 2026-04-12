# Screener — Top-Down Gate

## Vai trò

Screener là gate chạy **trước** toàn bộ multi-agent pipeline.
Không pass screener → không tốn một đồng API cost nào.

```
VN100 (100 mã)
  → Market Gate      → DOWNTREND: dừng tất cả, output CHỜ
  → Sector Filter    → Chỉ giữ sector có RS > 0 (outperform)
  → Stock Filter     → Chỉ giữ mã có setup rõ ràng
  → candidates[]     → Tối đa 10 mã xuống multi-agent
```

---

## Step 1 — Market Context (VN-Index)

```python
def get_market_context(vnindex_df: pd.DataFrame) -> MarketContext:
    close = vnindex_df["close"]
    ma20  = close.rolling(20).mean().iloc[-1]
    ma60  = close.rolling(60).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    current = close.iloc[-1]

    if current > ma20 > ma60 > ma200:
        trend = "UPTREND"
    elif current < ma20 < ma60:
        trend = "DOWNTREND"
    else:
        trend = "SIDEWAY"

    should_trade = (trend != "DOWNTREND")

    # Position trong range 20 phiên (dùng cho đồng pha SIDEWAY)
    high_20 = close.rolling(20).max().iloc[-1]
    low_20  = close.rolling(20).min().iloc[-1]
    rng = high_20 - low_20
    pos = (current - low_20) / rng if rng > 0 else 0.5

    if pos <= 0.25:   sideway_zone = "NEAR_SUPPORT"
    elif pos >= 0.75: sideway_zone = "NEAR_RESISTANCE"
    else:             sideway_zone = "MID_RANGE"

    return MarketContext(trend, should_trade, ma20, ma60, ma200, sideway_zone, pos)
```

---

## Step 2 — Uptrend Phase

```python
def get_uptrend_phase(close, rsi, ma20) -> str:
    current = close.iloc[-1]
    dist_ma20 = (current - ma20) / ma20 * 100

    if rsi > 75 or dist_ma20 > 10:
        return "OVERBOUGHT"   # Không vào mới
    elif current < ma20 * 1.02:
        return "PULLBACK"     # MID_TERM opportunity
    else:
        return "HEALTHY"      # SHORT_TERM opportunity
```

---

## Step 3 — Stock Setup Classification

### UPTREND setups

```python
# SHORT_TERM — Breakout
if (volume_surge      # volume > 1.5× TB20
    and breakout_20d  # vượt đỉnh 20 phiên cũ
    and big_candle):  # thân nến > 2%
    return "SHORT_TERM"

# MID_TERM — Pullback
if (pullback_pct >= 8          # giảm ≥8% từ đỉnh
    and volume_declining       # volume cạn dần
    and near_ma60):            # cách MA60 < 3%
    return "MID_TERM"
```

### SIDEWAY setups (ĐỒNG PHA — quan trọng)

```python
pos = (current - low_20) / (high_20 - low_20)

if pos <= 0.25:   # Cổ gần support
    if market_ctx.sideway_zone == "NEAR_SUPPORT":
        return "SIDEWAY_BUY"   # Đồng pha → MUA
    else:
        return None            # Không đồng pha → BỎ QUA

elif pos >= 0.75:  # Cổ gần resistance
    return "SIDEWAY_SELL"      # Không cần VN-Index xác nhận

else:
    return "MID_RANGE"         # Chờ về biên
```

> ⚠️ **SIDEWAY anti-pattern:** Breakout trong sideway = BẪY. Không đuổi theo.

---

## Step 4 — Sector Filter

```python
def get_outperform_sectors(sector_map, vnindex_df, n=20) -> set[str]:
    vni_ret = vnindex_df["close"].iloc[-1] / vnindex_df["close"].iloc[-n] - 1
    result = set()
    for sector, symbols in sector_map.items():
        sector_ret = mean([get_return(s, n) for s in symbols])
        if sector_ret - vni_ret > 0:   # outperform → giữ
            result.add(sector)
    return result
```

---

## Output

```python
@dataclass
class CandidateStock:
    symbol: str
    setup_type: str       # SHORT_TERM | MID_TERM | SIDEWAY_BUY | SIDEWAY_SELL
    sector: str
    sector_rs: float
    priority_score: float
    market_context: MarketContext

# Trả về tối đa 10 mã, sorted by priority_score DESC
```

---

## Anti-patterns — KHÔNG làm

- ❌ Vào SIDEWAY_BUY khi VN-Index đang MID_RANGE hoặc NEAR_RESISTANCE
- ❌ Đuổi theo breakout trong sideway
- ❌ Chạy multi-agent khi `should_trade = False`
- ❌ Fetch OHLCV từng mã trước khi pass market gate
- ❌ Fetch hơn 15 mã liên tiếp không sleep (rate limit)
