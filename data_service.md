# Data Service — fetcher.py

## Nguyên tắc

**Agents KHÔNG BAO GIỜ import vnstock trực tiếp.**
Tất cả data đi qua `fetcher.py`. Lý do:
- Rate limit management tập trung
- Cache logic tập trung
- Dễ swap data source mà không sửa agents
- Fixes pandas 3.x / vnstock compatibility ở 1 chỗ duy nhất

---

## Required Header (PHẢI có ở đầu fetcher.py)

```python
import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
import pandas as pd

# FIX: pandas 3.x + vnstock compatibility
if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = pd.DataFrame.map
```

---

## Public API

```python
def get_ohlcv(symbol: str, n_days: int = 200) -> pd.DataFrame
    # columns: date, open, high, low, close, volume
    # cache: cache/{symbol}_{date}_ohlcv.json

def get_fundamentals(symbol: str) -> dict
    # keys: pe, pb, roe, eps, revenue_growth, profit_growth, industry
    # cache: 1 ngày

def get_foreign_flow(symbol: str, n_days: int = 20) -> dict
    # keys: room_usage_pct, net_flow_5d, net_flow_20d, flow_history[]
    # cache: 1 ngày

def get_vnindex(n_days: int = 200) -> pd.DataFrame
    # OHLCV của VN-Index

def get_global_macro() -> dict
    # keys: sp500, dxy, oil_wti, gold, nikkei, kospi, hsi (mỗi key: {current, change_pct})
    # source: yfinance | cache: 4 giờ

def get_vn_macro() -> dict
    # keys: usd_vnd, interbank_rate, sbv_rate
    # cache: 1 ngày

def get_ohlcv_batch(symbols: list[str], n_days: int = 200) -> dict[str, pd.DataFrame]
    # Tự động sleep(60) sau mỗi 15 mã
```

---

## Cache Strategy

```python
CACHE_DIR = Path("multiagents_trading_assistant/cache")

# Key pattern: {SYMBOL}_{YYYY-MM-DD}_{type}.json
# Ví dụ: VNM_2026-04-12_ohlcv.json

def _load_cache(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None
```

---

## Rate Limit

```python
BATCH_SIZE = 15
SLEEP_SECONDS = 60   # vnstock free tier

def get_ohlcv_batch(symbols, n_days=200):
    result = {}
    for i, symbol in enumerate(symbols):
        if i > 0 and i % BATCH_SIZE == 0:
            print(f"⏳ Sleep {SLEEP_SECONDS}s sau {i} mã...")
            time.sleep(SLEEP_SECONDS)
        result[symbol] = get_ohlcv(symbol, n_days)
    return result
```

---

## Data Sources

| Data | Source | Library |
|------|--------|---------|
| OHLCV VN stocks | VCI | vnstock==3.4.2 |
| Fundamentals | VCI | vnstock==3.4.2 |
| Foreign flow | VCI | vnstock==3.4.2 |
| Global macro | Yahoo Finance | yfinance |
| USD/VND | Yahoo Finance (VND=X) | yfinance |
| News | CafeF, VnExpress | beautifulsoup4 |

```python
# Luôn dùng VCI source
from vnstock import Vnstock
stock = Vnstock().stock(symbol=symbol, source="VCI")
```
