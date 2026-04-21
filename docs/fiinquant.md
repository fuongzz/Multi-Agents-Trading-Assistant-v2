# FiinQuantX — Reference Guide

**Package**: `fiinquantx` (import as `FiinQuantX`)  
**Version**: 0.1.53  
**Install**:
```bash
pip install --extra-index-url https://fiinquant.github.io/fiinquantx/simple fiinquantx
```

> **Lưu ý**: KHÔNG đặt tên file Python trùng tên thư viện (FiinQuant).

---

## 1. Authentication

```python
from FiinQuantX import FiinSession

client = FiinSession(username="YOUR_USER", password="YOUR_PASS").login()
```

`client` là entry point cho toàn bộ API. Tất cả các method bên dưới đều gọi qua `client`.

---

## 2. Danh sách mã (TickerList)

```python
# Theo index
tickers = client.TickerList(ticker="VN30")    # list[str], 30 mã
tickers = client.TickerList(ticker="VN100")
tickers = client.TickerList(ticker="VNINDEX")
tickers = client.TickerList(ticker="HNXINDEX")

# Theo ngành (ICB)
tickers = client.TickerList(ticker="BANKS_L2")
tickers = client.TickerList(ticker="REAL_ESTATE_L2")
tickers = client.TickerList(ticker="STEEL_L4")

# Truyền nhiều mã thủ công
tickers = client.TickerList(tickers=["VCB", "HPG", "FPT"])
```

**Giới hạn**: tối đa 33 mã mỗi lần gọi dữ liệu giao dịch.

---

## 3. Dữ liệu giao dịch (Fetch_Trading_Data)

Hàm chính — lấy OHLCV + mua/bán chủ động + khối ngoại cho nhiều mã cùng lúc.

### Signature

```python
data = client.Fetch_Trading_Data(
    realtime   = False,           # True = realtime stream, False = historical
    tickers    = list[str],       # tối đa 33 mã
    fields     = list[str],       # xem bảng fields bên dưới
    adjusted   = True,            # True = giá đã điều chỉnh
    by         = "1d",            # timeframe: "1m","5m","15m","30m","1h","2h","4h","1d"
    from_date  = "2025-01-01",    # ngày bắt đầu (lịch sử tối đa 1 năm)
    to_date    = None,            # None = hôm nay
    period     = None,            # số nến gần nhất (thay thế from_date)
    lasted     = None,            # False = nến cuối là hoàn chỉnh (chỉ dùng khi realtime=False)
).get_data()                      # trả về pd.DataFrame
```

### Fields

| Field    | Mô tả                        |
|----------|------------------------------|
| `open`   | Giá mở cửa                   |
| `high`   | Giá cao nhất                 |
| `low`    | Giá thấp nhất                |
| `close`  | Giá đóng cửa                 |
| `volume` | Khối lượng khớp lệnh         |
| `bu`     | Mua chủ động (Buy Up)        |
| `sd`     | Bán chủ động (Sell Down)     |
| `fb`     | Khối ngoại mua (Foreign Buy) |
| `fs`     | Khối ngoại bán (Foreign Sell)|
| `fn`     | Khối ngoại ròng (Foreign Net)|
| `"full"` | Tất cả fields trên           |

### Ví dụ — lịch sử 30 mã VN30

```python
tickers = client.TickerList(ticker="VN30")

df = client.Fetch_Trading_Data(
    realtime  = False,
    tickers   = tickers,
    fields    = ["open", "high", "low", "close", "volume", "bu", "sd", "fb", "fs", "fn"],
    adjusted  = True,
    by        = "1d",
    from_date = "2025-01-01",
).get_data()
# columns: ticker, time, open, high, low, close, volume, bu, sd, fb, fs, fn
```

### Ví dụ — realtime stream

```python
def on_data(bar):          # bar: BarDataUpdate
    print(bar.to_dataFrame())

stream = client.Fetch_Trading_Data(
    realtime              = True,
    tickers               = ["VCB", "HPG"],
    fields                = ["close", "volume", "bu", "sd"],
    adjusted              = True,
    by                    = "1m",
    callback              = on_data,
    wait_for_full_timeFrame = False,
)
stream.start()
# ... stream.stop() để dừng
```

---

## 4. Realtime stream — Trading_Data_Stream

Stream tick-by-tick matching data (không phải nến).

```python
def on_tick(data):         # data: RealTimeData
    d = data.to_dict()
    print(d["Ticker"], d["Close"], d["Bu"], d["Sd"])

stream = client.Trading_Data_Stream(
    tickers  = ["VCB", "HPG", "FPT"],
    callback = on_tick,
)
stream.start()
# ... stream.stop()
```

**`RealTimeData` fields quan trọng**:
`Ticker`, `Close`, `Open`, `High`, `Low`, `TotalMatchVolume`, `MatchVolume`, `MatchValue`,
`Bu`, `Sd`, `ForeignBuyVolumeTotal`, `ForeignSellVolumeTotal`, `TotalBuyTradeVolume`, `TotalSellTradeVolume`

---

## 5. Sổ lệnh — BidAsk (realtime)

```python
def on_bidask(data):       # data: BidAskData
    d = data.to_dict()
    print(d["Ticker"], d["Best1Bid"], d["Best1Ask"])

stream = client.BidAsk(
    tickers  = ["VCB", "HPG"],
    callback = on_bidask,
)
stream.start()
# ... stream.stop()
```

**`BidAskData` fields**: `Best1–10Bid/Ask`, `Best1–10BidVolume/AskVolume`,
`TotalBidVolume`, `TotalAskVolume`, `OrderFlowImbalance`, `DepthImbalance`,
`VWAPBid`, `VWAPAsk`, `Spread`

---

## 6. PriceStatistics

```python
ps = client.PriceStatistics()

# Tổng quan giá (OHLCV + foreign theo kỳ)
df = ps.get_overview(
    tickers     = ["VCB", "HPG"],
    time_filter = "Daily",   # "Daily","Weekly","Monthly","Quarterly","Yearly"
    from_date   = "2025-01-01",
    to_date     = "2025-12-31",
)

# Giao dịch nước ngoài theo kỳ
df = ps.get_foreign(
    tickers     = ["VCB", "HPG"],
    time_filter = "Daily",
    from_date   = "2025-01-01",
)

# Giá trần/sàn lịch sử
df = ps.get_ceilingfloor(
    tickers   = ["VCB"],
    from_date = "2025-01-01",
)

# Giá trị giao dịch theo nhà đầu tư (tổ chức / cá nhân / nước ngoài)
df = ps.get_value_by_investor(
    tickers   = ["VCB", "HPG"],
    from_date = "2025-01-01",
)

# FreeFloat (tỷ lệ cổ phiếu tự do giao dịch)
df = ps.get_freefloat(
    tickers   = ["VCB", "HPG"],
    from_date = "2025-01-01",
)
```

---

## 7. MarketDepth — Định giá

```python
md = client.MarketDepth()

# Định giá từng cổ phiếu (P/E, P/B, EV/EBITDA, ...)
df = md.get_stock_valuation(
    tickers   = ["VCB", "HPG"],
    from_date = "2025-01-01",
    to_date   = "2025-12-31",
)

# Định giá ngành
df = md.get_sector_valuation(
    tickers   = ["BANKS_L2"],
    level     = 2,
    from_date = "2025-01-01",
)

# Định giá chỉ số
df = md.get_index_valuation(
    tickers   = ["VNINDEX", "VN30"],
    from_date = "2025-01-01",
)
```

---

## 8. MarketBreadth

```python
mb = client.MarketBreadth()
df = mb.get(tickers=["VNINDEX", "VN30", "HNX30"])
# Breadth: số mã tăng/giảm/đứng, tăng trần/giảm sàn
```

---

## 9. MoneyFlow

```python
mf = client.MoneyFlow()
df = mf.get_contribution(
    ticker           = "VNINDEX",
    contribution_day = "1Day",   # "1Day","5Day","10Day","20Day"
    type             = "topGainers",  # "topGainers","topLosers"
    top              = 15,
)
```

---

## 10. BasicInfor

```python
df = client.BasicInfor(tickers=["VCB", "HPG", "FPT"]).get()
# Tên DN, sàn GD, ngành ICB, ...
```

---

## 11. FundamentalAnalysis

```python
fa = client.FundamentalAnalysis()

# Báo cáo tài chính
df = fa.get_financial_statement(
    tickers   = ["VCB"],
    statement = "incomestatement",  # "balancesheet","incomestatement","cashflow","full"
    years     = [2023, 2024],
    type      = "consolidated",     # "consolidated","separate"
    quarters  = [1, 2, 3, 4],       # None = cả năm
)

# Chỉ số tài chính (P/E, ROE, ROA, ...)
df = fa.get_ratios(
    tickers  = ["VCB", "HPG"],
    years    = [2023, 2024],
    type     = "consolidated",
)
```

---

## 12. StockScreening

```python
sc = client.StockScreening()

# Lấy danh sách tên ngành theo level
icb_list = sc.get_icb_name_list(level=2)

# Lọc cổ phiếu theo tiêu chí
results = sc.get(
    filter        = [{"field": "pe", "operator": "<", "value": 15}],
    exchanges     = ["HOSE", "HNX"],
    screenerDate  = "2025-06-01",
    sectors       = ["BANKS_L2"],
)
```

---

## 13. RRG (Relative Rotation Graph)

```python
rrg = client.RRG(
    tickers   = ["VCB", "HPG", "FPT", "MBB"],
    benchmark = "VNINDEX",
    by        = "1d",
    from_date = "2025-01-01",
    period    = 52,
)

df    = rrg.get()   # RS-Ratio, RS-Momentum, phase cho từng mã
plot  = rrg.plot(latest_only=True)

# Filter theo phase
leading = rrg.filter(phase="Leading")
```

**Phases**: `"Leading"`, `"Weakening"`, `"Lagging"`, `"Improving"`

---

## 14. OrderBook — Theo dõi sổ lệnh

```python
ob = client.OrderBook()

ob.track_order_book_changes(
    ticker_config    = {"VCB": 48.5, "HPG": 27.0},  # {ticker: giá tham chiếu}
    callback         = lambda df: print(df),
    side             = "all",       # "buy","sell","all"
    action           = "cancel",    # "add","cancel"
    accumulate_window = 5,          # giây tích lũy
)
```

---

## 15. FiinIndicator (Technical Analysis)

```python
fi = client.FiinIndicator()

# Trend
fi.ema(close, window=20)
fi.sma(close, window=50)
fi.macd(close)
fi.macd_signal(close)
fi.adx(high, low, close, window=14)
fi.supertrend(high, low, close, multiplier=3.0)
fi.ichimoku_a / ichimoku_b / ichimoku_base_line / ichimoku_conversion_line(high, low, close)
fi.hma(close, period=9)
fi.alma(close, period=9)
fi.jma(close, length=7)

# Momentum
fi.rsi(close, window=14)
fi.stoch(high, low, close)
fi.williams_r(high, low, close)
fi.roc(close, window=12)
fi.coppock_curve(close)
fi.awesome_oscillator(high, low)

# Volatility
fi.bollinger_hband / bollinger_lband / bollinger_mband(close, window=20)
fi.bollinger_squeeze(close)
fi.atr(high, low, close)
fi.kc_upper / kc_lower / kc_midline(close, high, low)
fi.realized_volatility(close)

# Volume
fi.obv(close, volume)
fi.vwap(high, low, close, volume)
fi.mfi(high, low, close, volume)
fi.volume_profile(high, low, close, volume, bins=20)
fi.poc(close, volume)          # Point of Control

# Smart Money Concepts (SMC)
fi.fvg(open, high, low, close)                    # Fair Value Gap
fi.swing_HL(open, high, low, close)               # Swing High/Low
fi.break_of_structure(open, high, low, close)     # BOS
fi.chage_of_charactor(open, high, low, close)     # CHoCH
fi.liquidity(open, high, low, close)
fi.ob(open, high, low, close, volume)             # Order Blocks

# Money Flow
fi.mcdx_banker(close)       # Multi Color Dragon — Banker
fi.mcdx_hot_money(close)    # Hot Money
fi.mcdx_retail(close)       # Retail
fi.cmf(high, low, close, volume)  # Chaikin Money Flow

# Price Level
fi.fib_retracement_uptrend(high, low)    # dict các mức Fibonacci
fi.fib_retracement_downtrend(high, low)
```

---

## 16. Giới hạn dữ liệu

| Loại         | Số mã tối đa | Timeframe   | Lịch sử tối đa |
|--------------|-------------|-------------|----------------|
| Realtime     | 33          | 1m–4h, 1d   | 1 tháng        |
| Lịch sử      | 33          | 1d          | 1 năm          |

---

## 17. Data classes

### BarDataUpdate (từ Fetch_Trading_Data realtime callback)
```
timestamp, open, high, low, close, volume, ticker, bu, sd, fb, fs, fn
→ .to_dataFrame() → pd.DataFrame
```

### RealTimeData (từ Trading_Data_Stream callback)
```
Ticker, Open, High, Low, Close, TotalMatchVolume, MatchVolume, MatchValue,
TotalMatchValue, Bu, Sd, ForeignBuyVolumeTotal, ForeignSellVolumeTotal,
ForeignBuyValueTotal, ForeignSellValueTotal, TotalBuyTradeVolume,
TotalSellTradeVolume, ReferencePrice, Change, ChangePercent
→ .to_dict() | .to_dataFrame()
```

### BidAskData (từ BidAsk callback)
```
Ticker, Best1–10Bid, Best1–10Ask, Best1–10BidVolume, Best1–10AskVolume,
TotalBidVolume, TotalAskVolume, OrderFlowImbalance, DepthImbalance,
VWAPBid, VWAPAsk, VWAPBidAskSpread, Spread, SpreadDelta
→ .to_dict() | .to_dataFrame()
```
