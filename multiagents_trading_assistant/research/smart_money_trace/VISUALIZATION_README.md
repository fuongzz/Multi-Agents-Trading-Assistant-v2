# Smart Money Trace — Interactive Visualization

Generate interactive Plotly HTML dashboards để analyze Smart Money Trace indicators.

## Quick Start

### 1. Generate Market Dashboard

Market dashboard thể hiện:
- **Panel 1**: Smart Money Score (avg + median)
- **Panel 2**: Smart Money States distribution (HOT_BUT_STRONG / HOT_AND_EXHAUSTED / HOT_AND_DISTRIBUTING / RESET_IN_UPTREND)
- **Panel 3**: Accumulation vs Distribution Days
- **Panel 4**: Accumulation/Distribution Ratio

```bash
python -m multiagents_trading_assistant.research.smart_money_trace.viz_interactive_cli
```

Output: `data/research/smart_money_trace/interactive/01_market_dashboard_interactive.html`

### 2. Generate Symbol Dashboards — Single Symbol

Generate dashboard cho 1 symbol (VCB):

```bash
python -m multiagents_trading_assistant.research.smart_money_trace.viz_interactive_cli --symbol VCB
```

Output: `data/research/smart_money_trace/interactive/02_symbol_dashboard_VCB_interactive.html`

**4-panel per symbol dashboard:**
- **Panel 1**: Price + MA20/MA50/MA200
- **Panel 2**: Smart Money Score (colored by state)
- **Panel 3**: RS Score + CLV Score
- **Panel 4**: A/D Days + A/D Balance

### 3. Generate Multiple Symbols

```bash
python -m multiagents_trading_assistant.research.smart_money_trace.viz_interactive_cli --symbols VCB,FPT,VNM,ACB
```

### 4. Generate All VN30 Symbols

```bash
python -m multiagents_trading_assistant.research.smart_money_trace.viz_interactive_cli --universe vn30
```

**Pro tip**: Use `--skip-market` to only generate symbol dashboards (faster):

```bash
python -m multiagents_trading_assistant.research.smart_money_trace.viz_interactive_cli --universe vn30 --skip-market
```

### 5. Custom Data Directory

```bash
python -m multiagents_trading_assistant.research.smart_money_trace.viz_interactive_cli \
  --data-dir data/research/smart_money_trace \
  --output-dir data/research/smart_money_trace/interactive_custom \
  --universe vn100
```

## Interactive Features

Mở bất kỳ `.html` file trong browser:

| Tương tác | Cách dùng |
|---|---|
| **Zoom** | Scroll wheel lên/xuống |
| **Pan** | Click and drag để di chuyển |
| **Hover** | Trỏ chuột lên line để xem giá trị |
| **Toggle line** | Click legend item để ẩn/hiện line |
| **Download PNG** | Click camera icon trong toolbar |
| **Reset view** | Double click trên chart |

## Python API

```python
from multiagents_trading_assistant.research.smart_money_trace import run_interactive_charts

# Generate all charts
run_interactive_charts(
    data_dir="data/research/smart_money_trace",
    output_dir="data/research/smart_money_trace/interactive",
    symbols=["VCB", "FPT", "VNM"],
    skip_market=False,
)
```

### Generate Market Dashboard Only

```python
from pathlib import Path
import pandas as pd
from multiagents_trading_assistant.research.smart_money_trace import plot_market_dashboard_interactive

summary = pd.read_parquet("data/research/smart_money_trace/smart_money_daily_summary.parquet")
plot_market_dashboard_interactive(
    summary,
    Path("output.html")
)
```

### Generate Symbol Dashboard

```python
import pandas as pd
from multiagents_trading_assistant.research.smart_money_trace import plot_symbol_dashboard_interactive

by_symbol = pd.read_parquet("data/research/smart_money_trace/smart_money_by_symbol.parquet")
plot_symbol_dashboard_interactive(
    by_symbol,
    "VCB",
    Path("vcb_dashboard.html")
)
```

## Data Requirements

Cần có các file parquet trong `data/research/smart_money_trace/`:

- `smart_money_daily_summary.parquet` — tóm tắt hàng ngày
- `smart_money_by_symbol.parquet` — raw data per symbol

Nếu chưa có, chạy:

```bash
python -m multiagents_trading_assistant.research.smart_money_trace.cli \
  --input <OHLCV.parquet> \
  --output data/research/smart_money_trace
```

## Smart Money States Explained

| State | Meaning | Action |
|---|---|---|
| **HOT_BUT_STRONG** | Giá cao, dòng tiền smart vẫn mạnh | HOLD or BUY_PULLBACK |
| **HOT_AND_EXHAUSTED** | Giá cao, dòng tiền yếu | TIGHTEN_STOP, avoid chase |
| **HOT_AND_DISTRIBUTING** | Giá cao, dấu hiệu phân phối | REDUCE or EXIT |
| **RESET_IN_UPTREND** | Pull back trong uptrend lớn | BUY_PULLBACK_WATCHLIST |
| **NEUTRAL** | Không rõ signal | WAIT |

## Output Locations

```
data/research/smart_money_trace/
├── interactive/
│   ├── 01_market_dashboard_interactive.html
│   ├── 02_symbol_dashboard_VCB_interactive.html
│   ├── 02_symbol_dashboard_FPT_interactive.html
│   └── ...
├── smart_money_by_symbol.parquet
├── smart_money_daily_summary.parquet
└── smart_money_hot_states.parquet
```

## Troubleshooting

### Missing data for symbol

Nếu symbol không có data, log sẽ warning. Kiểm tra:
- Symbol có tồn tại trong `smart_money_by_symbol.parquet` không?
- Date range có data không?

```python
import pandas as pd
df = pd.read_parquet("data/research/smart_money_trace/smart_money_by_symbol.parquet")
print(df[df["symbol"] == "VCB"].head())
```

### Charts not rendering

- Kiểm tra browser có hỗ trợ Plotly không (hiện đại hầu hết đều hỗ trợ)
- Thử dùng Chrome / Firefox / Safari
- File `.html` có lớn, có thể mất vài giây để load

### Custom styling

Edit `visualization_interactive.py` để thay đổi colors, layouts, v.v.:
- `go.Scatter(..., line=dict(color="..."))` — change line color
- `fig.update_layout(...)` — change global layout
- `fig.add_hline(y=...)` — add reference lines

---

**Last Updated**: 2026-05-07
