# Money Cycle Research Module

## Overview

Compute two key market-structure indicators from OHLCV data:

1. **CHDM** (Money-In Position) — Where price stands within N-period range
2. **DS** (Distribution Share) — What fraction of stocks show selling pressure per N-period cycle

---

## Concepts

### CHDM — Position in Cycle

For each stock, each day, each window N:

```
CHDM_N = 100 × (Close - LowestLow_N) / (HighestHigh_N - LowestLow_N)
```

- **Range**: 0–100
- **CHDM near 0**: Price at bottom of N-period range (accumulation zone)
- **CHDM near 100**: Price at top of N-period range (distribution zone)
- **CHDM ~50**: Price mid-range

**Market level**: Median of all stocks (reduces outlier influence)

### DS — Distribution Pressure

For each stock, each day, each window N:

```
MA_N = Simple Moving Average of Close over N periods
sell_state = (Close < MA_N) AND (MA_N declining)
DS_N = 1 if sell_state else 0
```

- **DS_N = 1**: Stock in sell/downtrend mode (below declining MA)
- **DS_N = 0**: Stock in uptrend or MA is rising

**Market level**: Mean of all stocks (proportion in sell state) → 0.0 to 1.0

**Interpretation**:
- DS near 0: Few stocks under distribution pressure
- DS near 1: Most stocks in decline/sell mode
- Rising DS: Deteriorating breadth
- Falling DS: Improving breadth or recovery

---

## Output Files

All files saved to `data/research/money_cycle/`:

| File | Rows | Columns | Description |
|---|---|---|---|
| `chdm_market.parquet` | 1,579 dates | CHDM03–CHDM200 | Market median CHDM per day |
| `ds_market.parquet` | 1,579 dates | DS03–DS200 | Market mean DS per day (0–1) |
| `money_cycle_market.parquet` | 1,579 dates | All CHDM + DS | Merged for easy plotting |
| `chdm_by_symbol.parquet` | 153k rows | symbol, CHDM* | Per-stock CHDM (raw data) |
| `ds_by_symbol.parquet` | 153k rows | symbol, DS* | Per-stock DS (0 or 1) |

---

## Usage

### CLI (Recommended)

**Default windows [3, 5, 10, 20, 50, 100, 200]:**

```bash
python -m multiagents_trading_assistant.research.money_cycle.cli
```

**Custom windows:**

```bash
python -m multiagents_trading_assistant.research.money_cycle.cli \
  --windows 5,10,20,50,100
```

**Custom input/output:**

```bash
python -m multiagents_trading_assistant.research.money_cycle.cli \
  --input path/to/ohlcv.parquet \
  --output path/to/output \
  --windows 10,50,200
```

### Python API

```python
from multiagents_trading_assistant.research.money_cycle import run_money_cycle_research

stats = run_money_cycle_research(
    input_path="multiagents_trading_assistant/data/ohlcv_master.parquet",
    output_dir="data/research/money_cycle",
    windows=[3, 5, 10, 20, 50, 100, 200],
)
```

### Load & Analyze Results

```python
import pandas as pd

# Load market-level indicators
mc = pd.read_parquet("data/research/money_cycle/money_cycle_market.parquet")

# Plot CHDM
import matplotlib.pyplot as plt
mc.plot(x="date", y=["CHDM03", "CHDM50", "CHDM200"], figsize=(12, 4))

# Identify chop/washout zones
washout = (mc["CHDM03"] < 20) & (mc["CDHM05"] < 20) & (mc["CHDM10"] < 20)
print(f"Washout days: {washout.sum()}")

# Monitor breadth deterioration
print(mc[["DS03", "DS50", "DS200"]].describe())
```

---

## Data Quality

| Aspect | Details |
|---|---|
| **Look-ahead bias** | None. Rolling windows use only current + past data |
| **NaN handling** | Insufficient data → NaN (e.g., CHDM03 NaN in first 2 rows) |
| **Symbols** | 100 (VN30 + VN100) |
| **Date range** | 2020-01-02 to 2026-05-06 (1,579 trading days) |
| **Rows** | 153,328 (100 symbols × 1,533 trading days) |

---

## Typical Workflow

1. **Backfill** (one-time):
   ```bash
   python -m multiagents_trading_assistant.research.money_cycle.cli
   ```

2. **Daily incremental** (can add to APScheduler):
   ```bash
   python -m multiagents_trading_assistant.research.money_cycle.cli
   # Re-run daily to update market files (overwrites with fresh computation)
   ```

3. **Visualization** (next step):
   - Plot 4-panel dashboard (like QMV/QMH)
   - Mark washout clusters, distribution tops
   - Overlay on price chart

4. **Backtesting** (downstream):
   - Use CHDM chop zones to filter for mean-reversion
   - Use high DS to avoid shorting, low DS for swing trades
   - Use cross-timeframe alignment (e.g., CHDM200 > 50 as long-term filter)

---

## Module Structure

```
multiagents_trading_assistant/research/money_cycle/
  __init__.py                    — Module exports
  compute_money_cycle.py         — Core computation functions
  cli.py                         — CLI entry point
```

### Key Functions

- `load_ohlcv(path)` — Load parquet/CSV
- `normalize_ohlcv_columns(df)` — Standardize column names
- `compute_chdm_by_symbol(df, windows)` — Per-stock CHDM
- `compute_ds_by_symbol(df, windows)` — Per-stock DS
- `compute_chdm_market(df, windows)` — Market median
- `compute_ds_market(df, windows)` — Market mean
- `run_money_cycle_research(input, output, windows)` — Full pipeline

---

## Notes

- **Performance**: ~30s for 100 symbols × 1,533 days on modern CPU
- **Storage**: ~7.5 MB total (parquet with compression)
- **Customization**: Easy to add new windows or aggregate differently
- **Extensible**: Can add weight-by-volume, sector-specific breakdowns, etc.

---

**Last Updated**: 2026-05-06  
**Status**: Production Ready
