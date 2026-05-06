# OHLCV Standardized Data Store

## Overview

Standardized Parquet-based data warehouse for OHLCV data (VN30 + VN100). Single master file with 10 standardized columns, supporting incremental daily updates.

## Schema

```
date       datetime64[ns]   — trading date (Asia/Ho_Chi_Minh, normalized)
symbol     str              — stock symbol (uppercase, 3-char)
open       float64
high       float64
low        float64
close      float64
volume     int64            — trading volume (shares)
value      float64          — trading value = volume × close (VND)
exchange   str              — "HOSE" or "HNX"
industry   str              — ICB L2 industry name (e.g. "Banks", "Real Estate")
```

## Storage

**File**: `multiagents_trading_assistant/data/ohlcv_master.parquet`
- ~160k rows (130 symbols × 5 years × 250 trading days)
- ~5MB (snappy compressed)
- Single file, read-modify-write for incremental append

## Getting Started

### 1. Initial Backfill (One-Time)

Fetch full historical data from 2020 onwards. Takes ~10–15 minutes:

```bash
python scripts/build_ohlcv_history.py --start 2020-01-01 --workers 10
```

**Options**:
- `--start YYYY-MM-DD` — start date (default: 2020-01-01)
- `--end YYYY-MM-DD` — end date (default: today)
- `--workers N` — parallel fetch threads (default: 10)
- `--clear-first` — delete existing parquet before backfill

**Output**:
```
OK: Processed 130/130 symbols
  Successful: 130
  Total rows: 162,500
  Date range: 2020-01-01 → 2026-05-06
```

### 2. Daily Incremental Update

Runs automatically after market close (15:35) via APScheduler. Can also run manually:

```bash
python scripts/update_ohlcv_daily.py
```

**Logic**:
1. Check store's last date
2. If last_date == today → skip (already updated)
3. Fetch data from (last_date + 1) → today
4. Append to parquet

**Scheduler job** (in `pipeline_runner.py`):
- **When**: Monday–Friday 15:35 (after market close 15:30)
- **ID**: `ohlcv_daily_update`

### 3. Query Data

```python
from multiagents_trading_assistant.data import ohlcv_store

# Load all data
df = ohlcv_store.load()

# Filter by symbols
df = ohlcv_store.load(symbols=["VCB", "FPT", "VHM"])

# Filter by date range
from datetime import date
df = ohlcv_store.load(
    symbols=["VCB"],
    start=date(2024, 1, 1),
    end=date(2024, 12, 31)
)

# Get last date in store
last = ohlcv_store.get_last_date()
last_vcb = ohlcv_store.get_last_date("VCB")

# Append new data (used by daily update)
rows_added = ohlcv_store.append(df_new)
```

## Data Quality

- **Deduplication**: Rows are deduplicated by (date, symbol) — latest version kept
- **Exchange**: Fetched from `Reference.equity.by_exchange()`
- **Industry**: Fetched from `Reference.equity.list_by_industry()` (ICB L2)
- **Value**: Calculated as `volume × close` (approximation)
- **Timezone**: All dates normalized to Asia/Ho_Chi_Minh (UTC+7)

## Integration with Pipelines

Currently used by:
- (Future: Analysis pipelines can query `ohlcv_store.load()` for batch OHLCV)
- (Future: Backtester can use unified OHLCV source)

## Troubleshooting

### No data returned from backfill
- Check vnstock_data availability: `python -c "from vnstock_data import Market; m = Market().equity('VCB').ohlcv('2024-01-01', '2024-12-31'); print(m.shape)"`
- Check DNSE fallback: make sure `DNSE_API_KEY` is set in `.env`

### Parquet file is huge / slow
- Current size: ~5MB (normal)
- If >100MB: run `ohlcv_store.clear()` and backfill fresh
- Parquet is column-oriented → fast filtering by date/symbol

### Missing industry for some symbols
- Fallback: `"Unknown"`
- `list_by_industry()` may not return all symbols
- Safe to proceed — filter or use "Unknown" as placeholder

## Files

- `multiagents_trading_assistant/data/ohlcv_store.py` — Storage layer (load, append, get_last_date)
- `scripts/build_ohlcv_history.py` — One-time backfill script
- `scripts/update_ohlcv_daily.py` — Daily incremental script
- `multiagents_trading_assistant/orchestrator/pipeline_runner.py` — Scheduler integration

---

**Status**: Ready for production  
**Last Updated**: 2026-05-06
