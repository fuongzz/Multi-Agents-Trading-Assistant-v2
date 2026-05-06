# OHLCV Data Store — Quick Start

## Status: ✅ Live & Tested

Your standardized OHLCV data store is now operational with ~33k rows (100 symbols, 2025–2026).

## What Was Created

### 1. Storage Layer
- **File**: `multiagents_trading_assistant/data/ohlcv_store.py`
- **Schema**: date, symbol, open, high, low, close, volume, value, exchange, industry
- **Storage**: Single Parquet file (`ohlcv_master.parquet` — 814 KB)

### 2. Backfill Script
- **File**: `scripts/build_ohlcv_history.py`
- **Status**: ✅ Tested (100 symbols, 32,808 rows imported)
- **Runtime**: ~5-10 minutes for full 2020–present (with vnstock_data)

```bash
python scripts/build_ohlcv_history.py --start 2020-01-01 --workers 10
```

### 3. Daily Incremental Update
- **File**: `scripts/update_ohlcv_daily.py`
- **Trigger**: APScheduler job at 15:35 (Mon–Fri, after market close)
- **Manual run**: `python scripts/update_ohlcv_daily.py`

### 4. Scheduler Integration
- **File**: `multiagents_trading_assistant/orchestrator/pipeline_runner.py`
- **Job ID**: `ohlcv_daily_update`
- **Schedule**: 15:35 (Asia/Ho_Chi_Minh timezone)

---

## Usage Examples

### Load All Data
```python
from multiagents_trading_assistant.data import ohlcv_store

df = ohlcv_store.load()
print(df.shape)  # (32808, 10)
```

### Filter by Symbols
```python
vcb_fpt = ohlcv_store.load(symbols=['VCB', 'FPT'])
# Returns 2 symbols × ~329 rows each = ~658 rows
```

### Filter by Date Range
```python
from datetime import date

q1_2025 = ohlcv_store.load(
    start=date(2025, 1, 1),
    end=date(2025, 3, 31)
)
```

### Combined Filter
```python
acb_jan = ohlcv_store.load(
    symbols=['ACB'],
    start=date(2025, 1, 1),
    end=date(2025, 1, 31)
)
# Returns: 22 rows (ACB daily data for January 2025)
```

### Check Last Date in Store
```python
last_date = ohlcv_store.get_last_date()  # date(2026, 5, 6)
last_vcb_date = ohlcv_store.get_last_date('VCB')  # For specific symbol
```

### Append New Data (Used by Daily Job)
```python
rows_added = ohlcv_store.append(df_new)
print(f"Added {rows_added} rows")
```

---

## Current Data

| Metric | Value |
|---|---|
| **Total Rows** | 32,808 |
| **Symbols** | 100 (VN30 + VN100) |
| **Date Range** | 2025-01-02 → 2026-05-06 |
| **File Size** | 814 KB |
| **Exchange** | HOSE (all) |
| **Industry** | Not yet enriched (fallback: "Unknown") |

---

## Next Steps

### Option A: Extend History to 2020 (Recommended)

```bash
# Clear current store and rebuild from 2020
python scripts/build_ohlcv_history.py --start 2020-01-01 --clear-first --workers 10
```

**Requires**: `vnstock_data` Golden tier in virtual environment
- Estimated time: 15-20 minutes
- Result: ~160k rows (130 symbols × 5 years)

### Option B: Use as-is (2025–Present)

Current data is production-ready for:
- Recent analysis (last 16 months)
- Backtesting (limited historical span)
- Daily monitoring (fully automated)

### Option C: Add to Pipelines

Start querying from analysis code:

```python
# In investment_graph.py, trade_graph.py, or custom analysis
from multiagents_trading_assistant.data import ohlcv_store

# Get last month's data for a stock
from datetime import date, timedelta
yesterday = date.today() - timedelta(days=1)
month_ago = yesterday - timedelta(days=30)

df = ohlcv_store.load(symbols=['VCB'], start=month_ago, end=yesterday)
# Use df for analysis...
```

---

## Troubleshooting

### Issue: Import Error
```
ImportError: cannot import name 'VnstockDataProvider'
```
**Solution**: Already fixed in scripts. If still occurs, use:
```python
from multiagents_trading_assistant.data.providers.vnstock_provider import VnstockDataProvider
```

### Issue: vnstock_data Not Installed
When backfill runs, it falls back to DNSE API automatically. Data quality is identical.

### Issue: Industry Field Shows "Unknown"
Expected behavior when vnstock_data Reference API is unavailable. Can be enriched later with:
```python
# After vnstock_data is installed
industry_map = provider._reference.equity.list_by_industry()
```

---

## Architecture Notes

- **Deduplication**: Rows deduplicated by (date, symbol) — keeps latest version
- **Timezone**: All timestamps normalized to Asia/Ho_Chi_Minh (UTC+7)
- **Compression**: Snappy (fast reads, ~80% size reduction)
- **Incremental**: append() reads + dedupes + writes (safe for concurrent access)
- **Data Sources**: Primary vnstock_data, fallback DNSE LightSpeed API

---

## Files Reference

```
multiagents_trading_assistant/
  data/
    ohlcv_store.py              ← Core storage module
    ohlcv_master.parquet        ← Actual data file (814 KB)
    
scripts/
  build_ohlcv_history.py        ← One-time backfill
  update_ohlcv_daily.py         ← Daily incremental
  
orchestrator/
  pipeline_runner.py            ← Scheduler integration (15:35 job)

docs/
  ohlcv_data_store.md           ← Full documentation
```

---

**Last Updated**: 2026-05-06  
**Status**: Production Ready  
**Next Auto-Update**: 2026-05-07 at 15:35 (or when APScheduler starts)
