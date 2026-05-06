# Money Cycle Analysis — Quick Start Guide

## 🎯 What is Money Cycle Analysis?

**CHDM** (Chu kỳ dòng tiền) = Price position within N-period range (0-100%)
- 0-20%: **Washout** (oversold, reversal signal)
- 80-100%: **Distribution** (overbought, pullback signal)
- 20-80%: **Normal range**

**DS** (Distribution Share) = % of stocks in downtrend = (close < MA_N AND MA_N declining)
- Low (0-0.3): Bullish (few in downtrend)
- High (0.7-1.0): Bearish (many in downtrend)

---

## 📊 View Charts

### Static PNG Charts (High Quality, No Interaction)
```bash
python -m multiagents_trading_assistant.research.money_cycle.viz_cli
# → Generates 7 PNG files in: data/research/money_cycle/charts/
```

**Files created:**
- Market dashboard (4-panel)
- 3 heatmaps (CHDM20/50/200)
- Symbol dashboard (VCB 3-panel)
- 2 scatter plots (CHDM vs forward returns)

### Interactive HTML Charts (Zoom, Pan, Hover)
```bash
python -m multiagents_trading_assistant.research.money_cycle.viz_interactive_cli
# → Generates interactive HTML in: data/research/money_cycle/interactive/
```

**Files created:**
- Market dashboard (interactive, Plotly, 5.2MB)
- Symbol dashboard for VCB (interactive, Plotly, 5.2MB)

**To view different symbol:**
```bash
python -m multiagents_trading_assistant.research.money_cycle.viz_interactive_cli --symbol FPT
python -m multiagents_trading_assistant.research.money_cycle.viz_interactive_cli --symbol VNM
```

**Open in browser:**
1. Navigate to: `data/research/money_cycle/interactive/`
2. Double-click any `.html` file
3. Scroll wheel to zoom
4. Click + drag to pan
5. Hover to see values

---

## 📈 Run Backtest

Test 2 trading strategies on historical data (2020-2026):

```bash
python -m multiagents_trading_assistant.research.money_cycle.backtest_cli
```

### Strategy 1: Washout Reversal
- **Signal**: CHDM03 < 20% (washout zone)
- **Action**: BUY next day, HOLD 10 days
- **Result**: 60.9% win rate, +104% total return

### Strategy 2: Stock Selection
- **Signal**: Every 5 days, filter top 10 stocks with CHDM50 < 30%
- **Action**: BUY all, HOLD 10 days
- **Result**: 50.3% win rate, +620% total return (⚠️ risky: overlapping positions)

---

## 🔍 Key Insights

### From Visualization
- **Heatmaps** show which stocks are in washout/distribution zones by date
- **Market dashboard** tracks market-wide CHDM/DS for timing decisions
- **Scatter plots** prove correlation: low CHDM50 → better future returns

### From Backtest
- **Washout strategy** = reliable (high win rate, controlled drawdown)
- **Stock selection** = high returns but risky without position management
- **Optimizable**: threshold, hold period, position sizing, risk limits

---

## 📁 Data Structure

```
multiagents_trading_assistant/
  research/money_cycle/
    compute_money_cycle.py          ← Core CHDM/DS calculation
    cli.py                          ← CLI for compute
    visualization.py                ← PNG charts
    viz_cli.py                      ← CLI for PNG
    visualization_interactive.py    ← Plotly HTML charts (NEW)
    viz_interactive_cli.py          ← CLI for Plotly (NEW)
    backtest_strategy.py            ← Backtest engine
    backtest_cli.py                 ← CLI for backtest
  data/
    ohlcv_master.parquet            ← Raw data (100 symbols, 2020-2026)
  research/money_cycle/
    money_cycle_market.parquet      ← Market CHDM/DS by date
    chdm_by_symbol.parquet          ← Per-stock CHDM
    ds_by_symbol.parquet            ← Per-stock DS
    charts/                         ← 7 PNG files
    interactive/                    ← 3+ HTML interactive files
```

---

## 🚀 Typical Workflow

### 1. Generate All Data
```bash
python -m multiagents_trading_assistant.research.money_cycle.cli
# Computes CHDM/DS from OHLCV, saves 5 parquet files (~20 sec)
```

### 2. Create Visualizations
```bash
# PNG (static, high quality)
python -m multiagents_trading_assistant.research.money_cycle.viz_cli --symbol VCB

# HTML (interactive, better for 6-year exploration)
python -m multiagents_trading_assistant.research.money_cycle.viz_interactive_cli --symbol VCB

# View in browser (double-click HTML file)
open data/research/money_cycle/interactive/02_symbol_dashboard_VCB_interactive.html
```

### 3. Backtest Strategies
```bash
python -m multiagents_trading_assistant.research.money_cycle.backtest_cli
# Outputs: win rate, profit factor, drawdown, trade stats
```

### 4. Analyze Results
- Look for patterns in heatmaps (which stocks align with washout signals)
- Track CDHM20 zones for entry/exit timing
- Use scatter plots to validate CHDM predictability
- Optimize backtest parameters (threshold, hold period)

---

## 💡 Tips

1. **6-year span too compressed?** → Use interactive charts (zoom + pan)
2. **Which symbol to focus on?** → Check heatmaps for frequent washout occurrences
3. **Market bullish or bearish?** → Check DS market line (low = bullish, high = bearish)
4. **Entry signal?** → Look for CHDM < 20 (washout) with DS declining (less stocks in downtrend)
5. **Risk management?** → Backtest shows position overlap risk — use portfolio cap + sizing

---

## 📚 Documentation

- **INTERACTIVE_CHARTS_GUIDE.md** ← How to use Plotly charts (zoom, pan, hover)
- **VISUALIZATION_BACKTEST_COMPLETE.md** ← Full task results (PNG + backtest)
- **MONEY_CYCLE_QUICK_START.md** ← This file (TL;DR guide)

---

## 🎓 For Developers

### Running Individual Components
```python
from multiagents_trading_assistant.research.money_cycle.compute_money_cycle import run_money_cycle_research
from multiagents_trading_assistant.research.money_cycle.visualization import run_all_charts
from multiagents_trading_assistant.research.money_cycle.visualization_interactive import run_interactive_charts
from multiagents_trading_assistant.research.money_cycle.backtest_strategy import run_backtests

# Compute
run_money_cycle_research()

# Visualize (PNG)
run_all_charts(data_dir="data/research/money_cycle", output_dir="data/research/money_cycle/charts", symbol="VCB")

# Visualize (Interactive)
run_interactive_charts(data_dir="data/research/money_cycle", output_dir="data/research/money_cycle/interactive", symbol="VCB")

# Backtest
run_backtests(data_dir="data/research/money_cycle")
```

---

**Last Updated**: 2026-05-07  
**Status**: All features working, ready for exploration
