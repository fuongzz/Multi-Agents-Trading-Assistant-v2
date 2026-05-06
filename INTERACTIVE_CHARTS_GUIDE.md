# Money Cycle Interactive Charts Guide

## 🎯 Overview

Interactive HTML charts with **Plotly** allow you to zoom, pan, and hover over data for detailed exploration. Perfect for viewing 6+ years of data with granular time selection.

---

## 📊 Available Charts

### 1. Market Dashboard (01_market_dashboard_interactive.html)
**4-panel market-wide analysis:**
- **Panel 1**: CHDM03/10/50/200 — Price position across timeframes (0-100%)
- **Panel 2**: DS03/50/200 — Distribution state across timeframes (0-1)
- **Panel 3**: CHDM20 with zones — Washout (<20%, red) and distribution (>80%, green) shaded areas
- **Panel 4**: DS20 vs DS200 — Short-term vs long-term distribution comparison

### 2. Symbol Dashboard (02_symbol_dashboard_{SYMBOL}_interactive.html)
**3-panel analysis for a specific stock (default: VCB):**
- **Panel 1**: Price + Moving Averages (MA50, MA200)
- **Panel 2**: CHDM03/20/50/200 — Stock price position in cycles
- **Panel 3**: DS03/20/200 — Stock distribution state

---

## 🖱️ How to Use

### Open in Browser
```bash
# Windows Explorer
open data/research/money_cycle/interactive/01_market_dashboard_interactive.html

# Or use file manager to navigate:
# C:\Users\NC\Desktop\AI-Trading-Assistant\data\research\money_cycle\interactive\
```

### Mouse Controls
| Action | Behavior |
|---|---|
| **Scroll wheel** | Zoom in/out on time axis |
| **Click + drag** | Pan (move left/right) to view different time ranges |
| **Hover** | See exact values at that date with unified tooltip |
| **Double-click** | Reset zoom to full range |
| **Legend click** | Toggle line visibility (e.g., hide MA50 to focus on price) |
| **Camera icon** | Download chart as PNG |

### Navigation Example
1. **View 6-year span**: Open chart (shows full 2020-2026 data)
2. **Zoom to 2023**: Scroll wheel to zoom into 2023 region
3. **Pan to March 2023**: Click and drag to center on March
4. **See exact values**: Hover over any point to see date + CHDM/DS values
5. **Hide MA200**: Click "MA200" in legend to toggle off
6. **Download**: Click camera icon to save as PNG

---

## 🔧 Generate Charts for Different Symbols

### Default (VCB)
```bash
python -m multiagents_trading_assistant.research.money_cycle.viz_interactive_cli
```
Output: `02_symbol_dashboard_VCB_interactive.html`

### Any Symbol (e.g., FPT, VNM, VHM)
```bash
python -m multiagents_trading_assistant.research.money_cycle.viz_interactive_cli --symbol FPT
python -m multiagents_trading_assistant.research.money_cycle.viz_interactive_cli --symbol VNM
python -m multiagents_trading_assistant.research.money_cycle.viz_interactive_cli --symbol VHM
```
Output: `02_symbol_dashboard_{SYMBOL}_interactive.html`

### Custom Paths
```bash
python -m multiagents_trading_assistant.research.money_cycle.viz_interactive_cli \
  --data-dir data/research/money_cycle \
  --output-dir data/research/money_cycle/interactive \
  --symbol VHM
```

---

## 📈 Interpreting the Indicators

### CHDM (Price Position %)
- **0-20%** = Washout zone (oversold, potential reversal)
- **20-80%** = Normal trading range
- **80-100%** = Distribution zone (overbought, potential pullback)

### DS (Distribution Share 0-1)
- **0.0-0.3** = Few stocks in downtrend (bullish)
- **0.3-0.7** = Mixed market (neutral)
- **0.7-1.0** = Many stocks in downtrend (bearish)

### Moving Averages (MA)
- **MA50** = 50-day moving average (short-term trend)
- **MA200** = 200-day moving average (long-term trend)
- **Golden Cross** = MA50 > MA200 (bullish signal)
- **Death Cross** = MA50 < MA200 (bearish signal)

---

## 📁 File Locations

```
data/research/money_cycle/interactive/
  ├── 01_market_dashboard_interactive.html  (5.2 MB)
  └── 02_symbol_dashboard_VCB_interactive.html (5.2 MB)
```

---

## 🚀 Quick Start

1. **Run CLI**: `python -m multiagents_trading_assistant.research.money_cycle.viz_interactive_cli`
2. **Open file**: `data/research/money_cycle/interactive/01_market_dashboard_interactive.html`
3. **Zoom** with mouse wheel
4. **Pan** with click + drag
5. **Hover** to see values

---

## 💡 Tips

- **Charts are self-contained HTML** — Works offline, no internet needed
- **Responsive design** — Zoom level is remembered while panning
- **Unified hover** — See all panel values at same timestamp
- **High resolution** — HTML is 5.2MB (embedded Plotly.js + data)
- **Export as PNG** — Use camera icon to save static image

---

**Generated**: 2026-05-07  
**Interactive Library**: Plotly 6.7.0  
**Data Source**: OHLCV store (2020-2026, 100 symbols)
