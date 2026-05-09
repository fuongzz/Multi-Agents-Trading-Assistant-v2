# Price-Volume Intelligence Layer

**Module**: `multiagents_trading_assistant/price_volume.py`  
**Type**: Pure rule-based, deterministic, no LLM  
**Score range**: -100 … +100  

---

## Purpose

Price and volume are the two most fundamental signals in technical analysis. This layer extracts structured, actionable intelligence from raw OHLCV data:

- Detect **real accumulation** vs **distribution** behind price moves
- Evaluate **breakout quality** (confirmed, weak, or fake)
- Identify **volume dry-up** bases (coiled spring setups)
- Catch **washout** reversals (shakeout + recovery)
- Flag **buying climax** exhaustion
- Measure **effort vs result** (absorption vs distribution)

The layer is fully deterministic: given the same OHLCV data and `as_of_date`, it always produces the same output. This makes it safe for both live trading and bar-by-bar backtesting.

---

## Entry Point

```python
from multiagents_trading_assistant.price_volume import analyze_price_volume

result = analyze_price_volume(df, as_of_date="2024-06-15")
```

**Input**:
- `df`: pandas DataFrame with columns `open`, `high`, `low`, `close`, `volume` (case-insensitive). An optional `date` or `time` column is used for filtering.
- `as_of_date`: ISO date string. Only bars on or before this date are used. **Required for correct backtesting** — prevents future-data leakage.

**Output**:
```python
{
    "price_volume_score": int,        # -100 … +100
    "entry_bias": str,                # "bullish" | "neutral" | "bearish" | "avoid"
    "signals": dict[str, bool],       # 14 named signals
    "risk_flags": list[str],          # active risk flags
    "setup_tags": list[str],          # active setup quality tags
    "interpretation": str,            # short deterministic English explanation
    "metrics": dict,                  # raw computed metrics
}
```

---

## Signal Definitions

All signals are evaluated on the **latest bar only**, using only data available up to that bar.

| Signal | Condition |
|---|---|
| `breakout_confirmed` | close > resistance_20 AND vol > 1.5×MA20 AND close_pos > 0.70 |
| `weak_breakout` | close > resistance_20 AND vol < MA20 |
| `fake_breakout_risk` | high > resistance_20 AND close < resistance_20 AND vol > 1.3×MA20 |
| `volume_dry_up` | vol_ma5 < 0.7×MA20 AND price_range_5d < 0.5×price_range_20d |
| `accumulation_day` | close > open AND close > prev_close AND vol > 1.2×MA20 AND close_pos > 0.65 |
| `distribution_day` | close < prev_close AND vol > 1.2×MA20 AND close_pos < 0.40 |
| `high_effort_low_result` | effort > 1.8 AND result < 0.5 |
| `absorption_signal` | high_effort_low_result AND close ≥ open AND close_pos > 0.50 |
| `distribution_signal` | high_effort_low_result AND close < open AND close_pos < 0.50 |
| `washout` | low < support_20 AND close > open AND close_pos > 0.60 AND vol > 1.5×MA20 |
| `buying_climax` | close > MA20×1.15 AND vol > 2.5×MA20 AND close_pos < 0.60 |
| `constructive_pullback` | MA20×0.97 ≤ close ≤ MA20×1.05 AND vol < MA20 AND close ≥ prev_close×0.97 |
| `bearish_volume_expansion` | close < prev_close AND vol > 1.5×MA20 AND close_pos < 0.35 |
| `bullish_volume_expansion` | close > prev_close AND vol > 1.5×MA20 AND close_pos > 0.65 |

**Key definitions**:

- `close_position_in_range`: `(close - low) / (high - low)`. Returns 0.5 when high == low.
- `resistance_20`: highest high of the **previous 20 bars**, excluding the current bar.
- `support_20`: lowest low of the **previous 20 bars**, excluding the current bar.
- `ATR14`: rolling 14-bar mean of True Range (`max(H-L, |H-prev_C|, |L-prev_C|)`).
- `effort`: `volume / vol_ma20` (current bar relative to 20-bar average).
- `result`: `|close - prev_close| / ATR14` (price move relative to volatility).

---

## Scoring Logic

Start at 0. Apply bonuses and penalties:

**Positive contributors**:

| Condition | Points |
|---|---|
| `breakout_confirmed` | +25 |
| `accumulation_count_20 ≥ 5` | +25 |
| `accumulation_count_20 ≥ 3` | +15 |
| `volume_dry_up` | +10 |
| `washout` | +15 |
| `absorption_signal` | +10 |
| `constructive_pullback` | +10 |
| `bullish_volume_expansion` | +15 |

**Negative contributors**:

| Condition | Points |
|---|---|
| `weak_breakout` | -15 |
| `fake_breakout_risk` | -25 |
| `distribution_count_20 ≥ 5` | -35 |
| `distribution_count_20 ≥ 3` | -20 |
| `buying_climax` | -25 |
| `distribution_signal` | -15 |
| `bearish_volume_expansion` | -20 |

**Final score is clamped to [-100, +100].**

---

## Entry Bias

| Condition (checked in order) | Bias |
|---|---|
| `distribution_count_20 ≥ 5` OR `fake_breakout_risk` | `"avoid"` |
| `price_volume_score ≥ 35` | `"bullish"` |
| `price_volume_score ≤ -25` | `"bearish"` |
| otherwise | `"neutral"` |

---

## Risk Flags

Active when the corresponding signal is true:

| Flag | Trigger |
|---|---|
| `FAKE_BREAKOUT_RISK` | `fake_breakout_risk` signal |
| `HEAVY_DISTRIBUTION` | `distribution_count_20 ≥ 5` |
| `BUYING_CLIMAX` | `buying_climax` signal |
| `BEARISH_VOLUME_EXPANSION` | `bearish_volume_expansion` signal |
| `WEAK_BREAKOUT` | `weak_breakout` signal |
| `DISTRIBUTION_SIGNAL` | `distribution_signal` signal |

---

## Setup Tags

Active when the corresponding condition is true:

| Tag | Condition |
|---|---|
| `HIGH_CONVICTION_BREAKOUT` | `breakout_confirmed` AND `accumulation_count_20 ≥ 3` |
| `VOLUME_DRY_UP_BASE` | `volume_dry_up` |
| `WASHOUT_REVERSAL` | `washout` |
| `ABSORPTION` | `absorption_signal` |
| `CONSTRUCTIVE_PULLBACK` | `constructive_pullback` |
| `BULLISH_VOLUME_EXPANSION` | `bullish_volume_expansion` |

---

## Integration Map

### 1. `indicators.py` → `compute_indicators()`

`analyze_price_volume(df)` is called at the end of `compute_indicators()`. The result is stored under `indicators["price_volume"]`. Key scalars (`price_volume_score`, `pv_entry_bias`, `pv_risk_flags`, `pv_setup_tags`) are also flattened to the top level for quick access.

### 2. `technical_agent.py`

`_attach_price_volume()` copies the PV result from the indicator dict into `technical_analysis["price_volume"]` and adds scalar keys to `indicator_snapshot`. PV data flows downstream in the TradeState as part of `technical_analysis`.

### 3. `synthesis_agent.py` — Confluence weighting

PV contributes **25%** to the master `confluence_score`:

| Component | Weight |
|---|---|
| Technical setup score | 30% |
| Foreign flow | 15% |
| Money flow (Blackbox) | 19% |
| Sentiment | 11% |
| **Price-Volume score** | **25%** |

The PV score (-100…+100) is mapped to 0…100 before weighting. `entry_bias = "avoid"` caps the mapped score at 20/100 regardless of the raw number.

### 4. `trade_screener.py` — Candidate ranking

`priority_score` now includes a signed PV contribution of ±20 points:

```
priority_score = vol_ratio × 32 + avg_vol_20M × 5 + pv_score × 0.20
```

Stocks with `entry_bias = "avoid"` are hard-capped at `priority_score ≤ 30` and cannot appear in the top of the candidate list.

### 5. `risk_trade.py` — Hard gates

Applied before max-loss sizing, for `BUY` actions only:

| Flag | Action |
|---|---|
| `FAKE_BREAKOUT_RISK` | Override → CHỜ |
| `HEAVY_DISTRIBUTION` | Override → CHỜ |
| `BUYING_CLIMAX` | Override → CHỜ (unless position held + confluence ≥ 80) |
| `BEARISH_VOLUME_EXPANSION` | Sizing × 0.7 + warning |

### 6. `trader_trade.py` — LLM prompt

A `=== Price-Volume Intelligence ===` section is injected into the Sonnet prompt with score, bias, interpretation, risk flags, and setup tags. The agent is informed that FAKE_BREAKOUT_RISK and HEAVY_DISTRIBUTION have already been blocked upstream.

---

## Metrics Reference

| Key | Description |
|---|---|
| `volume_ma5` | 5-bar volume moving average |
| `volume_ma20` | 20-bar volume moving average |
| `volume_ratio_20` | Current volume / vol_ma20 |
| `ma20` | 20-bar price moving average |
| `atr14` | 14-bar Average True Range |
| `resistance_20` | Highest high of previous 20 bars (current excluded) |
| `support_20` | Lowest low of previous 20 bars (current excluded) |
| `close_position_in_range` | (close − low) / (high − low); 0.5 if H==L |
| `price_range_5d` | High − Low over last 5 bars |
| `price_range_20d` | High − Low over last 20 bars |
| `accumulation_count_20` | Number of accumulation days in last 20 bars |
| `distribution_count_20` | Number of distribution days in last 20 bars |
| `effort` | volume / vol_ma20 for current bar |
| `result` | \|close − prev_close\| / ATR14 for current bar |

---

## Interaction with Other Layers

| Layer | Interaction |
|---|---|
| **Regime (Market Context)** | Regime determines which setups are valid. PV score is independent of regime — it can confirm or contradict regime bias. |
| **Money Flow (Blackbox)** | Money Flow detects institutional order flow; PV measures raw supply/demand footprint from candle structure. They are complementary. |
| **Technical Score** | Technical score is pattern-based (MA, RSI, MACD). PV catches what technical patterns miss: volume conviction quality. |
| **Sentiment** | News sentiment is external; PV is purely price-internal. Low correlation — both should be checked. |

---

## Known Limitations

1. **Single-bar latency**: All signals evaluate only the last bar. Multi-bar patterns (e.g., prolonged accumulation) are captured via the 20-bar accumulation/distribution count but the setup_tag fires only once the count threshold is met.

2. **Volume normalization**: Uses a 20-bar rolling mean as the baseline. In low-liquidity stocks, one block trade can distort all signals for 20 bars.

3. **No price context above 20 bars**: Resistance/support are computed over only the last 20 bars. Major historical levels are captured separately by `compute_support_resistance()`.

4. **Effort/result interpretation**: High effort + low result is bullish (absorption) when the bar closes green, bearish (distribution) when red. But in thin VN markets, a single block trade can create a misleading signal on low-volume stocks.

5. **No regime filter inside PV**: The module does not adjust thresholds based on UPTREND/SIDEWAY/DOWNTREND. The caller (synthesis, risk, screener) applies regime-specific interpretation.

---

## Testing

```bash
pytest tests/test_price_volume.py -v
```

Tests cover: empty/short data, NaN handling, zero-volume, `as_of_date` isolation, all 14 signals, all risk flags, all setup tags, score bounds, schema completeness, and no-look-ahead for resistance/support.
