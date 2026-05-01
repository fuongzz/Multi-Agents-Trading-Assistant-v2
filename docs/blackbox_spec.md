# Blackbox Money Flow Spec

## Status

Draft implementation spec.

This document defines a QMV-inspired money-flow engine for HOSE/HNX/UPCOM stocks. It is not a clone of the proprietary QMV Blackbox. Public QMV material describes Blackbox as a tool to identify money inflow/outflow, but does not publish the full algorithm. The implementation below converts the public concepts into transparent, backtestable indicators.

## References

- QMV Group overview: top-down approach from macro policy to sectors and stocks, with emphasis on money rotation.
  - https://qmvgroup.com/gioithieu.html
- QMV technical analysis course: supply/demand, game theory, buy/sell zones, timing, QMV Blackbox, and QMV Screener.
  - https://qmvgroup.com/KT01.html
- QMV trading signal guide: Screener, Blackbox money inflow/outflow, and emotional valuation.
  - https://www.studocu.vn/vn/document/truong-cao-dang-hang-hai-ii/kinh-te/huong-dan-su-dung-trading-signal-system/122207103
- Quach Manh Hao market view: "dung nganh, dung dong tien".
  - https://24hmoney.vn/news/tro-choi-dung-cua-qmv-la-dung-nganh-dung-dong-tien-c30a1782117.html

## Goal

Build a rule-based money-flow layer that answers four questions:

1. Is money entering or leaving the stock?
2. Is the stock under accumulation, distribution, or neutral behavior?
3. Is the signal supported by market and sector context?
4. Should the trade pipeline treat this symbol as buy candidate, watchlist, avoid, or sell/exit warning?

The engine must be explainable, deterministic, and backtestable. It should run before expensive LLM nodes when used in screening, and as a confirmation layer before Trader/Risk nodes when used in the multi-agent pipeline.

## Non-Goals

- Do not infer or claim to reproduce QMV's proprietary formula.
- Do not use LLMs for the numeric scoring.
- Do not generate buy/sell recommendations alone. It produces evidence and gating signals for the rest of the trading system.
- Do not trust volume spikes blindly. Every money-flow signal must be interpreted with price location, market regime, and liquidity quality.

## Design Principles

1. Top-down first: market regime -> sector rotation -> stock money flow.
2. Price and value beat raw volume: value traded is more comparable across stocks than volume alone.
3. Accumulation is quiet; distribution is noisy.
4. Breakout requires confirmation from money flow, not only price.
5. Foreign flow is a confirmation signal, not the primary signal.
6. Sideway markets punish chasing. In sideway regimes, prefer accumulation near support and confirmed breakouts only after market context improves.

## Inputs

### Required OHLCV

Daily bars, adjusted consistently:

```python
{
    "time": datetime | str,
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": float,
}
```

The engine derives:

```python
value = close * volume
ret_1d = close.pct_change()
```

### Required Context

```python
{
    "symbol": "HPG",
    "exchange": "HOSE",
    "sector": "Steel",
    "market_context": {
        "trend": "UPTREND" | "SIDEWAY" | "DOWNTREND",
        "sideway_zone": "NEAR_SUPPORT" | "MID_RANGE" | "NEAR_RESISTANCE",
        "vni_change_pct": float,
        "current_price": float,
    },
    "sector_context": {
        "relative_strength_20d": float,
        "is_outperforming": bool,
    }
}
```

### Optional Flow Data

```python
{
    "foreign_buy_value": float,
    "foreign_sell_value": float,
    "proprietary_buy_value": float | None,
    "proprietary_sell_value": float | None,
    "active_buy_value": float | None,
    "active_sell_value": float | None,
}
```

If optional data is missing, the engine must degrade gracefully and mark the reason in `data_quality`.

## Feature Set

### Liquidity Quality

Purpose: remove noisy symbols before scoring.

```python
avg_value_20 = value.rolling(20).mean()
liquidity_ok = avg_value_20 >= min_avg_value
```

Default:

```python
min_avg_value = 20_000_000_000  # VND 20B/day
```

For looser universe scanning, allow VND 5B/day, but lower confidence.

### Abnormal Volume and Value

Purpose: detect unusual participation.

```python
volume_ratio_20 = volume / volume.rolling(20).mean()
value_ratio_20 = value / value.rolling(20).mean()
volume_z_60 = (volume - volume.rolling(60).mean()) / volume.rolling(60).std()
value_z_60 = (value - value.rolling(60).mean()) / value.rolling(60).std()
```

Interpretation:

| Signal | Condition | Meaning |
|---|---:|---|
| Normal | `value_ratio_20 < 1.3` | No unusual money flow |
| Active | `1.3 <= value_ratio_20 < 1.8` | More participation than usual |
| Surge | `value_ratio_20 >= 1.8` or `value_z_60 >= 2` | Abnormal money flow |

### Close Location

Purpose: distinguish buying pressure from selling pressure.

```python
close_position = (close - low) / (high - low)
```

Interpretation:

| Zone | Condition | Meaning |
|---|---:|---|
| Strong close | `close_position >= 0.70` | Buyers controlled the session |
| Neutral close | `0.40 <= close_position < 0.70` | Mixed control |
| Weak close | `close_position < 0.40` | Sellers controlled the session |

### Money In

Purpose: identify clear money entering.

```python
money_in = (
    ret_1d > 0
    and value_ratio_20 >= 1.5
    and close_position >= 0.60
)
```

Stronger condition:

```python
strong_money_in = (
    ret_1d >= 0.02
    and value_ratio_20 >= 1.8
    and close_position >= 0.70
)
```

### Money Out

Purpose: identify clear money leaving.

```python
money_out = (
    ret_1d < 0
    and value_ratio_20 >= 1.5
    and close_position <= 0.40
)
```

Stronger condition:

```python
strong_money_out = (
    ret_1d <= -0.02
    and value_ratio_20 >= 1.8
    and close_position <= 0.30
)
```

### Accumulation

Purpose: identify quiet bases where supply is drying up.

Core idea: price compresses, volume cools down, and relative strength stops falling.

```python
range_20 = (high.rolling(20).max() - low.rolling(20).min()) / close
range_120_q35 = range_20.rolling(120).quantile(0.35)
ret_vol_20 = ret_1d.rolling(20).std()
ret_vol_120 = ret_1d.rolling(120).std()

accumulation = (
    range_20 < range_120_q35
    and volume_ratio_20 < 0.9
    and ret_vol_20 < ret_vol_120
    and relative_strength_20d >= -0.03
)
```

Preferred location:

```python
near_support = close <= low.rolling(60).min() * 1.10
```

An accumulation signal becomes high quality only when the stock later produces `breakout_flow`.

### Distribution

Purpose: identify selling by stronger hands into liquidity.

```python
failed_breakout = (
    high > high.rolling(20).max().shift(1)
    and close < high.rolling(20).max().shift(1)
    and value_ratio_20 >= 1.5
)

distribution = (
    money_out
    or failed_breakout
    or (
        value_ratio_20 >= 1.8
        and close_position <= 0.35
        and ret_1d <= 0
    )
)
```

Distribution is a hard negative. It should block new buy signals unless explicitly overridden by a later bullish reversal.

### Breakout Flow

Purpose: identify price leaving a base with real participation.

```python
resistance_20 = high.rolling(20).max().shift(1)

breakout_flow = (
    close > resistance_20
    and value_ratio_20 >= 1.8
    and close_position >= 0.70
    and ret_1d >= 0.015
)
```

High-quality breakout:

```python
high_quality_breakout = (
    breakout_flow
    and sector_is_outperforming
    and market_trend != "DOWNTREND"
    and not distribution
)
```

### Foreign Flow

Purpose: capture foreign participation, but avoid overfitting.

```python
foreign_net_value = foreign_buy_value - foreign_sell_value
foreign_net_ratio = foreign_net_value / value
foreign_net_5d = foreign_net_value.rolling(5).sum()
foreign_net_20d = foreign_net_value.rolling(20).sum()
```

Scoring:

| Condition | Score |
|---|---:|
| `foreign_net_ratio >= 0.05` | +1 |
| `foreign_net_5d > 0 and foreign_net_20d > 0` | +1 |
| `foreign_net_ratio <= -0.05` | -1 |
| `foreign_net_5d < 0 and foreign_net_20d < 0` | -1 |

Foreign selling is not a hard blocker by itself. In Vietnam, strong domestic flow can overpower foreign selling.

### Relative Strength

Purpose: enforce "right sector, right money".

```python
stock_ret_20 = close / close.shift(20) - 1
benchmark_ret_20 = vnindex_close / vnindex_close.shift(20) - 1
relative_strength_20d = stock_ret_20 - benchmark_ret_20
```

Sector relative strength should be calculated similarly using sector basket returns.

## Scoring Model

### Daily Score

```python
daily_score = 0

if money_in:
    daily_score += 2
if strong_money_in:
    daily_score += 1
if breakout_flow:
    daily_score += 3
if accumulation:
    daily_score += 1
if sector_is_outperforming:
    daily_score += 1
if relative_strength_20d > 0:
    daily_score += 1
if foreign_score > 0:
    daily_score += foreign_score

if money_out:
    daily_score -= 2
if strong_money_out:
    daily_score -= 1
if distribution:
    daily_score -= 4
if market_trend == "DOWNTREND":
    daily_score -= 2
if foreign_score < 0:
    daily_score += foreign_score
```

Clamp:

```python
daily_score = max(-10, min(10, daily_score))
```

### Regime Label

```python
if distribution or daily_score <= -4:
    regime = "DISTRIBUTION"
elif breakout_flow and daily_score >= 4:
    regime = "BREAKOUT_FLOW"
elif accumulation and daily_score >= 1:
    regime = "ACCUMULATION"
elif daily_score >= 3:
    regime = "MONEY_IN"
elif daily_score <= -2:
    regime = "MONEY_OUT"
else:
    regime = "NEUTRAL"
```

### Action Bias

```python
if regime == "BREAKOUT_FLOW" and market_trend != "DOWNTREND":
    action_bias = "BUY_CANDIDATE"
elif regime == "ACCUMULATION":
    action_bias = "WATCHLIST"
elif regime in {"DISTRIBUTION", "MONEY_OUT"}:
    action_bias = "AVOID_OR_EXIT"
else:
    action_bias = "NEUTRAL"
```

## Output Schema

```python
{
    "symbol": "HPG",
    "as_of": "2026-05-01",
    "engine": "blackbox_money_flow_v1",
    "regime": "BREAKOUT_FLOW",
    "action_bias": "BUY_CANDIDATE",
    "score": 6,
    "confidence": "HIGH",
    "signals": {
        "liquidity_ok": True,
        "money_in": True,
        "strong_money_in": False,
        "money_out": False,
        "accumulation": False,
        "distribution": False,
        "breakout_flow": True,
        "foreign_score": 1,
        "sector_is_outperforming": True,
        "relative_strength_20d": 0.052
    },
    "metrics": {
        "value_ratio_20": 2.15,
        "volume_ratio_20": 1.88,
        "value_z_60": 2.35,
        "close_position": 0.82,
        "ret_1d": 0.034,
        "avg_value_20": 125000000000
    },
    "reasons": [
        "Breakout above 20-day resistance with value spike",
        "Strong close near session high",
        "Sector outperforming benchmark"
    ],
    "blockers": [],
    "data_quality": {
        "ohlcv_days": 250,
        "has_foreign_flow": True,
        "has_intraday_flow": False,
        "warnings": []
    }
}
```

## Confidence Rules

```python
if not liquidity_ok:
    confidence = "LOW"
elif data_quality["ohlcv_days"] < 120:
    confidence = "LOW"
elif abs(score) >= 5 and market_trend != "DOWNTREND":
    confidence = "HIGH"
elif abs(score) >= 3:
    confidence = "MEDIUM"
else:
    confidence = "LOW"
```

Downgrade confidence when:

- Foreign flow is missing and the stock is highly foreign-sensitive.
- Latest bar is not a completed trading day.
- Average value is below threshold.
- There is a corporate action or abnormal price adjustment in the last 20 sessions.

## Pipeline Integration

### Screener Stage

Use this engine after the existing market and sector gates:

```text
VN100/VNAllShare
  -> Market Gate
  -> Sector Filter
  -> Liquidity Filter
  -> Blackbox Money Flow
  -> Candidate Ranking
  -> Multi-agent pipeline
```

Candidate rank:

```python
priority_score = (
    blackbox_score * 10
    + max(relative_strength_20d, 0) * 100
    + min(value_ratio_20, 3) * 5
)
```

### Trade Pipeline Stage

Add to `state` as:

```python
state["money_flow_analysis"] = blackbox_output
```

Synthesis should treat:

| Action bias | Pipeline behavior |
|---|---|
| `BUY_CANDIDATE` | Add bullish confluence |
| `WATCHLIST` | Do not buy yet unless other setup confirms |
| `AVOID_OR_EXIT` | Add blocker; risk node may override buy |
| `NEUTRAL` | No strong impact |

### Risk Stage

Risk manager should block new buys when:

```python
blackbox.regime == "DISTRIBUTION"
or blackbox.action_bias == "AVOID_OR_EXIT"
```

Exception: allow sell/exit actions even when distribution is detected.

## Minimal Implementation Plan

1. Create `multiagents_trading_assistant/indicators/money_flow.py` or extend existing `indicators.py`.
2. Implement pure pandas functions:
   - `add_money_flow_features(df)`
   - `classify_money_flow(df, context)`
   - `build_blackbox_output(symbol, df, context, foreign_flow=None)`
3. Add unit tests using synthetic OHLCV:
   - breakout with high value -> `BREAKOUT_FLOW`
   - falling price with high value and weak close -> `DISTRIBUTION`
   - flat range with falling volume -> `ACCUMULATION`
   - low liquidity -> low confidence
4. Wire into screener as a filter/ranking field.
5. Backtest thresholds before enabling hard buy filters.

## Default Parameters

```python
BLACKBOX_DEFAULTS = {
    "min_avg_value": 20_000_000_000,
    "volume_window": 20,
    "value_window": 20,
    "zscore_window": 60,
    "resistance_window": 20,
    "base_window": 20,
    "long_window": 120,
    "money_in_value_ratio": 1.5,
    "breakout_value_ratio": 1.8,
    "distribution_value_ratio": 1.8,
    "strong_close": 0.70,
    "weak_close": 0.40,
    "foreign_ratio_threshold": 0.05,
}
```

## Calibration Notes for HOSE

- Large caps: use value traded, not volume. VND 100B/day is normal for leaders.
- Mid caps: VND 20B/day is a reasonable liquid threshold.
- Small caps: signals are noisier; require stricter close location and repeated confirmation.
- Bank and brokerage stocks often lead sector rotation; their sector context matters more than single-day spikes.
- Real estate and speculative stocks often produce false breakouts; require a second confirmation day or tighter stop.
- ATC-driven spikes can distort close location. If intraday data exists, separate continuous session flow from ATC flow.

## Example Decision Logic

```python
def blackbox_decision(output: dict) -> str:
    if output["regime"] == "DISTRIBUTION":
        return "AVOID_OR_EXIT"
    if output["regime"] == "BREAKOUT_FLOW" and output["score"] >= 4:
        return "BUY_CANDIDATE"
    if output["regime"] == "ACCUMULATION":
        return "WATCHLIST"
    return "NEUTRAL"
```

## Backtest Metrics

Track these metrics by regime:

- 5-day, 10-day, 20-day forward return.
- Hit rate above 3%, 5%, and 8%.
- Max drawdown after signal.
- False breakout rate: close falls back below breakout level within 5 sessions.
- Average liquidity at signal.
- Performance split by market regime and sector.

Useful acceptance targets before production:

```text
BREAKOUT_FLOW:
  10d hit rate > 55%
  avg 10d return > VNINDEX 10d return + 2%
  false breakout rate < 35%

DISTRIBUTION:
  avg 10d return < VNINDEX 10d return
  drawdown warning should trigger before large selloffs often enough to be useful
```

## Open Questions

- Which universe should be default: VN100, VNAllShare, or all HOSE liquid names?
- Should sector baskets come from vnstock listing metadata or a custom mapping?
- Is foreign flow available daily for all symbols in the chosen data source?
- Should intraday active buy/sell be added from DNSE websocket or another provider?
- Should this be used as a hard gate or only as a confluence score in the first release?
