# Kalman Filter Application

## Why It Fits This Project

Kalman filter is useful when the signal we care about is hidden behind noisy
daily price movement. In this project, the practical hidden state is not
"tomorrow's price"; it is the smoother market state behind OHLCV:

- estimated fair trend line,
- short-term trend slope,
- distance between actual close and the estimated trend,
- regime confirmation when combined with Money Cycle and Smart Money Trace.

The best integration point is `multiagents_trading_assistant.edge_lab.features`.
That module already builds the research feature table used by hypotheses,
ranking, and portfolio backtests. Adding Kalman features there keeps the
backtest engine unchanged and avoids look-ahead bias.

## Implemented Feature Columns

`edge_lab.features._add_ta_columns()` now adds three causal Kalman columns per
symbol:

| Column | Meaning | Typical Use |
|---|---|---|
| `kalman_close` | Smoothed estimated price level | Replace/compare with MA trend line |
| `kalman_trend_5d` | Estimated 5-day log-trend, expressed as percent-like return | Filter/rank trend strength |
| `kalman_residual_pct` | Close versus Kalman estimate | Detect pullback/overextension |

It also adds adaptive gate columns for VN daily data:

| Column | Meaning | Typical Use |
|---|---|---|
| `kalman_adaptive_close` | Robust smoothed price level | Diagnostic trend line |
| `kalman_adaptive_trend_5d` | Adaptive 5-day trend estimate | Pullback trend confirmation |
| `kalman_adaptive_residual_pct` | Close versus adaptive estimate | Pullback/overextension gate |
| `kalman_confidence` | 0-1 quality score for the current Kalman update | Avoid low-quality noisy bars |
| `kalman_shock_score` | Normalized innovation shock | Avoid one-bar gap/range shocks |

The implementation uses a local-linear state model on log close:

```text
state = [log_price_level, log_price_slope]
observation = current log(close)
```

It is causal: each row only uses current and past prices.

## Practical Hypothesis Patterns

### 1. Trend Continuation Filter

Use this to reduce false breakouts where price is above MA but the smoother
state is not improving:

```json
{"column": "kalman_trend_5d", "op": ">=", "value": 0.005}
```

Good companions:

- `above_ma50 == true`
- `smart_money_score >= 60`
- `mkt_regime_state not_in ["RISK_OFF"]`

### 2. Pullback Into Healthy Trend

Use residual to find stocks pulling back near the Kalman line without breaking
the underlying trend:

```json
{"column": "kalman_trend_5d", "op": ">=", "value": 0.002}
{"column": "kalman_residual_pct", "op": "between", "value": [-0.04, 0.015]}
```

This fits `mean_reversion_uptrend_ma20_v1` and `leader_pullback_*` strategies.

### 3. Overextension Brake

Avoid buying stretched candles even when other momentum signals are valid:

```json
{"column": "kalman_residual_pct", "op": "<=", "value": 0.08}
```

This is most useful for breakout and ADX-style setups.

## Suggested Next Experiment

Run a grid search on an existing hypothesis rather than enabling Kalman as a
hard global rule immediately:

```powershell
python -m multiagents_trading_assistant.edge_lab.research_cli `
  --hypothesis leader_pullback_market_healthy_v2 `
  --grid "{kalman_trend_5d:[0.0,0.002,0.005],kalman_residual_pct:[-0.04,-0.02,0.02]}"
```

If the grid result improves out-of-sample Sharpe or Calmar without sharply
reducing trade count, then promote the best thresholds into the hypothesis
config.

## Current Experiment Read

The 2026-05-12 MVP experiments show that Kalman works best as a selective gate:

- Do not gate `breakout_after_accumulation_v3` globally.
- Prefer adaptive Kalman on `mean_reversion_uptrend_ma50_v1`.
- VN30 responds best to loose adaptive gates.
- VN100 recent responds best when breakout is left unchanged and Kalman supports pullback entries.

## What Not To Do

- Do not use Kalman output alone as a buy/sell signal.
- Do not tune thresholds on one short period only.
- Do not compare today's signal to future-smoothed values.
- Do not add external Kalman packages unless the current simple model is proven
  insufficient.
