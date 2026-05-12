# Adaptive Kalman MVP Experiment - 2026-05-12

## Change Tested

Added adaptive Kalman gate features alongside the earlier static Kalman columns:

- `kalman_adaptive_close`
- `kalman_adaptive_trend_5d`
- `kalman_adaptive_residual_pct`
- `kalman_confidence`
- `kalman_shock_score`

The adaptive filter is designed as a support gate, not a signal:

- trusts price less on wide-range, gap, low-volume, or low-value bars,
- clips extreme innovation so one shock day cannot fully pull the estimated trend,
- lets trend adapt faster when regime, CHDM, DS, and smart-money context are healthier.

## Output Files

Directory:

`backtest_results/adaptive_kalman_mvp_experiment_2026-05-12/`

Key files:

- `combined_summary.csv`
- `combined_summary_readable.csv`
- `combined_deltas.csv`
- `combined_deltas_readable.csv`
- `vn30_2022_summary.csv`
- `vn30_2022_strategy_breakdown.csv`
- `vn100_2025_summary.csv`
- `vn100_2025_strategy_breakdown.csv`

## Readable Run Names

The readable summary files use these labels:

| Label | Meaning |
|---|---|
| `No Kalman - Baseline` | Original MVP, no Kalman gate |
| `Static Kalman Gate - Setup Aware` | Earlier fixed-threshold Kalman gate |
| `Adaptive Kalman Gate - Loose` | Light confidence/shock/residual gate |
| `Adaptive Kalman Gate - Quality/Shock` | Stricter confidence and shock filter |
| `Adaptive Kalman Gate - All MVP Setups` | Adaptive gate applied to breakout, compression, and pullback |
| `Adaptive Kalman Gate - Pullback Only` | Breakout unchanged; adaptive Kalman gate only supports pullback/compression style setups |

Each run is also prefixed with:

- `VN30 2022` or `VN100 2025`: tested universe and period.
- `Family`: always evaluates the MVP strategy family.
- `Router`: applies the MVP regime router before evaluating strategies.

## VN30, 2022-05-08 to 2026-05-08

| Run | Return | Sharpe | Max DD | Profit Factor | Read |
|---|---:|---:|---:|---:|---|
| VN30 2022 - Family - No Kalman - Baseline | 65.34% | 0.880 | -14.73% | 1.78 | Reference |
| VN30 2022 - Family - Adaptive Kalman Gate - Loose | 76.51% | 0.964 | -14.98% | 1.79 | Best adaptive family |
| VN30 2022 - Router - No Kalman - Baseline | 50.91% | 0.738 | -16.91% | 1.59 | Reference |
| VN30 2022 - Router - Adaptive Kalman Gate - Loose | 60.20% | 0.824 | -17.74% | 1.68 | Best adaptive router |
| VN30 2022 - Router - Static Kalman Gate - Setup Aware | 69.18% | 0.884 | -18.55% | 1.67 | Higher return, worse DD |

Interpretation:

- Adaptive loose is the best VN30 adaptive configuration.
- It improves return and Sharpe for both family and router.
- Drawdown is slightly worse, so it is an alpha/timing improvement rather than a pure risk-control improvement.
- Quality/pullback variants were too restrictive and hurt performance.

## VN100, 2025-01-01 to 2026-05-08

| Run | Return | Sharpe | Max DD | Profit Factor | Read |
|---|---:|---:|---:|---:|---|
| VN100 2025 - Family - No Kalman - Baseline | 52.92% | 2.099 | -10.78% | 2.32 | Reference |
| VN100 2025 - Family - Adaptive Kalman Gate - Pullback Only | 54.41% | 2.210 | -9.43% | 2.31 | Best adaptive family |
| VN100 2025 - Router - No Kalman - Baseline | 43.49% | 1.833 | -11.60% | 2.21 | Reference |
| VN100 2025 - Router - Adaptive Kalman Gate - Pullback Only | 67.98% | 2.599 | -9.43% | 2.49 | Best overall |

Interpretation:

- The strongest result is **adaptive no-breakout router**.
- This means Kalman should not gate `breakout_after_accumulation_v3`.
- It should support `mean_reversion_uptrend_ma50_v1` and, with caution, `compression_breakout_smt_v1`.
- On VN100 recent, adaptive no-breakout improves return, Sharpe, drawdown, win rate, and profit factor.

## Strategy-Level Read

VN30:

- `adaptive_loose` helps the family mostly by improving compression and maintaining enough breakout/pullback participation.
- Tight confidence/shock gates remove too many useful trades.

VN100 recent:

- Leaving breakout untouched is critical.
- Adaptive pullback gate improves `mean_reversion_uptrend_ma50_v1` strongly:
  - router avg trade: 2.49% baseline to 4.17%
  - router profit factor: 2.19 baseline to 3.91
- Compression remains unstable; use Kalman there only as a soft rank/gate, not a strict filter.

## Recommendation

Do not use one global Kalman gate.

Recommended production/shadow plan:

1. Keep `breakout_after_accumulation_v3` unchanged.
2. Add adaptive Kalman gate only to pullback/mean-reversion style MVP entries.
3. For VN30 router, test `adaptive_loose` as a shadow variant.
4. For VN100 router, test `adaptive_no_breakout` as the preferred shadow variant.
5. Promote only after a walk-forward check confirms the VN100 recent result is not a single-period artifact.

Suggested gate direction:

```json
[
  {"column": "kalman_confidence", "op": ">=", "value": 0.30},
  {"column": "kalman_shock_score", "op": "<=", "value": 3.5},
  {"column": "kalman_adaptive_trend_5d", "op": ">=", "value": -0.002},
  {"column": "kalman_adaptive_residual_pct", "op": "between", "value": [-0.08, 0.05]}
]
```

Use this first on `mean_reversion_uptrend_ma50_v1`, not on breakout.
