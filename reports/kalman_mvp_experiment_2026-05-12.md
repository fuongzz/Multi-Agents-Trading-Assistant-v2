# Kalman MVP Experiment - 2026-05-12

## Scope

Test whether Kalman-derived features improve the current MVP strategy family:

- `breakout_after_accumulation_v3`
- `compression_breakout_smt_v1`
- `mean_reversion_uptrend_ma50_v1`

Two execution styles were tested:

- **family**: always evaluate the three MVP strategies.
- **router**: mimic `regime_router_mvp_v1`, enabling the MVP family only when market regime allows it.

Kalman variants:

- `loose`: light trend/residual constraints.
- `trend_only`: require non-negative Kalman trend.
- `setup_aware`: stricter trend/residual for breakout, pullback-band for mean reversion.
- `strict`: stronger trend and residual constraints.

Output files:

- `backtest_results/kalman_mvp_experiment_2026-05-12/vn30_summary.csv`
- `backtest_results/kalman_mvp_experiment_2026-05-12/vn30_trades.csv`
- `backtest_results/kalman_mvp_experiment_2026-05-12/vn100_recent_summary.csv`
- `backtest_results/kalman_mvp_experiment_2026-05-12/vn100_recent_trades.csv`

## VN30, 2022-05-08 to 2026-05-08

| Run | Trades | Return | Sharpe | Max DD | Profit Factor |
|---|---:|---:|---:|---:|---:|
| family baseline | 235 | 65.34% | 0.880 | -14.73% | 1.78 |
| family kalman strict | 218 | 53.67% | 0.764 | -15.12% | 1.82 |
| router baseline | 241 | 50.91% | 0.738 | -16.91% | 1.59 |
| router kalman setup-aware | 220 | 69.18% | 0.884 | -18.55% | 1.67 |
| router kalman strict | 215 | 53.39% | 0.767 | -18.90% | 1.72 |

Key read:

- Kalman **does not improve** the fixed MVP family on VN30.
- Kalman **does improve router return and Sharpe** when setup-aware:
  - return: 50.91% to 69.18%
  - Sharpe: 0.738 to 0.884
  - profit factor: 1.59 to 1.67
- But drawdown worsens from -16.91% to -18.55%, so this is not a clean risk improvement.

## VN100, 2025-01-01 to 2026-05-08

| Run | Trades | Return | Sharpe | Max DD | Profit Factor |
|---|---:|---:|---:|---:|---:|
| family baseline | 133 | 52.92% | 2.099 | -10.78% | 2.32 |
| family kalman setup-aware | 128 | 48.88% | 2.054 | -11.18% | 2.01 |
| family kalman strict | 127 | 52.06% | 2.080 | -11.02% | 2.10 |
| router baseline | 129 | 43.49% | 1.833 | -11.60% | 2.21 |
| router kalman setup-aware | 130 | 40.22% | 1.729 | -13.19% | 1.78 |
| router kalman strict | 124 | 43.61% | 1.781 | -11.02% | 2.05 |

Key read:

- VN100 recent does **not confirm strong Kalman alpha**.
- `router kalman strict` is almost flat versus baseline return:
  - return: 43.49% to 43.61%
  - max drawdown improves: -11.60% to -11.02%
  - Sharpe and profit factor decline.
- `setup_aware` is harmful on VN100 recent, especially for mean reversion.

## Strategy-Level Observations

VN30:

- Kalman strict improves `mean_reversion_uptrend_ma50_v1` trade quality:
  - family avg trade rises from 1.78% to 2.65%
  - router avg trade rises from 1.56% to 2.23%
- Kalman hurts or fails to improve `breakout_after_accumulation_v3` in most variants.
- `compression_breakout_smt_v1` has too few trades to trust but often benefits in average trade quality.

VN100 recent:

- Baseline `breakout_after_accumulation_v3` is already strong; Kalman filters remove some winners.
- Setup-aware Kalman badly weakens `mean_reversion_uptrend_ma50_v1` on the router run.
- Strict Kalman is safer than setup-aware but mostly neutral, not clearly additive.

## Conclusion

Kalman has a positive effect only in a narrow use case:

> It can improve VN30 MVP router selection, mainly by tightening mean-reversion and compression entries.

It should **not** be promoted as a global MVP filter yet.

## Recommendation

Do not change the live MVP default immediately.

Next safe step:

1. Add Kalman only as a shadow metric in live signal output.
2. Run a dedicated grid on:
   - `mean_reversion_uptrend_ma50_v1`
   - `compression_breakout_smt_v1`
3. Avoid applying Kalman to `breakout_after_accumulation_v3` unless a separate breakout-only grid proves benefit.
4. If used in production, prefer strict Kalman as a risk-control option, not setup-aware as currently parameterized.

