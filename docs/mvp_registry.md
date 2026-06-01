# MVP Strategy Registry

Last updated: 2026-06-01

This file is the source of truth for "MVP" references in this repo. Read it
before answering questions about MVP results, MVP dashboards, or MVP strategy
membership.

## Active HTML MVP Dashboards

These are the five paper sleeves currently built by
`run_combined_paper_trading_demo.bat` into
`reports/combined_paper_trading_demo/index.html`.

| Sleeve ID | Label | Logic | Universe | Key Config | Dashboard |
|---|---|---|---|---|---|
| `flow_v2` | Flow V2 Rotation | Flow Money V2 high-RS flow-heavy rotation | VN100 | positions 2, rebalance 10d, `risk_on_or_strong_neutral`, `high_rs`, `flow_heavy` | `reports/combined_paper_trading_demo/flow_v2` |
| `core_mvp9_rank2` | Core MVP9 p2 rank2 compound | Core MVP9 compound-equity rank2 active sleeve | VN100 | max positions 2, max edge rank 2, max candidates 10, lookback 14d, compound equity | `reports/combined_paper_trading_demo/core_mvp9_rank2` |
| `hostile_combo_long` | Hostile Combo Long p1 | High-drawdown research/watch paper sleeve for single-stock hostile-market leadership | VN100 | top1, 100% sleeve-equity exposure, rebalance 3 sessions, score `0.35*RS60 + 0.35*RS120 + 0.20*RS20 + 0.10*liquidity_rank`, MA20 next-open exit | `reports/combined_paper_trading_demo/hostile_combo_long` |
| `flow_v2_baseline` | Baseline fresh-signal top2 | Flow V2 audited fresh-signal baseline | VN100 | positions 2, rebalance 10d, `high_rs`, `flow_heavy`, no added early-exit overlay | `reports/combined_paper_trading_demo/flow_v2_baseline` |
| `flow_v2_tiered` | Early-exit hai tầng top2 | Flow V2 audited tiered early-exit candidate | VN100 | positions 2, rebalance 10d, `high_rs`, `flow_heavy`, `flow_momentum_tiered` | `reports/combined_paper_trading_demo/flow_v2_tiered` |

There is also a standalone Flow V2 HTML dashboard:
`reports/flow_v2_production_demo_live/index.html`.

Paper ledger status from 2026-05-25: the original five dashboard sleeves were reset to
cash and are configured for a fresh parallel run beginning 2026-05-26. Each
sleeve now has a dedicated open-holdings page exported from the combined
dashboard.

On 2026-05-31 `hostile_combo_long` was added as a separate research/opportunity
paper sleeve in the combined HTML. It does not replace Flow V2 or Core MVP
logic. The first exported paper target was `VHM` on signal date 2026-05-29.
On 2026-06-01 it was capped to 25% sleeve-equity satellite exposure because
full-size longer-horizon research showed very large drawdowns.
Later on 2026-06-01 it was removed from the active paper/account set because
the capped version reduced drawdown only by sacrificing too much 2020-now
return, while 50%-75% exposure remained inferior to Flow/Core on drawdown or
risk-adjusted return. Keep it as research/watchlist only unless a new variant
beats the active Flow/Core sleeves on VN100 without mixing universes.
After user approval later on 2026-06-01, it was reopened as a full-size
high-drawdown paper/watch sleeve to observe live behavior. It remains
research-only for interpretation and must not be treated as a Core/Flow
replacement.

Later on 2026-05-31 `mvp_p4` and `mvp_p5` were deactivated from the combined
HTML/paper runner because their active paper behavior was not competitive with
the current Flow/Core stack. Their old report folders and ledger
history are retained for audit/reference, but they no longer contribute to the
combined dashboard, active paper account totals, or new paper entries.

After that, `core_mvp9_rank2` was added as a clean Core MVP sleeve without
reusing the old `mvp_p4`/`mvp_p5` identities. It uses the research-selected
Core MVP9 p2/rank2 compound-equity contract. Its sleeve activation date is
stored in `paper_ledger_meta.json` as `2026-05-30`, so old signals before
activation are not backfilled into paper holdings.

On 2026-05-31 the `mvp_p4` and `mvp_p5` dashboard slots were switched from the
old p4/p5 fixed-size Core MVP contract to the research-selected Core MVP9
p2/rank2 compound-equity contract. Existing open paper positions from the old
slots are not forcibly reset by this config change; new signals and new fills
use max positions 2, edge rank <= 2, and current-equity sizing.

The dashboard comparison export for 100m VND per sleeve is:
`reports/combined_paper_trading_demo/strategy_backtest_comparison.csv`.

Correction recorded on 2026-05-25: the Core MVP `combos_unbiased` rows use
the baseline `_edge_exit_decision` path and do not include the
`PositionStateExitAgent` that is enabled in current paper ledger operation.
They are verified baseline-exit comparisons, not exact current-paper contract
results. The closest existing position-exit-agent reference ends on
2026-05-19; a through-2026-05-22 exact paper-contract rerun remains pending.

| Sleeve | Period | Return | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| `mvp_p5`, baseline exit only | 2025-01-01 to 2026-05-22 | 45.47% | 1.629 | -12.73% |
| `mvp_p5`, position-exit-agent reference | 2025-01-01 to 2026-05-19 | 102.99% | 2.914 | -11.40% |
| `mvp_p4`, baseline exit only | 2025-01-01 to 2026-05-22 | 63.53% | 2.122 | -11.95% |
| `mvp_p4`, position-exit-agent reference | 2025-01-01 to 2026-05-19 | 114.00% | 3.055 | -10.19% |
| `flow_v2_baseline` | 2025-01-01 to 2026-05-22 | 85.44% | 1.972 | -10.57% |
| `flow_v2_baseline` | 2020-01-01 to 2026-05-22 | 884.55% | 1.434 | -31.11% |
| `flow_v2_tiered` | 2025-01-01 to 2026-05-22 | 83.48% | 1.944 | -11.00% |
| `flow_v2_tiered` | 2020-01-01 to 2026-05-22 | 923.95% | 1.527 | -22.97% |

Retest recorded on 2026-05-31 using VN100 data through 2026-05-29, corrected
VND execution, top2, rebalance 10 sessions, `high_rs`, `flow_heavy`, 100m VND,
and source `reports/flow_v2_risk_management_experiment_vn100_2025_to_2026-05-29/summary.csv`:

| Sleeve | Period | Return | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| `flow_v2_baseline` | 2025-01-01 to 2026-05-29 | 85.44% | 1.958 | -10.57% |
| `flow_v2_tiered` / enhanced | 2025-01-01 to 2026-05-29 | 83.48% | 1.930 | -11.00% |

Full-history retest source:
`reports/flow_v2_risk_management_experiment_vn100_2020_to_2026-05-29/summary.csv`.

| Sleeve | Period | Return | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| `flow_v2_baseline` | 2020-01-01 to 2026-05-29 | 884.55% | 1.432 | -31.11% |
| `flow_v2_tiered` / enhanced | 2020-01-01 to 2026-05-29 | 923.95% | 1.525 | -22.97% |

`flow_v2` has a separate paper exit contract from the two audited Flow
variants; do not map audited results onto that sleeve. Exact current-paper
contract Core MVP9 fresh-start results for `mvp_p4` and `mvp_p5`, including
the enabled position-exit agent, were rerun on 2026-05-31 through 2026-05-29.
Source:
`reports/combined_paper_trading_demo/current_paper_contract_backtest/summary.csv`.

| Sleeve | Period | Return | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| `mvp_p5`, exact current-paper contract, fresh 2025 start | 2025-01-01 to 2026-05-29 | 25.50% | 1.188 | -13.54% |
| `mvp_p4`, exact current-paper contract, fresh 2025 start | 2025-01-01 to 2026-05-29 | 25.40% | 1.114 | -16.10% |

After the 2026-05-31 dashboard-slot switch, the active `mvp_p4` and `mvp_p5`
slots both use Core MVP9 p2/rank2 compound-equity. Source:
`reports/combined_paper_trading_demo/current_paper_contract_backtest/summary.csv`.

| Sleeve | Period | Return | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| `mvp_p5`, active p2/rank2 compound-equity | 2025-01-01 to 2026-05-29 | 95.78% | 2.132 | -19.12% |
| `mvp_p4`, active p2/rank2 compound-equity | 2025-01-01 to 2026-05-29 | 95.78% | 2.132 | -19.12% |

Full-history exact current-paper contract rerun recorded on 2026-05-31 using
VN100 data through 2026-05-29, 100m VND initial capital, max candidates 10,
and the enabled position-exit agent. Source:
`reports/combined_paper_trading_demo/current_paper_contract_backtest_2020_to_2026-05-29/summary.csv`.

| Sleeve | Period | Return | Sharpe | Max DD |
|---|---:|---:|---:|---:|
| `mvp_p5`, exact current-paper contract | 2020-01-01 to 2026-05-29 | 124.56% | 1.144 | -12.56% |
| `mvp_p4`, exact current-paper contract | 2020-01-01 to 2026-05-29 | 134.75% | 1.163 | -14.11% |

### Core MVP Reinvestment Research

Research rerun recorded on 2026-05-31. These are **not promoted paper sleeves**.
They test whether Core MVP entries can become comparable to Flow V2 by changing
position sizing from fixed initial-capital slots to current-equity
reinvestment. Source:
`reports/mvp_current_paper_contract_compound_p1_optimization_vn100_2020_to_2026-05-29/summary.csv`.

| Candidate | Period | Return | Sharpe | Max DD | 2025-now Return | Note |
|---|---:|---:|---:|---:|---:|---|
| `top5_full_history_net`, p1, compound equity | 2020-01-01 to 2026-05-29 | 1835.74% | 1.500 | -55.13% | 188.85% | High return, drawdown too large for direct promotion |
| `core_mvp9`, p2, compound equity | 2020-01-01 to 2026-05-29 | 743.26% | 1.450 | -29.03% | 43.99% | Closest balanced Core MVP candidate tested |
| `core_minus_breakout_accum`, p2, compound equity | 2020-01-01 to 2026-05-29 | 705.98% | 1.422 | -29.03% | 43.09% | Similar to core p2; no clear improvement |
| `cycle_breakout_compression_accum`, p1, compound equity | 2020-01-01 to 2026-05-29 | 1342.20% | 1.422 | -57.97% | 146.63% | High return, drawdown too large for direct promotion |

Interpretation: reinvesting equity and concentrating to two positions is the
main lever that moves Core MVP toward Flow V2-like full-history returns. One
position can exceed Flow V2 return, but drawdown around -55% makes it a research
candidate only. Further work should focus on risk gates for compound p2 before
changing the active HTML/paper sleeves.

Risk-gate follow-up source:
`reports/mvp_current_paper_contract_risk_gate_optimization_vn100_2020_to_2026-05-29/summary.csv`.

| Candidate | Period | Return | Sharpe | Max DD | 2025-now Return | Note |
|---|---:|---:|---:|---:|---:|---|
| `core_mvp9`, p2, compound equity, rank2 only | 2020-01-01 to 2026-05-29 | 855.34% | 1.523 | -29.26% | 48.56% | Best balanced Core MVP candidate so far; return/Sharpe close to Flow V2 enhanced but drawdown still worse |

Indicator-complexity research recorded on 2026-05-31. This is **not a promoted
paper sleeve** and uses the generic edge_lab research portfolio contract, not
the exact current paper ledger. It compares simple price rules, imported
classic indicators, QMV/Ichimoku, and local MVP9 money-flow logic on VN100.
Source: `reports/vn_market_indicator_complexity_vn100/summary.csv`.

| Candidate | Period | Return | Sharpe | Max DD | Note |
|---|---:|---:|---:|---:|---|
| `complex_local_money_flow_mvp9`, generic research contract | 2025-01-01 to 2026-05-29 | 61.16% | 1.854 | -17.54% | Confirms local flow/regime/RS logic remains useful; research-only because contract differs from active paper sleeves |

### Hostile-Market Combo-Long Research

Research rerun recorded on 2026-05-31. This is **not promoted to paper**. It
formalizes the hostile-market opportunity sleeve idea: VN100, top-1 rotation,
rebalance every 3 sessions, score `0.35*RS60 + 0.35*RS120 + 0.20*RS20 +
0.10*liquidity_rank`, MA20 exit, 100m VND initial capital, corrected VND
execution, fees, slippage, and board lots. Source:
`reports/hostile_combo_long_vn100_2025_check/summary.csv`.

| Candidate | Period | Return | Sharpe | Max DD | Note |
|---|---:|---:|---:|---:|---|
| `hostile_combo_long`, p1/r3/MA20 | 2025-01-01 to 2026-05-29 | 73.68% | 1.027 | -40.00% | Research-only; exceeds 60% return but drawdown is too large for full-size promotion |

Longer-horizon robustness source:
`reports/hostile_combo_long_vn100_long/summary.csv`.

| Start | End | Return | Sharpe | Max DD | Note |
|---|---:|---:|---:|---:|---|
| 2016-01-01 | 2026-05-29 | 16.67% | 0.255 | -76.42% | Not robust over the full local history |
| 2018-01-01 | 2026-05-29 | -34.22% | 0.125 | -79.75% | Fails when started before the favorable 2020+ cycle |
| 2020-01-01 | 2026-05-29 | 1330.30% | 1.096 | -52.48% | High return, unacceptable drawdown for core allocation |
| 2025-12-01 | 2026-05-29 | 74.92% | 2.268 | -24.21% | Recent hostile-market opportunity capture only |

Satellite retest recorded on 2026-06-01 after capping the active HTML/paper
contract to 25% sleeve-equity exposure. This cap is no longer active paper;
it is retained as research evidence. Source:
`reports/hostile_combo_long_satellite_vn100_2026-06-01/summary.csv`.

| Start | End | Return | Sharpe | Max DD | Note |
|---|---:|---:|---:|---:|---|
| 2016-01-01 | 2026-05-29 | 27.12% | 0.261 | -29.02% | Satellite cap materially reduces full-size drawdown but remains research/opportunity only |
| 2018-01-01 | 2026-05-29 | 8.12% | 0.138 | -31.05% | Still weak when started before the favorable 2020+ cycle |
| 2020-01-01 | 2026-05-29 | 139.16% | 1.112 | -14.34% | Better dashboard risk fit than full-size p1 |
| 2025-01-01 | 2026-05-29 | 16.69% | 0.891 | -12.43% | Recent return is lower, but DD falls from -40.00% to -12.43% |
| 2025-12-01 | 2026-05-29 | 12.18% | 1.763 | -7.60% | Recent hostile opportunity capture at capped size |

Margin-agent research is exported separately at
`reports/margin_agent_research_vn100_2020_to_2026-05-22/index.html`.
It is not activated in the paper/live dashboard. The study uses 100m VND,
12% annual borrowing cost, 0.20% one-side incremental financed-trading cost
(including turnover of the financed tranche on underlying rebalances),
and a forced-deleverage cooldown after a -20% financed-account drawdown.
The leverage decision for each session uses observations only through the
previous session. Verified highlights:

| Sleeve | Period | Margin policy | Return | Sharpe | Max DD | Forced deleverage |
|---|---:|---|---:|---:|---:|---:|
| `mvp_p4` | 2025-01-01 to 2026-05-22 | static max 1.50x | 82.51% | 1.947 | -16.56% | 0 |
| `mvp_p5` | 2025-01-01 to 2026-05-22 | static max 1.50x | 55.58% | 1.440 | -17.47% | 0 |
| `flow_v2_baseline` | 2025-01-01 to 2026-05-22 | static max 1.50x | 124.76% | 1.820 | -15.51% | 0 |
| `flow_v2_tiered` | 2025-01-01 to 2026-05-22 | static max 1.50x | 119.12% | 1.775 | -16.23% | 0 |
| `flow_v2_baseline` | 2020-01-01 to 2026-05-22 | static max 1.25x | 1173.03% | 1.361 | -34.81% | 23 |
| `flow_v2_tiered` | 2020-01-01 to 2026-05-22 | static max 1.50x | 1129.05% | 1.277 | -31.68% | 33 |

`flow_v2` legacy is excluded from exact margin comparison because its live
paper exit contract is not identical to either audited Flow curve. Full-history
Core `mvp_p4` and `mvp_p5` margin rows remain unavailable until the exact
underlying full-history sleeves are rerun.

`VN100` is the default operating universe. `Liquid150` means the local top-150
liquid-stock research comparison universe in
`reports/liquid_universe_2020_now/top150_symbols_only.csv`; keep Liquid150
results separate and do not treat it as the default.

## Core MVP Edge Strategy Set

The `mvp_p5` and `mvp_p4` sleeves use the production core MVP strategy set from
`multiagents_trading_assistant.edge_lab.strategy_sleeves.MVP_EDGE_STRATEGIES`:

- `leader_pullback_market_regime_v3`
- `leader_pullback_market_healthy_v2`
- `breakout_55_smt_v1`
- `money_cycle_reset_smt_confirm`
- `smart_money_strong_market_healthy`
- `accumulation_breakout_smt_v1`
- `breakout_after_accumulation_v3`
- `compression_breakout_smt_v1`
- `mean_reversion_uptrend_ma50_v1`

Do not confuse this production core set with all 60 hypotheses in
`multiagents_trading_assistant/edge_lab/configs/vn30_money_smt_hypotheses.json`.

## Verified Backtest Results From 2025-01-01

Unless otherwise stated, the end date is 2026-05-22 and the source data is local
parquet data in this repo.

### Active HTML MVP / Portfolio-Level Results

Flow V2 backtest rows produced before the 2026-05-24 execution-price fix
used parquet prices in thousand-VND units as if they were VND when sizing
orders. They are retained below only as invalidated historical references
until rerun with corrected execution units.

| MVP / Portfolio | Universe | Period | Return | Sharpe | Max DD | Source |
|---|---|---:|---:|---:|---:|---|
| Flow V2 Rotation, prior HTML-config backtest, invalidated by price-unit execution bug (`high_rs`, `flow_heavy`, capital 100m) | VN100 | 2025-01-01 to 2026-05-22 | invalid (prior 99.04%) | invalid | invalid | `backtest_results/flow_v2_rotation_production_like/flow_v2_promoted_highrs_flowheavy_2025now_vn100_2025-01-01_2026-05-22/summary.csv` |
| Flow V2 Rotation, prior Liquid150 comparison, invalidated by price-unit execution bug | Liquid150 | 2025-01-01 to 2026-05-22 | invalid (prior 63.93%) | invalid | invalid | `backtest_results/flow_v2_rotation_production_like/flow_v2_active_liquid150_2025now_liquid150_2025-01-01_2026-05-22/summary.csv` |
| MVP p5 / entry market gate variant | VN100 | 2025-01-01 to 2026-05-14 | 99.37% | 2.898 | -11.40% | `backtest_results/entry_market_gate_variants/mvp_2025_to_20260514_vn100_2025-01-01_2026-05-14_p5/summary.csv` |
| MVP p5 / soft gate variant | VN100 | 2025-01-01 to 2026-05-16 | 98.48% | 2.885 | -11.40% | `backtest_results/entry_market_gate_variants/mvp_2025now_soft_gate_vn100_2025-01-01_2026-05-16_p5/summary.csv` |
| MVP p5 / rerun fix variant | VN100 | 2025-01-01 to 2026-05-16 | 95.30% | 2.796 | -11.40% | `backtest_results/entry_market_gate_variants/rerun_fix_20260518_mvp_2025now_vn100_2025-01-01_2026-05-16_p5/summary.csv` |
| Core MVP9 combined, MPT `inverse_volatility` | VN100 | 2025-01-01 to 2026-05-22 | 71.71% | 2.166 | -9.12% | `backtest_results/mpt_allocator_comparison_2025_to_now_vn100_mvp9/allocator_summary.csv` |
| Core MVP9 combined, baseline `equal_weight` | VN100 | 2025-01-01 to 2026-05-22 | 70.37% | 2.165 | -9.12% | `backtest_results/mpt_allocator_comparison_2025_to_now_vn100_mvp9/allocator_summary.csv` |
| Flow V2 fresh signals, baseline top2, corrected VND execution | VN100 | 2025-01-01 to 2026-05-22 | 87.76% | 1.969 | -11.20% | `reports/flow_v2_reversal_guard_experiment_vn100_2025_to_2026-05-22/summary.csv` |
| Flow V2 fresh signals, early-exit flow/momentum break top2, corrected VND execution | VN100 | 2025-01-01 to 2026-05-22 | 85.99% | 1.946 | -11.19% | `reports/flow_v2_reversal_guard_experiment_vn100_2025_to_2026-05-22/summary.csv` |
| Flow V2 fresh signals, early-exit tiered top2 | VN100 | 2025-01-01 to 2026-05-22 | 86.54% | 1.954 | -11.17% | `reports/flow_v2_risk_management_experiment_vn100_2025_to_2026-05-22/summary.csv` |
| Flow V2 fresh signals, early-exit plus neutral 70% exposure top2 | VN100 | 2025-01-01 to 2026-05-22 | 85.99% | 1.946 | -11.19% | `reports/flow_v2_risk_management_experiment_vn100_2025_to_2026-05-22/summary.csv` |
| Flow V2 fresh signals, early-exit plus inverse ATR top2 | VN100 | 2025-01-01 to 2026-05-22 | 80.14% | 1.882 | -11.32% | `reports/flow_v2_risk_management_experiment_vn100_2025_to_2026-05-22/summary.csv` |
| Flow V2 fresh signals, early-exit plus distinct industry top2 | VN100 | 2025-01-01 to 2026-05-22 | 67.16% | 1.621 | -11.17% | `reports/flow_v2_risk_management_experiment_vn100_2025_to_2026-05-22/summary.csv` |
| Flow V2 fresh signals, combined position-risk overlays top2 | VN100 | 2025-01-01 to 2026-05-22 | 63.34% | 1.586 | -11.29% | `reports/flow_v2_risk_management_experiment_vn100_2025_to_2026-05-22/summary.csv` |
| Core MVP9 combined, baseline `equal_weight` | VN30 | 2025-01-01 to 2026-05-22 | 20.55% | 0.880 | -13.87% | `backtest_results/mvp_strategy_breakdown_2025_to_now/mvp_strategy_summary.csv` |

### Active Flow V2 Full-History Validation

| MVP / Portfolio | Universe | Period | Return | Sharpe | Max DD | Source |
|---|---|---:|---:|---:|---:|---|
| Flow V2 prior 100m backtest, invalidated by price-unit execution bug (`high_rs`, `flow_heavy`, top2, rebalance10) | VN100 | 2020-01-01 to 2026-05-22 | invalid (prior 631.84%) | invalid | invalid | `backtest_results/flow_v2_rotation_production_like/flow_v2_promoted_highrs_flowheavy_2020now_vn100_2020-01-01_2026-05-22/summary.csv` |
| Flow V2 frozen historical targets replayed with corrected VND execution, capital 1bn | VN100 | 2020-01-01 to 2026-05-22 | 712.03% | 1.354 | -28.99% | `reports/flow_v2_high_performance_capital_1bn_vn100_2020_to_2026-05-22/summary.csv` |
| Flow V2 fresh signals, baseline top2 corrected VND execution, capital 1bn, audited | VN100 | 2020-01-01 to 2026-05-22 | 886.55% | 1.430 | -31.24% | `reports/flow_v2_detailed_audit_vn100_2020_to_2026-05-22/full_period_summary.csv` |
| Flow V2 fresh signals, early-exit flow/momentum break top2, capital 1bn, rerun after position-limit fix | VN100 | 2020-01-01 to 2026-05-22 | 873.70% | 1.484 | -24.54% | `reports/flow_v2_reversal_guard_experiment_vn100_2020_to_2026-05-22/summary.csv` |
| Flow V2 fresh signals, early-exit tiered top2, capital 1bn, audited | VN100 | 2020-01-01 to 2026-05-22 | 927.78% | 1.523 | -23.01% | `reports/flow_v2_detailed_audit_vn100_2020_to_2026-05-22/full_period_summary.csv` |

The corrected 1bn figure is an execution replay of the previously saved target
decisions, preserving the strategy's historical choices while fixing order
sizing, board-lot handling, and VND price units. It is not a freshly selected
signal run from mutable feature inputs.

The fresh-signal reversal-guard experiment recomputes choices from synchronized
OHLCV, Money Cycle, and Smart Money Trace data. Its early-exit variant generates
the exit after close on date T and executes only at open on T+1; it is not a
same-bar liquidation rule.

### Flow V2 Market-Gate Comparison Through 2026-05-26

On 2026-05-26 a corrected-VND research comparison was run with 100m VND,
VN100, top2, rebalance every 10 sessions, `high_rs`, `flow_heavy`, and the
same open-T+1 execution contract used for the dashboard family. `market_gate=none`
removes only the market regime filter and retains stock-level quality filters.
This is research output, not a promoted paper sleeve.

Source:
`reports/flow_v2_gate_comparison_vn100_2020_to_2026-05-26/summary.csv`.

| Period | Gate | Exit mode | Return | Sharpe | Max DD |
|---|---|---|---:|---:|---:|
| 2025-01-01 to 2026-05-26 | none | baseline | 126.34% | 1.806 | -30.38% |
| 2025-01-01 to 2026-05-26 | `risk_on_or_strong_neutral` | baseline | 85.44% | 1.967 | -10.57% |
| 2025-01-01 to 2026-05-26 | strict | baseline | 74.75% | 1.811 | -11.18% |
| 2020-01-01 to 2026-05-26 | none | baseline | 1141.26% | 1.254 | -46.43% |
| 2020-01-01 to 2026-05-26 | none | `flow_momentum_tiered` | 983.01% | 1.273 | -46.48% |
| 2020-01-01 to 2026-05-26 | `risk_on_or_strong_neutral` | baseline | 884.55% | 1.433 | -31.11% |
| 2020-01-01 to 2026-05-26 | `risk_on_or_strong_neutral` | `flow_momentum_tiered` | 923.95% | 1.526 | -22.97% |

Interpretation: removing the market gate lifts raw return by maintaining
exposure through weak regimes, but materially increases drawdown and lowers
risk-adjusted quality. The active gate remains preferable for paper trading;
the no-gate signal list is useful as display-only discovery rather than as an
execution replacement.

Retest through 2026-05-29 confirms the active gate is not overly strict for
the enhanced/tiered Flow V2 sleeve. Source:
`reports/flow_v2_gate_comparison_vn100_2020_to_2026-05-29/summary.csv`.

| Period | Gate | Exit mode | Return | Sharpe | Max DD |
|---|---|---|---:|---:|---:|
| 2025-01-01 to 2026-05-29 | none | `flow_momentum_tiered` | 114.79% | 1.731 | -38.23% |
| 2025-01-01 to 2026-05-29 | `base` | `flow_momentum_tiered` | 70.15% | 1.658 | -15.68% |
| 2025-01-01 to 2026-05-29 | `risk_on_or_strong_neutral` | `flow_momentum_tiered` | 83.48% | 1.930 | -11.00% |
| 2025-01-01 to 2026-05-29 | strict | `flow_momentum_tiered` | 73.62% | 1.781 | -11.14% |
| 2020-01-01 to 2026-05-29 | none | `flow_momentum_tiered` | 969.59% | 1.265 | -46.48% |
| 2020-01-01 to 2026-05-29 | `base` | `flow_momentum_tiered` | 943.48% | 1.505 | -22.97% |
| 2020-01-01 to 2026-05-29 | `risk_on_or_strong_neutral` | `flow_momentum_tiered` | 923.95% | 1.525 | -22.97% |
| 2020-01-01 to 2026-05-29 | strict | `flow_momentum_tiered` | 563.30% | 1.337 | -22.94% |

Interpretation: no-gate is too loose because drawdown expands materially.
Strict is too tight because it sacrifices too much return. The active
`risk_on_or_strong_neutral` gate has the best recent risk-adjusted profile and
the best full-history Sharpe among enhanced/tiered gate variants, while
preserving the same full-history drawdown as `base`.

The 2026-05-24 detailed audit added `signal_date` traces and fixed a position
limit defect: when `VIX` lacked an executable open price on 2020-12-31, older
fresh-signal runs could retain `VIX` and still buy both new top2 candidates,
temporarily holding three symbols. The corrected engine reserves a slot for a
position that cannot be sold. The audit passes T+1, top2, non-negative cash,
and VND price-unit checks. It still uses the VN100 list resolved at run time,
so survivorship bias is not fully removed.

### Research / Grid Variants With Return Greater Than 80%

These are historical research references. Flow V2 rows in this subsection
predate the VND execution-price fix and must be rerun before being treated as
verified comparisons.

| Variant | Universe | Period | Return | Sharpe | Max DD | Source |
|---|---|---:|---:|---:|---:|---|
| Flow V2 high-RS, positions 2, rebalance 10, `high_rs`, `balanced` | Liquid150 | 2025-01-01 to 2026-05-22 | 80.32% | 1.505 | -16.64% | `backtest_results/flow_v2_rotation_production_like/flow_v2_high_cases_compare_2025now_liquid150_2025-01-01_2026-05-22/summary.csv` |
| Flow V2 sector-heavy, positions 2, rebalance 8, `loose_flow`, `sector_heavy` | Liquid150 | 2025-01-01 to 2026-05-22 | 74.52% | 1.553 | -19.55% | `backtest_results/flow_v2_rotation_production_like/flow_v2_high_cases_compare_2025now_liquid150_2025-01-01_2026-05-22/summary.csv` |
| Flow V2 sector-heavy, positions 2, rebalance 8, `clean_flow`, `sector_heavy` | Liquid150 | 2025-01-01 to 2026-05-22 | 69.83% | 1.506 | -19.55% | `backtest_results/flow_v2_rotation_production_like/flow_v2_high_cases_compare_2025now_liquid150_2025-01-01_2026-05-22/summary.csv` |
| Flow V2 balanced, positions 2, rebalance 10, `loose_flow`, `balanced` | Liquid150 | 2025-01-01 to 2026-05-22 | 69.82% | 1.419 | -16.88% | `backtest_results/flow_v2_rotation_production_like/flow_v2_high_cases_compare_2025now_liquid150_2025-01-01_2026-05-22/summary.csv` |
| Flow V2 tune5, positions 2, rebalance 8, `loose_flow`, `momentum_heavy` | VN100 | 2025-01-01 to 2026-05-19 | 129.82% | 2.571 | -11.74% | `backtest_results/flow_v2_rotation_production_like/flow_v2_rotation_prodlike_tune5_filter_score_2025now_vn100_2025-01-01_2026-05-19/summary.csv` |
| Flow V2 tune5, positions 2, rebalance 8, `loose_flow`, `balanced` | VN100 | 2025-01-01 to 2026-05-19 | 120.95% | 2.562 | -11.74% | `backtest_results/flow_v2_rotation_production_like/flow_v2_rotation_prodlike_tune5_filter_score_2025now_vn100_2025-01-01_2026-05-19/summary.csv` |
| Flow V2 tune5, positions 2, rebalance 8, `clean_flow`, `sector_heavy` | VN100 | 2025-01-01 to 2026-05-19 | 118.39% | 2.344 | -11.16% | `backtest_results/flow_v2_rotation_production_like/flow_v2_rotation_prodlike_tune5_filter_score_2025now_vn100_2025-01-01_2026-05-19/summary.csv` |
| Cycle-first strategy-specific p4 baseline/no-weak2 | VN100 | 2025-01-01 to 2026-05-19 | 114.00% | 3.055 | -10.19% | `backtest_results/cycle_first_strategy_specific/cycle_specific_baseline_grid_2025now_vn100_2025-01-01_2026-05-19/summary.csv` |
| Flow V2 current-family config, positions 2, rebalance 10, `clean_flow`, `balanced` | VN100 | 2025-01-01 to 2026-05-19 | 95.86% | 2.216 | -11.07% | `backtest_results/flow_v2_rotation_production_like/flow_v2_rotation_prodlike_tune5_filter_score_2025now_vn100_2025-01-01_2026-05-19/summary.csv` |

Additional verified MVP/grid rows above 80%:

- Flow V2 retune best-candidates file has four rows above 80%:
  `clean_flow/balanced` 98.84%, `clean_flow/sector_heavy` 88.68%,
  `loose_flow/sector_heavy` 86.41%, and `loose_flow/balanced` 86.38%.
  Source:
  `backtest_results/flow_v2_rotation_production_like/flow_v2_retune_best_candidates_2025now_vn100_2025-01-01_2026-05-21/summary.csv`.
- Flow V2 tune5 file has 37 rows above 80%; top rows include 129.82%,
  120.95%, 118.39%, 113.06%, 112.76%, 111.45%, 110.54%,
  108.79%, 105.58%, and 102.68%. Source:
  `backtest_results/flow_v2_rotation_production_like/flow_v2_rotation_prodlike_tune5_filter_score_2025now_vn100_2025-01-01_2026-05-19/summary.csv`.
- Cycle-first overlay file has six rows above 80%: `stock_cycle_tiebreak`
  103.07%, `weak_cycle_filter` 102.99%, `weak_cycle_penalty` 102.99%,
  `none` 102.99%, `quality_or_penalty` 88.52%, and
  `small_cycle_tiebreak` 82.46%. Source:
  `backtest_results/cycle_first_overlays/cycle_overlay_2025now_vn100_2025-01-01_2026-05-19/summary.csv`.
- Cycle-first strategy-specific baseline grid has eight rows above 80%:
  p4/c5-c15 114.00%, p5/c5-c15 102.99%, and p6/c10-c15 83.11%.
  Source:
  `backtest_results/cycle_first_strategy_specific/cycle_specific_baseline_grid_2025now_vn100_2025-01-01_2026-05-19/summary.csv`.

### Individual Core MVP Strategies With Return Greater Than 60%

| Strategy | Universe | Period | Return | Sharpe | Max DD | Source |
|---|---|---:|---:|---:|---:|---|
| `breakout_55_smt_v1` | VN100 | 2025-01-01 to 2026-05-22 | 63.96% | 1.528 | -14.68% | `backtest_results/vn100_mvp_strategy_breakdown_2025_to_now/vn100_mvp_strategy_summary.csv` |
| `breakout_55_smt_v1` | VN30 | 2025-01-01 to 2026-05-22 | 62.41% | 1.589 | -11.45% | `backtest_results/mvp_strategy_breakdown_2025_to_now/mvp_strategy_summary.csv` |

## Answering Rules

- If the user says "MVP" without qualifiers, assume the active HTML MVP set:
  `mvp_p5`, `mvp_p4`, and `flow_v2`.
- If the user asks about strategy membership, use the nine strategy IDs listed
  above for core MVP edge strategies.
- If the user asks about return from 2025-01-01, cite the verified tables above
  and specify universe and config. Do not mix VN30, VN100, and Liquid150 results.
- If a new MVP is promoted or a new strategy exceeds 60% return from 2025-01-01,
  update this file in the same turn.
