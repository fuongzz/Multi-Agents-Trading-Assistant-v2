# Strategy And Parameter Catalog

Last updated: 2026-05-23

This file summarizes all strategy definitions and portfolio-level MVP/Flow parameters currently discoverable from local repository configs. The full row-level strategy table is in `docs/strategy_parameter_catalog.csv`.

## Scope

- Config files parsed: 8
- Strategy rows including duplicates across files: 101
- Unique strategy names: 101
- Core MVP strategy names: 9

## Config Inventory

| Config | Strategy rows |
|---|---|
| global_market_hypotheses.json | 10 |
| oil_gas_rotation_hypotheses.json | 4 |
| qmv_ichimoku_turtle_hypotheses.json | 3 |
| quant_next_cycle_hypotheses.json | 10 |
| sector_rotation_hypotheses.json | 7 |
| theme_flow_hypotheses.json | 6 |
| vn30_money_smt_hypotheses.json | 60 |
| vn30_money_smt_hypotheses_smart_exit.json | 1 |

## Active / High-Priority Portfolio Strategies

| Sleeve / Variant | Logic | Universe | Key parameters | Sizing | Status |
|---|---|---|---|---|---|
| mvp_p5 | Core MVP shared-pool | VN100 | max_positions=5; max_candidates=10; lookback=14d | equal/shared-slot sizing | HTML active |
| mvp_p4 | Core MVP shared-pool | VN100 | max_positions=4; max_candidates=10; lookback=14d | equal/shared-slot sizing | HTML active |
| flow_v2 | Flow V2 Rotation | VN100 | positions=2; rebalance_days=10; market_gate=risk_on_or_strong_neutral; pool_filter=clean_flow; score_mode=balanced | equal weight current; inverse_volatility recommended risk overlay | HTML active; verified 98.84% on 2025-01-01 to 2026-05-21 |
| Flow V2 tune5 best | Flow V2 Rotation | VN100 | positions=2; rebalance_days=8; pool_filter=loose_flow; score_mode=momentum_heavy | equal weight archived | research/grid; 129.82% |
| cycle-first p4 | Core MVP shared-pool variant | VN100 | max_positions=4; max_candidates=5/10/15 | equal/shared-slot sizing | research/grid; 114.00% |

## Core MVP Strategy Parameters

| Strategy | Source | Tags | Stop | Target | Max hold |
|---|---|---|---|---|---|
| accumulation_breakout_smt_v1 | vn30_money_smt_hypotheses.json | breakout,accumulation,money_cycle,smart_money_trace | 0.09 | 0.35 | 45 |
| breakout_55_smt_v1 | vn30_money_smt_hypotheses.json | breakout,55day,smart_money_trace,money_cycle,high_conviction | 0.09 | 0.4 | 60 |
| breakout_after_accumulation_v3 | vn30_money_smt_hypotheses.json | breakout,accumulation,smart_money_trace,money_cycle,ma_break | 0.09 | 0.35 | 45 |
| compression_breakout_smt_v1 | vn30_money_smt_hypotheses.json | compression,breakout,smart_money_trace,money_cycle,legacy_replacement | 0.07 | 0.22 | 30 |
| leader_pullback_market_healthy_v2 | vn30_money_smt_hypotheses.json | money_cycle,smart_money_trace,leader_pullback,grid_selected | 0.08 | 0.25 | 60 |
| leader_pullback_market_regime_v3 | vn30_money_smt_hypotheses.json | money_cycle,smart_money_trace,leader_pullback,market_regime | 0.08 | 0.25 | 60 |
| mean_reversion_uptrend_ma50_v1 | vn30_money_smt_hypotheses.json | mean_reversion,uptrend,ma50,smart_money_trace,money_cycle | 0.08 | 0.25 | 45 |
| money_cycle_reset_smt_confirm | vn30_money_smt_hypotheses.json | money_cycle,smart_money_trace,reset | 0.08 | 0.25 | 60 |
| smart_money_strong_market_healthy | vn30_money_smt_hypotheses.json | money_cycle,smart_money_trace,continuation | 0.08 | 0.25 | 60 |

## MPT Allocation Methods

| Method | Main parameters | Use case | Observed result |
|---|---|---|---|
| equal_weight | 1/N per entry batch | Baseline for MVP/Flow | Best for core MVP replay |
| rank_weight | weights by _edge_rank | Available in allocator | Not yet production default |
| inverse_volatility | 60-bar inverse vol + rank blend | Flow V2 risk overlay | Improved Sharpe/MaxDD for Flow active |
| minimum_variance | shrunk covariance min-var | Experiment/risk-off | Can help some Flow variants; not core MVP |
| max_sharpe | shrunk covariance mean-variance | Research only | Return slightly up in Flow active but DD worse |

## MPT Result Files

| File | Status |
|---|---|
| backtest_results\mpt_flow_v2_allocator_check_2025_to_now\summary.csv | available |
| backtest_results\mpt_impact_mvp_gt80_2025_to_now\mpt_impact_summary.csv | available |

## Unique Strategy Index

| Strategy | Status | Source | Universe | Tags | Stop | Target | Max hold |
|---|---|---|---|---|---|---|---|
| accumulation_breakout_smt_v1 | CORE_MVP | vn30_money_smt_hypotheses.json | VN30 | breakout,accumulation,money_cycle,smart_money_trace | 0.09 | 0.35 | 45 |
| breakout_55_smt_v1 | CORE_MVP | vn30_money_smt_hypotheses.json | VN100 | breakout,55day,smart_money_trace,money_cycle,high_conviction | 0.09 | 0.4 | 60 |
| breakout_after_accumulation_v3 | CORE_MVP | vn30_money_smt_hypotheses.json | VN30 | breakout,accumulation,smart_money_trace,money_cycle,ma_break | 0.09 | 0.35 | 45 |
| compression_breakout_smt_v1 | CORE_MVP | vn30_money_smt_hypotheses.json | VN100 | compression,breakout,smart_money_trace,money_cycle,legacy_replacement | 0.07 | 0.22 | 30 |
| leader_pullback_market_healthy_v2 | CORE_MVP | vn30_money_smt_hypotheses.json | VN30 | money_cycle,smart_money_trace,leader_pullback,grid_selected | 0.08 | 0.25 | 60 |
| leader_pullback_market_regime_v3 | CORE_MVP | vn30_money_smt_hypotheses.json | VN30 | money_cycle,smart_money_trace,leader_pullback,market_regime | 0.08 | 0.25 | 60 |
| mean_reversion_uptrend_ma50_v1 | CORE_MVP | vn30_money_smt_hypotheses.json | VN30 | mean_reversion,uptrend,ma50,smart_money_trace,money_cycle | 0.08 | 0.25 | 45 |
| money_cycle_reset_smt_confirm | CORE_MVP | vn30_money_smt_hypotheses.json | VN30 | money_cycle,smart_money_trace,reset | 0.08 | 0.25 | 60 |
| smart_money_strong_market_healthy | CORE_MVP | vn30_money_smt_hypotheses.json | VN30 | money_cycle,smart_money_trace,continuation | 0.08 | 0.25 | 60 |
| antonacci_dual_momentum_v1 | research/shadow | global_market_hypotheses.json | VN30 | antonacci,dual_momentum,rotation,monthly,us_classic | 0.12 | 0.4 | 20 |
| connors_rsi2_mean_reversion_v1 | research/shadow | global_market_hypotheses.json | VN30 | connors,rsi2,mean_reversion,short_term,us_classic | 0.05 | 0.08 | 8 |
| darvas_box_breakout_v1 | research/shadow | global_market_hypotheses.json | VN30 | darvas,box_breakout,us_classic | 0.07 | 0.22 | 35 |
| faber_gtaa_trend_filter_v1 | research/shadow | global_market_hypotheses.json | VN30 | faber,gtaa,trend_filter,monthly,us_classic | 0.1 | 0.25 | 20 |
| fiftytwo_week_high_momentum_v1 | research/shadow | global_market_hypotheses.json | VN30 | academic,52_week_high,momentum,us_classic | 0.1 | 0.3 | 60 |
| minervini_trend_template_v1 | research/shadow | global_market_hypotheses.json | VN30 | minervini,sepa,stage2,trend_template,us_classic | 0.08 | 0.3 | 60 |
| oneil_canslim_base_breakout_v1 | research/shadow | global_market_hypotheses.json | VN30 | oneil,canslim,base_breakout,us_classic | 0.07 | 0.25 | 40 |
| turtle_donchian_20d_breakout_v1 | research/shadow | global_market_hypotheses.json | VN30 | turtle,donchian,trend_following,breakout,us_classic | 0.1 | 0.4 | 60 |
| weinstein_stage2_breakout_v1 | research/shadow | global_market_hypotheses.json | VN30 | weinstein,stage_analysis,stage2_breakout,us_classic | 0.08 | 0.35 | 80 |
| wyckoff_spring_accumulation_v1 | research/shadow | global_market_hypotheses.json | VN30 | wyckoff,spring,accumulation,reversal,us_classic | 0.06 | 0.2 | 25 |
| oil_gas_breakout_rotation_v1 | research/shadow | oil_gas_rotation_hypotheses.json | VN100 | oil_gas,theme_rotation,breakout,volume_expansion | 0.07 | 0.25 | 30 |
| oil_gas_leader_breakout_v2 | research/shadow | oil_gas_rotation_hypotheses.json | VN100 | oil_gas,theme_rotation,leader_breakout,volume_expansion | 0.07 | 0.25 | 25 |
| oil_gas_quality_recovery_v2 | research/shadow | oil_gas_rotation_hypotheses.json | VN100 | oil_gas,theme_rotation,quality_recovery,smart_money | 0.075 | 0.22 | 22 |
| oil_gas_smart_money_recovery_v1 | research/shadow | oil_gas_rotation_hypotheses.json | VN100 | oil_gas,theme_rotation,smart_money,recovery | 0.08 | 0.22 | 25 |
| qmv_ichimoku_kijun_bounce_v1 | research/shadow | qmv_ichimoku_turtle_hypotheses.json | VN100 | qmv,ichimoku,kijun_bounce,pullback,trend_following | 0.07 | 0.25 | 45 |
| qmv_ichimoku_perfect_bull_v1 | research/shadow | qmv_ichimoku_turtle_hypotheses.json | VN100 | qmv,ichimoku,perfect_bullish,trend_following | 0.08 | 0.35 | 80 |
| qmv_turtle_ichimoku_breakout_v1 | research/shadow | qmv_ichimoku_turtle_hypotheses.json | VN100 | qmv,turtle,donchian,ichimoku,breakout,trend_following | 0.1 | 0.45 | 90 |
| cci_trend_continuation_v1 | research/shadow | quant_next_cycle_hypotheses.json | VN100 | momentum,cci,trend,smart_money_trace,research_shadow | 0.075 | 0.24 | 32 |
| ichimoku_volume_surge_breakout_v1 | research/shadow | quant_next_cycle_hypotheses.json | VN100 | breakout,ichimoku,volume,smart_money_trace,research_shadow | 0.08 | 0.29 | 32 |
| long_term_support_bounce_v1 | research/shadow | quant_next_cycle_hypotheses.json | VN100 | support,mean_reversion,ma200,long_term,smart_money_trace,research_shadow | 0.06 | 0.16 | 25 |
| long_term_support_bounce_v2_quality | research/shadow | quant_next_cycle_hypotheses.json | VN100 | support,mean_reversion,ma200,long_term,quality_gate,research_shadow | 0.06 | 0.16 | 25 |
| long_term_support_bounce_v3_above_ma200 | research/shadow | quant_next_cycle_hypotheses.json | VN100 | support,mean_reversion,ma200,above_ma200,research_shadow | 0.055 | 0.15 | 22 |
| long_term_support_bounce_v4_shallow_break | research/shadow | quant_next_cycle_hypotheses.json | VN100 | support,mean_reversion,ma200,shallow_break,research_shadow | 0.055 | 0.15 | 20 |
| relative_strength_ma50_pullback_v1 | research/shadow | quant_next_cycle_hypotheses.json | VN100 | mean_reversion,relative_strength,ma50,smart_money_trace,research_shadow | 0.07 | 0.2 | 25 |
| sector_ranked_reclaim_ma20_v1 | research/shadow | quant_next_cycle_hypotheses.json | VN100 | mean_reversion,sector_leadership,ma20,smart_money_trace,research_shadow | 0.075 | 0.24 | 30 |
| spring_leader_reversal_v1 | research/shadow | quant_next_cycle_hypotheses.json | VN100 | reversal,spring,leader,smart_money_trace,research_shadow | 0.06 | 0.17 | 20 |
| volatility_squeeze_leader_break_v1 | research/shadow | quant_next_cycle_hypotheses.json | VN100 | volatility,compression,breakout,sector_leadership,research_shadow | 0.08 | 0.27 | 35 |
| breakout_market_cycle_defensive_v1 | research/shadow | sector_rotation_hypotheses.json | WIDE | breakout,accumulation,market_cycle,defensive_gate | 0.09 | 0.35 | 45 |
| breakout_sector_anti_risk_v1 | research/shadow | sector_rotation_hypotheses.json | WIDE | breakout,accumulation,sector_rotation,anti_risk | 0.09 | 0.35 | 45 |
| breakout_sector_soft_rank_v1 | research/shadow | sector_rotation_hypotheses.json | WIDE | breakout,accumulation,sector_rotation,soft_rank | 0.09 | 0.35 | 45 |
| breakout_with_rotation_gate_v1 | research/shadow | sector_rotation_hypotheses.json | WIDE | breakout,accumulation,rotation_gate | 0.09 | 0.35 | 45 |
| rotation_early_entry_v1 | research/shadow | sector_rotation_hypotheses.json | WIDE | rotation,sector_state,early_entry,money_cycle,smart_money_trace | 0.07 | 0.22 | 30 |
| rotation_sector_leader_v1 | research/shadow | sector_rotation_hypotheses.json | WIDE | rotation,sector_state,top_down,money_cycle,smart_money_trace | 0.08 | 0.28 | 40 |
| rotation_sector_leader_v2 | research/shadow | sector_rotation_hypotheses.json | WIDE | rotation,sector_state,tightened | 0.08 | 0.28 | 35 |
| theme_flow_breakout_quality_v2 | research/shadow | theme_flow_hypotheses.json | VN100 | theme_flow,breakout_quality,smart_money,market_wide | 0.075 | 0.28 | 35 |
| theme_flow_breakout_v1 | research/shadow | theme_flow_hypotheses.json | VN100 | theme_flow,breakout,smart_money,market_wide | 0.075 | 0.25 | 30 |
| theme_flow_leader_quality_v2 | research/shadow | theme_flow_hypotheses.json | VN100 | theme_flow,leader_quality,smart_money,market_wide | 0.075 | 0.26 | 30 |
| theme_flow_recovery_v1 | research/shadow | theme_flow_hypotheses.json | VN100 | theme_flow,smart_money,recovery,market_wide | 0.08 | 0.24 | 25 |
| theme_flow_sector_leader_v1 | research/shadow | theme_flow_hypotheses.json | VN100 | theme_flow,sector_leader,smart_money,market_wide | 0.08 | 0.26 | 30 |
| theme_flow_shock_reclaim_v2 | research/shadow | theme_flow_hypotheses.json | VN100 | theme_flow,shock_reclaim,smart_money,market_wide | 0.08 | 0.24 | 18 |
| accumulation_breakout_smt_v2 | research/shadow | vn30_money_smt_hypotheses.json | VN30 | breakout,accumulation,money_cycle,smart_money_trace,grid_selected | 0.09 | 0.35 | 45 |
| adx_trend_refined_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | adx,trend,smart_money_trace,money_cycle,legacy_refined | 0.08 | 0.28 | 45 |
| any_regime_ma20_spring_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | reversal,spring,all_regime,ma20,smart_money_trace | 0.06 | 0.17 | 20 |
| any_regime_stoch_reclaim_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | mean_reversion,all_regime,stochastic,reclaim,smart_money_trace | 0.06 | 0.16 | 18 |
| any_regime_volume_reclaim_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | reversal,all_regime,volume,climax,smart_money_trace | 0.06 | 0.15 | 18 |
| benchmark_leader_breakout_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | breakout,benchmark_relative,sector_leadership,smart_money_trace,money_cycle | 0.085 | 0.32 | 42 |
| breakout_55_smt_v2 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | breakout,55day,smart_money_trace,money_cycle,value_flow,high_conviction | 0.08 | 0.35 | 50 |
| bullish_divergence_flow_confirm_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | reversal,bullish_signal,positive_divergence,money_flow,smart_money_trace | 0.055 | 0.18 | 20 |
| cci_momentum_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | momentum,breakout,cci,smart_money_trace | 0.08 | 0.26 | 38 |
| cmf_expansion_smt_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | cmf,money_flow,accumulation,smart_money_trace,money_cycle | 0.07 | 0.22 | 30 |
| cmf_expansion_smt_v2 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | cmf,money_flow,accumulation,smart_money_trace,money_cycle,value_flow | 0.07 | 0.22 | 28 |
| composite_edge_score_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN30 | score,money_cycle,smart_money_trace,relative_strength,market_regime | 0.08 | 0.3 | 60 |
| composite_edge_score_v2 | research/shadow | vn30_money_smt_hypotheses.json | VN30 | score,money_cycle,smart_money_trace,relative_strength,market_regime,grid_selected | 0.08 | 0.3 | 60 |
| dry_retest_breakout_smt_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | retest,breakout,dry_volume,smart_money_trace,money_cycle,value_flow | 0.07 | 0.24 | 32 |
| hammer_refined_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | hammer,reversal,smart_money_trace,money_cycle,legacy_refined | 0.06 | 0.18 | 18 |
| inside_bar_refined_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | inside_bar,compression,breakout,smart_money_trace,money_cycle,legacy_refined | 0.07 | 0.22 | 25 |
| kalman_ma50_pullback_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | mean_reversion,ma50,kalman,smart_money_trace,money_cycle,value_flow | 0.08 | 0.24 | 34 |
| kalman_trend_momentum_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | trend,momentum,kalman,smart_money_trace | 0.08 | 0.28 | 40 |
| leader_pullback_market_healthy | research/shadow | vn30_money_smt_hypotheses.json | VN30 | money_cycle,smart_money_trace,leader_pullback | 0.08 | 0.25 | 60 |
| leader_trend_continuation_smt_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | trend,continuation,smart_money_trace,money_cycle,legacy_replacement | 0.08 | 0.28 | 45 |
| linreg_momentum_refined_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | linreg,momentum,trend,smart_money_trace,money_cycle,legacy_refined | 0.08 | 0.28 | 40 |
| mean_reversion_uptrend_ma20_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN30 | mean_reversion,uptrend,ma20,smart_money_trace,money_cycle | 0.07 | 0.22 | 35 |
| morning_star_reversal_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | reversal,candlestick,morning_star,smart_money_trace | 0.07 | 0.2 | 25 |
| neutral_range_bounce_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | mean_reversion,neutral_regime,oversold,range,smart_money_trace | 0.06 | 0.15 | 20 |
| nr7_refined_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | nr7,compression,smart_money_trace,money_cycle,legacy_refined | 0.065 | 0.2 | 25 |
| nr7_value_breakout_smt_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | nr7,compression,breakout,smart_money_trace,money_cycle,value_flow | 0.075 | 0.28 | 34 |
| oversold_mean_reversion_refined_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | oversold,mean_reversion,smart_money_trace,money_cycle,legacy_refined | 0.065 | 0.18 | 22 |
| psar_trend_rider_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | trend,continuation,psar,smart_money_trace | 0.08 | 0.26 | 40 |
| pullback_deep_quality_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | mean_reversion,deep_pullback,smart_money_trace,money_cycle,quality_gate | 0.09 | 0.28 | 45 |
| pullback_deep_quality_v2 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | mean_reversion,deep_pullback,smart_money_trace,money_cycle,quality_gate,value_flow | 0.08 | 0.24 | 35 |
| reclaim_ma20_quality_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | mean_reversion,ma20,reclaim,smart_money_trace,money_cycle | 0.07 | 0.22 | 28 |
| reclaim_ma20_quality_v2 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | mean_reversion,ma20,reclaim,smart_money_trace,money_cycle,value_flow | 0.07 | 0.2 | 24 |
| retest_refined_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | retest,breakout,pullback,smart_money_trace,money_cycle,legacy_refined | 0.07 | 0.22 | 30 |
| risk_on_loose_breakout_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | breakout,risk_on,trend,smart_money_trace,high_frequency | 0.07 | 0.22 | 30 |
| rotation_monthly_smt_money_cycle_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN30 | rotation,monthly,smart_money_trace,money_cycle,relative_strength | 0.1 | 0.3 | 20 |
| rotation_weekly_smt_money_cycle_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN30 | rotation,weekly,smart_money_trace,money_cycle,relative_strength | 0.08 | 0.18 | 5 |
| sector_leader_pullback_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | mean_reversion,benchmark_relative,sector_leadership,ma20,smart_money_trace | 0.07 | 0.24 | 32 |
| spring_reclaim_smt_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | reversal,spring,reclaim,smart_money_trace,money_cycle,legacy_replacement | 0.07 | 0.2 | 28 |
| spring_refined_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | spring,reversal,smart_money_trace,money_cycle,legacy_refined | 0.07 | 0.2 | 28 |
| spring_washout_smt_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | spring,washout,reversal,smart_money_trace,money_cycle,value_flow | 0.07 | 0.22 | 24 |
| stoch_oversold_smt_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | mean_reversion,oversold,stochastic,smart_money_trace | 0.07 | 0.2 | 28 |
| three_white_soldiers_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | continuation,candlestick,three_white_soldiers,smart_money_trace | 0.08 | 0.28 | 35 |
| triple_ma_pullback_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | trend,pullback,triple_ma,smart_money_trace | 0.08 | 0.28 | 45 |
| vol_climax_reversal_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | reversal,volume,climax,smart_money_trace | 0.07 | 0.2 | 25 |
| volatility_squeeze_break_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | volatility,breakout,compression,smart_money_trace | 0.08 | 0.28 | 35 |
| volume_surge_breakout_smt_v1 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | volume,surge,breakout,smart_money_trace,money_cycle,climactic | 0.08 | 0.3 | 35 |
| volume_surge_breakout_smt_v2 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | volume,surge,breakout,smart_money_trace,money_cycle,climactic,value_flow | 0.08 | 0.26 | 28 |
| volume_surge_breakout_smt_v3 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | volume,surge,breakout,smart_money_trace,money_cycle,climactic,value_flow | 0.08 | 0.28 | 32 |
| volume_surge_breakout_smt_v4 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | volume,surge,breakout,smart_money_trace,money_cycle,climactic,value_flow | 0.08 | 0.28 | 32 |
| volume_surge_breakout_smt_v5 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | volume,surge,breakout,smart_money_trace,money_cycle,climactic,value_flow | 0.08 | 0.32 | 38 |
| volume_surge_breakout_smt_v6 | research/shadow | vn30_money_smt_hypotheses.json | VN100 | volume,surge,breakout,smart_money_trace,money_cycle,climactic,value_flow | 0.075 | 0.28 | 32 |
| smart_money_strong_market_healthy_smart_exit | research/shadow | vn30_money_smt_hypotheses_smart_exit.json | VN30 | money_cycle,smart_money_trace,continuation,smart_exit | 0.08 | 0.25 | 60 |

## Duplicate Strategy Names Across Configs

| Strategy | Sources |
|---|---|
| None |  |

## Notes

- `mvp_p5` and `mvp_p4` use the nine `CORE_MVP` strategy IDs, not every research hypothesis in the config files.
- `flow_v2` is a rotation strategy implemented in Python, not a hypothesis JSON strategy; its key parameters are listed in the portfolio table.
- Historical >80% variants are cataloged in `docs/mvp_registry.md`; this file focuses on definitions and parameters.
