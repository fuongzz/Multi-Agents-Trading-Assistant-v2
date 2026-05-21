# Fintech Lab Research Spec: Adaptive ExitAgent

Date: 2026-05-20

## 1. Context

The current Quant system already has entry strategies and static exit rules. The baseline backtest exits positions using fixed risk parameters such as stop loss, ATR stop, trailing ATR stop, take profit, maximum holding bars, and final end-of-data close.

This works as a controlled baseline, but it has two known weaknesses:

1. Some losing positions show weakness before they hit stop loss, causing avoidable drawdown and wasted capital slots.
2. Some winning positions are exited too mechanically, preventing the system from holding stronger trend/ranking winners longer.

The requested module is an adaptive `ExitAgent` that reviews open positions after each daily bar and proposes exit/risk-management decisions without changing the entry engine.

## 2. Objective

Design, implement, and backtest an adaptive ExitAgent that can:

- Exit early when a position thesis is weakening before the static stop is hit.
- Exit or deprioritize positions that go sideways for too long and consume portfolio slots.
- Raise stops or protect profit when a position becomes extended or starts losing momentum.
- Hold runners longer when trend, volume, and market context remain favorable.
- Improve risk-adjusted return without materially reducing absolute return.

The module must be deterministic and backtestable. If a machine-learning component is proposed, it must still produce auditable decision reasons and must be benchmarked against deterministic policies.

## 3. Current Baseline Exit Logic

Baseline trade flow:

- Signal is generated on day `T`.
- Entry is executed on the next tradable bar, usually `T+1`, using the configured open-price/slippage model.
- Exit decisions are based only on data available up to the current bar.
- Exit is executable on the next tradable bar unless the existing backtest harness explicitly models another fill rule.

Current exit priority:

1. Static percent stop loss.
2. Initial ATR stop.
3. Trailing ATR stop after activation threshold.
4. Static take profit.
5. Maximum holding bars.
6. Final close at end of data.

Existing implementation references:

- `multiagents_trading_assistant/agentic/position_exit_agent.py`
- `scripts/backtest_position_exit_agent.py`
- `tests/test_position_exit_agent.py`

## 4. Sample Data Package

Use the sample data in:

- `data/samples/exit_agent_lab/sample_trades.csv`
- `data/samples/exit_agent_lab/sample_trade_paths.csv`
- `data/samples/exit_agent_lab/baseline_summary.csv`
- `data/samples/exit_agent_lab/README.md`

The sample contains representative trades from:

- YTD window: `2025-01-01` to `2026-05-16`
- Validation window: `2022-01-01` to `2024-12-31`

The sample is for interface design and initial hypothesis testing. Final evaluation must use the full backtest universe, not only this sample.

## 5. Required Input Contract

The ExitAgent should support an input structure equivalent to the following fields.

### 5.1 Position State

- `symbol`
- `strategy_name`
- `setup_type`
- `signal_date`
- `entry_date`
- `entry_price`
- `shares`
- `entry_value`
- `current_stop`
- `current_take_profit`
- `highest_price_since_entry`
- `lowest_price_since_entry`
- `holding_bars`
- `unrealized_pct`
- `unrealized_atr`
- `edge_score`
- `edge_rank`
- `edge_rank_score`
- `edge_risk`

### 5.2 Price and Feature History

Daily, causal history up to `as_of_date` only:

- `date`
- `open`, `high`, `low`, `close`, `volume`
- `atr14`, `atr_pct`
- `ma20`, `ma50`
- `distance_to_ma20`, `distance_to_ma50`
- `volume_ratio_20`
- `range_pct`
- `close_location`
- optional support/resistance and market-regime features if available

### 5.3 Portfolio State

- `cash`
- `equity`
- `open_positions`
- `max_positions`
- `available_slots`
- `gross_exposure`
- `current_drawdown_pct`
- `slot_pressure`: whether new high-quality signals are competing for capital

### 5.4 Market State

- VNINDEX trend/regime state.
- Market breadth if available.
- Panic/crash guard status if available.
- Sector strength if available.

## 6. Required Output Contract

The ExitAgent must return an auditable decision object.

Required fields:

- `action`: one of `HOLD`, `EXIT_NEXT_OPEN`, `RAISE_STOP`, `HOLD_RUNNER`, `REDUCE_NEXT_OPEN`
- `confidence`: float from `0.0` to `1.0`
- `reason_codes`: list of stable reason codes
- `reason_text`: short human-readable explanation
- `suggested_stop`: nullable float
- `suggested_take_profit`: nullable float
- `size_fraction`: float, defaults to `1.0`
- `valid_until`: nullable date
- `next_bar_actionable`: boolean
- `risk_invariant_passed`: boolean

Recommended reason codes:

- `THESIS_BROKEN`
- `MOMENTUM_DECAY`
- `ATR_BREAKDOWN`
- `MA20_LOST`
- `WEAK_CLOSE_LOCATION`
- `VOLUME_DRY_UP`
- `DEAD_CAPITAL`
- `PROFIT_PROTECT`
- `RUNNER_CONFIRMED`
- `MARKET_RISK_OFF`
- `NO_ACTION`

## 7. Decision Lanes To Research

### 7.1 Early Damage Control

Exit before the static stop when several weakness conditions appear together, for example:

- Price closes below MA20/MA50 after entry.
- Close repeatedly lands in the lower part of the daily range.
- Drawdown expands in ATR terms.
- Volume expands on down days.
- Market regime turns risk-off.

This lane must reduce average loser size without cutting too many normal pullbacks.

### 7.2 Dead-Capital Exit

Exit when the position fails to progress after a reasonable holding period, for example:

- `holding_bars` exceeds a setup-specific threshold.
- Unrealized return remains near zero or negative.
- ATR compresses and volume dries up.
- There is slot pressure from better-ranked signals.

This lane must be evaluated by freed-slot contribution, not only by the exited trade's PnL.

### 7.3 Profit Protection

Raise stop or exit when profit is at risk:

- Position has reached a meaningful unrealized gain.
- Momentum deteriorates.
- Close loses short-term moving average or breaks a recent low.
- Market weakens.

This lane should improve drawdown and profit giveback without capping all winners.

### 7.4 Runner Extension

Hold beyond static take-profit behavior when:

- Trend remains intact.
- Volume confirms continuation.
- Pullbacks are shallow.
- Market and sector conditions remain supportive.

This lane must preserve or increase top-decile winners.

## 8. Hard Guardrails

The ExitAgent must not:

- Use future bars or any information after `as_of_date`.
- Change entry selection, entry timing, or position sizing unless explicitly part of `REDUCE_NEXT_OPEN`.
- Trigger same-day exits from end-of-day signals.
- Convert a profitable baseline into a lower-return system just to improve win rate.
- Depend on LLM-only judgement that cannot be replayed deterministically in backtest.
- Override the existing market panic guard without an explicit and tested rule.

## 9. Evaluation Requirements

Run at least these comparisons:

1. Baseline static exit.
2. Adaptive ExitAgent full policy.
3. Ablation without early damage control.
4. Ablation without dead-capital exit.
5. Ablation without runner extension.
6. Ablation without market-state features.

Required reporting:

- Total return.
- CAGR if the backtest window is long enough.
- Sharpe ratio.
- Max drawdown.
- MAR or return/drawdown ratio.
- Win rate.
- Number of trades.
- Median holding bars for winners and losers.
- Average loser size.
- Average winner size.
- Top-decile winner preservation.
- Stop-loss hit rate.
- Action distribution by reason code.
- Performance by strategy, year, market regime, and setup type.

Promotion criteria:

- Validation performance must not reduce total return by more than 5% relative to baseline unless max drawdown improves by at least 25%.
- Sharpe or MAR should improve on the validation window.
- Average loser size should decline.
- Top-decile winners should not materially shrink.
- Action distribution must be explainable and not dominated by noisy exits.

## 10. Expected Deliverables From Fintech Lab

1. Research memo with proposed rules/features and rationale.
2. Python module implementing the ExitAgent interface.
3. Backtest integration patch or standalone harness compatible with existing trade simulation.
4. Full backtest report on YTD and 2022-2024 validation windows.
5. Ablation report showing which decision lanes add value.
6. Failure analysis with examples where the ExitAgent hurts performance.
7. Recommendation: reject, shadow-run, or promote to candidate production.

## 11. Suggested Implementation Shape

Start with a deterministic policy engine:

- `DamageControlPolicy`
- `DeadCapitalPolicy`
- `ProfitProtectionPolicy`
- `RunnerExtensionPolicy`
- `PolicyAggregator`

Each policy should emit a score, reason codes, and a proposed action. The aggregator resolves conflicts conservatively:

1. Existing hard risk stops remain the final safety net.
2. High-confidence damage control can exit early.
3. Profit protection can raise stop before it exits.
4. Runner extension can suppress static take-profit only when risk is already protected.
5. Dead-capital exit should require either low progress or explicit portfolio slot pressure.

Machine learning, if used, should be introduced only after a deterministic benchmark exists.

## 12. Open Research Questions

- Which weakness signals predict future stop-loss hits best in Vietnamese equities?
- Are dead-capital exits helpful after transaction cost and replacement trade quality?
- Should thresholds be global, strategy-specific, or regime-specific?
- Does runner extension improve only breakout setups, or also pullback setups?
- How should ExitAgent interact with market panic guard and sector rotation?
