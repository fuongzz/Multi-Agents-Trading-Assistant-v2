# Modern Portfolio Allocation Design

## Goal

The current edge-lab portfolio first ranks signals and then gives each selected
candidate the same budget inside the next entry batch. That is a strong
baseline, but it ignores volatility and correlation between candidates.

This design keeps the existing signal pipeline intact and adds a second stage:

```text
hypothesis signals -> top candidates -> allocation optimizer -> order budgets
```

## Allocation Scope

The optimizer runs only on the candidate batch scheduled for the next entry
date. It does not change exits, signal filters, market regime exposure, or
position slot limits.

Historical returns are estimated from close-to-close data strictly before the
entry date, so the allocator can be used in walk-forward backtests without
entry-day leakage.

## Supported Methods

- `equal_weight`: current baseline behavior.
- `rank_weight`: weights by `_edge_rank`.
- `inverse_volatility`: gives lower weight to high-volatility candidates.
- `minimum_variance`: long-only approximation using a shrunk covariance matrix.
- `max_sharpe`: long-only approximation using historical mean returns and a
  shrunk covariance matrix.

All optimized methods can blend back toward signal rank with
`allocation_rank_blend`.

## Risk Controls

Weights are constrained per candidate:

- `allocation_min_weight`
- `allocation_max_weight`

The existing portfolio constraints still apply:

- `max_positions`
- `top_n`
- market-regime target exposure
- cash availability
- liquidity participation cap
- lot size
- stop-loss and exit rules

## Configuration Example

```python
from multiagents_trading_assistant.edge_lab.backtest import PortfolioConfig

cfg = PortfolioConfig(
    max_positions=8,
    top_n=5,
    allocation_method="minimum_variance",
    allocation_lookback_bars=60,
    allocation_min_weight=0.05,
    allocation_max_weight=0.35,
    allocation_rank_blend=0.25,
    allocation_shrinkage=0.20,
)
```

## Evaluation Plan

Compare each allocator against `equal_weight` over the same hypothesis set:

- total return
- Sharpe ratio
- max drawdown
- win rate and profit factor
- average gross exposure
- turnover and number of trades

Promotion rule: an allocator should improve risk-adjusted return without
raising max drawdown or depending on a single market regime.
