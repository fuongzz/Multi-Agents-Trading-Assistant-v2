"""QuantAgents-inspired permutation backtesting toolkit."""

from multiagents_trading_assistant.quantagents_backtest.backtest_engine import (
    BacktestConfig,
    BacktestResult,
    Trade,
    backtest_strategy,
    backtest_ensemble,
)
from multiagents_trading_assistant.quantagents_backtest.strategy_generator import (
    Strategy,
    generate_strategy_pool,
)
from multiagents_trading_assistant.quantagents_backtest.walk_forward import (
    WalkForwardConfig,
    run_walk_forward,
)
from multiagents_trading_assistant.quantagents_backtest.vn_quantagents import (
    RiskGateConfig,
    VNQuantAgentsConfig,
    run_vn_quantagents,
    save_vn_quantagents_result,
)
from multiagents_trading_assistant.quantagents_backtest.vn_portfolio_engine import (
    VNMarketCostConfig,
    VNPortfolioConfig,
    backtest_vn_portfolio,
)
from multiagents_trading_assistant.quantagents_backtest.vn_universe import (
    CURRENT_VN30,
    UniverseSnapshot,
    get_universe_snapshot,
    load_historical_constituents,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Strategy",
    "Trade",
    "RiskGateConfig",
    "CURRENT_VN30",
    "UniverseSnapshot",
    "VNMarketCostConfig",
    "VNQuantAgentsConfig",
    "VNPortfolioConfig",
    "WalkForwardConfig",
    "backtest_ensemble",
    "backtest_strategy",
    "backtest_vn_portfolio",
    "generate_strategy_pool",
    "get_universe_snapshot",
    "load_historical_constituents",
    "run_walk_forward",
    "run_vn_quantagents",
    "save_vn_quantagents_result",
]
