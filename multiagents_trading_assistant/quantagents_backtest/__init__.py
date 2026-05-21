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
    build_oos_research_portfolio,
    run_vn_quantagents,
    run_oos_execution_portfolio,
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
from multiagents_trading_assistant.quantagents_backtest.local_data import (
    DEFAULT_INDEX_PATH,
    DEFAULT_OHLCV_PATH,
    LocalUniverseConfig,
    load_local_index,
    load_local_universe,
)
from multiagents_trading_assistant.quantagents_backtest.run_local import (
    run_local_quantagents,
    save_local_quantagents_result,
)
from multiagents_trading_assistant.quantagents_backtest.edge_research import (
    DEFAULT_COMBOS,
    EdgeResearchConfig,
    ResearchStrategy,
    load_research_strategies,
    run_edge_vn_quantagents,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Strategy",
    "Trade",
    "build_oos_research_portfolio",
    "RiskGateConfig",
    "CURRENT_VN30",
    "DEFAULT_INDEX_PATH",
    "DEFAULT_OHLCV_PATH",
    "DEFAULT_COMBOS",
    "EdgeResearchConfig",
    "ResearchStrategy",
    "UniverseSnapshot",
    "LocalUniverseConfig",
    "VNMarketCostConfig",
    "VNQuantAgentsConfig",
    "VNPortfolioConfig",
    "WalkForwardConfig",
    "backtest_ensemble",
    "backtest_strategy",
    "backtest_vn_portfolio",
    "generate_strategy_pool",
    "get_universe_snapshot",
    "load_local_index",
    "load_local_universe",
    "load_historical_constituents",
    "run_local_quantagents",
    "load_research_strategies",
    "run_edge_vn_quantagents",
    "run_oos_execution_portfolio",
    "run_walk_forward",
    "run_vn_quantagents",
    "save_local_quantagents_result",
    "save_vn_quantagents_result",
]
