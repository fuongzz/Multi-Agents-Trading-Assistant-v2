"""TradingAgents-VN: Multi-agent LLM trading analysis for Vietnam market.

Adapted from https://github.com/tauricresearch/tradingagents
Uses VnstockDataProvider (vnstock_data Golden) instead of yfinance/Alpha Vantage.
Completely separate from the existing production pipeline.
"""

from multiagents_trading_assistant.tradingagents_vn.graph import TradingAgentsVN

__all__ = ["TradingAgentsVN"]
