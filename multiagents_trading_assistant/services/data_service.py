"""Data service — re-export fetcher functions.

Cả hai pipeline (invest + trade) dùng chung fetcher để không duplicate API call.
"""

from multiagents_trading_assistant.fetcher import (
    get_ohlcv,
    get_ohlcv_batch,
    get_vnindex,
    get_vn30_symbols,
    get_vn100_symbols,
    get_price_board,
    get_fundamentals,
    get_foreign_flow,
    get_global_macro,
    get_vn_macro,
)

__all__ = [
    "get_ohlcv",
    "get_ohlcv_batch",
    "get_vnindex",
    "get_vn30_symbols",
    "get_vn100_symbols",
    "get_price_board",
    "get_fundamentals",
    "get_foreign_flow",
    "get_global_macro",
    "get_vn_macro",
]
