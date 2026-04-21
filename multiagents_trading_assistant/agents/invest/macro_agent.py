"""macro_agent.py — Vĩ mô cho Investment pipeline.

Port nguyên từ _legacy/agents/macro_agent.py — chỉ đổi import path.
Cache 1 lần/ngày, shared giữa Investment và Trade pipeline.
"""

from multiagents_trading_assistant._legacy.agents.macro_agent import (
    get_macro_context,
    _crawl_macro_headlines,
    _neutral_fallback,
    CACHE_DIR,
)

__all__ = ["get_macro_context", "CACHE_DIR"]
