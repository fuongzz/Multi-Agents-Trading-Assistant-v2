"""LLM service — re-export base runners from agent.py.

Hai function:
  run_agent()      → Sonnet (quyết định cuối / debate)
  run_agent_lite() → Haiku  (analyst nhẹ)

Giữ nguyên prompt caching của agent.py để tiết kiệm ~90% input token.
"""

from multiagents_trading_assistant.agent import (
    run_agent,
    run_agent_lite,
    MODEL_SONNET,
    MODEL_HAIKU,
)

__all__ = ["run_agent", "run_agent_lite", "MODEL_SONNET", "MODEL_HAIKU"]
