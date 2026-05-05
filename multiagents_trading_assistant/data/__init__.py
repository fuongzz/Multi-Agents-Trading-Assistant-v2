"""Internal data access layer.

Application code should import provider-facing helpers from repository.py,
not vendor SDKs directly. The active implementation currently uses
vnstock_data Golden, but the interface is intentionally vendor-neutral.
"""

from multiagents_trading_assistant.data.repository import get_data_provider

__all__ = ["get_data_provider"]
