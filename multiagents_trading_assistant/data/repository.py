"""Provider factory for the application's data layer."""

from __future__ import annotations

import os
from functools import lru_cache

from multiagents_trading_assistant.data.providers.base import DataProvider
from multiagents_trading_assistant.data.providers.vnstock_provider import (
    VnstockDataProvider,
)


@lru_cache(maxsize=1)
def get_data_provider() -> DataProvider:
    """Return the configured data provider.

    The indirection keeps vendor SDKs out of agents and pipelines. When this
    product grows its own data library, add another provider here and switch
    MATA_DATA_PROVIDER without changing downstream code.
    """

    provider = os.getenv("MATA_DATA_PROVIDER", "vnstock").strip().lower()
    if provider in {"vnstock", "vnstock_data", "golden"}:
        return VnstockDataProvider()
    raise ValueError(f"Unsupported MATA_DATA_PROVIDER: {provider}")
