"""Provider contracts for market and fundamental data."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class DataProvider(Protocol):
    """Vendor-neutral data provider contract used by the app."""

    name: str

    def get_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1D",
    ) -> pd.DataFrame:
        """Return OHLCV with columns: date, open, high, low, close, volume."""

    def get_price_board(self, symbols: list[str]) -> pd.DataFrame:
        """Return a normalized realtime/snapshot price board."""

    def get_symbols_by_group(self, group: str) -> list[str]:
        """Return symbols for a market group such as VN30 or VN100."""

    def get_symbols_by_exchange(self, exchange: str = "HOSE") -> list[str]:
        """Return listed symbols for an exchange."""

    def get_fundamentals(self, symbol: str) -> dict:
        """Return normalized valuation and financial metric keys."""

    def get_foreign_flow(self, symbol: str, n_days: int = 20) -> dict:
        """Return normalized foreign flow summary and history."""
