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

    def get_corporate_events(self, symbol: str) -> pd.DataFrame:
        """Return normalized corporate events (dividends, AGM, issuance, insider).

        Columns: ticker, category, event_name_vi, action_type_vi, title,
        public_date, record_date, exright_date, issue_date, payout_date,
        exercise_ratio, value_per_share. Date columns are tz-naive Timestamps.
        category values include DIVIDEND, SHAREHOLDER_MEETING,
        MAJOR_SHAREHOLDER_TRADING, OTHER.
        """
