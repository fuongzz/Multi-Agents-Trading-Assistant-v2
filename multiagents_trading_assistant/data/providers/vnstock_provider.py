"""vnstock_data Golden implementation of the internal data provider."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import pandas as pd


def _safe_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_change(curr: float | None, prev: float | None) -> float | None:
    if curr is None or prev in (None, 0):
        return None
    return round((curr - prev) / abs(prev) * 100, 1)


@dataclass
class VnstockDataProvider:
    """Data provider backed by vnstock_data Unified UI."""

    name: str = "vnstock_data"

    @cached_property
    def _market(self):
        from vnstock_data import Market

        return Market()

    @cached_property
    def _reference(self):
        from vnstock_data import Reference

        return Reference()

    @cached_property
    def _fundamental(self):
        from vnstock_data import Fundamental

        return Fundamental()

    @cached_property
    def _listing(self):
        from vnstock_data import Listing

        return Listing(source="vci")

    def get_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1D",
    ) -> pd.DataFrame:
        df = self._market.equity(symbol.upper()).ohlcv(
            start=start,
            end=end,
            interval=interval,
        )
        return self._normalize_ohlcv(df)

    def get_price_board(self, symbols: list[str]) -> pd.DataFrame:
        # price_board API columns: symbol, exchange, reference_price, ceiling_price,
        # floor_price, open_price, high_price, low_price, close_price,
        # bid/ask price+vol x3, foreign_buy_volume, foreign_sell_volume
        # NOTE: listed_share, current_room, total_room are NOT in this endpoint.
        # foreign_ownership_pct comes from summary() per symbol.
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            df = self._market.equity(symbol.upper()).price_board()
            if df is None or df.empty:
                continue
            # Enrich with foreign_ownership_pct from summary
            try:
                summ = self._market.equity(symbol.upper()).summary()
                if summ is not None and not summ.empty:
                    df = df.copy()
                    df["foreign_ownership_pct"] = _safe_float(
                        summ["foreign_ownership_pct"].iloc[0]
                    )
            except Exception:
                df = df.copy()
                df["foreign_ownership_pct"] = None
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)
        out = pd.DataFrame()
        out["symbol"] = raw.get("symbol", pd.Series(dtype=str))
        out["price"] = raw.get("close_price", pd.Series(dtype=float))
        out["foreign_buy_vol"] = raw.get("foreign_buy_volume", pd.Series(dtype=float))
        out["foreign_sell_vol"] = raw.get("foreign_sell_volume", pd.Series(dtype=float))
        # foreign_ownership_pct: 0-100, e.g. 20.14 means foreigners own 20.14%
        out["foreign_ownership_pct"] = raw.get(
            "foreign_ownership_pct", pd.Series(dtype=float)
        )
        return out

    def get_symbols_by_group(self, group: str) -> list[str]:
        series = self._reference.equity.by_group(group.upper())
        if hasattr(series, "tolist"):
            return [str(s).upper() for s in series.tolist()]
        return [str(s).upper() for s in series]

    def get_symbols_by_exchange(self, exchange: str = "HOSE") -> list[str]:
        df = self._reference.equity.by_exchange()
        if df is None or df.empty:
            return []
        exchange = exchange.upper()
        if "exchange" in df.columns:
            df = df[df["exchange"].astype(str).str.upper() == exchange]
        if "type" in df.columns:
            df = df[df["type"].astype(str).str.upper() == "STOCK"]
        symbols = df["symbol"].dropna().astype(str).str.upper().unique()
        return sorted(s for s in symbols if len(s) == 3 and s.isalpha())

    def get_fundamentals(self, symbol: str) -> dict:
        symbol = symbol.upper()
        result = {
            "pe": None,
            "pb": None,
            "roe": None,
            "eps": None,
            "revenue_growth": None,
            "profit_growth": None,
            "industry": None,
        }

        equity = self._fundamental.equity(symbol)
        ratio = equity.ratio(limit=8)
        if ratio is not None and not ratio.empty:
            latest = ratio.iloc[0]
            result["pe"] = _safe_float(latest.get("pe"))
            result["pb"] = _safe_float(latest.get("pb"))
            result["eps"] = _safe_float(latest.get("trailing_eps"))

        income = equity.income_statement(limit=8)
        if income is not None and not income.empty:
            income = income.copy()
            profit_col = self._first_existing_col(
                income,
                [
                    "net_profit_after_tax",
                    "profit_after_tax_for_shareholders_of_parent_company",
                    "profit_after_tax",
                ],
            )
            revenue_col = self._first_existing_col(
                income,
                [
                    "net_revenue",
                    "revenue",
                    "net_interest_income",
                    "operating_income",
                ],
            )
            result["profit_growth"] = self._trailing_growth(income, profit_col)
            result["revenue_growth"] = self._trailing_growth(income, revenue_col)

        try:
            health = equity.financial_health(limit=4)
            if health is not None and not health.empty:
                latest_health = health.iloc[-1]
                profit = _safe_float(latest_health.get("net_profit_after_tax"))
                total_assets = _safe_float_from_row(latest_health, "total_assets")
                liabilities = _safe_float_from_row(latest_health, "liabilities")
                if profit is not None and total_assets is not None:
                    if liabilities is not None:
                        owners_equity = total_assets - liabilities
                    else:
                        owners_equity = _safe_float_from_row(
                            latest_health,
                            "owners_equity",
                        )
                    if owners_equity and owners_equity > 0:
                        result["roe"] = round(profit * 4 / owners_equity * 100, 1)
        except Exception:
            pass

        if result["roe"] is None:
            try:
                balance = equity.balance_sheet(limit=1)
                income_one = equity.income_statement(limit=1)
                if (
                    balance is not None
                    and not balance.empty
                    and income_one is not None
                    and not income_one.empty
                ):
                    bal = balance.iloc[0]
                    inc = income_one.iloc[0]
                    profit = _safe_float_from_row(inc, "net_profit_after_tax")
                    total_assets = _safe_float_from_row(bal, "total_assets")
                    liabilities = _safe_float_from_row(bal, "liabilities")
                    owners_equity = _safe_float_from_row(bal, "owners_equity")
                    if owners_equity is None and None not in (total_assets, liabilities):
                        owners_equity = total_assets - liabilities
                    if profit is not None and owners_equity and owners_equity > 0:
                        result["roe"] = round(profit * 4 / owners_equity * 100, 1)
            except Exception:
                pass

        result["industry"] = self._lookup_industry(symbol)
        return result

    def get_foreign_flow(self, symbol: str, n_days: int = 20) -> dict:
        df = self._market.equity(symbol.upper()).foreign_flow()
        result = {
            "room_usage_pct": None,
            "net_flow_5d": None,
            "net_flow_20d": None,
            "flow_history": [],
        }
        if df is None or df.empty:
            return result
        df = df.copy().head(max(n_days, 20))
        net_col = "net_vol" if "net_vol" in df.columns else "net_val"
        net = pd.to_numeric(df[net_col], errors="coerce").dropna()
        result["net_flow_5d"] = float(net.head(5).sum()) if not net.empty else None
        result["net_flow_20d"] = (
            float(net.head(n_days).sum()) if not net.empty else None
        )
        date_col = "trading_date" if "trading_date" in df.columns else df.columns[0]
        result["flow_history"] = [
            {"date": str(row[date_col]), "net_flow": float(row[net_col])}
            for _, row in df[[date_col, net_col]].head(n_days).iterrows()
            if pd.notna(row[net_col])
        ]
        return result

    def get_corporate_events(self, symbol: str) -> pd.DataFrame:
        # Company requires symbol in constructor; source must be 'VCI' or 'KBS'.
        # VCI.events() returns dividends, AGM, issuance AND major-shareholder
        # (insider) trading in one frame — both Catalyst and Insider agents read it.
        from vnstock_data import Company

        df = Company(symbol=symbol.upper(), source="VCI").events()
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        out = pd.DataFrame()
        out["ticker"] = df.get("ticker", pd.Series([symbol.upper()] * len(df)))
        out["category"] = df.get("category", pd.Series(dtype=str))
        out["event_name_vi"] = df.get("event_name_vi", pd.Series(dtype=str))
        out["action_type_vi"] = df.get("action_type_vi", pd.Series(dtype=str))
        out["title"] = df.get("event_title_vi", pd.Series(dtype=str))
        out["exercise_ratio"] = df.get("exercise_ratio", pd.Series(dtype=float))
        out["value_per_share"] = df.get("value_per_share", pd.Series(dtype=float))
        for col in ("public_date", "record_date", "exright_date", "issue_date", "payout_date"):
            out[col] = pd.to_datetime(df.get(col), errors="coerce").dt.tz_localize(None).dt.normalize()
        return out.reset_index(drop=True)

    def _normalize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={"time": "date", "ticker": "symbol"}).copy()
        needed = ["date", "open", "high", "low", "close", "volume"]
        cols = [col for col in needed if col in df.columns]
        df = df[cols]
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"], errors="coerce")
            df["date"] = dates.dt.tz_localize(None).dt.normalize()
            df = df.dropna(subset=["date"])
            df = df.drop_duplicates(subset=["date"], keep="last")
        return df.sort_values("date").reset_index(drop=True)

    def _lookup_industry(self, symbol: str) -> str | None:
        try:
            df = self._reference.equity.list_by_industry(lang="vi")
            if df is None or df.empty:
                return None
            rows = df[
                (df["symbol"].astype(str).str.upper() == symbol)
                & (pd.to_numeric(df["icb_level"], errors="coerce") == 2)
            ]
            if rows.empty:
                rows = df[df["symbol"].astype(str).str.upper() == symbol]
            if rows.empty:
                return None
            return str(rows.iloc[0].get("icb_name") or "") or None
        except Exception:
            return None

    def _trailing_growth(self, df: pd.DataFrame, col: str | None) -> float | None:
        if not col or col not in df.columns or df.empty:
            return None
        series = pd.to_numeric(df[col], errors="coerce")
        if len(series) >= 5 and pd.notna(series.iloc[4]):
            return _pct_change(_safe_float(series.iloc[0]), _safe_float(series.iloc[4]))
        if len(series) >= 2 and pd.notna(series.iloc[1]):
            return _pct_change(_safe_float(series.iloc[0]), _safe_float(series.iloc[1]))
        return None

    def _first_existing_col(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        return None


def _safe_float_from_row(row: pd.Series, key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, pd.Series):
        value = value.dropna()
        if value.empty:
            return None
        value = value.iloc[0]
    return _safe_float(value)
