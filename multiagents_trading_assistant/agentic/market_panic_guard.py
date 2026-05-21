from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


HARD_STOP_REASONS = {
    "SL",
    "EDGE_STOP_LOSS",
    "EDGE_ATR_STOP",
    "EDGE_TRAILING_ATR_STOP",
    "AGENT_RAISED_STOP",
}


@dataclass(frozen=True)
class MarketPanicGuardConfig:
    min_symbols: int = 50
    median_gap_down_pct: float = -0.035
    gap_down_5_pct_fraction: float = 0.40
    low_down_6_5_pct_fraction: float = 0.50
    close_down_5_pct_fraction: float = 0.55
    index_close_down_pct: float = -0.025


@dataclass(frozen=True)
class MarketPanicState:
    date: pd.Timestamp
    is_panic: bool
    symbols: int
    median_gap_pct: float
    gap_down_5_fraction: float
    low_down_6_5_fraction: float
    close_down_5_fraction: float
    index_return_pct: float | None
    reason: str


class MarketPanicGuard:
    """Detect broad forced-selling tapes where hard stops should not be fire-sale orders.

    The guard uses same-day market breadth to model an execution/risk-control
    decision: during market-wide panic, hard stops are deferred instead of
    force-selling at the panic print. It does not use future bars.
    """

    def __init__(self, config: MarketPanicGuardConfig | None = None) -> None:
        self.config = config or MarketPanicGuardConfig()

    def state_for_date(
        self,
        *,
        data: dict[str, pd.DataFrame],
        index_data: pd.DataFrame,
        date: pd.Timestamp,
    ) -> MarketPanicState:
        date = pd.Timestamp(date).normalize()
        rows: list[dict[str, float]] = []
        for symbol, frame in data.items():
            if frame.empty:
                continue
            history = frame[frame["date"] <= date].tail(2)
            if len(history) < 2:
                continue
            prev = history.iloc[-2]
            row = history.iloc[-1]
            if pd.Timestamp(row["date"]).normalize() != date:
                continue
            prev_close = float(prev["close"])
            if prev_close <= 0:
                continue
            rows.append(
                {
                    "gap": float(row["open"]) / prev_close - 1.0,
                    "low_ret": float(row["low"]) / prev_close - 1.0,
                    "close_ret": float(row["close"]) / prev_close - 1.0,
                }
            )

        if len(rows) < self.config.min_symbols:
            return MarketPanicState(date, False, len(rows), 0.0, 0.0, 0.0, 0.0, None, "INSUFFICIENT_BREADTH")

        breadth = pd.DataFrame(rows)
        median_gap = float(breadth["gap"].median())
        gap_down_5 = float((breadth["gap"] <= -0.05).mean())
        low_down_6_5 = float((breadth["low_ret"] <= -0.065).mean())
        close_down_5 = float((breadth["close_ret"] <= -0.05).mean())
        index_ret = self._index_return(index_data, date)

        breadth_panic = (
            median_gap <= self.config.median_gap_down_pct
            and gap_down_5 >= self.config.gap_down_5_pct_fraction
            and low_down_6_5 >= self.config.low_down_6_5_pct_fraction
        )
        close_panic = close_down_5 >= self.config.close_down_5_pct_fraction
        index_panic = index_ret is not None and index_ret <= self.config.index_close_down_pct
        is_panic = breadth_panic and (close_panic or index_panic)
        reason = (
            f"median_gap={median_gap:.3f};gap_down_5={gap_down_5:.3f};"
            f"low_down_6_5={low_down_6_5:.3f};close_down_5={close_down_5:.3f};"
            f"index_ret={index_ret if index_ret is not None else 'NA'}"
        )
        return MarketPanicState(
            date=date,
            is_panic=is_panic,
            symbols=len(rows),
            median_gap_pct=median_gap,
            gap_down_5_fraction=gap_down_5,
            low_down_6_5_fraction=low_down_6_5,
            close_down_5_fraction=close_down_5,
            index_return_pct=index_ret,
            reason=reason,
        )

    @staticmethod
    def _index_return(index_data: pd.DataFrame, date: pd.Timestamp) -> float | None:
        if index_data.empty:
            return None
        history = index_data[index_data["date"] <= date].tail(2)
        if len(history) < 2:
            return None
        row = history.iloc[-1]
        if pd.Timestamp(row["date"]).normalize() != date:
            return None
        prev_close = float(history.iloc[-2]["close"])
        if prev_close <= 0:
            return None
        return float(row["close"]) / prev_close - 1.0

    @staticmethod
    def should_defer_exit(exit_reason: str | None, state: MarketPanicState | None) -> bool:
        return bool(state and state.is_panic and exit_reason in HARD_STOP_REASONS)
