from __future__ import annotations

import pandas as pd

from multiagents_trading_assistant.agentic.market_panic_guard import MarketPanicGuard, MarketPanicGuardConfig


def _frame(symbol: str, prev_close: float, open_: float, low: float, close: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-03-06"), "symbol": symbol, "open": prev_close, "high": prev_close, "low": prev_close, "close": prev_close, "volume": 1_000_000},
            {"date": pd.Timestamp("2026-03-09"), "symbol": symbol, "open": open_, "high": open_, "low": low, "close": close, "volume": 2_000_000},
        ]
    )


def test_market_panic_guard_detects_broad_forced_selling() -> None:
    data = {
        f"S{i:03d}": _frame(f"S{i:03d}", 100.0, 94.0, 93.0, 94.0)
        for i in range(80)
    }
    index = _frame("VNINDEX", 1000.0, 960.0, 950.0, 960.0)
    guard = MarketPanicGuard(MarketPanicGuardConfig(min_symbols=50))

    state = guard.state_for_date(data=data, index_data=index, date=pd.Timestamp("2026-03-09"))

    assert state.is_panic
    assert guard.should_defer_exit("EDGE_STOP_LOSS", state)
    assert not guard.should_defer_exit("EDGE_TAKE_PROFIT", state)


def test_market_panic_guard_ignores_single_stock_stop() -> None:
    data = {
        f"S{i:03d}": _frame(f"S{i:03d}", 100.0, 100.0, 99.0, 100.0)
        for i in range(80)
    }
    data["S001"] = _frame("S001", 100.0, 92.0, 91.0, 93.0)
    index = _frame("VNINDEX", 1000.0, 1000.0, 995.0, 1000.0)
    guard = MarketPanicGuard(MarketPanicGuardConfig(min_symbols=50))

    state = guard.state_for_date(data=data, index_data=index, date=pd.Timestamp("2026-03-09"))

    assert not state.is_panic
    assert not guard.should_defer_exit("EDGE_STOP_LOSS", state)
