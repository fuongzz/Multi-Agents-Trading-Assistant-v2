import pandas as pd
import pytest

from multiagents_trading_assistant.research.money_cycle.backtest_strategy import (
    backtest_stock_selection,
    backtest_washout_reversal,
)


def _ohlcv(symbols=("AAA", "BBB"), periods=8):
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    rows = []
    for i, date in enumerate(dates):
        for offset, symbol in enumerate(symbols):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": 100.0 + i + 10 * offset,
                }
            )
    return pd.DataFrame(rows)


def test_washout_reversal_enters_next_trading_day():
    ohlcv = _ohlcv(symbols=("AAA",), periods=5)
    dates = sorted(ohlcv["date"].unique())
    market = pd.DataFrame({"date": dates, "CHDM03": [50.0, 10.0, 50.0, 50.0, 50.0]})

    result = backtest_washout_reversal(market, ohlcv, hold_days=2)

    assert len(result.trades) == 1
    assert result.trades[0].entry_date == pd.Timestamp(dates[2])


def test_stock_selection_requires_ds_improvement():
    ohlcv = _ohlcv(periods=6)
    dates = sorted(ohlcv["date"].unique())
    chdm = pd.DataFrame(
        [
            {"date": date, "symbol": symbol, "CHDM50": 20.0}
            for date in dates
            for symbol in ["AAA", "BBB"]
        ]
    )
    ds = pd.DataFrame(
        [{"date": dates[0], "symbol": "AAA", "DS50": 1.0}]
        + [{"date": date, "symbol": "AAA", "DS50": 0.0} for date in dates[1:]]
        + [{"date": date, "symbol": "BBB", "DS50": 1.0} for date in dates]
    )

    result = backtest_stock_selection(chdm, ds, ohlcv, hold_days=2, top_n_stocks=5)

    assert {trade.symbol for trade in result.trades} == {"AAA"}


def test_stock_selection_rebalance_days_is_explicit():
    ohlcv = _ohlcv(symbols=("AAA",), periods=6)
    dates = sorted(ohlcv["date"].unique())
    chdm = pd.DataFrame(
        [{"date": date, "symbol": "AAA", "CHDM50": 20.0} for date in dates]
    )
    ds = pd.DataFrame(
        [{"date": date, "symbol": "AAA", "DS50": 0.0} for date in dates]
    )

    scan_all = backtest_stock_selection(chdm, ds, ohlcv, hold_days=1, top_n_stocks=1)
    rebalance = backtest_stock_selection(
        chdm,
        ds,
        ohlcv,
        hold_days=1,
        top_n_stocks=1,
        rebalance_days=5,
    )

    assert len(scan_all.trades) > len(rebalance.trades)


def test_stock_selection_rejects_missing_ds_column():
    ohlcv = _ohlcv(symbols=("AAA",), periods=3)
    dates = sorted(ohlcv["date"].unique())
    chdm = pd.DataFrame(
        [{"date": date, "symbol": "AAA", "CHDM50": 20.0} for date in dates]
    )
    ds = pd.DataFrame(
        [{"date": date, "symbol": "AAA", "DS20": 0.0} for date in dates]
    )

    with pytest.raises(ValueError, match="DS50"):
        backtest_stock_selection(chdm, ds, ohlcv)
