from __future__ import annotations

import json

import numpy as np
import pandas as pd

from multiagents_trading_assistant.quantagents_backtest.local_data import (
    LocalUniverseConfig,
    load_local_index,
    load_local_universe,
)
from multiagents_trading_assistant.quantagents_backtest.run_local import (
    run_local_quantagents,
    save_local_quantagents_result,
)
from multiagents_trading_assistant.quantagents_backtest.vn_quantagents import VNQuantAgentsConfig


def _sample_symbol(symbol: str, start: str, periods: int, value: float = 5_000_000_000.0) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    close = np.linspace(20.0, 40.0, periods) + np.sin(np.arange(periods) / 8.0)
    open_ = close * (1 + 0.001)
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = np.full(periods, 500_000, dtype=np.int64)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "value": np.full(periods, value),
            "exchange": "HOSE",
            "industry": "Test",
        }
    )


def _sample_index(start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    close = np.linspace(1000.0, 1200.0, periods)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "VNINDEX",
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(periods, 100_000_000, dtype=np.int64),
        }
    )


def test_load_local_universe_applies_history_and_liquidity_filters(tmp_path):
    ohlcv = pd.concat(
        [
            _sample_symbol("AAA", "2024-01-01", 180),
            _sample_symbol("BBB", "2024-01-01", 120),
            _sample_symbol("CCC", "2024-01-01", 180, value=500_000_000.0),
        ],
        ignore_index=True,
    )
    ohlcv_path = tmp_path / "ohlcv.parquet"
    ohlcv.to_parquet(ohlcv_path, index=False)

    data, coverage, snapshot = load_local_universe(
        LocalUniverseConfig(
            start="2024-01-01",
            end="2024-12-31",
            universe_name="VN30",
            min_history_bars=150,
            min_median_daily_value=3_000_000_000.0,
        ),
        ohlcv_path=ohlcv_path,
        explicit_symbols=["AAA", "BBB", "CCC"],
    )

    assert snapshot.source == "explicit_symbols"
    assert sorted(data) == ["AAA"]
    reasons = dict(zip(coverage["symbol"], coverage["reason"]))
    assert reasons["BBB"] == "insufficient_history"
    assert reasons["CCC"] == "insufficient_liquidity"


def test_load_local_universe_builds_vn100_liquidity_proxy(tmp_path):
    ohlcv = pd.concat(
        [
            _sample_symbol("AAA", "2024-01-01", 180, value=9_000_000_000.0),
            _sample_symbol("BBB", "2024-01-01", 180, value=8_000_000_000.0),
            _sample_symbol("CCC", "2024-01-01", 180, value=500_000_000.0),
        ],
        ignore_index=True,
    )
    ohlcv_path = tmp_path / "ohlcv.parquet"
    ohlcv.to_parquet(ohlcv_path, index=False)

    data, coverage, snapshot = load_local_universe(
        LocalUniverseConfig(
            start="2024-01-01",
            end="2024-12-31",
            universe_name="VN100",
            min_history_bars=150,
            min_median_daily_value=3_000_000_000.0,
        ),
        ohlcv_path=ohlcv_path,
    )

    assert snapshot.source == "local_vn100_liquidity_proxy"
    assert not snapshot.is_historical
    assert sorted(data) == ["AAA", "BBB"]
    assert set(coverage["symbol"]) == {"AAA", "BBB"}


def test_run_local_quantagents_and_save_outputs(tmp_path):
    periods = 180
    ohlcv = pd.concat(
        [
            _sample_symbol("AAA", "2024-01-01", periods),
            _sample_symbol("BBB", "2024-01-01", periods, value=6_000_000_000.0),
            _sample_symbol("CCC", "2024-01-01", periods, value=7_000_000_000.0),
        ],
        ignore_index=True,
    )
    index_df = _sample_index("2024-01-01", periods)
    ohlcv_path = tmp_path / "ohlcv.parquet"
    index_path = tmp_path / "index.parquet"
    ohlcv.to_parquet(ohlcv_path, index=False)
    index_df.to_parquet(index_path, index=False)

    result = run_local_quantagents(
        universe_config=LocalUniverseConfig(
            start="2024-01-01",
            end="2024-12-31",
            min_history_bars=120,
            min_median_daily_value=1_000_000_000.0,
        ),
        qa_config=VNQuantAgentsConfig(
            start="2024-01-01",
            end="2024-12-31",
            n_strategies=50,
            top_k=5,
            train_bars=80,
            test_bars=40,
            step_bars=40,
        ),
        ohlcv_path=ohlcv_path,
        index_path=index_path,
        explicit_symbols=["AAA", "BBB", "CCC"],
    )

    assert result["metadata"]["mode"] == "research_test_only"
    assert result["metadata"]["universe_included"] == 3
    assert not result["strategy_memory"].empty
    assert not result["portfolio_summary"].empty
    assert not result["data_coverage"].empty

    out = save_local_quantagents_result(result, tmp_path / "out")
    assert (out / "metadata.json").exists()
    assert (out / "data_coverage.csv").exists()
    assert (out / "family_summary.csv").exists()

    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["universe_included"] == 3
    assert metadata["snapshot_source"] == "explicit_symbols"


def test_load_local_index_returns_datetime_index(tmp_path):
    index_df = _sample_index("2024-01-01", 30)
    index_path = tmp_path / "index.parquet"
    index_df.to_parquet(index_path, index=False)

    loaded = load_local_index(index_path=index_path, start="2024-01-10", end="2024-02-15")
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.min() >= pd.Timestamp("2024-01-10")
    assert loaded.index.max() <= pd.Timestamp("2024-02-15")
