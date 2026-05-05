import numpy as np
import pandas as pd

from multiagents_trading_assistant.backtest import live_pipeline
from multiagents_trading_assistant.backtest.live_pipeline import (
    LivePipelineBacktestConfig,
    run_live_pipeline_backtest,
)


def _sample_ohlcv(n=140, offset=0.0):
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(20 + offset, 30 + offset, n) + np.sin(np.arange(n) / 5.0)
    open_ = close * 1.001
    high = np.maximum(open_, close) * 1.02
    low = np.minimum(open_, close) * 0.98
    volume = 1_000_000 + (np.arange(n) % 10) * 50_000
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_live_pipeline_backtest_replays_candidates_and_portfolio(monkeypatch):
    def always_pass(_df, _ind):
        return True, ["unit detector"]

    monkeypatch.setattr(live_pipeline, "LIVE_STRATEGIES", [("UNIT_SETUP", always_pass)])
    monkeypatch.setattr(
        live_pipeline,
        "_money_flow_at",
        lambda *_args, **_kwargs: {"regime": "MONEY_IN", "score": 3},
    )
    monkeypatch.setattr(
        live_pipeline,
        "compute_priority_score",
        lambda *_args, **_kwargs: 80.0,
    )

    result = run_live_pipeline_backtest(
        {"AAA": _sample_ohlcv(), "BBB": _sample_ohlcv(offset=2.0)},
        _sample_ohlcv(),
        config=LivePipelineBacktestConfig(
            start_date="2024-05-01",
            end_date="2024-06-28",
            lookback=80,
            max_positions=2,
            max_candidates_per_day=2,
            max_hold_bars=5,
            settlement_bars=1,
            lot_size=10,
        ),
    )

    assert not result["equity_curve"].empty
    assert result["metrics"]["number_of_trades"] > 0
    assert result["equity_frame"]["positions"].max() <= 2
    assert {trade["setup_type"] for trade in result["trades"]} == {"UNIT_SETUP"}
