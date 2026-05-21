import pandas as pd

from multiagents_trading_assistant.backtest.live_pipeline import (
    LivePipelineBacktestConfig,
    LivePosition,
    _edge_exit_decision,
)


def _position(**risk_overrides) -> LivePosition:
    risk = {
        "stop_loss": 0.08,
        "take_profit": 0.20,
        "initial_atr_stop_mult": 2.0,
        "trailing_atr_mult": 2.0,
        "trailing_profit_activation": 0.10,
        "max_holding_bars": 30,
    }
    risk.update(risk_overrides)
    return LivePosition(
        symbol="AAA",
        setup_type="EDGE",
        signal_date=pd.Timestamp("2024-01-01"),
        entry_date=pd.Timestamp("2024-01-02"),
        entry_idx=1,
        entry_price=100.0,
        stop_loss=92.0,
        take_profit=120.0,
        shares=100,
        priority_score=80.0,
        reasons=["unit"],
        edge_strategy_name="unit",
        edge_strategy_passed=True,
        edge_score=80.0,
        edge_rank=1.0,
        edge_rank_score=1.0,
        edge_risk=risk,
        entry_atr=5.0,
        highest_price=100.0,
        holding_bars=3,
    )


def test_edge_stop_loss_uses_trigger_or_gap_not_same_bar_open():
    cfg = LivePipelineBacktestConfig(slippage_rate=0.001)
    row = pd.Series({"open": 98.0, "high": 101.0, "low": 90.0, "close": 95.0})

    reason, exit_price = _edge_exit_decision(row, _position(), cfg)

    assert reason == "EDGE_STOP_LOSS"
    assert exit_price == 91.908


def test_edge_take_profit_uses_trigger_or_gap_up_not_same_bar_open():
    cfg = LivePipelineBacktestConfig(slippage_rate=0.001)
    row = pd.Series({"open": 105.0, "high": 125.0, "low": 104.0, "close": 121.0})
    position = _position(
        initial_atr_stop_mult=None,
        trailing_atr_mult=None,
        trailing_profit_activation=None,
    )

    reason, exit_price = _edge_exit_decision(row, position, cfg)

    assert reason == "EDGE_TAKE_PROFIT"
    assert exit_price == 119.88


def test_edge_trailing_atr_stop_uses_trailing_trigger_not_same_bar_open():
    cfg = LivePipelineBacktestConfig(slippage_rate=0.001)
    row = pd.Series({"open": 112.0, "high": 114.0, "low": 104.0, "close": 109.0})
    position = _position(stop_loss=None, take_profit=None)
    position.highest_price = 115.0

    reason, exit_price = _edge_exit_decision(row, position, cfg)

    assert reason == "EDGE_TRAILING_ATR_STOP"
    assert exit_price == 104.895
