from __future__ import annotations

import pandas as pd

from multiagents_trading_assistant.backtest.live_pipeline import LivePosition
from scripts.backtest_position_exit_agent import _apply_pending_reviews, _exit_decision_with_agent_stop


def test_pending_take_profit_review_sells_at_next_open_not_review_day():
    date = pd.Timestamp("2026-01-11")
    data = {
        "AAA": pd.DataFrame(
            [
                {"date": date, "open": 121.0, "high": 125.0, "low": 120.0, "close": 124.0, "volume": 1000}
            ]
        )
    }
    position = LivePosition(
        symbol="AAA",
        setup_type="EDGE",
        signal_date=pd.Timestamp("2026-01-01"),
        entry_date=pd.Timestamp("2026-01-02"),
        entry_idx=0,
        entry_price=100.0,
        stop_loss=92.0,
        take_profit=120.0,
        shares=100,
        priority_score=1.0,
        reasons=[],
        edge_risk={"stop_loss": 0.08, "take_profit": 0.2},
    )
    positions = {"AAA": position}
    pending = {
        date: {
            "AAA": {
                "action": "TAKE_PROFIT_NEXT_OPEN",
                "as_of_date": "2026-01-10",
                "reason_codes": ["WEAK_CLOSE"],
            }
        }
    }

    from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig

    cfg = LivePipelineBacktestConfig(settlement_bars=2)
    trades = []
    cash = _apply_pending_reviews(
        date=date,
        day_idx=4,
        data=data,
        pending_reviews=pending,
        positions=positions,
        cash=0.0,
        trades=trades,
        cfg=cfg,
        counters={"agent_tp_next_open": 0, "agent_raise_stop": 0, "agent_hold_runner": 0},
    )

    assert cash > 0
    assert "AAA" not in positions
    assert trades[0]["exit_date"] == "2026-01-11"
    assert trades[0]["exit_reason"] == "AGENT_TAKE_PROFIT_NEXT_OPEN"
    assert trades[0]["agent_review_as_of"] == "2026-01-10"


def test_pending_raise_stop_only_arms_next_bar_without_same_day_exit():
    date = pd.Timestamp("2026-01-11")
    data = {
        "AAA": pd.DataFrame(
            [
                {"date": date, "open": 119.0, "high": 122.0, "low": 118.0, "close": 121.0, "volume": 1000}
            ]
        )
    }
    position = LivePosition(
        symbol="AAA",
        setup_type="EDGE",
        signal_date=pd.Timestamp("2026-01-01"),
        entry_date=pd.Timestamp("2026-01-02"),
        entry_idx=0,
        entry_price=100.0,
        stop_loss=92.0,
        take_profit=125.0,
        shares=100,
        priority_score=1.0,
        reasons=[],
        edge_risk={"stop_loss": 0.08, "take_profit": 0.25},
    )
    positions = {"AAA": position}
    pending = {
        date: {
            "AAA": {
                "action": "RAISE_TRAILING_STOP",
                "as_of_date": "2026-01-10",
                "reason_codes": ["PROTECT_PROFIT"],
                "suggested_stop_loss": 115.0,
            }
        }
    }

    from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig

    cfg = LivePipelineBacktestConfig(settlement_bars=2)
    trades = []
    cash = _apply_pending_reviews(
        date=date,
        day_idx=4,
        data=data,
        pending_reviews=pending,
        positions=positions,
        cash=0.0,
        trades=trades,
        cfg=cfg,
        counters={"agent_tp_next_open": 0, "agent_raise_stop": 0, "agent_hold_runner": 0},
    )

    assert cash == 0.0
    assert trades == []
    assert positions["AAA"].stop_loss == 115.0
    assert getattr(positions["AAA"], "agent_stop_active") is True


def test_agent_raised_stop_uses_next_bar_trigger_or_gap_price():
    from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig

    position = LivePosition(
        symbol="AAA",
        setup_type="EDGE",
        signal_date=pd.Timestamp("2026-01-01"),
        entry_date=pd.Timestamp("2026-01-02"),
        entry_idx=0,
        entry_price=100.0,
        stop_loss=115.0,
        take_profit=125.0,
        shares=100,
        priority_score=1.0,
        reasons=[],
        edge_risk={"stop_loss": 0.08, "take_profit": 0.25},
    )
    setattr(position, "agent_stop_active", True)

    row = pd.Series({"open": 112.0, "high": 118.0, "low": 110.0, "close": 117.0})
    reason, exit_price = _exit_decision_with_agent_stop(row, position, LivePipelineBacktestConfig(slippage_rate=0.001))

    assert reason == "AGENT_RAISED_STOP"
    assert exit_price == 111.888
