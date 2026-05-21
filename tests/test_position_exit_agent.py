from __future__ import annotations

import pandas as pd

from multiagents_trading_assistant.agentic.position_exit_agent import (
    PositionState,
    PositionStateExitAgent,
)


def _position() -> PositionState:
    return PositionState(
        symbol="VCB",
        entry_date="2026-01-02",
        entry_price=100.0,
        stop_loss_price=92.0,
        take_profit_price=120.0,
        shares=100,
        setup_type="EDGE_BREAKOUT",
        holding_bars=10,
    )


def test_exit_agent_holds_runner_when_near_tp_and_state_strong():
    history = pd.DataFrame(
        [
            {
                "date": "2026-01-10",
                "close": 119.0,
                "atr14": 3.0,
                "mkt_regime_score": 70,
                "rs_percentile_20": 0.75,
                "value_ratio_20": 1.4,
                "close_location": 0.85,
                "above_ma20": True,
                "above_ma50": True,
                "upper_wick_pct": 0.02,
                "distribution_days_10": 0,
            }
        ]
    )

    review = PositionStateExitAgent().review(_position(), history, as_of_date="2026-01-10")

    assert review.action == "HOLD_RUNNER"
    assert review.next_bar_actionable is True
    assert review.suggested_stop_loss is not None
    assert review.suggested_stop_loss >= 111.2


def test_exit_agent_takes_profit_next_open_when_near_tp_and_exhausted():
    history = pd.DataFrame(
        [
            {
                "date": "2026-01-10",
                "close": 119.0,
                "atr14": 3.0,
                "mkt_regime_score": 56,
                "rs_percentile_20": 0.50,
                "value_ratio_20": 0.9,
                "close_location": 0.30,
                "above_ma20": False,
                "above_ma50": True,
                "upper_wick_pct": 0.12,
                "distribution_days_10": 3,
            }
        ]
    )

    review = PositionStateExitAgent().review(_position(), history, as_of_date="2026-01-10")

    assert review.action == "TAKE_PROFIT_NEXT_OPEN"
    assert "WEAK_CLOSE" in review.reason_codes
    assert "SUPPLY_WICK" in review.reason_codes


def test_exit_agent_ignores_future_rows_after_as_of_date():
    causal = pd.DataFrame(
        [
            {
                "date": "2026-01-10",
                "close": 119.0,
                "atr14": 3.0,
                "mkt_regime_score": 70,
                "rs_percentile_20": 0.75,
                "value_ratio_20": 1.4,
                "close_location": 0.85,
                "above_ma20": True,
                "above_ma50": True,
                "upper_wick_pct": 0.02,
                "distribution_days_10": 0,
            }
        ]
    )
    with_future_bearish = pd.concat(
        [
            causal,
            pd.DataFrame(
                [
                    {
                        "date": "2026-01-11",
                        "close": 90.0,
                        "atr14": 8.0,
                        "mkt_regime_score": 10,
                        "rs_percentile_20": 0.10,
                        "value_ratio_20": 0.5,
                        "close_location": 0.10,
                        "above_ma20": False,
                        "above_ma50": False,
                        "upper_wick_pct": 0.20,
                        "distribution_days_10": 5,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    agent = PositionStateExitAgent()
    base_review = agent.review(_position(), causal, as_of_date="2026-01-10")
    future_review = agent.review(_position(), with_future_bearish, as_of_date="2026-01-10")

    assert future_review.action == base_review.action
    assert future_review.suggested_stop_loss == base_review.suggested_stop_loss
    assert future_review.reason_codes == base_review.reason_codes
