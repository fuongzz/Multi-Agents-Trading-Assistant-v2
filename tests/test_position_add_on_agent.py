from __future__ import annotations

import pandas as pd

from multiagents_trading_assistant.agentic.position_add_on_agent import (
    PositionAddOnState,
    PositionAddOnAgent,
)


def _position() -> PositionAddOnState:
    return PositionAddOnState(
        symbol="AAA",
        entry_date="2026-01-02",
        entry_price=100.0,
        stop_loss_price=94.0,
        take_profit_price=130.0,
        shares=1000,
        add_on_count=0,
        setup_type="EDGE_BREAKOUT",
        strategy_name="breakout_55_smt_v1",
        holding_bars=12,
    )


def _history(last_close: float = 116.0) -> pd.DataFrame:
    rows = []
    for idx in range(18):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=idx)
        high = 113.0 if idx < 17 else 116.5
        low = 100.0 if idx < 17 else 112.0
        close = 111.0 if idx < 17 else last_close
        rows.append(
            {
                "date": date,
                "open": 108.0,
                "high": high,
                "low": low,
                "close": close,
                "ma20": 106.0,
                "atr14": 3.0,
                "mkt_regime_score": 70,
                "rs_percentile_20": 0.78,
                "value_ratio_20": 1.35,
                "close_location": 0.82,
                "above_ma20": True,
                "above_ma50": True,
                "upper_wick_pct": 0.03,
            }
        )
    return pd.DataFrame(rows)


def test_add_on_agent_adds_only_after_confirmed_rebreakout():
    review = PositionAddOnAgent().review(_position(), _history(), as_of_date="2026-01-18")

    assert review.action == "ADD_ON_NEXT_OPEN"
    assert review.next_bar_actionable is True
    assert review.suggested_stop_loss is not None
    assert "RE_BREAKOUT_CLOSE" in review.reason_codes


def test_add_on_agent_ignores_future_rows_after_as_of_date():
    causal = _history(last_close=116.0)
    future = pd.concat(
        [
            causal,
            pd.DataFrame(
                [
                    {
                        "date": "2026-01-19",
                        "open": 90.0,
                        "high": 91.0,
                        "low": 80.0,
                        "close": 82.0,
                        "ma20": 105.0,
                        "atr14": 8.0,
                        "mkt_regime_score": 20,
                        "rs_percentile_20": 0.10,
                        "value_ratio_20": 0.50,
                        "close_location": 0.10,
                        "above_ma20": False,
                        "above_ma50": False,
                        "upper_wick_pct": 0.30,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    agent = PositionAddOnAgent()
    base_review = agent.review(_position(), causal, as_of_date="2026-01-18")
    future_review = agent.review(_position(), future, as_of_date="2026-01-18")

    assert future_review.action == base_review.action
    assert future_review.reason_codes == base_review.reason_codes
    assert future_review.suggested_stop_loss == base_review.suggested_stop_loss


def test_add_on_agent_holds_when_market_is_not_strong():
    history = _history()
    history.loc[history.index[-1], "mkt_regime_score"] = 40

    review = PositionAddOnAgent().review(_position(), history, as_of_date="2026-01-18")

    assert review.action == "HOLD"
    assert "MARKET_NOT_UPTREND" in review.reason_codes
