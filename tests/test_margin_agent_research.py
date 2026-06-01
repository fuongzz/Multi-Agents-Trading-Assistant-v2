from __future__ import annotations

import pandas as pd

from scripts.run_margin_agent_research import MarginPolicy, run_margin_overlay


def _curve(equities: list[float], cash: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-02", periods=len(equities), freq="B"),
            "equity": equities,
            "cash": [cash] * len(equities),
        }
    )


def test_margin_charges_interest_when_underlying_is_flat() -> None:
    policy = MarginPolicy("static", "Static", 1.5)
    equity, summary = run_margin_overlay(_curve([100.0, 100.0, 100.0]), policy, initial_capital=100.0)
    assert equity["margin_equity"].iloc[-1] < 100.0
    assert equity["interest_paid_cumulative"].iloc[-1] > 0.0


def test_margin_amplifies_positive_invested_curve() -> None:
    policy = MarginPolicy("static", "Static", 1.5)
    _, leveraged = run_margin_overlay(_curve([100.0, 110.0, 121.0]), policy, initial_capital=100.0)
    _, cash = run_margin_overlay(
        _curve([100.0, 110.0, 121.0]),
        MarginPolicy("cash", "Cash", 1.0),
        initial_capital=100.0,
    )
    assert leveraged["total_return_pct"] > cash["total_return_pct"]


def test_drawdown_breach_triggers_deleveraging_cooldown() -> None:
    policy = MarginPolicy("static", "Static", 1.5)
    equity, summary = run_margin_overlay(
        _curve([100.0, 120.0, 80.0, 90.0, 100.0]),
        policy,
        initial_capital=100.0,
        forced_deleverage_drawdown=-0.20,
        cooldown_sessions=2,
    )
    assert summary["margin_calls"] == 1
    assert equity.loc[3, "borrow_ratio"] == 0.0
    assert equity.loc[3, "decision_reason"] == "FORCED_COOLDOWN"


def test_margin_pays_incremental_cost_when_underlying_rotates() -> None:
    policy = MarginPolicy("static", "Static", 1.5)
    turnover = pd.DataFrame({"date": ["2025-01-03"], "gross_value": [50.0]})
    without_rotation, _ = run_margin_overlay(_curve([100.0, 110.0]), policy, initial_capital=100.0)
    with_rotation, _ = run_margin_overlay(
        _curve([100.0, 110.0]),
        policy,
        turnover=turnover,
        initial_capital=100.0,
    )
    assert (
        with_rotation["financing_trade_cost_cumulative"].iloc[-1]
        > without_rotation["financing_trade_cost_cumulative"].iloc[-1]
    )
