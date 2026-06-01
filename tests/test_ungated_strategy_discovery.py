from __future__ import annotations

import pandas as pd

from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, evaluate_filters
from scripts.backtest_flow_v2_rotation_production_like import _eligible_pool
from scripts.export_combined_paper_dashboard import _build_signal_plans
from scripts.run_flow_v2_production_demo import _ungated_discovery_rows as flow_discovery_rows
from scripts.run_production_dry_run import _ungated_discovery_rows as mvp_discovery_rows
from scripts.run_production_dry_run import _without_market_filters


def test_mvp_discovery_removes_market_filter_only_for_display() -> None:
    hypothesis = Hypothesis(
        name="strategy_demo",
        filters=[
            {"column": "mkt_regime_state", "op": "in", "value": ["RISK_ON"]},
            {"column": "smart_money_score", "op": ">=", "value": 60},
        ],
        rank=[{"column": "smart_money_score", "ascending": False, "weight": 1.0}],
    )
    latest = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "mkt_regime_state": "RISK_OFF",
                "smart_money_score": 70.0,
                "rs_percentile_20": 0.72,
                "value_ratio_20": 1.20,
            }
        ]
    )

    assert not evaluate_filters(latest, hypothesis).any()

    relaxed, removed = _without_market_filters(hypothesis)
    rows = mvp_discovery_rows(
        latest_features=latest,
        hypotheses=[hypothesis],
        as_of=pd.Timestamp("2026-05-25"),
    )

    assert removed == ["mkt_regime_state"]
    assert len(hypothesis.filters) == 2
    assert evaluate_filters(latest, relaxed).all()
    assert [row["symbol"] for row in rows] == ["AAA"]
    assert rows[0]["display_only"] is True


def test_flow_discovery_has_rows_while_execution_gate_stays_empty() -> None:
    day = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-25"),
                "symbol": "AAA",
                "mkt_regime_state": "RISK_OFF",
                "mkt_regime_score": 20.0,
                "mkt_CHDM20": 30.0,
                "mkt_DS20": 0.80,
                "mkt_DS50": 0.75,
                "above_ma50": True,
                "distribution_pressure_score": 35.0,
                "flow_sponsorship_score": 70.0,
                "flow_absorption_score": 64.0,
                "value_ratio_20": 1.25,
                "rs_percentile_20": 0.78,
                "flow_v2_rotation_score_flow_heavy": 76.0,
                "close": 50.0,
                "sector_cycle_score": 60.0,
                "flow_quality_v2_score": 65.0,
                "stock_lifecycle_score": 60.0,
            }
        ]
    )

    gated = _eligible_pool(day, "risk_on_or_strong_neutral", "high_rs")
    discovery = flow_discovery_rows(
        day,
        score_col="flow_v2_rotation_score_flow_heavy",
        pool_filter="high_rs",
        market_gate="risk_on_or_strong_neutral",
        limit=10,
    )

    assert gated.empty
    assert [row["symbol"] for row in discovery] == ["AAA"]
    assert discovery[0]["display_only"] is True


def test_signal_plan_displays_entry_band_without_turning_ungated_idea_into_order() -> None:
    discovery = pd.DataFrame(
        [
            {
                "date": "2026-05-25",
                "sleeves": "mvp_p4",
                "symbol": "AAA",
                "strategy_name": "strategy_demo",
                "close": 50.0,
                "market_regime_state": "RISK_OFF",
                "risk": '{"stop_loss": 0.08, "take_profit": 0.25, "max_holding_bars": 30}',
                "relaxed_rule": "Bỏ qua filter thị trường",
            }
        ]
    )

    plans = _build_signal_plans(pd.DataFrame(), pd.DataFrame(), discovery)
    idea = plans.iloc[0]

    assert idea["decision_status"] == "OBSERVE_ONLY_GATE_REMOVED"
    assert idea["display_only"]
    assert idea["reference_close_vnd"] == 50000.0
    assert idea["entry_zone_low_vnd"] == 49500.0
    assert idea["entry_zone_high_vnd"] == 50500.0
    assert idea["stop_loss_vnd"] == 46000.0
    assert idea["take_profit_vnd"] == 62500.0


def test_signal_plan_preserves_audited_flow_exit_contract() -> None:
    discovery = pd.DataFrame(
        [
            {
                "date": "2026-05-25",
                "sleeves": "flow_v2_baseline, flow_v2_tiered",
                "symbol": "AAA",
                "close": 50.0,
                "relaxed_rule": "Bỏ qua market gate",
            }
        ]
    )

    plans = _build_signal_plans(pd.DataFrame(), pd.DataFrame(), discovery)

    assert plans["stop_loss_vnd"].isna().all()
    assert plans["take_profit_vnd"].isna().all()
    assert "rebalance" in plans.loc[plans["sleeve_id"] == "flow_v2_baseline", "expectation"].iloc[0]
    assert "flow_momentum_tiered" in plans.loc[plans["sleeve_id"] == "flow_v2_tiered", "expectation"].iloc[0]
