import pandas as pd

import scripts.backtest_flow_v2_rotation_production_like as flow_v2
from scripts.backtest_flow_v2_rotation_production_like import (
    RotationConfig,
    _eligible_pool,
    _execute_rebalance,
    _select_symbols,
    _target_weights,
)


def test_dropped_symbol_is_fully_sold_without_round_lot_residual():
    cfg = RotationConfig(
        universe="vn100",
        start="2020-01-01",
        end="2020-12-31",
        positions=2,
        rebalance_days=10,
    )
    positions = {"BCM": 15_707_500}
    trades = []

    cash, _ = _execute_rebalance(
        date=pd.Timestamp("2020-09-24"),
        target_symbols=[],
        open_by_symbol=pd.Series({"BCM": 37.95}),
        close_by_symbol=pd.Series({"BCM": 37.95}),
        cash=0.0,
        positions=positions,
        hold_counts={"BCM": 1},
        cfg=cfg,
        trades=trades,
    )

    assert cash > 0
    assert positions == {}
    assert trades[0]["shares"] == 15_707_500


def test_blocked_sale_reserves_slot_and_prevents_extra_position():
    cfg = RotationConfig(
        universe="vn100",
        start="2020-01-01",
        end="2020-12-31",
        positions=2,
        rebalance_days=10,
        min_trade_value=0.0,
    )
    positions = {"BLOCKED": 10_000}
    trades = []

    cash, _ = _execute_rebalance(
        date=pd.Timestamp("2020-12-31"),
        target_symbols=["AAA", "BBB"],
        open_by_symbol=pd.Series({"AAA": 10_000.0, "BBB": 10_000.0}),
        close_by_symbol=pd.Series({"BLOCKED": 10_000.0, "AAA": 10_000.0, "BBB": 10_000.0}),
        cash=1_000_000_000.0,
        positions=positions,
        hold_counts={"BLOCKED": 1},
        cfg=cfg,
        trades=trades,
    )

    assert cash > 0
    assert set(positions) == {"BLOCKED", "AAA"}
    assert len(positions) == 2


def test_backtest_converts_parquet_thousand_vnd_price_before_sizing(monkeypatch):
    features = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "symbol": "AAA", "open": 10.0, "close": 10.0},
            {"date": pd.Timestamp("2024-01-03"), "symbol": "AAA", "open": 10.0, "close": 10.0},
        ]
    )
    for column in [
        "flow_v2_rotation_score_balanced",
        "sector_cycle_score",
        "flow_sponsorship_score",
        "flow_absorption_score",
        "distribution_pressure_score",
        "rs_percentile_20",
        "value_ratio_20",
    ]:
        features[column] = 50.0
    monkeypatch.setattr(flow_v2, "_select_symbols", lambda *_args, **_kwargs: ["AAA"])
    cfg = RotationConfig(
        universe="vn100",
        start="2024-01-02",
        end="2024-01-03",
        positions=1,
        rebalance_days=10,
        initial_capital=10_000_000.0,
        min_trade_value=0.0,
        price_unit_multiplier=1000.0,
    )

    result = flow_v2.run_backtest(cfg, features=features)
    buy = result["trades"].loc[result["trades"]["side"] == "BUY"].iloc[0]

    assert buy["price"] == 10_005.0
    assert buy["shares"] == 900


def test_forward_paper_mode_keeps_open_position_at_latest_close(monkeypatch):
    features = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "symbol": "AAA", "open": 10.0, "close": 10.0},
            {"date": pd.Timestamp("2024-01-03"), "symbol": "AAA", "open": 10.0, "close": 10.2},
        ]
    )
    for column in [
        "flow_v2_rotation_score_balanced",
        "sector_cycle_score",
        "flow_sponsorship_score",
        "flow_absorption_score",
        "distribution_pressure_score",
        "rs_percentile_20",
        "value_ratio_20",
    ]:
        features[column] = 50.0
    monkeypatch.setattr(flow_v2, "_select_symbols", lambda *_args, **_kwargs: ["AAA"])
    cfg = RotationConfig(
        universe="vn100",
        start="2024-01-02",
        end="2024-01-03",
        positions=1,
        rebalance_days=10,
        initial_capital=10_000_000.0,
        min_trade_value=0.0,
        price_unit_multiplier=1000.0,
        liquidate_at_end=False,
    )

    result = flow_v2.run_backtest(cfg, features=features)

    assert result["open_positions"] == {"AAA": 900}
    assert "FINAL_LIQUIDATION" not in set(result["trades"]["reason"])


def test_entry_extension_cap_excludes_overextended_candidate():
    day = pd.DataFrame(
        [
            {
                "symbol": "NORMAL",
                "mkt_regime_state": "RISK_ON",
                "mkt_regime_score": 60,
                "mkt_CHDM20": 55,
                "mkt_DS20": 0.40,
                "above_ma50": True,
                "distribution_pressure_score": 45,
                "flow_sponsorship_score": 55,
                "value_ratio_20": 1.1,
                "rs_percentile_20": 0.70,
                "ret_20d_local": 0.18,
                "distance_ma20": 0.08,
            },
            {
                "symbol": "HOT",
                "mkt_regime_state": "RISK_ON",
                "mkt_regime_score": 60,
                "mkt_CHDM20": 55,
                "mkt_DS20": 0.40,
                "above_ma50": True,
                "distribution_pressure_score": 45,
                "flow_sponsorship_score": 55,
                "value_ratio_20": 1.1,
                "rs_percentile_20": 0.70,
                "ret_20d_local": 0.35,
                "distance_ma20": 0.21,
            },
        ]
    )

    pool = _eligible_pool(
        day,
        "risk_on_or_strong_neutral",
        "high_rs",
        max_entry_ret_20d=0.20,
        max_entry_distance_ma20=0.15,
    )

    assert list(pool["symbol"]) == ["NORMAL"]

    pool_with_existing_position = _eligible_pool(
        day,
        "risk_on_or_strong_neutral",
        "high_rs",
        max_entry_ret_20d=0.20,
        max_entry_distance_ma20=0.15,
        current_symbols={"HOT"},
    )

    assert set(pool_with_existing_position["symbol"]) == {"NORMAL", "HOT"}


def test_no_market_gate_keeps_stock_quality_filter_but_allows_risk_off_candidate():
    day = pd.DataFrame(
        [
            {
                "symbol": "QUALITY",
                "mkt_regime_state": "RISK_OFF",
                "mkt_regime_score": 20,
                "mkt_CHDM20": 30,
                "mkt_DS20": 0.80,
                "above_ma50": True,
                "distribution_pressure_score": 45,
                "flow_sponsorship_score": 55,
                "value_ratio_20": 1.1,
                "rs_percentile_20": 0.70,
            },
            {
                "symbol": "WEAK",
                "mkt_regime_state": "RISK_OFF",
                "mkt_regime_score": 20,
                "mkt_CHDM20": 30,
                "mkt_DS20": 0.80,
                "above_ma50": False,
                "distribution_pressure_score": 75,
                "flow_sponsorship_score": 20,
                "value_ratio_20": 0.5,
                "rs_percentile_20": 0.30,
            },
        ]
    )

    gated = _eligible_pool(day, "risk_on_or_strong_neutral", "high_rs")
    no_gate = _eligible_pool(day, "none", "high_rs")

    assert gated.empty
    assert list(no_gate["symbol"]) == ["QUALITY"]


def test_early_exit_uses_next_open_after_flow_momentum_break(monkeypatch):
    features = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "symbol": "AAA", "open": 10.0, "close": 10.0, "above_ma50": True},
            {"date": pd.Timestamp("2024-01-03"), "symbol": "AAA", "open": 10.0, "close": 9.8, "above_ma50": False},
            {"date": pd.Timestamp("2024-01-04"), "symbol": "AAA", "open": 9.7, "close": 9.7, "above_ma50": False},
        ]
    )
    for column in [
        "flow_v2_rotation_score_balanced",
        "sector_cycle_score",
        "flow_sponsorship_score",
        "flow_absorption_score",
        "distribution_pressure_score",
        "rs_percentile_20",
        "value_ratio_20",
    ]:
        features[column] = 50.0
    monkeypatch.setattr(flow_v2, "_select_symbols", lambda *_args, **_kwargs: ["AAA"])
    cfg = RotationConfig(
        universe="vn100",
        start="2024-01-02",
        end="2024-01-04",
        positions=1,
        rebalance_days=10,
        initial_capital=10_000_000.0,
        min_trade_value=0.0,
        price_unit_multiplier=1000.0,
        early_exit_mode="flow_momentum_break",
    )

    result = flow_v2.run_backtest(cfg, features=features)
    sell = result["trades"].loc[result["trades"]["side"] == "SELL"].iloc[0]

    assert sell["date"] == "2024-01-04"
    assert sell["signal_date"] == "2024-01-03"
    assert sell["reason"] == "EARLY_EXIT_FLOW_MOMENTUM_BREAK"


def test_tiered_exit_reduces_position_once_at_next_open(monkeypatch):
    features = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-02"), "symbol": "AAA", "open": 10.0, "close": 10.0},
            {"date": pd.Timestamp("2024-01-03"), "symbol": "AAA", "open": 10.0, "close": 9.9},
            {"date": pd.Timestamp("2024-01-04"), "symbol": "AAA", "open": 9.8, "close": 9.8},
            {"date": pd.Timestamp("2024-01-05"), "symbol": "AAA", "open": 9.8, "close": 9.8},
        ]
    )
    features["above_ma50"] = True
    features["flow_sponsorship_score"] = [55.0, 44.0, 44.0, 44.0]
    features["rs_percentile_20"] = [0.70, 0.55, 0.55, 0.55]
    features["distribution_pressure_score"] = 50.0
    for column in [
        "flow_v2_rotation_score_balanced",
        "sector_cycle_score",
        "flow_absorption_score",
        "value_ratio_20",
    ]:
        features[column] = 50.0
    monkeypatch.setattr(flow_v2, "_select_symbols", lambda *_args, **_kwargs: ["AAA"])
    cfg = RotationConfig(
        universe="vn100",
        start="2024-01-02",
        end="2024-01-05",
        positions=1,
        rebalance_days=10,
        initial_capital=10_000_000.0,
        min_trade_value=0.0,
        price_unit_multiplier=1000.0,
        early_exit_mode="flow_momentum_tiered",
    )

    trades = flow_v2.run_backtest(cfg, features=features)["trades"]
    partial = trades.loc[trades["reason"] == "PARTIAL_EXIT_FLOW_WEAKNESS"]

    assert list(partial["date"]) == ["2024-01-04"]
    assert partial.iloc[0]["signal_date"] == "2024-01-03"
    assert partial.iloc[0]["shares"] == 400


def test_inverse_atr_and_neutral_buffer_define_target_weights():
    day = pd.DataFrame(
        [
            {"symbol": "LOW_VOL", "close": 10.0, "atr14": 0.2, "mkt_regime_state": "NEUTRAL"},
            {"symbol": "HIGH_VOL", "close": 10.0, "atr14": 0.4, "mkt_regime_state": "NEUTRAL"},
        ]
    )
    cfg = RotationConfig(
        universe="vn100",
        start="2024-01-01",
        end="2024-01-02",
        positions=2,
        rebalance_days=10,
        allocation_mode="inverse_atr",
        neutral_gross_exposure=0.70,
    )

    weights = _target_weights(day, ["LOW_VOL", "HIGH_VOL"], cfg)

    assert round(sum(weights.values()), 2) == 0.70
    assert round(weights["LOW_VOL"], 4) == 0.4667
    assert round(weights["HIGH_VOL"], 4) == 0.2333


def test_sector_cap_chooses_second_symbol_from_different_industry():
    day = pd.DataFrame(
        [
            {"symbol": "BANK_A", "industry": "Bank", "flow_v2_rotation_score_flow_heavy": 90.0},
            {"symbol": "BANK_B", "industry": "Bank", "flow_v2_rotation_score_flow_heavy": 89.0},
            {"symbol": "STEEL", "industry": "Steel", "flow_v2_rotation_score_flow_heavy": 80.0},
        ]
    )
    for column, value in {
        "mkt_regime_state": "RISK_ON",
        "mkt_regime_score": 60,
        "mkt_CHDM20": 55,
        "mkt_DS20": 0.40,
        "above_ma50": True,
        "distribution_pressure_score": 45,
        "flow_sponsorship_score": 55,
        "value_ratio_20": 1.1,
        "rs_percentile_20": 0.70,
    }.items():
        day[column] = value

    selected = _select_symbols(
        day,
        2,
        "risk_on_or_strong_neutral",
        "high_rs",
        "flow_heavy",
        diversification_mode="distinct_industry",
    )

    assert selected == ["BANK_A", "STEEL"]
