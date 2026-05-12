import pandas as pd

from multiagents_trading_assistant.edge_lab.backtest import (
    PortfolioConfig,
    Position,
    _atr_exit_reason,
    _regime_strategy_names,
    run_portfolio,
)
from multiagents_trading_assistant.edge_lab.features import (
    _add_derived_columns,
    _add_market_derived_columns,
    _kalman_adaptive_price_gate,
    _kalman_price_trend,
)
from multiagents_trading_assistant.edge_lab.hypothesis import (
    Hypothesis,
    evaluate_filters,
    rank_candidates,
)
from multiagents_trading_assistant.edge_lab.research_cli import _load_grid_arg


def test_hypothesis_filter_ops_and_ranking():
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "score": [70, 60, 80],
            "state": ["A", "B", "A"],
            "x": [2, 3, 1],
            "x_prev": [1, 4, 1],
        }
    )
    hypothesis = Hypothesis(
        name="demo",
        filters=[
            {"column": "score", "op": ">=", "value": 65},
            {"column": "state", "op": "in", "value": ["A"]},
            {"column": "x", "op": "up"},
        ],
        rank=[
            {"column": "score", "ascending": False, "weight": 1.0},
        ],
    )

    filtered = frame[evaluate_filters(frame, hypothesis)]
    ranked = rank_candidates(filtered, hypothesis)

    assert ranked["symbol"].tolist() == ["AAA"]


def test_between_and_not_up_ops():
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "chdm": [30, 50, 40],
            "ds": [0.2, 0.3, 0.4],
            "ds_prev": [0.3, 0.2, 0.4],
        }
    )
    hypothesis = Hypothesis(
        name="reset",
        filters=[
            {"column": "chdm", "op": "between", "value": [20, 45]},
            {"column": "ds", "op": "not_up"},
        ],
    )

    filtered = frame[evaluate_filters(frame, hypothesis)]

    assert filtered["symbol"].tolist() == ["AAA", "CCC"]


def test_load_grid_arg_accepts_powershell_stripped_keys():
    grid = _load_grid_arg("{smart_money_score:[62,65],mkt_DS20:[0.45]}")

    assert grid == {"smart_money_score": [62, 65], "mkt_DS20": [0.45]}


def test_market_delta_is_date_level_not_symbol_level():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "mkt_CHDM20": [50.0, 50.0, 55.0, 55.0],
            "mkt_DS20": [0.40, 0.40, 0.35, 0.35],
            "mkt_CHDM50": [45.0, 45.0, 46.0, 46.0],
            "mkt_DS50": [0.45, 0.45, 0.44, 0.44],
        }
    )

    _add_market_derived_columns(frame)

    second_day = frame[frame["date"] == pd.Timestamp("2024-01-02")]
    assert second_day["mkt_CHDM20_delta"].tolist() == [5.0, 5.0]
    assert second_day["mkt_regime_state"].isin(["RISK_ON", "NEUTRAL", "RISK_OFF"]).all()


def test_derived_columns_add_edge_score_and_breakout():
    closes = list(range(10, 34)) + [40]
    highs = list(range(11, 35)) + [41]
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=25),
            "symbol": ["AAA"] * 25,
            "open": range(10, 35),
            "high": highs,
            "low": range(9, 34),
            "close": closes,
            "ma20": [20.0] * 25,
            "ma50": [18.0] * 25,
            "mkt_CHDM20": [55.0] * 25,
            "mkt_DS20": [0.35] * 25,
            "mkt_CHDM50": [50.0] * 25,
            "mkt_DS50": [0.40] * 25,
            "smart_money_score": [70.0] * 25,
            "rs_percentile_20": [0.7] * 25,
            "value_flow_quality_score": [65.0] * 25,
            "CHDM50": [60.0] * 25,
            "DS20": [0.3] * 25,
            "distribution_days_10": [0.0] * 25,
            "value_ratio_20": [1.4] * 25,
        }
    )

    _add_derived_columns(frame)

    assert "edge_score" in frame.columns
    assert frame["edge_score"].iloc[-1] > 60
    assert bool(frame["breakout_20"].iloc[-1])
    assert {"kalman_close", "kalman_trend_5d", "kalman_residual_pct"}.issubset(frame.columns)
    assert {
        "kalman_adaptive_close",
        "kalman_adaptive_trend_5d",
        "kalman_adaptive_residual_pct",
        "kalman_confidence",
        "kalman_shock_score",
    }.issubset(frame.columns)


def test_kalman_price_trend_tracks_smooth_uptrend():
    close = pd.Series([10.0, 10.2, 10.1, 10.5, 10.8, 11.1, 11.4, 11.8, 12.0, 12.3])

    kalman = _kalman_price_trend(close)

    assert kalman["kalman_close"].notna().all()
    assert kalman["kalman_close"].iloc[-1] > kalman["kalman_close"].iloc[0]
    assert kalman["kalman_trend_5d"].iloc[-1] > 0
    assert abs(kalman["kalman_residual_pct"].iloc[-1]) < 0.08


def test_adaptive_kalman_outputs_confidence_and_shock_score():
    close = pd.Series([10.0, 10.1, 10.2, 10.25, 10.3, 11.8, 10.5, 10.6])
    frame = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": [1_000_000, 1_100_000, 900_000, 1_000_000, 950_000, 300_000, 900_000, 1_000_000],
            "value_ratio_20": [1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 1.0, 1.0],
            "volume_ratio_20": [1.0, 1.1, 0.9, 1.0, 0.95, 0.3, 0.9, 1.0],
            "smart_money_score_delta": [0.0] * 8,
            "CHDM50": [55.0] * 8,
            "DS20": [0.35] * 8,
            "mkt_regime_score": [55.0] * 8,
        }
    )

    kalman = _kalman_adaptive_price_gate(frame)

    assert kalman["kalman_adaptive_close"].notna().all()
    assert kalman["kalman_confidence"].between(0.0, 1.0).all()
    assert kalman["kalman_shock_score"].iloc[5] > kalman["kalman_shock_score"].iloc[2]


def test_atr_trailing_exit_after_activation():
    position = Position(
        hypothesis="demo",
        symbol="AAA",
        entry_date=pd.Timestamp("2024-01-01"),
        entry_price=100.0,
        shares=100,
        entry_value=10_000.0,
        risk={"trailing_atr_mult": 2.0, "trailing_profit_activation": 0.10},
        entry_atr=5.0,
        highest_price=115.0,
    )

    assert _atr_exit_reason(104.0, position) == "TRAILING_ATR_STOP"


def test_regime_strategy_names_switches_by_market_state():
    risk_on = pd.DataFrame(
        {
            "mkt_regime_state": ["RISK_ON"],
            "mkt_regime_score": [70.0],
            "mkt_CHDM20_delta": [1.0],
            "mkt_DS20_delta": [0.0],
            "mkt_DS20": [0.35],
        }
    )
    risk_off = risk_on.copy()
    risk_off["mkt_regime_state"] = "RISK_OFF"
    risk_off["mkt_regime_score"] = 30.0

    names, label = _regime_strategy_names(risk_on)
    off_names, off_label = _regime_strategy_names(risk_off)

    assert label == "RISK_ON"
    assert "leader_pullback_market_healthy_v2" in names
    assert off_label == "RISK_OFF"
    assert off_names == []


def test_portfolio_deduplicates_same_symbol_orders_in_one_batch():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    features = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1], dates[2], dates[2]],
            "symbol": ["AAA", "BBB"] * 3,
            "open": [10.0, 20.0] * 3,
            "high": [10.5, 20.5] * 3,
            "low": [9.5, 19.5] * 3,
            "close": [10.0, 20.0] * 3,
            "volume": [100_000, 100_000] * 3,
            "value": [1_000_000, 2_000_000] * 3,
            "score": [100.0, 50.0] * 3,
        }
    )
    price_map = {
        "AAA": features[features["symbol"] == "AAA"].copy(),
        "BBB": features[features["symbol"] == "BBB"].copy(),
    }
    h1 = Hypothesis(name="h1", filters=[{"column": "score", "op": ">=", "value": 90}], rank=[{"column": "score"}])
    h2 = Hypothesis(name="h2", filters=[{"column": "score", "op": ">=", "value": 90}], rank=[{"column": "score"}])

    trades, _ = run_portfolio(
        features,
        price_map,
        [h1, h2],
        "2024-01-01",
        "2024-01-03",
        config=PortfolioConfig(max_positions=2, top_n=2),
    )

    assert len(trades[trades["symbol"] == "AAA"]) == 1
