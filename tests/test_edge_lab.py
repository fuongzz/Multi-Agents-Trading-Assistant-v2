import pandas as pd

from multiagents_trading_assistant.edge_lab.features import _add_market_derived_columns
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
