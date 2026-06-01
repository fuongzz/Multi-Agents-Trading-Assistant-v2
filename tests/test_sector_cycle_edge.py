import pandas as pd

from multiagents_trading_assistant.research.sector_rotation.sector_cycle_edge import (
    add_forward_sector_returns,
    summarize_leader_spread,
)


def _sector_frame() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    rows = []
    for sector, rank, returns in (
        ("Leader", 1.0, [0.01, 0.02, 0.03, 0.04, 0.05]),
        ("Middle", 0.5, [0.00, 0.01, 0.01, 0.02, 0.02]),
        ("Laggard", 0.1, [-0.01, -0.02, -0.03, -0.04, -0.05]),
        ("Weak", 0.2, [-0.01, -0.01, -0.02, -0.02, -0.03]),
    ):
        for date, ret in zip(dates, returns):
            rows.append(
                {
                    "date": date,
                    "sector_l1": sector,
                    "sector_ret_1d": ret,
                    "mkt_ret_1d": 0.0,
                    "sector_rs_rank_20d": rank,
                    "money_cycle_state": "MARKUP",
                }
            )
    return pd.DataFrame(rows)


def test_forward_return_is_shifted_from_future_row():
    result = add_forward_sector_returns(_sector_frame(), horizons=(1,))
    leader = result[result["sector_l1"] == "Leader"].sort_values("date")

    assert leader.iloc[0]["fwd_ret_1d"] == 0.02
    assert pd.isna(leader.iloc[-1]["fwd_ret_1d"])


def test_leader_spread_rewards_future_leadership():
    prepared = add_forward_sector_returns(_sector_frame(), horizons=(1,))
    summary = summarize_leader_spread(prepared, top_n=1, horizons=(1,))
    all_regimes = summary[summary["money_cycle_state"] == "ALL"].iloc[0]

    assert all_regimes["leader_minus_laggard_pct"] > 0
    assert all_regimes["spread_positive_rate_pct"] == 100.0
