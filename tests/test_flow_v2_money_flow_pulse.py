from __future__ import annotations

import pandas as pd

from scripts.run_flow_v2_production_demo import _money_flow_pulse


def test_money_flow_pulse_surfaces_observations_when_market_gate_is_risk_off() -> None:
    dates = pd.date_range("2026-05-18", periods=6, freq="D")
    rows: list[dict[str, object]] = []
    for idx, date in enumerate(dates):
        for symbol, score, sponsorship, absorption, distribution, value_ratio in [
            ("AAA", 60.0 + idx * 2.0, 58.0, 62.0, 35.0, 1.30),
            ("BBB", 54.0 + idx, 52.0, 50.0, 45.0, 1.05),
            ("CCC", 45.0 - idx, 35.0, 30.0, 78.0, 0.70),
        ]:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "industry": "Test",
                    "flow_v2_rotation_score_flow_heavy": score,
                    "flow_sponsorship_score": sponsorship,
                    "flow_absorption_score": absorption,
                    "distribution_pressure_score": distribution,
                    "value_ratio_20": value_ratio,
                    "rs_percentile_20": 0.75,
                    "mkt_CHDM20": 45.0 - idx,
                    "mkt_DS20": 0.50 + idx * 0.02,
                    "mkt_regime_state": "RISK_OFF",
                }
            )

    pulse, today, recent = _money_flow_pulse(
        pd.DataFrame(rows),
        dates[-1],
        "flow_v2_rotation_score_flow_heavy",
    )

    assert pulse["pulse_state"] == "REGIME_CHAN_GIAI_NGAN"
    assert pulse["strong_inflow_symbols_today"] == 2
    assert pulse["persistent_inflow_symbols_5d"] == 2
    assert [row["symbol"] for row in today] == ["AAA", "BBB"]
    assert today[0]["flow_signal"] == "DONG_TIEN_VAO_MANH"
    assert today[0]["flow_trend_5d"] == "DUY_TRI_TIEN_VAO"
    assert [row["symbol"] for row in recent] == ["AAA", "BBB"]
    assert recent[0]["inflow_days_5d"] == 5
    assert recent[0]["flow_score_avg_5d"] == 66.0
