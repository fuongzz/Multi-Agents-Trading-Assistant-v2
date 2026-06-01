from __future__ import annotations

import json

import pandas as pd

import scripts.build_combined_paper_pnl as pnl
from scripts.run_hostile_combo_long_production_demo import DEFAULT_TARGET_EXPOSURE


def test_hostile_candidate_carries_satellite_exposure(tmp_path) -> None:
    sleeve_dir = tmp_path / "hostile_combo_long"
    sleeve_dir.mkdir()
    (sleeve_dir / "hostile_status.json").write_text(
        json.dumps(
            {
                "as_of_date": "2026-05-29",
                "sleeve_id": "hostile_combo_long",
                "sleeve_label": "Hostile Combo Long p1 satellite",
                "target_exposure": DEFAULT_TARGET_EXPOSURE,
                "paper_trading_enabled": True,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "sleeve_id": "hostile_combo_long",
                "signal_date": "2026-05-29",
                "symbol": "VHM",
                "strategy_name": "hostile_combo_long",
                "rank": 1,
                "action": "PAPER_BUY_CANDIDATE",
                "target_exposure": DEFAULT_TARGET_EXPOSURE,
                "target_value": 25_000_000.0,
            }
        ]
    ).to_csv(sleeve_dir / "hostile_target_plan.csv", index=False)

    rows = pnl._hostile_candidates(sleeve_dir, pd.Timestamp("2026-06-01"))

    assert rows[0]["target_exposure"] == DEFAULT_TARGET_EXPOSURE


def test_hostile_entry_uses_target_exposure_for_compound_sizing(tmp_path) -> None:
    sleeve_dir = tmp_path / "hostile_combo_long"
    sleeve_dir.mkdir()
    (sleeve_dir / "hostile_status.json").write_text(
        json.dumps(
            {
                "as_of_date": "2026-05-29",
                "sleeve_id": "hostile_combo_long",
                "sleeve_label": "Hostile Combo Long p1 satellite",
                "target_exposure": DEFAULT_TARGET_EXPOSURE,
                "paper_trading_enabled": True,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "symbol": "VHM",
                "strategy_name": "hostile_combo_long",
                "rank": 1,
                "action": "PAPER_BUY_CANDIDATE",
                "target_exposure": DEFAULT_TARGET_EXPOSURE,
            }
        ]
    ).to_csv(sleeve_dir / "hostile_target_plan.csv", index=False)

    prices = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-06-01"), "symbol": "VHM", "open": 50.0, "close": 51.0},
        ]
    )
    open_, events = pnl._enter_positions(
        out_dir=tmp_path,
        open_=pd.DataFrame(columns=pnl.OPEN_COLUMNS),
        closed=pd.DataFrame(columns=pnl.CLOSED_COLUMNS),
        events=pd.DataFrame(columns=pnl.EVENT_COLUMNS),
        prices=prices,
        as_of=pd.Timestamp("2026-06-01"),
        capital=100_000_000.0,
        lot_size=100,
        price_unit_multiplier=1000.0,
        risk_by_strategy={},
        start_date=None,
    )

    assert len(open_) == 1
    assert open_.iloc[0]["shares"] == 500
    assert open_.iloc[0]["cost_value"] == 25_000_000.0
    assert events.iloc[0]["reason"] == "PAPER_BUY_CANDIDATE"
