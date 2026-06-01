from __future__ import annotations

from datetime import datetime

import pandas as pd

import scripts.update_combined_paper_realtime as realtime


def _seed_files(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "sleeve_id": "mvp_p5",
                "symbol": "GVR",
                "strategy_name": "money_cycle_reset_smt_confirm",
                "signal_date": "2026-05-26",
                "decision_status": "PAPER_TARGET",
                "reference_close_vnd": 35000.0,
                "score": 2.2,
            }
        ]
    ).to_csv(tmp_path / "17_signal_plans.csv", index=False)
    pd.DataFrame(
        [
            {
                "sleeve_id": "mvp_p5",
                "symbol": "GVR",
                "action": "PAPER_BUY_CANDIDATE",
                "max_positions_context": 5,
            }
        ]
    ).to_csv(tmp_path / "05_orders_targets.csv", index=False)
    pd.DataFrame(columns=realtime.OPEN_COLUMNS).to_csv(tmp_path / "paper_ledger_open.csv", index=False)
    pd.DataFrame(columns=["sleeve_id", "symbol", "signal_date"]).to_csv(tmp_path / "paper_ledger_closed.csv", index=False)
    pd.DataFrame(columns=realtime.EVENT_COLUMNS).to_csv(tmp_path / "paper_ledger_events.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "paper_open_holdings.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / "paper_account_pnl.csv", index=False)


def _seed_rolling_history(tmp_path) -> None:
    (tmp_path / "mvp_p5").mkdir()
    pd.DataFrame(
        [
            {"sleeve_id": "mvp_p5", "date": "2026-05-27", "signal_count": 2, "paper_buy_candidate_count": 2},
            {"sleeve_id": "mvp_p5", "date": "2026-05-28", "signal_count": 2, "paper_buy_candidate_count": 2},
        ]
    ).to_csv(tmp_path / "09_rolling_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-05-27",
                "priority": 1,
                "symbol": "GVR",
                "action": "PAPER_BUY_CANDIDATE",
                "max_positions_context": 5,
                "strategy_name": "money_cycle_reset_smt_confirm",
                "edge_rank_score": 2.7,
            },
            {
                "date": "2026-05-27",
                "priority": 2,
                "symbol": "HCM",
                "action": "PAPER_BUY_CANDIDATE",
                "max_positions_context": 5,
                "strategy_name": "money_cycle_reset_smt_confirm",
                "edge_rank_score": 1.8,
            },
            {
                "date": "2026-05-28",
                "priority": 1,
                "symbol": "VPL",
                "action": "PAPER_BUY_CANDIDATE",
                "max_positions_context": 5,
                "strategy_name": "money_cycle_reset_smt_confirm",
                "edge_rank_score": 1.9,
            },
        ]
    ).to_csv(tmp_path / "mvp_p5" / "rolling_candidate_actions_14d.csv", index=False)


def test_realtime_paper_entry_records_once(tmp_path) -> None:
    _seed_files(tmp_path)

    first = realtime._enter_realtime_paper_positions(
        tmp_path,
        {"GVR": 35100.0},
        pd.Timestamp("2026-05-27"),
    )
    second = realtime._enter_realtime_paper_positions(
        tmp_path,
        {"GVR": 35200.0},
        pd.Timestamp("2026-05-27"),
    )

    assert len(first) == 1
    assert second == []
    open_ = pd.read_csv(tmp_path / "paper_ledger_open.csv")
    events = pd.read_csv(tmp_path / "paper_ledger_events.csv")
    assert len(open_) == 1
    assert open_.iloc[0]["fill_model"] == "live_quote_after_market_open_T_plus_1_compound_equity"
    assert events.iloc[0]["reason"] == "REALTIME_PAPER_BUY_CANDIDATE"


def test_realtime_hostile_entry_respects_target_exposure(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "sleeve_id": "hostile_combo_long",
                "symbol": "VHM",
                "strategy_name": "hostile_combo_long",
                "signal_date": "2026-05-29",
                "decision_status": "PAPER_TARGET",
                "reference_close_vnd": 50000.0,
                "score": 0.97,
            }
        ]
    ).to_csv(tmp_path / "17_signal_plans.csv", index=False)
    pd.DataFrame(
        [
            {
                "sleeve_id": "hostile_combo_long",
                "symbol": "VHM",
                "action": "PAPER_BUY_CANDIDATE",
                "max_positions_context": 1,
                "target_exposure": 0.25,
            }
        ]
    ).to_csv(tmp_path / "05_orders_targets.csv", index=False)
    pd.DataFrame(columns=realtime.OPEN_COLUMNS).to_csv(tmp_path / "paper_ledger_open.csv", index=False)
    pd.DataFrame(columns=["sleeve_id", "symbol", "signal_date"]).to_csv(tmp_path / "paper_ledger_closed.csv", index=False)
    pd.DataFrame(columns=realtime.EVENT_COLUMNS).to_csv(tmp_path / "paper_ledger_events.csv", index=False)

    entries = realtime._enter_realtime_paper_positions(
        tmp_path,
        {"VHM": 50000.0},
        pd.Timestamp("2026-06-01"),
    )

    assert len(entries) == 1
    assert entries[0]["shares"] == 500
    assert entries[0]["cost_value"] == 25_000_000.0


def test_realtime_pending_survives_signal_plan_regeneration(tmp_path) -> None:
    _seed_files(tmp_path)
    pd.DataFrame(
        [
            {
                "sleeve_id": "mvp_p5",
                "symbol": "GVR",
                "strategy_name": "money_cycle_reset_smt_confirm",
                "signal_date": "2026-05-27",
                "decision_status": "PAPER_TARGET",
                "score": 2.7,
                "reference_close_vnd": 34400.0,
                "max_positions_context": 5,
                "created_at": "2026-05-27T15:30:00",
                "source": "signal_plan_export",
            },
            {
                "sleeve_id": "mvp_p5",
                "symbol": "HCM",
                "strategy_name": "money_cycle_reset_smt_confirm",
                "signal_date": "2026-05-27",
                "decision_status": "PAPER_TARGET",
                "score": 1.8,
                "reference_close_vnd": 27500.0,
                "max_positions_context": 5,
                "created_at": "2026-05-27T15:30:00",
                "source": "signal_plan_export",
            },
        ]
    ).to_csv(tmp_path / "paper_signal_pending.csv", index=False)
    pd.DataFrame(
        [
            {
                "sleeve_id": "mvp_p5",
                "symbol": "VPL",
                "strategy_name": "money_cycle_reset_smt_confirm",
                "signal_date": "2026-05-28",
                "decision_status": "PAPER_TARGET",
                "reference_close_vnd": 92800.0,
                "score": 1.9,
            }
        ]
    ).to_csv(tmp_path / "17_signal_plans.csv", index=False)

    entries = realtime._enter_realtime_paper_positions(
        tmp_path,
        {"GVR": 34.4, "HCM": 27.5, "VPL": 92.8},
        pd.Timestamp("2026-05-28"),
    )

    assert [row["symbol"] for row in entries] == ["GVR", "HCM"]
    assert entries[0]["entry_price"] == 34400.0
    assert entries[1]["entry_price"] == 27500.0


def test_refresh_does_not_open_realtime_entry_before_market_session(tmp_path, monkeypatch) -> None:
    _seed_files(tmp_path)
    monkeypatch.setattr(realtime, "_export_dashboard", lambda _out_dir: None)
    monkeypatch.setattr(realtime, "get_live_price", lambda _symbols: {"GVR": 35100.0})

    status = realtime.refresh_once(
        tmp_path,
        realtime_paper_entries=True,
        now=datetime(2026, 5, 27, 8, 30),
    )

    open_ = pd.read_csv(tmp_path / "paper_ledger_open.csv")
    assert open_.empty
    assert status["paper_entries"] == 0
    flow_status = pd.read_json(tmp_path / "realtime_money_flow_status.json", typ="series")
    assert flow_status["state"] == "WAITING_FOR_MARKET_SESSION"


def test_realtime_flow_overlay_is_confirmation_only() -> None:
    watchlist = pd.DataFrame(
        [{"rank": 1, "symbol": "VND", "industry": "Broker", "flow_signal": "DONG_TIEN_VAO_MANH", "flow_score_today": 75.4}]
    )
    board = pd.DataFrame(
        [{"symbol": "VND", "reference_price": 18000, "close_price": 18200, "foreign_buy_volume": 50000, "foreign_sell_volume": 10000}]
    )

    result = realtime._build_realtime_flow_rows(watchlist, board, priced_at="2026-05-27T09:05:00")

    assert result.iloc[0]["realtime_confirmation"] == "XAC_NHAN_VAO_TRONG_PHIEN"
    assert result.iloc[0]["foreign_net_volume"] == 40000
    assert result.iloc[0]["intraday_change_pct"] == 1.11
    assert "display only" in result.iloc[0]["data_basis"]
