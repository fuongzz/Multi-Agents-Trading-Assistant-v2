from __future__ import annotations

import json

import pandas as pd

from scripts.audit_current_paper_contract import build_audit, write_audit


def _seed_dashboard(tmp_path) -> None:
    pd.DataFrame(
        [
            {"sleeve_id": "mvp_p5", "label": "MVP p5", "paper_equity": 101.0, "open_positions": 1},
            {"sleeve_id": "mvp_p4", "label": "MVP p4", "paper_equity": 102.0, "open_positions": 1},
            {"sleeve_id": "flow_v2", "label": "Flow V2", "paper_equity": 103.0, "open_positions": 1},
            {"sleeve_id": "flow_v2_baseline", "label": "Flow baseline", "paper_equity": 104.0, "open_positions": 2},
            {"sleeve_id": "flow_v2_tiered", "label": "Flow tiered", "paper_equity": 105.0, "open_positions": 2},
            {"sleeve_id": "hostile_combo_long", "label": "Hostile", "paper_equity": 100.0, "open_positions": 0},
            {"sleeve_id": "core_mvp9_rank2", "label": "Core rank2", "paper_equity": 100.0, "open_positions": 0},
        ]
    ).to_csv(tmp_path / "00_account_overview.csv", index=False)
    pd.DataFrame(
        [
            {
                "sleeve_id": "mvp_p5",
                "status": "Verified baseline only",
                "total_return_pct": 45.47,
                "source_note": "Baseline exit only, not exact paper contract.",
            },
            {
                "sleeve_id": "mvp_p4",
                "status": "Reference only",
                "total_return_pct": 114.0,
                "source_note": "Closest current-paper reference.",
            },
            {
                "sleeve_id": "flow_v2",
                "status": "Chua xac minh",
                "total_return_pct": None,
                "source_note": "Separate paper exit contract.",
            },
            {
                "sleeve_id": "flow_v2_baseline",
                "status": "Verified",
                "total_return_pct": 85.44,
                "source_note": "Audited engine.",
            },
            {
                "sleeve_id": "flow_v2_tiered",
                "status": "Verified",
                "total_return_pct": 83.48,
                "source_note": "Audited engine.",
            },
        ]
    ).to_csv(tmp_path / "13_backtest_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "sleeve_id": "mvp_p5",
                "symbol": "AAA",
                "fill_model": "signal_close_T_fill_next_open_T_plus_1",
            },
            {
                "sleeve_id": "mvp_p4",
                "symbol": "BBB",
                "fill_model": "signal_close_T_fill_next_open_T_plus_1",
            },
        ]
    ).to_csv(tmp_path / "paper_ledger_open.csv", index=False)
    pd.DataFrame(columns=["sleeve_id", "symbol"]).to_csv(tmp_path / "paper_ledger_closed.csv", index=False)
    pd.DataFrame(columns=["sleeve_id", "symbol"]).to_csv(tmp_path / "paper_ledger_events.csv", index=False)
    for sleeve_id in ["flow_v2_baseline", "flow_v2_tiered"]:
        (tmp_path / sleeve_id).mkdir()
        pd.DataFrame([{"date": "2026-05-29", "equity": 100.0}]).to_csv(
            tmp_path / sleeve_id / "paper_forward_equity.csv",
            index=False,
        )
        pd.DataFrame([{"date": "2026-05-29", "side": "BUY"}]).to_csv(
            tmp_path / sleeve_id / "paper_forward_orders.csv",
            index=False,
        )


def test_current_paper_contract_audit_marks_unverified_contracts(tmp_path) -> None:
    _seed_dashboard(tmp_path)

    audit, summary = build_audit(tmp_path)
    status_by_sleeve = dict(zip(audit["sleeve_id"], audit["audit_status"]))

    assert status_by_sleeve["flow_v2_baseline"] == "VERIFIED"
    assert status_by_sleeve["flow_v2_tiered"] == "VERIFIED"
    assert "mvp_p5" not in status_by_sleeve
    assert "mvp_p4" not in status_by_sleeve
    assert status_by_sleeve["flow_v2"] == "NEEDS_EXACT_RERUN"
    assert status_by_sleeve["hostile_combo_long"] == "RESEARCH_ONLY"
    assert status_by_sleeve["core_mvp9_rank2"] == "NEEDS_EXACT_RERUN"
    assert summary["verified_count"] == 2
    assert summary["needs_exact_rerun_count"] == 2
    assert summary["ready_for_strategy_optimization"] is False


def test_current_paper_contract_audit_writes_csv_and_json(tmp_path) -> None:
    _seed_dashboard(tmp_path)

    csv_path, json_path, audit, summary = write_audit(tmp_path)

    assert csv_path.exists()
    assert json_path.exists()
    assert len(audit) == 5
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["needs_exact_rerun_count"] == summary["needs_exact_rerun_count"]


def test_current_paper_contract_audit_ignores_deactivated_core_mvp_report(tmp_path) -> None:
    _seed_dashboard(tmp_path)
    exact_dir = tmp_path / "current_paper_contract_backtest"
    exact_dir.mkdir()
    pd.DataFrame(
        [
            {
                "sleeve_id": "mvp_p5",
                "status": "Verified",
                "total_return_pct": 12.34,
                "source_note": "Exact current paper contract backtest",
                "contract_id": "signal_close_T_fill_next_open_T_plus_1_no_fee_agent_exit_v1",
            },
            {
                "sleeve_id": "mvp_p4",
                "status": "Verified",
                "total_return_pct": 10.0,
                "source_note": "Exact current paper contract backtest",
                "contract_id": "signal_close_T_fill_next_open_T_plus_1_no_fee_agent_exit_v1",
            },
        ]
    ).to_csv(exact_dir / "summary.csv", index=False)

    audit, summary = build_audit(tmp_path)
    status_by_sleeve = dict(zip(audit["sleeve_id"], audit["audit_status"]))
    return_by_sleeve = dict(zip(audit["sleeve_id"], audit["backtest_return_pct"]))

    assert "mvp_p5" not in status_by_sleeve
    assert "mvp_p4" not in status_by_sleeve
    assert "mvp_p5" not in return_by_sleeve
    assert summary["verified_count"] == 2
    assert summary["needs_exact_rerun_count"] == 2
