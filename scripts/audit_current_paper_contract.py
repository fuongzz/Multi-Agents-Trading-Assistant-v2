"""Audit the current combined paper contract status.

This script is intentionally read-only. It consolidates the dashboard CSVs into
one machine-readable contract audit so later optimization work starts from a
clear baseline instead of scattered notes in the HTML and registry.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "reports" / "combined_paper_trading_demo"

ACTIVE_SLEEVES = [
    "flow_v2",
    "flow_v2_baseline",
    "flow_v2_tiered",
    "hostile_combo_long",
    "core_mvp9_rank2",
]

AUDITED_FLOW_SLEEVES = {"flow_v2_baseline", "flow_v2_tiered"}
CORE_MVP_SLEEVES = {"mvp_p5", "mvp_p4", "core_mvp9_rank2"}
EXACT_CONTRACT_DIRNAME = "current_paper_contract_backtest"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _first_text(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    values = frame[column].dropna().astype(str)
    return "" if values.empty else values.iloc[0]


def _sleeve_label(account_row: pd.DataFrame, sleeve_id: str) -> str:
    label = _first_text(account_row, "label")
    return label or sleeve_id


def _comparison_for_sleeve(comparison: pd.DataFrame, sleeve_id: str) -> pd.DataFrame:
    if comparison.empty or "sleeve_id" not in comparison:
        return pd.DataFrame()
    return comparison[comparison["sleeve_id"].astype(str).eq(sleeve_id)].copy()


def _best_comparison_status(rows: pd.DataFrame) -> tuple[str, float | None, str]:
    if rows.empty:
        return "missing", None, "No strategy_backtest_comparison row found."
    exact_verified = rows[
        rows.get("status", pd.Series(dtype=str)).astype(str).str.lower().eq("verified")
        & rows.get("total_return_pct", pd.Series(dtype=str)).notna()
    ]
    if not exact_verified.empty:
        row = exact_verified.iloc[0]
        return str(row.get("status") or ""), _to_float(row.get("total_return_pct")), str(row.get("source_note") or "")
    row = rows.iloc[0]
    return str(row.get("status") or ""), _to_float(row.get("total_return_pct")), str(row.get("source_note") or "")


def _ledger_stats(out_dir: Path, sleeve_id: str) -> dict[str, Any]:
    open_ = _read_csv(out_dir / "paper_ledger_open.csv")
    closed = _read_csv(out_dir / "paper_ledger_closed.csv")
    events = _read_csv(out_dir / "paper_ledger_events.csv")
    out: dict[str, Any] = {
        "ledger_open_rows": 0,
        "ledger_closed_rows": 0,
        "ledger_event_rows": 0,
        "ledger_fill_models": "",
    }
    if not open_.empty and "sleeve_id" in open_:
        sleeve_open = open_[open_["sleeve_id"].astype(str).eq(sleeve_id)].copy()
        out["ledger_open_rows"] = int(len(sleeve_open))
        if "fill_model" in sleeve_open:
            out["ledger_fill_models"] = ",".join(sorted(sleeve_open["fill_model"].dropna().astype(str).unique()))
    if not closed.empty and "sleeve_id" in closed:
        out["ledger_closed_rows"] = int((closed["sleeve_id"].astype(str) == sleeve_id).sum())
    if not events.empty and "sleeve_id" in events:
        out["ledger_event_rows"] = int((events["sleeve_id"].astype(str) == sleeve_id).sum())
    return out


def _audited_flow_stats(out_dir: Path, sleeve_id: str) -> dict[str, Any]:
    equity = _read_csv(out_dir / sleeve_id / "paper_forward_equity.csv")
    orders = _read_csv(out_dir / sleeve_id / "paper_forward_orders.csv")
    return {
        "forward_equity_rows": int(len(equity)),
        "forward_order_rows": int(len(orders)),
        "forward_equity_latest": _first_text(equity.tail(1), "date"),
    }


def _exact_contract_rows(out_dir: Path) -> pd.DataFrame:
    summary = _read_csv(out_dir / EXACT_CONTRACT_DIRNAME / "summary.csv")
    if summary.empty or "sleeve_id" not in summary:
        return pd.DataFrame()
    return summary.copy()


def _exact_contract_for_sleeve(exact_contract: pd.DataFrame, sleeve_id: str) -> pd.DataFrame:
    if exact_contract.empty or "sleeve_id" not in exact_contract:
        return pd.DataFrame()
    return exact_contract[exact_contract["sleeve_id"].astype(str).eq(sleeve_id)].copy()


def _exact_contract_status(rows: pd.DataFrame) -> tuple[bool, float | None, str, str]:
    if rows.empty:
        return False, None, "", ""
    row = rows.iloc[0]
    verified = str(row.get("status") or "").lower() == "verified"
    note = str(row.get("source_note") or "")
    contract_id = str(row.get("contract_id") or "")
    exact_note = "exact current paper contract" in note.lower() or "current-paper" in note.lower()
    return verified and exact_note, _to_float(row.get("total_return_pct")), note, contract_id


def _audit_row(out_dir: Path, sleeve_id: str, account: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, Any]:
    account_row = account[account["sleeve_id"].astype(str).eq(sleeve_id)].copy() if "sleeve_id" in account else pd.DataFrame()
    comparison_rows = _comparison_for_sleeve(comparison, sleeve_id)
    backtest_status, backtest_return, source_note = _best_comparison_status(comparison_rows)
    row: dict[str, Any] = {
        "sleeve_id": sleeve_id,
        "label": _sleeve_label(account_row, sleeve_id),
        "paper_equity": _to_float(account_row.iloc[0].get("paper_equity")) if not account_row.empty else None,
        "open_positions": _to_float(account_row.iloc[0].get("open_positions")) if not account_row.empty else None,
        "backtest_status": backtest_status,
        "backtest_return_pct": backtest_return,
        "source_note": source_note,
    }

    if sleeve_id in CORE_MVP_SLEEVES:
        row.update({"contract_type": "core_mvp_ledger_with_position_exit_agent"})
        row.update(_ledger_stats(out_dir, sleeve_id))
        exact_rows = _exact_contract_for_sleeve(_exact_contract_rows(out_dir), sleeve_id)
        exact_verified, exact_return, exact_note, exact_contract_id = _exact_contract_status(exact_rows)
        row["exact_contract_status"] = "VERIFIED" if exact_verified else "missing"
        row["exact_contract_return_pct"] = exact_return
        row["exact_contract_note"] = exact_note
        row["exact_contract_id"] = exact_contract_id
        if account_row.empty:
            row["audit_status"] = "FAIL_MISSING_PAPER_ACCOUNT"
            row["action_required"] = "Rebuild paper PnL before optimizing."
        elif exact_verified:
            row["audit_status"] = "VERIFIED"
            row["backtest_return_pct"] = exact_return
            row["source_note"] = exact_note
            row["action_required"] = ""
        else:
            row["audit_status"] = "NEEDS_EXACT_RERUN"
            row["action_required"] = (
                "Run exact current-paper contract backtest through the target period; "
                "existing comparison is baseline/reference only."
            )
        return row

    if sleeve_id in AUDITED_FLOW_SLEEVES:
        row.update({"contract_type": "audited_flow_forward_replay"})
        row.update(_audited_flow_stats(out_dir, sleeve_id))
        has_forward = int(row["forward_equity_rows"]) > 0
        verified = str(backtest_status).lower() == "verified"
        if account_row.empty:
            row["audit_status"] = "FAIL_MISSING_PAPER_ACCOUNT"
            row["action_required"] = "Rebuild paper PnL before optimizing."
        elif not has_forward:
            row["audit_status"] = "FAIL_MISSING_FORWARD_AUDIT"
            row["action_required"] = f"Rebuild {sleeve_id} forward equity/orders."
        elif not verified:
            row["audit_status"] = "NEEDS_EXACT_RERUN"
            row["action_required"] = "Backtest comparison is not marked Verified."
        else:
            row["audit_status"] = "VERIFIED"
            row["action_required"] = ""
        return row

    if sleeve_id == "hostile_combo_long":
        row.update({"contract_type": "research_hostile_opportunity_sleeve"})
        row.update(_ledger_stats(out_dir, sleeve_id))
        if account_row.empty:
            row["audit_status"] = "FAIL_MISSING_PAPER_ACCOUNT"
            row["action_required"] = "Rebuild paper PnL before optimizing."
        else:
            row["audit_status"] = "RESEARCH_ONLY"
            row["action_required"] = "Do not treat as a verified core sleeve; monitor paper behavior separately."
        return row

    row.update({"contract_type": "legacy_flow_v2_dashboard_overlay"})
    row.update(_ledger_stats(out_dir, sleeve_id))
    if account_row.empty:
        row["audit_status"] = "FAIL_MISSING_PAPER_ACCOUNT"
        row["action_required"] = "Rebuild paper PnL before optimizing."
    else:
        row["audit_status"] = "NEEDS_EXACT_RERUN"
        row["action_required"] = "Legacy flow_v2 has a separate paper exit contract; rerun exact contract."
    return row


def build_audit(out_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    account = _read_csv(out_dir / "00_account_overview.csv")
    if account.empty:
        account = _read_csv(out_dir / "paper_account_pnl.csv")
    comparison = _read_csv(out_dir / "13_backtest_comparison.csv")
    if comparison.empty:
        comparison = _read_csv(out_dir / "strategy_backtest_comparison.csv")

    rows = [_audit_row(out_dir, sleeve_id, account, comparison) for sleeve_id in ACTIVE_SLEEVES]
    audit = pd.DataFrame(rows)

    statuses = audit["audit_status"].astype(str) if "audit_status" in audit else pd.Series(dtype=str)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": str(out_dir),
        "verified_count": int((statuses == "VERIFIED").sum()),
        "needs_exact_rerun_count": int((statuses == "NEEDS_EXACT_RERUN").sum()),
        "fail_count": int(statuses.str.startswith("FAIL_").sum()),
        "active_sleeves": ACTIVE_SLEEVES,
        "ready_for_strategy_optimization": bool(statuses.isin({"VERIFIED", "RESEARCH_ONLY"}).all()),
    }
    return audit, summary


def write_audit(out_dir: Path) -> tuple[Path, Path, pd.DataFrame, dict[str, Any]]:
    audit, summary = build_audit(out_dir)
    csv_path = out_dir / "19_current_paper_contract_audit.csv"
    json_path = out_dir / "current_paper_contract_audit.json"
    audit.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps({"summary": summary, "rows": audit.to_dict("records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path, audit, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit combined paper contract verification status.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--fail-on-unverified", action="store_true")
    args = parser.parse_args()

    csv_path, json_path, _audit, summary = write_audit(Path(args.out_dir))
    print(csv_path)
    print(json_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_unverified and not summary["ready_for_strategy_optimization"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
