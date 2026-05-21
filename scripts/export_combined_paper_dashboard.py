"""Export one HTML dashboard for independent paper strategy sleeves."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from scripts.inject_sortable_tables import SORTABLE_JS


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:,.4g}"
    return str(value)


def _to_number(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace("%", "").replace("+", "").replace("VND", "").replace(" ", "")
    if "." in cleaned and "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_vn_number(value: float, decimals: int = 0, signed: bool = False, suffix: str = "") -> str:
    sign = "+" if signed and value > 0 else ""
    if decimals <= 0:
        text = f"{abs(value):,.0f}".replace(",", ".")
        prefix = "-" if value < 0 else sign
        return f"{prefix}{text}{suffix}"
    text = f"{abs(value):,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    prefix = "-" if value < 0 else sign
    return f"{prefix}{text}{suffix}"


def _column_kind(column: str) -> str:
    name = column.lower()
    if name in {"sleeve_id", "symbol", "strategy_name", "strategy", "label", "logic", "mode", "status"}:
        return "text"
    if "date" in name:
        return "date"
    if name in {"shares", "open_positions", "closed_trades", "candidate_count", "target_count", "max_positions", "days_held", "priority", "rank"}:
        return "integer"
    if "pct" in name or "return" in name or "win_rate" in name:
        return "percent"
    if "weight" in name:
        return "weight"
    if any(token in name for token in ["pnl", "equity", "cash", "value", "price", "capital", "fees", "cost"]):
        return "money"
    if "score" in name:
        return "score"
    return "default"


def _format_cell(column: str, value: Any) -> str:
    number = _to_number(value)
    kind = _column_kind(column)
    if number is None:
        return "" if value is None else str(value)
    if kind in {"money", "integer"}:
        return _format_vn_number(number, decimals=0)
    if kind == "percent":
        return _format_vn_number(number, decimals=2, signed=True, suffix="%")
    if kind == "weight":
        pct = number * 100.0 if abs(number) <= 1.0 else number
        return _format_vn_number(pct, decimals=2, signed=False, suffix="%")
    if kind == "score":
        return _format_vn_number(number, decimals=2)
    return _format_vn_number(number, decimals=2) if not float(number).is_integer() else _format_vn_number(number)


def _cell_class(column: str, value: Any) -> str:
    name = column.lower()
    text = str(value or "").upper()
    number = _to_number(value)
    if text in {"RISK_ON", "HOLD_STRONG"}:
        return "win"
    if text in {"RISK_OFF", "WATCH_WEAKENING"}:
        return "loss"
    if text in {"NEUTRAL", "UNKNOWN"}:
        return "flat"
    if number is None:
        return ""
    if any(token in name for token in ["pnl", "return", "pct"]):
        if number > 0:
            return "win"
        if number < 0:
            return "loss"
        return "flat"
    return ""


def _frame_to_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<table class="data-table"><thead><tr><th>status</th></tr></thead><tbody><tr><td>No rows</td></tr></tbody></table>'
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    body_rows: list[str] = []
    for row in frame.to_dict("records"):
        cells = []
        for column in frame.columns:
            raw = row.get(column)
            klass = _cell_class(str(column), raw)
            class_attr = f' class="{klass}"' if klass else ""
            cells.append(f"<td{class_attr}>{html.escape(_format_cell(str(column), raw))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<table class="data-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def _write_table_page(path: Path, title: str, frame: pd.DataFrame, note: str) -> None:
    display = frame.copy()
    if len(display) > 300:
        display = display.head(300)
        note = f"{note} Showing first 300 rows only; CSV contains full export.".strip()
    table = _frame_to_html(display)
    content = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ --ink: #14213d; --muted: #5f6b7a; --line: #d9dee7; --bg: #f5f1e8; --panel: #fffaf0; }}
    body {{ margin: 0; padding: 28px; font-family: Georgia, 'Times New Roman', serif; background: var(--bg); color: var(--ink); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .note {{ margin: 0 0 18px; color: var(--muted); font-family: Segoe UI, Arial, sans-serif; }}
    .wrap {{ overflow: auto; background: white; border: 1px solid var(--line); box-shadow: 0 14px 36px rgba(20,33,61,.12); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; white-space: nowrap; font-family: Segoe UI, Arial, sans-serif; }}
    th {{ position: sticky; top: 0; background: var(--ink); color: white; text-align: left; cursor: pointer; user-select: none; }}
    th::after {{ content: " <>"; color: rgba(255,255,255,.55); font-weight: 400; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf0f5; }}
    tr:nth-child(even) {{ background: #fbfbf8; }}
    .win {{ color: #0f8b4c; font-weight: 700; }}
    .loss {{ color: #c63d2f; font-weight: 700; }}
    .flat {{ color: var(--muted); font-weight: 700; }}
    a {{ color: var(--ink); }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="note"><a href="index.html">Index</a> | {html.escape(note)}</p>
  <div class="wrap">{table}</div>
  <script src="sortable_tables.js"></script>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def _open_pnl_summary(pnl_overview: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "open_positions",
        "open_cost_value",
        "open_market_value",
        "unrealized_pnl",
        "unrealized_pnl_pct",
        "paper_equity",
        "paper_cash",
    ]
    if pnl_overview.empty:
        return pd.DataFrame(columns=columns)

    numeric_cols = [
        "open_positions",
        "open_cost_value",
        "open_market_value",
        "unrealized_pnl",
        "paper_equity",
        "paper_cash",
    ]
    work = pnl_overview.copy()
    for column in numeric_cols:
        if column in work:
            work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
        else:
            work[column] = 0.0

    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        cost = float(row.get("open_cost_value", 0.0) or 0.0)
        pnl = float(row.get("unrealized_pnl", 0.0) or 0.0)
        rows.append(
            {
                "scope": row.get("sleeve_id"),
                "open_positions": row.get("open_positions", 0),
                "open_cost_value": cost,
                "open_market_value": row.get("open_market_value", 0.0),
                "unrealized_pnl": pnl,
                "unrealized_pnl_pct": (pnl / cost * 100.0) if cost else 0.0,
                "paper_equity": row.get("paper_equity", 0.0),
                "paper_cash": row.get("paper_cash", 0.0),
            }
        )

    total_cost = float(work["open_cost_value"].sum())
    total_pnl = float(work["unrealized_pnl"].sum())
    rows.insert(
        0,
        {
            "scope": "TOTAL_OPEN_HOLDINGS",
            "open_positions": int(work["open_positions"].sum()),
            "open_cost_value": total_cost,
            "open_market_value": float(work["open_market_value"].sum()),
            "unrealized_pnl": total_pnl,
            "unrealized_pnl_pct": (total_pnl / total_cost * 100.0) if total_cost else 0.0,
            "paper_equity": float(work["paper_equity"].sum()),
            "paper_cash": float(work["paper_cash"].sum()),
        },
    )
    return pd.DataFrame(rows, columns=columns)


def _mvp_summary(sleeve_id: str, label: str, path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    status = _read_json(path / "production_status.json")
    market = _read_json(path / "market_snapshot.json")
    actions = _read_csv(path / "candidate_actions.csv")
    signals = _read_csv(path / "production_signals.csv")
    buys = actions[actions.get("action", pd.Series(dtype=str)).astype(str).str.startswith("PAPER_BUY_CANDIDATE")] if not actions.empty and "action" in actions else pd.DataFrame()
    row = {
        "sleeve_id": sleeve_id,
        "label": label,
        "logic": "Core MVP shared-pool",
        "as_of": status.get("as_of_date"),
        "mode": status.get("mode"),
        "market_state": market.get("mkt_regime_state"),
        "market_score": market.get("mkt_regime_score"),
        "max_positions": (status.get("active_sleeve") or {}).get("max_positions"),
        "candidate_count": status.get("candidate_count"),
        "target_count": len(buys),
        "target_symbols": ",".join(buys.get("symbol", pd.Series(dtype=str)).dropna().astype(str).head(20).tolist()),
        "dashboard_dir": str(path),
    }
    export = actions.copy()
    if export.empty:
        export = signals.copy()
    if not export.empty:
        export.insert(0, "sleeve_id", sleeve_id)
    return row, export


def _flow_summary(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    status = _read_json(path / "flow_v2_status.json")
    market = _read_json(path / "market_snapshot.json")
    targets = _read_csv(path / "flow_v2_target_plan.csv")
    candidates = _read_csv(path / "flow_v2_candidates.csv")
    row = {
        "sleeve_id": "flow_v2",
        "label": "Flow V2 Rotation",
        "logic": "Flow Money V2 clean-flow rotation",
        "as_of": status.get("as_of_date"),
        "mode": status.get("mode"),
        "market_state": market.get("mkt_regime_state"),
        "market_score": market.get("mkt_regime_score"),
        "max_positions": status.get("positions"),
        "candidate_count": status.get("candidate_count"),
        "target_count": status.get("target_count"),
        "target_symbols": ",".join(status.get("target_symbols") or []),
        "dashboard_dir": str(path),
    }
    export = targets.copy()
    if export.empty:
        export = candidates.copy()
    if not export.empty:
        export.insert(0, "sleeve_id", "flow_v2")
    return row, export


def _flatten_json_rows(sleeve_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        rows.append({"sleeve_id": sleeve_id, "key": key, "value": value})
    return rows


def _empty_open_holdings() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sleeve_id",
            "symbol",
            "strategy_name",
            "entry_date",
            "entry_price",
            "shares",
            "cost_value",
            "market_price",
            "market_value",
            "unrealized_pnl",
            "unrealized_pnl_pct",
            "days_held",
            "stop_loss",
            "take_profit",
            "status",
        ]
    )


def _empty_closed_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sleeve_id",
            "symbol",
            "strategy_name",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "shares",
            "gross_pnl",
            "net_pnl",
            "pnl_pct",
            "days_held",
            "exit_reason",
        ]
    )


def _load_mvp_tables(sleeve_id: str, path: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "actions": _read_csv(path / "candidate_actions.csv"),
        "signals": _read_csv(path / "production_signals.csv"),
        "votes": _read_csv(path / "strategy_votes.csv"),
        "rolling_summary": _read_csv(path / "rolling_summary_14d.csv"),
        "status": pd.DataFrame(_flatten_json_rows(sleeve_id, _read_json(path / "production_status.json"))),
        "market": pd.DataFrame(_flatten_json_rows(sleeve_id, _read_json(path / "market_snapshot.json"))),
    }
    for frame in tables.values():
        if not frame.empty and "sleeve_id" not in frame.columns:
            frame.insert(0, "sleeve_id", sleeve_id)
    return tables


def _load_flow_tables(path: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "actions": _read_csv(path / "flow_v2_target_plan.csv"),
        "signals": _read_csv(path / "flow_v2_candidates.csv"),
        "votes": pd.DataFrame(),
        "rolling_summary": pd.DataFrame(),
        "status": pd.DataFrame(_flatten_json_rows("flow_v2", _read_json(path / "flow_v2_status.json"))),
        "market": pd.DataFrame(_flatten_json_rows("flow_v2", _read_json(path / "market_snapshot.json"))),
    }
    for frame in tables.values():
        if not frame.empty and "sleeve_id" not in frame.columns:
            frame.insert(0, "sleeve_id", "flow_v2")
    return tables


def export_dashboard(out_dir: Path, mvp_p5_dir: Path, mvp_p4_dir: Path, flow_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for row, frame in [
        _mvp_summary("mvp_p5", "MVP Original p5/c10", mvp_p5_dir),
        _mvp_summary("mvp_p4", "MVP Improved p4/c10", mvp_p4_dir),
        _flow_summary(flow_dir),
    ]:
        rows.append(row)
        if not frame.empty:
            detail_frames.append(frame)

    summary = pd.DataFrame(rows)
    details = pd.concat(detail_frames, ignore_index=True, sort=False) if detail_frames else pd.DataFrame()
    summary.to_csv(out_dir / "combined_sleeve_summary.csv", index=False, encoding="utf-8-sig")
    details.to_csv(out_dir / "combined_actions.csv", index=False, encoding="utf-8-sig")

    mvp_p5_tables = _load_mvp_tables("mvp_p5", mvp_p5_dir)
    mvp_p4_tables = _load_mvp_tables("mvp_p4", mvp_p4_dir)
    flow_tables = _load_flow_tables(flow_dir)
    all_tables = [mvp_p5_tables, mvp_p4_tables, flow_tables]

    def concat_table(name: str) -> pd.DataFrame:
        frames = [tables[name] for tables in all_tables if not tables[name].empty]
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    realtime_status = _read_json(out_dir / "paper_realtime_status.json")
    pnl_overview = _read_csv(out_dir / "paper_realtime_account_pnl.csv")
    pnl_open_holdings = _read_csv(out_dir / "paper_realtime_open_holdings.csv")
    strategy_pnl = _read_csv(out_dir / "paper_realtime_strategy_pnl.csv")
    using_realtime_overlay = not pnl_overview.empty or not pnl_open_holdings.empty or not strategy_pnl.empty
    if pnl_overview.empty:
        pnl_overview = _read_csv(out_dir / "paper_account_pnl.csv")
    if pnl_open_holdings.empty:
        pnl_open_holdings = _read_csv(out_dir / "paper_open_holdings.csv")
    if strategy_pnl.empty:
        strategy_pnl = _read_csv(out_dir / "paper_strategy_pnl.csv")
    pnl_closed_trades = _read_csv(out_dir / "paper_closed_trades.csv")
    ledger_events = _read_csv(out_dir / "paper_ledger_events.csv")
    open_pnl_summary = _open_pnl_summary(pnl_overview)
    open_holdings = pnl_open_holdings if not pnl_open_holdings.empty else _empty_open_holdings()
    closed_trades = pnl_closed_trades if not pnl_closed_trades.empty else _empty_closed_trades()
    orders_targets = concat_table("actions")
    candidates = concat_table("signals")
    votes = concat_table("votes")
    status_rows = pd.concat(
        [concat_table("status"), concat_table("market")],
        ignore_index=True,
        sort=False,
    )
    rolling_summary = concat_table("rolling_summary")
    account_overview = summary[
        [
            "sleeve_id",
            "label",
            "logic",
            "as_of",
            "mode",
            "market_state",
            "market_score",
            "max_positions",
            "candidate_count",
            "target_count",
            "target_symbols",
        ]
    ].copy()
    if not pnl_overview.empty:
        account_overview = account_overview.merge(pnl_overview, on="sleeve_id", how="left")
        if using_realtime_overlay:
            account_overview["note"] = f"Realtime price overlay at {realtime_status.get('priced_at', '-')}; ledger exits still run on daily OHLCV."
        else:
            account_overview["note"] = "PnL uses paper ledger: signal close T, fill next open T+1, mark latest close."
    else:
        account_overview["paper_equity"] = None
        account_overview["paper_cash"] = None
        account_overview["open_positions"] = 0
        account_overview["closed_trades"] = 0
        account_overview["realized_pnl"] = 0.0
        account_overview["unrealized_pnl"] = 0.0
        account_overview["note"] = "Dry-run signals only until first paper fill is recorded."

    account_overview.to_csv(out_dir / "00_account_overview.csv", index=False, encoding="utf-8-sig")
    open_holdings.to_csv(out_dir / "03_open_holdings.csv", index=False, encoding="utf-8-sig")
    closed_trades.to_csv(out_dir / "04_closed_trades.csv", index=False, encoding="utf-8-sig")
    orders_targets.to_csv(out_dir / "05_orders_targets.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(out_dir / "06_candidates.csv", index=False, encoding="utf-8-sig")
    votes.to_csv(out_dir / "07_strategy_votes.csv", index=False, encoding="utf-8-sig")
    status_rows.to_csv(out_dir / "08_status_config.csv", index=False, encoding="utf-8-sig")
    rolling_summary.to_csv(out_dir / "09_rolling_summary.csv", index=False, encoding="utf-8-sig")
    strategy_pnl.to_csv(out_dir / "10_strategy_pnl.csv", index=False, encoding="utf-8-sig")
    open_pnl_summary.to_csv(out_dir / "11_open_pnl_summary.csv", index=False, encoding="utf-8-sig")
    ledger_events.to_csv(out_dir / "12_ledger_events.csv", index=False, encoding="utf-8-sig")

    _write_table_page(out_dir / "00_account_overview.html", "00 Account Overview", account_overview, "Paper account state by sleeve.")
    _write_table_page(out_dir / "01_sleeve_summary.html", "01 Sleeve Summary", summary, "Independent paper strategy sleeves.")
    _write_table_page(out_dir / "02_combined_actions.html", "02 Combined Actions", details, "Paper actions/candidates from all sleeves.")
    _write_table_page(out_dir / "03_open_holdings.html", "03 Open Holdings", open_holdings, "Open paper positions by sleeve. Empty until first fill.")
    _write_table_page(out_dir / "04_closed_trades.html", "04 Closed Trades", closed_trades, "Closed paper trades by sleeve. Empty until first exit.")
    _write_table_page(out_dir / "05_orders_targets.html", "05 Orders & Targets", orders_targets, "Paper buy targets/orders emitted by each sleeve.")
    _write_table_page(out_dir / "06_candidates.html", "06 Candidates", candidates, "Ranked candidates/signals before portfolio allocation.")
    _write_table_page(out_dir / "07_strategy_votes.html", "07 Strategy Votes", votes, "MVP strategy-level votes. Flow V2 uses a single ranked score.")
    _write_table_page(out_dir / "08_status_config.html", "08 Status & Config", status_rows, "Run status, market snapshot and safety controls.")
    _write_table_page(out_dir / "09_rolling_summary.html", "09 Rolling Summary", rolling_summary, "Recent MVP signal history when available.")
    _write_table_page(out_dir / "10_strategy_pnl.html", "10 Strategy PnL", strategy_pnl, "PnL grouped by sleeve and strategy.")
    _write_table_page(out_dir / "11_open_pnl_summary.html", "11 Open PnL Summary", open_pnl_summary, "Current unrealized PnL for all open paper holdings.")
    _write_table_page(out_dir / "12_ledger_events.html", "12 Ledger Events", ledger_events, "Audit log for paper entries, stop updates and exits.")
    (out_dir / "sortable_tables.js").write_text(SORTABLE_JS, encoding="utf-8")

    pnl_by_sleeve = {}
    if not pnl_overview.empty:
        pnl_by_sleeve = {str(row.get("sleeve_id")): row for row in pnl_overview.to_dict("records")}
    total_open_pnl = open_pnl_summary.iloc[0].to_dict() if not open_pnl_summary.empty else {}
    total_pnl_value = total_open_pnl.get("unrealized_pnl", 0.0)
    total_pnl_pct = total_open_pnl.get("unrealized_pnl_pct", 0.0)
    total_pnl_class = _cell_class("unrealized_pnl", total_pnl_value)

    cards = []
    for row in rows:
        state = str(row.get("market_state") or "UNKNOWN")
        state_class = "win" if state == "RISK_ON" else ("loss" if state == "RISK_OFF" else "flat")
        symbols = row.get("target_symbols") or "Cash"
        sleeve_pnl = pnl_by_sleeve.get(str(row.get("sleeve_id")), {})
        sleeve_pnl_value = sleeve_pnl.get("unrealized_pnl", 0.0)
        sleeve_pnl_pct = sleeve_pnl.get("unrealized_pnl_pct", 0.0)
        sleeve_pnl_class = _cell_class("unrealized_pnl", sleeve_pnl_value)
        cards.append(
            f"""<div class="sleeve">
  <div class="sleeve-top"><strong>{html.escape(str(row["label"]))}</strong><span>{html.escape(str(row["logic"]))}</span></div>
  <div class="metric-row"><span>Market</span><b class="{state_class}">{html.escape(state)}</b></div>
  <div class="metric-row"><span>Open PnL</span><b class="{sleeve_pnl_class}">{html.escape(_format_cell("unrealized_pnl", sleeve_pnl_value))} ({html.escape(_format_cell("unrealized_pnl_pct", sleeve_pnl_pct))})</b></div>
  <div class="metric-row"><span>Score</span><b>{html.escape(_fmt(row.get("market_score")))}</b></div>
  <div class="metric-row"><span>Max positions</span><b>{html.escape(_fmt(row.get("max_positions")))}</b></div>
  <div class="metric-row"><span>Candidates</span><b>{html.escape(_fmt(row.get("candidate_count")))}</b></div>
  <div class="metric-row"><span>Targets</span><b>{html.escape(_fmt(symbols))}</b></div>
</div>"""
        )

    generated_at = max([str(item.get("as_of") or "") for item in rows] or [""])
    price_line = ""
    if using_realtime_overlay:
        price_line = (
            f" | Realtime prices: {html.escape(str(realtime_status.get('updated_holdings', realtime_status.get('updated', 0))))} holdings / "
            f"{html.escape(str(realtime_status.get('symbols', 0)))} symbols at "
            f"{html.escape(str(realtime_status.get('priced_at', '-')))}"
        )
    refresh_seconds = int(_to_number(realtime_status.get("html_refresh_seconds")) or 0) if using_realtime_overlay else 0
    refresh_meta = f'\n  <meta http-equiv="refresh" content="{refresh_seconds}">' if refresh_seconds > 0 else ""
    index = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  {refresh_meta}
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Combined Paper Trading Dashboard</title>
  <style>
    :root {{ --ink: #10243e; --gold: #b8872f; --paper: #f5f1e8; --card: #fffaf0; --muted: #667085; --win: #0f8b4c; --loss: #c63d2f; }}
    body {{ margin: 0; min-height: 100vh; background: radial-gradient(circle at top left, rgba(184,135,47,.24), transparent 32rem), linear-gradient(135deg, #f7f0df 0%, #eef3f6 100%); color: var(--ink); font-family: Georgia, 'Times New Roman', serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 44px 26px; }}
    h1 {{ font-size: 42px; margin: 0 0 8px; }}
    .sub {{ color: var(--muted); font-family: Segoe UI, Arial, sans-serif; font-size: 16px; margin-bottom: 24px; }}
    .flow {{ padding: 18px 20px; background: rgba(255,255,255,.68); border: 1px solid rgba(16,36,62,.12); border-radius: 18px; font-family: Consolas, monospace; line-height: 1.55; white-space: pre-wrap; box-shadow: 0 18px 48px rgba(16,36,62,.12); margin-bottom: 24px; }}
    .pnl-total {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .pnl-box {{ background: rgba(255,255,255,.76); border: 1px solid rgba(16,36,62,.14); border-radius: 8px; padding: 16px; font-family: Segoe UI, Arial, sans-serif; }}
    .pnl-box span {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 7px; }}
    .pnl-box strong {{ display: block; font-size: 24px; overflow-wrap: anywhere; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 14px; margin-bottom: 18px; }}
    .sleeve {{ background: var(--card); border: 1px solid rgba(16,36,62,.14); border-radius: 8px; padding: 18px; box-shadow: 0 10px 26px rgba(16,36,62,.08); }}
    .sleeve-top strong {{ display: block; font-size: 19px; margin-bottom: 5px; }}
    .sleeve-top span {{ color: var(--muted); font-family: Segoe UI, Arial, sans-serif; font-size: 13px; }}
    .metric-row {{ display: flex; justify-content: space-between; gap: 12px; padding: 9px 0; border-bottom: 1px solid rgba(16,36,62,.1); font-family: Segoe UI, Arial, sans-serif; }}
    .metric-row span {{ color: var(--muted); }}
    .metric-row b {{ text-align: right; overflow-wrap: anywhere; }}
    .win {{ color: var(--win); }}
    .loss {{ color: var(--loss); }}
    .flat {{ color: var(--muted); }}
    .links {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }}
    .card {{ display: block; text-decoration: none; color: var(--ink); background: rgba(255,255,255,.72); border: 1px solid rgba(16,36,62,.14); border-radius: 8px; padding: 16px; font-family: Segoe UI, Arial, sans-serif; }}
    .card strong {{ display: block; margin-bottom: 5px; }}
    .card span {{ color: var(--muted); font-size: 13px; }}
  </style>
</head>
<body>
  <main>
    <h1>Combined Paper Trading Dashboard</h1>
    <div class="sub">Three independent sleeves | As-of: {html.escape(generated_at)} | Broker orders disabled{price_line}</div>
    <div class="flow">OHLCV parquet + feature engine
        |
        +-- MVP Original p5/c10: Core MVP shared pool, max 5 slots
        |
        +-- MVP Improved p4/c10: same Core MVP logic, max 4 slots
        |
        +-- Flow V2 Rotation: clean-flow sector-heavy rotation, max 2 slots
        |
Combined paper dashboard only. No sleeve can overwrite another sleeve's logic.</div>
    <div class="pnl-total">
      <div class="pnl-box"><span>Total Open PnL</span><strong class="{total_pnl_class}">{html.escape(_format_cell("unrealized_pnl", total_pnl_value))}</strong></div>
      <div class="pnl-box"><span>Total Open PnL %</span><strong class="{total_pnl_class}">{html.escape(_format_cell("unrealized_pnl_pct", total_pnl_pct))}</strong></div>
      <div class="pnl-box"><span>Open Market Value</span><strong>{html.escape(_format_cell("open_market_value", total_open_pnl.get("open_market_value", 0.0)))}</strong></div>
      <div class="pnl-box"><span>Open Positions</span><strong>{html.escape(_format_cell("open_positions", total_open_pnl.get("open_positions", 0)))}</strong></div>
    </div>
    <div class="grid">{''.join(cards)}</div>
    <div class="links">
      <a class="card" href="00_account_overview.html"><strong>00 Account Overview</strong><span>Portfolio state, cash/equity placeholders and sleeve status.</span></a>
      <a class="card" href="01_sleeve_summary.html"><strong>01 Sleeve Summary</strong><span>High-level state for all paper sleeves.</span></a>
      <a class="card" href="02_combined_actions.html"><strong>02 Combined Actions</strong><span>Actions and candidates emitted by each independent sleeve.</span></a>
      <a class="card" href="03_open_holdings.html"><strong>03 Open Holdings</strong><span>Open paper positions by sleeve.</span></a>
      <a class="card" href="04_closed_trades.html"><strong>04 Closed Trades</strong><span>Closed paper trades and realized PnL.</span></a>
      <a class="card" href="05_orders_targets.html"><strong>05 Orders & Targets</strong><span>Paper targets for next session.</span></a>
      <a class="card" href="06_candidates.html"><strong>06 Candidates</strong><span>Full candidate list and scores.</span></a>
      <a class="card" href="07_strategy_votes.html"><strong>07 Strategy Votes</strong><span>Per-strategy MVP vote transparency.</span></a>
      <a class="card" href="08_status_config.html"><strong>08 Status & Config</strong><span>Market snapshot, config and safety notes.</span></a>
      <a class="card" href="09_rolling_summary.html"><strong>09 Rolling Summary</strong><span>Recent signal counts and market states.</span></a>
      <a class="card" href="10_strategy_pnl.html"><strong>10 Strategy PnL</strong><span>Per-strategy paper PnL by sleeve.</span></a>
      <a class="card" href="11_open_pnl_summary.html"><strong>11 Open PnL Summary</strong><span>Total and per-sleeve unrealized PnL for open holdings.</span></a>
      <a class="card" href="12_ledger_events.html"><strong>12 Ledger Events</strong><span>Audit log for entries, stop updates and exits.</span></a>
    </div>
  </main>
</body>
</html>
"""
    index_path = out_dir / "index.html"
    index_path.write_text(index, encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export combined paper trading dashboard.")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "combined_paper_trading_demo"))
    parser.add_argument("--mvp-p5-dir", required=True)
    parser.add_argument("--mvp-p4-dir", required=True)
    parser.add_argument("--flow-dir", required=True)
    args = parser.parse_args()
    path = export_dashboard(
        Path(args.out_dir),
        Path(args.mvp_p5_dir),
        Path(args.mvp_p4_dir),
        Path(args.flow_dir),
    )
    print(path)


if __name__ == "__main__":
    main()
