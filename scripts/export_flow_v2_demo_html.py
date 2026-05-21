"""Export Flow V2 production dry-run files to a clickable HTML dashboard."""

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


def _write_table_page(path: Path, title: str, frame: pd.DataFrame, note: str) -> None:
    display = frame.copy()
    if len(display) > 300:
        display = display.head(300)
        note = f"{note} Showing first 300 rows only; CSV contains full export.".strip()
    table = display.to_html(index=False, classes="data-table", border=0, escape=True)
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


def _metric(label: str, value: Any, klass: str = "") -> str:
    class_attr = f' class="{klass}"' if klass else ""
    return f'<div class="metric"><span>{html.escape(label)}</span><strong{class_attr}>{html.escape(_fmt(value))}</strong></div>'


def _card(href: str, title: str, note: str) -> str:
    return f"""<a class="card" href="{html.escape(href)}">
  <strong>{html.escape(title)}</strong>
  <span>{html.escape(note)}</span>
</a>"""


def _status_class(market_state: str) -> str:
    state = market_state.upper()
    if state == "RISK_ON":
        return "win"
    if state == "RISK_OFF":
        return "loss"
    return "flat"


def export_html(demo_dir: Path) -> Path:
    status = _read_json(demo_dir / "flow_v2_status.json")
    market = _read_json(demo_dir / "market_snapshot.json")
    candidates = _read_csv(demo_dir / "flow_v2_candidates.csv")
    targets = _read_csv(demo_dir / "flow_v2_target_plan.csv")

    market_rows = pd.DataFrame([{"key": key, "value": value} for key, value in market.items()])
    status_rows = pd.DataFrame(
        [
            {"key": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value}
            for key, value in status.items()
        ]
    )

    _write_table_page(demo_dir / "01_market_snapshot.html", "01 Market Snapshot", market_rows, "Market regime and money-cycle context.")
    _write_table_page(demo_dir / "02_flow_v2_candidates.html", "02 Flow V2 Candidates", candidates, "Ranked eligible pool after Flow V2 gates.")
    _write_table_page(demo_dir / "03_target_plan.html", "03 Paper Target Plan", targets, "Paper target allocation for next session.")
    _write_table_page(demo_dir / "04_status.html", "04 Production Status", status_rows, "Dry-run status and safety controls.")
    (demo_dir / "sortable_tables.js").write_text(SORTABLE_JS, encoding="utf-8")

    state = str(market.get("mkt_regime_state") or "UNKNOWN")
    target_symbols = status.get("target_symbols") or []
    if target_symbols:
        action_line = "Flow V2 target is ready for next session paper execution."
        action_class = "win"
    else:
        action_line = "No paper target today; system stays in cash/watch mode."
        action_class = "loss" if state.upper() == "RISK_OFF" else "flat"

    flow = """OHLCV parquet + VNINDEX
        |
Flow Money V2 feature engine
Sector cycle / sponsorship / absorption / distribution
        |
Market gate
Risk-on or strong neutral only
        |
Clean-flow stock filter
MA50 / sponsorship / absorption / RS / value
        |
Ranking engine
sector_heavy score
        |
Paper target plan
Top 2 / 50-50 weight / next-session execution
        |
Production demo
No broker order, manual approval required"""

    metrics = "\n".join(
        [
            _metric("As-of", status.get("as_of_date")),
            _metric("Mode", status.get("mode")),
            _metric("Market", state, _status_class(state)),
            _metric("Regime score", market.get("mkt_regime_score")),
            _metric("CHDM20", market.get("mkt_CHDM20")),
            _metric("DS20", market.get("mkt_DS20")),
            _metric("Candidates", status.get("candidate_count")),
            _metric("Targets", ", ".join(target_symbols) if target_symbols else "Cash"),
        ]
    )
    cards = "\n".join(
        [
            _card("01_market_snapshot.html", "01 Market Snapshot", "Regime, CHDM and distribution context."),
            _card("02_flow_v2_candidates.html", "02 Flow V2 Candidates", "Ranked symbols passing market and stock gates."),
            _card("03_target_plan.html", "03 Paper Target Plan", "Indicative allocation for next session."),
            _card("04_status.html", "04 Production Status", "Config, safety notes and dry-run metadata."),
        ]
    )

    index = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flow V2 Production Demo</title>
  <style>
    :root {{ --ink: #10243e; --gold: #b8872f; --paper: #f5f1e8; --card: #fffaf0; --muted: #667085; --win: #0f8b4c; --loss: #c63d2f; }}
    body {{ margin: 0; min-height: 100vh; background: radial-gradient(circle at top left, rgba(184,135,47,.24), transparent 32rem), linear-gradient(135deg, #f7f0df 0%, #eef3f6 100%); color: var(--ink); font-family: Georgia, 'Times New Roman', serif; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 44px 26px; }}
    h1 {{ font-size: 42px; margin: 0 0 8px; }}
    .sub {{ color: var(--muted); font-family: Segoe UI, Arial, sans-serif; font-size: 16px; margin-bottom: 28px; }}
    .banner {{ margin: 0 0 22px; padding: 14px 16px; border: 1px solid rgba(16,36,62,.12); border-radius: 8px; background: rgba(255,255,255,.72); font-family: Segoe UI, Arial, sans-serif; }}
    .banner strong {{ display: block; margin-bottom: 4px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin: 0 0 22px; font-family: Segoe UI, Arial, sans-serif; }}
    .metric {{ background: rgba(255,255,255,.72); border: 1px solid rgba(16,36,62,.12); border-radius: 8px; padding: 10px; min-width: 0; }}
    .metric span {{ display: block; color: var(--muted); font-size: 13px; }}
    .metric b, .metric strong {{ display: block; margin-top: 4px; font-size: 17px; overflow-wrap: anywhere; }}
    .win {{ color: var(--win); }}
    .loss {{ color: var(--loss); }}
    .flat {{ color: var(--muted); }}
    .flow {{ padding: 18px 20px; background: rgba(255,255,255,.68); border: 1px solid rgba(16,36,62,.12); border-radius: 18px; font-family: Consolas, monospace; line-height: 1.55; white-space: pre-wrap; box-shadow: 0 18px 48px rgba(16,36,62,.12); margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; }}
    .card {{ display: block; text-decoration: none; color: var(--ink); background: var(--card); border: 1px solid rgba(16,36,62,.14); border-radius: 8px; padding: 18px; box-shadow: 0 10px 26px rgba(16,36,62,.08); }}
    .card strong {{ display: block; font-size: 18px; margin-bottom: 7px; }}
    .card span {{ color: var(--muted); font-family: Segoe UI, Arial, sans-serif; font-size: 14px; }}
  </style>
</head>
<body>
  <main>
    <h1>Flow V2 Production Demo</h1>
    <div class="sub">Paper dry-run | Updated: {html.escape(str(status.get("generated_at") or ""))} | Universe: {html.escape(str(status.get("universe") or ""))}</div>
    <div class="banner"><strong class="{html.escape(action_class)}">{html.escape(action_line)}</strong><span>Signals use local data through close T. Any real fill belongs to next session after manual approval.</span></div>
    <div class="stats">{metrics}</div>
    <div class="flow">{html.escape(flow)}</div>
    <div class="grid">{cards}</div>
  </main>
</body>
</html>
"""
    index_path = demo_dir / "index.html"
    index_path.write_text(index, encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Flow V2 production demo HTML.")
    parser.add_argument("--demo-dir", required=True)
    args = parser.parse_args()
    path = export_html(Path(args.demo_dir))
    print(path)


if __name__ == "__main__":
    main()
