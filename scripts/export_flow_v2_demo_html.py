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


def _pulse_class(value: str) -> str:
    text = str(value).upper()
    if text in {"DONG_TIEN_DANG_CAI_THIEN", "DONG_TIEN_VAO_MANH", "DONG_TIEN_VAO", "DUY_TRI_TIEN_VAO", "DANG_TANG_TOC"}:
        return "win"
    if text in {"REGIME_CHAN_GIAI_NGAN", "AP_LUC_PHAN_PHOI_TANG", "AP_LUC_PHAN_PHOI", "DANG_GIAM_NHIET", "ROI_NHOM_TIEN_VAO"}:
        return "loss"
    return "flat"


def _pulse_label(value: Any) -> str:
    labels = {
        "REGIME_CHAN_GIAI_NGAN": "Regime chặn giải ngân",
        "DONG_TIEN_DANG_CAI_THIEN": "Dòng tiền đang cải thiện",
        "AP_LUC_PHAN_PHOI_TANG": "Áp lực phân phối tăng",
        "TRUNG_LAP_THEO_DOI": "Trung lập / theo dõi",
        "DONG_TIEN_VAO_MANH": "Vào mạnh",
        "DONG_TIEN_VAO": "Tiền vào",
        "AP_LUC_PHAN_PHOI": "Phân phối",
        "THEO_DOI": "Theo dõi",
        "DUY_TRI_TIEN_VAO": "Duy trì vào",
        "DANG_TANG_TOC": "Tăng tốc",
        "DANG_GIAM_NHIET": "Giảm nhiệt",
        "ROI_NHOM_TIEN_VAO": "Rời nhóm tiền vào",
        "DI_NGANG": "Đi ngang",
    }
    text = str(value or "")
    return labels.get(text, text or "-")


def _flow_bars(rows: pd.DataFrame, *, recent: bool = False) -> str:
    bars: list[str] = []
    score_key = "flow_score_avg_5d" if recent else "flow_score_today"
    for row in rows.head(10).to_dict("records"):
        symbol = html.escape(str(row.get("symbol") or "-"))
        score = float(row.get(score_key) or 0.0)
        width = max(3.0, min(100.0, score))
        signal = str(row.get("flow_signal") or "")
        trend = str(row.get("flow_trend_5d") or "")
        klass = _pulse_class(trend if recent else signal)
        if recent:
            detail = f"{_pulse_label(trend)} | {int(row.get('inflow_days_5d') or 0)}/5 phiên"
        else:
            detail = f"{_pulse_label(signal)} | {_pulse_label(trend)}"
        bars.append(
            f'<div class="flow-row"><b>{symbol}</b><div class="bar-track">'
            f'<i class="bar {klass}" style="width:{width:.1f}%"></i></div>'
            f'<strong>{score:.1f}</strong><span>{html.escape(detail)}</span></div>'
        )
    return "".join(bars) if bars else '<p class="empty">Không có mã thỏa điều kiện.</p>'


def _money_flow_panel(pulse: dict[str, Any], today: pd.DataFrame, recent: pd.DataFrame) -> str:
    if not pulse.get("available"):
        return ""
    state = str(pulse.get("pulse_state") or "TRUNG_LAP_THEO_DOI")
    summary_metrics = "\n".join(
        [
            _metric("Tín hiệu hôm nay", pulse.get("strong_inflow_symbols_today"), "win"),
            _metric("Có tín hiệu >=3/5 phiên", pulse.get("persistent_inflow_symbols_5d"), "win"),
            _metric("CHDM20 đổi 5 phiên", pulse.get("mkt_CHDM20_change_5d")),
            _metric("DS20 đổi 5 phiên", pulse.get("mkt_DS20_change_5d")),
        ]
    )
    note = str(pulse.get("interpretation_note") or "")
    return f"""
    <section class="pulse">
      <div class="pulse-head">
        <div>
          <h2>Dòng tiền quan sát - Flow V2</h2>
          <p>Ảnh chụp {html.escape(str(pulse.get("date") or "-"))} | VN100 | score đang dùng cho ranking Flow V2</p>
        </div>
        <strong class="pill {_pulse_class(state)}">{html.escape(_pulse_label(state))}</strong>
      </div>
      <div class="pulse-stats">{summary_metrics}</div>
      <div class="pulse-sections">
        <section class="pulse-block">
          <h3>1. Tiền vào hôm nay</h3>
          <p class="block-note">Chỉ gồm mã đạt proxy dòng tiền tại phiên mới nhất. Thanh biểu diễn score hôm nay.</p>
          <div class="flow-bars">{_flow_bars(today)}</div>
          <a class="detail-link" href="05_money_flow_today.html">Mở dữ liệu hôm nay</a>
        </section>
        <section class="pulse-block">
          <h3>2. Dòng tiền mạnh thời gian vừa qua</h3>
          <p class="block-note">Mã đạt proxy ít nhất 2/5 phiên gần nhất. Thanh biểu diễn score trung bình 5 phiên.</p>
          <div class="flow-bars">{_flow_bars(recent, recent=True)}</div>
          <a class="detail-link" href="06_money_flow_recent.html">Mở dữ liệu 5 phiên</a>
        </section>
      </div>
      <p class="disclaimer">{html.escape(note)}</p>
    </section>"""


def export_html(demo_dir: Path) -> Path:
    status = _read_json(demo_dir / "flow_v2_status.json")
    market = _read_json(demo_dir / "market_snapshot.json")
    pulse = _read_json(demo_dir / "money_flow_pulse.json")
    candidates = _read_csv(demo_dir / "flow_v2_candidates.csv")
    targets = _read_csv(demo_dir / "flow_v2_target_plan.csv")
    today = _read_csv(demo_dir / "flow_v2_money_flow_today.csv")
    recent = _read_csv(demo_dir / "flow_v2_money_flow_recent.csv")

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
    _write_table_page(
        demo_dir / "05_money_flow_today.html",
        "05 Tiền Vào Hôm Nay",
        today,
        "Symbols satisfying the Flow V2 price/volume inflow proxy on the latest session.",
    )
    _write_table_page(
        demo_dir / "06_money_flow_recent.html",
        "06 Dòng Tiền Mạnh 5 Phiên",
        recent,
        "Symbols satisfying the Flow V2 inflow proxy on at least two of the latest five sessions.",
    )
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
            _card("05_money_flow_today.html", "05 Tiền Vào Hôm Nay", "Mã đạt tín hiệu dòng tiền ở phiên mới nhất."),
            _card("06_money_flow_recent.html", "06 Dòng Tiền Mạnh 5 Phiên", "Mã có dòng tiền nổi bật lặp lại gần đây."),
        ]
    )
    money_flow_panel = _money_flow_panel(pulse, today, recent)

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
    .pulse {{ padding: 22px; margin: 0 0 24px; background: rgba(255,255,255,.78); border: 1px solid rgba(16,36,62,.12); border-radius: 18px; box-shadow: 0 18px 48px rgba(16,36,62,.10); font-family: Segoe UI, Arial, sans-serif; }}
    .pulse-head {{ display: flex; justify-content: space-between; align-items: start; gap: 14px; margin-bottom: 16px; }}
    .pulse h2 {{ margin: 0 0 4px; font-family: Georgia, 'Times New Roman', serif; font-size: 27px; }}
    .pulse h3 {{ margin: 18px 0 12px; font-size: 16px; }}
    .pulse-head p, .disclaimer {{ color: var(--muted); font-size: 13px; margin: 0; }}
    .pill {{ border: 1px solid currentColor; border-radius: 999px; padding: 8px 12px; font-size: 13px; white-space: nowrap; }}
    .pulse-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 9px; }}
    .pulse-sections {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; margin-top: 18px; }}
    .pulse-block {{ border: 1px solid rgba(16,36,62,.10); border-radius: 10px; padding: 14px; background: rgba(255,255,255,.54); }}
    .flow-bars {{ display: grid; gap: 8px; }}
    .flow-row {{ display: grid; grid-template-columns: 45px minmax(80px, 1fr) 40px; align-items: center; gap: 8px; font-size: 13px; margin-bottom: 4px; }}
    .flow-row > strong {{ text-align: right; }}
    .flow-row > span {{ color: var(--muted); grid-column: 2 / 4; font-size: 12px; margin-top: -5px; }}
    .block-note {{ min-height: 34px; margin: -5px 0 13px; color: var(--muted); font-size: 12px; }}
    .bar-track {{ height: 15px; background: #edf0f3; border-radius: 999px; overflow: hidden; }}
    .bar {{ display: block; height: 100%; border-radius: 999px; background: #8c98a7; }}
    .bar.win {{ background: var(--win); }}
    .bar.loss {{ background: var(--loss); }}
    .disclaimer {{ border-top: 1px solid rgba(16,36,62,.10); padding-top: 13px; margin-top: 16px; }}
    .detail-link {{ display: inline-block; margin-top: 12px; color: var(--ink); font-weight: 600; }}
    @media (max-width: 760px) {{ .pulse-head {{ display: block; }} .pill {{ display: inline-block; margin-top: 10px; }} .pulse-sections {{ grid-template-columns: 1fr; }} }}
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
    {money_flow_panel}
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
