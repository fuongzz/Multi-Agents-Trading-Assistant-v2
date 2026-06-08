"""Daily VN market-state agent.

The agent is deliberately deterministic. It reads local parquet research
artifacts first, then optionally enriches the report with live macro/news
context. The output is an operating view, not a broker order generator.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from multiagents_trading_assistant.edge_lab.features import build_feature_table


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "reports" / "daily_market_state"
OHLCV_PATH = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
INDEX_PATH = ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"


@dataclass(frozen=True)
class AllocationHint:
    stock_core_pct: float
    tactical_rebound_pct: float
    accumulation_watch_pct: float
    cash_pct: float
    term_deposit_pct: float
    note: str


@dataclass(frozen=True)
class MarketStateReport:
    generated_at: str
    as_of: str
    market_lane: str
    conviction: float
    action_bias: str
    risk_summary: str
    allocation: AllocationHint
    metrics: dict[str, Any]
    lane_scores: dict[str, float]
    reasons: list[str]
    watch_actions: list[str]
    sector_snapshot: list[dict[str, Any]]
    system_assessment: dict[str, Any] = field(default_factory=dict)
    global_context: dict[str, Any] = field(default_factory=dict)
    vn_macro_context: dict[str, Any] = field(default_factory=dict)
    news_context: dict[str, Any] = field(default_factory=dict)
    data_sources: list[str] = field(default_factory=list)


def build_daily_market_state_report(
    *,
    as_of: str | None = None,
    root: str | Path = ROOT,
    with_live_context: bool = False,
    news_limit: int = 12,
) -> MarketStateReport:
    """Build one daily report from local market data.

    Args:
        as_of: YYYY-MM-DD. Defaults to latest date in local OHLCV data.
        root: Repository root.
        with_live_context: If true, call optional macro/news fetchers.
        news_limit: Maximum expert/news headlines when live context is enabled.
    """
    repo = Path(root)
    ohlcv = pd.read_parquet(repo / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.normalize()
    ohlcv["symbol"] = ohlcv["symbol"].astype(str).str.upper()
    target = pd.Timestamp(as_of).normalize() if as_of else ohlcv["date"].max()
    symbols = sorted(ohlcv.loc[ohlcv["date"].eq(target), "symbol"].unique().tolist())
    if not symbols:
        latest = ohlcv["date"].max()
        raise ValueError(f"No symbols found for as_of={target.date()}; latest local date is {latest.date()}")

    start = target - pd.Timedelta(days=460)
    features, _ = build_feature_table(symbols, start=start, end=target, root=repo)
    daily_features = features[features["date"].eq(target)].copy()
    if daily_features.empty:
        raise ValueError(f"Feature table is empty for {target.date()}")

    index = _load_index(repo, target)
    metrics = _build_metrics(daily_features, features, index, target)
    global_ctx: dict[str, Any] = {}
    vn_macro_ctx: dict[str, Any] = {}
    news_ctx: dict[str, Any] = {}
    if with_live_context:
        global_ctx, vn_macro_ctx, news_ctx = _load_live_context(target, news_limit=news_limit)
        metrics.update(_macro_metrics(global_ctx, vn_macro_ctx))

    system_assessment = _build_system_assessment(metrics)
    metrics.update(
        {
            "system_overall_risk_score": system_assessment["overall_risk_score"],
            "system_state": system_assessment["state"],
            "vn_market_stress_score": _component_score(system_assessment, "VN market structure"),
            "global_risk_score": _component_score(system_assessment, "Global risk"),
            "policy_pressure_score": _component_score(system_assessment, "Policy / FX pressure"),
            "liquidity_flow_score": _component_score(system_assessment, "Liquidity / flow"),
        }
    )
    lane_scores = _score_lanes(metrics)
    market_lane = _choose_lane(lane_scores, metrics)
    allocation = _allocation_for_lane(market_lane, metrics, lane_scores)
    reasons = _reasons_for_lane(market_lane, metrics, lane_scores)
    watch_actions = _watch_actions(market_lane, metrics, allocation)
    sector_snapshot = _sector_snapshot(daily_features)
    action_bias = _action_bias(market_lane)
    risk_summary = _risk_summary(market_lane, metrics)

    return MarketStateReport(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        as_of=target.date().isoformat(),
        market_lane=market_lane,
        conviction=round(float(lane_scores.get(market_lane, 0.0)), 2),
        action_bias=action_bias,
        risk_summary=risk_summary,
        allocation=allocation,
        metrics=_json_safe(metrics),
        lane_scores={k: round(float(v), 2) for k, v in lane_scores.items()},
        reasons=reasons,
        watch_actions=watch_actions,
        sector_snapshot=sector_snapshot,
        system_assessment=_json_safe(system_assessment),
        global_context=_json_safe(global_ctx),
        vn_macro_context=_json_safe(vn_macro_ctx),
        news_context=_json_safe(news_ctx),
        data_sources=[
            "multiagents_trading_assistant/data/ohlcv_master.parquet",
            "multiagents_trading_assistant/data/index_master.parquet",
            "data/research/money_cycle/*.parquet",
            "data/research/smart_money_trace/smart_money_by_symbol.parquet",
            "optional: fetcher.get_global_macro/get_vn_macro and news_fetcher.get_expert_news",
        ],
    )


def write_market_state_report(report: MarketStateReport, out_dir: str | Path = DEFAULT_OUT_DIR) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "market_state_latest.json"
    md_path = out / "market_state_latest.md"
    html_path = out / "index.html"
    dated_json = out / f"market_state_{report.as_of}.json"
    dated_md = out / f"market_state_{report.as_of}.md"

    payload = asdict(report)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    markdown = render_markdown(report)
    html = render_html(report)

    json_path.write_text(json_text, encoding="utf-8")
    dated_json.write_text(json_text, encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    dated_md.write_text(markdown, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "html": str(html_path),
        "dated_json": str(dated_json),
        "dated_markdown": str(dated_md),
    }


def render_markdown(report: MarketStateReport) -> str:
    m = report.metrics
    alloc = report.allocation
    lines = [
        f"# Daily Market State - {report.as_of}",
        "",
        f"- Lane: **{report.market_lane}**",
        f"- Conviction: **{report.conviction:.2f}/100**",
        f"- Bias: **{report.action_bias}**",
        f"- Risk summary: {report.risk_summary}",
        "",
        "## Allocation Hint",
        "",
        "| Bucket | Percent |",
        "|---|---:|",
        f"| Core stock exposure | {alloc.stock_core_pct:.0f}% |",
        f"| Tactical rebound | {alloc.tactical_rebound_pct:.0f}% |",
        f"| Accumulation watch | {alloc.accumulation_watch_pct:.0f}% |",
        f"| Cash / T-bill-like liquidity | {alloc.cash_pct:.0f}% |",
        f"| Term deposit / savings | {alloc.term_deposit_pct:.0f}% |",
        "",
        f"Note: {alloc.note}",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in [
        "vnindex_close",
        "vnindex_ret_1d_pct",
        "vnindex_ret_20d_pct",
        "vnindex_ret_60d_pct",
        "mkt_regime_state",
        "mkt_regime_score",
        "mkt_CHDM20",
        "mkt_CHDM50",
        "mkt_DS20",
        "breadth_ma20_pct",
        "breadth_ma50_pct",
        "avg_smart_money_score",
        "accumulation_ratio_pct",
        "distribution_pressure",
    ]:
        if key in m:
            lines.append(f"| {key} | {_fmt(m[key])} |")
    lines.extend(["", "## Lane Scores", "", "| Lane | Score |", "|---|---:|"])
    for key, value in sorted(report.lane_scores.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {key} | {value:.2f} |")
    lines.extend(["", "## Reasons", ""])
    lines.extend([f"- {item}" for item in report.reasons])
    lines.extend(["", "## Watch Actions", ""])
    lines.extend([f"- {item}" for item in report.watch_actions])
    if report.sector_snapshot:
        lines.extend(["", "## Sector Snapshot", "", "| Industry | Count | Avg 20D Return | Avg Edge |", "|---|---:|---:|---:|"])
        for row in report.sector_snapshot[:10]:
            lines.append(
                f"| {row.get('industry', '')} | {row.get('count', 0)} | "
                f"{_fmt(row.get('avg_ret_20d_pct'))} | {_fmt(row.get('avg_edge_score'))} |"
            )
    if report.news_context.get("headlines"):
        lines.extend(["", "## News / Expert Headlines", ""])
        for item in report.news_context["headlines"][:8]:
            lines.append(f"- [{item.get('source', 'news')}] {item.get('headline', '')}")
    return "\n".join(lines) + "\n"


def render_html(report: MarketStateReport) -> str:
    m = report.metrics
    alloc = report.allocation
    lane_meta = _lane_meta(report.market_lane)
    quick_take = _plain_takeaway(report)
    caution_note = _caution_note(report)

    def metric_card(label: str, value: Any, note: str, tone: str = "") -> str:
        tone_class = f" {tone}" if tone else ""
        return (
            f'<div class="metric-card{tone_class}">'
            f'<div class="metric-label">{escape(label)}</div>'
            f'<div class="metric-value">{escape(_fmt(value))}</div>'
            f'<div class="metric-note">{escape(note)}</div>'
            "</div>"
        )

    def allocation_segment(label: str, value: float, css_class: str) -> str:
        width = max(0.0, min(100.0, float(value)))
        if width <= 0:
            return ""
        return f'<div class="alloc-seg {css_class}" style="width:{width:.1f}%"><span>{escape(label)} {width:.0f}%</span></div>'

    def macro_card(label: str, value: Any, note: str, tone: str = "") -> str:
        tone_class = f" {tone}" if tone else ""
        return (
            f'<div class="metric-card macro-card{tone_class}">'
            f'<div class="metric-label">{escape(label)}</div>'
            f'<div class="metric-value">{escape(_fmt(value))}</div>'
            f'<div class="metric-note">{escape(note)}</div>'
            "</div>"
        )

    score_rows = "\n".join(
        f"""
        <div class="score-row">
          <div class="score-name">{escape(_lane_label(key))}</div>
          <div class="score-track"><div class="score-fill {escape(_lane_class(key))}" style="width:{max(0.0, min(100.0, value)):.1f}%"></div></div>
          <div class="score-num">{value:.0f}</div>
        </div>
        """
        for key, value in sorted(report.lane_scores.items(), key=lambda item: item[1], reverse=True)
    )
    simple_reasons = "".join(f"<li>{escape(item)}</li>" for item in _simple_explanation(report))
    technical_reasons = "".join(f"<li>{escape(item)}</li>" for item in report.reasons[:6])
    reading_notes = "".join(f"<li>{escape(item)}</li>" for item in _retail_reading_notes(report))
    sector_rows = "".join(
        f"""
        <tr>
          <td>{escape(str(row.get("industry", "")))}</td>
          <td>{escape(_fmt(row.get("count")))}</td>
          <td class="{_signed_class(row.get("avg_ret_20d_pct"))}">{escape(_fmt(row.get("avg_ret_20d_pct")))}%</td>
          <td>{escape(_fmt(row.get("avg_edge_score")))}</td>
        </tr>
        """
        for row in report.sector_snapshot[:8]
    )
    headline_items = "".join(
        f'<li><span>{escape(str(item.get("source") or "news"))}</span>{escape(str(item.get("headline") or ""))}</li>'
        for item in (report.news_context.get("headlines") or [])[:6]
    )
    macro_cards = _macro_cards(report, macro_card)
    macro_notes = "".join(f"<li>{escape(item)}</li>" for item in _macro_reading_notes(report))
    system_cards = _system_assessment_cards(report, metric_card)
    system_notes = "".join(f"<li>{escape(item)}</li>" for item in _system_assessment_notes(report))
    metric_rows = "\n".join(
        f"<tr><td>{escape(_metric_label(k))}</td><td>{escape(_fmt(v))}</td></tr>"
        for k, v in report.metrics.items()
        if isinstance(v, (int, float, str, bool)) or v is None
    )
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Daily Market Brief - {escape(report.as_of)}</title>
  <style>
    :root {{
      --ink:#18212f; --muted:#667085; --line:#d9e0ea; --bg:#f4f6f8; --panel:#ffffff;
      --red:#c24132; --red-bg:#fff1ed; --amber:#b46a00; --amber-bg:#fff7df;
      --green:#137333; --green-bg:#eaf7ee; --blue:#22577a; --blue-bg:#e9f2f7;
      --cash:#526071; --savings:#7c5c2f; --watch:#2f6f73; --core:#1f7a4d; --rebound:#b85d2a;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family: Segoe UI, Arial, sans-serif; color:var(--ink); background:var(--bg); }}
    header {{ padding:22px 28px; background:#162231; color:white; }}
    main {{ padding:20px 28px 36px; max-width:1180px; margin:auto; }}
    h1 {{ margin:0 0 5px; font-size:25px; letter-spacing:0; }}
    h2 {{ margin:0 0 12px; font-size:18px; }}
    .sub {{ color:#c9d3df; font-size:13px; }}
    .hero {{ margin:18px 0; padding:18px; border:1px solid var(--line); border-left:8px solid {escape(lane_meta["color"])}; background:{escape(lane_meta["bg"])}; border-radius:8px; }}
    .hero-top {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; flex-wrap:wrap; }}
    .lane {{ font-size:28px; font-weight:800; margin:2px 0 8px; }}
    .plain {{ font-size:17px; line-height:1.45; max-width:860px; }}
    .badge {{ display:inline-flex; align-items:center; min-height:26px; padding:4px 9px; border-radius:6px; background:white; border:1px solid var(--line); font-size:13px; font-weight:700; }}
    .section {{ margin-top:16px; padding:16px; background:var(--panel); border:1px solid var(--line); border-radius:8px; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; }}
    .metric-card {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:#fff; min-height:105px; }}
    .metric-card.bad {{ background:var(--red-bg); border-color:#f0b4aa; }}
    .metric-card.warn {{ background:var(--amber-bg); border-color:#efce83; }}
    .metric-card.good {{ background:var(--green-bg); border-color:#aad8b6; }}
    .metric-label {{ color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; }}
    .metric-value {{ margin-top:7px; font-size:25px; font-weight:800; }}
    .metric-note {{ margin-top:6px; color:var(--muted); font-size:12px; line-height:1.35; }}
    .macro-card .metric-value {{ font-size:22px; }}
    .alloc-bar {{ display:flex; overflow:hidden; height:46px; border-radius:8px; border:1px solid var(--line); background:#eef1f5; }}
    .alloc-seg {{ display:flex; align-items:center; justify-content:center; min-width:38px; color:white; font-size:12px; font-weight:800; white-space:nowrap; }}
    .alloc-seg span {{ padding:0 6px; overflow:hidden; text-overflow:ellipsis; }}
    .core {{ background:var(--core); }} .rebound {{ background:var(--rebound); }} .watch {{ background:var(--watch); }}
    .cash {{ background:var(--cash); }} .savings {{ background:var(--savings); }}
    .hint {{ margin-top:10px; color:var(--muted); line-height:1.45; }}
    .note-box {{ padding:12px; border-radius:8px; background:#f7f9fb; border:1px solid var(--line); line-height:1.5; }}
    .two-col {{ display:grid; grid-template-columns:1.2fr .8fr; gap:14px; }}
    .score-row {{ display:grid; grid-template-columns:190px 1fr 44px; align-items:center; gap:10px; margin:10px 0; }}
    .score-name {{ font-size:13px; font-weight:700; }}
    .score-track {{ height:13px; border-radius:999px; background:#ecf0f4; overflow:hidden; }}
    .score-fill {{ height:100%; border-radius:999px; }}
    .risk {{ background:var(--red); }} .warning {{ background:var(--amber); }} .positive {{ background:var(--green); }} .neutral {{ background:var(--blue); }}
    .score-num {{ font-size:13px; font-weight:800; text-align:right; }}
    .system-summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:10px; }}
    ul.clean {{ margin:0; padding-left:20px; line-height:1.55; }}
    table {{ width:100%; border-collapse:collapse; background:white; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid #edf0f4; text-align:left; font-size:13px; }}
    th {{ background:#eef2f6; font-weight:800; }}
    .pos {{ color:var(--green); font-weight:700; }} .neg {{ color:var(--red); font-weight:700; }}
    .news-list {{ margin:0; padding-left:20px; line-height:1.5; }}
    .news-list span {{ display:inline-block; margin-right:8px; color:var(--blue); font-weight:800; }}
    details {{ margin-top:16px; }}
    summary {{ cursor:pointer; color:var(--blue); font-weight:800; }}
    @media (max-width: 820px) {{
      main {{ padding:14px; }}
      .two-col {{ grid-template-columns:1fr; }}
      .score-row {{ grid-template-columns:130px 1fr 38px; }}
      .lane {{ font-size:22px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Daily Market Brief</h1>
    <div class="sub">Tổng quan thị trường sau phiên {escape(report.as_of)} · Generated {escape(report.generated_at)}</div>
  </header>
  <main>
    <section class="hero">
      <div class="hero-top">
        <div>
          <div class="badge">{escape(lane_meta["label"])}</div>
          <div class="lane">{escape(lane_meta["headline"])}</div>
          <div class="plain">{escape(quick_take)}</div>
        </div>
        <div class="badge">Mức tín hiệu {report.conviction:.0f}/100</div>
      </div>
    </section>

    <section class="section">
      <h2>Đọc Nhanh</h2>
      <div class="metric-grid">
        {metric_card("VNINDEX hôm nay", f'{_fmt(m.get("vnindex_ret_1d_pct"))}%', "Mức giảm/tăng trong phiên", "bad" if _f(m, "vnindex_ret_1d_pct") < -1 else "good" if _f(m, "vnindex_ret_1d_pct") > 1 else "warn")}
        {metric_card("Market score", f'{_fmt(m.get("mkt_regime_score"))}/100', "Càng thấp càng risk-off", "bad" if _f(m, "mkt_regime_score") < 40 else "good" if _f(m, "mkt_regime_score") >= 60 else "warn")}
        {metric_card("Breadth MA20", f'{_fmt(m.get("breadth_ma20_pct"))}%', "Số cổ phiếu còn trên MA20", "bad" if _f(m, "breadth_ma20_pct") < 30 else "good" if _f(m, "breadth_ma20_pct") >= 55 else "warn")}
        {metric_card("Đóng cửa trong nến", f'{_fmt(m.get("vnindex_close_location"))}', "0=gần đáy, 1=gần đỉnh", "bad" if _f(m, "vnindex_close_location") < 0.3 else "good" if _f(m, "vnindex_close_location") > 0.6 else "warn")}
      </div>
    </section>

    <section class="section">
      <h2>Gợi Ý Mức Độ Thận Trọng</h2>
      <div class="alloc-bar">
        {allocation_segment("Cổ phiếu", alloc.stock_core_pct, "core")}
        {allocation_segment("Theo dõi hồi", alloc.tactical_rebound_pct, "rebound")}
        {allocation_segment("Watchlist", alloc.accumulation_watch_pct, "watch")}
        {allocation_segment("Tiền mặt", alloc.cash_pct, "cash")}
        {allocation_segment("Tiết kiệm", alloc.term_deposit_pct, "savings")}
      </div>
      <div class="hint">{escape(caution_note)}</div>
    </section>

    <div class="two-col">
      <section class="section">
        <h2>Nói Dễ Hiểu Là Gì?</h2>
        <ul class="clean">{simple_reasons}</ul>
      </section>
      <section class="section">
        <h2>Cách Đọc Hôm Nay</h2>
        <ul class="clean">{reading_notes}</ul>
      </section>
    </div>

    <section class="section">
      <h2>Khung Đánh Giá Hệ Thống</h2>
      <div class="system-summary">{system_cards}</div>
      <div class="note-box" style="margin-top:12px">
        <ul class="clean">{system_notes}</ul>
      </div>
    </section>

    <section class="section">
      <h2>Thị Trường Đang Nghiêng Về Kịch Bản Nào?</h2>
      {score_rows}
    </section>

    <section class="section">
      <h2>Vĩ Mô & Thế Giới</h2>
      <div class="metric-grid">{macro_cards}</div>
      <div class="note-box" style="margin-top:12px">
        <ul class="clean">{macro_notes}</ul>
      </div>
    </section>

    <section class="section">
      <h2>Nhóm Ngành Còn Tương Đối Khỏe</h2>
      <table><thead><tr><th>Ngành</th><th>Số mã</th><th>20 phiên</th><th>Edge</th></tr></thead><tbody>{sector_rows}</tbody></table>
    </section>

    <section class="section">
      <h2>Tin Tức Đáng Chú Ý</h2>
      <div class="note-box">Phần này dùng để hiểu bối cảnh thị trường, không phải tín hiệu mua bán. Tin tức chỉ giúp giải thích tâm lý và rủi ro trong ngày.</div>
      <ul class="news-list">{headline_items or "<li>Chưa có dữ liệu tin tức trong lần chạy này.</li>"}</ul>
    </section>

    <details>
      <summary>Chi tiết kỹ thuật</summary>
      <section class="section">
        <h2>Lý Do Kỹ Thuật</h2>
        <ul class="clean">{technical_reasons}</ul>
        <h2>Toàn Bộ Metrics</h2>
        <table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{metric_rows}</tbody></table>
      </section>
    </details>
  </main>
</body>
</html>
"""


def _build_system_assessment(m: dict[str, Any]) -> dict[str, Any]:
    components = [
        _vn_market_structure_component(m),
        _global_risk_component(m),
        _policy_fx_component(m),
        _liquidity_flow_component(m),
    ]
    weights = {
        "VN market structure": 0.40,
        "Global risk": 0.25,
        "Policy / FX pressure": 0.15,
        "Liquidity / flow": 0.20,
    }
    overall = sum(float(c["score"]) * weights.get(str(c["name"]), 0.0) for c in components)
    state = _risk_state(overall)
    return {
        "framework_version": "daily_market_state_v2",
        "overall_risk_score": round(overall, 2),
        "state": state,
        "state_label": _risk_state_label(state),
        "interpretation": _system_interpretation(state, components),
        "components": components,
    }


def _vn_market_structure_component(m: dict[str, Any]) -> dict[str, Any]:
    score = 0.0
    evidence: list[str] = []
    ret1 = _f(m, "vnindex_ret_1d_pct")
    ret20 = _f(m, "vnindex_ret_20d_pct")
    ret60 = _f(m, "vnindex_ret_60d_pct")
    regime = _f(m, "mkt_regime_score")
    breadth20 = _f(m, "breadth_ma20_pct")
    breadth50 = _f(m, "breadth_ma50_pct")
    ds20 = _f(m, "mkt_DS20")
    chdm20 = _f(m, "mkt_CHDM20")
    close_loc = _f(m, "vnindex_close_location", 0.5)

    if ret1 <= -2.0:
        score += 16
        evidence.append(f"VNINDEX giảm mạnh trong ngày ({_fmt(ret1)}%).")
    elif ret1 <= -1.0:
        score += 8
        evidence.append(f"VNINDEX giảm trên 1% ({_fmt(ret1)}%).")
    if ret20 <= -5.0:
        score += 12
        evidence.append(f"Xu hướng 20 phiên yếu ({_fmt(ret20)}%).")
    if ret60 <= -8.0:
        score += 14
        evidence.append(f"Xu hướng 60 phiên đã xấu rõ ({_fmt(ret60)}%).")
    if regime < 25:
        score += 18
        evidence.append(f"Market score rất thấp ({_fmt(regime)}/100).")
    elif regime < 40:
        score += 10
        evidence.append(f"Market score dưới trung bình ({_fmt(regime)}/100).")
    if breadth20 <= 20:
        score += 16
        evidence.append(f"Độ rộng MA20 rất hẹp ({_fmt(breadth20)}%).")
    elif breadth20 <= 35:
        score += 8
        evidence.append(f"Độ rộng MA20 yếu ({_fmt(breadth20)}%).")
    if breadth50 <= 30:
        score += 10
        evidence.append(f"Độ rộng MA50 yếu ({_fmt(breadth50)}%).")
    if ds20 >= 0.65:
        score += 10
        evidence.append(f"DS20 cao ({_fmt(ds20)}), áp lực phân phối/đứt xu hướng lớn.")
    if chdm20 < 25:
        score += 8
        evidence.append(f"CHDM20 thấp ({_fmt(chdm20)}), ít mã còn giữ momentum.")
    if close_loc < 0.30:
        score += 8
        evidence.append("Chỉ số đóng gần đáy phiên, lực đỡ cuối ngày yếu.")

    return _component("VN market structure", score, evidence, "Nội tại thị trường Việt Nam")


def _global_risk_component(m: dict[str, Any]) -> dict[str, Any]:
    score = 0.0
    evidence: list[str] = []
    sp500 = _f(m, "global_sp500_change_pct")
    sp500_5d = _f(m, "global_sp500_change_5d_pct")
    btc_5d = _f(m, "global_btc_change_5d_pct")
    gold_5d = _f(m, "global_gold_change_5d_pct")
    kospi = _f(m, "global_kospi_change_pct")
    hsi = _f(m, "global_hsi_change_pct")
    nikkei = _f(m, "global_nikkei_change_pct")

    if sp500_5d <= -1.0:
        score += 14
        evidence.append(f"S&P giảm trong nhịp gần đây ({_fmt(sp500_5d)}%).")
    elif sp500 <= -1.0:
        score += 10
        evidence.append(f"S&P giảm mạnh trong phiên gần nhất ({_fmt(sp500)}%).")
    if btc_5d <= -8.0:
        score += 18
        evidence.append(f"Bitcoin giảm rất mạnh khi VN nghỉ giao dịch ({_fmt(btc_5d)}%).")
    elif btc_5d <= -3.0:
        score += 10
        evidence.append(f"Bitcoin yếu trong nhịp gần đây ({_fmt(btc_5d)}%).")
    if gold_5d <= -2.0:
        score += 12
        evidence.append(f"Vàng giảm mạnh ({_fmt(gold_5d)}%), có thể là bán tài sản trú ẩn/giảm đòn bẩy.")
    if kospi <= -8.0:
        score += 22
        evidence.append(f"KOSPI giảm trên 8% ({_fmt(kospi)}%), vùng circuit breaker.")
    elif kospi <= -3.0:
        score += 12
        evidence.append(f"KOSPI giảm mạnh ({_fmt(kospi)}%).")
    asia_weak = sum(1 for v in [kospi, hsi, nikkei] if v <= -1.0)
    if asia_weak >= 2:
        score += 10
        evidence.append("Ít nhất 2 thị trường châu Á lớn giảm trên 1%.")

    return _component("Global risk", score, evidence, "Risk-off toàn cầu/khu vực")


def _policy_fx_component(m: dict[str, Any]) -> dict[str, Any]:
    score = 0.0
    evidence: list[str] = []
    dxy = _f(m, "global_dxy_change_pct")
    dxy_5d = _f(m, "global_dxy_change_5d_pct")
    oil = _f(m, "global_oil_change_pct")
    oil_5d = _f(m, "global_oil_change_5d_pct")
    usd_vnd = m.get("vn_usd_vnd")
    sbv_rate = m.get("vn_sbv_rate")

    if dxy_5d >= 0.7 or dxy >= 0.7:
        score += 20
        evidence.append(f"DXY tăng ({_fmt(dxy)}% / {_fmt(dxy_5d)}%), áp lực USD/tỷ giá.")
    elif dxy_5d >= 0.3:
        score += 10
        evidence.append(f"DXY nhích lên trong vài phiên ({_fmt(dxy_5d)}%).")
    if oil >= 1.5 or oil_5d >= 2.0:
        score += 12
        evidence.append(f"Dầu tăng ({_fmt(oil)}% / {_fmt(oil_5d)}%), nhạy với câu chuyện lạm phát.")
    if usd_vnd is not None:
        score += 8
        evidence.append(f"USD/VND quanh {_fmt(usd_vnd)}, cần theo dõi áp lực tỷ giá.")
    if sbv_rate is not None:
        score += 5
        evidence.append(f"SBV rate {_fmt(sbv_rate)}%, dùng để đọc mặt bằng tiền gửi/cổ phiếu.")

    return _component("Policy / FX pressure", score, evidence, "Áp lực tỷ giá, lãi suất, hàng hóa")


def _liquidity_flow_component(m: dict[str, Any]) -> dict[str, Any]:
    score = 0.0
    evidence: list[str] = []
    smart = _f(m, "avg_smart_money_score", 50.0)
    value_ratio = _f(m, "avg_value_ratio_20", 1.0)
    distribution = _f(m, "distribution_ratio_pct")
    accumulation = _f(m, "accumulation_ratio_pct")
    breadth_delta = _f(m, "breadth_ma20_delta_10d")

    if smart < 45:
        score += 14
        evidence.append(f"Smart money score thấp ({_fmt(smart)}).")
    elif smart >= 55:
        score -= 8
        evidence.append(f"Smart money score còn tương đối tốt ({_fmt(smart)}).")
    if value_ratio < 0.85:
        score += 10
        evidence.append(f"Thanh khoản/giá trị yếu hơn bình quân ({_fmt(value_ratio)}).")
    if distribution >= 20:
        score += 16
        evidence.append(f"Tỷ lệ phân phối cao ({_fmt(distribution)}%).")
    if accumulation >= distribution and accumulation >= 15:
        score -= 8
        evidence.append(f"Có một phần mã tích lũy ({_fmt(accumulation)}%).")
    if breadth_delta < -0.15:
        score += 12
        evidence.append("Độ rộng xấu đi nhanh trong 10 phiên.")

    return _component("Liquidity / flow", score, evidence, "Dòng tiền và độ rộng")


def _component(name: str, score: float, evidence: list[str], label: str) -> dict[str, Any]:
    score = max(0.0, min(100.0, float(score)))
    state = _risk_state(score)
    return {
        "name": name,
        "label": label,
        "score": round(score, 2),
        "state": state,
        "state_label": _risk_state_label(state),
        "tone": _risk_state_tone(state),
        "evidence": evidence[:4] or ["Chưa có tín hiệu nổi bật; giữ mức theo dõi trung tính."],
    }


def _component_score(assessment: dict[str, Any], name: str) -> float:
    for item in assessment.get("components", []):
        if item.get("name") == name:
            return float(item.get("score", 0.0))
    return 0.0


def _risk_state(score: float) -> str:
    if score >= 75:
        return "SEVERE"
    if score >= 55:
        return "STRESS"
    if score >= 35:
        return "WATCH"
    return "NORMAL"


def _risk_state_label(state: str) -> str:
    return {
        "SEVERE": "Rủi ro rất cao",
        "STRESS": "Rủi ro cao",
        "WATCH": "Cần theo dõi",
        "NORMAL": "Bình thường",
    }.get(state, state)


def _risk_state_tone(state: str) -> str:
    return {
        "SEVERE": "bad",
        "STRESS": "bad",
        "WATCH": "warn",
        "NORMAL": "good",
    }.get(state, "warn")


def _system_interpretation(state: str, components: list[dict[str, Any]]) -> str:
    worst = sorted(components, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:2]
    drivers = ", ".join(str(item.get("label")) for item in worst)
    if state == "SEVERE":
        return f"Rủi ro hệ thống rất cao; tác nhân chính là {drivers}."
    if state == "STRESS":
        return f"Thị trường đang trong vùng stress; cần đọc kỹ {drivers}."
    if state == "WATCH":
        return f"Rủi ro chưa cực đoan nhưng cần theo dõi {drivers}."
    return "Bối cảnh hệ thống chưa có áp lực lớn; vẫn theo dõi thay đổi từ từng component."


def _system_assessment_cards(report: MarketStateReport, card_fn: Any) -> str:
    assessment = report.system_assessment or {}
    cards = [
        card_fn(
            "Tổng rủi ro hệ thống",
            f"{_fmt(assessment.get('overall_risk_score'))}/100",
            str(assessment.get("state_label") or ""),
            _risk_state_tone(str(assessment.get("state") or "WATCH")),
        )
    ]
    for item in assessment.get("components", []):
        cards.append(
            card_fn(
                str(item.get("label") or item.get("name")),
                f"{_fmt(item.get('score'))}/100",
                str(item.get("state_label") or ""),
                str(item.get("tone") or "warn"),
            )
        )
    return "\n".join(cards)


def _system_assessment_notes(report: MarketStateReport) -> list[str]:
    assessment = report.system_assessment or {}
    notes = [str(assessment.get("interpretation") or "Chưa có đánh giá hệ thống.")]
    for item in assessment.get("components", []):
        evidence = item.get("evidence") or []
        if evidence:
            notes.append(f"{item.get('label')}: {'; '.join(str(x) for x in evidence[:2])}")
    notes.append("Khung này là lớp đọc bối cảnh; nó không sinh tín hiệu mua bán và không thay thế quản trị rủi ro cá nhân.")
    return notes


def _macro_cards(report: MarketStateReport, card_fn: Any) -> str:
    ctx = report.global_context
    m = report.metrics
    cards: list[str] = []

    def pct_card(label: str, key: str, note: str, tone_mode: str = "risk") -> None:
        item = ctx.get(key) if isinstance(ctx.get(key), dict) else {}
        one_day = item.get("change_pct")
        recent = item.get("change_5d_pct")
        date = item.get("date")
        value = f"{_fmt(one_day)}% / {_fmt(recent)}%"
        note_with_date = f"{note}. 1 phiên / 5-7 ngày" + (f"; dữ liệu {date}" if date else "")
        tone = _macro_tone(one_day, recent, mode=tone_mode)
        cards.append(card_fn(label, value, note_with_date, tone))

    pct_card("S&P", "sp500", "Khẩu vị rủi ro của thị trường Mỹ", "risk")
    pct_card("DXY", "dxy", "USD mạnh lên thường gây áp lực cho tài sản rủi ro", "inverse")
    pct_card("Dầu", "oil", "Dầu tăng mạnh có thể làm tăng lo ngại lạm phát/chi phí", "oil")
    pct_card("Vàng", "gold", "Vàng giảm mạnh hoặc tăng sốc đều là tín hiệu risk-off cần chú ý", "haven")
    pct_card("Bitcoin", "btc", "Crypto giao dịch cuối tuần nên có thể báo trước tâm lý risk-off", "risk")
    pct_card("Nikkei", "nikkei", "Tâm lý khu vực Nhật Bản", "risk")
    pct_card("Kospi", "kospi", "Tâm lý khu vực Hàn Quốc; giảm sâu/circuit breaker là tín hiệu xấu", "risk")
    pct_card("Hang Seng", "hsi", "Tâm lý khu vực Hong Kong/Trung Quốc", "risk")

    usd_vnd = m.get("vn_usd_vnd")
    sbv_rate = m.get("vn_sbv_rate")
    if usd_vnd is not None:
        cards.append(card_fn("USD/VND", _fmt(usd_vnd), "Tỷ giá cao/tăng nhanh là áp lực với dòng vốn và tâm lý", "warn"))
    if sbv_rate is not None:
        cards.append(card_fn("SBV rate", f"{_fmt(sbv_rate)}%", "Mặt bằng chính sách tiền tệ trong nước", "warn"))

    if not cards:
        return '<div class="note-box">Chưa có dữ liệu vĩ mô trong lần chạy này.</div>'
    return "\n".join(cards)


def _macro_tone(one_day: Any, recent: Any, *, mode: str) -> str:
    d1 = _as_float_or_none(one_day)
    d5 = _as_float_or_none(recent)
    stress = max(abs(v) for v in [d1, d5] if v is not None) if any(v is not None for v in [d1, d5]) else 0.0
    ref = d5 if d5 is not None else d1
    if ref is None:
        return "warn"
    if mode == "inverse":
        if ref >= 0.7 or (d1 is not None and d1 >= 0.7):
            return "bad"
        if ref <= -0.7:
            return "good"
        return "warn"
    if mode == "oil":
        if ref >= 1.5 or (d1 is not None and d1 >= 1.5):
            return "bad"
        return "warn"
    if mode == "haven":
        if ref <= -1.5 or stress >= 2.5:
            return "bad"
        return "warn"
    if ref <= -1.0 or (d1 is not None and d1 <= -1.0):
        return "bad"
    if ref >= 1.0:
        return "good"
    return "warn"


def _as_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _macro_reading_notes(report: MarketStateReport) -> list[str]:
    m = report.metrics
    ctx = report.global_context
    notes: list[str] = []
    sp500 = _f(m, "global_sp500_change_pct")
    sp500_5d = _f(m, "global_sp500_change_5d_pct")
    dxy = _f(m, "global_dxy_change_pct")
    dxy_5d = _f(m, "global_dxy_change_5d_pct")
    oil = _f(m, "global_oil_change_pct")
    oil_5d = _f(m, "global_oil_change_5d_pct")
    gold = _f(m, "global_gold_change_pct")
    gold_5d = _f(m, "global_gold_change_5d_pct")
    btc = _f(m, "global_btc_change_pct")
    btc_5d = _f(m, "global_btc_change_5d_pct")
    nikkei = _f(m, "global_nikkei_change_pct")
    kospi = _f(m, "global_kospi_change_pct")
    kospi_5d = _f(m, "global_kospi_change_5d_pct")
    hsi = _f(m, "global_hsi_change_pct")
    usd_vnd = m.get("vn_usd_vnd")
    sbv_rate = m.get("vn_sbv_rate")

    if sp500_5d < -1.0:
        notes.append("Mỹ đã yếu trong vài phiên gần đây; nếu VN giảm sau cuối tuần, cần đọc đây là một phần của nhịp risk-off toàn cầu.")
    elif sp500 > 0.5:
        notes.append("S&P phiên gần nhất hồi lại, nhưng vẫn cần so với nhịp vài ngày trước để tránh bỏ sót cú giảm cuối tuần/ngoài giờ VN.")
    elif sp500 < -0.5:
        notes.append("Mỹ đang tạo áp lực risk-off: S&P giảm, tâm lý toàn cầu có thể làm nhà đầu tư trong nước thận trọng hơn.")
    else:
        notes.append("S&P 500 biến động nhẹ, chưa phải nguồn áp lực lớn nhất trong bối cảnh hôm nay.")

    if dxy > 0.5 or dxy_5d > 0.7:
        notes.append("DXY tăng là điểm bất lợi: USD mạnh thường gây áp lực lên tỷ giá, vốn ngoại và nhóm tài sản rủi ro.")
    elif dxy < -0.5 or dxy_5d < -0.7:
        notes.append("DXY hạ nhiệt là điểm hỗ trợ: áp lực USD/tỷ giá bớt căng hơn.")

    asia = [v for v in [nikkei, kospi, hsi] if v != 0]
    if kospi <= -8.0:
        notes.append("KOSPI giảm trên 8% là tín hiệu xấu rõ rệt: thị trường Hàn Quốc đã rơi vào vùng circuit breaker, phản ánh risk-off khu vực rất mạnh.")
    elif kospi_5d <= -5.0:
        notes.append("KOSPI giảm mạnh trong vài phiên gần đây, nhất là nhóm công nghệ/bán dẫn; đây là áp lực khu vực với thị trường Việt Nam.")
    if asia and sum(1 for v in asia if v < -1.0) >= 2:
        notes.append("Châu Á đang yếu trên diện rộng, nên tâm lý khu vực là một yếu tố cần theo dõi cùng VNINDEX.")
    elif asia and sum(1 for v in asia if v > 1.0) >= 2:
        notes.append("Châu Á nhìn chung tích cực, đây là bối cảnh hỗ trợ nếu nội tại thị trường Việt Nam cải thiện.")

    if oil > 1.5 or oil_5d > 2.0:
        notes.append("Dầu tăng tương đối mạnh, có thể làm thị trường nhạy hơn với câu chuyện lạm phát và chi phí đầu vào.")
    if gold_5d < -1.5:
        notes.append("Vàng giảm mạnh trong nhịp gần đây; đây có thể là dấu hiệu bán tài sản trú ẩn/giảm đòn bẩy, không nên đọc là tích cực.")
    elif gold > 1.2:
        notes.append("Vàng tăng mạnh thường cho thấy nhu cầu trú ẩn cao hơn; cần đọc cùng tin địa chính trị và biến động USD.")
    if btc_5d < -3.0:
        notes.append("Bitcoin giảm mạnh qua cuối tuần là tín hiệu risk-off đáng chú ý vì crypto giao dịch khi thị trường Việt Nam đóng cửa.")
    elif btc > 2.0:
        notes.append("Bitcoin hồi lại trong phiên gần nhất, nhưng cần so với nhịp cuối tuần để biết risk-off đã thật sự hạ nhiệt hay chỉ hồi kỹ thuật.")

    if usd_vnd is not None:
        notes.append(f"USD/VND hiện quanh {_fmt(usd_vnd)}; tỷ giá là biến số quan trọng với vốn ngoại và tâm lý nhóm tài chính.")
    if sbv_rate is not None:
        notes.append(f"SBV rate đang ở mức {_fmt(sbv_rate)}%; mặt bằng lãi suất giúp đánh giá áp lực giữa cổ phiếu, tiền mặt và tiền gửi.")

    if not notes:
        notes.append("Chưa có đủ dữ liệu vĩ mô để kết luận bối cảnh bên ngoài; phần trạng thái thị trường vẫn dựa trên dữ liệu VNINDEX và độ rộng thị trường.")
    return notes


def _lane_meta(lane: str) -> dict[str, str]:
    mapping = {
        "PANIC_REBOUND": {
            "label": "Có thể hồi kỹ thuật",
            "headline": "Có nhịp hồi ngắn hạn, vẫn cần thận trọng",
            "color": "#b85d2a",
            "bg": "#fff7df",
        },
        "ACCUMULATION_WATCH": {
            "label": "Tích lũy / chờ xác nhận",
            "headline": "Lập watchlist, chờ tín hiệu rõ hơn",
            "color": "#2f6f73",
            "bg": "#e9f2f7",
        },
        "RISK_OFF_BREAKDOWN": {
            "label": "Risk-off rất xấu",
            "headline": "Thị trường yếu, ưu tiên quan sát",
            "color": "#c24132",
            "bg": "#fff1ed",
        },
        "STRUCTURAL_BEAR": {
            "label": "Bear dài hạn",
            "headline": "Bảo toàn vốn, tăng tiền gửi",
            "color": "#8f1f17",
            "bg": "#fff1ed",
        },
        "DISTRIBUTION_RISK": {
            "label": "Phân phối",
            "headline": "Dòng tiền yếu, cần đọc rủi ro kỹ hơn",
            "color": "#b46a00",
            "bg": "#fff7df",
        },
        "HEALTHY_RISK_ON": {
            "label": "Risk-on",
            "headline": "Bối cảnh thuận lợi hơn",
            "color": "#137333",
            "bg": "#eaf7ee",
        },
        "RISK_OFF_BASELINE": {
            "label": "Risk-off",
            "headline": "Phòng thủ, chờ tín hiệu rõ hơn",
            "color": "#b46a00",
            "bg": "#fff7df",
        },
        "NEUTRAL_CHOP": {
            "label": "Trung tính / nhiễu",
            "headline": "Chỉ chọn lọc, giữ linh hoạt",
            "color": "#22577a",
            "bg": "#e9f2f7",
        },
    }
    return mapping.get(lane, mapping["NEUTRAL_CHOP"])


def _plain_takeaway(report: MarketStateReport) -> str:
    lane = report.market_lane
    m = report.metrics
    if lane == "RISK_OFF_BREAKDOWN":
        return (
            "Thị trường đang xấu thật trong ngắn hạn: giảm mạnh, độ rộng yếu và đóng cửa gần đáy. "
            "Với người mới, đây là ngày nên đọc theo hướng phòng thủ: hiểu rủi ro trước, quan sát cổ phiếu khỏe sau."
        )
    if lane == "PANIC_REBOUND":
        return (
            "Thị trường giảm đủ mạnh để có thể xuất hiện nhịp hồi kỹ thuật. Dù vậy, nhịp hồi kiểu này thường chỉ là dấu hiệu ngắn hạn, "
            "chưa nói rằng thị trường đã khỏe trở lại."
        )
    if lane == "ACCUMULATION_WATCH":
        return (
            "Thị trường chưa khỏe hẳn nhưng đã có dấu hiệu gom nền. Việc hợp lý là lập danh sách cổ tích lũy, "
            "chờ breakout hoặc breadth cải thiện trước khi tăng tỷ trọng."
        )
    if lane == "STRUCTURAL_BEAR":
        return (
            "Rủi ro dài hạn đang lớn hơn cơ hội. Với người mới, nên ưu tiên bảo toàn vốn và giữ một phần tài sản ở tiền mặt hoặc tiền gửi ngắn hạn."
        )
    if lane == "HEALTHY_RISK_ON":
        return "Cấu trúc thị trường thuận lợi hơn: độ rộng và xu hướng cùng tốt lên, nhưng vẫn cần theo dõi rủi ro vĩ mô và biến động bất ngờ."
    return (
        f"VNINDEX {_fmt(m.get('vnindex_ret_1d_pct'))}% trong ngày, market score {_fmt(m.get('mkt_regime_score'))}. "
        "Chưa có lợi thế rõ ràng; ưu tiên quan sát và giữ ngân sách linh hoạt."
    )


def _simple_explanation(report: MarketStateReport) -> list[str]:
    m = report.metrics
    lane = report.market_lane
    down_text = f"VNINDEX giảm {_fmt(abs(_f(m, 'vnindex_ret_1d_pct')))}% trong một phiên."
    weak_breadth = f"Chỉ khoảng {_fmt(m.get('breadth_ma20_pct'))}% cổ phiếu còn giữ được xu hướng ngắn hạn, nghĩa là thị trường yếu trên diện rộng."
    close_location = _f(m, "vnindex_close_location", 0.5)
    close_text = (
        "Cuối phiên chỉ số đóng gần vùng thấp nhất ngày, nên lực mua đỡ giá chưa đủ rõ."
        if close_location < 0.3
        else "Cuối phiên có lực kéo lại nhất định, nên có thể theo dõi khả năng hồi kỹ thuật."
    )
    long_term = (
        "Điểm tích cực nhỏ là xu hướng 60 ngày chưa gãy hoàn toàn, nên chưa gọi đây là bear market dài hạn."
        if _f(m, "vnindex_ret_60d_pct") > -8 and _f(m, "vnindex_distance_ma200_pct") > -5
        else "Xu hướng trung hạn cũng đang yếu, nên cần phòng thủ mạnh hơn."
    )

    if lane == "RISK_OFF_BREAKDOWN":
        return [
            down_text,
            weak_breadth,
            close_text,
            long_term,
            "Kết luận dễ hiểu: hôm nay là ngày ưu tiên bảo vệ tài khoản và quan sát thêm, chưa phải ngày để vội lạc quan.",
        ]
    if lane == "PANIC_REBOUND":
        return [
            down_text,
            "Mức giảm đã đủ mạnh để có thể xuất hiện nhịp hồi kỹ thuật.",
            close_text,
            "Điểm quan trọng là phân biệt hồi kỹ thuật với thị trường khỏe thật; cần thêm dữ liệu ở các phiên sau.",
        ]
    if lane == "ACCUMULATION_WATCH":
        return [
            "Thị trường chưa khỏe hẳn, nhưng không còn xấu đồng loạt như giai đoạn bán tháo.",
            "Một số cổ phiếu bắt đầu có dấu hiệu tích lũy, phù hợp để đưa vào watchlist.",
            "Việc chính lúc này là chờ điểm nổ hoặc chờ thị trường xác nhận hồi phục rõ hơn.",
        ]
    if lane == "STRUCTURAL_BEAR":
        return [
            "Không chỉ một phiên xấu: cấu trúc trung hạn cũng đang yếu.",
            weak_breadth,
            "Ưu tiên bảo toàn vốn, giảm giao dịch và chuyển một phần tiền nhàn rỗi sang tiền gửi/kỳ hạn ngắn.",
        ]
    if lane == "DISTRIBUTION_RISK":
        return [
            "Thị trường có dấu hiệu bị bán ra/phân phối, tức là tiền lớn chưa ủng hộ chiều mua.",
            "Nhịp hồi trong bối cảnh này dễ chỉ là hồi yếu nếu độ rộng thị trường không cải thiện.",
            "Ưu tiên giảm rủi ro và giữ danh sách theo dõi.",
        ]
    if lane == "HEALTHY_RISK_ON":
        return [
            "Độ rộng thị trường và xu hướng chính đang ủng hộ bên mua.",
            "Có thể vận hành các chiến lược chính, nhưng vẫn cần giữ kỷ luật vị thế.",
        ]
    return [
        "Thị trường chưa cho tín hiệu đủ rõ để tăng rủi ro.",
        "Cách xử lý hợp lý là giữ tỷ trọng linh hoạt và chờ thêm xác nhận.",
    ]


def _caution_note(report: MarketStateReport) -> str:
    lane = report.market_lane
    if lane == "RISK_OFF_BREAKDOWN":
        return (
            "Mức thận trọng rất cao: nên đọc thị trường như một ngày risk-off rõ rệt. "
            "Trang này chỉ cung cấp bối cảnh và mức cảnh báo, không phải tín hiệu mua bán."
        )
    if lane == "PANIC_REBOUND":
        return (
            "Có khả năng xuất hiện nhịp hồi kỹ thuật, nhưng cần hiểu đây là tín hiệu ngắn hạn. "
            "Chỉ khi độ rộng, giá đóng cửa và dòng tiền cùng tốt lên thì bối cảnh mới đáng tin hơn."
        )
    if lane == "ACCUMULATION_WATCH":
        return (
            "Thị trường phù hợp để lập watchlist và theo dõi nhóm tích lũy. "
            "Điểm quan trọng là chờ dữ liệu xác nhận, không kết luận chỉ từ một phiên."
        )
    if lane == "STRUCTURAL_BEAR":
        return (
            "Mức phòng thủ cao: ưu tiên bảo toàn vốn và cân nhắc để một phần tài sản ở tiền mặt hoặc tiền gửi ngắn hạn. "
            "Đây là cách quản trị rủi ro cho người mới, không phải khuyến nghị cá nhân."
        )
    if lane == "HEALTHY_RISK_ON":
        return (
            "Bối cảnh thuận lợi hơn: xu hướng và độ rộng ủng hộ thị trường. "
            "Dù vậy vẫn cần theo dõi lịch vĩ mô, biến động quốc tế và rủi ro bất ngờ."
        )
    return (
        "Giữ cách đọc linh hoạt: thị trường chưa nghiêng hẳn về tốt hay xấu. "
        "Nên theo dõi thêm vài phiên để tránh kết luận quá sớm."
    )


def _retail_reading_notes(report: MarketStateReport) -> list[str]:
    lane = report.market_lane
    notes = [
        "Đọc phần đầu trang trước: nó cho biết hôm nay thị trường đang ở trạng thái nào.",
        "Dùng màu như tín hiệu cảnh báo: đỏ là thận trọng, vàng là chờ thêm, xanh là thuận lợi hơn.",
    ]
    if lane == "RISK_OFF_BREAKDOWN":
        notes.extend(
            [
                "Với người mới, ngày như hôm nay nên tập trung hiểu vì sao thị trường yếu thay vì vội tìm cơ hội.",
                "Theo dõi các phiên sau xem chỉ số có đóng cửa tốt hơn không và số cổ phiếu khỏe có tăng lên không.",
            ]
        )
    elif lane == "PANIC_REBOUND":
        notes.extend(
            [
                "Nếu có hồi sau một phiên giảm mạnh, hãy xem đó là nhịp hồi kỹ thuật trước khi gọi là thị trường khỏe.",
                "Tín hiệu tốt hơn cần đi kèm độ rộng cải thiện và đóng cửa không còn sát đáy.",
            ]
        )
    elif lane == "ACCUMULATION_WATCH":
        notes.extend(
            [
                "Đây là giai đoạn phù hợp để ghi lại các nhóm ngành và cổ phiếu giữ giá tốt hơn thị trường.",
                "Chờ thêm xác nhận từ độ rộng và dòng tiền trước khi nâng mức tin cậy.",
            ]
        )
    elif lane == "STRUCTURAL_BEAR":
        notes.extend(
            [
                "Khi rủi ro dài hạn cao, mục tiêu chính là tránh quyết định vội và giữ kế hoạch tài chính linh hoạt.",
                "Các kênh an toàn hơn như tiền mặt hoặc tiền gửi ngắn hạn có thể được dùng để giảm biến động danh mục.",
            ]
        )
    else:
        notes.extend(
            [
                "So sánh trạng thái hôm nay với vài phiên gần nhất để biết thị trường đang tốt lên hay xấu đi.",
                "Đừng đọc một chỉ báo đơn lẻ; hãy nhìn cùng lúc xu hướng, độ rộng, tin tức và vĩ mô.",
            ]
        )
    notes.append("Phần chi tiết kỹ thuật ở cuối trang dành cho người muốn kiểm tra sâu hơn.")
    return notes


def _lane_label(lane: str) -> str:
    return {
        "PANIC_REBOUND": "Bắt hồi kỹ thuật",
        "ACCUMULATION_WATCH": "Gom nền chờ nổ",
        "RISK_OFF_BREAKDOWN": "Risk-off breakdown",
        "STRUCTURAL_BEAR": "Bear dài hạn",
        "DISTRIBUTION_RISK": "Phân phối",
        "HEALTHY_RISK_ON": "Risk-on khỏe",
        "RISK_OFF_BASELINE": "Risk-off thường",
        "NEUTRAL_CHOP": "Trung tính/nhiễu",
    }.get(lane, lane)


def _lane_class(lane: str) -> str:
    if lane in {"RISK_OFF_BREAKDOWN", "STRUCTURAL_BEAR"}:
        return "risk"
    if lane in {"PANIC_REBOUND", "DISTRIBUTION_RISK", "RISK_OFF_BASELINE"}:
        return "warning"
    if lane in {"HEALTHY_RISK_ON", "ACCUMULATION_WATCH"}:
        return "positive"
    return "neutral"


def _signed_class(value: Any) -> str:
    try:
        return "pos" if float(value) >= 0 else "neg"
    except Exception:
        return ""


def _metric_label(key: str) -> str:
    labels = {
        "vnindex_close": "VNINDEX close",
        "vnindex_ret_1d_pct": "VNINDEX 1 ngày (%)",
        "vnindex_ret_3d_pct": "VNINDEX 3 ngày (%)",
        "vnindex_ret_20d_pct": "VNINDEX 20 ngày (%)",
        "vnindex_ret_60d_pct": "VNINDEX 60 ngày (%)",
        "vnindex_close_location": "Đóng cửa trong biên độ nến",
        "vnindex_volume_ratio_20": "Volume / MA20",
        "mkt_regime_state": "Trạng thái market",
        "mkt_regime_score": "Market score",
        "mkt_CHDM20": "CHDM20",
        "mkt_CHDM50": "CHDM50",
        "mkt_DS20": "DS20",
        "breadth_ma20_pct": "Breadth MA20 (%)",
        "breadth_ma50_pct": "Breadth MA50 (%)",
        "avg_smart_money_score": "Smart money score TB",
        "accumulation_ratio_pct": "Tỷ lệ tích lũy (%)",
        "distribution_ratio_pct": "Tỷ lệ phân phối (%)",
    }
    return labels.get(key, key)


def _load_index(repo: Path, target: pd.Timestamp) -> pd.DataFrame:
    index = pd.read_parquet(repo / "multiagents_trading_assistant" / "data" / "index_master.parquet")
    index["date"] = pd.to_datetime(index["date"]).dt.normalize()
    if "symbol" in index.columns:
        index = index[index["symbol"].astype(str).str.upper().eq("VNINDEX")]
    index = index[index["date"].le(target)].sort_values("date").copy()
    if index.empty:
        raise ValueError("index_master.parquet has no VNINDEX rows up to requested date")
    return index


def _build_metrics(
    features: pd.DataFrame,
    feature_history: pd.DataFrame,
    index: pd.DataFrame,
    target: pd.Timestamp,
) -> dict[str, Any]:
    latest_idx = index.iloc[-1]
    close = pd.to_numeric(index["close"], errors="coerce")
    volume = pd.to_numeric(index.get("volume", pd.Series(index=close.index)), errors="coerce")
    ma20 = close.rolling(20, min_periods=5).mean()
    ma50 = close.rolling(50, min_periods=10).mean()
    ma200 = close.rolling(200, min_periods=40).mean()
    vol20 = volume.rolling(20, min_periods=5).mean()
    idx_range = (pd.to_numeric(index["high"], errors="coerce") - pd.to_numeric(index["low"], errors="coerce")).replace(0, pd.NA)
    close_location = ((pd.to_numeric(index["close"], errors="coerce") - pd.to_numeric(index["low"], errors="coerce")) / idx_range).clip(0, 1)

    m = {
        "date": target.date().isoformat(),
        "vnindex_close": _round(latest_idx.get("close")),
        "vnindex_open": _round(latest_idx.get("open")),
        "vnindex_low": _round(latest_idx.get("low")),
        "vnindex_high": _round(latest_idx.get("high")),
        "vnindex_ret_1d_pct": _pct(close.pct_change().iloc[-1]),
        "vnindex_ret_3d_pct": _pct(close.pct_change(3).iloc[-1]),
        "vnindex_ret_20d_pct": _pct(close.pct_change(20).iloc[-1]),
        "vnindex_ret_60d_pct": _pct(close.pct_change(60).iloc[-1]),
        "vnindex_above_ma20": bool(close.iloc[-1] > ma20.iloc[-1]) if pd.notna(ma20.iloc[-1]) else None,
        "vnindex_above_ma50": bool(close.iloc[-1] > ma50.iloc[-1]) if pd.notna(ma50.iloc[-1]) else None,
        "vnindex_above_ma200": bool(close.iloc[-1] > ma200.iloc[-1]) if pd.notna(ma200.iloc[-1]) else None,
        "vnindex_distance_ma20_pct": _pct(close.iloc[-1] / ma20.iloc[-1] - 1) if pd.notna(ma20.iloc[-1]) else None,
        "vnindex_distance_ma50_pct": _pct(close.iloc[-1] / ma50.iloc[-1] - 1) if pd.notna(ma50.iloc[-1]) else None,
        "vnindex_distance_ma200_pct": _pct(close.iloc[-1] / ma200.iloc[-1] - 1) if pd.notna(ma200.iloc[-1]) else None,
        "vnindex_volume_ratio_20": _round(volume.iloc[-1] / vol20.iloc[-1]) if pd.notna(vol20.iloc[-1]) and vol20.iloc[-1] else None,
        "vnindex_close_location": _round(close_location.iloc[-1], 4),
    }

    m.update(
        {
            "mkt_regime_score": _median(features, "mkt_regime_score"),
            "mkt_regime_state": _mode(features, "mkt_regime_state"),
            "mkt_CHDM20": _median(features, "mkt_CHDM20"),
            "mkt_CHDM50": _median(features, "mkt_CHDM50"),
            "mkt_DS20": _median(features, "mkt_DS20"),
            "mkt_DS50": _median(features, "mkt_DS50"),
            "mkt_CHDM20_delta": _median(features, "mkt_CHDM20_delta"),
            "mkt_CHDM50_delta": _median(features, "mkt_CHDM50_delta"),
            "mkt_DS20_delta": _median(features, "mkt_DS20_delta"),
            "breadth_ma20_pct": _pct(features.get("above_ma20", pd.Series(dtype=bool)).mean()),
            "breadth_ma50_pct": _pct(features.get("above_ma50", pd.Series(dtype=bool)).mean()),
            "breadth_ma200_pct": _pct(features.get("above_ma200", pd.Series(dtype=bool)).mean()),
            "avg_smart_money_score": _mean(features, "smart_money_score"),
            "avg_edge_score": _mean(features, "edge_score"),
            "avg_value_ratio_20": _mean(features, "value_ratio_20"),
            "accumulation_ratio_pct": _pct((pd.to_numeric(features.get("accumulation_days_10", 0), errors="coerce") >= 2).mean()),
            "distribution_ratio_pct": _pct((pd.to_numeric(features.get("distribution_days_10", 0), errors="coerce") >= 2).mean()),
            "distribution_pressure": _mean(features, "distribution_days_10"),
            "hot_state_count": int(features.get("smart_money_state", pd.Series(dtype=str)).astype(str).str.contains("HOT").sum()),
            "reset_in_uptrend_count": int(features.get("smart_money_state", pd.Series(dtype=str)).astype(str).eq("RESET_IN_UPTREND").sum()),
            "breakout_20_count": int(features.get("breakout_20", pd.Series(dtype=bool)).fillna(False).sum()),
            "strong_rs_count": int((pd.to_numeric(features.get("rs_percentile_20", 0), errors="coerce") >= 0.8).sum()),
        }
    )

    previous_days = _daily_breadth_history(feature_history)
    if len(previous_days) >= 2:
        m["breadth_ma20_delta_10d"] = _round(previous_days["breadth_ma20"].iloc[-1] - previous_days["breadth_ma20"].shift(10).iloc[-1], 4)
        m["breadth_ma50_delta_20d"] = _round(previous_days["breadth_ma50"].iloc[-1] - previous_days["breadth_ma50"].shift(20).iloc[-1], 4)
    return m


def _daily_breadth_history(features: pd.DataFrame) -> pd.DataFrame:
    return (
        features.groupby("date", sort=True)
        .agg(
            breadth_ma20=("above_ma20", "mean"),
            breadth_ma50=("above_ma50", "mean"),
        )
        .reset_index()
    )


def _macro_metrics(global_ctx: dict[str, Any], vn_macro_ctx: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ["sp500", "dxy", "oil", "gold", "btc", "nikkei", "kospi", "hsi"]:
        value = global_ctx.get(key)
        if isinstance(value, dict):
            out[f"global_{key}_change_pct"] = value.get("change_pct")
            out[f"global_{key}_change_5d_pct"] = value.get("change_5d_pct")
            out[f"global_{key}_date"] = value.get("date")
    if vn_macro_ctx:
        out["vn_usd_vnd"] = vn_macro_ctx.get("usd_vnd")
        out["vn_sbv_rate"] = vn_macro_ctx.get("sbv_rate")
    return out


def _score_lanes(m: dict[str, Any]) -> dict[str, float]:
    ret1 = _f(m, "vnindex_ret_1d_pct")
    ret3 = _f(m, "vnindex_ret_3d_pct")
    ret20 = _f(m, "vnindex_ret_20d_pct")
    ret60 = _f(m, "vnindex_ret_60d_pct")
    score = _f(m, "mkt_regime_score")
    chdm20 = _f(m, "mkt_CHDM20")
    chdm50 = _f(m, "mkt_CHDM50")
    ds20 = _f(m, "mkt_DS20")
    ds_delta = _f(m, "mkt_DS20_delta")
    breadth20 = _f(m, "breadth_ma20_pct") / 100.0
    breadth50 = _f(m, "breadth_ma50_pct") / 100.0
    breadth20_delta = _f(m, "breadth_ma20_delta_10d")
    vol_ratio = _f(m, "vnindex_volume_ratio_20", 1.0)
    close_loc = _f(m, "vnindex_close_location", 0.5)
    dist_ma50 = _f(m, "vnindex_distance_ma50_pct")
    dist_ma200 = _f(m, "vnindex_distance_ma200_pct")
    smt = _f(m, "avg_smart_money_score", 50.0)
    accumulation = _f(m, "accumulation_ratio_pct")
    distribution = _f(m, "distribution_ratio_pct")
    sp500 = _f(m, "global_sp500_change_pct")
    dxy = _f(m, "global_dxy_change_pct")
    global_risk = _f(m, "global_risk_score")
    system_risk = _f(m, "system_overall_risk_score")

    panic_rebound = 0.0
    panic_rebound += 24.0 if ret1 <= -2.0 else 12.0 if ret1 <= -1.2 else 0.0
    panic_rebound += 16.0 if ret3 <= -4.0 else 8.0 if ret3 <= -2.5 else 0.0
    panic_rebound += 12.0 if vol_ratio >= 1.25 else 0.0
    panic_rebound += 12.0 if close_loc >= 0.45 else 0.0
    panic_rebound += 12.0 if breadth20 <= 0.25 else 6.0 if breadth20 <= 0.35 else 0.0
    panic_rebound += 10.0 if -8.0 <= dist_ma50 <= -2.0 else 0.0
    panic_rebound -= 16.0 if ret60 <= -10.0 and dist_ma200 <= -6.0 else 0.0

    accumulation_watch = 0.0
    accumulation_watch += 18.0 if score >= 32.0 else 6.0
    accumulation_watch += 16.0 if chdm20 >= 45.0 or _f(m, "mkt_CHDM20_delta") > 0 else 0.0
    accumulation_watch += 14.0 if ds20 <= 0.45 or ds_delta < 0 else 0.0
    accumulation_watch += 14.0 if accumulation >= distribution and accumulation >= 15.0 else 0.0
    accumulation_watch += 12.0 if smt >= 45.0 else 0.0
    accumulation_watch += 10.0 if breadth20_delta >= -0.05 else 0.0
    accumulation_watch += 8.0 if ret20 > -6.0 else 0.0

    structural_bear = 0.0
    structural_bear += 20.0 if score < 35.0 else 8.0 if score < 45.0 else 0.0
    structural_bear += 18.0 if ret60 <= -8.0 else 0.0
    structural_bear += 16.0 if dist_ma200 <= -5.0 else 8.0 if dist_ma200 <= -2.0 else 0.0
    structural_bear += 14.0 if breadth50 <= 0.35 else 0.0
    structural_bear += 12.0 if ds20 >= 0.55 else 0.0
    structural_bear += 10.0 if chdm50 < 40.0 else 0.0
    structural_bear += 8.0 if sp500 < -1.0 or dxy > 0.8 else 0.0
    structural_bear += 8.0 if system_risk >= 75 and ret60 <= -5.0 else 0.0

    distribution_risk = 0.0
    distribution_risk += 18.0 if distribution >= 20.0 else 8.0 if distribution >= 12.0 else 0.0
    distribution_risk += 16.0 if ds_delta > 0.04 else 0.0
    distribution_risk += 14.0 if breadth20_delta < -0.15 else 0.0
    distribution_risk += 12.0 if ret20 > 0 and ret1 < -1.0 else 0.0
    distribution_risk += 10.0 if vol_ratio >= 1.2 and close_loc < 0.40 else 0.0
    distribution_risk += 8.0 if global_risk >= 70.0 and ret1 < -1.0 else 0.0

    healthy_risk_on = 0.0
    healthy_risk_on += 22.0 if score >= 60.0 else 0.0
    healthy_risk_on += 18.0 if ret20 > 0 and ret60 > 0 else 0.0
    healthy_risk_on += 16.0 if breadth50 >= 0.55 else 0.0
    healthy_risk_on += 12.0 if chdm50 >= 55.0 and ds20 <= 0.35 else 0.0
    healthy_risk_on += 10.0 if dist_ma50 > 0 else 0.0
    healthy_risk_on -= 20.0 if global_risk >= 70.0 else 0.0
    healthy_risk_on -= 12.0 if system_risk >= 60.0 else 0.0

    risk_off_breakdown = 0.0
    risk_off_breakdown += 18.0 if score < 20.0 else 0.0
    risk_off_breakdown += 16.0 if breadth20 <= 0.20 else 0.0
    risk_off_breakdown += 14.0 if breadth50 <= 0.30 else 0.0
    risk_off_breakdown += 14.0 if ds20 >= 0.65 else 0.0
    risk_off_breakdown += 12.0 if chdm20 < 25.0 else 0.0
    risk_off_breakdown += 10.0 if ret1 <= -2.0 else 0.0
    risk_off_breakdown += 8.0 if close_loc < 0.30 else 0.0
    risk_off_breakdown += 8.0 if ret60 > -8.0 and dist_ma200 > -5.0 else 0.0
    risk_off_breakdown += 10.0 if global_risk >= 70.0 and ret1 <= -1.5 else 0.0
    risk_off_breakdown += 8.0 if system_risk >= 70.0 else 0.0

    return {
        "PANIC_REBOUND": max(0.0, min(100.0, panic_rebound)),
        "ACCUMULATION_WATCH": max(0.0, min(100.0, accumulation_watch)),
        "RISK_OFF_BREAKDOWN": max(0.0, min(100.0, risk_off_breakdown)),
        "STRUCTURAL_BEAR": max(0.0, min(100.0, structural_bear)),
        "DISTRIBUTION_RISK": max(0.0, min(100.0, distribution_risk)),
        "HEALTHY_RISK_ON": max(0.0, min(100.0, healthy_risk_on)),
    }


def _choose_lane(scores: dict[str, float], m: dict[str, Any]) -> str:
    if scores["STRUCTURAL_BEAR"] >= 58 and scores["PANIC_REBOUND"] < 60:
        return "STRUCTURAL_BEAR"
    if scores["PANIC_REBOUND"] >= 58:
        return "PANIC_REBOUND"
    if scores.get("RISK_OFF_BREAKDOWN", 0.0) >= 58:
        return "RISK_OFF_BREAKDOWN"
    if scores["DISTRIBUTION_RISK"] >= 55:
        return "DISTRIBUTION_RISK"
    if scores["ACCUMULATION_WATCH"] >= 55:
        return "ACCUMULATION_WATCH"
    if scores["HEALTHY_RISK_ON"] >= 55:
        return "HEALTHY_RISK_ON"
    state = str(m.get("mkt_regime_state") or "")
    return "RISK_OFF_BASELINE" if state == "RISK_OFF" else "NEUTRAL_CHOP"


def _allocation_for_lane(lane: str, m: dict[str, Any], scores: dict[str, float]) -> AllocationHint:
    if lane == "PANIC_REBOUND":
        return AllocationHint(10, 10, 10, 50, 20, "Chỉ dùng sleeve nhỏ cho nhịp hồi; không nâng core khi xu hướng chính chưa xác nhận.")
    if lane == "ACCUMULATION_WATCH":
        return AllocationHint(15, 0, 15, 45, 25, "Ưu tiên gom quan sát và chờ breakout xác nhận; tiền nhàn rỗi dài hơn có thể đưa sang tiết kiệm.")
    if lane == "STRUCTURAL_BEAR":
        return AllocationHint(0, 0, 5, 35, 60, "Thị trường yếu mang tính cấu trúc; bảo toàn vốn và kỳ hạn tiền gửi cao hơn cổ phiếu.")
    if lane == "RISK_OFF_BREAKDOWN":
        return AllocationHint(0, 0, 5, 55, 40, "Risk-off rất xấu trong ngắn hạn nhưng chưa xác nhận bear dài hạn; giữ tiền và đợi reversal evidence.")
    if lane == "DISTRIBUTION_RISK":
        return AllocationHint(5, 0, 5, 55, 35, "Giảm rủi ro vì dòng tiền phân phối; chỉ giữ danh sách theo dõi.")
    if lane == "HEALTHY_RISK_ON":
        return AllocationHint(55, 0, 20, 20, 5, "Có thể vận hành chiến lược chính với kiểm soát vị thế bình thường.")
    if lane == "RISK_OFF_BASELINE":
        return AllocationHint(5, 0, 10, 55, 30, "Risk-off nhưng chưa đủ điều kiện bắt hồi hoặc xác nhận bear dài hạn.")
    return AllocationHint(20, 0, 10, 50, 20, "Thị trường trung tính/nhiễu; giữ ngân sách linh hoạt.")


def _reasons_for_lane(lane: str, m: dict[str, Any], scores: dict[str, float]) -> list[str]:
    reasons = [
        f"VNINDEX 1D {_fmt(m.get('vnindex_ret_1d_pct'))}%, 20D {_fmt(m.get('vnindex_ret_20d_pct'))}%, 60D {_fmt(m.get('vnindex_ret_60d_pct'))}%.",
        f"Market score {_fmt(m.get('mkt_regime_score'))}, state {m.get('mkt_regime_state')}.",
        f"Breadth MA20 {_fmt(m.get('breadth_ma20_pct'))}%, MA50 {_fmt(m.get('breadth_ma50_pct'))}%.",
        f"CHDM20 {_fmt(m.get('mkt_CHDM20'))}, DS20 {_fmt(m.get('mkt_DS20'))}, Smart Money avg {_fmt(m.get('avg_smart_money_score'))}.",
    ]
    if lane == "PANIC_REBOUND":
        reasons.append("Giảm mạnh kèm điều kiện washout/oversold đủ để theo dõi nhịp hồi, nhưng vẫn là tactical lane.")
    elif lane == "ACCUMULATION_WATCH":
        reasons.append("Dòng tiền/độ rộng có dấu hiệu ổn định tương đối; ưu tiên watchlist gom nền và chờ nổ.")
    elif lane == "STRUCTURAL_BEAR":
        reasons.append("Xu hướng trung hạn yếu, breadth hẹp và/hoặc VNINDEX nằm xa dưới các MA dài hơn.")
    elif lane == "RISK_OFF_BREAKDOWN":
        reasons.append("Phiên giảm đóng gần đáy, breadth rất hẹp và DS cao; chưa đủ điều kiện bắt hồi dù chưa phải bear dài hạn.")
    elif lane == "DISTRIBUTION_RISK":
        reasons.append("Áp lực phân phối hoặc độ rộng xấu đi nhanh; tránh bắt dao rơi.")
    elif lane == "HEALTHY_RISK_ON":
        reasons.append("Cấu trúc thị trường đủ khỏe để cho phép sleeve chính hoạt động.")
    reasons.append(
        "Lane scores: "
        + ", ".join(f"{k}={v:.1f}" for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True))
    )
    return reasons


def _watch_actions(lane: str, m: dict[str, Any], allocation: AllocationHint) -> list[str]:
    actions = [
        "Không phát sinh broker order từ report này; dùng như lớp điều phối ngân sách/rủi ro.",
        "Core/Flow chỉ nên chạy khi market lane cho phép hoặc strategy riêng có risk gate độc lập.",
    ]
    if lane == "PANIC_REBOUND":
        actions.extend(
            [
                "Tìm cổ phiếu RS cao, giảm ít hơn thị trường, hoặc có close_location tốt sau phiên washout.",
                "Nhịp hồi chỉ nên dùng tỷ trọng nhỏ và có stop rõ; không bình quân xuống cổ yếu.",
            ]
        )
    elif lane == "ACCUMULATION_WATCH":
        actions.extend(
            [
                "Lọc cổ phiếu tích lũy: DS giảm, CHDM/SMT cải thiện, giá co hẹp gần MA20/MA50.",
                "Chờ tín hiệu nổ: breakout nền + value_ratio_20 tăng + market score hồi lại.",
            ]
        )
    elif lane in {"STRUCTURAL_BEAR", "RISK_OFF_BREAKDOWN", "DISTRIBUTION_RISK", "RISK_OFF_BASELINE"}:
        actions.extend(
            [
                "Ưu tiên tiền mặt, tiền gửi/kỳ hạn ngắn, và giảm tần suất trade.",
                "Chỉ giữ discovery/watchlist để chuẩn bị khi breadth hoặc CHDM quay đầu.",
            ]
        )
    else:
        actions.append("Có thể dùng ngân sách core theo rule hiện tại, nhưng vẫn giới hạn theo drawdown/account risk.")
    return actions


def _sector_snapshot(features: pd.DataFrame) -> list[dict[str, Any]]:
    if "industry" not in features.columns:
        return []
    out = (
        features.dropna(subset=["industry"])
        .groupby("industry", sort=False)
        .agg(
            count=("symbol", "count"),
            avg_ret_20d_pct=("ret_20d_local", lambda s: _pct(pd.to_numeric(s, errors="coerce").mean())),
            avg_edge_score=("edge_score", "mean"),
            avg_smart_money_score=("smart_money_score", "mean"),
        )
        .reset_index()
    )
    out = out.sort_values(["avg_edge_score", "avg_ret_20d_pct"], ascending=False)
    return [_json_safe(row) for row in out.head(12).to_dict("records")]


def _load_live_context(target: pd.Timestamp, *, news_limit: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    global_ctx: dict[str, Any] = {}
    vn_macro_ctx: dict[str, Any] = {}
    news_ctx: dict[str, Any] = {}
    try:
        from multiagents_trading_assistant.fetcher import get_global_macro, get_vn_macro

        global_ctx = get_global_macro() or {}
        vn_macro_ctx = get_vn_macro() or {}
    except Exception as exc:
        vn_macro_ctx["error"] = f"{type(exc).__name__}: {exc}"
    try:
        from multiagents_trading_assistant import news_fetcher

        items = news_fetcher.get_expert_news(date=target.date().isoformat(), max_articles=news_limit) or []
        news_ctx = {
            "news_count": len(items),
            "headlines": [
                {
                    "source": item.get("source"),
                    "headline": item.get("headline"),
                    "reliability": item.get("reliability"),
                    "url": item.get("url"),
                }
                for item in items[:news_limit]
            ],
        }
    except Exception as exc:
        news_ctx = {"news_count": 0, "error": f"{type(exc).__name__}: {exc}", "headlines": []}
    return global_ctx, vn_macro_ctx, news_ctx


def _action_bias(lane: str) -> str:
    return {
        "PANIC_REBOUND": "TACTICAL_REBOUND_ONLY",
        "ACCUMULATION_WATCH": "WATCH_ACCUMULATION_AND_WAIT_CONFIRMATION",
        "STRUCTURAL_BEAR": "CAPITAL_PRESERVATION",
        "RISK_OFF_BREAKDOWN": "DEFENSIVE_WAIT_FOR_REVERSAL_EVIDENCE",
        "DISTRIBUTION_RISK": "REDUCE_RISK",
        "HEALTHY_RISK_ON": "ALLOW_CORE_STRATEGIES",
        "RISK_OFF_BASELINE": "DEFENSIVE_WAIT",
        "NEUTRAL_CHOP": "SELECTIVE_ONLY",
    }.get(lane, "SELECTIVE_ONLY")


def _risk_summary(lane: str, m: dict[str, Any]) -> str:
    if lane == "PANIC_REBOUND":
        return "Risk-off có thể quá bán ngắn hạn; hồi kỹ thuật có xác suất nhưng không coi là đảo trend."
    if lane == "ACCUMULATION_WATCH":
        return "Risk-off/neutral nhưng có dấu hiệu gom nền; ưu tiên quan sát cổ khỏe hơn thị trường."
    if lane == "STRUCTURAL_BEAR":
        return "Rủi ro xu hướng dài hơn đang chiếm ưu thế; giảm equity budget và tăng cash/savings."
    if lane == "RISK_OFF_BREAKDOWN":
        return "Risk-off ngắn hạn rất xấu: giảm mạnh, breadth yếu, đóng gần đáy; chưa nên bắt hồi khi chưa có reversal evidence."
    if lane == "DISTRIBUTION_RISK":
        return "Áp lực phân phối làm giảm chất lượng tín hiệu; tránh mở vị thế mới trừ setup đặc biệt."
    if lane == "HEALTHY_RISK_ON":
        return "Cấu trúc thị trường thuận lợi tương đối; vẫn áp dụng quản trị rủi ro bình thường."
    return "Chưa có edge rõ cho risk-on hoặc bắt hồi; giữ phòng thủ."


def _median(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    value = pd.to_numeric(df[col], errors="coerce").median()
    return _round(value)


def _mean(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    value = pd.to_numeric(df[col], errors="coerce").mean()
    return _round(value)


def _mode(df: pd.DataFrame, col: str) -> str | None:
    if col not in df.columns or df[col].dropna().empty:
        return None
    return str(df[col].mode().iloc[0])


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def _pct(value: Any) -> float | None:
    rounded = _round(value)
    return None if rounded is None else round(rounded * 100.0, 2)


def _f(m: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = m.get(key, default)
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
