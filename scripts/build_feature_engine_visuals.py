"""Build an offline visual page for the layered demo feature engine.

The dashboard is a static file export, so this script writes a self-contained
HTML page with inline SVG charts. It intentionally avoids external JS/CSS.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "reports" / "layered_pipeline_demo_2026-05-17"
DATA_DIR = ROOT / "multiagents_trading_assistant" / "data"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(str(value).replace(",", "").replace("%", "").strip())
    except Exception:
        return default


def _scale(values: list[float], lo: float, hi: float) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [(lo + hi) / 2 for _ in values]
    return [hi - ((value - vmin) / (vmax - vmin)) * (hi - lo) for value in values]


def _bar_chart(frame: pd.DataFrame, metric: str, title: str, *, color: str, max_items: int = 15) -> str:
    if frame.empty or metric not in frame.columns:
        return "<p class='note'>No data.</p>"
    rows = frame[["symbol", metric]].copy()
    rows[metric] = rows[metric].map(_num)
    rows = rows.sort_values(metric, ascending=False).head(max_items)
    width = 920
    row_h = 30
    left = 96
    top = 34
    chart_w = 760
    height = top + len(rows) * row_h + 34
    max_v = max(rows[metric].max(), 1)
    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>",
        f"<text x='0' y='18' class='chart-title'>{html.escape(title)}</text>",
    ]
    for i, row in enumerate(rows.to_dict("records")):
        y = top + i * row_h
        symbol = str(row["symbol"])
        value = float(row[metric])
        bar_w = max(2, value / max_v * chart_w)
        parts.extend(
            [
                f"<text x='0' y='{y + 18}' class='axis'>{html.escape(symbol)}</text>",
                f"<rect x='{left}' y='{y}' width='{bar_w:.1f}' height='19' rx='3' fill='{color}' />",
                f"<text x='{left + bar_w + 8:.1f}' y='{y + 15}' class='value'>{value:.2f}</text>",
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def _multi_metric_heatmap(frame: pd.DataFrame) -> str:
    metrics = [
        ("edge_rank_score", "Rank"),
        ("edge_score", "Edge"),
        ("rs_percentile_20", "RS20"),
        ("smart_money_score", "SMT"),
        ("value_ratio_20", "Value"),
        ("mkt_regime_score", "Regime"),
    ]
    present = [(key, label) for key, label in metrics if key in frame.columns]
    if frame.empty or not present:
        return "<p class='note'>No data.</p>"
    rows = frame.copy()
    rows["edge_rank_score"] = rows.get("edge_rank_score", 0).map(_num)
    rows = rows.sort_values("edge_rank_score", ascending=False).head(15)
    cell_w = 96
    cell_h = 30
    left = 96
    top = 42
    width = left + len(present) * cell_w + 20
    height = top + len(rows) * cell_h + 24
    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Feature heatmap'>",
        "<text x='0' y='18' class='chart-title'>Feature Heatmap</text>",
    ]
    for j, (_key, label) in enumerate(present):
        parts.append(f"<text x='{left + j * cell_w + 8}' y='36' class='axis'>{html.escape(label)}</text>")
    for i, row in enumerate(rows.to_dict("records")):
        y = top + i * cell_h
        parts.append(f"<text x='0' y='{y + 20}' class='axis'>{html.escape(str(row.get('symbol', '')))}</text>")
        for j, (key, _label) in enumerate(present):
            value = _num(row.get(key))
            if key == "rs_percentile_20":
                normalized = max(0.0, min(1.0, value))
            elif key == "value_ratio_20":
                normalized = max(0.0, min(1.0, value / 3.0))
            elif key == "edge_rank_score":
                normalized = max(0.0, min(1.0, value / 4.0))
            else:
                normalized = max(0.0, min(1.0, value / 100.0))
            red = int(230 - normalized * 150)
            green = int(90 + normalized * 120)
            blue = int(80 + normalized * 40)
            x = left + j * cell_w
            parts.append(f"<rect x='{x}' y='{y}' width='{cell_w - 6}' height='24' rx='4' fill='rgb({red},{green},{blue})' />")
            parts.append(f"<text x='{x + 8}' y='{y + 17}' class='heat-value'>{value:.2f}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _price_line_chart(symbol: str, ohlcv: pd.DataFrame) -> str:
    rows = ohlcv[ohlcv["symbol"].astype(str).str.upper().eq(symbol)].sort_values("date").tail(30).copy()
    if rows.empty:
        return "<p class='note'>No OHLCV data.</p>"
    closes = [_num(v) for v in rows["close"].tolist()]
    volumes = [_num(v) for v in rows.get("volume", pd.Series([0] * len(rows))).tolist()]
    width = 920
    height = 270
    left = 48
    right = 16
    top = 34
    mid = 170
    bottom = 238
    xs = [left + i * ((width - left - right) / max(len(closes) - 1, 1)) for i in range(len(closes))]
    ys = _scale(closes, top, mid)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    vol_max = max(volumes) if volumes else 1
    parts = [
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(symbol)} price line'>",
        f"<text x='0' y='18' class='chart-title'>{html.escape(symbol)} Price / Volume (30 bars)</text>",
        f"<line x1='{left}' y1='{mid}' x2='{width-right}' y2='{mid}' stroke='#d9dee7' />",
        f"<polyline points='{points}' fill='none' stroke='#14213d' stroke-width='2.4' />",
    ]
    for x, volume in zip(xs, volumes):
        bar_h = 55 * (volume / vol_max) if vol_max else 0
        parts.append(f"<rect x='{x - 4:.1f}' y='{bottom - bar_h:.1f}' width='8' height='{bar_h:.1f}' fill='#b8872f' opacity='0.55' />")
    first_date = str(rows.iloc[0]["date"])[:10]
    last_date = str(rows.iloc[-1]["date"])[:10]
    parts.extend(
        [
            f"<text x='{left}' y='260' class='axis'>{html.escape(first_date)}</text>",
            f"<text x='{width-right-90}' y='260' class='axis'>{html.escape(last_date)}</text>",
            f"<text x='{left}' y='{top + 10}' class='axis'>{max(closes):.2f}</text>",
            f"<text x='{left}' y='{mid - 6}' class='axis'>{min(closes):.2f}</text>",
            "</svg>",
        ]
    )
    return "".join(parts)


def build() -> Path:
    features_path = DEMO_DIR / "02_feature_engine.csv"
    watchlist_path = DEMO_DIR / "08_watchlist.csv"
    source = pd.read_csv(watchlist_path if watchlist_path.exists() else features_path)
    if "symbol" in source.columns:
        source["symbol"] = source["symbol"].astype(str).str.upper()
    ohlcv = pd.read_parquet(DATA_DIR / "ohlcv_master.parquet")
    ohlcv["symbol"] = ohlcv["symbol"].astype(str).str.upper()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.tz_localize(None).dt.normalize()
    top_symbols = source.sort_values("edge_rank_score", ascending=False).head(6)["symbol"].tolist()
    cards = "\n".join(
        f"<a class='chip' href='symbols/{html.escape(symbol)}.html'>{html.escape(symbol)}</a>"
        for symbol in top_symbols
    )
    price_charts = "\n".join(f"<section>{_price_line_chart(symbol, ohlcv)}</section>" for symbol in top_symbols[:4])
    html_text = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>02 Feature Engine Visuals</title>
  <style>
    :root {{ --ink:#14213d; --muted:#5f6b7a; --line:#d9dee7; --bg:#f5f1e8; --panel:#fffaf0; }}
    body {{ margin:0; padding:28px; font-family:Georgia,'Times New Roman',serif; background:var(--bg); color:var(--ink); }}
    h1 {{ margin:0 0 8px; font-size:30px; }}
    h2 {{ margin:28px 0 12px; font-size:20px; }}
    .note {{ margin:0 0 18px; color:var(--muted); font-family:Segoe UI,Arial,sans-serif; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px; }}
    section {{ background:white; border:1px solid var(--line); border-radius:8px; padding:16px; box-shadow:0 14px 36px rgba(20,33,61,.10); overflow:auto; }}
    svg {{ width:100%; min-width:640px; height:auto; display:block; }}
    .chart-title {{ font:700 16px Segoe UI,Arial,sans-serif; fill:var(--ink); }}
    .axis {{ font:12px Segoe UI,Arial,sans-serif; fill:var(--muted); }}
    .value {{ font:12px Segoe UI,Arial,sans-serif; fill:var(--ink); }}
    .heat-value {{ font:11px Segoe UI,Arial,sans-serif; fill:white; font-weight:700; }}
    .chips {{ display:flex; gap:8px; flex-wrap:wrap; margin:12px 0 20px; }}
    .chip {{ color:var(--ink); text-decoration:none; border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:7px 10px; font-family:Segoe UI,Arial,sans-serif; }}
    a {{ color:var(--ink); }}
  </style>
</head>
<body>
  <h1>02 Feature Engine Visuals</h1>
  <p class="note"><a href="index.html">Index</a> | <a href="02_feature_engine.html">Feature table</a> | Biểu diễn trực quan snapshot feature và giá gần đây.</p>
  <div class="chips">{cards}</div>
  <div class="grid">
    <section>{_bar_chart(source, "edge_rank_score", "Edge Rank Score", color="#14213d")}</section>
    <section>{_bar_chart(source, "smart_money_score", "Smart Money Score", color="#0f8b4c")}</section>
    <section>{_bar_chart(source, "rs_percentile_20", "Relative Strength Percentile 20", color="#2d6cdf")}</section>
    <section>{_bar_chart(source, "value_ratio_20", "Value Ratio 20", color="#b8872f")}</section>
  </div>
  <h2>Feature Heatmap</h2>
  <section>{_multi_metric_heatmap(source)}</section>
  <h2>Top Symbol Price Lines</h2>
  <div class="grid">{price_charts}</div>
</body>
</html>
"""
    out = DEMO_DIR / "02_feature_engine_visuals.html"
    out.write_text(html_text, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
