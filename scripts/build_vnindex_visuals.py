"""Build an offline VNINDEX visual analysis page for the layered demo."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "reports" / "layered_pipeline_demo_2026-05-17"
INDEX_MASTER = ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return ""


def _scale(values: list[float], low: float, high: float) -> list[float]:
    values = [float(v) for v in values if pd.notna(v)]
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [(low + high) / 2 for _ in values]
    return [high - ((v - vmin) / (vmax - vmin)) * (high - low) for v in values]


def _polyline(points: list[tuple[float, float]], color: str, width: float = 2.5) -> str:
    if not points:
        return ""
    data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f"<polyline points='{data}' fill='none' stroke='{color}' stroke-width='{width}' stroke-linejoin='round' stroke-linecap='round'/>"


def _trend_label(row: pd.Series) -> str:
    close = _num(row.get("close"))
    ma20 = _num(row.get("ma20"))
    ma50 = _num(row.get("ma50"))
    ma200 = _num(row.get("ma200"))
    if close > ma20 > ma50:
        return "Uptrend ngắn hạn tốt"
    if close > ma50 and close > ma200:
        return "Xu hướng lớn vẫn tích cực, ngắn hạn cần theo dõi"
    if close < ma20 < ma50:
        return "Ngắn hạn suy yếu"
    return "Trung tính / nhiễu"


def _risk_label(row: pd.Series) -> str:
    ret20 = _num(row.get("ret_20d_pct"))
    ds20 = _num(row.get("DS20"))
    chdm20 = _num(row.get("CHDM20"))
    if ret20 > 5 and ds20 > 0 and chdm20 >= 45:
        return "Bối cảnh thuận lợi: có thể ưu tiên tín hiệu mạnh"
    if ret20 < -3 or chdm20 < 35:
        return "Rủi ro tăng: nên giảm mua đuổi và yêu cầu confluence cao hơn"
    return "Trung tính: chọn lọc theo ranking và risk gate"


def _build_frame() -> pd.DataFrame:
    df = pd.read_parquet(INDEX_MASTER)
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper().eq("VNINDEX")]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").tail(160).copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(120, min_periods=50).mean()
    df["ret_1d_pct"] = df["close"].pct_change(1) * 100
    df["ret_5d_pct"] = df["close"].pct_change(5) * 100
    df["ret_20d_pct"] = df["close"].pct_change(20) * 100
    df["vol_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean()

    money = ROOT / "data" / "research" / "money_cycle" / "money_cycle_market.parquet"
    if money.exists():
        mc = pd.read_parquet(money)
        mc["date"] = pd.to_datetime(mc["date"]).dt.tz_localize(None).dt.normalize()
        cols = [c for c in ["date", "CHDM20", "CHDM50", "DS20", "DS50"] if c in mc.columns]
        df = df.merge(mc[cols], on="date", how="left")
    return df


def _chart(df: pd.DataFrame) -> str:
    plot = df.tail(90).copy().reset_index(drop=True)
    width, height = 980, 420
    left, right, top, price_bottom = 58, 24, 28, 290
    vol_top, bottom = 318, 394
    xs = [left + i * ((width - left - right) / max(len(plot) - 1, 1)) for i in range(len(plot))]

    price_values = []
    for col in ["close", "ma20", "ma50"]:
        price_values.extend([_num(v) for v in plot[col].dropna().tolist()])
    ys_close = _scale([_num(v) for v in plot["close"]], top, price_bottom)
    ys_ma20 = _scale([_num(v) for v in plot["ma20"].fillna(plot["close"])], top, price_bottom)
    ys_ma50 = _scale([_num(v) for v in plot["ma50"].fillna(plot["close"])], top, price_bottom)

    v_max = max([_num(v) for v in plot["volume"]], default=1)
    bars = []
    for i, row in plot.iterrows():
        vol = _num(row.get("volume"))
        bar_h = 2 if v_max <= 0 else vol / v_max * (bottom - vol_top)
        color = "#0f8b4c" if _num(row.get("close")) >= _num(row.get("open")) else "#c63d2f"
        bars.append(f"<rect x='{xs[i]-3:.1f}' y='{bottom-bar_h:.1f}' width='6' height='{bar_h:.1f}' fill='{color}' opacity='.55'/>")

    latest = plot.iloc[-1]
    first = plot.iloc[0]
    labels = [
        f"<text x='{left}' y='18' class='label'>VNINDEX 90 phiên: {first['date']:%Y-%m-%d} -> {latest['date']:%Y-%m-%d}</text>",
        f"<text x='{width-right-220}' y='18' class='legend close'>Close</text>",
        f"<text x='{width-right-150}' y='18' class='legend ma20'>MA20</text>",
        f"<text x='{width-right-80}' y='18' class='legend ma50'>MA50</text>",
        f"<line x1='{left}' y1='{price_bottom}' x2='{width-right}' y2='{price_bottom}' stroke='#d9dee7'/>",
        f"<line x1='{left}' y1='{bottom}' x2='{width-right}' y2='{bottom}' stroke='#d9dee7'/>",
    ]
    for j in [0, len(plot)//2, len(plot)-1]:
        labels.append(f"<text x='{xs[j]-34:.1f}' y='{height-8}' class='axis'>{plot.iloc[j]['date']:%m-%d}</text>")

    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='VNINDEX line chart'>"
        + "".join(labels)
        + "".join(bars)
        + _polyline(list(zip(xs, ys_close)), "#14213d", 3.0)
        + _polyline(list(zip(xs, ys_ma20)), "#b8872f", 2.0)
        + _polyline(list(zip(xs, ys_ma50)), "#3b82f6", 2.0)
        + "</svg>"
    )


def build(demo_dir: Path = DEMO_DIR) -> Path:
    demo_dir.mkdir(parents=True, exist_ok=True)
    df = _build_frame()
    latest = df.iloc[-1]
    prior = df.iloc[-2] if len(df) > 1 else latest
    cards = [
        ("Close", _fmt(latest.get("close")), f"{_fmt(latest.get('ret_1d_pct'))}% 1D"),
        ("5D / 20D", f"{_fmt(latest.get('ret_5d_pct'))}% / {_fmt(latest.get('ret_20d_pct'))}%", "momentum"),
        ("MA20 / MA50", f"{_fmt(latest.get('ma20'))} / {_fmt(latest.get('ma50'))}", _trend_label(latest)),
        ("Volume ratio 20", _fmt(latest.get("vol_ratio_20")), "so với trung bình 20 phiên"),
        ("CHDM20 / DS20", f"{_fmt(latest.get('CHDM20'))} / {_fmt(latest.get('DS20'), 3)}", "money cycle"),
        ("Risk read", _risk_label(latest), ""),
    ]
    card_html = "".join(
        f"<div class='metric'><span>{html.escape(k)}</span><strong>{html.escape(str(v))}</strong><em>{html.escape(str(n))}</em></div>"
        for k, v, n in cards
    )
    note = (
        f"Phiên mới nhất {latest['date']:%Y-%m-%d}: VNINDEX đóng cửa {_fmt(latest.get('close'))}, "
        f"{_fmt(latest.get('ret_1d_pct'))}% so với phiên trước ({_fmt(prior.get('close'))}). "
        f"{_trend_label(latest)}. {_risk_label(latest)}."
    )
    table = df.tail(30)[
        [c for c in ["date", "open", "high", "low", "close", "volume", "ret_1d_pct", "ret_5d_pct", "ret_20d_pct", "ma20", "ma50", "vol_ratio_20", "CHDM20", "DS20"] if c in df.columns]
    ].copy()
    table["date"] = table["date"].dt.date.astype(str)
    table_html = table.to_html(index=False, classes="data-table", border=0)
    out = demo_dir / "01_vnindex_visuals.html"
    out.write_text(
        f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>01 VNINDEX Visual Analysis</title>
  <style>
    :root {{ --ink:#14213d; --muted:#5f6b7a; --line:#d9dee7; --bg:#f5f1e8; --panel:#fffaf0; }}
    body {{ margin:0; padding:28px; font-family:Georgia,'Times New Roman',serif; background:var(--bg); color:var(--ink); }}
    h1 {{ margin:0 0 8px; font-size:30px; }}
    h2 {{ margin:26px 0 12px; font-size:20px; }}
    .note {{ color:var(--muted); font-family:Segoe UI,Arial,sans-serif; line-height:1.55; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:18px 0; }}
    .metric {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .metric span,.metric em {{ display:block; color:var(--muted); font:13px Segoe UI,Arial,sans-serif; }}
    .metric strong {{ display:block; margin:6px 0; font-size:20px; }}
    .panel {{ background:white; border:1px solid var(--line); padding:16px; overflow:auto; }}
    svg {{ width:100%; min-width:760px; }}
    .label,.axis,.legend {{ font-family:Segoe UI,Arial,sans-serif; fill:#5f6b7a; font-size:12px; }}
    .legend.close {{ fill:#14213d; }} .legend.ma20 {{ fill:#b8872f; }} .legend.ma50 {{ fill:#3b82f6; }}
    table {{ border-collapse:collapse; width:100%; font:13px Segoe UI,Arial,sans-serif; white-space:nowrap; }}
    th {{ background:var(--ink); color:white; position:sticky; top:0; text-align:left; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid #edf0f5; }}
    a {{ color:var(--ink); }}
  </style>
</head>
<body>
  <h1>01 VNINDEX Visual Analysis</h1>
  <p class="note"><a href="index.html">Index</a> | <a href="01_vnindex.html">VNINDEX table</a> | Cập nhật từ index_master.parquet.</p>
  <div class="cards">{card_html}</div>
  <div class="panel">{_chart(df)}</div>
  <h2>Phân tích nhanh</h2>
  <p class="note">{html.escape(note)}</p>
  <h2>30 phiên gần nhất</h2>
  <div class="panel">{table_html}</div>
  <script src="sortable_tables.js"></script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return out


def main() -> None:
    path = build()
    print(path)


if __name__ == "__main__":
    main()
