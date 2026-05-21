"""Generate demo assets for the bias-fixed MVP backtest.

This script does not run a backtest. It reads trusted artifacts from
``backtest_results`` and writes presentation-ready CSV/HTML files under
``reports/demo_mvp_2026-05-16/assets``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "backtest_results" / "combos_unbiased_vn100_pure_top3_p3c3_20260516_2025-01-01_2026-05-16"
GRID_SUMMARY = ROOT / "backtest_results" / "mvp_unbiased_pure_top3_grid_2025now_20260516" / "summary.csv"
OUT_DIR = ROOT / "reports" / "demo_mvp_2026-05-16" / "assets"


def _max_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def _format_vnd(value: float) -> str:
    return f"{value:,.0f}"


def _scale(values: pd.Series, low: float, high: float, invert: bool = False) -> list[float]:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax == vmin:
        midpoint = (low + high) / 2
        return [midpoint for _ in values]
    scaled = low + (values - vmin) / (vmax - vmin) * (high - low)
    if invert:
        scaled = high - (scaled - low)
    return [float(item) for item in scaled]


def _write_line_chart(
    path: Path,
    *,
    title: str,
    dates: pd.Series,
    values: pd.Series,
    y_label: str,
    color: str,
    fill_to_zero: bool = False,
) -> None:
    width = 1100
    height = 560
    left = 80
    right = 30
    top = 70
    bottom = 80
    x_values = pd.Series(range(len(values)))
    xs = _scale(x_values, left, width - right)
    ys = _scale(values, top, height - bottom, invert=True)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    y_min = float(values.min())
    y_max = float(values.max())
    zero_y = _scale(pd.Series([0.0, y_min, y_max]), top, height - bottom, invert=True)[0]
    area = ""
    if fill_to_zero:
        area = (
            f'<polygon points="{left},{zero_y:.1f} {points} {width-right},{zero_y:.1f}" '
            f'fill="{color}" opacity="0.18"></polygon>'
        )
    first_date = pd.Timestamp(dates.iloc[0]).date().isoformat()
    last_date = pd.Timestamp(dates.iloc[-1]).date().isoformat()
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #18212f; }}
    .card {{ max-width: 1180px; border: 1px solid #d7dde8; border-radius: 16px; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .sub {{ color: #5b6472; margin-bottom: 18px; }}
    svg {{ width: 100%; height: auto; background: #fbfcff; border-radius: 12px; }}
    .axis {{ stroke: #9aa4b2; stroke-width: 1; }}
    .grid {{ stroke: #e6ebf2; stroke-width: 1; }}
    .line {{ fill: none; stroke: {color}; stroke-width: 3; }}
    .label {{ fill: #5b6472; font-size: 13px; }}
    .metric {{ font-weight: 600; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>{title}</h1>
    <div class="sub">{first_date} to {last_date} | {y_label}</div>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
      <line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"></line>
      <line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"></line>
      <line class="grid" x1="{left}" y1="{top}" x2="{width-right}" y2="{top}"></line>
      <line class="grid" x1="{left}" y1="{(top + height - bottom)/2:.1f}" x2="{width-right}" y2="{(top + height - bottom)/2:.1f}"></line>
      {area}
      <polyline class="line" points="{points}"></polyline>
      <text class="label metric" x="{left}" y="34">{y_max:,.2f}</text>
      <text class="label metric" x="{left}" y="{height-38}">{y_min:,.2f}</text>
      <text class="label" x="{left}" y="{height-22}">{first_date}</text>
      <text class="label" x="{width-right-90}" y="{height-22}">{last_date}</text>
    </svg>
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _write_trade_bar_chart(path: Path, trades: pd.DataFrame) -> None:
    width = 1100
    height = 620
    left = 80
    right = 30
    top = 70
    bottom = 110
    frame = trades.sort_values("exit_date").reset_index(drop=True)
    max_abs = float(frame["pnl"].abs().max())
    zero_y = (top + height - bottom) / 2
    bar_area = (height - bottom - top) / 2 - 20
    bar_width = max(10, (width - left - right) / max(1, len(frame)) * 0.62)
    step = (width - left - right) / max(1, len(frame))
    bars = []
    labels = []
    for idx, row in frame.iterrows():
        pnl = float(row["pnl"])
        x = left + idx * step + (step - bar_width) / 2
        h = abs(pnl) / max_abs * bar_area if max_abs else 0
        y = zero_y - h if pnl >= 0 else zero_y
        color = "#15803d" if pnl >= 0 else "#b91c1c"
        label = f"{row['symbol']} {pnl/1_000_000:.1f}m"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" '
            f'fill="{color}"><title>{label}</title></rect>'
        )
        if idx % 2 == 0:
            labels.append(
                f'<text class="label" transform="translate({x+bar_width/2:.1f},{height-72}) rotate(55)">'
                f'{row["symbol"]}</text>'
            )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Closed Trade PnL</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #18212f; }}
    .card {{ max-width: 1180px; border: 1px solid #d7dde8; border-radius: 16px; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .sub {{ color: #5b6472; margin-bottom: 18px; }}
    svg {{ width: 100%; height: auto; background: #fbfcff; border-radius: 12px; }}
    .axis {{ stroke: #9aa4b2; stroke-width: 1; }}
    .label {{ fill: #5b6472; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Closed Trade PnL</h1>
    <div class="sub">Green = profitable trade, red = loss. Hover bars for symbol and PnL.</div>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Closed Trade PnL">
      <line class="axis" x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}"></line>
      {''.join(bars)}
      {''.join(labels)}
      <text class="label" x="{left}" y="36">+/- {max_abs/1_000_000:.1f}m VND</text>
    </svg>
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    equity = pd.read_csv(SOURCE_DIR / "PURE_TOP3_equity.csv", parse_dates=["date"])
    trades = pd.read_csv(SOURCE_DIR / "PURE_TOP3_trades.csv", parse_dates=["entry_date", "exit_date", "signal_date"])
    grid = pd.read_csv(GRID_SUMMARY)

    equity["nav_million"] = equity["equity"] / 1_000_000
    equity["drawdown_pct"] = _max_drawdown(equity["equity"]) * 100
    equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)

    first_equity = float(equity["equity"].iloc[0])
    last_equity = float(equity["equity"].iloc[-1])
    total_return = last_equity / first_equity - 1.0
    wins = int((trades["pnl"] > 0).sum())
    losses = int((trades["pnl"] <= 0).sum())

    metrics = pd.DataFrame(
        [
            {"metric": "Start NAV", "value": _format_vnd(first_equity), "raw": first_equity},
            {"metric": "Final NAV", "value": _format_vnd(last_equity), "raw": last_equity},
            {"metric": "Total return", "value": f"{total_return * 100:.2f}%", "raw": total_return},
            {"metric": "Max drawdown", "value": f"{equity['drawdown_pct'].min():.2f}%", "raw": equity["drawdown_pct"].min() / 100},
            {"metric": "Trades", "value": str(len(trades)), "raw": len(trades)},
            {"metric": "Win rate", "value": f"{wins / len(trades) * 100:.2f}%", "raw": wins / len(trades)},
            {"metric": "Winning trades", "value": str(wins), "raw": wins},
            {"metric": "Losing trades", "value": str(losses), "raw": losses},
            {"metric": "First equity date", "value": equity["date"].iloc[0].date().isoformat(), "raw": ""},
            {"metric": "Last equity date", "value": equity["date"].iloc[-1].date().isoformat(), "raw": ""},
        ]
    )
    metrics.to_csv(OUT_DIR / "demo_metrics.csv", index=False)

    top_trades = (
        trades.assign(pnl_pct_display=(trades["pnl_pct"] * 100).round(2))
        .sort_values("pnl", ascending=False)
        [[
            "symbol",
            "entry_date",
            "exit_date",
            "pnl",
            "pnl_pct_display",
            "exit_reason",
            "edge_strategy_name",
        ]]
        .head(12)
    )
    top_trades.to_csv(OUT_DIR / "top_trades.csv", index=False)

    strategy_breakdown = (
        trades.groupby("edge_strategy_name", dropna=False)
        .agg(
            trades=("symbol", "count"),
            pnl=("pnl", "sum"),
            avg_pnl_pct=("pnl_pct", "mean"),
            win_rate=("pnl", lambda s: (s > 0).mean()),
        )
        .reset_index()
        .sort_values("pnl", ascending=False)
    )
    strategy_breakdown.to_csv(OUT_DIR / "strategy_breakdown.csv", index=False)

    symbol_breakdown = (
        trades.groupby("symbol", dropna=False)
        .agg(trades=("symbol", "count"), pnl=("pnl", "sum"), avg_pnl_pct=("pnl_pct", "mean"))
        .reset_index()
        .sort_values("pnl", ascending=False)
    )
    symbol_breakdown.to_csv(OUT_DIR / "symbol_breakdown.csv", index=False)

    monthly = (
        equity.set_index("date")["equity"]
        .resample("ME")
        .last()
        .pct_change()
        .dropna()
        .mul(100)
        .rename("monthly_return_pct")
        .reset_index()
    )
    monthly.to_csv(OUT_DIR / "monthly_returns.csv", index=False)

    _write_line_chart(
        OUT_DIR / "equity_curve.html",
        title="PURE_TOP3 MVP NAV - Bias-Fixed Backtest",
        dates=equity["date"],
        values=equity["nav_million"],
        y_label="NAV (VND million)",
        color="#0f766e",
    )
    _write_line_chart(
        OUT_DIR / "drawdown.html",
        title="Drawdown",
        dates=equity["date"],
        values=equity["drawdown_pct"],
        y_label="Drawdown (%)",
        color="#b91c1c",
        fill_to_zero=True,
    )
    _write_trade_bar_chart(OUT_DIR / "trade_pnl.html", trades)

    grid.sort_values("total_return_pct", ascending=False).to_csv(OUT_DIR / "pure_top3_grid_ranked.csv", index=False)

    print(f"Wrote demo assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
