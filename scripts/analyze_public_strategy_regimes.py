"""Slice public-strategy backtests across Vietnam market periods."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "backtest_results/public_strategy_fishing/public_strategy_fishing_combined_2022-01-01_2026-05-19.csv"
OUT_DIR = ROOT / "backtest_results/public_strategy_fishing"
REPORTS_DIR = ROOT / "reports"


PERIODS = [
    ("2022_bear", "2022-01-01", "2022-12-31"),
    ("2023_recovery", "2023-01-01", "2023-12-31"),
    ("2024_sideways_up", "2024-01-01", "2024-12-31"),
    ("2025_2026_current", "2025-01-01", "2026-05-19"),
]


def _period_return(equity: pd.DataFrame, start: str, end: str) -> tuple[float | None, float | None, float | None]:
    frame = equity[(equity["date"] >= pd.Timestamp(start)) & (equity["date"] <= pd.Timestamp(end))].copy()
    if len(frame) < 2:
        return None, None, None
    frame = frame.sort_values("date")
    daily = frame["equity"].pct_change().dropna()
    ret = frame["equity"].iloc[-1] / frame["equity"].iloc[0] - 1.0
    dd = (frame["equity"] / frame["equity"].cummax() - 1.0).min()
    sharpe = None
    if not daily.empty and daily.std(ddof=0) > 0:
        sharpe = (daily.mean() / daily.std(ddof=0)) * (252 ** 0.5)
    return ret, dd, sharpe


def _trade_stats(trades: pd.DataFrame, start: str, end: str) -> tuple[int, float | None, float | None]:
    if trades.empty or "entry_date" not in trades.columns:
        return 0, None, None
    frame = trades[(trades["entry_date"] >= pd.Timestamp(start)) & (trades["entry_date"] <= pd.Timestamp(end))]
    if frame.empty:
        return 0, None, None
    win_rate = (frame["pnl"] > 0).mean()
    avg_pnl = frame["pnl_pct"].mean() if "pnl_pct" in frame.columns else None
    return int(len(frame)), float(win_rate), None if avg_pnl is None else float(avg_pnl)


def _vnindex_periods(periods: list[tuple[str, str, str]]) -> dict[str, float | None]:
    path = ROOT / "multiagents_trading_assistant/data/index_master.parquet"
    if not path.exists():
        return {name: None for name, _, _ in periods}
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "symbol" in df.columns:
        df = df[df["symbol"].astype(str).str.upper() == "VNINDEX"]
    out: dict[str, float | None] = {}
    for name, start, end in periods:
        frame = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].sort_values("date")
        if len(frame) < 2:
            out[name] = None
        else:
            out[name] = float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0)
    return out


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df[cols].iterrows():
        values = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                if col.endswith("_pct") or col in {"period_return_pct", "max_dd_pct", "win_rate_pct", "avg_trade_pct", "vnindex_return_pct"}:
                    values.append(f"{value:.2f}")
                elif "sharpe" in col:
                    values.append(f"{value:.3f}")
                else:
                    values.append(f"{value:.2f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def analyze(summary_path: Path, periods: list[tuple[str, str, str]]) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    summary = pd.read_csv(summary_path)
    rows: list[dict] = []
    vnindex_returns = _vnindex_periods(periods)

    for item in summary.to_dict("records"):
        summary_file = Path(str(item["summary_file"]))
        equity_file = Path(str(summary_file).replace("_summary.csv", "_top10_eod_next_open_equity.csv"))
        trades_file = Path(str(summary_file).replace("_summary.csv", "_top10_eod_next_open_trades.csv"))
        if not equity_file.exists():
            continue
        equity = pd.read_csv(equity_file)
        equity["date"] = pd.to_datetime(equity["date"]).dt.normalize()
        trades = pd.read_csv(trades_file) if trades_file.exists() else pd.DataFrame()
        if not trades.empty:
            for col in ["entry_date", "exit_date", "signal_date"]:
                if col in trades.columns:
                    trades[col] = pd.to_datetime(trades[col]).dt.normalize()

        for period, start, end in periods:
            ret, dd, sharpe = _period_return(equity, start, end)
            n_trades, win_rate, avg_pnl = _trade_stats(trades, start, end)
            rows.append(
                {
                    "period": period,
                    "period_start": start,
                    "period_end": end,
                    "source_family": item.get("source_family", item.get("label", "")),
                    "strategy": item["strategy"],
                    "period_return_pct": None if ret is None else ret * 100.0,
                    "max_dd_pct": None if dd is None else dd * 100.0,
                    "sharpe": sharpe,
                    "trades": n_trades,
                    "win_rate_pct": None if win_rate is None else win_rate * 100.0,
                    "avg_trade_pct": None if avg_pnl is None else avg_pnl * 100.0,
                    "vnindex_return_pct": None if vnindex_returns.get(period) is None else vnindex_returns[period] * 100.0,
                }
            )

    detail = pd.DataFrame(rows)
    pivot = (
        detail.pivot_table(index=["source_family", "strategy"], columns="period", values="period_return_pct", aggfunc="first")
        .reset_index()
        .sort_values(["2025_2026_current", "2024_sideways_up", "2023_recovery"], ascending=[False, False, False], na_position="last")
    )
    return detail, pivot, summary_path


def write_outputs(detail: pd.DataFrame, pivot: pd.DataFrame, label: str) -> tuple[Path, Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUT_DIR / f"public_strategy_regime_detail_{label}.csv"
    pivot_path = OUT_DIR / f"public_strategy_regime_pivot_{label}.csv"
    report_path = REPORTS_DIR / f"public_strategy_regime_report_{label}.md"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    pivot.to_csv(pivot_path, index=False, encoding="utf-8-sig")

    lines = [
        "# Public Strategy Regime Slicing",
        "",
        "Method: slice the already generated full-period `eod_next_open` equity curves and trades by market period. This is a regime diagnostic, not a fresh capital reset backtest for each period.",
        "",
        f"Detail CSV: `{detail_path}`",
        f"Pivot CSV: `{pivot_path}`",
        "",
        "## Period Leaderboard",
        "",
    ]
    for period in [p[0] for p in PERIODS]:
        part = detail[detail["period"] == period].sort_values(["period_return_pct", "sharpe"], ascending=[False, False])
        lines.append(f"### {period}")
        lines.append("")
        lines.append(
            _markdown_table(
                part.head(6),
                ["source_family", "strategy", "period_return_pct", "sharpe", "max_dd_pct", "trades", "win_rate_pct", "vnindex_return_pct"],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Return Pivot",
            "",
            _markdown_table(
                pivot,
                ["source_family", "strategy", "2022_bear", "2023_recovery", "2024_sideways_up", "2025_2026_current"],
            ),
            "",
            "## Interpretation",
            "",
            "- 2022 is the stress test. A strategy that survives here is more valuable than one that only works in 2023 recovery.",
            "- 2023 shows recovery/bounce behavior.",
            "- 2024 and 2025-2026 show whether the entry logic still works after the easy rebound.",
            "- Treat low-trade periods as directional evidence only.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return detail_path, pivot_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--label", default="2022-2026")
    args = parser.parse_args()
    detail, pivot, _ = analyze(Path(args.summary), PERIODS)
    detail_path, pivot_path, report_path = write_outputs(detail, pivot, args.label)
    print(detail_path)
    print(pivot_path)
    print(report_path)


if __name__ == "__main__":
    main()
