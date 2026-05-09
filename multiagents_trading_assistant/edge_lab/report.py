"""Report writers for edge lab runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from multiagents_trading_assistant.edge_lab.metrics import equity_metrics, trade_metrics


def build_summary(
    signal_trades: pd.DataFrame,
    portfolio_trades: pd.DataFrame,
    equity: pd.DataFrame,
    initial_capital: float,
    benchmark: dict,
) -> pd.DataFrame:
    rows = []
    if not signal_trades.empty:
        for (hypothesis, hold), group in signal_trades.groupby(["hypothesis", "hold"]):
            rows.append(
                {
                    "scope": "signal_quality",
                    "hypothesis": hypothesis,
                    "hold": hold,
                    **trade_metrics(group),
                    **benchmark,
                }
            )
    if not portfolio_trades.empty or not equity.empty:
        rows.append(
            {
                "scope": "portfolio",
                "hypothesis": "combined",
                "hold": "",
                **trade_metrics(portfolio_trades),
                **equity_metrics(equity, initial_capital),
                **benchmark,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    output_dir: str | Path,
    signal_trades: pd.DataFrame,
    portfolio_trades: pd.DataFrame,
    equity: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    signal_trades.to_csv(out / "signal_quality_trades.csv", index=False)
    portfolio_trades.to_csv(out / "portfolio_trades.csv", index=False)
    equity.to_csv(out / "portfolio_equity.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    with pd.ExcelWriter(out / "edge_lab_report.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        portfolio_trades.to_excel(writer, sheet_name="portfolio_trades", index=False)
        equity.to_excel(writer, sheet_name="portfolio_equity", index=False)
        signal_trades.head(50000).to_excel(writer, sheet_name="signal_quality", index=False)

