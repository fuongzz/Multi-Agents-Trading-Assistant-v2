"""
Smart Money Trace Interactive Visualization - Plotly (zoom, pan, hover)

Generate HTML files that can be opened in browser with full interactivity:
  - Zoom in/out
  - Pan (drag to move)
  - Hover for values
  - Click legend to hide/show lines
  - Download as PNG
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def plot_market_dashboard_interactive(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Interactive 4-panel market dashboard with Plotly.

    Args:
        summary: DataFrame with date, avg_smart_money_score, *_count columns
        output_path: Output HTML file path
    """
    summary = summary.copy()
    summary["date"] = pd.to_datetime(summary["date"])
    summary = summary.sort_values("date")

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Smart Money Score (Avg & Median)",
            "Smart Money States Distribution",
            "Accumulation vs Distribution Days",
            "Accumulation/Distribution Ratio"
        ),
        specs=[[{"secondary_y": False}] for _ in range(4)],
        vertical_spacing=0.08,
    )

    # Panel 1: Smart Money Score
    fig.add_trace(
        go.Scatter(
            x=summary["date"],
            y=summary["avg_smart_money_score"],
            mode="lines",
            name="Avg Smart Money Score",
            line=dict(color="blue", width=2),
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=summary["date"],
            y=summary["median_smart_money_score"],
            mode="lines",
            name="Median Smart Money Score",
            line=dict(color="orange", width=2, dash="dash"),
        ),
        row=1, col=1
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3, row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="green", opacity=0.3, row=1, col=1)
    fig.update_yaxes(title_text="Score", range=[0, 100], row=1, col=1)

    # Panel 2: States Distribution
    fig.add_trace(
        go.Scatter(
            x=summary["date"],
            y=summary["hot_but_strong_count"],
            mode="lines",
            name="HOT_BUT_STRONG",
            line=dict(color="darkgreen", width=2),
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=summary["date"],
            y=summary["hot_and_exhausted_count"],
            mode="lines",
            name="HOT_AND_EXHAUSTED",
            line=dict(color="orange", width=2),
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=summary["date"],
            y=summary["hot_and_distributing_count"],
            mode="lines",
            name="HOT_AND_DISTRIBUTING",
            line=dict(color="red", width=2),
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=summary["date"],
            y=summary["reset_in_uptrend_count"],
            mode="lines",
            name="RESET_IN_UPTREND",
            line=dict(color="purple", width=2),
        ),
        row=2, col=1
    )
    fig.update_yaxes(title_text="Count", row=2, col=1)

    # Panel 3: A/D Days
    fig.add_trace(
        go.Scatter(
            x=summary["date"],
            y=summary["accumulation_total"],
            mode="lines",
            name="Accumulation Days",
            line=dict(color="green", width=2),
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=summary["date"],
            y=summary["distribution_total"],
            mode="lines",
            name="Distribution Days",
            line=dict(color="red", width=2),
        ),
        row=3, col=1
    )
    fig.update_yaxes(title_text="Days Count", row=3, col=1)

    # Panel 4: A/D Ratio
    fig.add_trace(
        go.Scatter(
            x=summary["date"],
            y=summary["accumulation_distribution_ratio"],
            mode="lines",
            name="A/D Ratio",
            line=dict(color="purple", width=2.5),
        ),
        row=4, col=1
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.3, row=4, col=1)
    fig.update_yaxes(title_text="Ratio", range=[0, 1], row=4, col=1)

    fig.update_xaxes(title_text="Ngày", row=4, col=1, fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    fig.update_layout(
        title_text="Smart Money Trace Market Dashboard (Interactive)",
        height=1200,
        hovermode="x unified",
        template="plotly_white",
        dragmode=None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, config={"scrollZoom": True})
    logger.info(f"Saved interactive market dashboard: {output_path}")


def plot_symbol_dashboard_interactive(
    df: pd.DataFrame,
    symbol: str,
    output_path: Path,
) -> None:
    """
    Interactive 4-panel symbol dashboard with Plotly.

    Args:
        df: Full smart_money_by_symbol DataFrame
        symbol: Stock symbol
        output_path: Output HTML file path
    """
    symbol = symbol.upper()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df_sym = df[df["symbol"] == symbol].sort_values("date")

    if df_sym.empty:
        logger.warning(f"No data for {symbol}")
        return

    df_sym["ma20"] = df_sym["close"].rolling(20, min_periods=1).mean()
    df_sym["ma50"] = df_sym["close"].rolling(50, min_periods=1).mean()
    df_sym["ma200"] = df_sym["close"].rolling(200, min_periods=1).mean()

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f"{symbol} Price + MA",
            f"{symbol} Smart Money Score",
            f"{symbol} Relative Strength & CLV",
            f"{symbol} Accumulation/Distribution Days"
        ),
        specs=[[{"secondary_y": False}] for _ in range(4)],
        vertical_spacing=0.08,
    )

    # Panel 1: Price + MA
    fig.add_trace(
        go.Scatter(
            x=df_sym["date"],
            y=df_sym["close"],
            mode="lines",
            name="Close",
            line=dict(color="black", width=1.5),
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_sym["date"],
            y=df_sym["ma20"],
            mode="lines",
            name="MA20",
            line=dict(color="blue", width=1.5, dash="dash"),
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_sym["date"],
            y=df_sym["ma50"],
            mode="lines",
            name="MA50",
            line=dict(color="orange", width=1.5, dash="dash"),
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_sym["date"],
            y=df_sym["ma200"],
            mode="lines",
            name="MA200",
            line=dict(color="red", width=1.5, dash="dash"),
        ),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Price (VND)", row=1, col=1)

    # Panel 2: Smart Money Score
    # Color by state
    state_colors = {
        "HOT_BUT_STRONG": "darkgreen",
        "HOT_AND_EXHAUSTED": "orange",
        "HOT_AND_DISTRIBUTING": "red",
        "RESET_IN_UPTREND": "purple",
        "NEUTRAL": "gray",
    }

    for state, color in state_colors.items():
        mask = df_sym["smart_money_state"] == state
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=df_sym.loc[mask, "date"],
                    y=df_sym.loc[mask, "smart_money_score"],
                    mode="markers+lines",
                    name=state,
                    marker=dict(color=color, size=4),
                    line=dict(color=color, width=1),
                ),
                row=2, col=1
            )

    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.3, row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="green", opacity=0.3, row=2, col=1)
    fig.update_yaxes(title_text="Smart Money Score", range=[0, 100], row=2, col=1)

    # Panel 3: RS & CLV
    fig.add_trace(
        go.Scatter(
            x=df_sym["date"],
            y=df_sym["rs_score"],
            mode="lines",
            name="RS Score",
            line=dict(color="blue", width=2),
        ),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_sym["date"],
            y=df_sym["clv_score"],
            mode="lines",
            name="CLV Score",
            line=dict(color="orange", width=2),
        ),
        row=3, col=1
    )
    fig.update_yaxes(title_text="Score", range=[0, 100], row=3, col=1)

    # Panel 4: A/D Balance
    fig.add_trace(
        go.Scatter(
            x=df_sym["date"],
            y=df_sym["accumulation_days_10"],
            mode="lines",
            name="Accumulation Days (10)",
            line=dict(color="green", width=1.5),
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_sym["date"],
            y=df_sym["distribution_days_10"],
            mode="lines",
            name="Distribution Days (10)",
            line=dict(color="red", width=1.5),
        ),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df_sym["date"],
            y=df_sym["ad_balance_10"],
            mode="lines",
            name="AD Balance (10)",
            line=dict(color="purple", width=2, dash="dash"),
        ),
        row=4, col=1
    )
    fig.update_yaxes(title_text="Count / Balance", row=4, col=1)

    fig.update_xaxes(title_text="Ngày", row=4, col=1, fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    fig.update_layout(
        title_text=f"{symbol} - Smart Money Trace Dashboard (Interactive)",
        height=1200,
        hovermode="x unified",
        template="plotly_white",
        dragmode=None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, config={"scrollZoom": True})
    logger.info(f"Saved interactive symbol dashboard: {output_path}")


def run_interactive_charts(
    data_dir: str | Path,
    output_dir: str | Path,
    symbols: str | list[str] | None = None,
    skip_market: bool = False,
) -> None:
    """
    Generate all interactive HTML charts.

    Args:
        data_dir: Path to smart_money_trace parquet files
        output_dir: Path to save HTML files
        symbols: Symbol or list of symbols for per-symbol dashboards.
                If None, auto-detect from data
        skip_market: If True, skip the market dashboard
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    summary = pd.read_parquet(data_dir / "smart_money_daily_summary.parquet")
    by_symbol = pd.read_parquet(data_dir / "smart_money_by_symbol.parquet")

    if symbols is None:
        # Auto-detect top symbols by frequency
        symbols = by_symbol["symbol"].value_counts().head(30).index.tolist()
    elif isinstance(symbols, str):
        symbols = [symbols]
    symbols = [s.upper() for s in symbols]

    logger.info(f"Generating interactive charts for {len(symbols)} symbol(s)...")

    if not skip_market:
        logger.info("Market dashboard (interactive)...")
        plot_market_dashboard_interactive(summary, output_dir / "01_market_dashboard_interactive.html")

    for idx, symbol in enumerate(symbols, start=1):
        logger.info(f"[{idx}/{len(symbols)}] Symbol dashboard ({symbol})...")
        plot_symbol_dashboard_interactive(
            by_symbol,
            symbol,
            output_dir / f"02_symbol_dashboard_{symbol}_interactive.html"
        )

    logger.info(f"All interactive charts saved to {output_dir}")
