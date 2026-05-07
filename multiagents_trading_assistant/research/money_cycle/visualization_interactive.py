"""
Money Cycle Interactive Visualization - Plotly (zoom, pan, hover)

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
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def plot_market_dashboard_interactive(
    mc_market: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Interactive 4-panel market dashboard with Plotly.

    Args:
        mc_market: DataFrame with date, CHDM*, DS* columns
        output_path: Output HTML file path
    """
    mc_market = mc_market.copy()
    mc_market["date"] = pd.to_datetime(mc_market["date"])
    mc_market = mc_market.sort_values("date")

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=("CHDM Market (3-10-50-200)", "DS Market (3-50-200)",
                       "CHDM20 + Washout/Distribution Zones", "DS20 vs DS200"),
        specs=[[{"secondary_y": False}] for _ in range(4)],
        vertical_spacing=0.08,
    )

    # Panel 1: CHDM
    fig.add_trace(
        go.Scatter(x=mc_market["date"], y=mc_market["CHDM03"], mode="lines",
                  name="CHDM03", line=dict(color="blue", width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=mc_market["date"], y=mc_market["CHDM10"], mode="lines",
                  name="CHDM10", line=dict(color="orange", width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=mc_market["date"], y=mc_market["CHDM50"], mode="lines",
                  name="CHDM50", line=dict(color="green", width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=mc_market["date"], y=mc_market["CHDM200"], mode="lines",
                  name="CHDM200", line=dict(color="red", width=1.5)),
        row=1, col=1
    )
    fig.add_hline(y=20, line_dash="dash", line_color="red", opacity=0.3, row=1, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="green", opacity=0.3, row=1, col=1)
    fig.update_yaxes(title_text="CHDM (%)", range=[0, 100], row=1, col=1)

    # Panel 2: DS
    fig.add_trace(
        go.Scatter(x=mc_market["date"], y=mc_market["DS03"], mode="lines",
                  name="DS03", line=dict(color="blue", width=1.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=mc_market["date"], y=mc_market["DS50"], mode="lines",
                  name="DS50", line=dict(color="orange", width=1.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=mc_market["date"], y=mc_market["DS200"], mode="lines",
                  name="DS200", line=dict(color="red", width=1.5)),
        row=2, col=1
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.3, row=2, col=1)
    fig.update_yaxes(title_text="DS (Tỷ lệ)", range=[0, 1], row=2, col=1)

    # Panel 3: CHDM20 với vùng
    fig.add_trace(
        go.Scatter(x=mc_market["date"], y=mc_market["CHDM20"], mode="lines",
                  name="CHDM20", line=dict(color="purple", width=2.5)),
        row=3, col=1
    )

    # Washout zones (CHDM20 < 20)
    washout = mc_market["CHDM20"] < 20
    for i in range(len(mc_market) - 1):
        if washout.iloc[i]:
            fig.add_vrect(
                x0=mc_market["date"].iloc[i], x1=mc_market["date"].iloc[i+1],
                fillcolor="red", opacity=0.1, layer="below", line_width=0,
                row=3, col=1
            )

    # Distribution zones (CHDM20 > 80)
    distribution = mc_market["CHDM20"] > 80
    for i in range(len(mc_market) - 1):
        if distribution.iloc[i]:
            fig.add_vrect(
                x0=mc_market["date"].iloc[i], x1=mc_market["date"].iloc[i+1],
                fillcolor="green", opacity=0.1, layer="below", line_width=0,
                row=3, col=1
            )

    fig.add_hline(y=20, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    fig.update_yaxes(title_text="CHDM20 (%)", range=[0, 100], row=3, col=1)

    # Panel 4: DS20 vs DS200
    fig.add_trace(
        go.Scatter(x=mc_market["date"], y=mc_market["DS20"], mode="lines",
                  name="DS20 (ngắn)", line=dict(color="orange", width=2.5)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=mc_market["date"], y=mc_market["DS200"], mode="lines",
                  name="DS200 (dài)", line=dict(color="red", width=2.5)),
        row=4, col=1
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.3, row=4, col=1)
    fig.update_yaxes(title_text="DS (Tỷ lệ)", range=[0, 1], row=4, col=1)

    fig.update_xaxes(title_text="Ngày", row=4, col=1, fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    fig.update_layout(
        title_text="Money Cycle Market Dashboard (Interactive)",
        height=1200,
        hovermode="x unified",
        template="plotly_white",
        dragmode=None,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, config={"scrollZoom": True})
    logger.info(f"Saved interactive market dashboard: {output_path}")


def plot_symbol_dashboard_interactive(
    ohlcv: pd.DataFrame,
    chdm: pd.DataFrame,
    ds: pd.DataFrame,
    symbol: str,
    output_path: Path,
) -> None:
    """
    Interactive 3-panel symbol dashboard with Plotly.

    Args:
        ohlcv: OHLCV data
        chdm: CHDM by symbol
        ds: DS by symbol
        symbol: Stock symbol
        output_path: Output HTML file path
    """
    symbol = symbol.upper()

    ohlcv = ohlcv.copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    chdm = chdm.copy()
    chdm["date"] = pd.to_datetime(chdm["date"])
    ds = ds.copy()
    ds["date"] = pd.to_datetime(ds["date"])

    ohlcv_sym = ohlcv[ohlcv["symbol"] == symbol].sort_values("date")
    chdm_sym = chdm[chdm["symbol"] == symbol].sort_values("date")
    ds_sym = ds[ds["symbol"] == symbol].sort_values("date")

    if ohlcv_sym.empty:
        logger.warning(f"No data for {symbol}")
        return

    ohlcv_sym["MA50"] = ohlcv_sym["close"].rolling(50, min_periods=1).mean()
    ohlcv_sym["MA200"] = ohlcv_sym["close"].rolling(200, min_periods=1).mean()

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f"{symbol} Price + MA", f"{symbol} CHDM", f"{symbol} DS"),
        specs=[[{"secondary_y": False}] for _ in range(3)],
        vertical_spacing=0.08,
    )

    # Panel 1: Price
    fig.add_trace(
        go.Scatter(x=ohlcv_sym["date"], y=ohlcv_sym["close"], mode="lines",
                  name="Close", line=dict(color="black", width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ohlcv_sym["date"], y=ohlcv_sym["MA50"], mode="lines",
                  name="MA50", line=dict(color="blue", width=1.5, dash="dash")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=ohlcv_sym["date"], y=ohlcv_sym["MA200"], mode="lines",
                  name="MA200", line=dict(color="red", width=1.5, dash="dash")),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Price (VND)", row=1, col=1)

    # Panel 2: CHDM
    fig.add_trace(
        go.Scatter(x=chdm_sym["date"], y=chdm_sym["CHDM03"], mode="lines",
                  name="CHDM03", line=dict(color="blue", width=1)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=chdm_sym["date"], y=chdm_sym["CHDM20"], mode="lines",
                  name="CHDM20", line=dict(color="orange", width=1.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=chdm_sym["date"], y=chdm_sym["CHDM50"], mode="lines",
                  name="CHDM50", line=dict(color="green", width=1.5)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=chdm_sym["date"], y=chdm_sym["CHDM200"], mode="lines",
                  name="CHDM200", line=dict(color="red", width=1.5)),
        row=2, col=1
    )
    fig.add_hline(y=20, line_dash="dash", line_color="red", opacity=0.3, row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="green", opacity=0.3, row=2, col=1)
    fig.update_yaxes(title_text="CHDM (%)", range=[0, 100], row=2, col=1)

    # Panel 3: DS
    fig.add_trace(
        go.Scatter(x=ds_sym["date"], y=ds_sym["DS03"], mode="lines",
                  name="DS03", line=dict(color="blue", width=1)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=ds_sym["date"], y=ds_sym["DS20"], mode="lines",
                  name="DS20", line=dict(color="orange", width=1.5)),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=ds_sym["date"], y=ds_sym["DS200"], mode="lines",
                  name="DS200", line=dict(color="red", width=1.5)),
        row=3, col=1
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.3, row=3, col=1)
    fig.update_yaxes(title_text="DS (Sell State)", range=[0, 1], row=3, col=1)

    fig.update_xaxes(title_text="Ngày", row=3, col=1, fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    fig.update_layout(
        title_text=f"{symbol} - Money Cycle Dashboard (Interactive)",
        height=1000,
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
    symbol: str = "VCB",
) -> None:
    """
    Generate all interactive HTML charts.

    Args:
        data_dir: Path to money_cycle parquet files
        output_dir: Path to save HTML files
        symbol: Symbol for symbol dashboard
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    mc_market = pd.read_parquet(data_dir / "money_cycle_market.parquet")
    chdm_by_symbol = pd.read_parquet(data_dir / "chdm_by_symbol.parquet")
    ds_by_symbol = pd.read_parquet(data_dir / "ds_by_symbol.parquet")

    from multiagents_trading_assistant.data import ohlcv_store
    ohlcv = ohlcv_store.load()

    logger.info("Generating interactive charts...")

    # Market dashboard
    logger.info("1. Market dashboard (interactive)...")
    plot_market_dashboard_interactive(mc_market, output_dir / "01_market_dashboard_interactive.html")

    # Symbol dashboard
    logger.info(f"2. Symbol dashboard ({symbol}) (interactive)...")
    plot_symbol_dashboard_interactive(
        ohlcv, chdm_by_symbol, ds_by_symbol, symbol,
        output_dir / f"02_symbol_dashboard_{symbol}_interactive.html"
    )

    logger.info(f"All interactive charts saved to {output_dir}")
