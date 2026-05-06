"""
Money Cycle Visualization - 4 biểu đồ phân tích chu kỳ tiền.

1. Market Dashboard (4-panel): CHDM/DS thị trường theo thời gian
2. CHDM Heatmap: 100 mã × ngày (thấy vị trí giá của từng mã)
3. Symbol Dashboard: Price + CHDM + DS của 1 mã cụ thể
4. Scatter: CHDM vs Forward Return (tương quan hiệu năng)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


def plot_market_dashboard(
    mc_market: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Plot 4-panel market dashboard: CHDM/DS over time.

    Args:
        mc_market: DataFrame với date, CHDM*, DS* columns
        output_path: Đường dẫn lưu PNG
    """
    mc_market = mc_market.copy()
    mc_market["date"] = pd.to_datetime(mc_market["date"])
    mc_market = mc_market.sort_values("date")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle("Money Cycle Market Dashboard", fontsize=16, fontweight="bold")

    # Panel 1: CHDM ngắn/trung/dài hạn
    ax = axes[0]
    ax.plot(mc_market["date"], mc_market["CHDM03"], label="CHDM03", color="blue", linewidth=1.5)
    ax.plot(mc_market["date"], mc_market["CHDM10"], label="CHDM10", color="orange", linewidth=1.5)
    ax.plot(mc_market["date"], mc_market["CHDM50"], label="CHDM50", color="green", linewidth=1.5)
    ax.plot(mc_market["date"], mc_market["CHDM200"], label="CHDM200", color="red", linewidth=1.5)
    ax.axhline(20, color="red", linestyle="--", alpha=0.3, linewidth=1, label="Washout (20)")
    ax.axhline(80, color="green", linestyle="--", alpha=0.3, linewidth=1, label="Distribution (80)")
    ax.set_ylabel("CHDM (%)", fontweight="bold")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    # Panel 2: DS ba chu kỳ
    ax = axes[1]
    ax.plot(mc_market["date"], mc_market["DS03"], label="DS03", color="blue", linewidth=1.5)
    ax.plot(mc_market["date"], mc_market["DS50"], label="DS50", color="orange", linewidth=1.5)
    ax.plot(mc_market["date"], mc_market["DS200"], label="DS200", color="red", linewidth=1.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    ax.set_ylabel("DS (Tỷ lệ bị bán)", fontweight="bold")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Panel 3: CHDM20 với vùng washout/distribution highlight
    ax = axes[2]
    ax.plot(mc_market["date"], mc_market["CHDM20"], label="CHDM20", color="purple", linewidth=2)
    washout = mc_market["CHDM20"] < 20
    distribution = mc_market["CHDM20"] > 80
    ax.fill_between(mc_market["date"], 0, 100, where=washout, color="red", alpha=0.1, label="Washout Zone")
    ax.fill_between(mc_market["date"], 0, 100, where=distribution, color="green", alpha=0.1, label="Distribution Zone")
    ax.axhline(20, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(80, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_ylabel("CHDM20 (%)", fontweight="bold")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    # Panel 4: DS20 vs DS200 (so ngắn vs dài hạn)
    ax = axes[3]
    ax.plot(mc_market["date"], mc_market["DS20"], label="DS20 (ngắn hạn)", color="orange", linewidth=2)
    ax.plot(mc_market["date"], mc_market["DS200"], label="DS200 (dài hạn)", color="red", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Ngày", fontweight="bold")
    ax.set_ylabel("DS (Tỷ lệ)", fontweight="bold")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved market dashboard: {output_path}")
    plt.close()


def plot_chdm_heatmap(
    chdm_by_symbol: pd.DataFrame,
    window: int,
    output_path: Path,
) -> None:
    """
    Plot heatmap: 100 symbols × dates, color by CHDM value.

    Args:
        chdm_by_symbol: DataFrame với date, symbol, CHDM* columns
        window: Window để plot (20, 50, 200)
        output_path: Đường dẫn lưu PNG
    """
    col_name = f"CHDM{window:02d}"
    if col_name not in chdm_by_symbol.columns:
        logger.error(f"Column {col_name} not found")
        return

    # Pivot: symbols × dates
    pivot_df = chdm_by_symbol.pivot(index="symbol", columns="date", values=col_name)

    # Sample dates (monthly) để heatmap không quá rộng
    dates_list = pivot_df.columns.tolist()
    step = max(1, len(dates_list) // 50)  # Max 50 columns
    pivot_df = pivot_df.iloc[:, ::step]

    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        pivot_df,
        cmap="RdYlGn",
        cbar_kws={"label": f"CHDM{window:02d} (%)"},
        ax=ax,
        vmin=0,
        vmax=100,
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(f"CHDM{window:02d} Heatmap - 100 Symbols (Green=Low, Red=High)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Symbol", fontweight="bold")

    # Format x-axis dates
    xticks = ax.get_xticks()
    xticklabels = [pivot_df.columns[int(i)].strftime("%Y-%m-%d") if i < len(pivot_df.columns) else "" for i in xticks]
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    logger.info(f"Saved CHDM heatmap: {output_path}")
    plt.close()


def plot_symbol_dashboard(
    ohlcv: pd.DataFrame,
    chdm: pd.DataFrame,
    ds: pd.DataFrame,
    symbol: str,
    output_path: Path,
) -> None:
    """
    Plot 3-panel dashboard for one symbol: Price + CHDM + DS.

    Args:
        ohlcv: OHLCV data with date, symbol, close, volume
        chdm: CHDM by symbol with CHDM* columns
        ds: DS by symbol with DS* columns
        symbol: Mã cổ phiếu (VCB, FPT, ...)
        output_path: Đường dẫn lưu PNG
    """
    symbol = symbol.upper()

    # Filter data cho symbol
    ohlcv_sym = ohlcv[ohlcv["symbol"] == symbol].sort_values("date").copy()
    ohlcv_sym["date"] = pd.to_datetime(ohlcv_sym["date"])
    chdm_sym = chdm[chdm["symbol"] == symbol].sort_values("date").copy()
    chdm_sym["date"] = pd.to_datetime(chdm_sym["date"])
    ds_sym = ds[ds["symbol"] == symbol].sort_values("date").copy()
    ds_sym["date"] = pd.to_datetime(ds_sym["date"])

    if ohlcv_sym.empty or chdm_sym.empty:
        logger.warning(f"No data for symbol {symbol}")
        return

    # Compute MA
    ohlcv_sym["MA50"] = ohlcv_sym["close"].rolling(50, min_periods=1).mean()
    ohlcv_sym["MA200"] = ohlcv_sym["close"].rolling(200, min_periods=1).mean()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f"{symbol} - Money Cycle Dashboard", fontsize=16, fontweight="bold")

    # Panel 1: Price + MA
    ax = axes[0]
    ax.plot(ohlcv_sym["date"], ohlcv_sym["close"], label="Close", color="black", linewidth=1.5)
    ax.plot(ohlcv_sym["date"], ohlcv_sym["MA50"], label="MA50", color="blue", linewidth=1.5, alpha=0.7)
    ax.plot(ohlcv_sym["date"], ohlcv_sym["MA200"], label="MA200", color="red", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Price (VND)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: CHDM
    ax = axes[1]
    ax.plot(chdm_sym["date"], chdm_sym["CHDM03"], label="CHDM03", color="blue", linewidth=1)
    ax.plot(chdm_sym["date"], chdm_sym["CHDM20"], label="CHDM20", color="orange", linewidth=1.5)
    ax.plot(chdm_sym["date"], chdm_sym["CHDM50"], label="CHDM50", color="green", linewidth=1.5)
    ax.plot(chdm_sym["date"], chdm_sym["CHDM200"], label="CHDM200", color="red", linewidth=1.5)
    ax.axhline(20, color="red", linestyle="--", alpha=0.3)
    ax.axhline(80, color="green", linestyle="--", alpha=0.3)
    ax.set_ylabel("CHDM (%)", fontweight="bold")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    # Panel 3: DS
    ax = axes[2]
    ax.plot(ds_sym["date"], ds_sym["DS03"], label="DS03", color="blue", linewidth=1)
    ax.plot(ds_sym["date"], ds_sym["DS20"], label="DS20", color="orange", linewidth=1.5)
    ax.plot(ds_sym["date"], ds_sym["DS200"], label="DS200", color="red", linewidth=1.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("DS (Sell State)", fontweight="bold")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved symbol dashboard: {output_path}")
    plt.close()


def plot_scatter_chdm_vs_return(
    chdm_by_symbol: pd.DataFrame,
    ohlcv: pd.DataFrame,
    chdm_window: int,
    fwd_days: int,
    output_path: Path,
) -> None:
    """
    Plot scatter: CHDM_N vs forward return T+N.

    Args:
        chdm_by_symbol: CHDM data
        ohlcv: OHLCV data
        chdm_window: Window (50 cho T+10, 200 cho T+20)
        fwd_days: Forward days (10 hoặc 20)
        output_path: Đường dẫn lưu PNG
    """
    col_chdm = f"CHDM{chdm_window:02d}"

    # Merge CHDM + OHLCV
    chdm_by_symbol = chdm_by_symbol[["date", "symbol", col_chdm]].copy()
    chdm_by_symbol["date"] = pd.to_datetime(chdm_by_symbol["date"])
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])

    merged = pd.merge(
        ohlcv[["date", "symbol", "close"]],
        chdm_by_symbol,
        on=["date", "symbol"],
    )

    # Compute forward return
    merged = merged.sort_values(["symbol", "date"])
    merged["close_fwd"] = merged.groupby("symbol")["close"].shift(-fwd_days)
    merged["return_pct"] = 100 * (merged["close_fwd"] - merged["close"]) / merged["close"]

    # Remove NaN
    merged_clean = merged.dropna(subset=[col_chdm, "return_pct"])

    if len(merged_clean) < 10:
        logger.warning(f"Insufficient data for scatter plot")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        merged_clean[col_chdm],
        merged_clean["return_pct"],
        alpha=0.4,
        s=20,
        c=merged_clean["return_pct"],
        cmap="RdYlGn",
        vmin=merged_clean["return_pct"].quantile(0.05),
        vmax=merged_clean["return_pct"].quantile(0.95),
    )

    # Regression line
    z = np.polyfit(merged_clean[col_chdm], merged_clean["return_pct"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged_clean[col_chdm].min(), merged_clean[col_chdm].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Trend line")

    # Stats
    corr = merged_clean[col_chdm].corr(merged_clean["return_pct"])
    ax.text(
        0.05, 0.95, f"Correlation: {corr:.3f}\nN: {len(merged_clean)}",
        transform=ax.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(f"{col_chdm} (%)", fontweight="bold")
    ax.set_ylabel(f"Forward Return T+{fwd_days} (%)", fontweight="bold")
    ax.set_title(f"CHDM vs Forward Return (T+{fwd_days})", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax, label="Return (%)")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved scatter plot: {output_path}")
    plt.close()


def run_all_charts(
    data_dir: str | Path,
    output_dir: str | Path,
    symbol: str = "VCB",
) -> None:
    """
    Run all 4 charts.

    Args:
        data_dir: Path to money_cycle parquet files
        output_dir: Path to save PNG files
        symbol: Symbol cho dashboard (VCB, FPT, ...)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {data_dir}")

    # Load market data
    mc_market = pd.read_parquet(data_dir / "money_cycle_market.parquet")
    chdm_by_symbol = pd.read_parquet(data_dir / "chdm_by_symbol.parquet")
    ds_by_symbol = pd.read_parquet(data_dir / "ds_by_symbol.parquet")

    # Load OHLCV
    from multiagents_trading_assistant.data import ohlcv_store
    ohlcv = ohlcv_store.load()

    logger.info("Generating 4 charts...")

    # Chart 1: Market dashboard
    logger.info("1. Market dashboard...")
    plot_market_dashboard(mc_market, output_dir / "01_market_dashboard.png")

    # Chart 2: CHDM heatmap (windows 20, 50, 200)
    for window in [20, 50, 200]:
        logger.info(f"2. CHDM{window:02d} heatmap...")
        plot_chdm_heatmap(chdm_by_symbol, window, output_dir / f"02_chdm_heatmap_{window:03d}.png")

    # Chart 3: Symbol dashboard
    logger.info(f"3. Symbol dashboard ({symbol})...")
    plot_symbol_dashboard(
        ohlcv, chdm_by_symbol, ds_by_symbol, symbol,
        output_dir / f"03_symbol_dashboard_{symbol}.png"
    )

    # Chart 4: Scatter plots
    logger.info("4. Scatter plots...")
    plot_scatter_chdm_vs_return(chdm_by_symbol, ohlcv, 50, 10, output_dir / "04_scatter_chdm50_vs_return_T10.png")
    plot_scatter_chdm_vs_return(chdm_by_symbol, ohlcv, 200, 20, output_dir / "04_scatter_chdm200_vs_return_T20.png")

    logger.info(f"All charts saved to {output_dir}")
