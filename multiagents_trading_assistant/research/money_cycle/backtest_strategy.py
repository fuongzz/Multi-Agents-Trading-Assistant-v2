"""
Money Cycle Backtest - Test trading strategies based on CHDM/DS indicators.

Strategies:
1. Washout reversal: CHDM < 20 trên thị trường → long signal toàn thị trường
2. Stock selection: Mua cổ phiếu có CHDM thấp + DS đang giảm
3. Cross-timeframe: CHDM200 > 50 làm bộ lọc dài hạn cho short-term trades
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str  # "TP", "SL", "TIMEOUT"

    @property
    def pnl_pct(self) -> float:
        return 100 * (self.exit_price - self.entry_price) / self.entry_price

    @property
    def days_held(self) -> int:
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    """Summary of backtest results."""
    strategy: str
    trades: list[Trade]

    @property
    def win_count(self) -> int:
        return sum(1 for t in self.trades if t.pnl_pct > 0)

    @property
    def loss_count(self) -> int:
        return sum(1 for t in self.trades if t.pnl_pct < 0)

    @property
    def win_rate(self) -> float:
        if len(self.trades) == 0:
            return 0.0
        return self.win_count / len(self.trades)

    @property
    def avg_win_pct(self) -> float:
        wins = [t.pnl_pct for t in self.trades if t.pnl_pct > 0]
        return np.mean(wins) if wins else 0.0

    @property
    def avg_loss_pct(self) -> float:
        losses = [t.pnl_pct for t in self.trades if t.pnl_pct < 0]
        return np.mean(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss."""
        total_wins = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        total_losses = abs(sum(t.pnl_pct for t in self.trades if t.pnl_pct < 0))
        if total_losses == 0:
            return float('inf') if total_wins > 0 else 0.0
        return total_wins / total_losses

    @property
    def total_return_pct(self) -> float:
        return sum(t.pnl_pct for t in self.trades)

    @property
    def avg_hold_days(self) -> float:
        days = [t.days_held for t in self.trades]
        return np.mean(days) if days else 0.0

    @property
    def max_drawdown_pct(self) -> float:
        """Maximum peak-to-trough drawdown."""
        if not self.trades:
            return 0.0
        cumulative = 0
        peak = 0
        max_dd = 0
        for trade in self.trades:
            cumulative += trade.pnl_pct
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_dd = max(max_dd, drawdown)
        return max_dd

    def print_report(self) -> None:
        """Print backtest report."""
        print(f"\n{'='*70}")
        print(f"BACKTEST RESULT: {self.strategy}")
        print(f"{'='*70}")
        print(f"Total Trades:     {len(self.trades)}")
        print(f"Wins/Losses:      {self.win_count}/{self.loss_count}")
        print(f"Win Rate:         {self.win_rate*100:.1f}%")
        print(f"Avg Win:          +{self.avg_win_pct:.2f}%")
        print(f"Avg Loss:         {self.avg_loss_pct:.2f}%")
        print(f"Profit Factor:    {self.profit_factor:.2f}x")
        print(f"Total Return:     {self.total_return_pct:+.2f}%")
        print(f"Avg Hold Days:    {self.avg_hold_days:.0f}")
        print(f"Max Drawdown:     {self.max_drawdown_pct:.2f}%")
        print(f"{'='*70}\n")


def backtest_washout_reversal(
    mc_market: pd.DataFrame,
    ohlcv: pd.DataFrame,
    hold_days: int = 10,
    chdm_threshold: float = 20.0,
) -> BacktestResult:
    """
    Backtest: Mua khi CHDM03 < 20 (washout), giữ N ngày.

    Args:
        mc_market: Market CHDM/DS data
        ohlcv: OHLCV data
        hold_days: Số ngày giữ vị thế (T+N)
        chdm_threshold: CHDM threshold cho washout

    Returns:
        BacktestResult
    """
    trades = []

    # Merge market signal vào OHLCV
    signals = mc_market[["date", "CHDM03"]].copy()
    signals["date"] = pd.to_datetime(signals["date"])

    # VNI proxy: tính từ tất cả stocks
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    daily_close = ohlcv.groupby("date")["close"].mean()  # Market average
    daily_close = daily_close.sort_index()

    # Identify washout entry dates
    signals = signals.sort_values("date")
    signal_dates = signals[signals["CHDM03"] < chdm_threshold]["date"].values

    for signal_date in signal_dates:
        signal_date = pd.Timestamp(signal_date)
        future_entry_dates = daily_close.index[daily_close.index > signal_date]
        if len(future_entry_dates) == 0:
            continue

        entry_date = future_entry_dates[0]
        exit_date_target = entry_date + pd.Timedelta(days=hold_days)

        # Signal is known after the signal date close, so enter next trading day.
        entry_price = daily_close[entry_date]

        # Find exit date (first date >= entry_date_target)
        exit_dates_after = daily_close.index[daily_close.index >= exit_date_target]
        if len(exit_dates_after) == 0:
            exit_date = daily_close.index[-1]
        else:
            exit_date = exit_dates_after[0]

        exit_price = daily_close[exit_date]

        trade = Trade(
            symbol="VNI_MARKET",
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_reason="TIMEOUT",
        )
        trades.append(trade)

    return BacktestResult(
        strategy=f"Washout Reversal (CHDM03 < {chdm_threshold}, next-day entry, hold {hold_days}d)",
        trades=trades,
    )


def backtest_stock_selection(
    chdm_by_symbol: pd.DataFrame,
    ds_by_symbol: pd.DataFrame,
    ohlcv: pd.DataFrame,
    hold_days: int = 10,
    top_n_stocks: int = 10,
    chdm_threshold: float = 30.0,
    ds_window: int = 50,
    rebalance_days: Optional[int] = None,
) -> BacktestResult:
    """
    Backtest: Mua top N cổ phiếu có CHDM thấp + DS đang giảm.

    Args:
        chdm_by_symbol: Per-stock CHDM
        ds_by_symbol: Per-stock DS
        ohlcv: OHLCV data
        hold_days: Số ngày giữ
        top_n_stocks: Số cổ phiếu mua
        chdm_threshold: CHDM threshold

    Returns:
        BacktestResult
    """
    trades = []

    ohlcv = ohlcv.copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv["symbol"] = ohlcv["symbol"].astype(str).str.upper()

    chdm_by_symbol = chdm_by_symbol.copy()
    chdm_by_symbol["date"] = pd.to_datetime(chdm_by_symbol["date"])
    chdm_by_symbol["symbol"] = chdm_by_symbol["symbol"].astype(str).str.upper()

    ds_col = f"DS{ds_window:02d}"
    if ds_col not in ds_by_symbol.columns:
        raise ValueError(f"Missing required DS column: {ds_col}")

    ds_by_symbol = ds_by_symbol.copy()
    ds_by_symbol["date"] = pd.to_datetime(ds_by_symbol["date"])
    ds_by_symbol["symbol"] = ds_by_symbol["symbol"].astype(str).str.upper()
    ds_by_symbol = ds_by_symbol.sort_values(["symbol", "date"])
    ds_by_symbol[f"{ds_col}_prev"] = ds_by_symbol.groupby("symbol", sort=False)[ds_col].shift(1)

    # Get unique dates
    dates = [pd.Timestamp(d) for d in sorted(ohlcv["date"].unique())]
    if rebalance_days is not None:
        if rebalance_days <= 0:
            raise ValueError("rebalance_days must be positive when provided")
        selected_dates = []
        last_date: pd.Timestamp | None = None
        for signal_date in dates:
            if last_date is None or (signal_date - last_date).days >= rebalance_days:
                selected_dates.append(signal_date)
                last_date = signal_date
        dates = selected_dates

    for date in dates:
        # Get CHDM/DS at this date for all stocks
        chdm_on_date = chdm_by_symbol[chdm_by_symbol["date"] == date]
        ds_on_date = ds_by_symbol[ds_by_symbol["date"] == date][
            ["symbol", ds_col, f"{ds_col}_prev"]
        ]

        if len(chdm_on_date) == 0 or len(ds_on_date) == 0:
            continue

        # Filter: CHDM < threshold and DS sell-state is improving.
        candidates = chdm_on_date.merge(ds_on_date, on="symbol", how="inner")
        ds_improving = (
            (candidates[f"{ds_col}_prev"].notna() & (candidates[ds_col] < candidates[f"{ds_col}_prev"]))
            | (candidates[ds_col] == 0)
        )
        candidates = candidates[
            (candidates["CHDM50"] < chdm_threshold)
            & ds_improving
        ].copy()
        if len(candidates) == 0:
            continue

        # Select top N by lowest CHDM50 among names whose DS state is improving.
        top_stocks = candidates.nsmallest(top_n_stocks, "CHDM50")["symbol"].values

        # Entry at this date's close
        for symbol in top_stocks:
            ohlcv_sym = ohlcv[ohlcv["symbol"] == symbol.upper()]
            dates_sym = sorted(ohlcv_sym["date"].unique())

            if date not in dates_sym:
                continue
            idx = dates_sym.index(date)
            if idx >= len(dates_sym) - 1:
                continue
            entry_date = dates_sym[idx + 1]  # Next trading day

            entry_row = ohlcv_sym[ohlcv_sym["date"] == entry_date]
            if entry_row.empty:
                continue

            entry_price = entry_row["close"].iloc[0]

            # Exit after hold_days
            exit_date_target = entry_date + pd.Timedelta(days=hold_days)
            future_dates = dates_sym[dates_sym.index(entry_date) + 1:]
            exit_dates = [d for d in future_dates if d >= exit_date_target]

            if not exit_dates:
                exit_date = dates_sym[-1]
            else:
                exit_date = exit_dates[0]

            exit_row = ohlcv_sym[ohlcv_sym["date"] == exit_date]
            if exit_row.empty:
                continue

            exit_price = exit_row["close"].iloc[0]

            trade = Trade(
                symbol=symbol,
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=exit_date,
                exit_price=exit_price,
                exit_reason="TIMEOUT",
            )
            trades.append(trade)

    return BacktestResult(
        strategy=(
            f"Stock Selection (CHDM50 < {chdm_threshold}, {ds_col} improving, "
            f"top {top_n_stocks}, hold {hold_days}d)"
        ),
        trades=trades,
    )


def run_backtests(
    data_dir: str | Path,
    ohlcv_path: Optional[str | Path] = None,
    ds_window: int = 50,
    rebalance_days: Optional[int] = None,
) -> None:
    """
    Run all backtest strategies.

    Args:
        data_dir: Path to money_cycle parquet files
        ohlcv_path: Path to OHLCV data (default: from ohlcv_store)
        ds_window: DS window used by stock-selection strategy
        rebalance_days: Optional calendar-day spacing between signal dates
    """
    data_dir = Path(data_dir)

    logger.info(f"Loading data from {data_dir}")
    mc_market = pd.read_parquet(data_dir / "money_cycle_market.parquet")
    chdm_by_symbol = pd.read_parquet(data_dir / "chdm_by_symbol.parquet")
    ds_by_symbol = pd.read_parquet(data_dir / "ds_by_symbol.parquet")

    # Load OHLCV
    if ohlcv_path:
        ohlcv = pd.read_parquet(ohlcv_path)
    else:
        from multiagents_trading_assistant.data import ohlcv_store
        ohlcv = ohlcv_store.load()

    logger.info("Running backtests...")

    # Strategy 1: Washout reversal
    result1 = backtest_washout_reversal(mc_market, ohlcv, hold_days=10)
    result1.print_report()

    # Strategy 2: Stock selection
    result2 = backtest_stock_selection(
        chdm_by_symbol, ds_by_symbol, ohlcv,
        hold_days=10,
        top_n_stocks=10,
        chdm_threshold=30.0,
        ds_window=ds_window,
        rebalance_days=rebalance_days,
    )
    result2.print_report()

    # Summary comparison
    print(f"\n{'='*70}")
    print("STRATEGY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Strategy':<50} {'Win %':>8} {'PF':>8} {'Return %':>10}")
    print(f"{'-'*70}")
    print(f"{result1.strategy:<50} {result1.win_rate*100:>7.1f}% {result1.profit_factor:>8.2f} {result1.total_return_pct:>9.2f}%")
    print(f"{result2.strategy:<50} {result2.win_rate*100:>7.1f}% {result2.profit_factor:>8.2f} {result2.total_return_pct:>9.2f}%")
    print(f"{'='*70}\n")
