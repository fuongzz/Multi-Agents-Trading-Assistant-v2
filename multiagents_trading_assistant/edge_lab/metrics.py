"""Metrics for edge research reports."""

from __future__ import annotations

import numpy as np
import pandas as pd


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0, "win_rate": 0.0, "avg_trade_pct": 0.0,
            "median_trade_pct": 0.0, "profit_factor": 0.0,
        }
    pnl = trades["pnl_pct"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.inf
    return {
        "trades": int(len(trades)),
        "symbols": int(trades["symbol"].nunique()) if "symbol" in trades else 0,
        "win_rate": float((pnl > 0).mean()),
        "avg_trade_pct": float(pnl.mean()),
        "median_trade_pct": float(pnl.median()),
        "avg_win_pct": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_pct": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(profit_factor),
        "best_trade_pct": float(pnl.max()),
        "worst_trade_pct": float(pnl.min()),
        "avg_hold_bars": float(trades["holding_bars"].mean()) if "holding_bars" in trades else 0.0,
    }


def equity_metrics(equity: pd.DataFrame, initial_capital: float) -> dict:
    if equity.empty:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    curve = equity["equity"].astype(float)
    returns = curve.pct_change().fillna(0.0)
    total_return = curve.iloc[-1] / initial_capital - 1.0
    max_drawdown = (curve / curve.cummax() - 1.0).min()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "final_equity": float(curve.iloc[-1]),
    }


def benchmark_metrics(price_map: dict[str, pd.DataFrame], start: str, end: str) -> dict:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frames = []
    for symbol, frame in price_map.items():
        data = frame[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)][["date", "close"]].copy()
        data = data.rename(columns={"close": symbol})
        frames.append(data.set_index("date"))
    if not frames:
        return {}
    prices = pd.concat(frames, axis=1).sort_index().ffill()
    daily_ret = prices.pct_change().dropna(how="all")
    ew_curve = (1.0 + daily_ret.mean(axis=1)).cumprod()
    common = prices.dropna(how="any")
    bh_curve = (common / common.iloc[0]).mean(axis=1) if not common.empty else ew_curve
    return {
        "benchmark_ew_return": float(ew_curve.iloc[-1] - 1.0) if not ew_curve.empty else 0.0,
        "benchmark_bh_return": float(bh_curve.iloc[-1] - 1.0) if not bh_curve.empty else 0.0,
        "benchmark_ew_maxdd": float((ew_curve / ew_curve.cummax() - 1.0).min()) if not ew_curve.empty else 0.0,
        "benchmark_bh_maxdd": float((bh_curve / bh_curve.cummax() - 1.0).min()) if not bh_curve.empty else 0.0,
    }

