"""Performance metrics for strategy ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    equity_curve: pd.Series,
    trades: list[dict],
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    returns = equity_curve.pct_change().fillna(0.0)
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
    sharpe = _sharpe(returns, periods_per_year)
    max_dd = max_drawdown(equity_curve)
    win_rate = _win_rate(trades)
    n_trades = len(trades)
    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "number_of_trades": n_trades,
    }


def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    drawdown = equity_curve / peak - 1.0
    return float(drawdown.min())


def score_metrics(metrics: dict, stability_penalty: float = 0.0) -> float:
    """QuantAgents ranking score with an instability penalty."""

    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    total_return = float(metrics.get("total_return", 0.0))
    drawdown_abs = abs(float(metrics.get("max_drawdown", 0.0)))
    return sharpe * 0.5 + total_return * 0.3 - drawdown_abs * 0.2 - stability_penalty


def _sharpe(returns: pd.Series, periods_per_year: int) -> float:
    std = returns.std(ddof=0)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((returns.mean() / std) * np.sqrt(periods_per_year))


def _win_rate(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for trade in trades if trade.get("pnl_pct", 0.0) > 0.0)
    return wins / len(trades)


def summarize_equity(equity_curve: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "equity": equity_curve,
            "return": equity_curve.pct_change().fillna(0.0),
            "drawdown": equity_curve / equity_curve.cummax() - 1.0,
        }
    )
