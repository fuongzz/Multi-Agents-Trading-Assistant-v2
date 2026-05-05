"""Long-only full-allocation backtest engine with next-bar execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from multiagents_trading_assistant.quantagents_backtest.metrics import compute_metrics
from multiagents_trading_assistant.quantagents_backtest.strategy_generator import (
    Strategy,
    evaluate_strategy_signal,
)


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    periods_per_year: int = 252
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005


@dataclass(frozen=True)
class Trade:
    strategy_id: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    holding_bars: int


@dataclass(frozen=True)
class BacktestResult:
    strategy: Strategy | None
    equity_curve: pd.Series
    trades: list[dict]
    metrics: dict
    signals: pd.Series


def backtest_strategy(
    data: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Backtest one strategy using full allocation per trade.

    Entry and opposite exits use yesterday's close-of-bar signal and execute at
    today's open. Stop loss and take profit are checked against today's low/high.
    """

    cfg = config or BacktestConfig()
    _validate_backtest_data(data)

    raw_signal = evaluate_strategy_signal(data, strategy)
    tradable_signal = raw_signal.shift(1, fill_value=False).astype(bool)

    cash = cfg.initial_capital
    shares = 0.0
    entry_price = 0.0
    entry_time: pd.Timestamp | None = None
    holding_bars = 0
    trades: list[dict] = []
    equity_values: list[float] = []
    equity_index: list[pd.Timestamp] = []

    for timestamp, row in data.iterrows():
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        current_signal = bool(tradable_signal.loc[timestamp])

        if shares > 0:
            holding_bars += 1
            if not current_signal:
                cash, trade = _close_trade(
                    strategy,
                    shares,
                    entry_price,
                    entry_time,
                    timestamp,
                    open_price * (1 - cfg.slippage_rate),
                    "OPPOSITE_SIGNAL",
                    holding_bars,
                    cfg,
                )
                shares = 0.0
                trades.append(trade)
            else:
                stop_price = entry_price * (1 - strategy.risk.stop_loss)
                target_price = entry_price * (1 + strategy.risk.take_profit)
                exit_price: float | None = None
                exit_reason: str | None = None
                if low_price <= stop_price:
                    exit_price = stop_price * (1 - cfg.slippage_rate)
                    exit_reason = "STOP_LOSS"
                elif high_price >= target_price:
                    exit_price = target_price * (1 - cfg.slippage_rate)
                    exit_reason = "TAKE_PROFIT"
                elif holding_bars >= strategy.risk.max_holding_bars:
                    exit_price = close_price * (1 - cfg.slippage_rate)
                    exit_reason = "MAX_HOLDING"

                if exit_price is not None and exit_reason is not None:
                    cash, trade = _close_trade(
                        strategy,
                        shares,
                        entry_price,
                        entry_time,
                        timestamp,
                        exit_price,
                        exit_reason,
                        holding_bars,
                        cfg,
                    )
                    shares = 0.0
                    trades.append(trade)

        if shares == 0 and current_signal and np.isfinite(open_price) and open_price > 0:
            fill_price = open_price * (1 + cfg.slippage_rate)
            shares = cash * (1 - cfg.commission_rate) / fill_price
            entry_price = fill_price
            entry_time = timestamp
            cash = 0.0
            holding_bars = 0

        equity = cash if shares == 0 else shares * close_price
        equity_values.append(float(equity))
        equity_index.append(timestamp)

    if shares > 0 and entry_time is not None:
        timestamp = data.index[-1]
        exit_price = float(data["close"].iloc[-1]) * (1 - cfg.slippage_rate)
        cash, trade = _close_trade(
            strategy,
            shares,
            entry_price,
            entry_time,
            timestamp,
            exit_price,
            "END_OF_DATA",
            holding_bars,
            cfg,
        )
        trades.append(trade)
        equity_values[-1] = cash

    equity_curve = pd.Series(equity_values, index=equity_index, name=strategy.strategy_id)
    metrics = compute_metrics(equity_curve, trades, cfg.periods_per_year)
    return BacktestResult(strategy, equity_curve, trades, metrics, raw_signal)


def backtest_ensemble(
    data: pd.DataFrame,
    strategies: list[Strategy],
    config: BacktestConfig | None = None,
    mode: str = "equal_capital",
) -> BacktestResult:
    """Backtest top strategies in parallel and average their equity curves."""

    if not strategies:
        raise ValueError("At least one strategy is required for ensemble backtest")
    cfg = config or BacktestConfig()
    results = [backtest_strategy(data, strategy, cfg) for strategy in strategies]

    if mode == "average_signal":
        votes = pd.concat([result.signals for result in results], axis=1).fillna(False)
        synthetic_signal = votes.mean(axis=1) >= 0.5
        equity_curve = _equity_from_signal(data, synthetic_signal, cfg, name="ensemble")
        trades: list[dict] = []
        metrics = compute_metrics(equity_curve, trades, cfg.periods_per_year)
        return BacktestResult(None, equity_curve, trades, metrics, synthetic_signal)

    curves = pd.concat([result.equity_curve for result in results], axis=1).ffill()
    equity_curve = curves.mean(axis=1)
    equity_curve.name = "ensemble"
    trades = [trade for result in results for trade in result.trades]
    metrics = compute_metrics(equity_curve, trades, cfg.periods_per_year)
    signal = pd.concat([result.signals for result in results], axis=1).mean(axis=1) >= 0.5
    return BacktestResult(None, equity_curve, trades, metrics, signal)


def _close_trade(
    strategy: Strategy,
    shares: float,
    entry_price: float,
    entry_time: pd.Timestamp | None,
    exit_time: pd.Timestamp,
    exit_price: float,
    exit_reason: str,
    holding_bars: int,
    cfg: BacktestConfig,
) -> tuple[float, dict]:
    gross = shares * exit_price
    cash = gross * (1 - cfg.commission_rate)
    entry_value = shares * entry_price
    pnl = cash - entry_value
    pnl_pct = pnl / entry_value if entry_value else 0.0
    trade = Trade(
        strategy_id=strategy.strategy_id,
        entry_time=entry_time or exit_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        shares=shares,
        pnl=pnl,
        pnl_pct=pnl_pct,
        exit_reason=exit_reason,
        holding_bars=holding_bars,
    )
    return cash, trade.__dict__


def _equity_from_signal(
    data: pd.DataFrame,
    raw_signal: pd.Series,
    cfg: BacktestConfig,
    name: str,
) -> pd.Series:
    cash = cfg.initial_capital
    shares = 0.0
    entry_price = 0.0
    equity_values: list[float] = []
    signal = raw_signal.shift(1, fill_value=False).astype(bool)
    for timestamp, row in data.iterrows():
        open_price = float(row["open"])
        close_price = float(row["close"])
        if shares > 0 and not bool(signal.loc[timestamp]):
            cash = shares * open_price * (1 - cfg.slippage_rate) * (1 - cfg.commission_rate)
            shares = 0.0
        if shares == 0 and bool(signal.loc[timestamp]):
            entry_price = open_price * (1 + cfg.slippage_rate)
            shares = cash * (1 - cfg.commission_rate) / entry_price
            cash = 0.0
        equity_values.append(cash if shares == 0 else shares * close_price)
    result = pd.Series(equity_values, index=data.index, name=name)
    return result


def _validate_backtest_data(data: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Backtest data is missing columns: {sorted(missing)}")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Backtest data must use a DatetimeIndex")
