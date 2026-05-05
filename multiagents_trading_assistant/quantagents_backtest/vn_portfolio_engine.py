"""Portfolio-level VN execution engine for QuantAgents strategies.

This engine converts the research backtests into a closer-to-real portfolio
simulation: one shared capital pool, daily universe scan, position limits,
VN-style fees/tax, basic liquidity participation, T+ sell lock, price-band
awareness, and market-regime-aware sizing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from multiagents_trading_assistant.quantagents_backtest.indicators import add_indicators
from multiagents_trading_assistant.quantagents_backtest.metrics import compute_metrics
from multiagents_trading_assistant.quantagents_backtest.strategy_generator import (
    Strategy,
    evaluate_strategy_signal,
)


@dataclass(frozen=True)
class VNMarketCostConfig:
    commission_rate: float = 0.001
    sell_tax_rate: float = 0.001
    slippage_rate: float = 0.0005
    lot_size: int = 100
    max_participation_rate: float = 0.10
    hose_price_band: float = 0.07
    settlement_bars: int = 2


@dataclass(frozen=True)
class VNPortfolioConfig:
    initial_capital: float = 100_000.0
    max_positions: int = 8
    min_position_value: float = 1_000.0
    periods_per_year: int = 252
    risk_on_exposure: float = 1.0
    neutral_exposure: float = 0.7
    risk_off_exposure: float = 0.3
    crash_exposure: float = 0.0
    allow_new_in_risk_off: bool = False
    mean_reversion_requires_trend: bool = True
    costs: VNMarketCostConfig = VNMarketCostConfig()


@dataclass
class VNPosition:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    shares: int
    entry_value: float
    stop_loss: float
    take_profit: float
    max_holding_bars: int
    strategy_ids: tuple[str, ...]
    family_votes: tuple[str, ...]
    holding_bars: int = 0


def backtest_vn_portfolio(
    universe_data: dict[str, pd.DataFrame],
    strategies: list[Strategy],
    market_regime: pd.Series | None = None,
    config: VNPortfolioConfig | None = None,
) -> dict:
    """Backtest top strategies as one VN portfolio with shared capital."""

    cfg = config or VNPortfolioConfig()
    prepared = _prepare_universe(universe_data, strategies, market_regime)
    if not prepared:
        raise ValueError("No usable OHLCV data for portfolio backtest")

    calendar = sorted(set().union(*(frame.index for frame in prepared.values())))
    cash = cfg.initial_capital
    positions: dict[str, VNPosition] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []

    for timestamp in calendar:
        regime = _regime_at(market_regime, timestamp)
        target_exposure = _target_exposure(regime, cfg)
        allow_new = _allow_new_positions(regime, cfg)

        # Exit first so freed cash can be reused by new entries on the same day.
        for symbol in list(positions):
            frame = prepared.get(symbol)
            if frame is None or timestamp not in frame.index:
                continue
            row = frame.loc[timestamp]
            position = positions[symbol]
            position.holding_bars += 1
            exit_reason = _exit_reason(row, position, regime)
            if exit_reason and _can_sell(row, position, cfg):
                cash, trade = _sell_position(
                    timestamp,
                    row,
                    position,
                    cash,
                    exit_reason,
                    cfg,
                )
                trades.append(trade)
                del positions[symbol]

        if allow_new and target_exposure > 0:
            candidates = _entry_candidates(prepared, strategies, timestamp, regime, cfg)
            open_slots = max(0, cfg.max_positions - len(positions))
            candidates = [candidate for candidate in candidates if candidate["symbol"] not in positions]
            candidates = candidates[:open_slots]
            if candidates:
                equity_before_entries = cash + _mark_to_market(positions, prepared, timestamp)
                target_capital = equity_before_entries * target_exposure
                current_exposure = _position_value(positions, prepared, timestamp)
                deployable = max(0.0, min(cash, target_capital - current_exposure))
                per_trade_budget = deployable / len(candidates) if candidates else 0.0
                for candidate in candidates:
                    if per_trade_budget < cfg.min_position_value:
                        continue
                    cash, position = _buy_position(
                        timestamp,
                        prepared[candidate["symbol"]].loc[timestamp],
                        candidate,
                        per_trade_budget,
                        cash,
                        cfg,
                    )
                    if position is not None:
                        positions[position.symbol] = position

        equity = cash + _mark_to_market(positions, prepared, timestamp)
        equity_rows.append(
            {
                "date": timestamp,
                "equity": equity,
                "cash": cash,
                "positions": len(positions),
                "regime": regime,
                "gross_exposure": _position_value(positions, prepared, timestamp) / equity if equity else 0.0,
            }
        )

    # Liquidate final positions for clean trade logs when possible.
    final_time = pd.Timestamp(calendar[-1])
    for symbol in list(positions):
        frame = prepared[symbol]
        if final_time in frame.index and _can_sell(frame.loc[final_time], positions[symbol], cfg, force=True):
            cash, trade = _sell_position(
                final_time,
                frame.loc[final_time],
                positions[symbol],
                cash,
                "END_OF_DATA",
                cfg,
            )
            trades.append(trade)
            del positions[symbol]
    if equity_rows:
        equity_rows[-1]["equity"] = cash + _mark_to_market(positions, prepared, final_time)
        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["positions"] = len(positions)
        equity_rows[-1]["gross_exposure"] = (
            _position_value(positions, prepared, final_time) / equity_rows[-1]["equity"]
            if equity_rows[-1]["equity"]
            else 0.0
        )

    equity_frame = pd.DataFrame(equity_rows).set_index("date")
    equity_curve = equity_frame["equity"].rename("vn_quantagents_portfolio")
    metrics = compute_metrics(equity_curve, trades, cfg.periods_per_year)
    return {
        "equity_curve": equity_curve,
        "equity_frame": equity_frame,
        "trades": trades,
        "metrics": metrics,
        "open_positions": positions,
    }


def _prepare_universe(
    universe_data: dict[str, pd.DataFrame],
    strategies: list[Strategy],
    market_regime: pd.Series | None,
) -> dict[str, pd.DataFrame]:
    prepared = {}
    for symbol, raw in universe_data.items():
        if raw.empty:
            continue
        frame = add_indicators(raw)
        signals = []
        for strategy in strategies:
            signal = evaluate_strategy_signal(frame, strategy)
            signal = _apply_vn_strategy_filter(frame, signal, strategy, market_regime)
            signals.append(signal.rename(strategy.strategy_id))
        signal_frame = pd.concat(signals, axis=1).fillna(False) if signals else pd.DataFrame(index=frame.index)
        frame = frame.copy()
        frame["qa_vote_count"] = signal_frame.sum(axis=1)
        frame["qa_signal"] = frame["qa_vote_count"] > 0
        frame["_signal_frame"] = list(signal_frame.to_dict("records"))
        prepared[symbol] = frame
    return prepared


def _apply_vn_strategy_filter(
    frame: pd.DataFrame,
    signal: pd.Series,
    strategy: Strategy,
    market_regime: pd.Series | None,
) -> pd.Series:
    filtered = signal.fillna(False).astype(bool)
    if strategy.family == "mean_reversion":
        # Avoid catching falling knives in broad market stress or broken trends.
        regime = market_regime.reindex(frame.index).ffill().bfill() if market_regime is not None else None
        if regime is not None:
            filtered &= ~regime.isin(["RISK_OFF", "CRASH"])
        if "ema200" in frame.columns:
            filtered &= frame["close"] >= frame["ema200"] * 0.92
        if "breakdown_20_low" in frame.columns:
            filtered &= ~frame["breakdown_20_low"].fillna(False)
        if "adx_14" in frame.columns:
            filtered &= frame["adx_14"].fillna(0) < 35
    return filtered


def _entry_candidates(
    prepared: dict[str, pd.DataFrame],
    strategies: list[Strategy],
    timestamp: pd.Timestamp,
    regime: str,
    cfg: VNPortfolioConfig,
) -> list[dict]:
    candidates = []
    by_id = {strategy.strategy_id: strategy for strategy in strategies}
    for symbol, frame in prepared.items():
        if timestamp not in frame.index:
            continue
        row = frame.loc[timestamp]
        if not bool(row.get("qa_signal", False)):
            continue
        if not _can_buy(row, cfg):
            continue
        signal_record = row.get("_signal_frame", {})
        active_ids = tuple(strategy_id for strategy_id, active in signal_record.items() if active)
        if not active_ids:
            continue
        active_strategies = [by_id[strategy_id] for strategy_id in active_ids if strategy_id in by_id]
        risk = _blend_risk(active_strategies)
        families = tuple(sorted({strategy.family for strategy in active_strategies}))
        candidates.append(
            {
                "symbol": symbol,
                "vote_count": int(row["qa_vote_count"]),
                "momentum": float(row.get("roc_9", 0.0) if pd.notna(row.get("roc_9", np.nan)) else 0.0),
                "active_ids": active_ids,
                "families": families,
                "risk": risk,
                "regime": regime,
            }
        )
    candidates.sort(key=lambda item: (item["vote_count"], item["momentum"]), reverse=True)
    return candidates


def _blend_risk(strategies: list[Strategy]) -> dict:
    if not strategies:
        return {"stop_loss": 0.08, "take_profit": 0.18, "max_holding_bars": 60}
    return {
        "stop_loss": float(np.median([strategy.risk.stop_loss for strategy in strategies])),
        "take_profit": float(np.median([strategy.risk.take_profit for strategy in strategies])),
        "max_holding_bars": int(np.median([strategy.risk.max_holding_bars for strategy in strategies])),
    }


def _buy_position(
    timestamp: pd.Timestamp,
    row: pd.Series,
    candidate: dict,
    budget: float,
    cash: float,
    cfg: VNPortfolioConfig,
) -> tuple[float, VNPosition | None]:
    open_price = float(row["open"])
    if not np.isfinite(open_price) or open_price <= 0:
        return cash, None
    fill_price = open_price * (1 + cfg.costs.slippage_rate)
    max_by_liquidity = float(row["volume"]) * cfg.costs.max_participation_rate
    gross_shares = min(budget / fill_price, max_by_liquidity)
    shares = int(gross_shares // cfg.costs.lot_size * cfg.costs.lot_size)
    if shares <= 0:
        return cash, None
    gross = shares * fill_price
    total_cost = gross * (1 + cfg.costs.commission_rate)
    if total_cost > cash:
        shares = int((cash / (fill_price * (1 + cfg.costs.commission_rate))) // cfg.costs.lot_size * cfg.costs.lot_size)
        gross = shares * fill_price
        total_cost = gross * (1 + cfg.costs.commission_rate)
    if shares <= 0 or gross < cfg.min_position_value:
        return cash, None
    risk = candidate["risk"]
    position = VNPosition(
        symbol=candidate["symbol"],
        entry_time=timestamp,
        entry_price=fill_price,
        shares=shares,
        entry_value=gross,
        stop_loss=float(risk["stop_loss"]),
        take_profit=float(risk["take_profit"]),
        max_holding_bars=int(risk["max_holding_bars"]),
        strategy_ids=tuple(candidate["active_ids"]),
        family_votes=tuple(candidate["families"]),
    )
    return cash - total_cost, position


def _sell_position(
    timestamp: pd.Timestamp,
    row: pd.Series,
    position: VNPosition,
    cash: float,
    exit_reason: str,
    cfg: VNPortfolioConfig,
) -> tuple[float, dict]:
    open_price = float(row["open"])
    fill_price = open_price * (1 - cfg.costs.slippage_rate)
    gross = position.shares * fill_price
    fees = gross * (cfg.costs.commission_rate + cfg.costs.sell_tax_rate)
    net = gross - fees
    pnl = net - position.entry_value
    pnl_pct = pnl / position.entry_value if position.entry_value else 0.0
    trade = {
        "symbol": position.symbol,
        "entry_time": position.entry_time,
        "exit_time": timestamp,
        "entry_price": position.entry_price,
        "exit_price": fill_price,
        "shares": position.shares,
        "entry_value": position.entry_value,
        "exit_value": net,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "exit_reason": exit_reason,
        "holding_bars": position.holding_bars,
        "strategy_ids": ",".join(position.strategy_ids),
        "family_votes": ",".join(position.family_votes),
    }
    return cash + net, trade


def _exit_reason(row: pd.Series, position: VNPosition, regime: str) -> str | None:
    if regime == "CRASH":
        return "REGIME_CRASH"
    close = float(row["close"])
    low = float(row["low"])
    high = float(row["high"])
    if low <= position.entry_price * (1 - position.stop_loss):
        return "STOP_LOSS"
    if high >= position.entry_price * (1 + position.take_profit):
        return "TAKE_PROFIT"
    if position.holding_bars >= position.max_holding_bars:
        return "MAX_HOLDING"
    if not bool(row.get("qa_signal", False)) and close < position.entry_price:
        return "SIGNAL_OFF"
    return None


def _can_buy(row: pd.Series, cfg: VNPortfolioConfig) -> bool:
    if not np.isfinite(float(row["open"])) or float(row["volume"]) <= 0:
        return False
    prev_close = row.get("close", np.nan) / (1 + row.get("close_return", 0.0)) if row.get("close_return", np.nan) else np.nan
    if np.isfinite(prev_close) and prev_close > 0:
        ceiling = prev_close * (1 + cfg.costs.hose_price_band)
        if float(row["open"]) >= ceiling * 0.999:
            return False
    return True


def _can_sell(
    row: pd.Series,
    position: VNPosition,
    cfg: VNPortfolioConfig,
    force: bool = False,
) -> bool:
    if not force and position.holding_bars < cfg.costs.settlement_bars:
        return False
    prev_close = row.get("close", np.nan) / (1 + row.get("close_return", 0.0)) if row.get("close_return", np.nan) else np.nan
    if np.isfinite(prev_close) and prev_close > 0:
        floor = prev_close * (1 - cfg.costs.hose_price_band)
        if float(row["open"]) <= floor * 1.001:
            return False
    return True


def _target_exposure(regime: str, cfg: VNPortfolioConfig) -> float:
    return {
        "RISK_ON": cfg.risk_on_exposure,
        "NEUTRAL": cfg.neutral_exposure,
        "RISK_OFF": cfg.risk_off_exposure,
        "CRASH": cfg.crash_exposure,
    }.get(regime, cfg.neutral_exposure)


def _allow_new_positions(regime: str, cfg: VNPortfolioConfig) -> bool:
    if regime == "CRASH":
        return False
    if regime == "RISK_OFF" and not cfg.allow_new_in_risk_off:
        return False
    return True


def _regime_at(regime: pd.Series | None, timestamp: pd.Timestamp) -> str:
    if regime is None or regime.empty:
        return "NEUTRAL"
    aligned = regime.loc[:timestamp]
    if aligned.empty:
        return "NEUTRAL"
    return str(aligned.iloc[-1])


def _mark_to_market(
    positions: dict[str, VNPosition],
    prepared: dict[str, pd.DataFrame],
    timestamp: pd.Timestamp,
) -> float:
    value = 0.0
    for symbol, position in positions.items():
        frame = prepared.get(symbol)
        if frame is None:
            continue
        rows = frame.loc[:timestamp]
        if rows.empty:
            value += position.shares * position.entry_price
        else:
            value += position.shares * float(rows["close"].iloc[-1])
    return value


def _position_value(
    positions: dict[str, VNPosition],
    prepared: dict[str, pd.DataFrame],
    timestamp: pd.Timestamp,
) -> float:
    return _mark_to_market(positions, prepared, timestamp)
