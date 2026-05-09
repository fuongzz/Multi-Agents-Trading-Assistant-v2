"""Generic signal-quality and portfolio backtests for edge hypotheses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from multiagents_trading_assistant.edge_lab.hypothesis import (
    Hypothesis,
    evaluate_filters,
    rank_candidates,
)


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 100_000.0
    max_positions: int = 8
    top_n: int = 5
    commission_rate: float = 0.001
    sell_tax_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_participation_rate: float = 0.10
    lot_size: int = 100
    min_position_value: float = 1_000.0
    stop_loss: float = 0.08
    take_profit: float = 0.25
    max_holding_bars: int = 60
    risk_on_exposure: float = 1.0
    neutral_exposure: float = 0.7
    risk_off_exposure: float = 0.3


@dataclass
class Position:
    hypothesis: str
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    entry_value: float
    risk: dict
    holding_bars: int = 0


def run_signal_quality(
    features: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    hypotheses: list[Hypothesis],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    holds: list[int] | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    holds = holds or [10, 20, 30, 60]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    date_index = _date_index(price_map)
    rows = []
    daily = features[(features["date"] >= start_ts) & (features["date"] <= end_ts)]
    for hypothesis in hypotheses:
        for signal_date, day in daily.groupby("date", sort=True):
            candidates = rank_candidates(day[evaluate_filters(day, hypothesis)], hypothesis).head(top_n)
            for candidate in candidates.to_dict("records"):
                symbol = candidate["symbol"]
                idx = date_index.get(symbol, {}).get(pd.Timestamp(signal_date))
                if idx is None or idx + 1 >= len(price_map[symbol]):
                    continue
                entry_idx = idx + 1
                entry = price_map[symbol].iloc[entry_idx]
                if pd.Timestamp(entry["date"]) > end_ts:
                    continue
                entry_price = _entry_price(entry)
                for hold in holds:
                    exit_idx = min(entry_idx + hold, len(price_map[symbol]) - 1)
                    exit_row = price_map[symbol].iloc[exit_idx]
                    exit_price = float(exit_row["close"])
                    rows.append(
                        {
                            "hypothesis": hypothesis.name,
                            "symbol": symbol,
                            "signal_date": pd.Timestamp(signal_date),
                            "entry_date": pd.Timestamp(entry["date"]),
                            "exit_date": pd.Timestamp(exit_row["date"]),
                            "hold": hold,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl_pct": 100.0 * (exit_price - entry_price) / entry_price,
                            "holding_bars": exit_idx - entry_idx,
                            "rank": candidate.get("_edge_rank", np.nan),
                        }
                    )
    return pd.DataFrame(rows)


def run_portfolio(
    features: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    hypotheses: list[Hypothesis],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    config: PortfolioConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or PortfolioConfig()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    date_index = _date_index(price_map)
    feature_history = _feature_history(features)
    calendar = sorted(features[(features["date"] >= start_ts) & (features["date"] <= end_ts)]["date"].unique())

    cash = cfg.initial_capital
    positions: dict[str, Position] = {}
    pending: dict[pd.Timestamp, list[dict]] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []

    for raw_date in calendar:
        date = pd.Timestamp(raw_date)

        for symbol in list(positions):
            idx = date_index.get(symbol, {}).get(date)
            if idx is None:
                continue
            position = positions[symbol]
            position.holding_bars += 1
            row = price_map[symbol].iloc[idx]
            prior_features = _latest_features_before(feature_history, symbol, date)
            reason = _exit_reason(row, prior_features, position, cfg)
            if reason:
                cash, trade = _sell(date, row, position, cash, reason, cfg)
                trades.append(trade)
                del positions[symbol]

        todays_orders = pending.pop(date, [])
        if todays_orders:
            candidates = pd.DataFrame(todays_orders).sort_values("_edge_rank", ascending=False)
            candidates = candidates[~candidates["symbol"].isin(positions)]
            open_slots = max(0, cfg.max_positions - len(positions))
            candidates = candidates.head(min(cfg.top_n, open_slots))
            if not candidates.empty:
                equity = cash + _mark_to_market(positions, price_map, date)
                exposure = _target_exposure(candidates)
                current_exposure = _mark_to_market(positions, price_map, date)
                deployable = max(0.0, min(cash, equity * exposure - current_exposure))
                per_trade_budget = deployable / len(candidates) if len(candidates) else 0.0
                for candidate in candidates.to_dict("records"):
                    symbol = candidate["symbol"]
                    idx = date_index.get(symbol, {}).get(date)
                    if idx is None:
                        continue
                    cash, position = _buy(date, price_map[symbol].iloc[idx], candidate, per_trade_budget, cash, cfg)
                    if position is not None:
                        positions[symbol] = position

        day = features[features["date"] == date]
        for hypothesis in hypotheses:
            signals = rank_candidates(day[evaluate_filters(day, hypothesis)], hypothesis)
            for signal in signals.head(cfg.top_n).to_dict("records"):
                symbol = signal["symbol"]
                idx = date_index.get(symbol, {}).get(date)
                if idx is None or idx + 1 >= len(price_map[symbol]):
                    continue
                next_date = pd.Timestamp(price_map[symbol].iloc[idx + 1]["date"])
                if next_date > end_ts:
                    continue
                signal["_hypothesis"] = hypothesis.name
                signal["_risk"] = hypothesis.risk
                pending.setdefault(next_date, []).append(signal)

        equity = cash + _mark_to_market(positions, price_map, date)
        equity_rows.append(
            {
                "date": date,
                "equity": equity,
                "cash": cash,
                "positions": len(positions),
                "gross_exposure": _mark_to_market(positions, price_map, date) / equity if equity else 0.0,
            }
        )

    if calendar:
        final_date = pd.Timestamp(calendar[-1])
        for symbol in list(positions):
            idx = date_index[symbol].get(final_date)
            if idx is not None:
                cash, trade = _sell(final_date, price_map[symbol].iloc[idx], positions[symbol], cash, "END_OF_DATA", cfg)
                trades.append(trade)
                del positions[symbol]
        if equity_rows:
            equity_rows[-1]["equity"] = cash
            equity_rows[-1]["cash"] = cash
            equity_rows[-1]["positions"] = 0
            equity_rows[-1]["gross_exposure"] = 0.0

    return pd.DataFrame(trades), pd.DataFrame(equity_rows)


def _date_index(price_map: dict[str, pd.DataFrame]) -> dict[str, dict[pd.Timestamp, int]]:
    return {
        symbol: {pd.Timestamp(date): idx for idx, date in enumerate(frame["date"])}
        for symbol, frame in price_map.items()
    }


def _feature_history(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        symbol: frame.sort_values("date").reset_index(drop=True)
        for symbol, frame in features.groupby("symbol", sort=False)
    }


def _latest_features_before(
    history: dict[str, pd.DataFrame],
    symbol: str,
    date: pd.Timestamp,
) -> pd.Series | None:
    frame = history.get(symbol)
    if frame is None or frame.empty:
        return None
    rows = frame[frame["date"] < date]
    if rows.empty:
        return None
    return rows.iloc[-1]


def _entry_price(row: pd.Series) -> float:
    open_price = float(row["open"])
    return open_price if np.isfinite(open_price) and open_price > 0 else float(row["close"])


def _buy(date, row, candidate, budget, cash, cfg):
    open_price = float(row["open"])
    volume = float(row["volume"])
    if budget < cfg.min_position_value or open_price <= 0 or volume <= 0:
        return cash, None
    fill_price = open_price * (1.0 + cfg.slippage_rate)
    gross_shares = min(budget / fill_price, volume * cfg.max_participation_rate)
    shares = int(gross_shares // cfg.lot_size * cfg.lot_size)
    if shares <= 0:
        return cash, None
    gross = shares * fill_price
    total_cost = gross * (1.0 + cfg.commission_rate)
    if total_cost > cash:
        shares = int((cash / (fill_price * (1.0 + cfg.commission_rate))) // cfg.lot_size * cfg.lot_size)
        gross = shares * fill_price
        total_cost = gross * (1.0 + cfg.commission_rate)
    if shares <= 0 or gross < cfg.min_position_value:
        return cash, None
    position = Position(
        hypothesis=str(candidate["_hypothesis"]),
        symbol=str(candidate["symbol"]),
        entry_date=date,
        entry_price=fill_price,
        shares=shares,
        entry_value=gross,
        risk=dict(candidate.get("_risk") or {}),
    )
    return cash - total_cost, position


def _sell(date, row, position, cash, reason, cfg):
    fill_price = float(row["open"]) * (1.0 - cfg.slippage_rate)
    gross = position.shares * fill_price
    fees = gross * (cfg.commission_rate + cfg.sell_tax_rate)
    net = gross - fees
    pnl = net - position.entry_value
    trade = {
        "hypothesis": position.hypothesis,
        "symbol": position.symbol,
        "entry_date": position.entry_date,
        "exit_date": date,
        "entry_price": position.entry_price,
        "exit_price": fill_price,
        "shares": position.shares,
        "entry_value": position.entry_value,
        "exit_value": net,
        "pnl": pnl,
        "pnl_pct": 100.0 * pnl / position.entry_value if position.entry_value else 0.0,
        "exit_reason": reason,
        "holding_bars": position.holding_bars,
    }
    return cash + net, trade


def _exit_reason(row, prior_features, position, cfg) -> str | None:
    low = float(row["low"])
    high = float(row["high"])
    stop_loss = float(position.risk.get("stop_loss", cfg.stop_loss))
    take_profit = float(position.risk.get("take_profit", cfg.take_profit))
    max_holding = int(position.risk.get("max_holding_bars", cfg.max_holding_bars))
    if low <= position.entry_price * (1.0 - stop_loss):
        return "STOP_LOSS"
    if high >= position.entry_price * (1.0 + take_profit):
        return "TAKE_PROFIT"
    if prior_features is not None:
        reason = _feature_exit_reason(prior_features, position.risk)
        if reason:
            return reason
    if position.holding_bars >= max_holding:
        return "MAX_HOLDING"
    return None


def _feature_exit_reason(features: pd.Series, risk: dict) -> str | None:
    score_below = risk.get("exit_smart_money_score_below")
    if score_below is not None and _value(features, "smart_money_score") < float(score_below):
        return "SMART_MONEY_SCORE_EXIT"

    dist_gte = risk.get("exit_distribution_days_gte")
    if dist_gte is not None and _value(features, "distribution_days_10") >= float(dist_gte):
        return "DISTRIBUTION_EXIT"

    mkt_ds20_gte = risk.get("exit_mkt_DS20_gte")
    if mkt_ds20_gte is not None and _value(features, "mkt_DS20") >= float(mkt_ds20_gte):
        return "MARKET_DS_EXIT"

    chdm20_below = risk.get("exit_mkt_CHDM20_below")
    if chdm20_below is not None and _value(features, "mkt_CHDM20") < float(chdm20_below):
        return "MARKET_CHDM_EXIT"

    regime_score_below = risk.get("exit_mkt_regime_score_below")
    if regime_score_below is not None and _value(features, "mkt_regime_score") < float(regime_score_below):
        return "MARKET_REGIME_EXIT"

    ma50_break = risk.get("exit_close_below_ma50")
    if ma50_break and _value(features, "close") < _value(features, "ma50"):
        return "MA50_EXIT"

    return None


def _value(row: pd.Series, column: str) -> float:
    value = row.get(column, np.nan)
    return float(value) if pd.notna(value) else np.nan


def _mark_to_market(positions, price_map, date) -> float:
    total = 0.0
    for symbol, position in positions.items():
        frame = price_map[symbol]
        rows = frame[frame["date"] <= date]
        close = float(rows["close"].iloc[-1]) if not rows.empty else position.entry_price
        total += position.shares * close
    return total


def _target_exposure(candidates: pd.DataFrame) -> float:
    if candidates.empty:
        return 0.0
    row = candidates.iloc[0]
    risk = row.get("_risk") if isinstance(row.get("_risk"), dict) else {}
    regime_exposure = row.get("mkt_regime_exposure")
    if risk.get("use_regime_exposure") and pd.notna(regime_exposure):
        return float(regime_exposure)
    if row.get("mkt_CHDM50", 0) > 60 and row.get("mkt_DS50", 1) < 0.35:
        return 1.0
    if row.get("mkt_CHDM20", 0) > 45 and row.get("mkt_DS20", 1) < 0.50:
        return 0.7
    return 0.3
