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
    entry_atr: float = np.nan
    highest_price: float = np.nan
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
    return _run_portfolio(features, price_map, hypotheses, start, end, config=config)


def run_regime_switching_portfolio(
    features: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    hypotheses: list[Hypothesis],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    config: PortfolioConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one portfolio that switches active strategies by market regime."""
    hypothesis_map = {item.name: item for item in hypotheses}

    def selector(date: pd.Timestamp, day: pd.DataFrame) -> tuple[list[Hypothesis], str]:
        selected_names, regime_label = _regime_strategy_names(day)
        selected = [hypothesis_map[name] for name in selected_names if name in hypothesis_map]
        return selected, regime_label

    return _run_portfolio(features, price_map, hypotheses, start, end, config=config, strategy_selector=selector)


def run_symbol_strategy_router_portfolio(
    features: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    hypotheses: list[Hypothesis],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    config: PortfolioConfig | None = None,
    train_months: int = 12,
    test_months: int = 3,
    min_train_trades: int = 4,
    min_profit_factor: float = 1.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Walk-forward router that assigns one strategy per symbol.

    Each fold trains on the prior window, chooses the best strategy for each
    symbol, then trades the next test window with those symbol-strategy pairs.
    """
    cfg = config or PortfolioConfig()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    folds = _walk_forward_folds(start_ts, end_ts, train_months, test_months)
    selected_rows: list[dict] = []
    route: dict[pd.Timestamp, dict[str, str]] = {}

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds, start=1):
        train_features = features[(features["date"] >= train_start) & (features["date"] <= train_end)]
        selected = _select_symbol_strategies(
            train_features,
            price_map,
            hypotheses,
            train_start,
            train_end,
            cfg,
            min_train_trades,
            min_profit_factor,
        )
        route[test_start] = dict(zip(selected["symbol"], selected["selected_hypothesis"])) if not selected.empty else {}
        if not selected.empty:
            selected = selected.copy()
            selected["fold"] = fold_idx
            selected["train_start"] = train_start
            selected["train_end"] = train_end
            selected["test_start"] = test_start
            selected["test_end"] = test_end
            selected_rows.extend(selected.to_dict("records"))

    def signal_filter(hypothesis: Hypothesis, signal: dict, date: pd.Timestamp) -> bool:
        fold_start = _active_fold_start(route, date)
        if fold_start is None:
            return False
        symbol_route = route.get(fold_start, {})
        return symbol_route.get(str(signal.get("symbol"))) == hypothesis.name

    test_start = folds[0][2] if folds else start_ts
    trades, equity = _run_portfolio(
        features,
        price_map,
        hypotheses,
        test_start,
        end_ts,
        config=cfg,
        signal_filter=signal_filter,
    )
    return trades, equity, pd.DataFrame(selected_rows)


def run_symbol_strategy_router_v2_portfolio(
    features: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    hypotheses: list[Hypothesis],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    config: PortfolioConfig | None = None,
    train_months: int = 12,
    test_months: int = 3,
    validation_months: int = 3,
    min_train_trades: int = 4,
    min_validation_trades: int = 1,
    min_profit_factor: float = 1.20,
    min_avg_trade_pct: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Conservative symbol-strategy router with recent validation scoring."""
    cfg = config or PortfolioConfig()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    folds = _walk_forward_folds(start_ts, end_ts, train_months, test_months)
    selected_rows: list[dict] = []
    route: dict[pd.Timestamp, dict[str, str]] = {}

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds, start=1):
        validation_start = max(train_start, train_end - pd.DateOffset(months=validation_months) + pd.Timedelta(days=1))
        selected = _select_symbol_strategies_v2(
            features,
            price_map,
            hypotheses,
            train_start,
            train_end,
            validation_start,
            cfg,
            min_train_trades,
            min_validation_trades,
            min_profit_factor,
            min_avg_trade_pct,
        )
        route[test_start] = dict(zip(selected["symbol"], selected["selected_hypothesis"])) if not selected.empty else {}
        if not selected.empty:
            selected = selected.copy()
            selected["fold"] = fold_idx
            selected["train_start"] = train_start
            selected["train_end"] = train_end
            selected["validation_start"] = validation_start
            selected["test_start"] = test_start
            selected["test_end"] = test_end
            selected_rows.extend(selected.to_dict("records"))

    def signal_filter(hypothesis: Hypothesis, signal: dict, date: pd.Timestamp) -> bool:
        fold_start = _active_fold_start(route, date)
        if fold_start is None:
            return False
        return route.get(fold_start, {}).get(str(signal.get("symbol"))) == hypothesis.name

    test_start = folds[0][2] if folds else start_ts
    trades, equity = _run_portfolio(
        features,
        price_map,
        hypotheses,
        test_start,
        end_ts,
        config=cfg,
        signal_filter=signal_filter,
    )
    return trades, equity, pd.DataFrame(selected_rows)


def _run_portfolio(
    features: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    hypotheses: list[Hypothesis],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    config: PortfolioConfig | None = None,
    strategy_selector=None,
    signal_filter=None,
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
            position.highest_price = max(position.highest_price, float(row["high"]))
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
            candidates = candidates.drop_duplicates("symbol", keep="first")
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
        regime_label = ""
        active_hypotheses = hypotheses
        if strategy_selector is not None:
            active_hypotheses, regime_label = strategy_selector(date, day)
        for hypothesis in active_hypotheses:
            if not _can_emit_signal(hypothesis, date, calendar):
                continue
            signals = rank_candidates(day[evaluate_filters(day, hypothesis)], hypothesis)
            for signal in signals.head(cfg.top_n).to_dict("records"):
                symbol = signal["symbol"]
                if signal_filter is not None and not signal_filter(hypothesis, signal, date):
                    continue
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
                "regime": regime_label,
                "active_strategies": ",".join(item.name for item in active_hypotheses),
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


def _walk_forward_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int,
    test_months: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds = []
    train_start = start.normalize()
    while True:
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        if test_start > end:
            break
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), end)
        folds.append((train_start, train_end, test_start, test_end))
        train_start = train_start + pd.DateOffset(months=test_months)
    return folds


def _select_symbol_strategies(
    features: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    hypotheses: list[Hypothesis],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: PortfolioConfig,
    min_trades: int,
    min_profit_factor: float,
) -> pd.DataFrame:
    rows = []
    for hypothesis in hypotheses:
        trades, _ = run_portfolio(features, price_map, [hypothesis], start, end, config=cfg)
        if trades.empty:
            continue
        for symbol, group in trades.groupby("symbol", sort=False):
            pnl = group["pnl_pct"].astype(float)
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.inf
            rows.append(
                {
                    "symbol": symbol,
                    "selected_hypothesis": hypothesis.name,
                    "train_trades": int(len(group)),
                    "train_win_rate": float((pnl > 0).mean()),
                    "train_avg_trade_pct": float(pnl.mean()),
                    "train_total_pnl_pct": float(pnl.sum()),
                    "train_profit_factor": float(profit_factor),
                }
            )
    if not rows:
        return pd.DataFrame()
    candidates = pd.DataFrame(rows)
    candidates = candidates[
        (candidates["train_trades"] >= min_trades)
        & (candidates["train_profit_factor"] >= min_profit_factor)
    ].copy()
    if candidates.empty:
        return candidates
    candidates["router_score"] = (
        0.45 * candidates["train_profit_factor"].clip(0, 5)
        + 0.35 * candidates["train_avg_trade_pct"].clip(-10, 20) / 5
        + 0.20 * candidates["train_win_rate"] * 2
    )
    return (
        candidates.sort_values(["symbol", "router_score", "train_trades"], ascending=[True, False, False])
        .drop_duplicates("symbol", keep="first")
        .reset_index(drop=True)
    )


def _select_symbol_strategies_v2(
    features: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    hypotheses: list[Hypothesis],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    validation_start: pd.Timestamp,
    cfg: PortfolioConfig,
    min_train_trades: int,
    min_validation_trades: int,
    min_profit_factor: float,
    min_avg_trade_pct: float,
) -> pd.DataFrame:
    rows = []
    for hypothesis in hypotheses:
        trades, _ = run_portfolio(features, price_map, [hypothesis], train_start, train_end, config=cfg)
        if trades.empty:
            continue
        for symbol, group in trades.groupby("symbol", sort=False):
            train_stats = _group_trade_stats(group)
            validation = group[pd.to_datetime(group["entry_date"]) >= validation_start]
            validation_stats = _group_trade_stats(validation)
            rows.append(
                {
                    "symbol": symbol,
                    "selected_hypothesis": hypothesis.name,
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"validation_{k}": v for k, v in validation_stats.items()},
                }
            )
    if not rows:
        return pd.DataFrame()
    candidates = pd.DataFrame(rows)
    candidates = candidates[
        (candidates["train_trades"] >= min_train_trades)
        & (candidates["validation_trades"] >= min_validation_trades)
        & (candidates["train_profit_factor"] >= min_profit_factor)
        & (candidates["train_avg_trade_pct"] >= min_avg_trade_pct)
        & (candidates["validation_avg_trade_pct"] > 0)
        & (candidates["train_worst_trade_pct"] > -18.0)
    ].copy()
    if candidates.empty:
        return candidates

    trade_count_score = (candidates["train_trades"].clip(0, 10) / 10.0)
    pf_score = candidates["train_profit_factor"].replace(np.inf, 5.0).clip(0, 5) / 5.0
    avg_score = candidates["train_avg_trade_pct"].clip(-5, 15) / 15.0
    val_score = candidates["validation_avg_trade_pct"].clip(-5, 15) / 15.0
    win_score = candidates["train_win_rate"].clip(0, 1)
    drawdown_penalty = (candidates["train_worst_trade_pct"].abs().clip(0, 20) / 20.0)
    instability_penalty = (candidates["train_avg_trade_pct"] - candidates["validation_avg_trade_pct"]).clip(lower=0) / 15.0

    candidates["router_score"] = (
        0.24 * pf_score
        + 0.22 * avg_score
        + 0.24 * val_score
        + 0.16 * win_score
        + 0.14 * trade_count_score
        - 0.10 * drawdown_penalty
        - 0.10 * instability_penalty
    )
    return (
        candidates.sort_values(["symbol", "router_score", "train_trades"], ascending=[True, False, False])
        .drop_duplicates("symbol", keep="first")
        .reset_index(drop=True)
    )


def _group_trade_stats(group: pd.DataFrame) -> dict:
    if group.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_trade_pct": 0.0,
            "median_trade_pct": 0.0,
            "profit_factor": 0.0,
            "worst_trade_pct": 0.0,
        }
    pnl = group["pnl_pct"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else np.inf
    return {
        "trades": int(len(group)),
        "win_rate": float((pnl > 0).mean()),
        "avg_trade_pct": float(pnl.mean()),
        "median_trade_pct": float(pnl.median()),
        "profit_factor": float(profit_factor),
        "worst_trade_pct": float(pnl.min()),
    }


def _active_fold_start(route: dict[pd.Timestamp, dict[str, str]], date: pd.Timestamp) -> pd.Timestamp | None:
    starts = [fold_start for fold_start in route if fold_start <= date]
    return max(starts) if starts else None


def _regime_strategy_names(day: pd.DataFrame) -> tuple[list[str], str]:
    if day.empty:
        return [], "NO_DATA"
    row = day.iloc[0]
    state = str(row.get("mkt_regime_state", "NEUTRAL"))
    score = _safe_float(row.get("mkt_regime_score", np.nan))
    chdm_delta = _safe_float(row.get("mkt_CHDM20_delta", 0.0))
    ds_delta = _safe_float(row.get("mkt_DS20_delta", 0.0))
    ds20 = _safe_float(row.get("mkt_DS20", np.nan))

    early_recovery = (
        np.isfinite(score)
        and score >= 45
        and chdm_delta >= 5
        and ds_delta <= 0.05
        and (not np.isfinite(ds20) or ds20 <= 0.55)
    )
    if state == "RISK_ON":
        if early_recovery:
            return ["breakout_after_accumulation_v3", "leader_pullback_market_healthy_v2"], "RISK_ON_RECOVERY"
        return ["leader_pullback_market_healthy_v2", "breakout_after_accumulation_v3"], "RISK_ON"
    if early_recovery:
        return ["breakout_after_accumulation_v3", "accumulation_breakout_smt_v2"], "EARLY_RECOVERY"
    if state == "NEUTRAL":
        return ["composite_edge_score_v2", "mean_reversion_uptrend_ma50_v1"], "NEUTRAL"
    return [], "RISK_OFF"


def _can_emit_signal(hypothesis: Hypothesis, date: pd.Timestamp, calendar: list) -> bool:
    every_n_bars = hypothesis.risk.get("entry_every_n_bars")
    if every_n_bars is None:
        return True
    every_n_bars = max(1, int(every_n_bars))
    try:
        idx = calendar.index(np.datetime64(date))
    except ValueError:
        idx = next((i for i, item in enumerate(calendar) if pd.Timestamp(item) == date), 0)
    return idx % every_n_bars == 0


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
        entry_atr=_safe_float(candidate.get("atr14", np.nan)),
        highest_price=fill_price,
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
    atr_reason = _atr_exit_reason(low, position)
    if atr_reason:
        return atr_reason
    if high >= position.entry_price * (1.0 + take_profit):
        return "TAKE_PROFIT"
    if prior_features is not None:
        reason = _feature_exit_reason(prior_features, position.risk)
        if reason:
            return reason
    if position.holding_bars >= max_holding:
        return "MAX_HOLDING"
    return None


def _atr_exit_reason(low: float, position: Position) -> str | None:
    atr = _safe_float(position.entry_atr)
    if not np.isfinite(atr) or atr <= 0:
        return None

    initial_mult = position.risk.get("initial_atr_stop_mult")
    if initial_mult is not None and low <= position.entry_price - float(initial_mult) * atr:
        return "ATR_STOP"

    trailing_mult = position.risk.get("trailing_atr_mult")
    if trailing_mult is None:
        return None

    activation = float(position.risk.get("trailing_profit_activation", 0.0))
    if position.highest_price < position.entry_price * (1.0 + activation):
        return None
    if low <= position.highest_price - float(trailing_mult) * atr:
        return "TRAILING_ATR_STOP"
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


def _safe_float(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


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
