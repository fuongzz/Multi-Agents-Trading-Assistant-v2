"""Production-like backtest for Flow Money V2 rotation.

Execution model:
- compute ranking after close T;
- execute target portfolio at open T+1;
- use cash/shares, lot size, commission, sell tax, and slippage;
- mark positions to market at daily close;
- rebalance every N trading days.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from multiagents_trading_assistant.edge_lab.features import build_feature_table


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "flow_v2_rotation_production_like"


@dataclass(frozen=True)
class RotationConfig:
    universe: str
    start: str
    end: str
    positions: int
    rebalance_days: int
    initial_capital: float = 1_000_000_000.0
    commission_rate: float = 0.0010
    sell_tax_rate: float = 0.0010
    slippage_rate: float = 0.0005
    lot_size: int = 100
    price_unit_multiplier: float = 1000.0
    min_trade_value: float = 2_000_000.0
    max_symbol_weight: float | None = None
    min_hold_rebalance: int = 0
    selection_mode: str = "base"
    hold_buffer: int = 0
    min_score_improvement: float = 0.0
    market_gate: str = "base"
    pool_filter: str = "base"
    score_mode: str = "balanced"
    max_entry_ret_20d: float | None = None
    max_entry_distance_ma20: float | None = None
    early_exit_mode: str = "none"
    allocation_mode: str = "equal_weight"
    diversification_mode: str = "none"
    neutral_gross_exposure: float | None = None
    liquidate_at_end: bool = True


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def _sharpe(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty or float(daily_returns.std()) == 0.0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(252.0))


def _prepare_features(universe: str, start: str, end: str) -> pd.DataFrame:
    symbols = combo_harness.resolve_symbols(universe)
    features, _ = build_feature_table(symbols, start, end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    features["value_rank"] = features.groupby("date", sort=False)["value_ratio_20"].rank(pct=True)
    features["dist_rank"] = features.groupby("date", sort=False)["distribution_pressure_score"].rank(pct=True)
    features["flow_v2_rotation_score_balanced"] = (
        0.28 * features["sector_cycle_score"].fillna(50.0)
        + 0.24 * features["flow_sponsorship_score"].fillna(50.0)
        + 0.18 * features["flow_absorption_score"].fillna(50.0)
        + 0.15 * (features["rs_percentile_20"].fillna(0.5) * 100.0)
        + 0.15 * (features["value_rank"].fillna(0.5) * 100.0)
        - 0.15 * (features["dist_rank"].fillna(0.5) * 100.0)
    )
    features["flow_v2_rotation_score_flow_heavy"] = (
        0.20 * features["sector_cycle_score"].fillna(50.0)
        + 0.30 * features["flow_sponsorship_score"].fillna(50.0)
        + 0.22 * features["flow_absorption_score"].fillna(50.0)
        + 0.13 * (features["rs_percentile_20"].fillna(0.5) * 100.0)
        + 0.10 * (features["value_rank"].fillna(0.5) * 100.0)
        - 0.17 * (features["dist_rank"].fillna(0.5) * 100.0)
    )
    features["flow_v2_rotation_score_momentum_heavy"] = (
        0.22 * features["sector_cycle_score"].fillna(50.0)
        + 0.20 * features["flow_sponsorship_score"].fillna(50.0)
        + 0.15 * features["flow_absorption_score"].fillna(50.0)
        + 0.25 * (features["rs_percentile_20"].fillna(0.5) * 100.0)
        + 0.12 * (features["value_rank"].fillna(0.5) * 100.0)
        - 0.16 * (features["dist_rank"].fillna(0.5) * 100.0)
    )
    features["flow_v2_rotation_score_sector_heavy"] = (
        0.35 * features["sector_cycle_score"].fillna(50.0)
        + 0.20 * features["flow_sponsorship_score"].fillna(50.0)
        + 0.15 * features["flow_absorption_score"].fillna(50.0)
        + 0.15 * (features["rs_percentile_20"].fillna(0.5) * 100.0)
        + 0.10 * (features["value_rank"].fillna(0.5) * 100.0)
        - 0.15 * (features["dist_rank"].fillna(0.5) * 100.0)
    )
    features["flow_v2_rotation_score"] = features["flow_v2_rotation_score_balanced"]
    return features


def _score_column(score_mode: str) -> str:
    allowed = {"balanced", "flow_heavy", "momentum_heavy", "sector_heavy"}
    if score_mode not in allowed:
        raise ValueError(f"Unknown score_mode={score_mode!r}; expected one of {sorted(allowed)}")
    return f"flow_v2_rotation_score_{score_mode}"


def _eligible_pool(
    day: pd.DataFrame,
    market_gate: str,
    pool_filter: str,
    max_entry_ret_20d: float | None = None,
    max_entry_distance_ma20: float | None = None,
    current_symbols: set[str] | None = None,
) -> pd.DataFrame:
    if market_gate == "none":
        market_mask = pd.Series(True, index=day.index)
    elif market_gate == "strict":
        market_mask = (
            (day["mkt_regime_state"] == "RISK_ON")
            & (day["mkt_regime_score"] >= 60)
            & (day["mkt_CHDM20"] > 50)
            & (day["mkt_DS20"] <= 0.45)
            & (day["mkt_DS50"] <= 0.50)
        )
    elif market_gate == "risk_on_or_strong_neutral":
        market_mask = (
            (day["mkt_regime_state"].isin(["RISK_ON", "NEUTRAL"]))
            & (day["mkt_regime_score"] >= 55)
            & (day["mkt_CHDM20"] > 48)
            & (day["mkt_DS20"] <= 0.50)
        )
    else:
        market_mask = (
            (day["mkt_regime_state"] != "RISK_OFF")
            & (day["mkt_regime_score"] >= 50)
            & (day["mkt_DS20"] <= 0.60)
        )

    if pool_filter == "clean_flow":
        stock_mask = (
            (day["above_ma50"] == True)
            & (day["distribution_pressure_score"] <= 55)
            & (day["flow_sponsorship_score"] >= 50)
            & (day["flow_absorption_score"] >= 45)
            & (day["value_ratio_20"] >= 0.90)
            & (day["rs_percentile_20"] >= 0.50)
        )
    elif pool_filter == "high_rs":
        stock_mask = (
            (day["above_ma50"] == True)
            & (day["distribution_pressure_score"] <= 65)
            & (day["flow_sponsorship_score"] >= 45)
            & (day["value_ratio_20"] >= 0.80)
            & (day["rs_percentile_20"] >= 0.60)
        )
    elif pool_filter == "loose_flow":
        stock_mask = (
            (day["above_ma50"] == True)
            & (day["distribution_pressure_score"] <= 70)
            & (day["flow_sponsorship_score"] >= 42)
            & (day["value_ratio_20"] >= 0.70)
            & (day["rs_percentile_20"] >= 0.40)
        )
    else:
        stock_mask = (
            (day["above_ma50"] == True)
            & (day["distribution_pressure_score"] <= 65)
            & (day["flow_sponsorship_score"] >= 45)
            & (day["value_ratio_20"] >= 0.80)
            & (day["rs_percentile_20"] >= 0.45)
        )

    held_mask = day["symbol"].isin(current_symbols or set())
    if max_entry_ret_20d is not None:
        stock_mask = stock_mask & ((day["ret_20d_local"] <= max_entry_ret_20d) | held_mask)
    if max_entry_distance_ma20 is not None:
        stock_mask = stock_mask & ((day["distance_ma20"] <= max_entry_distance_ma20) | held_mask)

    pool = day[market_mask & stock_mask].copy()
    return pool


def _select_symbols(
    day: pd.DataFrame,
    positions: int,
    market_gate: str = "base",
    pool_filter: str = "base",
    score_mode: str = "balanced",
    max_entry_ret_20d: float | None = None,
    max_entry_distance_ma20: float | None = None,
    current_symbols: set[str] | None = None,
    diversification_mode: str = "none",
) -> list[str]:
    pool = _eligible_pool(
        day,
        market_gate,
        pool_filter,
        max_entry_ret_20d,
        max_entry_distance_ma20,
        current_symbols,
    )
    if pool.empty:
        return []
    return _take_ranked_symbols(pool, positions, score_mode, diversification_mode)


def _take_ranked_symbols(
    pool: pd.DataFrame,
    positions: int,
    score_mode: str,
    diversification_mode: str = "none",
) -> list[str]:
    score_col = _score_column(score_mode)
    ranked = pool.sort_values(score_col, ascending=False)
    if diversification_mode == "none":
        return list(ranked.head(positions)["symbol"])
    if diversification_mode != "distinct_industry":
        raise ValueError(f"Unknown diversification_mode={diversification_mode!r}")

    selected: list[str] = []
    used_industries: set[str] = set()
    for _, row in ranked.iterrows():
        industry = str(row.get("industry", "")).strip()
        known_industry = industry.upper() not in {"", "UNKNOWN", "NAN", "NONE"}
        if known_industry and industry in used_industries:
            continue
        selected.append(str(row["symbol"]))
        if known_industry:
            used_industries.add(industry)
        if len(selected) >= positions:
            break
    return selected


def _select_symbols_turnover_aware(
    day: pd.DataFrame,
    *,
    positions: int,
    current_symbols: set[str],
    market_gate: str,
    pool_filter: str,
    score_mode: str,
    hold_buffer: int,
    min_score_improvement: float,
    max_entry_ret_20d: float | None = None,
    max_entry_distance_ma20: float | None = None,
    diversification_mode: str = "none",
) -> list[str]:
    pool = _eligible_pool(
        day,
        market_gate,
        pool_filter,
        max_entry_ret_20d,
        max_entry_distance_ma20,
        current_symbols,
    )
    if pool.empty:
        return []
    score_col = _score_column(score_mode)
    ranked = pool.sort_values(score_col, ascending=False).reset_index(drop=True)
    if diversification_mode == "distinct_industry":
        eligible_symbols = set(_take_ranked_symbols(pool, len(pool), score_mode, diversification_mode))
        ranked = ranked.loc[ranked["symbol"].isin(eligible_symbols)].reset_index(drop=True)
    elif diversification_mode != "none":
        raise ValueError(f"Unknown diversification_mode={diversification_mode!r}")
    score_by_symbol = ranked.set_index("symbol")[score_col].to_dict()
    top_limit = max(positions, positions + hold_buffer)
    top_symbols = list(ranked.head(top_limit)["symbol"])
    selected: list[str] = []

    for symbol in top_symbols:
        if symbol in current_symbols and len(selected) < positions:
            selected.append(symbol)

    for symbol in list(current_symbols):
        if len(selected) >= positions:
            break
        if symbol in score_by_symbol and symbol not in selected:
            selected.append(symbol)

    for symbol in list(ranked["symbol"]):
        if len(selected) >= positions:
            break
        if symbol in selected:
            continue
        if len(selected) < positions:
            if not current_symbols or min_score_improvement <= 0:
                selected.append(symbol)
                continue
            weakest_current_score = min(
                [float(score_by_symbol.get(item, -1e9)) for item in selected if item in current_symbols] or [-1e9]
            )
            candidate_score = float(score_by_symbol.get(symbol, -1e9))
            if candidate_score >= weakest_current_score + min_score_improvement:
                selected.append(symbol)
            elif len([item for item in selected if item in current_symbols]) < positions:
                selected.append(symbol)

    return selected[:positions]


def _early_exit_actions(
    day: pd.DataFrame,
    current_symbols: set[str],
    reduced_symbols: set[str],
    cfg: RotationConfig,
) -> dict[str, tuple[float, str]]:
    if cfg.early_exit_mode == "none" or not current_symbols:
        return {}
    if cfg.early_exit_mode not in {"flow_momentum_break", "flow_momentum_tiered"}:
        raise ValueError(f"Unknown early_exit_mode={cfg.early_exit_mode!r}")

    held = day.loc[day["symbol"].isin(current_symbols)].copy()
    if held.empty:
        return {}
    momentum_break = held["above_ma50"].fillna(False) != True
    weak_flow = (held["flow_sponsorship_score"] < 40) & (held["rs_percentile_20"] < 0.50)
    distribution_break = (held["distribution_pressure_score"] > 70) & (held["rs_percentile_20"] < 0.50)
    hard_symbols = set(held.loc[momentum_break | weak_flow | distribution_break, "symbol"])
    if cfg.early_exit_mode == "flow_momentum_break":
        return {str(symbol): (1.0, "EARLY_EXIT_FLOW_MOMENTUM_BREAK") for symbol in hard_symbols}

    soft_break = (
        ((held["flow_sponsorship_score"] < 45) & (held["rs_percentile_20"] < 0.60))
        | ((held["distribution_pressure_score"] > 65) & (held["rs_percentile_20"] < 0.60))
    )
    soft_symbols = set(held.loc[soft_break, "symbol"]) - hard_symbols - reduced_symbols
    actions = {str(symbol): (1.0, "EARLY_EXIT_FLOW_MOMENTUM_BREAK") for symbol in hard_symbols}
    actions.update({str(symbol): (0.5, "PARTIAL_EXIT_FLOW_WEAKNESS") for symbol in soft_symbols})
    return actions


def _early_exit_symbols(day: pd.DataFrame, current_symbols: set[str], cfg: RotationConfig) -> list[str]:
    return list(_early_exit_actions(day, current_symbols, set(), cfg))


def _target_weights(day: pd.DataFrame, target_symbols: list[str], cfg: RotationConfig) -> dict[str, float]:
    if not target_symbols:
        return {}
    exposure = 1.0
    if cfg.neutral_gross_exposure is not None and str(day["mkt_regime_state"].iloc[0]) == "NEUTRAL":
        exposure = min(1.0, max(0.0, cfg.neutral_gross_exposure))
    selected = day.loc[day["symbol"].isin(target_symbols)].set_index("symbol")
    if cfg.allocation_mode == "equal_weight":
        raw = pd.Series(1.0, index=target_symbols)
    elif cfg.allocation_mode == "inverse_atr":
        relative_atr = selected["atr14"] / selected["close"].replace(0, np.nan)
        raw = 1.0 / relative_atr.reindex(target_symbols).replace(0, np.nan)
        raw = raw.replace([np.inf, -np.inf], np.nan)
        if raw.isna().any() or float(raw.sum()) <= 0:
            raw = pd.Series(1.0, index=target_symbols)
    else:
        raise ValueError(f"Unknown allocation_mode={cfg.allocation_mode!r}")
    weights = raw / float(raw.sum()) * exposure
    if cfg.max_symbol_weight is not None:
        weights = weights.clip(upper=cfg.max_symbol_weight)
    return {str(symbol): float(weight) for symbol, weight in weights.items()}


def _portfolio_value(cash: float, positions: dict[str, int], close_by_symbol: pd.Series) -> float:
    value = cash
    for symbol, shares in positions.items():
        close = close_by_symbol.get(symbol)
        if pd.notna(close):
            value += float(shares) * float(close)
    return float(value)


def _sell_to_target(
    *,
    date: pd.Timestamp,
    symbol: str,
    shares_to_sell: int,
    open_price: float,
    positions: dict[str, int],
    cash: float,
    cfg: RotationConfig,
    trades: list[dict[str, Any]],
    reason: str,
    signal_date: str | None = None,
) -> float:
    shares_to_sell = min(shares_to_sell, positions.get(symbol, 0))
    shares_to_sell = (shares_to_sell // cfg.lot_size) * cfg.lot_size
    if shares_to_sell <= 0:
        return cash
    gross_price = open_price * (1.0 - cfg.slippage_rate)
    gross_value = shares_to_sell * gross_price
    if gross_value < cfg.min_trade_value:
        return cash
    fees = gross_value * (cfg.commission_rate + cfg.sell_tax_rate)
    cash += gross_value - fees
    positions[symbol] -= shares_to_sell
    if positions[symbol] <= 0:
        del positions[symbol]
    trades.append(
        {
            "date": date.date().isoformat(),
            "signal_date": signal_date,
            "symbol": symbol,
            "side": "SELL",
            "shares": shares_to_sell,
            "price": gross_price,
            "gross_value": gross_value,
            "fees": fees,
            "reason": reason,
        }
    )
    return cash


def _buy_to_target(
    *,
    date: pd.Timestamp,
    symbol: str,
    target_value: float,
    open_price: float,
    positions: dict[str, int],
    cash: float,
    cfg: RotationConfig,
    trades: list[dict[str, Any]],
    signal_date: str | None = None,
) -> float:
    current_shares = positions.get(symbol, 0)
    current_value = current_shares * open_price
    buy_budget = max(0.0, target_value - current_value)
    if buy_budget < cfg.min_trade_value:
        return cash
    buy_price = open_price * (1.0 + cfg.slippage_rate)
    shares = int(buy_budget / (buy_price * (1.0 + cfg.commission_rate)))
    shares = (shares // cfg.lot_size) * cfg.lot_size
    if shares <= 0:
        return cash
    gross_value = shares * buy_price
    fees = gross_value * cfg.commission_rate
    total_cost = gross_value + fees
    if total_cost > cash:
        affordable = int(cash / (buy_price * (1.0 + cfg.commission_rate)))
        shares = (affordable // cfg.lot_size) * cfg.lot_size
        if shares <= 0:
            return cash
        gross_value = shares * buy_price
        fees = gross_value * cfg.commission_rate
        total_cost = gross_value + fees
    cash -= total_cost
    positions[symbol] = positions.get(symbol, 0) + shares
    trades.append(
        {
            "date": date.date().isoformat(),
            "signal_date": signal_date,
            "symbol": symbol,
            "side": "BUY",
            "shares": shares,
            "price": buy_price,
            "gross_value": gross_value,
            "fees": fees,
            "reason": "TARGET_REBALANCE",
        }
    )
    return cash


def _execute_rebalance(
    *,
    date: pd.Timestamp,
    target_symbols: list[str],
    open_by_symbol: pd.Series,
    close_by_symbol: pd.Series,
    cash: float,
    positions: dict[str, int],
    hold_counts: dict[str, int],
    cfg: RotationConfig,
    trades: list[dict[str, Any]],
    target_weights: dict[str, float] | None = None,
    signal_date: str | None = None,
) -> tuple[float, dict[str, float]]:
    # At the next-open execution point, same-day close is not known yet.
    # Use open prices for sizing the target portfolio to avoid look-ahead sizing.
    nav = _portfolio_value(cash, positions, open_by_symbol)
    if nav <= 0:
        return cash, {}

    if target_weights is None:
        max_weight = cfg.max_symbol_weight if cfg.max_symbol_weight is not None else 1.0 / max(1, cfg.positions)
        target_weight = min(max_weight, 1.0 / max(1, len(target_symbols))) if target_symbols else 0.0
        target_values = {symbol: nav * target_weight for symbol in target_symbols}
    else:
        target_values = {symbol: nav * float(target_weights.get(symbol, 0.0)) for symbol in target_symbols}

    for symbol in list(positions):
        open_price = open_by_symbol.get(symbol)
        if pd.isna(open_price):
            continue
        target_value = target_values.get(symbol, 0.0)
        current_value = positions[symbol] * float(open_price)
        if target_value <= 0 and hold_counts.get(symbol, 0) < cfg.min_hold_rebalance:
            target_value = current_value
        excess_value = current_value - target_value
        if excess_value > cfg.min_trade_value:
            # A symbol dropped from the target must be closed exactly. Recomputing
            # shares from floating-point value can leave an unintended board lot.
            shares_to_sell = (
                positions[symbol]
                if target_value <= 0
                else int(excess_value / float(open_price))
            )
            cash = _sell_to_target(
                date=date,
                symbol=symbol,
                shares_to_sell=shares_to_sell,
                open_price=float(open_price),
                positions=positions,
                cash=cash,
                cfg=cfg,
                trades=trades,
                reason="TARGET_REBALANCE" if symbol in target_symbols else "DROP_FROM_TARGET",
                signal_date=signal_date,
            )

    # A held symbol without an executable open price cannot be sold today.
    # Reserve its slot instead of opening additional names beyond the position cap.
    existing_targets = [symbol for symbol in target_symbols if symbol in positions]
    remaining_slots = max(0, cfg.positions - len(positions))
    admitted_new_targets = [symbol for symbol in target_symbols if symbol not in positions][:remaining_slots]
    executable_targets = set(existing_targets + admitted_new_targets)
    for symbol in target_symbols:
        if symbol not in executable_targets:
            continue
        open_price = open_by_symbol.get(symbol)
        if pd.isna(open_price):
            continue
        cash = _buy_to_target(
            date=date,
            symbol=symbol,
            target_value=target_values[symbol],
            open_price=float(open_price),
            positions=positions,
            cash=cash,
            cfg=cfg,
            trades=trades,
            signal_date=signal_date,
        )

    weights = {}
    nav_after = _portfolio_value(cash, positions, close_by_symbol)
    if nav_after > 0:
        for symbol, shares in positions.items():
            close = close_by_symbol.get(symbol)
            if pd.notna(close):
                weights[symbol] = float(shares) * float(close) / nav_after
    return cash, weights


def run_backtest(
    cfg: RotationConfig,
    features: pd.DataFrame | None = None,
    locked_targets: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if features is None:
        features = _prepare_features(cfg.universe, cfg.start, cfg.end)
    start_ts = pd.Timestamp(cfg.start).normalize()
    end_ts = pd.Timestamp(cfg.end).normalize()
    calendar = sorted(features.loc[(features["date"] >= start_ts) & (features["date"] <= end_ts), "date"].unique())
    by_date = {date: group for date, group in features.groupby("date", sort=True)}

    cash = cfg.initial_capital
    positions: dict[str, int] = {}
    hold_counts: dict[str, int] = {}
    pending_target: list[str] | None = None
    pending_target_weights: dict[str, float] | None = None
    pending_target_signal_date: str | None = None
    pending_early_exits: dict[str, tuple[float, str]] = {}
    pending_early_exit_signal_date: str | None = None
    reduced_symbols: set[str] = set()
    rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    rebalance_counter = 0
    last_close: dict[str, float] = {}
    locked_by_date: dict[str, list[str]] = {}
    if locked_targets is not None and not locked_targets.empty:
        locked_by_date = (
            locked_targets.sort_values(["signal_date", "rank"])
            .groupby("signal_date", sort=False)["symbol"]
            .apply(list)
            .to_dict()
        )

    for idx, date in enumerate(calendar):
        day = by_date[date]
        open_by_symbol = day.set_index("symbol")["open"] * cfg.price_unit_multiplier
        todays_close = day.set_index("symbol")["close"].dropna() * cfg.price_unit_multiplier
        last_close.update({str(symbol): float(close) for symbol, close in todays_close.items()})
        close_by_symbol = pd.Series(last_close)

        if pending_target is not None:
            cash, weights = _execute_rebalance(
                date=date,
                target_symbols=pending_target,
                open_by_symbol=open_by_symbol,
                close_by_symbol=close_by_symbol,
                cash=cash,
                positions=positions,
                hold_counts=hold_counts,
                cfg=cfg,
                trades=trades,
                target_weights=pending_target_weights,
                signal_date=pending_target_signal_date,
            )
            pending_target = None
            pending_target_weights = None
            pending_target_signal_date = None
            reduced_symbols.intersection_update(positions)
            reduced_symbols.clear()
            rebalance_counter += 1
        elif pending_early_exits:
            for symbol, (sell_fraction, reason) in pending_early_exits.items():
                open_price = open_by_symbol.get(symbol)
                if symbol not in positions or pd.isna(open_price):
                    continue
                cash = _sell_to_target(
                    date=date,
                    symbol=symbol,
                    shares_to_sell=int(positions[symbol] * sell_fraction),
                    open_price=float(open_price),
                    positions=positions,
                    cash=cash,
                    cfg=cfg,
                    trades=trades,
                    reason=reason,
                    signal_date=pending_early_exit_signal_date,
                )
                if sell_fraction < 1.0 and symbol in positions:
                    reduced_symbols.add(symbol)
                elif symbol not in positions:
                    reduced_symbols.discard(symbol)
            pending_early_exits = {}
            pending_early_exit_signal_date = None
            weights = {}
        else:
            weights = {}

        for symbol in list(hold_counts):
            if symbol in positions:
                hold_counts[symbol] += 1
            else:
                del hold_counts[symbol]
        for symbol in positions:
            hold_counts.setdefault(symbol, 1)

        nav = _portfolio_value(cash, positions, close_by_symbol)
        rows.append(
            {
                "date": date.date().isoformat(),
                "equity": nav,
                "cash": cash,
                "cash_weight": cash / nav if nav > 0 else 0.0,
                "positions": len(positions),
                "holdings": ",".join(sorted(positions)),
            }
        )

        if idx % cfg.rebalance_days == 0:
            signal_date = date.date().isoformat()
            if locked_targets is not None:
                target = locked_by_date.get(signal_date, [])
            elif cfg.selection_mode == "turnover_aware":
                target = _select_symbols_turnover_aware(
                    day,
                    positions=cfg.positions,
                    current_symbols=set(positions),
                    market_gate=cfg.market_gate,
                    pool_filter=cfg.pool_filter,
                    score_mode=cfg.score_mode,
                    hold_buffer=cfg.hold_buffer,
                    min_score_improvement=cfg.min_score_improvement,
                    max_entry_ret_20d=cfg.max_entry_ret_20d,
                    max_entry_distance_ma20=cfg.max_entry_distance_ma20,
                    diversification_mode=cfg.diversification_mode,
                )
            else:
                target = _select_symbols(
                    day,
                    cfg.positions,
                    cfg.market_gate,
                    cfg.pool_filter,
                    cfg.score_mode,
                    cfg.max_entry_ret_20d,
                    cfg.max_entry_distance_ma20,
                    set(positions),
                    cfg.diversification_mode,
                )
            pending_target = target
            pending_target_weights = _target_weights(day, target, cfg)
            pending_target_signal_date = signal_date
            score_col = _score_column(cfg.score_mode)
            for rank, symbol in enumerate(target, start=1):
                item = day.loc[day["symbol"] == symbol].iloc[0]
                target_rows.append(
                    {
                        "signal_date": date.date().isoformat(),
                        "execute_date": calendar[idx + 1].date().isoformat() if idx + 1 < len(calendar) else None,
                        "rank": rank,
                        "symbol": symbol,
                        "flow_v2_rotation_score": round(float(item[score_col]), 4),
                        "score_mode": cfg.score_mode,
                        "pool_filter": cfg.pool_filter,
                        "sector_cycle_score": round(float(item["sector_cycle_score"]), 4),
                        "flow_sponsorship_score": round(float(item["flow_sponsorship_score"]), 4),
                        "flow_absorption_score": round(float(item["flow_absorption_score"]), 4),
                        "distribution_pressure_score": round(float(item["distribution_pressure_score"]), 4),
                        "rs_percentile_20": round(float(item["rs_percentile_20"]), 4),
                        "value_ratio_20": round(float(item["value_ratio_20"]), 4),
                        "target_weight": round(float(pending_target_weights.get(symbol, 0.0)), 6),
                    }
                )
        elif idx + 1 < len(calendar):
            pending_early_exits = _early_exit_actions(day, set(positions), reduced_symbols, cfg)
            pending_early_exit_signal_date = date.date().isoformat() if pending_early_exits else None

    # Liquidate at final close for a clean ending NAV/trade log.
    if cfg.liquidate_at_end and calendar and positions:
        final_date = calendar[-1]
        final_close = by_date[final_date].set_index("symbol")["close"].dropna() * cfg.price_unit_multiplier
        last_close.update({str(symbol): float(close) for symbol, close in final_close.items()})
        close_by_symbol = pd.Series(last_close)
        for symbol in list(positions):
            close = close_by_symbol.get(symbol)
            if pd.isna(close):
                continue
            cash = _sell_to_target(
                date=final_date,
                symbol=symbol,
                shares_to_sell=positions[symbol],
                open_price=float(close),
                positions=positions,
                cash=cash,
                cfg=cfg,
                trades=trades,
                reason="FINAL_LIQUIDATION",
                signal_date=final_date.date().isoformat(),
            )
        if rows:
            rows[-1]["equity"] = cash
            rows[-1]["cash"] = cash
            rows[-1]["cash_weight"] = 1.0
            rows[-1]["positions"] = 0
            rows[-1]["holdings"] = ""

    equity_frame = pd.DataFrame(rows)
    equity_series = equity_frame.set_index(pd.to_datetime(equity_frame["date"]))["equity"]
    daily_returns = equity_series.pct_change()
    total_return = float(equity_series.iloc[-1] / cfg.initial_capital - 1.0) if not equity_series.empty else 0.0
    trade_frame = pd.DataFrame(trades)
    turnover = float(trade_frame["gross_value"].sum() / cfg.initial_capital) if not trade_frame.empty else 0.0
    summary = {
        "positions": cfg.positions,
        "rebalance_days": cfg.rebalance_days,
        "price_unit_multiplier": cfg.price_unit_multiplier,
        "max_entry_ret_20d": cfg.max_entry_ret_20d,
        "max_entry_distance_ma20": cfg.max_entry_distance_ma20,
        "early_exit_mode": cfg.early_exit_mode,
        "allocation_mode": cfg.allocation_mode,
        "diversification_mode": cfg.diversification_mode,
        "neutral_gross_exposure": cfg.neutral_gross_exposure,
        "total_return_pct": round(total_return * 100.0, 2),
        "sharpe_ratio": round(_sharpe(daily_returns), 3),
        "max_drawdown_pct": round(_max_drawdown(equity_series) * 100.0, 2),
        "ending_equity": round(float(equity_series.iloc[-1]), 2) if not equity_series.empty else cfg.initial_capital,
        "number_of_orders": int(len(trade_frame)),
        "buy_orders": int((trade_frame["side"] == "BUY").sum()) if not trade_frame.empty else 0,
        "sell_orders": int((trade_frame["side"] == "SELL").sum()) if not trade_frame.empty else 0,
        "gross_turnover_x": round(turnover, 3),
        "total_fees": round(float(trade_frame["fees"].sum()), 2) if not trade_frame.empty else 0.0,
        "rebalance_count": rebalance_counter,
    }
    return {
        "summary": summary,
        "equity": equity_frame,
        "targets": pd.DataFrame(target_rows),
        "trades": trade_frame,
        "open_positions": dict(positions),
        "ending_cash": cash,
    }


def run_grid(
    *,
    universe: str,
    start: str,
    end: str,
    label: str,
    positions_values: list[int],
    rebalance_values: list[int],
    selection_modes: list[str],
    market_gates: list[str],
    pool_filters: list[str],
    score_modes: list[str],
    hold_buffer_values: list[int],
    min_score_improvement_values: list[float],
    initial_capital: float,
    commission_rate: float,
    sell_tax_rate: float,
    slippage_rate: float,
    lot_size: int,
    price_unit_multiplier: float,
) -> Path:
    out_dir = OUT_ROOT / f"{label}_{universe}_{start}_{end}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    features = _prepare_features(universe, start, end)
    for selection_mode in selection_modes:
        for market_gate in market_gates:
            for pool_filter in pool_filters:
                for score_mode in score_modes:
                    for hold_buffer in hold_buffer_values:
                        for min_score_improvement in min_score_improvement_values:
                            for positions in positions_values:
                                for rebalance_days in rebalance_values:
                                    cfg = RotationConfig(
                                        universe=universe,
                                        start=start,
                                        end=end,
                                        positions=positions,
                                        rebalance_days=rebalance_days,
                                        initial_capital=initial_capital,
                                        commission_rate=commission_rate,
                                        sell_tax_rate=sell_tax_rate,
                                        slippage_rate=slippage_rate,
                                        lot_size=lot_size,
                                        price_unit_multiplier=price_unit_multiplier,
                                        selection_mode=selection_mode,
                                        market_gate=market_gate,
                                        pool_filter=pool_filter,
                                        score_mode=score_mode,
                                        hold_buffer=hold_buffer,
                                        min_score_improvement=min_score_improvement,
                                    )
                                    print(
                                        f"[flow-v2-prod] run {selection_mode}/{market_gate}/{pool_filter}/{score_mode}/"
                                        f"h{hold_buffer}/m{min_score_improvement:g}/top{positions}/"
                                        f"rebalance{rebalance_days}"
                                    )
                                    result = run_backtest(cfg, features=features)
                                    summary = result["summary"]
                                    summary["selection_mode"] = selection_mode
                                    summary["market_gate"] = market_gate
                                    summary["pool_filter"] = pool_filter
                                    summary["score_mode"] = score_mode
                                    summary["hold_buffer"] = hold_buffer
                                    summary["min_score_improvement"] = min_score_improvement
                                    rows.append(summary)
                                    stem = (
                                        f"{selection_mode}_{market_gate}_{pool_filter}_{score_mode}_"
                                        f"h{hold_buffer}_m{min_score_improvement:g}_"
                                        f"top{positions}_rebalance{rebalance_days}"
                                    )
                                    result["equity"].to_csv(out_dir / f"{stem}_equity.csv", index=False)
                                    result["targets"].to_csv(out_dir / f"{stem}_targets.csv", index=False)
                                    result["trades"].to_csv(out_dir / f"{stem}_orders.csv", index=False)
                                    pd.DataFrame(rows).sort_values("total_return_pct", ascending=False).to_csv(
                                        out_dir / "summary.csv",
                                        index=False,
                                    )
                                    print(
                                        f"  -> ret={summary['total_return_pct']:+.2f}% "
                                        f"sharpe={summary['sharpe_ratio']:.3f} "
                                        f"mdd={summary['max_drawdown_pct']:.2f}% "
                                        f"orders={summary['number_of_orders']} turnover={summary['gross_turnover_x']:.2f}x"
                                    )
    pd.DataFrame(rows).sort_values(["total_return_pct", "sharpe_ratio"], ascending=[False, False]).to_csv(
        out_dir / "summary.csv",
        index=False,
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Production-like Flow Money V2 rotation backtest")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-19")
    parser.add_argument("--label", default="flow_v2_rotation_prodlike_2025now")
    parser.add_argument("--positions", default="3,5,10")
    parser.add_argument("--rebalance-days", default="5,10")
    parser.add_argument("--selection-modes", default="base")
    parser.add_argument("--market-gates", default="base")
    parser.add_argument("--pool-filters", default="base")
    parser.add_argument("--score-modes", default="balanced")
    parser.add_argument("--hold-buffer", default="0")
    parser.add_argument("--min-score-improvement", default="0.0")
    parser.add_argument("--initial-capital", type=float, default=1_000_000_000.0)
    parser.add_argument("--commission-rate", type=float, default=0.0010)
    parser.add_argument("--sell-tax-rate", type=float, default=0.0010)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument(
        "--price-unit-multiplier",
        type=float,
        default=1000.0,
        help="Multiplier converting stored parquet prices to executable VND prices.",
    )
    args = parser.parse_args()

    out_dir = run_grid(
        universe=args.universe,
        start=args.start,
        end=args.end,
        label=args.label,
        positions_values=[int(item.strip()) for item in args.positions.split(",") if item.strip()],
        rebalance_values=[int(item.strip()) for item in args.rebalance_days.split(",") if item.strip()],
        selection_modes=[item.strip() for item in args.selection_modes.split(",") if item.strip()],
        market_gates=[item.strip() for item in args.market_gates.split(",") if item.strip()],
        pool_filters=[item.strip() for item in args.pool_filters.split(",") if item.strip()],
        score_modes=[item.strip() for item in args.score_modes.split(",") if item.strip()],
        hold_buffer_values=[int(item.strip()) for item in args.hold_buffer.split(",") if item.strip()],
        min_score_improvement_values=[
            float(item.strip()) for item in args.min_score_improvement.split(",") if item.strip()
        ],
        initial_capital=float(args.initial_capital),
        commission_rate=float(args.commission_rate),
        sell_tax_rate=float(args.sell_tax_rate),
        slippage_rate=float(args.slippage_rate),
        lot_size=int(args.lot_size),
        price_unit_multiplier=float(args.price_unit_multiplier),
    )
    print(f"[flow-v2-prod] saved {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
