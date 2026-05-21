"""QuantAgents-style no-leak workflow using edge_lab research strategies.

This module bridges the existing QuantAgents orchestration layer with the
project's VN-specific research hypotheses and combos. It stays test-only and
deterministic: all signals come from cached daily features, and strategy
selection is walk-forward / out-of-sample only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import (
    Hypothesis,
    evaluate_filters,
    load_hypotheses,
    rank_candidates,
)
from multiagents_trading_assistant.quantagents_backtest.metrics import (
    compute_metrics,
    score_metrics,
    summarize_equity,
)
from multiagents_trading_assistant.quantagents_backtest.vn_quantagents import (
    RiskGateConfig,
    VNQuantAgentsConfig,
    _curve_only_summary,
    _dominant_regime,
    _equal_weight_portfolio,
    _mean,
    _std,
    compute_market_regime,
)


DEFAULT_COMBOS = [
    {
        "name": "LIVE_CURRENT_3core",
        "strategies": [
            "breakout_after_accumulation_v3",
            "compression_breakout_smt_v1",
            "mean_reversion_uptrend_ma50_v1",
        ],
    },
    {
        "name": "PURE_TOP3",
        "strategies": [
            "leader_pullback_market_regime_v3",
            "leader_pullback_market_healthy_v2",
            "breakout_55_smt_v1",
        ],
    },
    {
        "name": "TOP3_pullback_breakout_meanrev",
        "strategies": [
            "leader_pullback_market_regime_v3",
            "breakout_55_smt_v1",
            "mean_reversion_uptrend_ma20_v1",
        ],
    },
    {
        "name": "BREAKOUT_TRIO",
        "strategies": [
            "breakout_after_accumulation_v3",
            "breakout_55_smt_v1",
            "linreg_momentum_refined_v1",
        ],
    },
    {
        "name": "MEANREV_TRIO",
        "strategies": [
            "mean_reversion_uptrend_ma20_v1",
            "mean_reversion_uptrend_ma50_v1",
            "reclaim_ma20_quality_v1",
        ],
    },
    {
        "name": "SMT_TRIO",
        "strategies": [
            "money_cycle_reset_smt_confirm",
            "smart_money_strong_market_healthy",
            "accumulation_breakout_smt_v1",
        ],
    },
]


@dataclass(frozen=True)
class ResearchStrategy:
    strategy_id: str
    family: str
    description: str
    risk: dict[str, Any]
    members: tuple[str, ...]
    kind: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class EdgeResearchConfig:
    config_path: str
    strategy_source: str = "edge_hypotheses"
    combo_names: tuple[str, ...] = ()
    top_candidates_per_day: int = 5
    warmup_days: int = 370


@dataclass
class ResearchStrategyMemory:
    strategy_id: str
    family: str
    kind: str
    description: str
    risk: dict
    members: tuple[str, ...]
    symbol_scores: list[float] = field(default_factory=list)
    train_scores: list[float] = field(default_factory=list)
    test_returns: list[float] = field(default_factory=list)
    test_sharpes: list[float] = field(default_factory=list)
    test_drawdowns: list[float] = field(default_factory=list)
    win_rates: list[float] = field(default_factory=list)
    trades: int = 0
    symbols: set[str] = field(default_factory=set)
    risk_rejections: int = 0

    def record_train(self, score: float) -> None:
        if np.isfinite(score):
            self.train_scores.append(float(score))

    def record_test(self, symbol: str, score: float, metrics: dict) -> None:
        self.symbols.add(symbol)
        self.symbol_scores.append(float(score))
        self.test_returns.append(float(metrics.get("total_return", 0.0)))
        self.test_sharpes.append(float(metrics.get("sharpe_ratio", 0.0)))
        self.test_drawdowns.append(float(metrics.get("max_drawdown", 0.0)))
        self.win_rates.append(float(metrics.get("win_rate", 0.0)))
        self.trades += int(metrics.get("number_of_trades", 0))

    def to_row(self, cfg: VNQuantAgentsConfig) -> dict:
        avg_score = _mean(self.symbol_scores)
        score_std = _std(self.symbol_scores)
        avg_drawdown = _mean(self.test_drawdowns)
        symbols_tested = len(self.symbols)
        concentration_penalty = max(0, cfg.risk_gate.min_symbols_tested - symbols_tested)
        qa_score = (
            avg_score
            - cfg.risk_gate.instability_penalty_weight * score_std
            - cfg.risk_gate.drawdown_penalty_weight * abs(min(avg_drawdown, 0.0))
            - cfg.risk_gate.concentration_penalty_weight * concentration_penalty
            + 0.02 * symbols_tested
        )
        return {
            "strategy_id": self.strategy_id,
            "family": self.family,
            "kind": self.kind,
            "description": self.description,
            "members": ",".join(self.members),
            "risk": self.risk,
            "symbols_tested": symbols_tested,
            "avg_score": avg_score,
            "score_std": score_std,
            "avg_total_return": _mean(self.test_returns),
            "avg_sharpe_ratio": _mean(self.test_sharpes),
            "avg_max_drawdown": avg_drawdown,
            "avg_win_rate": _mean(self.win_rates),
            "total_trades": self.trades,
            "risk_rejections": self.risk_rejections,
            "qa_score": qa_score,
        }


def load_research_strategies(config: EdgeResearchConfig) -> tuple[list[ResearchStrategy], dict[str, Hypothesis]]:
    hypotheses = load_hypotheses(config.config_path)
    by_name = {item.name: item for item in hypotheses}
    if config.strategy_source == "edge_hypotheses":
        strategies = [
            ResearchStrategy(
                strategy_id=item.name,
                family=_family_from_hypothesis(item),
                description=item.description,
                risk=dict(item.risk),
                members=(item.name,),
                kind="hypothesis",
                tags=tuple(item.tags),
            )
            for item in hypotheses
        ]
        return strategies, by_name

    combos = DEFAULT_COMBOS
    if config.combo_names:
        wanted = {name.strip() for name in config.combo_names}
        combos = [combo for combo in combos if combo["name"] in wanted]
    strategies: list[ResearchStrategy] = []
    for combo in combos:
        members = tuple(name for name in combo["strategies"] if name in by_name)
        if not members:
            continue
        strategies.append(
            ResearchStrategy(
                strategy_id=combo["name"],
                family="combo",
                description=f"Combo of {', '.join(members)}",
                risk=_blend_risk([by_name[name].risk for name in members]),
                members=members,
                kind="combo",
                tags=("combo",),
            )
        )
    return strategies, by_name


def run_edge_vn_quantagents(
    universe: list[str],
    market_index: pd.DataFrame,
    config: VNQuantAgentsConfig,
    research_config: EdgeResearchConfig,
    root: str = ".",
) -> dict:
    features, price_map = build_feature_table(
        universe,
        start=config.start,
        end=config.end,
        root=root,
        warmup_days=research_config.warmup_days,
    )
    strategies, hypotheses = load_research_strategies(research_config)
    regime = compute_market_regime(market_index.set_index("date") if "date" in market_index.columns else market_index, config)
    memory = _init_memory(strategies)
    symbol_reports: list[dict] = []
    fold_reports: list[dict] = []
    best_fold_curves: dict[str, list[pd.Series]] = {}
    ensemble_fold_curves: dict[str, list[pd.Series]] = {}
    execution_schedule: dict[int, dict] = {}

    for symbol, raw_price in price_map.items():
        if symbol not in {item.upper() for item in universe}:
            continue
        price = raw_price.copy()
        price["date"] = pd.to_datetime(price["date"])
        price = price[(price["date"] >= pd.Timestamp(config.start)) & (price["date"] <= pd.Timestamp(config.end))].copy()
        if price.empty:
            continue
        price = price[["date", "open", "high", "low", "close", "volume"]].drop_duplicates("date").sort_values("date")
        price = price.set_index("date")
        symbol_features = features[features["symbol"].astype(str).str.upper() == symbol].copy()
        symbol_features["date"] = pd.to_datetime(symbol_features["date"])
        symbol_features = symbol_features[(symbol_features["date"] >= price.index.min()) & (symbol_features["date"] <= price.index.max())]
        symbol_features = symbol_features.drop_duplicates("date").sort_values("date")
        folds = _make_folds(price, config.train_bars, config.test_bars, config.step_bars)
        if not folds:
            continue
        symbol_signal_tables = _precompute_signal_tables(
            symbol_features,
            strategies,
            hypotheses,
            research_config.top_candidates_per_day,
        )
        symbol_best_rows = []

        for fold_id, (train_price, test_price) in enumerate(folds, start=1):
            train_features = symbol_features[symbol_features["date"].between(train_price.index[0], train_price.index[-1])].copy()
            test_features = symbol_features[symbol_features["date"].between(test_price.index[0], test_price.index[-1])].copy()
            fold_regime = regime.reindex(test_price.index).ffill().bfill()
            ranked = _rank_train_window_edge(
                train_price,
                train_features,
                strategies,
                hypotheses,
                config,
                memory,
                research_config,
                symbol_signal_tables,
            )
            selected = _select_with_risk_gate_edge(ranked, config, memory)
            if not selected:
                selected = [row["strategy"] for row in ranked[: config.top_k]]
            selected = selected[: config.top_k]
            test_results = [
                backtest_edge_strategy(
                    test_price,
                    test_features,
                    strategy,
                    hypotheses,
                    research_config.top_candidates_per_day,
                    signal_table=symbol_signal_tables.get(strategy.strategy_id),
                )
                for strategy in selected
            ]
            ensemble_curve, ensemble_metrics = _ensemble_edge_results(test_results, fold_regime, config)
            if test_results:
                best_fold_curves.setdefault(symbol, []).append(apply_regime_scaling_edge(test_results[0].equity_curve, fold_regime, config))
            ensemble_fold_curves.setdefault(symbol, []).append(ensemble_curve)
            execution_schedule.setdefault(
                fold_id,
                {
                    "test_start": test_price.index[0],
                    "test_end": test_price.index[-1],
                    "market_regime": fold_regime,
                    "strategy_map": {},
                    "window_universe": {},
                    "window_features": {},
                    "window_signals": {},
                },
            )
            execution_schedule[fold_id]["strategy_map"][symbol] = selected
            execution_schedule[fold_id]["window_universe"][symbol] = test_price.copy()
            execution_schedule[fold_id]["window_features"][symbol] = test_features.copy()
            execution_schedule[fold_id]["window_signals"][symbol] = {
                strategy.strategy_id: _slice_signal_table(
                    symbol_signal_tables.get(strategy.strategy_id),
                    test_price.index[0],
                    test_price.index[-1],
                )
                for strategy in selected
            }

            for result in test_results:
                metrics = compute_metrics(
                    apply_regime_scaling_edge(result.equity_curve, fold_regime, config),
                    result.trades,
                    config.backtest.periods_per_year,
                )
                score = score_metrics(metrics)
                memory[result.strategy.strategy_id].record_test(symbol, score, metrics)
                symbol_best_rows.append(
                    {
                        "symbol": symbol,
                        "fold": fold_id,
                        "strategy_id": result.strategy.strategy_id,
                        "score": score,
                        **metrics,
                    }
                )

            fold_reports.append(
                {
                    "symbol": symbol,
                    "fold": fold_id,
                    "test_start": test_price.index[0],
                    "test_end": test_price.index[-1],
                    "market_regime": _dominant_regime(fold_regime),
                    "selected_strategy_ids": ",".join(strategy.strategy_id for strategy in selected),
                    **{f"ensemble_{key}": value for key, value in ensemble_metrics.items()},
                }
            )

        if symbol_best_rows:
            best_df = pd.DataFrame(symbol_best_rows)
            symbol_reports.append(
                {
                    "symbol": symbol,
                    "folds": best_df["fold"].nunique(),
                    "avg_score": best_df["score"].mean(),
                    "avg_total_return": best_df["total_return"].mean(),
                    "avg_sharpe_ratio": best_df["sharpe_ratio"].mean(),
                    "avg_max_drawdown": best_df["max_drawdown"].mean(),
                    "avg_win_rate": best_df["win_rate"].mean(),
                    "total_trades": int(best_df["number_of_trades"].sum()),
                }
            )

    strategy_memory = pd.DataFrame([item.to_row(config) for item in memory.values()]).sort_values("qa_score", ascending=False)
    research_summary, best_equity, ensemble_equity = _build_edge_research_summary(best_fold_curves, ensemble_fold_curves, config)
    execution_portfolio = run_oos_edge_execution_portfolio(execution_schedule, hypotheses, config, research_config)
    portfolio_summary = pd.DataFrame([{"portfolio": "qa_execution_schedule", **execution_portfolio["metrics"]}])
    return {
        "strategy_memory": strategy_memory.reset_index(drop=True),
        "symbol_reports": pd.DataFrame(symbol_reports),
        "fold_reports": pd.DataFrame(fold_reports),
        "top_strategies": strategy_memory.head(config.top_k)["strategy_id"].tolist(),
        "portfolio_summary": portfolio_summary,
        "research_summary": research_summary,
        "best_strategy_equity": best_equity,
        "ensemble_equity": ensemble_equity,
        "execution_portfolio": execution_portfolio,
        "regime": regime,
        "features": features,
        "price_map": price_map,
    }


@dataclass(frozen=True)
class EdgeBacktestResult:
    strategy: ResearchStrategy
    equity_curve: pd.Series
    trades: list[dict]
    metrics: dict
    signal: pd.Series


def backtest_edge_strategy(
    price: pd.DataFrame,
    features: pd.DataFrame,
    strategy: ResearchStrategy,
    hypotheses: dict[str, Hypothesis],
    top_candidates_per_day: int,
    signal_table: pd.DataFrame | None = None,
) -> EdgeBacktestResult:
    if signal_table is None:
        signal_table = build_edge_signal_table(features, strategy, hypotheses, top_candidates_per_day)
    signal_table = _slice_signal_table(signal_table, price.index.min(), price.index.max())
    signal = signal_table.set_index("date")["passed"].reindex(price.index).fillna(False).astype(bool)
    rank_score = signal_table.set_index("date")["rank_score"].reindex(price.index).fillna(0.0)
    cash = 100_000.0
    shares = 0.0
    entry_price = 0.0
    entry_time: pd.Timestamp | None = None
    holding_bars = 0
    trades: list[dict] = []
    equity_values: list[float] = []
    tradable = signal.shift(1, fill_value=False)
    risk = strategy.risk or {"stop_loss": 0.08, "take_profit": 0.18, "max_holding_bars": 60}
    stop_loss = float(risk.get("stop_loss", 0.08))
    take_profit = float(risk.get("take_profit", 0.18))
    max_holding_bars = int(risk.get("max_holding_bars", 60))

    for timestamp, row in price.iterrows():
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])
        current_signal = bool(tradable.loc[timestamp])
        if shares > 0:
            holding_bars += 1
            exit_price = None
            exit_reason = None
            if not current_signal:
                exit_price = open_price * (1 - 0.0005)
                exit_reason = "OPPOSITE_SIGNAL"
            elif low_price <= entry_price * (1 - stop_loss):
                exit_price = entry_price * (1 - stop_loss) * (1 - 0.0005)
                exit_reason = "STOP_LOSS"
            elif high_price >= entry_price * (1 + take_profit):
                exit_price = entry_price * (1 + take_profit) * (1 - 0.0005)
                exit_reason = "TAKE_PROFIT"
            elif holding_bars >= max_holding_bars:
                exit_price = close_price * (1 - 0.0005)
                exit_reason = "MAX_HOLDING"
            if exit_price is not None:
                gross = shares * exit_price
                cash = gross * (1 - 0.001)
                entry_value = shares * entry_price
                pnl = cash - entry_value
                trades.append(
                    {
                        "strategy_id": strategy.strategy_id,
                        "entry_time": entry_time or timestamp,
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "shares": shares,
                        "pnl": pnl,
                        "pnl_pct": pnl / entry_value if entry_value else 0.0,
                        "exit_reason": exit_reason,
                        "holding_bars": holding_bars,
                    }
                )
                shares = 0.0
        if shares == 0 and current_signal and rank_score.loc[timestamp] > 0 and open_price > 0:
            fill_price = open_price * (1 + 0.0005)
            shares = cash * (1 - 0.001) / fill_price
            entry_price = fill_price
            entry_time = timestamp
            cash = 0.0
            holding_bars = 0
        equity_values.append(cash if shares == 0 else shares * close_price)

    if shares > 0 and entry_time is not None:
        timestamp = price.index[-1]
        exit_price = float(price["close"].iloc[-1]) * (1 - 0.0005)
        gross = shares * exit_price
        cash = gross * (1 - 0.001)
        entry_value = shares * entry_price
        pnl = cash - entry_value
        trades.append(
            {
                "strategy_id": strategy.strategy_id,
                "entry_time": entry_time,
                "exit_time": timestamp,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "pnl": pnl,
                "pnl_pct": pnl / entry_value if entry_value else 0.0,
                "exit_reason": "END_OF_DATA",
                "holding_bars": holding_bars,
            }
        )
        equity_values[-1] = cash

    equity_curve = pd.Series(equity_values, index=price.index, name=strategy.strategy_id)
    metrics = compute_metrics(equity_curve, trades, 252)
    return EdgeBacktestResult(strategy, equity_curve, trades, metrics, signal)


def build_edge_signal_table(
    features: pd.DataFrame,
    strategy: ResearchStrategy,
    hypotheses: dict[str, Hypothesis],
    top_candidates_per_day: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for date, day in features.groupby("date", sort=True):
        daily_rows = []
        for member in strategy.members:
            hypothesis = hypotheses.get(member)
            if hypothesis is None:
                continue
            mask = evaluate_filters(day, hypothesis)
            ranked = rank_candidates(day[mask], hypothesis).head(top_candidates_per_day)
            for rank_idx, row in enumerate(ranked.to_dict("records"), start=1):
                daily_rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "symbol": str(row["symbol"]).upper(),
                        "hypothesis": member,
                        "passed": True,
                        "rank_score": float(row.get("_edge_rank", 0.0)),
                        "rank_idx": rank_idx,
                    }
                )
        if not daily_rows:
            continue
        ranked_day = pd.DataFrame(daily_rows).sort_values(["symbol", "rank_score"], ascending=[True, False])
        ranked_day = ranked_day.drop_duplicates("symbol", keep="first")
        rows.extend(ranked_day.to_dict("records"))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "symbol", "passed", "rank_score"])


def _precompute_signal_tables(
    features: pd.DataFrame,
    strategies: list[ResearchStrategy],
    hypotheses: dict[str, Hypothesis],
    top_candidates_per_day: int,
) -> dict[str, pd.DataFrame]:
    return {
        strategy.strategy_id: build_edge_signal_table(features, strategy, hypotheses, top_candidates_per_day)
        for strategy in strategies
    }


def _slice_signal_table(
    signal_table: pd.DataFrame | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if signal_table is None or signal_table.empty:
        return pd.DataFrame(columns=["date", "symbol", "passed", "rank_score"])
    frame = signal_table.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()


def apply_regime_scaling_edge(equity_curve: pd.Series, regime: pd.Series, config: VNQuantAgentsConfig) -> pd.Series:
    aligned = regime.reindex(equity_curve.index).ffill().fillna("NEUTRAL")
    returns = equity_curve.pct_change().fillna(0.0)
    scale = pd.Series(1.0, index=equity_curve.index)
    scale[aligned == "RISK_OFF"] = config.regime_risk_off_scale
    scale[aligned == "CRASH"] = 0.0
    return ((1.0 + returns * scale).cumprod() * float(equity_curve.iloc[0])).rename(equity_curve.name)


def _ensemble_edge_results(results: list[EdgeBacktestResult], regime: pd.Series, config: VNQuantAgentsConfig) -> tuple[pd.Series, dict]:
    if not results:
        return pd.Series(dtype=float), {"total_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "number_of_trades": 0}
    curves = pd.concat(
        [apply_regime_scaling_edge(result.equity_curve, regime, config) for result in results],
        axis=1,
    ).ffill()
    ensemble_curve = curves.mean(axis=1)
    ensemble_curve.name = "edge_ensemble"
    metrics = compute_metrics(ensemble_curve, [trade for result in results for trade in result.trades], 252)
    return ensemble_curve, metrics


def run_oos_edge_execution_portfolio(
    execution_schedule: dict[int, dict],
    hypotheses: dict[str, Hypothesis],
    config: VNQuantAgentsConfig,
    research_config: EdgeResearchConfig,
) -> dict:
    fold_ids = sorted(execution_schedule)
    current_capital = config.portfolio.initial_capital
    all_trades: list[dict] = []
    equity_frames: list[pd.DataFrame] = []
    for fold_id in fold_ids:
        fold = execution_schedule[fold_id]
        result = backtest_edge_schedule_window(
            fold["window_universe"],
            fold["window_features"],
            fold["strategy_map"],
            fold["market_regime"],
            current_capital,
            hypotheses,
            research_config.top_candidates_per_day,
            config,
            window_signal_tables=fold.get("window_signals"),
        )
        frame = result["equity_frame"]
        if equity_frames:
            frame = frame.loc[~frame.index.isin(equity_frames[-1].index)]
        equity_frames.append(frame)
        all_trades.extend(result["trades"])
        if not result["equity_curve"].empty:
            current_capital = float(result["equity_curve"].iloc[-1])
    equity_frame = pd.concat(equity_frames).sort_index() if equity_frames else pd.DataFrame()
    equity_curve = equity_frame["equity"].rename("edge_quantagents_oos") if not equity_frame.empty else pd.Series(dtype=float)
    metrics = compute_metrics(equity_curve, all_trades, config.portfolio.periods_per_year) if not equity_curve.empty else {}
    return {
        "equity_curve": equity_curve,
        "equity_frame": equity_frame,
        "trades": all_trades,
        "metrics": metrics,
        "open_positions": {},
    }


def backtest_edge_schedule_window(
    window_universe: dict[str, pd.DataFrame],
    window_features: dict[str, pd.DataFrame],
    strategy_map: dict[str, list[ResearchStrategy]],
    market_regime: pd.Series,
    initial_capital: float,
    hypotheses: dict[str, Hypothesis],
    top_candidates_per_day: int,
    config: VNQuantAgentsConfig,
    window_signal_tables: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> dict:
    cash = initial_capital
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []
    calendar = sorted(set().union(*(frame.index for frame in window_universe.values())))
    signal_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol, strategies in strategy_map.items():
        feature_frame = window_features.get(symbol)
        if feature_frame is None or feature_frame.empty:
            continue
        for strategy in strategies:
            supplied = (window_signal_tables or {}).get(symbol, {}).get(strategy.strategy_id)
            signal_cache[(symbol, strategy.strategy_id)] = (
                supplied
                if supplied is not None
                else build_edge_signal_table(feature_frame, strategy, hypotheses, top_candidates_per_day)
            )

    for timestamp in calendar:
        regime = market_regime.reindex([timestamp]).ffill().fillna("NEUTRAL").iloc[0]
        for symbol in list(positions):
            frame = window_universe.get(symbol)
            if frame is None or timestamp not in frame.index:
                continue
            row = frame.loc[timestamp]
            pos = positions[symbol]
            pos["holding_bars"] += 1
            exit_reason = _edge_exit_reason(row, pos, str(regime))
            if exit_reason:
                fill_price = float(row["open"]) * (1 - config.portfolio.costs.slippage_rate)
                gross = pos["shares"] * fill_price
                fees = gross * (config.portfolio.costs.commission_rate + config.portfolio.costs.sell_tax_rate)
                net = gross - fees
                pnl = net - pos["entry_value"]
                trades.append(
                    {
                        "symbol": symbol,
                        "entry_time": pos["entry_time"],
                        "exit_time": timestamp,
                        "entry_price": pos["entry_price"],
                        "exit_price": fill_price,
                        "shares": pos["shares"],
                        "entry_value": pos["entry_value"],
                        "exit_value": net,
                        "pnl": pnl,
                        "pnl_pct": pnl / pos["entry_value"] if pos["entry_value"] else 0.0,
                        "exit_reason": exit_reason,
                        "holding_bars": pos["holding_bars"],
                        "strategy_ids": ",".join(pos["strategy_ids"]),
                    }
                )
                cash += net
                del positions[symbol]

        allow_new = not (str(regime) == "RISK_OFF" and not config.portfolio.allow_new_in_risk_off) and str(regime) != "CRASH"
        if allow_new:
            candidates = _edge_entry_candidates(timestamp, window_universe, strategy_map, signal_cache, positions)
            candidates = candidates[: max(0, config.portfolio.max_positions - len(positions))]
            if candidates:
                target_exposure = _edge_target_exposure(str(regime), config)
                equity_before = cash + _edge_mark_to_market(positions, window_universe, timestamp)
                deployable = max(0.0, min(cash, equity_before * target_exposure - _edge_mark_to_market(positions, window_universe, timestamp)))
                budget = deployable / len(candidates) if candidates else 0.0
                for candidate in candidates:
                    if budget < config.portfolio.min_position_value:
                        continue
                    row = window_universe[candidate["symbol"]].loc[timestamp]
                    open_price = float(row["open"])
                    fill = open_price * (1 + config.portfolio.costs.slippage_rate)
                    gross_shares = min(budget / fill, float(row["volume"]) * config.portfolio.costs.max_participation_rate)
                    shares = int(gross_shares // config.portfolio.costs.lot_size * config.portfolio.costs.lot_size)
                    gross = shares * fill
                    total_cost = gross * (1 + config.portfolio.costs.commission_rate)
                    if shares <= 0 or total_cost > cash or gross < config.portfolio.min_position_value:
                        continue
                    risk = candidate["risk"]
                    positions[candidate["symbol"]] = {
                        "entry_time": timestamp,
                        "entry_price": fill,
                        "shares": shares,
                        "entry_value": gross,
                        "stop_loss": float(risk.get("stop_loss", 0.08)),
                        "take_profit": float(risk.get("take_profit", 0.20)),
                        "max_holding_bars": int(risk.get("max_holding_bars", 60)),
                        "strategy_ids": tuple(candidate["strategy_ids"]),
                        "holding_bars": 0,
                    }
                    cash -= total_cost

        equity = cash + _edge_mark_to_market(positions, window_universe, timestamp)
        equity_rows.append(
            {
                "date": timestamp,
                "equity": equity,
                "cash": cash,
                "positions": len(positions),
                "regime": regime,
                "gross_exposure": _edge_mark_to_market(positions, window_universe, timestamp) / equity if equity else 0.0,
            }
        )

    equity_frame = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame()
    equity_curve = equity_frame["equity"].rename("edge_schedule_window") if not equity_frame.empty else pd.Series(dtype=float)
    return {"equity_curve": equity_curve, "equity_frame": equity_frame, "trades": trades}


def _edge_entry_candidates(
    timestamp: pd.Timestamp,
    window_universe: dict[str, pd.DataFrame],
    strategy_map: dict[str, list[ResearchStrategy]],
    signal_cache: dict[tuple[str, str], pd.DataFrame],
    positions: dict[str, dict],
) -> list[dict]:
    rows = []
    for symbol, strategies in strategy_map.items():
        if symbol in positions or symbol not in window_universe or timestamp not in window_universe[symbol].index:
            continue
        active = []
        for strategy in strategies:
            table = signal_cache.get((symbol, strategy.strategy_id))
            if table is None or table.empty:
                continue
            matched = table[table["date"] == timestamp]
            if matched.empty:
                continue
            active.append((strategy, float(matched["rank_score"].max())))
        if not active:
            continue
        active.sort(key=lambda item: item[1], reverse=True)
        rows.append(
            {
                "symbol": symbol,
                "vote_count": len(active),
                "rank_score": float(np.mean([item[1] for item in active])),
                "strategy_ids": tuple(item[0].strategy_id for item in active),
                "risk": _blend_risk([item[0].risk for item in active]),
            }
        )
    rows.sort(key=lambda item: (item["vote_count"], item["rank_score"]), reverse=True)
    return rows


def _rank_train_window_edge(
    train_price: pd.DataFrame,
    train_features: pd.DataFrame,
    strategies: list[ResearchStrategy],
    hypotheses: dict[str, Hypothesis],
    config: VNQuantAgentsConfig,
    memory: dict[str, ResearchStrategyMemory],
    research_config: EdgeResearchConfig,
    signal_tables: dict[str, pd.DataFrame] | None = None,
) -> list[dict]:
    ranked = []
    for strategy in strategies:
        result = backtest_edge_strategy(
            train_price,
            train_features,
            strategy,
            hypotheses,
            research_config.top_candidates_per_day,
            signal_table=(signal_tables or {}).get(strategy.strategy_id),
        )
        metrics = result.metrics
        base_score = score_metrics(metrics)
        memory[strategy.strategy_id].record_train(base_score)
        unstable = _std(memory[strategy.strategy_id].train_scores)
        risk_reject = (
            metrics["number_of_trades"] < config.risk_gate.min_train_trades
            or metrics["max_drawdown"] < config.risk_gate.max_train_drawdown
            or metrics["sharpe_ratio"] < config.risk_gate.min_train_sharpe
            or unstable > config.risk_gate.max_score_std
        )
        adjusted_score = (
            base_score
            - config.risk_gate.instability_penalty_weight * unstable
            - config.risk_gate.drawdown_penalty_weight * abs(min(metrics["max_drawdown"], 0.0))
        )
        ranked.append(
            {
                "strategy": strategy,
                "risk_reject": risk_reject,
                "unstable_score": unstable,
                "adjusted_score": adjusted_score,
                **metrics,
            }
        )
    ranked.sort(key=lambda row: row["adjusted_score"], reverse=True)
    return ranked


def _select_with_risk_gate_edge(
    ranked: list[dict],
    config: VNQuantAgentsConfig,
    memory: dict[str, ResearchStrategyMemory],
) -> list[ResearchStrategy]:
    selected = []
    for row in ranked:
        strategy = row["strategy"]
        if row["risk_reject"]:
            memory[strategy.strategy_id].risk_rejections += 1
            continue
        selected.append(strategy)
        if len(selected) >= config.top_k:
            break
    return selected


def _init_memory(strategies: list[ResearchStrategy]) -> dict[str, ResearchStrategyMemory]:
    return {
        strategy.strategy_id: ResearchStrategyMemory(
            strategy_id=strategy.strategy_id,
            family=strategy.family,
            kind=strategy.kind,
            description=strategy.description,
            risk=strategy.risk,
            members=strategy.members,
        )
        for strategy in strategies
    }


def _make_folds(
    price: pd.DataFrame,
    train_bars: int,
    test_bars: int,
    step_bars: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    folds = []
    start = 0
    while start + train_bars + test_bars <= len(price):
        train = price.iloc[start : start + train_bars].copy()
        test = price.iloc[start + train_bars : start + train_bars + test_bars].copy()
        folds.append((train, test))
        start += step_bars
    return folds


def _build_edge_research_summary(
    best_fold_curves: dict[str, list[pd.Series]],
    ensemble_fold_curves: dict[str, list[pd.Series]],
    config: VNQuantAgentsConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    best_curves = {symbol: _stitch_fold_curves(curves) for symbol, curves in best_fold_curves.items() if curves}
    ensemble_curves = {
        symbol: _stitch_fold_curves(curves) for symbol, curves in ensemble_fold_curves.items() if curves
    }
    best_equity = _equal_weight_portfolio(best_curves, config.initial_capital, "edge_best_strategy_oos")
    ensemble_equity = _equal_weight_portfolio(ensemble_curves, config.initial_capital, "edge_ensemble_oos")
    summary = pd.DataFrame(
        [
            _curve_only_summary(best_equity, "edge_best_strategy_oos", config.backtest.periods_per_year),
            _curve_only_summary(ensemble_equity, "edge_ensemble_oos", config.backtest.periods_per_year),
        ]
    )
    return summary, best_equity, ensemble_equity


def _stitch_fold_curves(curves: list[pd.Series]) -> pd.Series:
    stitched = pd.concat(curves).sort_index()
    return stitched[~stitched.index.duplicated(keep="last")]


def _edge_mark_to_market(positions: dict[str, dict], window_universe: dict[str, pd.DataFrame], timestamp: pd.Timestamp) -> float:
    total = 0.0
    for symbol, position in positions.items():
        frame = window_universe.get(symbol)
        if frame is None or timestamp not in frame.index:
            continue
        total += position["shares"] * float(frame.loc[timestamp, "close"])
    return total


def _edge_target_exposure(regime: str, config: VNQuantAgentsConfig) -> float:
    if regime == "RISK_ON":
        return config.portfolio.risk_on_exposure
    if regime == "RISK_OFF":
        return config.portfolio.risk_off_exposure
    if regime == "CRASH":
        return config.portfolio.crash_exposure
    return config.portfolio.neutral_exposure


def _edge_exit_reason(row: pd.Series, position: dict, regime: str) -> str | None:
    if regime == "CRASH":
        return "REGIME_CRASH"
    low = float(row["low"])
    high = float(row["high"])
    if low <= position["entry_price"] * (1 - position["stop_loss"]):
        return "STOP_LOSS"
    if high >= position["entry_price"] * (1 + position["take_profit"]):
        return "TAKE_PROFIT"
    if position["holding_bars"] >= position["max_holding_bars"]:
        return "MAX_HOLDING"
    return None


def _family_from_hypothesis(hypothesis: Hypothesis) -> str:
    if hypothesis.tags:
        return str(hypothesis.tags[0])
    return "hypothesis"


def _blend_risk(risks: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [risk for risk in risks if risk]
    if not clean:
        return {"stop_loss": 0.08, "take_profit": 0.18, "max_holding_bars": 60}
    return {
        "stop_loss": float(np.median([float(item.get("stop_loss", 0.08)) for item in clean])),
        "take_profit": float(np.median([float(item.get("take_profit", 0.18)) for item in clean])),
        "max_holding_bars": int(np.median([int(item.get("max_holding_bars", 60)) for item in clean])),
    }
