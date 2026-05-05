"""Rolling walk-forward validation with instability penalty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from multiagents_trading_assistant.quantagents_backtest.backtest_engine import (
    BacktestConfig,
    BacktestResult,
    backtest_ensemble,
    backtest_strategy,
)
from multiagents_trading_assistant.quantagents_backtest.indicators import add_indicators
from multiagents_trading_assistant.quantagents_backtest.metrics import score_metrics
from multiagents_trading_assistant.quantagents_backtest.strategy_generator import Strategy


@dataclass(frozen=True)
class WalkForwardConfig:
    train_bars: int = 504
    test_bars: int = 126
    step_bars: int = 63
    top_k: int = 7
    min_trades_train: int = 3
    instability_weight: float = 0.25
    backtest: BacktestConfig = BacktestConfig()


def run_walk_forward(
    ohlcv: pd.DataFrame,
    strategies: list[Strategy],
    config: WalkForwardConfig | None = None,
) -> dict:
    """Run rolling train/test strategy selection and ensemble evaluation."""

    cfg = config or WalkForwardConfig()
    data = add_indicators(ohlcv)
    folds = _make_folds(data, cfg.train_bars, cfg.test_bars, cfg.step_bars)
    if not folds:
        raise ValueError("Not enough data for the requested walk-forward windows")

    train_scores: dict[str, list[float]] = {strategy.strategy_id: [] for strategy in strategies}
    rows: list[dict] = []
    fold_payloads: list[dict] = []

    for fold_id, (train, test) in enumerate(folds, start=1):
        train_ranked = _rank_on_window(
            train,
            strategies,
            cfg,
            train_scores=train_scores,
        )
        selected = [row["strategy"] for row in train_ranked[: cfg.top_k]]
        test_results = [backtest_strategy(test, strategy, cfg.backtest) for strategy in selected]
        ensemble = backtest_ensemble(test, selected, cfg.backtest)

        for result in test_results:
            metrics = result.metrics
            strategy = result.strategy
            if strategy is None:
                continue
            rows.append(
                {
                    "fold": fold_id,
                    "strategy_id": strategy.strategy_id,
                    "family": strategy.family,
                    "score": score_metrics(metrics),
                    **metrics,
                }
            )

        fold_payloads.append(
            {
                "fold": fold_id,
                "train_start": train.index[0],
                "train_end": train.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "selected_strategy_ids": [strategy.strategy_id for strategy in selected],
                "test_results": test_results,
                "ensemble_result": ensemble,
            }
        )

    ranked = _aggregate_rankings(pd.DataFrame(rows), strategies, train_scores, cfg)
    top_strategies = _strategies_from_ids(strategies, ranked.head(cfg.top_k)["strategy_id"].tolist())
    best_strategy = top_strategies[0]
    best_full = backtest_strategy(data, best_strategy, cfg.backtest)
    ensemble_full = backtest_ensemble(data, top_strategies, cfg.backtest)

    return {
        "ranked_table": ranked,
        "best_strategy": best_strategy,
        "best_strategy_result": best_full,
        "ensemble_result": ensemble_full,
        "folds": fold_payloads,
        "feature_data": data,
    }


def _rank_on_window(
    train: pd.DataFrame,
    strategies: list[Strategy],
    cfg: WalkForwardConfig,
    train_scores: dict[str, list[float]],
) -> list[dict]:
    ranked = []
    for strategy in strategies:
        result = backtest_strategy(train, strategy, cfg.backtest)
        if result.metrics["number_of_trades"] < cfg.min_trades_train:
            base_score = -999.0
        else:
            base_score = score_metrics(result.metrics)
        train_scores[strategy.strategy_id].append(base_score)
        instability = _instability_penalty(train_scores[strategy.strategy_id], cfg.instability_weight)
        ranked.append(
            {
                "strategy": strategy,
                "train_score": base_score,
                "instability_penalty": instability,
                "stable_score": base_score - instability,
                **result.metrics,
            }
        )
    ranked.sort(key=lambda row: row["stable_score"], reverse=True)
    return ranked


def _aggregate_rankings(
    rows: pd.DataFrame,
    strategies: list[Strategy],
    train_scores: dict[str, list[float]],
    cfg: WalkForwardConfig,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()

    grouped = rows.groupby("strategy_id", as_index=False).agg(
        family=("family", "first"),
        folds=("fold", "nunique"),
        total_return=("total_return", "mean"),
        sharpe_ratio=("sharpe_ratio", "mean"),
        max_drawdown=("max_drawdown", "mean"),
        win_rate=("win_rate", "mean"),
        number_of_trades=("number_of_trades", "sum"),
        test_score=("score", "mean"),
        score_std=("score", "std"),
    )
    grouped["score_std"] = grouped["score_std"].fillna(0.0)
    grouped["instability_penalty"] = grouped["strategy_id"].map(
        lambda strategy_id: _instability_penalty(train_scores.get(strategy_id, []), cfg.instability_weight)
    )
    grouped["final_score"] = grouped["test_score"] - grouped["score_std"] * cfg.instability_weight - grouped[
        "instability_penalty"
    ]

    descriptions = {strategy.strategy_id: strategy.describe() for strategy in strategies}
    grouped["conditions"] = grouped["strategy_id"].map(
        lambda strategy_id: " | ".join(descriptions[strategy_id]["conditions"])
    )
    grouped["logic"] = grouped["strategy_id"].map(lambda strategy_id: descriptions[strategy_id]["logic"])
    grouped["risk"] = grouped["strategy_id"].map(lambda strategy_id: descriptions[strategy_id]["risk"])
    return grouped.sort_values("final_score", ascending=False).reset_index(drop=True)


def _make_folds(
    data: pd.DataFrame,
    train_bars: int,
    test_bars: int,
    step_bars: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    folds = []
    start = 0
    while start + train_bars + test_bars <= len(data):
        train = data.iloc[start : start + train_bars].copy()
        test = data.iloc[start + train_bars : start + train_bars + test_bars].copy()
        folds.append((train, test))
        start += step_bars
    return folds


def _instability_penalty(scores: list[float], weight: float) -> float:
    clean = [score for score in scores if np.isfinite(score) and score > -100]
    if len(clean) < 2:
        return 0.0
    return float(np.std(clean, ddof=0) * weight)


def _strategies_from_ids(strategies: list[Strategy], ids: list[str]) -> list[Strategy]:
    by_id = {strategy.strategy_id: strategy for strategy in strategies}
    return [by_id[strategy_id] for strategy_id in ids]
