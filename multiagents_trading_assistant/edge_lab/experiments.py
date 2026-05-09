"""Experiment helpers for robustness and threshold sweeps."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from multiagents_trading_assistant.edge_lab.backtest import PortfolioConfig, run_portfolio
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis
from multiagents_trading_assistant.edge_lab.metrics import benchmark_metrics, equity_metrics, trade_metrics
from multiagents_trading_assistant.edge_lab.universe import get_universe


def run_subperiod_robustness(
    hypotheses: list[Hypothesis],
    universe_name: str,
    periods: list[tuple[str, str]],
    config: PortfolioConfig,
) -> pd.DataFrame:
    rows = []
    universe = get_universe(universe_name)
    for start, end in periods:
        features, price_map = build_feature_table(universe, start, end)
        benchmark = benchmark_metrics(price_map, start, end)
        for hypothesis in hypotheses:
            trades, equity = run_portfolio(features, price_map, [hypothesis], start, end, config=config)
            rows.append(
                {
                    "period_start": start,
                    "period_end": end,
                    "hypothesis": hypothesis.name,
                    **trade_metrics(trades),
                    **equity_metrics(equity, config.initial_capital),
                    **benchmark,
                }
            )
    return pd.DataFrame(rows)


def generate_threshold_variants(
    hypothesis: Hypothesis,
    grid: dict[str, list[Any]],
) -> list[Hypothesis]:
    """Create hypothesis variants by replacing filter values by column name."""
    if not grid:
        return [hypothesis]
    keys = list(grid)
    variants = []
    for values in product(*(grid[key] for key in keys)):
        replacements = dict(zip(keys, values))
        filters = []
        suffix = []
        for rule in hypothesis.filters:
            new_rule = dict(rule)
            column = str(new_rule.get("column"))
            if column in replacements:
                new_rule["value"] = replacements[column]
                suffix.append(f"{column}={replacements[column]}")
            filters.append(new_rule)
        name = hypothesis.name + "__" + "__".join(suffix)
        variants.append(replace(hypothesis, name=name, filters=filters))
    return variants


def generate_risk_variants(
    hypothesis: Hypothesis,
    risk_grid: dict[str, list[Any]],
) -> list[Hypothesis]:
    """Create hypothesis variants by replacing risk config values."""
    if not risk_grid:
        return [hypothesis]
    keys = list(risk_grid)
    variants = []
    for values in product(*(risk_grid[key] for key in keys)):
        replacements = dict(zip(keys, values))
        risk = dict(hypothesis.risk)
        risk.update(replacements)
        suffix = "__".join(f"{key}={value}" for key, value in replacements.items())
        variants.append(replace(hypothesis, name=f"{hypothesis.name}__{suffix}", risk=risk))
    return variants


def run_threshold_grid(
    base_hypothesis: Hypothesis,
    grid: dict[str, list[Any]],
    universe_name: str,
    start: str,
    end: str,
    config: PortfolioConfig,
) -> pd.DataFrame:
    universe = get_universe(universe_name)
    features, price_map = build_feature_table(universe, start, end)
    benchmark = benchmark_metrics(price_map, start, end)
    rows = []
    for hypothesis in generate_threshold_variants(base_hypothesis, grid):
        trades, equity = run_portfolio(features, price_map, [hypothesis], start, end, config=config)
        rows.append(
            {
                "hypothesis": hypothesis.name,
                **trade_metrics(trades),
                **equity_metrics(equity, config.initial_capital),
                **benchmark,
            }
        )
    return pd.DataFrame(rows).sort_values(["sharpe", "total_return"], ascending=False)


def run_risk_grid(
    base_hypothesis: Hypothesis,
    risk_grid: dict[str, list[Any]],
    universe_name: str,
    start: str,
    end: str,
    config: PortfolioConfig,
) -> pd.DataFrame:
    universe = get_universe(universe_name)
    features, price_map = build_feature_table(universe, start, end)
    benchmark = benchmark_metrics(price_map, start, end)
    rows = []
    for hypothesis in generate_risk_variants(base_hypothesis, risk_grid):
        trades, equity = run_portfolio(features, price_map, [hypothesis], start, end, config=config)
        rows.append(
            {
                "hypothesis": hypothesis.name,
                "risk": hypothesis.risk,
                **trade_metrics(trades),
                **equity_metrics(equity, config.initial_capital),
                **benchmark,
            }
        )
    return pd.DataFrame(rows).sort_values(["sharpe", "total_return"], ascending=False)


def run_walk_forward_grid(
    base_hypothesis: Hypothesis,
    threshold_grid: dict[str, list[Any]],
    risk_grid: dict[str, list[Any]],
    universe_name: str,
    start: str,
    end: str,
    config: PortfolioConfig,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
    min_train_trades: int = 20,
    objective: str = "sharpe",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward parameter selection.

    For each fold, generate threshold and risk variants, select the best variant
    on the train window, then evaluate that exact variant on the next test window.
    """
    universe = get_universe(universe_name)
    full_features, full_price_map = build_feature_table(universe, start, end)
    threshold_variants = generate_threshold_variants(base_hypothesis, threshold_grid)
    variants = []
    for item in threshold_variants:
        variants.extend(generate_risk_variants(item, risk_grid))

    folds = _make_monthly_folds(start, end, train_months, test_months, step_months)
    selected_rows = []
    test_rows = []

    for fold_id, (train_start, train_end, test_start, test_end) in enumerate(folds, start=1):
        train_scores = []
        for hypothesis in variants:
            train_trades, train_equity = run_portfolio(
                full_features,
                full_price_map,
                [hypothesis],
                train_start,
                train_end,
                config=config,
            )
            metrics = {
                **trade_metrics(train_trades),
                **equity_metrics(train_equity, config.initial_capital),
            }
            if metrics.get("trades", 0) < min_train_trades:
                continue
            train_scores.append({"hypothesis_obj": hypothesis, **metrics})
        if not train_scores:
            continue

        train_df = pd.DataFrame([{k: v for k, v in row.items() if k != "hypothesis_obj"} for row in train_scores])
        best_idx = _select_best_index(train_df, objective)
        selected = train_scores[best_idx]["hypothesis_obj"]
        selected_train = train_df.iloc[best_idx].to_dict()

        test_trades, test_equity = run_portfolio(
            full_features,
            full_price_map,
            [selected],
            test_start,
            test_end,
            config=config,
        )
        test_metrics = {
            **trade_metrics(test_trades),
            **equity_metrics(test_equity, config.initial_capital),
            **benchmark_metrics(full_price_map, test_start, test_end),
        }
        selected_rows.append(
            {
                "fold": fold_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "selected_hypothesis": selected.name,
                "selected_filters": selected.filters,
                "selected_risk": selected.risk,
                **{f"train_{k}": v for k, v in selected_train.items()},
            }
        )
        test_rows.append(
            {
                "fold": fold_id,
                "test_start": test_start,
                "test_end": test_end,
                "selected_hypothesis": selected.name,
                "selected_risk": selected.risk,
                **test_metrics,
            }
        )

    return pd.DataFrame(selected_rows), pd.DataFrame(test_rows)


def _make_monthly_folds(
    start: str,
    end: str,
    train_months: int,
    test_months: int,
    step_months: int,
) -> list[tuple[str, str, str, str]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    folds = []
    train_start = start_ts
    while True:
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_start > end_ts:
            break
        if test_end > end_ts:
            test_end = end_ts
        if (test_end - test_start).days < 30:
            break
        folds.append((
            str(train_start.date()),
            str(train_end.date()),
            str(test_start.date()),
            str(test_end.date()),
        ))
        train_start = train_start + pd.DateOffset(months=step_months)
    return folds


def _select_best_index(frame: pd.DataFrame, objective: str) -> int:
    objective = objective.lower()
    data = frame.copy()
    if objective == "return":
        return int(data["total_return"].astype(float).idxmax())
    if objective == "profit_factor":
        return int(data["profit_factor"].replace([float("inf")], 999.0).astype(float).idxmax())
    if objective == "calmar":
        score = data["total_return"].astype(float) / data["max_drawdown"].astype(float).abs().replace(0, pd.NA)
        return int(score.fillna(-999).idxmax())
    # Default: robust Sharpe, tie-break by return and drawdown.
    score = (
        data["sharpe"].astype(float)
        + 0.25 * data["total_return"].astype(float)
        - 0.25 * data["max_drawdown"].astype(float).abs()
    )
    return int(score.idxmax())


def default_periods() -> list[tuple[str, str]]:
    return [
        ("2020-01-01", "2021-12-31"),
        ("2022-01-01", "2023-12-31"),
        ("2024-01-01", "2026-05-08"),
        ("2022-05-08", "2026-05-08"),
    ]


def write_experiment(path: str | Path, frames: dict[str, pd.DataFrame]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
