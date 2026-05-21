"""Vietnam-focused QuantAgents-style simulated trading workflow.

This module keeps the QuantAgents idea in the deterministic simulated-trading
layer: strategy development, risk control, market regime context, memory, and
top-k ensemble selection. It intentionally avoids debate/LLM agents so results
remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from multiagents_trading_assistant.quantagents_backtest.backtest_engine import (
    BacktestConfig,
    backtest_ensemble,
    backtest_strategy,
)
from multiagents_trading_assistant.quantagents_backtest.indicators import add_indicators
from multiagents_trading_assistant.quantagents_backtest.metrics import (
    compute_metrics,
    score_metrics,
    summarize_equity,
)
from multiagents_trading_assistant.quantagents_backtest.strategy_generator import (
    Strategy,
    generate_strategy_pool,
)
from multiagents_trading_assistant.quantagents_backtest.vn_portfolio_engine import (
    VNPortfolioConfig,
    backtest_vn_portfolio,
)


@dataclass(frozen=True)
class RiskGateConfig:
    """Risk-control thresholds inspired by QA's risk analyst."""

    min_train_trades: int = 3
    max_train_drawdown: float = -0.35
    min_train_sharpe: float = -0.25
    max_score_std: float = 1.25
    min_symbols_tested: int = 5
    drawdown_penalty_weight: float = 0.35
    instability_penalty_weight: float = 0.35
    concentration_penalty_weight: float = 0.03


@dataclass(frozen=True)
class VNQuantAgentsConfig:
    start: str = "2021-01-01"
    end: str = "2026-05-03"
    initial_capital: float = 100_000.0
    n_strategies: int = 120
    seed: int = 43
    top_k: int = 7
    train_bars: int = 504
    test_bars: int = 126
    step_bars: int = 63
    regime_ma_fast: int = 50
    regime_ma_slow: int = 200
    regime_risk_off_scale: float = 0.5
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk_gate: RiskGateConfig = field(default_factory=RiskGateConfig)
    portfolio: VNPortfolioConfig = field(default_factory=VNPortfolioConfig)


@dataclass
class StrategyMemory:
    """Aggregated simulated-trading memory for one strategy."""

    strategy_id: str
    family: str
    logic: str
    conditions: str
    risk: dict
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
            "logic": self.logic,
            "conditions": self.conditions,
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


def run_vn_quantagents(
    universe_data: dict[str, pd.DataFrame],
    market_index: pd.DataFrame | None = None,
    config: VNQuantAgentsConfig | None = None,
    strategies: list[Strategy] | None = None,
) -> dict:
    """Run Vietnam QuantAgents-style simulated strategy development.

    Args:
        universe_data: ``symbol -> OHLCV`` with DatetimeIndex.
        market_index: optional VNINDEX/VN30INDEX OHLCV for regime scaling.
        config: workflow parameters.
        strategies: optional pre-generated strategy pool.
    """

    cfg = config or VNQuantAgentsConfig()
    pool = strategies or generate_strategy_pool(cfg.n_strategies, cfg.seed)
    backtest_cfg = BacktestConfig(
        initial_capital=cfg.initial_capital,
        periods_per_year=cfg.backtest.periods_per_year,
        commission_rate=cfg.backtest.commission_rate,
        slippage_rate=cfg.backtest.slippage_rate,
    )
    regime = compute_market_regime(market_index, cfg) if market_index is not None else None
    memory = _init_memory(pool)
    symbol_reports: list[dict] = []
    fold_reports: list[dict] = []
    best_fold_curves: dict[str, list[pd.Series]] = {}
    ensemble_fold_curves: dict[str, list[pd.Series]] = {}
    execution_schedule: dict[int, dict] = {}

    for symbol, raw_frame in universe_data.items():
        if raw_frame.empty:
            continue
        data = add_indicators(raw_frame)
        symbol_regime = _align_regime(regime, data.index)
        folds = _make_folds(data, cfg.train_bars, cfg.test_bars, cfg.step_bars)
        if not folds:
            continue

        symbol_best_rows = []
        for fold_id, (train, test) in enumerate(folds, start=1):
            fold_regime = _align_regime(symbol_regime, test.index)
            ranked = _rank_train_window(train, pool, cfg, backtest_cfg, memory)
            selected = _select_with_risk_gate(ranked, cfg, memory)
            if not selected:
                selected = [row["strategy"] for row in ranked[: cfg.top_k]]
            selected = selected[: cfg.top_k]
            test_results = [backtest_strategy(test, strategy, backtest_cfg) for strategy in selected]
            ensemble = backtest_ensemble(test, selected, backtest_cfg)
            ensemble_curve = apply_regime_scaling(ensemble.equity_curve, fold_regime, cfg)
            ensemble_metrics = compute_metrics(ensemble_curve, ensemble.trades, backtest_cfg.periods_per_year)
            if selected:
                best_curve = apply_regime_scaling(test_results[0].equity_curve, fold_regime, cfg)
                best_fold_curves.setdefault(symbol, []).append(best_curve)
            ensemble_fold_curves.setdefault(symbol, []).append(ensemble_curve)
            execution_schedule.setdefault(
                fold_id,
                {
                    "test_start": test.index[0],
                    "test_end": test.index[-1],
                    "market_regime": fold_regime,
                    "strategy_map": {},
                    "window_universe": {},
                },
            )
            execution_schedule[fold_id]["strategy_map"][symbol] = selected
            execution_schedule[fold_id]["window_universe"][symbol] = test.copy()

            for result in test_results:
                strategy = result.strategy
                if strategy is None:
                    continue
                scaled_curve = apply_regime_scaling(result.equity_curve, fold_regime, cfg)
                metrics = compute_metrics(scaled_curve, result.trades, backtest_cfg.periods_per_year)
                score = score_metrics(metrics)
                memory[strategy.strategy_id].record_test(symbol, score, metrics)
                symbol_best_rows.append(
                    {
                        "symbol": symbol,
                        "fold": fold_id,
                        "strategy_id": strategy.strategy_id,
                        "score": score,
                        **metrics,
                    }
                )

            fold_reports.append(
                {
                    "symbol": symbol,
                    "fold": fold_id,
                    "test_start": test.index[0],
                    "test_end": test.index[-1],
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

    strategy_memory = pd.DataFrame([item.to_row(cfg) for item in memory.values()])
    strategy_memory = strategy_memory.sort_values("qa_score", ascending=False).reset_index(drop=True)
    top_strategies = _strategies_from_ids(pool, strategy_memory.head(cfg.top_k)["strategy_id"].tolist())
    portfolio = build_oos_research_portfolio(best_fold_curves, ensemble_fold_curves, cfg, backtest_cfg)
    execution_portfolio = run_oos_execution_portfolio(execution_schedule, cfg)
    portfolio_summary = pd.DataFrame([{"portfolio": "qa_execution_schedule", **execution_portfolio["metrics"]}])

    return {
        "strategy_memory": strategy_memory,
        "symbol_reports": pd.DataFrame(symbol_reports),
        "fold_reports": pd.DataFrame(fold_reports),
        "top_strategies": top_strategies,
        "portfolio_summary": portfolio_summary,
        "research_summary": portfolio["research_summary"],
        "best_strategy_equity": portfolio["best_strategy_equity"],
        "ensemble_equity": portfolio["ensemble_equity"],
        "execution_portfolio": execution_portfolio,
        "regime": regime,
    }


def compute_market_regime(market_index: pd.DataFrame, cfg: VNQuantAgentsConfig) -> pd.Series:
    """Classify VN market regime from index trend and drawdown."""

    data = add_indicators(market_index)
    close = data["close"]
    fast = close.rolling(cfg.regime_ma_fast, min_periods=cfg.regime_ma_fast).mean()
    slow = close.rolling(cfg.regime_ma_slow, min_periods=cfg.regime_ma_slow).mean()
    drawdown_63 = close / close.rolling(63, min_periods=20).max() - 1.0
    ret_20 = close.pct_change(20)
    regime = pd.Series("NEUTRAL", index=data.index)
    regime[(close > slow) & (fast > slow) & (ret_20 > 0)] = "RISK_ON"
    regime[(close < slow) | (drawdown_63 < -0.12)] = "RISK_OFF"
    regime[(drawdown_63 < -0.20)] = "CRASH"
    return regime


def apply_regime_scaling(
    equity_curve: pd.Series,
    regime: pd.Series | None,
    cfg: VNQuantAgentsConfig,
) -> pd.Series:
    """Scale daily returns in risk-off regimes to mimic QA risk alerts."""

    if regime is None or equity_curve.empty:
        return equity_curve
    aligned = regime.reindex(equity_curve.index).ffill().fillna("NEUTRAL")
    returns = equity_curve.pct_change().fillna(0.0)
    scale = pd.Series(1.0, index=equity_curve.index)
    scale[aligned == "RISK_OFF"] = cfg.regime_risk_off_scale
    scale[aligned == "CRASH"] = 0.0
    scaled = (1.0 + returns * scale).cumprod() * float(equity_curve.iloc[0])
    scaled.name = equity_curve.name
    return scaled


def build_oos_research_portfolio(
    best_fold_curves: dict[str, list[pd.Series]],
    ensemble_fold_curves: dict[str, list[pd.Series]],
    cfg: VNQuantAgentsConfig,
    backtest_cfg: BacktestConfig,
) -> dict:
    best_curves = {symbol: _stitch_fold_curves(curves) for symbol, curves in best_fold_curves.items() if curves}
    ensemble_curves = {
        symbol: _stitch_fold_curves(curves) for symbol, curves in ensemble_fold_curves.items() if curves
    }
    best_equity = _equal_weight_portfolio(best_curves, cfg.initial_capital, "qa_best_strategy_oos")
    ensemble_equity = _equal_weight_portfolio(ensemble_curves, cfg.initial_capital, "qa_ensemble_oos")
    summary = pd.DataFrame(
        [
            _curve_only_summary(best_equity, "qa_best_strategy_oos", backtest_cfg.periods_per_year),
            _curve_only_summary(ensemble_equity, "qa_ensemble_oos", backtest_cfg.periods_per_year),
        ]
    )
    return {
        "research_summary": summary,
        "best_strategy_equity": best_equity,
        "ensemble_equity": ensemble_equity,
    }


def run_oos_execution_portfolio(execution_schedule: dict[int, dict], cfg: VNQuantAgentsConfig) -> dict:
    """Execute fold-selected strategies only in their future test windows."""

    fold_ids = sorted(execution_schedule)
    if not fold_ids:
        empty_curve = pd.Series(dtype=float, name="vn_quantagents_portfolio_oos")
        empty_frame = pd.DataFrame(columns=["equity", "cash", "positions", "regime", "gross_exposure"])
        return {
            "equity_curve": empty_curve,
            "equity_frame": empty_frame,
            "trades": [],
            "metrics": compute_metrics(pd.Series([cfg.portfolio.initial_capital]), []),
            "open_positions": {},
        }

    current_capital = cfg.portfolio.initial_capital
    equity_frames: list[pd.DataFrame] = []
    trades: list[dict] = []
    open_positions = {}

    for fold_id in fold_ids:
        fold = execution_schedule[fold_id]
        fold_cfg = VNPortfolioConfig(
            initial_capital=current_capital,
            max_positions=cfg.portfolio.max_positions,
            min_position_value=cfg.portfolio.min_position_value,
            periods_per_year=cfg.portfolio.periods_per_year,
            risk_on_exposure=cfg.portfolio.risk_on_exposure,
            neutral_exposure=cfg.portfolio.neutral_exposure,
            risk_off_exposure=cfg.portfolio.risk_off_exposure,
            crash_exposure=cfg.portfolio.crash_exposure,
            allow_new_in_risk_off=cfg.portfolio.allow_new_in_risk_off,
            mean_reversion_requires_trend=cfg.portfolio.mean_reversion_requires_trend,
            costs=cfg.portfolio.costs,
        )
        result = backtest_vn_portfolio(
            fold["window_universe"],
            [],
            market_regime=fold["market_regime"],
            config=fold_cfg,
            strategy_map=fold["strategy_map"],
        )
        fold_equity = result["equity_frame"].copy()
        if equity_frames:
            fold_equity = fold_equity.loc[~fold_equity.index.isin(equity_frames[-1].index)]
        equity_frames.append(fold_equity)
        trades.extend(result["trades"])
        current_capital = float(result["equity_curve"].iloc[-1]) if not result["equity_curve"].empty else current_capital
        open_positions = result["open_positions"]

    equity_frame = pd.concat(equity_frames).sort_index() if equity_frames else pd.DataFrame()
    equity_curve = equity_frame["equity"].rename("vn_quantagents_portfolio_oos") if not equity_frame.empty else pd.Series(dtype=float)
    metrics = compute_metrics(equity_curve, trades, cfg.portfolio.periods_per_year) if not equity_curve.empty else {}
    return {
        "equity_curve": equity_curve,
        "equity_frame": equity_frame,
        "trades": trades,
        "metrics": metrics,
        "open_positions": open_positions,
    }


def save_vn_quantagents_result(result: dict, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result["strategy_memory"].to_csv(out / "strategy_memory.csv", index=False)
    result["symbol_reports"].to_csv(out / "symbol_reports.csv", index=False)
    result["fold_reports"].to_csv(out / "fold_reports.csv", index=False)
    result["portfolio_summary"].to_csv(out / "portfolio_summary.csv", index=False)
    if result.get("research_summary") is not None:
        result["research_summary"].to_csv(out / "research_summary.csv", index=False)
    summarize_equity(result["best_strategy_equity"]).to_csv(out / "equity_best_strategy.csv")
    summarize_equity(result["ensemble_equity"]).to_csv(out / "equity_ensemble.csv")
    if result.get("execution_portfolio") is not None:
        execution = result["execution_portfolio"]
        execution["equity_frame"].to_csv(out / "execution_equity_frame.csv")
        pd.DataFrame(execution["trades"]).to_csv(out / "execution_trades.csv", index=False)
        pd.DataFrame([execution["metrics"]]).to_csv(out / "execution_metrics.csv", index=False)
    if result.get("regime") is not None:
        result["regime"].to_frame("regime").to_csv(out / "market_regime.csv")
    return out


def _rank_train_window(
    train: pd.DataFrame,
    strategies: list[Strategy],
    cfg: VNQuantAgentsConfig,
    backtest_cfg: BacktestConfig,
    memory: dict[str, StrategyMemory],
) -> list[dict]:
    ranked = []
    for strategy in strategies:
        result = backtest_strategy(train, strategy, backtest_cfg)
        metrics = result.metrics
        base_score = score_metrics(metrics)
        memory[strategy.strategy_id].record_train(base_score)
        unstable = _std(memory[strategy.strategy_id].train_scores)
        risk_reject = (
            metrics["number_of_trades"] < cfg.risk_gate.min_train_trades
            or metrics["max_drawdown"] < cfg.risk_gate.max_train_drawdown
            or metrics["sharpe_ratio"] < cfg.risk_gate.min_train_sharpe
            or unstable > cfg.risk_gate.max_score_std
        )
        adjusted_score = (
            base_score
            - cfg.risk_gate.instability_penalty_weight * unstable
            - cfg.risk_gate.drawdown_penalty_weight * abs(min(metrics["max_drawdown"], 0.0))
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


def _select_with_risk_gate(
    ranked: list[dict],
    cfg: VNQuantAgentsConfig,
    memory: dict[str, StrategyMemory],
) -> list[Strategy]:
    selected = []
    for row in ranked:
        strategy = row["strategy"]
        if row["risk_reject"]:
            memory[strategy.strategy_id].risk_rejections += 1
            continue
        selected.append(strategy)
        if len(selected) >= cfg.top_k:
            break
    return selected


def _init_memory(strategies: list[Strategy]) -> dict[str, StrategyMemory]:
    memory = {}
    for strategy in strategies:
        desc = strategy.describe()
        memory[strategy.strategy_id] = StrategyMemory(
            strategy_id=strategy.strategy_id,
            family=strategy.family,
            logic=strategy.logic,
            conditions=" | ".join(desc["conditions"]),
            risk=desc["risk"],
        )
    return memory


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


def _align_regime(regime: pd.Series | None, index: pd.Index) -> pd.Series | None:
    if regime is None:
        return None
    return regime.reindex(index).ffill().bfill()


def _dominant_regime(regime: pd.Series | None) -> str:
    if regime is None or regime.empty:
        return "UNKNOWN"
    return str(regime.value_counts().idxmax())


def _equal_weight_portfolio(
    curves: dict[str, pd.Series],
    initial_capital: float,
    name: str,
) -> pd.Series:
    normalized = []
    for curve in curves.values():
        clean = curve.dropna()
        if clean.empty:
            continue
        normalized.append(clean / float(clean.iloc[0]))
    if not normalized:
        return pd.Series(dtype=float, name=name)
    frame = pd.concat(normalized, axis=1).sort_index().ffill().dropna(how="all")
    portfolio = frame.mean(axis=1) * initial_capital
    portfolio.name = name
    return portfolio


def _stitch_fold_curves(curves: list[pd.Series]) -> pd.Series:
    stitched = pd.concat(curves).sort_index()
    stitched = stitched[~stitched.index.duplicated(keep="last")]
    return stitched


def _curve_only_summary(equity_curve: pd.Series, portfolio: str, periods_per_year: int) -> dict:
    if equity_curve.empty:
        return {
            "portfolio": portfolio,
            "metric_scope": "research_curve_only",
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": None,
            "number_of_trades": None,
        }
    metrics = compute_metrics(equity_curve, [], periods_per_year)
    metrics["win_rate"] = None
    metrics["number_of_trades"] = None
    return {"portfolio": portfolio, "metric_scope": "research_curve_only", **metrics}


def _strategies_from_ids(strategies: list[Strategy], ids: list[str]) -> list[Strategy]:
    by_id = {strategy.strategy_id: strategy for strategy in strategies}
    return [by_id[strategy_id] for strategy_id in ids if strategy_id in by_id]


def _mean(values: list[float]) -> float:
    clean = [value for value in values if np.isfinite(value)]
    return float(np.mean(clean)) if clean else 0.0


def _std(values: list[float]) -> float:
    clean = [value for value in values if np.isfinite(value)]
    return float(np.std(clean, ddof=0)) if len(clean) > 1 else 0.0
