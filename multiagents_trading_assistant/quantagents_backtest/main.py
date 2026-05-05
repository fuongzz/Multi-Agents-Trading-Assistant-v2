"""CLI-style entrypoint for the QuantAgents-inspired backtesting pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from multiagents_trading_assistant.quantagents_backtest.metrics import summarize_equity
from multiagents_trading_assistant.quantagents_backtest.strategy_generator import generate_strategy_pool
from multiagents_trading_assistant.quantagents_backtest.walk_forward import (
    WalkForwardConfig,
    run_walk_forward,
)


def run_pipeline(
    ohlcv: pd.DataFrame,
    n_strategies: int = 120,
    seed: int = 42,
    top_k: int = 7,
    train_bars: int = 504,
    test_bars: int = 126,
    step_bars: int = 63,
) -> dict:
    strategies = generate_strategy_pool(n_strategies=n_strategies, seed=seed)
    config = WalkForwardConfig(
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        top_k=top_k,
    )
    return run_walk_forward(ohlcv, strategies, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QuantAgents-style walk-forward backtest")
    parser.add_argument("--csv", required=True, help="CSV with datetime index or date/time column and OHLCV columns")
    parser.add_argument("--output-dir", default="backtest_results/quantagents", help="Directory for output CSV files")
    parser.add_argument("--n-strategies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=7)
    args = parser.parse_args()

    ohlcv = _read_ohlcv_csv(Path(args.csv))
    result = run_pipeline(
        ohlcv,
        n_strategies=args.n_strategies,
        seed=args.seed,
        top_k=args.top_k,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = result["ranked_table"]
    best_equity = summarize_equity(result["best_strategy_result"].equity_curve)
    ensemble_equity = summarize_equity(result["ensemble_result"].equity_curve)

    ranked.to_csv(output_dir / "ranked_strategies.csv", index=False)
    best_equity.to_csv(output_dir / "best_strategy_equity.csv")
    ensemble_equity.to_csv(output_dir / "ensemble_equity.csv")

    print(ranked.head(args.top_k).to_string(index=False))
    print(f"\nBest strategy: {result['best_strategy'].strategy_id}")
    print(f"Outputs written to: {output_dir.resolve()}")


def _read_ohlcv_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((col for col in ("date", "time", "datetime") if col in df.columns), None)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        df.index = pd.to_datetime(df.index)
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing OHLCV columns: {missing}")
    return df[required].sort_index()


if __name__ == "__main__":
    main()
