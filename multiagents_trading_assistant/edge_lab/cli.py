"""CLI for the edge discovery lab."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from multiagents_trading_assistant.edge_lab.backtest import (
    PortfolioConfig,
    run_portfolio,
    run_regime_switching_portfolio,
    run_signal_quality,
)
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import load_hypotheses
from multiagents_trading_assistant.edge_lab.metrics import benchmark_metrics
from multiagents_trading_assistant.edge_lab.metrics import equity_metrics, trade_metrics
from multiagents_trading_assistant.edge_lab.report import build_summary, write_outputs
from multiagents_trading_assistant.edge_lab.universe import get_universe


DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "vn30_money_smt_hypotheses.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run edge discovery research")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Hypothesis JSON config")
    parser.add_argument("--universe", default="VN30")
    parser.add_argument("--from", dest="from_date", default="2022-05-08")
    parser.add_argument("--to", dest="to_date", default="2026-05-08")
    parser.add_argument("--output-dir", default="backtest_results/edge_lab_vn30_money_smt")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-positions", type=int, default=8)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--holds", default="10,20,30,60", help="Comma-separated signal-quality holds")
    args = parser.parse_args()

    universe = get_universe(args.universe)
    hypotheses = load_hypotheses(args.config)
    holds = [int(item.strip()) for item in args.holds.split(",") if item.strip()]

    print("Building feature table...")
    features, price_map = build_feature_table(universe, args.from_date, args.to_date)
    print(f"Features: {len(features):,} rows | symbols: {len(price_map)}")

    print("Running signal-quality backtest...")
    signal_trades = run_signal_quality(
        features,
        price_map,
        hypotheses,
        args.from_date,
        args.to_date,
        holds=holds,
        top_n=args.top_n,
    )

    print("Running portfolio backtests...")
    portfolio_cfg = PortfolioConfig(
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        top_n=args.top_n,
    )
    portfolio_runs = []
    for hypothesis in hypotheses:
        trades, run_equity = run_portfolio(
            features,
            price_map,
            [hypothesis],
            args.from_date,
            args.to_date,
            config=portfolio_cfg,
        )
        trades["portfolio_run"] = hypothesis.name
        run_equity["portfolio_run"] = hypothesis.name
        portfolio_runs.append((hypothesis.name, trades, run_equity))

    combined_trades, combined_equity = run_portfolio(
        features,
        price_map,
        hypotheses,
        args.from_date,
        args.to_date,
        config=portfolio_cfg,
    )
    combined_trades["portfolio_run"] = "combined"
    combined_equity["portfolio_run"] = "combined"
    portfolio_runs.append(("combined", combined_trades, combined_equity))

    regime_trades, regime_equity = run_regime_switching_portfolio(
        features,
        price_map,
        hypotheses,
        args.from_date,
        args.to_date,
        config=portfolio_cfg,
    )
    regime_trades["portfolio_run"] = "regime_switching"
    regime_equity["portfolio_run"] = "regime_switching"
    portfolio_runs.append(("regime_switching", regime_trades, regime_equity))

    portfolio_trades = pd.concat([item[1] for item in portfolio_runs], ignore_index=True)
    equity = pd.concat([item[2] for item in portfolio_runs], ignore_index=True)

    benchmark = benchmark_metrics(price_map, args.from_date, args.to_date)
    summary = build_summary(signal_trades, pd.DataFrame(), pd.DataFrame(), args.initial_capital, benchmark)
    portfolio_rows = []
    for run_name, trades, run_equity in portfolio_runs:
        portfolio_rows.append(
            {
                "scope": "portfolio",
                "hypothesis": run_name,
                "hold": "",
                **trade_metrics(trades),
                **equity_metrics(run_equity, args.initial_capital),
                **benchmark,
            }
        )
    summary = pd.concat([summary, pd.DataFrame(portfolio_rows)], ignore_index=True)
    write_outputs(args.output_dir, signal_trades, portfolio_trades, equity, summary)
    print(summary.round(4).to_string(index=False))
    print(f"\nOutputs: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
