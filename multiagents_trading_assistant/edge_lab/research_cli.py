"""Robustness and grid-search CLI for edge lab hypotheses."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from multiagents_trading_assistant.edge_lab.backtest import PortfolioConfig
from multiagents_trading_assistant.edge_lab.cli import DEFAULT_CONFIG
from multiagents_trading_assistant.edge_lab.experiments import (
    default_periods,
    run_walk_forward_grid,
    run_subperiod_robustness,
    run_risk_grid,
    run_threshold_grid,
    write_experiment,
)
from multiagents_trading_assistant.edge_lab.hypothesis import load_hypotheses


def _load_grid_arg(value: str) -> dict:
    """Load JSON grid args, tolerating PowerShell-stripped object keys."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        fixed = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', value)
        return json.loads(fixed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run edge lab robustness and grid experiments")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--universe", default="VN30")
    parser.add_argument("--from", dest="from_date", default="2022-05-08")
    parser.add_argument("--to", dest="to_date", default="2026-05-08")
    parser.add_argument("--output-dir", default="backtest_results/edge_lab_experiments")
    parser.add_argument("--hypothesis", default="leader_pullback_market_healthy")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward parameter selection")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument("--min-train-trades", type=int, default=20)
    parser.add_argument(
        "--objective",
        choices=["sharpe", "return", "profit_factor", "calmar"],
        default="sharpe",
        help="Walk-forward training objective",
    )
    parser.add_argument(
        "--grid",
        default='{"smart_money_score":[62,65,68,70],"mkt_DS20":[0.45,0.50],"mkt_CHDM20":[45,50]}',
        help="JSON object mapping filter column to values",
    )
    parser.add_argument(
        "--risk-grid",
        default='{"stop_loss":[0.06,0.08,0.10],"take_profit":[0.20,0.25,0.30],"max_holding_bars":[45,60,90]}',
        help="JSON object mapping risk key to values",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hypotheses = load_hypotheses(args.config)
    cfg = PortfolioConfig()

    print("Running subperiod robustness...")
    robustness = run_subperiod_robustness(hypotheses, args.universe, default_periods(), cfg)
    robustness.to_csv(out_dir / "robustness.csv", index=False)

    target = next((item for item in hypotheses if item.name == args.hypothesis), None)
    if target is None:
        raise ValueError(f"Hypothesis not found: {args.hypothesis}")

    print("Running threshold grid...")
    grid = _load_grid_arg(args.grid)
    grid_result = run_threshold_grid(target, grid, args.universe, args.from_date, args.to_date, cfg)
    grid_result.to_csv(out_dir / "grid_search.csv", index=False)

    print("Running risk grid...")
    risk_grid = _load_grid_arg(args.risk_grid)
    risk_result = run_risk_grid(target, risk_grid, args.universe, args.from_date, args.to_date, cfg)
    risk_result.to_csv(out_dir / "risk_grid.csv", index=False)

    frames = {
        "robustness": robustness,
        "grid_search": grid_result,
        "risk_grid": risk_result,
    }

    if args.walk_forward:
        print("Running walk-forward validation...")
        walk_selected, walk_oos = run_walk_forward_grid(
            target,
            grid,
            risk_grid,
            args.universe,
            args.from_date,
            args.to_date,
            cfg,
            train_months=args.train_months,
            test_months=args.test_months,
            step_months=args.step_months,
            min_train_trades=args.min_train_trades,
            objective=args.objective,
        )
        walk_selected.to_csv(out_dir / "walk_forward_selection.csv", index=False)
        walk_oos.to_csv(out_dir / "walk_forward_oos.csv", index=False)
        frames["walk_forward_selection"] = walk_selected
        frames["walk_forward_oos"] = walk_oos

    write_experiment(out_dir / "edge_lab_experiments.xlsx", frames)

    print("\nRobustness:")
    print(robustness.round(4).to_string(index=False))
    print("\nGrid search top 20:")
    print(grid_result.head(20).round(4).to_string(index=False))
    print("\nRisk grid top 20:")
    print(risk_result.head(20).round(4).to_string(index=False))
    if args.walk_forward:
        print("\nWalk-forward OOS:")
        print(walk_oos.round(4).to_string(index=False))
    print(f"\nOutputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
