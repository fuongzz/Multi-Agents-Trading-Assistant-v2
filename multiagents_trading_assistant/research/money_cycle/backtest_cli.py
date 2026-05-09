"""
Money Cycle Backtest CLI

Usage:
    python -m multiagents_trading_assistant.research.money_cycle.backtest_cli
    python -m multiagents_trading_assistant.research.money_cycle.backtest_cli --data-dir data/research/money_cycle
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from multiagents_trading_assistant.research.money_cycle.backtest_strategy import run_backtests

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run Money Cycle backtest on historical data."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/research/money_cycle",
        help="Path to money_cycle parquet files",
    )
    parser.add_argument(
        "--ohlcv-path",
        type=str,
        default=None,
        help="Path to OHLCV parquet (default: from ohlcv_store)",
    )
    parser.add_argument(
        "--ds-window",
        type=int,
        default=50,
        help="DS window for stock selection (default: 50)",
    )
    parser.add_argument(
        "--rebalance-days",
        type=int,
        default=None,
        help="Optional calendar-day spacing between stock-selection signal dates",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Check input directory
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    required_files = [
        "money_cycle_market.parquet",
        "chdm_by_symbol.parquet",
        "ds_by_symbol.parquet",
    ]
    missing = [f for f in required_files if not (data_dir / f).exists()]
    if missing:
        logger.error(f"Missing files: {missing}")
        sys.exit(1)

    try:
        run_backtests(
            data_dir,
            ohlcv_path=args.ohlcv_path,
            ds_window=args.ds_window,
            rebalance_days=args.rebalance_days,
        )
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
