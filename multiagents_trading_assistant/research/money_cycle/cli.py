"""
Money Cycle Research CLI.

Usage:
    python -m multiagents_trading_assistant.research.money_cycle.cli
    python -m multiagents_trading_assistant.research.money_cycle.cli \
        --input multiagents_trading_assistant/data/ohlcv_master.parquet \
        --output data/research/money_cycle \
        --windows 3,5,10,20,50,100,200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from multiagents_trading_assistant.data import ohlcv_store
from multiagents_trading_assistant.research.money_cycle.compute_money_cycle import (
    run_money_cycle_research,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Money Cycle Research - Compute CHDM and DS indicators from OHLCV data."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(ohlcv_store.STORE_PATH),
        help=f"Input OHLCV parquet/csv file (default: {ohlcv_store.STORE_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/research/money_cycle",
        help="Output directory for results (default: data/research/money_cycle)",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="3,5,10,20,50,100,200",
        help="Comma-separated list of windows (default: 3,5,10,20,50,100,200)",
    )

    args = parser.parse_args()

    # Parse windows
    try:
        windows = [int(w.strip()) for w in args.windows.split(",")]
    except ValueError:
        logger.error(f"Invalid windows format: {args.windows}")
        sys.exit(1)

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        stats = run_money_cycle_research(
            input_path=input_path,
            output_dir=args.output,
            windows=windows,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("MONEY CYCLE RESEARCH COMPLETE")
        print("=" * 60)
        print(f"Input file:      {stats['input_path']}")
        print(f"Output dir:      {stats['output_dir']}")
        print(f"Windows:         {stats['windows']}")
        print(f"Input rows:      {stats['n_rows_input']:,}")
        print(f"Symbols:         {stats['n_symbols']}")
        print(f"Date range:      {stats['date_min']} to {stats['date_max']}")
        print(f"\nOutput files:")
        print(f"  chdm_market.parquet:       {stats['n_rows_chdm_market']} dates")
        print(f"  ds_market.parquet:         {stats['n_rows_ds_market']} dates")
        print(f"  money_cycle_market.parquet: {stats['n_rows_mc_market']} dates")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
