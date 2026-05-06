"""
Money Cycle Visualization CLI

Usage:
    python -m multiagents_trading_assistant.research.money_cycle.viz_cli
    python -m multiagents_trading_assistant.research.money_cycle.viz_cli --symbol FPT
    python -m multiagents_trading_assistant.research.money_cycle.viz_cli --data-dir data/research/money_cycle
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from multiagents_trading_assistant.research.money_cycle.visualization import run_all_charts

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Money Cycle visualization charts (4 types)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/research/money_cycle",
        help="Path to money_cycle parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/research/money_cycle/charts",
        help="Output directory for PNG files",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="VCB",
        help="Symbol for symbol dashboard (default: VCB)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

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
        run_all_charts(data_dir, output_dir, symbol=args.symbol)
        print("\n" + "=" * 70)
        print("VISUALIZATION COMPLETE")
        print("=" * 70)
        print(f"Output directory: {output_dir}")
        print(f"\nGenerated charts:")
        print(f"  1. 01_market_dashboard.png")
        print(f"  2. 02_chdm_heatmap_020.png")
        print(f"  2. 02_chdm_heatmap_050.png")
        print(f"  2. 02_chdm_heatmap_200.png")
        print(f"  3. 03_symbol_dashboard_{args.symbol}.png")
        print(f"  4. 04_scatter_chdm50_vs_return_T10.png")
        print(f"  4. 04_scatter_chdm200_vs_return_T20.png")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
