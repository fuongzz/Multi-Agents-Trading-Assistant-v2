"""
Money Cycle Interactive Visualization CLI

Usage:
    python -m multiagents_trading_assistant.research.money_cycle.viz_interactive_cli
    python -m multiagents_trading_assistant.research.money_cycle.viz_interactive_cli --symbol FPT
    python -m multiagents_trading_assistant.research.money_cycle.viz_interactive_cli --data-dir data/research/money_cycle
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from multiagents_trading_assistant.research.money_cycle.visualization_interactive import run_interactive_charts

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Money Cycle interactive HTML charts (Plotly with zoom, pan, hover)."
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
        default="data/research/money_cycle/interactive",
        help="Output directory for HTML files",
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
        run_interactive_charts(data_dir, output_dir, symbol=args.symbol)
        print("\n" + "=" * 70)
        print("INTERACTIVE CHARTS GENERATED")
        print("=" * 70)
        print(f"Output directory: {output_dir}")
        print(f"\nGenerated interactive charts (HTML):")
        print(f"  1. 01_market_dashboard_interactive.html")
        print(f"  2. 02_symbol_dashboard_{args.symbol}_interactive.html")
        print(f"\nOpen in browser to interact:")
        print(f"  - Zoom: scroll wheel")
        print(f"  - Pan: click and drag")
        print(f"  - Hover: see values")
        print(f"  - Legend toggle: click line name")
        print(f"  - Download PNG: camera icon in toolbar")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
