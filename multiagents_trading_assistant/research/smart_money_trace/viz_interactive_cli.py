"""
Smart Money Trace Interactive Visualization CLI

Usage:
    python -m multiagents_trading_assistant.research.smart_money_trace.viz_interactive_cli
    python -m multiagents_trading_assistant.research.smart_money_trace.viz_interactive_cli --symbol VCB
    python -m multiagents_trading_assistant.research.smart_money_trace.viz_interactive_cli --symbols VCB,FPT,VNM
    python -m multiagents_trading_assistant.research.smart_money_trace.viz_interactive_cli --universe vn30
    python -m multiagents_trading_assistant.research.smart_money_trace.viz_interactive_cli --data-dir data/research/smart_money_trace
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from multiagents_trading_assistant.research.smart_money_trace.visualization_interactive import run_interactive_charts

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _resolve_symbols(args) -> list[str] | None:
    """Resolve final symbol list. Priority: --universe > --symbols > --symbol > None."""
    if args.universe:
        from multiagents_trading_assistant.fetcher import (
            get_vn30_symbols,
            get_vn100_symbols,
            get_liquid_symbols,
        )
        u = args.universe.lower()
        if u == "vn30":
            return get_vn30_symbols()
        if u == "vn100":
            return get_vn100_symbols()
        if u == "liquid":
            return get_liquid_symbols(min_avg_vol=500_000)
        raise ValueError(f"Unknown universe: {args.universe}")

    if args.symbols:
        return [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if args.symbol:
        return [args.symbol.upper()]

    return None  # Auto-detect from data


def main():
    parser = argparse.ArgumentParser(
        description="Generate Smart Money Trace interactive HTML charts (Plotly with zoom, pan, hover)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/research/smart_money_trace",
        help="Path to smart_money_trace parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/research/smart_money_trace/interactive",
        help="Output directory for HTML files",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Single symbol for symbol dashboard (default: auto-detect top 30)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols, e.g. 'VCB,FPT,VNM' (overrides --symbol)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        default=None,
        choices=["vn30", "vn100", "liquid"],
        help="Render all symbols in a market group (overrides --symbols and --symbol)",
    )
    parser.add_argument(
        "--skip-market",
        action="store_true",
        help="Skip rendering the market dashboard (only per-symbol dashboards)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    required_files = [
        "smart_money_daily_summary.parquet",
        "smart_money_by_symbol.parquet",
    ]
    missing = [f for f in required_files if not (data_dir / f).exists()]
    if missing:
        logger.error(f"Missing files: {missing}")
        sys.exit(1)

    try:
        symbols = _resolve_symbols(args)

        if symbols:
            logger.info(f"Resolved {len(symbols)} symbol(s): {', '.join(symbols[:10])}"
                       + (" ..." if len(symbols) > 10 else ""))
        else:
            logger.info("No symbols specified — auto-detecting top 30 from data")

        run_interactive_charts(
            data_dir, output_dir,
            symbols=symbols,
            skip_market=args.skip_market,
        )

        print("\n" + "=" * 70)
        print("INTERACTIVE CHARTS GENERATED (Smart Money Trace)")
        print("=" * 70)
        print(f"Output directory: {output_dir}")
        print(f"\nGenerated interactive charts (HTML):")
        if not args.skip_market:
            print(f"  - 01_market_dashboard_interactive.html")
        if symbols:
            print(f"  - {len(symbols)} symbol dashboard(s):")
            for s in symbols[:10]:
                print(f"      02_symbol_dashboard_{s}_interactive.html")
            if len(symbols) > 10:
                print(f"      ... and {len(symbols) - 10} more")
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
