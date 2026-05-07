"""CLI for Smart Money Trace research module.

Usage:
    python -m multiagents_trading_assistant.research.smart_money_trace.cli
    python -m multiagents_trading_assistant.research.smart_money_trace.cli \\
        --input data/processed/OHLCV.parquet \\
        --output data/research/smart_money_trace
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from multiagents_trading_assistant.research.smart_money_trace.compute_smart_money_trace import (
    run_smart_money_trace,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_FALLBACK_INPUT = Path("multiagents_trading_assistant/data/ohlcv_master.parquet")
_DEFAULT_OUTPUT = Path("data/research/smart_money_trace")
_DEFAULT_CHDM = Path("data/research/money_cycle/chdm_by_symbol.parquet")


def _resolve_input(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    if _FALLBACK_INPUT.exists():
        logger.warning("Input '%s' not found. Falling back to '%s'.", raw, _FALLBACK_INPUT)
        return _FALLBACK_INPUT
    raise FileNotFoundError(
        f"Input file not found: {raw}\n"
        f"Also tried fallback: {_FALLBACK_INPUT}\n"
        "Please specify a valid --input path."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smart Money Trace - detect smart money footprints from OHLCV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m multiagents_trading_assistant.research.smart_money_trace.cli\n"
            "  python -m multiagents_trading_assistant.research.smart_money_trace.cli"
            " --input data/processed/OHLCV.parquet --output data/research/smart_money_trace\n"
            "  python -m multiagents_trading_assistant.research.smart_money_trace.cli"
            " --input multiagents_trading_assistant/data/ohlcv_master.parquet\n"
        ),
    )
    parser.add_argument(
        "--input",
        default=str(_FALLBACK_INPUT),
        help="Path to OHLCV parquet/csv (default: ohlcv_master.parquet)",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help="Output directory (default: data/research/smart_money_trace)",
    )
    parser.add_argument(
        "--chdm",
        default=None,
        help="Path to chdm_by_symbol.parquet (auto-detect if omitted)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    try:
        input_path = _resolve_input(args.input)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    chdm_path = args.chdm

    print()
    print("=" * 60)
    print("  Smart Money Trace")
    print("=" * 60)
    print("  Input  :", input_path)
    print("  Output :", args.output)
    if chdm_path:
        print("  CHDM   :", chdm_path)
    else:
        print("  CHDM   : auto-detect (", _DEFAULT_CHDM, ")")
    print()

    stats = run_smart_money_trace(
        input_path=input_path,
        output_dir=args.output,
        chdm_path=chdm_path,
    )

    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  OHLCV rows     : {stats['n_rows_input']:,}")
    print(f"  Symbols        : {stats['n_symbols']:,}")
    print(f"  Date range     : {stats['date_min']} to {stats['date_max']}")
    print(f"  CHDM from file : {stats['chdm_loaded_from_file']}")
    print(f"  Output rows    : {stats['n_rows_output']:,}")
    print()
    print("  Smart Money States:")
    state_counts = stats.get("state_counts", {})
    total_classified = sum(state_counts.values())
    for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
        pct = count / total_classified * 100 if total_classified else 0
        print(f"    {state:<28} {count:>6,}  ({pct:.1f}%)")
    print()
    print("  Output files:")
    out_dir = Path(args.output)
    for fname in [
        "smart_money_by_symbol.parquet",
        "smart_money_daily_summary.parquet",
        "smart_money_hot_states.parquet",
    ]:
        fpath = out_dir / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"    [OK] {fname}  ({size_kb:.0f} KB)")
        else:
            print(f"    [!!] {fname}  (MISSING)")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
