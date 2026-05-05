"""main.py — CLI entry point cho AI Trading Assistant (dual pipeline).

Cách dùng:
  python -m multiagents_trading_assistant.main --pipeline invest
  python -m multiagents_trading_assistant.main --pipeline trade
  python -m multiagents_trading_assistant.main --pipeline all
  python -m multiagents_trading_assistant.main --pipeline invest --symbol VCB
  python -m multiagents_trading_assistant.main --pipeline trade  --symbol HPG
  python -m multiagents_trading_assistant.main --schedule          # APScheduler nền
"""

import argparse
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import importlib.metadata  # noqa: F401  (fix pandas-ta AttributeError Python 3.11)

_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _today() -> str:
    return datetime.now(tz=_VN_TZ).strftime("%Y-%m-%d")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Trading Assistant — Dual Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ (live pipeline):
  python -m multiagents_trading_assistant.main --pipeline invest
  python -m multiagents_trading_assistant.main --pipeline trade
  python -m multiagents_trading_assistant.main --pipeline all
  python -m multiagents_trading_assistant.main --pipeline invest --symbol VCB
  python -m multiagents_trading_assistant.main --pipeline trade  --symbol HPG
  python -m multiagents_trading_assistant.main --schedule

Ví dụ (backtest — không dùng LLM):
  python -m multiagents_trading_assistant.main --backtest --symbol VCB
  python -m multiagents_trading_assistant.main --backtest --symbol VCB --from 2024-01-01 --to 2024-12-31
  python -m multiagents_trading_assistant.main --backtest --universe vn30 --from 2023-01-01
  python -m multiagents_trading_assistant.main --backtest --universe liquid --setup BREAKOUT

Ví dụ (Bob — Strategy Development Meeting):
  python -m multiagents_trading_assistant.main --bob
  python -m multiagents_trading_assistant.main --bob --date 2024-12-31
        """,
    )
    parser.add_argument(
        "--pipeline",
        choices=["invest", "trade", "all"],
        default="all",
        help="Pipeline cần chạy: invest | trade | all (mặc định: all)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Chỉ phân tích 1 mã (VD: VCB). Nếu không có → dùng screener.",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Ngày phân tích YYYY-MM-DD (mặc định: hôm nay)",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Khởi APScheduler (invest Thứ 2 08:00, trade hàng ngày 08:30)",
    )

    # ── Backtest args (lazy-loaded qua cli.py) ──
    from multiagents_trading_assistant.backtest.cli import add_backtest_args
    add_backtest_args(parser)

    # ── Bob args ──
    parser.add_argument(
        "--bob",
        action="store_true",
        help="Chạy Bob's Strategy Development Meeting (simulated trading + ℳₛ update)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    date = args.date or _today()
    symbol = args.symbol.upper() if args.symbol else None

    # Lazy import để tránh circular và giữ import nhanh ở --help
    from multiagents_trading_assistant.orchestrator.pipeline_runner import (
        run_investment_pipeline,
        run_trade_pipeline,
        start_scheduler,
    )

    if args.schedule:
        start_scheduler()
        sys.exit(0)

    if getattr(args, "backtest", False):
        from multiagents_trading_assistant.backtest.cli import run_backtest_cli
        run_backtest_cli(args)
        sys.exit(0)

    if getattr(args, "bob", False):
        from multiagents_trading_assistant.bob.simulator import run_strategy_development_meeting
        active = run_strategy_development_meeting(to_date=date if args.date else None)
        print(f"\n[Bob] {len(active)} active strategies returned.")
        sys.exit(0)

    if args.pipeline in ("invest", "all"):
        run_investment_pipeline(symbol=symbol, date=date)

    if args.pipeline in ("trade", "all"):
        run_trade_pipeline(symbol=symbol, date=date)


if __name__ == "__main__":
    main()
