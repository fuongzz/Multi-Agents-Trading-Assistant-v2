"""backtest — Walk-forward backtester cho 14 trading setups.

Không dùng LLM. Tái dụng logic detect_* từ trade_screener.py.
Chống look-ahead bias bằng walk-forward slicing: df.iloc[:i+1].

Cách dùng nhanh:
    from multiagents_trading_assistant.backtest import run_symbol, print_report
    from multiagents_trading_assistant.fetcher import get_ohlcv

    df = get_ohlcv("VCB", n_days=500)
    trades = run_symbol("VCB", df, from_date="2024-01-01")
    print_report(trades, label="VCB", from_date="2024-01-01")
"""

from multiagents_trading_assistant.backtest.engine import run_symbol, run_universe
from multiagents_trading_assistant.backtest.metrics import (
    compute_metrics,
    compute_setup_breakdown,
    compute_symbol_breakdown,
)
from multiagents_trading_assistant.backtest.report import print_report, save_report

__all__ = [
    "run_symbol",
    "run_universe",
    "compute_metrics",
    "compute_setup_breakdown",
    "compute_symbol_breakdown",
    "print_report",
    "save_report",
]
