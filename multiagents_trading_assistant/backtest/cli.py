"""CLI backtest — tích hợp vào main.py qua --backtest flag.

Cách dùng:
  python -m multiagents_trading_assistant.main --backtest --symbol VCB
  python -m multiagents_trading_assistant.main --backtest --symbol VCB --from 2024-01-01
  python -m multiagents_trading_assistant.main --backtest --universe vn30 --from 2023-01-01
  python -m multiagents_trading_assistant.main --backtest --universe liquid --setup BREAKOUT
"""

import argparse
import importlib.metadata  # noqa: F401 — pandas-ta Python 3.11 fix
from datetime import date as _date, datetime


def add_backtest_args(parser: argparse.ArgumentParser) -> None:
    """Thêm backtest-specific args vào parser chính của main.py."""
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Chạy backtest thay vì live pipeline",
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        type=str,
        default="2024-01-01",
        metavar="YYYY-MM-DD",
        help="Ngày bắt đầu backtest (mặc định 2024-01-01)",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Ngày kết thúc backtest (mặc định: hôm nay)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        choices=["vn30", "vn100", "liquid"],
        help="Backtest toàn bộ universe: vn30 | vn100 | liquid (vol≥500k)",
    )
    parser.add_argument(
        "--setup",
        type=str,
        metavar="SETUP_NAME",
        help=(
            "Chỉ test một setup (VD: BREAKOUT, MA_PULLBACK, RSI_BOUNCE). "
            "Có thể kết hợp với --symbol hoặc --universe."
        ),
    )
    parser.add_argument(
        "--max-hold",
        dest="max_hold",
        type=int,
        default=120,
        metavar="N",
        help="Safety cap tuyệt đối — số bar tối đa (mặc định 120 ≈ 6 tháng)",
    )
    parser.add_argument(
        "--rr",
        type=float,
        default=1.5,
        metavar="RATIO",
        help="R:R ratio tối thiểu để tính TP (mặc định 1.5)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Không lưu kết quả ra CSV",
    )


def run_backtest_cli(args: argparse.Namespace) -> None:
    """Entry point backtest, gọi từ main()."""
    from multiagents_trading_assistant.backtest.engine import run_symbol, run_universe
    from multiagents_trading_assistant.backtest.report import print_report, save_report
    from multiagents_trading_assistant.fetcher import (
        get_liquid_symbols,
        get_ohlcv,
        get_ohlcv_batch,
        get_vn100_symbols,
        get_vn30_symbols,
    )

    from_date = args.from_date
    to_date   = args.to_date or _date.today().strftime("%Y-%m-%d")
    setups    = [args.setup.upper()] if getattr(args, "setup", None) else None
    symbol    = args.symbol.upper() if getattr(args, "symbol", None) else None
    max_hold  = getattr(args, "max_hold", 120)
    rr        = getattr(args, "rr", 1.5)
    universe  = getattr(args, "universe", None)
    no_save   = getattr(args, "no_save", False)

    # Tính n_days cần fetch: khoảng thời gian + buffer lookback (60 bars) + ngày nghỉ
    start_dt      = datetime.strptime(from_date, "%Y-%m-%d")
    end_dt        = datetime.strptime(to_date,   "%Y-%m-%d")
    calendar_span = (end_dt - start_dt).days
    # ~0.72 trading days / calendar day; thêm 90 ngày buffer cho lookback
    n_days = max(300, int((calendar_span + 90) * 0.72) + 90)

    if symbol:
        # ── Single-symbol mode ────────────────────────────────────
        print(f"[backtest] Fetching {symbol} ({n_days} bars)...")
        df = get_ohlcv(symbol, n_days=n_days)
        if df.empty:
            print(f"[backtest] Không có data cho {symbol}")
            return

        trades = run_symbol(
            symbol=symbol,
            df=df,
            max_hold=max_hold,
            rr_ratio=rr,
            from_date=from_date,
            to_date=to_date,
            setups=setups,
        )
        print_report(trades, label=symbol, from_date=from_date, to_date=to_date)
        if not no_save:
            save_report(trades, label=symbol, from_date=from_date, to_date=to_date)

    elif universe:
        # ── Universe mode ─────────────────────────────────────────
        if universe == "vn30":
            symbols = get_vn30_symbols()
        elif universe == "vn100":
            symbols = get_vn100_symbols()
        else:
            symbols = get_liquid_symbols(min_avg_vol=500_000)

        print(f"[backtest] Universe {universe.upper()}: {len(symbols)} mã")
        print(f"[backtest] Fetching OHLCV batch ({n_days} bars)...")
        ohlcv_map = get_ohlcv_batch(symbols, n_days=n_days)

        results = run_universe(
            ohlcv_map=ohlcv_map,
            max_hold=max_hold,
            rr_ratio=rr,
            from_date=from_date,
            to_date=to_date,
            setups=setups,
        )

        all_trades = [t for sym_trades in results.values() for t in sym_trades]
        print_report(
            all_trades,
            label=universe.upper(),
            from_date=from_date,
            to_date=to_date,
            show_symbol_breakdown=True,
        )
        if not no_save:
            save_report(all_trades, label=universe.upper(), from_date=from_date, to_date=to_date)

    else:
        print("[backtest] Cần chỉ định --symbol <MÃ> hoặc --universe <vn30|vn100|liquid>")
        print("  VD: --backtest --symbol VCB --from 2024-01-01")
        print("  VD: --backtest --universe vn30 --from 2023-01-01 --to 2024-12-31")
