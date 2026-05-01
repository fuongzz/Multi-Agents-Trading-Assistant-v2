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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ta", "money-flow"],
        default="ta",
        help="Chế độ backtest: ta (TA setups, mặc định) | money-flow (Blackbox regime)",
    )
    parser.add_argument(
        "--min-score",
        dest="min_score",
        type=int,
        default=3,
        metavar="N",
        help="[money-flow] Điểm Blackbox tối thiểu để vào lệnh (mặc định 3)",
    )
    parser.add_argument(
        "--allow-downtrend",
        dest="allow_downtrend",
        action="store_true",
        help="[money-flow] Cho phép vào lệnh khi VNI đang DOWNTREND (mặc định tắt)",
    )
    parser.add_argument(
        "--min-value",
        dest="min_value_b",
        type=float,
        default=None,
        metavar="B",
        help="[money-flow] Ngưỡng giá trị giao dịch tối thiểu tính bằng tỷ VND (mặc định 20B). Dùng 5 cho mid-cap.",
    )
    parser.add_argument(
        "--no-money-flow",
        dest="use_money_flow",
        action="store_false",
        default=True,
        help="[ta] Tat money-flow confirmation trong TA backtest.",
    )
    parser.add_argument(
        "--pipeline-review",
        action="store_true",
        help="[ta] Bat tang review thu 2 mo phong synthesis/trader/risk, khong goi LLM.",
    )


def run_backtest_cli(args: argparse.Namespace) -> None:
    """Entry point backtest, gọi từ main()."""
    from multiagents_trading_assistant.backtest.engine import run_symbol, run_universe
    from multiagents_trading_assistant.backtest.pipeline_review import (
        new_stats as new_pipeline_review_stats,
        print_review_stats,
    )
    from multiagents_trading_assistant.backtest.report import print_report, save_report
    from multiagents_trading_assistant.fetcher import (
        get_liquid_symbols,
        get_ohlcv,
        get_ohlcv_batch,
        get_vn100_symbols,
        get_vn30_symbols,
        get_vnindex,
    )

    from_date       = args.from_date
    to_date         = args.to_date or _date.today().strftime("%Y-%m-%d")
    setups          = [args.setup.upper()] if getattr(args, "setup", None) else None
    symbol          = args.symbol.upper() if getattr(args, "symbol", None) else None
    max_hold        = getattr(args, "max_hold", 120)
    rr              = getattr(args, "rr", 1.5)
    universe        = getattr(args, "universe", None)
    no_save         = getattr(args, "no_save", False)
    mode            = getattr(args, "mode", "ta")
    min_score       = getattr(args, "min_score", 3)
    allow_downtrend = getattr(args, "allow_downtrend", False)
    min_value_b     = getattr(args, "min_value_b", None)
    use_money_flow  = getattr(args, "use_money_flow", True)
    pipeline_review = getattr(args, "pipeline_review", False)
    mf_params       = {"min_avg_value": int(min_value_b * 1_000_000_000)} if min_value_b is not None else None
    review_stats    = new_pipeline_review_stats() if pipeline_review else None

    # ── Money-flow mode ────────────────────────────────────────────────────────
    if mode == "money-flow":
        _run_money_flow_backtest(
            symbol=symbol,
            universe=universe,
            from_date=from_date,
            to_date=to_date,
            rr=rr,
            min_score=min_score,
            filter_downtrend=not allow_downtrend,
            min_value_b=min_value_b,
            no_save=no_save,
            get_ohlcv=get_ohlcv,
            get_ohlcv_batch=get_ohlcv_batch,
            get_vn30_symbols=get_vn30_symbols,
            get_vn100_symbols=get_vn100_symbols,
            get_liquid_symbols=get_liquid_symbols,
            get_vnindex=get_vnindex,
            print_report=print_report,
            save_report=save_report,
        )
        return

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
            use_money_flow=use_money_flow,
            money_flow_min_score=min_score,
            money_flow_params=mf_params,
            pipeline_review=pipeline_review,
            pipeline_review_stats=review_stats,
        )
        print_review_stats(review_stats)
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
            use_money_flow=use_money_flow,
            money_flow_min_score=min_score,
            money_flow_params=mf_params,
            pipeline_review=pipeline_review,
            pipeline_review_stats=review_stats,
        )

        all_trades = [t for sym_trades in results.values() for t in sym_trades]
        print_review_stats(review_stats)
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


def _run_money_flow_backtest(
    symbol, universe, from_date, to_date, rr, min_score, filter_downtrend,
    min_value_b, no_save, get_ohlcv, get_ohlcv_batch, get_vn30_symbols,
    get_vn100_symbols, get_liquid_symbols, get_vnindex, print_report, save_report,
) -> None:
    """Entry point cho --mode money-flow."""
    from multiagents_trading_assistant.backtest.money_flow_engine import (
        run_money_flow_symbol,
        run_money_flow_universe,
        print_money_flow_report,
    )

    start_dt      = datetime.strptime(from_date, "%Y-%m-%d")
    end_dt        = datetime.strptime(to_date,   "%Y-%m-%d")
    calendar_span = (end_dt - start_dt).days
    # Money flow cần 140 bar lookback thay vì 60 — thêm buffer lớn hơn
    n_days = max(400, int((calendar_span + 140) * 0.72) + 140)

    print(f"[money_flow_bt] Fetching VNINDEX ({n_days} bars)...")
    vnindex_df = get_vnindex(n_days=n_days)
    if vnindex_df is None or vnindex_df.empty:
        print("[money_flow_bt] Không lấy được VNINDEX — bỏ market trend filter")
        vnindex_df = None

    # Build params override nếu user chỉ định --min-value
    mf_params = {"min_avg_value": int(min_value_b * 1_000_000_000)} if min_value_b is not None else None

    filter_str = "có" if filter_downtrend else "không"
    min_val_str = f"{min_value_b}B VND" if min_value_b is not None else "default (20B)"
    print(
        f"[money_flow_bt] mode=money-flow, min_score={min_score}, "
        f"filter_downtrend={filter_str}, min_value={min_val_str}"
    )

    if symbol:
        print(f"[money_flow_bt] Fetching {symbol} ({n_days} bars)...")
        df = get_ohlcv(symbol, n_days=n_days)
        if df.empty:
            print(f"[money_flow_bt] Không có data cho {symbol}")
            return

        trades = run_money_flow_symbol(
            symbol=symbol,
            df=df,
            vnindex_df=vnindex_df,
            min_score=min_score,
            filter_downtrend=filter_downtrend,
            params=mf_params,
            from_date=from_date,
            to_date=to_date,
            rr_ratio=rr,
        )
        print_money_flow_report(trades, label=symbol, from_date=from_date, to_date=to_date)
        if not no_save:
            save_report(trades, label=f"MF_{symbol}", from_date=from_date, to_date=to_date)

    elif universe:
        if universe == "vn30":
            symbols = get_vn30_symbols()
        elif universe == "vn100":
            symbols = get_vn100_symbols()
        else:
            symbols = get_liquid_symbols(min_avg_vol=500_000)

        print(f"[money_flow_bt] Universe {universe.upper()}: {len(symbols)} mã")
        print(f"[money_flow_bt] Fetching OHLCV batch ({n_days} bars)...")
        ohlcv_map = get_ohlcv_batch(symbols, n_days=n_days)

        results = run_money_flow_universe(
            ohlcv_map=ohlcv_map,
            vnindex_df=vnindex_df,
            min_score=min_score,
            filter_downtrend=filter_downtrend,
            params=mf_params,
            from_date=from_date,
            to_date=to_date,
            rr_ratio=rr,
        )
        all_trades = [t for ts in results.values() for t in ts]
        print_money_flow_report(
            all_trades,
            label=universe.upper(),
            from_date=from_date,
            to_date=to_date,
            show_symbol_breakdown=True,
        )
        if not no_save:
            save_report(all_trades, label=f"MF_{universe.upper()}", from_date=from_date, to_date=to_date)

    else:
        print("[money_flow_bt] Cần --symbol <MÃ> hoặc --universe <vn30|vn100|liquid>")
        print("  VD: --backtest --mode money-flow --symbol HPG --from 2024-01-01")
        print("  VD: --backtest --mode money-flow --universe vn30 --from 2023-01-01")
