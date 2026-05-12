"""CLI backtest — tích hợp vào main.py qua --backtest flag.

Cách dùng:
  python -m multiagents_trading_assistant.main --backtest --symbol VCB
  python -m multiagents_trading_assistant.main --backtest --symbol VCB --from 2024-01-01
  python -m multiagents_trading_assistant.main --backtest --universe vn30 --from 2023-01-01
  python -m multiagents_trading_assistant.main --backtest --universe liquid --setup BREAKOUT
"""

import argparse
import importlib.metadata  # noqa: F401 — pandas-ta Python 3.11 fix
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as _date, datetime, timedelta

# Setups phù hợp cho blue-chip / ngân hàng (ổn định, biến động thấp):
# Loại bỏ: NR7, RSI_BOUNCE, DOUBLE_BOTTOM, BULLISH_ENGULFING, INSIDE_BAR, HAMMER, PIN_BAR, KIJUN_BOUNCE
# (các setup này cần biến động cao / pattern đảo chiều rõ — không phù hợp VCB, BID, CTG, ...)
_BLUECHIP_SETUPS = [
    "SPRING", "BB_SQUEEZE", "BREAKOUT_RETEST_ENTRY", "MACD_CROSSOVER",
    "TREND_PULLBACK", "MA_PULLBACK", "GOLDEN_CROSS", "RETEST",
    "BREAKOUT", "FLAG_PENNANT", "MOMENTUM_SURGE",
    "KUMO_BREAKOUT", "TK_CROSS", "KUMO_TWIST_ENTRY",
]


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
            "Chỉ test một hoặc nhiều setup, phân tách bằng dấu phẩy "
            "(VD: BREAKOUT hoặc KUMO_BREAKOUT,TK_CROSS). "
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
        choices=["ta", "money-flow", "lifecycle", "live-pipeline", "llm-graph", "pv-signals"],
        default="ta",
        help=(
            "Chế độ backtest: ta (TA setups, mặc định) | money-flow | lifecycle | "
            "live-pipeline | llm-graph (TradingAgents-VN với LLM decisions) | "
            "pv-signals (Price-Volume Intelligence Layer standalone)"
        ),
    )
    parser.add_argument(
        "--pv-gate",
        dest="pv_gate",
        action="store_true",
        help=(
            "[ta] Bật PV gate: chặn TA entry khi PV bias='avoid' hoặc có "
            "FAKE_BREAKOUT_RISK / HEAVY_DISTRIBUTION. Kết hợp với --mode ta."
        ),
    )
    parser.add_argument(
        "--pv-min-score",
        dest="pv_min_score",
        type=int,
        default=0,
        metavar="N",
        help=(
            "[pv-signals] Ngưỡng price_volume_score tối thiểu để vào lệnh "
            "(mặc định 0 = chỉ cần entry_bias bullish). Đề xuất: 20-35."
        ),
    )
    parser.add_argument(
        "--pv-signals",
        dest="pv_signals",
        type=str,
        default=None,
        metavar="A,B,...",
        help=(
            "[pv-signals] Lọc PV signal: PV_BREAKOUT,PV_WASHOUT,PV_ABSORPTION,"
            "PV_PULLBACK,PV_BULL_EXPANSION. Mặc định: tất cả 5 signal."
        ),
    )
    # llm-graph specific args
    parser.add_argument(
        "--cheap",
        action="store_true",
        help="[llm-graph] Dùng Haiku cho tất cả agent (~$0.03/mã)",
    )
    parser.add_argument(
        "--train-end",
        dest="train_period_end",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="[llm-graph] Ngày kết thúc train period. Performance chỉ tính từ sau ngày này.",
    )
    parser.add_argument(
        "--max-candidates",
        dest="max_candidates",
        type=int,
        default=5,
        metavar="N",
        help="[llm-graph] Số candidates top tối đa để deep analyze mỗi ngày (mặc định 5)",
    )
    parser.add_argument(
        "--score-threshold",
        dest="score_threshold",
        type=float,
        default=0.60,
        metavar="FLOAT",
        help="[llm-graph] Plan score threshold để vào lệnh (mặc định 0.60)",
    )
    parser.add_argument(
        "--exec-model",
        dest="exec_model",
        choices=["next_open", "zone_only", "zone_with_slippage"],
        default="zone_with_slippage",
        help="[llm-graph] Entry execution model (mặc định zone_with_slippage)",
    )
    parser.add_argument(
        "--lifecycle-playbook",
        choices=["single", "scale-in", "auto"],
        default="auto",
        help="[lifecycle] Playbook quản trị vị thế: single | scale-in | auto (mặc định auto)",
    )
    parser.add_argument(
        "--lifecycle-total-pct",
        type=float,
        default=9.0,
        metavar="PCT",
        help="[lifecycle] Tổng % NAV dự kiến khi scale-in full (mặc định 9%%)",
    )
    parser.add_argument(
        "--max-total-risk",
        dest="max_total_risk_pct",
        type=float,
        default=6.0,
        metavar="PCT",
        help="[lifecycle] Tổng rủi ro tối đa nếu toàn bộ SL hit, tính theo %% NAV (mặc định 6%%)",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=10.0,
        metavar="BPS",
        help="[lifecycle] Slippage mô phỏng cho fill, basis points mỗi chiều (mặc định 10)",
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
    parser.add_argument(
        "--candidate-llm-backtest",
        action="store_true",
        help=(
            "[ta] Backtest pre-LLM candidate filter de uoc tinh so call LLM tiet kiem. "
            "Tu dong bat --pipeline-review."
        ),
    )
    parser.add_argument(
        "--llm-lite-calls-per-candidate",
        type=int,
        default=4,
        metavar="N",
        help="[candidate-llm] So Haiku/lite calls moi candidate live (mac dinh 4).",
    )
    parser.add_argument(
        "--llm-sonnet-calls-per-candidate",
        type=int,
        default=1,
        metavar="N",
        help="[candidate-llm] So Sonnet calls moi candidate live (mac dinh 1).",
    )
    parser.add_argument(
        "--llm-lite-cost",
        type=float,
        default=float(os.getenv("LLM_LITE_COST_USD_PER_CALL", "0") or 0),
        metavar="USD",
        help="[candidate-llm] Optional USD/call cho lite model.",
    )
    parser.add_argument(
        "--llm-sonnet-cost",
        type=float,
        default=float(os.getenv("LLM_SONNET_COST_USD_PER_CALL", "0") or 0),
        metavar="USD",
        help="[candidate-llm] Optional USD/call cho Sonnet model.",
    )
    parser.add_argument(
        "--pending-entry",
        action="store_true",
        help="[ta] Mo phong recommendation-only: signal cho vao danh sach cho, chi khop khi gia cham entry zone.",
    )
    parser.add_argument(
        "--pending-bars",
        dest="pending_bars",
        type=int,
        default=3,
        metavar="N",
        help="[ta] So phien giu signal cho khop khi dung --pending-entry (mac dinh 3).",
    )
    parser.add_argument(
        "--ichimoku-params",
        type=str,
        default=None,
        metavar="T,K,SENKOUB,DISP,CHIKOU",
        help="[ta] Override Ichimoku params, VD: 9,17,26,26,26. Mac dinh: 9,26,52,26,26.",
    )
    parser.add_argument(
        "--cooldown",
        dest="cooldown_bars",
        type=int,
        default=0,
        metavar="N",
        help=(
            "[ta] Số bar chờ sau mỗi exit trước khi tìm signal mới. "
            "SL/TSL → N bar, profitable exit → N//2 bar. "
            "0 = tắt (mặc định). Đề xuất: 5 cho single-symbol, 3 cho universe."
        ),
    )
    parser.add_argument(
        "--loss-streak-pause",
        dest="loss_streak_pause",
        type=int,
        default=0,
        metavar="N",
        help=(
            "[ta] Sau N lần lỗ liên tiếp, kéo dài cooldown thêm cooldown × 2 bar. "
            "0 = tắt (mặc định). Đề xuất: 3."
        ),
    )
    parser.add_argument(
        "--max-positions",
        dest="max_positions",
        type=int,
        default=0,
        metavar="N",
        help=(
            "[ta] [universe] Giới hạn số vị thế đồng thời tối đa. "
            "Khi đầy, bỏ qua signal mới; khi nhiều signal cùng ngày, ưu tiên confluence cao hơn. "
            "0 = không giới hạn (mặc định). Đề xuất: 5 cho universe."
        ),
    )
    parser.add_argument(
        "--initial-capital",
        dest="initial_capital",
        type=float,
        default=100_000_000.0,
        metavar="VND",
        help="[live-pipeline] Initial capital in VND.",
    )
    parser.add_argument(
        "--max-candidates-per-day",
        dest="max_candidates_per_day",
        type=int,
        default=10,
        metavar="N",
        help="[live-pipeline] Max screener candidates accepted per day.",
    )
    parser.add_argument(
        "--setups",
        dest="setups",
        type=str,
        default=None,
        metavar="A,B,...",
        help="[live-pipeline] Whitelist setup (comma-separated). VD: BB_SQUEEZE,FLAG_PENNANT",
    )
    parser.add_argument(
        "--require-uptrend",
        dest="require_uptrend",
        action="store_true",
        default=False,
        help="[live-pipeline] Chỉ vào lệnh khi VNINDEX/reference đang UPTREND (bỏ qua SIDEWAY).",
    )
    parser.add_argument(
        "--backtest-mode",
        dest="backtest_mode",
        choices=["raw_screener", "broad_pool", "simulated_llm_gate"],
        default="raw_screener",
        help=(
            "[live-pipeline] raw_screener=screener tự trade (mặc định); "
            "broad_pool=đo candidate flow, không execute; "
            "simulated_llm_gate=broad + proxy LLM filter + execute."
        ),
    )
    parser.add_argument(
        "--edge-strategy",
        dest="edge_strategy_name",
        type=str,
        default=None,
        metavar="NAME",
        help="[live-pipeline] Chỉ trade candidate pass edge strategy này, ví dụ breakout_after_accumulation_v3.",
    )
    parser.add_argument(
        "--label",
        dest="label",
        type=str,
        default=None,
        metavar="TEXT",
        help="[live-pipeline] Custom label cho tên thư mục kết quả.",
    )
    parser.add_argument(
        "--stock-profile",
        dest="stock_profile",
        type=str,
        choices=["bluechip", "banking", "speculative", "default"],
        default="default",
        help=(
            "[ta] Lọc setup phù hợp với đặc điểm cổ phiếu: "
            "bluechip/banking = loại bỏ setup cần biến động cao (NR7, RSI_BOUNCE, DOUBLE_BOTTOM, ...); "
            "speculative/default = toàn bộ 22 setup. "
            "Bị override bởi --setup nếu chỉ định rõ."
        ),
    )


def run_backtest_cli(args: argparse.Namespace) -> None:
    """Entry point backtest, gọi từ main()."""
    import pandas as pd
    from multiagents_trading_assistant.backtest.engine import run_symbol, run_universe
    from multiagents_trading_assistant.backtest.pipeline_review import (
        new_stats as new_pipeline_review_stats,
        print_candidate_llm_budget,
        print_review_stats,
    )
    from multiagents_trading_assistant.backtest.report import print_report, save_report
    from multiagents_trading_assistant.fetcher import (
        get_liquid_symbols,
        get_ohlcv,
        get_ohlcv_batch,
        get_ohlcv_history,
        get_vn100_symbols,
        get_vn30_symbols,
        get_vnindex,
    )

    from_date       = args.from_date
    to_date         = args.to_date or _date.today().strftime("%Y-%m-%d")
    stock_profile   = getattr(args, "stock_profile", "default")
    # --setup explicit overrides --stock-profile
    if getattr(args, "setup", None):
        setups = [s.strip().upper() for s in args.setup.split(",") if s.strip()]
    elif stock_profile in ("bluechip", "banking"):
        setups = _BLUECHIP_SETUPS
        print(f"[backtest] stock-profile={stock_profile} → {len(setups)} setups (loại bỏ NR7, RSI_BOUNCE, DOUBLE_BOTTOM, ...)")
    else:
        setups = None
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
    candidate_llm_backtest = getattr(args, "candidate_llm_backtest", False)
    pipeline_review = getattr(args, "pipeline_review", False) or candidate_llm_backtest
    pending_entry    = getattr(args, "pending_entry", False)
    pending_bars     = getattr(args, "pending_bars", 3)
    cooldown_bars    = getattr(args, "cooldown_bars", 0)
    loss_streak_pause = getattr(args, "loss_streak_pause", 0)
    max_positions    = getattr(args, "max_positions", 0)
    pv_gate          = getattr(args, "pv_gate", False)
    pv_min_score     = getattr(args, "pv_min_score", 0)
    pv_signals_raw   = getattr(args, "pv_signals", None)
    pv_signals_filter = (
        [s.strip().upper() for s in pv_signals_raw.split(",") if s.strip()]
        if pv_signals_raw else None
    )
    ichimoku_params  = getattr(args, "ichimoku_params", None)
    mf_params        = {"min_avg_value": int(min_value_b * 1_000_000_000)} if min_value_b is not None else None
    review_stats    = new_pipeline_review_stats() if pipeline_review else None
    if ichimoku_params:
        os.environ["ICHIMOKU_PARAMS"] = ichimoku_params
        print(f"[backtest] Ichimoku params: {ichimoku_params}")
    else:
        os.environ.pop("ICHIMOKU_PARAMS", None)

    # Tính khoảng history cần fetch cho TA/lifecycle modes.
    start_dt      = datetime.strptime(from_date, "%Y-%m-%d")
    end_dt        = datetime.strptime(to_date,   "%Y-%m-%d")
    calendar_span = (end_dt - start_dt).days
    n_days = max(300, int((calendar_span + 90) * 0.72) + 90)
    history_start = (start_dt - timedelta(days=220)).strftime("%Y-%m-%d")

    # ── LLM-graph mode ────────────────────────────────────────────────────────
    if mode == "llm-graph":
        _run_llm_graph_backtest(args, from_date, to_date)
        return

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

    if mode == "lifecycle":
        _run_lifecycle_backtest(
            symbol=symbol,
            universe=universe,
            from_date=from_date,
            to_date=to_date,
            setups=setups,
            rr=rr,
            min_score=min_score,
            use_money_flow=use_money_flow,
            mf_params=mf_params,
            pipeline_review=pipeline_review,
            no_save=no_save,
            get_ohlcv_history=get_ohlcv_history,
            get_vn30_symbols=get_vn30_symbols,
            get_vn100_symbols=get_vn100_symbols,
            get_liquid_symbols=get_liquid_symbols,
            playbook=getattr(args, "lifecycle_playbook", "auto"),
            lifecycle_total_pct=getattr(args, "lifecycle_total_pct", 9.0),
            max_positions=max_positions if max_positions > 0 else 5,
            max_total_risk_pct=getattr(args, "max_total_risk_pct", 6.0),
            slippage_bps=getattr(args, "slippage_bps", 10.0),
            history_start=history_start,
        )
        return

    # ── PV-signals mode ──────────────────────────────────────────────────────
    if mode == "pv-signals":
        _run_pv_signals_backtest(
            symbol=symbol,
            universe=universe,
            from_date=from_date,
            to_date=to_date,
            rr=rr,
            max_hold=max_hold,
            pv_min_score=pv_min_score,
            pv_signals_filter=pv_signals_filter,
            cooldown_bars=cooldown_bars,
            loss_streak_pause=loss_streak_pause,
            max_positions=max_positions,
            no_save=no_save,
            get_ohlcv_history=get_ohlcv_history,
            get_vn30_symbols=get_vn30_symbols,
            get_vn100_symbols=get_vn100_symbols,
            get_liquid_symbols=get_liquid_symbols,
            print_report=print_report,
            save_report=save_report,
            history_start=history_start,
        )
        return

    if mode == "live-pipeline":
        _run_live_pipeline_backtest_cli(
            symbol=symbol,
            universe=universe,
            from_date=from_date,
            to_date=to_date,
            initial_capital=getattr(args, "initial_capital", 100_000_000.0),
            max_positions=max_positions if max_positions > 0 else 5,
            max_candidates_per_day=getattr(args, "max_candidates_per_day", 10),
            max_hold=max_hold,
            rr=rr,
            slippage_bps=getattr(args, "slippage_bps", 10.0),
            allow_downtrend=allow_downtrend,
            require_uptrend=getattr(args, "require_uptrend", False),
            setups=getattr(args, "setups", None),
            backtest_mode=getattr(args, "backtest_mode", "raw_screener"),
            edge_strategy_name=getattr(args, "edge_strategy_name", None),
            custom_label=getattr(args, "label", None),
            no_save=no_save,
        )
        return

    if symbol:
        # ── Single-symbol mode ────────────────────────────────────
        print(f"[backtest] Fetching {symbol} history {history_start} → {to_date}...")
        df = get_ohlcv_history(symbol, start=history_start, end=to_date)
        if df.empty:
            print(f"[backtest] Không có data cho {symbol}")
            return

        if pv_gate:
            print(f"[backtest] PV gate ON — chặn entry khi PV bias=avoid / FAKE_BREAKOUT_RISK / HEAVY_DISTRIBUTION")
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
            pending_entry=pending_entry,
            pending_bars=pending_bars,
            cooldown_bars=cooldown_bars,
            loss_streak_pause=loss_streak_pause,
            pv_gate=pv_gate,
        )
        print_review_stats(review_stats)
        if candidate_llm_backtest:
            print_candidate_llm_budget(
                review_stats,
                trades,
                lite_calls_per_candidate=getattr(args, "llm_lite_calls_per_candidate", 4),
                sonnet_calls_per_candidate=getattr(args, "llm_sonnet_calls_per_candidate", 1),
                lite_cost_per_call=getattr(args, "llm_lite_cost", 0.0),
                sonnet_cost_per_call=getattr(args, "llm_sonnet_cost", 0.0),
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
        print(f"[backtest] Fetching OHLCV history batch {history_start} → {to_date} ({len(symbols)} mã, 8 workers)...")
        ohlcv_map: dict = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            fut_map = {ex.submit(get_ohlcv_history, sym, history_start, to_date): sym for sym in symbols}
            for fut in as_completed(fut_map):
                sym = fut_map[fut]
                try:
                    ohlcv_map[sym] = fut.result()
                except Exception as e:
                    print(f"[backtest] {sym} fetch error: {e}")
                    ohlcv_map[sym] = pd.DataFrame()

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
            pending_entry=pending_entry,
            pending_bars=pending_bars,
            cooldown_bars=cooldown_bars,
            loss_streak_pause=loss_streak_pause,
            pv_gate=pv_gate,
        )

        all_trades = [t for sym_trades in results.values() for t in sym_trades]
        if max_positions > 0:
            from multiagents_trading_assistant.backtest.engine import apply_portfolio_filter
            before = len(all_trades)
            all_trades = apply_portfolio_filter(all_trades, max_positions=max_positions)
            print(f"[backtest] Portfolio filter (max={max_positions}): {before} → {len(all_trades)} trades")
        print_review_stats(review_stats)
        if candidate_llm_backtest:
            print_candidate_llm_budget(
                review_stats,
                all_trades,
                lite_calls_per_candidate=getattr(args, "llm_lite_calls_per_candidate", 4),
                sonnet_calls_per_candidate=getattr(args, "llm_sonnet_calls_per_candidate", 1),
                lite_cost_per_call=getattr(args, "llm_lite_cost", 0.0),
                sonnet_cost_per_call=getattr(args, "llm_sonnet_cost", 0.0),
            )
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


def _run_pv_signals_backtest(
    *,
    symbol,
    universe,
    from_date,
    to_date,
    rr,
    max_hold,
    pv_min_score,
    pv_signals_filter,
    cooldown_bars,
    loss_streak_pause,
    max_positions,
    no_save,
    get_ohlcv_history,
    get_vn30_symbols,
    get_vn100_symbols,
    get_liquid_symbols,
    print_report,
    save_report,
    history_start,
) -> None:
    """Entry point cho --mode pv-signals: dùng PV Intelligence Layer làm entry trigger."""
    from multiagents_trading_assistant.backtest.backtest_pv import (
        run_pv_symbol,
        run_pv_universe,
    )
    from multiagents_trading_assistant.backtest.engine import apply_portfolio_filter

    sig_label = ",".join(pv_signals_filter) if pv_signals_filter else "ALL"
    print(
        f"[pv-backtest] mode=pv-signals | signals={sig_label} | "
        f"min_score={pv_min_score} | rr={rr} | cooldown={cooldown_bars}"
    )

    if symbol:
        print(f"[pv-backtest] Fetching {symbol} history {history_start} → {to_date}...")
        df = get_ohlcv_history(symbol, start=history_start, end=to_date)
        if df.empty:
            print(f"[pv-backtest] Không có data cho {symbol}")
            return

        trades = run_pv_symbol(
            symbol=symbol,
            df=df,
            signals=pv_signals_filter,
            min_pv_score=pv_min_score,
            from_date=from_date,
            to_date=to_date,
            rr_ratio=rr,
            max_hold=max_hold,
            cooldown_bars=cooldown_bars,
            loss_streak_pause=loss_streak_pause,
        )
        print_report(trades, label=f"PV_{symbol}", from_date=from_date, to_date=to_date)
        if not no_save:
            save_report(trades, label=f"PV_{symbol}", from_date=from_date, to_date=to_date)

    elif universe:
        if universe == "vn30":
            symbols = get_vn30_symbols()
        elif universe == "vn100":
            symbols = get_vn100_symbols()
        else:
            symbols = get_liquid_symbols(min_avg_vol=500_000)

        print(f"[pv-backtest] Universe {universe.upper()}: {len(symbols)} mã")
        print(f"[pv-backtest] Fetching OHLCV history batch {history_start} → {to_date}...")
        ohlcv_map: dict = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            fut_map = {ex.submit(get_ohlcv_history, sym, history_start, to_date): sym for sym in symbols}
            for fut in as_completed(fut_map):
                sym = fut_map[fut]
                try:
                    ohlcv_map[sym] = fut.result()
                except Exception as e:
                    print(f"[pv-backtest] {sym} fetch error: {e}")
                    ohlcv_map[sym] = __import__("pandas").DataFrame()

        results = run_pv_universe(
            ohlcv_map=ohlcv_map,
            signals=pv_signals_filter,
            min_pv_score=pv_min_score,
            from_date=from_date,
            to_date=to_date,
            rr_ratio=rr,
            max_hold=max_hold,
            cooldown_bars=cooldown_bars,
            loss_streak_pause=loss_streak_pause,
        )

        all_trades = [t for ts in results.values() for t in ts]
        if max_positions > 0:
            before = len(all_trades)
            all_trades = apply_portfolio_filter(all_trades, max_positions=max_positions)
            print(f"[pv-backtest] Portfolio filter (max={max_positions}): {before} → {len(all_trades)} trades")

        print_report(
            all_trades,
            label=f"PV_{universe.upper()}",
            from_date=from_date,
            to_date=to_date,
            show_symbol_breakdown=True,
        )
        if not no_save:
            save_report(all_trades, label=f"PV_{universe.upper()}", from_date=from_date, to_date=to_date)

    else:
        print("[pv-backtest] Cần --symbol <MÃ> hoặc --universe <vn30|vn100|liquid>")
        print("  VD: --backtest --mode pv-signals --symbol VCB --from 2024-01-01")
        print("  VD: --backtest --mode pv-signals --universe vn30 --from 2023-01-01 --pv-min-score 25")
        print("  VD: --backtest --mode pv-signals --universe liquid --pv-signals PV_BREAKOUT,PV_WASHOUT")


def _run_llm_graph_backtest(
    args: "argparse.Namespace",
    from_date: str,
    to_date: str,
) -> None:
    """Dispatch cho --mode llm-graph.

    Chạy TradingAgentsVN trên mỗi ngày trong khoảng backtest.
    Kết quả in ra console; không lưu CSV (phase 1).
    """
    from multiagents_trading_assistant.backtest.execution import LLMExecutionConfig
    from multiagents_trading_assistant.backtest.llm_backtest import (
        LLMBacktestConfig,
        run_llm_backtest,
    )

    symbol = getattr(args, "symbol", None)
    symbol = symbol.upper() if symbol else None
    universe = getattr(args, "universe", None)

    if not symbol and not universe:
        print("[llm-graph] Cần --symbol hoặc --universe")
        return

    exec_cfg = LLMExecutionConfig(
        model=getattr(args, "exec_model", "zone_with_slippage"),
        slippage_bps=20.0,
        max_position_vol_pct=0.10,
    )

    cfg = LLMBacktestConfig(
        symbol=symbol,
        universe=universe,
        from_date=from_date,
        to_date=to_date,
        train_period_end=getattr(args, "train_period_end", None),
        cheap_mode=getattr(args, "cheap", True),
        execution=exec_cfg,
        max_candidates_per_day=getattr(args, "max_candidates", 5),
        score_threshold=getattr(args, "score_threshold", 0.60),
        use_cache=True,
        verbose=True,
    )

    print(f"\n[llm-graph] Backtest {symbol or universe} | {from_date} → {to_date}")
    if cfg.train_period_end:
        print(f"[llm-graph] Train period: {from_date} → {cfg.train_period_end}")
        print(f"[llm-graph] Test period:  {cfg.train_period_end} → {to_date}")
    print(f"[llm-graph] cheap={cfg.cheap_mode} | exec_model={exec_cfg.model} | threshold={cfg.score_threshold}")

    trades = run_llm_backtest(cfg)

    test_trades = [t for t in trades if t.is_test_period]
    print(f"\n[llm-graph] Tổng trades (test period): {len(test_trades)}")
    print(f"[llm-graph] Decision log: backtest_results/llm_decisions_{cfg.run_id}.jsonl")


def _run_live_pipeline_backtest_cli(
    *,
    symbol: str | None,
    universe: str | None,
    from_date: str,
    to_date: str,
    initial_capital: float,
    max_positions: int,
    max_candidates_per_day: int,
    max_hold: int,
    rr: float,
    slippage_bps: float,
    allow_downtrend: bool,
    require_uptrend: bool = False,
    setups: str | None = None,
    backtest_mode: str = "raw_screener",
    edge_strategy_name: str | None = None,
    custom_label: str | None = None,
    no_save: bool = False,
) -> None:
    import pandas as pd
    from multiagents_trading_assistant.backtest.live_pipeline import (
        LivePipelineBacktestConfig,
        run_live_pipeline_backtest,
        save_live_pipeline_results,
    )
    from multiagents_trading_assistant.fetcher import (
        get_ohlcv_history,
        get_vn100_symbols,
        get_vn30_symbols,
    )

    if symbol:
        symbols = [symbol]
        label = symbol
    elif universe == "vn30":
        symbols = get_vn30_symbols()
        label = "VN30"
    elif universe == "vn100":
        symbols = get_vn100_symbols()
        label = "VN100"
    elif universe:
        symbols = [item.strip().upper() for item in universe.split(",") if item.strip()]
        label = universe.upper().replace(",", "_")
    else:
        print("[live-backtest] Can chi dinh --symbol hoac --universe vn30|vn100|A,B,C")
        return

    warmup_start = (pd.Timestamp(from_date) - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
    print(f"[live-backtest] Universe {label}: {len(symbols)} symbols")
    print(f"[live-backtest] Fetching history {warmup_start} -> {to_date}")
    data = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(get_ohlcv_history, sym, warmup_start, to_date): sym for sym in symbols}
        for idx, fut in enumerate(as_completed(futures), start=1):
            sym = futures[fut]
            try:
                df = fut.result()
            except Exception as e:
                print(f"[live-backtest] {sym} fetch error: {e}")
                df = pd.DataFrame()
            if not df.empty:
                data[sym] = df
            if idx % 10 == 0:
                print(f"[live-backtest] fetched {idx}/{len(symbols)}")

    setup_whitelist = (
        frozenset(s.strip().upper() for s in setups.split(",") if s.strip())
        if setups else None
    )
    mode_tag = backtest_mode
    if require_uptrend:
        mode_tag += "_uptrend"
    if setup_whitelist:
        mode_tag += "_whitelist"
    if edge_strategy_name:
        mode_tag += f"_{edge_strategy_name}"
    report_label = custom_label or f"{label}_{mode_tag}_{from_date}_{to_date}"

    vnindex = get_ohlcv_history("VNINDEX", warmup_start, to_date)
    cfg = LivePipelineBacktestConfig(
        initial_capital=initial_capital,
        start_date=from_date,
        end_date=to_date,
        max_positions=max_positions,
        max_candidates_per_day=max_candidates_per_day,
        max_hold_bars=max_hold,
        rr_ratio=rr,
        slippage_rate=slippage_bps / 10000.0,
        allow_downtrend_entries=allow_downtrend,
        require_uptrend=require_uptrend,
        setup_whitelist=setup_whitelist,
        backtest_mode=backtest_mode,
        edge_strategy_name=edge_strategy_name,
    )
    result = run_live_pipeline_backtest(data, vnindex, config=cfg)
    print()
    print("LIVE PIPELINE BACKTEST")
    print(f"  Mode          : {backtest_mode}")
    print(f"  Label         : {report_label}")
    if setup_whitelist:
        print(f"  Setups        : {', '.join(sorted(setup_whitelist))}")
    if edge_strategy_name:
        print(f"  Edge strategy : {edge_strategy_name}")
    if require_uptrend:
        print(f"  Market gate   : UPTREND only")

    if backtest_mode == "broad_pool":
        pool = result.get("pool_summary", [])
        cdf = result.get("candidate_log")
        total = len(cdf) if cdf is not None and not cdf.empty else 0
        print(f"  Total candidate appearances : {total}")
        if pool:
            print("  Setup pool summary:")
            for row in pool:
                print(
                    f"    {row['setup_type']:<28} appearances={row['appearances']:>5} "
                    f"symbols={row['unique_symbols']:>3} avg_score={row['avg_score']:>5.1f} "
                    f"top_flag={row['top_risk_flag']}"
                )
    else:
        metrics = result["metrics"]
        print(f"  Total return  : {metrics.get('total_return', 0.0) * 100:+.2f}%")
        print(f"  Sharpe        : {metrics.get('sharpe_ratio', 0.0):.2f}")
        print(f"  Max drawdown  : {metrics.get('max_drawdown', 0.0) * 100:.2f}%")
        print(f"  Win rate      : {metrics.get('win_rate', 0.0) * 100:.1f}%")
        print(f"  Trades        : {metrics.get('number_of_trades', 0)}")
        if result["setup_breakdown"]:
            print("  All setups    :")
            for row in result["setup_breakdown"]:
                pf = row.get("profit_factor")
                pf_str = f"{pf:.2f}" if pf is not None and pf != float("inf") else ("inf" if pf == float("inf") else "n/a")
                print(
                    f"    {row['setup_type']:<28} trades={row['trades']:>4} "
                    f"wr={row['win_rate'] * 100:>5.1f}% avg={row['avg_pnl_pct'] * 100:+.2f}% pf={pf_str}"
                )
    if not no_save:
        outdir = save_live_pipeline_results(result, label=report_label)
        print(f"[live-backtest] Saved -> {outdir}")


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


def _run_lifecycle_backtest(
    *,
    symbol,
    universe,
    from_date,
    to_date,
    setups,
    rr,
    min_score,
    use_money_flow,
    mf_params,
    pipeline_review,
    no_save,
    get_ohlcv_history,
    get_vn30_symbols,
    get_vn100_symbols,
    get_liquid_symbols,
    playbook,
    lifecycle_total_pct,
    max_positions,
    max_total_risk_pct,
    slippage_bps,
    history_start,
) -> None:
    """Lifecycle backtest entry point.

    Phase 1 supports single-symbol first. Universe lifecycle needs a chronological
    event loop across symbols to keep portfolio cash/exposure exact.
    """
    if universe:
        print("[lifecycle_bt] Universe mode chưa bật cho lifecycle.")
        print("[lifecycle_bt] Lý do: cần event loop theo ngày trên toàn universe để portfolio cash/exposure không bị lệch.")
        print("[lifecycle_bt] Hãy chạy trước với --symbol <MÃ>.")
        return
    if not symbol:
        print("[lifecycle_bt] Cần --symbol <MÃ> cho lifecycle mode.")
        return

    from multiagents_trading_assistant.backtest.execution import ExecutionConfig
    from multiagents_trading_assistant.backtest.lifecycle_detector import LifecycleSignalDetector
    from multiagents_trading_assistant.backtest.lifecycle_engine import LifecycleEngine
    from multiagents_trading_assistant.backtest.lifecycle_metrics import (
        portfolio_metrics,
        position_metrics,
    )
    from multiagents_trading_assistant.backtest.pipeline_review import (
        new_stats as new_pipeline_review_stats,
        print_review_stats,
    )
    from multiagents_trading_assistant.backtest.sector_rotation import (
        DEFAULT_SECTOR_MAP,
        SectorRotationModel,
    )
    from multiagents_trading_assistant.backtest.playbook import (
        CoreTrendPlaybook,
        DefensiveExitPlaybook,
        ProbeOnlyBearPlaybook,
        PlaybookRouter,
        RangeReversalPlaybook,
        ScaleInReversalPlaybook,
        SingleEntryPlaybook,
        TrendFollowingPlaybook,
    )
    from multiagents_trading_assistant.backtest.portfolio import (
        PortfolioConstraints,
        PortfolioEngine,
    )

    print(f"[lifecycle_bt] Fetching {symbol} history {history_start} → {to_date}...")
    df = get_ohlcv_history(symbol, start=history_start, end=to_date)
    if df.empty:
        print(f"[lifecycle_bt] Không có data cho {symbol}")
        return
    sector_model = _build_lifecycle_sector_model(
        symbol=symbol,
        symbol_df=df,
        history_start=history_start,
        to_date=to_date,
        get_ohlcv_history=get_ohlcv_history,
        sector_map=DEFAULT_SECTOR_MAP,
    )

    single = SingleEntryPlaybook()
    scale = ScaleInReversalPlaybook(intended_total_pct=float(lifecycle_total_pct))
    core = CoreTrendPlaybook(core_pct=95.0)
    trend = TrendFollowingPlaybook(max_position_pct=25.0)
    range_pb = RangeReversalPlaybook()
    bear = ProbeOnlyBearPlaybook()
    defensive = DefensiveExitPlaybook()
    if playbook == "single":
        router = PlaybookRouter(playbooks=[single], default=single)
    elif playbook == "scale-in":
        router = PlaybookRouter(playbooks=[scale], default=scale)
    else:
        router = PlaybookRouter(
            playbooks=[core, trend, range_pb, bear, defensive, scale, single],
            default=single,
        )

    constraints = PortfolioConstraints(
        max_open_positions=int(max_positions),
        max_total_exposure_pct=100.0 if playbook == "auto" else 50.0,
        max_total_risk_pct=float(max_total_risk_pct),
        max_position_pct=100.0 if playbook == "auto" else max(10.0, float(lifecycle_total_pct)),
    )
    portfolio = PortfolioEngine(constraints)
    engine = LifecycleEngine(
        router=router,
        portfolio=portfolio,
        execution_cfg=ExecutionConfig(slippage_bps=float(slippage_bps)),
        lookback=60,
    )

    # Lifecycle should approximate live pipeline by default, so keep review on
    # even if user did not pass --pipeline-review.
    review_stats = new_pipeline_review_stats()
    detector = LifecycleSignalDetector(
        symbol,
        df,
        setups=setups,
        use_money_flow=use_money_flow,
        money_flow_min_score=min_score,
        money_flow_params=mf_params,
        pipeline_review=True if not pipeline_review else pipeline_review,
        pipeline_review_stats=review_stats,
        rr_ratio=rr,
        sector_model=sector_model,
    )

    result = engine.run_symbol(
        symbol,
        df,
        signal_detector=detector,
        from_date=from_date,
        to_date=to_date,
    )
    closed = result["closed_positions"]
    print_review_stats(review_stats)
    _print_lifecycle_report(
        closed,
        audit_log=result["audit_log"],
        label=f"{symbol} lifecycle/{playbook}",
        from_date=from_date,
        to_date=to_date,
        benchmark_df=df,
    )
    if not no_save:
        print("[lifecycle_bt] CSV save chưa nối cho Position lifecycle; dùng --no-save hoặc đọc audit log trong memory.")


def _build_lifecycle_sector_model(
    *,
    symbol,
    symbol_df,
    history_start,
    to_date,
    get_ohlcv_history,
    sector_map,
):
    from multiagents_trading_assistant.backtest.sector_rotation import SectorRotationModel

    symbols = sorted({s for names in sector_map.values() for s in names})
    ohlcv_map = {symbol.upper(): symbol_df}
    fetch_symbols = [s for s in symbols if s != symbol.upper()]
    print(f"[lifecycle_bt] Building sector rotation model ({len(symbols)} symbols, 8 workers)...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut_map = {ex.submit(get_ohlcv_history, sym, history_start, to_date): sym for sym in fetch_symbols}
        for fut in as_completed(fut_map):
            sym = fut_map[fut]
            try:
                df = fut.result()
                if not df.empty:
                    ohlcv_map[sym] = df
            except Exception as e:
                print(f"[lifecycle_bt] sector fetch skip {sym}: {e}")
    return SectorRotationModel(sector_map, ohlcv_map)


def _print_lifecycle_report(closed_positions, *, audit_log, label, from_date, to_date, benchmark_df=None) -> None:
    from multiagents_trading_assistant.backtest.lifecycle_metrics import (
        buy_hold_benchmark,
        portfolio_metrics,
        position_metrics,
    )

    sep = "═" * 62
    print()
    print(sep)
    print(f"  LIFECYCLE BACKTEST: {label}   {from_date} → {to_date}")
    print(sep)
    bh = buy_hold_benchmark(benchmark_df, from_date, to_date) if benchmark_df is not None else {}
    if not closed_positions:
        print("  Không có position nào đóng trong giai đoạn này.")
        if bh:
            print(
                "  Buy & hold benchmark : "
                f"{bh['open_to_close_pct']:+.2f}% open-to-close "
                f"({bh['start_date']} {bh['first_open']} -> {bh['end_date']} {bh['last_close']})"
            )
        rejects = [e for e in audit_log if str(e.get("status", "")).startswith("reject")]
        portfolio_rejects = [e for e in audit_log if "portfolio_reject" in str(e.get("status", ""))]
        print(f"  Audit events: {len(audit_log)} | rejects={len(rejects)} | portfolio_rejects={len(portfolio_rejects)}")
        return

    m = portfolio_metrics(closed_positions, label=label)
    pf_str = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "∞"
    print()
    print("  TỔNG QUAN")
    print(f"    Tổng positions       : {m['total_positions']}")
    print(f"    Win rate             : {m['win_rate_pct']:.1f}%  ({m['win_count']}W / {m['loss_count']}L)")
    print(f"    Avg PnL / position   : {m['avg_pnl_nav_pct']:+.3f}% NAV")
    print(f"    Avg PnL on capital   : {m['avg_pnl_capital_pct']:+.2f}%")
    print(f"    Total realized       : {m['total_realized_nav_pct']:+.3f}% NAV")
    if bh:
        relative = m["total_realized_nav_pct"] - bh["open_to_close_pct"]
        print(
            f"    Buy & hold benchmark : {bh['open_to_close_pct']:+.2f}% "
            f"({bh['start_date']} open {bh['first_open']} -> {bh['end_date']} close {bh['last_close']})"
        )
        print(f"    Excess vs buy & hold : {relative:+.2f}% NAV")
    print(f"    Profit factor        : {pf_str}")
    print(f"    Best / Worst         : {m['best_position_nav']:+.3f}% / {m['worst_position_nav']:+.3f}% NAV")
    prog = m["stage_progression"]
    print(
        "    Stage progression    : "
        f"probe_only={prog['PROBE_ONLY']}, added={prog['REACHED_ADDED']}, full={prog['REACHED_FULL']}"
    )

    print()
    print("  POSITIONS")
    print(f"    {'Open':>10} {'Close':>10} {'Legs':>4} {'Buys':>4} {'MaxNAV':>7} {'PnL NAV':>9} {'Exit':>14}")
    print("    " + "─" * 70)
    for pos in closed_positions[-15:]:
        pm = position_metrics(pos)
        print(
            f"    {str(pm['open_date'])[:10]:>10} {str(pm['close_date'])[:10]:>10} "
            f"{pm['num_legs']:>4} {pm['num_buys']:>4} {pm['max_nav_pct']:>6.2f}% "
            f"{pm['realized_pnl_nav_pct']:>+8.3f}% {pm['exit_reason']:>14}"
        )

    rejects = [e for e in audit_log if str(e.get("status", "")).startswith("reject")]
    portfolio_rejects = [e for e in audit_log if "portfolio_reject" in str(e.get("status", ""))]
    print()
    print(f"  Audit events: {len(audit_log)} | rejects={len(rejects)} | portfolio_rejects={len(portfolio_rejects)}")
