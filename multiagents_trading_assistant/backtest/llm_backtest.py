"""LLM-Driven Backtest Engine.

Walk-forward qua ngày lịch sử. Mỗi ngày:
  1. Screener với as_of_date filter → top candidates
  2. TradingAgentsVN.propagate() với as_of_date → TradePlan (có cache)
  3. validate_trade_plan() → reject nếu invalid
  4. evaluate_plan() score → reject nếu < threshold
  5. simulate_llm_entry() với next_day OHLCV → LLMFillResult
  6. Walk-forward SL/TP/timeout → ghi kết quả

Anti-lookahead:
  - OHLCV trim tại as_of_date (screener + tools)
  - News filter published_before as_of_date (tools)
  - Memory chỉ đọc entries < as_of_date (graph)
  - Entry tại ATO ngày kế (execution_date = next trading day)

Train/Test split:
  - train_period_end: set None để bỏ qua split
  - Chỉ log + count performance từ ngày sau train_period_end
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import pandas as pd

from multiagents_trading_assistant.backtest.execution import (
    LLMExecutionConfig,
    simulate_llm_entry,
)
from multiagents_trading_assistant.backtest.llm_cache import (
    INDICATOR_VERSION,
    PROMPT_VERSION,
    build_cache_key,
    compute_data_hash,
    get_cached,
    log_decision,
    set_cache,
)
from multiagents_trading_assistant.backtest.plan_scorer import SCORE_THRESHOLD, evaluate_plan
from multiagents_trading_assistant.backtest.validator import validate_trade_plan
from multiagents_trading_assistant.agentic import (
    build_strategy_signal_from_candidate,
)
from multiagents_trading_assistant.agentic.evidence_builder import EvidenceBuilder
from multiagents_trading_assistant.tradingagents_vn.schema import TradePlan


# ── Max hold theo nhóm setup ──────────────────────────────────────────────────

_MAX_HOLD_BY_SETUP: dict[str, int] = {
    "BREAKOUT": 15, "NR7": 3, "INSIDE_BAR": 3, "HAMMER": 3,
    "PIN_BAR": 3, "BULLISH_ENGULFING": 3, "MOMENTUM_SURGE": 5,
    "BB_SQUEEZE": 5, "SPRING": 5, "RETEST": 5, "MACD_CROSSOVER": 5,
    "FLAG_PENNANT": 5, "DOUBLE_BOTTOM": 5, "TREND_PULLBACK": 5,
    "BREAKOUT_RETEST_ENTRY": 5, "GOLDEN_CROSS": 8, "RSI_BOUNCE": 8,
    "MA_PULLBACK": 8,
}
_DEFAULT_MAX_HOLD = 5


@dataclass
class LLMBacktestConfig:
    # Universe
    symbol: str | None = None
    universe: str | None = None          # "vn30" | "vn100" | "liquid"
    # Date range
    from_date: str = "2024-01-01"
    to_date: str | None = None           # None = today
    train_period_end: str | None = None  # Đóng băng prompt sau ngày này; None = không split
    # Capital
    initial_capital: float = 1_000_000_000  # 1 tỷ VND
    # Execution
    execution: LLMExecutionConfig = field(default_factory=LLMExecutionConfig)
    max_positions: int = 5
    # Graph
    cheap_mode: bool = True              # Haiku cho tất cả agent (rẻ ~$0.03/mã)
    max_invest_rounds: int = 1           # Rút ngắn để tiết kiệm tiền khi backtest
    max_risk_rounds: int = 1
    # Score
    score_threshold: float = SCORE_THRESHOLD
    # Cache
    use_cache: bool = True
    # Screener
    max_candidates_per_day: int = 5      # Chỉ deep analyze top N candidates
    # Output
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    verbose: bool = True


@dataclass
class LLMTrade:
    symbol: str
    signal_date: str
    execution_date: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_pct: float
    max_hold: int
    setup_type: str
    plan_score: float
    exit_date: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""    # SL | TP | TIMEOUT
    pnl_pct: float = 0.0     # % return trên vốn đầu vào
    bars_held: int = 0
    is_test_period: bool = True   # False nếu trong train_period


def _trading_days(from_date: str, to_date: str) -> list[str]:
    """Sinh list ngày giao dịch (bỏ Sat/Sun). Không lọc nghỉ lễ VN — phase 1."""
    start = date.fromisoformat(from_date)
    end = date.fromisoformat(to_date)
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:   # Mon–Fri
            days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return days


def _next_trading_day(d: str) -> str:
    """Ngày giao dịch kế tiếp (bỏ Sat/Sun)."""
    cur = date.fromisoformat(d) + timedelta(days=1)
    while cur.weekday() >= 5:
        cur += timedelta(days=1)
    return cur.strftime("%Y-%m-%d")


def _get_bar(df: pd.DataFrame, target_date: str) -> dict | None:
    """Lấy OHLCV bar tại target_date."""
    if df is None or df.empty:
        return None
    col = "date" if "date" in df.columns else None
    if col:
        rows = df[df[col].astype(str) == target_date]
    else:
        rows = df[df.index.astype(str) == target_date]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "open": float(row.get("open", 0)),
        "high": float(row.get("high", 0)),
        "low": float(row.get("low", 0)),
        "close": float(row.get("close", 0)),
        "volume": float(row.get("volume", 0)),
    }


def _simulate_trade_walk(
    symbol: str,
    entry: float,
    sl: float,
    tp: float,
    max_hold: int,
    execution_date: str,
    ohlcv_df: pd.DataFrame,
    settlement_days: int = 2,
) -> tuple[str, float, str, int]:
    """Walk-forward từ execution_date đến khi hit SL/TP/timeout.

    Returns:
        (exit_date, exit_price, exit_reason, bars_held)
    """
    all_days = _trading_days(execution_date, ohlcv_df["date"].max() if "date" in ohlcv_df.columns else "2099-12-31")
    bars_held = 0

    for i, d in enumerate(all_days):
        if i == 0:
            continue   # skip execution_date itself (entry bar)
        bar = _get_bar(ohlcv_df, d)
        if bar is None:
            continue

        bars_held += 1
        low = bar["low"]
        high = bar["high"]
        close = bar["close"]

        # SL hit — fill at min(open, sl) to handle gap-down opens realistically
        if low <= sl:
            open_bar = float(bar.get("open", sl))
            return d, min(open_bar, sl), "SL", bars_held

        # TP hit — fill at max(open, tp) to handle gap-up opens realistically
        if high >= tp:
            open_bar = float(bar.get("open", tp))
            return d, max(open_bar, tp), "TP", bars_held

        # Timeout
        if bars_held >= max_hold:
            # Có thể bán T+settlement đúng rule, dùng close làm proxy
            if bars_held >= settlement_days:
                return d, close, "TIMEOUT", bars_held

    # Hết data → close cuối
    last_day = all_days[-1] if all_days else execution_date
    last_bar = _get_bar(ohlcv_df, last_day)
    exit_price = last_bar["close"] if last_bar else entry
    return last_day, exit_price, "DATA_END", bars_held


def run_llm_backtest(config: LLMBacktestConfig) -> list[LLMTrade]:
    """Main backtest loop.

    Returns:
        List[LLMTrade] — tất cả trades (train + test period).
        Filter is_test_period=True để lấy kết quả đánh giá thực.
    """
    from multiagents_trading_assistant.fetcher import get_ohlcv_history, get_ohlcv
    from multiagents_trading_assistant.screener.trade_screener import run_screener
    from multiagents_trading_assistant.tradingagents_vn.graph import TradingAgentsVN

    to_date = config.to_date or date.today().strftime("%Y-%m-%d")
    trading_days = _trading_days(config.from_date, to_date)

    if config.verbose:
        print(f"\n{'='*60}")
        print(f"[llm_backtest] run_id={config.run_id}")
        print(f"[llm_backtest] {config.from_date} → {to_date} ({len(trading_days)} ngày)")
        print(f"[llm_backtest] cheap={config.cheap_mode} | cache={config.use_cache}")
        print(f"[llm_backtest] max_candidates/day={config.max_candidates_per_day}")
        if config.train_period_end:
            print(f"[llm_backtest] train≤{config.train_period_end} | test>{config.train_period_end}")
        print(f"{'='*60}\n")

    graph = TradingAgentsVN(
        cheap=config.cheap_mode,
        max_invest_rounds=config.max_invest_rounds,
        max_risk_rounds=config.max_risk_rounds,
    )

    # Pre-fetch VNINDEX history đủ dài cho toàn bộ backtest range (1 lần duy nhất)
    hist_start = (date.fromisoformat(config.from_date) - timedelta(days=400)).strftime("%Y-%m-%d")
    if config.verbose:
        print(f"[llm_backtest] Pre-fetching VNINDEX {hist_start} → {to_date}...")
    vnindex_history = get_ohlcv_history("VNINDEX", hist_start, to_date)
    if vnindex_history.empty:
        print("[llm_backtest] WARNING: không lấy được VNINDEX history — screener sẽ tự fetch từng ngày")
        vnindex_history = None

    # Pre-fetch OHLCV cho toàn bộ universe từ hist_start → to_date
    # Tránh get_ohlcv_batch() dùng cache ngày-hôm-nay, sẽ rỗng khi trim sang date lịch sử
    from multiagents_trading_assistant.services.data_service import get_liquid_symbols
    universe_symbols = config.symbol.split(",") if config.symbol else get_liquid_symbols(min_avg_vol=300_000)
    if config.verbose:
        print(f"[llm_backtest] Pre-fetching OHLCV {len(universe_symbols)} symbols {hist_start} → {to_date}...")

    # Fetch tuần tự (tránh rate limit); ohlcv_cache dùng lại toàn vòng lặp
    ohlcv_cache: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(universe_symbols):
        try:
            df = get_ohlcv_history(sym, hist_start, to_date)
            if df is not None and not df.empty:
                ohlcv_cache[sym] = df
        except Exception:
            pass
        if config.verbose and (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(universe_symbols)} fetched")
    if config.verbose:
        print(f"[llm_backtest] Fetched {len(ohlcv_cache)}/{len(universe_symbols)} symbols with data\n")

    open_positions: dict[str, LLMTrade] = {}   # symbol → trade đang mở
    all_trades: list[LLMTrade] = []

    model_label = "haiku" if config.cheap_mode else "sonnet"
    is_test = lambda d: (config.train_period_end is None) or (d > config.train_period_end)

    for signal_date in trading_days:
        execution_date = _next_trading_day(signal_date)
        if config.verbose:
            print(f"\n[{signal_date}] open_positions={list(open_positions.keys())}")

        # ── Update open positions với next-day prices ──────────────────────
        closed_symbols = []
        for sym, trade in open_positions.items():
            df = ohlcv_cache.get(sym)
            if df is None:
                continue
            exit_date, exit_price, reason, bars = _simulate_trade_walk(
                sym, trade.entry_price, trade.stop_loss, trade.take_profit,
                trade.max_hold, trade.execution_date, df,
                config.execution.settlement_days,
            )
            if reason != "TIMEOUT" or bars >= trade.max_hold:
                pnl = (exit_price - trade.entry_price) / trade.entry_price * 100
                trade.exit_date = exit_date
                trade.exit_price = exit_price
                trade.exit_reason = reason
                trade.pnl_pct = round(pnl, 3)
                trade.bars_held = bars
                all_trades.append(trade)
                closed_symbols.append(sym)
                if config.verbose:
                    print(f"  CLOSE {sym}: {reason} | pnl={pnl:+.2f}% ({bars}bars)")
        for sym in closed_symbols:
            del open_positions[sym]

        # ── Skip nếu đã đủ positions ───────────────────────────────────────
        if len(open_positions) >= config.max_positions:
            continue

        # ── Screener ───────────────────────────────────────────────────────
        # Pass slice VNINDEX đến signal_date để tránh fetch lại
        vni_slice = None
        if vnindex_history is not None and not vnindex_history.empty:
            if "date" in vnindex_history.columns:
                vni_slice = vnindex_history[vnindex_history["date"].astype(str) <= signal_date]

        try:
            _, candidates = run_screener(
                symbols=list(ohlcv_cache.keys()),
                max_candidates=config.max_candidates_per_day * 3,
                as_of_date=signal_date,
                vnindex_df=vni_slice,
                ohlcv_map=ohlcv_cache,
            )
        except Exception as e:
            if config.verbose:
                print(f"  [screener] error: {e}")
            continue

        # Skip nếu không nên trade (downtrend mạnh)
        if candidates and not candidates[0].market_context.should_trade:
            if config.verbose:
                print(f"  [screener] should_trade=False → skip ngày này")
            continue

        top = candidates[:config.max_candidates_per_day]

        # ── Phân tích từng candidate ───────────────────────────────────────
        for candidate in top:
            sym = candidate.symbol
            if sym in open_positions:
                continue

            # Lấy OHLCV từ pre-fetched cache
            df = ohlcv_cache.get(sym)
            if df is None or df.empty:
                continue

            # current_price tại as_of_date
            bar_today = _get_bar(df, signal_date)
            current_price = bar_today["close"] if bar_today else 0.0

            # data_hash
            df_filtered = df[df["date"].astype(str) <= signal_date] if "date" in df.columns else df
            data_hash = compute_data_hash(df_filtered, signal_date)

            # Build cache key
            cache_key = build_cache_key(
                sym, signal_date, "deep", model_label, data_hash,
                PROMPT_VERSION, INDICATOR_VERSION,
            )

            # Try cache first
            plan_dict = None
            cache_hit = False
            if config.use_cache:
                cached = get_cached(cache_key)
                if cached:
                    plan_dict = cached.get("plan")
                    cache_hit = True
                    if config.verbose:
                        print(f"  {sym}: cache hit")

            plan: TradePlan | None = None

            if plan_dict:
                try:
                    plan = TradePlan.model_validate(plan_dict)
                except Exception:
                    plan = None

            if plan is None:
                # Gọi LLM
                try:
                    strategy_signal = build_strategy_signal_from_candidate(
                        candidate,
                        signal_date,
                        source="core3" if (candidate.indicators or {}).get("edge_strategy_analysis", {}).get("passed") else "broad_screener",
                        require_edge_passed=False,
                    )
                    evidence_packet = EvidenceBuilder(
                        signal_date, backtest_mode=True
                    ).build(candidate, signal=strategy_signal)
                    plan = graph.propagate(
                        sym,
                        signal_date,
                        as_of_date=signal_date,
                        setup_type=candidate.setup_type,
                        strategy_signal=strategy_signal.model_dump() if strategy_signal else None,
                        evidence_packet=evidence_packet.model_dump(),
                    )
                except Exception as e:
                    if config.verbose:
                        print(f"  {sym}: LLM error — {e}")
                    continue

                if config.use_cache and plan:
                    set_cache(cache_key, {"plan": plan.model_dump()})

            # Validate
            if plan is None:
                log_decision(config.run_id, sym, signal_date, None, 0.0, False, cache_hit)
                continue

            vr = validate_trade_plan(plan, current_price)
            if not vr:
                if config.verbose:
                    print(f"  {sym}: validate FAIL — {vr.errors}")
                log_decision(config.run_id, sym, signal_date, plan.model_dump(), 0.0, False, cache_hit)
                continue

            # Score
            features = dict(candidate.indicators)
            features["market_trend"] = candidate.market_context.reference_trend
            score, should_trade = evaluate_plan(plan, features, config.score_threshold)

            log_decision(config.run_id, sym, signal_date, plan.model_dump(), score, should_trade, cache_hit)

            if not should_trade:
                if config.verbose:
                    print(f"  {sym}: score={score:.3f} < {config.score_threshold} → skip")
                continue

            if plan.action != "MUA":
                continue

            # Execute
            bar_exec = _get_bar(df, execution_date)
            if bar_exec is None:
                if config.verbose:
                    print(f"  {sym}: no bar at execution_date {execution_date}")
                continue

            bar_exec["prev_close"] = bar_today["close"] if bar_today else bar_exec["open"]
            avg_vol = float(df_filtered["volume"].tail(20).mean()) if not df_filtered.empty else 0.0

            fill = simulate_llm_entry(
                entry_zone=plan.entry_zone,
                stop_loss=plan.stop_loss,
                position_pct=plan.position_pct,
                nav=config.initial_capital,
                next_bar=bar_exec,
                avg_volume=avg_vol,
                cfg=config.execution,
            )

            if not fill.filled:
                if config.verbose:
                    print(f"  {sym}: fill FAIL — {fill.reason}")
                continue

            max_hold = _MAX_HOLD_BY_SETUP.get(plan.setup_type, _DEFAULT_MAX_HOLD)
            trade = LLMTrade(
                symbol=sym,
                signal_date=signal_date,
                execution_date=execution_date,
                entry_price=fill.price,
                stop_loss=plan.stop_loss,
                take_profit=plan.take_profit,
                position_pct=plan.position_pct,
                max_hold=max_hold,
                setup_type=plan.setup_type,
                plan_score=score,
                is_test_period=is_test(signal_date),
            )
            open_positions[sym] = trade

            if config.verbose:
                print(
                    f"  OPEN {sym}: entry={fill.price} SL={plan.stop_loss} "
                    f"TP={plan.take_profit} score={score:.3f}"
                )

    # Close tất cả positions còn mở ở cuối
    for sym, trade in open_positions.items():
        df = ohlcv_cache.get(sym)
        if df is not None and not df.empty:
            last_close = float(df["close"].iloc[-1])
        else:
            last_close = trade.entry_price
        pnl = (last_close - trade.entry_price) / trade.entry_price * 100
        trade.exit_date = to_date
        trade.exit_price = last_close
        trade.exit_reason = "END_OF_BACKTEST"
        trade.pnl_pct = round(pnl, 3)
        trade.bars_held = -1
        all_trades.append(trade)

    if config.verbose:
        _print_summary(all_trades, config)

    return all_trades


def _print_summary(trades: list[LLMTrade], config: LLMBacktestConfig) -> None:
    test_trades = [t for t in trades if t.is_test_period]
    all_t = test_trades or trades

    print(f"\n{'='*60}")
    print(f"[llm_backtest] SUMMARY run_id={config.run_id}")
    print(f"  Total trades (test period): {len(all_t)}")
    if not all_t:
        print("  Không có trade nào.")
        return

    wins = [t for t in all_t if t.pnl_pct > 0]
    losses = [t for t in all_t if t.pnl_pct <= 0]
    win_rate = len(wins) / len(all_t) * 100
    avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0
    total_pnl = sum(t.pnl_pct * t.position_pct for t in all_t)

    print(f"  Win rate: {win_rate:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg win: {avg_win:+.2f}% | Avg loss: {avg_loss:+.2f}%")
    print(f"  Total PnL (weighted): {total_pnl:+.2f}%")

    exit_counts = {}
    for t in all_t:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
    print(f"  Exit reasons: {exit_counts}")
    print(f"{'='*60}\n")
