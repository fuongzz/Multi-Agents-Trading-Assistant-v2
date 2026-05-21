"""pipeline_runner.py — APScheduler runner cho cả hai pipeline.

Schedule:
  Investment : Thứ 2 hàng tuần lúc 08:00
  Trade      : Hàng ngày lúc 08:30 (trước ATO 09:00)

Chế độ --no-schedule / run_once=True: chạy một lần rồi thoát (dùng cho test).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo

from multiagents_trading_assistant import database as db
from multiagents_trading_assistant.orchestrator.investment_graph import run_pipeline as run_invest
from multiagents_trading_assistant.orchestrator.trade_graph import run_pipeline as run_trade
from multiagents_trading_assistant.screener.invest_screener import run_screener as invest_screener
from multiagents_trading_assistant.screener.trade_screener import run_screener as trade_screener
from multiagents_trading_assistant.services import output_service
from multiagents_trading_assistant.services.output_service import send_pipeline_alert
from multiagents_trading_assistant.services.memory_service import save_trade_decision
from multiagents_trading_assistant.orchestrator.session_monitor import run_session_monitor
from multiagents_trading_assistant.bob.simulator import run_strategy_development_meeting
from multiagents_trading_assistant.memory.strategy_memory import StrategyMemory
from multiagents_trading_assistant.bob.reward_tracker import RewardTracker
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_LIVE_EDGE_FAMILY, get_edge_strategy_signals

_reward_tracker = RewardTracker()

_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _today() -> str:
    return datetime.now(tz=_VN_TZ).strftime("%Y-%m-%d")


def _load_active_strategies() -> list[dict]:
    """Load active strategies from ℳₛ for injection into Otto's context.

    Returns a list of dicts (serialisable) sorted by profit_factor desc.
    Returns [] gracefully if memory file does not exist yet.
    """
    try:
        mem = StrategyMemory()
        records = mem.get_active_strategies()
        return [
            {
                "setup": r.setup,
                "regime": r.regime,
                "win_rate": r.win_rate,
                "profit_factor": r.profit_factor,
                "avg_rr": r.avg_rr,
                "sample_size": r.sample_size,
                "updated_at": r.updated_at,
            }
            for r in records
        ]
    except Exception as e:
        print(f"[runner] ℳₛ load failed: {e}")
        return []


def run_bob_strategy_meeting() -> None:
    """Bob's weekly Strategy Development Meeting — updates ℳₛ.

    Paper: "Bob executes simulated trading backtests on all potential new
    strategies (μ't) ... selects strategies with strongest historical
    performance to form new strategy set (μ')."
    """
    print(f"\n{'=' * 60}\n[runner] BOB STRATEGY MEETING — {_today()}\n{'=' * 60}")
    try:
        active = run_strategy_development_meeting()
        print(f"[runner] Bob complete — {len(active)} active strategies in ℳₛ")
    except Exception as e:
        print(f"[runner] Bob meeting FAIL: {e}")
        send_pipeline_alert("bob", e, date=_today())


def _update_ohlcv_daily() -> None:
    """Daily OHLCV data update — runs after market close (15:35).

    Fetches new trading data and appends to ohlcv_master.parquet.
    """
    print(f"\n{'=' * 60}\n[runner] OHLCV DAILY UPDATE — {_today()}\n{'=' * 60}")
    try:
        from pathlib import Path

        script_path = Path(__file__).parent.parent.parent / "scripts" / "update_ohlcv_daily.py"
        subprocess.run([sys.executable, str(script_path)], check=False, timeout=600)
        print("[runner] OHLCV update complete")
    except Exception as e:
        print(f"[runner] OHLCV update FAIL: {e}")


def _update_vnindex() -> None:
    """Append latest VNINDEX bar to index_master.parquet."""
    print(f"\n{'-' * 60}\n[runner] VNINDEX update — {_today()}\n{'-' * 60}")
    try:
        from pathlib import Path

        import pandas as pd

        from multiagents_trading_assistant.fetcher import get_vnindex

        vni = get_vnindex(n_days=10)
        if vni is None or vni.empty:
            print("[runner] VNINDEX fetch returned empty — skip")
            return
        vni = vni.copy()
        vni["date"] = pd.to_datetime(vni["date"]).dt.normalize()
        vni["symbol"] = "VNINDEX"

        master_path = (
            Path(__file__).parent.parent / "data" / "index_master.parquet"
        )
        if master_path.exists():
            master = pd.read_parquet(master_path)
            master["date"] = pd.to_datetime(master["date"]).dt.normalize()
            existing = set(
                master[master["symbol"].astype(str).str.upper() == "VNINDEX"]["date"]
            )
            new_rows = vni[~vni["date"].isin(existing)][
                ["date", "symbol", "open", "high", "low", "close", "volume"]
            ]
            if new_rows.empty:
                print("[runner] VNINDEX already up to date")
                return
            combined = (
                pd.concat([master, new_rows], ignore_index=True)
                .sort_values(["symbol", "date"])
                .drop_duplicates(["symbol", "date"], keep="last")
                .reset_index(drop=True)
            )
        else:
            combined = vni[
                ["date", "symbol", "open", "high", "low", "close", "volume"]
            ]
        master_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(master_path, index=False)
        latest = combined[combined["symbol"] == "VNINDEX"]["date"].max()
        print(f"[runner] VNINDEX latest: {latest}")
    except Exception as e:
        print(f"[runner] VNINDEX update FAIL: {e}")


def _compute_money_cycle() -> None:
    """Recompute CHDM / DS parquets from latest OHLCV master."""
    print(f"\n{'-' * 60}\n[runner] Money cycle compute — {_today()}\n{'-' * 60}")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "multiagents_trading_assistant.research.money_cycle.cli",
            ],
            check=False,
            timeout=900,
        )
        print("[runner] Money cycle compute done")
    except Exception as e:
        print(f"[runner] Money cycle compute FAIL: {e}")


def _compute_smart_money_trace() -> None:
    """Recompute smart_money_by_symbol parquet (depends on CHDM)."""
    print(f"\n{'-' * 60}\n[runner] Smart money trace compute — {_today()}\n{'-' * 60}")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "multiagents_trading_assistant.research.smart_money_trace.cli",
            ],
            check=False,
            timeout=900,
        )
        print("[runner] Smart money trace done")
    except Exception as e:
        print(f"[runner] Smart money trace FAIL: {e}")


def _run_daily_data_refresh() -> None:
    """Sequential daily data refresh — chạy sau market close hoặc trước ATO.

    Sequence (cần đúng thứ tự dependency):
        1. OHLCV master       (per-symbol price)
        2. VNINDEX master     (market index)
        3. Money cycle        (CHDM, DS — depends on OHLCV)
        4. Smart money trace  (depends on OHLCV + CHDM)
    """
    print(f"\n{'=' * 60}\n[runner] DAILY DATA REFRESH — {_today()}\n{'=' * 60}")
    _update_ohlcv_daily()
    _update_vnindex()
    _compute_money_cycle()
    _compute_smart_money_trace()
    print(f"[runner] Daily data refresh COMPLETE\n")


def _run_daily_data_refresh_if_stale() -> None:
    """Morning catch-up: only refresh if EOD job didn't run (e.g. PC was off).

    Idempotent — safe to call. Skips if all data layers are fresh.
    """
    ok, stale = _check_data_freshness()
    if ok:
        print(f"[runner] Morning data check — all fresh, skip refresh")
        return
    print(f"[runner] Morning catch-up triggered. Stale layers: {stale}")
    _run_daily_data_refresh()


def _check_data_freshness(date: str | None = None) -> tuple[bool, list[str]]:
    """Verify all data layers are at most 1 trading day stale.

    Returns (ok, stale_layers). Used as gate before trade pipeline scan.
    """
    from datetime import timedelta
    from pathlib import Path

    import pandas as pd

    target = pd.Timestamp(date or _today()).normalize()
    # Allow data 3 calendar days old (weekend + Monday morning before refresh)
    cutoff = target - pd.Timedelta(days=3)

    repo_root = Path(__file__).parent.parent.parent
    checks = {
        "ohlcv_master": repo_root / "multiagents_trading_assistant/data/ohlcv_master.parquet",
        "index_master": repo_root / "multiagents_trading_assistant/data/index_master.parquet",
        "money_cycle/chdm": repo_root / "data/research/money_cycle/chdm_by_symbol.parquet",
        "money_cycle/ds": repo_root / "data/research/money_cycle/ds_by_symbol.parquet",
        "smart_money_trace": repo_root
        / "data/research/smart_money_trace/smart_money_by_symbol.parquet",
    }
    stale: list[str] = []
    for name, path in checks.items():
        if not path.exists():
            stale.append(f"{name} (missing file)")
            continue
        try:
            df = pd.read_parquet(path, columns=["date"])
            latest = pd.to_datetime(df["date"]).max().normalize()
            if latest < cutoff:
                stale.append(f"{name} (latest={latest.date()})")
        except Exception as e:
            stale.append(f"{name} (read error: {e})")
    return (not stale, stale)


# ──────────────────────────────────────────────
# Investment pipeline
# ──────────────────────────────────────────────

def run_investment_pipeline(
    symbol: str | None = None,
    date: str | None = None,
) -> list[dict]:
    date = date or _today()
    print(f"\n{'=' * 60}\n[runner] INVESTMENT PIPELINE — {date}\n{'=' * 60}")
    try:
        return _run_investment_pipeline_inner(symbol, date)
    except Exception as e:
        send_pipeline_alert("invest", e, date=date)
        raise


def _run_investment_pipeline_inner(
    symbol: str | None,
    date: str,
) -> list[dict]:

    if symbol:
        candidates_sym = [symbol]
    else:
        candidates = invest_screener()
        candidates_sym = [c.symbol for c in candidates]

    results = []
    for sym in candidates_sym:
        print(f"\n[runner] → Invest: {sym}")
        try:
            state = run_invest(symbol=sym, date=date)
            results.append(state)
        except Exception as e:
            print(f"[runner] Invest {sym} FAIL: {e}")
            send_pipeline_alert("invest", e, symbol=sym, date=date)

    _print_invest_summary(results, date)
    return results


# ──────────────────────────────────────────────
# Trade pipeline
# ──────────────────────────────────────────────

import os as _os

_GRAPH_MODE = _os.environ.get("MATA_GRAPH_MODE", "fast").lower()
_EDGE_STRATEGY_ONLY = _os.environ.get("MATA_EDGE_STRATEGY_ONLY", DEFAULT_LIVE_EDGE_FAMILY).strip()


def run_trade_pipeline(
    symbol: str | None = None,
    date: str | None = None,
) -> list[dict]:
    date = date or _today()
    db.init_db()
    print(f"\n{'=' * 60}\n[runner] TRADE PIPELINE — {date} | mode={_GRAPH_MODE}\n{'=' * 60}")

    # Data freshness gate — refuse to scan with stale data (signals will all
    # fail filter chain due to NaN in SMT/CHDM, masquerading as "no opportunity")
    ok, stale = _check_data_freshness(date)
    if not ok:
        msg = f"Trade pipeline aborted — stale data layers: {stale}"
        print(f"[runner] ⚠ {msg}")
        try:
            output_service.send_to_channel(
                "trade",
                f"⚠️ **{date}** — Trade pipeline tạm dừng: data stale.\n"
                f"Stale: {', '.join(stale)}\nĐang thử refresh.",
            )
        except Exception:
            pass
        _run_daily_data_refresh()
        ok2, stale2 = _check_data_freshness(date)
        if not ok2:
            print(f"[runner] Data refresh failed to fix: {stale2}. Skip scan.")
            send_pipeline_alert(
                "trade",
                RuntimeError(f"Stale data after refresh: {stale2}"),
                date=date,
            )
            return []

    try:
        if _GRAPH_MODE == "deep":
            return _run_trade_pipeline_deep(symbol, date)
        return _run_trade_pipeline_inner(symbol, date)
    except Exception as e:
        send_pipeline_alert("trade", e, date=date)
        raise


def _run_trade_pipeline_deep(
    symbol: str | None,
    date: str,
) -> list[dict]:
    """Deep mode: screener → top 3–5 candidates → TradingAgentsVN full debate → batch Discord."""
    from multiagents_trading_assistant.agentic import (
        build_strategy_signal_from_candidate,
    )
    from multiagents_trading_assistant.agentic.evidence_builder import EvidenceBuilder
    from multiagents_trading_assistant.tradingagents_vn.graph import TradingAgentsVN
    from multiagents_trading_assistant.backtest.validator import validate_trade_plan
    from multiagents_trading_assistant.backtest.plan_scorer import evaluate_plan
    from multiagents_trading_assistant.services.data_service import get_ohlcv

    _DEEP_TOP_N = int(_os.environ.get("MATA_DEEP_TOP_N", "3"))
    cheap = _os.environ.get("MATA_DEEP_CHEAP", "true").lower() == "true"

    graph = TradingAgentsVN(cheap=cheap, max_invest_rounds=2, max_risk_rounds=1)

    if symbol:
        candidates_sym = [symbol.upper()]
        candidate_by_symbol = {}
    else:
        _market_ctx, candidates = trade_screener()
        if _EDGE_STRATEGY_ONLY:
            candidates = _filter_edge_strategy_candidates(candidates, _EDGE_STRATEGY_ONLY)
        if not _market_ctx.should_trade:
            print(f"[runner:deep] should_trade=False — skip")
            output_service.send_to_channel(
                "trade",
                f"📊 **{date}** — Thị trường không thuận, deep mode bỏ qua phiên hôm nay.\n"
                f"Trend: {_market_ctx.reference_trend} | VNI: {_market_ctx.vni_change_pct:+.2f}%",
            )
            return []
        selected_candidates = candidates[:_DEEP_TOP_N]
        candidate_by_symbol = {c.symbol.upper(): c for c in selected_candidates}
        candidates_sym = [c.symbol for c in selected_candidates]

    print(f"[runner:deep] Phân tích sâu: {candidates_sym}")

    results = []
    mua_plans = []

    for sym in candidates_sym:
        print(f"\n[runner:deep] → {sym}")
        try:
            candidate = candidate_by_symbol.get(sym.upper())
            strategy_signal = None
            evidence_packet = None
            setup_type = "UNKNOWN"
            features = {}
            if candidate is not None:
                strategy_signal = build_strategy_signal_from_candidate(candidate, date)
                if strategy_signal is None:
                    print(f"[runner:deep] {sym}: skip — không có core3 StrategySignal")
                    continue
                evidence_packet = EvidenceBuilder(date, backtest_mode=False).build(
                    candidate, signal=strategy_signal
                )
                setup_type = strategy_signal.setup_type
                features = dict(candidate.indicators or {})
                features["market_trend"] = candidate.market_context.reference_trend

            # Lấy current price để validate
            df = get_ohlcv(sym, 5)
            current_price = float(df["close"].iloc[-1]) if not df.empty else 0.0

            plan = graph.propagate(
                sym,
                signal_date=date,
                setup_type=setup_type,
                strategy_signal=strategy_signal.model_dump() if strategy_signal else None,
                evidence_packet=evidence_packet.model_dump() if evidence_packet else None,
            )

            if plan is None:
                print(f"[runner:deep] {sym}: không parse được TradePlan")
                continue

            vr = validate_trade_plan(plan, current_price)
            if not vr:
                print(f"[runner:deep] {sym}: validate fail — {vr.errors}")
                continue

            score, should_trade = evaluate_plan(plan, features)
            print(f"[runner:deep] {sym}: action={plan.action} score={score:.3f}")

            state_dict = {
                "symbol": sym,
                "date": date,
                "strategy_signal": strategy_signal.model_dump() if strategy_signal else None,
                "plan": plan.model_dump(),
                "score": score,
                "should_trade": should_trade,
            }
            results.append(state_dict)

            if should_trade and plan.action == "MUA":
                mua_plans.append((sym, plan, score))

        except Exception as e:
            print(f"[runner:deep] {sym} FAIL: {e}")
            send_pipeline_alert("trade", e, symbol=sym, date=date)

    # Gửi Discord: 1 summary + detail top 1–3 MUA
    _send_deep_mode_discord(date, results, mua_plans)
    return results


def _send_deep_mode_discord(
    date: str,
    all_results: list[dict],
    mua_plans: list,
) -> None:
    """Gửi 1 summary embed + detail cho top 1–3 MUA. Không spam từng mã."""
    analyzed = len(all_results)
    mua_count = len(mua_plans)

    summary_lines = [f"**TradingAgents-VN Deep Mode | {date}**", ""]
    summary_lines.append(f"Đã phân tích: {analyzed} mã | MUA: {mua_count} mã")
    summary_lines.append("")

    for sym, plan, score in mua_plans[:3]:
        entry_low, entry_high = plan.entry_zone
        summary_lines.append(
            f"🟢 **{sym}** | Entry: {entry_low:.1f}–{entry_high:.1f} | "
            f"SL: {plan.stop_loss:.1f} | TP: {plan.take_profit:.1f} | "
            f"R:R: {plan.rr_ratio:.2f} | Score: {score:.2f}"
        )

    if not mua_plans:
        summary_lines.append("Không có tín hiệu MUA đủ chất lượng hôm nay.")

    try:
        output_service.send_to_channel("trade", "\n".join(summary_lines))
    except Exception as e:
        print(f"[runner:deep] Discord send fail: {e}")


def _run_trade_pipeline_inner(
    symbol: str | None,
    date: str,
) -> list[dict]:

    # Retrieve active strategies from ℳₛ for Otto — paper: ϕD(st, ℳret, μt)
    active_strategies = _load_active_strategies()

    if symbol:
        edge = _edge_context_for_symbol(symbol, date)
        if _EDGE_STRATEGY_ONLY and not edge.get("passed"):
            print(f"[runner] {symbol.upper()} skipped: {_EDGE_STRATEGY_ONLY} not passed")
            _print_trade_summary([], date)
            return []
        setup = "BREAKOUT" if edge.get("passed") else "UNKNOWN"
        candidates_sym = [(symbol, setup, {
            "_portfolio_checked": False,
            "active_strategies": active_strategies,
            "edge_strategy_analysis": edge,
        })]
    else:
        _market_ctx, candidates = trade_screener()
        if _EDGE_STRATEGY_ONLY:
            candidates = _filter_edge_strategy_candidates(candidates, _EDGE_STRATEGY_ONLY)
        candidates_sym = []
        for c in candidates:
            ctx = asdict(c.market_context)
            ctx["stock_current_price"] = c.indicators.get("current_price")
            ctx["stock_day_change_pct"] = c.indicators.get("stock_day_change_pct", 0.0)
            ctx["screener_money_flow"] = c.indicators.get("money_flow_analysis", {})
            ctx["avg_vol_20d"] = c.indicators.get("volume_ma20")
            ctx["edge_strategy_analysis"] = c.indicators.get("edge_strategy_analysis", {})
            ctx["active_strategies"] = active_strategies
            candidates_sym.append((c.symbol, c.setup_type, ctx))

    results = []
    for i, (sym, setup, mkt_ctx) in enumerate(candidates_sym):
        # portfolio_monitor chỉ chạy với symbol đầu tiên trong batch mỗi ngày
        mkt_ctx["_portfolio_checked"] = (i > 0)
        print(f"\n[runner] → Trade: {sym} ({setup})")
        try:
            state = run_trade(symbol=sym, setup_type=setup, market_context=mkt_ctx, date=date)
            results.append(state)
            try:
                save_trade_decision(state)
            except Exception as e:
                print(f"[runner] memory save fail ({sym}): {e}")
            # Record entry cho dual-reward tracking
            _record_trade_entry(state, sym, setup, date)
        except Exception as e:
            print(f"[runner] Trade {sym} FAIL: {e}")
            send_pipeline_alert("trade", e, symbol=sym, date=date)

    _print_trade_summary(results, date)
    return results


# ──────────────────────────────────────────────
# Scheduler
# ──────────────────────────────────────────────

def _filter_edge_strategy_candidates(candidates: list, strategy_name: str) -> list:
    allowed = {item.strip() for item in strategy_name.split(",") if item.strip()}
    filtered = []
    for candidate in candidates:
        indicators = getattr(candidate, "indicators", {}) or {}
        edge = indicators.get("edge_strategy_analysis", {}) or {}
        candidate_strategy = str(edge.get("strategy_name") or "")
        candidate_family = str(edge.get("strategy_family") or "")
        if edge.get("passed") and (
            candidate_strategy in allowed
            or candidate_family == strategy_name
            or (not allowed and candidate_strategy == strategy_name)
        ):
            filtered.append(candidate)
    print(f"[runner] Edge-only {strategy_name}: {len(filtered)}/{len(candidates)} candidates kept")
    return filtered


def _edge_context_for_symbol(symbol: str, date: str) -> dict:
    try:
        return get_edge_strategy_signals(
            [symbol],
            as_of_date=date,
            strategy_name=DEFAULT_LIVE_EDGE_FAMILY,
        ).get(symbol.upper(), {})
    except Exception as e:
        print(f"[runner] edge strategy signal fail ({symbol}): {e}")
        return {
            "strategy_name": DEFAULT_LIVE_EDGE_FAMILY,
            "passed": False,
            "available": False,
            "error": str(e),
        }


def start_scheduler() -> None:
    """Khởi APScheduler: invest Thứ 2 08:00, trade hàng ngày 08:30."""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print("[runner] apscheduler not installed — pip install apscheduler")
        return

    scheduler = BlockingScheduler(timezone=_VN_TZ)

    scheduler.add_job(
        run_investment_pipeline,
        CronTrigger(day_of_week="mon", hour=8, minute=0, timezone=_VN_TZ),
        id="invest_weekly",
    )
    scheduler.add_job(
        run_trade_pipeline,
        CronTrigger(hour=8, minute=30, timezone=_VN_TZ),
        id="trade_daily",
    )
    scheduler.add_job(
        run_bob_strategy_meeting,
        CronTrigger(day_of_week="fri", hour=20, minute=0, timezone=_VN_TZ),
        id="bob_strategy_meeting",
    )
    scheduler.add_job(
        _run_cleanup,
        CronTrigger(day_of_week="sun", hour=2, minute=0, timezone=_VN_TZ),
        id="cleanup_weekly",
    )
    scheduler.add_job(
        run_session_monitor,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-11,13-14",
            minute="*/5",
            timezone=_VN_TZ,
        ),
        id="session_monitor",
    )
    # Full data refresh after market close (Mon-Fri 15:35)
    scheduler.add_job(
        _run_daily_data_refresh,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=35, timezone=_VN_TZ),
        id="daily_data_refresh_eod",
    )
    # Morning catch-up at 07:30 — only re-runs if data is stale (e.g. PC was off
    # last night). Idempotent: OHLCV/VNINDEX updaters skip if up-to-date.
    scheduler.add_job(
        _run_daily_data_refresh_if_stale,
        CronTrigger(day_of_week="mon-fri", hour=7, minute=30, timezone=_VN_TZ),
        id="daily_data_refresh_morning",
    )

    print("[runner] APScheduler started:")
    print("  - Bob (M_s)     : Thu 6 20:00 VN — Strategy Development Meeting")
    print("  - Investment    : Thu 2 08:00 VN")
    print("  - Trade         : Hang ngay 08:30 VN (co data freshness gate)")
    print("  - Session mon.  : Moi 5 phut (09:00-14:35) — real-time risk")
    print("  - Data refresh  : 15:35 EOD + 07:30 morning catch-up")
    print("                    (OHLCV + VNINDEX + Money Cycle + Smart Money Trace)")
    print("  - Cleanup       : Chu nhat 02:00 VN")
    print("  Ctrl+C de dung.\n")

    # Khởi động DNSE WebSocket price feed (background thread)
    _start_ws_price_feed()

    _send_startup_notification()

    # Kiem tra code moi moi 1 gio — tu restart neu co commit moi
    scheduler.add_job(
        _check_and_restart_if_new_code,
        CronTrigger(minute=0, timezone=_VN_TZ),
        id="code_update_check",
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n[runner] Scheduler dung.")


# ──────────────────────────────────────────────
# Startup notification + Auto-update check
# ──────────────────────────────────────────────

def _git_commit() -> str:
    """Lay commit hash hien tai (7 ky tu dau)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_commit_message() -> str:
    """Lay commit message hien tai."""
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()[:80]
    except Exception:
        return ""


def _send_startup_notification() -> None:
    """Gui Discord khi scheduler khoi dong — hien thi version dang chay."""
    import os, requests as _req
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not webhook:
        return

    commit  = _git_commit()
    msg     = _git_commit_message()
    now_str = datetime.now(tz=_VN_TZ).strftime("%Y-%m-%d %H:%M")

    payload = {
        "embeds": [{
            "title":       "AI Trading Assistant — Online",
            "description": f"Scheduler khoi dong thanh cong",
            "color":       0x00AA88,
            "fields": [
                {"name": "Version",   "value": f"`{commit}`", "inline": True},
                {"name": "Commit",    "value": msg or "—",    "inline": True},
                {"name": "Thoi gian", "value": now_str,       "inline": True},
                {"name": "Jobs",      "value": "Trade 08:30 | Invest T2 08:00 | Risk monitor 5min | Cleanup CN 02:00", "inline": False},
            ],
            "footer": {"text": "AI Trading Assistant"},
        }]
    }
    try:
        _req.post(webhook, json=payload, timeout=5)
        print(f"[runner] Startup notification sent (commit: {commit})")
    except Exception as e:
        print(f"[runner] Startup notification fail: {e}")


def _check_and_restart_if_new_code() -> None:
    """Kiem tra code moi tren GitHub moi 1 gio. Restart process neu co commit moi."""
    print(f"[runner] Code update check — {datetime.now(tz=_VN_TZ).strftime('%H:%M')}")

    commit_before = _git_commit()

    try:
        # Fetch origin, khong merge
        subprocess.run(
            ["git", "fetch", "origin", "main"],
            capture_output=True, timeout=15,
        )
        # So sanh local HEAD vs origin/main
        behind = subprocess.check_output(
            ["git", "rev-list", "HEAD..origin/main", "--count"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()

        if behind == "0":
            print(f"[runner] Code up-to-date (commit: {commit_before})")
            return

        print(f"[runner] Phat hien {behind} commit moi — dang pull va restart...")

        subprocess.run(
            ["git", "pull", "origin", "main"],
            capture_output=True, timeout=30,
        )

        commit_after = _git_commit()
        msg = _git_commit_message()

        # Notify Discord truoc khi restart
        import os, requests as _req
        webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
        if webhook:
            try:
                _req.post(webhook, json={"embeds": [{
                    "title":       "Code moi — Dang restart...",
                    "description": f"{commit_before} → {commit_after}",
                    "color":       0x5865F2,
                    "fields": [{"name": "Commit moi", "value": msg or "—", "inline": False}],
                    "footer":      {"text": "AI Trading Assistant"},
                }]}, timeout=5)
            except Exception:
                pass

        # Restart process — run.bat se tu dong khoi dong lai
        print("[runner] Restarting process for code update...")
        sys.exit(0)

    except Exception as e:
        print(f"[runner] Code update check fail: {e}")


# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────

def _start_ws_price_feed() -> None:
    """Khởi động DNSE WebSocket price feed với toàn bộ liquid symbols."""
    try:
        from multiagents_trading_assistant.services.data_service import get_liquid_symbols
        from multiagents_trading_assistant.services.dnse_ws_price import start_ws_price_feed
        symbols = get_liquid_symbols(min_avg_vol=300_000)
        if symbols:
            start_ws_price_feed(symbols)
            print(f"[runner] DNSE WebSocket price feed started — {len(symbols)} mã")
        else:
            print("[runner] WS price feed: không lấy được symbol list")
    except Exception as e:
        print(f"[runner] WS price feed start fail: {e}")


def _run_cleanup() -> None:
    print(f"\n[runner] CLEANUP — {_today()}")
    try:
        stats = db.cleanup_old_data(news_keep_days=90, decisions_keep_days=180)
        print(f"[runner] Cleanup OK: {stats}")
    except Exception as e:
        print(f"[runner] Cleanup fail: {e}")
        send_pipeline_alert("cleanup", e, date=_today())


# ──────────────────────────────────────────────
# Summary helpers
# ──────────────────────────────────────────────

def _record_trade_entry(state: dict, symbol: str, setup: str, date: str) -> None:
    """Record MUA decision vào RewardTracker cho dual-reward tracking."""
    try:
        if state.get("risk_output", {}).get("final_action") != "MUA":
            return
        entry_price = (
            state.get("trader_decision", {}).get("entry_price")
            or state.get("market_context", {}).get("stock_current_price")
        )
        if not entry_price:
            return
        _reward_tracker.record_entry(
            symbol=symbol,
            setup=setup,
            entry_date=date,
            entry_price=float(entry_price),
        )
    except Exception as e:
        print(f"[runner] reward_tracker record_entry fail ({symbol}): {e}")


def _print_invest_summary(results: list[dict], date: str) -> None:
    mua  = sum(1 for s in results if s.get("risk_output", {}).get("final_action") == "MUA")
    cho  = sum(1 for s in results if s.get("risk_output", {}).get("final_action") == "CHỜ")
    tranh = sum(1 for s in results if s.get("risk_output", {}).get("final_action") == "TRÁNH")

    print(f"\n{'=' * 60}")
    print(f"  INVEST SUMMARY — {date}  ({len(results)} mã)")
    print(f"  🟢 MUA={mua}  🟡 CHỜ={cho}  🔴 TRÁNH={tranh}")
    print(f"{'=' * 60}")
    for s in results:
        sym = s.get("symbol", "?")
        action = s.get("risk_output", {}).get("final_action", "?")
        mos = s.get("valuation_analysis", {}).get("margin_of_safety")
        mos_str = f"MoS={mos:.1f}%" if mos is not None else ""
        icon = {"MUA": "🟢", "CHỜ": "🟡", "TRÁNH": "🔴"}.get(action, "⚪")
        print(f"  {sym:<6} {icon}{action:<6} {mos_str}")
    print()


def _print_trade_summary(results: list[dict], date: str) -> None:
    mua  = sum(1 for s in results if s.get("risk_output", {}).get("final_action") == "MUA")
    cho  = sum(1 for s in results if s.get("risk_output", {}).get("final_action") == "CHỜ")

    print(f"\n{'=' * 60}")
    print(f"  TRADE SUMMARY — {date}  ({len(results)} mã)")
    print(f"  🟢 MUA={mua}  🟡 CHỜ={cho}")
    print(f"{'=' * 60}")
    for s in results:
        sym = s.get("symbol", "?")
        action = s.get("risk_output", {}).get("final_action", "?")
        conf = s.get("synthesis", {}).get("confluence_score")
        conf_str = f"conf={conf:.0f}" if conf is not None else ""
        icon = {"MUA": "🟢", "CHỜ": "🟡"}.get(action, "⚪")
        print(f"  {sym:<6} {icon}{action:<6} {conf_str}")
    print()
