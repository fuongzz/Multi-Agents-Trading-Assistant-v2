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

_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _today() -> str:
    return datetime.now(tz=_VN_TZ).strftime("%Y-%m-%d")


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

def run_trade_pipeline(
    symbol: str | None = None,
    date: str | None = None,
) -> list[dict]:
    date = date or _today()
    db.init_db()
    print(f"\n{'=' * 60}\n[runner] TRADE PIPELINE — {date}\n{'=' * 60}")
    try:
        return _run_trade_pipeline_inner(symbol, date)
    except Exception as e:
        send_pipeline_alert("trade", e, date=date)
        raise


def _run_trade_pipeline_inner(
    symbol: str | None,
    date: str,
) -> list[dict]:

    if symbol:
        candidates_sym = [(symbol, "UNKNOWN", {})]
    else:
        _market_ctx, candidates = trade_screener()
        candidates_sym = [
            (c.symbol, c.setup_type, asdict(c.market_context))
            for c in candidates
        ]

    results = []
    for sym, setup, mkt_ctx in candidates_sym:
        print(f"\n[runner] → Trade: {sym} ({setup})")
        try:
            state = run_trade(symbol=sym, setup_type=setup, market_context=mkt_ctx, date=date)
            results.append(state)
            try:
                save_trade_decision(state)
            except Exception as e:
                print(f"[runner] memory save fail ({sym}): {e}")
        except Exception as e:
            print(f"[runner] Trade {sym} FAIL: {e}")
            send_pipeline_alert("trade", e, symbol=sym, date=date)

    _print_trade_summary(results, date)
    return results


# ──────────────────────────────────────────────
# Scheduler
# ──────────────────────────────────────────────

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
        _run_cleanup,
        CronTrigger(day_of_week="sun", hour=2, minute=0, timezone=_VN_TZ),
        id="cleanup_weekly",
    )
    scheduler.add_job(
        run_session_monitor,
        CronTrigger(
            day_of_week="mon-fri",
            hour="9-11,13-14",
            minute="0,15,30,45",
            timezone=_VN_TZ,
        ),
        id="session_monitor",
    )

    print("[runner] APScheduler started:")
    print("  - Investment    : Thu 2 08:00 VN")
    print("  - Trade         : Hang ngay 08:30 VN")
    print("  - Session mon.  : Moi 15 phut (09:00-14:45)")
    print("  - Cleanup       : Chu nhat 02:00 VN")
    print("  Ctrl+C de dung.\n")

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
                {"name": "Jobs",      "value": "Trade 08:30 | Invest T2 08:00 | Monitor 15min | Cleanup CN 02:00", "inline": False},
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
