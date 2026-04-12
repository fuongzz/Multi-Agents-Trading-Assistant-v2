"""
scheduler.py — APScheduler cron tự động cho AI Trading Assistant.

Lịch cố định (múi giờ Việt Nam):
  06:00 — morning_fetch:  Macro + foreign flow + news (pre-market)
  08:45 — morning_brief:  Scan VN100 → full pipeline → Discord
  15:10 — session_close:  Recap phiên + lưu closing prices
  20:00 — evening_fetch:  Compress memory + cập nhật news outcomes

Chạy:
  py -3.11 multiagents_trading_assistant/scheduler.py
"""

import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
import json
import os
import signal
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from multiagents_trading_assistant import database as db
from multiagents_trading_assistant import discord_bot
from multiagents_trading_assistant import orchestrator
from multiagents_trading_assistant import screener
from multiagents_trading_assistant.agents import macro_agent
from multiagents_trading_assistant.memory.memory_system import get_memory

_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _today() -> str:
    return datetime.now(tz=_VN_TZ).strftime("%Y-%m-%d")


def _log(msg: str) -> None:
    ts = datetime.now(tz=_VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[scheduler {ts}] {msg}")


# ──────────────────────────────────────────────
# Job 1 — 06:00: Morning Fetch
# ──────────────────────────────────────────────

def job_morning_fetch() -> None:
    """
    Pre-market fetch: macro + news.
    Cache kết quả để morning_brief dùng ngay, không chờ.
    """
    date = _today()
    _log(f"=== MORNING FETCH bắt đầu — {date} ===")

    try:
        # Fetch + cache macro (LLM call ~20s)
        ctx = macro_agent.get_macro_context(date)
        bias = ctx.get("macro_bias", "NEUTRAL")
        _log(f"Macro cache OK: {bias} / score={ctx.get('macro_score',0)}")

        # Gửi daily brief sớm lên Discord
        discord_bot.alert_daily_brief(ctx, date)
        _log("Daily brief → Discord OK")

        # Cảnh báo sớm nếu BEARISH
        if bias == "BEARISH":
            discord_bot.alert_macro_bearish(ctx)
            _log("MACRO BEARISH alert → Discord")

    except Exception as e:
        _log(f"LỖI morning_fetch: {e}")
        discord_bot.send_message(f"⚠️ Morning fetch lỗi {date}: {e}")

    _log("=== MORNING FETCH xong ===")


# ──────────────────────────────────────────────
# Job 2 — 08:45: Morning Brief (full pipeline)
# ──────────────────────────────────────────────

def job_morning_brief() -> None:
    """
    Full pipeline: screener → orchestrator → Discord signals.
    Đây là job chính hàng ngày.
    """
    date = _today()
    _log(f"=== MORNING BRIEF bắt đầu — {date} ===")

    try:
        # Đọc macro từ cache (đã fetch lúc 06:00)
        from pathlib import Path
        cache_dir  = Path(__file__).parent / "cache"
        cache_path = cache_dir / f"macro_{date}.json"

        if cache_path.exists():
            macro_ctx = json.loads(cache_path.read_text(encoding="utf-8"))
            _log(f"Macro cache loaded: {macro_ctx.get('macro_bias','?')}")
        else:
            _log("Không có macro cache — fetch lại...")
            macro_ctx = macro_agent.get_macro_context(date)

        macro_bias = macro_ctx.get("macro_bias", "NEUTRAL")

        # Dừng nếu BEARISH
        if macro_bias == "BEARISH":
            _log("Macro BEARISH — dừng pipeline!")
            discord_bot.alert_macro_bearish(macro_ctx)
            return

        # Screener
        _log("Chạy screener...")
        screen_result = screener.run_screener(date=date)
        should_trade  = screen_result.get("should_trade", False)
        candidates    = screen_result.get("candidates", [])
        market_state  = screen_result.get("market_state", "?")
        _log(f"Screener: {market_state} | {len(candidates)} candidates | should_trade={should_trade}")

        if not should_trade:
            msg = f"⛔ **KHÔNG GIAO DỊCH — {date}**\nMarket: {market_state}\nGiữ tiền mặt."
            discord_bot.send_message(msg)
            return

        if not candidates:
            discord_bot.send_message(f"📊 Screener {date}: Không có mã đạt tiêu chí.")
            return

        # Pipeline từng mã
        _log(f"Pipeline {len(candidates)} mã...")
        results = []
        for c in candidates:
            sym   = c.get("symbol", "")
            setup = c.get("setup_type", "SHORT_TERM")
            mkt   = c.get("market_context", {})
            if not sym:
                continue
            try:
                state = orchestrator.run_pipeline(sym, setup_type=setup, market_context=mkt, date=date)
                results.append(state)

                final = state.get("risk_output", {}).get("final_action", "CHỜ")
                if final in ("MUA", "BÁN"):
                    discord_bot.alert_signal(state)
                    _log(f"{sym} → {final} — signal đã gửi Discord")
            except Exception as e:
                _log(f"LỖI pipeline {sym}: {e}")

        # Batch summary
        discord_bot.alert_batch_summary(results, date)
        _log(f"Batch summary → Discord: {len(results)} mã")

        # Lưu DB
        _save_decisions(results, date)

    except Exception as e:
        _log(f"LỖI NGHIÊM TRỌNG morning_brief: {e}")
        discord_bot.send_message(f"❌ Morning brief lỗi {date}: {e}")

    _log("=== MORNING BRIEF xong ===")


# ──────────────────────────────────────────────
# Job 3 — 15:10: Session Close Recap
# ──────────────────────────────────────────────

def job_session_close() -> None:
    """
    Recap sau đóng cửa:
      - Tổng hợp kết quả các signals đã gửi sáng
      - Cập nhật P&L tạm tính (so entry vs close)
    """
    date = _today()
    _log(f"=== SESSION CLOSE — {date} ===")

    try:
        decisions_today = db.get_decisions(date=date)
        mua_signals = [d for d in decisions_today if d.get("final_action") == "MUA"]
        ban_signals = [d for d in decisions_today if d.get("final_action") == "BÁN"]

        lines = [
            f"📊 **RECAP PHIÊN — {date}**",
            f"",
            f"Tổng quyết định: {len(decisions_today)}",
            f"🟢 Tín hiệu MUA: {len(mua_signals)} mã",
            f"🔴 Tín hiệu BÁN: {len(ban_signals)} mã",
        ]

        if mua_signals:
            syms = [d.get("symbol","?") for d in mua_signals]
            lines.append(f"MUA: {', '.join(syms)}")

        msg = "\n".join(lines)
        discord_bot.send_message(msg)
        _log(f"Recap gửi Discord: {len(decisions_today)} decisions")

    except Exception as e:
        _log(f"LỖI session_close: {e}")

    _log("=== SESSION CLOSE xong ===")


# ──────────────────────────────────────────────
# Job 4 — 20:00: Evening Fetch (Compress Memory)
# ──────────────────────────────────────────────

def job_evening_fetch() -> None:
    """
    Evening job:
      - Compress decisions hôm nay → L2 ChromaDB
      - Log stats
    """
    date = _today()
    _log(f"=== EVENING FETCH — {date} ===")

    try:
        mem = get_memory()
        summary = mem.compress_daily(date)
        _log(
            f"Memory compress: {summary['decisions_today']} decisions, "
            f"{summary['saved_to_l2']} → L2, "
            f"{summary['total_positions']} positions"
        )

    except Exception as e:
        _log(f"LỖI evening_fetch: {e}")

    _log("=== EVENING FETCH xong ===")


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _save_decisions(results: list[dict], date: str) -> None:
    saved = 0
    for state in results:
        try:
            sym    = state.get("symbol", "")
            trader = state.get("trader_decision", {})
            risk   = state.get("risk_output", {})
            if not sym:
                continue
            db.save_decision(
                symbol          = sym,
                date            = date,
                action          = trader.get("action", "CHỜ"),
                final_action    = risk.get("final_action", "CHỜ"),
                strategy        = state.get("setup_type"),
                quality_score   = trader.get("quality_score"),
                confidence      = trader.get("confidence"),
                entry           = trader.get("entry"),
                sl              = trader.get("sl"),
                tp              = trader.get("tp"),
                nav_pct         = trader.get("nav_pct"),
                override_reason = risk.get("override_reason"),
                full_output     = {k: v for k, v in state.items()
                                   if isinstance(v, (str, int, float, dict, list, bool, type(None)))},
            )
            saved += 1
        except Exception as e:
            _log(f"Lỗi lưu decision {state.get('symbol','?')}: {e}")
    _log(f"Đã lưu {saved}/{len(results)} decisions vào DB")


# ──────────────────────────────────────────────
# Build + Start Scheduler
# ──────────────────────────────────────────────

def build_scheduler() -> BlockingScheduler:
    """Tạo APScheduler với 4 jobs theo lịch VN."""
    scheduler = BlockingScheduler(timezone=_VN_TZ)

    scheduler.add_job(
        job_morning_fetch,
        CronTrigger(hour=6, minute=0, timezone=_VN_TZ),
        id="morning_fetch",
        name="Morning Fetch (macro + news)",
        misfire_grace_time=300,  # 5 phút grace nếu chậm
    )

    scheduler.add_job(
        job_morning_brief,
        CronTrigger(hour=8, minute=45, timezone=_VN_TZ),
        id="morning_brief",
        name="Morning Brief (full pipeline)",
        misfire_grace_time=600,  # 10 phút grace
    )

    scheduler.add_job(
        job_session_close,
        CronTrigger(hour=15, minute=10, timezone=_VN_TZ),
        id="session_close",
        name="Session Close Recap",
        misfire_grace_time=300,
    )

    scheduler.add_job(
        job_evening_fetch,
        CronTrigger(hour=20, minute=0, timezone=_VN_TZ),
        id="evening_fetch",
        name="Evening Fetch (compress memory)",
        misfire_grace_time=600,
    )

    return scheduler


def run_scheduler() -> None:
    """Khởi động scheduler — blocking, chạy mãi mãi."""
    db.init_db()

    scheduler = build_scheduler()

    # Graceful shutdown khi nhận Ctrl+C hoặc SIGTERM
    def _shutdown(signum, frame):
        _log("Nhận tín hiệu dừng — shutdown scheduler...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    _log("Scheduler khởi động — đang chờ các jobs theo lịch VN:")
    for job in scheduler.get_jobs():
        _log(f"  {job.name}: {job.trigger}")

    discord_bot.send_message(
        f"🤖 **AI Trading Assistant ONLINE** — {_today()}\n"
        f"Scheduler chạy với 4 jobs: 06:00 / 08:45 / 15:10 / 20:00 VN"
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        _log("Scheduler đã dừng.")


# ──────────────────────────────────────────────
# Test nhanh (chạy 1 job ngay)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    db.init_db()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "morning_fetch":
            job_morning_fetch()
        elif cmd == "morning_brief":
            job_morning_brief()
        elif cmd == "session_close":
            job_session_close()
        elif cmd == "evening_fetch":
            job_evening_fetch()
        elif cmd == "start":
            run_scheduler()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: py -3.11 scheduler.py [morning_fetch|morning_brief|session_close|evening_fetch|start]")
    else:
        # Mặc định: print jobs info
        scheduler = build_scheduler()
        print(f"Scheduler config (timezone: Asia/Ho_Chi_Minh):")
        for job in scheduler.get_jobs():
            print(f"  [{job.id}] {job.name}: {job.trigger}")
        print("Để chạy thật: py -3.11 scheduler.py start")
