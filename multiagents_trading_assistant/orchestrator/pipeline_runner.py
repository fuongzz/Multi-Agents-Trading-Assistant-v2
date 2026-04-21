"""pipeline_runner.py — APScheduler runner cho cả hai pipeline.

Schedule:
  Investment : Thứ 2 hàng tuần lúc 08:00
  Trade      : Hàng ngày lúc 08:30 (trước ATO 09:00)

Chế độ --no-schedule / run_once=True: chạy một lần rồi thoát (dùng cho test).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from multiagents_trading_assistant.orchestrator.investment_graph import run_pipeline as run_invest
from multiagents_trading_assistant.orchestrator.trade_graph import run_pipeline as run_trade
from multiagents_trading_assistant.screener.invest_screener import run_screener as invest_screener
from multiagents_trading_assistant.screener.trade_screener import run_screener as trade_screener
from multiagents_trading_assistant.services import output_service

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

    if symbol:
        candidates_sym = [symbol]
    else:
        candidates = invest_screener()
        candidates_sym = [c.symbol for c in candidates]

    results = []
    for sym in candidates_sym:
        print(f"\n[runner] → Invest: {sym}")
        state = run_invest(symbol=sym, date=date)
        results.append(state)

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
    print(f"\n{'=' * 60}\n[runner] TRADE PIPELINE — {date}\n{'=' * 60}")

    if symbol:
        candidates_sym = [(symbol, "UNKNOWN", {})]
    else:
        candidates = trade_screener()
        candidates_sym = [
            (c.symbol, getattr(c, "setup_type", "UNKNOWN"), {})
            for c in candidates
        ]

    results = []
    for sym, setup, mkt_ctx in candidates_sym:
        print(f"\n[runner] → Trade: {sym} ({setup})")
        state = run_trade(symbol=sym, setup_type=setup, market_context=mkt_ctx, date=date)
        results.append(state)

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

    print("[runner] APScheduler started:")
    print("  - Investment : Thứ 2 08:00 VN")
    print("  - Trade      : Hàng ngày 08:30 VN")
    print("  Ctrl+C để dừng.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n[runner] Scheduler dừng.")


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
