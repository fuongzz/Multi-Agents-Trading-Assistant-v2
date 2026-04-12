"""
main.py — CLI entry point cho AI Trading Assistant.

Dùng để chạy thủ công:
  py -3.11 main.py                     # chạy pipeline đầy đủ hôm nay
  py -3.11 main.py --symbol VNM        # phân tích 1 mã
  py -3.11 main.py --macro-only        # chỉ chạy macro agent
  py -3.11 main.py --date 2026-04-10   # phân tích ngày cụ thể
  py -3.11 main.py --screener          # chỉ chạy screener, in candidates
  py -3.11 main.py --no-discord        # chạy pipeline nhưng không gửi Discord
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Fix pandas-ta-openbb AttributeError Python 3.11
import importlib.metadata  # noqa: F401

from multiagents_trading_assistant import database as db
from multiagents_trading_assistant.agents import macro_agent
from multiagents_trading_assistant import discord_bot
from multiagents_trading_assistant import orchestrator
from multiagents_trading_assistant import screener

_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _today() -> str:
    return datetime.now(tz=_VN_TZ).strftime("%Y-%m-%d")


# ──────────────────────────────────────────────
# Sub-commands
# ──────────────────────────────────────────────

def cmd_macro(date: str) -> dict:
    """Chạy macro agent và in kết quả."""
    print(f"\n{'='*60}")
    print(f"[main] MACRO AGENT — {date}")
    print(f"{'='*60}")

    ctx = macro_agent.get_macro_context(date)
    print(f"\n[main] Kết quả macro:")
    print(json.dumps(ctx, ensure_ascii=False, indent=2))
    return ctx


def cmd_screener(date: str) -> list[dict]:
    """Chạy screener và in candidates."""
    print(f"\n{'='*60}")
    print(f"[main] SCREENER — {date}")
    print(f"{'='*60}")

    result = screener.run_screener(date=date)
    should_trade = result.get("should_trade", False)
    candidates   = result.get("candidates", [])
    market_state = result.get("market_state", "?")

    print(f"\n[main] Market state: {market_state}")
    print(f"[main] Should trade: {should_trade}")
    print(f"[main] Candidates: {len(candidates)} mã")

    for c in candidates[:10]:
        print(f"  - {c.get('symbol','?')} | {c.get('setup_type','?')} | score={c.get('priority_score','?')}")

    return candidates


def cmd_pipeline_single(symbol: str, date: str, send_discord: bool = True) -> dict:
    """Chạy pipeline cho 1 mã."""
    print(f"\n{'='*60}")
    print(f"[main] PIPELINE: {symbol} — {date}")
    print(f"{'='*60}")

    result = orchestrator.run_pipeline(symbol, date=date)
    action = result.get("risk_output", {}).get("final_action", "?")
    print(f"\n[main] Kết quả: {symbol} → {action}")

    if send_discord and action in ("MUA", "BÁN"):
        discord_bot.alert_signal(result)

    return result


def cmd_run_full(date: str, send_discord: bool = True) -> None:
    """
    Chạy toàn bộ pipeline:
      1. Macro agent
      2. Kiểm tra BEARISH → dừng nếu có
      3. Screener
      4. Pipeline từng mã trong candidates
      5. Gửi Discord
    """
    print(f"\n{'='*60}")
    print(f"[main] BẮT ĐẦU FULL PIPELINE — {date}")
    print(f"{'='*60}")

    # Bước 1: Macro
    macro_ctx = macro_agent.get_macro_context(date)
    macro_bias = macro_ctx.get("macro_bias", "NEUTRAL")

    if send_discord:
        discord_bot.alert_daily_brief(macro_ctx, date)

    # Bước 2: Dừng nếu BEARISH
    if macro_bias == "BEARISH":
        print(f"\n[main] Macro BEARISH — dừng pipeline!")
        if send_discord:
            discord_bot.alert_macro_bearish(macro_ctx)
        return

    # Bước 3: Screener
    screen_result = screener.run_screener(date=date)
    should_trade  = screen_result.get("should_trade", False)
    candidates    = screen_result.get("candidates", [])
    market_state  = screen_result.get("market_state", "DOWNTREND")

    if not should_trade:
        print(f"\n[main] Market state {market_state} — không giao dịch hôm nay")
        if send_discord:
            msg = (
                f"⛔ **KHÔNG GIAO DỊCH — {date}**\n"
                f"Market state: {market_state}\n"
                f"Đứng ngoài, giữ tiền mặt."
            )
            discord_bot.send_message(msg)
        return

    if not candidates:
        print(f"\n[main] Screener không có candidates — bỏ qua pipeline")
        if send_discord:
            discord_bot.send_message(f"📊 Screener {date}: Không có mã nào đạt tiêu chí.")
        return

    print(f"\n[main] {len(candidates)} candidates — bắt đầu pipeline...")

    # Bước 4: Pipeline từng mã
    results = []
    for candidate in candidates:
        sym       = candidate.get("symbol", "")
        setup     = candidate.get("setup_type", "SHORT_TERM")
        mkt_ctx   = candidate.get("market_context", {})

        if not sym:
            continue

        state = orchestrator.run_pipeline(sym, setup_type=setup, market_context=mkt_ctx, date=date)
        results.append(state)

        # Gửi signal ngay nếu MUA/BÁN
        if send_discord:
            final_action = state.get("risk_output", {}).get("final_action", "CHỜ")
            if final_action in ("MUA", "BÁN"):
                discord_bot.alert_signal(state)

    # Bước 5: Batch summary
    print(f"\n[main] Xong {len(results)} mã")
    if send_discord:
        discord_bot.alert_batch_summary(results, date)

    # Lưu decisions vào DB
    _save_all_decisions(results, date)


def _save_all_decisions(results: list[dict], date: str) -> None:
    """Lưu toàn bộ kết quả pipeline vào database."""
    saved = 0
    for state in results:
        try:
            symbol     = state.get("symbol", "")
            trader     = state.get("trader_decision", {})
            risk_out   = state.get("risk_output", {})

            if not symbol:
                continue

            db.save_decision(
                symbol         = symbol,
                date           = date,
                action         = trader.get("action", "CHỜ"),
                final_action   = risk_out.get("final_action", "CHỜ"),
                strategy       = state.get("setup_type"),
                quality_score  = trader.get("quality_score"),
                confidence     = trader.get("confidence"),
                entry          = trader.get("entry"),
                sl             = trader.get("sl"),
                tp             = trader.get("tp"),
                nav_pct        = trader.get("nav_pct"),
                override_reason= risk_out.get("override_reason"),
                full_output    = {k: v for k, v in state.items()
                                  if isinstance(v, (str, int, float, dict, list, bool, type(None)))},
            )
            saved += 1
        except Exception as e:
            print(f"[main] Lỗi lưu decision {state.get('symbol','?')}: {e}")

    print(f"[main] Đã lưu {saved}/{len(results)} decisions vào DB")


# ──────────────────────────────────────────────
# CLI Parser
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Trading Assistant — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  py -3.11 main.py                        # Full pipeline hôm nay
  py -3.11 main.py --symbol VNM           # 1 mã
  py -3.11 main.py --macro-only           # Chỉ macro
  py -3.11 main.py --screener             # Chỉ screener
  py -3.11 main.py --date 2026-04-10      # Ngày cụ thể
  py -3.11 main.py --no-discord           # Không gửi Discord
        """,
    )

    parser.add_argument("--symbol",     type=str,  help="Phân tích 1 mã cụ thể (VD: VNM)")
    parser.add_argument("--date",       type=str,  help="Ngày phân tích YYYY-MM-DD (mặc định hôm nay)")
    parser.add_argument("--macro-only", action="store_true", help="Chỉ chạy macro agent")
    parser.add_argument("--screener",   action="store_true", help="Chỉ chạy screener")
    parser.add_argument("--no-discord", action="store_true", help="Không gửi Discord")
    parser.add_argument("--init-db",    action="store_true", help="Khởi tạo database và thoát")

    return parser


def main() -> None:
    """Entry point chính."""
    db.init_db()

    parser = build_parser()
    args   = parser.parse_args()
    date   = args.date or _today()
    send_discord = not args.no_discord

    if args.init_db:
        print(f"[main] Database đã khởi tạo tại: {db.DB_PATH}")
        sys.exit(0)

    if args.macro_only:
        cmd_macro(date)

    elif args.screener:
        cmd_screener(date)

    elif args.symbol:
        cmd_pipeline_single(args.symbol.upper(), date, send_discord)

    else:
        cmd_run_full(date, send_discord)


if __name__ == "__main__":
    main()
