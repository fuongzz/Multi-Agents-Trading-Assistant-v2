"""session_monitor.py — Quản trị rủi ro real-time trong phiên, mỗi 5 phút (09:00–14:35).

Hai nhóm kiểm tra độc lập chạy mỗi chu kỳ:

[A] Hard check vị thế đang giữ (positions table) — ƯU TIÊN CAO
    - SL_HIT + t3_available  → close_position() + Discord "⚡ THOÁT NGAY"
    - SL_HIT + T+0/T+1/T+2  → Discord "🔴 BỊ KẸP" mỗi 5 phút đến khi thoát được
    - TP_HIT + t3_available  → close_position() + Discord "💰 CHỐT LỜI"
    - TP_HIT + T+0/T+1/T+2  → Discord "⚠️ GẦN TP - kẹp T+3" (nhắc nhở)

[B] Re-analysis tín hiệu MUA hôm nay (decisions table)
    - Confluence giảm >= 3 điểm  → "Signal yếu đi"
    - Setup quality TOT -> YEU    → "Setup đảo chiều"
    - Giá sắp chạm SL (còn 2%)   → "Sắp chạm SL" (warning, chưa hit)
    - Giá sắp chạm TP (còn 2%)   → "Gần TP - chốt lời?"
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from multiagents_trading_assistant import database as db
from multiagents_trading_assistant import fetcher

_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Ngưỡng re-analysis
_CONFLUENCE_DROP_THRESHOLD = 3  # điểm trên thang 0-10
_SL_WARNING_PCT = 0.02          # giá <= SL * 1.02 → "sắp chạm SL"
_TP_WARNING_PCT = 0.02          # giá >= TP * 0.98 → "gần TP"


# ──────────────────────────────────────────────
# Entry point — gọi từ APScheduler mỗi 5 phút
# ──────────────────────────────────────────────

def run_session_monitor() -> None:
    now = datetime.now(tz=_VN_TZ)
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    if not _is_market_open(now):
        print(f"[session_monitor] {time_str} — ngoài giờ giao dịch, bỏ qua")
        return

    print(f"\n[session_monitor] === {time_str} ===")

    # ── [A] Hard check vị thế đang giữ ──
    _check_open_positions(date, time_str)

    # ── [B] Re-analysis MUA signals hôm nay ──
    _reanalyze_today_signals(date, time_str)


# ──────────────────────────────────────────────
# [A] Hard check — positions table
# ──────────────────────────────────────────────

def _check_open_positions(date: str, time_str: str) -> None:
    """
    Kiểm tra tất cả vị thế đang giữ (không chỉ hôm nay).
    Tự đóng DB khi SL/TP hit + t3 sẵn sàng.
    Alert kẹp T+0 lặp lại mỗi 5 phút khi chưa thoát được.
    """
    positions = db.get_all_positions()
    if not positions:
        return

    symbols = [p["symbol"] for p in positions]
    live_prices = _fetch_live_prices(symbols)

    for pos in positions:
        sym = pos["symbol"]
        current = live_prices.get(sym)
        if not current:
            print(f"[session_monitor] {sym} — không lấy được giá live")
            continue

        sl = pos.get("sl")
        tp = pos.get("tp")
        entry_price = pos.get("entry_price", 0)
        quantity = pos.get("quantity", 0)
        strategy = pos.get("strategy", "")

        # Tính T+3 availability
        entry_date_str = pos.get("entry_date", "")
        try:
            entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
            days_held = (datetime.now(tz=_VN_TZ).date() - entry_date).days
            t3_available = days_held >= 3
        except (ValueError, TypeError):
            days_held = 0
            t3_available = False

        pnl_pct = (current - entry_price) / entry_price * 100 if entry_price > 0 else 0

        # ── Cập nhật peak_price ──
        old_peak = pos.get("peak_price") or 0
        if current > old_peak:
            db.update_position_peak(sym, current)

        # ── SL bị hit ──
        if sl and current <= sl:
            if t3_available:
                # Thoát được → đóng DB + alert "THOÁT NGAY"
                try:
                    db.close_position(
                        symbol=sym,
                        exit_price=current,
                        exit_date=date,
                        exit_reason="SL_HIT",
                        entry_price=entry_price,
                        quantity=quantity,
                        strategy=strategy,
                    )
                    print(f"[session_monitor] ⚡ {sym} SL_HIT + đã đóng DB @ {current:,.0f}")
                except Exception as e:
                    print(f"[session_monitor] Lỗi close_position {sym}: {e}")

                _send_hard_exit_alert(
                    symbol=sym, exit_type="SL_HIT", current_price=current,
                    entry_price=entry_price, sl=sl, tp=tp,
                    pnl_pct=pnl_pct, days_held=days_held,
                    time_str=time_str, t3_available=True,
                )
            else:
                # Kẹp T+0/T+1/T+2 → alert đỏ, nhắc mỗi 5 phút
                print(f"[session_monitor] 🔴 {sym} SL_HIT nhưng kẹp T+{days_held} @ {current:,.0f}")
                _send_trapped_alert(
                    symbol=sym, current_price=current,
                    entry_price=entry_price, sl=sl, tp=tp,
                    pnl_pct=pnl_pct, days_held=days_held,
                    time_str=time_str,
                )
            continue  # skip TP check nếu đã xử lý SL

        # ── Đạt initial target → nâng SL, tiếp tục giữ ──
        if tp and current >= tp:
            # Tính SL mới đề xuất: entry + (current - entry) × 0.5 (khoá ≥50% lãi)
            suggested_sl = round(entry_price + (current - entry_price) * 0.5, 0)
            old_sl = sl or 0.0

            if suggested_sl > old_sl:
                try:
                    db.update_position_sl(sym, suggested_sl)
                    print(f"[session_monitor] 📈 {sym} đạt target — nâng SL {old_sl:,.0f}→{suggested_sl:,.0f}")
                except Exception as e:
                    print(f"[session_monitor] Lỗi update_position_sl {sym}: {e}")

            _send_profit_trail_alert(
                symbol=sym, current_price=current,
                entry_price=entry_price, old_sl=old_sl,
                suggested_sl=suggested_sl, initial_target=tp,
                pnl_pct=pnl_pct, days_held=days_held,
                time_str=time_str, t3_available=t3_available,
            )


# ──────────────────────────────────────────────
# [B] Re-analysis — decisions table (hôm nay)
# ──────────────────────────────────────────────

def _reanalyze_today_signals(date: str, time_str: str) -> None:
    """Re-analyze tín hiệu MUA hôm nay — confluence, quality, proximity."""
    from multiagents_trading_assistant.agents.trade import technical_agent

    buy_decisions = _get_today_buys(date)
    if not buy_decisions:
        return

    symbols = [d["symbol"] for d in buy_decisions]
    print(f"[session_monitor] Re-analyze {len(symbols)} tín hiệu hôm nay: {symbols}")

    live_prices = _fetch_live_prices(symbols)

    for decision in buy_decisions:
        try:
            _check_signal(decision, live_prices, date, time_str, technical_agent)
        except Exception as e:
            print(f"[session_monitor] {decision['symbol']} re-analysis fail: {e}")


def _check_signal(
    decision: dict,
    live_prices: dict[str, float],
    date: str,
    time_str: str,
    technical_agent,
) -> None:
    symbol = decision["symbol"]

    # Bỏ qua nếu vị thế đã bị đóng bởi [A]
    if not db.has_position(symbol):
        return

    full = decision.get("full_output") or {}
    if isinstance(full, str):
        try:
            full = json.loads(full)
        except Exception:
            full = {}

    morning_tech = full.get("technical_analysis", {})
    morning_conf = morning_tech.get("confluence_score")
    morning_qual = morning_tech.get("setup_quality", "")

    trader = full.get("trader_decision", {})
    entry_zone = trader.get("entry_zone")
    sl = trader.get("stop_loss")
    tp = trader.get("take_profit")
    rr = trader.get("rr_ratio", "?")

    current_price = live_prices.get(symbol)
    if not current_price:
        return

    alerts: list[dict] = []

    # Re-run technical agent (Haiku — ~2s)
    new_tech = technical_agent.analyze(symbol, date)
    new_conf = new_tech.get("confluence_score")
    new_qual = new_tech.get("setup_quality", "")
    new_summary = new_tech.get("technical_summary", "")

    print(
        f"[session_monitor] {symbol} conf: {morning_conf}→{new_conf} "
        f"qual: {morning_qual}→{new_qual} price: {current_price:,.0f}"
    )

    if (morning_conf is not None and new_conf is not None
            and morning_conf - new_conf >= _CONFLUENCE_DROP_THRESHOLD):
        alerts.append({
            "type":  "CONFLUENCE_DROP",
            "title": f"Signal yếu đi — {symbol}",
            "desc":  f"Confluence: {morning_conf}→{new_conf}/10 | {new_summary}",
            "color": 0xFFA500,
        })

    if morning_qual == "TOT" and new_qual == "YEU":
        alerts.append({
            "type":  "QUALITY_FLIP",
            "title": f"Setup đảo chiều — {symbol}",
            "desc":  f"Quality: TOT→YEU | {new_summary}",
            "color": 0xFF4444,
        })

    # Cảnh báo gần SL (chưa hit — đã xử lý ở [A] nếu hit rồi)
    if sl and current_price > sl and current_price <= sl * (1 + _SL_WARNING_PCT):
        sl_pct = (current_price - sl) / sl * 100
        alerts.append({
            "type":  "NEAR_SL",
            "title": f"Sắp chạm SL — {symbol}",
            "desc":  f"Giá: {current_price:,.0f} | SL: {sl:,.0f} ({sl_pct:+.1f}%)",
            "color": 0xFF6600,
        })

    if tp and current_price < tp and current_price >= tp * (1 - _TP_WARNING_PCT):
        tp_pct = (current_price - tp) / tp * 100
        alerts.append({
            "type":  "NEAR_TARGET",
            "title": f"Gần target ban đầu — {symbol}",
            "desc":  f"Giá: {current_price:,.0f} | Target: {tp:,.0f} ({tp_pct:+.1f}%) — Chuẩn bị nâng SL khi chạm.",
            "color": 0x00CC66,
        })

    if not alerts:
        print(f"[session_monitor] {symbol} — OK")
        return

    for alert in alerts:
        _send_reanalysis_alert(
            alert=alert, symbol=symbol, time_str=time_str,
            entry_zone=entry_zone, sl=sl, tp=tp, rr=rr,
            new_conf=new_conf, new_qual=new_qual,
        )


# ──────────────────────────────────────────────
# Discord senders
# ──────────────────────────────────────────────

def _send_hard_exit_alert(
    symbol: str, exit_type: str, current_price: float,
    entry_price: float, sl, tp, pnl_pct: float,
    days_held: int, time_str: str, t3_available: bool,
) -> None:
    """Alert khi đóng vị thế thực sự (SL/TP hit + t3 sẵn sàng)."""
    webhook = _get_webhook()

    if exit_type == "SL_HIT":
        icon, color = "⚡", 0xFF0000
        title = f"⚡ THOÁT NGAY — {symbol} — SL HIT"
        desc = f"**Đặt lệnh BÁN ngay!** Vị thế đã được ghi nhận đóng trong hệ thống."
    else:
        icon, color = "💰", 0x00CC66
        title = f"💰 CHỐT LỜI — {symbol} — TP HIT"
        desc = f"**Đặt lệnh BÁN ngay!** Vị thế đã được ghi nhận đóng trong hệ thống."

    pnl_sign = "+" if pnl_pct >= 0 else ""
    fields = [
        {"name": "Thời gian",  "value": time_str,                                "inline": True},
        {"name": "Giá hiện tại","value": f"{current_price:,.0f}",               "inline": True},
        {"name": "P&L",        "value": f"{pnl_sign}{pnl_pct:.1f}%",            "inline": True},
        {"name": "Giá vào",    "value": f"{entry_price:,.0f}",                   "inline": True},
        {"name": "SL",         "value": f"{sl:,.0f}" if sl else "N/A",           "inline": True},
        {"name": "TP",         "value": f"{tp:,.0f}" if tp else "N/A",           "inline": True},
        {"name": "Giữ",        "value": f"{days_held} ngày",                     "inline": True},
        {"name": "T+3",        "value": "✅ Có thể bán",                         "inline": True},
    ]
    _post_embed(webhook, title, desc, color, fields, footer="AI Trading Assistant — Real-time Risk")


def _send_trapped_alert(
    symbol: str, current_price: float,
    entry_price: float, sl, tp,
    pnl_pct: float, days_held: int, time_str: str,
) -> None:
    """Alert kẹp T+0/T+1/T+2 — SL đã bị hit nhưng chưa bán được."""
    webhook = _get_webhook()

    days_to_free = 3 - days_held
    pnl_sign = "+" if pnl_pct >= 0 else ""

    # Ước tính lỗ thêm nếu giá tiếp tục giảm (worst case -7%)
    worst_case_price = current_price * 0.93
    worst_pnl_pct = (worst_case_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

    fields = [
        {"name": "Thời gian",      "value": time_str,                                "inline": True},
        {"name": "Giá hiện tại",   "value": f"{current_price:,.0f}",                "inline": True},
        {"name": "P&L hiện tại",   "value": f"{pnl_sign}{pnl_pct:.1f}%",            "inline": True},
        {"name": "Giá vào",        "value": f"{entry_price:,.0f}",                   "inline": True},
        {"name": "SL đã hit",      "value": f"{sl:,.0f}" if sl else "N/A",           "inline": True},
        {"name": "Kẹp còn",        "value": f"~{days_to_free} ngày nữa",             "inline": True},
        {"name": "Worst case -7%", "value": f"{worst_case_price:,.0f} ({worst_pnl_pct:.1f}%)", "inline": True},
        {"name": "TP",             "value": f"{tp:,.0f}" if tp else "N/A",           "inline": True},
        {"name": "Khuyến nghị",    "value": "Theo dõi chặt, đặt lệnh bán ngay khi T+3 đến hạn", "inline": False},
    ]
    _post_embed(
        webhook,
        title=f"🔴 BỊ KẸP T+{days_held} — {symbol} — SL ĐÃ HIT",
        desc=f"**Chưa thể bán — cần thêm {days_to_free} ngày!** Nhắc lại mỗi 5 phút.",
        color=0xCC0000,
        fields=fields,
        footer="AI Trading Assistant — Real-time Risk",
    )


def _send_profit_trail_alert(
    symbol: str, current_price: float,
    entry_price: float, old_sl: float, suggested_sl: float,
    initial_target: float, pnl_pct: float,
    days_held: int, time_str: str, t3_available: bool,
) -> None:
    """Alert khi giá đạt initial_target — nâng SL để khoá lãi, tiếp tục giữ."""
    webhook = _get_webhook()
    pnl_sign = "+" if pnl_pct >= 0 else ""
    t3_str = "✅ Có thể bán nếu muốn" if t3_available else f"⏳ Kẹp T+{days_held}"

    fields = [
        {"name": "Thời gian",       "value": time_str,                                  "inline": True},
        {"name": "Giá hiện tại",    "value": f"{current_price:,.0f}",                   "inline": True},
        {"name": "P&L",             "value": f"{pnl_sign}{pnl_pct:.1f}%",              "inline": True},
        {"name": "SL cũ",           "value": f"{old_sl:,.0f}" if old_sl else "N/A",    "inline": True},
        {"name": "SL mới (đề xuất)","value": f"{suggested_sl:,.0f}",                   "inline": True},
        {"name": "Target ban đầu",  "value": f"{initial_target:,.0f}",                 "inline": True},
        {"name": "Giữ",             "value": f"{days_held} ngày",                       "inline": True},
        {"name": "T+3",             "value": t3_str,                                    "inline": True},
        {"name": "Hành động",       "value": "SL đã được nâng tự động. Tiếp tục giữ — thoát khi price action đảo chiều.", "inline": False},
    ]
    _post_embed(
        webhook,
        title=f"📈 ĐẠT TARGET — {symbol} — NÂNG SL",
        desc=f"Giá chạm target ban đầu. **Không chốt** — để lệnh chạy, SL đã được nâng để khoá lãi.",
        color=0x00CC66,
        fields=fields,
        footer="AI Trading Assistant — Trailing Exit",
    )


def _send_reanalysis_alert(
    alert: dict, symbol: str, time_str: str,
    entry_zone, sl, tp, rr, new_conf, new_qual,
) -> None:
    webhook = _get_webhook()
    ez_str = (
        f"{entry_zone[0]:,.0f}–{entry_zone[1]:,.0f}"
        if entry_zone else "N/A"
    )
    fields = [
        {"name": "Thời gian",  "value": time_str,                            "inline": True},
        {"name": "Conf mới",   "value": f"{new_conf}/10 ({new_qual})",        "inline": True},
        {"name": "Entry zone", "value": ez_str,                              "inline": True},
        {"name": "SL",         "value": f"{sl:,.0f}" if sl else "N/A",       "inline": True},
        {"name": "TP",         "value": f"{tp:,.0f}" if tp else "N/A",       "inline": True},
        {"name": "R:R",        "value": str(rr),                             "inline": True},
    ]
    _post_embed(
        webhook, alert["title"], alert["desc"], alert["color"],
        fields, footer="AI Trading Assistant — Session Monitor",
    )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _get_webhook() -> str:
    return os.getenv("DISCORD_WEBHOOK_TRADE") or os.getenv("DISCORD_WEBHOOK_URL", "")


def _post_embed(
    webhook: str, title: str, desc: str, color: int,
    fields: list[dict], footer: str = "AI Trading Assistant",
) -> None:
    if not webhook:
        print(f"  [{title}] {desc}")
        return
    payload = {"embeds": [{"title": title, "description": desc, "color": color,
                           "fields": fields, "footer": {"text": footer}}]}
    try:
        resp = requests.post(webhook, json=payload, timeout=5)
        if resp.status_code not in (200, 204):
            print(f"[session_monitor] Discord HTTP {resp.status_code}")
    except Exception as e:
        print(f"[session_monitor] Discord fail: {e}")


def _fetch_live_prices(symbols: list[str]) -> dict[str, float]:
    """
    Lấy giá live theo thứ tự ưu tiên:
      1. DNSE WebSocket cache (instant, real-time tick)
      2. DNSE HTTP poll fallback (không cache)
    """
    try:
        from multiagents_trading_assistant.services.dnse_ws_price import (
            get_ws_price, is_connected, subscribe_symbols,
        )
        if is_connected():
            # Đảm bảo các symbol mới đã được subscribe
            subscribe_symbols(symbols)

            ws_prices = {}
            missing = []
            for sym in symbols:
                p = get_ws_price(sym, max_age_seconds=60.0)
                if p is not None:
                    ws_prices[sym] = p
                else:
                    missing.append(sym)

            if missing:
                # Fallback HTTP cho các mã chưa có tick
                http_prices = fetcher.get_live_price(missing)
                ws_prices.update(http_prices)

            return ws_prices
    except Exception as e:
        print(f"[session_monitor] WS price lỗi: {e}")

    # Fallback hoàn toàn sang HTTP
    try:
        return fetcher.get_live_price(symbols)
    except Exception as e:
        print(f"[session_monitor] Lấy giá live fail: {e}")
        return {}


def _is_market_open(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    morning   = (9, 0) <= (h, m) <= (11, 30)
    afternoon = (13, 0) <= (h, m) <= (14, 35)
    return morning or afternoon


def _get_today_buys(date: str) -> list[dict]:
    try:
        return [d for d in db.get_decisions(date=date) if d.get("final_action") == "MUA"]
    except Exception as e:
        print(f"[session_monitor] get_decisions fail: {e}")
        return []
