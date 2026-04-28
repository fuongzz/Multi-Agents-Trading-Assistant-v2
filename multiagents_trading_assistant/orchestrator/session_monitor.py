"""session_monitor.py — Re-analyze trong phiên, mỗi 15 phút (09:00–14:30).

Logic mỗi lần chạy:
  1. Lấy tất cả MUA decisions hôm nay từ database
  2. Với mỗi symbol:
     a. Fetch giá live (không cache)
     b. Re-run technical_agent (Haiku — ~2s, rẻ)
     c. So sánh confluence mới vs sáng
     d. Check SL/TP proximity
  3. Gửi Discord alert nếu có thay đổi đáng kể

Alert triggers:
  - Confluence giảm >= 3 điểm (0-10 scale)  → "Signal yeu di"
  - Setup quality doi: TOT -> YEU             → "Setup dao chieu"
  - Gia <= SL * 1.02                          → "Sap cham SL"
  - Gia >= TP * 0.98                          → "Gan TP - chot loi"
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from multiagents_trading_assistant import database as db
from multiagents_trading_assistant import fetcher
from multiagents_trading_assistant.agents.trade import technical_agent

_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Nguong thay doi confluence de alert (tren thang 0-10)
_CONFLUENCE_DROP_THRESHOLD = 3

# % gia cach SL/TP de bat dau canh bao
_SL_WARNING_PCT = 0.02   # gia <= SL * (1 + 2%) -> "sap cham SL"
_TP_WARNING_PCT = 0.02   # gia >= TP * (1 - 2%) -> "sap cham TP"


# ──────────────────────────────────────────────
# Entry point — goi tu APScheduler
# ──────────────────────────────────────────────

def run_session_monitor() -> None:
    """Chay re-analysis cho tat ca MUA signals hom nay."""
    now  = datetime.now(tz=_VN_TZ)
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    # Chi chay trong gio giao dich 09:00 - 14:35
    if not _is_market_open(now):
        print(f"[session_monitor] {time_str} — ngoai gio giao dich, bo qua")
        return

    print(f"\n[session_monitor] === Re-analysis {time_str} ===")

    # Lay tat ca MUA decisions hom nay
    buy_decisions = _get_today_buys(date)
    if not buy_decisions:
        print("[session_monitor] Khong co MUA signal nao hom nay")
        return

    symbols = [d["symbol"] for d in buy_decisions]
    print(f"[session_monitor] Monitor {len(symbols)} ma: {symbols}")

    # Fetch gia live 1 lan cho tat ca symbol
    live_prices = fetcher.get_live_price(symbols)

    for decision in buy_decisions:
        try:
            _check_symbol(decision, live_prices, date, time_str)
        except Exception as e:
            print(f"[session_monitor] {decision['symbol']} fail: {e}")


# ──────────────────────────────────────────────
# Core check logic
# ──────────────────────────────────────────────

def _check_symbol(
    decision: dict,
    live_prices: dict[str, float],
    date: str,
    time_str: str,
) -> None:
    symbol = decision["symbol"]
    print(f"[session_monitor] Checking {symbol}...")

    # Lay du lieu sang tu full_output
    full = decision.get("full_output") or {}
    if isinstance(full, str):
        try:
            full = json.loads(full)
        except Exception:
            full = {}

    morning_tech  = full.get("technical_analysis", {})
    morning_conf  = morning_tech.get("confluence_score")   # 0-10
    morning_qual  = morning_tech.get("setup_quality", "")

    trader  = full.get("trader_decision", {})
    entry_zone = trader.get("entry_zone")   # [low, high]
    sl      = trader.get("stop_loss")
    tp      = trader.get("take_profit")
    rr      = trader.get("rr_ratio", "?")

    # Gia hien tai
    current_price = live_prices.get(symbol)
    if not current_price:
        print(f"[session_monitor] {symbol} — khong lay duoc gia live")
        return

    alerts: list[dict] = []

    # ── Re-run technical agent (Haiku) ──
    new_tech = technical_agent.analyze(symbol, date)
    new_conf = new_tech.get("confluence_score")   # 0-10
    new_qual = new_tech.get("setup_quality", "")
    new_summary = new_tech.get("technical_summary", "")

    print(
        f"[session_monitor] {symbol} — "
        f"conf: {morning_conf} -> {new_conf} | "
        f"quality: {morning_qual} -> {new_qual} | "
        f"price: {current_price:,.0f}"
    )

    # ── Alert: confluence giam manh ──
    if (morning_conf is not None and new_conf is not None
            and morning_conf - new_conf >= _CONFLUENCE_DROP_THRESHOLD):
        alerts.append({
            "type":  "CONFLUENCE_DROP",
            "title": f"Signal yeu di — {symbol}",
            "desc":  f"Confluence: {morning_conf} -> {new_conf}/10 | {new_summary}",
            "color": 0xFFA500,
        })

    # ── Alert: setup quality dao chieu ──
    if morning_qual == "TOT" and new_qual == "YEU":
        alerts.append({
            "type":  "QUALITY_FLIP",
            "title": f"Setup dao chieu — {symbol}",
            "desc":  f"Quality: TOT -> YEU | {new_summary}",
            "color": 0xFF4444,
        })

    # ── Alert: gia sap cham SL ──
    if sl and current_price <= sl * (1 + _SL_WARNING_PCT):
        sl_pct = (current_price - sl) / sl * 100
        alerts.append({
            "type":  "NEAR_SL",
            "title": f"Sap cham SL — {symbol}",
            "desc":  f"Gia hien tai: {current_price:,.0f} | SL: {sl:,.0f} ({sl_pct:+.1f}%)",
            "color": 0xFF0000,
        })

    # ── Alert: gia sap cham TP ──
    if tp and current_price >= tp * (1 - _TP_WARNING_PCT):
        tp_pct = (current_price - tp) / tp * 100
        alerts.append({
            "type":  "NEAR_TP",
            "title": f"Gan TP — {symbol} — Chot loi?",
            "desc":  f"Gia hien tai: {current_price:,.0f} | TP: {tp:,.0f} ({tp_pct:+.1f}%)",
            "color": 0x00CC66,
        })

    if not alerts:
        print(f"[session_monitor] {symbol} — OK, khong co thay doi dang ke")
        return

    # Gui Discord
    for alert in alerts:
        _send_monitor_alert(
            alert=alert,
            symbol=symbol,
            time_str=time_str,
            entry_zone=entry_zone,
            sl=sl,
            tp=tp,
            rr=rr,
            new_conf=new_conf,
            new_qual=new_qual,
        )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _is_market_open(now: datetime) -> bool:
    """True neu dang trong gio giao dich HOSE (09:00-14:35 ca sang va chieu)."""
    if now.weekday() >= 5:   # Thu 7, Chu nhat
        return False
    h, m = now.hour, now.minute
    morning   = (9, 0) <= (h, m) <= (11, 30)
    afternoon = (13, 0) <= (h, m) <= (14, 35)
    return morning or afternoon


def _get_today_buys(date: str) -> list[dict]:
    """Lay cac MUA decisions hom nay, kem full_output."""
    try:
        decisions = db.get_decisions(date=date)
        return [
            d for d in decisions
            if d.get("final_action") == "MUA"
        ]
    except Exception as e:
        print(f"[session_monitor] get_decisions fail: {e}")
        return []


def _send_monitor_alert(
    alert: dict,
    symbol: str,
    time_str: str,
    entry_zone,
    sl,
    tp,
    rr,
    new_conf,
    new_qual,
) -> None:
    webhook_url = (
        os.getenv("DISCORD_WEBHOOK_TRADE")
        or os.getenv("DISCORD_WEBHOOK_URL", "")
    )
    if not webhook_url:
        print(f"[session_monitor] Discord webhook chua set — in ra terminal")
        print(f"  [{alert['type']}] {alert['title']}: {alert['desc']}")
        return

    ez_str = (
        f"{entry_zone[0]:,.0f}–{entry_zone[1]:,.0f}"
        if entry_zone else "N/A"
    )

    fields = [
        {"name": "Thoi gian",  "value": time_str,                           "inline": True},
        {"name": "Conf moi",   "value": f"{new_conf}/10 ({new_qual})",       "inline": True},
        {"name": "Entry zone", "value": ez_str,                             "inline": True},
        {"name": "SL",         "value": f"{sl:,.0f}" if sl else "N/A",      "inline": True},
        {"name": "TP",         "value": f"{tp:,.0f}" if tp else "N/A",      "inline": True},
        {"name": "R:R",        "value": str(rr),                            "inline": True},
    ]

    payload = {
        "embeds": [{
            "title":       alert["title"],
            "description": alert["desc"],
            "color":       alert["color"],
            "fields":      fields,
            "footer":      {"text": f"AI Trading Assistant — Session Monitor"},
        }]
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=5)
        if resp.status_code not in (200, 204):
            print(f"[session_monitor] Discord HTTP {resp.status_code}")
        else:
            print(f"[session_monitor] Discord alert sent: {alert['type']} {symbol}")
    except Exception as e:
        print(f"[session_monitor] Discord fail: {e}")
