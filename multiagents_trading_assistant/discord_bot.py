"""
discord_bot.py — Gửi alert và mini analyst note lên Discord.

2 cách gửi:
  1. Webhook (send_webhook) — đơn giản, không cần bot token
  2. Bot (send_bot_message) — cần DISCORD_BOT_TOKEN, hỗ trợ nhiều kênh

Format message:
  - Daily Brief: bối cảnh macro + thị trường
  - Signal Alert: mini analyst note cho từng mã (MUA/BÁN/CHỜ)
  - Batch Summary: tóm tắt toàn bộ batch cuối ngày
"""

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
BOT_TOKEN   = os.getenv("DISCORD_BOT_TOKEN", "")
CHANNEL_ID  = os.getenv("DISCORD_CHANNEL_ID", "")

# Giới hạn ký tự Discord per message
_MAX_CHARS = 1900


# ──────────────────────────────────────────────
# Low-level send
# ──────────────────────────────────────────────

def send_webhook(content: str, embeds: list | None = None) -> bool:
    """
    Gửi message qua Discord Webhook.

    Args:
        content: Text message (hỗ trợ markdown Discord)
        embeds:  List embed objects (optional)

    Returns:
        True nếu thành công, False nếu lỗi
    """
    if not WEBHOOK_URL:
        print("[discord] DISCORD_WEBHOOK_URL chưa set — bỏ qua gửi")
        return False

    payload: dict = {}
    if content:
        payload["content"] = content[:_MAX_CHARS]
    if embeds:
        payload["embeds"] = embeds

    try:
        resp = requests.post(
            WEBHOOK_URL,
            json=payload,
            timeout=10,
        )
        if resp.status_code in (200, 204):
            print(f"[discord] Webhook OK ({resp.status_code})")
            return True
        else:
            print(f"[discord] Webhook lỗi {resp.status_code}: {resp.text[:200]}")
            return False
    except requests.RequestException as e:
        print(f"[discord] Webhook exception: {e}")
        return False


def send_bot_message(content: str, channel_id: str | None = None) -> bool:
    """
    Gửi message qua Discord Bot API.

    Args:
        content:    Text message
        channel_id: ID kênh Discord (mặc định DISCORD_CHANNEL_ID trong .env)

    Returns:
        True nếu thành công
    """
    if not BOT_TOKEN:
        print("[discord] DISCORD_BOT_TOKEN chưa set — fallback webhook")
        return send_webhook(content)

    cid = channel_id or CHANNEL_ID
    if not cid:
        print("[discord] DISCORD_CHANNEL_ID chưa set")
        return False

    url = f"https://discord.com/api/v10/channels/{cid}/messages"
    headers = {
        "Authorization": f"Bot {BOT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"content": content[:_MAX_CHARS]}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            print(f"[discord] Bot gửi OK → channel {cid}")
            return True
        else:
            print(f"[discord] Bot lỗi {resp.status_code}: {resp.text[:200]}")
            return False
    except requests.RequestException as e:
        print(f"[discord] Bot exception: {e}")
        return False


def send_message(content: str) -> bool:
    """
    Gửi message — tự chọn bot hoặc webhook tuỳ config.
    Ưu tiên webhook nếu có, không thì dùng bot.
    """
    if WEBHOOK_URL:
        return send_webhook(content)
    return send_bot_message(content)


# ──────────────────────────────────────────────
# Formatter — Daily Brief
# ──────────────────────────────────────────────

def format_daily_brief(macro_context: dict, date: str | None = None) -> str:
    """
    Tạo Daily Brief message từ macro context.

    Format:
      📊 DAILY BRIEF — [date]
      Macro: [bias] [score] | [confidence]
      VN: [vn_summary]
      Global: [global_summary]
      Sectors hưởng lợi: ...
      Sectors bị ảnh hưởng: ...
    """
    if date is None:
        date = datetime.now(tz=_VN_TZ).strftime("%Y-%m-%d")

    bias       = macro_context.get("macro_bias", "NEUTRAL")
    score      = macro_context.get("macro_score", 0)
    confidence = macro_context.get("confidence", "LOW")
    vn_sum     = macro_context.get("vn_summary", "")
    global_sum = macro_context.get("global_summary", "")
    reasoning  = macro_context.get("reasoning", "")
    beneficiary = macro_context.get("beneficiary_sectors", [])
    affected    = macro_context.get("affected_sectors", [])
    expert_con  = macro_context.get("expert_consensus", "")

    # Emoji theo bias
    bias_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(bias, "⚪")

    lines = [
        f"📊 **DAILY BRIEF — {date}**",
        f"",
        f"{bias_emoji} **Macro:** {bias} (score {score:+d}) | Độ tin cậy: {confidence}",
    ]

    if vn_sum:
        lines.append(f"🇻🇳 **VN:** {vn_sum}")
    if global_sum:
        lines.append(f"🌍 **Global:** {global_sum}")
    if expert_con:
        lines.append(f"👥 **Chuyên gia:** {expert_con}")

    if reasoning:
        # Giới hạn reasoning để không quá dài
        short_reason = reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
        lines.append(f"")
        lines.append(f"📝 {short_reason}")

    if beneficiary:
        top3 = beneficiary[:3]
        lines.append(f"")
        lines.append(f"✅ **Hưởng lợi:** {' | '.join(top3)}")

    if affected:
        top3 = affected[:3]
        lines.append(f"⚠️ **Chịu áp lực:** {' | '.join(top3)}")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Formatter — Signal Alert (1 mã)
# ──────────────────────────────────────────────

def format_signal_alert(state: dict) -> str:
    """
    Tạo mini analyst note từ TradingState sau pipeline.

    Format:
      🟢 [MUA] VNM — 2026-04-12
      Kỹ thuật | Cơ bản | Khối ngoại | Sentiment
      Entry / SL / TP / %NAV
      Lý do + Risk override nếu có
    """
    symbol   = state.get("symbol", "?")
    date     = state.get("date", "?")
    risk_out = state.get("risk_output", {})
    trader   = state.get("trader_decision", {})
    ptkt     = state.get("ptkt_analysis", {})
    fa       = state.get("fa_analysis", {})
    ff       = state.get("foreign_flow_analysis", {})
    sent     = state.get("sentiment_analysis", {})
    debate   = state.get("debate_synthesis", {})

    action     = risk_out.get("final_action", "CHỜ")
    override   = risk_out.get("override_reason")
    warnings   = risk_out.get("warnings", [])
    sizing_mod = risk_out.get("sizing_modifier", 1.0)
    confidence = trader.get("confidence", "?")
    entry      = trader.get("entry")
    sl         = trader.get("sl")
    tp         = trader.get("tp")
    nav_pct    = trader.get("nav_pct", 0)
    reason     = trader.get("primary_reason", "")
    risks      = trader.get("risks", [])

    action_emoji = {"MUA": "🟢", "BÁN": "🔴", "CHỜ": "🟡"}.get(action, "⚪")

    lines = [
        f"**{action_emoji} [{action}] {symbol}** — {date}",
        f"",
        f"**📈 Kỹ thuật:** {ptkt.get('ma_trend','?')} | Score {ptkt.get('confluence_score','?')}/10 | {ptkt.get('setup_quality','?')}",
        f"**💼 Cơ bản:** {fa.get('valuation','?')} | {fa.get('financial_health','?')}",
        f"**🌐 Khối ngoại:** {ff.get('flow_trend','?')} | Room {ff.get('room_status','?')} | ×{ff.get('sizing_modifier',1.0)}",
        f"**💬 Sentiment:** {sent.get('sentiment_label','?')} ({sent.get('sentiment_score','?')}/100)",
        f"",
    ]

    if action in ("MUA", "BÁN") and entry:
        nav_actual = round(nav_pct * sizing_mod, 1) if sizing_mod != 1.0 else nav_pct
        entry_line = f"**Entry:** {entry:,.0f}"
        if sl and tp:
            rr = round((tp - entry) / (entry - sl), 1) if sl != entry else "?"
            entry_line += f" | **SL:** {sl:,.0f} | **TP:** {tp:,.0f} | R:R {rr}"
        lines.append(entry_line)
        lines.append(f"**Sizing:** {nav_actual}% NAV | **Độ tin cậy:** {confidence}")
        lines.append("")

    if reason:
        lines.append(f"**Lý do:** {reason[:250]}")

    if override:
        lines.append(f"⚠️ **Risk override:** {override}")

    if warnings:
        lines.append(f"⚠️ **Cảnh báo:** {' | '.join(warnings[:2])}")

    if risks:
        lines.append(f"**Rủi ro:** {' | '.join(str(r) for r in risks[:2])}")

    if debate.get("key_risk"):
        lines.append(f"**Key risk:** {debate['key_risk'][:100]}")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Formatter — Batch Summary
# ──────────────────────────────────────────────

def format_batch_summary(results: list[dict], date: str | None = None) -> str:
    """
    Tóm tắt toàn bộ batch — gửi cuối phiên.

    Args:
        results: List TradingState từ orchestrator.run_batch()
        date:    Ngày phân tích
    """
    if date is None:
        date = datetime.now(tz=_VN_TZ).strftime("%Y-%m-%d")

    total = len(results)
    mua_list   = []
    ban_list   = []
    cho_list   = []
    error_list = []

    for r in results:
        symbol = r.get("symbol", "?")
        action = r.get("risk_output", {}).get("final_action", "CHỜ")
        if r.get("error"):
            error_list.append(symbol)
        elif action == "MUA":
            mua_list.append(symbol)
        elif action == "BÁN":
            ban_list.append(symbol)
        else:
            cho_list.append(symbol)

    lines = [
        f"📊 **BATCH SUMMARY — {date}**",
        f"Tổng: {total} mã phân tích",
        f"",
        f"🟢 **MUA ({len(mua_list)}):** {', '.join(mua_list) if mua_list else '—'}",
        f"🔴 **BÁN ({len(ban_list)}):** {', '.join(ban_list) if ban_list else '—'}",
        f"🟡 **CHỜ ({len(cho_list)}):** {', '.join(cho_list[:10]) if cho_list else '—'}",
    ]

    if error_list:
        lines.append(f"❌ **Lỗi ({len(error_list)}):** {', '.join(error_list)}")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Public API — gửi từng loại
# ──────────────────────────────────────────────

def alert_daily_brief(macro_context: dict, date: str | None = None) -> bool:
    """Gửi Daily Brief lên Discord."""
    msg = format_daily_brief(macro_context, date)
    print(f"[discord] Gửi Daily Brief...")
    return send_message(msg)


def alert_signal(state: dict) -> bool:
    """
    Gửi Signal Alert cho 1 mã.
    Chỉ gửi nếu final_action là MUA hoặc BÁN — bỏ qua CHỜ.
    """
    action = state.get("risk_output", {}).get("final_action", "CHỜ")
    if action == "CHỜ":
        print(f"[discord] {state.get('symbol','?')} → CHỜ, không gửi alert")
        return True  # Không lỗi — chỉ skip

    msg = format_signal_alert(state)
    print(f"[discord] Gửi Signal Alert: {state.get('symbol','?')} → {action}")
    return send_message(msg)


def alert_batch_summary(results: list[dict], date: str | None = None) -> bool:
    """Gửi Batch Summary cuối phiên."""
    msg = format_batch_summary(results, date)
    print(f"[discord] Gửi Batch Summary ({len(results)} mã)...")
    return send_message(msg)


def alert_macro_bearish(macro_context: dict) -> bool:
    """
    Gửi cảnh báo đặc biệt khi macro BEARISH — dừng toàn bộ pipeline.
    """
    score = macro_context.get("macro_score", 0)
    risk  = macro_context.get("key_risk", "")
    date  = macro_context.get("date", datetime.now(tz=_VN_TZ).strftime("%Y-%m-%d"))

    msg = (
        f"🚨 **CẢNH BÁO MACRO BEARISH — {date}**\n"
        f"\n"
        f"Macro score: {score:+d} — Đã dừng toàn bộ pipeline\n"
        f"Lý do: {risk[:300] if risk else 'Xem macro report'}\n"
        f"\n"
        f"⛔ Không có trading signal hôm nay. Giữ tiền mặt."
    )
    print(f"[discord] Gửi MACRO BEARISH alert...")
    return send_message(msg)


# ──────────────────────────────────────────────
# Test trực tiếp
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Test formatter (không cần Discord credentials)
    macro = {
        "date": "2026-04-12",
        "macro_bias": "NEUTRAL",
        "macro_score": 0,
        "confidence": "MEDIUM",
        "vn_summary": "USD/VND 26310, SBV 4.5%",
        "global_summary": "S&P500 -0.11%, DXY -0.17%",
        "reasoning": "Dầu giảm hỗ trợ lạm phát. Fed pivot kỳ vọng. VN-Index sideway.",
        "beneficiary_sectors": ["Bán lẻ", "Ngân hàng", "Điện tử"],
        "affected_sectors": ["Hàng không", "Dầu khí", "Bất động sản"],
        "expert_consensus": "MIXED (15 nguồn)",
    }
    print("=== Daily Brief ===")
    print(format_daily_brief(macro))

    print("\n=== Signal Alert (MUA) ===")
    fake_state = {
        "symbol": "VNM",
        "date": "2026-04-12",
        "risk_output": {
            "final_action": "MUA",
            "override_reason": None,
            "warnings": [],
            "sizing_modifier": 1.0,
        },
        "trader_decision": {
            "action": "MUA",
            "entry": 72500,
            "sl": 69000,
            "tp": 82000,
            "nav_pct": 5.0,
            "confidence": "CAO",
            "primary_reason": "Breakout trên MA60, volume xác nhận, FA tốt",
            "risks": ["Thị trường sideway", "Room ngoại 80%"],
        },
        "ptkt_analysis":         {"ma_trend": "UPTREND", "confluence_score": 7.5, "setup_quality": "TỐT"},
        "fa_analysis":           {"valuation": "FAIR", "financial_health": "TỐT"},
        "foreign_flow_analysis": {"flow_trend": "MUA RÒNG", "room_status": "NORMAL", "sizing_modifier": 1.0},
        "sentiment_analysis":    {"sentiment_label": "TÍCH CỰC", "sentiment_score": 72},
        "debate_synthesis":      {"key_risk": "Cần xác nhận breakout phiên tiếp theo"},
    }
    print(format_signal_alert(fake_state))

    print("\n=== Batch Summary ===")
    fake_results = [
        {"symbol": "VNM", "risk_output": {"final_action": "MUA"}, "error": None},
        {"symbol": "HPG", "risk_output": {"final_action": "CHỜ"}, "error": None},
        {"symbol": "FPT", "risk_output": {"final_action": "MUA"}, "error": None},
        {"symbol": "VIC", "risk_output": {"final_action": "BÁN"}, "error": None},
        {"symbol": "MSN", "risk_output": {"final_action": "CHỜ"}, "error": "Timeout"},
    ]
    print(format_batch_summary(fake_results))

    # Gửi thực nếu có webhook
    if WEBHOOK_URL:
        print("\n=== Gửi Daily Brief thật ===")
        alert_daily_brief(macro)
    else:
        print("\n[discord] DISCORD_WEBHOOK_URL chưa set — chỉ test formatter")
