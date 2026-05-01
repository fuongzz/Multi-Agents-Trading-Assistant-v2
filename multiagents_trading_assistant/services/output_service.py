"""Output service — thay thế Discord bot.

Ghi tín hiệu ra:
  1. stdout (có màu qua ANSI escape — Windows 10+ hỗ trợ)
  2. file JSON tại cache/signals/{invest|trade}/{date}_{symbol}.json

Khi nào muốn nối Discord lại, chỉ cần sửa hai hàm write_*_signal để
thêm bước gọi webhook.
"""

import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import requests

_BASE_DIR     = Path(__file__).resolve().parent.parent
_SIGNAL_DIR   = _BASE_DIR / "cache" / "signals"
_INVEST_DIR   = _SIGNAL_DIR / "invest"
_TRADE_DIR    = _SIGNAL_DIR / "trade"

for _d in (_INVEST_DIR, _TRADE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── ANSI colors ──

_RESET  = "\x1b[0m"
_BOLD   = "\x1b[1m"
_GREEN  = "\x1b[32m"
_TEAL   = "\x1b[36m"
_ORANGE = "\x1b[33m"
_RED    = "\x1b[31m"
_GREY   = "\x1b[90m"


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def write_invest_signal(state: dict, formatted_text: str) -> Path:
    """In invest signal ra stdout + ghi JSON + gửi Discord."""
    date   = state.get("date", datetime.now().strftime("%Y-%m-%d"))
    symbol = state.get("symbol", "UNKNOWN")

    print(f"\n{_TEAL}{_BOLD}══ INVEST SIGNAL ══{_RESET}")
    print(f"{_TEAL}{formatted_text}{_RESET}")

    out_path = _INVEST_DIR / f"{date}_{symbol}.json"
    _write_json(out_path, state)
    print(f"{_GREY}  → saved: {out_path.relative_to(_BASE_DIR)}{_RESET}\n")

    webhook = os.getenv("DISCORD_WEBHOOK_INVEST") or os.getenv("DISCORD_WEBHOOK_URL", "")
    _send_signal_to_discord(webhook, _invest_embed(state))

    return out_path


def write_trade_signal(state: dict, formatted_text: str) -> Path:
    """In trade signal ra stdout + ghi JSON + gửi Discord."""
    date   = state.get("date", datetime.now().strftime("%Y-%m-%d"))
    symbol = state.get("symbol", "UNKNOWN")

    print(f"\n{_ORANGE}{_BOLD}══ TRADE SIGNAL ══{_RESET}")
    print(f"{_ORANGE}{formatted_text}{_RESET}")

    out_path = _TRADE_DIR / f"{date}_{symbol}.json"
    _write_json(out_path, state)
    print(f"{_GREY}  → saved: {out_path.relative_to(_BASE_DIR)}{_RESET}\n")

    webhook = os.getenv("DISCORD_WEBHOOK_TRADE") or os.getenv("DISCORD_WEBHOOK_URL", "")
    _send_signal_to_discord(webhook, _trade_embed(state))

    return out_path


def write_summary(pipeline: str, total: int, actionable: int, date: str) -> None:
    """Tóm tắt cuối run."""
    color = _TEAL if pipeline == "invest" else _ORANGE
    print(f"\n{color}{_BOLD}══ {pipeline.upper()} SUMMARY — {date} ══{_RESET}")
    print(f"{color}  Processed: {total} | Actionable (MUA): {actionable}{_RESET}")
    print(f"{_GREY}  Output dir: cache/signals/{pipeline}/{_RESET}\n")


# ──────────────────────────────────────────────
# Internal
# ──────────────────────────────────────────────

def _send_signal_to_discord(webhook_url: str, embed: dict) -> None:
    """POST 1 embed lên Discord webhook. Không raise exception."""
    if not webhook_url:
        return
    try:
        resp = requests.post(webhook_url, json={"embeds": [embed]}, timeout=5)
        if resp.status_code not in (200, 204):
            print(f"[output_service] Discord signal HTTP {resp.status_code}")
    except Exception as e:
        print(f"[output_service] Discord signal fail: {e}")


def _trade_embed(state: dict) -> dict:
    """Build Discord embed cho Trade signal."""
    symbol  = state.get("symbol", "?")
    date    = state.get("date", "?")
    setup   = state.get("setup_type", "?")
    trader  = state.get("trader_decision", {})
    risk    = state.get("risk_output", {})
    synth   = state.get("synthesis", {})

    action  = risk.get("final_action") or trader.get("action", "CHO")
    icon    = {"MUA": "🟢", "CHO": "🟡", "TRANH": "🔴"}.get(action, "⚪")
    color   = {"MUA": 0xFF8C00, "CHO": 0xFFCC00, "TRANH": 0xFF4444}.get(action, 0x888888)

    conf    = synth.get("confluence_score", "?")
    reason  = str(trader.get("primary_reason", ""))[:200]
    override = risk.get("override_reason", "")

    fields = [
        {"name": "Setup",       "value": setup,              "inline": True},
        {"name": "Confluence",  "value": f"{conf}/100",      "inline": True},
        {"name": "Confidence",  "value": trader.get("confidence", "?"), "inline": True},
    ]

    if action == "MUA" and trader.get("entry_zone"):
        ez  = trader["entry_zone"]
        sl  = trader.get("stop_loss")
        target = trader.get("initial_target") or trader.get("take_profit")
        rr  = trader.get("rr_ratio", "?")
        pos = int((trader.get("position_pct", 0) or 0) * risk.get("sizing_modifier", 1.0))
        fields += [
            {"name": "Entry",   "value": f"{ez[0]:,.0f} – {ez[1]:,.0f}", "inline": True},
            {"name": "SL",      "value": f"{sl:,.0f}" if sl else "N/A",   "inline": True},
            {"name": "Target",  "value": f"{target:,.0f}" if target else "N/A", "inline": True},
            {"name": "R:R",     "value": str(rr),  "inline": True},
            {"name": "Size",    "value": f"{pos}% NAV", "inline": True},
        ]

    if reason:
        fields.append({"name": "Ly do", "value": reason, "inline": False})
    if override:
        fields.append({"name": "Risk override", "value": override, "inline": False})

    return {
        "title":       f"{icon} [TRADE] {symbol} — {action}",
        "description": f"Ngay: {date}",
        "color":       color,
        "fields":      fields,
        "footer":      {"text": "AI Trading Assistant"},
    }


def _invest_embed(state: dict) -> dict:
    """Build Discord embed cho Invest signal."""
    symbol  = state.get("symbol", "?")
    date    = state.get("date", "?")
    trader  = state.get("trader_decision", {})
    risk    = state.get("risk_output", {})
    val     = state.get("valuation_analysis", {})

    action  = risk.get("final_action") or trader.get("action", "CHO")
    icon    = {"MUA": "🟢", "CHO": "🟡", "TRANH": "🔴"}.get(action, "⚪")
    color   = {"MUA": 0x00AA88, "CHO": 0xFFCC00, "TRANH": 0xFF4444}.get(action, 0x888888)

    target  = trader.get("target_price")
    mos     = val.get("margin_of_safety")
    horizon = trader.get("holding_horizon", "?")
    reason  = str(trader.get("primary_reason", ""))[:200]
    exit_c  = str(trader.get("exit_condition", ""))[:150]
    override = risk.get("override_reason", "")

    fields = [
        {"name": "Horizon",    "value": horizon,                              "inline": True},
        {"name": "Confidence", "value": trader.get("confidence", "?"),        "inline": True},
        {"name": "Target",     "value": f"{target:,.0f}" if target else "N/A","inline": True},
    ]
    if mos is not None:
        fields.append({"name": "Margin of Safety", "value": f"{mos:.1f}%", "inline": True})
    if reason:
        fields.append({"name": "Luan diem",    "value": reason,  "inline": False})
    if exit_c:
        fields.append({"name": "Exit khi nao", "value": exit_c,  "inline": False})
    if override:
        fields.append({"name": "Risk override","value": override, "inline": False})

    return {
        "title":       f"{icon} [INVEST] {symbol} — {action}",
        "description": f"Ngay: {date}",
        "color":       color,
        "fields":      fields,
        "footer":      {"text": "AI Trading Assistant"},
    }


def send_pipeline_alert(
    pipeline: str,
    error: Exception | str,
    symbol: str | None = None,
    date: str | None = None,
) -> None:
    """Gửi Discord alert khi pipeline fail.

    Dùng webhook — không cần bot token. Gọi từ pipeline_runner trong except block.
    Không raise exception — alert không được làm crash thêm.
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        print("[output_service] DISCORD_WEBHOOK_URL chưa set — bỏ qua alert")
        return

    date_str   = date or datetime.now().strftime("%Y-%m-%d %H:%M")
    symbol_str = f" | {symbol}" if symbol else ""
    err_text   = str(error)[:800]
    tb_text    = traceback.format_exc()[-600:] if not isinstance(error, str) else ""

    # Color: đỏ cho pipeline fail, vàng cho symbol fail riêng lẻ
    color = 0xFF4444 if symbol is None else 0xFFA500

    payload = {
        "embeds": [{
            "title":       f"⚠️ [{pipeline.upper()}] Pipeline Alert{symbol_str}",
            "description": f"**Lỗi:** {err_text}",
            "color":       color,
            "fields": [
                {"name": "Thời gian", "value": date_str, "inline": True},
                {"name": "Pipeline",  "value": pipeline, "inline": True},
            ],
            "footer": {"text": "AI Trading Assistant — Auto Monitor"},
        }]
    }
    if tb_text:
        payload["embeds"][0]["fields"].append(
            {"name": "Traceback", "value": f"```{tb_text}```", "inline": False}
        )

    try:
        resp = requests.post(webhook_url, json=payload, timeout=5)
        if resp.status_code not in (200, 204):
            print(f"[output_service] Discord alert HTTP {resp.status_code}")
    except Exception as e:
        print(f"[output_service] Discord alert fail: {e}")


def _write_json(path: Path, state: dict) -> None:
    """Serialize state (skip non-JSON-serializable values)."""
    try:
        # Deep-filter các giá trị serializable được
        clean = _clean_for_json(state)
        path.write_text(json.dumps(clean, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        print(f"{_RED}[output_service] Lỗi ghi {path.name}: {e}{_RESET}")


def _clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(x) for x in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    # fallback — stringify
    return str(obj)


# ──────────────────────────────────────────────
# Portfolio management — exit alerts & summary
# ──────────────────────────────────────────────

def send_exit_alert(exit_signal: dict) -> None:
    """Gửi Discord alert khi một vị thế được đóng (SL/TP/momentum)."""
    webhook = os.getenv("DISCORD_WEBHOOK_TRADE") or os.getenv("DISCORD_WEBHOOK_URL", "")

    symbol = exit_signal.get("symbol", "?")
    exit_type = exit_signal.get("exit_type", "UNKNOWN")
    entry_price = exit_signal.get("entry_price", 0)
    current_price = exit_signal.get("current_price", 0)
    pnl_pct = exit_signal.get("unrealized_pnl_pct", 0)
    pnl_vnd = exit_signal.get("unrealized_pnl_vnd", 0)
    sl = exit_signal.get("sl")
    tp = exit_signal.get("tp")
    days_held = exit_signal.get("days_held", 0)
    strategy = exit_signal.get("strategy", "?")

    # Màu theo loại thoát
    color_map = {"SL_HIT": 0xFF4444, "TP_HIT": 0x00CC66, "MOMENTUM_LOSS": 0xFF8C00}
    color = color_map.get(exit_type, 0x888888)

    icon_map = {"SL_HIT": "🔴", "TP_HIT": "🟢", "MOMENTUM_LOSS": "🟠"}
    icon = icon_map.get(exit_type, "⚪")

    pnl_sign = "+" if pnl_pct >= 0 else ""
    fields = [
        {"name": "Lý do",   "value": exit_type,                              "inline": True},
        {"name": "Giá vào", "value": f"{entry_price:,.0f}",                  "inline": True},
        {"name": "Giá ra",  "value": f"{current_price:,.0f}",                "inline": True},
        {"name": "P&L",     "value": f"{pnl_sign}{pnl_pct:.1f}% ({pnl_sign}{pnl_vnd:,.0f} VNĐ)", "inline": True},
        {"name": "Giữ",     "value": f"{days_held} ngày",                    "inline": True},
        {"name": "Setup",   "value": strategy,                               "inline": True},
    ]
    if sl:
        fields.append({"name": "SL", "value": f"{sl:,.0f}", "inline": True})
    if tp:
        fields.append({"name": "TP", "value": f"{tp:,.0f}", "inline": True})

    embed = {
        "title":       f"{icon} [EXIT] {symbol} — {exit_type}",
        "color":       color,
        "fields":      fields,
        "footer":      {"text": "AI Trading Assistant — Portfolio Monitor"},
    }

    # stdout
    print(f"\n{_RED if exit_type == 'SL_HIT' else _GREEN}{_BOLD}══ EXIT SIGNAL ══{_RESET}")
    print(f"  {symbol} | {exit_type} | {pnl_sign}{pnl_pct:.1f}% @ {current_price:,.0f}")

    _send_signal_to_discord(webhook, embed)


def send_portfolio_summary(positions: list[dict], stats: dict, date: str) -> None:
    """Gửi Discord embed tóm tắt danh mục hàng sáng."""
    webhook = os.getenv("DISCORD_WEBHOOK_TRADE") or os.getenv("DISCORD_WEBHOOK_URL", "")

    open_count = len(positions)
    win_rate = stats.get("win_rate", 0)
    total_trades = stats.get("total_trades", 0)
    total_pnl = stats.get("total_realized_pnl_vnd", 0)
    avg_rr = stats.get("avg_realized_rr", 0)
    wins = stats.get("win_trades", 0)

    pnl_sign = "+" if total_pnl >= 0 else ""

    if open_count == 0:
        description = "Không có vị thế đang mở."
    else:
        description = f"**{open_count} vị thế đang mở** | Win rate: {win_rate:.0f}% ({total_trades} GD đã đóng)"

    fields = []
    for pos in positions[:5]:  # tối đa 5
        sym = pos.get("symbol", "?")
        entry = pos.get("entry_price", 0)
        current = pos.get("current_price", entry)
        pnl = pos.get("unrealized_pnl_pct", 0)
        sl_dist = pos.get("distance_to_sl_pct")
        tp_dist = pos.get("distance_to_tp_pct")
        target_review = pos.get("target_review") or {}

        pnl_str = f"{'+' if pnl >= 0 else ''}{pnl:.1f}%"
        sl_str = f"SL -{abs(sl_dist):.1f}%" if sl_dist is not None else ""
        if tp_dist is None:
            tp_str = ""
        elif tp_dist >= 0:
            tp_str = f"Target +{tp_dist:.1f}%"
        else:
            tp_str = f"Vượt target {abs(tp_dist):.1f}%"
        detail = f"{entry:,.0f}→{current:,.0f} | P&L {pnl_str}"
        if sl_str:
            detail += f" | {sl_str}"
        if tp_str:
            detail += f" | {tp_str}"
        if target_review.get("should_alert"):
            detail += f" | Target: {target_review.get('recommendation')}"

        fields.append({"name": sym, "value": detail, "inline": False})

    if len(positions) > 5:
        fields.append({"name": "...", "value": f"và {len(positions)-5} vị thế khác", "inline": False})

    fields.append({
        "name": "Tổng kết",
        "value": f"Realized P&L: {pnl_sign}{total_pnl:,.0f} VNĐ | Avg R:R: {avg_rr:.2f} | Win: {wins}/{total_trades}",
        "inline": False,
    })

    embed = {
        "title":       f"📊 Portfolio — {date}",
        "description": description,
        "color":       0x5865F2,
        "fields":      fields,
        "footer":      {"text": "AI Trading Assistant — Portfolio Monitor"},
    }

    # stdout
    print(f"\n{_TEAL}{_BOLD}══ PORTFOLIO SUMMARY {date} ══{_RESET}")
    print(f"  Đang giữ: {open_count} | Win rate: {win_rate:.0f}% | Realized P&L: {pnl_sign}{total_pnl:,.0f} VNĐ")

    _send_signal_to_discord(webhook, embed)
