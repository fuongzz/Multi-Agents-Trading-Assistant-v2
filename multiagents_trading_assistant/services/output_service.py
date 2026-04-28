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
        tp  = trader.get("take_profit")
        rr  = trader.get("rr_ratio", "?")
        pos = int((trader.get("position_pct", 0) or 0) * risk.get("sizing_modifier", 1.0))
        fields += [
            {"name": "Entry",   "value": f"{ez[0]:,.0f} – {ez[1]:,.0f}", "inline": True},
            {"name": "SL",      "value": f"{sl:,.0f}" if sl else "N/A",   "inline": True},
            {"name": "TP",      "value": f"{tp:,.0f}" if tp else "N/A",   "inline": True},
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
