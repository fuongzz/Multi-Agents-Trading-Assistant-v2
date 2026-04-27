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
    """In invest signal ra stdout + ghi JSON.

    Args:
        state:          InvestState cuối pipeline
        formatted_text: text đã format sẵn từ formatters/invest_output.py

    Returns:
        path file JSON đã ghi
    """
    date   = state.get("date", datetime.now().strftime("%Y-%m-%d"))
    symbol = state.get("symbol", "UNKNOWN")

    print(f"\n{_TEAL}{_BOLD}══ INVEST SIGNAL ══{_RESET}")
    print(f"{_TEAL}{formatted_text}{_RESET}")

    out_path = _INVEST_DIR / f"{date}_{symbol}.json"
    _write_json(out_path, state)
    print(f"{_GREY}  → saved: {out_path.relative_to(_BASE_DIR)}{_RESET}\n")
    return out_path


def write_trade_signal(state: dict, formatted_text: str) -> Path:
    """In trade signal ra stdout + ghi JSON."""
    date   = state.get("date", datetime.now().strftime("%Y-%m-%d"))
    symbol = state.get("symbol", "UNKNOWN")

    print(f"\n{_ORANGE}{_BOLD}══ TRADE SIGNAL ══{_RESET}")
    print(f"{_ORANGE}{formatted_text}{_RESET}")

    out_path = _TRADE_DIR / f"{date}_{symbol}.json"
    _write_json(out_path, state)
    print(f"{_GREY}  → saved: {out_path.relative_to(_BASE_DIR)}{_RESET}\n")
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
