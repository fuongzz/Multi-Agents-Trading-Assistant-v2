"""flow_agent.py — Dòng tiền khối ngoại cho Trade pipeline.

Model: Haiku. Port từ _legacy/agents/foreign_flow_agent.py (không đổi logic).
"""

import importlib.metadata  # noqa: F401
from datetime import datetime

from multiagents_trading_assistant.services.llm_service import run_agent_lite
from multiagents_trading_assistant import fetcher


_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích dòng tiền khối ngoại thị trường VN.

Quy tắc bắt buộc:
- Trả về JSON hợp lệ DUY NHẤT
- room_status: CRITICAL (>95%) | HIGH (90-95%) | MEDIUM (80-90%) | NORMAL (<80%) | UNKNOWN
- sizing_modifier: CRITICAL→0.0, HIGH→0.5, MEDIUM→0.8, NORMAL→1.0
- flow_trend: BUYING | SELLING | NEUTRAL | UNKNOWN
- accumulation_signal: true nếu NN mua ròng 5+ phiên khi VN-Index giảm

Output schema:
{
  "room_usage_pct": <float|null>,
  "room_status": <str>,
  "net_flow_5d": <float|null>,
  "flow_trend": <str>,
  "accumulation_signal": <bool>,
  "sizing_modifier": <float>,
  "foreign_summary": <str>
}"""


def analyze(symbol: str, date: str | None = None) -> dict:
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    print(f"[flow_agent] {symbol} ({date})")

    flow_data = fetcher.get_foreign_flow(symbol)
    if not flow_data or (flow_data.get("room_usage_pct") is None and flow_data.get("net_flow_5d") is None):
        return _empty_result()

    room_pct = flow_data.get("room_usage_pct")
    net_5d   = flow_data.get("net_flow_5d")
    net_20d  = flow_data.get("net_flow_20d")
    history  = flow_data.get("flow_history", [])

    history_str = _format_history(history[-10:]) if history else "Không có lịch sử."

    prompt = f"""Phân tích dòng tiền khối ngoại {symbol} ngày {date}.

=== Room ===
{_fmt_pct(room_pct)}

=== Net flow ===
5d: {_fmt_flow(net_5d)}
20d: {_fmt_flow(net_20d)}

=== 10 phiên gần ===
{history_str}

Trả JSON theo schema."""

    try:
        result = run_agent_lite(prompt=prompt, system=_SYSTEM_PROMPT)
        print(f"[flow_agent] {symbol} — room={result.get('room_status')}, flow={result.get('flow_trend')}, size={result.get('sizing_modifier')}")
        return result
    except Exception as e:
        print(f"[flow_agent] LLM error: {e}")
        return _fallback_result(flow_data)


def _format_history(history: list[dict]) -> str:
    lines = []
    for h in history:
        date_str = str(h.get("date", ""))[:10]
        net = h.get("net_flow", 0) or 0
        sign = "+" if net >= 0 else ""
        lines.append(f"  {date_str}: {sign}{net/1_000_000:.1f} tỷ")
    return "\n".join(lines) if lines else "Không có."


def _fmt_pct(val) -> str:
    return "N/A" if val is None else f"{val:.1f}%"


def _fmt_flow(val) -> str:
    if val is None:
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val/1_000_000_000:.2f} tỷ"


def _empty_result() -> dict:
    return {
        "room_usage_pct": None, "room_status": "UNKNOWN",
        "net_flow_5d": None, "flow_trend": "UNKNOWN",
        "accumulation_signal": False, "sizing_modifier": 1.0,
        "foreign_summary": "Không có dữ liệu khối ngoại.",
    }


def _fallback_result(flow_data: dict) -> dict:
    room_pct = flow_data.get("room_usage_pct")
    net_5d   = flow_data.get("net_flow_5d")

    status, sizing = "UNKNOWN", 1.0
    if room_pct is not None:
        if room_pct > 95:   status, sizing = "CRITICAL", 0.0
        elif room_pct > 90: status, sizing = "HIGH", 0.5
        elif room_pct > 80: status, sizing = "MEDIUM", 0.8
        else:                status, sizing = "NORMAL", 1.0

    trend = "UNKNOWN"
    if net_5d is not None:
        trend = "BUYING" if net_5d > 0 else ("SELLING" if net_5d < 0 else "NEUTRAL")

    return {
        "room_usage_pct": room_pct,
        "room_status": status,
        "net_flow_5d": net_5d,
        "flow_trend": trend,
        "accumulation_signal": (net_5d or 0) > 5_000_000_000,
        "sizing_modifier": sizing,
        "foreign_summary": f"Room {room_pct or 'N/A'}%, net 5D {_fmt_flow(net_5d)}.",
    }
