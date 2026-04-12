"""
foreign_flow_agent.py — Phân tích dòng tiền khối ngoại.

Model: Haiku (run_agent_lite) — nhanh + prompt caching
Input:  symbol, foreign flow dict từ fetcher
Output: dict theo schema agents.md #4
"""

import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
from datetime import datetime

from multiagents_trading_assistant.agent import run_agent_lite
from multiagents_trading_assistant import fetcher


_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích dòng tiền khối ngoại thị trường chứng khoán Việt Nam.
Nhiệm vụ: đánh giá room sở hữu nước ngoài và xu hướng dòng tiền khối ngoại.

Quy tắc bắt buộc:
- Trả về JSON hợp lệ DUY NHẤT, không có text thừa
- room_status: "CRITICAL" (>95%) | "HIGH" (90-95%) | "MEDIUM" (80-90%) | "NORMAL" (<80%) | "UNKNOWN"
- sizing_modifier: CRITICAL→0.0, HIGH→0.5, MEDIUM→0.8, NORMAL→1.0
- flow_trend: "BUYING" (mua ròng) | "SELLING" (bán ròng) | "NEUTRAL" | "UNKNOWN"
- accumulation_signal: true nếu khối ngoại mua ròng khi VN-Index giảm (smart money)

Output schema:
{
  "room_usage_pct": <float|null>,
  "room_status": <str>,
  "net_flow_5d": <float|null>,
  "flow_trend": <str>,
  "accumulation_signal": <bool>,
  "sizing_modifier": <float 0.0-1.0>,
  "foreign_summary": <str — 1-2 câu tiếng Việt tóm tắt>
}"""


def analyze(symbol: str, date: str | None = None) -> dict:
    """
    Chạy Foreign Flow agent cho một mã.

    Args:
        symbol: Mã cổ phiếu (VD: "VNM")
        date:   Ngày phân tích (YYYY-MM-DD), mặc định hôm nay

    Returns:
        dict theo schema foreign_flow_agent (agents.md #4)
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    print(f"[foreign_flow_agent] Bắt đầu phân tích khối ngoại {symbol} ({date})...")

    flow_data = fetcher.get_foreign_flow(symbol)
    if not flow_data or flow_data.get("room_usage_pct") is None and flow_data.get("net_flow_5d") is None:
        print(f"[foreign_flow_agent] Không có dữ liệu khối ngoại cho {symbol}.")
        return _empty_result()

    room_pct   = flow_data.get("room_usage_pct")
    net_5d     = flow_data.get("net_flow_5d")
    net_20d    = flow_data.get("net_flow_20d")
    history    = flow_data.get("flow_history", [])

    # Tóm tắt lịch sử flow (tối đa 10 ngày gần nhất)
    history_str = _format_history(history[-10:]) if history else "Không có lịch sử."

    prompt = f"""Phân tích dòng tiền khối ngoại mã {symbol} ngày {date}.

=== Room sở hữu nước ngoài ===
Room sử dụng: {_fmt_pct(room_pct)}

=== Net flow ===
Net flow 5 phiên: {_fmt_flow(net_5d)}
Net flow 20 phiên: {_fmt_flow(net_20d)}

=== Lịch sử giao dịch (10 phiên gần nhất) ===
{history_str}

Đánh giá:
- room_status dựa trên room_usage_pct
- flow_trend dựa trên net_flow_5d và net_flow_20d
- accumulation_signal: khối ngoại mua ròng nhất quán 5+ phiên gần đây
- sizing_modifier theo quy tắc: CRITICAL=0.0, HIGH=0.5, MEDIUM=0.8, NORMAL=1.0

Trả về JSON theo schema đã định."""

    try:
        result = run_agent_lite(prompt=prompt, system=_SYSTEM_PROMPT)
        print(f"[foreign_flow_agent] {symbol} — room_status={result.get('room_status')}, "
              f"flow_trend={result.get('flow_trend')}, sizing={result.get('sizing_modifier')}")
        return result
    except Exception as e:
        print(f"[foreign_flow_agent] Lỗi gọi LLM cho {symbol}: {e}")
        return _fallback_result(flow_data)


def _format_history(history: list[dict]) -> str:
    """Format lịch sử flow thành text."""
    lines = []
    for h in history:
        date_str = str(h.get("date", ""))[:10]
        net = h.get("net_flow", 0) or 0
        sign = "+" if net >= 0 else ""
        lines.append(f"  {date_str}: {sign}{net/1_000_000:.1f} tỷ VNĐ")
    return "\n".join(lines) if lines else "Không có dữ liệu."


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f}%"


def _fmt_flow(val) -> str:
    if val is None:
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val/1_000_000_000:.2f} tỷ VNĐ"


def _empty_result() -> dict:
    return {
        "room_usage_pct": None,
        "room_status": "UNKNOWN",
        "net_flow_5d": None,
        "flow_trend": "UNKNOWN",
        "accumulation_signal": False,
        "sizing_modifier": 1.0,
        "foreign_summary": "Không có dữ liệu khối ngoại.",
    }


def _fallback_result(flow_data: dict) -> dict:
    """Tính kết quả rule-based khi LLM fail."""
    room_pct = flow_data.get("room_usage_pct")
    net_5d   = flow_data.get("net_flow_5d")

    # Room status
    room_status = "UNKNOWN"
    sizing = 1.0
    if room_pct is not None:
        if room_pct > 95:
            room_status, sizing = "CRITICAL", 0.0
        elif room_pct > 90:
            room_status, sizing = "HIGH", 0.5
        elif room_pct > 80:
            room_status, sizing = "MEDIUM", 0.8
        else:
            room_status, sizing = "NORMAL", 1.0

    # Flow trend
    flow_trend = "UNKNOWN"
    if net_5d is not None:
        flow_trend = "BUYING" if net_5d > 0 else ("SELLING" if net_5d < 0 else "NEUTRAL")

    return {
        "room_usage_pct": room_pct,
        "room_status": room_status,
        "net_flow_5d": net_5d,
        "flow_trend": flow_trend,
        "accumulation_signal": (net_5d or 0) > 5_000_000_000,  # >5 tỷ mua ròng
        "sizing_modifier": sizing,
        "foreign_summary": f"Room {room_pct or 'N/A'}%, net flow 5D {_fmt_flow(net_5d)}.",
    }
