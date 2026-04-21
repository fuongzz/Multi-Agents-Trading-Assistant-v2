"""synthesis_agent.py — Tổng hợp tín hiệu TA + Flow + Sentiment.

Tạo ra confluence_score 0-100 để trader_trade dùng làm bộ lọc chính.
Logic chính là rule-based (trọng số cố định); LLM chỉ để viết drivers/blockers tiếng Việt.

Weights (tổng 1.0):
  technical  0.50  — confluence_score (0-10) × 10 = 0-100
  flow       0.30  — mapped từ flow_trend + accumulation_signal + room_status
  sentiment  0.20  — sentiment_score (0-100) trực tiếp

Setup quality derived:
  STRONG ≥ 70  |  MEDIUM 50-69  |  WEAK < 50
"""

import importlib.metadata  # noqa: F401

from multiagents_trading_assistant.services.llm_service import run_agent_lite


_W_TECH, _W_FLOW, _W_SENT = 0.50, 0.30, 0.20

_SYSTEM_PROMPT = """Bạn là analyst tổng hợp tín hiệu trading VN.
Nhiệm vụ: nhận confluence_score rule-based + 3 phân tích thành phần, viết drivers/blockers ngắn gọn tiếng Việt.

Quy tắc bắt buộc:
- Trả JSON hợp lệ DUY NHẤT
- drivers: tối đa 3 điểm ủng hộ MUA (tiếng Việt, ngắn gọn)
- blockers: tối đa 3 điểm cản trở MUA
- synthesis_summary: 1-2 câu

Output schema:
{
  "drivers": [<str>, ...],
  "blockers": [<str>, ...],
  "synthesis_summary": <str>
}"""


def _flow_points(flow: dict) -> float:
    """Map flow_agent output → 0-100 score."""
    room = flow.get("room_status", "UNKNOWN")
    trend = flow.get("flow_trend", "UNKNOWN")
    accum = bool(flow.get("accumulation_signal"))

    if room == "CRITICAL":
        return 0.0

    base = {
        "BUYING": 70.0,
        "NEUTRAL": 50.0,
        "SELLING": 25.0,
        "UNKNOWN": 40.0,
    }.get(trend, 40.0)

    if accum:
        base += 15.0
    if room == "HIGH":
        base -= 15.0
    elif room == "MEDIUM":
        base -= 5.0
    return max(0.0, min(100.0, base))


def run(
    technical_analysis: dict,
    foreign_flow_analysis: dict,
    sentiment_analysis: dict,
    setup_type: str,
) -> dict:
    """Tính confluence_score 0-100 + gọi LLM viết drivers/blockers."""
    tech_score = float(technical_analysis.get("confluence_score") or 0) * 10.0
    flow_score = _flow_points(foreign_flow_analysis)
    sent_score = float(sentiment_analysis.get("sentiment_score") or 50)

    confluence = _W_TECH * tech_score + _W_FLOW * flow_score + _W_SENT * sent_score
    confluence = round(confluence, 1)

    if confluence >= 70:
        quality = "STRONG"
    elif confluence >= 50:
        quality = "MEDIUM"
    else:
        quality = "WEAK"

    prompt = f"""Setup type: {setup_type}
Confluence score: {confluence}/100 ({quality})

Thành phần:
- Technical ({tech_score:.0f}/100): {technical_analysis.get('technical_summary', '')}
  Trend {technical_analysis.get('ma_trend')}, RSI {technical_analysis.get('rsi_signal')}, MACD {technical_analysis.get('macd_signal')}
- Flow ({flow_score:.0f}/100): {foreign_flow_analysis.get('foreign_summary', '')}
  Room {foreign_flow_analysis.get('room_status')}, trend {foreign_flow_analysis.get('flow_trend')}
- Sentiment ({sent_score:.0f}/100): {sentiment_analysis.get('sentiment_summary', '')}
  Positive: {sentiment_analysis.get('key_positive', [])}
  Negative: {sentiment_analysis.get('key_negative', [])}

Viết drivers (điểm cộng) và blockers (điểm trừ) ngắn gọn, synthesis_summary 1-2 câu tiếng Việt.
Trả JSON theo schema."""

    try:
        llm_out = run_agent_lite(prompt=prompt, system=_SYSTEM_PROMPT)
    except Exception as e:
        print(f"[synthesis_agent] LLM error: {e} — fallback")
        llm_out = {
            "drivers": [],
            "blockers": [],
            "synthesis_summary": f"Confluence {confluence}/100 ({quality}).",
        }

    return {
        "confluence_score": confluence,
        "setup_quality": quality,
        "tech_score": round(tech_score, 1),
        "flow_score": round(flow_score, 1),
        "sentiment_score": round(sent_score, 1),
        "drivers": llm_out.get("drivers", []),
        "blockers": llm_out.get("blockers", []),
        "synthesis_summary": llm_out.get("synthesis_summary", ""),
    }
