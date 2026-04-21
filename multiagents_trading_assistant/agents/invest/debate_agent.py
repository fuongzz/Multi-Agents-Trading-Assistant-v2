"""debate_agent.py — Bull/Bear debate cho Investment pipeline (thesis dài hạn).

Port từ _legacy/research/debate_agent.py.
Điều chỉnh prompt: Bull/Bear phải lập luận theo khung DÀI HẠN (quý/năm),
dựa trên valuation gap và catalyst cơ bản — KHÔNG dùng RSI/MACD.

Model: Sonnet (Bull/Bear), Haiku (Synthesizer).
"""

import importlib.metadata  # noqa: F401

from multiagents_trading_assistant.services.llm_service import run_agent, run_agent_lite
from multiagents_trading_assistant.agent import _get_client, MODEL_SONNET, MAX_TOKENS_SONNET


_BULL_SYSTEM = """Bạn là chuyên gia đầu tư giá trị (Value Investor) vai BULL.
Nhiệm vụ: xây dựng luận điểm MUA dài hạn (3-12 tháng) dựa trên nền tảng cơ bản.

Quy tắc:
- Tập trung vào: margin of safety, tăng trưởng EPS bền vững, ROE cao, valuation gap
- KHÔNG dùng RSI/MACD hay tín hiệu kỹ thuật ngắn hạn
- Trích dẫn cụ thể: P/E vs ngành, ROE, EPS growth, intrinsic value
- Thừa nhận rủi ro nhưng lập luận tại sao story dài hạn còn nguyên
- Viết tiếng Việt, súc tích (150-250 từ), kết thúc bằng 1 câu tóm tắt BULL"""

_BEAR_SYSTEM = """Bạn là chuyên gia đầu tư thận trọng vai BEAR.
Nhiệm vụ: phản biện luận điểm Bull theo khung dài hạn (3-12 tháng).

Quy tắc:
- Tập trung vào: định giá quá cao, tăng trưởng không bền vững, rủi ro macro, cạnh tranh
- KHÔNG dùng RSI/MACD
- Phản bác trực tiếp các số liệu Bull đưa ra
- Xem xét: chu kỳ ngành, nợ/vốn, margin xu hướng, catalyst có còn không
- Viết tiếng Việt, súc tích (150-250 từ), kết thúc bằng 1 câu tóm tắt BEAR"""

_SYNTHESIZER_SYSTEM = """Bạn là chuyên gia tổng hợp đầu tư trung lập (Investment pipeline).
Nhiệm vụ: tổng hợp debate Bull vs Bear dài hạn thành đánh giá cân bằng.

Quy tắc bắt buộc:
- Trả về JSON hợp lệ DUY NHẤT
- bull_key_points: tối đa 3 điểm mạnh nhất (tiếng Việt, ngắn, dài hạn)
- bear_key_points: tối đa 3 điểm mạnh nhất (tiếng Việt, ngắn, dài hạn)
- balance: "STRONG_BULL" | "BULL_SLIGHT_EDGE" | "NEUTRAL" | "BEAR_SLIGHT_EDGE" | "STRONG_BEAR"
- key_risk: rủi ro dài hạn lớn nhất (1 câu)
- debate_conclusion: kết luận 1-2 câu — có nên giữ dài hạn không

Output schema:
{
  "bull_key_points": [<str>, ...],
  "bear_key_points": [<str>, ...],
  "balance": <str>,
  "key_risk": <str>,
  "debate_conclusion": <str>
}"""


def run_bull(state: dict) -> dict:
    symbol = state["symbol"]
    context = _build_context(state)
    prompt = f"""Xây dựng luận điểm BULL dài hạn (3-12 tháng) cho mã {symbol}.

{context}

Lập luận tại sao NÊN ĐẦU TƯ {symbol} theo khung dài hạn — valuation gap, story tăng trưởng.
Viết tiếng Việt (150-250 từ)."""

    print(f"[debate_agent] BULL → {symbol}")
    try:
        text = _run_text(prompt, _BULL_SYSTEM)
        return {"bull_argument": text}
    except Exception as e:
        print(f"[debate_agent] BULL error: {e}")
        return {"bull_argument": f"[BULL lỗi]: {e}"}


def run_bear(state: dict) -> dict:
    symbol = state["symbol"]
    bull = state.get("bull_argument", "")
    context = _build_context(state)

    prompt_r1 = f"""Phản bác luận điểm BULL dài hạn sau cho mã {symbol}.

{context}

=== Luận điểm BULL ===
{bull}

Xây dựng case KHÔNG NÊN ĐẦU TƯ hoặc CHỜ theo khung dài hạn.
Viết tiếng Việt (150-250 từ)."""

    print(f"[debate_agent] BEAR → {symbol}")
    try:
        bear_r1 = _run_text(prompt_r1, _BEAR_SYSTEM)
    except Exception as e:
        bear_r1 = f"[BEAR R1 lỗi]: {e}"

    prompt_r2 = f"""Round 2 — Quan điểm BEAR cuối cùng cho {symbol}.

=== BEAR Round 1 ===
{bear_r1}

Tóm tắt rủi ro dài hạn lớn nhất, nhấn mạnh tại sao KHÔNG NÊN ĐẦU TƯ ngay bây giờ.
(100-150 từ)"""
    try:
        bear_r2 = _run_text(prompt_r2, _BEAR_SYSTEM)
    except Exception as e:
        bear_r2 = ""

    full = f"=== BEAR Round 1 ===\n{bear_r1}\n\n=== BEAR Final ===\n{bear_r2}" if bear_r2 else bear_r1
    return {"bear_argument": full}


def synthesize(state: dict) -> dict:
    symbol = state["symbol"]
    bull = state.get("bull_argument", "")
    bear = state.get("bear_argument", "")

    prompt = f"""Tổng hợp debate Bull vs Bear DÀI HẠN cho mã {symbol}.

=== BULL ===
{bull[:800]}

=== BEAR ===
{bear[:800]}

Trả JSON theo schema."""

    print(f"[debate_agent] Synthesizer → {symbol}")
    try:
        result = run_agent_lite(prompt=prompt, system=_SYNTHESIZER_SYSTEM)
        print(f"[debate_agent] balance={result.get('balance')}")
        return {"debate_synthesis": result}
    except Exception as e:
        print(f"[debate_agent] Synthesizer error: {e}")
        return {"debate_synthesis": {
            "bull_key_points": [], "bear_key_points": [],
            "balance": "NEUTRAL", "key_risk": "Không tổng hợp được.",
            "debate_conclusion": "Lỗi synthesizer.",
        }}


def _build_context(state: dict) -> str:
    fa = state.get("fundamental_analysis", {})
    val = state.get("valuation_analysis", {})
    macro = state.get("macro_context", {})
    symbol = state.get("symbol", "?")

    return "\n".join([
        f"Mã: {symbol}",
        "",
        "=== Cơ bản ===",
        f"ROE: {fa.get('roe', 'N/A')}% | EPS growth: {fa.get('eps_growth_yoy', 'N/A')}%",
        f"Revenue growth: {fa.get('revenue_growth', 'N/A')}%",
        f"Sức khỏe: {fa.get('financial_health', 'N/A')} | Tăng trưởng: {fa.get('growth_quality', 'N/A')}",
        f"So ngành: {fa.get('vs_industry', 'N/A')}",
        f"Tóm tắt: {fa.get('fa_summary', '')}",
        "",
        "=== Định giá ===",
        f"Intrinsic: {val.get('intrinsic_value', 'N/A')} VNĐ | P/E fair: {val.get('pe_fair', 'N/A')}",
        f"Margin of Safety: {val.get('margin_of_safety', 'N/A')}% | Valuation: {val.get('valuation', 'N/A')}",
        f"Tóm tắt: {val.get('valuation_summary', '')}",
        "",
        "=== Vĩ mô ===",
        f"Bias: {macro.get('macro_bias', 'N/A')} | Score: {macro.get('macro_score', 'N/A')}",
        f"Beneficiary sectors: {macro.get('beneficiary_sectors', [])}",
        f"Tóm tắt: {macro.get('overall_summary', '')}",
    ])


def _run_text(prompt: str, system: str) -> str:
    client = _get_client()
    response = client.messages.create(
        model=MODEL_SONNET,
        max_tokens=MAX_TOKENS_SONNET,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
