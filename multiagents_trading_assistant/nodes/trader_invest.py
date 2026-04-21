"""trader_invest.py — Quyết định MUA/CHỜ/TRÁNH cho Investment pipeline (dài hạn).

Model: Sonnet. Output KHÔNG có SL/TP kỹ thuật.
Thay vào đó: target_price, exit_condition, max_drawdown_tolerance.
"""

import importlib.metadata  # noqa: F401

from multiagents_trading_assistant.services.llm_service import run_agent


_SYSTEM_PROMPT = """Bạn là Portfolio Manager đầu tư giá trị dài hạn (Value Investor).
Nhiệm vụ: dựa trên FA + Valuation + Macro + Debate, ra quyết định đầu tư 3-12 tháng.

=== Quy tắc bắt buộc ===
1. action chỉ nhận: "MUA" | "CHỜ" | "TRÁNH"
2. target_price = intrinsic_value từ valuation_agent (KHÔNG tự tính lại)
3. holding_horizon cố định: "3-12 tháng"
4. position_pct: 3-5% NAV (cao MoS + debate STRONG_BULL → 5%, thấp → 3%)
5. exit_condition: story thay đổi (ROE sụt 3Q, EPS giảm) HOẶC giá ≥ intrinsic × 1.1
6. max_drawdown_tolerance: 15-20% (KHÔNG đặt SL kỹ thuật)
7. Valuation ĐẮTS + debate STRONG_BEAR → TRÁNH
8. MoS < 0 (giá > intrinsic) → CHỜ, không MUA
9. financial_health = YẾU → TRÁNH

=== Output schema (JSON hợp lệ DUY NHẤT) ===
{
  "action": "MUA" | "CHỜ" | "TRÁNH",
  "target_price": <float|null>,
  "position_pct": <int 0-5>,
  "holding_horizon": "3-12 tháng",
  "exit_condition": <str — 1-2 câu>,
  "max_drawdown_tolerance": <int 15-20>,
  "confidence": "THẤP" | "TRUNG_BÌNH" | "CAO",
  "primary_reason": <str — 1-2 câu tiếng Việt>,
  "risks": [<str>, ...]
}"""


def decide(state: dict) -> dict:
    symbol = state.get("symbol", "?")
    print(f"[trader_invest] Sonnet — {symbol}")

    prompt = _build_prompt(state)
    try:
        result = run_agent(prompt=prompt, system=_SYSTEM_PROMPT)
        result = _validate(result, state)
        print(f"[trader_invest] {symbol} → {result.get('action')} | conf={result.get('confidence')} | pos={result.get('position_pct')}%")
        return {"trader_decision": result}
    except Exception as e:
        print(f"[trader_invest] LLM error: {e}")
        return {"trader_decision": _fallback(str(e))}


def _build_prompt(state: dict) -> str:
    symbol = state.get("symbol", "?")
    date = state.get("date", "?")
    fa = state.get("fundamental_analysis", {})
    val = state.get("valuation_analysis", {})
    macro = state.get("macro_context", {})
    debate = state.get("debate_synthesis", {})

    return "\n".join([
        f"Quyết định đầu tư dài hạn cho {symbol} ngày {date}.",
        "",
        "=== Cơ bản ===",
        f"ROE: {fa.get('roe', 'N/A')}% | EPS growth: {fa.get('eps_growth_yoy', 'N/A')}%",
        f"Revenue growth: {fa.get('revenue_growth', 'N/A')}%",
        f"Sức khỏe: {fa.get('financial_health', 'N/A')} | Tăng trưởng: {fa.get('growth_quality', 'N/A')}",
        f"Tóm tắt: {fa.get('fa_summary', '')}",
        "",
        "=== Định giá ===",
        f"Intrinsic (DCF): {val.get('intrinsic_value', 'N/A')} VNĐ",
        f"P/E fair: {val.get('pe_fair', 'N/A')} | P/B fair: {val.get('pb_fair', 'N/A')}",
        f"Margin of Safety: {val.get('margin_of_safety', 'N/A')}%",
        f"Valuation: {val.get('valuation', 'N/A')}",
        f"Tóm tắt: {val.get('valuation_summary', '')}",
        "",
        "=== Vĩ mô ===",
        f"Bias: {macro.get('macro_bias', 'N/A')} | Score: {macro.get('macro_score', 'N/A')}",
        f"Key risk: {macro.get('key_risk', '')}",
        f"Tóm tắt: {macro.get('overall_summary', '')}",
        "",
        "=== Debate Bull vs Bear (dài hạn) ===",
        f"Balance: {debate.get('balance', 'N/A')}",
        f"Bull points: {debate.get('bull_key_points', [])}",
        f"Bear points: {debate.get('bear_key_points', [])}",
        f"Key risk: {debate.get('key_risk', '')}",
        f"Kết luận: {debate.get('debate_conclusion', '')}",
        "",
        "Quyết định theo khung 3-12 tháng. Trả JSON theo schema.",
    ])


def _validate(result: dict, state: dict) -> dict:
    action = str(result.get("action", "CHỜ")).upper().strip()
    if action not in ("MUA", "CHỜ", "TRÁNH"):
        action = "CHỜ"
    result["action"] = action
    result["holding_horizon"] = "3-12 tháng"

    # Clamp position_pct
    conf = result.get("confidence", "THẤP")
    max_pos = {"THẤP": 3, "TRUNG_BÌNH": 4, "CAO": 5}.get(conf, 3)
    result["position_pct"] = min(int(result.get("position_pct") or 0), max_pos)

    # Sanity: không MUA nếu MoS < 0
    val = state.get("valuation_analysis", {})
    mos = val.get("margin_of_safety")
    if action == "MUA" and mos is not None and mos < 0:
        print(f"[trader_invest] ⚠ MoS {mos}% < 0 → override CHỜ")
        result["action"] = "CHỜ"
        result["position_pct"] = 0
        result["primary_reason"] = f"Giá cao hơn intrinsic (MoS={mos}%) — chờ điều chỉnh."

    if result["action"] in ("CHỜ", "TRÁNH"):
        result["position_pct"] = 0

    result.setdefault("target_price", val.get("intrinsic_value"))
    result.setdefault("exit_condition", "Story thay đổi hoặc giá ≥ intrinsic × 1.1")
    result.setdefault("max_drawdown_tolerance", 15)
    result.setdefault("risks", [])
    return result


def _fallback(err: str) -> dict:
    return {
        "action": "CHỜ", "target_price": None, "position_pct": 0,
        "holding_horizon": "3-12 tháng", "exit_condition": "N/A",
        "max_drawdown_tolerance": 15, "confidence": "THẤP",
        "primary_reason": f"LLM lỗi — CHỜ. {err}".strip(),
        "risks": ["trader_invest fail"],
    }
