"""risk_invest.py — Risk gate cho Investment pipeline. KHÔNG dùng LLM.

Rules:
  1. Margin of safety ≥ 30% → nếu không, downgrade MUA → CHỜ
  2. Sector concentration ≤ 25% NAV (memory_service)
  3. Macro BEARISH → downgrade MUA → CHỜ
  4. financial_health = YẾU → VETO (TRÁNH)
"""

from multiagents_trading_assistant.services.memory_service import L3_VN_RULES


def check(state: dict) -> dict:
    symbol = state.get("symbol", "")
    trader = state.get("trader_decision", {})
    val = state.get("valuation_analysis", {})
    fa = state.get("fundamental_analysis", {})
    macro = state.get("macro_context", {})

    action = trader.get("action", "CHỜ")
    original = action
    warnings: list[str] = []

    print(f"[risk_invest] {symbol} — action={action}")

    # Rule 4: financial_health = YẾU → VETO
    health = fa.get("financial_health", "UNKNOWN")
    if health == "YẾU":
        return _override(original, "TRÁNH",
                         f"financial_health=YẾU — không đầu tư", warnings)

    # Rule 3: Macro BEARISH → downgrade MUA → CHỜ
    macro_bias = macro.get("macro_bias", "NEUTRAL")
    if action == "MUA" and macro_bias == "BEARISH":
        return _override(original, "CHỜ",
                         f"Macro BEARISH — hoãn đầu tư", warnings)

    # Rule 1: MoS ≥ 30%
    mos = val.get("margin_of_safety")
    if action == "MUA" and mos is not None and mos < 30:
        print(f"[risk_invest] MoS {mos:.1f}% < 30% → CHỜ")
        return _override(original, "CHỜ",
                         f"Margin of safety {mos:.1f}% < 30% — chờ giá tốt hơn", warnings)

    # Rule 2: Sector concentration (simplified — không có real portfolio data)
    # Chỉ log cảnh báo, không block
    industry = fa.get("vs_industry", "")
    if industry:
        warnings.append(f"Kiểm tra tập trung ngành {industry} ≤ 25% NAV trước khi vào")

    if warnings:
        print(f"[risk_invest] warnings: {'; '.join(warnings)}")
    print(f"[risk_invest] OK — final={action}")

    return {
        "final_action": action,
        "override_reason": None,
        "warnings": warnings,
        "sizing_modifier": 1.0,
        "original_action": original,
    }


def _override(orig: str, new_action: str, reason: str, warnings: list) -> dict:
    print(f"[risk_invest] OVERRIDE {orig} → {new_action}: {reason}")
    return {
        "final_action": new_action,
        "override_reason": reason,
        "warnings": warnings,
        "sizing_modifier": 0.0,
        "original_action": orig,
    }
