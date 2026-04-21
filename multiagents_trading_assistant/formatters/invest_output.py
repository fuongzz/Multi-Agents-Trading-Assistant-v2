"""invest_output.py — Format Investment signal text.

KHÔNG có SL/TP kỹ thuật, confluence_score — những trường đó thuộc Trade pipeline.
"""

_ACTION_ICON = {"MUA": "🟢", "CHỜ": "🟡", "TRÁNH": "🔴"}


def format_invest_signal(state: dict) -> str:
    symbol = state.get("symbol", "?")
    date = state.get("date", "?")

    trader = state.get("trader_decision", {})
    risk = state.get("risk_output", {})
    val = state.get("valuation_analysis", {})
    fa = state.get("fundamental_analysis", {})
    debate = state.get("debate_synthesis", {})

    action = risk.get("final_action") or trader.get("action", "CHỜ")
    icon = _ACTION_ICON.get(action, "⚪")

    mos = val.get("margin_of_safety")
    intrinsic = val.get("intrinsic_value")
    target = trader.get("target_price") or intrinsic
    pos = trader.get("position_pct", 0)
    override = risk.get("override_reason")
    warnings = risk.get("warnings", [])

    lines = [
        f"📗 [INVEST] {symbol} — {icon} {action}",
        f"   Ngày: {date} | Horizon: {trader.get('holding_horizon', '3-12 tháng')}",
    ]

    if action == "MUA" and target:
        lines += [
            f"   Target:    {target:,.0f} VNĐ (intrinsic DCF)",
            f"   MoS:       {mos:+.1f}%" if mos is not None else "   MoS:       N/A",
            f"   Position:  {pos}% NAV",
            f"   Max drawdown tolerance: {trader.get('max_drawdown_tolerance', 15)}%",
        ]

    lines.append(f"   Confidence: {trader.get('confidence', '?')}")
    lines.append(f"   Lý do: {trader.get('primary_reason', '')}")

    if trader.get("exit_condition"):
        lines.append(f"   Exit: {trader['exit_condition']}")

    if fa.get("financial_health"):
        lines.append(f"   Health: {fa['financial_health']} | ROE {fa.get('roe', '?')}% | EPS growth {fa.get('eps_growth_yoy', '?')}%")

    if val.get("valuation_summary"):
        lines.append(f"   Định giá: {val['valuation_summary']}")

    if debate.get("balance"):
        lines.append(f"   Debate: {debate['balance']} — {debate.get('debate_conclusion', '')}")

    if override:
        lines.append(f"   ⚠ Risk override: {override}")
    if warnings:
        lines.append(f"   ⚠ Warnings: {' | '.join(warnings[:2])}")

    risks = trader.get("risks", [])
    if risks:
        lines.append(f"   Rủi ro: {' | '.join(risks[:2])}")

    return "\n".join(lines)
