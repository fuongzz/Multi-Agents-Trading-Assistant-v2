"""invest_output.py — Format Investment signal text."""

_ACTION_ICON = {"MUA": "🟢", "CHO": "🟡", "TRANH": "🔴"}


def format_invest_signal(state: dict) -> str:
    symbol = state.get("symbol", "?")
    date   = state.get("date", "?")

    trader = state.get("trader_decision", {})
    risk   = state.get("risk_output", {})
    val    = state.get("valuation_analysis", {})
    fa     = state.get("fundamental_analysis", {})
    debate = state.get("debate_synthesis", {})

    action   = risk.get("final_action") or trader.get("action", "CHO")
    icon     = _ACTION_ICON.get(action, "o")
    mos      = val.get("margin_of_safety")
    intrinsic = val.get("intrinsic_value")
    target   = trader.get("target_price") or intrinsic
    pos      = trader.get("position_pct", 0)
    override = risk.get("override_reason")
    warnings = risk.get("warnings", [])

    lines = [
        f"{'='*55}",
        f"[INVEST] {symbol} | {icon} {action}",
        f"  Ngay: {date} | Horizon: {trader.get('holding_horizon','3-12 thang')} | Confidence: {trader.get('confidence','?')}",
        f"{'='*55}",
    ]

    # Target / MoS / sizing -- luon hien thi neu co
    if target:
        lines.append(f"  Target:  {target:,.0f} VND")
    if mos is not None:
        lines.append(f"  MoS:     {mos:+.1f}%")
    if pos:
        lines.append(f"  Size:    {pos}% NAV | Max DD: {trader.get('max_drawdown_tolerance',15)}%")

    lines.append(f"{'-'*55}")

    # Ly do -- luon hien thi du MUA hay CHO
    reason = trader.get("primary_reason", "")
    if reason:
        lines.append(f"  Ly do:   {reason}")

    # Exit condition
    if trader.get("exit_condition"):
        lines.append(f"  Exit:    {trader['exit_condition']}")

    # Fundamental snapshot
    if fa.get("financial_health"):
        lines.append(
            f"  FA:      {fa['financial_health']} | "
            f"ROE {fa.get('roe','?')}% | EPS growth {fa.get('eps_growth_yoy','?')}%"
        )

    # Valuation
    if val.get("valuation_summary"):
        lines.append(f"  Val:     {val['valuation_summary']}")

    # Debate balance
    if debate.get("balance"):
        lines.append(f"  Debate:  {debate['balance']} -- {debate.get('debate_conclusion','')}")

    # Note khi CHO/TRANH
    if action in ("CHO", "TRANH"):
        note = trader.get("trader_note", "")
        if note:
            lines.append(f"  Note:    {note}")

    # Rui ro
    risks = trader.get("risks", [])
    if risks:
        lines.append(f"  Risk:    {' | '.join(risks[:2])}")

    if override:
        lines.append(f"  [!] Override: {override}")
    if warnings:
        lines.append(f"  [!] Warnings: {' | '.join(warnings[:2])}")

    lines.append(f"{'='*55}")
    return "\n".join(lines)
