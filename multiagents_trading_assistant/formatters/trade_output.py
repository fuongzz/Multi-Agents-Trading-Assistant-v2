"""trade_output.py — Format Trade signal text.

Bảng tiếng Việt ngắn gọn — KHÔNG có intrinsic_value/margin_of_safety
(những trường đó thuộc Investment pipeline).
"""


_ACTION_ICON = {"MUA": "🟢", "CHỜ": "🟡", "TRÁNH": "🔴"}


def format_trade_signal(state: dict) -> str:
    symbol = state.get("symbol", "?")
    date = state.get("date", "?")
    setup = state.get("setup_type", "?")

    trader = state.get("trader_decision", {})
    risk = state.get("risk_output", {})
    synth = state.get("synthesis", {})
    flow = state.get("foreign_flow_analysis", {})

    action = risk.get("final_action") or trader.get("action", "CHỜ")
    icon = _ACTION_ICON.get(action, "⚪")

    confluence = synth.get("confluence_score", "?")
    quality = synth.get("setup_quality", "?")
    drivers = synth.get("drivers", [])
    blockers = synth.get("blockers", [])
    override = risk.get("override_reason")
    warnings = risk.get("warnings", [])
    sizing = risk.get("sizing_modifier", 1.0)

    lines = [
        f"📙 [TRADE] {symbol} — {icon} {action}  ({setup}, confluence {confluence}/100, {quality})",
        f"   Ngày: {date}",
    ]

    if action == "MUA" and trader.get("entry_zone"):
        ez = trader["entry_zone"]
        sl = trader.get("stop_loss")
        tp = trader.get("take_profit")
        rr = trader.get("rr_ratio", "?")
        pos = int((trader.get("position_pct", 0) or 0) * sizing)
        entry_mid = (ez[0] + ez[1]) / 2

        sl_pct = (sl - entry_mid) / entry_mid * 100 if sl else 0
        tp_pct = (tp - entry_mid) / entry_mid * 100 if tp else 0

        lines += [
            f"   Entry: {ez[0]:,.0f}–{ez[1]:,.0f}",
            f"   SL:    {sl:,.0f} ({sl_pct:+.1f}%)" if sl else "   SL:    N/A",
            f"   TP:    {tp:,.0f} ({tp_pct:+.1f}%)" if tp else "   TP:    N/A",
            f"   R:R:   {rr} | Position: {pos}% NAV | Horizon: {trader.get('holding_horizon', '1-4 tuần')}",
        ]

    lines.append(f"   Confidence: {trader.get('confidence', '?')}")
    lines.append(f"   Lý do: {trader.get('primary_reason', '')}")

    if drivers:
        lines.append(f"   Drivers (+): {' | '.join(drivers[:3])}")
    if blockers:
        lines.append(f"   Blockers (-): {' | '.join(blockers[:3])}")

    if flow.get("flow_trend"):
        lines.append(f"   NN: {flow.get('flow_trend')} | Room: {flow.get('room_status', '?')}")

    if override:
        lines.append(f"   ⚠ Risk override: {override}")
    if warnings:
        lines.append(f"   ⚠ Warnings: {' | '.join(warnings[:2])}")

    risks = trader.get("risks", [])
    if risks:
        lines.append(f"   Rủi ro: {' | '.join(risks[:2])}")

    return "\n".join(lines)
