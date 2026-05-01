"""trade_output.py — Format Trade signal text.

Bảng tiếng Việt ngắn gọn — KHÔNG có intrinsic_value/margin_of_safety
(những trường đó thuộc Investment pipeline).
"""

import unicodedata

_ACTION_ICON = {
    "MUA": "🟢",
    "CHỜ": "🟡",
    "TRÁNH": "🔴",
    "GIỮ": "🔵",
    "GIA_TĂNG": "🟢",
    "GIẢM": "🟠",
    "BÁN": "🔴",
}

def _action_key(action: str) -> str:
    text = str(action or "").upper().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("Đ", "D")


def format_trade_signal(state: dict) -> str:
    symbol = state.get("symbol", "?")
    date = state.get("date", "?")
    setup = state.get("setup_type", "?")

    trader = state.get("trader_decision", {})
    risk = state.get("risk_output", {})
    synth = state.get("synthesis", {})
    flow = state.get("foreign_flow_analysis", {})
    exec_plan = state.get("execution_plan", {})
    memory = (state.get("memory_context") or {}).get("internal", {})
    current_position = memory.get("current_position") or {}

    action = risk.get("final_action") or trader.get("action", "CHỜ")
    icon = _ACTION_ICON.get(action, "⚪")
    action_key = _action_key(action)

    confluence = synth.get("confluence_score", "?")
    quality = synth.get("setup_quality", "?")
    drivers = synth.get("drivers", [])
    blockers = synth.get("blockers", [])
    override = risk.get("override_reason")
    warnings = risk.get("warnings", [])
    sizing = risk.get("sizing_modifier", 1.0)

    lines = [
        f"{'='*55}",
        f"[TRADE] {symbol} | {icon} {action}",
        f"  Setup: {setup} | Confluence: {confluence}/100 | Quality: {quality}",
        f"  Ngay:  {date} | Confidence: {trader.get('confidence', '?')}",
        f"{'='*55}",
    ]

    # Entry / SL / TP — chỉ khi có khuyến nghị mở mới/gia tăng
    if action_key in ("MUA", "GIA_TANG") and trader.get("entry_zone"):
        ez  = trader["entry_zone"]
        sl  = trader.get("stop_loss")
        target = trader.get("initial_target") or trader.get("take_profit")
        rr  = trader.get("rr_ratio", "?")
        pos = risk.get("adjusted_position_pct")
        if pos is None:
            pos = int((trader.get("position_pct", 0) or 0) * sizing)
        mid = (ez[0] + ez[1]) / 2
        sl_pct = (sl - mid) / mid * 100 if sl else 0
        target_pct = (target - mid) / mid * 100 if target else 0
        lines += [
            f"  Entry:  {ez[0]:,.0f} – {ez[1]:,.0f}",
            f"  SL:     {sl:,.0f} ({sl_pct:+.1f}%)" if sl else "  SL:     N/A",
            f"  Target: {target:,.0f} ({target_pct:+.1f}%)" if target else "  Target: N/A",
            f"  R:R:    {rr} | Size: {pos}% NAV | Hold: {trader.get('holding_horizon','1-4 tuan')}",
            f"{'-'*55}",
        ]

    # Position management — khi đã có vị thế
    if action_key in ("GIU", "GIAM", "BAN") and current_position:
        lines += [
            f"  Position: entry {current_position.get('entry_price', 'N/A')} | "
            f"SL {current_position.get('sl', 'N/A')} | Target {current_position.get('tp', 'N/A')} | "
            f"{current_position.get('nav_pct', 'N/A')}% NAV",
        ]
        if trader.get("management_plan"):
            lines.append(f"  Plan:   {trader.get('management_plan')}")
        if trader.get("reversal_watchlist"):
            lines.append(f"  Watch:  {' | '.join(trader.get('reversal_watchlist', [])[:3])}")
        lines.append(f"{'-'*55}")

    # Lý do — luôn hiển thị dù MUA hay CHO
    reason = trader.get("primary_reason", "")
    if reason:
        lines.append(f"  Ly do:  {reason}")

    # Drivers / Blockers
    if drivers:
        lines.append(f"  (+):    {' | '.join(drivers[:3])}")
    if blockers:
        lines.append(f"  (-):    {' | '.join(blockers[:3])}")

    # Tại sao CHO/TRANH — note riêng
    if action_key in ("CHO", "TRANH", "GIU", "GIAM", "BAN"):
        note = trader.get("trader_note", "")
        if note:
            lines.append(f"  Note:   {note}")

    # Dòng ngoại
    if flow.get("flow_trend"):
        lines.append(f"  NN:     {flow.get('flow_trend')} | Room: {flow.get('room_status','?')}")

    # Rủi ro
    risks = trader.get("risks", [])
    if risks:
        lines.append(f"  Risk:   {' | '.join(risks[:2])}")

    # Override / warnings
    if override:
        lines.append(f"  [!] Risk override: {override}")
    if warnings:
        lines.append(f"  [!] Warnings: {' | '.join(warnings[:2])}")

    if exec_plan.get("status") == "RECOMMENDATION_ONLY":
        lines.append("  Mode:   Khuyen nghi, chua ghi thanh vi the/giao dich.")

    lines.append(f"{'='*55}")
    return "\n".join(lines)
