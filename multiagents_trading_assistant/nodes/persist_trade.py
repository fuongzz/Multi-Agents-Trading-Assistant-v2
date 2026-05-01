"""
persist_trade.py — Ghi nhận execution plan cho khuyến nghị trade.

Chạy sau risk_trade, trước format_output.
Không tự động ghi positions/trades khi hệ thống chỉ phát khuyến nghị.
positions/trades chỉ nên được ghi sau khi user xác nhận đã khớp lệnh thực tế.
"""

from datetime import datetime


def run_persist_trade(state: dict) -> dict:
    """
    LangGraph node — chuẩn hóa output thành recommendation-only execution plan.

    Returns dict để formatter/consumer biết đây là khuyến nghị, không phải lệnh đã khớp.
    """
    risk = state.get("risk_output") or {}
    final_action = risk.get("final_action", "")
    trader = state.get("trader_decision") or {}

    date = state.get("date") or datetime.now().strftime("%Y-%m-%d")
    symbol = state.get("symbol", "")

    execution_plan = {
        "status": "RECOMMENDATION_ONLY",
        "symbol": symbol,
        "date": date,
        "action": final_action,
        "entry_zone": trader.get("entry_zone"),
        "stop_loss": trader.get("stop_loss"),
        "initial_target": trader.get("initial_target") or trader.get("take_profit"),
        "position_pct": risk.get("adjusted_position_pct") or trader.get("position_pct") or 0,
        "sizing_modifier": risk.get("sizing_modifier", 1.0),
        "note": "Khuyến nghị chưa được ghi thành vị thế. Chỉ ghi positions/trades sau khi user xác nhận đã khớp lệnh.",
    }

    print(f"[persist_trade] {symbol} {final_action} — recommendation only, no DB position/trade write.")
    return {"execution_plan": execution_plan}
