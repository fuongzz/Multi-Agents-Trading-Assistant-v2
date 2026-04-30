"""
persist_trade.py — Tự động ghi nhận vị thế khi AI quyết định MUA.

Chạy sau risk_trade, trước format_output.
Chỉ ghi khi final_action == 'MUA' và chưa có vị thế cho symbol này.
"""

import math
from datetime import datetime

from multiagents_trading_assistant import database as db
from multiagents_trading_assistant.services.portfolio_service import get_total_nav_vnd


def run_persist_trade(state: dict) -> dict:
    """
    LangGraph node — ghi nhận lệnh MUA vào DB (positions + trades).

    Returns {} trong mọi trường hợp (không thay đổi state).
    """
    risk = state.get("risk_output") or {}
    final_action = risk.get("final_action", "")

    if final_action != "MUA":
        return {}

    symbol = state.get("symbol", "")
    date = state.get("date") or datetime.now().strftime("%Y-%m-%d")

    # Guard: tránh duplicate khi pipeline rerun cùng ngày
    if db.has_position(symbol):
        print(f"[persist_trade] {symbol} đã có vị thế — skip.")
        return {}

    trader = state.get("trader_decision") or {}
    market = state.get("market_context") or {}

    entry_zone = trader.get("entry_zone")
    stop_loss = trader.get("stop_loss")
    take_profit = trader.get("take_profit")
    position_pct = trader.get("position_pct") or 2
    sizing_modifier = risk.get("sizing_modifier") or 1.0
    setup_type = state.get("setup_type") or trader.get("setup_type") or "UNKNOWN"
    exchange = market.get("exchange") or "HOSE"
    confidence = trader.get("confidence") or ""

    # Tính entry_price = midpoint của entry_zone
    if entry_zone and len(entry_zone) == 2:
        entry_price = (entry_zone[0] + entry_zone[1]) / 2
    elif entry_zone and len(entry_zone) == 1:
        entry_price = entry_zone[0]
    else:
        # Fallback: dùng current_price từ market context
        entry_price = market.get("current_price", 0)

    if entry_price <= 0:
        print(f"[persist_trade] {symbol} entry_price không hợp lệ ({entry_price}) — skip.")
        return {}

    # Tính quantity
    total_nav = get_total_nav_vnd()
    effective_nav_pct = position_pct * sizing_modifier
    raw_quantity = (effective_nav_pct / 100.0 * total_nav) / entry_price
    # Làm tròn xuống bội số 100 (lot size VN)
    quantity = max(100, math.floor(raw_quantity / 100) * 100)

    effective_nav_pct_actual = round(effective_nav_pct, 2)

    try:
        db.add_position(
            symbol=symbol,
            exchange=exchange,
            entry_price=round(entry_price),
            quantity=quantity,
            entry_date=date,
            strategy=setup_type,
            sl=round(stop_loss) if stop_loss else None,
            tp=round(take_profit) if take_profit else None,
            nav_pct=effective_nav_pct_actual,
        )
        db.record_trade(
            symbol=symbol,
            action="MUA",
            price=round(entry_price),
            quantity=quantity,
            trade_date=date,
            strategy=setup_type,
            note=f"Auto | conf={confidence} | nav={effective_nav_pct_actual:.1f}%",
        )
        print(
            f"[persist_trade] ✅ {symbol} MUA @ {entry_price:,.0f} | "
            f"qty={quantity:,} | nav={effective_nav_pct_actual:.1f}% | "
            f"SL={stop_loss} TP={take_profit}"
        )
    except Exception as e:
        print(f"[persist_trade] Lỗi ghi DB cho {symbol}: {e}")

    return {}
