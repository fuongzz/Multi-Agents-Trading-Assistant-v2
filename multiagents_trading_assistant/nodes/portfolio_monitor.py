"""
portfolio_monitor.py — Node kiểm tra danh mục mỗi sáng.

Chạy một lần duy nhất mỗi ngày (symbol đầu tiên trong batch).
Guard: market_context["_portfolio_checked"] == True → skip.

Luồng:
  1. Lấy toàn bộ vị thế đang giữ
  2. Lấy giá live batch (1 API call)
  3. Cập nhật peak_price
  4. Phát hiện exit signals (SL/TP/momentum)
  5. Đóng vị thế + gửi Discord alert
  6. Gửi portfolio summary
"""

from datetime import datetime

from multiagents_trading_assistant import database as db
from multiagents_trading_assistant.services.portfolio_service import (
    check_exit_signals,
    compute_performance_stats,
    get_enriched_positions,
)


def run_portfolio_monitor(state: dict) -> dict:
    """
    LangGraph node — kiểm tra danh mục và phát tín hiệu thoát.

    Returns dict với key portfolio_summary (hoặc {} nếu skip/không có vị thế).
    """
    # Guard: chỉ chạy với symbol đầu tiên trong batch mỗi ngày
    if state.get("market_context", {}).get("_portfolio_checked", False):
        return {}

    positions = db.get_all_positions()
    if not positions:
        print("[portfolio_monitor] Không có vị thế đang giữ — skip.")
        return {"portfolio_summary": {"open_positions_count": 0}}

    date = state.get("date") or datetime.now().strftime("%Y-%m-%d")
    symbols = [p["symbol"] for p in positions]

    # Lấy giá live batch
    live_prices: dict[str, float] = {}
    try:
        from multiagents_trading_assistant.services.data_service import get_price_board
        df = get_price_board(symbols)
        if df is not None and not df.empty and "symbol" in df.columns and "price" in df.columns:
            live_prices = dict(zip(df["symbol"].astype(str), df["price"].astype(float)))
    except Exception as e:
        print(f"[portfolio_monitor] Lấy giá live lỗi: {e} — dùng entry price thay thế.")

    # Cập nhật peak_price
    for pos in positions:
        sym = pos["symbol"]
        current = live_prices.get(sym, pos.get("entry_price", 0))
        old_peak = pos.get("peak_price") or 0
        if current > old_peak:
            db.update_position_peak(sym, current)
            pos["peak_price"] = current  # cập nhật local để check_exit_signals dùng

    enriched = get_enriched_positions(live_prices)
    exit_signals = check_exit_signals(enriched)

    # Xử lý tín hiệu thoát
    exited_symbols = []
    for sig in exit_signals:
        sym = sig["symbol"]
        try:
            db.close_position(
                symbol=sym,
                exit_price=sig["current_price"],
                exit_date=date,
                exit_reason=sig["exit_type"],
                entry_price=sig["entry_price"],
                quantity=sig.get("quantity", 0),
                strategy=sig.get("strategy"),
            )
            exited_symbols.append(sym)
            print(f"[portfolio_monitor] {sym} đã thoát: {sig['exit_type']} @ {sig['current_price']:,.0f}")

            # Gửi exit alert lên Discord
            _send_exit_alert(sig)

        except Exception as e:
            print(f"[portfolio_monitor] Lỗi khi đóng {sym}: {e}")

    # Loại bỏ vị thế đã thoát khỏi danh sách enriched để tổng hợp đúng
    remaining = [p for p in enriched if p["symbol"] not in exited_symbols]

    # Tính stats và gửi portfolio summary
    stats = compute_performance_stats(live_prices)
    _send_portfolio_summary(remaining, stats, date)

    return {
        "portfolio_summary": {
            "date": date,
            "open_positions_count": len(remaining),
            "exit_signals_count": len(exit_signals),
            "exited_symbols": exited_symbols,
            "stats": stats,
        }
    }


def _send_exit_alert(signal: dict) -> None:
    """Gọi output_service.send_exit_alert — lazy import để tránh circular."""
    try:
        from multiagents_trading_assistant.services.output_service import send_exit_alert
        send_exit_alert(signal)
    except Exception as e:
        print(f"[portfolio_monitor] send_exit_alert lỗi: {e}")


def _send_portfolio_summary(positions: list[dict], stats: dict, date: str) -> None:
    """Gọi output_service.send_portfolio_summary — lazy import."""
    try:
        from multiagents_trading_assistant.services.output_service import send_portfolio_summary
        send_portfolio_summary(positions, stats, date)
    except Exception as e:
        print(f"[portfolio_monitor] send_portfolio_summary lỗi: {e}")
