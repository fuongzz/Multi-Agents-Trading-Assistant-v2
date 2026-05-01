"""
portfolio_service.py — Quản trị danh mục đầu tư.

Ba nhóm chức năng:
  1. Định giá vị thế (unrealized P&L, khoảng cách SL/TP)
  2. Phát hiện tín hiệu thoát (SL_HIT / TP_HIT / MOMENTUM_LOSS)
  3. Thống kê hiệu suất (win rate, R:R, equity curve)
"""

import os
from datetime import datetime, timedelta

from multiagents_trading_assistant import database as db


# ──────────────────────────────────────────────
# NAV helpers
# ──────────────────────────────────────────────

def get_total_nav_vnd() -> float:
    """Đọc tổng NAV (VNĐ) từ env → DB → mặc định 1 tỷ."""
    env_val = os.environ.get("TOTAL_NAV_VND")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    db_val = db.get_portfolio_config("total_nav_vnd")
    if db_val:
        try:
            return float(db_val)
        except ValueError:
            pass
    return 1_000_000_000.0


def _get_momentum_drawdown_pct() -> float:
    """Đọc ngưỡng momentum drawdown từ env → DB → mặc định 3%."""
    env_val = os.environ.get("MOMENTUM_DRAWDOWN_PCT")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    db_val = db.get_portfolio_config("momentum_drawdown_pct")
    if db_val:
        try:
            return float(db_val)
        except ValueError:
            pass
    return 3.0


def compute_profit_trailing_sl(
    *,
    entry_price: float,
    current_price: float,
    old_sl: float | None,
    initial_target: float | None = None,
    peak_price: float | None = None,
) -> float | None:
    """Return a higher SL for winners; initial_target is not a hard TP."""
    if entry_price <= 0 or current_price <= entry_price:
        return None

    reference_price = max(current_price, peak_price or current_price)
    gain_pct = (reference_price - entry_price) / entry_price * 100
    target_hit = bool(initial_target and reference_price >= initial_target)
    if gain_pct < 5.0 and not target_hit:
        return None

    # Wider leash as profit grows: protect gains without capping upside.
    if gain_pct >= 20.0:
        lock_ratio = 0.55
    elif gain_pct >= 10.0:
        lock_ratio = 0.60
    else:
        lock_ratio = 0.50

    suggested = round(entry_price + (reference_price - entry_price) * lock_ratio, 0)
    old = old_sl or 0.0
    if suggested <= old:
        return None
    return suggested


def review_target_decision(position: dict) -> dict:
    """Reason about what to do when price reaches the reference target."""
    entry = float(position.get("entry_price") or 0)
    current = float(position.get("current_price") or 0)
    target = position.get("tp")
    peak = float(position.get("peak_price") or current or 0)
    old_sl = position.get("sl")
    new_sl = position.get("suggested_trailing_sl")
    t3_ok = bool(position.get("t3_available", False))

    if not target or entry <= 0 or current < target:
        return {
            "recommendation": "NO_TARGET_REVIEW",
            "reason": "Chưa chạm target tham chiếu.",
            "should_alert": False,
        }

    pnl_pct = (current - entry) / entry * 100
    target_extension_pct = (current - target) / target * 100 if target else 0.0
    drawdown_pct = (peak - current) / peak * 100 if peak > 0 else 0.0
    protected_pct = None
    if new_sl and new_sl > entry:
        protected_pct = (new_sl - entry) / entry * 100
    elif old_sl and old_sl > entry:
        protected_pct = (old_sl - entry) / entry * 100

    reasons: list[str] = []
    if not t3_ok:
        reasons.append("Chưa qua T+3 nên không thể chủ động bán.")
        recommendation = "HOLD_RAISE_SL"
    elif drawdown_pct >= 4.0:
        reasons.append(f"Giá đã hụt {drawdown_pct:.1f}% từ đỉnh sau khi đạt target.")
        recommendation = "TAKE_PROFIT"
    elif drawdown_pct >= 2.5:
        reasons.append(f"Giá đang trả lại lợi nhuận {drawdown_pct:.1f}% từ đỉnh.")
        recommendation = "TAKE_PARTIAL"
    elif pnl_pct >= 15.0 and target_extension_pct < 1.0:
        reasons.append(f"Lợi nhuận {pnl_pct:.1f}% nhưng giá chưa vượt target rõ.")
        recommendation = "TAKE_PARTIAL"
    elif protected_pct is not None and protected_pct >= pnl_pct * 0.45:
        reasons.append(f"SL mới khóa khoảng {protected_pct:.1f}% lợi nhuận.")
        recommendation = "HOLD_RAISE_SL"
    elif new_sl:
        reasons.append("Có thể nâng SL để khóa lãi và tiếp tục cho lệnh chạy.")
        recommendation = "HOLD_RAISE_SL"
    else:
        reasons.append("Đạt target nhưng chưa nâng được SL bảo vệ lợi nhuận.")
        recommendation = "TAKE_PARTIAL"

    if target_extension_pct >= 2.0 and drawdown_pct < 2.0:
        reasons.append(f"Giá vượt target {target_extension_pct:.1f}% và chưa bị xả mạnh.")
        if recommendation == "TAKE_PARTIAL":
            recommendation = "HOLD_RAISE_SL"

    action_text = {
        "HOLD_RAISE_SL": "Tiếp tục nắm giữ, nâng SL để khóa lãi.",
        "TAKE_PARTIAL": "Cân nhắc chốt 1 phần, phần còn lại giữ theo trailing SL.",
        "TAKE_PROFIT": "Cân nhắc chốt lời chủ động vì giá đang suy yếu sau target.",
    }.get(recommendation, "Theo dõi thêm.")

    return {
        "recommendation": recommendation,
        "action_text": action_text,
        "reason": " ".join(reasons),
        "pnl_pct": round(pnl_pct, 2),
        "target_extension_pct": round(target_extension_pct, 2),
        "drawdown_from_peak_pct": round(drawdown_pct, 2),
        "protected_profit_pct": round(protected_pct, 2) if protected_pct is not None else None,
        "should_alert": True,
    }


def compute_nav_pct_actual(position: dict, total_nav: float) -> float:
    """Tính % NAV thực tế = (entry_price * quantity) / total_nav * 100."""
    if total_nav <= 0:
        return 0.0
    return round(position.get("entry_price", 0) * position.get("quantity", 0) / total_nav * 100, 2)


# ──────────────────────────────────────────────
# Định giá vị thế
# ──────────────────────────────────────────────

def get_enriched_positions(live_prices: dict[str, float]) -> list[dict]:
    """
    Với mỗi vị thế trong DB, tính các chỉ số thị trường thực tế.

    Args:
        live_prices: {symbol: current_price} — lấy từ get_price_board()

    Returns:
        list[dict] — mỗi dict là vị thế gốc + các trường bổ sung:
          current_price, unrealized_pnl_vnd, unrealized_pnl_pct,
          nav_pct_actual, distance_to_sl_pct, distance_to_tp_pct,
          days_held, t3_available
    """
    positions = db.get_all_positions()
    total_nav = get_total_nav_vnd()
    today = datetime.now().date()
    enriched = []

    for pos in positions:
        symbol = pos["symbol"]
        current_price = live_prices.get(symbol)
        if current_price is None or current_price <= 0:
            current_price = pos.get("entry_price", 0)

        entry_price = pos.get("entry_price", 0)
        quantity = pos.get("quantity", 0)
        sl = pos.get("sl")
        tp = pos.get("tp")
        peak = pos.get("peak_price") or current_price
        target_reached = bool(tp and current_price >= tp)
        suggested_trailing_sl = compute_profit_trailing_sl(
            entry_price=entry_price,
            current_price=current_price,
            old_sl=sl,
            initial_target=tp,
            peak_price=peak,
        )

        # P&L
        unrealized_pnl_vnd = (current_price - entry_price) * quantity
        unrealized_pnl_pct = (
            round((current_price - entry_price) / entry_price * 100, 2)
            if entry_price > 0 else 0.0
        )

        # Khoảng cách SL/TP
        distance_to_sl_pct = (
            round((current_price - sl) / current_price * 100, 2)
            if sl and current_price > 0 else None
        )
        distance_to_tp_pct = (
            round((tp - current_price) / current_price * 100, 2)
            if tp and current_price > 0 else None
        )

        # T+2.5: hàng về chiều T+2 (2 ngày lịch sau ngày mua)
        entry_date_str = pos.get("entry_date", "")
        try:
            entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
            days_held = (today - entry_date).days
            t3_available = days_held >= 2
        except (ValueError, TypeError):
            days_held = 0
            t3_available = False

        enriched_pos = {
            **pos,
            "current_price": current_price,
            "unrealized_pnl_vnd": round(unrealized_pnl_vnd),
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "nav_pct_actual": compute_nav_pct_actual(pos, total_nav),
            "distance_to_sl_pct": distance_to_sl_pct,
            "distance_to_tp_pct": distance_to_tp_pct,
            "target_reached": target_reached,
            "suggested_trailing_sl": suggested_trailing_sl,
            "days_held": days_held,
            "t3_available": t3_available,
        }
        enriched_pos["target_review"] = review_target_decision(enriched_pos)
        enriched.append(enriched_pos)

    return enriched


# ──────────────────────────────────────────────
# Phát hiện tín hiệu thoát
# ──────────────────────────────────────────────

def check_exit_signals(enriched_positions: list[dict]) -> list[dict]:
    """
    Kiểm tra các điều kiện thoát lệnh cho mỗi vị thế đã định giá.

    Điều kiện:
      - SL_HIT       : current_price <= sl
      - MOMENTUM_LOSS: current đã giảm >= momentum_drawdown_pct% từ peak
                       VÀ t3_available=True (không thoát T+0/T+1, chờ T+2.5)

    Returns:
        list[dict] — mỗi dict là vị thế + exit_type
    """
    momentum_threshold = _get_momentum_drawdown_pct()
    signals = []

    for pos in enriched_positions:
        symbol = pos["symbol"]
        current = pos["current_price"]
        sl = pos.get("sl")
        peak = pos.get("peak_price") or current
        t3_ok = pos.get("t3_available", False)

        exit_type = None

        if sl and current <= sl:
            exit_type = "SL_HIT"
        elif t3_ok and peak > 0:
            drawdown_pct = (peak - current) / peak * 100
            if drawdown_pct >= momentum_threshold:
                exit_type = "MOMENTUM_LOSS"

        if exit_type:
            signals.append({
                **pos,
                "exit_type": exit_type,
                "peak_price": peak,
                "drawdown_from_peak_pct": round((peak - current) / peak * 100, 2) if peak > 0 else 0,
            })

    return signals


# ──────────────────────────────────────────────
# Thống kê hiệu suất
# ──────────────────────────────────────────────

def compute_performance_stats(live_prices: dict[str, float] | None = None) -> dict:
    """
    Tính thống kê hiệu suất từ lịch sử giao dịch đã đóng.

    Returns:
        dict với các keys:
          total_trades, win_trades, lose_trades, win_rate,
          avg_realized_rr, total_realized_pnl_vnd,
          pnl_by_setup, equity_curve,
          open_positions_count, open_unrealized_pnl_vnd
    """
    closed = db.get_closed_trades(limit=500)

    total = len(closed)
    wins = sum(1 for t in closed if (t.get("realized_pnl") or 0) > 0)
    loses = total - wins
    win_rate = round(wins / total * 100, 1) if total > 0 else 0.0

    rr_values = [t["realized_rr"] for t in closed if t.get("realized_rr") is not None]
    avg_rr = round(sum(rr_values) / len(rr_values), 2) if rr_values else 0.0

    total_pnl = sum(t.get("realized_pnl") or 0 for t in closed)

    # P&L theo setup type
    pnl_by_setup: dict[str, dict] = {}
    for t in closed:
        setup = t.get("strategy") or "UNKNOWN"
        if setup not in pnl_by_setup:
            pnl_by_setup[setup] = {"trades": 0, "wins": 0, "rr_sum": 0.0, "total_pnl": 0.0}
        pnl_by_setup[setup]["trades"] += 1
        if (t.get("realized_pnl") or 0) > 0:
            pnl_by_setup[setup]["wins"] += 1
        pnl_by_setup[setup]["rr_sum"] += t.get("realized_rr") or 0
        pnl_by_setup[setup]["total_pnl"] += t.get("realized_pnl") or 0

    for setup in pnl_by_setup:
        d = pnl_by_setup[setup]
        d["avg_rr"] = round(d["rr_sum"] / d["trades"], 2) if d["trades"] > 0 else 0.0
        d["win_rate"] = round(d["wins"] / d["trades"] * 100, 1) if d["trades"] > 0 else 0.0
        del d["rr_sum"]

    # Equity curve — tích lũy P&L theo ngày
    sorted_closed = sorted(closed, key=lambda t: t.get("trade_date") or "")
    equity_curve = []
    cumulative = 0.0
    for t in sorted_closed:
        cumulative += t.get("realized_pnl") or 0
        equity_curve.append({
            "date": t.get("trade_date"),
            "cumulative_pnl": round(cumulative),
        })

    # Vị thế đang mở
    open_positions = db.get_all_positions()
    open_unrealized = 0.0
    if live_prices and open_positions:
        for pos in open_positions:
            sym = pos["symbol"]
            curr = live_prices.get(sym, pos.get("entry_price", 0))
            open_unrealized += (curr - pos.get("entry_price", 0)) * pos.get("quantity", 0)

    return {
        "total_trades": total,
        "win_trades": wins,
        "lose_trades": loses,
        "win_rate": win_rate,
        "avg_realized_rr": avg_rr,
        "total_realized_pnl_vnd": round(total_pnl),
        "pnl_by_setup": pnl_by_setup,
        "equity_curve": equity_curve,
        "open_positions_count": len(open_positions),
        "open_unrealized_pnl_vnd": round(open_unrealized),
    }
