"""
risk_manager.py — Rule-based engine kiểm tra 5 hard rules VN.

KHÔNG dùng LLM — chỉ logic thuần.
Chạy sau Trader Agent (Node 8), có quyền override bất kỳ quyết định nào.
"""

from datetime import datetime, time
from zoneinfo import ZoneInfo

from multiagents_trading_assistant import database as db

# Timezone Việt Nam
_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _get_vietnam_time() -> time:
    """Trả về giờ hiện tại theo múi giờ Việt Nam."""
    return datetime.now(tz=_VN_TZ).time()


def _override(original: str, new_action: str, reason: str, warnings: list, sizing: float) -> dict:
    """Tạo dict override — thay action và ghi lý do."""
    print(f"[risk_manager] OVERRIDE {original} → {new_action}: {reason}")
    return {
        "final_action":    new_action,
        "override_reason": reason,
        "warnings":        warnings,
        "sizing_modifier": sizing,
        "original_action": original,
    }


def check(state: dict) -> dict:
    """
    Kiểm tra 5 hard rules VN — override nếu vi phạm.

    Args:
        state: TradingState từ orchestrator

    Returns:
        dict với keys: final_action, override_reason, warnings, sizing_modifier, original_action
    """
    symbol       = state.get("symbol", "")
    trader_dec   = state.get("trader_decision", {})
    market_ctx   = state.get("market_context", {})
    ff_analysis  = state.get("foreign_flow_analysis", {})

    action          = trader_dec.get("action", "CHỜ")
    original_action = action
    warnings: list[str] = []
    sizing_modifier = 1.0

    print(f"[risk_manager] Kiểm tra {symbol} — action={action}")

    # ──────────────────────────────────────────────
    # Rule 1 — Circuit Breaker: VN-Index giảm > 3%
    # ──────────────────────────────────────────────
    vni_change = market_ctx.get("vni_change_pct")
    if vni_change is not None and vni_change < -3.0:
        return _override(
            action, "CHỜ",
            f"VN-Index giảm {vni_change:.1f}% — circuit breaker kích hoạt",
            warnings, 0.0,
        )

    # ──────────────────────────────────────────────
    # Rule 2 — Foreign Room Trap
    # ──────────────────────────────────────────────
    room_pct = ff_analysis.get("room_usage_pct")

    # Thử fallback từ room_status nếu không có room_usage_pct
    if room_pct is None:
        room_status = ff_analysis.get("room_status", "")
        if room_status == "CRITICAL":
            room_pct = 96.0  # > 95 → force CHỜ
        elif room_status == "HIGH":
            room_pct = 92.0  # > 90 → giảm sizing

    if room_pct is not None:
        if room_pct > 95:
            return _override(
                action, "CHỜ",
                f"Room ngoại {room_pct:.0f}% — không thể mua thêm",
                warnings, 0.0,
            )
        elif room_pct > 90:
            sizing_modifier *= 0.5
            warnings.append(f"Room ngoại {room_pct:.0f}% (HIGH) — giảm sizing 50%")
            print(f"[risk_manager] Room cao {room_pct:.0f}% → sizing ×0.5")
        elif room_pct > 80:
            sizing_modifier *= 0.8
            warnings.append(f"Room ngoại {room_pct:.0f}% (MEDIUM) — giảm sizing 20%")

    # ──────────────────────────────────────────────
    # Rule 3 — T+3 Constraint: không mua lại trong 3 ngày
    # ──────────────────────────────────────────────
    if action == "MUA" and symbol:
        recent_buys = db.get_buys_last_n_days(symbol, 3)
        if recent_buys:
            latest = recent_buys[0].get("trade_date", "?")
            return _override(
                action, "CHỜ",
                f"Đã mua {symbol} ngày {latest} — chưa qua T+3",
                warnings, 0.0,
            )

    # ──────────────────────────────────────────────
    # Rule 4 — Biên Độ Giá: giá đã gần hết biên độ
    # ──────────────────────────────────────────────
    stock_day_change = abs(market_ctx.get("stock_day_change_pct", 0) or 0)
    exchange = market_ctx.get("exchange", "HOSE")
    limit    = 7.0 if exchange == "HOSE" else 10.0
    used_pct = (stock_day_change / limit) * 100  # % đã dùng của biên độ

    if used_pct > 71.4:  # ~5% của 7% HOSE = 71.4%
        sizing_modifier *= 0.5
        msg = f"Giá đã dùng {stock_day_change:.1f}% biên độ ({used_pct:.0f}% biên) — giảm sizing"
        warnings.append(msg)
        print(f"[risk_manager] {msg}")

    # ──────────────────────────────────────────────
    # Rule 5 — Trading Window: chỉ lệnh trong giờ GD
    # ──────────────────────────────────────────────
    if action in ("MUA", "BÁN"):
        now_vn       = _get_vietnam_time()
        in_morning   = time(9, 0) <= now_vn <= time(11, 30)
        in_afternoon = time(13, 0) <= now_vn <= time(14, 25)  # 14:25 là trước ATC

        if not (in_morning or in_afternoon):
            return _override(
                action, "CHỜ",
                f"Ngoài giờ giao dịch ({now_vn.strftime('%H:%M')} VN) — chờ phiên sau",
                warnings, 0.0,
            )

    # ──────────────────────────────────────────────
    # Rule BÁN — không short selling
    # ──────────────────────────────────────────────
    if action == "BÁN" and symbol:
        if not db.has_position(symbol):
            return _override(
                action, "CHỜ",
                f"Không có vị thế {symbol} để bán — không short selling",
                warnings, 0.0,
            )

    # ──────────────────────────────────────────────
    # Tất cả rules OK — trả về action gốc
    # ──────────────────────────────────────────────
    if warnings:
        print(f"[risk_manager] Cảnh báo: {'; '.join(warnings)}")

    print(f"[risk_manager] OK — final_action={action}, sizing={sizing_modifier:.1f}")
    return {
        "final_action":    action,
        "override_reason": None,
        "warnings":        warnings,
        "sizing_modifier": sizing_modifier,
        "original_action": original_action,
    }


# ──────────────────────────────────────────────
# Test trực tiếp
# ──────────────────────────────────────────────

if __name__ == "__main__":
    db.init_db()

    print("=== Test 1: Normal MUA trong giờ GD ===")
    state1 = {
        "symbol": "VNM",
        "trader_decision": {"action": "MUA"},
        "market_context": {"vni_change_pct": -0.5, "stock_day_change_pct": 1.2, "exchange": "HOSE"},
        "foreign_flow_analysis": {"room_usage_pct": 75.0},
    }
    r1 = check(state1)
    print("→", r1)

    print("\n=== Test 2: Circuit Breaker (VNI -3.5%) ===")
    state2 = {
        "symbol": "HPG",
        "trader_decision": {"action": "MUA"},
        "market_context": {"vni_change_pct": -3.5, "stock_day_change_pct": 0.5, "exchange": "HOSE"},
        "foreign_flow_analysis": {},
    }
    r2 = check(state2)
    print("→", r2)

    print("\n=== Test 3: Room ngoại CRITICAL (96%) ===")
    state3 = {
        "symbol": "VIC",
        "trader_decision": {"action": "MUA"},
        "market_context": {"vni_change_pct": 0.3, "stock_day_change_pct": 0.8, "exchange": "HOSE"},
        "foreign_flow_analysis": {"room_usage_pct": 96.5},
    }
    r3 = check(state3)
    print("→", r3)

    print("\n=== Test 4: BÁN khi không có vị thế ===")
    state4 = {
        "symbol": "MSN",
        "trader_decision": {"action": "BÁN"},
        "market_context": {"vni_change_pct": 0.2, "stock_day_change_pct": -0.5, "exchange": "HOSE"},
        "foreign_flow_analysis": {},
    }
    r4 = check(state4)
    print("→", r4)

    print("\n=== Test 5: CHỜ — bỏ qua trading window ===")
    state5 = {
        "symbol": "FPT",
        "trader_decision": {"action": "CHỜ"},
        "market_context": {"vni_change_pct": 0.1, "stock_day_change_pct": 0.3, "exchange": "HOSE"},
        "foreign_flow_analysis": {},
    }
    r5 = check(state5)
    print("→", r5)
