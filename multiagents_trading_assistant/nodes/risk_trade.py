"""risk_trade.py — Rule-based risk gate cho Trade pipeline.

Port từ _legacy/risk_manager.py — giữ 5 hard rules VN, thêm:
  + rr_ratio ≥ 2 → nếu không, downgrade
  + max_loss_vnd ≤ 2% NAV (estimate)

KHÔNG dùng LLM.
"""

from datetime import datetime, time
from zoneinfo import ZoneInfo

from multiagents_trading_assistant import database as db


_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
_NAV_DEFAULT = 1_000_000_000  # 1 tỷ giả định để check max_loss


def _now_vn() -> time:
    return datetime.now(tz=_VN_TZ).time()


def _override(orig: str, new_action: str, reason: str, warnings: list, sizing: float) -> dict:
    print(f"[risk_trade] OVERRIDE {orig} → {new_action}: {reason}")
    return {
        "final_action": new_action,
        "override_reason": reason,
        "warnings": warnings,
        "sizing_modifier": sizing,
        "original_action": orig,
    }


def check(state: dict) -> dict:
    symbol = state.get("symbol", "")
    trader = state.get("trader_decision", {})
    mkt = state.get("market_context", {})
    flow = state.get("foreign_flow_analysis", {})

    action = trader.get("action", "CHỜ")
    original = action
    warnings: list[str] = []
    sizing = 1.0

    print(f"[risk_trade] {symbol} — action={action}")

    # Rule 1: Circuit breaker VN-Index < -3%
    vni_chg = mkt.get("vni_change_pct")
    if vni_chg is not None and vni_chg < -3.0:
        return _override(action, "CHỜ", f"VN-Index {vni_chg:.1f}% — circuit breaker", warnings, 0.0)

    # Rule 2: Foreign room
    room_pct = flow.get("room_usage_pct")
    if room_pct is None:
        room_status = flow.get("room_status", "")
        if room_status == "CRITICAL":
            room_pct = 96.0
        elif room_status == "HIGH":
            room_pct = 92.0

    if room_pct is not None:
        if room_pct > 95:
            return _override(action, "CHỜ", f"Room ngoại {room_pct:.0f}% — không mua thêm", warnings, 0.0)
        elif room_pct > 90:
            sizing *= 0.5
            warnings.append(f"Room {room_pct:.0f}% (HIGH) — sizing ×0.5")
        elif room_pct > 80:
            sizing *= 0.8
            warnings.append(f"Room {room_pct:.0f}% (MEDIUM) — sizing ×0.8")

    # Rule 3: T+2.5 — hàng về chiều T+2, không mua lại trong 2 ngày
    if action == "MUA" and symbol:
        try:
            recent = db.get_buys_last_n_days(symbol, 2)
            if recent:
                latest = recent[0].get("trade_date", "?")
                return _override(action, "CHỜ", f"Đã mua {symbol} {latest} — chưa qua T+2.5", warnings, 0.0)
        except Exception as e:
            print(f"[risk_trade] T+2.5 check skipped: {e}")

    # Rule 4: Biên độ giá — HOSE ±7%, HNX ±10%, UPCoM ±15%
    raw_chg = mkt.get("stock_day_change_pct", 0) or 0
    stk_chg = abs(raw_chg)
    exch = mkt.get("exchange", "HOSE")
    if exch == "HOSE":
        limit = 7.0
    elif exch == "HNX":
        limit = 10.0
    else:  # UPCoM
        limit = 15.0
    used = (stk_chg / limit) * 100
    if used > 71.4:
        sizing *= 0.5
        warnings.append(f"Đã dùng {stk_chg:.1f}%/{limit}% biên — sizing ×0.5")
    if raw_chg <= -(limit * 0.72):
        warnings.append(f"Gần sàn ({raw_chg:.1f}%/{limit}%) — rủi ro nhốt sàn, mất thanh khoản")

    # Rule 5: Trading window
    if action == "MUA":
        now = _now_vn()
        in_morn = time(9, 0) <= now <= time(11, 30)
        in_aft = time(13, 0) <= now <= time(14, 25)
        if not (in_morn or in_aft):
            warnings.append(f"Ngoài giờ GD ({now.strftime('%H:%M')} VN) — signal cho phiên kế tiếp")

    # Rule extra A: R:R ≥ 2
    if action == "MUA":
        rr = trader.get("rr_ratio")
        if rr is not None and rr < 2.0:
            return _override(action, "CHỜ", f"R:R {rr:.2f} < 2 — không đủ rủi ro/lợi", warnings, 0.0)

    # Rule extra B: max_loss ≤ 2% NAV
    if action == "MUA":
        ez = trader.get("entry_zone")
        sl = trader.get("stop_loss")
        pos_pct = trader.get("position_pct", 0) or 0
        if ez and sl and pos_pct:
            entry_mid = (ez[0] + ez[1]) / 2
            if entry_mid > sl:
                loss_pct_per_share = (entry_mid - sl) / entry_mid
                position_vnd = _NAV_DEFAULT * (pos_pct / 100)
                max_loss_vnd = position_vnd * loss_pct_per_share
                max_loss_pct_nav = max_loss_vnd / _NAV_DEFAULT * 100
                if max_loss_pct_nav > 2.0:
                    new_pos = max(1, int(pos_pct * (2.0 / max_loss_pct_nav)))
                    sizing *= (new_pos / pos_pct)
                    warnings.append(f"Max loss {max_loss_pct_nav:.2f}% NAV > 2% — giảm position {pos_pct}→{new_pos}%")

    if warnings:
        print(f"[risk_trade] warnings: {'; '.join(warnings)}")
    print(f"[risk_trade] OK — final={action}, sizing={sizing:.2f}")

    return {
        "final_action": action,
        "override_reason": None,
        "warnings": warnings,
        "sizing_modifier": sizing,
        "original_action": original,
    }
