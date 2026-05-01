"""risk_trade.py — Rule-based risk gate cho Trade pipeline.

Hard rules VN:
  1. Circuit breaker VNI < -3%
  2. Foreign room > 95% — CHỈ chặn MUA, không chặn BÁN/THOÁT
  3. T+2.5 — không mua lại trong 2 ngày
  4. Không mua đuổi khi mã đã tăng ≥ 5% trong phiên
  5. Blackbox money flow
  6. Liquidity gate — avg_vol_20d < 200k CP/ngày
  7. R:R ≥ 1.5 — THIẾU rr_ratio → CHỜ (không pass mặc định)
  8. Max loss ≤ 2% NAV — tính thuần túy relative:
     position_pct × (entry - SL) / entry ≤ 2%
     (NAV triệt tiêu → không cần biết số tuyệt đối, áp dụng cho mọi user)

KHÔNG dùng LLM.
"""

from datetime import datetime, time
from zoneinfo import ZoneInfo

from multiagents_trading_assistant import database as db


_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# R:R tối thiểu — 1.5 phù hợp VN (biên ±7%, thanh khoản mỏng)
_MIN_RR = 1.5
_CHASE_THRESHOLD = 5.0
_BREAKAWAY_CONFLUENCE = 75.0
_BREAKAWAY_SETUPS = {
    "BREAKOUT",
    "MOMENTUM_SURGE",
    "BB_SQUEEZE",
    "GOLDEN_CROSS",
    "BREAKOUT_RETEST_ENTRY",
}
_POLICY_LARGE_CAPS = {
    "PLX", "GAS", "VCB", "BID", "CTG", "MBB", "TCB", "ACB",
    "FPT", "HPG", "VIC", "VHM", "VNM", "MSN", "GVR",
}

# Thanh khoản tối thiểu — safety net dưới screener (screener đã lọc 500k)
# Chỉ chặn các trường hợp cực đoan: --symbol test, volume sụt mạnh sau ngày screener
_MIN_LIQUIDITY_VOL = 200_000  # CP/ngày
_BUY_ACTIONS = {"MUA", "GIA_TĂNG"}
_POSITION_ACTIONS = {"GIỮ", "GIA_TĂNG", "GIẢM", "BÁN"}


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
        "adjusted_position_pct": None,
    }


def _is_breakaway_exception(
    symbol: str,
    setup_type: str,
    confluence: float,
    raw_chg: float,
    limit: float,
) -> bool:
    """Allow high-quality breakaway momentum instead of a blanket chase block."""
    if setup_type not in _BREAKAWAY_SETUPS:
        return False
    if confluence < _BREAKAWAY_CONFLUENCE:
        return False
    if raw_chg >= limit * 0.97 and symbol not in _POLICY_LARGE_CAPS:
        return False
    return symbol in _POLICY_LARGE_CAPS or raw_chg < limit * 0.97


def check(state: dict) -> dict:
    symbol = state.get("symbol", "")
    trader = state.get("trader_decision", {})
    mkt    = state.get("market_context", {})
    flow   = state.get("foreign_flow_analysis", {})
    synth  = state.get("synthesis", {})

    action   = trader.get("action", "CHỜ")
    original = action
    warnings: list[str] = []
    sizing = 1.0
    adjusted_position_pct = None  # set nếu max-loss rule cắt giảm position
    internal_ctx = (state.get("memory_context") or {}).get("internal", {})
    has_position = bool(internal_ctx.get("has_position")) or (bool(symbol) and db.has_position(symbol))

    print(f"[risk_trade] {symbol} — action={action}")

    # State guard: signal entry và quản trị vị thế là 2 mode khác nhau.
    if has_position and action == "MUA":
        warnings.append("Đang có vị thế — chuyển MUA mới thành GIỮ để quản trị vị thế hiện tại")
        return {
            "final_action": "GIỮ",
            "override_reason": "Đang có vị thế — không phát MUA mới nếu chưa có rule GIA_TĂNG",
            "warnings": warnings,
            "sizing_modifier": 0.0,
            "original_action": original,
            "adjusted_position_pct": None,
        }

    if not has_position and action in _POSITION_ACTIONS:
        return _override(action, "CHỜ", f"Chưa có vị thế — action {action} không hợp lệ", warnings, 0.0)

    if action == "GIA_TĂNG":
        pos = internal_ctx.get("current_position") or db.get_position(symbol) or {}
        current_price = state.get("technical_analysis", {}).get("current_price") or state.get("market_context", {}).get("stock_current_price")
        entry_price = pos.get("entry_price")
        sl = pos.get("sl")
        nav_pct = float(pos.get("nav_pct") or 0)
        if not current_price or not entry_price:
            return _override(action, "GIỮ", "Thiếu giá hiện tại/entry hiện hữu — không gia tăng", warnings, 0.0)
        pnl_pct = (float(current_price) - float(entry_price)) / float(entry_price) * 100
        if pnl_pct < 5.0:
            return _override(action, "GIỮ", f"Vị thế mới lãi {pnl_pct:.1f}% < 5% — chưa đủ đệm để gia tăng", warnings, 0.0)
        if sl and float(sl) < float(entry_price) * 0.995:
            return _override(action, "GIỮ", "SL hiện tại chưa lên gần hòa vốn — không gia tăng", warnings, 0.0)
        if nav_pct >= 7.0:
            return _override(action, "GIỮ", f"Vị thế hiện hữu {nav_pct:.1f}% NAV đã lớn — không gia tăng", warnings, 0.0)

    # Rule 1: Circuit breaker VN-Index < -3%
    vni_chg = mkt.get("vni_change_pct")
    if vni_chg is not None and vni_chg < -3.0:
        return _override(action, "CHỜ", f"VN-Index {vni_chg:.1f}% — circuit breaker", warnings, 0.0)

    # Rule 2: Foreign room — CHỈ chặn MUA, không ảnh hưởng BÁN/THOÁT
    # Mục đích: room cao → khó kéo thêm dòng ngoại; nhưng bán vẫn phải cho phép
    if action in _BUY_ACTIONS:
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
    if action in _BUY_ACTIONS and symbol:
        try:
            recent = db.get_buys_last_n_days(symbol, 2)
            if recent:
                latest = recent[0].get("trade_date", "?")
                return _override(action, "CHỜ", f"Đã mua {symbol} {latest} — chưa qua T+2.5", warnings, 0.0)
        except Exception as e:
            print(f"[risk_trade] T+2.5 check skipped: {e}")

    # Rule 4: Không mua đuổi — mã đã tăng ≥ 5% trong phiên
    raw_chg = mkt.get("stock_day_change_pct", 0) or 0
    exch = mkt.get("exchange", "HOSE")
    limit = {"HOSE": 7.0, "HNX": 10.0}.get(exch, 15.0)

    setup_type = state.get("setup_type", "")
    confluence = float(synth.get("confluence_score") or 0.0)
    is_breakaway = (
        action in _BUY_ACTIONS
        and raw_chg >= _CHASE_THRESHOLD
        and _is_breakaway_exception(symbol, setup_type, confluence, raw_chg, limit)
    )

    if action in _BUY_ACTIONS and raw_chg >= _CHASE_THRESHOLD and not is_breakaway:
        return _override(action, "CHỜ", f"Mã đã tăng {raw_chg:.1f}% trong phiên — không mua đuổi", warnings, 0.0)

    if is_breakaway:
        sizing *= 0.5
        warnings.append(
            f"Breakaway momentum {raw_chg:.1f}%/{limit:.0f}% bien - "
            f"khong cho retest, sizing x0.5"
        )

    if raw_chg <= -(limit * 0.72):
        warnings.append(f"Gần sàn ({raw_chg:.1f}%/{limit}%) — rủi ro nhốt sàn, mất thanh khoản")

    # Rule 5: Blackbox money flow — chặn mua khi phân phối (sell/exit vẫn cho qua)
    if action in _BUY_ACTIONS:
        mfa = state.get("money_flow_analysis", {})
        mfa_regime = mfa.get("regime", "")
        mfa_bias   = mfa.get("action_bias", "")
        if mfa_regime == "DISTRIBUTION":
            return _override(action, "CHỜ", f"Blackbox: DISTRIBUTION (score={mfa.get('score', '?')}) — tay to xả hàng", warnings, 0.0)
        if mfa_bias == "AVOID_OR_EXIT":
            return _override(action, "CHỜ", f"Blackbox: {mfa_regime} → AVOID_OR_EXIT", warnings, 0.0)

    # Rule 6: Liquidity gate — safety net dưới screener (screener đã lọc ≥500k)
    # Chặn nếu avg_vol_20d < 200k — chỉ áp dụng cho MUA
    if action in _BUY_ACTIONS:
        avg_vol = mkt.get("avg_vol_20d")
        if avg_vol is not None and avg_vol < _MIN_LIQUIDITY_VOL:
            return _override(
                action, "CHỜ",
                f"Thanh khoản {avg_vol:,.0f} CP/ngày < {_MIN_LIQUIDITY_VOL:,} — quá mỏng để thoát",
                warnings, 0.0,
            )

    # Rule 7: Trading window
    if action in _BUY_ACTIONS:
        now = _now_vn()
        in_morn = time(9, 0) <= now <= time(11, 30)
        in_aft  = time(13, 0) <= now <= time(14, 25)
        if not (in_morn or in_aft):
            warnings.append(f"Ngoài giờ GD ({now.strftime('%H:%M')} VN) — signal cho phiên kế tiếp")

    # Rule 8: R:R ≥ 1.5 — THIẾU rr_ratio cũng là fail (không pass mặc định)
    # Lý do: trade ngắn hạn VN không có SL/TP rõ ràng = không đủ điều kiện ra lệnh
    if action in _BUY_ACTIONS:
        rr = trader.get("rr_ratio")
        if rr is None:
            return _override(action, "CHỜ", "Thiếu R:R ratio — không đủ thông tin để vào lệnh", warnings, 0.0)
        if rr < _MIN_RR:
            return _override(action, "CHỜ", f"R:R {rr:.2f} < {_MIN_RR} — không đủ rủi ro/lợi", warnings, 0.0)

    # Rule 9: Max loss ≤ 2% NAV — relative, không cần NAV tuyệt đối
    # max_loss% = position_pct × (entry - SL) / entry
    if action in _BUY_ACTIONS:
        ez      = trader.get("entry_zone")
        sl      = trader.get("stop_loss")
        pos_pct = trader.get("position_pct", 0) or 0
        if ez and sl and pos_pct:
            entry_mid = (ez[0] + ez[1]) / 2
            if entry_mid > sl:
                loss_per_share_pct = (entry_mid - sl) / entry_mid
                max_loss_pct       = (pos_pct / 100) * loss_per_share_pct * 100
                if max_loss_pct > 2.0:
                    new_pos = max(1, int(pos_pct * (2.0 / max_loss_pct)))
                    sizing *= (new_pos / pos_pct)
                    adjusted_position_pct = new_pos
                    warnings.append(
                        f"Max loss {max_loss_pct:.2f}% NAV > 2% — giảm position {pos_pct}→{new_pos}%"
                    )
                else:
                    adjusted_position_pct = pos_pct

    if warnings:
        print(f"[risk_trade] warnings: {'; '.join(warnings)}")
    print(f"[risk_trade] OK — final={action}, sizing={sizing:.2f}")

    return {
        "final_action":           action,
        "override_reason":        None,
        "warnings":               warnings,
        "sizing_modifier":        sizing,
        "original_action":        original,
        "adjusted_position_pct":  adjusted_position_pct,
    }
