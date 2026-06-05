"""intraday_execution_agent.py — Định thời vào lệnh trong phiên (Trade pipeline).

Thuần RULE-BASED (không LLM) — formalize logic đã validate trong nghiên cứu
`scripts/run_flow_v2_intraday_entry_timing_experiment.py`:

  - Dynamic entry: vào lệnh tại checkpoint đầu tiên mà giá T+1 vẫn validate setup T
      ret_from_signal_close ≥ -0.5%  AND  price ≥ ma20 × 0.995
  - Same-clock intraday guard: HỦY chỉ khi giá phá rõ setup T
      drop_from_signal_close ≤ -2.5%  OR  price < ma20 × 0.985

Biến khuyến nghị MUA (đặt ATO mù) thành điểm vào tối ưu trong phiên:
  ENTER_NOW : giá đã về vùng hợp lệ, vào lệnh
  WAIT      : giá chưa về vùng tốt nhưng chưa phá — chờ checkpoint sau
  CANCEL    : giá phá setup — bỏ tín hiệu

Dùng ở session_monitor (nhánh [C], mỗi 5 phút) với giá live DNSE WS.
Không gọi API nặng — chỉ tính toán trên số đầu vào.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Ngưỡng đã tune (khớp scripts/run_flow_v2_intraday_entry_timing_experiment.py)
_MIN_RET_FOR_ENTRY = -0.005      # giá ≥ signal_close × 0.995 mới đủ điều kiện vào
_MA20_ENTER_BUFFER = 0.995       # vào: price ≥ ma20 × 0.995
_MAX_DROP_FOR_CANCEL = -0.025    # drop ≤ -2.5% so signal_close → hủy
_MA20_CANCEL_BUFFER = 0.985      # hủy: price < ma20 × 0.985
_CHASE_ABOVE_ZONE = 0.005        # giá > entry_high × 1.005 → coi như mua đuổi → WAIT


def decide(
    symbol: str,
    signal_close: float,
    current_price: float,
    *,
    ma20: float | None = None,
    entry_zone: tuple[float, float] | list[float] | None = None,
    now_clock: str | None = None,
) -> dict:
    """Quyết định thời điểm vào lệnh trong phiên cho 1 khuyến nghị MUA.

    Args:
        signal_close: giá đóng cửa phiên ra tín hiệu (cơ sở của setup).
        current_price: giá khớp lệnh hiện tại (live).
        ma20: MA20 tại phiên tín hiệu (None = bỏ qua điều kiện MA).
        entry_zone: (low, high) vùng vào dự kiến — chống mua đuổi.
        now_clock: "HH:MM" VN (None = giờ hiện tại).

    Returns:
        dict: action (ENTER_NOW/WAIT/CANCEL), suggested_price, ret_from_signal_close,
        reason, clock.
    """
    clock = now_clock or datetime.now(tz=_VN_TZ).strftime("%H:%M")

    if not signal_close or signal_close <= 0 or not current_price or current_price <= 0:
        return _result("WAIT", current_price, None, clock,
                       "Thiếu giá tham chiếu — chờ dữ liệu.")

    ret = current_price / signal_close - 1.0

    # ── CANCEL: giá phá rõ setup ──
    if ret <= _MAX_DROP_FOR_CANCEL:
        return _result("CANCEL", current_price, ret, clock,
                       f"Giá rớt {ret*100:.1f}% so phiên tín hiệu (≤ -2.5%) — setup gãy.")
    if ma20 and current_price < ma20 * _MA20_CANCEL_BUFFER:
        return _result("CANCEL", current_price, ret, clock,
                       f"Giá {current_price:,.0f} thủng MA20×0.985 ({ma20*_MA20_CANCEL_BUFFER:,.0f}) — setup gãy.")

    # ── Chống mua đuổi: giá vượt xa entry zone ──
    if entry_zone and len(entry_zone) == 2 and entry_zone[1]:
        if current_price > entry_zone[1] * (1 + _CHASE_ABOVE_ZONE):
            return _result("WAIT", entry_zone[1], ret, clock,
                           f"Giá {current_price:,.0f} vượt entry zone ({entry_zone[1]:,.0f}) — không mua đuổi, chờ pullback.")

    # ── ENTER_NOW: giá đã validate setup ──
    ma_ok = (ma20 is None) or (current_price >= ma20 * _MA20_ENTER_BUFFER)
    if ret >= _MIN_RET_FOR_ENTRY and ma_ok:
        return _result("ENTER_NOW", current_price, ret, clock,
                       f"Giá {current_price:,.0f} validate setup (ret {ret*100:+.1f}%, trên MA20) — vào lệnh.")

    # ── WAIT: giá tụt dưới ngưỡng vào nhưng chưa phá ──
    return _result("WAIT", current_price, ret, clock,
                   f"Giá {current_price:,.0f} dưới ngưỡng vào (ret {ret*100:+.1f}%) nhưng chưa gãy — chờ hồi.")


def _result(action: str, suggested_price, ret, clock: str, reason: str) -> dict:
    return {
        "action": action,
        "suggested_price": round(float(suggested_price), 0) if suggested_price else None,
        "ret_from_signal_close": round(ret, 4) if ret is not None else None,
        "clock": clock,
        "reason": reason,
    }


if __name__ == "__main__":
    import json
    # signal_close=100, ma20=98
    cases = [
        ("validate", 100.0, 100.5, 98.0, (99.0, 101.0)),
        ("pullback_ok", 100.0, 99.8, 98.0, (99.0, 101.0)),
        ("broken_drop", 100.0, 97.0, 98.0, (99.0, 101.0)),
        ("broken_ma", 100.0, 96.3, 98.0, (99.0, 101.0)),
        ("chase", 100.0, 102.0, 98.0, (99.0, 101.0)),
        ("wait_dip", 100.0, 99.0, 98.0, (99.0, 101.0)),
    ]
    for name, sc, cp, ma, ez in cases:
        print(name, "→", json.dumps(decide("TEST", sc, cp, ma20=ma, entry_zone=ez, now_clock="10:30"), ensure_ascii=False))
