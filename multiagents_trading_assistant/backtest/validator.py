"""Trade Plan Validator — kiểm tra TradePlan trước khi simulate.

Backtest chỉ simulate khi is_valid=True.
LLM chỉ cần lú một phát là backtest bẩn — validator ngăn điều đó.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from multiagents_trading_assistant.tradingagents_vn.schema import TradePlan

MIN_RR = 1.5
MIN_POSITION_PCT = 0.01
MAX_POSITION_PCT = 0.10
MAX_ZONE_WIDTH_PCT = 0.04       # entry_zone không rộng hơn 4% của mid
MAX_ZONE_DISTANCE_PCT = 0.07    # entry_zone không xa current_price quá 7%
MAX_SL_PCT = 0.06               # SL không quá 6% dưới entry_mid


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


def validate_trade_plan(
    plan: TradePlan,
    current_price: float,
) -> ValidationResult:
    """Validate TradePlan trước khi simulate.

    Args:
        plan: TradePlan đã parse từ LLM output.
        current_price: Giá đóng cửa cuối cùng tại as_of_date (reference để check zone distance).

    Returns:
        ValidationResult(is_valid, errors)
    """
    errors: list[str] = []

    # Chỉ validate khi action = MUA
    if plan.action != "MUA":
        return ValidationResult(is_valid=True)

    entry_low, entry_high = plan.entry_zone
    entry_mid = plan.entry_mid
    sl = plan.stop_loss
    tp = plan.take_profit

    # 1. SL phải dưới entry
    if sl >= entry_low:
        errors.append(f"SL ({sl}) >= entry_zone_low ({entry_low}) — SL phải thấp hơn vùng vào")

    # 2. TP phải trên entry
    if tp <= entry_high:
        errors.append(f"TP ({tp}) <= entry_zone_high ({entry_high}) — TP phải cao hơn vùng vào")

    # 3. R:R >= MIN_RR
    rr = plan.rr_ratio
    if rr < MIN_RR:
        errors.append(f"R:R = {rr:.2f} < {MIN_RR} — quá thấp để giao dịch")

    # 4. position_pct hợp lệ (đã validate ở Pydantic nhưng double-check)
    if not (MIN_POSITION_PCT <= plan.position_pct <= MAX_POSITION_PCT):
        errors.append(
            f"position_pct={plan.position_pct:.2%} ngoài range "
            f"[{MIN_POSITION_PCT:.0%}, {MAX_POSITION_PCT:.0%}]"
        )

    # 5. Zone không quá rộng
    zone_width_pct = (entry_high - entry_low) / entry_mid if entry_mid > 0 else 0
    if zone_width_pct > MAX_ZONE_WIDTH_PCT:
        errors.append(
            f"entry_zone quá rộng: {zone_width_pct:.1%} > {MAX_ZONE_WIDTH_PCT:.0%} "
            f"(low={entry_low}, high={entry_high})"
        )

    # 6. Zone không xa current_price quá 7% (không mua quá cao / quá xa)
    if current_price > 0:
        dist_pct = abs(entry_mid - current_price) / current_price
        if dist_pct > MAX_ZONE_DISTANCE_PCT:
            errors.append(
                f"entry_zone mid ({entry_mid}) cách current_price ({current_price}) "
                f"{dist_pct:.1%} > {MAX_ZONE_DISTANCE_PCT:.0%}"
            )

    # 7. SL không quá sâu
    if entry_mid > 0 and sl > 0:
        sl_depth = (entry_mid - sl) / entry_mid
        if sl_depth > MAX_SL_PCT:
            errors.append(
                f"SL depth {sl_depth:.1%} > {MAX_SL_PCT:.0%} — SL quá xa, R:R không thực tế"
            )

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
