"""TradePlan — schema cứng cho output của TradingAgents-VN.

LLM propose → parse vào đây → validator check → system decide.
Thiếu field bất kỳ → ValidationError, không tự đoán.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# MAX_HOLD theo setup type — system quyết định, không để LLM tự đặt
MAX_HOLD_BY_SETUP: dict[str, int] = {
    # Momentum ngắn
    "BREAKOUT": 15,
    "NR7": 3,
    "INSIDE_BAR": 3,
    "HAMMER": 3,
    "PIN_BAR": 3,
    "BULLISH_ENGULFING": 3,
    "MOMENTUM_SURGE": 5,
    # Default
    "BB_SQUEEZE": 5,
    "SPRING": 5,
    "RETEST": 5,
    "MACD_CROSSOVER": 5,
    "FLAG_PENNANT": 5,
    "DOUBLE_BOTTOM": 5,
    "TREND_PULLBACK": 5,
    "BREAKOUT_RETEST_ENTRY": 5,
    # Trend / Reversal
    "GOLDEN_CROSS": 8,
    "RSI_BOUNCE": 8,
    "MA_PULLBACK": 8,
}
_DEFAULT_MAX_HOLD = 5


def get_max_hold(setup_type: str) -> int:
    return MAX_HOLD_BY_SETUP.get(setup_type.upper(), _DEFAULT_MAX_HOLD)


class TradePlan(BaseModel):
    """Quyết định giao dịch từ TradingAgents-VN Portfolio Manager.

    holding_bars KHÔNG có — do system quyết định qua get_max_hold(setup_type).
    LLM chỉ propose action + zone + levels + confidence.
    """

    action: Literal["MUA", "CHO", "BAN", "TRANH"] = Field(
        description="Hành động: MUA=buy, CHO=wait, BAN=sell, TRANH=avoid"
    )
    entry_zone: tuple[float, float] = Field(
        description="Vùng giá vào lệnh (thấp, cao) — VND nghìn"
    )
    stop_loss: float = Field(description="Giá dừng lỗ — VND nghìn")
    take_profit: float = Field(description="Giá chốt lời — VND nghìn")
    position_pct: float = Field(
        ge=0.01, le=0.10,
        description="Tỷ trọng NAV đề xuất: 0.01–0.10 (1%–10%)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Mức độ tự tin: 0.0–1.0"
    )
    setup_type: str = Field(
        default="UNKNOWN",
        description="Loại setup từ screener (BREAKOUT, HAMMER, ...)"
    )
    reasons: list[str] = Field(default_factory=list, description="Lý do vào lệnh")
    risks: list[str] = Field(default_factory=list, description="Rủi ro cần lưu ý")

    @model_validator(mode="after")
    def _check_zone_order(self) -> "TradePlan":
        low, high = self.entry_zone
        if low >= high:
            raise ValueError(f"entry_zone[0] ({low}) phải < entry_zone[1] ({high})")
        return self

    @property
    def entry_mid(self) -> float:
        return round((self.entry_zone[0] + self.entry_zone[1]) / 2, 2)

    @property
    def rr_ratio(self) -> float:
        risk = self.entry_mid - self.stop_loss
        reward = self.take_profit - self.entry_mid
        if risk <= 0:
            return 0.0
        return round(reward / risk, 2)

    @property
    def max_hold(self) -> int:
        return get_max_hold(self.setup_type)
