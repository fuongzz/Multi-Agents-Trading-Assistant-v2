"""Trade dataclass cho backtest engine."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Trade:
    """Một giao dịch — mở tại entry_date, đóng tại exit_date."""

    symbol: str
    setup_type: str
    signal_date: str        # bar phát hiện setup
    entry_date: str         # bar kế tiếp — open ATO
    entry_price: float
    stop_loss: float        # SL ban đầu (bất biến, dùng để tính R:R)
    take_profit: float
    entry_atr: float = 0.0        # ATR tại thời điểm vào lệnh — dùng cho trailing SL
    confluence_score: float = 0.0 # 0 = không có review (dùng default sizing), >0 = pipeline_review score
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    # exit_reason values:
    #   "SL"                   — SL gốc bị chạm (chưa trail)
    #   "TSL"                  — Trailing SL (swing-low) bị chạm
    #   "REVERSAL_MA20_BREAK"  — 2 nến đóng dưới MA20 + MA20 giảm
    #   "REVERSAL_DOUBLE_TOP"  — 2 đỉnh ngang nhau, pull back qua midpoint
    #   "REVERSAL_TREND_BREAK" — 2 nến đóng dưới prior swing low
    #   "SAFETY_CAP"           — Đạt 120 bar tuyệt đối (zombie prevention)
    holding_bars: int = 0
    pnl_pct: Optional[float] = None
    reasons: list[str] = field(default_factory=list)

    @property
    def is_win(self) -> bool:
        return (self.pnl_pct or 0.0) > 0.0

    @property
    def is_loss(self) -> bool:
        return (self.pnl_pct or 0.0) < 0.0

    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None

    def close(self, date: str, price: float, reason: str, bars: int) -> None:
        self.exit_date = date
        self.exit_price = round(float(price), 0)
        self.exit_reason = reason
        self.holding_bars = bars
        self.pnl_pct = round(
            (float(price) - self.entry_price) / self.entry_price * 100, 2
        )
