"""Position lifecycle abstractions — Signal, Leg, Position, PositionStage.

Triết lý:
  - Signal = mô tả thị trường (setup phát hiện được). Không quyết định mua.
  - Position = trạng thái sống của một mã (gồm nhiều Leg).
  - Leg = một lần mua/bán cụ thể.
  - PositionStage = anh đang ở đâu trong vòng đời vị thế.

Mỗi Position có thể có nhiều Leg. Avg price, total nav_pct được tính lại sau
mỗi Leg. SL có thể được nâng lên (trailing) qua nhiều bước trong cùng Position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PositionStage(str, Enum):
    """5 trạng thái của vòng đời vị thế."""

    NO_POSITION    = "no_position"     # chưa có gì
    PROBE          = "probe"           # đã mua thử, chưa xác nhận
    ADDED_STRENGTH = "added_strength"  # đã add lần 1, trend xác nhận
    FULL           = "full"            # đã full vị thế
    REDUCING       = "reducing"        # đang thoát dần
    CLOSED         = "closed"          # đã đóng hoàn toàn


class LegReason(str, Enum):
    """Lý do của mỗi Leg — để playbook log lại quyết định."""

    PROBE          = "probe"           # mua thăm dò 30%
    ADD_STRENGTH   = "add_strength"    # add khi xác nhận
    ADD_TREND      = "add_trend"       # add lần cuối khi trend rõ
    REDUCE_HALF    = "reduce_half"     # giảm một nửa
    REDUCE_PARTIAL = "reduce_partial"  # giảm một phần
    EXIT_ALL       = "exit_all"        # thoát hết
    EXIT_SL        = "exit_sl"         # SL bị chạm
    EXIT_REVERSAL  = "exit_reversal"   # tín hiệu đảo chiều


@dataclass
class Signal:
    """Mô tả tín hiệu thị trường — KHÔNG bao gồm quyết định mua/bán."""

    symbol: str
    date: str
    setup_type: str              # tên setup (BREAKOUT, DOUBLE_BOTTOM, ...)
    confluence_score: float      # 0-100, từ synthesis
    entry_zone_low: float
    entry_zone_high: float
    suggested_sl: float          # SL kỹ thuật (dưới swing low)
    atr: float = 0.0
    regime: str = "UNKNOWN"      # UPTREND / SIDEWAY / DOWNTREND
    money_flow_score: int = 0    # mf_score
    money_flow_regime: str = "NEUTRAL"
    playbook_hint: str = ""
    risk_multiplier: float = 1.0
    sector_name: str = ""
    sector_regime: str = "UNKNOWN"
    sector_rank: int = 0
    sector_score: float = 0.0
    regime_notes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class Leg:
    """Một lần mua hoặc bán."""

    date: str
    price: float                 # giá fill
    qty_pct: float               # % NAV (positive=buy, negative=sell)
    reason: LegReason
    sl_at_leg: float = 0.0       # SL hiện tại sau leg này
    bar_idx: int = -1            # index bar khớp lệnh, dùng cho T+2 theo từng leg

    @property
    def is_buy(self) -> bool:
        return self.qty_pct > 0

    @property
    def is_sell(self) -> bool:
        return self.qty_pct < 0


@dataclass
class Position:
    """Trạng thái sống của một mã.

    Các invariants:
      - stage chỉ chuyển theo state machine: NO_POSITION → PROBE → ADDED_STRENGTH
        → FULL → REDUCING → CLOSED.
      - avg_price luôn = weighted average của các buy leg còn lại.
      - total_nav_pct = sum của các leg (buy + sell). Khi = 0 → CLOSED.
      - current_sl chỉ tăng, không bao giờ giảm trong vòng đời position.
    """

    symbol: str
    stage: PositionStage = PositionStage.NO_POSITION
    legs: list[Leg] = field(default_factory=list)
    current_sl: float = 0.0
    intended_total_pct: float = 0.0   # tổng NAV dự kiến nếu vào full (vd 9%)

    # Stage-specific tracking
    peak_close: float = 0.0           # đỉnh close kể từ lúc mở vị thế
    entry_atr: float = 0.0            # ATR tại bar đầu tiên
    open_date: Optional[str] = None
    close_date: Optional[str] = None
    realized_pnl_nav_pct: float = 0.0 # PnL đã chốt, đơn vị % NAV portfolio
    playbook_name: Optional[str] = None

    # ─── Derived properties ──────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self.stage not in (PositionStage.NO_POSITION, PositionStage.CLOSED)

    @property
    def total_nav_pct(self) -> float:
        """% NAV đang chiếm (sau khi đã trừ các sell leg)."""
        return round(sum(leg.qty_pct for leg in self.legs), 4)

    @property
    def avg_price(self) -> float:
        """Weighted average price của các buy leg, sau khi đã trừ sell theo FIFO.

        Implementation đơn giản: nếu chỉ có buy → weighted avg. Nếu có sell,
        coi sell như giảm proportionally (không FIFO chính xác để tránh phức tạp).
        """
        buy_value = 0.0
        buy_qty = 0.0
        for leg in self.legs:
            if leg.is_buy:
                buy_value += leg.price * leg.qty_pct
                buy_qty += leg.qty_pct
        if buy_qty <= 0:
            return 0.0
        return round(buy_value / buy_qty, 2)

    @property
    def total_invested_pct(self) -> float:
        """Tổng % NAV đã mua vào (gross, không trừ sell)."""
        return round(sum(leg.qty_pct for leg in self.legs if leg.is_buy), 4)

    @property
    def fill_ratio(self) -> float:
        """Tỷ lệ đã lấp đầy so với intended (0.0 - 1.0)."""
        if self.intended_total_pct <= 0:
            return 0.0
        return min(1.0, self.total_nav_pct / self.intended_total_pct)

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """% PnL trên vốn của leg (so với avg price)."""
        ap = self.avg_price
        if ap <= 0:
            return 0.0
        return round((current_price - ap) / ap * 100, 2)

    def unrealized_pnl_nav_pct(self, current_price: float) -> float:
        """% PnL trên NAV portfolio (đã nhân với position size)."""
        return round(self.unrealized_pnl_pct(current_price) * self.total_nav_pct / 100, 4)

    def max_loss_if_sl_hit_nav_pct(self) -> float:
        """Rủi ro tối đa nếu SL bị chạm — tính theo % NAV portfolio."""
        ap = self.avg_price
        if ap <= 0 or self.current_sl <= 0:
            return 0.0
        loss_pct = (ap - self.current_sl) / ap
        return round(max(0.0, loss_pct * self.total_nav_pct), 4)

    # ─── Mutations (chỉ qua các method, không sửa trực tiếp) ────────────────

    def add_leg(self, leg: Leg) -> None:
        """Thêm một leg và cập nhật stage tự động (do playbook quyết định stage mới)."""
        if self.is_open is False and leg.is_buy:
            self.open_date = leg.date
        self.legs.append(leg)
        # SL không bao giờ giảm
        if leg.sl_at_leg > 0:
            self.current_sl = max(self.current_sl, leg.sl_at_leg)
        # Nếu sell làm total về 0 → close
        if abs(self.total_nav_pct) < 1e-6:
            self.stage = PositionStage.CLOSED
            self.close_date = leg.date

    def update_peak(self, close: float) -> None:
        if close > self.peak_close:
            self.peak_close = close

    def update_sl(self, new_sl: float) -> None:
        """SL chỉ tăng."""
        if new_sl > self.current_sl:
            self.current_sl = round(new_sl, 2)


# ─── Stage transition rules ──────────────────────────────────────────────────

# Allowed transitions — playbook PHẢI tuân theo bảng này.
ALLOWED_TRANSITIONS: dict[PositionStage, set[PositionStage]] = {
    PositionStage.NO_POSITION:    {PositionStage.PROBE, PositionStage.FULL},
    PositionStage.PROBE:          {PositionStage.ADDED_STRENGTH, PositionStage.REDUCING, PositionStage.CLOSED},
    PositionStage.ADDED_STRENGTH: {PositionStage.FULL, PositionStage.REDUCING, PositionStage.CLOSED},
    PositionStage.FULL:           {PositionStage.REDUCING, PositionStage.CLOSED},
    PositionStage.REDUCING:       {PositionStage.CLOSED, PositionStage.FULL},  # có thể add lại
    PositionStage.CLOSED:         {PositionStage.NO_POSITION},                  # reset
}


def is_valid_transition(from_stage: PositionStage, to_stage: PositionStage) -> bool:
    return to_stage in ALLOWED_TRANSITIONS.get(from_stage, set()) or from_stage == to_stage
