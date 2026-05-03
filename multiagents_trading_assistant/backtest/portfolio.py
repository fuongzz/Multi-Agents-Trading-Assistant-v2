"""PortfolioEngine — quản lý NAV, cash, exposure, risk toàn danh mục.

Khác với Position (state của 1 mã), Portfolio quản lý cả tập:
  - cash_pct còn lại (chưa invest)
  - tổng exposure (sum total_nav_pct của tất cả position đang mở)
  - max symbols, sector exposure cap
  - total risk (sum max_loss_if_sl_hit của tất cả position)
  - realized PnL tích lũy

NAV được normalize về 100% (không track absolute VND). Mọi % đều là % của
NAV ban đầu — backtest không cần biết NAV tuyệt đối thực tế.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from multiagents_trading_assistant.backtest.lifecycle import Position, PositionStage


@dataclass
class PortfolioConstraints:
    """Ràng buộc danh mục — giống risk_trade nhưng cho backtest."""

    max_open_positions: int = 5            # tối đa 5 mã mở cùng lúc
    max_total_exposure_pct: float = 50.0   # tổng NAV invest tối đa 50%
    max_total_risk_pct: float = 6.0        # tổng max_loss nếu tất cả SL hit ≤ 6% NAV
    max_position_pct: float = 10.0         # 1 mã không quá 10% NAV
    max_per_sector_pct: float = 20.0       # 1 ngành không quá 20% NAV (Phase 2)


@dataclass
class PortfolioState:
    cash_pct: float = 100.0
    realized_pnl_nav_pct: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    closed_positions: list[Position] = field(default_factory=list)

    @property
    def open_positions(self) -> dict[str, Position]:
        return {s: p for s, p in self.positions.items() if p.is_open}

    @property
    def total_exposure_pct(self) -> float:
        return round(sum(p.total_nav_pct for p in self.open_positions.values()), 4)

    @property
    def total_risk_pct(self) -> float:
        return round(
            sum(p.max_loss_if_sl_hit_nav_pct() for p in self.open_positions.values()),
            4,
        )

    @property
    def num_open(self) -> int:
        return len(self.open_positions)

    def equity_pct(self, mark_to_market: dict[str, float]) -> float:
        """NAV hiện tại = cash + realized + unrealized. Reference = 100.0."""
        unreal = 0.0
        for sym, pos in self.open_positions.items():
            px = mark_to_market.get(sym)
            if px is not None:
                unreal += pos.unrealized_pnl_nav_pct(px)
        return round(100.0 + self.realized_pnl_nav_pct + unreal, 4)


class PortfolioEngine:
    """Validate hành động trước khi commit, và commit khi pass."""

    def __init__(self, constraints: Optional[PortfolioConstraints] = None):
        self.constraints = constraints or PortfolioConstraints()
        self.state = PortfolioState()

    # ─── Validation ──────────────────────────────────────────────────────────

    def can_open_new(
        self,
        symbol: str,
        intended_pct: float,
        *,
        entry_price: float | None = None,
        stop_loss: float | None = None,
    ) -> tuple[bool, str]:
        """Kiểm tra có được mở vị thế mới không (không tính add vào vị thế cũ)."""
        s = self.state
        if symbol in s.open_positions:
            return False, f"{symbol} already open"
        if s.num_open >= self.constraints.max_open_positions:
            return False, f"max_open_positions ({self.constraints.max_open_positions}) reached"
        if intended_pct > self.constraints.max_position_pct:
            return False, f"intended {intended_pct}% > max_position_pct"
        if s.total_exposure_pct + intended_pct > self.constraints.max_total_exposure_pct:
            return False, "would exceed max_total_exposure_pct"
        if intended_pct > s.cash_pct:
            return False, f"insufficient cash ({s.cash_pct}%)"
        projected_risk = self._projected_total_risk_for_new(
            intended_pct, entry_price, stop_loss
        )
        if projected_risk > self.constraints.max_total_risk_pct:
            return False, "would exceed max_total_risk_pct"
        return True, "ok"

    def can_add(
        self,
        symbol: str,
        add_pct: float,
        position: Position,
        *,
        entry_price: float | None = None,
        stop_loss: float | None = None,
    ) -> tuple[bool, str]:
        s = self.state
        if add_pct <= 0:
            return False, "add_pct must be positive"
        if position.total_nav_pct + add_pct > self.constraints.max_position_pct:
            return False, "would exceed max_position_pct"
        if s.total_exposure_pct + add_pct > self.constraints.max_total_exposure_pct:
            return False, "would exceed max_total_exposure_pct"
        if add_pct > s.cash_pct:
            return False, f"insufficient cash ({s.cash_pct}%)"
        projected_risk = self._projected_total_risk_for_add(
            position, add_pct, entry_price, stop_loss
        )
        if projected_risk > self.constraints.max_total_risk_pct:
            return False, "would exceed max_total_risk_pct"
        return True, "ok"

    def _projected_total_risk_for_new(
        self,
        qty_pct: float,
        entry_price: float | None,
        stop_loss: float | None,
    ) -> float:
        return round(self.state.total_risk_pct + self._risk_nav(qty_pct, entry_price, stop_loss), 4)

    def _projected_total_risk_for_add(
        self,
        position: Position,
        add_pct: float,
        entry_price: float | None,
        stop_loss: float | None,
    ) -> float:
        old_risk = position.max_loss_if_sl_hit_nav_pct()
        new_sl = max(position.current_sl, float(stop_loss or 0.0))
        total_qty = position.total_nav_pct + add_pct
        if total_qty <= 0:
            new_risk = 0.0
        else:
            ap = position.avg_price
            px = float(entry_price or 0.0)
            if ap <= 0 or px <= 0 or new_sl <= 0:
                new_risk = old_risk
            else:
                new_avg = ((ap * position.total_nav_pct) + (px * add_pct)) / total_qty
                new_risk = max(0.0, (new_avg - new_sl) / new_avg * total_qty)
        return round(self.state.total_risk_pct - old_risk + new_risk, 4)

    @staticmethod
    def _risk_nav(
        qty_pct: float,
        entry_price: float | None,
        stop_loss: float | None,
    ) -> float:
        entry = float(entry_price or 0.0)
        sl = float(stop_loss or 0.0)
        if qty_pct <= 0 or entry <= 0 or sl <= 0:
            return 0.0
        return max(0.0, (entry - sl) / entry * qty_pct)

    # ─── Commit ──────────────────────────────────────────────────────────────

    def get_or_create(self, symbol: str) -> Position:
        if symbol not in self.state.positions:
            self.state.positions[symbol] = Position(symbol=symbol)
        return self.state.positions[symbol]

    def commit_buy(self, position: Position, qty_pct: float, price: float) -> None:
        """Trừ cash. Không thay đổi realized PnL."""
        self.state.cash_pct = round(self.state.cash_pct - qty_pct, 4)

    def commit_sell(self, position: Position, qty_pct_abs: float, price: float) -> None:
        """qty_pct_abs là số dương — số % NAV bán ra.

        Realized PnL = (price - avg_price) / avg_price × qty_pct_abs (% NAV).
        Cash += qty_pct_abs (proxy: vốn quay lại) + PnL.
        """
        ap = position.avg_price
        if ap <= 0:
            self.state.cash_pct = round(self.state.cash_pct + qty_pct_abs, 4)
            return
        pnl_on_capital_pct = (price - ap) / ap         # vd 0.05 = +5%
        realized_nav = round(pnl_on_capital_pct * qty_pct_abs, 4)
        self.state.realized_pnl_nav_pct = round(
            self.state.realized_pnl_nav_pct + realized_nav, 4
        )
        # Cash quay về theo MTM (capital × (1 + return))
        cash_back = round(qty_pct_abs * (1 + pnl_on_capital_pct), 4)
        self.state.cash_pct = round(self.state.cash_pct + cash_back, 4)
        position.realized_pnl_nav_pct = round(
            position.realized_pnl_nav_pct + realized_nav, 4
        )

    def archive_if_closed(self, position: Position) -> None:
        if position.stage == PositionStage.CLOSED:
            self.state.closed_positions.append(position)
            del self.state.positions[position.symbol]
