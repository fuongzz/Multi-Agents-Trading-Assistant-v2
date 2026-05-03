"""StrategyPlaybook — quyết định "với signal này, ở stage này, làm gì?"

Tách biệt hoàn toàn khỏi:
  - Detector (chỉ trả lời "có gì xảy ra?")
  - Risk gate (kiểm tra hành động có an toàn không)
  - Execution (mô phỏng fill)

Một Playbook thuần là pure function:
    on_signal(position, signal, market_ctx) -> Action

Hệ thống có thể có nhiều Playbook, mỗi cái phù hợp với một loại setup/regime.
Phase 1 ship 2 cái:
  - SingleEntryPlaybook: vào 1 lần, behavior backward-compatible với engine cũ.
  - ScaleInReversalPlaybook: 30/30/30 cho reversal setups.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from multiagents_trading_assistant.backtest.lifecycle import (
    LegReason,
    Position,
    PositionStage,
    Signal,
)


# ─── Action types ────────────────────────────────────────────────────────────


class ActionType(str, Enum):
    HOLD       = "hold"        # không làm gì
    OPEN_PROBE = "open_probe"  # mở probe (NO_POSITION → PROBE)
    OPEN_FULL  = "open_full"   # mở full ngay (NO_POSITION → FULL) — single entry
    ADD        = "add"         # add tiếp một leg (PROBE → ADDED_STRENGTH, ADDED_STRENGTH → FULL)
    REDUCE     = "reduce"      # giảm một phần
    EXIT       = "exit"        # thoát hết
    UPDATE_SL  = "update_sl"   # chỉ nâng SL


@dataclass
class Action:
    """Quyết định của playbook — playbook không tự execute, chỉ trả ý định."""

    type: ActionType
    qty_pct: float = 0.0          # % NAV (positive=buy, negative=sell). 0 với HOLD/UPDATE_SL.
    new_sl: float = 0.0           # SL mới (có thể = SL cũ nếu không đổi)
    new_stage: Optional[PositionStage] = None
    leg_reason: Optional[LegReason] = None
    note: str = ""                # giải thích quyết định
    extras: dict = field(default_factory=dict)


# ─── Market context (truyền vào playbook) ────────────────────────────────────


@dataclass
class MarketBar:
    """Snapshot 1 bar — playbook chỉ thấy thông tin tới bar hiện tại (no look-ahead)."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    atr: float = 0.0
    ma20: float = 0.0
    ma50: float = 0.0
    rsi: float = 0.0
    swing_low_recent: float = 0.0   # swing low gần nhất (5-7 bar)
    is_higher_high: bool = False
    is_higher_low: bool = False
    bullish_close: bool = False     # close > open
    volume_vs_ma20: float = 1.0     # volume / volume_ma20


# ─── Playbook ABC ────────────────────────────────────────────────────────────


class StrategyPlaybook(ABC):
    """Pure function: (position, signal_or_none, market_bar) -> Action."""

    name: str = "abstract"

    @abstractmethod
    def on_bar(
        self,
        position: Position,
        signal: Optional[Signal],
        bar: MarketBar,
    ) -> Action:
        """Gọi mỗi bar.

        - signal != None  → có setup mới phát hiện ở bar này.
        - signal == None  → chỉ là routine check vị thế hiện hữu.
        """

    def is_applicable(self, signal: Signal) -> bool:
        """Playbook có nhận signal này không? (router dùng để chọn playbook)."""
        return True


# ─── Helper: check exit conditions chung ─────────────────────────────────────


def _check_sl_hit(position: Position, bar: MarketBar) -> bool:
    """SL bị chạm trong bar hiện tại."""
    return position.is_open and position.current_sl > 0 and bar.low <= position.current_sl


def _check_invalidation(position: Position, bar: MarketBar) -> bool:
    """Tín hiệu invalidation: 2 nến đỏ liên tiếp + mất MA20 + thủng swing low."""
    if not position.is_open:
        return False
    # Đơn giản hóa: close < swing low gần nhất
    if bar.swing_low_recent > 0 and bar.close < bar.swing_low_recent:
        return True
    return False


# ─── 1. SingleEntryPlaybook ──────────────────────────────────────────────────


class SingleEntryPlaybook(StrategyPlaybook):
    """Vào 1 lần, exit 1 lần — wrap behavior cũ của engine.

    NO_POSITION + signal → OPEN_FULL (intended_total_pct theo confluence)
    FULL + SL hit         → EXIT
    FULL + invalidation   → EXIT
    """

    name = "single_entry"

    def __init__(self, sizing_map: Optional[dict[str, float]] = None):
        # confluence range → % NAV
        self.sizing_map = sizing_map or {
            "high":   5.0,   # ≥70
            "medium": 3.0,   # 55-69
            "low":    2.0,   # <55
        }

    def _size_for(self, confluence: float) -> float:
        if confluence >= 70:
            return self.sizing_map["high"]
        if confluence >= 55:
            return self.sizing_map["medium"]
        return self.sizing_map["low"]

    def on_bar(self, position, signal, bar):
        # Exit checks first
        if position.is_open:
            position.update_peak(bar.close)
            if _check_sl_hit(position, bar):
                return Action(
                    type=ActionType.EXIT,
                    qty_pct=-position.total_nav_pct,
                    new_stage=PositionStage.CLOSED,
                    leg_reason=LegReason.EXIT_SL,
                    note=f"SL hit at {position.current_sl}",
                )
            gain_pct = position.unrealized_pnl_pct(position.peak_close)
            if gain_pct >= 5.0 and _check_invalidation(position, bar):
                return Action(
                    type=ActionType.EXIT,
                    qty_pct=-position.total_nav_pct,
                    new_stage=PositionStage.CLOSED,
                    leg_reason=LegReason.EXIT_REVERSAL,
                    note="Reversal/invalidation after profitable move",
                )
            new_sl = position.current_sl
            if gain_pct >= 5.0 and bar.swing_low_recent > 0:
                buffer = 0.5 * bar.atr if bar.atr > 0 else position.avg_price * 0.01
                new_sl = max(new_sl, round(bar.swing_low_recent - buffer, 2))
            return Action(type=ActionType.HOLD, new_sl=new_sl)

        # NO_POSITION: chỉ vào khi có signal
        if signal is None:
            return Action(type=ActionType.HOLD)

        size = round(self._size_for(signal.confluence_score) * max(0.0, signal.risk_multiplier), 2)
        if size <= 0:
            return Action(type=ActionType.HOLD, note="Regime risk multiplier blocks new buy")
        return Action(
            type=ActionType.OPEN_FULL,
            qty_pct=size,
            new_sl=signal.suggested_sl,
            new_stage=PositionStage.FULL,
            leg_reason=LegReason.ADD_TREND,
            note=f"Single entry {signal.setup_type} conf={signal.confluence_score} regime={signal.regime}",
            extras={"intended_total_pct": size},
        )


# ─── 2. ScaleInReversalPlaybook ──────────────────────────────────────────────


REVERSAL_SETUPS = {
    "DOUBLE_BOTTOM",
    "RSI_BOUNCE",
    "HAMMER",
    "BULLISH_ENGULFING",
    "PIN_BAR",
}


class ScaleInReversalPlaybook(StrategyPlaybook):
    """30/30/30 scale-in cho reversal setups.

    NO_POSITION + reversal signal:
        → PROBE 30% intended, SL dưới đáy gần nhất.

    PROBE:
        - Lãi ≥ 3% + bullish close + volume xác nhận → ADD 30%, nâng SL lên hòa vốn.
        - Mất swing low / SL hit → EXIT all.

    ADDED_STRENGTH (60%):
        - HH + HL + breakout/retest → ADD 30% nốt → FULL.
        - Phân phối / mất MA20 → REDUCE half (về PROBE) hoặc EXIT.

    FULL (90%):
        - SL hit / reversal → EXIT.
        - Momentum yếu → REDUCE half.
        - Otherwise: trail SL theo swing low.
    """

    name = "scale_in_reversal"

    def __init__(
        self,
        intended_total_pct: float = 9.0,   # 9% NAV nếu vào full
        probe_fraction: float = 0.33,
        confirm_pnl_pct: float = 3.0,      # cần lãi 3% mới add
        full_pnl_pct: float = 5.0,         # cần lãi 5% + HH/HL mới add nốt
    ):
        self.intended_total_pct = intended_total_pct
        self.probe_fraction = probe_fraction
        self.confirm_pnl_pct = confirm_pnl_pct
        self.full_pnl_pct = full_pnl_pct

    def is_applicable(self, signal: Signal) -> bool:
        return signal.setup_type in REVERSAL_SETUPS

    # ─── State handlers ──────────────────────────────────────────────────────

    def _on_no_position(self, signal: Optional[Signal], bar: MarketBar) -> Action:
        if signal is None or not self.is_applicable(signal):
            return Action(type=ActionType.HOLD)
        # Probe size = intended × fraction
        probe_size = round(self.intended_total_pct * self.probe_fraction * max(0.0, signal.risk_multiplier), 2)
        if probe_size <= 0:
            return Action(type=ActionType.HOLD, note="Regime risk multiplier blocks probe")
        return Action(
            type=ActionType.OPEN_PROBE,
            qty_pct=probe_size,
            new_sl=signal.suggested_sl,
            new_stage=PositionStage.PROBE,
            leg_reason=LegReason.PROBE,
            note=f"Probe 30% on {signal.setup_type} conf={signal.confluence_score} regime={signal.regime}",
            extras={"intended_total_pct": round(self.intended_total_pct * max(0.0, signal.risk_multiplier), 2)},
        )

    def _on_probe(self, position: Position, bar: MarketBar) -> Action:
        # Exit if SL hit
        if _check_sl_hit(position, bar):
            return Action(
                type=ActionType.EXIT,
                qty_pct=-position.total_nav_pct,
                new_stage=PositionStage.CLOSED,
                leg_reason=LegReason.EXIT_SL,
                note="Probe SL hit",
            )
        # Exit if invalidation (mất swing low ngay khi đang probe)
        if _check_invalidation(position, bar):
            return Action(
                type=ActionType.EXIT,
                qty_pct=-position.total_nav_pct,
                new_stage=PositionStage.CLOSED,
                leg_reason=LegReason.EXIT_REVERSAL,
                note="Probe invalidated — broke swing low",
            )

        pnl = position.unrealized_pnl_pct(bar.close)

        # Add điều kiện: lãi ≥ confirm_pnl + bullish close + volume xác nhận
        if (
            pnl >= self.confirm_pnl_pct
            and bar.bullish_close
            and bar.volume_vs_ma20 >= 1.0
        ):
            add_size = round(self.intended_total_pct * self.probe_fraction, 2)
            # Nâng SL lên gần hòa vốn (avg price - 0.5%)
            breakeven_sl = round(position.avg_price * 0.995, 2)
            new_sl = max(position.current_sl, breakeven_sl)
            return Action(
                type=ActionType.ADD,
                qty_pct=add_size,
                new_sl=new_sl,
                new_stage=PositionStage.ADDED_STRENGTH,
                leg_reason=LegReason.ADD_STRENGTH,
                note=f"Confirm add (+{pnl:.1f}%, vol {bar.volume_vs_ma20:.1f}x)",
            )

        return Action(type=ActionType.HOLD, new_sl=position.current_sl)

    def _on_added_strength(self, position: Position, bar: MarketBar) -> Action:
        if _check_sl_hit(position, bar):
            return Action(
                type=ActionType.EXIT,
                qty_pct=-position.total_nav_pct,
                new_stage=PositionStage.CLOSED,
                leg_reason=LegReason.EXIT_SL,
                note="ADDED_STRENGTH SL hit",
            )

        pnl = position.unrealized_pnl_pct(bar.close)

        # Reduce nếu mất MA20
        if bar.ma20 > 0 and bar.close < bar.ma20:
            return Action(
                type=ActionType.REDUCE,
                qty_pct=-round(position.total_nav_pct * 0.5, 2),
                new_stage=PositionStage.REDUCING,
                leg_reason=LegReason.REDUCE_HALF,
                note="Lost MA20 — reduce half",
            )

        # Add nốt nếu structure xác nhận
        if (
            pnl >= self.full_pnl_pct
            and bar.is_higher_high
            and bar.is_higher_low
            and bar.bullish_close
        ):
            remaining = self.intended_total_pct - position.total_nav_pct
            if remaining > 0.5:
                # Trail SL theo swing low
                new_sl = max(position.current_sl, bar.swing_low_recent)
                return Action(
                    type=ActionType.ADD,
                    qty_pct=round(remaining, 2),
                    new_sl=new_sl,
                    new_stage=PositionStage.FULL,
                    leg_reason=LegReason.ADD_TREND,
                    note=f"Full add (HH+HL, +{pnl:.1f}%)",
                )

        # Trail SL nhẹ
        new_sl = position.current_sl
        if bar.swing_low_recent > 0:
            new_sl = max(new_sl, bar.swing_low_recent)
        return Action(type=ActionType.HOLD, new_sl=new_sl)

    def _on_full(self, position: Position, bar: MarketBar) -> Action:
        if _check_sl_hit(position, bar):
            return Action(
                type=ActionType.EXIT,
                qty_pct=-position.total_nav_pct,
                new_stage=PositionStage.CLOSED,
                leg_reason=LegReason.EXIT_SL,
                note="FULL SL hit (trailing)",
            )

        # Mất MA20 + 2 nến đỏ → exit
        if bar.ma20 > 0 and bar.close < bar.ma20 and not bar.bullish_close:
            return Action(
                type=ActionType.REDUCE,
                qty_pct=-round(position.total_nav_pct * 0.5, 2),
                new_stage=PositionStage.REDUCING,
                leg_reason=LegReason.REDUCE_HALF,
                note="FULL distribution — reduce half",
            )

        # Trail SL theo swing low
        new_sl = position.current_sl
        if bar.swing_low_recent > 0:
            new_sl = max(new_sl, bar.swing_low_recent)
        return Action(type=ActionType.HOLD, new_sl=new_sl)

    def _on_reducing(self, position: Position, bar: MarketBar) -> Action:
        # Đơn giản: ở REDUCING → tiếp tục thoát hết nếu có thêm yếu.
        if _check_sl_hit(position, bar):
            return Action(
                type=ActionType.EXIT,
                qty_pct=-position.total_nav_pct,
                new_stage=PositionStage.CLOSED,
                leg_reason=LegReason.EXIT_SL,
                note="REDUCING SL hit",
            )
        if bar.ma20 > 0 and bar.close < bar.ma20 * 0.98:
            return Action(
                type=ActionType.EXIT,
                qty_pct=-position.total_nav_pct,
                new_stage=PositionStage.CLOSED,
                leg_reason=LegReason.EXIT_ALL,
                note="REDUCING further weakness — full exit",
            )
        return Action(type=ActionType.HOLD, new_sl=position.current_sl)

    # ─── Dispatch ────────────────────────────────────────────────────────────

    def on_bar(self, position, signal, bar):
        position.update_peak(bar.close)

        if position.stage == PositionStage.NO_POSITION:
            return self._on_no_position(signal, bar)
        if position.stage == PositionStage.PROBE:
            return self._on_probe(position, bar)
        if position.stage == PositionStage.ADDED_STRENGTH:
            return self._on_added_strength(position, bar)
        if position.stage == PositionStage.FULL:
            return self._on_full(position, bar)
        if position.stage == PositionStage.REDUCING:
            return self._on_reducing(position, bar)
        return Action(type=ActionType.HOLD)


# ─── Router ──────────────────────────────────────────────────────────────────


class TrendFollowingPlaybook(StrategyPlaybook):
    """Uptrend playbook: ride winners, pyramid only when profitable."""

    name = "trend_following"

    def __init__(self, max_position_pct: float = 25.0):
        self.max_position_pct = max_position_pct

    def is_applicable(self, signal: Signal) -> bool:
        return signal.playbook_hint == self.name

    def on_bar(self, position: Position, signal: Optional[Signal], bar: MarketBar) -> Action:
        if position.is_open:
            position.update_peak(bar.close)
            pnl = position.unrealized_pnl_pct(bar.close)
            if _check_sl_hit(position, bar):
                return Action(ActionType.EXIT, -position.total_nav_pct, new_stage=PositionStage.CLOSED,
                              leg_reason=LegReason.EXIT_SL, note="Trend SL hit")
            if bar.ma20 > 0 and bar.close < bar.ma20:
                return Action(ActionType.EXIT, -position.total_nav_pct, new_stage=PositionStage.CLOSED,
                              leg_reason=LegReason.EXIT_REVERSAL, note="Trend failed: close below MA20")

            sold_pct = sum(-l.qty_pct for l in position.legs if l.is_sell)
            if pnl >= 20.0 and sold_pct < position.total_invested_pct * 0.55:
                return Action(ActionType.REDUCE, -round(position.total_nav_pct * 0.30, 2),
                              new_stage=PositionStage.REDUCING, leg_reason=LegReason.REDUCE_PARTIAL,
                              note=f"Trend partial TP +20% ({pnl:.1f}%)")
            if pnl >= 10.0 and sold_pct < position.total_invested_pct * 0.25:
                return Action(ActionType.REDUCE, -round(position.total_nav_pct * 0.30, 2),
                              new_stage=PositionStage.REDUCING, leg_reason=LegReason.REDUCE_PARTIAL,
                              note=f"Trend partial TP +10% ({pnl:.1f}%)")

            if (
                pnl >= 5.0
                and position.total_nav_pct < self.max_position_pct
                and bar.is_higher_high
                and bar.is_higher_low
                and bar.bullish_close
            ):
                add = min(7.5, self.max_position_pct - position.total_nav_pct)
                if add >= 2.0:
                    new_sl = max(position.current_sl, position.avg_price * 0.995, bar.swing_low_recent)
                    return Action(ActionType.ADD, round(add, 2), new_sl=new_sl,
                                  new_stage=PositionStage.FULL, leg_reason=LegReason.ADD_TREND,
                                  note=f"Trend pyramid winner +{pnl:.1f}%")

            new_sl = position.current_sl
            if pnl >= 5.0 and bar.swing_low_recent > 0:
                buffer = 0.5 * bar.atr if bar.atr > 0 else position.avg_price * 0.01
                new_sl = max(new_sl, round(bar.swing_low_recent - buffer, 2))
            return Action(ActionType.HOLD, new_sl=new_sl)

        if signal is None:
            return Action(ActionType.HOLD)
        setup = signal.setup_type.upper()
        if setup in {"SPRING", "DOUBLE_BOTTOM", "RSI_BOUNCE", "HAMMER", "BULLISH_ENGULFING", "PIN_BAR"}:
            base = 7.5
        elif setup in {"BREAKOUT", "MOMENTUM_SURGE", "KUMO_BREAKOUT", "TK_CROSS"}:
            base = 15.0
        else:
            base = 12.0
        size = round(min(self.max_position_pct, base * max(0.0, signal.risk_multiplier)), 2)
        if size <= 0:
            return Action(ActionType.HOLD, note="Trend regime blocks entry")
        return Action(ActionType.OPEN_FULL, size, new_sl=signal.suggested_sl,
                      new_stage=PositionStage.FULL, leg_reason=LegReason.ADD_TREND,
                      note=f"Trend entry {signal.setup_type} conf={signal.confluence_score} regime={signal.regime}",
                      extras={"intended_total_pct": min(self.max_position_pct, size)})


class CoreTrendPlaybook(StrategyPlaybook):
    """Benchmark-aware core sleeve for established uptrends.

    Tactical playbooks try to time setups. This one exists to avoid the worst
    failure mode in a long bull move: sitting mostly in cash while buy-and-hold
    compounds. It buys a large core, lets winners run, and only exits when the
    intermediate trend breaks.
    """

    name = "core_trend"

    def __init__(
        self,
        core_pct: float = 95.0,
        hard_stop_pct: float = 0.055,
        peak_drawdown_exit_pct: float = 0.22,
    ):
        self.core_pct = core_pct
        self.hard_stop_pct = hard_stop_pct
        self.peak_drawdown_exit_pct = peak_drawdown_exit_pct

    def is_applicable(self, signal: Signal) -> bool:
        return signal.playbook_hint == self.name or signal.setup_type.upper() == "CORE_TREND"

    def on_bar(self, position: Position, signal: Optional[Signal], bar: MarketBar) -> Action:
        if position.is_open:
            position.update_peak(bar.close)
            pnl = position.unrealized_pnl_pct(bar.close)
            drawdown_from_peak = (
                (position.peak_close - bar.close) / position.peak_close
                if position.peak_close > 0
                else 0.0
            )
            if _check_sl_hit(position, bar):
                return Action(ActionType.EXIT, -position.total_nav_pct, new_stage=PositionStage.CLOSED,
                              leg_reason=LegReason.EXIT_SL, note="Core hard stop hit")
            if (
                bar.ma50 > 0
                and bar.ma20 > 0
                and bar.close < bar.ma50
                and bar.ma20 < bar.ma50
            ):
                return Action(ActionType.EXIT, -position.total_nav_pct, new_stage=PositionStage.CLOSED,
                              leg_reason=LegReason.EXIT_REVERSAL, note="Core trend break: close < MA50 and MA20 < MA50")
            if pnl > 20.0 and drawdown_from_peak >= self.peak_drawdown_exit_pct:
                return Action(ActionType.EXIT, -position.total_nav_pct, new_stage=PositionStage.CLOSED,
                              leg_reason=LegReason.EXIT_REVERSAL, note=f"Core peak drawdown {drawdown_from_peak * 100:.1f}%")

            new_sl = position.current_sl
            if pnl >= 12.0 and bar.ma50 > 0:
                new_sl = max(new_sl, round(bar.ma50 * 0.94, 2))
            return Action(ActionType.HOLD, new_sl=new_sl)

        if signal is None:
            return Action(ActionType.HOLD)
        size = round(self.core_pct * max(0.0, signal.risk_multiplier), 2)
        if size <= 0:
            return Action(ActionType.HOLD, note="Core trend regime blocks entry")
        hard_sl = round(max(signal.suggested_sl, bar.close * (1.0 - self.hard_stop_pct)), 2)
        if hard_sl <= 0:
            hard_sl = round(bar.close * (1.0 - self.hard_stop_pct), 2)
        return Action(ActionType.OPEN_FULL, size, new_sl=hard_sl,
                      new_stage=PositionStage.FULL, leg_reason=LegReason.ADD_TREND,
                      note=f"Core trend entry conf={signal.confluence_score} regime={signal.regime}",
                      extras={"intended_total_pct": size, "core": True})


class RangeReversalPlaybook(StrategyPlaybook):
    """Sideway playbook: buy low, sell fast, do not dream of trend."""

    name = "range_reversal"

    def __init__(self, entry_pct: float = 10.0, stop_pct: float = 0.03, take_profit_pct: float = 0.05):
        self.entry_pct = entry_pct
        self.stop_pct = stop_pct
        self.take_profit_pct = take_profit_pct

    def is_applicable(self, signal: Signal) -> bool:
        return signal.playbook_hint == self.name

    def on_bar(self, position: Position, signal: Optional[Signal], bar: MarketBar) -> Action:
        if position.is_open:
            pnl = position.unrealized_pnl_pct(bar.close)
            if _check_sl_hit(position, bar) or pnl <= -(self.stop_pct * 100):
                return Action(ActionType.EXIT, -position.total_nav_pct, new_stage=PositionStage.CLOSED,
                              leg_reason=LegReason.EXIT_SL, note="Range stop hit")
            if pnl >= self.take_profit_pct * 100:
                return Action(ActionType.EXIT, -position.total_nav_pct, new_stage=PositionStage.CLOSED,
                              leg_reason=LegReason.EXIT_ALL, note=f"Range TP {pnl:.1f}%")
            if bar.ma20 > 0 and bar.close < bar.ma20 * 0.98:
                return Action(ActionType.EXIT, -position.total_nav_pct, new_stage=PositionStage.CLOSED,
                              leg_reason=LegReason.EXIT_REVERSAL, note="Range failed below MA20")
            return Action(ActionType.HOLD, new_sl=position.current_sl)

        if signal is None:
            return Action(ActionType.HOLD)
        size = round(self.entry_pct * max(0.0, signal.risk_multiplier), 2)
        if size <= 0:
            return Action(ActionType.HOLD, note="Range regime blocks entry")
        tight_sl = max(signal.suggested_sl, signal.entry_zone_low * (1 - self.stop_pct))
        return Action(ActionType.OPEN_FULL, size, new_sl=round(tight_sl, 2),
                      new_stage=PositionStage.FULL, leg_reason=LegReason.PROBE,
                      note=f"Range reversal {signal.setup_type} conf={signal.confluence_score} regime={signal.regime}",
                      extras={"intended_total_pct": size})


class ProbeOnlyBearPlaybook(StrategyPlaybook):
    """Downtrend playbook: survive first, probe small, take quick profit."""

    name = "probe_only_bear"

    def __init__(self, entry_pct: float = 7.5, stop_pct: float = 0.02, take_profit_pct: float = 0.02):
        self.entry_pct = entry_pct
        self.stop_pct = stop_pct
        self.take_profit_pct = take_profit_pct

    def is_applicable(self, signal: Signal) -> bool:
        return signal.playbook_hint == self.name

    def on_bar(self, position: Position, signal: Optional[Signal], bar: MarketBar) -> Action:
        if position.is_open:
            pnl = position.unrealized_pnl_pct(bar.close)
            if _check_sl_hit(position, bar) or pnl <= -(self.stop_pct * 100):
                return Action(ActionType.EXIT, -position.total_nav_pct, new_stage=PositionStage.CLOSED,
                              leg_reason=LegReason.EXIT_SL, note="Bear probe stop")
            if pnl >= self.take_profit_pct * 100:
                return Action(ActionType.EXIT, -position.total_nav_pct, new_stage=PositionStage.CLOSED,
                              leg_reason=LegReason.EXIT_ALL, note=f"Bear quick TP {pnl:.1f}%")
            return Action(ActionType.HOLD, new_sl=position.current_sl)

        if signal is None:
            return Action(ActionType.HOLD)
        size = round(self.entry_pct * max(0.0, signal.risk_multiplier), 2)
        if size <= 0:
            return Action(ActionType.HOLD, note="Bear regime blocks entry")
        tight_sl = max(signal.suggested_sl, signal.entry_zone_low * (1 - self.stop_pct))
        return Action(ActionType.OPEN_PROBE, size, new_sl=round(tight_sl, 2),
                      new_stage=PositionStage.PROBE, leg_reason=LegReason.PROBE,
                      note=f"Bear probe {signal.setup_type} conf={signal.confluence_score} regime={signal.regime}",
                      extras={"intended_total_pct": size})


class DefensiveExitPlaybook(StrategyPlaybook):
    """Distribution playbook: no new buys, reduce/exit existing positions."""

    name = "defensive_exit"

    def is_applicable(self, signal: Signal) -> bool:
        return signal.playbook_hint == self.name

    def on_bar(self, position: Position, signal: Optional[Signal], bar: MarketBar) -> Action:
        if not position.is_open:
            return Action(ActionType.HOLD, note="Defensive: no new buy")
        if _check_sl_hit(position, bar):
            return Action(ActionType.EXIT, -position.total_nav_pct, new_stage=PositionStage.CLOSED,
                          leg_reason=LegReason.EXIT_SL, note="Defensive SL hit")
        if bar.ma20 > 0 and bar.close < bar.ma20:
            return Action(ActionType.EXIT, -position.total_nav_pct, new_stage=PositionStage.CLOSED,
                          leg_reason=LegReason.EXIT_ALL, note="Defensive structure break")
        if bar.volume_vs_ma20 >= 1.8 and not bar.bullish_close:
            return Action(ActionType.REDUCE, -round(position.total_nav_pct * 0.20, 2),
                          new_stage=PositionStage.REDUCING, leg_reason=LegReason.REDUCE_PARTIAL,
                          note="Defensive distribution reduce 20%")
        return Action(ActionType.HOLD, new_sl=position.current_sl)


class PlaybookRouter:
    """Chọn playbook phù hợp với signal. Khi position đã mở, vẫn dùng playbook
    đã chọn lúc đầu (lưu trong Position.extras tương lai — Phase 1: 1 playbook/symbol)."""

    def __init__(self, playbooks: list[StrategyPlaybook], default: StrategyPlaybook):
        self.playbooks = playbooks
        self.default = default
        self._by_name = {pb.name: pb for pb in [*playbooks, default]}

    def by_name(self, name: str | None) -> StrategyPlaybook | None:
        if not name:
            return None
        return self._by_name.get(name)

    def select(self, signal: Optional[Signal]) -> StrategyPlaybook:
        if signal is None:
            return self.default
        hinted = self.by_name(signal.playbook_hint)
        if hinted is not None:
            return hinted
        for pb in self.playbooks:
            if pb.is_applicable(signal):
                return pb
        return self.default
