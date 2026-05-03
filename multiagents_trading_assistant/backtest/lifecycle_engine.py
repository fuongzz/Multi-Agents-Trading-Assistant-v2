"""Walk-forward backtest engine using Position Lifecycle abstractions.

Khác với engine.py cũ (1 trade tại 1 thời điểm), engine này:
  - Cho phép 1 mã có Position nhiều legs (probe → add → full).
  - Tách biệt 4 lớp: Signal detection / Playbook / Portfolio / Execution.
  - Không có concept "Trade" — chỉ có Position và Leg.

Pipeline mỗi bar i:
  1. Compute MarketBar từ window df[:i+1].
  2. Detect signal (nếu có) — dùng detect_* hiện có hoặc Signal được inject.
  3. Lấy position hiện tại (có thể là NO_POSITION).
  4. Hỏi playbook: action gì?
  5. Validate với portfolio (cash, exposure, risk).
  6. Simulate fill ở bar i+1.
  7. Commit leg vào position + cập nhật portfolio.

Zero look-ahead: tại bar i chỉ thấy df[:i+1]. Fill ở open[i+1].
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from multiagents_trading_assistant.backtest.execution import (
    ExecutionConfig,
    can_sell_today,
    simulate_fill,
)
from multiagents_trading_assistant.backtest.lifecycle import (
    Leg,
    LegReason,
    Position,
    PositionStage,
    Signal,
    is_valid_transition,
)
from multiagents_trading_assistant.backtest.playbook import (
    Action,
    ActionType,
    MarketBar,
    PlaybookRouter,
    StrategyPlaybook,
)
from multiagents_trading_assistant.backtest.portfolio import (
    PortfolioConstraints,
    PortfolioEngine,
)
from multiagents_trading_assistant.setup_scoring import (
    COMPRESSION_SETUPS,
    PULLBACK_SETUPS,
    REVERSAL_SETUPS,
)


SignalDetector = Callable[[pd.DataFrame, int], Optional[Signal]]


# ─── Helpers: compute MarketBar from raw OHLCV window ────────────────────────


def _compute_market_bar(df: pd.DataFrame, i: int, symbol: str) -> MarketBar:
    """Tính MarketBar tại bar i từ window df[:i+1]."""
    row = df.iloc[i]
    win = df.iloc[: i + 1]
    closes = win["close"].astype(float).values
    highs = win["high"].astype(float).values
    lows = win["low"].astype(float).values
    vols = win["volume"].astype(float).values

    ma20 = float(closes[-20:].mean()) if len(closes) >= 20 else 0.0
    ma50 = float(closes[-50:].mean()) if len(closes) >= 50 else 0.0
    vol_ma20 = float(vols[-20:].mean()) if len(vols) >= 20 else float(vols.mean())

    # ATR(14)
    atr = 0.0
    if len(win) >= 15:
        h = highs[-15:]
        l = lows[-15:]
        c = closes[-16:-1] if len(closes) >= 16 else closes[-15:]
        c = np.concatenate([[c[0]], c]) if len(c) < len(h) else c
        tr = np.maximum.reduce([h - l, np.abs(h - c[: len(h)]), np.abs(l - c[: len(h)])])
        atr = float(np.mean(tr))

    # RSI(14) — đơn giản
    rsi = 50.0
    if len(closes) >= 15:
        deltas = np.diff(closes[-15:])
        gains = np.where(deltas > 0, deltas, 0).mean()
        losses = np.where(deltas < 0, -deltas, 0).mean()
        if losses > 0:
            rs = gains / losses
            rsi = 100 - 100 / (1 + rs)
        elif gains > 0:
            rsi = 100.0

    # Swing low gần nhất (5 bar)
    swing_low = float(np.min(lows[-5:])) if len(lows) >= 5 else float(lows.min())

    # HH/HL (so với 5 bar trước)
    is_hh = False
    is_hl = False
    if len(win) >= 6:
        is_hh = highs[-1] > float(np.max(highs[-6:-1]))
        is_hl = lows[-1] > float(np.min(lows[-6:-1]))

    return MarketBar(
        date=str(row["date"])[:10],
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row["volume"]),
        atr=atr,
        ma20=ma20,
        ma50=ma50,
        rsi=rsi,
        swing_low_recent=swing_low,
        is_higher_high=is_hh,
        is_higher_low=is_hl,
        bullish_close=float(row["close"]) > float(row["open"]),
        volume_vs_ma20=float(row["volume"]) / vol_ma20 if vol_ma20 > 0 else 1.0,
    )


# ─── Engine ─────────────────────────────────────────────────────────────────


class LifecycleEngine:
    """Walk-forward backtest single-symbol hoặc multi-symbol."""

    def __init__(
        self,
        router: PlaybookRouter,
        portfolio: Optional[PortfolioEngine] = None,
        execution_cfg: Optional[ExecutionConfig] = None,
        lookback: int = 60,
    ):
        self.router = router
        self.portfolio = portfolio or PortfolioEngine()
        self.execution_cfg = execution_cfg or ExecutionConfig()
        self.lookback = lookback
        self.audit_log: list[dict] = []

    # ─── Per-bar processing ─────────────────────────────────────────────────

    def _process_bar(
        self,
        symbol: str,
        df: pd.DataFrame,
        i: int,
        signal: Optional[Signal],
    ) -> None:
        bar = _compute_market_bar(df, i, symbol)
        position = self.portfolio.get_or_create(symbol)

        # Khi position đã mở, luôn dùng lại playbook đã mở position đó.
        # Signal mới trong lúc đang có vị thế chỉ là context phụ, không đổi strategy owner.
        playbook = self.router.by_name(position.playbook_name) if position.is_open else None
        if playbook is None:
            playbook = self.router.select(signal)

        action = playbook.on_bar(position, signal, bar)

        # Validate transition
        if action.new_stage and not is_valid_transition(position.stage, action.new_stage):
            self._log(symbol, bar, action, fill=None, status="invalid_transition")
            return

        # ─── HOLD / UPDATE_SL: không fill, có thể nâng SL ──────────────────
        if action.type in (ActionType.HOLD, ActionType.UPDATE_SL):
            if action.new_sl > 0:
                position.update_sl(action.new_sl)
            position.update_peak(bar.close)
            return

        # ─── EXIT: fill ngay tại close của bar hiện tại nếu SL hit ──────────
        # (SL hit là intra-bar, dùng SL price làm fill — tệ hơn close).
        # Khác với entry, exit do SL không chờ bar kế.
        if action.type == ActionType.EXIT and action.leg_reason == LegReason.EXIT_SL:
            sellable = self._settled_nav_pct(position, i)
            if sellable <= 0:
                # T+2 chưa cho bán — giữ position
                self._log(symbol, bar, action, fill=None, status="t+2_blocked")
                return
            qty_to_sell = min(abs(action.qty_pct), sellable)
            # Gap-down: nếu open đã dưới SL thì thoát tại open; ngược lại thoát tại SL.
            fill_price = bar.open if bar.open < position.current_sl else position.current_sl
            self._commit_sell(symbol, position, qty_to_sell, fill_price, bar.date, action, i)
            self._log(symbol, bar, action, fill=fill_price, status="sl_filled")
            return

        # ─── Các action khác cần fill ở bar kế (ATO) ────────────────────────
        if i + 1 >= len(df):
            self._log(symbol, bar, action, fill=None, status="no_next_bar")
            return
        next_bar = df.iloc[i + 1]
        side = "buy" if action.qty_pct > 0 else "sell"
        fill = simulate_fill(
            side=side,
            next_bar_open=float(next_bar["open"]),
            next_bar_high=float(next_bar["high"]),
            next_bar_low=float(next_bar["low"]),
            prev_close=bar.close,
            cfg=self.execution_cfg,
        )
        if not fill.filled:
            self._log(symbol, bar, action, fill=None, status=f"reject:{fill.reason}")
            return

        if action.qty_pct > 0 and not self._buy_fill_in_entry_zone(fill.price, signal):
            self._log(symbol, bar, action, fill=fill.price, status="reject:outside_entry_zone")
            return

        # ─── Validate với portfolio rồi commit ──────────────────────────────
        if action.type in (ActionType.OPEN_PROBE, ActionType.OPEN_FULL):
            ok, why = self.portfolio.can_open_new(
                symbol,
                action.qty_pct,
                entry_price=fill.price,
                stop_loss=action.new_sl,
            )
            if not ok:
                self._log(symbol, bar, action, fill=fill.price, status=f"portfolio_reject:{why}")
                return
            position.intended_total_pct = action.extras.get(
                "intended_total_pct", action.qty_pct
            )
            position.entry_atr = bar.atr
            position.open_date = next_bar["date"] if hasattr(next_bar["date"], "strftime") else str(next_bar["date"])[:10]
            position.playbook_name = playbook.name
            self._commit_buy(symbol, position, action.qty_pct, fill.price, position.open_date, action, i + 1)

        elif action.type == ActionType.ADD:
            ok, why = self.portfolio.can_add(
                symbol,
                action.qty_pct,
                position,
                entry_price=fill.price,
                stop_loss=action.new_sl or position.current_sl,
            )
            if not ok:
                self._log(symbol, bar, action, fill=fill.price, status=f"portfolio_reject:{why}")
                return
            self._commit_buy(symbol, position, action.qty_pct, fill.price, str(next_bar["date"])[:10], action, i + 1)

        elif action.type in (ActionType.REDUCE, ActionType.EXIT):
            sellable = self._settled_nav_pct(position, i + 1)
            if sellable <= 0:
                self._log(symbol, bar, action, fill=fill.price, status="t+2_blocked")
                return
            qty_to_sell = min(abs(action.qty_pct), sellable)
            self._commit_sell(symbol, position, qty_to_sell, fill.price, str(next_bar["date"])[:10], action, i + 1)

        position.update_peak(bar.close)

    # ─── Commit helpers ─────────────────────────────────────────────────────

    def _commit_buy(
        self,
        symbol: str,
        position: Position,
        qty_pct: float,
        price: float,
        date: str,
        action: Action,
        open_idx: Optional[int] = None,
    ) -> None:
        leg = Leg(
            date=date,
            price=price,
            qty_pct=qty_pct,
            reason=action.leg_reason or LegReason.PROBE,
            sl_at_leg=action.new_sl,
            bar_idx=open_idx if open_idx is not None else -1,
        )
        position.add_leg(leg)
        if action.new_stage == PositionStage.CLOSED and position.total_nav_pct > 1e-6:
            position.stage = PositionStage.REDUCING
        elif action.new_stage:
            position.stage = action.new_stage
        self.portfolio.commit_buy(position, qty_pct, price)
        self._log(symbol, None, action, fill=price, status="filled_buy", leg=leg)

    def _commit_sell(
        self,
        symbol: str,
        position: Position,
        qty_pct_abs: float,
        price: float,
        date: str,
        action: Action,
        bar_idx: int = -1,
    ) -> None:
        # qty_pct của leg là số âm (sell)
        leg = Leg(
            date=date,
            price=price,
            qty_pct=-qty_pct_abs,
            reason=action.leg_reason or LegReason.EXIT_ALL,
            sl_at_leg=position.current_sl,
            bar_idx=bar_idx,
        )
        position.add_leg(leg)
        if action.new_stage:
            position.stage = action.new_stage
        self.portfolio.commit_sell(position, qty_pct_abs, price)
        self._log(symbol, None, action, fill=price, status="filled_sell", leg=leg)
        self.portfolio.archive_if_closed(position)

    def _settled_nav_pct(self, position: Position, today_idx: int) -> float:
        """Return net sellable NAV pct using per-buy-leg T+N settlement."""
        settled_buys = sum(
            leg.qty_pct
            for leg in position.legs
            if leg.is_buy and leg.bar_idx >= 0 and can_sell_today(leg.bar_idx, today_idx, self.execution_cfg)
        )
        already_sold = sum(-leg.qty_pct for leg in position.legs if leg.is_sell)
        return round(max(0.0, settled_buys - already_sold), 4)

    @staticmethod
    def _buy_fill_in_entry_zone(fill_price: float, signal: Optional[Signal]) -> bool:
        if signal is None:
            return True
        low = float(signal.entry_zone_low or 0.0)
        high = float(signal.entry_zone_high or 0.0)
        if low <= 0 or high <= 0:
            return True
        if low <= fill_price <= high:
            return True
        setup = signal.setup_type.upper()
        if fill_price < low and setup in (PULLBACK_SETUPS | REVERSAL_SETUPS | COMPRESSION_SETUPS):
            # Pullback/range/reversal can accept a slightly better fill below
            # the proposed zone, but not a breakdown through the technical SL.
            return fill_price >= max(float(signal.suggested_sl or 0.0), low * 0.98)
        return False

    # ─── Audit ──────────────────────────────────────────────────────────────

    def _log(self, symbol, bar, action, fill, status, leg=None):
        self.audit_log.append({
            "symbol": symbol,
            "date": bar.date if bar else (leg.date if leg else "?"),
            "action": action.type.value,
            "qty_pct": action.qty_pct,
            "fill": fill,
            "stage_after": action.new_stage.value if action.new_stage else None,
            "status": status,
            "note": action.note,
        })

    # ─── Public API ─────────────────────────────────────────────────────────

    def run_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        signal_detector: SignalDetector,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> dict:
        """Chạy backtest 1 mã.

        Args:
          symbol: mã.
          df: OHLCV df, cột date/open/high/low/close/volume, sorted theo time tăng.
          signal_detector: callback (df, i) -> Optional[Signal]. Trả None nếu không có setup.
          from_date / to_date: chỉ phát signal trong range này. Lookback vẫn dùng full history.
        """
        df = df.reset_index(drop=True).copy()
        from_idx = 0 if from_date is None else int(
            df.index[df["date"].astype(str).str[:10] >= from_date][0]
        ) if (df["date"].astype(str).str[:10] >= from_date).any() else len(df)

        for i in range(self.lookback, len(df) - 1):
            d_i = str(df.iloc[i]["date"])[:10]
            if to_date and d_i > to_date:
                break
            # Chỉ phát signal trong [from_date, to_date]; ngoài range vẫn process bar
            # để cho exit hoạt động.
            pos = self.portfolio.state.positions.get(symbol)
            is_flat = pos is None or not pos.is_open
            allow_signal = is_flat and (i >= from_idx) and (not to_date or d_i <= to_date)
            sig = signal_detector(df, i) if allow_signal else None
            self._process_bar(symbol, df, i, sig)

        # Force-close vị thế còn mở ở bar cuối
        if symbol in self.portfolio.state.open_positions:
            pos = self.portfolio.state.positions[symbol]
            last = df.iloc[-1]
            self._commit_sell(
                symbol, pos, abs(pos.total_nav_pct), float(last["close"]),
                str(last["date"])[:10],
                Action(
                    type=ActionType.EXIT,
                    qty_pct=-pos.total_nav_pct,
                    leg_reason=LegReason.EXIT_ALL,
                    new_stage=PositionStage.CLOSED,
                    note="end_of_data force close",
                ),
            )

        return {
            "symbol": symbol,
            "closed_positions": [p for p in self.portfolio.state.closed_positions if p.symbol == symbol],
            "audit_log": self.audit_log,
            "portfolio_state": self.portfolio.state,
        }
