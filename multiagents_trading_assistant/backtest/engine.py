"""Walk-forward backtester — zero LLM, pure rule-based.

Triết lý exit (price-action based, không time-based):
  - Không có fixed TP. Để lệnh chạy miễn là price action còn tốt.
  - Exit khi: SL bị chạm | Reversal signal | Safety cap (120 bar)

Trailing SL — swing-low based:
  - Gain < 5%  : giữ nguyên SL gốc (tránh noise đầu lệnh)
  - Gain 5-10% : trail về swing low gần nhất (7 bar lookback) − 0.5×ATR
  - Gain 10-20%: swing low (12 bar lookback) − 0.5×ATR
  - Gain ≥ 20% : swing low (20 bar lookback) − 0.5×ATR
  → SL chỉ tăng, không bao giờ giảm.

Reversal detection (chỉ kích hoạt khi đang lãi ≥ 5%):
  1. MA20_BREAK  : 2 nến liên tiếp đóng cửa dưới MA20 + MA20 đang giảm
  2. DOUBLE_TOP  : 2 đỉnh cách nhau ≤ 2.5%, đỉnh 2 pull back qua midpoint (gain ≥ 8%)
  3. TREND_BREAK : 2 nến liên tiếp đóng dưới swing low gần nhất (gain ≥ 10%)

Safety cap:
  _SAFETY_CAP = 120 bar (~6 tháng giao dịch) — ngăn zombie position.

Nguyên tắc chống look-ahead bias:
  - Tại bar i, window = df.iloc[:i+1] — chỉ dữ liệu đã biết tính đến bar i.
  - Indicator + detect_* đều tính trên window này.
  - Entry: open của bar i+1 (ATO hôm sau) — không dùng giá đóng cửa tín hiệu.
  - Exit: check SL/Reversal theo lo/hi/cl của từng bar kế tiếp, bar-by-bar.
  - Không nhìn trước bất kỳ bước nào.
"""

import importlib.metadata  # noqa: F401 — pandas-ta Python 3.11 fix
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd

from multiagents_trading_assistant.indicators import compute_indicators
from multiagents_trading_assistant.screener.trade_screener import (
    detect_bb_squeeze,
    detect_breakout,
    detect_double_bottom,
    detect_flag_pennant,
    detect_golden_cross,
    detect_hammer,
    detect_inside_bar,
    detect_ma_pullback,
    detect_macd_crossover,
    detect_momentum_surge,
    detect_nr7,
    detect_retest,
    detect_rsi_bounce,
    detect_spring,
    # Price Action setups
    detect_pin_bar,
    detect_bullish_engulfing,
    detect_trend_pullback,
    detect_breakout_retest_entry,
)
from multiagents_trading_assistant.backtest.positions import Trade


# Absolute backstop — ngăn zombie position (không phải exit chủ động).
_SAFETY_CAP = 120

# Thứ tự ưu tiên theo _STRATEGY_BASE_SCORE — setup score cao hơn được check trước.
_STRATEGIES = [
    # ── Original 14 ──
    ("BREAKOUT",               detect_breakout),
    ("FLAG_PENNANT",           detect_flag_pennant),
    ("BB_SQUEEZE",             detect_bb_squeeze),
    ("RETEST",                 detect_retest),
    ("SPRING",                 detect_spring),
    ("GOLDEN_CROSS",           detect_golden_cross),
    ("DOUBLE_BOTTOM",          detect_double_bottom),
    ("MOMENTUM_SURGE",         detect_momentum_surge),
    ("MACD_CROSSOVER",         detect_macd_crossover),
    ("MA_PULLBACK",            detect_ma_pullback),
    ("INSIDE_BAR",             detect_inside_bar),
    ("NR7",                    detect_nr7),
    ("HAMMER",                 detect_hammer),
    ("RSI_BOUNCE",             detect_rsi_bounce),
    # ── Price Action setups ──
    ("TREND_PULLBACK",         detect_trend_pullback),
    ("BREAKOUT_RETEST_ENTRY",  detect_breakout_retest_entry),
    ("BULLISH_ENGULFING",      detect_bullish_engulfing),
    ("PIN_BAR",                detect_pin_bar),
]


def _date_str(val) -> str:
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    return str(val)[:10]


def _compute_initial_sl(entry: float, ind: dict) -> float:
    """SL ban đầu = entry − max(1.5×ATR, 2% entry), capped tối đa 5% dưới entry."""
    atr = ind.get("atr") or 0.0
    if atr > 0 and entry > 0:
        sl_dist = min(max(1.5 * atr, entry * 0.02), entry * 0.05)
    else:
        sl_dist = entry * 0.04
    return round(entry - sl_dist, 0)


def _swing_low_sl(
    lows: np.ndarray,
    entry_price: float,
    entry_atr: float,
    current_sl: float,
    peak_close: float,
) -> float:
    """Trail SL lên swing low gần nhất theo mức lãi hiện tại.

    Lookback mở rộng khi lãi nhiều hơn — cho giá "thở" theo biên độ xu hướng.
    SL chỉ tăng, không bao giờ giảm (bảo vệ lãi).
    """
    gain_pct = (peak_close - entry_price) / entry_price * 100
    if gain_pct < 5.0:
        return current_sl  # Chưa có đủ đệm, giữ nguyên SL gốc

    n = len(lows)
    if gain_pct >= 20.0:
        lookback = min(20, n)
    elif gain_pct >= 10.0:
        lookback = min(12, n)
    else:
        lookback = min(7, n)

    swing_low = float(np.min(lows[-lookback:]))
    buffer = entry_atr * 0.5 if entry_atr > 0 else entry_price * 0.01
    new_sl = round(swing_low - buffer, 0)

    # SL chỉ tăng — không bao giờ kéo xuống
    return max(current_sl, new_sl)


def _detect_reversal(
    window: pd.DataFrame,
    entry_price: float,
    peak_close: float,
) -> tuple[bool, str]:
    """Phát hiện tín hiệu đảo chiều price-action.

    Chỉ kích hoạt khi đang lãi ≥ 5% để tránh noise đầu lệnh.

    3 loại reversal:
      MA20_BREAK  — 2 nến liên tiếp đóng dưới MA20 + MA20 đang giảm
      DOUBLE_TOP  — 2 đỉnh cách nhau ≤ 2.5%, pull back qua midpoint
      TREND_BREAK — 2 nến liên tiếp đóng dưới prior swing low
    """
    gain_pct = (peak_close - entry_price) / entry_price * 100
    if gain_pct < 5.0:
        return False, ""

    closes = window["close"].values.astype(float)
    highs  = window["high"].values.astype(float)
    lows   = window["low"].values.astype(float)
    n = len(closes)

    if n < 25:
        return False, ""

    # ── 1. MA20 breakdown ──────────────────────────────────────────────────
    ma20 = pd.Series(closes).rolling(20).mean().values
    if not (np.isnan(ma20[-1]) or np.isnan(ma20[-2]) or np.isnan(ma20[-6])):
        if closes[-1] < ma20[-1] and closes[-2] < ma20[-2]:
            # MA20 phải đang giảm (so với 5 bar trước)
            if ma20[-1] < ma20[-6] * 0.9995:
                return True, "MA20_BREAK"

    # ── 2. Double top (gain ≥ 8%, cần ít nhất 20 bar) ─────────────────────
    if gain_pct >= 8.0 and n >= 20:
        recent_highs = highs[-20:]
        peak_idx = int(np.argmax(recent_highs))
        # Đỉnh phải nằm giữa window (không phải bar đầu/cuối)
        if 4 <= peak_idx <= 16:
            first_section = recent_highs[: peak_idx - 2]
            if len(first_section) >= 3:
                first_peak  = float(np.max(first_section))
                second_peak = float(recent_highs[peak_idx])
                # Hai đỉnh trong phạm vi 2.5%
                if abs(first_peak - second_peak) / second_peak < 0.025:
                    # Neckline = đáy giữa hai đỉnh
                    mid_lows = lows[-20 + peak_idx :]
                    neckline = float(np.min(mid_lows)) if len(mid_lows) > 0 else float(np.min(lows[-5:]))
                    midpoint = (second_peak + neckline) / 2
                    if closes[-1] < midpoint:
                        return True, "DOUBLE_TOP"

    # ── 3. Trend break (gain ≥ 10%, cần ít nhất 15 bar) ──────────────────
    if gain_pct >= 10.0 and n >= 15:
        # Prior swing low = đáy thấp nhất từ 3-15 bar trước (không phải bar hiện tại)
        prior_swing_low = float(np.min(lows[-15:-3]))
        if closes[-1] < prior_swing_low and closes[-2] < prior_swing_low:
            return True, "TREND_BREAK"

    return False, ""


def run_symbol(
    symbol: str,
    df: pd.DataFrame,
    lookback: int = 60,
    max_hold: int = _SAFETY_CAP,
    rr_ratio: float = 1.5,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    setups: Optional[list[str]] = None,
) -> list[Trade]:
    """Walk-forward backtest cho một mã cổ phiếu.

    Args:
        symbol:    Mã cổ phiếu.
        df:        OHLCV DataFrame (columns: date, open, high, low, close, volume).
        lookback:  Số bar tối thiểu trước khi phát tín hiệu (≥ 60 cho MA60).
        max_hold:  Safety cap tuyệt đối (mặc định _SAFETY_CAP=120). Không dùng cho exit chủ động.
        rr_ratio:  Tỷ lệ dùng tính initial_target tham chiếu (không phải hard TP).
        from_date: Bắt đầu tính tín hiệu (YYYY-MM-DD). Lookback vẫn tích lũy trước ngày này.
        to_date:   Kết thúc tính tín hiệu.
        setups:    Lọc theo tên setup (None = toàn bộ 18 setups).

    Returns:
        Danh sách Trade đã đóng.

    Exit logic (price-action based, không time-based):
        1. SL/TSL hit — trailing swing-low SL
        2. Reversal signal — MA20_BREAK | DOUBLE_TOP | TREND_BREAK
        3. Safety cap — _SAFETY_CAP bar tuyệt đối
    """
    if df.empty or len(df) < lookback + 2:
        return []

    df = df.sort_values("date").reset_index(drop=True)

    strategies = [
        (name, fn) for name, fn in _STRATEGIES
        if setups is None or name in setups
    ]

    from_ts = pd.Timestamp(from_date) if from_date else None
    to_ts   = pd.Timestamp(to_date)   if to_date   else None

    trades: list[Trade] = []
    open_pos: Optional[Trade] = None
    entry_bar_idx: int = -1

    # Trailing SL state — reset khi mở lệnh mới
    dynamic_sl: float = 0.0
    peak_close: float = 0.0

    for i in range(lookback, len(df)):
        bar    = df.iloc[i]
        date   = _date_str(bar["date"])
        lo     = float(bar["low"])
        hi     = float(bar["high"])  # noqa: F841 — kept for clarity
        cl     = float(bar["close"])
        op     = float(bar["open"])
        bar_ts = pd.Timestamp(date)

        # ── 1. Exit check ──────────────────────────────────────────────────
        if open_pos is not None:
            bars_held = i - entry_bar_idx

            # Cập nhật peak close (dùng close, không phải high — tránh intraday wick)
            peak_close = max(peak_close, cl)

            # Swing-low trailing SL
            held_lows  = df.iloc[entry_bar_idx : i + 1]["low"].values
            dynamic_sl = _swing_low_sl(
                held_lows, open_pos.entry_price, open_pos.entry_atr, dynamic_sl, peak_close
            )

            # Exit 1: SL bị chạm
            if lo <= dynamic_sl:
                reason  = "TSL" if dynamic_sl > open_pos.stop_loss else "SL"
                # Gap-down: nếu open đã dưới SL, thoát tại open (thực tế hơn)
                exit_px = min(dynamic_sl, op)
                open_pos.close(date, exit_px, reason, bars_held)
                trades.append(open_pos)
                open_pos = None

            # Exit 2: Reversal signal (chỉ khi đang có lãi ≥ 5%)
            elif (peak_close - open_pos.entry_price) / open_pos.entry_price >= 0.05:
                window_now = df.iloc[: i + 1]
                rev, rev_reason = _detect_reversal(window_now, open_pos.entry_price, peak_close)
                if rev:
                    open_pos.close(date, cl, f"REVERSAL_{rev_reason}", bars_held)
                    trades.append(open_pos)
                    open_pos = None

            # Exit 3: Safety cap — absolute backstop
            if open_pos is not None and bars_held >= _SAFETY_CAP:
                open_pos.close(date, cl, "SAFETY_CAP", bars_held)
                trades.append(open_pos)
                open_pos = None

        # ── 2. Signal detection ─────────────────────────────────────────────
        if open_pos is not None:
            continue
        if from_ts and bar_ts < from_ts:
            continue
        if to_ts and bar_ts > to_ts:
            break
        if i + 1 >= len(df):
            continue

        # window: chỉ dữ liệu đến bar i — không nhìn bar i+1
        window = df.iloc[: i + 1]
        ind    = compute_indicators(window)
        if not ind:
            continue

        for setup_name, detect_fn in strategies:
            passed, reasons = detect_fn(window, ind)
            if not passed:
                continue

            next_bar    = df.iloc[i + 1]
            entry_price = float(next_bar["open"])
            entry_date  = _date_str(next_bar["date"])

            if pd.isna(entry_price) or entry_price <= 0:
                continue

            sl  = _compute_initial_sl(entry_price, ind)
            atr = ind.get("atr") or 0.0

            if sl <= 0 or sl >= entry_price:
                continue

            # initial_target = tham chiếu R:R — không dùng làm hard exit
            initial_target = round(entry_price + rr_ratio * (entry_price - sl), 0)

            open_pos = Trade(
                symbol=symbol,
                setup_type=setup_name,
                signal_date=date,
                entry_date=entry_date,
                entry_price=entry_price,
                stop_loss=sl,
                take_profit=initial_target,  # lưu để tham chiếu trong metrics
                entry_atr=atr,
                reasons=reasons,
            )
            entry_bar_idx = i + 1
            # Reset trailing state cho lệnh mới
            dynamic_sl = sl
            peak_close = entry_price
            break

    return trades


def run_universe(
    ohlcv_map: dict[str, pd.DataFrame],
    lookback: int = 60,
    max_hold: int = _SAFETY_CAP,
    rr_ratio: float = 1.5,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    setups: Optional[list[str]] = None,
    max_workers: int = 8,
) -> dict[str, list[Trade]]:
    """Walk-forward backtest cho nhiều mã, chạy song song.

    Returns:
        dict[symbol → list[Trade đã đóng]]
    """
    results: dict[str, list[Trade]] = {}

    def _run_one(sym: str, df: pd.DataFrame) -> tuple[str, list[Trade]]:
        try:
            t = run_symbol(sym, df, lookback, max_hold, rr_ratio, from_date, to_date, setups)
            return sym, t
        except Exception as e:
            print(f"[backtest] {sym} lỗi: {e}")
            return sym, []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, sym, df): sym for sym, df in ohlcv_map.items()}
        done = 0
        for future in as_completed(futures):
            done += 1
            sym, sym_trades = future.result()
            results[sym] = sym_trades
            if done % 10 == 0:
                total = sum(len(t) for t in results.values())
                print(f"[backtest] {done}/{len(ohlcv_map)} mã xong — {total} trades")

    total = sum(len(t) for t in results.values())
    print(f"[backtest] Xong: {len(results)} mã, {total} trades tổng cộng")
    return results
