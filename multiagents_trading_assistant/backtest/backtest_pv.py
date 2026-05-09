"""PV-signal backtest engine — dùng Price-Volume Intelligence Layer làm entry trigger.

Entry triggers (from analyze_price_volume()):
  breakout_confirmed      → PV_BREAKOUT
  washout                 → PV_WASHOUT
  absorption_signal       → PV_ABSORPTION
  constructive_pullback   → PV_PULLBACK
  bullish_volume_expansion→ PV_BULL_EXPANSION

Điều kiện vào lệnh:
  - entry_bias == "bullish"  (không vào khi "avoid" hoặc "bearish")
  - price_volume_score >= min_pv_score  (mặc định 0 — chỉ cần bias bullish)
  - Không vào khi có FAKE_BREAKOUT_RISK hoặc HEAVY_DISTRIBUTION

Exit mechanics: giống TA engine (SL → TSL swing-low → Reversal → Safety cap 120 bar).

Anti-look-ahead:
  - Mỗi bar i: window = df.iloc[:i+1] → analyze_price_volume(window)
  - Entry: open của bar i+1 (ATO hôm sau)
  - Exit: check SL/reversal bar-by-bar từ i+1 trở đi
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd

from multiagents_trading_assistant.price_volume import analyze_price_volume
from multiagents_trading_assistant.backtest.engine import (
    _compute_initial_sl,
    _swing_low_sl,
    _detect_reversal,
    _calc_cooldown,
    _SAFETY_CAP,
)
from multiagents_trading_assistant.backtest.positions import Trade

# PV signal → setup_type name mapping
_PV_SIGNAL_MAP: dict[str, str] = {
    "breakout_confirmed":       "PV_BREAKOUT",
    "washout":                  "PV_WASHOUT",
    "absorption_signal":        "PV_ABSORPTION",
    "constructive_pullback":    "PV_PULLBACK",
    "bullish_volume_expansion": "PV_BULL_EXPANSION",
}

# Risk flags yang blokir entry
_BLOCK_FLAGS = {"FAKE_BREAKOUT_RISK", "HEAVY_DISTRIBUTION"}


def _date_str(val) -> str:
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    return str(val)[:10]


def _check_pv_entry(
    window: pd.DataFrame,
    signals_filter: Optional[list[str]],
    min_pv_score: int,
) -> tuple[Optional[str], list[str], dict]:
    """Kiểm tra PV signals trên window hiện tại.

    Returns:
        (setup_name, reasons, pv_result) — setup_name=None nếu không có signal.
    """
    pv = analyze_price_volume(window)
    if not pv or pv.get("price_volume_score") is None:
        return None, [], pv

    bias = pv.get("entry_bias", "neutral")
    if bias in ("avoid", "bearish"):
        return None, [], pv

    score = int(pv.get("price_volume_score") or 0)
    if score < min_pv_score:
        return None, [], pv

    flags = set(pv.get("risk_flags") or [])
    if flags & _BLOCK_FLAGS:
        return None, [], pv

    signals: dict = pv.get("signals") or {}
    for sig_key, setup_name in _PV_SIGNAL_MAP.items():
        if signals_filter and setup_name not in signals_filter:
            continue
        if signals.get(sig_key):
            reasons = [
                f"pv_signal={sig_key}",
                f"pv_score={score}",
                f"pv_bias={bias}",
            ]
            tags = pv.get("setup_tags") or []
            if tags:
                reasons.append(f"pv_tags={','.join(tags)}")
            return setup_name, reasons, pv

    return None, [], pv


def run_pv_symbol(
    symbol: str,
    df: pd.DataFrame,
    signals: Optional[list[str]] = None,
    min_pv_score: int = 0,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    rr_ratio: float = 1.5,
    max_hold: int = _SAFETY_CAP,
    lookback: int = 30,
    cooldown_bars: int = 0,
    loss_streak_pause: int = 0,
) -> list[Trade]:
    """Walk-forward PV-signal backtest cho một mã cổ phiếu.

    Args:
        symbol:        Mã cổ phiếu.
        df:            OHLCV DataFrame (columns: date, open, high, low, close, volume).
        signals:       Lọc theo PV setup names (None = toàn bộ 5 signal).
                       VD: ["PV_BREAKOUT", "PV_WASHOUT"]
        min_pv_score:  Ngưỡng score tối thiểu để vào lệnh (mặc định 0 = bias đủ).
        from_date:     Bắt đầu tính signal (YYYY-MM-DD). Lookback vẫn tích lũy.
        to_date:       Kết thúc tính signal.
        rr_ratio:      Tỷ lệ R:R cho initial_target tham chiếu (không phải hard TP).
        max_hold:      Safety cap tuyệt đối (mặc định 120 bar).
        lookback:      Số bar warmup trước khi phát signal (tối thiểu 25 cho ATR14 + MA20).
        cooldown_bars: Số bar chờ sau exit trước khi tìm signal mới.
        loss_streak_pause: Sau N lần lỗ liên tiếp → kéo dài cooldown.

    Returns:
        list[Trade] đã đóng, tương thích với print_report/save_report/compute_metrics.
    """
    if df.empty or len(df) < max(lookback, 30) + 2:
        return []

    df = df.sort_values("date").reset_index(drop=True)

    from_ts = pd.Timestamp(from_date) if from_date else None
    to_ts   = pd.Timestamp(to_date)   if to_date   else None

    trades: list[Trade] = []
    open_pos: Optional[Trade] = None
    entry_bar_idx: int = -1
    dynamic_sl: float = 0.0
    peak_close: float = 0.0
    cooldown_until: int = 0
    cur_loss_streak: int = 0

    min_start = max(lookback, 30)  # đảm bảo đủ warmup cho ATR14 + MA20

    for i in range(min_start, len(df)):
        bar    = df.iloc[i]
        date   = _date_str(bar["date"])
        lo     = float(bar["low"])
        hi     = float(bar["high"])  # noqa: F841
        cl     = float(bar["close"])
        op     = float(bar["open"])
        bar_ts = pd.Timestamp(date)

        # ── 1. Exit check ──────────────────────────────────────────────────────
        if open_pos is not None:
            bars_held = i - entry_bar_idx
            peak_close = max(peak_close, cl)

            held_lows = df.iloc[entry_bar_idx: i + 1]["low"].values
            dynamic_sl = _swing_low_sl(
                held_lows, open_pos.entry_price, open_pos.entry_atr, dynamic_sl, peak_close
            )

            # Exit 1: SL / TSL
            if lo <= dynamic_sl:
                reason  = "TSL" if dynamic_sl > open_pos.stop_loss else "SL"
                exit_px = min(dynamic_sl, op)
                open_pos.close(date, exit_px, reason, bars_held)
                trades.append(open_pos)
                cooldown_until, cur_loss_streak = _calc_cooldown(
                    open_pos, i, cooldown_bars, loss_streak_pause, cur_loss_streak)
                open_pos = None

            # Exit 2: Reversal (chỉ khi đang lãi ≥ 5%)
            elif (peak_close - open_pos.entry_price) / open_pos.entry_price >= 0.05:
                window_now = df.iloc[: i + 1]
                rev, rev_reason = _detect_reversal(window_now, open_pos.entry_price, peak_close)
                if rev:
                    open_pos.close(date, cl, f"REVERSAL_{rev_reason}", bars_held)
                    trades.append(open_pos)
                    cooldown_until, cur_loss_streak = _calc_cooldown(
                        open_pos, i, cooldown_bars, loss_streak_pause, cur_loss_streak)
                    open_pos = None

            # Exit 3: Safety cap
            if open_pos is not None and bars_held >= max_hold:
                open_pos.close(date, cl, "SAFETY_CAP", bars_held)
                trades.append(open_pos)
                cooldown_until, cur_loss_streak = _calc_cooldown(
                    open_pos, i, cooldown_bars, loss_streak_pause, cur_loss_streak)
                open_pos = None

        # ── 2. Signal detection ────────────────────────────────────────────────
        if open_pos is not None:
            continue
        if cooldown_bars > 0 and i < cooldown_until:
            continue
        if from_ts and bar_ts < from_ts:
            continue
        if to_ts and bar_ts > to_ts:
            break
        if i + 1 >= len(df):
            continue

        window = df.iloc[: i + 1]
        setup_name, reasons, pv_result = _check_pv_entry(window, signals, min_pv_score)
        if setup_name is None:
            continue

        next_bar    = df.iloc[i + 1]
        entry_price = float(next_bar["open"])
        entry_date  = _date_str(next_bar["date"])

        if to_ts and pd.Timestamp(entry_date) > to_ts:
            continue
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # Compute SL from PV metrics (ATR14 available in pv_result.metrics)
        metrics = pv_result.get("metrics") or {}
        atr = float(metrics.get("atr14") or 0.0)
        mock_ind = {"atr": atr}
        sl = _compute_initial_sl(entry_price, mock_ind)

        if sl <= 0 or sl >= entry_price:
            continue

        initial_target = round(entry_price + rr_ratio * (entry_price - sl), 0)
        pv_score = int(pv_result.get("price_volume_score") or 0)

        open_pos = Trade(
            symbol=symbol,
            setup_type=setup_name,
            signal_date=date,
            entry_date=entry_date,
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=initial_target,
            entry_atr=atr,
            confluence_score=float(max(0, pv_score)),
            reasons=reasons,
        )
        entry_bar_idx = i + 1
        dynamic_sl  = sl
        peak_close  = entry_price

    # Force-close lệnh còn mở cuối kỳ
    if open_pos is not None:
        last = df.iloc[-1]
        bars_held = len(df) - 1 - entry_bar_idx
        open_pos.close(_date_str(last["date"]), float(last["close"]), "END_OF_DATA", bars_held)
        trades.append(open_pos)

    return trades


def run_pv_universe(
    ohlcv_map: dict[str, pd.DataFrame],
    signals: Optional[list[str]] = None,
    min_pv_score: int = 0,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    rr_ratio: float = 1.5,
    max_hold: int = _SAFETY_CAP,
    lookback: int = 30,
    cooldown_bars: int = 0,
    loss_streak_pause: int = 0,
    max_workers: int = 8,
) -> dict[str, list[Trade]]:
    """Walk-forward PV backtest cho nhiều mã, chạy song song.

    Returns:
        dict[symbol → list[Trade đã đóng]]
    """
    results: dict[str, list[Trade]] = {}

    def _run_one(sym: str, df: pd.DataFrame) -> tuple[str, list[Trade]]:
        try:
            t = run_pv_symbol(
                sym, df,
                signals=signals,
                min_pv_score=min_pv_score,
                from_date=from_date,
                to_date=to_date,
                rr_ratio=rr_ratio,
                max_hold=max_hold,
                lookback=lookback,
                cooldown_bars=cooldown_bars,
                loss_streak_pause=loss_streak_pause,
            )
            return sym, t
        except Exception as e:
            print(f"[pv-backtest] {sym} lỗi: {e}")
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
                print(f"[pv-backtest] {done}/{len(ohlcv_map)} mã xong — {total} trades")

    total = sum(len(t) for t in results.values())
    print(f"[pv-backtest] Xong: {len(results)} mã, {total} trades tổng cộng")
    return results
