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
from multiagents_trading_assistant.setup_scoring import (
    MF_STRICT_SETUPS as _MF_STRICT_SETUPS,
    MF_COMPRESSION_SETUPS as _MF_COMPRESSION_SETUPS,
    MF_REVERSAL_SETUPS as _MF_REVERSAL_SETUPS,
)
from multiagents_trading_assistant.agents.trade.money_flow_agent import (
    BLACKBOX_DEFAULTS,
    add_money_flow_features,
)
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
    # Ichimoku setups
    detect_kumo_breakout,
    detect_tk_cross,
    detect_kijun_bounce,
    detect_kumo_twist_entry,
)
from multiagents_trading_assistant.backtest.positions import Trade
from multiagents_trading_assistant.backtest.pipeline_review import (
    merge_stats as _merge_pipeline_review_stats,
    new_stats as _new_pipeline_review_stats,
    record_stats as _record_pipeline_review_stats,
    review_entry as _review_pipeline_entry,
)


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
    # Ichimoku setups
    ("KUMO_BREAKOUT",          detect_kumo_breakout),
    ("TK_CROSS",               detect_tk_cross),
    ("KIJUN_BOUNCE",           detect_kijun_bounce),
    ("KUMO_TWIST_ENTRY",       detect_kumo_twist_entry),
]

SETUP_REGIME_REQUIREMENT = {
    "BREAKOUT": ["UPTREND"],
    "INSIDE_BAR": ["UPTREND"],
    "MOMENTUM_SURGE": ["UPTREND"],
    "FLAG_PENNANT": ["UPTREND"],
    "GOLDEN_CROSS": ["UPTREND", "SIDEWAY"],
    "DOUBLE_BOTTOM": ["SIDEWAY", "DOWNTREND"],
    "RSI_BOUNCE": ["SIDEWAY", "DOWNTREND"],
    "BB_SQUEEZE": ["SIDEWAY"],
    "NR7": ["UPTREND", "SIDEWAY"],
    "BULLISH_ENGULFING": ["UPTREND", "SIDEWAY", "DOWNTREND"],
    "KUMO_BREAKOUT": ["UPTREND"],
    "TK_CROSS": ["UPTREND"],
    "KIJUN_BOUNCE": ["UPTREND"],
    "KUMO_TWIST_ENTRY": ["UPTREND", "SIDEWAY"],
}


def _is_setup_allowed_in_regime(setup_type: str, ref_trend: str) -> bool:
    allowed = SETUP_REGIME_REQUIREMENT.get(setup_type)
    return allowed is None or ref_trend in allowed


def _date_str(val) -> str:
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    return str(val)[:10]


def _prep_ta_money_flow_features(df: pd.DataFrame, params: Optional[dict] = None) -> pd.DataFrame:
    """Precompute causal money-flow fields for TA entries."""
    p = {**BLACKBOX_DEFAULTS, **(params or {})}
    feat = add_money_flow_features(df, p)
    vr = feat["value_ratio_20"].fillna(1.0)
    cp = feat["close_position"].fillna(0.5)
    ret = feat["ret_1d"].fillna(0.0)
    ret_5d = feat["ret_5d"].fillna(0.0)
    dist_ma20 = feat["dist_ma20"].fillna(0.0)
    av = feat["avg_value_20"].fillna(0.0)
    res = feat["resistance_20"]

    money_in = (ret > 0) & (vr >= p["money_in_value_ratio"]) & (cp >= 0.60)
    money_out = (ret < 0) & (vr >= p["money_in_value_ratio"]) & (cp <= p["weak_close"])
    breakout_flow = (
        res.notna()
        & (res > 0)
        & (feat["close"] > res)
        & (vr >= p["breakout_value_ratio"])
        & (cp >= p["strong_close"])
        & (ret >= 0.015)
    )
    distribution = money_out | (
        (vr >= p["distribution_value_ratio"]) & (cp <= 0.35) & (ret <= 0)
    )
    recent_distribution = (
        distribution.shift(1)
        .rolling(int(p["recent_distribution_window"]))
        .sum()
        .fillna(0)
        > 0
    )
    early_money_in = (
        money_in
        & ~recent_distribution
        & (ret_5d < p["early_ret_5d_max"])
        & (dist_ma20 < p["early_dist_ma20_max"])
    )
    chase_money_in = money_in & (
        (ret_5d >= p["chase_ret_5d_min"])
        | (dist_ma20 >= p["chase_dist_ma20_min"])
    )
    exhaustion_inflow = money_in & (vr >= 2.5) & ((cp < 0.70) | chase_money_in)
    base_tight = feat["range_20"].notna() & feat["range_120_q35"].notna() & (
        feat["range_20"] < feat["range_120_q35"] * 1.15
    )
    volume_dry_before = (
        feat["volume_ratio_20"].shift(1).rolling(5).mean().fillna(1.0) < 1.1
    )
    confirmed_breakout = breakout_flow & ~recent_distribution & (base_tight | volume_dry_before)
    score = (
        money_in.astype(int) * 2
        + early_money_in.astype(int) * 1
        + confirmed_breakout.astype(int) * 3
        + (breakout_flow & ~confirmed_breakout).astype(int) * 1
        - money_out.astype(int) * 2
        - chase_money_in.astype(int) * 2
        - exhaustion_inflow.astype(int) * 3
        - recent_distribution.astype(int) * 2
        - distribution.astype(int) * 4
    ).clip(-10, 10)

    regime = pd.Series("NEUTRAL", index=feat.index)
    regime[score >= 3] = "MONEY_IN"
    regime[early_money_in & (score >= 2)] = "EARLY_MONEY_IN"
    regime[chase_money_in] = "CHASE_MONEY_IN"
    regime[confirmed_breakout & (score >= 4)] = "BREAKOUT_FLOW"
    regime[exhaustion_inflow] = "EXHAUSTION_INFLOW"
    regime[money_out | (score <= -2)] = "MONEY_OUT"
    regime[distribution | (score <= -4)] = "DISTRIBUTION"

    feat["mf_score"] = score
    feat["mf_regime"] = regime
    feat["liquidity_ok"] = av >= p["min_avg_value"]
    feat["distribution"] = distribution
    feat["recent_distribution"] = recent_distribution
    feat["money_in"] = money_in
    feat["early_money_in"] = early_money_in
    feat["chase_money_in"] = chase_money_in
    feat["exhaustion_inflow"] = exhaustion_inflow
    feat["breakout_flow"] = breakout_flow
    feat["confirmed_breakout_flow"] = confirmed_breakout
    return feat.reset_index(drop=True)


# _MF_STRICT/COMPRESSION/REVERSAL_SETUPS imported from setup_scoring


def _money_flow_entry_ok(row: pd.Series, min_score: int, setup_name: str) -> tuple[bool, list[str]]:
    if not bool(row.get("liquidity_ok", False)):
        return False, []
    regime = str(row.get("mf_regime", "NEUTRAL"))
    score = int(row.get("mf_score", 0))
    recent_distribution = bool(row.get("recent_distribution", False))
    if regime in {"DISTRIBUTION", "MONEY_OUT", "EXHAUSTION_INFLOW"}:
        return False, []

    if setup_name in _MF_STRICT_SETUPS:
        if regime not in {"BREAKOUT_FLOW", "EARLY_MONEY_IN", "MONEY_IN"} or score < min_score:
            return False, []
    elif setup_name in _MF_COMPRESSION_SETUPS:
        if recent_distribution or regime == "CHASE_MONEY_IN":
            return False, []
    elif setup_name in _MF_REVERSAL_SETUPS:
        if recent_distribution:
            return False, []
    elif score < min_score and regime not in {"BREAKOUT_FLOW", "EARLY_MONEY_IN", "MONEY_IN"}:
        return False, []

    return True, [f"money_flow={regime}", f"mf_score={score}"]


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


def _calc_cooldown(
    trade: "Trade",
    bar_idx: int,
    cooldown_bars: int,
    loss_streak_pause: int,
    cur_streak: int,
) -> tuple[int, int]:
    """Tính cooldown_until và cur_loss_streak mới sau khi một lệnh đóng.

    SL/TSL:         cooldown đầy đủ (reassess sau thua)
    Profitable exit: cooldown / 2 (thở ngắn, không phải reassess)
    Loss streak:    sau N lần lỗ liên tiếp → thêm cooldown × 2
    """
    is_loss = (trade.pnl_pct or 0.0) < 0.0
    new_streak = cur_streak + 1 if is_loss else 0

    if cooldown_bars <= 0:
        return 0, new_streak

    # SL/TSL: stop-out surprise → full cooldown. SAFETY_CAP: zombie exit → full cooldown.
    # Profitable reversals: lighter cooldown (half), voluntary, price-action confirmed.
    hard_exit = trade.exit_reason in {"SL", "TSL", "SAFETY_CAP"}
    base = cooldown_bars if hard_exit else max(0, cooldown_bars // 2)

    streak_bonus = 0
    if loss_streak_pause > 0 and new_streak >= loss_streak_pause:
        streak_bonus = cooldown_bars * 2

    return bar_idx + base + streak_bonus, new_streak


def run_symbol(
    symbol: str,
    df: pd.DataFrame,
    lookback: int = 60,
    max_hold: int = _SAFETY_CAP,
    rr_ratio: float = 1.5,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    setups: Optional[list[str]] = None,
    use_money_flow: bool = True,
    money_flow_min_score: int = 3,
    money_flow_params: Optional[dict] = None,
    pipeline_review: bool = False,
    pipeline_review_stats: Optional[dict] = None,
    pending_entry: bool = False,
    pending_bars: int = 3,
    cooldown_bars: int = 0,
    loss_streak_pause: int = 0,
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
        setups:              Lọc theo tên setup (None = toàn bộ 18 setups).
        cooldown_bars:       Số bar chờ sau mỗi exit trước khi tìm signal mới (0 = tắt).
                             SL/TSL exit → cooldown_bars đầy đủ.
                             Profitable exit → cooldown_bars // 2 (min 0).
        loss_streak_pause:   Sau N lần lỗ liên tiếp, tự động kéo dài cooldown thêm
                             cooldown_bars × 2. 0 = tắt.

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
    mf_feat = _prep_ta_money_flow_features(df, money_flow_params) if use_money_flow else None

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

    # Cooldown / loss-streak state
    cooldown_until: int = 0   # bar index không được vào lệnh (i < cooldown_until)
    cur_loss_streak: int = 0  # số lần lỗ liên tiếp gần nhất

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
                cooldown_until, cur_loss_streak = _calc_cooldown(
                    open_pos, i, cooldown_bars, loss_streak_pause, cur_loss_streak)
                open_pos = None

            # Exit 2: Reversal signal (chỉ khi đang có lãi ≥ 5%)
            elif (peak_close - open_pos.entry_price) / open_pos.entry_price >= 0.05:
                window_now = df.iloc[: i + 1]
                rev, rev_reason = _detect_reversal(window_now, open_pos.entry_price, peak_close)
                if rev:
                    open_pos.close(date, cl, f"REVERSAL_{rev_reason}", bars_held)
                    trades.append(open_pos)
                    cooldown_until, cur_loss_streak = _calc_cooldown(
                        open_pos, i, cooldown_bars, loss_streak_pause, cur_loss_streak)
                    open_pos = None

            # Exit 3: Safety cap — absolute backstop
            if open_pos is not None and bars_held >= _SAFETY_CAP:
                open_pos.close(date, cl, "SAFETY_CAP", bars_held)
                trades.append(open_pos)
                cooldown_until, cur_loss_streak = _calc_cooldown(
                    open_pos, i, cooldown_bars, loss_streak_pause, cur_loss_streak)
                open_pos = None

        # ── 2. Signal detection ─────────────────────────────────────────────
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

        # window: chỉ dữ liệu đến bar i — không nhìn bar i+1
        window = df.iloc[: i + 1]
        ind    = compute_indicators(window)
        if not ind:
            continue
        mf_row = mf_feat.iloc[i] if mf_feat is not None else None

        for setup_name, detect_fn in strategies:
            ref_trend = str(ind.get("ma_trend") or "SIDEWAY")
            if not _is_setup_allowed_in_regime(setup_name, ref_trend):
                continue
            passed, reasons = detect_fn(window, ind)
            if not passed:
                continue
            if mf_row is not None:
                mf_ok, mf_reasons = _money_flow_entry_ok(mf_row, money_flow_min_score, setup_name)
                if not mf_ok:
                    continue
                reasons = [*reasons, *mf_reasons]

            next_bar    = df.iloc[i + 1]
            entry_price = float(next_bar["open"])
            entry_date  = _date_str(next_bar["date"])
            entry_bar_pos = i + 1
            fill_ind = ind
            fill_mf_row = mf_row
            fill_reasons = list(reasons)

            if pending_entry:
                fill = _resolve_pending_entry(
                    df=df,
                    signal_idx=i,
                    setup_name=setup_name,
                    ind=ind,
                    max_wait_bars=pending_bars,
                    to_ts=to_ts,
                )
                if fill is None:
                    continue
                entry_bar_pos, entry_price, entry_date, pending_reason = fill
                fill_reasons.append(pending_reason)
                if entry_bar_pos > i + 1:
                    fill_window = df.iloc[:entry_bar_pos]
                    fill_ind = compute_indicators(fill_window) or ind
                    if mf_feat is not None:
                        fill_mf_row = mf_feat.iloc[entry_bar_pos - 1]

            if to_ts and pd.Timestamp(entry_date) > to_ts:
                continue

            if pd.isna(entry_price) or entry_price <= 0:
                continue

            sl  = _compute_initial_sl(entry_price, fill_ind)
            atr = fill_ind.get("atr") or 0.0

            if sl <= 0 or sl >= entry_price:
                continue

            # initial_target = tham chiếu R:R — không dùng làm hard exit
            initial_target = round(entry_price + rr_ratio * (entry_price - sl), 0)

            trade_confluence = 0.0
            if pipeline_review:
                review = _review_pipeline_entry(
                    setup_name=setup_name,
                    ind=fill_ind,
                    mf_row=fill_mf_row,
                    entry_price=entry_price,
                    stop_loss=sl,
                    rr_ratio=rr_ratio,
                )
                _record_pipeline_review_stats(pipeline_review_stats, review)
                if not review.approved:
                    continue
                fill_reasons = [*fill_reasons, *review.reasons]
                trade_confluence = review.confluence_score

            open_pos = Trade(
                symbol=symbol,
                setup_type=setup_name,
                signal_date=date,
                entry_date=entry_date,
                entry_price=entry_price,
                stop_loss=sl,
                take_profit=initial_target,  # lưu để tham chiếu trong metrics
                entry_atr=atr,
                confluence_score=trade_confluence,
                reasons=fill_reasons,
            )
            entry_bar_idx = entry_bar_pos
            # Reset trailing state cho lệnh mới
            dynamic_sl = sl
            peak_close = entry_price
            break

    # Force-close lệnh còn mở cuối kỳ tại giá close bar cuối
    if open_pos is not None:
        last = df.iloc[-1]
        bars_held = len(df) - 1 - entry_bar_idx
        open_pos.close(_date_str(last["date"]), float(last["close"]), "END_OF_DATA", bars_held)
        trades.append(open_pos)

    return trades


def _resolve_pending_entry(
    *,
    df: pd.DataFrame,
    signal_idx: int,
    setup_name: str,
    ind: dict,
    max_wait_bars: int,
    to_ts: pd.Timestamp | None,
) -> tuple[int, float, str, str] | None:
    """Resolve a recommendation-only signal into a pending entry fill."""
    if signal_idx + 1 >= len(df):
        return None

    signal_close = float(df.iloc[signal_idx]["close"])
    zone_low, zone_high = _pending_entry_zone(setup_name, signal_close, ind)
    if zone_low <= 0 or zone_high <= 0 or zone_low > zone_high:
        return None

    expiry = min(len(df) - 1, signal_idx + max(1, max_wait_bars))
    for j in range(signal_idx + 1, expiry + 1):
        bar = df.iloc[j]
        entry_date = _date_str(bar["date"])
        if to_ts and pd.Timestamp(entry_date) > to_ts:
            return None

        op = float(bar["open"])
        hi = float(bar["high"])
        lo = float(bar["low"])
        cl = float(bar["close"])

        # Avoid stale recommendations that break down before filling.
        if cl < zone_low * 0.97:
            return None
        if op > zone_high * 1.03 and lo > zone_high:
            return None

        if lo <= zone_high and hi >= zone_low:
            if zone_low <= op <= zone_high:
                fill_price = op
            elif op < zone_low:
                fill_price = zone_low
            else:
                fill_price = zone_high
            reason = (
                f"pending_entry_fill:{setup_name} "
                f"zone={zone_low:.0f}-{zone_high:.0f} wait={j - signal_idx}d"
            )
            return j, round(float(fill_price), 0), entry_date, reason

    return None


def _pending_entry_zone(setup_name: str, signal_close: float, ind: dict) -> tuple[float, float]:
    """Approximate live entry_zone deterministically for backtests."""
    atr = float(ind.get("atr") or 0.0)
    ma20 = ind.get("ma20")
    supports = ind.get("support_levels") or []
    resistances = ind.get("resistance_levels") or []

    if setup_name in {"BREAKOUT", "MOMENTUM_SURGE", "MACD_CROSSOVER"}:
        return signal_close, signal_close * 1.01
    if setup_name in {"KUMO_BREAKOUT", "TK_CROSS"}:
        cloud_top = ind.get("ichimoku_cloud_top") or signal_close
        level = max(signal_close, float(cloud_top))
        return level, level * 1.01
    if setup_name == "KIJUN_BOUNCE" and ind.get("ichimoku_kijun"):
        level = float(ind["ichimoku_kijun"])
        return level * 0.99, level * 1.01
    if setup_name == "KUMO_TWIST_ENTRY":
        trigger = max(
            float(ind.get("ichimoku_tenkan") or signal_close),
            float(ind.get("ichimoku_kijun") or signal_close),
        )
        return trigger * 0.995, trigger * 1.015
    if setup_name in {"BB_SQUEEZE", "INSIDE_BAR", "NR7", "FLAG_PENNANT"}:
        return signal_close * 0.995, signal_close * 1.01
    if setup_name in {"RETEST", "TREND_PULLBACK", "BREAKOUT_RETEST_ENTRY"}:
        level = float(supports[0]) if supports else signal_close
        return level * 0.99, level * 1.01
    if setup_name == "MA_PULLBACK" and ma20:
        level = float(ma20)
        return level * 0.99, level * 1.01
    if setup_name in {"DOUBLE_BOTTOM", "SPRING", "HAMMER", "RSI_BOUNCE", "BULLISH_ENGULFING", "PIN_BAR"}:
        low = signal_close * 0.99
        if atr > 0:
            low = min(low, signal_close - 0.5 * atr)
        high = signal_close * 1.01
        if resistances:
            high = min(high, float(resistances[0]) * 0.97)
        return low, max(low, high)
    return signal_close * 0.99, signal_close * 1.01


def apply_portfolio_filter(
    all_trades: list["Trade"],
    max_positions: int = 0,
) -> list["Trade"]:
    """Lọc trades theo giới hạn vị thế đồng thời (portfolio cap).

    Chạy chronologically: mỗi ngày, nếu portfolio đầy (>= max_positions) và
    symbol chưa có vị thế mở → bỏ qua signal, nhường slot cho signal tốt hơn đã vào.
    Signal với confluence cao hơn được ưu tiên khi cùng ngày.

    Args:
        all_trades:    Toàn bộ trades từ run_universe (chưa lọc portfolio).
        max_positions: Số vị thế đồng thời tối đa (0 = không giới hạn).

    Returns:
        Danh sách trades đã qua lọc portfolio.
    """
    if max_positions <= 0:
        return all_trades

    # Sort by entry_date ASC, confluence_score DESC (ưu tiên signal mạnh hơn cùng ngày)
    sorted_trades = sorted(
        all_trades,
        key=lambda t: (t.entry_date, -(t.confluence_score or 0.0)),
    )

    accepted: list["Trade"] = []
    open_positions: dict[str, "Trade"] = {}  # symbol → open trade đang held

    for trade in sorted_trades:
        # Đóng các vị thế đã exit trước ngày entry của trade này
        closed_syms = [
            sym for sym, t in open_positions.items()
            if t.exit_date is not None and t.exit_date <= trade.entry_date
        ]
        for sym in closed_syms:
            del open_positions[sym]

        # Bỏ qua nếu symbol đang có vị thế mở
        if trade.symbol in open_positions:
            continue

        # Bỏ qua nếu portfolio đầy
        if len(open_positions) >= max_positions:
            continue

        accepted.append(trade)
        open_positions[trade.symbol] = trade

    return accepted


def run_universe(
    ohlcv_map: dict[str, pd.DataFrame],
    lookback: int = 60,
    max_hold: int = _SAFETY_CAP,
    rr_ratio: float = 1.5,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    setups: Optional[list[str]] = None,
    max_workers: int = 8,
    use_money_flow: bool = True,
    money_flow_min_score: int = 3,
    money_flow_params: Optional[dict] = None,
    pipeline_review: bool = False,
    pipeline_review_stats: Optional[dict] = None,
    pending_entry: bool = False,
    pending_bars: int = 3,
    cooldown_bars: int = 0,
    loss_streak_pause: int = 0,
) -> dict[str, list[Trade]]:
    """Walk-forward backtest cho nhiều mã, chạy song song.

    Returns:
        dict[symbol → list[Trade đã đóng]]
    """
    results: dict[str, list[Trade]] = {}

    def _run_one(sym: str, df: pd.DataFrame) -> tuple[str, list[Trade], dict]:
        local_stats = _new_pipeline_review_stats() if pipeline_review else {}
        try:
            t = run_symbol(
                sym, df, lookback, max_hold, rr_ratio, from_date, to_date, setups,
                use_money_flow=use_money_flow,
                money_flow_min_score=money_flow_min_score,
                money_flow_params=money_flow_params,
                pipeline_review=pipeline_review,
                pipeline_review_stats=local_stats,
                pending_entry=pending_entry,
                pending_bars=pending_bars,
                cooldown_bars=cooldown_bars,
                loss_streak_pause=loss_streak_pause,
            )
            return sym, t, local_stats
        except Exception as e:
            print(f"[backtest] {sym} lỗi: {e}")
            return sym, [], local_stats

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, sym, df): sym for sym, df in ohlcv_map.items()}
        done = 0
        for future in as_completed(futures):
            done += 1
            sym, sym_trades, local_stats = future.result()
            results[sym] = sym_trades
            if pipeline_review_stats is not None:
                _merge_pipeline_review_stats(pipeline_review_stats, local_stats)
            if done % 10 == 0:
                total = sum(len(t) for t in results.values())
                print(f"[backtest] {done}/{len(ohlcv_map)} mã xong — {total} trades")

    total = sum(len(t) for t in results.values())
    print(f"[backtest] Xong: {len(results)} mã, {total} trades tổng cộng")
    return results
