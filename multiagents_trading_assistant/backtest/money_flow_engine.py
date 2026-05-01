"""
money_flow_engine.py — Walk-forward backtester cho Blackbox Money Flow signals.

Entry : open[i+1] khi regime ∈ entry_regimes AND liquidity_ok
        (mặc định BREAKOUT_FLOW, MONEY_IN)

Exit  :
  1. SL/TSL     — trailing swing-low (giống engine.py)
  2. DIST_EXIT  — regime đổi sang DISTRIBUTION → exit tại open[i+1]
  3. Reversal   — MA20_BREAK | DOUBLE_TOP | TREND_BREAK (giống engine.py)
  4. Safety cap — 120 bar tuyệt đối

Zero look-ahead:
  - Pre-compute features bằng rolling causal của pandas (không nhìn trước).
  - Entry tại open[i+1], không dùng close của bar tín hiệu.
  - Regime tại bar i chỉ dùng dữ liệu đến bar i.

Phân tách kết quả theo regime để so sánh chất lượng tín hiệu.
"""

from __future__ import annotations

import importlib.metadata  # noqa: F401 — pandas-ta Python 3.11 fix
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd

from multiagents_trading_assistant.agents.trade.money_flow_agent import (
    BLACKBOX_DEFAULTS,
    add_money_flow_features,
)
from multiagents_trading_assistant.backtest.engine import (
    _SAFETY_CAP,
    _compute_initial_sl,
    _date_str,
    _detect_reversal,
    _swing_low_sl,
)
from multiagents_trading_assistant.backtest.positions import Trade

# Lookback tối thiểu — range_120_q35 cần 120 + 20 = 140 bars
_MIN_LOOKBACK = 140


# ── Feature prep (vectorized, causal) ─────────────────────────────────────────

def _prep_features(
    df: pd.DataFrame,
    vnindex_df: Optional[pd.DataFrame] = None,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Tính toàn bộ money-flow features và regime cho mỗi bar.

    Không look-ahead: pandas rolling là causal theo thiết kế.
    Pre-compute một lần trên full DataFrame, sau đó đọc từng row trong walk-forward.

    Trả về df_feat với DatetimeIndex và các cột bổ sung:
        atr, market_trend, rs_20d,
        regime, mf_score, distribution, breakout_flow, money_in, accumulation, liquidity_ok
    """
    p = {**BLACKBOX_DEFAULTS, **(params or {})}

    # ── Money flow features ────────────────────────────────────────────────────
    df_feat = add_money_flow_features(df, params)  # sets DatetimeIndex

    # ── ATR (True Range 14-bar) ────────────────────────────────────────────────
    prev_close = df_feat["close"].shift(1)
    tr = pd.concat([
        df_feat["high"] - df_feat["low"],
        (df_feat["high"] - prev_close).abs(),
        (df_feat["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df_feat["atr"] = tr.rolling(14).mean()

    # ── Market trend from VNINDEX ──────────────────────────────────────────────
    if vnindex_df is not None and not vnindex_df.empty:
        vni = vnindex_df.copy()
        if "date" in vni.columns:
            vni = vni.set_index("date")
        vni.index = pd.to_datetime(vni.index)
        vni = vni.sort_index()

        vni_close  = vni["close"].reindex(df_feat.index, method="ffill")
        vni_ma20   = vni_close.rolling(20).mean()
        vni_ma50   = vni_close.rolling(50).mean()

        uptrend    = (vni_close > vni_ma20) & (vni_ma20 > vni_ma50)
        downtrend  = (vni_close < vni_ma20) & (vni_close < vni_ma50)

        df_feat["market_trend"] = "SIDEWAY"
        df_feat.loc[uptrend,   "market_trend"] = "UPTREND"
        df_feat.loc[downtrend, "market_trend"] = "DOWNTREND"

        # Relative strength 20d (stock vs VNI)
        stock_ret_20 = df_feat["close"] / df_feat["close"].shift(20) - 1
        vni_ret_20   = vni_close         / vni_close.shift(20)        - 1
        df_feat["rs_20d"] = (stock_ret_20 - vni_ret_20).fillna(0.0)
    else:
        df_feat["market_trend"] = "SIDEWAY"
        df_feat["rs_20d"]       = 0.0

    # ── Vectorized signals ────────────────────────────────────────────────────
    mir = p["money_in_value_ratio"]
    bvr = p["breakout_value_ratio"]
    dvr = p["distribution_value_ratio"]
    sc  = p["strong_close"]
    wc  = p["weak_close"]

    vr   = df_feat["value_ratio_20"].fillna(1.0)
    volr = df_feat["volume_ratio_20"].fillna(1.0)
    cp   = df_feat["close_position"].fillna(0.5)
    ret  = df_feat["ret_1d"].fillna(0.0)
    ret5 = df_feat["ret_5d"].fillna(0.0)
    dist_ma20 = df_feat["dist_ma20"].fillna(0.0)
    av   = df_feat["avg_value_20"].fillna(0.0)
    r20  = df_feat["range_20"]
    r120 = df_feat["range_120_q35"]
    rv20 = df_feat["ret_vol_20"]
    rv120= df_feat["ret_vol_120"]
    res  = df_feat["resistance_20"]
    high = df_feat["high"]
    close= df_feat["close"]
    rs   = df_feat["rs_20d"]

    money_in      = (ret > 0)     & (vr >= mir)  & (cp >= 0.60)
    strong_mi     = (ret >= 0.02) & (vr >= bvr)  & (cp >= sc)
    money_out     = (ret < 0)     & (vr >= mir)  & (cp <= wc)
    strong_mo     = (ret <= -0.02)& (vr >= bvr)  & (cp <= 0.30)

    rng_v  = r20.notna() & r120.notna()
    vol_v  = rv20.notna() & rv120.notna()
    accum  = rng_v & (r20 < r120) & (volr < 0.9) & vol_v & (rv20 < rv120)

    res_v  = res.notna() & (res > 0)
    failed = res_v & (high > res) & (close < res) & (vr >= mir)
    distrib= money_out | failed | ((vr >= dvr) & (cp <= 0.35) & (ret <= 0))

    brkout = res_v & (close > res) & (vr >= bvr) & (cp >= sc) & (ret >= 0.015)
    recent_distrib = (
        distrib.shift(1)
        .rolling(int(p["recent_distribution_window"]))
        .sum()
        .fillna(0)
        > 0
    )
    early_mi = money_in & ~recent_distrib & (ret5 < p["early_ret_5d_max"]) & (dist_ma20 < p["early_dist_ma20_max"])
    chase_mi = money_in & ((ret5 >= p["chase_ret_5d_min"]) | (dist_ma20 >= p["chase_dist_ma20_min"]))
    exhaust = money_in & (vr >= 2.5) & ((cp < 0.70) | chase_mi)
    base_tight = rng_v & (r20 < r120 * 1.15)
    vol_dry_before = df_feat["volume_ratio_20"].shift(1).rolling(5).mean().fillna(1.0) < 1.1
    confirmed_brkout = brkout & ~recent_distrib & (base_tight | vol_dry_before)

    # ── Accumulation: thêm rs >= -0.03 như classify_money_flow ──────────────────
    accum = accum & (rs >= -0.03)

    # ── Score ────────────────────────────────────────────────────────────────────
    # spec có HAI check riêng: sector_is_outperforming (+1) và rs_20d > 0 (+1).
    # Trong backtest không có sector data nên cả hai được proxy bằng rs > 0 → +2 total.
    downtrend_mask = df_feat["market_trend"] == "DOWNTREND"

    score = (
        money_in.astype(int)  * 2
        + strong_mi.astype(int)  * 1
        + early_mi.astype(int)   * 1
        + confirmed_brkout.astype(int) * 3
        + (brkout & ~confirmed_brkout).astype(int) * 1
        + accum.astype(int)      * 1
        + (rs > 0).astype(int)   * 2  # sector_is_outperforming +1 VÀ rs_20d>0 +1
        + money_out.astype(int)  * (-2)
        + strong_mo.astype(int)  * (-1)
        + chase_mi.astype(int)   * (-2)
        + exhaust.astype(int)    * (-3)
        + recent_distrib.astype(int) * (-2)
        + distrib.astype(int)    * (-4)
        + downtrend_mask.astype(int) * (-2)
    ).clip(-10, 10)

    # ── Regime (priority: DISTRIBUTION > BREAKOUT_FLOW > ACCUMULATION > MONEY_IN > MONEY_OUT > NEUTRAL) ──
    regime = pd.Series("NEUTRAL", index=df_feat.index)
    regime[(score <= -2)]                       = "MONEY_OUT"
    regime[(score >= 3)]                        = "MONEY_IN"
    regime[early_mi & (score >= 2)]              = "EARLY_MONEY_IN"
    regime[accum & (score >= 1)]                = "ACCUMULATION"
    regime[chase_mi]                             = "CHASE_MONEY_IN"
    regime[confirmed_brkout & (score >= 4)]      = "BREAKOUT_FLOW"
    regime[exhaust]                              = "EXHAUSTION_INFLOW"
    regime[distrib | (score <= -4)]             = "DISTRIBUTION"  # highest priority

    df_feat["regime"]       = regime
    df_feat["mf_score"]     = score
    df_feat["distribution"] = distrib
    df_feat["recent_distribution"] = recent_distrib
    df_feat["breakout_flow"]= confirmed_brkout
    df_feat["money_in"]     = money_in
    df_feat["early_money_in"] = early_mi
    df_feat["chase_money_in"] = chase_mi
    df_feat["exhaustion_inflow"] = exhaust
    df_feat["accumulation"] = accum
    df_feat["liquidity_ok"] = av >= p["min_avg_value"]

    # ── Debug stats ───────────────────────────────────────────────────────────
    regime_counts = regime.value_counts().to_dict()
    liq_pass = int((av >= p["min_avg_value"]).sum())
    print(
        f"[money_flow_bt] features OK — "
        f"regimes: {regime_counts} | liq_ok: {liq_pass}/{len(df_feat)} bars "
        f"| avg_value: {av.mean()/1e9:.1f}B VND/ngày"
    )

    return df_feat


# ── Single-symbol walk-forward ────────────────────────────────────────────────

def run_money_flow_symbol(
    symbol: str,
    df: pd.DataFrame,
    vnindex_df: Optional[pd.DataFrame] = None,
    entry_regimes: Optional[set[str]] = None,
    min_score: int = 3,
    filter_downtrend: bool = True,
    params: Optional[dict] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    rr_ratio: float = 1.5,
) -> list[Trade]:
    """
    Walk-forward backtest Blackbox Money Flow cho một mã.

    Args:
        symbol:           Mã cổ phiếu.
        df:               OHLCV DataFrame (date, open, high, low, close, volume).
        vnindex_df:       OHLCV VNINDEX để tính market trend + relative strength.
        entry_regimes:    Tập regime cho phép vào lệnh (mặc định BREAKOUT_FLOW + MONEY_IN).
        min_score:        Điểm tối thiểu để vào lệnh (mặc định 3).
        filter_downtrend: Bỏ qua tín hiệu mua khi thị trường DOWNTREND (mặc định True).
        params:           Override BLACKBOX_DEFAULTS.
        from_date:        Ngày bắt đầu phát tín hiệu (YYYY-MM-DD).
        to_date:          Ngày kết thúc phát tín hiệu.
        rr_ratio:         R:R dùng tính initial_target tham chiếu.

    Returns:
        list[Trade] đã đóng.
    """
    if entry_regimes is None:
        entry_regimes = {"BREAKOUT_FLOW", "EARLY_MONEY_IN", "MONEY_IN"}

    if df.empty or len(df) < _MIN_LOOKBACK + 2:
        return []

    df = df.sort_values("date").reset_index(drop=True)

    df_feat = _prep_features(df, vnindex_df, params)
    # df_feat và df cùng số dòng, cùng thứ tự → dùng iloc[i] trực tiếp, không cần lookup
    assert len(df_feat) == len(df), "df_feat và df phải có cùng độ dài"

    from_ts = pd.Timestamp(from_date) if from_date else None
    to_ts   = pd.Timestamp(to_date)   if to_date   else None

    trades: list[Trade] = []
    open_pos: Optional[Trade] = None
    entry_bar_idx: int = -1
    dynamic_sl: float = 0.0
    peak_close: float = 0.0

    # debug counters
    _dbg_regime_ok = 0
    _dbg_score_ok  = 0
    _dbg_liq_ok    = 0
    _dbg_trend_ok  = 0

    for i in range(_MIN_LOOKBACK, len(df)):
        bar    = df.iloc[i]
        date   = _date_str(bar["date"])
        lo     = float(bar["low"])
        cl     = float(bar["close"])
        op     = float(bar["open"])
        bar_ts = pd.Timestamp(date)

        # df_feat chia sẻ thứ tự với df — an toàn dùng positional
        feat_row    = df_feat.iloc[i]
        regime      = str(feat_row["regime"])
        distribution= bool(feat_row["distribution"])

        # ── 1. Exit check ──────────────────────────────────────────────────────
        if open_pos is not None:
            bars_held  = i - entry_bar_idx
            peak_close = max(peak_close, cl)

            # Trailing swing-low SL
            held_lows  = df.iloc[entry_bar_idx : i + 1]["low"].values
            dynamic_sl = _swing_low_sl(
                held_lows, open_pos.entry_price, open_pos.entry_atr, dynamic_sl, peak_close
            )

            # Exit 1: SL/TSL hit
            if lo <= dynamic_sl:
                reason  = "TSL" if dynamic_sl > open_pos.stop_loss else "SL"
                exit_px = min(dynamic_sl, op)
                open_pos.close(date, exit_px, reason, bars_held)
                trades.append(open_pos)
                open_pos = None

            # Exit 2: Distribution flip → exit tại mở phiên kế tiếp
            elif distribution and i + 1 < len(df):
                next_open = float(df.iloc[i + 1]["open"])
                open_pos.close(date, next_open, "DIST_EXIT", bars_held)
                trades.append(open_pos)
                open_pos = None

            # Exit 3: Reversal price-action (chỉ khi đang lãi ≥ 5%)
            elif (peak_close - open_pos.entry_price) / open_pos.entry_price >= 0.05:
                window_now = df.iloc[: i + 1]
                rev, rev_reason = _detect_reversal(window_now, open_pos.entry_price, peak_close)
                if rev:
                    open_pos.close(date, cl, f"REVERSAL_{rev_reason}", bars_held)
                    trades.append(open_pos)
                    open_pos = None

            # Exit 4: Safety cap
            if open_pos is not None and bars_held >= _SAFETY_CAP:
                open_pos.close(date, cl, "SAFETY_CAP", bars_held)
                trades.append(open_pos)
                open_pos = None

        # ── 2. Entry check ─────────────────────────────────────────────────────
        if open_pos is not None:
            continue
        if from_ts and bar_ts < from_ts:
            continue
        if to_ts   and bar_ts > to_ts:
            break
        if i + 1 >= len(df):
            continue

        # Điều kiện vào lệnh
        score      = int(feat_row["mf_score"])
        liq_ok     = bool(feat_row["liquidity_ok"])
        mkt_trend  = str(feat_row["market_trend"])

        if regime not in entry_regimes:
            continue
        _dbg_regime_ok += 1

        if score < min_score:
            continue
        _dbg_score_ok += 1

        if not liq_ok:
            continue
        _dbg_liq_ok += 1

        if filter_downtrend and mkt_trend == "DOWNTREND":
            continue
        _dbg_trend_ok += 1

        # Entry tại ATO hôm sau
        next_bar    = df.iloc[i + 1]
        entry_price = float(next_bar["open"])
        entry_date  = _date_str(next_bar["date"])
        if to_ts and pd.Timestamp(entry_date) > to_ts:
            continue

        if pd.isna(entry_price) or entry_price <= 0:
            continue

        atr = float(feat_row["atr"]) if not pd.isna(feat_row["atr"]) else 0.0
        sl  = _compute_initial_sl(entry_price, {"atr": atr})
        if sl <= 0 or sl >= entry_price:
            continue

        initial_target = round(entry_price + rr_ratio * (entry_price - sl), 0)

        reasons = [f"regime={regime}", f"score={score}", f"mkt={mkt_trend}"]
        if bool(feat_row["breakout_flow"]):
            reasons.append("breakout_flow")
        if bool(feat_row["money_in"]):
            reasons.append("money_in")
        if bool(feat_row["accumulation"]):
            reasons.append("accumulation")

        open_pos = Trade(
            symbol=symbol,
            setup_type=regime,          # BREAKOUT_FLOW | MONEY_IN | ACCUMULATION
            signal_date=date,
            entry_date=entry_date,
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=initial_target,
            entry_atr=atr,
            reasons=reasons,
        )
        entry_bar_idx = i + 1
        dynamic_sl    = sl
        peak_close    = entry_price

    # Force-close lệnh còn mở cuối kỳ tại giá close bar cuối
    if open_pos is not None:
        last = df.iloc[-1]
        bars_held = len(df) - 1 - entry_bar_idx
        open_pos.close(_date_str(last["date"]), float(last["close"]), "END_OF_DATA", bars_held)
        trades.append(open_pos)

    print(
        f"[money_flow_bt] {symbol}: "
        f"regime_ok={_dbg_regime_ok} → score_ok={_dbg_score_ok} → "
        f"liq_ok={_dbg_liq_ok} → trend_ok={_dbg_trend_ok} → "
        f"trades={len(trades)}"
    )
    return trades


# ── Universe parallel runner ──────────────────────────────────────────────────

def run_money_flow_universe(
    ohlcv_map: dict[str, pd.DataFrame],
    vnindex_df: Optional[pd.DataFrame] = None,
    entry_regimes: Optional[set[str]] = None,
    min_score: int = 3,
    filter_downtrend: bool = True,
    params: Optional[dict] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    rr_ratio: float = 1.5,
    max_workers: int = 8,
) -> dict[str, list[Trade]]:
    """Chạy walk-forward money flow backtest song song cho nhiều mã."""
    results: dict[str, list[Trade]] = {}

    def _run_one(sym: str, df: pd.DataFrame) -> tuple[str, list[Trade]]:
        try:
            t = run_money_flow_symbol(
                sym, df, vnindex_df, entry_regimes, min_score,
                filter_downtrend, params, from_date, to_date, rr_ratio,
            )
            return sym, t
        except Exception as e:
            print(f"[money_flow_bt] {sym} lỗi: {e}")
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
                print(f"[money_flow_bt] {done}/{len(ohlcv_map)} mã xong — {total} trades")

    total = sum(len(t) for t in results.values())
    print(f"[money_flow_bt] Xong: {len(results)} mã, {total} trades")
    return results


# ── Metrics bổ sung cho money flow ───────────────────────────────────────────

def compute_regime_breakdown(trades: list[Trade]) -> list[dict]:
    """
    Tính stats phân theo regime (BREAKOUT_FLOW, MONEY_IN, ACCUMULATION).
    Dùng setup_type = regime name từ run_money_flow_symbol.
    """
    from multiagents_trading_assistant.backtest.metrics import compute_metrics
    from collections import defaultdict

    by_regime: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        if t.is_closed:
            by_regime[t.setup_type].append(t)

    rows = []
    for regime, rt in by_regime.items():
        m = compute_metrics(rt, label=regime)
        # Thêm exit reason breakdown
        dist_exits   = sum(1 for t in rt if t.exit_reason == "DIST_EXIT")
        m["dist_exit_count"] = dist_exits
        m["dist_exit_pct"]   = round(dist_exits / len(rt) * 100, 1) if rt else 0.0
        rows.append(m)

    rows.sort(key=lambda r: r.get("total_trades", 0), reverse=True)
    return rows


def print_money_flow_report(
    trades: list[Trade],
    label: str = "MONEY_FLOW",
    from_date: str = "",
    to_date: str = "",
    show_symbol_breakdown: bool = False,
) -> None:
    """In báo cáo backtest money flow ra console."""
    from multiagents_trading_assistant.backtest.report import print_report

    # Dùng lại print_report hiện có (setup breakdown = regime breakdown)
    print_report(
        trades,
        label=f"MONEY_FLOW/{label}",
        from_date=from_date,
        to_date=to_date,
        show_symbol_breakdown=show_symbol_breakdown,
    )

    # Thêm exit reason chi tiết
    regime_rows = compute_regime_breakdown(trades)
    if regime_rows:
        sep = "─" * 62
        print(f"  {sep}")
        print("  DISTRIBUTION EXIT BREAKDOWN (per regime)")
        print(f"    {'Regime':<18} {'Trades':>6} {'DistExit':>9} {'DistExit%':>10}")
        print("    " + "─" * 48)
        for r in regime_rows:
            print(
                f"    {r['label']:<18} {r['total_trades']:>6} "
                f"{r['dist_exit_count']:>9} {r['dist_exit_pct']:>9.1f}%"
            )
        print()
