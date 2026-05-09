"""price_volume.py — Deterministic Price-Volume Intelligence Layer.

Pure rule-based, testable, backtestable, zero LLM.
Detects accumulation, distribution, breakout quality, fake-breakout risk,
volume dry-up, washout, buying climax, and effort-vs-result behaviour.

Input:  OHLCV DataFrame (date/open/high/low/close/volume columns).
Output: dict with price_volume_score (-100..+100), entry_bias, signals,
        risk_flags, setup_tags, interpretation, metrics.

All rolling windows look only at data up to the current bar (no look-ahead).
Resistance/support use the *previous* 20 bars, excluding the current bar,
so the current bar cannot self-confirm a breakout.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

_EMPTY: dict = {
    "price_volume_score": 0,
    "entry_bias": "neutral",
    "signals": {},
    "risk_flags": [],
    "setup_tags": [],
    "interpretation": "Insufficient data for price-volume analysis.",
    "metrics": {},
}


def analyze_price_volume(
    df: pd.DataFrame,
    as_of_date: Optional[str] = None,
) -> dict:
    """Compute the Price-Volume Intelligence snapshot for the most recent bar.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns open, high, low, close, volume.
        Any date column (named 'date' or 'time') is used for as_of_date filtering.
    as_of_date : str, optional
        ISO date string (YYYY-MM-DD).  Only bars on or before this date are used.
        Prevents future-data leakage in backtesting.

    Returns
    -------
    dict with keys:
        price_volume_score  int     -100 … +100
        entry_bias          str     "bullish" | "neutral" | "bearish" | "avoid"
        signals             dict    each signal name → bool
        risk_flags          list    active risk-flag strings
        setup_tags          list    active setup-tag strings
        interpretation      str     short deterministic explanation
        metrics             dict    raw computed numeric metrics
    """
    df = _prepare(df, as_of_date)
    if df is None or len(df) < 5:
        return _EMPTY.copy()

    try:
        return _compute(df)
    except Exception as exc:  # noqa: BLE001
        return {**_EMPTY, "interpretation": f"Computation error: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def _prepare(df: pd.DataFrame, as_of_date: Optional[str]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None

    # Apply as_of_date filter — no future bars in backtest
    if as_of_date:
        date_col = next((c for c in ("date", "time") if c in df.columns), None)
        if date_col:
            df = df[df[date_col].astype(str) <= as_of_date]

    # Sort ascending so iloc[-1] is always the latest bar
    for col in ("date", "time"):
        if col in df.columns:
            try:
                df = df.sort_values(col)
            except Exception:
                pass
            break

    # Coerce to float
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(required))

    # Zero-volume rows are meaningless — forward-fill from last valid bar
    df = df.copy()
    df.loc[df["volume"] == 0, "volume"] = float("nan")
    df["volume"] = df["volume"].ffill()
    df = df.dropna(subset=["volume"])

    return df if len(df) >= 5 else None


# ─────────────────────────────────────────────────────────────────────────────
# Core computation (vectorisation-friendly, bar-by-bar safe)
# ─────────────────────────────────────────────────────────────────────────────

def _compute(df: pd.DataFrame) -> dict:  # noqa: C901 — structured sequentially by design
    close  = df["close"].reset_index(drop=True)
    high   = df["high"].reset_index(drop=True)
    low    = df["low"].reset_index(drop=True)
    volume = df["volume"].reset_index(drop=True)
    open_  = df["open"].reset_index(drop=True)

    n = len(df)

    # ── Latest-bar scalars ────────────────────────────────────────────────
    cur_close  = float(close.iloc[-1])
    cur_open   = float(open_.iloc[-1])
    cur_high   = float(high.iloc[-1])
    cur_low    = float(low.iloc[-1])
    cur_vol    = float(volume.iloc[-1])
    prev_close = float(close.iloc[-2]) if n >= 2 else cur_close

    # ── close_position_in_range ───────────────────────────────────────────
    h_l = cur_high - cur_low
    close_pos = 0.5 if h_l == 0 else (cur_close - cur_low) / h_l

    # ── Volume moving averages ────────────────────────────────────────────
    vol_ma5  = float(volume.rolling(5).mean().iloc[-1])  if n >= 5  else cur_vol
    vol_ma20 = float(volume.rolling(20).mean().iloc[-1]) if n >= 20 else float(volume.mean())
    vol_ma20 = vol_ma20 if vol_ma20 > 0 else 1.0

    volume_ratio_20 = cur_vol / vol_ma20

    # ── MA20 (price) ──────────────────────────────────────────────────────
    ma20 = float(close.rolling(20).mean().iloc[-1]) if n >= 20 else float(close.mean())

    # ── ATR-14 (standard true range, backward-looking only) ──────────────
    if n >= 2:
        prev_closes = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_closes).abs(),
            (low  - prev_closes).abs(),
        ], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1]) if n >= 14 else float(tr.dropna().mean() or h_l)
    else:
        atr14 = h_l
    atr14 = atr14 if atr14 > 0 else 1.0

    # ── Rolling resistance / support (exclude current bar) ───────────────
    # Use the 20 bars *before* the current bar so the current candle cannot
    # self-confirm a breakout — critical for look-ahead protection.
    if n >= 22:
        resistance_20 = float(high.iloc[-21:-1].max())
        support_20    = float(low.iloc[-21:-1].min())
    elif n >= 3:
        resistance_20 = float(high.iloc[:-1].max())
        support_20    = float(low.iloc[:-1].min())
    else:
        resistance_20 = cur_high
        support_20    = cur_low

    # ── Price ranges ──────────────────────────────────────────────────────
    price_range_5d  = float(high.iloc[-5:].max()  - low.iloc[-5:].min())  if n >= 5  else h_l
    price_range_20d = float(high.iloc[-20:].max() - low.iloc[-20:].min()) if n >= 20 else price_range_5d
    price_range_20d = price_range_20d if price_range_20d > 0 else 1.0

    # ── Accumulation / Distribution counts over the last 20 complete bars ─
    # Each bar uses only rolling vol MA of bars up to and including itself —
    # no forward reference.
    accum_count = 0
    distr_count = 0
    start_idx = max(1, n - 20)
    for i in range(start_idx, n):
        o_i   = float(open_.iloc[i])
        c_i   = float(close.iloc[i])
        h_i   = float(high.iloc[i])
        l_i   = float(low.iloc[i])
        v_i   = float(volume.iloc[i])
        pc_i  = float(close.iloc[i - 1])
        hl_i  = h_i - l_i
        cp_i  = 0.5 if hl_i == 0 else (c_i - l_i) / hl_i
        # vol MA20 at bar i — only backward-looking bars [max(0,i-19)…i]
        v_ma20_i = float(volume.iloc[max(0, i - 19): i + 1].mean())

        if c_i > o_i and c_i > pc_i and v_i > 1.2 * v_ma20_i and cp_i > 0.65:
            accum_count += 1
        if c_i < pc_i and v_i > 1.2 * v_ma20_i and cp_i < 0.4:
            distr_count += 1

    # ── Effort / Result ───────────────────────────────────────────────────
    effort = volume_ratio_20
    result = abs(cur_close - prev_close) / atr14

    # ─────────────────────────────────────────────────────────────────────
    # Signal evaluation
    # ─────────────────────────────────────────────────────────────────────

    sig: dict[str, bool] = {}

    sig["breakout_confirmed"] = (
        cur_close > resistance_20
        and cur_vol > 1.5 * vol_ma20
        and close_pos > 0.7
    )

    sig["weak_breakout"] = (
        cur_close > resistance_20
        and cur_vol < vol_ma20
    )

    sig["fake_breakout_risk"] = (
        cur_high > resistance_20
        and cur_close < resistance_20
        and cur_vol > 1.3 * vol_ma20
    )

    sig["volume_dry_up"] = (
        vol_ma5 < 0.7 * vol_ma20
        and price_range_5d < 0.5 * price_range_20d
    )

    sig["accumulation_day"] = (
        cur_close > cur_open
        and cur_close > prev_close
        and cur_vol > 1.2 * vol_ma20
        and close_pos > 0.65
    )

    sig["distribution_day"] = (
        cur_close < prev_close
        and cur_vol > 1.2 * vol_ma20
        and close_pos < 0.4
    )

    _helo = effort > 1.8 and result < 0.5      # high effort, low result
    sig["high_effort_low_result"] = _helo

    sig["absorption_signal"] = (
        _helo
        and cur_close >= cur_open
        and close_pos > 0.5
    )

    sig["distribution_signal"] = (
        _helo
        and cur_close < cur_open
        and close_pos < 0.5
    )

    sig["washout"] = (
        cur_low < support_20
        and cur_close > cur_open
        and close_pos > 0.6
        and cur_vol > 1.5 * vol_ma20
    )

    sig["buying_climax"] = (
        cur_close > ma20 * 1.15
        and cur_vol > 2.5 * vol_ma20
        and close_pos < 0.6
    )

    sig["constructive_pullback"] = (
        ma20 * 0.97 <= cur_close <= ma20 * 1.05
        and cur_vol < vol_ma20
        and cur_close >= prev_close * 0.97
    )

    sig["bearish_volume_expansion"] = (
        cur_close < prev_close
        and cur_vol > 1.5 * vol_ma20
        and close_pos < 0.35
    )

    sig["bullish_volume_expansion"] = (
        cur_close > prev_close
        and cur_vol > 1.5 * vol_ma20
        and close_pos > 0.65
    )

    # ─────────────────────────────────────────────────────────────────────
    # Scoring
    # ─────────────────────────────────────────────────────────────────────

    score = 0

    # Positive contributors
    if sig["breakout_confirmed"]:       score += 25
    if accum_count >= 5:               score += 25
    elif accum_count >= 3:             score += 15
    if sig["volume_dry_up"]:           score += 10
    if sig["washout"]:                 score += 15
    if sig["absorption_signal"]:       score += 10
    if sig["constructive_pullback"]:   score += 10
    if sig["bullish_volume_expansion"]: score += 15

    # Negative contributors
    if sig["weak_breakout"]:           score -= 15
    if sig["fake_breakout_risk"]:      score -= 25
    if distr_count >= 5:               score -= 35
    elif distr_count >= 3:             score -= 20
    if sig["buying_climax"]:           score -= 25
    if sig["distribution_signal"]:     score -= 15
    if sig["bearish_volume_expansion"]: score -= 20

    score = max(-100, min(100, score))

    # ─────────────────────────────────────────────────────────────────────
    # Entry bias
    # ─────────────────────────────────────────────────────────────────────

    if distr_count >= 5 or sig["fake_breakout_risk"]:
        entry_bias = "avoid"
    elif score >= 35:
        entry_bias = "bullish"
    elif score <= -25:
        entry_bias = "bearish"
    else:
        entry_bias = "neutral"

    # ─────────────────────────────────────────────────────────────────────
    # Risk flags
    # ─────────────────────────────────────────────────────────────────────

    risk_flags: list[str] = []
    if sig["fake_breakout_risk"]:        risk_flags.append("FAKE_BREAKOUT_RISK")
    if distr_count >= 5:                 risk_flags.append("HEAVY_DISTRIBUTION")
    if sig["buying_climax"]:             risk_flags.append("BUYING_CLIMAX")
    if sig["bearish_volume_expansion"]:  risk_flags.append("BEARISH_VOLUME_EXPANSION")
    if sig["weak_breakout"]:             risk_flags.append("WEAK_BREAKOUT")
    if sig["distribution_signal"]:       risk_flags.append("DISTRIBUTION_SIGNAL")

    # ─────────────────────────────────────────────────────────────────────
    # Setup tags
    # ─────────────────────────────────────────────────────────────────────

    setup_tags: list[str] = []
    if sig["breakout_confirmed"] and accum_count >= 3:
        setup_tags.append("HIGH_CONVICTION_BREAKOUT")
    if sig["volume_dry_up"]:
        setup_tags.append("VOLUME_DRY_UP_BASE")
    if sig["washout"]:
        setup_tags.append("WASHOUT_REVERSAL")
    if sig["absorption_signal"]:
        setup_tags.append("ABSORPTION")
    if sig["constructive_pullback"]:
        setup_tags.append("CONSTRUCTIVE_PULLBACK")
    if sig["bullish_volume_expansion"]:
        setup_tags.append("BULLISH_VOLUME_EXPANSION")

    # ─────────────────────────────────────────────────────────────────────
    # Interpretation (deterministic)
    # ─────────────────────────────────────────────────────────────────────

    interpretation = _interpret(sig, score, entry_bias, accum_count, distr_count)

    # ─────────────────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────────────────

    metrics: dict = {
        "volume_ma5":               round(vol_ma5),
        "volume_ma20":              round(vol_ma20),
        "volume_ratio_20":          round(volume_ratio_20, 3),
        "ma20":                     round(ma20, 2),
        "atr14":                    round(atr14, 4),
        "resistance_20":            round(resistance_20, 2),
        "support_20":               round(support_20, 2),
        "close_position_in_range":  round(close_pos, 4),
        "price_range_5d":           round(price_range_5d, 2),
        "price_range_20d":          round(price_range_20d, 2),
        "accumulation_count_20":    accum_count,
        "distribution_count_20":    distr_count,
        "effort":                   round(effort, 3),
        "result":                   round(result, 3),
    }

    return {
        "price_volume_score": int(score),
        "entry_bias":         entry_bias,
        "signals":            sig,
        "risk_flags":         risk_flags,
        "setup_tags":         setup_tags,
        "interpretation":     interpretation,
        "metrics":            metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Interpretation builder — no LLM, fully deterministic
# ─────────────────────────────────────────────────────────────────────────────

def _interpret(
    sig: dict[str, bool],
    score: int,
    entry_bias: str,
    accum_count: int,
    distr_count: int,
) -> str:
    if entry_bias == "avoid":
        if sig.get("fake_breakout_risk"):
            return (
                "Avoid new long entries: fake breakout risk — "
                "price was rejected at resistance on high volume."
            )
        return (
            f"Avoid new long entries: heavy distribution detected "
            f"({distr_count} distribution bars in the last 20 sessions)."
        )

    if entry_bias == "bearish":
        parts: list[str] = []
        if sig.get("bearish_volume_expansion"):
            parts.append("bearish volume expansion")
        if sig.get("distribution_signal"):
            parts.append("distribution signal (high effort, weak close)")
        if sig.get("buying_climax"):
            parts.append("buying climax — exhaustion likely")
        if distr_count >= 3:
            parts.append(f"{distr_count} distribution days in 20 bars")
        reason = "; ".join(parts) or "net negative price-volume pressure"
        return f"Risky price-volume structure: {reason}."

    if entry_bias == "bullish":
        parts = []
        if sig.get("breakout_confirmed"):
            parts.append("confirmed breakout with volume surge")
        if accum_count >= 3:
            parts.append(f"{accum_count} accumulation days in 20 bars")
        if sig.get("washout"):
            parts.append("washout reversal (shakeout then recovery)")
        if sig.get("absorption_signal"):
            parts.append("absorption (supply absorbed on high volume)")
        if sig.get("constructive_pullback"):
            parts.append("constructive pullback near MA20")
        if sig.get("bullish_volume_expansion"):
            parts.append("bullish volume expansion")
        if sig.get("volume_dry_up"):
            parts.append("volume dry-up base (selling pressure fading)")
        reason = "; ".join(parts) or "positive price-volume balance"
        return f"Constructive price-volume structure: {reason}."

    return "Neutral price-volume structure: no strong confirmation from volume."
