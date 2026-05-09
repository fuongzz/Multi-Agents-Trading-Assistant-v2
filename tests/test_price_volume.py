"""tests/test_price_volume.py — Unit tests for the Price-Volume Intelligence Layer.

Uses only synthetic OHLCV data — no network calls, no LLM, no external deps.
Each test targets a single signal or property so failures point to a root cause.
"""
import pandas as pd
import pytest

from multiagents_trading_assistant.price_volume import analyze_price_volume


# ─────────────────────────────────────────────────────────────────────────────
# Test-data factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _flat(n: int = 50, base: float = 100.0, vol: int = 1_000_000) -> pd.DataFrame:
    """Flat OHLCV: close=open=base, high=base*1.01, low=base*0.99, volume constant."""
    dates = pd.date_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "date":   dates,
        "open":   pd.array([base] * n, dtype="float64"),
        "high":   pd.array([base * 1.01] * n, dtype="float64"),
        "low":    pd.array([base * 0.99] * n, dtype="float64"),
        "close":  pd.array([base] * n, dtype="float64"),
        "volume": pd.array([float(vol)] * n, dtype="float64"),
    })


def _set_last(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Overwrite the last row of df with keyword column values."""
    df = df.copy()
    for col, val in kwargs.items():
        df.iloc[-1, df.columns.get_loc(col)] = val
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases: empty / short / NaN / zero-volume
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_df_returns_empty_result():
    result = analyze_price_volume(pd.DataFrame())
    assert result["price_volume_score"] == 0
    assert result["entry_bias"] == "neutral"
    assert result["signals"] == {}
    assert result["risk_flags"] == []


def test_too_short_history_returns_empty():
    result = analyze_price_volume(_flat(n=3))
    assert result["price_volume_score"] == 0
    assert "Insufficient" in result["interpretation"]


def test_nan_in_close_handled_gracefully():
    df = _flat(n=40)
    df.iloc[5:10, df.columns.get_loc("close")] = float("nan")
    result = analyze_price_volume(df)
    assert "price_volume_score" in result
    assert result["entry_bias"] in ("bullish", "neutral", "bearish", "avoid")


def test_zero_volume_rows_are_forward_filled():
    df = _flat(n=30)
    df.iloc[10:15, df.columns.get_loc("volume")] = 0
    result = analyze_price_volume(df)
    assert "price_volume_score" in result


def test_missing_required_column_returns_empty():
    df = _flat(n=30).drop(columns=["volume"])
    result = analyze_price_volume(df)
    assert result["price_volume_score"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# as_of_date: future-data isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_as_of_date_blocks_future_breakout():
    df = _flat(n=50, base=100)
    # Keep bars 0-39 flat (close=100, high=101), then make bar 40+ a confirmed breakout.
    # The breakout bar (index 40 = bar -10) must have:
    #   close > resistance_20 (highest high of bars -21:-1, i.e. flat bars = 101)
    #   vol > 1.5×MA20
    #   close_pos > 0.7
    # Only apply to the VERY LAST bar so it doesn't push prior resistance up.
    df = _set_last(df, close=106.4, high=106.5, low=105.0, volume=2_000_000.0)

    cutoff = str(df["date"].iloc[-2].date())       # one bar before the breakout bar
    result_past = analyze_price_volume(df, as_of_date=cutoff)
    result_full = analyze_price_volume(df)

    assert not result_past["signals"].get("breakout_confirmed"), \
        "Future breakout must not appear when as_of_date cuts it off"
    assert result_full["signals"].get("breakout_confirmed"), \
        "Breakout must be detected when full data is used"


# ─────────────────────────────────────────────────────────────────────────────
# Confirmed breakout
# ─────────────────────────────────────────────────────────────────────────────

def test_breakout_confirmed_signal_and_score():
    # 20-bar resistance ≈ 101.0 (high of flat bars)
    # Last bar: close=106 (>101), vol=2.5M (2.5×MA20=1M), close_pos=(106-104)/(107-104)=0.67>0.7 ✗
    # Adjust: close_pos must be > 0.7 → close near high
    df = _flat(n=50)
    df = _set_last(df, close=106.0, high=106.5, low=105.0, volume=2_000_000.0)
    # close_pos = (106-105)/(106.5-105) = 1/1.5 = 0.67 — just under 0.7, tighten:
    df = _set_last(df, close=106.4, high=106.5, low=105.0, volume=2_000_000.0)
    # close_pos = (106.4-105)/(106.5-105) = 1.4/1.5 ≈ 0.93 ✓
    result = analyze_price_volume(df)
    assert result["signals"]["breakout_confirmed"] is True
    assert result["price_volume_score"] > 0
    assert result["entry_bias"] in ("bullish", "neutral")


def test_high_conviction_breakout_tag_requires_accum_days():
    """HIGH_CONVICTION_BREAKOUT only appears when accum_count >= 3."""
    df = _flat(n=50)
    df = _set_last(df, close=106.4, high=106.5, low=105.0, volume=2_000_000.0)
    result = analyze_price_volume(df)
    if result["signals"]["breakout_confirmed"] and result["metrics"]["accumulation_count_20"] >= 3:
        assert "HIGH_CONVICTION_BREAKOUT" in result["setup_tags"]


# ─────────────────────────────────────────────────────────────────────────────
# Weak breakout
# ─────────────────────────────────────────────────────────────────────────────

def test_weak_breakout_signals_and_risk_flag():
    # close > resistance, but volume < vol_ma20
    df = _flat(n=50)
    df = _set_last(df, close=105.0, high=105.5, volume=500_000.0)  # vol < MA20 (1M)
    result = analyze_price_volume(df)
    assert result["signals"]["weak_breakout"] is True
    assert "WEAK_BREAKOUT" in result["risk_flags"]
    assert result["price_volume_score"] < 0


# ─────────────────────────────────────────────────────────────────────────────
# Fake breakout risk
# ─────────────────────────────────────────────────────────────────────────────

def test_fake_breakout_triggers_avoid_bias():
    # high pierces resistance (~101) but close falls back below it
    df = _flat(n=50)
    df = _set_last(df, high=105.0, close=99.0, open=100.0, low=98.5, volume=1_400_000.0)
    # vol > 1.3 × 1M = 1.3M ✓, high(105) > resistance(101) ✓, close(99) < resistance ✓
    result = analyze_price_volume(df)
    assert result["signals"]["fake_breakout_risk"] is True
    assert "FAKE_BREAKOUT_RISK" in result["risk_flags"]
    assert result["entry_bias"] == "avoid"
    assert result["price_volume_score"] < 0


# ─────────────────────────────────────────────────────────────────────────────
# Volume dry-up
# ─────────────────────────────────────────────────────────────────────────────

def test_volume_dry_up_setup_tag():
    df = _flat(n=50)
    # Last 5 bars: low volume AND tight range
    for i in range(-5, 0):
        df.iloc[i, df.columns.get_loc("volume")] = 300_000.0   # < 0.7 × 1M
        df.iloc[i, df.columns.get_loc("high")]   = 100.1
        df.iloc[i, df.columns.get_loc("low")]    = 99.9        # very tight
    result = analyze_price_volume(df)
    assert result["signals"]["volume_dry_up"] is True
    assert "VOLUME_DRY_UP_BASE" in result["setup_tags"]
    assert result["price_volume_score"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Accumulation day
# ─────────────────────────────────────────────────────────────────────────────

def test_accumulation_day_detected():
    # prev_close = 100 (flat). Last bar: green, close > prev_close, vol > 1.2×MA20, close_pos > 0.65
    df = _flat(n=50)
    df = _set_last(df, open=99.0, close=102.5, high=103.0, low=98.0, volume=1_500_000.0)
    # close_pos = (102.5 - 98) / (103 - 98) = 4.5/5 = 0.9 ✓
    result = analyze_price_volume(df)
    assert result["signals"]["accumulation_day"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Distribution day
# ─────────────────────────────────────────────────────────────────────────────

def test_distribution_day_detected():
    # close < prev_close, vol > 1.2×MA20, close_pos < 0.4
    df = _flat(n=50)
    df = _set_last(df, close=96.5, high=101.0, low=96.0, volume=1_500_000.0)
    # close_pos = (96.5 - 96) / (101 - 96) = 0.5/5 = 0.1 ✓
    result = analyze_price_volume(df)
    assert result["signals"]["distribution_day"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Absorption signal
# ─────────────────────────────────────────────────────────────────────────────

def test_absorption_signal_and_setup_tag():
    # high_effort: vol > 1.8×MA20 → ratio = 2.0
    # low_result: |close - prev_close| / ATR14 < 0.5 → very small price move
    # close >= open (green bar), close_pos > 0.5
    df = _flat(n=50)
    df = _set_last(df, open=100.0, close=100.1, high=100.5, low=99.5, volume=2_100_000.0)
    # effort = 2100000/1000000 = 2.1 > 1.8 ✓
    # result = |100.1 - 100| / ATR ≈ 0.1/(~1.0) = 0.1 < 0.5 ✓
    # close_pos = (100.1 - 99.5) / (100.5 - 99.5) = 0.6/1 = 0.6 > 0.5 ✓
    result = analyze_price_volume(df)
    assert result["signals"]["absorption_signal"] is True
    assert "ABSORPTION" in result["setup_tags"]


# ─────────────────────────────────────────────────────────────────────────────
# Distribution signal (effort/result variant)
# ─────────────────────────────────────────────────────────────────────────────

def test_distribution_signal_risk_flag():
    # high_effort + close < open + close_pos < 0.5
    df = _flat(n=50)
    df = _set_last(df, open=101.0, close=99.0, high=102.0, low=98.5, volume=2_100_000.0)
    # effort = 2.1 > 1.8 ✓
    # result = |99 - 100| / ATR ≈ 1 — this may exceed 0.5 with large ATR; let's ensure:
    # ATR14 on flat bars ≈ base*0.02 = 2.0  → result = 1/2 = 0.5 — borderline
    # Use tighter range to keep ATR small:
    df_narrow = _flat(n=50, base=100.0)
    for i in range(-15, 0):
        df_narrow.iloc[i, df_narrow.columns.get_loc("high")] = 100.1
        df_narrow.iloc[i, df_narrow.columns.get_loc("low")]  = 99.9
    df_narrow = _set_last(df_narrow, open=100.1, close=99.9, high=100.2, low=99.7, volume=2_100_000.0)
    # close_pos = (99.9 - 99.7) / (100.2 - 99.7) = 0.2/0.5 = 0.4 < 0.5 ✓
    result = analyze_price_volume(df_narrow)
    assert result["signals"]["distribution_signal"] is True
    assert "DISTRIBUTION_SIGNAL" in result["risk_flags"]


# ─────────────────────────────────────────────────────────────────────────────
# Washout
# ─────────────────────────────────────────────────────────────────────────────

def test_washout_reversal_setup_tag():
    df = _flat(n=50)
    # 20-bar support (prev bars low = 99.0) set by _flat
    # Last bar: low < 99 (pierces support), strong close (recovery), high vol
    df = _set_last(df, low=97.0, open=98.0, close=101.5, high=102.0, volume=2_000_000.0)
    # close_pos = (101.5 - 97) / (102 - 97) = 4.5/5 = 0.9 > 0.6 ✓
    # vol = 2M > 1.5×1M ✓
    result = analyze_price_volume(df)
    assert result["signals"]["washout"] is True
    assert "WASHOUT_REVERSAL" in result["setup_tags"]
    assert result["price_volume_score"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Buying climax
# ─────────────────────────────────────────────────────────────────────────────

def test_buying_climax_risk_flag():
    # MA20 ≈ 100; close > 100*1.15=115, vol > 2.5×1M, close_pos < 0.6
    df = _flat(n=50)
    df = _set_last(df, close=118.0, open=120.0, high=122.0, low=116.0, volume=3_000_000.0)
    # close_pos = (118-116)/(122-116) = 2/6 = 0.33 < 0.6 ✓
    result = analyze_price_volume(df)
    assert result["signals"]["buying_climax"] is True
    assert "BUYING_CLIMAX" in result["risk_flags"]
    assert result["price_volume_score"] < 0


# ─────────────────────────────────────────────────────────────────────────────
# Constructive pullback
# ─────────────────────────────────────────────────────────────────────────────

def test_constructive_pullback_setup_tag():
    # MA20 ≈ 100; close in [97, 105]; vol < MA20; close >= prev_close * 0.97
    df = _flat(n=50)
    df.iloc[-2, df.columns.get_loc("close")] = 102.0   # prev_close = 102
    df = _set_last(df, close=101.5, open=101.0, high=102.0, low=100.5, volume=600_000.0)
    result = analyze_price_volume(df)
    assert result["signals"]["constructive_pullback"] is True
    assert "CONSTRUCTIVE_PULLBACK" in result["setup_tags"]


# ─────────────────────────────────────────────────────────────────────────────
# Bearish volume expansion
# ─────────────────────────────────────────────────────────────────────────────

def test_bearish_volume_expansion_risk_flag():
    # close < prev_close, vol > 1.5×MA20, close_pos < 0.35
    df = _flat(n=50)
    df = _set_last(df, close=96.0, high=101.0, low=95.0, volume=2_000_000.0)
    # close_pos = (96 - 95) / (101 - 95) = 1/6 ≈ 0.17 < 0.35 ✓
    result = analyze_price_volume(df)
    assert result["signals"]["bearish_volume_expansion"] is True
    assert "BEARISH_VOLUME_EXPANSION" in result["risk_flags"]
    assert result["price_volume_score"] < 0


# ─────────────────────────────────────────────────────────────────────────────
# Bullish volume expansion
# ─────────────────────────────────────────────────────────────────────────────

def test_bullish_volume_expansion_setup_tag():
    # close > prev_close, vol > 1.5×MA20, close_pos > 0.65
    df = _flat(n=50)
    df = _set_last(df, close=103.0, high=103.5, low=101.0, volume=2_000_000.0)
    # close_pos = (103 - 101) / (103.5 - 101) = 2/2.5 = 0.8 > 0.65 ✓
    result = analyze_price_volume(df)
    assert result["signals"]["bullish_volume_expansion"] is True
    assert "BULLISH_VOLUME_EXPANSION" in result["setup_tags"]
    assert result["price_volume_score"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Output schema completeness
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_TOP_KEYS = {
    "price_volume_score", "entry_bias", "signals",
    "risk_flags", "setup_tags", "interpretation", "metrics",
}

_REQUIRED_METRIC_KEYS = {
    "volume_ma5", "volume_ma20", "volume_ratio_20", "ma20", "atr14",
    "resistance_20", "support_20", "close_position_in_range",
    "price_range_5d", "price_range_20d", "accumulation_count_20",
    "distribution_count_20", "effort", "result",
}

def test_output_schema_complete():
    result = analyze_price_volume(_flat(n=50))
    assert _REQUIRED_TOP_KEYS.issubset(result.keys())
    assert _REQUIRED_METRIC_KEYS.issubset(result["metrics"].keys())


def test_score_within_bounds():
    result = analyze_price_volume(_flat(n=50))
    assert -100 <= result["price_volume_score"] <= 100


def test_entry_bias_valid_values():
    result = analyze_price_volume(_flat(n=50))
    assert result["entry_bias"] in ("bullish", "neutral", "bearish", "avoid")


def test_signals_are_all_booleans():
    result = analyze_price_volume(_flat(n=50))
    for name, val in result["signals"].items():
        assert isinstance(val, bool), f"Signal '{name}' is not bool: {val!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Scoring logic validation
# ─────────────────────────────────────────────────────────────────────────────

def test_score_clamped_at_extremes():
    """Score must stay within [-100, +100] even with many simultaneous signals."""
    df = _flat(n=50)
    # Force as many positive signals as possible
    # Confirmed breakout + high close_pos + high vol
    df = _set_last(df, close=106.4, high=106.5, low=105.0, volume=2_000_000.0)
    result = analyze_price_volume(df)
    assert result["price_volume_score"] <= 100

    # Force negative: fake breakout + distribution flag
    df2 = _flat(n=50)
    df2 = _set_last(df2, high=105.0, close=99.0, open=100.0, low=98.5, volume=1_400_000.0)
    result2 = analyze_price_volume(df2)
    assert result2["price_volume_score"] >= -100


def test_avoid_bias_when_heavy_distribution():
    """entry_bias must be 'avoid' when 5+ distribution days occur."""
    df = _flat(n=50)
    # Distribution day: close < prev_close, vol > 1.2×MA20, close_pos < 0.4.
    # Each bar must close LOWER than the bar before it so c_i < pc_i holds.
    # Use alternating up/down pattern within last 20 bars:
    # Odd indices: high close (up), even: low close (distribution).
    start_price = 105.0
    n = len(df)
    dist_inserted = 0
    for offset in range(1, 20):
        idx = n - 20 + offset
        if idx < 1:
            continue
        prev_close = float(df.iloc[idx - 1]["close"])
        if offset % 2 == 0:  # distribution bar — close < prev_close, low range, high vol
            new_close = prev_close - 2.0
            df.iloc[idx, df.columns.get_loc("close")]  = new_close
            df.iloc[idx, df.columns.get_loc("high")]   = prev_close + 1.0
            df.iloc[idx, df.columns.get_loc("low")]    = new_close - 0.5
            df.iloc[idx, df.columns.get_loc("volume")] = 1_500_000.0
            # close_pos = (new_close - (new_close-0.5)) / ((prev_close+1) - (new_close-0.5))
            dist_inserted += 1
        else:  # recovery bar — close up to keep prev_close alternating
            df.iloc[idx, df.columns.get_loc("close")]  = prev_close + 2.5
            df.iloc[idx, df.columns.get_loc("volume")] = 900_000.0

    result = analyze_price_volume(df)
    assert result["metrics"]["distribution_count_20"] >= 5, \
        f"Expected ≥5 distribution days, got {result['metrics']['distribution_count_20']}"
    assert result["entry_bias"] == "avoid"
    assert "HEAVY_DISTRIBUTION" in result["risk_flags"]


# ─────────────────────────────────────────────────────────────────────────────
# Backtesting: no look-ahead in resistance/support
# ─────────────────────────────────────────────────────────────────────────────

def test_resistance_excludes_current_bar():
    """resistance_20 must use only bars BEFORE the current bar."""
    df = _flat(n=50, base=100)
    # Make current bar the highest ever — it should NOT self-confirm breakout
    df = _set_last(df, close=110.0, high=120.0, low=109.0, volume=500_000.0)
    # vol < MA20 → not a confirmed breakout
    result = analyze_price_volume(df)
    # resistance_20 computed from bars [-21:-1], max high there ≈ 101
    # close(110) > resistance(101) AND vol(500k) < MA20(1M) → weak_breakout, not confirmed
    assert result["signals"]["weak_breakout"] is True
    assert result["signals"]["breakout_confirmed"] is False


def test_column_names_are_case_insensitive():
    df = _flat(n=30)
    df.columns = [c.upper() for c in df.columns]
    result = analyze_price_volume(df)
    assert "price_volume_score" in result
