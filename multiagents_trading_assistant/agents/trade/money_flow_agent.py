"""
money_flow_agent.py — QMV-inspired Blackbox Money Flow engine.

Rule-based, deterministic. No LLM. Implements blackbox_spec.md in full.

LangGraph wiring (trade_graph.py run_analysts):
    state["money_flow_analysis"] = analyze(symbol, date, market_context, sector_context)

Risk gate (risk_trade.py):
    mfa = state.get("money_flow_analysis", {})
    if mfa.get("regime") == "DISTRIBUTION" or mfa.get("action_bias") == "AVOID_OR_EXIT":
        # block new buys (sell/exit still allowed)

Synthesis use:
    BUY_CANDIDATE  → add bullish confluence
    WATCHLIST      → hold unless other setup confirms
    AVOID_OR_EXIT  → add blocker; risk node may override
    NEUTRAL        → no strong impact
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from multiagents_trading_assistant import fetcher

logger = logging.getLogger(__name__)

# ── Default parameters ────────────────────────────────────────────────────────

BLACKBOX_DEFAULTS: dict = {
    "min_avg_value":             20_000_000_000,  # VND 20B/day — mid-cap liquid threshold
    "price_unit_multiplier":     "auto",          # vnstock often returns stock prices in kVND
    "volume_window":             20,
    "value_window":              20,
    "zscore_window":             60,
    "resistance_window":         20,
    "base_window":               20,
    "long_window":               120,
    "money_in_value_ratio":      1.5,
    "breakout_value_ratio":      1.8,
    "distribution_value_ratio":  1.8,
    "strong_close":              0.70,
    "weak_close":                0.40,
    "foreign_ratio_threshold":   0.05,
    "early_ret_5d_max":          0.08,
    "early_dist_ma20_max":       0.08,
    "chase_ret_5d_min":          0.12,
    "chase_dist_ma20_min":       0.12,
    "recent_distribution_window": 5,
}


# ── Feature engineering ───────────────────────────────────────────────────────

def _infer_price_unit_multiplier(close: pd.Series, setting: str | int | float = "auto") -> float:
    """Return multiplier that converts quoted close prices to VND/share."""
    if setting != "auto":
        return float(setting)

    median_close = pd.to_numeric(close, errors="coerce").dropna().median()
    if pd.isna(median_close):
        return 1.0

    # VN stock APIs commonly quote equities as 27.75 for 27,750 VND.
    # Index levels are around 1,000+, but money-flow value is only used for stocks.
    return 1000.0 if median_close < 1000 else 1.0

def add_money_flow_features(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Add all money-flow derived columns to an OHLCV DataFrame.

    Accepts either a 'date' column or a DatetimeIndex.
    Returns a copy sorted by date with a DatetimeIndex.
    """
    p  = {**BLACKBOX_DEFAULTS, **(params or {})}
    df = df.copy()

    if "date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    vw = p["value_window"]
    zw = p["zscore_window"]
    bw = p["base_window"]
    lw = p["long_window"]
    rw = p["resistance_window"]

    price_multiplier = _infer_price_unit_multiplier(df["close"], p["price_unit_multiplier"])

    df["value"]    = df["close"] * price_multiplier * df["volume"]
    df["ret_1d"]   = df["close"].pct_change()
    df["ret_5d"]   = df["close"] / df["close"].shift(5) - 1.0
    df["ma20"]     = df["close"].rolling(20).mean()
    df["dist_ma20"] = df["close"] / df["ma20"] - 1.0

    # Liquidity
    df["avg_value_20"]    = df["value"].rolling(vw).mean()

    # Abnormal volume / value
    df["volume_ratio_20"] = df["volume"] / df["volume"].rolling(vw).mean()
    df["value_ratio_20"]  = df["value"]  / df["avg_value_20"]

    v60_mean          = df["volume"].rolling(zw).mean()
    v60_std           = df["volume"].rolling(zw).std().replace(0, np.nan)
    df["volume_z_60"] = (df["volume"] - v60_mean) / v60_std

    val60_mean       = df["value"].rolling(zw).mean()
    val60_std        = df["value"].rolling(zw).std().replace(0, np.nan)
    df["value_z_60"] = (df["value"] - val60_mean) / val60_std

    # Close location: 1.0 = closed at high, 0.0 = closed at low
    spread               = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_position"] = (df["close"] - df["low"]) / spread

    # Accumulation components
    df["range_20"]      = (df["high"].rolling(bw).max() - df["low"].rolling(bw).min()) / df["close"]
    df["range_120_q35"] = df["range_20"].rolling(lw).quantile(0.35)
    df["ret_vol_20"]    = df["ret_1d"].rolling(bw).std()
    df["ret_vol_120"]   = df["ret_1d"].rolling(lw).std()

    # Support / resistance
    df["resistance_20"]  = df["high"].rolling(rw).max().shift(1)
    df["support_60_min"] = df["low"].rolling(60).min()

    return df


# ── Classification ────────────────────────────────────────────────────────────

def classify_money_flow(
    df: pd.DataFrame,
    context: dict,
    foreign_flow: dict | None = None,
    params: dict | None = None,
) -> dict:
    """
    Classify the latest bar of df (must already have features from
    add_money_flow_features) into a money-flow regime and score.

    Returns full blackbox output dict per blackbox_spec.md §Output Schema.
    """
    p    = {**BLACKBOX_DEFAULTS, **(params or {})}
    sym  = context.get("symbol", "UNKNOWN")
    mkt  = context.get("market_context", {})
    sec  = context.get("sector_context", {})

    market_trend          = mkt.get("trend", "SIDEWAY")
    sector_rs             = float(sec.get("relative_strength_20d", 0.0) or 0.0)
    sector_is_outperform  = bool(sec.get("is_outperforming", sector_rs > 0))

    warnings   = []
    ohlcv_days = int(df["close"].notna().sum())

    row = df.iloc[-1]

    def _f(col: str, default: float = 0.0) -> float:
        v = row.get(col)
        if v is None or pd.isna(v):
            return default
        return float(v)

    value_ratio_20  = _f("value_ratio_20",  1.0)
    volume_ratio_20 = _f("volume_ratio_20", 1.0)
    value_z_60      = _f("value_z_60",      0.0)
    close_position  = _f("close_position",  0.5)
    ret_1d          = _f("ret_1d",          0.0)
    ret_5d          = _f("ret_5d",          0.0)
    dist_ma20       = _f("dist_ma20",       0.0)
    avg_value_20    = _f("avg_value_20",    0.0)
    current_close   = _f("close",           0.0)
    current_high    = _f("high",            0.0)
    value_traded    = _f("value",           0.0)

    range_20       = _f("range_20",       np.nan)
    range_120_q35  = _f("range_120_q35",  np.nan)
    ret_vol_20     = _f("ret_vol_20",     np.nan)
    ret_vol_120    = _f("ret_vol_120",    np.nan)
    resistance_20  = _f("resistance_20",  np.nan)
    support_60_min = _f("support_60_min", np.nan)

    mir = p["money_in_value_ratio"]
    bvr = p["breakout_value_ratio"]
    dvr = p["distribution_value_ratio"]
    sc  = p["strong_close"]
    wc  = p["weak_close"]

    # ── Liquidity quality ─────────────────────────────────────────────────────
    liquidity_ok = avg_value_20 >= p["min_avg_value"]
    if not liquidity_ok:
        warnings.append(
            f"avg_value_20 {avg_value_20/1e9:.1f}B < threshold {p['min_avg_value']/1e9:.0f}B"
        )

    # ── Money In / Out ────────────────────────────────────────────────────────
    money_in = bool(
        ret_1d > 0 and value_ratio_20 >= mir and close_position >= 0.60
    )
    strong_money_in = bool(
        ret_1d >= 0.02 and value_ratio_20 >= bvr and close_position >= sc
    )
    money_out = bool(
        ret_1d < 0 and value_ratio_20 >= mir and close_position <= wc
    )
    strong_money_out = bool(
        ret_1d <= -0.02 and value_ratio_20 >= bvr and close_position <= 0.30
    )

    # ── Accumulation ──────────────────────────────────────────────────────────
    rng_valid = not (np.isnan(range_20) or np.isnan(range_120_q35))
    vol_valid = not (np.isnan(ret_vol_20) or np.isnan(ret_vol_120))
    accumulation = bool(
        rng_valid and range_20 < range_120_q35
        and volume_ratio_20 < 0.9
        and vol_valid and ret_vol_20 < ret_vol_120
        and sector_rs >= -0.03
    )
    near_support = (
        not np.isnan(support_60_min)
        and support_60_min > 0
        and current_close <= support_60_min * 1.10
    )

    # ── Distribution ──────────────────────────────────────────────────────────
    failed_breakout = False
    if not np.isnan(resistance_20) and resistance_20 > 0:
        failed_breakout = bool(
            current_high > resistance_20
            and current_close < resistance_20
            and value_ratio_20 >= mir
        )

    distribution = bool(
        money_out
        or failed_breakout
        or (value_ratio_20 >= dvr and close_position <= 0.35 and ret_1d <= 0)
    )

    vr_s = df["value_ratio_20"].fillna(1.0)
    cp_s = df["close_position"].fillna(0.5)
    ret_s = df["ret_1d"].fillna(0.0)
    res_s = df["resistance_20"]
    recent_money_out = (ret_s < 0) & (vr_s >= mir) & (cp_s <= wc)
    recent_failed = res_s.notna() & (res_s > 0) & (df["high"] > res_s) & (df["close"] < res_s) & (vr_s >= mir)
    distribution_s = recent_money_out | recent_failed | ((vr_s >= dvr) & (cp_s <= 0.35) & (ret_s <= 0))
    recent_distribution = bool(
        distribution_s.shift(1)
        .tail(int(p["recent_distribution_window"]))
        .fillna(False)
        .any()
    )

    early_money_in = bool(
        money_in
        and not recent_distribution
        and ret_5d < p["early_ret_5d_max"]
        and dist_ma20 < p["early_dist_ma20_max"]
    )
    chase_money_in = bool(
        money_in
        and (
            ret_5d >= p["chase_ret_5d_min"]
            or dist_ma20 >= p["chase_dist_ma20_min"]
        )
    )
    exhaustion_inflow = bool(
        money_in
        and value_ratio_20 >= 2.5
        and (close_position < 0.70 or chase_money_in)
    )

    # ── Breakout flow ─────────────────────────────────────────────────────────
    breakout_flow = False
    if not np.isnan(resistance_20) and resistance_20 > 0:
        breakout_flow = bool(
            current_close > resistance_20
            and value_ratio_20 >= bvr
            and close_position >= sc
            and ret_1d >= 0.015
        )

    base_tight = bool(rng_valid and range_20 < range_120_q35 * 1.15)
    volume_dry_before = bool(
        df["volume_ratio_20"].shift(1).tail(5).fillna(1.0).mean() < 1.1
    )
    confirmed_breakout_flow = bool(
        breakout_flow
        and not recent_distribution
        and (base_tight or volume_dry_before)
    )

    high_quality_breakout = bool(
        confirmed_breakout_flow
        and sector_is_outperform
        and market_trend != "DOWNTREND"
        and not distribution
    )

    # ── Foreign flow scoring ──────────────────────────────────────────────────
    foreign_score    = 0
    has_foreign_flow = False

    if foreign_flow:
        net5d  = foreign_flow.get("net_flow_5d")
        net20d = foreign_flow.get("net_flow_20d")
        history = foreign_flow.get("flow_history", [])

        if net5d is not None or net20d is not None:
            has_foreign_flow = True
            net5d  = float(net5d  or 0.0)
            net20d = float(net20d or 0.0)

            # Latest day's net flow for ratio
            day_net = float(history[-1]["net_flow"]) if history else 0.0
            safe_value = value_traded if value_traded > 0 else 1.0
            foreign_net_ratio = day_net / safe_value

            if foreign_net_ratio >= p["foreign_ratio_threshold"]:
                foreign_score += 1
            if net5d > 0 and net20d > 0:
                foreign_score += 1
            if foreign_net_ratio <= -p["foreign_ratio_threshold"]:
                foreign_score -= 1
            if net5d < 0 and net20d < 0:
                foreign_score -= 1

    # ── Daily score ───────────────────────────────────────────────────────────
    score = 0
    if money_in:                score += 2
    if strong_money_in:         score += 1
    if early_money_in:          score += 1
    if confirmed_breakout_flow: score += 3
    elif breakout_flow:         score += 1
    if accumulation:            score += 1
    if sector_is_outperform:    score += 1
    if sector_rs > 0:           score += 1
    if foreign_score > 0:       score += foreign_score

    if money_out:               score -= 2
    if strong_money_out:        score -= 1
    if chase_money_in:          score -= 2
    if exhaustion_inflow:       score -= 3
    if recent_distribution:     score -= 2
    if distribution:            score -= 4
    if market_trend == "DOWNTREND": score -= 2
    if foreign_score < 0:       score += foreign_score

    score = max(-10, min(10, score))

    # ── Regime ───────────────────────────────────────────────────────────────
    if distribution or score <= -4:
        regime = "DISTRIBUTION"
    elif exhaustion_inflow:
        regime = "EXHAUSTION_INFLOW"
    elif confirmed_breakout_flow and score >= 4:
        regime = "BREAKOUT_FLOW"
    elif chase_money_in:
        regime = "CHASE_MONEY_IN"
    elif early_money_in and score >= 2:
        regime = "EARLY_MONEY_IN"
    elif accumulation and score >= 1:
        regime = "ACCUMULATION"
    elif score >= 3:
        regime = "MONEY_IN"
    elif score <= -2:
        regime = "MONEY_OUT"
    else:
        regime = "NEUTRAL"

    # ── Action bias ───────────────────────────────────────────────────────────
    if regime in {"BREAKOUT_FLOW", "EARLY_MONEY_IN"} and market_trend != "DOWNTREND":
        action_bias = "BUY_CANDIDATE"
    elif regime == "ACCUMULATION":
        action_bias = "WATCHLIST"
    elif regime in {"DISTRIBUTION", "MONEY_OUT", "CHASE_MONEY_IN", "EXHAUSTION_INFLOW"}:
        action_bias = "AVOID_OR_EXIT"
    else:
        action_bias = "NEUTRAL"

    # ── Confidence ────────────────────────────────────────────────────────────
    if not liquidity_ok:
        confidence = "LOW"
    elif ohlcv_days < 120:
        confidence = "LOW"
    elif abs(score) >= 5 and market_trend != "DOWNTREND":
        confidence = "HIGH"
    elif abs(score) >= 3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # ── Reasons & blockers ────────────────────────────────────────────────────
    reasons  = []
    blockers = []

    if high_quality_breakout:
        reasons.append("High-quality breakout: value spike + sector aligned + market healthy")
    elif confirmed_breakout_flow:
        reasons.append("Confirmed breakout flow after a tight/dry base")
    elif breakout_flow:
        reasons.append("Breakout above 20-day resistance with value spike")
    if early_money_in:
        reasons.append("Early money inflow without extended 5-day run or MA20 stretch")
    if strong_money_in:
        reasons.append("Strong close near session high with abnormal money inflow")
    elif money_in:
        reasons.append("Positive return with above-average money inflow")
    if accumulation and near_support:
        reasons.append("Volume compression near support — accumulation base forming")
    elif accumulation:
        reasons.append("Range and volume compression — possible accumulation")
    if sector_is_outperform:
        reasons.append("Sector outperforming benchmark")
    if foreign_score > 0:
        reasons.append("Foreign investors net buying")

    if distribution:
        blockers.append("Distribution — strong hands selling into liquidity")
    if recent_distribution:
        blockers.append("Recent distribution in the last 5 sessions")
    if chase_money_in:
        blockers.append("Chase inflow — stock already extended from recent run/MA20")
    if exhaustion_inflow:
        blockers.append("Possible exhaustion inflow — high value after extended move")
    if failed_breakout:
        blockers.append("Failed breakout — closed below resistance despite high volume")
    if money_out:
        blockers.append("Money outflow — price down with high value and weak close")
    if market_trend == "DOWNTREND":
        blockers.append("Market in DOWNTREND — reduces buy signal reliability")
    if not liquidity_ok:
        blockers.append(f"Low liquidity (avg {avg_value_20/1e9:.1f}B VND/day)")

    as_of = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])[:10]

    return {
        "symbol":      sym,
        "as_of":       as_of,
        "engine":      "blackbox_money_flow_v1",
        "regime":      regime,
        "action_bias": action_bias,
        "score":       score,
        "confidence":  confidence,
        "signals": {
            "liquidity_ok":             liquidity_ok,
            "money_in":                 money_in,
            "strong_money_in":          strong_money_in,
            "early_money_in":           early_money_in,
            "chase_money_in":           chase_money_in,
            "exhaustion_inflow":        exhaustion_inflow,
            "money_out":                money_out,
            "accumulation":             accumulation,
            "distribution":             distribution,
            "recent_distribution":       recent_distribution,
            "breakout_flow":            breakout_flow,
            "confirmed_breakout_flow":  confirmed_breakout_flow,
            "foreign_score":            foreign_score,
            "sector_is_outperforming":  sector_is_outperform,
            "relative_strength_20d":    round(sector_rs, 4),
        },
        "metrics": {
            "value_ratio_20":  round(value_ratio_20,  3),
            "volume_ratio_20": round(volume_ratio_20, 3),
            "value_z_60":      round(value_z_60,      3),
            "close_position":  round(close_position,  3),
            "ret_1d":          round(ret_1d,           4),
            "ret_5d":          round(ret_5d,           4),
            "dist_ma20":       round(dist_ma20,        4),
            "avg_value_20":    int(avg_value_20),
        },
        "reasons":  reasons,
        "blockers": blockers,
        "data_quality": {
            "ohlcv_days":       ohlcv_days,
            "has_foreign_flow": has_foreign_flow,
            "has_intraday_flow": False,
            "warnings":         warnings,
        },
    }


# ── Top-level builder ─────────────────────────────────────────────────────────

def build_blackbox_output(
    symbol: str,
    df: pd.DataFrame,
    context: dict,
    foreign_flow: dict | None = None,
    params: dict | None = None,
) -> dict:
    """Feature-engineer df, then classify. Returns full blackbox output dict."""
    df = add_money_flow_features(df, params)
    return classify_money_flow(df, context, foreign_flow, params)


# ── Public agent entry point ──────────────────────────────────────────────────

def analyze(
    symbol: str,
    date: str | None = None,
    market_context: dict | None = None,
    sector_context: dict | None = None,
) -> dict:
    """
    Entry point matching the existing agent interface (flow_agent.analyze, etc.).

    Fetches OHLCV (250 bars) + VNINDEX (for relative strength) + foreign flow,
    builds context, and returns the full blackbox output dict.

    Args:
        symbol:         Ticker, e.g. "HPG"
        date:           As-of date string YYYY-MM-DD (defaults to today)
        market_context: {'trend': 'UPTREND'|'SIDEWAY'|'DOWNTREND', 'vni_change_pct': float, ...}
        sector_context: {'relative_strength_20d': float, 'is_outperforming': bool}
                        If omitted, RS is computed from VNINDEX automatically.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    print(f"[money_flow_agent] {symbol} ({date})")

    # ── 1. OHLCV ──────────────────────────────────────────────────────────────
    try:
        df = fetcher.get_ohlcv(symbol, n_days=250)
    except Exception as exc:
        logger.warning("[money_flow_agent] get_ohlcv %s: %s", symbol, exc)
        return _make_fallback(symbol, date, str(exc))

    if df is None or df.empty or len(df) < 20:
        return _make_fallback(symbol, date, "Insufficient OHLCV data")

    # ── 2. Relative strength vs VNINDEX ───────────────────────────────────────
    rs_20d = 0.0
    if sector_context is None or "relative_strength_20d" not in sector_context:
        try:
            vni = fetcher.get_vnindex(n_days=25)
            if vni is not None and not vni.empty and len(vni) >= 21:
                stock_close = (
                    df.sort_values("date")["close"]
                    if "date" in df.columns
                    else df["close"]
                )
                vni_close = (
                    vni.sort_values("date")["close"]
                    if "date" in vni.columns
                    else vni["close"]
                )
                if len(stock_close) >= 21 and len(vni_close) >= 21:
                    stock_ret = float(stock_close.iloc[-1] / stock_close.iloc[-21] - 1)
                    vni_ret   = float(vni_close.iloc[-1]   / vni_close.iloc[-21]   - 1)
                    rs_20d    = round(stock_ret - vni_ret, 4)
        except Exception as exc:
            logger.debug("[money_flow_agent] vnindex RS %s: %s", symbol, exc)

    # ── 3. Build context ──────────────────────────────────────────────────────
    if sector_context is None:
        sector_context = {
            "relative_strength_20d": rs_20d,
            "is_outperforming":      rs_20d > 0,
        }
    else:
        # Merge: computed rs_20d as fallback, caller-provided values win
        sector_context = {
            "relative_strength_20d": rs_20d,
            "is_outperforming":      rs_20d > 0,
            **sector_context,
        }

    context = {
        "symbol":         symbol,
        "market_context": market_context or {"trend": "SIDEWAY", "vni_change_pct": 0.0},
        "sector_context": sector_context,
    }

    # ── 4. Foreign flow ───────────────────────────────────────────────────────
    fflow: dict | None = None
    try:
        raw = fetcher.get_foreign_flow(symbol)
        if raw and (raw.get("net_flow_5d") is not None or raw.get("net_flow_20d") is not None):
            fflow = raw
    except Exception as exc:
        logger.debug("[money_flow_agent] foreign_flow %s: %s", symbol, exc)

    # ── 5. Classify ───────────────────────────────────────────────────────────
    try:
        result = build_blackbox_output(symbol, df, context, fflow)
        print(
            f"[money_flow_agent] {symbol} — regime={result['regime']}, "
            f"score={result['score']}, bias={result['action_bias']}, "
            f"confidence={result['confidence']}"
        )
        return result
    except Exception as exc:
        logger.warning("[money_flow_agent] classify %s: %s", symbol, exc)
        return _make_fallback(symbol, date, str(exc))


# ── Screener priority score ───────────────────────────────────────────────────

def compute_priority_score(result: dict) -> float:
    """
    Candidate rank formula from spec §Pipeline Integration:

        priority_score = blackbox_score * 10
                       + max(relative_strength_20d, 0) * 100
                       + min(value_ratio_20, 3) * 5
    """
    score = result.get("score", 0)
    rs    = result.get("signals", {}).get("relative_strength_20d", 0.0)
    vr    = result.get("metrics", {}).get("value_ratio_20", 1.0)
    return score * 10 + max(float(rs), 0.0) * 100 + min(float(vr), 3.0) * 5


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_fallback(symbol: str, date: str, reason: str = "") -> dict:
    return {
        "symbol":      symbol,
        "as_of":       date,
        "engine":      "blackbox_money_flow_v1",
        "regime":      "NEUTRAL",
        "action_bias": "NEUTRAL",
        "score":       0,
        "confidence":  "LOW",
        "signals":     {},
        "metrics":     {},
        "reasons":     [],
        "blockers":    [reason] if reason else ["Data unavailable"],
        "data_quality": {
            "ohlcv_days":       0,
            "has_foreign_flow": False,
            "has_intraday_flow": False,
            "warnings":         [reason] if reason else [],
        },
    }
