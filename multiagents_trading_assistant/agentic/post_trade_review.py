"""Post-trade review — classify losing trades into enumerated categories.

Deterministic rule-based classifier. Reads a trades DataFrame (from backtest
or live persistence) and assigns each loss one LossCategory. The output is a
report DataFrame suitable for human review or for feeding back into gate
threshold tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from multiagents_trading_assistant.agentic.schemas import LossCategory


@dataclass(frozen=True)
class ReviewThresholds:
    entry_chase_gap_pct: float = 3.0       # entry_open vs signal_close gap ≥ this %
    false_breakout_recover_bars: int = 5   # SL hit then price recovers within N bars
    exit_too_early_recover_pct: float = 5.0  # after SL, price runs ≥ this % within window
    exit_too_late_max_hold_ratio: float = 0.95  # holding_bars ≥ ratio × max_hold


DEFAULT_REVIEW_THRESHOLDS = ReviewThresholds()


def classify_loss(
    trade: dict,
    *,
    ohlcv_after: pd.DataFrame | None = None,
    market_after: pd.DataFrame | None = None,
    thresholds: ReviewThresholds | None = None,
) -> LossCategory:
    """Classify one losing trade.

    Args:
        trade: dict-like with keys: setup_type, signal_date, entry_date,
            entry_price, exit_price, exit_date, exit_reason, holding_bars,
            edge_risk (dict with max_holding_bars), priority_score, ...
        ohlcv_after: optional OHLCV bars starting at exit_date for follow-up
            check (used by FALSE_BREAKOUT and EXIT_TOO_EARLY).
        market_after: optional VNINDEX bars over the holding period (used by
            REGIME_SHIFT).

    Returns one LossCategory. UNCLASSIFIED if no rule matches confidently.
    """
    t = thresholds or DEFAULT_REVIEW_THRESHOLDS

    pnl_pct = float(trade.get("pnl_pct") or 0.0)
    if pnl_pct >= 0:
        return LossCategory.UNCLASSIFIED

    setup_type = str(trade.get("setup_type") or "").upper()
    exit_reason = str(trade.get("exit_reason") or "").upper()
    holding_bars = int(trade.get("holding_bars") or 0)

    edge_risk = trade.get("edge_risk") or {}
    if isinstance(edge_risk, str):
        try:
            import ast
            edge_risk = ast.literal_eval(edge_risk)
        except Exception:
            edge_risk = {}
    max_hold = int(edge_risk.get("max_holding_bars") or 30) if isinstance(edge_risk, dict) else 30

    # 1. EXIT_TOO_LATE — held nearly to max_hold and still lost
    if holding_bars >= int(max_hold * t.exit_too_late_max_hold_ratio):
        return LossCategory.EXIT_TOO_LATE

    # 2. ENTRY_CHASE — gap up on entry day vs signal close
    signal_close = trade.get("signal_close_price") or trade.get("close_at_signal")
    entry_price = float(trade.get("entry_price") or 0.0)
    if signal_close and entry_price:
        gap_pct = (entry_price - float(signal_close)) / float(signal_close) * 100.0
        if gap_pct >= t.entry_chase_gap_pct:
            return LossCategory.ENTRY_CHASE

    # 3. FALSE_BREAKOUT / premise-fail — quick SL hit on ANY setup (not just breakout).
    # Any setup that's stopped out within recover_bars window means the entry premise
    # didn't follow through — semantically a "false signal" regardless of setup name.
    sl_exits = ("EDGE_ATR_STOP", "EDGE_STOP_LOSS", "EDGE_TRAILING_ATR_STOP", "SL", "SL_HIT", "STOP_LOSS")
    if exit_reason in sl_exits and holding_bars <= t.false_breakout_recover_bars:
        return LossCategory.FALSE_BREAKOUT

    # 3b. STRUCTURE_BREAK — SL after the early window but well before max_hold.
    # The signal worked initially, price moved favorably or ranged, then structure broke.
    # Different root cause from FALSE_BREAKOUT (entry timing) — this is mid-trade trend failure.
    if exit_reason in sl_exits:
        upper = int(max_hold * 0.6)  # 60% of max_hold = "mid-trade"
        if t.false_breakout_recover_bars < holding_bars <= upper:
            return LossCategory.STRUCTURE_BREAK

    # 4. REGIME_SHIFT — VNI moved to risk-off during the hold
    if market_after is not None and not market_after.empty:
        try:
            vni_start = float(market_after.iloc[0]["close"])
            vni_min = float(market_after["close"].min())
            vni_dd = (vni_min - vni_start) / vni_start * 100.0
            if vni_dd <= -3.0:  # VNI lost ≥3% during the hold
                return LossCategory.REGIME_SHIFT
        except (KeyError, IndexError, ValueError):
            pass

    # 5. EXIT_TOO_EARLY — after SL, price ran back up within window
    if ohlcv_after is not None and not ohlcv_after.empty and exit_reason in ("EDGE_ATR_STOP", "EDGE_STOP_LOSS", "EDGE_TRAILING_ATR_STOP", "SL", "SL_HIT", "STOP_LOSS"):
        try:
            window = ohlcv_after.head(t.false_breakout_recover_bars + 5)
            recovery = (window["close"].max() - float(trade["exit_price"])) / float(trade["exit_price"]) * 100.0
            if recovery >= t.exit_too_early_recover_pct:
                return LossCategory.EXIT_TOO_EARLY
        except (KeyError, ValueError):
            pass

    # 6. WEAK_SECTOR — sector RS was already weak at entry
    sector_rs = trade.get("sector_rs_at_entry")
    if sector_rs is not None:
        try:
            rs = float(sector_rs)
            if rs < 1.0:
                rs *= 100.0
            if rs < 30:
                return LossCategory.WEAK_SECTOR
        except (TypeError, ValueError):
            pass

    # 7. DATA_ISSUE — abnormal gap or missing data flagged at entry
    if trade.get("data_quality_ok") is False:
        return LossCategory.DATA_ISSUE

    return LossCategory.UNCLASSIFIED


def review_trades_df(
    trades_df: pd.DataFrame,
    *,
    ohlcv_map: dict[str, pd.DataFrame] | None = None,
    vnindex_df: pd.DataFrame | None = None,
    thresholds: ReviewThresholds | None = None,
) -> pd.DataFrame:
    """Apply classify_loss across a trades DataFrame.

    Returns the original frame with a new `loss_category` column.
    """
    out = trades_df.copy()
    categories: list[str] = []

    for _, row in out.iterrows():
        trade = row.to_dict()
        ohlcv_after = None
        market_after = None
        if float(trade.get("pnl_pct") or 0.0) < 0:
            if ohlcv_map is not None:
                sym_df = ohlcv_map.get(str(trade.get("symbol")))
                if sym_df is not None and "date" in sym_df.columns:
                    after = sym_df[sym_df["date"] > pd.to_datetime(trade.get("exit_date"))]
                    ohlcv_after = after.head(20) if not after.empty else None
            if vnindex_df is not None and "date" in vnindex_df.columns:
                entry = pd.to_datetime(trade.get("entry_date"))
                exit_d = pd.to_datetime(trade.get("exit_date"))
                slice_ = vnindex_df[(vnindex_df["date"] >= entry) & (vnindex_df["date"] <= exit_d)]
                market_after = slice_ if not slice_.empty else None
            cat = classify_loss(
                trade,
                ohlcv_after=ohlcv_after,
                market_after=market_after,
                thresholds=thresholds,
            )
        else:
            cat = LossCategory.UNCLASSIFIED  # winners not classified
        categories.append(cat.value)

    out["loss_category"] = categories
    return out


def summarize_losses(reviewed_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate loss categories: count, total_pnl, avg_pnl_pct."""
    losers = reviewed_df[reviewed_df["pnl_pct"] < 0].copy()
    if losers.empty:
        return pd.DataFrame(columns=["loss_category", "count", "total_pnl", "avg_pnl_pct"])
    grouped = (
        losers.groupby("loss_category")
        .agg(
            count=("symbol", "count"),
            total_pnl=("pnl", "sum"),
            avg_pnl_pct=("pnl_pct", "mean"),
        )
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return grouped
