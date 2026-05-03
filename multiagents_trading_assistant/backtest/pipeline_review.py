"""Deterministic second-stage review for backtests.

This module mirrors the live trade pipeline's review role without calling LLMs
or writing side effects. It is intentionally conservative and reproducible:
candidate TA/money-flow entries are scored, then vetoed by the same kinds of
checks that synthesis/trader/risk apply in live mode.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd

from multiagents_trading_assistant.setup_scoring import (
    score_setup,
    MF_STRICT_SETUPS as _STRICT_SETUPS,
    MF_COMPRESSION_SETUPS as _COMPRESSION_SETUPS,
    MF_REVERSAL_SETUPS as _REVERSAL_SETUPS,
    MIN_CONFLUENCE_BY_REGIME as _MIN_CONFLUENCE_BY_REGIME,
)


@dataclass(frozen=True)
class ReviewDecision:
    approved: bool
    confluence_score: float
    quality: str
    reasons: list[str]
    blockers: list[str]


def new_stats() -> dict[str, Any]:
    return {
        "reviewed": 0,
        "approved": 0,
        "rejected": 0,
        "reject_reasons": Counter(),
        "approved_quality": Counter(),
    }


def merge_stats(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in ("reviewed", "approved", "rejected"):
        dst[key] = int(dst.get(key, 0)) + int(src.get(key, 0))
    dst.setdefault("reject_reasons", Counter()).update(src.get("reject_reasons", Counter()))
    dst.setdefault("approved_quality", Counter()).update(src.get("approved_quality", Counter()))


def record_stats(stats: dict[str, Any] | None, decision: ReviewDecision) -> None:
    if stats is None:
        return
    stats["reviewed"] = int(stats.get("reviewed", 0)) + 1
    if decision.approved:
        stats["approved"] = int(stats.get("approved", 0)) + 1
        stats.setdefault("approved_quality", Counter()).update([decision.quality])
        return
    stats["rejected"] = int(stats.get("rejected", 0)) + 1
    reason = decision.blockers[0] if decision.blockers else "review_reject"
    stats.setdefault("reject_reasons", Counter()).update([reason])


def record_soft_approval(stats: dict[str, Any] | None, decision: ReviewDecision) -> None:
    if stats is None:
        return
    stats["reviewed"] = int(stats.get("reviewed", 0)) + 1
    stats["approved"] = int(stats.get("approved", 0)) + 1
    stats["soft_approved"] = int(stats.get("soft_approved", 0)) + 1
    stats.setdefault("approved_quality", Counter()).update([decision.quality])
    reason = decision.blockers[0] if decision.blockers else "soft_approved"
    stats.setdefault("soft_approve_reasons", Counter()).update([reason])


def print_review_stats(stats: dict[str, Any] | None) -> None:
    if not stats:
        return
    reviewed = int(stats.get("reviewed", 0))
    if reviewed == 0:
        print("[pipeline_review] Khong co candidate nao de review.")
        return
    approved = int(stats.get("approved", 0))
    rejected = int(stats.get("rejected", 0))
    soft_approved = int(stats.get("soft_approved", 0))
    rate = approved / reviewed * 100
    soft_suffix = f", soft_approved={soft_approved}" if soft_approved else ""
    print(
        f"[pipeline_review] reviewed={reviewed}, approved={approved} "
        f"({rate:.1f}%), rejected={rejected}{soft_suffix}"
    )
    rejects = stats.get("reject_reasons", Counter())
    if rejects:
        top = ", ".join(f"{k}:{v}" for k, v in rejects.most_common(5))
        print(f"[pipeline_review] top reject reasons: {top}")
    soft_reasons = stats.get("soft_approve_reasons", Counter())
    if soft_reasons:
        top = ", ".join(f"{k}:{v}" for k, v in soft_reasons.most_common(5))
        print(f"[pipeline_review] top soft approve reasons: {top}")


def print_candidate_llm_budget(
    stats: dict[str, Any] | None,
    trades: list[Any],
    *,
    lite_calls_per_candidate: int = 4,
    sonnet_calls_per_candidate: int = 1,
    lite_cost_per_call: float = 0.0,
    sonnet_cost_per_call: float = 0.0,
) -> None:
    """Print LLM call/cost savings from pre-LLM candidate filtering.

    The live trade pipeline currently spends per candidate on:
      - lite analyst calls: technical, flow, sentiment, synthesis
      - sonnet decision call: trader

    Backtest review is deterministic, so it estimates how many candidates
    would be rejected before those live LLM calls are made.
    """
    if not stats:
        return

    reviewed = int(stats.get("reviewed", 0))
    approved = int(stats.get("approved", 0))
    rejected = int(stats.get("rejected", 0))
    if reviewed == 0:
        print("[candidate_llm] No reviewed candidates; run with --candidate-llm-backtest on a wider period/universe.")
        return

    lite_calls_per_candidate = max(0, int(lite_calls_per_candidate))
    sonnet_calls_per_candidate = max(0, int(sonnet_calls_per_candidate))

    baseline_lite = reviewed * lite_calls_per_candidate
    baseline_sonnet = reviewed * sonnet_calls_per_candidate
    optimized_lite = approved * lite_calls_per_candidate
    optimized_sonnet = approved * sonnet_calls_per_candidate
    saved_lite = baseline_lite - optimized_lite
    saved_sonnet = baseline_sonnet - optimized_sonnet
    total_baseline = baseline_lite + baseline_sonnet
    total_optimized = optimized_lite + optimized_sonnet
    total_saved = total_baseline - total_optimized

    approval_rate = approved / reviewed * 100
    closed_trades = len([t for t in trades if getattr(t, "is_closed", False)])

    print()
    print("[candidate_llm] PRE-LLM CANDIDATE BACKTEST")
    print(f"[candidate_llm] candidates={reviewed}, pass_pre_llm={approved} ({approval_rate:.1f}%), rejected={rejected}")
    print(f"[candidate_llm] closed_trades_after_backtest={closed_trades}")
    print(
        "[candidate_llm] baseline calls: "
        f"lite={baseline_lite}, sonnet={baseline_sonnet}, total={total_baseline}"
    )
    print(
        "[candidate_llm] optimized calls: "
        f"lite={optimized_lite}, sonnet={optimized_sonnet}, total={total_optimized}"
    )
    print(
        "[candidate_llm] saved calls: "
        f"lite={saved_lite}, sonnet={saved_sonnet}, total={total_saved}"
    )

    if lite_cost_per_call > 0 or sonnet_cost_per_call > 0:
        baseline_usd = baseline_lite * lite_cost_per_call + baseline_sonnet * sonnet_cost_per_call
        optimized_usd = optimized_lite * lite_cost_per_call + optimized_sonnet * sonnet_cost_per_call
        saved_usd = baseline_usd - optimized_usd
        print(
            "[candidate_llm] estimated cost: "
            f"baseline=${baseline_usd:.4f}, optimized=${optimized_usd:.4f}, saved=${saved_usd:.4f}"
        )

    rejects = stats.get("reject_reasons", Counter())
    if rejects:
        top = ", ".join(f"{k}:{v}" for k, v in rejects.most_common(8))
        print(f"[candidate_llm] reject mix: {top}")


def review_entry(
    *,
    setup_name: str,
    ind: dict,
    mf_row: pd.Series | None,
    entry_price: float,
    stop_loss: float,
    rr_ratio: float,
) -> ReviewDecision:
    mf_dict = _mf_dict(mf_row)
    setup_scoring = score_setup(
        setup_name,
        ind,
        money_flow=mf_dict,
        reference_trend=str(ind.get("ma_trend") or "SIDEWAY"),
    )
    tech_score = float(setup_scoring.get("score") or 0.0)
    money_score = _money_points(mf_row)
    confluence = round(0.40 * tech_score + 0.25 * money_score + 17.5, 1)
    quality = "STRONG" if confluence >= 70 else "MEDIUM" if confluence >= 50 else "WEAK"

    reasons = [
        f"pipeline_conf={confluence}",
        f"pipeline_quality={quality}",
        f"setup_score={tech_score:.0f}",
        f"pipeline_money={money_score:.0f}",
    ]
    blockers: list[str] = []

    rsi = _num(ind.get("rsi"))
    ma_phase = str(ind.get("ma_phase") or "")
    macd_label = str(ind.get("macd_signal_label") or "")
    ma_trend = str(ind.get("ma_trend") or "")
    volume_surge = bool(ind.get("volume_surge"))
    rr_to_nearest_res = _nearest_resistance_rr(ind, entry_price, stop_loss)

    mf_regime = str(mf_row.get("mf_regime", "NEUTRAL")) if mf_row is not None else "NEUTRAL"
    mf_score = int(mf_row.get("mf_score", 0)) if mf_row is not None else 0
    recent_distribution = bool(mf_row.get("recent_distribution", False)) if mf_row is not None else False
    liquidity_ok = bool(mf_row.get("liquidity_ok", True)) if mf_row is not None else True

    if not liquidity_ok:
        blockers.append("liquidity_low")
    if tech_score < 50:
        blockers.append(f"weak_setup_score_{setup_name.lower()}")
    min_confluence = _MIN_CONFLUENCE_BY_REGIME.get(ma_trend, 62.0)
    if confluence < min_confluence:
        blockers.append(f"regime_confluence_{ma_trend.lower() or 'unknown'}")
    if mf_regime in {"DISTRIBUTION", "MONEY_OUT", "EXHAUSTION_INFLOW"}:
        blockers.append(f"money_flow_{mf_regime.lower()}")
    if recent_distribution:
        blockers.append("recent_distribution")
    if mf_regime == "CHASE_MONEY_IN":
        blockers.append("chase_money_in")
    if rsi is not None and rsi >= 78:
        blockers.append("rsi_extreme")
    if ma_phase == "OVERBOUGHT" and setup_name not in {"BREAKOUT", "MOMENTUM_SURGE"}:
        blockers.append("overbought_non_momentum")
    if (
        setup_name in _STRICT_SETUPS
        and rr_to_nearest_res is not None
        and rr_to_nearest_res < 0.8
    ):
        blockers.append("poor_room_to_resistance")

    if setup_name in _STRICT_SETUPS:
        if confluence < 60:
            blockers.append("weak_pipeline_confluence")
        if mf_regime not in {"BREAKOUT_FLOW", "EARLY_MONEY_IN", "MONEY_IN"} or mf_score < 3:
            blockers.append("strict_setup_no_money_confirm")
        if macd_label == "BEARISH":
            blockers.append("bearish_momentum")
    elif setup_name in _COMPRESSION_SETUPS:
        if confluence < 50:
            blockers.append("weak_pipeline_confluence")
        if ma_trend == "DOWNTREND" and not volume_surge:
            blockers.append("compression_in_downtrend")
    elif setup_name in _REVERSAL_SETUPS:
        if confluence < 52:
            blockers.append("weak_pipeline_confluence")
        if macd_label == "BEARISH" and mf_regime not in {"EARLY_MONEY_IN", "MONEY_IN", "BREAKOUT_FLOW"}:
            blockers.append("reversal_without_flow")
        if rsi is not None and rsi > 68 and setup_name in {"RSI_BOUNCE", "HAMMER", "BULLISH_ENGULFING"}:
            blockers.append("late_reversal")

    approved = not blockers
    return ReviewDecision(
        approved=approved,
        confluence_score=confluence,
        quality=quality,
        reasons=reasons,
        blockers=blockers,
    )


def _money_points(mf_row: pd.Series | None) -> float:
    if mf_row is None:
        return 50.0
    regime = str(mf_row.get("mf_regime", "NEUTRAL"))
    score = float(mf_row.get("mf_score", 0) or 0)
    liquidity_ok = bool(mf_row.get("liquidity_ok", True))
    base = {
        "BREAKOUT_FLOW": 85.0,
        "EARLY_MONEY_IN": 78.0,
        "MONEY_IN": 72.0,
        "ACCUMULATION": 58.0,
        "CHASE_MONEY_IN": 35.0,
        "EXHAUSTION_INFLOW": 15.0,
        "NEUTRAL": 50.0,
        "MONEY_OUT": 25.0,
        "DISTRIBUTION": 5.0,
    }.get(regime, 50.0)
    base += max(-15.0, min(15.0, score * 3.0))
    if not liquidity_ok:
        base -= 20.0
    return max(0.0, min(100.0, base))


def _mf_dict(mf_row: pd.Series | None) -> dict[str, Any]:
    if mf_row is None:
        return {}
    return {
        "regime": str(mf_row.get("mf_regime", "NEUTRAL")),
        "score": float(mf_row.get("mf_score", 0) or 0),
        "signals": {"liquidity_ok": bool(mf_row.get("liquidity_ok", True))},
    }


def _nearest_resistance_rr(ind: dict, entry: float, stop_loss: float) -> float | None:
    risk = entry - stop_loss
    if risk <= 0:
        return None
    resistances = ind.get("resistance_levels") or []
    above = [_num(r) for r in resistances]
    above = [r for r in above if r is not None and r > entry]
    if not above:
        return None
    return (min(above) - entry) / risk


def _num(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None
