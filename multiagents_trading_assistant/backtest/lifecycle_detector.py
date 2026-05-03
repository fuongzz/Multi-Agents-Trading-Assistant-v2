"""Bridge existing TA candidate logic into lifecycle Signals.

This module intentionally reuses the current deterministic backtest filters:
TA detectors, regime gating, money-flow gating, pending-entry zones, and the
pipeline review layer. Its only job is to convert an approved candidate into a
Signal so the lifecycle engine can manage the position with multi-leg logic.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from multiagents_trading_assistant.backtest.engine import (
    _STRATEGIES,
    _compute_initial_sl,
    _is_setup_allowed_in_regime,
    _money_flow_entry_ok,
    _pending_entry_zone,
    _prep_ta_money_flow_features,
)
from multiagents_trading_assistant.backtest.lifecycle import Signal
from multiagents_trading_assistant.backtest.pipeline_review import (
    record_stats as record_pipeline_review_stats,
    record_soft_approval as record_pipeline_review_soft_approval,
    review_entry,
)
from multiagents_trading_assistant.backtest.regime import (
    Regime,
    classify_regime,
    is_setup_allowed_for_regime,
    playbook_hint_for_signal,
)
from multiagents_trading_assistant.backtest.sector_rotation import SectorRotationModel
from multiagents_trading_assistant.setup_scoring import PULLBACK_SETUPS, MOMENTUM_SETUPS
from multiagents_trading_assistant.indicators import compute_indicators
from multiagents_trading_assistant.setup_scoring import score_setup


class LifecycleSignalDetector:
    """Stateful single-symbol detector for LifecycleEngine.run_symbol()."""

    def __init__(
        self,
        symbol: str,
        df: pd.DataFrame,
        *,
        setups: Optional[list[str]] = None,
        use_money_flow: bool = True,
        money_flow_min_score: int = 3,
        money_flow_params: Optional[dict] = None,
        pipeline_review: bool = True,
        pipeline_review_stats: Optional[dict] = None,
        rr_ratio: float = 1.5,
        sector_model: Optional[SectorRotationModel] = None,
    ):
        self.symbol = symbol
        self.df = df.reset_index(drop=True)
        self.setups = setups
        self.use_money_flow = use_money_flow
        self.money_flow_min_score = money_flow_min_score
        self.pipeline_review = pipeline_review
        self.pipeline_review_stats = pipeline_review_stats
        self.rr_ratio = rr_ratio
        self.sector_model = sector_model
        self.mf_feat = (
            _prep_ta_money_flow_features(self.df, money_flow_params)
            if use_money_flow
            else None
        )
        self.strategies = [
            (name, fn)
            for name, fn in _STRATEGIES
            if setups is None or name in setups
        ]

    def __call__(self, df: pd.DataFrame, i: int) -> Optional[Signal]:
        window = df.iloc[: i + 1]
        ind = compute_indicators(window)
        if not ind:
            return None

        ref_trend = str(ind.get("ma_trend") or "SIDEWAY")
        mf_row = self.mf_feat.iloc[i] if self.mf_feat is not None else None
        regime = classify_regime(window, mf_row)
        row = df.iloc[i]
        sector = self._sector_snapshot(str(row["date"])[:10])

        for setup_name, detect_fn in self.strategies:
            if not is_setup_allowed_for_regime(setup_name, regime):
                continue
            if not _is_setup_allowed_in_regime(setup_name, ref_trend):
                # The legacy setup-regime table is still useful, but too strict
                # after regime-first routing. In strong uptrends we allow
                # pullback/retest/shakeout setups to continue with smaller
                # regime-adjusted sizing.
                if regime.regime != Regime.UPTREND or setup_name not in {
                    "SPRING",
                    "BULLISH_ENGULFING",
                    "PIN_BAR",
                    "RSI_BOUNCE",
                    "HAMMER",
                    "DOUBLE_BOTTOM",
                    "MA_PULLBACK",
                    "TREND_PULLBACK",
                    "RETEST",
                    "BREAKOUT_RETEST_ENTRY",
                }:
                    continue
            passed, reasons = detect_fn(window, ind)
            if not passed:
                continue
            if mf_row is not None:
                mf_ok, mf_reasons = _money_flow_entry_ok(
                    mf_row, self.money_flow_min_score, setup_name
                )
                if not mf_ok:
                    if not self._soft_allow_money_flow(setup_name, regime, mf_row):
                        continue
                    mf_reasons = [*mf_reasons, "money_flow_soft_allowed"]
                reasons = [*reasons, *mf_reasons]

            signal_close = float(df.iloc[i]["close"])
            zone_low, zone_high = _pending_entry_zone(setup_name, signal_close, ind)
            if zone_low <= 0 or zone_high <= 0 or zone_low > zone_high:
                continue

            # LifecycleEngine fills on i+1. Use next open for the same risk/review
            # estimate the legacy engine used before creating Trade.
            if i + 1 >= len(df):
                return None
            entry_price = float(df.iloc[i + 1]["open"])
            if entry_price <= 0:
                continue

            sl = _compute_initial_sl(entry_price, ind)
            if sl <= 0 or sl >= entry_price:
                continue

            confluence = 0.0
            if self.pipeline_review:
                review = review_entry(
                    setup_name=setup_name,
                    ind=ind,
                    mf_row=mf_row,
                    entry_price=entry_price,
                    stop_loss=sl,
                    rr_ratio=self.rr_ratio,
                )
                if not review.approved:
                    soft_ok, soft_reasons = self._soft_allow_review(
                        setup_name, regime, review.blockers
                    )
                    if not soft_ok:
                        record_pipeline_review_stats(self.pipeline_review_stats, review)
                        continue
                    record_pipeline_review_soft_approval(
                        self.pipeline_review_stats, review
                    )
                    reasons = [*reasons, *soft_reasons]
                else:
                    record_pipeline_review_stats(self.pipeline_review_stats, review)
                confluence = review.confluence_score
                reasons = [*reasons, *review.reasons]
            else:
                mf_dict = {}
                if mf_row is not None:
                    mf_dict = {
                        "regime": str(mf_row.get("mf_regime", "NEUTRAL")),
                        "score": float(mf_row.get("mf_score", 0) or 0),
                    }
                scored = score_setup(
                    setup_name,
                    ind,
                    money_flow=mf_dict,
                    reference_trend=ref_trend,
                )
                confluence = float(scored.get("score") or 0.0)

            return Signal(
                symbol=self.symbol,
                date=str(row["date"])[:10],
                setup_type=setup_name,
                confluence_score=confluence,
                entry_zone_low=round(float(zone_low), 2),
                entry_zone_high=round(float(zone_high), 2),
                suggested_sl=float(sl),
                atr=float(ind.get("atr") or 0.0),
                regime=regime.regime.value,
                money_flow_score=int(mf_row.get("mf_score", 0)) if mf_row is not None else 0,
                money_flow_regime=str(mf_row.get("mf_regime", "NEUTRAL")) if mf_row is not None else "NEUTRAL",
                playbook_hint=playbook_hint_for_signal(setup_name, regime),
                risk_multiplier=regime.risk_multiplier,
                sector_name=sector.sector,
                sector_regime=sector.regime.value,
                sector_rank=sector.rank,
                sector_score=sector.score,
                regime_notes=regime.notes,
                reasons=[
                    *reasons,
                    f"regime={regime.regime.value}",
                    f"risk_mult={regime.risk_multiplier:.2f}",
                    *sector.notes,
                ],
            )

        return self._core_trend_signal(df, i, ind, mf_row, regime, sector)

    def _sector_snapshot(self, date: str):
        if self.sector_model is None:
            from multiagents_trading_assistant.backtest.sector_rotation import _unknown_snapshot

            return _unknown_snapshot("UNKNOWN", "sector_model_disabled")
        return self.sector_model.snapshot(self.symbol, date)

    def _core_trend_signal(self, df: pd.DataFrame, i: int, ind: dict, mf_row, regime, sector) -> Optional[Signal]:
        """Emit a benchmark-aware core entry when trend is already established.

        Pattern detectors are intentionally tactical; they can miss long, clean
        uptrends. This fallback lets lifecycle hold a core sleeve instead of
        sitting in cash until a narrow setup appears.
        """
        if regime.regime != Regime.UPTREND:
            return None
        if self.sector_model is not None and not sector.allows_core:
            return None
        if i + 1 >= len(df):
            return None
        close = float(df.iloc[i]["close"])
        if close <= 0:
            return None

        ma20 = float(ind.get("ma20") or 0.0)
        ma50 = float(ind.get("ma50") or ind.get("ma60") or 0.0)
        ma_trend = str(ind.get("ma_trend") or "")
        mf_regime = str(mf_row.get("mf_regime", "NEUTRAL")) if mf_row is not None else "NEUTRAL"
        recent_distribution = bool(mf_row.get("recent_distribution", False)) if mf_row is not None else False
        if ma20 <= 0 or ma50 <= 0 or close < ma20 or ma20 < ma50:
            return None
        if ma_trend == "DOWNTREND":
            return None
        if recent_distribution or mf_regime in {"DISTRIBUTION", "MONEY_OUT", "EXHAUSTION_INFLOW"}:
            return None

        sl = min(close * 0.93, ma50 * 0.97)
        if sl <= 0 or sl >= close:
            return None
        zone_high = close * 1.035
        return Signal(
            symbol=self.symbol,
            date=str(df.iloc[i]["date"])[:10],
            setup_type="CORE_TREND",
            confluence_score=max(62.0, min(85.0, 60.0 + regime.trend_score * 6.0)),
            entry_zone_low=round(close * 0.95, 2),
            entry_zone_high=round(zone_high, 2),
            suggested_sl=round(sl, 2),
            atr=float(ind.get("atr") or 0.0),
            regime=regime.regime.value,
            money_flow_score=int(mf_row.get("mf_score", 0)) if mf_row is not None else 0,
            money_flow_regime=mf_regime,
            playbook_hint=playbook_hint_for_signal("CORE_TREND", regime),
            risk_multiplier=regime.risk_multiplier,
            sector_name=sector.sector,
            sector_regime=sector.regime.value,
            sector_rank=sector.rank,
            sector_score=sector.score,
            regime_notes=regime.notes,
            reasons=[
                "core_trend_fallback",
                f"regime={regime.regime.value}",
                f"trend_score={regime.trend_score:.1f}",
                f"risk_mult={regime.risk_multiplier:.2f}",
                *sector.notes,
            ],
        )

    @staticmethod
    def _soft_allow_money_flow(setup_name: str, regime, mf_row: pd.Series) -> bool:
        if regime.regime != Regime.UPTREND:
            return False
        mf_regime = str(mf_row.get("mf_regime", "NEUTRAL"))
        recent_distribution = bool(mf_row.get("recent_distribution", False))
        if recent_distribution or mf_regime in {"DISTRIBUTION", "MONEY_OUT", "EXHAUSTION_INFLOW"}:
            return False
        setup = setup_name.upper()
        return setup in (PULLBACK_SETUPS | MOMENTUM_SETUPS | {
            "RETEST",
            "BREAKOUT_RETEST_ENTRY",
            "SPRING",
            "BULLISH_ENGULFING",
            "PIN_BAR",
        })

    @staticmethod
    def _soft_allow_review(setup_name: str, regime, blockers: list[str]) -> tuple[bool, list[str]]:
        if regime.regime != Regime.UPTREND:
            return False, []
        hard_prefixes = (
            "liquidity_low",
            "money_flow_distribution",
            "money_flow_money_out",
            "money_flow_exhaustion_inflow",
            "recent_distribution",
            "bearish_momentum",
            "poor_room_to_resistance",
            "rsi_extreme",
        )
        if any(any(b.startswith(prefix) for prefix in hard_prefixes) for b in blockers):
            return False, []
        softable = [
            b for b in blockers
            if b.startswith("regime_confluence_")
            or b in {"weak_pipeline_confluence", "reversal_without_flow"}
            or b.startswith("weak_setup_score_")
        ]
        if softable and len(softable) == len(blockers):
            return True, [f"pipeline_soft_allowed:{','.join(softable[:3])}"]
        return False, []
