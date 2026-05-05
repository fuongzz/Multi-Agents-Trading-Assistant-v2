"""Project-level portfolio risk rules."""

from __future__ import annotations

from multiagents_trading_assistant.risk.models import (
    OrderSide,
    RiskInput,
    RiskRuleResult,
    RuleSeverity,
)


def _risk_nav(nav_pct: float, entry_price: float | None, stop_loss: float | None) -> float:
    entry = float(entry_price or 0.0)
    sl = float(stop_loss or 0.0)
    if nav_pct <= 0 or entry <= 0 or sl <= 0:
        return 0.0
    return max(0.0, (entry - sl) / entry * nav_pct)


def validate_portfolio_limits(inp: RiskInput, nav_pct: float) -> RiskRuleResult:
    if inp.side != OrderSide.BUY:
        return RiskRuleResult("IR-002", RuleSeverity.PASS, "not a buy order")

    p = inp.portfolio
    lim = inp.limits
    max_allowed = min(
        max(0.0, p.cash_pct),
        max(0.0, lim.max_total_exposure_pct - p.total_exposure_pct),
        max(0.0, lim.max_position_pct - p.symbol_exposure_pct),
        max(0.0, lim.max_per_sector_pct - p.sector_exposure_pct),
    )
    if p.open_positions >= lim.max_open_positions and p.symbol_exposure_pct <= 0:
        return RiskRuleResult("IR-002", RuleSeverity.HARD_REJECT, "max open positions reached")
    if max_allowed <= 0:
        return RiskRuleResult("IR-002", RuleSeverity.HARD_REJECT, "no portfolio capacity")

    projected_risk = p.total_risk_pct + _risk_nav(nav_pct, inp.entry_price or inp.limit_price, inp.stop_loss)
    if projected_risk > lim.max_total_risk_pct:
        available_risk = max(0.0, lim.max_total_risk_pct - p.total_risk_pct)
        entry = float(inp.entry_price or inp.limit_price or 0.0)
        sl = float(inp.stop_loss or 0.0)
        risk_per_nav = max(0.0, (entry - sl) / entry) if entry > 0 and sl > 0 else 0.0
        risk_capped_nav = available_risk / risk_per_nav if risk_per_nav > 0 else 0.0
        max_allowed = min(max_allowed, risk_capped_nav)

    if max_allowed <= 0:
        return RiskRuleResult("IR-002", RuleSeverity.HARD_REJECT, "max total risk reached")
    if nav_pct > max_allowed:
        multiplier = max_allowed / nav_pct if nav_pct > 0 else 0.0
        return RiskRuleResult(
            "IR-002",
            RuleSeverity.SOFT_ADJUST,
            f"portfolio cap reduces NAV from {nav_pct:.2f}% to {max_allowed:.2f}%",
            sizing_multiplier=round(multiplier, 4),
        )
    return RiskRuleResult("IR-002", RuleSeverity.PASS, "portfolio limits ok")


def validate_market_regime(inp: RiskInput) -> RiskRuleResult:
    if inp.side != OrderSide.BUY:
        return RiskRuleResult("IR-001", RuleSeverity.PASS, "not a buy order")
    if inp.vni_change_pct is not None and inp.vni_change_pct <= -3.0:
        return RiskRuleResult("IR-001", RuleSeverity.HARD_REJECT, "VN-Index circuit breaker")
    if inp.vni_above_ma100 is False:
        return RiskRuleResult("IR-001", RuleSeverity.SOFT_ADJUST, "VN-Index below MA100", sizing_multiplier=0.5)
    return RiskRuleResult("IR-001", RuleSeverity.PASS, "market regime ok")
