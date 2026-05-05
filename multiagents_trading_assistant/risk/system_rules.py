"""System and data-integrity risk rules."""

from __future__ import annotations

from multiagents_trading_assistant.risk.models import (
    OrderSide,
    RiskInput,
    RiskRuleResult,
    RuleSeverity,
)


def validate_data_integrity(inp: RiskInput) -> RiskRuleResult:
    if not inp.symbol:
        return RiskRuleResult("IR-005", RuleSeverity.HARD_REJECT, "missing symbol")
    if inp.side == OrderSide.HOLD:
        return RiskRuleResult("IR-005", RuleSeverity.PASS, "hold action")
    if not inp.exchange:
        return RiskRuleResult("IR-005", RuleSeverity.HARD_REJECT, "missing exchange")
    if inp.reference_price is None or inp.reference_price <= 0:
        return RiskRuleResult("IR-005", RuleSeverity.HARD_REJECT, "missing reference price")
    if inp.floor_price is None or inp.ceiling_price is None:
        return RiskRuleResult("IR-005", RuleSeverity.HARD_REJECT, "missing floor/ceiling")
    if inp.side == OrderSide.SELL and inp.settled_sellable_qty <= 0:
        return RiskRuleResult("IR-005", RuleSeverity.HARD_REJECT, "missing sellable quantity")
    return RiskRuleResult("IR-005", RuleSeverity.PASS, "required data present")


def validate_no_short(inp: RiskInput, normalized_qty: int) -> RiskRuleResult:
    if inp.side != OrderSide.SELL:
        return RiskRuleResult("IR-004", RuleSeverity.PASS, "not a sell order")
    if normalized_qty > inp.position_qty:
        return RiskRuleResult("IR-004", RuleSeverity.HARD_REJECT, "sell quantity exceeds position quantity")
    return RiskRuleResult("IR-004", RuleSeverity.PASS, "no net short")
