"""Trade-level risk rules."""

from __future__ import annotations

from multiagents_trading_assistant.risk.models import (
    OrderSide,
    RiskInput,
    RiskRuleResult,
    RuleSeverity,
)


def validate_stop_loss(inp: RiskInput) -> RiskRuleResult:
    if inp.side != OrderSide.BUY:
        return RiskRuleResult("IR-003", RuleSeverity.PASS, "not a buy order")
    entry = float(inp.entry_price or inp.limit_price or 0.0)
    sl = float(inp.stop_loss or 0.0)
    if entry <= 0:
        return RiskRuleResult("IR-003", RuleSeverity.HARD_REJECT, "missing entry price")
    if sl <= 0:
        return RiskRuleResult("IR-003", RuleSeverity.HARD_REJECT, "missing stop loss")
    if sl >= entry:
        return RiskRuleResult("IR-003", RuleSeverity.HARD_REJECT, "stop loss must be below entry")
    return RiskRuleResult("IR-003", RuleSeverity.PASS, "stop loss ok")


def validate_reward_risk(inp: RiskInput) -> RiskRuleResult:
    if inp.side != OrderSide.BUY or inp.reward_risk is None:
        return RiskRuleResult("IR-003-RR", RuleSeverity.PASS, "reward/risk not supplied")
    if inp.reward_risk < 1.0:
        return RiskRuleResult("IR-003-RR", RuleSeverity.HARD_REJECT, "reward/risk below 1.0")
    if inp.reward_risk < 1.5:
        return RiskRuleResult("IR-003-RR", RuleSeverity.SOFT_ADJUST, "reward/risk below preferred 1.5R", 0.7)
    return RiskRuleResult("IR-003-RR", RuleSeverity.PASS, "reward/risk ok")


def validate_liquidity(inp: RiskInput) -> RiskRuleResult:
    if inp.side != OrderSide.BUY or inp.avg_value_20 is None:
        return RiskRuleResult("IR-LIQ-001", RuleSeverity.PASS, "liquidity not supplied")
    if inp.avg_value_20 <= 0:
        return RiskRuleResult("IR-LIQ-001", RuleSeverity.HARD_REJECT, "invalid liquidity")
    if inp.avg_value_20 < inp.min_avg_value * 0.5:
        return RiskRuleResult("IR-LIQ-001", RuleSeverity.SOFT_ADJUST, "liquidity far below target", 0.3)
    if inp.avg_value_20 < inp.min_avg_value:
        return RiskRuleResult("IR-LIQ-001", RuleSeverity.SOFT_ADJUST, "liquidity below target", 0.6)
    return RiskRuleResult("IR-LIQ-001", RuleSeverity.PASS, "liquidity ok")


def validate_extension(inp: RiskInput) -> RiskRuleResult:
    if inp.side != OrderSide.BUY or inp.distance_to_ma20_pct is None:
        return RiskRuleResult("IR-EXT-001", RuleSeverity.PASS, "extension not supplied")
    if inp.distance_to_ma20_pct >= 15.0:
        return RiskRuleResult("IR-EXT-001", RuleSeverity.SOFT_ADJUST, "price very extended from MA20", 0.4)
    if inp.distance_to_ma20_pct >= 10.0:
        return RiskRuleResult("IR-EXT-001", RuleSeverity.SOFT_ADJUST, "price extended from MA20", 0.7)
    return RiskRuleResult("IR-EXT-001", RuleSeverity.PASS, "extension ok")


def validate_volume_confirmation(inp: RiskInput) -> RiskRuleResult:
    if inp.side != OrderSide.BUY or inp.volume_vs_ma20 is None:
        return RiskRuleResult("IR-VOL-001", RuleSeverity.PASS, "volume not supplied")
    if inp.volume_vs_ma20 < 0.7:
        return RiskRuleResult("IR-VOL-001", RuleSeverity.SOFT_ADJUST, "weak volume confirmation", 0.7)
    return RiskRuleResult("IR-VOL-001", RuleSeverity.PASS, "volume ok")
