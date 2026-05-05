"""Risk engine that combines hard market rules and soft system rules."""

from __future__ import annotations

from multiagents_trading_assistant.risk.market_rules import (
    normalize_board_lot,
    round_to_tick,
    validate_foreign_room,
    validate_lot_size,
    validate_price_band,
    validate_restricted_security,
    validate_sellable_quantity,
    validate_trading_session,
)
from multiagents_trading_assistant.risk.models import (
    NormalizedOrder,
    OrderSide,
    RiskDecision,
    RiskInput,
    RiskRuleResult,
    RuleSeverity,
)
from multiagents_trading_assistant.risk.portfolio_rules import (
    validate_market_regime,
    validate_portfolio_limits,
)
from multiagents_trading_assistant.risk.system_rules import (
    validate_data_integrity,
    validate_no_short,
)
from multiagents_trading_assistant.risk.trade_rules import (
    validate_extension,
    validate_liquidity,
    validate_reward_risk,
    validate_stop_loss,
    validate_volume_confirmation,
)


class RiskEngine:
    """Rule-based final gate before an order can be accepted."""

    def evaluate(self, inp: RiskInput) -> RiskDecision:
        normalized_qty = normalize_board_lot(inp.quantity, allow_odd_lot=inp.allow_odd_lot)
        raw_price = inp.limit_price if inp.limit_price is not None else inp.entry_price
        normalized_price = (
            round_to_tick(float(raw_price), side=inp.side)
            if raw_price is not None and raw_price > 0 and inp.side != OrderSide.HOLD
            else raw_price
        )

        results: list[RiskRuleResult] = []

        # Mandatory gates first.
        results.append(validate_data_integrity(inp))
        results.append(validate_no_short(inp, normalized_qty))
        results.append(validate_trading_session(inp))
        results.append(validate_restricted_security(inp))
        results.append(validate_lot_size(inp, normalized_qty))
        results.append(validate_price_band(inp, normalized_price))
        results.append(validate_sellable_quantity(inp, normalized_qty))
        results.append(validate_foreign_room(inp, normalized_qty))

        sizing_modifier = self._combined_multiplier(results, include_soft=inp.allow_soft_adjust)
        nav_after_market = round(inp.proposed_nav_pct * sizing_modifier, 4)

        # Internal risk can reduce sizing further, or reject when no capacity.
        internal_results = [
            validate_portfolio_limits(inp, nav_after_market),
            validate_stop_loss(inp),
            validate_reward_risk(inp),
            validate_liquidity(inp),
            validate_extension(inp),
            validate_volume_confirmation(inp),
            validate_market_regime(inp),
        ]
        results.extend(internal_results)
        sizing_modifier = self._combined_multiplier(results, include_soft=inp.allow_soft_adjust)

        rejected = [r.rule_id for r in results if r.severity == RuleSeverity.HARD_REJECT]
        allowed = len(rejected) == 0

        nav_pct = round(inp.proposed_nav_pct * sizing_modifier, 4)
        final_qty = normalized_qty
        if inp.proposed_nav_pct > 0 and nav_pct < inp.proposed_nav_pct and normalized_qty > 0:
            final_qty = normalize_board_lot(
                int(normalized_qty * nav_pct / inp.proposed_nav_pct),
                allow_odd_lot=inp.allow_odd_lot,
            )

        if inp.side == OrderSide.HOLD:
            final_action = OrderSide.HOLD
            allowed = True
        elif not allowed or final_qty <= 0:
            final_action = OrderSide.HOLD
            if final_qty <= 0 and not rejected:
                rejected.append("IR-SIZE-001")
                results.append(RiskRuleResult("IR-SIZE-001", RuleSeverity.HARD_REJECT, "sizing rounds to zero"))
            allowed = False
        else:
            final_action = inp.side

        warnings = [
            r.message for r in results
            if r.severity in {RuleSeverity.WARNING, RuleSeverity.SOFT_ADJUST}
        ]
        override_reason = None
        if not allowed:
            first_reject = next((r for r in results if r.severity == RuleSeverity.HARD_REJECT), None)
            override_reason = first_reject.message if first_reject else "risk rejected order"

        normalized_order = NormalizedOrder(
            symbol=inp.symbol,
            exchange=inp.exchange,
            side=final_action,
            order_type=inp.order_type,
            quantity=final_qty if final_action != OrderSide.HOLD else 0,
            limit_price=normalized_price,
            nav_pct=nav_pct if final_action != OrderSide.HOLD else 0.0,
        )
        return RiskDecision(
            allowed=allowed,
            final_action=final_action,
            normalized_order=normalized_order,
            override_reason=override_reason,
            warnings=warnings,
            sizing_modifier=sizing_modifier,
            evaluated_rules=results,
            rejected_by=rejected,
        )

    @staticmethod
    def _combined_multiplier(results: list[RiskRuleResult], *, include_soft: bool) -> float:
        if not include_soft:
            return 1.0
        multiplier = 1.0
        for result in results:
            if result.severity == RuleSeverity.SOFT_ADJUST:
                multiplier *= max(0.0, min(1.0, result.sizing_multiplier))
        return round(multiplier, 6)
