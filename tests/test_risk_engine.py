from datetime import datetime

from multiagents_trading_assistant.risk import (
    AccountType,
    OrderSide,
    OrderType,
    RiskEngine,
    RiskInput,
)
from multiagents_trading_assistant.risk.market_rules import hose_tick_size, round_to_tick


def _base(**overrides):
    data = dict(
        symbol="VCB",
        exchange="HOSE",
        side=OrderSide.BUY,
        order_type=OrderType.LO,
        signal_time=datetime(2026, 5, 4, 10, 0),
        quantity=1200,
        limit_price=87430,
        reference_price=86000,
        floor_price=80000,
        ceiling_price=92000,
        proposed_nav_pct=10.0,
        entry_price=87430,
        stop_loss=83000,
        reward_risk=1.8,
        avg_value_20=20_000_000_000,
        vni_above_ma100=True,
        distance_to_ma20_pct=4.0,
        volume_vs_ma20=1.1,
    )
    data.update(overrides)
    return RiskInput(**data)


def test_hose_tick_size_and_rounding():
    assert hose_tick_size(10_000) == 10
    assert hose_tick_size(10_050) == 50
    assert hose_tick_size(50_000) == 100
    assert round_to_tick(87_430, side=OrderSide.BUY) == 87_400
    assert round_to_tick(87_430, side=OrderSide.SELL) == 87_500


def test_valid_buy_is_allowed_and_normalized():
    decision = RiskEngine().evaluate(_base())
    assert decision.allowed is True
    assert decision.final_action == OrderSide.BUY
    assert decision.normalized_order.limit_price == 87_400
    assert decision.normalized_order.quantity == 1200


def test_outside_session_rejects_order():
    decision = RiskEngine().evaluate(_base(signal_time=datetime(2026, 5, 4, 12, 0)))
    assert decision.allowed is False
    assert decision.final_action == OrderSide.HOLD
    assert "MR-HOSE-001" in decision.rejected_by


def test_odd_lot_requires_lo_and_explicit_enablement():
    blocked = RiskEngine().evaluate(_base(quantity=55))
    assert blocked.allowed is False
    assert "MR-HOSE-008" in blocked.rejected_by

    allowed = RiskEngine().evaluate(_base(quantity=55, allow_odd_lot=True, order_type=OrderType.LO))
    assert allowed.allowed is True
    assert allowed.normalized_order.quantity == 55

    wrong_type = RiskEngine().evaluate(_base(quantity=55, allow_odd_lot=True, order_type=OrderType.ATO))
    assert wrong_type.allowed is False
    assert "MR-HOSE-004" in wrong_type.rejected_by or "MR-HOSE-002" in wrong_type.rejected_by


def test_sell_cannot_exceed_settled_sellable_quantity():
    decision = RiskEngine().evaluate(
        _base(
            side=OrderSide.SELL,
            quantity=1000,
            position_qty=1000,
            settled_sellable_qty=500,
            stop_loss=None,
            proposed_nav_pct=0,
        )
    )
    assert decision.allowed is False
    assert "MR-SETTLE-001" in decision.rejected_by


def test_foreign_account_room_is_hard_reject():
    decision = RiskEngine().evaluate(
        _base(account_type=AccountType.FOREIGN, foreign_room_qty=500, quantity=1000)
    )
    assert decision.allowed is False
    assert "MR-HOSE-013" in decision.rejected_by


def test_soft_rules_reduce_size_without_rejecting():
    decision = RiskEngine().evaluate(
        _base(
            vni_above_ma100=False,
            distance_to_ma20_pct=12.0,
            volume_vs_ma20=0.6,
            avg_value_20=7_000_000_000,
        )
    )
    assert decision.allowed is True
    assert decision.final_action == OrderSide.BUY
    assert 0 < decision.sizing_modifier < 1
    assert decision.normalized_order.nav_pct < 10.0
