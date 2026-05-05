"""Exchange-mandated and market-structure rules.

The HOSE rules here are based on the FPTS trading regulation page referenced
from risk_rules.md.
"""

from __future__ import annotations

from datetime import time
from math import floor

from multiagents_trading_assistant.risk.models import (
    AccountType,
    OrderSide,
    OrderType,
    RiskInput,
    RiskRuleResult,
    RuleSeverity,
)


FPTS_HOSE_URL = (
    "https://www.fpts.com.vn/customer-service/securities-trading/"
    "stock-trading-guide/trading-regulations/hose-trading-regulations/"
)
FPTS_SETTLEMENT_URL = (
    "https://www.fpts.com.vn/customer-service/securities-trading/"
    "securities-trading-settlement/"
)


def hose_tick_size(price: float, *, instrument_type: str = "stock") -> int:
    """Return HOSE tick size in VND."""
    if instrument_type.lower() in {"etf", "warrant", "covered_warrant"}:
        return 10
    if price <= 10_000:
        return 10
    if price < 50_000:
        return 50
    return 100


def round_to_tick(price: float, *, side: OrderSide, instrument_type: str = "stock") -> float:
    """Conservatively round a price to a valid HOSE tick.

    Buy prices round down to avoid accidentally crossing above the user's
    intended limit. Sell prices round up for the symmetric conservative behavior.
    """
    tick = hose_tick_size(price, instrument_type=instrument_type)
    if side == OrderSide.SELL:
        return float(((int(price) + tick - 1) // tick) * tick)
    return float((int(price) // tick) * tick)


def normalize_board_lot(quantity: int, *, allow_odd_lot: bool) -> int:
    if quantity <= 0:
        return 0
    if allow_odd_lot and quantity < 100:
        return quantity
    return floor(quantity / 100) * 100


def _session_name(t: time) -> str:
    if time(9, 0) <= t < time(9, 15):
        return "ATO"
    if time(9, 15) <= t <= time(11, 30):
        return "CONTINUOUS"
    if time(13, 0) <= t < time(14, 30):
        return "CONTINUOUS"
    if time(14, 30) <= t <= time(14, 45):
        return "ATC"
    return "CLOSED"


def validate_trading_session(inp: RiskInput) -> RiskRuleResult:
    if inp.side == OrderSide.HOLD:
        return RiskRuleResult("MR-HOSE-001", RuleSeverity.PASS, "hold action")
    if inp.exchange.upper() != "HOSE":
        return RiskRuleResult(
            "MR-HOSE-001",
            RuleSeverity.WARNING,
            f"session validator is HOSE-focused, got {inp.exchange}",
            source=FPTS_HOSE_URL,
        )

    session = _session_name(inp.signal_time.time())
    if session == "CLOSED":
        return RiskRuleResult(
            "MR-HOSE-001",
            RuleSeverity.HARD_REJECT,
            "outside HOSE matching sessions",
            source=FPTS_HOSE_URL,
        )
    if inp.order_type == OrderType.ATO and session != "ATO":
        return RiskRuleResult("MR-HOSE-004", RuleSeverity.HARD_REJECT, "ATO only valid 09:00-09:15", source=FPTS_HOSE_URL)
    if inp.order_type == OrderType.ATC and session != "ATC":
        return RiskRuleResult("MR-HOSE-004", RuleSeverity.HARD_REJECT, "ATC only valid 14:30-14:45", source=FPTS_HOSE_URL)
    if inp.order_type == OrderType.MTL and session != "CONTINUOUS":
        return RiskRuleResult("MR-HOSE-004", RuleSeverity.HARD_REJECT, "MTL only valid in continuous sessions", source=FPTS_HOSE_URL)
    return RiskRuleResult("MR-HOSE-001", RuleSeverity.PASS, f"valid HOSE session {session}", source=FPTS_HOSE_URL)


def validate_restricted_security(inp: RiskInput) -> RiskRuleResult:
    if not inp.is_restricted_security:
        return RiskRuleResult("MR-HOSE-003", RuleSeverity.PASS, "not restricted")
    if inp.order_type in {OrderType.ATO, OrderType.ATC}:
        return RiskRuleResult("MR-HOSE-003", RuleSeverity.WARNING, "restricted security requires auction-aware handling", source=FPTS_HOSE_URL)
    return RiskRuleResult(
        "MR-HOSE-003",
        RuleSeverity.HARD_REJECT,
        "restricted security cannot use normal continuous-session assumptions",
        source=FPTS_HOSE_URL,
    )


def validate_lot_size(inp: RiskInput, normalized_qty: int) -> RiskRuleResult:
    if inp.side == OrderSide.HOLD:
        return RiskRuleResult("MR-HOSE-008", RuleSeverity.PASS, "hold action")
    if inp.quantity <= 0:
        return RiskRuleResult("MR-HOSE-008", RuleSeverity.HARD_REJECT, "quantity must be positive", source=FPTS_HOSE_URL)
    if inp.quantity > 500_000:
        return RiskRuleResult("MR-HOSE-008", RuleSeverity.HARD_REJECT, "quantity exceeds 500,000 shares/order", source=FPTS_HOSE_URL)
    if 0 < inp.quantity < 100:
        if not inp.allow_odd_lot:
            return RiskRuleResult("MR-HOSE-008", RuleSeverity.HARD_REJECT, "odd-lot disabled and quantity < 100", source=FPTS_HOSE_URL)
        if inp.order_type != OrderType.LO:
            return RiskRuleResult("MR-HOSE-002", RuleSeverity.HARD_REJECT, "odd-lot orders must use LO", source=FPTS_HOSE_URL)
    if normalized_qty <= 0:
        return RiskRuleResult("MR-HOSE-008", RuleSeverity.HARD_REJECT, "quantity rounds to zero board lot", source=FPTS_HOSE_URL)
    if normalized_qty != inp.quantity:
        return RiskRuleResult("MR-HOSE-008", RuleSeverity.SOFT_ADJUST, f"quantity rounded down to {normalized_qty}", source=FPTS_HOSE_URL)
    return RiskRuleResult("MR-HOSE-008", RuleSeverity.PASS, "valid lot size", source=FPTS_HOSE_URL)


def validate_price_band(inp: RiskInput, normalized_price: float | None) -> RiskRuleResult:
    if inp.side == OrderSide.HOLD:
        return RiskRuleResult("MR-HOSE-010", RuleSeverity.PASS, "hold action")
    price = normalized_price if normalized_price is not None else inp.entry_price
    if price is None or price <= 0:
        return RiskRuleResult("MR-HOSE-010", RuleSeverity.HARD_REJECT, "missing executable price", source=FPTS_HOSE_URL)
    if inp.floor_price is None or inp.ceiling_price is None:
        return RiskRuleResult("MR-HOSE-010", RuleSeverity.HARD_REJECT, "missing official floor/ceiling price", source=FPTS_HOSE_URL)
    if price < inp.floor_price or price > inp.ceiling_price:
        return RiskRuleResult(
            "MR-HOSE-010",
            RuleSeverity.HARD_REJECT,
            f"price {price:g} outside floor/ceiling {inp.floor_price:g}-{inp.ceiling_price:g}",
            source=FPTS_HOSE_URL,
        )
    band_used = 0.0
    if inp.reference_price and inp.reference_price > 0:
        band_used = abs(price - inp.reference_price) / inp.reference_price
    if inp.side == OrderSide.BUY and inp.ceiling_price and inp.ceiling_price > 0:
        near_ceiling = price >= inp.ceiling_price * 0.985 or band_used >= 0.06
        if near_ceiling:
            return RiskRuleResult("MR-HOSE-010", RuleSeverity.SOFT_ADJUST, "buy price near ceiling", 0.5, FPTS_HOSE_URL)
    return RiskRuleResult("MR-HOSE-010", RuleSeverity.PASS, "price inside floor/ceiling", source=FPTS_HOSE_URL)


def validate_sellable_quantity(inp: RiskInput, normalized_qty: int) -> RiskRuleResult:
    if inp.side != OrderSide.SELL:
        return RiskRuleResult("MR-SETTLE-001", RuleSeverity.PASS, "not a sell order", source=FPTS_SETTLEMENT_URL)
    if normalized_qty > inp.settled_sellable_qty:
        return RiskRuleResult(
            "MR-SETTLE-001",
            RuleSeverity.HARD_REJECT,
            f"sell quantity {normalized_qty} exceeds settled sellable {inp.settled_sellable_qty}",
            source=FPTS_SETTLEMENT_URL,
        )
    return RiskRuleResult("MR-SETTLE-001", RuleSeverity.PASS, "sell quantity is settled", source=FPTS_SETTLEMENT_URL)


def validate_foreign_room(inp: RiskInput, normalized_qty: int) -> RiskRuleResult:
    if inp.side != OrderSide.BUY or inp.foreign_room_qty is None:
        return RiskRuleResult("MR-HOSE-013", RuleSeverity.PASS, "foreign room not applicable", source=FPTS_HOSE_URL)
    if inp.account_type == AccountType.FOREIGN and inp.foreign_room_qty < normalized_qty:
        return RiskRuleResult(
            "MR-HOSE-013",
            RuleSeverity.HARD_REJECT,
            f"foreign room {inp.foreign_room_qty} < quantity {normalized_qty}",
            source=FPTS_HOSE_URL,
        )
    if inp.account_type == AccountType.DOMESTIC and inp.foreign_room_qty < normalized_qty:
        return RiskRuleResult(
            "MR-HOSE-013",
            RuleSeverity.WARNING,
            "foreign room is tight, but domestic account is not hard-blocked",
            source=FPTS_HOSE_URL,
        )
    return RiskRuleResult("MR-HOSE-013", RuleSeverity.PASS, "foreign room ok", source=FPTS_HOSE_URL)
