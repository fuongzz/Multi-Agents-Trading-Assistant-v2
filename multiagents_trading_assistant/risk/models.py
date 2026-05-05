"""Data contracts for the deterministic risk layer."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class OrderType(str, Enum):
    LO = "LO"
    ATO = "ATO"
    ATC = "ATC"
    MTL = "MTL"


class AccountType(str, Enum):
    DOMESTIC = "DOMESTIC"
    FOREIGN = "FOREIGN"


class RuleSeverity(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    SOFT_ADJUST = "SOFT_ADJUST"
    HARD_REJECT = "HARD_REJECT"


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Minimal portfolio state needed by risk rules.

    Percent fields are expressed as percent of NAV, not fractions.
    """

    cash_pct: float = 100.0
    open_positions: int = 0
    total_exposure_pct: float = 0.0
    total_risk_pct: float = 0.0
    symbol_exposure_pct: float = 0.0
    sector_exposure_pct: float = 0.0


@dataclass(frozen=True)
class PortfolioLimits:
    max_open_positions: int = 5
    max_total_exposure_pct: float = 50.0
    max_total_risk_pct: float = 6.0
    max_position_pct: float = 10.0
    max_per_sector_pct: float = 20.0


@dataclass(frozen=True)
class RiskInput:
    symbol: str
    exchange: str
    side: OrderSide
    order_type: OrderType
    signal_time: datetime
    quantity: int = 0
    limit_price: Optional[float] = None
    reference_price: Optional[float] = None
    ceiling_price: Optional[float] = None
    floor_price: Optional[float] = None
    account_type: AccountType = AccountType.DOMESTIC
    foreign_room_qty: Optional[int] = None
    position_qty: int = 0
    settled_sellable_qty: int = 0
    proposed_nav_pct: float = 0.0
    stop_loss: Optional[float] = None
    entry_price: Optional[float] = None
    reward_risk: Optional[float] = None
    avg_value_20: Optional[float] = None
    min_avg_value: float = 10_000_000_000.0
    vni_change_pct: Optional[float] = None
    vni_above_ma100: Optional[bool] = None
    distance_to_ma20_pct: Optional[float] = None
    volume_vs_ma20: Optional[float] = None
    is_restricted_security: bool = False
    allow_odd_lot: bool = False
    allow_soft_adjust: bool = True
    portfolio: PortfolioSnapshot = field(default_factory=PortfolioSnapshot)
    limits: PortfolioLimits = field(default_factory=PortfolioLimits)

    def copy_with(self, **updates) -> "RiskInput":
        return replace(self, **updates)


@dataclass(frozen=True)
class NormalizedOrder:
    symbol: str
    exchange: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: Optional[float]
    nav_pct: float


@dataclass(frozen=True)
class RiskRuleResult:
    rule_id: str
    severity: RuleSeverity
    message: str
    sizing_multiplier: float = 1.0
    source: Optional[str] = None


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    final_action: OrderSide
    normalized_order: NormalizedOrder
    override_reason: Optional[str]
    warnings: list[str]
    sizing_modifier: float
    evaluated_rules: list[RiskRuleResult]
    rejected_by: list[str]
