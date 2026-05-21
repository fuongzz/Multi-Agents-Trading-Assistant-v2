"""Phase 1 agentic schemas.

These schemas freeze deterministic strategy output before GenAI review. The
LLM layer may review or propose execution details, but it must not mutate the
source strategy signal.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrategySignal(BaseModel):
    """Immutable signal produced by a deterministic alpha engine."""

    symbol: str
    signal_date: str
    as_of_date: str
    source: str = "core3"
    strategy_name: str
    strategy_family: str = ""
    setup_type: str = "EDGE"
    feature_date: str | None = None
    priority_score: float | None = None
    edge_score: float | None = None
    edge_rank: float | None = None
    edge_rank_score: float | None = None
    filters_failed: list[str] = Field(default_factory=list)
    risk: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    immutable: bool = True


class EvidencePacket(BaseModel):
    """Broker-grade context passed to agentic review.

    Field names are the canonical Phase 2 broker-grade taxonomy. Legacy names
    (`market_context`, `indicators`, `edge_strategy`, `fundamentals`) remain
    accessible via aliases (input/serialization) and read-only `@property`
    accessors for in-process readers.
    """

    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    as_of_date: str
    market: dict[str, Any] = Field(default_factory=dict, alias="market_context")
    sector: dict[str, Any] = Field(default_factory=dict)
    technical: dict[str, Any] = Field(default_factory=dict, alias="indicators")
    money_flow: dict[str, Any] = Field(default_factory=dict)
    edge: dict[str, Any] = Field(default_factory=dict, alias="edge_strategy")
    fundamental: dict[str, Any] = Field(default_factory=dict, alias="fundamentals")
    news: dict[str, Any] = Field(default_factory=dict)
    portfolio: dict[str, Any] = Field(default_factory=dict)
    data_quality: dict[str, Any] = Field(default_factory=dict)

    @property
    def market_context(self) -> dict[str, Any]:
        return self.market

    @property
    def indicators(self) -> dict[str, Any]:
        return self.technical

    @property
    def edge_strategy(self) -> dict[str, Any]:
        return self.edge

    @property
    def fundamentals(self) -> dict[str, Any]:
        return self.fundamental


class ReasonCode(str, Enum):
    """Enumerated reasons the triage gate may use to downgrade a signal.

    The gate may only choose from this fixed menu. LLM output that does not
    match an enum value is rejected by the validator and falls back to
    APPROVE (fail-safe — gate never invents new risks).
    """

    NEWS_RISK = "NEWS_RISK"
    BAD_LIQUIDITY = "BAD_LIQUIDITY"
    MARKET_RISK_OFF = "MARKET_RISK_OFF"
    OVEREXTENDED_GAP = "OVEREXTENDED_GAP"
    SECTOR_WEAK = "SECTOR_WEAK"
    DATA_QUALITY_ISSUE = "DATA_QUALITY_ISSUE"
    PORTFOLIO_CONCENTRATION = "PORTFOLIO_CONCENTRATION"
    RECENT_SAME_SYMBOL = "RECENT_SAME_SYMBOL"


class LossCategory(str, Enum):
    """Enumerated categories for post-trade review of losing trades."""

    ENTRY_CHASE = "ENTRY_CHASE"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    REGIME_SHIFT = "REGIME_SHIFT"
    WEAK_SECTOR = "WEAK_SECTOR"
    DATA_ISSUE = "DATA_ISSUE"
    EXIT_TOO_EARLY = "EXIT_TOO_EARLY"
    EXIT_TOO_LATE = "EXIT_TOO_LATE"
    STRUCTURE_BREAK = "STRUCTURE_BREAK"   # SL mid-trade: signal worked initially then broke
    UNCLASSIFIED = "UNCLASSIFIED"


class AgenticReview(BaseModel):
    """Phase 3a triage gate output. Bounded decisions over a frozen signal."""

    symbol: str
    as_of_date: str
    action: Literal["APPROVE", "REDUCE_SIZE", "SKIP", "WAIT_FOR_ENTRY"]
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    size_multiplier: float | None = Field(default=None, ge=0.3, le=1.0)
    wait_condition: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = ""

    @field_validator("rationale")
    @classmethod
    def _rationale_len(cls, v: str) -> str:
        return v[:200] if v else ""

    def model_post_init(self, __context: Any) -> None:  # pragma: no cover - schema invariant
        if self.action == "REDUCE_SIZE" and self.size_multiplier is None:
            raise ValueError("REDUCE_SIZE requires size_multiplier")
        if self.action == "WAIT_FOR_ENTRY" and not self.wait_condition:
            raise ValueError("WAIT_FOR_ENTRY requires wait_condition")
        if self.action in ("APPROVE", "SKIP", "WAIT_FOR_ENTRY"):
            object.__setattr__(self, "size_multiplier", None) if self.action != "WAIT_FOR_ENTRY" else None
        if self.action != "APPROVE" and not self.reason_codes:
            raise ValueError(f"{self.action} requires at least one reason_code")


class ExecutionProposal(BaseModel):
    """Execution-only proposal. Final validation belongs to risk."""

    action: Literal["BUY", "SELL", "HOLD"]
    entry_zone: tuple[float, float] | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    position_pct: float | None = Field(default=None, ge=0.0, le=1.0)
    order_type: str = "LO"
    rationale: list[str] = Field(default_factory=list)


class FinalDecision(BaseModel):
    """Deterministic final control output."""

    symbol: str
    as_of_date: str
    final_action: Literal["BUY", "SELL", "HOLD"]
    allowed: bool
    source_strategy: str = ""
    override_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    source_rule_ids: list[str] = Field(default_factory=list)
