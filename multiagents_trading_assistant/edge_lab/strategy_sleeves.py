"""Strategy catalog and portfolio sleeve defaults.

The screener and the trading portfolio have different objectives:

- Screener: expose every useful setup/filter so users can discover ideas.
- Portfolio: allocate scarce capital slots only to validated sleeves.

This module is intentionally configuration-oriented. It prevents research
strategies from being accidentally treated as production capital allocators just
because they are useful as user-facing filters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


StrategyEngine = Literal["edge_lab", "custom_ichimoku", "custom_darvas"]
StrategyStatus = Literal["production", "research", "shadow"]
SleeveStatus = Literal["active", "shadow", "disabled"]


@dataclass(frozen=True)
class StrategyCatalogItem:
    strategy_id: str
    label: str
    family: str
    engine: StrategyEngine
    status: StrategyStatus
    screener_enabled: bool
    portfolio_role: str
    native_exit_required: bool = False
    notes: str = ""


@dataclass(frozen=True)
class PortfolioSleeve:
    sleeve_id: str
    label: str
    status: SleeveStatus
    strategy_ids: tuple[str, ...]
    max_positions: int
    max_nav_fraction: float
    allowed_regimes: tuple[str, ...] = ("RISK_ON_UPTREND", "NEUTRAL_UPTREND")
    native_exit_required: bool = False
    notes: str = ""


@dataclass(frozen=True)
class StrategyArchitecture:
    screener_strategy_ids: tuple[str, ...]
    portfolio_sleeves: tuple[PortfolioSleeve, ...]
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "screener_strategy_ids": list(self.screener_strategy_ids),
            "portfolio_sleeves": [asdict(item) for item in self.portfolio_sleeves],
            "notes": self.notes,
        }


MVP_EDGE_STRATEGIES: tuple[str, ...] = (
    "leader_pullback_market_regime_v3",
    "leader_pullback_market_healthy_v2",
    "breakout_55_smt_v1",
    "money_cycle_reset_smt_confirm",
    "smart_money_strong_market_healthy",
    "accumulation_breakout_smt_v1",
    "breakout_after_accumulation_v3",
    "compression_breakout_smt_v1",
    "mean_reversion_uptrend_ma50_v1",
)


RESEARCH_EDGE_STRATEGIES: tuple[str, ...] = (
    "faber_gtaa_trend_filter_v1",
    "minervini_trend_template_v1",
    "weinstein_stage2_breakout_v1",
    "breakout_market_cycle_defensive_v1",
    "oil_gas_smart_money_recovery_v1",
    "oil_gas_breakout_rotation_v1",
    "oil_gas_quality_recovery_v2",
    "oil_gas_leader_breakout_v2",
    "theme_flow_recovery_v1",
    "theme_flow_breakout_v1",
    "theme_flow_sector_leader_v1",
    "theme_flow_leader_quality_v2",
    "theme_flow_shock_reclaim_v2",
    "theme_flow_breakout_quality_v2",
)


CUSTOM_STRATEGIES: tuple[StrategyCatalogItem, ...] = (
    StrategyCatalogItem(
        strategy_id="ich_vn_fast_turtle55_ich_cloud_close_uptrend_rs50",
        label="Ichimoku Turtle55 Trend",
        family="ichimoku_trend",
        engine="custom_ichimoku",
        status="research",
        screener_enabled=True,
        portfolio_role="aggressive_trend_sleeve",
        native_exit_required=True,
        notes="Strong standalone trend sleeve, but should keep cloud-close/MAX_HOLD style exit.",
    ),
    StrategyCatalogItem(
        strategy_id="ich_qmv_fast_cloud_break_ich_ladder_bull_rs50_kumoq",
        label="QMV-like Ichimoku Cloud Break Ladder",
        family="ichimoku_trend",
        engine="custom_ichimoku",
        status="research",
        screener_enabled=True,
        portfolio_role="robust_trend_sleeve",
        native_exit_required=True,
        notes="More robust long-window candidate; not validated as blind shared-pool signal.",
    ),
    StrategyCatalogItem(
        strategy_id="darvas_focus_w20_breakout_atr_trail_uptrend_rs75_v1p2",
        label="Darvas Box Breakout W20",
        family="darvas_breakout",
        engine="custom_darvas",
        status="research",
        screener_enabled=True,
        portfolio_role="diversification_or_watchlist_sleeve",
        native_exit_required=True,
        notes="Useful as breakout/watchlist filter; weak as direct shared-pool allocator.",
    ),
)


EDGE_CATALOG: tuple[StrategyCatalogItem, ...] = tuple(
    StrategyCatalogItem(
        strategy_id=strategy_id,
        label=strategy_id,
        family="mvp_edge",
        engine="edge_lab",
        status="production",
        screener_enabled=True,
        portfolio_role="core_mvp",
        native_exit_required=False,
        notes="Validated inside current live pipeline shared-pool execution stack.",
    )
    for strategy_id in MVP_EDGE_STRATEGIES
) + tuple(
    StrategyCatalogItem(
        strategy_id=strategy_id,
        label=strategy_id,
        family="research_edge",
        engine="edge_lab",
        status="shadow",
        screener_enabled=True,
        portfolio_role="research_filter",
        native_exit_required=False,
        notes="Available for screener/research; not enabled for capital by default.",
    )
    for strategy_id in RESEARCH_EDGE_STRATEGIES
)


STRATEGY_CATALOG: tuple[StrategyCatalogItem, ...] = EDGE_CATALOG + CUSTOM_STRATEGIES


DEFAULT_PORTFOLIO_SLEEVES: tuple[PortfolioSleeve, ...] = (
    PortfolioSleeve(
        sleeve_id="core_mvp",
        label="Core MVP Shared Pool",
        status="active",
        strategy_ids=MVP_EDGE_STRATEGIES,
        max_positions=5,
        max_nav_fraction=1.0,
        allowed_regimes=("RISK_ON_UPTREND", "RISK_ON_RECOVERY", "NEUTRAL_UPTREND"),
        native_exit_required=False,
        notes="Current best production allocation. Do not let research filters blindly displace these slots.",
    ),
    PortfolioSleeve(
        sleeve_id="darvas_breakout_watchlist",
        label="Darvas Breakout Watchlist Sleeve",
        status="shadow",
        strategy_ids=("darvas_focus_w20_breakout_atr_trail_uptrend_rs75_v1p2",),
        max_positions=1,
        max_nav_fraction=0.15,
        allowed_regimes=("RISK_ON_UPTREND",),
        native_exit_required=True,
        notes="Shadow by default. Promote only after native-exit sleeve backtest beats displacement risk.",
    ),
    PortfolioSleeve(
        sleeve_id="ichimoku_trend_research",
        label="Ichimoku Trend Research Sleeve",
        status="shadow",
        strategy_ids=(
            "ich_vn_fast_turtle55_ich_cloud_close_uptrend_rs50",
            "ich_qmv_fast_cloud_break_ich_ladder_bull_rs50_kumoq",
        ),
        max_positions=1,
        max_nav_fraction=0.20,
        allowed_regimes=("RISK_ON_UPTREND",),
        native_exit_required=True,
        notes="Requires native cloud/Kijun exit logic; do not route through generic shared-pool exits.",
    ),
)


def catalog_by_id() -> dict[str, StrategyCatalogItem]:
    return {item.strategy_id: item for item in STRATEGY_CATALOG}


def screener_strategy_ids(include_shadow: bool = True) -> list[str]:
    allowed = {"production", "research", "shadow"} if include_shadow else {"production", "research"}
    return [
        item.strategy_id
        for item in STRATEGY_CATALOG
        if item.screener_enabled and item.status in allowed
    ]


def active_portfolio_sleeves(include_shadow: bool = False) -> list[PortfolioSleeve]:
    statuses = {"active", "shadow"} if include_shadow else {"active"}
    return [item for item in DEFAULT_PORTFOLIO_SLEEVES if item.status in statuses]


def default_architecture(include_shadow_sleeves: bool = False) -> StrategyArchitecture:
    return StrategyArchitecture(
        screener_strategy_ids=tuple(screener_strategy_ids(include_shadow=True)),
        portfolio_sleeves=tuple(active_portfolio_sleeves(include_shadow=include_shadow_sleeves)),
        notes=(
            "Screener exposes broad strategy coverage for users. Portfolio uses "
            "sleeves so scarce capital slots are not blindly displaced by research filters."
        ),
    )


def catalog_rows() -> list[dict]:
    return [asdict(item) for item in STRATEGY_CATALOG]
