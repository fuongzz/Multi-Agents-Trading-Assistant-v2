"""Smoke test cho Position Lifecycle architecture.

Tạo synthetic OHLCV với pattern reversal rõ:
  - 30 bar đi xuống (downtrend)
  - 5 bar tạo đáy
  - 30 bar đi lên (uptrend với HH/HL)

Inject signal DOUBLE_BOTTOM tại bar tạo đáy → kỳ vọng:
  PROBE (30%) → ADDED_STRENGTH (60%) → FULL (90%) → CLOSED có lãi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from multiagents_trading_assistant.backtest.execution import ExecutionConfig
from multiagents_trading_assistant.backtest.lifecycle import (
    Leg,
    LegReason,
    Position,
    PositionStage,
    Signal,
)
from multiagents_trading_assistant.backtest.lifecycle_engine import LifecycleEngine
from multiagents_trading_assistant.backtest.lifecycle_metrics import (
    portfolio_metrics,
    position_metrics,
)
from multiagents_trading_assistant.backtest.playbook import (
    CoreTrendPlaybook,
    PlaybookRouter,
    ScaleInReversalPlaybook,
    SingleEntryPlaybook,
)
from multiagents_trading_assistant.backtest.portfolio import (
    PortfolioConstraints,
    PortfolioEngine,
)
from multiagents_trading_assistant.backtest.regime import (
    Regime,
    classify_regime,
    is_setup_allowed_for_regime,
)
from multiagents_trading_assistant.backtest.sector_rotation import SectorRotationModel


def _make_reversal_df(n_down=30, n_bottom=5, n_up=40) -> pd.DataFrame:
    """OHLCV synthetic: downtrend → bottom → strong uptrend."""
    rng = np.random.default_rng(42)
    bars = []
    price = 100.0
    start_date = pd.Timestamp("2024-01-02")

    # Downtrend
    for _ in range(n_down):
        chg = rng.uniform(-0.02, 0.005)
        new = price * (1 + chg)
        o, c = price, new
        h = max(o, c) * (1 + abs(rng.uniform(0, 0.01)))
        l = min(o, c) * (1 - abs(rng.uniform(0, 0.01)))
        bars.append([o, h, l, c, rng.uniform(800_000, 1_200_000)])
        price = new

    # Bottom (sideways near lows)
    bottom_price = price
    for _ in range(n_bottom):
        chg = rng.uniform(-0.005, 0.005)
        new = bottom_price * (1 + chg)
        o, c = bottom_price, new
        h = max(o, c) * 1.005
        l = min(o, c) * 0.995
        bars.append([o, h, l, c, rng.uniform(900_000, 1_500_000)])
        bottom_price = new

    # Uptrend (HH+HL with steady gains)
    up_price = bottom_price
    for i in range(n_up):
        chg = rng.uniform(0.005, 0.025)
        new = up_price * (1 + chg)
        o = up_price
        c = new
        h = c * (1 + abs(rng.uniform(0, 0.005)))
        l = o * (1 - abs(rng.uniform(0, 0.005)))
        bars.append([o, h, l, c, rng.uniform(1_200_000, 2_000_000)])
        up_price = new

    df = pd.DataFrame(bars, columns=["open", "high", "low", "close", "volume"])
    df.insert(0, "date", pd.date_range(start_date, periods=len(df), freq="D"))
    return df


def _signal_at_bottom_factory(signal_bar_idx: int):
    """Detector phát Signal DOUBLE_BOTTOM tại 1 bar duy nhất."""

    def detector(df, i):
        if i != signal_bar_idx:
            return None
        row = df.iloc[i]
        recent_low = float(df["low"].iloc[max(0, i - 7):i + 1].min())
        return Signal(
            symbol="VCB",
            date=str(row["date"])[:10],
            setup_type="DOUBLE_BOTTOM",
            confluence_score=68.0,
            entry_zone_low=float(row["close"]) * 0.99,
            entry_zone_high=float(row["close"]) * 1.01,
            suggested_sl=recent_low * 0.97,
            atr=float(row["close"]) * 0.02,
            regime="DOWNTREND",
            reasons=["test signal"],
        )

    return detector


# ─────────────────────────────────────────────────────────────────────────────


def test_scale_in_reversal_full_lifecycle():
    df = _make_reversal_df()
    signal_bar = 32  # ngay sau khi tạo đáy 5 bar

    pb = ScaleInReversalPlaybook(intended_total_pct=9.0)
    router = PlaybookRouter(playbooks=[pb], default=pb)
    portfolio = PortfolioEngine(PortfolioConstraints(max_open_positions=5))
    engine = LifecycleEngine(
        router=router,
        portfolio=portfolio,
        execution_cfg=ExecutionConfig(settlement_days=2),
        lookback=20,
    )

    result = engine.run_symbol(
        "VCB",
        df,
        signal_detector=_signal_at_bottom_factory(signal_bar),
    )

    closed = result["closed_positions"]
    assert len(closed) == 1, f"Expected 1 closed position, got {len(closed)}"
    pos = closed[0]
    pm = position_metrics(pos)

    # Phải có ít nhất 1 buy + 1 sell
    assert pm["num_buys"] >= 1, f"No buys recorded: {pm}"
    assert pm["num_sells"] >= 1, f"No sells recorded: {pm}"

    # Stage progression — kỳ vọng ít nhất reach ADDED_STRENGTH
    stages = pm["stages_visited"]
    assert "PROBE" in stages, f"Stages: {stages}"

    # Print kết quả cho người chạy thấy
    print("\n=== Position Metrics ===")
    for k, v in pm.items():
        print(f"  {k}: {v}")
    print("\n=== Audit Log (last 15) ===")
    for entry in result["audit_log"][-15:]:
        print(f"  {entry['date']} | {entry['action']:12s} | {entry['status']:25s} | {entry['note']}")
    print("\n=== Portfolio metrics ===")
    pf_metrics = portfolio_metrics(closed, label="scale_in_reversal_smoke")
    for k, v in pf_metrics.items():
        print(f"  {k}: {v}")


def test_single_entry_backward_compat():
    """SingleEntryPlaybook chạy đúng theo behavior cũ: 1 buy → 1 sell."""
    df = _make_reversal_df()
    signal_bar = 32

    pb = SingleEntryPlaybook()
    router = PlaybookRouter(playbooks=[pb], default=pb)
    engine = LifecycleEngine(
        router=router,
        portfolio=PortfolioEngine(),
        lookback=20,
    )

    result = engine.run_symbol(
        "VCB",
        df,
        signal_detector=_signal_at_bottom_factory(signal_bar),
    )
    closed = result["closed_positions"]
    assert len(closed) == 1
    pos = closed[0]
    pm = position_metrics(pos)
    assert pm["num_buys"] == 1, f"SingleEntry must have exactly 1 buy: {pm}"
    print("\n=== Single Entry Metrics ===")
    for k, v in pm.items():
        print(f"  {k}: {v}")


def test_portfolio_constraint_blocks_open():
    """PortfolioConstraints chặn được khi cash không đủ."""
    df = _make_reversal_df()
    pb = SingleEntryPlaybook(sizing_map={"high": 5.0, "medium": 3.0, "low": 2.0})
    router = PlaybookRouter(playbooks=[pb], default=pb)

    # Đặt max_position_pct = 1.5% → 2-5% buy luôn bị reject
    portfolio = PortfolioEngine(PortfolioConstraints(max_position_pct=1.5))
    engine = LifecycleEngine(router=router, portfolio=portfolio, lookback=20)

    result = engine.run_symbol(
        "VCB", df, signal_detector=_signal_at_bottom_factory(32)
    )
    assert len(result["closed_positions"]) == 0
    rejects = [e for e in result["audit_log"] if "portfolio_reject" in e["status"]]
    assert len(rejects) > 0, "Should have at least 1 portfolio_reject"
    print(f"\nBlocked {len(rejects)} open attempts (expected).")


def test_open_position_keeps_original_playbook_when_default_differs():
    """Scale-in position must keep scaling even when router default is single-entry."""
    df = _make_reversal_df()
    scale = ScaleInReversalPlaybook(intended_total_pct=9.0)
    single = SingleEntryPlaybook()
    router = PlaybookRouter(playbooks=[scale], default=single)
    engine = LifecycleEngine(
        router=router,
        portfolio=PortfolioEngine(),
        execution_cfg=ExecutionConfig(settlement_days=2),
        lookback=20,
    )

    result = engine.run_symbol(
        "VCB",
        df,
        signal_detector=_signal_at_bottom_factory(32),
    )
    closed = result["closed_positions"]
    assert len(closed) == 1
    pm = position_metrics(closed[0])
    assert pm["num_buys"] >= 2, f"Expected scale-in adds, got: {pm}"
    assert closed[0].playbook_name == "scale_in_reversal"


def test_buy_rejected_when_fill_outside_entry_zone():
    """A signal should not buy if next open has run outside the recommended zone."""
    df = _make_reversal_df()

    def detector(data, i):
        if i != 32:
            return None
        row = data.iloc[i]
        return Signal(
            symbol="VCB",
            date=str(row["date"])[:10],
            setup_type="DOUBLE_BOTTOM",
            confluence_score=68.0,
            entry_zone_low=1.0,
            entry_zone_high=2.0,
            suggested_sl=0.9,
        )

    pb = SingleEntryPlaybook()
    engine = LifecycleEngine(
        router=PlaybookRouter(playbooks=[pb], default=pb),
        portfolio=PortfolioEngine(),
        lookback=20,
    )
    result = engine.run_symbol("VCB", df, signal_detector=detector)
    assert len(result["closed_positions"]) == 0
    assert any(e["status"] == "reject:outside_entry_zone" for e in result["audit_log"])


def test_portfolio_risk_cap_blocks_wide_stop():
    df = _make_reversal_df()
    pb = SingleEntryPlaybook()
    engine = LifecycleEngine(
        router=PlaybookRouter(playbooks=[pb], default=pb),
        portfolio=PortfolioEngine(PortfolioConstraints(max_total_risk_pct=0.01)),
        lookback=20,
    )
    result = engine.run_symbol(
        "VCB",
        df,
        signal_detector=_signal_at_bottom_factory(32),
    )
    assert len(result["closed_positions"]) == 0
    assert any("max_total_risk_pct" in e["status"] for e in result["audit_log"])


def test_settlement_is_tracked_per_buy_leg():
    pb = SingleEntryPlaybook()
    engine = LifecycleEngine(
        router=PlaybookRouter(playbooks=[pb], default=pb),
        execution_cfg=ExecutionConfig(settlement_days=2),
    )
    pos = Position(symbol="VCB", stage=PositionStage.FULL)
    pos.add_leg(Leg("2024-01-01", 100.0, 3.0, LegReason.PROBE, 95.0, bar_idx=5))
    pos.add_leg(Leg("2024-01-02", 105.0, 3.0, LegReason.ADD_STRENGTH, 100.0, bar_idx=8))

    assert engine._settled_nav_pct(pos, 6) == 0.0
    assert engine._settled_nav_pct(pos, 7) == 3.0
    assert engine._settled_nav_pct(pos, 9) == 3.0
    assert engine._settled_nav_pct(pos, 10) == 6.0


def test_regime_filters_trend_setup_in_downtrend():
    df = _make_reversal_df(n_down=60, n_bottom=0, n_up=0)
    snap = classify_regime(df)
    assert snap.regime == Regime.DOWNTREND
    assert not is_setup_allowed_for_regime("BREAKOUT", snap)
    assert is_setup_allowed_for_regime("DOUBLE_BOTTOM", snap)


def test_core_trend_playbook_can_hold_large_winner():
    df = _make_reversal_df(n_down=0, n_bottom=0, n_up=90)

    def detector(data, i):
        if i != 20:
            return None
        row = data.iloc[i]
        return Signal(
            symbol="VCB",
            date=str(row["date"])[:10],
            setup_type="CORE_TREND",
            confluence_score=84.0,
            entry_zone_low=float(row["close"]) * 0.95,
            entry_zone_high=float(row["close"]) * 1.05,
            suggested_sl=float(row["close"]) * 0.945,
            regime="UPTREND",
            playbook_hint="core_trend",
        )

    pb = CoreTrendPlaybook(core_pct=95.0)
    engine = LifecycleEngine(
        router=PlaybookRouter(playbooks=[pb], default=pb),
        portfolio=PortfolioEngine(
            PortfolioConstraints(
                max_total_exposure_pct=100.0,
                max_position_pct=100.0,
                max_total_risk_pct=6.0,
            )
        ),
        lookback=20,
    )
    result = engine.run_symbol("VCB", df, signal_detector=detector)
    closed = result["closed_positions"]
    assert len(closed) == 1
    pm = position_metrics(closed[0])
    assert pm["max_nav_pct"] == 95.0
    assert pm["realized_pnl_nav_pct"] > 30.0, pm


def test_sector_rotation_ranks_leading_sector():
    strong_a = _make_reversal_df(n_down=0, n_bottom=0, n_up=90)
    strong_b = _make_reversal_df(n_down=0, n_bottom=0, n_up=90)
    weak_a = _make_reversal_df(n_down=70, n_bottom=0, n_up=20)
    weak_b = _make_reversal_df(n_down=70, n_bottom=0, n_up=20)
    model = SectorRotationModel(
        {
            "STRONG": ["AAA", "AAB"],
            "WEAK": ["BBA", "BBB"],
        },
        {
            "AAA": strong_a,
            "AAB": strong_b,
            "BBA": weak_a,
            "BBB": weak_b,
        },
    )
    snap = model.snapshot("AAA", str(strong_a.iloc[-1]["date"])[:10])
    assert snap.sector == "STRONG"
    assert snap.rank == 1
    assert snap.allows_core


if __name__ == "__main__":
    test_scale_in_reversal_full_lifecycle()
    print("\n" + "=" * 70)
    test_single_entry_backward_compat()
    print("\n" + "=" * 70)
    test_portfolio_constraint_blocks_open()
    print("\n" + "=" * 70)
    test_open_position_keeps_original_playbook_when_default_differs()
    print("\n" + "=" * 70)
    test_buy_rejected_when_fill_outside_entry_zone()
    print("\n" + "=" * 70)
    test_portfolio_risk_cap_blocks_wide_stop()
    print("\n" + "=" * 70)
    test_settlement_is_tracked_per_buy_leg()
    print("\n" + "=" * 70)
    test_regime_filters_trend_setup_in_downtrend()
    print("\n" + "=" * 70)
    test_core_trend_playbook_can_hold_large_winner()
    print("\n" + "=" * 70)
    test_sector_rotation_ranks_leading_sector()
