"""Tests for Phase 2 EvidenceBuilder."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multiagents_trading_assistant.agentic.evidence_builder import (
    EvidenceBuilder,
    _LIVE_ONLY_FIELDS,
)
from multiagents_trading_assistant.agentic.schemas import EvidencePacket, StrategySignal


@dataclass
class _Market:
    reference_trend: str = "UPTREND"
    vni_change_pct: float = 0.5
    trend: str = "UPTREND"


@dataclass
class _Candidate:
    symbol: str
    indicators: dict = field(default_factory=dict)
    priority_score: float = 88.0
    setup_type: str = "EDGE_BREAKOUT"
    market_context: _Market = field(default_factory=_Market)
    reasons: list[str] = field(default_factory=list)


def _make_candidate(symbol: str = "VCB", **indicators_kw) -> _Candidate:
    indicators = {
        "current_price": 50.0,
        "rsi": 60.0,
        "edge_strategy_analysis": {
            "passed": True,
            "strategy_name": "breakout_after_accumulation_v3",
            "setup_type": "EDGE_BREAKOUT",
            "edge_score": 72.5,
            "risk": {"stop_loss": 0.05},
        },
        "money_flow_analysis": {"regime": "BREAKOUT_FLOW", "action_bias": "BUY_CANDIDATE"},
    }
    indicators.update(indicators_kw)
    return _Candidate(symbol=symbol, indicators=indicators)


def _seed_parquets(root: Path) -> None:
    """Create mock smart_money / chdm / ds parquets with rows spanning a year."""
    smt_path = root / "data/research/smart_money_trace/smart_money_by_symbol.parquet"
    chdm_path = root / "data/research/money_cycle/chdm_by_symbol.parquet"
    ds_path = root / "data/research/money_cycle/ds_by_symbol.parquet"
    mkt_path = root / "data/research/money_cycle/money_cycle_market.parquet"
    for p in (smt_path, chdm_path, ds_path, mkt_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2024-01-01", "2024-12-31", freq="B")
    smt = pd.DataFrame({
        "date": dates,
        "symbol": ["VCB"] * len(dates),
        "smart_money_score": np.linspace(10, 90, len(dates)),
        "smart_money_state": ["HOT_BUT_STRONG"] * len(dates),
        "rs_percentile_20": np.linspace(20, 95, len(dates)),
        "sector_leadership_score": np.linspace(0, 100, len(dates)),
    })
    smt.to_parquet(smt_path, index=False)

    chdm = pd.DataFrame({
        "date": dates,
        "symbol": ["VCB"] * len(dates),
        "CHDM03": np.linspace(-5, 5, len(dates)),
        "CHDM20": np.linspace(-3, 3, len(dates)),
    })
    chdm.to_parquet(chdm_path, index=False)

    ds = pd.DataFrame({
        "date": dates,
        "symbol": ["VCB"] * len(dates),
        "DS20": np.linspace(0, 100, len(dates)),
    })
    ds.to_parquet(ds_path, index=False)

    mkt = pd.DataFrame({
        "date": dates,
        "mkt_chdm03": np.linspace(-1, 1, len(dates)),
        "mkt_ds20": np.linspace(0, 100, len(dates)),
        "mkt_regime": ["UPTREND"] * len(dates),
    })
    mkt.to_parquet(mkt_path, index=False)


def test_build_minimal(tmp_path):
    builder = EvidenceBuilder("2024-06-15", backtest_mode=True, root=tmp_path, write_cache=False)
    packet = builder.build(_make_candidate())
    assert isinstance(packet, EvidencePacket)
    assert packet.symbol == "VCB"
    assert packet.as_of_date == "2024-06-15"
    assert packet.edge["passed"] is True
    assert packet.money_flow["classification"]["regime"] == "BREAKOUT_FLOW"


def test_parquet_anti_lookahead(tmp_path):
    _seed_parquets(tmp_path)
    as_of = "2024-06-15"
    builder = EvidenceBuilder(as_of, backtest_mode=True, root=tmp_path, write_cache=False)
    packet = builder.build(_make_candidate())

    sector_row_date = packet.sector.get("as_of_row_date")
    assert sector_row_date is not None
    assert sector_row_date <= as_of, f"sector pulled future row: {sector_row_date}"

    smt = packet.technical["smart_money"]
    assert smt["as_of_row_date"] <= as_of
    mc = packet.technical["money_cycle"]
    assert mc["chdm_row_date"] <= as_of
    assert mc["ds_row_date"] <= as_of

    mc_market = packet.market["money_cycle_market"]
    assert mc_market["date"] <= as_of


def test_backtest_mode_skips_live_fields(tmp_path):
    builder = EvidenceBuilder("2024-06-15", backtest_mode=True, root=tmp_path, write_cache=False)
    packet = builder.build(_make_candidate())
    assert packet.news == {}
    assert packet.portfolio == {}
    assert packet.fundamental == {}
    assert "global_macro" not in packet.market
    assert "foreign_flow" not in packet.money_flow
    skipped = packet.data_quality["skipped_in_backtest"]
    for field_name in _LIVE_ONLY_FIELDS:
        assert field_name in skipped


def test_fetcher_error_resilience(tmp_path, monkeypatch):
    """When a live fetcher raises, packet still builds with empty field."""
    from multiagents_trading_assistant import fetcher as _fetcher
    from multiagents_trading_assistant import news_fetcher as _news_fetcher
    from multiagents_trading_assistant import database as _database

    def _boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(_fetcher, "get_fundamentals", _boom)
    monkeypatch.setattr(_fetcher, "get_foreign_flow", _boom)
    monkeypatch.setattr(_fetcher, "get_global_macro", _boom)
    monkeypatch.setattr(_news_fetcher, "get_stock_news", _boom)
    monkeypatch.setattr(_database, "get_position", _boom)

    builder = EvidenceBuilder("2024-06-15", backtest_mode=False, root=tmp_path, write_cache=False)
    packet = builder.build(_make_candidate())
    assert packet.fundamental == {}
    assert packet.news == {}
    assert packet.portfolio == {}
    # Builder catches exceptions inside _build_* (returns {}) so they don't
    # bubble up to the ThreadPoolExecutor wrapper. Pipeline does not crash.


def test_json_serializable_no_nan_no_timestamp(tmp_path):
    candidate = _make_candidate(
        timestamp_field=pd.Timestamp("2024-06-14"),
        nan_field=float("nan"),
        np_nan_field=np.nan,
    )
    builder = EvidenceBuilder("2024-06-15", backtest_mode=True, root=tmp_path, write_cache=False)
    packet = builder.build(candidate)
    blob = packet.model_dump_json(by_alias=False)
    parsed = json.loads(blob)
    assert parsed["symbol"] == "VCB"
    # No raw NaN in serialized form: cleaned to None
    assert "NaN" not in blob


def test_backward_compat_old_field_names(tmp_path):
    builder = EvidenceBuilder("2024-06-15", backtest_mode=True, root=tmp_path, write_cache=False)
    packet = builder.build(_make_candidate())
    assert packet.edge_strategy["passed"] is True
    assert packet.market_context == packet.market
    assert packet.indicators == packet.technical
    assert packet.fundamentals == packet.fundamental


def test_cache_write(tmp_path):
    cache_root = tmp_path / "cache"
    builder = EvidenceBuilder(
        "2024-06-15",
        backtest_mode=True,
        root=tmp_path,
        cache_root=cache_root,
        write_cache=True,
    )
    builder.build(_make_candidate())
    out = cache_root / "2024-06-15" / "VCB.json"
    assert out.exists()
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["symbol"] == "VCB"
    assert parsed["as_of_date"] == "2024-06-15"
    # canonical (new) field names are present in serialized form
    assert "market" in parsed
    assert "technical" in parsed
    assert "edge" in parsed


def test_signal_param_attaches_to_data_quality(tmp_path):
    signal = StrategySignal(
        symbol="VCB",
        signal_date="2024-06-15",
        as_of_date="2024-06-15",
        strategy_name="breakout_after_accumulation_v3",
    )
    builder = EvidenceBuilder("2024-06-15", backtest_mode=True, root=tmp_path, write_cache=False)
    packet = builder.build(_make_candidate(), signal=signal)
    assert packet.data_quality["signal_strategy"] == "breakout_after_accumulation_v3"
    assert packet.data_quality["signal_source"] == "core3"
