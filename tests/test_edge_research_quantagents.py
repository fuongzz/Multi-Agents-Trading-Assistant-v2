from __future__ import annotations

import json

import numpy as np
import pandas as pd

from multiagents_trading_assistant.quantagents_backtest.edge_research import (
    EdgeResearchConfig,
    build_edge_signal_table,
    load_research_strategies,
    run_edge_vn_quantagents,
)
from multiagents_trading_assistant.quantagents_backtest.run_local import run_local_quantagents
from multiagents_trading_assistant.quantagents_backtest.local_data import LocalUniverseConfig
from multiagents_trading_assistant.quantagents_backtest.vn_quantagents import VNQuantAgentsConfig


def _write_hypothesis_config(path):
    payload = {
        "hypotheses": [
            {
                "name": "trend_alpha",
                "description": "Simple trend alpha",
                "universe": "VN30",
                "tags": ["trend"],
                "filters": [
                    {"column": "above_ma20", "op": "==", "value": True},
                    {"column": "mkt_regime_score", "op": ">=", "value": 40},
                ],
                "rank": [{"column": "edge_score", "ascending": False, "weight": 1.0}],
                "risk": {"stop_loss": 0.07, "take_profit": 0.20, "max_holding_bars": 20},
            },
            {
                "name": "pullback_alpha",
                "description": "Simple pullback alpha",
                "universe": "VN30",
                "tags": ["mean_reversion"],
                "filters": [
                    {"column": "reclaim_ma20", "op": "==", "value": True},
                    {"column": "smart_money_score", "op": ">=", "value": 45},
                ],
                "rank": [{"column": "pullback_quality_score", "ascending": False, "weight": 1.0}],
                "risk": {"stop_loss": 0.06, "take_profit": 0.15, "max_holding_bars": 15},
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sample_local_ohlcv(symbol: str, start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    close = np.linspace(20.0, 35.0, periods) + np.sin(np.arange(periods) / 5.0)
    open_ = close * 0.999
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = np.full(periods, 500_000, dtype=np.int64)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "value": close * volume,
            "exchange": "HOSE",
            "industry": "Tech",
        }
    )


def _sample_index_df(start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    close = np.linspace(1000.0, 1100.0, periods)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "VNINDEX",
            "open": close * 0.999,
            "high": close * 1.003,
            "low": close * 0.997,
            "close": close,
            "volume": np.full(periods, 100_000_000, dtype=np.int64),
        }
    )


def _write_research_artifacts(root, symbols: list[str], start="2024-01-01", periods=180):
    money_cycle_dir = root / "data" / "research" / "money_cycle"
    smt_dir = root / "data" / "research" / "smart_money_trace"
    money_cycle_dir.mkdir(parents=True, exist_ok=True)
    smt_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start, periods=periods, freq="B")
    rows = []
    smt_rows = []
    for symbol in symbols:
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "CHDM03": 55 + (idx % 5),
                    "CHDM20": 52 + (idx % 7),
                    "CHDM50": 50 + (idx % 9),
                    "DS20": 0.3,
                    "DS50": 0.35,
                }
            )
            smt_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "smart_money_score": 60 + (idx % 3),
                    "smart_money_state": "HEALTHY",
                    "rs_percentile_20": 0.7,
                    "distribution_days_10": 0,
                    "accumulation_days_10": 2,
                    "rs_score": 65,
                    "value_flow_quality_score": 60,
                    "clv_score": 58,
                    "accumulation_distribution_score": 62,
                    "pullback_quality_score": 66,
                    "sector_leadership_score": 55,
                    "clv_ma5": 0.6,
                    "ma20": 25 + idx * 0.05,
                    "ma50": 24 + idx * 0.04,
                    "ma200": 22 + idx * 0.02,
                    "atr14": 1.1,
                    "cmf20": 0.1,
                    "mfi14": 55,
                    "ret_5d": 0.02,
                    "ret_20d": 0.05,
                    "value_ma20": 9_000_000,
                    "value_ratio_20": 1.2,
                    "volume_ma20": 450_000,
                    "volume_ratio_20": 1.1,
                    "pullback_depth_20": -0.03,
                }
            )
    pd.DataFrame(rows).to_parquet(money_cycle_dir / "chdm_by_symbol.parquet", index=False)
    pd.DataFrame(rows).to_parquet(money_cycle_dir / "ds_by_symbol.parquet", index=False)
    pd.DataFrame(
        {
            "date": dates,
            "CHDM10": 55,
            "CHDM50": 55,
            "CHDM200": 55,
            "DS10": 0.3,
            "DS50": 0.35,
            "DS200": 0.4,
        }
    ).to_parquet(money_cycle_dir / "money_cycle_market.parquet", index=False)
    pd.DataFrame(smt_rows).to_parquet(smt_dir / "smart_money_by_symbol.parquet", index=False)


def test_load_research_strategies_hypotheses_and_combos(tmp_path):
    config_path = tmp_path / "hyp.json"
    _write_hypothesis_config(config_path)
    hypotheses, by_name = load_research_strategies(
        EdgeResearchConfig(config_path=str(config_path), strategy_source="edge_hypotheses")
    )
    combos, _ = load_research_strategies(
        EdgeResearchConfig(
            config_path=str(config_path),
            strategy_source="edge_combos",
            combo_names=("MEANREV_TRIO",),
        )
    )
    assert len(hypotheses) == 2
    assert "trend_alpha" in by_name
    assert combos == []


def test_build_edge_signal_table(tmp_path):
    config_path = tmp_path / "hyp.json"
    _write_hypothesis_config(config_path)
    strategies, by_name = load_research_strategies(
        EdgeResearchConfig(config_path=str(config_path), strategy_source="edge_hypotheses")
    )
    features = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01"), "symbol": "AAA", "above_ma20": True, "mkt_regime_score": 50, "edge_score": 0.9, "reclaim_ma20": False, "smart_money_score": 50, "pullback_quality_score": 0.2},
            {"date": pd.Timestamp("2024-01-01"), "symbol": "BBB", "above_ma20": True, "mkt_regime_score": 55, "edge_score": 0.8, "reclaim_ma20": True, "smart_money_score": 60, "pullback_quality_score": 0.7},
        ]
    )
    table = build_edge_signal_table(features, strategies[0], by_name, top_candidates_per_day=5)
    assert not table.empty
    assert set(table["symbol"]) == {"AAA", "BBB"}


def test_run_local_quantagents_edge_hypotheses(tmp_path, monkeypatch):
    config_path = tmp_path / "hyp.json"
    _write_hypothesis_config(config_path)
    ohlcv = pd.concat(
        [
            _sample_local_ohlcv("AAA", "2024-01-01", 180),
            _sample_local_ohlcv("BBB", "2024-01-01", 180),
        ],
        ignore_index=True,
    )
    index_df = _sample_index_df("2024-01-01", 180)
    ohlcv_path = tmp_path / "ohlcv.parquet"
    index_path = tmp_path / "index.parquet"
    ohlcv.to_parquet(ohlcv_path, index=False)
    index_df.to_parquet(index_path, index=False)
    _write_research_artifacts(tmp_path, ["AAA", "BBB"])
    monkeypatch.chdir(tmp_path)

    result = run_local_quantagents(
        universe_config=LocalUniverseConfig(
            start="2024-01-01",
            end="2024-12-31",
            min_history_bars=120,
            min_median_daily_value=1_000_000.0,
        ),
        qa_config=VNQuantAgentsConfig(
            start="2024-01-01",
            end="2024-12-31",
            top_k=2,
            train_bars=80,
            test_bars=40,
            step_bars=40,
        ),
        ohlcv_path=ohlcv_path,
        index_path=index_path,
        explicit_symbols=["AAA", "BBB"],
        strategy_source="edge_hypotheses",
        research_config=EdgeResearchConfig(
            config_path=str(config_path),
            strategy_source="edge_hypotheses",
            top_candidates_per_day=3,
        ),
    )

    assert result["metadata"]["strategy_source"] == "edge_hypotheses"
    assert not result["strategy_memory"].empty
    assert not result["portfolio_summary"].empty
    assert not result["research_summary"].empty
