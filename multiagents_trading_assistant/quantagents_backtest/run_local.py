"""Research-only runner for local VN QuantAgents MVP backtests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd

from multiagents_trading_assistant.quantagents_backtest.edge_research import (
    EdgeResearchConfig,
    run_edge_vn_quantagents,
)
from multiagents_trading_assistant.quantagents_backtest import (
    VNQuantAgentsConfig,
    run_vn_quantagents,
    save_vn_quantagents_result,
)
from multiagents_trading_assistant.quantagents_backtest.local_data import (
    DEFAULT_INDEX_PATH,
    DEFAULT_OHLCV_PATH,
    LocalUniverseConfig,
    load_local_index,
    load_local_universe,
)


def run_local_quantagents(
    *,
    universe_config: LocalUniverseConfig | None = None,
    qa_config: VNQuantAgentsConfig | None = None,
    ohlcv_path: str | Path = DEFAULT_OHLCV_PATH,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    constituents_path: str | Path | None = None,
    explicit_symbols: list[str] | None = None,
    strategy_source: str = "generic",
    research_config: EdgeResearchConfig | None = None,
) -> dict:
    """Run the QuantAgents-style research workflow on local parquet data."""

    uc = universe_config or LocalUniverseConfig()
    qc = qa_config or VNQuantAgentsConfig(start=uc.start, end=uc.end)
    universe_data, coverage, snapshot = load_local_universe(
        uc,
        ohlcv_path=ohlcv_path,
        constituents_path=constituents_path,
        explicit_symbols=explicit_symbols,
    )
    if not universe_data:
        raise ValueError("Universe is empty after local history/liquidity filters")

    market_index = load_local_index(index_path=index_path, start=uc.start, end=uc.end)
    if strategy_source == "generic":
        result = run_vn_quantagents(
            universe_data=universe_data,
            market_index=market_index,
            config=qc,
        )
    else:
        rc = research_config
        if rc is None:
            raise ValueError("research_config is required when strategy_source is not `generic`")
        _stage_edge_research_inputs(Path("."), ohlcv_path, index_path)
        result = run_edge_vn_quantagents(
            universe=sorted(universe_data),
            market_index=market_index.reset_index().rename(columns={"index": "date"}) if isinstance(market_index.index, pd.DatetimeIndex) else market_index,
            config=qc,
            research_config=rc,
            root=".",
        )
    result["data_coverage"] = coverage
    result["metadata"] = {
        "mode": "research_test_only",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe_name": uc.universe_name,
        "snapshot_date": snapshot.date.isoformat(),
        "snapshot_source": snapshot.source,
        "is_historical_snapshot": snapshot.is_historical,
        "explicit_symbols": explicit_symbols or [],
        "input_ohlcv_path": str(Path(ohlcv_path).resolve()),
        "input_index_path": str(Path(index_path).resolve()),
        "constituents_path": str(Path(constituents_path).resolve()) if constituents_path else None,
        "universe_requested": len(snapshot.symbols),
        "universe_included": len(universe_data),
        "filters": {
            "min_history_bars": uc.min_history_bars,
            "min_median_daily_value": uc.min_median_daily_value,
            "min_last_close": uc.min_last_close,
            "allowed_exchanges": list(uc.allowed_exchanges),
        },
        "strategy_source": strategy_source,
        "research_config": _json_ready(asdict(rc)) if strategy_source != "generic" and (rc := research_config) is not None else None,
        "quantagents_config": _json_ready(asdict(qc)),
    }
    return result


def save_local_quantagents_result(result: dict, output_dir: str | Path) -> Path:
    """Save core CSV outputs plus research metadata."""

    out = save_vn_quantagents_result(result, output_dir)
    if "data_coverage" in result and isinstance(result["data_coverage"], pd.DataFrame):
        result["data_coverage"].to_csv(out / "data_coverage.csv", index=False)
    if "metadata" in result:
        (out / "metadata.json").write_text(
            json.dumps(_json_ready(result["metadata"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    strategy_memory = result.get("strategy_memory")
    if isinstance(strategy_memory, pd.DataFrame) and not strategy_memory.empty:
        family_summary = (
            strategy_memory.groupby("family", dropna=False)
            .agg(
                strategies=("strategy_id", "count"),
                avg_qa_score=("qa_score", "mean"),
                avg_return=("avg_total_return", "mean"),
                avg_sharpe=("avg_sharpe_ratio", "mean"),
            )
            .reset_index()
            .sort_values("avg_qa_score", ascending=False)
        )
        family_summary.to_csv(out / "family_summary.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local QuantAgents MVP backtest on parquet data")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-05-03")
    parser.add_argument("--universe", default="VN30")
    parser.add_argument("--symbols", nargs="*", help="Optional explicit symbols instead of index snapshot")
    parser.add_argument("--ohlcv-path", default=str(DEFAULT_OHLCV_PATH))
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--constituents-path", help="Historical constituents CSV for anti-survivorship snapshots")
    parser.add_argument("--min-history-bars", type=int, default=252)
    parser.add_argument("--min-median-daily-value", type=float, default=50_000_000.0)
    parser.add_argument("--min-last-close", type=float, default=5.0)
    parser.add_argument("--n-strategies", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--train-bars", type=int, default=504)
    parser.add_argument("--test-bars", type=int, default=126)
    parser.add_argument("--step-bars", type=int, default=63)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--strategy-source",
        default="generic",
        choices=["generic", "edge_hypotheses", "edge_combos"],
        help="Use generic indicator pool or project research hypotheses/combos",
    )
    parser.add_argument(
        "--research-config-path",
        default="multiagents_trading_assistant/edge_lab/configs/vn30_money_smt_hypotheses.json",
    )
    parser.add_argument(
        "--combo-names",
        default="",
        help="Comma-separated subset of default research combos when strategy-source=edge_combos",
    )
    parser.add_argument("--top-candidates-per-day", type=int, default=5)
    parser.add_argument("--output-dir", default="backtest_results/quantagents_local")
    args = parser.parse_args()

    universe_config = LocalUniverseConfig(
        start=args.start,
        end=args.end,
        universe_name=args.universe,
        min_history_bars=args.min_history_bars,
        min_median_daily_value=args.min_median_daily_value,
        min_last_close=args.min_last_close,
    )
    qa_config = VNQuantAgentsConfig(
        start=args.start,
        end=args.end,
        n_strategies=args.n_strategies,
        seed=args.seed,
        top_k=args.top_k,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
    )
    research_config = None
    if args.strategy_source != "generic":
        combo_names = tuple(item.strip() for item in args.combo_names.split(",") if item.strip())
        research_config = EdgeResearchConfig(
            config_path=args.research_config_path,
            strategy_source=args.strategy_source,
            combo_names=combo_names,
            top_candidates_per_day=args.top_candidates_per_day,
        )
    result = run_local_quantagents(
        universe_config=universe_config,
        qa_config=qa_config,
        ohlcv_path=args.ohlcv_path,
        index_path=args.index_path,
        constituents_path=args.constituents_path,
        explicit_symbols=args.symbols,
        strategy_source=args.strategy_source,
        research_config=research_config,
    )
    output_dir = save_local_quantagents_result(result, args.output_dir)
    summary = result["portfolio_summary"]
    print(summary.to_string(index=False))
    print(f"\nUniverse included: {result['metadata']['universe_included']}/{result['metadata']['universe_requested']}")
    print(f"Outputs written to: {output_dir.resolve()}")


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _stage_edge_research_inputs(root: Path, ohlcv_path: str | Path, index_path: str | Path) -> None:
    data_dir = root / "multiagents_trading_assistant" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ohlcv_target = data_dir / "ohlcv_master.parquet"
    index_target = data_dir / "index_master.parquet"
    ohlcv_source = Path(ohlcv_path).resolve()
    index_source = Path(index_path).resolve()
    if ohlcv_source != ohlcv_target.resolve():
        shutil.copyfile(ohlcv_source, ohlcv_target)
    if index_source != index_target.resolve():
        shutil.copyfile(index_source, index_target)


if __name__ == "__main__":
    main()
