"""Research oil/gas thematic rotation strategies against the MVP shared pool."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.backtest_combos_unbiased as combo_harness
from scripts.backtest_position_exit_agent import run_with_exit_agent
from scripts.research_shared_pool_strategy_mix import GLOBAL_CONFIG, SECTOR_CONFIG
from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, evaluate_filters, load_hypotheses, rank_candidates
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG
from multiagents_trading_assistant.edge_lab.strategy_sleeves import MVP_EDGE_STRATEGIES
from multiagents_trading_assistant.research.sector_rotation.enrich_features import enrich_features_with_rotation

CONFIG_DIR = ROOT / "multiagents_trading_assistant" / "edge_lab" / "configs"
OIL_CONFIG = CONFIG_DIR / "oil_gas_rotation_hypotheses.json"
OUT_ROOT = ROOT / "backtest_results" / "oil_gas_rotation_research"
OIL_GAS_SYMBOLS = ("BSR", "OIL", "PLX", "PVB", "PVC", "PVD", "PVS", "GAS", "PVT")


def _load_hypothesis_map() -> dict[str, Hypothesis]:
    out: dict[str, Hypothesis] = {}
    for config in [Path(DEFAULT_CONFIG), GLOBAL_CONFIG, SECTOR_CONFIG, OIL_CONFIG]:
        for hyp in load_hypotheses(config):
            out[hyp.name] = hyp
    return out


def _row(
    *,
    case_name: str,
    max_positions: int,
    result: dict[str, Any],
    start: str,
    end: str,
    strategies: list[str],
) -> dict[str, Any]:
    metrics = result["metrics"]
    trades = int(metrics.get("number_of_trades", 0))
    weeks = max(1.0, (pd.Timestamp(end) - pd.Timestamp(start)).days / 7.0)
    return {
        "case": case_name,
        "max_positions": max_positions,
        "n_strategies": len(strategies),
        "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100.0, 2),
        "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100.0, 2),
        "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100.0, 2),
        "number_of_trades": trades,
        "trades_per_week": round(trades / weeks, 2),
        "signals": int(metrics.get("signals", 0)),
        "fills": int(metrics.get("fills", 0)),
        "agent_reviews": int(metrics.get("agent_reviews", 0)),
        "agent_hold_runner": int(metrics.get("agent_hold_runner", 0)),
        "agent_raise_stop": int(metrics.get("agent_raise_stop", 0)),
        "strategies": " + ".join(strategies),
    }


def _export_recent_oil_passes(
    *,
    features: pd.DataFrame,
    hypotheses: list[Hypothesis],
    start: str,
    end: str,
    out_dir: Path,
) -> None:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    rows: list[dict[str, Any]] = []
    recent = features[
        features["symbol"].isin(OIL_GAS_SYMBOLS)
        & (features["date"] >= start_ts)
        & (features["date"] <= end_ts)
    ].copy()
    for date, daily in recent.groupby("date", sort=True):
        latest = daily.reset_index(drop=True)
        for hyp in hypotheses:
            mask = evaluate_filters(latest, hyp)
            ranked = rank_candidates(latest[mask], hyp)
            for rank_idx, item in enumerate(ranked.to_dict("records"), start=1):
                rows.append(
                    {
                        "date": pd.Timestamp(date).date().isoformat(),
                        "symbol": item.get("symbol"),
                        "strategy": hyp.name,
                        "rank": rank_idx,
                        "edge_rank_score": round(float(item.get("_edge_rank") or 0.0), 4),
                        "edge_score": item.get("edge_score"),
                        "smart_money_score": item.get("smart_money_score"),
                        "smart_money_delta": item.get("smart_money_score_delta"),
                        "rs20": item.get("rs_percentile_20"),
                        "value_ratio_20": item.get("value_ratio_20"),
                        "CHDM50": item.get("CHDM50"),
                        "DS20": item.get("DS20"),
                        "mkt_regime": item.get("mkt_regime_state"),
                        "sector_l1": item.get("sector_l1"),
                        "sector_rs_rank_20d": item.get("sector_rs_rank_20d"),
                    }
                )
    pd.DataFrame(rows).to_csv(out_dir / "recent_oil_gas_hypothesis_passes.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest oil/gas thematic rotation research hypotheses")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-14")
    parser.add_argument("--label", default="oil_gas_2025now")
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.label}_{args.universe}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(args.universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, args.start, args.end)

    print(f"[oil-gas] universe={args.universe} symbols={len(clean_symbols)}")
    print("[oil-gas] building features...")
    feature_result = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = feature_result[0] if isinstance(feature_result, tuple) else feature_result
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    features = enrich_features_with_rotation(features, root=ROOT)

    hypotheses = _load_hypothesis_map()
    oil_strategies = ["oil_gas_smart_money_recovery_v1", "oil_gas_breakout_rotation_v1"]
    oil_quality_strategies = ["oil_gas_quality_recovery_v2", "oil_gas_leader_breakout_v2"]
    oil_all_strategies = oil_strategies + oil_quality_strategies
    mvp_strategies = list(MVP_EDGE_STRATEGIES)
    cases = [
        {"name": "OIL_GAS_ONLY_p2", "strategies": oil_strategies, "max_positions": 2},
        {"name": "OIL_GAS_ONLY_p5", "strategies": oil_strategies, "max_positions": 5},
        {"name": "OIL_GAS_QUALITY_ONLY_p2", "strategies": oil_quality_strategies, "max_positions": 2},
        {"name": "OIL_GAS_QUALITY_ONLY_p5", "strategies": oil_quality_strategies, "max_positions": 5},
        {"name": "OIL_GAS_ALL_RESEARCH_p3", "strategies": oil_all_strategies, "max_positions": 3},
        {"name": "MVP_BASELINE_p5", "strategies": mvp_strategies, "max_positions": 5},
        {"name": "MVP_PLUS_OIL_SHARED_p5", "strategies": mvp_strategies + oil_strategies, "max_positions": 5},
        {"name": "MVP_PLUS_OIL_SHARED_p6", "strategies": mvp_strategies + oil_strategies, "max_positions": 6},
        {"name": "MVP_PLUS_OIL_QUALITY_SHARED_p5", "strategies": mvp_strategies + oil_quality_strategies, "max_positions": 5},
        {"name": "MVP_PLUS_OIL_QUALITY_SHARED_p6", "strategies": mvp_strategies + oil_quality_strategies, "max_positions": 6},
    ]

    _export_recent_oil_passes(
        features=features,
        hypotheses=[hypotheses[name] for name in oil_all_strategies],
        start="2026-05-04",
        end=args.end,
        out_dir=out_dir,
    )

    rows: list[dict[str, Any]] = []
    for case in cases:
        missing = [name for name in case["strategies"] if name not in hypotheses]
        if missing:
            raise KeyError(f"{case['name']} missing hypotheses: {missing}")

        print(f"[oil-gas] case={case['name']} strategies={len(case['strategies'])}")
        signal_cache = combo_harness.build_signal_cache_for_combo(
            features,
            [hypotheses[name] for name in case["strategies"]],
            case["name"],
            clean_symbols,
        )
        combo_harness.install_edge_cache(signal_cache)

        cfg = LivePipelineBacktestConfig(
            start_date=args.start,
            end_date=args.end,
            max_positions=int(case["max_positions"]),
            max_candidates_per_day=args.max_candidates_per_day,
            edge_strategy_name=case["name"],
        )
        result = run_with_exit_agent(universe_data, vnindex, features, cfg, enabled=True)
        row = _row(
            case_name=case["name"],
            max_positions=int(case["max_positions"]),
            result=result,
            start=args.start,
            end=args.end,
            strategies=list(case["strategies"]),
        )
        rows.append(row)

        stem = case["name"]
        pd.DataFrame(result["trades"]).to_csv(out_dir / f"{stem}_trades.csv", index=False)
        if not result["equity_frame"].empty:
            result["equity_frame"].to_csv(out_dir / f"{stem}_equity.csv")
        pd.DataFrame(rows).sort_values("total_return_pct", ascending=False).to_csv(out_dir / "summary.csv", index=False)
        print(
            f"  -> ret={row['total_return_pct']:+.2f}% sharpe={row['sharpe_ratio']:.3f} "
            f"mdd={row['max_drawdown_pct']:.2f}% wr={row['win_rate_pct']:.2f}% trades={row['number_of_trades']}"
        )

    summary = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"[oil-gas] saved {out_dir / 'summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
