"""Research cycle-first soft ranking boosts for the MVP strategy pool.

The experiment keeps MVP filters and risk rules unchanged. It only clones MVP
hypotheses in memory and adjusts ranking weights for cycle-first evidence:
sector leadership, smart-money no-sector, relative strength, and DS/CHDM shape.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from scripts.backtest_position_exit_agent import run_with_exit_agent
from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG
from multiagents_trading_assistant.edge_lab.strategy_sleeves import MVP_EDGE_STRATEGIES


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "cycle_first_soft_boost"


BOOST_VARIANTS: dict[str, list[dict[str, Any]]] = {
    "sector_smt_rs_boost_v1": [
        {"column": "sector_leadership_score", "ascending": False, "weight": 1.1},
        {"column": "smart_money_score_no_sector", "ascending": False, "weight": 0.9},
        {"column": "rs_percentile_20", "ascending": False, "weight": 0.8},
    ],
    "sector_smt_rs_ds_boost_v2": [
        {"column": "sector_leadership_score", "ascending": False, "weight": 1.4},
        {"column": "smart_money_score_no_sector", "ascending": False, "weight": 1.0},
        {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
        {"column": "DS20", "ascending": True, "weight": 0.5},
    ],
    "strong_leader_boost_v3": [
        {"column": "sector_leadership_score", "ascending": False, "weight": 1.6},
        {"column": "smart_money_score_no_sector", "ascending": False, "weight": 1.2},
        {"column": "rs_percentile_20", "ascending": False, "weight": 1.2},
        {"column": "CHDM50", "ascending": False, "weight": 0.4},
        {"column": "distribution_days_10", "ascending": True, "weight": 0.4},
    ],
}


def _clone_with_rank_boost(hypothesis: Hypothesis, variant: str, boosts: list[dict[str, Any]]) -> Hypothesis:
    rank = [dict(item) for item in hypothesis.rank]
    rank.extend(dict(item) for item in boosts)
    return replace(
        hypothesis,
        name=f"{hypothesis.name}__{variant}",
        description=f"{hypothesis.description} [cycle-first soft rank boost: {variant}]",
        tags=[*hypothesis.tags, "cycle_first_soft_boost", variant],
        rank=rank,
    )


def _load_mvp_hypotheses() -> list[Hypothesis]:
    by_name = {hyp.name: hyp for hyp in load_hypotheses(DEFAULT_CONFIG)}
    return [by_name[name] for name in MVP_EDGE_STRATEGIES if name in by_name]


def _row(
    *,
    case: str,
    variant: str,
    max_positions: int,
    max_candidates_per_day: int,
    result: dict[str, Any],
    start: str,
    end: str,
) -> dict[str, Any]:
    metrics = result["metrics"]
    trades = int(metrics.get("number_of_trades", 0))
    weeks = max(1.0, (pd.Timestamp(end) - pd.Timestamp(start)).days / 7.0)
    return {
        "case": case,
        "variant": variant,
        "max_positions": max_positions,
        "max_candidates_per_day": max_candidates_per_day,
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
    }


def _strategy_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "edge_strategy_name" not in trades.columns:
        return pd.DataFrame()
    out = (
        trades.groupby("edge_strategy_name")
        .agg(
            trades=("symbol", "count"),
            sum_pnl_pct=("pnl_pct", "sum"),
            avg_pnl_pct=("pnl_pct", "mean"),
            win_rate_pct=("pnl_pct", lambda s: float((s > 0).mean() * 100.0)),
            pnl=("pnl", "sum"),
        )
        .reset_index()
        .sort_values("pnl", ascending=False)
    )
    out["sum_pnl_pct"] *= 100.0
    out["avg_pnl_pct"] *= 100.0
    return out


def run_period(
    *,
    universe: str,
    start: str,
    end: str,
    label: str,
    max_positions_values: list[int],
    max_candidates_values: list[int],
) -> Path:
    out_dir = OUT_ROOT / f"{label}_{universe}_{start}_{end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, start, end)

    print(f"[cycle-boost] universe={universe} symbols={len(clean_symbols)} period={start}->{end}")
    print("[cycle-boost] building feature table once...")
    features, _ = build_feature_table(symbols, start, end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)

    base_hypotheses = _load_mvp_hypotheses()
    cases: list[tuple[str, str, list[Hypothesis]]] = [("MVP_BASELINE", "baseline", base_hypotheses)]
    for variant, boosts in BOOST_VARIANTS.items():
        cases.append(
            (
                f"MVP_{variant}",
                variant,
                [_clone_with_rank_boost(hyp, variant, boosts) for hyp in base_hypotheses],
            )
        )

    rows: list[dict[str, Any]] = []
    for case_name, variant, hypotheses in cases:
        print(f"[cycle-boost] cache {case_name}")
        signal_cache = combo_harness.build_signal_cache_for_combo(
            features,
            hypotheses,
            case_name,
            clean_symbols,
        )
        combo_harness.install_edge_cache(signal_cache)
        for max_positions in max_positions_values:
            for max_candidates_per_day in max_candidates_values:
                cfg = LivePipelineBacktestConfig(
                    start_date=start,
                    end_date=end,
                    max_positions=max_positions,
                    max_candidates_per_day=max_candidates_per_day,
                    edge_strategy_name=case_name,
                )
                print(f"[cycle-boost] run {case_name} p{max_positions}/c{max_candidates_per_day}")
                result = run_with_exit_agent(universe_data, vnindex, features, cfg, enabled=True)
                row = _row(
                    case=case_name,
                    variant=variant,
                    max_positions=max_positions,
                    max_candidates_per_day=max_candidates_per_day,
                    result=result,
                    start=start,
                    end=end,
                )
                rows.append(row)
                stem = f"{case_name}_p{max_positions}_c{max_candidates_per_day}"
                trades = pd.DataFrame(result["trades"])
                trades.to_csv(out_dir / f"{stem}_trades.csv", index=False)
                if not result["equity_frame"].empty:
                    result["equity_frame"].to_csv(out_dir / f"{stem}_equity.csv")
                breakdown = _strategy_breakdown(trades)
                if not breakdown.empty:
                    breakdown.to_csv(out_dir / f"{stem}_strategy_breakdown.csv", index=False)
                pd.DataFrame(rows).sort_values("total_return_pct", ascending=False).to_csv(out_dir / "summary.csv", index=False)
                print(
                    f"  -> ret={row['total_return_pct']:+.2f}% "
                    f"sharpe={row['sharpe_ratio']:.3f} "
                    f"mdd={row['max_drawdown_pct']:.2f}% "
                    f"trades={row['number_of_trades']}"
                )

    summary = pd.DataFrame(rows).sort_values(["total_return_pct", "sharpe_ratio"], ascending=[False, False])
    summary.to_csv(out_dir / "summary.csv", index=False)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Research MVP cycle-first soft ranking boosts")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-19")
    parser.add_argument("--label", default="cycle_boost_2025now")
    parser.add_argument("--max-positions", default="5")
    parser.add_argument("--max-candidates-per-day", default="10")
    args = parser.parse_args()

    max_positions = [int(item.strip()) for item in args.max_positions.split(",") if item.strip()]
    max_candidates = [int(item.strip()) for item in args.max_candidates_per_day.split(",") if item.strip()]
    out_dir = run_period(
        universe=args.universe,
        start=args.start,
        end=args.end,
        label=args.label,
        max_positions_values=max_positions,
        max_candidates_values=max_candidates,
    )
    print(f"[cycle-boost] saved {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
