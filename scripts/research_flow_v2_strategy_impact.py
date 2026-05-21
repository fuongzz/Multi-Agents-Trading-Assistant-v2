"""Test how decomposed Flow V2 scores affect existing MVP strategies.

The script keeps strategy configs unchanged. It swaps feature columns in memory
before building the signal cache so we can measure whether the root flow score
improvement helps or hurts portfolio behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from scripts.backtest_position_exit_agent import run_with_exit_agent
from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig
from multiagents_trading_assistant.edge_lab.features import _edge_score, build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG
from multiagents_trading_assistant.edge_lab.strategy_sleeves import MVP_EDGE_STRATEGIES


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "flow_v2_strategy_impact"


VARIANT_MAP: dict[str, dict[str, str]] = {
    "baseline": {},
    "legacy_dist_veto_80": {},
    "legacy_dist_veto_70": {},
    "legacy_flow_quality_boost": {},
    "smt_to_flow_quality_v2": {"smart_money_score": "flow_quality_v2_score"},
    "smt_to_flow_pure": {"smart_money_score": "flow_pure_score"},
    "smt_to_sponsorship": {"smart_money_score": "flow_sponsorship_score"},
    "smt_to_absorption": {"smart_money_score": "flow_absorption_score"},
    "value_to_flow_quality_v2": {"value_flow_quality_score": "flow_quality_v2_score"},
    "smt_and_value_to_flow_quality_v2": {
        "smart_money_score": "flow_quality_v2_score",
        "value_flow_quality_score": "flow_quality_v2_score",
    },
}

POST_SIGNAL_VARIANTS = {"legacy_dist_veto_80", "legacy_dist_veto_70", "legacy_flow_quality_boost"}


def _load_mvp_hypotheses() -> list[Hypothesis]:
    by_name = {hyp.name: hyp for hyp in load_hypotheses(DEFAULT_CONFIG)}
    return [by_name[name] for name in MVP_EDGE_STRATEGIES if name in by_name]


def _apply_variant(features: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = features.copy()
    for target, source in VARIANT_MAP[variant].items():
        if source not in out.columns:
            raise ValueError(f"Missing source column for {variant}: {source}")
        out[target] = out[source]
    if variant != "baseline":
        out["smart_money_score_delta"] = out.groupby("symbol", sort=False)["smart_money_score"].diff()
        out["edge_score"] = _edge_score(out, smart_money_col="smart_money_score")
        out["edge_score_no_sector"] = _edge_score(out, smart_money_col="smart_money_score_no_sector")
    return out


def _apply_post_signal_variant(signal_cache: dict, features: pd.DataFrame, variant: str) -> dict:
    if variant not in POST_SIGNAL_VARIANTS:
        return signal_cache
    cols = ["date", "symbol", "distribution_pressure_score", "flow_quality_v2_score"]
    lookup = features[cols].set_index(["date", "symbol"]).to_dict("index")
    threshold = 80.0 if variant == "legacy_dist_veto_80" else 70.0
    out = {}
    for date, day in signal_cache.items():
        patched = {}
        for symbol, payload in day.items():
            current = dict(payload)
            row = lookup.get((pd.Timestamp(date).normalize(), symbol))
            if current.get("passed") and row is not None:
                dist = float(row.get("distribution_pressure_score") or 0.0)
                flow = float(row.get("flow_quality_v2_score") or 50.0)
                if variant in {"legacy_dist_veto_80", "legacy_dist_veto_70"} and dist >= threshold:
                    current["passed"] = False
                    current["filters_failed"] = [*(current.get("filters_failed") or []), f"{variant}:distribution_pressure"]
                elif variant == "legacy_flow_quality_boost":
                    current["edge_rank_score"] = round(
                        float(current.get("edge_rank_score") or 0.0) + max(-0.25, min(0.25, (flow - 50.0) / 100.0)),
                        4,
                    )
            patched[symbol] = current
        out[date] = patched
    return out


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


def _row(
    *,
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
        "agent_raise_stop": int(metrics.get("agent_raise_stop", 0)),
    }


def run_period(
    *,
    universe: str,
    start: str,
    end: str,
    label: str,
    variants: list[str],
    max_positions_values: list[int],
    max_candidates_values: list[int],
) -> Path:
    out_dir = OUT_ROOT / f"{label}_{universe}_{start}_{end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, start, end)

    print(f"[flow-v2] universe={universe} symbols={len(clean_symbols)} period={start}->{end}")
    print("[flow-v2] building feature table once...")
    base_features, _ = build_feature_table(symbols, start, end, root=ROOT)
    base_features = base_features.sort_values(["date", "symbol"]).reset_index(drop=True)
    hypotheses = _load_mvp_hypotheses()

    rows: list[dict[str, Any]] = []
    for variant in variants:
        features = _apply_variant(base_features, variant)
        signal_cache = combo_harness.build_signal_cache_for_combo(
            features,
            hypotheses,
            f"MVP_FLOW_V2_{variant}",
            clean_symbols,
        )
        signal_cache = _apply_post_signal_variant(signal_cache, features, variant)
        combo_harness.install_edge_cache(signal_cache)
        for max_positions in max_positions_values:
            for max_candidates_per_day in max_candidates_values:
                cfg = LivePipelineBacktestConfig(
                    start_date=start,
                    end_date=end,
                    max_positions=max_positions,
                    max_candidates_per_day=max_candidates_per_day,
                    edge_strategy_name=f"MVP_FLOW_V2_{variant}",
                )
                print(f"[flow-v2] run {variant} p{max_positions}/c{max_candidates_per_day}")
                result = run_with_exit_agent(universe_data, vnindex, features, cfg, enabled=True)
                row = _row(
                    variant=variant,
                    max_positions=max_positions,
                    max_candidates_per_day=max_candidates_per_day,
                    result=result,
                    start=start,
                    end=end,
                )
                rows.append(row)
                stem = f"{variant}_p{max_positions}_c{max_candidates_per_day}"
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

    pd.DataFrame(rows).sort_values(["total_return_pct", "sharpe_ratio"], ascending=[False, False]).to_csv(
        out_dir / "summary.csv",
        index=False,
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Research Flow V2 impact on MVP strategies")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-19")
    parser.add_argument("--label", default="flow_v2_impact_2025now")
    parser.add_argument("--variants", default=",".join(VARIANT_MAP))
    parser.add_argument("--max-positions", default="4,5")
    parser.add_argument("--max-candidates-per-day", default="10")
    args = parser.parse_args()

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(VARIANT_MAP))
    if unknown:
        raise ValueError(f"Unsupported variants: {unknown}")
    max_positions = [int(item.strip()) for item in args.max_positions.split(",") if item.strip()]
    max_candidates = [int(item.strip()) for item in args.max_candidates_per_day.split(",") if item.strip()]
    out_dir = run_period(
        universe=args.universe,
        start=args.start,
        end=args.end,
        label=args.label,
        variants=variants,
        max_positions_values=max_positions,
        max_candidates_values=max_candidates,
    )
    print(f"[flow-v2] saved {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
