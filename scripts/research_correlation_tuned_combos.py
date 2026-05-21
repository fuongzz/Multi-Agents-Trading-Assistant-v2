"""Correlation-aware combo generation and backtest.

This research runner does not randomly generate strategy combos. It first
estimates signal overlap between hypotheses, keeps low-overlap/diverse combos,
then tunes portfolio constraints on a train split and validates top candidates
on a later split.
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from scripts.backtest_position_exit_agent import run_with_exit_agent
from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, evaluate_filters, load_hypotheses, rank_candidates
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "correlation_tuned_combo_research"


DEFAULT_POOL = [
    "leader_pullback_market_regime_v3",
    "leader_pullback_market_healthy_v2",
    "breakout_55_smt_v1",
    "money_cycle_reset_smt_confirm",
    "smart_money_strong_market_healthy",
    "accumulation_breakout_smt_v1",
    "breakout_after_accumulation_v3",
    "compression_breakout_smt_v1",
    "mean_reversion_uptrend_ma50_v1",
    "reclaim_ma20_quality_v1",
    "cmf_expansion_smt_v1",
    "sector_leader_pullback_v1",
    "benchmark_leader_breakout_v1",
    "dry_retest_breakout_smt_v1",
    "nr7_value_breakout_smt_v1",
]


def _combo_name(combo: tuple[str, ...]) -> str:
    digest = hashlib.md5("|".join(combo).encode("utf-8")).hexdigest()[:8]
    return "CORR_" + "__".join(_short_name(item) for item in combo) + f"__{digest}"


def _short_name(name: str) -> str:
    parts = name.replace("_v1", "").replace("_v2", "").replace("_v3", "").split("_")
    return "".join(part[:4] for part in parts[:3])


def _family(hyp: Hypothesis) -> str:
    tags = set(hyp.tags)
    if "leader_pullback" in tags or "pullback" in tags:
        return "pullback"
    if "breakout" in tags or "55day" in tags:
        return "breakout"
    if "mean_reversion" in tags:
        return "mean_reversion"
    if "rotation" in tags:
        return "rotation"
    if "money_cycle" in tags or "smart_money_trace" in tags:
        return "smart_money"
    return sorted(tags)[0] if tags else "unknown"


def _top_signal_keys(features: pd.DataFrame, hypothesis: Hypothesis, top_per_day: int) -> set[str]:
    keys: set[str] = set()
    for date, daily in features.groupby("date", sort=True):
        latest = daily.reset_index(drop=True)
        mask = evaluate_filters(latest, hypothesis)
        ranked = rank_candidates(latest[mask], hypothesis)
        for row in ranked.head(top_per_day).to_dict("records"):
            keys.add(f"{pd.Timestamp(date).date().isoformat()}|{row['symbol']}")
    return keys


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def _build_signal_profiles(
    *,
    features: pd.DataFrame,
    hypotheses: dict[str, Hypothesis],
    pool: list[str],
    top_per_day: int,
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    profiles = {name: _top_signal_keys(features, hypotheses[name], top_per_day) for name in pool}
    rows = []
    for a, b in combinations(pool, 2):
        rows.append({"strategy_a": a, "strategy_b": b, "jaccard": round(_jaccard(profiles[a], profiles[b]), 4)})
    return profiles, pd.DataFrame(rows).sort_values("jaccard")


def _combo_overlap(combo: tuple[str, ...], profiles: dict[str, set[str]]) -> tuple[float, int]:
    overlaps = [_jaccard(profiles[a], profiles[b]) for a, b in combinations(combo, 2)]
    union_count = len(set().union(*(profiles[item] for item in combo))) if combo else 0
    return (sum(overlaps) / len(overlaps) if overlaps else 0.0), union_count


def _generate_candidates(
    *,
    pool: list[str],
    hypotheses: dict[str, Hypothesis],
    profiles: dict[str, set[str]],
    combo_sizes: list[int],
    max_avg_overlap: float,
    min_union_signals: int,
    candidate_limit: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for size in combo_sizes:
        for combo in combinations(pool, size):
            avg_overlap, union_count = _combo_overlap(combo, profiles)
            if avg_overlap > max_avg_overlap or union_count < min_union_signals:
                continue
            families = {_family(hypotheses[name]) for name in combo}
            diversity = len(families)
            score = union_count * (1.0 - avg_overlap) + 25.0 * diversity
            rows.append(
                {
                    "combo": _combo_name(combo),
                    "strategies": " + ".join(combo),
                    "n_strategies": size,
                    "avg_signal_overlap": round(avg_overlap, 4),
                    "union_signal_count": union_count,
                    "family_count": diversity,
                    "families": ",".join(sorted(families)),
                    "pre_score": round(score, 3),
                }
            )
    return pd.DataFrame(rows).sort_values("pre_score", ascending=False).head(candidate_limit)


def _install_combo_cache(
    *,
    features: pd.DataFrame,
    hypotheses: dict[str, Hypothesis],
    combo: tuple[str, ...],
    clean_symbols: list[str],
) -> None:
    combo_harness.install_edge_cache(
        combo_harness.build_signal_cache_for_combo(
            features,
            [hypotheses[name] for name in combo],
            _combo_name(combo),
            clean_symbols,
        )
    )


def _run_combo(
    *,
    combo: tuple[str, ...],
    features: pd.DataFrame,
    hypotheses: dict[str, Hypothesis],
    clean_symbols: list[str],
    universe_data: dict[str, pd.DataFrame],
    vnindex: pd.DataFrame,
    start: str,
    end: str,
    max_positions: int,
    max_candidates_per_day: int,
    exit_agent: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _install_combo_cache(features=features, hypotheses=hypotheses, combo=combo, clean_symbols=clean_symbols)
    cfg = LivePipelineBacktestConfig(
        start_date=start,
        end_date=end,
        max_positions=max_positions,
        max_candidates_per_day=max_candidates_per_day,
        edge_strategy_name=_combo_name(combo),
    )
    result = run_with_exit_agent(universe_data, vnindex, features, cfg, enabled=exit_agent)
    metrics = result["metrics"]
    row = {
        "combo": _combo_name(combo),
        "strategies": " + ".join(combo),
        "max_positions": max_positions,
        "max_candidates_per_day": max_candidates_per_day,
        "exit_agent": exit_agent,
        "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
        "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
        "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
        "number_of_trades": int(metrics.get("number_of_trades", 0)),
        "fills": int(metrics.get("fills", 0)),
        "agent_raise_stop": int(metrics.get("agent_raise_stop", 0)),
    }
    return row, result


def _research_score(row: dict[str, Any]) -> float:
    ret = float(row.get("total_return_pct") or 0.0)
    sharpe = float(row.get("sharpe_ratio") or 0.0)
    mdd = abs(float(row.get("max_drawdown_pct") or 0.0))
    trades = int(row.get("number_of_trades") or 0)
    return ret + 8.0 * sharpe - max(0.0, mdd - 16.0) * 2.0 - max(0, 35 - trades) * 0.7


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlation-aware combo tuning research")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--train-start", default="2025-01-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--valid-start", default="2026-01-01")
    parser.add_argument("--valid-end", default="2026-05-16")
    parser.add_argument("--label", default="corr_tuned_2025_train_2026_valid")
    parser.add_argument("--pool", default="")
    parser.add_argument("--combo-sizes", default="2,3")
    parser.add_argument("--profile-top-per-day", type=int, default=5)
    parser.add_argument("--max-avg-overlap", type=float, default=0.38)
    parser.add_argument("--min-union-signals", type=int, default=45)
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument("--top-n-validate", type=int, default=5)
    parser.add_argument("--max-positions-grid", default="3,5,7")
    parser.add_argument("--max-candidates-grid", default="10,15")
    parser.add_argument("--include-exit-agent", action="store_true")
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    hypotheses = {hyp.name: hyp for hyp in load_hypotheses(DEFAULT_CONFIG)}
    pool = [item.strip() for item in args.pool.split(",") if item.strip()] if args.pool.strip() else DEFAULT_POOL
    missing = [name for name in pool if name not in hypotheses]
    if missing:
        raise KeyError(f"Missing hypotheses: {missing}")

    full_start = min(args.train_start, args.valid_start)
    full_end = max(args.train_end, args.valid_end)
    symbols = combo_harness.resolve_symbols(args.universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, full_start, full_end)

    print("[corr] building features once...")
    features, _ = build_feature_table(symbols, full_start, full_end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    train_features = features[
        (features["date"] >= pd.Timestamp(args.train_start))
        & (features["date"] <= pd.Timestamp(args.train_end))
    ].copy()

    print("[corr] building signal profiles...")
    profiles, overlap = _build_signal_profiles(
        features=train_features,
        hypotheses=hypotheses,
        pool=pool,
        top_per_day=args.profile_top_per_day,
    )
    overlap.to_csv(out_dir / "pairwise_signal_overlap.csv", index=False)

    candidates = _generate_candidates(
        pool=pool,
        hypotheses=hypotheses,
        profiles=profiles,
        combo_sizes=[int(item.strip()) for item in args.combo_sizes.split(",") if item.strip()],
        max_avg_overlap=args.max_avg_overlap,
        min_union_signals=args.min_union_signals,
        candidate_limit=args.candidate_limit,
    )
    candidates.to_csv(out_dir / "generated_candidates.csv", index=False)
    print(f"[corr] generated candidates={len(candidates)}")

    max_positions_grid = [int(item.strip()) for item in args.max_positions_grid.split(",") if item.strip()]
    max_candidates_grid = [int(item.strip()) for item in args.max_candidates_grid.split(",") if item.strip()]
    exit_agent_grid = [False, True] if args.include_exit_agent else [False]

    train_rows: list[dict[str, Any]] = []
    for _, cand in candidates.iterrows():
        combo = tuple(str(cand["strategies"]).split(" + "))
        for max_positions in max_positions_grid:
            for max_candidates in max_candidates_grid:
                for exit_agent in exit_agent_grid:
                    print(f"[corr] train {cand['combo']} p{max_positions} c{max_candidates} exit={exit_agent}")
                    row, _ = _run_combo(
                        combo=combo,
                        features=features,
                        hypotheses=hypotheses,
                        clean_symbols=clean_symbols,
                        universe_data=universe_data,
                        vnindex=vnindex,
                        start=args.train_start,
                        end=args.train_end,
                        max_positions=max_positions,
                        max_candidates_per_day=max_candidates,
                        exit_agent=exit_agent,
                    )
                    row["split"] = "train"
                    row["avg_signal_overlap"] = cand["avg_signal_overlap"]
                    row["union_signal_count"] = cand["union_signal_count"]
                    row["family_count"] = cand["family_count"]
                    row["research_score"] = round(_research_score(row), 3)
                    train_rows.append(row)
                    pd.DataFrame(train_rows).sort_values("research_score", ascending=False).to_csv(out_dir / "train_tuned_summary.csv", index=False)

    train_summary = pd.DataFrame(train_rows).sort_values("research_score", ascending=False)
    train_summary.to_csv(out_dir / "train_tuned_summary.csv", index=False)
    # Validate distinct strategy mixes first; otherwise one strong combo can
    # consume all validation slots with near-identical portfolio settings.
    selected = train_summary.drop_duplicates(subset=["combo"], keep="first").head(args.top_n_validate)
    selected.to_csv(out_dir / "selected_for_validation.csv", index=False)

    valid_rows: list[dict[str, Any]] = []
    for _, item in selected.iterrows():
        combo = tuple(str(item["strategies"]).split(" + "))
        exit_agent = str(item["exit_agent"]).lower() == "true"
        print(f"[corr] valid {item['combo']} p{item['max_positions']} c{item['max_candidates_per_day']} exit={exit_agent}")
        row, result = _run_combo(
            combo=combo,
            features=features,
            hypotheses=hypotheses,
            clean_symbols=clean_symbols,
            universe_data=universe_data,
            vnindex=vnindex,
            start=args.valid_start,
            end=args.valid_end,
            max_positions=int(item["max_positions"]),
            max_candidates_per_day=int(item["max_candidates_per_day"]),
            exit_agent=exit_agent,
        )
        row["split"] = "validation"
        row["train_total_return_pct"] = item["total_return_pct"]
        row["train_sharpe_ratio"] = item["sharpe_ratio"]
        row["train_max_drawdown_pct"] = item["max_drawdown_pct"]
        row["train_number_of_trades"] = item["number_of_trades"]
        row["avg_signal_overlap"] = item["avg_signal_overlap"]
        row["union_signal_count"] = item["union_signal_count"]
        row["family_count"] = item["family_count"]
        row["research_score"] = round(_research_score(row), 3)
        valid_rows.append(row)
        stem = f"{row['combo']}_p{row['max_positions']}_c{row['max_candidates_per_day']}_{'exit' if exit_agent else 'base'}"
        pd.DataFrame(result["trades"]).to_csv(out_dir / f"{stem}_valid_trades.csv", index=False)
        if not result["equity_frame"].empty:
            result["equity_frame"].to_csv(out_dir / f"{stem}_valid_equity.csv")
        pd.DataFrame(valid_rows).sort_values("research_score", ascending=False).to_csv(out_dir / "validation_summary.csv", index=False)

    validation = pd.DataFrame(valid_rows).sort_values("research_score", ascending=False)
    validation.to_csv(out_dir / "validation_summary.csv", index=False)
    print(f"[corr] saved {out_dir}")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
