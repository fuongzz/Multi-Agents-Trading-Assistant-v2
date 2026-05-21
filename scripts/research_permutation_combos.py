"""Wide permutation research for strategy combinations.

This is a research-only runner. It creates combinations from existing
hypotheses, evaluates them on a train period, then validates top candidates on a
separate period. It is deliberately conservative to reduce data mining.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from scripts.backtest_position_exit_agent import run_with_exit_agent
from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "permutation_combo_research"


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


def _combo_name(items: tuple[str, ...]) -> str:
    return "PERM_" + "__".join(_short_name(item) for item in items)


def _short_name(name: str) -> str:
    parts = name.replace("_v1", "").replace("_v2", "").replace("_v3", "").split("_")
    return "".join(part[:4] for part in parts[:3])


def _build_combo_cache(
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


def _run_one(
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
    _build_combo_cache(features=features, hypotheses=hypotheses, combo=combo, clean_symbols=clean_symbols)
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
        "n_strategies": len(combo),
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


def _score(row: dict[str, Any]) -> float:
    ret = float(row.get("total_return_pct") or 0.0)
    sharpe = float(row.get("sharpe_ratio") or 0.0)
    mdd = abs(float(row.get("max_drawdown_pct") or 0.0))
    trades = int(row.get("number_of_trades") or 0)
    trade_penalty = max(0, 25 - trades) * 0.8
    dd_penalty = max(0.0, mdd - 18.0) * 1.8
    return ret + 8.0 * sharpe - dd_penalty - trade_penalty


def _resolve_pool(args_pool: str, hypotheses: dict[str, Hypothesis]) -> list[str]:
    if args_pool.strip():
        pool = [item.strip() for item in args_pool.split(",") if item.strip()]
    else:
        pool = DEFAULT_POOL
    missing = [name for name in pool if name not in hypotheses]
    if missing:
        raise KeyError(f"Missing hypotheses in pool: {missing}")
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Wide permutation combo research")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--train-start", default="2025-01-01")
    parser.add_argument("--train-end", default="2025-12-31")
    parser.add_argument("--valid-start", default="2026-01-01")
    parser.add_argument("--valid-end", default="2026-05-16")
    parser.add_argument("--label", default="wide_perm_2025_train_2026_valid")
    parser.add_argument("--pool", default="", help="Comma-separated hypothesis names; default uses curated pool")
    parser.add_argument("--combo-sizes", default="2,3,4")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=15)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--exit-agent-train", action="store_true", help="Also evaluate exit-agent variants in train")
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    hypotheses = {hyp.name: hyp for hyp in load_hypotheses(DEFAULT_CONFIG)}
    pool = _resolve_pool(args.pool, hypotheses)
    combo_sizes = [int(item.strip()) for item in args.combo_sizes.split(",") if item.strip()]
    combos = [combo for size in combo_sizes for combo in combinations(pool, size)]
    print(f"[perm] pool={len(pool)} combos={len(combos)} sizes={combo_sizes}")

    symbols = combo_harness.resolve_symbols(args.universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})

    # Load full range once so validation has enough warmup for indicators.
    full_start = min(args.train_start, args.valid_start)
    full_end = max(args.train_end, args.valid_end)
    universe_data, vnindex = combo_harness.load_local_history(symbols, full_start, full_end)
    print("[perm] building features once...")
    features, _ = build_feature_table(symbols, full_start, full_end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)

    train_rows: list[dict[str, Any]] = []
    variants = [False, True] if args.exit_agent_train else [False]
    for idx, combo in enumerate(combos, start=1):
        for exit_agent in variants:
            print(f"[perm] train {idx}/{len(combos)} exit_agent={exit_agent} {_combo_name(combo)}")
            try:
                row, _ = _run_one(
                    combo=combo,
                    features=features,
                    hypotheses=hypotheses,
                    clean_symbols=clean_symbols,
                    universe_data=universe_data,
                    vnindex=vnindex,
                    start=args.train_start,
                    end=args.train_end,
                    max_positions=args.max_positions,
                    max_candidates_per_day=args.max_candidates_per_day,
                    exit_agent=exit_agent,
                )
                row["split"] = "train"
                row["research_score"] = round(_score(row), 3)
                train_rows.append(row)
                pd.DataFrame(train_rows).sort_values("research_score", ascending=False).to_csv(out_dir / "train_summary.csv", index=False)
            except Exception as exc:
                train_rows.append({"combo": _combo_name(combo), "strategies": " + ".join(combo), "split": "train", "error": str(exc)})

    train_summary = pd.DataFrame(train_rows).sort_values("research_score", ascending=False, na_position="last")
    train_summary.to_csv(out_dir / "train_summary.csv", index=False)
    top = train_summary[train_summary["error"].isna() if "error" in train_summary.columns else slice(None)].head(args.top_n)
    top.to_csv(out_dir / "selected_for_validation.csv", index=False)

    valid_rows: list[dict[str, Any]] = []
    for _, item in top.iterrows():
        combo = tuple(str(item["strategies"]).split(" + "))
        exit_agent = str(item.get("exit_agent", "False")).lower() == "true"
        print(f"[perm] valid {item['combo']} exit_agent={exit_agent}")
        row, result = _run_one(
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
        row["train_combo"] = item["combo"]
        row["train_total_return_pct"] = item["total_return_pct"]
        row["train_sharpe_ratio"] = item["sharpe_ratio"]
        row["train_max_drawdown_pct"] = item["max_drawdown_pct"]
        row["train_number_of_trades"] = item["number_of_trades"]
        row["research_score"] = round(_score(row), 3)
        valid_rows.append(row)
        stem = f"{row['combo']}_{'exit' if exit_agent else 'base'}"
        pd.DataFrame(result["trades"]).to_csv(out_dir / f"{stem}_valid_trades.csv", index=False)
        if not result["equity_frame"].empty:
            result["equity_frame"].to_csv(out_dir / f"{stem}_valid_equity.csv")
        pd.DataFrame(valid_rows).sort_values("research_score", ascending=False).to_csv(out_dir / "validation_summary.csv", index=False)

    validation = pd.DataFrame(valid_rows).sort_values("research_score", ascending=False)
    validation.to_csv(out_dir / "validation_summary.csv", index=False)
    print(f"[perm] saved {out_dir}")
    print(validation.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
