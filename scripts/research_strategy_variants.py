"""Research-only strategy variants for the bias-fixed combo harness.

The script does not modify the production hypothesis config. It clones selected
baseline hypotheses in memory, adds targeted filters/ranking changes, and runs
the patched live-pipeline backtest harness.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "strategy_variant_research"


def _with_suffix(hyp: Hypothesis, suffix: str, *, description: str, extra_filters: list[dict[str, Any]], extra_rank: list[dict[str, Any]] | None = None) -> Hypothesis:
    return replace(
        hyp,
        name=f"{hyp.name}{suffix}",
        description=f"{hyp.description} Research variant: {description}",
        filters=[*hyp.filters, *extra_filters],
        rank=[*(extra_rank or []), *hyp.rank],
        tags=[*hyp.tags, "research_variant"],
    )


def _risk_tuned(hyp: Hypothesis, suffix: str, *, description: str, risk_updates: dict[str, Any]) -> Hypothesis:
    risk = dict(hyp.risk)
    risk.update(risk_updates)
    return replace(
        hyp,
        name=f"{hyp.name}{suffix}",
        description=f"{hyp.description} Research variant: {description}",
        risk=risk,
        tags=[*hyp.tags, "research_variant"],
    )


def build_research_hypotheses(base: dict[str, Hypothesis]) -> dict[str, Hypothesis]:
    """Return cloned hypotheses for targeted research ideas."""

    sector_filters = [
        {"column": "sector_leadership_score", "op": ">=", "value": 62},
        {"column": "sector_breadth_ma20", "op": ">=", "value": 0.55},
        {"column": "sector_value_ratio_20", "op": ">=", "value": 1.00},
    ]
    sector_rank = [
        {"column": "sector_leadership_score", "ascending": False, "weight": 0.9},
        {"column": "sector_breadth_ma20", "ascending": False, "weight": 0.5},
    ]

    bulltrap_filters = [
        {"column": "close_location", "op": ">=", "value": 0.55},
        {"column": "upper_wick_pct", "op": "<=", "value": 0.06},
        {"column": "volume_ratio_20", "op": ">=", "value": 1.05},
        {"column": "sector_leadership_score", "op": ">=", "value": 55},
    ]
    bulltrap_rank = [
        {"column": "close_location", "ascending": False, "weight": 0.6},
        {"column": "sector_leadership_score", "ascending": False, "weight": 0.5},
    ]

    strict_regime_filters = [
        {"column": "mkt_regime_state", "op": "==", "value": "RISK_ON"},
        {"column": "mkt_regime_score", "op": ">=", "value": 62},
        {"column": "mkt_DS20", "op": "<=", "value": 0.38},
    ]

    kalman_filters = [
        {"column": "kalman_adaptive_trend_5d", "op": ">=", "value": -0.005},
        {"column": "kalman_confidence", "op": ">=", "value": 0.35},
        {"column": "kalman_shock_score", "op": "<=", "value": 1.25},
    ]

    out: dict[str, Hypothesis] = {}
    for name in [
        "leader_pullback_market_regime_v3",
        "leader_pullback_market_healthy_v2",
        "breakout_55_smt_v1",
    ]:
        hyp = base[name]
        out[hyp.name] = hyp
        out[f"{name}__sector"] = _with_suffix(
            hyp,
            "__sector",
            description="require sector leadership and sector breadth confirmation",
            extra_filters=sector_filters,
            extra_rank=sector_rank,
        )
        out[f"{name}__strict_regime"] = _with_suffix(
            hyp,
            "__strict_regime",
            description="only trade in stronger market regimes",
            extra_filters=strict_regime_filters,
        )
        out[f"{name}__kalman"] = _with_suffix(
            hyp,
            "__kalman",
            description="require adaptive Kalman trend/confidence and avoid shock bars",
            extra_filters=kalman_filters,
        )

    out["breakout_55_smt_v1__bulltrap_guard"] = _with_suffix(
        base["breakout_55_smt_v1"],
        "__bulltrap_guard",
        description="avoid weak breakout candles with poor close location or large upper wick",
        extra_filters=bulltrap_filters,
        extra_rank=bulltrap_rank,
    )
    out["breakout_55_smt_v1__sector_bulltrap"] = _with_suffix(
        out["breakout_55_smt_v1__sector"],
        "__bulltrap",
        description="combine sector confirmation and bull-trap guard",
        extra_filters=bulltrap_filters,
        extra_rank=bulltrap_rank,
    )

    out["leader_pullback_market_regime_v3__faster_protect"] = _risk_tuned(
        base["leader_pullback_market_regime_v3"],
        "__faster_protect",
        description="same entry, tighter profit protection and shorter max hold",
        risk_updates={
            "take_profit": 0.22,
            "max_holding_bars": 45,
            "trailing_atr_mult": 2.5,
            "trailing_profit_activation": 0.08,
        },
    )
    return out


def research_combos() -> list[dict[str, Any]]:
    return [
        {
            "name": "BASE_PURE_TOP3_LOCKED",
            "strategies": [
                "leader_pullback_market_regime_v3",
                "leader_pullback_market_healthy_v2",
                "breakout_55_smt_v1",
            ],
        },
        {
            "name": "SECTOR_TOP3",
            "strategies": [
                "leader_pullback_market_regime_v3__sector",
                "leader_pullback_market_healthy_v2__sector",
                "breakout_55_smt_v1__sector",
            ],
        },
        {
            "name": "SECTOR_PULLBACK_BASE_BREAKOUT",
            "strategies": [
                "leader_pullback_market_regime_v3__sector",
                "leader_pullback_market_healthy_v2__sector",
                "breakout_55_smt_v1",
            ],
        },
        {
            "name": "PULLBACK_BASE_BREAKOUT_BULLTRAP",
            "strategies": [
                "leader_pullback_market_regime_v3",
                "leader_pullback_market_healthy_v2",
                "breakout_55_smt_v1__bulltrap_guard",
            ],
        },
        {
            "name": "SECTOR_BULLTRAP_TOP3",
            "strategies": [
                "leader_pullback_market_regime_v3__sector",
                "leader_pullback_market_healthy_v2__sector",
                "breakout_55_smt_v1__sector_bulltrap",
            ],
        },
        {
            "name": "STRICT_REGIME_TOP3",
            "strategies": [
                "leader_pullback_market_regime_v3__strict_regime",
                "leader_pullback_market_healthy_v2__strict_regime",
                "breakout_55_smt_v1__strict_regime",
            ],
        },
        {
            "name": "KALMAN_TOP3",
            "strategies": [
                "leader_pullback_market_regime_v3__kalman",
                "leader_pullback_market_healthy_v2__kalman",
                "breakout_55_smt_v1__kalman",
            ],
        },
        {
            "name": "FAST_PROTECT_PULLBACK_BASE_BREAKOUT",
            "strategies": [
                "leader_pullback_market_regime_v3__faster_protect",
                "leader_pullback_market_healthy_v2",
                "breakout_55_smt_v1",
            ],
        },
    ]


def run_combo(
    *,
    combo: dict[str, Any],
    hypotheses: dict[str, Hypothesis],
    features: pd.DataFrame,
    symbols: list[str],
    universe_data: dict[str, pd.DataFrame],
    vnindex: pd.DataFrame,
    start: str,
    end: str,
    max_positions: int,
    max_candidates_per_day: int,
) -> dict[str, Any]:
    clean_symbols = sorted({s.upper().strip() for s in symbols})
    hyps = [hypotheses[name] for name in combo["strategies"]]
    signal_cache = combo_harness.build_signal_cache_for_combo(features, hyps, combo["name"], clean_symbols)
    combo_harness.install_edge_cache(signal_cache)
    cfg = LivePipelineBacktestConfig(
        start_date=start,
        end_date=end,
        max_positions=max_positions,
        max_candidates_per_day=max_candidates_per_day,
        edge_strategy_name=combo["name"],
    )
    result = combo_harness.run_one(universe_data, vnindex, cfg)
    metrics = result["metrics"]
    row = {
        "combo": combo["name"],
        "max_positions": max_positions,
        "max_candidates_per_day": max_candidates_per_day,
        "strategies": " + ".join(combo["strategies"]),
        "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
        "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
        "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
        "number_of_trades": int(metrics.get("number_of_trades", 0)),
        "signal_days": int(metrics.get("signal_days", 0)),
        "fills": int(metrics.get("fills", 0)),
    }
    return {"row": row, "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run research-only strategy variants")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--label", default="top3_variants_2025now")
    parser.add_argument("--max-positions", default="3,4,5", help="Comma-separated max position values")
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.label}_{args.universe}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(args.universe)
    universe_data, vnindex = combo_harness.load_local_history(symbols, args.start, args.end)
    print(f"[research] universe={args.universe} symbols={len(symbols)}")
    print("[research] building features once...")
    t0 = time.time()
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    print(f"[research] features={len(features):,} elapsed={time.time() - t0:.1f}s")

    base = {item.name: item for item in load_hypotheses(DEFAULT_CONFIG)}
    hypotheses = build_research_hypotheses(base)
    combos = research_combos()
    max_position_values = [int(item.strip()) for item in args.max_positions.split(",") if item.strip()]

    rows: list[dict[str, Any]] = []
    for max_positions in max_position_values:
        for idx, combo in enumerate(combos, start=1):
            print(f"[research] p{max_positions} [{idx}/{len(combos)}] {combo['name']}")
            t_combo = time.time()
            try:
                out = run_combo(
                    combo=combo,
                    hypotheses=hypotheses,
                    features=features,
                    symbols=symbols,
                    universe_data=universe_data,
                    vnindex=vnindex,
                    start=args.start,
                    end=args.end,
                    max_positions=max_positions,
                    max_candidates_per_day=args.max_candidates_per_day,
                )
                row = out["row"]
                row["elapsed_sec"] = round(time.time() - t_combo, 1)
                rows.append(row)
                stem = f"p{max_positions}_{combo['name']}"
                pd.DataFrame(out["result"]["trades"]).to_csv(out_dir / f"{stem}_trades.csv", index=False)
                if not out["result"]["equity_frame"].empty:
                    out["result"]["equity_frame"].to_csv(out_dir / f"{stem}_equity.csv")
                print(
                    f"  -> ret={row['total_return_pct']:+.2f}% sharpe={row['sharpe_ratio']:.2f} "
                    f"mdd={row['max_drawdown_pct']:.2f}% trades={row['number_of_trades']} "
                    f"({row['elapsed_sec']}s)"
                )
            except Exception as exc:
                rows.append(
                    {
                        "combo": combo["name"],
                        "max_positions": max_positions,
                        "max_candidates_per_day": args.max_candidates_per_day,
                        "strategies": " + ".join(combo["strategies"]),
                        "error": str(exc),
                    }
                )
                print(f"  !! failed: {exc}")
            pd.DataFrame(rows).sort_values("total_return_pct", ascending=False, na_position="last").to_csv(
                out_dir / "summary.csv",
                index=False,
            )

    summary = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False, na_position="last")
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n[research] saved {out_dir / 'summary.csv'}")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
