"""Research combo-set backtests with PositionExitAgentV2.

This tests whether combining strategy families increases sample size without
destroying risk-adjusted performance. It is research-only and does not change
the locked demo baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from scripts.backtest_position_exit_agent import run_with_exit_agent
from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "exit_agent_combo_set_research"


COMBO_SETS: list[dict[str, Any]] = [
    {
        "name": "PURE_TOP3",
        "strategies": [
            "leader_pullback_market_regime_v3",
            "leader_pullback_market_healthy_v2",
            "breakout_55_smt_v1",
        ],
    },
    {
        "name": "PURE_TOP3_PLUS_SMT",
        "strategies": [
            "leader_pullback_market_regime_v3",
            "leader_pullback_market_healthy_v2",
            "breakout_55_smt_v1",
            "money_cycle_reset_smt_confirm",
            "smart_money_strong_market_healthy",
            "accumulation_breakout_smt_v1",
        ],
    },
    {
        "name": "PURE_TOP3_PLUS_LIVE_CORE3",
        "strategies": [
            "leader_pullback_market_regime_v3",
            "leader_pullback_market_healthy_v2",
            "breakout_55_smt_v1",
            "breakout_after_accumulation_v3",
            "compression_breakout_smt_v1",
            "mean_reversion_uptrend_ma50_v1",
        ],
    },
    {
        "name": "PURE_TOP3_PLUS_SMT_LIVE",
        "strategies": [
            "leader_pullback_market_regime_v3",
            "leader_pullback_market_healthy_v2",
            "breakout_55_smt_v1",
            "money_cycle_reset_smt_confirm",
            "smart_money_strong_market_healthy",
            "accumulation_breakout_smt_v1",
            "breakout_after_accumulation_v3",
            "compression_breakout_smt_v1",
            "mean_reversion_uptrend_ma50_v1",
        ],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest combo sets with PositionExitAgentV2")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--label", default="combo_sets_2025now")
    parser.add_argument("--max-positions", default="3,5,7")
    parser.add_argument("--max-candidates-per-day", type=int, default=15)
    parser.add_argument("--combo-sets", default="", help="Comma-separated subset of combo set names")
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.label}_{args.universe}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(args.universe)
    universe_data, vnindex = combo_harness.load_local_history(symbols, args.start, args.end)
    print(f"[combo-set] universe={args.universe} symbols={len(symbols)}")
    print("[combo-set] building features once...")
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)

    hypotheses = {hyp.name: hyp for hyp in load_hypotheses(DEFAULT_CONFIG)}
    wanted = {item.strip() for item in args.combo_sets.split(",") if item.strip()}
    combo_sets = [item for item in COMBO_SETS if not wanted or item["name"] in wanted]
    max_positions_values = [int(item.strip()) for item in args.max_positions.split(",") if item.strip()]
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})

    rows: list[dict[str, Any]] = []
    for combo_set in combo_sets:
        missing = [name for name in combo_set["strategies"] if name not in hypotheses]
        if missing:
            raise KeyError(f"{combo_set['name']} missing hypotheses: {missing}")
        print(f"[combo-set] cache {combo_set['name']} ({len(combo_set['strategies'])} strategies)")
        signal_cache = combo_harness.build_signal_cache_for_combo(
            features,
            [hypotheses[name] for name in combo_set["strategies"]],
            combo_set["name"],
            clean_symbols,
        )
        combo_harness.install_edge_cache(signal_cache)

        for max_positions in max_positions_values:
            for enabled in [False, True]:
                variant = "EXIT_AGENT" if enabled else "BASELINE"
                cfg = LivePipelineBacktestConfig(
                    start_date=args.start,
                    end_date=args.end,
                    max_positions=max_positions,
                    max_candidates_per_day=args.max_candidates_per_day,
                    edge_strategy_name=combo_set["name"],
                )
                print(f"[combo-set] {combo_set['name']} {variant} p{max_positions}")
                result = run_with_exit_agent(universe_data, vnindex, features, cfg, enabled=enabled)
                metrics = result["metrics"]
                row = {
                    "combo_set": combo_set["name"],
                    "variant": variant,
                    "n_strategies": len(combo_set["strategies"]),
                    "max_positions": max_positions,
                    "max_candidates_per_day": args.max_candidates_per_day,
                    "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
                    "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
                    "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
                    "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
                    "number_of_trades": int(metrics.get("number_of_trades", 0)),
                    "agent_reviews": int(metrics.get("agent_reviews", 0)),
                    "agent_tp_next_open": int(metrics.get("agent_tp_next_open", 0)),
                    "agent_hold_runner": int(metrics.get("agent_hold_runner", 0)),
                    "agent_raise_stop": int(metrics.get("agent_raise_stop", 0)),
                    "fills": int(metrics.get("fills", 0)),
                    "strategies": " + ".join(combo_set["strategies"]),
                }
                rows.append(row)
                stem = f"{combo_set['name']}_p{max_positions}_{variant.lower()}"
                pd.DataFrame(result["trades"]).to_csv(out_dir / f"{stem}_trades.csv", index=False)
                if not result["equity_frame"].empty:
                    result["equity_frame"].to_csv(out_dir / f"{stem}_equity.csv")
                pd.DataFrame(rows).sort_values("total_return_pct", ascending=False).to_csv(out_dir / "summary.csv", index=False)
                print(
                    f"  -> ret={row['total_return_pct']:+.2f}% sharpe={row['sharpe_ratio']:.2f} "
                    f"mdd={row['max_drawdown_pct']:.2f}% trades={row['number_of_trades']}"
                )

    summary = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"[combo-set] saved {out_dir / 'summary.csv'}")
    print(summary.head(24).to_string(index=False))


if __name__ == "__main__":
    main()
