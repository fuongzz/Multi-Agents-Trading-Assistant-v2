"""Backtest mixed hypothesis combos from multiple config files.

Useful for researching new external/global strategy hypotheses together with
the existing VN money-cycle / smart-money library.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "mixed_hypothesis_combos"
GLOBAL_CONFIG = ROOT / "multiagents_trading_assistant" / "edge_lab" / "configs" / "global_market_hypotheses.json"


COMBOS: list[dict[str, Any]] = [
    {
        "name": "GLOBAL_TOP3_FABER_MINERVINI_WEINSTEIN",
        "strategies": [
            "faber_gtaa_trend_filter_v1",
            "minervini_trend_template_v1",
            "weinstein_stage2_breakout_v1",
        ],
    },
    {
        "name": "HIGH_WIN_ROTATION_PLUS_FABER",
        "strategies": [
            "rotation_weekly_smt_money_cycle_v1",
            "rotation_monthly_smt_money_cycle_v1",
            "accumulation_breakout_smt_v1",
            "faber_gtaa_trend_filter_v1",
        ],
    },
    {
        "name": "FABER_PLUS_CONNORS_ACCUM",
        "strategies": [
            "faber_gtaa_trend_filter_v1",
            "connors_rsi2_mean_reversion_v1",
            "accumulation_breakout_smt_v1",
        ],
    },
    {
        "name": "FABER_PLUS_PURE_TOP3",
        "strategies": [
            "faber_gtaa_trend_filter_v1",
            "leader_pullback_market_regime_v3",
            "leader_pullback_market_healthy_v2",
            "breakout_55_smt_v1",
        ],
    },
]


def _load_hypothesis_map(configs: list[Path]) -> dict[str, Hypothesis]:
    out: dict[str, Hypothesis] = {}
    for config in configs:
        for hyp in load_hypotheses(config):
            out[hyp.name] = hyp
    return out


def _row(result: dict[str, Any], combo: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    metrics = result["metrics"]
    trades = int(metrics.get("number_of_trades", 0))
    weeks = max(1.0, (pd.Timestamp(args.end) - pd.Timestamp(args.start)).days / 7.0)
    return {
        "combo": combo["name"],
        "strategies": " + ".join(combo["strategies"]),
        "start": args.start,
        "end": args.end,
        "max_positions": args.max_positions,
        "max_candidates_per_day": args.max_candidates_per_day,
        "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
        "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
        "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
        "number_of_trades": trades,
        "trades_per_week": round(trades / weeks, 2),
        "fills": int(metrics.get("fills", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest mixed hypothesis combos")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--label", default="mixed_global_research")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=15)
    parser.add_argument("--combos", default="", help="Comma-separated combo subset")
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.label}_{args.universe}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    hypotheses = _load_hypothesis_map([Path(DEFAULT_CONFIG), GLOBAL_CONFIG])
    wanted = {item.strip() for item in args.combos.split(",") if item.strip()}
    combos = [combo for combo in COMBOS if not wanted or combo["name"] in wanted]

    symbols = combo_harness.resolve_symbols(args.universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, args.start, args.end)

    print("[mixed] building features once...")
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for combo in combos:
        missing = [name for name in combo["strategies"] if name not in hypotheses]
        if missing:
            raise KeyError(f"{combo['name']} missing hypotheses: {missing}")
        print(f"[mixed] {combo['name']}")
        cache = combo_harness.build_signal_cache_for_combo(
            features,
            [hypotheses[name] for name in combo["strategies"]],
            combo["name"],
            clean_symbols,
        )
        combo_harness.install_edge_cache(cache)
        cfg = LivePipelineBacktestConfig(
            start_date=args.start,
            end_date=args.end,
            max_positions=args.max_positions,
            max_candidates_per_day=args.max_candidates_per_day,
            edge_strategy_name=combo["name"],
        )
        result = combo_harness.run_one(universe_data, vnindex, cfg)
        row = _row(result, combo, args)
        rows.append(row)
        stem = combo["name"]
        pd.DataFrame(result["trades"]).to_csv(out_dir / f"{stem}_trades.csv", index=False)
        if not result["equity_frame"].empty:
            result["equity_frame"].to_csv(out_dir / f"{stem}_equity.csv")
        pd.DataFrame(rows).sort_values("total_return_pct", ascending=False).to_csv(out_dir / "summary.csv", index=False)
        print(
            f"  -> ret={row['total_return_pct']:+.2f}% sharpe={row['sharpe_ratio']:.2f} "
            f"mdd={row['max_drawdown_pct']:.2f}% wr={row['win_rate_pct']:.2f}% "
            f"trades={row['number_of_trades']} tpw={row['trades_per_week']:.2f}"
        )

    summary = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"[mixed] saved {out_dir / 'summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
