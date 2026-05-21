"""Research shared-pool strategy mixes across default, global, and cycle configs.

All strategies in a combo compete for the same portfolio slots. This is the
production-fair way to test whether adding more strategies improves the current
MVP without implicitly increasing exposure.
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
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG
from multiagents_trading_assistant.research.sector_rotation.enrich_features import (
    enrich_features_with_rotation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "multiagents_trading_assistant" / "edge_lab" / "configs"
GLOBAL_CONFIG = CONFIG_DIR / "global_market_hypotheses.json"
SECTOR_CONFIG = CONFIG_DIR / "sector_rotation_hypotheses.json"
OIL_GAS_CONFIG = CONFIG_DIR / "oil_gas_rotation_hypotheses.json"
THEME_FLOW_CONFIG = CONFIG_DIR / "theme_flow_hypotheses.json"
OUT_ROOT = ROOT / "backtest_results" / "shared_pool_strategy_mix"


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
    {
        "name": "PURE_TOP3_PLUS_FABER",
        "strategies": [
            "leader_pullback_market_regime_v3",
            "leader_pullback_market_healthy_v2",
            "breakout_55_smt_v1",
            "faber_gtaa_trend_filter_v1",
        ],
    },
    {
        "name": "PURE_TOP3_PLUS_GLOBAL_TOP3",
        "strategies": [
            "leader_pullback_market_regime_v3",
            "leader_pullback_market_healthy_v2",
            "breakout_55_smt_v1",
            "faber_gtaa_trend_filter_v1",
            "minervini_trend_template_v1",
            "weinstein_stage2_breakout_v1",
        ],
    },
    {
        "name": "PURE_TOP3_PLUS_MARKET_CYCLE_DEF",
        "strategies": [
            "leader_pullback_market_regime_v3",
            "leader_pullback_market_healthy_v2",
            "breakout_55_smt_v1",
            "breakout_market_cycle_defensive_v1",
        ],
    },
    {
        "name": "PURE_TOP3_PLUS_SMT_FABER",
        "strategies": [
            "leader_pullback_market_regime_v3",
            "leader_pullback_market_healthy_v2",
            "breakout_55_smt_v1",
            "money_cycle_reset_smt_confirm",
            "smart_money_strong_market_healthy",
            "accumulation_breakout_smt_v1",
            "faber_gtaa_trend_filter_v1",
        ],
    },
    {
        "name": "PURE_TOP3_PLUS_SMT_LIVE_FABER",
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
            "faber_gtaa_trend_filter_v1",
        ],
    },
]


def _load_hypothesis_map() -> dict[str, Hypothesis]:
    out: dict[str, Hypothesis] = {}
    for config in [Path(DEFAULT_CONFIG), GLOBAL_CONFIG, SECTOR_CONFIG, OIL_GAS_CONFIG, THEME_FLOW_CONFIG]:
        for hyp in load_hypotheses(config):
            out[hyp.name] = hyp
    return out


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _row(
    *,
    combo_set: dict[str, Any],
    variant: str,
    max_positions: int,
    max_candidates_per_day: int,
    metrics: dict[str, Any],
    start: str,
    end: str,
) -> dict[str, Any]:
    trades = int(metrics.get("number_of_trades", 0))
    weeks = max(1.0, (pd.Timestamp(end) - pd.Timestamp(start)).days / 7.0)
    return {
        "combo_set": combo_set["name"],
        "variant": variant,
        "n_strategies": len(combo_set["strategies"]),
        "max_positions": max_positions,
        "max_candidates_per_day": max_candidates_per_day,
        "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
        "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
        "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
        "number_of_trades": trades,
        "trades_per_week": round(trades / weeks, 2),
        "agent_reviews": int(metrics.get("agent_reviews", 0)),
        "agent_raise_stop": int(metrics.get("agent_raise_stop", 0)),
        "fills": int(metrics.get("fills", 0)),
        "strategies": " + ".join(combo_set["strategies"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest shared-pool strategy mixes")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--label", default="shared_pool_2025now")
    parser.add_argument("--max-positions", default="3,5,7")
    parser.add_argument("--max-candidates-per-day", default="10,15")
    parser.add_argument("--combo-sets", default="", help="Comma-separated subset names")
    parser.add_argument("--variants", default="BASELINE,EXIT_AGENT")
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.label}_{args.universe}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(args.universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, args.start, args.end)

    print(f"[shared-pool] universe={args.universe} symbols={len(symbols)}")
    print("[shared-pool] building features once...")
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    print("[shared-pool] enriching sector/cycle fields...")
    features = enrich_features_with_rotation(features, root=ROOT)

    hypotheses = _load_hypothesis_map()
    wanted = {item.strip() for item in args.combo_sets.split(",") if item.strip()}
    combo_sets = [item for item in COMBO_SETS if not wanted or item["name"] in wanted]
    max_positions_values = _parse_ints(args.max_positions)
    max_candidates_values = _parse_ints(args.max_candidates_per_day)
    variants = {item.strip().upper() for item in args.variants.split(",") if item.strip()}

    rows: list[dict[str, Any]] = []
    for combo_set in combo_sets:
        missing = [name for name in combo_set["strategies"] if name not in hypotheses]
        if missing:
            raise KeyError(f"{combo_set['name']} missing hypotheses: {missing}")
        print(f"[shared-pool] cache {combo_set['name']} ({len(combo_set['strategies'])} strategies)")
        signal_cache = combo_harness.build_signal_cache_for_combo(
            features,
            [hypotheses[name] for name in combo_set["strategies"]],
            combo_set["name"],
            clean_symbols,
        )
        combo_harness.install_edge_cache(signal_cache)

        for max_positions in max_positions_values:
            for max_candidates_per_day in max_candidates_values:
                for enabled in [False, True]:
                    variant = "EXIT_AGENT" if enabled else "BASELINE"
                    if variant not in variants:
                        continue
                    cfg = LivePipelineBacktestConfig(
                        start_date=args.start,
                        end_date=args.end,
                        max_positions=max_positions,
                        max_candidates_per_day=max_candidates_per_day,
                        edge_strategy_name=combo_set["name"],
                    )
                    print(
                        f"[shared-pool] {combo_set['name']} {variant} "
                        f"p{max_positions}/c{max_candidates_per_day}"
                    )
                    result = run_with_exit_agent(
                        universe_data,
                        vnindex,
                        features,
                        cfg,
                        enabled=enabled,
                    )
                    row = _row(
                        combo_set=combo_set,
                        variant=variant,
                        max_positions=max_positions,
                        max_candidates_per_day=max_candidates_per_day,
                        metrics=result["metrics"],
                        start=args.start,
                        end=args.end,
                    )
                    rows.append(row)
                    stem = f"{combo_set['name']}_p{max_positions}_c{max_candidates_per_day}_{variant.lower()}"
                    pd.DataFrame(result["trades"]).to_csv(out_dir / f"{stem}_trades.csv", index=False)
                    if not result["equity_frame"].empty:
                        result["equity_frame"].to_csv(out_dir / f"{stem}_equity.csv")
                    pd.DataFrame(rows).sort_values("total_return_pct", ascending=False).to_csv(
                        out_dir / "summary.csv",
                        index=False,
                    )
                    print(
                        f"  -> ret={row['total_return_pct']:+.2f}% "
                        f"sharpe={row['sharpe_ratio']:.2f} "
                        f"mdd={row['max_drawdown_pct']:.2f}% "
                        f"wr={row['win_rate_pct']:.2f}% "
                        f"trades={row['number_of_trades']}"
                    )

    summary = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"[shared-pool] saved {out_dir / 'summary.csv'}")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
