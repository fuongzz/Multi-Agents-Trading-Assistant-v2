"""Compare entry market-gate variants on the current MVP shared pool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from scripts.backtest_position_exit_agent import run_with_exit_agent
from scripts.research_shared_pool_strategy_mix import _load_hypothesis_map
from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.strategy_sleeves import MVP_EDGE_STRATEGIES
from multiagents_trading_assistant.research.sector_rotation.enrich_features import (
    enrich_features_with_rotation,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "entry_market_gate_variants"


def _row(*, mode: str, result: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    metrics = result["metrics"]
    trades = int(metrics.get("number_of_trades", 0))
    weeks = max(1.0, (pd.Timestamp(end) - pd.Timestamp(start)).days / 7.0)
    return {
        "mode": mode,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest MVP entry market-gate variants")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--label", default="mvp_2025now")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    parser.add_argument("--modes", default="baseline,riskoff_veto,riskoff_recovery")
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.label}_{args.universe}_{args.start}_{args.end}_p{args.max_positions}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(args.universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, args.start, args.end)

    print("[entry-gate] building features...")
    feature_result = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = feature_result[0] if isinstance(feature_result, tuple) else feature_result
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    features = enrich_features_with_rotation(features, root=ROOT)

    hypotheses = _load_hypothesis_map()
    missing = [name for name in MVP_EDGE_STRATEGIES if name not in hypotheses]
    if missing:
        raise KeyError(f"Missing MVP hypotheses: {missing}")

    signal_cache = combo_harness.build_signal_cache_for_combo(
        features,
        [hypotheses[name] for name in MVP_EDGE_STRATEGIES],
        "PURE_TOP3_PLUS_SMT_LIVE",
        clean_symbols,
    )
    combo_harness.install_edge_cache(signal_cache)

    rows: list[dict[str, Any]] = []
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        cfg = LivePipelineBacktestConfig(
            start_date=args.start,
            end_date=args.end,
            max_positions=args.max_positions,
            max_candidates_per_day=args.max_candidates_per_day,
            edge_strategy_name="PURE_TOP3_PLUS_SMT_LIVE",
        )
        print(f"[entry-gate] mode={mode}")
        result = run_with_exit_agent(
            universe_data,
            vnindex,
            features,
            cfg,
            enabled=True,
            entry_market_gate=mode,
        )
        row = _row(mode=mode, result=result, start=args.start, end=args.end)
        rows.append(row)
        stem = f"{mode}_p{args.max_positions}_c{args.max_candidates_per_day}"
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
    print(f"[entry-gate] saved {out_dir / 'summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
