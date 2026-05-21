"""Research strategy-specific cycle/runner variants for the MVP pool.

The earlier global cycle boosts were too blunt. This script keeps the MVP pool
intact and tests narrower interventions:

- strategy-specific weak-cycle vetoes after the original signal passes;
- leader/accumulation runner risk rules to avoid early exits in bull phases;
- combinations of the above.
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
from multiagents_trading_assistant.edge_lab.strategy_sleeves import MVP_EDGE_STRATEGIES


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "cycle_first_strategy_specific"


VARIANTS = [
    "baseline",
    "money_reset_veto_moderate",
    "money_reset_veto_strict",
    "leader_runner_risk",
    "leader_runner_plus_money_veto",
    "accumulation_runner_risk",
]

LEADER_STRATEGIES = {
    "leader_pullback_market_regime_v3",
    "leader_pullback_market_healthy_v2",
    "breakout_55_smt_v1",
}

ACCUMULATION_STRATEGIES = {
    "accumulation_breakout_smt_v1",
    "compression_breakout_smt_v1",
}

MONEY_RESET = "money_cycle_reset_smt_confirm"


def _load_mvp_hypotheses(
    *,
    include_strategies: set[str] | None = None,
    exclude_strategies: set[str] | None = None,
) -> list[Hypothesis]:
    by_name = {hyp.name: hyp for hyp in load_hypotheses(DEFAULT_CONFIG)}
    names = list(MVP_EDGE_STRATEGIES)
    if include_strategies:
        names = [name for name in names if name in include_strategies]
    if exclude_strategies:
        names = [name for name in names if name not in exclude_strategies]
    return [by_name[name] for name in names if name in by_name]


def _num(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    try:
        value = row.get(key, default)
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _risk_patch(strategy: str, variant: str) -> dict[str, Any]:
    if variant in {"leader_runner_risk", "leader_runner_plus_money_veto"} and strategy in LEADER_STRATEGIES:
        return {
            "take_profit": 0.45,
            "max_holding_bars": 90,
            "trailing_atr_mult": 3.2,
            "trailing_profit_activation": 0.14,
        }
    if variant == "accumulation_runner_risk" and strategy in ACCUMULATION_STRATEGIES:
        return {
            "take_profit": 0.45,
            "max_holding_bars": 75,
            "trailing_atr_mult": 3.0,
            "trailing_profit_activation": 0.12,
        }
    return {}


def _money_veto(row: dict[str, Any] | None, variant: str) -> bool:
    if variant not in {"money_reset_veto_moderate", "leader_runner_plus_money_veto", "money_reset_veto_strict"}:
        return False
    rs20 = _num(row, "rs_percentile_20")
    ds20 = _num(row, "DS20", 1.0)
    sector = _num(row, "sector_leadership_score")
    smt_no_sector = _num(row, "smart_money_score_no_sector")
    dist10 = _num(row, "distribution_days_10")
    if variant == "money_reset_veto_strict":
        return rs20 < 0.65 or ds20 > 0.35 or smt_no_sector < 62 or sector < 50 or dist10 > 0
    return (rs20 < 0.60 and sector < 55) or ds20 > 0.55 or smt_no_sector < 55


def transform_signal_cache(signal_cache: dict, features: pd.DataFrame, variant: str) -> dict:
    if variant == "baseline":
        return signal_cache
    cols = [
        "date",
        "symbol",
        "rs_percentile_20",
        "DS20",
        "distribution_days_10",
        "sector_leadership_score",
        "smart_money_score_no_sector",
    ]
    available = [col for col in cols if col in features.columns]
    lookup = features[available].set_index(["date", "symbol"]).to_dict("index")
    out = {}
    for date, day in signal_cache.items():
        patched = {}
        for symbol, payload in day.items():
            current = dict(payload)
            strategy = str(current.get("strategy_name") or "")
            if current.get("passed"):
                row = lookup.get((pd.Timestamp(date).normalize(), symbol))
                if strategy == MONEY_RESET and _money_veto(row, variant):
                    current["passed"] = False
                    current["filters_failed"] = [*(current.get("filters_failed") or []), f"{variant}:money_reset_veto"]
                patch = _risk_patch(strategy, variant)
                if patch:
                    risk = dict(current.get("risk") or {})
                    risk.update(patch)
                    current["risk"] = risk
            patched[symbol] = current
        out[date] = patched
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
    variants: list[str],
    include_strategies: set[str] | None = None,
    exclude_strategies: set[str] | None = None,
) -> Path:
    out_dir = OUT_ROOT / f"{label}_{universe}_{start}_{end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, start, end)

    print(f"[cycle-specific] universe={universe} symbols={len(clean_symbols)} period={start}->{end}")
    print("[cycle-specific] building feature table once...")
    features, _ = build_feature_table(symbols, start, end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)

    base_cache = combo_harness.build_signal_cache_for_combo(
        features,
        _load_mvp_hypotheses(
            include_strategies=include_strategies,
            exclude_strategies=exclude_strategies,
        ),
        "MVP_STRATEGY_SPECIFIC",
        clean_symbols,
    )

    rows: list[dict[str, Any]] = []
    for variant in variants:
        signal_cache = transform_signal_cache(base_cache, features, variant)
        combo_harness.install_edge_cache(signal_cache)
        for max_positions in max_positions_values:
            for max_candidates_per_day in max_candidates_values:
                cfg = LivePipelineBacktestConfig(
                    start_date=start,
                    end_date=end,
                    max_positions=max_positions,
                    max_candidates_per_day=max_candidates_per_day,
                    edge_strategy_name=f"MVP_STRATEGY_SPECIFIC_{variant}",
                )
                print(f"[cycle-specific] run {variant} p{max_positions}/c{max_candidates_per_day}")
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
    parser = argparse.ArgumentParser(description="Research MVP strategy-specific cycle/runner variants")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-19")
    parser.add_argument("--label", default="cycle_specific_2025now")
    parser.add_argument("--max-positions", default="5")
    parser.add_argument("--max-candidates-per-day", default="10")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument(
        "--include-strategies",
        default="",
        help="Comma-separated strategy ids to include from MVP_EDGE_STRATEGIES. Empty keeps all MVP strategies.",
    )
    parser.add_argument(
        "--exclude-strategies",
        default="",
        help="Comma-separated strategy ids to exclude from MVP_EDGE_STRATEGIES.",
    )
    args = parser.parse_args()

    max_positions = [int(item.strip()) for item in args.max_positions.split(",") if item.strip()]
    max_candidates = [int(item.strip()) for item in args.max_candidates_per_day.split(",") if item.strip()]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    include_strategies = {item.strip() for item in args.include_strategies.split(",") if item.strip()} or None
    exclude_strategies = {item.strip() for item in args.exclude_strategies.split(",") if item.strip()} or None
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unsupported variants: {unknown}")
    out_dir = run_period(
        universe=args.universe,
        start=args.start,
        end=args.end,
        label=args.label,
        max_positions_values=max_positions,
        max_candidates_values=max_candidates,
        variants=variants,
        include_strategies=include_strategies,
        exclude_strategies=exclude_strategies,
    )
    print(f"[cycle-specific] saved {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
