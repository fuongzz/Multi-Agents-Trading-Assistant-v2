"""Research Flow Money V2-native strategies.

These hypotheses use the decomposed Flow V2 columns directly instead of trying
to replace the legacy ``smart_money_score`` inside old strategies.
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
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "flow_v2_native_strategies"


def _hyp(data: dict[str, Any]) -> Hypothesis:
    return Hypothesis.from_dict(data)


FLOW_V2_HYPOTHESES: dict[str, Hypothesis] = {
    "flow_v2_sector_leader_pullback": _hyp(
        {
            "name": "flow_v2_sector_leader_pullback",
            "description": "Sector-cycle leader, constructive sponsorship, pullback near short trend.",
            "universe": "VN100",
            "tags": ["flow_v2", "sector_cycle", "leader_pullback"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "mkt_regime_score", "op": ">=", "value": 50},
                {"column": "mkt_CHDM20", "op": ">", "value": 45},
                {"column": "mkt_DS20", "op": "<=", "value": 0.55},
                {"column": "sector_cycle_score", "op": ">=", "value": 62},
                {"column": "flow_quality_v2_score", "op": ">=", "value": 58},
                {"column": "flow_sponsorship_score", "op": ">=", "value": 55},
                {"column": "distribution_pressure_score", "op": "<=", "value": 62},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "distance_ma20", "op": "between", "value": [-0.04, 0.07]},
                {"column": "DS20", "op": "not_up"},
            ],
            "rank": [
                {"column": "sector_cycle_score", "ascending": False, "weight": 1.5},
                {"column": "flow_quality_v2_score", "ascending": False, "weight": 1.2},
                {"column": "flow_sponsorship_score", "ascending": False, "weight": 1.0},
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
                {"column": "distance_ma20", "ascending": True, "weight": 0.4},
            ],
            "risk": {
                "stop_loss": 0.08,
                "take_profit": 0.30,
                "max_holding_bars": 55,
                "initial_atr_stop_mult": 2.3,
                "trailing_atr_mult": 2.8,
                "trailing_profit_activation": 0.10,
                "use_regime_exposure": True,
            },
        }
    ),
    "flow_v2_sponsorship_breakout20": _hyp(
        {
            "name": "flow_v2_sponsorship_breakout20",
            "description": "20-day breakout with sponsorship and sector cycle confirmation.",
            "universe": "VN100",
            "tags": ["flow_v2", "sponsorship", "breakout"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "mkt_regime_score", "op": ">=", "value": 50},
                {"column": "breakout_20", "op": "==", "value": True},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "ma50_slope_10", "op": ">=", "value": -0.01},
                {"column": "sector_cycle_score", "op": ">=", "value": 60},
                {"column": "flow_sponsorship_score", "op": ">=", "value": 62},
                {"column": "flow_quality_v2_score", "op": ">=", "value": 58},
                {"column": "distribution_pressure_score", "op": "<=", "value": 65},
                {"column": "value_ratio_20", "op": ">=", "value": 1.10},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.58},
                {"column": "DS20", "op": "<=", "value": 0.55},
            ],
            "rank": [
                {"column": "flow_sponsorship_score", "ascending": False, "weight": 1.5},
                {"column": "sector_cycle_score", "ascending": False, "weight": 1.2},
                {"column": "value_ratio_20", "ascending": False, "weight": 1.0},
                {"column": "rs_percentile_20", "ascending": False, "weight": 0.8},
                {"column": "distribution_pressure_score", "ascending": True, "weight": 0.6},
            ],
            "risk": {
                "stop_loss": 0.09,
                "take_profit": 0.36,
                "max_holding_bars": 55,
                "initial_atr_stop_mult": 2.5,
                "trailing_atr_mult": 3.0,
                "trailing_profit_activation": 0.12,
                "use_regime_exposure": True,
            },
        }
    ),
    "flow_v2_sponsorship_breakout55": _hyp(
        {
            "name": "flow_v2_sponsorship_breakout55",
            "description": "55-day breakout from a longer base with sponsorship and low distribution.",
            "universe": "VN100",
            "tags": ["flow_v2", "sponsorship", "breakout", "55day"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "mkt_regime_score", "op": ">=", "value": 50},
                {"column": "breakout_55", "op": "==", "value": True},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "sector_cycle_score", "op": ">=", "value": 60},
                {"column": "flow_sponsorship_score", "op": ">=", "value": 60},
                {"column": "flow_quality_v2_score", "op": ">=", "value": 56},
                {"column": "distribution_pressure_score", "op": "<=", "value": 65},
                {"column": "value_ratio_20", "op": ">=", "value": 1.10},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.58},
                {"column": "CHDM50", "op": ">=", "value": 45},
                {"column": "DS20", "op": "<=", "value": 0.55},
            ],
            "rank": [
                {"column": "sector_cycle_score", "ascending": False, "weight": 1.4},
                {"column": "flow_sponsorship_score", "ascending": False, "weight": 1.3},
                {"column": "flow_quality_v2_score", "ascending": False, "weight": 1.0},
                {"column": "value_ratio_20", "ascending": False, "weight": 0.8},
                {"column": "rs_percentile_20", "ascending": False, "weight": 0.8},
            ],
            "risk": {
                "stop_loss": 0.09,
                "take_profit": 0.40,
                "max_holding_bars": 65,
                "initial_atr_stop_mult": 2.5,
                "trailing_atr_mult": 3.0,
                "trailing_profit_activation": 0.12,
                "use_regime_exposure": True,
            },
        }
    ),
    "flow_v2_absorption_reset": _hyp(
        {
            "name": "flow_v2_absorption_reset",
            "description": "Absorption/reset setup after pullback, only when distribution pressure is controlled.",
            "universe": "VN100",
            "tags": ["flow_v2", "absorption", "reset"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "mkt_DS20", "op": "<=", "value": 0.60},
                {"column": "flow_absorption_score", "op": ">=", "value": 62},
                {"column": "stock_lifecycle_score", "op": ">=", "value": 55},
                {"column": "flow_quality_v2_score", "op": ">=", "value": 55},
                {"column": "distribution_pressure_score", "op": "<=", "value": 58},
                {"column": "sector_cycle_score", "op": ">=", "value": 52},
                {"column": "CHDM03", "op": "between", "value": [20, 58]},
                {"column": "CHDM03", "op": "up"},
                {"column": "DS20", "op": "not_up"},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.48},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "distance_ma20", "op": "between", "value": [-0.07, 0.06]},
            ],
            "rank": [
                {"column": "flow_absorption_score", "ascending": False, "weight": 1.4},
                {"column": "stock_lifecycle_score", "ascending": False, "weight": 1.0},
                {"column": "distribution_pressure_score", "ascending": True, "weight": 1.0},
                {"column": "sector_cycle_score", "ascending": False, "weight": 0.8},
                {"column": "CHDM03", "ascending": True, "weight": 0.5},
            ],
            "risk": {
                "stop_loss": 0.08,
                "take_profit": 0.28,
                "max_holding_bars": 45,
                "initial_atr_stop_mult": 2.2,
                "trailing_atr_mult": 2.7,
                "trailing_profit_activation": 0.10,
                "use_regime_exposure": True,
            },
        }
    ),
    "flow_v2_sector_cycle_continuation": _hyp(
        {
            "name": "flow_v2_sector_cycle_continuation",
            "description": "Top sector cycle, persistent flow, no chase continuation.",
            "universe": "VN100",
            "tags": ["flow_v2", "sector_cycle", "continuation"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "mkt_regime_score", "op": ">=", "value": 52},
                {"column": "sector_cycle_score", "op": ">=", "value": 68},
                {"column": "flow_quality_v2_score", "op": ">=", "value": 58},
                {"column": "flow_sponsorship_score", "op": ">=", "value": 56},
                {"column": "distribution_pressure_score", "op": "<=", "value": 64},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "ma20_slope_5", "op": ">=", "value": 0},
                {"column": "distance_ma20", "op": "between", "value": [-0.02, 0.08]},
                {"column": "ret_5d", "op": "between", "value": [-0.04, 0.12]},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.56},
                {"column": "DS20", "op": "<=", "value": 0.55},
            ],
            "rank": [
                {"column": "sector_cycle_score", "ascending": False, "weight": 1.7},
                {"column": "flow_quality_v2_score", "ascending": False, "weight": 1.1},
                {"column": "flow_sponsorship_score", "ascending": False, "weight": 1.0},
                {"column": "rs_percentile_20", "ascending": False, "weight": 0.8},
                {"column": "distance_ma20", "ascending": True, "weight": 0.4},
            ],
            "risk": {
                "stop_loss": 0.08,
                "take_profit": 0.32,
                "max_holding_bars": 50,
                "initial_atr_stop_mult": 2.3,
                "trailing_atr_mult": 2.8,
                "trailing_profit_activation": 0.10,
                "use_regime_exposure": True,
            },
        }
    ),
    "flow_v2_momentum_leader": _hyp(
        {
            "name": "flow_v2_momentum_leader",
            "description": "Bull-market leader: V2 sponsorship/value flow plus strong RS and controlled distribution.",
            "universe": "VN100",
            "tags": ["flow_v2", "momentum", "leader"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "mkt_regime_score", "op": ">=", "value": 55},
                {"column": "mkt_CHDM20", "op": ">", "value": 48},
                {"column": "mkt_DS20", "op": "<=", "value": 0.50},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "ma20_slope_5", "op": ">=", "value": 0},
                {"column": "ma50_slope_10", "op": ">=", "value": -0.005},
                {"column": "distance_ma20", "op": "between", "value": [-0.025, 0.075]},
                {"column": "ret_5d", "op": "between", "value": [-0.035, 0.12]},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.68},
                {"column": "value_ratio_20", "op": ">=", "value": 1.15},
                {"column": "flow_sponsorship_score", "op": ">=", "value": 55},
                {"column": "flow_quality_v2_score", "op": ">=", "value": 50},
                {"column": "distribution_pressure_score", "op": "<=", "value": 55},
                {"column": "CHDM50", "op": ">=", "value": 50},
                {"column": "DS20", "op": "<=", "value": 0.45},
            ],
            "rank": [
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.4},
                {"column": "value_ratio_20", "ascending": False, "weight": 1.2},
                {"column": "flow_sponsorship_score", "ascending": False, "weight": 1.1},
                {"column": "sector_cycle_score", "ascending": False, "weight": 0.9},
                {"column": "distribution_pressure_score", "ascending": True, "weight": 0.7},
            ],
            "risk": {
                "stop_loss": 0.08,
                "take_profit": 0.32,
                "max_holding_bars": 45,
                "initial_atr_stop_mult": 2.3,
                "trailing_atr_mult": 2.8,
                "trailing_profit_activation": 0.10,
                "use_regime_exposure": True,
            },
        }
    ),
    "flow_v2_momentum_pullback": _hyp(
        {
            "name": "flow_v2_momentum_pullback",
            "description": "Momentum leader pullback: high RS, high value expansion, V2 sponsorship, no short-term distribution.",
            "universe": "VN100",
            "tags": ["flow_v2", "momentum", "pullback"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "mkt_regime_score", "op": ">=", "value": 55},
                {"column": "mkt_DS20", "op": "<=", "value": 0.50},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "distance_ma20", "op": "between", "value": [-0.045, 0.045]},
                {"column": "ret_5d", "op": "between", "value": [-0.06, 0.08]},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.65},
                {"column": "value_ratio_20", "op": ">=", "value": 1.05},
                {"column": "flow_absorption_score", "op": ">=", "value": 55},
                {"column": "flow_sponsorship_score", "op": ">=", "value": 48},
                {"column": "distribution_pressure_score", "op": "<=", "value": 52},
                {"column": "CHDM50", "op": ">=", "value": 45},
                {"column": "DS20", "op": "not_up"},
            ],
            "rank": [
                {"column": "flow_absorption_score", "ascending": False, "weight": 1.2},
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.2},
                {"column": "value_ratio_20", "ascending": False, "weight": 1.0},
                {"column": "distribution_pressure_score", "ascending": True, "weight": 0.9},
                {"column": "distance_ma20", "ascending": True, "weight": 0.4},
            ],
            "risk": {
                "stop_loss": 0.075,
                "take_profit": 0.28,
                "max_holding_bars": 42,
                "initial_atr_stop_mult": 2.1,
                "trailing_atr_mult": 2.6,
                "trailing_profit_activation": 0.09,
                "use_regime_exposure": True,
            },
        }
    ),
}


SUITES: dict[str, list[str]] = {
    "all_v2": list(FLOW_V2_HYPOTHESES),
    "leaders": ["flow_v2_sector_leader_pullback", "flow_v2_sector_cycle_continuation"],
    "breakouts": ["flow_v2_sponsorship_breakout20", "flow_v2_sponsorship_breakout55"],
    "pullback_absorption": ["flow_v2_sector_leader_pullback", "flow_v2_absorption_reset"],
    "no_absorption": [
        "flow_v2_sector_leader_pullback",
        "flow_v2_sponsorship_breakout20",
        "flow_v2_sponsorship_breakout55",
        "flow_v2_sector_cycle_continuation",
    ],
    "momentum_v2": ["flow_v2_momentum_leader", "flow_v2_momentum_pullback"],
    "momentum_plus_absorption": [
        "flow_v2_momentum_leader",
        "flow_v2_momentum_pullback",
        "flow_v2_absorption_reset",
    ],
    "momentum_plus_leader": [
        "flow_v2_momentum_leader",
        "flow_v2_momentum_pullback",
        "flow_v2_sector_leader_pullback",
    ],
    **{name: [name] for name in FLOW_V2_HYPOTHESES},
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
            win_rate_pct=("pnl_pct", lambda item: float((item > 0).mean() * 100.0)),
            pnl=("pnl", "sum"),
        )
        .reset_index()
        .sort_values("pnl", ascending=False)
    )
    out["sum_pnl_pct"] *= 100.0
    out["avg_pnl_pct"] *= 100.0
    return out


def _summary_row(
    *,
    suite: str,
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
        "suite": suite,
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
    suites: list[str],
    max_positions_values: list[int],
    max_candidates_values: list[int],
) -> Path:
    out_dir = OUT_ROOT / f"{label}_{universe}_{start}_{end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, start, end)

    print(f"[flow-v2-native] universe={universe} symbols={len(clean_symbols)} period={start}->{end}")
    print("[flow-v2-native] building feature table once...")
    features, _ = build_feature_table(symbols, start, end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for suite in suites:
        hypotheses = [FLOW_V2_HYPOTHESES[name] for name in SUITES[suite]]
        signal_cache = combo_harness.build_signal_cache_for_combo(
            features,
            hypotheses,
            f"FLOW_V2_NATIVE_{suite}",
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
                    edge_strategy_name=f"FLOW_V2_NATIVE_{suite}",
                )
                print(f"[flow-v2-native] run {suite} p{max_positions}/c{max_candidates_per_day}")
                result = run_with_exit_agent(universe_data, vnindex, features, cfg, enabled=True)
                row = _summary_row(
                    suite=suite,
                    max_positions=max_positions,
                    max_candidates_per_day=max_candidates_per_day,
                    result=result,
                    start=start,
                    end=end,
                )
                rows.append(row)

                stem = f"{suite}_p{max_positions}_c{max_candidates_per_day}"
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
    parser = argparse.ArgumentParser(description="Research V2-native flow money strategies")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-19")
    parser.add_argument("--label", default="flow_v2_native_2025now")
    parser.add_argument("--suites", default="all_v2,leaders,breakouts,pullback_absorption,no_absorption")
    parser.add_argument("--max-positions", default="4,5")
    parser.add_argument("--max-candidates-per-day", default="10")
    args = parser.parse_args()

    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    unknown = sorted(set(suites) - set(SUITES))
    if unknown:
        raise ValueError(f"Unsupported suites: {unknown}. Available: {sorted(SUITES)}")
    max_positions = [int(item.strip()) for item in args.max_positions.split(",") if item.strip()]
    max_candidates = [int(item.strip()) for item in args.max_candidates_per_day.split(",") if item.strip()]
    out_dir = run_period(
        universe=args.universe,
        start=args.start,
        end=args.end,
        label=args.label,
        suites=suites,
        max_positions_values=max_positions,
        max_candidates_values=max_candidates,
    )
    print(f"[flow-v2-native] saved {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
