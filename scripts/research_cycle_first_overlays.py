"""Research cycle-first portfolio overlays for the MVP strategy pool.

Unlike ``research_cycle_first_soft_boost.py``, this script does not alter each
strategy's internal rank formula. It first builds the normal MVP signal cache,
then applies small post-signal boosts/penalties using sector, money-cycle, and
stock-cycle features. This tests whether cycle evidence is useful as portfolio
selection context instead of as a primary setup definition.
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
OUT_ROOT = ROOT / "backtest_results" / "cycle_first_overlays"


OVERLAYS = [
    "none",
    "small_cycle_tiebreak",
    "weak_cycle_penalty",
    "weak_cycle_filter",
    "quality_or_penalty",
    "stock_cycle_tiebreak",
]


def _load_mvp_hypotheses() -> list[Hypothesis]:
    by_name = {hyp.name: hyp for hyp in load_hypotheses(DEFAULT_CONFIG)}
    return [by_name[name] for name in MVP_EDGE_STRATEGIES if name in by_name]


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


def _cycle_points(row: dict[str, Any] | None) -> int:
    if row is None:
        return 0
    points = 0
    points += int(_num(row, "sector_leadership_score") >= 70)
    points += int(_num(row, "smart_money_score_no_sector") >= 70)
    points += int(_num(row, "rs_percentile_20") >= 0.80)
    chdm = _num(row, "CHDM50")
    points += int(50 <= chdm <= 65)
    points += int(_num(row, "DS20", 1.0) <= 0.35)
    return points


def _weak_cycle(row: dict[str, Any] | None) -> bool:
    if row is None:
        return True
    return (
        _num(row, "sector_leadership_score") < 45
        and _num(row, "smart_money_score_no_sector") < 50
        and _num(row, "rs_percentile_20") < 0.55
    )


def _overlay_delta(row: dict[str, Any] | None, overlay: str) -> float:
    if overlay == "small_cycle_tiebreak":
        return min(0.35, _cycle_points(row) * 0.07)
    if overlay == "weak_cycle_penalty":
        return -0.35 if _weak_cycle(row) else 0.0
    if overlay == "quality_or_penalty":
        points = _cycle_points(row)
        if points >= 3:
            return 0.25
        if _weak_cycle(row):
            return -0.35
        return 0.0
    if overlay == "stock_cycle_tiebreak":
        delta = 0.0
        delta += 0.12 if _num(row, "rs_percentile_20") >= 0.80 else 0.0
        delta += 0.10 if _num(row, "DS20", 1.0) <= 0.35 else 0.0
        delta += 0.10 if 50 <= _num(row, "CHDM50") <= 65 else 0.0
        return delta
    return 0.0


def apply_overlay(signal_cache: dict, features: pd.DataFrame, overlay: str) -> dict:
    if overlay == "none":
        return signal_cache
    cols = [
        "date",
        "symbol",
        "sector_leadership_score",
        "smart_money_score_no_sector",
        "rs_percentile_20",
        "CHDM50",
        "DS20",
    ]
    available = [col for col in cols if col in features.columns]
    lookup = features[available].set_index(["date", "symbol"]).to_dict("index")
    out = {}
    for date, day in signal_cache.items():
        patched = {}
        for symbol, payload in day.items():
            current = dict(payload)
            if current.get("passed"):
                row = lookup.get((pd.Timestamp(date).normalize(), symbol))
                if overlay == "weak_cycle_filter" and _weak_cycle(row):
                    current["passed"] = False
                    current["filters_failed"] = [*(current.get("filters_failed") or []), "cycle_overlay:weak_cycle"]
                else:
                    current["edge_rank_score"] = round(
                        float(current.get("edge_rank_score") or 0.0) + _overlay_delta(row, overlay),
                        4,
                    )
            patched[symbol] = current
        out[date] = patched
    return out


def _row(
    *,
    overlay: str,
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
        "overlay": overlay,
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
) -> Path:
    out_dir = OUT_ROOT / f"{label}_{universe}_{start}_{end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, start, end)

    print(f"[cycle-overlay] universe={universe} symbols={len(clean_symbols)} period={start}->{end}")
    print("[cycle-overlay] building feature table once...")
    features, _ = build_feature_table(symbols, start, end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)

    base_cache = combo_harness.build_signal_cache_for_combo(
        features,
        _load_mvp_hypotheses(),
        "MVP_OVERLAY",
        clean_symbols,
    )

    rows: list[dict[str, Any]] = []
    for overlay in OVERLAYS:
        signal_cache = apply_overlay(base_cache, features, overlay)
        combo_harness.install_edge_cache(signal_cache)
        for max_positions in max_positions_values:
            for max_candidates_per_day in max_candidates_values:
                cfg = LivePipelineBacktestConfig(
                    start_date=start,
                    end_date=end,
                    max_positions=max_positions,
                    max_candidates_per_day=max_candidates_per_day,
                    edge_strategy_name=f"MVP_OVERLAY_{overlay}",
                )
                print(f"[cycle-overlay] run {overlay} p{max_positions}/c{max_candidates_per_day}")
                result = run_with_exit_agent(universe_data, vnindex, features, cfg, enabled=True)
                row = _row(
                    overlay=overlay,
                    max_positions=max_positions,
                    max_candidates_per_day=max_candidates_per_day,
                    result=result,
                    start=start,
                    end=end,
                )
                rows.append(row)
                stem = f"{overlay}_p{max_positions}_c{max_candidates_per_day}"
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
    parser = argparse.ArgumentParser(description="Research MVP cycle-first portfolio overlays")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-19")
    parser.add_argument("--label", default="cycle_overlay_2025now")
    parser.add_argument("--max-positions", default="5")
    parser.add_argument("--max-candidates-per-day", default="10")
    args = parser.parse_args()

    max_positions = [int(item.strip()) for item in args.max_positions.split(",") if item.strip()]
    max_candidates = [int(item.strip()) for item in args.max_candidates_per_day.split(",") if item.strip()]
    out_dir = run_period(
        universe=args.universe,
        start=args.start,
        end=args.end,
        label=args.label,
        max_positions_values=max_positions,
        max_candidates_values=max_candidates,
    )
    print(f"[cycle-overlay] saved {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
