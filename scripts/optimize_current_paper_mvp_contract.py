"""Search Core MVP strategy subsets under the exact current-paper contract."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG
from multiagents_trading_assistant.edge_lab.strategy_sleeves import MVP_EDGE_STRATEGIES
from scripts.backtest_combos_unbiased import build_signal_cache_for_combo, load_local_history, resolve_symbols
from scripts.run_current_paper_contract_backtest import CONTRACT_ID, run_sleeve


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "reports" / "mvp_current_paper_contract_optimization_vn100_2020_to_2026-05-29"


VARIANTS: list[dict[str, Any]] = [
    {
        "variant": "core_mvp9",
        "description": "Current production Core MVP9 set.",
        "strategies": list(MVP_EDGE_STRATEGIES),
    },
    {
        "variant": "core_minus_breakout_accum",
        "description": "Drop breakout_after_accumulation_v3, the weakest full-history contributor.",
        "strategies": [s for s in MVP_EDGE_STRATEGIES if s != "breakout_after_accumulation_v3"],
    },
    {
        "variant": "no_leader_pullbacks",
        "description": "Remove the two leader-pullback entries that add many low-edge recent trades.",
        "strategies": [
            s
            for s in MVP_EDGE_STRATEGIES
            if s not in {"leader_pullback_market_regime_v3", "leader_pullback_market_healthy_v2"}
        ],
    },
    {
        "variant": "top5_full_history_net",
        "description": "Keep strongest full-history contributors from the exact-contract attribution.",
        "strategies": [
            "money_cycle_reset_smt_confirm",
            "breakout_55_smt_v1",
            "compression_breakout_smt_v1",
            "mean_reversion_uptrend_ma50_v1",
            "smart_money_strong_market_healthy",
        ],
    },
    {
        "variant": "cycle_breakout_compression",
        "description": "Concentrated sleeve: money cycle, breakout 55, compression breakout.",
        "strategies": [
            "money_cycle_reset_smt_confirm",
            "breakout_55_smt_v1",
            "compression_breakout_smt_v1",
        ],
    },
    {
        "variant": "cycle_breakout_compression_accum",
        "description": "Add accumulation SMT to the concentrated cycle/breakout sleeve.",
        "strategies": [
            "money_cycle_reset_smt_confirm",
            "breakout_55_smt_v1",
            "compression_breakout_smt_v1",
            "accumulation_breakout_smt_v1",
        ],
    },
    {
        "variant": "cycle_only",
        "description": "Single highest net contributor.",
        "strategies": ["money_cycle_reset_smt_confirm"],
    },
    {
        "variant": "breakout55_only",
        "description": "Single breakout 55 SMT contributor.",
        "strategies": ["breakout_55_smt_v1"],
    },
    {
        "variant": "compression_only",
        "description": "Single compression breakout contributor.",
        "strategies": ["compression_breakout_smt_v1"],
    },
]


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float((series / series.cummax() - 1.0).min())


def _period_return(equity: pd.DataFrame, start: str, end: str) -> tuple[float, float]:
    if equity.empty:
        return 0.0, 0.0
    frame = equity.copy()
    dates = pd.to_datetime(frame.index if "date" not in frame.columns else frame["date"]).normalize()
    frame = frame.assign(_date=dates)
    period = frame[(frame["_date"] >= pd.Timestamp(start)) & (frame["_date"] <= pd.Timestamp(end))]
    if period.empty:
        return 0.0, 0.0
    values = period["equity"].astype(float)
    return float(values.iloc[-1] / values.iloc[0] - 1.0) * 100.0, _max_drawdown(values) * 100.0


def _rows_for_result(
    *,
    variant: str,
    description: str,
    sleeve_id: str,
    max_positions: int,
    strategies: list[str],
    result: dict[str, Any],
    start: str,
    end: str,
    sizing_mode: str,
) -> dict[str, Any]:
    metrics = result["metrics"]
    equity = result["equity_frame"]
    recent_return, recent_dd = _period_return(equity, "2025-01-01", end)
    return {
        "variant": variant,
        "description": description,
        "sleeve_id": sleeve_id,
        "max_positions": max_positions,
        "n_strategies": len(strategies),
        "strategies": " + ".join(strategies),
        "start": start,
        "end": end,
        "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100.0, 2),
        "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100.0, 2),
        "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100.0, 2),
        "number_of_trades": int(metrics.get("number_of_trades", 0)),
        "fills": int(metrics.get("fills", 0)),
        "open_positions": len(result["open_positions"]),
        "return_2025_now_pct": round(recent_return, 2),
        "drawdown_2025_now_pct": round(recent_dd, 2),
        "contract_id": CONTRACT_ID,
        "sizing_mode": sizing_mode,
    }


def run_optimization(out_dir: Path, start: str, end: str, capital: float, sizing_modes: list[str]) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = resolve_symbols("vn100")
    print(f"[mvp-opt] symbols={len(symbols)} start={start} end={end}")
    universe_data, vnindex = load_local_history(symbols, start, end)
    print("[mvp-opt] building features once...")
    features, _ = build_feature_table(symbols, start, end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    hypotheses_by_name = {h.name: h for h in load_hypotheses(DEFAULT_CONFIG)}
    clean_symbols = sorted({s.upper() for s in symbols})
    rows: list[dict[str, Any]] = []

    for spec in VARIANTS:
        variant = str(spec["variant"])
        strategies = [s for s in spec["strategies"] if s in hypotheses_by_name]
        if not strategies:
            continue
        print(f"[mvp-opt] {variant}: {strategies}")
        signal_cache = build_signal_cache_for_combo(
            features,
            [hypotheses_by_name[s] for s in strategies],
            variant,
            clean_symbols,
        )
        for sizing_mode in sizing_modes:
            for sleeve_id, max_positions in [("p5", 5), ("p4", 4), ("p3", 3), ("p2", 2), ("p1", 1)]:
                t0 = time.time()
                result = run_sleeve(
                    sleeve_id=f"{variant}_{sleeve_id}_{sizing_mode}",
                    max_positions=max_positions,
                    universe_data=universe_data,
                    vnindex=vnindex,
                    features=features,
                    signal_cache=signal_cache,
                    start=start,
                    end=end,
                    initial_capital=capital,
                    max_candidates_per_day=10,
                    lot_size=100,
                    price_unit_multiplier=1000.0,
                    use_exit_agent=True,
                    sizing_mode=sizing_mode,
                )
                row = _rows_for_result(
                    variant=variant,
                    description=str(spec["description"]),
                    sleeve_id=sleeve_id,
                    max_positions=max_positions,
                    strategies=strategies,
                    result=result,
                    start=start,
                    end=end,
                    sizing_mode=sizing_mode,
                )
                row["elapsed_sec"] = round(time.time() - t0, 1)
                rows.append(row)
                stem = f"{variant}_{sleeve_id}_{sizing_mode}"
                result["equity_frame"].to_csv(out_dir / f"{stem}_equity.csv", encoding="utf-8-sig")
                pd.DataFrame(result["trades"]).to_csv(out_dir / f"{stem}_trades.csv", index=False, encoding="utf-8-sig")
                print(
                    f"  {sleeve_id}/{sizing_mode}: ret={row['total_return_pct']:+.2f}% "
                    f"sharpe={row['sharpe_ratio']:.3f} dd={row['max_drawdown_pct']:.2f}% "
                    f"2025now={row['return_2025_now_pct']:+.2f}%"
                )

    summary = pd.DataFrame(rows).sort_values(["sharpe_ratio", "total_return_pct"], ascending=[False, False])
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "generated_at": pd.Timestamp.now().isoformat(),
                "start": start,
                "end": end,
                "capital": capital,
                "contract_id": CONTRACT_ID,
                "sizing_modes": sizing_modes,
                "note": "Research only. Does not change paper trading pipeline.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize MVP strategy subsets under exact current-paper contract.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-29")
    parser.add_argument("--capital", type=float, default=100_000_000.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--sizing-modes",
        default="fixed_initial_capital,compound_equity",
        help="Comma-separated sizing modes: fixed_initial_capital, compound_equity.",
    )
    args = parser.parse_args()
    sizing_modes = [item.strip() for item in args.sizing_modes.split(",") if item.strip()]
    summary = run_optimization(args.out_dir, args.start, args.end, args.capital, sizing_modes)
    print(summary[["variant", "sleeve_id", "sizing_mode", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "return_2025_now_pct", "number_of_trades"]].to_string(index=False))
    print(args.out_dir / "summary.csv")


if __name__ == "__main__":
    main()
