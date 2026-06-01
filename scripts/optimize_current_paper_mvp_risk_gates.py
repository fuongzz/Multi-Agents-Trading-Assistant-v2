"""Optimize risk gates for compound Core MVP exact-contract candidates."""

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
DEFAULT_OUT_DIR = ROOT / "reports" / "mvp_current_paper_contract_risk_gate_optimization_vn100_2020_to_2026-05-29"


STRATEGY_SETS: list[dict[str, Any]] = [
    {"name": "core_mvp9", "strategies": list(MVP_EDGE_STRATEGIES)},
    {
        "name": "top5_full_history_net",
        "strategies": [
            "money_cycle_reset_smt_confirm",
            "breakout_55_smt_v1",
            "compression_breakout_smt_v1",
            "mean_reversion_uptrend_ma50_v1",
            "smart_money_strong_market_healthy",
        ],
    },
    {
        "name": "cycle_breakout_compression_accum",
        "strategies": [
            "money_cycle_reset_smt_confirm",
            "breakout_55_smt_v1",
            "compression_breakout_smt_v1",
            "accumulation_breakout_smt_v1",
        ],
    },
]


GATES: list[dict[str, Any]] = [
    {"name": "no_extra_gate"},
    {"name": "rank2_only", "max_edge_rank": 2},
    {"name": "regime60", "min_mkt_regime_score": 60.0},
    {"name": "risk_on_only", "allowed_mkt_regime_states": {"RISK_ON"}},
    {"name": "risk_on_regime60", "allowed_mkt_regime_states": {"RISK_ON"}, "min_mkt_regime_score": 60.0},
    {"name": "chdm55_ds045", "min_chdm50": 55.0, "max_ds20": 0.45},
    {
        "name": "rank2_regime60_ds050",
        "max_edge_rank": 2,
        "min_mkt_regime_score": 60.0,
        "max_ds20": 0.50,
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


def _row(
    *,
    strategy_set: str,
    gate: dict[str, Any],
    sleeve_id: str,
    max_positions: int,
    result: dict[str, Any],
    start: str,
    end: str,
) -> dict[str, Any]:
    metrics = result["metrics"]
    equity = result["equity_frame"]
    recent_ret, recent_dd = _period_return(equity, "2025-01-01", end)
    return {
        "strategy_set": strategy_set,
        "gate": gate["name"],
        "sleeve_id": sleeve_id,
        "max_positions": max_positions,
        "sizing_mode": "compound_equity",
        "start": start,
        "end": end,
        "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100.0, 2),
        "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100.0, 2),
        "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100.0, 2),
        "number_of_trades": int(metrics.get("number_of_trades", 0)),
        "fills": int(metrics.get("fills", 0)),
        "return_2025_now_pct": round(recent_ret, 2),
        "drawdown_2025_now_pct": round(recent_dd, 2),
        "contract_id": CONTRACT_ID,
    }


def run_grid(out_dir: Path, start: str, end: str, capital: float) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = resolve_symbols("vn100")
    print(f"[mvp-gate] symbols={len(symbols)} start={start} end={end}")
    universe_data, vnindex = load_local_history(symbols, start, end)
    print("[mvp-gate] building features once...")
    features, _ = build_feature_table(symbols, start, end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    hypotheses_by_name = {h.name: h for h in load_hypotheses(DEFAULT_CONFIG)}
    clean_symbols = sorted({s.upper() for s in symbols})
    rows: list[dict[str, Any]] = []

    for spec in STRATEGY_SETS:
        strategy_set = spec["name"]
        strategies = [s for s in spec["strategies"] if s in hypotheses_by_name]
        signal_cache = build_signal_cache_for_combo(
            features,
            [hypotheses_by_name[s] for s in strategies],
            strategy_set,
            clean_symbols,
        )
        for gate in GATES:
            for sleeve_id, max_positions in [("p2", 2), ("p1", 1)]:
                t0 = time.time()
                print(f"[mvp-gate] {strategy_set} {gate['name']} {sleeve_id}")
                result = run_sleeve(
                    sleeve_id=f"{strategy_set}_{gate['name']}_{sleeve_id}",
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
                    sizing_mode="compound_equity",
                    max_edge_rank=gate.get("max_edge_rank"),
                    min_edge_rank_score=gate.get("min_edge_rank_score"),
                    min_mkt_regime_score=gate.get("min_mkt_regime_score"),
                    allowed_mkt_regime_states=gate.get("allowed_mkt_regime_states"),
                    min_chdm50=gate.get("min_chdm50"),
                    max_ds20=gate.get("max_ds20"),
                )
                item = _row(
                    strategy_set=strategy_set,
                    gate=gate,
                    sleeve_id=sleeve_id,
                    max_positions=max_positions,
                    result=result,
                    start=start,
                    end=end,
                )
                item["elapsed_sec"] = round(time.time() - t0, 1)
                rows.append(item)
                stem = f"{strategy_set}_{gate['name']}_{sleeve_id}"
                result["equity_frame"].to_csv(out_dir / f"{stem}_equity.csv", encoding="utf-8-sig")
                pd.DataFrame(result["trades"]).to_csv(out_dir / f"{stem}_trades.csv", index=False, encoding="utf-8-sig")
                print(
                    f"  ret={item['total_return_pct']:+.2f}% sharpe={item['sharpe_ratio']:.3f} "
                    f"dd={item['max_drawdown_pct']:.2f}% 2025={item['return_2025_now_pct']:+.2f}%"
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
                "note": "Research only. Does not change paper trading pipeline.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize risk gates for compound Core MVP candidates.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-29")
    parser.add_argument("--capital", type=float, default=100_000_000.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_grid(args.out_dir, args.start, args.end, args.capital)
    print(
        summary[
            [
                "strategy_set",
                "gate",
                "sleeve_id",
                "total_return_pct",
                "sharpe_ratio",
                "max_drawdown_pct",
                "return_2025_now_pct",
                "number_of_trades",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )
    print(args.out_dir / "summary.csv")


if __name__ == "__main__":
    main()
