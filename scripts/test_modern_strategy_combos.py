"""Test modern strategy combos and regime-router variants."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.fish_modern_strategies import MODERN_STRATEGIES


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "multiagents_trading_assistant/edge_lab/configs/vn30_money_smt_hypotheses.json"
CFG_BACKUP = CFG_PATH.with_suffix(".modern_combo.bak")
OUT_DIR = ROOT / "backtest_results/modern_strategy_fishing"
REPORTS_DIR = ROOT / "reports"


ROUTER_STRATEGIES = [
    {
        "name": "router_ep_stress_v1",
        "description": "Regime-router EP sleeve: only in weak/stress markets with extreme value shock.",
        "universe": "VN100",
        "tags": ["modern", "router", "stress", "episodic_pivot"],
        "filters": [
            {"column": "mkt_regime_score", "op": "<=", "value": 48},
            {"column": "data_quality_ok", "op": "==", "value": True},
            {"column": "ret_1d", "op": ">=", "value": 0.045},
            {"column": "body_pct", "op": ">=", "value": 0.035},
            {"column": "close_location", "op": ">=", "value": 0.70},
            {"column": "value_ratio_20", "op": ">=", "value": 1.90},
            {"column": "volume_ratio_20", "op": ">=", "value": 1.60},
            {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
            {"column": "distribution_days_10", "op": "<=", "value": 2},
        ],
        "rank": [
            {"column": "value_ratio_20", "ascending": False, "weight": 1.5},
            {"column": "ret_1d", "ascending": False, "weight": 1.0},
            {"column": "close_location", "ascending": False, "weight": 0.7},
        ],
        "risk": {"stop_loss": 0.055, "take_profit": 0.14, "max_holding_bars": 7, "initial_atr_stop_mult": 1.7, "trailing_atr_mult": 1.9, "trailing_profit_activation": 0.045, "use_regime_exposure": True},
    },
    {
        "name": "router_pocket_recovery_v1",
        "description": "Regime-router recovery sleeve: pocket pivot in recovery/normal market.",
        "universe": "VN100",
        "tags": ["modern", "router", "recovery", "pocket_pivot"],
        "filters": [
            {"column": "mkt_regime_score", "op": "between", "value": [45, 68]},
            {"column": "data_quality_ok", "op": "==", "value": True},
            {"column": "reclaim_ma20_after_pullback", "op": "==", "value": True},
            {"column": "above_ma50", "op": "==", "value": True},
            {"column": "value_ratio_20", "op": ">=", "value": 1.20},
            {"column": "close_location", "op": ">=", "value": 0.60},
            {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
            {"column": "sector_leadership_score", "op": ">=", "value": 55},
            {"column": "distribution_days_10", "op": "<=", "value": 2},
        ],
        "rank": [
            {"column": "pullback_quality_score", "ascending": False, "weight": 1.2},
            {"column": "value_ratio_20", "ascending": False, "weight": 1.0},
            {"column": "sector_leadership_score", "ascending": False, "weight": 0.8},
        ],
        "risk": {"stop_loss": 0.065, "take_profit": 0.18, "max_holding_bars": 20, "initial_atr_stop_mult": 2.0, "trailing_atr_mult": 2.3, "trailing_profit_activation": 0.06, "use_regime_exposure": True},
    },
    {
        "name": "router_sector_normal_v1",
        "description": "Regime-router normal sleeve: sector RS in normal/up markets.",
        "universe": "VN100",
        "tags": ["modern", "router", "sector_rotation"],
        "filters": [
            {"column": "mkt_regime_score", "op": "between", "value": [50, 72]},
            {"column": "data_quality_ok", "op": "==", "value": True},
            {"column": "sector_rs_rank_20d", "op": ">=", "value": 0.70},
            {"column": "sector_breadth_ma20", "op": ">=", "value": 0.55},
            {"column": "sector_leadership_score", "op": ">=", "value": 62},
            {"column": "above_ma50", "op": "==", "value": True},
            {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
            {"column": "smart_money_score_no_sector", "op": ">=", "value": 50},
        ],
        "rank": [
            {"column": "sector_leadership_score", "ascending": False, "weight": 1.4},
            {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
            {"column": "smart_money_score_no_sector", "ascending": False, "weight": 0.8},
        ],
        "risk": {"stop_loss": 0.075, "take_profit": 0.22, "max_holding_bars": 35, "initial_atr_stop_mult": 2.3, "trailing_atr_mult": 2.6, "trailing_profit_activation": 0.08, "use_regime_exposure": True},
    },
    {
        "name": "router_kalman_bull_v1",
        "description": "Regime-router bull sleeve: Kalman trend in strong market regimes.",
        "universe": "VN100",
        "tags": ["modern", "router", "bull", "kalman"],
        "filters": [
            {"column": "mkt_regime_score", "op": ">=", "value": 62},
            {"column": "data_quality_ok", "op": "==", "value": True},
            {"column": "kalman_strong_uptrend", "op": "==", "value": True},
            {"column": "kalman_confidence", "op": ">=", "value": 0.45},
            {"column": "kalman_shock_score", "op": "<=", "value": 65},
            {"column": "above_ma50", "op": "==", "value": True},
            {"column": "sector_leadership_score", "op": ">=", "value": 55},
            {"column": "smart_money_score_no_sector", "op": ">=", "value": 52},
        ],
        "rank": [
            {"column": "kalman_confidence", "ascending": False, "weight": 1.2},
            {"column": "sector_leadership_score", "ascending": False, "weight": 0.9},
            {"column": "smart_money_score_no_sector", "ascending": False, "weight": 0.9},
        ],
        "risk": {"stop_loss": 0.075, "take_profit": 0.23, "max_holding_bars": 35, "initial_atr_stop_mult": 2.2, "trailing_atr_mult": 2.6, "trailing_profit_activation": 0.08, "use_regime_exposure": True},
    },
    {
        "name": "router_vol_bull_v1",
        "description": "Regime-router bull sleeve: volatility squeeze breakout in strong regimes.",
        "universe": "VN100",
        "tags": ["modern", "router", "bull", "squeeze"],
        "filters": [
            {"column": "mkt_regime_score", "op": ">=", "value": 58},
            {"column": "data_quality_ok", "op": "==", "value": True},
            {"column": "volatility_squeeze_break", "op": "==", "value": True},
            {"column": "above_ma20", "op": "==", "value": True},
            {"column": "value_ratio_20", "op": ">=", "value": 1.10},
            {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
        ],
        "rank": [
            {"column": "value_ratio_20", "ascending": False, "weight": 1.0},
            {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
            {"column": "range_pct", "ascending": True, "weight": 0.5},
        ],
        "risk": {"stop_loss": 0.07, "take_profit": 0.20, "max_holding_bars": 25, "initial_atr_stop_mult": 2.1, "trailing_atr_mult": 2.4, "trailing_profit_activation": 0.07, "use_regime_exposure": True},
    },
    {
        "name": "router_pocket_bull_only_v2",
        "description": "Bull-only pocket pivot: strict VNINDEX and money-cycle gate.",
        "universe": "VN100",
        "tags": ["modern", "router", "bull_only", "pocket_pivot"],
        "filters": [
            {"column": "mkt_regime_score", "op": ">=", "value": 62},
            {"column": "vni_above_ma20", "op": "==", "value": True},
            {"column": "vni_above_ma50", "op": "==", "value": True},
            {"column": "mkt_DS20", "op": "<=", "value": 0.50},
            {"column": "data_quality_ok", "op": "==", "value": True},
            {"column": "reclaim_ma20_after_pullback", "op": "==", "value": True},
            {"column": "above_ma50", "op": "==", "value": True},
            {"column": "value_ratio_20", "op": ">=", "value": 1.15},
            {"column": "close_location", "op": ">=", "value": 0.60},
            {"column": "sector_leadership_score", "op": ">=", "value": 58},
            {"column": "rs_percentile_20", "op": ">=", "value": 0.58},
        ],
        "rank": [
            {"column": "edge_score_no_sector", "ascending": False, "weight": 1.1},
            {"column": "pullback_quality_score", "ascending": False, "weight": 1.0},
            {"column": "sector_leadership_score", "ascending": False, "weight": 0.9},
        ],
        "risk": {"stop_loss": 0.06, "take_profit": 0.18, "max_holding_bars": 18, "initial_atr_stop_mult": 1.9, "trailing_atr_mult": 2.2, "trailing_profit_activation": 0.06, "use_regime_exposure": True},
    },
    {
        "name": "router_sector_bull_only_v2",
        "description": "Bull-only sector RS: strict market gate plus strong sector breadth.",
        "universe": "VN100",
        "tags": ["modern", "router", "bull_only", "sector_rotation"],
        "filters": [
            {"column": "mkt_regime_score", "op": ">=", "value": 62},
            {"column": "vni_above_ma20", "op": "==", "value": True},
            {"column": "vni_above_ma50", "op": "==", "value": True},
            {"column": "mkt_DS20", "op": "<=", "value": 0.50},
            {"column": "data_quality_ok", "op": "==", "value": True},
            {"column": "sector_rs_rank_20d", "op": ">=", "value": 0.72},
            {"column": "sector_breadth_ma20", "op": ">=", "value": 0.60},
            {"column": "sector_leadership_score", "op": ">=", "value": 66},
            {"column": "above_ma50", "op": "==", "value": True},
            {"column": "rs_percentile_20", "op": ">=", "value": 0.60},
            {"column": "smart_money_score_no_sector", "op": ">=", "value": 52},
            {"column": "distribution_days_10", "op": "<=", "value": 2},
        ],
        "rank": [
            {"column": "sector_leadership_score", "ascending": False, "weight": 1.4},
            {"column": "edge_score_no_sector", "ascending": False, "weight": 1.0},
            {"column": "rs_percentile_20", "ascending": False, "weight": 0.8},
        ],
        "risk": {"stop_loss": 0.065, "take_profit": 0.20, "max_holding_bars": 30, "initial_atr_stop_mult": 2.0, "trailing_atr_mult": 2.4, "trailing_profit_activation": 0.07, "use_regime_exposure": True},
    },
]


COMBOS = {
    "combo_pocket_sector": "modern_pocket_pivot_vn_v1,modern_sector_rs_vn_v1",
    "combo_modern_top4": "modern_pocket_pivot_vn_v1,modern_sector_rs_vn_v1,modern_kalman_regime_trend_v1,modern_vol_squeeze_momentum_v1",
    "combo_pocket_family": "modern_pocket_pivot_vn_v1,modern_pocket_pivot_runner_v3,modern_pocket_pivot_quality_v2,modern_canary_pocket_vn_v2",
    "router_regime_v1": "router_ep_stress_v1,router_pocket_recovery_v1,router_sector_normal_v1,router_kalman_bull_v1,router_vol_bull_v1",
    "router_regime_no_ep_v2": "router_pocket_recovery_v1,router_sector_normal_v1,router_kalman_bull_v1,router_vol_bull_v1",
    "router_bull_only_v3": "router_pocket_bull_only_v2,router_sector_bull_only_v2,router_kalman_bull_v1,router_vol_bull_v1",
}


def install_hypotheses() -> None:
    if not CFG_BACKUP.exists():
        shutil.copy(CFG_PATH, CFG_BACKUP)
    cfg = json.loads(CFG_BACKUP.read_text(encoding="utf-8"))
    names = {item["name"] for item in cfg["hypotheses"]}
    for item in [s["hypothesis"] for s in MODERN_STRATEGIES] + ROUTER_STRATEGIES:
        if item["name"] not in names:
            cfg["hypotheses"].append(item)
    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def restore_config() -> None:
    if CFG_BACKUP.exists():
        shutil.copy(CFG_BACKUP, CFG_PATH)


def run_combo(label: str, strategy_names: str, start: str, end: str, max_positions: int, max_candidates: int) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.compare_live_pipeline_sizing_grid",
            "--universe",
            "vn100",
            "--start",
            start,
            "--end",
            end,
            "--sizing-mode",
            "cash_split_orders",
            "--edge-strategy",
            strategy_names,
            "--label",
            label,
            "--max-positions",
            str(max_positions),
            "--max-candidates-per-day",
            str(max_candidates),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"label": label, "strategy": strategy_names, "error": result.stderr[-4000:]}
    files = sorted((ROOT / "backtest_results/mvp_sizing_grid").glob(f"vn100_{label}_*_{start}_{end}_summary.csv"))
    if not files:
        return {"label": label, "strategy": strategy_names, "error": "missing summary"}
    df = pd.read_csv(files[-1])
    eod = df[df["model"] == "eod_next_open"].iloc[0]
    intra = df[df["model"] == "intraday_touch"].iloc[0]
    return {
        "label": label,
        "strategy": strategy_names,
        "eod_return_pct": float(eod["total_return_pct"]),
        "eod_sharpe": float(eod["sharpe_ratio"]),
        "eod_max_dd_pct": float(eod["max_drawdown_pct"]),
        "eod_win_rate_pct": float(eod["win_rate_pct"]),
        "eod_trades": int(eod["number_of_trades"]),
        "eod_signals": int(eod["signals"]),
        "intraday_return_pct": float(intra["total_return_pct"]),
        "intraday_sharpe": float(intra["sharpe_ratio"]),
        "summary_file": str(files[-1]),
    }


def _markdown_table(df: pd.DataFrame) -> str:
    cols = ["label", "eod_return_pct", "eod_sharpe", "eod_max_dd_pct", "eod_win_rate_pct", "eod_trades", "eod_signals"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        values = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.3f}" if "sharpe" in col else f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_outputs(rows: list[dict], start: str, end: str) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["eod_sharpe", "eod_return_pct"], ascending=[False, False], na_position="last")
    csv_path = OUT_DIR / f"modern_strategy_combos_{start}_{end}.csv"
    report_path = REPORTS_DIR / f"modern_strategy_combos_{end}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    lines = [
        f"# Modern Strategy Combo Tests - VN100 ({start} to {end})",
        "",
        "Primary model: `eod_next_open`, long-only VN100, max 5 positions, top 10 candidates/day.",
        "",
        f"CSV: `{csv_path}`",
        "",
        "## Ranked Results",
        "",
        _markdown_table(df),
        "",
        "## Combo Definitions",
        "",
    ]
    for label, strategies in COMBOS.items():
        lines.append(f"- `{label}`: `{strategies}`")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-05-19")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    combos = COMBOS
    if args.only.strip():
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        combos = {key: value for key, value in COMBOS.items() if key in wanted}

    install_hypotheses()
    rows: list[dict] = []
    try:
        for label, strategies in combos.items():
            print(f"[combo] {label}")
            row = run_combo(label, strategies, args.start, args.end, args.max_positions, args.max_candidates_per_day)
            rows.append(row)
            if "error" in row:
                print(f"  ERROR: {row['error'][-300:]}")
            else:
                print(
                    f"  return={row['eod_return_pct']:+.2f}% "
                    f"sharpe={row['eod_sharpe']:.3f} "
                    f"dd={row['eod_max_dd_pct']:+.2f}% "
                    f"trades={row['eod_trades']}"
                )
    finally:
        restore_config()

    csv_path, report_path = write_outputs(rows, args.start, args.end)
    print(csv_path)
    print(report_path)


if __name__ == "__main__":
    main()
