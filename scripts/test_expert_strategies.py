"""Test 4 expert strategies vs core3 baseline.

Strategy: Minervini Trend Template, CANSLIM Technical, Connors RSI MR, Weinstein Stage 2.

Adds each to hypothesis config (backed up), runs single-strategy backtest, restores.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "multiagents_trading_assistant/edge_lab/configs/vn30_money_smt_hypotheses.json"
CFG_BACKUP = CFG_PATH.with_suffix(".bak2")
OUT_DIR = ROOT / "backtest_results/expert_strategies"
OUT_DIR.mkdir(parents=True, exist_ok=True)


EXPERT_STRATEGIES_ALL = [
    {
        "name": "minervini_trend_template_v1",
        "description": "Minervini stage-2 leader: stacked MAs up + RS>=70 + accumulation days",
        "universe": "VN100",
        "tags": ["minervini", "trend_template", "leader"],
        "filters": [
            {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
            {"column": "data_quality_ok", "op": "==", "value": True},
            {"column": "above_ma20", "op": "==", "value": True},
            {"column": "above_ma50", "op": "==", "value": True},
            {"column": "ma20_slope_5", "op": ">=", "value": 0},
            {"column": "ma50_slope_10", "op": ">=", "value": 0.003},
            {"column": "distance_ma50", "op": "between", "value": [0.0, 0.30]},
            {"column": "rs_percentile_20", "op": ">=", "value": 0.70},
            {"column": "excess_ret_60d_pctile", "op": ">=", "value": 0.60},
            {"column": "accumulation_days_10", "op": ">=", "value": 2},
            {"column": "distribution_days_10", "op": "<=", "value": 2},
        ],
        "rank": [
            {"column": "rs_percentile_20", "ascending": False, "weight": 1.5},
            {"column": "smart_money_score_no_sector", "ascending": False, "weight": 1.0},
            {"column": "ma50_slope_10", "ascending": False, "weight": 0.5},
        ],
        "risk": {
            "stop_loss": 0.08, "take_profit": 0.30, "max_holding_bars": 60,
            "initial_atr_stop_mult": 2.5, "trailing_atr_mult": 3.0,
            "trailing_profit_activation": 0.10, "use_regime_exposure": True,
        },
    },
    {
        "name": "canslim_technical_v1",
        "description": "O'Neil CANSLIM technicals: M+L+N+S+I (new 55-day high + extreme volume + leader)",
        "universe": "VN100",
        "tags": ["canslim", "oneil", "breakout", "leader"],
        "filters": [
            {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
            {"column": "mkt_regime_score", "op": ">=", "value": 45},
            {"column": "data_quality_ok", "op": "==", "value": True},
            {"column": "breakout_55", "op": "==", "value": True},
            {"column": "volume_ratio_20", "op": ">=", "value": 1.8},
            {"column": "value_ratio_20", "op": ">=", "value": 1.5},
            {"column": "rs_percentile_20", "op": ">=", "value": 0.75},
            {"column": "excess_ret_60d_pctile", "op": ">=", "value": 0.70},
            {"column": "smart_money_score_no_sector", "op": ">=", "value": 60},
            {"column": "above_ma50", "op": "==", "value": True},
        ],
        "rank": [
            {"column": "smart_money_score_no_sector", "ascending": False, "weight": 1.5},
            {"column": "volume_ratio_20", "ascending": False, "weight": 1.0},
            {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
        ],
        "risk": {
            "stop_loss": 0.08, "take_profit": 0.30, "max_holding_bars": 45,
            "initial_atr_stop_mult": 2.5, "trailing_atr_mult": 2.8,
            "trailing_profit_activation": 0.08, "use_regime_exposure": True,
        },
    },
    {
        "name": "connors_rsi_pullback_v1",
        "description": "Connors-style oversold RSI14 in long-term uptrend, short bounce play (relaxed for VN)",
        "universe": "VN100",
        "tags": ["connors", "mean_reversion", "oversold", "short_hold"],
        "filters": [
            {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
            {"column": "data_quality_ok", "op": "==", "value": True},
            {"column": "above_ma50", "op": "==", "value": True},
            {"column": "rsi14", "op": "<=", "value": 42},
            {"column": "distance_ma20", "op": "between", "value": [-0.12, 0.0]},
            {"column": "smart_money_score_no_sector", "op": ">=", "value": 45},
            {"column": "distribution_days_10", "op": "<=", "value": 3},
        ],
        "rank": [
            {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
            {"column": "rsi14", "ascending": True, "weight": 1.0},
            {"column": "smart_money_score_no_sector", "ascending": False, "weight": 0.5},
        ],
        "risk": {
            "stop_loss": 0.06, "take_profit": 0.12, "max_holding_bars": 10,
            "initial_atr_stop_mult": 2.0, "trailing_atr_mult": 2.0,
            "trailing_profit_activation": 0.04, "use_regime_exposure": True,
        },
    },
    {
        "name": "weinstein_stage2_breakout_v1",
        "description": "Stan Weinstein stage 2 transition: trend up + base breakout + volume",
        "universe": "VN100",
        "tags": ["weinstein", "stage2", "breakout", "ma200"],
        "filters": [
            {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
            {"column": "data_quality_ok", "op": "==", "value": True},
            {"column": "above_ma50", "op": "==", "value": True},
            {"column": "ma50_slope_10", "op": ">=", "value": 0.003},
            {"column": "breakout_20", "op": "==", "value": True},
            {"column": "volume_ratio_20", "op": ">=", "value": 1.5},
            {"column": "distance_ma50", "op": "between", "value": [0.01, 0.25]},
            {"column": "smart_money_score_no_sector", "op": ">=", "value": 55},
            {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
        ],
        "rank": [
            {"column": "breakout_20_quality", "ascending": False, "weight": 1.5},
            {"column": "smart_money_score_no_sector", "ascending": False, "weight": 1.0},
            {"column": "volume_ratio_20", "ascending": False, "weight": 0.5},
        ],
        "risk": {
            "stop_loss": 0.08, "take_profit": 0.30, "max_holding_bars": 50,
            "initial_atr_stop_mult": 2.5, "trailing_atr_mult": 3.0,
            "trailing_profit_activation": 0.10, "use_regime_exposure": True,
        },
    },
]


def install_strategies(strategies):
    """Add experimental strategies to config, backing up first."""
    if not CFG_BACKUP.exists():
        shutil.copy(CFG_PATH, CFG_BACKUP)
        print(f"[backup] {CFG_PATH.name} saved")
    cfg = json.loads(CFG_BACKUP.read_text(encoding="utf-8"))
    existing_names = {h["name"] for h in cfg["hypotheses"]}
    for s in strategies:
        if s["name"] not in existing_names:
            cfg["hypotheses"].append(s)
    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[install] added {len(strategies)} strategies to config")


def restore_config():
    if CFG_BACKUP.exists():
        shutil.copy(CFG_BACKUP, CFG_PATH)
        print(f"[restore] original config restored")


def run_backtest(name, start, end):
    label = name.replace("_", "")[:30]
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.compare_live_pipeline_sizing_grid",
            "--universe", "vn100",
            "--start", start, "--end", end,
            "--sizing-mode", "cash_split_orders",
            "--edge-strategy", name,
            "--label", label,
            "--max-positions", "5",
            "--max-candidates-per-day", "10",
        ],
        capture_output=True, text=True, cwd=ROOT,
    )
    summary_files = list((ROOT / "backtest_results/mvp_sizing_grid").glob(f"vn100_{label}_*_summary.csv"))
    if not summary_files:
        return None
    summary = pd.read_csv(summary_files[-1])
    eod = summary[summary["model"] == "eod_next_open"].iloc[0]
    return {
        "name": name,
        "return_pct": round(float(eod["total_return_pct"]), 2),
        "sharpe": round(float(eod["sharpe_ratio"]), 2),
        "max_dd": round(float(eod["max_drawdown_pct"]), 2),
        "win_rate": round(float(eod["win_rate_pct"]), 1),
        "n_trades": int(eod["number_of_trades"]),
        "signals": int(eod["signals"]),
        "fill_rate": round(float(eod["fill_rate_pct"]), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-05-12")
    parser.add_argument("--only", default="", help="comma-separated strategy names to run (default: all)")
    args = parser.parse_args()

    if args.only:
        names = set(args.only.split(","))
        EXPERT_STRATEGIES = [s for s in EXPERT_STRATEGIES_ALL if s["name"] in names]
    else:
        EXPERT_STRATEGIES = EXPERT_STRATEGIES_ALL

    install_strategies(EXPERT_STRATEGIES)

    rows = []
    try:
        for s in EXPERT_STRATEGIES:
            print(f"\n{'=' * 70}\n[backtest] {s['name']}\n{'=' * 70}")
            m = run_backtest(s["name"], args.start, args.end)
            if m:
                pf_calc = "N/A"
                rows.append({"label": s["name"], **m, "tags": ",".join(s["tags"])})
                print(f"  return={m['return_pct']:+.2f}%  Sharpe={m['sharpe']:.2f}  DD={m['max_dd']:+.2f}%  WR={m['win_rate']}%  n={m['n_trades']}  signals={m['signals']}  fill={m['fill_rate']}%")
            else:
                print(f"  FAILED to read result")
    finally:
        restore_config()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"expert_strategies_{args.start}_{args.end}.csv", index=False)

    print("\n" + "=" * 90)
    print(f"{'Strategy':<35s} {'Return':>8s} {'Sharpe':>7s} {'DD':>7s} {'WR':>6s} {'n':>5s} {'Signals':>8s}")
    print("-" * 90)
    for r in rows:
        print(f"{r['label']:<35s} {r['return_pct']:>+7.2f}% {r['sharpe']:>6.2f} {r['max_dd']:>+6.2f}% {r['win_rate']:>5}% {r['n_trades']:>5d} {r['signals']:>8d}")
    print()
    print(f"Saved: {OUT_DIR}/expert_strategies_{args.start}_{args.end}.csv")


if __name__ == "__main__":
    main()
