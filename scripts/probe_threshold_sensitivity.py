"""Probe A: shift filter thresholds in core3 hypotheses ±10%/20%, run backtest.

Output: how sensitive baseline +131% is to each threshold value.
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
CFG_BACKUP = CFG_PATH.with_suffix(".bak")
OUT_DIR = ROOT / "backtest_results/probe_threshold_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORE3 = [
    "breakout_after_accumulation_v3",
    "compression_breakout_smt_v1",
    "mean_reversion_uptrend_ma50_v1",
]

# Thresholds to perturb. (column_name, scale_factor) — scale 0.9, 1.1 = ±10%
TARGETS = [
    ("mkt_regime_score", "min_score"),
    ("smart_money_score_no_sector", "smt"),
    ("smart_money_score", "smt"),
    ("rs_percentile_20", "rs"),
    ("CHDM50", "chdm"),
    ("DS20", "ds"),
    ("value_ratio_20", "vol"),
]
SHIFTS = [0.8, 0.9, 1.1, 1.2]  # ±10%, ±20%


def find_threshold(filt, col):
    return filt.get("column") == col and filt.get("op") in (">=", "<=", ">", "<")


def shift_value(val, op, factor):
    """Shift threshold value by factor. For >= and >: scale value. For <= <: inverse."""
    if op in (">=", ">"):
        return round(val * factor, 4)
    if op in ("<=", "<"):
        # Tightening for <=: smaller factor. Inverse logic.
        return round(val * (2 - factor), 4)  # 0.8 -> 1.2x tighter, 1.2 -> 0.8x looser
    return val


def make_shifted_config(col_target, factor, label):
    """Clone full JSON, shift `col_target` threshold by factor across core3 hypotheses."""
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    new_cfg = deepcopy(cfg)
    n_changed = 0
    for h in new_cfg["hypotheses"]:
        if h["name"] not in CORE3:
            continue
        for filt in h["filters"]:
            if filt.get("column") == col_target and filt.get("op") in (">=", "<=", ">", "<"):
                orig = filt["value"]
                if isinstance(orig, (int, float)):
                    filt["value"] = shift_value(orig, filt["op"], factor)
                    n_changed += 1
    out_path = OUT_DIR / f"cfg_{label}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, indent=2, ensure_ascii=False)
    return out_path, n_changed


def run_backtest(label, start, end):
    """Run the existing sizing grid script with this config installed."""
    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.compare_live_pipeline_sizing_grid",
            "--universe", "vn100",
            "--start", start,
            "--end", end,
            "--sizing-mode", "cash_split_orders",
            "--label", label,
            "--max-positions", "5",
            "--max-candidates-per-day", "10",
        ],
        capture_output=True, text=True, cwd=ROOT,
    )
    # Find the summary CSV (script writes to mvp_sizing_grid)
    summary_files = list((ROOT / "backtest_results/mvp_sizing_grid").glob(f"vn100_{label}_*_summary.csv"))
    if not summary_files:
        return None
    summary = pd.read_csv(summary_files[-1])
    eod = summary[summary["model"] == "eod_next_open"].iloc[0]
    return {
        "return_pct": eod["total_return_pct"],
        "sharpe": eod["sharpe_ratio"],
        "max_dd_pct": eod["max_drawdown_pct"],
        "win_rate_pct": eod["win_rate_pct"],
        "n_trades": int(eod["number_of_trades"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-05-12")
    parser.add_argument("--shifts", default="0.8,0.9,1.0,1.1,1.2")
    args = parser.parse_args()
    shifts = [float(s) for s in args.shifts.split(",")]

    # Backup original config
    if not CFG_BACKUP.exists():
        shutil.copy(CFG_PATH, CFG_BACKUP)
        print(f"[backup] {CFG_PATH} -> {CFG_BACKUP}")

    rows = []
    try:
        # Baseline (factor 1.0)
        print("[probe] BASELINE (factor 1.0)")
        # Restore original
        shutil.copy(CFG_BACKUP, CFG_PATH)
        m = run_backtest("baseline_check2", args.start, args.end)
        if m:
            rows.append({"col": "baseline", "factor": 1.0, "n_changed": 0, **m})
            print(f"  -> return={m['return_pct']:+.2f}% Sharpe={m['sharpe']:.2f} n={m['n_trades']}")

        # Iterate threshold × factor
        for col, short in TARGETS:
            for f in shifts:
                if abs(f - 1.0) < 0.01:
                    continue
                label_short = f"shift_{short}_{int(f*100)}"
                cfg_path, n_changed = make_shifted_config(col, f, label_short)
                if n_changed == 0:
                    print(f"[probe] {col} factor {f}: no matching filters -> skip")
                    continue
                shutil.copy(cfg_path, CFG_PATH)
                print(f"[probe] {col} × {f} ({n_changed} filters)")
                m = run_backtest(label_short, args.start, args.end)
                if m:
                    rows.append({"col": col, "factor": f, "n_changed": n_changed, **m})
                    print(f"  -> return={m['return_pct']:+.2f}% Sharpe={m['sharpe']:.2f} n={m['n_trades']}")
                else:
                    rows.append({"col": col, "factor": f, "n_changed": n_changed,
                                 "return_pct": None, "sharpe": None})
                    print(f"  -> backtest FAILED")
    finally:
        # Always restore original config
        shutil.copy(CFG_BACKUP, CFG_PATH)
        print(f"[restore] Original config restored")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "sensitivity_summary.csv"
    df.to_csv(out_csv, index=False)

    # Pretty print
    print("\n" + "=" * 90)
    print(f"{'Threshold':<35s} {'Factor':>7s} {'Return%':>9s} {'Sharpe':>7s} {'DD%':>7s} {'Trades':>7s}")
    print("-" * 90)
    base_ret = df[df["col"] == "baseline"]["return_pct"].iloc[0] if (df["col"] == "baseline").any() else None
    for _, r in df.iterrows():
        if r["return_pct"] is None:
            continue
        delta = f"  ({r['return_pct']-base_ret:+.1f}pp)" if base_ret and r["col"] != "baseline" else ""
        print(f"{r['col']:<35s} {r['factor']:>7.2f} {r['return_pct']:>+8.2f}%{delta}  {r['sharpe']:>6.2f} {r['max_dd_pct']:>+6.2f}% {int(r['n_trades']):>6d}")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
