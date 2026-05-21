"""Fine-tune a target hypothesis with parameter variations.

Generates N variants of a base hypothesis with shifted thresholds, runs
backtest for each, compares metrics. Restores config when done.
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
import numpy as np
import glob

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "multiagents_trading_assistant/edge_lab/configs/vn30_money_smt_hypotheses.json"
CFG_BACKUP = CFG_PATH.with_suffix(".bak3")
OUT_DIR = ROOT / "backtest_results/finetune"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Variants per target hypothesis (column → new value)
VARIANTS = {
    "reclaim_ma20_quality_v1": [
        # Combo winners from previous fine-tune
        {"label": "best_combo", "patch": {"mkt_regime_score": 55, "smart_money_score": 55}},
        {"label": "best_combo_tight_sl", "patch": {"mkt_regime_score": 55, "smart_money_score": 55}, "patch_risk": {"stop_loss": 0.06}},
        {"label": "best_combo_loose_smt55_rs55", "patch": {"smart_money_score": 55, "rs_percentile_20": 0.55}},
        {"label": "tight_regime", "patch": {"mkt_regime_score": 55}},
        {"label": "loose_regime", "patch": {"mkt_regime_score": 45}},
        {"label": "tight_smt", "patch": {"smart_money_score": 62}},
        {"label": "loose_smt", "patch": {"smart_money_score": 55}},
        {"label": "tight_rs", "patch": {"rs_percentile_20": 0.62}},
        {"label": "loose_rs", "patch": {"rs_percentile_20": 0.50}},
        {"label": "tight_close_loc", "patch": {"close_location": 0.72}},
        {"label": "loose_close_loc", "patch": {"close_location": 0.55}},
        {"label": "tight_sl", "patch_risk": {"stop_loss": 0.06}},
        {"label": "loose_tp", "patch_risk": {"take_profit": 0.28}},
        {"label": "longer_hold", "patch_risk": {"max_holding_bars": 40}},
        {"label": "shorter_hold", "patch_risk": {"max_holding_bars": 20}},
    ],
    "leader_pullback_market_regime_v3": [
        {"label": "tight_regime", "patch": {"mkt_regime_score": 60}},
        {"label": "loose_regime", "patch": {"mkt_regime_score": 50}},
        {"label": "tight_smt", "patch": {"smart_money_score": 65}},
        {"label": "loose_smt", "patch": {"smart_money_score": 58}},
        {"label": "tight_rs", "patch": {"rs_percentile_20": 0.68}},
        {"label": "loose_rs", "patch": {"rs_percentile_20": 0.55}},
    ],
    "breakout_55_smt_v1": [
        {"label": "tight_smt", "patch": {"smart_money_score": 65}},
        {"label": "loose_smt", "patch": {"smart_money_score": 58}},
        {"label": "tight_vol", "patch": {"value_ratio_20": 1.30}},
        {"label": "loose_vol", "patch": {"value_ratio_20": 1.05}},
        {"label": "tight_rs", "patch": {"rs_percentile_20": 0.68}},
        {"label": "tight_sl", "patch_risk": {"stop_loss": 0.07}},
    ],
}


def install_variant(base_name: str, variant: dict):
    """Create modified hypothesis, add to config with new name."""
    cfg = json.loads(CFG_BACKUP.read_text(encoding="utf-8"))
    base = next(h for h in cfg["hypotheses"] if h["name"] == base_name)
    new_h = deepcopy(base)
    new_name = f"{base_name}__{variant['label']}"
    new_h["name"] = new_name
    # Patch filter values
    patch = variant.get("patch", {})
    for col, val in patch.items():
        for f in new_h["filters"]:
            if f.get("column") == col and "value" in f:
                if isinstance(f["value"], (int, float)):
                    f["value"] = val
                break
    # Patch risk
    patch_risk = variant.get("patch_risk", {})
    for k, v in patch_risk.items():
        new_h["risk"][k] = v
    cfg["hypotheses"].append(new_h)
    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return new_name


def run_backtest(name, start, end):
    label = name.replace("_", "")[:40]
    subprocess.run(
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
    # Compute true metrics from trades + equity
    trade_files = glob.glob(f"backtest_results/mvp_sizing_grid/vn100_{label}_*{start}_{end}_top10_eod_next_open_trades.csv")
    eq_files = glob.glob(f"backtest_results/mvp_sizing_grid/vn100_{label}_*{start}_{end}_top10_eod_next_open_equity.csv")
    if not trade_files:
        return None
    df = pd.read_csv(trade_files[-1])
    if len(df) == 0:
        return {"n_trades": 0, "return_pct": 0, "pf": None, "sharpe": 0, "max_dd": 0, "win_rate": 0}
    wins = df[df["pnl"] > 0]; losses = df[df["pnl"] < 0]
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) > 0 else 99
    wr = (df["pnl"] > 0).mean() * 100
    sharpe = max_dd = ret = None
    if eq_files:
        try:
            eq = pd.read_csv(eq_files[-1])
            eq["date"] = pd.to_datetime(eq["date"])
            eq = eq.set_index("date").sort_index()
            rets = eq["equity"].pct_change().dropna()
            if rets.std() > 0:
                sharpe = float(rets.mean() / rets.std() * np.sqrt(252))
            peak = eq["equity"].cummax()
            dd = (eq["equity"] - peak) / peak * 100
            max_dd = float(dd.min())
            ret = float(eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1) * 100
        except Exception:
            pass
    return {
        "n_trades": len(df),
        "return_pct": round(ret, 2) if ret is not None else None,
        "pf": round(float(pf), 2),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "max_dd": round(max_dd, 2) if max_dd is not None else None,
        "win_rate": round(wr, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="base hypothesis name")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-05-12")
    parser.add_argument("--only", default="", help="comma-separated variant labels to run (default: all)")
    args = parser.parse_args()

    if not CFG_BACKUP.exists():
        shutil.copy(CFG_PATH, CFG_BACKUP)
    if args.target not in VARIANTS:
        print(f"ERR: no variants defined for {args.target}")
        return 1

    variants = VARIANTS[args.target]
    if args.only:
        only = set(args.only.split(","))
        variants = [v for v in variants if v["label"] in only]
    print(f"Fine-tuning {args.target}: {len(variants)} variants on {args.start}->{args.end}")

    rows = []
    try:
        # Baseline (original)
        print(f"\n[baseline] {args.target}")
        m = run_backtest(args.target, args.start, args.end)
        if m:
            rows.append({"variant": "ORIGINAL", **m})
            print(f"  PF={m['pf']} Sharpe={m['sharpe']} DD={m['max_dd']} Return={m['return_pct']} n={m['n_trades']}")

        for v in variants:
            shutil.copy(CFG_BACKUP, CFG_PATH)  # restore before patching
            new_name = install_variant(args.target, v)
            print(f"\n[variant] {v['label']}")
            m = run_backtest(new_name, args.start, args.end)
            if m:
                rows.append({"variant": v["label"], **m})
                print(f"  PF={m['pf']} Sharpe={m['sharpe']} DD={m['max_dd']} Return={m['return_pct']} n={m['n_trades']}")
    finally:
        shutil.copy(CFG_BACKUP, CFG_PATH)
        print(f"\n[restore] original config restored")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / f"finetune_{args.target}.csv", index=False)
    out_sorted = out.copy()
    out_sorted["score"] = out_sorted["pf"].fillna(0) * out_sorted["sharpe"].fillna(0)
    out_sorted = out_sorted.sort_values("score", ascending=False)

    print(f"\n\n=== Fine-tune results: {args.target} ===")
    print(out_sorted.to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main() or 0)
