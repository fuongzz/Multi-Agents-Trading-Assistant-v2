"""Test Option C: replace compression_breakout_smt_v1 in core3 with reclaim_ma20_quality_v1__best_combo.

Compares:
- Baseline core3 (BAA + COMPRESS + MR50)
- Option C: core3' (BAA + RECLAIM_BEST_COMBO + MR50)
- Option B: core3 + RECLAIM_BEST_COMBO (4 strategies, same max_pos=5)
- Option A: RECLAIM_BEST_COMBO alone (already known)

Outputs comparison on YTD 2025 + 4y.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import glob
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "multiagents_trading_assistant/edge_lab/configs/vn30_money_smt_hypotheses.json"
CFG_BACKUP = CFG_PATH.with_suffix(".bak4")
OUT_DIR = ROOT / "backtest_results/option_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORE3_ORIG = [
    "breakout_after_accumulation_v3",
    "compression_breakout_smt_v1",
    "mean_reversion_uptrend_ma50_v1",
]

BEST_COMBO_NAME = "reclaim_ma20_quality_v1__best_combo"


def install_best_combo():
    """Add the tuned reclaim_ma20_quality_v1__best_combo to the config."""
    if not CFG_BACKUP.exists():
        shutil.copy(CFG_PATH, CFG_BACKUP)
    cfg = json.loads(CFG_BACKUP.read_text(encoding="utf-8"))
    base = next(h for h in cfg["hypotheses"] if h["name"] == "reclaim_ma20_quality_v1")
    new_h = deepcopy(base)
    new_h["name"] = BEST_COMBO_NAME
    # Apply best_combo patch: mkt_regime_score>=55, smart_money_score>=55
    for f in new_h["filters"]:
        if f.get("column") == "mkt_regime_score":
            f["value"] = 55
        elif f.get("column") == "smart_money_score":
            f["value"] = 55
    cfg["hypotheses"].append(new_h)
    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def restore_config():
    if CFG_BACKUP.exists():
        shutil.copy(CFG_BACKUP, CFG_PATH)


def run_backtest(strategy_list, label, start, end):
    edge_str = ",".join(strategy_list)
    subprocess.run(
        [
            sys.executable, "-m", "scripts.compare_live_pipeline_sizing_grid",
            "--universe", "vn100",
            "--start", start, "--end", end,
            "--sizing-mode", "cash_split_orders",
            "--edge-strategy", edge_str,
            "--label", label,
            "--max-positions", "5",
            "--max-candidates-per-day", "10",
        ],
        capture_output=True, text=True, cwd=ROOT,
    )
    trade_files = glob.glob(f"backtest_results/mvp_sizing_grid/vn100_{label}_*{start}_{end}_top10_eod_next_open_trades.csv")
    eq_files = glob.glob(f"backtest_results/mvp_sizing_grid/vn100_{label}_*{start}_{end}_top10_eod_next_open_equity.csv")
    if not trade_files:
        return None
    df = pd.read_csv(trade_files[-1])
    if len(df) == 0:
        return {"label": label, "n_trades": 0, "return_pct": 0, "pf": None, "sharpe": 0, "max_dd": 0, "win_rate": 0}
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
        "label": label,
        "n_trades": len(df),
        "return_pct": round(ret, 2) if ret is not None else None,
        "pf": round(float(pf), 2),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "max_dd": round(max_dd, 2) if max_dd is not None else None,
        "win_rate": round(wr, 1),
    }


def main():
    install_best_combo()
    try:
        configs = [
            ("baseline_core3",   CORE3_ORIG),
            ("optionC_swap_compr", ["breakout_after_accumulation_v3", BEST_COMBO_NAME, "mean_reversion_uptrend_ma50_v1"]),
            ("optionB_combo4",   CORE3_ORIG + [BEST_COMBO_NAME]),
            ("optionA_alone",    [BEST_COMBO_NAME]),
        ]
        all_rows = []
        for period_label, start, end in [("YTD", "2025-01-01", "2026-05-12"), ("4Y", "2022-01-01", "2026-05-12")]:
            for name, strategies in configs:
                full_label = f"{name}_{period_label}"
                print(f"\n[{period_label}] {name}: {len(strategies)} strategies")
                m = run_backtest(strategies, full_label, start, end)
                if m:
                    m["period"] = period_label
                    m["config"] = name
                    all_rows.append(m)
                    print(f"  return={m['return_pct']}% PF={m['pf']} Sharpe={m['sharpe']} DD={m['max_dd']}% n={m['n_trades']} WR={m['win_rate']}%")
    finally:
        restore_config()

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "option_c_compare.csv", index=False)
    print("\n\n=== SUMMARY ===")
    for period in ["YTD", "4Y"]:
        sub = df[df["period"] == period]
        if len(sub) == 0: continue
        print(f"\n--- {period} ---")
        print(sub[["config", "return_pct", "pf", "sharpe", "max_dd", "n_trades", "win_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
