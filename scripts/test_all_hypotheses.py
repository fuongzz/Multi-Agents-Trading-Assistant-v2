"""Mass-test all hypotheses in vn30_money_smt_hypotheses.json standalone.

Iterates every hypothesis name → runs single-strategy backtest with baseline config.
Saves results incrementally so progress isn't lost on interrupt.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "multiagents_trading_assistant/edge_lab/configs/vn30_money_smt_hypotheses.json"
OUT_DIR = ROOT / "backtest_results/all_hypotheses_mass_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_hypotheses():
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    return [(h["name"], h.get("universe", "VN100"), h.get("tags", [])) for h in cfg["hypotheses"]]


def run_backtest(name, start, end):
    label = name.replace("_", "")[:35]
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
    summary_files = list((ROOT / "backtest_results/mvp_sizing_grid").glob(f"vn100_{label}_*_summary.csv"))
    if not summary_files:
        return None
    summary = pd.read_csv(summary_files[-1])
    eod = summary[summary["model"] == "eod_next_open"].iloc[0]
    # Compute PF from trades
    trades_files = list((ROOT / "backtest_results/mvp_sizing_grid").glob(f"vn100_{label}_*top10_eod_next_open_trades.csv"))
    pf = None
    avg_pct = None
    if trades_files:
        try:
            tdf = pd.read_csv(trades_files[-1])
            wins = tdf[tdf["pnl"] > 0]
            losses = tdf[tdf["pnl"] < 0]
            if len(losses) > 0:
                pf = wins["pnl"].sum() / abs(losses["pnl"].sum())
            else:
                pf = float("inf") if len(wins) > 0 else 0
            avg_pct = tdf["pnl_pct"].mean() * 100
        except Exception:
            pass
    return {
        "name": name,
        "return_pct": round(float(eod["total_return_pct"]), 2),
        "sharpe": round(float(eod["sharpe_ratio"]), 2),
        "max_dd": round(float(eod["max_drawdown_pct"]), 2),
        "win_rate": round(float(eod["win_rate_pct"]), 1),
        "n_trades": int(eod["number_of_trades"]),
        "signals": int(eod["signals"]),
        "fill_rate": round(float(eod["fill_rate_pct"]), 1),
        "pf": round(float(pf), 2) if pf is not None and pf != float("inf") else (None if pf is None else 999),
        "avg_pct": round(avg_pct, 2) if avg_pct is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-12")
    parser.add_argument("--label", default="ytd2025")
    parser.add_argument("--skip", default="", help="Comma-separated names to skip")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of hypotheses (0 = all)")
    args = parser.parse_args()

    hyps = list_hypotheses()
    skip_set = set(args.skip.split(",")) if args.skip else set()
    hyps = [(n, u, t) for n, u, t in hyps if n not in skip_set]
    if args.limit > 0:
        hyps = hyps[: args.limit]

    print(f"Mass-test {len(hyps)} hypotheses on {args.start} -> {args.end}")
    out_csv = OUT_DIR / f"mass_test_{args.label}.csv"
    rows = []
    # Resume support: if CSV exists, skip already-done ones
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        done = set(existing["name"])
        rows = existing.to_dict("records")
        hyps = [(n, u, t) for n, u, t in hyps if n not in done]
        print(f"  Resume: {len(done)} already done, {len(hyps)} remaining")

    for i, (name, univ, tags) in enumerate(hyps, 1):
        print(f"\n[{i}/{len(hyps)}] {name} ({univ})")
        try:
            m = run_backtest(name, args.start, args.end)
        except Exception as e:
            print(f"  ERR: {e}")
            m = None
        if m is None:
            m = {"name": name, "return_pct": None, "sharpe": None, "max_dd": None,
                 "win_rate": None, "n_trades": 0, "signals": 0, "fill_rate": 0, "pf": None, "avg_pct": None}
        m["universe"] = univ
        m["tags"] = ",".join(tags)
        rows.append(m)
        if m["return_pct"] is not None:
            print(f"  return={m['return_pct']:+.2f}% Sharpe={m['sharpe']:.2f} PF={m['pf']} DD={m['max_dd']:+.2f}% n={m['n_trades']}")
        else:
            print(f"  no result")
        # Save incrementally
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    # Sort by PF then Sharpe
    df["sort_key"] = df["pf"].fillna(0) * df["sharpe"].fillna(-99).clip(lower=0)
    df_sorted = df.sort_values("sort_key", ascending=False)
    df_sorted.to_csv(OUT_DIR / f"mass_test_{args.label}_sorted.csv", index=False)
    print(f"\n\n=== TOP 10 by PF×Sharpe ===")
    show_cols = ["name", "return_pct", "sharpe", "pf", "max_dd", "win_rate", "n_trades", "signals"]
    print(df_sorted[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
