"""Mức 2 Walk-Forward Audit.

Sử dụng dữ liệu trade 4y có sẵn (chronological, không tune lại params) để:
1. Compute per-fold metrics: expanding window 5 fold
2. Per-quarter + monthly Sharpe distribution
3. Identify failure modes
4. Map problems to proposed remediations

Không tốn compute — chỉ phân tích dữ liệu đã có.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADES = ROOT / "backtest_results/mvp_sizing_grid/vn100_baseline_cash_split_orders_maxpos1_2022-01-01_2026-05-12_top10_eod_next_open_trades.csv"
EQUITY = ROOT / "backtest_results/mvp_sizing_grid/vn100_baseline_cash_split_orders_maxpos1_2022-01-01_2026-05-12_top10_eod_next_open_equity.csv"
VNI = ROOT / "multiagents_trading_assistant/data/index_master.parquet"
OUT = ROOT / "backtest_results/walk_forward_audit"
OUT.mkdir(parents=True, exist_ok=True)


def load():
    t = pd.read_csv(TRADES)
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    e = pd.read_csv(EQUITY)
    e["date"] = pd.to_datetime(e["date"])
    e = e.set_index("date").sort_index()
    v = pd.read_parquet(VNI)
    v = v[v["symbol"] == "VNINDEX"].copy()
    v["date"] = pd.to_datetime(v["date"])
    v = v.set_index("date").sort_index()
    return t, e, v


def metrics(trades, equity, vni=None):
    if len(trades) == 0:
        return None
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    pf = wins["pnl"].sum() / abs(losses["pnl"].sum()) if len(losses) else float("inf")
    wr = len(wins) / len(trades) * 100
    avg = trades["pnl_pct"].mean() * 100
    win_avg = wins["pnl_pct"].mean() * 100 if len(wins) else 0
    loss_avg = losses["pnl_pct"].mean() * 100 if len(losses) else 0
    wl_ratio = abs(win_avg / loss_avg) if loss_avg else float("inf")
    out = {
        "n": len(trades),
        "wr_pct": round(wr, 1),
        "pf": round(pf, 2),
        "avg_pct": round(avg, 2),
        "win_avg": round(win_avg, 2),
        "loss_avg": round(loss_avg, 2),
        "wl_ratio": round(wl_ratio, 2),
    }
    if equity is not None and len(equity) >= 30:
        rets = equity["equity"].pct_change().dropna()
        if rets.std() > 0:
            out["sharpe"] = round(float(rets.mean() / rets.std() * np.sqrt(252)), 2)
        peak = equity["equity"].cummax()
        dd = (equity["equity"] - peak) / peak * 100
        out["max_dd"] = round(float(dd.min()), 2)
        out["total_ret"] = round(float(equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1) * 100, 2)
    if vni is not None and len(vni) >= 30 and "open" in vni.columns:
        v_ret = (vni["close"].iloc[-1] / vni["close"].iloc[0] - 1) * 100
        out["vni_ret"] = round(float(v_ret), 2)
    return out


def walk_forward_expanding(trades, equity, vni):
    """5 expanding-window folds, fixed params throughout."""
    cutoffs = ["2022-12-31", "2023-12-31", "2024-12-31", "2025-12-31", "2026-05-12"]
    rows = []
    for i, end_str in enumerate(cutoffs, 1):
        end = pd.Timestamp(end_str)
        t_sub = trades[trades["entry_date"] <= end]
        e_sub = equity[equity.index <= end]
        v_sub = vni[(vni.index >= pd.Timestamp("2022-01-04")) & (vni.index <= end)]
        m = metrics(t_sub, e_sub, v_sub)
        m["fold"] = f"F{i}"
        m["period"] = f"2022-01 → {end_str}"
        rows.append(m)
    return pd.DataFrame(rows)


def walk_forward_rolling(trades, equity, vni):
    """Rolling 12-month windows — does each year stand alone?"""
    rows = []
    for year in [2022, 2023, 2024, 2025]:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        t_sub = trades[(trades["entry_date"] >= start) & (trades["entry_date"] <= end)]
        e_sub = equity[(equity.index >= start) & (equity.index <= end)]
        v_sub = vni[(vni.index >= start) & (vni.index <= end)]
        m = metrics(t_sub, e_sub, v_sub)
        if m:
            m["year"] = year
            rows.append(m)
    # 2026 partial
    start = pd.Timestamp("2026-01-01")
    end = pd.Timestamp("2026-05-12")
    t_sub = trades[(trades["entry_date"] >= start) & (trades["entry_date"] <= end)]
    e_sub = equity[(equity.index >= start) & (equity.index <= end)]
    v_sub = vni[(vni.index >= start) & (vni.index <= end)]
    m = metrics(t_sub, e_sub, v_sub)
    if m:
        m["year"] = "2026 (5mo)"
        rows.append(m)
    return pd.DataFrame(rows)


def quarterly(trades, equity, vni):
    rows = []
    for year in [2022, 2023, 2024, 2025, 2026]:
        for q in [1, 2, 3, 4]:
            start = pd.Timestamp(f"{year}-{1+(q-1)*3:02d}-01")
            if q == 4:
                end = pd.Timestamp(f"{year+1}-01-01") - pd.Timedelta(days=1)
            else:
                end = pd.Timestamp(f"{year}-{3*q+1:02d}-01") - pd.Timedelta(days=1)
            if year == 2026 and q > 2:
                continue
            t_sub = trades[(trades["entry_date"] >= start) & (trades["entry_date"] <= end)]
            e_sub = equity[(equity.index >= start) & (equity.index <= end)]
            v_sub = vni[(vni.index >= start) & (vni.index <= end)]
            if len(t_sub) == 0:
                continue
            m = metrics(t_sub, e_sub, v_sub)
            if m:
                m["period"] = f"{year}Q{q}"
                rows.append(m)
    return pd.DataFrame(rows)


def monthly_sharpe_distribution(equity):
    """Distribution of monthly returns (annualized → Sharpe per month)."""
    monthly_ret = equity["equity"].resample("ME").last().pct_change().dropna()
    return {
        "n_months": len(monthly_ret),
        "mean_pct": round(float(monthly_ret.mean() * 100), 2),
        "median_pct": round(float(monthly_ret.median() * 100), 2),
        "std_pct": round(float(monthly_ret.std() * 100), 2),
        "best_pct": round(float(monthly_ret.max() * 100), 2),
        "worst_pct": round(float(monthly_ret.min() * 100), 2),
        "n_negative": int((monthly_ret < 0).sum()),
        "pct_negative": round(float((monthly_ret < 0).mean() * 100), 1),
        "max_streak_negative": _max_streak(monthly_ret < 0),
        "annualized_sharpe": round(float(monthly_ret.mean() / monthly_ret.std() * np.sqrt(12)), 2) if monthly_ret.std() > 0 else 0,
    }


def _max_streak(bool_series):
    max_s = 0
    cur = 0
    for v in bool_series:
        if v:
            cur += 1
            max_s = max(max_s, cur)
        else:
            cur = 0
    return max_s


def deterioration_table(wf_expand, wf_roll):
    """OOS deterioration: each rolling year vs full-period train."""
    full_pf = wf_expand.iloc[-1]["pf"]
    rows = []
    for _, r in wf_roll.iterrows():
        ratio = r["pf"] / full_pf if full_pf else 0
        rows.append({
            "year": r["year"],
            "pf": r["pf"],
            "vs_full_pf": round(ratio, 2),
            "verdict": "ROBUST" if ratio > 0.7 else ("MARGINAL" if ratio > 0.5 else "WEAK"),
        })
    return pd.DataFrame(rows)


def main():
    trades, equity, vni = load()

    print("=" * 80)
    print("WALK-FORWARD AUDIT — Mức 2")
    print("=" * 80)

    # Expanding window
    print("\n## Expanding window (cumulative metric khi thêm năm)\n")
    wf_expand = walk_forward_expanding(trades, equity, vni)
    print(wf_expand[["fold", "period", "n", "wr_pct", "pf", "avg_pct", "sharpe", "max_dd", "total_ret", "vni_ret"]].to_string(index=False))

    # Rolling 12-month
    print("\n## Rolling 12-month (mỗi năm độc lập)\n")
    wf_roll = walk_forward_rolling(trades, equity, vni)
    print(wf_roll[["year", "n", "wr_pct", "pf", "avg_pct", "wl_ratio", "sharpe", "max_dd", "total_ret", "vni_ret"]].to_string(index=False))

    # Deterioration
    print("\n## Deterioration: each year vs 4y full PF\n")
    det = deterioration_table(wf_expand, wf_roll)
    print(det.to_string(index=False))

    # Quarterly
    print("\n## Quarterly breakdown (17 quarter)\n")
    q = quarterly(trades, equity, vni)
    print(q[["period", "n", "wr_pct", "pf", "avg_pct", "total_ret"]].to_string(index=False))

    # Monthly Sharpe distribution
    print("\n## Monthly return distribution (proxy stability)\n")
    md = monthly_sharpe_distribution(equity)
    for k, v in md.items():
        print(f"  {k:30s}: {v}")

    # Save
    wf_expand.to_csv(OUT / "expanding_window.csv", index=False)
    wf_roll.to_csv(OUT / "rolling_yearly.csv", index=False)
    det.to_csv(OUT / "deterioration.csv", index=False)
    q.to_csv(OUT / "quarterly.csv", index=False)
    (OUT / "monthly_dist.json").write_text(json.dumps(md, indent=2), encoding="utf-8")

    print(f"\nFiles saved to {OUT}")


if __name__ == "__main__":
    main()
