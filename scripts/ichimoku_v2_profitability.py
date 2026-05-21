"""Profitability sim for Ichimoku v2 tunings."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "backtest_results" / "ichimoku_research" / "v2"
INITIAL_NAV = 1_000_000_000
POSITION_PCT = 0.05
MAX_POSITIONS = 5

TUNES = ["T1_BASELINE", "T2_TIGHT_SL_3PCT", "T3_VNI_UPTREND", "T4_NO_CHASE",
         "T5_CLOUD_THICK", "T6_EXIT_CONFIRM", "T7_PARAM_VN", "T8_STACK"]


def simulate(trades_df):
    if trades_df.empty:
        return {"final_nav": INITIAL_NAV, "total_ret": 0, "cagr": 0,
                "n_taken": 0, "n_skipped": 0, "max_dd": 0}
    trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])

    nav = INITIAL_NAV
    open_pos = []
    n_taken = n_skipped = 0
    equity = []

    dates = pd.date_range(trades_df["entry_date"].min(),
                          trades_df["exit_date"].max(), freq="D")
    trades_iter = iter(trades_df.iterrows())
    nxt = next(trades_iter, None)

    for d in dates:
        still = []
        for ex_d, ex_p, cap in open_pos:
            if ex_d <= d:
                nav += cap * (ex_p / 100)
            else:
                still.append((ex_d, ex_p, cap))
        open_pos = still

        while nxt is not None:
            _, t = nxt
            if t["entry_date"] > d:
                break
            if t["entry_date"] == d:
                if len(open_pos) < MAX_POSITIONS:
                    cap = nav * POSITION_PCT
                    open_pos.append((t["exit_date"], float(t["pnl_pct"]), cap))
                    n_taken += 1
                else:
                    n_skipped += 1
            nxt = next(trades_iter, None)
        equity.append((d, nav))

    for _, ex_p, cap in open_pos:
        nav += cap * (ex_p / 100)

    eq = pd.DataFrame(equity, columns=["date", "nav"])
    eq["dd"] = (eq["nav"] - eq["nav"].cummax()) / eq["nav"].cummax() * 100
    days = (trades_df["exit_date"].max() - trades_df["entry_date"].min()).days
    years = max(days / 365.25, 0.1)
    cagr = ((nav / INITIAL_NAV) ** (1 / years) - 1) * 100

    return {"final_nav": nav,
            "total_ret": (nav - INITIAL_NAV) / INITIAL_NAV * 100,
            "cagr": cagr, "max_dd": float(eq["dd"].min()),
            "n_taken": n_taken, "n_skipped": n_skipped}


def main():
    print(f"{'Tune':18s}  {'Final NAV':>14s}  {'TotRet':>8s}  {'CAGR':>7s}  "
          f"{'MaxDD':>7s}  {'Taken':>5s}  {'Skip':>5s}")
    print("-" * 80)
    for v in TUNES:
        f = V2_DIR / f"{v}_trades.csv"
        if not f.exists():
            print(f"{v}: missing")
            continue
        df = pd.read_csv(f)
        r = simulate(df)
        print(f"{v:18s}  {r['final_nav']:>14,.0f}  "
              f"{r['total_ret']:>+7.1f}%  {r['cagr']:>+6.1f}%  "
              f"{r['max_dd']:>+6.1f}%  {r['n_taken']:>5d}  {r['n_skipped']:>5d}")


if __name__ == "__main__":
    main()
