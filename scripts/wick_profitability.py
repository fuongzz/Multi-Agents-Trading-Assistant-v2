"""Profitability analysis — equity curve + CAGR cho các variant wick v2.

Mô phỏng portfolio thực tế:
  - NAV ban đầu: 1,000,000,000 (1 tỷ VND)
  - Mỗi trade: 5% NAV (sizing thực tế của pipeline)
  - Max concurrent positions: 5
  - Chạy theo thứ tự thời gian, skip nếu portfolio đầy
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "backtest_results" / "monthly_wick_research" / "v2"

INITIAL_NAV = 1_000_000_000
POSITION_PCT = 0.05  # 5% NAV per trade
MAX_POSITIONS = 5

VARIANTS = ["U1_BASELINE", "U2_VOL_CONFIRM", "U3_VNI_REGIME", "U4_TIGHT_SL", "U5_STACK_ALL"]


def simulate_portfolio(trades_df: pd.DataFrame) -> dict:
    """Equity curve với portfolio cap + compounding."""
    if trades_df.empty:
        return {"final_nav": INITIAL_NAV, "total_return_pct": 0.0, "cagr": 0.0,
                "n_taken": 0, "n_skipped": 0, "max_drawdown_pct": 0.0}

    trades_df = trades_df.sort_values("entry_date").reset_index(drop=True)
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
    trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])

    nav = INITIAL_NAV
    open_positions = []  # list of (exit_date, exit_pnl_pct, capital_deployed)
    equity_curve = []  # (date, nav)
    n_taken = 0
    n_skipped = 0

    all_dates = pd.date_range(trades_df["entry_date"].min(),
                              trades_df["exit_date"].max(), freq="D")

    trades_iter = iter(trades_df.iterrows())
    next_trade = next(trades_iter, None)

    for d in all_dates:
        # Close exits due today
        still_open = []
        for ex_date, ex_pnl, cap in open_positions:
            if ex_date <= d:
                nav += cap * (ex_pnl / 100)
            else:
                still_open.append((ex_date, ex_pnl, cap))
        open_positions = still_open

        # Open new trades entering today
        while next_trade is not None:
            _, t = next_trade
            if t["entry_date"] > d:
                break
            if t["entry_date"] == d:
                if len(open_positions) < MAX_POSITIONS:
                    cap = nav * POSITION_PCT
                    open_positions.append((t["exit_date"], float(t["pnl_pct"]), cap))
                    n_taken += 1
                else:
                    n_skipped += 1
            next_trade = next(trades_iter, None)

        # Mark-to-market: just count NAV without unrealized (simple approximation)
        equity_curve.append((d, nav))

    # Close any still open
    for _, ex_pnl, cap in open_positions:
        nav += cap * (ex_pnl / 100)

    eq = pd.DataFrame(equity_curve, columns=["date", "nav"])
    eq["dd"] = (eq["nav"] - eq["nav"].cummax()) / eq["nav"].cummax() * 100
    max_dd = float(eq["dd"].min())

    total_ret = (nav - INITIAL_NAV) / INITIAL_NAV * 100
    days = (trades_df["exit_date"].max() - trades_df["entry_date"].min()).days
    years = max(days / 365.25, 0.1)
    cagr = ((nav / INITIAL_NAV) ** (1 / years) - 1) * 100

    # Yearly breakdown
    trades_df["year"] = trades_df["entry_date"].dt.year
    by_year = trades_df.groupby("year").agg(
        n=("pnl_pct", "size"),
        avg_pnl=("pnl_pct", "mean"),
        sum_pnl=("pnl_pct", "sum"),
    )

    return {
        "final_nav": nav,
        "total_return_pct": total_ret,
        "cagr": cagr,
        "n_taken": n_taken,
        "n_skipped": n_skipped,
        "max_drawdown_pct": max_dd,
        "years": years,
        "by_year": by_year,
    }


def main():
    print(f"{'Variant':17s}  {'Final NAV':>14s}  {'TotalRet':>9s}  {'CAGR':>7s}  "
          f"{'MaxDD':>7s}  {'Taken':>6s}  {'Skip':>5s}")
    print("-" * 80)
    results = []
    for v in VARIANTS:
        f = V2_DIR / f"{v}_trades.csv"
        if not f.exists():
            print(f"{v}: no trades file")
            continue
        df = pd.read_csv(f)
        r = simulate_portfolio(df)
        results.append((v, r))
        print(f"{v:17s}  {r['final_nav']:>14,.0f}  "
              f"{r['total_return_pct']:>+8.1f}%  {r['cagr']:>+6.1f}%  "
              f"{r['max_drawdown_pct']:>+6.1f}%  {r['n_taken']:>6d}  {r['n_skipped']:>5d}")

    print("\n--- Lợi nhuận theo năm (sum of pnl_pct per trade) ---")
    for v, r in results:
        print(f"\n{v}:")
        print(r["by_year"].to_string())


if __name__ == "__main__":
    main()
