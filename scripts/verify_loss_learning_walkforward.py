"""Rolling walk-forward: each year, learn deprecate-list from prior years' losses,
apply to current year, compute full metrics. Then aggregate 2024-2026 cumulatively.

Run: PYTHONIOENCODING=utf-8 python scripts/verify_loss_learning_walkforward.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from multiagents_trading_assistant.agentic.post_trade_review import review_trades_df


INITIAL_NAV = 1_000_000_000  # 1B VND


def load_all_trades() -> pd.DataFrame:
    files = glob.glob("backtest_results/all_strategies_unbiased_vn100_2022-01-01_2026-05-12/*_trades.csv")
    df = pd.concat([pd.read_csv(f) for f in files if Path(f).exists()], ignore_index=True)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["pnl_pct"] = df["pnl_pct"] * 100
    return df


def attach_regime(df: pd.DataFrame, vni: pd.DataFrame) -> pd.DataFrame:
    vni = vni.sort_values("date").reset_index(drop=True)
    vni["ma50"] = vni["close"].rolling(50).mean()
    vni["ma50_slope"] = vni["ma50"].diff(20)
    vni["regime"] = "SIDEWAY"
    vni.loc[(vni["close"] > vni["ma50"]) & (vni["ma50_slope"] > 0), "regime"] = "UPTREND"
    vni.loc[(vni["close"] < vni["ma50"]) & (vni["ma50_slope"] < 0), "regime"] = "DOWNTREND"

    df = df.copy()
    df["signal_date_dt"] = pd.to_datetime(df.get("signal_date", df["entry_date"]))
    return pd.merge_asof(
        df.sort_values("signal_date_dt"),
        vni[["date", "regime"]].rename(columns={"date": "signal_date_dt"}).sort_values("signal_date_dt"),
        on="signal_date_dt",
        direction="backward",
    )


def find_deprecate_list(
    train_df: pd.DataFrame,
    *,
    fb_n: int = 8,
    sb_n: int = 12,
    min_avg: float = -3.0,
) -> set[tuple[str, str]]:
    if len(train_df) < 30:
        return set()
    reviewed = review_trades_df(train_df)
    losers = reviewed[reviewed["pnl_pct"] < 0]
    if losers.empty:
        return set()
    pat = (
        losers.groupby(["setup_type", "regime", "loss_category"])
        .agg(n=("symbol", "count"), avg_pct=("pnl_pct", "mean"))
        .reset_index()
    )
    out = set()
    for _, r in pat.iterrows():
        n, avg, cat = int(r["n"]), float(r["avg_pct"]), r["loss_category"]
        if avg > min_avg:
            continue
        if (cat == "FALSE_BREAKOUT" and n >= fb_n) or (cat == "STRUCTURE_BREAK" and n >= sb_n):
            out.add((r["setup_type"], r["regime"]))
    return out


def build_equity(df: pd.DataFrame, initial_nav: float, dates: pd.DatetimeIndex) -> pd.Series:
    eq = pd.Series(0.0, index=dates)
    if not df.empty:
        daily = df.groupby(df["exit_date"].dt.normalize())["pnl"].sum()
        common = daily.index.intersection(dates)
        eq.loc[common] = daily.loc[common].values
    return (initial_nav + eq.cumsum()).rename("equity")


def metrics(df: pd.DataFrame, initial_nav: float, dates: pd.DatetimeIndex) -> dict:
    n = len(df)
    losers = df[df["pnl_pct"] < 0]
    winners = df[df["pnl_pct"] >= 0]
    total_pnl = float(df["pnl"].sum()) if n else 0
    gw = float(winners["pnl"].sum()) if len(winners) else 0
    gl = float(losers["pnl"].sum()) if len(losers) else 0
    pf = (gw / abs(gl)) if gl != 0 else (float("inf") if gw > 0 else 0)

    eq = build_equity(df, initial_nav, dates)
    rets = eq.pct_change().fillna(0)
    sharpe = float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0.0
    rolling_max = eq.cummax()
    mdd = float(((eq - rolling_max) / rolling_max).min()) * 100

    return {
        "trades": n,
        "wins": len(winners),
        "losses": len(losers),
        "win_rate_pct": (len(winners) / n * 100) if n else 0.0,
        "pnl": total_pnl,
        "return_pct": total_pnl / initial_nav * 100,
        "mdd_pct": mdd,
        "sharpe": sharpe,
        "profit_factor": pf,
        "equity_final": float(eq.iloc[-1]),
        "equity_curve": eq,
    }


def fmt_row(name: str, b: dict, f: dict, key: str, fmt: str) -> str:
    bv, fv = b[key], f[key]
    d = fv - bv
    if fmt == "int":
        return f"{name:<18}  {bv:>14,}  {fv:>14,}  {d:>+14,}"
    if fmt == "pct":
        return f"{name:<18}  {bv:>13.2f}%  {fv:>13.2f}%  {d:>+13.2f}%"
    if fmt == "money":
        return f"{name:<18}  {bv:>14,.0f}  {fv:>14,.0f}  {d:>+14,.0f}"
    if fmt == "ratio":
        return f"{name:<18}  {bv:>14.2f}  {fv:>14.2f}  {d:>+14.2f}"


def print_block(label: str, baseline: dict, filtered: dict, skipped_count: int) -> None:
    print(f"\n{'=' * 80}")
    print(f"{label}")
    print(f"{'=' * 80}")
    print(f"{'METRIC':<18}  {'BASELINE':>14}  {'FILTERED':>14}  {'DELTA':>14}")
    print("-" * 80)
    print(fmt_row("Trades",         baseline, filtered, "trades",        "int"))
    print(fmt_row("Wins",           baseline, filtered, "wins",          "int"))
    print(fmt_row("Losses",         baseline, filtered, "losses",        "int"))
    print(fmt_row("Win rate",       baseline, filtered, "win_rate_pct",  "pct"))
    print(fmt_row("PnL (VND)",      baseline, filtered, "pnl",           "money"))
    print(fmt_row("Return %",       baseline, filtered, "return_pct",    "pct"))
    print(fmt_row("MDD %",          baseline, filtered, "mdd_pct",       "pct"))
    print(fmt_row("Sharpe (ann.)",  baseline, filtered, "sharpe",        "ratio"))
    print(fmt_row("Profit Factor",  baseline, filtered, "profit_factor", "ratio"))
    print(fmt_row("Equity final",   baseline, filtered, "equity_final",  "money"))
    print(f"\nSkipped {skipped_count} trades (deprecated keys)")


def main() -> None:
    from multiagents_trading_assistant.services.data_service import get_vnindex
    df = load_all_trades()
    vni = get_vnindex(2000)
    df = attach_regime(df, vni)

    print(f"Total trades 2022-2026: {len(df)}")
    print(f"Date range: {df['entry_date'].min().date()} → {df['entry_date'].max().date()}")
    print(f"Setups: {dict(df['setup_type'].value_counts())}")

    test_years = [2024, 2025, 2026]
    cumulative_baseline = []
    cumulative_filtered = []

    for year in test_years:
        train = df[df["entry_date"] < f"{year}-01-01"]
        test = df[(df["entry_date"] >= f"{year}-01-01") & (df["entry_date"] < f"{year + 1}-01-01")]
        if test.empty:
            continue

        deprecated = find_deprecate_list(train)
        print(f"\n{'#' * 80}")
        print(f"# YEAR {year} — train on {len(train)} prior trades, test on {len(test)} trades")
        print(f"{'#' * 80}")
        print(f"Deprecate list: {sorted(deprecated) if deprecated else '(empty)'}")

        keys = list(zip(test["setup_type"], test["regime"]))
        skip_mask = pd.Series([k in deprecated for k in keys], index=test.index)
        kept = test[~skip_mask]

        end_date = min(pd.Timestamp(f"{year}-12-31"), df["exit_date"].max())
        dates = pd.date_range(f"{year}-01-01", end_date, freq="D")

        b = metrics(test, INITIAL_NAV, dates)
        f = metrics(kept, INITIAL_NAV, dates)
        print_block(f"YEAR {year}", b, f, int(skip_mask.sum()))

        cumulative_baseline.append(test)
        cumulative_filtered.append(kept)

    # ── Cumulative 2024-2026 ────────────────────────────────────────────────
    if cumulative_baseline:
        all_b = pd.concat(cumulative_baseline)
        all_f = pd.concat(cumulative_filtered)
        end_date = df["exit_date"].max()
        dates = pd.date_range("2024-01-01", end_date, freq="D")
        b = metrics(all_b, INITIAL_NAV, dates)
        f = metrics(all_f, INITIAL_NAV, dates)
        print_block(
            f"CUMULATIVE 2024-01-01 → {end_date.date()} (rolling walk-forward)",
            b, f, len(all_b) - len(all_f),
        )

        # Annualized return
        years = (end_date - pd.Timestamp("2024-01-01")).days / 365.25
        cagr_b = (b["equity_final"] / INITIAL_NAV) ** (1 / years) - 1
        cagr_f = (f["equity_final"] / INITIAL_NAV) ** (1 / years) - 1
        print(f"\nDuration: {years:.2f} years")
        print(f"CAGR baseline: {cagr_b * 100:+.2f}%   |   CAGR filtered: {cagr_f * 100:+.2f}%   |   Delta: {(cagr_f - cagr_b) * 100:+.2f}pp")


if __name__ == "__main__":
    main()
