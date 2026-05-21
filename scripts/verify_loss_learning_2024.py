"""Out-of-sample validation: train deprecate-list on 2022-23 losses,
apply to 2024, measure if avoiding deprecated patterns improves PnL.

Run: python scripts/verify_loss_learning_2024.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from multiagents_trading_assistant.agentic.post_trade_review import review_trades_df


def load_all_trades() -> pd.DataFrame:
    files = glob.glob("backtest_results/all_strategies_unbiased_vn100_2022-01-01_2026-05-12/*_trades.csv")
    dfs = [pd.read_csv(f) for f in files if Path(f).exists()]
    df = pd.concat(dfs, ignore_index=True)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["pnl_pct"] = df["pnl_pct"] * 100  # fraction → percent for classifier
    return df


def attach_regime(df: pd.DataFrame, vni: pd.DataFrame) -> pd.DataFrame:
    """Tag each trade with regime at signal_date based on VNI 50d trend."""
    vni = vni.sort_values("date").reset_index(drop=True)
    vni["ma50"] = vni["close"].rolling(50).mean()
    vni["ma50_slope"] = vni["ma50"].diff(20)
    vni["regime"] = "SIDEWAY"
    vni.loc[(vni["close"] > vni["ma50"]) & (vni["ma50_slope"] > 0), "regime"] = "UPTREND"
    vni.loc[(vni["close"] < vni["ma50"]) & (vni["ma50_slope"] < 0), "regime"] = "DOWNTREND"

    df = df.copy()
    df["signal_date_dt"] = pd.to_datetime(df.get("signal_date", df["entry_date"]))
    merged = pd.merge_asof(
        df.sort_values("signal_date_dt"),
        vni[["date", "regime"]].rename(columns={"date": "signal_date_dt"}).sort_values("signal_date_dt"),
        on="signal_date_dt",
        direction="backward",
    )
    return merged


def find_deprecate_list(
    train_df: pd.DataFrame,
    *,
    false_break_threshold: int = 8,
    structure_break_threshold: int = 12,
    min_avg_loss_pct: float = -3.0,
) -> list[tuple[str, str, str, int, float]]:
    """Return list of (setup, regime, reason_category, count, avg_pnl_pct) to deprecate."""
    reviewed = review_trades_df(train_df)
    losers = reviewed[reviewed["pnl_pct"] < 0]
    pat = (
        losers.groupby(["setup_type", "regime", "loss_category"])
        .agg(n=("symbol", "count"), avg_pct=("pnl_pct", "mean"))
        .reset_index()
    )

    deprecated = []
    for _, r in pat.iterrows():
        setup, regime, cat = r["setup_type"], r["regime"], r["loss_category"]
        n, avg = int(r["n"]), float(r["avg_pct"])
        if avg > min_avg_loss_pct:
            continue
        if cat == "FALSE_BREAKOUT" and n >= false_break_threshold:
            deprecated.append((setup, regime, cat, n, avg))
        elif cat == "STRUCTURE_BREAK" and n >= structure_break_threshold:
            deprecated.append((setup, regime, cat, n, avg))
    return deprecated


def build_equity_curve(df: pd.DataFrame, initial_nav: float, date_range) -> pd.Series:
    """Daily equity curve = initial_nav + cumulative realized PnL on exit_date."""
    if df.empty:
        return pd.Series(initial_nav, index=date_range, name="equity")
    daily = df.groupby(pd.to_datetime(df["exit_date"]))["pnl"].sum()
    eq = pd.Series(0.0, index=date_range)
    eq.update(daily.reindex(daily.index.intersection(date_range), fill_value=0))
    return (initial_nav + eq.cumsum()).rename("equity")


def evaluate(df: pd.DataFrame, initial_nav: float, date_range) -> dict:
    import numpy as np
    n = len(df)
    losers = df[df["pnl_pct"] < 0]
    winners = df[df["pnl_pct"] >= 0]
    total_pnl = float(df["pnl"].sum()) if n else 0
    gross_wins = float(winners["pnl"].sum()) if len(winners) else 0
    gross_losses = float(losers["pnl"].sum()) if len(losers) else 0
    pf = (gross_wins / abs(gross_losses)) if gross_losses != 0 else float("inf") if gross_wins > 0 else 0

    eq = build_equity_curve(df, initial_nav, date_range)
    returns = eq.pct_change().fillna(0)
    sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max
    mdd = float(dd.min())

    return {
        "trades": n,
        "wins": len(winners),
        "losses": len(losers),
        "win_rate": len(winners) / n if n else 0,
        "total_pnl": total_pnl,
        "return_pct": total_pnl / initial_nav * 100,
        "mdd_pct": mdd * 100,
        "sharpe": float(sharpe),
        "profit_factor": pf,
        "avg_pnl_pct": float(df["pnl_pct"].mean()) if n else 0,
        "gross_loss": gross_losses,
        "equity_final": float(eq.iloc[-1]),
    }


def main() -> None:
    from multiagents_trading_assistant.services.data_service import get_vnindex
    df = load_all_trades()
    vni = get_vnindex(2000)
    df = attach_regime(df, vni)

    train = df[(df["entry_date"] >= "2022-01-01") & (df["entry_date"] <= "2023-12-31")]
    test = df[(df["entry_date"] >= "2024-01-01") & (df["entry_date"] <= "2024-12-31")]

    print("=" * 78)
    print(f"TRAIN: 2022-01 → 2023-12   ({len(train)} trades)")
    print(f"TEST : 2024-01 → 2024-12   ({len(test)} trades)")
    print("=" * 78)
    print(f"Regime distribution train: {dict(train['regime'].value_counts())}")
    print(f"Regime distribution test : {dict(test['regime'].value_counts())}")

    # ── Step 1: classify training losses (regime-aware) ────────────────────
    deprecated = find_deprecate_list(train)

    print("\nDeprecate list learned from 2022-2023 losses (setup × regime):")
    if not deprecated:
        print("  (no patterns met the threshold)")
        return
    for setup, regime, cat, n, avg in deprecated:
        print(f"  ✗ {setup:<22}  {regime:<10}  {cat:<18}  n={n:>3}  avg={avg:6.2f}%")

    deprecated_keys = {(s, r) for s, r, _, _, _ in deprecated}

    # ── Step 2: evaluate 2024 — baseline vs filtered ───────────────────────
    test_keys = list(zip(test["setup_type"], test["regime"]))
    skip_mask = pd.Series([k in deprecated_keys for k in test_keys], index=test.index)
    filtered_df = test[~skip_mask]
    skipped_df = test[skip_mask]

    INITIAL_NAV = 1_000_000_000  # 1B VND
    date_range = pd.date_range("2024-01-01", "2024-12-31", freq="D")

    baseline = evaluate(test, INITIAL_NAV, date_range)
    filtered = evaluate(filtered_df, INITIAL_NAV, date_range)

    print("\n" + "=" * 86)
    print(f"2024 PERFORMANCE — Initial NAV: {INITIAL_NAV:,.0f} VND")
    print("=" * 86)
    print(f"{'METRIC':<22}  {'BASELINE':>18}  {'FILTERED':>18}  {'DELTA':>18}")
    print("-" * 86)

    def fmt_int(x): return f"{x:>18,}"
    def fmt_pct(x): return f"{x:>17.2f}%"
    def fmt_money(x): return f"{x:>18,.0f}"
    def fmt_ratio(x): return f"{x:>18.2f}"

    rows = [
        ("Trades",         baseline["trades"],        filtered["trades"],        fmt_int),
        ("Wins",           baseline["wins"],          filtered["wins"],          fmt_int),
        ("Losses",         baseline["losses"],        filtered["losses"],        fmt_int),
        ("Win rate",       baseline["win_rate"]*100,  filtered["win_rate"]*100,  fmt_pct),
        ("PnL (VND)",      baseline["total_pnl"],     filtered["total_pnl"],     fmt_money),
        ("Return %",       baseline["return_pct"],    filtered["return_pct"],    fmt_pct),
        ("MDD %",          baseline["mdd_pct"],       filtered["mdd_pct"],       fmt_pct),
        ("Sharpe (annual)",baseline["sharpe"],        filtered["sharpe"],        fmt_ratio),
        ("Profit Factor",  baseline["profit_factor"], filtered["profit_factor"], fmt_ratio),
        ("Avg PnL %",      baseline["avg_pnl_pct"],   filtered["avg_pnl_pct"],   fmt_pct),
        ("Gross loss",     baseline["gross_loss"],    filtered["gross_loss"],    fmt_money),
        ("Equity final",   baseline["equity_final"],  filtered["equity_final"],  fmt_money),
    ]
    for name, b, f, fn in rows:
        d = f - b
        print(f"{name:<22}  {fn(b)}  {fn(f)}  {fn(d)}")

    print(f"\nSkipped: {len(skipped_df)} trades  |  Kept: {len(filtered_df)} trades")

    # ── Step 3: what was skipped, was it actually bad in 2024? ─────────────
    if len(skipped_df):
        per = (
            skipped_df.groupby(["setup_type", "regime"])
            .agg(n=("symbol", "count"),
                 wins=("pnl_pct", lambda x: int((x >= 0).sum())),
                 losses=("pnl_pct", lambda x: int((x < 0).sum())),
                 total_pnl=("pnl", "sum"),
                 avg_pct=("pnl_pct", "mean"))
            .reset_index()
        )
        print("\nWhat was skipped in 2024 (deprecated keys):")
        print(per.to_string(index=False))
        skipped_pnl = float(skipped_df["pnl"].sum())
        verdict = "AVOIDED LOSS ✓" if skipped_pnl < 0 else "MISSED PROFIT ✗"
        print(f"\n→ Skipped trades' actual 2024 PnL: {skipped_pnl:+,.0f}  [{verdict}]")


if __name__ == "__main__":
    main()
