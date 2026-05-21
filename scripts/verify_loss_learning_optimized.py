"""4-variant comparison of closed-loop learning approaches.

Variants:
  A. BASELINE        — no learning, take all signals
  B. STATIC          — deprecate from ALL historical losses (current production design)
  C. ROLLING_365     — deprecate only from last 365 days of losses
  D. ROLLING+PROMOTE — C + auto-undo deprecation if recent 90d expectancy positive

Each evaluated on 2025-01-01 → today, walk-forward (re-learn at each year boundary).

Run: PYTHONIOENCODING=utf-8 python scripts/verify_loss_learning_optimized.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from verify_loss_learning_walkforward import (
    INITIAL_NAV,
    attach_regime,
    build_equity,
    load_all_trades,
    metrics,
)
from multiagents_trading_assistant.agentic.post_trade_review import review_trades_df


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


def find_promote_list(
    recent_df: pd.DataFrame,
    candidates: set[tuple[str, str]],
    *,
    min_n: int = 5,
    min_expectancy_pct: float = 0.5,
) -> set[tuple[str, str]]:
    """Among `candidates` (currently deprecated), promote back if recent expectancy positive."""
    promoted = set()
    if recent_df.empty:
        return promoted
    grouped = (
        recent_df.groupby(["setup_type", "regime"])
        .agg(n=("pnl_pct", "count"), exp=("pnl_pct", "mean"))
    )
    for key in candidates:
        if key not in grouped.index:
            continue
        row = grouped.loc[key]
        if int(row["n"]) >= min_n and float(row["exp"]) >= min_expectancy_pct:
            promoted.add(key)
    return promoted


def variant_filter(test: pd.DataFrame, deprecated: set[tuple[str, str]], *, mode: str = "skip", size_factor: float = 0.5) -> pd.DataFrame:
    """Apply deprecate logic. mode='skip' removes trades; mode='reduce' keeps but scales pnl by size_factor."""
    if not deprecated:
        return test
    keys = list(zip(test["setup_type"], test["regime"]))
    mask = pd.Series([k in deprecated for k in keys], index=test.index)
    if mode == "skip":
        return test[~mask]
    out = test.copy()
    out.loc[mask, "pnl"] = out.loc[mask, "pnl"] * size_factor
    return out


def evaluate_period(
    df: pd.DataFrame, start: str, end_date: pd.Timestamp,
    *, variant: str, all_data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[set]]:
    """Walk year-by-year, return aggregated kept trades + per-year deprecate lists."""
    kept = []
    dep_lists = []
    for year in (2025, 2026):
        train_all = all_data[all_data["entry_date"] < f"{year}-01-01"]
        train_recent = all_data[
            (all_data["entry_date"] >= f"{year - 1}-01-01") &
            (all_data["entry_date"] < f"{year}-01-01")
        ]
        test = df[(df["entry_date"] >= f"{year}-01-01") & (df["entry_date"] < f"{year + 1}-01-01")]
        if test.empty:
            continue

        if variant == "A":
            deprecated = set()
        elif variant == "B":
            deprecated = find_deprecate_list(train_all)
        elif variant == "C":
            deprecated = find_deprecate_list(train_all.tail(0).pipe(  # last 365d only
                lambda _: train_all[train_all["entry_date"] >= pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=365)]
            ))
        elif variant == "D":
            cutoff_train = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=365)
            cutoff_recent = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=90)
            train_window = train_all[train_all["entry_date"] >= cutoff_train]
            recent_window = train_all[train_all["entry_date"] >= cutoff_recent]
            deprecated = find_deprecate_list(train_window)
            promoted = find_promote_list(recent_window, deprecated)
            deprecated = deprecated - promoted
        elif variant == "E":
            # Static deprecate (all-history) BUT allow promote-back from last 180d
            deprecated = find_deprecate_list(train_all)
            cutoff_recent = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=180)
            recent_window = train_all[train_all["entry_date"] >= cutoff_recent]
            promoted = find_promote_list(recent_window, deprecated)
            deprecated = deprecated - promoted
        elif variant == "F":
            deprecated = find_deprecate_list(train_all)
            cutoff_recent = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=180)
            recent_window = train_all[train_all["entry_date"] >= cutoff_recent]
            promoted = find_promote_list(recent_window, deprecated, min_n=7, min_expectancy_pct=0.0)
            deprecated = deprecated - promoted
        elif variant == "G":
            # Static deprecate, but size-REDUCE 50% instead of full skip
            deprecated = find_deprecate_list(train_all)
        elif variant == "H":
            # Static deprecate, size-REDUCE 30% (more aggressive cap)
            deprecated = find_deprecate_list(train_all)
        elif variant == "I":
            # Stricter threshold (n≥15 FB, n≥20 SB, avg≤-5%) — only kill very-confident bad
            deprecated = find_deprecate_list(train_all, fb_n=15, sb_n=20, min_avg=-5.0)
        else:
            raise ValueError(variant)

        dep_lists.append(deprecated)
        if variant == "G":
            kept.append(variant_filter(test, deprecated, mode="reduce", size_factor=0.5))
        elif variant == "H":
            kept.append(variant_filter(test, deprecated, mode="reduce", size_factor=0.3))
        else:
            kept.append(variant_filter(test, deprecated))
    return pd.concat(kept) if kept else pd.DataFrame(), dep_lists


def main() -> None:
    from multiagents_trading_assistant.services.data_service import get_vnindex
    df = load_all_trades()
    df = attach_regime(df, get_vnindex(2000))
    end_date = df["exit_date"].max()
    dates = pd.date_range("2025-01-01", end_date, freq="D")
    test_universe = df[(df["entry_date"] >= "2025-01-01") & (df["entry_date"] <= end_date)]

    print(f"Test period: 2025-01-01 → {end_date.date()}  ({len(test_universe)} baseline trades)")
    print(f"Initial NAV: {INITIAL_NAV:,.0f} VND\n")

    variants = {
        "A. BASELINE":      "A",
        "B. STATIC-SKIP":   "B",
        "G. STATIC-RED50":  "G",  # size-reduce 50% on bad keys
        "H. STATIC-RED30":  "H",  # size-reduce 30% on bad keys
        "I. STRICT-SKIP":   "I",  # stricter threshold, only kill very-bad
    }

    results = {}
    for label, code in variants.items():
        kept, dep_lists = evaluate_period(test_universe, "2025-01-01", end_date, variant=code, all_data=df)
        if code == "A":
            kept = test_universe
        m = metrics(kept, INITIAL_NAV, dates)
        m["dep_lists"] = dep_lists
        results[label] = m

    # ── Comparison table ───────────────────────────────────────────────────
    keys = [
        ("Trades",         "trades",        "{:>12,}"),
        ("Win rate",       "win_rate_pct",  "{:>11.2f}%"),
        ("PnL (VND)",      "pnl",           "{:>12,.0f}"),
        ("Return %",       "return_pct",    "{:>11.2f}%"),
        ("MDD %",          "mdd_pct",       "{:>11.2f}%"),
        ("Sharpe (ann.)",  "sharpe",        "{:>12.2f}"),
        ("Profit Factor",  "profit_factor", "{:>12.2f}"),
        ("Equity final",   "equity_final",  "{:>12,.0f}"),
    ]

    header = f"{'METRIC':<16}" + "".join(f"{label:>17}" for label in variants.keys())
    print(header)
    print("-" * len(header))
    for name, key, fmt in keys:
        row = f"{name:<16}"
        for label in variants.keys():
            v = results[label][key]
            cell = fmt.format(v)
            row += f"{cell:>17}"
        print(row)

    # CAGR
    yrs = (end_date - pd.Timestamp("2025-01-01")).days / 365.25
    cagr_row = f"{'CAGR':<16}"
    for label in variants.keys():
        eq = results[label]["equity_final"]
        cagr = (eq / INITIAL_NAV) ** (1 / yrs) - 1
        cagr_row += f"{cagr * 100:>16.2f}%"
    print(cagr_row)

    for show in ("B. STATIC-SKIP", "I. STRICT-SKIP"):
        print(f"\n--- Deprecate list per year (variant {show}) ---")
        for year, dep in zip((2025, 2026), results[show]["dep_lists"]):
            print(f"  {year}: {sorted(dep) if dep else '(empty)'}")


if __name__ == "__main__":
    main()
