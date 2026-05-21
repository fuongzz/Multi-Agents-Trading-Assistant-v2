"""Apply learned deprecate-list to ACTUAL live_pipeline output (3 core strategies).

Train: 2022-2024 trades from same 3 strategies (offline backtest CSVs)
Test : 2025-01 to 2026-05-12 ACTUAL live_pipeline trades (real portfolio engine)

Compares full equity metrics with vs without deprecate filter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from verify_loss_learning_walkforward import attach_regime, build_equity, metrics
from multiagents_trading_assistant.agentic.post_trade_review import review_trades_df
from multiagents_trading_assistant.services.data_service import get_vnindex


CORE3 = ("breakout_after_accumulation_v3",
         "compression_breakout_smt_v1",
         "mean_reversion_uptrend_ma50_v1")

LIVE_DIR = Path("backtest_results/live_pipeline_regime_router_mvp_v3_core3_2025_to_now_20260511_221800")
TRAIN_DIR = Path("backtest_results/all_strategies_unbiased_vn100_2022-01-01_2026-05-12")
INITIAL_NAV = 100_000_000.0  # live_pipeline default


def build_deprecate_list() -> set[tuple[str, str]]:
    """Train on 2022-2024 trades from 3 core strategies."""
    train_files = [TRAIN_DIR / f"{s}_trades.csv" for s in CORE3]
    train_files = [f for f in train_files if f.exists()]
    if not train_files:
        print(f"⚠ No train CSVs found for core3 in {TRAIN_DIR}")
        return set()
    train = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)
    train["entry_date"] = pd.to_datetime(train["entry_date"])
    train = train[train["entry_date"] < "2025-01-01"]
    train["pnl_pct"] = train["pnl_pct"] * 100
    # Map strategy_name → setup_type label used in classifier
    train["setup_type"] = train["edge_strategy_name"]

    vni = get_vnindex(2000)
    train = attach_regime(train, vni)

    print(f"Train: {len(train)} core3 trades 2022 → 2024-12-31")
    print(f"  Per strategy: {dict(train['edge_strategy_name'].value_counts())}")

    reviewed = review_trades_df(train)
    losers = reviewed[reviewed["pnl_pct"] < 0]
    pat = (
        losers.groupby(["setup_type", "regime", "loss_category"])
        .agg(n=("symbol", "count"), avg_pct=("pnl_pct", "mean"))
        .reset_index()
    )
    print(f"\nTrain loss patterns (n ≥ 5):")
    print(pat[pat["n"] >= 5].to_string(index=False))

    # Production thresholds (from sweep): n≥15 FB, n≥20 SB, avg≤-5%
    deprecated = set()
    for _, r in pat.iterrows():
        n, avg, cat = int(r["n"]), float(r["avg_pct"]), r["loss_category"]
        if avg > -5.0:
            continue
        if (cat == "FALSE_BREAKOUT" and n >= 15) or (cat == "STRUCTURE_BREAK" and n >= 20):
            deprecated.add((r["setup_type"], r["regime"]))
    return deprecated


def main() -> None:
    # Load ACTUAL live_pipeline output for 3 core strategies, 2025-now
    live_trades = pd.read_csv(LIVE_DIR / "trades.csv")
    live_trades["entry_date"] = pd.to_datetime(live_trades["entry_date"])
    live_trades["exit_date"] = pd.to_datetime(live_trades["exit_date"])
    live_trades["pnl_pct"] = live_trades["pnl_pct"] * 100  # csv stores fraction

    # Tag with strategy_name + regime
    # core3 trades.csv stores edge_strategy_name in 'setup_type' column already? check
    live_eq = pd.read_csv(LIVE_DIR / "equity_curve.csv")
    live_eq["date"] = pd.to_datetime(live_eq["date"])
    print(f"\nLIVE 2025-now actual: {len(live_trades)} trades")
    print(f"Date range: {live_trades['entry_date'].min().date()} → {live_trades['entry_date'].max().date()}")

    # Need strategy_name in live trades — load from trades.csv if column exists, else infer
    sym_cols = list(live_trades.columns)
    print(f"Cols: {sym_cols}")
    if "edge_strategy_name" not in live_trades.columns:
        # mvp_v3 trades.csv may not have this column; infer from setup_type → may map back
        print("⚠ edge_strategy_name missing — using setup_type as fallback")
        live_trades["edge_strategy_name"] = live_trades["setup_type"]
    live_trades["setup_type"] = live_trades["edge_strategy_name"]

    vni = get_vnindex(2000)
    live_trades = attach_regime(live_trades, vni)

    # Build deprecate list from train
    deprecated = build_deprecate_list()
    print(f"\nDeprecate list (from 2022-2024 core3 losses):")
    for k in sorted(deprecated):
        print(f"  ✗ {k}")
    if not deprecated:
        print("  (empty — no patterns met production threshold)")

    # Apply gate
    keys = list(zip(live_trades["edge_strategy_name"], live_trades["regime"]))
    skip = pd.Series([k in deprecated for k in keys], index=live_trades.index)
    kept = live_trades[~skip]
    skipped = live_trades[skip]

    end = live_trades["exit_date"].max()
    dates = pd.date_range("2025-01-01", end, freq="D")

    base_m = metrics(live_trades, INITIAL_NAV, dates)
    filt_m = metrics(kept, INITIAL_NAV, dates)

    print("\n" + "=" * 86)
    print(f"3 CORE STRATEGIES — LIVE PIPELINE 2025-01 → {end.date()}  (NAV {INITIAL_NAV:,.0f})")
    print("=" * 86)
    print(f"{'METRIC':<20}  {'BASELINE':>18}  {'FILTERED':>18}  {'DELTA':>18}")
    print("-" * 86)
    rows = [
        ("Trades",         "trades",        "{:>18,}"),
        ("Win rate",       "win_rate_pct",  "{:>17.2f}%"),
        ("PnL (VND)",      "pnl",           "{:>18,.0f}"),
        ("Return %",       "return_pct",    "{:>17.2f}%"),
        ("MDD %",          "mdd_pct",       "{:>17.2f}%"),
        ("Sharpe (ann.)",  "sharpe",        "{:>18.2f}"),
        ("Profit Factor",  "profit_factor", "{:>18.2f}"),
        ("Equity final",   "equity_final",  "{:>18,.0f}"),
    ]
    for name, key, fmt in rows:
        bv, fv = base_m[key], filt_m[key]
        print(f"{name:<20}  {fmt.format(bv)}  {fmt.format(fv)}  {fmt.format(fv - bv)}")

    print(f"\nSkipped {len(skipped)} of {len(live_trades)} trades")
    if len(skipped):
        print("\nSkipped trades (what deprecate would have avoided):")
        print(skipped[["symbol", "edge_strategy_name", "regime", "entry_date", "exit_date", "pnl", "pnl_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
