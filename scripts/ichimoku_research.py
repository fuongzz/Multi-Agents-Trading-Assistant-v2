"""Ichimoku research — chuẩn lý thuyết, độc lập với pipeline.

Mục đích: tìm rule Ichimoku có edge thật trên thị trường VN.

Chạy: python scripts/ichimoku_research.py
Output: backtest_results/ichimoku_research/{entry}_{exit}_trades.csv + summary.csv
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OHLCV_PATH = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
OUT_DIR = ROOT / "backtest_results" / "ichimoku_research"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Period
FROM_DATE = "2022-01-01"
TO_DATE = "2026-05-01"

# Risk
INITIAL_SL_PCT = 0.05  # 5% capped initial SL
MAX_HOLD_BARS = 120
MIN_AVG_VOL_20D = 200_000


# ───────────────────────── Indicators ────────────────────────────────────
def ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Classical Ichimoku 9/26/52/26 (Senkou shift, Chikou lag)."""
    h, l, c = df["high"], df["low"], df["close"]
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    senkou_a_raw = (tenkan + kijun) / 2
    senkou_b_raw = (h.rolling(52).max() + l.rolling(52).min()) / 2
    # Cloud at current bar = senkou plotted from 26 bars ago
    senkou_a = senkou_a_raw.shift(26)
    senkou_b = senkou_b_raw.shift(26)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bot = np.minimum(senkou_a, senkou_b)
    # Chikou_free at bar i  ⇔  close[i] > close[i-26]  (lagging plot above price)
    chikou_free = c > c.shift(26)
    # Future cloud green: senkou_a_raw > senkou_b_raw (no shift) — predicts cloud color 26 bars ahead
    future_green = senkou_a_raw > senkou_b_raw
    out = df.copy()
    out["tenkan"] = tenkan
    out["kijun"] = kijun
    out["senkou_a"] = senkou_a
    out["senkou_b"] = senkou_b
    out["cloud_top"] = cloud_top
    out["cloud_bot"] = cloud_bot
    out["chikou_free"] = chikou_free
    out["future_green"] = future_green
    out["senkou_a_raw"] = senkou_a_raw
    out["senkou_b_raw"] = senkou_b_raw
    # ATR for SL sizing
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    out["vol_ma20"] = df["volume"].rolling(20).mean()
    return out


# ───────────────────────── Entry rules ───────────────────────────────────
def entry_tk_cross_raw(row, prev) -> bool:
    return prev["tenkan"] <= prev["kijun"] and row["tenkan"] > row["kijun"]


def entry_tk_cross_above_cloud(row, prev) -> bool:
    return (
        prev["tenkan"] <= prev["kijun"]
        and row["tenkan"] > row["kijun"]
        and row["close"] > row["cloud_top"]
    )


def entry_price_break_cloud(row, prev) -> bool:
    return prev["close"] <= prev["cloud_top"] and row["close"] > row["cloud_top"]


def entry_kijun_bounce(row, prev) -> bool:
    # Uptrend (price above cloud), pullback touches Kijun, close back above Kijun, bullish candle
    return (
        row["close"] > row["cloud_top"]
        and row["low"] <= row["kijun"] * 1.01
        and row["close"] > row["kijun"]
        and row["close"] > row["open"]
    )


def entry_kumo_twist_bull(row, prev) -> bool:
    # Future cloud (senkou_a_raw vs senkou_b_raw, no shift) flips bullish
    return (
        prev["senkou_a_raw"] <= prev["senkou_b_raw"]
        and row["senkou_a_raw"] > row["senkou_b_raw"]
        and row["close"] > row["kijun"]
    )


def entry_chikou_free(row, prev) -> bool:
    # Chikou crosses above past price (i.e., close > close[-26] now, but wasn't yesterday)
    return (not prev["chikou_free"]) and bool(row["chikou_free"])


def entry_triple_screen(row, prev) -> bool:
    return (
        prev["tenkan"] <= prev["kijun"]
        and row["tenkan"] > row["kijun"]
        and row["close"] > row["cloud_top"]
        and bool(row["chikou_free"])
    )


def entry_perfect_bullish(row, prev) -> bool:
    # Daily reading: full bullish alignment (not a "cross" event — entry on first bar all conditions met)
    aligned_now = (
        row["tenkan"] > row["kijun"]
        and row["close"] > row["tenkan"]
        and row["close"] > row["cloud_top"]
        and row["senkou_a"] > row["senkou_b"]  # current cloud green
        and bool(row["future_green"])
        and bool(row["chikou_free"])
    )
    aligned_prev = (
        prev["tenkan"] > prev["kijun"]
        and prev["close"] > prev["tenkan"]
        and prev["close"] > prev["cloud_top"]
        and prev["senkou_a"] > prev["senkou_b"]
        and bool(prev["future_green"])
        and bool(prev["chikou_free"])
    )
    return aligned_now and not aligned_prev


def entry_kijun_break(row, prev) -> bool:
    return prev["close"] <= prev["kijun"] and row["close"] > row["kijun"]


ENTRY_RULES: dict[str, Callable] = {
    "TK_CROSS_RAW": entry_tk_cross_raw,
    "TK_CROSS_ABOVE_CLOUD": entry_tk_cross_above_cloud,
    "PRICE_BREAK_CLOUD": entry_price_break_cloud,
    "KIJUN_BOUNCE": entry_kijun_bounce,
    "KUMO_TWIST_BULL": entry_kumo_twist_bull,
    "CHIKOU_FREE": entry_chikou_free,
    "TRIPLE_SCREEN": entry_triple_screen,
    "PERFECT_BULLISH": entry_perfect_bullish,
    "KIJUN_BREAK": entry_kijun_break,
}


# ───────────────────────── Exit rules ────────────────────────────────────
def exit_kijun_cross_down(row, prev, state) -> bool:
    return prev["close"] >= prev["kijun"] and row["close"] < row["kijun"]


def exit_tk_death_cross(row, prev, state) -> bool:
    return prev["tenkan"] >= prev["kijun"] and row["tenkan"] < row["kijun"]


def exit_trailing_swing(row, prev, state) -> bool:
    """Engine-style trailing swing low — state tracks dynamic_sl."""
    gain = (state["peak_close"] - state["entry_price"]) / state["entry_price"]
    if gain >= 0.05:
        if gain >= 0.20:
            lookback = 20
        elif gain >= 0.10:
            lookback = 12
        else:
            lookback = 7
        lows = state["lows"][-lookback:]
        atr = state["entry_atr"]
        new_sl = min(lows) - 0.5 * atr if atr > 0 else min(lows) * 0.99
        state["dynamic_sl"] = max(state["dynamic_sl"], new_sl)
    return row["low"] <= state["dynamic_sl"]


EXIT_RULES: dict[str, Callable] = {
    "KIJUN_CROSS_DOWN": exit_kijun_cross_down,
    "TK_DEATH_CROSS": exit_tk_death_cross,
    "TRAILING_SWING": exit_trailing_swing,
}


# ───────────────────────── Backtest core ─────────────────────────────────
@dataclass
class TradeRec:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    bars_held: int
    exit_reason: str


def backtest_symbol(
    df: pd.DataFrame,
    symbol: str,
    entry_fn: Callable,
    exit_fn: Callable,
    exit_name: str,
) -> list[TradeRec]:
    df = ichimoku(df).reset_index(drop=True)
    trades: list[TradeRec] = []
    pos = None
    state: dict = {}

    n = len(df)
    for i in range(60, n - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        if pd.isna(row["senkou_b"]) or pd.isna(row["kijun"]):
            continue
        if row["vol_ma20"] < MIN_AVG_VOL_20D:
            continue

        # ── Exit check ─────────────────────────────────────────────────
        if pos is not None:
            state["peak_close"] = max(state["peak_close"], float(row["close"]))
            state["lows"].append(float(row["low"]))
            bars_held = i - state["entry_idx"]

            # Initial SL stop check (independent of exit rule)
            if float(row["low"]) <= state["initial_sl"]:
                exit_px = min(state["initial_sl"], float(row["open"]))
                pnl = (exit_px - state["entry_price"]) / state["entry_price"] * 100
                trades.append(TradeRec(
                    symbol, state["entry_date"], str(row["date"])[:10],
                    state["entry_price"], exit_px, pnl, bars_held, "SL"))
                pos = None
                continue

            # Exit rule check (uses close of bar i → exit at next open)
            should_exit = False
            try:
                should_exit = bool(exit_fn(row, prev, state))
            except Exception:
                should_exit = False

            if should_exit:
                if i + 1 < n:
                    next_open = float(df.iloc[i + 1]["open"])
                    next_date = str(df.iloc[i + 1]["date"])[:10]
                else:
                    next_open = float(row["close"])
                    next_date = str(row["date"])[:10]
                pnl = (next_open - state["entry_price"]) / state["entry_price"] * 100
                trades.append(TradeRec(
                    symbol, state["entry_date"], next_date,
                    state["entry_price"], next_open, pnl, bars_held + 1, exit_name))
                pos = None
                continue

            if bars_held >= MAX_HOLD_BARS:
                pnl = (float(row["close"]) - state["entry_price"]) / state["entry_price"] * 100
                trades.append(TradeRec(
                    symbol, state["entry_date"], str(row["date"])[:10],
                    state["entry_price"], float(row["close"]), pnl, bars_held, "SAFETY_CAP"))
                pos = None
                continue

        # ── Entry check ────────────────────────────────────────────────
        if pos is None and i + 1 < n:
            try:
                fire = bool(entry_fn(row, prev))
            except Exception:
                fire = False
            if fire:
                entry_px = float(df.iloc[i + 1]["open"])
                if entry_px <= 0:
                    continue
                atr = float(row["atr14"]) if not pd.isna(row["atr14"]) else 0.0
                sl_dist = min(max(1.5 * atr, entry_px * 0.02), entry_px * INITIAL_SL_PCT)
                pos = True
                state = {
                    "entry_date": str(df.iloc[i + 1]["date"])[:10],
                    "entry_idx": i + 1,
                    "entry_price": entry_px,
                    "entry_atr": atr,
                    "initial_sl": entry_px - sl_dist,
                    "dynamic_sl": entry_px - sl_dist,
                    "peak_close": entry_px,
                    "lows": [float(df.iloc[i + 1]["low"])],
                }
    return trades


# ───────────────────────── Metrics ───────────────────────────────────────
def summarize(trades: list[TradeRec]) -> dict:
    if not trades:
        return {"n_trades": 0, "win_rate": 0.0, "avg_return": 0.0, "median_return": 0.0,
                "profit_factor": 0.0, "avg_bars": 0.0, "sl_pct": 0.0, "rev_pct": 0.0,
                "max_dd_trade": 0.0, "expectancy": 0.0, "total_return": 0.0}
    pnls = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 1e-9
    sl_count = sum(1 for t in trades if t.exit_reason == "SL")
    rev_count = sum(1 for t in trades if t.exit_reason not in {"SL", "SAFETY_CAP"})
    return {
        "n_trades": len(trades),
        "win_rate": float((pnls > 0).mean() * 100),
        "avg_return": float(pnls.mean()),
        "median_return": float(np.median(pnls)),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_bars": float(np.mean([t.bars_held for t in trades])),
        "sl_pct": float(sl_count / len(trades) * 100),
        "rev_pct": float(rev_count / len(trades) * 100),
        "max_dd_trade": float(pnls.min()),
        "expectancy": float(pnls.mean()),
        "total_return": float(pnls.sum()),
    }


# ───────────────────────── Runner ────────────────────────────────────────
def load_universe() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(OHLCV_PATH)
    df = df[(df["date"] >= "2021-01-01") & (df["date"] <= TO_DATE)]
    df = df[df["exchange"] == "HSX"] if "HSX" in df["exchange"].unique() else df
    # Keep liquid only — avg volume last 60d ≥ 500k
    last_60 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]
    liq = last_60.groupby("symbol")["volume"].mean()
    liquid_syms = liq[liq >= 500_000].index.tolist()
    df = df[df["symbol"].isin(liquid_syms)]
    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) >= 200:
            out[sym] = g
    return out


def run_combo(args):
    entry_name, exit_name, universe = args
    entry_fn = ENTRY_RULES[entry_name]
    exit_fn = EXIT_RULES[exit_name]
    all_trades: list[TradeRec] = []
    for sym, df in universe.items():
        df_window = df[df["date"] >= FROM_DATE].reset_index(drop=True)
        # Include lookback context — keep last 60 bars before FROM_DATE
        if len(df_window) < 100:
            continue
        # Re-attach lookback: take from full df starting 60 bars before from_date
        from_idx = df.index[df["date"] >= FROM_DATE]
        if len(from_idx) == 0:
            continue
        start = max(0, from_idx[0] - 60)
        df_full = df.iloc[start:].reset_index(drop=True)
        trades = backtest_symbol(df_full, sym, entry_fn, exit_fn, exit_name)
        # Drop trades whose entry pre-dates FROM_DATE (lookback bleed)
        trades = [t for t in trades if t.entry_date >= FROM_DATE]
        all_trades.extend(trades)
    return entry_name, exit_name, all_trades


def main():
    print(f"[ichimoku] Loading universe from {OHLCV_PATH.name}")
    universe = load_universe()
    print(f"[ichimoku] {len(universe)} symbols, period {FROM_DATE} - {TO_DATE}")

    summary_rows = []
    combos = [(e, x, universe) for e in ENTRY_RULES for x in EXIT_RULES]
    print(f"[ichimoku] Testing {len(combos)} combos (entry × exit)")

    for i, combo in enumerate(combos, 1):
        entry_name, exit_name, trades = run_combo(combo)
        stats = summarize(trades)
        stats["entry"] = entry_name
        stats["exit"] = exit_name
        summary_rows.append(stats)
        print(f"  [{i:2d}/{len(combos)}] {entry_name:22s} x {exit_name:18s}  "
              f"n={stats['n_trades']:4d}  WR={stats['win_rate']:5.1f}%  "
              f"AvgRet={stats['avg_return']:+5.2f}%  PF={stats['profit_factor']:.2f}")
        # Save individual trades
        if trades:
            pd.DataFrame([asdict(t) for t in trades]).to_csv(
                OUT_DIR / f"{entry_name}_{exit_name}_trades.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    cols = ["entry", "exit", "n_trades", "win_rate", "avg_return", "median_return",
            "profit_factor", "expectancy", "total_return", "avg_bars",
            "sl_pct", "rev_pct", "max_dd_trade"]
    summary_df = summary_df[cols].sort_values("profit_factor", ascending=False)
    out_path = OUT_DIR / "summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n[ichimoku] Summary written: {out_path}")
    print("\nTop 5 combos by profit factor:")
    print(summary_df.head(5).to_string(index=False))
    print("\nBottom 5:")
    print(summary_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
