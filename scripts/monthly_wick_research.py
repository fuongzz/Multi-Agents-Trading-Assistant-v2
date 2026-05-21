"""Monthly Wick (Dao Gam) bounce research — VN trader heuristic.

Idea: Sau khi giá tạo "dao gam" tháng (long lower-wick rejection candle),
zone bóng dưới [monthly_low, min(open,close)] trở thành "demand cứng".
Khi giá pull-back về zone này trên khung daily, thường có bounce.

Chiến lược:
  Entry — daily retest vào wick zone + bullish reversal + monthly chưa phá zone
  Exit  — SL dưới wick_low | trailing swing | fixed 2R TP

5 entry variants x 3 exit variants = 15 combos. Output ra summary.csv.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OHLCV_PATH = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
OUT_DIR = ROOT / "backtest_results" / "monthly_wick_research"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FROM_DATE = "2022-01-01"
TO_DATE = "2026-05-01"
MIN_AVG_VOL_20D = 200_000
INITIAL_SL_BUFFER = 0.015  # SL = wick_low x (1 - buffer)
MAX_HOLD_BARS = 60
RALLY_THRESHOLD = 0.05  # need 5% rally above wick_top after the wick before retest counts


# ───────────────────────── Monthly wick detection ────────────────────────
def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily to monthly OHLCV."""
    d = df.set_index("date")
    m = d.resample("MS").agg({"open": "first", "high": "max", "low": "min",
                              "close": "last", "volume": "sum"}).dropna()
    return m.reset_index()


def detect_dao_gam(m: pd.DataFrame) -> pd.DataFrame:
    """Mark each monthly bar — is it a dao gam? Return monthly df with wick zone fields."""
    body = (m["close"] - m["open"]).abs()
    body_safe = body.replace(0, 1e-9)
    upper_wick = m["high"] - m[["open", "close"]].max(axis=1)
    lower_wick = m[["open", "close"]].min(axis=1) - m["low"]
    rng = (m["high"] - m["low"]).replace(0, 1e-9)
    close_pos = (m["close"] - m["low"]) / rng

    is_dao_gam = (
        (lower_wick >= 2.0 * body_safe)
        & (lower_wick >= 1.5 * upper_wick.replace(0, 1e-9))
        & (close_pos >= 0.5)
        & (lower_wick / rng >= 0.4)  # wick is at least 40% of total range
    )
    out = m.copy()
    out["is_dao_gam"] = is_dao_gam
    out["wick_low"] = m["low"]
    out["wick_top"] = m[["open", "close"]].min(axis=1)
    return out


# ───────────────────────── Indicators ────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]
    h, l = out["high"], out["low"]
    out["vol_ma20"] = out["volume"].rolling(20).mean()
    # RSI 14
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, 1e-9))
    # ATR 14
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    return out


def attach_active_wick(df: pd.DataFrame, monthly: pd.DataFrame, max_age_months: int) -> pd.DataFrame:
    """For each daily bar, find the most recent VALID dao gam wick zone.

    Valid =
      - The dao gam month has ended (we don't peek into in-progress wick)
      - Within max_age_months
      - Monthly close (since dao gam) hasn't broken below wick_low
      - Some monthly close after dao gam closed ≥ wick_top * (1 + RALLY_THRESHOLD)
    """
    out = df.copy()
    out["wick_low"] = np.nan
    out["wick_top"] = np.nan
    out["wick_age_months"] = np.nan
    out["wick_rallied"] = False
    out["monthly_trend_up"] = False

    if monthly.empty:
        return out

    monthly = monthly.copy()
    monthly["month_end"] = monthly["date"] + pd.offsets.MonthEnd(0)
    monthly["ma6"] = monthly["close"].rolling(6).mean()

    # Walk daily bars and pick the freshest valid wick
    for idx in out.index:
        cur_date = out.at[idx, "date"]
        # Only consider monthly candles whose month has fully closed before today
        eligible = monthly[monthly["month_end"] < cur_date]
        if eligible.empty:
            continue

        # Monthly trend
        last_m = eligible.iloc[-1]
        if pd.notna(last_m["ma6"]):
            out.at[idx, "monthly_trend_up"] = bool(last_m["close"] >= last_m["ma6"])

        # Search backward through dao gam candles
        dgs = eligible[eligible["is_dao_gam"]]
        if dgs.empty:
            continue

        for _, dg in dgs.iloc[::-1].iterrows():
            age_days = (cur_date - dg["month_end"]).days
            age_months = age_days / 30.5
            if age_months > max_age_months:
                break  # older ones only get more stale
            wick_low = float(dg["wick_low"])
            wick_top = float(dg["wick_top"])
            # Subsequent monthly closes (after this dao gam)
            after = eligible[eligible["date"] > dg["date"]]
            if not after["close"].empty:
                if (after["close"] < wick_low).any():
                    continue  # zone violated
                rallied = (after["close"] >= wick_top * (1 + RALLY_THRESHOLD)).any()
            else:
                rallied = False
            out.at[idx, "wick_low"] = wick_low
            out.at[idx, "wick_top"] = wick_top
            out.at[idx, "wick_age_months"] = age_months
            out.at[idx, "wick_rallied"] = rallied
            break
    return out


# ───────────────────────── Entry rules ───────────────────────────────────
def _basic_retest(row, prev) -> bool:
    if pd.isna(row["wick_low"]) or pd.isna(row["wick_top"]):
        return False
    if not row["wick_rallied"]:
        return False
    # Retest: daily low pierces into wick zone (with small upper buffer)
    if not (row["low"] <= row["wick_top"] * 1.02 and row["low"] >= row["wick_low"] * 0.97):
        return False
    # Bullish reversal candle
    rng = max(row["high"] - row["low"], 1e-6)
    close_pos = (row["close"] - row["low"]) / rng
    bullish = row["close"] > row["open"] and close_pos >= 0.55
    return bool(bullish)


def entry_v1_basic(row, prev) -> bool:
    return _basic_retest(row, prev)


def entry_v2_oversold(row, prev) -> bool:
    if not _basic_retest(row, prev):
        return False
    return bool(row["rsi14"] < 40)


def entry_v3_monthly_up(row, prev) -> bool:
    if not _basic_retest(row, prev):
        return False
    return bool(row["monthly_trend_up"])


def entry_v4_fresh(row, prev) -> bool:
    if not _basic_retest(row, prev):
        return False
    return bool(row["wick_age_months"] <= 3)


def entry_v5_strict_all(row, prev) -> bool:
    return (
        _basic_retest(row, prev)
        and bool(row["rsi14"] < 45)
        and bool(row["monthly_trend_up"])
        and bool(row["wick_age_months"] <= 3)
    )


ENTRIES = {
    "V1_BASIC":       entry_v1_basic,
    "V2_OVERSOLD":    entry_v2_oversold,
    "V3_MONTHLY_UP":  entry_v3_monthly_up,
    "V4_FRESH":       entry_v4_fresh,
    "V5_STRICT_ALL":  entry_v5_strict_all,
}


# ───────────────────────── Exits ─────────────────────────────────────────
def exit_e1_wick_sl(row, prev, state) -> tuple[bool, str]:
    """SL fixed dưới wick_low; trailing swing after profit."""
    # Hard SL at initial level
    if row["low"] <= state["initial_sl"]:
        return True, "SL"
    # Trailing swing after gain
    gain = (state["peak_close"] - state["entry_price"]) / state["entry_price"]
    if gain >= 0.05:
        lookback = 20 if gain >= 0.20 else (12 if gain >= 0.10 else 7)
        new_sl = min(state["lows"][-lookback:]) - 0.5 * state["entry_atr"]
        state["dynamic_sl"] = max(state["dynamic_sl"], new_sl)
    if row["low"] <= state["dynamic_sl"]:
        return True, "TSL"
    return False, ""


def exit_e2_trailing(row, prev, state) -> tuple[bool, str]:
    if row["low"] <= state["initial_sl"]:
        return True, "SL"
    gain = (state["peak_close"] - state["entry_price"]) / state["entry_price"]
    if gain >= 0.05:
        lookback = 20 if gain >= 0.20 else (12 if gain >= 0.10 else 7)
        new_sl = min(state["lows"][-lookback:]) - 0.5 * state["entry_atr"]
        state["dynamic_sl"] = max(state["dynamic_sl"], new_sl)
    if row["low"] <= state["dynamic_sl"]:
        return True, "TSL"
    return False, ""


def exit_e3_fixed_rr2(row, prev, state) -> tuple[bool, str]:
    if row["low"] <= state["initial_sl"]:
        return True, "SL"
    if row["high"] >= state["fixed_tp"]:
        return True, "TP_2R"
    return False, ""


EXITS = {
    "E1_WICK_SL":   exit_e1_wick_sl,
    "E2_TRAILING":  exit_e2_trailing,
    "E3_FIXED_RR2": exit_e3_fixed_rr2,
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
    wick_low: float
    wick_top: float
    wick_age_months: float


def backtest_symbol(
    df: pd.DataFrame,
    symbol: str,
    entry_fn: Callable,
    exit_fn: Callable,
    exit_name: str,
    max_age_months: int = 6,
) -> list[TradeRec]:
    monthly = detect_dao_gam(build_monthly(df))
    df = add_indicators(df)
    df = attach_active_wick(df, monthly, max_age_months)
    df = df.reset_index(drop=True)

    trades: list[TradeRec] = []
    pos: Optional[dict] = None
    n = len(df)
    for i in range(60, n - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        if pd.isna(row["vol_ma20"]) or row["vol_ma20"] < MIN_AVG_VOL_20D:
            continue
        bar_date = str(row["date"])[:10]

        # ── Exit ─────────────────────────────────────────────────────────
        if pos is not None:
            pos["peak_close"] = max(pos["peak_close"], float(row["close"]))
            pos["lows"].append(float(row["low"]))
            bars_held = i - pos["entry_idx"]
            done, reason = exit_fn(row, prev, pos)
            if done:
                if reason == "SL":
                    exit_px = min(pos["initial_sl"], float(row["open"]))
                elif reason == "TSL":
                    exit_px = min(pos["dynamic_sl"], float(row["open"]))
                elif reason == "TP_2R":
                    exit_px = max(pos["fixed_tp"], float(row["open"]))
                else:
                    exit_px = float(row["close"])
                pnl = (exit_px - pos["entry_price"]) / pos["entry_price"] * 100
                trades.append(TradeRec(
                    symbol, pos["entry_date"], bar_date, pos["entry_price"], exit_px,
                    pnl, bars_held, reason, pos["wick_low"], pos["wick_top"], pos["wick_age"]))
                pos = None
                continue
            if bars_held >= MAX_HOLD_BARS:
                pnl = (float(row["close"]) - pos["entry_price"]) / pos["entry_price"] * 100
                trades.append(TradeRec(
                    symbol, pos["entry_date"], bar_date, pos["entry_price"], float(row["close"]),
                    pnl, bars_held, "MAX_HOLD", pos["wick_low"], pos["wick_top"], pos["wick_age"]))
                pos = None
                continue

        # ── Entry ────────────────────────────────────────────────────────
        if pos is None and i + 1 < n:
            try:
                fire = bool(entry_fn(row, prev))
            except Exception:
                fire = False
            if fire:
                entry_px = float(df.iloc[i + 1]["open"])
                if entry_px <= 0:
                    continue
                wick_low = float(row["wick_low"])
                sl_level = wick_low * (1 - INITIAL_SL_BUFFER)
                if sl_level >= entry_px:
                    continue  # entry would be below SL, skip
                # Cap SL distance at 8% (don't risk too much on wide wicks)
                max_sl = entry_px * 0.92
                sl_level = max(sl_level, max_sl)
                risk = entry_px - sl_level
                fixed_tp = entry_px + 2 * risk
                atr = float(row["atr14"]) if not pd.isna(row["atr14"]) else entry_px * 0.02
                pos = {
                    "entry_date": str(df.iloc[i + 1]["date"])[:10],
                    "entry_idx": i + 1,
                    "entry_price": entry_px,
                    "entry_atr": atr,
                    "initial_sl": sl_level,
                    "dynamic_sl": sl_level,
                    "fixed_tp": fixed_tp,
                    "peak_close": entry_px,
                    "lows": [float(df.iloc[i + 1]["low"])],
                    "wick_low": wick_low,
                    "wick_top": float(row["wick_top"]),
                    "wick_age": float(row["wick_age_months"]),
                }
    return trades


# ───────────────────────── Metrics ───────────────────────────────────────
def summarize(trades: list[TradeRec]) -> dict:
    if not trades:
        return {"n_trades": 0, "win_rate": 0.0, "avg_return": 0.0, "median_return": 0.0,
                "profit_factor": 0.0, "expectancy": 0.0, "total_return": 0.0,
                "avg_bars": 0.0, "sl_pct": 0.0, "max_dd_trade": 0.0,
                "avg_winner": 0.0, "avg_loser": 0.0}
    pnls = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 1e-9
    sl_count = sum(1 for t in trades if t.exit_reason == "SL")
    return {
        "n_trades": len(trades),
        "win_rate": float((pnls > 0).mean() * 100),
        "avg_return": float(pnls.mean()),
        "median_return": float(np.median(pnls)),
        "profit_factor": float(gross_win / gross_loss),
        "expectancy": float(pnls.mean()),
        "total_return": float(pnls.sum()),
        "avg_bars": float(np.mean([t.bars_held for t in trades])),
        "sl_pct": float(sl_count / len(trades) * 100),
        "max_dd_trade": float(pnls.min()),
        "avg_winner": float(wins.mean()) if len(wins) else 0.0,
        "avg_loser": float(losses.mean()) if len(losses) else 0.0,
    }


# ───────────────────────── Runner ────────────────────────────────────────
def load_universe() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(OHLCV_PATH)
    df = df[(df["date"] >= "2020-01-01") & (df["date"] <= TO_DATE)]
    last_90 = df[df["date"] >= df["date"].max() - pd.Timedelta(days=90)]
    liq = last_90.groupby("symbol")["volume"].mean()
    liquid = liq[liq >= 500_000].index.tolist()
    df = df[df["symbol"].isin(liquid)]
    out = {}
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) >= 300:
            out[sym] = g
    return out


def main():
    print(f"[wick] Loading {OHLCV_PATH.name}")
    universe = load_universe()
    print(f"[wick] {len(universe)} symbols, period {FROM_DATE} - {TO_DATE}")

    summary_rows = []
    combos = [(e, x) for e in ENTRIES for x in EXITS]
    print(f"[wick] Testing {len(combos)} combos")

    for ci, (entry_name, exit_name) in enumerate(combos, 1):
        entry_fn = ENTRIES[entry_name]
        exit_fn = EXITS[exit_name]
        all_trades: list[TradeRec] = []
        for sym, df in universe.items():
            try:
                trades = backtest_symbol(df, sym, entry_fn, exit_fn, exit_name)
            except Exception as e:
                print(f"  [warn] {sym}: {e}")
                continue
            trades = [t for t in trades if t.entry_date >= FROM_DATE]
            all_trades.extend(trades)
        stats = summarize(all_trades)
        stats["entry"] = entry_name
        stats["exit"] = exit_name
        summary_rows.append(stats)
        print(f"  [{ci:2d}/{len(combos)}] {entry_name:14s} x {exit_name:13s}  "
              f"n={stats['n_trades']:4d}  WR={stats['win_rate']:5.1f}%  "
              f"Avg={stats['avg_return']:+5.2f}%  PF={stats['profit_factor']:.2f}  "
              f"AvgWin={stats['avg_winner']:+5.2f}  AvgLoss={stats['avg_loser']:+5.2f}")
        if all_trades:
            pd.DataFrame([asdict(t) for t in all_trades]).to_csv(
                OUT_DIR / f"{entry_name}_{exit_name}_trades.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    cols = ["entry", "exit", "n_trades", "win_rate", "avg_return", "median_return",
            "profit_factor", "expectancy", "total_return", "avg_bars",
            "sl_pct", "max_dd_trade", "avg_winner", "avg_loser"]
    summary = summary[cols].sort_values("profit_factor", ascending=False)
    out_path = OUT_DIR / "summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n[wick] Summary: {out_path}")
    print("\nTop 5 by profit factor:")
    print(summary.head(5).to_string(index=False))
    print("\nBottom 3:")
    print(summary.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
