"""Monthly Wick v2 — edge upgrades + period analysis.

Baseline (from v1): V3_MONTHLY_UP x E2_TRAILING — PF 1.36, AvgRet +1.49%, n=187.

4 upgrades tested independently + stacked:
  U1_BASELINE       — V3_MONTHLY_UP (monthly trend filter only)
  U2_VOL_CONFIRM    — + bounce volume >= 1.2x MA20 (real buyers stepping in)
  U3_VNI_REGIME     — + VNI not in DOWNTREND (close > MA50 or MA50 rising)
  U4_TIGHT_SL       — baseline + SL = max(wick_low_buf, entry - 1.5*ATR), capped 5%
  U5_STACK_ALL      — vol + VNI + tight SL combined

Then break out edge by:
  - Calendar year (2022 / 2023 / 2024 / 2025-26)
  - VNI regime at entry (UPTREND / SIDEWAY / DOWNTREND)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OHLCV_PATH = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
INDEX_PATH = ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"
OUT_DIR = ROOT / "backtest_results" / "monthly_wick_research" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FROM_DATE = "2022-01-01"
TO_DATE = "2026-05-01"
MIN_AVG_VOL_20D = 200_000
INITIAL_SL_BUFFER = 0.015
MAX_HOLD_BARS = 60
RALLY_THRESHOLD = 0.05


# ───────────────────────── Monthly wick detection ────────────────────────
def build_monthly(df: pd.DataFrame) -> pd.DataFrame:
    d = df.set_index("date")
    m = d.resample("MS").agg({"open": "first", "high": "max", "low": "min",
                              "close": "last", "volume": "sum"}).dropna()
    return m.reset_index()


def detect_dao_gam(m: pd.DataFrame) -> pd.DataFrame:
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
        & (lower_wick / rng >= 0.4)
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
    out["vol_ratio"] = out["volume"] / out["vol_ma20"]
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, 1e-9))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    return out


def attach_active_wick(df: pd.DataFrame, monthly: pd.DataFrame, max_age_months: int = 6) -> pd.DataFrame:
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

    for idx in out.index:
        cur_date = out.at[idx, "date"]
        eligible = monthly[monthly["month_end"] < cur_date]
        if eligible.empty:
            continue

        last_m = eligible.iloc[-1]
        if pd.notna(last_m["ma6"]):
            out.at[idx, "monthly_trend_up"] = bool(last_m["close"] >= last_m["ma6"])

        dgs = eligible[eligible["is_dao_gam"]]
        if dgs.empty:
            continue

        for _, dg in dgs.iloc[::-1].iterrows():
            age_months = (cur_date - dg["month_end"]).days / 30.5
            if age_months > max_age_months:
                break
            wick_low = float(dg["wick_low"])
            wick_top = float(dg["wick_top"])
            after = eligible[eligible["date"] > dg["date"]]
            if not after["close"].empty:
                if (after["close"] < wick_low).any():
                    continue
                rallied = (after["close"] >= wick_top * (1 + RALLY_THRESHOLD)).any()
            else:
                rallied = False
            out.at[idx, "wick_low"] = wick_low
            out.at[idx, "wick_top"] = wick_top
            out.at[idx, "wick_age_months"] = age_months
            out.at[idx, "wick_rallied"] = rallied
            break
    return out


# ───────────────────────── VNI regime ────────────────────────────────────
def load_vni() -> pd.DataFrame:
    idx = pd.read_parquet(INDEX_PATH)
    vni = idx[idx["symbol"] == "VNINDEX"].sort_values("date").reset_index(drop=True)
    vni["ma20"] = vni["close"].rolling(20).mean()
    vni["ma50"] = vni["close"].rolling(50).mean()
    vni["ma50_slope"] = vni["ma50"].diff(10)  # 10-day slope of MA50
    # Regime: UPTREND = close > MA50 AND MA50 rising
    #         DOWNTREND = close < MA50 AND MA50 falling
    #         else SIDEWAY
    def _regime(row):
        if pd.isna(row["ma50"]):
            return "UNKNOWN"
        if row["close"] > row["ma50"] and row["ma50_slope"] > 0:
            return "UPTREND"
        if row["close"] < row["ma50"] and row["ma50_slope"] < 0:
            return "DOWNTREND"
        return "SIDEWAY"
    vni["regime"] = vni.apply(_regime, axis=1)
    return vni[["date", "close", "ma50", "ma50_slope", "regime"]].rename(
        columns={"close": "vni_close", "ma50": "vni_ma50",
                 "ma50_slope": "vni_ma50_slope", "regime": "vni_regime"})


# ───────────────────────── Entry/Exit rules ──────────────────────────────
def _retest_bar(row) -> bool:
    if pd.isna(row["wick_low"]) or pd.isna(row["wick_top"]):
        return False
    if not row["wick_rallied"]:
        return False
    if not (row["low"] <= row["wick_top"] * 1.02 and row["low"] >= row["wick_low"] * 0.97):
        return False
    rng = max(row["high"] - row["low"], 1e-6)
    close_pos = (row["close"] - row["low"]) / rng
    return bool(row["close"] > row["open"] and close_pos >= 0.55)


def entry_u1_baseline(row) -> bool:
    return _retest_bar(row) and bool(row["monthly_trend_up"])


def entry_u2_vol(row) -> bool:
    if not entry_u1_baseline(row):
        return False
    return bool(row["vol_ratio"] >= 1.2)


def entry_u3_vni(row) -> bool:
    if not entry_u1_baseline(row):
        return False
    return row.get("vni_regime") != "DOWNTREND"


def entry_u5_stack(row) -> bool:
    if not entry_u1_baseline(row):
        return False
    if not (row["vol_ratio"] >= 1.2):
        return False
    return row.get("vni_regime") != "DOWNTREND"


# U4 is same entry as U1 but different SL — handled at SL computation stage
ENTRIES = {
    "U1_BASELINE":    (entry_u1_baseline, "wick"),
    "U2_VOL_CONFIRM": (entry_u2_vol,      "wick"),
    "U3_VNI_REGIME":  (entry_u3_vni,      "wick"),
    "U4_TIGHT_SL":    (entry_u1_baseline, "atr"),  # tight ATR-based SL
    "U5_STACK_ALL":   (entry_u5_stack,    "atr"),
}


def exit_trailing(row, prev, state):
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


# ───────────────────────── Backtest ──────────────────────────────────────
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
    vni_regime_at_entry: str
    year: int


def backtest_symbol(
    df: pd.DataFrame, symbol: str, entry_fn: Callable, sl_mode: str,
) -> list[TradeRec]:
    monthly = detect_dao_gam(build_monthly(df))
    df = add_indicators(df)
    df = attach_active_wick(df, monthly, 6)
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

        if pos is not None:
            pos["peak_close"] = max(pos["peak_close"], float(row["close"]))
            pos["lows"].append(float(row["low"]))
            bars_held = i - pos["entry_idx"]
            done, reason = exit_trailing(row, prev, pos)
            if done:
                if reason == "SL":
                    exit_px = min(pos["initial_sl"], float(row["open"]))
                elif reason == "TSL":
                    exit_px = min(pos["dynamic_sl"], float(row["open"]))
                else:
                    exit_px = float(row["close"])
                pnl = (exit_px - pos["entry_price"]) / pos["entry_price"] * 100
                trades.append(TradeRec(
                    symbol, pos["entry_date"], bar_date, pos["entry_price"], exit_px,
                    pnl, bars_held, reason, pos["vni_regime"],
                    int(pos["entry_date"][:4])))
                pos = None
                continue
            if bars_held >= MAX_HOLD_BARS:
                pnl = (float(row["close"]) - pos["entry_price"]) / pos["entry_price"] * 100
                trades.append(TradeRec(
                    symbol, pos["entry_date"], bar_date, pos["entry_price"], float(row["close"]),
                    pnl, bars_held, "MAX_HOLD", pos["vni_regime"],
                    int(pos["entry_date"][:4])))
                pos = None
                continue

        if pos is None and i + 1 < n:
            try:
                fire = bool(entry_fn(row))
            except Exception:
                fire = False
            if fire:
                entry_px = float(df.iloc[i + 1]["open"])
                if entry_px <= 0:
                    continue
                atr = float(row["atr14"]) if not pd.isna(row["atr14"]) else entry_px * 0.02
                wick_low = float(row["wick_low"])

                if sl_mode == "atr":
                    # Tight ATR-based SL — 1.5*ATR or 2% min, 5% cap
                    sl_dist = min(max(1.5 * atr, entry_px * 0.02), entry_px * 0.05)
                    sl_level = entry_px - sl_dist
                else:
                    # Original wick-based SL
                    sl_level = max(wick_low * (1 - INITIAL_SL_BUFFER), entry_px * 0.92)

                if sl_level >= entry_px:
                    continue

                pos = {
                    "entry_date": str(df.iloc[i + 1]["date"])[:10],
                    "entry_idx": i + 1,
                    "entry_price": entry_px,
                    "entry_atr": atr,
                    "initial_sl": sl_level,
                    "dynamic_sl": sl_level,
                    "peak_close": entry_px,
                    "lows": [float(df.iloc[i + 1]["low"])],
                    "vni_regime": str(row.get("vni_regime", "UNKNOWN")),
                }
    return trades


# ───────────────────────── Metrics ───────────────────────────────────────
def summarize(trades: list[TradeRec], label: str = "") -> dict:
    if not trades:
        return {"label": label, "n_trades": 0, "win_rate": 0.0, "avg_return": 0.0,
                "median_return": 0.0, "profit_factor": 0.0, "total_return": 0.0,
                "avg_bars": 0.0, "sl_pct": 0.0, "avg_winner": 0.0, "avg_loser": 0.0,
                "max_dd_trade": 0.0}
    pnls = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 1e-9
    sl_count = sum(1 for t in trades if t.exit_reason == "SL")
    return {
        "label": label,
        "n_trades": len(trades),
        "win_rate": float((pnls > 0).mean() * 100),
        "avg_return": float(pnls.mean()),
        "median_return": float(np.median(pnls)),
        "profit_factor": float(gross_win / gross_loss),
        "total_return": float(pnls.sum()),
        "avg_bars": float(np.mean([t.bars_held for t in trades])),
        "sl_pct": float(sl_count / len(trades) * 100),
        "avg_winner": float(wins.mean()) if len(wins) else 0.0,
        "avg_loser": float(losses.mean()) if len(losses) else 0.0,
        "max_dd_trade": float(pnls.min()),
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
    print(f"[wick-v2] Loading data")
    universe = load_universe()
    vni = load_vni()
    print(f"[wick-v2] {len(universe)} symbols, period {FROM_DATE} - {TO_DATE}")

    overall_rows = []
    by_year_rows = []
    by_regime_rows = []

    for variant_name, (entry_fn, sl_mode) in ENTRIES.items():
        print(f"\n[wick-v2] Running {variant_name} (SL={sl_mode})")
        all_trades: list[TradeRec] = []
        for sym, df in universe.items():
            # Merge VNI regime into daily df
            df_with_vni = df.merge(vni, on="date", how="left")
            try:
                trades = backtest_symbol(df_with_vni, sym, entry_fn, sl_mode)
            except Exception as e:
                print(f"  [warn] {sym}: {e}")
                continue
            trades = [t for t in trades if t.entry_date >= FROM_DATE]
            all_trades.extend(trades)

        # Overall
        stats = summarize(all_trades, label=variant_name)
        overall_rows.append(stats)
        print(f"  OVERALL  n={stats['n_trades']:4d}  WR={stats['win_rate']:5.1f}%  "
              f"Avg={stats['avg_return']:+5.2f}%  PF={stats['profit_factor']:.2f}  "
              f"AvgWin={stats['avg_winner']:+5.2f}  AvgLoss={stats['avg_loser']:+5.2f}")

        # By year
        for yr in sorted(set(t.year for t in all_trades)):
            yr_trades = [t for t in all_trades if t.year == yr]
            s = summarize(yr_trades, label=f"{variant_name}_{yr}")
            s["variant"] = variant_name
            s["year"] = yr
            by_year_rows.append(s)
            print(f"    {yr}  n={s['n_trades']:3d}  WR={s['win_rate']:5.1f}%  "
                  f"Avg={s['avg_return']:+5.2f}%  PF={s['profit_factor']:.2f}")

        # By VNI regime at entry
        for reg in ["UPTREND", "SIDEWAY", "DOWNTREND"]:
            reg_trades = [t for t in all_trades if t.vni_regime_at_entry == reg]
            s = summarize(reg_trades, label=f"{variant_name}_{reg}")
            s["variant"] = variant_name
            s["vni_regime"] = reg
            by_regime_rows.append(s)
            print(f"    VNI={reg:9s}  n={s['n_trades']:3d}  WR={s['win_rate']:5.1f}%  "
                  f"Avg={s['avg_return']:+5.2f}%  PF={s['profit_factor']:.2f}")

        if all_trades:
            pd.DataFrame([asdict(t) for t in all_trades]).to_csv(
                OUT_DIR / f"{variant_name}_trades.csv", index=False)

    pd.DataFrame(overall_rows).to_csv(OUT_DIR / "overall_summary.csv", index=False)
    pd.DataFrame(by_year_rows).to_csv(OUT_DIR / "by_year.csv", index=False)
    pd.DataFrame(by_regime_rows).to_csv(OUT_DIR / "by_regime.csv", index=False)
    print(f"\n[wick-v2] Reports in {OUT_DIR}")


if __name__ == "__main__":
    main()
