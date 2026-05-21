"""Ichimoku v2 — tunings on top of PERFECT_BULLISH × TK_DEATH_CROSS.

Baseline (from v1): PF 1.57, AvgRet +1.63%, WR 28%, n=1362.

8 tunings + period analysis (year + VNI regime).
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
OUT_DIR = ROOT / "backtest_results" / "ichimoku_research" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FROM_DATE = "2022-01-01"
TO_DATE = "2026-05-01"
MIN_AVG_VOL_20D = 200_000
MAX_HOLD_BARS = 120


# ───────────────────────── Ichimoku ──────────────────────────────────────
def ichimoku(df: pd.DataFrame, t_p: int = 9, k_p: int = 26, b_p: int = 52,
             shift_p: int = 26) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    tenkan = (h.rolling(t_p).max() + l.rolling(t_p).min()) / 2
    kijun = (h.rolling(k_p).max() + l.rolling(k_p).min()) / 2
    senkou_a_raw = (tenkan + kijun) / 2
    senkou_b_raw = (h.rolling(b_p).max() + l.rolling(b_p).min()) / 2
    senkou_a = senkou_a_raw.shift(shift_p)
    senkou_b = senkou_b_raw.shift(shift_p)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bot = np.minimum(senkou_a, senkou_b)
    chikou_free = c > c.shift(shift_p)
    future_green = senkou_a_raw > senkou_b_raw
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out = df.copy()
    out["tenkan"] = tenkan
    out["kijun"] = kijun
    out["senkou_a"] = senkou_a
    out["senkou_b"] = senkou_b
    out["cloud_top"] = cloud_top
    out["cloud_bot"] = cloud_bot
    out["cloud_thickness_pct"] = (cloud_top - cloud_bot) / c * 100
    out["chikou_free"] = chikou_free
    out["future_green"] = future_green
    out["atr14"] = tr.rolling(14).mean()
    out["vol_ma20"] = df["volume"].rolling(20).mean()
    return out


# VNI regime
def load_vni() -> pd.DataFrame:
    idx = pd.read_parquet(INDEX_PATH)
    vni = idx[idx["symbol"] == "VNINDEX"].sort_values("date").reset_index(drop=True)
    vni["ma50"] = vni["close"].rolling(50).mean()
    vni["ma50_slope"] = vni["ma50"].diff(10)
    def _regime(r):
        if pd.isna(r["ma50"]):
            return "UNKNOWN"
        if r["close"] > r["ma50"] and r["ma50_slope"] > 0:
            return "UPTREND"
        if r["close"] < r["ma50"] and r["ma50_slope"] < 0:
            return "DOWNTREND"
        return "SIDEWAY"
    vni["regime"] = vni.apply(_regime, axis=1)
    return vni[["date", "regime"]].rename(columns={"regime": "vni_regime"})


# ───────────────────────── Entry: PERFECT_BULLISH ────────────────────────
def perfect_bullish(row, prev) -> bool:
    """Full bullish alignment — entry on first bar all conditions hold."""
    aligned_now = (
        row["tenkan"] > row["kijun"]
        and row["close"] > row["tenkan"]
        and row["close"] > row["cloud_top"]
        and row["senkou_a"] > row["senkou_b"]
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


# ───────────────────────── Tunings ───────────────────────────────────────
@dataclass
class TuneConfig:
    name: str
    sl_mode: str = "atr_5pct"      # "atr_5pct" or "atr_3pct"
    vni_filter: str = "any"        # "any", "uptrend_only"
    no_chase: bool = False          # skip if close > kijun*1.05
    cloud_thick_min: float = 0.0   # min cloud thickness %
    exit_confirm: bool = False     # TK_DEATH_CROSS + close < kijun
    ichimoku_params: tuple = (9, 26, 52, 26)  # tenkan, kijun, senkou_b, shift


TUNES = [
    TuneConfig("T1_BASELINE"),
    TuneConfig("T2_TIGHT_SL_3PCT", sl_mode="atr_3pct"),
    TuneConfig("T3_VNI_UPTREND", vni_filter="uptrend_only"),
    TuneConfig("T4_NO_CHASE", no_chase=True),
    TuneConfig("T5_CLOUD_THICK", cloud_thick_min=1.5),
    TuneConfig("T6_EXIT_CONFIRM", exit_confirm=True),
    TuneConfig("T7_PARAM_VN", ichimoku_params=(7, 22, 44, 22)),
    # T8 stack — will set after seeing top 2-3
    TuneConfig("T8_STACK", sl_mode="atr_3pct", vni_filter="uptrend_only",
               no_chase=True, cloud_thick_min=1.5),
]


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


def backtest_symbol(df: pd.DataFrame, symbol: str, cfg: TuneConfig) -> list[TradeRec]:
    t_p, k_p, b_p, shift_p = cfg.ichimoku_params
    df = ichimoku(df, t_p, k_p, b_p, shift_p).reset_index(drop=True)
    trades: list[TradeRec] = []
    pos: Optional[dict] = None
    n = len(df)
    for i in range(60, n - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        if pd.isna(row["senkou_b"]) or pd.isna(row["kijun"]):
            continue
        if pd.isna(row["vol_ma20"]) or row["vol_ma20"] < MIN_AVG_VOL_20D:
            continue
        bar_date = str(row["date"])[:10]

        # ── Exit ─────────────────────────────────────────────────────────
        if pos is not None:
            bars_held = i - pos["entry_idx"]

            # Hard SL
            if float(row["low"]) <= pos["initial_sl"]:
                exit_px = min(pos["initial_sl"], float(row["open"]))
                pnl = (exit_px - pos["entry_price"]) / pos["entry_price"] * 100
                trades.append(TradeRec(
                    symbol, pos["entry_date"], bar_date, pos["entry_price"], exit_px,
                    pnl, bars_held, "SL", pos["vni_regime"], int(pos["entry_date"][:4])))
                pos = None
                continue

            # TK death cross exit (with optional confirm)
            tk_death = prev["tenkan"] >= prev["kijun"] and row["tenkan"] < row["kijun"]
            if tk_death:
                exit_ok = True
                if cfg.exit_confirm:
                    exit_ok = bool(row["close"] < row["kijun"])
                if exit_ok and i + 1 < n:
                    next_open = float(df.iloc[i + 1]["open"])
                    next_date = str(df.iloc[i + 1]["date"])[:10]
                    pnl = (next_open - pos["entry_price"]) / pos["entry_price"] * 100
                    trades.append(TradeRec(
                        symbol, pos["entry_date"], next_date, pos["entry_price"], next_open,
                        pnl, bars_held + 1, "TK_DEATH",
                        pos["vni_regime"], int(pos["entry_date"][:4])))
                    pos = None
                    continue

            if bars_held >= MAX_HOLD_BARS:
                pnl = (float(row["close"]) - pos["entry_price"]) / pos["entry_price"] * 100
                trades.append(TradeRec(
                    symbol, pos["entry_date"], bar_date, pos["entry_price"], float(row["close"]),
                    pnl, bars_held, "MAX_HOLD", pos["vni_regime"], int(pos["entry_date"][:4])))
                pos = None
                continue

        # ── Entry ────────────────────────────────────────────────────────
        if pos is None and i + 1 < n:
            try:
                fire = perfect_bullish(row, prev)
            except Exception:
                fire = False
            if not fire:
                continue

            # Apply filters
            if cfg.vni_filter == "uptrend_only" and row.get("vni_regime") != "UPTREND":
                continue
            if cfg.no_chase and row["close"] > row["kijun"] * 1.05:
                continue
            if cfg.cloud_thick_min > 0:
                ct = row.get("cloud_thickness_pct", 0.0)
                if pd.isna(ct) or ct < cfg.cloud_thick_min:
                    continue

            entry_px = float(df.iloc[i + 1]["open"])
            if entry_px <= 0:
                continue
            atr = float(row["atr14"]) if not pd.isna(row["atr14"]) else entry_px * 0.02

            if cfg.sl_mode == "atr_3pct":
                sl_dist = min(max(1.5 * atr, entry_px * 0.015), entry_px * 0.03)
            else:  # atr_5pct
                sl_dist = min(max(1.5 * atr, entry_px * 0.02), entry_px * 0.05)
            sl_level = entry_px - sl_dist
            if sl_level >= entry_px:
                continue

            pos = {
                "entry_date": str(df.iloc[i + 1]["date"])[:10],
                "entry_idx": i + 1,
                "entry_price": entry_px,
                "initial_sl": sl_level,
                "entry_atr": atr,
                "vni_regime": str(row.get("vni_regime", "UNKNOWN")),
            }
    return trades


def summarize(trades: list[TradeRec]) -> dict:
    if not trades:
        return {"n_trades": 0, "win_rate": 0.0, "avg_return": 0.0, "profit_factor": 0.0,
                "total_return": 0.0, "avg_winner": 0.0, "avg_loser": 0.0,
                "sl_pct": 0.0, "max_dd_trade": 0.0}
    pnls = np.array([t.pnl_pct for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gw = wins.sum() if len(wins) else 0.0
    gl = -losses.sum() if len(losses) else 1e-9
    sl_count = sum(1 for t in trades if t.exit_reason == "SL")
    return {
        "n_trades": len(trades),
        "win_rate": float((pnls > 0).mean() * 100),
        "avg_return": float(pnls.mean()),
        "profit_factor": float(gw / gl),
        "total_return": float(pnls.sum()),
        "avg_winner": float(wins.mean()) if len(wins) else 0.0,
        "avg_loser": float(losses.mean()) if len(losses) else 0.0,
        "sl_pct": float(sl_count / len(trades) * 100),
        "max_dd_trade": float(pnls.min()),
    }


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
    print("[ich-v2] Loading data")
    universe = load_universe()
    vni = load_vni()
    print(f"[ich-v2] {len(universe)} symbols, period {FROM_DATE} - {TO_DATE}")

    overall_rows = []
    by_year_rows = []
    by_regime_rows = []

    for cfg in TUNES:
        print(f"\n[ich-v2] Running {cfg.name}")
        all_trades: list[TradeRec] = []
        for sym, df in universe.items():
            df_v = df.merge(vni, on="date", how="left")
            try:
                trades = backtest_symbol(df_v, sym, cfg)
            except Exception as e:
                print(f"  [warn] {sym}: {e}")
                continue
            trades = [t for t in trades if t.entry_date >= FROM_DATE]
            all_trades.extend(trades)

        s = summarize(all_trades)
        s["tune"] = cfg.name
        overall_rows.append(s)
        print(f"  OVERALL  n={s['n_trades']:4d}  WR={s['win_rate']:5.1f}%  "
              f"Avg={s['avg_return']:+5.2f}%  PF={s['profit_factor']:.2f}  "
              f"AvgWin={s['avg_winner']:+5.2f}  AvgLoss={s['avg_loser']:+5.2f}  "
              f"SL%={s['sl_pct']:.0f}")

        for yr in sorted(set(t.year for t in all_trades)):
            yt = [t for t in all_trades if t.year == yr]
            ys = summarize(yt)
            ys["tune"] = cfg.name
            ys["year"] = yr
            by_year_rows.append(ys)
            print(f"    {yr}  n={ys['n_trades']:4d}  WR={ys['win_rate']:5.1f}%  "
                  f"Avg={ys['avg_return']:+5.2f}%  PF={ys['profit_factor']:.2f}")

        for reg in ["UPTREND", "SIDEWAY", "DOWNTREND"]:
            rt = [t for t in all_trades if t.vni_regime_at_entry == reg]
            rs = summarize(rt)
            rs["tune"] = cfg.name
            rs["vni_regime"] = reg
            by_regime_rows.append(rs)
            print(f"    VNI={reg:9s}  n={rs['n_trades']:4d}  "
                  f"WR={rs['win_rate']:5.1f}%  Avg={rs['avg_return']:+5.2f}%  "
                  f"PF={rs['profit_factor']:.2f}")

        if all_trades:
            pd.DataFrame([asdict(t) for t in all_trades]).to_csv(
                OUT_DIR / f"{cfg.name}_trades.csv", index=False)

    pd.DataFrame(overall_rows).to_csv(OUT_DIR / "overall.csv", index=False)
    pd.DataFrame(by_year_rows).to_csv(OUT_DIR / "by_year.csv", index=False)
    pd.DataFrame(by_regime_rows).to_csv(OUT_DIR / "by_regime.csv", index=False)
    print(f"\n[ich-v2] Reports in {OUT_DIR}")


if __name__ == "__main__":
    main()
