"""Portfolio simulator: replay core3 trades with different sizing policies.

Reads a trades.csv from the existing core3 backtest (each row has signal_date,
entry_date, exit_date, entry_price, exit_price, pnl_pct, edge_score, ...) and
re-simulates portfolio NAV under a chosen sizing rule.

Each trade is "replayed" using `pnl_pct` (per-share return) — actual VND pnl
scales with the new entry size. If portfolio rules block entry (no cash, slot
cap), the trade is skipped.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# ── Sizing rules ───────────────────────────────────────────────────────────

SizerFn = Callable[[dict, "Portfolio"], float]


def sizer_fixed(pct: float) -> SizerFn:
    def f(trade: dict, port: "Portfolio") -> float:
        return pct
    f.__name__ = f"fixed_{int(pct*100)}"
    return f


def sizer_cap(cap: float) -> SizerFn:
    """Take MIN of (cash-split among remaining slots, cap %)."""
    def f(trade: dict, port: "Portfolio") -> float:
        slots_left = port.max_positions - len(port.open)
        if slots_left <= 0:
            return 0.0
        cash_split_pct = port.cash / port.equity() / slots_left
        return min(cash_split_pct, cap)
    f.__name__ = f"cap_{int(cap*100)}"
    return f


def sizer_cash_split() -> SizerFn:
    """Original baseline: divide remaining cash by remaining slots."""
    def f(trade: dict, port: "Portfolio") -> float:
        slots_left = port.max_positions - len(port.open)
        if slots_left <= 0:
            return 0.0
        return port.cash / port.equity() / slots_left
    f.__name__ = "cash_split"
    return f


def sizer_tiered(
    strong_pct: float = 0.40,
    normal_pct: float = 0.25,
    weak_pct: float = 0.15,
    strong_score: float = 80.0,
    weak_score: float = 55.0,
    cap: float = 0.50,
) -> SizerFn:
    """edge_score-driven tiered sizing with hard cap."""
    def f(trade: dict, port: "Portfolio") -> float:
        score = float(trade.get("edge_score") or 50.0)
        if score >= strong_score:
            target = strong_pct
        elif score <= weak_score:
            target = weak_pct
        else:
            target = normal_pct
        slots_left = port.max_positions - len(port.open)
        if slots_left <= 0:
            return 0.0
        # Still respect cash availability
        cash_pct = port.cash / port.equity()
        return min(target, cap, cash_pct)
    f.__name__ = f"tiered_{int(strong_pct*100)}_{int(normal_pct*100)}_{int(weak_pct*100)}"
    return f


def sizer_score_boost(
    base: float = 0.20,
    max_pct: float = 0.40,
    min_score: float = 40.0,
    max_score: float = 100.0,
) -> SizerFn:
    """Linear interpolation between base and max_pct based on edge_score."""
    def f(trade: dict, port: "Portfolio") -> float:
        score = float(trade.get("edge_score") or 50.0)
        norm = max(0.0, min(1.0, (score - min_score) / (max_score - min_score)))
        target = base + (max_pct - base) * norm
        cash_pct = port.cash / port.equity()
        return min(target, cash_pct)
    f.__name__ = f"boost_{int(base*100)}_{int(max_pct*100)}"
    return f


# ── Portfolio ─────────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_value: float
    pnl_pct: float
    setup_type: str


@dataclass
class Portfolio:
    capital: float
    max_positions: int
    cash: float = 0.0
    open: list[OpenPosition] = field(default_factory=list)
    closed: list[dict] = field(default_factory=list)
    equity_history: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    skipped: int = 0
    skipped_no_slot: int = 0
    skipped_no_cash: int = 0

    def __post_init__(self):
        self.cash = self.capital

    def equity(self) -> float:
        # Mark-to-cost: cash + invested cost of open positions
        return self.cash + sum(p.entry_value for p in self.open)

    def record(self, date: pd.Timestamp):
        self.equity_history.append((date, self.equity()))


def simulate(trades_df: pd.DataFrame, sizer: SizerFn,
             capital: float = 1e8, max_positions: int = 5) -> Portfolio:
    port = Portfolio(capital=capital, max_positions=max_positions)
    df = trades_df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    # Build event list: exits before entries on same date
    events: list[tuple[pd.Timestamp, str, dict]] = []
    for _, r in df.iterrows():
        events.append((r["entry_date"], "ENTRY", r.to_dict()))
        events.append((r["exit_date"], "EXIT", r.to_dict()))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "EXIT" else 1))

    for date, etype, trade in events:
        if etype == "EXIT":
            # Find matching open position (by symbol + entry_date)
            sym = trade["symbol"]
            entry_d = trade["entry_date"]
            idx = next((i for i, p in enumerate(port.open)
                        if p.symbol == sym and p.entry_date == entry_d), None)
            if idx is None:
                continue
            pos = port.open.pop(idx)
            realized_pnl = pos.entry_value * pos.pnl_pct
            port.cash += pos.entry_value + realized_pnl
            port.closed.append({
                "symbol": pos.symbol,
                "entry_date": pos.entry_date,
                "exit_date": pos.exit_date,
                "entry_value": pos.entry_value,
                "pnl": realized_pnl,
                "pnl_pct": pos.pnl_pct,
                "setup_type": pos.setup_type,
            })
            port.record(date)
        else:  # ENTRY
            if len(port.open) >= port.max_positions:
                port.skipped += 1
                port.skipped_no_slot += 1
                continue
            size_pct = sizer(trade, port)
            if size_pct <= 0:
                port.skipped += 1
                continue
            invested = port.equity() * size_pct
            invested = min(invested, port.cash * 0.99)  # leave dust
            if invested < port.capital * 0.05:  # min 5% NAV — too small ignore
                port.skipped += 1
                port.skipped_no_cash += 1
                continue
            port.cash -= invested
            port.open.append(OpenPosition(
                symbol=trade["symbol"],
                entry_date=trade["entry_date"],
                exit_date=trade["exit_date"],
                entry_value=invested,
                pnl_pct=float(trade["pnl_pct"]),
                setup_type=str(trade["setup_type"]),
            ))
            port.record(date)
    return port


# ── Metrics ───────────────────────────────────────────────────────────────

def metrics(port: Portfolio) -> dict:
    if not port.closed:
        return {"return_pct": 0.0, "sharpe": 0.0, "max_dd_pct": 0.0,
                "win_rate_pct": 0.0, "n_trades": 0, "n_skipped": port.skipped}

    eq = pd.DataFrame(port.equity_history, columns=["date", "equity"])
    if eq.empty:
        return {}
    eq = eq.sort_values("date").drop_duplicates("date", keep="last")
    eq["equity"] = eq["equity"].astype(float)
    final = eq["equity"].iloc[-1]
    ret = (final - port.capital) / port.capital * 100.0

    # Daily resample for sharpe
    eq_idx = eq.set_index("date")["equity"].asfreq("B").ffill()
    rets = eq_idx.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    peak = eq_idx.cummax()
    dd = (eq_idx - peak) / peak * 100.0
    max_dd = float(dd.min())

    closed_df = pd.DataFrame(port.closed)
    wr = (closed_df["pnl"] > 0).mean() * 100.0
    winners = closed_df.loc[closed_df["pnl"] > 0, "pnl"].sum()
    losers = closed_df.loc[closed_df["pnl"] < 0, "pnl"].abs().sum()
    pf = float(winners / losers) if losers > 0 else None

    return {
        "n_trades": int(len(closed_df)),
        "n_skipped": int(port.skipped),
        "skipped_no_slot": int(port.skipped_no_slot),
        "skipped_no_cash": int(port.skipped_no_cash),
        "return_pct": round(ret, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd, 2),
        "win_rate_pct": round(wr, 2),
        "profit_factor": round(pf, 2) if pf else None,
        "final_equity": round(final, 0),
    }


# ── Setup breakdown ──────────────────────────────────────────────────────

def setup_breakdown(port: Portfolio) -> pd.DataFrame:
    if not port.closed:
        return pd.DataFrame()
    df = pd.DataFrame(port.closed)
    grouped = df.groupby("setup_type").agg(
        trades=("symbol", "count"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
        total_pnl=("pnl", "sum"),
        avg_pnl_pct=("pnl_pct", "mean"),
    ).reset_index()
    return grouped


# ── Main ───────────────────────────────────────────────────────────────────

POLICIES: dict[str, SizerFn] = {
    "cash_split": sizer_cash_split(),
    "cap_50": sizer_cap(0.50),
    "cap_33": sizer_cap(0.33),
    "fixed_20": sizer_fixed(0.20),
    "fixed_25": sizer_fixed(0.25),
    "fixed_30": sizer_fixed(0.30),
    "fixed_33": sizer_fixed(0.33),
    "fixed_35": sizer_fixed(0.35),
    "fixed_40": sizer_fixed(0.40),
    "fixed_45": sizer_fixed(0.45),
    "fixed_50": sizer_fixed(0.50),
    "tiered_40_25_15": sizer_tiered(0.40, 0.25, 0.15),
    "tiered_50_33_20": sizer_tiered(0.50, 0.33, 0.20),
    "boost_20_40": sizer_score_boost(0.20, 0.40),
    "boost_25_50": sizer_score_boost(0.25, 0.50),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True)
    parser.add_argument("--capital", type=float, default=1e8)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="sizing_compare")
    args = parser.parse_args()

    trades_df = pd.read_csv(args.trades)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for name, sizer in POLICIES.items():
        port = simulate(trades_df, sizer, capital=args.capital,
                        max_positions=args.max_positions)
        m = metrics(port)
        m["policy"] = name
        summary_rows.append(m)
        # Save per-policy artifacts
        eq_df = pd.DataFrame(port.equity_history, columns=["date", "equity"])
        eq_df.to_csv(out_dir / f"equity_{name}.csv", index=False)
        bd = setup_breakdown(port)
        if not bd.empty:
            bd.to_csv(out_dir / f"setup_breakdown_{name}.csv", index=False)

    summary = pd.DataFrame(summary_rows).set_index("policy")
    summary = summary[[
        "n_trades", "n_skipped", "return_pct", "sharpe", "max_dd_pct",
        "win_rate_pct", "profit_factor",
    ]].sort_values("return_pct", ascending=False)

    summary_path = out_dir / "sizing_summary.csv"
    summary.to_csv(summary_path)

    print(f"\n=== {args.label} | {len(trades_df)} trades | capital={args.capital:.0e} | max_pos={args.max_positions} ===\n")
    print(summary.to_string())
    print(f"\nOutput: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
