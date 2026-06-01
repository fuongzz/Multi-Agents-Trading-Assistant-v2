"""Backtest hostile-market combo-long leadership strategy.

This script formalizes the best recent hostile-market research candidate:
top-1 leadership rotation, 3-session rebalance, and MA20 exit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiagents_trading_assistant.fetcher import get_vn100_symbols

MASTER = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
DEFAULT_OUT_DIR = ROOT / "reports" / "hostile_combo_long_vn100_long"


def _load_prices(start: str, end: str) -> pd.DataFrame:
    symbols = set(get_vn100_symbols())
    warmup = pd.Timestamp(start) - pd.Timedelta(days=260)
    df = pd.read_parquet(MASTER)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df[df["symbol"].isin(symbols) & (df["date"] >= warmup) & (df["date"] <= pd.Timestamp(end))].copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("symbol", sort=False)
    out["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    out["ret20"] = g["close"].pct_change(20)
    out["ret60"] = g["close"].pct_change(60)
    out["ret120"] = g["close"].pct_change(120)
    out["value20"] = g.apply(
        lambda x: (x["close"] * 1000.0 * x["volume"]).rolling(20).mean(),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    out["rs20"] = out.groupby("date")["ret20"].rank(pct=True)
    out["rs60"] = out.groupby("date")["ret60"].rank(pct=True)
    out["rs120"] = out.groupby("date")["ret120"].rank(pct=True)
    out["value_rank"] = out.groupby("date")["value20"].rank(pct=True)
    out["score"] = (
        0.35 * out["rs60"].fillna(0.0)
        + 0.35 * out["rs120"].fillna(0.0)
        + 0.20 * out["rs20"].fillna(0.0)
        + 0.10 * out["value_rank"].fillna(0.0)
    )
    out["above_ma20"] = out["close"] > out["ma20"]
    return out


def _lot_shares(value: float, price: float, lot_size: int) -> int:
    if value <= 0 or price <= 0:
        return 0
    return int(value // price // lot_size) * lot_size


def _max_drawdown(values: pd.Series) -> float:
    return float((values / values.cummax() - 1.0).min()) if not values.empty else 0.0


def _sharpe(values: pd.Series) -> float:
    returns = values.pct_change().fillna(0.0)
    std = returns.std(ddof=0)
    return 0.0 if std == 0 or np.isnan(std) else float(returns.mean() / std * np.sqrt(252))


def _pick_candidate(day: pd.DataFrame, min_value_vnd: float) -> str | None:
    pool = day[day["value20"].ge(min_value_vnd)].sort_values("score", ascending=False)
    if pool.empty:
        return None
    return str(pool.iloc[0]["symbol"])


def run_backtest(
    features: pd.DataFrame,
    start: str,
    end: str,
    *,
    initial_capital: float = 100_000_000.0,
    min_value_vnd: float = 40_000_000_000.0,
    rebalance_days: int = 3,
    commission_rate: float = 0.001,
    sell_tax_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    lot_size: int = 100,
    target_exposure: float = 1.0,
) -> dict[str, Any]:
    by_date = {date: day.copy() for date, day in features.groupby("date", sort=True)}
    dates = [d for d in sorted(by_date) if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    cash = initial_capital
    position: dict[str, Any] | None = None
    pending_target: str | None = None
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    last_rebalance_idx = -10000

    for idx, date in enumerate(dates):
        day = by_date[date]
        open_px = day.set_index("symbol")["open"] * 1000.0
        close_px = day.set_index("symbol")["close"] * 1000.0

        if pending_target is not None:
            target = pending_target
            pending_target = None
            current_symbol = position["symbol"] if position else None
            if position and current_symbol != target:
                price = float(open_px.get(current_symbol, close_px.get(current_symbol, np.nan))) * (1.0 - slippage_rate)
                if not np.isnan(price):
                    gross = position["shares"] * price
                    fees = gross * (commission_rate + sell_tax_rate)
                    cash += gross - fees
                    pnl_pct = (price / position["entry_price"] - 1.0) * 100.0
                    trades.append({
                        "date": date.date().isoformat(),
                        "symbol": current_symbol,
                        "side": "SELL",
                        "price": round(price, 2),
                        "shares": position["shares"],
                        "reason": "REBALANCE_OR_STOP",
                        "pnl_pct": round(pnl_pct, 2),
                    })
                    position = None
            if target and position is None and target in open_px.index:
                price = float(open_px[target]) * (1.0 + slippage_rate)
                shares = _lot_shares(cash * target_exposure, price, lot_size)
                cost = shares * price * (1.0 + commission_rate)
                if shares > 0 and cost <= cash:
                    cash -= cost
                    position = {"symbol": target, "shares": shares, "entry_price": price, "entry_date": date}
                    trades.append({
                        "date": date.date().isoformat(),
                        "symbol": target,
                        "side": "BUY",
                        "price": round(price, 2),
                        "shares": shares,
                        "reason": "TARGET_BUY",
                        "pnl_pct": 0.0,
                    })

        if position:
            row = day[day["symbol"].eq(position["symbol"])]
            if not row.empty and bool(row.iloc[0]["above_ma20"]) is False:
                if idx + 1 < len(dates):
                    pending_target = ""

        if idx - last_rebalance_idx >= rebalance_days:
            target = _pick_candidate(day, min_value_vnd)
            if target and idx + 1 < len(dates):
                pending_target = target
                last_rebalance_idx = idx

        symbols = position["symbol"] if position else ""
        market_value = 0.0
        if position:
            market_value = position["shares"] * float(close_px.get(position["symbol"], position["entry_price"]))
        equity_rows.append({
            "date": date,
            "equity": cash + market_value,
            "cash": cash,
            "symbols": symbols,
        })

    equity = pd.DataFrame(equity_rows)
    values = equity["equity"].astype(float)
    sell_trades = [t for t in trades if t["side"] == "SELL"]
    return {
        "summary": {
            "start": start,
            "end": end,
            "total_return_pct": round((values.iloc[-1] / values.iloc[0] - 1.0) * 100.0, 2),
            "sharpe_ratio": round(_sharpe(values), 3),
            "max_drawdown_pct": round(_max_drawdown(values) * 100.0, 2),
            "ending_equity": round(float(values.iloc[-1]), 2),
            "orders": len(trades),
            "sell_orders": len(sell_trades),
            "win_rate_pct": round(sum(1 for t in sell_trades if t["pnl_pct"] > 0) / len(sell_trades) * 100.0, 2) if sell_trades else 0.0,
            "current_symbol": str(equity.iloc[-1]["symbols"]),
            "target_exposure": round(target_exposure, 4),
        },
        "equity": equity,
        "trades": pd.DataFrame(trades),
    }


def _period_metrics(equity: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    g = equity[(equity["date"] >= pd.Timestamp(start)) & (equity["date"] <= pd.Timestamp(end))]
    if g.empty:
        return {"return_pct": 0.0, "max_drawdown_pct": 0.0}
    values = g["equity"].astype(float)
    return {
        "return_pct": round((values.iloc[-1] / values.iloc[0] - 1.0) * 100.0, 2),
        "max_drawdown_pct": round(_max_drawdown(values) * 100.0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", default="2026-05-29")
    parser.add_argument("--starts", nargs="+", default=["2016-01-01", "2018-01-01", "2020-01-01", "2025-12-01"])
    parser.add_argument("--target-exposure", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    first_start = min(pd.Timestamp(s) for s in args.starts).date().isoformat()
    prices = _load_prices(first_start, args.end)
    features = _add_features(prices)

    rows: list[dict[str, Any]] = []
    for start in args.starts:
        result = run_backtest(features, start, args.end, target_exposure=args.target_exposure)
        summary = result["summary"]
        last_1m = _period_metrics(result["equity"], "2026-05-01", args.end)
        last_6m = _period_metrics(result["equity"], "2025-12-01", args.end)
        summary.update({
            "return_1m_pct": last_1m["return_pct"],
            "drawdown_1m_pct": last_1m["max_drawdown_pct"],
            "return_6m_pct": last_6m["return_pct"],
            "drawdown_6m_pct": last_6m["max_drawdown_pct"],
        })
        rows.append(summary)
        tag = start.replace("-", "")
        result["equity"].to_csv(args.out_dir / f"combo_long_{tag}_equity.csv", index=False, encoding="utf-8-sig")
        result["trades"].to_csv(args.out_dir / f"combo_long_{tag}_trades.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.out_dir / "summary.csv", index=False, encoding="utf-8-sig")

    last_day = features[features["date"].eq(pd.Timestamp(args.end))].copy()
    candidates = last_day[last_day["value20"].ge(40_000_000_000.0)].sort_values("score", ascending=False)
    cols = ["date", "symbol", "score", "ret20", "ret60", "ret120", "rs20", "rs60", "rs120", "value20", "above_ma20"]
    candidates[cols].head(20).to_csv(args.out_dir / f"latest_candidates_{args.end}.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))
    print(args.out_dir / "summary.csv")


if __name__ == "__main__":
    main()
