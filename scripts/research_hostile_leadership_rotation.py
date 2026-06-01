"""Research hostile-market leadership rotation strategies.

This is intentionally separate from Core MVP and Flow V2. It searches for the
few stocks that can still lead when broad market breadth is weak.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multiagents_trading_assistant.fetcher import get_vn100_symbols


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "reports" / "hostile_leadership_rotation_vn100_2020_to_2026-05-29"
MASTER = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"


@dataclass(frozen=True)
class Config:
    name: str
    positions: int = 2
    rebalance_days: int = 5
    min_value_vnd: float = 40_000_000_000.0
    min_rs60_pct: float = 0.70
    min_ret20: float | None = None
    min_ret60: float = 0.0
    require_high60_reclaim: bool = False
    require_above_ma20: bool = True
    require_above_ma50: bool = True
    max_distance_ma20: float | None = 0.22
    stop_ma20: bool = True
    stop_ret20: float | None = -0.10
    score_mode: str = "leader"
    compound: bool = True
    initial_capital: float = 100_000_000.0
    price_unit_multiplier: float = 1000.0
    commission_rate: float = 0.001
    sell_tax_rate: float = 0.001
    slippage_rate: float = 0.0005
    lot_size: int = 100


CONFIGS = [
    Config("leader_rs60_p2_r5", positions=2, rebalance_days=5, min_rs60_pct=0.70, min_ret20=None),
    Config("leader_rs60_p2_r10", positions=2, rebalance_days=10, min_rs60_pct=0.70, min_ret20=None),
    Config("leader_breakout_p2_r5", positions=2, rebalance_days=5, min_rs60_pct=0.75, min_ret20=0.0, require_high60_reclaim=True),
    Config("leader_pullback_p2_r5", positions=2, rebalance_days=5, min_rs60_pct=0.70, min_ret20=-0.08, require_high60_reclaim=False, max_distance_ma20=0.12),
    Config("leader_rs60_p1_r5", positions=1, rebalance_days=5, min_rs60_pct=0.75, min_ret20=None),
    Config("leader_rs60_p3_r5", positions=3, rebalance_days=5, min_rs60_pct=0.65, min_ret20=None),
    Config("leader_no_ma20_stop_p2", positions=2, rebalance_days=5, min_rs60_pct=0.70, min_ret20=None, stop_ma20=False),
    Config("leader_strict_liquidity_p2", positions=2, rebalance_days=5, min_rs60_pct=0.75, min_value_vnd=100_000_000_000.0, min_ret20=0.0),
]


def _load_prices(start: str, end: str) -> pd.DataFrame:
    symbols = set(get_vn100_symbols())
    df = pd.read_parquet(MASTER)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    warmup = pd.Timestamp(start) - pd.Timedelta(days=300)
    df = df[df["symbol"].isin(symbols) & (df["date"] >= warmup) & (df["date"] <= pd.Timestamp(end))].copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def _add_features(df: pd.DataFrame, price_unit_multiplier: float) -> pd.DataFrame:
    g = df.groupby("symbol", sort=False)
    out = df.copy()
    out["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    out["ma50"] = g["close"].transform(lambda s: s.rolling(50).mean())
    out["ma100"] = g["close"].transform(lambda s: s.rolling(100).mean())
    out["ret5"] = g["close"].pct_change(5)
    out["ret20"] = g["close"].pct_change(20)
    out["ret60"] = g["close"].pct_change(60)
    out["ret120"] = g["close"].pct_change(120)
    out["high60_prev"] = g["high"].transform(lambda s: s.shift(1).rolling(60).max())
    out["high120_prev"] = g["high"].transform(lambda s: s.shift(1).rolling(120).max())
    out["value_vnd"] = out["close"] * price_unit_multiplier * out["volume"]
    g = out.groupby("symbol", sort=False)
    out["value20_vnd"] = g["value_vnd"].transform(lambda s: s.rolling(20).mean())
    out["distance_ma20"] = out["close"] / out["ma20"] - 1.0
    out["distance_ma50"] = out["close"] / out["ma50"] - 1.0
    out["above_ma20"] = out["close"] > out["ma20"]
    out["above_ma50"] = out["close"] > out["ma50"]
    out["rs60_pct"] = out.groupby("date")["ret60"].rank(pct=True)
    out["rs20_pct"] = out.groupby("date")["ret20"].rank(pct=True)
    out["rs120_pct"] = out.groupby("date")["ret120"].rank(pct=True)
    out["value_pct"] = out.groupby("date")["value20_vnd"].rank(pct=True)
    out["leader_score"] = (
        0.38 * out["rs60_pct"].fillna(0.0)
        + 0.24 * out["rs120_pct"].fillna(0.0)
        + 0.22 * out["rs20_pct"].fillna(0.0)
        + 0.16 * out["value_pct"].fillna(0.0)
    )
    return out


def _eligible(day: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    mask = (
        (day["value20_vnd"] >= cfg.min_value_vnd)
        & (day["rs60_pct"] >= cfg.min_rs60_pct)
        & (day["ret60"] >= cfg.min_ret60)
    )
    if cfg.min_ret20 is not None:
        mask &= day["ret20"] >= cfg.min_ret20
    if cfg.require_above_ma20:
        mask &= day["above_ma20"]
    if cfg.require_above_ma50:
        mask &= day["above_ma50"]
    if cfg.max_distance_ma20 is not None:
        mask &= day["distance_ma20"] <= cfg.max_distance_ma20
    if cfg.require_high60_reclaim:
        mask &= day["close"] >= day["high60_prev"] * 0.98
    return day[mask].sort_values("leader_score", ascending=False)


def _lot_shares(value: float, price: float, lot_size: int) -> int:
    if value <= 0 or price <= 0:
        return 0
    raw = int(value // price)
    return (raw // lot_size) * lot_size


def _max_drawdown(values: pd.Series) -> float:
    return float((values / values.cummax() - 1.0).min()) if not values.empty else 0.0


def _sharpe(values: pd.Series) -> float:
    returns = values.pct_change().fillna(0.0)
    std = returns.std(ddof=0)
    return 0.0 if std == 0 or np.isnan(std) else float(returns.mean() / std * np.sqrt(252))


def run_backtest(features: pd.DataFrame, cfg: Config, start: str, end: str) -> dict[str, Any]:
    by_date = {date: day.copy() for date, day in features.groupby("date", sort=True)}
    dates = [d for d in sorted(by_date) if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    cash = cfg.initial_capital
    positions: dict[str, dict[str, Any]] = {}
    pending_targets: dict[pd.Timestamp, list[str]] = {}
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    last_rebalance_idx = -10_000

    for idx, date in enumerate(dates):
        day = by_date[date]
        open_prices = day.set_index("symbol")["open"] * cfg.price_unit_multiplier
        close_prices = day.set_index("symbol")["close"] * cfg.price_unit_multiplier

        targets = pending_targets.pop(date, None)
        if targets is not None:
            target_set = set(targets)
            # Sell names no longer targeted at open T+1.
            for symbol in list(positions):
                if symbol in target_set:
                    continue
                price = float(open_prices.get(symbol, close_prices.get(symbol, np.nan))) * (1.0 - cfg.slippage_rate)
                if np.isnan(price):
                    continue
                pos = positions.pop(symbol)
                gross = pos["shares"] * price
                fees = gross * (cfg.commission_rate + cfg.sell_tax_rate)
                cash += gross - fees
                trades.append({
                    "date": date.date().isoformat(), "symbol": symbol, "side": "SELL", "price": round(price, 2),
                    "shares": pos["shares"], "reason": "REBALANCE_OUT", "pnl_pct": round((price / pos["entry_price"] - 1) * 100, 2)
                })
            # Buy/resize targets equally.
            current_value = cash + sum(pos["shares"] * float(close_prices.get(sym, pos["entry_price"])) for sym, pos in positions.items())
            target_value = current_value / max(1, cfg.positions) if cfg.compound else cfg.initial_capital / max(1, cfg.positions)
            for symbol in targets:
                if symbol in positions or symbol not in open_prices:
                    continue
                price = float(open_prices[symbol]) * (1.0 + cfg.slippage_rate)
                value = min(target_value, cash)
                shares = _lot_shares(value, price, cfg.lot_size)
                cost = shares * price * (1.0 + cfg.commission_rate)
                if shares <= 0 or cost > cash:
                    continue
                cash -= cost
                positions[symbol] = {"shares": shares, "entry_price": price, "entry_date": date}
                trades.append({"date": date.date().isoformat(), "symbol": symbol, "side": "BUY", "price": round(price, 2), "shares": shares, "reason": "TARGET_BUY", "pnl_pct": 0.0})

        # Fast exit after close, executed next open via pending target replacement.
        hold_symbols = set(positions)
        if hold_symbols:
            rows = day[day["symbol"].isin(hold_symbols)]
            exits = set()
            for _, row in rows.iterrows():
                symbol = row["symbol"]
                if cfg.stop_ma20 and bool(row.get("above_ma20")) is False:
                    exits.add(symbol)
                if cfg.stop_ret20 is not None and float(row.get("ret20") or 0.0) < cfg.stop_ret20:
                    exits.add(symbol)
            if exits:
                next_idx = idx + 1
                if next_idx < len(dates):
                    pending_targets[dates[next_idx]] = [sym for sym in positions if sym not in exits]

        # Rebalance after close T, execute at next open.
        if idx - last_rebalance_idx >= cfg.rebalance_days:
            pool = _eligible(day, cfg)
            targets = list(pool.head(cfg.positions)["symbol"])
            if targets:
                next_idx = idx + 1
                if next_idx < len(dates):
                    pending_targets[dates[next_idx]] = targets
                    last_rebalance_idx = idx

        equity = cash + sum(pos["shares"] * float(close_prices.get(sym, pos["entry_price"])) for sym, pos in positions.items())
        equity_rows.append({"date": date, "equity": equity, "cash": cash, "positions": len(positions), "symbols": ",".join(sorted(positions))})

    equity = pd.DataFrame(equity_rows)
    values = equity["equity"].astype(float)
    sells = [t for t in trades if t["side"] == "SELL"]
    return {
        "summary": {
            "variant": cfg.name,
            "positions": cfg.positions,
            "rebalance_days": cfg.rebalance_days,
            "total_return_pct": round((values.iloc[-1] / values.iloc[0] - 1.0) * 100.0, 2),
            "sharpe_ratio": round(_sharpe(values), 3),
            "max_drawdown_pct": round(_max_drawdown(values) * 100.0, 2),
            "ending_equity": round(float(values.iloc[-1]), 2),
            "number_of_orders": len(trades),
            "sell_orders": len(sells),
            "win_rate_pct": round(sum(1 for t in sells if t["pnl_pct"] > 0) / len(sells) * 100.0, 2) if sells else 0.0,
        },
        "equity": equity,
        "trades": pd.DataFrame(trades),
    }


def _period_metrics(equity: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    g = equity[(equity["date"] >= pd.Timestamp(start)) & (equity["date"] <= pd.Timestamp(end))]
    if g.empty:
        return {"return_pct": 0.0, "drawdown_pct": 0.0}
    v = g["equity"].astype(float)
    return {"return_pct": round((v.iloc[-1] / v.iloc[0] - 1.0) * 100.0, 2), "drawdown_pct": round(_max_drawdown(v) * 100.0, 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-29")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prices = _load_prices(args.start, args.end)
    features = _add_features(prices, 1000.0)
    rows: list[dict[str, Any]] = []
    for cfg in CONFIGS:
        print(f"[hostile-leader] {cfg.name}")
        result = run_backtest(features, cfg, args.start, args.end)
        summary = result["summary"]
        recent = _period_metrics(result["equity"], "2025-12-01", args.end)
        ytd = _period_metrics(result["equity"], "2025-01-01", args.end)
        summary.update({
            "return_6m_pct": recent["return_pct"],
            "drawdown_6m_pct": recent["drawdown_pct"],
            "return_2025_now_pct": ytd["return_pct"],
            "drawdown_2025_now_pct": ytd["drawdown_pct"],
        })
        rows.append(summary)
        result["equity"].to_csv(args.out_dir / f"{cfg.name}_equity.csv", index=False, encoding="utf-8-sig")
        result["trades"].to_csv(args.out_dir / f"{cfg.name}_trades.csv", index=False, encoding="utf-8-sig")
        print(summary)
    summary_df = pd.DataFrame(rows).sort_values(["return_6m_pct", "sharpe_ratio"], ascending=[False, False])
    summary_df.to_csv(args.out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))
    print(args.out_dir / "summary.csv")


if __name__ == "__main__":
    main()
