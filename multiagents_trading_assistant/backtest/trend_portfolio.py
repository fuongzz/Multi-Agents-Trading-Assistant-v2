"""Trend-following paper portfolio backtest for VN30.

This is intentionally separate from lifecycle single-symbol backtests. It uses
a portfolio event loop: rank VN30 stocks by causal uptrend features, hold fewer
than 10 names, trade next open, and mark NAV daily.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import inf
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class TrendPortfolioConfig:
    start: str = "2025-01-01"
    end: str = "2025-12-31"
    history_start: str = "2024-01-01"
    max_holdings: int = 8
    target_gross: float = 0.95
    rebalance_days: int = 20
    min_hold_days: int = 10
    slippage: float = 0.001
    stop_from_entry: float = 0.075
    trail_from_peak: float = 0.18
    long_trend_hold: bool = True
    winner_lock_pct: float = 0.25
    long_trend_ma: str = "ma120"
    market_regime_gate: bool = True
    uptrend_gross: float = 0.95
    narrow_leadership_gross: float = 0.70
    sideway_gross: float = 0.35
    downtrend_gross: float = 0.0
    narrow_leadership_max_holdings: int = 3
    sideway_max_holdings: int = 4
    sideway_take_profit: float = 0.08
    sideway_stop: float = 0.05
    sideway_max_hold_days: int = 25
    use_regime_strategies: bool = True


@dataclass
class _Position:
    symbol: str
    qty: float
    entry: float
    entry_date: pd.Timestamp
    peak: float
    strategy: str = "trend"


def run_vn30_trend_portfolio(
    *,
    get_symbols: Callable[[], list[str]],
    get_history: Callable[..., pd.DataFrame],
    cfg: TrendPortfolioConfig | None = None,
) -> dict:
    cfg = cfg or TrendPortfolioConfig()
    symbols = [s.upper() for s in get_symbols()]
    frames = _load_frames(symbols, get_history, cfg)
    for sym, df in list(frames.items()):
        frames[sym] = _add_features(df).set_index("date")
    market = _build_market_regime(frames)

    all_dates = sorted(set().union(*[set(df.index) for df in frames.values()]))
    dates = [d for d in all_dates if pd.Timestamp(cfg.start) <= d <= pd.Timestamp(cfg.end)]
    if len(dates) < 2:
        return {"error": "not_enough_dates"}

    cash = 100.0
    positions: dict[str, _Position] = {}
    trades: list[dict] = []
    equity_curve: list[dict] = []
    candidate_log: list[tuple] = []
    last_rebalance_idx = -10**9

    def price(symbol: str, date: pd.Timestamp, field: str) -> float | None:
        df = frames[symbol]
        if date in df.index:
            return float(df.loc[date, field])
        prev = df[df.index <= date]
        if prev.empty:
            return None
        return float(prev.iloc[-1][field])

    def equity(date: pd.Timestamp) -> float:
        value = cash
        for pos in positions.values():
            px = price(pos.symbol, date, "close")
            if px and px > 0:
                value += pos.qty * px
        return value

    def trend_candidates(date: pd.Timestamp) -> list[dict]:
        rows = []
        for sym, df in frames.items():
            hist = df[df.index <= date]
            if hist.empty:
                continue
            row = hist.iloc[-1]
            if not bool(row.get("uptrend", False)):
                continue
            score = float(row.get("trend_score", np.nan))
            if not np.isfinite(score):
                continue
            rows.append(
                {
                    "symbol": sym,
                    "score": score,
                    "ret20": float(row["ret20"]),
                    "ret60": float(row["ret60"]),
                    "ret120": float(row["ret120"]),
                    "strategy": "trend",
                }
            )
        return sorted(rows, key=lambda x: x["score"], reverse=True)

    def range_candidates(date: pd.Timestamp) -> list[dict]:
        rows = []
        for sym, df in frames.items():
            hist = df[df.index <= date]
            if hist.empty:
                continue
            row = hist.iloc[-1]
            close = float(row["close"])
            ma50 = float(row.get("ma50", 0.0) or 0.0)
            ma120 = float(row.get("ma120", 0.0) or 0.0)
            rsi = float(row.get("rsi14", np.nan))
            dd20 = float(row.get("dd20", np.nan))
            ret60 = float(row.get("ret60", np.nan))
            if ma50 <= 0 or ma120 <= 0 or not np.isfinite(rsi) or not np.isfinite(dd20):
                continue
            # Sideway playbook: buy pullback near support, avoid broken long-term trends.
            if close < ma120 * 0.97 or close > ma50 * 1.08:
                continue
            if not (32 <= rsi <= 52):
                continue
            if not (-0.18 <= dd20 <= -0.025):
                continue
            if np.isfinite(ret60) and ret60 < -0.18:
                continue
            score = (52 - rsi) * 0.02 + abs(dd20) * 1.5 + max(0.0, ret60) * 0.2
            rows.append(
                {
                    "symbol": sym,
                    "score": score,
                    "ret20": float(row["ret20"]),
                    "ret60": float(row["ret60"]),
                    "ret120": float(row["ret120"]),
                    "strategy": "range",
                }
            )
        return sorted(rows, key=lambda x: x["score"], reverse=True)

    def candidates(date: pd.Timestamp, regime: str | None = None) -> list[dict]:
        if not cfg.use_regime_strategies:
            return trend_candidates(date)
        if regime == "SIDEWAY":
            return range_candidates(date)
        if regime == "DOWNTREND":
            return []
        return trend_candidates(date)

    def market_regime(date: pd.Timestamp) -> dict:
        if not cfg.market_regime_gate or market.empty:
            return {"regime": "UPTREND", "target_gross": cfg.target_gross}
        hist = market[market.index <= date]
        if hist.empty:
            return {"regime": "UNKNOWN", "target_gross": cfg.sideway_gross}
        row = hist.iloc[-1]
        regime = str(row["regime"])
        gross = (
            cfg.uptrend_gross
            if regime == "UPTREND"
            else cfg.narrow_leadership_gross
            if regime == "NARROW_LEADERSHIP"
            else cfg.sideway_gross
            if regime == "SIDEWAY"
            else cfg.downtrend_gross
        )
        return {"regime": regime, "target_gross": gross, "breadth20": float(row["breadth20"])}

    for idx, date in enumerate(dates[:-1]):
        next_day = dates[idx + 1]
        mkt = market_regime(date)
        exits: list[tuple[str, str]] = []
        for sym, pos in list(positions.items()):
            row = frames[sym][frames[sym].index <= date].iloc[-1]
            close = float(row["close"])
            pos.peak = max(pos.peak, close)
            held_days = (date - pos.entry_date).days
            if pos.strategy == "range":
                range_tp = close >= pos.entry * (1 + cfg.sideway_take_profit)
                range_sl = close <= pos.entry * (1 - cfg.sideway_stop)
                range_timeout = held_days >= cfg.sideway_max_hold_days
                range_break = close < float(row.get("ma120", 0.0) or 0.0) * 0.95
                market_exit = cfg.market_regime_gate and mkt["regime"] == "DOWNTREND"
                if market_exit or range_tp or range_sl or range_timeout or range_break:
                    exits.append((sym, "market_downtrend" if market_exit else "range_tp" if range_tp else "range_sl" if range_sl else "range_timeout" if range_timeout else "range_break"))
            else:
                locked = _is_locked_winner(pos, row, cfg)
                trend_break = (
                    bool(close < float(row["ma50"]) or float(row["ma20"]) < float(row["ma50"]))
                    if not locked
                    else _long_trend_break(row, cfg)
                )
                hard_stop = close <= pos.entry * (1 - cfg.stop_from_entry)
                trail_stop = held_days >= cfg.min_hold_days and close <= pos.peak * (1 - cfg.trail_from_peak)
                market_exit = cfg.market_regime_gate and mkt["regime"] == "DOWNTREND"
                if market_exit or hard_stop or trend_break or trail_stop:
                    exits.append((sym, "market_downtrend" if market_exit else "hard_stop" if hard_stop else "trend_break" if trend_break else "trail_stop"))

        for sym, reason in exits:
            if sym not in positions:
                continue
            pos = positions.pop(sym)
            op = price(sym, next_day, "open")
            if not op:
                continue
            fill = op * (1 - cfg.slippage)
            proceeds = pos.qty * fill
            cash += proceeds
            trades.append(_sell_trade(pos, next_day, fill, proceeds - pos.qty * pos.entry, reason))

        do_rebalance = (idx - last_rebalance_idx) >= cfg.rebalance_days or idx == 0
        if do_rebalance:
            ranked = candidates(date, mkt["regime"])
            max_holdings_today = (
                cfg.max_holdings
                if mkt["regime"] == "UPTREND"
                else cfg.narrow_leadership_max_holdings
                if mkt["regime"] == "NARROW_LEADERSHIP"
                else cfg.sideway_max_holdings
                if mkt["regime"] == "SIDEWAY"
                else 0
            )
            top = [] if mkt["target_gross"] <= 0 else ranked[:max_holdings_today]
            candidate_log.append(
                (
                    date.date(),
                    [(x["symbol"], x.get("strategy", "trend"), round(x["score"], 3), round(x["ret60"] * 100, 1)) for x in top],
                )
            )
            target_symbols = [x["symbol"] for x in top]
            target_strategy = {x["symbol"]: x.get("strategy", "trend") for x in top}
            for sym in list(positions.keys()):
                if sym in target_symbols:
                    continue
                row = frames[sym][frames[sym].index <= date].iloc[-1]
                if positions[sym].strategy != "range" and _is_locked_winner(positions[sym], row, cfg):
                    continue
                pos = positions.pop(sym)
                op = price(sym, next_day, "open")
                if not op:
                    continue
                fill = op * (1 - cfg.slippage)
                proceeds = pos.qty * fill
                cash += proceeds
                trades.append(_sell_trade(pos, next_day, fill, proceeds - pos.qty * pos.entry, "rebalance_out"))

            nav = equity(date)
            target_value = nav * mkt["target_gross"] / max(1, len(target_symbols)) if target_symbols else 0.0
            for sym in target_symbols:
                if sym in positions:
                    continue
                op = price(sym, next_day, "open")
                if not op or target_value <= 0:
                    continue
                spend = min(cash, target_value)
                if spend < nav * 0.03:
                    continue
                fill = op * (1 + cfg.slippage)
                cash -= spend
                positions[sym] = _Position(sym, spend / fill, fill, next_day, fill, target_strategy.get(sym, "trend"))
                trades.append(
                    {
                        "symbol": sym,
                        "entry_date": next_day.date(),
                        "exit_date": None,
                        "entry": fill,
                        "exit": None,
                        "pnl": None,
                        "pnl_pct": None,
                        "reason": "buy",
                    }
                )
            last_rebalance_idx = idx

        equity_curve.append(
            {
                "date": date.date(),
                "nav": round(equity(date), 4),
                "cash": round(cash, 4),
                "holdings": sorted(positions.keys()),
                "market_regime": mkt["regime"],
            }
        )

    final_day = dates[-1]
    for sym, pos in list(positions.items()):
        px = price(sym, final_day, "close")
        if not px:
            continue
        fill = px * (1 - cfg.slippage)
        proceeds = pos.qty * fill
        cash += proceeds
        trades.append(_sell_trade(pos, final_day, fill, proceeds - pos.qty * pos.entry, "end"))
    positions.clear()

    final_nav = cash
    realized = [t for t in trades if t.get("exit_date") is not None and t.get("pnl") is not None]
    wins = [t for t in realized if t["pnl"] > 0]
    losses = [t for t in realized if t["pnl"] < 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    first_date, first_top = _first_full_candidate_date(
        lambda d: trend_candidates(d),
        dates,
        cfg.max_holdings,
    )

    return {
        "config": cfg,
        "symbols": symbols,
        "final_nav": round(final_nav, 4),
        "return_pct": round(final_nav - 100.0, 4),
        "vn30_equal_weight_buy_hold_pct": round(_equal_weight_buy_hold(frames, pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)), 4),
        "initial_top_buy_hold_pct": round(_selected_buy_hold(frames, first_top, first_date, pd.Timestamp(cfg.end)), 4),
        "initial_top_date": first_date.date() if first_date is not None else None,
        "initial_top_symbols": first_top,
        "closed_trades": len(realized),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / max(1, len(realized)) * 100, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else inf,
        "candidate_log": candidate_log,
        "market_regime_counts": market.loc[(market.index >= pd.Timestamp(cfg.start)) & (market.index <= pd.Timestamp(cfg.end)), "regime"].value_counts().to_dict() if not market.empty else {},
        "equity_curve": equity_curve,
        "trades": trades,
    }


def _load_frames(symbols: list[str], get_history, cfg: TrendPortfolioConfig) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(get_history, sym, start=cfg.history_start, end=cfg.end): sym for sym in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                df = fut.result()
            except Exception:
                continue
            if df is None or df.empty or len(df) <= 180:
                continue
            data = df.copy()
            data["date"] = pd.to_datetime(data["date"]).dt.normalize()
            data = data.sort_values("date").drop_duplicates("date")
            for col in ("open", "high", "low", "close", "volume"):
                data[col] = pd.to_numeric(data[col], errors="coerce")
            frames[sym] = data.dropna(subset=["open", "close"])
    return frames


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = data["close"].astype(float)
    volume = data["volume"].astype(float)
    data["ma20"] = close.rolling(20).mean()
    data["ma50"] = close.rolling(50).mean()
    data["ma120"] = close.rolling(120).mean()
    data["ma20_prev"] = data["ma20"].shift(10)
    data["ret20"] = close / close.shift(20) - 1
    data["ret60"] = close / close.shift(60) - 1
    data["ret120"] = close / close.shift(120) - 1
    data["vol_ratio"] = volume / volume.rolling(20).mean()
    data["rsi14"] = _rsi(close, 14)
    data["dd20"] = close / close.rolling(20).max() - 1
    data["dd60"] = close / close.rolling(60).max() - 1
    data["uptrend"] = (
        (close > data["ma20"])
        & (data["ma20"] > data["ma50"])
        & (data["ma50"] > data["ma120"])
        & (data["ma20"] > data["ma20_prev"])
    )
    data["trend_score"] = (
        data["ret60"] * 1.2
        + data["ret20"] * 0.7
        + data["ret120"] * 0.35
        + (data["vol_ratio"].clip(0, 3) - 1) * 0.03
        - data["dd60"].abs() * 0.15
    )
    return data


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _build_market_regime(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a causal VN30 equal-weight market regime proxy.

    VNINDEX absolute history is incomplete in some local data sources for early
    2020, so the portfolio uses its own tradable universe as the market proxy.
    """
    series = []
    breadth = []
    for sym, df in frames.items():
        data = df.copy()
        if data.empty:
            continue
        close = data["close"].astype(float)
        norm = close / close.dropna().iloc[0]
        series.append(norm.rename(sym))
        breadth.append((close > data["ma20"]).astype(float).rename(sym))
    if not series:
        return pd.DataFrame()
    idx = pd.concat(series, axis=1).sort_index().ffill().mean(axis=1)
    br20 = pd.concat(breadth, axis=1).sort_index().ffill().mean(axis=1)
    out = pd.DataFrame({"close": idx, "breadth20": br20})
    out["ma20"] = out["close"].rolling(20).mean()
    out["ma50"] = out["close"].rolling(50).mean()
    out["ma120"] = out["close"].rolling(120).mean()
    out["ma20_prev"] = out["ma20"].shift(10)
    up = (
        (out["close"] > out["ma20"])
        & (out["ma20"] > out["ma50"])
        & (out["ma50"] > out["ma120"])
        & (out["ma20"] > out["ma20_prev"])
        & (out["breadth20"] >= 0.50)
    )
    down = (
        ((out["close"] < out["ma50"]) & (out["ma20"] < out["ma50"]))
        | ((out["close"] < out["ma120"]) & (out["breadth20"] < 0.40))
    )
    leader_strength = _leader_strength(frames, out.index)
    out["leader_count"] = leader_strength["leader_count"]
    out["leader_score"] = leader_strength["leader_score"]
    narrow = (
        ~up
        & ~down
        & (out["leader_count"] >= 2)
        & (out["leader_score"] >= 0.65)
        & (out["close"] > out["ma50"])
    )
    out["regime"] = np.where(
        up,
        "UPTREND",
        np.where(down, "DOWNTREND", np.where(narrow, "NARROW_LEADERSHIP", "SIDEWAY")),
    )
    return out


def _leader_strength(frames: dict[str, pd.DataFrame], dates: pd.Index) -> pd.DataFrame:
    rows = []
    for date in dates:
        scores = []
        for df in frames.values():
            hist = df[df.index <= date]
            if hist.empty:
                continue
            row = hist.iloc[-1]
            if not bool(row.get("uptrend", False)):
                continue
            score = float(row.get("trend_score", np.nan))
            ret60 = float(row.get("ret60", np.nan))
            if np.isfinite(score) and np.isfinite(ret60) and ret60 > 0.25:
                scores.append(score)
        scores = sorted(scores, reverse=True)
        rows.append(
            {
                "leader_count": len(scores),
                "leader_score": float(np.mean(scores[:3])) if scores else 0.0,
            }
        )
    return pd.DataFrame(rows, index=dates)


def _sell_trade(pos: _Position, exit_date: pd.Timestamp, fill: float, pnl: float, reason: str) -> dict:
    return {
        "symbol": pos.symbol,
        "strategy": pos.strategy,
        "entry_date": pos.entry_date.date(),
        "exit_date": exit_date.date(),
        "entry": pos.entry,
        "exit": fill,
        "pnl": pnl,
        "pnl_pct": fill / pos.entry - 1,
        "reason": reason,
    }


def _is_locked_winner(pos: _Position, row, cfg: TrendPortfolioConfig) -> bool:
    if not cfg.long_trend_hold:
        return False
    close = float(row["close"])
    ma_long = float(row.get(cfg.long_trend_ma, 0.0) or 0.0)
    if ma_long <= 0:
        return False
    return close >= pos.entry * (1.0 + cfg.winner_lock_pct) and close > ma_long


def _long_trend_break(row, cfg: TrendPortfolioConfig) -> bool:
    close = float(row["close"])
    ma_long = float(row.get(cfg.long_trend_ma, 0.0) or 0.0)
    ma50 = float(row.get("ma50", 0.0) or 0.0)
    if ma_long <= 0:
        return True
    return close < ma_long or (ma50 > 0 and ma50 < ma_long)


def _equal_weight_buy_hold(frames: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> float:
    returns = []
    for df in frames.values():
        period = df[(df.index >= start) & (df.index <= end)]
        if not period.empty:
            returns.append(float(period.iloc[-1]["close"] / period.iloc[0]["open"] - 1))
    return sum(returns) / max(1, len(returns)) * 100


def _selected_buy_hold(frames: dict[str, pd.DataFrame], symbols: list[str], start: pd.Timestamp | None, end: pd.Timestamp) -> float:
    if start is None or not symbols:
        return 0.0
    returns = []
    for sym in symbols:
        df = frames[sym]
        period = df[(df.index >= start) & (df.index <= end)]
        if not period.empty:
            returns.append(float(period.iloc[-1]["close"] / period.iloc[0]["open"] - 1))
    return sum(returns) / max(1, len(returns)) * 100


def _first_full_candidate_date(candidates_fn, dates: list[pd.Timestamp], max_holdings: int) -> tuple[pd.Timestamp | None, list[str]]:
    for date in dates:
        rows = candidates_fn(date)
        if len(rows) >= max_holdings:
            return date, [r["symbol"] for r in rows[:max_holdings]]
    return None, []


def print_summary(result: dict) -> None:
    print("\n=== TREND FOLLOWING PAPER PORTFOLIO VN30 ===")
    cfg = result["config"]
    print(
        f"period={cfg.start}->{cfg.end} max_holdings={cfg.max_holdings} "
        f"target_gross={cfg.target_gross:.0%} rebalance_days={cfg.rebalance_days}"
    )
    print(f"final_nav={result['final_nav']:.2f} return={result['return_pct']:+.2f}%")
    print(f"vn30_equal_weight_buy_hold={result['vn30_equal_weight_buy_hold_pct']:+.2f}%")
    print(
        f"initial_top{cfg.max_holdings}_buy_hold_from_{result['initial_top_date']}="
        f"{result['initial_top_buy_hold_pct']:+.2f}% {result['initial_top_symbols']}"
    )
    print(
        f"closed_trades={result['closed_trades']} wins={result['wins']} losses={result['losses']} "
        f"win_rate={result['win_rate_pct']:.1f}% profit_factor={result['profit_factor']:.2f}"
    )
    print(f"market_regimes={result.get('market_regime_counts', {})}")
    print("\nRebalance snapshots:")
    for date, items in result["candidate_log"][:8]:
        print(date, items)
    print("\nTop trades:")
    closed = [t for t in result["trades"] if t.get("exit_date") is not None and t.get("pnl") is not None]
    for trade in sorted(closed, key=lambda x: x["pnl"], reverse=True)[:12]:
        print(trade)


if __name__ == "__main__":
    from multiagents_trading_assistant.fetcher import get_ohlcv_history, get_vn30_symbols

    print_summary(run_vn30_trend_portfolio(get_symbols=get_vn30_symbols, get_history=get_ohlcv_history))
