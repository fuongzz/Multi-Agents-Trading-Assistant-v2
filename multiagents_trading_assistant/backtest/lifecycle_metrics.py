"""Multi-leg metrics — tính hiệu suất từ Position thay vì Trade.

Khác với metrics.py cũ:
  - Một Position có thể có nhiều Buy/Sell leg → tính realized PnL theo avg cost.
  - Position size là tổng các buy leg (gross), không phải cố định 2/3/5%.
  - Equity curve dựa trên realized_pnl_nav_pct (đã được scale theo NAV).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from multiagents_trading_assistant.backtest.lifecycle import Position, PositionStage


def position_metrics(pos: Position) -> dict:
    """Metrics 1 Position đã đóng."""
    if pos.stage != PositionStage.CLOSED or not pos.legs:
        return {"symbol": pos.symbol, "status": "open"}

    buys = [l for l in pos.legs if l.is_buy]
    sells = [l for l in pos.legs if l.is_sell]
    total_buy_qty = sum(l.qty_pct for l in buys)
    total_sell_qty = sum(-l.qty_pct for l in sells)
    avg_buy = (
        sum(l.price * l.qty_pct for l in buys) / total_buy_qty if total_buy_qty > 0 else 0.0
    )
    avg_sell = (
        sum(l.price * (-l.qty_pct) for l in sells) / total_sell_qty
        if total_sell_qty > 0
        else 0.0
    )

    pnl_pct_on_capital = (
        (avg_sell - avg_buy) / avg_buy * 100 if avg_buy > 0 else 0.0
    )

    return {
        "symbol": pos.symbol,
        "open_date": pos.open_date,
        "close_date": pos.close_date,
        "num_legs": len(pos.legs),
        "num_buys": len(buys),
        "num_sells": len(sells),
        "max_nav_pct": round(max(_running_nav(pos.legs)), 4),
        "avg_buy_price": round(avg_buy, 2),
        "avg_sell_price": round(avg_sell, 2),
        "pnl_pct_on_capital": round(pnl_pct_on_capital, 2),
        "realized_pnl_nav_pct": round(pos.realized_pnl_nav_pct, 4),
        "exit_reason": pos.legs[-1].reason.value if pos.legs else None,
        "stages_visited": _stages_from_legs(pos.legs),
    }


def _running_nav(legs) -> list[float]:
    """% NAV qua từng leg — để tính max exposure."""
    cur = 0.0
    out = [0.0]
    for l in legs:
        cur += l.qty_pct
        out.append(cur)
    return out


def _stages_from_legs(legs) -> list[str]:
    """Suy ra các stage đã đi qua từ leg reasons."""
    seq = []
    for l in legs:
        r = l.reason.value
        if r == "probe":
            seq.append("PROBE")
        elif r == "add_strength":
            seq.append("ADDED_STRENGTH")
        elif r == "add_trend":
            seq.append("FULL")
        elif r in ("reduce_half", "reduce_partial"):
            seq.append("REDUCING")
        elif r in ("exit_all", "exit_sl", "exit_reversal"):
            seq.append("CLOSED")
    return seq


def portfolio_metrics(closed_positions: list[Position], label: Optional[str] = None) -> dict:
    """Tổng hợp toàn bộ closed positions."""
    if not closed_positions:
        return {"label": label or "", "total_positions": 0}

    pms = [position_metrics(p) for p in closed_positions]
    pnls_nav = [p["realized_pnl_nav_pct"] for p in pms if p.get("status") != "open"]
    pnls_cap = [p["pnl_pct_on_capital"] for p in pms if p.get("status") != "open"]

    wins = [p for p in pnls_nav if p > 0]
    losses = [p for p in pnls_nav if p < 0]
    sum_wins = sum(wins) if wins else 0.0
    sum_losses = abs(sum(losses)) if losses else 0.0
    pf = round(sum_wins / sum_losses, 2) if sum_losses > 0 else float("inf")

    # Stage progression analysis
    stage_counts = {"PROBE_ONLY": 0, "REACHED_ADDED": 0, "REACHED_FULL": 0}
    for p in pms:
        seq = p.get("stages_visited", [])
        if "FULL" in seq:
            stage_counts["REACHED_FULL"] += 1
        elif "ADDED_STRENGTH" in seq:
            stage_counts["REACHED_ADDED"] += 1
        else:
            stage_counts["PROBE_ONLY"] += 1

    return {
        "label":               label or "",
        "total_positions":     len(closed_positions),
        "win_count":           len(wins),
        "loss_count":          len(losses),
        "win_rate_pct":        round(len(wins) / max(1, len(pnls_nav)) * 100, 1),
        "avg_pnl_nav_pct":     round(sum(pnls_nav) / max(1, len(pnls_nav)), 3),
        "avg_pnl_capital_pct": round(sum(pnls_cap) / max(1, len(pnls_cap)), 2),
        "total_realized_nav_pct": round(sum(pnls_nav), 3),
        "profit_factor":       pf,
        "best_position_nav":   round(max(pnls_nav), 3) if pnls_nav else 0.0,
        "worst_position_nav":  round(min(pnls_nav), 3) if pnls_nav else 0.0,
        "stage_progression":   stage_counts,
    }


def buy_hold_benchmark(df: pd.DataFrame, from_date: str, to_date: str) -> dict:
    """Simple single-symbol buy-and-hold benchmark for the reported window."""
    if df is None or df.empty:
        return {}

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    start = pd.Timestamp(from_date)
    end = pd.Timestamp(to_date)
    data = data[(data["date"] >= start) & (data["date"] <= end)].sort_values("date")
    if data.empty:
        return {}

    first = data.iloc[0]
    last = data.iloc[-1]
    first_open = float(first["open"])
    first_close = float(first["close"])
    last_close = float(last["close"])
    return {
        "start_date": str(first["date"].date()),
        "end_date": str(last["date"].date()),
        "first_open": round(first_open, 2),
        "first_close": round(first_close, 2),
        "last_close": round(last_close, 2),
        "open_to_close_pct": round((last_close / first_open - 1.0) * 100.0, 2)
        if first_open > 0
        else 0.0,
        "close_to_close_pct": round((last_close / first_close - 1.0) * 100.0, 2)
        if first_close > 0
        else 0.0,
    }
