"""Tính thống kê hiệu suất backtest."""

from collections import defaultdict
from typing import Optional

from multiagents_trading_assistant.backtest.positions import Trade


def compute_metrics(trades: list[Trade], label: Optional[str] = None) -> dict:
    """Tính các chỉ số hiệu suất từ danh sách Trade đã đóng.

    Returns dict với keys:
        label, total_trades, win_count, loss_count, win_rate_pct,
        avg_pnl_pct, avg_win_pct, avg_loss_pct, profit_factor,
        max_consec_losses, avg_hold_bars, sl_count, tp_count, timeout_count,
        best_trade_pct, worst_trade_pct, total_return_pct (geometric approx)
    """
    closed = [t for t in trades if t.is_closed and t.pnl_pct is not None]
    if not closed:
        return {"label": label or "", "total_trades": 0}

    pnls    = [t.pnl_pct for t in closed]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p < 0]

    sl_count      = sum(1 for t in closed if t.exit_reason == "SL")
    tsl_count     = sum(1 for t in closed if t.exit_reason == "TSL")
    tp_count      = sum(1 for t in closed if t.exit_reason == "TP")
    timeout_count = sum(1 for t in closed if t.exit_reason == "TIMEOUT")

    # Max consecutive losses
    max_consec = cur_consec = 0
    for t in closed:
        if t.is_loss:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    sum_wins   = sum(wins)          if wins   else 0.0
    sum_losses = abs(sum(losses))   if losses else 0.0
    pf = round(sum_wins / sum_losses, 2) if sum_losses > 0 else float("inf")

    # Dynamic position sizing: confluence ≥70 → 5% NAV, 55-69 → 3%, <55 but >0 → 2%, no review → 3%
    equity = 1.0
    pos_sizes: list[float] = []
    for t in closed:
        c = t.confluence_score
        if c >= 70:
            pos = 0.05
        elif c >= 55:
            pos = 0.03
        elif c > 0:
            pos = 0.02
        else:
            pos = 0.03  # baseline: no pipeline_review info
        pos_sizes.append(pos)
        equity *= 1.0 + (t.pnl_pct / 100.0) * pos
    total_return_pct = round((equity - 1.0) * 100, 2)
    avg_position_pct = round(sum(pos_sizes) / len(pos_sizes) * 100, 1)

    return {
        "label":             label or "",
        "total_trades":      len(closed),
        "win_count":         len(wins),
        "loss_count":        len(losses),
        "win_rate_pct":      round(len(wins) / len(closed) * 100, 1),
        "avg_pnl_pct":       round(sum(pnls) / len(pnls), 2),
        "avg_win_pct":       round(sum_wins / len(wins), 2)       if wins   else 0.0,
        "avg_loss_pct":      round(-sum_losses / len(losses), 2)  if losses else 0.0,
        "profit_factor":     pf,
        "max_consec_losses": max_consec,
        "avg_hold_bars":     round(sum(t.holding_bars for t in closed) / len(closed), 1),
        "sl_count":          sl_count,
        "tsl_count":         tsl_count,
        "tp_count":          tp_count,
        "timeout_count":     timeout_count,
        "best_trade_pct":    round(max(pnls), 2),
        "worst_trade_pct":   round(min(pnls), 2),
        "total_return_pct":  total_return_pct,
        "avg_position_pct":  avg_position_pct,
    }


def compute_setup_breakdown(trades: list[Trade]) -> list[dict]:
    """Nhóm trades theo setup_type và tính thống kê cho từng nhóm.

    Returns:
        List dict mỗi setup, sort theo total_trades giảm dần.
    """
    by_setup: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        if t.is_closed:
            by_setup[t.setup_type].append(t)

    rows = []
    for setup_type, setup_trades in by_setup.items():
        m = compute_metrics(setup_trades, label=setup_type)
        rows.append(m)

    rows.sort(key=lambda r: r.get("total_trades", 0), reverse=True)
    return rows


def compute_symbol_breakdown(trades: list[Trade]) -> list[dict]:
    """Nhóm trades theo symbol và tính thống kê cho từng mã.

    Returns:
        List dict mỗi mã, sort theo avg_pnl_pct giảm dần.
    """
    by_sym: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        if t.is_closed:
            by_sym[t.symbol].append(t)

    rows = []
    for sym, sym_trades in by_sym.items():
        m = compute_metrics(sym_trades, label=sym)
        rows.append(m)

    rows.sort(key=lambda r: r.get("avg_pnl_pct", 0.0), reverse=True)
    return rows
