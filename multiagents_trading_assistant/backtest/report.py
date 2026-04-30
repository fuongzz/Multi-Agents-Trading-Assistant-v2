"""Xuất kết quả backtest ra console và CSV."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from multiagents_trading_assistant.backtest.positions import Trade
from multiagents_trading_assistant.backtest.metrics import (
    compute_metrics,
    compute_setup_breakdown,
    compute_symbol_breakdown,
)

_RESULTS_DIR = Path(__file__).parent.parent.parent / "backtest_results"


def print_report(
    trades: list[Trade],
    label: str = "UNIVERSE",
    from_date: str = "",
    to_date: str = "",
    show_symbol_breakdown: bool = False,
    tail: int = 15,
) -> None:
    """In báo cáo backtest ra console."""
    metrics     = compute_metrics(trades, label)
    setup_rows  = compute_setup_breakdown(trades)
    closed      = [t for t in trades if t.is_closed]

    sep = "═" * 62
    print()
    print(sep)
    print(f"  BACKTEST: {label}   {from_date} → {to_date}")
    print(sep)

    if metrics.get("total_trades", 0) == 0:
        print("  Không có trade nào trong giai đoạn này.\n")
        return

    m = metrics
    pf_str = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "∞"

    print()
    print("  TỔNG QUAN")
    print(f"    Tổng giao dịch  : {m['total_trades']}")
    print(f"    Win rate        : {m['win_rate_pct']:.1f}%  ({m['win_count']}W / {m['loss_count']}L)")
    print(f"    Avg P&L / trade : {m['avg_pnl_pct']:+.2f}%")
    print(f"    Avg win         : {m['avg_win_pct']:+.2f}%")
    print(f"    Avg loss        : {m['avg_loss_pct']:+.2f}%  (ký hiệu âm)")
    print(f"    Profit factor   : {pf_str}")
    print(f"    Best / Worst    : {m['best_trade_pct']:+.2f}% / {m['worst_trade_pct']:+.2f}%")
    print(f"    Total return    : {m['total_return_pct']:+.2f}%  (geo, pos=3% NAV)")
    print(f"    Max consec loss : {m['max_consec_losses']}")
    print(f"    Avg hold        : {m['avg_hold_bars']:.1f} bars")
    print(f"    SL / TSL / TP / TIMEOUT: {m['sl_count']} / {m.get('tsl_count', 0)} / {m['tp_count']} / {m['timeout_count']}")

    # ── Setup breakdown ──────────────────────────────────────────
    if setup_rows:
        print()
        print("  SETUP BREAKDOWN")
        hdr = f"    {'Setup':<14} {'Trades':>6} {'WR%':>6} {'AvgPnL':>8} {'PF':>6}"
        print(hdr)
        print("    " + "─" * 46)
        for r in setup_rows:
            pf = r.get("profit_factor", 0)
            pf_s = f"{pf:.2f}" if pf != float("inf") else "∞"
            print(
                f"    {r['label']:<14} {r['total_trades']:>6} "
                f"{r['win_rate_pct']:>5.1f}% {r['avg_pnl_pct']:>+7.2f}% {pf_s:>6}"
            )

    # ── Symbol breakdown (universe mode) ─────────────────────────
    if show_symbol_breakdown:
        sym_rows = compute_symbol_breakdown(trades)
        if sym_rows:
            print()
            print("  TOP SYMBOLS (theo avg P&L)")
            print(f"    {'Symbol':<8} {'Trades':>6} {'WR%':>6} {'AvgPnL':>8}")
            print("    " + "─" * 34)
            for r in sym_rows[:15]:
                print(
                    f"    {r['label']:<8} {r['total_trades']:>6} "
                    f"{r['win_rate_pct']:>5.1f}% {r['avg_pnl_pct']:>+7.2f}%"
                )

    # ── Recent trades ─────────────────────────────────────────────
    if closed:
        n = min(tail, len(closed))
        print()
        print(f"  {n} GIAO DỊCH GẦN NHẤT")
        is_universe = label not in [t.symbol for t in closed[:1]]
        sym_col = "Symbol" if is_universe else ""
        print(
            f"    {'EntryDate':>10} {sym_col:>6} {'Setup':>12} "
            f"{'Entry':>9} {'Exit':>9} {'PnL%':>7} {'Reason':>8} {'Bars':>4}"
        )
        print("    " + "─" * 75)
        for t in closed[-n:]:
            sym_str = t.symbol if is_universe else ""
            print(
                f"    {t.entry_date:>10} {sym_str:>6} {t.setup_type:>12} "
                f"{t.entry_price:>9,.0f} {t.exit_price:>9,.0f} "
                f"{t.pnl_pct:>+7.2f}% {t.exit_reason:>8} {t.holding_bars:>4}"
            )

    print()


def save_report(
    trades: list[Trade],
    label: str = "UNIVERSE",
    from_date: str = "",
    to_date: str = "",
) -> str:
    """Lưu toàn bộ trade list ra CSV. Trả về đường dẫn file."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"backtest_{label}_{from_date}_{to_date}_{ts_str}.csv"
    filepath = _RESULTS_DIR / filename

    closed = [t for t in trades if t.is_closed]
    if not closed:
        return ""

    fieldnames = [
        "symbol", "setup_type", "signal_date", "entry_date",
        "entry_price", "stop_loss", "take_profit",
        "exit_date", "exit_price", "exit_reason",
        "pnl_pct", "holding_bars", "reasons",
    ]

    with filepath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in closed:
            row = {k: getattr(t, k, "") for k in fieldnames}
            row["reasons"] = " | ".join(t.reasons) if t.reasons else ""
            writer.writerow(row)

    print(f"[backtest] Đã lưu {len(closed)} trades → {filepath}")
    return str(filepath)
