"""Rebuild the Flow V2 readable report using frozen targets and valid VND fills."""

from __future__ import annotations

import argparse
from collections import deque
from html import escape
from pathlib import Path

import pandas as pd

from scripts.backtest_flow_v2_rotation_production_like import RotationConfig, _prepare_features, run_backtest


def _fifo_realized(orders: pd.DataFrame) -> pd.DataFrame:
    queues: dict[str, deque[dict[str, float | str]]] = {}
    rows: list[dict[str, object]] = []
    for order in orders.itertuples(index=False):
        symbol = str(order.symbol)
        queue = queues.setdefault(symbol, deque())
        shares = int(order.shares)
        if order.side == "BUY":
            queue.append(
                {
                    "entry_date": str(order.date),
                    "shares": shares,
                    "unit_cost": (float(order.gross_value) + float(order.fees)) / shares,
                }
            )
            continue
        shares_left = shares
        unit_proceeds = (float(order.gross_value) - float(order.fees)) / shares
        while shares_left and queue:
            lot = queue[0]
            used = min(shares_left, int(lot["shares"]))
            cost = used * float(lot["unit_cost"])
            proceeds = used * unit_proceeds
            pnl = proceeds - cost
            rows.append(
                {
                    "symbol": symbol,
                    "entry_date": lot["entry_date"],
                    "exit_date": str(order.date),
                    "shares": used,
                    "cost_value": round(cost, 2),
                    "exit_value": round(proceeds, 2),
                    "net_pnl": round(pnl, 2),
                    "pnl_pct": round(pnl / cost * 100.0, 2) if cost else 0.0,
                    "exit_reason": order.reason,
                    "outcome": "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT",
                }
            )
            lot["shares"] = int(lot["shares"]) - used
            shares_left -= used
            if not lot["shares"]:
                queue.popleft()
    return pd.DataFrame(rows)


def _decision_windows(result: dict[str, object], targets: pd.DataFrame, cfg: RotationConfig) -> pd.DataFrame:
    equity = result["equity"].copy()
    orders = result["trades"].copy()
    dates = list(equity["date"])
    decisions = dates[:: cfg.rebalance_days]
    equity_by_date = equity.set_index("date")
    target_map = targets.groupby("signal_date", sort=False)["symbol"].apply(list).to_dict()
    rows: list[dict[str, object]] = []
    for index, signal_date in enumerate(decisions):
        execute_date = dates[dates.index(signal_date) + 1] if dates.index(signal_date) + 1 < len(dates) else None
        period_end = decisions[index + 1] if index + 1 < len(decisions) else dates[-1]
        selected = target_map.get(signal_date, [])
        execution = orders[orders["date"] == execute_date] if execute_date else orders.iloc[0:0]
        start_nav = float(equity_by_date.loc[signal_date, "equity"])
        end_nav = float(equity_by_date.loc[period_end, "equity"])
        pnl = end_nav - start_nav
        holdings = str(equity_by_date.loc[period_end, "holdings"] or "")
        rows.append(
            {
                "rebalance_no": index + 1,
                "signal_date": signal_date,
                "execute_date": execute_date,
                "period_end": period_end,
                "targets": ", ".join(selected) if selected else "CASH / GATE_CLOSED",
                "has_targets": "Yes" if selected else "No",
                "executed": "Yes" if execute_date else "No",
                "start_equity": start_nav,
                "end_equity": end_nav,
                "period_pnl": pnl,
                "period_return_pct": round(pnl / start_nav * 100.0, 2) if start_nav else 0.0,
                "outcome": "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT",
                "_buy_symbols": ", ".join(sorted(execution.loc[execution["side"] == "BUY", "symbol"])),
                "_sell_symbols": ", ".join(sorted(execution.loc[execution["side"] == "SELL", "symbol"])),
                "_buy_orders": int((execution["side"] == "BUY").sum()),
                "_sell_orders": int((execution["side"] == "SELL").sum()),
                "_buy_value": float(execution.loc[execution["side"] == "BUY", "gross_value"].sum()),
                "_sell_value": float(execution.loc[execution["side"] == "SELL", "gross_value"].sum()),
                "_fees": float(execution["fees"].sum()),
                "_holdings": holdings if holdings else "CASH",
                "_cash_weight": float(equity_by_date.loc[period_end, "cash_weight"]) * 100.0,
            }
        )
    return pd.DataFrame(rows)


def _journal(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in windows.iterrows():
        if row["_buy_orders"] or row["_sell_orders"]:
            kind = "BAN_VE_TIEN_MAT" if row["targets"] == "CASH / GATE_CLOSED" else "MUA_GIU_DANH_MUC_MUC_TIEU"
        else:
            kind = "GIU_TIEN_MAT_GATE_DONG"
        rows.append(
            {
                "Lan_rebalance": row["rebalance_no"],
                "Loai_quyet_dinh": kind,
                "Ngay_tin_hieu_T": row["signal_date"],
                "Ngay_mua_ban_rebalance_T1": row["execute_date"],
                "Ngay_chot_PnL_ky": row["period_end"],
                "Danh_muc_muc_tieu": row["targets"],
                "Ma_mua_them": row["_buy_symbols"] or "-",
                "Ma_ban_ra": row["_sell_symbols"] or "-",
                "So_lenh_mua": row["_buy_orders"],
                "So_lenh_ban": row["_sell_orders"],
                "Gia_tri_mua": row["_buy_value"],
                "Gia_tri_ban": row["_sell_value"],
                "Phi_thue_ky": row["_fees"],
                "NAV_dau_ky": row["start_equity"],
                "NAV_cuoi_ky": row["end_equity"],
                "Lai_lo_rong_ky": row["period_pnl"],
                "Ty_suat_ky_pct": row["period_return_pct"],
                "Ket_qua": "LAI" if row["outcome"] == "WIN" else "LO" if row["outcome"] == "LOSS" else "DI_NGANG",
                "Nam_giu_cuoi_ky": row["_holdings"],
                "Ty_trong_tien_mat_cuoi_ky_pct": round(row["_cash_weight"], 2),
                "Da_thuc_thi": "CO" if row["executed"] == "Yes" else "KHONG",
            }
        )
    return pd.DataFrame(rows)


def _annual(equity: pd.DataFrame) -> pd.DataFrame:
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    rows = []
    for year, group in frame.groupby("year", sort=True):
        start = float(group.iloc[0]["equity"])
        end = float(group.iloc[-1]["equity"])
        rows.append(
            {
                "year": year,
                "start_date": group.iloc[0]["date"].date().isoformat(),
                "end_date": group.iloc[-1]["date"].date().isoformat(),
                "starting_equity": start,
                "ending_equity": end,
                "pnl": end - start,
                "return_pct": round((end / start - 1.0) * 100.0, 2),
            }
        )
    return pd.DataFrame(rows)


def _export_index(
    out_dir: Path,
    report_metrics: list[tuple[str, object]],
    annual: pd.DataFrame,
    journal: pd.DataFrame,
) -> None:
    metric_cards = "".join(
        f"<div class='metric'><span>{escape(str(label))}</span><strong>{escape(str(value))}</strong></div>"
        for label, value in report_metrics
    )
    html = (
        "<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
        "<title>Flow V2 - bao cao da sua don vi gia</title>"
        "<style>*{box-sizing:border-box}body{margin:0;background:#f4f7f8;color:#18252c;"
        "font:14px Segoe UI,Arial,sans-serif}header,main{padding:22px 28px;max-width:1500px;margin:auto}"
        "header{background:#fff;max-width:none;border-bottom:1px solid #dce4e6}"
        "h1{margin:0 0 7px;font-size:24px}p{color:#52656d;line-height:1.5}"
        ".metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:8px;margin-bottom:16px}"
        ".metric,section{background:#fff;border:1px solid #d9e2e5;border-radius:6px}.metric{padding:10px}"
        ".metric span{display:block;color:#64747b;font-size:12px}.metric strong{display:block;margin-top:5px}"
        "section{padding:15px;margin:14px 0;overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}"
        "td,th{padding:6px 8px;border-bottom:1px solid #e5ecee;text-align:left}th{background:#edf3f3}</style>"
        "</head><body><header><h1>Flow V2 hieu suat cao - ban da sua thuc thi VND</h1>"
        "<p>Target lich su duoc khoa; khoi luong va NAV duoc tinh lai voi gia parquet x 1.000 va lo 100 co phieu.</p>"
        "<p><a href='bao_cao_dien_giai_rebalance_von_1_ty.html'>Bao cao dien giai de doc</a></p>"
        f"</header><main><div class='metrics'>{metric_cards}</div>"
        f"<section><h2>Hieu qua theo nam</h2>{annual.to_html(index=False, border=0)}</section>"
        f"<section><h2>Nhat ky rebalance</h2>{journal.to_html(index=False, border=0)}</section>"
        "</main></body></html>"
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def rebuild(out_dir: Path) -> dict[str, object]:
    locked_targets = pd.read_csv(out_dir / "all_targets.csv")
    cfg = RotationConfig(
        universe="vn100",
        start="2020-01-01",
        end="2026-05-22",
        positions=2,
        rebalance_days=10,
        initial_capital=1_000_000_000.0,
        selection_mode="turnover_aware",
        market_gate="risk_on_or_strong_neutral",
        pool_filter="high_rs",
        score_mode="flow_heavy",
        price_unit_multiplier=1000.0,
    )
    features = _prepare_features(cfg.universe, cfg.start, cfg.end)
    result = run_backtest(cfg, features=features, locked_targets=locked_targets)
    result["trades"].to_csv(out_dir / "all_orders.csv", index=False, encoding="utf-8-sig")
    result["equity"].to_csv(out_dir / "equity_curve_execution_corrected.csv", index=False, encoding="utf-8-sig")
    windows = _decision_windows(result, locked_targets, cfg)
    windows.drop(columns=[column for column in windows.columns if column.startswith("_")]).to_csv(
        out_dir / "all_rebalance_windows.csv", index=False, encoding="utf-8-sig"
    )
    windows[(windows["has_targets"] == "Yes")].drop(
        columns=[column for column in windows.columns if column.startswith("_")]
    ).to_csv(out_dir / "active_rebalance_windows.csv", index=False, encoding="utf-8-sig")
    journal = _journal(windows)
    journal.to_csv(out_dir / "flow_v2_nhat_ky_rebalance_von_1_ty.csv", index=False, encoding="utf-8-sig")
    realized = _fifo_realized(result["trades"])
    realized.to_csv(out_dir / "realized_roundtrips_fifo.csv", index=False, encoding="utf-8-sig")
    symbol = (
        realized.groupby("symbol", as_index=False)
        .agg(deals=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), wins=("net_pnl", lambda s: int((s > 0).sum())), losses=("net_pnl", lambda s: int((s < 0).sum())))
    )
    symbol["win_rate_pct"] = (symbol["wins"] / symbol["deals"] * 100.0).round(2)
    symbol.sort_values("net_pnl", ascending=False).to_csv(out_dir / "pnl_by_symbol.csv", index=False, encoding="utf-8-sig")
    annual = _annual(result["equity"])
    annual.to_csv(out_dir / "annual_breakdown.csv", index=False, encoding="utf-8-sig")
    summary = result["summary"]
    target_windows = windows[windows["has_targets"] == "Yes"]
    report_metrics = [
        ("Strategy", "Flow V2 high performance - locked targets, VND execution corrected"),
        ("Universe", "VN100"),
        ("Configuration", "high_rs + flow_heavy | top2 | rebalance10 | risk_on_or_strong_neutral"),
        ("Signal / Execution", "Frozen historical targets / Open T+1 in VND"),
        ("Price unit multiplier", cfg.price_unit_multiplier),
        ("Initial capital", cfg.initial_capital),
        ("From date", result["equity"].iloc[0]["date"]),
        ("To date", result["equity"].iloc[-1]["date"]),
        ("Ending equity", summary["ending_equity"]),
        ("Net P&L", summary["ending_equity"] - cfg.initial_capital),
        ("Total return (%)", summary["total_return_pct"]),
        ("Sharpe", summary["sharpe_ratio"]),
        ("Max drawdown (%)", summary["max_drawdown_pct"]),
        ("Decision windows", len(windows)),
        ("Executed rebalances", summary["rebalance_count"]),
        ("Target-bearing windows", len(target_windows)),
        ("Target windows won", int((target_windows["period_pnl"] > 0).sum())),
        ("Target windows lost", int((target_windows["period_pnl"] < 0).sum())),
        ("Buy orders", summary["buy_orders"]),
        ("Sell orders", summary["sell_orders"]),
        ("Realized FIFO segments", len(realized)),
        ("FIFO winning segments", int((realized["net_pnl"] > 0).sum())),
        ("FIFO losing segments", int((realized["net_pnl"] < 0).sum())),
        ("Fees and sell tax", summary["total_fees"]),
        ("Gross turnover (x initial capital)", summary["gross_turnover_x"]),
    ]
    pd.DataFrame(report_metrics, columns=["metric", "value"]).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_dir / "flow_v2_nhat_ky_rebalance_von_1_ty.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(report_metrics, columns=["metric", "value"]).to_excel(writer, sheet_name="Tong quan", index=False)
        journal.to_excel(writer, sheet_name="Nhat ky rebalance", index=False)
        realized.to_excel(writer, sheet_name="Giao dich FIFO", index=False)
        annual.to_excel(writer, sheet_name="Theo nam", index=False)
    _export_index(out_dir, report_metrics, annual, journal)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = rebuild(args.out_dir)
    print(summary)


if __name__ == "__main__":
    main()
