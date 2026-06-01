"""Detailed from-source audit rerun for Flow V2 on the operating VN100 universe."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from scripts.backtest_flow_v2_rotation_production_like import RotationConfig, _prepare_features, run_backtest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "flow_v2_detailed_audit_vn100_2020_to_2026-05-22"
INITIAL_CAPITAL = 1_000_000_000.0
START = "2020-01-01"
END = "2026-05-22"


def _config(start: str, end: str, tiered_exit: bool) -> RotationConfig:
    return RotationConfig(
        universe="vn100",
        start=start,
        end=end,
        positions=2,
        rebalance_days=10,
        initial_capital=INITIAL_CAPITAL,
        market_gate="risk_on_or_strong_neutral",
        pool_filter="high_rs",
        score_mode="flow_heavy",
        price_unit_multiplier=1000.0,
        early_exit_mode="flow_momentum_tiered" if tiered_exit else "none",
    )


def _money(value: float) -> str:
    return f"{value / 1_000_000:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".") + " triệu"


def _annual_rows(equity: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    rows: list[dict[str, Any]] = []
    prior = INITIAL_CAPITAL
    for year, group in frame.groupby("year", sort=True):
        end_value = float(group.iloc[-1]["equity"])
        rows.append(
            {
                "variant": label,
                "year": int(year),
                "starting_equity": round(prior, 2),
                "ending_equity": round(end_value, 2),
                "pnl": round(end_value - prior, 2),
                "return_pct": round((end_value / prior - 1.0) * 100.0, 2),
            }
        )
        prior = end_value
    return rows


def _drawdown_detail(equity: pd.DataFrame, label: str) -> dict[str, Any]:
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    series = frame.set_index("date")["equity"].astype(float)
    running_peak = series.cummax()
    drawdown = series / running_peak - 1.0
    trough_date = drawdown.idxmin()
    peak_value = float(running_peak.loc[trough_date])
    peak_date = series.loc[:trough_date][series.loc[:trough_date] == peak_value].index[-1]
    after = series.loc[trough_date:]
    recovered = after[after >= peak_value]
    recovery_date = recovered.index[0].date().isoformat() if not recovered.empty else "Chua hoi phuc den cuoi ky"
    return {
        "variant": label,
        "peak_date": peak_date.date().isoformat(),
        "peak_equity": round(peak_value, 2),
        "trough_date": trough_date.date().isoformat(),
        "trough_equity": round(float(series.loc[trough_date]), 2),
        "drawdown_pct": round(float(drawdown.loc[trough_date]) * 100.0, 2),
        "recovery_date": recovery_date,
    }


def _audit_execution(result: dict[str, Any], label: str) -> list[dict[str, str]]:
    trades = result["trades"].copy()
    targets = result["targets"].copy()
    equity = result["equity"].copy()
    strategy_trades = trades.loc[trades["reason"] != "FINAL_LIQUIDATION"].copy()
    strategy_signal = pd.to_datetime(strategy_trades["signal_date"], errors="coerce")
    strategy_execution = pd.to_datetime(strategy_trades["date"], errors="coerce")
    executed_targets = targets.dropna(subset=["execute_date"]).copy()
    target_signal = pd.to_datetime(executed_targets["signal_date"], errors="coerce")
    target_execution = pd.to_datetime(executed_targets["execute_date"], errors="coerce")
    checks = [
        (
            "Tín hiệu -> lệnh chiến lược",
            strategy_signal.notna().all() and (strategy_execution > strategy_signal).all(),
            f"{len(strategy_trades)} lệnh có signal_date và đều khớp sau ngày tín hiệu",
        ),
        (
            "Target rebalance -> ngày thi hành",
            (target_execution > target_signal).all(),
            f"{len(executed_targets)} target có execute_date > signal_date",
        ),
        (
            "Giới hạn số mã",
            int(equity["positions"].max()) <= 2,
            f"tối đa quan sát thấy {int(equity['positions'].max())} mã đang nắm giữ",
        ),
        (
            "Tiền mặt không âm",
            float(equity["cash"].min()) >= -0.01,
            f"cash thấp nhất {_money(float(equity['cash'].min()))}",
        ),
        (
            "Đơn vị giá thi hành",
            not trades.empty and float(trades["price"].median()) > 1000.0,
            f"giá khớp trung vị {_money(float(trades['price'].median()))}/cổ phiếu; parquet đã nhân 1.000",
        ),
    ]
    return [
        {"variant": label, "check": name, "status": "PASS" if passed else "FAIL", "detail": detail}
        for name, passed, detail in checks
    ]


def _run_variant(features: pd.DataFrame, start: str, end: str, label: str, tiered_exit: bool) -> dict[str, Any]:
    result = run_backtest(_config(start, end, tiered_exit), features=features)
    result["summary"].update({"variant": label, "start": start, "end": end})
    return result


def _table(rows: list[str], header: str) -> str:
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _html_report(
    out_dir: Path,
    full_summary: pd.DataFrame,
    period_summary: pd.DataFrame,
    annual: pd.DataFrame,
    drawdowns: pd.DataFrame,
    checks: pd.DataFrame,
    exits: pd.DataFrame,
    data_audit: list[tuple[str, str]],
    symbol_count: int,
) -> None:
    full_rows = [
        "<tr>"
        f"<td><strong>{escape(str(row.variant))}</strong></td>"
        f"<td>{float(row.total_return_pct):+.2f}%</td><td>{_money(float(row.ending_equity))}</td>"
        f"<td>{float(row.sharpe_ratio):.3f}</td><td>{float(row.max_drawdown_pct):.2f}%</td>"
        f"<td>{int(row.number_of_orders)}</td><td>{_money(float(row.total_fees))}</td></tr>"
        for row in full_summary.itertuples()
    ]
    period_rows = [
        "<tr>"
        f"<td>{escape(str(row.period))}</td><td>{escape(str(row.variant))}</td>"
        f"<td>{float(row.total_return_pct):+.2f}%</td><td>{float(row.sharpe_ratio):.3f}</td>"
        f"<td>{float(row.max_drawdown_pct):.2f}%</td><td>{_money(float(row.ending_equity))}</td></tr>"
        for row in period_summary.itertuples()
    ]
    annual_rows = [
        "<tr>"
        f"<td>{int(row.year)}</td><td>{escape(str(row.variant))}</td>"
        f"<td>{float(row.return_pct):+.2f}%</td><td>{_money(float(row.pnl))}</td>"
        f"<td>{_money(float(row.ending_equity))}</td></tr>"
        for row in annual.itertuples()
    ]
    dd_rows = [
        "<tr>"
        f"<td>{escape(str(row.variant))}</td><td>{row.peak_date}</td><td>{_money(float(row.peak_equity))}</td>"
        f"<td>{row.trough_date}</td><td>{_money(float(row.trough_equity))}</td>"
        f"<td>{float(row.drawdown_pct):.2f}%</td><td>{escape(str(row.recovery_date))}</td></tr>"
        for row in drawdowns.itertuples()
    ]
    check_rows = [
        "<tr>"
        f"<td>{escape(str(row.variant))}</td><td>{escape(str(row.check))}</td>"
        f"<td class=\"{str(row.status).lower()}\">{row.status}</td><td>{escape(str(row.detail))}</td></tr>"
        for row in checks.itertuples()
    ]
    exit_rows = [
        "<tr>"
        f"<td>{row.signal_date}</td><td>{row.date}</td><td>{escape(str(row.symbol))}</td>"
        f"<td>{'Giảm 1/2 vị thế' if row.reason == 'PARTIAL_EXIT_FLOW_WEAKNESS' else 'Thoát hết sớm'}</td>"
        f"<td>{int(row.shares):,}</td><td>{_money(float(row.gross_value))}</td><td>{_money(float(row.fees))}</td></tr>"
        for row in exits.itertuples()
    ]
    source_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{escape(value)}</td></tr>" for name, value in data_audit
    )
    document = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>Flow V2 - backtest kiểm định chi tiết</title>
<style>
body {{ margin:0; background:#f4f7f8; color:#16242b; font:15px Segoe UI,Arial,sans-serif; line-height:1.55; }}
main {{ max-width:1260px; margin:22px auto; padding:30px; background:#fff; border:1px solid #dce4e6; border-radius:6px; }}
h1 {{ font-size:27px; margin:0 0 6px; color:#146b62; }} h2 {{ font-size:19px; color:#146b62; margin:30px 0 12px; border-top:1px solid #e2e9ea; padding-top:18px; }}
.meta {{ color:#52656d; margin-bottom:18px; }} .note {{ border-left:4px solid #146b62; background:#f2f7f6; padding:11px 14px; margin:14px 0; }}
.warning {{ border-left:4px solid #a55b14; background:#fff7ed; padding:11px 14px; margin:14px 0; }}
table {{ border-collapse:collapse; width:100%; margin:11px 0; font-size:14px; }}
th, td {{ border:1px solid #dce4e6; padding:8px 9px; text-align:right; vertical-align:top; }}
th {{ background:#edf4f3; color:#214342; }} th:first-child,td:first-child {{ text-align:left; }}
td:nth-child(2) {{ text-align:left; }} .pass {{ color:#12654e; font-weight:700; }} .fail {{ color:#a41e22; font-weight:700; }}
a {{ color:#146b62; font-weight:600; }} code {{ font-size:13px; }}
</style></head><body><main>
<h1>Backtest kiểm định lại từ đầu - Flow V2 VN100</h1>
<p class="meta">Vốn giả lập 1 tỷ đồng | 01/01/2020 - 22/05/2026 | top2 | rebalance 10 phiên | tín hiệu sau close T, khớp open T+1 | chi phí all-in một deal mua-bán xấp xỉ 0,40%</p>
<p class="note"><strong>Mục đích:</strong> Tái chạy trực tiếp từ dữ liệu nguồn đã đồng bộ, có lưu <code>signal_date</code> trên mỗi lệnh để kiểm tra look-ahead. Hai cấu hình được đối chiếu là baseline fresh-signal và early-exit hai tầng top2 đã được chọn từ vòng nghiên cứu trước; đây không phải một lần tìm tham số mới.</p>
<p class="warning"><strong>Hạn chế quan trọng:</strong> Backtest dùng {symbol_count} mã VN100 được resolve tại thời điểm chạy. Chưa có bảng thành phần VN100 point-in-time theo từng ngày, nên kết quả vẫn có thể bị survivorship bias. Kết quả này dùng để so sánh logic trên cùng tập mã, chưa đủ để coi là ước tính lợi nhuận live không thiên lệch.</p>
<h2>1. Kiểm tra nguồn và quy tắc</h2>
<table><tbody>{source_rows}</tbody></table>
{_table(check_rows, "<th>Biến thể</th><th>Kiểm tra</th><th>Trạng thái</th><th>Chi tiết</th>")}
<h2>2. Kết quả toàn kỳ tái chạy</h2>
{_table(full_rows, "<th>Cấu hình</th><th>Lợi nhuận</th><th>NAV cuối kỳ</th><th>Sharpe</th><th>Max DD</th><th>Số lệnh</th><th>Phí/thuế</th>")}
<p>Early-exit hai tầng giảm nửa vị thế khi flow suy yếu; chỉ thoát hết khi momentum/flow gãy mạnh. Trên cùng pipeline tái chạy, biến thể này cân đối lại lợi nhuận và mức sụt giảm tốt hơn baseline.</p>
<h2>3. Kiểm tra theo từng giai đoạn, khởi tạo lại vốn 1 tỷ</h2>
<p>Mỗi dòng dưới đây là một backtest khởi tạo lại vốn riêng biệt trong giai đoạn đó. Đây là kiểm tra độ bền, không phải out-of-sample thuần túy vì overlay đã được đề xuất dựa trên quan sát lịch sử trước đó.</p>
{_table(period_rows, "<th>Giai đoạn</th><th>Cấu hình</th><th>Lợi nhuận</th><th>Sharpe</th><th>Max DD</th><th>NAV cuối kỳ</th>")}
<h2>4. Kết quả từng năm trên đường NAV toàn kỳ</h2>
{_table(annual_rows, "<th>Năm</th><th>Cấu hình</th><th>Lợi nhuận</th><th>Lãi/lỗ</th><th>NAV cuối năm</th>")}
<h2>5. Max drawdown lớn nhất</h2>
{_table(dd_rows, "<th>Cấu hình</th><th>Ngày đỉnh</th><th>NAV đỉnh</th><th>Ngày đáy</th><th>NAV đáy</th><th>Drawdown</th><th>Ngày hồi phục</th>")}
<h2>6. Các lệnh thoát sớm của early-exit hai tầng</h2>
<p>Bảng này hiển thị tất cả lệnh do overlay tạo ra; các giao dịch rebalance định kỳ vẫn nằm trong tệp lệnh chi tiết kèm theo.</p>
{_table(exit_rows, "<th>Ngày tín hiệu</th><th>Ngày khớp</th><th>Mã</th><th>Hành động</th><th>Số cổ phiếu</th><th>Giá trị</th><th>Phí/thuế</th>")}
<h2>7. Tệp kiểm tra kèm theo</h2>
<p><a href="full_period_summary.csv">Tổng hợp toàn kỳ</a> | <a href="period_robustness_summary.csv">Tổng hợp theo giai đoạn</a> | <a href="annual_breakdown.csv">Từng năm</a> | <a href="execution_audit_checks.csv">Checklist T+1</a> | <a href="early_exit_tiered_top2_orders.csv">Toàn bộ lệnh của phương án hai tầng</a> | <a href="early_exit_tiered_top2_targets.csv">Danh mục mục tiêu</a></p>
</main></body></html>"""
    (out_dir / "index.html").write_text(document, encoding="utf-8")


def run_audit(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[flow-v2-audit] preparing full-period features with built-in warmup")
    full_features = _prepare_features("vn100", START, END)
    variants = [
        ("fresh_baseline_top2", False),
        ("early_exit_tiered_top2", True),
    ]
    full_results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    drawdown_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, str]] = []
    for label, tiered in variants:
        print(f"[flow-v2-audit] running full period: {label}")
        result = _run_variant(full_features, START, END, label, tiered)
        full_results[label] = result
        summary_rows.append(result["summary"])
        annual_rows.extend(_annual_rows(result["equity"], label))
        drawdown_rows.append(_drawdown_detail(result["equity"], label))
        audit_rows.extend(_audit_execution(result, label))
        result["equity"].to_csv(out_dir / f"{label}_equity.csv", index=False)
        result["targets"].to_csv(out_dir / f"{label}_targets.csv", index=False)
        result["trades"].to_csv(out_dir / f"{label}_orders.csv", index=False)

    periods = [
        ("2020-2021 phục hồi/tăng mạnh", "2020-01-01", "2021-12-31"),
        ("2022-2023 điều chỉnh/hồi phục", "2022-01-01", "2023-12-31"),
        ("2024 kiểm tra sideway", "2024-01-01", "2024-12-31"),
        ("2025 đến 22/05/2026 gần đây", "2025-01-01", END),
    ]
    period_rows: list[dict[str, Any]] = []
    for period, start, end in periods:
        print(f"[flow-v2-audit] running clean-capital period: {period}")
        for label, tiered in variants:
            result = _run_variant(full_features, start, end, label, tiered)
            row = dict(result["summary"])
            row["period"] = period
            period_rows.append(row)

    full_summary = pd.DataFrame(summary_rows)
    period_summary = pd.DataFrame(period_rows)
    annual = pd.DataFrame(annual_rows)
    drawdowns = pd.DataFrame(drawdown_rows)
    checks = pd.DataFrame(audit_rows)
    exits = full_results["early_exit_tiered_top2"]["trades"].loc[
        full_results["early_exit_tiered_top2"]["trades"]["reason"].isin(
            ["PARTIAL_EXIT_FLOW_WEAKNESS", "EARLY_EXIT_FLOW_MOMENTUM_BREAK"]
        )
    ].copy()

    raw_paths = [
        ("OHLCV master", ROOT / "multiagents_trading_assistant/data/ohlcv_master.parquet"),
        ("Money Cycle thị trường", ROOT / "data/research/money_cycle/money_cycle_market.parquet"),
        ("Smart Money Trace", ROOT / "data/research/smart_money_trace/smart_money_by_symbol.parquet"),
    ]
    data_audit: list[tuple[str, str]] = []
    for name, path in raw_paths:
        frame = pd.read_parquet(path, columns=["date"])
        dates = pd.to_datetime(frame["date"])
        data_audit.append((name, f"{dates.min().date().isoformat()} đến {dates.max().date().isoformat()}, {len(frame):,} dòng"))
    data_audit.extend(
        [
            ("Warm-up feature", "370 ngày lịch trước ngày bắt đầu mỗi lần backtest"),
            ("Quy tắc tín hiệu", "Dùng dữ liệu đến close T; mọi lệnh chiến lược khớp open T+1"),
            ("Thanh lý cuối kỳ", "Bán tại close ngày 22/05/2026 chỉ để chốt NAV; không tính là tín hiệu giao dịch T+1"),
            ("Chi phí giao dịch", "Mua 0,10% phí + 0,05% trượt giá; bán 0,10% phí + 0,10% thuế + 0,05% trượt giá"),
            ("Lô giao dịch", "100 cổ phiếu; đơn vị giá parquet được nhân 1.000 sang VND"),
        ]
    )

    full_summary.to_csv(out_dir / "full_period_summary.csv", index=False)
    period_summary.to_csv(out_dir / "period_robustness_summary.csv", index=False)
    annual.to_csv(out_dir / "annual_breakdown.csv", index=False)
    drawdowns.to_csv(out_dir / "max_drawdown_detail.csv", index=False)
    checks.to_csv(out_dir / "execution_audit_checks.csv", index=False)
    exits.to_csv(out_dir / "early_exit_events.csv", index=False)
    symbol_count = len(combo_harness.resolve_symbols("vn100"))
    _html_report(out_dir, full_summary, period_summary, annual, drawdowns, checks, exits, data_audit, symbol_count)
    return full_summary, period_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed Flow V2 audit rerun")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    full, periods = run_audit(args.out_dir)
    print(full[["variant", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "ending_equity"]].to_string(index=False))
    print(periods[["period", "variant", "total_return_pct", "sharpe_ratio", "max_drawdown_pct"]].to_string(index=False))
    print(f"[flow-v2-audit] saved {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
