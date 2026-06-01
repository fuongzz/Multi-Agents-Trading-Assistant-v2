"""Compare Flow V2 reversal guards on the VN100 production-like execution model."""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.backtest_flow_v2_rotation_production_like import RotationConfig, _max_drawdown, _prepare_features, run_backtest


ROOT = Path(__file__).resolve().parents[1]


def _period_metrics(equity: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    period = equity.loc[(equity["date"] >= start) & (equity["date"] <= end)].copy()
    if period.empty:
        return {"return_pct": 0.0, "max_drawdown_pct": 0.0}
    values = period["equity"].astype(float)
    return {
        "return_pct": round(float(values.iloc[-1] / values.iloc[0] - 1.0) * 100.0, 2),
        "max_drawdown_pct": round(_max_drawdown(values) * 100.0, 2),
    }


def _configs(start: str, end: str, capital: float) -> list[tuple[str, str, RotationConfig]]:
    base: dict[str, Any] = {
        "universe": "vn100",
        "start": start,
        "end": end,
        "rebalance_days": 10,
        "initial_capital": capital,
        "market_gate": "risk_on_or_strong_neutral",
        "pool_filter": "high_rs",
        "score_mode": "flow_heavy",
        "price_unit_multiplier": 1000.0,
    }
    return [
        ("baseline_top2", "Goc: top2, khong lop chong dao chieu", RotationConfig(positions=2, **base)),
        (
            "cap20_top2",
            "Top2, chan mua neu tang 20 phien >20% hoac cach MA20 >15%",
            RotationConfig(positions=2, max_entry_ret_20d=0.20, max_entry_distance_ma20=0.15, **base),
        ),
        (
            "exit_top2",
            "Top2, thoat som khi momentum/flow gay",
            RotationConfig(positions=2, early_exit_mode="flow_momentum_break", **base),
        ),
        (
            "exit_top3",
            "Top3, thoat som khi momentum/flow gay",
            RotationConfig(positions=3, early_exit_mode="flow_momentum_break", **base),
        ),
        (
            "exit_top4",
            "Top4, thoat som khi momentum/flow gay",
            RotationConfig(positions=4, early_exit_mode="flow_momentum_break", **base),
        ),
        (
            "guard20_top2",
            "Top2, cap20 + thoat som",
            RotationConfig(
                positions=2,
                max_entry_ret_20d=0.20,
                max_entry_distance_ma20=0.15,
                early_exit_mode="flow_momentum_break",
                **base,
            ),
        ),
        (
            "guard20_top3",
            "Top3, cap20 + thoat som",
            RotationConfig(
                positions=3,
                max_entry_ret_20d=0.20,
                max_entry_distance_ma20=0.15,
                early_exit_mode="flow_momentum_break",
                **base,
            ),
        ),
        (
            "guard20_top4",
            "Top4, cap20 + thoat som",
            RotationConfig(
                positions=4,
                max_entry_ret_20d=0.20,
                max_entry_distance_ma20=0.15,
                early_exit_mode="flow_momentum_break",
                **base,
            ),
        ),
        (
            "guard30_top2",
            "Top2, cap long hon: tang 20 phien <=30%, cach MA20 <=20%, va thoat som",
            RotationConfig(
                positions=2,
                max_entry_ret_20d=0.30,
                max_entry_distance_ma20=0.20,
                early_exit_mode="flow_momentum_break",
                **base,
            ),
        ),
        (
            "guard30_top3",
            "Top3, cap long hon: tang 20 phien <=30%, cach MA20 <=20%, va thoat som",
            RotationConfig(
                positions=3,
                max_entry_ret_20d=0.30,
                max_entry_distance_ma20=0.20,
                early_exit_mode="flow_momentum_break",
                **base,
            ),
        ),
        (
            "guard30_top4",
            "Top4, cap long hon: tang 20 phien <=30%, cach MA20 <=20%, va thoat som",
            RotationConfig(
                positions=4,
                max_entry_ret_20d=0.30,
                max_entry_distance_ma20=0.20,
                early_exit_mode="flow_momentum_break",
                **base,
            ),
        ),
    ]


def _export_html(out_dir: Path, rows: pd.DataFrame, start: str, end: str) -> None:
    ordered = rows.sort_values(["max_drawdown_pct", "total_return_pct"], ascending=[False, False])
    best_return = rows.sort_values("total_return_pct", ascending=False).iloc[0]
    best_2024_dd = rows.sort_values("return_2024_pct", ascending=False).iloc[0]
    table_rows = []
    for _, row in ordered.iterrows():
        table_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(row['variant'])}</strong><br><span>{html.escape(row['description'])}</span></td>"
            f"<td>{row['positions']}</td>"
            f"<td>{row['total_return_pct']:+.2f}%</td>"
            f"<td>{row['sharpe_ratio']:.3f}</td>"
            f"<td>{row['max_drawdown_pct']:.2f}%</td>"
            f"<td>{row['return_2024_pct']:+.2f}%</td>"
            f"<td>{row['drawdown_2024_pct']:.2f}%</td>"
            f"<td>{row['return_2025_now_pct']:+.2f}%</td>"
            f"<td>{int(row['early_exit_orders'])}</td>"
            f"<td>{int(row['number_of_orders'])}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>Flow V2 - thu nghiem chong dao chieu nganh</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #1b2430; line-height: 1.45; }}
h1 {{ font-size: 25px; margin-bottom: 6px; }} h2 {{ margin-top: 28px; font-size: 19px; }}
.sub {{ color: #52616d; margin-top: 0; }} .note {{ background: #f4f7f8; border-left: 4px solid #287271; padding: 12px 16px; max-width: 1060px; }}
table {{ border-collapse: collapse; width: 100%; max-width: 1280px; margin-top: 14px; font-size: 14px; }}
th, td {{ border: 1px solid #d8dee3; padding: 9px 10px; text-align: right; vertical-align: top; }}
th:first-child, td:first-child {{ text-align: left; }} th {{ background: #eef3f5; }} td span {{ color: #61727f; font-size: 12px; }}
code {{ background: #f1f3f5; padding: 1px 4px; }} ul {{ max-width: 1060px; }}
</style></head><body>
<h1>MVP Flow V2: thử nghiệm lớp chống đảo chiều ngành</h1>
<p class="sub">VN100 | {start} đến {end} | vốn 1 tỷ đồng | rebalance 10 phiên | khớp open T+1 | giá VND và lô 100 cổ phiếu</p>
<p class="note">Đây là backtest chọn lại tín hiệu trên feature đã đồng bộ đến 22/05/2026, không phải replay target lịch sử đã khóa trong báo cáo +712,03%. Mỗi quyết định dùng dữ liệu sau đóng cửa T và chỉ được giao dịch ở open T+1; lớp thoát sớm cũng tuân theo nguyên tắc này.</p>
<h2>Kết quả so sánh</h2>
<table><thead><tr><th>Biến thể</th><th>Top</th><th>Lợi nhuận toàn kỳ</th><th>Sharpe</th><th>Max DD</th><th>2024</th><th>DD 2024</th><th>2025 đến nay</th><th>Lệnh thoát sớm</th><th>Tổng lệnh</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<h2>Logic được kiểm thử</h2>
<ul>
<li><strong>Entry extension cap:</strong> không mở mua mới khi lợi nhuận 20 phiên vượt ngưỡng hoặc giá đã cách MA20 quá xa; mục tiêu là tránh mua cuối nhịp kéo ngành.</li>
<li><strong>Early exit:</strong> trong ngày không phải kỳ rebalance, nếu mã đang giữ rơi dưới MA50, hoặc flow yếu đi đồng thời RS giảm, hoặc distribution tăng đồng thời RS giảm, hệ thống phát lệnh bán cho phiên kế tiếp.</li>
<li><strong>Phân tán:</strong> cùng một overlay được thử với top2, top3 và top4 để đo đổi chác giữa tập trung và drawdown.</li>
</ul>
<h2>Điểm nổi bật</h2>
<p>Biến thể có lợi nhuận toàn kỳ cao nhất trong nhóm kiểm thử là <strong>{html.escape(best_return['variant'])}</strong>, đạt <strong>{best_return['total_return_pct']:+.2f}%</strong>, Sharpe {best_return['sharpe_ratio']:.3f}, max drawdown {best_return['max_drawdown_pct']:.2f}%.</p>
<p>Biến thể có kết quả năm 2024 tốt nhất là <strong>{html.escape(best_2024_dd['variant'])}</strong>, đạt <strong>{best_2024_dd['return_2024_pct']:+.2f}%</strong> trong năm này với drawdown {best_2024_dd['drawdown_2024_pct']:.2f}%.</p>
<p>Các file chi tiết theo biến thể gồm <code>*_equity.csv</code>, <code>*_orders.csv</code>, <code>*_targets.csv</code>; bảng tổng hợp nằm tại <code>summary.csv</code>.</p>
</body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def run_experiment(out_dir: Path, start: str, end: str, capital: float) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    features = _prepare_features("vn100", start, end)
    rows: list[dict[str, Any]] = []
    for variant, description, cfg in _configs(start, end, capital):
        print(f"[reversal-guard] running {variant}")
        result = run_backtest(cfg, features=features)
        trades = result["trades"]
        p2024 = _period_metrics(result["equity"], "2024-01-01", "2024-12-31")
        p2025_now = _period_metrics(result["equity"], "2025-01-01", end)
        row = dict(result["summary"])
        row.update(
            {
                "variant": variant,
                "description": description,
                "return_2024_pct": p2024["return_pct"],
                "drawdown_2024_pct": p2024["max_drawdown_pct"],
                "return_2025_now_pct": p2025_now["return_pct"],
                "drawdown_2025_now_pct": p2025_now["max_drawdown_pct"],
                "early_exit_orders": int(
                    (trades["reason"] == "EARLY_EXIT_FLOW_MOMENTUM_BREAK").sum()
                )
                if not trades.empty
                else 0,
            }
        )
        rows.append(row)
        result["equity"].to_csv(out_dir / f"{variant}_equity.csv", index=False)
        result["trades"].to_csv(out_dir / f"{variant}_orders.csv", index=False)
        result["targets"].to_csv(out_dir / f"{variant}_targets.csv", index=False)
        print(
            f"  ret={row['total_return_pct']:+.2f}% sharpe={row['sharpe_ratio']:.3f} "
            f"dd={row['max_drawdown_pct']:.2f}% 2024={row['return_2024_pct']:+.2f}%"
        )
    summary = pd.DataFrame(rows).sort_values(["total_return_pct", "sharpe_ratio"], ascending=[False, False])
    summary.to_csv(out_dir / "summary.csv", index=False)
    _export_html(out_dir, summary, start, end)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow V2 reversal guard comparison on VN100")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-22")
    parser.add_argument("--capital", type=float, default=1_000_000_000.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "flow_v2_reversal_guard_experiment_vn100_2020_to_2026-05-22",
    )
    args = parser.parse_args()
    summary = run_experiment(args.out_dir, args.start, args.end, args.capital)
    print(summary[["variant", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "return_2024_pct"]].to_string(index=False))
    print(f"[reversal-guard] saved {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
