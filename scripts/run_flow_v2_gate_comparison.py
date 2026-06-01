"""Compare Flow V2 market gates under the corrected execution contract."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.backtest_flow_v2_rotation_production_like import RotationConfig, _prepare_features, run_backtest


ROOT = Path(__file__).resolve().parents[1]
GATES = [
    ("none", "Không gate: chỉ giữ bộ lọc chất lượng cổ phiếu"),
    ("base", "Gate mềm: không vào khi RISK_OFF"),
    ("risk_on_or_strong_neutral", "Gate đang dùng: RISK_ON hoặc NEUTRAL mạnh"),
    ("strict", "Gate chặt: chỉ RISK_ON mạnh"),
]
EXITS = [
    ("none", "Baseline rebalance"),
    ("flow_momentum_tiered", "Early-exit hai tầng"),
]


def _config(start: str, end: str, capital: float, gate: str, exit_mode: str) -> RotationConfig:
    return RotationConfig(
        universe="vn100",
        start=start,
        end=end,
        positions=2,
        rebalance_days=10,
        initial_capital=capital,
        market_gate=gate,
        pool_filter="high_rs",
        score_mode="flow_heavy",
        price_unit_multiplier=1000.0,
        early_exit_mode=exit_mode,
    )


def _run_period(
    features: pd.DataFrame,
    period: str,
    start: str,
    end: str,
    capital: float,
    out_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate, gate_label in GATES:
        for exit_mode, exit_label in EXITS:
            variant = f"{gate}_{exit_mode}"
            print(f"[flow-v2-gates] {period} {variant}", flush=True)
            result = run_backtest(_config(start, end, capital, gate, exit_mode), features=features)
            summary = dict(result["summary"])
            summary.update(
                {
                    "period": period,
                    "start": start,
                    "end": end,
                    "market_gate": gate,
                    "gate_label": gate_label,
                    "exit_mode": exit_mode,
                    "exit_label": exit_label,
                }
            )
            rows.append(summary)
            stem = f"{period}_{variant}"
            result["equity"].to_csv(out_dir / f"{stem}_equity.csv", index=False)
            result["targets"].to_csv(out_dir / f"{stem}_targets.csv", index=False)
            result["trades"].to_csv(out_dir / f"{stem}_orders.csv", index=False)
            print(
                f"  return={summary['total_return_pct']:+.2f}% "
                f"sharpe={summary['sharpe_ratio']:.3f} dd={summary['max_drawdown_pct']:.2f}% "
                f"orders={summary['number_of_orders']}",
                flush=True,
            )
    return rows


def _export_html(out_dir: Path, summary: pd.DataFrame, capital: float) -> None:
    sections: list[str] = []
    for period, frame in summary.groupby("period", sort=False):
        rows = []
        for row in frame.sort_values(["exit_mode", "market_gate"]).to_dict("records"):
            rows.append(
                "<tr>"
                f"<td>{escape(str(row['gate_label']))}</td>"
                f"<td>{escape(str(row['exit_label']))}</td>"
                f"<td>{float(row['total_return_pct']):+.2f}%</td>"
                f"<td>{float(row['sharpe_ratio']):.3f}</td>"
                f"<td>{float(row['max_drawdown_pct']):.2f}%</td>"
                f"<td>{int(row['number_of_orders'])}</td>"
                f"<td>{float(row['gross_turnover_x']):.2f}x</td>"
                "</tr>"
            )
        start = str(frame.iloc[0]["start"])
        end = str(frame.iloc[0]["end"])
        sections.append(
            f"<h2>{escape(str(period))} <small>{escape(start)} đến {escape(end)}</small></h2>"
            "<table><thead><tr><th>Market gate</th><th>Thoát lệnh</th>"
            "<th>Return</th><th>Sharpe</th><th>Max DD</th><th>Orders</th><th>Turnover</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
    document = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>Flow V2 - Gate comparison</title>
<style>
body {{ font: 15px Segoe UI, Arial, sans-serif; color: #122435; margin: 32px; line-height: 1.5; }}
h1 {{ margin-bottom: 5px; }} h2 {{ margin-top: 28px; font-size: 20px; }}
h2 small {{ color: #65768a; font-weight: normal; font-size: 13px; }}
.note {{ max-width: 1200px; background: #edf5f5; border-left: 4px solid #176e65; padding: 12px 15px; }}
table {{ border-collapse: collapse; min-width: 920px; margin-top: 12px; }}
th, td {{ border: 1px solid #dce4e6; padding: 9px 12px; text-align: right; }}
th:nth-child(1), td:nth-child(1), th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #edf4f3; }}
</style></head><body>
<h1>Flow V2: Gate regime và rủi ro</h1>
<p>VN100 | vốn {capital:,.0f} VND | top2 | rebalance 10 phiên | high_rs + flow_heavy | signal close T, khớp open T+1 | giá thi hành VND đã sửa.</p>
<p class="note"><strong>Không gate</strong> ở đây chỉ bỏ điều kiện market regime; bộ lọc chất lượng cổ phiếu vẫn giữ nguyên. <strong>Early-exit hai tầng</strong> là lớp quản trị sau khi đã vào vị thế, khác với gate quyết định có được mở vị thế hay không.</p>
{''.join(sections)}
</body></html>"""
    (out_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Flow V2 market gate levels and exit overlay.")
    parser.add_argument("--end", default="2026-05-26")
    parser.add_argument("--recent-start", default="2025-01-01")
    parser.add_argument("--full-start", default="2020-01-01")
    parser.add_argument("--capital", type=float, default=100_000_000.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "flow_v2_gate_comparison_vn100_2020_to_2026-05-26",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    full_features = _prepare_features("vn100", args.full_start, args.end)
    rows = _run_period(full_features, "recent", args.recent_start, args.end, args.capital, args.out_dir)
    rows.extend(_run_period(full_features, "full_history", args.full_start, args.end, args.capital, args.out_dir))
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    _export_html(args.out_dir, summary, args.capital)
    print(f"[flow-v2-gates] saved {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
