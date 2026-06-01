"""Evaluate Flow V2 position-risk overlays on corrected VN100 execution."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.backtest_flow_v2_rotation_production_like import RotationConfig, _max_drawdown, _prepare_features, run_backtest


ROOT = Path(__file__).resolve().parents[1]


def _period_metrics(equity: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    period = equity.loc[(equity["date"] >= start) & (equity["date"] <= end), "equity"].astype(float)
    if period.empty:
        return {"return_pct": 0.0, "drawdown_pct": 0.0}
    return {
        "return_pct": round(float(period.iloc[-1] / period.iloc[0] - 1.0) * 100.0, 2),
        "drawdown_pct": round(_max_drawdown(period) * 100.0, 2),
    }


def _configs(start: str, end: str, capital: float) -> list[tuple[str, str, RotationConfig]]:
    base: dict[str, Any] = {
        "universe": "vn100",
        "start": start,
        "end": end,
        "positions": 2,
        "rebalance_days": 10,
        "initial_capital": capital,
        "market_gate": "risk_on_or_strong_neutral",
        "pool_filter": "high_rs",
        "score_mode": "flow_heavy",
        "price_unit_multiplier": 1000.0,
    }
    return [
        ("fresh_baseline_top2", "Khong overlay quan tri vi the", RotationConfig(**base)),
        (
            "early_exit_full_top2",
            "Ban het open T+1 khi flow/momentum gay",
            RotationConfig(early_exit_mode="flow_momentum_break", **base),
        ),
        (
            "early_exit_tiered_top2",
            "Flow yeu thi giam nua vi the; gay manh thi ban het",
            RotationConfig(early_exit_mode="flow_momentum_tiered", **base),
        ),
        (
            "early_exit_inverse_atr_top2",
            "Thoat som + chia trong so ty le nghich ATR",
            RotationConfig(early_exit_mode="flow_momentum_break", allocation_mode="inverse_atr", **base),
        ),
        (
            "early_exit_sector_cap_top2",
            "Thoat som + toi da mot ma tren moi nganh",
            RotationConfig(early_exit_mode="flow_momentum_break", diversification_mode="distinct_industry", **base),
        ),
        (
            "early_exit_sideway_buffer_top2",
            "Thoat som + NEUTRAL chi giai ngan toi da 70% NAV",
            RotationConfig(early_exit_mode="flow_momentum_break", neutral_gross_exposure=0.70, **base),
        ),
        (
            "combined_risk_overlay_top2",
            "Thoat hai tang + inverse ATR + khac nganh + NEUTRAL 70%",
            RotationConfig(
                early_exit_mode="flow_momentum_tiered",
                allocation_mode="inverse_atr",
                diversification_mode="distinct_industry",
                neutral_gross_exposure=0.70,
                **base,
            ),
        ),
    ]


def _export_html(out_dir: Path, summary: pd.DataFrame, start: str, end: str) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            "<tr>"
            f"<td><strong>{escape(str(row['variant']))}</strong><br><span>{escape(str(row['description']))}</span></td>"
            f"<td>{float(row['total_return_pct']):+.2f}%</td>"
            f"<td>{float(row['sharpe_ratio']):.3f}</td>"
            f"<td>{float(row['max_drawdown_pct']):.2f}%</td>"
            f"<td>{float(row['return_2024_pct']):+.2f}%</td>"
            f"<td>{float(row['drawdown_2024_pct']):.2f}%</td>"
            f"<td>{int(row['full_exit_orders'])}</td>"
            f"<td>{int(row['partial_exit_orders'])}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>Flow V2 risk overlay comparison</title>
<style>
body {{ font: 15px Segoe UI, Arial, sans-serif; color: #16242b; margin: 30px; line-height: 1.5; }}
h1 {{ font-size: 25px; margin-bottom: 4px; }} h2 {{ font-size: 19px; margin-top: 28px; }}
.note {{ background: #f3f7f7; border-left: 4px solid #146b62; padding: 12px 15px; max-width: 1150px; }}
table {{ border-collapse: collapse; width: 100%; max-width: 1250px; margin-top: 14px; font-size: 14px; }}
th, td {{ border: 1px solid #dce4e6; padding: 10px; text-align: right; vertical-align: top; }}
th:first-child, td:first-child {{ text-align: left; }} th {{ background: #edf4f3; }} td span {{ color: #596c74; font-size: 12px; }}
</style></head><body>
<h1>Flow V2: so sanh overlay quan tri rui ro</h1>
<p>VN100 | {start} den {end} | von 1 ty dong | top2 | rebalance 10 phien | open T+1 | VND execution corrected</p>
<p class="note">Tat ca bien the fresh-signal dung cung mot pipeline du lieu da dong bo. Tin hieu thoat, phan bo ATR, gioi han nganh va giam exposure NEUTRAL deu chi dung feature biet duoc sau dong cua T; lenh chi khop o open T+1.</p>
<h2>Ket qua</h2>
<table><thead><tr><th>Bien the</th><th>Return toan ky</th><th>Sharpe</th><th>Max DD</th><th>2024</th><th>DD 2024</th><th>Full exit</th><th>Partial exit</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    (out_dir / "index.html").write_text(document, encoding="utf-8")


def run_experiment(out_dir: Path, start: str, end: str, capital: float) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    features = _prepare_features("vn100", start, end)
    rows: list[dict[str, Any]] = []
    for variant, description, cfg in _configs(start, end, capital):
        print(f"[flow-v2-risk] running {variant}")
        result = run_backtest(cfg, features=features)
        p2024 = _period_metrics(result["equity"], "2024-01-01", "2024-12-31")
        trades = result["trades"]
        row = dict(result["summary"])
        row.update(
            {
                "variant": variant,
                "description": description,
                "return_2024_pct": p2024["return_pct"],
                "drawdown_2024_pct": p2024["drawdown_pct"],
                "full_exit_orders": int((trades["reason"] == "EARLY_EXIT_FLOW_MOMENTUM_BREAK").sum())
                if not trades.empty
                else 0,
                "partial_exit_orders": int((trades["reason"] == "PARTIAL_EXIT_FLOW_WEAKNESS").sum())
                if not trades.empty
                else 0,
            }
        )
        rows.append(row)
        result["equity"].to_csv(out_dir / f"{variant}_equity.csv", index=False)
        result["targets"].to_csv(out_dir / f"{variant}_targets.csv", index=False)
        result["trades"].to_csv(out_dir / f"{variant}_orders.csv", index=False)
        print(
            f"  ret={row['total_return_pct']:+.2f}% sharpe={row['sharpe_ratio']:.3f} "
            f"dd={row['max_drawdown_pct']:.2f}% 2024={row['return_2024_pct']:+.2f}%"
        )
    summary = pd.DataFrame(rows).sort_values(["sharpe_ratio", "total_return_pct"], ascending=[False, False])
    summary.to_csv(out_dir / "summary.csv", index=False)
    _export_html(out_dir, summary, start, end)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Flow V2 risk overlays")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-22")
    parser.add_argument("--capital", type=float, default=1_000_000_000.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "flow_v2_risk_management_experiment_vn100_2020_to_2026-05-22",
    )
    args = parser.parse_args()
    summary = run_experiment(args.out_dir, args.start, args.end, args.capital)
    print(summary[["variant", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "return_2024_pct"]].to_string(index=False))
    print(f"[flow-v2-risk] saved {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
