"""Probe: vì sao 06/05 sinh signal, các ngày khác không.

In ra:
- Per-day: thống kê market_regime_state/score, sector_breadth, breakout_20 count
- Top mã pass ngày 06/05 với feature values
- Compare 05/05 vs 06/05 vs 07/05 trên 3 mã cùng (VCG, HSG, GEX)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import evaluate_filters, load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG
from multiagents_trading_assistant.fetcher import get_vn100_symbols


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-05-04")
    parser.add_argument("--end", default="2026-05-13")
    args = parser.parse_args()

    symbols = get_vn100_symbols()
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    start_ts = pd.Timestamp(args.start).normalize()
    end_ts = pd.Timestamp(args.end).normalize()
    window = features[(features["date"] >= start_ts) & (features["date"] <= end_ts)].copy()

    print("=== Market-wide aggregates per day (VN100) ===")
    cols = [
        "mkt_regime_state", "mkt_regime_score",
        "breakout_20", "above_ma20", "above_ma50",
        "ma20_slope_5", "ma50_slope_10",
        "value_ratio_20", "smart_money_score_no_sector",
        "smart_money_score_no_sector_delta",
        "accumulation_days_10", "distribution_days_10",
        "CHDM50", "DS20",
        "data_quality_ok",
    ]
    for col in cols:
        if col not in window.columns:
            print(f"  (missing column: {col})")

    summary_rows = []
    for date, grp in window.groupby("date"):
        row = {"date": date.date()}
        row["regime_state"] = grp["mkt_regime_state"].iloc[0] if "mkt_regime_state" in grp else None
        row["regime_score"] = round(float(grp["mkt_regime_score"].iloc[0]), 1) if "mkt_regime_score" in grp else None
        row["n_breakout_20"] = int(grp.get("breakout_20", pd.Series(dtype=bool)).fillna(False).sum())
        row["n_above_ma20"] = int(grp.get("above_ma20", pd.Series(dtype=bool)).fillna(False).sum())
        row["pct_value_gt_1_1"] = round((grp.get("value_ratio_20", pd.Series(dtype=float)).fillna(0) >= 1.1).mean() * 100, 1)
        row["pct_smt_gt_58"] = round((grp.get("smart_money_score_no_sector", pd.Series(dtype=float)).fillna(0) >= 58).mean() * 100, 1)
        row["pct_smt_delta_pos"] = round((grp.get("smart_money_score_no_sector_delta", pd.Series(dtype=float)).fillna(0) >= 0).mean() * 100, 1)
        row["pct_rs_gt_58"] = round((grp.get("rs_percentile_20", pd.Series(dtype=float)).fillna(0) >= 0.58).mean() * 100, 1)
        row["pct_chdm50_gt_45"] = round((grp.get("CHDM50", pd.Series(dtype=float)).fillna(0) >= 45).mean() * 100, 1)
        row["pct_ds20_lt_055"] = round((grp.get("DS20", pd.Series(dtype=float)).fillna(0) <= 0.55).mean() * 100, 1)
        summary_rows.append(row)
    sumdf = pd.DataFrame(summary_rows)
    print(sumdf.to_string(index=False))
    print()

    # Per-strategy passes detail
    cfg_path = Path(DEFAULT_CONFIG)
    hyps = {h.name: h for h in load_hypotheses(cfg_path)}
    core3 = [
        "breakout_after_accumulation_v3",
        "compression_breakout_smt_v1",
        "mean_reversion_uptrend_ma50_v1",
    ]

    print("=== Mã pass mỗi strategy ngày 06/05 ===")
    target = window[window["date"] == pd.Timestamp("2026-05-06")]
    for name in core3:
        h = hyps[name]
        mask = evaluate_filters(target, h)
        passed = target[mask]
        print(f"\n[{name}] {len(passed)} mã pass:")
        if not passed.empty:
            show_cols = ["symbol", "close", "value_ratio_20", "smart_money_score_no_sector",
                         "smart_money_score_no_sector_delta", "rs_percentile_20", "CHDM50", "DS20"]
            show_cols = [c for c in show_cols if c in passed.columns]
            print(passed[show_cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
