"""Probe: vì sao mỗi ngày có/không có signal core3 trong window cho trước.

Cho mỗi ngày: in regime VN-Index + số mã pass mỗi strategy (core3) + reason
nếu market gate block.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from multiagents_trading_assistant.backtest.live_pipeline import (
    LivePipelineBacktestConfig,
    _market_context_at,
    _prepare_data,
    _prepare_single_frame,
)
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import evaluate_filters, load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import (
    DEFAULT_CONFIG,
    DEFAULT_LIVE_EDGE_FAMILY,
    _parse_strategy_names,
)
from multiagents_trading_assistant.fetcher import get_vn100_symbols


ROOT = Path(__file__).resolve().parents[1]


def load_local(symbols, start, end):
    df = pd.read_parquet(ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    warmup = pd.Timestamp(start) - pd.Timedelta(days=420)
    sub = df[
        df["symbol"].isin(symbols)
        & (df["date"] >= warmup)
        & (df["date"] <= pd.Timestamp(end))
    ].copy()
    data = {
        s: g[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        for s, g in sub.groupby("symbol", sort=False)
    }
    idx = pd.read_parquet(ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet")
    idx = idx[idx["symbol"].astype(str).str.upper() == "VNINDEX"][["date", "open", "high", "low", "close", "volume"]]
    idx["date"] = pd.to_datetime(idx["date"]).dt.normalize()
    idx = idx[(idx["date"] >= warmup) & (idx["date"] <= pd.Timestamp(end))].sort_values("date").reset_index(drop=True)
    return data, idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-05-04")
    parser.add_argument("--end", default="2026-05-13")
    args = parser.parse_args()

    symbols = get_vn100_symbols()
    print(f"VN100 symbols: {len(symbols)}")
    data_raw, vni_raw = load_local(symbols, args.start, args.end)
    data = _prepare_data(data_raw)
    vni = _prepare_single_frame(vni_raw)

    # Build feature table once
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()

    # Load hypotheses for core3
    cfg_path = Path(DEFAULT_CONFIG)
    strategy_names = _parse_strategy_names(DEFAULT_LIVE_EDGE_FAMILY)
    hypotheses = []
    for name in strategy_names:
        try:
            h = next(h for h in load_hypotheses(cfg_path) if h.name == name)
            hypotheses.append(h)
        except StopIteration:
            print(f"warn: hypothesis {name} not found")

    print(f"Core3 strategies: {[h.name for h in hypotheses]}")
    print()

    cfg = LivePipelineBacktestConfig()
    calendar = sorted(set(vni["date"]))
    start_ts = pd.Timestamp(args.start).normalize()
    end_ts = pd.Timestamp(args.end).normalize()

    print(f"{'Date':<12} {'Trend':<10} {'VNI%':>7} {'should':>7} | {'breakout':>9} {'compress':>9} {'meanrev':>8} | {'TOTAL':>6}")
    print("-" * 90)
    for date in calendar:
        if date < start_ts or date > end_ts:
            continue
        mctx = _market_context_at(vni, date, cfg.lookback)
        # Per-strategy pass counts
        feat_today = features[features["date"] == date]
        counts = []
        for h in hypotheses:
            try:
                mask = evaluate_filters(feat_today, h)
                counts.append(int(mask.sum()))
            except Exception as e:
                counts.append(-1)
        total = sum(c for c in counts if c >= 0)
        vni_pct = mctx.vni_change_pct
        trend = mctx.reference_trend
        should = "Y" if mctx.should_trade else "N"
        print(f"{date.date()} {trend:<10} {vni_pct:>+6.2f}% {should:>7} | "
              f"{counts[0]:>9} {counts[1]:>9} {counts[2]:>8} | {total:>6}")


if __name__ == "__main__":
    main()
