"""Break down Flow V2 production-like equity by year and market regime."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest_flow_v2_rotation_production_like import ROOT, _prepare_features


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def _sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty or float(returns.std()) == 0.0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(252.0))


def _summarize(group: pd.DataFrame) -> pd.Series:
    returns = group["daily_return"].fillna(0.0)
    equity = group["equity"]
    total_return = float((1.0 + returns).prod() - 1.0)
    return pd.Series(
        {
            "trading_days": int(len(group)),
            "total_return_pct": round(total_return * 100.0, 2),
            "sharpe": round(_sharpe(returns), 3),
            "max_drawdown_pct": round(_max_drawdown(equity) * 100.0, 2),
            "avg_cash_weight_pct": round(float(group["cash_weight"].mean()) * 100.0, 2),
            "avg_positions": round(float(group["positions"].mean()), 2),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Flow V2 equity by year and market regime")
    parser.add_argument("--equity", required=True)
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-05-19")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--label", default="flow_v2_best_full_cycle")
    args = parser.parse_args()

    equity = pd.read_csv(args.equity)
    equity["date"] = pd.to_datetime(equity["date"]).dt.normalize()
    equity = equity.sort_values("date").reset_index(drop=True)
    equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
    equity["year"] = equity["date"].dt.year

    features = _prepare_features(args.universe, args.start, args.end)
    regime_cols = ["date", "mkt_regime_state", "mkt_regime_score", "mkt_DS20", "mkt_CHDM20"]
    regime = (
        features[regime_cols]
        .drop_duplicates("date")
        .assign(date=lambda frame: pd.to_datetime(frame["date"]).dt.normalize())
        .sort_values("date")
    )

    merged = equity.merge(regime, on="date", how="left")
    merged["mkt_regime_state"] = merged["mkt_regime_state"].fillna("UNKNOWN")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annual = merged.groupby("year", sort=True).apply(_summarize, include_groups=False).reset_index()
    regime_summary = (
        merged.groupby("mkt_regime_state", sort=True).apply(_summarize, include_groups=False).reset_index()
    )
    year_regime = (
        merged.groupby(["year", "mkt_regime_state"], sort=True)
        .apply(_summarize, include_groups=False)
        .reset_index()
    )

    annual.to_csv(out_dir / f"{args.label}_annual_breakdown.csv", index=False)
    regime_summary.to_csv(out_dir / f"{args.label}_regime_breakdown.csv", index=False)
    year_regime.to_csv(out_dir / f"{args.label}_year_regime_breakdown.csv", index=False)
    merged.to_csv(out_dir / f"{args.label}_daily_regime_equity.csv", index=False)

    print("[flow-v2-breakdown] annual")
    print(annual.to_string(index=False))
    print("[flow-v2-breakdown] regime")
    print(regime_summary.to_string(index=False))
    print("[flow-v2-breakdown] year_regime")
    print(year_regime.to_string(index=False))


if __name__ == "__main__":
    main()
