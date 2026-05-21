"""Flow Money V2 rotation research strategy.

This is intentionally separate from the entry/stop strategy pipeline. Flow V2
behaves better as a cross-sectional ranking model, so this script tests a
periodic sector/stock rotation portfolio.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from multiagents_trading_assistant.edge_lab.features import build_feature_table


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "flow_v2_rotation_strategy"


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def _sharpe(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty or float(daily_returns.std()) == 0.0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(252.0))


def _prepare_features(universe: str, start: str, end: str) -> pd.DataFrame:
    symbols = combo_harness.resolve_symbols(universe)
    features, _ = build_feature_table(symbols, start, end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    features["daily_close_ret"] = features.groupby("symbol", sort=False)["close"].pct_change()
    features["value_rank"] = features.groupby("date", sort=False)["value_ratio_20"].rank(pct=True)
    features["dist_rank"] = features.groupby("date", sort=False)["distribution_pressure_score"].rank(pct=True)
    features["flow_v2_rotation_score"] = (
        0.28 * features["sector_cycle_score"].fillna(50.0)
        + 0.24 * features["flow_sponsorship_score"].fillna(50.0)
        + 0.18 * features["flow_absorption_score"].fillna(50.0)
        + 0.15 * (features["rs_percentile_20"].fillna(0.5) * 100.0)
        + 0.15 * (features["value_rank"].fillna(0.5) * 100.0)
        - 0.15 * (features["dist_rank"].fillna(0.5) * 100.0)
    )
    return features


def _select_symbols(day: pd.DataFrame, positions: int) -> list[str]:
    pool = day[
        (day["mkt_regime_state"] != "RISK_OFF")
        & (day["mkt_regime_score"] >= 50)
        & (day["mkt_DS20"] <= 0.60)
        & (day["above_ma50"] == True)
        & (day["distribution_pressure_score"] <= 65)
        & (day["flow_sponsorship_score"] >= 45)
        & (day["value_ratio_20"] >= 0.80)
        & (day["rs_percentile_20"] >= 0.45)
    ].copy()
    if pool.empty:
        return []
    return list(pool.sort_values("flow_v2_rotation_score", ascending=False).head(positions)["symbol"])


def run_rotation(
    *,
    features: pd.DataFrame,
    start: str,
    positions: int,
    rebalance_days: int,
    cost_bps: float,
) -> dict[str, Any]:
    start_ts = pd.Timestamp(start).normalize()
    calendar = sorted(features.loc[features["date"] >= start_ts, "date"].unique())
    by_date = {date: group for date, group in features.groupby("date", sort=True)}
    cost_rate = cost_bps / 10_000.0

    equity = 1.0
    holdings: list[str] = []
    previous_set: set[str] = set()
    rows: list[dict[str, Any]] = []
    holding_rows: list[dict[str, Any]] = []
    rebalance_count = 0
    turnover_sum = 0.0
    selected_slots = 0

    for idx, date in enumerate(calendar):
        day = by_date[date]
        daily_ret = 0.0
        if holdings:
            returns = day.set_index("symbol").reindex(holdings)["daily_close_ret"].dropna()
            if not returns.empty:
                daily_ret = float(returns.mean())
                equity *= 1.0 + daily_ret

        rebalanced = idx % rebalance_days == 0
        if rebalanced:
            new_holdings = _select_symbols(day, positions)
            new_set = set(new_holdings)
            turnover = len(new_set.symmetric_difference(previous_set)) / max(1, positions)
            if idx > 0 and turnover > 0:
                equity *= 1.0 - cost_rate * turnover
            holdings = new_holdings
            previous_set = new_set
            rebalance_count += 1
            turnover_sum += turnover
            selected_slots += len(new_holdings)
            for rank, symbol in enumerate(new_holdings, start=1):
                item = day.loc[day["symbol"] == symbol].iloc[0]
                holding_rows.append(
                    {
                        "date": pd.Timestamp(date).date().isoformat(),
                        "rank": rank,
                        "symbol": symbol,
                        "flow_v2_rotation_score": round(float(item["flow_v2_rotation_score"]), 4),
                        "sector_cycle_score": round(float(item["sector_cycle_score"]), 4),
                        "flow_sponsorship_score": round(float(item["flow_sponsorship_score"]), 4),
                        "flow_absorption_score": round(float(item["flow_absorption_score"]), 4),
                        "distribution_pressure_score": round(float(item["distribution_pressure_score"]), 4),
                        "rs_percentile_20": round(float(item["rs_percentile_20"]), 4),
                        "value_ratio_20": round(float(item["value_ratio_20"]), 4),
                    }
                )

        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "equity": equity,
                "daily_return": daily_ret,
                "positions": len(holdings),
                "rebalanced": rebalanced,
                "holdings": ",".join(holdings),
            }
        )

    equity_frame = pd.DataFrame(rows)
    equity_series = equity_frame.set_index(pd.to_datetime(equity_frame["date"]))["equity"]
    daily_returns = equity_series.pct_change()
    return {
        "summary": {
            "positions": positions,
            "rebalance_days": rebalance_days,
            "cost_bps": cost_bps,
            "total_return_pct": round((float(equity_series.iloc[-1]) - 1.0) * 100.0, 2) if not equity_series.empty else 0.0,
            "sharpe_ratio": round(_sharpe(daily_returns), 3),
            "max_drawdown_pct": round(_max_drawdown(equity_series) * 100.0, 2),
            "rebalance_count": rebalance_count,
            "avg_turnover": round(turnover_sum / max(1, rebalance_count), 3),
            "selected_slots": selected_slots,
        },
        "equity": equity_frame,
        "holdings": pd.DataFrame(holding_rows),
    }


def run_grid(
    *,
    universe: str,
    start: str,
    end: str,
    label: str,
    positions_values: list[int],
    rebalance_values: list[int],
    cost_bps: float,
) -> Path:
    out_dir = OUT_ROOT / f"{label}_{universe}_{start}_{end}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[flow-v2-rotation] universe={universe} period={start}->{end}")
    print("[flow-v2-rotation] building feature table once...")
    features = _prepare_features(universe, start, end)

    summaries: list[dict[str, Any]] = []
    for positions in positions_values:
        for rebalance_days in rebalance_values:
            print(f"[flow-v2-rotation] run top{positions}/rebalance{rebalance_days}")
            result = run_rotation(
                features=features,
                start=start,
                positions=positions,
                rebalance_days=rebalance_days,
                cost_bps=cost_bps,
            )
            summary = result["summary"]
            summaries.append(summary)
            stem = f"top{positions}_rebalance{rebalance_days}"
            result["equity"].to_csv(out_dir / f"{stem}_equity.csv", index=False)
            result["holdings"].to_csv(out_dir / f"{stem}_holdings.csv", index=False)
            pd.DataFrame(summaries).sort_values("total_return_pct", ascending=False).to_csv(out_dir / "summary.csv", index=False)
            print(
                f"  -> ret={summary['total_return_pct']:+.2f}% "
                f"sharpe={summary['sharpe_ratio']:.3f} "
                f"mdd={summary['max_drawdown_pct']:.2f}% "
                f"slots={summary['selected_slots']}"
            )

    pd.DataFrame(summaries).sort_values(["total_return_pct", "sharpe_ratio"], ascending=[False, False]).to_csv(
        out_dir / "summary.csv",
        index=False,
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Research Flow Money V2 rotation strategy")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-19")
    parser.add_argument("--label", default="flow_v2_rotation_2025now")
    parser.add_argument("--positions", default="3,4,5,6,8,10")
    parser.add_argument("--rebalance-days", default="3,5,10")
    parser.add_argument("--cost-bps", type=float, default=15.0)
    args = parser.parse_args()

    out_dir = run_grid(
        universe=args.universe,
        start=args.start,
        end=args.end,
        label=args.label,
        positions_values=[int(item.strip()) for item in args.positions.split(",") if item.strip()],
        rebalance_values=[int(item.strip()) for item in args.rebalance_days.split(",") if item.strip()],
        cost_bps=float(args.cost_bps),
    )
    print(f"[flow-v2-rotation] saved {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
