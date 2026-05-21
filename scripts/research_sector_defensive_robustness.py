"""Robustness grid for sector/cycle defensive breakout variants."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from multiagents_trading_assistant.edge_lab.backtest import PortfolioConfig, run_portfolio
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, load_hypotheses
from multiagents_trading_assistant.edge_lab.metrics import equity_metrics, trade_metrics
from multiagents_trading_assistant.edge_lab.universe import get_universe
from multiagents_trading_assistant.research.sector_rotation.enrich_features import enrich_features_with_rotation


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "multiagents_trading_assistant" / "edge_lab" / "configs"
CORE_CFG = CONFIGS / "vn30_money_smt_hypotheses.json"
ROTATION_CFG = CONFIGS / "sector_rotation_hypotheses.json"
OUT_ROOT = ROOT / "backtest_results" / "sector_rotation_defensive_robustness"


STRATEGIES = [
    ("core_breakout", CORE_CFG, "breakout_after_accumulation_v3"),
    ("market_cycle_defensive", ROTATION_CFG, "breakout_market_cycle_defensive_v1"),
    ("sector_anti_risk", ROTATION_CFG, "breakout_sector_anti_risk_v1"),
    ("sector_soft_rank", ROTATION_CFG, "breakout_sector_soft_rank_v1"),
]

PERIODS = [
    ("train_2022_2024", "2022-01-01", "2024-12-31"),
    ("valid_2025_now", "2025-01-01", "2026-05-13"),
    ("full_2022_now", "2022-01-01", "2026-05-13"),
]

GRID = [
    (5, 3),
    (5, 5),
    (8, 5),
    (8, 8),
    (10, 5),
]


def _hypothesis_map() -> dict[str, Hypothesis]:
    out: dict[str, Hypothesis] = {}
    for config in [CORE_CFG, ROTATION_CFG]:
        for hyp in load_hypotheses(config):
            out[hyp.name] = hyp
    return out


def _run_one(
    *,
    features: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    hypothesis: Hypothesis,
    period_label: str,
    start: str,
    end: str,
    max_positions: int,
    top_n: int,
) -> dict[str, Any]:
    cfg = PortfolioConfig(
        initial_capital=100_000.0,
        max_positions=max_positions,
        top_n=top_n,
    )
    trades, equity = run_portfolio(features, price_map, [hypothesis], pd.Timestamp(start), pd.Timestamp(end), cfg)
    em = equity_metrics(equity, cfg.initial_capital)
    tm = trade_metrics(trades)
    return {
        "period": period_label,
        "start": start,
        "end": end,
        "strategy": hypothesis.name,
        "max_positions": max_positions,
        "top_n": top_n,
        "total_return_pct": round(float(em.get("total_return", 0.0)) * 100, 2),
        "sharpe": round(float(em.get("sharpe", 0.0)), 3),
        "max_drawdown_pct": round(float(em.get("max_drawdown", 0.0)) * 100, 2),
        "final_equity": round(float(em.get("final_equity", 0.0)), 2),
        "trades": int(tm.get("trades", 0)),
        "symbols": int(tm.get("symbols", 0)),
        "win_rate_pct": round(float(tm.get("win_rate", 0.0)) * 100, 2),
        "profit_factor": round(float(tm.get("profit_factor", 0.0)), 3),
        "avg_trade_pct": round(float(tm.get("avg_trade_pct", 0.0)), 3),
        "median_trade_pct": round(float(tm.get("median_trade_pct", 0.0)), 3),
    }


def _score(row: dict[str, Any]) -> float:
    ret = float(row["total_return_pct"])
    sharpe = float(row["sharpe"])
    mdd = abs(float(row["max_drawdown_pct"]))
    pf = float(row["profit_factor"])
    trades = int(row["trades"])
    return ret + 8.0 * sharpe + 5.0 * min(pf, 3.0) - max(0.0, mdd - 18.0) - max(0, 40 - trades) * 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Sector defensive breakout robustness grid")
    parser.add_argument("--universe", default="WIDE")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-05-13")
    parser.add_argument("--label", default="v1")
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = get_universe(args.universe)
    print(f"[robust] universe={args.universe} symbols={len(symbols)}")
    print("[robust] building features once...")
    features, price_map = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = enrich_features_with_rotation(features, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)

    hypotheses = _hypothesis_map()
    rows: list[dict[str, Any]] = []
    for period_label, start, end in PERIODS:
        for max_positions, top_n in GRID:
            for alias, _, hyp_name in STRATEGIES:
                print(f"[robust] {period_label} {alias} p{max_positions} top{top_n}")
                row = _run_one(
                    features=features,
                    price_map=price_map,
                    hypothesis=hypotheses[hyp_name],
                    period_label=period_label,
                    start=start,
                    end=end,
                    max_positions=max_positions,
                    top_n=top_n,
                )
                row["alias"] = alias
                row["score"] = round(_score(row), 3)
                rows.append(row)
                pd.DataFrame(rows).to_csv(out_dir / "grid_summary.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "grid_summary.csv", index=False)
    piv = (
        summary.sort_values(["period", "score"], ascending=[True, False])
        .groupby("period", as_index=False)
        .head(12)
    )
    piv.to_csv(out_dir / "top_by_period.csv", index=False)

    stability_rows = []
    for keys, grp in summary.groupby(["alias", "strategy", "max_positions", "top_n"], dropna=False):
        by_period = {row["period"]: row for _, row in grp.iterrows()}
        if not all(label in by_period for label, _, _ in PERIODS):
            continue
        valid = by_period["valid_2025_now"]
        train = by_period["train_2022_2024"]
        full = by_period["full_2022_now"]
        stability_rows.append(
            {
                "alias": keys[0],
                "strategy": keys[1],
                "max_positions": keys[2],
                "top_n": keys[3],
                "train_return_pct": train["total_return_pct"],
                "train_sharpe": train["sharpe"],
                "train_mdd_pct": train["max_drawdown_pct"],
                "valid_return_pct": valid["total_return_pct"],
                "valid_sharpe": valid["sharpe"],
                "valid_mdd_pct": valid["max_drawdown_pct"],
                "full_return_pct": full["total_return_pct"],
                "full_sharpe": full["sharpe"],
                "full_mdd_pct": full["max_drawdown_pct"],
                "full_win_rate_pct": full["win_rate_pct"],
                "full_trades": full["trades"],
                "min_return_pct": min(train["total_return_pct"], valid["total_return_pct"], full["total_return_pct"]),
                "avg_sharpe": round((train["sharpe"] + valid["sharpe"] + full["sharpe"]) / 3.0, 3),
                "max_abs_mdd_pct": max(abs(train["max_drawdown_pct"]), abs(valid["max_drawdown_pct"]), abs(full["max_drawdown_pct"])),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability["stability_score"] = (
        stability["min_return_pct"]
        + 10.0 * stability["avg_sharpe"]
        - (stability["max_abs_mdd_pct"] - 18.0).clip(lower=0.0)
    ).round(3)
    stability = stability.sort_values("stability_score", ascending=False)
    stability.to_csv(out_dir / "stability_summary.csv", index=False)

    print(f"[robust] saved {out_dir}")
    print(stability.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
