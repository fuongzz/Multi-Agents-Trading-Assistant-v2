"""Compare simple vs complex indicator strategies on VN100.

This research runner answers a practical question:
do Vietnamese equities reward complex indicator stacks, or are simpler
momentum/trend rules enough?

The runner uses local cached daily data and the existing edge_lab no-lookahead
contract: signal at close T, fill next open T+1, with commission/slippage.
It is research-only and does not modify any paper/live sleeve.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiagents_trading_assistant.edge_lab.backtest import PortfolioConfig, run_portfolio
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, load_hypotheses
from multiagents_trading_assistant.edge_lab.strategy_sleeves import MVP_EDGE_STRATEGIES
from multiagents_trading_assistant.edge_lab.universe import get_universe
from multiagents_trading_assistant.quantagents_backtest.metrics import compute_metrics


DEFAULT_OUT_DIR = ROOT / "reports" / "vn_market_indicator_complexity_vn100"


def _simple_hypotheses() -> list[Hypothesis]:
    return [
        Hypothesis(
            name="simple_rs60_top_rank",
            description="Simple cross-sectional relative strength: favor VN100 stocks with high 60d excess return and liquid trading.",
            universe="VN100",
            tags=["simple", "relative_strength", "momentum"],
            filters=[
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "value_ma20", "op": ">=", "value": 20_000_000.0},
                {"column": "excess_ret_60d_pctile", "op": ">=", "value": 0.70},
            ],
            rank=[
                {"column": "excess_ret_60d_pctile", "ascending": False, "weight": 1.5},
                {"column": "excess_ret_20d_pctile", "ascending": False, "weight": 0.8},
                {"column": "value_ma20", "ascending": False, "weight": 0.2},
            ],
            risk={"stop_loss": 0.10, "take_profit": 0.30, "max_holding_bars": 40},
        ),
        Hypothesis(
            name="simple_ma20_ma50_uptrend",
            description="Simple trend filter: above MA20 and MA50, both not weakening, with decent liquidity.",
            universe="VN100",
            tags=["simple", "trend"],
            filters=[
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "ma20_slope_5", "op": ">=", "value": 0.0},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
            ],
            rank=[
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
                {"column": "ma20_slope_5", "ascending": False, "weight": 0.7},
                {"column": "value_ratio_20", "ascending": False, "weight": 0.3},
            ],
            risk={"stop_loss": 0.08, "take_profit": 0.25, "max_holding_bars": 35},
        ),
        Hypothesis(
            name="simple_rsi14_pullback_uptrend",
            description="Simple pullback: buy mild RSI14 weakness only when the stock remains above MA50.",
            universe="VN100",
            tags=["simple", "rsi", "mean_reversion"],
            filters=[
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "rsi14", "op": "between", "value": [35, 50]},
                {"column": "distance_ma20", "op": "between", "value": [-0.08, 0.02]},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.45},
            ],
            rank=[
                {"column": "rsi14", "ascending": True, "weight": 0.9},
                {"column": "rs_percentile_20", "ascending": False, "weight": 0.7},
                {"column": "smart_money_score", "ascending": False, "weight": 0.2},
            ],
            risk={"stop_loss": 0.07, "take_profit": 0.18, "max_holding_bars": 25},
        ),
    ]


def _load_named(path: Path, names: Iterable[str]) -> list[Hypothesis]:
    wanted = set(names)
    return [item for item in load_hypotheses(path) if item.name in wanted]


def _strategy_groups() -> dict[str, list[Hypothesis]]:
    config_dir = ROOT / "multiagents_trading_assistant" / "edge_lab" / "configs"
    global_classics = _load_named(
        config_dir / "global_market_hypotheses.json",
        [
            "turtle_donchian_20d_breakout_v1",
            "minervini_trend_template_v1",
            "weinstein_stage2_breakout_v1",
            "connors_rsi2_mean_reversion_v1",
            "fiftytwo_week_high_momentum_v1",
        ],
    )
    qmv_complex = load_hypotheses(config_dir / "qmv_ichimoku_turtle_hypotheses.json")
    all_local = load_hypotheses(config_dir / "vn30_money_smt_hypotheses.json")
    local_mvp = [item for item in all_local if item.name in set(MVP_EDGE_STRATEGIES)]
    return {
        "simple_price_rules": _simple_hypotheses(),
        "classic_imported_indicators": global_classics,
        "complex_ichimoku_qmv": qmv_complex,
        "complex_local_money_flow_mvp9": local_mvp,
    }


def _periods(start: str, end: str) -> list[tuple[str, str, str]]:
    periods = [
        ("full_2020_now", start, end),
        ("stress_2020_2022", "2020-01-01", "2022-12-31"),
        ("recovery_2023_2024", "2023-01-01", "2024-12-31"),
        ("recent_2025_now", "2025-01-01", end),
    ]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return [
        (label, max(pd.Timestamp(s), start_ts).date().isoformat(), min(pd.Timestamp(e), end_ts).date().isoformat())
        for label, s, e in periods
        if max(pd.Timestamp(s), start_ts) <= min(pd.Timestamp(e), end_ts)
    ]


def _summarize_equity(equity: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int]:
    if equity.empty:
        return {
            "return_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": 0,
            "win_rate_pct": 0.0,
        }
    metrics = compute_metrics(equity["equity"].astype(float), trades.to_dict("records") if not trades.empty else [])
    return {
        "return_pct": round(float(metrics["total_return"]) * 100.0, 2),
        "sharpe": round(float(metrics["sharpe_ratio"]), 3),
        "max_drawdown_pct": round(float(metrics["max_drawdown"]) * 100.0, 2),
        "trades": int(metrics["number_of_trades"]),
        "win_rate_pct": round(float(metrics["win_rate"]) * 100.0, 2),
    }


def _write_readme(out_dir: Path, summary: pd.DataFrame, start: str, end: str) -> None:
    full = summary[summary["period"].eq("full_2020_now")].sort_values("return_pct", ascending=False)
    recent = summary[summary["period"].eq("recent_2025_now")].sort_values("return_pct", ascending=False)
    lines = [
        "# VN100 Indicator Complexity Research",
        "",
        f"Period tested: `{start}` to `{end}`.",
        "",
        "All rows use VN100, local cached daily bars, signal close T, fill next open T+1, commission/slippage, board lots, and the same portfolio engine.",
        "",
        "## Full-period ranking",
        "",
        _markdown_table(full),
        "",
        "## Recent-period ranking",
        "",
        _markdown_table(recent),
        "",
        "## Interpretation",
        "",
        "- Simple rules are useful baselines, but they are not automatically robust allocators.",
        "- Imported complex indicators are fragile when used mechanically; they need local market/regime adaptation.",
        "- The strongest existing evidence in this repo favors local flow/relative-strength/regime logic over indicator complexity for its own sake.",
        "- Complexity is acceptable only when it improves out-of-sample return, drawdown, turnover quality, or risk timing.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    """Small markdown renderer that avoids requiring optional tabulate."""
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in frame.iterrows():
        values = [str(row[col]) for col in cols]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-29")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    universe = get_universe("VN100")
    features, price_map = build_feature_table(universe, args.start, args.end, root=ROOT)
    cfg = PortfolioConfig(
        initial_capital=100_000.0,
        max_positions=5,
        top_n=5,
        commission_rate=0.001,
        sell_tax_rate=0.001,
        slippage_rate=0.0005,
        max_participation_rate=0.10,
        lot_size=100,
        min_position_value=1_000.0,
        allocation_method="equal_weight",
    )

    rows: list[dict[str, object]] = []
    groups = _strategy_groups()
    for group_name, hypotheses in groups.items():
        if not hypotheses:
            continue
        for period_name, period_start, period_end in _periods(args.start, args.end):
            trades, equity = run_portfolio(features, price_map, hypotheses, period_start, period_end, config=cfg)
            tag = f"{group_name}_{period_name}"
            trades.to_csv(args.out_dir / f"{tag}_trades.csv", index=False, encoding="utf-8-sig")
            equity.to_csv(args.out_dir / f"{tag}_equity.csv", index=False, encoding="utf-8-sig")
            rows.append(
                {
                    "group": group_name,
                    "period": period_name,
                    "start": period_start,
                    "end": period_end,
                    "strategies": len(hypotheses),
                    **_summarize_equity(equity, trades),
                }
            )

    summary = pd.DataFrame(rows).sort_values(["period", "return_pct"], ascending=[True, False])
    summary.to_csv(args.out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    _write_readme(args.out_dir, summary, args.start, args.end)
    print(summary.to_string(index=False))
    print(args.out_dir / "summary.csv")


if __name__ == "__main__":
    main()
