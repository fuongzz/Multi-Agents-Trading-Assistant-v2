"""Research regime-aware strategy routing.

The goal is not to maximize one in-sample return. This runner:

1. Tags each signal/trade with a causal market regime at signal date.
2. Runs individual strategy backtests on a training period.
3. Learns allowed (strategy, regime) pairs from win rate/expectancy/trade-count.
4. Backtests a regime router on validation and full periods.

This is research-only; it does not modify production strategy config.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import evaluate_filters, load_hypotheses, rank_candidates
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG, _round_or_none, _setup_type_from_tags


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "backtest_results" / "regime_strategy_router"


DEFAULT_POOL = [
    "leader_pullback_market_regime_v3",
    "leader_pullback_market_healthy_v2",
    "breakout_55_smt_v1",
    "money_cycle_reset_smt_confirm",
    "smart_money_strong_market_healthy",
    "accumulation_breakout_smt_v1",
    "breakout_after_accumulation_v3",
    "compression_breakout_smt_v1",
    "mean_reversion_uptrend_ma20_v1",
    "mean_reversion_uptrend_ma50_v1",
    "reclaim_ma20_quality_v1",
    "cmf_expansion_smt_v1",
    "sector_leader_pullback_v1",
    "benchmark_leader_breakout_v1",
    "dry_retest_breakout_smt_v1",
    "nr7_value_breakout_smt_v1",
    "rotation_weekly_smt_money_cycle_v1",
    "composite_edge_score_v2",
]


def _regime_table(features: pd.DataFrame) -> pd.DataFrame:
    daily = (
        features[
            [
                "date",
                "mkt_regime_state",
                "mkt_regime_score",
                "vni_above_ma50",
                "vni_ret_20d",
                "vni_ret_60d",
            ]
        ]
        .drop_duplicates("date")
        .copy()
    )
    daily["market_regime"] = daily.apply(_classify_regime_row, axis=1)
    return daily.sort_values("date").reset_index(drop=True)


def _classify_regime_row(row: pd.Series) -> str:
    money_state = str(row.get("mkt_regime_state") or "UNKNOWN").upper()
    above_ma50 = bool(row.get("vni_above_ma50")) if pd.notna(row.get("vni_above_ma50")) else False
    ret20 = _safe_float(row.get("vni_ret_20d"), 0.0)
    ret60 = _safe_float(row.get("vni_ret_60d"), 0.0)
    if money_state == "RISK_OFF":
        return "RISK_OFF"
    if money_state == "RISK_ON" and above_ma50 and ret60 >= 0:
        return "RISK_ON_UPTREND"
    if money_state == "RISK_ON":
        return "RISK_ON_RECOVERY"
    if money_state == "NEUTRAL" and above_ma50 and ret20 >= -0.03:
        return "NEUTRAL_UPTREND"
    return "NEUTRAL_SIDEWAY"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _features_with_regime(features: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    out = features.drop(columns=[col for col in ["market_regime"] if col in features.columns]).merge(
        regimes[["date", "market_regime"]],
        on="date",
        how="left",
    )
    out["market_regime"] = out["market_regime"].fillna("UNKNOWN")
    return out


def _build_router_cache(
    *,
    features: pd.DataFrame,
    hypotheses: dict[str, Any],
    strategy_names: list[str],
    clean_symbols: list[str],
    combo_label: str,
    allowed_pairs: set[tuple[str, str]] | None,
    regime_strategy_order: dict[str, list[str]] | None = None,
) -> dict[pd.Timestamp, dict[str, dict[str, Any]]]:
    signal_cache: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    hyps = [hypotheses[name] for name in strategy_names]
    for date, daily in features.groupby("date", sort=True):
        latest = daily.reset_index(drop=True)
        regime = str(latest["market_regime"].iloc[0]) if "market_regime" in latest.columns and not latest.empty else "UNKNOWN"
        day_out: dict[str, dict[str, Any]] = {}
        for _, row in latest.iterrows():
            symbol = str(row["symbol"])
            day_out[symbol] = {
                "strategy_name": combo_label,
                "strategy_family": combo_label,
                "passed": False,
                "available": True,
                "feature_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "market_regime": row.get("market_regime"),
                "edge_rank": None,
                "edge_rank_score": 0.0,
                "edge_score": _round_or_none(row.get("edge_score")),
                "smart_money_score": _round_or_none(row.get("smart_money_score")),
                "smart_money_score_delta": _round_or_none(row.get("smart_money_score_delta")),
                "rs_percentile_20": _round_or_none(row.get("rs_percentile_20")),
                "value_ratio_20": _round_or_none(row.get("value_ratio_20")),
                "CHDM50": _round_or_none(row.get("CHDM50")),
                "DS20": _round_or_none(row.get("DS20")),
                "mkt_regime_state": row.get("mkt_regime_state"),
                "mkt_regime_score": _round_or_none(row.get("mkt_regime_score")),
                "filters_failed": [],
                "risk": {},
            }
        active_hyps = hyps
        if regime_strategy_order is not None:
            active_hyps = [hypotheses[name] for name in regime_strategy_order.get(regime, []) if name in hypotheses]
        for hyp in active_hyps:
            if allowed_pairs is not None and (hyp.name, regime) not in allowed_pairs:
                continue
            mask = evaluate_filters(latest, hyp)
            ranked = rank_candidates(latest[mask], hyp)
            for rank_idx, row in enumerate(ranked.to_dict("records"), start=1):
                symbol = str(row["symbol"])
                current = day_out.get(symbol)
                if not current:
                    continue
                rank_score = float(row.get("_edge_rank") or 0.0)
                if rank_score <= float(current.get("edge_rank_score") or 0.0):
                    continue
                current.update(
                    {
                        "strategy_name": hyp.name,
                        "strategy_family": combo_label,
                        "description": hyp.description,
                        "passed": True,
                        "edge_rank": rank_idx,
                        "edge_rank_score": round(rank_score, 4),
                        "setup_type": _setup_type_from_tags(hyp),
                        "risk": hyp.risk,
                    }
                )
        for symbol in clean_symbols:
            day_out.setdefault(
                symbol,
                {"strategy_name": combo_label, "passed": False, "available": False, "error": "No latest feature row"},
            )
        signal_cache[pd.Timestamp(date).normalize()] = day_out
    return signal_cache


def _run_router(
    *,
    features: pd.DataFrame,
    hypotheses: dict[str, Any],
    strategy_names: list[str],
    clean_symbols: list[str],
    universe_data: dict[str, pd.DataFrame],
    vnindex: pd.DataFrame,
    start: str,
    end: str,
    max_positions: int,
    max_candidates_per_day: int,
    label: str,
    allowed_pairs: set[tuple[str, str]] | None,
    regime_strategy_order: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    cache = _build_router_cache(
        features=features,
        hypotheses=hypotheses,
        strategy_names=strategy_names,
        clean_symbols=clean_symbols,
        combo_label=label,
        allowed_pairs=allowed_pairs,
        regime_strategy_order=regime_strategy_order,
    )
    combo_harness.install_edge_cache(cache)
    cfg = LivePipelineBacktestConfig(
        start_date=start,
        end_date=end,
        max_positions=max_positions,
        max_candidates_per_day=max_candidates_per_day,
        edge_strategy_name=label,
    )
    result = combo_harness.run_one(universe_data, vnindex, cfg)
    metrics = result["metrics"]
    return {
        "label": label,
        "start": start,
        "end": end,
        "max_positions": max_positions,
        "max_candidates_per_day": max_candidates_per_day,
        "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
        "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
        "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
        "number_of_trades": int(metrics.get("number_of_trades", 0)),
        "fills": int(metrics.get("fills", 0)),
        "trades": result["trades"],
        "equity_frame": result["equity_frame"],
    }


def _attach_trade_regime(trades: list[dict], regimes: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(trades)
    if df.empty:
        return df
    df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.normalize()
    out = df.merge(regimes[["date", "market_regime"]], left_on="signal_date", right_on="date", how="left")
    out.drop(columns=["date"], inplace=True)
    out["market_regime"] = out["market_regime"].fillna("UNKNOWN")
    out["is_win"] = out["pnl_pct"] > 0
    return out


def _trade_group_stats(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    grouped = trades.groupby(group_cols, dropna=False)
    rows = []
    for key, grp in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        row.update(
            {
                "trades": len(grp),
                "win_rate_pct": round(float((grp["pnl_pct"] > 0).mean()) * 100, 2),
                "avg_pnl_pct": round(float(grp["pnl_pct"].mean()) * 100, 3),
                "median_pnl_pct": round(float(grp["pnl_pct"].median()) * 100, 3),
                "total_pnl_pct_sum": round(float(grp["pnl_pct"].sum()) * 100, 3),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["win_rate_pct", "avg_pnl_pct", "trades"], ascending=[False, False, False])


def _select_allowed_pairs(
    stats: pd.DataFrame,
    *,
    min_trades: int,
    min_win_rate: float,
    min_avg_pnl: float,
) -> set[tuple[str, str]]:
    if stats.empty:
        return set()
    selected = stats[
        (stats["trades"] >= min_trades)
        & (stats["win_rate_pct"] >= min_win_rate)
        & (stats["avg_pnl_pct"] >= min_avg_pnl)
    ].copy()
    return {(str(row["edge_strategy_name"]), str(row["market_regime"])) for _, row in selected.iterrows()}


def _weeks_between(start: str, end: str) -> float:
    return max(1.0, (pd.Timestamp(end) - pd.Timestamp(start)).days / 7.0)


def _summary_row(result: dict[str, Any], *, kind: str, allowed_pairs_count: int) -> dict[str, Any]:
    weeks = _weeks_between(result["start"], result["end"])
    trades = int(result["number_of_trades"])
    return {
        "kind": kind,
        "label": result["label"],
        "start": result["start"],
        "end": result["end"],
        "allowed_pairs": allowed_pairs_count,
        "max_positions": result["max_positions"],
        "max_candidates_per_day": result["max_candidates_per_day"],
        "total_return_pct": result["total_return_pct"],
        "sharpe_ratio": result["sharpe_ratio"],
        "max_drawdown_pct": result["max_drawdown_pct"],
        "win_rate_pct": result["win_rate_pct"],
        "number_of_trades": trades,
        "trades_per_week": round(trades / weeks, 2),
        "fills": result["fills"],
    }


def _parse_manual_router(value: str) -> dict[str, list[str]]:
    """Parse REGIME=strat+strat;REGIME=strat into a router map."""
    out: dict[str, list[str]] = {}
    if not value.strip():
        return out
    for chunk in value.split(";"):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid manual router chunk: {chunk!r}")
        regime, strategies = chunk.split("=", 1)
        names = [item.strip() for item in strategies.replace(",", "+").split("+") if item.strip()]
        out[regime.strip()] = names
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Research regime-aware strategy router")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--train-start", default="2022-01-01")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--valid-start", default="2025-01-01")
    parser.add_argument("--valid-end", default="2026-05-16")
    parser.add_argument("--full-start", default="2022-01-01")
    parser.add_argument("--full-end", default="2026-05-16")
    parser.add_argument("--label", default="regime_router_2022_2026")
    parser.add_argument("--pool", default="")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=15)
    parser.add_argument("--min-train-trades", type=int, default=5)
    parser.add_argument("--min-train-win-rate", type=float, default=50.0)
    parser.add_argument("--min-train-avg-pnl", type=float, default=0.0)
    parser.add_argument(
        "--manual-router",
        default="",
        help="Optional REGIME=strategy+strategy;REGIME=strategy map. Bypasses learned allowed pairs for router runs.",
    )
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    full_start = min(args.train_start, args.valid_start, args.full_start)
    full_end = max(args.train_end, args.valid_end, args.full_end)
    symbols = combo_harness.resolve_symbols(args.universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, full_start, full_end)

    print("[regime] building features once...")
    features, _ = build_feature_table(symbols, full_start, full_end, root=ROOT)
    features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
    regimes = _regime_table(features)
    features = _features_with_regime(features, regimes)
    regimes.to_csv(out_dir / "market_regime_daily.csv", index=False)
    regimes[
        (regimes["date"] >= pd.Timestamp(args.full_start)) & (regimes["date"] <= pd.Timestamp(args.full_end))
    ]["market_regime"].value_counts().rename_axis("market_regime").reset_index(name="days").to_csv(
        out_dir / "market_regime_distribution.csv",
        index=False,
    )

    hypotheses = {hyp.name: hyp for hyp in load_hypotheses(DEFAULT_CONFIG)}
    pool = [item.strip() for item in args.pool.split(",") if item.strip()] if args.pool.strip() else DEFAULT_POOL
    missing = [name for name in pool if name not in hypotheses]
    if missing:
        raise KeyError(f"Missing hypotheses: {missing}")
    manual_router = _parse_manual_router(args.manual_router)
    manual_missing = sorted({name for names in manual_router.values() for name in names if name not in hypotheses})
    if manual_missing:
        raise KeyError(f"Manual router missing hypotheses: {manual_missing}")

    print(f"[regime] train baseline pool strategies={len(pool)}")
    train_all = _run_router(
        features=features,
        hypotheses=hypotheses,
        strategy_names=pool,
        clean_symbols=clean_symbols,
        universe_data=universe_data,
        vnindex=vnindex,
        start=args.train_start,
        end=args.train_end,
        max_positions=args.max_positions,
        max_candidates_per_day=args.max_candidates_per_day,
        label="REGIME_POOL_BASELINE",
        allowed_pairs=None,
    )
    train_trades = _attach_trade_regime(train_all["trades"], regimes)
    train_trades.to_csv(out_dir / "train_pool_trades_with_regime.csv", index=False)
    strategy_regime = _trade_group_stats(train_trades, ["edge_strategy_name", "market_regime"])
    strategy_regime.to_csv(out_dir / "train_strategy_regime_stats.csv", index=False)
    regime_stats = _trade_group_stats(train_trades, ["market_regime"])
    regime_stats.to_csv(out_dir / "train_regime_stats.csv", index=False)
    strategy_stats = _trade_group_stats(train_trades, ["edge_strategy_name"])
    strategy_stats.to_csv(out_dir / "train_strategy_stats.csv", index=False)

    allowed_pairs = _select_allowed_pairs(
        strategy_regime,
        min_trades=args.min_train_trades,
        min_win_rate=args.min_train_win_rate,
        min_avg_pnl=args.min_train_avg_pnl,
    )
    if manual_router:
        allowed_pairs = {
            (strategy, regime)
            for regime, strategies in manual_router.items()
            for strategy in strategies
        }
    pd.DataFrame(
        [{"edge_strategy_name": strategy, "market_regime": regime} for strategy, regime in sorted(allowed_pairs)]
    ).to_csv(out_dir / "selected_strategy_regime_pairs.csv", index=False)
    print(f"[regime] selected allowed pairs={len(allowed_pairs)}")

    rows = [_summary_row(train_all, kind="train_pool_baseline", allowed_pairs_count=0)]
    for kind, start, end in [
        ("valid_pool_baseline", args.valid_start, args.valid_end),
        ("valid_regime_router", args.valid_start, args.valid_end),
        ("full_pool_baseline", args.full_start, args.full_end),
        ("full_regime_router", args.full_start, args.full_end),
    ]:
        allowed = allowed_pairs if kind.endswith("router") else None
        label = "REGIME_ROUTER" if allowed is not None else "REGIME_POOL_BASELINE"
        print(f"[regime] run {kind} {start}..{end}")
        result = _run_router(
            features=features,
            hypotheses=hypotheses,
            strategy_names=pool,
            clean_symbols=clean_symbols,
            universe_data=universe_data,
            vnindex=vnindex,
            start=start,
            end=end,
            max_positions=args.max_positions,
            max_candidates_per_day=args.max_candidates_per_day,
            label=label,
            allowed_pairs=allowed,
            regime_strategy_order=manual_router if allowed is not None and manual_router else None,
        )
        trades = _attach_trade_regime(result["trades"], regimes)
        trades.to_csv(out_dir / f"{kind}_trades_with_regime.csv", index=False)
        if not result["equity_frame"].empty:
            result["equity_frame"].to_csv(out_dir / f"{kind}_equity.csv")
        rows.append(_summary_row(result, kind=kind, allowed_pairs_count=len(allowed_pairs) if allowed else 0))
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"[regime] saved {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
