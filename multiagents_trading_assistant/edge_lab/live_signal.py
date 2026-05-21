"""Live/shadow evaluation for backtested edge strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import (
    Hypothesis,
    evaluate_filters,
    load_hypotheses,
    rank_candidates,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "vn30_money_smt_hypotheses.json"
DEFAULT_STRATEGY = "breakout_after_accumulation_v3"
DEFAULT_LIVE_EDGE_FAMILY = (
    "breakout_after_accumulation_v3,"
    "compression_breakout_smt_v1,"
    "mean_reversion_uptrend_ma50_v1"
)
REGIME_ROUTER_MVP = "regime_router_mvp_v1"
REGIME_ROUTER_RESEARCH = "regime_router_research_v1"


def get_edge_strategy_signals(
    symbols: list[str],
    as_of_date: str | None = None,
    strategy_name: str = DEFAULT_STRATEGY,
    config_path: str | Path = DEFAULT_CONFIG,
    root: str | Path = ".",
) -> dict[str, dict[str, Any]]:
    """Evaluate a backtested edge strategy for the latest available feature date.

    Returns one record per requested symbol. `passed` is true only when the
    strategy filters pass on the latest available feature row for that symbol.
    """
    clean_symbols = sorted({item.upper().strip() for item in symbols if item})
    if not clean_symbols:
        return {}

    strategy_names = _parse_strategy_names(strategy_name)
    end = pd.Timestamp(as_of_date) if as_of_date else pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=30)

    try:
        features, _ = build_feature_table(clean_symbols, start, end, root=root)
    except Exception as exc:
        return {
            symbol: {
                "strategy_name": strategy_name,
                "passed": False,
                "available": False,
                "error": str(exc),
            }
            for symbol in clean_symbols
        }

    if features.empty:
        return {
            symbol: {
                "strategy_name": strategy_name,
                "passed": False,
                "available": False,
                "error": "No edge feature data available",
            }
            for symbol in clean_symbols
        }

    if as_of_date:
        features = features[features["date"] <= end]
    if features.empty:
        return {}

    latest = (
        features.sort_values(["symbol", "date"])
        .groupby("symbol", sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    latest["data_age_days"] = (end.normalize() - pd.to_datetime(latest["date"]).dt.normalize()).dt.days
    strategy_names = _resolve_strategy_names_for_latest(strategy_names, latest)
    hypotheses = [_load_strategy(name, config_path) for name in strategy_names]
    out: dict[str, dict[str, Any]] = {}
    for row_idx, row in latest.iterrows():
        symbol = str(row["symbol"])
        out[symbol] = {
            "strategy_name": strategy_name,
            "description": "",
            "passed": False,
            "available": True,
            "feature_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
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
            "data_age_days": int(row.get("data_age_days") or 0),
            "data_quality_ok": bool(row.get("data_quality_ok", True)),
            "filters_failed": _failed_filters(row, hypotheses[0]) if len(hypotheses) == 1 else [],
            "risk": {},
        }

    active_latest = latest[latest["data_age_days"] <= 14].copy()
    for hypothesis in hypotheses:
        mask = evaluate_filters(active_latest, hypothesis)
        ranked = rank_candidates(active_latest[mask], hypothesis)
        for rank_idx, row in enumerate(ranked.to_dict("records"), start=1):
            symbol = str(row["symbol"])
            rank_score = float(row.get("_edge_rank") or 0.0)
            current = out.get(symbol)
            if not current or rank_score <= float(current.get("edge_rank_score") or 0.0):
                continue
            current.update(
                {
                    "strategy_name": hypothesis.name,
                    "strategy_family": strategy_name if len(hypotheses) > 1 else hypothesis.name,
                    "description": hypothesis.description,
                    "passed": True,
                    "edge_rank": rank_idx,
                    "edge_rank_score": round(rank_score, 4),
                    "setup_type": _setup_type_from_tags(hypothesis),
                    "filters_failed": [],
                    "risk": hypothesis.risk,
                }
            )

    for symbol in clean_symbols:
        out.setdefault(
            symbol,
            {
                "strategy_name": strategy_name,
                "passed": False,
                "available": False,
                "error": "No latest feature row for symbol",
            },
        )
        if symbol in out and out[symbol].get("available") and int(out[symbol].get("data_age_days") or 0) > 14:
            out[symbol].update(
                {
                    "passed": False,
                    "available": False,
                    "error": f"Stale feature row: {out[symbol].get('feature_date')}",
                }
            )
    return out


def _parse_strategy_names(strategy_name: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(strategy_name, (list, tuple)):
        items = strategy_name
    else:
        items = str(strategy_name).split(",")
    names = [str(item).strip() for item in items if str(item).strip()]
    return names or [DEFAULT_STRATEGY]


def _resolve_strategy_names_for_latest(strategy_names: list[str], latest: pd.DataFrame) -> list[str]:
    resolved: list[str] = []
    for name in strategy_names:
        if name == REGIME_ROUTER_MVP:
            resolved.extend(_router_mvp_strategies(latest))
        elif name == REGIME_ROUTER_RESEARCH:
            resolved.extend(_router_research_strategies(latest))
        else:
            resolved.append(name)
    # keep order while de-duplicating
    return list(dict.fromkeys(resolved))


def _router_mvp_strategies(latest: pd.DataFrame) -> list[str]:
    if latest.empty:
        return [DEFAULT_STRATEGY]
    row = latest.iloc[0]
    regime = str(row.get("mkt_regime_state") or "")
    score = _safe_float(row.get("mkt_regime_score")) or 0.0
    chdm20 = _safe_float(row.get("mkt_CHDM20")) or 0.0
    ds20 = _safe_float(row.get("mkt_DS20")) or 1.0

    if regime == "RISK_ON" and score >= 60 and chdm20 > 50 and ds20 <= 0.45:
        return [
            "breakout_after_accumulation_v3",
            "compression_breakout_smt_v1",
            "mean_reversion_uptrend_ma50_v1",
        ]
    if regime in {"RISK_ON", "NEUTRAL"} and score >= 50 and ds20 <= 0.55:
        return [
            "breakout_after_accumulation_v3",
            "compression_breakout_smt_v1",
            "mean_reversion_uptrend_ma50_v1",
        ]
    return []


def _router_research_strategies(latest: pd.DataFrame) -> list[str]:
    """Regime router for research/shadow evaluation.

    This intentionally covers every non-rotation hypothesis family under the
    regime where it has a plausible edge. It is not the live default unless
    explicitly selected via --edge-strategy or MATA_EDGE_STRATEGY_ONLY.
    """
    if latest.empty:
        return [DEFAULT_STRATEGY]
    row = latest.iloc[0]
    regime = str(row.get("mkt_regime_state") or "")
    score = _safe_float(row.get("mkt_regime_score")) or 0.0
    chdm20 = _safe_float(row.get("mkt_CHDM20")) or 0.0
    ds20 = _safe_float(row.get("mkt_DS20")) or 1.0
    ds50 = _safe_float(row.get("mkt_DS50")) or 1.0
    chdm_delta = _safe_float(row.get("mkt_CHDM20_delta")) or 0.0

    if regime == "RISK_ON" and score >= 62 and chdm20 > 52 and ds20 <= 0.42:
        return [
            "breakout_after_accumulation_v3",
            "compression_breakout_smt_v1",
            "composite_edge_score_v2",
            "mean_reversion_uptrend_ma50_v1",
            "linreg_momentum_refined_v1",
            "adx_trend_refined_v1",
        ]

    if regime in {"RISK_ON", "NEUTRAL"} and score >= 52 and ds20 <= 0.55 and ds50 <= 0.55:
        strategies = [
            "breakout_after_accumulation_v3",
            "compression_breakout_smt_v1",
            "mean_reversion_uptrend_ma50_v1",
            "nr7_refined_v1",
        ]
        if chdm_delta >= -5:
            strategies.extend(["spring_refined_v1", "spring_reclaim_smt_v1"])
        return strategies

    if regime == "NEUTRAL" and score >= 45 and ds20 <= 0.65:
        return [
            "compression_breakout_smt_v1",
            "mean_reversion_uptrend_ma50_v1",
            "spring_refined_v1",
            "spring_reclaim_smt_v1",
        ]

    # Risk-off remains defensive. We still include the reversal research setup
    # names here so they are exercised in shadow; their own filters are strict
    # and currently reject most RISK_OFF cases.
    return [
        "spring_refined_v1",
        "spring_reclaim_smt_v1",
    ]


def _load_strategy(strategy_name: str, config_path: str | Path) -> Hypothesis:
    for hypothesis in load_hypotheses(config_path):
        if hypothesis.name == strategy_name:
            return hypothesis
    raise ValueError(f"Strategy not found: {strategy_name}")


def _setup_type_from_tags(hypothesis: Hypothesis) -> str:
    if hypothesis.risk.get("setup_type"):
        return str(hypothesis.risk["setup_type"])
    tags = set(hypothesis.tags)
    if "compression" in tags:
        return "EDGE_COMPRESSION"
    if "reversal" in tags or "spring" in tags or "reclaim" in tags:
        return "EDGE_REVERSAL"
    if "breakout" in tags:
        return "EDGE_BREAKOUT"
    if "trend" in tags or "continuation" in tags:
        return "EDGE_TREND"
    if "mean_reversion" in tags:
        return "EDGE_MEAN_REVERSION"
    return "EDGE"


def _failed_filters(row: pd.Series, hypothesis: Hypothesis) -> list[str]:
    failed = []
    frame = pd.DataFrame([row])
    for rule in hypothesis.filters:
        if not bool(evaluate_filters(frame, Hypothesis(name=hypothesis.name, filters=[rule])).iloc[0]):
            failed.append(_format_rule(rule, row.get(rule["column"])))
    return failed


def _format_rule(rule: dict[str, Any], actual: Any) -> str:
    return f"{rule.get('column')} {rule.get('op')} {rule.get('value')} (actual={_round_or_none(actual)})"


def _round_or_none(value: Any) -> float | str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return value


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None
