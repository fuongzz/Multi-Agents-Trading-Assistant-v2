"""Adapters from existing pipeline objects to Phase 1 agentic schemas."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from multiagents_trading_assistant.agentic.schemas import EvidencePacket, StrategySignal
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_LIVE_EDGE_FAMILY


def build_strategy_signal_from_candidate(
    candidate: Any,
    as_of_date: str,
    *,
    source: str = "core3",
    require_edge_passed: bool = True,
) -> StrategySignal | None:
    """Create a frozen StrategySignal from a screener candidate.

    By default this returns None unless the candidate has a passed edge strategy.
    That keeps deep agentic mode anchored to core3 instead of broad candidates.
    """

    indicators = getattr(candidate, "indicators", {}) or {}
    edge = indicators.get("edge_strategy_analysis", {}) or {}
    if require_edge_passed and not edge.get("passed"):
        return None

    symbol = str(getattr(candidate, "symbol", "")).upper()
    strategy_name = str(edge.get("strategy_name") or edge.get("strategy_family") or "")
    if not strategy_name:
        strategy_name = str(getattr(candidate, "setup_type", "") or "UNKNOWN")

    return StrategySignal(
        symbol=symbol,
        signal_date=as_of_date,
        as_of_date=as_of_date,
        source=source,
        strategy_name=strategy_name,
        strategy_family=str(edge.get("strategy_family") or DEFAULT_LIVE_EDGE_FAMILY),
        setup_type=str(edge.get("setup_type") or getattr(candidate, "setup_type", "") or "EDGE"),
        feature_date=_clean_scalar(edge.get("feature_date")),
        priority_score=_safe_float(getattr(candidate, "priority_score", None)),
        edge_score=_safe_float(edge.get("edge_score")),
        edge_rank=_safe_float(edge.get("edge_rank")),
        edge_rank_score=_safe_float(edge.get("edge_rank_score")),
        filters_failed=[str(item) for item in edge.get("filters_failed", [])],
        risk=_clean_mapping(edge.get("risk", {}) or {}),
        reasons=[str(item) for item in getattr(candidate, "reasons", [])],
        data_quality={
            "available": bool(edge.get("available", True)),
            "data_age_days": _safe_float(edge.get("data_age_days")),
            "data_quality_ok": bool(edge.get("data_quality_ok", True)),
            "error": _clean_scalar(edge.get("error")),
        },
    )


def build_evidence_packet_from_candidate(
    candidate: Any,
    as_of_date: str,
    *,
    backtest_mode: bool = False,
    signal: Any = None,
) -> EvidencePacket:
    """Build a replayable broker-grade evidence packet.

    Delegates to :class:`EvidenceBuilder` (Phase 2). Kept as a thin function
    wrapper so existing call sites keep working without changes.
    """

    from multiagents_trading_assistant.agentic.evidence_builder import EvidenceBuilder

    return EvidenceBuilder(as_of_date, backtest_mode=backtest_mode).build(
        candidate, signal=signal
    )


def format_agentic_context(
    strategy_signal: StrategySignal | dict | None,
    evidence_packet: EvidencePacket | dict | None,
) -> str:
    """Render compact context for LLM prompts."""

    if strategy_signal is None and evidence_packet is None:
        return ""

    signal = _model_or_dict(strategy_signal)
    evidence = _model_or_dict(evidence_packet)
    payload = {
        "strategy_signal": signal,
        "evidence_packet": _compact_evidence(evidence),
        "instructions": [
            "Treat strategy_signal as immutable deterministic core output.",
            "Review, reject, or propose bounded execution changes only.",
            "Do not invent a different source strategy.",
            "Final approval belongs to deterministic risk validation.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _compact_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    if not evidence:
        return {}
    technical = dict(evidence.get("technical") or evidence.get("indicators") or {})
    inner_indicators = dict(technical.get("indicators") or technical or {})
    keep = {
        key: inner_indicators.get(key)
        for key in [
            "current_price", "rsi", "macd_hist", "ma20", "ma60", "ma200",
            "volume_ratio_20", "avg_volume_20d", "stock_day_change_pct",
            "confluence_score", "pv_setup_tags_screener",
        ]
        if key in inner_indicators
    }
    return {
        "symbol": evidence.get("symbol"),
        "as_of_date": evidence.get("as_of_date"),
        "market": evidence.get("market") or evidence.get("market_context", {}),
        "sector": evidence.get("sector", {}),
        "technical": {
            "indicators": keep,
            "smart_money": technical.get("smart_money", {}),
            "money_cycle": technical.get("money_cycle", {}),
        },
        "money_flow": evidence.get("money_flow", {}),
        "edge": evidence.get("edge") or evidence.get("edge_strategy", {}),
        "fundamental": evidence.get("fundamental") or evidence.get("fundamentals", {}),
        "news": evidence.get("news", {}),
        "portfolio": evidence.get("portfolio", {}),
        "data_quality": evidence.get("data_quality", {}),
    }


def _model_or_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return _clean_mapping(value)
    return _clean_obj(value)


def _clean_obj(value: Any) -> Any:
    if is_dataclass(value):
        return _clean_mapping(asdict(value))
    if isinstance(value, dict):
        return _clean_mapping(value)
    if isinstance(value, (list, tuple, set)):
        return [_clean_obj(item) for item in value]
    return _clean_scalar(value)


def _clean_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _clean_obj(v) for k, v in value.items()}


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _safe_float(value: Any) -> float | None:
    value = _clean_scalar(value)
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None
