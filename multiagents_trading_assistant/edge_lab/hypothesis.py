"""Strategy hypothesis config and signal evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Hypothesis:
    name: str
    description: str = ""
    universe: str = "VN30"
    filters: list[dict[str, Any]] = field(default_factory=list)
    rank: list[dict[str, Any]] = field(default_factory=list)
    risk: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hypothesis":
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            universe=str(data.get("universe", "VN30")),
            filters=list(data.get("filters", [])),
            rank=list(data.get("rank", [])),
            risk=dict(data.get("risk", {})),
            tags=list(data.get("tags", [])),
        )


def load_hypotheses(path: str | Path) -> list[Hypothesis]:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    items = raw["hypotheses"] if isinstance(raw, dict) and "hypotheses" in raw else raw
    return [Hypothesis.from_dict(item) for item in items]


def evaluate_filters(frame: pd.DataFrame, hypothesis: Hypothesis) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for rule in hypothesis.filters:
        mask &= _eval_rule(frame, rule)
    return mask.fillna(False)


def rank_candidates(frame: pd.DataFrame, hypothesis: Hypothesis) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
    ranked["_edge_rank"] = 0.0
    for item in hypothesis.rank:
        col = item["column"]
        if col not in ranked.columns:
            continue
        ascending = bool(item.get("ascending", False))
        weight = float(item.get("weight", 1.0))
        ranked["_edge_rank"] += weight * ranked[col].rank(ascending=ascending, pct=True)
    return ranked.sort_values("_edge_rank", ascending=False)


def _eval_rule(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    col = rule["column"]
    op = str(rule["op"]).lower()
    value = rule.get("value")
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    left = frame[col]
    if op == ">":
        return left > value
    if op == ">=":
        return left >= value
    if op == "<":
        return left < value
    if op == "<=":
        return left <= value
    if op in {"==", "eq"}:
        return left == value
    if op in {"!=", "ne"}:
        return left != value
    if op == "between":
        lo, hi = value
        return left.between(lo, hi)
    if op == "in":
        return left.isin(value)
    if op == "not_in":
        return ~left.isin(value)
    if op == "up":
        return left > frame[str(rule.get("prev_column", f"{col}_prev"))]
    if op == "down":
        return left < frame[str(rule.get("prev_column", f"{col}_prev"))]
    if op == "not_up":
        return left <= frame[str(rule.get("prev_column", f"{col}_prev"))]
    raise ValueError(f"Unsupported filter op: {op}")

