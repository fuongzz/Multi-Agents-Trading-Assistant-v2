"""Evolutionary strategy search engine.

Trường phái 2: parameter space được định nghĩa trước, LLM không dùng ở đây.
Chỉ dùng Python thuần để search, backtest, và báo cáo.

Usage:
    python -m multiagents_trading_assistant.edge_lab.evolve \\
        --universe vn100 --start 2023-01-01 --end 2025-12-31 \\
        --pop-size 40 --generations 50 --workers 4
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from multiagents_trading_assistant.edge_lab.backtest import run_signal_quality
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, load_hypotheses
from multiagents_trading_assistant.edge_lab.universe import get_universe

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "vn30_money_smt_hypotheses.json"

# ── Parameter search space ────────────────────────────────────────────────────

# Numeric filter columns: (lo, hi, is_int)
FILTER_SPACE: dict[str, tuple[float, float, bool]] = {
    "mkt_regime_score": (40.0, 65.0, False),
    "smart_money_score": (46.0, 76.0, False),
    "smart_money_score_no_sector": (46.0, 76.0, False),
    "smart_money_score_no_sector_delta": (-5.0, 10.0, False),
    "smart_money_score_delta": (-5.0, 10.0, False),
    "rs_percentile_20": (0.35, 0.78, False),
    "distribution_days_10": (0.0, 3.0, True),
    "accumulation_days_10": (0.0, 3.0, True),
    "CHDM50": (25.0, 72.0, False),
    "CHDM20": (25.0, 72.0, False),
    "DS20": (0.25, 0.75, False),
    "DS50": (0.25, 0.75, False),
    "value_ratio_20": (0.85, 1.65, False),
    "distance_ma20": (-0.10, 0.15, False),
    "distance_ma50": (-0.12, 0.12, False),
    "rsi14": (25.0, 78.0, False),
    "adx_14": (12.0, 38.0, False),
    "stoch_k": (15.0, 82.0, False),
    "cci_14": (-160.0, 220.0, False),
    "edge_score": (48.0, 78.0, False),
    "edge_score_no_sector": (48.0, 78.0, False),
    "pullback_quality_score": (38.0, 78.0, False),
    "value_flow_quality_score": (38.0, 78.0, False),
    "clv_score": (38.0, 78.0, False),
    "ma50_slope_10": (-0.05, 0.05, False),
    "ma20_slope_5": (-0.03, 0.03, False),
    "atr_ratio": (1.05, 2.5, False),
    "kalman_trend_5d": (0.005, 0.04, False),
    "kalman_confidence": (0.30, 0.85, False),
    "excess_ret_20d_pctile": (0.40, 0.80, False),
    "sector_rs_rank_20d": (0.40, 0.80, False),
}

# Boolean/flag columns injectable as {"column": x, "op": "==", "value": true}
BOOLEAN_FILTER_POOL: list[str] = [
    "breakout_20", "breakout_55", "above_ma20", "above_ma50",
    "spring_20", "hammer_like", "nr7", "tight_range_20",
    "retest_breakout_level", "inside_bar_breakout", "breakout_20_quality",
    "stoch_cross_up", "stoch_oversold_cross",
    "cci_momentum", "cci_oversold_bounce",
    "psar_bullish", "psar_flip_bull",
    "three_white_soldiers", "morning_star",
    "doji_at_support", "vol_climax_reversal",
    "triple_ma_align", "triple_ma_pullback",
    "volatility_squeeze_break", "kalman_strong_uptrend",
    "adx_bullish", "reclaim_ma20", "reclaim_ma50",
    "ich_price_above_cloud", "ich_tk_bull", "ich_kijun_bounce",
]

# Risk param bounds: (lo, hi)
RISK_BOUNDS: dict[str, tuple[float, float]] = {
    "stop_loss": (0.05, 0.13),
    "take_profit": (0.15, 0.48),
    "max_holding_bars": (18.0, 65.0),
    "initial_atr_stop_mult": (1.5, 3.2),
    "trailing_atr_mult": (2.0, 3.8),
    "trailing_profit_activation": (0.05, 0.18),
}

# Required filters that must not be removed (regime guard)
REQUIRED_COLS = {"mkt_regime_state", "data_quality_ok"}


# ── Fitness ───────────────────────────────────────────────────────────────────

def compute_fitness(
    results: pd.DataFrame,
    primary_hold: int = 20,
    min_trades: int = 15,
    wr_min: float = 0.40,
    rr_min: float = 1.0,
    target_trades: int = 0,   # 0 = no frequency bonus (legacy mode)
    n_filters: int = 0,       # for filter-complexity penalty
    min_freq_ratio: float = 0.20,  # min fraction of target_trades required (hard gate)
) -> float:
    """Score a hypothesis from signal-quality forward-return results.

    target_trades > 0 activates frequency-aware mode:
      - minimum frequency gate: must have >= min_freq_ratio * target_trades signals
      - frequency bonus rewards strategies closer to target signal count
      - filter complexity penalty: extra filters above 5 reduce fitness
      - wr_min / rr_min gates are enforced strictly
    """
    if results.empty:
        return 0.0
    sub = results[results["hold"] == primary_hold]
    if len(sub) < min_trades:
        return 0.0
    pnl = sub["pnl_pct"].dropna()
    if len(pnl) < min_trades:
        return 0.0

    # Hard frequency gate: reject strategies with too few signals relative to target
    if target_trades > 0 and len(pnl) < max(min_trades, int(target_trades * min_freq_ratio)):
        return 0.0

    mean_pnl = pnl.mean()
    std_pnl = pnl.std()
    if std_pnl < 0.01 or mean_pnl <= 0:
        return 0.0

    wr = float((pnl > 0).mean())
    if wr < wr_min:
        return 0.0

    pos = pnl[pnl > 0]
    neg = pnl[pnl < 0]
    avg_win  = float(pos.mean()) if len(pos) > 0 else 0.0
    avg_loss = float(abs(neg.mean())) if len(neg) > 0 else 1e-6
    rr = avg_win / avg_loss
    if rr < rr_min:
        return 0.0

    pf = pos.sum() / (abs(neg.sum()) + 1e-6)
    sharpe = mean_pnl / std_pnl * math.sqrt(252.0 / primary_hold)
    pf_mult = 1.0 if pf >= 1.20 else (0.7 if pf >= 0.90 else 0.0)

    # Filter complexity penalty: each filter above 5 reduces fitness by 8%
    # This pressures the optimizer to find simple, general strategies
    complexity_discount = max(0.3, 1.0 - max(0, n_filters - 5) * 0.08) if n_filters > 0 else 1.0

    if target_trades > 0:
        # Frequency bonus: peaks at target_trades, diminishes above 2× target
        freq_ratio = len(pnl) / target_trades
        freq_bonus = math.sqrt(min(freq_ratio, 2.0))          # 0 → √2  as n → 2×target
        # Quality bonus: extra reward for beating wr_min by a margin
        wr_bonus = 1.0 + max(0.0, (wr - wr_min) / (1.0 - wr_min)) * 0.5
        return max(0.0, sharpe) * freq_bonus * wr_bonus * pf_mult * complexity_discount
    else:
        # Legacy mode: flat n_bonus
        n_bonus = min(math.sqrt(len(pnl) / 20.0), 2.0)
        return max(0.0, sharpe) * n_bonus * pf_mult * complexity_discount


def _eval_hypothesis(
    hyp_dict: dict[str, Any],
    features: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    start: str,
    end: str,
    holds: list[int],
    min_trades: int,
    primary_hold: int,
    wr_min: float = 0.40,
    rr_min: float = 1.0,
    target_trades: int = 0,
    min_freq_ratio: float = 0.20,
) -> tuple[float, dict[str, Any]]:
    """Worker function (runs in subprocess)."""
    try:
        hyp = Hypothesis.from_dict(hyp_dict)
        n_filters = len(hyp_dict.get("filters", []))
        results = run_signal_quality(
            features, price_map, [hyp], start, end,
            holds=holds, top_n=10,
        )
        score = compute_fitness(
            results, primary_hold=primary_hold, min_trades=min_trades,
            wr_min=wr_min, rr_min=rr_min, target_trades=target_trades,
            n_filters=n_filters, min_freq_ratio=min_freq_ratio,
        )
        metrics = _summary_metrics(results, primary_hold)
        return score, {**hyp_dict, "_fitness": score, "_n_filters": n_filters, **metrics}
    except Exception as exc:
        return 0.0, {**hyp_dict, "_fitness": 0.0, "_error": str(exc)}


def _summary_metrics(results: pd.DataFrame, hold: int) -> dict[str, Any]:
    if results.empty:
        return {}
    sub = results[results["hold"] == hold]
    if sub.empty:
        return {}
    pnl = sub["pnl_pct"].dropna()
    pos = pnl[pnl > 0]
    neg = pnl[pnl < 0]
    avg_win  = float(pos.mean()) if len(pos) > 0 else 0.0
    avg_loss = float(abs(neg.mean())) if len(neg) > 0 else 1e-6
    return {
        "_n_trades": len(pnl),
        "_win_rate": round(float((pnl > 0).mean()), 3),
        "_mean_pnl": round(float(pnl.mean()), 3),
        "_profit_factor": round(float(pos.sum() / (abs(neg.sum()) + 1e-6)), 3),
        "_rr": round(avg_win / avg_loss, 2),
        "_sharpe": round(float(pnl.mean() / (pnl.std() + 1e-6) * math.sqrt(252.0 / hold)), 3),
    }


# ── Genetic operators ─────────────────────────────────────────────────────────

def _perturb_numeric(value: float, lo: float, hi: float, is_int: bool, rng: random.Random) -> float:
    span = hi - lo
    delta = rng.gauss(0, span * 0.12)
    new_val = float(np.clip(value + delta, lo, hi))
    if is_int:
        return float(int(round(new_val)))
    # round to 2 significant decimals relative to span
    precision = max(0, 2 - int(math.log10(span + 1e-9)))
    return round(new_val, precision + 2)


def mutate(hyp_dict: dict[str, Any], rng: random.Random, gen: int = 0, max_filters: int = 12) -> dict[str, Any]:
    """Return a mutated copy of hyp_dict.

    Bias toward removing filters when overloaded to counteract convergence:
    - always available: perturb_filter, perturb_risk
    - add more if conditions met: inject_filter (only under max), remove_filter, remove_bool_filter
    - add toggle_bool_filter only when bool count < 3 (prevent boolean pile-up)
    """
    d = copy.deepcopy(hyp_dict)
    filters: list[dict] = d.get("filters", [])
    risk: dict = d.get("risk", {})

    bool_filters = [f for f in filters if f.get("op") == "==" and f.get("value") is True]
    bool_count = len(bool_filters)
    total_count = len(filters)

    # Build weighted op pool — removal ops are weighted higher when overloaded
    ops: list[str] = ["perturb_filter", "perturb_risk"]
    weights: list[float] = [1.0, 0.8]

    if bool_count >= 2:
        # remove a boolean filter — weighted by how many booleans exist
        ops.append("remove_bool_filter")
        weights.append(0.5 + 0.3 * bool_count)  # more booleans → more pressure to remove

    if total_count > 4:
        ops.append("remove_filter")
        weights.append(0.4 + 0.2 * max(0, total_count - 5))  # more filters → more removal pressure

    if bool_count < 3:
        # only add more booleans when we have few — prevents boolean pile-up
        ops.append("toggle_bool_filter")
        weights.append(0.4 if bool_count == 0 else 0.25)

    if total_count < max_filters:
        ops.append("inject_filter")
        weights.append(0.5 if total_count < 4 else 0.2)  # less incentive to inject when already complex

    op = rng.choices(ops, weights=weights, k=1)[0]

    if op == "perturb_filter":
        numeric = [
            (i, f) for i, f in enumerate(filters)
            if f.get("op") in (">=", "<=", ">", "<", "between")
            and f.get("column") in FILTER_SPACE
        ]
        if numeric:
            i, f = rng.choice(numeric)
            col = f["column"]
            lo, hi, is_int = FILTER_SPACE[col]
            f = dict(f)
            if f["op"] == "between":
                bounds = list(f["value"])
                idx = rng.randint(0, 1)
                bounds[idx] = _perturb_numeric(bounds[idx], lo, hi, is_int, rng)
                bounds.sort()
                f["value"] = bounds
            else:
                f["value"] = _perturb_numeric(float(f["value"]), lo, hi, is_int, rng)
            filters[i] = f

    elif op == "perturb_risk":
        param = rng.choice(list(RISK_BOUNDS.keys()))
        lo, hi = RISK_BOUNDS[param]
        current = float(risk.get(param, (lo + hi) / 2))
        new_val = float(np.clip(rng.gauss(current, (hi - lo) * 0.10), lo, hi))
        if param == "max_holding_bars":
            risk[param] = int(round(new_val))
        else:
            risk[param] = round(new_val, 3)

    elif op == "remove_bool_filter":
        # Remove one random boolean filter (never required cols)
        removable_bool = [
            i for i, f in enumerate(filters)
            if f.get("op") == "==" and f.get("value") is True
            and f.get("column") not in REQUIRED_COLS
        ]
        if removable_bool:
            filters.pop(rng.choice(removable_bool))

    elif op == "toggle_bool_filter":
        existing_bool = {f["column"] for f in filters if f.get("op") == "==" and f.get("value") is True}
        candidates = [c for c in BOOLEAN_FILTER_POOL if c not in existing_bool]
        if candidates:
            col = rng.choice(candidates)
            filters.append({"column": col, "op": "==", "value": True})

    elif op == "remove_filter":
        removable = [i for i, f in enumerate(filters) if f.get("column") not in REQUIRED_COLS]
        if removable:
            filters.pop(rng.choice(removable))

    elif op == "inject_filter":
        col = rng.choice(list(FILTER_SPACE.keys()))
        if col not in {f["column"] for f in filters}:
            lo, hi, is_int = FILTER_SPACE[col]
            val = rng.uniform(lo + (hi - lo) * 0.2, hi - (hi - lo) * 0.2)
            if is_int:
                val = float(int(round(val)))
            op_str = rng.choice([">=", "<="])
            filters.append({"column": col, "op": op_str, "value": round(val, 3)})

    if "max_holding_bars" in risk:
        risk["max_holding_bars"] = int(round(float(risk["max_holding_bars"])))
    d["filters"] = filters
    d["risk"] = risk
    d["name"] = f"evo_{abs(hash(json.dumps(d, sort_keys=True, default=str))) % 10_000_000:07d}"
    seen_tags: set[str] = set()
    clean_tags: list[str] = []
    for t in list(d.get("tags", [])) + ["evolved"]:
        if t not in seen_tags:
            seen_tags.add(t)
            clean_tags.append(t)
    d["tags"] = clean_tags
    d["description"] = f"Evolved from {hyp_dict.get('name', 'unknown')}"
    return d


def crossover(a: dict[str, Any], b: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Single-point crossover on filters list."""
    a_f = a.get("filters", [])
    b_f = b.get("filters", [])
    if not a_f or not b_f:
        return copy.deepcopy(a)
    split = rng.randint(1, max(1, len(a_f) - 1))
    child_filters = a_f[:split] + b_f[split:]
    # deduplicate by column (keep last)
    seen: dict[str, dict] = {}
    for f in child_filters:
        seen[f["column"]] = f
    child_filters = list(seen.values())

    child = copy.deepcopy(a)
    child["filters"] = child_filters
    # blend risk params
    for param in RISK_BOUNDS:
        va = float(a.get("risk", {}).get(param, 0))
        vb = float(b.get("risk", {}).get(param, 0))
        child["risk"][param] = round(va * 0.5 + vb * 0.5, 3)
    child["name"] = f"evo_{abs(hash(json.dumps(child, sort_keys=True, default=str))) % 10_000_000:07d}"
    child["tags"] = list(a.get("tags", [])) + ["evolved"]
    child["description"] = f"Crossover({a.get('name','?')}, {b.get('name','?')})"
    return child


# ── Diversity helpers ────────────────────────────────────────────────────────

def _filter_content_hash(d: dict[str, Any]) -> int:
    """Stable hash of a strategy's filter set (ignores name/description)."""
    canonical = sorted(
        (f.get("column", ""), f.get("op", ""), str(f.get("value", "")))
        for f in d.get("filters", [])
    )
    return hash(json.dumps(canonical, sort_keys=True))


def _filter_key(d: dict[str, Any]) -> frozenset[tuple[str, str]]:
    """(column, op) pairs — used for Jaccard similarity."""
    return frozenset(
        (f.get("column", ""), f.get("op", ""))
        for f in d.get("filters", [])
    )


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _apply_diversity_pressure(
    scored: list[tuple[float, dict[str, Any]]],
    penalty: float = 0.25,
) -> list[tuple[float, dict[str, Any]]]:
    """Penalize candidates whose filters are too similar to higher-ranked ones.

    Works by comparing (column, op) Jaccard similarity. A strategy that is
    85% similar to the top candidate gets its fitness reduced by 85% × penalty.
    This prevents the population from collapsing to a single strategy.
    """
    if penalty <= 0 or not scored:
        return scored
    result: list[tuple[float, dict[str, Any]]] = []
    seen_keys: list[frozenset] = []
    for score, d in scored:
        key = _filter_key(d)
        max_sim = max((_jaccard(key, s) for s in seen_keys), default=0.0)
        adjusted = score * (1.0 - penalty * max_sim)
        result.append((adjusted, d))
        seen_keys.append(key)
    return sorted(result, key=lambda x: x[0], reverse=True)


def _select_diverse_elite(
    scored: list[tuple[float, dict[str, Any]]],
    n_elite: int,
    max_jaccard: float = 0.55,
) -> list[tuple[float, dict[str, Any]]]:
    """Greedy niche selection: build elite pool ensuring no two members are
    more than max_jaccard similar.

    Strategy:
    1. Always take the top-scoring candidate.
    2. For each subsequent candidate (in fitness order), only add it if its
       max Jaccard similarity to any existing elite member is < max_jaccard.
    3. If we can't fill n_elite with diverse strategies, fill remaining slots
       with the best remaining (even if similar) to keep population size stable.
    """
    if not scored:
        return []
    elite: list[tuple[float, dict[str, Any]]] = []
    elite_keys: list[frozenset] = []
    overflow: list[tuple[float, dict[str, Any]]] = []

    for score, d in scored:
        if score <= 0:
            overflow.append((score, d))
            continue
        key = _filter_key(d)
        max_sim = max((_jaccard(key, k) for k in elite_keys), default=0.0)
        if max_sim < max_jaccard:
            elite.append((score, d))
            elite_keys.append(key)
        else:
            overflow.append((score, d))
        if len(elite) >= n_elite:
            break

    # fill remaining slots from overflow to keep population size
    while len(elite) < n_elite and overflow:
        elite.append(overflow.pop(0))

    return elite[:n_elite]


# ── Evolution config & engine ─────────────────────────────────────────────────

# ── Discord reporting ─────────────────────────────────────────────────────────

DISCORD_REPORT_EVERY_N = 5  # send summary every N generations


def _discord_send(content: str, webhook_url: str = "") -> None:
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")
    if not url:
        return
    try:
        requests.post(url, json={"content": content[:2000]}, timeout=10)
    except Exception as exc:
        logger.warning("Discord send failed: %s", exc)


def _fmt_top(scored: list[tuple[float, dict[str, Any]]], n: int = 5) -> str:
    lines = []
    for i, (score, d) in enumerate(scored[:n], 1):
        lines.append(
            f"  {i}. `{d.get('name','?')[:30]}` "
            f"fitness=**{score:.3f}** n={d.get('_n_trades','?')} "
            f"wr={d.get('_win_rate','?')} pf={d.get('_profit_factor','?')}"
        )
    return "\n".join(lines) if lines else "  (none)"


@dataclass
class EvolutionConfig:
    universe: list[str]
    start: str
    end: str
    pop_size: int = 40
    elite_frac: float = 0.30
    max_generations: int = 50
    holds: list[int] = field(default_factory=lambda: [10, 20])
    primary_hold: int = 20
    min_trades: int = 15
    n_workers: int = 2
    seed_config: Path = field(default_factory=lambda: DEFAULT_CONFIG)
    output_dir: Path = field(default_factory=lambda: Path("multiagents_trading_assistant/edge_lab/evolved"))
    seed_expand_factor: int = 3
    timeout_hours: float = 0.0
    random_seed: int = 42
    discord_webhook: str = ""
    validate_end: str = ""
    # Quality gates (enforced in fitness function)
    wr_min: float = 0.40          # minimum win rate; set 0.60 for "weekly quality" mode
    rr_min: float = 1.0           # minimum avg_win / avg_loss ratio
    target_trades: int = 0        # 0 = legacy mode; set e.g. 120 for weekly-signal mode
    min_freq_ratio: float = 0.20  # fraction of target_trades required as hard gate
    # Diversity & filter constraints
    max_filters: int = 12         # cap filter count to prevent over-restriction
    diversity_pressure: float = 0.25  # 0 = off, 1 = max penalty for similar strategies (soft)
    max_jaccard: float = 0.55     # hard niche threshold: elites must differ by this much
    immigrant_frac: float = 0.15  # fraction of new generation from fresh random seeds


def _strip_booleans(hyp_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with all boolean filters removed (keep only numeric + required)."""
    d = copy.deepcopy(hyp_dict)
    d["filters"] = [
        f for f in d.get("filters", [])
        if not (f.get("op") == "==" and f.get("value") is True)
        or f.get("column") in REQUIRED_COLS
    ]
    d["name"] = f"bare_{d.get('name', 'seed')}"
    d["description"] = f"Stripped booleans from {hyp_dict.get('name', 'seed')}"
    return d


def _init_population(
    seeds: list[Hypothesis],
    pop_size: int,
    expand_factor: int,
    rng: random.Random,
    max_filters: int = 12,
) -> list[dict[str, Any]]:
    """Expand seed hypotheses into initial population via mutation.

    Includes both the original seeds AND boolean-stripped versions so the
    optimizer starts with some high-frequency, low-filter candidates.
    """
    seed_dicts = [_hyp_to_dict(h) for h in seeds]
    # Create stripped (bare) versions of every seed
    bare_dicts = [_strip_booleans(d) for d in seed_dicts]
    # Deduplicate bare seeds by content
    seen_hashes: set[int] = set()
    unique_bare: list[dict[str, Any]] = []
    for d in bare_dicts:
        h = _filter_content_hash(d)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_bare.append(d)

    all_seeds = seed_dicts + unique_bare
    population: list[dict[str, Any]] = list(all_seeds)  # keep originals + bare
    target = max(pop_size, len(seed_dicts) * expand_factor)
    while len(population) < target:
        parent = rng.choice(all_seeds)
        population.append(mutate(parent, rng, max_filters=max_filters))
    return population[:target]


def _hyp_to_dict(hyp: Hypothesis) -> dict[str, Any]:
    return {
        "name": hyp.name,
        "description": hyp.description,
        "universe": hyp.universe,
        "tags": list(hyp.tags),
        "filters": [dict(f) for f in hyp.filters],
        "rank": [dict(r) for r in hyp.rank],
        "risk": dict(hyp.risk),
    }


class EvolutionEngine:
    def __init__(self, cfg: EvolutionConfig, root: str | Path = ".") -> None:
        self.cfg = cfg
        self.root = Path(root)
        self.rng = random.Random(cfg.random_seed)
        self.np_rng = np.random.default_rng(cfg.random_seed)
        self.all_results: list[dict[str, Any]] = []  # cumulative across generations

    def run(self) -> list[dict[str, Any]]:
        cfg = self.cfg
        logger.info("Building feature table for %d symbols…", len(cfg.universe))
        t0 = time.time()
        features, price_map = build_feature_table(
            cfg.universe, cfg.start, cfg.end, root=self.root
        )
        logger.info("Features built in %.1fs — shape %s", time.time() - t0, features.shape)

        seeds = [h for h in load_hypotheses(cfg.seed_config) if not h.name.startswith("regime_router")]
        population = _init_population(seeds, cfg.pop_size, cfg.seed_expand_factor, self.rng, cfg.max_filters)
        self._seed_dicts = [_hyp_to_dict(h) for h in seeds]  # kept for immigrant injection
        logger.info("Initial population: %d candidates from %d seeds", len(population), len(seeds))

        hook = cfg.discord_webhook or os.getenv("DISCORD_WEBHOOK_URL", "")
        _discord_send(
            f"🧬 **Evolution started** | universe={len(cfg.universe)} symbols "
            f"| pop={cfg.pop_size} | gens={cfg.max_generations} "
            f"| seeds={len(seeds)} → init_pop={len(population)}\n"
            f"Period: `{cfg.start}` → `{cfg.end}`",
            hook,
        )

        best_ever: dict[str, Any] | None = None
        deadline = time.time() + cfg.timeout_hours * 3600 if cfg.timeout_hours > 0 else float("inf")

        for gen in range(cfg.max_generations):
            if time.time() > deadline:
                logger.info("Timeout reached at generation %d", gen)
                break

            gen_t0 = time.time()
            scored = self._eval_population(population, features, price_map)
            scored.sort(key=lambda x: x[0], reverse=True)
            # Step 1: soft re-ranking via diversity pressure
            scored = _apply_diversity_pressure(scored, cfg.diversity_pressure)
            # Step 2: hard niche enforcement — elite must be genuinely diverse
            n_elite = max(3, int(len(scored) * cfg.elite_frac))
            top = _select_diverse_elite(scored, n_elite, max_jaccard=cfg.max_jaccard)
            self.all_results.extend(x[1] for x in scored)

            if top:
                current_best = top[0][1]
                if best_ever is None or top[0][0] > best_ever.get("_fitness", 0):
                    best_ever = current_best

            self._report_generation(gen, scored, time.time() - gen_t0)
            self._save_generation(gen, [x[1] for x in top])

            # Discord report every N generations
            if (gen + 1) % DISCORD_REPORT_EVERY_N == 0 or gen == 0:
                n_viable = sum(1 for s, _ in scored if s > 0)
                elapsed_total = time.time() - t0
                _discord_send(
                    f"🧬 **Gen {gen+1}/{cfg.max_generations}** "
                    f"| viable={n_viable}/{len(scored)} "
                    f"| elapsed={elapsed_total/60:.1f}min\n"
                    f"**Top 5:**\n{_fmt_top(scored)}",
                    hook,
                )

            elite_dicts = [x[1] for x in top]
            population = self._next_generation(elite_dicts)

        top20 = self._save_final(hook)

        if cfg.validate_end:
            val_start = (pd.Timestamp(cfg.end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            self._run_validation(top20, val_start, cfg.validate_end, hook)

        return top20

    def _eval_population(
        self,
        population: list[dict[str, Any]],
        features: pd.DataFrame,
        price_map: dict[str, pd.DataFrame],
    ) -> list[tuple[float, dict[str, Any]]]:
        cfg = self.cfg
        results: list[tuple[float, dict[str, Any]]] = []

        if cfg.n_workers <= 1:
            for hyp_dict in population:
                score, info = _eval_hypothesis(
                    hyp_dict, features, price_map,
                    cfg.start, cfg.end, cfg.holds, cfg.min_trades, cfg.primary_hold,
                    cfg.wr_min, cfg.rr_min, cfg.target_trades, cfg.min_freq_ratio,
                )
                results.append((score, info))
        else:
            with ProcessPoolExecutor(max_workers=cfg.n_workers) as pool:
                futures = {
                    pool.submit(
                        _eval_hypothesis, hyp_dict, features, price_map,
                        cfg.start, cfg.end, cfg.holds, cfg.min_trades, cfg.primary_hold,
                        cfg.wr_min, cfg.rr_min, cfg.target_trades, cfg.min_freq_ratio,
                    ): hyp_dict
                    for hyp_dict in population
                }
                for fut in as_completed(futures):
                    try:
                        score, info = fut.result(timeout=120)
                    except Exception as exc:
                        hyp_dict = futures[fut]
                        score, info = 0.0, {**hyp_dict, "_fitness": 0.0, "_error": str(exc)}
                    results.append((score, info))
        return results

    def _next_generation(self, elite: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build next generation from diverse elite + immigrants.

        immigrants_frac of slots are filled from fresh seed mutations (not elite),
        preventing the population from collapsing to a single elite lineage.
        """
        cfg = self.cfg
        rng = self.rng
        new_pop: list[dict[str, Any]] = list(elite)

        n_immigrants = max(1, int(cfg.pop_size * cfg.immigrant_frac))
        n_offspring = cfg.pop_size - len(elite) - n_immigrants

        # Offspring: mutations and crossovers from diverse elite
        for _ in range(max(0, n_offspring)):
            op = rng.choices(["mutate", "crossover"], weights=[0.60, 0.40])[0]
            if op == "mutate":
                parent = rng.choice(elite)
                new_pop.append(mutate(parent, rng, gen=len(self.all_results), max_filters=cfg.max_filters))
            else:
                if len(elite) >= 2:
                    a, b = rng.sample(elite, 2)
                    child = crossover(a, b, rng)
                    new_pop.append(mutate(child, rng, max_filters=cfg.max_filters))
                else:
                    new_pop.append(mutate(elite[0], rng, max_filters=cfg.max_filters))

        # Immigrants: fresh mutations from original seeds (bypasses elite lineage)
        seed_pool = getattr(self, "_seed_dicts", []) or elite
        for _ in range(n_immigrants):
            parent = rng.choice(seed_pool)
            # 50% chance to strip booleans first → high-frequency fresh blood
            if rng.random() < 0.5:
                parent = _strip_booleans(parent)
            new_pop.append(mutate(parent, rng, max_filters=cfg.max_filters))

        return new_pop[:cfg.pop_size]

    def _report_generation(
        self,
        gen: int,
        scored: list[tuple[float, dict[str, Any]]],
        elapsed: float,
    ) -> None:
        n_viable = sum(1 for s, _ in scored if s > 0)
        top3 = scored[:3]
        lines = [
            f"\n{'='*60}",
            f"Generation {gen+1:3d} | viable={n_viable}/{len(scored)} | {elapsed:.1f}s",
            f"{'='*60}",
        ]
        for rank, (score, d) in enumerate(top3, 1):
            lines.append(
                f"  #{rank} fitness={score:.4f} | n={d.get('_n_trades','?')} "
                f"wr={d.get('_win_rate','?')} pf={d.get('_profit_factor','?')} "
                f"sharpe={d.get('_sharpe','?')} filters={d.get('_n_filters','?')} | {d.get('name','?')}"
            )
        print("\n".join(lines))
        logger.info("Gen %d: top_fitness=%.4f viable=%d/%d", gen+1, scored[0][0] if scored else 0, n_viable, len(scored))

    def _save_generation(self, gen: int, elite: list[dict[str, Any]]) -> None:
        out_dir = self.root / self.cfg.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"gen_{gen+1:03d}_elite.json"
        _write_json({"generation": gen + 1, "hypotheses": elite}, path)

    def _save_final(self, hook: str = "") -> list[dict[str, Any]]:
        out_dir = self.root / self.cfg.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        ranked = sorted(self.all_results, key=lambda d: d.get("_fitness", 0), reverse=True)
        seen_filter_hashes: set[int] = set()
        top: list[dict[str, Any]] = []
        for d in ranked:
            if d.get("_fitness", 0) <= 0:
                continue
            fhash = _filter_content_hash(d)
            if fhash not in seen_filter_hashes:
                seen_filter_hashes.add(fhash)
                top.append(d)
            if len(top) >= 20:
                break

        clean = [{k: v for k, v in d.items() if not k.startswith("_")} for d in top]
        _write_json({"hypotheses": clean}, out_dir / "evolved_top20.json")
        print(f"\n✓ Saved top {len(clean)} evolved hypotheses → {out_dir / 'evolved_top20.json'}")

        df = pd.DataFrame(self.all_results)
        df.to_csv(out_dir / "evolution_log.csv", index=False)
        print(f"✓ Full evolution log → {out_dir / 'evolution_log.csv'} ({len(df)} candidates)")

        cfg = self.cfg
        val_note = f"\n⏳ Running validation `{(pd.Timestamp(cfg.end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}` → `{cfg.validate_end}`…" if cfg.validate_end else ""
        top_lines = [
            f"  {i}. `{d.get('name','?')[:28]}` "
            f"fit=**{d.get('_fitness',0):.3f}** "
            f"n={d.get('_n_trades','?')} wr={d.get('_win_rate','?')} "
            f"pf={d.get('_profit_factor','?')} sharpe={d.get('_sharpe','?')}"
            for i, d in enumerate(top[:10], 1)
        ]
        _discord_send(
            f"✅ **Evolution done** | {len(self.all_results)} candidates | "
            f"train `{cfg.start}`→`{cfg.end}`\n"
            f"**Top {len(top)} saved** → `evolved_top20.json`\n\n"
            + "\n".join(top_lines) + val_note,
            hook,
        )
        return top

    def _run_validation(
        self,
        train_top: list[dict[str, Any]],
        val_start: str,
        val_end: str,
        hook: str = "",
    ) -> None:
        cfg = self.cfg
        out_dir = self.root / cfg.output_dir
        print(f"\n{'='*60}")
        print(f"Validation: {val_start} → {val_end}  ({len(train_top)} strategies)")
        print(f"{'='*60}")

        try:
            val_features, val_price_map = build_feature_table(
                cfg.universe, val_start, val_end, root=self.root
            )
        except Exception as exc:
            msg = f"❌ Validation feature build failed: {exc}"
            print(msg)
            _discord_send(msg, hook)
            return

        rows: list[dict[str, Any]] = []
        for d in train_top:
            try:
                hyp = Hypothesis.from_dict({k: v for k, v in d.items() if not k.startswith("_")})
                results = run_signal_quality(
                    val_features, val_price_map, [hyp],
                    val_start, val_end,
                    holds=cfg.holds, top_n=10,
                )
                val_score = compute_fitness(results, primary_hold=cfg.primary_hold, min_trades=8)
                val_metrics = _summary_metrics(results, cfg.primary_hold)
            except Exception as exc:
                val_score = 0.0
                val_metrics = {"_val_error": str(exc)}

            rows.append({
                "name": d.get("name"),
                "train_fitness": d.get("_fitness", 0),
                "train_wr": d.get("_win_rate"),
                "train_pf": d.get("_profit_factor"),
                "train_sharpe": d.get("_sharpe"),
                "train_n": d.get("_n_trades"),
                "val_fitness": val_score,
                "val_wr": val_metrics.get("_win_rate"),
                "val_pf": val_metrics.get("_profit_factor"),
                "val_sharpe": val_metrics.get("_sharpe"),
                "val_n": val_metrics.get("_n_trades"),
            })

        val_df = pd.DataFrame(rows).sort_values("val_fitness", ascending=False)
        val_df.to_csv(out_dir / "validation_results.csv", index=False)
        print(f"✓ Validation results → {out_dir / 'validation_results.csv'}")

        # Build Discord report: train vs val side-by-side
        survived = val_df[val_df["val_fitness"] > 0]
        overfit = val_df[val_df["val_fitness"] <= 0]

        lines = [
            f"📊 **Validation complete** | `{val_start}` → `{val_end}`",
            f"✅ Survived: **{len(survived)}/{len(val_df)}** strategies",
            f"❌ Overfit/dead: {len(overfit)}",
            "",
            "**Top by val_fitness** (train → val):",
        ]
        for _, row in val_df.head(8).iterrows():
            arrow = "✅" if (row["val_fitness"] or 0) > 0 else "❌"
            lines.append(
                f"  {arrow} `{str(row['name'])[:26]}` "
                f"fit {row['train_fitness']:.2f}→**{row['val_fitness']:.2f}** "
                f"wr {row['train_wr']}→{row['val_wr']} "
                f"pf {row['train_pf']}→{row['val_pf']}"
            )

        # Save survived strategies as deployable config
        survived_names = set(survived["name"].tolist())
        deployable = [
            {k: v for k, v in d.items() if not k.startswith("_")}
            for d in train_top
            if d.get("name") in survived_names
        ]
        _write_json({"hypotheses": deployable}, out_dir / "evolved_validated.json")
        lines.append(f"\n💾 Deployable: **{len(deployable)}** strategies → `evolved_validated.json`")

        _discord_send("\n".join(lines), hook)
        print(f"✓ Deployable config → {out_dir / 'evolved_validated.json'} ({len(deployable)} strategies)")


def _write_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evolutionary strategy search for VN market",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--universe", default="vn100", help="vn30 | vn100 | liquid | comma-separated symbols")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--pop-size", type=int, default=40)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--workers", type=int, default=2, help="Parallel workers (processes)")
    p.add_argument("--min-trades", type=int, default=15)
    p.add_argument("--primary-hold", type=int, default=20, help="Forward-return hold period for fitness")
    p.add_argument("--seed-config", default=str(DEFAULT_CONFIG))
    p.add_argument("--output-dir", default="multiagents_trading_assistant/edge_lab/evolved")
    p.add_argument("--timeout-hours", type=float, default=0.0, help="Stop after N hours (0=unlimited)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--root", default=".", help="Project root directory")
    p.add_argument("--validate-end", default="", help="Auto-validate on (end+1d to this date) after evolution")
    p.add_argument("--wr-min", type=float, default=0.40, help="Min win rate gate (0.60 for weekly-quality mode)")
    p.add_argument("--rr-min", type=float, default=1.0,  help="Min avg_win/avg_loss ratio gate")
    p.add_argument("--target-trades", type=int, default=0, help="Target signal count for freq bonus (e.g. 120 = ~1/week over 3yr)")
    p.add_argument("--min-freq-ratio", type=float, default=0.20, help="Hard gate: min fraction of target-trades required (default 0.20 = 20pct)")
    p.add_argument("--max-filters", type=int, default=12, help="Max filter count per strategy (lower = more signals)")
    p.add_argument("--diversity-pressure", type=float, default=0.25, help="Soft penalty for similar strategies (0=off, 1=max)")
    p.add_argument("--max-jaccard", type=float, default=0.55, help="Hard niche threshold: elite members must differ by at least (1 - max_jaccard)")
    p.add_argument("--immigrant-frac", type=float, default=0.15, help="Fraction of each generation from fresh random seeds (prevents lineage collapse)")
    p.add_argument("--discord-webhook", default="", help="Override Discord webhook URL")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    root = Path(args.root)

    # resolve universe
    uni_arg = args.universe.lower()
    if uni_arg in ("vn30", "vn100", "hnx30", "upcom30", "wide", "all_research"):
        try:
            symbols = get_universe(uni_arg)
        except Exception as exc:
            raise SystemExit(f"Cannot resolve universe '{uni_arg}': {exc}")
    else:
        symbols = [s.strip().upper() for s in uni_arg.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols resolved. Use --universe vn30|vn100 or comma-separated tickers.")

    cfg = EvolutionConfig(
        universe=symbols,
        start=args.start,
        end=args.end,
        pop_size=args.pop_size,
        max_generations=args.generations,
        n_workers=args.workers,
        min_trades=args.min_trades,
        primary_hold=args.primary_hold,
        seed_config=Path(args.seed_config),
        output_dir=Path(args.output_dir),
        timeout_hours=args.timeout_hours,
        random_seed=args.seed,
        discord_webhook=args.discord_webhook,
        validate_end=args.validate_end,
        wr_min=args.wr_min,
        rr_min=args.rr_min,
        target_trades=args.target_trades,
        min_freq_ratio=args.min_freq_ratio,
        max_filters=args.max_filters,
        diversity_pressure=args.diversity_pressure,
        max_jaccard=args.max_jaccard,
        immigrant_frac=args.immigrant_frac,
    )

    engine = EvolutionEngine(cfg, root=root)
    top = engine.run()

    print(f"\n{'='*60}")
    print(f"Evolution complete — top strategies:")
    for i, d in enumerate(top[:10], 1):
        print(
            f"  {i:2d}. fitness={d.get('_fitness',0):.4f} "
            f"n={d.get('_n_trades','?')} wr={d.get('_win_rate','?')} "
            f"pf={d.get('_profit_factor','?')} | {d.get('name','?')}"
        )


if __name__ == "__main__":
    main()
