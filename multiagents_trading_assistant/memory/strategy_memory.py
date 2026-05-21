"""strategy_memory.py — Strategy Memory (ℳₛ) from QuantAgents paper.

Paper: "Analysis of strategies from both simulated and real-world trading.
Stores strategy characteristics, performance metrics, risk profiles.
Updated weekly after Strategy Development Meetings."

Bob writes here after each Strategy Development Meeting.
Otto retrieves K=10 similar strategies at decision time via K-NN.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

_MEMORY_PATH = (
    Path(__file__).parent.parent / "data" / "strategy_memory.json"
)

# Thresholds for a strategy to be considered "active" (eligible for Otto)
_MIN_SAMPLE = 10
_MIN_PROFIT_FACTOR = 1.0


@dataclass
class StrategyRecord:
    """One strategy × regime performance snapshot from a simulation window."""

    strategy_id: str      # "{setup}_{regime}_{YYYY-MM}"
    setup: str            # e.g. BREAKOUT, MA_PULLBACK
    regime: str           # UPTREND | SIDEWAY | DOWNTREND
    win_rate: float       # 0.0–1.0
    profit_factor: float  # gross_profit / gross_loss
    avg_rr: float         # avg_win / avg_loss ratio
    avg_pnl_pct: float    # mean P&L per trade (%)
    sample_size: int      # number of simulated trades
    is_active: bool       # profit_factor >= 1.0 and sample_size >= 10
    updated_at: str       # ISO date of last simulation run
    market_snapshot: dict = field(default_factory=dict)


class StrategyMemory:
    """ℳₛ: persists strategy records and provides K-NN retrieval.

    Storage: JSON at data/strategy_memory.json — no external dependency.
    Retrieval: cosine similarity on a 4-dim normalised market condition vector.
    """

    def __init__(self, path: Path = _MEMORY_PATH) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[StrategyRecord] = self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> list[StrategyRecord]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return [StrategyRecord(**r) for r in raw]
        except Exception:
            return []

    def _save(self) -> None:
        self._path.write_text(
            json.dumps([asdict(r) for r in self._records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Update (Bob writes here) ──────────────────────────────────────────────

    def update(self, records: list[StrategyRecord]) -> None:
        """Upsert records by strategy_id, then persist to disk."""
        index = {r.strategy_id: r for r in self._records}
        for rec in records:
            index[rec.strategy_id] = rec
        self._records = list(index.values())
        self._save()

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_active_strategies(self, regime: str | None = None) -> list[StrategyRecord]:
        """Return active strategies, optionally filtered by regime.

        Active = profit_factor >= 1.0 AND sample_size >= 10.
        Sorted by profit_factor descending.
        """
        result = [r for r in self._records if r.is_active]
        if regime:
            result = [r for r in result if r.regime == regime]
        return sorted(result, key=lambda r: r.profit_factor, reverse=True)

    def retrieve_similar(
        self,
        market_conditions: dict,
        k: int = 10,
        regime: str | None = None,
    ) -> list[StrategyRecord]:
        """K-NN retrieval: return K most similar active strategy records.

        Similarity metric: cosine similarity on a 4-dim market vector.
        market_conditions keys:
          regime_score   float  -1=DOWNTREND, 0=SIDEWAY, 1=UPTREND
          vni_change_pct float  daily VNI change
          ma_ratio       float  MA20 / MA60 (trend strength proxy)
          vol_ratio      float  recent_vol / avg_vol_20d
        """
        query_vec = _to_vec(market_conditions)
        pool = self.get_active_strategies(regime=regime)
        if not pool:
            return []

        scored = sorted(
            pool,
            key=lambda r: _cosine(query_vec, _to_vec(r.market_snapshot)),
            reverse=True,
        )
        return scored[:k]

    def summary(self) -> dict:
        active = self.get_active_strategies()
        return {
            "total_records": len(self._records),
            "active_count": len(active),
            "top_strategies": [
                {
                    "setup": r.setup,
                    "regime": r.regime,
                    "profit_factor": r.profit_factor,
                    "win_rate": r.win_rate,
                    "sample_size": r.sample_size,
                }
                for r in active[:10]
            ],
        }


# ── Vector helpers ────────────────────────────────────────────────────────────

def _regime_score(regime: str) -> float:
    return {"UPTREND": 1.0, "SIDEWAY": 0.0, "DOWNTREND": -1.0}.get(regime, 0.0)


def _to_vec(snapshot: dict) -> list[float]:
    """Normalised 4-dim market condition vector."""
    return [
        float(snapshot.get("regime_score", 0.0)),
        float(snapshot.get("vni_change_pct", 0.0)) / 3.0,   # scale: ±3% typical
        float(snapshot.get("ma_ratio", 1.0)) - 1.0,          # centre around 0
        float(snapshot.get("vol_ratio", 1.0)) - 1.0,
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def deprecate_from_loss_patterns(
    memory: StrategyMemory,
    patterns: list[dict],
    *,
    regime: str,
    false_break_threshold: int = 15,
    structure_break_threshold: int = 20,
    min_total_loss_pct: float = -5.0,
) -> list[dict]:
    # Thresholds tuned via walk-forward sweep 2025-01 → 2026-05 (1.36 yrs):
    # (15, 20, -5) maximizes PnL +38.99M (vs baseline +35.19M, vs naive (8,12,-3) +33.80M)
    # while keeping Sharpe 1.06 (best), MDD -3.16% (2.2× better than baseline -6.92%).
    # Lower thresholds over-deprecate winners; higher leaves bad patterns active.
    """Closed-loop level 2: deprecate ℳₛ entries based on real-loss patterns.

    Flips `is_active=False` on (setup, regime) pairs where:
      - FALSE_BREAKOUT count ≥ false_break_threshold,  OR
      - STRUCTURE_BREAK count ≥ structure_break_threshold,  AND
      - average loss % is materially negative.

    `patterns` is the output of database.aggregate_loss_patterns().
    Only acts on records matching the current regime — strategies that lost
    badly in UPTREND don't necessarily fail in DOWNTREND.

    Returns list of {strategy_id, reason} for what was deprecated.
    """
    deprecated: list[dict] = []
    by_setup: dict[str, dict[str, dict]] = {}
    for p in patterns:
        if p.get("regime") and p["regime"] != regime and p["regime"] != "UNKNOWN":
            continue
        by_setup.setdefault(p["setup"], {})[p["loss_category"]] = p

    for record in memory._records:
        if not record.is_active or record.regime != regime:
            continue
        cat_map = by_setup.get(record.setup) or {}
        false_break = cat_map.get("FALSE_BREAKOUT") or {}
        struct_break = cat_map.get("STRUCTURE_BREAK") or {}

        fb_count = int(false_break.get("count") or 0)
        sb_count = int(struct_break.get("count") or 0)
        fb_avg = float(false_break.get("avg_pnl_pct") or 0.0)
        sb_avg = float(struct_break.get("avg_pnl_pct") or 0.0)

        reason = None
        if fb_count >= false_break_threshold and fb_avg <= min_total_loss_pct:
            reason = f"FALSE_BREAKOUT={fb_count} avg={fb_avg:.2f}% (≥{false_break_threshold})"
        elif sb_count >= structure_break_threshold and sb_avg <= min_total_loss_pct:
            reason = f"STRUCTURE_BREAK={sb_count} avg={sb_avg:.2f}% (≥{structure_break_threshold})"

        if reason:
            record.is_active = False
            deprecated.append({"strategy_id": record.strategy_id, "reason": reason})

    if deprecated:
        memory._save()
    return deprecated


def build_market_snapshot(
    regime: str,
    vni_change_pct: float,
    ma20: float,
    ma60: float,
    vol_ratio: float,
) -> dict:
    """Build the normalised market snapshot dict stored alongside each record."""
    return {
        "regime": regime,
        "regime_score": _regime_score(regime),
        "vni_change_pct": vni_change_pct,
        "ma_ratio": ma20 / ma60 if ma60 > 0 else 1.0,
        "vol_ratio": vol_ratio,
    }
