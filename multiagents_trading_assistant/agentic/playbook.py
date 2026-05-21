"""Adaptive exit playbook — (setup_type, regime) → fixed exit params.

The LLM may only choose a playbook KEY from this finite menu. It may not
invent params. Each playbook should be backtested before being added here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Playbook:
    trail_atr_mult: float
    tp_ratio: float           # tp_target × tp_ratio (1.0 = no change vs strategy default)
    max_hold_bars: int
    time_stop_bars: int | None = None  # force exit if no progress in N bars
    breakeven_at_pct: float | None = None  # move SL to entry after +X%
    note: str = ""


# Keys: (setup_type, regime). regime ∈ {"UPTREND", "SIDEWAY", "DOWNTREND", "*"}
PLAYBOOKS: dict[tuple[str, str], Playbook] = {
    ("EDGE_BREAKOUT", "UPTREND"): Playbook(
        trail_atr_mult=2.8, tp_ratio=1.0, max_hold_bars=45,
        breakeven_at_pct=5.0,
        note="Strong trend — give breakout room to run",
    ),
    ("EDGE_BREAKOUT", "SIDEWAY"): Playbook(
        trail_atr_mult=2.2, tp_ratio=0.7, max_hold_bars=25,
        breakeven_at_pct=4.0,
        note="Choppy — take profit earlier, trail tighter",
    ),
    ("EDGE_COMPRESSION", "UPTREND"): Playbook(
        trail_atr_mult=2.5, tp_ratio=0.85, max_hold_bars=25,
        time_stop_bars=15, breakeven_at_pct=4.0,
        note="Compression breakouts decay fast — time stop",
    ),
    ("EDGE_COMPRESSION", "SIDEWAY"): Playbook(
        trail_atr_mult=2.0, tp_ratio=0.6, max_hold_bars=15,
        time_stop_bars=10, breakeven_at_pct=3.0,
        note="High false-break risk — tight trail",
    ),
    ("EDGE_MEAN_REVERSION", "UPTREND"): Playbook(
        trail_atr_mult=3.0, tp_ratio=0.7, max_hold_bars=35,
        time_stop_bars=20, breakeven_at_pct=4.0,
        note="Pullback in uptrend — quick TP, time stop if no recovery",
    ),
    ("EDGE_MEAN_REVERSION", "SIDEWAY"): Playbook(
        trail_atr_mult=2.5, tp_ratio=0.5, max_hold_bars=20,
        time_stop_bars=12, breakeven_at_pct=3.0,
        note="MR in chop — TP fast, exit if stalls",
    ),
}


DEFAULT_PLAYBOOK = Playbook(
    trail_atr_mult=2.5,
    tp_ratio=1.0,
    max_hold_bars=30,
    note="Fallback when no specific playbook matches",
)


def get_playbook(setup_type: str, regime: str) -> Playbook:
    """Lookup playbook with regime wildcard fallback."""
    setup_up = (setup_type or "").upper()
    regime_up = (regime or "").upper()
    key_exact = (setup_up, regime_up)
    if key_exact in PLAYBOOKS:
        return PLAYBOOKS[key_exact]
    key_any = (setup_up, "*")
    if key_any in PLAYBOOKS:
        return PLAYBOOKS[key_any]
    return DEFAULT_PLAYBOOK


def list_playbook_keys() -> list[tuple[str, str]]:
    return list(PLAYBOOKS.keys())
