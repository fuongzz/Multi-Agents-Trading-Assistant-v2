"""Realtime paper runner for side-by-side production pipeline comparison.

This module runs two paper portfolios in parallel:

- live_current: the existing intraday edge family.
- core_mvp9_shadow: the full 9-strategy core_mvp sleeve.

It does not send broker orders and does not write to the real positions table.
State is stored as JSON under data/runtime/realtime_shadow and dashboards are
written as static auto-refreshing HTML files under reports/realtime_shadow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, time as dt_time
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from multiagents_trading_assistant.edge_lab.live_signal import (
    DEFAULT_LIVE_EDGE_FAMILY,
    get_edge_strategy_signals,
)
from multiagents_trading_assistant.edge_lab.strategy_sleeves import MVP_EDGE_STRATEGIES, catalog_rows
from multiagents_trading_assistant.fetcher import (
    get_live_price,
    get_vn30_symbols,
    get_vn100_symbols,
)
from multiagents_trading_assistant.orchestrator.trade_graph import run_pipeline as run_trade_graph
from multiagents_trading_assistant.services.portfolio_service import compute_profit_trailing_sl


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "data" / "runtime" / "realtime_shadow"
REPORT_DIR = ROOT / "reports" / "realtime_shadow"
LAYERED_DEMO_DIR = ROOT / "reports" / "layered_pipeline_demo_2026-05-17"
DATA_DIR = ROOT / "multiagents_trading_assistant" / "data"
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


@dataclass(frozen=True)
class PipelineSpec:
    portfolio_id: str
    label: str
    strategy_family: str
    max_positions: int = 5
    max_expanded_positions: int = 5
    max_candidates: int = 15
    allow_dynamic_expansion: bool = False
    allow_rotation: bool = False


PIPELINES: tuple[PipelineSpec, ...] = (
    PipelineSpec(
        portfolio_id="live_current",
        label="Live Current Edge Family",
        strategy_family=DEFAULT_LIVE_EDGE_FAMILY,
        max_positions=5,
        max_expanded_positions=5,
        max_candidates=15,
        allow_dynamic_expansion=False,
        allow_rotation=False,
    ),
    PipelineSpec(
        portfolio_id="core_mvp9_shadow",
        label="Core MVP9 Shadow",
        strategy_family=",".join(MVP_EDGE_STRATEGIES),
        max_positions=5,
        max_expanded_positions=8,
        max_candidates=15,
        allow_dynamic_expansion=False,
        allow_rotation=False,
    ),
)


def _now() -> datetime:
    return datetime.now(VN_TZ)


def _today() -> str:
    return _now().strftime("%Y-%m-%d")


def _is_market_session(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    current = now.time()
    morning = dt_time(9, 0) <= current <= dt_time(11, 30)
    afternoon = dt_time(13, 0) <= current <= dt_time(14, 45)
    return morning or afternoon


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value)


def _load_state(spec: PipelineSpec, initial_capital: float) -> dict[str, Any]:
    path = _state_path(spec)
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            state.setdefault("cash", initial_capital)
            state.setdefault("initial_capital", initial_capital)
            state.setdefault("holdings", [])
            state.setdefault("closed", [])
            state.setdefault("watchlist", [])
            state.setdefault("events", [])
            state.setdefault("stats", {})
            return state
        except Exception:
            pass
    return {
        "portfolio_id": spec.portfolio_id,
        "label": spec.label,
        "strategy_family": spec.strategy_family,
        "initial_capital": initial_capital,
        "cash": initial_capital,
        "holdings": [],
        "closed": [],
        "watchlist": [],
        "events": [],
        "stats": {},
        "watchlist_date": None,
        "created_at": _now().isoformat(timespec="seconds"),
        "updated_at": None,
    }


def _summary_metric(frame: pd.DataFrame, name: str, default: float = 0.0) -> float:
    if frame.empty or "metric" not in frame.columns or "value" not in frame.columns:
        return default
    rows = frame[frame["metric"].astype(str).eq(name)]
    if rows.empty:
        return default
    return _safe_float(rows.iloc[-1]["value"], default) or default


def _seed_continuation_from_layered_demo(
    spec: PipelineSpec,
    state: dict[str, Any],
    layered_demo_dir: Path | None,
    *,
    reset: bool = False,
) -> dict[str, Any]:
    if spec.portfolio_id != "core_mvp9_shadow" or not layered_demo_dir:
        return state
    if state.get("continuation_mode") == "layered_baseline" and not reset:
        return state

    summary_path = layered_demo_dir / "06a_portfolio_overview.csv"
    holdings_path = layered_demo_dir / "06c_open_holdings.csv"
    if not summary_path.exists() or not holdings_path.exists():
        return state

    summary = pd.read_csv(summary_path)
    holdings = pd.read_csv(holdings_path)
    if holdings.empty:
        return state
    if "source" in holdings.columns:
        holdings = holdings[holdings["source"].astype(str).ne("realtime_shadow")].copy()

    initial_nav = _summary_metric(summary, "Initial NAV", 100_000_000.0)
    cash = _summary_metric(summary, "Cash", 0.0)
    baseline_realized = _summary_metric(summary, "Closed realized PnL", 0.0)
    baseline_closed_trades = int(_summary_metric(summary, "Closed trades", 0.0))
    baseline_win_rate = _summary_metric(summary, "Closed win rate", 0.0)
    start_rows = summary[summary["metric"].astype(str).eq("Start date")] if "metric" in summary.columns else pd.DataFrame()
    start_date = str(start_rows.iloc[-1]["value"]) if not start_rows.empty else state.get("created_at", "")[:10]

    live_holdings: list[dict[str, Any]] = []
    for row in holdings.to_dict("records"):
        symbol = str(row.get("symbol") or "").upper().strip()
        entry = _safe_float(row.get("entry_price"), 0.0) or 0.0
        last = _safe_float(row.get("last_close"), entry) or entry
        qty = int(_safe_float(row.get("shares"), _safe_float(row.get("quantity"), 0.0)) or 0)
        if not symbol or entry <= 0 or qty <= 0:
            continue
        # The baseline open holdings ended at "END_OF_DATA", so we create a
        # live continuation SL/TP envelope from the latest marked price.
        sl = round(max(entry * 0.92, last * 0.92), 4)
        tp = round(max(entry * 1.2456, last * 1.12), 4)
        live_holdings.append(
            {
                "symbol": symbol,
                "strategy_name": row.get("strategy_name"),
                "setup_type": row.get("setup_type"),
                "edge_rank_score": row.get("edge_rank_score"),
                "edge_score": row.get("edge_score"),
                "entry_price": round(entry, 4),
                "quantity": qty,
                "entry_date": str(row.get("entry_date") or start_date),
                "entry_time": str(row.get("entry_date") or start_date),
                "sl": sl,
                "tp": tp,
                "peak_price": round(max(entry, last), 4),
                "last_price": round(last, 4),
                "status": "OPEN",
                "entry_source": "baseline_backtest_continuation",
                "source": "baseline_continuation_live",
                "portfolio_max_positions_at_entry": spec.max_positions,
            }
        )

    if not live_holdings:
        return state

    state.update(
        {
            "portfolio_id": spec.portfolio_id,
            "label": f"{spec.label} - Baseline Continuation",
            "strategy_family": spec.strategy_family,
            "initial_capital": initial_nav,
            "cash": round(cash, 2),
            "holdings": live_holdings,
            "closed": [] if reset else state.get("closed", []),
            "watchlist": [] if reset else state.get("watchlist", []),
            "events": [] if reset else state.get("events", []),
            "created_at": str(start_date),
            "continuation_mode": "layered_baseline",
            "continuation_source": str(layered_demo_dir),
            "baseline_realized_pnl": baseline_realized,
            "baseline_closed_trades": baseline_closed_trades,
            "baseline_win_rate_pct": baseline_win_rate,
            "baseline_seeded_at": _now().isoformat(timespec="seconds"),
        }
    )
    _event(
        state,
        "BASELINE_CONTINUATION_SEEDED",
        cash=round(cash, 2),
        holdings=len(live_holdings),
        source=str(layered_demo_dir),
    )
    return state


def _save_state(spec: PipelineSpec, state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now().isoformat(timespec="seconds")
    _state_path(spec).write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _state_path(spec: PipelineSpec) -> Path:
    return STATE_DIR / f"{spec.portfolio_id}.json"


def _event(state: dict[str, Any], message: str, **fields: Any) -> None:
    events = state.setdefault("events", [])
    events.append({"time": _now().isoformat(timespec="seconds"), "message": message, **fields})
    del events[:-250]


def _current_symbols(state: dict[str, Any]) -> set[str]:
    return {str(item.get("symbol", "")).upper() for item in state.get("holdings", [])}


def _build_market_context(symbol: str, edge: dict[str, Any], live_price: float | None) -> dict[str, Any]:
    return {
        "_portfolio_checked": True,
        "exchange": "HOSE",
        "stock_current_price": live_price,
        "stock_day_change_pct": 0.0,
        "avg_vol_20d": None,
        "reference_trend": str(edge.get("mkt_regime_state") or "SIDEWAY"),
        "trend": str(edge.get("mkt_regime_state") or "SIDEWAY"),
        "vni_change_pct": 0.0,
        "edge_strategy_analysis": edge,
        "active_strategies": [],
        "paper_portfolio_id": "shadow",
    }


def _normalize_entry_zone(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        low = float(value[0])
        high = float(value[1])
    except Exception:
        return None
    if low <= 0 or high <= 0 or low > high:
        return None
    return low, high


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except Exception:
        return default


def _fallback_sl_tp(price: float, edge: dict[str, Any]) -> tuple[float, float]:
    risk = edge.get("risk") or {}
    stop = _safe_float(risk.get("stop_loss"), 0.08) or 0.08
    target = _safe_float(risk.get("take_profit"), 0.25) or 0.25
    return round(price * (1.0 - stop), 2), round(price * (1.0 + target), 2)


def _scale_plan_prices(
    *,
    entry_zone: tuple[float, float],
    sl: float,
    tp: float,
    live_price: float | None,
) -> tuple[tuple[float, float], float, float]:
    """Normalize LLM prices that may be expressed in thousands of VND."""
    if not live_price or live_price <= 1000:
        return entry_zone, sl, tp
    if entry_zone[1] < 1000 and live_price / max(entry_zone[1], 1.0) > 100:
        return (
            (entry_zone[0] * 1000.0, entry_zone[1] * 1000.0),
            sl * 1000.0 if sl < 1000 else sl,
            tp * 1000.0 if tp < 1000 else tp,
        )
    return entry_zone, sl, tp


def _refresh_watchlist(
    *,
    spec: PipelineSpec,
    state: dict[str, Any],
    symbols: list[str],
    force: bool,
    once: bool,
) -> None:
    graph_enabled = os.getenv("ALLOW_ANTHROPIC_SHADOW_GRAPH", "").strip() == "1"
    if not graph_enabled:
        once = True
    today = _today()
    if not force and state.get("watchlist_date") == today and state.get("watchlist"):
        return

    print(f"[{spec.portfolio_id}] rebuilding watchlist for {today}")
    edge_signals = get_edge_strategy_signals(
        symbols,
        as_of_date=today,
        strategy_name=spec.strategy_family,
        root=ROOT,
    )
    passed = [
        {**edge, "symbol": symbol}
        for symbol, edge in edge_signals.items()
        if edge.get("passed")
    ]
    passed.sort(key=lambda item: float(item.get("edge_rank_score") or 0.0), reverse=True)
    passed = passed[: spec.max_candidates]
    prices = get_live_price([str(item["symbol"]) for item in passed]) if passed else {}
    held = _current_symbols(state)
    watchlist: list[dict[str, Any]] = []

    for edge in passed:
        symbol = str(edge.get("symbol") or "").upper()
        if not symbol or symbol in held:
            continue
        live_price = prices.get(symbol)
        entry_zone: tuple[float, float] | None = None
        trader: dict[str, Any] = {}
        risk_output: dict[str, Any] = {}
        error = None
        if not once:
            try:
                graph_state = run_trade_graph(
                    symbol=symbol,
                    setup_type=str(edge.get("setup_type") or "EDGE"),
                    market_context=_build_market_context(symbol, edge, live_price),
                    date=today,
                    backtest_mode=False,
                )
                trader = dict(graph_state.get("trader_decision") or {})
                risk_output = dict(graph_state.get("risk_output") or {})
                entry_zone = _normalize_entry_zone(trader.get("entry_zone"))
            except Exception as exc:
                error = str(exc)
        if entry_zone is None and once:
            ref_price = live_price or _safe_float(edge.get("close")) or 0.0
            if ref_price > 0:
                entry_zone = (round(ref_price * 0.995, 2), round(ref_price * 1.01, 2))
                trader = {"action": "GRAPH_DISABLED"}
                risk_output = {
                    "final_action": "NO_GRAPH",
                    "override_reason": "Trader/risk graph skipped by --no-graph; this is watchlist-only.",
                }
        if entry_zone is None:
            _event(state, "WATCHLIST_SKIP_NO_ENTRY", symbol=symbol, error=error)
            continue

        ref_price = live_price or sum(entry_zone) / 2.0
        fallback_sl, fallback_tp = _fallback_sl_tp(ref_price, edge)
        sl = _safe_float(trader.get("stop_loss"), fallback_sl) or fallback_sl
        tp = (
            _safe_float(trader.get("initial_target"))
            or _safe_float(trader.get("take_profit"))
            or fallback_tp
        )
        entry_zone, sl, tp = _scale_plan_prices(
            entry_zone=entry_zone,
            sl=float(sl),
            tp=float(tp),
            live_price=live_price,
        )
        watchlist.append(
            {
                "symbol": symbol,
                "strategy_name": edge.get("strategy_name"),
                "setup_type": edge.get("setup_type"),
                "edge_rank": edge.get("edge_rank"),
                "edge_rank_score": edge.get("edge_rank_score"),
                "edge_score": edge.get("edge_score"),
                "mkt_regime_state": edge.get("mkt_regime_state"),
                "mkt_regime_score": edge.get("mkt_regime_score"),
                "rs_percentile_20": edge.get("rs_percentile_20"),
                "smart_money_score": edge.get("smart_money_score"),
                "value_ratio_20": edge.get("value_ratio_20"),
                "close_location": edge.get("close_location"),
                "above_ma20": edge.get("above_ma20"),
                "above_ma50": edge.get("above_ma50"),
                "distribution_days_10": edge.get("distribution_days_10"),
                "feature_date": edge.get("feature_date"),
                "entry_zone": [round(entry_zone[0], 2), round(entry_zone[1], 2)],
                "stop_loss": round(float(sl), 2),
                "take_profit": round(float(tp), 2),
                "last_price": live_price,
                "trader_action": trader.get("action"),
                "risk_final_action": risk_output.get("final_action"),
                "risk_note": risk_output.get("override_reason"),
                "created_at": _now().isoformat(timespec="seconds"),
            }
        )

    state["watchlist"] = watchlist
    state["watchlist_date"] = today
    _event(state, "WATCHLIST_REFRESH", count=len(watchlist))
    print(f"[{spec.portfolio_id}] watchlist ready: {len(watchlist)}")


def _portfolio_value(state: dict[str, Any], prices: dict[str, float]) -> float:
    value = float(state.get("cash") or 0.0)
    for pos in state.get("holdings", []):
        symbol = str(pos.get("symbol"))
        ref = _safe_float(pos.get("entry_price"), _safe_float(pos.get("last_price")))
        price = (
            _normalize_live_price_to_position_unit(prices.get(symbol), ref)
            or _safe_float(pos.get("last_price"))
            or _safe_float(pos.get("entry_price"))
            or 0.0
        )
        value += price * int(pos.get("quantity") or 0)
    return value


def _pnl_pct(pos: dict[str, Any], price: float | None = None) -> float:
    entry = _safe_float(pos.get("entry_price"), 0.0) or 0.0
    last = price or _safe_float(pos.get("last_price"), entry) or entry
    if entry <= 0:
        return 0.0
    return (last / entry - 1.0) * 100.0


def _holding_bars_approx(pos: dict[str, Any]) -> int:
    try:
        entry = pd.Timestamp(str(pos.get("entry_date"))).normalize()
        return max(0, int((pd.Timestamp(_today()) - entry).days))
    except Exception:
        return 0


def _candidate_score(candidate: dict[str, Any]) -> float:
    return _safe_float(candidate.get("edge_rank_score"), 0.0) or 0.0


def _holding_score(pos: dict[str, Any]) -> float:
    return _safe_float(pos.get("edge_rank_score"), 0.0) or 0.0


def _portfolio_policy(spec: PipelineSpec, state: dict[str, Any]) -> dict[str, Any]:
    watchlist = list(state.get("watchlist", []))
    scores = sorted((_candidate_score(item) for item in watchlist), reverse=True)
    top_scores = scores[: min(8, len(scores))]
    avg_top_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
    regimes = [str(item.get("mkt_regime_state") or "").upper() for item in watchlist]
    risk_off_count = sum(1 for item in regimes if item == "RISK_OFF")
    constructive_count = sum(1 for item in regimes if item in {"RISK_ON_UPTREND", "RISK_ON_RECOVERY", "NEUTRAL_UPTREND", "NEUTRAL"})

    max_positions = spec.max_positions
    expansion_reason = "BASE_MAX_POSITIONS"
    if (
        spec.allow_dynamic_expansion
        and
        spec.max_expanded_positions > spec.max_positions
        and len(watchlist) >= 8
        and avg_top_score >= 2.5
        and constructive_count >= max(4, len(watchlist) // 2)
        and risk_off_count <= max(1, len(watchlist) // 4)
    ):
        max_positions = spec.max_expanded_positions
        expansion_reason = "EXPAND_STRONG_BREADTH_AND_CANDIDATES"

    policy = {
        "base_max_positions": spec.max_positions,
        "effective_max_positions": max_positions,
        "expanded_max_positions": spec.max_expanded_positions,
        "watchlist_count": len(watchlist),
        "avg_top_edge_rank_score": round(avg_top_score, 3),
        "constructive_regime_count": constructive_count,
        "risk_off_count": risk_off_count,
        "reason": expansion_reason,
    }
    state["portfolio_policy"] = policy
    return policy


def _effective_max_positions(spec: PipelineSpec, state: dict[str, Any]) -> int:
    policy = state.get("portfolio_policy") or _portfolio_policy(spec, state)
    return int(policy.get("effective_max_positions") or spec.max_positions)


def _rotation_reason(pos: dict[str, Any], candidate: dict[str, Any], price: float) -> str | None:
    pnl = _pnl_pct(pos, price=_safe_float(pos.get("last_price")))
    peak = _safe_float(pos.get("peak_price"), price) or price
    drawdown_from_peak = (price / peak - 1.0) * 100.0 if peak > 0 else 0.0
    age = _holding_bars_approx(pos)
    score_gap = _candidate_score(candidate) - _holding_score(pos)

    weak_loser = pnl <= -3.0 and score_gap >= 0.75
    stale_sideways = age >= 15 and abs(pnl) <= 2.0 and score_gap >= 1.0
    winner_weakening = pnl >= 8.0 and drawdown_from_peak <= -3.0 and score_gap >= 0.5
    low_quality_vs_candidate = _holding_score(pos) > 0 and score_gap >= 1.5 and pnl <= 4.0

    if weak_loser:
        return "ROTATE_WEAK_LOSER_TO_STRONGER_CANDIDATE"
    if stale_sideways:
        return "ROTATE_STALE_SIDEWAYS_TO_STRONGER_CANDIDATE"
    if winner_weakening:
        return "ROTATE_WINNER_WEAKENING_TO_STRONGER_CANDIDATE"
    if low_quality_vs_candidate:
        return "ROTATE_LOWER_SCORE_TO_STRONGER_CANDIDATE"
    return None


def _maybe_free_slot_for_candidate(
    *,
    spec: PipelineSpec,
    state: dict[str, Any],
    candidate: dict[str, Any],
    prices: dict[str, float],
) -> bool:
    if not spec.allow_rotation:
        return False
    if len(state.get("holdings", [])) < _effective_max_positions(spec, state):
        return True
    candidate_score = _candidate_score(candidate)
    if candidate_score < 2.5:
        return False

    choices: list[tuple[float, dict[str, Any], str]] = []
    for pos in state.get("holdings", []):
        symbol = str(pos.get("symbol"))
        price = prices.get(symbol) or _safe_float(pos.get("last_price")) or _safe_float(pos.get("entry_price"))
        if not price:
            continue
        reason = _rotation_reason(pos, candidate, float(price))
        if not reason:
            continue
        priority = candidate_score - _holding_score(pos)
        priority += max(0.0, -_pnl_pct(pos, float(price)) / 10.0)
        choices.append((priority, pos, reason))
    if not choices:
        return False

    choices.sort(key=lambda item: item[0], reverse=True)
    _priority, pos, reason = choices[0]
    symbol = str(pos.get("symbol"))
    exit_price = prices.get(symbol) or _safe_float(pos.get("last_price")) or _safe_float(pos.get("entry_price"))
    if not exit_price:
        return False
    _close_position(state=state, pos=pos, price=float(exit_price), reason=reason)
    _event(
        state,
        "ROTATION_SLOT_FREED",
        symbol=symbol,
        replacement=candidate.get("symbol"),
        reason=reason,
        old_score=round(_holding_score(pos), 3),
        new_score=round(candidate_score, 3),
    )
    return True


def _open_position(
    *,
    spec: PipelineSpec,
    state: dict[str, Any],
    candidate: dict[str, Any],
    price: float,
) -> bool:
    holdings = state.setdefault("holdings", [])
    max_positions = _effective_max_positions(spec, state)
    if len(holdings) >= max_positions:
        return False
    if str(candidate.get("symbol")).upper() in _current_symbols(state):
        return False
    cash = float(state.get("cash") or 0.0)
    slots = max(1, max_positions - len(holdings))
    budget = cash / slots
    quantity = int((budget // price) // 100 * 100)
    if quantity <= 0:
        quantity = int(budget // price)
    if quantity <= 0:
        return False
    cost = quantity * price
    if cost > cash:
        return False
    state["cash"] = round(cash - cost, 2)
    pos = {
        "symbol": str(candidate["symbol"]).upper(),
        "strategy_name": candidate.get("strategy_name"),
        "setup_type": candidate.get("setup_type"),
        "edge_rank_score": candidate.get("edge_rank_score"),
        "edge_score": candidate.get("edge_score"),
        "mkt_regime_state": candidate.get("mkt_regime_state"),
        "entry_price": round(price, 2),
        "quantity": quantity,
        "entry_date": _today(),
        "entry_time": _now().isoformat(timespec="seconds"),
        "sl": _safe_float(candidate.get("stop_loss")),
        "tp": _safe_float(candidate.get("take_profit")),
        "peak_price": round(price, 2),
        "last_price": round(price, 2),
        "status": "OPEN",
        "entry_source": "realtime_shadow_trigger",
        "portfolio_max_positions_at_entry": max_positions,
    }
    holdings.append(pos)
    _event(state, "OPEN", symbol=pos["symbol"], price=price, quantity=quantity, strategy=pos["strategy_name"])
    return True


def _close_position(
    *,
    state: dict[str, Any],
    pos: dict[str, Any],
    price: float,
    reason: str,
) -> None:
    holdings = state.setdefault("holdings", [])
    quantity = int(pos.get("quantity") or 0)
    entry = float(pos.get("entry_price") or 0.0)
    proceeds = quantity * price
    pnl = (price - entry) * quantity
    pnl_pct = ((price / entry) - 1.0) * 100.0 if entry > 0 else 0.0
    state["cash"] = round(float(state.get("cash") or 0.0) + proceeds, 2)
    closed = {
        **pos,
        "exit_price": round(price, 2),
        "exit_date": _today(),
        "exit_time": _now().isoformat(timespec="seconds"),
        "exit_reason": reason,
        "realized_pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "status": "CLOSED",
    }
    state.setdefault("closed", []).append(closed)
    holdings[:] = [item for item in holdings if item is not pos]
    _event(state, "CLOSE", symbol=pos.get("symbol"), price=price, reason=reason, pnl=round(pnl, 2))


def _scan_entries_and_exits(spec: PipelineSpec, state: dict[str, Any]) -> None:
    _portfolio_policy(spec, state)
    watch_symbols = [str(item["symbol"]) for item in state.get("watchlist", [])]
    holding_symbols = [str(item["symbol"]) for item in state.get("holdings", [])]
    symbols = sorted(set(watch_symbols + holding_symbols))
    prices = get_live_price(symbols) if symbols else {}
    now_iso = _now().isoformat(timespec="seconds")

    for pos in list(state.get("holdings", [])):
        symbol = str(pos.get("symbol"))
        entry_ref = _safe_float(pos.get("entry_price"), _safe_float(pos.get("last_price")))
        price = _normalize_live_price_to_position_unit(prices.get(symbol), entry_ref)
        if not price:
            continue
        pos["last_price"] = round(price, 2)
        pos["last_checked"] = now_iso
        pos["peak_price"] = max(float(pos.get("peak_price") or price), price)
        new_sl = compute_profit_trailing_sl(
            entry_price=float(pos.get("entry_price") or 0.0),
            current_price=price,
            old_sl=_safe_float(pos.get("sl")),
            initial_target=_safe_float(pos.get("tp")),
            peak_price=_safe_float(pos.get("peak_price")),
        )
        if new_sl:
            pos["sl"] = round(float(new_sl), 2)
            _event(state, "RAISE_SL", symbol=symbol, new_sl=pos["sl"], price=round(price, 2))
        sl = _safe_float(pos.get("sl"))
        tp = _safe_float(pos.get("tp"))
        if sl and price <= sl:
            _close_position(state=state, pos=pos, price=price, reason="SL_HIT")
        elif tp and price >= tp:
            _close_position(state=state, pos=pos, price=price, reason="TP_HIT")

    held = _current_symbols(state)
    remaining_watchlist = []
    for candidate in state.get("watchlist", []):
        symbol = str(candidate.get("symbol"))
        zone = _normalize_entry_zone(candidate.get("entry_zone"))
        ref_price = ((zone[0] + zone[1]) / 2.0) if zone else _safe_float(candidate.get("last_price"))
        price = _normalize_live_price_to_position_unit(prices.get(symbol), ref_price)
        if price:
            candidate["last_price"] = round(price, 2)
            candidate["last_checked"] = now_iso
        if not price or not zone or symbol in held:
            remaining_watchlist.append(candidate)
            continue
        if zone[0] <= price <= zone[1]:
            final_action = str(candidate.get("risk_final_action") or candidate.get("trader_action") or "").upper()
            if final_action != "MUA":
                if candidate.get("last_reject_price") != round(price, 2):
                    candidate["last_reject_price"] = round(price, 2)
                    candidate["last_reject_time"] = now_iso
                    _event(
                        state,
                        "TRIGGER_NO_BUY_APPROVAL",
                        symbol=symbol,
                        price=round(price, 2),
                        reason=final_action or "NO_ACTION",
                    )
                remaining_watchlist.append(candidate)
                continue
            if len(state.get("holdings", [])) >= _effective_max_positions(spec, state):
                if not _maybe_free_slot_for_candidate(
                    spec=spec,
                    state=state,
                    candidate=candidate,
                    prices=prices,
                ):
                    if candidate.get("last_no_slot_price") != round(price, 2):
                        candidate["last_no_slot_price"] = round(price, 2)
                        candidate["last_no_slot_time"] = now_iso
                        _event(
                            state,
                            "TRIGGER_NO_SLOT_NO_ROTATION",
                            symbol=symbol,
                            price=round(price, 2),
                            score=round(_candidate_score(candidate), 3),
                        )
                    remaining_watchlist.append(candidate)
                    continue
                held = _current_symbols(state)
            if _open_position(spec=spec, state=state, candidate=candidate, price=price):
                held.add(symbol)
                continue
        remaining_watchlist.append(candidate)
    state["watchlist"] = remaining_watchlist


def _update_stats(state: dict[str, Any]) -> None:
    symbols = [str(item.get("symbol")) for item in state.get("holdings", [])]
    prices = get_live_price(symbols) if symbols else {}
    equity = _portfolio_value(state, prices)
    initial = float(state.get("initial_capital") or equity or 1.0)
    closed = state.get("closed", [])
    wins = [item for item in closed if float(item.get("realized_pnl") or 0.0) > 0]
    baseline_realized = float(state.get("baseline_realized_pnl") or 0.0)
    baseline_closed = int(float(state.get("baseline_closed_trades") or 0.0))
    baseline_win_rate = _safe_float(state.get("baseline_win_rate_pct"))
    baseline_wins = int(round(baseline_closed * baseline_win_rate / 100.0)) if baseline_win_rate is not None else 0
    total_closed = baseline_closed + len(closed)
    total_wins = baseline_wins + len(wins)
    realized_total = baseline_realized + sum(float(item.get("realized_pnl") or 0.0) for item in closed)
    state["stats"] = {
        "equity": round(equity, 2),
        "cash": round(float(state.get("cash") or 0.0), 2),
        "return_pct": round((equity / initial - 1.0) * 100.0, 2) if initial else 0.0,
        "open_positions": len(state.get("holdings", [])),
        "watchlist": len(state.get("watchlist", [])),
        "closed_trades": total_closed,
        "win_rate_pct": round(total_wins / total_closed * 100.0, 2) if total_closed else None,
        "realized_pnl": round(realized_total, 2),
        "effective_max_positions": (state.get("portfolio_policy") or {}).get("effective_max_positions"),
        "portfolio_policy_reason": (state.get("portfolio_policy") or {}).get("reason"),
    }


def _fmt_money(value: Any) -> str:
    num = _safe_float(value)
    if num is None:
        return ""
    return f"{num:,.0f}"


def _fmt_pct(value: Any) -> str:
    num = _safe_float(value)
    if num is None:
        return ""
    return f"{num:.2f}%"


def _render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p class='muted'>No rows.</p>"
    head = "".join(f"<th>{escape(label)}</th>" for key, label in columns)
    body = []
    for row in rows:
        cells = []
        for key, _label in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:,.2f}"
            cells.append(f"<td>{escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _symbol_detail_path(symbol: Any) -> str:
    safe = "".join(ch for ch in str(symbol or "").upper().strip() if ch.isalnum() or ch in ("_", "-"))
    return f"symbols/{safe}.html" if safe else ""


def _link_symbol(value: Any) -> str:
    symbol = str(value or "").upper().strip()
    href = _symbol_detail_path(symbol)
    if not symbol or not href:
        return escape(symbol)
    return f'<a href="{escape(href)}">{escape(symbol)}</a>'


def _link_symbol_column(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns:
        return frame
    display = frame.copy()
    display["symbol"] = display["symbol"].map(_link_symbol)
    return display


def _rows_to_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame[columns]


def _write_layer_table(out_dir: Path, stem: str, title: str, note: str, frame: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    display = frame.copy()
    if display.empty:
        display = pd.DataFrame(
            [
                {
                    "status": "NO_ROWS",
                    "note": note,
                    "updated_at": _now().isoformat(timespec="seconds"),
                }
            ]
        )
    display.to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    html_display = _link_symbol_column(display)
    html = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="15">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{ --ink: #14213d; --muted: #5f6b7a; --line: #d9dee7; --bg: #f5f1e8; --panel: #fffaf0; }}
    body {{ margin: 0; padding: 28px; font-family: Georgia, 'Times New Roman', serif; background: var(--bg); color: var(--ink); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .note {{ margin: 0 0 18px; color: var(--muted); font-family: Segoe UI, Arial, sans-serif; }}
    .wrap {{ overflow: auto; background: white; border: 1px solid var(--line); box-shadow: 0 14px 36px rgba(20,33,61,.12); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; white-space: nowrap; font-family: Segoe UI, Arial, sans-serif; }}
    th {{ position: sticky; top: 0; background: var(--ink); color: white; text-align: left; cursor: pointer; user-select: none; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf0f5; }}
    tr:nth-child(even) {{ background: #fbfbf8; }}
    a {{ color: var(--ink); }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="note"><a href="index.html">Index</a> | {escape(note)}</p>
  <div class="wrap">{html_display.to_html(index=False, classes="data-table", border=0, escape=False)}</div>
</body>
</html>
"""
    (out_dir / f"{stem}.html").write_text(html, encoding="utf-8")


def _link_symbols_in_table_html(html: str, out_dir: Path) -> str:
    try:
        import re

        symbols: set[str] = set()
        for csv_name in ("08_watchlist.csv", "06c_open_holdings.csv", "04_ranking_signals.csv", "05_risk_gate_actions.csv"):
            frame = _read_layer_csv(out_dir / csv_name)
            if not frame.empty and "symbol" in frame.columns:
                symbols.update(frame["symbol"].dropna().astype(str).str.upper().str.strip().tolist())
        for symbol in sorted(symbols, key=len, reverse=True):
            if not symbol:
                continue
            href = _symbol_detail_path(symbol)
            html = re.sub(
                rf"(<td[^>]*>)({re.escape(symbol)})(</td>)",
                rf'\1<a href="{href}">\2</a>\3',
                html,
            )
        return html
    except Exception:
        return html


def _read_layer_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 5:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _detail_section(title: str, frame: pd.DataFrame, *, columns: list[str] | None = None) -> str:
    if frame.empty:
        return f"<h2>{escape(title)}</h2><p class='note'>No rows.</p>"
    display = frame.copy()
    if columns:
        keep = [column for column in columns if column in display.columns]
        display = display[keep] if keep else display
    return f"<h2>{escape(title)}</h2><div class='wrap'>{display.to_html(index=False, border=0, escape=True)}</div>"


def _symbol_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns:
        return pd.DataFrame()
    return frame[frame["symbol"].astype(str).str.upper().eq(symbol)].copy()


def _load_symbol_ohlcv_from_master(symbol: str, days: int = 30) -> pd.DataFrame:
    path = DATA_DIR / "ohlcv_master.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_parquet(path)
        if "symbol" not in frame.columns or "date" not in frame.columns:
            return pd.DataFrame()
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        rows = frame[frame["symbol"].eq(symbol)].sort_values("date").tail(days).copy()
        return rows.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _holding_rationale_frame(symbol: str, holding: pd.DataFrame) -> pd.DataFrame:
    if holding.empty:
        return pd.DataFrame()
    row = holding.iloc[0].to_dict()
    rows = [
        {"field": "status", "value": "OPEN_HOLDING"},
        {"field": "why_in_system", "value": "Position is still open in the baseline/backtest portfolio and is being marked to market with live prices."},
        {"field": "strategy_name", "value": row.get("strategy_name", "")},
        {"field": "setup_type", "value": row.get("setup_type", "")},
        {"field": "signal_date", "value": row.get("signal_date", "")},
        {"field": "entry_date", "value": row.get("entry_date", "")},
        {"field": "entry_price", "value": row.get("entry_price", "")},
        {"field": "last_live_price", "value": row.get("last_close", "")},
        {"field": "edge_score", "value": row.get("edge_score", "")},
        {"field": "edge_rank_score", "value": row.get("edge_rank_score", "")},
        {"field": "planned_exit_reason", "value": row.get("planned_exit_reason", "")},
        {"field": "source", "value": row.get("source", "")},
    ]
    return pd.DataFrame(rows)


def _symbol_value(frame: pd.DataFrame, symbol: str, column: str, default: Any = "") -> Any:
    if frame.empty or "symbol" not in frame.columns or column not in frame.columns:
        return default
    rows = frame[frame["symbol"].astype(str).str.upper().eq(symbol)]
    if rows.empty:
        return default
    value = rows.iloc[0].get(column, default)
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return default if pd.isna(value) else value
    except Exception:
        return value


def _write_symbol_detail_pages(
    *,
    out_dir: Path,
    raw_ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    ranking: pd.DataFrame,
    actions: pd.DataFrame,
    holdings: pd.DataFrame,
    closed: pd.DataFrame,
    watchlist: pd.DataFrame,
    exit_advisory: pd.DataFrame | None = None,
) -> None:
    symbols_dir = out_dir / "symbols"
    symbols_dir.mkdir(parents=True, exist_ok=True)
    exit_advisory = exit_advisory if exit_advisory is not None else pd.DataFrame()
    frames = [raw_ohlcv, features, ranking, actions, holdings, closed, watchlist, exit_advisory]
    symbols: set[str] = set()
    for frame in frames:
        if not frame.empty and "symbol" in frame.columns:
            symbols.update(frame["symbol"].dropna().astype(str).str.upper().str.strip().tolist())

    for symbol in sorted(symbols):
        if not symbol:
            continue
        raw_s = _symbol_frame(raw_ohlcv, symbol)
        features_s = _symbol_frame(features, symbol)
        ranking_s = _symbol_frame(ranking, symbol)
        actions_s = _symbol_frame(actions, symbol)
        holdings_s = _symbol_frame(holdings, symbol)
        closed_s = _symbol_frame(closed, symbol)
        watch_s = _symbol_frame(watchlist, symbol)
        exit_s = _symbol_frame(exit_advisory, symbol)
        if raw_s.empty:
            raw_s = _load_symbol_ohlcv_from_master(symbol)
        holding_rationale = _holding_rationale_frame(symbol, holdings_s)

        action = _symbol_value(actions_s, symbol, "action", _symbol_value(watch_s, symbol, "watch_action", ""))
        strategy = _symbol_value(watch_s, symbol, "strategy_name", _symbol_value(holdings_s, symbol, "strategy_name", ""))
        score = _symbol_value(watch_s, symbol, "edge_rank_score", _symbol_value(ranking_s, symbol, "edge_rank_score", _symbol_value(holdings_s, symbol, "edge_rank_score", "")))
        entry = _symbol_value(watch_s, symbol, "entry_zone", _symbol_value(holdings_s, symbol, "entry_price", ""))
        sl = _symbol_value(watch_s, symbol, "stop_loss", _symbol_value(actions_s, symbol, "sl", _symbol_value(holdings_s, symbol, "sl", "N/A")))
        tp = _symbol_value(watch_s, symbol, "take_profit", _symbol_value(actions_s, symbol, "tp", _symbol_value(holdings_s, symbol, "planned_exit_reason", "N/A")))
        risk_action = _symbol_value(watch_s, symbol, "risk_final_action", _symbol_value(actions_s, symbol, "risk_final_action", ""))
        trader_action = _symbol_value(watch_s, symbol, "trader_action", "")
        note = _symbol_value(
            watch_s,
            symbol,
            "risk_note",
            _symbol_value(actions_s, symbol, "note", _symbol_value(holdings_s, symbol, "planned_exit_reason", "")),
        )

        cards = f"""
<div class="cards">
  <div class="metric"><span>Action</span><strong>{escape(str(action))}</strong></div>
  <div class="metric"><span>Strategy</span><strong>{escape(str(strategy))}</strong></div>
  <div class="metric"><span>Score</span><strong>{escape(str(score))}</strong></div>
  <div class="metric"><span>Entry Zone</span><strong>{escape(str(entry))}</strong></div>
  <div class="metric"><span>SL</span><strong>{escape(str(sl))}</strong></div>
  <div class="metric"><span>TP</span><strong>{escape(str(tp))}</strong></div>
  <div class="metric"><span>Trader</span><strong>{escape(str(trader_action))}</strong></div>
  <div class="metric"><span>Risk Gate</span><strong>{escape(str(risk_action))}</strong></div>
</div>
"""
        sections = "\n".join(
            [
                _detail_section(
                    "Why This Symbol Is Here",
                    watch_s if not watch_s.empty else holding_rationale,
                    columns=[
                        "watch_action", "symbol", "strategy_name", "setup_type", "edge_rank", "edge_rank_score",
                        "feature_date", "entry_zone", "last_price", "stop_loss", "take_profit",
                        "trader_action", "risk_final_action", "risk_note",
                        "field", "value",
                    ],
                ),
                _detail_section(
                    "Ranking / Confluence Features",
                    ranking_s if not ranking_s.empty else features_s if not features_s.empty else holdings_s,
                    columns=[
                        "symbol", "strategy_name", "setup_type", "edge_rank", "edge_rank_score",
                        "edge_score", "mkt_regime_state", "mkt_regime_score", "rs_percentile_20",
                        "smart_money_score", "value_ratio_20", "close_location", "above_ma20",
                        "above_ma50", "distribution_days_10", "feature_date",
                        "signal_date", "entry_date", "planned_exit_reason",
                    ],
                ),
                _detail_section(
                    "Portfolio / PnL",
                    holdings_s,
                    columns=[
                        "symbol", "strategy_name", "entry_date", "entry_price", "shares", "last_close",
                        "market_value", "unrealized_pnl", "unrealized_pnl_pct", "sl", "tp",
                        "exit_agent_action", "weakness_score", "exit_agent_reason", "shadow_exit_price",
                        "price_source", "priced_at", "source",
                    ],
                ),
                _detail_section(
                    "Exit Agent Advisory",
                    exit_s,
                    columns=[
                        "symbol", "exit_agent_action", "weakness_score", "exit_agent_reason",
                        "pnl_pct", "drawdown_from_peak_pct", "distance_to_sl_pct", "distance_to_tp_pct",
                        "replacement_pressure", "shadow_exit_price", "shadow_realized_pnl",
                        "shadow_only", "promotion_guard", "updated_at",
                    ],
                ),
                _detail_section(
                    "Risk Gate / Action State",
                    actions_s,
                    columns=[
                        "rank", "symbol", "action", "strategy_name", "edge_rank_score", "entry_zone",
                        "last_price", "sl", "tp", "risk_final_action", "portfolio_policy", "note",
                    ],
                ),
                _detail_section(
                    "Closed Trade History",
                    closed_s,
                    columns=[
                        "symbol", "strategy_name", "entry_date", "exit_date", "entry_price", "exit_price",
                        "shares", "quantity", "pnl", "realized_pnl", "pnl_pct", "exit_reason", "source",
                    ],
                ),
                _detail_section(
                    "Recent OHLCV",
                    raw_s.tail(20),
                    columns=["date", "time", "symbol", "open", "high", "low", "close", "volume", "value"],
                ),
            ]
        )
        html = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="15">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(symbol)} Detail</title>
  <style>
    :root {{ --ink: #14213d; --muted: #5f6b7a; --line: #d9dee7; --bg: #f5f1e8; --panel: #fffaf0; }}
    body {{ margin: 0; padding: 28px; font-family: Georgia, 'Times New Roman', serif; background: var(--bg); color: var(--ink); }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 26px 0 12px; font-size: 20px; }}
    .note {{ margin: 0 0 18px; color: var(--muted); font-family: Segoe UI, Arial, sans-serif; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 18px 0 24px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; box-shadow: 0 10px 26px rgba(20,33,61,.08); }}
    .metric span {{ display: block; color: var(--muted); font-family: Segoe UI, Arial, sans-serif; font-size: 12px; }}
    .metric strong {{ display: block; font-size: 16px; margin-top: 6px; overflow-wrap: anywhere; }}
    .wrap {{ overflow: auto; background: white; border: 1px solid var(--line); box-shadow: 0 14px 36px rgba(20,33,61,.12); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; white-space: nowrap; font-family: Segoe UI, Arial, sans-serif; }}
    th {{ position: sticky; top: 0; background: var(--ink); color: white; text-align: left; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf0f5; }}
    tr:nth-child(even) {{ background: #fbfbf8; }}
    a {{ color: var(--ink); }}
  </style>
</head>
<body>
  <h1>{escape(symbol)} Detail</h1>
  <p class="note"><a href="../index.html">Index</a> | <a href="../08_watchlist.html">Watchlist</a> | <a href="../06c_open_holdings.html">Open Holdings</a></p>
  <p class="note">{escape(str(note))}</p>
  {cards}
  {sections}
</body>
</html>
"""
        (symbols_dir / f"{symbol}.html").write_text(html, encoding="utf-8")


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(str(value).replace(",", "").replace("%", "").strip())
    except Exception:
        return default


def _summary_lookup(summary: pd.DataFrame, metric: str, default: float = 0.0) -> float:
    if summary.empty or "metric" not in summary.columns or "value" not in summary.columns:
        return default
    rows = summary[summary["metric"].astype(str) == metric]
    if rows.empty:
        return default
    return _numeric(rows.iloc[-1]["value"], default)


def _set_summary_metric(summary: pd.DataFrame, metric: str, value: Any) -> pd.DataFrame:
    if summary.empty or "metric" not in summary.columns or "value" not in summary.columns:
        return pd.DataFrame([{"metric": metric, "value": value}])
    mask = summary["metric"].astype(str) == metric
    if mask.any():
        summary.loc[mask, "value"] = value
        return summary
    return pd.concat([summary, pd.DataFrame([{"metric": metric, "value": value}])], ignore_index=True)


def _normalize_live_price_to_position_unit(live_price: float | None, reference_price: float | None) -> float | None:
    if live_price is None or live_price <= 0:
        return None
    if reference_price is None or reference_price <= 0:
        return float(live_price)

    price = float(live_price)
    ref = float(reference_price)
    if price / ref > 100.0:
        return price / 1000.0
    if ref / price > 100.0:
        return price * 1000.0
    return price


def _mark_holdings_to_live(holdings: pd.DataFrame) -> tuple[pd.DataFrame, int, str]:
    if holdings.empty or "symbol" not in holdings.columns:
        return holdings, 0, ""

    refreshed = holdings.copy()
    symbols = refreshed["symbol"].dropna().astype(str).str.upper().str.strip().unique().tolist()
    if not symbols:
        return refreshed, 0, ""

    prices = get_live_price(symbols)
    priced_at = _now().isoformat(timespec="seconds")
    updated = 0
    for idx, row in refreshed.iterrows():
        symbol = str(row.get("symbol") or "").upper().strip()
        entry_price = _numeric(row.get("entry_price"), 0.0)
        shares = _numeric(row.get("shares"), _numeric(row.get("quantity"), 0.0))
        raw_live_price = prices.get(symbol)
        live_price = _normalize_live_price_to_position_unit(raw_live_price, entry_price)
        fallback_price = _numeric(row.get("last_close"), _numeric(row.get("last_price"), 0.0))
        price = live_price if live_price and live_price > 0 else fallback_price

        refreshed.at[idx, "price_source"] = "live" if live_price and live_price > 0 else "snapshot"
        refreshed.at[idx, "raw_live_price"] = round(float(raw_live_price), 4) if raw_live_price else None
        refreshed.at[idx, "priced_at"] = priced_at if live_price and live_price > 0 else str(row.get("priced_at") or "")
        if price <= 0 or entry_price <= 0 or shares <= 0:
            continue

        market_value = shares * price
        unrealized_pnl = (price - entry_price) * shares
        refreshed.at[idx, "last_close"] = round(price, 4)
        refreshed.at[idx, "market_value"] = round(market_value, 0)
        refreshed.at[idx, "unrealized_pnl"] = round(unrealized_pnl, 0)
        refreshed.at[idx, "unrealized_pnl_pct"] = round((price / entry_price - 1.0) * 100.0, 2)
        if live_price and live_price > 0:
            updated += 1

    return refreshed, updated, priced_at


def _update_summary_from_holdings(summary: pd.DataFrame, holdings: pd.DataFrame, updated: int, priced_at: str) -> pd.DataFrame:
    if holdings.empty:
        return summary

    market_value = sum(_numeric(v) for v in holdings.get("market_value", []))
    open_pnl = sum(_numeric(v) for v in holdings.get("unrealized_pnl", []))
    entry_value = market_value - open_pnl
    cash = _summary_lookup(summary, "Cash")
    initial_nav = _summary_lookup(summary, "Initial NAV", 100_000_000.0)
    max_positions = int(_summary_lookup(summary, "Max positions", 5))
    current_nav = cash + market_value

    summary = _set_summary_metric(summary, "Current NAV", round(current_nav, 0))
    summary = _set_summary_metric(summary, "Return", round((current_nav / initial_nav - 1.0) * 100.0, 2) if initial_nav else None)
    summary = _set_summary_metric(summary, "Cash / NAV", round(cash / current_nav * 100.0, 2) if current_nav else None)
    summary = _set_summary_metric(summary, "Open positions", len(holdings))
    summary = _set_summary_metric(summary, "Available slots", max(0, max_positions - len(holdings)))
    summary = _set_summary_metric(summary, "Open holdings value", round(market_value, 0))
    summary = _set_summary_metric(summary, "Open unrealized PnL", round(open_pnl, 0))
    summary = _set_summary_metric(summary, "Open unrealized PnL %", round(open_pnl / entry_value * 100.0, 2) if entry_value else None)
    summary = _set_summary_metric(summary, "Price updated at", priced_at)
    summary = _set_summary_metric(summary, "Live prices updated", updated)
    return summary


def _merge_realtime_portfolio_pages(
    *,
    out_dir: Path,
    summary: pd.DataFrame,
    holdings_frame: pd.DataFrame,
    closed_frame: pd.DataFrame,
    backtest_dir: Path | None = None,
    continuation_mode: bool = False,
) -> None:
    """Merge realtime shadow rows into the baseline 06a/06b/06c pages."""
    from scripts.export_pipeline_layer_demo import BASELINE_BACKTEST_DIR, _portfolio_overview_page
    from scripts.update_demo_open_holdings_live import (
        _closed_trades_page_with_totals,
        _open_holdings_page_with_totals,
        _with_meta_refresh,
    )

    backtest_dir = backtest_dir or BASELINE_BACKTEST_DIR
    base_summary = _read_layer_csv(out_dir / "06a_portfolio_overview.csv")
    base_holdings = _read_layer_csv(out_dir / "06c_open_holdings.csv")
    base_closed = _read_layer_csv(out_dir / "06b_closed_trades.csv")

    if continuation_mode:
        base_holdings = pd.DataFrame()
    elif base_holdings.empty:
        base_holdings = pd.DataFrame()
    elif "source" in base_holdings.columns:
        base_holdings = base_holdings[base_holdings["source"].astype(str) != "realtime_shadow"].copy()
    else:
        base_holdings = base_holdings.copy()
        base_holdings["source"] = "baseline_backtest"

    if not continuation_mode:
        base_holdings, base_updated, base_priced_at = _mark_holdings_to_live(base_holdings)
        base_summary = _update_summary_from_holdings(base_summary, base_holdings, base_updated, base_priced_at)

    realtime_holdings = holdings_frame.copy()
    if not realtime_holdings.empty:
        realtime_holdings["source"] = "realtime_shadow"
        realtime_holdings["planned_exit_reason"] = "LIVE_SHADOW_SL_TP"
    combined_holdings = pd.concat([base_holdings, realtime_holdings], ignore_index=True, sort=False)

    if base_closed.empty:
        base_closed = pd.DataFrame()
    elif "source" in base_closed.columns:
        base_closed = base_closed[base_closed["source"].astype(str) != "realtime_shadow"].copy()
    else:
        base_closed = base_closed.copy()
        base_closed["source"] = "baseline_backtest"

    realtime_closed = closed_frame.copy()
    if not realtime_closed.empty:
        realtime_closed["source"] = "realtime_shadow"
        if "realized_pnl" in realtime_closed.columns and "pnl" not in realtime_closed.columns:
            realtime_closed["pnl"] = realtime_closed["realized_pnl"]
        if "quantity" in realtime_closed.columns and "shares" not in realtime_closed.columns:
            realtime_closed["shares"] = realtime_closed["quantity"]
    combined_closed = pd.concat([base_closed, realtime_closed], ignore_index=True, sort=False)

    display_summary = summary.copy() if continuation_mode else (base_summary.copy() if not base_summary.empty else summary.copy())
    baseline_open_pnl = _summary_lookup(display_summary, "Open unrealized PnL")
    baseline_realized = _summary_lookup(display_summary, "Closed realized PnL")
    baseline_open_positions = _summary_lookup(display_summary, "Open positions")
    realtime_open_pnl = _summary_lookup(summary, "Open unrealized PnL")
    realtime_realized = _summary_lookup(summary, "Closed realized PnL")
    realtime_cash = _summary_lookup(summary, "Cash")
    realtime_open_positions = _summary_lookup(summary, "Open positions")

    if continuation_mode:
        display_summary = _set_summary_metric(display_summary, "Continuation source", "baseline_backtest_live")
    else:
        display_summary = _set_summary_metric(display_summary, "Realtime shadow cash", round(realtime_cash, 0))
        display_summary = _set_summary_metric(display_summary, "Realtime shadow open positions", int(realtime_open_positions))
        display_summary = _set_summary_metric(display_summary, "Realtime shadow open unrealized PnL", round(realtime_open_pnl, 0))
        display_summary = _set_summary_metric(display_summary, "Realtime shadow realized PnL", round(realtime_realized, 0))
        display_summary = _set_summary_metric(display_summary, "Combined open positions", int(baseline_open_positions + realtime_open_positions))
        display_summary = _set_summary_metric(display_summary, "Combined open unrealized PnL", round(baseline_open_pnl + realtime_open_pnl, 0))
        display_summary = _set_summary_metric(display_summary, "Combined realized + open PnL", round(baseline_realized + baseline_open_pnl + realtime_realized + realtime_open_pnl, 0))
    display_summary = _set_summary_metric(display_summary, "Watchlist", _summary_lookup(summary, "Watchlist"))
    display_summary = _set_summary_metric(display_summary, "Realtime updated", _now().isoformat(timespec="seconds"))

    display_summary.to_csv(out_dir / "06a_portfolio_overview.csv", index=False, encoding="utf-8-sig")
    combined_holdings.to_csv(out_dir / "06c_open_holdings.csv", index=False, encoding="utf-8-sig")
    combined_closed.to_csv(out_dir / "06b_closed_trades.csv", index=False, encoding="utf-8-sig")

    (out_dir / "06a_portfolio_overview.html").write_text(
        _with_meta_refresh(_portfolio_overview_page(display_summary, backtest_dir), 15),
        encoding="utf-8",
    )
    holdings_html = _link_symbols_in_table_html(_open_holdings_page_with_totals(combined_holdings, 15), out_dir)
    closed_html = _link_symbols_in_table_html(_closed_trades_page_with_totals(combined_closed, 15), out_dir)
    (out_dir / "06c_open_holdings.html").write_text(holdings_html, encoding="utf-8")
    (out_dir / "06b_closed_trades.html").write_text(closed_html, encoding="utf-8")


def _layered_index_html(spec: PipelineSpec, state: dict[str, Any], symbols: list[str]) -> str:
    stats = state.get("stats", {})
    updated = str(state.get("updated_at") or "")
    layers = [
        ("01 Raw OHLCV", "01_raw_ohlcv.html", "Dữ liệu giá/khối lượng gần nhất cho watchlist và holdings."),
        ("01 VNINDEX", "01_vnindex.html", "Context thị trường gần nhất dùng để quan sát regime."),
        ("02 Feature Engine", "02_feature_engine.html", "Feature/signal snapshot từ Core MVP9 shadow watchlist."),
        ("03 Strategy Pool", "03_strategy_pool.html", "9 production strategies trong Core MVP."),
        ("04 Ranking", "04_ranking_signals.html", "Candidates realtime sau strategy filters và ranking."),
        ("05 Shared Pool Actions", "05_risk_gate_actions.html", "Action hiện tại: paper-buy/watchlist/holding/closed."),
        ("06a Portfolio Overview", "06a_portfolio_overview.html", "NAV, cash, slots, realized/unrealized PnL."),
        ("06b Closed Trades", "06b_closed_trades.html", "Các vị thế paper đã đóng do SL/TP hoặc lý do khác."),
        ("06c Open Holdings", "06c_open_holdings.html", "Các vị thế paper đang mở, SL/TP và PnL realtime."),
        ("07 Production Status", "07_production_status.html", "Trạng thái runner, safety controls và chế độ paper."),
    ]
    layers.extend(
        [
            ("08 Watchlist", "08_watchlist.html", "Realtime candidates waiting for entry/risk approval."),
            ("09 Exit Agent", "09_exit_agent_advisory.html", "Advisory + shadow exit agent; no discretionary selling rights."),
        ]
    )
    cards = "\n".join(
        f"""<a class="card" href="{href}">
  <strong>{escape(title)}</strong>
  <span>{escape(desc)}</span>
</a>"""
        for title, href, desc in layers
    )
    flow = """OHLCV + VNINDEX
        ↓
Feature Engine
RS / Smart Money / CHDM / DS / Volume / Regime
        ↓
Strategy Pool
9 production Core MVP strategies
        ↓
Ranking Engine
Chọn best candidates realtime
        ↓
Shared Pool Portfolio
Max 5 slots / Paper-buy / Watchlist / Exit Agent
        ↓
Paper Portfolio State
NAV / Cash / Holdings / Closed / SL / TP
        ↓
Realtime Shadow Run
Cập nhật HTML mỗi 15 giây"""
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="15">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trading System Layer Demo</title>
  <style>
    :root {{ --ink: #10243e; --gold: #b8872f; --paper: #f5f1e8; --card: #fffaf0; --muted: #667085; }}
    body {{ margin: 0; min-height: 100vh; background: radial-gradient(circle at top left, rgba(184,135,47,.24), transparent 32rem), linear-gradient(135deg, #f7f0df 0%, #eef3f6 100%); color: var(--ink); font-family: Georgia, 'Times New Roman', serif; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 44px 26px; }}
    h1 {{ font-size: 42px; margin: 0 0 8px; }}
    .sub {{ color: var(--muted); font-family: Segoe UI, Arial, sans-serif; font-size: 16px; margin-bottom: 28px; }}
    .flow {{ padding: 18px 20px; background: rgba(255,255,255,.68); border: 1px solid rgba(16,36,62,.12); border-radius: 18px; font-family: Consolas, monospace; line-height: 1.55; white-space: pre-wrap; box-shadow: 0 18px 48px rgba(16,36,62,.12); margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; }}
    .card {{ display: block; text-decoration: none; color: var(--ink); background: var(--card); border: 1px solid rgba(16,36,62,.14); border-radius: 16px; padding: 18px; box-shadow: 0 10px 26px rgba(16,36,62,.08); }}
    .card strong {{ display: block; font-size: 18px; margin-bottom: 7px; }}
    .card span {{ color: var(--muted); font-family: Segoe UI, Arial, sans-serif; font-size: 14px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin: 0 0 22px; font-family: Segoe UI, Arial, sans-serif; }}
    .metric {{ background: rgba(255,255,255,.72); border: 1px solid rgba(16,36,62,.12); border-radius: 8px; padding: 10px; }}
    .metric b {{ display: block; margin-top: 4px; font-size: 17px; }}
  </style>
</head>
<body>
  <main>
    <h1>Trading System Layer Demo</h1>
    <div class="sub">Realtime Core MVP9 Shadow · Updated: {escape(updated)} · Symbols shown: {escape(', '.join(symbols[:18]))}</div>
    <div class="stats">
      <div class="metric">Equity<b>{_fmt_money(stats.get("equity"))}</b></div>
      <div class="metric">Return<b>{_fmt_pct(stats.get("return_pct"))}</b></div>
      <div class="metric">Cash<b>{_fmt_money(stats.get("cash"))}</b></div>
      <div class="metric">Open<b>{stats.get("open_positions", 0)}</b></div>
      <div class="metric">Slots<b>{stats.get("open_positions", 0)} / {stats.get("effective_max_positions") or spec.max_positions}</b></div>
      <div class="metric">Watchlist<b>{stats.get("watchlist", 0)}</b></div>
      <div class="metric">Closed<b>{stats.get("closed_trades", 0)}</b></div>
    </div>
    <div class="sub">Portfolio policy: {escape(str(stats.get("portfolio_policy_reason") or ""))}</div>
    <div class="flow">{escape(flow)}</div>
    <div class="grid">{cards}</div>
  </main>
</body>
</html>
"""


def _safe_read_recent_parquet(path: Path, symbols: list[str] | None, days: int = 14) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        end = frame["date"].max()
        start = end - pd.Timedelta(days=days)
        frame = frame[frame["date"].between(start, end)]
    if symbols and "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str).str.upper()
        frame = frame[frame["symbol"].isin(symbols)]
    return frame.sort_values([c for c in ["date", "symbol"] if c in frame.columns]).reset_index(drop=True)


def _enrich_vnindex_context(vnindex: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    if vnindex.empty or "date" not in vnindex.columns:
        return vnindex

    frame = vnindex.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame = frame.sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    volume = pd.to_numeric(frame.get("volume"), errors="coerce")
    frame["ret_1d_pct"] = (close.pct_change(1) * 100.0).round(2)
    frame["ret_5d_pct"] = (close.pct_change(5) * 100.0).round(2)
    frame["ret_20d_pct"] = (close.pct_change(20) * 100.0).round(2)
    frame["ma20"] = close.rolling(20).mean().round(2)
    frame["ma50"] = close.rolling(50).mean().round(2)
    frame["ma200"] = close.rolling(200).mean().round(2)
    frame["dist_ma20_pct"] = ((close / frame["ma20"] - 1.0) * 100.0).round(2)
    frame["dist_ma50_pct"] = ((close / frame["ma50"] - 1.0) * 100.0).round(2)
    frame["vol_ratio_20"] = (volume / volume.rolling(20).mean()).round(2)

    merge_specs = [
        (
            ROOT / "data" / "research" / "money_cycle" / "money_cycle_market.parquet",
            ["date", "CHDM20", "CHDM50", "DS20", "DS50"],
        ),
        (
            ROOT / "data" / "research" / "sector_rotation" / "market_regime.parquet",
            [
                "date", "regime", "risk_state", "avg_smart_money_score",
                "median_smart_money_score", "accumulation_distribution_ratio",
                "ds20_delta_5d", "sm_delta_5d", "chdm50_delta_5d",
            ],
        ),
        (
            ROOT / "data" / "research" / "smart_money_trace" / "smart_money_daily_summary.parquet",
            ["date", "accumulation_total", "distribution_total"],
        ),
    ]
    for path, columns in merge_specs:
        if not path.exists():
            continue
        try:
            extra = pd.read_parquet(path)
            extra["date"] = pd.to_datetime(extra["date"]).dt.tz_localize(None).dt.normalize()
            keep = [column for column in columns if column in extra.columns]
            if "date" in keep:
                frame = frame.merge(extra[keep], on="date", how="left")
        except Exception:
            continue

    if "risk_state" in frame.columns and "regime" in frame.columns:
        frame["market_context"] = frame["risk_state"].fillna("NEUTRAL").astype(str) + " / " + frame["regime"].fillna("NEUTRAL").astype(str)
    return frame.tail(days).reset_index(drop=True)


def _recent_symbol_features(raw_ohlcv: pd.DataFrame, symbol: str, entry: float, last: float) -> dict[str, Any]:
    if raw_ohlcv.empty or "symbol" not in raw_ohlcv.columns:
        return {}
    frame = raw_ohlcv[raw_ohlcv["symbol"].astype(str).str.upper().eq(symbol.upper())].copy()
    if frame.empty or "close" not in frame.columns:
        return {}
    frame = frame.sort_values("date") if "date" in frame.columns else frame
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    volume = pd.to_numeric(frame.get("volume"), errors="coerce").dropna() if "volume" in frame.columns else pd.Series(dtype=float)
    if close.empty:
        return {}
    ma5 = float(close.tail(5).mean()) if len(close) >= 5 else float(close.iloc[-1])
    ma10 = float(close.tail(10).mean()) if len(close) >= 10 else ma5
    ret_3d = (float(close.iloc[-1]) / float(close.iloc[-4]) - 1.0) * 100.0 if len(close) >= 4 and float(close.iloc[-4]) else 0.0
    ret_5d = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 and float(close.iloc[-6]) else 0.0
    vol_ratio_5 = 1.0
    if len(volume) >= 6 and float(volume.tail(6).head(5).mean() or 0.0) > 0:
        vol_ratio_5 = float(volume.iloc[-1]) / float(volume.tail(6).head(5).mean())
    return {
        "ma5": round(ma5, 4),
        "ma10": round(ma10, 4),
        "ret_3d_pct": round(ret_3d, 2),
        "ret_5d_pct": round(ret_5d, 2),
        "vol_ratio_5": round(vol_ratio_5, 2),
        "below_ma5": bool(last < ma5) if last else False,
        "below_ma10": bool(last < ma10) if last else False,
        "entry_gain_pct": round((last / entry - 1.0) * 100.0, 2) if entry else 0.0,
    }


def _exit_agent_advisory_rows(
    *,
    holdings: list[dict[str, Any]],
    watchlist: list[dict[str, Any]],
    raw_ohlcv: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Advisory-only exit agent.

    This deliberately does not close positions. It produces a parallel shadow
    opinion so the live demo can observe exit quality before any promotion.
    """
    best_candidate_score = max(
        [_safe_float(item.get("edge_rank_score"), 0.0) or 0.0 for item in watchlist] or [0.0]
    )
    rows: list[dict[str, Any]] = []
    for pos in holdings:
        symbol = str(pos.get("symbol") or "").upper()
        entry = _safe_float(pos.get("entry_price"), 0.0) or 0.0
        last = _safe_float(pos.get("last_price"), entry) or entry
        sl = _safe_float(pos.get("sl"), 0.0) or 0.0
        tp = _safe_float(pos.get("tp"), 0.0) or 0.0
        peak = _safe_float(pos.get("peak_price"), last) or last
        pnl_pct = (last / entry - 1.0) * 100.0 if entry else 0.0
        drawdown_from_peak = (last / peak - 1.0) * 100.0 if peak else 0.0
        dist_to_sl = (last / sl - 1.0) * 100.0 if sl else None
        dist_to_tp = (tp / last - 1.0) * 100.0 if last and tp else None
        feat = _recent_symbol_features(raw_ohlcv, symbol, entry, last)

        weakness = 0
        reasons: list[str] = []
        if dist_to_sl is not None and dist_to_sl <= 3.0:
            weakness += 35
            reasons.append("near SL")
        if pnl_pct > 12.0 and drawdown_from_peak <= -6.0:
            weakness += 25
            reasons.append("winner pulling back from peak")
        if bool(feat.get("below_ma5")):
            weakness += 12
            reasons.append("below MA5")
        if bool(feat.get("below_ma10")):
            weakness += 18
            reasons.append("below MA10")
        if float(feat.get("ret_3d_pct") or 0.0) <= -4.0:
            weakness += 15
            reasons.append("3d momentum weakening")
        if float(feat.get("vol_ratio_5") or 1.0) >= 1.7 and float(feat.get("ret_3d_pct") or 0.0) < 0:
            weakness += 10
            reasons.append("down move on heavier volume")
        replacement_pressure = max(0.0, best_candidate_score - (_safe_float(pos.get("edge_rank_score"), 0.0) or 0.0))
        if replacement_pressure >= 1.0:
            weakness += 10
            reasons.append("stronger replacement candidate")

        weakness = min(100, int(round(weakness)))
        if weakness >= 70:
            action = "EXIT_SUGGESTED"
        elif weakness >= 45:
            action = "REDUCE_SUGGESTED"
        elif weakness >= 25:
            action = "WATCH_WEAKENING"
        else:
            action = "HOLD_STRONG"

        shadow_exit_price = last if action in {"EXIT_SUGGESTED", "REDUCE_SUGGESTED"} else ""
        shadow_realized = round((last - entry) * int(pos.get("quantity") or 0), 0) if shadow_exit_price != "" else ""
        rows.append(
            {
                "symbol": symbol,
                "strategy_name": pos.get("strategy_name"),
                "entry_date": pos.get("entry_date"),
                "entry_price": entry,
                "last_price": last,
                "pnl_pct": round(pnl_pct, 2),
                "sl": sl,
                "tp": tp,
                "peak_price": peak,
                "drawdown_from_peak_pct": round(drawdown_from_peak, 2),
                "distance_to_sl_pct": round(dist_to_sl, 2) if dist_to_sl is not None else "",
                "distance_to_tp_pct": round(dist_to_tp, 2) if dist_to_tp is not None else "",
                "weakness_score": weakness,
                "replacement_pressure": round(replacement_pressure, 3),
                "exit_agent_action": action,
                "exit_agent_reason": "; ".join(reasons) if reasons else "trend/price action still acceptable",
                "shadow_exit_price": shadow_exit_price,
                "shadow_realized_pnl": shadow_realized,
                "shadow_only": True,
                "promotion_guard": "OFF: advisory + shadow only; no automatic agent discretionary sell",
                "updated_at": _now().isoformat(timespec="seconds"),
            }
        )
    return rows


def _write_layered_realtime_demo(spec: PipelineSpec, state: dict[str, Any], symbols: list[str], out_dir: Path) -> None:
    if spec.portfolio_id != "core_mvp9_shadow":
        return
    _portfolio_policy(spec, state)
    _update_stats(state)
    watchlist = list(state.get("watchlist", []))
    holdings = deepcopy(state.get("holdings", []))
    closed = list(reversed(state.get("closed", [])))
    shown_symbols = sorted(
        {
            str(item.get("symbol", "")).upper()
            for item in [*watchlist, *holdings, *closed[:20]]
            if item.get("symbol")
        }
    ) or symbols[:12]

    raw_ohlcv = _safe_read_recent_parquet(DATA_DIR / "ohlcv_master.parquet", shown_symbols)
    vnindex = _safe_read_recent_parquet(DATA_DIR / "index_master.parquet", None)
    if "symbol" in vnindex.columns:
        vnindex = vnindex[vnindex["symbol"].astype(str).str.upper().eq("VNINDEX")]
    vnindex = _enrich_vnindex_context(vnindex)
    exit_advisory_rows = _exit_agent_advisory_rows(
        holdings=holdings,
        watchlist=watchlist,
        raw_ohlcv=raw_ohlcv,
    )
    state["exit_advisory"] = exit_advisory_rows
    exit_advisory_by_symbol = {str(item.get("symbol") or "").upper(): item for item in exit_advisory_rows}

    feature_columns = [
        "symbol", "strategy_name", "setup_type", "edge_rank", "edge_rank_score", "feature_date",
        "edge_score", "mkt_regime_state", "mkt_regime_score", "rs_percentile_20", "smart_money_score",
        "value_ratio_20", "last_price", "entry_zone", "stop_loss", "take_profit", "trader_action",
        "risk_final_action", "risk_note",
    ]
    features = _rows_to_frame(watchlist, feature_columns)
    watchlist_frame = features.copy()
    if not watchlist_frame.empty:
        watchlist_frame.insert(
            0,
            "watch_action",
            [
                "PAPER_BUY_CANDIDATE" if idx < spec.max_positions else "WATCHLIST_ONLY"
                for idx in range(len(watchlist_frame))
            ],
        )
    strategy_catalog = pd.DataFrame(catalog_rows())
    if not strategy_catalog.empty and "strategy_id" in strategy_catalog.columns:
        strategy_catalog = strategy_catalog[strategy_catalog["strategy_id"].isin(MVP_EDGE_STRATEGIES)].copy()
        strategy_catalog["runtime_status"] = "ACTIVE_IN_CORE_MVP9_SHADOW"

    ranking = features.sort_values("edge_rank_score", ascending=False, na_position="last") if not features.empty else features
    action_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(watchlist, start=1):
        action_rows.append(
            {
                "rank": idx,
                "symbol": item.get("symbol"),
                "action": "PAPER_BUY_CANDIDATE" if idx <= spec.max_positions else "WATCHLIST_ONLY",
                "strategy_name": item.get("strategy_name"),
                "edge_rank_score": item.get("edge_rank_score"),
                "entry_zone": item.get("entry_zone"),
                "last_price": item.get("last_price"),
                "sl": item.get("stop_loss"),
                "tp": item.get("take_profit"),
                "risk_final_action": item.get("risk_final_action"),
                "portfolio_policy": (state.get("portfolio_policy") or {}).get("reason"),
                "note": "Realtime shadow only: no broker order is sent.",
            }
        )
    for item in holdings:
        exit_view = exit_advisory_by_symbol.get(str(item.get("symbol") or "").upper(), {})
        action_rows.append(
            {
                "rank": "",
                "symbol": item.get("symbol"),
                "action": "HOLDING_OPEN",
                "strategy_name": item.get("strategy_name"),
                "edge_rank_score": "",
                "entry_zone": "",
                "last_price": item.get("last_price"),
                "sl": item.get("sl"),
                "tp": item.get("tp"),
                "risk_final_action": "",
                "portfolio_policy": (state.get("portfolio_policy") or {}).get("reason"),
                "exit_agent_action": exit_view.get("exit_agent_action"),
                "weakness_score": exit_view.get("weakness_score"),
                "note": "Managed by realtime SL/TP/trailing SL. Exit agent is advisory/shadow only.",
            }
        )
    actions = pd.DataFrame(action_rows)

    stats = state.get("stats", {})
    initial = float(state.get("initial_capital") or 0.0)
    cash = float(stats.get("cash") or state.get("cash") or 0.0)
    open_unrealized = 0.0
    holding_rows: list[dict[str, Any]] = []
    for item in holdings:
        exit_view = exit_advisory_by_symbol.get(str(item.get("symbol") or "").upper(), {})
        entry = _safe_float(item.get("entry_price"), 0.0) or 0.0
        last = _safe_float(item.get("last_price"), entry) or entry
        qty = int(item.get("quantity") or 0)
        pnl = (last - entry) * qty
        pnl_pct = (last / entry - 1.0) * 100.0 if entry else 0.0
        open_unrealized += pnl
        holding_rows.append(
            {
                "symbol": item.get("symbol"),
                "strategy_name": item.get("strategy_name"),
                "entry_date": item.get("entry_date"),
                "entry_price": item.get("entry_price"),
                "shares": qty,
                "last_close": last,
                "market_value": round(last * qty, 0),
                "unrealized_pnl": round(pnl, 0),
                "unrealized_pnl_pct": round(pnl_pct, 2),
                "sl": item.get("sl"),
                "tp": item.get("tp"),
                "peak_price": item.get("peak_price"),
                "exit_agent_action": exit_view.get("exit_agent_action"),
                "weakness_score": exit_view.get("weakness_score"),
                "exit_agent_reason": exit_view.get("exit_agent_reason"),
                "shadow_exit_price": exit_view.get("shadow_exit_price"),
                "shadow_only": exit_view.get("shadow_only"),
            }
        )
    realized = float(stats.get("realized_pnl") or 0.0)
    summary = pd.DataFrame(
        [
            {"metric": "Start date", "value": state.get("created_at", "")[:10]},
            {"metric": "Mode", "value": "realtime_core_mvp9_shadow"},
            {"metric": "Initial NAV", "value": round(initial, 0)},
            {"metric": "Current NAV", "value": round(float(stats.get("equity") or initial), 0)},
            {"metric": "Return", "value": stats.get("return_pct")},
            {"metric": "Cash", "value": round(cash, 0)},
            {"metric": "Cash / NAV", "value": round(cash / float(stats.get("equity") or initial) * 100.0, 2) if float(stats.get("equity") or initial) else None},
            {"metric": "Closed realized PnL", "value": round(realized, 0)},
            {"metric": "Open unrealized PnL", "value": round(open_unrealized, 0)},
            {"metric": "Total realized + open PnL", "value": round(realized + open_unrealized, 0)},
            {"metric": "Open positions", "value": len(holdings)},
            {"metric": "Max positions", "value": spec.max_positions},
            {"metric": "Base max positions", "value": spec.max_positions},
            {"metric": "Effective max positions", "value": _effective_max_positions(spec, state)},
            {"metric": "Expanded max positions", "value": spec.max_expanded_positions},
            {"metric": "Available slots", "value": max(0, _effective_max_positions(spec, state) - len(holdings))},
            {"metric": "Closed win rate", "value": stats.get("win_rate_pct")},
            {"metric": "Portfolio policy", "value": (state.get("portfolio_policy") or {}).get("reason")},
            {"metric": "Avg top candidate score", "value": (state.get("portfolio_policy") or {}).get("avg_top_edge_rank_score")},
            {"metric": "Watchlist", "value": len(watchlist)},
            {"metric": "Exit agent mode", "value": "ADVISORY_AND_SHADOW_ONLY"},
            {"metric": "Exit suggestions", "value": sum(1 for item in exit_advisory_rows if item.get("exit_agent_action") in {"EXIT_SUGGESTED", "REDUCE_SUGGESTED"})},
            {"metric": "Closed trades", "value": stats.get("closed_trades")},
            {"metric": "Updated", "value": state.get("updated_at")},
        ]
    )
    closed_frame = _rows_to_frame(
        closed,
        [
            "symbol", "strategy_name", "entry_date", "exit_date", "entry_price", "exit_price",
            "quantity", "pnl_pct", "realized_pnl", "exit_reason", "exit_time",
        ],
    )
    holdings_frame = pd.DataFrame(holding_rows)
    status = pd.DataFrame(
        [
            {"key": "mode", "value": "realtime_shadow_paper"},
            {"key": "live_orders_enabled", "value": False},
            {"key": "portfolio_id", "value": spec.portfolio_id},
            {"key": "strategy_count", "value": len(MVP_EDGE_STRATEGIES)},
            {"key": "candidate_count", "value": len(watchlist)},
            {"key": "open_positions", "value": len(holdings)},
            {"key": "base_max_positions", "value": spec.max_positions},
            {"key": "effective_max_positions", "value": _effective_max_positions(spec, state)},
            {"key": "expanded_max_positions", "value": spec.max_expanded_positions},
            {"key": "dynamic_expansion_enabled", "value": spec.allow_dynamic_expansion},
            {"key": "rotation_enabled", "value": spec.allow_rotation},
            {"key": "portfolio_policy", "value": (state.get("portfolio_policy") or {}).get("reason")},
            {"key": "closed_trades", "value": len(state.get("closed", []))},
            {"key": "auto_sl_tp", "value": True},
            {"key": "auto_rotation", "value": spec.allow_rotation},
            {"key": "exit_agent_mode", "value": "ADVISORY_AND_SHADOW_ONLY"},
            {"key": "exit_agent_can_close_positions", "value": False},
            {"key": "exit_agent_promotion_guard", "value": "Requires separate backtest + live shadow validation before execution rights."},
            {"key": "html_refresh_seconds", "value": 15},
            {"key": "updated_at", "value": state.get("updated_at")},
        ]
    )
    exit_advisory = pd.DataFrame(exit_advisory_rows)

    _write_layer_table(out_dir, "01_raw_ohlcv", "01 Raw OHLCV", "Realtime shadow symbols, recent local OHLCV bars.", raw_ohlcv)
    _write_layer_table(out_dir, "01_vnindex", "01 VNINDEX", "Recent local VNINDEX context.", vnindex)
    _write_layer_table(out_dir, "02_feature_engine", "02 Feature Engine", "Realtime Core MVP9 feature/signal snapshot.", features)
    _write_layer_table(out_dir, "03_strategy_pool", "03 Strategy Pool", "Active Core MVP9 production strategy pool.", strategy_catalog)
    _write_layer_table(out_dir, "04_ranking_signals", "04 Ranking Engine", "Realtime ranked watchlist candidates.", ranking)
    _write_layer_table(out_dir, "05_risk_gate_actions", "05 Shared Pool Actions", "Realtime paper actions and holding states.", actions)
    _merge_realtime_portfolio_pages(
        out_dir=out_dir,
        summary=summary,
        holdings_frame=holdings_frame,
        closed_frame=closed_frame,
        continuation_mode=state.get("continuation_mode") == "layered_baseline",
    )
    _write_layer_table(out_dir, "07_production_status", "07 Production Status", "Safety and operating mode.", status)
    _write_layer_table(out_dir, "08_watchlist", "08 Watchlist", "Realtime candidates waiting for entry zone and risk approval.", watchlist_frame)
    _write_layer_table(
        out_dir,
        "09_exit_agent_advisory",
        "09 Exit Agent Advisory",
        "Advisory-only and shadow exit assessment. It does not close positions or alter portfolio PnL.",
        exit_advisory,
    )
    _write_symbol_detail_pages(
        out_dir=out_dir,
        raw_ohlcv=raw_ohlcv,
        features=features,
        ranking=ranking,
        actions=actions,
        holdings=_read_layer_csv(out_dir / "06c_open_holdings.csv"),
        closed=_read_layer_csv(out_dir / "06b_closed_trades.csv"),
        watchlist=watchlist_frame,
        exit_advisory=exit_advisory,
    )
    manifest = {
        "generated_at": _now().isoformat(timespec="seconds"),
        "mode": "realtime_core_mvp9_shadow",
        "portfolio_id": spec.portfolio_id,
        "symbols": shown_symbols,
        "entrypoint": str(out_dir / "index.html"),
        "state_file": str(_state_path(spec)),
    }
    (out_dir / "index.html").write_text(_layered_index_html(spec, state, shown_symbols), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        from scripts.inject_sortable_tables import inject

        for html_path in out_dir.rglob("*.html"):
            inject(html_path, out_dir)
    except Exception as exc:
        print(f"[{spec.portfolio_id}] sortable table injection failed: {type(exc).__name__}: {exc}")


def _write_dashboard(spec: PipelineSpec, state: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _portfolio_policy(spec, state)
    _update_stats(state)
    stats = state.get("stats", {})
    holdings = deepcopy(state.get("holdings", []))
    for pos in holdings:
        entry = _safe_float(pos.get("entry_price"), 0.0) or 0.0
        last = _safe_float(pos.get("last_price"), entry) or entry
        qty = int(pos.get("quantity") or 0)
        pos["unrealized_pnl"] = round((last - entry) * qty, 2)
        pos["pnl_pct"] = round((last / entry - 1.0) * 100.0, 2) if entry else 0.0

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="15">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(spec.label)}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #17212b; background: #f6f8fa; }}
    h1 {{ margin: 0 0 4px; font-size: 24px; }}
    h2 {{ margin-top: 26px; font-size: 18px; }}
    .muted {{ color: #6a737d; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .metric {{ background: white; border: 1px solid #d8dee4; border-radius: 8px; padding: 12px; }}
    .metric b {{ display: block; font-size: 18px; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d8dee4; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #eaeef2; text-align: left; font-size: 13px; }}
    th {{ background: #eef2f6; font-weight: 600; }}
    tr:hover {{ background: #f8fafc; }}
    .links a {{ margin-right: 12px; }}
  </style>
</head>
<body>
  <h1>{escape(spec.label)}</h1>
  <div class="muted">Updated {escape(str(state.get("updated_at") or ""))} | Paper only, no broker orders | Auto-refresh 15s</div>
  <div class="links">
    <a href="live_current.html">live_current</a>
    <a href="core_mvp9_shadow.html">core_mvp9_shadow</a>
  </div>
  <div class="grid">
    <div class="metric">Equity<b>{_fmt_money(stats.get("equity"))}</b></div>
    <div class="metric">Return<b>{_fmt_pct(stats.get("return_pct"))}</b></div>
    <div class="metric">Cash<b>{_fmt_money(stats.get("cash"))}</b></div>
    <div class="metric">Open<b>{stats.get("open_positions", 0)}</b></div>
    <div class="metric">Slots<b>{stats.get("open_positions", 0)} / {stats.get("effective_max_positions") or spec.max_positions}</b></div>
    <div class="metric">Watchlist<b>{stats.get("watchlist", 0)}</b></div>
    <div class="metric">Closed<b>{stats.get("closed_trades", 0)}</b></div>
    <div class="metric">Win Rate<b>{_fmt_pct(stats.get("win_rate_pct"))}</b></div>
    <div class="metric">Realized PnL<b>{_fmt_money(stats.get("realized_pnl"))}</b></div>
  </div>
  <h2>Holdings</h2>
  {_render_table(holdings, [
      ("symbol", "Symbol"), ("strategy_name", "Strategy"), ("entry_price", "Entry"),
      ("last_price", "Last"), ("quantity", "Qty"), ("sl", "SL"), ("tp", "TP"),
      ("edge_rank_score", "Score"), ("pnl_pct", "PnL %"), ("unrealized_pnl", "Unrealized"), ("entry_time", "Entry Time")
  ])}
  <h2>Watchlist</h2>
  {_render_table(state.get("watchlist", []), [
      ("symbol", "Symbol"), ("strategy_name", "Strategy"), ("edge_rank_score", "Score"),
      ("mkt_regime_state", "Regime"), ("entry_zone", "Entry Zone"), ("last_price", "Last"),
      ("stop_loss", "SL"), ("take_profit", "TP"), ("feature_date", "Feature Date")
  ])}
  <h2>Closed Trades</h2>
  {_render_table(list(reversed(state.get("closed", [])))[:100], [
      ("symbol", "Symbol"), ("strategy_name", "Strategy"), ("entry_price", "Entry"),
      ("exit_price", "Exit"), ("quantity", "Qty"), ("exit_reason", "Reason"),
      ("pnl_pct", "PnL %"), ("realized_pnl", "Realized"), ("exit_time", "Exit Time")
  ])}
  <h2>Event Log</h2>
  {_render_table(list(reversed(state.get("events", [])))[:80], [
      ("time", "Time"), ("message", "Event"), ("symbol", "Symbol"), ("price", "Price"),
      ("reason", "Reason"), ("pnl", "PnL")
  ])}
</body>
</html>
"""
    (REPORT_DIR / f"{spec.portfolio_id}.html").write_text(html, encoding="utf-8")
    (REPORT_DIR / "index.html").write_text(
        "<meta http-equiv='refresh' content='0; url=live_current.html'>",
        encoding="utf-8",
    )


class DualRealtimeShadowRunner:
    def __init__(
        self,
        symbols: list[str],
        *,
        poll_seconds: float,
        initial_capital: float,
        market_hours_only: bool,
        rebuild_minutes: int,
        layered_demo_dir: Path | None,
        stop_at: dt_time | None,
        continue_layered_baseline: bool = False,
        reset_continuation: bool = False,
    ) -> None:
        self.symbols = sorted({item.upper().strip() for item in symbols if item.strip()})
        self.poll_seconds = poll_seconds
        self.initial_capital = initial_capital
        self.market_hours_only = market_hours_only
        self.rebuild_minutes = rebuild_minutes
        self.layered_demo_dir = layered_demo_dir
        self.stop_at = stop_at
        self.states = {spec.portfolio_id: _load_state(spec, initial_capital) for spec in PIPELINES}
        if continue_layered_baseline:
            for spec in PIPELINES:
                self.states[spec.portfolio_id] = _seed_continuation_from_layered_demo(
                    spec,
                    self.states[spec.portfolio_id],
                    layered_demo_dir,
                    reset=reset_continuation,
                )
        self.last_rebuild_minute: dict[str, str] = {}

    def run(self, *, once: bool = False, no_graph: bool = False) -> None:
        if once or not self.market_hours_only or _is_market_session(_now()):
            for spec in PIPELINES:
                state = self.states[spec.portfolio_id]
                _refresh_watchlist(spec=spec, state=state, symbols=self.symbols, force=True, once=no_graph)
                _scan_entries_and_exits(spec, state)
                _save_state(spec, state)
                _write_dashboard(spec, state)
                if self.layered_demo_dir:
                    _write_layered_realtime_demo(spec, state, self.symbols, self.layered_demo_dir)

        if once:
            return

        print(
            f"[dual-shadow] running {len(PIPELINES)} paper pipelines | symbols={len(self.symbols)} | "
            f"poll={self.poll_seconds:g}s"
        )
        print(f"[dual-shadow] dashboards: {REPORT_DIR}")
        while True:
            now = _now()
            try:
                if self.stop_at and now.time() >= self.stop_at:
                    print(f"[dual-shadow] stop-at reached: {self.stop_at.strftime('%H:%M')}")
                    return
                if self.market_hours_only and not _is_market_session(now):
                    time.sleep(max(self.poll_seconds, 30))
                    continue

                for spec in PIPELINES:
                    state = self.states[spec.portfolio_id]
                    force = self._should_rebuild(spec, now)
                    _refresh_watchlist(spec=spec, state=state, symbols=self.symbols, force=force, once=no_graph)
                    _scan_entries_and_exits(spec, state)
                    _save_state(spec, state)
                    _write_dashboard(spec, state)
                    if self.layered_demo_dir:
                        _write_layered_realtime_demo(spec, state, self.symbols, self.layered_demo_dir)
                time.sleep(self.poll_seconds)
            except KeyboardInterrupt:
                print("\n[dual-shadow] stopped by user")
                return
            except Exception as exc:
                print(f"[dual-shadow] error: {exc}")
                time.sleep(max(self.poll_seconds, 10))

    def _should_rebuild(self, spec: PipelineSpec, now: datetime) -> bool:
        if self.rebuild_minutes <= 0:
            return False
        minute_bucket = now.strftime(f"%Y-%m-%d %H:{now.minute // self.rebuild_minutes:02d}")
        if self.last_rebuild_minute.get(spec.portfolio_id) == minute_bucket:
            return False
        self.last_rebuild_minute[spec.portfolio_id] = minute_bucket
        return True

    def _heartbeat(self, note: str) -> None:
        stamp = _now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] dual-shadow {note}")
        for spec in PIPELINES:
            state = self.states[spec.portfolio_id]
            _event(state, "HEARTBEAT", note=note)
            _save_state(spec, state)
            _write_dashboard(spec, state)
            if self.layered_demo_dir:
                _write_layered_realtime_demo(spec, state, self.symbols, self.layered_demo_dir)


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.universe == "vn100":
        return get_vn100_symbols()
    if args.universe == "vn30":
        return get_vn30_symbols()
    return [item.strip().upper() for item in args.symbols.split(",") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dual realtime paper/shadow portfolios")
    parser.add_argument("--universe", choices=["vn30", "vn100", "custom"], default="vn100")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols when --universe custom")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--initial-capital", type=float, default=100_000_000.0)
    parser.add_argument("--rebuild-minutes", type=int, default=30)
    parser.add_argument(
        "--layered-demo-dir",
        default=str(LAYERED_DEMO_DIR),
        help="Write the Core MVP9 realtime layer dashboard to this demo directory. Use empty string to disable.",
    )
    parser.add_argument("--all-hours", action="store_true")
    parser.add_argument("--stop-at", default="", help="Stop the runner at local HH:MM, useful for scheduled market sessions.")
    parser.add_argument(
        "--continue-layered-baseline",
        action="store_true",
        help="Use the current layered demo 06a/06c portfolio as the live paper state for core_mvp9_shadow.",
    )
    parser.add_argument(
        "--reset-continuation",
        action="store_true",
        help="Force reseeding continuation state from the layered demo open holdings.",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Fast smoke mode: skip trade_graph and use fallback entry/SL/TP from live price and edge risk.",
    )
    return parser


def _parse_stop_at(value: str) -> dt_time | None:
    if not value.strip():
        return None
    try:
        hour, minute = value.strip().split(":", 1)
        return dt_time(int(hour), int(minute))
    except Exception as exc:
        raise SystemExit(f"Invalid --stop-at value {value!r}; expected HH:MM") from exc


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    symbols = _resolve_symbols(args)
    if not symbols:
        raise SystemExit("No symbols resolved")
    runner = DualRealtimeShadowRunner(
        symbols,
        poll_seconds=args.poll_seconds,
        initial_capital=args.initial_capital,
        market_hours_only=not args.all_hours,
        rebuild_minutes=args.rebuild_minutes,
        layered_demo_dir=Path(args.layered_demo_dir) if args.layered_demo_dir else None,
        stop_at=_parse_stop_at(args.stop_at),
        continue_layered_baseline=args.continue_layered_baseline,
        reset_continuation=args.reset_continuation,
    )
    runner.run(once=args.once, no_graph=args.no_graph)


if __name__ == "__main__":
    main()
