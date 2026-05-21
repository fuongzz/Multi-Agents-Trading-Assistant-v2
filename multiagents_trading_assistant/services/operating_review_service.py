"""Operating review service.

Builds an end-of-day / weekly operating review from live trade history and
trusted backtest artifacts. The goal is to treat P&L as a system diagnostic,
not just a scoreboard.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
import csv

from multiagents_trading_assistant import database as db
from multiagents_trading_assistant.memory.strategy_memory import StrategyMemory
from multiagents_trading_assistant.services.portfolio_service import compute_performance_stats


ROOT = Path(__file__).resolve().parents[2]
BACKTEST_RESULTS = ROOT / "backtest_results"

TRUSTED_SUMMARY_FILES = {
    "single_strategy_ytd": BACKTEST_RESULTS / "all_strategies_unbiased_vn100_2025-01-01_2026-05-12" / "summary.csv",
    "combo_strategy_ytd": BACKTEST_RESULTS / "combos_unbiased_vn100_all_combos_2025ytd_2025-01-01_2026-05-12" / "summary.csv",
    "combo_strategy_4y": BACKTEST_RESULTS / "combos_unbiased_vn100_top5_4year_2022-01-01_2026-05-12" / "summary.csv",
}


@dataclass(frozen=True)
class ReviewLeaderboardEntry:
    name: str
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    number_of_trades: int
    source: str


def _parse_trade_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d")
    except ValueError:
        return None


def _filter_closed_trades(
    trades: list[dict],
    *,
    as_of_date: str | None = None,
    lookback_days: int = 20,
) -> list[dict]:
    if not trades:
        return []
    as_of = _parse_trade_date(as_of_date) if as_of_date else max(
        (_parse_trade_date(t.get("trade_date")) for t in trades),
        default=None,
    )
    if as_of is None:
        return []
    cutoff = as_of - timedelta(days=lookback_days - 1)
    filtered = []
    for trade in trades:
        trade_dt = _parse_trade_date(trade.get("trade_date"))
        if trade_dt is None:
            continue
        if cutoff <= trade_dt <= as_of:
            filtered.append(trade)
    return sorted(filtered, key=lambda row: row.get("trade_date") or "")


def _setup_breakdown(trades: list[dict]) -> list[dict]:
    by_setup: dict[str, dict] = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0, "total_pnl_pct": 0.0})
    for trade in trades:
        setup = str(trade.get("strategy") or trade.get("setup_type") or "UNKNOWN")
        row = by_setup[setup]
        pnl = float(trade.get("realized_pnl") or 0.0)
        pnl_pct = float(trade.get("pnl_pct") or 0.0)
        row["trades"] += 1
        row["total_pnl"] += pnl
        row["total_pnl_pct"] += pnl_pct
        if pnl > 0:
            row["wins"] += 1

    result = []
    for setup, row in by_setup.items():
        trades_n = row["trades"]
        result.append(
            {
                "setup": setup,
                "trades": trades_n,
                "win_rate_pct": round(row["wins"] / trades_n * 100, 2) if trades_n else 0.0,
                "avg_pnl_pct": round(row["total_pnl_pct"] / trades_n, 2) if trades_n else 0.0,
                "total_pnl": round(row["total_pnl"], 2),
            }
        )
    return sorted(result, key=lambda item: (item["total_pnl"], item["avg_pnl_pct"]), reverse=True)


def _loss_breakdown(trades: list[dict]) -> list[dict]:
    losers = [t for t in trades if float(t.get("realized_pnl") or 0.0) < 0]
    by_category: dict[str, dict] = defaultdict(lambda: {"count": 0, "total_pnl": 0.0, "total_pnl_pct": 0.0})
    for trade in losers:
        key = str(trade.get("loss_category") or "UNCLASSIFIED")
        row = by_category[key]
        row["count"] += 1
        row["total_pnl"] += float(trade.get("realized_pnl") or 0.0)
        row["total_pnl_pct"] += float(trade.get("pnl_pct") or 0.0)

    result = []
    for category, row in by_category.items():
        count = row["count"]
        result.append(
            {
                "loss_category": category,
                "count": count,
                "avg_pnl_pct": round(row["total_pnl_pct"] / count, 2) if count else 0.0,
                "total_pnl": round(row["total_pnl"], 2),
            }
        )
    return sorted(result, key=lambda item: item["total_pnl"])


def _trusted_leaderboard_entry(path: Path, *, name_field: str) -> ReviewLeaderboardEntry | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    def _score(row: dict) -> tuple[float, float]:
        return (float(row.get("total_return_pct") or 0.0), float(row.get("sharpe_ratio") or 0.0))

    best = max(rows, key=_score)
    return ReviewLeaderboardEntry(
        name=str(best.get(name_field) or "UNKNOWN"),
        total_return_pct=round(float(best.get("total_return_pct") or 0.0), 2),
        sharpe_ratio=round(float(best.get("sharpe_ratio") or 0.0), 3),
        max_drawdown_pct=round(float(best.get("max_drawdown_pct") or 0.0), 2),
        win_rate_pct=round(float(best.get("win_rate_pct") or 0.0), 2),
        number_of_trades=int(float(best.get("number_of_trades") or 0)),
        source=str(path),
    )


def load_trusted_leaderboards() -> dict[str, dict] | None:
    leaders = {
        "single_strategy_ytd": _trusted_leaderboard_entry(TRUSTED_SUMMARY_FILES["single_strategy_ytd"], name_field="strategy"),
        "combo_strategy_ytd": _trusted_leaderboard_entry(TRUSTED_SUMMARY_FILES["combo_strategy_ytd"], name_field="combo"),
        "combo_strategy_4y": _trusted_leaderboard_entry(TRUSTED_SUMMARY_FILES["combo_strategy_4y"], name_field="combo"),
    }
    return {key: asdict(value) for key, value in leaders.items() if value is not None}


def build_operating_review(
    *,
    as_of_date: str | None = None,
    lookback_days: int = 20,
    live_prices: dict[str, float] | None = None,
    strategy_memory: StrategyMemory | None = None,
) -> dict:
    closed = db.get_closed_trades(limit=1000)
    recent = _filter_closed_trades(closed, as_of_date=as_of_date, lookback_days=lookback_days)

    realized_pnl = round(sum(float(t.get("realized_pnl") or 0.0) for t in recent), 2)
    wins = sum(1 for t in recent if float(t.get("realized_pnl") or 0.0) > 0)
    losses = sum(1 for t in recent if float(t.get("realized_pnl") or 0.0) < 0)
    total = len(recent)

    latest_trade_date = recent[-1].get("trade_date") if recent else None
    perf = compute_performance_stats(live_prices=live_prices)
    memory = strategy_memory or StrategyMemory()

    return {
        "as_of_date": as_of_date or latest_trade_date,
        "review_window_days": lookback_days,
        "recent_realized": {
            "trade_count": total,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(wins / total * 100, 2) if total else 0.0,
            "realized_pnl_vnd": realized_pnl,
            "avg_pnl_pct": round(sum(float(t.get("pnl_pct") or 0.0) for t in recent) / total, 2) if total else 0.0,
        },
        "setup_breakdown": _setup_breakdown(recent),
        "loss_breakdown": _loss_breakdown(recent),
        "portfolio_snapshot": {
            "open_positions_count": perf.get("open_positions_count", 0),
            "open_unrealized_pnl_vnd": perf.get("open_unrealized_pnl_vnd", 0),
            "total_realized_pnl_vnd": perf.get("total_realized_pnl_vnd", 0),
            "win_rate_pct_all_time": perf.get("win_rate", 0.0),
            "avg_realized_rr": perf.get("avg_realized_rr", 0.0),
        },
        "trusted_leaderboards": load_trusted_leaderboards(),
        "strategy_memory": memory.summary(),
    }
