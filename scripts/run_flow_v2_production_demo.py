"""Create a Flow Money V2 production dry-run package.

This is not a backtest and does not send broker orders. It answers:
"At the latest available close, what would Flow V2 prepare for the next
session?"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.backtest_flow_v2_rotation_production_like import (
    ROOT,
    RotationConfig,
    _eligible_pool,
    _prepare_features,
    _score_column,
)


DATA_DIR = ROOT / "multiagents_trading_assistant" / "data"


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None


def _latest_data_date() -> pd.Timestamp:
    frame = pd.read_parquet(DATA_DIR / "ohlcv_master.parquet", columns=["date"])
    return pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize().max()


def _market_snapshot(day: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, Any]:
    if day.empty:
        return {"available": False, "date": as_of.strftime("%Y-%m-%d")}
    row = day.iloc[0]
    columns = [
        "mkt_regime_state",
        "mkt_regime_score",
        "mkt_CHDM20",
        "mkt_CHDM50",
        "mkt_DS20",
        "mkt_DS50",
        "mkt_ret_20d",
        "mkt_ret_60d",
        "mkt_drawdown_60d",
        "mkt_above_ma50",
        "mkt_above_ma200",
    ]
    out: dict[str, Any] = {"available": True, "date": as_of.strftime("%Y-%m-%d")}
    for column in columns:
        if column not in row:
            continue
        value = row[column]
        out[column] = value if isinstance(value, str) else _round_or_none(value)
    return out


def _candidate_rows(pool: pd.DataFrame, score_col: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ranked = pool.sort_values(score_col, ascending=False).head(limit)
    for rank, row in enumerate(ranked.to_dict("records"), start=1):
        rows.append(
            {
                "rank": rank,
                "symbol": row["symbol"],
                "flow_v2_rotation_score": _round_or_none(row.get(score_col)),
                "close": _round_or_none(row.get("close"), 2),
                "target_weight": None,
                "sector_cycle_score": _round_or_none(row.get("sector_cycle_score")),
                "flow_sponsorship_score": _round_or_none(row.get("flow_sponsorship_score")),
                "flow_absorption_score": _round_or_none(row.get("flow_absorption_score")),
                "distribution_pressure_score": _round_or_none(row.get("distribution_pressure_score")),
                "flow_quality_v2_score": _round_or_none(row.get("flow_quality_v2_score")),
                "stock_lifecycle_score": _round_or_none(row.get("stock_lifecycle_score")),
                "rs_percentile_20": _round_or_none(row.get("rs_percentile_20")),
                "value_ratio_20": _round_or_none(row.get("value_ratio_20")),
                "above_ma50": bool(row.get("above_ma50")) if pd.notna(row.get("above_ma50")) else None,
            }
        )
    return rows


def _target_plan(
    candidates: list[dict[str, Any]],
    *,
    capital: float,
    lot_size: int,
    price_unit_multiplier: float,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    target_weight = 1.0 / len(candidates)
    rows = []
    for item in candidates:
        close = item.get("close")
        target_value = capital * target_weight
        executable_price = float(close) * price_unit_multiplier if close and close > 0 else 0.0
        if executable_price > 0:
            shares = int(target_value / executable_price)
            shares = (shares // lot_size) * lot_size
            indicative_value = shares * executable_price
        else:
            shares = 0
            indicative_value = 0.0
        rows.append(
            {
                "symbol": item["symbol"],
                "action": "PAPER_TARGET_BUY",
                "signal_date": item.get("signal_date"),
                "execution_assumption": "next_session_open_or_limit_after_manual_approval",
                "reference_close": close,
                "price_unit_multiplier": price_unit_multiplier,
                "reference_price_vnd": round(executable_price, 2),
                "target_weight": round(target_weight, 4),
                "target_value": round(target_value, 2),
                "indicative_shares_by_close": shares,
                "indicative_value_by_close": round(indicative_value, 2),
                "requires_manual_approval": True,
            }
        )
    return rows


def _write_readme(out_dir: Path, status: dict[str, Any]) -> None:
    content = f"""# Flow V2 Production Demo

Mode: `paper_trading_dry_run`

As-of close: `{status["as_of_date"]}`

Config:

- Universe: `{status["universe"]}`
- Market gate: `{status["market_gate"]}`
- Pool filter: `{status["pool_filter"]}`
- Score mode: `{status["score_mode"]}`
- Max positions: `{status["positions"]}`
- Rebalance cadence: `{status["rebalance_days"]}` trading days
- Broker orders: disabled

Files:

- `flow_v2_status.json`: run status and safety notes.
- `market_snapshot.json`: market regime and money-cycle context at as-of close.
- `flow_v2_candidates.csv`: ranked eligible pool.
- `flow_v2_target_plan.csv`: paper target allocation for the next session.

Operational rule:

The signal is computed after close T using local parquet data through T. Any
real fill must be decided at open/limit on T+1 after manual approval. The share
counts in the target plan are indicative only because they use close T as a
reference price.
"""
    (out_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Flow Money V2 production dry-run artifacts.")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--as-of", default="", help="YYYY-MM-DD. Defaults to latest local price date.")
    parser.add_argument("--start", default="2020-01-01", help="Feature warm-up start date.")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--capital", type=float, default=1_000_000_000.0)
    parser.add_argument("--positions", type=int, default=2)
    parser.add_argument("--rebalance-days", type=int, default=20)
    parser.add_argument("--market-gate", default="risk_on_or_strong_neutral")
    parser.add_argument("--pool-filter", default="clean_flow")
    parser.add_argument("--score-mode", default="sector_heavy")
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument(
        "--price-unit-multiplier",
        type=float,
        default=1000.0,
        help="Multiplier from parquet price unit to VND. Local OHLCV prices are usually in thousand VND.",
    )
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else _latest_data_date()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "reports" / f"flow_v2_production_demo_{as_of:%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RotationConfig(
        universe=args.universe,
        start=args.start,
        end=as_of.strftime("%Y-%m-%d"),
        positions=args.positions,
        rebalance_days=args.rebalance_days,
        initial_capital=args.capital,
        lot_size=args.lot_size,
        selection_mode="turnover_aware",
        market_gate=args.market_gate,
        pool_filter=args.pool_filter,
        score_mode=args.score_mode,
    )

    features = _prepare_features(cfg.universe, cfg.start, cfg.end)
    day = features[features["date"] == as_of].copy()
    if day.empty:
        latest = features["date"].max()
        raise RuntimeError(f"No features for {as_of:%Y-%m-%d}; latest available feature date is {latest:%Y-%m-%d}")

    score_col = _score_column(cfg.score_mode)
    pool = _eligible_pool(day, cfg.market_gate, cfg.pool_filter)
    candidates = _candidate_rows(pool, score_col, args.max_candidates)
    for row in candidates:
        row["signal_date"] = as_of.strftime("%Y-%m-%d")
        row["score_mode"] = cfg.score_mode
        row["pool_filter"] = cfg.pool_filter
        row["market_gate"] = cfg.market_gate
    selected = candidates[: cfg.positions]
    targets = _target_plan(
        selected,
        capital=cfg.initial_capital,
        lot_size=cfg.lot_size,
        price_unit_multiplier=args.price_unit_multiplier,
    )

    pd.DataFrame(candidates).to_csv(out_dir / "flow_v2_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(targets).to_csv(out_dir / "flow_v2_target_plan.csv", index=False, encoding="utf-8-sig")

    market = _market_snapshot(day, as_of)
    status = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "paper_trading_dry_run",
        "live_orders_enabled": False,
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "feature_start": args.start,
        "universe": cfg.universe,
        "positions": cfg.positions,
        "rebalance_days": cfg.rebalance_days,
        "market_gate": cfg.market_gate,
        "pool_filter": cfg.pool_filter,
        "score_mode": cfg.score_mode,
        "capital": cfg.initial_capital,
        "price_unit_multiplier": args.price_unit_multiplier,
        "candidate_count": len(candidates),
        "target_count": len(targets),
        "target_symbols": [item["symbol"] for item in targets],
        "safety_notes": [
            "No broker order is sent.",
            "Signals use only local parquet data through the as-of close.",
            "Indicative shares use close T; real execution must use next-session open or limit.",
            "Current universe is not point-in-time; this is acceptable for demo paper mode, not final production validation.",
        ],
    }
    (out_dir / "flow_v2_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (out_dir / "market_snapshot.json").write_text(
        json.dumps(market, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_readme(out_dir, status)
    print(json.dumps(status, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
