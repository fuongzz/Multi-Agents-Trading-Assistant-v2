"""Generate paper signals for the hostile-market combo-long sleeve."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.backtest_hostile_combo_long import _add_features, _load_prices


ROOT = Path(__file__).resolve().parents[1]
SLEEVE_ID = "hostile_combo_long"
SLEEVE_LABEL = "Hostile Combo Long p1"
LOGIC_LABEL = "Top-1 hostile-market leadership rotation; full-size watch sleeve; rebalance 3 sessions; MA20 exit"
DEFAULT_TARGET_EXPOSURE = 0.25


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(
    out_dir: Path,
    start: str,
    end: str | None,
    min_value_vnd: float,
    target_exposure: float,
    paper_enabled: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prices = _load_prices(start, end or pd.Timestamp.today().date().isoformat())
    features = _add_features(prices)
    as_of = pd.Timestamp(end).normalize() if end else pd.to_datetime(features["date"]).max().normalize()
    latest = features[features["date"].eq(as_of)].copy()
    if latest.empty:
        as_of = pd.to_datetime(features["date"]).max().normalize()
        latest = features[features["date"].eq(as_of)].copy()

    candidates = latest[latest["value20"].ge(min_value_vnd)].sort_values("score", ascending=False).copy()
    candidates["sleeve_id"] = SLEEVE_ID
    candidates["strategy_name"] = SLEEVE_ID
    candidates["signal_date"] = as_of.date().isoformat()
    candidates["reference_price_vnd"] = candidates["close"] * 1000.0
    candidates["market_regime_state"] = "HOSTILE_OPPORTUNITY"
    candidates["risk"] = json.dumps(
        {
            "exit_model": "MA20_NEXT_OPEN",
            "stop_loss": 0.9999,
            "take_profit": 99.0,
            "max_holding_bars": 100000,
        },
        ensure_ascii=False,
    )
    candidate_cols = [
        "sleeve_id",
        "signal_date",
        "symbol",
        "strategy_name",
        "score",
        "reference_price_vnd",
        "ret20",
        "ret60",
        "ret120",
        "rs20",
        "rs60",
        "rs120",
        "value20",
        "above_ma20",
        "market_regime_state",
        "risk",
    ]
    candidates[candidate_cols].head(30).to_csv(out_dir / "hostile_candidates.csv", index=False, encoding="utf-8-sig")

    target = candidates.head(1).copy() if paper_enabled else candidates.head(0).copy()
    target["rank"] = 1
    target["action"] = "PAPER_BUY_CANDIDATE"
    target["max_positions_context"] = 1
    target["target_exposure"] = target_exposure
    target["target_value"] = 100_000_000.0 * target_exposure
    target_cols = [
        "sleeve_id",
        "signal_date",
        "symbol",
        "strategy_name",
        "rank",
        "action",
        "score",
        "reference_price_vnd",
        "max_positions_context",
        "target_exposure",
        "target_value",
        "market_regime_state",
        "risk",
    ]
    target[target_cols].to_csv(out_dir / "hostile_target_plan.csv", index=False, encoding="utf-8-sig")

    status = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "paper_trading_dry_run" if paper_enabled else "observe_only_research",
        "live_orders_enabled": False,
        "as_of_date": as_of.date().isoformat(),
        "feature_start": start,
        "universe": "vn100",
        "sleeve_id": SLEEVE_ID,
        "sleeve_label": SLEEVE_LABEL,
        "logic_label": LOGIC_LABEL,
        "positions": 1,
        "target_exposure": target_exposure,
        "rebalance_days": 3,
        "score_formula": "0.35*RS60 + 0.35*RS120 + 0.20*RS20 + 0.10*liquidity_rank",
        "exit_model": "close_below_MA20_after_close_T_exit_next_open_T_plus_1",
        "min_value_vnd": min_value_vnd,
        "candidate_count": int(len(candidates)),
        "target_count": int(len(target)),
        "target_symbols": target["symbol"].astype(str).tolist() if not target.empty else [],
        "paper_trading_enabled": paper_enabled,
        "risk_note": "Research/opportunity paper-watch sleeve only; full-size mode is high drawdown and must not replace Flow V2 or Core MVP.",
    }
    _write_json(out_dir / "hostile_status.json", status)
    _write_json(
        out_dir / "market_snapshot.json",
        {
            "as_of_date": as_of.date().isoformat(),
            "mkt_regime_state": "HOSTILE_OPPORTUNITY",
            "mkt_regime_score": None,
            "note": "This sleeve intentionally searches for single-stock leadership even when broad market gates are defensive.",
        },
    )
    print(out_dir / "hostile_status.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hostile combo-long paper signal generator.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--min-value-vnd", type=float, default=40_000_000_000.0)
    parser.add_argument("--target-exposure", type=float, default=DEFAULT_TARGET_EXPOSURE)
    parser.add_argument("--paper-enabled", action="store_true")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "combined_paper_trading_demo" / SLEEVE_ID))
    args = parser.parse_args()
    run(Path(args.out_dir), args.start, args.end or None, args.min_value_vnd, args.target_exposure, args.paper_enabled)


if __name__ == "__main__":
    main()
