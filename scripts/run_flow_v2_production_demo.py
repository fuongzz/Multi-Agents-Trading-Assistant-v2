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


def _ungated_discovery_pool(day: pd.DataFrame, pool_filter: str) -> pd.DataFrame:
    """Apply stock-quality filters only for the display-only idea list."""
    if pool_filter == "clean_flow":
        stock_mask = (
            (day["above_ma50"] == True)
            & (day["distribution_pressure_score"] <= 55)
            & (day["flow_sponsorship_score"] >= 50)
            & (day["flow_absorption_score"] >= 45)
            & (day["value_ratio_20"] >= 0.90)
            & (day["rs_percentile_20"] >= 0.50)
        )
    elif pool_filter == "high_rs":
        stock_mask = (
            (day["above_ma50"] == True)
            & (day["distribution_pressure_score"] <= 65)
            & (day["flow_sponsorship_score"] >= 45)
            & (day["value_ratio_20"] >= 0.80)
            & (day["rs_percentile_20"] >= 0.60)
        )
    elif pool_filter == "loose_flow":
        stock_mask = (
            (day["above_ma50"] == True)
            & (day["distribution_pressure_score"] <= 70)
            & (day["flow_sponsorship_score"] >= 42)
            & (day["value_ratio_20"] >= 0.70)
            & (day["rs_percentile_20"] >= 0.40)
        )
    else:
        stock_mask = (
            (day["above_ma50"] == True)
            & (day["distribution_pressure_score"] <= 65)
            & (day["flow_sponsorship_score"] >= 45)
            & (day["value_ratio_20"] >= 0.80)
            & (day["rs_percentile_20"] >= 0.45)
        )
    return day[stock_mask.fillna(False)].copy()


def _ungated_discovery_rows(
    day: pd.DataFrame,
    *,
    score_col: str,
    pool_filter: str,
    market_gate: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Rank Flow V2 observations without its market gate; never drives targets."""
    rows = _candidate_rows(_ungated_discovery_pool(day, pool_filter), score_col, limit)
    regime = str(day.iloc[0].get("mkt_regime_state") or "UNKNOWN") if not day.empty else "UNKNOWN"
    signal_date = pd.Timestamp(day.iloc[0]["date"]).strftime("%Y-%m-%d") if not day.empty else None
    for row in rows:
        row.update(
            {
                "date": signal_date,
                "family": "Flow V2 Rotation",
                "sleeves": "flow_v2, flow_v2_baseline, flow_v2_tiered",
                "strategy_name": f"{pool_filter} + flow_heavy ranking",
                "score": row.get("flow_v2_rotation_score"),
                "smart_money_score": None,
                "market_regime_state": regime,
                "relaxed_rule": f"Bỏ qua market gate: {market_gate}",
                "display_only": True,
                "explanation": (
                    f"Đạt lọc cổ phiếu Flow V2: sponsorship {row.get('flow_sponsorship_score')}, "
                    f"absorption {row.get('flow_absorption_score')}, RS {row.get('rs_percentile_20')}, "
                    f"thanh khoản {row.get('value_ratio_20')}x; chỉ quan sát khi bỏ gate thị trường."
                ),
            }
        )
    return rows


def _money_flow_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify high-quality flow proxy observations used for display only."""
    return (
        (frame["flow_sponsorship_score"] >= 50)
        & (frame["flow_absorption_score"] >= 45)
        & (frame["distribution_pressure_score"] <= 55)
        & (frame["value_ratio_20"] >= 0.90)
    ).fillna(False)


def _flow_label(row: pd.Series) -> str:
    if bool(row.get("strong_flow_today")) and float(row.get("value_ratio_20", 0.0) or 0.0) >= 1.20:
        return "DONG_TIEN_VAO_MANH"
    if bool(row.get("strong_flow_today")):
        return "DONG_TIEN_VAO"
    if float(row.get("distribution_pressure_score", 0.0) or 0.0) >= 70:
        return "AP_LUC_PHAN_PHOI"
    return "THEO_DOI"


def _flow_trend(row: pd.Series) -> str:
    persistent_days = int(row.get("inflow_days_5d", 0) or 0)
    change = float(row.get("flow_score_change_5d", 0.0) or 0.0)
    active_today = bool(row.get("strong_flow_today"))
    if persistent_days >= 3 and active_today and change >= -5:
        return "DUY_TRI_TIEN_VAO"
    if persistent_days >= 3 and not active_today:
        return "ROI_NHOM_TIEN_VAO"
    if change >= 8:
        return "DANG_TANG_TOC"
    if change <= -8:
        return "DANG_GIAM_NHIET"
    return "DI_NGANG"


def _money_flow_pulse(
    features: pd.DataFrame,
    as_of: pd.Timestamp,
    score_col: str,
    *,
    limit: int = 15,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize observable price/volume flow proxies without affecting trading."""
    dates = (
        features.loc[features["date"] <= as_of, "date"]
        .drop_duplicates()
        .sort_values()
        .tail(6)
        .tolist()
    )
    if not dates:
        return {"available": False, "date": as_of.strftime("%Y-%m-%d")}, [], []

    recent = features[features["date"].isin(dates)].copy()
    recent["strong_flow"] = _money_flow_mask(recent)
    latest = recent[recent["date"] == dates[-1]].copy()
    latest["strong_flow_today"] = latest["strong_flow"]
    five_dates = dates[-5:]
    five_session = recent[recent["date"].isin(five_dates)]
    persistent = five_session.groupby("symbol")["strong_flow"].sum().rename("inflow_days_5d")
    average_score = five_session.groupby("symbol")[score_col].mean().rename("flow_score_avg_5d")

    prior = recent[recent["date"] == dates[0]][["symbol", score_col]].rename(
        columns={score_col: "flow_score_5d_ago"}
    )
    latest = (
        latest.merge(persistent, on="symbol", how="left")
        .merge(average_score, on="symbol", how="left")
        .merge(prior, on="symbol", how="left")
    )
    latest["inflow_days_5d"] = latest["inflow_days_5d"].fillna(0).astype(int)
    latest["flow_score_change_5d"] = latest[score_col] - latest["flow_score_5d_ago"]
    latest["flow_signal"] = latest.apply(_flow_label, axis=1)
    latest["flow_trend_5d"] = latest.apply(_flow_trend, axis=1)

    market_daily = recent.drop_duplicates("date").sort_values("date")
    today_market = market_daily.iloc[-1]
    prior_market = market_daily.iloc[0]
    chdm_delta = float(today_market["mkt_CHDM20"] - prior_market["mkt_CHDM20"])
    ds_delta = float(today_market["mkt_DS20"] - prior_market["mkt_DS20"])
    state = str(today_market.get("mkt_regime_state") or "UNKNOWN")
    if state == "RISK_OFF":
        pulse_state = "REGIME_CHAN_GIAI_NGAN"
    elif chdm_delta >= 5 and ds_delta <= -0.05:
        pulse_state = "DONG_TIEN_DANG_CAI_THIEN"
    elif chdm_delta <= -5 or ds_delta >= 0.05:
        pulse_state = "AP_LUC_PHAN_PHOI_TANG"
    else:
        pulse_state = "TRUNG_LAP_THEO_DOI"

    def format_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rank, row in enumerate(frame.to_dict("records"), start=1):
            rows.append(
                {
                    "rank": rank,
                    "symbol": row["symbol"],
                    "industry": row.get("industry"),
                    "flow_signal": row["flow_signal"],
                    "flow_trend_5d": row["flow_trend_5d"],
                    "inflow_days_5d": int(row["inflow_days_5d"]),
                    "flow_score_today": _round_or_none(row.get(score_col)),
                    "flow_score_avg_5d": _round_or_none(row.get("flow_score_avg_5d")),
                    "flow_score_change_5d": _round_or_none(row.get("flow_score_change_5d")),
                    "flow_sponsorship_score": _round_or_none(row.get("flow_sponsorship_score")),
                    "flow_absorption_score": _round_or_none(row.get("flow_absorption_score")),
                    "distribution_pressure_score": _round_or_none(row.get("distribution_pressure_score")),
                    "value_ratio_20": _round_or_none(row.get("value_ratio_20")),
                    "rs_percentile_20": _round_or_none(row.get("rs_percentile_20")),
                }
            )
        return rows

    today_ranked = latest[latest["strong_flow_today"]].sort_values(score_col, ascending=False).head(limit)
    recent_ranked = (
        latest[latest["inflow_days_5d"] >= 2]
        .sort_values(["inflow_days_5d", "flow_score_avg_5d", score_col], ascending=[False, False, False])
        .head(limit)
    )
    today_rows = format_rows(today_ranked)
    recent_rows = format_rows(recent_ranked)

    summary = {
        "available": True,
        "date": dates[-1].strftime("%Y-%m-%d"),
        "data_basis": "price_volume_flow_proxy_not_investor_net_buy",
        "lookback_sessions": len(five_dates),
        "pulse_state": pulse_state,
        "market_regime_state": state,
        "mkt_CHDM20": _round_or_none(today_market.get("mkt_CHDM20")),
        "mkt_CHDM20_change_5d": _round_or_none(chdm_delta),
        "mkt_DS20": _round_or_none(today_market.get("mkt_DS20")),
        "mkt_DS20_change_5d": _round_or_none(ds_delta),
        "strong_inflow_symbols_today": int(latest["strong_flow_today"].sum()),
        "persistent_inflow_symbols_5d": int((latest["inflow_days_5d"] >= 3).sum()),
        "displayed_today_leaders": len(today_rows),
        "displayed_recent_leaders": len(recent_rows),
        "interpretation_note": (
            "Tín hiệu dòng tiền là proxy từ giá, thanh khoản, sponsorship, absorption "
            "và distribution của Flow V2; không phải giá trị mua ròng theo lệnh."
        ),
    }
    return summary, today_rows, recent_rows


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
- `money_flow_pulse.json`: market-level observable money-flow pulse and trend context.
- `flow_v2_money_flow_today.csv`: symbols satisfying the observable inflow proxy today.
- `flow_v2_money_flow_recent.csv`: symbols repeatedly showing the proxy over the last five sessions.
- `ungated_discovery_candidates.csv`: display-only symbols after removing the market gate; never used for paper targets.
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
    parser.add_argument("--sleeve-id", default="flow_v2")
    parser.add_argument("--sleeve-label", default="Flow V2 Rotation")
    parser.add_argument("--logic-label", default="Flow Money V2 high-RS flow-heavy rotation")
    parser.add_argument("--early-exit-mode", default="none")
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
    money_flow_pulse, money_flow_today, money_flow_recent = _money_flow_pulse(features, as_of, score_col)
    ungated_discovery = _ungated_discovery_rows(
        day,
        score_col=score_col,
        pool_filter=cfg.pool_filter,
        market_gate=cfg.market_gate,
        limit=args.max_candidates,
    )

    pd.DataFrame(candidates).to_csv(out_dir / "flow_v2_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(targets).to_csv(out_dir / "flow_v2_target_plan.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(money_flow_today).to_csv(
        out_dir / "flow_v2_money_flow_today.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(money_flow_recent).to_csv(
        out_dir / "flow_v2_money_flow_recent.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(ungated_discovery).to_csv(
        out_dir / "ungated_discovery_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

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
        "sleeve_id": args.sleeve_id,
        "sleeve_label": args.sleeve_label,
        "logic_label": args.logic_label,
        "early_exit_mode": args.early_exit_mode,
        "candidate_count": len(candidates),
        "target_count": len(targets),
        "target_symbols": [item["symbol"] for item in targets],
        "ungated_discovery_count": len(ungated_discovery),
        "ungated_discovery_display_only": True,
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
    (out_dir / "money_flow_pulse.json").write_text(
        json.dumps(money_flow_pulse, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_readme(out_dir, status)
    print(json.dumps(status, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
