"""Run a production-style dry run from the latest local market data.

This script is intentionally not a backtest. It answers the production demo
question: "At the latest available close, what would the system prepare for the
next trading session?"
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import (
    Hypothesis,
    evaluate_filters,
    load_hypotheses,
    rank_candidates,
)
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG
from multiagents_trading_assistant.edge_lab.strategy_sleeves import (
    MVP_EDGE_STRATEGIES,
    active_portfolio_sleeves,
    catalog_rows,
)
from multiagents_trading_assistant.research.sector_rotation.enrich_features import (
    enrich_features_with_rotation,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "multiagents_trading_assistant" / "data"
CONFIG_DIR = ROOT / "multiagents_trading_assistant" / "edge_lab" / "configs"
GLOBAL_CONFIG = CONFIG_DIR / "global_market_hypotheses.json"
SECTOR_CONFIG = CONFIG_DIR / "sector_rotation_hypotheses.json"
OIL_GAS_CONFIG = CONFIG_DIR / "oil_gas_rotation_hypotheses.json"
THEME_FLOW_CONFIG = CONFIG_DIR / "theme_flow_hypotheses.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
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


def _load_hypothesis_map() -> dict[str, Hypothesis]:
    out: dict[str, Hypothesis] = {}
    for config in [Path(DEFAULT_CONFIG), GLOBAL_CONFIG, SECTOR_CONFIG, OIL_GAS_CONFIG, THEME_FLOW_CONFIG]:
        if not config.exists():
            continue
        for hypothesis in load_hypotheses(config):
            out[hypothesis.name] = hypothesis
    return out


def _latest_index_snapshot(as_of: pd.Timestamp) -> dict[str, Any]:
    path = DATA_DIR / "index_master.parquet"
    if not path.exists():
        return {"available": False}
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    if "symbol" in frame.columns:
        frame = frame[frame["symbol"].astype(str).str.upper() == "VNINDEX"]
    frame = frame[frame["date"] <= as_of].sort_values("date")
    if frame.empty:
        return {"available": False}
    last = frame.iloc[-1]
    close = float(last["close"])
    prev_close = float(frame.iloc[-2]["close"]) if len(frame) >= 2 else None
    ret_1d = (close / prev_close - 1.0) if prev_close else None
    return {
        "available": True,
        "symbol": "VNINDEX",
        "date": pd.Timestamp(last["date"]).strftime("%Y-%m-%d"),
        "close": _round_or_none(close, 2),
        "ret_1d_pct": _round_or_none(ret_1d * 100 if ret_1d is not None else None, 2),
        "volume": _round_or_none(last.get("volume"), 0),
    }


def _market_snapshot(features: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, Any]:
    daily = features[pd.to_datetime(features["date"]).dt.normalize() == as_of]
    if daily.empty:
        return {"available": False, "date": as_of.strftime("%Y-%m-%d")}
    row = daily.iloc[0]
    market_columns = [
        "mkt_regime_state",
        "mkt_regime_score",
        "mkt_above_ma50",
        "mkt_above_ma200",
        "mkt_ret_20d",
        "mkt_ret_60d",
        "mkt_drawdown_60d",
        "mkt_CHDM20",
        "mkt_DS20",
        "sector_rotation_state",
        "sector_rotation_score",
    ]
    snapshot = {"available": True, "date": as_of.strftime("%Y-%m-%d")}
    for column in market_columns:
        if column not in row:
            continue
        value = row[column]
        snapshot[column] = value if isinstance(value, str) else _round_or_none(value)
    snapshot["vnindex"] = _latest_index_snapshot(as_of)
    return snapshot


def _candidate_rows(day_signals: dict[str, dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    rows = []
    for symbol, signal in day_signals.items():
        if not signal.get("passed"):
            continue
        rows.append(
            {
                "symbol": symbol,
                "strategy_name": signal.get("strategy_name"),
                "strategy_family": signal.get("strategy_family"),
                "setup_type": signal.get("setup_type"),
                "edge_rank": signal.get("edge_rank"),
                "edge_rank_score": signal.get("edge_rank_score"),
                "edge_score": signal.get("edge_score"),
                "smart_money_score": signal.get("smart_money_score"),
                "smart_money_score_delta": signal.get("smart_money_score_delta"),
                "rs_percentile_20": signal.get("rs_percentile_20"),
                "value_ratio_20": signal.get("value_ratio_20"),
                "CHDM50": signal.get("CHDM50"),
                "DS20": signal.get("DS20"),
                "mkt_regime_state": signal.get("mkt_regime_state"),
                "mkt_regime_score": signal.get("mkt_regime_score"),
                "feature_date": signal.get("feature_date"),
                "risk": json.dumps(signal.get("risk") or {}, ensure_ascii=False),
            }
        )
    return sorted(rows, key=lambda item: float(item.get("edge_rank_score") or 0.0), reverse=True)[
        :max_candidates
    ]


def _vote_rows(
    *,
    latest_features: pd.DataFrame,
    hypotheses: list[Hypothesis],
    as_of: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        mask = evaluate_filters(latest_features, hypothesis)
        ranked = rank_candidates(latest_features[mask], hypothesis)
        for rank_idx, row in enumerate(ranked.to_dict("records"), start=1):
            rows.append(
                {
                    "date": as_of.strftime("%Y-%m-%d"),
                    "symbol": row["symbol"],
                    "strategy_name": hypothesis.name,
                    "rank_in_strategy": rank_idx,
                    "edge_rank_score": _round_or_none(row.get("_edge_rank")),
                    "edge_score": _round_or_none(row.get("edge_score")),
                    "smart_money_score": _round_or_none(row.get("smart_money_score")),
                    "rs_percentile_20": _round_or_none(row.get("rs_percentile_20")),
                    "value_ratio_20": _round_or_none(row.get("value_ratio_20")),
                    "CHDM50": _round_or_none(row.get("CHDM50")),
                    "DS20": _round_or_none(row.get("DS20")),
                }
            )
    return rows


def _action_rows(candidates: list[dict[str, Any]], *, max_positions: int, market_state: str | None) -> list[dict[str, Any]]:
    normalized_state = str(market_state or "UNKNOWN").upper()
    rows = []
    for idx, candidate in enumerate(candidates, start=1):
        production_decision = "PAPER_BUY_CANDIDATE" if idx <= max_positions else "WATCHLIST_ONLY"
        risk_label = "HIGH_RISK_REGIME" if normalized_state in {"RISK_OFF", "CRASH", "PANIC"} else "NORMAL_RISK_REVIEW"
        rows.append(
            {
                "priority": idx,
                "symbol": candidate["symbol"],
                "action": production_decision,
                "execution_plan": "next_session_open_or_limit_after_manual_approval",
                "requires_manual_approval": True,
                "max_positions_context": max_positions,
                "strategy_name": candidate.get("strategy_name"),
                "edge_rank_score": candidate.get("edge_rank_score"),
                "mkt_regime_state": market_state or candidate.get("mkt_regime_state"),
                "risk_label": risk_label,
                "gate_note": "baseline MVP does not hard-veto market regime; regime is an audit/risk label",
                "note": (
                    "Dry-run only: no broker order is sent. Baseline MVP ranks candidates, "
                    "allocates shared slots, then relies on exit/risk management."
                ),
            }
        )
    return rows


def _is_paper_buy_action(action: Any) -> bool:
    return str(action or "").startswith("PAPER_BUY_CANDIDATE")


def _write_demo_script(
    *,
    out_dir: Path,
    as_of: pd.Timestamp,
    universe: str,
    max_positions: int,
    candidate_count: int,
) -> None:
    content = f"""# Production Demo Dry-Run

## Mục tiêu demo

Chứng minh hệ thống không chỉ backtest, mà có pipeline production dry-run:

1. Nạp dữ liệu OHLCV local mới nhất.
2. Tính feature và market regime chỉ bằng dữ liệu đến ngày `{as_of:%Y-%m-%d}`.
3. Chạy sleeve production `core_mvp` gồm 9 strategy đã validate.
4. Sinh danh sách tín hiệu cho phiên giao dịch kế tiếp.
5. Tách rõ `PAPER_BUY_CANDIDATE` và `WATCHLIST_ONLY`; không gửi lệnh thật trong demo.

## Tham số demo

- Universe: `{universe}`
- As-of close: `{as_of:%Y-%m-%d}`
- Active sleeve: `core_mvp`
- Max positions: `{max_positions}`
- Candidate count: `{candidate_count}`
- Mode: `paper_trading_dry_run`
- Broker order: `disabled`
- Production baseline: không hard-veto theo `RISK_OFF`; market regime chỉ là risk label/audit context vì backtest MVP +98.48% không dùng hard gate này.

## File cần mở khi demo với thầy

- `production_status.json`: trạng thái hệ thống, ngày dữ liệu, sleeve active, cơ chế an toàn.
- `market_snapshot.json`: ảnh chụp trạng thái thị trường tại ngày as-of.
- `production_signals.csv`: tín hiệu đã pass filter, đã xếp hạng.
- `strategy_votes.csv`: mã nào được strategy nào vote, dùng để giải thích transparency.
- `candidate_actions.csv`: hành động mô phỏng baseline cho phiên kế tiếp.
- `strategy_catalog.csv`: toàn bộ strategy được expose cho screener và strategy đang dùng cho portfolio.

## Câu nói demo ngắn

Backtest dùng để chọn hệ thống. Production dry-run dùng để trả lời: nếu hôm nay đóng cửa,
hệ thống baseline chuẩn bị mua gì vào phiên tới và vì sao mua. Demo này cố ý disable broker order
để tránh nhầm giữa kiểm thử và vận hành vốn thật.

Market regime vẫn được hiển thị để audit rủi ro, nhưng không được trình bày như hard gate vì hard gate
đã backtest kém hơn baseline.

## Lưu ý không look-ahead

Tín hiệu chỉ dùng feature có `feature_date = {as_of:%Y-%m-%d}` hoặc trước đó. Hành động là
cho phiên kế tiếp, không dùng giá tương lai để ra quyết định.
"""
    (out_dir / "PRODUCTION_DEMO_SCRIPT.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create production dry-run demo artifacts.")
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--as-of", default="", help="YYYY-MM-DD. Defaults to latest local data date.")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--max-candidates", type=int, default=15)
    parser.add_argument("--max-positions", type=int, default=0, help="Override core MVP max positions. 0 uses sleeve default.")
    parser.add_argument("--lookback-days", type=int, default=0, help="Also export rolling dry-run signals for recent calendar days.")
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else _latest_data_date()
    start_date = as_of - pd.Timedelta(days=max(0, args.lookback_days))
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "reports" / f"production_demo_{datetime.now():%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(args.universe)
    hypothesis_map = _load_hypothesis_map()
    missing = [name for name in MVP_EDGE_STRATEGIES if name not in hypothesis_map]
    if missing:
        raise RuntimeError(f"Missing production hypotheses: {missing}")
    hypotheses = [hypothesis_map[name] for name in MVP_EDGE_STRATEGIES]

    feature_result = build_feature_table(
        universe=symbols,
        start=start_date.strftime("%Y-%m-%d"),
        end=as_of.strftime("%Y-%m-%d"),
        root=ROOT,
    )
    features = feature_result[0] if isinstance(feature_result, tuple) else feature_result
    features = enrich_features_with_rotation(features, ROOT)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    latest_features = features[features["date"] == as_of].reset_index(drop=True)
    if latest_features.empty:
        available = features["date"].max()
        raise RuntimeError(f"No features for as-of {as_of:%Y-%m-%d}. Latest available feature date: {available:%Y-%m-%d}")

    signal_cache = combo_harness.build_signal_cache_for_combo(
        features=features,
        hypotheses_list=hypotheses,
        combo_label="PRODUCTION_CORE_MVP",
        clean_symbols=symbols,
    )
    day_signals = signal_cache.get(as_of, {})
    candidates = _candidate_rows(day_signals, args.max_candidates)

    active_sleeves = active_portfolio_sleeves(include_shadow=False)
    core_sleeve = next(item for item in active_sleeves if item.sleeve_id == "core_mvp")
    max_positions = int(args.max_positions) if int(args.max_positions) > 0 else core_sleeve.max_positions
    market = _market_snapshot(features, as_of)
    market_state = market.get("mkt_regime_state") if isinstance(market, dict) else None
    actions = _action_rows(
        candidates,
        max_positions=max_positions,
        market_state=market_state,
    )
    votes = _vote_rows(latest_features=latest_features, hypotheses=hypotheses, as_of=as_of)

    pd.DataFrame(candidates).to_csv(out_dir / "production_signals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(actions).to_csv(out_dir / "candidate_actions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(votes).to_csv(out_dir / "strategy_votes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(catalog_rows()).to_csv(out_dir / "strategy_catalog.csv", index=False, encoding="utf-8-sig")

    rolling_signal_rows: list[dict[str, Any]] = []
    rolling_action_rows: list[dict[str, Any]] = []
    rolling_summary_rows: list[dict[str, Any]] = []
    if args.lookback_days > 0:
        dates = [
            date
            for date in sorted(signal_cache)
            if start_date <= pd.Timestamp(date).normalize() <= as_of
        ]
        for date in dates:
            day_candidates = _candidate_rows(signal_cache.get(date, {}), args.max_candidates)
            day_market = _market_snapshot(features, pd.Timestamp(date).normalize())
            day_market_state = day_market.get("mkt_regime_state") if isinstance(day_market, dict) else None
            day_actions = _action_rows(
                day_candidates,
                max_positions=max_positions,
                market_state=day_market_state,
            )
            for row in day_candidates:
                rolling_signal_rows.append({"date": pd.Timestamp(date).strftime("%Y-%m-%d"), **row})
            for row in day_actions:
                rolling_action_rows.append({"date": pd.Timestamp(date).strftime("%Y-%m-%d"), **row})
            rolling_summary_rows.append(
                {
                    "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "mkt_regime_state": day_market_state,
                    "mkt_regime_score": day_market.get("mkt_regime_score") if isinstance(day_market, dict) else None,
                    "signal_count": len(day_candidates),
                    "paper_buy_candidate_count": sum(1 for item in day_actions if _is_paper_buy_action(item.get("action"))),
                    "vetoed_candidate_count": sum(
                        1 for item in day_actions if str(item.get("action", "")).startswith("VETO_")
                    ),
                    "symbols": ",".join(item["symbol"] for item in day_candidates),
                }
            )
        pd.DataFrame(rolling_signal_rows).to_csv(
            out_dir / f"rolling_signals_{args.lookback_days}d.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(rolling_action_rows).to_csv(
            out_dir / f"rolling_candidate_actions_{args.lookback_days}d.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(rolling_summary_rows).to_csv(
            out_dir / f"rolling_summary_{args.lookback_days}d.csv",
            index=False,
            encoding="utf-8-sig",
        )

    status = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "paper_trading_dry_run",
        "live_orders_enabled": False,
        "broker_integration_required_for_real_orders": True,
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "next_session_assumption": "Use generated candidates for the next trading session after as-of close.",
        "universe": args.universe,
        "symbols": len(symbols),
        "active_sleeve": {**core_sleeve.__dict__, "max_positions": max_positions},
        "production_strategy_count": len(MVP_EDGE_STRATEGIES),
        "candidate_count": len(candidates),
        "paper_buy_candidate_count": sum(1 for item in actions if _is_paper_buy_action(item.get("action"))),
        "vetoed_candidate_count": sum(1 for item in actions if str(item.get("action", "")).startswith("VETO_")),
        "lookback_days": args.lookback_days,
        "lookback_start_date": start_date.strftime("%Y-%m-%d") if args.lookback_days > 0 else None,
        "safety_notes": [
            "No broker order is sent by this dry-run.",
            "Signals use only same-day close features and are intended for the next session.",
            "Final production mode still needs persistent portfolio state, broker adapter, and order audit log.",
        ],
    }
    (out_dir / "production_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (out_dir / "market_snapshot.json").write_text(
        json.dumps(market, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_demo_script(
        out_dir=out_dir,
        as_of=as_of,
        universe=args.universe,
        max_positions=max_positions,
        candidate_count=len(candidates),
    )

    print(json.dumps(status, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
