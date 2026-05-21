"""Test vnstock Gold flow data as filters on the current shared-pool MVP.

Gold endpoints used:
- Market().equity(symbol).foreign_flow(start, end, limit)
- Market().equity(symbol).proprietary_flow(start, end, limit)
- Market().equity(symbol).trade_history(start, end, limit)

Signals remain causal: flow data on date T is merged into the feature row for
date T and can only generate orders for next open T+1 in the existing harness.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from scripts.backtest_position_exit_agent import run_with_exit_agent
from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "multiagents_trading_assistant" / "edge_lab" / "configs"
OUT_ROOT = ROOT / "backtest_results" / "vnstock_gold_flow_filters"
DATA_ROOT = ROOT / "data" / "research" / "vnstock_gold_flow"


COMBO_SET = {
    "name": "PURE_TOP3_PLUS_SMT_LIVE",
    "strategies": [
        "leader_pullback_market_regime_v3",
        "leader_pullback_market_healthy_v2",
        "breakout_55_smt_v1",
        "money_cycle_reset_smt_confirm",
        "smart_money_strong_market_healthy",
        "accumulation_breakout_smt_v1",
        "breakout_after_accumulation_v3",
        "compression_breakout_smt_v1",
        "mean_reversion_uptrend_ma50_v1",
    ],
}


def _load_hypothesis_map() -> dict[str, Hypothesis]:
    out: dict[str, Hypothesis] = {}
    for config in [
        Path(DEFAULT_CONFIG),
        CONFIG_DIR / "global_market_hypotheses.json",
        CONFIG_DIR / "sector_rotation_hypotheses.json",
        CONFIG_DIR / "qmv_ichimoku_turtle_hypotheses.json",
    ]:
        if not config.exists():
            continue
        for hyp in load_hypotheses(config):
            out[hyp.name] = hyp
    return out


def collect_gold_flow(symbols: list[str], start: str, end: str, *, limit: int, refresh: bool) -> pd.DataFrame:
    cache_dir = DATA_ROOT / f"vn100_{start}_{end}_limit{limit}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "gold_flow_features.parquet"
    if out_path.exists() and not refresh:
        return pd.read_parquet(out_path)

    from vnstock_data import Market

    rows: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []
    for idx, symbol in enumerate(symbols, 1):
        print(f"[gold-flow] {idx:03d}/{len(symbols):03d} {symbol}")
        equity = Market().equity(symbol)
        try:
            foreign = equity.foreign_flow(start=start, end=end, limit=limit)
        except Exception as exc:
            errors.append({"symbol": symbol, "endpoint": "foreign_flow", "error": f"{type(exc).__name__}: {exc}"})
            foreign = pd.DataFrame()
        try:
            prop = equity.proprietary_flow(start=start, end=end, limit=limit)
        except Exception as exc:
            errors.append({"symbol": symbol, "endpoint": "proprietary_flow", "error": f"{type(exc).__name__}: {exc}"})
            prop = pd.DataFrame()
        try:
            trade = equity.trade_history(start=start, end=end, limit=limit)
        except Exception as exc:
            errors.append({"symbol": symbol, "endpoint": "trade_history", "error": f"{type(exc).__name__}: {exc}"})
            trade = pd.DataFrame()

        frame = _build_symbol_flow_features(symbol, foreign, prop, trade)
        if not frame.empty:
            rows.append(frame)
        time.sleep(0.05)

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not combined.empty:
        combined.to_parquet(out_path, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(cache_dir / "errors.csv", index=False)
    return combined


def _build_symbol_flow_features(symbol: str, foreign: pd.DataFrame, prop: pd.DataFrame, trade: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    if not foreign.empty:
        f = foreign.copy()
        f["date"] = pd.to_datetime(f["trading_date"]).dt.normalize()
        f = f[["date", "net_val", "net_vol"]].rename(columns={
            "net_val": "foreign_net_val",
            "net_vol": "foreign_net_vol",
        })
        pieces.append(f)
    if not prop.empty:
        p = prop.copy()
        p["date"] = pd.to_datetime(p["trading_date"]).dt.normalize()
        p = p[["date", "net_val", "net_vol"]].rename(columns={
            "net_val": "prop_net_val",
            "net_vol": "prop_net_vol",
        })
        pieces.append(p)
    if not trade.empty:
        t = trade.copy()
        t["date"] = pd.to_datetime(t["trading_date"]).dt.normalize()
        keep = [
            "date", "total_value", "total_volume", "matched_value", "deal_value",
            "total_buy_unmatched_volume", "total_sell_unmatched_volume", "total_net_trade_volume",
            "average_buy_trade_volume", "average_sell_trade_volume",
        ]
        keep = [col for col in keep if col in t.columns]
        pieces.append(t[keep])
    if not pieces:
        return pd.DataFrame()
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on="date", how="outer")
    out["symbol"] = symbol
    out = out.sort_values("date").reset_index(drop=True)
    for col in ["foreign_net_val", "prop_net_val"]:
        if col in out.columns:
            out[f"{col}_5d"] = out[col].rolling(5, min_periods=2).sum()
            out[f"{col}_20d"] = out[col].rolling(20, min_periods=5).sum()
    total_value = out.get("total_value", pd.Series(0.0, index=out.index)).replace(0, pd.NA)
    total_volume = out.get("total_volume", pd.Series(0.0, index=out.index)).replace(0, pd.NA)
    if "foreign_net_val" in out.columns:
        out["foreign_net_val_ratio"] = out["foreign_net_val"] / total_value
        out["foreign_net_val_5d_ratio"] = out["foreign_net_val_5d"] / out["total_value"].rolling(5, min_periods=2).sum().replace(0, pd.NA)
    if "prop_net_val" in out.columns:
        out["prop_net_val_ratio"] = out["prop_net_val"] / total_value
        out["prop_net_val_5d_ratio"] = out["prop_net_val_5d"] / out["total_value"].rolling(5, min_periods=2).sum().replace(0, pd.NA)
    if "total_net_trade_volume" in out.columns:
        out["net_trade_pressure"] = out["total_net_trade_volume"] / total_volume
        out["net_trade_pressure_5d"] = out["total_net_trade_volume"].rolling(5, min_periods=2).sum() / out["total_volume"].rolling(5, min_periods=2).sum().replace(0, pd.NA)
    out["institutional_net_val_5d"] = out.get("foreign_net_val_5d", 0.0) + out.get("prop_net_val_5d", 0.0)
    out["institutional_net_buy_5d"] = out["institutional_net_val_5d"] > 0
    out["foreign_net_buy_5d"] = out.get("foreign_net_val_5d", 0.0) > 0
    out["prop_net_buy_5d"] = out.get("prop_net_val_5d", 0.0) > 0
    out["positive_pressure_5d"] = out.get("net_trade_pressure_5d", 0.0) > 0
    return out


def add_gold_flow_features(features: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    if flow.empty:
        return features.copy()
    flow = flow.copy()
    flow["date"] = pd.to_datetime(flow["date"]).dt.normalize()
    flow["symbol"] = flow["symbol"].astype(str).str.upper()
    keep = [
        "date", "symbol", "foreign_net_val_5d", "foreign_net_val_20d", "foreign_net_val_5d_ratio",
        "prop_net_val_5d", "prop_net_val_20d", "prop_net_val_5d_ratio",
        "institutional_net_val_5d", "institutional_net_buy_5d",
        "foreign_net_buy_5d", "prop_net_buy_5d",
        "net_trade_pressure_5d", "positive_pressure_5d",
    ]
    keep = [col for col in keep if col in flow.columns]
    return features.merge(flow[keep], on=["date", "symbol"], how="left")


def overlay_signal_cache(signal_cache: dict, features: pd.DataFrame, overlay: str) -> dict:
    if overlay == "none":
        return signal_cache
    cols = [
        "date", "symbol", "foreign_net_buy_5d", "prop_net_buy_5d", "institutional_net_buy_5d",
        "positive_pressure_5d", "foreign_net_val_5d_ratio", "prop_net_val_5d_ratio",
    ]
    available = [col for col in cols if col in features.columns]
    lookup = features[available].set_index(["date", "symbol"]).to_dict("index")
    out = {}
    for date, day in signal_cache.items():
        patched = {}
        for symbol, payload in day.items():
            current = dict(payload)
            if current.get("passed") and overlay.startswith("boost_"):
                row = lookup.get((pd.Timestamp(date).normalize(), symbol))
                current["edge_rank_score"] = round(
                    float(current.get("edge_rank_score") or 0.0) + _overlay_boost(row, overlay),
                    4,
                )
            elif current.get("passed"):
                row = lookup.get((pd.Timestamp(date).normalize(), symbol))
                if not _overlay_passes(row, overlay):
                    current["passed"] = False
                    current["filters_failed"] = [*(current.get("filters_failed") or []), f"gold_overlay:{overlay}"]
            patched[symbol] = current
        out[date] = patched
    return out


def _overlay_boost(row: dict[str, Any] | None, overlay: str) -> float:
    if row is None:
        return 0.0
    parts = overlay.split("_")
    try:
        weight = float(parts[-1].replace("p", "."))
    except Exception:
        weight = 0.75
    if overlay.startswith("boost_foreign_"):
        return weight if bool(row.get("foreign_net_buy_5d", False)) else 0.0
    if overlay.startswith("boost_prop_"):
        return weight if bool(row.get("prop_net_buy_5d", False)) else 0.0
    if overlay.startswith("boost_institutional_"):
        return weight if bool(row.get("institutional_net_buy_5d", False)) else 0.0
    if overlay.startswith("boost_pressure_"):
        return weight if bool(row.get("positive_pressure_5d", False)) else 0.0
    if overlay.startswith("boost_foreign_or_pressure_"):
        return weight if (
            bool(row.get("foreign_net_buy_5d", False)) or bool(row.get("positive_pressure_5d", False))
        ) else 0.0
    return 0.0


def _overlay_passes(row: dict[str, Any] | None, overlay: str) -> bool:
    if row is None:
        return False
    if overlay == "foreign_5d_positive":
        return bool(row.get("foreign_net_buy_5d", False))
    if overlay == "prop_5d_positive":
        return bool(row.get("prop_net_buy_5d", False))
    if overlay == "institutional_5d_positive":
        return bool(row.get("institutional_net_buy_5d", False))
    if overlay == "pressure_5d_positive":
        return bool(row.get("positive_pressure_5d", False))
    if overlay == "foreign_or_pressure":
        return bool(row.get("foreign_net_buy_5d", False)) or bool(row.get("positive_pressure_5d", False))
    if overlay == "institutional_or_pressure":
        return bool(row.get("institutional_net_buy_5d", False)) or bool(row.get("positive_pressure_5d", False))
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--label", default="gold_flow_filters_2025now")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.label}_{args.universe}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(args.universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, args.start, args.end)
    print("[gold-flow] collecting/loading flow features...")
    flow = collect_gold_flow(clean_symbols, args.start, args.end, limit=args.limit, refresh=args.refresh)
    print(f"[gold-flow] flow rows={len(flow):,} symbols={flow['symbol'].nunique() if not flow.empty else 0}")

    print("[gold-flow] building core features...")
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = add_gold_flow_features(features.sort_values(["date", "symbol"]).reset_index(drop=True), flow)

    hypotheses = _load_hypothesis_map()
    signal_cache_base = combo_harness.build_signal_cache_for_combo(
        features,
        [hypotheses[name] for name in COMBO_SET["strategies"]],
        COMBO_SET["name"],
        clean_symbols,
    )

    overlays = [
        "none",
        "boost_foreign_0p75",
        "boost_prop_0p75",
        "boost_institutional_0p75",
        "boost_pressure_0p75",
        "boost_foreign_or_pressure_0p75",
        "boost_foreign_1p5",
        "boost_prop_1p5",
        "boost_institutional_1p5",
        "boost_pressure_1p5",
        "boost_foreign_or_pressure_1p5",
        "foreign_5d_positive",
        "prop_5d_positive",
        "institutional_5d_positive",
        "pressure_5d_positive",
        "foreign_or_pressure",
        "institutional_or_pressure",
    ]
    rows = []
    for overlay in overlays:
        print(f"[gold-flow] overlay={overlay}")
        signal_cache = overlay_signal_cache(signal_cache_base, features, overlay)
        combo_harness.install_edge_cache(signal_cache)
        cfg = LivePipelineBacktestConfig(
            start_date=args.start,
            end_date=args.end,
            max_positions=args.max_positions,
            max_candidates_per_day=args.max_candidates_per_day,
            edge_strategy_name=COMBO_SET["name"],
        )
        result = run_with_exit_agent(universe_data, vnindex, features, cfg, enabled=True)
        metrics = result["metrics"]
        row = {
            "overlay": overlay,
            "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
            "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
            "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
            "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
            "number_of_trades": int(metrics.get("number_of_trades", 0)),
            "fills": int(metrics.get("fills", 0)),
        }
        rows.append(row)
        stem = f"{COMBO_SET['name']}_p{args.max_positions}_{overlay}"
        pd.DataFrame(result["trades"]).to_csv(out_dir / f"{stem}_trades.csv", index=False)
        if not result["equity_frame"].empty:
            result["equity_frame"].to_csv(out_dir / f"{stem}_equity.csv")
        print(
            f"  -> ret={row['total_return_pct']:+.2f}% sharpe={row['sharpe_ratio']:.2f} "
            f"mdd={row['max_drawdown_pct']:.2f}% wr={row['win_rate_pct']:.2f}% trades={row['number_of_trades']}"
        )

    summary = pd.DataFrame(rows).sort_values("total_return_pct", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"[gold-flow] saved {out_dir / 'summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
