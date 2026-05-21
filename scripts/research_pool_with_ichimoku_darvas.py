"""Test MVP shared pool with external Ichimoku and Darvas signals.

This is a portfolio-level screening test. External strategies compete for the
same max_positions as the MVP pool via the live signal cache. Exits are handled
by the existing live pipeline / ExitAgent risk path, not by the standalone
research runners' custom exits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

import scripts.backtest_combos_unbiased as combo_harness
from scripts.backtest_position_exit_agent import run_with_exit_agent
from scripts.research_darvas_box_strategies import (
    DarvasConfig,
    add_features as add_darvas_features,
    entry_signal as darvas_entry_signal,
    load_data as load_darvas_data,
)
from scripts.research_ichimoku_vn_strategies import (
    IchConfig,
    add_features as add_ich_features,
    entry_signal as ich_entry_signal,
    load_data as load_ich_data,
)
from multiagents_trading_assistant.backtest.live_pipeline import LivePipelineBacktestConfig
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import Hypothesis, load_hypotheses
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_CONFIG
from multiagents_trading_assistant.research.sector_rotation.enrich_features import enrich_features_with_rotation


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "multiagents_trading_assistant" / "edge_lab" / "configs"
GLOBAL_CONFIG = CONFIG_DIR / "global_market_hypotheses.json"
SECTOR_CONFIG = CONFIG_DIR / "sector_rotation_hypotheses.json"
OUT_ROOT = ROOT / "backtest_results" / "pool_with_ichimoku_darvas"

MVP_STRATEGIES = [
    "leader_pullback_market_regime_v3",
    "leader_pullback_market_healthy_v2",
    "breakout_55_smt_v1",
    "money_cycle_reset_smt_confirm",
    "smart_money_strong_market_healthy",
    "accumulation_breakout_smt_v1",
    "breakout_after_accumulation_v3",
    "compression_breakout_smt_v1",
    "mean_reversion_uptrend_ma50_v1",
]


def _load_hypothesis_map() -> dict[str, Hypothesis]:
    out: dict[str, Hypothesis] = {}
    for config in [Path(DEFAULT_CONFIG), GLOBAL_CONFIG, SECTOR_CONFIG]:
        for hyp in load_hypotheses(config):
            out[hyp.name] = hyp
    return out


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _rs_rank(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    rs = pd.concat(
        [frame[["date", "ret20"]].assign(symbol=symbol) for symbol, frame in frames.items()],
        ignore_index=True,
    )
    rs["rs_pct"] = rs.groupby("date")["ret20"].rank(pct=True)
    out = {}
    for symbol, frame in frames.items():
        out[symbol] = frame.merge(rs[rs["symbol"] == symbol][["date", "rs_pct"]], on="date", how="left")
    return out


def _attach_index(frame: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ["vni_uptrend", "vni_healthy", "vni_bull_phase", "drawdown_120"] if col in index.columns]
    return frame.merge(index[["date", *cols]], on="date", how="left")


def build_ich_signals(symbols: list[str], start: str, end: str, cfg: IchConfig, label: str) -> dict[pd.Timestamp, dict[str, dict[str, Any]]]:
    data, index = load_ich_data(symbols, start, end)
    frames = {symbol: add_ich_features(df, cfg) for symbol, df in data.items()}
    frames = _rs_rank(frames)
    frames = {symbol: _attach_index(frame, index) for symbol, frame in frames.items()}
    signals: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    for symbol, frame in frames.items():
        for row in frame.to_dict("records"):
            date = pd.Timestamp(row["date"]).normalize()
            if not ich_entry_signal(pd.Series(row), cfg):
                continue
            score = (
                1.0
                + float(row.get("rs_pct") or 0.0)
                + 0.20 * float(row.get("volume_ratio_20") or 0.0)
                - 0.20 * max(float(row.get("kijun_distance") or 0.0), 0.0)
            )
            signals.setdefault(date, {})[symbol] = {
                "strategy_name": label,
                "strategy_family": label,
                "description": label,
                "passed": True,
                "available": True,
                "feature_date": date.strftime("%Y-%m-%d"),
                "edge_rank": None,
                "edge_rank_score": round(score, 4),
                "edge_score": None,
                "rs_percentile_20": round(float(row.get("rs_pct") or 0.0), 4),
                "value_ratio_20": round(float(row.get("volume_ratio_20") or 0.0), 4),
                "setup_type": "EDGE_TREND",
                "filters_failed": [],
                "risk": {
                    "stop_loss": cfg.stop_loss,
                    "take_profit": None,
                    "initial_atr_stop_mult": 2.0,
                    "trailing_atr_mult": 3.0,
                    "trailing_profit_activation": 0.15,
                    "max_holding_bars": cfg.max_holding_bars,
                },
            }
    return signals


def build_darvas_signals(symbols: list[str], start: str, end: str, cfg: DarvasConfig, label: str) -> dict[pd.Timestamp, dict[str, dict[str, Any]]]:
    data, index = load_darvas_data(symbols, start, end)
    frames = {symbol: add_darvas_features(df, cfg) for symbol, df in data.items()}
    frames = _rs_rank(frames)
    frames = {symbol: _attach_index(frame, index) for symbol, frame in frames.items()}
    signals: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    for symbol, frame in frames.items():
        for row in frame.to_dict("records"):
            date = pd.Timestamp(row["date"]).normalize()
            if not darvas_entry_signal(pd.Series(row), cfg):
                continue
            score = (
                0.85
                + float(row.get("rs_pct") or 0.0)
                + 0.15 * float(row.get("volume_ratio_20") or 0.0)
                - 1.0 * float(row.get("box_width") or 0.0)
            )
            signals.setdefault(date, {})[symbol] = {
                "strategy_name": label,
                "strategy_family": label,
                "description": label,
                "passed": True,
                "available": True,
                "feature_date": date.strftime("%Y-%m-%d"),
                "edge_rank": None,
                "edge_rank_score": round(score, 4),
                "edge_score": None,
                "rs_percentile_20": round(float(row.get("rs_pct") or 0.0), 4),
                "value_ratio_20": round(float(row.get("volume_ratio_20") or 0.0), 4),
                "setup_type": "EDGE_BREAKOUT",
                "filters_failed": [],
                "risk": {
                    "stop_loss": cfg.stop_loss,
                    "take_profit": None,
                    "initial_atr_stop_mult": 2.0,
                    "trailing_atr_mult": cfg.trailing_atr_mult,
                    "trailing_profit_activation": cfg.trailing_activation,
                    "max_holding_bars": cfg.max_holding_bars,
                },
            }
    return signals


def merge_signal_caches(base: dict, overlays: list[dict]) -> dict:
    out = {
        date: {symbol: dict(payload) for symbol, payload in day.items()}
        for date, day in base.items()
    }
    for overlay in overlays:
        for date, day in overlay.items():
            target_day = out.setdefault(pd.Timestamp(date).normalize(), {})
            for symbol, payload in day.items():
                current = target_day.get(symbol)
                if current is None or not current.get("passed") or float(payload.get("edge_rank_score") or 0.0) > float(current.get("edge_rank_score") or 0.0):
                    target_day[symbol] = dict(payload)
    return out


def _row(name: str, max_positions: int, metrics: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    trades = int(metrics.get("number_of_trades", 0))
    weeks = max(1.0, (pd.Timestamp(end) - pd.Timestamp(start)).days / 7.0)
    return {
        "pool": name,
        "max_positions": max_positions,
        "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
        "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
        "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
        "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
        "number_of_trades": trades,
        "trades_per_week": round(trades / weeks, 2),
        "agent_reviews": int(metrics.get("agent_reviews", 0)),
        "fills": int(metrics.get("fills", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--label", default="pool_ich_darvas_2025now")
    parser.add_argument("--max-positions", default="5,7,10")
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.label}_{args.universe}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = combo_harness.resolve_symbols(args.universe)
    clean_symbols = sorted({symbol.upper().strip() for symbol in symbols})
    universe_data, vnindex = combo_harness.load_local_history(symbols, args.start, args.end)

    print("[pool-ext] building MVP features/cache...")
    features, _ = build_feature_table(symbols, args.start, args.end, root=ROOT)
    features = enrich_features_with_rotation(features.sort_values(["date", "symbol"]).reset_index(drop=True), root=ROOT)
    hypotheses = _load_hypothesis_map()
    base_cache = combo_harness.build_signal_cache_for_combo(
        features,
        [hypotheses[name] for name in MVP_STRATEGIES],
        "MVP",
        clean_symbols,
    )

    print("[pool-ext] building external signals...")
    ich_turtle = build_ich_signals(
        symbols,
        args.start,
        args.end,
        IchConfig(
            name="ich_turtle55_uptrend_rs50",
            tenkan=7,
            kijun=22,
            senkou_b=44,
            shift=22,
            entry="turtle55_ich",
            exit="cloud_close",
            vni_filter="uptrend",
            rs_min=0.50,
            market_dd_120_max=1.0,
            max_holding_bars=90,
        ),
        "ICH_TURTLE55",
    )
    ich_ladder = build_ich_signals(
        symbols,
        args.start,
        args.end,
        IchConfig(
            name="ich_cloud_break_ladder_bull_rs50_kumoq",
            tenkan=7,
            kijun=22,
            senkou_b=44,
            shift=22,
            entry="cloud_break",
            exit="ich_ladder",
            vni_filter="bull",
            rs_min=0.50,
            cloud_width_min=0.005,
            cloud_width_max=0.12,
            market_dd_120_max=1.0,
            max_holding_bars=70,
        ),
        "ICH_CLOUD_LADDER",
    )
    darvas = build_darvas_signals(
        symbols,
        args.start,
        args.end,
        DarvasConfig(
            name="darvas_w20_breakout_atr_uptrend_rs75_v1p2",
            box_window=20,
            entry="breakout",
            exit="atr_trail",
            vni_filter="uptrend",
            rs_min=0.75,
            volume_ratio_min=1.2,
            box_width_min=0.035,
            box_width_max=0.16,
            max_holding_bars=90,
        ),
        "DARVAS_W20",
    )
    print(
        f"[pool-ext] external signal days: turtle={len(ich_turtle)} "
        f"ladder={len(ich_ladder)} darvas={len(darvas)}"
    )

    pools = {
        "MVP": [],
        "MVP_PLUS_ICH_TURTLE": [ich_turtle],
        "MVP_PLUS_ICH_LADDER": [ich_ladder],
        "MVP_PLUS_DARVAS": [darvas],
        "MVP_PLUS_TURTLE_DARVAS": [ich_turtle, darvas],
        "MVP_PLUS_ALL": [ich_turtle, ich_ladder, darvas],
    }

    rows = []
    for pool_name, overlays in pools.items():
        signal_cache = merge_signal_caches(base_cache, overlays)
        combo_harness.install_edge_cache(signal_cache)
        for max_positions in _parse_ints(args.max_positions):
            cfg = LivePipelineBacktestConfig(
                start_date=args.start,
                end_date=args.end,
                max_positions=max_positions,
                max_candidates_per_day=args.max_candidates_per_day,
                edge_strategy_name=pool_name,
            )
            print(f"[pool-ext] {pool_name} p{max_positions}")
            result = run_with_exit_agent(universe_data, vnindex, features, cfg, enabled=True)
            row = _row(pool_name, max_positions, result["metrics"], args.start, args.end)
            rows.append(row)
            stem = f"{pool_name}_p{max_positions}"
            pd.DataFrame(result["trades"]).to_csv(out_dir / f"{stem}_trades.csv", index=False)
            if not result["equity_frame"].empty:
                result["equity_frame"].to_csv(out_dir / f"{stem}_equity.csv")
            pd.DataFrame(rows).sort_values(["max_positions", "total_return_pct"], ascending=[True, False]).to_csv(out_dir / "summary.csv", index=False)
            print(
                f"  -> ret={row['total_return_pct']:+.2f}% sharpe={row['sharpe_ratio']:.2f} "
                f"mdd={row['max_drawdown_pct']:.2f}% wr={row['win_rate_pct']:.2f}% trades={row['number_of_trades']}"
            )

    summary = pd.DataFrame(rows).sort_values(["max_positions", "total_return_pct"], ascending=[True, False])
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"[pool-ext] saved {out_dir}")


if __name__ == "__main__":
    main()
