"""
Comparative backtest: 3 core strategies + 2 sector-rotation strategies vs VN-Index buy-hold.

Universe: VN100 + HNX30 + UPCOM30 (the "WIDE" pre-filtered to liquid in ohlcv_master).
Period:   default 2025-01-01 -> latest available date in master.

Reports per strategy: PnL, Return%, MDD, Sharpe, Profit Factor, Win Rate, # Trades.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from multiagents_trading_assistant.edge_lab.backtest import PortfolioConfig, run_portfolio
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import load_hypotheses
from multiagents_trading_assistant.edge_lab.metrics import equity_metrics, trade_metrics
from multiagents_trading_assistant.edge_lab.universe import get_universe
from multiagents_trading_assistant.research.sector_rotation.enrich_features import (
    enrich_features_with_rotation,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "multiagents_trading_assistant" / "edge_lab" / "configs"

CORE_CFG = CONFIGS / "vn30_money_smt_hypotheses.json"
ROTATION_CFG = CONFIGS / "sector_rotation_hypotheses.json"

# The 3 production core strategies (commit 47d785df). The JSON also holds
# legacy/research hypotheses; the runner only uses these by default.
CORE_NAMES = (
    "breakout_after_accumulation_v3",
    "mean_reversion_uptrend_ma20_v1",
    "mean_reversion_uptrend_ma50_v1",
)

logger = logging.getLogger("backtest_rotation")


def vnindex_buyhold(root: Path, start: pd.Timestamp, end: pd.Timestamp,
                    initial_capital: float) -> tuple[pd.DataFrame, dict]:
    idx_path = root / "multiagents_trading_assistant" / "data" / "index_master.parquet"
    if not idx_path.exists():
        return pd.DataFrame(), {"total_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    vni = pd.read_parquet(idx_path)
    vni["date"] = pd.to_datetime(vni["date"]).dt.tz_localize(None).dt.normalize()
    if "symbol" in vni.columns:
        vni = vni[vni["symbol"].astype(str).str.upper() == "VNINDEX"]
    vni = vni[(vni["date"] >= start) & (vni["date"] <= end)].sort_values("date")
    if vni.empty:
        return pd.DataFrame(), {"total_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}
    base = vni["close"].iloc[0]
    eq = pd.DataFrame({"date": vni["date"].to_numpy(),
                       "equity": initial_capital * vni["close"].to_numpy() / base})
    return eq, equity_metrics(eq, initial_capital)


def run_one(name: str, hypotheses, features, price_map, start, end, cfg):
    logger.info("Running %s ...", name)
    trades, equity = run_portfolio(features, price_map, hypotheses, start, end, cfg)
    em = equity_metrics(equity, cfg.initial_capital)
    tm = trade_metrics(trades)
    return {"name": name, **em, **tm, "_trades": trades, "_equity": equity}


def fmt_pct(x):
    return f"{x*100:7.2f}%" if pd.notna(x) else "    n/a"


def print_summary(rows: list[dict], initial_capital: float, bh_metrics: dict):
    print()
    print("=" * 108)
    print(f"{'Strategy':<32} {'PnL':>14} {'Return':>9} {'MDD':>9} {'Sharpe':>7} {'PF':>6} {'WinRt':>7} {'Trades':>7}")
    print("-" * 108)
    for r in rows:
        pnl = (r.get("final_equity") or initial_capital) - initial_capital
        print(f"{r['name']:<32} {pnl:>14,.0f} "
              f"{fmt_pct(r.get('total_return'))} "
              f"{fmt_pct(r.get('max_drawdown'))} "
              f"{r.get('sharpe', 0):>7.2f} "
              f"{r.get('profit_factor', 0):>6.2f} "
              f"{fmt_pct(r.get('win_rate', 0))} "
              f"{r.get('trades', 0):>7d}")
    print("-" * 108)
    if bh_metrics:
        pnl = initial_capital * bh_metrics.get("total_return", 0)
        print(f"{'VN-Index Buy & Hold':<32} {pnl:>14,.0f} "
              f"{fmt_pct(bh_metrics.get('total_return'))} "
              f"{fmt_pct(bh_metrics.get('max_drawdown'))} "
              f"{bh_metrics.get('sharpe', 0):>7.2f} "
              f"{'n/a':>6} {'n/a':>7} {'n/a':>7}")
    print("=" * 108)


def save_artifacts(rows: list[dict], out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in rows:
        safe_name = r["name"].replace("/", "_")
        trades = r["_trades"]
        equity = r["_equity"]
        if not trades.empty:
            trades.to_csv(out_dir / f"trades_{label}_{safe_name}.csv", index=False)
        if not equity.empty:
            equity.to_csv(out_dir / f"equity_{label}_{safe_name}.csv", index=False)
        summary.append({k: v for k, v in r.items() if not k.startswith("_")})
    pd.DataFrame(summary).to_csv(out_dir / f"summary_{label}.csv", index=False)
    logger.info("Artifacts saved -> %s", out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default=None, help="Default: latest available date in ohlcv_master")
    p.add_argument("--universe", default="WIDE")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--max-positions", type=int, default=8)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--out", default="backtest_results/sector_rotation")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # latest date from master
    ohlcv = pd.read_parquet(ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet",
                             columns=["date"])
    latest = pd.to_datetime(ohlcv["date"]).max().normalize()
    end_ts = pd.Timestamp(args.end) if args.end else latest
    start_ts = pd.Timestamp(args.start)
    logger.info("Period: %s -> %s", start_ts.date(), end_ts.date())

    symbols = get_universe(args.universe)
    logger.info("Universe %s: %d symbols", args.universe, len(symbols))

    logger.info("Building feature table...")
    features, price_map = build_feature_table(symbols, start_ts, end_ts, root=ROOT)
    logger.info("features rows=%d, price_map symbols=%d", len(features), len(price_map))

    logger.info("Enriching with sector_rotation + market_regime...")
    features = enrich_features_with_rotation(features, root=ROOT)

    all_core = load_hypotheses(CORE_CFG)
    core_hyps = [h for h in all_core if h.name in CORE_NAMES]
    rot_hyps = load_hypotheses(ROTATION_CFG)
    logger.info("Loaded %d core (of %d in file) + %d rotation hypotheses",
                len(core_hyps), len(all_core), len(rot_hyps))

    cfg = PortfolioConfig(
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
        top_n=args.top_n,
    )

    rows: list[dict] = []
    # Each core strategy individually
    for hyp in core_hyps:
        rows.append(run_one(f"core/{hyp.name}", [hyp], features, price_map, start_ts, end_ts, cfg))
    # Combined 3-core
    rows.append(run_one("core/COMBINED_3", core_hyps, features, price_map, start_ts, end_ts, cfg))
    # Rotation strategies individually
    for hyp in rot_hyps:
        rows.append(run_one(f"rotation/{hyp.name}", [hyp], features, price_map, start_ts, end_ts, cfg))
    # Combined rotation
    rows.append(run_one("rotation/COMBINED", rot_hyps, features, price_map, start_ts, end_ts, cfg))
    # All-in: core + rotation
    rows.append(run_one("ALL/core+rotation", core_hyps + rot_hyps, features, price_map, start_ts, end_ts, cfg))

    _, bh_metrics = vnindex_buyhold(ROOT, start_ts, end_ts, args.initial_capital)

    print_summary(rows, args.initial_capital, bh_metrics)
    out_dir = ROOT / args.out
    label = f"{start_ts.date()}_{end_ts.date()}"
    save_artifacts(rows, out_dir, label)


if __name__ == "__main__":
    main()
