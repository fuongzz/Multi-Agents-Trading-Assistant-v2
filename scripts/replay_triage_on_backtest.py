"""Replay the Phase 3a triage gate against an existing core3 backtest.

For each fill in trades.csv:
    1. Reconstruct a minimal EvidencePacket at signal_date from the feature
       table + smart_money_trace parquet.
    2. Run RuleBasedTriageGate.review().
    3. Record the action and counterfactual PnL.

Then compute portfolio-level metrics for two scenarios:
    * core3_pure   — original baseline
    * core3_gated  — apply gate: SKIP → drop trade, REDUCE_SIZE → scale pnl,
                     WAIT_FOR_ENTRY → drop trade (re-evaluate next session).

Usage:
    python scripts/replay_triage_on_backtest.py \
        --trades backtest_results/mvp_intraday_cached/VN100_vn100_ytd2025_core3_mvp_no_sector_smt_top5_2025-01-01_2026-05-12_top5_eod_next_open_trades.csv \
        --out backtest_results/triage_shadow/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from multiagents_trading_assistant.agentic.schemas import (
    EvidencePacket,
    StrategySignal,
)
from multiagents_trading_assistant.agentic.triage_gate import (
    GateThresholds,
    RuleBasedTriageGate,
)


# ── Data loaders ────────────────────────────────────────────────────────────

def load_parquets(root: Path) -> dict[str, pd.DataFrame]:
    """Load reference parquets needed to rebuild evidence at signal_date."""
    paths = {
        "ohlcv": root / "multiagents_trading_assistant/data/ohlcv_master.parquet",
        "smt": root / "data/research/smart_money_trace/smart_money_by_symbol.parquet",
        "chdm": root / "data/research/money_cycle/chdm_by_symbol.parquet",
        "ds": root / "data/research/money_cycle/ds_by_symbol.parquet",
        "market": root / "data/research/money_cycle/money_cycle_market.parquet",
    }
    out: dict[str, pd.DataFrame] = {}
    for name, p in paths.items():
        if not p.exists():
            print(f"[warn] missing parquet: {p}")
            continue
        df = pd.read_parquet(p)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str).str.upper()
        out[name] = df
    return out


# ── Evidence reconstruction ─────────────────────────────────────────────────

def _row_le(df: pd.DataFrame, symbol: str, as_of: pd.Timestamp) -> dict:
    sub = df[(df["symbol"] == symbol) & (df["date"] <= as_of)]
    if sub.empty:
        return {}
    return sub.sort_values("date").iloc[-1].to_dict()


def _row_market_le(df: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    sub = df[df["date"] <= as_of]
    if sub.empty:
        return {}
    return sub.sort_values("date").iloc[-1].to_dict()


def rebuild_evidence(
    symbol: str,
    signal_date: str,
    setup_type: str,
    parquets: dict[str, pd.DataFrame],
) -> EvidencePacket:
    as_of = pd.Timestamp(signal_date).normalize()
    symbol = symbol.upper()

    ohlcv_row = _row_le(parquets["ohlcv"], symbol, as_of) if "ohlcv" in parquets else {}
    smt_row = _row_le(parquets["smt"], symbol, as_of) if "smt" in parquets else {}
    mkt_row = _row_market_le(parquets["market"], as_of) if "market" in parquets else {}

    # Compute 1-day change from OHLCV
    day_chg_pct = None
    if "ohlcv" in parquets:
        sub = parquets["ohlcv"][
            (parquets["ohlcv"]["symbol"] == symbol)
            & (parquets["ohlcv"]["date"] <= as_of)
        ].sort_values("date").tail(2)
        if len(sub) == 2:
            prev = float(sub.iloc[0]["close"])
            curr = float(sub.iloc[1]["close"])
            day_chg_pct = (curr - prev) / prev * 100.0 if prev else None

    # OHLCV / SMT `value` is in kVND (price × volume where price is in 1000-VND).
    # Convert to raw VND so gate threshold (10B raw VND) is consistent.
    value_ma20_kvnd = smt_row.get("value_ma20")
    avg_value_vnd = float(value_ma20_kvnd) * 1000.0 if value_ma20_kvnd else None
    if avg_value_vnd is None:
        v = ohlcv_row.get("value")
        avg_value_vnd = float(v) * 1000.0 if v else None
    volume_ratio_20 = smt_row.get("volume_ratio_20")
    value_ratio_20 = smt_row.get("value_ratio_20")
    rs_pct = smt_row.get("rs_percentile_20")
    sector_lead = smt_row.get("sector_leadership_score")

    # rs_percentile_20 may be on [0,1]
    if rs_pct is not None and rs_pct <= 1.0:
        rs_pct = rs_pct * 100.0

    # Determine market regime
    vni_chg = None
    if isinstance(mkt_row, dict):
        for k in ("vni_change_pct", "mkt_vni_change_pct", "mkt_ret_1d"):
            if k in mkt_row and mkt_row[k] is not None:
                v = float(mkt_row[k])
                vni_chg = v * 100.0 if abs(v) < 1.0 else v
                break
    regime = "UPTREND"
    if vni_chg is not None and vni_chg <= -1.5:
        regime = "DOWNTREND"
    elif vni_chg is not None and -1.5 < vni_chg < -0.5:
        regime = "SIDEWAY"

    indicators_block = {
        "stock_day_change_pct": day_chg_pct,
        "volume_ratio_20": volume_ratio_20,
        "value_ratio_20": value_ratio_20,
        "avg_value_20": avg_value_vnd,
    }
    # Drop None
    indicators_block = {k: v for k, v in indicators_block.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}

    return EvidencePacket(
        symbol=symbol,
        as_of_date=signal_date,
        market={
            "context": {
                "reference_trend": regime,
                "trend": regime,
                "vni_change_pct": vni_chg if vni_chg is not None else 0.0,
            },
            "money_cycle_market": {
                "mkt_regime": str(mkt_row.get("mkt_regime") or ""),
            } if mkt_row else {},
        },
        sector={
            k: v
            for k, v in {
                "rs_percentile_20": rs_pct,
                "sector_leadership_score": sector_lead,
                "smart_money_state": smt_row.get("smart_money_state"),
            }.items()
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        },
        technical={
            "indicators": indicators_block,
        },
        money_flow={},
        edge={"passed": True, "setup_type": setup_type},
        fundamental={},
        news={},  # not available in replay (would leak future news)
        portfolio={"has_position": False},  # replay assumes greenfield
        data_quality={
            "missing_fields": [],
            "data_quality_ok": True,
            "skipped_in_backtest": ["news", "fundamental", "portfolio", "foreign_flow", "global_macro"],
        },
    )


# ── Replay loop ─────────────────────────────────────────────────────────────

@dataclass
class GatedTrade:
    original: dict
    action: str
    reason_codes: list[str]
    size_multiplier: float
    rationale: str
    gated_pnl: float
    gated_pnl_pct: float
    kept: bool


def replay(
    trades_df: pd.DataFrame,
    parquets: dict[str, pd.DataFrame],
    thresholds: GateThresholds | None = None,
) -> list[GatedTrade]:
    gate = RuleBasedTriageGate(thresholds=thresholds)
    out: list[GatedTrade] = []

    for _, trade in trades_df.iterrows():
        symbol = str(trade["symbol"]).upper()
        signal_date = str(trade["signal_date"])
        setup_type = str(trade["setup_type"])

        try:
            ev = rebuild_evidence(symbol, signal_date, setup_type, parquets)
            signal = StrategySignal(
                symbol=symbol,
                signal_date=signal_date,
                as_of_date=signal_date,
                strategy_name=str(trade.get("edge_strategy_name") or "unknown"),
                setup_type=setup_type,
            )
            review = gate.review(signal, ev)
        except Exception as exc:
            # Fail-safe: any error → APPROVE (gate is non-blocking)
            print(f"[warn] gate error for {symbol}@{signal_date}: {exc}")
            out.append(GatedTrade(
                original=trade.to_dict(),
                action="APPROVE",
                reason_codes=[],
                size_multiplier=1.0,
                rationale=f"gate_error:{exc}",
                gated_pnl=float(trade["pnl"]),
                gated_pnl_pct=float(trade["pnl_pct"]),
                kept=True,
            ))
            continue

        kept = review.action in ("APPROVE", "REDUCE_SIZE")
        size_mult = review.size_multiplier if review.action == "REDUCE_SIZE" else 1.0
        size_mult = size_mult or 1.0

        if not kept:
            gated_pnl = 0.0
            gated_pnl_pct = 0.0
        else:
            gated_pnl = float(trade["pnl"]) * size_mult
            gated_pnl_pct = float(trade["pnl_pct"])  # pct doesn't change with sizing

        out.append(GatedTrade(
            original=trade.to_dict(),
            action=review.action,
            reason_codes=[r.value for r in review.reason_codes],
            size_multiplier=size_mult,
            rationale=review.rationale,
            gated_pnl=gated_pnl,
            gated_pnl_pct=gated_pnl_pct,
            kept=kept,
        ))

    return out


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame, pnl_col: str = "pnl",
                    pnl_pct_col: str = "pnl_pct", capital: float = 1e8) -> dict:
    if trades.empty:
        return {"return_pct": 0, "sharpe": 0, "max_dd_pct": 0,
                "win_rate_pct": 0, "n_trades": 0, "total_pnl": 0}

    pnl = trades[pnl_col].astype(float)
    pnl_pct = trades[pnl_pct_col].astype(float)
    total_pnl = pnl.sum()
    return_pct = total_pnl / capital * 100.0
    win_rate = (pnl > 0).mean() * 100.0

    # Daily equity from entry/exit -- approximate via exit_date cumulation
    t = trades.copy()
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    daily = t.groupby("exit_date")[pnl_col].sum().sort_index()
    equity = daily.cumsum() + capital
    if len(equity) > 1:
        rets = equity.pct_change().dropna()
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
        peak = equity.cummax()
        dd = (equity - peak) / peak * 100.0
        max_dd = float(dd.min())
    else:
        sharpe = 0.0
        max_dd = 0.0

    losers = pnl[pnl < 0].abs().sum()
    winners = pnl[pnl > 0].sum()
    pf = float(winners / losers) if losers > 0 else float("inf")

    return {
        "n_trades": int(len(trades)),
        "win_rate_pct": round(win_rate, 2),
        "return_pct": round(return_pct, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd, 2),
        "total_pnl": round(total_pnl, 0),
        "profit_factor": round(pf, 2) if pf != float("inf") else None,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True, help="Path to trades.csv from core3 backtest")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--root", default=".", help="Repo root (for parquet paths)")
    parser.add_argument("--capital", type=float, default=1e8, help="Notional capital for return calc")
    parser.add_argument("--gap-threshold", type=float, default=5.0,
                        help="OVEREXTENDED_GAP %% threshold (default 5.0)")
    parser.add_argument("--vol-ratio-threshold", type=float, default=0.5,
                        help="BAD_LIQUIDITY vol_ratio_20 threshold (default 0.5)")
    parser.add_argument("--avg-value-min-bn", type=float, default=10.0,
                        help="BAD_LIQUIDITY avg_value_20 minimum (B VND)")
    parser.add_argument("--label", default="default", help="Output sub-label")
    args = parser.parse_args()

    trades_df = pd.read_csv(args.trades)
    root = Path(args.root)
    parquets = load_parquets(root)

    thresholds = GateThresholds(
        overextended_gap_pct=args.gap_threshold,
        volume_ratio_min=args.vol_ratio_threshold,
        avg_value_min_bn=args.avg_value_min_bn,
    )
    print(f"[replay] Loaded {len(trades_df)} trades. Running gate ({args.label})...")
    print(f"         thresholds: gap>={args.gap_threshold}%, vol_ratio<{args.vol_ratio_threshold}, avg_value<{args.avg_value_min_bn}B")
    gated = replay(trades_df, parquets, thresholds=thresholds)

    # Build gated trades DataFrame
    rows = []
    for g in gated:
        row = dict(g.original)
        row["gate_action"] = g.action
        row["gate_reasons"] = ",".join(g.reason_codes)
        row["gate_size_mult"] = g.size_multiplier
        row["gate_rationale"] = g.rationale
        row["gated_pnl"] = g.gated_pnl
        row["gated_pnl_pct"] = g.gated_pnl_pct
        row["kept"] = g.kept
        rows.append(row)
    gated_df = pd.DataFrame(rows)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gated_csv = out_dir / "gated_trades.csv"
    gated_df.to_csv(gated_csv, index=False)

    # Action histogram
    print("\n=== Gate action distribution ===")
    print(gated_df["gate_action"].value_counts().to_string())

    print("\n=== Reason code distribution ===")
    all_reasons = gated_df[gated_df["gate_action"] != "APPROVE"]["gate_reasons"].str.split(",").explode()
    print(all_reasons.value_counts().to_string() if not all_reasons.empty else "(no triggers)")

    # Metrics: pure vs gated
    pure = compute_metrics(gated_df, "pnl", "pnl_pct", capital=args.capital)
    kept_df = gated_df[gated_df["kept"]]
    gated_metrics = compute_metrics(kept_df, "gated_pnl", "gated_pnl_pct", capital=args.capital)

    # False skip: trades that would have profited but were skipped
    skipped = gated_df[~gated_df["kept"]]
    false_skips = skipped[skipped["pnl"] > 0]
    saved_skips = skipped[skipped["pnl"] <= 0]

    setup_breakdown = {}
    for st in gated_df["setup_type"].unique():
        sub_pure = gated_df[gated_df["setup_type"] == st]
        sub_gated = kept_df[kept_df["setup_type"] == st]
        setup_breakdown[st] = {
            "pure": compute_metrics(sub_pure, "pnl", "pnl_pct", capital=args.capital),
            "gated": compute_metrics(sub_gated, "gated_pnl", "gated_pnl_pct", capital=args.capital),
            "skipped_count": int((gated_df["setup_type"] == st).sum() - (kept_df["setup_type"] == st).sum()),
        }

    summary = {
        "trades_path": args.trades,
        "n_trades_pure": int(len(gated_df)),
        "n_trades_gated_kept": int(len(kept_df)),
        "n_skipped": int(len(skipped)),
        "n_reduce_size": int((gated_df["gate_action"] == "REDUCE_SIZE").sum()),
        "n_wait_for_entry": int((gated_df["gate_action"] == "WAIT_FOR_ENTRY").sum()),
        "n_skip": int((gated_df["gate_action"] == "SKIP").sum()),
        "false_skip_count": int(len(false_skips)),
        "false_skip_pnl_pct_sum": round(false_skips["pnl_pct"].sum() * 100.0, 2) if not false_skips.empty else 0.0,
        "saved_skip_count": int(len(saved_skips)),
        "saved_skip_pnl": round(saved_skips["pnl"].sum(), 0) if not saved_skips.empty else 0.0,
        "pure": pure,
        "gated": gated_metrics,
        "delta": {
            "return_pct": round(gated_metrics["return_pct"] - pure["return_pct"], 2),
            "sharpe": round(gated_metrics["sharpe"] - pure["sharpe"], 3),
            "max_dd_pct": round(gated_metrics["max_dd_pct"] - pure["max_dd_pct"], 2),
            "win_rate_pct": round(gated_metrics["win_rate_pct"] - pure["win_rate_pct"], 2),
        },
        "setup_breakdown": setup_breakdown,
    }

    summary_path = out_dir / "shadow_comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # Pretty print headline
    print("\n=== Shadow comparison ===")
    print(f"core3 pure        return={pure['return_pct']:+.2f}% sharpe={pure['sharpe']:.2f} dd={pure['max_dd_pct']:+.2f}% wr={pure['win_rate_pct']:.1f}% n={pure['n_trades']}")
    print(f"core3 + AI gate   return={gated_metrics['return_pct']:+.2f}% sharpe={gated_metrics['sharpe']:.2f} dd={gated_metrics['max_dd_pct']:+.2f}% wr={gated_metrics['win_rate_pct']:.1f}% n={gated_metrics['n_trades']}")
    print(f"Delta                 return={summary['delta']['return_pct']:+.2f}pp sharpe={summary['delta']['sharpe']:+.2f} dd={summary['delta']['max_dd_pct']:+.2f}pp")
    print(f"False skips: {summary['false_skip_count']} ({summary['false_skip_pnl_pct_sum']:.1f}% of pnl)")
    print(f"Saved skips (avoided losers): {summary['saved_skip_count']} (pnl saved: {summary['saved_skip_pnl']:,.0f})")
    print(f"\nOutputs: {gated_csv}, {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
