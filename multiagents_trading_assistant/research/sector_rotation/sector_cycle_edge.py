"""Conditional edge study for money cycle and sector leadership.

The study is deliberately diagnostic, not a portfolio backtest:
- inputs are restricted to the requested universe;
- a signal is observed at close T;
- forward returns are measured close T to close T+h;
- rebalance sampling avoids counting every overlapping daily observation.

Use a production-like next-open execution engine before promoting a trading rule.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from multiagents_trading_assistant.research.money_cycle.compute_money_cycle import (
    compute_chdm_market,
    compute_ds_market,
    compute_money_cycle_market,
)

from .compute_sector_rotation import compute_sector_rotation

logger = logging.getLogger(__name__)

HORIZONS = (5, 10, 20)
ROOT = Path(__file__).resolve().parents[3]


def _market_cycle_state(row: pd.Series) -> str:
    chdm = row["CHDM50"]
    ds = row["DS20"]
    ds_delta = row["ds20_delta_5d"]
    if pd.isna(chdm) or pd.isna(ds):
        return "NEUTRAL"
    if chdm > 65 and ds_delta > 0.08:
        return "DISTRIBUTION"
    if chdm >= 50 and ds < 0.45:
        return "MARKUP"
    if chdm < 45 and ds > 0.55:
        return "MARKDOWN"
    if chdm < 35 and ds_delta < 0:
        return "ACCUMULATION"
    return "NEUTRAL"


def build_market_cycle_for_universe(
    chdm_by_symbol: pd.DataFrame,
    ds_by_symbol: pd.DataFrame,
    symbols: Iterable[str],
) -> pd.DataFrame:
    """Aggregate CHDM/DS within one named universe and classify its cycle."""
    symbol_set = {str(symbol).upper() for symbol in symbols}
    chdm = chdm_by_symbol.copy()
    ds = ds_by_symbol.copy()
    chdm["symbol"] = chdm["symbol"].astype(str).str.upper()
    ds["symbol"] = ds["symbol"].astype(str).str.upper()
    chdm = chdm[chdm["symbol"].isin(symbol_set)]
    ds = ds[ds["symbol"].isin(symbol_set)]
    windows = [
        int(column.removeprefix("CHDM"))
        for column in chdm.columns
        if column.startswith("CHDM")
    ]
    market = compute_money_cycle_market(
        compute_chdm_market(chdm, windows),
        compute_ds_market(ds, windows),
    ).sort_values("date")
    market["ds20_delta_5d"] = market["DS20"].diff(5)
    market["money_cycle_state"] = market.apply(_market_cycle_state, axis=1)
    return market


def add_forward_sector_returns(
    sector: pd.DataFrame,
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    """Attach future sector and excess returns using only trailing-return columns."""
    out = sector.sort_values(["sector_l1", "date"]).copy()
    grouped = out.groupby("sector_l1", sort=False)
    for horizon in horizons:
        sector_col = f"sector_ret_{horizon}d"
        market_col = f"mkt_ret_{horizon}d"
        out[f"fwd_ret_{horizon}d"] = grouped[sector_col].shift(-horizon)
        out[f"fwd_mkt_ret_{horizon}d"] = grouped[market_col].shift(-horizon)
        out[f"fwd_excess_{horizon}d"] = (
            out[f"fwd_ret_{horizon}d"] - out[f"fwd_mkt_ret_{horizon}d"]
        )
    return out.sort_values(["date", "sector_l1"]).reset_index(drop=True)


def sample_rebalance_dates(
    frame: pd.DataFrame,
    start: str,
    end: str,
    rebalance_days: int,
) -> pd.DataFrame:
    """Take observations every N trading sessions to reduce overlapping outcomes."""
    if rebalance_days <= 0:
        raise ValueError("rebalance_days must be positive")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    selected = frame[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)].copy()
    dates = sorted(selected["date"].drop_duplicates())
    sampled_dates = set(dates[::rebalance_days])
    return selected[selected["date"].isin(sampled_dates)].copy()


def summarize_states(
    sampled: pd.DataFrame,
    group_col: str,
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, group in sampled.groupby(group_col, dropna=False):
        for horizon in horizons:
            excess = group[f"fwd_excess_{horizon}d"].dropna()
            forward = group.loc[excess.index, f"fwd_ret_{horizon}d"]
            rows.append(
                {
                    "group_type": group_col,
                    "group": str(group_name),
                    "horizon_days": horizon,
                    "observations": int(len(excess)),
                    "avg_forward_return_pct": float(forward.mean() * 100.0) if len(excess) else np.nan,
                    "avg_excess_return_pct": float(excess.mean() * 100.0) if len(excess) else np.nan,
                    "median_excess_return_pct": float(excess.median() * 100.0) if len(excess) else np.nan,
                    "outperform_rate_pct": float((excess > 0).mean() * 100.0) if len(excess) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_leader_spread(
    sampled: pd.DataFrame,
    top_n: int = 2,
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    """Compare current RS leaders with laggards on the same signal dates."""
    rows: list[dict[str, object]] = []
    for date, day in sampled.groupby("date"):
        ranked = day.dropna(subset=["sector_rs_rank_20d"]).sort_values("sector_rs_rank_20d")
        if len(ranked) < top_n * 2:
            continue
        leaders = ranked.tail(top_n)
        laggards = ranked.head(top_n)
        for horizon in horizons:
            top_ret = leaders[f"fwd_excess_{horizon}d"].mean()
            bottom_ret = laggards[f"fwd_excess_{horizon}d"].mean()
            if pd.isna(top_ret) or pd.isna(bottom_ret):
                continue
            rows.append(
                {
                    "date": date,
                    "money_cycle_state": str(day["money_cycle_state"].iloc[0]),
                    "horizon_days": horizon,
                    "leaders_excess_pct": float(top_ret * 100.0),
                    "laggards_excess_pct": float(bottom_ret * 100.0),
                    "leader_minus_laggard_pct": float((top_ret - bottom_ret) * 100.0),
                }
            )
    observations = pd.DataFrame(rows)
    if observations.empty:
        return observations
    summary = (
        observations.groupby(["money_cycle_state", "horizon_days"], dropna=False)
        .agg(
            observations=("leader_minus_laggard_pct", "size"),
            leaders_excess_pct=("leaders_excess_pct", "mean"),
            laggards_excess_pct=("laggards_excess_pct", "mean"),
            leader_minus_laggard_pct=("leader_minus_laggard_pct", "mean"),
            spread_positive_rate_pct=("leader_minus_laggard_pct", lambda value: (value > 0).mean() * 100.0),
        )
        .reset_index()
    )
    all_regimes = (
        observations.groupby("horizon_days")
        .agg(
            observations=("leader_minus_laggard_pct", "size"),
            leaders_excess_pct=("leaders_excess_pct", "mean"),
            laggards_excess_pct=("laggards_excess_pct", "mean"),
            leader_minus_laggard_pct=("leader_minus_laggard_pct", "mean"),
            spread_positive_rate_pct=("leader_minus_laggard_pct", lambda value: (value > 0).mean() * 100.0),
        )
        .reset_index()
    )
    all_regimes.insert(0, "money_cycle_state", "ALL")
    return pd.concat([all_regimes, summary], ignore_index=True)


def _pct(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:+.2f}%"


def write_markdown_report(
    out_path: Path,
    *,
    universe: str,
    symbols_count: int,
    start: str,
    end: str,
    rebalance_days: int,
    state_summary: pd.DataFrame,
    spread_summary: pd.DataFrame,
    period_spread_summary: pd.DataFrame,
    latest: pd.DataFrame,
) -> None:
    sector_20 = state_summary[
        (state_summary["group_type"] == "sector_state")
        & (state_summary["horizon_days"] == 20)
    ].sort_values("avg_excess_return_pct", ascending=False)
    market_20 = state_summary[
        (state_summary["group_type"] == "money_cycle_state")
        & (state_summary["horizon_days"] == 20)
    ].sort_values("avg_excess_return_pct", ascending=False)
    spread_20 = spread_summary[
        spread_summary["horizon_days"] == 20
    ].sort_values("leader_minus_laggard_pct", ascending=False)
    all_spread_20 = spread_20[spread_20["money_cycle_state"] == "ALL"]
    lines = [
        f"# Money Cycle va Sector Cycle Edge Study - {universe.upper()}",
        "",
        "## Pham vi",
        "",
        f"- Universe: `{universe.upper()}` hien tai, {symbols_count} ma.",
        f"- Ky quan sat: `{start}` den `{end}`.",
        f"- Lay mau moi {rebalance_days} phien de giam trung lap ket qua forward.",
        "- Day la conditional study close-to-close, chua phai backtest khop lenh open T+1 va chua tru chi phi.",
        "- Universe hien tai ap cho lich su nen con survivorship bias.",
        "",
        "## Sector State Va Loi The Forward 20 Phien",
        "",
        "| Sector state | Mau | Loi nhuan forward TB | Excess so voi thi truong | Ty le outperform |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in sector_20.iterrows():
        lines.append(
            f"| {row['group']} | {int(row['observations'])} | "
            f"{_pct(row['avg_forward_return_pct'])} | {_pct(row['avg_excess_return_pct'])} | "
            f"{row['outperform_rate_pct']:.1f}% |"
        )
    lines.extend(["", "## Relative Strength Leader Spread", ""])
    if not all_spread_20.empty:
        row = all_spread_20.iloc[0]
        lines.extend(
            [
                f"Tren toan bo market cycle, top-2 nganh RS tru bottom-2 nganh RS o forward 20 phien: "
                f"**{_pct(row['leader_minus_laggard_pct'])}** "
                f"(duong o {row['spread_positive_rate_pct']:.1f}% mau, n={int(row['observations'])}).",
                "",
            ]
        )
    lines.extend(
        [
            "| Money-cycle state | Mau | Leader - laggard excess | Ty le spread duong |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in spread_20[spread_20["money_cycle_state"] != "ALL"].iterrows():
        lines.append(
            f"| {row['money_cycle_state']} | {int(row['observations'])} | "
            f"{_pct(row['leader_minus_laggard_pct'])} | {row['spread_positive_rate_pct']:.1f}% |"
        )
    period_20 = period_spread_summary[
        (period_spread_summary["money_cycle_state"] == "ALL")
        & (period_spread_summary["horizon_days"] == 20)
    ]
    lines.extend(
        [
            "",
            "## Kiem Tra Do Ben Theo Giai Doan",
            "",
            "| Giai doan | Mau | Leader - laggard excess 20 phien | Ty le spread duong |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in period_20.iterrows():
        lines.append(
            f"| {row['period']} | {int(row['observations'])} | "
            f"{_pct(row['leader_minus_laggard_pct'])} | {row['spread_positive_rate_pct']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Money Cycle Va Forward Sector Excess",
            "",
            "| Money-cycle state | Mau sector | Excess TB 20 phien | Ty le outperform |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in market_20.iterrows():
        lines.append(
            f"| {row['group']} | {int(row['observations'])} | "
            f"{_pct(row['avg_excess_return_pct'])} | {row['outperform_rate_pct']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Anh chup gan nhat",
            "",
            "| Nganh | State | RS rank 20d | CHDM50 | DS20 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in latest.head(6).iterrows():
        lines.append(
            f"| {row['sector_l1']} | {row['sector_state']} | "
            f"{row['sector_rs_rank_20d']:.2f} | {row['sector_chdm_50']:.1f} | "
            f"{row['sector_ds_20']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Ket luan nghien cuu",
            "",
            "- Neu leader spread duong va on dinh theo market cycle, sector leadership co the dung lam ranking/sizing overlay.",
            "- Neu state hard gate loai bo nhieu co hoi tot, chi dung no nhu canh bao rui ro thay vi dieu kien vao lenh bat buoc.",
            "- Buoc kiem dinh tiep theo la chay production-like open T+1 tren cung VN100 voi chi phi va gioi han tap trung nganh.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_study(
    *,
    symbols: list[str],
    universe: str,
    start: str,
    end: str,
    out_dir: Path,
    rebalance_days: int = 10,
) -> dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chdm_path = ROOT / "data" / "research" / "money_cycle" / "chdm_by_symbol.parquet"
    ds_path = ROOT / "data" / "research" / "money_cycle" / "ds_by_symbol.parquet"
    sector = compute_sector_rotation(
        ohlcv_path=ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet",
        chdm_path=chdm_path,
        ds_path=ds_path,
        out_path=out_dir / "sector_rotation.parquet",
        symbols=symbols,
    )
    chdm = pd.read_parquet(chdm_path)
    ds = pd.read_parquet(ds_path)
    market = build_market_cycle_for_universe(chdm, ds, symbols)
    market.to_parquet(out_dir / "money_cycle_market.parquet", index=False)
    prepared = add_forward_sector_returns(sector).merge(
        market[["date", "money_cycle_state", "CHDM50", "DS20"]],
        on="date",
        how="left",
    )
    sampled = sample_rebalance_dates(prepared, start, end, rebalance_days)
    state_summary = pd.concat(
        [
            summarize_states(sampled, "sector_state"),
            summarize_states(sampled, "money_cycle_state"),
        ],
        ignore_index=True,
    )
    spread_summary = summarize_leader_spread(sampled)
    periods = [
        ("full", start, end),
        ("pre_2025", start, min(end, "2024-12-31")),
        ("2025_now", max(start, "2025-01-01"), end),
    ]
    period_frames: list[pd.DataFrame] = []
    for label, period_start, period_end in periods:
        if period_start > period_end:
            continue
        period_sample = sample_rebalance_dates(prepared, period_start, period_end, rebalance_days)
        period_spread = summarize_leader_spread(period_sample)
        if not period_spread.empty:
            period_spread.insert(0, "period", label)
            period_frames.append(period_spread)
    period_spread_summary = pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()
    latest = (
        prepared[prepared["date"] == prepared["date"].max()]
        .sort_values("sector_rs_rank_20d", ascending=False)
        [["date", "sector_l1", "sector_state", "sector_rs_rank_20d", "sector_chdm_50", "sector_ds_20"]]
    )
    sampled.to_parquet(out_dir / "sampled_sector_observations.parquet", index=False)
    state_summary.to_csv(out_dir / "state_forward_summary.csv", index=False)
    spread_summary.to_csv(out_dir / "leader_spread_summary.csv", index=False)
    period_spread_summary.to_csv(out_dir / "leader_spread_period_summary.csv", index=False)
    latest.to_csv(out_dir / "latest_sector_snapshot.csv", index=False)
    write_markdown_report(
        out_dir / "README.md",
        universe=universe,
        symbols_count=len(symbols),
        start=start,
        end=end,
        rebalance_days=rebalance_days,
        state_summary=state_summary,
        spread_summary=spread_summary,
        period_spread_summary=period_spread_summary,
        latest=latest,
    )
    return {
        "state_summary": state_summary,
        "spread_summary": spread_summary,
        "period_spread_summary": period_spread_summary,
        "latest": latest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Study VN money cycle and sector rotation conditional edge")
    parser.add_argument("--universe", default="vn100", choices=["vn100"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-26")
    parser.add_argument("--rebalance-days", type=int, default=10)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "sector_cycle_edge_vn100_2020_to_2026-05-26",
    )
    args = parser.parse_args()
    from multiagents_trading_assistant.fetcher import get_vn100_symbols

    symbols = get_vn100_symbols()
    results = run_study(
        symbols=symbols,
        universe=args.universe,
        start=args.start,
        end=args.end,
        out_dir=args.out_dir,
        rebalance_days=args.rebalance_days,
    )
    print(f"[sector-cycle] symbols={len(symbols)} report={args.out_dir / 'README.md'}")
    print(results["spread_summary"].to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
