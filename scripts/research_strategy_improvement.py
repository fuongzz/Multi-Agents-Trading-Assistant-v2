from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SINGLE_SUMMARY = ROOT / "backtest_results" / "all_strategies_unbiased_vn100_2025-01-01_2026-05-12" / "summary.csv"
DEFAULT_COMBO_YTD = ROOT / "backtest_results" / "combos_unbiased_vn100_all_combos_2025ytd_2025-01-01_2026-05-12"
DEFAULT_COMBO_4Y = ROOT / "backtest_results" / "combos_unbiased_vn100_top5_4year_2022-01-01_2026-05-12"
DEFAULT_OUT = ROOT / "reports" / "strategy_improvement_research.md"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):+.2f}%"


def _fmt_num(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def top_rows(summary: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return summary.sort_values(["total_return_pct", "sharpe_ratio"], ascending=False).head(n).reset_index(drop=True)


def trade_diagnostics(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if trades.empty:
        return {"by_strategy": pd.DataFrame(), "by_exit_reason": pd.DataFrame(), "worst_trades": pd.DataFrame()}

    frame = trades.copy()
    frame["pnl_pct"] = pd.to_numeric(frame.get("pnl_pct"), errors="coerce").fillna(0.0)
    frame["pnl"] = pd.to_numeric(frame.get("pnl"), errors="coerce").fillna(0.0)
    frame["is_win"] = frame["pnl"] > 0

    by_strategy = (
        frame.groupby("edge_strategy_name", dropna=False)
        .agg(
            trades=("symbol", "count"),
            win_rate_pct=("is_win", lambda s: round(float(s.mean()) * 100, 2)),
            avg_pnl_pct=("pnl_pct", lambda s: round(float(s.mean()) * 100, 2)),
            total_pnl=("pnl", "sum"),
        )
        .reset_index()
        .sort_values("total_pnl", ascending=False)
    )
    by_strategy["total_pnl"] = by_strategy["total_pnl"].round(0)

    by_exit = (
        frame.groupby("exit_reason", dropna=False)
        .agg(
            trades=("symbol", "count"),
            win_rate_pct=("is_win", lambda s: round(float(s.mean()) * 100, 2)),
            avg_pnl_pct=("pnl_pct", lambda s: round(float(s.mean()) * 100, 2)),
            total_pnl=("pnl", "sum"),
        )
        .reset_index()
        .sort_values("total_pnl", ascending=True)
    )
    by_exit["total_pnl"] = by_exit["total_pnl"].round(0)

    worst = frame.sort_values("pnl").head(10)[
        ["symbol", "edge_strategy_name", "signal_date", "entry_date", "exit_date", "exit_reason", "pnl_pct", "pnl"]
    ].copy()
    worst["pnl_pct"] = (worst["pnl_pct"] * 100).round(2)
    worst["pnl"] = worst["pnl"].round(0)

    return {"by_strategy": by_strategy, "by_exit_reason": by_exit, "worst_trades": worst}


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No data._"
    view = df[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in view.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, sep, *rows])


def build_report(
    *,
    single_summary_path: Path = DEFAULT_SINGLE_SUMMARY,
    combo_ytd_dir: Path = DEFAULT_COMBO_YTD,
    combo_4y_dir: Path = DEFAULT_COMBO_4Y,
) -> str:
    single = _read_csv(single_summary_path)
    combo_ytd = _read_csv(combo_ytd_dir / "summary.csv")
    combo_4y = _read_csv(combo_4y_dir / "summary.csv")

    top_single = top_rows(single, 10)
    top_ytd = top_rows(combo_ytd, 8)
    top_4y = top_rows(combo_4y, 8)

    best_combo = top_ytd.iloc[0] if not top_ytd.empty else None
    best_combo_name = str(best_combo["combo"]) if best_combo is not None else ""
    best_trades = _read_csv(combo_ytd_dir / f"{best_combo_name}_trades.csv") if best_combo_name else pd.DataFrame()
    diag = trade_diagnostics(best_trades)

    robust_name = top_4y.iloc[0]["combo"] if not top_4y.empty else "n/a"
    ytd_name = top_ytd.iloc[0]["combo"] if not top_ytd.empty else "n/a"

    lines = [
        "# Strategy Improvement Research",
        "",
        "## Guardrails",
        "",
        "- Use only `unbiased` / post-fix artifacts as research input.",
        "- Treat higher return from a small YTD sample as a candidate, not proof.",
        "- Promote a variant only after it improves train and does not degrade the 4-year check materially.",
        "- Keep entry execution next-bar and exit execution at trigger/gap levels; no same-bar-open exit fill.",
        "",
        "## Current Baseline",
        "",
        f"- Best YTD combo: `{ytd_name}` with {_fmt_pct(top_ytd.iloc[0]['total_return_pct']) if not top_ytd.empty else 'n/a'} return, "
        f"Sharpe {_fmt_num(top_ytd.iloc[0]['sharpe_ratio'], 3) if not top_ytd.empty else 'n/a'}, "
        f"WR {_fmt_pct(top_ytd.iloc[0]['win_rate_pct']) if not top_ytd.empty else 'n/a'}.",
        f"- Best 4-year combo: `{robust_name}` with {_fmt_pct(top_4y.iloc[0]['total_return_pct']) if not top_4y.empty else 'n/a'} return, "
        f"Sharpe {_fmt_num(top_4y.iloc[0]['sharpe_ratio'], 3) if not top_4y.empty else 'n/a'}, "
        f"WR {_fmt_pct(top_4y.iloc[0]['win_rate_pct']) if not top_4y.empty else 'n/a'}.",
        "",
        "## Top Single Strategies YTD",
        "",
        _markdown_table(
            top_single,
            ["strategy", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "number_of_trades"],
        ),
        "",
        "## Top Combos YTD",
        "",
        _markdown_table(
            top_ytd,
            ["combo", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "number_of_trades"],
        ),
        "",
        "## Top Combos 4Y Check",
        "",
        _markdown_table(
            top_4y,
            ["combo", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "number_of_trades"],
        ),
        "",
        f"## Diagnostics For `{best_combo_name or 'n/a'}`",
        "",
        "### By Strategy",
        "",
        _markdown_table(
            diag["by_strategy"],
            ["edge_strategy_name", "trades", "win_rate_pct", "avg_pnl_pct", "total_pnl"],
        ),
        "",
        "### By Exit Reason",
        "",
        _markdown_table(
            diag["by_exit_reason"],
            ["exit_reason", "trades", "win_rate_pct", "avg_pnl_pct", "total_pnl"],
        ),
        "",
        "### Worst Trades",
        "",
        _markdown_table(
            diag["worst_trades"],
            ["symbol", "edge_strategy_name", "signal_date", "entry_date", "exit_date", "exit_reason", "pnl_pct", "pnl"],
        ),
        "",
        "## Research Queue",
        "",
        "1. Test `leader_pullback_market_regime_v3` as the production core because it is strongest on the 4-year check.",
        "2. Keep `PURE_TOP3` as a YTD alpha candidate, but require a degradation check because its 4-year return is below `TOP1_leader_pullback_regime`.",
        "3. Investigate whether `breakout_55_smt_v1` improves YTD return at the cost of 4-year stability; test a market-regime or volatility gate before promotion.",
        "4. Reduce false starts by studying the worst trades above for common entry gap, symbol, date cluster, or exit reason patterns.",
        "5. Add any new filter as shadow first, then run YTD and 4-year summaries before touching live strategy config.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build no-lookahead strategy improvement research report")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(str(args.out))


if __name__ == "__main__":
    main()
