"""Research margin overlays for the five sleeves shown in the paper dashboard.

This script is deliberately separate from live/paper execution. It applies a
causal financing overlay to verified unlevered equity curves:

- the leverage decision for session T uses only observations through T-1;
- borrowing accrues interest every leveraged session;
- changes in borrowed exposure pay incremental trading friction;
- a drawdown breach forces a cooldown at cash-account leverage.

It is a portfolio overlay study, not a broker margin simulator: it does not
model symbol margin eligibility, liquidity, forced-sale auction prices, or
re-ranking caused by altered buying power.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from scripts.backtest_flow_v2_rotation_production_like import RotationConfig, _prepare_features, run_backtest


ROOT = Path(__file__).resolve().parents[1]
END_DATE = "2026-05-22"
INITIAL_CAPITAL = 100_000_000.0


@dataclass(frozen=True)
class MarginPolicy:
    policy_id: str
    label: str
    max_leverage: float
    guarded: bool = False


POLICIES = (
    MarginPolicy("cash_1_00x", "Khong margin (1.00x)", 1.0),
    MarginPolicy("static_1_25x", "Margin co dinh toi da 1.25x", 1.25),
    MarginPolicy("guarded_1_25x", "Margin kiem soat rui ro toi da 1.25x", 1.25, True),
    MarginPolicy("static_1_50x", "Margin co dinh toi da 1.50x", 1.50),
    MarginPolicy("guarded_1_50x", "Margin kiem soat rui ro toi da 1.50x", 1.50, True),
)


def _sharpe(returns: pd.Series) -> float:
    values = returns.dropna()
    if values.empty or float(values.std()) == 0.0:
        return 0.0
    return float(values.mean() / values.std() * np.sqrt(252.0))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def _normalize_equity(frame: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result = result.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if result.empty:
        return result
    scale = initial_capital / float(result["equity"].iloc[0])
    result["equity"] = result["equity"].astype(float) * scale
    result["cash"] = result["cash"].astype(float) * scale
    return result


def run_margin_overlay(
    underlying: pd.DataFrame,
    policy: MarginPolicy,
    turnover: pd.DataFrame | None = None,
    initial_capital: float = INITIAL_CAPITAL,
    annual_interest_rate: float = 0.12,
    incremental_side_cost: float = 0.002,
    forced_deleverage_drawdown: float = -0.20,
    cooldown_sessions: int = 10,
    guard_drawdown: float = -0.08,
    guard_annual_volatility: float = 0.35,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply financing without using future information to choose leverage."""
    base = _normalize_equity(underlying, initial_capital)
    if base.empty:
        return pd.DataFrame(), {}

    base["base_return"] = base["equity"].pct_change().fillna(0.0)
    base["base_drawdown"] = base["equity"] / base["equity"].cummax() - 1.0
    base["exposure"] = ((base["equity"] - base["cash"]) / base["equity"]).clip(0.0, 1.0).fillna(0.0)
    base["volatility_20"] = base["base_return"].rolling(20).std() * np.sqrt(252.0)
    turnover_by_date: dict[pd.Timestamp, float] = {}
    if turnover is not None and not turnover.empty:
        normalized_turnover = turnover.copy()
        normalized_turnover["date"] = pd.to_datetime(normalized_turnover["date"]).dt.normalize()
        turnover_by_date = normalized_turnover.groupby("date")["gross_value"].sum().astype(float).to_dict()

    nav = initial_capital
    peak = initial_capital
    previous_borrow_ratio = 0.0
    cooldown_left = 0
    margin_calls = 0
    interest_paid = 0.0
    financing_trade_cost = 0.0
    resize_trade_cost = 0.0
    rotation_trade_cost = 0.0
    borrowed_sessions = 0
    rows: list[dict[str, Any]] = []

    for idx, row in base.iterrows():
        reason = "INITIAL"
        borrow_ratio = 0.0
        if idx > 0:
            prior = base.iloc[idx - 1]
            prior_exposure = float(prior["exposure"])
            can_borrow = cooldown_left == 0 and policy.max_leverage > 1.0 and prior_exposure > 0.0
            if can_borrow and policy.guarded:
                observed_vol = float(prior["volatility_20"]) if pd.notna(prior["volatility_20"]) else np.inf
                can_borrow = (
                    float(prior["base_drawdown"]) > guard_drawdown
                    and observed_vol <= guard_annual_volatility
                )
                reason = "GUARD_ALLOW" if can_borrow else "GUARD_BLOCK"
            elif can_borrow:
                reason = "STATIC_MARGIN"
            elif cooldown_left > 0:
                reason = "FORCED_COOLDOWN"
            else:
                reason = "NO_EXPOSURE"
            if can_borrow:
                borrow_ratio = (policy.max_leverage - 1.0) * prior_exposure

            base_return = float(row["base_return"])
            interest = nav * borrow_ratio * annual_interest_rate / 252.0
            resize_cost = nav * abs(borrow_ratio - previous_borrow_ratio) * incremental_side_cost
            tranche_multiplier = borrow_ratio / prior_exposure if prior_exposure > 0.0 else 0.0
            rotation_cost = (
                turnover_by_date.get(pd.Timestamp(row["date"]).normalize(), 0.0)
                * tranche_multiplier
                * incremental_side_cost
            )
            trading_cost = resize_cost + rotation_cost
            nav = nav * (1.0 + (1.0 + borrow_ratio) * base_return) - interest - trading_cost
            interest_paid += interest
            financing_trade_cost += trading_cost
            resize_trade_cost += resize_cost
            rotation_trade_cost += rotation_cost
            if borrow_ratio > 0.0:
                borrowed_sessions += 1
            peak = max(peak, nav)
            drawdown = nav / peak - 1.0
            if borrow_ratio > 0.0 and drawdown <= forced_deleverage_drawdown and cooldown_left == 0:
                margin_calls += 1
                cooldown_left = cooldown_sessions
                reason = "MARGIN_DD_BREACH"
            elif cooldown_left > 0:
                cooldown_left -= 1
            previous_borrow_ratio = borrow_ratio
        else:
            drawdown = 0.0

        rows.append(
            {
                "date": row["date"].date().isoformat(),
                "base_equity": float(row["equity"]),
                "margin_equity": nav,
                "base_return": float(row["base_return"]),
                "borrow_ratio": borrow_ratio,
                "gross_leverage": 1.0 + borrow_ratio,
                "interest_paid_cumulative": interest_paid,
                "financing_trade_cost_cumulative": financing_trade_cost,
                "financing_resize_cost_cumulative": resize_trade_cost,
                "incremental_rotation_cost_cumulative": rotation_trade_cost,
                "margin_drawdown": drawdown,
                "decision_reason": reason,
            }
        )

    equity = pd.DataFrame(rows)
    margin_returns = equity["margin_equity"].pct_change()
    years = max((len(equity) - 1) / 252.0, 1.0 / 252.0)
    total_return = float(equity["margin_equity"].iloc[-1] / initial_capital - 1.0)
    summary = {
        "policy_id": policy.policy_id,
        "policy": policy.label,
        "max_leverage_x": policy.max_leverage,
        "total_return_pct": round(total_return * 100.0, 2),
        "cagr_pct": round(((1.0 + total_return) ** (1.0 / years) - 1.0) * 100.0, 2)
        if total_return > -1.0
        else -100.0,
        "sharpe_ratio": round(_sharpe(margin_returns), 3),
        "max_drawdown_pct": round(_max_drawdown(equity["margin_equity"]) * 100.0, 2),
        "ending_nav": round(float(equity["margin_equity"].iloc[-1]), 0),
        "interest_paid": round(interest_paid, 0),
        "incremental_trade_cost": round(financing_trade_cost, 0),
        "financing_resize_cost": round(resize_trade_cost, 0),
        "incremental_rotation_cost": round(rotation_trade_cost, 0),
        "borrowed_sessions": borrowed_sessions,
        "average_gross_leverage_x": round(float(equity["gross_leverage"].mean()), 3),
        "margin_calls": margin_calls,
    }
    return equity, summary


def _flow_config(start: str, end: str, capital: float, early_exit_mode: str) -> RotationConfig:
    return RotationConfig(
        universe="vn100",
        start=start,
        end=end,
        positions=2,
        rebalance_days=10,
        initial_capital=capital,
        market_gate="risk_on_or_strong_neutral",
        pool_filter="high_rs",
        score_mode="flow_heavy",
        price_unit_multiplier=1000.0,
        early_exit_mode=early_exit_mode,
    )


def load_underlyings(out_dir: Path, capital: float) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    curves: list[dict[str, Any]] = []
    scope = [
        {
            "sleeve_id": "flow_v2",
            "label": "Flow V2 Rotation",
            "period": "2025-2026 YTD",
            "status": "Khong backtest margin exact",
            "note": "Sleeve live co exit contract rieng, khong gan ket qua audited vao sleeve nay.",
        },
        {
            "sleeve_id": "flow_v2",
            "label": "Flow V2 Rotation",
            "period": "2020-2026 YTD",
            "status": "Khong backtest margin exact",
            "note": "Can dong bo engine exit cua sleeve live truoc khi danh gia margin.",
        },
    ]
    core_sources = {
        "mvp_p5": (
            "MVP Original p5/c10",
            ROOT
            / "backtest_results"
            / "combos_unbiased_vn100_core_mvp9_compare_p5_2025now_2025-01-01_2026-05-22"
            / "CORE_MVP9_equity.csv",
        ),
        "mvp_p4": (
            "MVP Improved p4/c10",
            ROOT
            / "backtest_results"
            / "combos_unbiased_vn100_core_mvp9_compare_p4_2025now_2025-01-01_2026-05-22"
            / "CORE_MVP9_equity.csv",
        ),
    }
    for sleeve_id, (label, source) in core_sources.items():
        curve = pd.read_csv(source)
        trades = pd.read_csv(source.with_name("CORE_MVP9_trades.csv"))
        entry_turnover = trades[["entry_date", "entry_value"]].rename(
            columns={"entry_date": "date", "entry_value": "gross_value"}
        )
        exit_turnover = trades[["exit_date", "exit_price", "shares"]].copy()
        exit_turnover["gross_value"] = exit_turnover["exit_price"].astype(float) * exit_turnover["shares"].astype(float)
        exit_turnover = exit_turnover[["exit_date", "gross_value"]].rename(columns={"exit_date": "date"})
        curves.append(
            {
                "sleeve_id": sleeve_id,
                "label": label,
                "period": "2025-2026 YTD",
                "start": "2025-01-01",
                "end": END_DATE,
                "source": str(source.relative_to(ROOT)),
                "equity": curve,
                "turnover": pd.concat([entry_turnover, exit_turnover], ignore_index=True),
            }
        )
        scope.extend(
            [
                {
                    "sleeve_id": sleeve_id,
                    "label": label,
                    "period": "2025-2026 YTD",
                    "status": "Backtest margin",
                    "note": "Duong NAV exact cua sleeve HTML hien tai.",
                },
                {
                    "sleeve_id": sleeve_id,
                    "label": label,
                    "period": "2020-2026 YTD",
                    "status": "Chua xac minh",
                    "note": "Chua co full-history rerun exact cho current contract.",
                },
            ]
        )

    for period, start in (("2025-2026 YTD", "2025-01-01"), ("2020-2026 YTD", "2020-01-01")):
        print(f"[margin-research] preparing Flow features for {period}")
        features = _prepare_features("vn100", start, END_DATE)
        for sleeve_id, label, exit_mode in (
            ("flow_v2_baseline", "Baseline fresh-signal top2", "none"),
            ("flow_v2_tiered", "Early-exit hai tang top2", "flow_momentum_tiered"),
        ):
            print(f"[margin-research] rerun {sleeve_id} {period}")
            result = run_backtest(_flow_config(start, END_DATE, capital, exit_mode), features=features)
            underlying_file = out_dir / "underlying" / f"{sleeve_id}_{start}_{END_DATE}_equity.csv"
            underlying_file.parent.mkdir(parents=True, exist_ok=True)
            result["equity"].to_csv(underlying_file, index=False)
            curves.append(
                {
                    "sleeve_id": sleeve_id,
                    "label": label,
                    "period": period,
                    "start": start,
                    "end": END_DATE,
                    "source": str(underlying_file.relative_to(ROOT)),
                    "equity": result["equity"],
                    "turnover": result["trades"][["date", "gross_value"]].copy(),
                }
            )
            scope.append(
                {
                    "sleeve_id": sleeve_id,
                    "label": label,
                    "period": period,
                    "status": "Backtest margin",
                    "note": "Tai chay engine audited, capital 100m, VN100, open T+1.",
                }
            )
    return curves, pd.DataFrame(scope)


def _money(value: float) -> str:
    return f"{value / 1_000_000:+.2f} trieu"


def _html_table(frame: pd.DataFrame, columns: list[tuple[str, str]], classes: str = "") -> str:
    heads = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    rows = []
    for _, row in frame.iterrows():
        cells = "".join(f"<td>{escape(str(row[column]))}</td>" for column, _ in columns)
        rows.append(f"<tr>{cells}</tr>")
    return f'<table class="{classes}"><thead><tr>{heads}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def export_html(out_dir: Path, summaries: pd.DataFrame, scope: pd.DataFrame, assumptions: dict[str, Any]) -> None:
    display = summaries.copy()
    display["Return"] = display["total_return_pct"].map(lambda x: f"{x:+.2f}%")
    display["Sharpe"] = display["sharpe_ratio"].map(lambda x: f"{x:.3f}")
    display["Max DD"] = display["max_drawdown_pct"].map(lambda x: f"{x:.2f}%")
    display["NAV cuoi"] = display["ending_nav"].map(_money)
    display["Lai vay"] = display["interest_paid"].map(lambda x: _money(-x))
    display["CP giao dich vay"] = display["incremental_trade_cost"].map(lambda x: _money(-x))
    display["Avg lev"] = display["average_gross_leverage_x"].map(lambda x: f"{x:.3f}x")
    display["Call"] = display["margin_calls"].astype(str)

    best_rows = []
    for (period, sleeve_id), group in summaries.groupby(["period", "sleeve_id"], sort=False):
        base = group.loc[group["policy_id"] == "cash_1_00x"].iloc[0]
        best = group.sort_values("total_return_pct", ascending=False).iloc[0]
        best_rows.append(
            {
                "period": period,
                "sleeve": base["label"],
                "baseline": f"{base['total_return_pct']:+.2f}%",
                "best": best["policy"],
                "return": f"{best['total_return_pct']:+.2f}%",
                "delta": f"{best['total_return_pct'] - base['total_return_pct']:+.2f} diem %",
                "dd": f"{best['max_drawdown_pct']:.2f}%",
                "calls": str(int(best["margin_calls"])),
            }
        )
    best_frame = pd.DataFrame(best_rows)
    sharpe_wins = 0
    comparisons = 0
    for _, group in summaries.groupby(["period", "sleeve_id"], sort=False):
        comparisons += 1
        best_sharpe_policy = group.sort_values("sharpe_ratio", ascending=False).iloc[0]["policy_id"]
        sharpe_wins += int(best_sharpe_policy == "cash_1_00x")

    scope_display = scope.rename(
        columns={"label": "Chien luoc", "period": "Giai doan", "status": "Trang thai", "note": "Ghi chu"}
    )
    summary_table = _html_table(
        display,
        [
            ("period", "Giai doan"),
            ("label", "Chien luoc"),
            ("policy", "Margin agent"),
            ("Return", "Return"),
            ("Sharpe", "Sharpe"),
            ("Max DD", "Max DD"),
            ("NAV cuoi", "NAV cuoi"),
            ("Lai vay", "Lai vay"),
            ("CP giao dich vay", "CP tang/giam vay"),
            ("Avg lev", "Don bay TB"),
            ("Call", "Forced deleverage"),
        ],
    )
    best_table = _html_table(
        best_frame,
        [
            ("period", "Giai doan"),
            ("sleeve", "Chien luoc"),
            ("baseline", "Khong margin"),
            ("best", "Agent return cao nhat"),
            ("return", "Return"),
            ("delta", "Tang them"),
            ("dd", "Max DD"),
            ("calls", "Forced deleverage"),
        ],
    )
    coverage_table = _html_table(
        scope_display,
        [("Chien luoc", "Chien luoc HTML"), ("Giai doan", "Giai doan"), ("Trang thai", "Trang thai"), ("Ghi chu", "Pham vi")],
    )
    document = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>Margin agents research - VN100</title>
<style>
body {{ margin: 30px; max-width: 1500px; font: 14px Segoe UI, Arial, sans-serif; line-height: 1.48; color: #15262c; }}
h1 {{ font-size: 27px; margin: 0 0 5px; }} h2 {{ font-size: 19px; margin: 28px 0 10px; }}
.sub {{ color: #52666d; margin: 0 0 18px; }} .note {{ border-left: 4px solid #146b62; background: #f2f7f6; padding: 12px 15px; margin: 14px 0; }}
.warn {{ border-left-color: #9a5a14; background: #fff7ed; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 18px; font-size: 13px; }}
th, td {{ padding: 8px 9px; border: 1px solid #d9e2e5; vertical-align: top; text-align: right; }}
th {{ background: #edf4f3; white-space: nowrap; }} th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
code {{ background: #eef2f2; padding: 1px 4px; }}
</style></head><body>
<h1>Nghien cuu Margin Agents cho 5 chien luoc HTML</h1>
<p class="sub">VN100 | von tham chieu 100 trieu dong | du lieu den {END_DATE} | bao cao research, khong bat margin cho paper/live</p>
<p class="note">Mo hinh la lop tai tro tren duong NAV khong margin da backtest: quyet dinh vay phien T chi dung NAV, exposure va bien dong quan sat den het T-1. Lenh goc van giu dung pipeline va quy tac open T+1 cua tung engine co the xac minh.</p>
<p class="note warn">Ket qua khong phai mo phong margin broker day du. Bao cao chua mo hinh hoa danh sach ma duoc cap margin, han muc theo tung ma, thanh khoan khi call margin, gia ban giai chap hoac lai suat thay doi. Do do chi dung de chon gia thuyet can kiem thu tiep.</p>
<h2>Pham vi 5 chien luoc tren HTML</h2>
{coverage_table}
<h2>Gia dinh margin</h2>
<ul>
<li>Lai vay: {assumptions['annual_interest_rate'] * 100:.1f}%/nam, tinh moi phien vay tren 252 phien/nam.</li>
<li>Chi phi mua/ban phan von vay: {assumptions['incremental_side_cost'] * 100:.2f}% moi chieu; bi tru ca khi tang/giam du no va khi tranche vay giao dich theo cac lan co cau cua danh muc goc, tuong duong khoang 0.40% cho mot vong mua-ban day du.</li>
<li>Agent guarded chi mo vay neu drawdown cua chien luoc co so truoc phien lon hon -8% va volatility 20 phien nam hoa khong vuot 35%.</li>
<li>Neu NAV margin sut {assumptions['forced_deleverage_drawdown'] * 100:.0f}% so voi dinh trong luc dang vay, he thong dung vay {assumptions['cooldown_sessions']} phien tiep theo.</li>
</ul>
<h2>Ket luan nghien cuu</h2>
<p class="note warn">Margin lam tang return tuyet doi trong nhung duong NAV thuan loi, nhung <strong>khong margin co Sharpe cao nhat o {sharpe_wins}/{comparisons} phep so sanh co du lieu</strong>. O toan ky Flow, margin co dinh phat sinh forced deleverage lap lai va max drawdown tang ro. Vi vay chua co co so de bat vay cho paper/live; buoc tiep theo hop ly la test han muc nho chi trong che do market/flow du manh va ap margin eligibility theo ma.</p>
<h2>Agent co return cao nhat trong tung sleeve</h2>
{best_table}
<h2>Bang ket qua day du</h2>
{summary_table}
<p class="sub">CSV: <code>margin_summary.csv</code>, <code>strategy_coverage.csv</code>; dien bien NAV tung agent nam trong thu muc <code>equity/</code>.</p>
</body></html>"""
    (out_dir / "index.html").write_text(document, encoding="utf-8")


def run_research(out_dir: Path, capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    out_dir = out_dir.resolve()
    assumptions = {
        "annual_interest_rate": 0.12,
        "incremental_side_cost": 0.002,
        "forced_deleverage_drawdown": -0.20,
        "cooldown_sessions": 10,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    curves, scope = load_underlyings(out_dir, capital)
    scope.to_csv(out_dir / "strategy_coverage.csv", index=False)

    summaries: list[dict[str, Any]] = []
    equity_dir = out_dir / "equity"
    equity_dir.mkdir(exist_ok=True)
    for item in curves:
        for policy in POLICIES:
            equity, metrics = run_margin_overlay(
                item["equity"],
                policy,
                turnover=item["turnover"],
                initial_capital=capital,
                **assumptions,
            )
            metrics.update(
                {
                    "sleeve_id": item["sleeve_id"],
                    "label": item["label"],
                    "period": item["period"],
                    "period_start": item["start"],
                    "period_end": item["end"],
                    "universe": "VN100",
                    "initial_capital": capital,
                    "underlying_source": item["source"],
                }
            )
            summaries.append(metrics)
            equity.to_csv(
                equity_dir / f"{item['sleeve_id']}_{item['start']}_{policy.policy_id}.csv",
                index=False,
            )
    summary = pd.DataFrame(summaries).sort_values(["period_start", "sleeve_id", "total_return_pct"])
    summary.to_csv(out_dir / "margin_summary.csv", index=False)
    export_html(out_dir, summary, scope, assumptions)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Research margin-agent overlays for active HTML paper sleeves.")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "margin_agent_research_vn100_2020_to_2026-05-22",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=PerformanceWarning)
    summary = run_research(args.out_dir, args.capital)
    print(
        summary[
            ["period", "sleeve_id", "policy_id", "total_return_pct", "sharpe_ratio", "max_drawdown_pct", "margin_calls"]
        ].to_string(index=False)
    )
    print(f"[margin-research] saved {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
