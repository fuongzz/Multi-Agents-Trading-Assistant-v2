"""Export a client-readable Flow V2 Enhanced comparison report."""

from __future__ import annotations

import argparse
import math
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.backtest_combos_unbiased import resolve_symbols


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"
OHLCV_PATH = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
FLOW_DIR = ROOT / "reports" / "flow_v2_gate_comparison_vn100_2020_to_2026-05-26"
DEFAULT_OUT = ROOT / "reports" / "flow_v2_enhanced_review_vn100_to_2026-05-26"
CAPITAL = 100_000_000.0
END_DATE = "2026-05-26"
PERIODS = [
    ("recent", "Từ 2025 đến hiện tại", "2025-01-01"),
    ("full_history", "Toàn lịch sử kiểm chứng", "2020-01-01"),
]
FLOW_SERIES = [
    (
        "Flow V2 Enhanced - baseline có gate",
        "risk_on_or_strong_neutral_none",
        "#167d68",
        "Chiến lược",
    ),
    (
        "Flow V2 Enhanced - tiered exit có gate",
        "risk_on_or_strong_neutral_flow_momentum_tiered",
        "#14609b",
        "Chiến lược",
    ),
    (
        "Flow V2 không gate - nghiên cứu",
        "none_none",
        "#d47710",
        "Nghiên cứu",
    ),
]
DEPOSIT_SCENARIOS = [
    (
        "Tiền gửi 12T tham chiếu 4.6%/năm",
        0.046,
        "#737f8b",
        "Tham chiếu ít rủi ro",
        "Mức 12 tháng của Vietcombank được VietinBank dẫn chiếu tại 23/10/2025; dùng làm kịch bản, không phải báo giá ngày mở sổ.",
    ),
    (
        "CCTG VCB 6.6%/năm - đợt 02/2026",
        0.066,
        "#a18d42",
        "Tham chiếu kỳ hạn",
        "Sản phẩm chứng chỉ tiền gửi Vietcombank phát hành giới hạn từ 03/02/2026 đến 28/02/2026; dùng làm kịch bản cao hơn tiền gửi thường.",
    ),
]


def _fmt_pct(value: float) -> str:
    return f"{value:+,.2f}%"


def _fmt_vnd(value: float) -> str:
    return f"{value / 1_000_000:,.1f} triệu"


def _metrics(equity: pd.Series) -> dict[str, float]:
    equity = equity.dropna().astype(float)
    returns = equity.pct_change().dropna()
    drawdown = equity / equity.cummax() - 1.0
    sharpe = 0.0
    if not returns.empty and float(returns.std()) > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252.0))
    return {
        "ending_equity": float(equity.iloc[-1]),
        "return_pct": float((equity.iloc[-1] / equity.iloc[0] - 1.0) * 100.0),
        "sharpe": sharpe,
        "max_drawdown_pct": float(drawdown.min() * 100.0),
    }


def _read_flow_curve(period: str, variant: str) -> pd.Series:
    frame = pd.read_csv(FLOW_DIR / f"{period}_{variant}_equity.csv", parse_dates=["date"])
    return frame.set_index("date")["equity"].sort_index()


def _vnindex_hold(start: str, end: str) -> pd.Series:
    frame = pd.read_parquet(INDEX_PATH, columns=["date", "symbol", "close"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[
        frame["symbol"].astype(str).str.upper().eq("VNINDEX")
        & frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ].sort_values("date")
    prices = frame.set_index("date")["close"]
    return CAPITAL * prices / float(prices.iloc[0])


def _vn100_equal_hold(start: str, end: str, symbols: list[str]) -> tuple[pd.Series, int]:
    frame = pd.read_parquet(OHLCV_PATH, columns=["date", "symbol", "close"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[
        frame["symbol"].isin(symbols)
        & frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ]
    panel = frame.pivot(index="date", columns="symbol", values="close").sort_index()
    initial = panel.iloc[0].dropna()
    normalized = panel[list(initial.index)].ffill().divide(initial)
    return CAPITAL * normalized.mean(axis=1), len(initial.index)


def _deposit_curve(index: pd.DatetimeIndex, annual_rate: float) -> pd.Series:
    elapsed_days = (index - index[0]).days
    return pd.Series(CAPITAL * (1.0 + annual_rate) ** (elapsed_days / 365.0), index=index)


def _path_points(series: pd.Series, min_log: float, max_log: float, width: int, height: int) -> str:
    left, top, chart_w, chart_h = 56, 22, width - 78, height - 54
    values = series.to_numpy(dtype=float)
    xs = np.linspace(left, left + chart_w, len(values))
    logs = np.log10(values / CAPITAL)
    span = max(max_log - min_log, 0.1)
    ys = top + chart_h * (1.0 - ((logs - min_log) / span))
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def _chart(curves: dict[str, tuple[pd.Series, str]], title: str) -> str:
    width, height = 930, 340
    log_values = np.concatenate([np.log10(series.to_numpy(dtype=float) / CAPITAL) for series, _ in curves.values()])
    min_log = min(0.0, float(log_values.min()))
    max_log = float(log_values.max())
    plot_lines: list[str] = []
    legend: list[str] = []
    for label, (series, color) in curves.items():
        points = _path_points(series, min_log, max_log, width, height)
        plot_lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend.append(
            f'<span class="legend-item"><i style="background:{color}"></i>{escape(label)}</span>'
        )
    ticks: list[str] = []
    for fraction in [0.0, 0.25, 0.5, 0.75, 1.0]:
        log_value = min_log + (max_log - min_log) * fraction
        nav = CAPITAL * (10**log_value)
        y = 22 + (height - 54) * (1.0 - fraction)
        ticks.append(
            f'<line x1="56" y1="{y:.1f}" x2="{width - 22}" y2="{y:.1f}" stroke="#e7ecef"/>'
            f'<text x="4" y="{y + 4:.1f}" fill="#627182" font-size="11">{nav / 1_000_000:,.0f}tr</text>'
        )
    return (
        f'<article class="chart-card"><h3>{escape(title)}</h3>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f'{"".join(ticks)}{"".join(plot_lines)}</svg>'
        f'<div class="legend">{"".join(legend)}</div></article>'
    )


def _result_rows(period: str, start: str, symbols: list[str]) -> tuple[pd.DataFrame, dict[str, tuple[pd.Series, str]], int]:
    rows: list[dict[str, Any]] = []
    curves: dict[str, tuple[pd.Series, str]] = {}
    for label, variant, color, group in FLOW_SERIES:
        equity = _read_flow_curve(period, variant)
        row = _metrics(equity)
        row.update({"phuong_an": label, "nhom": group, "ghi_chu": "Đã gồm phí/thuế/slippage; khớp open T+1."})
        rows.append(row)
        curves[label] = (equity, color)

    vnindex = _vnindex_hold(start, END_DATE)
    row = _metrics(vnindex)
    row.update(
        {
            "phuong_an": "VNINDEX mua và giữ",
            "nhom": "Benchmark thị trường",
            "ghi_chu": "Chỉ số giá, không gồm cổ tức hoặc phí mô phỏng ETF.",
        }
    )
    rows.append(row)
    curves["VNINDEX mua và giữ"] = (vnindex, "#4e5970")

    vn100_hold, count = _vn100_equal_hold(start, END_DATE, symbols)
    row = _metrics(vn100_hold)
    row.update(
        {
            "phuong_an": f"VN100 mua và giữ đều ({count} mã có dữ liệu đầu kỳ)",
            "nhom": "Benchmark hold",
            "ghi_chu": "Rổ cố định theo danh sách VN100 hiện tại; chưa trừ phí/cổ tức.",
        }
    )
    rows.append(row)
    curves["VN100 equal-weight hold"] = (vn100_hold, "#7e51a0")

    calendar = _read_flow_curve(period, "risk_on_or_strong_neutral_none").index
    for label, rate, color, group, note in DEPOSIT_SCENARIOS:
        deposit = _deposit_curve(calendar, rate)
        row = _metrics(deposit)
        row["sharpe"] = np.nan
        row.update({"phuong_an": label, "nhom": group, "ghi_chu": note})
        rows.append(row)
        curves[label.split(" - ")[0]] = (deposit, color)
    return pd.DataFrame(rows), curves, count


def _table(frame: pd.DataFrame) -> str:
    rows: list[str] = []
    for record in frame.to_dict("records"):
        risk_class = "risk" if record["max_drawdown_pct"] <= -20 else ""
        sharpe = "-" if pd.isna(record["sharpe"]) else f'{record["sharpe"]:.3f}'
        rows.append(
            f'<tr class="{risk_class}"><td>{escape(str(record["phuong_an"]))}</td>'
            f'<td>{escape(str(record["nhom"]))}</td>'
            f'<td class="num">{_fmt_vnd(record["ending_equity"])}</td>'
            f'<td class="num"><strong>{_fmt_pct(record["return_pct"])}</strong></td>'
            f'<td class="num">{sharpe}</td>'
            f'<td class="num">{_fmt_pct(record["max_drawdown_pct"])}</td>'
            f'<td>{escape(str(record["ghi_chu"]))}</td></tr>'
        )
    return (
        '<table><thead><tr><th>Phương án</th><th>Nhóm</th><th>NAV cuối kỳ</th>'
        '<th>Lợi nhuận</th><th>Sharpe</th><th>Max DD</th><th>Cách đọc</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _export_html(out_dir: Path, results: dict[str, pd.DataFrame], charts: dict[str, str], counts: dict[str, int]) -> None:
    recent = results["recent"].set_index("phuong_an")
    full = results["full_history"].set_index("phuong_an")
    baseline = "Flow V2 Enhanced - baseline có gate"
    tiered = "Flow V2 Enhanced - tiered exit có gate"
    no_gate = "Flow V2 không gate - nghiên cứu"
    html = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<title>Flow V2 Enhanced - Logic và so sánh hiệu suất</title>
<style>
:root {{ --ink:#13283a; --muted:#617385; --teal:#167d68; --blue:#14609b; --gold:#a86a0a; --border:#dce5e8; --bg:#f5f8f9; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); font:15px "Segoe UI", Arial, sans-serif; line-height:1.52; background:var(--bg); }}
main {{ max-width:1280px; margin:0 auto; padding:30px 28px 48px; }}
a {{ color:var(--blue); }} h1 {{ margin:8px 0 6px; font-size:32px; }} h2 {{ margin:34px 0 12px; font-size:23px; }}
h3 {{ margin:0 0 10px; font-size:17px; }} .sub {{ color:var(--muted); margin:0 0 20px; }}
.hero {{ background:white; padding:26px 28px; border:1px solid var(--border); border-radius:16px; }}
.badge {{ display:inline-block; padding:5px 10px; border-radius:14px; background:#e3f3ee; color:var(--teal); font-weight:700; font-size:12px; }}
.cards {{ display:grid; grid-template-columns:repeat(4, minmax(190px, 1fr)); gap:12px; margin-top:22px; }}
.card {{ background:white; border:1px solid var(--border); border-radius:12px; padding:15px; }}
.card b {{ display:block; font-size:25px; margin-top:5px; }} .card span {{ color:var(--muted); font-size:13px; }}
.good b {{ color:var(--teal); }} .caution b {{ color:var(--gold); }}
.box {{ background:white; border:1px solid var(--border); border-radius:13px; padding:18px 20px; }}
.logic {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }}
.logic strong {{ display:block; margin-bottom:5px; color:var(--teal); }}
.callout {{ margin-top:15px; padding:13px 16px; background:#fff6e3; border-left:4px solid #d89b22; border-radius:4px; }}
table {{ width:100%; border-collapse:collapse; background:white; font-size:13px; }}
th, td {{ border:1px solid var(--border); padding:9px 10px; vertical-align:top; }} th {{ background:#edf4f3; text-align:left; }}
td.num {{ text-align:right; white-space:nowrap; }} tr.risk td:nth-child(6) {{ color:#a6342d; font-weight:600; }}
.chart-card {{ background:white; border:1px solid var(--border); border-radius:13px; padding:16px; margin:12px 0; }}
.chart-card svg {{ display:block; width:100%; height:auto; }} .legend {{ display:flex; flex-wrap:wrap; gap:16px; color:var(--muted); font-size:13px; }}
.legend-item i {{ width:18px; height:3px; display:inline-block; vertical-align:middle; margin-right:6px; }}
.issues {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }} .issues b {{ display:block; margin-bottom:5px; }}
ul {{ margin:7px 0 0; padding-left:20px; }} code {{ background:#eef2f4; padding:1px 5px; border-radius:4px; }}
.footer {{ margin-top:32px; font-size:13px; color:var(--muted); }}
@media(max-width:880px) {{ .cards,.logic,.issues {{ grid-template-columns:1fr; }} main {{ padding:18px 12px; }} }}
</style></head><body><main>
<p><a href="../combined_paper_trading_demo/index.html">← Dashboard paper trading</a> | <a href="../flow_v2_gate_comparison_vn100_2020_to_2026-05-26/index.html">Nghiên cứu gate chi tiết</a></p>
<section class="hero">
  <span class="badge">FLOW V2 ENHANCED | VN100 | dữ liệu đến 2026-05-26</span>
  <h1>Dòng tiền có chọn lọc, không phải đánh cược toàn thị trường</h1>
  <p class="sub">Báo cáo giải thích logic, hiệu suất lịch sử và lý do phần giao dịch paper vẫn giữ market gate. Vốn so sánh: 100 triệu VND.</p>
  <div class="cards">
    <div class="card good"><span>Enhanced baseline có gate<br>2025 đến nay</span><b>{_fmt_pct(recent.loc[baseline, "return_pct"])}</b><span>Max DD {_fmt_pct(recent.loc[baseline, "max_drawdown_pct"])}</span></div>
    <div class="card good"><span>Enhanced tiered exit<br>2020 đến nay</span><b>{_fmt_pct(full.loc[tiered, "return_pct"])}</b><span>Max DD {_fmt_pct(full.loc[tiered, "max_drawdown_pct"])}</span></div>
    <div class="card"><span>VNINDEX hold<br>2025 đến nay</span><b>{_fmt_pct(recent.loc["VNINDEX mua và giữ", "return_pct"])}</b><span>Benchmark thị trường</span></div>
    <div class="card caution"><span>Không gate nghiên cứu<br>Max DD 2020 đến nay</span><b>{_fmt_pct(full.loc[no_gate, "max_drawdown_pct"])}</b><span>Return cao hơn nhưng rủi ro sâu hơn</span></div>
  </div>
</section>

<h2>Flow V2 Enhanced nghĩ gì?</h2>
<div class="logic">
 <div class="box"><strong>1. Xác định nơi có dòng tiền</strong>Điểm <code>flow_heavy</code> ưu tiên sponsorship 30%, absorption 22%, chu kỳ ngành 20%, relative strength 13%, thanh khoản 10%, trừ áp lực phân phối 17%.</div>
 <div class="box"><strong>2. Chỉ chọn cổ phiếu đủ chất lượng</strong>VN100, trên MA50, distribution pressure không quá cao, flow sponsorship tối thiểu, thanh khoản tương đối và RS percentile cao. Danh mục chỉ giữ top 2.</div>
 <div class="box"><strong>3. Gate quyết định có giao dịch</strong>Paper active chỉ vào khi thị trường ở <code>RISK_ON</code> hoặc <code>NEUTRAL</code> mạnh. Danh sách bỏ gate chỉ phục vụ giải thích/cơ hội quan sát.</div>
 <div class="box"><strong>4. Thực thi có độ trễ thực tế</strong>Tín hiệu được chốt sau close ngày T, lệnh khớp open T+1; rebalance mỗi 10 phiên. Backtest Flow đã tính lot, phí, thuế và slippage.</div>
 <div class="box"><strong>5. Hai cách quản trị vị thế</strong><code>baseline</code> giữ đến nhịp rebalance; <code>tiered exit</code> thoát sớm khi dòng tiền/momentum suy yếu nhằm giảm drawdown dài hạn.</div>
 <div class="box"><strong>6. Giá trị với người dùng</strong>Hệ thống vừa đưa ra paper target đủ gate, vừa cho thấy mã nổi bật bị regime chặn để người dùng hiểu cơ hội mà không biến chúng thành lệnh tự động.</div>
</div>

<h2>So sánh từ 2025-01-01 đến 2026-05-26</h2>
<p class="sub">Đây là giai đoạn thị trường tăng mạnh; kiểm tra khả năng bắt xu hướng và mức đánh đổi khi bỏ gate.</p>
{_table(results["recent"])}
{charts["recent"]}
<div class="callout"><strong>Kết luận giai đoạn gần:</strong> bỏ gate đưa return baseline lên {_fmt_pct(recent.loc[no_gate, "return_pct"])}, nhưng Max DD tăng từ {_fmt_pct(recent.loc[baseline, "max_drawdown_pct"])} lên {_fmt_pct(recent.loc[no_gate, "max_drawdown_pct"])}. Vì vậy no-gate phù hợp làm danh sách quan sát, không thay thế pipeline paper.</div>

<h2>So sánh toàn lịch sử từ 2020-01-01 đến 2026-05-26</h2>
<p class="sub">Giai đoạn này chứa nhiều regime khác nhau, phù hợp hơn để đánh giá độ bền của quản trị rủi ro.</p>
{_table(results["full_history"])}
{charts["full_history"]}
<div class="callout"><strong>Tư duy Enhanced:</strong> baseline có gate tăng {_fmt_pct(full.loc[baseline, "return_pct"])}; tiered exit có gate tăng {_fmt_pct(full.loc[tiered, "return_pct"])} nhưng Max DD tốt hơn baseline ({_fmt_pct(full.loc[tiered, "max_drawdown_pct"])} so với {_fmt_pct(full.loc[baseline, "max_drawdown_pct"])}). No-gate đạt lợi nhuận thô cao hơn nhưng từng giảm {_fmt_pct(full.loc[no_gate, "max_drawdown_pct"])}.</div>

<h2>Hiểu đúng các benchmark</h2>
<div class="issues">
 <div class="box"><b>Hold VN100 không phải chiến lược Flow</b>Benchmark mua-giữ chia đều dùng {counts["recent"]} mã có giá đầu kỳ cho giai đoạn gần và {counts["full_history"]} mã cho toàn lịch sử, theo danh sách VN100 được resolve hiện tại. Nó có survivorship bias và chưa trừ phí/cổ tức.</div>
 <div class="box"><b>VNINDEX là nhịp thị trường, không phải tài khoản giao dịch</b>Mua-giữ VNINDEX dùng chỉ số giá local từ <code>index_master.parquet</code>, không gồm cổ tức hay chi phí sản phẩm mô phỏng chỉ số.</div>
 <div class="box"><b>Tiền gửi là mức nền rủi ro thấp</b>Kịch bản 4.6%/năm tham chiếu lãi suất VCB 12 tháng được công bố/dẫn chiếu ngày 23/10/2025. Kịch bản 6.6% là chứng chỉ tiền gửi VCB đợt 03/02-28/02/2026, không được coi là sản phẩm còn mở hiện tại.</div>
 <div class="box"><b>So sánh không đồng nghĩa cam kết</b>Tiền gửi có lãi cố định theo điều kiện sản phẩm; Flow/VNINDEX/VN100 chịu biến động giá và có thể thua lỗ. Backtest không bảo đảm hiệu quả tương lai.</div>
</div>

<h2>Các vấn đề đã phát hiện và cách đang xử lý</h2>
<div class="issues">
 <div class="box"><b>Đã sửa lỗi đơn vị giá thi hành</b>Các kết quả Flow cũ trước `2026-05-24` dùng sai đơn vị nghìn VND khi sizing và đã bị loại khỏi so sánh. Trang này chỉ dùng rerun giá VND đã sửa.</div>
 <div class="box"><b>Đã kiểm tra thực thi T+1 và giới hạn top2</b>Audit xác nhận tín hiệu xảy ra trước lệnh, tiền mặt không âm và tối đa hai mã; lỗi giữ thừa vị thế khi không có giá mở cửa khả dụng đã được sửa.</div>
 <div class="box"><b>Chưa loại bỏ hoàn toàn survivorship bias</b>Universe VN100 được resolve tại thời điểm chạy, chưa phải danh sách thành phần lịch sử theo từng ngày. Điều này có thể làm đẹp kết quả cả Flow lẫn hold.</div>
 <div class="box"><b>Paper sleeve và kết quả nghiên cứu phải tách biệt</b>Sleeve <code>flow_v2</code> trên dashboard có exit contract paper riêng; hiệu suất trình bày ở đây thuộc các biến thể Enhanced được kiểm tra, không được gắn thành PnL paper thực tế.</div>
</div>

<h2>Cách dùng trong sản phẩm</h2>
<ul>
 <li><strong>Khối tự giao dịch:</strong> giữ gate hiện tại và các rule thi hành đã kiểm tra.</li>
 <li><strong>Khối tư vấn/giải thích:</strong> cho người dùng xem dòng tiền hôm nay, mã mạnh gần đây và danh sách no-gate được gắn nhãn quan sát.</li>
 <li><strong>Khối báo cáo khách hàng:</strong> dùng benchmark VNINDEX, hold VN100 và tiền gửi để giải thích kỳ vọng lợi nhuận đi cùng rủi ro, không chỉ trình bày return.</li>
</ul>
<p class="footer">Nguồn local: <code>flow_v2_gate_comparison_vn100_2020_to_2026-05-26</code>, <code>index_master.parquet</code>, <code>ohlcv_master.parquet</code>. Nguồn tham chiếu lãi suất: <a href="https://www.vietinbank.vn/vi/vietinbank-thong-bao-lai-suat-trai-phieu-ctg2232t2-02-ky-3-ma-chung-khoan-ctg123034-20251024101721-00-html">VietinBank/Vietcombank 12T ngày 23/10/2025</a>; <a href="https://www.vietcombank.com.vn/vi-VN/KHCN/Truy-cap-nhanh/Tin-noi-bat/Articles/2026/02/11/Chung-chi-tien-gui-dot-2">Vietcombank CCTG đợt 02/2026</a>. Báo cáo phục vụ nghiên cứu và demo, không phải khuyến nghị đầu tư cá nhân hóa.</p>
</main></body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def export_report(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = list(resolve_symbols("vn100"))
    result_frames: dict[str, pd.DataFrame] = {}
    charts: dict[str, str] = {}
    counts: dict[str, int] = {}
    exports: list[pd.DataFrame] = []
    for period, label, start in PERIODS:
        frame, curves, count = _result_rows(period, start, symbols)
        frame.insert(0, "period", period)
        frame.insert(1, "period_label", label)
        result_frames[period] = frame.drop(columns=["period", "period_label"])
        charts[period] = _chart(curves, f"Đường cong NAV - {label} (thang log)")
        counts[period] = count
        exports.append(frame)
        curve_export = pd.concat({key: series for key, (series, _) in curves.items()}, axis=1)
        curve_export.index.name = "date"
        curve_export.to_csv(out_dir / f"{period}_nav_curves.csv", encoding="utf-8-sig")
    comparison = pd.concat(exports, ignore_index=True)
    comparison.to_csv(out_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")
    _export_html(out_dir, result_frames, charts, counts)
    return out_dir / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Flow V2 Enhanced logic and benchmark comparison report.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(export_report(args.out_dir))


if __name__ == "__main__":
    main()
