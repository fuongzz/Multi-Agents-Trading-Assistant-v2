"""Export a clickable layer-by-layer production pipeline demo.

The output is a static HTML/CSV package that can be opened during a teacher demo:
raw data -> features -> strategy pool -> ranking -> risk gate -> production dry-run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.strategy_sleeves import catalog_rows
from multiagents_trading_assistant.research.sector_rotation.enrich_features import (
    enrich_features_with_rotation,
)
from scripts.inject_sortable_tables import SORTABLE_JS


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "multiagents_trading_assistant" / "data"
PRODUCTION_DIR = ROOT / "reports" / "production_demo_2026-05-17"
BASELINE_BACKTEST_DIR = (
    ROOT
    / "backtest_results"
    / "entry_market_gate_variants"
    / "mvp_2025now_soft_gate_vn100_2025-01-01_2026-05-16_p5"
)


def _latest_data_date() -> pd.Timestamp:
    frame = pd.read_parquet(DATA_DIR / "ohlcv_master.parquet", columns=["date"])
    return pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize().max()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _selected_symbols(production_dir: Path, fallback: list[str], max_symbols: int) -> list[str]:
    frames = [
        _read_csv_if_exists(production_dir / "rolling_signals_14d.csv"),
        _read_csv_if_exists(production_dir / "production_signals.csv"),
    ]
    symbols: list[str] = []
    for frame in frames:
        if "symbol" not in frame.columns:
            continue
        for symbol in frame["symbol"].dropna().astype(str).str.upper().tolist():
            if symbol not in symbols:
                symbols.append(symbol)
            if len(symbols) >= max_symbols:
                return symbols
    for symbol in fallback:
        if symbol not in symbols:
            symbols.append(symbol)
        if len(symbols) >= max_symbols:
            break
    return symbols


def _write_table(frame: pd.DataFrame, csv_path: Path, html_path: Path, title: str, note: str = "") -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    html = _page(title, frame, note=note)
    html_path.write_text(html, encoding="utf-8")


def _page(title: str, frame: pd.DataFrame, note: str = "") -> str:
    display = frame.copy()
    if len(display) > 300:
        display = display.head(300)
        note = f"{note} Showing first 300 rows only in HTML; CSV contains full export.".strip()
    table_html = display.to_html(index=False, classes="data-table", border=0)
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ --ink: #14213d; --muted: #5f6b7a; --line: #d9dee7; --bg: #f5f1e8; --panel: #fffaf0; }}
    body {{ margin: 0; padding: 28px; font-family: Georgia, 'Times New Roman', serif; background: var(--bg); color: var(--ink); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .note {{ margin: 0 0 18px; color: var(--muted); font-family: Segoe UI, Arial, sans-serif; }}
    .wrap {{ overflow: auto; background: white; border: 1px solid var(--line); box-shadow: 0 14px 36px rgba(20,33,61,.12); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; white-space: nowrap; font-family: Segoe UI, Arial, sans-serif; }}
    th {{ position: sticky; top: 0; background: var(--ink); color: white; text-align: left; cursor: pointer; user-select: none; }}
    th::after {{ content: " ↕"; color: rgba(255,255,255,.55); font-weight: 400; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf0f5; }}
    tr:nth-child(even) {{ background: #fbfbf8; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="note">{note}</p>
  <div class="wrap">{table_html}</div>
  <script src="sortable_tables.js"></script>
</body>
</html>
"""


def _styled_page(title: str, body_html: str, note: str = "") -> str:
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ --ink: #14213d; --muted: #5f6b7a; --line: #d9dee7; --bg: #f5f1e8; --win: #0f8b4c; --loss: #c63d2f; --panel: #fffaf0; }}
    body {{ margin: 0; padding: 28px; font-family: Georgia, 'Times New Roman', serif; background: var(--bg); color: var(--ink); }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 26px 0 12px; font-size: 20px; }}
    .note {{ margin: 0 0 18px; color: var(--muted); font-family: Segoe UI, Arial, sans-serif; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 18px 0 24px; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 16px; box-shadow: 0 10px 26px rgba(20,33,61,.08); }}
    .metric span {{ display: block; color: var(--muted); font-family: Segoe UI, Arial, sans-serif; font-size: 13px; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    .win {{ color: var(--win); font-weight: 700; }}
    .loss {{ color: var(--loss); font-weight: 700; }}
    .flat {{ color: var(--muted); font-weight: 700; }}
    .wrap {{ overflow: auto; background: white; border: 1px solid var(--line); box-shadow: 0 14px 36px rgba(20,33,61,.12); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; white-space: nowrap; font-family: Segoe UI, Arial, sans-serif; }}
    th {{ position: sticky; top: 0; background: var(--ink); color: white; text-align: left; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf0f5; }}
    tr:nth-child(even) {{ background: #fbfbf8; }}
    a {{ color: var(--ink); }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="note">{note}</p>
  {body_html}
  <script>
    function parseCell(text) {{
      const raw = String(text || '').trim();
      const cleaned = raw.replace(/[,%]/g, '').replace(/^\\+/, '');
      if (/^-?\\d+(\\.\\d+)?$/.test(cleaned)) return Number(cleaned);
      const date = Date.parse(raw);
      if (!Number.isNaN(date) && /^\\d{{4}}-\\d{{2}}-\\d{{2}}/.test(raw)) return date;
      return raw.toLowerCase();
    }}
    document.querySelectorAll('table').forEach((table) => {{
      const headers = table.querySelectorAll('th');
      headers.forEach((th, index) => {{
        th.addEventListener('click', () => {{
          const tbody = table.querySelector('tbody');
          const rows = Array.from(tbody.querySelectorAll('tr'));
          const current = th.dataset.sortDir === 'asc' ? 'desc' : 'asc';
          headers.forEach(h => h.dataset.sortDir = '');
          th.dataset.sortDir = current;
          rows.sort((a, b) => {{
            const av = parseCell(a.children[index]?.innerText);
            const bv = parseCell(b.children[index]?.innerText);
            if (typeof av === 'number' && typeof bv === 'number') return current === 'asc' ? av - bv : bv - av;
            return current === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
          }});
          rows.forEach(row => tbody.appendChild(row));
        }});
      }});
    }});
  </script>
  <script src="sortable_tables.js"></script>
</body>
</html>
"""


def _fmt_money(value: Any) -> str:
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return ""


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):+.2f}%"
    except Exception:
        return ""


def _pnl_class(value: Any) -> str:
    try:
        if isinstance(value, str):
            cleaned = value.replace(",", "").replace("%", "").replace("+", "").strip()
            numeric = float(cleaned)
        else:
            numeric = float(value)
    except Exception:
        return "flat"
    if numeric > 0:
        return "win"
    if numeric < 0:
        return "loss"
    return "flat"


def _html_table(frame: pd.DataFrame, *, pnl_columns: tuple[str, ...] = ()) -> str:
    if frame.empty:
        return "<p class=\"note\">No rows.</p>"
    headers = "".join(f"<th>{column}</th>" for column in frame.columns)
    body_rows = []
    for row in frame.to_dict("records"):
        cells = []
        for column in frame.columns:
            value = row.get(column, "")
            klass = _pnl_class(value) if column in pnl_columns else ""
            text = "" if pd.isna(value) else str(value)
            cells.append(f"<td class=\"{klass}\">{text}</td>" if klass else f"<td>{text}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<div class=\"wrap\"><table><thead><tr>{headers}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"


def _load_raw_ohlcv(symbols: list[str], as_of: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
    frame = pd.read_parquet(DATA_DIR / "ohlcv_master.parquet")
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    start = as_of - pd.Timedelta(days=lookback_days)
    return (
        frame[(frame["symbol"].isin(symbols)) & (frame["date"].between(start, as_of))]
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )


def _load_vnindex(as_of: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
    frame = pd.read_parquet(DATA_DIR / "index_master.parquet")
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    if "symbol" in frame.columns:
        frame = frame[frame["symbol"].astype(str).str.upper() == "VNINDEX"]
    start = as_of - pd.Timedelta(days=lookback_days)
    return frame[frame["date"].between(start, as_of)].sort_values("date").reset_index(drop=True)


def _build_features(symbols: list[str], as_of: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
    start = as_of - pd.Timedelta(days=lookback_days)
    result = build_feature_table(
        universe=symbols,
        start=start.strftime("%Y-%m-%d"),
        end=as_of.strftime("%Y-%m-%d"),
        root=ROOT,
    )
    features = result[0] if isinstance(result, tuple) else result
    features = enrich_features_with_rotation(features, ROOT)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    keep = [
        "date",
        "symbol",
        "close",
        "volume",
        "value",
        "value_ratio_20",
        "rs_percentile_20",
        "smart_money_score",
        "smart_money_score_delta",
        "CHDM50",
        "DS20",
        "edge_score",
        "mkt_regime_state",
        "mkt_regime_score",
        "mkt_CHDM20",
        "mkt_DS20",
        "vni_close",
        "vni_ret_20d",
    ]
    existing = [column for column in keep if column in features.columns]
    return features[features["date"].between(start, as_of)][existing].sort_values(["date", "symbol"]).reset_index(drop=True)


def _strategy_catalog() -> pd.DataFrame:
    frame = pd.DataFrame(catalog_rows())
    if frame.empty:
        return frame
    return frame.sort_values(["status", "portfolio_role", "strategy_id"]).reset_index(drop=True)


def _load_status(production_dir: Path) -> pd.DataFrame:
    path = production_dir / "production_status.json"
    if not path.exists():
        return pd.DataFrame()
    raw = json.loads(path.read_text(encoding="utf-8"))
    sleeve = raw.get("active_sleeve", {})
    rows = [
        {"key": "mode", "value": raw.get("mode")},
        {"key": "live_orders_enabled", "value": raw.get("live_orders_enabled")},
        {"key": "as_of_date", "value": raw.get("as_of_date")},
        {"key": "universe", "value": raw.get("universe")},
        {"key": "symbols", "value": raw.get("symbols")},
        {"key": "active_sleeve", "value": sleeve.get("sleeve_id")},
        {"key": "max_positions", "value": sleeve.get("max_positions")},
        {"key": "allowed_regimes", "value": ", ".join(sleeve.get("allowed_regimes", []))},
        {"key": "candidate_count", "value": raw.get("candidate_count")},
        {"key": "paper_buy_candidate_count", "value": raw.get("paper_buy_candidate_count")},
        {"key": "vetoed_candidate_count", "value": raw.get("vetoed_candidate_count")},
    ]
    return pd.DataFrame(rows)


def _latest_price_map(as_of: pd.Timestamp) -> dict[str, float]:
    frame = pd.read_parquet(DATA_DIR / "ohlcv_master.parquet")
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame = frame[frame["date"] <= as_of].sort_values(["symbol", "date"])
    latest = frame.groupby("symbol", sort=False).tail(1)
    return {str(row["symbol"]): float(row["close"]) for row in latest.to_dict("records")}


def _load_paper_portfolio_state(backtest_dir: Path, as_of: pd.Timestamp, max_positions: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades_path = backtest_dir / "baseline_p5_c10_trades.csv"
    equity_path = backtest_dir / "baseline_p5_c10_equity.csv"
    if not trades_path.exists() or not equity_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    trades = pd.read_csv(trades_path)
    equity = pd.read_csv(equity_path)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.normalize()
    trades["exit_date"] = pd.to_datetime(trades["exit_date"]).dt.normalize()
    equity["date"] = pd.to_datetime(equity["date"]).dt.normalize()

    end_of_data_open = (trades["exit_date"] == as_of) & (trades.get("exit_reason", "") == "END_OF_DATA")
    open_trades = trades[(trades["entry_date"] <= as_of) & ((trades["exit_date"] > as_of) | end_of_data_open)].copy()
    latest_prices = _latest_price_map(as_of)
    open_rows: list[dict[str, Any]] = []
    for row in open_trades.to_dict("records"):
        symbol = str(row["symbol"]).upper()
        entry_price = float(row["entry_price"])
        shares = int(float(row["shares"]))
        last_close = latest_prices.get(symbol)
        market_value = shares * last_close if last_close is not None else None
        unrealized_pnl = (last_close - entry_price) * shares if last_close is not None else None
        unrealized_pnl_pct = (last_close / entry_price - 1.0) * 100.0 if last_close is not None and entry_price else None
        open_rows.append(
            {
                "symbol": symbol,
                "strategy_name": row.get("edge_strategy_name"),
                "setup_type": row.get("setup_type"),
                "signal_date": row.get("signal_date"),
                "entry_date": pd.Timestamp(row["entry_date"]).strftime("%Y-%m-%d"),
                "entry_price": round(entry_price, 4),
                "shares": shares,
                "last_close": round(last_close, 4) if last_close is not None else None,
                "market_value": round(market_value, 0) if market_value is not None else None,
                "unrealized_pnl": round(unrealized_pnl, 0) if unrealized_pnl is not None else None,
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 2) if unrealized_pnl_pct is not None else None,
                "planned_exit_date": pd.Timestamp(row["exit_date"]).strftime("%Y-%m-%d"),
                "planned_exit_reason": row.get("exit_reason"),
                "edge_score": row.get("edge_score"),
                "edge_rank_score": row.get("edge_rank_score"),
            }
        )

    closed = trades[(trades["exit_date"] <= as_of) & ~end_of_data_open].copy()
    closed_rows = []
    for row in closed.to_dict("records"):
        closed_rows.append(
            {
                "symbol": row.get("symbol"),
                "strategy_name": row.get("edge_strategy_name"),
                "setup_type": row.get("setup_type"),
                "signal_date": row.get("signal_date"),
                "entry_date": pd.Timestamp(row["entry_date"]).strftime("%Y-%m-%d"),
                "exit_date": pd.Timestamp(row["exit_date"]).strftime("%Y-%m-%d"),
                "entry_price": round(float(row["entry_price"]), 4),
                "exit_price": round(float(row["exit_price"]), 4),
                "shares": int(float(row["shares"])),
                "pnl": round(float(row["pnl"]), 0),
                "pnl_pct": round(float(row["pnl_pct"]) * 100.0, 2),
                "holding_bars": int(float(row["holding_bars"])),
                "exit_reason": row.get("exit_reason"),
            }
        )

    latest_equity = equity[equity["date"] <= as_of].sort_values("date").tail(1)
    if latest_equity.empty:
        summary = pd.DataFrame()
    else:
        eq = latest_equity.iloc[0]
        nav = float(eq["equity"])
        cash = float(eq["cash"])
        positions = int(float(eq["positions"]))
        initial_nav = 100_000_000.0
        summary = pd.DataFrame(
            [
                {"metric": "Start date", "value": "2025-01-01"},
                {"metric": "As-of date", "value": as_of.strftime("%Y-%m-%d")},
                {"metric": "Initial NAV", "value": round(initial_nav, 0)},
                {"metric": "Current NAV", "value": round(nav, 0)},
                {"metric": "Return", "value": round((nav / initial_nav - 1.0) * 100.0, 2)},
                {"metric": "Cash", "value": round(cash, 0)},
                {"metric": "Cash / NAV", "value": round(cash / nav * 100.0, 2) if nav else None},
                {"metric": "Open positions", "value": positions},
                {"metric": "Max positions", "value": max_positions},
                {"metric": "Available slots", "value": max(0, max_positions - positions)},
            ]
        )
    holdings = pd.DataFrame(open_rows).sort_values("market_value", ascending=False) if open_rows else pd.DataFrame()
    closed_frame = pd.DataFrame(closed_rows).sort_values("exit_date", ascending=False) if closed_rows else pd.DataFrame()
    return summary, holdings, closed_frame


def _portfolio_overview_page(summary: pd.DataFrame, backtest_dir: Path) -> str:
    values = {str(row["metric"]): row["value"] for row in summary.to_dict("records")} if not summary.empty else {}

    def _plain_pct(value: Any) -> str:
        try:
            return f"{float(value):.2f}%"
        except Exception:
            return ""

    cards = [
        ("Start date", values.get("Start date", "")),
        ("Initial NAV", _fmt_money(values.get("Initial NAV"))),
        ("Current NAV", _fmt_money(values.get("Current NAV"))),
        ("Return", _fmt_pct(values.get("Return"))),
        ("Cash", _fmt_money(values.get("Cash"))),
        ("Closed realized PnL", _fmt_money(values.get("Closed realized PnL"))),
        ("Open unrealized PnL", _fmt_money(values.get("Open unrealized PnL"))),
        ("Total realized + open PnL", _fmt_money(values.get("Total realized + open PnL"))),
        ("Open / Max positions", f"{values.get('Open positions', '')} / {values.get('Max positions', '')}"),
        ("Available slots", values.get("Available slots", "")),
        ("Cash / NAV", _fmt_pct(values.get("Cash / NAV"))),
        ("Closed win rate", _plain_pct(values.get("Closed win rate"))),
        ("Closed profit factor", values.get("Closed profit factor", "")),
    ]
    card_html = "".join(
        f"<div class=\"metric\"><span>{label}</span><strong class=\"{_pnl_class(value) if label in {'Return', 'Closed realized PnL', 'Open unrealized PnL', 'Total realized + open PnL'} else ''}\">{value}</strong></div>"
        for label, value in cards
    )
    body = f"""
<div class="cards">{card_html}</div>
<h2>Portfolio Pages</h2>
<p class="note"><a href="06b_closed_trades.html">Closed trades</a> · <a href="06c_open_holdings.html">Open holdings</a></p>
<h2>Source</h2>
<p class="note">Reconstructed from baseline backtest files in <code>{backtest_dir}</code>.</p>
"""
    return _styled_page(
        "06a Paper Portfolio Overview",
        body,
        "Simulated portfolio state for the baseline MVP, not a broker account.",
    )


def _closed_trades_page(frame: pd.DataFrame) -> str:
    view = frame.copy()
    if not view.empty:
        view["pnl"] = view["pnl"].map(_fmt_money)
        view["pnl_pct_raw"] = frame["pnl_pct"]
        view["pnl_pct"] = frame["pnl_pct"].map(_fmt_pct)
        view = view.drop(columns=["pnl_pct_raw"], errors="ignore")
    return _styled_page(
        "06b Closed Trades",
        _html_table(view, pnl_columns=("pnl", "pnl_pct")),
        "Closed baseline trades: strategy, entry/exit date, realized PnL, holding bars, exit reason.",
    )


def _open_holdings_page(frame: pd.DataFrame) -> str:
    view = frame.copy()
    if not view.empty:
        for column in ["market_value", "unrealized_pnl"]:
            view[column] = view[column].map(_fmt_money)
        view["unrealized_pnl_pct_raw"] = frame["unrealized_pnl_pct"]
        view["unrealized_pnl_pct"] = frame["unrealized_pnl_pct"].map(_fmt_pct)
        view = view.drop(columns=["unrealized_pnl_pct_raw"], errors="ignore")
    return _styled_page(
        "06c Open Holdings",
        _html_table(view, pnl_columns=("unrealized_pnl", "unrealized_pnl_pct")),
        "Open baseline holdings as of latest local data. Green is unrealized gain; red is unrealized loss.",
    )


def _index_html(out_dir: Path, as_of: pd.Timestamp, symbols: list[str]) -> str:
    layers = [
        ("01 Raw OHLCV", "01_raw_ohlcv.html", "Dữ liệu giá/khối lượng từng mã từ parquet."),
        ("01 VNINDEX", "01_vnindex.html", "Context thị trường dùng để tính regime."),
        ("02 Feature Engine", "02_feature_engine.html", "RS, Smart Money, CHDM, DS, edge score, regime."),
        ("03 Strategy Pool", "03_strategy_pool.html", "Production strategies và research/shadow strategies."),
        ("04 Ranking", "04_ranking_signals.html", "Candidates sau strategy filters và ranking."),
        ("05 Shared Pool Actions", "05_risk_gate_actions.html", "Baseline actions: paper-buy hoặc watchlist; regime là risk label, không hard gate."),
        ("06a Portfolio Overview", "06a_portfolio_overview.html", "Ngày bắt đầu, NAV ban đầu/hiện tại, return, cash và slots."),
        ("06b Closed Trades", "06b_closed_trades.html", "Các vị thế đã đóng: ngày vào/ra, chiến lược, lãi/lỗ, exit reason."),
        ("06c Open Holdings", "06c_open_holdings.html", "Các vị thế đang mở: giá vào, giá hiện tại, PnL xanh/đỏ."),
        ("07 Production Status", "07_production_status.html", "Trạng thái dry-run và safety controls."),
        ("08 Watchlist", "08_watchlist.html", "Realtime watchlist waiting for entry zone and risk approval."),
    ]
    cards = "\n".join(
        f"""<a class="card" href="{href}">
  <strong>{title}</strong>
  <span>{desc}</span>
</a>"""
        for title, href, desc in layers
    )
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Trading System Layer Demo</title>
  <style>
    :root {{ --ink: #10243e; --gold: #b8872f; --paper: #f5f1e8; --card: #fffaf0; --muted: #667085; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(184,135,47,.24), transparent 32rem),
        linear-gradient(135deg, #f7f0df 0%, #eef3f6 100%);
      color: var(--ink);
      font-family: Georgia, 'Times New Roman', serif;
    }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 44px 26px; }}
    h1 {{ font-size: 42px; margin: 0 0 8px; letter-spacing: -.03em; }}
    .sub {{ color: var(--muted); font-family: Segoe UI, Arial, sans-serif; font-size: 16px; margin-bottom: 28px; }}
    .flow {{
      padding: 18px 20px;
      background: rgba(255,255,255,.68);
      border: 1px solid rgba(16,36,62,.12);
      border-radius: 18px;
      font-family: Consolas, monospace;
      line-height: 1.55;
      white-space: pre-wrap;
      box-shadow: 0 18px 48px rgba(16,36,62,.12);
      margin-bottom: 24px;
    }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; }}
    .card {{
      display: block;
      text-decoration: none;
      color: var(--ink);
      background: var(--card);
      border: 1px solid rgba(16,36,62,.14);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 10px 26px rgba(16,36,62,.08);
      transition: transform .16s ease, box-shadow .16s ease;
    }}
    .card:hover {{ transform: translateY(-2px); box-shadow: 0 16px 34px rgba(16,36,62,.14); }}
    .card strong {{ display: block; font-size: 18px; margin-bottom: 7px; }}
    .card span {{ color: var(--muted); font-family: Segoe UI, Arial, sans-serif; font-size: 14px; }}
  </style>
</head>
<body>
  <main>
    <h1>Trading System Layer Demo</h1>
    <div class="sub">As-of: {as_of:%Y-%m-%d} · Symbols shown: {", ".join(symbols)}</div>
    <div class="flow">OHLCV + VNINDEX
        ↓
Feature Engine
RS / Smart Money / CHDM / DS / Volume / Regime
        ↓
Strategy Pool
9 production strategies + research strategies shadow
        ↓
Ranking Engine
Chọn best candidates
        ↓
Shared Pool Portfolio
Max 5 slots / Paper-buy / Watchlist / Exit Agent
        ↓
Paper Portfolio State
NAV / Cash / Holdings / Available slots
        ↓
Production Dry-run
Tín hiệu cho phiên kế tiếp</div>
    <div class="grid">{cards}</div>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Export layer-by-layer pipeline demo.")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--production-dir", default=str(PRODUCTION_DIR))
    parser.add_argument("--backtest-dir", default=str(BASELINE_BACKTEST_DIR))
    parser.add_argument("--as-of", default="")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--max-symbols", type=int, default=12)
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of).normalize() if args.as_of else _latest_data_date()
    production_dir = Path(args.production_dir)
    backtest_dir = Path(args.backtest_dir)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "reports" / f"layered_pipeline_demo_{datetime.now():%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sortable_tables.js").write_text(SORTABLE_JS, encoding="utf-8")

    symbols = _selected_symbols(
        production_dir=production_dir,
        fallback=["CTG", "GEX", "HCM", "VCG", "MWG", "TCB"],
        max_symbols=args.max_symbols,
    )

    raw_ohlcv = _load_raw_ohlcv(symbols, as_of, args.lookback_days)
    vnindex = _load_vnindex(as_of, args.lookback_days)
    features = _build_features(symbols, as_of, args.lookback_days)
    strategy_catalog = _strategy_catalog()
    ranking = _read_csv_if_exists(production_dir / "rolling_signals_14d.csv")
    risk_gate = _read_csv_if_exists(production_dir / "rolling_candidate_actions_14d.csv")
    status = _load_status(production_dir)
    portfolio_summary, portfolio_holdings, closed_trades = _load_paper_portfolio_state(backtest_dir, as_of, max_positions=5)

    _write_table(raw_ohlcv, out_dir / "01_raw_ohlcv.csv", out_dir / "01_raw_ohlcv.html", "01 Raw OHLCV", "Raw stock bars from ohlcv_master.parquet.")
    _write_table(vnindex, out_dir / "01_vnindex.csv", out_dir / "01_vnindex.html", "01 VNINDEX", "Market index context from index_master.parquet.")
    _write_table(features, out_dir / "02_feature_engine.csv", out_dir / "02_feature_engine.html", "02 Feature Engine", "Causal features computed from data up to each date.")
    _write_table(strategy_catalog, out_dir / "03_strategy_pool.csv", out_dir / "03_strategy_pool.html", "03 Strategy Pool", "Production strategies are active; research/shadow strategies are exposed but not capital allocators.")
    _write_table(ranking, out_dir / "04_ranking_signals.csv", out_dir / "04_ranking_signals.html", "04 Ranking Engine", "Signals after strategy filters and ranking.")
    _write_table(
        risk_gate,
        out_dir / "05_risk_gate_actions.csv",
        out_dir / "05_risk_gate_actions.html",
        "05 Shared Pool Actions",
        "Baseline MVP actions. Market regime is shown as audit/risk context, not as a hard veto.",
    )
    portfolio_summary.to_csv(out_dir / "06a_portfolio_overview.csv", index=False, encoding="utf-8-sig")
    portfolio_holdings.to_csv(out_dir / "06c_open_holdings.csv", index=False, encoding="utf-8-sig")
    closed_trades.to_csv(out_dir / "06b_closed_trades.csv", index=False, encoding="utf-8-sig")
    (out_dir / "06a_portfolio_overview.html").write_text(_portfolio_overview_page(portfolio_summary, backtest_dir), encoding="utf-8")
    (out_dir / "06b_closed_trades.html").write_text(_closed_trades_page(closed_trades), encoding="utf-8")
    (out_dir / "06c_open_holdings.html").write_text(_open_holdings_page(portfolio_holdings), encoding="utf-8")
    legacy_portfolio = """<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=06a_portfolio_overview.html">
<title>Paper Portfolio</title></head>
<body><p>Portfolio pages were split for readability:
<a href="06a_portfolio_overview.html">Overview</a>,
<a href="06b_closed_trades.html">Closed trades</a>,
<a href="06c_open_holdings.html">Open holdings</a>.</p></body></html>
"""
    (out_dir / "06_paper_portfolio.html").write_text(legacy_portfolio, encoding="utf-8")
    _write_table(status, out_dir / "07_production_status.csv", out_dir / "07_production_status.html", "07 Production Dry-Run Status", "Safety and operating mode.")

    (out_dir / "index.html").write_text(_index_html(out_dir, as_of, symbols), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of": as_of.strftime("%Y-%m-%d"),
        "symbols": symbols,
        "out_dir": str(out_dir),
        "entrypoint": str(out_dir / "index.html"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
