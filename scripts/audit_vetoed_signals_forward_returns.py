"""Look-ahead audit for recently vetoed production dry-run signals.

This is not production logic. It intentionally looks forward to evaluate whether
the risk gate was helpful or too conservative on recent signals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OHLCV_PATH = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"


def _load_prices() -> pd.DataFrame:
    frame = pd.read_parquet(OHLCV_PATH)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def _future_row(symbol_frame: pd.DataFrame, signal_date: pd.Timestamp, bars_forward: int) -> pd.Series | None:
    rows = symbol_frame[symbol_frame["date"] >= signal_date].reset_index(drop=True)
    if rows.empty:
        return None
    idx = bars_forward
    if idx >= len(rows):
        return None
    return rows.iloc[idx]


def _next_row(symbol_frame: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Series | None:
    rows = symbol_frame[symbol_frame["date"] > signal_date].reset_index(drop=True)
    if rows.empty:
        return None
    return rows.iloc[0]


def audit(actions_path: Path, out_path: Path) -> pd.DataFrame:
    actions = pd.read_csv(actions_path)
    if actions.empty:
        out = pd.DataFrame()
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        return out

    actions["date"] = pd.to_datetime(actions["date"]).dt.normalize()
    actions["symbol"] = actions["symbol"].astype(str).str.upper()
    prices = _load_prices()
    by_symbol = {symbol: group.reset_index(drop=True) for symbol, group in prices.groupby("symbol", sort=False)}

    rows = []
    for item in actions.to_dict("records"):
        symbol = item["symbol"]
        signal_date = pd.Timestamp(item["date"]).normalize()
        sf = by_symbol.get(symbol)
        if sf is None or sf.empty:
            continue
        signal_rows = sf[sf["date"] == signal_date]
        if signal_rows.empty:
            continue
        signal_bar = signal_rows.iloc[0]
        next_bar = _next_row(sf, signal_date)
        entry_close = float(signal_bar["close"])
        entry_next_open = float(next_bar["open"]) if next_bar is not None else None
        row = {
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "action": item.get("action"),
            "strategy_name": item.get("strategy_name"),
            "mkt_regime_state": item.get("mkt_regime_state"),
            "edge_rank_score": item.get("edge_rank_score"),
            "signal_close": entry_close,
            "next_open_date": pd.Timestamp(next_bar["date"]).strftime("%Y-%m-%d") if next_bar is not None else None,
            "next_open": entry_next_open,
        }
        for bars in [1, 3, 5, 10]:
            future = _future_row(sf, signal_date, bars)
            close_key = f"ret_close_to_plus_{bars}b_pct"
            open_key = f"ret_next_open_to_plus_{bars}b_close_pct"
            if future is None:
                row[close_key] = None
                row[open_key] = None
                continue
            future_close = float(future["close"])
            row[f"plus_{bars}b_date"] = pd.Timestamp(future["date"]).strftime("%Y-%m-%d")
            row[f"plus_{bars}b_close"] = future_close
            row[close_key] = round((future_close / entry_close - 1.0) * 100.0, 2)
            if entry_next_open and entry_next_open > 0:
                row[open_key] = round((future_close / entry_next_open - 1.0) * 100.0, 2)
            else:
                row[open_key] = None
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rows = []
    for action, group in frame.groupby("action", dropna=False):
        row = {"action": action, "count": len(group)}
        for col in [
            "ret_close_to_plus_1b_pct",
            "ret_close_to_plus_3b_pct",
            "ret_close_to_plus_5b_pct",
            "ret_close_to_plus_10b_pct",
        ]:
            valid = pd.to_numeric(group[col], errors="coerce").dropna() if col in group else pd.Series(dtype=float)
            row[f"{col}_avg"] = round(float(valid.mean()), 2) if not valid.empty else None
            row[f"{col}_win_rate"] = round(float((valid > 0).mean()) * 100.0, 2) if not valid.empty else None
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Look-ahead audit for vetoed recent dry-run signals.")
    parser.add_argument(
        "--actions",
        default=str(ROOT / "reports" / "production_demo_2026-05-17" / "rolling_candidate_actions_14d.csv"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "reports" / "production_demo_2026-05-17" / "risk_gate_forward_audit_14d.csv"),
    )
    args = parser.parse_args()

    out = audit(Path(args.actions), Path(args.out))
    summary = _summary(out)
    summary_path = Path(args.out).with_name(Path(args.out).stem + "_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("Audit rows:", len(out))
    print(out.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nExported: {args.out}")
    print(f"Exported: {summary_path}")


if __name__ == "__main__":
    main()
