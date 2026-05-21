from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from multiagents_trading_assistant.backtest.live_pipeline import (
    LivePipelineBacktestConfig,
    LivePosition,
    _breakdown,
    _close_position,
    _compute_initial_sl,
    _edge_initial_sl,
    _edge_take_profit,
    _exit_decision,
    _last_row_on_or_before,
    _mark_to_market,
    _market_context_at,
    _next_calendar_date,
    _prepare_data,
    _prepare_single_frame,
    _row_at,
)
from multiagents_trading_assistant.edge_lab.features import build_feature_table
from multiagents_trading_assistant.edge_lab.hypothesis import evaluate_filters, load_hypotheses, rank_candidates
from multiagents_trading_assistant.edge_lab.live_signal import DEFAULT_LIVE_EDGE_FAMILY
from multiagents_trading_assistant.fetcher import get_vn30_symbols, get_vn100_symbols
from multiagents_trading_assistant.quantagents_backtest.metrics import compute_metrics


OUT_DIR = Path("backtest_results") / "mvp_execution_compare"
MASTER_PARQUET = Path("multiagents_trading_assistant") / "data" / "ohlcv_master.parquet"
CACHE_DIR = Path("multiagents_trading_assistant") / "cache"
AS_OF = "2026-05-12"
START_2025 = "2025-01-01"
START_4Y = "2022-05-12"
HYPOTHESIS_CONFIG = Path("multiagents_trading_assistant") / "edge_lab" / "configs" / "vn30_money_smt_hypotheses.json"


def _edge_setup_type(name: str) -> str:
    if "breakout" in name or "compression" in name:
        return "EDGE_BREAKOUT" if "breakout" in name else "EDGE_COMPRESSION"
    if "mean_reversion" in name:
        return "EDGE_MEAN_REVERSION"
    return "EDGE"


def build_signal_map(
    universe: str,
    start: str,
    end: str,
) -> tuple[dict[pd.Timestamp, list[dict]], dict[str, pd.DataFrame]]:
    if universe == "VN30":
        symbols = get_vn30_symbols()
    elif universe == "VN100":
        symbols = get_vn100_symbols()
    else:
        raise ValueError(f"Unsupported universe: {universe}")

    features, price_map = build_feature_table(symbols, start, end, root=".")
    if features.empty:
        return {}, price_map

    strategy_names = [item.strip() for item in DEFAULT_LIVE_EDGE_FAMILY.split(",") if item.strip()]
    wanted = {name: hyp for hyp in load_hypotheses(HYPOTHESIS_CONFIG) for name in [hyp.name] if hyp.name in strategy_names}
    signals_by_date: dict[pd.Timestamp, list[dict]] = {}

    for date, daily in features.groupby("date", sort=True):
        best_by_symbol: dict[str, dict] = {}
        for name in strategy_names:
            hypothesis = wanted.get(name)
            if hypothesis is None:
                continue
            mask = evaluate_filters(daily, hypothesis)
            ranked = rank_candidates(daily[mask], hypothesis)
            if ranked.empty:
                continue
            for rank_idx, row in enumerate(ranked.to_dict("records"), start=1):
                symbol = str(row["symbol"])
                rank_score = float(row.get("_edge_rank") or 0.0)
                candidate = {
                    "symbol": symbol,
                    "setup_type": _edge_setup_type(name),
                    "signal_date": pd.Timestamp(date),
                    "priority_score": min(100.0, 50.0 + rank_score * 50.0),
                    "indicators": {
                        "current_price": float(row.get("close") or 0.0),
                        "atr": float(row.get("atr14") or 0.0),
                        "ema50": float(row.get("ma50") or 0.0),
                        "support_levels": [float(row.get("ma50"))] if pd.notna(row.get("ma50")) else [],
                    },
                    "reasons": [f"Edge strategy passed: {name}"],
                    "edge_strategy_name": name,
                    "edge_strategy_passed": True,
                    "edge_score": row.get("edge_score"),
                    "edge_rank": rank_idx,
                    "edge_rank_score": round(rank_score, 4),
                    "edge_risk": dict(hypothesis.risk or {}),
                }
                current = best_by_symbol.get(symbol)
                if current is None or rank_score > float(current.get("edge_rank_score") or 0.0):
                    best_by_symbol[symbol] = candidate
        if best_by_symbol:
            orders = sorted(best_by_symbol.values(), key=lambda item: float(item["edge_rank_score"]), reverse=True)
            signals_by_date[pd.Timestamp(date)] = orders
    return signals_by_date, price_map


def derive_entry_zone(order: dict) -> tuple[float, float] | None:
    ind = order.get("indicators") or {}
    signal_close = float(ind.get("current_price") or 0.0)
    if signal_close <= 0:
        return None

    edge_name = str(order.get("edge_strategy_name") or "")
    supports = ind.get("support_levels") or []
    ema50 = float(ind.get("ema50") or 0.0)

    if edge_name == "breakout_after_accumulation_v3":
        return round(signal_close, 2), round(signal_close * 1.01, 2)
    if edge_name == "compression_breakout_smt_v1":
        return round(signal_close * 0.995, 2), round(signal_close * 1.01, 2)
    if edge_name == "mean_reversion_uptrend_ma50_v1":
        if supports:
            level = float(supports[0])
        elif ema50 > 0:
            level = ema50
        else:
            level = signal_close
        return round(level * 0.99, 2), round(level * 1.01, 2)

    setup_type = str(order.get("setup_type") or "")
    if setup_type == "EDGE_BREAKOUT":
        return round(signal_close, 2), round(signal_close * 1.01, 2)
    if setup_type == "EDGE_COMPRESSION":
        return round(signal_close * 0.995, 2), round(signal_close * 1.01, 2)
    if setup_type == "EDGE_MEAN_REVERSION":
        level = float(supports[0]) if supports else (ema50 if ema50 > 0 else signal_close)
        return round(level * 0.99, 2), round(level * 1.01, 2)
    return None


def intraday_touch_fill(order: dict, row: pd.Series) -> float | None:
    zone = derive_entry_zone(order)
    if not zone:
        return None
    low_zone, high_zone = zone
    open_ = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    if open_ <= 0:
        return None
    if low_zone <= open_ <= high_zone:
        return round(open_, 2)
    if open_ < low_zone and high >= low_zone:
        return round(low_zone, 2)
    if open_ > high_zone and low <= high_zone:
        return round(high_zone, 2)
    if low <= low_zone and high >= high_zone:
        return round(low_zone, 2)
    return None


def fill_entries(
    orders: list[dict],
    data: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    cash: float,
    positions: dict[str, LivePosition],
    cfg: LivePipelineBacktestConfig,
    model: str,
    counters: dict,
) -> float:
    slots = max(0, cfg.max_positions - len(positions))
    if slots <= 0:
        return cash
    orders = [order for order in orders if order["symbol"] not in positions][:slots]
    if not orders:
        return cash
    budget = cash / max(1, min(slots, len(orders)))
    for order in orders:
        counters["attempted_orders"] += 1
        row = _row_at(data[order["symbol"]], date)
        if row is None:
            continue
        if model == "eod_next_open":
            entry = float(row["open"]) * (1.0 + cfg.slippage_rate)
        else:
            entry = intraday_touch_fill(order, row)
        if entry is None or entry <= 0:
            continue
        gross_shares = int(budget // entry)
        shares = (gross_shares // cfg.lot_size) * cfg.lot_size if cfg.lot_size > 1 else gross_shares
        if shares <= 0:
            continue
        gross = shares * entry
        cost = gross * (1.0 + cfg.commission_rate)
        if cost > cash:
            continue

        ind = order["indicators"]
        edge_risk = dict(order.get("edge_risk") or {})
        stop_loss = _edge_initial_sl(entry, ind, edge_risk) if edge_risk else _compute_initial_sl(entry, ind)
        if stop_loss <= 0 or stop_loss >= entry:
            continue
        take_profit = _edge_take_profit(entry, stop_loss, edge_risk, cfg)

        cash -= cost
        counters["fills"] += 1
        positions[order["symbol"]] = LivePosition(
            symbol=order["symbol"],
            setup_type=order["setup_type"],
            signal_date=pd.Timestamp(order["signal_date"]),
            entry_date=date,
            entry_idx=-1,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            shares=shares,
            priority_score=float(order["priority_score"]),
            reasons=list(order.get("reasons") or []),
            edge_strategy_name=str(order.get("edge_strategy_name") or ""),
            edge_strategy_passed=bool(order.get("edge_strategy_passed")),
            edge_score=order.get("edge_score"),
            edge_rank=order.get("edge_rank"),
            edge_rank_score=order.get("edge_rank_score"),
            edge_risk=edge_risk,
            entry_atr=ind.get("atr"),
            highest_price=entry,
            holding_bars=0,
        )
    return cash


def run_model(
    universe_data: dict[str, pd.DataFrame],
    vnindex: pd.DataFrame,
    signal_map: dict[pd.Timestamp, list[dict]],
    *,
    cfg: LivePipelineBacktestConfig,
    model: str,
) -> dict:
    data = _prepare_data(universe_data)
    index_data = _prepare_single_frame(vnindex)
    calendar = sorted(set(index_data["date"]))
    cash = float(cfg.initial_capital)
    positions: dict[str, LivePosition] = {}
    pending_orders: dict[pd.Timestamp, list[dict]] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []
    counters = {"signal_days": 0, "signals": 0, "attempted_orders": 0, "fills": 0}

    date_to_pos = {date: idx for idx, date in enumerate(calendar)}
    start_ts = pd.Timestamp(cfg.start_date).normalize() if cfg.start_date else None
    end_ts = pd.Timestamp(cfg.end_date).normalize() if cfg.end_date else None

    last_processed_date = None
    for date in calendar[cfg.lookback :]:
        if end_ts is not None and date > end_ts:
            break
        last_processed_date = date
        day_idx = date_to_pos[date]

        todays_orders = pending_orders.pop(date, [])
        if todays_orders:
            cash = fill_entries(todays_orders, data, date, cash, positions, cfg, model, counters)

        for symbol in list(positions):
            row = _row_at(data[symbol], date)
            if row is None:
                continue
            position = positions[symbol]
            position.holding_bars = day_idx - date_to_pos.get(position.entry_date, day_idx)
            exit_reason, exit_price = _exit_decision(row, position, cfg)
            if exit_reason and position.holding_bars >= cfg.settlement_bars:
                cash, trade = _close_position(date, exit_price, exit_reason, position, cash, cfg)
                trades.append(trade)
                del positions[symbol]

        if start_ts is None or date >= start_ts:
            equity = cash + _mark_to_market(positions, data, date)
            equity_rows.append(
                {
                    "date": date,
                    "equity": equity,
                    "cash": cash,
                    "positions": len(positions),
                }
            )

        next_date = _next_calendar_date(calendar, day_idx)
        if next_date is None:
            continue
        if start_ts is not None and date < start_ts:
            continue
        market_ctx = _market_context_at(index_data, date, cfg.lookback)
        if not market_ctx.should_trade and not cfg.allow_downtrend_entries:
            continue
        if cfg.require_uptrend and market_ctx.reference_trend != "UPTREND":
            continue
        candidates = list(signal_map.get(date, []))
        candidates = [c for c in candidates if c["symbol"] not in positions]
        if candidates:
            counters["signal_days"] += 1
            counters["signals"] += len(candidates[: cfg.max_candidates_per_day])
            pending_orders.setdefault(next_date, []).extend(candidates[: cfg.max_candidates_per_day])

    final_date = last_processed_date or calendar[-1]
    for symbol in list(positions):
        row = _row_at(data[symbol], final_date)
        if row is None:
            row = _last_row_on_or_before(data[symbol], final_date)
        if row is None:
            continue
        cash, trade = _close_position(final_date, float(row["close"]), "END_OF_DATA", positions[symbol], cash, cfg)
        trades.append(trade)
        del positions[symbol]

    equity_frame = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame()
    equity_curve = equity_frame["equity"].rename(f"{model}_equity") if not equity_frame.empty else pd.Series(dtype=float)
    metrics = compute_metrics(equity_curve, trades) if not equity_curve.empty else {}
    metrics.update(
        {
            "signal_days": counters["signal_days"],
            "signals": counters["signals"],
            "attempted_orders": counters["attempted_orders"],
            "fills": counters["fills"],
            "fill_rate": (counters["fills"] / counters["attempted_orders"]) if counters["attempted_orders"] else 0.0,
            "avg_pnl_pct_per_trade": (sum(t["pnl_pct"] for t in trades) / len(trades)) if trades else 0.0,
        }
    )
    return {
        "metrics": metrics,
        "trades": trades,
        "setup_breakdown": _breakdown(trades, "setup_type"),
        "symbol_breakdown": _breakdown(trades, "symbol"),
        "equity_curve": equity_curve,
        "equity_frame": equity_frame,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    periods = [
        ("2025_to_2026-05-12", START_2025, AS_OF),
        ("2022-05-12_to_2026-05-12", START_4Y, AS_OF),
    ]
    universes = ["VN30", "VN100"]
    rows: list[dict] = []

    for universe in universes:
        print(f"\n=== Build local signal map {universe} up to {AS_OF} ===")
        signal_map, _ = build_signal_map(universe, START_4Y, AS_OF)
        print(f"[{universe}] signal dates={len(signal_map)}")
        print(f"=== Load local history {universe} up to {AS_OF} ===")
        universe_data, vnindex = load_local_history(universe, START_4Y, AS_OF)
        for label, start_date, end_date in periods:
            cfg = LivePipelineBacktestConfig(
                start_date=start_date,
                end_date=end_date,
                edge_strategy_name=DEFAULT_LIVE_EDGE_FAMILY,
            )
            print(f"\n[{universe}] {label} | model=EOD")
            eod = run_model(universe_data, vnindex, signal_map, cfg=cfg, model="eod_next_open")
            print(f"[{universe}] {label} | model=INTRADAY")
            intraday = run_model(universe_data, vnindex, signal_map, cfg=cfg, model="intraday_touch")

            for model_name, result in [("EOD", eod), ("INTRADAY", intraday)]:
                m = result["metrics"]
                rows.append(
                    {
                        "universe": universe,
                        "period": label,
                        "model": model_name,
                        "total_return_pct": round(float(m.get("total_return", 0.0)) * 100, 2),
                        "sharpe_ratio": round(float(m.get("sharpe_ratio", 0.0)), 3),
                        "max_drawdown_pct": round(float(m.get("max_drawdown", 0.0)) * 100, 2),
                        "win_rate_pct": round(float(m.get("win_rate", 0.0)) * 100, 2),
                        "number_of_trades": int(m.get("number_of_trades", 0)),
                        "signal_days": int(m.get("signal_days", 0)),
                        "signals": int(m.get("signals", 0)),
                        "attempted_orders": int(m.get("attempted_orders", 0)),
                        "fills": int(m.get("fills", 0)),
                        "fill_rate_pct": round(float(m.get("fill_rate", 0.0)) * 100, 2),
                        "avg_pnl_pct_per_trade": round(float(m.get("avg_pnl_pct_per_trade", 0.0)) * 100, 2),
                    }
                )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    pivot_parts = []
    for (universe, period), grp in summary.groupby(["universe", "period"], sort=False):
        row = {"universe": universe, "period": period}
        eod = grp[grp["model"] == "EOD"].iloc[0].to_dict()
        intraday = grp[grp["model"] == "INTRADAY"].iloc[0].to_dict()
        for key in [
            "total_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "win_rate_pct",
            "number_of_trades",
            "fills",
            "fill_rate_pct",
            "avg_pnl_pct_per_trade",
        ]:
            row[f"eod_{key}"] = eod[key]
            row[f"intraday_{key}"] = intraday[key]
            row[f"delta_{key}"] = round(float(intraday[key]) - float(eod[key]), 2)
        pivot_parts.append(row)
    compare = pd.DataFrame(pivot_parts)
    compare.to_csv(OUT_DIR / "comparison.csv", index=False)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== COMPARISON ===")
    print(compare.to_string(index=False))


def load_local_history(universe: str, start: str, end: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    df = pd.read_parquet(MASTER_PARQUET)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if universe == "VN30":
        symbols = get_vn30_symbols()
    elif universe == "VN100":
        symbols = get_vn100_symbols()
    else:
        raise ValueError(f"Unsupported universe: {universe}")

    warmup_start = start_ts - pd.Timedelta(days=420)
    sub = df[
        (df["symbol"].isin(symbols))
        & (df["date"] >= warmup_start)
        & (df["date"] <= end_ts)
    ].copy()
    universe_data: dict[str, pd.DataFrame] = {}
    for symbol, grp in sub.groupby("symbol", sort=False):
        frame = grp[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        universe_data[str(symbol)] = frame

    vnindex = load_cached_vnindex(start, end)
    return universe_data, vnindex


def load_cached_vnindex(start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    warmup_start = start_ts - pd.Timedelta(days=420)
    index_master = ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"
    if index_master.exists():
        try:
            frame = pd.read_parquet(index_master)
            if not frame.empty:
                frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
                if "symbol" in frame.columns:
                    frame = frame[frame["symbol"].astype(str).str.upper() == "VNINDEX"]
                frame = frame[(frame["date"] >= warmup_start) & (frame["date"] <= end_ts)]
                if not frame.empty:
                    return frame[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    candidates = sorted(CACHE_DIR.glob("VNINDEX_*_1D_hist.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            frame = pd.read_json(path)
            if frame.empty:
                continue
            frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
            frame = frame[(frame["date"] >= warmup_start) & (frame["date"] <= end_ts)]
            if frame.empty:
                continue
            return frame[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        except Exception:
            continue
    raise FileNotFoundError("No cached VNINDEX history found")


if __name__ == "__main__":
    main()
