"""Backtest hybrid Money Cycle + Smart Money Trace strategies on VN30.

This script uses:
- Money Cycle as market/regime and timing filter.
- Smart Money Trace as stock quality/ranking filter.
- A shared portfolio with VN-style costs, position caps, and next-open entries.

It intentionally does not use Price Volume or sector filters so the result
isolates the two money indicators.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float = 100_000.0
    max_positions: int = 8
    commission_rate: float = 0.001
    sell_tax_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_participation_rate: float = 0.10
    lot_size: int = 100
    min_position_value: float = 1_000.0
    stop_loss: float = 0.08
    take_profit: float = 0.25
    max_holding_bars: int = 60
    risk_on_exposure: float = 1.0
    neutral_exposure: float = 0.7
    risk_off_exposure: float = 0.3


@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    entry_value: float
    holding_bars: int = 0
    strategy: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Money Cycle + Smart Money Trace VN30 backtest")
    parser.add_argument("--from", dest="from_date", default="2022-05-08")
    parser.add_argument("--to", dest="to_date", default="2026-05-08")
    parser.add_argument("--output-dir", default="backtest_results/money_smt_hybrid_vn30")
    parser.add_argument("--max-positions", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    cfg = PortfolioConfig(max_positions=args.max_positions)
    start = pd.Timestamp(args.from_date)
    end = pd.Timestamp(args.to_date)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ohlcv, features = load_data(start, end)
    price_map = {s: g.reset_index(drop=True) for s, g in ohlcv.groupby("symbol")}
    date_index = {
        s: {pd.Timestamp(d): i for i, d in enumerate(g["date"])}
        for s, g in price_map.items()
    }

    strategies = {
        "MC_HEALTHY_SMT_STRONG": signal_mc_healthy_smt_strong,
        "MC_RESET_SMT_CONFIRM": signal_mc_reset_smt_confirm,
        "MC_RISK_ON_SMT_TOP": signal_mc_risk_on_smt_top,
        "MC_HYBRID_BALANCED": signal_mc_hybrid_balanced,
    }

    all_summaries = []
    for name, signal_fn in strategies.items():
        trades, equity = run_portfolio(
            name=name,
            features=features,
            price_map=price_map,
            date_index=date_index,
            signal_fn=signal_fn,
            start=start,
            end=end,
            top_n=args.top_n,
            cfg=cfg,
        )
        trades_path = output_dir / f"{name}_trades.csv"
        equity_path = output_dir / f"{name}_equity.csv"
        trades.to_csv(trades_path, index=False)
        equity.to_csv(equity_path, index=False)
        all_summaries.append({"strategy": name, **summarize_portfolio(trades, equity, cfg)})

    summary = pd.DataFrame(all_summaries).sort_values("total_return", ascending=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"\nOutputs: {output_dir.resolve()}")


def load_data(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(".")
    ohlcv = pd.read_parquet(root / "multiagents_trading_assistant/data/ohlcv_master.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.normalize()
    ohlcv["symbol"] = ohlcv["symbol"].astype(str).str.upper()
    ohlcv = ohlcv[
        ohlcv["symbol"].isin(VN30)
        & (ohlcv["date"] >= start - pd.Timedelta(days=370))
        & (ohlcv["date"] <= end)
    ].sort_values(["symbol", "date"]).reset_index(drop=True)

    chdm = pd.read_parquet(root / "data/research/money_cycle/chdm_by_symbol.parquet")
    ds = pd.read_parquet(root / "data/research/money_cycle/ds_by_symbol.parquet")
    mc = pd.read_parquet(root / "data/research/money_cycle/money_cycle_market.parquet")
    smt = pd.read_parquet(root / "data/research/smart_money_trace/smart_money_by_symbol.parquet")
    for df in [chdm, ds, mc, smt]:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for df in [chdm, ds, smt]:
        df["symbol"] = df["symbol"].astype(str).str.upper()
        df.query("symbol in @VN30", inplace=True)

    features = (
        ohlcv[["date", "symbol", "open", "high", "low", "close", "volume"]]
        .merge(chdm, on=["date", "symbol"], how="left")
        .merge(ds, on=["date", "symbol"], how="left")
        .merge(
            smt[
                [
                    "date", "symbol", "smart_money_score", "smart_money_state",
                    "rs_percentile_20", "distribution_days_10", "accumulation_days_10",
                    "value_flow_quality_score", "pullback_quality_score", "clv_ma5",
                    "ma20", "ma50",
                ]
            ],
            on=["date", "symbol"],
            how="left",
        )
        .merge(mc.add_prefix("mkt_").rename(columns={"mkt_date": "date"}), on="date", how="left")
        .sort_values(["symbol", "date"])
    )
    features["DS20_prev"] = features.groupby("symbol")["DS20"].shift(1)
    features["DS50_prev"] = features.groupby("symbol")["DS50"].shift(1)
    features["CHDM03_prev"] = features.groupby("symbol")["CHDM03"].shift(1)
    features["mkt_DS20_prev"] = features["mkt_DS20"].shift(1)
    return ohlcv, features


def signal_mc_healthy_smt_strong(day: pd.DataFrame) -> pd.Series:
    return (
        (day["mkt_CHDM20"] > 50)
        & (day["mkt_DS20"] < 0.45)
        & day["smart_money_state"].isin(["HOT_BUT_STRONG", "RESET_IN_UPTREND"])
        & (day["smart_money_score"] >= 65)
        & (day["rs_percentile_20"] >= 0.65)
        & (day["distribution_days_10"] <= 1)
    ).fillna(False)


def signal_mc_reset_smt_confirm(day: pd.DataFrame) -> pd.Series:
    return (
        (day["mkt_DS20"] < 0.55)
        & (day["CHDM50"] >= 50)
        & day["CHDM03"].between(20, 45)
        & (day["CHDM03"] > day["CHDM03_prev"])
        & (day["DS20"] <= day["DS20_prev"])
        & (day["smart_money_score"] >= 60)
        & (day["rs_percentile_20"] >= 0.55)
        & (day["distribution_days_10"] <= 1)
    ).fillna(False)


def signal_mc_risk_on_smt_top(day: pd.DataFrame) -> pd.Series:
    return (
        (day["mkt_CHDM50"] > 55)
        & (day["mkt_DS50"] < 0.40)
        & (day["mkt_DS20"] <= day["mkt_DS20_prev"])
        & (day["smart_money_score"] >= 70)
        & (day["rs_percentile_20"] >= 0.70)
        & (day["value_flow_quality_score"] >= 60)
        & (day["distribution_days_10"] <= 1)
        & (day["close"] > day["ma50"])
    ).fillna(False)


def signal_mc_hybrid_balanced(day: pd.DataFrame) -> pd.Series:
    healthy = (
        (day["mkt_CHDM20"] > 45)
        & (day["mkt_DS20"] < 0.50)
        & (day["mkt_DS50"] < 0.50)
    )
    stock_quality = (
        (day["smart_money_score"] >= 68)
        & (day["rs_percentile_20"] >= 0.62)
        & (day["value_flow_quality_score"] >= 55)
        & (day["distribution_days_10"] <= 1)
        & (day["close"] > day["ma50"])
    )
    pullback_timing = (
        (day["CHDM50"] >= 45)
        & (day["CHDM03"] > day["CHDM03_prev"])
        & (day["DS20"] <= day["DS20_prev"])
    )
    return (healthy & stock_quality & pullback_timing).fillna(False)


def run_portfolio(
    name: str,
    features: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    date_index: dict[str, dict[pd.Timestamp, int]],
    signal_fn,
    start: pd.Timestamp,
    end: pd.Timestamp,
    top_n: int,
    cfg: PortfolioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash = cfg.initial_capital
    positions: dict[str, Position] = {}
    trades = []
    equity_rows = []
    signal_calendar: dict[pd.Timestamp, pd.DataFrame] = {}

    dates = sorted(features[(features["date"] >= start) & (features["date"] <= end)]["date"].unique())
    for timestamp in dates:
        timestamp = pd.Timestamp(timestamp)

        for symbol in list(positions):
            frame = price_map[symbol]
            idx = date_index[symbol].get(timestamp)
            if idx is None:
                continue
            row = frame.iloc[idx]
            position = positions[symbol]
            position.holding_bars += 1
            reason = exit_reason(row, position, cfg)
            if reason:
                cash, trade = sell_position(timestamp, row, position, cash, reason, cfg)
                trades.append(trade)
                del positions[symbol]

        todays_signals = signal_calendar.pop(timestamp, pd.DataFrame())
        if not todays_signals.empty:
            equity = cash + mark_to_market(positions, price_map, timestamp)
            exposure = target_exposure(todays_signals)
            deployable = max(0.0, min(cash, equity * exposure - position_value(positions, price_map, timestamp)))
            open_slots = max(0, cfg.max_positions - len(positions))
            candidates = todays_signals[~todays_signals["symbol"].isin(positions)].head(min(top_n, open_slots))
            if not candidates.empty and deployable >= cfg.min_position_value:
                per_trade_budget = deployable / len(candidates)
                for candidate in candidates.to_dict("records"):
                    symbol = candidate["symbol"]
                    idx = date_index[symbol].get(timestamp)
                    if idx is None:
                        continue
                    cash, pos = buy_position(timestamp, price_map[symbol].iloc[idx], candidate, per_trade_budget, cash, cfg)
                    if pos is not None:
                        pos.strategy = name
                        positions[symbol] = pos

        day = features[features["date"] == timestamp].copy()
        signals = day[signal_fn(day)].copy()
        if not signals.empty:
            signals["_rank"] = (
                signals["smart_money_score"].rank(ascending=False, pct=True)
                + signals["rs_percentile_20"].rank(ascending=False, pct=True)
                + signals["value_flow_quality_score"].rank(ascending=False, pct=True)
                + signals["CHDM50"].rank(ascending=False, pct=True)
            )
            signals = signals.sort_values("_rank", ascending=False)
            next_open_rows = []
            for row in signals.to_dict("records"):
                symbol = row["symbol"]
                idx = date_index[symbol].get(timestamp)
                frame = price_map[symbol]
                if idx is not None and idx + 1 < len(frame):
                    next_date = pd.Timestamp(frame.iloc[idx + 1]["date"])
                    if next_date <= end:
                        next_open_rows.append((next_date, row))
            for next_date, row in next_open_rows:
                signal_calendar.setdefault(next_date, []).append(row)

        for next_date, rows in list(signal_calendar.items()):
            if isinstance(rows, list):
                signal_calendar[next_date] = pd.DataFrame(rows).sort_values("_rank", ascending=False)

        equity = cash + mark_to_market(positions, price_map, timestamp)
        equity_rows.append(
            {
                "date": timestamp,
                "equity": equity,
                "cash": cash,
                "positions": len(positions),
                "gross_exposure": position_value(positions, price_map, timestamp) / equity if equity else 0.0,
            }
        )

    final_date = pd.Timestamp(dates[-1])
    for symbol in list(positions):
        frame = price_map[symbol]
        idx = date_index[symbol].get(final_date)
        if idx is not None:
            cash, trade = sell_position(final_date, frame.iloc[idx], positions[symbol], cash, "END_OF_DATA", cfg)
            trades.append(trade)
            del positions[symbol]
    if equity_rows:
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["positions"] = 0
        equity_rows[-1]["gross_exposure"] = 0.0

    return pd.DataFrame(trades), pd.DataFrame(equity_rows)


def target_exposure(signals: pd.DataFrame) -> float:
    if signals.empty:
        return 0.0
    row = signals.iloc[0]
    if row.get("mkt_CHDM50", 0) > 60 and row.get("mkt_DS50", 1) < 0.35:
        return 1.0
    if row.get("mkt_CHDM20", 0) > 45 and row.get("mkt_DS20", 1) < 0.50:
        return 0.7
    return 0.3


def buy_position(timestamp, row, candidate, budget, cash, cfg):
    open_price = float(row["open"])
    volume = float(row["volume"])
    if not np.isfinite(open_price) or open_price <= 0 or volume <= 0:
        return cash, None
    fill_price = open_price * (1 + cfg.slippage_rate)
    gross_shares = min(budget / fill_price, volume * cfg.max_participation_rate)
    shares = int(gross_shares // cfg.lot_size * cfg.lot_size)
    if shares <= 0:
        return cash, None
    gross = shares * fill_price
    total_cost = gross * (1 + cfg.commission_rate)
    if total_cost > cash:
        shares = int((cash / (fill_price * (1 + cfg.commission_rate))) // cfg.lot_size * cfg.lot_size)
        gross = shares * fill_price
        total_cost = gross * (1 + cfg.commission_rate)
    if shares <= 0 or gross < cfg.min_position_value:
        return cash, None
    pos = Position(
        symbol=str(candidate["symbol"]),
        entry_date=timestamp,
        entry_price=fill_price,
        shares=shares,
        entry_value=gross,
    )
    return cash - total_cost, pos


def sell_position(timestamp, row, position, cash, reason, cfg):
    open_price = float(row["open"])
    fill_price = open_price * (1 - cfg.slippage_rate)
    gross = position.shares * fill_price
    fees = gross * (cfg.commission_rate + cfg.sell_tax_rate)
    net = gross - fees
    pnl = net - position.entry_value
    trade = {
        "strategy": position.strategy,
        "symbol": position.symbol,
        "entry_date": position.entry_date,
        "exit_date": timestamp,
        "entry_price": position.entry_price,
        "exit_price": fill_price,
        "shares": position.shares,
        "entry_value": position.entry_value,
        "exit_value": net,
        "pnl": pnl,
        "pnl_pct": pnl / position.entry_value if position.entry_value else 0.0,
        "exit_reason": reason,
        "holding_bars": position.holding_bars,
    }
    return cash + net, trade


def exit_reason(row, position, cfg):
    low = float(row["low"])
    high = float(row["high"])
    if low <= position.entry_price * (1 - cfg.stop_loss):
        return "STOP_LOSS"
    if high >= position.entry_price * (1 + cfg.take_profit):
        return "TAKE_PROFIT"
    if position.holding_bars >= cfg.max_holding_bars:
        return "MAX_HOLDING"
    return None


def mark_to_market(positions, price_map, timestamp):
    total = 0.0
    for symbol, position in positions.items():
        frame = price_map[symbol]
        rows = frame[frame["date"] <= timestamp]
        close = float(rows["close"].iloc[-1]) if not rows.empty else position.entry_price
        total += position.shares * close
    return total


def position_value(positions, price_map, timestamp):
    return mark_to_market(positions, price_map, timestamp)


def summarize_portfolio(trades: pd.DataFrame, equity: pd.DataFrame, cfg: PortfolioConfig) -> dict:
    if equity.empty:
        return {}
    eq = equity["equity"].astype(float)
    returns = eq.pct_change().fillna(0.0)
    total_return = eq.iloc[-1] / cfg.initial_capital - 1
    max_dd = (eq / eq.cummax() - 1).min()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    if trades.empty:
        win_rate = 0.0
        avg_trade = 0.0
        profit_factor = 0.0
        trade_count = 0
    else:
        pnl = trades["pnl"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = float((pnl > 0).mean())
        avg_trade = float(trades["pnl_pct"].mean())
        profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else np.inf
        trade_count = len(trades)
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": trade_count,
        "win_rate": win_rate,
        "avg_trade_pct": avg_trade,
        "profit_factor": profit_factor,
        "final_equity": eq.iloc[-1],
    }


if __name__ == "__main__":
    main()
