"""Research Darvas Box breakout variants for Vietnamese equities.

Causality:
- Box top/bottom at T are computed from bars strictly before T.
- Breakout signals are evaluated after close of T.
- Entries and signal exits are filled at next open T+1.
- Intraday hard stops use same-day low and fill at min(open, stop).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multiagents_trading_assistant.fetcher import get_vn100_symbols, get_vn30_symbols
from multiagents_trading_assistant.quantagents_backtest.metrics import compute_metrics


ROOT = Path(__file__).resolve().parents[1]
OHLCV_PATH = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
INDEX_PATH = ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"
OUT_ROOT = ROOT / "backtest_results" / "darvas_box_research"


@dataclass(frozen=True)
class DarvasConfig:
    name: str
    box_window: int
    entry: str
    exit: str
    vni_filter: str = "uptrend"
    rs_min: float = 0.50
    volume_ratio_min: float = 1.0
    box_width_min: float = 0.03
    box_width_max: float = 0.18
    pretrend_min: float = 0.0
    breakout_buffer: float = 0.0
    stop_buffer: float = 0.01
    stop_loss: float = 0.09
    max_holding_bars: int = 70
    trailing_atr_mult: float = 2.8
    trailing_activation: float = 0.12


@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_idx: int
    entry_price: float
    shares: int
    stop_price: float
    highest_price: float
    entry_atr: float
    cfg_name: str


def resolve_symbols(universe: str) -> list[str]:
    name = universe.lower()
    if name == "vn100":
        return get_vn100_symbols()
    if name == "vn30":
        return get_vn30_symbols()
    return [item.strip().upper() for item in universe.split(",") if item.strip()]


def load_data(symbols: list[str], start: str, end: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    warmup = start_ts - pd.Timedelta(days=520)

    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.normalize()
    ohlcv["symbol"] = ohlcv["symbol"].astype(str).str.upper()
    ohlcv = ohlcv[
        ohlcv["symbol"].isin(symbols)
        & (ohlcv["date"] >= warmup)
        & (ohlcv["date"] <= end_ts)
    ].sort_values(["symbol", "date"])

    data = {
        symbol: group[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        for symbol, group in ohlcv.groupby("symbol", sort=False)
    }

    idx = pd.read_parquet(INDEX_PATH)
    idx["date"] = pd.to_datetime(idx["date"]).dt.tz_localize(None).dt.normalize()
    if "symbol" in idx.columns:
        idx = idx[idx["symbol"].astype(str).str.upper() == "VNINDEX"]
    idx = idx[(idx["date"] >= warmup) & (idx["date"] <= end_ts)].copy()
    idx = idx[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
    idx["ma20"] = idx["close"].rolling(20, min_periods=10).mean()
    idx["ma50"] = idx["close"].rolling(50, min_periods=25).mean()
    idx["ma200"] = idx["close"].rolling(200, min_periods=120).mean()
    idx["ma50_slope_10"] = idx["ma50"].pct_change(10)
    idx["ret_20d"] = idx["close"].pct_change(20)
    idx["vni_uptrend"] = (idx["close"] > idx["ma50"]) & (idx["ma50_slope_10"] > 0)
    idx["vni_healthy"] = (idx["close"] > idx["ma50"]) | (idx["ret_20d"] > 0)
    idx["vni_bull_phase"] = (
        (idx["close"] > idx["ma50"])
        & (idx["close"] > idx["ma200"])
        & (idx["ma50_slope_10"] > 0)
        & (idx["ret_20d"] >= 0)
    )
    return data, idx


def add_features(df: pd.DataFrame, cfg: DarvasConfig) -> pd.DataFrame:
    out = df.copy()
    h, l, c = out["high"], out["low"], out["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)

    box_top = h.shift(1).rolling(cfg.box_window, min_periods=max(10, cfg.box_window // 2)).max()
    box_bot = l.shift(1).rolling(cfg.box_window, min_periods=max(10, cfg.box_window // 2)).min()
    box_width = (box_top - box_bot) / c.replace(0, np.nan)

    out["box_top"] = box_top
    out["box_bot"] = box_bot
    out["box_mid"] = (box_top + box_bot) / 2.0
    out["box_width"] = box_width
    out["box_tight"] = box_width.between(cfg.box_width_min, cfg.box_width_max)
    out["breakout"] = c > box_top * (1.0 + cfg.breakout_buffer)
    out["breakout_new"] = out["breakout"] & (c.shift(1) <= box_top.shift(1) * (1.0 + cfg.breakout_buffer))
    out["box_retest"] = (
        out["breakout"].shift(1).rolling(8, min_periods=1).max().fillna(False).astype(bool)
        & (l <= box_top * 1.015)
        & (c > box_top)
        & (c > out["open"])
    )
    out["vol_ma20"] = out["volume"].rolling(20, min_periods=10).mean()
    out["volume_ratio_20"] = out["volume"] / out["vol_ma20"].replace(0, np.nan)
    out["ret20"] = c.pct_change(20)
    out["ret60"] = c.pct_change(60)
    out["ma20"] = c.rolling(20, min_periods=10).mean()
    out["ma50"] = c.rolling(50, min_periods=25).mean()
    out["atr14"] = tr.rolling(14, min_periods=14).mean()
    out["near_high_120"] = c >= h.shift(1).rolling(120, min_periods=60).max() * 0.85
    return out


def build_feature_maps(data: dict[str, pd.DataFrame], cfg: DarvasConfig, index: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {symbol: add_features(df, cfg) for symbol, df in data.items()}
    rs = pd.concat(
        [frame[["date", "ret20"]].assign(symbol=symbol) for symbol, frame in frames.items()],
        ignore_index=True,
    )
    rs["rs_pct"] = rs.groupby("date")["ret20"].rank(pct=True)
    for symbol, frame in frames.items():
        frame = frame.merge(rs[rs["symbol"] == symbol][["date", "rs_pct"]], on="date", how="left")
        frame = frame.merge(index[["date", "vni_uptrend", "vni_healthy", "vni_bull_phase"]], on="date", how="left")
        frames[symbol] = frame
    return frames


def entry_signal(row: pd.Series, cfg: DarvasConfig) -> bool:
    if cfg.vni_filter == "uptrend" and not bool(row.get("vni_uptrend", False)):
        return False
    if cfg.vni_filter == "healthy" and not bool(row.get("vni_healthy", False)):
        return False
    if cfg.vni_filter == "bull" and not bool(row.get("vni_bull_phase", False)):
        return False
    if float(row.get("rs_pct") or 0.0) < cfg.rs_min:
        return False
    if float(row.get("volume_ratio_20") or 0.0) < cfg.volume_ratio_min:
        return False
    if not bool(row.get("box_tight", False)):
        return False
    if float(row.get("ret60") or 0.0) < cfg.pretrend_min:
        return False
    if cfg.entry == "breakout":
        return bool(row.get("breakout_new", False))
    if cfg.entry == "breakout_near_high":
        return bool(row.get("breakout_new", False)) and bool(row.get("near_high_120", False))
    if cfg.entry == "retest":
        return bool(row.get("box_retest", False))
    return False


def exit_signal(row: pd.Series, pos: Position, holding_bars: int, cfg: DarvasConfig) -> tuple[str | None, float | None]:
    low = float(row["low"])
    open_ = float(row["open"])
    close = float(row["close"])
    if low <= pos.stop_price:
        return "BOX_STOP", min(open_, pos.stop_price)
    pos.highest_price = max(pos.highest_price, float(row["high"]))
    if cfg.exit == "atr_trail" and pos.highest_price >= pos.entry_price * (1.0 + cfg.trailing_activation):
        pos.stop_price = max(pos.stop_price, pos.highest_price - cfg.trailing_atr_mult * pos.entry_atr)
    if holding_bars >= cfg.max_holding_bars:
        return "MAX_HOLD", close
    if cfg.exit == "box_mid_close" and close < float(row.get("box_mid") or -np.inf):
        return "BOX_MID_CLOSE_NEXT_OPEN", None
    if cfg.exit == "ma20_close" and close < float(row.get("ma20") or -np.inf):
        return "MA20_CLOSE_NEXT_OPEN", None
    if cfg.exit == "box_bottom_close" and close < float(row.get("box_bot") or -np.inf):
        return "BOX_BOTTOM_CLOSE_NEXT_OPEN", None
    return None, None


def run_portfolio(data: dict[str, pd.DataFrame], index: pd.DataFrame, cfg: DarvasConfig, args) -> dict[str, Any]:
    frames = build_feature_maps(data, cfg, index)
    calendar = sorted(set(index["date"]))
    date_to_idx = {date: idx for idx, date in enumerate(calendar)}
    row_maps = {symbol: frame.set_index("date", drop=False) for symbol, frame in frames.items()}
    start_ts = pd.Timestamp(args.start).normalize()
    end_ts = pd.Timestamp(args.end).normalize()
    cash = float(args.initial_capital)
    positions: dict[str, Position] = {}
    pending_entries: dict[pd.Timestamp, list[str]] = {}
    pending_exits: dict[pd.Timestamp, list[tuple[str, str]]] = {}
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for date in calendar:
        if date < start_ts or date > end_ts:
            continue
        day_idx = date_to_idx[date]
        for symbol, reason in pending_exits.pop(date, []):
            pos = positions.get(symbol)
            row = row_maps.get(symbol, pd.DataFrame()).loc[date] if symbol in row_maps and date in row_maps[symbol].index else None
            if pos is None or row is None:
                continue
            exit_price = float(row["open"])
            cash += pos.shares * exit_price * (1.0 - args.commission_rate)
            trades.append(_trade(pos, date, exit_price, reason))
            del positions[symbol]

        orders = pending_entries.pop(date, [])
        slots = max(0, args.max_positions - len(positions))
        if slots and orders:
            budget = cash / min(slots, len(orders))
            for symbol in orders[:slots]:
                if symbol in positions or symbol not in row_maps or date not in row_maps[symbol].index:
                    continue
                row = row_maps[symbol].loc[date]
                entry_price = float(row["open"])
                atr = float(row.get("atr14") or entry_price * 0.03)
                box_stop = float(row.get("box_bot") or entry_price * (1.0 - cfg.stop_loss)) * (1.0 - cfg.stop_buffer)
                stop_price = max(entry_price * (1.0 - cfg.stop_loss), box_stop)
                shares = int((budget // entry_price) // args.lot_size) * args.lot_size
                cost = shares * entry_price * (1.0 + args.commission_rate)
                if shares <= 0 or cost > cash:
                    continue
                cash -= cost
                positions[symbol] = Position(symbol, date, day_idx, entry_price, shares, stop_price, entry_price, atr, cfg.name)

        next_date = _next_date(calendar, day_idx)
        for symbol in list(positions):
            if symbol not in row_maps or date not in row_maps[symbol].index:
                continue
            pos = positions[symbol]
            reason, price = exit_signal(row_maps[symbol].loc[date], pos, day_idx - pos.entry_idx, cfg)
            if reason is None:
                continue
            if price is None and next_date is not None:
                pending_exits.setdefault(next_date, []).append((symbol, reason))
                continue
            cash += pos.shares * float(price) * (1.0 - args.commission_rate)
            trades.append(_trade(pos, date, float(price), reason))
            del positions[symbol]

        equity = cash + sum(pos.shares * _close_at(row_maps, pos.symbol, date, pos.entry_price) for pos in positions.values())
        equity_rows.append({"date": date, "equity": equity, "cash": cash, "positions": len(positions)})

        if next_date is None:
            continue
        candidates = []
        for symbol, frame in row_maps.items():
            if symbol in positions or date not in frame.index:
                continue
            row = frame.loc[date]
            if entry_signal(row, cfg):
                score = (
                    float(row.get("rs_pct") or 0.0)
                    + 0.25 * float(row.get("volume_ratio_20") or 0.0)
                    - 1.5 * float(row.get("box_width") or 0.0)
                    + 0.2 * float(row.get("ret60") or 0.0)
                )
                candidates.append((score, symbol))
        candidates.sort(reverse=True)
        if candidates:
            pending_entries.setdefault(next_date, []).extend([symbol for _, symbol in candidates[: args.max_candidates_per_day]])

    final_date = min(pd.Timestamp(args.end).normalize(), calendar[-1])
    for symbol, pos in list(positions.items()):
        price = _close_at(row_maps, symbol, final_date, pos.entry_price)
        cash += pos.shares * price * (1.0 - args.commission_rate)
        trades.append(_trade(pos, final_date, price, "END_OF_DATA"))
        del positions[symbol]

    equity_frame = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame()
    metrics = compute_metrics(equity_frame["equity"], trades) if not equity_frame.empty else {}
    return {"metrics": metrics, "trades": trades, "equity": equity_frame}


def _trade(pos: Position, exit_date: pd.Timestamp, exit_price: float, reason: str) -> dict[str, Any]:
    return {
        "symbol": pos.symbol,
        "entry_date": pos.entry_date,
        "exit_date": exit_date,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "shares": pos.shares,
        "pnl": pos.shares * (exit_price - pos.entry_price),
        "pnl_pct": exit_price / pos.entry_price - 1.0,
        "holding_bars": (exit_date - pos.entry_date).days,
        "exit_reason": reason,
        "strategy": pos.cfg_name,
    }


def _next_date(calendar: list[pd.Timestamp], idx: int) -> pd.Timestamp | None:
    return calendar[idx + 1] if idx + 1 < len(calendar) else None


def _close_at(row_maps: dict[str, pd.DataFrame], symbol: str, date: pd.Timestamp, default: float) -> float:
    frame = row_maps.get(symbol)
    if frame is None or frame.empty:
        return default
    rows = frame[frame.index <= date]
    if rows.empty:
        return default
    return float(rows.iloc[-1]["close"])


def generate_configs() -> list[DarvasConfig]:
    configs = []
    for box_window in [15, 20, 30, 40]:
        for entry in ["breakout", "breakout_near_high", "retest"]:
            for exit_name in ["box_bottom_close", "box_mid_close", "ma20_close", "atr_trail"]:
                for regime in ["uptrend", "bull"]:
                    for rs_min in [0.50, 0.65]:
                        for vol_min in [1.0, 1.3]:
                            name = f"darvas_w{box_window}_{entry}_{exit_name}_{regime}_rs{int(rs_min*100)}_v{str(vol_min).replace('.', 'p')}"
                            configs.append(DarvasConfig(
                                name=name,
                                box_window=box_window,
                                entry=entry,
                                exit=exit_name,
                                vni_filter=regime,
                                rs_min=rs_min,
                                volume_ratio_min=vol_min,
                                pretrend_min=0.0 if entry != "breakout_near_high" else 0.05,
                                max_holding_bars=90 if exit_name in {"atr_trail", "box_bottom_close"} else 55,
                            ))
    return configs


def generate_focused_configs() -> list[DarvasConfig]:
    configs = []
    for box_window in [20, 30, 40]:
        for entry in ["breakout", "breakout_near_high"]:
            for exit_name in ["box_bottom_close", "atr_trail"]:
                for regime in ["uptrend", "bull"]:
                    for rs_min in [0.50, 0.65, 0.75]:
                        for vol_min in [1.0, 1.2]:
                            name = f"darvas_focus_w{box_window}_{entry}_{exit_name}_{regime}_rs{int(rs_min*100)}_v{str(vol_min).replace('.', 'p')}"
                            configs.append(DarvasConfig(
                                name=name,
                                box_window=box_window,
                                entry=entry,
                                exit=exit_name,
                                vni_filter=regime,
                                rs_min=rs_min,
                                volume_ratio_min=vol_min,
                                box_width_min=0.035,
                                box_width_max=0.16,
                                pretrend_min=0.0 if entry == "breakout" else 0.05,
                                max_holding_bars=90,
                            ))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--label", default="darvas_box_search")
    parser.add_argument("--initial-capital", type=float, default=100_000_000)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    parser.add_argument("--commission-rate", type=float, default=0.001)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--focused", action="store_true")
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.label}_{args.universe}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)
    symbols = resolve_symbols(args.universe)
    data, index = load_data(symbols, args.start, args.end)
    configs = generate_focused_configs() if args.focused else generate_configs()
    if args.limit > 0:
        configs = configs[: args.limit]
    print(f"[darvas] configs={len(configs)} symbols={len(data)} period={args.start}->{args.end}")

    rows = []
    payloads = []
    for i, cfg in enumerate(configs, 1):
        result = run_portfolio(data, index, cfg, args)
        metrics = result["metrics"]
        row = {
            **asdict(cfg),
            "total_return_pct": round(float(metrics.get("total_return", 0.0)) * 100, 2),
            "sharpe_ratio": round(float(metrics.get("sharpe_ratio", 0.0)), 3),
            "max_drawdown_pct": round(float(metrics.get("max_drawdown", 0.0)) * 100, 2),
            "win_rate_pct": round(float(metrics.get("win_rate", 0.0)) * 100, 2),
            "number_of_trades": int(metrics.get("number_of_trades", 0)),
        }
        rows.append(row)
        if i % 25 == 0:
            print(f"[darvas] {i}/{len(configs)}")
        if len(result["trades"]) >= 10:
            payloads.append((row["total_return_pct"], cfg.name, result))

    summary = pd.DataFrame(rows).sort_values(["total_return_pct", "sharpe_ratio"], ascending=[False, False])
    summary.to_csv(out_dir / "summary.csv", index=False)
    for _, name, payload in sorted(payloads, reverse=True)[: args.top]:
        pd.DataFrame(payload["trades"]).to_csv(out_dir / f"{name}_trades.csv", index=False)
        if not payload["equity"].empty:
            payload["equity"].to_csv(out_dir / f"{name}_equity.csv")
    print(summary.head(args.top).to_string(index=False))
    print(f"[darvas] saved {out_dir}")


if __name__ == "__main__":
    main()
