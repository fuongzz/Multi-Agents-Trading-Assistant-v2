"""Search Ichimoku variants adapted for Vietnamese equities.

The runner is intentionally independent from edge_lab so it can test Ichimoku
parameter sets and exits that are not expressible as static JSON filters yet.

Causality:
- Signals are evaluated after close of date T.
- Entries and signal exits are filled at next open T+1.
- Hard stops use same-day low and fill at min(open, stop), a conservative
  exchange-realistic assumption for gap-down stops.
- Chikou confirmation is represented as close(T) > close(T-shift), not future
  data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from multiagents_trading_assistant.fetcher import get_vn100_symbols, get_vn30_symbols
from multiagents_trading_assistant.quantagents_backtest.metrics import compute_metrics


ROOT = Path(__file__).resolve().parents[1]
OHLCV_PATH = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
INDEX_PATH = ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"
OUT_ROOT = ROOT / "backtest_results" / "ichimoku_vn_research"


@dataclass(frozen=True)
class IchConfig:
    name: str
    tenkan: int
    kijun: int
    senkou_b: int
    shift: int
    entry: str
    exit: str
    vni_filter: str = "uptrend"
    rs_min: float = 0.50
    volume_ratio_min: float = 0.80
    no_chase_max_kijun: float = 0.12
    cloud_width_min: float = 0.0
    cloud_width_max: float = 0.18
    market_dd_120_max: float = 0.16
    stop_loss: float = 0.08
    take_profit: float | None = None
    max_holding_bars: int = 80
    trailing_atr_mult: float = 2.5
    trailing_activation: float = 0.10


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
    idx["ma50"] = idx["close"].rolling(50, min_periods=25).mean()
    idx["ma200"] = idx["close"].rolling(200, min_periods=120).mean()
    idx["ma50_slope_10"] = idx["ma50"].pct_change(10)
    idx["ret_20d"] = idx["close"].pct_change(20)
    idx["drawdown_120"] = idx["close"] / idx["close"].rolling(120, min_periods=60).max() - 1.0
    idx["vni_uptrend"] = (idx["close"] > idx["ma50"]) & (idx["ma50_slope_10"] > 0)
    idx["vni_healthy"] = (idx["close"] > idx["ma50"]) | (idx["ret_20d"] > 0)
    idx["vni_bull_phase"] = (
        (idx["close"] > idx["ma50"])
        & (idx["close"] > idx["ma200"])
        & (idx["ma50_slope_10"] > 0)
        & (idx["ret_20d"] >= 0)
    )
    return data, idx


def add_features(df: pd.DataFrame, cfg: IchConfig) -> pd.DataFrame:
    out = df.copy()
    h, l, c = out["high"], out["low"], out["close"]
    tenkan = (h.rolling(cfg.tenkan, min_periods=cfg.tenkan).max() + l.rolling(cfg.tenkan, min_periods=cfg.tenkan).min()) / 2
    kijun = (h.rolling(cfg.kijun, min_periods=cfg.kijun).max() + l.rolling(cfg.kijun, min_periods=cfg.kijun).min()) / 2
    senkou_a_raw = (tenkan + kijun) / 2
    senkou_b_raw = (h.rolling(cfg.senkou_b, min_periods=cfg.senkou_b).max() + l.rolling(cfg.senkou_b, min_periods=cfg.senkou_b).min()) / 2
    senkou_a = senkou_a_raw.shift(cfg.shift)
    senkou_b = senkou_b_raw.shift(cfg.shift)
    cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)

    out["tenkan"] = tenkan
    out["kijun"] = kijun
    out["cloud_top"] = cloud_top
    out["cloud_bot"] = cloud_bot
    out["future_green"] = senkou_a_raw > senkou_b_raw
    out["cloud_green"] = senkou_a > senkou_b
    out["chikou_free"] = c > c.shift(cfg.shift)
    out["chikou_high_free"] = c > h.rolling(cfg.shift, min_periods=max(5, cfg.shift // 2)).max().shift(cfg.shift)
    out["tk_bull"] = tenkan > kijun
    out["tk_cross_bull"] = out["tk_bull"] & (out["tk_bull"].shift(1) == False)
    out["tk_death"] = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))
    out["above_cloud"] = c > cloud_top
    out["below_cloud"] = c < cloud_bot
    out["cloud_break"] = out["above_cloud"] & (c.shift(1) <= cloud_top.shift(1))
    out["cloud_width"] = (cloud_top - cloud_bot) / c.replace(0, np.nan)
    out["cloud_width_expanding"] = out["cloud_width"] > out["cloud_width"].shift(5)
    out["recent_cloud_break_30"] = out["cloud_break"].rolling(30, min_periods=1).max().shift(1).fillna(False).astype(bool)
    out["perfect_bull"] = out["tk_bull"] & out["above_cloud"] & out["cloud_green"] & out["future_green"] & out["chikou_free"]
    out["perfect_bull_signal"] = out["perfect_bull"] & (out["perfect_bull"].shift(1) == False)
    out["kijun_bounce"] = out["above_cloud"] & (out["low"] <= kijun * 1.015) & (c > kijun) & (c > out["open"])
    out["qmv_retest"] = (
        out["recent_cloud_break_30"]
        & out["above_cloud"]
        & out["tk_bull"]
        & (out["low"] <= pd.concat([kijun * 1.02, cloud_top * 1.02], axis=1).max(axis=1))
        & (c > pd.concat([kijun, cloud_top], axis=1).max(axis=1))
        & (c > out["open"])
    )
    out["kumo_reclaim"] = (
        out["above_cloud"]
        & (c.shift(1) <= cloud_top.shift(1) * 1.01)
        & out["future_green"]
        & (tenkan >= kijun * 0.995)
    )
    out["high20_prev"] = h.rolling(20, min_periods=10).max().shift(1)
    out["high55_prev"] = h.rolling(55, min_periods=25).max().shift(1)
    out["turtle20"] = c > out["high20_prev"]
    out["turtle55"] = c > out["high55_prev"]
    out["atr14"] = tr.rolling(14, min_periods=14).mean()
    out["vol_ma20"] = out["volume"].rolling(20, min_periods=10).mean()
    out["volume_ratio_20"] = out["volume"] / out["vol_ma20"].replace(0, np.nan)
    out["ret20"] = c.pct_change(20)
    out["kijun_distance"] = c / kijun.replace(0, np.nan) - 1.0
    return out


def build_feature_maps(data: dict[str, pd.DataFrame], cfg: IchConfig, index: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {symbol: add_features(df, cfg) for symbol, df in data.items()}
    rs_rows = []
    for symbol, frame in frames.items():
        rs_rows.append(frame[["date", "ret20"]].assign(symbol=symbol))
    rs = pd.concat(rs_rows, ignore_index=True)
    rs["rs_pct"] = rs.groupby("date")["ret20"].rank(pct=True)
    for symbol, frame in frames.items():
        frame = frame.merge(rs[rs["symbol"] == symbol][["date", "rs_pct"]], on="date", how="left")
        frame = frame.merge(index[["date", "vni_uptrend", "vni_healthy", "vni_bull_phase", "drawdown_120"]], on="date", how="left")
        frames[symbol] = frame
    return frames


def entry_signal(row: pd.Series, cfg: IchConfig) -> bool:
    if cfg.vni_filter == "uptrend" and not bool(row.get("vni_uptrend", False)):
        return False
    if cfg.vni_filter == "healthy" and not bool(row.get("vni_healthy", False)):
        return False
    if cfg.vni_filter == "bull" and not bool(row.get("vni_bull_phase", False)):
        return False
    if float(row.get("drawdown_120") or 0.0) < -cfg.market_dd_120_max:
        return False
    if float(row.get("rs_pct") or 0.0) < cfg.rs_min:
        return False
    if float(row.get("volume_ratio_20") or 0.0) < cfg.volume_ratio_min:
        return False
    if float(row.get("kijun_distance") or 99.0) > cfg.no_chase_max_kijun:
        return False
    cloud_width = float(row.get("cloud_width") or 0.0)
    if cloud_width < cfg.cloud_width_min or cloud_width > cfg.cloud_width_max:
        return False
    if cfg.entry == "perfect_bull":
        return bool(row.get("perfect_bull_signal", False))
    if cfg.entry == "tk_cross_cloud":
        return bool(row.get("tk_cross_bull", False)) and bool(row.get("above_cloud", False)) and bool(row.get("chikou_free", False))
    if cfg.entry == "cloud_break":
        return bool(row.get("cloud_break", False)) and bool(row.get("tk_bull", False)) and bool(row.get("chikou_free", False))
    if cfg.entry == "kijun_bounce":
        return bool(row.get("kijun_bounce", False)) and bool(row.get("tk_bull", False)) and bool(row.get("chikou_free", False))
    if cfg.entry == "qmv_retest":
        return (
            bool(row.get("qmv_retest", False))
            and bool(row.get("future_green", False))
            and bool(row.get("chikou_high_free", False))
        )
    if cfg.entry == "chikou_high_break":
        return (
            bool(row.get("cloud_break", False))
            and bool(row.get("tk_bull", False))
            and bool(row.get("future_green", False))
            and bool(row.get("chikou_high_free", False))
        )
    if cfg.entry == "kumo_reclaim":
        return bool(row.get("kumo_reclaim", False)) and bool(row.get("chikou_free", False))
    if cfg.entry == "turtle20_ich":
        return bool(row.get("turtle20", False)) and bool(row.get("above_cloud", False)) and bool(row.get("tk_bull", False))
    if cfg.entry == "turtle55_ich":
        return bool(row.get("turtle55", False)) and bool(row.get("above_cloud", False)) and bool(row.get("chikou_free", False))
    return False


def exit_signal(row: pd.Series, pos: Position, holding_bars: int, cfg: IchConfig) -> tuple[str | None, float | None]:
    low = float(row["low"])
    open_ = float(row["open"])
    close = float(row["close"])
    if low <= pos.stop_price:
        return "SL", min(open_, pos.stop_price)
    if cfg.take_profit is not None and float(row["high"]) >= pos.entry_price * (1.0 + cfg.take_profit):
        return "TP", max(open_, pos.entry_price * (1.0 + cfg.take_profit))
    pos.highest_price = max(pos.highest_price, float(row["high"]))
    if cfg.exit == "atr_trail" and pos.highest_price >= pos.entry_price * (1.0 + cfg.trailing_activation):
        trail = pos.highest_price - cfg.trailing_atr_mult * pos.entry_atr
        pos.stop_price = max(pos.stop_price, trail)
    if holding_bars >= cfg.max_holding_bars:
        return "MAX_HOLD", close
    if cfg.exit == "tk_death" and bool(row.get("tk_death", False)):
        return "TK_DEATH_NEXT_OPEN", None
    if cfg.exit == "kijun_close" and close < float(row.get("kijun") or -np.inf):
        return "KIJUN_CLOSE_NEXT_OPEN", None
    if cfg.exit == "cloud_close" and close < float(row.get("cloud_bot") or -np.inf):
        return "CLOUD_CLOSE_NEXT_OPEN", None
    if cfg.exit == "ich_ladder":
        pnl = close / pos.entry_price - 1.0
        tenkan = float(row.get("tenkan") or -np.inf)
        kijun = float(row.get("kijun") or -np.inf)
        cloud_bot = float(row.get("cloud_bot") or -np.inf)
        if close < cloud_bot:
            return "LADDER_CLOUD_LOSS_NEXT_OPEN", None
        if pnl >= cfg.trailing_activation and close < kijun:
            return "LADDER_KIJUN_PROFIT_NEXT_OPEN", None
        if pnl < 0.0 and close < tenkan and bool(row.get("tk_death", False)):
            return "LADDER_TK_FAILED_NEXT_OPEN", None
    if cfg.exit == "kijun_trail":
        pnl = close / pos.entry_price - 1.0
        if pnl >= cfg.trailing_activation and close < float(row.get("kijun") or -np.inf):
            return "KIJUN_TRAIL_NEXT_OPEN", None
        if close < float(row.get("cloud_bot") or -np.inf):
            return "KIJUN_TRAIL_CLOUD_NEXT_OPEN", None
    return None, None


def run_portfolio(data: dict[str, pd.DataFrame], index: pd.DataFrame, cfg: IchConfig, args) -> dict[str, Any]:
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
                stop_price = max(entry_price * (1.0 - cfg.stop_loss), entry_price - 2.0 * atr)
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
            row = row_maps[symbol].loc[date]
            reason, price = exit_signal(row, pos, day_idx - pos.entry_idx, cfg)
            if reason is None:
                continue
            if price is None and next_date is not None:
                pending_exits.setdefault(next_date, []).append((symbol, reason))
                continue
            if price is not None:
                cash += pos.shares * price * (1.0 - args.commission_rate)
                trades.append(_trade(pos, date, price, reason))
                del positions[symbol]

        equity = cash + sum(
            pos.shares * _close_at(row_maps, pos.symbol, date, pos.entry_price)
            for pos in positions.values()
        )
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
                    + 0.3 * float(row.get("volume_ratio_20") or 0.0)
                    - 0.2 * max(float(row.get("kijun_distance") or 0.0), 0.0)
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
    pnl_pct = exit_price / pos.entry_price - 1.0
    return {
        "symbol": pos.symbol,
        "entry_date": pos.entry_date,
        "exit_date": exit_date,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "shares": pos.shares,
        "pnl": pos.shares * (exit_price - pos.entry_price),
        "pnl_pct": pnl_pct,
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


def generate_configs() -> list[IchConfig]:
    params = [
        ("fast", 7, 22, 44, 22),
        ("classic", 9, 26, 52, 26),
        ("slow", 10, 30, 60, 30),
    ]
    entries = ["perfect_bull", "tk_cross_cloud", "cloud_break", "kijun_bounce", "turtle20_ich", "turtle55_ich"]
    exits = ["tk_death", "kijun_close", "cloud_close", "atr_trail"]
    configs = []
    for label, tenkan, kijun, senkou_b, shift in params:
        for entry in entries:
            for exit_name in exits:
                for vni in ["uptrend", "healthy"]:
                    for rs_min in [0.50, 0.65]:
                        name = f"ich_vn_{label}_{entry}_{exit_name}_{vni}_rs{int(rs_min*100)}"
                        configs.append(IchConfig(
                            name=name,
                            tenkan=tenkan,
                            kijun=kijun,
                            senkou_b=senkou_b,
                            shift=shift,
                            entry=entry,
                            exit=exit_name,
                            vni_filter=vni,
                            rs_min=rs_min,
                            take_profit=None if exit_name in {"tk_death", "kijun_close", "cloud_close"} else 0.35,
                            max_holding_bars=90 if entry.startswith("turtle") else 60,
                        ))
    return configs


def generate_focused_configs(regimes: list[str] | None = None) -> list[IchConfig]:
    regimes = regimes or ["uptrend", "bull"]
    params = [
        ("fast", 7, 22, 44, 22),
        ("classic", 9, 26, 52, 26),
    ]
    entries = [
        "perfect_bull",
        "cloud_break",
        "chikou_high_break",
        "kijun_bounce",
        "qmv_retest",
        "kumo_reclaim",
        "turtle20_ich",
        "turtle55_ich",
    ]
    exits = ["kijun_close", "cloud_close", "atr_trail", "ich_ladder", "kijun_trail"]
    configs = []
    for label, tenkan, kijun, senkou_b, shift in params:
        for entry in entries:
            for exit_name in exits:
                for regime in regimes:
                    for rs_min in [0.50, 0.65, 0.75]:
                        for cloud_width_min, cloud_width_max, suffix in [
                            (0.0, 0.18, ""),
                            (0.005, 0.12, "_kumoq"),
                        ]:
                            name = f"ich_vn_{label}_{entry}_{exit_name}_{regime}_rs{int(rs_min*100)}{suffix}"
                            configs.append(IchConfig(
                                name=name,
                                tenkan=tenkan,
                                kijun=kijun,
                                senkou_b=senkou_b,
                                shift=shift,
                                entry=entry,
                                exit=exit_name,
                                vni_filter=regime,
                                rs_min=rs_min,
                                cloud_width_min=cloud_width_min,
                                cloud_width_max=cloud_width_max,
                                take_profit=None if exit_name in {"kijun_close", "cloud_close", "ich_ladder", "kijun_trail"} else 0.35,
                                max_holding_bars=90 if entry.startswith("turtle") else 60,
                                trailing_activation=0.08 if exit_name in {"ich_ladder", "kijun_trail"} else 0.10,
                            ))
    return configs


def generate_qmv_like_configs(regimes: list[str] | None = None) -> list[IchConfig]:
    regimes = regimes or ["bull"]
    params = [
        ("fast", 7, 22, 44, 22),
        ("classic", 9, 26, 52, 26),
    ]
    entries = ["cloud_break", "chikou_high_break", "qmv_retest", "kumo_reclaim", "turtle55_ich"]
    exits = ["cloud_close", "ich_ladder", "kijun_trail"]
    configs = []
    for label, tenkan, kijun, senkou_b, shift in params:
        for entry in entries:
            for exit_name in exits:
                for regime in regimes:
                    for rs_min in [0.50, 0.65, 0.75]:
                        for cloud_width_min, cloud_width_max, suffix in [
                            (0.0, 0.18, ""),
                            (0.005, 0.12, "_kumoq"),
                        ]:
                            for market_dd_120_max, dd_suffix in [(0.16, ""), (0.10, "_dd10"), (0.08, "_dd8")]:
                                configs.append(IchConfig(
                                    name=f"ich_qmv_{label}_{entry}_{exit_name}_{regime}_rs{int(rs_min*100)}{suffix}{dd_suffix}",
                                    tenkan=tenkan,
                                    kijun=kijun,
                                    senkou_b=senkou_b,
                                    shift=shift,
                                    entry=entry,
                                    exit=exit_name,
                                    vni_filter=regime,
                                    rs_min=rs_min,
                                    cloud_width_min=cloud_width_min,
                                    cloud_width_max=cloud_width_max,
                                    market_dd_120_max=market_dd_120_max,
                                    take_profit=None,
                                    max_holding_bars=100 if entry.startswith("turtle") else 70,
                                    trailing_activation=0.08,
                                ))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="vn100")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-05-16")
    parser.add_argument("--label", default="ichimoku_vn_search")
    parser.add_argument("--initial-capital", type=float, default=100_000_000)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    parser.add_argument("--commission-rate", type=float, default=0.001)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--focused", action="store_true")
    parser.add_argument("--qmv-like", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--regimes", default="uptrend,bull")
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"{args.label}_{args.universe}_{args.start}_{args.end}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = resolve_symbols(args.universe)
    data, index = load_data(symbols, args.start, args.end)
    rows = []
    best_payloads = []
    regimes = [item.strip() for item in args.regimes.split(",") if item.strip()]
    if args.qmv_like:
        configs = generate_qmv_like_configs(regimes=regimes)
    elif args.focused:
        configs = generate_focused_configs(regimes=regimes)
    else:
        configs = generate_configs()
    if args.limit > 0:
        configs = configs[: args.limit]
    print(f"[ich-vn] configs={len(configs)} symbols={len(data)} period={args.start}->{args.end}")
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
            print(f"[ich-vn] {i}/{len(configs)}")
        if len(result["trades"]) >= 20:
            best_payloads.append((row["total_return_pct"], cfg.name, result))

    summary = pd.DataFrame(rows).sort_values(
        ["total_return_pct", "sharpe_ratio"], ascending=[False, False]
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    for _, name, payload in sorted(best_payloads, reverse=True)[: args.top]:
        pd.DataFrame(payload["trades"]).to_csv(out_dir / f"{name}_trades.csv", index=False)
        if not payload["equity"].empty:
            payload["equity"].to_csv(out_dir / f"{name}_equity.csv")
    print(summary.head(args.top).to_string(index=False))
    print(f"[ich-vn] saved {out_dir}")


if __name__ == "__main__":
    main()
