"""Local parquet loaders for QuantAgents MVP research runs.

These helpers intentionally stay in the deterministic research layer. They load
locally cached OHLCV/index data, apply simple liquidity-history filters, and
return symbol -> OHLCV frames ready for the QuantAgents-style backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from multiagents_trading_assistant.quantagents_backtest.vn_universe import (
    UniverseSnapshot,
    get_universe_snapshot,
    load_historical_constituents,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OHLCV_PATH = ROOT / "multiagents_trading_assistant" / "data" / "ohlcv_master.parquet"
DEFAULT_INDEX_PATH = ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"


@dataclass(frozen=True)
class LocalUniverseConfig:
    start: str = "2021-01-01"
    end: str = "2026-05-03"
    universe_name: str = "VN30"
    as_of: str | None = None
    min_history_bars: int = 252
    min_median_daily_value: float = 50_000_000.0
    min_last_close: float = 5.0
    allowed_exchanges: tuple[str, ...] = ("HOSE", "HNX")


def load_local_universe(
    config: LocalUniverseConfig | None = None,
    *,
    ohlcv_path: str | Path = DEFAULT_OHLCV_PATH,
    constituents_path: str | Path | None = None,
    explicit_symbols: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, UniverseSnapshot]:
    """Load local OHLCV into per-symbol backtest frames.

    Returns:
        ``(universe_data, coverage_df, snapshot)``
    """

    cfg = config or LocalUniverseConfig()
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    as_of = pd.Timestamp(cfg.as_of or cfg.end)
    raw = _read_parquet(ohlcv_path)

    if explicit_symbols:
        snapshot = UniverseSnapshot(
            date=as_of,
            symbols=sorted({symbol.upper().strip() for symbol in explicit_symbols}),
            source="explicit_symbols",
            is_historical=False,
        )
    else:
        constituents = None
        if constituents_path is not None:
            constituents = load_historical_constituents(constituents_path)
        try:
            snapshot = get_universe_snapshot(as_of, index_name=cfg.universe_name, constituents=constituents)
        except ValueError:
            if cfg.universe_name.upper() != "VN100":
                raise
            snapshot = _build_local_vn100_proxy(raw, cfg, start, end, as_of)

    symbols = {symbol.upper().strip() for symbol in snapshot.symbols}
    frame = raw[raw["symbol"].astype(str).str.upper().isin(symbols)].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    if "exchange" in frame.columns:
        frame["exchange"] = frame["exchange"].astype(str).str.upper()
    else:
        frame["exchange"] = ""
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    if cfg.allowed_exchanges:
        frame = frame[frame["exchange"].isin(cfg.allowed_exchanges)]

    required = ["date", "symbol", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"OHLCV parquet is missing columns: {missing}")

    coverage_rows: list[dict] = []
    universe_data: dict[str, pd.DataFrame] = {}

    for symbol in sorted(symbols):
        symbol_frame = frame[frame["symbol"] == symbol].copy()
        if symbol_frame.empty:
            coverage_rows.append(
                {
                    "symbol": symbol,
                    "included": False,
                    "reason": "missing_symbol_data",
                    "bars": 0,
                    "median_daily_value": 0.0,
                    "last_close": 0.0,
                    "exchange": "",
                    "industry": "",
                }
            )
            continue

        symbol_frame = symbol_frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        median_daily_value = float(symbol_frame.get("value", pd.Series(dtype=float)).median())
        last_close = float(symbol_frame["close"].iloc[-1])
        exchange = str(symbol_frame.get("exchange", pd.Series([""])).iloc[-1])
        industry = str(symbol_frame.get("industry", pd.Series([""])).iloc[-1])

        reason = "ok"
        included = True
        if len(symbol_frame) < cfg.min_history_bars:
            included = False
            reason = "insufficient_history"
        elif pd.notna(median_daily_value) and median_daily_value < cfg.min_median_daily_value:
            included = False
            reason = "insufficient_liquidity"
        elif last_close < cfg.min_last_close:
            included = False
            reason = "price_floor"

        coverage_rows.append(
            {
                "symbol": symbol,
                "included": included,
                "reason": reason,
                "bars": int(len(symbol_frame)),
                "median_daily_value": median_daily_value if pd.notna(median_daily_value) else 0.0,
                "last_close": last_close,
                "exchange": exchange,
                "industry": industry,
            }
        )

        if not included:
            continue

        prepared = symbol_frame[["date", "open", "high", "low", "close", "volume"]].copy()
        prepared = prepared.set_index("date").sort_index()
        prepared.index = pd.DatetimeIndex(prepared.index)
        universe_data[symbol] = prepared

    coverage = pd.DataFrame(coverage_rows).sort_values(["included", "symbol"], ascending=[False, True]).reset_index(drop=True)
    return universe_data, coverage, snapshot


def load_local_index(
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    symbol: str = "VNINDEX",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load local market index history into a backtest-ready frame."""

    frame = _read_parquet(index_path).copy()
    if "symbol" not in frame.columns:
        raise ValueError("Index parquet is missing `symbol` column")
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["symbol"] == symbol.upper()].copy()
    if start is not None:
        frame = frame[frame["date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["date"] <= pd.Timestamp(end)]
    if frame.empty:
        raise ValueError(f"No index data found for {symbol}")
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Index parquet is missing columns: {missing}")
    frame = frame[required].sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return frame.set_index("date")


def _read_parquet(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Parquet file not found: {resolved}")
    return pd.read_parquet(resolved)


def _build_local_vn100_proxy(
    raw: pd.DataFrame,
    cfg: LocalUniverseConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    as_of: pd.Timestamp,
) -> UniverseSnapshot:
    """Build a test-only VN100 proxy from local liquidity when constituents are unavailable."""

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["symbol"] = frame["symbol"].astype(str).str.upper().str.strip()
    if "exchange" in frame.columns:
        frame["exchange"] = frame["exchange"].astype(str).str.upper()
    else:
        frame["exchange"] = ""
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    if cfg.allowed_exchanges:
        frame = frame[frame["exchange"].isin(cfg.allowed_exchanges)]
    if frame.empty:
        raise ValueError("Cannot build VN100 proxy because local OHLCV data is empty")

    if "value" not in frame.columns:
        frame["value"] = frame["close"].astype(float) * frame["volume"].astype(float)

    grouped = (
        frame.groupby("symbol", as_index=False)
        .agg(
            bars=("date", "nunique"),
            median_daily_value=("value", "median"),
            last_date=("date", "max"),
            last_close=("close", "last"),
        )
        .sort_values(["median_daily_value", "bars"], ascending=[False, False])
    )
    eligible = grouped[
        (grouped["bars"] >= cfg.min_history_bars)
        & (grouped["median_daily_value"] >= cfg.min_median_daily_value)
        & (grouped["last_close"] >= cfg.min_last_close)
    ].copy()
    symbols = eligible.head(100)["symbol"].tolist()
    if not symbols:
        raise ValueError("Cannot build VN100 proxy because no symbols pass local history/liquidity filters")
    return UniverseSnapshot(
        date=as_of,
        symbols=sorted(symbols),
        source="local_vn100_liquidity_proxy",
        is_historical=False,
    )
