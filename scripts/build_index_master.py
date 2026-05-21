"""Build a standardized market index parquet from local VNINDEX data.

The backtest code historically read ``VNINDEX_*_1D_hist.json`` directly from
cache. This script materializes that data as a stable parquet artifact so
feature generation and data-quality checks have one canonical source. It also
fetches recent VNINDEX bars so the parquet does not stop at the last cache file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CACHE_DIR = ROOT / "multiagents_trading_assistant" / "cache"
DEFAULT_OUT = ROOT / "multiagents_trading_assistant" / "data" / "index_master.parquet"


def load_index_cache(cache_dir: Path, symbol: str = "VNINDEX") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(cache_dir.glob(f"{symbol}_*_1D_hist.json")):
        try:
            frame = pd.read_json(path)
        except Exception as exc:
            print(f"skip {path.name}: {exc}")
            continue
        if frame.empty or "date" not in frame.columns:
            continue
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
        frame["symbol"] = symbol
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in frame.columns:
                frame[col] = pd.NA
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frames.append(frame[["date", "symbol", "open", "high", "low", "close", "volume"]])

    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date", "close"])
    out = out.drop_duplicates(["date", "symbol"], keep="last")
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out["volume"] = out["volume"].fillna(0).astype("int64")
    return out


def fetch_recent_index(symbol: str = "VNINDEX", n_days: int = 30) -> pd.DataFrame:
    """Fetch recent index bars from the project's data provider."""
    if symbol.upper() != "VNINDEX":
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    try:
        from multiagents_trading_assistant.fetcher import get_vnindex

        frame = get_vnindex(n_days=n_days)
    except Exception as exc:
        print(f"recent {symbol.upper()} fetch failed: {exc}")
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])

    if frame is None or frame.empty or "date" not in frame.columns:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    frame["symbol"] = symbol.upper()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in frame.columns:
            frame[col] = pd.NA
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    frame["volume"] = frame["volume"].fillna(0).astype("int64")
    return frame[["date", "symbol", "open", "high", "low", "close", "volume"]]


def merge_index_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=["date", "close"])
    out = out.drop_duplicates(["date", "symbol"], keep="last")
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype("int64")
    return out[["date", "symbol", "open", "high", "low", "close", "volume"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build index_master.parquet from cache and recent provider data")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--symbol", default="VNINDEX")
    parser.add_argument("--recent-days", type=int, default=30)
    parser.add_argument("--no-fetch-latest", action="store_true")
    args = parser.parse_args()

    cached = load_index_cache(args.cache_dir, args.symbol.upper())
    recent = (
        pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume"])
        if args.no_fetch_latest
        else fetch_recent_index(args.symbol.upper(), args.recent_days)
    )
    frame = merge_index_frames([cached, recent])
    if frame.empty:
        raise SystemExit(f"No {args.symbol.upper()} cache data found in {args.cache_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f"{args.out.stem}.tmp{args.out.suffix}")
    tmp.unlink(missing_ok=True)
    frame.to_parquet(tmp, index=False)
    tmp.replace(args.out)
    print(f"Wrote {len(frame)} rows to {args.out}")
    print(f"Date range: {frame['date'].min().date()} -> {frame['date'].max().date()}")


if __name__ == "__main__":
    main()
