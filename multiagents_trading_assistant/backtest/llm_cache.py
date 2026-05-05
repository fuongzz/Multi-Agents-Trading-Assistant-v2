"""LLM Decision Cache — tránh gọi lại LLM khi chạy backtest nhiều lần.

Cache key bao gồm: symbol + signal_date + graph_mode + model + prompt_version
                  + data_hash + indicator_version

data_hash: hash OHLCV cuối của symbol tại as_of_date (thay đổi → invalidate).
indicator_version: const trong module này — tăng thủ công khi logic indicator thay đổi.

Storage: multiagents_trading_assistant/cache/llm_decisions/{key[:16]}.json
Decision log: backtest_results/llm_decisions_{run_id}.jsonl (dùng để replay)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Tăng khi thay đổi indicator logic hoặc prompt system
INDICATOR_VERSION = "v1.0"
PROMPT_VERSION = "v1.0"

_CACHE_DIR = Path(__file__).parent.parent / "cache" / "llm_decisions"
_RESULTS_DIR = Path(__file__).parent.parent.parent / "backtest_results"


def _ensure_dirs() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_data_hash(df: pd.DataFrame, as_of_date: str) -> str:
    """Hash OHLCV dataframe filtered tại as_of_date.

    Dùng để detect khi data thay đổi → invalidate cache.
    """
    if df is None or df.empty:
        return "empty"
    # Chỉ hash 3 cột quan trọng + shape
    try:
        subset = df[["close", "volume", "high"]].tail(10)
        raw = subset.to_csv(index=False) + as_of_date
    except Exception:
        raw = str(df.shape) + as_of_date
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def build_cache_key(
    symbol: str,
    signal_date: str,
    graph_mode: str,
    model: str,
    data_hash: str,
    prompt_version: str = PROMPT_VERSION,
    indicator_version: str = INDICATOR_VERSION,
) -> str:
    raw = (
        f"{symbol}__{signal_date}__{graph_mode}__{model}__"
        f"{prompt_version}__{indicator_version}__{data_hash}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(key: str) -> dict | None:
    """Đọc cache. Trả None nếu không có."""
    _ensure_dirs()
    path = _CACHE_DIR / f"{key[:24]}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def set_cache(key: str, data: dict) -> None:
    """Ghi cache."""
    _ensure_dirs()
    path = _CACHE_DIR / f"{key[:24]}.json"
    payload = {**data, "_cached_at": datetime.now().isoformat(), "_key": key}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def log_decision(
    run_id: str,
    symbol: str,
    signal_date: str,
    plan_dict: dict | None,
    score: float,
    accepted: bool,
    cache_hit: bool,
) -> None:
    """Append quyết định vào JSONL log. Dùng để replay không cần LLM."""
    _ensure_dirs()
    path = _RESULTS_DIR / f"llm_decisions_{run_id}.jsonl"
    entry: dict[str, Any] = {
        "ts": datetime.now().isoformat(),
        "symbol": symbol,
        "signal_date": signal_date,
        "plan": plan_dict,
        "score": score,
        "accepted": accepted,
        "cache_hit": cache_hit,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
