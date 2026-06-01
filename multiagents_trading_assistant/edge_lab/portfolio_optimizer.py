"""Portfolio allocation helpers for edge-lab candidate batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


AllocationMethod = Literal[
    "equal_weight",
    "rank_weight",
    "inverse_volatility",
    "minimum_variance",
    "max_sharpe",
]


@dataclass(frozen=True)
class AllocationConfig:
    method: AllocationMethod = "equal_weight"
    lookback_bars: int = 60
    min_weight: float = 0.05
    max_weight: float = 0.35
    rank_blend: float = 0.25
    shrinkage: float = 0.20
    risk_free_daily: float = 0.0


def allocate_candidate_budgets(
    candidates: pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    entry_date: pd.Timestamp,
    *,
    deployable: float,
    config: AllocationConfig,
) -> pd.DataFrame:
    """Attach target weights and budgets to an entry batch.

    The optimizer only uses price history strictly before ``entry_date`` so it
    can be used inside walk-forward backtests without leaking entry-day data.
    """
    result = candidates.copy()
    if result.empty:
        result["_target_weight"] = []
        result["_budget"] = []
        return result

    if deployable <= 0:
        result["_target_weight"] = 0.0
        result["_budget"] = 0.0
        return result

    returns = _candidate_returns(
        result["symbol"].astype(str).tolist(),
        price_map,
        entry_date,
        config.lookback_bars,
    )
    weights = candidate_weights(result, returns, config)
    result["_target_weight"] = weights
    result["_budget"] = result["_target_weight"] * float(deployable)
    return result


def candidate_weights(
    candidates: pd.DataFrame,
    returns: pd.DataFrame,
    config: AllocationConfig,
) -> np.ndarray:
    if candidates.empty:
        return np.array([], dtype=float)

    n = len(candidates)
    if config.method == "equal_weight" or returns.empty:
        raw = np.ones(n, dtype=float) / n
    elif config.method == "rank_weight":
        raw = _rank_weights(candidates)
    else:
        aligned_returns = returns.reindex(columns=candidates["symbol"].astype(str).tolist())
        if aligned_returns.dropna(how="all").empty:
            raw = np.ones(n, dtype=float) / n
        elif config.method == "inverse_volatility":
            raw = _inverse_volatility_weights(aligned_returns)
        elif config.method == "minimum_variance":
            raw = _minimum_variance_weights(aligned_returns, config.shrinkage)
        elif config.method == "max_sharpe":
            raw = _max_sharpe_weights(
                aligned_returns,
                config.shrinkage,
                config.risk_free_daily,
            )
        else:
            raw = np.ones(n, dtype=float) / n

    if config.method not in {"equal_weight", "rank_weight"} and config.rank_blend > 0:
        rank = _rank_weights(candidates)
        blend = float(np.clip(config.rank_blend, 0.0, 1.0))
        raw = (1.0 - blend) * raw + blend * rank

    return _constrain_weights(raw, config.min_weight, config.max_weight)


def _candidate_returns(
    symbols: list[str],
    price_map: dict[str, pd.DataFrame],
    entry_date: pd.Timestamp,
    lookback_bars: int,
) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        frame = price_map.get(symbol)
        if frame is None or frame.empty:
            continue
        history = frame[pd.to_datetime(frame["date"]) < entry_date].tail(
            max(lookback_bars + 1, 2)
        )
        if len(history) < 2:
            continue
        close = pd.Series(
            history["close"].astype(float).values,
            index=pd.to_datetime(history["date"]),
            name=symbol,
        )
        frames.append(close.pct_change().rename(symbol))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna(how="all")


def _rank_weights(candidates: pd.DataFrame) -> np.ndarray:
    rank = pd.to_numeric(
        candidates.get("_edge_rank", pd.Series(1.0, index=candidates.index)),
        errors="coerce",
    )
    score = (
        rank.fillna(rank.median() if rank.notna().any() else 1.0)
        .clip(lower=0.0)
        .to_numpy(dtype=float)
    )
    if score.sum() <= 0:
        return np.ones(len(candidates), dtype=float) / len(candidates)
    return score / score.sum()


def _inverse_volatility_weights(returns: pd.DataFrame) -> np.ndarray:
    vol = returns.std(skipna=True).replace(0.0, np.nan)
    inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    if inv.sum() <= 0:
        return np.ones(len(returns.columns), dtype=float) / len(returns.columns)
    return inv / inv.sum()


def _minimum_variance_weights(returns: pd.DataFrame, shrinkage: float) -> np.ndarray:
    cov = _shrunk_covariance(returns, shrinkage)
    ones = np.ones(cov.shape[0], dtype=float)
    try:
        inv_ones = np.linalg.pinv(cov) @ ones
    except np.linalg.LinAlgError:
        return _inverse_volatility_weights(returns)
    return _positive_normalize(inv_ones)


def _max_sharpe_weights(
    returns: pd.DataFrame,
    shrinkage: float,
    risk_free_daily: float,
) -> np.ndarray:
    cov = _shrunk_covariance(returns, shrinkage)
    mu = returns.mean(skipna=True).fillna(0.0).to_numpy(dtype=float) - float(risk_free_daily)
    if np.nanmax(mu) <= 0:
        return _minimum_variance_weights(returns, shrinkage)
    try:
        raw = np.linalg.pinv(cov) @ mu
    except np.linalg.LinAlgError:
        raw = mu
    return _positive_normalize(raw)


def _shrunk_covariance(returns: pd.DataFrame, shrinkage: float) -> np.ndarray:
    clean = returns.fillna(0.0)
    cov = clean.cov().to_numpy(dtype=float)
    if cov.size == 0:
        return np.eye(len(returns.columns), dtype=float)
    diag = np.diag(np.diag(cov))
    lam = float(np.clip(shrinkage, 0.0, 1.0))
    shrunk = (1.0 - lam) * cov + lam * diag
    eps = max(float(np.nanmean(np.diag(shrunk))) * 1e-6, 1e-10)
    return shrunk + np.eye(shrunk.shape[0]) * eps


def _positive_normalize(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    raw = np.clip(raw, 0.0, None)
    if raw.sum() <= 0:
        return np.ones(len(raw), dtype=float) / len(raw)
    return raw / raw.sum()


def _constrain_weights(weights: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    raw = _positive_normalize(weights)
    n = len(raw)
    if n == 0:
        return raw
    lo = max(0.0, float(min_weight))
    hi = min(1.0, max(float(max_weight), 1.0 / n))
    if lo * n > 1.0:
        lo = 0.0

    constrained = np.clip(raw, lo, hi)
    for _ in range(20):
        diff = 1.0 - constrained.sum()
        if abs(diff) < 1e-10:
            break
        if diff > 0:
            room = hi - constrained
            mask = room > 1e-12
            if not mask.any():
                break
            constrained[mask] += diff * room[mask] / room[mask].sum()
        else:
            room = constrained - lo
            mask = room > 1e-12
            if not mask.any():
                break
            constrained[mask] += diff * room[mask] / room[mask].sum()
    return _positive_normalize(constrained)
