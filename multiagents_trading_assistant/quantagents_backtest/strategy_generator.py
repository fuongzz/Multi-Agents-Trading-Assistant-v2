"""Strategy pool generation and signal evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
import operator
import random
from typing import Literal

import pandas as pd


Logic = Literal["AND", "OR"]
ConditionKind = Literal["threshold", "column_compare", "crossover"]

_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


@dataclass(frozen=True)
class Condition:
    left: str
    op: str
    right: str | float | bool
    kind: ConditionKind = "threshold"

    def label(self) -> str:
        if self.kind == "crossover":
            direction = "crosses_above" if self.op == ">" else "crosses_below"
            return f"{self.left} {direction} {self.right}"
        return f"{self.left} {self.op} {self.right}"


@dataclass(frozen=True)
class RiskRule:
    stop_loss: float
    take_profit: float
    max_holding_bars: int = 60


@dataclass(frozen=True)
class Strategy:
    strategy_id: str
    conditions: tuple[Condition, ...]
    logic: Logic
    risk: RiskRule
    family: str

    def describe(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "family": self.family,
            "logic": self.logic,
            "conditions": [condition.label() for condition in self.conditions],
            "risk": {
                "stop_loss": self.risk.stop_loss,
                "take_profit": self.risk.take_profit,
                "max_holding_bars": self.risk.max_holding_bars,
            },
        }


def generate_strategy_pool(n_strategies: int = 120, seed: int = 42) -> list[Strategy]:
    """Generate a reproducible, diverse pool of 50-200 strategies."""

    if not 50 <= n_strategies <= 200:
        raise ValueError("n_strategies must be between 50 and 200")

    rng = random.Random(seed)
    templates = _condition_templates()
    risks = [
        RiskRule(0.03, 0.08, 30),
        RiskRule(0.05, 0.12, 45),
        RiskRule(0.05, 0.15, 60),
        RiskRule(0.08, 0.18, 75),
        RiskRule(0.08, 0.25, 90),
    ]

    strategies: list[Strategy] = []
    seen: set[tuple] = set()
    ids = count(1)

    families = list(templates)
    while len(strategies) < n_strategies:
        family = families[len(strategies) % len(families)]
        candidates = templates[family]
        size = rng.choice([2, 2, 3, 3, 4])
        conditions = tuple(rng.sample(candidates, k=min(size, len(candidates))))
        logic: Logic = rng.choice(["AND", "AND", "OR"])
        risk = rng.choice(risks)
        key = (
            tuple(sorted(condition.label() for condition in conditions)),
            logic,
            risk.stop_loss,
            risk.take_profit,
            risk.max_holding_bars,
        )
        if key in seen:
            continue
        seen.add(key)
        strategies.append(
            Strategy(
                strategy_id=f"qa_{next(ids):03d}",
                conditions=conditions,
                logic=logic,
                risk=risk,
                family=family,
            )
        )

    return strategies


def evaluate_strategy_signal(df: pd.DataFrame, strategy: Strategy) -> pd.Series:
    """Evaluate raw close-of-bar entry signals for a strategy."""

    condition_values = [_evaluate_condition(df, condition) for condition in strategy.conditions]
    if not condition_values:
        return pd.Series(False, index=df.index)

    signal = condition_values[0]
    for value in condition_values[1:]:
        if strategy.logic == "AND":
            signal = signal & value
        else:
            signal = signal | value
    return signal.fillna(False).astype(bool)


def _evaluate_condition(df: pd.DataFrame, condition: Condition) -> pd.Series:
    if condition.left not in df.columns:
        return pd.Series(False, index=df.index)

    left = df[condition.left]
    op = _OPS[condition.op]

    if condition.kind == "crossover":
        if not isinstance(condition.right, str) or condition.right not in df.columns:
            return pd.Series(False, index=df.index)
        right = df[condition.right]
        if condition.op == ">":
            return (left.shift(1) <= right.shift(1)) & (left > right)
        return (left.shift(1) >= right.shift(1)) & (left < right)

    right = df[condition.right] if isinstance(condition.right, str) else condition.right
    return op(left, right)


def _condition_templates() -> dict[str, list[Condition]]:
    return {
        "trend": [
            Condition("close", ">", "sma20", "column_compare"),
            Condition("close", ">", "sma50", "column_compare"),
            Condition("close", ">", "ema20", "column_compare"),
            Condition("close", ">", "ema50", "column_compare"),
            Condition("close", ">", "ema200", "column_compare"),
            Condition("ema20", ">", "ema50", "column_compare"),
            Condition("ema50", ">", "ema200", "column_compare"),
            Condition("adx_14", ">", 20),
            Condition("adx_14", ">", 25),
            Condition("dmp_14", ">", "dmn_14", "column_compare"),
            Condition("aroon_up_14", ">", 60),
            Condition("aroon_osc_14", ">", 0),
            Condition("supertrend_dir", ">", 0),
            Condition("linreg_slope_14", ">", 0),
            Condition("ema20", ">", "ema50", "crossover"),
            Condition("close", ">", "ema20", "crossover"),
        ],
        "momentum": [
            Condition("rsi_7", ">", 55),
            Condition("rsi_14", ">", 50),
            Condition("rsi_14", ">", 60),
            Condition("rsi_21", ">", 55),
            Condition("willr_14", ">", -50),
            Condition("willr_14", ">", -20),
            Condition("cmo_9", ">", 0),
            Condition("cmo_9", ">", 20),
            Condition("stoch_k", ">", "stoch_d", "column_compare"),
            Condition("stoch_k", ">", 50),
            Condition("roc_9", ">", 0),
            Condition("mom_10", ">", 0),
            Condition("macd", ">", "macd_signal", "column_compare"),
            Condition("macd_hist", ">", 0),
            Condition("macd", ">", "macd_signal", "crossover"),
            Condition("stoch_k", ">", "stoch_d", "crossover"),
        ],
        "volume": [
            Condition("volume", ">", "volume_ma20", "column_compare"),
            Condition("volume_ratio_20", ">", 1.2),
            Condition("volume_ratio_20", ">", 1.5),
            Condition("volume_spike", ">=", True),
            Condition("obv", ">", "obv_ema20", "column_compare"),
            Condition("close", ">", "vwma20", "column_compare"),
            Condition("close", ">", "ema20", "column_compare"),
            Condition("breakout_20_high", ">=", True),
        ],
        "volatility": [
            Condition("close", ">", "bb_mid", "column_compare"),
            Condition("close", ">", "bb_upper", "column_compare"),
            Condition("close", "<=", "bb_lower", "column_compare"),
            Condition("bb_percent", ">", 0.5),
            Condition("bb_percent", "<", 0.2),
            Condition("bb_width", "<", 0.12),
            Condition("atr_pct", "<", 0.05),
            Condition("stdev_14", "<", "atr_14", "column_compare"),
            Condition("close", ">", "kc_mid", "column_compare"),
            Condition("close", ">", "kc_upper", "column_compare"),
            Condition("kc_width", "<", 0.12),
            Condition("volume_ratio_20", ">", 1.2),
        ],
        "structure": [
            Condition("breakout_20_high", ">=", True),
            Condition("breakdown_20_low", "<=", False),
            Condition("close", ">", "high_20_prev", "column_compare"),
            Condition("close", ">", "ema50", "column_compare"),
            Condition("close", ">", "supertrend_10_3", "column_compare"),
            Condition("aroon_up_14", ">", "aroon_down_14", "column_compare"),
            Condition("dmp_14", ">", "dmn_14", "column_compare"),
            Condition("volume_ratio_20", ">", 1.5),
            Condition("rsi_14", ">", 55),
        ],
        "hybrid": [
            Condition("close", ">", "ema50", "column_compare"),
            Condition("ema20", ">", "ema50", "column_compare"),
            Condition("rsi_14", ">", 50),
            Condition("macd_hist", ">", 0),
            Condition("volume_ratio_20", ">", 1.2),
            Condition("atr_pct", "<", 0.06),
            Condition("adx_14", ">", 20),
            Condition("supertrend_dir", ">", 0),
            Condition("obv", ">", "obv_ema20", "column_compare"),
            Condition("breakout_20_high", ">=", True),
        ],
        "mean_reversion": [
            Condition("rsi_14", "<", 35),
            Condition("willr_14", "<", -80),
            Condition("stoch_k", "<", 25),
            Condition("close", "<=", "bb_lower", "column_compare"),
            Condition("bb_percent", "<", 0.2),
            Condition("adx_14", "<", 25),
            Condition("volume_ratio_20", ">", 1.0),
            Condition("close", ">", "kc_lower", "column_compare"),
        ],
        "qa_broad": [
            Condition("roc_9", ">", 0),
            Condition("mom_10", ">", 0),
            Condition("cmo_9", ">", 0),
            Condition("dmp_14", ">", "dmn_14", "column_compare"),
            Condition("aroon_osc_14", ">", 0),
            Condition("linreg_slope_14", ">", 0),
            Condition("close", ">", "vwma20", "column_compare"),
            Condition("close", ">", "supertrend_10_3", "column_compare"),
        ],
    }
