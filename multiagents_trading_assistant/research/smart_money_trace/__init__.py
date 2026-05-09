"""Smart Money Trace Research — định lượng dấu vết dòng tiền thông minh."""

from multiagents_trading_assistant.research.smart_money_trace.compute_smart_money_trace import (
    run_smart_money_trace,
    load_ohlcv,
    normalize_ohlcv_columns,
    compute_basic_features,
    compute_clv,
    compute_atr,
    compute_relative_strength,
    compute_accumulation_distribution,
    compute_pullback_quality,
    compute_value_flow_quality,
    compute_sector_leadership,
    compute_baseline_indicators,
    compute_smart_money_scores,
    classify_smart_money_state,
    summarize_daily,
    save_outputs,
)

__all__ = [
    "run_smart_money_trace",
    "load_ohlcv",
    "normalize_ohlcv_columns",
    "compute_basic_features",
    "compute_clv",
    "compute_atr",
    "compute_relative_strength",
    "compute_accumulation_distribution",
    "compute_pullback_quality",
    "compute_value_flow_quality",
    "compute_sector_leadership",
    "compute_baseline_indicators",
    "compute_smart_money_scores",
    "classify_smart_money_state",
    "summarize_daily",
    "save_outputs",
    "plot_market_dashboard_interactive",
    "plot_symbol_dashboard_interactive",
    "run_interactive_charts",
]


def __getattr__(name: str):
    if name in {
        "plot_market_dashboard_interactive",
        "plot_symbol_dashboard_interactive",
        "run_interactive_charts",
    }:
        from multiagents_trading_assistant.research.smart_money_trace import visualization_interactive

        return getattr(visualization_interactive, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
