"""Fish public trading strategies, tune them for Vietnam, and backtest on VN100.

The script temporarily injects public-rule inspired hypotheses into the
edge-lab config, runs the existing live-pipeline sizing-grid backtester, then
restores the original config.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "multiagents_trading_assistant/edge_lab/configs/vn30_money_smt_hypotheses.json"
CFG_BACKUP = CFG_PATH.with_suffix(".public_fishing.bak")
OUT_DIR = ROOT / "backtest_results/public_strategy_fishing"
REPORTS_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


PUBLIC_STRATEGIES = [
    {
        "source_family": "Minervini Trend Template",
        "public_rule": "Price above key MAs, MA stack rising, strong relative strength.",
        "source_url": "https://www.chartmill.com/documentation/stock-screener/technical-analysis-trading-strategies/496-Mark-Minervini-Trend-Template-A-Step-by-Step-Guide-for-Beginners",
        "hypothesis": {
            "name": "public_minervini_vn_tuned_v1",
            "description": "Public Minervini trend template tuned for VN: MA20/50/200 trend, RS leadership, no distribution cluster.",
            "universe": "VN100",
            "tags": ["public", "minervini", "trend_template", "leader"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "above_ma200", "op": "==", "value": True},
                {"column": "ma20_slope_5", "op": ">=", "value": 0.0},
                {"column": "ma50_slope_10", "op": ">=", "value": 0.002},
                {"column": "ma200_slope_20", "op": ">=", "value": -0.005},
                {"column": "distance_ma50", "op": "between", "value": [0.0, 0.28]},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.72},
                {"column": "excess_ret_60d_pctile", "op": ">=", "value": 0.62},
                {"column": "distribution_days_10", "op": "<=", "value": 2},
            ],
            "rank": [
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.4},
                {"column": "smart_money_score_no_sector", "ascending": False, "weight": 1.0},
                {"column": "ma50_slope_10", "ascending": False, "weight": 0.6},
            ],
            "risk": {"stop_loss": 0.08, "take_profit": 0.30, "max_holding_bars": 60, "initial_atr_stop_mult": 2.5, "trailing_atr_mult": 3.0, "trailing_profit_activation": 0.10, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "CANSLIM / O'Neil breakout",
        "public_rule": "Buy leading stocks breaking out of bases on strong demand/volume.",
        "source_url": "https://en.wikipedia.org/wiki/CAN_SLIM",
        "hypothesis": {
            "name": "public_canslim_breakout_vn_tuned_v1",
            "description": "CANSLIM technical proxy: 55-day breakout, high value/volume, market not risk-off, RS leadership.",
            "universe": "VN100",
            "tags": ["public", "canslim", "breakout", "volume", "leader"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "mkt_regime_score", "op": ">=", "value": 45},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "breakout_55", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "volume_ratio_20", "op": ">=", "value": 1.5},
                {"column": "value_ratio_20", "op": ">=", "value": 1.35},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.70},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 55},
                {"column": "distribution_days_10", "op": "<=", "value": 2},
            ],
            "rank": [
                {"column": "value_ratio_20", "ascending": False, "weight": 1.2},
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
                {"column": "smart_money_score_no_sector", "ascending": False, "weight": 1.0},
            ],
            "risk": {"stop_loss": 0.08, "take_profit": 0.30, "max_holding_bars": 45, "initial_atr_stop_mult": 2.4, "trailing_atr_mult": 2.8, "trailing_profit_activation": 0.08, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Turtle / Donchian 20 breakout",
        "public_rule": "Buy new 20-day highs; manage exits with trend/risk controls.",
        "source_url": "https://www.turtelli.com/articles/donchian-channels-for-traders",
        "hypothesis": {
            "name": "public_turtle20_vn_tuned_v1",
            "description": "Donchian/Turtle 20-day breakout tuned for VN with volume, MA50 and market-regime filters.",
            "universe": "VN100",
            "tags": ["public", "turtle", "donchian", "breakout"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "breakout_20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "ma50_slope_10", "op": ">=", "value": -0.005},
                {"column": "value_ratio_20", "op": ">=", "value": 1.15},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
                {"column": "distance_ma20", "op": "between", "value": [0.0, 0.15]},
            ],
            "rank": [
                {"column": "breakout_20_quality", "ascending": False, "weight": 1.2},
                {"column": "value_ratio_20", "ascending": False, "weight": 1.0},
                {"column": "rs_percentile_20", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.09, "take_profit": 0.28, "max_holding_bars": 35, "initial_atr_stop_mult": 2.4, "trailing_atr_mult": 2.8, "trailing_profit_activation": 0.09, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Turtle / Donchian 55 breakout",
        "public_rule": "Buy new 55-day highs for slower trend-following entries.",
        "source_url": "https://quantest.andgenie.jp/en/blog/donchian-channel-breakout/",
        "hypothesis": {
            "name": "public_turtle55_vn_tuned_v1",
            "description": "Slower Donchian 55-day breakout tuned for VN leaders.",
            "universe": "VN100",
            "tags": ["public", "turtle", "donchian", "trend_following"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "breakout_55", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "ma50_slope_10", "op": ">=", "value": 0.0},
                {"column": "value_ratio_20", "op": ">=", "value": 1.10},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.60},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 52},
            ],
            "rank": [
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.2},
                {"column": "value_ratio_20", "ascending": False, "weight": 1.0},
                {"column": "smart_money_score_no_sector", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.10, "take_profit": 0.35, "max_holding_bars": 60, "initial_atr_stop_mult": 2.8, "trailing_atr_mult": 3.2, "trailing_profit_activation": 0.12, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Moving-average crossover + ADX",
        "public_rule": "Fast MA above slow MA; add ADX/regime filters to reduce chop.",
        "source_url": "https://crosstrade.io/learn/trading-strategies/moving-average-crossover",
        "hypothesis": {
            "name": "public_ma_trend_adx_vn_tuned_v1",
            "description": "MA trend/crossover proxy tuned for VN: MA20 above MA50, ADX bullish, no chase.",
            "universe": "VN100",
            "tags": ["public", "moving_average", "trend", "adx"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "ma20_slope_5", "op": ">=", "value": 0.0},
                {"column": "ma50_slope_10", "op": ">=", "value": 0.0},
                {"column": "adx_14", "op": ">=", "value": 18},
                {"column": "adx_bullish", "op": "==", "value": True},
                {"column": "distance_ma20", "op": "between", "value": [-0.02, 0.08]},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
            ],
            "rank": [
                {"column": "adx_14", "ascending": False, "weight": 0.8},
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
                {"column": "smart_money_score_no_sector", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.075, "take_profit": 0.22, "max_holding_bars": 45, "initial_atr_stop_mult": 2.2, "trailing_atr_mult": 2.7, "trailing_profit_activation": 0.08, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Connors RSI(2)",
        "public_rule": "Buy very oversold pullbacks only when the long-term trend is up.",
        "source_url": "https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2",
        "hypothesis": {
            "name": "public_connors_rsi_vn_tuned_v1",
            "description": "Connors-style VN proxy using RSI14/pullback due to available feature set; short holding period.",
            "universe": "VN100",
            "tags": ["public", "connors", "mean_reversion", "oversold"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "ma50_slope_10", "op": ">=", "value": -0.005},
                {"column": "rsi14", "op": "<=", "value": 43},
                {"column": "distance_ma20", "op": "between", "value": [-0.12, 0.01]},
                {"column": "distribution_days_10", "op": "<=", "value": 3},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 45},
            ],
            "rank": [
                {"column": "rsi14", "ascending": True, "weight": 1.1},
                {"column": "pullback_quality_score", "ascending": False, "weight": 1.0},
                {"column": "rs_percentile_20", "ascending": False, "weight": 0.7},
            ],
            "risk": {"stop_loss": 0.06, "take_profit": 0.12, "max_holding_bars": 10, "initial_atr_stop_mult": 2.0, "trailing_atr_mult": 2.0, "trailing_profit_activation": 0.04, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Stan Weinstein Stage 2",
        "public_rule": "Price above rising 30-week/150-day MA, relative strength, breakout volume.",
        "source_url": "https://www.stage2stocks.com/learn",
        "hypothesis": {
            "name": "public_weinstein_stage2_vn_tuned_v1",
            "description": "Weinstein Stage 2 proxy: rising MA200/MA50, 20-day breakout, volume and RS confirmation.",
            "universe": "VN100",
            "tags": ["public", "weinstein", "stage2", "breakout"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "above_ma200", "op": "==", "value": True},
                {"column": "ma50_slope_10", "op": ">=", "value": 0.001},
                {"column": "ma200_slope_20", "op": ">=", "value": -0.003},
                {"column": "breakout_20", "op": "==", "value": True},
                {"column": "volume_ratio_20", "op": ">=", "value": 1.35},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.60},
                {"column": "distance_ma50", "op": "between", "value": [0.0, 0.25]},
            ],
            "rank": [
                {"column": "breakout_20_quality", "ascending": False, "weight": 1.3},
                {"column": "volume_ratio_20", "ascending": False, "weight": 0.8},
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
            ],
            "risk": {"stop_loss": 0.08, "take_profit": 0.30, "max_holding_bars": 50, "initial_atr_stop_mult": 2.5, "trailing_atr_mult": 3.0, "trailing_profit_activation": 0.10, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Ichimoku trend following",
        "public_rule": "Bullish cloud alignment plus trend confirmation.",
        "source_url": "https://www.investopedia.com/terms/i/ichimoku-cloud.asp",
        "hypothesis": {
            "name": "public_ichimoku_cloud_vn_tuned_v1",
            "description": "Ichimoku bullish alignment tuned for VN with no-chase and smart-money filters.",
            "universe": "VN100",
            "tags": ["public", "ichimoku", "trend"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "ich_strong_alignment", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 50},
                {"column": "distribution_days_10", "op": "<=", "value": 2},
            ],
            "rank": [
                {"column": "ich_cloud_width", "ascending": True, "weight": 0.5},
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
                {"column": "smart_money_score_no_sector", "ascending": False, "weight": 1.0},
            ],
            "risk": {"stop_loss": 0.08, "take_profit": 0.25, "max_holding_bars": 45, "initial_atr_stop_mult": 2.4, "trailing_atr_mult": 2.8, "trailing_profit_activation": 0.09, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Bollinger squeeze breakout",
        "public_rule": "Low volatility squeeze followed by upside breakout and volume confirmation.",
        "source_url": "https://www.investopedia.com/articles/trading/07/bollinger.asp",
        "hypothesis": {
            "name": "public_bollinger_squeeze_vn_tuned_v1",
            "description": "Bollinger/squeeze proxy: tight range, breakout quality, value expansion.",
            "universe": "VN100",
            "tags": ["public", "bollinger", "squeeze", "breakout"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "breakout_20_quality", "op": "==", "value": True},
                {"column": "tight_range_20", "op": "==", "value": True},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "value_ratio_20", "op": ">=", "value": 1.20},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
            ],
            "rank": [
                {"column": "value_ratio_20", "ascending": False, "weight": 1.1},
                {"column": "range_pct", "ascending": True, "weight": 0.7},
                {"column": "rs_percentile_20", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.075, "take_profit": 0.22, "max_holding_bars": 35, "initial_atr_stop_mult": 2.2, "trailing_atr_mult": 2.6, "trailing_profit_activation": 0.08, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Darvas Box",
        "public_rule": "Buy breakouts from a box/base when price makes new highs on demand.",
        "source_url": "https://www.investopedia.com/terms/d/darvasbox.asp",
        "hypothesis": {
            "name": "public_darvas_box_vn_tuned_v1",
            "description": "Darvas box proxy: inside-bar/base breakout, controlled distance from MA20, RS and volume confirmation.",
            "universe": "VN100",
            "tags": ["public", "darvas", "box", "breakout"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "inside_bar_breakout", "op": "==", "value": True},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "distance_ma20", "op": "between", "value": [0.0, 0.10]},
                {"column": "volume_ratio_20", "op": ">=", "value": 1.05},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
                {"column": "distribution_days_10", "op": "<=", "value": 2},
            ],
            "rank": [
                {"column": "volume_ratio_20", "ascending": False, "weight": 0.8},
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
                {"column": "smart_money_score_no_sector", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.075, "take_profit": 0.24, "max_holding_bars": 35, "initial_atr_stop_mult": 2.3, "trailing_atr_mult": 2.6, "trailing_profit_activation": 0.08, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Turtle / Donchian 20 breakout",
        "public_rule": "Second-pass VN tuning: keep the 20-day breakout, but require stronger market regime, value expansion, and lower distribution.",
        "source_url": "https://www.turtelli.com/articles/donchian-channels-for-traders",
        "hypothesis": {
            "name": "public_turtle20_defensive_vn_tuned_v2",
            "description": "Defensive Donchian 20 for VN: stricter regime, cleaner demand, tighter risk.",
            "universe": "VN100",
            "tags": ["public", "turtle", "donchian", "breakout", "defensive"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "mkt_regime_score", "op": ">=", "value": 55},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "breakout_20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "ma50_slope_10", "op": ">=", "value": 0.0},
                {"column": "value_ratio_20", "op": ">=", "value": 1.35},
                {"column": "volume_ratio_20", "op": ">=", "value": 1.20},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.62},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 58},
                {"column": "distribution_days_10", "op": "<=", "value": 1},
                {"column": "distance_ma20", "op": "between", "value": [0.0, 0.10]},
            ],
            "rank": [
                {"column": "edge_score_no_sector", "ascending": False, "weight": 1.2},
                {"column": "value_ratio_20", "ascending": False, "weight": 1.0},
                {"column": "rs_percentile_20", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.07, "take_profit": 0.22, "max_holding_bars": 30, "initial_atr_stop_mult": 2.1, "trailing_atr_mult": 2.5, "trailing_profit_activation": 0.07, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Minervini Trend Template",
        "public_rule": "Second-pass VN tuning: require trend-template leadership plus an actual breakout trigger.",
        "source_url": "https://www.chartmill.com/documentation/stock-screener/technical-analysis-trading-strategies/496-Mark-Minervini-Trend-Template-A-Step-by-Step-Guide-for-Beginners",
        "hypothesis": {
            "name": "public_minervini_breakout_vn_tuned_v2",
            "description": "Minervini VN variant with explicit breakout-quality trigger.",
            "universe": "VN100",
            "tags": ["public", "minervini", "trend_template", "breakout", "leader"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "mkt_regime_score", "op": ">=", "value": 50},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "above_ma200", "op": "==", "value": True},
                {"column": "breakout_20_quality", "op": "==", "value": True},
                {"column": "ma50_slope_10", "op": ">=", "value": 0.002},
                {"column": "ma200_slope_20", "op": ">=", "value": -0.003},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.70},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 55},
                {"column": "distribution_days_10", "op": "<=", "value": 1},
            ],
            "rank": [
                {"column": "edge_score_no_sector", "ascending": False, "weight": 1.2},
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
                {"column": "value_ratio_20", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.075, "take_profit": 0.25, "max_holding_bars": 45, "initial_atr_stop_mult": 2.3, "trailing_atr_mult": 2.8, "trailing_profit_activation": 0.09, "use_regime_exposure": True},
        },
    },
]


def install_strategies(strategies: list[dict]) -> None:
    if not CFG_BACKUP.exists():
        shutil.copy(CFG_PATH, CFG_BACKUP)
    cfg = json.loads(CFG_BACKUP.read_text(encoding="utf-8"))
    names = {item["name"] for item in cfg["hypotheses"]}
    for item in strategies:
        hyp = item["hypothesis"]
        if hyp["name"] not in names:
            cfg["hypotheses"].append(hyp)
    CFG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def restore_config() -> None:
    if CFG_BACKUP.exists():
        shutil.copy(CFG_BACKUP, CFG_PATH)


def _read_summary(label: str, start: str, end: str) -> dict | None:
    files = sorted((ROOT / "backtest_results/mvp_sizing_grid").glob(f"vn100_{label}_*_{start}_{end}_summary.csv"))
    if not files:
        return None
    summary = pd.read_csv(files[-1])
    row = summary[summary["model"] == "eod_next_open"].iloc[0].to_dict()
    intraday = summary[summary["model"] == "intraday_touch"].iloc[0].to_dict()
    return {
        "label": label,
        "start": start,
        "end": end,
        "eod_return_pct": round(float(row["total_return_pct"]), 2),
        "eod_sharpe": round(float(row["sharpe_ratio"]), 3),
        "eod_max_dd_pct": round(float(row["max_drawdown_pct"]), 2),
        "eod_win_rate_pct": round(float(row["win_rate_pct"]), 2),
        "eod_trades": int(row["number_of_trades"]),
        "eod_signals": int(row["signals"]),
        "eod_fill_rate_pct": round(float(row["fill_rate_pct"]), 2),
        "intraday_return_pct": round(float(intraday["total_return_pct"]), 2),
        "intraday_sharpe": round(float(intraday["sharpe_ratio"]), 3),
        "intraday_max_dd_pct": round(float(intraday["max_drawdown_pct"]), 2),
        "summary_file": str(files[-1]),
    }


def run_backtest(strategy_name: str, start: str, end: str, max_positions: int, max_candidates: int) -> dict | None:
    label = strategy_name.replace("public_", "").replace("_vn_tuned_v1", "")[:32]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.compare_live_pipeline_sizing_grid",
            "--universe",
            "vn100",
            "--start",
            start,
            "--end",
            end,
            "--sizing-mode",
            "cash_split_orders",
            "--edge-strategy",
            strategy_name,
            "--label",
            label,
            "--max-positions",
            str(max_positions),
            "--max-candidates-per-day",
            str(max_candidates),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {
            "label": label,
            "strategy": strategy_name,
            "error": result.stderr[-4000:],
        }
    metrics = _read_summary(label, start, end)
    if metrics:
        metrics["strategy"] = strategy_name
    return metrics


def write_report(rows: list[dict], start: str, end: str) -> Path:
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / f"public_strategy_fishing_{start}_{end}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    source_map = {
        item["hypothesis"]["name"]: {
            "source_family": item["source_family"],
            "public_rule": item["public_rule"],
            "source_url": item["source_url"],
        }
        for item in PUBLIC_STRATEGIES
    }
    if not df.empty and "strategy" in df.columns:
        df["source_family"] = df["strategy"].map(lambda name: source_map.get(name, {}).get("source_family", ""))
        df["public_rule"] = df["strategy"].map(lambda name: source_map.get(name, {}).get("public_rule", ""))
        df = df.sort_values(["eod_sharpe", "eod_return_pct"], ascending=[False, False], na_position="last")

    report_path = REPORTS_DIR / f"public_strategy_fishing_{end}.md"
    lines = [
        f"# Public Strategy Fishing - VN100 ({start} to {end})",
        "",
        "Scope: public-rule strategies encoded as VN100 long-only hypotheses, tuned with VN-specific regime, liquidity/value, relative-strength, and distribution filters. Backtest model uses next-open entries (`eod_next_open`) as the primary anti-lookahead result; `intraday_touch` is reported as a sensitivity check.",
        "",
        f"CSV: `{csv_path}`",
        "",
        "## Ranked Results",
        "",
    ]
    if df.empty:
        lines.append("No rows produced.")
    else:
        show_cols = [
            "source_family",
            "strategy",
            "eod_return_pct",
            "eod_sharpe",
            "eod_max_dd_pct",
            "eod_win_rate_pct",
            "eod_trades",
            "eod_signals",
            "intraday_return_pct",
            "intraday_sharpe",
        ]
        lines.append(_markdown_table(df[show_cols]))
    lines.extend(["", "## Sources Used", ""])
    for item in PUBLIC_STRATEGIES:
        lines.append(f"- {item['source_family']}: {item['public_rule']} Source: {item['source_url']}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is research output, not a trading recommendation.",
            "- Strategies with very few trades should be treated as unproven even if headline Sharpe is high.",
            "- The tuning is constrained to public-rule intent plus VN market hygiene filters; no symbol-specific manual fitting was used.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a simple Markdown table without pandas' optional tabulate dependency."""
    headers = [str(col) for col in df.columns]
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|"))
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-05-19")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    strategies = PUBLIC_STRATEGIES
    if args.only.strip():
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        strategies = [item for item in PUBLIC_STRATEGIES if item["hypothesis"]["name"] in wanted]

    install_strategies(strategies)
    rows: list[dict] = []
    try:
        for item in strategies:
            name = item["hypothesis"]["name"]
            print(f"[backtest] {name}")
            row = run_backtest(name, args.start, args.end, args.max_positions, args.max_candidates_per_day)
            if row:
                rows.append(row)
                if "error" in row:
                    print(f"  ERROR: {row['error'][-300:]}")
                else:
                    print(
                        f"  return={row['eod_return_pct']:+.2f}% "
                        f"sharpe={row['eod_sharpe']:.3f} "
                        f"dd={row['eod_max_dd_pct']:+.2f}% "
                        f"trades={row['eod_trades']}"
                    )
    finally:
        restore_config()

    report_path = write_report(rows, args.start, args.end)
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
