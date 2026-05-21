"""Backtest modern expert-inspired VN strategy hypotheses.

These are not classic textbook indicators. They translate newer systematic
ideas into the project's existing VN100 feature and live-pipeline backtester:
multi-factor momentum, sector rotation, canary gates, EP proxies, and
volatility-aware trend sleeves.
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
CFG_BACKUP = CFG_PATH.with_suffix(".modern_fishing.bak")
OUT_DIR = ROOT / "backtest_results/modern_strategy_fishing"
REPORTS_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


MODERN_STRATEGIES = [
    {
        "source_family": "SystemTrader Gemini / multi-factor momentum",
        "source_url": "https://www.systemtrader.co/gemini/methodology",
        "hypothesis": {
            "name": "modern_gemini_vn_momentum_v1",
            "description": "Multi-factor momentum score: RS, excess return, smart-money, value flow, low distribution, regime gate.",
            "universe": "VN100",
            "tags": ["modern", "momentum", "multi_factor", "gemini"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "mkt_regime_score", "op": ">=", "value": 45},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "ma50_slope_10", "op": ">=", "value": -0.005},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.68},
                {"column": "excess_ret_60d_pctile", "op": ">=", "value": 0.62},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 55},
                {"column": "value_ratio_20", "op": ">=", "value": 1.05},
                {"column": "distribution_days_10", "op": "<=", "value": 2},
            ],
            "rank": [
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.4},
                {"column": "excess_ret_60d_pctile", "ascending": False, "weight": 1.1},
                {"column": "smart_money_score_no_sector", "ascending": False, "weight": 1.0},
                {"column": "value_ratio_20", "ascending": False, "weight": 0.7},
                {"column": "distribution_days_10", "ascending": True, "weight": 0.4},
            ],
            "risk": {"stop_loss": 0.075, "take_profit": 0.24, "max_holding_bars": 35, "initial_atr_stop_mult": 2.3, "trailing_atr_mult": 2.6, "trailing_profit_activation": 0.08, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "SystemTrader Gemini / multi-factor momentum",
        "source_url": "https://www.systemtrader.co/gemini/methodology",
        "hypothesis": {
            "name": "modern_gemini_vn_momentum_strict_v2",
            "description": "Stricter Gemini-style VN momentum: high regime, high RS, high smart-money, low DS.",
            "universe": "VN100",
            "tags": ["modern", "momentum", "multi_factor", "strict"],
            "filters": [
                {"column": "mkt_regime_score", "op": ">=", "value": 55},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "ma20_slope_5", "op": ">=", "value": 0.0},
                {"column": "ma50_slope_10", "op": ">=", "value": 0.0},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.75},
                {"column": "excess_ret_60d_pctile", "op": ">=", "value": 0.70},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 62},
                {"column": "DS20", "op": "<=", "value": 0.52},
                {"column": "distribution_days_10", "op": "<=", "value": 1},
            ],
            "rank": [
                {"column": "edge_score_no_sector", "ascending": False, "weight": 1.5},
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
                {"column": "smart_money_score_no_sector_delta", "ascending": False, "weight": 0.7},
            ],
            "risk": {"stop_loss": 0.07, "take_profit": 0.22, "max_holding_bars": 30, "initial_atr_stop_mult": 2.1, "trailing_atr_mult": 2.5, "trailing_profit_activation": 0.07, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Modern sector rotation",
        "source_url": "https://blog.traderspost.io/article/sector-rotation-trading-strategies",
        "hypothesis": {
            "name": "modern_sector_rs_vn_v1",
            "description": "Sector rotation proxy: only buy stocks in top RS sectors with sector breadth and value confirmation.",
            "universe": "VN100",
            "tags": ["modern", "sector_rotation", "relative_strength"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "sector_rs_rank_20d", "op": ">=", "value": 0.70},
                {"column": "sector_breadth_ma20", "op": ">=", "value": 0.55},
                {"column": "sector_value_ratio_20", "op": ">=", "value": 1.00},
                {"column": "sector_leadership_score", "op": ">=", "value": 62},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 50},
            ],
            "rank": [
                {"column": "sector_leadership_score", "ascending": False, "weight": 1.4},
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
                {"column": "smart_money_score_no_sector", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.08, "take_profit": 0.25, "max_holding_bars": 45, "initial_atr_stop_mult": 2.4, "trailing_atr_mult": 2.8, "trailing_profit_activation": 0.09, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Modern sector rotation",
        "source_url": "https://www.systemtrader.co/etfs/sector-rotation",
        "hypothesis": {
            "name": "modern_sector_breakout_vn_v2",
            "description": "Sector leadership plus breakout trigger: strongest sectors, stock breakout quality, value expansion.",
            "universe": "VN100",
            "tags": ["modern", "sector_rotation", "breakout"],
            "filters": [
                {"column": "mkt_regime_score", "op": ">=", "value": 48},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "sector_rs_rank_20d", "op": ">=", "value": 0.70},
                {"column": "sector_breadth_ma20", "op": ">=", "value": 0.55},
                {"column": "breakout_20_quality", "op": "==", "value": True},
                {"column": "value_ratio_20", "op": ">=", "value": 1.15},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.60},
                {"column": "distribution_days_10", "op": "<=", "value": 2},
            ],
            "rank": [
                {"column": "sector_leadership_score", "ascending": False, "weight": 1.2},
                {"column": "breakout_20_quality", "ascending": False, "weight": 1.0},
                {"column": "value_ratio_20", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.075, "take_profit": 0.23, "max_holding_bars": 35, "initial_atr_stop_mult": 2.2, "trailing_atr_mult": 2.6, "trailing_profit_activation": 0.08, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Canary / tactical allocation gate",
        "source_url": "https://reblnc.com/strategies/haa",
        "hypothesis": {
            "name": "modern_canary_gate_breakout_vn_v1",
            "description": "Canary-gated breakout: market money-cycle and VNINDEX trend must be healthy before taking breakouts.",
            "universe": "VN100",
            "tags": ["modern", "canary", "risk_gate", "breakout"],
            "filters": [
                {"column": "mkt_regime_score", "op": ">=", "value": 58},
                {"column": "vni_above_ma20", "op": "==", "value": True},
                {"column": "vni_above_ma50", "op": "==", "value": True},
                {"column": "mkt_CHDM50", "op": ">=", "value": 45},
                {"column": "mkt_DS20", "op": "<=", "value": 0.55},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "breakout_20_quality", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.58},
            ],
            "rank": [
                {"column": "edge_score_no_sector", "ascending": False, "weight": 1.2},
                {"column": "value_ratio_20", "ascending": False, "weight": 0.8},
                {"column": "rs_percentile_20", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.07, "take_profit": 0.22, "max_holding_bars": 30, "initial_atr_stop_mult": 2.1, "trailing_atr_mult": 2.5, "trailing_profit_activation": 0.08, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Stockbee / TraderLion episodic pivot",
        "source_url": "https://traderlion.com/podcast/pradeep-bonde-episodic-pivots/",
        "hypothesis": {
            "name": "modern_ep_proxy_vn_v1",
            "description": "EP proxy without news: one-day price/value shock, strong close, RS, short holding period.",
            "universe": "VN100",
            "tags": ["modern", "episodic_pivot", "momentum_burst"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "ret_1d", "op": ">=", "value": 0.045},
                {"column": "body_pct", "op": ">=", "value": 0.035},
                {"column": "close_location", "op": ">=", "value": 0.70},
                {"column": "value_ratio_20", "op": ">=", "value": 1.80},
                {"column": "volume_ratio_20", "op": ">=", "value": 1.50},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.60},
                {"column": "distribution_days_10", "op": "<=", "value": 2},
            ],
            "rank": [
                {"column": "value_ratio_20", "ascending": False, "weight": 1.4},
                {"column": "ret_1d", "ascending": False, "weight": 1.0},
                {"column": "close_location", "ascending": False, "weight": 0.6},
            ],
            "risk": {"stop_loss": 0.06, "take_profit": 0.16, "max_holding_bars": 8, "initial_atr_stop_mult": 1.8, "trailing_atr_mult": 2.0, "trailing_profit_activation": 0.05, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Pocket Pivot / modern momentum continuation",
        "source_url": "https://stockrps.com/en-us/doc/terms/pocket/",
        "hypothesis": {
            "name": "modern_pocket_pivot_vn_v1",
            "description": "Pocket-pivot proxy: reclaim MA20 after pullback on high value with leader/sector confirmation.",
            "universe": "VN100",
            "tags": ["modern", "pocket_pivot", "pullback_continuation"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "reclaim_ma20_after_pullback", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "value_ratio_20", "op": ">=", "value": 1.20},
                {"column": "close_location", "op": ">=", "value": 0.60},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
                {"column": "sector_leadership_score", "op": ">=", "value": 55},
                {"column": "distribution_days_10", "op": "<=", "value": 2},
            ],
            "rank": [
                {"column": "pullback_quality_score", "ascending": False, "weight": 1.2},
                {"column": "value_ratio_20", "ascending": False, "weight": 1.0},
                {"column": "sector_leadership_score", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.065, "take_profit": 0.18, "max_holding_bars": 20, "initial_atr_stop_mult": 2.0, "trailing_atr_mult": 2.3, "trailing_profit_activation": 0.06, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Volatility squeeze / momentum burst",
        "source_url": "https://www.tradewink.com/learn/what-is-market-regime",
        "hypothesis": {
            "name": "modern_vol_squeeze_momentum_v1",
            "description": "Volatility squeeze breakout with momentum and volume confirmation.",
            "universe": "VN100",
            "tags": ["modern", "volatility_squeeze", "momentum"],
            "filters": [
                {"column": "mkt_regime_state", "op": "not_in", "value": ["RISK_OFF"]},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "volatility_squeeze_break", "op": "==", "value": True},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "value_ratio_20", "op": ">=", "value": 1.10},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.55},
            ],
            "rank": [
                {"column": "value_ratio_20", "ascending": False, "weight": 1.0},
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.0},
                {"column": "range_pct", "ascending": True, "weight": 0.5},
            ],
            "risk": {"stop_loss": 0.07, "take_profit": 0.20, "max_holding_bars": 25, "initial_atr_stop_mult": 2.1, "trailing_atr_mult": 2.4, "trailing_profit_activation": 0.07, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Regime-aware trend model",
        "source_url": "https://arxiv.org/abs/2510.14986",
        "hypothesis": {
            "name": "modern_kalman_regime_trend_v1",
            "description": "Regime-aware Kalman trend: smoother trend confirmation, shock control, sector and market filters.",
            "universe": "VN100",
            "tags": ["modern", "regime", "kalman", "trend"],
            "filters": [
                {"column": "mkt_regime_score", "op": ">=", "value": 48},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "kalman_strong_uptrend", "op": "==", "value": True},
                {"column": "kalman_confidence", "op": ">=", "value": 0.45},
                {"column": "kalman_shock_score", "op": "<=", "value": 65},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "sector_leadership_score", "op": ">=", "value": 55},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 52},
            ],
            "rank": [
                {"column": "kalman_confidence", "ascending": False, "weight": 1.2},
                {"column": "sector_leadership_score", "ascending": False, "weight": 0.9},
                {"column": "smart_money_score_no_sector", "ascending": False, "weight": 0.9},
            ],
            "risk": {"stop_loss": 0.075, "take_profit": 0.23, "max_holding_bars": 35, "initial_atr_stop_mult": 2.2, "trailing_atr_mult": 2.6, "trailing_profit_activation": 0.08, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Pocket Pivot / modern momentum continuation",
        "source_url": "https://stockrps.com/en-us/doc/terms/pocket/",
        "hypothesis": {
            "name": "modern_pocket_pivot_quality_v2",
            "description": "Tuned pocket pivot: stronger sector/SMT quality and shorter risk window.",
            "universe": "VN100",
            "tags": ["modern", "pocket_pivot", "quality_tune"],
            "filters": [
                {"column": "mkt_regime_score", "op": ">=", "value": 50},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "reclaim_ma20_after_pullback", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "value_ratio_20", "op": ">=", "value": 1.35},
                {"column": "close_location", "op": ">=", "value": 0.65},
                {"column": "pullback_quality_score", "op": ">=", "value": 55},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.60},
                {"column": "sector_rs_rank_20d", "op": ">=", "value": 0.60},
                {"column": "sector_leadership_score", "op": ">=", "value": 60},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 55},
                {"column": "DS20", "op": "<=", "value": 0.55},
                {"column": "distribution_days_10", "op": "<=", "value": 1},
            ],
            "rank": [
                {"column": "pullback_quality_score", "ascending": False, "weight": 1.2},
                {"column": "sector_leadership_score", "ascending": False, "weight": 1.0},
                {"column": "smart_money_score_no_sector", "ascending": False, "weight": 0.9},
                {"column": "value_ratio_20", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.055, "take_profit": 0.16, "max_holding_bars": 15, "initial_atr_stop_mult": 1.8, "trailing_atr_mult": 2.1, "trailing_profit_activation": 0.05, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Pocket Pivot / modern momentum continuation",
        "source_url": "https://stockrps.com/en-us/doc/terms/pocket/",
        "hypothesis": {
            "name": "modern_pocket_pivot_runner_v3",
            "description": "Tuned pocket pivot runner: same continuation setup, looser profit target for stronger regimes.",
            "universe": "VN100",
            "tags": ["modern", "pocket_pivot", "runner_tune"],
            "filters": [
                {"column": "mkt_regime_score", "op": ">=", "value": 55},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "reclaim_ma20_after_pullback", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "value_ratio_20", "op": ">=", "value": 1.20},
                {"column": "close_location", "op": ">=", "value": 0.60},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.62},
                {"column": "sector_leadership_score", "op": ">=", "value": 60},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 52},
                {"column": "distribution_days_10", "op": "<=", "value": 2},
            ],
            "rank": [
                {"column": "edge_score_no_sector", "ascending": False, "weight": 1.2},
                {"column": "pullback_quality_score", "ascending": False, "weight": 1.0},
                {"column": "sector_leadership_score", "ascending": False, "weight": 0.9},
            ],
            "risk": {"stop_loss": 0.065, "take_profit": 0.24, "max_holding_bars": 30, "initial_atr_stop_mult": 2.0, "trailing_atr_mult": 2.7, "trailing_profit_activation": 0.08, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Modern sector rotation",
        "source_url": "https://www.systemtrader.co/etfs/sector-rotation",
        "hypothesis": {
            "name": "modern_sector_quality_defensive_v2",
            "description": "Tuned sector quality: only strongest sectors, above MA20/50, low distribution.",
            "universe": "VN100",
            "tags": ["modern", "sector_rotation", "quality_tune"],
            "filters": [
                {"column": "mkt_regime_score", "op": ">=", "value": 50},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "sector_rs_rank_20d", "op": ">=", "value": 0.75},
                {"column": "sector_breadth_ma20", "op": ">=", "value": 0.60},
                {"column": "sector_leadership_score", "op": ">=", "value": 68},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.62},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 55},
                {"column": "DS20", "op": "<=", "value": 0.58},
                {"column": "distribution_days_10", "op": "<=", "value": 1},
            ],
            "rank": [
                {"column": "sector_leadership_score", "ascending": False, "weight": 1.5},
                {"column": "edge_score_no_sector", "ascending": False, "weight": 1.0},
                {"column": "rs_percentile_20", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.065, "take_profit": 0.20, "max_holding_bars": 30, "initial_atr_stop_mult": 2.0, "trailing_atr_mult": 2.4, "trailing_profit_activation": 0.07, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Canary / tactical allocation gate",
        "source_url": "https://reblnc.com/strategies/haa",
        "hypothesis": {
            "name": "modern_canary_pocket_vn_v2",
            "description": "Canary-gated pocket pivot: VNINDEX trend and market money-cycle must confirm before pullback continuation.",
            "universe": "VN100",
            "tags": ["modern", "canary", "pocket_pivot", "risk_gate"],
            "filters": [
                {"column": "mkt_regime_score", "op": ">=", "value": 58},
                {"column": "vni_above_ma20", "op": "==", "value": True},
                {"column": "vni_above_ma50", "op": "==", "value": True},
                {"column": "mkt_DS20", "op": "<=", "value": 0.55},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "reclaim_ma20_after_pullback", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "value_ratio_20", "op": ">=", "value": 1.20},
                {"column": "close_location", "op": ">=", "value": 0.60},
                {"column": "sector_leadership_score", "op": ">=", "value": 58},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.58},
            ],
            "rank": [
                {"column": "edge_score_no_sector", "ascending": False, "weight": 1.2},
                {"column": "pullback_quality_score", "ascending": False, "weight": 1.0},
                {"column": "sector_leadership_score", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.06, "take_profit": 0.18, "max_holding_bars": 18, "initial_atr_stop_mult": 1.9, "trailing_atr_mult": 2.2, "trailing_profit_activation": 0.06, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "SystemTrader Gemini / multi-factor momentum",
        "source_url": "https://www.systemtrader.co/gemini/methodology",
        "hypothesis": {
            "name": "modern_gemini_leader_runner_v3",
            "description": "Aggressive 2025 bull-market leader runner: strict momentum entry, wider stop, higher TP, longer hold.",
            "universe": "VN100",
            "tags": ["modern", "momentum", "leader", "runner"],
            "filters": [
                {"column": "mkt_regime_score", "op": ">=", "value": 50},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "ma20_slope_5", "op": ">=", "value": 0.0},
                {"column": "ma50_slope_10", "op": ">=", "value": -0.002},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.70},
                {"column": "excess_ret_60d_pctile", "op": ">=", "value": 0.65},
                {"column": "smart_money_score_no_sector", "op": ">=", "value": 58},
                {"column": "sector_leadership_score", "op": ">=", "value": 55},
                {"column": "distribution_days_10", "op": "<=", "value": 2},
            ],
            "rank": [
                {"column": "edge_score_no_sector", "ascending": False, "weight": 1.4},
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.2},
                {"column": "excess_ret_60d_pctile", "ascending": False, "weight": 1.0},
                {"column": "smart_money_score_no_sector_delta", "ascending": False, "weight": 0.6},
            ],
            "risk": {"stop_loss": 0.11, "take_profit": 0.55, "max_holding_bars": 90, "initial_atr_stop_mult": 3.2, "trailing_atr_mult": 4.2, "trailing_profit_activation": 0.18, "use_regime_exposure": True},
        },
    },
    {
        "source_family": "Volatility squeeze / momentum burst",
        "source_url": "https://www.tradewink.com/learn/what-is-market-regime",
        "hypothesis": {
            "name": "modern_vol_squeeze_leader_runner_v2",
            "description": "Vol squeeze with leader filter and wider runner exits for 2025-style bull trends.",
            "universe": "VN100",
            "tags": ["modern", "volatility_squeeze", "leader", "runner"],
            "filters": [
                {"column": "mkt_regime_score", "op": ">=", "value": 50},
                {"column": "data_quality_ok", "op": "==", "value": True},
                {"column": "volatility_squeeze_break", "op": "==", "value": True},
                {"column": "above_ma20", "op": "==", "value": True},
                {"column": "above_ma50", "op": "==", "value": True},
                {"column": "value_ratio_20", "op": ">=", "value": 1.10},
                {"column": "rs_percentile_20", "op": ">=", "value": 0.60},
                {"column": "sector_leadership_score", "op": ">=", "value": 55},
                {"column": "distribution_days_10", "op": "<=", "value": 2},
            ],
            "rank": [
                {"column": "rs_percentile_20", "ascending": False, "weight": 1.1},
                {"column": "value_ratio_20", "ascending": False, "weight": 1.0},
                {"column": "sector_leadership_score", "ascending": False, "weight": 0.8},
            ],
            "risk": {"stop_loss": 0.09, "take_profit": 0.35, "max_holding_bars": 60, "initial_atr_stop_mult": 2.8, "trailing_atr_mult": 3.4, "trailing_profit_activation": 0.12, "use_regime_exposure": True},
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


def _label(strategy_name: str) -> str:
    return strategy_name.replace("modern_", "")[:40]


def _read_summary(label: str, start: str, end: str) -> dict | None:
    files = sorted((ROOT / "backtest_results/mvp_sizing_grid").glob(f"vn100_{label}_*_{start}_{end}_summary.csv"))
    if not files:
        return None
    summary = pd.read_csv(files[-1])
    eod = summary[summary["model"] == "eod_next_open"].iloc[0].to_dict()
    intraday = summary[summary["model"] == "intraday_touch"].iloc[0].to_dict()
    return {
        "label": label,
        "eod_return_pct": round(float(eod["total_return_pct"]), 2),
        "eod_sharpe": round(float(eod["sharpe_ratio"]), 3),
        "eod_max_dd_pct": round(float(eod["max_drawdown_pct"]), 2),
        "eod_win_rate_pct": round(float(eod["win_rate_pct"]), 2),
        "eod_trades": int(eod["number_of_trades"]),
        "eod_signals": int(eod["signals"]),
        "eod_fill_rate_pct": round(float(eod["fill_rate_pct"]), 2),
        "intraday_return_pct": round(float(intraday["total_return_pct"]), 2),
        "intraday_sharpe": round(float(intraday["sharpe_ratio"]), 3),
        "summary_file": str(files[-1]),
    }


def run_backtest(strategy_name: str, start: str, end: str, max_positions: int, max_candidates: int) -> dict | None:
    label = _label(strategy_name)
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
        return {"label": label, "strategy": strategy_name, "error": result.stderr[-4000:]}
    metrics = _read_summary(label, start, end)
    if metrics:
        metrics["strategy"] = strategy_name
    return metrics


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        values = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.3f}" if "sharpe" in col else f"{value:.2f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(rows: list[dict], start: str, end: str) -> Path:
    df = pd.DataFrame(rows)
    source_map = {
        item["hypothesis"]["name"]: {"source_family": item["source_family"], "source_url": item["source_url"]}
        for item in MODERN_STRATEGIES
    }
    if not df.empty and "strategy" in df.columns:
        df["source_family"] = df["strategy"].map(lambda name: source_map.get(name, {}).get("source_family", ""))
        df["source_url"] = df["strategy"].map(lambda name: source_map.get(name, {}).get("source_url", ""))
        df = df.sort_values(["eod_sharpe", "eod_return_pct"], ascending=[False, False], na_position="last")
    csv_path = OUT_DIR / f"modern_strategy_fishing_{start}_{end}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    report_path = REPORTS_DIR / f"modern_strategy_fishing_{end}.md"
    lines = [
        f"# Modern Strategy Fishing - VN100 ({start} to {end})",
        "",
        "Primary model: `eod_next_open`, long-only VN100, max 5 positions, top 10 candidates/day, cash-split orders.",
        "",
        f"CSV: `{csv_path}`",
        "",
        "## Ranked Results",
        "",
    ]
    if df.empty:
        lines.append("No rows produced.")
    else:
        lines.append(
            _markdown_table(
                df,
                [
                    "source_family",
                    "strategy",
                    "eod_return_pct",
                    "eod_sharpe",
                    "eod_max_dd_pct",
                    "eod_win_rate_pct",
                    "eod_trades",
                    "eod_signals",
                ],
            )
        )
    lines.extend(["", "## Source URLs", ""])
    for item in MODERN_STRATEGIES:
        lines.append(f"- {item['source_family']}: {item['source_url']}")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-05-19")
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-candidates-per-day", type=int, default=10)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    strategies = MODERN_STRATEGIES
    if args.only.strip():
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        strategies = [item for item in MODERN_STRATEGIES if item["hypothesis"]["name"] in wanted]

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
