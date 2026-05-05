import numpy as np
import pandas as pd

from multiagents_trading_assistant.quantagents_backtest.backtest_engine import (
    BacktestConfig,
    backtest_strategy,
)
from multiagents_trading_assistant.quantagents_backtest.indicators import add_indicators
from multiagents_trading_assistant.quantagents_backtest.strategy_generator import (
    Condition,
    RiskRule,
    Strategy,
    generate_strategy_pool,
)
from multiagents_trading_assistant.quantagents_backtest.walk_forward import (
    WalkForwardConfig,
    run_walk_forward,
)
from multiagents_trading_assistant.quantagents_backtest.vn_quantagents import (
    RiskGateConfig,
    VNQuantAgentsConfig,
    run_vn_quantagents,
)
from multiagents_trading_assistant.quantagents_backtest.vn_portfolio_engine import (
    VNMarketCostConfig,
    VNPortfolioConfig,
    backtest_vn_portfolio,
)
from multiagents_trading_assistant.quantagents_backtest.vn_universe import (
    get_universe_snapshot,
    load_historical_constituents,
)


def _sample_ohlcv(n=220) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    trend = np.linspace(20.0, 45.0, n)
    cycle = np.sin(np.arange(n) / 7.0) * 1.5
    close = trend + cycle
    open_ = close * (1 + np.sin(np.arange(n)) * 0.002)
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = 1_000_000 + (np.arange(n) % 25) * 20_000
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_add_indicators_produces_required_columns():
    df = add_indicators(_sample_ohlcv(260))
    for col in [
        "ema20",
        "ema50",
        "ema200",
        "rsi_7",
        "rsi_14",
        "rsi_21",
        "macd",
        "macd_signal",
        "willr_14",
        "cmo_9",
        "stoch_k",
        "roc_9",
        "mom_10",
        "volume_ma20",
        "volume_spike",
        "obv",
        "vwma20",
        "bb_upper",
        "bb_percent",
        "atr_14",
        "adx_14",
        "aroon_up_14",
        "supertrend_dir",
        "kc_upper",
        "linreg_slope_14",
        "high_20_prev",
        "low_20_prev",
    ]:
        assert col in df.columns


def test_generate_strategy_pool_is_reproducible_and_diverse():
    left = generate_strategy_pool(n_strategies=80, seed=7)
    right = generate_strategy_pool(n_strategies=80, seed=7)
    assert [strategy.describe() for strategy in left] == [strategy.describe() for strategy in right]
    assert len(left) == 80
    assert len({strategy.family for strategy in left}) >= 5


def test_backtest_uses_next_bar_entry_without_lookahead():
    raw = _sample_ohlcv(80)
    df = add_indicators(raw)
    strategy = Strategy(
        strategy_id="unit",
        conditions=(Condition("close", ">", 0),),
        logic="AND",
        risk=RiskRule(stop_loss=0.5, take_profit=10.0, max_holding_bars=5),
        family="unit",
    )
    result = backtest_strategy(df, strategy, BacktestConfig(commission_rate=0.0, slippage_rate=0.0))

    assert result.trades
    first_trade = result.trades[0]
    assert first_trade["entry_time"] == df.index[1]
    assert first_trade["entry_price"] == df["open"].iloc[1]
    assert result.metrics["number_of_trades"] > 0


def test_walk_forward_returns_ranked_table_and_equity_curves():
    ohlcv = _sample_ohlcv(180)
    strategies = generate_strategy_pool(n_strategies=50, seed=11)
    result = run_walk_forward(
        ohlcv,
        strategies,
        WalkForwardConfig(train_bars=80, test_bars=40, step_bars=40, top_k=5, min_trades_train=1),
    )

    assert not result["ranked_table"].empty
    assert len(result["ranked_table"].head(5)) == 5
    assert not result["best_strategy_result"].equity_curve.empty
    assert not result["ensemble_result"].equity_curve.empty


def test_vn_quantagents_smoke_with_regime_and_risk_gate():
    universe = {
        "AAA": _sample_ohlcv(180),
        "BBB": _sample_ohlcv(180) * 1.05,
        "CCC": _sample_ohlcv(180) * 0.95,
    }
    strategies = generate_strategy_pool(n_strategies=50, seed=17)
    result = run_vn_quantagents(
        universe,
        market_index=_sample_ohlcv(180),
        strategies=strategies,
        config=VNQuantAgentsConfig(
            train_bars=80,
            test_bars=40,
            step_bars=40,
            top_k=5,
            n_strategies=50,
            risk_gate=RiskGateConfig(min_train_trades=1, min_symbols_tested=1),
        ),
    )

    assert not result["strategy_memory"].empty
    assert not result["portfolio_summary"].empty
    assert not result["best_strategy_equity"].empty
    assert not result["ensemble_equity"].empty
    assert "metrics" in result["execution_portfolio"]


def test_vn_portfolio_engine_uses_shared_capital_and_costs():
    universe = {
        "AAA": _sample_ohlcv(140),
        "BBB": _sample_ohlcv(140) * 1.02,
    }
    strategies = generate_strategy_pool(n_strategies=50, seed=19)[:5]
    regime = pd.Series("RISK_ON", index=_sample_ohlcv(140).index)
    result = backtest_vn_portfolio(
        universe,
        strategies,
        market_regime=regime,
        config=VNPortfolioConfig(
            initial_capital=100_000,
            max_positions=2,
            min_position_value=500,
            costs=VNMarketCostConfig(lot_size=10, settlement_bars=1),
        ),
    )

    assert not result["equity_curve"].empty
    assert result["metrics"]["number_of_trades"] >= 0
    assert result["equity_frame"]["positions"].max() <= 2


def test_historical_universe_loader_and_fallback(tmp_path):
    path = tmp_path / "constituents.csv"
    path.write_text(
        "effective_date,index,symbol\n"
        "2021-01-01,VN30,AAA\n"
        "2021-01-01,VN30,BBB\n"
        "2022-01-01,VN30,CCC\n",
        encoding="utf-8",
    )
    constituents = load_historical_constituents(path)
    snapshot = get_universe_snapshot("2021-06-01", constituents=constituents)
    assert snapshot.is_historical
    assert snapshot.symbols == ["AAA", "BBB"]

    fallback = get_universe_snapshot("2021-06-01")
    assert not fallback.is_historical
    assert len(fallback.symbols) == 30
