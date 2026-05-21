from pathlib import Path

from multiagents_trading_assistant.services import operating_review_service as svc


def test_filter_closed_trades_respects_as_of_date_and_window():
    trades = [
        {"trade_date": "2026-05-08", "realized_pnl": 100, "pnl_pct": 1.0},
        {"trade_date": "2026-05-10", "realized_pnl": -50, "pnl_pct": -0.5},
        {"trade_date": "2026-05-12", "realized_pnl": 200, "pnl_pct": 2.0},
        {"trade_date": "2026-05-13", "realized_pnl": 300, "pnl_pct": 3.0},
    ]

    recent = svc._filter_closed_trades(trades, as_of_date="2026-05-12", lookback_days=3)

    assert [row["trade_date"] for row in recent] == ["2026-05-10", "2026-05-12"]


def test_load_trusted_leaderboards_reads_only_trusted_summaries(tmp_path, monkeypatch):
    single = tmp_path / "single.csv"
    single.write_text(
        "strategy,total_return_pct,sharpe_ratio,max_drawdown_pct,win_rate_pct,number_of_trades\n"
        "alpha,12.0,1.1,-5.0,50.0,10\n"
        "beta,15.0,0.9,-6.0,48.0,12\n",
        encoding="utf-8",
    )
    combo_ytd = tmp_path / "combo_ytd.csv"
    combo_ytd.write_text(
        "combo,total_return_pct,sharpe_ratio,max_drawdown_pct,win_rate_pct,number_of_trades\n"
        "c1,20.0,1.5,-8.0,55.0,20\n",
        encoding="utf-8",
    )
    combo_4y = tmp_path / "combo_4y.csv"
    combo_4y.write_text(
        "combo,total_return_pct,sharpe_ratio,max_drawdown_pct,win_rate_pct,number_of_trades\n"
        "long_run,30.0,1.2,-12.0,52.0,50\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        svc,
        "TRUSTED_SUMMARY_FILES",
        {
            "single_strategy_ytd": single,
            "combo_strategy_ytd": combo_ytd,
            "combo_strategy_4y": combo_4y,
        },
    )

    leaders = svc.load_trusted_leaderboards()

    assert leaders["single_strategy_ytd"]["name"] == "beta"
    assert leaders["combo_strategy_ytd"]["name"] == "c1"
    assert leaders["combo_strategy_4y"]["name"] == "long_run"


def test_build_operating_review_aggregates_recent_pnl(monkeypatch):
    closed = [
        {"trade_date": "2026-05-10", "realized_pnl": 1_000_000, "pnl_pct": 2.0, "strategy": "A", "loss_category": "UNCLASSIFIED"},
        {"trade_date": "2026-05-11", "realized_pnl": -300_000, "pnl_pct": -1.0, "strategy": "B", "loss_category": "ENTRY_CHASE"},
        {"trade_date": "2026-05-12", "realized_pnl": 500_000, "pnl_pct": 1.5, "strategy": "A", "loss_category": "UNCLASSIFIED"},
    ]
    monkeypatch.setattr(svc.db, "get_closed_trades", lambda limit=1000: closed)
    monkeypatch.setattr(
        svc,
        "load_trusted_leaderboards",
        lambda: {"single_strategy_ytd": {"name": "leader", "total_return_pct": 40.0}},
    )
    monkeypatch.setattr(
        svc,
        "compute_performance_stats",
        lambda live_prices=None: {
            "open_positions_count": 2,
            "open_unrealized_pnl_vnd": 250_000,
            "total_realized_pnl_vnd": 1_200_000,
            "win_rate": 66.7,
            "avg_realized_rr": 1.8,
        },
    )

    class _Memory:
        def summary(self):
            return {"active_count": 3, "top_strategies": [{"setup": "A"}]}

    review = svc.build_operating_review(
        as_of_date="2026-05-12",
        lookback_days=3,
        strategy_memory=_Memory(),
    )

    assert review["recent_realized"]["trade_count"] == 3
    assert review["recent_realized"]["realized_pnl_vnd"] == 1_200_000.0
    assert review["setup_breakdown"][0]["setup"] == "A"
    assert review["loss_breakdown"][0]["loss_category"] == "ENTRY_CHASE"
    assert review["trusted_leaderboards"]["single_strategy_ytd"]["name"] == "leader"
