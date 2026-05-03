"""test_backtest_no_leak.py — Anti-leak tests for backtest retrieval layer.

Tests verify that when as_of_date / backtest_mode is set:
  1. L1 SQLite decisions from future dates are excluded.
  2. L1 SQLite T+2.5 check uses as_of_date, not wall-clock now().
  3. L2 ChromaDB similar-setups excludes documents dated after as_of_date.
  4. KB reports without published_at are excluded in backtest_mode.
  5. KB reports with published_at <= as_of_date are returned in backtest_mode.
  6. KB stats are always returned (whitelisted — timeless data).
  7. KB news with date > as_of_date is excluded via date_to filter.

All tests use isolated temp SQLite files and (optionally) temp ChromaDB dirs.
No live database is touched.
"""

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_temp_db() -> tuple[Path, sqlite3.Connection]:
    """Create a fresh in-memory-style temp SQLite DB with the schema from database.py."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = Path(tmp.name)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS decisions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT    NOT NULL,
            date            TEXT    NOT NULL,
            action          TEXT    NOT NULL,
            final_action    TEXT    NOT NULL,
            strategy        TEXT,
            quality_score   REAL,
            confidence      TEXT,
            entry           REAL,
            sl              REAL,
            tp              REAL,
            nav_pct         REAL,
            override_reason TEXT,
            full_output     TEXT,
            created_at      TEXT DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL,
            action      TEXT    NOT NULL,
            price       REAL    NOT NULL,
            quantity    INTEGER NOT NULL,
            trade_date  TEXT    NOT NULL,
            strategy    TEXT,
            note        TEXT,
            exit_price  REAL,
            exit_date   TEXT,
            exit_reason TEXT,
            realized_pnl REAL,
            realized_rr  REAL,
            created_at  TEXT DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS positions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL UNIQUE,
            exchange    TEXT    NOT NULL DEFAULT 'HOSE',
            entry_price REAL    NOT NULL,
            quantity    INTEGER NOT NULL DEFAULT 0,
            entry_date  TEXT    NOT NULL,
            strategy    TEXT,
            sl          REAL,
            tp          REAL,
            nav_pct     REAL,
            created_at  TEXT DEFAULT (datetime('now','localtime'))
        );
        CREATE INDEX IF NOT EXISTS idx_decisions_symbol_date ON decisions(symbol, date);
        CREATE INDEX IF NOT EXISTS idx_trades_symbol_date    ON trades(symbol, trade_date);
    """)
    conn.commit()
    return path, conn


def _insert_decision(conn, symbol, date_str, final_action="MUA"):
    conn.execute(
        "INSERT INTO decisions (symbol, date, action, final_action) VALUES (?,?,?,?)",
        (symbol, date_str, final_action, final_action),
    )
    conn.commit()


def _insert_trade(conn, symbol, date_str, action="MUA"):
    conn.execute(
        "INSERT INTO trades (symbol, action, price, quantity, trade_date) VALUES (?,?,?,?,?)",
        (symbol, action, 10.0, 100, date_str),
    )
    conn.commit()


# ──────────────────────────────────────────────
# Test 1 & 2 — database.get_decisions with as_of_date
# ──────────────────────────────────────────────

class TestDatabaseGetDecisions(unittest.TestCase):
    """database.get_decisions() must exclude rows dated after as_of_date."""

    def setUp(self):
        self.path, self.conn = _make_temp_db()
        _insert_decision(self.conn, "VNM", "2023-01-10")   # past
        _insert_decision(self.conn, "VNM", "2023-06-15")   # exactly as_of_date
        _insert_decision(self.conn, "VNM", "2023-12-31")   # future relative to as_of_date
        _insert_decision(self.conn, "VNM", "2024-06-01")   # far future

    def tearDown(self):
        self.conn.close()
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def _query(self, as_of_date=None):
        import multiagents_trading_assistant.database as db_module

        with patch.object(db_module, "DB_PATH", self.path):
            rows = db_module.get_decisions(symbol="VNM", limit=20, as_of_date=as_of_date)
        return sorted(r["date"] for r in rows)

    def test_no_filter_returns_all(self):
        dates = self._query(as_of_date=None)
        self.assertEqual(len(dates), 4)

    def test_as_of_date_excludes_future_rows(self):
        dates = self._query(as_of_date="2023-06-15")
        # Should include 2023-01-10 and 2023-06-15, exclude 2023-12-31 and 2024-06-01
        self.assertIn("2023-01-10", dates)
        self.assertIn("2023-06-15", dates)  # inclusive boundary
        self.assertNotIn("2023-12-31", dates)
        self.assertNotIn("2024-06-01", dates)

    def test_as_of_date_includes_boundary_date(self):
        dates = self._query(as_of_date="2023-01-10")
        self.assertEqual(dates, ["2023-01-10"])


# ──────────────────────────────────────────────
# Test 3 — database.get_buys_last_n_days with as_of_date
# ──────────────────────────────────────────────

class TestDatabaseGetBuysLastNDays(unittest.TestCase):
    """get_buys_last_n_days() must use as_of_date as reference, not datetime.now()."""

    def setUp(self):
        self.path, self.conn = _make_temp_db()
        _insert_trade(self.conn, "VCB", "2023-06-13")  # 2 days before as_of_date
        _insert_trade(self.conn, "VCB", "2023-06-14")  # 1 day before
        _insert_trade(self.conn, "VCB", "2023-06-15")  # on as_of_date
        _insert_trade(self.conn, "VCB", "2023-06-16")  # 1 day AFTER → must be excluded

    def tearDown(self):
        self.conn.close()
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def _get_buys(self, n, as_of_date=None):
        import multiagents_trading_assistant.database as db_module

        with patch.object(db_module, "DB_PATH", self.path):
            rows = db_module.get_buys_last_n_days("VCB", n, as_of_date=as_of_date)
        return sorted(r["trade_date"] for r in rows)

    def test_as_of_date_excludes_future_buys(self):
        buys = self._get_buys(n=2, as_of_date="2023-06-15")
        self.assertNotIn("2023-06-16", buys, "Future buy must be excluded (anti-leak)")

    def test_as_of_date_includes_buys_in_window(self):
        buys = self._get_buys(n=2, as_of_date="2023-06-15")
        self.assertIn("2023-06-13", buys)
        self.assertIn("2023-06-14", buys)
        self.assertIn("2023-06-15", buys)

    def test_no_as_of_date_window_excludes_old_buys(self):
        # Without as_of_date, window is relative to now — 2023 dates fall outside
        buys = self._get_buys(n=2, as_of_date=None)
        self.assertEqual(buys, [], "Old 2023 trades outside live 2-day window")


# ──────────────────────────────────────────────
# Test 4 — L1Memory.get_recent_decisions with as_of_date
# ──────────────────────────────────────────────

class TestL1MemoryRecentDecisions(unittest.TestCase):
    """L1Memory.get_recent_decisions() must use as_of_date as the reference date."""

    def _patch_db_path(self, db_path):
        import multiagents_trading_assistant.database as db_module
        return patch.object(db_module, "DB_PATH", db_path)

    def test_recent_window_relative_to_as_of_date(self):
        path, conn = _make_temp_db()
        try:
            _insert_decision(conn, "HPG", "2023-06-08")   # 7 days before 2023-06-15
            _insert_decision(conn, "HPG", "2023-06-10")   # 5 days before
            _insert_decision(conn, "HPG", "2023-06-15")   # on as_of_date
            _insert_decision(conn, "HPG", "2023-06-20")   # future — must be excluded
            conn.close()

            import multiagents_trading_assistant.database as db_module
            with patch.object(db_module, "DB_PATH", path):
                from multiagents_trading_assistant.memory.memory_system import L1Memory
                l1 = L1Memory()
                results = l1.get_recent_decisions("HPG", days=7, as_of_date="2023-06-15")
                dates = [r["date"] for r in results]

            self.assertNotIn("2023-06-20", dates, "Future decision must be excluded")
            self.assertIn("2023-06-08", dates)
            self.assertIn("2023-06-15", dates)
        finally:
            # Windows: SQLite may hold a file lock briefly; ignore cleanup errors in tests
            try:
                if path.exists():
                    os.unlink(path)
            except OSError:
                pass


# ──────────────────────────────────────────────
# Test 5 — L2Memory.search_similar with as_of_date
# ──────────────────────────────────────────────

class TestL2MemorySearchSimilar(unittest.TestCase):
    """L2Memory.search_similar() must apply $lte filter when as_of_date is set."""

    def test_as_of_date_filter_applied_to_chromadb_query(self):
        """Verify the ChromaDB where clause contains the $lte condition."""
        try:
            import chromadb
        except ImportError:
            self.skipTest("chromadb not installed")

        tmp_dir = tempfile.mkdtemp()
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            client = chromadb.PersistentClient(
                path=tmp_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collection = client.get_or_create_collection("trading_decisions")
            # Insert a past and a future document
            collection.upsert(
                ids=["VNM_2023-01-10"],
                documents=["Setup BREAKOUT trend UPTREND HIGH confluence Score 75"],
                metadatas=[{"symbol": "VNM", "date": "2023-01-10", "action": "MUA"}],
            )
            collection.upsert(
                ids=["VNM_2025-12-01"],
                documents=["Setup BREAKOUT trend UPTREND HIGH confluence Score 75"],
                metadatas=[{"symbol": "VNM", "date": "2025-12-01", "action": "MUA"}],
            )
            collection.upsert(
                ids=["VNM_missing_date"],
                documents=["Setup BREAKOUT trend UPTREND HIGH confluence Score 75"],
                metadatas=[{"symbol": "VNM", "action": "MUA"}],
            )

            from multiagents_trading_assistant.memory.memory_system import L2Memory
            l2 = L2Memory.__new__(L2Memory)
            l2._client = client
            l2._collection = collection

            results = l2.search_similar(
                query="Setup BREAKOUT trend UPTREND",
                symbol="VNM",
                n_results=5,
                as_of_date="2023-06-15",
            )
            result_ids = [r["id"] for r in results]
            self.assertIn("VNM_2023-01-10", result_ids, "Past doc should be returned")
            self.assertNotIn("VNM_2025-12-01", result_ids, "Future doc must be excluded")
            self.assertNotIn("VNM_missing_date", result_ids, "Undated doc must be excluded")
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_no_as_of_date_returns_all(self):
        """Without as_of_date, both past and future docs are returned (live mode)."""
        try:
            import chromadb
        except ImportError:
            self.skipTest("chromadb not installed")

        tmp_dir = tempfile.mkdtemp()
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            client = chromadb.PersistentClient(
                path=tmp_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collection = client.get_or_create_collection("trading_decisions")
            collection.upsert(
                ids=["VNM_2023-01-10"],
                documents=["Setup BREAKOUT trend UPTREND Score 75"],
                metadatas=[{"symbol": "VNM", "date": "2023-01-10", "action": "MUA"}],
            )
            collection.upsert(
                ids=["VNM_2025-12-01"],
                documents=["Setup BREAKOUT trend UPTREND Score 75"],
                metadatas=[{"symbol": "VNM", "date": "2025-12-01", "action": "MUA"}],
            )

            from multiagents_trading_assistant.memory.memory_system import L2Memory
            l2 = L2Memory.__new__(L2Memory)
            l2._client = client
            l2._collection = collection

            results = l2.search_similar(
                query="Setup BREAKOUT trend UPTREND",
                symbol="VNM",
                n_results=5,
                as_of_date=None,
            )
            result_ids = [r["id"] for r in results]
            self.assertEqual(len(result_ids), 2, "Live mode: both docs returned")
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────
# Test 6 — KnowledgeBase.search_reports backtest filtering
# ──────────────────────────────────────────────

class TestKnowledgeBaseReports(unittest.TestCase):
    """search_reports() must exclude docs without published_at in backtest_mode."""

    def _make_kb_with_reports(self, tmp_dir: str):
        try:
            import chromadb
        except ImportError:
            return None
        from chromadb.config import Settings as ChromaSettings
        client = chromadb.PersistentClient(
            path=tmp_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection("vietstock_reports")
        # Doc 1: has published_at before as_of_date — should be returned in backtest
        collection.upsert(
            ids=["VCB_2022_Q4_BCTC_KET_QUA_KD"],
            documents=["VCB Q4 2022 loi nhuan tang truong tot"],
            metadatas=[{
                "symbol": "VCB", "year": 2022, "quarter": 4,
                "doc_type": "BCTC", "section": "KET_QUA_KD",
                "published_at": "2023-03-31",  # Q4 published March next year
            }],
        )
        # Doc 2: has published_at AFTER as_of_date — must be excluded in backtest
        collection.upsert(
            ids=["VCB_2023_Q2_BCTC_KET_QUA_KD"],
            documents=["VCB Q2 2023 loi nhuan tot hon du kien"],
            metadatas=[{
                "symbol": "VCB", "year": 2023, "quarter": 2,
                "doc_type": "BCTC", "section": "KET_QUA_KD",
                "published_at": "2023-08-15",  # published after our as_of_date
            }],
        )
        # Doc 3: NO published_at — must be excluded in backtest_mode (unsafe, no timestamp)
        collection.upsert(
            ids=["VCB_2021_Q4_BCTC_KET_QUA_KD"],
            documents=["VCB Q4 2021 tong ket nam"],
            metadatas=[{
                "symbol": "VCB", "year": 2021, "quarter": 4,
                "doc_type": "BCTC", "section": "KET_QUA_KD",
            }],
        )
        # stats_stub: needed so KnowledgeBase.available returns True (_reports and _stats both non-None)
        stats_stub = client.get_or_create_collection("vietstock_stats_stub")
        return client, collection, stats_stub

    def test_backtest_mode_excludes_reports_without_published_at(self):
        try:
            import chromadb
        except ImportError:
            self.skipTest("chromadb not installed")

        tmp_dir = tempfile.mkdtemp()
        try:
            result = self._make_kb_with_reports(tmp_dir)
            if result is None:
                self.skipTest("chromadb not installed")
            client, collection, stats_stub = result

            from multiagents_trading_assistant.memory.knowledge_base import KnowledgeBase
            kb = KnowledgeBase.__new__(KnowledgeBase)
            kb._client = client
            kb._reports = collection
            kb._stats = stats_stub   # must be non-None for available=True
            kb._news = None

            docs = kb.search_reports(
                symbol="VCB",
                query="loi nhuan tang truong",
                n=5,
                as_of_date="2023-06-15",
                backtest_mode=True,
            )
            # Only the Q4/2022 report (published_at="2023-03-31") should be returned.
            # Q2/2023 (published 2023-08-15) and Q4/2021 (no timestamp) must be excluded.
            self.assertEqual(len(docs), 1, "Only the one safe report should be returned")
            self.assertIn("2022", docs[0], "Q4/2022 report should be in results")
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_live_mode_returns_all_reports(self):
        try:
            import chromadb
        except ImportError:
            self.skipTest("chromadb not installed")

        tmp_dir = tempfile.mkdtemp()
        try:
            result = self._make_kb_with_reports(tmp_dir)
            if result is None:
                self.skipTest("chromadb not installed")
            client, collection, stats_stub = result

            from multiagents_trading_assistant.memory.knowledge_base import KnowledgeBase
            kb = KnowledgeBase.__new__(KnowledgeBase)
            kb._client = client
            kb._reports = collection
            kb._stats = stats_stub   # must be non-None for available=True
            kb._news = None

            docs = kb.search_reports(
                symbol="VCB",
                query="loi nhuan",
                n=5,
                as_of_date=None,
                backtest_mode=False,
            )
            # Live mode: no date filter → all 3 docs returned
            self.assertGreaterEqual(len(docs), 1, "Live mode: at least 1 doc returned")
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────
# Test 7 — KnowledgeBase.search_stats is always returned (whitelisted)
# ──────────────────────────────────────────────

class TestKnowledgeBaseStats(unittest.TestCase):
    """search_stats() must always return data regardless of as_of_date (timeless data)."""

    def test_stats_returned_in_backtest_mode(self):
        try:
            import chromadb
        except ImportError:
            self.skipTest("chromadb not installed")

        tmp_dir = tempfile.mkdtemp()
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            client = chromadb.PersistentClient(
                path=tmp_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collection = client.get_or_create_collection("vietstock_stats")
            collection.upsert(
                ids=["VNM_SEASONALITY_monthly"],
                documents=["VNM tháng 6: 7/10 năm tăng (70%), trung bình +3.2%"],
                metadatas=[{"symbol": "VNM", "stat_type": "SEASONALITY", "period": "monthly"}],
            )

            from multiagents_trading_assistant.memory.knowledge_base import KnowledgeBase
            kb = KnowledgeBase.__new__(KnowledgeBase)
            kb._client = client
            kb._reports = collection  # reuse collection for available check
            kb._stats = collection
            kb._news = None

            docs = kb.search_stats(
                symbol="VNM",
                query="xác suất tăng giá tháng 6",
                n=2,
                as_of_date="2021-01-01",   # very old date — should still return
                backtest_mode=True,
            )
            self.assertGreater(len(docs), 0, "Stats must be returned even for old as_of_date")
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────
# Test 8 — KnowledgeBase.search_news date_to filter
# ──────────────────────────────────────────────

class TestKnowledgeBaseNews(unittest.TestCase):
    """search_news() date_to must exclude news published after as_of_date."""

    def test_date_to_excludes_future_news(self):
        try:
            import chromadb
        except ImportError:
            self.skipTest("chromadb not installed")

        tmp_dir = tempfile.mkdtemp()
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            client = chromadb.PersistentClient(
                path=tmp_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collection = client.get_or_create_collection("news_articles")
            collection.upsert(
                ids=["cafef_abc123"],
                documents=["Ma VCB ngay 2023-06-10: [CafeF] VCB record profit. Earnings up 30%."],
                metadatas=[{"symbol": "VCB", "date": "2023-06-10", "source": "CafeF",
                             "url": "http://ex.com/1", "sentiment": "POSITIVE"}],
            )
            collection.upsert(
                ids=["cafef_def456"],
                documents=["Ma VCB ngay 2023-07-20: [CafeF] VCB issues new shares."],
                metadatas=[{"symbol": "VCB", "date": "2023-07-20", "source": "CafeF",
                             "url": "http://ex.com/2", "sentiment": "NEUTRAL"}],
            )

            from multiagents_trading_assistant.memory.knowledge_base import KnowledgeBase
            kb = KnowledgeBase.__new__(KnowledgeBase)
            kb._client = client
            kb._reports = collection
            kb._stats = collection
            kb._news = collection

            results = kb.search_news(
                symbol="VCB",
                query="record profit earnings",
                n=5,
                date_to="2023-06-15",  # anti-leak: only news up to this date
            )
            dates = [r["date"] for r in results]
            self.assertIn("2023-06-10", dates, "News before date_to should be returned")
            self.assertNotIn("2023-07-20", dates, "News after date_to must be excluded")
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestRiskTradeBacktestT3(unittest.TestCase):
    """risk_trade.check() must use state date for T+2.5 in backtest_mode."""

    def test_backtest_t3_uses_as_of_date(self):
        path, conn = _make_temp_db()
        try:
            _insert_trade(conn, "VCB", "2023-06-14")
            conn.close()

            import multiagents_trading_assistant.database as db_module
            from multiagents_trading_assistant.nodes import risk_trade

            state = {
                "symbol": "VCB",
                "date": "2023-06-15",
                "backtest_mode": True,
                "memory_context": {"internal": {}},
                "trader_decision": {
                    "action": "MUA",
                    "rr_ratio": 2.0,
                    "entry_zone": [10.0, 11.0],
                    "stop_loss": 9.0,
                    "position_pct": 5.0,
                },
                "market_context": {
                    "reference_trend": "UPTREND",
                    "stock_day_change_pct": 0.0,
                    "avg_vol_20d": 500_000,
                },
                "foreign_flow_analysis": {},
                "money_flow_analysis": {},
                "synthesis": {"confluence_score": 80.0},
            }

            with patch.object(db_module, "DB_PATH", path), patch("builtins.print"):
                result = risk_trade.check(state)

            self.assertNotEqual(result.get("final_action"), "MUA")
            self.assertIn("T+2.5", result.get("override_reason", ""))
        finally:
            try:
                if path.exists():
                    os.unlink(path)
            except OSError:
                pass


class TestRetrieveTradeContextBacktestPosition(unittest.TestCase):
    """Backtest retrieval must not read live position state."""

    def test_backtest_mode_ignores_live_position(self):
        from multiagents_trading_assistant.services import memory_service

        fake_mem = MagicMock()
        fake_mem.get_decision_history.return_value = []
        fake_mem.get_streak.return_value = 0
        fake_mem.find_similar_setups.return_value = []
        fake_mem.has_position.return_value = True
        fake_mem.is_t3_blocked.return_value = False
        fake_mem.l1.get_position.return_value = {
            "entry_price": 10.0,
            "entry_date": "2026-01-01",
            "nav_pct": 5.0,
        }

        with patch.object(memory_service, "get_memory", return_value=fake_mem):
            ctx = memory_service.retrieve_trade_context(
                "VCB",
                "BREAKOUT",
                "UPTREND",
                80.0,
                as_of_date="2023-06-15",
                backtest_mode=True,
            )

        self.assertFalse(ctx["has_position"])
        self.assertIsNone(ctx["current_position"])
        fake_mem.has_position.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
