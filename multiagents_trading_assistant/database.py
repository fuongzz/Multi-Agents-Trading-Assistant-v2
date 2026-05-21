"""
database.py — SQLite interface cho AI Trading Assistant.

7 bảng chính:
  - positions          : vị thế đang giữ
  - decisions          : lịch sử AI ra quyết định
  - trades             : lịch sử MUA/BÁN thực tế (dùng cho T+2.5 check)
  - news_history       : lịch sử tin tức + sentiment
  - news_outcomes      : kết quả giá sau T+1/3/5/20 của mỗi tin
  - source_credibility : thống kê độ tin cậy theo nguồn báo
  - portfolio_config   : cấu hình danh mục (NAV, thresholds)
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Đường dẫn database — nằm cùng thư mục với file này
DB_PATH = Path(__file__).parent / "trading_assistant.db"


def get_connection() -> sqlite3.Connection:
    """Tạo connection với row_factory để trả về dict thay vì tuple."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # truy cập cột bằng tên: row["symbol"]
    conn.execute("PRAGMA journal_mode=WAL")  # an toàn hơn khi multi-process
    return conn


def init_db() -> None:
    """
    Khởi tạo database — tạo bảng nếu chưa tồn tại.
    Gọi 1 lần khi startup.
    """
    with get_connection() as conn:
        conn.executescript("""
            -- Vị thế đang giữ
            CREATE TABLE IF NOT EXISTS positions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT    NOT NULL UNIQUE,   -- mã CK, mỗi mã chỉ 1 dòng
                exchange    TEXT    NOT NULL DEFAULT 'HOSE',
                entry_price REAL    NOT NULL,
                quantity    INTEGER NOT NULL DEFAULT 0,
                entry_date  TEXT    NOT NULL,          -- YYYY-MM-DD
                strategy    TEXT,                      -- Breakout / MA_Pullback / ...
                sl          REAL,                      -- stop loss
                tp          REAL,                      -- take profit
                nav_pct     REAL,                      -- % NAV đã phân bổ
                created_at  TEXT    DEFAULT (datetime('now','localtime'))
            );

            -- Lịch sử quyết định của AI (mỗi lần pipeline chạy = 1 dòng)
            CREATE TABLE IF NOT EXISTS decisions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT    NOT NULL,
                date            TEXT    NOT NULL,      -- YYYY-MM-DD
                action          TEXT    NOT NULL,      -- MUA / BÁN / CHỜ
                final_action    TEXT    NOT NULL,      -- sau khi Risk Manager override
                strategy        TEXT,
                quality_score   REAL,                  -- AI chấm 1-10
                confidence      TEXT,                  -- THẤP / TRUNG_BÌNH / CAO / RẤT_CAO
                entry           REAL,
                sl              REAL,
                tp              REAL,
                nav_pct         REAL,
                override_reason TEXT,                  -- NULL nếu không override
                full_output     TEXT,                  -- JSON toàn bộ pipeline output
                created_at      TEXT DEFAULT (datetime('now','localtime'))
            );

            -- Lịch sử MUA/BÁN thực tế (dùng cho T+2.5 check)
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT    NOT NULL,
                action      TEXT    NOT NULL,          -- MUA / BÁN
                price       REAL    NOT NULL,
                quantity    INTEGER NOT NULL,
                trade_date  TEXT    NOT NULL,          -- YYYY-MM-DD
                strategy    TEXT,
                note        TEXT,
                created_at  TEXT DEFAULT (datetime('now','localtime'))
            );

            -- Index để query nhanh theo symbol + date
            CREATE INDEX IF NOT EXISTS idx_decisions_symbol_date
                ON decisions(symbol, date);

            CREATE INDEX IF NOT EXISTS idx_trades_symbol_date
                ON trades(symbol, trade_date);

            -- Lịch sử tin tức + sentiment lúc đăng
            CREATE TABLE IF NOT EXISTS news_history (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                date              TEXT    NOT NULL,          -- YYYY-MM-DD ngày lưu
                source            TEXT    NOT NULL,          -- CafeF / VnExpress / ...
                url               TEXT    UNIQUE,            -- tránh lưu trùng
                symbol            TEXT    NOT NULL,          -- mã CK liên quan
                headline          TEXT    NOT NULL,
                content           TEXT,                      -- tóm tắt / đoạn đầu bài
                sentiment         TEXT    DEFAULT 'NEUTRAL', -- POSITIVE / NEGATIVE / NEUTRAL
                price_at_publish  REAL,                      -- giá cổ phiếu lúc tin đăng
                credibility_score REAL,                      -- NULL → cập nhật sau khi có outcome
                is_suspicious     INTEGER DEFAULT 0,         -- 0=False, 1=True
                created_at        TEXT    DEFAULT (datetime('now','localtime'))
            );

            -- Kết quả giá sau T+1/3/5/20 phiên của 1 tin
            CREATE TABLE IF NOT EXISTS news_outcomes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                news_id    INTEGER NOT NULL UNIQUE REFERENCES news_history(id) ON DELETE CASCADE,
                price_t1   REAL,   -- giá đóng cửa T+1 phiên
                price_t3   REAL,
                price_t5   REAL,
                price_t20  REAL,
                return_t1  REAL,   -- % thay đổi so với price_at_publish
                return_t3  REAL,
                return_t5  REAL,
                return_t20 REAL,
                outcome    TEXT,   -- CORRECT / INCORRECT / TRAP
                updated_at TEXT    DEFAULT (datetime('now','localtime'))
            );

            -- Thống kê độ tin cậy theo nguồn báo
            CREATE TABLE IF NOT EXISTS source_credibility (
                source            TEXT PRIMARY KEY,
                total_news        INTEGER DEFAULT 0,
                correct_t5        INTEGER DEFAULT 0,   -- tin positive → giá tăng T+5
                trap_count        INTEGER DEFAULT 0,   -- giá tăng T+1 nhưng giảm T+5
                credibility_score REAL    DEFAULT 0.5, -- correct_t5 / total_news
                trap_rate         REAL    DEFAULT 0.0, -- trap_count / total_news
                last_updated      TEXT    DEFAULT (datetime('now','localtime'))
            );

            -- Index cho news_history
            CREATE INDEX IF NOT EXISTS idx_news_symbol_date
                ON news_history(symbol, date);

            CREATE INDEX IF NOT EXISTS idx_news_source
                ON news_history(source);

            -- Index cho news_outcomes
            CREATE INDEX IF NOT EXISTS idx_outcomes_news_id
                ON news_outcomes(news_id);

            -- Cấu hình danh mục (NAV, thresholds)
            CREATE TABLE IF NOT EXISTS portfolio_config (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now','localtime'))
            );
        """)
    # Mở rộng bảng positions — thêm cột portfolio management
    _alter_columns = [
        ("positions", "peak_price",   "REAL"),
        ("positions", "last_checked", "TEXT"),
        ("positions", "setup_type",   "TEXT"),
        ("positions", "signal_close", "REAL"),
        ("positions", "regime_at_entry", "TEXT"),
        ("trades",    "exit_price",   "REAL"),
        ("trades",    "exit_date",    "TEXT"),
        ("trades",    "exit_reason",  "TEXT"),
        ("trades",    "realized_pnl", "REAL"),
        ("trades",    "realized_rr",  "REAL"),
        ("trades",    "pnl_pct",       "REAL"),
        ("trades",    "holding_bars",  "INTEGER"),
        ("trades",    "loss_category", "TEXT"),
        ("trades",    "regime",        "TEXT"),
    ]
    with get_connection() as conn:
        for table, col, col_type in _alter_columns:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass  # column already exists

        # Seed portfolio_config nếu chưa có
        conn.execute("""
            INSERT OR IGNORE INTO portfolio_config (key, value)
            VALUES ('total_nav_vnd', '1000000000')
        """)
        conn.execute("""
            INSERT OR IGNORE INTO portfolio_config (key, value)
            VALUES ('momentum_drawdown_pct', '3.0')
        """)

    print(f"[database] DB ready: {DB_PATH}")


# ──────────────────────────────────────────────
# POSITIONS — vị thế đang giữ
# ──────────────────────────────────────────────

def has_position(symbol: str) -> bool:
    """
    Kiểm tra có đang giữ cổ phiếu này không.
    Risk Manager dùng để ngăn short selling.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id FROM positions WHERE symbol = ?", (symbol,)
        ).fetchone()
        return row is not None


def get_position(symbol: str) -> dict | None:
    """Lấy thông tin vị thế hiện tại của 1 mã. None nếu không có."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM positions WHERE symbol = ?", (symbol,)
        ).fetchone()
        return dict(row) if row else None


def get_all_positions() -> list[dict]:
    """Lấy toàn bộ vị thế đang giữ — dùng cho dashboard."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM positions ORDER BY entry_date DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def add_position(
    symbol: str,
    exchange: str,
    entry_price: float,
    quantity: int,
    entry_date: str,
    strategy: str = None,
    sl: float = None,
    tp: float = None,
    nav_pct: float = None,
) -> None:
    """Thêm vị thế mới sau khi MUA."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO positions
                (symbol, exchange, entry_price, quantity, entry_date, strategy, sl, tp, nav_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                entry_price = excluded.entry_price,
                quantity    = excluded.quantity,
                entry_date  = excluded.entry_date,
                strategy    = excluded.strategy,
                sl          = excluded.sl,
                tp          = excluded.tp,
                nav_pct     = excluded.nav_pct
        """, (symbol, exchange, entry_price, quantity, entry_date, strategy, sl, tp, nav_pct))


def remove_position(symbol: str) -> None:
    """Xóa vị thế sau khi BÁN hết."""
    with get_connection() as conn:
        conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))


def update_position_peak(symbol: str, peak_price: float) -> None:
    """Cập nhật đỉnh giá của vị thế — dùng để tính momentum drawdown."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE positions SET peak_price = ?, last_checked = date('now','localtime') WHERE symbol = ?",
            (peak_price, symbol),
        )


def update_position_sl(symbol: str, new_sl: float) -> None:
    """Nâng trailing SL của vị thế — gọi khi session_monitor phát hiện profit target đạt."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE positions SET sl = ?, last_checked = date('now','localtime') WHERE symbol = ?",
            (new_sl, symbol),
        )


def close_position(
    symbol: str,
    exit_price: float,
    exit_date: str,
    exit_reason: str,
    entry_price: float,
    quantity: int,
    strategy: str = None,
) -> None:
    """
    Đóng vị thế: ghi BÁN vào trades với các trường exit, rồi xóa khỏi positions.
    Đồng thời classify_loss() và lưu loss_category cho closed-loop learning.

    exit_reason: SL_HIT / TP_HIT / MOMENTUM_LOSS / MANUAL
    """
    risk = entry_price - (entry_price * 0.05)  # fallback nếu không có sl
    pnl = (exit_price - entry_price) * quantity
    rr = (exit_price - entry_price) / max(entry_price - risk, 1) if entry_price > risk else None
    pnl_pct = (exit_price - entry_price) / entry_price * 100.0 if entry_price > 0 else 0.0

    # Pull setup_type/signal_close/regime/entry_date from positions before delete
    pos = get_position(symbol) or {}
    setup_type = pos.get("setup_type") or strategy or ""
    signal_close = pos.get("signal_close")
    regime = pos.get("regime_at_entry")
    holding_bars = _trading_days_between(pos.get("entry_date"), exit_date)

    loss_category = _classify_close(
        pnl_pct=pnl_pct,
        setup_type=setup_type,
        exit_reason=exit_reason,
        holding_bars=holding_bars,
        signal_close=signal_close,
        entry_price=entry_price,
    )

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO trades
                (symbol, action, price, quantity, trade_date, strategy, note,
                 exit_price, exit_date, exit_reason, realized_pnl, realized_rr,
                 pnl_pct, holding_bars, loss_category, regime)
            VALUES (?, 'BÁN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (symbol, exit_price, quantity, exit_date, strategy,
              f"Auto exit: {exit_reason}",
              exit_price, exit_date, exit_reason, pnl, rr,
              pnl_pct, holding_bars, loss_category, regime))
        conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))


def _trading_days_between(entry_date: str | None, exit_date: str | None) -> int:
    """Approximate trading bars between two YYYY-MM-DD dates (weekdays only)."""
    if not entry_date or not exit_date:
        return 0
    try:
        d1 = datetime.strptime(entry_date, "%Y-%m-%d").date()
        d2 = datetime.strptime(exit_date, "%Y-%m-%d").date()
    except ValueError:
        return 0
    if d2 <= d1:
        return 0
    days = 0
    cur = d1
    while cur < d2:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


def _classify_close(
    *,
    pnl_pct: float,
    setup_type: str,
    exit_reason: str,
    holding_bars: int,
    signal_close: float | None,
    entry_price: float,
) -> str:
    """Lazy-import classify_loss to avoid circular deps. Returns LossCategory.value."""
    if pnl_pct >= 0:
        return "WINNER"
    try:
        from multiagents_trading_assistant.agentic.post_trade_review import classify_loss
        cat = classify_loss({
            "pnl_pct": pnl_pct,
            "setup_type": setup_type,
            "exit_reason": exit_reason,
            "holding_bars": holding_bars,
            "signal_close_price": signal_close,
            "entry_price": entry_price,
        })
        return cat.value
    except Exception:
        return "UNCLASSIFIED"


def reclassify_closed_losses(
    days: int,
    ohlcv_map: dict | None = None,
    vnindex_df=None,
) -> int:
    """Re-run classify_loss on closed losses with full OHLCV+VNI context.

    close_position() classifies at exit time without future bars or market slice,
    so REGIME_SHIFT and EXIT_TOO_EARLY can never trigger. Bob has this data —
    re-classify here to upgrade UNCLASSIFIED rows when better evidence is available.

    Returns count of rows actually updated (loss_category changed).
    """
    from multiagents_trading_assistant.agentic.post_trade_review import classify_loss

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    updated = 0
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id, symbol, exit_date, exit_reason, pnl_pct, holding_bars,
                   loss_category, strategy, regime, entry_price = price AS entry_price
            FROM trades
            WHERE action='BÁN' AND realized_pnl < 0 AND trade_date >= ?
        """, (cutoff,)).fetchall()

        for r in rows:
            row = dict(r)
            ohlcv_after = None
            market_after = None
            sym_df = (ohlcv_map or {}).get(row["symbol"])
            if sym_df is not None and len(sym_df) and "date" in sym_df.columns:
                import pandas as pd
                ex = pd.to_datetime(row["exit_date"])
                tail = sym_df[sym_df["date"] > ex]
                if not tail.empty:
                    ohlcv_after = tail.head(20)
            if vnindex_df is not None and len(vnindex_df) and "date" in vnindex_df.columns:
                import pandas as pd
                # Approximation: use last `holding_bars` of vni leading up to exit
                ex = pd.to_datetime(row["exit_date"])
                slice_ = vnindex_df[vnindex_df["date"] <= ex].tail(max(int(row.get("holding_bars") or 5), 5))
                if not slice_.empty:
                    market_after = slice_

            new_cat = classify_loss(
                {
                    "pnl_pct": row.get("pnl_pct") or 0.0,
                    "setup_type": row.get("strategy") or "",
                    "exit_reason": row.get("exit_reason") or "",
                    "holding_bars": row.get("holding_bars") or 0,
                },
                ohlcv_after=ohlcv_after,
                market_after=market_after,
            ).value

            if new_cat and new_cat != row.get("loss_category"):
                conn.execute(
                    "UPDATE trades SET loss_category=? WHERE id=?",
                    (new_cat, row["id"]),
                )
                updated += 1
    return updated


def aggregate_loss_patterns(
    days: int = 90,
    min_count: int = 2,
) -> list[dict]:
    """Group closed losses by (setup × loss_category × regime) over last N days.

    Returns sorted list of patterns Bob should pay attention to:
        [{setup, regime, loss_category, count, avg_pnl_pct, total_pnl}, ...]
    Sorted by total_pnl ascending (worst first).
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                COALESCE(strategy, '?')      AS setup,
                COALESCE(regime, 'UNKNOWN')  AS regime,
                COALESCE(loss_category, 'UNCLASSIFIED') AS loss_category,
                COUNT(*)                     AS count,
                AVG(pnl_pct)                 AS avg_pnl_pct,
                SUM(realized_pnl)            AS total_pnl
            FROM trades
            WHERE action = 'BÁN'
              AND realized_pnl < 0
              AND trade_date >= ?
            GROUP BY setup, regime, loss_category
            HAVING count >= ?
            ORDER BY total_pnl ASC
        """, (cutoff, min_count)).fetchall()
        return [dict(r) for r in rows]


def get_closed_trades(limit: int = 100) -> list[dict]:
    """Lấy lịch sử giao dịch đã đóng (BÁN rows) — dùng cho thống kê hiệu suất."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM trades
            WHERE action = 'BÁN'
            ORDER BY trade_date DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]


# ──────────────────────────────────────────────
# PORTFOLIO CONFIG
# ──────────────────────────────────────────────

def get_portfolio_config(key: str, default=None) -> str | None:
    """Đọc một cấu hình danh mục theo key."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT value FROM portfolio_config WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default


def set_portfolio_config(key: str, value: str) -> None:
    """Ghi/cập nhật một cấu hình danh mục."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO portfolio_config (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_at = datetime('now','localtime')
        """, (key, value))


# ──────────────────────────────────────────────
# TRADES — lịch sử giao dịch (T+2.5)
# ──────────────────────────────────────────────

def record_trade(
    symbol: str,
    action: str,
    price: float,
    quantity: int,
    trade_date: str,
    strategy: str = None,
    note: str = None,
) -> None:
    """Ghi nhận 1 giao dịch MUA/BÁN vào lịch sử."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO trades (symbol, action, price, quantity, trade_date, strategy, note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, action, price, quantity, trade_date, strategy, note))


def get_buys_last_n_days(symbol: str, n: int = 3, as_of_date: str | None = None) -> list[dict]:
    """
    Lấy danh sách lệnh MUA của 1 mã trong n ngày gần nhất tính từ as_of_date.

    Args:
        as_of_date: Ngày tham chiếu (YYYY-MM-DD). None = dùng datetime.now() (live mode).
                    Trong backtest, truyền ngày đang evaluate để tránh data leak.
    """
    # Anti-leak: compute cutoff relative to as_of_date, not wall-clock now()
    reference = datetime.strptime(as_of_date, "%Y-%m-%d") if as_of_date else datetime.now()
    cutoff = (reference - timedelta(days=n)).strftime("%Y-%m-%d")
    upper  = as_of_date or "9999-12-31"   # live mode: no upper bound needed
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM trades
            WHERE symbol = ? AND action = 'MUA'
              AND trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date DESC
        """, (symbol, cutoff, upper)).fetchall()
        return [dict(r) for r in rows]


def get_trade_history(symbol: str = None, limit: int = 50) -> list[dict]:
    """Lấy lịch sử giao dịch — dùng cho dashboard."""
    with get_connection() as conn:
        if symbol:
            rows = conn.execute("""
                SELECT * FROM trades WHERE symbol = ?
                ORDER BY trade_date DESC LIMIT ?
            """, (symbol, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM trades ORDER BY trade_date DESC LIMIT ?
            """, (limit,)).fetchall()
        return [dict(r) for r in rows]


# ──────────────────────────────────────────────
# DECISIONS — lịch sử quyết định AI
# ──────────────────────────────────────────────

def save_decision(
    symbol: str,
    date: str,
    action: str,
    final_action: str,
    strategy: str = None,
    quality_score: float = None,
    confidence: str = None,
    entry: float = None,
    sl: float = None,
    tp: float = None,
    nav_pct: float = None,
    override_reason: str = None,
    full_output: dict = None,
) -> None:
    """Lưu kết quả pipeline cho 1 mã vào 1 ngày."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO decisions
                (symbol, date, action, final_action, strategy, quality_score,
                 confidence, entry, sl, tp, nav_pct, override_reason, full_output)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, date, action, final_action, strategy, quality_score,
            confidence, entry, sl, tp, nav_pct, override_reason,
            json.dumps(full_output, ensure_ascii=False) if full_output else None,
        ))


def get_decisions(
    symbol: str = None,
    date: str = None,
    limit: int = 50,
    as_of_date: str | None = None,
) -> list[dict]:
    """Lấy lịch sử quyết định — dùng cho dashboard và backtest.

    Args:
        as_of_date: Nếu set, chỉ trả về decisions có date <= as_of_date.
                    Dùng trong backtest để tránh leak quyết định tương lai.
    """
    # Anti-leak: khi backtest, loại bỏ mọi decision sau ngày đang evaluate
    aod_clause = "AND date <= ?" if as_of_date else ""
    aod_param  = (as_of_date,) if as_of_date else ()

    with get_connection() as conn:
        if symbol and date:
            rows = conn.execute(f"""
                SELECT * FROM decisions
                WHERE symbol = ? AND date = ? {aod_clause}
                ORDER BY created_at DESC
            """, (symbol, date) + aod_param).fetchall()
        elif symbol:
            rows = conn.execute(f"""
                SELECT * FROM decisions
                WHERE symbol = ? {aod_clause}
                ORDER BY date DESC LIMIT ?
            """, (symbol,) + aod_param + (limit,)).fetchall()
        elif date:
            rows = conn.execute(f"""
                SELECT * FROM decisions
                WHERE date = ? {aod_clause}
                ORDER BY created_at DESC
            """, (date,) + aod_param).fetchall()
        else:
            rows = conn.execute(f"""
                SELECT * FROM decisions
                WHERE 1=1 {aod_clause}
                ORDER BY date DESC, created_at DESC LIMIT ?
            """, aod_param + (limit,)).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            if d.get("full_output"):
                try:
                    d["full_output"] = json.loads(d["full_output"])
                except Exception:
                    pass
            result.append(d)
        return result


# ──────────────────────────────────────────────
# NEWS HISTORY — lịch sử tin tức
# ──────────────────────────────────────────────

def save_news(news_item: dict) -> int | None:
    """
    Lưu 1 bài báo vào news_history.

    Args:
        news_item: dict với các key: date, source, url, symbol, headline,
                   content, sentiment, price_at_publish, is_suspicious

    Returns:
        id của dòng vừa insert, hoặc None nếu URL đã tồn tại.
    """
    with get_connection() as conn:
        try:
            cursor = conn.execute("""
                INSERT INTO news_history
                    (date, source, url, symbol, headline, content,
                     sentiment, price_at_publish, is_suspicious)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                news_item.get("date"),
                news_item.get("source"),
                news_item.get("url"),
                news_item.get("symbol"),
                news_item.get("headline"),
                news_item.get("content"),
                news_item.get("sentiment", "NEUTRAL"),
                news_item.get("price_at_publish"),
                int(news_item.get("is_suspicious", False)),
            ))
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # URL đã tồn tại (UNIQUE constraint) — bỏ qua
            return None


def update_news_outcome(news_id: int, prices_dict: dict) -> None:
    """
    Cập nhật kết quả giá sau T+1/3/5/20 cho 1 tin.

    Args:
        news_id: id trong bảng news_history
        prices_dict: {
            "price_at_publish": float,
            "price_t1": float, "price_t3": float,
            "price_t5": float, "price_t20": float
        }

    Logic outcome:
        - CORRECT  : sentiment POSITIVE + return_t5 > 0
        - TRAP     : return_t1 > 2% nhưng return_t5 < 0
        - INCORRECT: sentiment POSITIVE + return_t5 <= 0 (mà không phải TRAP)
    """
    base = prices_dict.get("price_at_publish")

    def pct(price):
        if base and price:
            return round((price - base) / base * 100, 2)
        return None

    p1  = prices_dict.get("price_t1")
    p3  = prices_dict.get("price_t3")
    p5  = prices_dict.get("price_t5")
    p20 = prices_dict.get("price_t20")
    r1  = pct(p1)
    r3  = pct(p3)
    r5  = pct(p5)
    r20 = pct(p20)

    # Xác định outcome
    outcome = None
    if r1 is not None and r5 is not None:
        if r1 > 2 and r5 < 0:
            outcome = "TRAP"
        elif r5 > 0:
            outcome = "CORRECT"
        else:
            outcome = "INCORRECT"

    with get_connection() as conn:
        # Upsert vào news_outcomes
        conn.execute("""
            INSERT INTO news_outcomes
                (news_id, price_t1, price_t3, price_t5, price_t20,
                 return_t1, return_t3, return_t5, return_t20, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(news_id) DO UPDATE SET
                price_t1  = excluded.price_t1,
                price_t3  = excluded.price_t3,
                price_t5  = excluded.price_t5,
                price_t20 = excluded.price_t20,
                return_t1 = excluded.return_t1,
                return_t3 = excluded.return_t3,
                return_t5 = excluded.return_t5,
                return_t20= excluded.return_t20,
                outcome   = excluded.outcome,
                updated_at= datetime('now','localtime')
        """, (news_id, p1, p3, p5, p20, r1, r3, r5, r20, outcome))

        # Cập nhật credibility_score trên news_history
        if outcome:
            conn.execute("""
                UPDATE news_history
                SET credibility_score = (
                    SELECT CAST(correct_t5 AS REAL) / NULLIF(total_news, 0)
                    FROM source_credibility sc
                    WHERE sc.source = (SELECT source FROM news_history WHERE id = ?)
                )
                WHERE id = ?
            """, (news_id, news_id))


def get_source_credibility(source: str) -> float:
    """
    Lấy credibility_score của 1 nguồn báo.
    Trả về 0.5 (neutral) nếu chưa có dữ liệu.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT credibility_score FROM source_credibility WHERE source = ?",
            (source,)
        ).fetchone()
        if row and row["credibility_score"] is not None:
            return float(row["credibility_score"])
        return 0.5


def update_source_stats(source: str) -> None:
    """
    Tính lại và cập nhật thống kê tổng hợp cho 1 nguồn báo.
    Gọi sau mỗi lần update_news_outcome().
    """
    with get_connection() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                                        AS total,
                SUM(CASE WHEN no.outcome = 'CORRECT' THEN 1 ELSE 0 END) AS correct,
                SUM(CASE WHEN no.outcome = 'TRAP'    THEN 1 ELSE 0 END) AS traps
            FROM news_history nh
            LEFT JOIN news_outcomes no ON nh.id = no.news_id
            WHERE nh.source = ? AND no.outcome IS NOT NULL
        """, (source,)).fetchone()

        if not row or row["total"] == 0:
            return

        total   = row["total"]
        correct = row["correct"] or 0
        traps   = row["traps"]   or 0
        cred    = round(correct / total, 4)
        trap_r  = round(traps   / total, 4)

        conn.execute("""
            INSERT INTO source_credibility
                (source, total_news, correct_t5, trap_count,
                 credibility_score, trap_rate, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now','localtime'))
            ON CONFLICT(source) DO UPDATE SET
                total_news        = excluded.total_news,
                correct_t5        = excluded.correct_t5,
                trap_count        = excluded.trap_count,
                credibility_score = excluded.credibility_score,
                trap_rate         = excluded.trap_rate,
                last_updated      = excluded.last_updated
        """, (source, total, correct, traps, cred, trap_r))


def get_suspicious_news(symbol: str, date: str) -> list[dict]:
    """
    Lấy danh sách tin nghi ngờ (is_suspicious=1 HOẶC source có trap_rate > 30%)
    cho 1 mã trong ngày cụ thể.

    Args:
        symbol: mã CK
        date  : YYYY-MM-DD

    Returns:
        list[dict] — các tin đáng ngờ, kèm credibility_score của nguồn
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT nh.*,
                   COALESCE(sc.trap_rate, 0.0)         AS src_trap_rate,
                   COALESCE(sc.credibility_score, 0.5) AS src_credibility
            FROM news_history nh
            LEFT JOIN source_credibility sc ON nh.source = sc.source
            WHERE nh.symbol = ?
              AND nh.date   = ?
              AND (nh.is_suspicious = 1 OR COALESCE(sc.trap_rate, 0.0) > 0.3)
            ORDER BY nh.created_at DESC
        """, (symbol, date)).fetchall()
        return [dict(r) for r in rows]


# ──────────────────────────────────────────────
# CLEANUP — xóa dữ liệu cũ định kỳ
# ──────────────────────────────────────────────

def cleanup_old_data(
    news_keep_days: int = 90,
    decisions_keep_days: int = 180,
) -> dict:
    """Xóa dữ liệu cũ khỏi SQLite + cache JSON + ChromaDB news_articles.

    Args:
        news_keep_days:      Giữ news_history N ngày gần nhất (default 90)
        decisions_keep_days: Giữ decisions N ngày gần nhất (default 180)

    Returns:
        dict thống kê: {"news_deleted", "decisions_deleted", "cache_deleted", "chroma_deleted"}
    """
    import shutil
    from pathlib import Path

    stats = {"news_deleted": 0, "decisions_deleted": 0, "cache_deleted": 0, "chroma_deleted": 0}

    news_cutoff      = (datetime.now() - timedelta(days=news_keep_days)).strftime("%Y-%m-%d")
    decision_cutoff  = (datetime.now() - timedelta(days=decisions_keep_days)).strftime("%Y-%m-%d")
    cache_cutoff     = datetime.now().timestamp() - 7 * 86400  # 7 ngày tính theo mtime

    # ── SQLite: news_history ──
    try:
        with get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM news_history WHERE date < ?", (news_cutoff,)
            )
            stats["news_deleted"] = cur.rowcount
    except Exception as e:
        print(f"[database] cleanup news_history fail: {e}")

    # ── SQLite: decisions ──
    try:
        with get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM decisions WHERE date < ?", (decision_cutoff,)
            )
            stats["decisions_deleted"] = cur.rowcount
    except Exception as e:
        print(f"[database] cleanup decisions fail: {e}")

    # ── Cache JSON: xóa file OHLCV/news cũ hơn 7 ngày ──
    cache_dir = Path(__file__).parent / "cache"
    try:
        for f in cache_dir.glob("*.json"):
            if f.stat().st_mtime < cache_cutoff:
                f.unlink(missing_ok=True)
                stats["cache_deleted"] += 1
    except Exception as e:
        print(f"[database] cleanup cache fail: {e}")

    # ── ChromaDB: xóa news_articles cũ ──
    try:
        from multiagents_trading_assistant.memory.knowledge_base import get_knowledge_base
        kb = get_knowledge_base()
        if kb._news is not None:
            old = kb._news.get(where={"date": {"$lt": news_cutoff}})
            ids_to_del = old.get("ids", [])
            if ids_to_del:
                kb._news.delete(ids=ids_to_del)
                stats["chroma_deleted"] = len(ids_to_del)
    except Exception as e:
        print(f"[database] cleanup ChromaDB fail: {e}")

    print(
        f"[database] Cleanup xong — "
        f"news={stats['news_deleted']}, decisions={stats['decisions_deleted']}, "
        f"cache={stats['cache_deleted']}, chroma={stats['chroma_deleted']}"
    )
    return stats


# ──────────────────────────────────────────────
# Chạy trực tiếp để test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    init_db()

    # Test thêm position
    add_position("VNM", "HOSE", 72500, 1000, "2026-04-12", "MA_Pullback", 69000, 82000, 5.0)
    print("Có VNM không?", has_position("VNM"))     # True
    print("Có HPG không?", has_position("HPG"))     # False
    print("Vị thế VNM:", get_position("VNM"))

    # Test ghi trade
    record_trade("VNM", "MUA", 72500, 1000, "2026-04-12", "MA_Pullback")
    print("Mua VNM trong 3 ngày:", get_buys_last_n_days("VNM", 3))

    # Test lưu decision
    save_decision("VNM", "2026-04-12", "MUA", "MUA", "MA_Pullback", 8.5, "CAO",
                  72500, 69000, 82000, 5.0, None, {"test": True})
    print("Decisions:", get_decisions("VNM"))

    print("\n[database] Tất cả test OK!")
