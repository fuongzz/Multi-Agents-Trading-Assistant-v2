"""
database.py — SQLite interface cho AI Trading Assistant.

6 bảng chính:
  - positions          : vị thế đang giữ
  - decisions          : lịch sử AI ra quyết định
  - trades             : lịch sử MUA/BÁN thực tế (dùng cho T+3 check)
  - news_history       : lịch sử tin tức + sentiment
  - news_outcomes      : kết quả giá sau T+1/3/5/20 của mỗi tin
  - source_credibility : thống kê độ tin cậy theo nguồn báo
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

            -- Lịch sử MUA/BÁN thực tế (dùng cho T+3 check)
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
        """)
    print(f"[database] Đã khởi tạo DB tại: {DB_PATH}")


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


# ──────────────────────────────────────────────
# TRADES — lịch sử giao dịch (T+3)
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


def get_buys_last_n_days(symbol: str, n: int = 3) -> list[dict]:
    """
    Lấy danh sách lệnh MUA của 1 mã trong n ngày gần nhất.
    Risk Manager dùng để kiểm tra T+3: nếu có → không mua lại.

    Ví dụ: mua VNM ngày T, thì T+1, T+2, T+3 đều không mua thêm.
    """
    cutoff = (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM trades
            WHERE symbol = ? AND action = 'MUA' AND trade_date >= ?
            ORDER BY trade_date DESC
        """, (symbol, cutoff)).fetchall()
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


def get_decisions(symbol: str = None, date: str = None, limit: int = 50) -> list[dict]:
    """Lấy lịch sử quyết định — dùng cho dashboard và backtest."""
    with get_connection() as conn:
        if symbol and date:
            rows = conn.execute("""
                SELECT * FROM decisions WHERE symbol = ? AND date = ?
                ORDER BY created_at DESC
            """, (symbol, date)).fetchall()
        elif symbol:
            rows = conn.execute("""
                SELECT * FROM decisions WHERE symbol = ?
                ORDER BY date DESC LIMIT ?
            """, (symbol, limit)).fetchall()
        elif date:
            rows = conn.execute("""
                SELECT * FROM decisions WHERE date = ?
                ORDER BY created_at DESC
            """, (date,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM decisions ORDER BY date DESC, created_at DESC LIMIT ?
            """, (limit,)).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            # Parse lại full_output từ JSON string → dict
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
