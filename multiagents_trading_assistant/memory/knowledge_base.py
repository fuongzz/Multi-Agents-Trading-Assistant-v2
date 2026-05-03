"""knowledge_base.py — External knowledge base (ChromaDB).

3 collections độc lập với trading_decisions:
  vietstock_reports — BCTC tóm tắt, giải trình LN, nghị quyết ĐHCĐ
                      Chunk theo Section (không theo char count)
                      Metadata: {symbol, year, quarter, doc_type, section}

  vietstock_stats   — Insight thống kê lịch sử 10 năm (seasonality, xác suất)
                      Format cô đọng: "Tháng 1: 7/10 năm tăng (70%), +4.5%"
                      Metadata: {symbol, stat_type, period}

  news_articles     — Tin CafeF/VnExpress crawl tự động hàng ngày
                      ID deterministic = source + url_hash → upsert không trùng
                      Metadata: {symbol, date, source, sentiment}

Search: filter-first theo symbol (hard-filter), rồi vector search trong đó.
Ingest Vietstock: gọi thủ công qua scripts/ingest_vietstock.py — tách khỏi pipeline.
Ingest News: tự động từ sentiment_agent.py sau mỗi lần crawl.
"""

import hashlib
from pathlib import Path
from typing import Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False

_CHROMA_PATH = Path(__file__).parent.parent / "cache" / "chromadb"


class KnowledgeBase:
    """External knowledge base — 3 collections trong ChromaDB."""

    def __init__(self):
        self._client = None
        self._reports: Optional[object] = None   # vietstock_reports
        self._stats: Optional[object] = None     # vietstock_stats
        self._news: Optional[object] = None      # news_articles
        if _HAS_CHROMA:
            self._init()

    def _init(self) -> None:
        try:
            _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(_CHROMA_PATH),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._reports = self._client.get_or_create_collection(
                name="vietstock_reports",
                metadata={"description": "BCTC, nghị quyết, giải trình — Vietstock"},
            )
            self._stats = self._client.get_or_create_collection(
                name="vietstock_stats",
                metadata={"description": "Insight thống kê lịch sử 10 năm — Vietstock"},
            )
            self._news = self._client.get_or_create_collection(
                name="news_articles",
                metadata={"description": "Tin CafeF/VnExpress crawl hàng ngày"},
            )
            print(
                f"[knowledge_base] ChromaDB OK — "
                f"reports={self._reports.count()}, "
                f"stats={self._stats.count()}, "
                f"news={self._news.count()}"
            )
        except Exception as e:
            print(f"[knowledge_base] ChromaDB loi: {e} — KB disabled")
            self._client = None
            self._reports = None
            self._stats = None
            self._news = None

    @property
    def available(self) -> bool:
        return self._reports is not None and self._stats is not None

    # ──────────────────────────────────────────────
    # Internal helper — query an toàn với fallback
    # ──────────────────────────────────────────────

    def _safe_query(self, collection, query: str, n: int, where: dict) -> list[str]:
        """Query ChromaDB với retry khi n > filtered count.

        ChromaDB raise ValueError nếu n_results > số doc trong filter.
        Retry với n=1 nếu lần đầu fail.
        """
        if collection.count() == 0:
            return []
        for n_try in (n, 1):
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=n_try,
                    where=where,
                )
                docs = results.get("documents", [[]])[0]
                return [d for d in docs if d and d.strip()]
            except Exception:
                if n_try == 1:
                    return []
        return []

    # ──────────────────────────────────────────────
    # Search API
    # ──────────────────────────────────────────────

    def search_reports(
        self,
        symbol: str,
        query: str,
        n: int = 3,
        as_of_date: str | None = None,
        backtest_mode: bool = False,
    ) -> list[str]:
        """Tìm BCTC/nghị quyết liên quan đến symbol.

        Filter theo symbol trước (hard-filter) để tránh lôi báo cáo của mã khác.

        Args:
            as_of_date:    Nếu set cùng backtest_mode=True, chỉ trả về báo cáo có
                           published_at <= as_of_date. Báo cáo không có published_at
                           bị loại — không có timestamp = unsafe cho backtest.
            backtest_mode: True = chế độ backtest, bật strict date filtering.
        """
        if not self.available:
            return []
        try:
            if backtest_mode and as_of_date:
                # ChromaDB $lte unsupported on string dates → Python post-filter.
                # Fetch extra candidates by symbol, then filter by published_at in Python.
                # Documents without published_at are excluded — no timestamp = unsafe.
                print(
                    f"[knowledge_base] backtest_mode: filtering reports by published_at <= {as_of_date}"
                )
                total = self._reports.count()
                if total == 0:
                    return []
                n_fetch = min(n * 5 + 1, total)
                raw = self._reports.query(
                    query_texts=[query],
                    n_results=n_fetch,
                    where={"symbol": symbol},
                )
                docs  = raw.get("documents", [[]])[0]
                metas = raw.get("metadatas",  [[]])[0]
                result = []
                for doc, meta in zip(docs, metas):
                    if not doc or not doc.strip():
                        continue
                    pub = meta.get("published_at", "")
                    if not pub:
                        # No timestamp = unsafe for backtest — exclude
                        continue
                    if pub > as_of_date:
                        # Published after backtest evaluation date — exclude (anti-leak)
                        continue
                    result.append(doc.strip())
                    if len(result) >= n:
                        break
                return result
            else:
                return self._safe_query(self._reports, query, n, where={"symbol": symbol})
        except Exception as e:
            print(f"[knowledge_base] search_reports fail ({symbol}): {e}")
            return []

    def search_stats(
        self,
        symbol: str,
        query: str,
        n: int = 2,
        as_of_date: str | None = None,
        backtest_mode: bool = False,
    ) -> list[str]:
        """Tìm insight thống kê lịch sử (seasonality, xác suất tăng/giảm).

        vietstock_stats là dữ liệu lịch sử tổng hợp (10 năm) — không gắn thời điểm
        cụ thể → WHITELISTED cho backtest, không cần filter theo as_of_date.
        """
        if not self.available:
            return []
        try:
            # Stats are timeless historical summaries — safe for any backtest date
            return self._safe_query(self._stats, query, n, where={"symbol": symbol})
        except Exception as e:
            print(f"[knowledge_base] search_stats fail ({symbol}): {e}")
            return []

    # ──────────────────────────────────────────────
    # Ingest API — gọi từ scripts/ingest_vietstock.py
    # ──────────────────────────────────────────────

    def ingest_report(
        self,
        symbol: str,
        year: int,
        quarter: int,            # 0 = báo cáo năm
        doc_type: str,           # "BCTC" | "NGHI_QUYET" | "GIAI_TRINH"
        section: str,            # "KET_QUA_KD" | "KE_HOACH" | "RUI_RO" | "CO_TUC" | ...
        text: str,
        published_at: str | None = None,  # YYYY-MM-DD — ngày báo cáo công bố chính thức
    ) -> bool:
        """Nạp 1 chunk báo cáo vào vietstock_reports.

        doc_id deterministic → upsert an toàn, chạy lại không duplicate.

        Args:
            published_at: Ngày công bố chính thức (YYYY-MM-DD). Nếu None, báo cáo này
                          sẽ bị loại trong backtest_mode vì không có timestamp xác nhận.
                          Ví dụ: Q4/2022 thường công bố tháng 3/2023 → "2023-03-31".
        """
        if not self.available or not text.strip():
            return False
        doc_id = f"{symbol}_{year}_Q{quarter}_{doc_type}_{section}"
        metadata: dict = {
            "symbol":   symbol,
            "year":     year,
            "quarter":  quarter,
            "doc_type": doc_type,
            "section":  section,
        }
        if published_at:
            # Store publication date to enable anti-leak filtering in backtest
            metadata["published_at"] = published_at
        try:
            self._reports.upsert(
                ids=[doc_id],
                documents=[text.strip()],
                metadatas=[metadata],
            )
            return True
        except Exception as e:
            print(f"[knowledge_base] ingest_report fail ({doc_id}): {e}")
            return False

    def ingest_stat(
        self,
        symbol: str,
        stat_type: str,  # "SEASONALITY" | "VOLATILITY" | "TREND" | "CORRELATION"
        period: str,     # "monthly" | "quarterly" | "annual"
        text: str,       # Đã format cô đọng: "Tháng 1: 7/10 năm tăng (70%), +4.5%"
    ) -> bool:
        """Nạp 1 insight thống kê vào vietstock_stats.

        1 insight per (symbol, stat_type, period) — upsert sẽ overwrite khi re-ingest.
        """
        if not self.available or not text.strip():
            return False
        doc_id = f"{symbol}_{stat_type}_{period}"
        try:
            self._stats.upsert(
                ids=[doc_id],
                documents=[text.strip()],
                metadatas=[{
                    "symbol":    symbol,
                    "stat_type": stat_type,
                    "period":    period,
                }],
            )
            return True
        except Exception as e:
            print(f"[knowledge_base] ingest_stat fail ({doc_id}): {e}")
            return False

    def ingest_news_article(
        self,
        symbol:    str,
        date:      str,
        source:    str,
        url:       str,
        title:     str,
        summary:   str,
        sentiment: str = "NEUTRAL",
    ) -> bool:
        """Nạp 1 bài báo vào news_articles.

        ID = source + md5(url)[:8] → upsert an toàn, chạy lại không duplicate.
        Text embed = "Mã {symbol} ngày {date}: [{source}] {title}. {summary}"
        — embed cả symbol/date để vector search được tập trung hơn.
        """
        if self._news is None or not title.strip():
            return False

        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        doc_id   = f"{source}_{url_hash}"

        text = f"Ma {symbol} ngay {date}: [{source}] {title}. {summary}".strip()

        try:
            self._news.upsert(
                ids       = [doc_id],
                documents = [text],
                metadatas = [{
                    "symbol":    symbol,
                    "date":      date,
                    "source":    source,
                    "url":       url,
                    "sentiment": sentiment,
                }],
            )
            return True
        except Exception as e:
            print(f"[knowledge_base] ingest_news fail ({doc_id}): {e}")
            return False

    def search_news(
        self,
        symbol:    str,
        query:     str,
        n:         int = 5,
        date_from: str | None = None,
        date_to:   str | None = None,
    ) -> list[dict]:
        """Tìm tin tức liên quan theo vector search.

        Args:
            symbol:    Lọc cứng theo mã CK
            query:     Câu truy vấn (VD: "tin tiêu cực về nợ xấu")
            n:         Số kết quả tối đa
            date_from: Chỉ lấy tin từ ngày này trở đi (YYYY-MM-DD)
            date_to:   Chỉ lấy tin đến ngày này (YYYY-MM-DD).
                       Trong backtest, set date_to=as_of_date để tránh leak tin tương lai.

        Returns:
            list[dict] — mỗi phần tử gồm: text, source, date, url, sentiment, distance
        """
        if self._news is None:
            return []

        # ChromaDB $lte/$gte unsupported on string dates → Python post-filter.
        # Query by symbol only; filter date_from / date_to in Python after retrieval.
        has_date_filter = bool(date_from or date_to)
        try:
            total   = self._news.count()
            if total == 0:
                return []
            # Fetch extra results to compensate for Python-side date post-filtering
            n_fetch = min(n * 4, total) if has_date_filter else min(n, total)
            raw = self._news.query(
                query_texts = [query],
                n_results   = n_fetch,
                where       = {"symbol": symbol},
            )
            docs      = raw.get("documents", [[]])[0]
            metas     = raw.get("metadatas",  [[]])[0]
            distances = raw.get("distances",  [[]])[0]

            result = []
            for doc, meta, dist in zip(docs, metas, distances):
                if not doc or not doc.strip():
                    continue
                d = meta.get("date", "")
                if date_from and d < date_from:
                    continue
                if date_to and d > date_to:
                    # Anti-leak: exclude news published after the backtest evaluation date
                    continue
                result.append({
                    "text":      doc,
                    "source":    meta.get("source"),
                    "date":      d,
                    "url":       meta.get("url"),
                    "sentiment": meta.get("sentiment"),
                    "distance":  round(dist, 4),
                })
                if len(result) >= n:
                    break
            return result
        except Exception as e:
            print(f"[knowledge_base] search_news fail ({symbol}): {e}")
            return []

    def get_counts(self) -> dict:
        """Số documents trong mỗi collection — dùng để monitor."""
        return {
            "reports": self._reports.count() if self._reports else 0,
            "stats":   self._stats.count()   if self._stats   else 0,
            "news":    self._news.count()     if self._news    else 0,
        }


# ──────────────────────────────────────────────
# Singleton
# ──────────────────────────────────────────────

_kb: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Lấy hoặc tạo singleton KnowledgeBase."""
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
