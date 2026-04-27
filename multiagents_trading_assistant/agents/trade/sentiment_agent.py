"""sentiment_agent.py — Sentiment tin tức cho Trade pipeline.

Model: Haiku. Port từ _legacy/research/sentiment_agent.py (không đổi logic).
"""

import importlib.metadata  # noqa: F401
from datetime import datetime

from multiagents_trading_assistant.services.llm_service import run_agent_lite
from multiagents_trading_assistant.news_fetcher import get_stock_news
from multiagents_trading_assistant import database as db
from multiagents_trading_assistant.memory.knowledge_base import get_knowledge_base


_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích sentiment tin tức tài chính VN.

Quy tắc bắt buộc:
- Trả về JSON hợp lệ DUY NHẤT
- sentiment_score: 0-100 (0=rất tiêu cực, 50=trung tính, 100=rất tích cực)
- sentiment_label: TIÊU_CỰC (0-30) | TRUNG_TÍNH (31-60) | TÍCH_CỰC (61-80) | RẤT_TÍCH_CỰC (81-100)
- key_positive / key_negative: tối đa 3 điểm mỗi bên (tiếng Việt, ngắn gọn)
- Không có bài → score=50, label=TRUNG_TÍNH

Output schema:
{
  "sentiment_score": <int>,
  "sentiment_label": <str>,
  "news_count": <int>,
  "key_positive": [<str>, ...],
  "key_negative": [<str>, ...],
  "sentiment_summary": <str>
}"""


def analyze(symbol: str, date: str | None = None, days: int = 3) -> dict:
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    print(f"[sentiment_agent] {symbol} ({date})")

    articles = get_stock_news(symbol, days=days, max_articles=20)
    _auto_ingest_news(symbol, date, articles)   # lưu vào SQLite + ChromaDB

    if not articles:
        return _neutral_result(0)

    news_text = _format_articles(articles)
    prompt = f"""Phân tích sentiment mã {symbol} ngày {date}.

Tổng: {len(articles)} bài (CafeF + VnExpress, {days} ngày)

=== Bài báo ===
{news_text}

Trả JSON theo schema."""

    try:
        result = run_agent_lite(prompt=prompt, system=_SYSTEM_PROMPT)
        result["news_count"] = len(articles)
        print(f"[sentiment_agent] {symbol} — score={result.get('sentiment_score')}, {len(articles)} bài")
        return result
    except Exception as e:
        print(f"[sentiment_agent] LLM error: {e}")
        return _neutral_result(len(articles))


def _format_articles(articles: list[dict]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        src  = a.get("source", "?")
        ttl  = a.get("title", "")[:120]
        smy  = a.get("summary", "")[:150]
        pub  = a.get("published_at", "")
        line = f"{i}. [{src}] {ttl}"
        if pub: line += f" ({pub})"
        if smy: line += f"\n   → {smy}"
        lines.append(line)
    return "\n".join(lines)


def _auto_ingest_news(symbol: str, date: str, articles: list[dict]) -> None:
    """Lưu bài báo vừa crawl vào 2 nơi song song — không raise exception.

    Tầng 1 — SQLite (database.news_history):
        Lưu có cấu trúc để tra cứu nhanh theo symbol/date/source.
        Dùng UNIQUE(url) nên chạy lại không bị trùng.

    Tầng 2 — ChromaDB (knowledge_base.news_articles):
        Vector hóa headline + summary để tìm kiếm ngữ nghĩa sau này.
        Dùng upsert với ID = source + url_hash nên chạy lại không bị trùng.
    """
    if not articles:
        return

    kb          = get_knowledge_base()
    saved_sql   = 0
    saved_chroma = 0

    for article in articles:
        url     = article.get("url", "")
        title   = article.get("title", "")
        summary = article.get("summary", "")
        source  = article.get("source", "unknown")

        if not url or not title:
            continue

        # ── Tầng 1: SQLite ──
        try:
            news_id = db.save_news({
                "date":             date,
                "source":           source,
                "url":              url,
                "symbol":           symbol,
                "headline":         title,
                "content":          summary,
                "sentiment":        "NEUTRAL",
                "price_at_publish": None,
                "is_suspicious":    False,
            })
            if news_id:
                saved_sql += 1
        except Exception as e:
            print(f"[sentiment_agent] SQLite ingest fail ({url[:60]}): {e}")

        # ── Tầng 2: ChromaDB ──
        try:
            ok = kb.ingest_news_article(
                symbol=symbol, date=date, source=source,
                url=url, title=title, summary=summary,
            )
            if ok:
                saved_chroma += 1
        except Exception as e:
            print(f"[sentiment_agent] ChromaDB ingest fail ({url[:60]}): {e}")

    print(
        f"[sentiment_agent] Auto-ingest {symbol}: "
        f"SQL={saved_sql}, Chroma={saved_chroma}/{len(articles)}"
    )


def _neutral_result(news_count: int) -> dict:
    return {
        "sentiment_score": 50,
        "sentiment_label": "TRUNG_TÍNH",
        "news_count": news_count,
        "key_positive": [],
        "key_negative": [],
        "sentiment_summary": "Không có tin đáng kể.",
    }
