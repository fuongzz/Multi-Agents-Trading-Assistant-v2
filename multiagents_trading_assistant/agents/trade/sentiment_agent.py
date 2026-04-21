"""sentiment_agent.py — Sentiment tin tức cho Trade pipeline.

Model: Haiku. Port từ _legacy/research/sentiment_agent.py (không đổi logic).
"""

import importlib.metadata  # noqa: F401
from datetime import datetime

from multiagents_trading_assistant.services.llm_service import run_agent_lite
from multiagents_trading_assistant.news_fetcher import get_stock_news


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


def _neutral_result(news_count: int) -> dict:
    return {
        "sentiment_score": 50,
        "sentiment_label": "TRUNG_TÍNH",
        "news_count": news_count,
        "key_positive": [],
        "key_negative": [],
        "sentiment_summary": "Không có tin đáng kể.",
    }
