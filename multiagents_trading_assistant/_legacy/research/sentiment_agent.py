"""
sentiment_agent.py — Phân tích cảm xúc tin tức cho một mã cổ phiếu.

Model: Haiku (run_agent_lite) — nhanh + prompt caching
Input:  symbol + list bài báo từ news_fetcher (CafeF + VnExpress, 3 ngày)
Output: dict theo schema agents.md #5
"""

import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
from datetime import datetime

from multiagents_trading_assistant.agent import run_agent_lite
from multiagents_trading_assistant.news_fetcher import get_stock_news


_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích sentiment tin tức tài chính Việt Nam.
Nhiệm vụ: đọc các tiêu đề + tóm tắt bài báo và đánh giá cảm xúc thị trường đối với mã cổ phiếu.

Quy tắc bắt buộc:
- Trả về JSON hợp lệ DUY NHẤT, không có text thừa
- sentiment_score: 0–100 (0=rất tiêu cực, 50=trung tính, 100=rất tích cực)
- sentiment_label: "TIÊU_CỰC" (0-30) | "TRUNG_TÍNH" (31-60) | "TÍCH_CỰC" (61-80) | "RẤT_TÍCH_CỰC" (81-100)
- key_positive: list tối đa 3 điểm tích cực nổi bật (tiếng Việt, ngắn gọn)
- key_negative: list tối đa 3 điểm tiêu cực nổi bật (tiếng Việt, ngắn gọn)
- Nếu không có bài nào: sentiment_score=50, sentiment_label="TRUNG_TÍNH"

Output schema:
{
  "sentiment_score": <int 0-100>,
  "sentiment_label": <str>,
  "news_count": <int>,
  "key_positive": [<str>, ...],
  "key_negative": [<str>, ...],
  "sentiment_summary": <str — 1-2 câu tiếng Việt tóm tắt>
}"""


def analyze(symbol: str, date: str | None = None, days: int = 3) -> dict:
    """
    Chạy Sentiment agent cho một mã.

    Args:
        symbol: Mã cổ phiếu (VD: "VNM")
        date:   Ngày phân tích (YYYY-MM-DD), mặc định hôm nay
        days:   Số ngày nhìn lại để lấy tin (mặc định 3)

    Returns:
        dict theo schema sentiment_agent (agents.md #5)
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    print(f"[sentiment_agent] Bắt đầu phân tích sentiment {symbol} ({date})...")

    # ── Lấy tin tức ──
    articles = get_stock_news(symbol, days=days, max_articles=20)

    if not articles:
        print(f"[sentiment_agent] Không có tin tức cho {symbol}, trả về neutral.")
        return _neutral_result(0)

    # ── Build prompt ──
    news_text = _format_articles(articles)
    prompt = f"""Phân tích sentiment tin tức mã {symbol} ngày {date}.

Tổng số bài: {len(articles)} bài (CafeF + VnExpress, {days} ngày qua)

=== Danh sách bài báo ===
{news_text}

Đánh giá:
- Tính sentiment_score tổng hợp (0-100)
- Xác định key_positive và key_negative từ nội dung bài
- sentiment_summary 1-2 câu mô tả tone tin tức hiện tại

Trả về JSON theo schema đã định."""

    try:
        result = run_agent_lite(prompt=prompt, system=_SYSTEM_PROMPT)
        # Đảm bảo news_count chính xác
        result["news_count"] = len(articles)
        print(f"[sentiment_agent] {symbol} — score={result.get('sentiment_score')}, "
              f"label={result.get('sentiment_label')}, {len(articles)} bài")
        return result
    except Exception as e:
        print(f"[sentiment_agent] Lỗi gọi LLM cho {symbol}: {e}")
        return _neutral_result(len(articles))


def _format_articles(articles: list[dict]) -> str:
    """Format list bài báo thành text cho prompt."""
    lines = []
    for i, a in enumerate(articles, 1):
        source = a.get("source", "?")
        title  = a.get("title", "")[:120]
        summary = a.get("summary", "")[:150]
        pub    = a.get("published_at", "")

        line = f"{i}. [{source}] {title}"
        if pub:
            line += f" ({pub})"
        if summary:
            line += f"\n   → {summary}"
        lines.append(line)

    return "\n".join(lines)


def _neutral_result(news_count: int) -> dict:
    return {
        "sentiment_score": 50,
        "sentiment_label": "TRUNG_TÍNH",
        "news_count": news_count,
        "key_positive": [],
        "key_negative": [],
        "sentiment_summary": "Không có tin tức đáng kể trong 3 ngày qua.",
    }
