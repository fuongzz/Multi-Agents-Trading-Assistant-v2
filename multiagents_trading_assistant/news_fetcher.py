"""
news_fetcher.py — Crawl tin tức từ nhiều nguồn, phát hiện bất thường volume,
                  và đánh giá độ tin cậy của tin.

Nguồn cơ bản (sentiment_agent):
  - CafeF tìm kiếm theo mã
  - VnExpress tìm kiếm theo mã

Nguồn chuyên gia (mới):
  - cafef.vn/thi-truong-chung-khoan
  - vietstock.vn/nhan-dinh-thi-truong
  - ssi.com.vn/tin-tuc/bao-cao-phan-tich

Agents KHÔNG import vnstock trực tiếp — dữ liệu volume qua fetcher.py.
"""

import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
import json
import re
import time

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR   = Path(__file__).parent
CACHE_DIR  = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9",
}

_TIMEOUT = 10  # giây

# ──────────────────────────────────────────────
# Bảng độ tin cậy cứng theo nguồn
# ──────────────────────────────────────────────

SOURCE_RELIABILITY: dict[str, float] = {
    "SSI Research":  0.90,
    "VCSC":          0.90,
    "VnDirect":      0.85,
    "MBS":           0.80,
    "CafeF":         0.70,
    "Vietstock":     0.65,
    "VnExpress":     0.60,
}

# Ngưỡng để đánh dấu is_suspicious
_SUSPICIOUS_CONTENT_THRESHOLD = 0.4  # content_score < ngưỡng này → nghi ngờ
_SUSPICIOUS_CREDIBILITY_THRESHOLD = 0.5  # credibility_score tổng < ngưỡng → nghi ngờ


# ──────────────────────────────────────────────
# Retry helper
# ──────────────────────────────────────────────

def _retry_request(url: str, max_attempts: int = 3) -> requests.Response:
    """requests.get với exponential backoff retry (1s → 2s → 4s)."""
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < max_attempts - 1:
                wait = 2 ** attempt
                print(f"[news_fetcher] Attempt {attempt + 1} thất bại, thử lại sau {wait}s: {e}")
                time.sleep(wait)
    raise last_exc


# ──────────────────────────────────────────────
# Public API — tin cơ bản theo mã
# ──────────────────────────────────────────────

def get_stock_news(symbol: str, days: int = 3, max_articles: int = 20) -> list[dict]:
    """
    Lấy tin tức liên quan đến một mã cổ phiếu trong N ngày gần nhất.

    Args:
        symbol:       Mã cổ phiếu (VD: "VNM")
        days:         Số ngày nhìn lại (mặc định 3)
        max_articles: Số bài tối đa trả về (mặc định 20)

    Returns:
        list[dict] mỗi phần tử gồm:
          - title:        tiêu đề bài viết
          - summary:      tóm tắt / đoạn đầu
          - source:       "CafeF" hoặc "VnExpress"
          - url:          đường dẫn bài viết
          - published_at: chuỗi ngày giờ (ISO hoặc raw từ site)
    """
    today = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"{symbol}_{today}_news"
    cached = _load_cache(cache_key)
    if cached is not None:
        print(f"[news_fetcher] Cache hit — {len(cached)} bài cho {symbol}")
        return cached[:max_articles]

    print(f"[news_fetcher] Crawl tin tức {symbol} ({days} ngày)...")
    articles: list[dict] = []

    # Nguồn 1: CafeF (retry 3 lần bên trong _crawl_cafef)
    cafef_ok = False
    try:
        cafef_articles = _crawl_cafef(symbol, days)
        articles.extend(cafef_articles)
        cafef_ok = bool(cafef_articles)
        print(f"[news_fetcher] CafeF: {len(cafef_articles)} bài")
    except Exception as e:
        print(f"[news_fetcher] CafeF thất bại hoàn toàn: {e}")

    time.sleep(1)

    # Nguồn 2: VnExpress — luôn thử để bổ sung bài, hoặc fallback khi CafeF trống
    try:
        vnexpress_articles = _crawl_vnexpress(symbol, days)
        articles.extend(vnexpress_articles)
        print(f"[news_fetcher] VnExpress: {len(vnexpress_articles)} bài")
    except Exception as e:
        print(f"[news_fetcher] VnExpress thất bại hoàn toàn: {e}")

    # Nguồn 3: Stale cache — fallback cuối khi cả hai crawler đều trống
    if not articles:
        stale = _load_stale_cache(symbol)
        if stale:
            print(f"[news_fetcher] Fallback stale cache: {len(stale)} bài (dữ liệu cũ)")
            return stale[:max_articles]
        print(f"[news_fetcher] Không lấy được tin {symbol}, trả về rỗng")
        return []

    _save_cache(cache_key, articles)
    print(f"[news_fetcher] Tổng: {len(articles)} bài cho {symbol}")
    return articles[:max_articles]


# ──────────────────────────────────────────────
# Public API — tin chuyên gia (nhận định thị trường)
# ──────────────────────────────────────────────

def get_expert_news(date: str | None = None, max_articles: int = 20) -> list[dict]:
    """
    Lấy nhận định/phân tích từ các nguồn chuyên gia:
      - CafeF thị trường chứng khoán
      - Vietstock nhận định
      - SSI Research báo cáo công khai

    Args:
        date:         Ngày tham chiếu (YYYY-MM-DD), mặc định hôm nay
        max_articles: Số bài tối đa

    Returns:
        list[dict] mỗi phần tử gồm:
          - source:            tên nguồn
          - reliability:       float 0-1 (từ SOURCE_RELIABILITY)
          - author:            tên tác giả nếu có
          - headline:          tiêu đề
          - content:           tóm tắt ≤ 200 từ
          - date:              YYYY-MM-DD
          - symbols_mentioned: list[str] — mã CK đề cập
          - url:               đường dẫn bài
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    cache_key = f"expert_{date}"
    cached = _load_cache(cache_key)
    if cached is not None:
        print(f"[news_fetcher] Cache expert hit — {len(cached)} bài ngày {date}")
        return cached[:max_articles]

    print(f"[news_fetcher] Crawl tin chuyên gia ({date})...")
    articles: list[dict] = []

    # CafeF — section thị trường chứng khoán
    try:
        items = _crawl_cafef_market_section()
        articles.extend(items)
        print(f"[news_fetcher] CafeF market section: {len(items)} bài")
    except Exception as e:
        print(f"[news_fetcher] Lỗi CafeF market section: {e}")

    time.sleep(1)

    # Vietstock — nhận định thị trường
    try:
        items = _crawl_vietstock_nhandinh()
        articles.extend(items)
        print(f"[news_fetcher] Vietstock: {len(items)} bài")
    except Exception as e:
        print(f"[news_fetcher] Lỗi Vietstock: {e}")

    time.sleep(1)

    # SSI Research — báo cáo phân tích công khai
    try:
        items = _crawl_ssi_research()
        articles.extend(items)
        print(f"[news_fetcher] SSI Research: {len(items)} bài")
    except Exception as e:
        print(f"[news_fetcher] Lỗi SSI Research: {e}")

    _save_cache(cache_key, articles)
    print(f"[news_fetcher] Tổng chuyên gia: {len(articles)} bài")
    return articles[:max_articles]


# ──────────────────────────────────────────────
# Public API — phát hiện bất thường volume
# ──────────────────────────────────────────────

def detect_volume_anomaly(symbol: str, date: str, lookback: int = 3) -> bool:
    """
    Kiểm tra volume 'lookback' ngày trước 'date' có bất thường không.

    Bất thường = bất kỳ ngày nào trong lookback ngày có volume > 2× TB20.

    Args:
        symbol:   Mã cổ phiếu
        date:     Ngày tham chiếu (YYYY-MM-DD)
        lookback: Số ngày nhìn lại (mặc định 3)

    Returns:
        True nếu volume bất thường, False nếu bình thường hoặc không lấy được data.
    """
    try:
        from multiagents_trading_assistant import fetcher

        # Lấy OHLCV — cần ít nhất 20+lookback phiên
        df = fetcher.get_ohlcv(symbol, n_days=30)
        if df is None or df.empty or "volume" not in df.columns:
            return False

        # Tính TB20 — date là cột thường, sort theo cột
        df = df.sort_values("date").reset_index(drop=True)
        df["vol_ma20"] = df["volume"].rolling(20, min_periods=10).mean()

        # Lấy lookback ngày gần nhất trước date
        try:
            cutoff = pd.Timestamp(date)
        except Exception:
            cutoff = pd.Timestamp.now()

        recent = df[df["date"] <= cutoff].tail(lookback)
        if recent.empty:
            return False

        for _, row in recent.iterrows():
            ma20 = row.get("vol_ma20")
            vol  = row.get("volume")
            if ma20 and vol and ma20 > 0 and vol > 2 * ma20:
                print(f"[news_fetcher] ⚠️ Volume anomaly {symbol}: "
                      f"{vol:,.0f} > 2× TB20 {ma20:,.0f}")
                return True

        return False

    except Exception as e:
        print(f"[news_fetcher] detect_volume_anomaly lỗi ({symbol}): {e}")
        return False


# ──────────────────────────────────────────────
# Public API — đánh giá độ tin cậy tin
# ──────────────────────────────────────────────

_CREDIBILITY_SYSTEM = """Bạn là chuyên gia phát hiện tin tức thao túng thị trường chứng khoán Việt Nam.
Nhiệm vụ: chấm điểm nội dung một bài báo dựa trên các dấu hiệu đáng ngờ (TRAP_SIGNALS).

TRAP_SIGNALS — tin nghi ngờ nếu có nhiều dấu hiệu:
- Ngôn ngữ mơ hồ, không có số liệu cụ thể ("sẽ tăng mạnh", "cơ hội không thể bỏ qua")
- Không trích dẫn nguồn hoặc tác giả ẩn danh
- Đề cập cổ phiếu penny / thanh khoản thấp kèm "cơ hội"
- Quá lạc quan mà không đề cập rủi ro
- Thông tin không thể kiểm chứng hoặc mâu thuẫn với số liệu thực tế
- Xuất hiện từ khóa kêu gọi hành động gấp ("mua ngay", "hôm nay cuối cùng")

Trả về JSON hợp lệ DUY NHẤT:
{
  "content_score": <float 0.0-1.0 — 1.0 là hoàn toàn đáng tin>,
  "trap_signals_found": [<str> — danh sách dấu hiệu phát hiện được],
  "assessment": <str — 1 câu đánh giá tiếng Việt>
}"""


def evaluate_news_credibility(news_item: dict) -> dict:
    """
    Đánh giá độ tin cậy tổng hợp của 1 tin tức.

    Kết hợp 3 nguồn:
      1. source_score  — tra SOURCE_RELIABILITY + database.get_source_credibility()
      2. volume_anomaly — detect_volume_anomaly() cho symbol liên quan
      3. content_score  — LLM (Haiku) chấm nội dung theo TRAP_SIGNALS

    Args:
        news_item: dict với các key tối thiểu: source, headline, content,
                   symbols_mentioned (hoặc symbol), date

    Returns:
        {
          "credibility_score": float 0.0-1.0,
          "is_suspicious":     bool,
          "source_score":      float,
          "volume_anomaly":    bool,
          "content_score":     float,
          "reasons":           list[str]
        }
    """
    source   = news_item.get("source", "")
    headline = news_item.get("headline") or news_item.get("title", "")
    content  = news_item.get("content") or news_item.get("summary", "")
    date     = news_item.get("date", datetime.now().strftime("%Y-%m-%d"))
    symbols  = news_item.get("symbols_mentioned") or (
        [news_item["symbol"]] if news_item.get("symbol") else []
    )

    reasons: list[str] = []

    # ── 1. Source score ──
    source_score = _get_combined_source_score(source)

    # ── 2. Volume anomaly ──
    volume_anomaly = False
    for sym in symbols[:2]:  # chỉ check 2 mã đầu để tiết kiệm thời gian
        try:
            if detect_volume_anomaly(sym, date):
                volume_anomaly = True
                reasons.append(f"Volume bất thường {sym} trùng thời điểm đăng tin")
                break
        except Exception:
            pass

    # ── 3. Content score (LLM) ──
    content_score = _llm_score_content(headline, content)

    # ── Tổng hợp credibility_score ──
    # Trọng số: source 40%, content 50%, volume_anomaly -10% penalty
    vol_penalty = 0.10 if volume_anomaly else 0.0
    credibility_score = round(
        source_score * 0.4 + content_score * 0.5 - vol_penalty,
        3,
    )
    credibility_score = max(0.0, min(1.0, credibility_score))

    # ── Xác định is_suspicious ──
    is_suspicious = (
        credibility_score < _SUSPICIOUS_CREDIBILITY_THRESHOLD
        or content_score < _SUSPICIOUS_CONTENT_THRESHOLD
    )

    # Ghi lý do thêm
    if source_score < 0.6:
        reasons.append(f"Nguồn '{source}' có độ tin cậy thấp ({source_score:.2f})")
    if content_score < _SUSPICIOUS_CONTENT_THRESHOLD:
        reasons.append(f"Nội dung có dấu hiệu TRAP (content_score={content_score:.2f})")

    return {
        "credibility_score": credibility_score,
        "is_suspicious":     is_suspicious,
        "source_score":      source_score,
        "volume_anomaly":    volume_anomaly,
        "content_score":     content_score,
        "reasons":           reasons,
    }


# ──────────────────────────────────────────────
# CafeF crawler — tìm kiếm theo mã
# ──────────────────────────────────────────────

def _crawl_cafef(symbol: str, days: int) -> list[dict]:
    """
    Crawl CafeF — tìm kiếm theo mã cổ phiếu.
    URL: https://cafef.vn/tim-kiem.chn?keywords={symbol}
    """
    url = f"https://cafef.vn/tim-kiem.chn?keywords={symbol}"
    articles = []

    try:
        resp = _retry_request(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        cutoff = datetime.now() - timedelta(days=days)

        items = soup.select("div.tlitem, div.item-news, li.item, div.news-item")
        if not items:
            items = soup.select("h3 a, h2 a, .title a")

        for item in items[:30]:
            try:
                article = _parse_cafef_item(item, cutoff)
                if article:
                    articles.append(article)
            except Exception:
                continue

    except requests.RequestException as e:
        print(f"[news_fetcher] CafeF request lỗi: {e}")

    return articles


def _parse_cafef_item(item, cutoff: datetime) -> dict | None:
    """Parse một item tin từ CafeF."""
    if item.name == "a":
        link_tag = item
        title = item.get_text(strip=True)
    else:
        link_tag = item.find("a")
        if not link_tag:
            return None
        title = link_tag.get_text(strip=True) or item.get_text(strip=True)[:100]

    href = link_tag.get("href", "")
    if not href:
        return None

    if href.startswith("/"):
        href = "https://cafef.vn" + href
    elif not href.startswith("http"):
        return None

    summary_tag = item.find("p") if item.name != "a" else None
    summary = summary_tag.get_text(strip=True) if summary_tag else ""

    time_tag = item.find("span", class_=lambda c: c and ("time" in c or "date" in c))
    published_at = time_tag.get_text(strip=True) if time_tag else ""

    if not title or len(title) < 10:
        return None

    return {
        "title": title,
        "summary": summary[:300],
        "source": "CafeF",
        "url": href,
        "published_at": published_at,
    }


# ──────────────────────────────────────────────
# CafeF — section thị trường chứng khoán
# ──────────────────────────────────────────────

def _crawl_cafef_market_section() -> list[dict]:
    """
    Crawl CafeF section nhận định thị trường chứng khoán.
    URL: https://cafef.vn/thi-truong-chung-khoan.chn
    """
    url = "https://cafef.vn/thi-truong-chung-khoan.chn"
    articles = []
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        resp = _retry_request(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        items = soup.select("div.tlitem, div.item-news, li.item, article")
        if not items:
            items = soup.select("h3 a, h2 a")

        for item in items[:20]:
            try:
                article = _parse_expert_cafef_item(item, today)
                if article:
                    articles.append(article)
            except Exception:
                continue

    except requests.RequestException as e:
        print(f"[news_fetcher] CafeF market section request lỗi: {e}")

    return articles


def _parse_expert_cafef_item(item, date: str) -> dict | None:
    """Parse một item chuyên gia từ CafeF, trả về expert schema."""
    if item.name == "a":
        link_tag = item
        headline = item.get_text(strip=True)
    else:
        link_tag = item.find("a")
        if not link_tag:
            return None
        headline = link_tag.get_text(strip=True)

    href = link_tag.get("href", "")
    if not href:
        return None
    if href.startswith("/"):
        href = "https://cafef.vn" + href
    elif not href.startswith("http"):
        return None

    if not headline or len(headline) < 10:
        return None

    # Tóm tắt
    summary_tag = item.find("p") if item.name != "a" else None
    content = summary_tag.get_text(strip=True)[:400] if summary_tag else headline

    # Tác giả
    author_tag = item.find(class_=lambda c: c and "author" in (c or ""))
    author = author_tag.get_text(strip=True) if author_tag else ""

    return {
        "source":            "CafeF",
        "reliability":       SOURCE_RELIABILITY.get("CafeF", 0.7),
        "author":            author,
        "headline":          headline,
        "content":           content[:800],
        "date":              date,
        "symbols_mentioned": _extract_symbols(headline + " " + content),
        "url":               href,
    }


# ──────────────────────────────────────────────
# VnExpress crawler — tìm kiếm theo mã
# ──────────────────────────────────────────────

def _crawl_vnexpress(symbol: str, days: int) -> list[dict]:
    """
    Crawl VnExpress — search theo mã cổ phiếu.
    URL: https://timkiem.vnexpress.net/?q={symbol}&cate_code=kinh-doanh
    """
    url = f"https://timkiem.vnexpress.net/?q={symbol}&cate_code=kinh-doanh"
    articles = []

    try:
        resp = _retry_request(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        cutoff = datetime.now() - timedelta(days=days)

        items = soup.select("article.item-news, div.item-news-common, .article-item")
        if not items:
            items = soup.select("h3.title-news a, h2.title-news a")

        for item in items[:30]:
            try:
                article = _parse_vnexpress_item(item, cutoff)
                if article:
                    articles.append(article)
            except Exception:
                continue

    except requests.RequestException as e:
        print(f"[news_fetcher] VnExpress request lỗi: {e}")

    return articles


def _parse_vnexpress_item(item, cutoff: datetime) -> dict | None:
    """Parse một item tin từ VnExpress."""
    if item.name == "a":
        link_tag = item
        title = item.get_text(strip=True)
    else:
        link_tag = item.find("a", class_=lambda c: c and "title" in c) or item.find("a")
        if not link_tag:
            return None
        title = link_tag.get_text(strip=True)

    href = link_tag.get("href", "")
    if not href or not href.startswith("http"):
        return None

    desc_tag = item.find("p", class_=lambda c: c and "description" in (c or ""))
    if not desc_tag:
        desc_tag = item.find("p")
    summary = desc_tag.get_text(strip=True) if desc_tag else ""

    time_tag = item.find("span", class_="time-count") or item.find("span", class_="date")
    published_at = time_tag.get_text(strip=True) if time_tag else ""

    if not title or len(title) < 10:
        return None

    return {
        "title": title,
        "summary": summary[:300],
        "source": "VnExpress",
        "url": href,
        "published_at": published_at,
    }


# ──────────────────────────────────────────────
# Vietstock — nhận định thị trường
# ──────────────────────────────────────────────

def _crawl_vietstock_nhandinh() -> list[dict]:
    """
    Crawl Vietstock section nhận định thị trường.
    URL: https://vietstock.vn/nhan-dinh-thi-truong
    """
    url = "https://vietstock.vn/nhan-dinh-thi-truong"
    articles = []
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        resp = _retry_request(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Vietstock dùng nhiều layout khác nhau — thử các selector phổ biến
        items = soup.select(
            "div.news-item, article.post, div.article-item, "
            "ul.list-news li, div.box-item"
        )
        if not items:
            items = soup.select("h3 a, h2 a, .title a")

        for item in items[:15]:
            try:
                article = _parse_vietstock_item(item, today)
                if article:
                    articles.append(article)
            except Exception:
                continue

    except requests.RequestException as e:
        print(f"[news_fetcher] Vietstock request lỗi: {e}")

    return articles


def _parse_vietstock_item(item, date: str) -> dict | None:
    """Parse một item từ Vietstock, trả về expert schema."""
    if item.name == "a":
        link_tag = item
        headline = item.get_text(strip=True)
    else:
        link_tag = item.find("a")
        if not link_tag:
            return None
        headline = link_tag.get_text(strip=True)

    href = link_tag.get("href", "")
    if not href:
        return None
    if href.startswith("/"):
        href = "https://vietstock.vn" + href
    elif not href.startswith("http"):
        return None

    if not headline or len(headline) < 10:
        return None

    # Tóm tắt từ thẻ p hoặc span mô tả
    desc_tag = item.find("p") or item.find("span", class_=lambda c: c and "desc" in (c or ""))
    content = desc_tag.get_text(strip=True)[:400] if desc_tag else headline

    # Tác giả
    author_tag = item.find(class_=lambda c: c and ("author" in (c or "") or "writer" in (c or "")))
    author = author_tag.get_text(strip=True) if author_tag else ""

    return {
        "source":            "Vietstock",
        "reliability":       SOURCE_RELIABILITY.get("Vietstock", 0.65),
        "author":            author,
        "headline":          headline,
        "content":           content[:800],
        "date":              date,
        "symbols_mentioned": _extract_symbols(headline + " " + content),
        "url":               href,
    }


# ──────────────────────────────────────────────
# SSI Research — báo cáo phân tích công khai
# ──────────────────────────────────────────────

def _crawl_ssi_research() -> list[dict]:
    """
    Crawl SSI báo cáo phân tích công khai.
    URL: https://www.ssi.com.vn/khach-hang-ca-nhan/bao-cao-phan-tich
    (trang public — chỉ lấy tiêu đề + tóm tắt)
    """
    url = "https://www.ssi.com.vn/khach-hang-ca-nhan/bao-cao-phan-tich"
    articles = []
    today = datetime.now().strftime("%Y-%m-%d")

    try:
        resp = _retry_request(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        # SSI dùng nhiều framework khác nhau qua các phiên bản
        items = soup.select(
            "div.report-item, div.news-item, article, "
            "li.item, div.item, tr.report-row"
        )
        if not items:
            items = soup.select("h3 a, h4 a, td a, .title a")

        for item in items[:15]:
            try:
                article = _parse_ssi_item(item, today)
                if article:
                    articles.append(article)
            except Exception:
                continue

    except requests.RequestException as e:
        print(f"[news_fetcher] SSI Research request lỗi: {e}")

    return articles


def _parse_ssi_item(item, date: str) -> dict | None:
    """Parse một item từ SSI Research, trả về expert schema."""
    if item.name == "a":
        link_tag = item
        headline = item.get_text(strip=True)
    else:
        link_tag = item.find("a")
        if not link_tag:
            return None
        headline = link_tag.get_text(strip=True)

    href = link_tag.get("href", "")
    if not href:
        return None
    if href.startswith("/"):
        href = "https://www.ssi.com.vn" + href
    elif not href.startswith("http"):
        return None

    if not headline or len(headline) < 10:
        return None

    desc_tag = item.find("p") or item.find("span") or item.find("td")
    content = desc_tag.get_text(strip=True)[:400] if desc_tag else headline

    # Tác giả / analyst thường có trong title hoặc meta
    author_tag = item.find(class_=lambda c: c and "analyst" in (c or ""))
    author = author_tag.get_text(strip=True) if author_tag else "SSI Research"

    return {
        "source":            "SSI Research",
        "reliability":       SOURCE_RELIABILITY.get("SSI Research", 0.9),
        "author":            author,
        "headline":          headline,
        "content":           content[:800],
        "date":              date,
        "symbols_mentioned": _extract_symbols(headline + " " + content),
        "url":               href,
    }


# ──────────────────────────────────────────────
# Helpers nội bộ
# ──────────────────────────────────────────────

def _extract_symbols(text: str) -> list[str]:
    """
    Trích xuất các mã cổ phiếu VN trong đoạn text.
    Mã VN: 2-4 chữ cái in hoa, không phải từ khóa tiếng Anh thông dụng.
    """
    _EXCLUDE = {
        "VN", "THE", "AND", "FOR", "CEO", "CFO", "IPO", "GDP", "FDI",
        "USD", "VND", "ETF", "NAV", "ROE", "EPS", "BSC", "SSI", "VPS",
        "MUA", "BAN", "CHO", "NOT", "BUT",
    }
    found = re.findall(r"\b([A-Z]{2,4})\b", text)
    return list({s for s in found if s not in _EXCLUDE})[:10]


def _get_combined_source_score(source: str) -> float:
    """
    Kết hợp SOURCE_RELIABILITY cứng và credibility từ database (nếu có).
    Trọng số: hardcode 60%, database history 40%.
    """
    hardcode = SOURCE_RELIABILITY.get(source, 0.55)

    try:
        from multiagents_trading_assistant.database import get_source_credibility
        db_score = get_source_credibility(source)
        # database trả về 0.5 nếu chưa có dữ liệu — bỏ qua trọng lượng đó
        if db_score != 0.5:
            return round(hardcode * 0.6 + db_score * 0.4, 3)
    except Exception:
        pass

    return hardcode


def _llm_score_content(headline: str, content: str) -> float:
    """
    Dùng Haiku chấm điểm nội dung theo TRAP_SIGNALS.
    Trả về float 0.0-1.0, mặc định 0.6 nếu LLM lỗi.
    """
    if not headline and not content:
        return 0.5

    try:
        from multiagents_trading_assistant.agent import run_agent_lite

        prompt = f"""Đánh giá độ tin cậy nội dung bài báo chứng khoán sau:

Tiêu đề: {headline[:200]}

Nội dung: {content[:500]}

Chấm điểm theo TRAP_SIGNALS và trả về JSON."""

        result = run_agent_lite(prompt=prompt, system=_CREDIBILITY_SYSTEM)
        score = result.get("content_score", 0.6)
        return max(0.0, min(1.0, float(score)))

    except Exception as e:
        print(f"[news_fetcher] LLM content score lỗi: {e}")
        return 0.6  # fallback trung tính


# ──────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────

def _load_cache(key: str) -> list | None:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _load_stale_cache(symbol: str) -> list | None:
    """Tìm cache cũ nhất còn tồn tại cho symbol (bất kỳ ngày nào).

    Dùng làm fallback cuối cùng khi tất cả crawler đều thất bại.
    """
    matches = sorted(CACHE_DIR.glob(f"{symbol}_*_news.json"), reverse=True)
    for p in matches:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data:
                print(f"[news_fetcher] Dùng stale cache: {p.name}")
                return data
        except Exception:
            continue
    return None


def _save_cache(key: str, data: list) -> None:
    try:
        p = CACHE_DIR / f"{key}.json"
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[news_fetcher] Lỗi ghi cache: {e}")


# ──────────────────────────────────────────────
# Test nhanh
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Test tin cơ bản
    articles = get_stock_news("VNM", days=3, max_articles=5)
    print(f"\nTin cơ bản VNM: {len(articles)} bài")
    for a in articles:
        print(f"  [{a['source']}] {a['title'][:80]}")

    # Test tin chuyên gia
    expert = get_expert_news(max_articles=5)
    print(f"\nTin chuyên gia: {len(expert)} bài")
    for a in expert:
        print(f"  [{a['source']} | {a['reliability']}] {a['headline'][:80]}")
        print(f"    Symbols: {a['symbols_mentioned']}")

    # Test volume anomaly
    anomaly = detect_volume_anomaly("VNM", "2026-04-12")
    print(f"\nVolume anomaly VNM: {anomaly}")

    # Test evaluate credibility
    sample_news = {
        "source":            "CafeF",
        "headline":          "VNM tăng mạnh, cơ hội không thể bỏ qua!",
        "content":           "Cổ phiếu VNM được dự báo sẽ tăng mạnh trong tuần tới.",
        "date":              "2026-04-12",
        "symbols_mentioned": ["VNM"],
    }
    cred = evaluate_news_credibility(sample_news)
    print(f"\nCredibility VNM sample: {cred}")
