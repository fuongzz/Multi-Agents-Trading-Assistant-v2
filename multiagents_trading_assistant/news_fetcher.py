"""
news_fetcher.py — Crawl tin tức từ nhiều nguồn, phát hiện bất thường volume,
                  và đánh giá độ tin cậy của tin.

Nguồn cơ bản (sentiment_agent) — qua vnstock_news RSS:
  - CafeF, VnExpress, CafeBiz, Vietstock (feed chung, filter per symbol)

Nguồn chuyên gia:
  - CafeF / Vietstock / CafeBiz (reuse feed đã fetch)
  - ssi.com.vn/tin-tuc/bao-cao-phan-tich (legacy crawler, SSI không có trong vnstock_news)

Chiến lược: fetch toàn bộ feed 1 lần/ngày, cache, rồi filter per symbol client-side.
Giảm từ ~160 HTTP requests/run xuống còn 4 requests cho toàn bộ screener.

Agents KHÔNG import vnstock trực tiếp — dữ liệu volume qua fetcher.py.
"""

import email.utils
import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
import json
import re
import time

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── vnstock_news import (optional — fallback về legacy nếu chưa cài) ──
try:
    from vnstock_news import Crawler as _VNNewsCrawler
    _VNSTOCK_NEWS_AVAILABLE = True
except ImportError:
    _VNSTOCK_NEWS_AVAILABLE = False

# ── vnstock.api news (Golden sponsor — FiinGroup per-symbol) ──
try:
    from vnstock.api.company import Company as _VnstockCompany
    _VNSTOCK_API_NEWS_AVAILABLE = True
except ImportError:
    _VNSTOCK_API_NEWS_AVAILABLE = False

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
    "FiinGroup":     0.85,
    "MBS":           0.80,
    "CafeF":         0.70,
    "CafeBiz":       0.65,
    "Vietstock":     0.65,
    "VnExpress":     0.60,
}

# Ngưỡng để đánh dấu is_suspicious
_SUSPICIOUS_CONTENT_THRESHOLD = 0.4  # content_score < ngưỡng này → nghi ngờ
_SUSPICIOUS_CREDIBILITY_THRESHOLD = 0.5  # credibility_score tổng < ngưỡng → nghi ngờ

# ──────────────────────────────────────────────
# vnstock_news config
# ──────────────────────────────────────────────

# CafeF không có RSS trong vnstock_news config mặc định.
# Sitemap chỉ trả về URL/lastmod (không có title) → không dùng được cho feed cache.
# CafeF symbol-specific news vẫn qua legacy crawler (Tier 2).
_VNSTOCK_SITES_RSS = ["vnexpress", "cafebiz", "vietstock"]

_SITE_DISPLAY: dict[str, str] = {
    "cafef":     "CafeF",
    "vnexpress": "VnExpress",
    "cafebiz":   "CafeBiz",
    "vietstock": "Vietstock",
}

# In-memory feed cache — tránh re-fetch trong cùng một process run
_FEED_CACHE: dict[str, list[dict]] = {}

# ──────────────────────────────────────────────
# Ánh xạ ticker → tên công ty (dùng để filter RSS feed)
# Bao gồm ~100 mã liquid nhất HOSE (VN30 + VN100 + blue chips phổ biến)
# ──────────────────────────────────────────────

_COMPANY_NAMES: dict[str, list[str]] = {
    # ── Ngân hàng ──
    "VCB":  ["Vietcombank", "Ngoại thương"],
    "BID":  ["BIDV", "Đầu tư và Phát triển"],
    "CTG":  ["VietinBank", "Vietinbank", "Công thương"],
    "MBB":  ["MB Bank", "MB", "Quân đội"],
    "TCB":  ["Techcombank"],
    "ACB":  ["ACB", "Á Châu"],
    "VPB":  ["VPBank", "VP Bank", "Việt Nam Thịnh Vượng"],
    "STB":  ["Sacombank", "Sài Gòn Thương Tín"],
    "HDB":  ["HDBank", "HD Bank"],
    "TPB":  ["TPBank", "TP Bank", "Tiên Phong"],
    "LPB":  ["LPBank", "LP Bank", "Lộc Phát"],
    "MSB":  ["MSB", "Hàng Hải"],
    "OCB":  ["OCB", "Phương Đông"],
    "VIB":  ["VIB", "Quốc tế"],
    "EIB":  ["Eximbank"],
    "SHB":  ["SHB", "Sài Gòn – Hà Nội"],
    "ABB":  ["ABBank", "An Bình"],
    "SSB":  ["SeABank", "SeA Bank"],

    # ── Bất động sản & Xây dựng ──
    "VIC":  ["Vingroup", "Vin Group"],
    "VHM":  ["Vinhomes"],
    "VRE":  ["Vincom Retail", "Vincom"],
    "NVL":  ["Novaland", "Nova Land"],
    "PDR":  ["Phát Đạt"],
    "DXG":  ["Đất Xanh"],
    "KDH":  ["Khang Điền"],
    "NLG":  ["Nam Long"],
    "DIG":  ["DIC Corp", "DIC"],
    "BCM":  ["Becamex"],
    "CEO":  ["CEO Group"],
    "HDG":  ["Hà Đô"],
    "SJS":  ["Sudico"],
    "ITC":  ["Đầu tư Kinh doanh nhà"],
    "LDG":  ["LDG"],
    "AGG":  ["An Gia"],
    "CII":  ["CII", "Hạ tầng Kỹ thuật"],
    "FCN":  ["FECON"],
    "HHV":  ["Hưng Hải Gia"],
    "PC1":  ["PC1", "PCC1"],

    # ── Thép & Vật liệu ──
    "HPG":  ["Hòa Phát", "Hoà Phát"],
    "HSG":  ["Hoa Sen"],
    "NKG":  ["Nam Kim"],
    "TLH":  ["Thép Tiến Lên"],
    "VGC":  ["Viglacera"],
    "BMP":  ["Bình Minh"],
    "CSV":  ["Hóa chất Cơ bản"],

    # ── Dầu khí & Năng lượng ──
    "GAS":  ["PV Gas", "PVGas", "Khí Việt Nam"],
    "PLX":  ["Petrolimex"],
    "PVD":  ["PV Drilling", "Khoan Dầu khí"],
    "PVS":  ["PV Technical Services", "PTSC", "Dịch vụ Kỹ thuật Dầu khí"],
    "BSR":  ["Bình Sơn", "Lọc hóa dầu Bình Sơn"],
    "OIL":  ["PV Oil", "PVOil"],
    "PGV":  ["PV Power", "PVPower", "Điện lực Dầu khí"],
    "POW":  ["PV Power", "Điện lực Dầu khí"],
    "EVF":  ["EVN Finance"],

    # ── Công nghệ & Viễn thông ──
    "FPT":  ["FPT"],
    "CMG":  ["CMC"],
    "ELC":  ["Elcom"],
    "VGI":  ["Viettel Global"],
    "FOX":  ["Fox"],

    # ── Bán lẻ & Tiêu dùng ──
    "MWG":  ["Thế Giới Di Động", "TGDĐ"],
    "PNJ":  ["Phú Nhuận", "PNJ"],
    "DGW":  ["Digiworld"],
    "FRT":  ["FPT Retail"],
    "VNM":  ["Vinamilk", "Sữa Việt Nam"],
    "SAB":  ["Sabeco", "Bia Sài Gòn"],
    "VHC":  ["Vĩnh Hoàn"],
    "ANV":  ["Nam Việt"],
    "IDI":  ["IDI", "Đại Dương Xanh"],
    "MSN":  ["Masan"],
    "MCH":  ["Masan Consumer"],
    "QNS":  ["Đường Quảng Ngãi"],
    "KDC":  ["Kido"],

    # ── Hàng không & Vận tải ──
    "HVN":  ["Vietnam Airlines", "Hàng không Việt Nam"],
    "VJC":  ["Vietjet", "Viet Jet"],
    "ACV":  ["ACV", "Cảng hàng không"],
    "GMD":  ["Gemadept"],
    "STG":  ["Sotrans"],
    "HAH":  ["Hải An"],
    "VSC":  ["Container Việt Nam"],
    "MVN":  ["VIMC", "Vosco"],

    # ── Chứng khoán & Tài chính ──
    "SSI":  ["SSI", "Chứng khoán SSI"],
    "VND":  ["VnDirect", "Chứng khoán VnDirect"],
    "HCM":  ["Chứng khoán HCM", "HCMS"],
    "VCI":  ["Vietcap", "Chứng khoán Vietcap"],
    "BSI":  ["BSC", "Chứng khoán BIDV"],
    "MBS":  ["MBS", "Chứng khoán MB"],
    "SHS":  ["Chứng khoán Sài Gòn – Hà Nội"],
    "VDS":  ["Rồng Việt"],
    "AGR":  ["Agribank Securities"],
    "FTS":  ["Fintrade"],

    # ── Điện & Tiện ích ──
    "REE":  ["REE", "Cơ điện lạnh"],
    "PC1":  ["PC1", "PCC1"],
    "TV2":  ["Tư vấn Điện 2"],
    "SBA":  ["Sông Ba"],
    "GEX":  ["Gelex"],
    "BWE":  ["Nước Biwase"],
    "TBC":  ["Thủy điện Thác Bà"],
    "NT2":  ["Nhiệt điện Nhơn Trạch 2"],
    "SGT":  ["Saigon Tel"],

    # ── Nông nghiệp & Thủy sản ──
    "HAG":  ["Hoàng Anh Gia Lai", "HAGL"],
    "HNG":  ["HAGL Agrico"],
    "BAF":  ["BA F", "BAF"],
    "LSS":  ["Đường Lam Sơn"],
    "SBT":  ["TTC AgriS", "Đường TTC"],
    "FMC":  ["Sao Ta"],

    # ── Dược phẩm & Y tế ──
    "DHG":  ["Dược Hậu Giang"],
    "IMP":  ["Imexpharm"],
    "DBD":  ["Dược Bình Định"],
    "TRA":  ["Traphaco"],
    "DMC":  ["Domesco"],

    # ── Bảo hiểm ──
    "BVH":  ["Bảo Việt"],
    "PVI":  ["PVI", "Bảo hiểm Dầu khí"],
    "BMI":  ["BaoMinh", "Bảo Minh"],
    "PTI":  ["Bảo hiểm Bưu điện"],
}

# Keywords nhận dạng bài chuyên gia/phân tích
_EXPERT_KEYWORDS = (
    "thị trường", "nhận định", "phân tích", "chứng khoán",
    "vn-index", "vnindex", "phiên giao dịch", "khuyến nghị",
    "triển vọng", "dự báo",
)


# ──────────────────────────────────────────────
# Retry helper (dùng cho SSI legacy crawler)
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
# Tier 0: vnstock.api — FiinGroup per-symbol news
# ──────────────────────────────────────────────

def _fetch_vnstock_api_news(symbol: str, days: int) -> list[dict]:
    """Lấy news từ FiinGroup qua vnstock.api — chính xác theo mã, không cần filter."""
    if not _VNSTOCK_API_NEWS_AVAILABLE:
        return []
    try:
        from vnstock.api.company import Company
        df = Company(symbol=symbol, source="VCI").news()
        if df is None or df.empty:
            return []

        cutoff = datetime.now() - timedelta(days=days)
        articles = []
        for _, row in df.iterrows():
            pub_str = str(row.get("public_date") or "")
            pub_dt = _parse_published_at(pub_str) if pub_str else None
            if pub_dt is not None and pub_dt < cutoff:
                continue
            title = str(row.get("news_title") or "").strip()
            if not title:
                continue
            articles.append({
                "title":        title,
                "summary":      str(row.get("news_short_content") or row.get("news_sub_title") or "")[:300].strip(),
                "source":       str(row.get("news_source") or "FiinGroup"),
                "url":          str(row.get("news_source_link") or ""),
                "published_at": pub_str,
                "credibility_score": SOURCE_RELIABILITY.get("FiinGroup", 0.80),
            })
        articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        return articles
    except Exception as e:
        print(f"[news_fetcher] vnstock.api news lỗi ({symbol}): {e}")
        return []


# ──────────────────────────────────────────────
# vnstock_news — fetch & cache toàn bộ feed
# ──────────────────────────────────────────────

def _normalize_vnstock_article(raw: dict, site_name: str) -> dict:
    """Chuyển vnstock_news dict sang output schema của news_fetcher."""
    pub = raw.get("publish_time")
    if pub is None:
        published_at = ""
    elif hasattr(pub, "isoformat"):
        published_at = pub.isoformat()
    else:
        published_at = str(pub)

    return {
        "title":        raw.get("title", "").strip(),
        "summary":      (raw.get("short_description") or raw.get("content") or "")[:300].strip(),
        "source":       _SITE_DISPLAY.get(site_name, site_name),
        "url":          raw.get("url", ""),
        "published_at": published_at,
    }


def _fetch_all_feeds_cached(date: str) -> list[dict]:
    """
    Fetch toàn bộ RSS feed từ _VNSTOCK_SITES, cache kết quả theo ngày.

    Level 1: in-memory _FEED_CACHE (cùng process)
    Level 2: CACHE_DIR/feeds_{date}.json (khởi động lại process)
    Level 3: fresh fetch từ vnstock_news

    Returns:
        list[dict] tất cả articles đã normalize, dedup by URL.
    """
    cache_key = f"feeds_{date}"

    # Level 1: in-memory
    if cache_key in _FEED_CACHE:
        return _FEED_CACHE[cache_key]

    # Level 2: file cache
    cached = _load_cache(cache_key)
    if cached is not None:
        _FEED_CACHE[cache_key] = cached
        print(f"[news_fetcher] Feed cache hit: {len(cached)} bài (file)")
        return cached

    if not _VNSTOCK_NEWS_AVAILABLE:
        print("[news_fetcher] ⚠️ vnstock_news chưa cài — dùng legacy crawlers")
        return []

    # Level 3: fresh fetch
    print(f"[news_fetcher] Fetch RSS feeds từ {len(_VNSTOCK_SITES_RSS)} nguồn...")
    all_articles: list[dict] = []
    seen_urls: set[str] = set()

    for site in _VNSTOCK_SITES_RSS:
        try:
            crawler = _VNNewsCrawler(site_name=site)
            raw_list = crawler.get_articles_from_feed(limit_per_feed=50)
            count = 0
            for raw in raw_list:
                url = raw.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                article = _normalize_vnstock_article(raw, site)
                if article["title"]:
                    all_articles.append(article)
                    count += 1
            print(f"[news_fetcher] {_SITE_DISPLAY[site]}: {count} bài (RSS)")
        except Exception as e:
            print(f"[news_fetcher] {site} RSS lỗi: {e}")

    _FEED_CACHE[cache_key] = all_articles
    _save_cache(cache_key, all_articles)
    print(f"[news_fetcher] Feed tổng: {len(all_articles)} bài (đã cache)")
    return all_articles


def _parse_published_at(pub_str: str) -> datetime | None:
    """Parse published_at string — hỗ trợ ISO, RFC 2822 và các format phổ biến."""
    if not pub_str:
        return None
    # RFC 2822: "Sun, 03 May 2026 16:01:16 +0700"
    try:
        t = email.utils.parsedate_to_datetime(pub_str)
        return t.replace(tzinfo=None)
    except Exception:
        pass
    # ISO: "2026-05-03T16:01:16" hoặc "2026-05-03"
    try:
        return datetime.fromisoformat(pub_str.split("+")[0].rstrip("Z"))
    except Exception:
        pass
    return None


def _build_symbol_patterns(symbol: str) -> list[re.Pattern]:
    """Tạo danh sách regex patterns cho ticker + tên công ty."""
    keywords = [symbol] + _COMPANY_NAMES.get(symbol, [])
    return [re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE) for k in keywords]


def _filter_by_symbol(articles: list[dict], symbol: str, days: int) -> list[dict]:
    """
    Lọc articles theo symbol và khoảng thời gian.

    Giữ bài nếu:
    - published_at >= cutoff (days ngày gần nhất), hoặc không parse được ngày
    - ticker HOẶC tên công ty xuất hiện trong title/summary
    """
    cutoff = datetime.now() - timedelta(days=days)
    patterns = _build_symbol_patterns(symbol)
    result = []

    for a in articles:
        # Lọc theo thời gian
        pub_dt = _parse_published_at(a.get("published_at", ""))
        if pub_dt is not None and pub_dt < cutoff:
            continue

        # Lọc theo ticker + tên công ty
        text = (a.get("title") or "") + " " + (a.get("summary") or "")
        if any(p.search(text) for p in patterns):
            result.append(a)

    result.sort(key=lambda x: x.get("published_at", ""), reverse=True)
    return result


# ──────────────────────────────────────────────
# Public API — tin cơ bản theo mã
# ──────────────────────────────────────────────

def get_stock_news(symbol: str, days: int = 3, max_articles: int = 20) -> list[dict]:
    """
    Lấy tin tức liên quan đến một mã cổ phiếu trong N ngày gần nhất.

    Chiến lược (3 tier):
      Tier 1: vnstock_news RSS feed (fetch once, filter per symbol)
      Tier 2: Legacy crawlers CafeF + VnExpress (fallback)
      Tier 3: Stale cache (fallback cuối)

    Args:
        symbol:       Mã cổ phiếu (VD: "VNM")
        days:         Số ngày nhìn lại (mặc định 3)
        max_articles: Số bài tối đa trả về (mặc định 20)

    Returns:
        list[dict] mỗi phần tử gồm:
          - title:        tiêu đề bài viết
          - summary:      tóm tắt / đoạn đầu
          - source:       "CafeF", "VnExpress", "CafeBiz", "Vietstock"
          - url:          đường dẫn bài viết
          - published_at: chuỗi ISO datetime
    """
    today = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"{symbol}_{today}_news"
    cached = _load_cache(cache_key)
    if cached is not None:
        print(f"[news_fetcher] Cache hit — {len(cached)} bài cho {symbol}")
        return cached[:max_articles]

    print(f"[news_fetcher] Lay tin {symbol} ({days} ngay)...")

    merged: list[dict] = []
    seen_titles: set[str] = set()

    def _add_articles(new_list: list[dict]) -> None:
        for a in new_list:
            t = (a.get("title") or "").strip()[:60]
            if t and t not in seen_titles:
                seen_titles.add(t)
                merged.append(a)

    # ── Tier 0: vnstock.api — FiinGroup (tin công bố chính thức, chính xác theo mã) ──
    fiin_articles = _fetch_vnstock_api_news(symbol, days)
    _add_articles(fiin_articles)
    print(f"[news_fetcher] FiinGroup: {len(fiin_articles)} bai cho {symbol}")

    # ── Tier 1: vnstock_news RSS feed ──
    all_feeds = _fetch_all_feeds_cached(today)
    rss_articles = _filter_by_symbol(all_feeds, symbol, days)
    _add_articles(rss_articles)
    if rss_articles:
        print(f"[news_fetcher] vnstock_news RSS: {len(rss_articles)} bai cho {symbol}")

    if merged:
        merged.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        print(f"[news_fetcher] Merged Tier0+1: {len(merged)} bai cho {symbol}")
        _save_cache(cache_key, merged)
        return merged[:max_articles]

    # ── Tier 2: Legacy crawlers ──
    print(f"[news_fetcher] vnstock_news không tìm được bài cho {symbol}, dùng legacy crawlers")
    legacy: list[dict] = []
    try:
        legacy.extend(_legacy_crawl_cafef(symbol, days))
    except Exception as e:
        print(f"[news_fetcher] Legacy CafeF lỗi: {e}")

    time.sleep(1)

    try:
        legacy.extend(_legacy_crawl_vnexpress(symbol, days))
    except Exception as e:
        print(f"[news_fetcher] Legacy VnExpress lỗi: {e}")

    if legacy:
        print(f"[news_fetcher] Legacy: {len(legacy)} bài cho {symbol}")
        _save_cache(cache_key, legacy)
        return legacy[:max_articles]

    # ── Tier 3: Stale cache ──
    stale = _load_stale_cache(symbol)
    if stale:
        print(f"[news_fetcher] Stale cache: {len(stale)} bài (dữ liệu cũ)")
        return stale[:max_articles]

    print(f"[news_fetcher] Không lấy được tin {symbol}, trả về rỗng")
    return []


# ──────────────────────────────────────────────
# Public API — tin chuyên gia (nhận định thị trường)
# ──────────────────────────────────────────────

def get_expert_news(date: str | None = None, max_articles: int = 20) -> list[dict]:
    """
    Lấy nhận định/phân tích từ các nguồn chuyên gia:
      - CafeF, Vietstock, CafeBiz (reuse RSS feed đã fetch)
      - SSI Research báo cáo công khai (legacy crawler)

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

    print(f"[news_fetcher] Lấy tin chuyên gia ({date})...")
    articles: list[dict] = []

    # ── Từ RSS feed đã fetch (reuse, không tốn thêm request) ──
    all_feeds = _fetch_all_feeds_cached(date)
    for a in all_feeds:
        text = ((a.get("title") or "") + " " + (a.get("summary") or "")).lower()
        if any(kw in text for kw in _EXPERT_KEYWORDS):
            expert_item = _normalize_feed_to_expert(a, date)
            if expert_item:
                articles.append(expert_item)

    print(f"[news_fetcher] Feed expert filter: {len(articles)} bài")

    # ── SSI Research — legacy crawler (SSI không có trong vnstock_news) ──
    try:
        ssi_items = _crawl_ssi_research()
        articles.extend(ssi_items)
        print(f"[news_fetcher] SSI Research: {len(ssi_items)} bài")
    except Exception as e:
        print(f"[news_fetcher] Lỗi SSI Research: {e}")

    _save_cache(cache_key, articles)
    print(f"[news_fetcher] Tổng chuyên gia: {len(articles)} bài")
    return articles[:max_articles]


def _normalize_feed_to_expert(article: dict, date: str) -> dict | None:
    """Chuyển article từ feed cache sang expert schema."""
    headline = article.get("title", "").strip()
    if not headline or len(headline) < 10:
        return None

    source = article.get("source", "")
    content = article.get("summary") or headline

    return {
        "source":            source,
        "reliability":       SOURCE_RELIABILITY.get(source, 0.60),
        "author":            "",
        "headline":          headline,
        "content":           content[:800],
        "date":              date,
        "symbols_mentioned": _extract_symbols(headline + " " + content),
        "url":               article.get("url", ""),
    }


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
# Legacy CafeF crawler — tìm kiếm theo mã (Tier 2 fallback)
# ──────────────────────────────────────────────

def _legacy_crawl_cafef(symbol: str, days: int) -> list[dict]:
    """
    Legacy crawler CafeF — tìm kiếm theo mã cổ phiếu.
    Dùng khi vnstock_news không tìm được bài cho symbol.
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
        print(f"[news_fetcher] Legacy CafeF request lỗi: {e}")

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
# Legacy VnExpress crawler — tìm kiếm theo mã (Tier 2 fallback)
# ──────────────────────────────────────────────

def _legacy_crawl_vnexpress(symbol: str, days: int) -> list[dict]:
    """
    Legacy crawler VnExpress — tìm kiếm theo mã cổ phiếu.
    Dùng khi vnstock_news không tìm được bài cho symbol.
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
        print(f"[news_fetcher] Legacy VnExpress request lỗi: {e}")

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
