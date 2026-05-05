"""LangChain tools wrapping VnstockDataProvider + news_fetcher cho TradingAgents-VN.

as_of_date: tất cả tool nhận tham số này để filter dữ liệu khi backtest.
  - None (default)  → dùng today, hành vi production như cũ.
  - "2024-06-01"    → end_date = as_of_date, tránh leak future data.

Tools là @tool decorator nên không nhận as_of_date trực tiếp từ LLM —
as_of_date được inject qua module-level variable set trước khi graph chạy.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date, timedelta
from functools import lru_cache

from langchain_core.tools import tool

from multiagents_trading_assistant.data.providers.vnstock_provider import VnstockDataProvider
from multiagents_trading_assistant.indicators import compute_indicators

# ── Module-level as_of_date (thread-local-style via context manager) ──────────
_CURRENT_AS_OF_DATE: str | None = None


@contextmanager
def as_of_date_context(as_of_date: str | None):
    """Context manager để set as_of_date cho tất cả tool calls trong scope."""
    global _CURRENT_AS_OF_DATE
    prev = _CURRENT_AS_OF_DATE
    _CURRENT_AS_OF_DATE = as_of_date
    try:
        yield
    finally:
        _CURRENT_AS_OF_DATE = prev


def _get_end_date() -> str:
    if _CURRENT_AS_OF_DATE:
        return _CURRENT_AS_OF_DATE
    return date.today().strftime("%Y-%m-%d")


@lru_cache(maxsize=1)
def _provider() -> VnstockDataProvider:
    return VnstockDataProvider()


def _default_dates(end_override: str | None = None) -> tuple[str, str]:
    end_str = end_override or _get_end_date()
    end = date.fromisoformat(end_str)
    start = end - timedelta(days=365)
    return start.strftime("%Y-%m-%d"), end_str


# ── Market Analyst tools ──────────────────────────────────────────────────────

@tool
def get_stock_data(symbol: str, start_date: str = "", end_date: str = "") -> str:
    """Lấy dữ liệu OHLCV lịch sử cho một mã cổ phiếu Việt Nam.

    Args:
        symbol: Mã cổ phiếu (VCB, HPG, FPT...)
        start_date: Ngày bắt đầu yyyy-mm-dd. Mặc định 1 năm trước.
        end_date: Ngày kết thúc yyyy-mm-dd. Mặc định as_of_date (hoặc hôm nay).

    Returns:
        Chuỗi CSV với các cột: date, open, high, low, close, volume
    """
    effective_end = end_date or _get_end_date()
    if not start_date:
        start_date, _ = _default_dates(effective_end)
    try:
        df = _provider().get_ohlcv(symbol, start_date, effective_end)
        if df.empty:
            return f"Không có dữ liệu OHLCV cho {symbol}"
        return df.tail(60).to_csv(index=False)
    except Exception as exc:
        return f"Lỗi khi lấy OHLCV cho {symbol}: {exc}"


@tool
def get_technical_indicators(symbol: str) -> str:
    """Tính toán các chỉ báo kỹ thuật cho mã cổ phiếu (200 phiên gần nhất).

    Args:
        symbol: Mã cổ phiếu (VCB, HPG, FPT...)

    Returns:
        JSON string với MA, RSI, MACD, Bollinger Bands, ATR, ADX...
    """
    end_str = _get_end_date()
    end = date.fromisoformat(end_str)
    extended_start = (end - timedelta(days=400)).strftime("%Y-%m-%d")
    try:
        df = _provider().get_ohlcv(symbol, extended_start, end_str)
        if df.empty:
            return f"Không có dữ liệu cho {symbol}"
        indicators = compute_indicators(df)
        clean = {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in indicators.items() if v is not None}
        return json.dumps(clean, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Lỗi khi tính chỉ báo cho {symbol}: {exc}"


# ── Fundamental Analyst tools ─────────────────────────────────────────────────

@tool
def get_fundamentals(symbol: str) -> str:
    """Lấy dữ liệu cơ bản của một mã cổ phiếu Việt Nam.

    Args:
        symbol: Mã cổ phiếu (VCB, HPG, FPT...)

    Returns:
        JSON string với PE, PB, ROE, EPS, tăng trưởng doanh thu/lợi nhuận, ngành
    """
    try:
        data = _provider().get_fundamentals(symbol)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Lỗi khi lấy dữ liệu cơ bản cho {symbol}: {exc}"


# ── News Analyst tools ────────────────────────────────────────────────────────

@tool
def get_stock_news(symbol: str, days: int = 5) -> str:
    """Lấy tin tức liên quan đến một mã cổ phiếu từ CafeF, VnExpress, CafeBiz.

    Args:
        symbol: Mã cổ phiếu (VCB, HPG, FPT...)
        days: Số ngày nhìn lại (mặc định 5)

    Returns:
        Chuỗi văn bản tổng hợp các bài tin tức kèm tiêu đề, nguồn, tóm tắt
    """
    as_of = _CURRENT_AS_OF_DATE
    try:
        from multiagents_trading_assistant.news_fetcher import get_stock_news as _fetch
        articles = _fetch(symbol, days=days, max_articles=15, before_date=as_of)
        if not articles:
            return f"Không tìm thấy tin tức nào cho {symbol} trong {days} ngày gần đây."
        lines = [f"=== TIN TỨC {symbol} ({days} ngày gần nhất) ===\n"]
        for i, a in enumerate(articles, 1):
            lines.append(
                f"{i}. [{a.get('source', '?')}] {a.get('title', '')}\n"
                f"   {a.get('summary', '')[:200]}\n"
                f"   {a.get('published_at', '')} | {a.get('url', '')}\n"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"Lỗi khi lấy tin tức cho {symbol}: {exc}"


# ── Flow Analyst tools ────────────────────────────────────────────────────────

@tool
def get_foreign_flow(symbol: str) -> str:
    """Lấy dữ liệu dòng tiền nhà đầu tư nước ngoài (NĐTNN) cho mã cổ phiếu.

    Args:
        symbol: Mã cổ phiếu (VCB, HPG, FPT...)

    Returns:
        JSON string với net_flow_5d, net_flow_20d, room_usage_pct, lịch sử 20 phiên
    """
    try:
        data = _provider().get_foreign_flow(symbol, n_days=20)
        # Filter flow_history nếu có as_of_date
        if _CURRENT_AS_OF_DATE and "flow_history" in data:
            data["flow_history"] = [
                row for row in data["flow_history"]
                if row.get("date", "9999") <= _CURRENT_AS_OF_DATE
            ]
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Lỗi khi lấy dòng tiền NĐTNN cho {symbol}: {exc}"


# ── Tool sets theo từng analyst ───────────────────────────────────────────────

MARKET_TOOLS = [get_stock_data, get_technical_indicators]
FUNDAMENTAL_TOOLS = [get_fundamentals]
NEWS_TOOLS = [get_stock_news]
FLOW_TOOLS = [get_foreign_flow]
