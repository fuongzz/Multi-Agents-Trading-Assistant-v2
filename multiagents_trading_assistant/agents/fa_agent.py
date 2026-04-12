"""
fa_agent.py — Phân tích cơ bản (FA) cho một mã cổ phiếu.

Model: Haiku (run_agent_lite) — nhanh + prompt caching
Input:  symbol, fundamentals dict từ fetcher
Output: dict theo schema agents.md #3
"""

import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
from datetime import datetime

from multiagents_trading_assistant.agent import run_agent_lite
from multiagents_trading_assistant import fetcher


# ── Median P/E theo ngành (VN market, cập nhật định kỳ) ──
# Dùng làm benchmark khi không có industry data từ API
_INDUSTRY_PE_MEDIAN: dict[str, float] = {
    "Ngân hàng":             12.0,
    "Bất động sản":          15.0,
    "Thép":                   8.0,
    "Bán lẻ":                18.0,
    "Thực phẩm & Đồ uống":  20.0,
    "Công nghệ":             22.0,
    "Dầu khí":               10.0,
    "Điện":                  13.0,
    "Vận tải":               14.0,
    "Chứng khoán":           15.0,
    "Hóa chất":              11.0,
    "Xây dựng":              10.0,
    "default":               15.0,
}

_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích cơ bản (FA) chứng khoán Việt Nam.
Nhiệm vụ: đánh giá sức khỏe tài chính và định giá cổ phiếu dựa trên dữ liệu được cung cấp.

Quy tắc bắt buộc:
- Trả về JSON hợp lệ DUY NHẤT, không có text thừa
- Tất cả field phải có giá trị (null nếu không có dữ liệu)
- valuation: "RẺ" (P/E < 70% median ngành) | "HỢP_LÝ" (70-130%) | "ĐẮT" (>130%) | "UNKNOWN"
- vs_industry: "TỐT_HƠN" | "TRUNG_BÌNH" | "KÉM_HƠN" | "UNKNOWN"
- financial_health: "KHỎE" (ROE>15%, tăng trưởng dương) | "TRUNG_BÌNH" | "YẾU" | "UNKNOWN"

Output schema:
{
  "pe_ratio": <float|null>,
  "pb_ratio": <float|null>,
  "roe": <float|null>,
  "eps_growth_yoy": <float|null>,
  "valuation": <str>,
  "vs_industry": <str>,
  "financial_health": <str>,
  "fa_summary": <str — 1-2 câu tiếng Việt tóm tắt điểm mạnh/yếu FA>
}"""


def analyze(symbol: str, date: str | None = None) -> dict:
    """
    Chạy FA agent cho một mã.

    Args:
        symbol: Mã cổ phiếu (VD: "VNM")
        date:   Ngày phân tích (YYYY-MM-DD), mặc định hôm nay

    Returns:
        dict theo schema fa_agent (agents.md #3)
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    print(f"[fa_agent] Bắt đầu phân tích cơ bản {symbol} ({date})...")

    fund = fetcher.get_fundamentals(symbol)
    if not fund or all(v is None for v in fund.values()):
        print(f"[fa_agent] Không lấy được fundamentals cho {symbol}.")
        return _empty_result()

    # Lấy median ngành
    industry = fund.get("industry") or "default"
    industry_pe = _get_industry_pe(industry)

    prompt = f"""Phân tích cơ bản mã {symbol} ngày {date}.

=== Dữ liệu tài chính ===
Ngành:                {industry}
P/E hiện tại:         {_fmt(fund.get('pe'))}
P/B hiện tại:         {_fmt(fund.get('pb'))}
ROE (%):              {_fmt(fund.get('roe'))}
EPS (VNĐ):            {_fmt(fund.get('eps'))}
Tăng trưởng DT YoY:   {_fmt_pct(fund.get('revenue_growth'))}
Tăng trưởng LN YoY:   {_fmt_pct(fund.get('profit_growth'))}

=== Benchmark ngành ===
P/E median ngành "{industry}": {industry_pe}

Đánh giá:
- valuation dựa trên P/E so với median ngành
- vs_industry dựa trên ROE, tăng trưởng so với benchmark ngành
- financial_health dựa trên ROE và tăng trưởng lợi nhuận

Trả về JSON theo schema đã định."""

    try:
        result = run_agent_lite(prompt=prompt, system=_SYSTEM_PROMPT)
        print(f"[fa_agent] {symbol} — valuation={result.get('valuation')}, "
              f"health={result.get('financial_health')}")
        return result
    except Exception as e:
        print(f"[fa_agent] Lỗi gọi LLM cho {symbol}: {e}")
        return _fallback_result(fund, industry_pe)


def _get_industry_pe(industry: str) -> float:
    """Lấy P/E median ngành. Fallback về default nếu không tìm thấy."""
    for key, val in _INDUSTRY_PE_MEDIAN.items():
        if key.lower() in industry.lower() or industry.lower() in key.lower():
            return val
    return _INDUSTRY_PE_MEDIAN["default"]


def _fmt(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}"


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f}%"


def _empty_result() -> dict:
    return {
        "pe_ratio": None, "pb_ratio": None, "roe": None, "eps_growth_yoy": None,
        "valuation": "UNKNOWN", "vs_industry": "UNKNOWN",
        "financial_health": "UNKNOWN",
        "fa_summary": "Không có dữ liệu cơ bản.",
    }


def _fallback_result(fund: dict, industry_pe: float) -> dict:
    """Tính valuation đơn giản khi LLM fail."""
    pe = fund.get("pe")
    roe = fund.get("roe")

    valuation = "UNKNOWN"
    if pe and industry_pe:
        ratio = pe / industry_pe
        valuation = "RẺ" if ratio < 0.7 else ("HỢP_LÝ" if ratio <= 1.3 else "ĐẮT")

    health = "UNKNOWN"
    if roe is not None:
        health = "KHỎE" if roe >= 15 else ("TRUNG_BÌNH" if roe >= 8 else "YẾU")

    return {
        "pe_ratio": pe,
        "pb_ratio": fund.get("pb"),
        "roe": roe,
        "eps_growth_yoy": fund.get("profit_growth"),
        "valuation": valuation,
        "vs_industry": "UNKNOWN",
        "financial_health": health,
        "fa_summary": f"P/E {pe or 'N/A'}, ROE {roe or 'N/A'}%.",
    }
