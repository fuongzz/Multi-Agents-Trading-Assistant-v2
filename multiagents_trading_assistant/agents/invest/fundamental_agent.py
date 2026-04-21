"""fundamental_agent.py — Sức khỏe tài chính & tăng trưởng cho Investment pipeline.

Model: Haiku. Port từ _legacy/agents/fa_agent.py.
Chỉ đánh giá financial_health và growth_quality — KHÔNG định giá
(valuation nội tại chuyển sang valuation_agent.py).
"""

import importlib.metadata  # noqa: F401
from datetime import datetime

from multiagents_trading_assistant.services.llm_service import run_agent_lite
from multiagents_trading_assistant import fetcher


_INDUSTRY_PE_MEDIAN: dict[str, float] = {
    "Ngân hàng": 12.0, "Bất động sản": 15.0, "Thép": 8.0,
    "Bán lẻ": 18.0, "Thực phẩm": 20.0, "Công nghệ": 22.0,
    "Dầu khí": 10.0, "Điện": 13.0, "Vận tải": 14.0,
    "Chứng khoán": 15.0, "Hóa chất": 11.0, "Xây dựng": 10.0,
    "default": 15.0,
}

_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích cơ bản chứng khoán Việt Nam (Investment pipeline — dài hạn).
Nhiệm vụ: đánh giá sức khỏe tài chính và chất lượng tăng trưởng. KHÔNG định giá nội tại (do valuation_agent xử lý).

Quy tắc bắt buộc:
- Trả về JSON hợp lệ DUY NHẤT
- financial_health: "KHỎE" (ROE≥15%, tăng trưởng dương) | "TRUNG_BÌNH" | "YẾU" | "UNKNOWN"
- growth_quality: "CAO" (profit_growth≥15%, bền vững) | "TRUNG_BÌNH" | "THẤP" | "UNKNOWN"
- vs_industry: "TỐT_HƠN" | "TRUNG_BÌNH" | "KÉM_HƠN" | "UNKNOWN"
- eps_growth_yoy: lấy từ profit_growth nếu không có EPS riêng

Output schema:
{
  "roe": <float|null>,
  "eps_growth_yoy": <float|null>,
  "revenue_growth": <float|null>,
  "financial_health": <str>,
  "growth_quality": <str>,
  "vs_industry": <str>,
  "fa_summary": <str — 1-2 câu tiếng Việt tóm tắt điểm mạnh/yếu>
}"""


def analyze(symbol: str, date: str | None = None) -> dict:
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    print(f"[fundamental_agent] {symbol} ({date})")

    fund = fetcher.get_fundamentals(symbol)
    if not fund or all(v is None for v in fund.values()):
        return _empty()

    industry = fund.get("industry") or "default"
    industry_pe = _get_industry_pe(industry)

    prompt = f"""Phân tích cơ bản (sức khỏe + tăng trưởng) mã {symbol} ngày {date}.

Ngành: {industry} | P/E median ngành: {industry_pe}
ROE: {_fmt(fund.get('roe'))}%
EPS: {_fmt(fund.get('eps'))} VNĐ
Tăng trưởng doanh thu YoY: {_fmt(fund.get('revenue_growth'))}%
Tăng trưởng lợi nhuận YoY: {_fmt(fund.get('profit_growth'))}%
P/E hiện tại: {_fmt(fund.get('pe'))} | P/B: {_fmt(fund.get('pb'))}

Đánh giá financial_health, growth_quality, vs_industry theo khung dài hạn (quý/năm).
Trả JSON theo schema."""

    try:
        result = run_agent_lite(prompt=prompt, system=_SYSTEM_PROMPT)
        result.setdefault("roe", fund.get("roe"))
        result.setdefault("eps_growth_yoy", fund.get("profit_growth"))
        result.setdefault("revenue_growth", fund.get("revenue_growth"))
        print(f"[fundamental_agent] {symbol} — health={result.get('financial_health')}, growth={result.get('growth_quality')}")
        return result
    except Exception as e:
        print(f"[fundamental_agent] LLM error: {e}")
        return _fallback(fund, industry_pe)


def get_industry_pe(industry: str) -> float:
    return _get_industry_pe(industry)


def _get_industry_pe(industry: str) -> float:
    for key, val in _INDUSTRY_PE_MEDIAN.items():
        if key.lower() in industry.lower() or industry.lower() in key.lower():
            return val
    return _INDUSTRY_PE_MEDIAN["default"]


def _fmt(val) -> str:
    return "N/A" if val is None else f"{val:.2f}"


def _empty() -> dict:
    return {
        "roe": None, "eps_growth_yoy": None, "revenue_growth": None,
        "financial_health": "UNKNOWN", "growth_quality": "UNKNOWN",
        "vs_industry": "UNKNOWN", "fa_summary": "Không có dữ liệu cơ bản.",
    }


def _fallback(fund: dict, industry_pe: float) -> dict:
    roe = fund.get("roe")
    health = "UNKNOWN"
    if roe is not None:
        health = "KHỎE" if roe >= 15 else ("TRUNG_BÌNH" if roe >= 8 else "YẾU")
    pg = fund.get("profit_growth")
    growth = "UNKNOWN"
    if pg is not None:
        growth = "CAO" if pg >= 15 else ("TRUNG_BÌNH" if pg >= 5 else "THẤP")
    return {
        "roe": roe, "eps_growth_yoy": pg,
        "revenue_growth": fund.get("revenue_growth"),
        "financial_health": health, "growth_quality": growth,
        "vs_industry": "UNKNOWN",
        "fa_summary": f"ROE {roe or 'N/A'}%, profit growth {pg or 'N/A'}%.",
    }
