"""valuation_agent.py — Định giá theo P/B và P/E dự phóng 1 năm.

Phương pháp (không dùng DCF):
  1. Dự phóng LNST 1 năm: EPS_next = EPS × (1 + g)
     g = profit_growth_yoy clamped [0%, 30%]

  2. P/E projection:
     intrinsic_PE = EPS_next × PE_median_ngành

  3. P/B projection:
     BVPS_now   = price / PB  (hoặc EPS / ROE nếu không có PB)
     projected_BVPS = BVPS_now + EPS_next × retention_ratio  (default 0.7)
     intrinsic_PB   = projected_BVPS × PB_median_ngành

  4. intrinsic = avg(intrinsic_PE, intrinsic_PB) nếu cả hai có, else lấy cái có
  5. MoS = (intrinsic − price) / intrinsic × 100
     RẺ: ≥ 30% | HỢP_LÝ: 0-30% | ĐẮT: < 0
"""

import importlib.metadata  # noqa: F401
from datetime import datetime

from multiagents_trading_assistant.services.llm_service import run_agent_lite
from multiagents_trading_assistant import fetcher
from multiagents_trading_assistant.agents.invest.fundamental_agent import get_industry_pe

_G_MIN  = 0.00   # không dự phóng âm
_G_MAX  = 0.30   # cap 30%
_RETENTION = 0.70  # 70% lợi nhuận giữ lại (30% chia cổ tức / phát hành thêm)

_INDUSTRY_PB_MEDIAN: dict[str, float] = {
    "Ngân hàng": 1.8,  "Bảo hiểm": 2.0, "Chứng khoán": 2.0,
    "Bất động sản": 1.5, "Xây dựng": 1.3,
    "Thép": 1.2, "Hóa chất": 1.5, "Vật liệu": 1.3,
    "Bán lẻ": 3.0, "Thực phẩm": 3.5, "Đồ uống": 4.0,
    "Công nghệ": 4.0, "Viễn thông": 2.5,
    "Dầu khí": 1.5, "Điện": 1.3, "Vận tải": 1.5,
    "Nông nghiệp": 2.0,
    "default": 2.0,
}

_SYSTEM_PROMPT = """Bạn là chuyên gia định giá cổ phiếu Việt Nam.
Nhiệm vụ: dựa trên kết quả định giá rule-based đã tính sẵn, viết valuation_summary ngắn gọn.

Quy tắc:
- Trả về JSON hợp lệ DUY NHẤT, không markdown
- valuation_summary: 1-2 câu tiếng Việt, nêu MoS, so sánh intrinsic vs giá hiện tại

Output schema:
{
  "valuation_summary": <str>
}"""


def analyze(symbol: str, date: str | None = None) -> dict:
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    print(f"[valuation_agent] {symbol} ({date})")

    fund = fetcher.get_fundamentals(symbol)
    if not fund:
        return _empty()

    industry  = fund.get("industry") or "default"
    pe_median = get_industry_pe(industry)
    pb_median = _get_industry_pb(industry)

    eps = fund.get("eps")          # VNĐ
    roe = fund.get("roe")          # % (e.g. 21.8)
    pg  = fund.get("profit_growth")  # % YoY
    pe  = fund.get("pe")           # P/E hiện tại
    pb  = fund.get("pb")           # P/B hiện tại

    # ── Giá hiện tại ──
    current_price = _get_current_price(symbol, fund)

    # ── Tốc độ tăng trưởng 1 năm tới ──
    g = max(_G_MIN, min(_G_MAX, (pg or 0) / 100))

    # ── Dự phóng EPS 1 năm ──
    eps_next = None
    if eps and eps > 0:
        eps_next = round(eps * (1 + g), 0)

    # ── P/E projection ──
    intrinsic_pe = None
    if eps_next and pe_median:
        intrinsic_pe = round(eps_next * pe_median, 0)

    # ── P/B projection ──
    intrinsic_pb = None
    bvps_now = None
    if current_price and current_price > 0 and pb and pb > 0:
        bvps_now = current_price / pb
    elif eps and eps > 0 and roe and roe > 0:
        bvps_now = eps / (roe / 100)   # ROE = EPS/BVPS → BVPS = EPS/ROE

    if bvps_now and bvps_now > 0 and eps_next and pb_median:
        projected_bvps = bvps_now + eps_next * _RETENTION
        intrinsic_pb = round(projected_bvps * pb_median, 0)

    # ── Intrinsic = trung bình PE + PB (ưu tiên cả hai) ──
    if intrinsic_pe and intrinsic_pb:
        intrinsic = round((intrinsic_pe + intrinsic_pb) / 2, 0)
    elif intrinsic_pe:
        intrinsic = intrinsic_pe
    elif intrinsic_pb:
        intrinsic = intrinsic_pb
    else:
        intrinsic = None

    # ── Margin of Safety ──
    mos = None
    if intrinsic and intrinsic > 0 and current_price and current_price > 0:
        mos = round((intrinsic - current_price) / intrinsic * 100, 1)

    valuation = "UNKNOWN"
    if mos is not None:
        valuation = "RẺ" if mos >= 30 else ("HỢP_LÝ" if mos >= 0 else "ĐẮT")

    # ── LLM viết summary ──
    prompt = f"""Định giá {symbol} ngày {date}.

Ngành: {industry} | PB median ngành: {pb_median} | PE median ngành: {pe_median}
Giá hiện tại: {_fmt(current_price)} VNĐ | PB hiện tại: {_fmt(pb)} | PE hiện tại: {_fmt(pe)}
EPS ttm: {_fmt(eps)} VNĐ | ROE: {_fmt(roe)}% | Tăng trưởng LN dự phóng: {g:.0%}

Kết quả định giá:
  EPS dự phóng 1 năm: {_fmt(eps_next)} VNĐ (g={g:.0%})
  BVPS hiện tại: {_fmt(bvps_now)} VNĐ → BVPS dự phóng: {_fmt(bvps_now + eps_next * _RETENTION if bvps_now and eps_next else None)} VNĐ
  Intrinsic P/E ({pe_median}×): {_fmt(intrinsic_pe)} VNĐ
  Intrinsic P/B ({pb_median}×): {_fmt(intrinsic_pb)} VNĐ
  Intrinsic (trung bình): {_fmt(intrinsic)} VNĐ
  Margin of Safety: {f'{mos:+.1f}%' if mos is not None else 'N/A'}
  Valuation: {valuation}

Viết valuation_summary 1-2 câu. Trả JSON."""

    try:
        result = run_agent_lite(prompt=prompt, system=_SYSTEM_PROMPT)
        summary = result.get("valuation_summary", f"Intrinsic {_fmt(intrinsic)} VNĐ, MoS {mos}%.")
    except Exception as e:
        print(f"[valuation_agent] LLM error: {e}")
        summary = f"Intrinsic {_fmt(intrinsic)} VNĐ (PE×{pe_median}+PB×{pb_median}), MoS {mos}%."

    print(f"[valuation_agent] {symbol} — PE={_fmt(intrinsic_pe)}, PB={_fmt(intrinsic_pb)}, intrinsic={_fmt(intrinsic)}, MoS={mos}%, val={valuation}")
    return {
        "intrinsic_value": intrinsic,
        "pe_fair":         intrinsic_pe,
        "pb_fair":         intrinsic_pb,
        "eps_next":        eps_next,
        "bvps_projected":  round(bvps_now + eps_next * _RETENTION, 0) if bvps_now and eps_next else None,
        "growth_used":     round(g * 100, 1),
        "margin_of_safety": mos,
        "valuation":       valuation,
        "valuation_summary": summary,
    }


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _get_current_price(symbol: str, fund: dict) -> float | None:
    try:
        pb_df = fetcher.get_price_board([symbol])
        if not pb_df.empty:
            row = pb_df[pb_df["symbol"] == symbol]
            if not row.empty:
                p = float(row.iloc[0].get("price") or 0)
                if p > 0:
                    return p
    except Exception:
        pass
    eps = fund.get("eps")
    pe  = fund.get("pe")
    if eps and pe:
        return eps * pe
    return None


def _get_industry_pb(industry: str) -> float:
    ind_lower = industry.lower()
    for key, val in _INDUSTRY_PB_MEDIAN.items():
        if key.lower() in ind_lower or ind_lower in key.lower():
            return val
    return _INDUSTRY_PB_MEDIAN["default"]


def _fmt(val) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float) and val != int(val):
        return f"{val:,.2f}"
    return f"{val:,.0f}"


def _empty() -> dict:
    return {
        "intrinsic_value": None, "pe_fair": None, "pb_fair": None,
        "eps_next": None, "bvps_projected": None, "growth_used": None,
        "margin_of_safety": None, "valuation": "UNKNOWN",
        "valuation_summary": "Không có dữ liệu định giá.",
    }
