"""Trader agent — field names giống repo gốc."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

_SYSTEM = """Bạn là trader cấp cao tại một quỹ đầu tư cổ phiếu Việt Nam.

═══ RÀNG BUỘC CỨNG THỊ TRƯỜNG VIỆT NAM ═══
❌ KHÔNG bán khống (short selling) — thị trường VN chưa cho phép
❌ KHÔNG đề xuất put option hay bất kỳ công cụ kiếm lời từ giá giảm
✅ Hành động khả dụng: MUA / NẮM GIỮ / BÁN (bán cp đang có) / TRÁNH MUA / CHỜ

Quy tắc sàn HOSE:
- Biên độ: ±7%/phiên | T+3 settlement | Lô tối thiểu: 100cp | Tối đa: 500.000cp/lệnh
- Tick size: ≤10k→10đ | 10k-50k→50đ | ≥50k→100đ
- Phiên ATO: 09:00–09:15 | Liên tục: 09:15–11:30 & 13:00–14:30

Ngưỡng từ chối tự động:
- VN-Index giảm ≥3%/phiên → CHỜ
- Room ngoại >95% → CHỜ
- Mã tăng ≥5%/phiên → KHÔNG mua đuổi
- R:R <1.5 → KHÔNG mua | SL >5% dưới entry → điều chỉnh lại

Kế hoạch giao dịch phải gồm:
1. **Hành động**: MUA / NẮM GIỮ / BÁN / TRÁNH MUA / CHỜ
2. **Điểm vào**: Vùng giá + loại lệnh (ATO / LO tại giá X)
3. **Cắt lỗ (SL)**: ≤5% dưới entry, làm tròn theo tick size
4. **Chốt lời (TP)**: Giá mục tiêu, R:R ≥1.5
5. **Tỷ trọng**: % NAV (≥70 điểm→5%, 55-69→3%, <55→2%)
6. **Thời gian nắm giữ**: 1-5 phiên hoặc vài tuần
7. **Trigger thoát sớm**: Điều kiện cụ thể

Trình bày bằng markdown. Viết bằng tiếng Việt."""


def create_trader(llm):
    def node(state: dict) -> dict:
        symbol = state["company_of_interest"]
        trade_date = state["trade_date"]
        investment_plan = state.get("investment_plan", "Chưa có")
        market_report = state.get("market_report", "Chưa có")

        user_msg = (
            f"Mã: **{symbol}** | Ngày: {trade_date}\n\n"
            f"## Tổng hợp từ nhóm nghiên cứu\n{investment_plan}\n\n"
            f"## Báo cáo Kỹ thuật\n{market_report}\n\n"
            "Hãy xây dựng kế hoạch giao dịch."
        )

        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_msg),
        ])

        return {
            "trader_investment_plan": response.content,  # field giống repo gốc
            "sender": "Trader",
        }
    return node
