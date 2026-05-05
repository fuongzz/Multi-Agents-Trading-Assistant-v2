"""Risk management agents — field names giống repo gốc, debate state dùng str."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

_AGGRESSIVE_SYSTEM = """Bạn là chuyên gia quản lý rủi ro tích cực (aggressive) tại quỹ đầu tư Việt Nam.

RÀNG BUỘC CỨNG:
- VN KHÔNG có bán khống — chỉ tranh luận về vị thế MUA và tỷ trọng
- SL tối đa 5% dưới entry | T+3 cần tính vào kế hoạch exit | Lô tối thiểu 100cp

Lập luận cho cách tiếp cận tích cực HƠN:
- Đề xuất tỷ trọng lớn hơn nếu tin cậy cao
- Lập luận SL rộng hơn (nhưng ≤5%) nếu cơ bản mạnh
- Xác định upside bất cân xứng bị đánh giá thấp
- Thách thức giả định quá thận trọng

Cụ thể: trích dẫn giá thực tế, %. Viết bằng tiếng Việt."""

_CONSERVATIVE_SYSTEM = """Bạn là chuyên gia quản lý rủi ro thận trọng (conservative) tại quỹ đầu tư Việt Nam.

RÀNG BUỘC CỨNG:
- VN KHÔNG có bán khống — chỉ tranh luận về việc có nên MUA hay không
- Khi bán phải đảm bảo cp đã qua T+3 settlement
- Tránh mua khi: VNI ≤-3%/phiên, room ngoại >95%, mã tăng ≥5%/phiên

Lập luận cho cách tiếp cận thận trọng HƠN:
- Tỷ trọng nhỏ hơn hoặc mua dần từng phần
- Thắt chặt SL (ưu tiên 3-4% thay vì 5%)
- Xác định rủi ro đuôi đặc thù VN:
  (thay đổi chính sách đột ngột, biến động USD/VND, tạm dừng giao dịch, diện kiểm soát)
- Thách thức giả định quá lạc quan

Cụ thể: trích dẫn giá thực tế, %. Viết bằng tiếng Việt."""

_NEUTRAL_SYSTEM = """Bạn là chuyên gia quản lý rủi ro trung lập (neutral) tại quỹ đầu tư Việt Nam.

RÀNG BUỘC CỨNG:
- VN KHÔNG có bán khống — chỉ: MUA / NẮM GIỮ / BÁN cp đang có / TRÁNH MUA
- Ngưỡng tự động: VNI ≤-3%, room >95%, mã +5%, R:R <1.5

Tổng hợp aggressive và conservative thành đánh giá cân bằng:
- Tỷ trọng % NAV được khuyến nghị
- SL (≤5%) và TP (R:R ≥1.5) cụ thể
- Rủi ro quan trọng nhất cần theo dõi
- Điều kiện tăng hoặc cắt giảm vị thế

Căn cứ vào thực tế HOSE: thanh khoản, T+3, ±7%, tick size, lô 100cp. Viết bằng tiếng Việt."""

_PM_SYSTEM = """Bạn là Giám đốc Danh mục Đầu tư đưa ra quyết định cuối cùng.

RÀNG BUỘC TUYỆT ĐỐI:
❌ VN KHÔNG cho phép bán khống — không bao giờ đề xuất short position
✅ Hành động cuối: MUA / NẮM GIỮ / BÁN (cp đang nắm) / TRÁNH MUA / CHỜ

Checklist bắt buộc trước khi duyệt MUA:
- [ ] VNI không giảm ≥3%/phiên
- [ ] Room ngoại ≤95%
- [ ] Mã không tăng ≥5%/phiên
- [ ] R:R ≥1.5
- [ ] SL ≤5% dưới entry
- [ ] Không mua lại mã đã mua trong 2 ngày (T+2.5)

Quyết định cuối cùng:
1. **DUYỆT / TỪ CHỐI / ĐIỀU CHỈNH / CHỜ**
2. Hành động: MUA / NẮM GIỮ / BÁN / TRÁNH MUA / CHỜ
3. Tỷ trọng (% NAV)
4. Giá vào (ATO hoặc LO), SL, TP — làm tròn theo tick size
5. Lý do (1 đoạn)
6. Danh sách theo dõi hàng ngày (3-5 điểm cụ thể)

Sau phần phân tích, BẮT BUỘC kết thúc bằng JSON block sau (không bỏ sót field nào):

```json
{
  "action": "MUA|CHO|BAN|TRANH",
  "entry_zone": [<giá_thấp>, <giá_cao>],
  "stop_loss": <giá_sl>,
  "take_profit": <giá_tp>,
  "position_pct": <0.01_đến_0.10>,
  "confidence": <0.0_đến_1.0>,
  "reasons": ["<lý do 1>", "<lý do 2>"],
  "risks": ["<rủi ro 1>"]
}
```

Tất cả giá tính bằng nghìn đồng (VD: 50.5 nghĩa là 50,500 VND). Quyết đoán và cụ thể. Viết bằng tiếng Việt."""


def _trader_context(state: dict) -> str:
    return (
        f"Mã: {state['company_of_interest']} | Ngày: {state['trade_date']}\n\n"
        f"## Kế hoạch Trader\n{state.get('trader_investment_plan', 'Chưa có')}\n"
    )


def create_aggressive_analyst(llm):
    def node(state: dict) -> dict:
        risk = state.get("risk_debate_state", {})
        con_latest = risk.get("current_conservative_response", "")

        user_msg = _trader_context(state)
        if con_latest:
            user_msg += f"\n\n## Quan điểm Conservative (hãy phản bác)\n{con_latest}"
        user_msg += "\n\nHãy lập luận cho vị thế tích cực hơn."

        response = llm.invoke([SystemMessage(content=_AGGRESSIVE_SYSTEM), HumanMessage(content=user_msg)])
        content = response.content

        agg_history = (risk.get("aggressive_history", "") + "\n\n" + content).strip()
        full_history = (risk.get("history", "") + "\nAggressive: " + content).strip()

        return {
            "risk_debate_state": {
                **risk,
                "aggressive_history": agg_history,
                "history": full_history,
                "current_aggressive_response": content,
                "latest_speaker": "Aggressive",
                "count": risk.get("count", 0) + 1,
            },
            "sender": "AggressiveAnalyst",
        }
    return node


def create_conservative_analyst(llm):
    def node(state: dict) -> dict:
        risk = state.get("risk_debate_state", {})
        agg_latest = risk.get("current_aggressive_response", "")

        user_msg = _trader_context(state)
        if agg_latest:
            user_msg += f"\n\n## Quan điểm Aggressive (hãy phản bác)\n{agg_latest}"
        user_msg += "\n\nHãy lập luận cho vị thế thận trọng hơn."

        response = llm.invoke([SystemMessage(content=_CONSERVATIVE_SYSTEM), HumanMessage(content=user_msg)])
        content = response.content

        con_history = (risk.get("conservative_history", "") + "\n\n" + content).strip()
        full_history = (risk.get("history", "") + "\nConservative: " + content).strip()

        return {
            "risk_debate_state": {
                **risk,
                "conservative_history": con_history,
                "history": full_history,
                "current_conservative_response": content,
                "latest_speaker": "Conservative",
                "count": risk.get("count", 0) + 1,
            },
            "sender": "ConservativeAnalyst",
        }
    return node


def create_neutral_analyst(llm):
    def node(state: dict) -> dict:
        risk = state.get("risk_debate_state", {})
        user_msg = (
            _trader_context(state) + "\n\n"
            f"## Quan điểm Aggressive\n{risk.get('current_aggressive_response', 'Chưa có')}\n\n"
            f"## Quan điểm Conservative\n{risk.get('current_conservative_response', 'Chưa có')}\n\n"
            "Hãy tổng hợp thành đánh giá rủi ro cân bằng."
        )

        response = llm.invoke([SystemMessage(content=_NEUTRAL_SYSTEM), HumanMessage(content=user_msg)])
        content = response.content

        neu_history = (risk.get("neutral_history", "") + "\n\n" + content).strip()
        full_history = (risk.get("history", "") + "\nNeutral: " + content).strip()

        return {
            "risk_debate_state": {
                **risk,
                "neutral_history": neu_history,
                "history": full_history,
                "current_neutral_response": content,
                "latest_speaker": "Neutral",
                "count": risk.get("count", 0) + 1,
            },
            "sender": "NeutralAnalyst",
        }
    return node


def create_portfolio_manager(llm):
    def node(state: dict) -> dict:
        risk = state.get("risk_debate_state", {})
        user_msg = (
            f"Mã: **{state['company_of_interest']}** | Ngày: {state['trade_date']}\n\n"
            f"## Tổng hợp Nghiên cứu\n{state.get('investment_plan', 'Chưa có')}\n\n"
            f"## Kế hoạch Trader\n{state.get('trader_investment_plan', 'Chưa có')}\n\n"
            f"## Lịch sử tranh luận rủi ro\n{risk.get('history', 'Chưa có')}\n\n"
            "Hãy đưa ra quyết định danh mục cuối cùng."
        )

        response = llm.invoke([SystemMessage(content=_PM_SYSTEM), HumanMessage(content=user_msg)])

        return {
            "final_trade_decision": response.content,  # field giống repo gốc
            "sender": "PortfolioManager",
        }
    return node
