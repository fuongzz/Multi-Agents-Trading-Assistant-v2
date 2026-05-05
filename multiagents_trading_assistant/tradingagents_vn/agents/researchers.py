"""Bull/Bear researcher agents và Research Manager — debate state dùng str như repo gốc."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

_BULL_SYSTEM = """Bạn là nhà nghiên cứu đầu tư lạc quan (bull) chuyên về thị trường chứng khoán Việt Nam.
Nhiệm vụ: xây dựng luận điểm tăng giá mạnh mẽ, có căn cứ dữ liệu.

RÀNG BUỘC — Thị trường VN KHÔNG có bán khống. Chỉ lập luận về việc NÊN MUA hay không.

Cấu trúc lập luận:
1. Luận điểm cốt lõi (1-2 câu)
2. Các catalyst chính (3-5 bullet)
3. Phản bác luận điểm bear (nếu bear đã lập luận trước)
4. Vùng mục tiêu giá và khung thời gian

Hãy cụ thể và định lượng. Trích dẫn số liệu thực tế. Viết bằng tiếng Việt."""

_BEAR_SYSTEM = """Bạn là nhà nghiên cứu đầu tư bi quan (bear) chuyên về thị trường chứng khoán Việt Nam.
Nhiệm vụ: xây dựng luận điểm tại sao KHÔNG NÊN MUA cổ phiếu này.

RÀNG BUỘC — Thị trường VN KHÔNG có bán khống, không có put option.
Chỉ lập luận: TRÁNH MUA hoặc BÁN cổ phiếu đang nắm giữ. Không đề xuất short position.

Cấu trúc lập luận:
1. Luận điểm rủi ro cốt lõi (1-2 câu)
2. Các rủi ro chính (3-5 bullet)
3. Phản bác luận điểm bull (nếu bull đã lập luận trước)
4. Kịch bản giảm giá và trigger rủi ro

Hãy cụ thể và định lượng. Trích dẫn số liệu thực tế. Viết bằng tiếng Việt."""

_MANAGER_SYSTEM = """Bạn là Trưởng phòng Nghiên cứu, tổng hợp tranh luận bull/bear cho cổ phiếu Việt Nam.

RÀNG BUỘC — VN không có bán khống. Khuyến nghị chỉ gồm: MUA / NẮM GIỮ / BÁN (cp đang nắm) / TRÁNH MUA.

Viết bản tổng hợp nghiên cứu cuối cùng:
1. Các điểm đồng thuận giữa bull và bear
2. Lập luận nào thuyết phục hơn và lý do
3. Khuyến nghị: MUA / NẮM GIỮ / BÁN / TRÁNH MUA — kèm mức độ tin cậy (Cao/Trung bình/Thấp)
4. Rủi ro quan trọng nhất cần theo dõi
5. Cách tiếp cận đề xuất (mua dần, chờ pullback, tránh, bán giải ngân nếu đang nắm)

Hãy quyết đoán. Viết bằng tiếng Việt."""


def _context(state: dict) -> str:
    return (
        f"Mã: {state['company_of_interest']} | Ngày: {state['trade_date']}\n\n"
        f"## Báo cáo Kỹ thuật\n{state.get('market_report', 'Chưa có')}\n\n"
        f"## Báo cáo Cơ bản\n{state.get('fundamentals_report', 'Chưa có')}\n\n"
        f"## Báo cáo Tin tức/Sentiment\n{state.get('sentiment_report', 'Chưa có')}\n\n"
        f"## Báo cáo Dòng tiền NĐTNN\n{state.get('flow_report', 'Chưa có')}\n"
    )


def create_bull_researcher(llm):
    def node(state: dict) -> dict:
        debate = state.get("investment_debate_state", {})
        bear_latest = debate.get("current_response", "")

        user_msg = _context(state)
        if bear_latest:
            user_msg += f"\n\n## Luận điểm mới nhất của Bear (hãy phản bác)\n{bear_latest}"
        user_msg += "\n\nHãy đưa ra luận điểm tăng giá của bạn."

        response = llm.invoke([
            SystemMessage(content=_BULL_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        content = response.content

        bull_history = debate.get("bull_history", "")
        full_history = debate.get("history", "")
        bull_history = (bull_history + "\n\n" + content).strip()
        full_history = (full_history + "\nBull: " + content).strip()

        return {
            "investment_debate_state": {
                **debate,
                "bull_history": bull_history,
                "history": full_history,
                "current_response": content,
                "count": debate.get("count", 0) + 1,
            },
            "sender": "BullResearcher",
        }
    return node


def create_bear_researcher(llm):
    def node(state: dict) -> dict:
        debate = state.get("investment_debate_state", {})
        bull_latest = debate.get("current_response", "")

        user_msg = _context(state)
        if bull_latest:
            user_msg += f"\n\n## Luận điểm mới nhất của Bull (hãy phản bác)\n{bull_latest}"
        user_msg += "\n\nHãy đưa ra luận điểm giảm giá của bạn."

        response = llm.invoke([
            SystemMessage(content=_BEAR_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        content = response.content

        bear_history = debate.get("bear_history", "")
        full_history = debate.get("history", "")
        bear_history = (bear_history + "\n\n" + content).strip()
        full_history = (full_history + "\nBear: " + content).strip()

        return {
            "investment_debate_state": {
                **debate,
                "bear_history": bear_history,
                "history": full_history,
                "current_response": content,
                "count": debate.get("count", 0) + 1,
            },
            "sender": "BearResearcher",
        }
    return node


def create_research_manager(llm):
    def node(state: dict) -> dict:
        debate = state.get("investment_debate_state", {})
        user_msg = (
            f"{_context(state)}\n\n"
            f"## Lịch sử tranh luận đầy đủ\n{debate.get('history', 'Chưa có')}\n\n"
            "Hãy viết bản tổng hợp nghiên cứu và khuyến nghị cuối cùng."
        )

        response = llm.invoke([
            SystemMessage(content=_MANAGER_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        synthesis = response.content

        return {
            "investment_plan": synthesis,        # field giống repo gốc
            "investment_debate_state": {
                **debate,
                "judge_decision": synthesis,
            },
            "sender": "ResearchManager",
        }
    return node
