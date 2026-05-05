"""Fundamental analyst — ToolNode pattern như repo gốc."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from multiagents_trading_assistant.tradingagents_vn.tools import FUNDAMENTAL_TOOLS

_SYSTEM = """Bạn là chuyên gia phân tích cơ bản các công ty niêm yết tại Việt Nam.
Nhiệm vụ của bạn là đánh giá sức khỏe tài chính, định giá và triển vọng tăng trưởng.

Tập trung vào:
1. Định giá: P/E so với trung vị ngành, P/B so với lịch sử
2. Khả năng sinh lời: xu hướng ROE, biên lợi nhuận, tăng trưởng EPS
3. Tăng trưởng: doanh thu và lợi nhuận (YoY, QoQ)
4. Bối cảnh ngành trong môi trường vĩ mô Việt Nam hiện tại

Lưu ý đặc thù ngành:
- Ngân hàng: NIM, tỷ lệ NPL, tăng trưởng tín dụng
- Bất động sản: quỹ đất, hàng tồn kho, dòng tiền
- Sản xuất/xuất khẩu: đơn hàng, tác động tỷ giá USD/VND
- Tiêu dùng: chu kỳ tiêu dùng nội địa

QUAN TRỌNG — Thị trường VN KHÔNG cho phép bán khống.
Đánh giá tổng thể: MUA MẠNH / MUA / NẮM GIỮ / BÁN (cp đang nắm) / TRÁNH MUA"""


def create_fundamental_analyst(llm):
    llm_with_tools = llm.bind_tools(FUNDAMENTAL_TOOLS)

    def node(state: dict) -> dict:
        symbol = state["company_of_interest"]
        trade_date = state["trade_date"]

        messages = list(state.get("messages", []))
        if not messages:
            messages = [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=(
                    f"Phân tích cơ bản mã **{symbol}** ngày {trade_date}. "
                    f"Dùng get_fundamentals để lấy dữ liệu."
                )),
            ]

        response = llm_with_tools.invoke(messages)
        report = response.content if not response.tool_calls else ""

        return {
            "messages": [response],
            "fundamentals_report": report,
            "sender": "FundamentalAnalyst",
        }

    return node


def create_fundamental_analyst_clear():
    def node(state: dict) -> dict:
        messages = list(state.get("messages", []))
        report = state.get("fundamentals_report", "")
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                report = msg.content
                break
        return {"messages": [], "fundamentals_report": report}
    return node
