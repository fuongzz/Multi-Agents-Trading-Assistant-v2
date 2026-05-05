"""Foreign flow analyst — VN-specific, thay thế Social Media Analyst."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from multiagents_trading_assistant.tradingagents_vn.tools import FLOW_TOOLS

_SYSTEM = """Bạn là chuyên gia phân tích dòng tiền nhà đầu tư nước ngoài (NĐTNN) trên HOSE.
Dòng tiền NĐTNN là tín hiệu quan trọng đặc thù của thị trường Việt Nam.

Phân tích dữ liệu và viết báo cáo gồm:
1. Xu hướng mua/bán ròng: 5 phiên và 20 phiên gần nhất
2. Tỷ lệ sử dụng room ngoại: >90% → NĐTNN gần như không thể mua thêm
3. Momentum: dòng tiền đang tăng tốc vào hay rút ra?
4. Diễn giải: tích lũy, phân phối hay trung lập?

Đánh giá:
- RẤT TÍCH CỰC: mua ròng liên tục + momentum tăng
- TÍCH CỰC: mua ròng nhưng chậm lại
- TRUNG LẬP: tín hiệu hỗn hợp
- TIÊU CỰC: bán ròng
- RẤT TIÊU CỰC: bán ròng mạnh + tăng tốc

Lưu ý: room ngoại >95% → không còn nhu cầu mua từ NĐTNN."""


def create_flow_analyst(llm):
    llm_with_tools = llm.bind_tools(FLOW_TOOLS)

    def node(state: dict) -> dict:
        symbol = state["company_of_interest"]
        trade_date = state["trade_date"]

        messages = list(state.get("messages", []))
        if not messages:
            messages = [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=(
                    f"Phân tích dòng tiền NĐTNN cho mã **{symbol}** tính đến ngày {trade_date}. "
                    f"Dùng get_foreign_flow để lấy dữ liệu."
                )),
            ]

        response = llm_with_tools.invoke(messages)
        report = response.content if not response.tool_calls else ""

        return {
            "messages": [response],
            "flow_report": report,
            "sender": "FlowAnalyst",
        }

    return node


def create_flow_analyst_clear():
    def node(state: dict) -> dict:
        messages = list(state.get("messages", []))
        report = state.get("flow_report", "")
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                report = msg.content
                break
        return {"messages": [], "flow_report": report}
    return node
