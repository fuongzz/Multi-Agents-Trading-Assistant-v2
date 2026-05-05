"""Market (technical) analyst — ToolNode pattern như repo gốc."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from multiagents_trading_assistant.tradingagents_vn.tools import MARKET_TOOLS

_SYSTEM = """Bạn là chuyên gia phân tích kỹ thuật thị trường chứng khoán Việt Nam (HOSE/HNX).
Nhiệm vụ của bạn là phân tích price action, khối lượng và các chỉ báo kỹ thuật.

Đặc thù thị trường Việt Nam:
- Biên độ giá: ±7% (HOSE), ±10% (HNX)
- T+3: không thể bán trong ngày mua
- Phiên ATO: 09:00–09:15 | Liên tục: 09:15–11:30 & 13:00–14:30 | ATC: 14:30–14:45

Dùng tools để lấy dữ liệu, sau đó viết báo cáo kỹ thuật gồm:
1. Xu hướng hiện tại (tăng/sideway/giảm)
2. Vùng hỗ trợ/kháng cự quan trọng
3. Tín hiệu momentum (RSI, MACD)
4. Biến động (Bollinger Bands, ATR)
5. Diễn biến khối lượng
6. Đánh giá tổng thể (Tích cực/Trung lập/Tiêu cực) kèm mức độ tin cậy

Trình bày bằng markdown rõ ràng, có bảng tóm tắt."""


def create_market_analyst(llm):
    llm_with_tools = llm.bind_tools(MARKET_TOOLS)

    def node(state: dict) -> dict:
        symbol = state["company_of_interest"]
        trade_date = state["trade_date"]
        past = state.get("past_context", "")

        system = _SYSTEM
        if past:
            system += f"\n\n## Lịch sử phân tích trước\n{past}"

        messages = list(state.get("messages", []))
        if not messages:
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=(
                    f"Phân tích kỹ thuật cho mã **{symbol}** ngày {trade_date}. "
                    f"Dùng get_stock_data và get_technical_indicators để lấy dữ liệu."
                )),
            ]

        response = llm_with_tools.invoke(messages)
        report = response.content if not response.tool_calls else ""

        return {
            "messages": [response],
            "market_report": report,
            "sender": "MarketAnalyst",
        }

    return node


def create_market_analyst_clear(report_key: str = "market_report"):
    """Node xóa messages sau khi analyst xong, lưu report vào state."""
    def node(state: dict) -> dict:
        # Tìm content cuối cùng từ messages để lưu report
        messages = list(state.get("messages", []))
        report = state.get(report_key, "")
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                report = msg.content
                break
        return {
            "messages": [],   # clear messages
            report_key: report,
        }
    return node
