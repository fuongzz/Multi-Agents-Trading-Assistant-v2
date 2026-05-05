"""News analyst — thay Social Media Analyst, dùng vnstock_news qua news_fetcher."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from multiagents_trading_assistant.tradingagents_vn.tools import NEWS_TOOLS

_SYSTEM = """Bạn là chuyên gia phân tích tin tức và tâm lý thị trường chứng khoán Việt Nam.
Nguồn tin: CafeF, VnExpress, CafeBiz, Vietstock (qua vnstock_news).

Phân tích các bài báo và viết báo cáo gồm:
1. Tin tức nổi bật liên quan trực tiếp đến doanh nghiệp (kết quả kinh doanh, nhân sự, M&A...)
2. Tin ngành và vĩ mô ảnh hưởng đến cổ phiếu
3. Tâm lý thị trường xung quanh mã này: tích cực / trung lập / tiêu cực
4. Rủi ro tin tức ngắn hạn cần lưu ý
5. Đánh giá tổng thể sentiment: RẤT TÍCH CỰC / TÍCH CỰC / TRUNG LẬP / TIÊU CỰC / RẤT TIÊU CỰC

Lưu ý: chỉ đánh giá tin tức, không đề xuất giao dịch tại bước này."""


def create_news_analyst(llm):
    llm_with_tools = llm.bind_tools(NEWS_TOOLS)

    def node(state: dict) -> dict:
        symbol = state["company_of_interest"]
        trade_date = state["trade_date"]

        messages = list(state.get("messages", []))
        if not messages:
            messages = [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=(
                    f"Phân tích tin tức và sentiment cho mã **{symbol}** tính đến ngày {trade_date}. "
                    f"Dùng get_stock_news để lấy tin tức gần đây."
                )),
            ]

        response = llm_with_tools.invoke(messages)
        report = response.content if not response.tool_calls else ""

        return {
            "messages": [response],
            "sentiment_report": report,
            "sender": "NewsAnalyst",
        }

    return node


def create_news_analyst_clear():
    def node(state: dict) -> dict:
        messages = list(state.get("messages", []))
        report = state.get("sentiment_report", "")
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                report = msg.content
                break
        return {"messages": [], "sentiment_report": report}
    return node
