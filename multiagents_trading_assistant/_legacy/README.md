# Legacy — Archived Code

Các file trong thư mục này là **kiến trúc single-pipeline cũ** (blended investment + speculation),
được archive khi refactor sang dual-pipeline theo `agents.md` (2026-04-19).

## Vì sao archive chứ không xóa
- Nhiều logic kỹ thuật đã được **port sang** `agents/trade/`, `screener/trade_screener.py`, `nodes/risk_trade.py` và `nodes/trader_trade.py`.
- Một số prompt và heuristic có giá trị tham khảo khi fine-tune pipeline mới.
- Có thể cần rollback nếu pipeline mới gặp vấn đề nghiêm trọng trong quá trình ổn định.

## Ánh xạ Port

| Legacy | Thay thế |
|---|---|
| `orchestrator.py` | `orchestrator/trade_graph.py` |
| `screener.py` | `screener/trade_screener.py` |
| `risk_manager.py` | `nodes/risk_trade.py` |
| `scheduler.py` + `discord_bot.py` | `orchestrator/pipeline_runner.py` + `services/output_service.py` |
| `agents/ptkt_agent.py` | `agents/trade/technical_agent.py` |
| `agents/foreign_flow_agent.py` | `agents/trade/flow_agent.py` |
| `agents/fa_agent.py` | `agents/invest/fundamental_agent.py` + `agents/invest/valuation_agent.py` |
| `agents/macro_agent.py` | `agents/invest/macro_agent.py` |
| `research/debate_agent.py` | `agents/invest/debate_agent.py` (thesis dài hạn) |
| `research/sentiment_agent.py` | `agents/trade/sentiment_agent.py` |
| `trader/trader_agent.py` | `nodes/trader_trade.py` (có SL/TP) |
| `dashboard.py` | — (chưa port, có thể cần trong tương lai) |

**Không import từ `_legacy/` trong code mới.**
