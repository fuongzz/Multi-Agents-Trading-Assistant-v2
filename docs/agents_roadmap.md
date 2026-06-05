# Agents Roadmap — Nghiệp vụ chứng khoán mở rộng

> Kế hoạch xây 4 agent nghiệp vụ bổ sung cho hệ thống dual-pipeline.
> Tạo: 2026-06-03. Bám pattern hiện có (xem `CLAUDE.md`).

## Nguyên tắc thiết kế (bắt buộc tuân thủ)

- **Agent file**: module phẳng, hàm `analyze()`/`run()`, có `_SYSTEM_PROMPT`, `_empty_result()`, `_fallback_result()` — theo `agents/trade/flow_agent.py`.
- **LLM**: `run_agent_lite()` (Haiku) cho phân loại/tóm tắt; rule-based cho tín hiệu xác định. KHÔNG gọi Sonnet trừ khi cần lập luận dài.
- **Data**: thêm method vào `data/providers/base.py` (Protocol) → implement `vnstock_provider.py` → expose qua `fetcher.py` (cache JSON theo ngày + fallback an toàn). KHÔNG gọi `vnstock_data` trực tiếp trong agent.
- **Persistence**: bảng SQLite mới trong `database.py`, theo style sẵn có (`init_db()` + `_alter_columns`).
- **Tích hợp graph**: thêm node vào `orchestrator/trade_graph.py`/`investment_graph.py` hoặc gate sớm.
- **Anti-leak**: mọi truy vấn nhận `as_of_date` + `backtest_mode`, theo chuẩn `retrieve_trade_context`.
- **Output**: Discord webhook qua `services/output_service.py` (webhook env var riêng cho kênh mới).

---

## Phase 1 — 📓 Trade Journal / Post-mortem (LLM layer)  [ƯU TIÊN 1]

**Trạng thái nền**: phần định lượng ĐÃ TỒN TẠI.
- `agentic/post_trade_review.py` — classify_loss() 9 loại (rule-based).
- `database.py` — tự classify khi `close_position()`; `aggregate_loss_patterns()`, `reclassify_closed_losses()`, `get_closed_trades()`.
- `bob/simulator.py` — Bob đã reclassify + aggregate + `deprecate_from_loss_patterns()`, nhưng **chỉ in stdout**.

**Việc mới (chỉ là lớp phủ)**:
- `agents/review/trade_journal_agent.py` (Haiku): đọc closed trades + loss/win patterns → LLM sinh narrative tiếng Việt: top bài học, setup/regime hỏng, đề xuất chỉnh ngưỡng. Có fallback rule-based.
- `formatters/journal_embed.py` — Discord embed.
- `output_service.send_trade_journal()` — kênh `#trade-journal` (env `DISCORD_WEBHOOK_JOURNAL`).
- Wire vào `run_bob_strategy_meeting()` trong `pipeline_runner.py`: gọi journal SAU khi Bob xong (Bob đã có patterns).

**Độ khó**: THẤP. Không cần data mới.

---

## Phase 2 — 🏛️ Insider + 📅 Catalyst (chung nền DataProvider lớp Company)  [ƯU TIÊN 2]

**Foundation chung (làm trước)**: mở rộng DataProvider sang lớp `Company` của vnstock_data.
- Verify method bằng `show_api()` / `show_doc("Company")` trên `~/.venv`.
- `base.py`: `get_corporate_events(symbol)`, `get_insider_deals(symbol)`.
- `vnstock_provider.py`: implement qua `Company.events()` / `dividends()` / `insider_deals()` (tên method cần xác minh).
- `fetcher.py`: `get_corporate_events()`, `get_insider_deals()` (cache theo ngày, fallback `{}`).

### 2a — Insider & Major-Holder Agent
- `agents/trade/insider_agent.py` (rule-based, schema kiểu flow_agent): `{insider_net_30d, signal: ACCUMULATION/DISTRIBUTION/NEUTRAL, score 0-100, summary}`.
- Tích hợp: slot parallel thứ 5 trong `run_analysts()`; tái cân bằng synthesis weights (đề xuất khi code, cần backtest lại để giữ benchmark +56%).

### 2b — Catalyst / Event Agent
- `agents/shared/catalyst_agent.py` (rule-based khoảng-cách-ngày + LLM tóm tắt 1 dòng).
- Phân loại: `EARNINGS_BLACKOUT` (BCTC ≤ T-3), `EX_DATE` (GDKHQ), `DILUTION` (phát hành).
- Tích hợp: node `catalyst_gate` trong cả 2 graph; thêm Rule #11 vào `risk_trade.py`; feed `risk_invest.py`.

**Độ khó**: TRUNG BÌNH. Phụ thuộc field vnstock_data Company; fallback bắt buộc.

---

## Phase 3 — ⏱️ Intraday Execution Agent  [ƯU TIÊN 3]

**Bối cảnh**: đã có `intraday_trade_mvp_monitor.py`, `intraday_money_flow_monitor.py`, DNSE WS feed, và loạt script `run_flow_v2_intraday_*` + `build_vci_intraday_warehouse.py` (đang làm dở).

**Việc mới**:
- `agents/trade/intraday_execution_agent.py`: input 1 BUY đã qua risk (có `entry_zone`) → theo dõi intraday (DNSE WS ≤60s) → `{entry_now, suggested_price, wait_reason, expire_at}`.
- Tái dùng "same-clock intraday guard" từ script nghiên cứu.
- Tích hợp: nhánh [C] "entry timing" trong `session_monitor.py` (đã chạy mỗi 5 phút 09:00–14:35).
- Backtest: cần intraday warehouse; validate không introduce intraday look-ahead.

**Độ khó**: CAO. Làm cuối khi nền vững.

---

## Tiến độ

- [x] Phase 1 — Trade Journal ✅ (2026-06-03)
  - `agents/review/trade_journal_agent.py` (Haiku + fallback rule-based)
  - `formatters/journal_embed.py` (embed tím)
  - `output_service.send_trade_journal()` (env `DISCORD_WEBHOOK_JOURNAL`)
  - Wire: `pipeline_runner.run_bob_strategy_meeting()` gọi journal sau Bob (Thứ 6 20:00)
- [x] Phase 2 — Insider + Catalyst ✅ (2026-06-03)
  - **Foundation**: `DataProvider.get_corporate_events()` → `vnstock_provider` (VCI `Company.events()`) → `fetcher.get_corporate_events()` (cache + parse dates)
  - **Phát hiện**: `insider_trading()` KHÔNG khả dụng (VCI NotImplementedError, KBS rỗng). Thay bằng `events()` category `MAJOR_SHAREHOLDER_TRADING` — title chứa khối lượng, parse bằng regex.
  - `agents/shared/catalyst_agent.py` (rule-based, không LLM — gate tính xác định): EX_DATE/AGM/DIVIDEND/DILUTION + days-to-event + anti-leak
  - `agents/trade/insider_agent.py` (rule-based, schema kiểu flow): net Mua/Bán volume 30/90d → ACCUMULATION/DISTRIBUTION/NEUTRAL score 0-100
  - **Trade**: 2 agent vào `run_analysts` parallel; `risk_trade` Rule 11 (catalyst blackout→CHỜ, dilution→×0.7) + Rule 12 (insider distribution→×0.7)
  - **Invest**: catalyst vào `run_analysts`; `risk_invest` Rule 6 (dilution cảnh báo + ×0.7)
  - **CHƯA làm** (cố ý, cần backtest): đổi synthesis core weights cho insider — hiện chỉ là tín hiệu mềm ở risk
- [x] Phase 3 — Intraday Execution Agent ✅ (2026-06-03)
  - `agents/trade/intraday_execution_agent.py` (rule-based, formalize logic đã validate trong `scripts/run_flow_v2_intraday_entry_timing_experiment.py`)
  - Ngưỡng: ENTER_NOW khi `ret ≥ -0.5%` AND `price ≥ ma20×0.995`; CANCEL khi `drop ≤ -2.5%` OR `price < ma20×0.985`; chống mua đuổi khi giá > entry_high×1.005 → WAIT
  - Tích hợp: `session_monitor` nhánh **[C]** (mỗi 5 phút) — chạy cho BUY signals hôm nay CHƯA có vị thế; alert ENTER_NOW/CANCEL (dedup 1 lần/ngày/mã), WAIT chỉ log
  - Nguồn giá live: `_fetch_live_prices` (DNSE WS → HTTP); warehouse backtest: `data/intraday/vn100_1m_vci.parquet` + `vn100_5m.parquet`

> Sau mỗi phase: cập nhật `CLAUDE.md` (bảng agents, file structure) + dòng này.

---

## ✅ Hoàn tất toàn bộ 4 agent (2026-06-03)

Tất cả test với dữ liệu thật + compile + import graph OK. Việc nên làm tiếp (cần backtest, để follow-up):
- Đưa **insider** vào synthesis core weights (hiện chỉ tín hiệu mềm ở risk) — cần walk-forward.
- Backtest nhánh [C] entry timing trên warehouse 1m/5m so với ATO mù để định lượng cải thiện.
