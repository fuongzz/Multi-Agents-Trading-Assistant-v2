# Backtest Bias Discovery — 2026-05-14

## Tóm tắt

Phát hiện **same-bar-open exit lookahead bias** trong `multiagents_trading_assistant/backtest/live_pipeline.py:_edge_exit_decision()`. Tất cả backtest "live_pipeline_*" và "mvp_intraday_cached*" trước ngày 2026-05-14 đều bị bias này — số liệu **phóng đại ~1.8 lần** so với thực tế.

## Bản chất bias

Trong `_edge_exit_decision()` cũ:

```python
if low <= entry * (1 - stop_pct):                       # SL fired by INTRADAY low
    return "EDGE_STOP_LOSS", open_ * (1 - slippage)     # ← exit at SAME bar's OPEN
```

Khi SL bị quét trong phiên (low chạm trigger), code trả về giá thoát là **giá mở cửa của chính bar đó** — tức là **giá trước khi wick xảy ra**. Đây là look-ahead bias: trong thực tế, lệnh stop order chỉ khớp ở mức trigger (hoặc tệ hơn nếu gap), không phải ở giá mở cửa.

Áp dụng tương tự cho `EDGE_ATR_STOP`, `EDGE_TRAILING_ATR_STOP`, `EDGE_TAKE_PROFIT`.

## Tác động đo lường được

Backtest `core3` trên VN100, 2025-01 → 2026-05 (cùng dataset, cùng 68 trades):

| Số liệu | Reported (biased) | TRUE (fixed) | Bias |
|---|---|---|---|
| Total return | **+101%** | **+56%** | +45pp inflated |
| Sharpe | 3.28 | 2.05 | +1.23 inflated |
| MaxDD | -11% | **-17%** | -6pp hidden |
| WR | 63% | 62% | (giống) |
| Avg SL loss | -5.79% | **-8.10%** | +2.31pp inflated |
| Avg ATR_STOP loss | -4.22% | **-5.40%** | +1.18pp inflated |

## Patch áp dụng (2026-05-14)

Đã sửa `_edge_exit_decision` trong `live_pipeline.py` để dùng exit price thực tế:

- SL/ATR/trailing exit: `min(open, trigger_level) * (1 - slip)` — exit tại trigger hoặc gap-down open (cái nào tệ hơn)
- TP exit: `max(open, tp_level) * (1 - slip)` — exit tại TP hoặc gap-up open (cái nào tốt hơn, bounded)

Patch giữ nguyên **logic intraday trigger** của v1 (low/high touch) — chỉ sửa giá thoát.

## Các file backtest bị ảnh hưởng

Toàn bộ thư mục sau chứa số liệu **biased**:

```
backtest_results/live_pipeline_*/
backtest_results/mvp_intraday_cached/
backtest_results/mvp_intraday_cached_fixed_slot/
backtest_results/edge_lab_*/ (signal_quality_trades.csv)
```

**Không xoá** — giữ lại để compare history. Số liệu mới (sau 2026-05-14) ở `backtest_results/mvp_intraday_cached_v2/` đều dùng exit logic đã sửa.

## True benchmark mới (3 core strategies, VN100, 2025 YTD)

| Metric | Giá trị thực |
|---|---|
| Total return | +56% (16 tháng) |
| CAGR | ~40%/năm |
| Sharpe | 2.05 |
| MaxDD | -17% |
| Win rate | 62% |
| Trades | 68 |

So với VN-Index buy-hold cùng kỳ (~30%/16th, MDD -25%): outperform +26pp với MDD thấp hơn 8pp.

## Hành động tiếp theo

1. ✅ Patch `live_pipeline.py` — fix exit logic
2. ✅ Cập nhật CLAUDE.md với benchmark thực
3. ⚠️ Toàn bộ docs/comments có nhắc đến "+101% return" hay "Sharpe 3.28" cần được verify lại
4. 📝 Khi report số liệu backtest cho stakeholder: dùng số sau ngày 2026-05-14, neo vào benchmark thực

## Liên quan

- Script tái tạo: `scripts/compare_live_pipeline_intraday_cached_v2.py --mode v1_fixed_bias`
- Bug fix commit: (sẽ tham chiếu khi commit)
