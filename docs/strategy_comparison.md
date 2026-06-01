# Strategy Comparison Cheat Sheet

Last updated: 2026-05-23

Use this file as the quick lookup table when there are too many MVP/Flow
variants to remember. Default operating/backtest universe is VN100. Keep VN30,
VN100, and Liquid150 results separate.

## Portfolio-Level Strategies

| Priority | Strategy / Variant | Status | Universe | Period | Return | Sharpe | Max DD | Best Use | Notes |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 1 | `mvp_p5` | Active HTML | VN100 | Current dry-run only | N/A | N/A | N/A | Main Core MVP sleeve with 5 slots | Uses the 9 core MVP strategies; current close 2026-05-22 has 0 candidates. |
| 2 | `mvp_p4` | Active HTML | VN100 | Current dry-run only | N/A | N/A | N/A | More concentrated Core MVP sleeve with 4 slots | Same 9 core MVP strategies; current close 2026-05-22 has 0 candidates. |
| 3 | `flow_v2` current config | Active HTML + verified | VN100 | 2025-01-01 to 2026-05-21 | 98.84% | 2.169 | -11.03% | Default Flow V2 sleeve | positions=2, rebalance=10d, `risk_on_or_strong_neutral`, `clean_flow`, `balanced`. |
| 4 | Flow V2 high-RS balanced | Research/grid | Liquid150 | 2025-01-01 to 2026-05-22 | 80.32% | 1.505 | -16.64% | Liquid150 comparison only | positions=2, rebalance=10d, `high_rs`, `balanced`; candidate retune, not active. |
| 5 | MVP p5 entry-market-gate | Historical benchmark | VN100 | 2025-01-01 to 2026-05-14 | 99.37% | 2.898 | -11.40% | Best old Core MVP p5 reference | VN100 only. |
| 6 | MVP p5 soft-gate | Historical benchmark | VN100 | 2025-01-01 to 2026-05-16 | 98.48% | 2.885 | -11.40% | Old robust Core MVP reference | VN100 only. |
| 7 | Flow V2 tune5 best | Research/grid | VN100 | 2025-01-01 to 2026-05-19 | 129.82% | 2.571 | -11.74% | Candidate for future retune | positions=2, rebalance=8d, `loose_flow`, `momentum_heavy`; not active. |
| 8 | Cycle-first p4 | Research/grid | VN100 | 2025-01-01 to 2026-05-19 | 114.00% | 3.055 | -10.19% | Candidate for future Core MVP retune | max_positions=4; not active HTML. |
| 9 | Core MVP9 + inverse volatility | Research allocator | VN100 | 2025-01-01 to 2026-05-22 | 71.71% | 2.166 | -9.12% | Risk overlay reference | Allocator comparison, not current active sleeve. |
| 10 | Core MVP9 + equal weight | Research allocator | VN100 | 2025-01-01 to 2026-05-22 | 70.37% | 2.165 | -9.12% | Baseline allocator reference | VN100 only. |

Latest direct comparison report:
`reports/mvp_vn100_vs_liquid150_2025now/README.md`.

## Core MVP Strategy Set

These 9 strategy IDs are the production Core MVP pool used by `mvp_p5` and
`mvp_p4`.

| Strategy | Style | Stop | Target | Max Hold | Quick Meaning |
|---|---|---:|---:|---:|---|
| `leader_pullback_market_regime_v3` | Pullback / regime-aware | 8% | 25% | 60 | Buy leading stocks pulling back in a supportive market regime. |
| `leader_pullback_market_healthy_v2` | Pullback / healthy market | 8% | 25% | 60 | Similar to leader pullback, with healthier market filters. |
| `breakout_55_smt_v1` | Breakout | 9% | 40% | 60 | 55-day breakout confirmed by smart money / money cycle. |
| `money_cycle_reset_smt_confirm` | Reset / continuation | 8% | 25% | 60 | Stock resets after cycle pressure and gets smart-money confirmation. |
| `smart_money_strong_market_healthy` | Continuation | 8% | 25% | 60 | Follow strong smart-money names in a healthy market. |
| `accumulation_breakout_smt_v1` | Accumulation breakout | 9% | 35% | 45 | Breakout after accumulation with smart-money confirmation. |
| `breakout_after_accumulation_v3` | Accumulation breakout | 9% | 35% | 45 | Breakout after accumulation / MA reclaim. |
| `compression_breakout_smt_v1` | Compression breakout | 7% | 22% | 30 | Tight range compression then breakout with smart-money support. |
| `mean_reversion_uptrend_ma50_v1` | Mean reversion | 8% | 25% | 45 | Buy pullbacks toward MA50 while uptrend remains intact. |

## Rule Of Thumb

| Question | Use |
|---|---|
| "Backtest mặc định từ giờ" | VN100. |
| "MVP hiện tại là gì?" | `mvp_p5`, `mvp_p4`, `flow_v2` on VN100. |
| "Cái nào đang có HTML/paper dashboard?" | The three active HTML sleeves above. |
| "Cái nào lợi nhuận cao nhất đã thấy?" | Flow V2 tune5 best on VN100, 129.82%, research only. |
| "Cái nào nên tin để vận hành hiện tại?" | VN100 active sleeves. |
| "Có được so Liquid150 với VN100 không?" | Yes, but label clearly as research comparison; do not mix them as one result set. |
