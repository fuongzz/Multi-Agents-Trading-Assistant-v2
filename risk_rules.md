# Risk Manager — 5 Hard Rules

## Nguyên tắc

- Rule-based engine — **KHÔNG dùng LLM**
- Chạy sau Trader Agent (Node 8), có quyền **override** bất kỳ quyết định nào
- Không thể bị bypass bởi bất kỳ agent hay prompt nào

---

## 5 Hard Rules

```python
def run_risk_manager(state: TradingState) -> dict:
    action = state["trader_decision"]["action"]
    warnings = []
    sizing_modifier = 1.0

    # Rule 1 — Circuit Breaker
    if state["market_context"]["vni_change_pct"] < -3.0:
        return override(action, "CHỜ", "VN-Index giảm >3% — circuit breaker")

    # Rule 2 — Foreign Room Trap
    room = state["foreign_flow_analysis"]["room_usage_pct"]
    if room is not None:
        if room > 95:
            return override(action, "CHỜ", f"Room ngoại {room}% — không mua được")
        elif room > 90:
            sizing_modifier *= 0.5
            warnings.append(f"Room ngoại {room}% — giảm sizing 50%")

    # Rule 3 — T+3 Constraint
    if action == "MUA":
        recent = db.get_buys_last_n_days(state["symbol"], 3)
        if recent:
            return override(action, "CHỜ", f"Đã mua {state['symbol']} trong T+3")

    # Rule 4 — Biên Độ
    price_move = abs(state["market_context"].get("stock_day_change_pct", 0))
    limit = 7.0 if state["market_context"]["exchange"] == "HOSE" else 10.0
    if price_move > limit * 0.714:
        sizing_modifier *= 0.5
        warnings.append(f"Đã dùng {price_move:.1f}% biên độ — giảm sizing")

    # Rule 5 — Trading Window
    now = get_vietnam_time()
    in_morning   = time(9, 15) <= now <= time(11, 30)
    in_afternoon = time(13, 0) <= now <= time(14, 25)
    if action in ["MUA", "BÁN"] and not (in_morning or in_afternoon):
        return override(action, "CHỜ", f"Ngoài giờ GD ({now.strftime('%H:%M')})")

    # Rule BÁN không có vị thế
    if action == "BÁN" and not db.has_position(state["symbol"]):
        return override(action, "CHỜ", "Không có vị thế để bán — không short")

    return {
        "final_action":    action,
        "override_reason": None,
        "warnings":        warnings,
        "sizing_modifier": sizing_modifier,
        "original_action": action,
    }
```

---

## Output Schema

```python
{
    "final_action":    "MUA" | "BÁN" | "CHỜ",
    "override_reason": str | None,      # None = không override
    "warnings":        list[str],
    "sizing_modifier": float,           # 1.0 | 0.5 | 0.0
    "original_action": str,             # action trước override
}
```

---

## VN Market Rules Reference

| Rule | Value |
|------|-------|
| Price band HOSE | ±7% |
| Price band HNX | ±10% |
| Settlement | T+3 |
| Short selling | ❌ Không được |
| ATC time | 14:25–14:30 |
| Morning session | 09:00–11:30 |
| Afternoon session | 13:00–14:30 |

---

## Foreign Room Levels

| Room % | Status | Hành động |
|--------|--------|-----------|
| > 95% | CRITICAL | Force CHỜ |
| 90–95% | HIGH | sizing × 0.5 |
| 80–90% | MEDIUM | sizing × 0.8 |
| < 80% | NORMAL | sizing × 1.0 |
