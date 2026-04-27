"""trader_trade.py — Quyết định MUA/BÁN/CHỜ cho Trade pipeline (ngắn hạn).

Model: Sonnet (run_agent). Port từ _legacy/trader/trader_agent.py.

Bắt buộc output:
  entry_zone: [low, high]   — vùng entry (không phải single price)
  stop_loss:  float          — SL kỹ thuật
  take_profit: float         — TP, R:R ≥ 2
  holding_horizon: "1-4 tuần"
  position_pct: 2-5% NAV

Tôn trọng biên độ ±7% HOSE / ±10% HNX / ±15% UPCoM và T+2.5.
"""

import importlib.metadata  # noqa: F401

from multiagents_trading_assistant.services.llm_service import run_agent


_SYSTEM_PROMPT = """Bạn là Trader đầu cơ ngắn hạn (1-4 tuần) thị trường VN.
Nhiệm vụ: dựa trên technical + flow + sentiment + synthesis, ra quyết định MUA/CHỜ/TRÁNH với entry_zone/SL/TP cụ thể.

=== Quy tắc bắt buộc ===
1. action chỉ nhận: "MUA" | "CHỜ" | "TRÁNH"
2. entry_zone: [low, high] — vùng giá entry hợp lệ (không single price)
3. SL ≤ 7% dưới low của entry_zone (HOSE), ≤ 10% (HNX), ≤ 15% (UPCoM)
4. TP: Risk/Reward ≥ 2 — (TP - entry_mid) ≥ 2 × (entry_mid - SL)
5. KHÔNG mua gần kháng cự — high của entry_zone phải cách resistance gần nhất ≥ 3%
6. holding_horizon cố định: "1-4 tuần" (Trade pipeline không hold dài hơn)
7. position_pct: 2-5% NAV — cao confluence → 5%, thấp → 2%
8. Confluence < 50 → bắt buộc CHỜ hoặc TRÁNH
9. Không setup rõ → CHỜ, entry_zone/SL/TP = null
10. T+2.5 — KHÔNG trade scalping: hàng về chiều T+2, không thể exit T+0/T+1. Holding ≥ 3 ngày là minimum thực tế.
11. Nhốt sàn/trần: cổ phiếu đã dùng >70% biên độ xuống trong phiên → thanh khoản kém, KHÔNG mua. Đặt SL phải có dư địa thoát lệnh.
12. Khối ngoại & tự doanh: dòng tiền "Tây" và CTCK có ảnh hưởng tâm lý lớn trên HOSE. Ưu tiên MUA khi khối ngoại mua ròng ≥ 20 tỷ/phiên. Thận trọng khi tự doanh bán ròng mạnh.

=== Output schema (JSON hợp lệ DUY NHẤT) ===
{
  "action": "MUA" | "CHỜ" | "TRÁNH",
  "entry_zone": [<float>, <float>] | null,
  "stop_loss": <float|null>,
  "take_profit": <float|null>,
  "rr_ratio": <float|null>,
  "position_pct": <int 0-5>,
  "holding_horizon": "1-4 tuần",
  "confidence": "THẤP" | "TRUNG_BÌNH" | "CAO",
  "primary_reason": <str — 1-2 câu tiếng Việt>,
  "risks": [<str>, ...],
  "trader_note": <str>
}"""


def decide(state: dict) -> dict:
    symbol = state.get("symbol", "?")
    print(f"[trader_trade] Sonnet — {symbol}")

    prompt = _build_prompt(state)

    try:
        result = run_agent(prompt=prompt, system=_SYSTEM_PROMPT)
        result = _validate(result, state)
        print(f"[trader_trade] {symbol} → {result.get('action')} | conf={result.get('confidence')} | pos={result.get('position_pct')}%")
        if result.get("entry_zone"):
            ez = result["entry_zone"]
            print(f"[trader_trade]   entry={ez[0]:,.0f}-{ez[1]:,.0f} | SL={result.get('stop_loss'):,.0f} | TP={result.get('take_profit'):,.0f} | R:R={result.get('rr_ratio')}")
        return {"trader_decision": result}
    except Exception as e:
        print(f"[trader_trade] LLM error: {e} — fallback CHỜ")
        return {"trader_decision": _fallback(str(e))}


def _build_prompt(state: dict) -> str:
    symbol = state.get("symbol", "?")
    date = state.get("date", "?")
    setup = state.get("setup_type", "?")

    tech = state.get("technical_analysis", {})
    flow = state.get("foreign_flow_analysis", {})
    sent = state.get("sentiment_analysis", {})
    synth = state.get("synthesis", {})
    macro = state.get("macro_context", {})
    mkt = state.get("market_context", {})

    current_price = mkt.get("current_price")
    supports = tech.get("support_levels", [])
    resistances = tech.get("resistance_levels", [])
    nearest_res = resistances[0] if resistances else None
    atr = tech.get("atr")

    room_to_res = None
    if current_price and nearest_res:
        room_to_res = round((nearest_res - current_price) / current_price * 100, 1)

    lines = [
        f"Quyết định Trade ngắn hạn cho {symbol} ngày {date}.",
        f"Setup: {setup}",
        f"Giá hiện tại: {_fmt(current_price)} VNĐ",
        f"Sàn: {mkt.get('exchange', 'HOSE')}",
        "",
        "=== VN-Index ===",
        f"Trend: {mkt.get('trend', '?')} | Δhôm nay: {mkt.get('vni_change_pct', '?')}%",
        "",
        "=== Synthesis (rule-based) ===",
        f"Confluence: {synth.get('confluence_score', '?')}/100 ({synth.get('setup_quality', '?')})",
        f"Drivers: {synth.get('drivers', [])}",
        f"Blockers: {synth.get('blockers', [])}",
        "",
        "=== Technical ===",
        f"MA trend: {tech.get('ma_trend', '?')} | Phase: {tech.get('ma_phase', '?')}",
        f"RSI: {tech.get('rsi', '?')} ({tech.get('rsi_signal', '?')})",
        f"MACD: {tech.get('macd_signal', '?')} | Bollinger: {tech.get('bollinger_position', '?')}",
        f"ATR: {_fmt(atr)}",
        f"Support: {supports}",
        f"Resistance: {resistances}",
        f"Room→R gần nhất: {room_to_res}%" if room_to_res is not None else "Room→R: N/A",
        "",
        "=== Foreign flow ===",
        f"Room: {flow.get('room_status', '?')} | Trend: {flow.get('flow_trend', '?')} | Sizing: ×{flow.get('sizing_modifier', 1.0)}",
        f"Tích lũy: {flow.get('accumulation_signal', False)}",
        "",
        "=== Sentiment ===",
        f"Score: {sent.get('sentiment_score', '?')}/100 ({sent.get('sentiment_label', '?')})",
        "",
        "=== Macro ===",
        f"Bias: {macro.get('macro_bias', '?')}",
        "",
        *_format_memory_section(state.get("memory_context", {})),
        "=== Hướng dẫn entry/SL/TP theo setup ===",
    ]

    if setup in ("BREAKOUT",):
        lines += [
            "BREAKOUT: entry_zone = [giá hiện tại, giá hiện tại × 1.01]",
            "SL = support gần nhất hoặc -5% entry_low",
            "TP = resistance kế tiếp hoặc entry_mid + 2×(entry_mid - SL)",
        ]
    elif setup in ("RETEST",):
        lines += [
            "RETEST: entry_zone = vùng support đã break (±1%)",
            "SL = -1.5×ATR dưới entry_low",
            "TP = đỉnh gần nhất hoặc R:R ≥ 2",
        ]
    elif setup in ("MA_PULLBACK",):
        lines += [
            "MA_PULLBACK: entry_zone = [MA20-1%, MA20+1%]",
            "SL = MA60 hoặc -1.5×ATR",
            "TP = đỉnh trước hoặc R:R ≥ 2",
        ]
    elif setup in ("RSI_BOUNCE",):
        lines += [
            "RSI_BOUNCE: entry_zone = [giá hiện tại, +1%]",
            "SL = -5% entry_low",
            "TP = MA20 hoặc R:R ≥ 2",
        ]
    elif setup in ("SPRING",):
        lines += [
            "SPRING: entry_zone quanh đáy giả vừa hồi",
            "SL = dưới đáy giả 2-3%",
            "TP = top range trước đó",
        ]
    else:
        lines += ["Tự xác định entry/SL/TP phù hợp."]

    lines += [
        "",
        f"Lưu ý: biên độ {({'HOSE': '±7%', 'HNX': '±10%'}.get(mkt.get('exchange', 'HOSE'), '±15%'))} và T+2.5 (không scalping).",
        "Trả JSON theo schema. Nếu không đủ điều kiện → CHỜ với entry_zone/SL/TP=null.",
    ]
    return "\n".join(lines)


def _format_memory_section(ctx: dict) -> list[str]:
    """Định dạng memory_context (nested) thành dòng tiếng Việt cho prompt.

    ctx = {"internal": {...}, "external": {...}}
    Trả về list[str] để unpack vào lines. Safe khi ctx rỗng (cold start).
    """
    if not ctx:
        return []

    internal = ctx.get("internal", {})
    external = ctx.get("external", {})

    lines = ["=== Lịch sử & Context ==="]

    # ── Vị thế / T+2.5 ──
    if internal.get("has_position"):
        pos = internal.get("current_position") or {}
        lines.append(
            f"⚠ Đang giữ vị thế: entry {pos.get('entry_price', '?')} "
            f"(từ {pos.get('entry_date', '?')}), SL {pos.get('sl', '?')}, "
            f"TP {pos.get('tp', '?')}, {pos.get('nav_pct', '?')}% NAV"
        )
    elif internal.get("t3_blocked"):
        lines.append("⚠ T+2.5: Đã mua trong 2 ngày qua — chưa qua T+2.5, không MUA thêm.")
    else:
        lines.append("Chưa có vị thế / không bị chặn T+2.5.")

    # ── Decisions gần đây (7 ngày) ──
    recent = internal.get("recent_decisions", [])
    streak = internal.get("cho_streak", 0)
    if recent:
        lines.append(f"Quyết định 7 ngày gần đây ({len(recent)} lần):")
        for d in recent[:3]:
            action_str = d.get("final_action") or d.get("action", "?")
            override   = f" [{d.get('override_reason')}]" if d.get("override_reason") else ""
            lines.append(
                f"  • {d.get('date', '?')}: {action_str} ({d.get('confidence', '?')}){override}"
            )
        if streak >= 3:
            lines.append(f"  → Đã CHỜ {streak} ngày liên tiếp — cân nhắc cẩn thận trước khi MUA.")
    else:
        lines.append("Chưa có lịch sử quyết định cho mã này.")

    # ── Setup tương tự trong quá khứ (L2) ──
    similar = internal.get("similar_setups", [])
    if similar:
        lines.append(f"Setup tương tự đã gặp ({len(similar)} trường hợp):")
        for s in similar[:3]:
            meta    = s.get("metadata", {})
            dist    = s.get("distance")
            sim_str = f" (similarity {1 - dist:.2f})" if dist is not None else ""
            lines.append(
                f"  • {meta.get('symbol', '?')} {meta.get('date', '?')}: "
                f"{meta.get('action', '?')} | setup {meta.get('setup_type', '?')} | "
                f"trend {meta.get('ma_trend', '?')}{sim_str}"
            )
    elif internal.get("l2_available") is False:
        lines.append("(ChromaDB chưa có dữ liệu — hệ thống mới.)")
    else:
        lines.append("Chưa tìm thấy setup tương tự trong lịch sử.")

    # ── External: Vietstock Knowledge Base ──
    fund_summary = external.get("fundamental_summary", "")
    hist_stats   = external.get("historical_stats", "")
    if fund_summary or hist_stats:
        lines.append("")
        lines.append("=== Kiến thức bổ trợ (Vietstock) ===")
        lines.append(
            "Dữ liệu nội tại & thống kê từ Vietstock. "
            "Nếu tín hiệu kỹ thuật MUA nhưng cơ bản có vấn đề "
            "(nợ xấu tăng, vi phạm, lãnh đạo thay đổi...) — "
            "hạ Confidence xuống, ghi rõ vào risks[]."
        )
        if fund_summary:
            lines.append(f"[Sức khỏe tài chính]: {fund_summary}")
        if hist_stats:
            lines.append(f"[Chu kỳ / Thống kê]: {hist_stats}")

    lines.append("")
    return lines


def _validate(result: dict, state: dict) -> dict:
    action = str(result.get("action", "CHỜ")).upper().strip()
    if action not in ("MUA", "CHỜ", "TRÁNH"):
        action = "CHỜ"
    result["action"] = action

    confidence = result.get("confidence", "THẤP")
    max_pos = {"THẤP": 2, "TRUNG_BÌNH": 3, "CAO": 5}.get(confidence, 2)
    result["position_pct"] = min(int(result.get("position_pct") or 0), max_pos)
    result["holding_horizon"] = "1-4 tuần"

    # Tính rr_ratio nếu thiếu
    if action == "MUA" and result.get("entry_zone") and result.get("stop_loss") and result.get("take_profit"):
        ez = result["entry_zone"]
        entry_mid = (ez[0] + ez[1]) / 2
        sl = result["stop_loss"]
        tp = result["take_profit"]
        if entry_mid > sl:
            result["rr_ratio"] = round((tp - entry_mid) / (entry_mid - sl), 2)

        # Hard rule: không mua gần kháng cự
        tech = state.get("technical_analysis", {})
        resistances = tech.get("resistance_levels", [])
        if resistances:
            nearest_res = resistances[0]
            room_pct = (nearest_res - ez[1]) / ez[1] * 100
            if room_pct < 3:
                print(f"[trader_trade] ⚠ entry_high quá gần R ({room_pct:.1f}%) → CHỜ")
                result.update({
                    "action": "CHỜ", "entry_zone": None, "stop_loss": None,
                    "take_profit": None, "rr_ratio": None, "position_pct": 0,
                    "primary_reason": f"Entry quá gần R ({room_pct:.1f}%) — chờ pullback.",
                })

    if result.get("action") == "CHỜ":
        result.setdefault("entry_zone", None)
        result.setdefault("stop_loss", None)
        result.setdefault("take_profit", None)
        result.setdefault("rr_ratio", None)
        result["position_pct"] = 0

    result.setdefault("risks", [])
    result.setdefault("trader_note", "")
    return result


def _fallback(err: str) -> dict:
    return {
        "action": "CHỜ", "entry_zone": None, "stop_loss": None, "take_profit": None,
        "rr_ratio": None, "position_pct": 0, "holding_horizon": "1-4 tuần",
        "confidence": "THẤP", "primary_reason": f"LLM lỗi — CHỜ. {err}".strip(),
        "risks": ["trader_trade fail"], "trader_note": "Fallback.",
    }


def _fmt(v) -> str:
    return "N/A" if v is None else f"{v:,.2f}"
