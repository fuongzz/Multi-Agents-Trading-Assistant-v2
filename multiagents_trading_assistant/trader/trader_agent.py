"""
trader_agent.py — Ra quyết định MUA/BÁN/CHỜ cuối cùng.

Model: Sonnet (run_agent) — cần reasoning sâu
Input:  TradingState đầy đủ (analysts + debate + market + macro)
Output: dict theo schema agents.md #7

Quy tắc vàng:
- KHÔNG mua gần kháng cự — entry phải có room lên đến TP
- KHÔNG bán gần hỗ trợ — nếu BÁN phải có room xuống rõ ràng
- action chỉ nhận: "MUA" | "BÁN" | "CHỜ"
- confidence → nav_pct: THẤP=3% | TRUNG_BÌNH=5% | CAO=8% | RẤT_CAO=10%
"""

import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11

from multiagents_trading_assistant.agent import run_agent


_SYSTEM_PROMPT = """Bạn là Trader chuyên nghiệp thị trường chứng khoán Việt Nam.
Nhiệm vụ: dựa trên toàn bộ phân tích (PTKT + FA + Khối ngoại + Sentiment + Debate), ra quyết định MUA/BÁN/CHỜ với entry/SL/TP cụ thể.

=== Quy tắc bắt buộc ===
1. action chỉ nhận đúng 3 giá trị: "MUA" | "BÁN" | "CHỜ"
2. KHÔNG mua gần kháng cự — khoảng cách entry → resistance gần nhất phải ≥ 5%
3. KHÔNG bán gần hỗ trợ — khoảng cách entry → support gần nhất phải ≥ 5% (nếu BÁN)
4. BÁN chỉ hợp lệ khi đang giữ vị thế (Risk Manager sẽ kiểm tra)
5. SL: tối đa 7% dưới entry (HOSE) / 10% (HNX)
6. TP: Risk/Reward ≥ 2:1 (TP - entry ≥ 2 × entry - SL)
7. nav_pct theo confidence: THẤP=3 | TRUNG_BÌNH=5 | CAO=8 | RẤT_CAO=10
8. Nếu không có setup rõ ràng → CHỜ, không ép lệnh

=== Ưu tiên signal ===
- PTKT confluence_score ≥ 6 + MA_TREND=UPTREND → điểm cộng lớn
- FA financial_health=KHỎE + valuation≠ĐẮT → hỗ trợ MUA
- Debate balance BULL_SLIGHT_EDGE hoặc STRONG_BULL → tăng confidence
- Room ngoại CRITICAL → CHỜ ngay (Risk Manager sẽ handle, nhưng Trader nên biết)
- Macro BEARISH → thiên về CHỜ

=== Output schema (JSON) ===
{
  "action": "MUA" | "BÁN" | "CHỜ",
  "entry": <float|null>,
  "sl": <float|null>,
  "tp": <float|null>,
  "nav_pct": <int 0-10>,
  "holding_period": <str — ví dụ "1-2 tuần" | "2-3 tuần" | "N/A">,
  "confidence": "THẤP" | "TRUNG_BÌNH" | "CAO" | "RẤT_CAO",
  "primary_reason": <str — 1-2 câu lý do chính tiếng Việt>,
  "risks": [<str>, ...],
  "trader_note": <str — nhận xét bổ sung hoặc điều kiện cần theo dõi>
}

Trả về JSON hợp lệ DUY NHẤT, không có text thừa."""


def decide(state: dict) -> dict:
    """
    Chạy Trader agent để ra quyết định MUA/BÁN/CHỜ.

    Args:
        state: TradingState đầy đủ từ orchestrator

    Returns:
        {"trader_decision": dict} theo schema agents.md #7
    """
    symbol = state["symbol"]
    print(f"\n[trader_agent] Chạy Trader (Sonnet) cho {symbol}...")

    prompt = _build_prompt(state)

    try:
        result = run_agent(prompt=prompt, system=_SYSTEM_PROMPT)

        # Validate và normalize
        result = _validate_decision(result, state)

        print(f"[trader_agent] {symbol} → action={result.get('action')}, "
              f"confidence={result.get('confidence')}, nav={result.get('nav_pct')}%")
        if result.get("entry"):
            print(f"[trader_agent] Entry={result.get('entry'):,.0f} | "
                  f"SL={result.get('sl'):,.0f} | TP={result.get('tp'):,.0f}")

        return {"trader_decision": result}

    except Exception as e:
        print(f"[trader_agent] Lỗi LLM: {e} — fallback CHỜ")
        return {"trader_decision": _fallback_decision(str(e))}


# ──────────────────────────────────────────────
# Build prompt
# ──────────────────────────────────────────────

def _build_prompt(state: dict) -> str:
    symbol  = state.get("symbol", "?")
    date    = state.get("date", "?")
    setup   = state.get("setup_type", "?")
    ptkt    = state.get("ptkt_analysis", {})
    fa      = state.get("fa_analysis", {})
    ff      = state.get("foreign_flow_analysis", {})
    sent    = state.get("sentiment_analysis", {})
    macro   = state.get("macro_context", {})
    debate  = state.get("debate_synthesis", {})
    mkt     = state.get("market_context", {})

    # Giá hiện tại và S/R
    current_price   = _get_current_price(ptkt, mkt)
    supports        = ptkt.get("support_levels", [])
    resistances     = ptkt.get("resistance_levels", [])
    nearest_res     = resistances[0] if resistances else None
    nearest_sup     = supports[0] if supports else None
    atr             = ptkt.get("atr")

    # Tính room đến kháng cự
    room_to_res = None
    if current_price and nearest_res:
        room_to_res = round((nearest_res - current_price) / current_price * 100, 1)

    lines = [
        f"Ra quyết định cho mã {symbol} ngày {date}.",
        f"Setup type: {setup}",
        f"Giá hiện tại: {current_price:,.0f} VNĐ" if current_price else "Giá hiện tại: N/A",
        "",
        "=== VN-Index Context ===",
        f"Trend: {mkt.get('trend','?')} | VNI thay đổi hôm nay: {mkt.get('vni_change_pct','?')}%",
        f"Sàn: {mkt.get('exchange','HOSE')}",
        "",
        "=== PTKT ===",
        f"MA trend: {ptkt.get('ma_trend','?')} | Phase: {ptkt.get('ma_phase','?')}",
        f"RSI: {ptkt.get('rsi','?')} ({ptkt.get('rsi_signal','?')})",
        f"MACD: {ptkt.get('macd_signal','?')} | Bollinger: {ptkt.get('bollinger_position','?')}",
        f"ATR: {atr:,.0f}" if atr else "ATR: N/A",
        f"Confluence: {ptkt.get('confluence_score','?')}/10 | Quality: {ptkt.get('setup_quality','?')}",
        f"Support:    {supports}",
        f"Resistance: {resistances}",
        f"Room đến kháng cự gần nhất: {room_to_res}%" if room_to_res is not None else "Room đến kháng cự: N/A",
        f"Tóm tắt PTKT: {ptkt.get('technical_summary','')}",
        "",
        "=== FA ===",
        f"P/E: {fa.get('pe_ratio','?')} | P/B: {fa.get('pb_ratio','?')} | ROE: {fa.get('roe','?')}%",
        f"EPS growth: {fa.get('eps_growth_yoy','?')}% | Định giá: {fa.get('valuation','?')}",
        f"Sức khỏe: {fa.get('financial_health','?')} | So ngành: {fa.get('vs_industry','?')}",
        f"Tóm tắt FA: {fa.get('fa_summary','')}",
        "",
        "=== Khối ngoại ===",
        f"Room: {ff.get('room_usage_pct','?')}% ({ff.get('room_status','?')})",
        f"Flow trend: {ff.get('flow_trend','?')} | Tích lũy: {ff.get('accumulation_signal','?')}",
        f"Sizing modifier: {ff.get('sizing_modifier',1.0)}×",
        "",
        "=== Sentiment ===",
        f"Score: {sent.get('sentiment_score','?')}/100 ({sent.get('sentiment_label','?')})",
        f"Key positive: {sent.get('key_positive',[])}",
        f"Key negative: {sent.get('key_negative',[])}",
        "",
        "=== Macro ===",
        f"Bias: {macro.get('macro_bias','?')} | Score: {macro.get('macro_score','?')}",
        f"Rủi ro vĩ mô: {macro.get('key_risks',[])}",
        f"Hỗ trợ vĩ mô: {macro.get('key_supports',[])}",
        "",
        "=== Kết quả Debate Bull vs Bear ===",
        f"Balance: {debate.get('balance','?')}",
        f"Bull points: {debate.get('bull_key_points',[])}",
        f"Bear points: {debate.get('bear_key_points',[])}",
        f"Key risk từ debate: {debate.get('key_risk','')}",
        f"Kết luận debate: {debate.get('debate_conclusion','')}",
        "",
        "=== Hướng dẫn tính entry/SL/TP ===",
    ]

    # Hướng dẫn tính entry/SL/TP theo setup type
    if setup in ("BREAKOUT", "SHORT_TERM"):
        lines += [
            "Setup BREAKOUT: Entry = giá hiện tại (đã break kháng cự)",
            "SL = support gần nhất hoặc 5-7% dưới entry",
            "TP = kháng cự tiếp theo hoặc entry + 2×(entry-SL)",
        ]
    elif setup in ("MA_PULLBACK", "MID_TERM", "RETEST"):
        lines += [
            "Setup PULLBACK/RETEST: Entry = vùng support / MA gần nhất",
            "SL = 1-2 ATR dưới entry hoặc đáy gần nhất",
            "TP = kháng cự gần nhất hoặc entry + 2×(entry-SL)",
        ]
    elif setup == "RSI_BOUNCE":
        lines += [
            "Setup RSI BOUNCE: Entry = giá hiện tại (RSI oversold)",
            "SL = 5% dưới entry",
            "TP = MA20 hoặc kháng cự gần nhất",
        ]
    else:
        lines += [
            "Tự xác định entry/SL/TP phù hợp với setup.",
        ]

    lines += [
        "",
        "Trả về JSON quyết định theo schema đã định.",
        "Nếu không đủ điều kiện → CHỜ, entry/sl/tp = null.",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _get_current_price(ptkt: dict, mkt: dict) -> float | None:
    """Lấy giá hiện tại từ ptkt hoặc market context."""
    # ptkt_agent không lưu current_price trực tiếp trong output
    # Dùng support[0] + khoảng cách làm proxy, hoặc lấy từ mkt
    return mkt.get("current_price") or None


def _validate_decision(result: dict, state: dict) -> dict:
    """
    Validate và normalize output từ Trader LLM.
    Đảm bảo:
    - action hợp lệ
    - nav_pct đúng theo confidence
    - Quy tắc KHÔNG mua gần kháng cự
    """
    # Normalize action
    action = str(result.get("action", "CHỜ")).upper().strip()
    if action not in ("MUA", "BÁN", "CHỜ"):
        action = "CHỜ"
    result["action"] = action

    # Validate confidence → nav_pct
    confidence = result.get("confidence", "THẤP")
    max_nav = {"THẤP": 3, "TRUNG_BÌNH": 5, "CAO": 8, "RẤT_CAO": 10}.get(confidence, 3)
    nav_pct = result.get("nav_pct", 0)
    result["nav_pct"] = min(int(nav_pct or 0), max_nav)

    # Kiểm tra quy tắc KHÔNG mua gần kháng cự
    if action == "MUA":
        entry      = result.get("entry")
        ptkt       = state.get("ptkt_analysis", {})
        resistances = ptkt.get("resistance_levels", [])

        if entry and resistances:
            nearest_res = resistances[0]
            room_pct = (nearest_res - entry) / entry * 100
            if room_pct < 3:  # entry quá gần kháng cự
                print(f"[trader_agent] ⚠️ Entry quá gần kháng cự ({room_pct:.1f}%) → override CHỜ")
                result["action"]         = "CHỜ"
                result["entry"]          = None
                result["sl"]             = None
                result["tp"]             = None
                result["nav_pct"]        = 0
                result["primary_reason"] = (
                    f"Entry {entry:,.0f} quá gần kháng cự {nearest_res:,.0f} "
                    f"(room chỉ {room_pct:.1f}%) — chờ pullback."
                )

    # Đảm bảo CHỜ không có entry/sl/tp
    if result.get("action") == "CHỜ":
        result.setdefault("entry", None)
        result.setdefault("sl", None)
        result.setdefault("tp", None)
        result["nav_pct"] = 0

    # Đảm bảo có đủ fields
    result.setdefault("holding_period", "N/A")
    result.setdefault("risks", [])
    result.setdefault("trader_note", "")

    return result


def _fallback_decision(error_msg: str = "") -> dict:
    """Trả về CHỜ khi LLM fail."""
    return {
        "action":         "CHỜ",
        "entry":          None,
        "sl":             None,
        "tp":             None,
        "nav_pct":        0,
        "holding_period": "N/A",
        "confidence":     "THẤP",
        "primary_reason": f"LLM timeout/lỗi — mặc định CHỜ. {error_msg}".strip(),
        "risks":          ["Trader agent không phản hồi"],
        "trader_note":    "Fallback do lỗi kỹ thuật.",
    }
