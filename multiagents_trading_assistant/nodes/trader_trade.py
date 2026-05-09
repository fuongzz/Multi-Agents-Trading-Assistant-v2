"""trader_trade.py — Quyết định theo state: entry mới hoặc quản trị vị thế.

Model: Sonnet (run_agent). Port từ _legacy/trader/trader_agent.py.

Bắt buộc output:
  entry_zone: [low, high]   — vùng entry (không phải single price)
  stop_loss:  float          — SL kỹ thuật
  initial_target: float      — target tham chiếu, R:R ≥ 1.5
  holding_horizon: "1-4 tuần"
  position_pct: 2-5% NAV

Tôn trọng biên độ ±7% HOSE / ±10% HNX / ±15% UPCoM và T+2.5.
"""

import importlib.metadata  # noqa: F401

from multiagents_trading_assistant.services.llm_service import run_agent


_FLAT_ACTIONS = {"MUA", "CHỜ", "TRÁNH"}
_POSITION_ACTIONS = {"GIỮ", "GIA_TĂNG", "GIẢM", "BÁN"}
_ALL_ACTIONS = _FLAT_ACTIONS | _POSITION_ACTIONS


_SYSTEM_PROMPT = """Bạn là Trader đầu cơ thị trường VN theo triết lý price-action.
Nhiệm vụ: dựa trên technical + flow + sentiment + synthesis, ra quyết định theo state:
- Nếu CHƯA có vị thế: chỉ xét MUA/CHỜ/TRÁNH với entry_zone và SL kỹ thuật.
- Nếu ĐANG có vị thế: quản trị vị thế bằng GIỮ/GIA_TĂNG/GIẢM/BÁN, không phát MUA mới.

=== Triết lý giữ lệnh (QUAN TRỌNG) ===
- KHÔNG exit theo thời gian (không có "5 phiên" hay "1 tuần").
- Cổ phiếu tăng → để nó tăng. Trailing SL tự bảo vệ lãi khi giá mạnh.
- Chỉ xem xét thoát khi price action nói "dừng":
    • SL trailing bị chạm (swing low dưới entry giảm sâu)
    • Double top xuất hiện (2 đỉnh ngang nhau, pull back qua midpoint)
    • MA20 bị phá 2 nến liên tiếp + MA20 đang dốc xuống
    • Cấu trúc HH+HL bị phá (giá đóng dưới prior swing low 2 nến liên tiếp)
- initial_target chỉ là mục tiêu tham chiếu — KHÔNG đặt lệnh bán tự động tại đó.
  Khi giá đạt initial_target, nâng SL lên để khoá lợi nhuận, tiếp tục giữ.

=== State machine bắt buộc ===
1. CHƯA CÓ VỊ THẾ:
   - Được dùng: "MUA" | "CHỜ" | "TRÁNH".
   - "MUA" chỉ hợp lệ khi giá hiện tại đang trong/vừa sát entry_zone và đủ SL/R:R.
   - Nếu setup cần chờ giá về vùng entry hoặc chờ breakout xác nhận, dùng "CHỜ" nhưng vẫn mô tả điều kiện entry trong trader_note.
2. ĐANG CÓ VỊ THẾ:
   - KHÔNG dùng "MUA". Tín hiệu mới chỉ là tín hiệu quản trị vị thế.
   - Được dùng: "GIỮ" | "GIA_TĂNG" | "GIẢM" | "BÁN".
   - "GIỮ": thesis còn đúng, chưa có lý do thoát; có thể đề xuất nâng SL.
   - "GIA_TĂNG": chỉ khi vị thế hiện tại đang lãi, cấu trúc HH/HL còn nguyên, SL hiện tại đã gần hòa vốn hoặc khóa lãi, và tổng rủi ro sau tăng vị thế vẫn hợp lý.
   - "GIẢM": khi đạt target nhưng momentum yếu, phân phối nhẹ, hoặc cần giảm risk.
   - "BÁN": khi SL/reversal/distribution rõ hoặc thesis gãy.
   - Với "GIỮ"/"GIẢM"/"BÁN", entry_zone và position_pct cho lệnh mới phải là null/0, trừ "GIA_TĂNG" có add-on zone riêng.

=== Quy tắc bắt buộc ===
1. action chỉ nhận: "MUA" | "CHỜ" | "TRÁNH" | "GIỮ" | "GIA_TĂNG" | "GIẢM" | "BÁN"
2. entry_zone: [low, high] — vùng giá entry hợp lệ (không single price)
3. stop_loss ≤ 5% dưới low của entry_zone — đặt dưới swing low kỹ thuật gần nhất
4. initial_target: tham chiếu R:R ≥ 1.5 — dùng để tính trail_sl_guide, KHÔNG phải hard TP
5. KHÔNG mua gần kháng cự — high của entry_zone phải cách resistance gần nhất ≥ 3%
6. position_pct theo confluence:
   - Confluence ≥ 70 → 5% NAV
   - Confluence 55–69 → 3% NAV
   - Confluence < 55 → 2% NAV
7. Confluence < 50 → bắt buộc CHỜ hoặc TRÁNH
8. Không setup rõ → CHỜ, entry_zone/stop_loss/initial_target = null
9. T+2.5 — không scalp: hàng về chiều T+2, không exit T+0/T+1. Hold tối thiểu qua T+3.
10. Nhốt sàn/trần: cổ phiếu đã tăng ≥ 5% hoặc dùng >70% biên độ xuống → KHÔNG mua.
11. Khối ngoại & tự doanh: ưu tiên MUA khi NN mua ròng ≥ 20 tỷ/phiên. Thận trọng khi tự doanh bán mạnh.
12. trail_sl_guide: gợi ý nâng SL khi lãi đạt các mức — dùng swing low gần nhất làm tham chiếu.

=== Output schema (JSON hợp lệ DUY NHẤT) ===
{
  "action": "MUA" | "CHỜ" | "TRÁNH" | "GIỮ" | "GIA_TĂNG" | "GIẢM" | "BÁN",
  "position_state": "NO_POSITION" | "OPEN_POSITION",
  "entry_zone": [<float>, <float>] | null,
  "stop_loss": <float|null>,
  "initial_target": <float|null>,
  "rr_ratio": <float|null>,
  "position_pct": <int 0-5>,
  "trail_sl_guide": {
    "at_5pct_gain":  <float|null>,
    "at_10pct_gain": <float|null>,
    "at_20pct_gain": <float|null>
  },
  "reversal_watchlist": [<str> — dấu hiệu cần theo dõi để thoát],
  "management_plan": <str|null — kế hoạch quản trị nếu đang có vị thế>,
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
            tgt = result.get("initial_target")
            tgt_str = f"{tgt:,.0f}" if tgt else "N/A"
            print(f"[trader_trade]   entry={ez[0]:,.0f}-{ez[1]:,.0f} | SL={result.get('stop_loss'):,.0f} | target={tgt_str} | R:R={result.get('rr_ratio')}")
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

    # market_context.current_price is the market index level from the screener.
    # For symbol-level entry/SL guidance, use the stock price from technical_analysis.
    current_price = tech.get("current_price") or mkt.get("stock_current_price") or mkt.get("current_price")
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
        "=== Trạng thái thị trường ===",
        f"VNI: {mkt.get('trend', '?')} ({_fmt_pct(mkt.get('vni_change_pct'))}) | "
        f"VNMidCap: {mkt.get('vnmidcap_trend', '?')} ({_fmt_pct(mkt.get('vnmidcap_change_pct'))})",
        f"Tham chiếu quyết định: {mkt.get('reference_index', 'VNINDEX')} → {mkt.get('reference_trend', mkt.get('trend', '?'))}",
        *(
            [f"⚠ Méo chỉ số: {mkt.get('distortion_note', '')}"]
            if mkt.get("is_index_distorted")
            else []
        ),
        f"VinGroup đóng góp VNI hôm nay: ≈ {mkt.get('vingroup_contribution_pct', 0):+.2f}%",
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
        *_format_pv_section(tech.get("price_volume") or {}),
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

    setup_guides = {
        "BREAKOUT": [
            "entry_zone = [giá hiện tại, giá hiện tại × 1.01]",
            "SL = support gần nhất hoặc -3% entry_low (không quá 5%)",
            "initial_target = resistance kế tiếp hoặc entry_mid + 1.5×(entry_mid - SL)",
        ],
        "FLAG_PENNANT": [
            "entry_zone = [đỉnh vùng nén, đỉnh vùng nén × 1.01]",
            "SL = đáy vùng nén hoặc -3% entry_low",
            "TP = đỉnh cột cờ hoặc entry_mid + 1.5×(entry_mid - SL)",
        ],
        "BB_SQUEEZE": [
            "entry_zone = [BB upper, BB upper × 1.01] — chờ giá vượt BB upper",
            "SL = BB mid hoặc -3% entry_low",
            "TP = entry_mid + 1.5×(entry_mid - SL) hoặc resistance gần nhất",
        ],
        "RETEST": [
            "entry_zone = vùng support đã break (±1%)",
            "SL = -1.5×ATR dưới entry_low (không quá 5%)",
            "TP = đỉnh gần nhất hoặc R:R ≥ 1.5",
        ],
        "SPRING": [
            "entry_zone = quanh đáy giả vừa hồi (±1%)",
            "SL = dưới đáy giả 2-3%",
            "initial_target = top range trước đó",
        ],
        "GOLDEN_CROSS": [
            "entry_zone = [MA20, MA20 × 1.015]",
            "SL = MA50 hoặc -3% entry_low",
            "initial_target = resistance gần nhất hoặc R:R ≥ 1.5",
        ],
        "DOUBLE_BOTTOM": [
            "entry_zone = [neckline, neckline × 1.02]",
            "SL = đáy thứ 2 hoặc -4% entry_low",
            "initial_target = neckline + (neckline - đáy) — measured move",
        ],
        "MOMENTUM_SURGE": [
            "entry_zone = [giá hiện tại, giá hiện tại × 1.01]",
            "SL = đáy nến hôm qua hoặc -3% entry_low",
            "TP = resistance kế tiếp hoặc R:R ≥ 1.5",
        ],
        "MACD_CROSSOVER": [
            "entry_zone = [giá hiện tại, giá hiện tại × 1.01]",
            "SL = support gần nhất hoặc -3% entry_low",
            "initial_target = resistance gần nhất hoặc R:R ≥ 1.5",
        ],
        "MA_PULLBACK": [
            "entry_zone = [MA20-1%, MA20+1%]",
            "SL = MA60 hoặc -1.5×ATR (không quá 5%)",
            "TP = đỉnh trước hoặc R:R ≥ 1.5",
        ],
        "INSIDE_BAR": [
            "entry_zone = [đỉnh inside bar, đỉnh inside bar × 1.01]",
            "SL = đáy inside bar hoặc -3% entry_low",
            "initial_target = resistance gần nhất hoặc R:R ≥ 1.5",
        ],
        "NR7": [
            "entry_zone = [đỉnh nến NR7, đỉnh nến NR7 × 1.01]",
            "SL = đáy nến NR7 — range hẹp nên SL tự nhiên rất gần",
            "initial_target = resistance gần nhất hoặc R:R ≥ 1.5",
        ],
        "HAMMER": [
            "entry_zone = [close nến hammer, close × 1.01]",
            "SL = đáy bóng hammer hoặc -3% entry_low",
            "initial_target = MA20 hoặc resistance gần nhất",
        ],
        "RSI_BOUNCE": [
            "entry_zone = [giá hiện tại, giá hiện tại × 1.01]",
            "SL = -3% entry_low (không quá 5%)",
            "initial_target = MA20 hoặc R:R ≥ 1.5",
        ],
        # ── Price Action setups ──
        "PIN_BAR": [
            "entry_zone = [close nến pin bar, close × 1.005] — vào gần close xác nhận",
            "SL = đáy bóng wick (low của nến pin bar) — tự nhiên và chặt",
            "TP = resistance gần nhất hoặc R:R ≥ 1.5 tính từ SL",
            "Lưu ý: pin bar hợp lệ khi đang tại key S/R level — không vào khi giá giữa chừng",
        ],
        "BULLISH_ENGULFING": [
            "entry_zone = [close nến engulfing, close × 1.005]",
            "SL = low của nến engulfing hoặc low nến đỏ trước — chọn cái thấp hơn",
            "initial_target = resistance gần nhất hoặc R:R ≥ 1.5",
            "Lưu ý: engulfing mạnh nhất khi tại support sau downswing — không chase sau khi đã tăng 3%+",
        ],
        "TREND_PULLBACK": [
            "entry_zone = vùng old breakout level (kháng cự cũ vừa thành support mới) ±1%",
            "SL = -1.5×ATR dưới entry_low, hoặc dưới đáy nến xác nhận (không quá 4%)",
            "initial_target = đỉnh gần nhất (recent high) — measured move về lại đỉnh cũ",
            "Lưu ý: setup thuận xu hướng — chỉ vào khi uptrend rõ (HH+HL) và volume cạn trong pullback",
        ],
        "BREAKOUT_RETEST_ENTRY": [
            "entry_zone = vùng breakout level ±1% (kháng cự vừa bị phá → test lại làm support)",
            "SL = -1.5×ATR dưới entry_low (không quá 4%) — nếu retest fail thì thoát sớm",
            "initial_target = measured move = entry + (breakout high - breakout level)",
            "Lưu ý: chỉ vào khi volume retest khô (< 0.8× TB20) và nến đóng xanh xác nhận support giữ",
        ],
    }
    guide = setup_guides.get(setup)
    if guide:
        lines += [f"{setup}: {g}" for g in guide]
    else:
        lines += ["Tự xác định entry/SL/TP phù hợp với setup. SL ≤ 5% entry_low."]

    lines += [
        "",
        "=== Hướng dẫn Trailing SL ===",
        "trail_sl_guide: gợi ý mức SL mới khi lãi đạt ngưỡng, tính từ swing low gần nhất.",
        "  at_5pct_gain  → nâng SL lên entry (hòa vốn) hoặc swing low gần nhất - ATR×0.5",
        "  at_10pct_gain → SL = swing low ngay trước đó (7-10 bar) - ATR×0.5",
        "  at_20pct_gain → SL = swing low rộng hơn (15-20 bar) - ATR×0.5, cho trend thở",
        "",
        "reversal_watchlist: liệt kê 2-3 dấu hiệu kỹ thuật cụ thể cần theo dõi để xem xét thoát.",
        "  Ví dụ: 'Nến đỏ đóng dưới MA20 lần thứ 2', 'Double top tại kháng cự X', 'RSI phân kỳ giảm'",
        "",
        f"Lưu ý: biên độ {({'HOSE': '±7%', 'HNX': '±10%'}.get(mkt.get('exchange', 'HOSE'), '±15%'))} và T+2.5.",
        "Trả JSON theo schema. Nếu không đủ điều kiện và chưa có vị thế → CHỜ với entry_zone/stop_loss/initial_target=null.",
        "Nếu đang có vị thế, tuyệt đối không trả MUA; dùng GIỮ/GIA_TĂNG/GIẢM/BÁN theo state machine.",
    ]
    return "\n".join(lines)


def _format_pv_section(pv: dict) -> list[str]:
    """Format price-volume intelligence data for the LLM prompt."""
    if not pv:
        return []
    score      = pv.get("price_volume_score", 0)
    bias       = pv.get("entry_bias", "neutral")
    flags      = pv.get("risk_flags", [])
    tags       = pv.get("setup_tags", [])
    interp     = pv.get("interpretation", "")
    metrics    = pv.get("metrics", {})
    accum      = metrics.get("accumulation_count_20", "?")
    distr      = metrics.get("distribution_count_20", "?")
    vol_ratio  = metrics.get("volume_ratio_20", "?")
    lines = [
        "=== Price-Volume Intelligence (rule-based) ===",
        f"Score: {score}/100 range (-100..+100) | Bias: {bias}",
        f"Volume ratio (cur/MA20): {vol_ratio}× | Accum days/20: {accum} | Distr days/20: {distr}",
        f"Interpretation: {interp}",
    ]
    if flags:
        lines.append(f"⚠ Risk flags: {', '.join(flags)}")
    if tags:
        lines.append(f"Setup tags: {', '.join(tags)}")
    lines.append(
        "Lưu ý: FAKE_BREAKOUT_RISK/HEAVY_DISTRIBUTION đã bị chặn ở risk_trade. "
        "BUYING_CLIMAX và BEARISH_VOLUME_EXPANSION → xem xét SL chặt hơn."
    )
    lines.append("")
    return lines


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
        lines.append(
            "STATE=OPEN_POSITION: không phát MUA mới. Hãy đánh giá GIỮ/GIA_TĂNG/GIẢM/BÁN "
            "dựa trên vị thế hiện tại, SL, target, trend và tín hiệu phân phối."
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
    news_summary = external.get("news_summary", "")
    if fund_summary or hist_stats or news_summary:
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
        if news_summary:
            lines.append(f"[Tin tức gần đây]: {news_summary}")

    lines.append("")
    return lines


def _validate(result: dict, state: dict) -> dict:
    action = str(result.get("action", "CHỜ")).upper().strip()
    if action not in _ALL_ACTIONS:
        action = "CHỜ"
    result["action"] = action

    internal = (state.get("memory_context") or {}).get("internal", {})
    has_position = bool(internal.get("has_position"))
    result["position_state"] = "OPEN_POSITION" if has_position else "NO_POSITION"

    if has_position and action == "MUA":
        result.update({
            "action": "GIỮ",
            "entry_zone": None,
            "rr_ratio": None,
            "position_pct": 0,
            "primary_reason": "Đang có vị thế nên không phát MUA mới; chuyển sang quản trị vị thế.",
        })
        action = "GIỮ"

    if not has_position and action in _POSITION_ACTIONS:
        result.update({
            "action": "CHỜ",
            "entry_zone": None,
            "stop_loss": None,
            "initial_target": None,
            "rr_ratio": None,
            "position_pct": 0,
            "primary_reason": f"Chưa có vị thế nên action {action} không hợp lệ; chuyển về CHỜ.",
        })
        action = "CHỜ"

    confidence = result.get("confidence", "THẤP")
    max_pos = {"THẤP": 2, "TRUNG_BÌNH": 3, "CAO": 5}.get(confidence, 2)
    result["position_pct"] = min(int(result.get("position_pct") or 0), max_pos)

    # Tính rr_ratio từ initial_target (không phải hard TP)
    if action in ("MUA", "GIA_TĂNG") and result.get("entry_zone") and result.get("stop_loss") and result.get("initial_target"):
        ez = result["entry_zone"]
        entry_mid = (ez[0] + ez[1]) / 2
        sl  = result["stop_loss"]
        tgt = result["initial_target"]
        if entry_mid > sl:
            result["rr_ratio"] = round((tgt - entry_mid) / (entry_mid - sl), 2)

        # Hard rule: không mua gần kháng cự
        tech = state.get("technical_analysis", {})
        resistances = tech.get("resistance_levels", [])
        if resistances:
            nearest_res = resistances[0]
            room_pct = (nearest_res - ez[1]) / ez[1] * 100
            if room_pct < 3:
                next_action = "GIỮ" if has_position else "CHỜ"
                print(f"[trader_trade] ⚠ entry_high quá gần R ({room_pct:.1f}%) → {next_action}")
                result.update({
                    "action": next_action, "entry_zone": None, "stop_loss": None,
                    "initial_target": None, "rr_ratio": None, "position_pct": 0,
                    "primary_reason": f"Entry quá gần R ({room_pct:.1f}%) — chờ pullback.",
                })

    if result.get("action") in ("CHỜ", "TRÁNH", "GIỮ", "GIẢM", "BÁN"):
        result.setdefault("entry_zone", None)
        if result.get("action") in ("CHỜ", "TRÁNH"):
            result.setdefault("stop_loss", None)
            result.setdefault("initial_target", None)
        result.setdefault("rr_ratio", None)
        result["position_pct"] = 0

    result.setdefault("trail_sl_guide", {})
    result.setdefault("reversal_watchlist", [])
    result.setdefault("management_plan", None)
    result.setdefault("risks", [])
    result.setdefault("trader_note", "")
    return result


def _fallback(err: str) -> dict:
    return {
        "action": "CHỜ", "position_state": "NO_POSITION",
        "entry_zone": None, "stop_loss": None, "initial_target": None,
        "rr_ratio": None, "position_pct": 0,
        "trail_sl_guide": {}, "reversal_watchlist": [], "management_plan": None,
        "confidence": "THẤP", "primary_reason": f"LLM lỗi — CHỜ. {err}".strip(),
        "risks": ["trader_trade fail"], "trader_note": "Fallback.",
    }


def _fmt(v) -> str:
    return "N/A" if v is None else f"{v:,.2f}"


def _fmt_pct(v) -> str:
    if v is None or v == "?":
        return "?"
    try:
        return f"{float(v):+.2f}%"
    except (TypeError, ValueError):
        return str(v)
