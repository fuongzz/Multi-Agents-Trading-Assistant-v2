"""
debate_agent.py — Bull vs Bear debate 2 vòng cho một mã cổ phiếu.

Flow:
  Round 1: Bull argument (Sonnet) — dựa trên aggregate analyst output
  Round 1: Bear counter (Sonnet)  — phản bác Bull
  Round 2: Bull rebuttal (Sonnet)
  Round 2: Bear final position (Sonnet)
  Synthesizer (Haiku): tổng hợp điểm mạnh cả 2 → debate_synthesis dict

Model: Sonnet cho Bull/Bear, Haiku cho Synthesizer
"""

import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11

from multiagents_trading_assistant.agent import run_agent, run_agent_lite


# ──────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────

_BULL_SYSTEM = """Bạn là chuyên gia phân tích cổ phiếu vai BULL (lạc quan).
Nhiệm vụ: xây dựng luận điểm MUA thuyết phục dựa trên dữ liệu được cung cấp.

Quy tắc:
- Tập trung vào điểm mạnh: kỹ thuật tốt, FA khỏe, dòng tiền tích cực, sentiment thuận lợi
- Trích dẫn số liệu cụ thể từ dữ liệu đầu vào (RSI, MA, P/E, room%, ...)
- Thừa nhận rủi ro nhưng lập luận tại sao upside lớn hơn downside
- Viết bằng tiếng Việt, rõ ràng, súc tích (150-250 từ)
- Kết thúc bằng 1 câu tóm tắt quan điểm BULL"""

_BEAR_SYSTEM = """Bạn là chuyên gia phân tích cổ phiếu vai BEAR (thận trọng/bi quan).
Nhiệm vụ: phản biện luận điểm Bull và xây dựng case KHÔNG MUA / CHỜ.

Quy tắc:
- Tập trung vào rủi ro: kỹ thuật yếu, định giá cao, dòng tiền bất lợi, macro tiêu cực
- Phản bác trực tiếp các điểm Bull đưa ra
- Trích dẫn số liệu cụ thể để bác bỏ
- Viết bằng tiếng Việt, rõ ràng, súc tích (150-250 từ)
- Kết thúc bằng 1 câu tóm tắt quan điểm BEAR"""

_SYNTHESIZER_SYSTEM = """Bạn là chuyên gia tổng hợp phân tích đầu tư trung lập.
Nhiệm vụ: tổng hợp cuộc tranh luận Bull vs Bear thành đánh giá cân bằng.

Quy tắc bắt buộc:
- Trả về JSON hợp lệ DUY NHẤT, không có text thừa
- bull_key_points: tối đa 3 điểm mạnh nhất của Bull (tiếng Việt, ngắn)
- bear_key_points: tối đa 3 điểm mạnh nhất của Bear (tiếng Việt, ngắn)
- balance: "STRONG_BULL" | "BULL_SLIGHT_EDGE" | "NEUTRAL" | "BEAR_SLIGHT_EDGE" | "STRONG_BEAR"
- key_risk: rủi ro lớn nhất cần theo dõi (1 câu)
- debate_conclusion: kết luận tổng hợp 1-2 câu tiếng Việt

Output schema:
{
  "bull_key_points": [<str>, ...],
  "bear_key_points": [<str>, ...],
  "balance": <str>,
  "key_risk": <str>,
  "debate_conclusion": <str>
}"""


# ──────────────────────────────────────────────
# Public functions — gọi từ orchestrator
# ──────────────────────────────────────────────

def run_bull(state: dict) -> dict:
    """
    Node bull_debate: xây dựng luận điểm BULL Round 1.
    Returns: {"bull_argument": str}
    """
    symbol  = state["symbol"]
    context = _build_analyst_context(state)

    prompt = f"""Xây dựng luận điểm BULL cho mã {symbol}.

{context}

Hãy lập luận tại sao NÊN MUA {symbol} dựa trên dữ liệu trên.
Trả lời bằng tiếng Việt (150-250 từ), kết thúc bằng 1 câu tóm tắt quan điểm."""

    print(f"[debate_agent] Chạy BULL argument cho {symbol}...")
    try:
        # run_agent trả về dict — Bull/Bear trả về text nên dùng cách khác
        bull_text = _run_text_agent(prompt, _BULL_SYSTEM)
        print(f"[debate_agent] BULL done — {len(bull_text)} ký tự")
        return {"bull_argument": bull_text}
    except Exception as e:
        print(f"[debate_agent] Lỗi BULL: {e}")
        return {"bull_argument": f"[BULL stub] Không tạo được luận điểm: {e}"}


def run_bear(state: dict) -> dict:
    """
    Node bear_debate: xây dựng luận điểm BEAR Round 1 + Round 2.
    Bao gồm cả Bear counter (Round 1) và Bear final (Round 2 rebuttal).
    Returns: {"bear_argument": str}
    """
    symbol      = state["symbol"]
    bull_arg    = state.get("bull_argument", "")
    context     = _build_analyst_context(state)

    # Round 1: Bear counter
    prompt_r1 = f"""Phản bác luận điểm BULL sau đây cho mã {symbol}.

=== Dữ liệu phân tích ===
{context}

=== Luận điểm BULL (Round 1) ===
{bull_arg}

Hãy phản bác và xây dựng case KHÔNG MUA / CHỜ.
Trả lời bằng tiếng Việt (150-250 từ)."""

    print(f"[debate_agent] Chạy BEAR counter Round 1 cho {symbol}...")
    try:
        bear_r1 = _run_text_agent(prompt_r1, _BEAR_SYSTEM)
    except Exception as e:
        print(f"[debate_agent] Lỗi BEAR R1: {e}")
        bear_r1 = f"[BEAR R1 lỗi]: {e}"

    # Round 2: Bull rebuttal → Bear final (gộp vào bear_argument để đơn giản)
    prompt_r2 = f"""Đây là Round 2. Bull đã phản hồi luận điểm của bạn.
Đưa ra quan điểm BEAR cuối cùng sau khi xem xét toàn bộ debate.

=== Bear Counter Round 1 của bạn ===
{bear_r1}

=== Context bổ sung ===
{context}

Tóm tắt lập luận BEAR cuối cùng (100-150 từ), nhấn mạnh rủi ro lớn nhất."""

    print(f"[debate_agent] Chạy BEAR final Round 2 cho {symbol}...")
    try:
        bear_r2 = _run_text_agent(prompt_r2, _BEAR_SYSTEM)
    except Exception as e:
        print(f"[debate_agent] Lỗi BEAR R2: {e}")
        bear_r2 = ""

    # Gộp cả 2 rounds
    full_bear = f"=== BEAR Round 1 ===\n{bear_r1}\n\n=== BEAR Final ===\n{bear_r2}" if bear_r2 else bear_r1
    print(f"[debate_agent] BEAR done — {len(full_bear)} ký tự")
    return {"bear_argument": full_bear}


def synthesize(state: dict) -> dict:
    """
    Node synthesize: Haiku tổng hợp Bull vs Bear → debate_synthesis dict.
    Returns: {"debate_synthesis": dict}
    """
    symbol   = state["symbol"]
    bull_arg = state.get("bull_argument", "")
    bear_arg = state.get("bear_argument", "")

    prompt = f"""Tổng hợp cuộc tranh luận Bull vs Bear cho mã {symbol}.

=== Luận điểm BULL ===
{bull_arg[:800]}

=== Luận điểm BEAR ===
{bear_arg[:800]}

Tổng hợp khách quan và trả về JSON theo schema đã định."""

    print(f"[debate_agent] Chạy Synthesizer (Haiku) cho {symbol}...")
    try:
        result = run_agent_lite(prompt=prompt, system=_SYNTHESIZER_SYSTEM)
        print(f"[debate_agent] Synthesis done — balance={result.get('balance')}")
        return {"debate_synthesis": result}
    except Exception as e:
        print(f"[debate_agent] Lỗi Synthesizer: {e}")
        return {"debate_synthesis": _fallback_synthesis(bull_arg, bear_arg)}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _run_text_agent(prompt: str, system: str) -> str:
    """
    Gọi Sonnet và trả về text thô (không parse JSON).
    Bull/Bear arguments là prose, không phải JSON.
    """
    import anthropic
    import os
    from dotenv import load_dotenv
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("Thiếu ANTHROPIC_API_KEY")

    client = anthropic.Anthropic(api_key=api_key)

    @retry(
        retry=retry_if_exception_type((anthropic.APIError, anthropic.APITimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        reraise=True,
    )
    def _call():
        from multiagents_trading_assistant.agent import MODEL_SONNET
        response = client.messages.create(
            model=MODEL_SONNET,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return _call()


def _build_analyst_context(state: dict) -> str:
    """Tóm tắt analyst outputs thành text để đưa vào debate prompt."""
    symbol  = state.get("symbol", "?")
    ptkt    = state.get("ptkt_analysis", {})
    fa      = state.get("fa_analysis", {})
    ff      = state.get("foreign_flow_analysis", {})
    sent    = state.get("sentiment_analysis", {})
    macro   = state.get("macro_context", {})
    setup   = state.get("setup_type", "?")

    lines = [
        f"Mã: {symbol} | Setup: {setup}",
        "",
        "=== PTKT ===",
        f"Xu hướng MA: {ptkt.get('ma_trend','?')} | Pha: {ptkt.get('ma_phase','?')}",
        f"RSI: {ptkt.get('rsi','?')} ({ptkt.get('rsi_signal','?')})",
        f"MACD: {ptkt.get('macd_signal','?')} | Bollinger: {ptkt.get('bollinger_position','?')}",
        f"Confluence score: {ptkt.get('confluence_score','?')}/10 | Quality: {ptkt.get('setup_quality','?')}",
        f"Support: {ptkt.get('support_levels',[])} | Resistance: {ptkt.get('resistance_levels',[])}",
        f"Tóm tắt: {ptkt.get('technical_summary','')}",
        "",
        "=== FA ===",
        f"P/E: {fa.get('pe_ratio','?')} | P/B: {fa.get('pb_ratio','?')} | ROE: {fa.get('roe','?')}%",
        f"EPS growth YoY: {fa.get('eps_growth_yoy','?')}%",
        f"Định giá: {fa.get('valuation','?')} | So ngành: {fa.get('vs_industry','?')} | Sức khỏe: {fa.get('financial_health','?')}",
        f"Tóm tắt: {fa.get('fa_summary','')}",
        "",
        "=== Khối ngoại ===",
        f"Room: {ff.get('room_usage_pct','?')}% ({ff.get('room_status','?')}) | Dòng tiền: {ff.get('flow_trend','?')}",
        f"Net flow 5D: {_fmt_flow(ff.get('net_flow_5d'))} | Tích lũy: {ff.get('accumulation_signal','?')}",
        f"Tóm tắt: {ff.get('foreign_summary','')}",
        "",
        "=== Sentiment ===",
        f"Score: {sent.get('sentiment_score','?')}/100 ({sent.get('sentiment_label','?')}) | {sent.get('news_count','?')} bài",
        f"Tích cực: {sent.get('key_positive',[])}",
        f"Tiêu cực: {sent.get('key_negative',[])}",
        "",
        "=== Macro ===",
        f"Bias: {macro.get('macro_bias','?')} | Score: {macro.get('macro_score','?')}",
        f"Tóm tắt: {macro.get('overall_summary','')}",
    ]
    return "\n".join(lines)


def _fmt_flow(val) -> str:
    if val is None:
        return "N/A"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val/1_000_000_000:.2f} tỷ"


def _fallback_synthesis(bull: str, bear: str) -> dict:
    """Fallback khi Synthesizer fail."""
    return {
        "bull_key_points": [bull[:100]] if bull else [],
        "bear_key_points": [bear[:100]] if bear else [],
        "balance": "NEUTRAL",
        "key_risk": "Không tổng hợp được — xem xét thủ công.",
        "debate_conclusion": "Debate synthesis lỗi — cần review thủ công.",
    }
