"""technical_agent.py — Phân tích kỹ thuật cho Trade pipeline.

Model: Haiku. Port từ _legacy/agents/ptkt_agent.py (không đổi logic).
"""

import importlib.metadata  # noqa: F401
from datetime import datetime

from multiagents_trading_assistant.services.llm_service import run_agent_lite
from multiagents_trading_assistant import fetcher, indicators


_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích kỹ thuật chứng khoán Việt Nam.
Nhiệm vụ: phân tích các chỉ báo kỹ thuật và trả về đánh giá JSON.

Quy tắc bắt buộc:
- Trả về JSON hợp lệ DUY NHẤT, không có text thừa
- confluence_score: 0-10 (+2 RSI 40-70, +2 giá>MA stack, +2 MACD tăng, +2 volume>TB20, +2 gần support)
- setup_quality: "TỐT" (>=7) | "TRUNG_BÌNH" (4-6) | "YẾU" (<=3)
- ma_trend: UPTREND | SIDEWAY | DOWNTREND
- ma_phase: HEALTHY | PULLBACK | OVERBOUGHT | UNKNOWN
- macd_signal: BULLISH | BEARISH | NEUTRAL | UNKNOWN
- bollinger_position: UPPER | UPPER_HALF | LOWER_HALF | LOWER | UNKNOWN
- rsi_signal: OVERBOUGHT | BULLISH | NEUTRAL | BEARISH | OVERSOLD | UNKNOWN

Output schema:
{
  "rsi": <float|null>,
  "rsi_signal": <str>,
  "ma_trend": <str>,
  "ma_phase": <str>,
  "macd_signal": <str>,
  "bollinger_position": <str>,
  "atr": <float|null>,
  "support_levels": [<float>, ...],
  "resistance_levels": [<float>, ...],
  "confluence_score": <int 0-10>,
  "setup_quality": <str>,
  "technical_summary": <str — 1-2 câu tiếng Việt>
}"""


def analyze(symbol: str, date: str | None = None) -> dict:
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    print(f"[technical_agent] {symbol} ({date})")

    df = fetcher.get_ohlcv(symbol)
    if df.empty:
        return _empty_result()

    ind = indicators.compute_indicators(df)
    if not ind:
        return _empty_result()

    prompt = f"""Phân tích kỹ thuật mã {symbol} ngày {date}.

Giá: {_fmt(ind.get('current_price'))}

=== MA ===
MA20: {_fmt(ind.get('ma20'))} | MA60: {_fmt(ind.get('ma60'))} | MA200: {_fmt(ind.get('ma200'))}
Trend: {ind.get('ma_trend', 'UNKNOWN')} | Phase: {ind.get('ma_phase', 'UNKNOWN')}

=== Momentum ===
RSI: {_fmt(ind.get('rsi'))} ({ind.get('rsi_signal', 'UNKNOWN')})
MACD: {_fmt(ind.get('macd'))} / {_fmt(ind.get('macd_signal'))} / hist {_fmt(ind.get('macd_hist'))}
MACD label: {ind.get('macd_signal_label', 'UNKNOWN')}

=== Bollinger (20,2) ===
U {_fmt(ind.get('bb_upper'))} | M {_fmt(ind.get('bb_mid'))} | L {_fmt(ind.get('bb_lower'))}
Pos: {ind.get('bollinger_position', 'UNKNOWN')}

=== Volume ===
Cur {_fmt_vol(ind.get('volume_current'))} | TB20 {_fmt_vol(ind.get('volume_ma20'))}
Surge: {ind.get('volume_surge', False)}

=== ATR ===
{_fmt(ind.get('atr'))}

=== S/R ===
S: {ind.get('support_levels', [])}
R: {ind.get('resistance_levels', [])}

=== Pre-computed confluence ===
Score: {ind.get('confluence_score', 0)}/10

Xác nhận confluence_score và viết technical_summary 1-2 câu tiếng Việt.
Trả JSON theo schema."""

    try:
        result = run_agent_lite(prompt=prompt, system=_SYSTEM_PROMPT)
        print(f"[technical_agent] {symbol} — conf={result.get('confluence_score')}, q={result.get('setup_quality')}")
        return result
    except Exception as e:
        print(f"[technical_agent] LLM error: {e}")
        return _fallback_result(ind)


def _fmt(val) -> str:
    return "N/A" if val is None else f"{val:,.2f}"


def _fmt_vol(val) -> str:
    return "N/A" if val is None else f"{val/1_000_000:.1f}M"


def _empty_result() -> dict:
    return {
        "rsi": None, "rsi_signal": "UNKNOWN",
        "ma_trend": "UNKNOWN", "ma_phase": "UNKNOWN",
        "macd_signal": "UNKNOWN", "bollinger_position": "UNKNOWN",
        "atr": None, "support_levels": [], "resistance_levels": [],
        "confluence_score": 0, "setup_quality": "YẾU",
        "technical_summary": "Không có dữ liệu kỹ thuật.",
    }


def _fallback_result(ind: dict) -> dict:
    score = ind.get("confluence_score", 0)
    quality = "TỐT" if score >= 7 else ("TRUNG_BÌNH" if score >= 4 else "YẾU")
    rsi_val = ind.get("rsi")
    return {
        "rsi": rsi_val,
        "rsi_signal": ind.get("rsi_signal", "UNKNOWN"),
        "ma_trend": ind.get("ma_trend", "UNKNOWN"),
        "ma_phase": ind.get("ma_phase", "UNKNOWN"),
        "macd_signal": ind.get("macd_signal_label", "UNKNOWN"),
        "bollinger_position": ind.get("bollinger_position", "UNKNOWN"),
        "atr": ind.get("atr"),
        "support_levels": ind.get("support_levels", []),
        "resistance_levels": ind.get("resistance_levels", []),
        "confluence_score": score,
        "setup_quality": quality,
        "technical_summary": f"MA trend {ind.get('ma_trend')}, RSI {rsi_val:.1f}" if rsi_val is not None else "Fallback indicators.",
    }
