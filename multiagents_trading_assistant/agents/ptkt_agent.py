"""
ptkt_agent.py — Phân tích kỹ thuật (PTKT) cho một mã cổ phiếu.

Model: Haiku (run_agent_lite) — nhanh + prompt caching
Input:  symbol, OHLCV DataFrame, indicators dict (từ fetcher + indicators)
Output: dict theo schema agents.md #2
"""

import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
import json
from datetime import datetime

from multiagents_trading_assistant.agent import run_agent_lite
from multiagents_trading_assistant import fetcher, indicators


# ── System prompt — cache lại sau lần đầu ──
_SYSTEM_PROMPT = """Bạn là chuyên gia phân tích kỹ thuật chứng khoán Việt Nam.
Nhiệm vụ: phân tích các chỉ báo kỹ thuật được cung cấp và trả về đánh giá JSON.

Quy tắc bắt buộc:
- Trả về JSON hợp lệ DUY NHẤT, không có text thừa bên ngoài
- Tất cả field phải có giá trị (null nếu không có dữ liệu)
- confluence_score: 0–10 (tính theo: +2 RSI 40-70, +2 giá>MA stack, +2 MACD tăng, +2 volume>TB20, +2 gần support)
- setup_quality: "TỐT" (score>=7) | "TRUNG_BÌNH" (score 4-6) | "YẾU" (score<=3)
- ma_trend: "UPTREND" | "SIDEWAY" | "DOWNTREND"
- ma_phase: "HEALTHY" | "PULLBACK" | "OVERBOUGHT" | "UNKNOWN"
- macd_signal: "BULLISH" | "BEARISH" | "NEUTRAL" | "UNKNOWN"
- bollinger_position: "UPPER" | "UPPER_HALF" | "LOWER_HALF" | "LOWER" | "UNKNOWN"
- rsi_signal: "OVERBOUGHT" | "BULLISH" | "NEUTRAL" | "BEARISH" | "OVERSOLD" | "UNKNOWN"

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
  "technical_summary": <str — 1-2 câu tiếng Việt tóm tắt>
}"""


def analyze(symbol: str, date: str | None = None) -> dict:
    """
    Chạy PTKT agent cho một mã.

    Args:
        symbol: Mã cổ phiếu (VD: "VNM")
        date:   Ngày phân tích (YYYY-MM-DD), mặc định hôm nay

    Returns:
        dict theo schema ptkt_agent (agents.md #2)
        Trả về dict rỗng nếu lỗi.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    print(f"[ptkt_agent] Bắt đầu phân tích kỹ thuật {symbol} ({date})...")

    # ── Lấy dữ liệu ──
    df = fetcher.get_ohlcv(symbol)
    if df.empty:
        print(f"[ptkt_agent] Không lấy được OHLCV cho {symbol}, bỏ qua.")
        return _empty_result()

    ind = indicators.compute_indicators(df)
    if not ind:
        print(f"[ptkt_agent] Không tính được indicators cho {symbol}, bỏ qua.")
        return _empty_result()

    # ── Build prompt ──
    current_price = ind.get("current_price")
    prompt = f"""Phân tích kỹ thuật mã {symbol} ngày {date}.

Giá hiện tại: {_fmt(current_price)} VNĐ

=== Moving Averages ===
MA20:  {_fmt(ind.get('ma20'))}
MA60:  {_fmt(ind.get('ma60'))}
MA200: {_fmt(ind.get('ma200'))}
Xu hướng MA: {ind.get('ma_trend', 'UNKNOWN')}
Pha MA: {ind.get('ma_phase', 'UNKNOWN')}

=== Momentum ===
RSI(14): {_fmt(ind.get('rsi'))}
Tín hiệu RSI: {ind.get('rsi_signal', 'UNKNOWN')}
MACD line:   {_fmt(ind.get('macd'))}
MACD signal: {_fmt(ind.get('macd_signal_label'))}
MACD hist:   {_fmt(ind.get('macd_hist'))}
Tín hiệu MACD: {ind.get('macd_signal_label', 'UNKNOWN')}

=== Bollinger Bands (20,2) ===
Upper: {_fmt(ind.get('bb_upper'))}
Mid:   {_fmt(ind.get('bb_mid'))}
Lower: {_fmt(ind.get('bb_lower'))}
Vị trí: {ind.get('bollinger_position', 'UNKNOWN')}

=== Volume ===
Volume hiện tại: {_fmt_vol(ind.get('volume_current'))}
Volume TB20:     {_fmt_vol(ind.get('volume_ma20'))}
Volume surge (>1.5×TB20): {ind.get('volume_surge', False)}

=== ATR(14) ===
ATR: {_fmt(ind.get('atr'))}

=== Support / Resistance ===
Support:    {ind.get('support_levels', [])}
Resistance: {ind.get('resistance_levels', [])}

=== Confluence Score (pre-computed) ===
Score: {ind.get('confluence_score', 0)}/10

Hãy xác nhận lại confluence_score và viết technical_summary 1-2 câu tiếng Việt.
Trả về JSON theo schema đã định."""

    # ── Gọi LLM ──
    try:
        result = run_agent_lite(prompt=prompt, system=_SYSTEM_PROMPT)
        print(f"[ptkt_agent] {symbol} — confluence_score={result.get('confluence_score')}, "
              f"setup_quality={result.get('setup_quality')}")
        return result
    except Exception as e:
        print(f"[ptkt_agent] Lỗi gọi LLM cho {symbol}: {e}")
        # Fallback: trả về kết quả từ indicators (không có LLM summary)
        return _fallback_result(ind)


def _fmt(val) -> str:
    """Format số float cho prompt."""
    if val is None:
        return "N/A"
    return f"{val:,.2f}"


def _fmt_vol(val) -> str:
    """Format volume (triệu)."""
    if val is None:
        return "N/A"
    return f"{val/1_000_000:.1f}M"


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
    """Trả về kết quả từ indicators khi LLM fail."""
    score = ind.get("confluence_score", 0)
    quality = "TỐT" if score >= 7 else ("TRUNG_BÌNH" if score >= 4 else "YẾU")
    return {
        "rsi": ind.get("rsi"),
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
        "technical_summary": f"MA trend: {ind.get('ma_trend')}, RSI: {ind.get('rsi'):.1f}" if ind.get('rsi') else "Dữ liệu indicators.",
    }
