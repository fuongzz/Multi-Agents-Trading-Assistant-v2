"""trade_journal_agent.py — Hậu kiểm giao dịch (Trade Journal / Post-mortem).

Model: Haiku. Đây là LỚP NARRATIVE phủ lên phần định lượng đã có sẵn:
  - database.get_closed_trades()       → lệnh đã đóng
  - database.aggregate_loss_patterns() → nhóm lỗ theo (setup × regime × loss_category)
  - agentic/post_trade_review.py       → classify_loss() (đã chạy lúc close_position)

Agent tính thống kê tổng hợp (win rate, profit factor, best/worst), rồi gọi LLM
sinh bài học tiếng Việt + đề xuất chỉnh ngưỡng. Có fallback rule-based khi LLM lỗi.

Chạy sau Bob (Thứ 6 20:00) trong pipeline_runner — Bob đã reclassify + aggregate.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from multiagents_trading_assistant.services.llm_service import run_agent_lite
from multiagents_trading_assistant import database


_SYSTEM_PROMPT = """Bạn là chuyên gia hậu kiểm giao dịch (trading post-mortem) thị trường chứng khoán VN.

Nhiệm vụ: đọc thống kê hiệu suất + các nhóm lỗ đã được phân loại, rút ra BÀI HỌC hành động được.

Quy tắc bắt buộc:
- Trả về JSON hợp lệ DUY NHẤT, không markdown.
- lessons: 2-4 bài học ngắn gọn, cụ thể, hành động được (tiếng Việt). Mỗi bài ≤ 140 ký tự.
- threshold_suggestions: 0-3 đề xuất chỉnh ngưỡng cụ thể (vd "siết confluence min UPTREND 62→66 do FALSE_BREAKOUT chiếm đa số"). Chỉ đề xuất khi dữ liệu đủ rõ.
- headline: 1 câu tóm tắt tuần (≤ 120 ký tự).
- Tuyệt đối KHÔNG bịa số liệu. Chỉ diễn giải con số được cung cấp.
- Nếu mẫu quá nhỏ (< 5 lệnh đóng), nói rõ "mẫu nhỏ, chưa kết luận" trong headline.

Các loại lỗ (loss_category):
  ENTRY_CHASE (mua đuổi gap), FALSE_BREAKOUT (tín hiệu giả, SL sớm),
  STRUCTURE_BREAK (gãy cấu trúc giữa lệnh), REGIME_SHIFT (thị trường đảo),
  WEAK_SECTOR (ngành yếu), EXIT_TOO_EARLY (thoát non), EXIT_TOO_LATE (thoát muộn),
  DATA_ISSUE, UNCLASSIFIED.

Output schema:
{
  "headline": <str>,
  "lessons": [<str>, ...],
  "threshold_suggestions": [<str>, ...],
  "worst_pattern": <str|null>,
  "best_setup": <str|null>
}"""


def analyze(period_days: int = 30, date: str | None = None) -> dict:
    """Sinh nhật ký giao dịch cho N ngày gần nhất.

    Args:
        period_days: cửa sổ thống kê (mặc định 30 ngày).
        date: ngày tham chiếu YYYY-MM-DD (None = hôm nay). Dùng để cắt cửa sổ.

    Returns:
        dict gồm stats định lượng + narrative LLM. Xem _empty_result() cho schema.
    """
    ref = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()
    cutoff = (ref - timedelta(days=period_days)).strftime("%Y-%m-%d")
    print(f"[trade_journal] period {cutoff} → {ref.strftime('%Y-%m-%d')} ({period_days}d)")

    closed = [
        t for t in database.get_closed_trades(limit=500)
        if str(t.get("trade_date") or "") >= cutoff
    ]
    if not closed:
        print("[trade_journal] không có lệnh đóng trong cửa sổ")
        return _empty_result(period_days)

    stats = _compute_stats(closed)
    loss_patterns = database.aggregate_loss_patterns(days=period_days, min_count=1)

    prompt = _build_prompt(stats, loss_patterns, period_days)
    try:
        llm = run_agent_lite(prompt=prompt, system=_SYSTEM_PROMPT)
    except Exception as e:
        print(f"[trade_journal] LLM error: {e}")
        llm = _fallback_narrative(stats, loss_patterns)

    result = {
        "period_days": period_days,
        "generated_at": ref.strftime("%Y-%m-%d"),
        **stats,
        "loss_patterns": loss_patterns[:10],
        "headline": llm.get("headline", ""),
        "lessons": llm.get("lessons", []) or [],
        "threshold_suggestions": llm.get("threshold_suggestions", []) or [],
        "worst_pattern": llm.get("worst_pattern"),
        "best_setup": llm.get("best_setup") or stats.get("best_setup"),
    }
    print(
        f"[trade_journal] n={stats['total_trades']} WR={stats['win_rate']:.0f}% "
        f"PF={stats['profit_factor']:.2f} → {len(result['lessons'])} bài học"
    )
    return result


# ──────────────────────────────────────────────
# Stats (định lượng — không LLM)
# ──────────────────────────────────────────────

def _compute_stats(closed: list[dict]) -> dict:
    """Tính win rate, profit factor, avg win/loss, best/worst setup từ lệnh đã đóng."""
    n = len(closed)
    pnls = [float(t.get("realized_pnl") or 0.0) for t in closed]
    pcts = [float(t.get("pnl_pct") or 0.0) for t in closed]

    wins = [p for p in pcts if p > 0]
    losses = [p for p in pcts if p <= 0]
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    # Best/worst setup theo tổng pnl
    by_setup: dict[str, float] = {}
    for t in closed:
        key = str(t.get("strategy") or t.get("loss_category") or "?")
        by_setup[key] = by_setup.get(key, 0.0) + float(t.get("realized_pnl") or 0.0)
    best_setup = max(by_setup, key=by_setup.get) if by_setup else None
    worst_setup = min(by_setup, key=by_setup.get) if by_setup else None

    best_trade = max(closed, key=lambda t: float(t.get("pnl_pct") or 0.0))
    worst_trade = min(closed, key=lambda t: float(t.get("pnl_pct") or 0.0))

    return {
        "total_trades": n,
        "win_rate": round(len(wins) / n * 100, 1) if n else 0.0,
        "total_pnl": round(sum(pnls), 0),
        "avg_win_pct": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss_pct": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 99.99,
        "best_setup": best_setup,
        "worst_setup": worst_setup,
        "best_trade": {
            "symbol": best_trade.get("symbol"),
            "pnl_pct": round(float(best_trade.get("pnl_pct") or 0.0), 1),
        },
        "worst_trade": {
            "symbol": worst_trade.get("symbol"),
            "pnl_pct": round(float(worst_trade.get("pnl_pct") or 0.0), 1),
        },
    }


# ──────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────

def _build_prompt(stats: dict, loss_patterns: list[dict], period_days: int) -> str:
    lp_lines = []
    for p in loss_patterns[:10]:
        lp_lines.append(
            f"  {str(p.get('setup'))[:18]:<18} {str(p.get('regime'))[:9]:<9} "
            f"{str(p.get('loss_category'))[:16]:<16} "
            f"n={p.get('count')}  avg={(p.get('avg_pnl_pct') or 0):.2f}%  "
            f"total={(p.get('total_pnl') or 0):,.0f}"
        )
    lp_str = "\n".join(lp_lines) if lp_lines else "  (chưa có nhóm lỗ ≥1 lần)"

    return f"""Hậu kiểm {period_days} ngày gần nhất.

=== Thống kê tổng hợp ===
Số lệnh đóng: {stats['total_trades']}
Win rate: {stats['win_rate']}%
Profit factor: {stats['profit_factor']}
Avg win: {stats['avg_win_pct']}% | Avg loss: {stats['avg_loss_pct']}%
Tổng P&L (VNĐ): {stats['total_pnl']:,.0f}
Setup tốt nhất (tổng P&L): {stats['best_setup']}
Setup tệ nhất (tổng P&L): {stats['worst_setup']}
Lệnh lãi nhất: {stats['best_trade']['symbol']} {stats['best_trade']['pnl_pct']:+.1f}%
Lệnh lỗ nhất: {stats['worst_trade']['symbol']} {stats['worst_trade']['pnl_pct']:+.1f}%

=== Nhóm lỗ (setup × regime × loại lỗi) ===
{lp_str}

Rút ra bài học hành động được. Trả JSON theo schema."""


# ──────────────────────────────────────────────
# Fallback (không LLM)
# ──────────────────────────────────────────────

def _fallback_narrative(stats: dict, loss_patterns: list[dict]) -> dict:
    lessons: list[str] = []
    worst_pattern = None
    if loss_patterns:
        p = loss_patterns[0]
        worst_pattern = f"{p.get('setup')} / {p.get('regime')} / {p.get('loss_category')}"
        lessons.append(
            f"Nhóm lỗ nặng nhất: {worst_pattern} "
            f"({p.get('count')} lần, avg {(p.get('avg_pnl_pct') or 0):.1f}%)."
        )
    if stats["profit_factor"] < 1.0:
        lessons.append("Profit factor < 1 — đang lỗ ròng, cần siết bộ lọc vào lệnh.")
    if stats["win_rate"] < 45:
        lessons.append(f"Win rate {stats['win_rate']}% thấp — xem lại chất lượng setup.")
    if not lessons:
        lessons.append("Hiệu suất ổn định, chưa phát hiện vấn đề hệ thống.")

    return {
        "headline": (
            f"{stats['total_trades']} lệnh, WR {stats['win_rate']}%, "
            f"PF {stats['profit_factor']}"
        ),
        "lessons": lessons,
        "threshold_suggestions": [],
        "worst_pattern": worst_pattern,
        "best_setup": stats.get("best_setup"),
    }


def _empty_result(period_days: int) -> dict:
    return {
        "period_days": period_days,
        "generated_at": datetime.now().strftime("%Y-%m-%d"),
        "total_trades": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "profit_factor": 0.0,
        "best_setup": None,
        "worst_setup": None,
        "best_trade": None,
        "worst_trade": None,
        "loss_patterns": [],
        "headline": "Chưa có lệnh đóng trong kỳ — chưa thể hậu kiểm.",
        "lessons": [],
        "threshold_suggestions": [],
        "worst_pattern": None,
    }


if __name__ == "__main__":
    import json
    database.init_db()
    out = analyze(period_days=90)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
