"""journal_embed.py — Discord embed cho Trade Journal (hậu kiểm tuần).

Màu tím (#9B59B6) để phân biệt với invest (xanh teal) / trade (cam).
"""

from __future__ import annotations


_PURPLE = 0x9B59B6
_RED = 0xFF4444
_GREEN = 0x00CC66


def build_journal_embed(journal: dict) -> dict:
    """Dựng Discord embed từ output của trade_journal_agent.analyze()."""
    period = journal.get("period_days", 30)
    n = journal.get("total_trades", 0)
    wr = journal.get("win_rate", 0.0)
    pf = journal.get("profit_factor", 0.0)
    total_pnl = journal.get("total_pnl", 0.0) or 0.0
    headline = journal.get("headline", "")

    # Màu theo profit factor
    color = _GREEN if pf >= 1.0 else (_RED if n > 0 else _PURPLE)
    pnl_sign = "+" if total_pnl >= 0 else ""

    fields = [
        {"name": "Số lệnh", "value": str(n), "inline": True},
        {"name": "Win rate", "value": f"{wr:.0f}%", "inline": True},
        {"name": "Profit factor", "value": f"{pf:.2f}", "inline": True},
    ]

    if n > 0:
        fields.append({
            "name": "Avg win / loss",
            "value": f"+{journal.get('avg_win_pct', 0):.1f}% / {journal.get('avg_loss_pct', 0):.1f}%",
            "inline": True,
        })
        fields.append({
            "name": "Tổng P&L",
            "value": f"{pnl_sign}{total_pnl:,.0f} VNĐ",
            "inline": True,
        })
        best = journal.get("best_trade") or {}
        worst = journal.get("worst_trade") or {}
        if best.get("symbol"):
            fields.append({
                "name": "Lãi / Lỗ nhất",
                "value": f"🟢 {best.get('symbol')} {best.get('pnl_pct'):+.1f}% | "
                         f"🔴 {worst.get('symbol')} {worst.get('pnl_pct'):+.1f}%",
                "inline": True,
            })

    lessons = journal.get("lessons") or []
    if lessons:
        fields.append({
            "name": "📌 Bài học",
            "value": "\n".join(f"• {x}" for x in lessons[:4])[:1000],
            "inline": False,
        })

    suggestions = journal.get("threshold_suggestions") or []
    if suggestions:
        fields.append({
            "name": "🔧 Đề xuất chỉnh ngưỡng",
            "value": "\n".join(f"• {x}" for x in suggestions[:3])[:1000],
            "inline": False,
        })

    worst_pattern = journal.get("worst_pattern")
    if worst_pattern:
        fields.append({
            "name": "⚠️ Nhóm lỗ nặng nhất",
            "value": str(worst_pattern)[:200],
            "inline": False,
        })

    return {
        "title": f"📓 Trade Journal — {period} ngày gần nhất",
        "description": headline or "Hậu kiểm giao dịch",
        "color": color,
        "fields": fields,
        "footer": {"text": "AI Trading Assistant — Post-mortem"},
    }
