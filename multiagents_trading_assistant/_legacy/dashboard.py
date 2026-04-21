"""
dashboard.py — Streamlit dashboard 4 tabs cho AI Trading Assistant.

Tab 1: 📊 Daily Brief      — Macro context + market overview
Tab 2: 🎯 Trading Signals  — Signals hôm nay + lịch sử
Tab 3: 💼 Portfolio        — Vị thế + P&L
Tab 4: 📰 News & Sentiment — Tin tức + credibility

Chạy:
  py -3.11 -m streamlit run multiagents_trading_assistant/dashboard.py
"""

import importlib.metadata  # FIX: pandas-ta-openbb
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from multiagents_trading_assistant import database as db

_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
_TODAY = datetime.now(tz=_VN_TZ).strftime("%Y-%m-%d")
_CACHE_DIR = Path(__file__).parent / "cache"


# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="AI Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS tối giản
st.markdown("""
<style>
    .stMetric { background: #1e2130; border-radius: 8px; padding: 12px; }
    .action-mua { color: #00c853; font-weight: bold; font-size: 1.1em; }
    .action-ban { color: #ff1744; font-weight: bold; font-size: 1.1em; }
    .action-cho { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_macro(date: str) -> dict:
    """Đọc macro cache cho ngày cụ thể."""
    path = _CACHE_DIR / f"macro_{date}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


@st.cache_data(ttl=60)
def load_decisions(date: str | None = None, limit: int = 100) -> list[dict]:
    """Load decisions từ DB."""
    db.init_db()
    return db.get_decisions(date=date, limit=limit)


@st.cache_data(ttl=60)
def load_positions() -> list[dict]:
    db.init_db()
    return db.get_all_positions()


@st.cache_data(ttl=120)
def load_news(symbol: str | None = None, days: int = 7) -> list[dict]:
    """Load news từ DB."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    with db.get_connection() as conn:
        if symbol:
            rows = conn.execute("""
                SELECT nh.*, sc.credibility_score AS src_cred, sc.trap_rate
                FROM news_history nh
                LEFT JOIN source_credibility sc ON nh.source = sc.source
                WHERE nh.symbol = ? AND nh.date >= ?
                ORDER BY nh.date DESC, nh.created_at DESC
                LIMIT 50
            """, (symbol, cutoff)).fetchall()
        else:
            rows = conn.execute("""
                SELECT nh.*, sc.credibility_score AS src_cred, sc.trap_rate
                FROM news_history nh
                LEFT JOIN source_credibility sc ON nh.source = sc.source
                WHERE nh.date >= ?
                ORDER BY nh.date DESC, nh.created_at DESC
                LIMIT 100
            """, (cutoff,)).fetchall()
    return [dict(r) for r in rows]


def _action_badge(action: str) -> str:
    colors = {"MUA": "🟢", "BÁN": "🔴", "CHỜ": "🟡"}
    return f"{colors.get(action, '⚪')} {action}"


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    selected_date = st.date_input(
        "Ngày phân tích",
        value=datetime.strptime(_TODAY, "%Y-%m-%d"),
        max_value=datetime.strptime(_TODAY, "%Y-%m-%d"),
    ).strftime("%Y-%m-%d")

    st.divider()
    st.caption(f"📅 Hôm nay: {_TODAY}")
    st.caption(f"🕐 {datetime.now(tz=_VN_TZ).strftime('%H:%M')} VN")

    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.title("📈 AI Trading Assistant")
st.caption(f"VN100 Multi-Agent System — {selected_date}")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Daily Brief",
    "🎯 Trading Signals",
    "💼 Portfolio",
    "📰 News & Sentiment",
])


# ──────────────────────────────────────────────
# Tab 1 — Daily Brief
# ──────────────────────────────────────────────

with tab1:
    macro = load_macro(selected_date)

    if not macro:
        st.warning(f"Chưa có macro data cho {selected_date}. Chạy macro agent trước.")
        st.code(f"py -3.11 multiagents_trading_assistant/main.py --macro-only --date {selected_date}")
    else:
        bias  = macro.get("macro_bias", "NEUTRAL")
        score = macro.get("macro_score", 0)
        conf  = macro.get("confidence", "?")

        bias_color = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(bias, "⚪")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Macro Bias", f"{bias_color} {bias}", f"Score {score:+d}")
        with col2:
            st.metric("Độ tin cậy", conf)
        with col3:
            ec = macro.get("expert_consensus", "?")
            st.metric("Chuyên gia", ec)
        with col4:
            horizon = macro.get("time_horizon", "?")
            st.metric("Horizon", horizon)

        st.divider()

        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("🌍 Global Context")
            st.info(macro.get("global_summary", "—"))

            st.subheader("🇻🇳 VN Market")
            st.info(macro.get("vn_summary", "—"))

            if macro.get("reasoning"):
                with st.expander("📝 Phân tích chi tiết"):
                    st.write(macro["reasoning"])

        with col_r:
            beneficiary = macro.get("beneficiary_sectors", [])
            affected    = macro.get("affected_sectors", [])

            if beneficiary:
                st.subheader("✅ Sector hưởng lợi")
                for s in beneficiary[:5]:
                    st.success(s)

            if affected:
                st.subheader("⚠️ Sector chịu áp lực")
                for s in affected[:5]:
                    st.error(s)

        risks = macro.get("key_risks", [])
        if risks:
            st.subheader("🚨 Key Risks")
            for r in risks[:3]:
                st.warning(r)

        experts = macro.get("experts_cited", [])
        if experts:
            with st.expander(f"👥 Chuyên gia đã tham khảo ({len(experts)})"):
                for e in experts[:10]:
                    st.caption(f"• {e}")


# ──────────────────────────────────────────────
# Tab 2 — Trading Signals
# ──────────────────────────────────────────────

with tab2:
    decisions = load_decisions(date=selected_date)

    if not decisions:
        st.info(f"Chưa có signals cho {selected_date}.")
        st.code(f"py -3.11 multiagents_trading_assistant/main.py --date {selected_date} --no-discord")
    else:
        # Summary metrics
        mua = [d for d in decisions if d.get("final_action") == "MUA"]
        ban = [d for d in decisions if d.get("final_action") == "BÁN"]
        cho = [d for d in decisions if d.get("final_action") == "CHỜ"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tổng phân tích", len(decisions))
        c2.metric("🟢 MUA", len(mua))
        c3.metric("🔴 BÁN", len(ban))
        c4.metric("🟡 CHỜ", len(cho))

        st.divider()

        # Filter
        filter_action = st.selectbox("Lọc theo action:", ["Tất cả", "MUA", "BÁN", "CHỜ"])
        if filter_action != "Tất cả":
            decisions = [d for d in decisions if d.get("final_action") == filter_action]

        # Table
        if decisions:
            rows = []
            for d in decisions:
                rows.append({
                    "Mã":         d.get("symbol", "?"),
                    "Action":     _action_badge(d.get("final_action", "?")),
                    "Strategy":   d.get("strategy", "—"),
                    "Confidence": d.get("confidence", "—"),
                    "Entry":      f"{d.get('entry'):,.0f}" if d.get("entry") else "—",
                    "SL":         f"{d.get('sl'):,.0f}" if d.get("sl") else "—",
                    "TP":         f"{d.get('tp'):,.0f}" if d.get("tp") else "—",
                    "%NAV":       f"{d.get('nav_pct', 0):.1f}%" if d.get("nav_pct") else "—",
                    "Override":   d.get("override_reason", "—") or "—",
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Detail expanders
        st.subheader("Chi tiết từng mã")
        for d in decisions:
            action = d.get("final_action", "?")
            symbol = d.get("symbol", "?")
            badge  = _action_badge(action)

            with st.expander(f"{badge} **{symbol}** — Score {d.get('quality_score', '?')} | {d.get('confidence', '?')}"):
                full = d.get("full_output")
                if isinstance(full, dict):
                    trader = full.get("trader_decision", {})
                    ptkt   = full.get("ptkt_analysis", {})
                    debate = full.get("debate_synthesis", {})

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Lý do MUA:** {trader.get('primary_reason','—')}")
                        if trader.get("risks"):
                            st.markdown(f"**Rủi ro:** {', '.join(str(r) for r in trader['risks'][:3])}")
                    with c2:
                        st.markdown(f"**PTKT:** {ptkt.get('ma_trend','?')} | Score {ptkt.get('confluence_score','?')}")
                        if debate.get("key_risk"):
                            st.markdown(f"**Key risk:** {debate['key_risk']}")

                if d.get("override_reason"):
                    st.warning(f"⚠️ Override: {d['override_reason']}")


# ──────────────────────────────────────────────
# Tab 3 — Portfolio
# ──────────────────────────────────────────────

with tab3:
    positions = load_positions()

    if not positions:
        st.info("Chưa có vị thế nào. Vị thế sẽ hiển thị sau khi thực hiện lệnh MUA.")
    else:
        st.subheader(f"💼 Vị thế đang giữ ({len(positions)} mã)")

        total_nav = sum(p.get("nav_pct", 0) for p in positions)
        c1, c2, c3 = st.columns(3)
        c1.metric("Số vị thế", len(positions))
        c2.metric("NAV đã dùng", f"{total_nav:.1f}%")
        c3.metric("NAV còn lại", f"{100 - total_nav:.1f}%")

        # Bảng positions
        rows = []
        for p in positions:
            rows.append({
                "Mã":         p.get("symbol"),
                "Sàn":        p.get("exchange"),
                "Chiến lược": p.get("strategy", "—"),
                "Entry":      f"{p.get('entry_price'):,.0f}",
                "SL":         f"{p.get('sl'):,.0f}" if p.get("sl") else "—",
                "TP":         f"{p.get('tp'):,.0f}" if p.get("tp") else "—",
                "SL Qty":     p.get("quantity", 0),
                "%NAV":       f"{p.get('nav_pct', 0):.1f}%",
                "Ngày vào":   p.get("entry_date"),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # Lịch sử giao dịch
    st.subheader("📋 Lịch sử giao dịch gần đây")
    trades = db.get_trade_history(limit=30)

    if trades:
        rows = []
        for t in trades:
            rows.append({
                "Ngày":       t.get("trade_date"),
                "Mã":         t.get("symbol"),
                "Action":     _action_badge(t.get("action", "?")),
                "Giá":        f"{t.get('price'):,.0f}",
                "SL lượng":   t.get("quantity"),
                "Chiến lược": t.get("strategy", "—"),
                "Ghi chú":    t.get("note", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có lịch sử giao dịch.")

    # Win rate thống kê
    st.divider()
    st.subheader("📊 Thống kê decisions")
    all_decisions = load_decisions(limit=200)

    if all_decisions:
        action_counts = {}
        for d in all_decisions:
            a = d.get("final_action", "?")
            action_counts[a] = action_counts.get(a, 0) + 1

        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng decisions", len(all_decisions))
        c2.metric("🟢 MUA tổng", action_counts.get("MUA", 0))
        c3.metric("🔴 BÁN tổng", action_counts.get("BÁN", 0))


# ──────────────────────────────────────────────
# Tab 4 — News & Sentiment
# ──────────────────────────────────────────────

with tab4:
    st.subheader("📰 Tin tức & Sentiment")

    col_filter1, col_filter2 = st.columns([1, 3])
    with col_filter1:
        news_days = st.selectbox("Số ngày:", [1, 3, 7, 14, 30], index=2)
    with col_filter2:
        news_symbol = st.text_input("Lọc theo mã (bỏ trống = tất cả):", "").upper().strip()

    news = load_news(symbol=news_symbol or None, days=news_days)

    if not news:
        st.info(f"Chưa có tin tức trong {news_days} ngày qua.")
    else:
        # Summary
        suspicious = [n for n in news if n.get("is_suspicious")]
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng tin", len(news))
        c2.metric("⚠️ Nghi ngờ", len(suspicious))
        c3.metric("Nguồn", len({n.get("source") for n in news}))

        # Source credibility table
        with st.expander("📊 Độ tin cậy nguồn"):
            with db.get_connection() as conn:
                sc_rows = conn.execute(
                    "SELECT * FROM source_credibility ORDER BY credibility_score DESC"
                ).fetchall()
            if sc_rows:
                sc_data = []
                for r in sc_rows:
                    r = dict(r)
                    sc_data.append({
                        "Nguồn":         r.get("source"),
                        "Tổng tin":      r.get("total_news", 0),
                        "Chính xác T5":  r.get("correct_t5", 0),
                        "Trap":          r.get("trap_count", 0),
                        "Credibility":   f"{r.get('credibility_score', 0.5):.2f}",
                        "Trap rate":     f"{r.get('trap_rate', 0):.1%}",
                        "Cập nhật":      r.get("last_updated", "—"),
                    })
                st.dataframe(pd.DataFrame(sc_data), use_container_width=True, hide_index=True)
            else:
                st.info("Chưa có dữ liệu credibility.")

        # News table
        st.subheader("Danh sách tin")
        for n in news[:30]:
            is_sus    = n.get("is_suspicious", 0)
            sentiment = n.get("sentiment", "NEUTRAL")
            cred      = n.get("credibility_score") or n.get("src_cred", 0.5)

            sent_emoji = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🟡"}.get(sentiment, "⚪")
            sus_badge  = " ⚠️" if is_sus else ""

            label = f"{sent_emoji} **{n.get('symbol','?')}** [{n.get('source','?')}] {n.get('date','?')}{sus_badge}"

            with st.expander(f"{label} — {n.get('headline','')[:80]}..."):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.write(n.get("headline", ""))
                    if n.get("content"):
                        st.caption(n["content"][:300])
                    if n.get("url"):
                        st.markdown(f"[Đọc tiếp]({n['url']})")
                with c2:
                    st.metric("Credibility", f"{float(cred):.2f}" if cred else "—")
                    if n.get("trap_rate") is not None:
                        st.metric("Trap rate", f"{n['trap_rate']:.1%}")
                    if is_sus:
                        st.error("⚠️ Tin nghi ngờ")
