"""catalyst_agent.py — Gate sự kiện doanh nghiệp (cả 2 pipeline).

Thuần RULE-BASED (không LLM) — gate cần tính xác định, độ trễ thấp, nhất quán
với money_flow_agent / synthesis_agent. Đọc lịch sự kiện từ
fetcher.get_corporate_events() (nguồn vnstock_data VCI).

Nhận diện rủi ro sự kiện PHÍA TRƯỚC tính từ `date`:
  - EX_DATE_SOON  : GDKHQ (exright_date) ≤ 2 ngày → giá điều chỉnh kỹ thuật,
                    tránh nhầm "thủng SL". Trade pipeline: blackout.
  - AGM_SOON      : ĐHCĐ record_date trong cửa sổ → biến động tin tức.
  - DIVIDEND_SOON : sắp chốt quyền cổ tức tiền mặt.
  - DILUTION      : phát hành cổ phiếu (issue_date) sắp tới → pha loãng.

Anti-leak: chỉ tính sự kiện đã công bố (public_date ≤ as_of_date) khi backtest.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from multiagents_trading_assistant import fetcher


# Cửa sổ nhìn trước (ngày dương lịch)
_EX_DATE_BLACKOUT_DAYS = 2     # GDKHQ ≤ 2 ngày → blackout cho trade
_AGM_WINDOW_DAYS = 5           # ĐHCĐ trong 5 ngày → cảnh báo
_DIVIDEND_WINDOW_DAYS = 5
_DILUTION_WINDOW_DAYS = 15     # phát hành ảnh hưởng dài hơn


def analyze(
    symbol: str,
    date: str | None = None,
    *,
    as_of_date: str | None = None,
    backtest_mode: bool = False,
) -> dict:
    """Phân tích rủi ro sự kiện phía trước cho 1 mã.

    Args:
        date: ngày tham chiếu YYYY-MM-DD (None = hôm nay).
        as_of_date: trần thời gian cho công bố sự kiện (anti-leak). None = live.
        backtest_mode: nếu True và as_of_date None → tự dùng `date` làm trần.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    ref = pd.Timestamp(date).normalize()
    aod = pd.Timestamp(as_of_date).normalize() if as_of_date else (
        ref if backtest_mode else None
    )
    print(f"[catalyst_agent] {symbol} ({date}){' [BT]' if backtest_mode else ''}")

    try:
        events = fetcher.get_corporate_events(symbol)
    except Exception as e:
        print(f"[catalyst_agent] events fail: {e}")
        return _empty_result()

    if events is None or events.empty:
        return _empty_result()

    # Anti-leak: chỉ giữ sự kiện đã công bố tính tới as_of_date
    if aod is not None and "public_date" in events.columns:
        events = events[events["public_date"].isna() | (events["public_date"] <= aod)]
    if events.empty:
        return _empty_result()

    upcoming: list[dict] = []
    upcoming += _scan(events, ref, "exright_date", _EX_DATE_BLACKOUT_DAYS,
                      lambda c: c == "DIVIDEND", "EX_DATE")
    upcoming += _scan(events, ref, "record_date", _AGM_WINDOW_DAYS,
                      lambda c: c == "SHAREHOLDER_MEETING", "AGM")
    upcoming += _scan(events, ref, "payout_date", _DIVIDEND_WINDOW_DAYS,
                      lambda c: c == "DIVIDEND", "DIVIDEND")
    upcoming += _scan(events, ref, "issue_date", _DILUTION_WINDOW_DAYS,
                      lambda c: c == "DIVIDEND" or "Phát hành" in str(c), "DILUTION",
                      name_filter="Phát hành")

    if not upcoming:
        return _empty_result()

    upcoming.sort(key=lambda e: e["days_to_event"])
    nearest = upcoming[0]

    # Blackout (trade): GDKHQ rất gần
    blackout = any(
        e["event_risk"] == "EX_DATE" and e["days_to_event"] <= _EX_DATE_BLACKOUT_DAYS
        for e in upcoming
    )
    dilution = any(e["event_risk"] == "DILUTION" for e in upcoming)

    result = {
        "has_catalyst": True,
        "event_risk": nearest["event_risk"],
        "days_to_event": nearest["days_to_event"],
        "blackout": blackout,
        "dilution_warning": dilution,
        "next_event": nearest["label"],
        "upcoming": upcoming[:5],
        "summary": _summary(nearest, blackout, dilution),
    }
    print(f"[catalyst_agent] {symbol} → {result['event_risk']} "
          f"in {result['days_to_event']}d, blackout={blackout}")
    return result


def _scan(events, ref, date_col, window, cat_pred, risk_label, name_filter=None):
    """Tìm sự kiện có `date_col` trong [ref, ref+window] thỏa category predicate."""
    if date_col not in events.columns:
        return []
    out = []
    for _, r in events.iterrows():
        cat = str(r.get("category") or "")
        if not cat_pred(cat):
            continue
        if name_filter and name_filter not in str(r.get("event_name_vi") or ""):
            continue
        d = r.get(date_col)
        if pd.isna(d):
            continue
        days = (pd.Timestamp(d).normalize() - ref).days
        if 0 <= days <= window:
            out.append({
                "event_risk": risk_label,
                "days_to_event": int(days),
                "date": str(pd.Timestamp(d).date()),
                "label": str(r.get("event_name_vi") or risk_label),
            })
    return out


def _summary(nearest: dict, blackout: bool, dilution: bool) -> str:
    risk = nearest["event_risk"]
    days = nearest["days_to_event"]
    label = nearest["label"]
    if blackout and risk == "EX_DATE":
        return f"GDKHQ trong {days} ngày ({label}) — giá sẽ điều chỉnh kỹ thuật, tránh nhầm thủng SL."
    if dilution:
        return f"Sắp phát hành cổ phiếu ({label}) trong {days} ngày — rủi ro pha loãng."
    if risk == "AGM":
        return f"ĐHCĐ trong {days} ngày — có thể biến động theo tin nghị quyết."
    if risk == "DIVIDEND":
        return f"Sắp chốt quyền cổ tức trong {days} ngày ({label})."
    return f"Sự kiện {label} trong {days} ngày."


def _empty_result() -> dict:
    return {
        "has_catalyst": False,
        "event_risk": "NONE",
        "days_to_event": None,
        "blackout": False,
        "dilution_warning": False,
        "next_event": None,
        "upcoming": [],
        "summary": "Không có sự kiện đáng chú ý phía trước.",
    }


if __name__ == "__main__":
    import json
    out = analyze("HPG")
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
