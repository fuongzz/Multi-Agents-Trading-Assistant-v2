"""insider_agent.py — Giao dịch nội bộ / cổ đông lớn (Trade pipeline).

Thuần RULE-BASED (không LLM) — trả score 0-100 theo style flow_agent để dễ
đưa vào synthesis/risk sau này. Đọc category == MAJOR_SHAREHOLDER_TRADING từ
fetcher.get_corporate_events() (nguồn vnstock_data VCI).

Tín hiệu mạnh ở TTCK VN: lãnh đạo/cổ đông lớn đăng ký Mua/Bán. Title chứa khối
lượng, ví dụ: "Trần Vũ Minh - Đăng kí Mua 50,000,000 HPG" → parse volume.

Anti-leak: chỉ tính giao dịch đã công bố (public_date ≤ as_of_date) khi backtest.
"""

from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

from multiagents_trading_assistant import fetcher


_WINDOW_30 = 30
_WINDOW_90 = 90
# Parse "... Đăng kí Mua 50,000,000 HPG" → 50000000
_VOL_RE = re.compile(r"(?:Mua|Bán|mua|bán)\s+([\d.,]+)")


def analyze(
    symbol: str,
    date: str | None = None,
    *,
    as_of_date: str | None = None,
    backtest_mode: bool = False,
) -> dict:
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    ref = pd.Timestamp(date).normalize()
    aod = pd.Timestamp(as_of_date).normalize() if as_of_date else (
        ref if backtest_mode else None
    )
    print(f"[insider_agent] {symbol} ({date}){' [BT]' if backtest_mode else ''}")

    try:
        events = fetcher.get_corporate_events(symbol)
    except Exception as e:
        print(f"[insider_agent] events fail: {e}")
        return _empty_result()

    if events is None or events.empty or "category" not in events.columns:
        return _empty_result()

    ins = events[events["category"] == "MAJOR_SHAREHOLDER_TRADING"].copy()
    if ins.empty:
        return _empty_result()

    # Anti-leak + window: chỉ giữ giao dịch công bố trong 90 ngày tính tới ref
    pub = ins["public_date"]
    upper = aod if aod is not None else ref
    ins = ins[pub.notna() & (pub <= upper) & ((upper - pub).dt.days <= _WINDOW_90)]
    if ins.empty:
        return _empty_result()

    buy_90 = sell_90 = buy_30 = sell_30 = 0.0
    buy_cnt = sell_cnt = 0
    for _, r in ins.iterrows():
        vol = _parse_vol(r.get("title"))
        action = str(r.get("action_type_vi") or "")
        days = (upper - pd.Timestamp(r["public_date"]).normalize()).days
        is_buy = action == "Mua"
        is_sell = action == "Bán"
        if is_buy:
            buy_90 += vol; buy_cnt += 1
            if days <= _WINDOW_30: buy_30 += vol
        elif is_sell:
            sell_90 += vol; sell_cnt += 1
            if days <= _WINDOW_30: sell_30 += vol

    total = buy_90 + sell_90
    if total <= 0:
        # Có giao dịch nhưng không parse được volume → dùng count
        total_cnt = buy_cnt + sell_cnt
        ratio = (buy_cnt - sell_cnt) / total_cnt if total_cnt else 0.0
    else:
        ratio = (buy_90 - sell_90) / total

    score = int(round(50 + ratio * 50))
    score = max(0, min(100, score))

    if score >= 60:
        signal = "ACCUMULATION"
    elif score <= 40:
        signal = "DISTRIBUTION"
    else:
        signal = "NEUTRAL"

    # Cảnh báo bán phân phối: cổ đông lớn đăng ký bán ròng đáng kể trong 30 ngày
    distribution_warning = (sell_30 - buy_30) > 0 and sell_30 > 0 and (
        sell_30 >= 2 * max(buy_30, 1)
    )

    result = {
        "insider_buy_vol_90d": buy_90,
        "insider_sell_vol_90d": sell_90,
        "insider_net_30d": buy_30 - sell_30,
        "buy_count": buy_cnt,
        "sell_count": sell_cnt,
        "signal": signal,
        "score": score,
        "distribution_warning": bool(distribution_warning),
        "summary": _summary(signal, buy_cnt, sell_cnt, buy_90, sell_90, distribution_warning),
    }
    print(f"[insider_agent] {symbol} → {signal} score={score} "
          f"(buy={buy_cnt}/{buy_90:,.0f}, sell={sell_cnt}/{sell_90:,.0f})")
    return result


def _parse_vol(title) -> float:
    if not title:
        return 0.0
    m = _VOL_RE.search(str(title))
    if not m:
        return 0.0
    try:
        return float(m.group(1).replace(",", "").replace(".", ""))
    except ValueError:
        return 0.0


def _summary(signal, buy_cnt, sell_cnt, buy_90, sell_90, dist_warn) -> str:
    if signal == "NEUTRAL" and buy_cnt == 0 and sell_cnt == 0:
        return "Không có giao dịch nội bộ/cổ đông lớn 90 ngày qua."
    base = (
        f"90 ngày: {buy_cnt} đăng ký mua ({buy_90:,.0f}) / "
        f"{sell_cnt} đăng ký bán ({sell_90:,.0f})."
    )
    if dist_warn:
        return base + " ⚠️ Cổ đông lớn đăng ký BÁN ròng gần đây."
    if signal == "ACCUMULATION":
        return base + " Nội bộ đang gom."
    if signal == "DISTRIBUTION":
        return base + " Nội bộ đang phân phối."
    return base


def _empty_result() -> dict:
    return {
        "insider_buy_vol_90d": 0.0,
        "insider_sell_vol_90d": 0.0,
        "insider_net_30d": 0.0,
        "buy_count": 0,
        "sell_count": 0,
        "signal": "NEUTRAL",
        "score": 50,
        "distribution_warning": False,
        "summary": "Không có dữ liệu giao dịch nội bộ.",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(analyze("HPG"), ensure_ascii=False, indent=2, default=str))
    print(json.dumps(analyze("VCB"), ensure_ascii=False, indent=2, default=str))
