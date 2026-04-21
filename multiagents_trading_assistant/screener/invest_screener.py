"""invest_screener.py — FA screener cho Investment pipeline.

Tiêu chí lọc:
  ROE TTM ≥ 15%
  EPS / profit_growth YoY > 0 (tăng trưởng dương)
  P/E < 1.3 × median ngành
  Volume 20MA ≥ 300k

Output: top 7 InvestCandidate sort theo valuation_score.
"""

import importlib.metadata  # noqa: F401
from dataclasses import dataclass

from multiagents_trading_assistant.services.data_service import (
    get_ohlcv_batch, get_vn30_symbols,
)
from multiagents_trading_assistant import fetcher
from multiagents_trading_assistant.agents.invest.fundamental_agent import (
    _INDUSTRY_PE_MEDIAN, get_industry_pe,
)


@dataclass
class InvestCandidate:
    symbol: str
    valuation_score: float   # 0-100, cao = rẻ + tốt
    pe_ratio: float | None
    roe: float | None
    growth_streak: int       # số quý tăng trưởng dương liên tiếp (estimate)
    industry: str
    reasons: list[str]


def run_screener(
    symbols: list[str] | None = None,
    max_candidates: int = 7,
) -> list[InvestCandidate]:
    if symbols is None:
        symbols = get_vn30_symbols()

    print(f"[invest_screener] Scan {len(symbols)} mã (FA criteria)...")

    ohlcv_map = get_ohlcv_batch(symbols, n_days=30)
    candidates: list[InvestCandidate] = []

    for symbol in symbols:
        df = ohlcv_map.get(symbol)
        # Liquidity check
        if df is None or df.empty:
            continue
        vol_ma20 = float(df["volume"].tail(20).mean()) if len(df) >= 20 else 0
        if vol_ma20 < 300_000:
            continue

        # Fundamentals
        try:
            fund = fetcher.get_fundamentals(symbol)
        except Exception as e:
            print(f"[invest_screener] fundamentals fail {symbol}: {e}")
            continue

        if not fund:
            continue

        roe = fund.get("roe")
        pe = fund.get("pe")
        profit_growth = fund.get("profit_growth")
        industry = fund.get("industry") or "default"
        pe_median = get_industry_pe(industry)

        reasons: list[str] = []
        score = 0.0

        # Hard filter 1: ROE ≥ 15%
        if roe is None or roe < 15:
            continue
        reasons.append(f"ROE {roe:.1f}%")
        score += min(roe, 30) / 30 * 30  # max 30 điểm

        # Hard filter 2: tăng trưởng dương
        if profit_growth is None or profit_growth <= 0:
            continue
        reasons.append(f"LN tăng {profit_growth:.1f}%")
        score += min(profit_growth, 30) / 30 * 30  # max 30 điểm

        # Hard filter 3: P/E < 1.3 × median ngành
        if pe is not None and pe > 1.3 * pe_median:
            continue
        if pe is not None:
            ratio = pe / pe_median
            reasons.append(f"P/E {pe:.1f}× (median {pe_median})")
            score += max(0, (1.3 - ratio) / 1.3) * 40  # max 40 điểm, rẻ hơn → điểm cao hơn
        else:
            score += 20  # không có PE → trung bình

        # growth_streak estimate: nếu profit_growth > 0, ít nhất 1
        streak = 1 if profit_growth and profit_growth > 0 else 0
        if profit_growth and profit_growth > 15:
            streak = 2
        if profit_growth and profit_growth > 30:
            streak = 3

        candidates.append(InvestCandidate(
            symbol=symbol,
            valuation_score=round(score, 1),
            pe_ratio=pe,
            roe=roe,
            growth_streak=streak,
            industry=industry,
            reasons=reasons,
        ))

    candidates.sort(key=lambda c: c.valuation_score, reverse=True)
    result = candidates[:max_candidates]

    print(f"[invest_screener] {len(result)} candidates pass FA filter:")
    for c in result:
        print(f"  {c.symbol:6s} | score={c.valuation_score:.1f} | {', '.join(c.reasons)}")

    return result
