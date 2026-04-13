"""
debug_screener.py — Kiểm tra tại sao screener không ra candidate.
Chạy: py -3.11 debug_screener.py
"""
import importlib.metadata
import sys
sys.path.insert(0, "multiagents_trading_assistant")

import pandas as pd
if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = pd.DataFrame.map

from fetcher import get_ohlcv, get_vnindex
from indicators import compute_indicators
from screener import (
    get_market_context,
    detect_breakout, detect_retest, detect_spring,
    detect_ma_pullback, detect_rsi_bounce,
)

TEST_SYMBOLS = ["VNM", "VCB", "HPG", "MBB", "FPT", "TCB", "ACB", "STB", "MWG", "SSI"]

# ── Bước 1: Market context ──
print("=== MARKET CONTEXT ===")
vnindex_df = get_vnindex(200)
if vnindex_df.empty:
    print("Không lấy được VN-Index!")
    sys.exit(1)

mkt = get_market_context(vnindex_df)
print(f"Trend: {mkt.trend} | Giá: {mkt.current_price:.0f}")
print(f"MA20={mkt.ma20:.0f} | MA60={mkt.ma60:.0f} | MA200={mkt.ma200:.0f}")
print(f"should_trade={mkt.should_trade}")
print()

# ── Bước 2: Kiểm tra từng mã ──
print("=== KIỂM TRA TỪNG MÃ ===")
for sym in TEST_SYMBOLS:
    df = get_ohlcv(sym, 200)
    if df is None or df.empty:
        print(f"{sym}: Không có data")
        continue

    ind = compute_indicators(df)
    if not ind:
        print(f"{sym}: compute_indicators trả về rỗng")
        continue

    price  = ind.get("current_price", 0)
    ma20   = ind.get("ma20", 0)
    ma60   = ind.get("ma60", 0)
    ma200  = ind.get("ma200", 0)
    rsi    = ind.get("rsi", 0)
    vol_s  = ind.get("volume_surge", False)

    # Kiểm tra từng điều kiện
    b_pass,  _ = detect_breakout(df, ind)
    r_pass,  _ = detect_retest(df, ind)
    sp_pass, _ = detect_spring(df, ind)
    m_pass,  _ = detect_ma_pullback(df, ind)
    rs_pass, _ = detect_rsi_bounce(df, ind)

    any_pass = any([b_pass, r_pass, sp_pass, m_pass, rs_pass])
    status = "✅ PASS" if any_pass else "❌ fail"

    print(f"{sym:6s} {status} | price={price:.0f} MA20={ma20:.0f} MA60={ma60:.0f} RSI={rsi:.1f} volSurge={vol_s}")
    if any_pass:
        signals = []
        if b_pass:  signals.append("BREAKOUT")
        if r_pass:  signals.append("RETEST")
        if sp_pass: signals.append("SPRING")
        if m_pass:  signals.append("MA_PULLBACK")
        if rs_pass: signals.append("RSI_BOUNCE")
        print(f"       → Signals: {signals}")
    else:
        # In lý do fail từng chiến lược
        v3 = [float(df["volume"].iloc[i]) for i in [-4, -3, -2]]
        vol_dec = (v3[0] > v3[1]) or (v3[1] > v3[2]) if len(df) >= 4 else False
        dist_ma60 = (price - ma60) / ma60 * 100 if ma60 else 0
        print(f"       MA_PULLBACK: uptrend={price > ma20 > ma60 > ma200} | "
              f"near_ma60={0 <= dist_ma60 <= 8.0:.0f}% ({dist_ma60:.1f}%) | "
              f"rsi_ok={30 <= rsi <= 60} ({rsi:.1f}) | vol_dec={vol_dec}")
        print(f"       RSI_BOUNCE:  rsi<40={rsi < 40} ({rsi:.1f}) | "
              f"above_ma200={price > ma200 if ma200 else '?'}")
