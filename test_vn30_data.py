"""Test data layer cho 30 mã VN30."""
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

# Xóa cache chỉ khi chạy với flag --fresh
if "--fresh" in sys.argv:
    for p in Path("multiagents_trading_assistant/cache").glob("*_fundamentals.json"):
        p.unlink()
    for p in Path("multiagents_trading_assistant/cache").glob("*_foreign_flow.json"):
        p.unlink()
    for p in Path("multiagents_trading_assistant/cache").glob("*_ohlcv_*.json"):
        p.unlink()
    print(">>> Cache đã xóa (--fresh mode)\n")

from multiagents_trading_assistant.fetcher import (
    get_fundamentals,
    get_vn30_symbols,
    get_ohlcv_batch,
    get_foreign_flow,
)

symbols = get_vn30_symbols()
print(f"\n{'='*70}")
print(f"  DATA LAYER TEST — {len(symbols)} mã VN30")
print(f"{'='*70}\n")

# ── 1. Fundamentals ──────────────────────────────────────────────────────
print(">>> [1] FUNDAMENTALS\n")
fundamentals = {}
for sym in symbols:
    fundamentals[sym] = get_fundamentals(sym)

print(f"\n{'Mã':<6} {'PE':>6} {'PB':>6} {'ROE%':>6} {'EPS':>7} {'Rev%':>6} {'Prf%':>6}  Industry")
print("-" * 72)
for sym in symbols:
    r = fundamentals[sym]
    pe      = f"{r['pe']:.1f}"              if r["pe"]             else "N/A"
    pb      = f"{r['pb']:.2f}"              if r["pb"]             else "N/A"
    roe     = f"{r['roe']:.1f}"             if r["roe"]            else "N/A"
    eps     = f"{r['eps']:.0f}"             if r["eps"]            else "N/A"
    rev_gr  = f"{r['revenue_growth']:.1f}"  if r["revenue_growth"] else "N/A"
    prof_gr = f"{r['profit_growth']:.1f}"   if r["profit_growth"]  else "N/A"
    ind     = (r["industry"] or "N/A")[:18]
    print(f"{sym:<6} {pe:>6} {pb:>6} {roe:>6} {eps:>7} {rev_gr:>6} {prof_gr:>6}  {ind}")

# Summary
api_ok   = sum(1 for r in fundamentals.values() if r["pe"] is not None)
roe_ok   = sum(1 for r in fundamentals.values() if r["roe"] is not None)
eps_ok   = sum(1 for r in fundamentals.values() if r["eps"] is not None)
print(f"\n  PE/PB live:  {api_ok}/{len(symbols)}")
print(f"  ROE live:    {roe_ok}/{len(symbols)}")
print(f"  EPS any:     {eps_ok}/{len(symbols)}")

# ── 2. OHLCV batch ───────────────────────────────────────────────────────
print(f"\n\n>>> [2] OHLCV (60 nến cuối)\n")
ohlcv = get_ohlcv_batch(symbols, n_days=60)
ok = sum(1 for v in ohlcv.values() if not v.empty)
print(f"\n  Lấy được: {ok}/{len(symbols)} mã")
for sym in symbols[:5]:
    df = ohlcv.get(sym)
    if df is not None and not df.empty:
        last = df.iloc[-1]
        print(f"  {sym}: {len(df)} nến, close={last['close']:.1f}, vol={last['volume']:,.0f}")

# ── 3. Foreign flow ──────────────────────────────────────────────────────
print(f"\n\n>>> [3] FOREIGN FLOW (10 mã đầu)\n")
ff_ok = 0
for sym in symbols[:10]:
    ff = get_foreign_flow(sym)
    has_history = len(ff.get("flow_history", [])) > 1
    if has_history:
        ff_ok += 1
    print(f"  {sym}: room={ff['room_usage_pct']}%, net5d={ff['net_flow_5d']:,.0f}, "
          f"net20d={ff['net_flow_20d']:,.0f}, history_days={len(ff.get('flow_history', []))}")
print(f"\n  Có lịch sử thực: {ff_ok}/10 mã")

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}\n")
