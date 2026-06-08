"""
Backfill OHLCV cho TOÀN BỘ HOSE đang niêm yết (~400 mã) — khử membership
look-ahead (survivorship Lớp 1).

Vì sao file riêng: master parquet của project chỉ có 100 mã VN100 *hiện tại*
→ khi backtest 2020 ta vô tình chỉ chọn trong winner tương lai. Mở rộng ra
toàn bộ HOSE + để bộ lọc thanh khoản point-in-time tự định nghĩa universe mỗi
kỳ = khử phần bias lớn nhất, KHÔNG cần danh sách thành phần lịch sử.

Dùng vnstock FREE (VCI) — đã chứng minh chạy được trong python global này
(vnstock_data Golden nằm ở ~/.venv, không gọi từ đây). Ghi ra parquet RIÊNG,
KHÔNG đụng master.

Chạy:
    python -m research.build_hose_universe --start 2019-06-01 --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)
for _n in ("vnstock", "vnstock.core.utils.client", "urllib3"):
    logging.getLogger(_n).setLevel(logging.CRITICAL)

_ROOT = Path(__file__).resolve().parents[1]
_CACHE_SYMS = _ROOT / "multiagents_trading_assistant" / "cache" / "all_symbols_2026-04-29.json"
_OUT_DIR = _ROOT / "research" / "data"
_OUT = _OUT_DIR / "ohlcv_hose_full.parquet"


def load_symbols() -> list[str]:
    """Danh sách mã HOSE: thử Listing API, fallback cache json. Lọc 3-ký-tự."""
    syms: list[str] = []
    try:
        from vnstock import Listing
        df = Listing(source="VCI").symbols_by_exchange()
        if "exchange" in df.columns:
            df = df[df["exchange"].astype(str).str.upper().str.contains("HOSE|HSX", na=False)]
        syms = df["symbol"].astype(str).str.upper().tolist()
    except Exception:
        pass
    if not syms:                                   # fallback: cache
        data = json.load(open(_CACHE_SYMS, encoding="utf-8"))
        syms = [d["symbol"] for d in data] if isinstance(data, list) else list(data)
    syms = sorted({s.upper().strip() for s in syms if isinstance(s, str)
                   and len(s.strip()) == 3 and s.strip().isalpha()})
    return syms


def fetch_one(sym: str, start: str, end: str, retries: int = 3):
    """Kéo OHLCV 1 mã qua VCI free, retry + backoff. Trả (sym, df|None, err)."""
    from vnstock import Vnstock
    for attempt in range(retries):
        try:
            q = Vnstock().stock(symbol=sym, source="VCI").quote.history(
                start=start, end=end, interval="1D")
            if q is None or len(q) == 0:
                return sym, None, "empty"
            q = q.rename(columns={"time": "date"})
            q["date"] = pd.to_datetime(q["date"])
            need = ["date", "open", "high", "low", "close", "volume"]
            if not all(c in q.columns for c in need):
                return sym, None, f"missing cols {list(q.columns)}"
            q = q[need].copy()
            q["symbol"] = sym
            q["value"] = q["volume"] * q["close"]      # đơn vị giống master (giá_nghìn × KL)
            return sym, q, None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return sym, None, f"{type(e).__name__}: {str(e)[:80]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-06-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="giới hạn số mã (debug)")
    args = ap.parse_args()
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")

    syms = load_symbols()
    if args.limit:
        syms = syms[: args.limit]
    print(f"Universe HOSE: {len(syms)} mã | {args.start} → {end} | workers={args.workers}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames, errors, done = [], {}, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, s, args.start, end): s for s in syms}
        for fut in as_completed(futs):
            sym, df, err = fut.result()
            done += 1
            if err:
                errors[sym] = err
            else:
                frames.append(df)
            if done % 25 == 0 or done == len(syms):
                print(f"  {done}/{len(syms)} | ok={len(frames)} err={len(errors)} "
                      f"| {time.time()-t0:.0f}s", flush=True)

    if not frames:
        print("❌ Không kéo được dữ liệu nào.")
        if errors:
            print("Lỗi mẫu:", list(errors.items())[:5])
        return

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["date", "symbol"]).sort_values(["symbol", "date"])
    out = out[["date", "symbol", "open", "high", "low", "close", "volume", "value"]]
    out.to_parquet(_OUT, index=False)
    print(f"\n✓ Ghi {_OUT}")
    print(f"  {len(out):,} dòng | {out['symbol'].nunique()} mã | "
          f"{out['date'].min().date()} → {out['date'].max().date()}")
    if errors:
        print(f"  {len(errors)} mã lỗi (vd: {list(errors)[:8]})")


if __name__ == "__main__":
    main()
