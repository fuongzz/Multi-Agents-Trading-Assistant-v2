"""
fetcher.py — Data layer: lấy toàn bộ dữ liệu cho pipeline.

Sources:
  - VCI (primary)   : vnstock.explorer.vci.Quote
  - KBS (fallback)  : vnstock.Quote(source="KBS")
  - FiinQuantX      : fundamentals (P/E, P/B, ROE, EPS, growth, industry)
  - yfinance        : global macro (S&P500, DXY, Oil, Gold, USD/VND)

Functions:
  OHLCV:       get_ohlcv(), get_ohlcv_batch(), get_vnindex()
  Fundamentals: get_fundamentals(symbol)
  Foreign flow: get_foreign_flow(symbol)
  Price board:  get_price_board(symbols)
  Macro:        get_global_macro(), get_vn_macro()
  Symbols:      get_vn30_symbols(), get_vn100_symbols()
"""

import importlib.metadata  # FIX: pandas-ta-openbb Python 3.11
import json
import os
import sys
import time
from datetime import date as _date, datetime, timedelta
from pathlib import Path

import pandas as pd

# FIX: Windows terminal không encode được tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# FIX: pandas 3.x không có applymap
if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = pd.DataFrame.map

# ── vnstock imports ──
from vnstock.explorer.vci import Quote as _VCIQuote
from vnstock import Quote as _KBSQuote, Vnstock, Trading as _Trading

# ── yfinance (global macro) ──
import yfinance as yf


# ──────────────────────────────────────────────
# Helpers (đặt sớm vì được dùng bởi FiinQuantX helpers bên dưới)
# ──────────────────────────────────────────────

def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# ── FiinQuantX (fundamentals) — lazy init ──
_fiin_client = None

def _get_fiin_client():
    """Khởi tạo FiinQuantX session lần đầu tiên gọi."""
    global _fiin_client
    if _fiin_client is not None:
        return _fiin_client
    user = os.getenv("FIINQUANT_USERNAME")
    pwd  = os.getenv("FIINQUANT_PASSWORD")
    if not user or not pwd:
        raise ValueError("FIINQUANT_USERNAME/PASSWORD chưa được cấu hình trong .env")
    from FiinQuantX import FiinSession
    _fiin_client = FiinSession(username=user, password=pwd).login()
    return _fiin_client


def _fiin_extract_income(fs_list: list, year: int) -> dict:
    """Lấy income statement dict cho năm cụ thể từ kết quả get_financial_statement."""
    for r in fs_list:
        if r.get("year") == year:
            inc = r.get("financialStatement", {}).get("incomeStatement", [])
            return inc[0] if isinstance(inc, list) and inc else (inc or {})
    return {}


def _fiin_extract_equity(fs_list: list, year: int) -> float:
    """Lấy totalEquity từ balancesheet cho năm cụ thể."""
    for r in fs_list:
        if r.get("year") == year:
            bs = r.get("financialStatement", {}).get("balanceSheet", [])
            bs = bs[0] if isinstance(bs, list) and bs else (bs or {})
            return _safe_float(bs.get("resources", {}).get("equity", {}).get("totalEquity", 0)) or 0.0
    return 0.0


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

_BASE_DIR  = Path(__file__).parent
_CACHE_DIR = _BASE_DIR / "cache"
_CACHE_DIR.mkdir(exist_ok=True)

_TODAY = datetime.now().strftime("%Y-%m-%d")

_VCI_MAX_REQ_PER_MIN = 14
_VCI_SLEEP_SECONDS   = 62
_vci_count = 0
_vci_window_start = time.time()


# ──────────────────────────────────────────────
# Rate limiter
# ──────────────────────────────────────────────

def _throttle() -> None:
    """Sleep nếu đã gần đạt giới hạn VCI free tier."""
    global _vci_count, _vci_window_start
    elapsed = time.time() - _vci_window_start

    if elapsed >= 60:
        _vci_count = 0
        _vci_window_start = time.time()

    if _vci_count >= _VCI_MAX_REQ_PER_MIN:
        wait = _VCI_SLEEP_SECONDS - elapsed
        if wait > 0:
            print(f"[fetcher] VCI rate limit — chờ {wait:.0f}s...")
            time.sleep(wait)
        _vci_count = 0
        _vci_window_start = time.time()

    _vci_count += 1


# ──────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.json"


def _load_cache(key: str) -> list[dict] | None:
    p = _cache_path(key)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return None


def _save_cache(key: str, records: list[dict]) -> None:
    try:
        _cache_path(key).write_text(
            json.dumps(records, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[fetcher] Lỗi ghi cache {key}: {e}")


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


# ──────────────────────────────────────────────
# Internal fetch — VCI với KBS fallback
# ──────────────────────────────────────────────

def _date_range(n_days: int) -> tuple[str, str]:
    """Tính start/end date để lấy ít nhất n_days nến (bù ngày nghỉ × 1.5)."""
    end   = _date.today()
    start = end - timedelta(days=int(n_days * 1.5) + 10)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _normalize_df(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Chuẩn hóa tên cột, lấy n_days nến cuối."""
    df = df.rename(columns={"time": "date", "ticker": "symbol"})
    needed = ["date", "open", "high", "low", "close", "volume"]
    cols   = [c for c in needed if c in df.columns]
    df     = df[cols].copy()
    df     = df.tail(n_days).reset_index(drop=True)
    return df


def _fetch_single(symbol: str, n_days: int) -> pd.DataFrame:
    """
    Lấy OHLCV 1 mã. VCI trước, KBS nếu lỗi.
    Trả về DataFrame trống nếu cả 2 đều thất bại.
    """
    start, end = _date_range(n_days)

    # ── VCI ──
    try:
        _throttle()
        df = _VCIQuote(symbol).history(start=start, end=end, interval="1D")
        if df is not None and not df.empty:
            df = _normalize_df(df, n_days)
            print(f"[fetcher] VCI ✓ {symbol}: {len(df)} nến")
            return df
    except Exception as e:
        print(f"[fetcher] VCI ✗ {symbol}: {e} — thử KBS...")

    # ── KBS fallback ──
    try:
        _throttle()
        df = _KBSQuote(symbol=symbol, source="KBS").history(
            start=start, end=end, interval="1D"
        )
        if df is not None and not df.empty:
            df = _normalize_df(df, n_days)
            print(f"[fetcher] KBS ✓ {symbol}: {len(df)} nến")
            return df
    except Exception as e:
        print(f"[fetcher] KBS ✗ {symbol}: {e}")

    print(f"[fetcher] Không lấy được data {symbol}")
    return pd.DataFrame()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def get_ohlcv(symbol: str, n_days: int = 200) -> pd.DataFrame:
    """
    Lấy OHLCV của 1 mã. Cache theo ngày.

    Args:
        symbol: Mã cổ phiếu, VD "VNM"
        n_days: Số nến cần lấy (mặc định 200)

    Returns:
        DataFrame với columns [date, open, high, low, close, volume]
        DataFrame trống nếu lỗi.
    """
    key    = f"{symbol}_{_TODAY}_ohlcv_{n_days}"
    cached = _load_cache(key)
    if cached is not None:
        print(f"[fetcher] Cache ✓ {symbol}")
        return _records_to_df(cached)

    df = _fetch_single(symbol, n_days)
    if not df.empty:
        _save_cache(key, _df_to_records(df))
    return df


def get_ohlcv_batch(
    symbols: list[str],
    n_days: int = 200,
) -> dict[str, pd.DataFrame]:
    """
    Lấy OHLCV nhiều mã. Mã đã cache trả về ngay, mã chưa có gọi API tuần tự.

    Args:
        symbols: Danh sách mã cổ phiếu
        n_days:  Số nến cần lấy

    Returns:
        dict[symbol → DataFrame]. DataFrame trống nếu lấy thất bại.
    """
    result:  dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for sym in symbols:
        key    = f"{sym}_{_TODAY}_ohlcv_{n_days}"
        cached = _load_cache(key)
        if cached is not None:
            result[sym] = _records_to_df(cached)
        else:
            missing.append(sym)

    if not missing:
        print(f"[fetcher] Tất cả {len(symbols)} mã đã có cache hôm nay")
        return result

    print(f"[fetcher] Cần lấy {len(missing)}/{len(symbols)} mã từ API...")

    for sym in missing:
        df = _fetch_single(sym, n_days)
        if not df.empty:
            key = f"{sym}_{_TODAY}_ohlcv_{n_days}"
            _save_cache(key, _df_to_records(df))
        result[sym] = df

    ok = sum(1 for v in result.values() if not v.empty)
    print(f"[fetcher] Batch xong: {ok}/{len(symbols)} mã có data")
    return result


def get_vnindex(n_days: int = 200) -> pd.DataFrame:
    """
    Lấy OHLCV VN-Index. Cache theo ngày.

    Returns:
        DataFrame với columns [date, open, high, low, close, volume]
    """
    return get_ohlcv("VNINDEX", n_days)


# ──────────────────────────────────────────────
# Symbol lists (static — cập nhật định kỳ)
# ──────────────────────────────────────────────

def get_vn30_symbols() -> list[str]:
    return [
        "ACB", "BCM", "BID", "BVH", "CTG",
        "FPT", "GAS", "GVR", "HDB", "HPG",
        "MBB", "MSN", "MWG", "PLX", "POW",
        "SAB", "SHB", "SSB", "SSI", "STB",
        "TCB", "TPB", "VCB", "VHM", "VIB",
        "VIC", "VJC", "VNM", "VPB", "VRE",
    ]


def get_vn100_symbols() -> list[str]:
    """VN100 từ Listing API. Fallback danh sách cứng 100 mã nếu API lỗi."""
    try:
        from vnstock import Listing
        df = Listing().symbols_by_group("VN100")
        symbols = df.tolist() if hasattr(df, "tolist") else df["symbol"].tolist()
        if len(symbols) >= 90:
            print(f"[fetcher] VN100 từ Listing API: {len(symbols)} mã")
            return symbols
    except Exception as e:
        print(f"[fetcher] VN100 API lỗi: {e} — dùng danh sách cứng")

    return [
        "ACB", "ANV", "BCM", "BID", "BMP", "BSI", "BSR", "BVH", "BWE", "CII",
        "CMG", "CTD", "CTG", "CTR", "CTS", "DBC", "DCM", "DGC", "DGW", "DIG",
        "DPM", "DSE", "DXG", "DXS", "EIB", "EVF", "FPT", "FRT", "FTS", "GAS",
        "GEE", "GEX", "GMD", "GVR", "HAG", "HCM", "HDB", "HDC", "HDG", "HHV",
        "HPG", "HSG", "HT1", "IMP", "KBC", "KDC", "KDH", "KOS", "LPB", "MBB",
        "MSB", "MSN", "MWG", "NAB", "NKG", "NLG", "NT2", "NVL", "OCB", "PAN",
        "PC1", "PDR", "PHR", "PLX", "PNJ", "POW", "PVD", "PVT", "REE", "SAB",
        "SBT", "SCS", "SHB", "SIP", "SJS", "SSB", "SSI", "STB", "SZC", "TCB",
        "TCH", "TPB", "VCB", "VCG", "VCI", "VGC", "VHC", "VHM", "VIB", "VIC",
        "VIX", "VJC", "VND", "VNM", "VPB", "VPI", "VPL", "VRE", "VSC", "VTP",
    ]


# ──────────────────────────────────────────────
# Price Board (real-time snapshot)
# ──────────────────────────────────────────────

def get_price_board(symbols: list[str]) -> pd.DataFrame:
    """
    Lấy bảng giá realtime + thông tin khối ngoại từ Trading.price_board (VCI).

    Returns:
        DataFrame với columns:
          symbol, price, listed_share,
          current_room, total_room,
          foreign_buy_vol, foreign_sell_vol,
          foreign_buy_val, foreign_sell_val
        DataFrame trống nếu lỗi.
    """
    key    = f"price_board_{_TODAY}_{'_'.join(sorted(symbols))[:50]}"
    cached = _load_cache(key)
    if cached is not None:
        return pd.DataFrame(cached)

    try:
        _throttle()
        raw = _Trading(source="VCI").price_board(symbols_list=symbols)
        raw.columns = ["_".join(str(c) for c in col).strip("_") for col in raw.columns]

        df = pd.DataFrame()
        df["symbol"]           = raw.get("listing_symbol",            pd.Series(dtype=str))
        df["price"]            = raw.get("match_match_price",         pd.Series(dtype=float))
        df["listed_share"]     = raw.get("listing_listed_share",      pd.Series(dtype=float))
        df["current_room"]     = raw.get("match_current_room",        pd.Series(dtype=float))
        df["total_room"]       = raw.get("match_total_room",          pd.Series(dtype=float))
        df["foreign_buy_vol"]  = raw.get("match_foreign_buy_volume",  pd.Series(dtype=float))
        df["foreign_sell_vol"] = raw.get("match_foreign_sell_volume", pd.Series(dtype=float))
        df["foreign_buy_val"]  = raw.get("match_foreign_buy_value",   pd.Series(dtype=float))
        df["foreign_sell_val"] = raw.get("match_foreign_sell_value",  pd.Series(dtype=float))

        _save_cache(key, _df_to_records(df))
        print(f"[fetcher] price_board ✓ {len(df)} mã")
        return df
    except Exception as e:
        print(f"[fetcher] price_board ✗: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# Fundamentals (P/E, P/B, ROE, EPS)
# ──────────────────────────────────────────────

# Bảng tĩnh VN30 — fallback khi API lỗi (BCTC 2024, đơn vị VNĐ)
_VN30_STATIC: dict[str, dict] = {
    "VCB":  {"eps": 5800,  "bvps": 38000, "roe": 17.2, "revenue_growth": 12.0, "profit_growth": 10.0, "industry": "Ngân hàng"},
    "BID":  {"eps": 3200,  "bvps": 22000, "roe": 15.1, "revenue_growth":  8.0, "profit_growth":  9.0, "industry": "Ngân hàng"},
    "CTG":  {"eps": 3500,  "bvps": 24000, "roe": 15.8, "revenue_growth": 10.0, "profit_growth": 11.0, "industry": "Ngân hàng"},
    "MBB":  {"eps": 3800,  "bvps": 25000, "roe": 18.5, "revenue_growth": 14.0, "profit_growth": 12.0, "industry": "Ngân hàng"},
    "TCB":  {"eps": 5200,  "bvps": 33000, "roe": 17.0, "revenue_growth": 11.0, "profit_growth": 10.0, "industry": "Ngân hàng"},
    "ACB":  {"eps": 4100,  "bvps": 25000, "roe": 20.5, "revenue_growth": 15.0, "profit_growth": 13.0, "industry": "Ngân hàng"},
    "HDB":  {"eps": 3200,  "bvps": 20000, "roe": 18.0, "revenue_growth": 16.0, "profit_growth": 14.0, "industry": "Ngân hàng"},
    "VPB":  {"eps": 2800,  "bvps": 22000, "roe": 13.5, "revenue_growth":  9.0, "profit_growth":  8.0, "industry": "Ngân hàng"},
    "STB":  {"eps": 2200,  "bvps": 16000, "roe": 14.2, "revenue_growth": 10.0, "profit_growth": 11.0, "industry": "Ngân hàng"},
    "SSB":  {"eps": 2100,  "bvps": 15000, "roe": 14.8, "revenue_growth": 12.0, "profit_growth": 10.0, "industry": "Ngân hàng"},
    "TPB":  {"eps": 2500,  "bvps": 17000, "roe": 15.5, "revenue_growth": 11.0, "profit_growth":  9.0, "industry": "Ngân hàng"},
    "SHB":  {"eps": 1800,  "bvps": 14000, "roe": 13.0, "revenue_growth":  9.0, "profit_growth":  8.0, "industry": "Ngân hàng"},
    "VIB":  {"eps": 3600,  "bvps": 22000, "roe": 19.0, "revenue_growth": 13.0, "profit_growth": 12.0, "industry": "Ngân hàng"},
    "HPG":  {"eps": 2200,  "bvps": 22000, "roe": 10.5, "revenue_growth": 18.0, "profit_growth": 25.0, "industry": "Thép"},
    "GAS":  {"eps": 9500,  "bvps": 50000, "roe": 20.0, "revenue_growth":  5.0, "profit_growth":  4.0, "industry": "Dầu khí"},
    "PLX":  {"eps": 2800,  "bvps": 24000, "roe": 12.0, "revenue_growth":  7.0, "profit_growth":  6.0, "industry": "Dầu khí"},
    "POW":  {"eps": 1200,  "bvps": 12000, "roe":  9.5, "revenue_growth":  4.0, "profit_growth":  5.0, "industry": "Điện"},
    "FPT":  {"eps": 5500,  "bvps": 28000, "roe": 24.0, "revenue_growth": 22.0, "profit_growth": 20.0, "industry": "Công nghệ"},
    "MWG":  {"eps": 3200,  "bvps": 20000, "roe": 16.0, "revenue_growth": 15.0, "profit_growth": 30.0, "industry": "Bán lẻ"},
    "MSN":  {"eps": 2100,  "bvps": 28000, "roe":  7.5, "revenue_growth": 10.0, "profit_growth": 15.0, "industry": "Thực phẩm"},
    "VNM":  {"eps": 4500,  "bvps": 32000, "roe": 14.0, "revenue_growth":  3.0, "profit_growth":  2.0, "industry": "Thực phẩm"},
    "SAB":  {"eps": 13000, "bvps": 55000, "roe": 25.0, "revenue_growth":  8.0, "profit_growth":  7.0, "industry": "Thực phẩm"},
    "VIC":  {"eps": 2500,  "bvps": 55000, "roe":  4.5, "revenue_growth": 12.0, "profit_growth": 20.0, "industry": "Bất động sản"},
    "VHM":  {"eps": 8500,  "bvps": 45000, "roe": 19.0, "revenue_growth": 15.0, "profit_growth": 18.0, "industry": "Bất động sản"},
    "VRE":  {"eps": 1500,  "bvps": 14000, "roe": 10.5, "revenue_growth":  8.0, "profit_growth": 10.0, "industry": "Bất động sản"},
    "BCM":  {"eps": 2000,  "bvps": 20000, "roe": 10.0, "revenue_growth": 10.0, "profit_growth":  8.0, "industry": "Bất động sản"},
    "GVR":  {"eps": 1200,  "bvps": 14000, "roe":  8.5, "revenue_growth":  5.0, "profit_growth":  4.0, "industry": "Nông nghiệp"},
    "VJC":  {"eps": 4800,  "bvps": 22000, "roe": 22.0, "revenue_growth": 25.0, "profit_growth": 40.0, "industry": "Vận tải"},
    "SSI":  {"eps": 3200,  "bvps": 28000, "roe": 12.0, "revenue_growth": 15.0, "profit_growth": 18.0, "industry": "Chứng khoán"},
    "BVH":  {"eps": 2800,  "bvps": 30000, "roe": 10.0, "revenue_growth":  8.0, "profit_growth":  7.0, "industry": "Bảo hiểm"},
}


def get_fundamentals(symbol: str) -> dict:
    """
    Lấy P/E, P/B, ROE, EPS, revenue_growth, profit_growth, industry.

    Thứ tự ưu tiên:
      1. Cache hôm nay
      2. FiinQuantX: MarketDepth (P/E, P/B live) + financial statements (ROE, EPS, growth) + BasicInfor (industry)
      3. Fallback bảng tĩnh VN30 cho EPS/industry khi FiinQuantX thiếu data

    Returns:
        dict với keys: pe, pb, roe, eps, revenue_growth, profit_growth, industry
        Giá trị None nếu không có data.
    """
    key    = f"{symbol}_{_TODAY}_fundamentals"
    cached = _load_cache(key)
    if cached is not None:
        return cached[0] if isinstance(cached, list) else cached

    result: dict = {
        "pe": None, "pb": None, "roe": None, "eps": None,
        "revenue_growth": None, "profit_growth": None, "industry": None,
    }

    # ── FiinQuantX ──
    try:
        client = _get_fiin_client()

        # P/E, P/B — live từ thị trường
        week_ago = (_date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
        val_df = client.MarketDepth().get_stock_valuation(
            tickers=[symbol], from_date=week_ago
        )
        if not val_df.empty:
            latest = val_df.sort_values("timestamp").iloc[-1]
            result["pe"] = _safe_float(latest.get("pe"))
            result["pb"] = _safe_float(latest.get("pb"))

        # Income statement 2 năm gần nhất
        curr_year = datetime.now().year - 1   # 2025
        prev_year = datetime.now().year - 2   # 2024
        fa = client.FundamentalAnalysis()
        inc_data = fa.get_financial_statement(
            tickers=[symbol], statement="incomestatement",
            years=[prev_year, curr_year], type="consolidated",
        )
        inc_curr = _fiin_extract_income(inc_data, curr_year)
        inc_prev = _fiin_extract_income(inc_data, prev_year)

        # Nếu năm hiện tại chưa có → thử lùi thêm 1 năm
        if not inc_curr:
            curr_year -= 1
            prev_year -= 1
            inc_data  = fa.get_financial_statement(
                tickers=[symbol], statement="incomestatement",
                years=[prev_year, curr_year], type="consolidated",
            )
            inc_curr = _fiin_extract_income(inc_data, curr_year)
            inc_prev = _fiin_extract_income(inc_data, prev_year)

        if inc_curr:
            profit_curr = _safe_float(inc_curr.get("netProfitAfterTax")) or 0.0
            rev_curr    = _safe_float(inc_curr.get("revenue", {}).get("netRevenue")) or 0.0
            eps_raw     = _safe_float(inc_curr.get("earningsPerShare", {}).get("epsBasic")) or 0.0
            result["eps"] = eps_raw if eps_raw else None

            if inc_prev:
                profit_prev = _safe_float(inc_prev.get("netProfitAfterTax")) or 0.0
                rev_prev    = _safe_float(inc_prev.get("revenue", {}).get("netRevenue")) or 0.0
                if profit_prev:
                    result["profit_growth"] = round(
                        (profit_curr - profit_prev) / abs(profit_prev) * 100, 1
                    )
                if rev_prev > 0:
                    result["revenue_growth"] = round(
                        (rev_curr - rev_prev) / rev_prev * 100, 1
                    )

            # ROE = netProfit / totalEquity
            bs_data = fa.get_financial_statement(
                tickers=[symbol], statement="balancesheet",
                years=[curr_year], type="consolidated",
            )
            equity = _fiin_extract_equity(bs_data, curr_year)
            if equity > 0 and profit_curr:
                result["roe"] = round(profit_curr / equity * 100, 1)

        # Industry từ BasicInfor
        bi_df = client.BasicInfor(tickers=[symbol]).get()
        if not bi_df.empty:
            row = bi_df[bi_df["ticker"] == symbol]
            if not row.empty:
                industry_val = str(row.iloc[0].get("icbNameL2") or "")
                if industry_val:
                    result["industry"] = industry_val

        print(
            f"[fetcher] fundamentals ✓ FiinQuantX {symbol}: "
            f"PE={result['pe']}, ROE={result['roe']}, EPS={result['eps']}"
        )

    except Exception as e:
        print(f"[fetcher] fundamentals FiinQuantX ✗ {symbol}: {e}")

    # ── Fallback bảng tĩnh cho trường còn thiếu ──
    static = _VN30_STATIC.get(symbol)
    if static:
        if result["eps"] is None:
            result["eps"] = float(static["eps"])
        if result["roe"] is None:
            result["roe"] = static["roe"]
        if result["revenue_growth"] is None:
            result["revenue_growth"] = static["revenue_growth"]
        if result["profit_growth"] is None:
            result["profit_growth"] = static["profit_growth"]
        if result["industry"] is None:
            result["industry"] = static["industry"]
        # P/E, P/B từ static chỉ dùng khi FiinQuantX hoàn toàn thất bại
        if result["pe"] is None and static.get("eps"):
            pb_df = get_price_board([symbol])
            if not pb_df.empty:
                row = pb_df[pb_df["symbol"] == symbol]
                if not row.empty:
                    price = _safe_float(row.iloc[0].get("price"))
                    if price and price > 0:
                        result["pe"] = round(price / static["eps"],  2)
                        result["pb"] = round(price / static["bvps"], 2)

    _save_cache(key, [result])
    return result


# ──────────────────────────────────────────────
# Foreign Flow (khối ngoại)
# ──────────────────────────────────────────────

def get_foreign_flow(symbol: str, n_days: int = 20) -> dict:
    """
    Lấy dữ liệu khối ngoại: room usage, net flow 5d/20d.

    Thứ tự ưu tiên:
      1. Cache hôm nay
      2. FiinQuantX Fetch_Trading_Data(fields=["fb","fs","fn"]) — lịch sử thực sự
      3. Fallback price_board snapshot (room usage + net flow ngày hiện tại)

    Returns:
        dict với keys: room_usage_pct, net_flow_5d, net_flow_20d, flow_history
    """
    key    = f"{symbol}_{_TODAY}_foreign_flow"
    cached = _load_cache(key)
    if cached is not None:
        return cached[0] if isinstance(cached, list) else cached

    result = {
        "room_usage_pct": None,
        "net_flow_5d":    None,
        "net_flow_20d":   None,
        "flow_history":   [],
    }

    # ── FiinQuantX: lịch sử fb/fs/fn ──────────────────────────────────────
    try:
        client  = _get_fiin_client()
        period  = max(n_days + 5, 25)  # lấy thêm buffer cho ngày lễ
        ff_df = client.Fetch_Trading_Data(
            realtime = False,
            tickers  = [symbol],
            fields   = ["fb", "fs", "fn"],
            adjusted = True,
            by       = "1d",
            period   = period,
        ).get_data()

        if ff_df is not None and not ff_df.empty:
            # Lọc đúng ticker (DataFrame có thể chứa nhiều mã)
            if "ticker" in ff_df.columns:
                ff_df = ff_df[ff_df["ticker"] == symbol].copy()

            if "fn" in ff_df.columns and not ff_df.empty:
                ff_df = ff_df.sort_values("time") if "time" in ff_df.columns else ff_df
                net_series = ff_df["fn"].dropna()
                result["net_flow_5d"]  = float(net_series.tail(5).sum())
                result["net_flow_20d"] = float(net_series.tail(n_days).sum())

                time_col = "time" if "time" in ff_df.columns else ff_df.columns[0]
                result["flow_history"] = [
                    {"date": str(row[time_col]), "net_flow": float(row["fn"])}
                    for _, row in ff_df[[time_col, "fn"]].tail(n_days).iterrows()
                ]
                print(f"[fetcher] foreign_flow FiinQuantX ✓ {symbol}: "
                      f"net5d={result['net_flow_5d']:,.0f}, net20d={result['net_flow_20d']:,.0f}")
    except Exception as e:
        print(f"[fetcher] foreign_flow FiinQuantX ✗ {symbol}: {e}")

    # ── Fallback price_board: room usage (+ net flow nếu FiinQuantX thất bại) ──
    try:
        pb = get_price_board([symbol])
        if not pb.empty:
            row = pb[pb["symbol"] == symbol]
            if not row.empty:
                r = row.iloc[0]
                total   = _safe_float(r.get("total_room"))
                current = _safe_float(r.get("current_room"))
                if total and total > 0:
                    used = total - (current or 0)
                    result["room_usage_pct"] = round(used / total * 100, 1)

                if result["net_flow_5d"] is None:
                    buy  = _safe_float(r.get("foreign_buy_vol"))  or 0.0
                    sell = _safe_float(r.get("foreign_sell_vol")) or 0.0
                    net  = buy - sell
                    result["net_flow_5d"]  = net
                    result["net_flow_20d"] = net
                    result["flow_history"] = [{"date": _TODAY, "net_flow": net}]
                print(f"[fetcher] foreign_flow price_board ✓ {symbol}: room={result['room_usage_pct']}%")
    except Exception as e:
        print(f"[fetcher] foreign_flow price_board ✗ {symbol}: {e}")

    _save_cache(key, [result])
    return result


# ──────────────────────────────────────────────
# Global Macro (yfinance)
# ──────────────────────────────────────────────

_MACRO_TICKERS = {
    "sp500":  "^GSPC",
    "dxy":    "DX-Y.NYB",
    "oil":    "CL=F",
    "gold":   "GC=F",
    "nikkei": "^N225",
    "kospi":  "^KS11",
    "hsi":    "^HSI",
}


def get_global_macro() -> dict:
    """
    Lấy giá thị trường toàn cầu từ yfinance. Cache theo ngày.

    Returns:
        dict[name → {current, change_pct}]
        Ví dụ: {"sp500": {"current": 5123.0, "change_pct": 0.5}, ...}
    """
    key    = f"global_macro_{_TODAY}"
    cached = _load_cache(key)
    if cached is not None:
        return cached[0] if isinstance(cached, list) else cached

    result = {}
    for name, ticker in _MACRO_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) >= 2:
                cur  = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                result[name] = {
                    "current":    round(cur, 2),
                    "change_pct": round((cur - prev) / prev * 100, 2),
                }
            else:
                result[name] = {"current": None, "change_pct": None}
        except Exception:
            result[name] = {"current": None, "change_pct": None}

    _save_cache(key, [result])
    print(f"[fetcher] global_macro ✓: {list(result.keys())}")
    return result


def get_vn_macro() -> dict:
    """
    Lấy USD/VND và lãi suất SBV. Cache theo ngày.

    Returns:
        dict với keys: usd_vnd, sbv_rate
    """
    key    = f"vn_macro_{_TODAY}"
    cached = _load_cache(key)
    if cached is not None:
        return cached[0] if isinstance(cached, list) else cached

    result = {"usd_vnd": None, "sbv_rate": 4.5}
    try:
        hist = yf.Ticker("VND=X").history(period="5d")
        if not hist.empty:
            result["usd_vnd"] = round(float(hist["Close"].iloc[-1]), 0)
        _save_cache(key, [result])
        print(f"[fetcher] vn_macro ✓: USD/VND={result['usd_vnd']}")
    except Exception as e:
        print(f"[fetcher] vn_macro ✗: {e}")

    return result


