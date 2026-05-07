"""
fetcher.py — Data layer: lấy toàn bộ dữ liệu cho pipeline.

Sources:
  - vnstock_data Golden (primary): Market, Reference, Fundamental
  - DNSE (fallback/broker)       : REST GET /price/ohlc (LightSpeed API)
  - FiinQuantX      : fundamentals (P/E, P/B, ROE, EPS, growth, industry)
  - FiinQuantX      : foreign flow net5d/net20d
  - yfinance        : global macro (S&P500, DXY, Oil, Gold, USD/VND)

Functions:
  OHLCV:       get_ohlcv(), get_ohlcv_batch(), get_vnindex()
  Fundamentals: get_fundamentals(symbol)
  Foreign flow: get_foreign_flow(symbol)
  Price board:  get_price_board(symbols)
  Macro:        get_global_macro(), get_vn_macro()
  Symbols:      get_vn30_symbols(), get_vn100_symbols(), get_hnx30_symbols(), get_upcom30_symbols()
"""

import importlib.metadata  # FIX: pandas-ta-openbb Python 3.11
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as _date, datetime, timedelta, timezone
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

# ── yfinance (global macro) ──
import yfinance as yf

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# ── DNSE LightSpeed API ──
from multiagents_trading_assistant.services.dnse_client import _get_dnse_client
from multiagents_trading_assistant.data import get_data_provider


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


def _load_cache_with_ttl(key: str, ttl_seconds: int | None = None) -> list[dict] | None:
    """Load cache with optional TTL. None means cache never expires."""
    p = _cache_path(key)
    if ttl_seconds is not None and p.exists():
        age = time.time() - p.stat().st_mtime
        if age > ttl_seconds:
            return None
    return _load_cache(key)


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


def _to_unix_ts(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' sang Unix timestamp (UTC midnight)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _normalize_df(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Chuẩn hóa tên cột, lấy n_days nến cuối."""
    df = df.rename(columns={"time": "date", "ticker": "symbol"})
    needed = ["date", "open", "high", "low", "close", "volume"]
    cols   = [c for c in needed if c in df.columns]
    df     = df[cols].copy()
    df     = df.tail(n_days).reset_index(drop=True)
    return df


def _normalize_history_df(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Chuẩn hóa OHLCV theo khoảng ngày tuyệt đối."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"time": "date", "ticker": "symbol"})
    needed = ["date", "open", "high", "low", "close", "volume"]
    cols = [c for c in needed if c in df.columns]
    df = df[cols].copy()
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = (
        pd.to_datetime(df["date"], utc=True, errors="coerce")
        .dt.tz_convert("Asia/Ho_Chi_Minh")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    return df.sort_values("date").reset_index(drop=True)


def _fetch_single(symbol: str, n_days: int) -> pd.DataFrame:
    """
    Lấy OHLCV 1 mã. vnstock_data Golden trước, DNSE fallback.
    Trả về DataFrame trống nếu các nguồn đều thất bại.
    """
    start, end = _date_range(n_days)

    # ── vnstock_data Golden (primary) ──
    try:
        df = get_data_provider().get_ohlcv(symbol, start, end, interval="1D")
        if df is not None and not df.empty:
            df = _normalize_df(df, n_days)
            print(f"[fetcher] vnstock_data ✓ {symbol}: {len(df)} nến")
            return df
        print(f"[fetcher] vnstock_data trống {symbol} — thử DNSE...")
    except Exception as e:
        print(f"[fetcher] vnstock_data ✗ {symbol}: {e} — thử DNSE...")

    # ── DNSE fallback ──
    try:
        client = _get_dnse_client()
        if client is not None:
            from_ts = _to_unix_ts(start)
            to_ts   = _to_unix_ts(end) + 86400  # +1 ngày để include ngày cuối
            df = client.get_ohlcv(symbol, from_ts, to_ts, resolution="1D")
            if df is not None and not df.empty:
                df = _normalize_df(df, n_days)
                print(f"[fetcher] DNSE ✓ {symbol}: {len(df)} nến")
                return df
            print(f"[fetcher] DNSE trống {symbol}")
    except Exception as e:
        print(f"[fetcher] DNSE ✗ {symbol}: {e}")

    print(f"[fetcher] Không lấy được data {symbol}")
    return pd.DataFrame()


def _fetch_history(symbol: str, start: str, end: str, resolution: str = "1D") -> pd.DataFrame:
    """
    Lấy OHLCV theo khoảng ngày tuyệt đối. vnstock_data Golden trước, DNSE fallback.
    """
    # ── vnstock_data Golden (primary) ──
    try:
        df = get_data_provider().get_ohlcv(symbol, start, end, interval=resolution)
        df = _normalize_history_df(df, start, end)
        if df is not None and not df.empty:
            print(f"[fetcher] vnstock_data hist ✓ {symbol}: {len(df)} nến")
            return df
        print(f"[fetcher] vnstock_data hist trống {symbol} — thử DNSE...")
    except Exception as e:
        print(f"[fetcher] vnstock_data hist ✗ {symbol}: {e} — thử DNSE...")

    # ── DNSE fallback ──
    try:
        client = _get_dnse_client()
        if client is not None:
            from_ts = _to_unix_ts(start)
            to_ts = _to_unix_ts(end) + 86400  # include end date
            df = client.get_ohlcv(
                symbol,
                from_ts,
                to_ts,
                resolution=resolution,
                max_pages=100,
            )
            df = _normalize_history_df(df, start, end)
            if df is not None and not df.empty:
                print(f"[fetcher] DNSE hist ✓ {symbol}: {len(df)} nến")
                return df
            print(f"[fetcher] DNSE hist trống {symbol}")
    except Exception as e:
        print(f"[fetcher] DNSE hist ✗ {symbol}: {e}")

    print(f"[fetcher] Không lấy được history {symbol} {start}→{end}")
    return pd.DataFrame()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def get_ohlcv_history(
    symbol: str,
    start: str,
    end: str | None = None,
    resolution: str = "1D",
) -> pd.DataFrame:
    """
    Lấy OHLCV theo khoảng ngày tuyệt đối.

    Args:
        symbol: Mã cổ phiếu, VD "VCB"
        start: Ngày bắt đầu YYYY-MM-DD
        end: Ngày kết thúc YYYY-MM-DD. None = hôm nay.
        resolution: Khung thời gian DNSE/vnstock, mặc định "1D"

    Returns:
        DataFrame với columns [date, open, high, low, close, volume]
    """
    symbol = symbol.upper().strip()
    end = end or _TODAY
    key = f"{symbol}_{start}_{end}_{resolution}_hist"

    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    today = _date.today()
    ttl = None if end_date < today else 4 * 60 * 60
    cached = _load_cache_with_ttl(key, ttl_seconds=ttl)
    if cached is not None:
        print(f"[fetcher] Cache hist ✓ {symbol}")
        return _records_to_df(cached)

    df = _fetch_history(symbol, start, end, resolution=resolution)
    if not df.empty:
        _save_cache(key, _df_to_records(df))
    return df


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
    key    = f"VNINDEX_{_TODAY}_ohlcv_{n_days}"
    cached = _load_cache(key)
    if cached is not None:
        print("[fetcher] Cache ✓ VNINDEX")
        return _records_to_df(cached)

    start, end = _date_range(n_days)
    df = pd.DataFrame()

    # ── vnstock_data Golden (primary) ──
    try:
        raw = get_data_provider().get_ohlcv("VNINDEX", start, end, interval="1D")
        if raw is not None and not raw.empty:
            df = _normalize_df(raw, n_days)
            print(f"[fetcher] vnstock_data ✓ VNINDEX: {len(df)} nến")
    except Exception as e:
        print(f"[fetcher] vnstock_data ✗ VNINDEX: {e} — thử DNSE...")

    # ── DNSE fallback ──
    if df.empty:
        try:
            client = _get_dnse_client()
            if client is not None:
                from_ts = _to_unix_ts(start)
                to_ts   = _to_unix_ts(end) + 86400
                df = client.get_ohlcv("VNINDEX", from_ts, to_ts, resolution="1D", asset_type="INDEX")
                if df is not None and not df.empty:
                    df = _normalize_df(df, n_days)
                    print(f"[fetcher] DNSE ✓ VNINDEX: {len(df)} nến")
        except Exception as e:
            print(f"[fetcher] DNSE ✗ VNINDEX: {e}")

    if not df.empty:
        _save_cache(key, _df_to_records(df))
    return df


def get_vnmidcap(n_days: int = 200) -> pd.DataFrame:
    """
    Lấy OHLCV VN Mid-Cap Index. Cache theo ngày.

    Dùng để phát hiện VNI bị méo bởi VinGroup — khi VNI và VNMidCap
    phân kỳ xu hướng, tham chiếu VNMidCap đáng tin hơn cho thị trường rộng.

    Returns:
        DataFrame với columns [date, open, high, low, close, volume]
        DataFrame trống nếu symbol không hỗ trợ.
    """
    return get_ohlcv("VNMIDCAP", n_days)


# ──────────────────────────────────────────────
# Symbol lists (static — cập nhật định kỳ)
# ──────────────────────────────────────────────

def get_vn30_symbols() -> list[str]:
    try:
        symbols = get_data_provider().get_symbols_by_group("VN30")
        if len(symbols) >= 30:
            print(f"[fetcher] VN30 from vnstock_data: {len(symbols)} symbols")
            return symbols
    except Exception as e:
        print(f"[fetcher] VN30 vnstock_data failed: {e} - using static list")

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
        symbols = get_data_provider().get_symbols_by_group("VN100")
        if len(symbols) >= 90:
            print(f"[fetcher] VN100 from vnstock_data: {len(symbols)} symbols")
            return symbols
    except Exception as e:
        print(f"[fetcher] VN100 vnstock_data failed: {e} - using static list")

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


def get_hnx30_symbols() -> list[str]:
    """
    Lấy 30 mã lớn nhất sàn HNX (chỉ số HNX30 chính thức).

    Dùng Reference.equity.by_group('HNX30') từ vnstock_data.
    Fallback về danh sách cứng HNX30 nếu API lỗi.

    Returns:
        Danh sách 30 mã HNX (thường là HNX30 index).
    """
    try:
        symbols = get_data_provider().get_symbols_by_group("HNX30")
        if len(symbols) >= 25:
            print(f"[fetcher] HNX30 from vnstock_data: {len(symbols)} symbols")
            return symbols
    except Exception as e:
        print(f"[fetcher] HNX30 vnstock_data failed: {e} - using static list")

    # Fallback — HNX30 cập nhật Q1/2025
    return [
        "ACE", "APS", "BVS", "CAP", "CEO",
        "DP3", "DTD", "DVM", "DXP", "HUT",
        "IDC", "IDV", "L14", "L18", "LAS",
        "LHC", "MBS", "NVB", "PLC", "PSD",
        "PTI", "PVB", "SHN", "SHS", "TIG",
        "TNG", "VCS", "VGS", "VNR", "WIN",
    ]


def get_upcom30_symbols() -> list[str]:
    """
    Lấy 30 mã lớn nhất sàn UPCOM theo vốn hóa.

    Vì UPCOM không có index chuẩn 30 mã, hàm lấy toàn bộ mã UPCOM
    từ Reference API rồi ưu tiên các mã trong danh sách blue-chip UPCOM.
    Fallback về danh sách cứng 30 mã lớn nhất nếu API lỗi.

    Returns:
        Danh sách 30 mã UPCOM vốn hóa lớn.
    """
    try:
        symbols = get_data_provider().get_symbols_by_exchange("UPCOM")
        # Lọc các mã hợp lệ (3-4 ký tự alpha)
        symbols = [s for s in symbols if 2 <= len(s) <= 4 and s.replace("-", "").isalpha()]
        if len(symbols) >= 30:
            # Ưu tiên các mã blue-chip UPCOM đã biết, bổ sung thêm để đủ 30
            _UPCOM_BLUECHIP = [
                "BSR", "OIL", "ABI", "VEA", "MCH", "QNS", "VGT", "VHG",
                "ACV", "POT", "VCR", "SBS", "HVN", "ABB", "BIC", "BMI",
                "HKB", "MSR", "NAV", "PGI", "PTL", "SBD", "SGH", "TCR",
                "TIG", "VGI", "VID", "VNT", "VOC", "VTJ",
            ]
            # Ưu tiên bluechip, bổ sung từ danh sách exchange nếu chưa đủ
            result = [s for s in _UPCOM_BLUECHIP if s in symbols]
            remaining = [s for s in symbols if s not in set(result)]
            result = (result + remaining)[:30]
            print(f"[fetcher] UPCOM30 from vnstock_data: {len(result)} symbols (total UPCOM={len(symbols)})")
            return result
    except Exception as e:
        print(f"[fetcher] UPCOM30 vnstock_data failed: {e} - using static list")

    # Fallback — 30 mã UPCOM vốn hóa lớn tiêu biểu
    return [
        "BSR", "OIL", "ABI", "VEA", "MCH",
        "QNS", "VGT", "ACV", "POT", "HVN",
        "ABB", "BIC", "BMI", "HKB", "MSR",
        "NAV", "PGI", "PTL", "SBD", "SGH",
        "TCR", "VGI", "VID", "VNT", "VOC",
        "VTJ", "VGS", "SBS", "BCG", "PLC",
    ]


def get_all_symbols() -> list[str]:
    """
    Lấy toàn bộ mã cổ phiếu niêm yết HOSE từ DNSE instruments API.
    Fallback về vnstock Listing nếu DNSE không khả dụng.
    Cache theo ngày.

    Returns:
        Danh sách mã cổ phiếu HOSE (thường ~400 mã).
    """
    key    = f"all_symbols_{_TODAY}"
    try:
        symbols = get_data_provider().get_symbols_by_exchange("HOSE")
        if len(symbols) >= 100:
            print(f"[fetcher] vnstock_data Reference primary: {len(symbols)} ma HOSE")
            _save_cache(key, [{"symbol": s, "source": "vnstock_data"} for s in symbols])
            return symbols
        print(f"[fetcher] vnstock_data Reference incomplete: {len(symbols)} ma")
    except Exception as e:
        print(f"[fetcher] vnstock_data Reference failed: {e} - trying cache/DNSE...")

    # DNSE instruments primary. `vnstock Listing` remains a fallback below.
    # Keep this block before cache loading during the migration so a stale
    # vnstock-sourced all_symbols cache does not hide DNSE diagnostics.
    try:
        client = _get_dnse_client()
        if client is not None:
            dnse_symbols: list[str] = []
            offset = 0
            limit = 200
            while True:
                status, body = client.get_instruments(
                    market_id="STO",
                    limit=limit,
                    offset=offset,
                )
                if status != 200 or not body:
                    break
                data = json.loads(body)
                items = data.get("data", []) if isinstance(data, dict) else data
                if not items:
                    break
                for item in items:
                    sym = item.get("symbol")
                    if sym and len(sym) == 3 and sym.isalpha():
                        dnse_symbols.append(str(sym).upper())
                total = data.get("total", 0) if isinstance(data, dict) else 0
                offset += limit
                if offset >= (total or offset + 1):
                    break
            dnse_symbols = sorted(set(dnse_symbols))
            if len(dnse_symbols) >= 100:
                print(f"[fetcher] DNSE instruments primary: {len(dnse_symbols)} ma HOSE")
                _save_cache(key, [{"symbol": s, "source": "DNSE"} for s in dnse_symbols])
                return dnse_symbols
            print(f"[fetcher] DNSE instruments incomplete: {len(dnse_symbols)} ma")
        else:
            print("[fetcher] DNSE instruments skipped: missing credentials")
    except Exception as e:
        print(f"[fetcher] DNSE instruments failed: {e} - trying cache/vnstock Listing...")

    cached = _load_cache(key)
    if cached is not None:
        symbols = [r["symbol"] for r in cached if r.get("symbol")]
        print(f"[fetcher] Cache ✓ all_symbols: {len(symbols)} mã")
        return symbols

    # ── vnstock Listing (primary — đầy đủ nhất) ──
    try:
        symbols = get_data_provider().get_symbols_by_exchange("HOSE")
        if len(symbols) >= 100:
            print(f"[fetcher] vnstock Listing ✓: {len(symbols)} mã HOSE")
            _save_cache(key, [{"symbol": s} for s in symbols])
            return symbols
    except Exception as e:
        print(f"[fetcher] vnstock Listing ✗: {e} — thử DNSE instruments...")

    # ── DNSE instruments fallback ──
    try:
        client = _get_dnse_client()
        if client is not None:
            symbols: list[str] = []
            offset = 0
            limit = 200
            while True:
                status, body = client.get_instruments(
                    market_id="STO",
                    limit=limit,
                    offset=offset,
                )
                if status != 200 or not body:
                    break
                data = json.loads(body)
                items = data.get("data", []) if isinstance(data, dict) else data
                if not items:
                    break
                for item in items:
                    sym = item.get("symbol")
                    if sym and len(sym) == 3 and sym.isalpha():
                        symbols.append(str(sym).upper())
                total = data.get("total", 0) if isinstance(data, dict) else 0
                offset += limit
                if offset >= (total or offset + 1):
                    break
            if len(symbols) >= 100:
                print(f"[fetcher] DNSE instruments ✓: {len(symbols)} mã HOSE")
                _save_cache(key, [{"symbol": s} for s in symbols])
                return symbols
    except Exception as e:
        print(f"[fetcher] DNSE instruments ✗: {e}")

    return get_vn100_symbols()


def get_liquid_symbols(
    min_avg_vol: int = 500_000,
    n_days: int = 20,
    max_workers: int = 20,
) -> list[str]:
    """
    Lấy danh sách mã có thanh khoản trung bình >= min_avg_vol cổ/ngày.
    Fetch OHLCV song song (ThreadPoolExecutor) để nhanh hơn.
    Cache theo ngày.

    Args:
        min_avg_vol: Volume trung bình tối thiểu (mặc định 500k)
        n_days:      Số ngày tính trung bình (mặc định 20)
        max_workers: Số thread song song (mặc định 20)

    Returns:
        Danh sách mã đủ điều kiện, sort theo avg volume giảm dần.
    """
    cache_key = f"liquid_symbols_{_TODAY}_{min_avg_vol}_{n_days}"
    cached = _load_cache(cache_key)
    if cached is not None:
        symbols = [r["symbol"] for r in cached if r.get("symbol")]
        print(f"[fetcher] Cache ✓ liquid_symbols: {len(symbols)} mã (vol≥{min_avg_vol:,})")
        return symbols

    all_syms = get_all_symbols()
    print(f"[fetcher] Lọc thanh khoản {len(all_syms)} mã (vol≥{min_avg_vol:,}, {n_days}d) — {max_workers} workers...")

    liquid: list[tuple[str, float]] = []  # (symbol, avg_vol)

    def _fetch_avg_vol(sym: str) -> tuple[str, float]:
        df = _fetch_single(sym, n_days)
        if df.empty:
            return sym, 0.0
        avg = float(df["volume"].tail(n_days).mean())
        return sym, avg

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_avg_vol, s): s for s in all_syms}
        done = 0
        for future in as_completed(futures):
            done += 1
            sym, avg_vol = future.result()
            if avg_vol >= min_avg_vol:
                liquid.append((sym, avg_vol))
            if done % 50 == 0:
                print(f"[fetcher] Đã xử lý {done}/{len(all_syms)} mã, liquid={len(liquid)}...")

    liquid.sort(key=lambda x: x[1], reverse=True)
    result = [sym for sym, _ in liquid]

    print(f"[fetcher] Liquid symbols: {len(result)}/{len(all_syms)} mã đủ vol≥{min_avg_vol:,}")
    _save_cache(cache_key, [{"symbol": s} for s in result])
    return result


# ──────────────────────────────────────────────
# Price Board (real-time snapshot)
# ──────────────────────────────────────────────

def get_price_board(symbols: list[str]) -> pd.DataFrame:
    """
    Lấy bảng giá realtime + thông tin khối ngoại từ data provider.

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
        df = get_data_provider().get_price_board(symbols)
        if df is None or df.empty:
            return pd.DataFrame()
        _save_cache(key, _df_to_records(df))
        print(f"[fetcher] price_board vnstock_data ok {len(df)} ma")
        return df
    except Exception as e:
        print(f"[fetcher] price_board vnstock_data failed: {e}")
        return pd.DataFrame()


def get_live_price(symbols: list[str]) -> dict[str, float]:
    """Lấy giá khớp lệnh hiện tại, KHÔNG cache — dùng cho intraday monitor.

    Thứ tự ưu tiên:
      1. DNSE WebSocket cache (instant, real-time tick)
      2. DNSE REST 1m OHLCV (candle gần nhất, ~1-2 phút trễ)
      3. vnstock_data price_board (fallback cuối)

    Returns:
        {symbol: price} — chỉ gồm các mã lấy được giá.
    """
    result: dict[str, float] = {}
    missing = list(symbols)

    # ── 1. DNSE WebSocket cache ──
    try:
        from multiagents_trading_assistant.services.dnse_ws_price import (
            get_ws_price, is_connected,
        )
        if is_connected():
            still_missing = []
            for sym in missing:
                p = get_ws_price(sym, max_age_seconds=60.0)
                if p is not None:
                    result[sym] = p
                else:
                    still_missing.append(sym)
            missing = still_missing
            if result:
                print(f"[fetcher] get_live_price WS ✓ {len(result)} mã, còn {len(missing)} cần REST")
    except Exception as e:
        print(f"[fetcher] get_live_price WS: {e}")

    # ── 2. DNSE REST 1m fallback ──
    if missing:
        try:
            client = _get_dnse_client()
            if client is not None:
                now_ts  = int(time.time())
                from_ts = now_ts - 300  # 5 phút buffer
                still_missing = []
                for sym in missing:
                    df = client.get_ohlcv(sym, from_ts, now_ts, resolution="1")
                    if df is not None and not df.empty:
                        result[sym] = float(df["close"].iloc[-1])
                    else:
                        still_missing.append(sym)
                if len(missing) - len(still_missing) > 0:
                    print(f"[fetcher] get_live_price DNSE REST ✓ {len(missing) - len(still_missing)} mã")
                missing = still_missing
        except Exception as e:
            print(f"[fetcher] get_live_price DNSE REST: {e}")

    # ── 3. vnstock_data fallback ──
    if missing:
        try:
            raw = get_data_provider().get_price_board(missing)
            for _, row in raw.iterrows():
                sym   = row.get("symbol")
                price = row.get("price")
                if sym and price and float(price) > 0:
                    result[str(sym)] = float(price)
            print(f"[fetcher] get_live_price vnstock_data fallback {len(missing)} ma")
        except Exception as e:
            print(f"[fetcher] get_live_price vnstock_data fail: {e}")

    return result


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

    try:
        vn_result = get_data_provider().get_fundamentals(symbol)
        if vn_result:
            for key_name, value in vn_result.items():
                if key_name in result:
                    result[key_name] = value
            if any(value is not None for value in result.values()):
                print(
                    f"[fetcher] fundamentals vnstock_data ok {symbol}: "
                    f"PE={result['pe']}, ROE={result['roe']}, EPS={result['eps']}"
                )
    except Exception as e:
        print(f"[fetcher] fundamentals vnstock_data failed {symbol}: {e}")

    if all(value is not None for value in result.values()):
        _save_cache(key, [result])
        return result

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

    try:
        vn_result = get_data_provider().get_foreign_flow(symbol, n_days=n_days)
        if vn_result:
            result.update(vn_result)
            if result["net_flow_5d"] is not None or result["net_flow_20d"] is not None:
                print(
                    f"[fetcher] foreign_flow vnstock_data ok {symbol}: "
                    f"net5d={result['net_flow_5d']}, net20d={result['net_flow_20d']}"
                )
    except Exception as e:
        print(f"[fetcher] foreign_flow vnstock_data failed {symbol}: {e}")

    if result["net_flow_5d"] is not None and result["net_flow_20d"] is not None:
        try:
            pb = get_price_board([symbol])
            if not pb.empty:
                row = pb[pb["symbol"] == symbol]
                if not row.empty:
                    r = row.iloc[0]
                    # Ưu tiên foreign_ownership_pct từ summary()
                    # Banks cap=30%, others cap=49%. Dùng 49% làm mẫu số an toàn.
                    own_pct = _safe_float(r.get("foreign_ownership_pct"))
                    if own_pct is not None:
                        result["room_usage_pct"] = round(own_pct / 49.0 * 100, 1)
        except Exception:
            pass
        _save_cache(key, [result])
        return result

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
                own_pct = _safe_float(r.get("foreign_ownership_pct"))
                if own_pct is not None:
                    result["room_usage_pct"] = round(own_pct / 49.0 * 100, 1)

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
