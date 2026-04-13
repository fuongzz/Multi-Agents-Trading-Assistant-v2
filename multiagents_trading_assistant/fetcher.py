import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# FIX: pandas 3.x + vnstock compatibility
if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = pd.DataFrame.map

from vnstock import Vnstock

# --- Cấu hình cache ---
BASE_DIR   = Path(__file__).parent
CACHE_DIR  = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

BATCH_SIZE    = 15
SLEEP_SECONDS = 60  # vnstock free tier ~15 req/phút
TODAY         = datetime.now().strftime("%Y-%m-%d")


# ──────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _load_cache(key: str) -> dict | None:
    p = _cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_cache(key: str, data) -> None:
    try:
        _cache_path(key).write_text(
            json.dumps(data, ensure_ascii=False, default=str),
            encoding="utf-8"
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
# OHLCV
# ──────────────────────────────────────────────

def _fetch_ohlcv_yfinance(symbol: str, n_days: int = 200) -> pd.DataFrame:
    """
    Lấy OHLCV qua yfinance với suffix .VN (ví dụ ACB → ACB.VN).
    Không phụ thuộc vnstock, không bị rate limit VCI.
    """
    ticker = yf.Ticker(f"{symbol}.VN")
    raw = ticker.history(period="1y", interval="1d")
    if raw.empty:
        return pd.DataFrame()
    raw = raw.reset_index()
    raw.columns = [c.lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    df = raw[["date", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("date").tail(n_days).reset_index(drop=True)
    return df


def get_ohlcv(symbol: str, n_days: int = 200) -> pd.DataFrame:
    """
    Lấy OHLCV của một mã. Cache theo ngày.
    Ưu tiên: yfinance (.VN) → VCI vnstock fallback
    """
    from datetime import date as _date, timedelta

    cache_key = f"{symbol}_{TODAY}_ohlcv"
    cached = _load_cache(cache_key)
    if cached:
        return _records_to_df(cached)

    # Ưu tiên yfinance — không phụ thuộc VCI server
    try:
        df = _fetch_ohlcv_yfinance(symbol, n_days)
        if not df.empty:
            _save_cache(cache_key, _df_to_records(df))
            return df
        print(f"[fetcher] yfinance {symbol}.VN trả về rỗng — thử VCI...")
    except Exception as e:
        print(f"[fetcher] yfinance lỗi {symbol}: {e} — thử VCI...")

    # Fallback VCI qua vnstock
    end_date   = _date.today().strftime("%Y-%m-%d")
    start_date = (_date.today() - timedelta(days=max(n_days * 2, 400))).strftime("%Y-%m-%d")
    try:
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        df = stock.quote.history(start=start_date, end=end_date, interval="1D")
        df = df.rename(columns={"time": "date"})
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df = df.tail(n_days).reset_index(drop=True)
        if not df.empty:
            _save_cache(cache_key, _df_to_records(df))
            return df
    except Exception as e:
        print(f"[fetcher] VCI lỗi {symbol}: {e}")

    return pd.DataFrame()


def get_ohlcv_batch(symbols: list[str], n_days: int = 200) -> dict[str, pd.DataFrame]:
    """
    Lấy OHLCV nhiều mã song song qua yf.download() — 1 request cho tất cả.
    Mã nào yfinance không có → fallback get_ohlcv() tuần tự.
    """
    result: dict[str, pd.DataFrame] = {}

    # Kiểm tra cache trước — bỏ qua mã đã có cache hôm nay
    missing = []
    for sym in symbols:
        cached = _load_cache(f"{sym}_{TODAY}_ohlcv")
        if cached:
            result[sym] = _records_to_df(cached)
        else:
            missing.append(sym)

    if not missing:
        print(f"[fetcher] Tất cả {len(symbols)} mã đã có cache hôm nay")
        return result

    print(f"[fetcher] Download song song {len(missing)} mã qua yfinance...")
    tickers_vn = [f"{s}.VN" for s in missing]

    try:
        import logging, warnings
        # Tắt warning "possibly delisted" của yfinance
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        warnings.filterwarnings("ignore")

        raw = yf.download(
            tickers   = tickers_vn,
            period    = "1y",
            interval  = "1d",
            group_by  = "ticker",
            auto_adjust = True,
            progress  = False,
            threads   = True,
        )
        warnings.filterwarnings("default")

        for sym, ticker_vn in zip(missing, tickers_vn):
            try:
                # Multi-ticker: raw[ticker_vn] là DataFrame riêng
                if len(missing) == 1:
                    df_sym = raw.copy()
                else:
                    df_sym = raw[ticker_vn].copy()

                df_sym = df_sym.dropna(subset=["Close"])
                if df_sym.empty:
                    raise ValueError("rỗng")

                df_sym = df_sym.reset_index()
                df_sym.columns = [c.lower() for c in df_sym.columns]
                df_sym["date"] = pd.to_datetime(df_sym["date"]).dt.strftime("%Y-%m-%d")
                df_sym = df_sym[["date", "open", "high", "low", "close", "volume"]].tail(n_days).reset_index(drop=True)
                _save_cache(f"{sym}_{TODAY}_ohlcv", _df_to_records(df_sym))
                result[sym] = df_sym
            except Exception:
                # Mã không có trên yfinance (delisted/không niêm yết) → skip, không fallback VCI
                result[sym] = pd.DataFrame()

    except Exception as e:
        print(f"[fetcher] yf.download lỗi: {e} — fallback tuần tự...")
        for sym in missing:
            result[sym] = get_ohlcv(sym, n_days)

    ok  = sum(1 for v in result.values() if v is not None and not v.empty)
    print(f"[fetcher] Batch xong: {ok}/{len(symbols)} mã có data")
    return result


# ──────────────────────────────────────────────
# VN-Index
# ──────────────────────────────────────────────

def get_vnindex(n_days: int = 200) -> pd.DataFrame:
    """Lấy OHLCV VN-Index."""
    cache_key = f"VNINDEX_{TODAY}_ohlcv"
    cached = _load_cache(cache_key)
    if cached:
        return _records_to_df(cached)

    from datetime import date as _date, timedelta
    end_date   = _date.today().strftime("%Y-%m-%d")
    start_date = (_date.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    # Thử VCI với date range (tránh RetryError khi dùng period="1Y")
    try:
        stock = Vnstock().stock(symbol="VNINDEX", source="VCI")
        df = stock.quote.history(start=start_date, end=end_date, interval="1D")
        df = df.rename(columns={"time": "date"})
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df = df.tail(n_days).reset_index(drop=True)
        if not df.empty:
            _save_cache(cache_key, _df_to_records(df))
            print("[fetcher] VNINDEX OK từ VCI (date range)")
            return df
    except Exception as e:
        print(f"[fetcher] get_vnindex VCI lỗi: {e}")

    # Fallback: VCI dùng MSN proxy (VN30 ETF E1VFVN30 → VNINDEX proxy)
    try:
        stock = Vnstock().stock(symbol="VNINDEX", source="MSN")
        df = stock.quote.history(start=start_date, end=end_date, interval="1D")
        df = df.rename(columns={"time": "date"})
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df = df.tail(n_days).reset_index(drop=True)
        if not df.empty:
            _save_cache(cache_key, _df_to_records(df))
            print("[fetcher] VNINDEX OK từ MSN")
            return df
    except Exception as e:
        print(f"[fetcher] get_vnindex MSN lỗi: {e}")

    # Fallback cuối: yfinance VN30 ETF làm proxy
    for yf_ticker in ("VNINDEX.VN", "E1VFVN30.VN"):
        try:
            print(f"[fetcher] Fallback VNINDEX → yfinance {yf_ticker}")
            raw = yf.Ticker(yf_ticker).history(period="1y", interval="1d")
            if raw.empty:
                continue
            raw = raw.reset_index()
            raw.columns = [c.lower() for c in raw.columns]
            raw["date"] = pd.to_datetime(raw["date"]).dt.date.astype(str)
            df = raw[["date", "open", "high", "low", "close", "volume"]].tail(n_days).reset_index(drop=True)
            if not df.empty:
                _save_cache(cache_key, _df_to_records(df))
                print(f"[fetcher] VNINDEX OK từ yfinance {yf_ticker}")
                return df
        except Exception as e:
            print(f"[fetcher] yfinance {yf_ticker} lỗi: {e}")

    return pd.DataFrame()


# ──────────────────────────────────────────────
# Fundamentals
# ──────────────────────────────────────────────

def get_fundamentals(symbol: str) -> dict:
    """Lấy P/E, P/B, ROE, EPS, tăng trưởng doanh thu/lợi nhuận."""
    cache_key = f"{symbol}_{TODAY}_fundamentals"
    cached = _load_cache(cache_key)
    if cached:
        return cached

    result = {
        "pe": None, "pb": None, "roe": None, "eps": None,
        "revenue_growth": None, "profit_growth": None, "industry": None,
    }

    # Thử VCI qua vnstock
    try:
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        overview = stock.company.overview()
        ratios   = stock.finance.ratio(period="year", lang="en")

        if overview is not None and not overview.empty:
            row = overview.iloc[0]
            result["industry"] = str(row.get("industryName", ""))

        if ratios is not None and not ratios.empty:
            latest = ratios.iloc[-1]
            result["pe"]             = _safe_float(latest.get("priceToEarning"))
            result["pb"]             = _safe_float(latest.get("priceToBook"))
            result["roe"]            = _safe_float(latest.get("roe"))
            result["eps"]            = _safe_float(latest.get("earningPerShare"))
            result["revenue_growth"] = _safe_float(latest.get("revenueGrowth"))
            result["profit_growth"]  = _safe_float(latest.get("postTaxProfitGrowth"))

        if any(v is not None for v in result.values()):
            _save_cache(cache_key, result)
            return result
    except Exception as e:
        print(f"[fetcher] get_fundamentals VCI lỗi {symbol}: {e} — thử yfinance...")

    # Fallback yfinance
    try:
        ticker = yf.Ticker(f"{symbol}.VN")
        info = ticker.info
        if info:
            result["pe"]       = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
            result["pb"]       = _safe_float(info.get("priceToBook"))
            result["roe"]      = _safe_float(info.get("returnOnEquity"))
            result["eps"]      = _safe_float(info.get("trailingEps"))
            result["industry"] = info.get("industry") or info.get("sector") or ""
            result["revenue_growth"] = _safe_float(info.get("revenueGrowth"))
            result["profit_growth"]  = _safe_float(info.get("earningsGrowth"))
    except Exception as e:
        print(f"[fetcher] get_fundamentals yfinance lỗi {symbol}: {e}")

    _save_cache(cache_key, result)
    return result


# ──────────────────────────────────────────────
# Foreign Flow
# ──────────────────────────────────────────────

def get_foreign_flow(symbol: str, n_days: int = 20) -> dict:
    """Lấy dữ liệu khối ngoại: room, net flow."""
    cache_key = f"{symbol}_{TODAY}_foreign_flow"
    cached = _load_cache(cache_key)
    if cached:
        return cached

    try:
        stock = Vnstock().stock(symbol=symbol, source="VCI")

        result = {
            "room_usage_pct": None,
            "net_flow_5d":    None,
            "net_flow_20d":   None,
            "flow_history":   [],
        }

        # Thử lấy dữ liệu khối ngoại qua quote history với cột buy/sell ngoại
        from datetime import date as _date, timedelta
        end_date   = _date.today().strftime("%Y-%m-%d")
        start_date = (_date.today() - timedelta(days=60)).strftime("%Y-%m-%d")

        try:
            fr = stock.quote.history(start=start_date, end=end_date, interval="1D")
            if fr is not None and not fr.empty:
                # Tìm cột liên quan đến khối ngoại
                buy_cols  = [c for c in fr.columns if "foreign" in c.lower() and "buy" in c.lower()]
                sell_cols = [c for c in fr.columns if "foreign" in c.lower() and "sell" in c.lower()]

                if buy_cols and sell_cols:
                    fr["net_flow"] = fr[buy_cols[0]] - fr[sell_cols[0]]
                    result["net_flow_5d"]  = float(fr["net_flow"].tail(5).sum())
                    result["net_flow_20d"] = float(fr["net_flow"].tail(n_days).sum())
                    result["flow_history"] = [
                        {"date": str(r.get("time", "")), "net_flow": r["net_flow"]}
                        for r in fr[["time", "net_flow"]].tail(n_days).to_dict("records")
                    ]
        except Exception:
            pass  # không có cột khối ngoại → trả về None, agent xử lý

        _save_cache(cache_key, result)
        return result
    except Exception as e:
        print(f"[fetcher] Lỗi get_foreign_flow {symbol}: {e}")
        return {"room_usage_pct": None, "net_flow_5d": None, "net_flow_20d": None, "flow_history": []}


# ──────────────────────────────────────────────
# Global Macro (yfinance)
# ──────────────────────────────────────────────

# Tickers yahoo finance tương ứng
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
    """Lấy giá thị trường toàn cầu. Cache 4 giờ."""
    cache_key = f"global_macro_{TODAY}"
    cached = _load_cache(cache_key)
    if cached:
        return cached

    result = {}
    try:
        for name, ticker in _MACRO_TICKERS.items():
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period="5d")
                if len(hist) >= 2:
                    current   = float(hist["Close"].iloc[-1])
                    prev      = float(hist["Close"].iloc[-2])
                    change_pct = round((current - prev) / prev * 100, 2)
                    result[name] = {"current": current, "change_pct": change_pct}
                else:
                    result[name] = {"current": None, "change_pct": None}
            except Exception:
                result[name] = {"current": None, "change_pct": None}

        _save_cache(cache_key, result)
    except Exception as e:
        print(f"[fetcher] Lỗi get_global_macro: {e}")

    return result


# ──────────────────────────────────────────────
# VN Macro
# ──────────────────────────────────────────────

def get_vn_macro() -> dict:
    """Lấy USD/VND và lãi suất. Cache 1 ngày."""
    cache_key = f"vn_macro_{TODAY}"
    cached = _load_cache(cache_key)
    if cached:
        return cached

    result = {"usd_vnd": None, "interbank_rate": None, "sbv_rate": 4.5}
    try:
        # USD/VND từ Yahoo Finance
        t = yf.Ticker("VND=X")
        hist = t.history(period="5d")
        if not hist.empty:
            result["usd_vnd"] = float(hist["Close"].iloc[-1])

        # Lãi suất liên ngân hàng (tạm hardcode SBV rate, thực tế cần scrape SBV)
        result["sbv_rate"] = 4.5  # % — cập nhật thủ công khi SBV thay đổi

        _save_cache(cache_key, result)
    except Exception as e:
        print(f"[fetcher] Lỗi get_vn_macro: {e}")

    return result


# ──────────────────────────────────────────────
# Sector data (dùng cho Screener)
# ──────────────────────────────────────────────

def get_sector_ohlcv(sector_index: str, n_days: int = 20) -> pd.DataFrame:
    """Lấy OHLCV của sector index (VN30, VNFIN, VNREAL, ...)."""
    cache_key = f"{sector_index}_{TODAY}_ohlcv"
    cached = _load_cache(cache_key)
    if cached:
        return _records_to_df(cached)

    try:
        stock = Vnstock().stock(symbol=sector_index, source="VCI")
        df = stock.quote.history(period="1Y", interval="1D")
        df = df.rename(columns={"time": "date"})
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df = df.tail(n_days).reset_index(drop=True)
        _save_cache(cache_key, _df_to_records(df))
        return df
    except Exception as e:
        print(f"[fetcher] Lỗi get_sector_ohlcv {sector_index}: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# VN100 symbol list
# ──────────────────────────────────────────────

def get_vn30_symbols() -> list[str]:
    """Trả về danh sách 30 mã VN30 cứng — tiết kiệm API call."""
    return [
        "ACB", "BCM", "BID", "BVH", "CTG",
        "FPT", "GAS", "GVR", "HDB", "HPG",
        "MBB", "MSN", "MWG", "PLX", "POW",
        "SAB", "SHB", "SSB", "SSI", "STB",
        "TCB", "TPB", "VCB", "VHM", "VIB",
        "VIC", "VJC", "VNM", "VPB", "VRE",
    ]


def get_vn100_symbols() -> list[str]:
    """Trả về danh sách mã VN100."""
    try:
        from vnstock import Vnstock
        vc = Vnstock()
        df = vc.stock(symbol="VN100", source="VCI").listing.symbols_by_industries()
        if df is not None and not df.empty and "symbol" in df.columns:
            return df["symbol"].tolist()
    except Exception as e:
        print(f"[fetcher] Không lấy được VN100, dùng danh sách cứng: {e}")

    # Danh sách cứng top 30 khi API lỗi
    return [
        "VNM", "VIC", "VHM", "VCB", "BID", "CTG", "MBB", "TCB",
        "HPG", "GAS", "SAB", "MSN", "VJC", "HDB", "STB", "FPT",
        "SSI", "VND", "MWG", "PNJ", "REE", "DGC", "NVL", "PDR",
        "DPM", "PLX", "BSR", "BCM", "ACB", "EIB",
    ]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _safe_float(val) -> float | None:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
