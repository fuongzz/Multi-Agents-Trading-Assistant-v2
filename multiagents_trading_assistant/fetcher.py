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

def get_ohlcv(symbol: str, n_days: int = 200) -> pd.DataFrame:
    """Lấy OHLCV của một mã. Cache theo ngày."""
    cache_key = f"{symbol}_{TODAY}_ohlcv"
    cached = _load_cache(cache_key)
    if cached:
        return _records_to_df(cached)

    try:
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        df = stock.quote.history(period="1Y", interval="1D")
        # Chuẩn hóa tên cột
        df = df.rename(columns={
            "time": "date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume"
        })
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df = df.tail(n_days).reset_index(drop=True)
        _save_cache(cache_key, _df_to_records(df))
        return df
    except Exception as e:
        print(f"[fetcher] Lỗi get_ohlcv {symbol}: {e}")
        return pd.DataFrame()


def get_ohlcv_batch(symbols: list[str], n_days: int = 200) -> dict[str, pd.DataFrame]:
    """Lấy OHLCV nhiều mã, tự sleep sau mỗi 15 mã."""
    result = {}
    for i, symbol in enumerate(symbols):
        if i > 0 and i % BATCH_SIZE == 0:
            print(f"[fetcher] Sleep {SLEEP_SECONDS}s sau {i} mã (rate limit)...")
            time.sleep(SLEEP_SECONDS)
        result[symbol] = get_ohlcv(symbol, n_days)
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

    try:
        stock = Vnstock().stock(symbol="VNINDEX", source="VCI")
        df = stock.quote.history(period="1Y", interval="1D")
        df = df.rename(columns={"time": "date"})
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df = df.tail(n_days).reset_index(drop=True)
        _save_cache(cache_key, _df_to_records(df))
        return df
    except Exception as e:
        print(f"[fetcher] Lỗi get_vnindex: {e}")
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

    try:
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        # Lấy thông tin định giá
        overview = stock.company.overview()
        ratios   = stock.finance.ratio(period="year", lang="en")

        result = {
            "pe":              None,
            "pb":              None,
            "roe":             None,
            "eps":             None,
            "revenue_growth":  None,
            "profit_growth":   None,
            "industry":        None,
        }

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

        _save_cache(cache_key, result)
        return result
    except Exception as e:
        print(f"[fetcher] Lỗi get_fundamentals {symbol}: {e}")
        return {"pe": None, "pb": None, "roe": None, "eps": None,
                "revenue_growth": None, "profit_growth": None, "industry": None}


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
        # Dữ liệu giao dịch nước ngoài
        fr = stock.trading.foreign_flow(n_days)

        result = {
            "room_usage_pct": None,
            "net_flow_5d":    None,
            "net_flow_20d":   None,
            "flow_history":   [],
        }

        if fr is not None and not fr.empty:
            # net_flow = buy_value - sell_value
            if "buyValue" in fr.columns and "sellValue" in fr.columns:
                fr["net_flow"] = fr["buyValue"] - fr["sellValue"]
                history = fr[["time", "net_flow"]].tail(n_days).to_dict("records")
                result["flow_history"] = [
                    {"date": str(r["time"]), "net_flow": r["net_flow"]}
                    for r in history
                ]
                result["net_flow_5d"]  = float(fr["net_flow"].tail(5).sum())
                result["net_flow_20d"] = float(fr["net_flow"].tail(20).sum())

            # Room sử dụng (%)
            if "roomUsedPct" in fr.columns:
                result["room_usage_pct"] = _safe_float(fr["roomUsedPct"].iloc[-1])

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
