"""
dnse_client.py — DNSE LightSpeed API REST client.

Authentication: HMAC SHA256 signature theo format của DNSE SDK chính thức.
Ref: https://github.com/dnse-tech/openapi-sdk/tree/main/python

Chỉ dùng stdlib + urllib3 (đã có sẵn qua requests). Không thêm dependency mới.
"""

import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from urllib import parse
from uuid import uuid4

import pandas as pd
import urllib3


_BASE_URL = "https://openapi.dnse.com.vn"

# ── index symbols DNSE nhận là INDEX type ──
_INDEX_SYMBOLS = {"VNINDEX", "VN30", "HNX", "UPCOM", "HNX30", "VN100", "VNMIDCAP",
                  "VNDIVIDEND", "VN50GROWTH", "VNXALLSHARE", "VNMITECH"}


def _build_signature(
    secret: str,
    method: str,
    path: str,
    date_value: str,
    nonce: str | None = None,
) -> tuple[str, str]:
    """
    Tạo chữ ký HMAC SHA256 theo format DNSE SDK.

    Signature string:
        "(request-target): {method} {path}\\n
         date: {date_value}"
        + "\\nnonce: {nonce}" nếu có nonce

    Returns (headers_list, url_encoded_signature).
    """
    header_key = "date"
    headers_list = f"(request-target) {header_key}"
    signature_string = f"(request-target): {method.lower()} {path}\n{header_key}: {date_value}"
    if nonce:
        signature_string += f"\nnonce: {nonce}"

    mac = hmac.new(
        secret.encode("utf-8"),
        signature_string.encode("utf-8"),
        hashlib.sha256,
    )
    encoded = base64.b64encode(mac.digest()).decode("utf-8")
    escaped = parse.quote(encoded, safe="")
    return headers_list, escaped


class DNSERestClient:
    """REST client cho DNSE LightSpeed API — OHLCV historical data."""

    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret
        self._http = urllib3.PoolManager(
            num_pools=5,
            maxsize=5,
            timeout=urllib3.Timeout(connect=10.0, read=30.0),
        )

    def _date_header(self) -> str:
        return datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")

    def _make_headers(self, method: str, path: str) -> dict:
        date_value = self._date_header()
        nonce = uuid4().hex
        headers_list, signature = _build_signature(
            self._api_secret, method, path, date_value, nonce=nonce
        )
        sig_header = (
            f'Signature keyId="{self._api_key}",algorithm="hmac-sha256",'
            f'headers="{headers_list}",signature="{signature}",nonce="{nonce}"'
        )
        return {
            "Date": date_value,
            "X-Signature": sig_header,
            "x-api-key": self._api_key,
            "Accept": "application/json",
        }

    def get_instruments(
        self,
        market_id: str | None = None,
        security_group_id: str | None = None,
        index_name: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> tuple[int, str]:
        """GET /instruments — danh sách chứng khoán."""
        path = "/instruments"
        query: dict = {"limit": limit, "offset": offset}
        if market_id:
            query["marketId"] = market_id
        if security_group_id:
            query["securityGroupId"] = security_group_id
        if index_name:
            query["indexName"] = index_name
        qs  = parse.urlencode(query)
        url = f"{_BASE_URL}{path}?{qs}"
        headers = self._make_headers("GET", path)
        try:
            resp = self._http.request("GET", url, headers=headers)
            return resp.status, resp.data.decode("utf-8")
        except Exception as e:
            return 500, str(e)

    def get_ohlcv_raw(
        self,
        symbol: str,
        resolution: str,
        from_ts: int,
        to_ts: int,
        asset_type: str = "STOCK",
    ) -> dict | None:
        """
        Gọi GET /price/ohlc cho 1 khoảng thời gian.
        Returns parsed JSON dict hoặc None nếu lỗi.
        """
        path = "/price/ohlc"
        query = {
            "symbol": symbol,
            "type": asset_type,
            "resolution": resolution,
            "from": str(from_ts),
            "to": str(to_ts),
        }
        url = f"{_BASE_URL}{path}?{parse.urlencode(query)}"
        headers = self._make_headers("GET", path)

        try:
            resp = self._http.request("GET", url, headers=headers)
            if resp.status == 200:
                return json.loads(resp.data.decode("utf-8"))
            print(f"[dnse] HTTP {resp.status}: {resp.data.decode('utf-8')[:200]}")
            return None
        except Exception as e:
            print(f"[dnse] Request error: {e}")
            return None

    def get_ohlcv(
        self,
        symbol: str,
        from_ts: int,
        to_ts: int,
        resolution: str = "1D",
        asset_type: str | None = None,
    ) -> pd.DataFrame:
        """
        Lấy OHLCV lịch sử, tự động paginate nếu nextTime > 0.

        Returns DataFrame [date, open, high, low, close, volume].
        DataFrame trống nếu lỗi hoặc không có data.
        """
        if asset_type is None:
            asset_type = "INDEX" if symbol.upper() in _INDEX_SYMBOLS else "STOCK"

        all_t, all_o, all_h, all_l, all_c, all_v = [], [], [], [], [], []
        current_from = from_ts
        max_pages = 20

        for _ in range(max_pages):
            data = self.get_ohlcv_raw(symbol, resolution, current_from, to_ts, asset_type)
            if not data or not data.get("t"):
                break

            all_t.extend(data["t"])
            all_o.extend(data["o"])
            all_h.extend(data["h"])
            all_l.extend(data["l"])
            all_c.extend(data["c"])
            all_v.extend(data["v"])

            next_time = data.get("nextTime", 0)
            if not next_time or next_time <= 0:
                break
            current_from = next_time

        if not all_t:
            return pd.DataFrame()

        df = pd.DataFrame({
            "date":   pd.to_datetime(all_t, unit="s", utc=True).tz_convert("Asia/Ho_Chi_Minh").tz_localize(None).normalize(),
            "open":   all_o,
            "high":   all_h,
            "low":    all_l,
            "close":  all_c,
            "volume": all_v,
        })
        df = df.sort_values("date").reset_index(drop=True)
        return df


# ── Lazy singleton ──

_dnse_client: DNSERestClient | None = None


def _get_dnse_client() -> DNSERestClient | None:
    """
    Trả về DNSERestClient singleton. Trả None nếu credentials chưa cấu hình.
    Không raise — caller tự xử lý fallback.
    """
    global _dnse_client
    if _dnse_client is not None:
        return _dnse_client

    api_key = os.getenv("DNSE_API_KEY", "").strip()
    api_secret = os.getenv("DNSE_API_SECRET", "").strip()

    if not api_key or not api_secret:
        return None

    _dnse_client = DNSERestClient(api_key, api_secret)
    return _dnse_client
