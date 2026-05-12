"""
dnse_ws_price.py — DNSE WebSocket real-time price feed.

Kết nối tới wss://ws-openapi.dnse.com.vn, subscribe Trade channel,
duy trì price cache trong bộ nhớ để session_monitor đọc tức thì.

Chạy asyncio event loop trong 1 daemon thread riêng — không block scheduler.

Dùng:
    from multiagents_trading_assistant.services.dnse_ws_price import (
        start_ws_price_feed,
        get_ws_price,
        subscribe_symbols,
        is_connected,
    )

    # Khởi động (1 lần khi scheduler start)
    start_ws_price_feed(["VCB", "VNM", "HPG", ...])

    # Lấy giá (thread-safe, instant)
    price = get_ws_price("VCB")   # None nếu chưa có tick
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

_log = logging.getLogger("dnse_ws")

_WS_URL = "wss://ws-openapi.dnse.com.vn"
_VN_TZ  = ZoneInfo("Asia/Ho_Chi_Minh")

# Boards HOSE thường dùng
_DEFAULT_BOARDS = ["G1", "G3", "G4"]

# ── Thread-safe price cache ──
# {symbol: {"price": float, "volume": int, "ts": float}}
_price_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()

# ── Trạng thái kết nối ──
_ws_thread: threading.Thread | None = None
_ws_loop:   asyncio.AbstractEventLoop | None = None
_connected  = threading.Event()
_subscribed_symbols: set[str] = set()


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def start_ws_price_feed(symbols: list[str]) -> None:
    """
    Khởi động WebSocket price feed trong background thread.
    Gọi 1 lần khi APScheduler start. Idempotent — gọi lại sẽ bị ignore.
    """
    global _ws_thread, _ws_loop

    if _ws_thread is not None and _ws_thread.is_alive():
        _log.info("[ws_price] Đã chạy, bỏ qua.")
        subscribe_symbols(symbols)
        return

    api_key    = os.getenv("DNSE_API_KEY", "").strip()
    api_secret = os.getenv("DNSE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        _log.warning("[ws_price] DNSE_API_KEY/SECRET chưa set — WebSocket bị tắt.")
        return

    _subscribed_symbols.update(symbols)

    def _run():
        global _ws_loop
        _ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_ws_loop)
        _ws_loop.run_until_complete(
            _ws_main(api_key, api_secret, list(_subscribed_symbols))
        )

    _ws_thread = threading.Thread(target=_run, daemon=True, name="dnse-ws-price")
    _ws_thread.start()
    _log.info(f"[ws_price] Thread khởi động, subscribe {len(symbols)} mã.")


def subscribe_symbols(symbols: list[str]) -> None:
    """
    Thêm symbols vào subscription trong khi WS đang chạy.
    Nếu WS chưa kết nối, lưu lại và subscribe khi connected.
    """
    new = set(symbols) - _subscribed_symbols
    if not new:
        return
    _subscribed_symbols.update(new)

    if _ws_loop and _connected.is_set():
        api_key    = os.getenv("DNSE_API_KEY", "").strip()
        api_secret = os.getenv("DNSE_API_SECRET", "").strip()
        asyncio.run_coroutine_threadsafe(
            _subscribe(_ws_loop._running_connection, list(new)),  # type: ignore
            _ws_loop,
        )


def get_ws_price(symbol: str, max_age_seconds: float = 30.0) -> float | None:
    """
    Lấy giá live từ WS cache.

    Args:
        symbol: mã CK
        max_age_seconds: nếu tick cũ hơn ngưỡng này → trả None (stale)

    Returns:
        float giá gần nhất, hoặc None nếu chưa có / stale / WS không chạy
    """
    with _cache_lock:
        entry = _price_cache.get(symbol)
    if not entry:
        return None
    age = time.time() - entry["ts"]
    if age > max_age_seconds:
        return None
    return entry["price"]


def get_ws_tick(symbol: str, max_age_seconds: float = 30.0) -> dict | None:
    """
    Lay snapshot tick moi nhat tu WS cache.

    Returns:
        {"symbol", "price", "volume", "ts", "age_seconds"} hoac None neu chua co/stale.
    """
    symbol = symbol.upper().strip()
    with _cache_lock:
        entry = dict(_price_cache.get(symbol) or {})
    if not entry:
        return None
    age = time.time() - float(entry.get("ts", 0))
    if age > max_age_seconds:
        return None
    return {
        "symbol": symbol,
        "price": entry.get("price"),
        "volume": entry.get("volume", 0),
        "ts": entry.get("ts"),
        "age_seconds": age,
    }


def get_all_ws_prices() -> dict[str, float]:
    """Trả về dict tất cả giá trong cache (không lọc staleness)."""
    with _cache_lock:
        return {sym: v["price"] for sym, v in _price_cache.items()}


def is_connected() -> bool:
    """True nếu WS đang kết nối và đã auth thành công."""
    return _connected.is_set()


def get_cache_stats() -> dict:
    """Debug info: số mã trong cache, last update."""
    with _cache_lock:
        count = len(_price_cache)
        if _price_cache:
            latest_ts = max(v["ts"] for v in _price_cache.values())
            latest_sym = max(_price_cache, key=lambda s: _price_cache[s]["ts"])
            age = round(time.time() - latest_ts, 1)
        else:
            latest_sym, age = None, None
    return {
        "connected": is_connected(),
        "symbols_cached": count,
        "latest_symbol": latest_sym,
        "latest_age_seconds": age,
    }


# ──────────────────────────────────────────────
# Internal — WebSocket lifecycle
# ──────────────────────────────────────────────

def _make_signature(api_key: str, api_secret: str, timestamp: int, nonce: str) -> str:
    """HMAC-SHA256 hex của '{api_key}:{timestamp}:{nonce}'."""
    message = f"{api_key}:{timestamp}:{nonce}"
    return hmac.new(
        api_secret.encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()


async def _authenticate(ws, api_key: str, api_secret: str) -> bool:
    """Gửi auth message, đợi auth_success. Returns True nếu thành công."""
    timestamp = int(time.time())
    nonce = str(int(time.time() * 1_000_000))
    signature = _make_signature(api_key, api_secret, timestamp, nonce)

    auth_msg = json.dumps({
        "action":    "auth",
        "api_key":   api_key,
        "signature": signature,
        "timestamp": timestamp,
        "nonce":     nonce,
    })
    await ws.send(auth_msg)

    try:
        resp = await asyncio.wait_for(ws.recv(), timeout=10.0)
        data = json.loads(resp)
        if data.get("action") == "auth_success":
            _log.info("[ws_price] Auth thành công.")
            return True
        _log.error(f"[ws_price] Auth thất bại: {data}")
        return False
    except asyncio.TimeoutError:
        _log.error("[ws_price] Auth timeout.")
        return False


async def _subscribe(ws, symbols: list[str], boards: list[str] | None = None) -> None:
    """Subscribe Trade channel cho danh sách symbol."""
    if boards is None:
        boards = _DEFAULT_BOARDS

    channels = [
        {"name": f"tick.{board}.json", "symbols": symbols}
        for board in boards
    ]
    msg = json.dumps({"action": "subscribe", "channels": channels})
    await ws.send(msg)
    _log.info(f"[ws_price] Subscribe {len(symbols)} mã trên boards {boards}")


def _handle_tick(data: dict) -> None:
    """Parse tick message và cập nhật cache."""
    # DNSE Trade model fields (từ SDK models.py)
    sym   = data.get("sym") or data.get("symbol")
    price = data.get("mp")  or data.get("matchPrice") or data.get("price")
    qty   = data.get("mq")  or data.get("matchQtty")  or data.get("quantity") or 0

    if not sym or not price:
        return

    try:
        p = float(price)
        q = int(qty)
    except (TypeError, ValueError):
        return

    with _cache_lock:
        _price_cache[sym] = {
            "price":  p,
            "volume": q,
            "ts":     time.time(),
        }


async def _ws_main(api_key: str, api_secret: str, symbols: list[str]) -> None:
    """Main coroutine — kết nối, auth, subscribe, đọc messages với auto-reconnect."""
    import websockets

    reconnect_delay = 5.0
    max_reconnect_delay = 60.0

    while True:
        try:
            _log.info(f"[ws_price] Kết nối {_WS_URL} ...")
            async with websockets.connect(
                _WS_URL,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10,
            ) as ws:
                # Nhận welcome message
                welcome_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                welcome = json.loads(welcome_raw)
                session_id = welcome.get("session_id") or welcome.get("sid", "?")
                _log.info(f"[ws_price] Connected. Session: {session_id}")

                # Auth
                if not await _authenticate(ws, api_key, api_secret):
                    _connected.clear()
                    await asyncio.sleep(reconnect_delay)
                    continue

                # Subscribe
                current_symbols = list(_subscribed_symbols) or symbols
                await _subscribe(ws, current_symbols)
                _connected.set()
                reconnect_delay = 5.0  # reset delay sau khi connect thành công

                # Read loop
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue

                    action = data.get("action") or data.get("type") or data.get("event")

                    if action in ("tick", "trade", None):
                        _handle_tick(data)
                    elif action == "subscribe_success":
                        _log.debug(f"[ws_price] Subscribe OK: {data.get('channels')}")
                    elif action == "error":
                        _log.warning(f"[ws_price] Server error: {data}")

        except Exception as e:
            _connected.clear()
            _log.warning(f"[ws_price] Ngắt kết nối: {e}. Reconnect sau {reconnect_delay:.0f}s ...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
