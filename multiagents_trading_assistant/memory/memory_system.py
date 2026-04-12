"""
memory_system.py — 3-layer memory cho AI Trading Assistant.

Layer 1 (L1) — SQLite: lịch sử quyết định, vị thế, tin tức (ngắn hạn, truy vấn nhanh)
Layer 2 (L2) — ChromaDB: vector search cho pattern matching (dài hạn, Phase 4 ChromaDB)
Layer 3 (L3) — Hardcoded: quy tắc bất biến VN market, không thay đổi theo thời gian

L2 (ChromaDB) là optional — fallback về L1 nếu không cài.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from multiagents_trading_assistant import database as db

_VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# Thử import ChromaDB (L2 — optional)
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False

# Thư mục lưu ChromaDB
_CHROMA_PATH = Path(__file__).parent.parent / "cache" / "chromadb"


# ──────────────────────────────────────────────
# L3 — Hardcoded VN Market Rules (bất biến)
# ──────────────────────────────────────────────

L3_VN_RULES = {
    "price_band_hose":     7.0,    # % biên độ HOSE
    "price_band_hnx":     10.0,    # % biên độ HNX
    "settlement_days":     3,      # T+3
    "short_selling":       False,  # Không short
    "atc_start":          "14:25", # ATC bắt đầu
    "atc_end":            "14:30", # ATC kết thúc
    "morning_open":       "09:00",
    "morning_close":      "11:30",
    "afternoon_open":     "13:00",
    "afternoon_close":    "14:30",
    "circuit_breaker_pct": -3.0,   # VNI giảm > 3% → dừng
    "room_critical_pct":   95.0,   # Foreign room > 95% → không mua
    "room_high_pct":       90.0,   # Foreign room > 90% → giảm sizing 50%
    "room_medium_pct":     80.0,   # Foreign room > 80% → giảm sizing 20%
    "max_nav_per_stock":   10.0,   # Tối đa 10% NAV / cổ phiếu
    "max_positions":       10,     # Tối đa 10 vị thế đồng thời
}

L3_SECTOR_MAP = {
    # Sector → danh sách mã tiêu biểu
    "Ngân hàng":      ["VCB", "BID", "CTG", "TCB", "VPB", "MBB", "ACB", "STB", "HDB", "LPB"],
    "Bất động sản":   ["VIC", "VHM", "NVL", "PDR", "DIG", "KDH", "NLG", "DXG", "CEO", "BCM"],
    "Chứng khoán":    ["SSI", "VND", "HCM", "MBS", "VCI", "BSI", "AGR", "CTS", "TVS", "SHS"],
    "Thép":           ["HPG", "HSG", "NKG", "TLH", "VGS", "SMC", "TVN", "TIS", "BVH"],
    "Dầu khí":        ["PVD", "GAS", "PVS", "BSR", "OIL", "PVC", "PVB", "ASP", "PCT"],
    "Tiêu dùng":      ["VNM", "SAB", "MSN", "MWG", "FRT", "PNJ", "DGC", "HAH", "REE"],
    "Công nghệ":      ["FPT", "VGI", "CMG", "ELC", "ITD", "SAM", "VTC", "ICT"],
    "Hàng không":     ["HVN", "VJC", "SCS", "ASC", "NCT"],
    "Điện":           ["REE", "PC1", "GEG", "BCG", "ASM", "POW", "NT2", "TBC", "VSH"],
    "Dược phẩm":      ["DHG", "IMP", "TRA", "PME", "OPC", "VMD", "DBD", "AMV"],
}


# ──────────────────────────────────────────────
# L1 — SQLite Memory (ngắn hạn)
# ──────────────────────────────────────────────

class L1Memory:
    """
    Memory layer 1 — dùng SQLite (database.py).
    Lưu và truy vấn lịch sử quyết định, vị thế, performance.
    """

    # ── Positions ──

    def get_positions(self) -> list[dict]:
        """Danh sách vị thế đang giữ."""
        return db.get_all_positions()

    def has_position(self, symbol: str) -> bool:
        return db.has_position(symbol)

    def get_position(self, symbol: str) -> dict | None:
        return db.get_position(symbol)

    # ── Decisions ──

    def get_recent_decisions(self, symbol: str = None, days: int = 7) -> list[dict]:
        """Lấy decisions gần đây (trong n ngày)."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        decisions = db.get_decisions(symbol=symbol, limit=100)
        return [d for d in decisions if d.get("date", "") >= cutoff]

    def get_decision_streak(self, symbol: str) -> dict:
        """
        Tính streak CHỜ liên tiếp của 1 mã.
        Dùng để phát hiện mã đã được phân tích nhiều ngày liên tiếp mà không có tín hiệu.
        """
        decisions = db.get_decisions(symbol=symbol, limit=20)
        streak_cho = 0
        last_action = None

        for d in decisions:
            fa = d.get("final_action", "CHỜ")
            if fa == "CHỜ":
                streak_cho += 1
            else:
                break
            last_action = fa

        return {
            "symbol":      symbol,
            "streak_cho":  streak_cho,
            "last_action": last_action,
        }

    def get_win_rate(self, symbol: str = None, days: int = 30) -> dict:
        """
        Tính win rate từ decisions có outcome.
        Đơn giản: MUA → giá tăng T+3 = win.
        """
        decisions = db.get_decisions(symbol=symbol, limit=200)
        cutoff    = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        recent    = [d for d in decisions if d.get("date", "") >= cutoff]

        mua_count   = sum(1 for d in recent if d.get("final_action") == "MUA")
        total_acted = sum(1 for d in recent if d.get("final_action") in ("MUA", "BÁN"))

        return {
            "total_decisions": len(recent),
            "total_acted":     total_acted,
            "mua_count":       mua_count,
            "symbol":          symbol or "ALL",
            "days":            days,
        }

    # ── Trades ──

    def get_recent_trades(self, symbol: str = None, days: int = 30) -> list[dict]:
        """Lấy giao dịch thực tế gần đây."""
        history = db.get_trade_history(symbol=symbol, limit=200)
        cutoff  = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [t for t in history if t.get("trade_date", "") >= cutoff]

    def get_t3_status(self, symbol: str) -> bool:
        """True nếu đã mua trong T+3 — không mua lại."""
        buys = db.get_buys_last_n_days(symbol, 3)
        return len(buys) > 0


# ──────────────────────────────────────────────
# L2 — ChromaDB Memory (dài hạn, vector search)
# ──────────────────────────────────────────────

class L2Memory:
    """
    Memory layer 2 — ChromaDB vector search.
    Lưu full pipeline output dưới dạng document để semantic search.
    Dùng Phase 4: "tìm các lần VNM có setup tương tự trong quá khứ"
    """

    def __init__(self):
        self._client = None
        self._collection = None
        if _HAS_CHROMA:
            self._init_chroma()

    def _init_chroma(self) -> None:
        """Kết nối ChromaDB — tạo nếu chưa có."""
        try:
            _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(_CHROMA_PATH),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name="trading_decisions",
                metadata={"description": "Lịch sử pipeline output — AI Trading Assistant"},
            )
            print(f"[memory_L2] ChromaDB kết nối — {self._collection.count()} documents")
        except Exception as e:
            print(f"[memory_L2] ChromaDB lỗi: {e} — L2 disabled")
            self._client = None
            self._collection = None

    @property
    def available(self) -> bool:
        return self._collection is not None

    def save_decision(self, state: dict) -> bool:
        """
        Lưu pipeline output vào ChromaDB.

        Document = text summary của decision.
        Metadata = structured fields để filter.
        """
        if not self.available:
            return False

        symbol = state.get("symbol", "")
        date   = state.get("date", "")
        if not symbol or not date:
            return False

        doc_id = f"{symbol}_{date}"

        # Tạo text document để embed
        trader  = state.get("trader_decision", {})
        risk    = state.get("risk_output", {})
        ptkt    = state.get("ptkt_analysis", {})
        fa      = state.get("fa_analysis", {})

        doc_text = (
            f"Symbol: {symbol} Date: {date} "
            f"Action: {risk.get('final_action','?')} "
            f"Setup: {state.get('setup_type','?')} "
            f"Trend: {ptkt.get('ma_trend','?')} Score: {ptkt.get('confluence_score','?')} "
            f"Valuation: {fa.get('valuation','?')} Health: {fa.get('financial_health','?')} "
            f"Reason: {trader.get('primary_reason','')[:200]}"
        )

        metadata = {
            "symbol":       symbol,
            "date":         date,
            "action":       risk.get("final_action", "CHỜ"),
            "setup_type":   state.get("setup_type", ""),
            "ma_trend":     ptkt.get("ma_trend", ""),
            "confidence":   trader.get("confidence", ""),
        }

        try:
            self._collection.upsert(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[metadata],
            )
            return True
        except Exception as e:
            print(f"[memory_L2] Lỗi save {doc_id}: {e}")
            return False

    def search_similar(
        self,
        query: str,
        symbol: str | None = None,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Tìm các decisions tương tự theo query text.

        Args:
            query:    Text mô tả situation hiện tại
            symbol:   Filter theo mã (optional)
            n_results: Số kết quả trả về

        Returns:
            List dict với fields: id, document, metadata, distance
        """
        if not self.available:
            return []

        where = {"symbol": symbol} if symbol else None

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )

            output = []
            for i, doc_id in enumerate(results["ids"][0]):
                output.append({
                    "id":       doc_id,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None,
                })
            return output
        except Exception as e:
            print(f"[memory_L2] Lỗi search: {e}")
            return []

    def get_historical_actions(self, symbol: str, limit: int = 10) -> list[dict]:
        """Lấy lịch sử actions của 1 mã từ ChromaDB."""
        if not self.available:
            return []

        try:
            results = self._collection.get(
                where={"symbol": symbol},
                limit=limit,
            )
            output = []
            for i, doc_id in enumerate(results["ids"]):
                output.append({
                    "id":       doc_id,
                    "metadata": results["metadatas"][i],
                })
            # Sort by date desc
            output.sort(key=lambda x: x["metadata"].get("date", ""), reverse=True)
            return output
        except Exception as e:
            print(f"[memory_L2] Lỗi get_historical {symbol}: {e}")
            return []


# ──────────────────────────────────────────────
# MemorySystem — Unified interface
# ──────────────────────────────────────────────

class MemorySystem:
    """
    Unified interface cho 3 lớp memory.
    Agents gọi qua đây — không import database trực tiếp.
    """

    def __init__(self):
        self.l1 = L1Memory()
        self.l2 = L2Memory()
        self.l3 = L3_VN_RULES  # dict static

    # ── L3: Rules ──

    def get_rule(self, key: str) -> Any:
        """Lấy hard rule VN (L3)."""
        return self.l3.get(key)

    def get_sector_stocks(self, sector: str) -> list[str]:
        """Danh sách mã trong sector (L3)."""
        return L3_SECTOR_MAP.get(sector, [])

    def get_all_sectors(self) -> list[str]:
        return list(L3_SECTOR_MAP.keys())

    # ── L1: SQLite ──

    def has_position(self, symbol: str) -> bool:
        return self.l1.has_position(symbol)

    def get_positions(self) -> list[dict]:
        return self.l1.get_positions()

    def is_t3_blocked(self, symbol: str) -> bool:
        """True nếu không được mua do T+3."""
        return self.l1.get_t3_status(symbol)

    def get_decision_history(self, symbol: str, days: int = 7) -> list[dict]:
        return self.l1.get_recent_decisions(symbol, days)

    def get_streak(self, symbol: str) -> int:
        """Số ngày CHỜ liên tiếp của mã."""
        return self.l1.get_decision_streak(symbol)["streak_cho"]

    # ── L2: ChromaDB ──

    def save_decision(self, state: dict) -> bool:
        """Lưu pipeline state vào L2 (nếu ChromaDB available)."""
        return self.l2.save_decision(state)

    def find_similar_setups(self, query: str, symbol: str = None, n: int = 3) -> list[dict]:
        """Tìm các setup tương tự trong quá khứ (L2)."""
        return self.l2.search_similar(query, symbol, n)

    # ── Compress (20:00 job) ──

    def compress_daily(self, date: str = None) -> dict:
        """
        Nén memory cuối ngày:
          - Lấy tất cả decisions hôm nay từ L1
          - Lưu vào L2 ChromaDB để searchable
          - Trả về summary

        Gọi bởi scheduler lúc 20:00.
        """
        if date is None:
            date = datetime.now(tz=_VN_TZ).strftime("%Y-%m-%d")

        decisions = db.get_decisions(date=date)
        saved_l2  = 0

        for decision in decisions:
            # Tái tạo minimal state để save vào L2
            minimal_state = {
                "symbol":    decision.get("symbol"),
                "date":      decision.get("date"),
                "setup_type": decision.get("strategy", ""),
                "trader_decision": {
                    "action":         decision.get("action"),
                    "confidence":     decision.get("confidence"),
                    "primary_reason": "",
                },
                "risk_output": {
                    "final_action": decision.get("final_action"),
                    "override_reason": decision.get("override_reason"),
                },
                "ptkt_analysis": {},
                "fa_analysis":   {},
            }
            if self.l2.save_decision(minimal_state):
                saved_l2 += 1

        summary = {
            "date":             date,
            "decisions_today":  len(decisions),
            "saved_to_l2":      saved_l2,
            "l2_available":     self.l2.available,
            "total_positions":  len(self.l1.get_positions()),
        }
        print(f"[memory] Compress xong: {len(decisions)} decisions, {saved_l2} → L2")
        return summary


# ──────────────────────────────────────────────
# Singleton instance
# ──────────────────────────────────────────────

_memory: MemorySystem | None = None

def get_memory() -> MemorySystem:
    """Lấy hoặc tạo singleton MemorySystem."""
    global _memory
    if _memory is None:
        _memory = MemorySystem()
    return _memory


# ──────────────────────────────────────────────
# Test trực tiếp
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from multiagents_trading_assistant import database as db
    db.init_db()

    mem = get_memory()

    print("=== L3 Rules ===")
    print(f"Price band HOSE: ±{mem.get_rule('price_band_hose')}%")
    print(f"Settlement: T+{mem.get_rule('settlement_days')}")
    print(f"Short selling: {mem.get_rule('short_selling')}")
    print(f"Circuit breaker: {mem.get_rule('circuit_breaker_pct')}%")

    print("\n=== L3 Sectors ===")
    print(f"Ngân hàng: {mem.get_sector_stocks('Ngân hàng')[:5]}")

    print("\n=== L1 Positions ===")
    positions = mem.get_positions()
    print(f"Số vị thế: {len(positions)}")
    for p in positions[:3]:
        print(f"  - {p.get('symbol')}: {p.get('entry_price')}")

    print("\n=== L1 T+3 check ===")
    print(f"VNM T+3 blocked: {mem.is_t3_blocked('VNM')}")
    print(f"HPG T+3 blocked: {mem.is_t3_blocked('HPG')}")

    print(f"\n=== L2 ChromaDB ===")
    print(f"Available: {mem.l2.available}")

    print("\n=== Compress Daily ===")
    summary = mem.compress_daily()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
