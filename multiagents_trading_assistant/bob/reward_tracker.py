"""reward_tracker.py — Dual-reward framework từ QuantAgents paper.

Paper: "Otto maximises expected discounted rewards combining:
  rₜˢⁱᵐ (simulated trading reward) và rₜʳᵉᵃˡ (live market reward)
  với adaptive weights wₜˢⁱᵐ, wₜʳᵉᵃˡ điều chỉnh theo relative performance."

Flow:
  1. pipeline_runner gọi record_entry() sau mỗi quyết định MUA
  2. session_monitor gọi record_exit() khi SL/TP hit
  3. Bob's simulator gọi get_adaptive_weights() để tính final_score
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

_STORE_PATH = Path(__file__).parent.parent / "data" / "reward_tracker.json"

# Minimum trades trước khi live weight có ý nghĩa thống kê
_MIN_LIVE_TRADES = 5

# Rolling window để tính live performance
_DEFAULT_WINDOW_DAYS = 30


@dataclass
class TradeOutcome:
    """Một trade thực tế từ Otto's decision."""
    outcome_id: str        # "{symbol}_{entry_date}"
    symbol: str
    setup: str             # setup type Otto dùng để ra quyết định
    entry_date: str        # ISO date
    entry_price: float
    exit_date: str | None = None
    exit_price: float | None = None
    exit_reason: str | None = None   # SL_HIT | TP_HIT | MANUAL
    pnl_pct: float | None = None     # None khi chưa đóng


class RewardTracker:
    """Track live trade outcomes và tính adaptive weights cho Bob.

    Storage: JSON tại data/reward_tracker.json — không cần DB riêng.
    """

    def __init__(self, path: Path = _STORE_PATH) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._outcomes: dict[str, TradeOutcome] = self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict[str, TradeOutcome]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return {r["outcome_id"]: TradeOutcome(**r) for r in raw}
        except Exception:
            return {}

    def _save(self) -> None:
        data = [asdict(o) for o in self._outcomes.values()]
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Record ───────────────────────────────────────────────────────────────

    def record_entry(
        self,
        symbol: str,
        setup: str,
        entry_date: str,
        entry_price: float,
    ) -> None:
        """Gọi sau khi Otto ra quyết định MUA."""
        oid = f"{symbol}_{entry_date}"
        self._outcomes[oid] = TradeOutcome(
            outcome_id=oid,
            symbol=symbol,
            setup=setup,
            entry_date=entry_date,
            entry_price=entry_price,
        )
        self._save()
        print(f"[reward_tracker] Entry recorded: {oid} setup={setup} @ {entry_price:,.0f}")

    def record_exit(
        self,
        symbol: str,
        entry_date: str,
        exit_price: float,
        exit_date: str,
        exit_reason: str,
    ) -> None:
        """Gọi khi session_monitor đóng position (SL/TP hit)."""
        oid = f"{symbol}_{entry_date}"
        outcome = self._outcomes.get(oid)
        if outcome is None:
            # position không được record qua record_entry — tạo stub
            outcome = TradeOutcome(
                outcome_id=oid,
                symbol=symbol,
                setup="UNKNOWN",
                entry_date=entry_date,
                entry_price=0.0,
            )
        outcome.exit_date = exit_date
        outcome.exit_price = exit_price
        outcome.exit_reason = exit_reason
        if outcome.entry_price > 0:
            outcome.pnl_pct = round(
                (exit_price - outcome.entry_price) / outcome.entry_price * 100, 4
            )
        self._outcomes[oid] = outcome
        self._save()
        print(
            f"[reward_tracker] Exit recorded: {oid} reason={exit_reason} "
            f"pnl={outcome.pnl_pct:+.2f}%" if outcome.pnl_pct is not None
            else f"[reward_tracker] Exit recorded: {oid} reason={exit_reason} pnl=n/a"
        )

    # ── Metrics ──────────────────────────────────────────────────────────────

    def get_live_metrics(
        self,
        setup: str | None = None,
        window_days: int = _DEFAULT_WINDOW_DAYS,
    ) -> dict:
        """Tính live performance metrics trong rolling window.

        Returns dict với win_rate, profit_factor, sample_size.
        Chỉ tính trades đã đóng (pnl_pct is not None).
        """
        cutoff = (date.today() - timedelta(days=window_days)).isoformat()
        closed = [
            o for o in self._outcomes.values()
            if o.pnl_pct is not None
            and o.entry_date >= cutoff
            and (setup is None or o.setup == setup)
        ]
        if not closed:
            return {"win_rate": 0.0, "profit_factor": 0.0, "sample_size": 0}

        wins = [o.pnl_pct for o in closed if o.pnl_pct > 0]
        losses = [o.pnl_pct for o in closed if o.pnl_pct <= 0]
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        pf = gross_profit / gross_loss if gross_loss > 0 else (1.0 if gross_profit > 0 else 0.0)

        return {
            "win_rate": len(wins) / len(closed),
            "profit_factor": round(pf, 4),
            "sample_size": len(closed),
        }

    def get_adaptive_weights(
        self,
        sim_profit_factor: float,
        window_days: int = _DEFAULT_WINDOW_DAYS,
    ) -> tuple[float, float]:
        """Tính (w_sim, w_real) dựa trên relative performance gần nhất.

        Paper: "Adaptive weights adjust dynamically based on recent relative
        performance between simulated and real-world trading."

        - Nếu live sample < _MIN_LIVE_TRADES → w_sim=0.8, w_real=0.2 (trust sim more)
        - Sau khi có đủ live data → weights tỉ lệ với profit_factor tương ứng
        """
        live_m = self.get_live_metrics(window_days=window_days)
        n_live = live_m["sample_size"]

        if n_live < _MIN_LIVE_TRADES:
            # Chưa đủ live data — dựa nhiều vào sim
            ramp = n_live / _MIN_LIVE_TRADES          # 0.0 → 1.0
            w_real = 0.2 * ramp
            w_sim = 1.0 - w_real
            return round(w_sim, 4), round(w_real, 4)

        live_pf = live_m["profit_factor"]
        total = sim_profit_factor + live_pf
        if total == 0.0:
            return 0.5, 0.5

        w_sim = sim_profit_factor / total
        w_real = live_pf / total
        return round(w_sim, 4), round(w_real, 4)

    def summary(self, window_days: int = _DEFAULT_WINDOW_DAYS) -> dict:
        live_m = self.get_live_metrics(window_days=window_days)
        open_count = sum(1 for o in self._outcomes.values() if o.pnl_pct is None)
        return {
            "total_recorded": len(self._outcomes),
            "open_positions": open_count,
            "live_window_days": window_days,
            **live_m,
        }
