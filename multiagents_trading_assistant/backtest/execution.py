"""ExecutionSimulator — mô phỏng fill thực tế của HOSE.

Ràng buộc thực tế cần mô phỏng:
  - ATO fill: lệnh từ bar i được fill ở open của bar i+1.
  - Biên độ ±7% (HOSE) — không fill vượt giá trần/sàn.
  - Slippage cơ bản: fill ở giá xấu hơn open một chút.
  - T+2.5: cổ phiếu mua hôm nay không bán được trong 2.5 ngày.
  - Liquidity: lệnh quá lớn so với volume bar → giả định fill được nhưng có thể
    thêm slippage. Phase 1 đơn giản hóa: bỏ qua, vì size 30%/3% NAV thường nhỏ
    so với volume HOSE.

API:
  simulate_fill(side, intended_price, next_bar) -> FillResult
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FillResult:
    filled: bool
    price: float
    reason: str = ""


@dataclass
class ExecutionConfig:
    price_band_pct: float = 7.0     # HOSE ±7%
    slippage_bps: float = 10.0      # 10 bps = 0.1% mỗi chiều
    settlement_days: int = 2        # T+2 (HOSE: hàng về T+2)
    use_open_fill: bool = True      # True = fill ở open bar kế (ATO)


def _apply_slippage(price: float, side: str, bps: float) -> float:
    factor = bps / 10000.0
    if side == "buy":
        return round(price * (1 + factor), 2)
    return round(price * (1 - factor), 2)


def _within_band(fill_price: float, prev_close: float, band_pct: float) -> bool:
    if prev_close <= 0:
        return True
    upper = prev_close * (1 + band_pct / 100.0)
    lower = prev_close * (1 - band_pct / 100.0)
    return lower <= fill_price <= upper


def simulate_fill(
    side: str,                      # "buy" or "sell"
    next_bar_open: float,
    next_bar_high: float,
    next_bar_low: float,
    prev_close: float,
    cfg: Optional[ExecutionConfig] = None,
) -> FillResult:
    """Mô phỏng fill ở open của bar kế tiếp.

    - Fill base price = next_bar_open.
    - Áp slippage theo chiều buy/sell.
    - Reject nếu fill vượt biên độ (giá trần/sàn).
    """
    cfg = cfg or ExecutionConfig()
    if next_bar_open <= 0:
        return FillResult(filled=False, price=0.0, reason="invalid open")

    base = next_bar_open
    slipped = _apply_slippage(base, side, cfg.slippage_bps)

    # Clamp trong range của bar kế (không fill vượt high/low của chính bar đó)
    if side == "buy":
        slipped = min(slipped, next_bar_high)
    else:
        slipped = max(slipped, next_bar_low)

    if not _within_band(slipped, prev_close, cfg.price_band_pct):
        return FillResult(
            filled=False,
            price=slipped,
            reason=f"out of {cfg.price_band_pct}% band vs prev_close",
        )

    return FillResult(filled=True, price=round(slipped, 2), reason="ok")


def can_sell_today(buy_bar_idx: int, today_idx: int, cfg: ExecutionConfig) -> bool:
    """T+N rule: hàng mua tại buy_bar_idx về tài khoản tại buy_bar_idx + N."""
    return today_idx >= buy_bar_idx + cfg.settlement_days
