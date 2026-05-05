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


# ── LLM Backtest execution layer ──────────────────────────────────────────────
# Dùng bởi llm_backtest.py — tách riêng để không ảnh hưởng rule-based engine.

from typing import Literal  # noqa: E402

@dataclass
class LLMExecutionConfig:
    """Config cho execution simulator trong LLM backtest.

    model:
      "next_open"          → fill ở open bar kế, luôn fill (trừ khi vi phạm band)
      "zone_only"          → chỉ fill nếu next_open nằm trong entry_zone
      "zone_with_slippage" → zone_only + thêm slippage

    max_position_vol_pct: max position size / avg_daily_volume
    """
    model: Literal["next_open", "zone_only", "zone_with_slippage"] = "zone_with_slippage"
    slippage_bps: float = 20.0              # 0.2%
    price_band_pct: float = 7.0             # HOSE ±7%
    max_position_vol_pct: float = 0.10      # max 10% avg daily volume
    settlement_days: int = 2                # T+2


@dataclass
class LLMFillResult:
    filled: bool
    price: float
    reason: str = ""
    skipped_volume: bool = False
    skipped_zone: bool = False


def simulate_llm_entry(
    entry_zone: tuple[float, float],
    stop_loss: float,
    position_pct: float,          # 0.01–0.10 (% NAV)
    nav: float,                   # NAV hiện tại (VND)
    next_bar: dict,               # {open, high, low, close, volume}
    avg_volume: float,
    cfg: LLMExecutionConfig | None = None,
) -> LLMFillResult:
    """Mô phỏng entry cho LLM backtest.

    Args:
        entry_zone: (low, high) vùng giá hợp lệ.
        stop_loss: Giá SL để tính position_shares.
        position_pct: Tỷ trọng NAV đề xuất.
        nav: NAV hiện tại dùng tính số cổ phiếu.
        next_bar: OHLCV bar ngày thực thi.
        avg_volume: Volume trung bình 20 phiên (để check liquidity).
        cfg: LLMExecutionConfig.

    Returns:
        LLMFillResult
    """
    cfg = cfg or LLMExecutionConfig()
    next_open = float(next_bar.get("open", 0))
    prev_close = float(next_bar.get("prev_close", next_open))

    if next_open <= 0:
        return LLMFillResult(filled=False, price=0.0, reason="invalid open")

    # Zone check
    entry_low, entry_high = entry_zone
    in_zone = entry_low <= next_open <= entry_high

    if cfg.model in ("zone_only", "zone_with_slippage") and not in_zone:
        return LLMFillResult(
            filled=False, price=next_open,
            reason=f"next_open {next_open} ngoài entry_zone [{entry_low}, {entry_high}]",
            skipped_zone=True,
        )

    # Slippage
    base_price = next_open
    if cfg.model == "zone_with_slippage":
        factor = cfg.slippage_bps / 10_000
        base_price = round(next_open * (1 + factor), 2)

    # Clamp trong high/low của bar
    bar_high = float(next_bar.get("high", base_price))
    bar_low = float(next_bar.get("low", base_price))
    fill_price = min(base_price, bar_high)

    # Price band check
    if prev_close > 0:
        upper = prev_close * (1 + cfg.price_band_pct / 100)
        lower = prev_close * (1 - cfg.price_band_pct / 100)
        if not (lower <= fill_price <= upper):
            return LLMFillResult(
                filled=False, price=fill_price,
                reason=f"fill {fill_price} vượt biên độ ±{cfg.price_band_pct}%",
            )

    # Volume constraint — check position_shares vs avg_volume
    if fill_price > 0 and avg_volume > 0 and nav > 0:
        position_value = nav * position_pct
        position_shares = position_value / fill_price
        if position_shares > avg_volume * cfg.max_position_vol_pct:
            return LLMFillResult(
                filled=False, price=fill_price,
                reason=(
                    f"position_shares {position_shares:.0f} > "
                    f"{cfg.max_position_vol_pct:.0%} avg_volume ({avg_volume:.0f})"
                ),
                skipped_volume=True,
            )

    return LLMFillResult(filled=True, price=fill_price, reason="ok")
