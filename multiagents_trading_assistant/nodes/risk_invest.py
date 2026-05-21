"""risk_invest.py — Risk gate cho Investment pipeline. KHÔNG dùng LLM.

Rules:
  1. financial_health = YẾU → VETO (TRÁNH)
  2. Macro BEARISH → downgrade MUA → CHỜ
  3. Margin of safety — ngưỡng động theo chất lượng mã:
       Bluechip (FPT, VCB, MWG, PNJ, REE, ACB, VNM, MBB, TCB): ≥ 20%
       Ngân hàng, FMCG, Tech: ≥ 25%
       Cyclical, BĐS, Chứng khoán, Unknown: ≥ 30%  (default)
  4. Sector concentration ≤ 25% NAV (ICB L2):
       Đếm số vị thế đang giữ cùng ngành từ DB
       ≥ 2 vị thế cùng sector → WARNING
       ≥ 3 vị thế cùng sector → BLOCK (CHỜ)
  5. Cycle gate (Layer-1 market regime + Layer-2 sector rotation):
       Market regime DISTRIBUTION / MARKDOWN     → BLOCK MUA (CHỜ)
       Sector L1 state DISTRIBUTION / MARKDOWN   → BLOCK MUA (CHỜ)
       Sector L1 state ACCUMULATION late stage  → WARNING (sizing prudent)
"""

from pathlib import Path

import pandas as pd

from multiagents_trading_assistant import database as db
from multiagents_trading_assistant.research.sector_rotation.icb_mapping import L2_TO_L1


# ── Bluechip — chất lượng cao, biên độ MoS thấp hơn ──
_BLUECHIP = {
    "FPT", "VCB", "MWG", "PNJ", "REE", "ACB", "VNM", "MBB", "TCB",
    "VHM", "VIC", "HPG", "VNM", "GAS",
}

# ── Ngưỡng MoS (%) theo nhóm ngành ICB (giống key trong FiinQuantX) ──
# Cyclical / BĐS / Chứng khoán cần biên an toàn cao hơn
_MOS_BY_SECTOR: dict[str, int] = {
    "BANKS_L2":            25,
    "TECHNOLOGY":          22,
    "FOOD_BEVERAGE":       22,
    "PHARMACEUTICALS":     25,
    "UTILITIES":           25,
    "INDUSTRIALS":         28,
    "REAL_ESTATE":         35,
    "SECURITIES":          35,
    "CONSTRUCTION":        32,
    "MATERIALS":           30,  # includes HPG, HSG
}
_MOS_BLUECHIP  = 20
_MOS_DEFAULT   = 30  # cyclical, unknown

# ── Sector cap — số vị thế cùng ngành tối đa ──
# Mapping ICB industry label → sector group để gom nhóm ngành
_SECTOR_GROUP: dict[str, str] = {
    "BANKS_L2":     "FINANCE",
    "SECURITIES":   "FINANCE",
    "INSURANCE":    "FINANCE",
    "REAL_ESTATE":  "REAL_ESTATE",
    "CONSTRUCTION": "REAL_ESTATE",
    "TECHNOLOGY":   "TECHNOLOGY",
}
_SECTOR_WARN_COUNT  = 2   # ≥ 2 vị thế cùng sector → warning
_SECTOR_BLOCK_COUNT = 3   # ≥ 3 vị thế cùng sector → CHỜ

# ── Cycle gate paths (parquets produced by research/sector_rotation) ──
_SECTOR_ROTATION_PATH = "data/research/sector_rotation/sector_rotation.parquet"
_MARKET_REGIME_PATH   = "data/research/sector_rotation/market_regime.parquet"

_CYCLE_BLOCK_MARKET = {"MARKDOWN", "DISTRIBUTION"}
_CYCLE_BLOCK_SECTOR = {"MARKDOWN", "DISTRIBUTION"}
_CYCLE_WARN_SECTOR  = {"ACCUMULATION"}  # đáy chưa xác nhận — sizing prudent


def _latest_market_regime(root: Path) -> tuple[str, str]:
    """Return (regime, risk_state) for the most recent date in the regime parquet."""
    path = root / _MARKET_REGIME_PATH
    if not path.exists():
        return ("NEUTRAL", "NEUTRAL")
    try:
        df = pd.read_parquet(path, columns=["date", "regime", "risk_state"])
        last = df.sort_values("date").iloc[-1]
        return (str(last["regime"]), str(last["risk_state"]))
    except Exception as exc:
        print(f"[risk_invest] cycle regime read skipped: {exc}")
        return ("NEUTRAL", "NEUTRAL")


def _latest_sector_state(root: Path, industry_l2: str) -> str:
    """Return latest sector L1 state for the L2 industry of this symbol."""
    if not industry_l2:
        return "NEUTRAL"
    sector_l1 = L2_TO_L1.get(industry_l2)
    if not sector_l1:
        return "NEUTRAL"
    path = root / _SECTOR_ROTATION_PATH
    if not path.exists():
        return "NEUTRAL"
    try:
        df = pd.read_parquet(path, columns=["date", "sector_l1", "sector_state"])
        df = df[df["sector_l1"] == sector_l1].sort_values("date")
        if df.empty:
            return "NEUTRAL"
        return str(df.iloc[-1]["sector_state"])
    except Exception as exc:
        print(f"[risk_invest] cycle sector read skipped: {exc}")
        return "NEUTRAL"


def _mos_threshold(symbol: str, industry: str) -> int:
    """Trả ngưỡng MoS phù hợp với chất lượng mã và ngành."""
    if symbol in _BLUECHIP:
        return _MOS_BLUECHIP
    return _MOS_BY_SECTOR.get(industry, _MOS_DEFAULT)


def _sector_position_count(industry: str) -> int:
    """Đếm số vị thế đang giữ thuộc cùng sector group với industry này."""
    if not industry:
        return 0
    try:
        group = _SECTOR_GROUP.get(industry, industry)
        positions = db.get_all_positions()
        count = 0
        for pos in positions:
            pos_sym = pos.get("symbol", "")
            # Dùng strategy field để đoán ngành — nếu không có, bỏ qua
            pos_industry = pos.get("industry", "") or ""
            pos_group = _SECTOR_GROUP.get(pos_industry, pos_industry)
            if pos_group and pos_group == group:
                count += 1
        return count
    except Exception as e:
        print(f"[risk_invest] sector count skipped: {e}")
        return 0


def check(state: dict) -> dict:
    symbol = state.get("symbol", "")
    trader = state.get("trader_decision", {})
    val    = state.get("valuation_analysis", {})
    fa     = state.get("fundamental_analysis", {})
    macro  = state.get("macro_context", {})

    action   = trader.get("action", "CHỜ")
    original = action
    warnings: list[str] = []
    sizing_modifier: float = 1.0

    print(f"[risk_invest] {symbol} — action={action}")

    # Rule 1: financial_health = YẾU → VETO
    health = fa.get("financial_health", "UNKNOWN")
    if health == "YẾU":
        return _override(original, "TRÁNH", "financial_health=YẾU — không đầu tư", warnings)

    # Rule 2: Macro BEARISH → downgrade MUA → CHỜ
    macro_bias = macro.get("macro_bias", "NEUTRAL")
    if action == "MUA" and macro_bias == "BEARISH":
        return _override(original, "CHỜ", "Macro BEARISH — hoãn đầu tư", warnings)

    # Rule 3: Margin of safety — ngưỡng động theo chất lượng mã
    industry = fa.get("vs_industry", "") or val.get("industry", "") or ""
    mos = val.get("margin_of_safety")
    if action == "MUA" and mos is not None:
        threshold = _mos_threshold(symbol, industry)
        if mos < threshold:
            print(f"[risk_invest] MoS {mos:.1f}% < {threshold}% ({symbol}/{industry or 'unknown'}) → CHỜ")
            return _override(
                original, "CHỜ",
                f"Margin of safety {mos:.1f}% < {threshold}% ({industry or 'default'}) — chờ giá tốt hơn",
                warnings,
            )

    # Rule 4: Sector concentration ≤ 25% NAV — đếm vị thế cùng ngành
    if action == "MUA" and industry:
        n = _sector_position_count(industry)
        group = _SECTOR_GROUP.get(industry, industry)
        if n >= _SECTOR_BLOCK_COUNT:
            return _override(
                original, "CHỜ",
                f"Sector {group}: đã có {n} vị thế — vượt giới hạn tập trung ngành",
                warnings,
            )
        if n >= _SECTOR_WARN_COUNT:
            warnings.append(
                f"Sector {group}: đã có {n} vị thế — kiểm tra tổng tỷ trọng ≤ 25% NAV trước khi vào"
            )
    elif action == "MUA":
        warnings.append("Kiểm tra tập trung ngành ≤ 25% NAV trước khi vào")

    # Rule 5: Cycle gate — market regime + sector L1 state
    if action == "MUA":
        repo_root = Path(__file__).resolve().parents[2]
        mkt_regime, mkt_risk = _latest_market_regime(repo_root)
        sector_state = _latest_sector_state(repo_root, industry)

        if mkt_regime in _CYCLE_BLOCK_MARKET:
            return _override(
                original, "CHỜ",
                f"Market regime {mkt_regime} — chu kỳ thị trường không ủng hộ mua mới",
                warnings,
            )
        if sector_state in _CYCLE_BLOCK_SECTOR:
            return _override(
                original, "CHỜ",
                f"Sector {L2_TO_L1.get(industry, industry)} đang {sector_state} — chờ chuyển pha",
                warnings,
            )
        if sector_state in _CYCLE_WARN_SECTOR:
            warnings.append(
                f"Sector {L2_TO_L1.get(industry, industry)} đang ACCUMULATION — đáy chưa xác nhận, sizing thận trọng"
            )
            sizing_modifier = 0.7
        if mkt_risk == "RISK_OFF":
            warnings.append("Market RISK_OFF — cân nhắc trì hoãn vào lệnh")

    if warnings:
        print(f"[risk_invest] warnings: {'; '.join(warnings)}")
    print(f"[risk_invest] OK — final={action}")

    return {
        "final_action":    action,
        "override_reason": None,
        "warnings":        warnings,
        "sizing_modifier": sizing_modifier,
        "original_action": original,
    }


def _override(orig: str, new_action: str, reason: str, warnings: list) -> dict:
    print(f"[risk_invest] OVERRIDE {orig} → {new_action}: {reason}")
    return {
        "final_action":    new_action,
        "override_reason": reason,
        "warnings":        warnings,
        "sizing_modifier": 0.0,
        "original_action": orig,
    }
