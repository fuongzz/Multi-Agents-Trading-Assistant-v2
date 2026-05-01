"""risk_invest.py — Risk gate cho Investment pipeline. KHÔNG dùng LLM.

Rules:
  1. financial_health = YẾU → VETO (TRÁNH)
  2. Macro BEARISH → downgrade MUA → CHỜ
  3. Margin of safety — ngưỡng động theo chất lượng mã:
       Bluechip (FPT, VCB, MWG, PNJ, REE, ACB, VNM, MBB, TCB): ≥ 20%
       Ngân hàng, FMCG, Tech: ≥ 25%
       Cyclical, BĐS, Chứng khoán, Unknown: ≥ 30%  (default)
  4. Sector concentration ≤ 25% NAV:
       Đếm số vị thế đang giữ cùng ngành từ DB
       ≥ 2 vị thế cùng sector → WARNING
       ≥ 3 vị thế cùng sector → BLOCK (CHỜ)
"""

from multiagents_trading_assistant import database as db


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

    if warnings:
        print(f"[risk_invest] warnings: {'; '.join(warnings)}")
    print(f"[risk_invest] OK — final={action}")

    return {
        "final_action":    action,
        "override_reason": None,
        "warnings":        warnings,
        "sizing_modifier": 1.0,
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
