"""Export one HTML dashboard for independent paper strategy sleeves."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from scripts.audit_current_paper_contract import write_audit
from scripts.inject_sortable_tables import SORTABLE_JS


ROOT = Path(__file__).resolve().parents[1]
PRICE_UNIT_MULTIPLIER = 1000.0
INDICATIVE_ENTRY_BAND_PCT = 0.01
PENDING_SIGNAL_COLUMNS = [
    "sleeve_id",
    "symbol",
    "strategy_name",
    "signal_date",
    "decision_status",
    "score",
    "reference_close_vnd",
    "max_positions_context",
    "created_at",
    "source",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:,.4g}"
    return str(value)


def _to_number(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace("%", "").replace("+", "").replace("VND", "").replace(" ", "")
    if "." in cleaned and "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_vn_number(value: float, decimals: int = 0, signed: bool = False, suffix: str = "") -> str:
    sign = "+" if signed and value > 0 else ""
    if decimals <= 0:
        text = f"{abs(value):,.0f}".replace(",", ".")
        prefix = "-" if value < 0 else sign
        return f"{prefix}{text}{suffix}"
    text = f"{abs(value):,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    prefix = "-" if value < 0 else sign
    return f"{prefix}{text}{suffix}"


def _column_kind(column: str) -> str:
    name = column.lower()
    if name in {"sleeve_id", "symbol", "strategy_name", "strategy", "label", "logic", "mode", "status"}:
        return "text"
    if "date" in name:
        return "date"
    if name in {"shares", "open_positions", "closed_trades", "candidate_count", "target_count", "max_positions", "days_held", "priority", "rank"}:
        return "integer"
    if "pct" in name or "return" in name or "win_rate" in name:
        return "percent"
    if "weight" in name:
        return "weight"
    if any(token in name for token in ["pnl", "equity", "cash", "value", "price", "capital", "fees", "cost"]):
        return "money"
    if "score" in name:
        return "score"
    return "default"


def _format_cell(column: str, value: Any) -> str:
    number = _to_number(value)
    kind = _column_kind(column)
    if number is None:
        return "" if value is None else str(value)
    if kind in {"money", "integer"}:
        return _format_vn_number(number, decimals=0)
    if kind == "percent":
        return _format_vn_number(number, decimals=2, signed=True, suffix="%")
    if kind == "weight":
        pct = number * 100.0 if abs(number) <= 1.0 else number
        return _format_vn_number(pct, decimals=2, signed=False, suffix="%")
    if kind == "score":
        return _format_vn_number(number, decimals=2)
    return _format_vn_number(number, decimals=2) if not float(number).is_integer() else _format_vn_number(number)


def _cell_class(column: str, value: Any) -> str:
    name = column.lower()
    text = str(value or "").upper()
    number = _to_number(value)
    if text in {"RISK_ON", "HOLD_STRONG"}:
        return "win"
    if text in {"RISK_OFF", "WATCH_WEAKENING"}:
        return "loss"
    if text in {"NEUTRAL", "UNKNOWN"}:
        return "flat"
    if number is None:
        return ""
    if any(token in name for token in ["pnl", "return", "pct"]):
        if number > 0:
            return "win"
        if number < 0:
            return "loss"
        return "flat"
    return ""


def _pulse_class(value: Any) -> str:
    text = str(value or "").upper()
    if text in {"DONG_TIEN_DANG_CAI_THIEN", "DONG_TIEN_VAO_MANH", "DONG_TIEN_VAO", "DUY_TRI_TIEN_VAO", "DANG_TANG_TOC"}:
        return "win"
    if text in {"REGIME_CHAN_GIAI_NGAN", "AP_LUC_PHAN_PHOI_TANG", "AP_LUC_PHAN_PHOI", "DANG_GIAM_NHIET", "ROI_NHOM_TIEN_VAO"}:
        return "loss"
    return "flat"


def _pulse_label(value: Any) -> str:
    labels = {
        "REGIME_CHAN_GIAI_NGAN": "Regime chặn giải ngân",
        "DONG_TIEN_DANG_CAI_THIEN": "Dòng tiền đang cải thiện",
        "AP_LUC_PHAN_PHOI_TANG": "Áp lực phân phối tăng",
        "TRUNG_LAP_THEO_DOI": "Trung lập / theo dõi",
        "DONG_TIEN_VAO_MANH": "Vào mạnh",
        "DONG_TIEN_VAO": "Tiền vào",
        "AP_LUC_PHAN_PHOI": "Phân phối",
        "THEO_DOI": "Theo dõi",
        "DUY_TRI_TIEN_VAO": "Duy trì vào",
        "DANG_TANG_TOC": "Tăng tốc",
        "DANG_GIAM_NHIET": "Giảm nhiệt",
        "ROI_NHOM_TIEN_VAO": "Rời nhóm tiền vào",
        "DI_NGANG": "Đi ngang",
    }
    text = str(value or "")
    return labels.get(text, text or "-")


def _money_flow_rows(frame: pd.DataFrame, *, recent: bool) -> str:
    score_column = "flow_score_avg_5d" if recent else "flow_score_today"
    bars: list[str] = []
    for row in frame.head(8).to_dict("records"):
        symbol = html.escape(str(row.get("symbol") or "-"))
        score = float(row.get(score_column) or 0.0)
        width = max(3.0, min(100.0, score))
        state = row.get("flow_trend_5d") if recent else row.get("flow_signal")
        detail = (
            f"{_pulse_label(row.get('flow_trend_5d'))} | {int(row.get('inflow_days_5d') or 0)}/5 phiên"
            if recent
            else f"{_pulse_label(row.get('flow_signal'))} | {_pulse_label(row.get('flow_trend_5d'))}"
        )
        bars.append(
            f'<div class="money-row"><b>{symbol}</b><div class="money-track">'
            f'<i class="money-bar {_pulse_class(state)}" style="width:{width:.1f}%"></i></div>'
            f'<strong>{score:.1f}</strong><span>{html.escape(detail)}</span></div>'
        )
    return "".join(bars) if bars else '<p class="money-empty">Không có mã thỏa điều kiện.</p>'


def _money_flow_panel(pulse: dict[str, Any], today: pd.DataFrame, recent: pd.DataFrame) -> str:
    if not pulse.get("available"):
        return ""
    pulse_state = str(pulse.get("pulse_state") or "TRUNG_LAP_THEO_DOI")
    return f"""<section class="money-panel">
  <div class="money-head">
    <div>
      <h2>Dòng tiền thị trường theo Flow V2</h2>
      <p>Dữ liệu quan sát tại {html.escape(str(pulse.get("date") or "-"))} | VN100 | Ngữ cảnh chung cho 5 sleeve đang chạy</p>
    </div>
    <strong class="money-pill {_pulse_class(pulse_state)}">{html.escape(_pulse_label(pulse_state))}</strong>
  </div>
  <div class="money-metrics">
    <div><span>Tín hiệu hôm nay</span><strong>{html.escape(_fmt(pulse.get("strong_inflow_symbols_today")))}</strong></div>
    <div><span>Có tín hiệu >=3/5 phiên</span><strong>{html.escape(_fmt(pulse.get("persistent_inflow_symbols_5d")))}</strong></div>
    <div><span>CHDM20 đổi 5 phiên</span><strong>{html.escape(_fmt(pulse.get("mkt_CHDM20_change_5d")))}</strong></div>
    <div><span>DS20 đổi 5 phiên</span><strong>{html.escape(_fmt(pulse.get("mkt_DS20_change_5d")))}</strong></div>
  </div>
  <div class="money-sections">
    <section class="money-block">
      <h3>1. Tiền vào hôm nay</h3>
      <p>Chỉ gồm mã đạt proxy dòng tiền ở phiên mới nhất; thanh là score hôm nay.</p>
      {_money_flow_rows(today, recent=False)}
      <a href="14_money_flow_today.html">Mở dữ liệu hôm nay</a>
    </section>
    <section class="money-block">
      <h3>2. Dòng tiền mạnh thời gian vừa qua</h3>
      <p>Mã đạt proxy ít nhất 2/5 phiên; thanh là score trung bình 5 phiên.</p>
      {_money_flow_rows(recent, recent=True)}
      <a href="15_money_flow_recent.html">Mở dữ liệu 5 phiên</a>
    </section>
  </div>
  <p class="money-note">{html.escape(str(pulse.get("interpretation_note") or ""))}</p>
</section>"""


def _realtime_flow_confirmation_label(value: Any) -> str:
    labels = {
        "XAC_NHAN_VAO_TRONG_PHIEN": "Giá tăng và khối ngoại mua ròng",
        "KHOI_NGOAI_MUA_RONG": "Khối ngoại mua ròng",
        "GIA_DANG_XAC_NHAN": "Giá đang xác nhận",
        "SUY_YEU_TRONG_PHIEN": "Giá giảm và khối ngoại bán ròng",
        "THEO_DOI_TRONG_PHIEN": "Theo dõi trong phiên",
        "CHUA_CO_SNAPSHOT": "Chưa có snapshot",
    }
    return labels.get(str(value or ""), str(value or "-"))


def _realtime_flow_confirmation_class(value: Any) -> str:
    if value in {"XAC_NHAN_VAO_TRONG_PHIEN", "KHOI_NGOAI_MUA_RONG", "GIA_DANG_XAC_NHAN"}:
        return "win"
    if value == "SUY_YEU_TRONG_PHIEN":
        return "loss"
    return "flat"


def _realtime_money_flow_panel(status: dict[str, Any], frame: pd.DataFrame) -> str:
    if not status:
        return ""
    state = str(status.get("state") or "WAITING_FOR_MARKET_SESSION")
    live = state in {"LIVE", "LIVE_CACHED", "LIVE_NO_QUOTES"}
    state_label = "ĐANG CẬP NHẬT LIVE" if live else "CHỜ GIỜ GIAO DỊCH"
    state_class = "win" if live else "flat"
    rows: list[str] = []
    for row in frame.head(8).to_dict("records"):
        symbol = html.escape(str(row.get("symbol") or "-"))
        price = html.escape(_format_cell("price", row.get("close_price")) or "-")
        change = html.escape(_format_cell("return_pct", row.get("intraday_change_pct")) or "-")
        foreign_net = html.escape(_format_cell("shares", row.get("foreign_net_volume")) or "-")
        confirmation = row.get("realtime_confirmation")
        rows.append(
            f'<div class="live-flow-row"><b>{symbol}</b><strong>{price} ({change})</strong>'
            f'<em class="{_realtime_flow_confirmation_class(confirmation)}">{html.escape(_realtime_flow_confirmation_label(confirmation))}</em>'
            f'<span>KL ngoại ròng: {foreign_net}</span></div>'
        )
    rows_html = "".join(rows) if rows else '<p class="money-empty">Chưa có snapshot live trong phiên hiện tại.</p>'
    last_update = html.escape(str(status.get("last_live_update") or "-"))
    return f"""<section class="live-flow-panel">
  <div class="money-head">
    <div>
      <h2>Xác nhận dòng tiền realtime</h2>
      <p>Watchlist lấy từ Flow V2 EOD; giá và khối lượng mua/bán khối ngoại cập nhật trong giờ giao dịch. Snapshot gần nhất: {last_update}</p>
    </div>
    <strong class="money-pill {state_class}">{state_label}</strong>
  </div>
  <div class="idea-warning"><strong>DISPLAY ONLY:</strong> Đây là lớp xác nhận intraday cho người dùng, không phải score Flow V2 đã backtest và không thay đổi gate, target hoặc lệnh paper.</div>
  <div class="live-flow-grid">{rows_html}</div>
  <a class="idea-link" href="20_realtime_money_flow.html">Mở bảng realtime chi tiết</a>
</section>"""


def _discovery_rows(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<p class="money-empty">Không có mã đạt điều kiện riêng khi bỏ market gate.</p>'
    rows: list[str] = []
    for row in frame.head(6).to_dict("records"):
        symbol = html.escape(str(row.get("symbol") or "-"))
        strategy = html.escape(str(row.get("strategy_name") or "-"))
        score = _format_cell("score", row.get("score"))
        explanation = html.escape(str(row.get("explanation") or ""))
        rows.append(
            f'<div class="idea-row"><div><b>{symbol}</b><strong>{strategy}</strong></div>'
            f'<em>Điểm {html.escape(score)}</em><p>{explanation}</p></div>'
        )
    return "".join(rows)


def _ungated_discovery_panel(core: pd.DataFrame, flow: pd.DataFrame, as_of: str) -> str:
    if core.empty and flow.empty:
        return ""
    if core.empty:
        return f"""<section class="idea-panel">
  <div class="idea-head">
    <div>
      <h2>Cơ hội quan sát khi bỏ market gate</h2>
      <p>Danh sách mở rộng tại {html.escape(as_of or "-")} | VN100 | Giải thích theo logic của các chiến lược đang chạy</p>
    </div>
    <strong class="idea-pill">KHÔNG ĐẶT LỆNH</strong>
  </div>
  <div class="idea-warning">Phần này chỉ gỡ bỏ điều kiện thị trường để người dùng quan sát cổ phiếu và lý do xuất hiện. Paper trading vẫn dùng gate và target hiện tại như cũ.</div>
  <div class="money-sections">
    <section class="money-block">
      <h3>Flow V2: 3 sleeve rotation</h3>
      <p>Giữ bộ lọc chất lượng dòng tiền/high-RS, bỏ market gate trước khi xếp hạng.</p>
      {_discovery_rows(flow)}
    </section>
  </div>
  <a class="idea-link" href="16_ungated_discovery.html">Mở dữ liệu và giải thích đầy đủ</a>
</section>"""
    return f"""<section class="idea-panel">
  <div class="idea-head">
    <div>
      <h2>Cơ hội quan sát khi bỏ market gate</h2>
      <p>Danh sách mở rộng tại {html.escape(as_of or "-")} | VN100 | Giải thích theo logic của các chiến lược đang chạy</p>
    </div>
    <strong class="idea-pill">KHÔNG ĐẶT LỆNH</strong>
  </div>
  <div class="idea-warning">Phần này chỉ gỡ bỏ điều kiện thị trường để người dùng quan sát cổ phiếu và lý do xuất hiện. Paper trading vẫn dùng gate và target hiện tại như cũ.</div>
  <div class="money-sections">
    <section class="money-block">
      <h3>Core MVP9 rank2</h3>
      <p>Giữ điều kiện riêng của từng strategy, bỏ các filter bắt đầu bằng <code>mkt_</code>; hiển thị dưới sleeve active <code>core_mvp9_rank2</code>.</p>
      {_discovery_rows(core)}
    </section>
    <section class="money-block">
      <h3>Flow V2: 3 sleeve rotation</h3>
      <p>Giữ bộ lọc chất lượng dòng tiền/high-RS, bỏ market gate trước khi xếp hạng.</p>
      {_discovery_rows(flow)}
    </section>
  </div>
  <a class="idea-link" href="16_ungated_discovery.html">Mở dữ liệu và giải thích đầy đủ</a>
</section>"""


def _parse_risk(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        if pd.isna(value):
            return {}
    except Exception:
        pass
    try:
        payload = json.loads(str(value))
        return payload if isinstance(payload, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _reference_price_vnd(row: dict[str, Any]) -> float | None:
    reference = _to_number(row.get("reference_price_vnd"))
    if reference and reference > 0:
        return reference
    close = _to_number(row.get("close") if row.get("close") is not None else row.get("reference_close"))
    return close * PRICE_UNIT_MULTIPLIER if close and close > 0 else None


def _strategy_contract(sleeve_id: str, risk: dict[str, Any]) -> tuple[float | None, float | None, str]:
    if sleeve_id == "hostile_combo_long":
        return None, None, "Thoát khi close T mất MA20, paper exit ở open T+1; không dùng TP-SL cố định."
    if sleeve_id == "flow_v2_baseline":
        return None, None, "Thoát theo vòng quay audited / kỳ rebalance; không gắn TP-SL cố định."
    if sleeve_id == "flow_v2_tiered":
        return None, None, "Thoát theo flow_momentum_tiered; không gắn TP-SL cố định."
    if sleeve_id == "flow_v2":
        return 0.08, 0.25, "Paper overlay: SL -8%, TP +25%, tối đa 20 phiên."
    stop_pct = _to_number(risk.get("stop_loss")) or 0.08
    target_pct = _to_number(risk.get("take_profit")) or 0.25
    holding = int(_to_number(risk.get("max_holding_bars")) or 30)
    return stop_pct, target_pct, f"Risk strategy: SL -{stop_pct * 100:.1f}%, TP +{target_pct * 100:.1f}%, tối đa {holding} phiên."


def _signal_plan_row(row: dict[str, Any], *, sleeve_id: str, source_status: str) -> dict[str, Any]:
    price = _reference_price_vnd(row)
    risk = _parse_risk(row.get("risk"))
    stop_pct, target_pct, expectation = _strategy_contract(sleeve_id, risk)
    is_target = source_status == "PAPER_TARGET"
    signal_date = row.get("signal_date") or row.get("feature_date") or row.get("date")
    return {
        "sleeve_id": sleeve_id,
        "symbol": row.get("symbol"),
        "strategy_name": row.get("strategy_name") or "Flow V2 high-RS flow-heavy ranking",
        "signal_date": signal_date,
        "decision_status": source_status,
        "market_regime_state": row.get("market_regime_state"),
        "score": row.get("score") if row.get("score") is not None else row.get("flow_v2_rotation_score", row.get("edge_rank_score")),
        "reference_close_vnd": round(price, 2) if price else None,
        "entry_zone_low_vnd": round(price * (1.0 - INDICATIVE_ENTRY_BAND_PCT), 2) if price else None,
        "entry_zone_high_vnd": round(price * (1.0 + INDICATIVE_ENTRY_BAND_PCT), 2) if price else None,
        "entry_execution": (
            "Paper có thể tự fill bằng quote live đầu tiên từ phiên T+1; lệnh thật vẫn cần phê duyệt."
            if is_target
            else "Vùng theo dõi +/-1% quanh close T; chỉ xét lệnh khi tín hiệu đủ gate."
        ),
        "stop_loss_vnd": round(price * (1.0 - stop_pct), 2) if price and stop_pct is not None else None,
        "take_profit_vnd": round(price * (1.0 + target_pct), 2) if price and target_pct is not None else None,
        "reward_risk_ratio": round(target_pct / stop_pct, 2) if stop_pct and target_pct is not None else None,
        "expectation": expectation,
        "gate_explanation": (
            "Đủ điều kiện paper; tín hiệu sẽ tuân theo execution contract."
            if is_target
            else str(row.get("relaxed_rule") or "Danh sách quan sát mở rộng; không tạo lệnh paper.")
        ),
        "display_only": not is_target,
    }


def _build_signal_plans(
    orders_targets: pd.DataFrame,
    candidates: pd.DataFrame,
    ungated_discovery: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_keys: set[tuple[str, str]] = set()
    candidate_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in candidates.to_dict("records"):
        key = (str(candidate.get("sleeve_id") or ""), str(candidate.get("symbol") or ""))
        candidate_lookup[key] = candidate
    for target in orders_targets.to_dict("records"):
        sleeve_id = str(target.get("sleeve_id") or "")
        symbol = str(target.get("symbol") or "")
        if not sleeve_id or not symbol:
            continue
        merged = {**candidate_lookup.get((sleeve_id, symbol), {}), **target}
        rows.append(_signal_plan_row(merged, sleeve_id=sleeve_id, source_status="PAPER_TARGET"))
        target_keys.add((sleeve_id, symbol))
    for idea in ungated_discovery.to_dict("records"):
        sleeves = [value.strip() for value in str(idea.get("sleeves") or "").split(",") if value.strip()]
        for sleeve_id in sleeves:
            key = (sleeve_id, str(idea.get("symbol") or ""))
            if key not in target_keys:
                rows.append(_signal_plan_row(idea, sleeve_id=sleeve_id, source_status="OBSERVE_ONLY_GATE_REMOVED"))
    return pd.DataFrame(rows)


def _append_pending_signal_orders(out_dir: Path, signal_plans: pd.DataFrame, orders_targets: pd.DataFrame) -> pd.DataFrame:
    path = out_dir / "paper_signal_pending.csv"
    if signal_plans.empty:
        existing = _read_csv(path)
        return existing if not existing.empty else pd.DataFrame(columns=PENDING_SIGNAL_COLUMNS)

    pending = signal_plans[signal_plans["decision_status"].astype(str).eq("PAPER_TARGET")].copy()
    if pending.empty:
        existing = _read_csv(path)
        return existing if not existing.empty else pd.DataFrame(columns=PENDING_SIGNAL_COLUMNS)

    details = pd.DataFrame()
    if not orders_targets.empty and {"sleeve_id", "symbol", "max_positions_context"}.issubset(orders_targets.columns):
        details = orders_targets[["sleeve_id", "symbol", "max_positions_context"]].drop_duplicates()
    if not details.empty:
        pending = pending.merge(details, on=["sleeve_id", "symbol"], how="left")
    elif "max_positions_context" not in pending:
        pending["max_positions_context"] = pd.NA

    pending["created_at"] = datetime.now().isoformat(timespec="seconds")
    pending["source"] = "signal_plan_export"
    for column in PENDING_SIGNAL_COLUMNS:
        if column not in pending:
            pending[column] = pd.NA
    pending = pending[PENDING_SIGNAL_COLUMNS].copy()
    pending["symbol"] = pending["symbol"].astype(str).str.upper().str.strip()
    pending["signal_date"] = pd.to_datetime(pending["signal_date"], errors="coerce").dt.date.astype(str)

    existing = _read_csv(path)
    if existing.empty:
        merged = pending
    else:
        for column in PENDING_SIGNAL_COLUMNS:
            if column not in existing:
                existing[column] = pd.NA
        merged = pd.concat([existing[PENDING_SIGNAL_COLUMNS], pending], ignore_index=True, sort=False)
    merged = merged.dropna(subset=["sleeve_id", "symbol", "signal_date"])
    merged = merged.drop_duplicates(["sleeve_id", "symbol", "signal_date"], keep="first")
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return merged


def _signal_desk_panel(signal_plans: pd.DataFrame, as_of: str) -> str:
    if signal_plans.empty:
        return ""
    priority = {"PAPER_TARGET": 0, "OBSERVE_ONLY_GATE_REMOVED": 1}
    display = signal_plans.copy()
    display["_priority"] = display["decision_status"].map(priority).fillna(2)
    display["_score"] = pd.to_numeric(display.get("score"), errors="coerce").fillna(-1)
    display = display.sort_values(["_priority", "sleeve_id", "_score"], ascending=[True, True, False]).drop_duplicates("sleeve_id")
    cards: list[str] = []
    for row in display.to_dict("records"):
        target = row.get("decision_status") == "PAPER_TARGET"
        status_text = "PAPER TARGET" if target else "QUAN SÁT - GATE CHẶN"
        status_class = "win" if target else "blocked"
        zone = (
            f"{_format_cell('entry_price', row.get('entry_zone_low_vnd'))} - "
            f"{_format_cell('entry_price', row.get('entry_zone_high_vnd'))}"
        )
        stop = _format_cell("stop_loss_price", row.get("stop_loss_vnd")) or "Theo exit model"
        take = _format_cell("take_profit_price", row.get("take_profit_vnd")) or "Theo exit model"
        cards.append(
            f"""<article class="signal-card">
  <header><div><b>{html.escape(str(row.get('symbol') or '-'))}</b><span>{html.escape(str(row.get('sleeve_id') or '-'))}</span></div><strong class="{status_class}">{status_text}</strong></header>
  <p class="signal-strategy">{html.escape(str(row.get('strategy_name') or '-'))}</p>
  <div class="signal-fields">
    <div><small>Phát tín hiệu</small><b>{html.escape(str(row.get('signal_date') or '-'))}</b></div>
    <div><small>Close tham chiếu</small><b>{html.escape(_format_cell('reference_price', row.get('reference_close_vnd')))}</b></div>
    <div><small>Vùng entry</small><b>{html.escape(zone)}</b></div>
    <div><small>SL / Kỳ vọng</small><b>{html.escape(stop)} / {html.escape(take)}</b></div>
  </div>
  <p>{html.escape(str(row.get('expectation') or ''))}</p>
  <p class="signal-gate">{html.escape(str(row.get('gate_explanation') or ''))}</p>
</article>"""
        )
    return f"""<section class="signal-panel">
  <div class="signal-head">
    <div>
      <h2>Bảng tín hiệu và kế hoạch vào lệnh</h2>
      <p>As-of {html.escape(as_of or "-")} | Một ý tưởng ưu tiên cho mỗi sleeve đang chạy | Giá hiển thị theo VND</p>
    </div>
    <a href="17_signal_plans.html">Xem toàn bộ tín hiệu</a>
  </div>
  <p class="signal-note">Vùng entry là dải tham chiếu quanh giá đóng cửa ngày phát tín hiệu. Với paper realtime, target hợp lệ có thể được fill tự động bằng quote live đầu tiên từ phiên kế tiếp; không gửi lệnh broker. Các dòng bị chặn không tạo lệnh.</p>
  <div class="signal-grid">{''.join(cards)}</div>
</section>"""


def _frame_to_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<table class="data-table"><thead><tr><th>status</th></tr></thead><tbody><tr><td>No rows</td></tr></tbody></table>'
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    body_rows: list[str] = []
    for row in frame.to_dict("records"):
        cells = []
        for column in frame.columns:
            raw = row.get(column)
            klass = _cell_class(str(column), raw)
            class_attr = f' class="{klass}"' if klass else ""
            cells.append(f"<td{class_attr}>{html.escape(_format_cell(str(column), raw))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<table class="data-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def _write_table_page(path: Path, title: str, frame: pd.DataFrame, note: str) -> None:
    display = frame.copy()
    if len(display) > 300:
        display = display.head(300)
        note = f"{note} Showing first 300 rows only; CSV contains full export.".strip()
    table = _frame_to_html(display)
    content = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ --ink: #14213d; --muted: #5f6b7a; --line: #d9dee7; --bg: #f5f1e8; --panel: #fffaf0; }}
    body {{ margin: 0; padding: 28px; font-family: Georgia, 'Times New Roman', serif; background: var(--bg); color: var(--ink); }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .note {{ margin: 0 0 18px; color: var(--muted); font-family: Segoe UI, Arial, sans-serif; }}
    .wrap {{ overflow: auto; background: white; border: 1px solid var(--line); box-shadow: 0 14px 36px rgba(20,33,61,.12); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; white-space: nowrap; font-family: Segoe UI, Arial, sans-serif; }}
    th {{ position: sticky; top: 0; background: var(--ink); color: white; text-align: left; cursor: pointer; user-select: none; }}
    th::after {{ content: " <>"; color: rgba(255,255,255,.55); font-weight: 400; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf0f5; }}
    tr:nth-child(even) {{ background: #fbfbf8; }}
    .win {{ color: #0f8b4c; font-weight: 700; }}
    .loss {{ color: #c63d2f; font-weight: 700; }}
    .flat {{ color: var(--muted); font-weight: 700; }}
    a {{ color: var(--ink); }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="note"><a href="index.html">Index</a> | {html.escape(note)}</p>
  <div class="wrap">{table}</div>
  <script src="sortable_tables.js"></script>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def _open_pnl_summary(pnl_overview: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "open_positions",
        "open_cost_value",
        "open_market_value",
        "unrealized_pnl",
        "unrealized_pnl_pct",
        "paper_equity",
        "paper_cash",
    ]
    if pnl_overview.empty:
        return pd.DataFrame(columns=columns)

    numeric_cols = [
        "open_positions",
        "open_cost_value",
        "open_market_value",
        "unrealized_pnl",
        "paper_equity",
        "paper_cash",
    ]
    work = pnl_overview.copy()
    for column in numeric_cols:
        if column in work:
            work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
        else:
            work[column] = 0.0

    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        cost = float(row.get("open_cost_value", 0.0) or 0.0)
        pnl = float(row.get("unrealized_pnl", 0.0) or 0.0)
        rows.append(
            {
                "scope": row.get("sleeve_id"),
                "open_positions": row.get("open_positions", 0),
                "open_cost_value": cost,
                "open_market_value": row.get("open_market_value", 0.0),
                "unrealized_pnl": pnl,
                "unrealized_pnl_pct": (pnl / cost * 100.0) if cost else 0.0,
                "paper_equity": row.get("paper_equity", 0.0),
                "paper_cash": row.get("paper_cash", 0.0),
            }
        )

    total_cost = float(work["open_cost_value"].sum())
    total_pnl = float(work["unrealized_pnl"].sum())
    rows.insert(
        0,
        {
            "scope": "TOTAL_OPEN_HOLDINGS",
            "open_positions": int(work["open_positions"].sum()),
            "open_cost_value": total_cost,
            "open_market_value": float(work["open_market_value"].sum()),
            "unrealized_pnl": total_pnl,
            "unrealized_pnl_pct": (total_pnl / total_cost * 100.0) if total_cost else 0.0,
            "paper_equity": float(work["paper_equity"].sum()),
            "paper_cash": float(work["paper_cash"].sum()),
        },
    )
    return pd.DataFrame(rows, columns=columns)


def _mvp_summary(sleeve_id: str, label: str, path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    status = _read_json(path / "production_status.json")
    market = _read_json(path / "market_snapshot.json")
    actions = _read_csv(path / "candidate_actions.csv")
    signals = _read_csv(path / "production_signals.csv")
    buys = actions[actions.get("action", pd.Series(dtype=str)).astype(str).str.startswith("PAPER_BUY_CANDIDATE")] if not actions.empty and "action" in actions else pd.DataFrame()
    row = {
        "sleeve_id": sleeve_id,
        "label": label,
        "logic": status.get("optimized_contract") or "Core MVP shared-pool",
        "as_of": status.get("as_of_date"),
        "mode": status.get("mode"),
        "market_state": market.get("mkt_regime_state"),
        "market_score": market.get("mkt_regime_score"),
        "max_positions": (status.get("active_sleeve") or {}).get("max_positions"),
        "max_edge_rank": status.get("max_edge_rank"),
        "sizing_mode": status.get("sizing_mode"),
        "candidate_count": status.get("candidate_count"),
        "target_count": len(buys),
        "target_symbols": ",".join(buys.get("symbol", pd.Series(dtype=str)).dropna().astype(str).head(20).tolist()),
        "dashboard_dir": str(path),
    }
    export = actions.copy()
    if export.empty:
        export = signals.copy()
    if not export.empty:
        export.insert(0, "sleeve_id", sleeve_id)
    return row, export


def _flow_summary(path: Path, default_sleeve_id: str = "flow_v2", default_label: str = "Flow V2 Rotation") -> tuple[dict[str, Any], pd.DataFrame]:
    status = _read_json(path / "flow_v2_status.json")
    market = _read_json(path / "market_snapshot.json")
    targets = _read_csv(path / "flow_v2_target_plan.csv")
    candidates = _read_csv(path / "flow_v2_candidates.csv")
    sleeve_id = str(status.get("sleeve_id") or default_sleeve_id)
    label = str(status.get("sleeve_label") or default_label)
    if sleeve_id == "flow_v2_tiered":
        label = "Early-exit hai tầng top2"
    row = {
        "sleeve_id": sleeve_id,
        "label": label,
        "logic": status.get("logic_label") or "Flow Money V2 high-RS flow-heavy rotation",
        "as_of": status.get("as_of_date"),
        "mode": status.get("mode"),
        "market_state": market.get("mkt_regime_state"),
        "market_score": market.get("mkt_regime_score"),
        "max_positions": status.get("positions"),
        "candidate_count": status.get("candidate_count"),
        "target_count": status.get("target_count"),
        "target_symbols": ",".join(status.get("target_symbols") or []),
        "dashboard_dir": str(path),
    }
    export = targets.copy()
    if export.empty:
        export = candidates.copy()
    if not export.empty:
        export.insert(0, "sleeve_id", sleeve_id)
    return row, export


def _hostile_summary(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    status = _read_json(path / "hostile_status.json")
    market = _read_json(path / "market_snapshot.json")
    targets = _read_csv(path / "hostile_target_plan.csv")
    candidates = _read_csv(path / "hostile_candidates.csv")
    sleeve_id = str(status.get("sleeve_id") or "hostile_combo_long")
    row = {
        "sleeve_id": sleeve_id,
        "label": status.get("sleeve_label") or "Hostile Combo Long p1",
        "logic": status.get("logic_label") or "Top-1 hostile-market leadership rotation",
        "as_of": status.get("as_of_date"),
        "mode": status.get("mode"),
        "market_state": market.get("mkt_regime_state"),
        "market_score": market.get("mkt_regime_score"),
        "max_positions": status.get("positions"),
        "max_edge_rank": "",
        "sizing_mode": "compound_equity_top1",
        "candidate_count": status.get("candidate_count"),
        "target_count": status.get("target_count"),
        "target_symbols": ",".join(status.get("target_symbols") or []),
        "dashboard_dir": str(path),
    }
    export = targets.copy()
    if export.empty:
        export = candidates.copy()
    if not export.empty and "sleeve_id" not in export:
        export.insert(0, "sleeve_id", sleeve_id)
    return row, export


def _flatten_json_rows(sleeve_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        rows.append({"sleeve_id": sleeve_id, "key": key, "value": value})
    return rows


def _empty_open_holdings() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sleeve_id",
            "symbol",
            "strategy_name",
            "entry_date",
            "entry_price",
            "shares",
            "cost_value",
            "market_price",
            "market_value",
            "unrealized_pnl",
            "unrealized_pnl_pct",
            "days_held",
            "stop_loss",
            "take_profit",
            "status",
        ]
    )


def _empty_closed_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sleeve_id",
            "symbol",
            "strategy_name",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "shares",
            "gross_pnl",
            "net_pnl",
            "pnl_pct",
            "days_held",
            "exit_reason",
        ]
    )


def _load_mvp_tables(sleeve_id: str, path: Path) -> dict[str, pd.DataFrame]:
    tables = {
        "actions": _read_csv(path / "candidate_actions.csv"),
        "signals": _read_csv(path / "production_signals.csv"),
        "votes": _read_csv(path / "strategy_votes.csv"),
        "rolling_summary": _read_csv(path / "rolling_summary_14d.csv"),
        "status": pd.DataFrame(_flatten_json_rows(sleeve_id, _read_json(path / "production_status.json"))),
        "market": pd.DataFrame(_flatten_json_rows(sleeve_id, _read_json(path / "market_snapshot.json"))),
    }
    for frame in tables.values():
        if not frame.empty and "sleeve_id" not in frame.columns:
            frame.insert(0, "sleeve_id", sleeve_id)
    return tables


def _load_flow_tables(path: Path, default_sleeve_id: str = "flow_v2") -> dict[str, pd.DataFrame]:
    status_payload = _read_json(path / "flow_v2_status.json")
    sleeve_id = str(status_payload.get("sleeve_id") or default_sleeve_id)
    tables = {
        "actions": _read_csv(path / "flow_v2_target_plan.csv"),
        "signals": _read_csv(path / "flow_v2_candidates.csv"),
        "votes": pd.DataFrame(),
        "rolling_summary": pd.DataFrame(),
        "status": pd.DataFrame(_flatten_json_rows(sleeve_id, status_payload)),
        "market": pd.DataFrame(_flatten_json_rows(sleeve_id, _read_json(path / "market_snapshot.json"))),
    }
    for frame in tables.values():
        if not frame.empty and "sleeve_id" not in frame.columns:
            frame.insert(0, "sleeve_id", sleeve_id)
    return tables


def _load_hostile_tables(path: Path) -> dict[str, pd.DataFrame]:
    status_payload = _read_json(path / "hostile_status.json")
    sleeve_id = str(status_payload.get("sleeve_id") or "hostile_combo_long")
    tables = {
        "actions": _read_csv(path / "hostile_target_plan.csv"),
        "signals": _read_csv(path / "hostile_candidates.csv"),
        "votes": pd.DataFrame(),
        "rolling_summary": pd.DataFrame(),
        "status": pd.DataFrame(_flatten_json_rows(sleeve_id, status_payload)),
        "market": pd.DataFrame(_flatten_json_rows(sleeve_id, _read_json(path / "market_snapshot.json"))),
    }
    for frame in tables.values():
        if not frame.empty and "sleeve_id" not in frame.columns:
            frame.insert(0, "sleeve_id", sleeve_id)
    return tables


def export_dashboard(
    out_dir: Path,
    mvp_p5_dir: Path | None,
    mvp_p4_dir: Path | None,
    flow_dir: Path,
    flow_baseline_dir: Path | None = None,
    flow_tiered_dir: Path | None = None,
    hostile_dir: Path | None = None,
    core_rank2_dir: Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_meta = _read_json(out_dir / "paper_ledger_meta.json")
    paper_started_at = str(ledger_meta.get("started_at") or "-")
    rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    sleeve_sources = [
        _flow_summary(flow_dir),
    ]
    if core_rank2_dir is not None and (core_rank2_dir / "production_status.json").exists():
        sleeve_sources.append(_mvp_summary("core_mvp9_rank2", "Core MVP9 p2 rank2 compound", core_rank2_dir))
    if flow_baseline_dir is not None and (flow_baseline_dir / "flow_v2_status.json").exists():
        sleeve_sources.append(_flow_summary(flow_baseline_dir, "flow_v2_baseline", "Baseline fresh-signal top2"))
    if flow_tiered_dir is not None and (flow_tiered_dir / "flow_v2_status.json").exists():
        sleeve_sources.append(_flow_summary(flow_tiered_dir, "flow_v2_tiered", "Early-exit hai tầng top2"))
    if hostile_dir is not None and (hostile_dir / "hostile_status.json").exists():
        sleeve_sources.append(_hostile_summary(hostile_dir))
    for row, frame in sleeve_sources:
        rows.append(row)
        if not frame.empty:
            detail_frames.append(frame)

    summary = pd.DataFrame(rows)
    active_sleeves = set(summary["sleeve_id"].astype(str)) if "sleeve_id" in summary else set()
    details = pd.concat(detail_frames, ignore_index=True, sort=False) if detail_frames else pd.DataFrame()
    summary.to_csv(out_dir / "combined_sleeve_summary.csv", index=False, encoding="utf-8-sig")
    details.to_csv(out_dir / "combined_actions.csv", index=False, encoding="utf-8-sig")

    flow_tables = _load_flow_tables(flow_dir)
    all_tables = [flow_tables]
    if core_rank2_dir is not None and (core_rank2_dir / "production_status.json").exists():
        all_tables.append(_load_mvp_tables("core_mvp9_rank2", core_rank2_dir))
    if flow_baseline_dir is not None and (flow_baseline_dir / "flow_v2_status.json").exists():
        all_tables.append(_load_flow_tables(flow_baseline_dir, "flow_v2_baseline"))
    if flow_tiered_dir is not None and (flow_tiered_dir / "flow_v2_status.json").exists():
        all_tables.append(_load_flow_tables(flow_tiered_dir, "flow_v2_tiered"))
    if hostile_dir is not None and (hostile_dir / "hostile_status.json").exists():
        all_tables.append(_load_hostile_tables(hostile_dir))

    def concat_table(name: str) -> pd.DataFrame:
        frames = [tables[name] for tables in all_tables if not tables[name].empty]
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    realtime_status = _read_json(out_dir / "paper_realtime_status.json")
    pnl_overview = _read_csv(out_dir / "paper_realtime_account_pnl.csv")
    pnl_open_holdings = _read_csv(out_dir / "paper_realtime_open_holdings.csv")
    strategy_pnl = _read_csv(out_dir / "paper_realtime_strategy_pnl.csv")
    using_realtime_overlay = bool(realtime_status) or not pnl_overview.empty or not pnl_open_holdings.empty or not strategy_pnl.empty
    if pnl_overview.empty:
        pnl_overview = _read_csv(out_dir / "paper_account_pnl.csv")
    else:
        base_overview = _read_csv(out_dir / "paper_account_pnl.csv")
        if not base_overview.empty and "sleeve_id" in base_overview and "sleeve_id" in pnl_overview:
            missing = base_overview.loc[~base_overview["sleeve_id"].astype(str).isin(set(pnl_overview["sleeve_id"].astype(str)))].copy()
            if not missing.empty:
                pnl_overview = pd.concat([pnl_overview, missing], ignore_index=True, sort=False)
    if pnl_open_holdings.empty:
        pnl_open_holdings = _read_csv(out_dir / "paper_open_holdings.csv")
    if strategy_pnl.empty:
        strategy_pnl = _read_csv(out_dir / "paper_strategy_pnl.csv")
    if active_sleeves:
        if not pnl_overview.empty and "sleeve_id" in pnl_overview:
            pnl_overview = pnl_overview[pnl_overview["sleeve_id"].astype(str).isin(active_sleeves)].copy()
        if not pnl_open_holdings.empty and "sleeve_id" in pnl_open_holdings:
            pnl_open_holdings = pnl_open_holdings[pnl_open_holdings["sleeve_id"].astype(str).isin(active_sleeves)].copy()
        if not strategy_pnl.empty and "sleeve_id" in strategy_pnl:
            strategy_pnl = strategy_pnl[strategy_pnl["sleeve_id"].astype(str).isin(active_sleeves)].copy()
    pnl_closed_trades = _read_csv(out_dir / "paper_closed_trades.csv")
    ledger_events = _read_csv(out_dir / "paper_ledger_events.csv")
    backtest_comparison = _read_csv(out_dir / "strategy_backtest_comparison.csv")
    if active_sleeves:
        if not pnl_closed_trades.empty and "sleeve_id" in pnl_closed_trades:
            pnl_closed_trades = pnl_closed_trades[pnl_closed_trades["sleeve_id"].astype(str).isin(active_sleeves)].copy()
        if not ledger_events.empty and "sleeve_id" in ledger_events:
            ledger_events = ledger_events[ledger_events["sleeve_id"].astype(str).isin(active_sleeves)].copy()
        if not backtest_comparison.empty and "sleeve_id" in backtest_comparison:
            backtest_comparison = backtest_comparison[backtest_comparison["sleeve_id"].astype(str).isin(active_sleeves)].copy()
    money_flow_pulse = _read_json(flow_dir / "money_flow_pulse.json")
    money_flow_today = _read_csv(flow_dir / "flow_v2_money_flow_today.csv")
    money_flow_recent = _read_csv(flow_dir / "flow_v2_money_flow_recent.csv")
    realtime_money_flow_status = _read_json(out_dir / "realtime_money_flow_status.json")
    realtime_money_flow = _read_csv(out_dir / "20_realtime_money_flow.csv")
    core_ungated_discovery = (
        _read_csv(core_rank2_dir / "ungated_discovery_candidates.csv")
        if core_rank2_dir is not None
        else pd.DataFrame()
    )
    if not core_ungated_discovery.empty:
        core_ungated_discovery["sleeves"] = "core_mvp9_rank2"
    flow_ungated_discovery = _read_csv(flow_dir / "ungated_discovery_candidates.csv")
    discovery_frames = [frame for frame in [core_ungated_discovery, flow_ungated_discovery] if not frame.empty]
    ungated_discovery = (
        pd.concat(discovery_frames, ignore_index=True, sort=False)
        if discovery_frames
        else pd.DataFrame()
    )
    open_pnl_summary = _open_pnl_summary(pnl_overview)
    open_holdings = pnl_open_holdings if not pnl_open_holdings.empty else _empty_open_holdings()
    closed_trades = pnl_closed_trades if not pnl_closed_trades.empty else _empty_closed_trades()
    orders_targets = concat_table("actions")
    candidates = concat_table("signals")
    signal_plans = _build_signal_plans(orders_targets, candidates, ungated_discovery)
    if active_sleeves and not signal_plans.empty and "sleeve_id" in signal_plans:
        signal_plans = signal_plans[signal_plans["sleeve_id"].astype(str).isin(active_sleeves)].copy()
    pending_signals = _append_pending_signal_orders(out_dir, signal_plans, orders_targets)
    votes = concat_table("votes")
    status_rows = pd.concat(
        [concat_table("status"), concat_table("market")],
        ignore_index=True,
        sort=False,
    )
    rolling_summary = concat_table("rolling_summary")
    account_overview = summary[
        [
            "sleeve_id",
            "label",
            "logic",
            "as_of",
            "mode",
            "market_state",
            "market_score",
            "max_positions",
            "max_edge_rank",
            "sizing_mode",
            "candidate_count",
            "target_count",
            "target_symbols",
        ]
    ].copy()
    if not pnl_overview.empty:
        account_overview = account_overview.merge(pnl_overview, on="sleeve_id", how="left")
        if using_realtime_overlay:
            account_overview["note"] = f"Realtime price overlay at {realtime_status.get('priced_at', '-')}; ledger exits still run on daily OHLCV."
        else:
            account_overview["note"] = "PnL uses paper ledger: signal close T, fill next open T+1, mark latest close."
    else:
        account_overview["paper_equity"] = None
        account_overview["paper_cash"] = None
        account_overview["open_positions"] = 0
        account_overview["closed_trades"] = 0
        account_overview["realized_pnl"] = 0.0
        account_overview["unrealized_pnl"] = 0.0
        account_overview["note"] = "Dry-run signals only until first paper fill is recorded."

    account_overview.to_csv(out_dir / "00_account_overview.csv", index=False, encoding="utf-8-sig")
    open_holdings.to_csv(out_dir / "03_open_holdings.csv", index=False, encoding="utf-8-sig")
    closed_trades.to_csv(out_dir / "04_closed_trades.csv", index=False, encoding="utf-8-sig")
    orders_targets.to_csv(out_dir / "05_orders_targets.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(out_dir / "06_candidates.csv", index=False, encoding="utf-8-sig")
    votes.to_csv(out_dir / "07_strategy_votes.csv", index=False, encoding="utf-8-sig")
    status_rows.to_csv(out_dir / "08_status_config.csv", index=False, encoding="utf-8-sig")
    rolling_summary.to_csv(out_dir / "09_rolling_summary.csv", index=False, encoding="utf-8-sig")
    strategy_pnl.to_csv(out_dir / "10_strategy_pnl.csv", index=False, encoding="utf-8-sig")
    open_pnl_summary.to_csv(out_dir / "11_open_pnl_summary.csv", index=False, encoding="utf-8-sig")
    ledger_events.to_csv(out_dir / "12_ledger_events.csv", index=False, encoding="utf-8-sig")
    backtest_comparison.to_csv(out_dir / "13_backtest_comparison.csv", index=False, encoding="utf-8-sig")
    money_flow_today.to_csv(out_dir / "14_money_flow_today.csv", index=False, encoding="utf-8-sig")
    money_flow_recent.to_csv(out_dir / "15_money_flow_recent.csv", index=False, encoding="utf-8-sig")
    ungated_discovery.to_csv(out_dir / "16_ungated_discovery.csv", index=False, encoding="utf-8-sig")
    signal_plans.to_csv(out_dir / "17_signal_plans.csv", index=False, encoding="utf-8-sig")
    pending_signals.to_csv(out_dir / "18_paper_signal_pending.csv", index=False, encoding="utf-8-sig")
    _, _, contract_audit, contract_audit_summary = write_audit(out_dir)

    _write_table_page(out_dir / "00_account_overview.html", "00 Account Overview", account_overview, "Paper account state by sleeve.")
    _write_table_page(out_dir / "01_sleeve_summary.html", "01 Sleeve Summary", summary, "Independent paper strategy sleeves.")
    _write_table_page(out_dir / "02_combined_actions.html", "02 Combined Actions", details, "Paper actions/candidates from all sleeves.")
    _write_table_page(out_dir / "03_open_holdings.html", "03 Open Holdings", open_holdings, "Open paper positions by sleeve. Empty until first fill.")
    _write_table_page(out_dir / "04_closed_trades.html", "04 Closed Trades", closed_trades, "Closed paper trades by sleeve. Empty until first exit.")
    _write_table_page(out_dir / "05_orders_targets.html", "05 Orders & Targets", orders_targets, "Paper buy targets/orders emitted by each sleeve.")
    _write_table_page(out_dir / "06_candidates.html", "06 Candidates", candidates, "Ranked candidates/signals before portfolio allocation.")
    _write_table_page(out_dir / "07_strategy_votes.html", "07 Strategy Votes", votes, "MVP strategy-level votes. Flow V2 uses a single ranked score.")
    _write_table_page(out_dir / "08_status_config.html", "08 Status & Config", status_rows, "Run status, market snapshot and safety controls.")
    _write_table_page(out_dir / "09_rolling_summary.html", "09 Rolling Summary", rolling_summary, "Recent MVP signal history when available.")
    _write_table_page(out_dir / "10_strategy_pnl.html", "10 Strategy PnL", strategy_pnl, "PnL grouped by sleeve and strategy.")
    _write_table_page(out_dir / "11_open_pnl_summary.html", "11 Open PnL Summary", open_pnl_summary, "Current unrealized PnL for all open paper holdings.")
    _write_table_page(out_dir / "12_ledger_events.html", "12 Ledger Events", ledger_events, "Audit log for paper entries, stop updates and exits.")
    _write_table_page(out_dir / "13_backtest_comparison.html", "13 Backtest Comparison", backtest_comparison, "VN100 comparisons only; unverified equivalents are explicitly marked.")
    _write_table_page(out_dir / "14_money_flow_today.html", "14 Tiền Vào Hôm Nay", money_flow_today, "Flow V2 observable inflow proxy on the latest session; context only, not an order list.")
    _write_table_page(out_dir / "15_money_flow_recent.html", "15 Dòng Tiền Mạnh 5 Phiên", money_flow_recent, "Flow V2 observable inflow proxy seen on at least two of the latest five sessions.")
    _write_table_page(
        out_dir / "16_ungated_discovery.html",
        "16 Cơ Hội Quan Sát Bỏ Market Gate",
        ungated_discovery,
        "Display-only expansion: paper trading targets and gates are unchanged.",
    )
    _write_table_page(
        out_dir / "17_signal_plans.html",
        "17 Bảng Tín Hiệu Và Kế Hoạch Vào Lệnh",
        signal_plans,
        "Decision display only: entry zones are indicative around signal-day close; paper fills remain governed by the strategy contract.",
    )
    _write_table_page(
        out_dir / "18_paper_signal_pending.html",
        "18 Tin Hieu Paper Dang Cho T+1",
        pending_signals,
        "Persistent paper-entry signal ledger. Realtime entry consumes this so a regenerated pipeline does not erase T+1 orders.",
    )
    _write_table_page(
        out_dir / "19_current_paper_contract_audit.html",
        "19 Current Paper Contract Audit",
        contract_audit,
        "Read-only baseline audit. Strategy optimization should wait until active contracts are verified or explicitly accepted as research-only.",
    )
    _write_table_page(
        out_dir / "20_realtime_money_flow.html",
        "20 Xác Nhận Dòng Tiền Realtime",
        realtime_money_flow,
        "Display only: EOD Flow V2 watchlist enriched with intraday price-board and foreign-volume confirmation. It does not change gate or paper orders.",
    )
    for row in rows:
        sleeve_id = str(row["sleeve_id"])
        sleeve_holdings = open_holdings.loc[open_holdings["sleeve_id"].astype(str) == sleeve_id].copy() if not open_holdings.empty and "sleeve_id" in open_holdings else _empty_open_holdings()
        _write_table_page(
            out_dir / f"open_holdings_{sleeve_id}.html",
            f"Open Holdings - {row['label']}",
            sleeve_holdings,
            f"Open paper holdings for {row['label']}. Fresh paper run activation date: {paper_started_at}.",
        )
    (out_dir / "sortable_tables.js").write_text(SORTABLE_JS, encoding="utf-8")

    pnl_by_sleeve = {}
    if not pnl_overview.empty:
        pnl_by_sleeve = {str(row.get("sleeve_id")): row for row in pnl_overview.to_dict("records")}
    total_open_pnl = open_pnl_summary.iloc[0].to_dict() if not open_pnl_summary.empty else {}
    total_pnl_value = total_open_pnl.get("unrealized_pnl", 0.0)
    total_pnl_pct = total_open_pnl.get("unrealized_pnl_pct", 0.0)
    total_pnl_class = _cell_class("unrealized_pnl", total_pnl_value)
    audit_ready = bool(contract_audit_summary.get("ready_for_strategy_optimization"))
    audit_status = "READY" if audit_ready else "NEEDS BASELINE RERUN"
    audit_class = "win" if audit_ready else "loss"
    generated_at = max([str(item.get("as_of") or "") for item in rows] or [""])
    money_flow_html = _money_flow_panel(money_flow_pulse, money_flow_today, money_flow_recent)
    realtime_money_flow_html = _realtime_money_flow_panel(realtime_money_flow_status, realtime_money_flow)
    discovery_date = (
        str(ungated_discovery.iloc[0].get("date") or generated_at)
        if not ungated_discovery.empty
        else generated_at
    )
    ungated_discovery_html = _ungated_discovery_panel(
        core_ungated_discovery,
        flow_ungated_discovery,
        discovery_date,
    )
    signal_desk_html = _signal_desk_panel(signal_plans, generated_at)

    cards = []
    for row in rows:
        state = str(row.get("market_state") or "UNKNOWN")
        state_class = "win" if state == "RISK_ON" else ("loss" if state == "RISK_OFF" else "flat")
        symbols = row.get("target_symbols") or "Cash"
        sleeve_pnl = pnl_by_sleeve.get(str(row.get("sleeve_id")), {})
        sleeve_pnl_value = sleeve_pnl.get("unrealized_pnl", 0.0)
        sleeve_pnl_pct = sleeve_pnl.get("unrealized_pnl_pct", 0.0)
        sleeve_pnl_class = _cell_class("unrealized_pnl", sleeve_pnl_value)
        flow_pulse_link = ""
        if str(row.get("sleeve_id")) == "flow_v2":
            flow_pulse_link = '<a class="holding-link" href="flow_v2/index.html">Dòng tiền quan sát Flow V2</a>'
        cards.append(
            f"""<div class="sleeve">
  <div class="sleeve-top"><strong>{html.escape(str(row["label"]))}</strong><span>{html.escape(str(row["logic"]))}</span></div>
  <div class="metric-row"><span>Market</span><b class="{state_class}">{html.escape(state)}</b></div>
  <div class="metric-row"><span>Open PnL</span><b class="{sleeve_pnl_class}">{html.escape(_format_cell("unrealized_pnl", sleeve_pnl_value))} ({html.escape(_format_cell("unrealized_pnl_pct", sleeve_pnl_pct))})</b></div>
  <div class="metric-row"><span>Score</span><b>{html.escape(_fmt(row.get("market_score")))}</b></div>
  <div class="metric-row"><span>Max positions</span><b>{html.escape(_fmt(row.get("max_positions")))}</b></div>
  <div class="metric-row"><span>Candidates</span><b>{html.escape(_fmt(row.get("candidate_count")))}</b></div>
  <div class="metric-row"><span>Targets</span><b>{html.escape(_fmt(symbols))}</b></div>
  <a class="holding-link" href="open_holdings_{html.escape(str(row['sleeve_id']))}.html">Open holdings riêng</a>
  {flow_pulse_link}
</div>"""
        )

    price_line = ""
    if using_realtime_overlay:
        price_line = (
            f" | Realtime prices: {html.escape(str(realtime_status.get('updated_holdings', realtime_status.get('updated', 0))))} holdings / "
            f"{html.escape(str(realtime_status.get('symbols', 0)))} symbols at "
            f"{html.escape(str(realtime_status.get('priced_at', '-')))}"
        )
    refresh_seconds = int(_to_number(realtime_status.get("html_refresh_seconds")) or 0) if using_realtime_overlay else 0
    refresh_meta = f'\n  <meta http-equiv="refresh" content="{refresh_seconds}">' if refresh_seconds > 0 else ""
    index = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  {refresh_meta}
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Combined Paper Trading Dashboard</title>
  <style>
    :root {{ --ink: #10243e; --gold: #b8872f; --paper: #f5f1e8; --card: #fffaf0; --muted: #667085; --win: #0f8b4c; --loss: #c63d2f; }}
    body {{ margin: 0; min-height: 100vh; background: radial-gradient(circle at top left, rgba(184,135,47,.24), transparent 32rem), linear-gradient(135deg, #f7f0df 0%, #eef3f6 100%); color: var(--ink); font-family: Georgia, 'Times New Roman', serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 44px 26px; }}
    h1 {{ font-size: 42px; margin: 0 0 8px; }}
    .sub {{ color: var(--muted); font-family: Segoe UI, Arial, sans-serif; font-size: 16px; margin-bottom: 24px; }}
    .flow {{ padding: 18px 20px; background: rgba(255,255,255,.68); border: 1px solid rgba(16,36,62,.12); border-radius: 18px; font-family: Consolas, monospace; line-height: 1.55; white-space: pre-wrap; box-shadow: 0 18px 48px rgba(16,36,62,.12); margin-bottom: 24px; }}
    .pnl-total {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .pnl-box {{ background: rgba(255,255,255,.76); border: 1px solid rgba(16,36,62,.14); border-radius: 8px; padding: 16px; font-family: Segoe UI, Arial, sans-serif; }}
    .pnl-box span {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 7px; }}
    .pnl-box strong {{ display: block; font-size: 24px; overflow-wrap: anywhere; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 14px; margin-bottom: 18px; }}
    .sleeve {{ background: var(--card); border: 1px solid rgba(16,36,62,.14); border-radius: 8px; padding: 18px; box-shadow: 0 10px 26px rgba(16,36,62,.08); }}
    .sleeve-top strong {{ display: block; font-size: 19px; margin-bottom: 5px; }}
    .sleeve-top span {{ color: var(--muted); font-family: Segoe UI, Arial, sans-serif; font-size: 13px; }}
    .metric-row {{ display: flex; justify-content: space-between; gap: 12px; padding: 9px 0; border-bottom: 1px solid rgba(16,36,62,.1); font-family: Segoe UI, Arial, sans-serif; }}
    .metric-row span {{ color: var(--muted); }}
    .metric-row b {{ text-align: right; overflow-wrap: anywhere; }}
    .holding-link {{ display: inline-block; margin-top: 12px; color: #175f58; font: 600 13px Segoe UI, Arial, sans-serif; text-decoration: none; }}
    .notice {{ margin: 0 0 20px; padding: 15px 18px; border: 1px solid #b8d8d1; background: #eef8f5; border-radius: 8px; font: 15px Segoe UI, Arial, sans-serif; }}
    h2 {{ margin: 26px 0 12px; font-size: 24px; }}
    .comparison {{ margin-bottom: 22px; overflow: auto; background: rgba(255,255,255,.76); border: 1px solid rgba(16,36,62,.14); border-radius: 8px; padding: 12px; }}
    .comparison table {{ width: 100%; border-collapse: collapse; font: 13px Segoe UI, Arial, sans-serif; white-space: nowrap; }}
    .comparison th, .comparison td {{ padding: 9px; border-bottom: 1px solid rgba(16,36,62,.1); text-align: left; }}
    .comparison th {{ color: var(--muted); }}
    .comparison-note {{ margin: -3px 0 12px; padding: 11px 13px; background: #fff4da; border-left: 4px solid #b8872f; color: #624716; font: 13px Segoe UI, Arial, sans-serif; }}
    .money-panel {{ margin: 22px 0; padding: 20px; background: rgba(255,255,255,.78); border: 1px solid rgba(16,36,62,.14); border-radius: 16px; box-shadow: 0 12px 30px rgba(16,36,62,.08); font-family: Segoe UI, Arial, sans-serif; }}
    .money-head {{ display: flex; justify-content: space-between; align-items: start; gap: 12px; }}
    .money-head h2 {{ margin: 0 0 5px; font-family: Georgia, 'Times New Roman', serif; }}
    .money-head p, .money-note, .money-block p {{ color: var(--muted); font-size: 13px; margin: 0; }}
    .money-pill {{ border: 1px solid currentColor; border-radius: 999px; padding: 8px 12px; white-space: nowrap; font-size: 13px; }}
    .money-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 9px; margin: 18px 0; }}
    .money-metrics div {{ padding: 10px; border: 1px solid rgba(16,36,62,.1); border-radius: 8px; background: #fff; }}
    .money-metrics span {{ display: block; color: var(--muted); font-size: 12px; }}
    .money-metrics strong {{ display: block; margin-top: 4px; font-size: 18px; }}
    .money-sections {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 15px; }}
    .money-block {{ padding: 14px; border: 1px solid rgba(16,36,62,.1); border-radius: 10px; background: rgba(255,255,255,.56); }}
    .money-block h3 {{ margin: 0 0 5px; font-size: 17px; }}
    .money-block p {{ margin-bottom: 12px; min-height: 32px; }}
    .money-block a {{ display: inline-block; margin-top: 12px; color: #175f58; font-weight: 600; font-size: 13px; }}
    .money-row {{ display: grid; grid-template-columns: 42px minmax(75px, 1fr) 42px; gap: 8px; align-items: center; font-size: 13px; margin-bottom: 7px; }}
    .money-row strong {{ text-align: right; }}
    .money-row span {{ grid-column: 2 / 4; margin-top: -6px; color: var(--muted); font-size: 12px; }}
    .money-track {{ background: #e9eef2; height: 14px; border-radius: 999px; overflow: hidden; }}
    .money-bar {{ display: block; height: 100%; background: #8c98a7; border-radius: 999px; }}
    .money-bar.win {{ background: var(--win); }}
    .money-bar.loss {{ background: var(--loss); }}
    .money-note {{ border-top: 1px solid rgba(16,36,62,.1); padding-top: 12px; margin-top: 15px; }}
    .live-flow-panel {{ margin: 22px 0; padding: 20px; background: #f7fcfb; border: 1px solid rgba(15,139,76,.26); border-radius: 16px; font-family: Segoe UI, Arial, sans-serif; }}
    .live-flow-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(245px, 1fr)); gap: 9px; margin: 14px 0; }}
    .live-flow-row {{ display: grid; grid-template-columns: 44px 1fr; gap: 5px 9px; padding: 10px; border: 1px solid rgba(16,36,62,.1); border-radius: 8px; background: white; font-size: 13px; }}
    .live-flow-row b {{ font-size: 16px; }}
    .live-flow-row strong {{ text-align: right; }}
    .live-flow-row em {{ grid-column: 1 / 3; font-style: normal; font-weight: 600; }}
    .live-flow-row span {{ grid-column: 1 / 3; color: var(--muted); font-size: 12px; }}
    .idea-panel {{ margin: 22px 0; padding: 20px; background: #fffdf7; border: 1px solid rgba(184,135,47,.34); border-radius: 16px; font-family: Segoe UI, Arial, sans-serif; }}
    .idea-head {{ display: flex; justify-content: space-between; align-items: start; gap: 12px; }}
    .idea-head h2 {{ margin: 0 0 5px; font-family: Georgia, 'Times New Roman', serif; }}
    .idea-head p {{ color: var(--muted); font-size: 13px; margin: 0; }}
    .idea-pill {{ border: 1px solid #b8872f; color: #8a621e; background: #fff4da; border-radius: 999px; padding: 8px 12px; white-space: nowrap; font-size: 13px; }}
    .idea-warning {{ margin: 17px 0; padding: 11px 13px; background: #fff4da; border-left: 4px solid #b8872f; font-size: 13px; }}
    .idea-row {{ padding: 9px 0; border-bottom: 1px solid rgba(16,36,62,.1); font-size: 13px; }}
    .idea-row div {{ display: flex; justify-content: space-between; gap: 8px; }}
    .idea-row div strong {{ color: var(--muted); font-size: 12px; text-align: right; }}
    .idea-row em {{ display: block; color: #8a621e; font-style: normal; font-weight: 700; font-size: 12px; margin-top: 4px; }}
    .idea-row p {{ margin: 4px 0 0; min-height: 0; }}
    .idea-link {{ display: inline-block; margin-top: 15px; color: #175f58; font-weight: 600; font-size: 13px; }}
    .signal-panel {{ margin: 22px 0; padding: 20px; background: rgba(255,255,255,.84); border: 1px solid rgba(16,36,62,.16); border-radius: 16px; font-family: Segoe UI, Arial, sans-serif; }}
    .signal-head {{ display: flex; justify-content: space-between; align-items: start; gap: 14px; margin-bottom: 12px; }}
    .signal-head h2 {{ margin: 0 0 5px; font-family: Georgia, 'Times New Roman', serif; }}
    .signal-head p {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .signal-head a {{ color: #175f58; font-size: 13px; font-weight: 600; white-space: nowrap; }}
    .signal-note {{ margin: 0 0 15px; padding: 11px 13px; background: #eef4f8; color: #425466; font-size: 13px; border-left: 4px solid #426b8a; }}
    .signal-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(335px, 1fr)); gap: 12px; }}
    .signal-card {{ background: #fff; border: 1px solid rgba(16,36,62,.12); border-radius: 10px; padding: 14px; }}
    .signal-card header {{ display: flex; justify-content: space-between; align-items: start; gap: 10px; }}
    .signal-card header b {{ display: block; font-size: 21px; font-family: Georgia, 'Times New Roman', serif; }}
    .signal-card header span {{ display: block; color: var(--muted); font-size: 12px; }}
    .signal-card header strong {{ font-size: 11px; border-radius: 999px; padding: 5px 8px; border: 1px solid currentColor; white-space: nowrap; }}
    .signal-card header strong.blocked {{ color: #a46713; background: #fff4da; }}
    .signal-strategy {{ font-weight: 600; color: #34475d; margin: 10px 0; font-size: 13px; }}
    .signal-fields {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin: 10px 0; }}
    .signal-fields div {{ padding: 8px; background: #f6f7f8; border-radius: 6px; }}
    .signal-fields small {{ display: block; color: var(--muted); font-size: 11px; margin-bottom: 3px; }}
    .signal-fields b {{ font-size: 13px; }}
    .signal-card p:not(.signal-strategy) {{ margin: 8px 0 0; color: #425466; font-size: 12px; line-height: 1.45; }}
    .signal-card .signal-gate {{ color: #8a621e; }}
    @media (max-width: 780px) {{ .money-head {{ display: block; }} .money-pill {{ display: inline-block; margin-top: 10px; }} .money-sections {{ grid-template-columns: 1fr; }} }}
    .win {{ color: var(--win); }}
    .loss {{ color: var(--loss); }}
    .flat {{ color: var(--muted); }}
    .links {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }}
    .card {{ display: block; text-decoration: none; color: var(--ink); background: rgba(255,255,255,.72); border: 1px solid rgba(16,36,62,.14); border-radius: 8px; padding: 16px; font-family: Segoe UI, Arial, sans-serif; }}
    .card strong {{ display: block; margin-bottom: 5px; }}
    .card span {{ color: var(--muted); font-size: 13px; }}
  </style>
</head>
<body>
  <main>
    <h1>Combined Paper Trading Dashboard</h1>
    <div class="sub">{len(rows)} independent sleeves | As-of: {html.escape(generated_at)} | Broker orders disabled{price_line}</div>
    <div class="notice"><strong>Fresh paper run:</strong> toàn bộ open holdings cũ đã được xoá. Các sleeve bắt đầu theo dõi độc lập từ phiên <strong>{html.escape(paper_started_at)}</strong>; chỉ tín hiệu hình thành từ ngày này trở đi mới được phép mở vị thế.</div>
    <div class="flow">OHLCV parquet + feature engine
        |
        +-- Flow V2 Rotation: high-RS flow-heavy rotation, max 2 slots
        |
        +-- Core MVP9 p2 rank2 compound: Core edge strategies, max 2, edge rank <= 2
        |
        +-- Baseline fresh-signal top2: audited Flow V2 baseline, max 2 slots
        |
        +-- Early-exit hai tầng top2: audited Flow V2 tiered exit, max 2 slots
        |
        +-- Hostile Combo Long p1: top-1 leadership sleeve, rebalance 3, MA20 exit
        |
Combined paper dashboard only. No sleeve can overwrite another sleeve's logic.</div>
    <div class="pnl-total">
      <div class="pnl-box"><span>Total Open PnL</span><strong class="{total_pnl_class}">{html.escape(_format_cell("unrealized_pnl", total_pnl_value))}</strong></div>
      <div class="pnl-box"><span>Total Open PnL %</span><strong class="{total_pnl_class}">{html.escape(_format_cell("unrealized_pnl_pct", total_pnl_pct))}</strong></div>
      <div class="pnl-box"><span>Open Market Value</span><strong>{html.escape(_format_cell("open_market_value", total_open_pnl.get("open_market_value", 0.0)))}</strong></div>
      <div class="pnl-box"><span>Open Positions</span><strong>{html.escape(_format_cell("open_positions", total_open_pnl.get("open_positions", 0)))}</strong></div>
      <div class="pnl-box"><span>Contract Audit</span><strong class="{audit_class}">{audit_status}</strong></div>
    </div>
    <div class="grid">{''.join(cards)}</div>
    {signal_desk_html}
    {realtime_money_flow_html}
    {money_flow_html}
    {ungated_discovery_html}
    <h2>Current paper contract audit</h2>
    <p class="comparison-note">Verified sleeves: {contract_audit_summary.get("verified_count", 0)}. Needs exact rerun: {contract_audit_summary.get("needs_exact_rerun_count", 0)}. Failures: {contract_audit_summary.get("fail_count", 0)}. Use this as the baseline gate before strategy optimization.</p>
    <div class="comparison">{_frame_to_html(contract_audit)}</div>
    <h2>Kết quả backtest các chiến lược đang chạy</h2>
    <p class="comparison-note">Đây là so sánh backtest lịch sử của các sleeve còn hoạt động trong HTML tổng, không phải PnL paper hiện tại. Các sleeve đã deactivated không còn được tính vào bảng này.</p>
    <div class="comparison">{_frame_to_html(backtest_comparison)}</div>
    <div class="links">
      <a class="card" href="00_account_overview.html"><strong>00 Account Overview</strong><span>Portfolio state, cash/equity placeholders and sleeve status.</span></a>
      <a class="card" href="01_sleeve_summary.html"><strong>01 Sleeve Summary</strong><span>High-level state for all paper sleeves.</span></a>
      <a class="card" href="02_combined_actions.html"><strong>02 Combined Actions</strong><span>Actions and candidates emitted by each independent sleeve.</span></a>
      <a class="card" href="03_open_holdings.html"><strong>03 Open Holdings</strong><span>Open paper positions by sleeve.</span></a>
      <a class="card" href="04_closed_trades.html"><strong>04 Closed Trades</strong><span>Closed paper trades and realized PnL.</span></a>
      <a class="card" href="05_orders_targets.html"><strong>05 Orders & Targets</strong><span>Paper targets for next session.</span></a>
      <a class="card" href="06_candidates.html"><strong>06 Candidates</strong><span>Full candidate list and scores.</span></a>
      <a class="card" href="07_strategy_votes.html"><strong>07 Strategy Votes</strong><span>Per-strategy MVP vote transparency.</span></a>
      <a class="card" href="08_status_config.html"><strong>08 Status & Config</strong><span>Market snapshot, config and safety notes.</span></a>
      <a class="card" href="09_rolling_summary.html"><strong>09 Rolling Summary</strong><span>Recent signal counts and market states.</span></a>
      <a class="card" href="10_strategy_pnl.html"><strong>10 Strategy PnL</strong><span>Per-strategy paper PnL by sleeve.</span></a>
      <a class="card" href="11_open_pnl_summary.html"><strong>11 Open PnL Summary</strong><span>Total and per-sleeve unrealized PnL for open holdings.</span></a>
      <a class="card" href="12_ledger_events.html"><strong>12 Ledger Events</strong><span>Audit log for entries, stop updates and exits.</span></a>
      <a class="card" href="13_backtest_comparison.html"><strong>13 Backtest Comparison</strong><span>So sánh kết quả các sleeve hiện chạy song song.</span></a>
      <a class="card" href="14_money_flow_today.html"><strong>14 Tiền Vào Hôm Nay</strong><span>Dữ liệu dòng tiền mới nhất dùng chung trên dashboard tổng.</span></a>
      <a class="card" href="15_money_flow_recent.html"><strong>15 Dòng Tiền Mạnh 5 Phiên</strong><span>Cổ phiếu có tín hiệu lặp lại trong thời gian vừa qua.</span></a>
      <a class="card" href="16_ungated_discovery.html"><strong>16 Cơ Hội Quan Sát Bỏ Market Gate</strong><span>Mã và chiến lược được mở rộng để giải thích; không tạo lệnh paper.</span></a>
      <a class="card" href="17_signal_plans.html"><strong>17 Bảng Tín Hiệu Và Kế Hoạch Vào Lệnh</strong><span>Ngày phát tín hiệu, vùng entry tham chiếu, kỳ vọng và trạng thái gate.</span></a>
      <a class="card" href="../flow_v2_gate_comparison_vn100_2020_to_2026-05-26/index.html"><strong>18 Flow V2: Gate Và Rủi Ro</strong><span>Backtest có gate/không gate và early-exit trên cùng contract VN100.</span></a>
      <a class="card" href="../flow_v2_enhanced_review_vn100_to_2026-05-26/index.html"><strong>19 Flow V2 Enhanced: Tư Duy Và Benchmark</strong><span>Logic chiến lược, so sánh hold/VNINDEX/tiền gửi và các giới hạn cần giải thích.</span></a>
      <a class="card" href="20_realtime_money_flow.html"><strong>20 Xác Nhận Dòng Tiền Realtime</strong><span>Giá live và khối ngoại mua/bán trên watchlist Flow V2; chỉ quan sát, không tạo lệnh paper.</span></a>
    </div>
  </main>
</body>
</html>
"""
    index_path = out_dir / "index.html"
    index_path.write_text(index, encoding="utf-8")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export combined paper trading dashboard.")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "combined_paper_trading_demo"))
    parser.add_argument("--mvp-p5-dir", default="")
    parser.add_argument("--mvp-p4-dir", default="")
    parser.add_argument("--flow-dir", required=True)
    parser.add_argument("--flow-baseline-dir", default="")
    parser.add_argument("--flow-tiered-dir", default="")
    parser.add_argument("--hostile-dir", default="")
    parser.add_argument("--core-rank2-dir", default="")
    args = parser.parse_args()
    path = export_dashboard(
        Path(args.out_dir),
        Path(args.mvp_p5_dir) if args.mvp_p5_dir else None,
        Path(args.mvp_p4_dir) if args.mvp_p4_dir else None,
        Path(args.flow_dir),
        Path(args.flow_baseline_dir) if args.flow_baseline_dir else None,
        Path(args.flow_tiered_dir) if args.flow_tiered_dir else None,
        Path(args.hostile_dir) if args.hostile_dir else None,
        Path(args.core_rank2_dir) if args.core_rank2_dir else None,
    )
    print(path)


if __name__ == "__main__":
    main()
