"""Export a client-facing Excel workbook from the combined paper dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.table import Table, TableStyleInfo


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "reports" / "combined_paper_trading_demo"
DEFAULT_OUT_DIR = ROOT / "reports" / "client_demo"

NAVY = "17324D"
BLUE = "2374AB"
TEAL = "087E8B"
GREEN = "138A58"
RED = "C73E3A"
GOLD = "D5A021"
PALE_BLUE = "EAF3F8"
PALE_GREEN = "E9F5EF"
PALE_GOLD = "FFF4D6"
PALE_RED = "FCE8E7"
INPUT = "FFF2CC"
WHITE = "FFFFFF"
GREY = "667085"
THIN_GREY = Side(style="thin", color="D7DFE7")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _title(ws, title: str, subtitle: str, last_column: int) -> None:
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=last_column)
    cell = ws.cell(1, 1, title)
    cell.font = Font(size=20, bold=True, color=WHITE)
    cell.fill = PatternFill("solid", fgColor=NAVY)
    cell.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 36
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=last_column)
    subtitle_cell = ws.cell(2, 1, subtitle)
    subtitle_cell.font = Font(size=10, italic=True, color=GREY)
    subtitle_cell.alignment = Alignment(horizontal="left")
    ws.row_dimensions[2].height = 22


def _header(ws, row: int, headers: list[str]) -> None:
    for col, label in enumerate(headers, start=1):
        cell = ws.cell(row, col, label)
        cell.font = Font(bold=True, color=WHITE)
        cell.fill = PatternFill("solid", fgColor=BLUE)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=Side(style="medium", color=NAVY))
    ws.row_dimensions[row].height = 34


def _table(ws, name: str, header_row: int, max_col: int, max_row: int) -> None:
    if max_row <= header_row:
        return
    ref = f"A{header_row}:{get_column_letter(max_col)}{max_row}"
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    ws.add_table(table)


def _finish_sheet(ws, widths: dict[str, float], freeze: str | None = None) -> None:
    for column, width in widths.items():
        ws.column_dimensions[column].width = width
    if freeze:
        ws.freeze_panes = freeze
    ws.sheet_view.showGridLines = False
    for row in ws.iter_rows():
        for cell in row:
            cell.border = Border(bottom=THIN_GREY)
            cell.alignment = Alignment(vertical="center", wrap_text=True)


def _vnd_cells(ws, columns: list[int], start_row: int, end_row: int) -> None:
    for col in columns:
        for row in range(start_row, end_row + 1):
            ws.cell(row, col).number_format = '#,##0" ₫"'


def _percent_cells(ws, columns: list[int], start_row: int, end_row: int) -> None:
    for col in columns:
        for row in range(start_row, end_row + 1):
            ws.cell(row, col).number_format = "0.00%"


def _unique_watchlist(plans: pd.DataFrame, limit: int = 24) -> pd.DataFrame:
    if plans.empty:
        return plans
    work = plans.copy()
    work["priority"] = work["decision_status"].map(
        {"PAPER_TARGET": 0, "OBSERVE_ONLY_GATE_REMOVED": 1}
    ).fillna(2)
    work["score_n"] = pd.to_numeric(work["score"], errors="coerce").fillna(-1)
    work = work.sort_values(["priority", "score_n"], ascending=[True, False])
    grouped: list[dict[str, Any]] = []
    for symbol, group in work.groupby("symbol", sort=False):
        first = group.iloc[0].to_dict()
        first["sleeves"] = ", ".join(sorted(group["sleeve_id"].dropna().astype(str).unique()))
        first["symbol"] = symbol
        grouped.append(first)
    return pd.DataFrame(grouped).head(limit)


def _targets(plans: pd.DataFrame) -> pd.DataFrame:
    if plans.empty:
        return plans
    target = plans.loc[plans["decision_status"] == "PAPER_TARGET"].copy()
    if target.empty:
        return target
    grouped: list[dict[str, Any]] = []
    for symbol, group in target.groupby("symbol", sort=False):
        first = group.iloc[0].to_dict()
        first["sleeves"] = ", ".join(sorted(group["sleeve_id"].dropna().astype(str).unique()))
        grouped.append(first)
    return pd.DataFrame(grouped)


def _overview_sheet(
    wb: Workbook,
    pulse: dict[str, Any],
    watchlist: pd.DataFrame,
    targets: pd.DataFrame,
    as_of: str,
) -> None:
    ws = wb.active
    ws.title = "01_Tong_quan"
    _title(ws, "AI Trading Assistant - Hồ sơ cơ hội đầu tư demo", f"Dữ liệu tín hiệu đến phiên {as_of} | VN100 | Paper / tham khảo, không phải lệnh môi giới", 8)
    ws["A4"] = "Tên khách hàng"
    ws["B4"] = ""
    ws["D4"] = "Người tư vấn"
    ws["E4"] = ""
    ws["A5"] = "Ngày gửi"
    ws["B5"] = ""
    ws["D5"] = "Mục tiêu trao đổi"
    ws["E5"] = ""
    for cell in ["B4", "E4", "B5", "E5"]:
        ws[cell].fill = PatternFill("solid", fgColor=INPUT)
    ws["A7"] = "Bức tranh hôm nay"
    ws["A7"].font = Font(bold=True, size=14, color=NAVY)
    metrics = [
        ("Ngày dữ liệu", as_of),
        ("Regime thị trường", pulse.get("market_regime_state", "-")),
        ("Trạng thái Flow", pulse.get("pulse_state", "-")),
        ("Mã có tiền vào hôm nay", pulse.get("strong_inflow_symbols_today", 0)),
        ("Mã bền bỉ >=3/5 phiên", pulse.get("persistent_inflow_symbols_5d", 0)),
        ("Paper target đang đề xuất", len(targets)),
        ("Mã trong watchlist demo", len(watchlist)),
        ("CHDM20 thay đổi 5 phiên", pulse.get("mkt_CHDM20_change_5d", "-")),
    ]
    for idx, (label, value) in enumerate(metrics):
        row = 9 + idx // 4 * 3
        col = 1 + (idx % 4) * 2
        ws.cell(row, col, label).font = Font(size=10, color=GREY)
        ws.cell(row + 1, col, value).font = Font(size=17, bold=True, color=NAVY)
        ws.cell(row + 1, col).fill = PatternFill("solid", fgColor=PALE_BLUE)
    ws["A16"] = "Thông điệp gửi khách hàng"
    ws["A16"].font = Font(bold=True, size=14, color=NAVY)
    ws.merge_cells("A17:H19")
    ws["A17"] = (
        "Danh sách này tổng hợp tín hiệu định lượng và dòng tiền quan sát từ dữ liệu giá/khối lượng. "
        "Các mức mua, cắt lỗ và kỳ vọng là vùng tham khảo để trao đổi; quyết định giao dịch thực tế "
        "phụ thuộc khẩu vị rủi ro, khả năng khớp lệnh và xác nhận của khách hàng."
    )
    ws["A17"].alignment = Alignment(wrap_text=True, vertical="top")
    ws["A17"].fill = PatternFill("solid", fgColor=PALE_GOLD)
    ws["A21"] = "Lưu ý rủi ro"
    ws["A21"].font = Font(bold=True, size=14, color=RED)
    ws.merge_cells("A22:H23")
    ws["A22"] = (
        "Market regime hiện đang là RISK_OFF. Flow V2 có gate nên không phát target mua trong điều kiện này; "
        "các mã bỏ gate được ghi là QUAN SÁT, không phải khuyến nghị đặt lệnh."
    )
    ws["A22"].fill = PatternFill("solid", fgColor=PALE_RED)
    _finish_sheet(ws, {column: 18 for column in "ABCDEFGH"})
    ws.column_dimensions["A"].width = 25
    ws.column_dimensions["E"].width = 26


def _watchlist_sheet(wb: Workbook, watchlist: pd.DataFrame, as_of: str) -> None:
    ws = wb.create_sheet("02_Watchlist")
    headers = [
        "STT", "Ngày tín hiệu", "Mã", "Phân loại", "Sleeve nguồn", "Chiến lược", "Regime",
        "Điểm", "Close tham chiếu", "Vùng mua từ", "Vùng mua đến", "Cắt lỗ đề xuất",
        "Mục tiêu đề xuất", "R:R", "Trạng thái KH", "Giá mua thực tế", "KL mua",
        "Vốn mua", "Giá hiện tại", "Lãi/lỗ tạm tính", "Lãi/lỗ %", "Hành động tiếp", "Ghi chú",
    ]
    _title(ws, "Watchlist tư vấn khách hàng", f"Seed từ dashboard ngày {as_of}. Ô màu vàng dành cho cập nhật thủ công.", len(headers))
    _header(ws, 4, headers)
    for idx, row in enumerate(watchlist.to_dict("records"), start=1):
        excel_row = idx + 4
        category = "PAPER TARGET" if row.get("decision_status") == "PAPER_TARGET" else "QUAN SÁT - GATE CHẶN"
        values = [
            idx, row.get("signal_date"), row.get("symbol"), category, row.get("sleeves"),
            row.get("strategy_name"), row.get("market_regime_state") or "RISK_OFF", _num(row.get("score")),
            _num(row.get("reference_close_vnd")), _num(row.get("entry_zone_low_vnd")),
            _num(row.get("entry_zone_high_vnd")), _num(row.get("stop_loss_vnd")),
            _num(row.get("take_profit_vnd")), _num(row.get("reward_risk_ratio")),
            "Đang theo dõi", None, None, f'=IF(AND(P{excel_row}<>"",Q{excel_row}<>""),P{excel_row}*Q{excel_row},"")',
            None, f'=IF(AND(R{excel_row}<>"",S{excel_row}<>""),Q{excel_row}*(S{excel_row}-P{excel_row}),"")',
            f'=IF(AND(P{excel_row}<>"",S{excel_row}<>""),(S{excel_row}/P{excel_row})-1,"")',
            "Theo dõi", "", 
        ]
        for col, value in enumerate(values, start=1):
            ws.cell(excel_row, col, value)
        if category == "PAPER TARGET":
            ws.cell(excel_row, 4).fill = PatternFill("solid", fgColor=PALE_GREEN)
        else:
            ws.cell(excel_row, 4).fill = PatternFill("solid", fgColor=PALE_GOLD)
        for col in [15, 16, 17, 19, 22, 23]:
            ws.cell(excel_row, col).fill = PatternFill("solid", fgColor=INPUT)
    last_row = 4 + len(watchlist)
    status_validation = DataValidation(type="list", formula1='"Đang theo dõi,Đã trao đổi,Khách quan tâm,Khách từ chối,Đã mua,Đã bán"', allow_blank=True)
    action_validation = DataValidation(type="list", formula1='"Theo dõi,Chờ vùng mua,Mua thử,Mua,Chốt lời,Cắt lỗ,Đóng theo dõi"', allow_blank=True)
    ws.add_data_validation(status_validation)
    ws.add_data_validation(action_validation)
    status_validation.add(f"O5:O{max(last_row, 5)}")
    action_validation.add(f"V5:V{max(last_row, 5)}")
    _vnd_cells(ws, [9, 10, 11, 12, 13, 16, 18, 19, 20], 5, last_row)
    _percent_cells(ws, [21], 5, last_row)
    green_fill = PatternFill("solid", fgColor=PALE_GREEN)
    red_fill = PatternFill("solid", fgColor=PALE_RED)
    ws.conditional_formatting.add(f"T5:T{last_row}", CellIsRule(operator="greaterThan", formula=["0"], fill=green_fill))
    ws.conditional_formatting.add(f"T5:T{last_row}", CellIsRule(operator="lessThan", formula=["0"], fill=red_fill))
    _table(ws, "ClientWatchlist", 4, len(headers), last_row)
    widths = {
        "A": 7, "B": 14, "C": 10, "D": 25, "E": 23, "F": 33, "G": 14, "H": 11,
        "I": 17, "J": 16, "K": 17, "L": 18, "M": 18, "N": 9, "O": 18, "P": 18,
        "Q": 11, "R": 17, "S": 17, "T": 18, "U": 13, "V": 18, "W": 30,
    }
    _finish_sheet(ws, widths, "A5")
    ws.auto_filter.ref = f"A4:{get_column_letter(len(headers))}{last_row}"


def _trade_plan_sheet(wb: Workbook, targets: pd.DataFrame, as_of: str) -> None:
    ws = wb.create_sheet("03_Ke_hoach_lenh")
    headers = [
        "Ngày tín hiệu", "Mã", "Sleeves", "Chiến lược", "Trạng thái", "Close tham chiếu",
        "Entry từ", "Entry đến", "Stop loss", "Mục tiêu", "R:R", "KL dự kiến",
        "Giá đặt lệnh", "Giá trị dự kiến", "Xác nhận KH", "Ghi chú",
    ]
    _title(ws, "Kế hoạch lệnh paper / đề xuất trao đổi", f"Tín hiệu phát sau close {as_of}; chỉ ghi nhận mua thật khi có xác nhận và khớp lệnh.", len(headers))
    _header(ws, 4, headers)
    for idx, row in enumerate(targets.to_dict("records"), start=5):
        values = [
            row.get("signal_date"), row.get("symbol"), row.get("sleeves"), row.get("strategy_name"),
            "Chờ phiên kế tiếp", _num(row.get("reference_close_vnd")), _num(row.get("entry_zone_low_vnd")),
            _num(row.get("entry_zone_high_vnd")), _num(row.get("stop_loss_vnd")),
            _num(row.get("take_profit_vnd")), _num(row.get("reward_risk_ratio")), None, None,
            f'=IF(AND(L{idx}<>"",M{idx}<>""),L{idx}*M{idx},"")', "Chưa xác nhận", "",
        ]
        for col, value in enumerate(values, start=1):
            ws.cell(idx, col, value)
        for col in [12, 13, 15, 16]:
            ws.cell(idx, col).fill = PatternFill("solid", fgColor=INPUT)
    last_row = max(4 + len(targets), 5)
    if targets.empty:
        ws.cell(5, 1, "Không có target paper tại ngày dữ liệu hiện tại.")
    approval = DataValidation(type="list", formula1='"Chưa xác nhận,Đồng ý,Không tham gia,Chờ thêm thông tin"', allow_blank=True)
    ws.add_data_validation(approval)
    approval.add(f"O5:O{last_row}")
    _vnd_cells(ws, [6, 7, 8, 9, 10, 13, 14], 5, last_row)
    _table(ws, "TradePlans", 4, len(headers), last_row if not targets.empty else 4)
    _finish_sheet(ws, {
        "A": 15, "B": 10, "C": 19, "D": 33, "E": 19, "F": 18, "G": 15, "H": 15,
        "I": 16, "J": 16, "K": 9, "L": 13, "M": 17, "N": 19, "O": 18, "P": 34,
    }, "A5")


def _trade_journal_sheet(wb: Workbook) -> None:
    ws = wb.create_sheet("04_Nhat_ky_mua_ban")
    headers = [
        "Ngày", "Mã", "Khách hàng/TK", "Loại lệnh", "Trạng thái", "Khối lượng",
        "Giá khớp", "Phí", "Thuế", "Dòng tiền ròng", "Lý do giao dịch", "Ghi chú",
    ]
    _title(ws, "Nhật ký mua bán cập nhật thủ công", "Nhập các giao dịch thực tế tại các ô vàng. Sheet này độc lập với paper ledger của hệ thống.", len(headers))
    _header(ws, 4, headers)
    for row in range(5, 55):
        for col in range(1, len(headers) + 1):
            ws.cell(row, col).fill = PatternFill("solid", fgColor=INPUT)
        ws.cell(
            row,
            10,
            f'=IF(AND(D{row}<>"",F{row}<>"",G{row}<>""),IF(D{row}="MUA",-F{row}*G{row}-H{row}-I{row},F{row}*G{row}-H{row}-I{row}),"")',
        )
        ws.cell(row, 10).fill = PatternFill("solid", fgColor=WHITE)
    side = DataValidation(type="list", formula1='"MUA,BÁN"', allow_blank=True)
    trade_status = DataValidation(type="list", formula1='"Dự kiến,Đã đặt,Đã khớp,Hủy"', allow_blank=True)
    ws.add_data_validation(side)
    ws.add_data_validation(trade_status)
    side.add("D5:D54")
    trade_status.add("E5:E54")
    _vnd_cells(ws, [7, 8, 9, 10], 5, 54)
    _table(ws, "TradeJournal", 4, len(headers), 54)
    _finish_sheet(ws, {
        "A": 15, "B": 10, "C": 22, "D": 13, "E": 14, "F": 13, "G": 17,
        "H": 15, "I": 15, "J": 19, "K": 35, "L": 35,
    }, "A5")


def _flow_sheet(wb: Workbook, name: str, title: str, frame: pd.DataFrame, as_of: str) -> None:
    ws = wb.create_sheet(name)
    headers = ["Hạng", "Mã", "Ngành", "Tín hiệu", "Xu hướng 5 phiên", "Số phiên có tín hiệu", "Điểm hôm nay", "Điểm TB 5 phiên", "Biến động điểm 5 phiên", "Thanh khoản x", "RS percentile"]
    _title(ws, title, f"Flow V2 proxy ngày {as_of}; không phải số liệu mua ròng nhà đầu tư.", len(headers))
    _header(ws, 4, headers)
    for idx, row in enumerate(frame.to_dict("records"), start=5):
        values = [
            row.get("rank"), row.get("symbol"), row.get("industry"), row.get("flow_signal"),
            row.get("flow_trend_5d"), row.get("inflow_days_5d"), _num(row.get("flow_score_today")),
            _num(row.get("flow_score_avg_5d")), _num(row.get("flow_score_change_5d")),
            _num(row.get("value_ratio_20")), _num(row.get("rs_percentile_20")),
        ]
        for col, value in enumerate(values, start=1):
            ws.cell(idx, col, value)
        ws.cell(idx, 4).fill = PatternFill("solid", fgColor=PALE_GREEN)
    last_row = max(4 + len(frame), 5)
    _table(ws, "Tbl" + name.replace("_", ""), 4, len(headers), last_row if not frame.empty else 4)
    if not frame.empty:
        chart = BarChart()
        chart.type = "bar"
        chart.title = title
        chart.y_axis.title = "Mã"
        chart.x_axis.title = "Điểm dòng tiền"
        chart.height = 8
        chart.width = 15
        data = Reference(ws, min_col=7, min_row=4, max_row=min(last_row, 14))
        cats = Reference(ws, min_col=2, min_row=5, max_row=min(last_row, 14))
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        ws.add_chart(chart, "M4")
    _finish_sheet(ws, {
        "A": 9, "B": 10, "C": 24, "D": 22, "E": 23, "F": 18,
        "G": 16, "H": 17, "I": 22, "J": 14, "K": 16,
    }, "A5")


def _positions_sheet(wb: Workbook, open_positions: pd.DataFrame) -> None:
    ws = wb.create_sheet("07_Vi_the_dang_giu")
    if open_positions.empty:
        _title(ws, "Vị thế đang giữ", "Paper ledger hiện chưa có vị thế khớp mở.", 6)
        ws["A5"] = "Chưa có vị thế mở. Các target phát ngày 2026-05-26 chờ phiên giao dịch kế tiếp để có thể khớp paper."
        ws.merge_cells("A5:F6")
        ws["A5"].fill = PatternFill("solid", fgColor=PALE_GOLD)
        _finish_sheet(ws, {col: 20 for col in "ABCDEF"})
        return
    _title(ws, "Vị thế đang giữ", "Tình trạng paper holding hiện tại.", len(open_positions.columns))
    headers = [str(column) for column in open_positions.columns]
    _header(ws, 4, headers)
    for row_idx, row in enumerate(open_positions.to_dict("records"), start=5):
        for col, value in enumerate(row.values(), start=1):
            ws.cell(row_idx, col, value)
    _table(ws, "OpenPositions", 4, len(headers), 4 + len(open_positions))
    _finish_sheet(ws, {get_column_letter(col): 18 for col in range(1, len(headers) + 1)}, "A5")


def _guidance_sheet(wb: Workbook, as_of: str) -> None:
    ws = wb.create_sheet("08_Huong_dan")
    _title(ws, "Hướng dẫn sử dụng file demo", f"Phiên dữ liệu nguồn: {as_of}", 6)
    guidance = [
        ("1. Watchlist", "Dùng sheet 02 để lọc mã trao đổi; cập nhật trạng thái khách hàng, giá mua thực tế, khối lượng, giá hiện tại và ghi chú ở ô vàng."),
        ("2. Kế hoạch lệnh", "Sheet 03 chỉ chứa target có thể đưa vào kịch bản giao dịch. Điền số lượng và giá đặt sau khi trao đổi với khách."),
        ("3. Nhật ký mua bán", "Mỗi lệnh thực tế nhập một dòng ở sheet 04. Công thức dòng tiền tự phân biệt MUA/BÁN."),
        ("4. Dòng tiền", "Sheet 05-06 là tín hiệu proxy từ giá, khối lượng và Flow V2; không phải dữ liệu mua ròng của khối ngoại/tự doanh."),
        ("5. Gate rủi ro", "Dòng QUAN SÁT - GATE CHẶN chỉ dùng để giải thích cơ hội, không được trình bày như lệnh mua của hệ thống."),
        ("6. Disclaimer", "Đây là tài liệu demo/paper trading phục vụ trao đổi. Không phải khuyến nghị đầu tư cá nhân hoá hoặc cam kết lợi nhuận."),
    ]
    for idx, (heading, text) in enumerate(guidance, start=5):
        ws.cell(idx, 1, heading).font = Font(bold=True, color=NAVY)
        ws.cell(idx, 2, text)
        ws.cell(idx, 2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[idx].height = 42
    ws.merge_cells("B5:F5")
    for row in range(6, 11):
        ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=6)
    _finish_sheet(ws, {"A": 22, "B": 35, "C": 18, "D": 18, "E": 18, "F": 18})


def export_workbook(dashboard_dir: Path, output_path: Path | None = None) -> Path:
    plans = _read_csv(dashboard_dir / "17_signal_plans.csv")
    today_flow = _read_csv(dashboard_dir / "14_money_flow_today.csv")
    recent_flow = _read_csv(dashboard_dir / "15_money_flow_recent.csv")
    open_positions = _read_csv(dashboard_dir / "03_open_holdings.csv")
    pulse = _read_json(dashboard_dir / "flow_v2" / "money_flow_pulse.json")
    as_of = str(pulse.get("date") or (plans["signal_date"].max() if not plans.empty else "-"))
    if output_path is None:
        output_path = DEFAULT_OUT_DIR / f"AI_Trading_Watchlist_Demo_{as_of}.xlsx"
    watchlist = _unique_watchlist(plans)
    targets = _targets(plans)

    wb = Workbook()
    wb.properties.title = f"AI Trading Watchlist Demo - {as_of}"
    wb.properties.subject = "Watchlist, trade planning and money-flow demo workbook"
    wb.properties.creator = "AI Trading Assistant"
    _overview_sheet(wb, pulse, watchlist, targets, as_of)
    _watchlist_sheet(wb, watchlist, as_of)
    _trade_plan_sheet(wb, targets, as_of)
    _trade_journal_sheet(wb)
    _flow_sheet(wb, "05_Dong_tien_hom_nay", "Dòng tiền nổi bật hôm nay", today_flow, as_of)
    _flow_sheet(wb, "06_Dong_tien_5_phien", "Dòng tiền mạnh 5 phiên", recent_flow, as_of)
    _positions_sheet(wb, open_positions)
    _guidance_sheet(wb, as_of)
    wb.active = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    check = load_workbook(output_path, data_only=False)
    check.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a client-facing Excel watchlist workbook.")
    parser.add_argument("--dashboard-dir", type=Path, default=DASHBOARD_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = export_workbook(args.dashboard_dir, args.output)
    print(path)


if __name__ == "__main__":
    main()
