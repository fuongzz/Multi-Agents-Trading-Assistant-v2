"""
ICB Level 2 -> Level 1 mapping for Vietnam stock market.

L2 labels in ohlcv_master.parquet are Vietnamese ICB sector names.
Map to 11 L1 super-sectors (Real Estate split out from Financials due
to its size and distinct cycle in VN market).
"""

from __future__ import annotations

L2_TO_L1: dict[str, str] = {
    # Financials (banks + non-bank financials)
    "Ngân hàng": "Financials",
    "Bảo hiểm": "Financials",
    "Dịch vụ tài chính": "Financials",
    # Real Estate (own L1 — large, distinct cycle in VN)
    "Bất động sản": "RealEstate",
    # Industrials
    "Xây dựng và Vật liệu": "Industrials",
    "Hàng & Dịch vụ Công nghiệp": "Industrials",
    # Consumer Goods
    "Thực phẩm và đồ uống": "ConsumerGoods",
    "Hàng cá nhân & Gia dụng": "ConsumerGoods",
    "Ô tô và phụ tùng": "ConsumerGoods",
    # Consumer Services
    "Bán lẻ": "ConsumerServices",
    "Du lịch và Giải trí": "ConsumerServices",
    # Basic Materials
    "Hóa chất": "BasicMaterials",
    "Tài nguyên Cơ bản": "BasicMaterials",
    # Oil & Gas
    "Dầu khí": "OilGas",
    # Utilities
    "Điện, nước & xăng dầu khí đốt": "Utilities",
    # Health Care
    "Y tế": "HealthCare",
    # Technology
    "Công nghệ Thông tin": "Technology",
    # Telecom
    "Viễn thông": "Telecom",
}

L1_SECTORS: tuple[str, ...] = (
    "Financials",
    "RealEstate",
    "Industrials",
    "ConsumerGoods",
    "ConsumerServices",
    "BasicMaterials",
    "OilGas",
    "Utilities",
    "HealthCare",
    "Technology",
    "Telecom",
)


def map_l2_to_l1(l2: str | None) -> str | None:
    if l2 is None:
        return None
    return L2_TO_L1.get(l2)
