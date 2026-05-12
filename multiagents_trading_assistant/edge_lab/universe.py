"""Universe definitions for edge research."""

from __future__ import annotations

VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]

VN100 = [
    "ACB", "ANV", "BCM", "BID", "BMP", "BSI", "BSR", "BVH", "BWE", "CII",
    "CMG", "CTD", "CTG", "CTR", "CTS", "DBC", "DCM", "DGC", "DGW", "DIG",
    "DPM", "DSE", "DXG", "DXS", "EIB", "EVF", "FPT", "FRT", "FTS", "GAS",
    "GEE", "GEX", "GMD", "GVR", "HAG", "HCM", "HDB", "HDC", "HDG", "HHV",
    "HPG", "HSG", "HT1", "IMP", "KBC", "KDC", "KDH", "KOS", "LPB", "MBB",
    "MSB", "MSN", "MWG", "NAB", "NKG", "NLG", "NT2", "NVL", "OCB", "PAN",
    "PC1", "PDR", "PHR", "PLX", "PNJ", "POW", "PVD", "PVT", "REE", "SAB",
    "SBT", "SCS", "SHB", "SIP", "SJS", "SSB", "SSI", "STB", "SZC", "TCB",
    "TCH", "TPB", "VCB", "VCG", "VCI", "VGC", "VHC", "VHM", "VIB", "VIC",
    "VIX", "VJC", "VND", "VNM", "VPB", "VPI", "VPL", "VRE", "VSC", "VTP",
]

HNX30 = [
    "BVS", "CAP", "CEO", "DP3", "DTD", "DVM", "DXP", "HUT", "IDC", "IDV",
    "L14", "L18", "LAS", "LHC", "MBS", "NDN", "NTP", "PLC", "PSD", "PVB",
    "PVC", "PVS", "SHS", "SLS", "TMB", "TNG", "TVD", "VC3", "VCS", "VFS",
]

UPCOM30 = [
    "OIL", "ABI", "VEA", "QNS", "VGT", "VHG", "ACV", "VCR", "SBS", "ABB",
    "HKB", "MSR", "SBD", "VGI", "AAH", "AAS", "ABC", "ABW", "ACE", "ACM",
    "ACS", "AGF", "AGM", "AGP", "AGX", "AIC", "AIG", "ALC", "ALV", "AMP",
]


def get_universe(name: str) -> list[str]:
    key = name.upper().replace("-", "_")
    if key == "VN30":
        return VN30.copy()
    if key == "VN100":
        return VN100.copy()
    if key == "HNX30":
        return HNX30.copy()
    if key == "UPCOM30":
        return UPCOM30.copy()
    if key in {"VN100_HNX30_UPCOM30", "WIDE", "ALL_RESEARCH"}:
        return sorted(set(VN100) | set(HNX30) | set(UPCOM30))
    raise ValueError(f"Unsupported universe: {name}")
