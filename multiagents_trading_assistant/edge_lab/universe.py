"""Universe definitions for edge research."""

from __future__ import annotations

VN30 = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]


def get_universe(name: str) -> list[str]:
    key = name.upper()
    if key == "VN30":
        return VN30.copy()
    raise ValueError(f"Unsupported universe: {name}")

