from __future__ import annotations

from datetime import date

import pandas as pd

from scripts.wait_for_combined_eod_data import eod_coverage


def test_eod_coverage_counts_only_required_symbols_on_target_date(tmp_path) -> None:
    path = tmp_path / "ohlcv.parquet"
    pd.DataFrame(
        {
            "date": ["2026-05-22", "2026-05-25", "2026-05-25", "2026-05-25"],
            "symbol": ["AAA", "AAA", "BBB", "ZZZ"],
        }
    ).to_parquet(path, index=False)
    result = eod_coverage(date(2026, 5, 25), {"AAA", "BBB", "CCC"}, path)
    assert result["available"] == 2
    assert result["required"] == 3
    assert result["latest_date"] == "2026-05-25"
