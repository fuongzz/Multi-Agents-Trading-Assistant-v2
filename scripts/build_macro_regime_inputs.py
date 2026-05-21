"""Fetch macro inputs for macro-aware regime research.

Uses vnstock_data Unified UI where available. The output is normalized CSVs
that can be joined causally by the add-on regime filter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from vnstock_data import Macro


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "research" / "macro_regime"


def _with_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        if "report_time" in out.columns:
            out["date"] = out["report_time"]
        else:
            out["date"] = out.index
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build macro regime input CSVs")
    parser.add_argument("--start-month", default="2022-01")
    parser.add_argument("--end-month", default="2026-05")
    parser.add_argument("--start-day", default="2022-01-01")
    parser.add_argument("--end-day", default="2026-05-16")
    parser.add_argument("--out", default=str(OUT_ROOT))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    macro = Macro()

    cpi = _with_date(macro.economy().cpi(start=args.start_month, end=args.end_month, period="month"))
    cpi.to_csv(out_dir / "cpi.csv", index=False)
    print(f"saved {out_dir / 'cpi.csv'} rows={len(cpi)}")

    fx = _with_date(macro.currency().exchange_rate(start=args.start_day, end=args.end_day, period="day"))
    fx.to_csv(out_dir / "fx.csv", index=False)
    print(f"saved {out_dir / 'fx.csv'} rows={len(fx)}")

    try:
        interest = _with_date(
            macro.currency().interest_rate(
                start=args.start_day,
                end=args.end_day,
                period="day",
                format="long",
            )
        )
    except Exception as exc:
        print(f"interest start/end fetch failed: {exc}; falling back to length='1Y'")
        interest = _with_date(macro.currency().interest_rate(length="1Y", period="day", format="long"))
    interest.to_csv(out_dir / "interest_rate.csv", index=False)
    print(f"saved {out_dir / 'interest_rate.csv'} rows={len(interest)}")


if __name__ == "__main__":
    main()
