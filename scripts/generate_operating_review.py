from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiagents_trading_assistant.services.operating_review_service import build_operating_review


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate operating review from trades + trusted backtests")
    parser.add_argument("--as-of-date", help="Reference date YYYY-MM-DD", default=None)
    parser.add_argument("--lookback-days", type=int, default=20)
    args = parser.parse_args()

    review = build_operating_review(
        as_of_date=args.as_of_date,
        lookback_days=args.lookback_days,
    )
    print(json.dumps(review, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
