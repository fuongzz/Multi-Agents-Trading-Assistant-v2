"""Export screener catalog and portfolio sleeve defaults.

This makes the intended architecture explicit:

- Many strategies are available to screeners/users.
- Only validated sleeves are active for portfolio allocation by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from multiagents_trading_assistant.edge_lab.strategy_sleeves import (
    catalog_rows,
    default_architecture,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports" / "strategy_architecture"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--include-shadow-sleeves", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    catalog = catalog_rows()
    architecture = default_architecture(include_shadow_sleeves=args.include_shadow_sleeves).to_dict()

    (out / "strategy_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "portfolio_sleeves.json").write_text(
        json.dumps(architecture, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(catalog).to_csv(out / "strategy_catalog.csv", index=False)
    pd.DataFrame(architecture["portfolio_sleeves"]).to_csv(out / "portfolio_sleeves.csv", index=False)

    print(f"[strategy-architecture] catalog={len(catalog)}")
    print(f"[strategy-architecture] active_sleeves={len(architecture['portfolio_sleeves'])}")
    print(f"[strategy-architecture] saved={out}")


if __name__ == "__main__":
    main()
