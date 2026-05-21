"""Inspect and export parquet files for demos and data audits.

Examples:
    python scripts/view_parquet.py multiagents_trading_assistant/data/ohlcv_master.parquet --info
    python scripts/view_parquet.py multiagents_trading_assistant/data/ohlcv_master.parquet --symbol CTG --tail 20
    python scripts/view_parquet.py multiagents_trading_assistant/data/ohlcv_master.parquet --where "date >= DATE '2026-05-01'" --export-html reports/ohlcv_may.html
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _resolve_path(path: str) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Parquet file not found: {resolved}")
    return resolved


def _metadata(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    return {
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "columns": len(schema.names),
        "schema": [(field.name, str(field.type)) for field in schema],
    }


def _print_info(path: Path) -> None:
    meta = _metadata(path)
    print(f"Path: {meta['path']}")
    print(f"Size: {meta['size_mb']} MB")
    print(f"Rows: {meta['rows']:,}")
    print(f"Row groups: {meta['row_groups']}")
    print(f"Columns: {meta['columns']}")
    print("\nSchema:")
    for name, dtype in meta["schema"]:
        print(f"  {name}: {dtype}")


def _build_query(args: argparse.Namespace, path: Path) -> str:
    columns = "*"
    if args.columns:
        columns = ", ".join(_quote_identifier(item.strip()) for item in args.columns.split(",") if item.strip())

    clauses: list[str] = []
    if args.symbol:
        symbols = ", ".join(_sql_literal(item.strip().upper()) for item in args.symbol.split(",") if item.strip())
        clauses.append(f"upper(symbol) IN ({symbols})")
    if args.start:
        clauses.append(f"date >= DATE {_sql_literal(args.start)}")
    if args.end:
        clauses.append(f"date <= DATE {_sql_literal(args.end)}")
    if args.where:
        clauses.append(f"({args.where})")

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order_sql = ""
    if args.order_by:
        order_sql = f"ORDER BY {args.order_by}"
    elif args.tail and "date" in [name for name, _ in _metadata(path)["schema"]]:
        order_sql = "ORDER BY date DESC"
    elif "date" in [name for name, _ in _metadata(path)["schema"]]:
        order_sql = "ORDER BY date ASC"

    limit = args.limit
    if args.tail:
        limit = args.tail
    limit_sql = f"LIMIT {int(limit)}" if limit and limit > 0 else ""
    parquet_path = str(path).replace("'", "''")
    return f"SELECT {columns} FROM read_parquet('{parquet_path}') {where_sql} {order_sql} {limit_sql}".strip()


def _read_frame(query: str) -> pd.DataFrame:
    with duckdb.connect(database=":memory:") as conn:
        return conn.execute(query).fetchdf()


def _write_html(frame: pd.DataFrame, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table_html = frame.to_html(index=False, classes="data-table", border=0)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f7f7f2; color: #151515; }}
    h1 {{ font-size: 22px; margin-bottom: 8px; }}
    .hint {{ color: #555; margin-bottom: 18px; }}
    .wrap {{ overflow: auto; border: 1px solid #ddd; background: white; box-shadow: 0 8px 24px rgba(0,0,0,.08); }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; white-space: nowrap; }}
    th {{ position: sticky; top: 0; background: #17324d; color: white; text-align: left; }}
    th, td {{ padding: 7px 9px; border-bottom: 1px solid #e8e8e8; }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="hint">Static parquet export. Use browser search or import CSV for heavier analysis.</div>
  <div class="wrap">{table_html}</div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Professional parquet inspector/exporter.")
    parser.add_argument("path", help="Path to parquet file")
    parser.add_argument("--info", action="store_true", help="Print metadata and schema")
    parser.add_argument("--columns", default="", help="Comma-separated selected columns")
    parser.add_argument("--symbol", default="", help="Comma-separated symbols, if the file has a symbol column")
    parser.add_argument("--start", default="", help="Start date YYYY-MM-DD, if the file has a date column")
    parser.add_argument("--end", default="", help="End date YYYY-MM-DD, if the file has a date column")
    parser.add_argument("--where", default="", help="Extra DuckDB SQL predicate, e.g. \"close > 10000\"")
    parser.add_argument("--order-by", default="", help="DuckDB ORDER BY expression without the ORDER BY keyword")
    parser.add_argument("--limit", type=int, default=50, help="Rows to display/export. Use 0 for all matched rows")
    parser.add_argument("--tail", type=int, default=0, help="Show latest N rows by date descending")
    parser.add_argument("--export-csv", default="", help="Export result to CSV")
    parser.add_argument("--export-html", default="", help="Export result to browser-friendly HTML")
    args = parser.parse_args()

    path = _resolve_path(args.path)
    if args.info:
        _print_info(path)
        if not any([args.export_csv, args.export_html, args.symbol, args.start, args.end, args.where, args.tail]):
            return

    query = _build_query(args, path)
    frame = _read_frame(query)

    if args.tail and not frame.empty and "date" in frame.columns:
        frame = frame.sort_values("date")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 80)
    print(f"\nQuery:\n{query}\n")
    print(f"Rows returned: {len(frame):,}")
    print(frame.to_string(index=False))

    if args.export_csv:
        out = Path(args.export_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\nExported CSV: {out}")
    if args.export_html:
        out = Path(args.export_html)
        _write_html(frame, out, title=f"Parquet View: {path.name}")
        print(f"\nExported HTML: {out}")


if __name__ == "__main__":
    main()
