"""Inject sortable-table behavior into static HTML reports."""

from __future__ import annotations

import argparse
from pathlib import Path


SORTABLE_JS = r"""(() => {
  if (window.__sortableTablesInstalled) return;
  window.__sortableTablesInstalled = true;

  const style = document.createElement("style");
  style.textContent = `
    table.sortable-table th {
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }
    table.sortable-table th::after {
      content: " sort";
      color: rgba(255,255,255,.55);
      font-weight: 400;
      font-size: 10px;
    }
    table.sortable-table th[data-sort-dir="desc"]::after {
      content: " desc";
      color: #fff;
    }
    table.sortable-table th[data-sort-dir="asc"]::after {
      content: " asc";
      color: #fff;
    }
  `;
  document.head.appendChild(style);

  function tableKey(table, tableIndex) {
    const headers = Array.from(table.querySelectorAll("thead th"))
      .map((th) => (th.innerText || "").trim())
      .join("|");
    return `sortable:${location.pathname}:${tableIndex}:${headers}`;
  }

  function cellText(row, index) {
    return (row.children[index]?.innerText || "").trim();
  }

  function parseValue(text) {
    const raw = String(text || "").trim();
    if (!raw) return { type: "empty", value: "" };

    const date = Date.parse(raw);
    if (!Number.isNaN(date) && /^\d{4}[-/]\d{1,2}[-/]\d{1,2}/.test(raw)) {
      return { type: "date", value: date };
    }

    const cleaned = raw
      .replace(/\s+/g, "")
      .replace(/[,%]/g, "")
      .replace(/VND|VNĐ|đ|₫/gi, "")
      .replace(/^\+/, "")
      .replace(/[()]/g, "");
    if (/^-?\d+(\.\d+)?$/.test(cleaned)) {
      return { type: "number", value: Number(cleaned) };
    }

    return { type: "text", value: raw.toLocaleLowerCase("vi-VN") };
  }

  function compareValues(a, b, dir) {
    const av = parseValue(a);
    const bv = parseValue(b);
    const multiplier = dir === "asc" ? 1 : -1;
    if (av.type === "empty" && bv.type !== "empty") return 1;
    if (bv.type === "empty" && av.type !== "empty") return -1;
    if ((av.type === "number" || av.type === "date") && av.type === bv.type) {
      return (av.value - bv.value) * multiplier;
    }
    return String(av.value).localeCompare(String(bv.value), "vi-VN", { numeric: true }) * multiplier;
  }

  function sortTable(table, index, dir, persist) {
    const tbody = table.querySelector("tbody");
    if (!tbody) return;
    const headers = Array.from(table.querySelectorAll("thead th"));
    headers.forEach((header) => header.dataset.sortDir = "");
    if (headers[index]) headers[index].dataset.sortDir = dir;

    const rows = Array.from(tbody.querySelectorAll("tr"));
    rows.sort((a, b) => compareValues(cellText(a, index), cellText(b, index), dir));
    rows.forEach((row) => tbody.appendChild(row));

    if (persist) {
      try {
        localStorage.setItem(table.dataset.sortKey, JSON.stringify({ index, dir }));
      } catch (_) {}
    }
  }

  function installTable(table, tableIndex) {
    if (!table || table.dataset.sortableInstalled === "1") return;
    const headers = table.querySelectorAll("thead th");
    const tbody = table.querySelector("tbody");
    if (!headers.length || !tbody) return;
    table.classList.add("sortable-table");
    table.dataset.sortableInstalled = "1";
    table.dataset.sortKey = tableKey(table, tableIndex);
    headers.forEach((th, index) => {
      th.title = "Click de sort. Lan dau: cao den thap. Trang realtime se nho sort sau refresh.";
      th.dataset.sortIndex = String(index);
    });

    try {
      const saved = JSON.parse(localStorage.getItem(table.dataset.sortKey) || "null");
      if (saved && Number.isInteger(saved.index) && (saved.dir === "asc" || saved.dir === "desc")) {
        window.requestAnimationFrame(() => sortTable(table, saved.index, saved.dir, false));
      }
    } catch (_) {}
  }

  function installAll() {
    document.querySelectorAll("table").forEach(installTable);
  }

  document.addEventListener("click", (event) => {
    const th = event.target.closest("th");
    if (!th) return;
    const table = th.closest("table.sortable-table");
    if (!table) return;

    event.preventDefault();
    event.stopImmediatePropagation();

    const headers = Array.from(table.querySelectorAll("thead th"));
    const index = Number(th.dataset.sortIndex || headers.indexOf(th));
    const nextDir = th.dataset.sortDir === "desc" ? "asc" : "desc";
    sortTable(table, index, nextDir, true);
  }, true);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", installAll);
  } else {
    installAll();
  }
})();
"""


def script_src_for(html_path: Path, report_dir: Path) -> str:
    if html_path.parent == report_dir:
        return "sortable_tables.js"
    return "../" * len(html_path.parent.relative_to(report_dir).parts) + "sortable_tables.js"


def inject(html_path: Path, report_dir: Path) -> bool:
    text = html_path.read_text(encoding="utf-8", errors="replace")
    if "sortable_tables.js" in text:
        return False
    src = script_src_for(html_path, report_dir)
    tag = f'<script src="{src}"></script>'
    lower = text.lower()
    idx = lower.rfind("</body>")
    if idx >= 0:
        text = text[:idx] + f"  {tag}\n" + text[idx:]
    else:
        text = text + "\n" + tag + "\n"
    html_path.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    (report_dir / "sortable_tables.js").write_text(SORTABLE_JS, encoding="utf-8")
    changed = 0
    for html_path in sorted(report_dir.rglob("*.html")):
        if inject(html_path, report_dir):
            changed += 1
    print(f"sortable tables installed: {changed} html files; asset={report_dir / 'sortable_tables.js'}")


if __name__ == "__main__":
    main()
