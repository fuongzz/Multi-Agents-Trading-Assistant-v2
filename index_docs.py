"""
scripts/index_docs.py — ChromaDB Docs Indexer

Index toàn bộ docs/ vào ChromaDB để Claude Code search được bằng ngôn ngữ tự nhiên.
Chạy lại mỗi khi thêm hoặc sửa doc.

Usage:
    py -3.11 scripts/index_docs.py
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "docs"
DB_PATH  = str(ROOT / "chroma_db")
COLLECTION_NAME = "trading_docs"


def check_chromadb() -> bool:
    try:
        import chromadb
        return True
    except ImportError:
        print("❌ chromadb chưa cài. Chạy: py -3.11 -m pip install chromadb")
        return False


def load_and_chunk_docs(docs_dir: Path) -> list[dict]:
    """Đọc tất cả .md, chia theo heading ##."""
    chunks = []
    for md_file in sorted(docs_dir.glob("**/*.md")):
        content = md_file.read_text(encoding="utf-8")
        rel = str(md_file.relative_to(docs_dir.parent))
        chunks.extend(_split_by_heading(content, rel))
    print(f"📄 {len(chunks)} chunks từ {docs_dir}")
    return chunks


def _split_by_heading(content: str, source: str) -> list[dict]:
    chunks = []
    current_heading = "intro"
    current_lines: list[str] = []
    idx = 0

    for line in content.split("\n"):
        if line.startswith("## ") and current_lines:
            _flush(chunks, current_lines, current_heading, source, idx)
            idx += 1
            current_heading = line.lstrip("# ").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    _flush(chunks, current_lines, current_heading, source, idx)
    return chunks


def _flush(chunks, lines, heading, source, idx):
    text = "\n".join(lines).strip()
    if len(text) > 80:
        chunks.append({
            "id":       f"{source}::{idx}::{heading}",
            "content":  text,
            "metadata": {"source": source, "heading": heading, "idx": idx},
        })


def index(documents: list[dict]) -> None:
    import chromadb

    print(f"🔌 ChromaDB: {DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️  Xóa index cũ")
    except Exception:
        pass

    col = client.create_collection(COLLECTION_NAME)

    BATCH = 50
    for i in range(0, len(documents), BATCH):
        batch = documents[i:i+BATCH]
        col.add(
            ids=[d["id"] for d in batch],
            documents=[d["content"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        print(f"  ✅ {min(i+BATCH, len(documents))}/{len(documents)}")

    print(f"\n🎉 Xong! {len(documents)} chunks → '{COLLECTION_NAME}'")

    # Test nhanh
    res = col.query(query_texts=["fix lỗi pandas-ta"], n_results=1)
    if res["documents"][0]:
        meta = res["metadatas"][0][0]
        print(f"\n🔍 Test search 'fix lỗi pandas-ta':")
        print(f"   → {meta['source']} — {meta['heading']}")


def main():
    if not check_chromadb():
        sys.exit(1)
    if not DOCS_DIR.exists():
        print(f"❌ Không tìm thấy docs/ tại: {DOCS_DIR}")
        sys.exit(1)

    docs = load_and_chunk_docs(DOCS_DIR)
    if not docs:
        print("⚠️  Không có document.")
        sys.exit(0)

    index(docs)
    print(f"\n✅ ChromaDB sẵn sàng tại: {DB_PATH}")
    print("   Xem cấu hình MCP → docs/mcp_setup.md")


if __name__ == "__main__":
    main()
