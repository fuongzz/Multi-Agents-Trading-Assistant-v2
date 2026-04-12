# MCP Setup — ChromaDB cho Claude Code

## Mục đích

Sau khi index `docs/` vào ChromaDB, Claude Code có thể **tự search tài liệu** trong lúc code
mà không cần bạn nhắc. Ví dụ: gặp lỗi pandas-ta → Claude Code tự tìm BUG-001 trong `bugs_and_fixes.md`.

---

## Bước 1 — Cài dependencies

```powershell
py -3.11 -m pip install chromadb
npm install -g @modelcontextprotocol/server-chroma
```

Node.js chưa có: https://nodejs.org (LTS)

---

## Bước 2 — Index docs lần đầu

```powershell
cd AI-Trading-Assistant
py -3.11 scripts/index_docs.py
```

Tạo thư mục `chroma_db/` (đã gitignore).

---

## Bước 3 — Cấu hình Claude Code

Tạo file `.claude/mcp_settings.json` ở root project:

```json
{
  "mcpServers": {
    "chroma": {
      "command": "npx",
      "args": [
        "@modelcontextprotocol/server-chroma",
        "--path", "C:\\Users\\<username>\\Documents\\AI-Trading-Assistant\\chroma_db",
        "--collection", "trading_docs"
      ]
    }
  }
}
```

Thay `<username>` bằng tên user Windows.

---

## Bước 4 — Verify

Trong Claude Code hỏi:
```
Search docs: cách fix lỗi pandas-ta?
```
→ Nếu trả lời đúng nội dung BUG-001 → thành công.

---

## Workflow hàng ngày

```
Thêm doc mới vào docs/
  → Cập nhật bảng 📚 trong CLAUDE.md
  → py -3.11 scripts/index_docs.py    ← re-index
```

`chroma_db/` gitignored — mỗi máy cần chạy index_docs.py lần đầu.
