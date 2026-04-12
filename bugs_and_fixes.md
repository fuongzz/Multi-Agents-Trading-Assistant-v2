# Bugs & Fixes

> Đọc file này TRƯỚC KHI DEBUG bất cứ thứ gì.
> Gặp bug mới → thêm vào đây + cập nhật bảng 📚 trong CLAUDE.md.

---

## Template

```
### [BUG-XXX] Tên lỗi
**Date:** YYYY-MM-DD | **File:** file.py
**Error:** (paste error)
**Root cause:** ...
**Fix:** (code)
```

---

### [BUG-001] pandas-ta-openbb AttributeError — Python 3.11
**Date:** 2026-04-12 | **File:** Mọi file import `pandas_ta`

**Error:**
```
AttributeError: module 'importlib.metadata' has no attribute ...
```
**Root cause:** pandas-ta-openbb dùng API importlib.metadata không tương thích Python 3.11
**Fix:**
```python
import importlib.metadata  # PHẢI có TRƯỚC khi import pandas_ta
import pandas_ta as ta
```

---

### [BUG-002] `applymap` deprecated — pandas 3.x + vnstock
**Date:** 2026-04-12 | **File:** `fetcher.py`

**Error:**
```
AttributeError: 'DataFrame' object has no attribute 'applymap'
```
**Root cause:** pandas 3.x đổi `applymap` → `map`. vnstock vẫn dùng `applymap`.
**Fix:**
```python
if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = pd.DataFrame.map
```
Đặt ở đầu `fetcher.py`, trước `from vnstock import ...`

---

### [BUG-003] Vnstock rate limit — HTTP 429
**Date:** 2026-04-12 | **File:** `fetcher.py`

**Error:**
```
requests.exceptions.HTTPError: 429 Too Many Requests
```
**Root cause:** Free tier ~15 requests/phút
**Fix:**
```python
if i > 0 and i % 15 == 0:
    time.sleep(60)
```
