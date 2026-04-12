"""
agent.py — Base LLM runner cho toàn bộ hệ thống.

2 functions chính:
  run_agent()      → Sonnet, dùng cho Debate + Trader (cần reasoning sâu)
  run_agent_lite() → Haiku + prompt caching, dùng cho PTKT / FA / Sentiment / ...

Cả 2:
  - Load API key từ .env
  - Retry 3 lần với exponential backoff (tenacity)
  - Output luôn là dict (JSON parsed)
  - Log tiếng Việt ra terminal
"""

import json
import os
import re

import anthropic
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Load .env từ thư mục gốc project
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Model constants (theo CLAUDE.md) ──
MODEL_SONNET = "claude-sonnet-4-5"
MODEL_HAIKU  = "claude-haiku-4-5-20251001"

# ── Giới hạn token ──
MAX_TOKENS_SONNET = 4096
MAX_TOKENS_HAIKU  = 4096


def _get_client() -> anthropic.Anthropic:
    """Tạo Anthropic client từ API key trong .env."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Thiếu ANTHROPIC_API_KEY — kiểm tra file .env"
        )
    return anthropic.Anthropic(api_key=api_key)


def _parse_json_response(text: str) -> dict:
    """
    Parse JSON từ response LLM.
    Xử lý cả trường hợp LLM bọc JSON trong ```json ... ``` code block.
    """
    # Thử parse thẳng trước
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Tìm JSON block trong markdown
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Tìm object JSON đầu tiên trong text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Không parse được JSON từ response:\n{text[:300]}...")


# ──────────────────────────────────────────────
# run_agent — Sonnet (Debate, Trader)
# ──────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((anthropic.APIError, anthropic.APITimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    reraise=True,
)
def run_agent(
    prompt: str,
    system: str,
    model: str = MODEL_SONNET,
    max_tokens: int = MAX_TOKENS_SONNET,
) -> dict:
    """
    Gọi Sonnet agent — dùng cho Debate và Trader (cần reasoning sâu).

    Args:
        prompt:     Nội dung câu hỏi / dữ liệu đầu vào
        system:     System prompt định nghĩa vai trò agent
        model:      Model ID (mặc định Sonnet)
        max_tokens: Giới hạn token output

    Returns:
        dict — JSON response đã parse

    Raises:
        RuntimeError nếu fail sau 3 lần retry
    """
    print(f"[agent] Gọi {model.split('-')[1].upper()} ({model})...")
    client = _get_client()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.content[0].text
        print(f"[agent] Nhận response — {response.usage.output_tokens} tokens output")

        result = _parse_json_response(raw_text)
        return result

    except (anthropic.APIError, anthropic.APITimeoutError) as e:
        print(f"[agent] Lỗi API ({type(e).__name__}): {e} — sẽ retry...")
        raise
    except ValueError as e:
        # Parse JSON fail → không retry, raise ngay
        print(f"[agent] Lỗi parse JSON: {e}")
        raise RuntimeError(f"Agent trả về format không hợp lệ: {e}") from e


# ──────────────────────────────────────────────
# run_agent_lite — Haiku + prompt caching
# ──────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((anthropic.APIError, anthropic.APITimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    reraise=True,
)
def run_agent_lite(
    prompt: str,
    system: str,
    model: str = MODEL_HAIKU,
    max_tokens: int = MAX_TOKENS_HAIKU,
) -> dict:
    """
    Gọi Haiku agent với prompt caching — dùng cho PTKT, FA, ForeignFlow, Sentiment.

    Prompt caching hoạt động: system prompt được cache sau lần gọi đầu,
    các lần sau chỉ tốn token cho phần thay đổi (data input).
    → Tiết kiệm ~80% chi phí khi chạy batch nhiều mã cùng ngày.

    Args:
        prompt:     Data input (thay đổi theo từng mã)
        system:     System prompt (cache sau lần đầu)
        model:      Model ID (mặc định Haiku)
        max_tokens: Giới hạn token output

    Returns:
        dict — JSON response đã parse
    """
    print(f"[agent_lite] Gọi Haiku ({model})...")
    client = _get_client()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            # Bật prompt caching cho system prompt
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.content[0].text

        # Log cache hit/miss để theo dõi chi phí
        usage = response.usage
        cache_read    = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_created = getattr(usage, "cache_creation_input_tokens", 0) or 0

        if cache_read > 0:
            print(f"[agent_lite] Cache HIT — đọc {cache_read} tokens từ cache ✓")
        elif cache_created > 0:
            print(f"[agent_lite] Cache MISS — tạo cache {cache_created} tokens")
        else:
            print(f"[agent_lite] Output: {usage.output_tokens} tokens")

        result = _parse_json_response(raw_text)
        return result

    except (anthropic.APIError, anthropic.APITimeoutError) as e:
        print(f"[agent_lite] Lỗi API ({type(e).__name__}): {e} — sẽ retry...")
        raise
    except ValueError as e:
        print(f"[agent_lite] Lỗi parse JSON: {e}")
        raise RuntimeError(f"Agent lite trả về format không hợp lệ: {e}") from e


# ──────────────────────────────────────────────
# Chạy trực tiếp để test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Test run_agent_lite (Haiku) ===")
    result = run_agent_lite(
        system="Bạn là assistant phân tích tài chính. Trả lời bằng JSON hợp lệ.",
        prompt='Cho tôi JSON đơn giản: {"status": "ok", "message": "Haiku hoạt động"}',
    )
    print("Kết quả:", result)

    print("\n=== Test run_agent (Sonnet) ===")
    result = run_agent(
        system="Bạn là assistant phân tích tài chính. Trả lời bằng JSON hợp lệ.",
        prompt='Cho tôi JSON đơn giản: {"status": "ok", "message": "Sonnet hoạt động"}',
    )
    print("Kết quả:", result)
