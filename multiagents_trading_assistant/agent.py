"""
agent.py — Base LLM runner cho toàn bộ hệ thống.

2 functions chính:
  run_agent()      → Sonnet (Debate + Trader, cần reasoning sâu)
  run_agent_lite() → Haiku  (PTKT / FA / Sentiment / ForeignFlow)

Backend: Anthropic API (claude-sonnet / claude-haiku)
  - Prompt caching trên system prompt → tiết kiệm ~90% input token
  - Retry tự động qua anthropic SDK (max_retries=3)
  - Output luôn là dict (JSON parsed)
  - Log tiếng Việt ra terminal
"""

import json
import os
import re

import anthropic
from dotenv import load_dotenv

# LangSmith tracing — optional, graceful fallback nếu chưa cài hoặc chưa set key
try:
    from langsmith import traceable
except ImportError:
    def traceable(fn=None, **kwargs):   # noqa: E301
        """Passthrough decorator khi langsmith chưa được cài."""
        if fn is not None:
            return fn
        return lambda f: f

# Load .env từ thư mục gốc project
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Model theo CLAUDE.md §5 ──
MODEL_SONNET = os.getenv("MODEL_SONNET", "claude-sonnet-4-5")         # Debate, Trader
MODEL_HAIKU  = os.getenv("MODEL_HAIKU",  "claude-haiku-4-5-20251001")  # PTKT, FA, Sentiment, ForeignFlow

# ── Giới hạn token ──
MAX_TOKENS_SONNET = 4096
MAX_TOKENS_HAIKU  = 4096


def _get_client() -> anthropic.Anthropic:
    """Tạo Anthropic client. API key lấy từ ANTHROPIC_API_KEY env."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY chưa được cấu hình trong .env")
    # SDK tự retry 429 và 5xx với exponential backoff (max_retries=3 mặc định)
    return anthropic.Anthropic(api_key=api_key, max_retries=3)


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

    raise ValueError(f"Khong parse duoc JSON tu response:\n{text[:300]}...")


# ──────────────────────────────────────────────
# run_agent — Sonnet (Debate, Trader)
# ──────────────────────────────────────────────

@traceable(run_type="llm", name="claude-sonnet | Trader/Debate")
def run_agent(
    prompt: str,
    system: str,
    model: str = MODEL_SONNET,
    max_tokens: int = MAX_TOKENS_SONNET,
) -> dict:
    """
    Gọi Sonnet — dùng cho Debate và Trader (cần reasoning sâu).

    System prompt được cache (cache_control) → tái sử dụng nhiều lần tiết kiệm cost.

    Args:
        prompt:     Nội dung câu hỏi / dữ liệu đầu vào (thay đổi theo từng mã)
        system:     System prompt định nghĩa vai trò agent (ổn định → cache được)
        model:      Model ID
        max_tokens: Giới hạn token output

    Returns:
        dict — JSON response đã parse
    """
    print(f"[agent] Goi Anthropic {model}...")
    client = _get_client()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,
            # Cache system prompt — tiết kiệm ~90% token khi gọi nhiều lần
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        raw_text = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        usage = response.usage
        print(
            f"[agent] Xong — input={usage.input_tokens} "
            f"cache_hit={getattr(usage, 'cache_read_input_tokens', 0)} "
            f"output={usage.output_tokens} tokens"
        )

        return _parse_json_response(raw_text)

    except anthropic.BadRequestError as e:
        print(f"[agent] BadRequest: {e.message}")
        raise
    except anthropic.RateLimitError:
        print("[agent] Rate limit — SDK se tu dong retry...")
        raise
    except anthropic.APIStatusError as e:
        print(f"[agent] API error {e.status_code}: {e.message}")
        raise
    except ValueError as e:
        print(f"[agent] Loi parse JSON: {e}")
        raise RuntimeError(f"Agent tra ve format khong hop le: {e}") from e


# ──────────────────────────────────────────────
# run_agent_lite — Haiku (PTKT, FA, Sentiment, ForeignFlow)
# ──────────────────────────────────────────────

@traceable(run_type="llm", name="claude-haiku | Analyst")
def run_agent_lite(
    prompt: str,
    system: str,
    model: str = MODEL_HAIKU,
    max_tokens: int = MAX_TOKENS_HAIKU,
) -> dict:
    """
    Gọi Haiku — dùng cho các agent nhẹ: PTKT, FA, ForeignFlow, Sentiment.

    Nhanh + rẻ + system prompt được cache.

    Args:
        prompt:     Data input (thay đổi theo từng mã)
        system:     System prompt định nghĩa vai trò agent (ổn định → cache được)
        model:      Model ID
        max_tokens: Giới hạn token output

    Returns:
        dict — JSON response đã parse
    """
    print(f"[agent_lite] Goi Anthropic {model}...")
    client = _get_client()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.2,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        raw_text = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        usage = response.usage
        print(
            f"[agent_lite] Xong — input={usage.input_tokens} "
            f"cache_hit={getattr(usage, 'cache_read_input_tokens', 0)} "
            f"output={usage.output_tokens} tokens"
        )

        return _parse_json_response(raw_text)

    except anthropic.BadRequestError as e:
        print(f"[agent_lite] BadRequest: {e.message}")
        raise
    except anthropic.RateLimitError:
        print("[agent_lite] Rate limit — SDK se tu dong retry...")
        raise
    except anthropic.APIStatusError as e:
        print(f"[agent_lite] API error {e.status_code}: {e.message}")
        raise
    except ValueError as e:
        print(f"[agent_lite] Loi parse JSON: {e}")
        raise RuntimeError(f"Agent lite tra ve format khong hop le: {e}") from e


# ──────────────────────────────────────────────
# Chạy trực tiếp để test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== Test Anthropic API ===")
    print(f"Sonnet: {MODEL_SONNET}")
    print(f"Haiku:  {MODEL_HAIKU}")

    print("\n--- Test run_agent_lite (Haiku) ---")
    result = run_agent_lite(
        system="Ban la assistant phan tich tai chinh. Tra loi bang JSON hop le duy nhat, khong them text ngoai JSON.",
        prompt='Cho toi JSON don gian: {"status": "ok", "message": "agent_lite hoat dong"}',
    )
    print("Ket qua:", result)

    print("\n--- Test run_agent (Sonnet) ---")
    result = run_agent(
        system="Ban la assistant phan tich tai chinh. Tra loi bang JSON hop le duy nhat, khong them text ngoai JSON.",
        prompt='Cho toi JSON don gian: {"status": "ok", "message": "agent hoat dong"}',
    )
    print("Ket qua:", result)
