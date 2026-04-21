"""
macro_agent.py — Phân tích vĩ mô toàn cầu + VN, cache 1 lần/ngày.

Chạy 1 lần/ngày lúc 06:00 (morning_fetch) → lưu cache/macro_{date}.json
Orchestrator load từ cache — không gọi LLM lại trong ngày.

Input:
  - fetcher.get_global_macro()  → S&P500, DXY, dầu, vàng, Nikkei, KOSPI, HSI
  - fetcher.get_vn_macro()      → USD/VND, lãi suất SBV
  - Headlines crawl từ VnExpress + CafeF (5-10 bài)

Output (cache/macro_{date}.json):
  macro_bias, macro_score, affected_sectors, beneficiary_sectors,
  key_risk, key_risks, key_supports, reasoning, time_horizon,
  global_summary, vn_summary, overall_summary, headlines_used
"""

import importlib.metadata  # FIX: pandas-ta-openbb AttributeError Python 3.11
import json
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from multiagents_trading_assistant.agent import run_agent_lite
from multiagents_trading_assistant import fetcher


BASE_DIR   = Path(__file__).parent.parent
CACHE_DIR  = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9",
}
_TIMEOUT = 10


# ──────────────────────────────────────────────
# System prompt — tư duy chuyên gia nhân quả
# ──────────────────────────────────────────────

_SYSTEM_PROMPT = """Bạn là chuyên gia kinh tế vĩ mô chuyên phân tích tác động đến thị trường chứng khoán Việt Nam.

=== TƯ DUY PHÂN TÍCH ===
Luôn phân tích theo chuỗi nhân quả: SỰ KIỆN → CƠ CHẾ TRUYỀN DẪN → HẬU QUẢ VỚI VN

4 kênh truyền dẫn cần xem xét:
1. GIÁ HÀNG HÓA: dầu↑ → chi phí sx tăng → lạm phát VN↑ → lãi suất↑ → P/E ngành bất động sản↓
2. DÒNG VỐN QUỐC TẾ: DXY↑ → USD mạnh → khối ngoại rút tiền khỏi EM → VN-Index áp lực bán
3. CHUỖI CUNG ỨNG: Nikkei↓ → Nhật giảm đơn hàng → xuất khẩu VN (điện tử, dệt may) ảnh hưởng
4. TÂM LÝ THỊ TRƯỜNG: S&P500↓ mạnh → fear contagion → tâm lý xấu lan sang VN ngay cả khi macro VN tốt

Với mỗi sự kiện, luôn tự hỏi:
- Tác động TRỰC TIẾP đến VN là gì?
- Tác động GIÁN TIẾP qua chuỗi truyền dẫn?
- NGẮN HẠN (1-2 tuần) vs DÀI HẠN (1-3 tháng) khác nhau thế nào?
- Ngành nào VN bị ảnh HƯỞng (affected) và ngành nào HƯỞNG LỢI (beneficiary)?

=== QUY TẮC OUTPUT ===
- Trả về JSON hợp lệ DUY NHẤT, không có text thừa
- macro_bias: "BULLISH" | "NEUTRAL" | "BEARISH"
  * BULLISH: đa số tín hiệu thuận, dòng vốn vào EM, hàng hóa ổn, tâm lý tốt
  * NEUTRAL: tín hiệu trái chiều, không rõ xu hướng
  * BEARISH: ≥2 trong 4 kênh tiêu cực, hoặc 1 kênh tiêu cực rất mạnh
- macro_score: -2 (rất xấu) → -1 → 0 → +1 → +2 (rất tốt)
- confidence: "LOW" | "MEDIUM" | "HIGH" (dựa trên số lượng tín hiệu nhất quán)
- affected_sectors: ngành VN bị ảnh hưởng tiêu cực (list tiếng Việt)
- beneficiary_sectors: ngành VN hưởng lợi (list tiếng Việt)
- key_risk: 1 câu ngắn mô tả rủi ro chính cần theo dõi
- key_risks: list 2-4 rủi ro (tiếng Việt)
- key_supports: list 2-4 yếu tố hỗ trợ (tiếng Việt)
- reasoning: chuỗi nhân quả đầy đủ, trình bày theo 4 kênh
- time_horizon: "1-2 tuần" | "2-4 tuần" | "1-3 tháng"
- global_summary: tóm tắt thị trường toàn cầu 1-2 câu
- vn_summary: tóm tắt macro VN 1-2 câu
- overall_summary: kết luận tổng hợp 1-2 câu
- headlines_used: list tiêu đề bài báo đã dùng để phân tích
- expert_consensus: "BULLISH" | "MIXED" | "BEARISH" — đồng thuận của chuyên gia
- experts_cited: list tên nguồn chuyên gia đã tham khảo

=== XỬ LÝ Ý KIẾN CHUYÊN GIA ===
Ý kiến chuyên gia được cung cấp kèm reliability score (0-1).
Đánh giá cao hơn nguồn có score cao và khi nhiều nguồn độc lập đồng thuận.
Broker reports có thể có conflict of interest — cần xem xét thận trọng.
Nếu chuyên gia score cao (≥0.85) đồng thuận BEARISH → tăng trọng số phán đoán BEARISH.

Output schema:
{
  "macro_bias": <str>,
  "macro_score": <int -2 đến 2>,
  "confidence": <str>,
  "affected_sectors": [<str>, ...],
  "beneficiary_sectors": [<str>, ...],
  "key_risk": <str>,
  "key_risks": [<str>, ...],
  "key_supports": [<str>, ...],
  "reasoning": <str>,
  "time_horizon": <str>,
  "global_summary": <str>,
  "vn_summary": <str>,
  "overall_summary": <str>,
  "headlines_used": [<str>, ...],
  "expert_consensus": <str>,
  "experts_cited": [<str>, ...]
}"""


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def get_macro_context(date: str | None = None, force_refresh: bool = False) -> dict:
    """
    Lấy macro context cho ngày hôm nay.
    Cache 1 lần/ngày — không gọi lại LLM nếu cache đã có.

    Args:
        date:          Ngày phân tích (YYYY-MM-DD), mặc định hôm nay
        force_refresh: Bỏ qua cache, gọi lại LLM

    Returns:
        dict macro context theo schema agents.md #1
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    cache_path = CACHE_DIR / f"macro_{date}.json"

    # Load cache nếu đã có và không force refresh
    if not force_refresh and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            print(f"[macro_agent] Cache hit macro_{date}.json — bias={cached.get('macro_bias')}")
            return cached
        except Exception as e:
            print(f"[macro_agent] Lỗi đọc cache: {e} — sẽ fetch lại")

    print(f"[macro_agent] Bắt đầu phân tích vĩ mô ngày {date}...")

    # ── Fetch dữ liệu ──
    global_macro = fetcher.get_global_macro()
    vn_macro     = fetcher.get_vn_macro()
    headlines    = _crawl_macro_headlines()

    # ── Lớp 3: Ý kiến chuyên gia ──
    expert_opinions = _fetch_filtered_expert_opinions(date)

    # ── Build prompt ──
    prompt = _build_prompt(global_macro, vn_macro, headlines, date, expert_opinions)

    # ── Gọi LLM ──
    try:
        result = run_agent_lite(prompt=prompt, system=_SYSTEM_PROMPT)
        result["date"] = date

        # Đảm bảo có 2 field mới ngay cả khi LLM bỏ sót
        result.setdefault("expert_consensus", "MIXED")
        result.setdefault("experts_cited", [op["source"] for op in expert_opinions])

        # Log BEARISH rõ ràng ra terminal
        if result.get("macro_bias") == "BEARISH":
            print("\n" + "="*60)
            print(f"[macro_agent] ⚠️  MACRO BEARISH — DỪNG PIPELINE")
            print(f"[macro_agent] Key risk: {result.get('key_risk','')}")
            print(f"[macro_agent] Reasoning: {result.get('reasoning','')[:300]}...")
            print(f"[macro_agent] Key risks: {result.get('key_risks',[])}")
            print("="*60 + "\n")
        else:
            print(f"[macro_agent] Macro bias={result.get('macro_bias')}, "
                  f"score={result.get('macro_score')}, confidence={result.get('confidence')}")
            print(f"[macro_agent] Beneficiary: {result.get('beneficiary_sectors',[])}")
            print(f"[macro_agent] Affected:    {result.get('affected_sectors',[])}")
            print(f"[macro_agent] Expert consensus: {result.get('expert_consensus')} "
                  f"({len(expert_opinions)} nguồn)")

        # Lưu cache
        _save_cache(cache_path, result)

        # ── Lưu tin vào database + cập nhật stats nguồn ──
        _persist_expert_news_to_db(expert_opinions, date)

        return result

    except Exception as e:
        print(f"[macro_agent] Lỗi LLM: {e} — fallback NEUTRAL")
        fallback = _neutral_fallback(date, global_macro, vn_macro)
        _save_cache(cache_path, fallback)
        return fallback


# ──────────────────────────────────────────────
# Crawl headlines
# ──────────────────────────────────────────────

def _crawl_macro_headlines(max_headlines: int = 10) -> list[str]:
    """
    Crawl 5-10 headlines kinh tế vĩ mô từ VnExpress + CafeF.
    Trả về list tiêu đề string.
    """
    headlines: list[str] = []

    # VnExpress — chuyên mục kinh doanh
    try:
        resp = requests.get(
            "https://vnexpress.net/kinh-doanh",
            headers=_HEADERS, timeout=_TIMEOUT
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Lấy tiêu đề các bài trong trang chủ kinh doanh
        for tag in soup.select("h3.title-news a, h2.title-news a, .title-news a")[:8]:
            title = tag.get_text(strip=True)
            if title and len(title) > 15:
                headlines.append(f"[VnExpress] {title}")

        print(f"[macro_agent] VnExpress: {len(headlines)} headlines")
    except Exception as e:
        print(f"[macro_agent] Lỗi crawl VnExpress: {e}")

    time.sleep(1)

    # CafeF — trang chủ tin kinh tế
    try:
        resp = requests.get(
            "https://cafef.vn/kinh-te-vi-mo.chn",
            headers=_HEADERS, timeout=_TIMEOUT
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        cafef_count = 0
        for tag in soup.select("h3 a, h2 a, .title a")[:10]:
            title = tag.get_text(strip=True)
            if title and len(title) > 15 and cafef_count < 5:
                headlines.append(f"[CafeF] {title}")
                cafef_count += 1

        print(f"[macro_agent] CafeF: {cafef_count} headlines")
    except Exception as e:
        print(f"[macro_agent] Lỗi crawl CafeF: {e}")

    return headlines[:max_headlines]


# ──────────────────────────────────────────────
# Build prompt
# ──────────────────────────────────────────────

def _build_prompt(
    global_macro: dict,
    vn_macro: dict,
    headlines: list[str],
    date: str,
    expert_opinions: list[dict] | None = None,
) -> str:
    """Tổng hợp dữ liệu vào prompt cho LLM."""

    # Format global macro
    def _fmt_asset(name: str, data: dict) -> str:
        cur = data.get("current")
        chg = data.get("change_pct")
        if cur is None:
            return f"{name}: N/A"
        chg_str = f"{chg:+.2f}%" if chg is not None else "?"
        return f"{name}: {cur:,.2f} ({chg_str})"

    sp500   = global_macro.get("sp500",  {})
    dxy     = global_macro.get("dxy",    {})
    oil     = global_macro.get("oil",    {})
    gold    = global_macro.get("gold",   {})
    nikkei  = global_macro.get("nikkei", {})
    kospi   = global_macro.get("kospi",  {})
    hsi     = global_macro.get("hsi",    {})

    usd_vnd = vn_macro.get("usd_vnd")
    sbv     = vn_macro.get("sbv_rate")

    headlines_text = "\n".join(f"  - {h}" for h in headlines) if headlines else "  (Không lấy được headlines)"

    # Format expert opinions
    if expert_opinions:
        expert_lines = []
        for op in expert_opinions:
            src   = op.get("source", "?")
            rel   = op.get("reliability", 0.0)
            title = op.get("headline", "")[:120]
            body  = op.get("content", "")[:200]
            line  = f"  - {src} (reliability {rel:.1f}): {title}"
            if body and body != title:
                line += f" — {body}"
            expert_lines.append(line)
        expert_text = "\n".join(expert_lines)
    else:
        expert_text = "  (Không có ý kiến chuyên gia)"

    return f"""Phân tích vĩ mô cho thị trường chứng khoán Việt Nam ngày {date}.

=== THỊ TRƯỜNG TOÀN CẦU ===
{_fmt_asset("S&P500",  sp500)}
{_fmt_asset("DXY",     dxy)}
{_fmt_asset("Dầu WTI", oil)}
{_fmt_asset("Vàng",    gold)}
{_fmt_asset("Nikkei",  nikkei)}
{_fmt_asset("KOSPI",   kospi)}
{_fmt_asset("HSI",     hsi)}

=== VĨ MÔ VIỆT NAM ===
USD/VND: {f"{usd_vnd:,.0f}" if usd_vnd else "N/A"}
Lãi suất SBV: {sbv}%

=== HEADLINES KINH TẾ GẦN NHẤT ===
{headlines_text}

=== LỚP 3 — Ý KIẾN CHUYÊN GIA ===
{expert_text}

=== YÊU CẦU PHÂN TÍCH ===
Phân tích theo 4 kênh truyền dẫn:
1. Giá hàng hóa → lạm phát → lãi suất VN → định giá cổ phiếu
2. DXY/dòng vốn → khối ngoại rút/vào EM → VN-Index
3. Châu Á (Nikkei/KOSPI/HSI) → chuỗi cung ứng → xuất khẩu VN
4. Tâm lý S&P500 → contagion → tâm lý VN

Xác định:
- macro_bias tổng hợp (BULLISH/NEUTRAL/BEARISH)
- Ngành VN nào bị ảnh hưởng xấu (affected_sectors)
- Ngành VN nào hưởng lợi (beneficiary_sectors)
- Rủi ro chính cần theo dõi (key_risk)
- Time horizon phù hợp để ra quyết định

Trả về JSON theo schema đã định (bao gồm expert_consensus và experts_cited)."""


# ──────────────────────────────────────────────
# Lớp 3 — Expert opinions helpers
# ──────────────────────────────────────────────

def _fetch_filtered_expert_opinions(date: str) -> list[dict]:
    """
    Lấy tin chuyên gia từ news_fetcher, lọc bỏ tin kém tin cậy.

    Quy tắc lọc:
      - credibility_score < 0.5  → bỏ
      - is_suspicious = True     → bỏ

    Trả về list expert opinion đã lọc, sắp xếp reliability giảm dần.
    """
    try:
        from multiagents_trading_assistant.news_fetcher import (
            get_expert_news,
            evaluate_news_credibility,
        )

        raw_opinions = get_expert_news(date=date, max_articles=15)
        if not raw_opinions:
            print("[macro_agent] Không có tin chuyên gia")
            return []

        filtered: list[dict] = []
        for op in raw_opinions:
            try:
                cred = evaluate_news_credibility(op)
                if cred.get("is_suspicious") or cred.get("credibility_score", 0) < 0.5:
                    print(f"[macro_agent] Lọc bỏ tin nghi ngờ: "
                          f"{op.get('source')} — {op.get('headline','')[:60]} "
                          f"(score={cred.get('credibility_score',0):.2f})")
                    continue
                # Gắn credibility info vào item để dùng sau
                op["_credibility"] = cred
                filtered.append(op)
            except Exception as e:
                print(f"[macro_agent] Lỗi evaluate tin: {e}")
                continue

        # Sắp xếp theo reliability giảm dần
        filtered.sort(key=lambda x: x.get("reliability", 0), reverse=True)
        print(f"[macro_agent] Expert opinions: {len(filtered)}/{len(raw_opinions)} tin sau lọc")
        return filtered

    except Exception as e:
        print(f"[macro_agent] Lỗi lấy expert opinions: {e}")
        return []


def _persist_expert_news_to_db(expert_opinions: list[dict], date: str) -> None:
    """
    Lưu các tin chuyên gia đã dùng vào database.news_history.
    Cập nhật source_credibility stats cho từng nguồn.
    """
    if not expert_opinions:
        return

    try:
        from multiagents_trading_assistant.database import save_news, update_source_stats

        sources_updated: set[str] = set()

        for op in expert_opinions:
            cred = op.get("_credibility", {})
            news_item = {
                "date":             date,
                "source":           op.get("source", ""),
                "url":              op.get("url", ""),
                "symbol":           (op.get("symbols_mentioned") or ["MARKET"])[0],
                "headline":         op.get("headline", ""),
                "content":          op.get("content", ""),
                "sentiment":        "NEUTRAL",  # macro news — không phân biệt positive/negative ở đây
                "price_at_publish": None,
                "is_suspicious":    cred.get("is_suspicious", False),
            }
            try:
                save_news(news_item)
            except Exception as e:
                print(f"[macro_agent] Lỗi save_news: {e}")
                continue

            sources_updated.add(op.get("source", ""))

        # Cập nhật stats cho từng nguồn đã dùng
        for source in sources_updated:
            try:
                update_source_stats(source)
            except Exception as e:
                print(f"[macro_agent] Lỗi update_source_stats({source}): {e}")

        print(f"[macro_agent] Đã lưu {len(expert_opinions)} tin + update {len(sources_updated)} nguồn")

    except Exception as e:
        print(f"[macro_agent] Lỗi persist to DB: {e}")


# ──────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────

def _save_cache(path: Path, data: dict) -> None:
    try:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8"
        )
        print(f"[macro_agent] Đã lưu cache → {path.name}")
    except Exception as e:
        print(f"[macro_agent] Lỗi lưu cache: {e}")


def _neutral_fallback(date: str, global_macro: dict, vn_macro: dict) -> dict:
    """Trả về NEUTRAL khi LLM fail."""
    sp500_chg = global_macro.get("sp500", {}).get("change_pct")
    dxy_chg   = global_macro.get("dxy",   {}).get("change_pct")

    return {
        "date":                date,
        "macro_bias":          "NEUTRAL",
        "macro_score":         0,
        "confidence":          "LOW",
        "affected_sectors":    [],
        "beneficiary_sectors": [],
        "key_risk":            "Không phân tích được macro — cần kiểm tra thủ công.",
        "key_risks":           ["LLM không phản hồi"],
        "key_supports":        [],
        "reasoning":           f"Fallback NEUTRAL. S&P500 {sp500_chg:+.2f}%, DXY {dxy_chg:+.2f}%." if sp500_chg and dxy_chg else "Fallback NEUTRAL.",
        "time_horizon":        "1-2 tuần",
        "global_summary":      f"S&P500 {sp500_chg:+.2f}%, DXY {dxy_chg:+.2f}%." if sp500_chg and dxy_chg else "Không có dữ liệu.",
        "vn_summary":          f"USD/VND {vn_macro.get('usd_vnd','N/A')}, SBV {vn_macro.get('sbv_rate','N/A')}%.",
        "overall_summary":     "LLM lỗi — dùng fallback NEUTRAL. Cần kiểm tra thủ công.",
        "headlines_used":      [],
        "expert_consensus":    "MIXED",
        "experts_cited":       [],
    }


# ──────────────────────────────────────────────
# Test nhanh
# ──────────────────────────────────────────────

if __name__ == "__main__":
    result = get_macro_context()
    print("\n=== Macro Context ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
