"""Read-only 360 public web search for Anon's researcher specialist."""
import html
import os
import re
import time
import urllib.parse
import urllib.request

TIMEOUT = float(os.getenv("ANON_SEARCH_TIMEOUT", "12"))
CACHE_TTL = 600
_cache: dict[str, tuple[float, str | None]] = {}
_headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.so.com/",
}


def _strip_html(value: str) -> str:
    value = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", value, flags=re.S | re.I)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def _log_safe(text: str) -> str:
    # query 可能来自群友原话：日志行里压平换行/控制符，防止伪造 [ANON ...] 标记行
    return re.sub(r"[\x00-\x1f\x7f]+", " ", text)


def search(query: str, max_results: int = 4) -> str | None:
    query = (query or "").strip()[:80]
    if not query:
        return None
    now = time.time()
    cached = _cache.get(query)
    if cached and now - cached[0] < CACHE_TTL:
        return cached[1]
    result = None
    try:
        url = "https://www.so.com/s?q=" + urllib.parse.quote(query) + "&pn=1"
        request = urllib.request.Request(url, headers=_headers)
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            page = response.read().decode("utf-8", "replace")
        items = []
        for block in re.split(r'<li class="res-list', page)[1:]:
            title_match = re.search(r"<h3[^>]*>\s*<a[^>]*>(.*?)</a>", block, re.S | re.I)
            if not title_match:
                continue
            title = _strip_html(title_match.group(1))
            if not title:
                continue
            descriptions = [
                _strip_html(match.group(1))
                for match in re.finditer(r"<p[^>]*>(.*?)</p>", block, re.S | re.I)
            ]
            description = max(descriptions, key=len, default="")
            if len(description) < 15:
                description = ""
            items.append(f"- {title[:80]}" + (f"：{description[:180]}" if description else ""))
            if len(items) >= max(1, min(max_results, 5)):
                break
        if items:
            result = "【公开网页搜索摘要】\n" + "\n".join(items)
            print(f"[ANON SEARCH] q={_log_safe(query)} hits={len(items)}", flush=True)
        else:
            print(f"[ANON SEARCH] q={_log_safe(query)} hits=0", flush=True)
    except Exception as exc:
        print(f"[ANON SEARCH] {type(exc).__name__}: {exc}", flush=True)
    _cache[query] = (now, result)
    if len(_cache) > 100:
        for old_query in list(_cache)[:50]:
            _cache.pop(old_query, None)
    return result
