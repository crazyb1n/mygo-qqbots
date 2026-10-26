"""websearch：给若叶睦的搜索工具（360 搜索后端，免 key）。

背景（2026-09-28 实测）：本机网络环境下必应/百度/搜狗/百度百科的爬取都被反爬拦截
（空壳页/安全验证/403），DuckDuckGo 在境内不可达；唯独 360 搜索（so.com）返回可解析的
服务端渲染结果页。就用它做唯一后端，失败就如实告诉上层"没搜到"，绝不编造。

用法：search("坦克世界 装填机") -> Optional[str]
返回注入提示词的文本块（【搜索结果】+ 若干条 标题：摘要），没搜到返回 None。
同关键词 10 分钟内走缓存；每次调用有超时，失败不抛异常。
"""
import re
import time
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.so.com/",
}
TIMEOUT = float(__import__("os").getenv("SEARCH_TIMEOUT", "12"))
CACHE_TTL = 600.0
_cache: dict = {}   # query -> (ts, result_or_None)


def _strip_html(s: str) -> str:
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s.replace("&nbsp;", " ")).strip()


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read().decode("utf-8", "replace")


def _parse(html: str, max_results: int) -> List[Tuple[str, str]]:
    """解析 so.com 结果页：<li class="res-list"> 块 -> (标题, 摘要)。摘要取块内最长的 <p>。"""
    out: List[Tuple[str, str]] = []
    blocks = re.split(r'<li class="res-list', html)[1:]
    for b in blocks:
        mt = re.search(r"<h3[^>]*>\s*<a[^>]*>(.*?)</a>", b, re.S)
        if not mt:
            continue
        title = _strip_html(mt.group(1))
        # 标题里混进的日期戳（"2025年12月15日 - "）去掉
        title = re.sub(r"^\d{4}年\d{1,2}月\d{1,2}日\s*-\s*", "", title)
        if not title:
            continue
        desc = ""
        best = ""
        for mp in re.finditer(r"<p[^>]*>(.*?)</p>", b, re.S):
            cand = _strip_html(mp.group(1))
            if len(cand) > len(best) and len(cand) <= 200:
                best = cand
        desc = best
        # 噪音摘要过滤：太短的、"xx反馈"类页脚、纯日期，都不要（宁缺毋滥）
        if desc and (len(desc) < 15 or re.match(r"^[\w.]+\.(com|cn|net)反馈", desc)
                     or re.match(r"^\d{4}年\d{1,2}月\d{1,2}日\s*-?\s*$", desc)):
            desc = ""
        desc = re.sub(r"&nbsp;?", " ", desc)
        if (title, desc) not in out:
            out.append((title[:60], desc[:150]))
        if len(out) >= max_results:
            break
    return out


def search(query: str, max_results: int = 5) -> Optional[str]:
    """搜索并返回文本块；失败/无结果返回 None（不抛异常）。"""
    query = (query or "").strip()
    if not query:
        return None
    now = time.time()
    hit = _cache.get(query)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]
    result: Optional[str] = None
    try:
        html = _fetch("https://www.so.com/s?q=" + urllib.parse.quote(query) + "&pn=1")
        results = _parse(html, max_results)
        if results:
            lines = [f"- {t}" + (f"：{s}" if s else "") for t, s in results]
            result = "【搜索结果】\n" + "\n".join(lines)
            print(f"[SEARCH] {query!r} -> {len(results)} 条", flush=True)
        else:
            print(f"[SEARCH] {query!r} -> 无结果", flush=True)
    except Exception as e:
        print(f"[SEARCH] {query!r} 失败 {type(e).__name__}: {e}", flush=True)
    if len(_cache) > 100:
        for k in list(_cache)[:50]:
            _cache.pop(k, None)
    _cache[query] = (now, result)
    return result
