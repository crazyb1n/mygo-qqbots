"""千早爱音图片理解（WP1，2026-10-26）。

只依赖标准库 + 可选 PIL；任何失败返回空串，绝不抛异常打断聊天。
日志前缀 [ANON VISION]，只打印 URL 的 sha256 前 8 位，不打印完整 URL。

背景（2026-10-26 10:03 实锤）：群友先发图、再引用该图问“这衣服怎么样”，
``_message_text`` 只把图片压成 ``[图片]``，模型没有收到任何图像内容，于是
voice pass 自行编出“图太糊/再发近点”。本模块负责把图真正读懂，读不出来
时由 anonbot 注入“读取失败，不要假装看见”的事实。
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
import urllib.request
from collections import OrderedDict

# ---- 公开常量（anonbot / 测试 / 探针可读）----
# 下载硬上限 6MB（超过直接失败，不截断）；送模型内联上限 3MB（超限先压缩）。
MAX_DOWNLOAD_BYTES = 6 * 1024 * 1024
INLINE_MAX_BYTES = 3 * 1024 * 1024
MAX_DESC_CHARS = 120
CACHE_TTL_SECONDS = 300
CACHE_MAX_ENTRIES = 64
NEGATIVE_TTL_SECONDS = 60
# QQ 防盗链占位图（rkey 直链 200 假成功只给这张）的 md5；不要 import 跨工程。
PLACEHOLDER_MD5 = "5bb9bec07e71ef10a1a47521d396811d"

VISION_ENABLED = os.getenv("ANON_VISION_ENABLED", "1").strip() == "1"
VISION_MODEL = os.getenv("VISION_MODEL", "deepseek/deepseek-v4.1-flash").strip()
VISION_BASE_URL = os.getenv(
    "VISION_BASE_URL", "https://api.commandcode.ai/provider/v1"
).strip().rstrip("/")
# 单次 HTTP 请求的 socket 超时。
VISION_TIMEOUT = float(os.getenv("VISION_TIMEOUT", "30"))
# 单张图从下载到模型返回的总预算。
ANON_VISION_TIMEOUT = float(os.getenv("ANON_VISION_TIMEOUT", "30"))
# 思考型模型（deepseek-v4.1-flash）的思考 token 与正文共用预算；
# 512/1536 会被思考吃光返回空串（主 bot 2026-10-10 事故），这里必须给足。
ANON_VISION_MAX_TOKENS = int(os.getenv("ANON_VISION_MAX_TOKENS", "6144"))
VISION_API_KEY = (
    os.getenv("VISION_API_KEY", "").strip()
    or os.getenv("DEEPSEEK_API_KEY", "").strip()
)

DESCRIBE_PROMPT = (
    "用一句中文描述这张图（不超过80字）：先说画面是什么；有文字就照抄能看清的部分；"
    "如果是衣服或物品，写清款式、颜色、图案；如果是表情包或人物，写清情绪或梗。"
    "不要猜品牌，认不出就写“看不清/不确定”。只输出这一句，不要客套。"
)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AnonBot/1.0"
_cache: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
_cache_lock = threading.Lock()
_warned = {"disabled": False, "no_key": False}


def _log(message: str) -> None:
    print(f"[ANON VISION] {message}", flush=True)


def _sha8(url: str) -> str:
    return hashlib.sha256(str(url or "").encode("utf-8")).hexdigest()[:8]


def _cache_key(url: str) -> str:
    return hashlib.sha256(str(url or "").encode("utf-8")).hexdigest()


def _cache_get(key: str):
    """命中返回字符串（``""`` 表示负缓存）；未命中/过期返回 ``None``。"""
    with _cache_lock:
        item = _cache.get(key)
        if item is None:
            return None
        expires_at, value = item
        if time.monotonic() >= expires_at:
            _cache.pop(key, None)
            return None
        _cache.move_to_end(key)
        return value


def _cache_put(key: str, value: str, ttl: int) -> None:
    """写入缓存；最多 CACHE_MAX_ENTRIES 条，按 OrderedDict 最旧淘汰。"""
    with _cache_lock:
        _cache[key] = (time.monotonic() + ttl, value)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)


def _cache_clear() -> None:
    """测试辅助：清空缓存。生产代码不调用。"""
    with _cache_lock:
        _cache.clear()


def _is_placeholder_raw(raw: bytes) -> bool:
    """是不是 QQ 防盗链占位图（rkey 直链的 200 假成功）。"""
    return (
        len(raw) == 3691
        and hashlib.md5(raw).hexdigest() == PLACEHOLDER_MD5
    )


def _mime_of(raw: bytes) -> str:
    head = raw[:12]
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _prepare_image(raw: bytes):
    """图片字节 -> ``(mime, data, reason)``；失败返回 ``("", b"", reason)``。

    3MB 是送模型的内联上限，6MB 是下载上限，两者不同。GIF 只取第一帧并转
    JPEG；PIL 在函数内延迟 import，没装也不能影响本模块导入。
    """
    mime = _mime_of(raw)
    if len(raw) <= INLINE_MAX_BYTES and mime != "image/gif":
        return mime, raw, ""
    try:
        from PIL import Image
    except Exception:
        return "", b"", "no_pil"
    try:
        image = Image.open(io.BytesIO(raw))
        try:
            image.seek(0)  # 动图只取第一帧
        except Exception:
            pass
        image = image.convert("RGB")
        image.thumbnail((1600, 1600))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        data = buffer.getvalue()
    except Exception:
        return "", b"", "resize_failed"
    if not data:
        return "", b"", "resize_failed"
    if len(data) > INLINE_MAX_BYTES:
        return "", b"", "too_large_after_resize"
    return "image/jpeg", data, ""


def _download(url: str, timeout: float) -> bytes:
    """下载图片原始字节；异常向上抛，由 describe_image 统一记录。"""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": _UA, "Referer": "https://gchat.qpic.cn/"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(MAX_DOWNLOAD_BYTES + 1)


def _vision_call(
    mime: str,
    b64: str,
    prompt: str,
    max_tokens: int,
    timeout: float | None = None,
    sha8: str = "",
) -> str:
    """OpenAI 兼容视觉调用；返回原始 content（可能为空串），异常向上抛。"""
    payload = json.dumps(
        {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            # 不设置 temperature：沿用网关默认，本轮不改现有 chat/task 温度。
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{VISION_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VISION_API_KEY}",
            "User-Agent": _UA,
        },
        method="POST",
    )
    with urllib.request.urlopen(
        request, timeout=timeout if timeout is not None else VISION_TIMEOUT
    ) as response:
        data = json.loads(response.read().decode("utf-8", "replace"))
    message = data["choices"][0]["message"]
    text = str(message.get("content") or "").strip()
    if not text:
        thinking = str(message.get("reasoning_content") or "")
        if thinking:
            _log(
                f"fail sha={sha8} reason=content_empty thinking={len(thinking)}"
            )
    return text


def _clean_description(text: str) -> str:
    """压平空白、取第一句、去首尾引号、截到 MAX_DESC_CHARS。"""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return ""
    first = re.split(r"[。！？\n]", value, maxsplit=1)[0].strip()
    candidate = (first or value).strip().strip("\"'“”‘’「」『』 ")
    if not candidate:
        candidate = value
    return candidate[:MAX_DESC_CHARS].strip()


def describe_image(url: str, prompt: str = DESCRIBE_PROMPT) -> str:
    """下载并描述一张图；成功返回一句中文（≤120 字），失败返回 ""。绝不抛异常。"""
    sha8 = _sha8(url)
    cache_key = ""
    try:
        text_url = str(url or "").strip()
        if not text_url.lower().startswith(("http://", "https://")):
            _log(f"fail sha={sha8} reason=bad_scheme")
            return ""
        if not VISION_ENABLED:
            if not _warned["disabled"]:
                _warned["disabled"] = True
                _log("disabled")
            return ""
        if not VISION_API_KEY:
            if not _warned["no_key"]:
                _warned["no_key"] = True
                _log("no api key")
            return ""

        cache_key = _cache_key(text_url)
        cached = _cache_get(cache_key)
        if cached is not None:
            if cached:
                _log(f"cache hit sha={sha8} chars={len(cached)}")
                return cached
            _log(f"cache negative sha={sha8}")
            return ""

        deadline = time.monotonic() + ANON_VISION_TIMEOUT
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            raise TimeoutError("vision budget exhausted")

        raw = _download(text_url, timeout=min(VISION_TIMEOUT, remaining))
        if not raw:
            _log(f"fail sha={sha8} reason=empty")
            _cache_put(cache_key, "", NEGATIVE_TTL_SECONDS)
            return ""
        if len(raw) > MAX_DOWNLOAD_BYTES:
            _log(f"fail sha={sha8} reason=too_large")
            _cache_put(cache_key, "", NEGATIVE_TTL_SECONDS)
            return ""
        if _is_placeholder_raw(raw):
            _log(f"fail sha={sha8} reason=placeholder")
            _cache_put(cache_key, "", NEGATIVE_TTL_SECONDS)
            return ""

        mime, data, reason = _prepare_image(raw)
        if not data:
            _log(f"fail sha={sha8} reason={reason or 'prepare_failed'}")
            _cache_put(cache_key, "", NEGATIVE_TTL_SECONDS)
            return ""

        remaining = deadline - time.monotonic()
        if remaining <= 1:
            raise TimeoutError("vision budget exhausted")
        text = _vision_call(
            mime,
            base64.b64encode(data).decode("ascii"),
            prompt,
            ANON_VISION_MAX_TOKENS,
            timeout=min(VISION_TIMEOUT, remaining),
            sha8=sha8,
        )
        desc = _clean_description(text)
        if not desc:
            _log(f"fail sha={sha8} reason=content_empty")
            _cache_put(cache_key, "", NEGATIVE_TTL_SECONDS)
            return ""
        _log(f"ok sha={sha8} chars={len(desc)}")
        _cache_put(cache_key, desc, CACHE_TTL_SECONDS)
        return desc
    except Exception as exc:
        # 只写异常类型名：urllib 的异常文本可能包含完整 URL。
        _log(f"fail sha={sha8} reason={type(exc).__name__}")
        if cache_key:
            _cache_put(cache_key, "", NEGATIVE_TTL_SECONDS)
        return ""
