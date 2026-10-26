"""识图模块：把群友发的图交给视觉模型，返回一段中文描述。

后端：Ollama 云（OpenAI 兼容 /v1/chat/completions），模型默认 gemma4:31b（支持视觉）。
（若叶睦那边配的是本地 ollama localhost:11434，但本机并未安装 ollama，所以那个其实没生效；
 这里直接用与文本同一个云端网关，稳定可用。）
"""
import base64
import json
import os
import urllib.request
from typing import Optional

VISION_MODEL = os.getenv("VISION_MODEL", "gemma4:31b")
VISION_BASE = os.getenv("VISION_BASE_URL", "https://ollama.com/v1")
VISION_KEY = os.getenv("VISION_API_KEY", "") or os.getenv("DEEPSEEK_API_KEY", "")
VISION_TIMEOUT = float(os.getenv("VISION_TIMEOUT", "45"))
VISION_MAX_BYTES = int(os.getenv("VISION_MAX_BYTES", str(3 * 1024 * 1024)))

PROMPT = (
    "用一句中文描述这张图：画的是什么、有没有文字（有就照抄）、什么情绪/梗。"
    "如果是表情包，直接说它表达什么情绪。只输出这一句描述，不要客套。"
)


def describe_image(url: str) -> Optional[str]:
    """下载图片并让视觉模型描述。失败返回 None（绝不抛异常打断主流程）。"""
    if not url or not VISION_KEY:
        return None
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            # 2026-09-29 实锤：gchat 群图直链必须带 gchat Referer（qzone 的会 400）
            "Referer": "https://gchat.qpic.cn/",
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read(VISION_MAX_BYTES + 1)
        if not raw or len(raw) > VISION_MAX_BYTES:
            return None
        b64 = base64.b64encode(raw).decode()
        # 猜 mime
        head = raw[:12]
        if head[:3] == b"\xff\xd8\xff":
            mime = "image/jpeg"
        elif head[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        elif head[:6] in (b"GIF87a", b"GIF89a"):
            mime = "image/gif"
        elif head[:4] == b"RIFF":
            mime = "image/webp"
        else:
            mime = "image/jpeg"
        payload = json.dumps({
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
            "max_tokens": 2048,
            "temperature": 0.6,
        }).encode()
        req2 = urllib.request.Request(
            f"{VISION_BASE}/chat/completions", data=payload,
            headers={"content-type": "application/json",
                     "Authorization": f"Bearer {VISION_KEY}",
                     "User-Agent": "saohuo-bot/1.0"})
        with urllib.request.urlopen(req2, timeout=VISION_TIMEOUT) as resp2:
            d = json.loads(resp2.read().decode())
        text = (d["choices"][0]["message"].get("content") or "").strip()
        if text:
            print(f"[VISION] {url[:50]}... -> {text[:70]!r}", flush=True)
            return text[:200]
        return None
    except Exception as e:
        print(f"[VISION] 识图失败 {type(e).__name__}: {e}", flush=True)
        return None
