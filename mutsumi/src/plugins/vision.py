"""识图模块：把群里的图片交给云端视觉模型，返回中文描述 / 提取文字。

历史教训：chat.py 原来配的是本地 ollama（localhost:11434 + qwen3-vl:4b），
但本机根本没装 ollama，识图一直空转 → 图片永远被判"不感兴趣"，
群友发图有概率被回"……难看"（实锤过，很败好感）。
2026-09-27 起改用与文本同一个云端网关 + gemma4:31b（扫货同款，实测可用）。

2026-10-10 实锤（"我引用了表情包，若叶睦说看不到"）：视觉模型换成
deepseek-v4.1-flash 后是**思考型**——先把推理写进 reasoning_content 再输出 content。
原来 max_tokens=512/1536 时预算被思考（实测 700~2100 字）整段吃光，
finish_reason=length、content 空串 → describe_* 返回 None → 引用图回退成"[图片]"。
思考 token 与输出共用同一份预算，所以所有视觉调用预算必须 ≥2048。
网关不支持 reasoning_effort/enable_thinking/thinking 关思考（都试过，400 或无效）。

同日第二雷：QQ 表情包原图动辄 6MB+（实测一张动画表情 6636949 字节），
超过 VISION_MAX_BYTES=3MB 后**静默丢弃**（下载完才判大小）→ 同样表现为"看不到"。
现在超限/动图先用 PIL 缩到长边 1600 转 JPEG 再送模型；下载上限单独放宽到 20MB。
"""
import base64
import hashlib
import io
import json
import os
import urllib.request
from typing import Optional, Tuple

VISION_MODEL = os.getenv("VISION_MODEL", "gemma4:31b")
VISION_BASE = os.getenv("VISION_BASE_URL", "https://ollama.com/v1")
VISION_KEY = os.getenv("VISION_API_KEY", "") or os.getenv("DEEPSEEK_API_KEY", "")
VISION_TIMEOUT = float(os.getenv("VISION_TIMEOUT", "45"))
VISION_MAX_BYTES = int(os.getenv("VISION_MAX_BYTES", str(3 * 1024 * 1024)))
# 视觉调用输出预算：必须给思考 token 留位置（见模块头，512 会被思考吃光返回空）。
# 2026-10-10 晚实测：加了"认出角色"要求后思考变长（简单图 72 字/3s，难的 3400~5700 字/30s+），
# 2048 不够、4096 对最难的图仍偶尔清空 → 提到 6144。max_tokens 只是上限，快图不会因此变慢。
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "6144"))
# 下载/读取上限单独放宽：大图先收下来再压缩，而不是直接丢（见模块头第二雷）
VISION_DOWNLOAD_MAX_BYTES = int(os.getenv("VISION_DOWNLOAD_MAX_BYTES",
                                          str(20 * 1024 * 1024)))
# 超限或动图压缩后的长边上限（OCR 要看清小字，别缩太狠）
VISION_SHRINK_MAX_SIDE = int(os.getenv("VISION_SHRINK_MAX_SIDE", "1600"))
# QQ 防盗链占位图（"此图片来自QQ空间 未经允许不可引用"）：rkey 直链会 200 假成功只给这张，
# md5/size 与 stickers.PLACEHOLDER_MD5 一致。2026-10-10 21:37 实锤：引用图旧接口 404 +
# rkey 直链拿到它 → 描述失败 → 她对"翻译图片里的韩语"回【沉默】。
# 注意 vision 不能 import stickers（stickers 依赖 vision，会成环），常量在这里重复一份。
PLACEHOLDER_MD5 = "5bb9bec07e71ef10a1a47521d396811d"


def _is_placeholder_raw(raw: bytes) -> bool:
    """是不是 QQ 防盗链占位图（rkey 直链的 200 假成功）。"""
    return len(raw) == 3691 and hashlib.md5(raw).hexdigest() == PLACEHOLDER_MD5


def _prepare_image(raw: bytes) -> Tuple[str, bytes]:
    """图片字节 -> (mime, 待发送字节)。超限或动图用 PIL 缩到长边 1600 内、转 JPEG。
    返回 ("", b"") 表示既超限又压不出来（只能放弃）。"""
    mime = _mime_of(raw)
    if len(raw) <= VISION_MAX_BYTES and mime != "image/gif":
        return mime, raw
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw))
        try:
            im.seek(0)          # 动图只取第一帧（模型看不懂动图，第一帧够表达情绪）
        except Exception:
            pass
        im = im.convert("RGB")
        im.thumbnail((VISION_SHRINK_MAX_SIDE, VISION_SHRINK_MAX_SIDE))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        out = buf.getvalue()
        if out:
            if len(raw) > VISION_MAX_BYTES:
                print(f"[VISION] 原图 {len(raw)//1024}KB 超限，压缩到 {len(out)//1024}KB 再送", flush=True)
            return "image/jpeg", out
    except Exception as e:
        print(f"[VISION] 图片压缩失败 {type(e).__name__}: {e}", flush=True)
    if len(raw) <= VISION_MAX_BYTES:
        return mime, raw
    return "", b""

DESCRIBE_PROMPT = (
    "用一句中文描述这张图（≤60字）：画面是什么、有没有文字（有就照抄）、什么情绪/梗。"
    "角色名一句话带过即可：一眼认得出就写（如\"若叶睦（BanG Dream!）\"），"
    "认不出就跳过，不要为此推敲。如果是表情包，直接说它表达什么情绪。"
    "只输出这一句，不要客套。"
)
OCR_PROMPT = (
    "请把图片中的所有文字原样提取出来，包括题目内容、数字、公式、选项。"
    "只输出提取到的文字，不要解释。"
)


def _mime_of(raw: bytes) -> str:
    head = raw[:12]
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF":
        return "image/webp"
    return "image/jpeg"


def _vision_call(mime: str, b64: str, prompt: str, max_tokens: int,
                 timeout: Optional[float] = None) -> Optional[str]:
    """base64 图 + 提示词 -> 视觉模型。任何失败返回 None（绝不抛异常打断主流程）。"""
    try:
        payload = json.dumps({
            "model": VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
            "max_tokens": max_tokens,
            "temperature": 0.6,
        }).encode()
        req2 = urllib.request.Request(
            f"{VISION_BASE}/chat/completions", data=payload,
            headers={"content-type": "application/json",
                     "Authorization": f"Bearer {VISION_KEY}",
                     # UA 不能换：默认 Python UA 会被网关的 Cloudflare 1010 拦掉
                     "User-Agent": "mutsumi-bot/1.0"})
        with urllib.request.urlopen(req2, timeout=timeout or VISION_TIMEOUT) as resp2:
            d = json.loads(resp2.read().decode())
        msg = d["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        if not text:
            rc = msg.get("reasoning_content") or ""
            if rc:
                print(f"[VISION] content 空、thinking {len(rc)} 字 —— 预算被思考吃光，"
                      f"需加大 max_tokens（当前 {max_tokens}）", flush=True)
        return text[:300] if text else None
    except Exception as e:
        print(f"[VISION] 调用失败 {type(e).__name__}: {e}", flush=True)
        return None


def _call_vision(url: str, prompt: str, max_tokens: int,
                 timeout: Optional[float] = None) -> Optional[str]:
    """下载图片 -> 视觉模型。任何失败返回 None（绝不抛异常打断主流程）。"""
    if not url or not VISION_KEY:
        return None
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            # 2026-09-29 实锤：gchat 群图直链必须带 gchat Referer（qzone 的会 400）
            "Referer": "https://gchat.qpic.cn/",
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read(VISION_DOWNLOAD_MAX_BYTES + 1)
        if not raw or len(raw) > VISION_DOWNLOAD_MAX_BYTES:
            return None
        if _is_placeholder_raw(raw):
            print("[VISION] 拿到的是 QQ 防盗链占位图（非真图），判失败", flush=True)
            return None
        mime, data = _prepare_image(raw)
        if not data:
            return None
        return _vision_call(mime, base64.b64encode(data).decode(), prompt,
                            max_tokens, timeout)
    except Exception as e:
        print(f"[VISION] 调用失败 {type(e).__name__}: {e}", flush=True)
        return None


def describe_file(path: str, prompt: str = DESCRIBE_PROMPT,
                  max_tokens: int = VISION_MAX_TOKENS,
                  timeout: Optional[float] = None) -> Optional[str]:
    """描述本地图片文件（表情库语义标注、视频抽帧描述用）。失败返回 None。
    max_tokens 给足 2048（见模块头）：思考型视觉模型的思考 token 也占预算，
    预算不够时正文被挤空、静默返回空串。timeout 供调用方收紧单次等待（视频帧预算）。"""
    try:
        with open(path, "rb") as f:
            raw = f.read(VISION_DOWNLOAD_MAX_BYTES + 1)
    except Exception as e:
        print(f"[VISION] 读文件失败 {os.path.basename(path)}: {e}", flush=True)
        return None
    if not raw or len(raw) > VISION_DOWNLOAD_MAX_BYTES:
        return None
    mime, data = _prepare_image(raw)
    if not data:
        return None
    text = _vision_call(mime, base64.b64encode(data).decode(), prompt,
                        max_tokens, timeout)
    if text:
        print(f"[VISION] {os.path.basename(path)} -> {text[:70]!r}", flush=True)
    return text


def describe_image(url: str, max_tokens: int = VISION_MAX_TOKENS) -> Optional[str]:
    """描述图片内容（一句中文）。失败返回 None。"""
    text = _call_vision(url, DESCRIBE_PROMPT, max_tokens)
    if text:
        print(f"[VISION] {url[:50]}... -> {text[:70]!r}", flush=True)
    return text


def extract_text(url: str, max_tokens: int = VISION_MAX_TOKENS) -> Optional[str]:
    """提取图片里的文字（解题 OCR 用）。失败返回 None。"""
    text = _call_vision(url, OCR_PROMPT, max_tokens)
    if text:
        print(f"[OCR] {url[:50]}... -> {text[:80]!r}", flush=True)
    return text
