"""LLM 降级通道（2026-10-10）：主网关撞额度/故障时，自动切到备用网关重试一次。

背景：两只 bot 共用同一把 commandcode key，它的 5 小时额度一满，两只一起哑掉，
群里表现为刷 "[AI异常] 大脑开小差了：RateLimitError"。

接法（调用方一行不用改，chat.completions.create 签名兼容）：
    from .llm_fallback import build_client
    client = build_client(AsyncOpenAI(api_key=..., base_url=...))

备用网关：默认科大 AI 网关（Qwen3.6-35B-A3B，用户自己的学校额度；2026-10-10 实测可用。
付费模型余额为 0，不要用）。用 .env 覆盖：
    FALLBACK_BASE_URL / FALLBACK_API_KEY / FALLBACK_MODEL
未配置 FALLBACK_API_KEY 时回退读 AGENT_API_KEY（薇欧拉侧直接复用她的 agent 配置）。
"""
import os
from typing import Optional

from openai import AsyncOpenAI

FALLBACK_BASE_URL = os.getenv("FALLBACK_BASE_URL", "https://ai-apigateway.must.edu.mo/openhub/v1")
FALLBACK_API_KEY = (os.getenv("FALLBACK_API_KEY", "")
                    or os.getenv("AGENT_API_KEY", ""))
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "Qwen3.6-35B-A3B")

# 这些情况下认为"主网关不可用"，值得切备用：额度限流 / 网关 5xx / 连不上 / 超时
_RETRY_STATUS = {429, 500, 502, 503, 504}
_RETRY_ERRORS = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}


def _should_fallback(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) in _RETRY_STATUS:
        return True
    return type(exc).__name__ in _RETRY_ERRORS


class _Completions:
    def __init__(self, primary: AsyncOpenAI, fallback: Optional[AsyncOpenAI],
                 fallback_model: str):
        self._primary = primary
        self._fallback = fallback
        self._fallback_model = fallback_model

    async def create(self, **kwargs):
        try:
            return await self._primary.chat.completions.create(**kwargs)
        except Exception as e:
            if self._fallback is None or not _should_fallback(e):
                raise
            print(f"[LLM-FALLBACK] 主网关失败({type(e).__name__}: {str(e)[:80]}) -> 切备用模型 "
                  f"{self._fallback_model}", flush=True)
            kw = dict(kwargs)
            kw["model"] = self._fallback_model
            # 备用网关不保证吃 reasoning_effort；去掉它（顶层与 extra_body 里都清）
            kw.pop("reasoning_effort", None)
            body = dict(kw.get("extra_body") or {})
            body.pop("reasoning_effort", None)
            if body:
                kw["extra_body"] = body
            else:
                kw.pop("extra_body", None)
            # 备用模型是思考型：调用点给的预算（如 2048）常被思考吃光导致正文为空，
            # 统一垫到至少 4096（只抬高上限，不额外计费）。
            try:
                if int(kw.get("max_tokens") or 0) < 4096:
                    kw["max_tokens"] = 4096
            except Exception:
                kw["max_tokens"] = 4096
            resp = await self._fallback.chat.completions.create(**kw)
            print(f"[LLM-FALLBACK] 备用网关响应成功", flush=True)
            return resp


class _Chat:
    def __init__(self, completions: _Completions):
        self.completions = completions


class FallbackClient:
    """鸭子类型包装：.chat.completions.create 走降级链，其余属性透传给主客户端。"""

    def __init__(self, primary: AsyncOpenAI, fallback: Optional[AsyncOpenAI],
                 fallback_model: str):
        self._primary = primary
        self.chat = _Chat(_Completions(primary, fallback, fallback_model))

    def __getattr__(self, name):
        return getattr(self._primary, name)


def build_client(primary: AsyncOpenAI,
                 fallback_model: str = "") -> AsyncOpenAI:
    """给主客户端套上降级链。未配置备用 key 时原样返回主客户端（不改变行为）。"""
    if not FALLBACK_API_KEY or not FALLBACK_BASE_URL:
        return primary
    fallback = AsyncOpenAI(api_key=FALLBACK_API_KEY, base_url=FALLBACK_BASE_URL)
    return FallbackClient(primary, fallback, fallback_model or FALLBACK_MODEL)
