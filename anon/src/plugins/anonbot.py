"""Anon QQ bot: DeepSeek handles chat/routing and complex tasks."""
import asyncio
import json
import os
import random
import re
import time
from collections import defaultdict, deque
from pathlib import Path

from nonebot import on_message
from nonebot.rule import Rule, to_me
from nonebot.adapters import Event
from nonebot.adapters.onebot.v11 import (
    Bot,
    GroupMessageEvent,
    Message,
    MessageSegment,
    PrivateMessageEvent,
)
from openai import AsyncOpenAI

from . import agent_core
from . import agent_search
from . import anon_context
from . import anon_experience as experience
from . import anon_people
from . import anon_search_policy
from . import anon_vision
from .llm_fallback import build_client

ROOT = Path(__file__).resolve().parents[2]
ROLE_FILE = ROOT / "roles" / "Anon.md"
STATE_FILE = ROOT / "anon_state.json"
MEMORY_FILE = ROOT / "anon_memory.json"

BOT_QQ = os.getenv("BOT_QQ", "").strip()
OTHER_BOTS = {"200000001", "200000002", "200000004", "200000003", "200000007", "200000006", "200000008", "200000005"}
ALLOWED_GROUPS = {
    int(value.strip())
    for value in os.getenv("ALLOWED_GROUPS", "100000001").split(",")
    if value.strip().isdigit()
}
MAX_HISTORY_MESSAGES = 18
MAX_IMAGES_PER_MESSAGE = 2
VISION_BUDGET_SECONDS = 35
MAX_GROUP_CONTEXT = 12
MAX_REPLY_CHARS = int(os.getenv("ANON_CHAT_CHARS", "35"))
MAX_TASK_REPLY_CHARS = agent_core.MAX_AGENT_OUTPUT_CHARS
FIRST_GREETING = "初次见面！我是千早爱音，请多指教～"

SUMMARY_ENABLED = os.getenv("ANON_SUMMARY_ENABLED", "1") == "1"
SUMMARY_INTERVAL = int(os.getenv("ANON_SUMMARY_INTERVAL", "30"))
SUMMARY_MAX_CHARS = int(os.getenv("ANON_SUMMARY_CHARS", "400"))
SEARCH_REPLY_CHARS = int(os.getenv("ANON_CHAT_SEARCH_CHARS", "100"))
# 2026-10-26 延迟回复保护：旧消息在对话翻篇后不再回答；planner 与 voice 各自限时，
# 避免网关卡住时同一个群的消息串行等待两分钟（19:27 装弹机/raychell 实锤）。
# 进 LLM 前 + 拿到会话锁后判定队列陈旧；闲聊答案生成后若总年龄超过 120s 也丢弃
# （对话已翻篇，宁可不说也不答非所问）；任务豁免（ack 已发，复杂任务允许多轮 240s）。
STALE_SECONDS = int(os.getenv("ANON_STALE_SECONDS", "120"))
PLANNER_TIMEOUT = int(os.getenv("ANON_PLANNER_TIMEOUT", "25"))
VOICE_TIMEOUT = int(os.getenv("ANON_VOICE_TIMEOUT", "35"))
# WP2（2026-10-26）：受限动作在模型失败时的固定拒绝短句（<=35 字），
# 保证“发图/转告/抢票”即使 LLM 全挂也不会假装执行。
ACTION_FALLBACK_REPLIES = {
    "send_media": "图片和语音我还发不了啦，想听什么我用文字说给你。",
    "contact_third_party": "替人传话这事我做不来，你自己说更靠谱，我可以帮你想措辞。",
    "purchase_login": "抢票登录这类我碰不了，开售时间我可以帮你查。",
}

# 2026-10-26 评审修复：模型会在受限请求上“先答应再说”（“我发张图给你看看”
# “好，我去抢票了”）。按分句判定假承诺，否定句（“我发不了图”）不算；
# 命中任一分句就整条换成固定拒绝短句。
_FAKE_ACTION_PREFIX = (
    r"(?:已经|已|刚刚|刚才|这就|马上|立刻|稍等|现在|{extra}"
    r"帮你|替你|给你|那我去?)"
)
_FAKE_ACTION_COMPLETION = (
    r"(?:(?:好|完|到|上|过去|出去)(?:了)?"
    r"|(?:了|过)"
    r"|(?:他|她|您|你|他们|大家)(?:了|过))"
)
_FAKE_ACTION_NEGATION_RE = re.compile(
    r"(?:不|没|无法|没法|不能|不会|做不到|办不到|碰不了|做不来)"
    r"|文字|打字|口头|描述|建议"
)


def _fake_action_pattern(
    verbs: str, extra: str, extra_pattern: str = ""
) -> "re.Pattern[str]":
    return re.compile(
        _FAKE_ACTION_PREFIX.format(extra=extra)
        + r"[^，。！？!?\n]{0,8}(?:" + verbs + r")"
        + r"|(?:" + verbs + r")" + _FAKE_ACTION_COMPLETION
        + r"|(?:发|传|唱|录|拍)你(?:一|张|个|段|首)?"
        + extra_pattern
    )


_FAKE_ACTION_PATTERNS = {
    "send_media": _fake_action_pattern(
        r"发|传|录|拍|唱", r"我去|我来|我先|",
        r"|我(?:发|传|录|拍)[^，。！？!?\n]{0,8}(?:给你|你)"
        r"|我唱[^，。！？!?\n]{0,6}给你",
    ),
    "contact_third_party": _fake_action_pattern(
        r"转告|带话|告诉|联系|私聊|问", r"",
        # “说”太宽（“帮你说句话”不是传话）；只把“跟/和/对……说了”算完成。
        r"|(?:跟|和|对)[^，。！？!?\n]{0,4}说(?:了|过)",
    ),
    "purchase_login": _fake_action_pattern(
        r"抢|买|登录|登陆|付款|支付|下单|签到|注册", r"我去|我来|我先|"
    ),
}


def _claims_fake_action(answer: str, kind: str) -> bool:
    """回复是否声称已经执行了受限动作（按分句判定，否定句不算）。"""
    pattern = _FAKE_ACTION_PATTERNS.get(kind)
    if pattern is None:
        return False
    for clause in re.split(r"[，,。！!？?；;\n]+", answer or ""):
        if not clause.strip() or _FAKE_ACTION_NEGATION_RE.search(clause):
            continue
        if pattern.search(clause):
            return True
    return False

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
CHAT_MODEL = os.getenv("ANON_CHAT_MODEL", DEFAULT_MODEL).strip()
TASK_MODEL = os.getenv("ANON_TASK_MODEL", DEFAULT_MODEL).strip()
CHAT_MAX_TOKENS = int(os.getenv("ANON_CHAT_MAX_TOKENS", "2048"))
TASK_MAX_TOKENS = int(os.getenv("ANON_TASK_MAX_TOKENS", "8192"))
CHAT_TIMEOUT_SECONDS = int(os.getenv("ANON_CHAT_TIMEOUT", "90"))
TASK_TIMEOUT_SECONDS = int(os.getenv("ANON_TASK_TIMEOUT", "240"))
CIRCUIT_WINDOW_SECONDS = int(os.getenv("CIRCUIT_WINDOW_SECONDS", "300"))
CIRCUIT_MAX_TOKENS = int(os.getenv("CIRCUIT_MAX_TOKENS", "200000"))
CIRCUIT_BREAK_SECONDS = int(os.getenv("CIRCUIT_BREAK_SECONDS", "600"))
_usage_window: deque[tuple[float, int]] = deque()
_circuit_until = 0.0
DISABLE_REASONING = os.getenv("DISABLE_REASONING", "0") == "1"
_ALLOWED_EFFORTS = {"off", "low", "medium", "high", "xhigh", "max"}


def _normalized_effort(value: str) -> str:
    effort = (value or "").strip().lower()
    if effort in {"", "none", "false", "0", "disabled"}:
        return "off"
    if effort in _ALLOWED_EFFORTS:
        return effort
    print(f"[ANON] unsupported reasoning_effort={effort!r}", flush=True)
    return ""


CHAT_REASONING_EFFORT = _normalized_effort(os.getenv("ANON_CHAT_REASONING_EFFORT", "low"))
TASK_REASONING_EFFORT = _normalized_effort(os.getenv("ANON_TASK_REASONING_EFFORT", "high"))


def _circuit_open() -> bool:
    """Short-window token breaker: stop paid LLM calls without dropping the bot."""
    global _circuit_until
    now = time.monotonic()
    if _circuit_until and now >= _circuit_until:
        _circuit_until = 0.0
        print("[ANON CIRCUIT] recovered", flush=True)
    return bool(_circuit_until and now < _circuit_until)


def _record_usage(usage) -> None:
    global _circuit_until
    if usage is None:
        return
    total = int(getattr(usage, "total_tokens", 0) or 0)
    if total <= 0:
        total = int(getattr(usage, "prompt_tokens", 0) or 0) + int(
            getattr(usage, "completion_tokens", 0) or 0
        )
    if total <= 0:
        return
    now = time.monotonic()
    _usage_window.append((now, total))
    while _usage_window and now - _usage_window[0][0] > CIRCUIT_WINDOW_SECONDS:
        _usage_window.popleft()
    used = sum(tokens for _, tokens in _usage_window)
    if used > CIRCUIT_MAX_TOKENS and not _circuit_open():
        _circuit_until = now + CIRCUIT_BREAK_SECONDS
        print(
            f"[ANON CIRCUIT] open used={used} window={CIRCUIT_WINDOW_SECONDS}s "
            f"break={CIRCUIT_BREAK_SECONDS}s",
            flush=True,
        )


client = None
if DEEPSEEK_API_KEY:
    client = build_client(AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL))
else:
    print("[ANON] DEEPSEEK_API_KEY 未配置，DeepSeek 调用会失败", flush=True)

try:
    SYSTEM_PROMPT = ROLE_FILE.read_text(encoding="utf-8").strip()
except Exception as exc:
    print(f"[ANON] role card load failed: {type(exc).__name__}: {exc}", flush=True)
    SYSTEM_PROMPT = "你将扮演千早爱音：MyGO!!!!! 的吉他手，开朗、爱自拍、渴望被夸奖的女高中生。"


def _load_greeted_users() -> set[str]:
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return {str(uid) for uid in payload.get("greeted_users", []) if str(uid)}
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return set()


def _load_group_summaries() -> dict:
    """滚动摘要持久化：读 MEMORY_FILE 的 {"group_summaries": {...}}；缺失/损坏置空。"""
    try:
        payload = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[ANON MEMORY] load failed ({type(exc).__name__}: {exc}), summaries start empty", flush=True)
        return {}
    summaries = payload.get("group_summaries") if isinstance(payload, dict) else None
    if not isinstance(summaries, dict):
        print("[ANON MEMORY] group_summaries invalid, summaries start empty", flush=True)
        return {}
    return {str(group): str(text) for group, text in summaries.items() if str(text)}


def _load_user_profiles() -> dict:
    """群友档案持久化（2026-10-26）：读 MEMORY_FILE 的 ``users`` 键；失败置空。

    档案由 ``anon_people`` 负责规整与上限控制：QQ → 见过的名字/时间/条数/最后一句。
    之前只有群摘要落盘、每用户历史只存内存，重启即“失忆”，因此不认识人。
    """
    try:
        payload = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[ANON PEOPLE] load failed ({type(exc).__name__}: {exc}), profiles start empty", flush=True)
        return {}
    return anon_people.load_users(payload)


greeted_users = _load_greeted_users()
histories: dict[tuple[str, str], deque] = defaultdict(
    lambda: deque(maxlen=MAX_HISTORY_MESSAGES)
)
group_contexts: dict[str, deque] = defaultdict(lambda: deque(maxlen=40))
group_recent_raw: dict = defaultdict(list)
group_msg_count: dict = defaultdict(int)
group_summaries: dict = _load_group_summaries()
user_profiles: dict = _load_user_profiles()
summary_tasks: set = set()
# 每群一把摘要锁：同一群同一时间只允许一个滚动摘要任务在跑（WP4 防重入）。
summary_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
DISTILL_INTERVAL = 60
_last_distill: dict = {}
distill_tasks: set = set()
scope_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
handler = on_message(priority=20, block=False)


def _save_greeted_users() -> None:
    try:
        temp_path = STATE_FILE.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps({"greeted_users": sorted(greeted_users)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(STATE_FILE)
    except OSError as exc:
        print(f"[ANON] state save failed: {type(exc).__name__}: {exc}", flush=True)


def _save_memory() -> None:
    try:
        temp_path = MEMORY_FILE.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(
                {"group_summaries": group_summaries, "users": user_profiles},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temp_path.replace(MEMORY_FILE)
    except OSError as exc:
        print(f"[ANON MEMORY] save failed: {type(exc).__name__}: {exc}", flush=True)


PROFILE_SAVE_INTERVAL = 60
_last_profile_save = 0.0


def _save_user_profiles() -> None:
    """档案落盘节流：最多每 60 秒一次（与主 bot 的 MEMORY_SAVE_INTERVAL 同思路）。

    每条群消息都写盘会放大 IO；脏杀最多丢 60 秒的新档案，可接受。
    群摘要写回（_refresh_group_summary）仍走 _save_memory()，一并带上 users。
    离线单测直接调用 handler（NoneBot 未初始化）时不落盘，避免污染真实档案。
    """
    global _last_profile_save
    now = time.monotonic()
    if now - _last_profile_save < PROFILE_SAVE_INTERVAL:
        return
    try:
        from nonebot import get_driver

        get_driver()
    except Exception:
        return
    _last_profile_save = now
    _save_memory()


def _record_profile(user_id: str, user_name: str, user_text: str, received_at: float) -> None:
    """更新并（节流）落盘一条群友档案；只在 handler 内调用，零 LLM 成本。"""
    status = anon_people.update(
        user_profiles, user_id, user_name, user_text, ts=received_at
    )
    if status:
        print(
            f"[ANON PEOPLE] {status} sender={user_id} name={user_name!r} "
            f"count={user_profiles.get(user_id, {}).get('count', '?')}",
            flush=True,
        )
    _save_user_profiles()


def _sender_name(event) -> str:
    sender = getattr(event, "sender", None)
    if sender is None:
        return str(getattr(event, "user_id", "群友"))
    return ((getattr(sender, "card", "") or "").strip()
            or (getattr(sender, "nickname", "") or "").strip()
            or str(getattr(event, "user_id", "群友")))


def _message_text(event) -> str:
    parts = []
    for segment in event.get_message():
        kind = segment.type
        data = segment.data
        if kind == "text":
            parts.append(data.get("text", ""))
        elif kind == "image":
            parts.append("[图片]")
        elif kind == "record":
            parts.append("[语音]")
        elif kind == "video":
            parts.append("[视频]")
        elif kind == "face":
            parts.append("[表情]")
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def _image_urls(event) -> list[str]:
    """提取本次触发要看的图片 URL：当前消息优先，其次被引用消息；只收 http/https，最多 2 张。

    2026-10-26 实锤：群友先发图、再引用该图问“这衣服怎么样”，图片只在 reply.message 里。
    只扫当前消息会漏掉真实场景，所以必须扫两层；顺序固定为 direct -> quote。
    """
    urls: list[str] = []
    seen: set[str] = set()

    def collect(message) -> None:
        if message is None:
            return
        try:
            segments = list(message)
        except (TypeError, AttributeError):
            return
        for segment in segments:
            if getattr(segment, "type", "") != "image":
                continue
            data = getattr(segment, "data", {}) or {}
            # NapCat 真图给 url；部分适配器/测试只把 URL 放在 file。只收 http(s)，
            # 本地路径（file:// 或 C:\...）一律跳过，避免读本机文件。
            url = str(data.get("url") or data.get("file") or "").strip()
            if not url.lower().startswith(("http://", "https://")):
                continue
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
            if len(urls) >= MAX_IMAGES_PER_MESSAGE:
                return

    try:
        direct_message = event.get_message()
    except (TypeError, AttributeError):
        direct_message = None
    collect(direct_message)
    reply = getattr(event, "reply", None)
    if len(urls) < MAX_IMAGES_PER_MESSAGE and reply is not None:
        collect(getattr(reply, "message", None))
    return urls


async def _augment_text_with_images(text: str, urls: list[str]) -> str:
    """把图片描述注入当前文本；失败/超时注入“读取失败”事实，绝不让模型假装看见。

    最多 2 张并发读；整轮预算 VISION_BUDGET_SECONDS=35 秒。日志只报数量与字符数，
    不打印 URL。describe_image 本身不抛异常，这里仍用 return_exceptions 兜底。
    """
    urls = [url for url in (urls or []) if url][:MAX_IMAGES_PER_MESSAGE]
    if not urls:
        return text

    results = []
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(asyncio.to_thread(anon_vision.describe_image, url) for url in urls),
                return_exceptions=True,
            ),
            timeout=VISION_BUDGET_SECONDS,
        )
    except Exception as exc:
        print(
            f"[ANON VISION] timeout images={len(urls)} budget={VISION_BUDGET_SECONDS}s "
            f"reason={type(exc).__name__}",
            flush=True,
        )
        results = []

    descriptions: list[str] = []
    for value in results:
        if isinstance(value, BaseException):
            continue
        desc = str(value or "").strip()
        if desc:
            descriptions.append(desc)

    described = len(descriptions)
    chars = sum(len(desc) for desc in descriptions)
    print(
        f"[ANON VISION] images={len(urls)} described={described} chars={chars}",
        flush=True,
    )

    if described:
        addition = " [图片内容：" + "；".join(descriptions) + "]"
        failed = len(urls) - described
        if failed > 0:
            addition += f"（另有{failed}张读取失败，不要假装看见）"
        return f"{text}{addition}"
    return f"{text} [图片内容：读取失败，你看不到这张图，不要假装看见]"


def _reply_sender_id(event) -> str:
    """Return the quoted message sender, if OneBot exposed it."""
    reply = getattr(event, "reply", None)
    if reply is not None:
        sender = getattr(reply, "sender", None)
        uid = getattr(sender, "user_id", None)
        if uid is not None:
            return str(uid)
    for segment in event.get_message():
        if segment.type == "reply":
            qq = str(segment.data.get("qq", "")).strip()
            if qq:
                return qq
    return ""


def _quoted_text(event) -> str:
    reply = getattr(event, "reply", None)
    message = getattr(reply, "message", None) if reply is not None else None
    if message is None:
        return ""
    parts = []
    try:
        for segment in message:
            if segment.type == "text":
                parts.append(segment.data.get("text", ""))
            elif segment.type in {"image", "video", "record"}:
                parts.append(f"[{segment.type}]")
    except (AttributeError, TypeError):
        return ""
    return " ".join(parts).strip()[:300]


def _is_triggered(event, text: str, bot_id: str) -> tuple[bool, str]:
    bot_id = str(bot_id).strip()
    for segment in event.get_message():
        if segment.type == "at" and str(segment.data.get("qq", "")).strip() == bot_id:
            return True, "at"
    # A reply to somebody else must not wake Anon. The old code treated any
    # reply segment as hers and barged into unrelated conversations.
    if _reply_sender_id(event) == bot_id:
        return True, "reply"
    raw_message = str(getattr(event, "raw_message", ""))
    if re.search(rf"\[CQ:at,qq={re.escape(bot_id)}(?:,|\])", raw_message, re.IGNORECASE):
        return True, "at-raw"
    try:
        if event.is_tome():
            return True, "tome"
    except (AttributeError, TypeError):
        pass
    if anon_context.is_name_mention(text):
        return True, "name"
    return False, ""


def _build_prompt(
    history,
    group_context,
    user_name: str,
    user_text: str,
    quoted_text: str,
    first_message: bool,
    summary: str = "",
    experience_hint: str = "",
    search_hint: str = "",
    max_chars: int = MAX_REPLY_CHARS,
    user_profile: str = "",
) -> str:
    return anon_context.build_chat_prompt(
        history,
        group_context,
        user_name,
        user_text,
        quoted_text,
        first_message,
        FIRST_GREETING,
        max_chars,
        summary=summary,
        experience_hint=experience_hint,
        search_hint=search_hint,
        user_profile=user_profile,
    )


def _clean_answer(answer: str, max_chars: int = MAX_REPLY_CHARS,
                  bubbles: bool = False) -> str:
    """清洗模型输出；bubbles=True 时把多行闲聊按 1~3 条、每行 35 字清洗。

    单行回复仍走原 clean_answer（搜索回复的 100 字上限不变）；任务回复不传 bubbles，
    保持“一条消息交付”。"""
    raw = str(answer or "").strip()
    if bubbles:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(lines) > 1:
            cleaned = [
                anon_context.clean_answer(line, MAX_REPLY_CHARS, MAX_REPLY_CHARS)
                for line in lines[:agent_core.MAX_CHAT_BUBBLES]
            ]
            return "\n".join(x for x in cleaned if x)
    return anon_context.clean_answer(raw, max_chars, MAX_REPLY_CHARS)


async def _quick_search_hint(text: str) -> str:
    """闲聊快搜（WP-B）：失败/超时/空结果一律返回 ""，绝不让搜索阻塞聊天。"""
    if _circuit_open():
        return ""
    try:
        found = await asyncio.wait_for(
            asyncio.to_thread(agent_search.search, (text or "").strip()[:60], 4), timeout=14
        )
    except Exception as exc:
        print(f"[ANON QUICK SEARCH] failed: {type(exc).__name__}", flush=True)
        return ""
    hint = (found or "").strip()
    if not hint:
        print("[ANON QUICK SEARCH] empty", flush=True)
    return hint


def _log_safe(text: str, limit: int = 60) -> str:
    """日志字段压平：去掉 QQ 噪声/换行/控制符，防止群友伪造 [ANON ...] 日志行。"""
    value = anon_search_policy.strip_noise(text or "")
    value = re.sub(r"[\x00-\x1f\x7f]+", " ", value)
    return value[:limit]


async def _quick_search_materials(queries: list[str]) -> str:
    """闲聊确定性快搜（WP2）：最多两条，第一条有结果就停；失败/空/熔断都不阻塞聊天。

    每条复用 ``_quick_search_hint``（它内部先查熔断、14 秒超时、异常返回 ""），
    只额外补一行带查询词的日志，方便从 bot_standalone.log 复盘“到底搜了什么”。
    """
    materials: list[str] = []
    seen: set[str] = set()
    attempted = 0
    for query in queries or []:
        if attempted >= 2:
            break
        value = (query or "").strip()[:60]
        if not value or value in seen:
            continue
        seen.add(value)
        attempted += 1
        if _circuit_open():
            print("[ANON QUICK SEARCH] circuit open", flush=True)
            break
        hint = (await _quick_search_hint(value) or "").strip()
        hits = sum(1 for line in hint.splitlines() if line.strip().startswith("- "))
        print(
            f"[ANON QUICK SEARCH] q={_log_safe(value)} hits={hits} used chars={len(hint)}",
            flush=True,
        )
        if hint:
            materials.append(hint)
            break
    return "\n\n".join(materials)


async def _send_reply(bot: Bot, event: Event, text: str, face: int | None = None,
                      quote: bool = True) -> None:
    """Send one coherent response; chat and task length policies are applied upstream.

    ``quote=False`` 只用于多气泡的后续气泡：不再引用原消息，避免同一回复刷出多条引用。
    """
    parts = [MessageSegment.text(text)]
    if quote and isinstance(event, GroupMessageEvent):
        parts.insert(0, MessageSegment.reply(event.message_id))
    if face is not None:
        parts.append(MessageSegment.face(face))
    await bot.send(event=event, message=Message(parts))


async def _send_bubbles(bot: Bot, event: Event, bubbles: list[str],
                        face: int | None = None) -> None:
    """逐条发送闲聊气泡：首条带引用、后续纯文本，间隔 0.6~1.2 秒；表情挂最后一条。"""
    total = len(bubbles)
    for i, line in enumerate(bubbles):
        await _send_reply(
            bot, event, line,
            face=face if i == total - 1 else None,
            quote=(i == 0),
        )
        if i < total - 1:
            await asyncio.sleep(random.uniform(0.6, 1.2))


def _log_sent(scope: str, text: str, lines: int = 1) -> None:
    """[SENT] 日志行（2026-10-12）：大肥鱼桌宠的状态面板靠它显示"最近说的一句"，
    格式与 chat.py（若叶睦/薇欧拉）保持一致；表情等附加动作另记 [ANON ACT] 行。"""
    print(f"[SENT] {scope} persona=千早爱音 lines={lines} text={text[:60]!r} sticker=None", flush=True)


HELP_TEXT = (
    "诶嘿嘿，想找我玩的话记住这些就好～\n"
    "· 群里 @我、引用我，或者直接叫“爱音”，我就会回。\n"
    "· 私聊也可以，我会认真听。\n"
    "· 聊天、吐槽、陪你说话都行；解题、写作、查资料、出主意这类正经事也可以交给我。\n"
    "· 我不会替你发消息、改文件、登录账号或做其他外部操作；太越界的话题我会拒绝。\n"
    "· 我记性一般，但你说过的小事我会尽量记住。"
)


def _is_help_request(event: Event) -> bool:
    return anon_context.is_help_request(_message_text(event))


help_matcher = on_message(rule=to_me() & Rule(_is_help_request), priority=5, block=True)


@help_matcher.handle()
async def handle_help(bot: Bot, event: Event):
    user_id = str(event.user_id)
    if user_id == BOT_QQ or user_id in OTHER_BOTS:
        return
    if isinstance(event, GroupMessageEvent):
        if ALLOWED_GROUPS and int(event.group_id) not in ALLOWED_GROUPS:
            return
        scope = f"group:{event.group_id}"
    else:
        scope = f"private:{user_id}"
    await _send_reply(bot, event, HELP_TEXT)
    _log_sent(scope, HELP_TEXT)
    print("[ANON HELP] sent", flush=True)


def _max_tokens_for(max_chars: int) -> int:
    return TASK_MAX_TOKENS if max_chars > agent_core.PLANNER_OUTPUT_CHARS else CHAT_MAX_TOKENS


async def _ask_deepseek(
    prompt: str,
    model: str,
    reasoning_effort: str,
    max_chars: int,
    timeout: int,
    system_prompt: str = "",
) -> str:
    if client is None:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    effort = "off" if DISABLE_REASONING else reasoning_effort
    print(f"[ANON MODEL] model={model} effort={effort or 'default'}", flush=True)
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": _max_tokens_for(max_chars),
        "temperature": 0.7,
    }
    if model.lower().endswith("reasoner"):
        kwargs.pop("temperature")
    if effort:
        kwargs["reasoning_effort"] = effort
    for attempt in range(2):
        if _circuit_open():
            raise RuntimeError("LLM circuit is open")
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(**kwargs), timeout=timeout
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"DeepSeek request exceeded {timeout}s") from exc
        usage = getattr(response, "usage", None)
        if usage is not None:
            cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            print(
                f"[ANON USAGE] model={model} prompt={prompt_tokens} "
                f"completion={completion_tokens} cached={cached} total={total_tokens}",
                flush=True,
            )
        _record_usage(usage)
        raw_answer = (response.choices[0].message.content or "").strip()
        answer = _clean_answer(raw_answer, max_chars=max_chars,
                               bubbles=max_chars <= SEARCH_REPLY_CHARS)
        if answer:
            return answer
        if attempt == 0:
            print("[ANON EMPTY] retrying", flush=True)
    raise RuntimeError("DeepSeek returned an empty reply")


async def _ask_chat(
    prompt: str, max_chars: int = MAX_REPLY_CHARS, timeout: int | None = None
) -> str:
    return await _ask_deepseek(
        prompt,
        CHAT_MODEL,
        CHAT_REASONING_EFFORT,
        max_chars,
        timeout if timeout is not None else CHAT_TIMEOUT_SECONDS,
        system_prompt=SYSTEM_PROMPT,
    )


async def _ask_task(prompt: str, max_chars: int = MAX_TASK_REPLY_CHARS) -> str:
    return await _ask_deepseek(
        prompt,
        TASK_MODEL,
        TASK_REASONING_EFFORT,
        max_chars,
        TASK_TIMEOUT_SECONDS,
        system_prompt=SYSTEM_PROMPT,
    )


async def _refresh_group_summary(group_id: str) -> None:
    """滚动摘要（WP4）：把攒下的群消息压缩成一段背景总结并落盘。

    失败时保留缓冲不清理，等下一轮计数再试。同一群同一时间只跑一个
    摘要任务：后续任务在锁上排队，等前一个写回之后才快照 group_summaries
    和消息缓冲，避免用启动时读到的旧摘要把刚落盘的新摘要覆盖回去。
    """
    async with summary_locks[group_id]:
        try:
            if _circuit_open():
                print("[ANON SUMMARY] circuit open, skip", flush=True)
                return
            raw = list(group_recent_raw.get(group_id, []))[:SUMMARY_INTERVAL]
            if len(raw) < 3:
                return
            lines = [anon_context.format_recent_line(item) for item in raw]
            prompt = anon_context.build_summary_prompt(
                group_summaries.get(group_id, ""), lines
            )
            answer = await _ask_deepseek(
                prompt, CHAT_MODEL, "off", SUMMARY_MAX_CHARS, 90, system_prompt=""
            )
            group_summaries[group_id] = answer[:SUMMARY_MAX_CHARS]
            _save_memory()
            print(f"[ANON SUMMARY] group={group_id} {answer[:60]!r}", flush=True)
            processed = {id(item) for item in raw}
            group_recent_raw[group_id] = [
                item for item in group_recent_raw.get(group_id, [])
                if id(item) not in processed
            ]
        except Exception as exc:
            print(f"[ANON SUMMARY ERROR] {type(exc).__name__}: {exc}", flush=True)


async def _distill(need: str, result: str) -> None:
    """任务后沉淀（WP5）：让小模型判断这次任务有没有值得记进经验库的东西。

    整个函数吞异常：沉淀失败绝不影响已交付的答案。"""
    try:
        prompt = (
            f"这次的任务：{need}\n交付内容开头：{result[:200]}\n\n"
            "只输出一行 JSON：{\"layer\":\"L2或L3或none\",\"text\":\"60字以内\"}"
            "（L2=关于这个群的人/梗/约定/喜好；L3=这类任务怎么做的经验；没有值得记的用 none）"
        )
        raw = await _ask_deepseek(prompt, CHAT_MODEL, "off", 200, 45, system_prompt="")
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            return
        data = json.loads(match.group(0))
        layer = str(data.get("layer", "")).strip().upper() if isinstance(data, dict) else ""
        text = str(data.get("text", "")).strip() if isinstance(data, dict) else ""
        if layer in ("L2", "L3") and text:
            tags = re.findall(r"[\u4e00-\u9fff]{2,4}", text)[:4]
            experience.learn(layer, text, tags=tags)
            print(f"[ANON DISTILL] {layer} {text[:40]!r}", flush=True)
    except Exception as exc:
        print(f"[ANON DISTILL ERROR] {type(exc).__name__}: {exc}", flush=True)


@handler.handle()
async def handle_message(bot: Bot, event: Event):
    received_at = time.time()
    user_id = str(event.user_id)
    if user_id == BOT_QQ or user_id in OTHER_BOTS:
        return

    message_type = getattr(event, "message_type", None)
    is_private = isinstance(event, PrivateMessageEvent) or message_type == "private"
    if not is_private:
        if not isinstance(event, GroupMessageEvent):
            return
        if ALLOWED_GROUPS and int(event.group_id) not in ALLOWED_GROUPS:
            return

    user_name = _sender_name(event)
    user_text = _message_text(event)
    profile_text = user_text  # 档案只记群友真实说的话；bare @ 的合成文本不写入

    group_id = str(getattr(event, "group_id", ""))
    if not is_private:
        group_contexts[group_id].append(
            {"uid": user_id, "name": user_name, "text": (user_text or "[空消息]")[:160]}
        )
        recent = group_recent_raw[group_id]
        recent.append((user_name, user_text[:80], time.time()))
        if len(recent) > SUMMARY_INTERVAL:
            del recent[: len(recent) - SUMMARY_INTERVAL]
        group_msg_count[group_id] += 1
        if SUMMARY_ENABLED and group_msg_count[group_id] >= SUMMARY_INTERVAL:
            group_msg_count[group_id] = 0
            task = asyncio.create_task(_refresh_group_summary(group_id))
            summary_tasks.add(task)
            task.add_done_callback(summary_tasks.discard)
        triggered, reason = _is_triggered(event, user_text, str(bot.self_id))
        if not triggered:
            # 未叫到她也要记档案（认人靠长期观察，不靠被叫到的那一次）。
            _record_profile(user_id, user_name, profile_text, received_at)
            return
        # WP1：只有触发消息才读图，避免每条群图都花钱；当前消息 + 被引用消息都扫。
        image_urls = _image_urls(event)
        if not user_text:
            # A bare @/reply carries no text; answer it instead of dropping the event.
            # 若裸 @ 前后带图，告诉模型“发了一张图片”，不要只说“没有说内容”。
            user_text = "[发了一张图片]" if image_urls else "[只是叫了你一下，没有说内容]"
        context_snapshot = list(group_contexts[group_id])[:-1][-MAX_GROUP_CONTEXT:]
        scope = f"group:{group_id}"
        current_summary = group_summaries.get(group_id, "")
    else:
        if not user_text:
            return
        reason = "private"
        image_urls = _image_urls(event)
        context_snapshot = []
        scope = f"private:{user_id}"
        current_summary = ""

    print(
        f"[ANON HANDLE] sender={user_id} name={user_name!r} scope={scope} "
        f"reason={reason} chars={len(user_text)}",
        flush=True,
    )
    # 认人上下文：先按“更新前”的档案渲染（上次互动/上次说过什么），再记录本条。
    # 顺序反了会让「TA 上次说」永远回显当前这句，等于没有记忆。
    user_profile = anon_people.render(user_profiles.get(user_id), user_id, user_name)
    if user_profile:
        print(
            f"[ANON PEOPLE] inject sender={user_id} name={user_name!r} "
            f"seen={user_profiles.get(user_id, {}).get('count', '?')}",
            flush=True,
        )
    _record_profile(user_id, user_name, profile_text, received_at)
    if anon_context.is_stale(received_at, time.time(), STALE_SECONDS):
        print(
            f"[ANON STALE] sender={user_id} scope={scope} "
            f"age={time.time() - received_at:.0f} stage=before_llm",
            flush=True,
        )
        return

    # WP1（2026-10-26）：触发消息（含被引用图）先读图；失败也注入“读取失败”事实。
    # 放在 stale 检查之后，避免对已经过期的消息白花视觉调用；放在 memo/planner 之前，
    # 让 planner、voice、task 三条链路都能看到 [图片内容：...]。
    if image_urls:
        user_text = await _augment_text_with_images(user_text, image_urls)

    # WP5 记忆指令：触发检查已通过、尚未进入任何 LLM 调用；命中就直接学习并收尾。
    memo = anon_context.parse_memory_command(user_text)
    if memo:
        experience.learn(
            "L2",
            memo,
            tags=re.findall(r"[\u4e00-\u9fff]{2,4}", memo)[:4],
            about=user_id,
        )
        reply = random.choice(
            ["记住了～你说过的小事我都会尽量记着。", "嗯嗯，记下来了！", "记下了，下次忘不了～"]
        )
        await _send_reply(bot, event, reply)
        _log_sent(scope, reply)
        print(f"[ANON MEMO] sender={user_id} learned={memo[:40]!r}", flush=True)
        return

    session_key = (scope, user_id)
    async with scope_locks[scope]:
        # 队列等待只有拿到锁之后才看得见：进 LLM 前那次检查几乎总是 0 秒。
        if anon_context.is_stale(received_at, time.time(), STALE_SECONDS):
            print(
                f"[ANON STALE] sender={user_id} scope={scope} "
                f"age={time.time() - received_at:.0f} stage=after_lock",
                flush=True,
            )
            return
        # 动作边界与 plan 分支无关：planner 失败（plan=None）或判成 task 时
        # 也必须拿到固定拒绝、假承诺改写和表情抑制所需的 kind。
        request_kind = anon_search_policy.classify_request(user_text)
        boundary_note = anon_search_policy.action_note(request_kind)
        history = histories[session_key]
        first_message = False  # 2026-10-25 用户指令：取消首见固定开场白，一律按已认识处理
        task_mode = False
        search_used = False
        plan = None  # understand() 抛错时 final send 仍要引用 plan，先兜底
        selected_model = CHAT_MODEL
        answer = ""
        ask_chat = lambda prompt, limit: _ask_chat(prompt, max_chars=limit)
        ask_plan = lambda prompt, limit: _ask_chat(
            prompt, max_chars=limit, timeout=PLANNER_TIMEOUT
        )
        ask_voice = lambda prompt, limit: _ask_chat(
            prompt, max_chars=limit, timeout=VOICE_TIMEOUT
        )
        ask_task = lambda prompt, limit: _ask_task(prompt, max_chars=limit)

        try:
            exp_hint = experience.render_hint(user_text)
        except Exception as exc:
            print(f"[ANON EXP HINT] render failed: {type(exc).__name__}: {exc}", flush=True)
            exp_hint = ""

        try:
            plan = await agent_core.understand(
                user_text,
                user_name,
                list(history),
                context_snapshot,
                ask_plan,
                first_message=first_message,
                summary=current_summary,
                experience_hint=exp_hint,
                user_profile=user_profile,
            )
            if plan and plan.get("mode") == "chat":
                answer = ""
                ctx_text = "\n".join(
                    anon_context.format_group_context(context_snapshot, limit=8, keep=160)
                )
                intent = anon_search_policy.detect_lookup(
                    user_text, summary=current_summary, context=ctx_text
                )
                gloss = anon_search_policy.glossary_hint(user_text)
                queries = list(intent.get("queries") or []) if intent else []
                if not queries and plan.get("fresh"):
                    queries = [anon_search_policy.strip_noise(user_text)[:60]]
                # WP2：词表只对“求定义”免搜（群内用法优先于搜索标题）；
                # explicit / performance / current 即使命中词表也照搜（演出评价题）。
                if intent and intent.get("reason") == "definition" and gloss:
                    queries = []
                queries = [
                    str(item).strip()[:60] for item in queries if str(item).strip()
                ][:2]
                print(
                    f"[ANON INTENT] kind={request_kind} "
                    f"lookup={intent.get('reason') if intent else 'none'} "
                    f"queries={'|'.join(_log_safe(item) for item in queries) if queries else '-'}",
                    flush=True,
                )
                search_material = await _quick_search_materials(queries)
                search_used = bool(search_material)
                limit = SEARCH_REPLY_CHARS if search_used else MAX_REPLY_CHARS
                draft = plan.get("reply", "")
                try:
                    answer = await asyncio.wait_for(
                        agent_core.compose_chat_reply(
                            user_text,
                            user_name,
                            list(history),
                            context_snapshot,
                            plan,
                            ask_voice,
                            summary=current_summary,
                            experience_hint=exp_hint,
                            search_hint=search_material,
                            glossary_hint=gloss,
                            action_note=boundary_note,
                            max_chars=limit,
                            draft=draft,
                            quoted_text=_quoted_text(event),
                            user_profile=user_profile,
                        ),
                        timeout=VOICE_TIMEOUT,
                    )
                except Exception as exc:
                    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                        print("[ANON TIMEOUT] stage=voice", flush=True)
                    print(
                        f"[ANON VOICE] failed: {type(exc).__name__}: {exc} "
                        f"draft={len(draft)} used=no",
                        flush=True,
                    )
                    answer = ""
                answer = (answer or "").strip()
                if not answer:
                    answer = _clean_answer(draft, limit)
                if not answer and request_kind in ACTION_FALLBACK_REPLIES:
                    answer = ACTION_FALLBACK_REPLIES[request_kind]
            elif plan and plan.get("mode") == "task" and plan.get("agent") in agent_core.SPECIALISTS:
                task_mode = True
                try:
                    await _send_reply(bot, event, "我看看，稍等一下～")
                    _log_sent(scope, "我看看，稍等一下～")
                    print("[ANON SENT] task acknowledgement", flush=True)
                except Exception as exc:
                    print(f"[ANON SEND] task acknowledgement failed: {type(exc).__name__}: {exc}", flush=True)
                # WP6（2026-10-26）：researcher 的 simple 任务也必须走 _research（内含 agent_search），
                # 否则 24h 7 个 task 全 complex=False，researcher 从未联网。
                if plan.get("complex") or plan.get("agent") == "researcher":
                    selected_model = TASK_MODEL
                    answer = await agent_core.run_subagent(
                        plan,
                        user_text,
                        user_name,
                        context_snapshot,
                        ask_task,
                        action_note=boundary_note,
                    )
                else:
                    answer = await agent_core.answer_simple_task(
                        plan,
                        user_text,
                        user_name,
                        list(history),
                        context_snapshot,
                        ask_chat,
                        action_note=boundary_note,
                    )
        except Exception as exc:
            print(f"[ANON ORCH] classification/task failed: {type(exc).__name__}: {exc}", flush=True)

        if not answer:
            prompt = _build_prompt(
                history,
                context_snapshot,
                user_name,
                user_text,
                _quoted_text(event),
                first_message,
                summary=current_summary,
                experience_hint=exp_hint,
                user_profile=user_profile,
            )
            if boundary_note:
                prompt += f"\n【本轮动作边界】（本轮边界以此为准）{boundary_note}"
            try:
                answer = await _ask_chat(prompt, timeout=VOICE_TIMEOUT)
            except Exception as exc:
                if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
                    print("[ANON TIMEOUT] stage=fallback", flush=True)
                print(f"[ANON ERROR] {type(exc).__name__}: {exc}", flush=True)
                # 模型全挂时，受限请求也要给固定拒绝，不能只发“线路忙”了事。
                answer = ACTION_FALLBACK_REPLIES.get(
                    request_kind, "诶？刚刚线路有点忙……等我缓一下再聊好不好？"
                )

        output_limit = (
            MAX_TASK_REPLY_CHARS if task_mode
            else (SEARCH_REPLY_CHARS if search_used else MAX_REPLY_CHARS)
        )
        answer = _clean_answer(answer, max_chars=output_limit, bubbles=not task_mode)
        # 受限请求不许在模型嘴里“假装已经执行”；命中承诺句就换成固定拒绝短句。
        # 任务分支同样适用：planner 把“发图/转告/抢票”判成 task 时也不能假装执行。
        if request_kind in ACTION_FALLBACK_REPLIES and _claims_fake_action(
            answer, request_kind
        ):
            answer = ACTION_FALLBACK_REPLIES[request_kind]
        # 闲聊答案在搜索/模型里拖过 120 秒就不再发：对话已经翻篇，宁可不说也不答非所问。
        # 任务不受此限：ack 已发出，复杂任务允许多轮 240 秒，答案必须交付。
        if not task_mode and anon_context.is_stale(
            received_at, time.time(), STALE_SECONDS
        ):
            print(
                f"[ANON STALE] sender={user_id} scope={scope} "
                f"age={time.time() - received_at:.0f} stage=before_send",
                flush=True,
            )
            return
        print(
            f"[ANON REPLY] generated chars={len(answer)} mode={'task' if task_mode else 'chat'} model={selected_model}",
            flush=True,
        )
        history.append({"role": "user", "name": user_name, "content": user_text[:500]})
        history.append({"role": "assistant", "content": answer})
        if first_message:
            greeted_users.add(user_id)
            _save_greeted_users()

        act = plan.get("act", "none") if isinstance(plan, dict) else "none"
        # WP2 动作边界：拒绝类请求不附表情（附了像答应了）；群友要表情时才补一个经典表情。
        # 任务分支也要抑制受限动作，否则 planner 把“发图/转告/抢票”判成 task 时
        # 仍可能附一个表情，等于变相答应。
        if request_kind in {"send_media", "contact_third_party", "purchase_login"}:
            act = "none"
        elif not task_mode and request_kind == "send_face":
            act = anon_search_policy.pick_face_action(user_text)
        face_id = agent_core.FACE_ACTIONS.get(act[5:]) if act.startswith("face:") else None
        send_bubbles = (
            [line for line in answer.splitlines() if line.strip()]
            if not task_mode else []
        )
        if len(send_bubbles) > 1:
            print(
                "[ANON BUBBLES] n=%d %s" % (
                    len(send_bubbles),
                    " | ".join(
                        f"{i + 1}:{bubble[:60]!r}"
                        for i, bubble in enumerate(send_bubbles)
                    ),
                ),
                flush=True,
            )
            await _send_bubbles(bot, event, send_bubbles, face=face_id)
        else:
            await _send_reply(bot, event, answer, face=face_id)
        _log_sent(scope, answer, lines=max(1, len(send_bubbles)))
        if face_id is not None:
            print(f"[ANON ACT] {act}", flush=True)
        print("[ANON SENT] one coherent message accepted by OneBot", flush=True)

        # WP5 任务后沉淀：任务答案已成功发出，节流 + 熔断检查后异步提炼经验。
        if task_mode and isinstance(plan, dict):
            need = plan.get("need") or user_text
            if (
                time.monotonic() - _last_distill.get(scope, 0) >= DISTILL_INTERVAL
                and not _circuit_open()
            ):
                _last_distill[scope] = time.monotonic()
                distill_task = asyncio.create_task(_distill(need, answer))
                distill_tasks.add(distill_task)
                distill_task.add_done_callback(distill_tasks.discard)
