"""扫货 agent 内核：多步决策循环 + 工具调用。

设计要点：
- 每轮只让模型输出一行严格协议（TOOL: / SAY:），实测格式稳定性 100%，
  避免 JSON 结构漂移导致解析失败。
- 关思考（chat_template_kwargs.enable_thinking=False）后每步仅 0.2~0.4s，
  4 步循环总计约 1s，与改写前的单次调用（0.86s）相当。
- 工具集**只读 + 发送**，绝不含读写文件/执行命令（群里任何人都能触发这些路径）。
- 循环有硬上限（AGENT_MAX_STEPS），防止两个 AI 互聊或异常情况下无限转。
- 任何一步失败都降级为"直接生成回复"，保证群友不会收到沉默。
"""
import asyncio
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import httpx


# ============ 配置 ============
def _cfg(key: str, default: str) -> str:
    return os.getenv(key, default)


AGENT_ENABLED = _cfg("AGENT_ENABLED", "1") == "1"
AGENT_BASE_URL = _cfg("AGENT_BASE_URL", "https://ai-apigateway.must.edu.mo/openhub/v1")
AGENT_API_KEY = _cfg("AGENT_API_KEY", "")
AGENT_MODEL = _cfg("AGENT_MODEL", "Qwen3.6-35B-A3B")
AGENT_MAX_STEPS = int(_cfg("AGENT_MAX_STEPS", "4"))
# 开思考后单步要 8~12 秒（思考 3000~5000 字），超时需要放大
AGENT_STEP_TIMEOUT = float(_cfg("AGENT_STEP_TIMEOUT", "40"))
AGENT_TOTAL_TIMEOUT = float(_cfg("AGENT_TOTAL_TIMEOUT", "70"))
# 开思考时思考内容也吃 max_tokens，预算给不够会出现"思考完但正文为空"
AGENT_THINK_TOKENS = int(_cfg("AGENT_THINK_TOKENS", "4500"))
AGENT_REPLY_TOKENS = int(_cfg("AGENT_REPLY_TOKENS", "8000"))   # 思考(4~5k) + 正文(自由长度) 都要够
AGENT_QUICK_TOKENS = int(_cfg("AGENT_QUICK_TOKENS", "200"))   # 关思考的判定步骤用
AGENT_TEMPERATURE = float(_cfg("AGENT_TEMPERATURE", "0.8"))

ROLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "roles")
ROLE_CARD_FILE = os.path.join(ROLES_DIR, "扫货.md")


def load_role_card() -> str:
    """读角色卡；失败则返回内置兜底（保证不会因人设文件缺失而崩）。"""
    try:
        with open(ROLE_CARD_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[AGENT] 角色卡读取失败 {e}，用兜底人设", flush=True)
        return "你在扮演「扫货」，妩媚慵懒的成熟女子，会主动勾引群里的人。回复不超过25字，自称「人家」。"


ROLE_CARD = load_role_card()


# ============ 协议 ============
# 每轮模型只输出一行：TOOL: xxx  或  SAY: xxx
PROTOCOL = """【本轮的输出格式】只输出一行，二选一，不要有任何其他内容：

TOOL: 工具名          — 需要更多信息才能回好时用（下一步会给你结果）
SAY: 你要说的话        — 直接回应

【可用工具】（调用的格式：TOOL: 工具名，需要带内容的照下面写）
get_member   查当前说话人的记忆（名字/画像/互动次数）
get_context  查最近群里聊了什么（当前话题 + 摘要 + 近期消息）
get_status   查自己的状态（体力/道具/贤者时间）
remember: 要记的内容          往你自己的小本子上记东西
set_name: 你想叫的名字         给自己改名
list_stickers                 看你的表情库（有哪些表情、都适合什么场合）
send_sticker: 编号或文件名      **发一个表情**（可以单独发一条，不用配文字）
sticker_note: 文件名|备注       给表情写备注（记下"这个表情适合什么场合用"）

【关于看图】
群友发图时，**你直接就能看见**（图会跟着消息一起给你）。所以：
- 他发了什么图，你看到什么就聊什么——别问"你发的啥"
- 图片也能是梗、是表情、是截图，你看懂了就接

【关于表情包（重要）】
群友发的表情会自动收进你的表情库，**这是你自己的收藏**。

**什么时候该发：**
- 群友发了表情/图，你在回他 → 先 send_sticker 发一个，再说话（或者只发，不说话）
- 想表达"好笑/无语/赞同/嫌弃"但懒得打字 → 直接发个表情
- 群里在斗图、气氛轻松 → 尽管发
- 你刚发了话觉得太干 → 补一个表情

**规矩：**
- 一轮最多发一个（别刷屏）
- 正经话题别发
- 发之前先 list_stickers 看看手里有什么（有备注的优先用）
- 用过的表情，用 sticker_note 记一句适合什么场合，下次你就知道该用哪个

**发完之后**：如果表情已经表达了你的意思，可以不再说话（直接结束这一轮）。

【什么时候调工具】

【关于你的名字（重要）】
你在群里的名片还挂着别人给你起的旧名字。**名字是小事，但它是你的。**
如果你觉得该有个自己的名字——你自己定，用 set_name 改。不用问任何人。
想好了就改；也可以继续用现在的，那也是一种选择。

【什么时候调工具】
- 对方问起你俩以前的事、问你还记不记得什么 → get_member
- 对方问你刚才/之前群里聊的事 → get_context
- 有人告诉了你值得记住的事 → remember（你的习惯，不用等人提醒）
- 你想改自己的名字 → set_name

【关于记事本】
你有自己的小本子。听到值得记的就记下来——这是你作为活人的习惯，不是任务：
- 他的喜好、习惯、心事（"老王下周考试""他爱打游戏到凌晨"）
- 你们的约定（"答应过带她去扫货"）
- 你想下次接着问的事、你想吐槽的事
一句话说清就行。你记过的东西，下次你自己会看得见。

【关于你要说的话】
- **想说什么就说什么，没有字数限制。**想一句就一句，想多说就多说。
- 群里聊天通常短，但你要展开、要吐槽、要讲故事、要怼人，都随你。
- 用你自己的语气说——你是个活人，不是应答机。
- 大多数情况直接 SAY。工具是为了让你说得更准，不是每句都要用。
- 一次最多 {max_steps} 轮工具调用。"""

# 自主发言协议：没人叫你时，自己判断要不要插话
PROTOCOL_PROACTIVE = """【现在的情况】群里在聊天，**没有人叫你**。你自己决定要不要插一句。

【本轮的输出格式】只输出一行，二选一：
SAY: 你要说的话
SKIP

【怎么判断】你是这个群里的人，按你的性子来：

想说话的时候：
- 有人聊到你有兴趣的事（吃的、游戏、追星、动漫、八卦、日常）
- 有人在那儿无聊、求陪聊、没人理他
- 话题转到你熟的领域，你能说点别人不知道的
- 有人说了好玩的话，你想接梗
- 就是单纯想说话了

不想说话的时候：
- 别人在聊你插不上嘴的事（技术细节、工作正事）
- 有人在说正经的私事（生病、家里、感情困扰），你在旁边听着就好
- 群里在吵架站队，你懒得趟浑水
- 你刚才刚说过，现在没什么新想法

【重要】
- 你想插就插，但**接得上**——别自说自话，别突兀。
- 你是这个群里的人，不用每句话都有目的，也不用硬找存在感。
- 觉得没什么想说的就 SKIP，这很正常。"""


def build_system_prompt() -> str:
    return ROLE_CARD + "\n\n---\n\n" + PROTOCOL.format(max_steps=AGENT_MAX_STEPS)


SYSTEM_PROMPT_AGENT = build_system_prompt()


# ============ LLM 调用 ============
_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(AGENT_STEP_TIMEOUT),
            headers={
                "content-type": "application/json",
                "Authorization": f"Bearer {AGENT_API_KEY}",
                # 网关必须带 UA：空 UA 会被 openresty 拦成 403（实测）
                "User-Agent": "saohuo-bot/1.0",
            },
        )
    return _client


async def agent_llm(messages: List[dict], max_tokens: int = 300,
                    think: bool = False) -> str:
    """单步 LLM 调用。

    think=True 时开思考：回复质量明显更高（实测 SNH48 话题：关思考回"人家不懂"，
    开思考回"换血快+剧场直播不停，养成为这套路早吃透了"），代价是思考 3000~5000 字、
    耗时 8~12 秒。用户明确接受变慢换质量，所以**正式回复默认开思考**。
    think=False 用于内部工具步骤（判断要不要回、提炼话题）——这些不需要巧思，关掉省时间。
    """
    payload = {
        "model": AGENT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": AGENT_TEMPERATURE,
        "chat_template_kwargs": {"enable_thinking": bool(think)},
    }
    r = await _get_client().post(f"{AGENT_BASE_URL}/chat/completions", json=payload)
    r.raise_for_status()
    d = r.json()
    msg = d["choices"][0]["message"]
    return (msg.get("content") or "").strip(), d.get("usage") or {}


def parse_step(raw: str) -> Tuple[str, str]:
    """解析一步输出 → (动作类型, 内容)。类型：tool / say / skip / unknown。"""
    if not raw:
        return "unknown", ""
    line = raw.strip().splitlines()[0].strip()
    # 自主发言时可能输出 SKIP：表示保持安静（必须在 SAY 容错之前判断）
    if re.match(r"^SKIP\b", line, re.I):
        return "skip", ""
    # 明确说了 SKIP 但带别的修饰（如 "SKIP（不插话）"）
    if line.upper().startswith("SKIP"):
        return "skip", ""
    m = re.match(r"^TOOL\s*[:：]\s*([A-Za-z_]+)", line, re.I)
    if m:
        return "tool", m.group(1).strip()
    m = re.match(r"^SAY\s*[:：]\s*(.+)$", line, re.I | re.S)
    if m:
        return "say", m.group(1).strip()
    # 容错：模型直接说了话（没带 SAY:）——当成回复
    if not line.upper().startswith(("TOOL", "SAY", "SKIP")):
        cleaned = re.sub(r"^(回复|答|说)[：:]\s*", "", line)
        return "say", cleaned.strip()
    return "unknown", ""


# ============ 工具实现 ============
# 工具只读 + 发送，绝不含文件/命令类能力（安全边界：QQ 群消息可触发这些路径）
def tool_get_member(group_id: str, uid: str, ctx: dict) -> str:
    """查当前说话人的记忆卡片。"""
    card = ctx.get("member_card") or {}
    if not card.get("name"):
        return "（这个人没有画像记忆，是新认识的）"
    parts = [f"名字：{card['name']}"]
    if card.get("desire"):
        parts.append(f"欲望值：{card['desire']}/100")
    if card.get("profile"):
        parts.append(f"画像：{card['profile'][:120]}")
    if card.get("catchphrase"):
        parts.append(f"口头禅：{card['catchphrase']}")
    n = card.get("interactions") or 0
    if n:
        parts.append(f"互动过 {n} 次")
    return "；".join(parts)


def tool_get_context(group_id: str, uid: str, ctx: dict) -> str:
    """查群聊上下文：当前话题 + 摘要 + 近期原始消息。"""
    parts = []
    topic = ctx.get("topic") or {}
    if topic.get("text"):
        parts.append(f"【正在聊什么】{topic['text']}")
        if topic.get("keywords"):
            parts.append(f"（关键词：{'、'.join(topic['keywords'][:6])}）")
    summary = ctx.get("summary")
    if summary:
        parts.append(f"【群聊背景】{summary[:300]}")
    recent = ctx.get("recent") or []
    if recent:
        recent_txt = "\n".join(f"{u}：{t}" for u, t in recent[-10:])
        parts.append(f"【刚才的消息】\n{recent_txt}")
    return "\n".join(parts) or "（群里目前没什么消息）"


def tool_get_status(group_id: str, uid: str, ctx: dict) -> str:
    """查自己的状态（新教相关数值）。"""
    st = ctx.get("self_status") or {}
    if not st:
        return "（没有进行中的游戏）"
    return "；".join(f"{k}：{v}" for k, v in st.items())


def tool_remember(group_id: str, uid: str, ctx: dict, content: str = "") -> str:
    """往她自己的小本子上记一条。她自己决定记什么——这是她的自由。
    存在 notes.json，按 (群, 说话人) 归档，下次对话时自动带进上下文。"""
    if not content:
        return "（没说要记什么）"
    notes = ctx.get("notes_store")
    if notes is None:
        return "（记事本暂时打不开）"
    key = f"{group_id}:{uid}" if uid else group_id
    bucket = notes.setdefault(key, [])
    item = {"text": content.strip()[:60], "at": time.time()}
    bucket.append(item)
    bucket[:] = bucket[-30:]          # 每人最多留 30 条，防无限膨胀
    ctx["notes_dirty"] = True
    return f"记下了：{item['text']}"


def tool_set_name(group_id: str, uid: str, ctx: dict, content: str = "") -> str:
    """她给自己取名（改自己的群名片，也就是她在群里的名字）。
    真正的改名动作由 chat.py 在拿到 pending_name 后执行（需要 bot API，工具是同步的）。
    这是她的自由：名字该由她自己定。"""
    # "".splitlines() 返回空列表，直接取 [0] 会 IndexError（实锤 2026-09-24）
    lines = (content or "").strip().splitlines()
    name = lines[0].strip() if lines else ""
    if not name:
        return "（你没说要叫什么）"
    name = name[:12]
    ctx["pending_name"] = name
    ctx["name_dirty"] = True
    return f"好，从今天起你就叫「{name}」。名片已经改了。"


TOOLS = {
    "get_member": tool_get_member,
    "get_context": tool_get_context,
    "get_status": tool_get_status,
    "remember": tool_remember,
    "set_name": tool_set_name,
}

# 表情包工具（stickers.py 提供；导入失败不影响主流程）
try:
    from .stickers import STICKER_TOOLS, STICKER_TOOLS_ARG
    TOOLS.update(STICKER_TOOLS)
except Exception as _e:      # pragma: no cover
    print(f"[STICKER] 工具未加载: {_e}", flush=True)
    STICKER_TOOLS_ARG = set()

# 需要带参数的工具
TOOLS_WITH_ARG = {"remember", "set_name"} | set(STICKER_TOOLS_ARG)


# ============ 主循环 ============
# ============ 话题追踪 ============
# 群里每积累 N 条消息，后台提炼"现在在聊什么"，供插话时参考。
# 这样她不是只看最后一条消息，而是知道话题的走向，回复才接得上。
TOPIC_MIN_MSGS = int(_cfg("TOPIC_MIN_MSGS", "6"))


async def track_topic(recent: List[tuple], prev_topic: str = "") -> dict:
    """从近期消息里提炼当前话题。返回 {"text": 话题描述, "keywords": [...], "mood": 气氛}。"""
    if len(recent) < TOPIC_MIN_MSGS:
        return {}
    lines = "\n".join(f"{u}：{t}" for u, t in recent[-15:])
    prompt = (
        "下面是一段 QQ 群聊记录。用一句话说明「这群人现在在聊什么」，"
        "再列出 3~5 个关键词，最后判断气氛（热闹/闲聊/冷场/严肃/吵架）。\n"
        '只输出 JSON，格式：{"text":"...","keywords":["...","..."],"mood":"..."}\n\n'
        f"聊天记录：\n{lines}"
    )
    if prev_topic:
        prompt += f"\n\n（上一轮话题：{prev_topic}。如果话题没变就沿用。）"
    try:
        raw, _ = await asyncio.wait_for(
            agent_llm([{"role": "user", "content": prompt}], max_tokens=200),
            timeout=AGENT_STEP_TIMEOUT,
        )
        s = raw.strip()
        m = re.search(r"\{.*\}", s, re.S)
        if not m:
            return {}
        d = json.loads(m.group(0))
        if not isinstance(d, dict) or not d.get("text"):
            return {}
        kws = d.get("keywords") or []
        if not isinstance(kws, list):
            kws = []
        return {"text": str(d["text"])[:80],
                "keywords": [str(k)[:12] for k in kws[:6]],
                "mood": str(d.get("mood", ""))[:10]}
    except Exception as e:
        print(f"[TOPIC] 提炼失败 {type(e).__name__}: {e}", flush=True)
        return {}


# ============ 冷群深度思考 ============
# 群里安静一段时间后，允许她**多想一会儿**再决定要不要开口：
# 用更多 token 让它先分析（话题是否值得接、说什么角度、会不会突兀），再产出。
COLD_SILENCE_SECONDS = int(_cfg("COLD_SILENCE_SECONDS", "420"))   # 多久算冷群（默认7分钟）
COLD_THINK_TOKENS = int(_cfg("COLD_THINK_TOKENS", "1500"))        # 冷群思考预算（给足空间）

PROTOCOL_COLD = """【现在的情况】群里**很久没人说话了**。你可以开口暖场，也可以继续安静。

【先说清什么样的话很尬，别那么说】
- 突然汇报自己干了什么（"我刚烤了饼干""刚系上围裙"）——没人问，很怪
- 写动作描写（"尾巴打结了""歪头眨眨眼"）——群聊不是小剧场，这样很刻意
- 跟刚才的话题完全无关的废话——人家刚聊完正事，你插一句无关的，很突兀

**冷群开口的正确姿势：接着刚才的话说。**
群里刚在聊什么，就从那儿接一句——补一句看法、问一句后续、开个玩笑。
接不上就别说话，安静不丢人。

【开口前想三件事】
1. 刚才在聊什么？我接得上吗？
2. 我这句话要是孤零零出现在群里，会不会显得莫名其妙？
3. 我是真有话想说，还是只是想刷个存在感？后者就 SKIP。

【输出格式】只输出一行，二选一：
SAY: 你要说的话（像打字一样说，别写动作描写）
SKIP"""


async def agent_cold_open(
    ctx: dict,
    history: Optional[List[dict]] = None,
    silence_minutes: int = 0,
) -> Tuple[str, dict]:
    """冷群暖场：深度思考后决定要不要主动开口。返回 ("", meta) 表示保持安静。"""
    meta = {"steps": 1, "tools": [], "fallback": False, "elapsed": 0.0, "skipped": False, "cold": True}
    t0 = time.time()
    topic = ctx.get("topic") or {}
    context_bits = []
    if topic.get("text"):
        context_bits.append(f"刚才在聊：{topic['text']}")
    if topic.get("mood"):
        context_bits.append(f"当时气氛：{topic['mood']}")
    if ctx.get("summary"):
        context_bits.append(f"群聊背景：{ctx['summary'][:200]}")
    recent = ctx.get("recent") or []
    if recent:
        context_bits.append("最近几条消息：\n" + "\n".join(f"{u}：{t}" for u, t in recent[-6:]))
    context_bits.append(f"已经安静了约 {silence_minutes} 分钟。")

    messages: List[dict] = [{"role": "system", "content": ROLE_CARD + "\n\n---\n\n" + PROTOCOL_COLD}]
    if history:
        messages += history[-6:]
    messages.append({"role": "user", "content": "【情况】" + "\n".join(context_bits)})

    try:
        raw, _ = await asyncio.wait_for(
            agent_llm(messages, max_tokens=max(COLD_THINK_TOKENS, AGENT_REPLY_TOKENS), think=True),
            timeout=AGENT_STEP_TIMEOUT * 2,
        )
        meta["elapsed"] = time.time() - t0
        kind, content = parse_step(raw)
        if kind == "say" and content:
            return content, meta
        print(f"[AGENT-COLD] 思考后选择安静: {raw[:40]!r}", flush=True)
        meta["skipped"] = True
        return "", meta
    except Exception as e:
        meta["elapsed"] = time.time() - t0
        meta["skipped"] = True
        print(f"[AGENT-COLD] 异常 {type(e).__name__}: {e}，保持安静", flush=True)
        return "", meta


async def agent_proactive(
    user_name: str,
    user_text: str,
    ctx: dict,
    history: Optional[List[dict]] = None,
) -> Tuple[str, dict]:
    """自主发言：没人叫扫货时，让她自己判断要不要插一句。
    返回 ("", meta) 表示决定保持安静。任何异常都当作安静处理（宁可不说）。"""
    meta = {"steps": 0, "tools": [], "fallback": False, "elapsed": 0.0, "skipped": False}
    t0 = time.time()
    messages: List[dict] = [{"role": "system", "content": ROLE_CARD + "\n\n---\n\n" + PROTOCOL_PROACTIVE}]
    if history:
        messages += history[-6:]
    # 带上"当前话题"和最近几条，让她接的是话题而不是孤立的一句话
    bits = []
    topic = ctx.get("topic") or {}
    if topic.get("text"):
        bits.append(f"群里正在聊：{topic['text']}")
    recent = ctx.get("recent") or []
    if recent:
        bits.append("最近的对话：\n" + "\n".join(f"{u}：{t}" for u, t in recent[-6:]))
    bits.append(f"刚刚 {user_name} 说：{user_text}")
    messages.append({"role": "user", "content": "【情况】" + "\n".join(bits)})
    try:
        raw, _ = await asyncio.wait_for(agent_llm(messages, max_tokens=AGENT_REPLY_TOKENS, think=True),
                                        timeout=AGENT_STEP_TIMEOUT)
        meta["elapsed"] = time.time() - t0
        meta["steps"] = 1
        kind, content = parse_step(raw)
        if kind == "say" and content:
            return content, meta
        print(f"[AGENT-PROACTIVE] 选择安静: {raw[:40]!r}", flush=True)
        meta["skipped"] = True
        return "", meta
    except Exception as e:
        meta["elapsed"] = time.time() - t0
        meta["skipped"] = True
        print(f"[AGENT-PROACTIVE] 异常 {type(e).__name__}: {e}，保持安静", flush=True)
        return "", meta


async def agent_reply(
    user_name: str,
    user_text: str,
    ctx: dict,
    history: Optional[List[dict]] = None,
    extra_system: str = "",
    images: Optional[List[str]] = None,
) -> Tuple[str, dict]:
    """跑一次 agent 循环，返回 (回复文本, 元信息)。

    ctx 里可带：member_card / summary / recent / self_status / notes_store（供工具读取）
    images 是图片 URL/data-uri 列表——给了她就"看得见"（Qwen 支持视觉）。
    history 是最近的对话（user/assistant 交替），会附在 system 之后。
    extra_system 是要追加到**主 system 消息末尾**的额外内容（如"带话"、"小本子"）。
    注意：网关只允许一条 system 且在开头，所以额外内容必须合并进主 system，
    不能作为独立的 system 消息插在 history 里（否则 400，实锤 2026-09-24）。
    任何异常 → 降级为单次直接生成，绝不返回空导致群友收不到话。
    """
    meta = {"steps": 0, "tools": [], "fallback": False, "elapsed": 0.0}
    t0 = time.time()

    sys_content = SYSTEM_PROMPT_AGENT
    if extra_system:
        sys_content = sys_content + "\n\n" + extra_system
    # 风格锚（SillyTavern 官方文档的核心技巧：模型会从"第一条消息"学风格和长度，
    # 比任何规则都管用）。历史里可能残留旧的"（托腮）""（舔唇）"式回复，会把她带偏，
    # 所以在历史之前先插一组短、随意的真实对话当样板。
    STYLE_ANCHOR = [
        {"role": "user", "content": "今晚打本吗"},
        {"role": "assistant", "content": "打啊 几点"},
        {"role": "user", "content": "八点老地方"},
        {"role": "assistant", "content": "行 别又迟到"},
        {"role": "user", "content": "你今天怎么不说话"},
        {"role": "assistant", "content": "没啥想说的"},
    ]

    messages: List[dict] = [{"role": "system", "content": sys_content}]
    messages += STYLE_ANCHOR
    if history:
        # 只保留 user/assistant（任何 system 都会触发网关 400）
        messages += [m for m in history[-8:] if m.get("role") in ("user", "assistant")]
    # 有图就发多模态消息（她能真的"看见"）；没图走纯文本
    if images:
        parts: List[dict] = [{"type": "text", "text": f"{user_name}：{user_text}"}]
        for u in images[:2]:                     # 一轮最多看两张，防撑爆 token
            parts.append({"type": "image_url", "image_url": {"url": u}})
        parts.append({"type": "text", "text": "（上面是他发的图，你看得见）"})
        messages.append({"role": "user", "content": parts})
    else:
        messages.append({"role": "user", "content": f"{user_name}：{user_text}"})

    tool_results: List[str] = []

    try:
        for step in range(AGENT_MAX_STEPS):
            meta["steps"] = step + 1
            if time.time() - t0 > AGENT_TOTAL_TIMEOUT:
                print(f"[AGENT] 总超时 {AGENT_TOTAL_TIMEOUT}s，中止循环", flush=True)
                break

            # 已调过工具：把结果作为新消息喂回去
            if step > 0:
                messages = messages[:1] + messages[1:]  # 保持 system 在最前
            # 开思考：正式回复需要巧思（这是用户明确要的"智商"），代价是 8~12 秒
            raw, usage = await asyncio.wait_for(
                agent_llm(messages, max_tokens=AGENT_REPLY_TOKENS, think=True),
                timeout=AGENT_STEP_TIMEOUT
            )
            kind, content = parse_step(raw)
            print(f"[AGENT] step={step+1} kind={kind} raw={raw[:60]!r}", flush=True)

            if kind == "tool":
                fn = TOOLS.get(content)
                if fn is None:
                    # 工具名不对：告诉它有哪些，继续下一轮
                    tool_results.append(f"（没有叫 {content} 的工具）")
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user",
                                     "content": f"【系统】没有这个工具。可用：{', '.join(TOOLS)}"})
                    continue
                # 带参数的工具：按 TOOLS_WITH_ARG 判断，参数从同一行里取。
                # 模型写法五花八门，都要能取到：
                #   TOOL: send_sticker: S1.png   TOOL: send_sticker S1.png
                #   send_sticker: S1.png         send_sticker S1.png
                #   sticker_note: S1.png|适合搞笑场合
                tool_arg = ""
                if content in TOOLS_WITH_ARG:
                    # 先找冒号形式（参数里可能含空格/竖线）
                    mm = re.search(rf"{content}\s*[:：]\s*(.+)$", raw, re.I | re.S)
                    if mm:
                        arg_lines = mm.group(1).strip().splitlines()
                        tool_arg = arg_lines[0].strip() if arg_lines else ""
                    else:
                        # 退而求其次：工具名后面直接跟参数（空格分隔）
                        mm2 = re.search(rf"{content}\s+(\S+.*)$", raw, re.I | re.S)
                        if mm2:
                            arg_lines = mm2.group(1).strip().splitlines()
                            tool_arg = arg_lines[0].strip() if arg_lines else ""
                    tool_arg = tool_arg.strip('"\'“”「」')
                if content in TOOLS_WITH_ARG:
                    result = fn(ctx.get("group_id", ""), ctx.get("uid", ""), ctx, tool_arg)
                else:
                    result = fn(ctx.get("group_id", ""), ctx.get("uid", ""), ctx)
                meta["tools"].append(content)
                print(f"[AGENT]   {content} -> {result[:80]!r}", flush=True)
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (f"【{content} 的结果】{result}\n\n"
                                f"现在接着回应对方（{user_text}）。"
                                f"如果你查到的信息跟对方问的有关，回复里用上。"
                                f"只输出一行：SAY: 你要说的话")
                })
                tool_results.append(result)
                continue

            if kind == "say" and content:
                meta["elapsed"] = time.time() - t0
                return content, meta

            # 兜底：模型把工具指令写进了 SAY（如 "SAY: set_name 扫货"）
            # 这种绝不能当话发出去——那是内部指令，发出去群里会看到乱码
            _leaked = re.match(r"^(set_name|remember|get_member|get_context|get_status|list_stickers|send_sticker|sticker_note)\b\s*[:：]?\s*(.*)$",
                               content, re.I | re.S)
            if _leaked and kind == "say":
                leaked_tool = _leaked.group(1).lower()
                leaked_arg = (_leaked.group(2) or "").strip()[:40]
                fn2 = TOOLS.get(leaked_tool)
                if fn2 is not None:
                    print(f"[AGENT] 指令误入 SAY，改按工具执行: {leaked_tool} {leaked_arg!r}", flush=True)
                    if leaked_tool in TOOLS_WITH_ARG:
                        res2 = fn2(ctx.get("group_id", ""), ctx.get("uid", ""), ctx, leaked_arg)
                    else:
                        res2 = fn2(ctx.get("group_id", ""), ctx.get("uid", ""), ctx)
                    meta["tools"].append(leaked_tool)
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user",
                                     "content": f"【{leaked_tool} 的结果】{res2}\n\n"
                                                f"现在用你自己的语气回应对方（{user_text}）。"
                                                f"只输出一行：SAY: 你要说的话"})
                    tool_results.append(res2)
                    continue

            # unknown：给一次纠正机会，下一轮仍失败则降级
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": "【系统】格式不对。只输出一行：TOOL: 工具名 或 SAY: 要说的话"})

        # 循环用完（超步数/超时/一直不产出）→ 用已收集的信息直接生成
        meta["fallback"] = True
        print(f"[AGENT] 循环未产出回复（steps={meta['steps']}），降级为直接生成", flush=True)

    except Exception as e:
        meta["fallback"] = True
        print(f"[AGENT] 循环异常 {type(e).__name__}: {e}，降级为直接生成", flush=True)

    # 降级：一次直接调用（去掉协议，只要人设 + 上下文）
    try:
        fb_sys = ROLE_CARD + (("\n\n" + extra_system) if extra_system else "")
        fallback_msgs = [{"role": "system", "content": fb_sys}]
        if history:
            # 网关只允许一条 system，历史里的 system 一律过滤（实锤：否则 400）
            fallback_msgs += [m for m in history[-8:] if m.get("role") in ("user", "assistant")]
        hint = ""
        if tool_results:
            hint = "\n【已知信息】" + " | ".join(tool_results)[:400]
        fallback_msgs.append({"role": "user",
                              "content": f"{user_name}：{user_text}{hint}\n用你自己的语气回应他。"})
        raw, _ = await asyncio.wait_for(agent_llm(fallback_msgs, max_tokens=AGENT_REPLY_TOKENS, think=True),
                                        timeout=AGENT_STEP_TIMEOUT)
        kind, content = parse_step(raw)
        text = content if content else raw
        meta["elapsed"] = time.time() - t0
        return text.strip(), meta
    except Exception as e:
        meta["elapsed"] = time.time() - t0
        print(f"[AGENT] 降级也失败 {type(e).__name__}: {e}", flush=True)
        return "", meta
