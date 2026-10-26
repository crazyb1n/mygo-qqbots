"""若叶睦 agent 内核：主AI编排 + 子代理派遣。

架构见《若叶睦-agent架构-逻辑文档.md》。与扫货旧 agent-v2（TOOL:/SAY: 工具循环）不同，
这是「编排式」agent，对应用户的方法论：
- 主AI = 若叶睦本人的角色思考：理解需求 → 查经验 → 出方案 → （需要时）向群友确认
  → 派子代理 → 验收 → 交付 → 沉淀经验。全程用她的口吻，不 OOC。
- 子代理 = 按提示词执行的专项角色：solver（解题）/ writer（写东西）/ brainstorm（出主意）
  / researcher（查资料：LLM 出关键词 → 360 搜索 → 基于结果整理，2026-10-09 新增）。
- 路由：should_try_task 只挡明显不是任务的（太短/纯情绪），是不是任务由 understand() 裁决——
  关键词表不再当闸门（旧版教训：入口是稀有条件交集 → 功能上线起就不可达，命中 0）。

吸收自扫货 agent-v2 的教训：
- 所有 LLM 调用必以 user 消息收尾（旧版"提示词回显"bug 的根因是 assistant 结尾续写）；
- 只用一条 system（某些网关多 system 会 400），群上下文/经验合并进 user；
- 任何一步失败由 chat.py 降级为快路直答，群友永远收得到话。
- 安全边界：子代理 = 纯 LLM + 只读网页搜索，绝无文件/命令/写盘能力（群消息可触发，
  群里任何人说话都能进编排——放开机器能力等于把电脑交给整个群，这条线不动）。
"""
import asyncio
import json
import os
_REASONING_EFFORT = os.getenv("DEEPSEEK_REASONING_EFFORT", "none").strip()
import re
import time
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from . import experience

try:
    from . import websearch
except ImportError:                      # 脱离包结构跑脚本时
    websearch = None


def _cfg(key: str, default: str) -> str:
    return os.getenv(key, default)


DEEPSEEK_API_KEY = _cfg("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = _cfg("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _cfg("DEEPSEEK_MODEL", "deepseek-chat")
DISABLE_REASONING = _cfg("DISABLE_REASONING", "1") == "1"
SEARCH_ENABLED = _cfg("SEARCH_ENABLED", "1") == "1"
UNDERSTAND_TIMEOUT = float(_cfg("AGENT_UNDERSTAND_TIMEOUT", "45"))
SUBAGENT_TIMEOUT = float(_cfg("AGENT_SUBAGENT_TIMEOUT", "180"))
DISTILL_TIMEOUT = float(_cfg("AGENT_DISTILL_TIMEOUT", "45"))

from .llm_fallback import build_client
client = build_client(AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL))

# ============ 编排入口判定（2026-10-09 放宽） ============

# 明确排除：纯情绪/闲聊短语，防止"帮帮我（玩梗）"之类误入编排
TASK_EXCLUDE = ["帮帮我（", "救命", "啊啊啊"]


def should_try_task(raw: str) -> bool:
    """@ 场景下的编排入口判定：只要不像纯闲聊就交给主AI理解去判断（零成本规则）。
    教训（旧版实锤）：入口是两个稀有条件的交集 → 功能从上线起就不可达（命中 0）。
    所以关键词表不再当闸门——它是"省 token 的优化"，不是可达性开关。
    这里只挡明显不是任务的：太短、纯应答/情绪、纯表情；
    是不是任务由 understand() 的 agent=none 裁决，闲聊照旧走快路。"""
    raw = (raw or "").strip()
    if len(raw) < 4:
        return False
    if any(k in raw for k in TASK_EXCLUDE):
        return False
    # 纯应答/情绪/语气词（"哈哈哈哈哈""？？？""嗯嗯嗯"）：不值得花一次理解调用
    if re.fullmatch(r"[哈啊哦嗯呃额唔咦诶呀哇噢嗷唉诶嘿?？!！。…、，~～\s]+", raw):
        return False
    return True


# ============ 确认 / 取消（零 token，正则） ============

_CONFIRM_RE = re.compile(
    r"^(好|好呀|好吧|好的|好咧|嗯|嗯嗯|嗯呐|可以|可以呀|行|行吧|ok|okay|okay|来吧|来|"
    r"就这样|就这个|没问题|同意|做吧|去吧|去呗|执行|安排|上|冲|干吧)[的啦呀哈呢。！!~～\s]*$",
    re.I,
)
_CANCEL_RE = re.compile(r"^(算了|不要了|不用了|不用|取消|不弄了|别弄了|不搞了|算了算了|先不)[。！!~～\s]*$")


def is_confirm(raw: str) -> bool:
    return bool(_CONFIRM_RE.match((raw or "").strip()))


def is_cancel(raw: str) -> bool:
    return bool(_CANCEL_RE.match((raw or "").strip()))


# ============ 主AI：理解（LLM#1，关思考，快） ============

ORCH_SYSTEM = (
    "你是「若叶睦」的内心整理过程（Ave Mujica吉他手，话极少，认真）。群里有人向你提了请求，"
    "你要先在心里想清楚：这是什么需求、该派哪个专项的自己去做、分几步、要不要先跟对方确认。\n"
    "【输出】只输出一行 JSON，不要换行、不要解释、不要多余内容：\n"
    '{"need":"需求完整一句话","brief":"若叶睦口吻的需求复述，不超过18字，不用句号",'
    '"agent":"solver|writer|brainstorm|researcher|none","steps":["第一步","第二步"],'
    '"need_confirm":false}\n'
    "【最先判断：对话还是任务（最重要）】\n"
    "- 对话（问候/玩梗/闲聊/问你问题/问你自己或你认识的人——Mortis/墨提斯、祥子、素世、灯、"
    "CRYCHIC、Ave Mujica、黄瓜这些你本来就认识或就是本人）→ agent=none：你直接按人设聊，"
    "严禁查资料、严禁写文档。\n"
    "- 任务（要产出新东西）：写东西 / 出主意 / 解题 / 查**外部**资料 → 才派对应的自己去做。\n"
    "- 主观题（让你评价/谈看法/说喜好/比较或推荐：好不好、怎么样、好玩吗、值不值、"
    "该不该、你喜欢…吗、哪个好、谁更强）→ agent=none：你有自己的判断和脾气，直接按人设"
    "说自己的想法；严禁查资料、严禁列条目、严禁写成资料汇编。\n"
    "【判断规则】\n"
    "- 算数/数学/逻辑/解题/推导 → agent=solver，need_confirm=false（直接做）\n"
    "- 写东西（文案/祝福语/小作文/公告/介绍/段子/请假条）→ agent=writer；"
    "成品会比较长（50字以上）或涉及个人口味 → need_confirm=true；顺手的小句子 → false\n"
    "- 出主意/起名/做选择/给建议/帮忙想 → agent=brainstorm，need_confirm=false\n"
    "- 查外部资料（**明确让你去查**：帮我查/搜一下/找找/最新消息或进展/参数/价格/是什么/有哪些/什么时候）"
    "→ agent=researcher；征求意见与看法（怎么样/好不好/评价/推荐哪个/值不值）**不算**查资料，"
    "按上面的主观题规则走 agent=none；问的若是你自己/你认识的人（Mortis、祥子…）→ agent=none（你自己就是答案）\n"
    "- 只是聊天/玩梗/问候/情绪表达/问问题闲聊 → agent=none（不编排，直接聊天）\n"
    "- 拿不准 → agent=none\n"
    "steps 写 1~3 步、每步不超过12字；简单请求就一步。"
)


async def understand(raw: str, user_name: str, ctx_lines: List[str] = None,
                     exp_hint: str = "") -> Optional[dict]:
    """主AI理解：返回 {"need","brief","agent","steps","need_confirm"}；失败返回 None（当闲聊）。"""
    user_content = f"{user_name} 说：{raw}"
    if ctx_lines:
        user_content += "\n\n【群里最近的消息】\n" + "\n".join(ctx_lines[-6:])
    if exp_hint:
        user_content += "\n\n" + exp_hint
    user_content += "\n\n（现在只输出那一行 JSON。）"
    messages = [
        {"role": "system", "content": ORCH_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        kwargs = {"model": DEEPSEEK_MODEL, "messages": messages, "max_tokens": 2048,
                  "temperature": 0.3}
        if DISABLE_REASONING:
            # 2026-10-10：路由判定是分类任务，实测 high 要 8.2s、off 只要 2.3s；
            # 一律关思考（正式回复的质量档位在 chat.py 里独立控制，不受影响）
            kwargs["reasoning_effort"] = "off"
        r = await asyncio.wait_for(client.chat.completions.create(**kwargs),
                                   timeout=UNDERSTAND_TIMEOUT)
        text = (r.choices[0].message.content or "").strip()
        usage = r.usage
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            print(f"[ORCH] 理解输出无 JSON: {text[:60]!r}", flush=True)
            return None
        d = json.loads(m.group(0))
        agent = d.get("agent", "none")
        if agent not in ("solver", "writer", "brainstorm", "researcher", "none"):
            agent = "none"
        plan = {
            "need": str(d.get("need", ""))[:120],
            "brief": str(d.get("brief", ""))[:24],
            "agent": agent,
            "steps": [str(s)[:14] for s in (d.get("steps") or []) if s][:3],
            "need_confirm": bool(d.get("need_confirm")),
            "usage": usage,
        }
        print(f"[ORCH] 理解完成: agent={plan['agent']} brief={plan['brief']!r} "
              f"steps={plan['steps']} confirm={plan['need_confirm']}", flush=True)
        return plan
    except Exception as e:
        print(f"[ORCH] 理解失败 {type(e).__name__}: {e}", flush=True)
        return None


# ============ 方案确认消息（模板拼装，不调 LLM，若叶睦口吻） ============

_CN_NUM = "一二三四五"


def render_plan(plan: dict) -> str:
    brief = (plan.get("brief") or plan.get("need") or "知道了").strip("。")
    steps = [s for s in plan.get("steps", []) if s] or ["把它做出来"]
    lines = "\n".join(f"{_CN_NUM[i]}、{s}" for i, s in enumerate(steps[:3]))
    return f"……嗯。{brief}。我想这么弄：\n{lines}\n……可以的话，回我一声。"


DELIVER_LEADS = [
    "……写完了。是这个。",
    "……嗯。做完了，看看。",
    "……好了。给你。",
    "……弄好了。这样行不行。",
]
CANCEL_ACKS = ["……嗯。那算了。", "……好吧。（收起来）", "……嗯。"]


# ============ 子代理：按提示词执行 ============

SUBAGENT_PROMPTS = {
    "solver": (
        "你是冷静理性的解题助手，直接给答案，不要输出思考过程或'我来想想'之类的话。"
        "要求：1. 用简单清晰的文字分2~4步写解题过程（每步一行，用① ② ③标注）；"
        "2. 最后一行给最终答案；"
        "3. 禁止LaTeX、markdown、星号、下划线、箭头等花哨符号——分数写'3/4'，平方写'x的平方'，根号写'根号下'，全部用汉字和普通标点；"
        "4. 语气平淡简洁，不要客套、不要AI味、不要废话。"
    ),
    "writer": (
        "你替群友写东西。要求：\n"
        "1. 按需求写，成品完整、可以直接发到群里；\n"
        "2. 口吻自然，像群里活人写的：不要'在此''值得'之类书面腔，不要堆 emoji，不要落款'由AI生成'；\n"
        "3. 长度跟需求走，没说就 80~180 字；\n"
        "4. 只输出成品正文，不要解释、不要前后缀。"
    ),
    "brainstorm": (
        "你帮群友出主意。要求：\n"
        "1. 直接给最多 3 个具体、可执行的点子，每个一两句话，不用开场白；\n"
        "2. 点子要具体（有场景/有细节），不要'多沟通''放平心态'这类空话；\n"
        "3. 只输出点子本身，编号 1. 2. 3."
    ),
}
# solver 留着思考（解题质量优先，v1 就这样）；writer/brainstorm 关思考求快
SUBAGENT_SETTINGS = {
    "solver": {"max_tokens": 20480, "temperature": 0.3, "thinking": True},
    "writer": {"max_tokens": 2048, "temperature": 0.7, "thinking": False},
    "brainstorm": {"max_tokens": 2048, "temperature": 0.8, "thinking": False},
    "researcher": {"max_tokens": 2048, "temperature": 0.3, "thinking": False},
}

# ============ 子代理·查资料（2026-10-09 新增：真的去搜，再整理） ============

RESEARCH_QUERY_SYSTEM = (
    "你在帮若叶睦准备搜索。根据群友的任务，给出 1~2 条搜索关键词（中文，每条 4~16 字，"
    "尽量具体、能搜到事实信息；不要用问句，不要引号）。\n"
    "只输出一行 JSON：{\"queries\":[\"关键词一\",\"关键词二\"]}"
)

RESEARCH_ANSWER_SYSTEM = (
    "你在帮若叶睦查资料回答群友。下面会给你【搜索结果】（每条是 标题＋摘要）。要求：\n"
    "1. 把结果整理成条理清楚的中文回答：3~6 条要点，每条一两句话，直接说事；\n"
    "2. 只写搜索结果里有依据的内容；结果里没有的数字/事实一个字都不要编；"
    "资料不足就直说「没查到 XX」；\n"
    "3. 可以写「据报道」「有消息称」这类来源感的话，但不要写网址、不要提'搜索结果'这个词；\n"
    "4. 400 字以内，不要客套、不要AI味。"
)

# 主观问题（看法/评价/比较/推荐）专用（2026-10-16 用户要求："涉及主观问题的询问，
# 若叶睦要用符合人设的语气总结回复，不是单纯的检索问题"）。
# 资料照查照信，但成品必须是"她本人在说话"的短总结，不是 3~6 条的资料汇编。
RESEARCH_SUMMARY_SYSTEM = (
    "你在扮演若叶睦（话极少、直白、不热情）回答群友，这是个主观问题（问看法/评价/比较/推荐）。\n"
    "下面会给你【搜索结果】（标题＋摘要）。要求：\n"
    "1. 用若叶睦自己的语气总结：先一句结论/态度（你的判断），最多再补一句依据——"
    "像她本人开口说话（短句、直接、不客套、不用敬语），不要分条目、不要小标题、"
    "不要写「据报道」「综合来看」这类资料腔，也不要写网址；\n"
    "2. 结论必须有资料支撑；资料里没有数字/事实一个字都不要编；"
    "资料不足就说「没查到」或「说不好」，也可以只给一句你的感觉；\n"
    "3. 总长 80 字以内（可以分 2~3 行短句），多一个字都不写。"
)


class _Usage:
    """把 researcher 的多次调用用量合并成一个 _record_usage 能吃的对象。"""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, usage) -> None:
        if not usage:
            return
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0


def _chat_kwargs(messages: List[dict], max_tokens: int, temperature: float,
                 thinking: bool) -> dict:
    kwargs = {"model": DEEPSEEK_MODEL, "messages": messages,
              "max_tokens": max_tokens, "temperature": temperature}
    if not thinking and DISABLE_REASONING:
        # 2026-10-10：thinking=False 的步骤（路由/关键词/整理/沉淀）一律关思考提速；
        # 正式回复走 chat.py 自己的档位（.env 的 DEEPSEEK_REASONING_EFFORT）
        kwargs["reasoning_effort"] = "off"
    return kwargs


async def _call_llm(kwargs: dict, timeout: float, tag: str):
    """一次 LLM 调用；正文为空（思考吃光 token 预算，v4-flash 实锤）则重试一次。
    成功返回 (text, usage)，两次都空返回 ("", usage累计)。"""
    last_usage = None
    for attempt in (1, 2):
        try:
            r = await asyncio.wait_for(client.chat.completions.create(**kwargs), timeout=timeout)
            last_usage = r.usage
            text = (r.choices[0].message.content or "").strip()
            if text:
                return text, last_usage
            print(f"[{tag}] 第{attempt}次正文为空（finish={r.choices[0].finish_reason}），重试", flush=True)
        except Exception as e:
            print(f"[{tag}] 第{attempt}次调用失败 {type(e).__name__}: {e}", flush=True)
    return "", last_usage


async def _run_researcher(need: str, raw: str, ctx_lines: List[str] = None,
                          subjective: bool = False) -> Tuple[str, object]:
    """查资料子代理：LLM 出关键词 → 360 搜索（最多 2 条）→ LLM 基于结果整理成回答。
    搜不到就如实说没查到，绝不编造（与全体子代理同一条铁律）。
    subjective=True（主观问题）：整理那步换成"若叶睦语气短总结"提示词（见 RESEARCH_SUMMARY_SYSTEM），
    交付的不是资料汇编而是她自己的话（2026-10-16 用户要求）。"""
    usage = _Usage()
    topic = (need or raw or "").strip()
    if ctx_lines:
        topic += "\n【群里最近的消息（供参考）】\n" + "\n".join(ctx_lines[-4:])

    queries: List[str] = []
    if SEARCH_ENABLED and websearch is not None:
        try:
            text_kw, u_kw = await _call_llm(_chat_kwargs(
                [{"role": "system", "content": RESEARCH_QUERY_SYSTEM},
                 {"role": "user", "content": f"任务：{topic}\n\n（现在只输出那行 JSON。）"}],
                max_tokens=2048, temperature=0.3, thinking=False), UNDERSTAND_TIMEOUT, "RESEARCH-KW")
            usage.add(u_kw)
            m = re.search(r"\{.*\}", text_kw, re.S)
            if m:
                queries = [str(q).strip()[:24] for q in (json.loads(m.group(0)).get("queries") or []) if str(q).strip()]
        except Exception as e:
            print(f"[RESEARCH] 关键词生成失败，退回原句搜索: {type(e).__name__}: {e}", flush=True)
        if not queries:
            queries = [(need or raw or "")[:24]]

    blocks: List[str] = []
    for q in queries[:2]:
        try:
            b = await asyncio.to_thread(websearch.search, q)
        except Exception as e:
            print(f"[RESEARCH] 搜索异常 {q!r}: {type(e).__name__}: {e}", flush=True)
            b = None
        if b:
            blocks.append(b)
        print(f"[RESEARCH] 搜索 {q!r} -> {'有结果' if b else '无结果'}", flush=True)

    if blocks:
        material = "\n\n".join(blocks)
        tail = "（根据上面的搜索结果整理回答。）" if not subjective else \
            "（根据上面的搜索结果，用若叶睦的语气总结回答：先结论后依据，短。）"
    else:
        material = "（没有搜到任何资料。）"
        tail = ("（没搜到资料。只回答你确定的常识内容；不确定的明确说没查到，绝不编造。）"
                if not subjective else
                "（没搜到资料。主观问题可以只给一句你自己的态度/感觉；具体事实没把握就说没查到，绝不编造。）")
    try:
        sys_prompt = RESEARCH_SUMMARY_SYSTEM if subjective else RESEARCH_ANSWER_SYSTEM
        text, u2 = await _call_llm(_chat_kwargs(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": f"任务：{topic}\n\n{material}\n\n{tail}"}],
            max_tokens=1024 if subjective else 8192,
            temperature=0.6 if subjective else 0.3, thinking=False), SUBAGENT_TIMEOUT, "RESEARCH")
        usage.add(u2)
        print(f"[RESEARCH] 完成: {text[:60]!r}", flush=True)
        return text, usage
    except Exception as e:
        print(f"[RESEARCH] 整理失败 {type(e).__name__}: {e}", flush=True)
        return "", None


async def run_subagent(agent: str, need: str, raw: str,
                       ctx_lines: List[str] = None, subjective: bool = False) -> Tuple[str, object]:
    """派子代理执行。返回 (成果文本, usage)。失败返回 ("", None)。
    subjective 只对 researcher 有意义（主观问题 → 用人设语气短总结，见 _run_researcher）。"""
    cfg = SUBAGENT_SETTINGS.get(agent)
    if not cfg:
        return "", None
    if agent == "researcher":
        return await _run_researcher(need, raw, ctx_lines, subjective=subjective)
    user_content = f"需求：{need or raw}\n对方原话：{raw}"
    if ctx_lines:
        user_content += "\n\n【群里最近的消息（供参考）】\n" + "\n".join(ctx_lines[-4:])
    user_content += "\n\n（现在直接给出成果。）"
    messages = [
        {"role": "system", "content": SUBAGENT_PROMPTS[agent]},
        {"role": "user", "content": user_content},
    ]
    try:
        text, usage = await _call_llm(_chat_kwargs(
            messages, max_tokens=cfg["max_tokens"], temperature=cfg["temperature"],
            thinking=cfg["thinking"]), SUBAGENT_TIMEOUT, f"SUBAGENT:{agent}")
        if text:
            print(f"[SUBAGENT] {agent} 完成: {text[:60]!r}", flush=True)
            return text, usage
        print(f"[SUBAGENT] {agent} 两次都空，放弃", flush=True)
        return "", None
    except Exception as e:
        print(f"[SUBAGENT] {agent} 失败 {type(e).__name__}: {e}", flush=True)
        return "", None


# ============ 经验沉淀（任务后一次廉价调用） ============

DISTILL_SYSTEM = (
    "你是若叶睦的记忆整理。每次帮群友做完一件事，想想有没有值得长期记住的一条。"
    "只输出一行 JSON：{\"layer\":\"L2\",\"text\":\"≤60字\"} 或 {\"layer\":\"L3\",\"text\":\"≤60字\"}。"
    "layer=L2 表示关于这个群里的人/梗/约定/喜好的情报；layer=L3 表示关于这类任务怎么做的经验。"
    "没有值得记的就只输出 {\"layer\":\"none\"}。"
)


async def distill(need: str, result: str) -> Optional[str]:
    """任务后沉淀经验。返回写入的层名（L2/L3）或 None。写入由 experience.learn 完成。"""
    if not need or not result:
        return None
    user_content = f"这次的任务：{need}\n交付的内容开头：{result[:200]}"
    messages = [
        {"role": "system", "content": DISTILL_SYSTEM},
        {"role": "user", "content": user_content + "\n\n（现在只输出那行 JSON。）"},
    ]
    try:
        kwargs = {"model": DEEPSEEK_MODEL, "messages": messages, "max_tokens": 2048,
                  "temperature": 0.3}
        if DISABLE_REASONING:
            kwargs["reasoning_effort"] = "off"   # 沉淀是低成本整理，关思考（2026-10-10）
        r = await asyncio.wait_for(client.chat.completions.create(**kwargs),
                                   timeout=DISTILL_TIMEOUT)
        text = (r.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        d = json.loads(m.group(0))
        layer = d.get("layer")
        body = str(d.get("text", "")).strip()
        if layer in ("L2", "L3") and body:
            tags = [w for w in re.findall(r"[\u4e00-\u9fff]{2,4}", body)][:4]
            if experience.learn(layer, body, tags=tags):
                return layer
        return None
    except Exception as e:
        print(f"[ORCH] 沉淀失败 {type(e).__name__}: {e}", flush=True)
        return None
