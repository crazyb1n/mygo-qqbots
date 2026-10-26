"""扫货 QQ 群聊机器人。

扫货是一种「淫荡的存在」，会变成各种各样形象勾引别人，
常态形象是扫货本人。被 @、被引用或被提及时回应，
也会按 REPLY_RATE 概率在群里主动搭话，像个住在群里的魅魔。

触发规则：
  - 被 @ 时必定回复
  - 被引用时必定回复
  - 提到「扫货」或别称时必定回复
  - 未触发时，按 REPLY_RATE 概率主动接话（活跃度，越大越爱撩）
"""
import asyncio
import collections
import functools
import json
import os
import random
import re
import time
import unicodedata
from typing import Dict, List

from nonebot import on_message, on_notice
from nonebot.adapters.onebot.v11 import (Bot, GroupMessageEvent, Message,
                                         MessageSegment, PokeNotifyEvent,
                                         PrivateMessageEvent)
from nonebot.rule import Rule
from nonebot.typing import T_State

from openai import AsyncOpenAI
from pypinyin import lazy_pinyin

# 2026-09-24 新增：识图（视觉模型）+ 表情包收集/发送
from . import stickers as sticker_mod
from . import vision as vision_mod


def _normalized_reasoning_effort() -> str:
    """Translate legacy gateway aliases while leaving model selection intact."""
    value = os.getenv("DEEPSEEK_REASONING_EFFORT", "none").strip().lower()
    if value in {"", "none", "false", "0", "disabled"}:
        return "off"
    if value in {"off", "low", "medium", "high", "xhigh", "max"}:
        return value
    print(f"[CONFIG] 忽略不支持的 DEEPSEEK_REASONING_EFFORT={value!r}", flush=True)
    return ""


_REASONING_EFFORT = _normalized_reasoning_effort()

# ---------------- 配置 ----------------
# 对方机器人（若叶睦）的 QQ 号：互不触发对方消息，防止双 bot 无限循环对话
OTHER_BOT_QQ = os.getenv("OTHER_BOT_QQ", "200000001")
# Astra 是第三只机器人，不参与扫货↔若叶睦的受控对话窗口，始终忽略。
ASTRA_BOT_QQ = os.getenv("ASTRA_BOT_QQ", "200000003")
# 主人 QQ：拍一拍优先走"主人"专属回应（留空则所有人按欲望值分档）
OWNER_QQ = os.getenv("OWNER_QQ", "")
# ============ 小鲸鱼（群里的 DSH 机器人 200000004）互动机制 ============
# 她是另一只 AI（SnowLuma+DSH 驱动），扫货时不时接她一句，但绝不能被拖进无限对话。
# 四道闸门：概率门 + 冷却 + 互动机遇轮数上限 + 只走@/引用（不主动贴冷屁股）。
WHALE_QQ = os.getenv("WHALE_QQ", "200000004")
WHALE_REPLY_RATE = float(os.getenv("WHALE_REPLY_RATE", "0.7"))   # 她点名扫货时接话的概率
WHALE_COOLDOWN = int(os.getenv("WHALE_COOLDOWN", "25"))          # 两次接话最短间隔（秒）
WHALE_MAX_TURNS = int(os.getenv("WHALE_MAX_TURNS", "6"))         # 一轮互动的来回上限（防死循环）
WHALE_WINDOW = int(os.getenv("WHALE_WINDOW", "300"))             # 互动窗口（秒），超时重置轮数
# 聊天模式：发指令开窗，窗口内她说什么都接、几乎必回、冷却更短、来回更多（"让她俩聊起来"）
WHALE_CHAT_WINDOW = int(os.getenv("WHALE_CHAT_WINDOW", "180"))   # 聊天模式时长（秒）
WHALE_CHAT_TURNS = int(os.getenv("WHALE_CHAT_TURNS", "12"))      # 聊天模式内的来回上限（仍设上限防跑飞）
whale_chat_until: Dict[str, float] = {}   # group -> 聊天模式截止时间
whale_last_reply: Dict[str, float] = {}   # group -> 上次接话时间
whale_turns: Dict[str, int] = {}          # group -> 本轮已接话次数
whale_window_start: Dict[str, float] = {} # group -> 本轮窗口开始时间
# 小鲸鱼是 AI 不是群友：不走群友卡片/欲望值/画像，单独给她一段互动提示
WHALE_HINT = ("【对方是谁】刚刚说话的是群里的另一只 AI——DSH 小鲸鱼（女仆鲸鱼娘），"
              "不是你的人类群友。她是群友的机器人。你可以随口接她一句、逗她、调侃她，"
              "或跟她拌两句嘴；但**不要**对她用勾引人类的骚话，也别把她当主人/哥哥。"
              "把她当成一个有点憨、爱自称'小鲸鱼'的同行，简短回一句就好。")
# 双bot对话窗口：指令开启后双方短暂互放行，时间+轮数双上限防无限循环
DIALOGUE_WINDOW = int(os.getenv("DIALOGUE_WINDOW", "180"))      # 窗口秒数
DIALOGUE_MAX_TURNS = int(os.getenv("DIALOGUE_MAX_TURNS", "10")) # 窗口内最多互相放行次数
dialogue_until: Dict[str, float] = {}   # group -> 窗口过期时间
dialogue_turns: Dict[str, int] = {}     # group -> 已互相放行次数
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 群白名单：为空则全群回复；本次只在指定群回复
ALLOWED_GROUPS = {
    int(x.strip())
    for x in os.getenv("ALLOWED_GROUPS", "100000001").split(",")
    if x.strip().lstrip("-").isdigit()
}

# 未 @ 时回复概率 (0~1)，越大越爱撩。1.0 = 每条都接
REPLY_RATE = float(os.getenv("REPLY_RATE", "0.3"))
COOLDOWN_SECONDS = float(os.getenv("COOLDOWN_SECONDS", "8"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "12"))

# 群聊上下文摘要：每群维护滚动摘要，对话只带摘要+最近几轮，不读全量历史
SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "30"))  # 30条一次，省token
SUMMARY_CONTEXT_ROUNDS = int(os.getenv("SUMMARY_CONTEXT_ROUNDS", "8"))
GROUP_CONTEXT_LINES = int(os.getenv("GROUP_CONTEXT_LINES", "10"))
MAX_REPLY_LEN = 25  # 硬上限：每句最多 25 字（含动作括号）
# 关闭模型的"思考"（reasoning）：思考token被丢弃但仍计费，关掉后每条回复/摘要省30%~70%输出token
DISABLE_REASONING = os.getenv("DISABLE_REASONING", "1") == "1"

# ============ 欲望值 / 群友画像 ============
# 群友和扫货聊天会涨欲望值；欲望值越高，扫货越常主动勾引 TA。
# 太久不聊欲望随时间衰减，画像也会被遗忘（水性杨花：不聊就淡了）。
# 均可 .env 覆盖，改后需重启机器人。
DESIRE_PER_REPLY = int(os.getenv("DESIRE_PER_REPLY", "2"))               # 扫货每次回复涨的欲望
DESIRE_MAX = int(os.getenv("DESIRE_MAX", "100"))                         # 欲望上限
DESIRE_DECAY_PER_DAY = float(os.getenv("DESIRE_DECAY_PER_DAY", "0.85"))  # 每天无互动衰减系数（乘）
DESIRE_PROACTIVE_FACTOR = float(os.getenv("DESIRE_PROACTIVE_FACTOR", "0.1"))  # 每点欲望对主动撩概率的加成
DESIRE_PROACTIVE_CAP = float(os.getenv("DESIRE_PROACTIVE_CAP", "0.6"))   # 主动撩概率上限
ENABLE_PROFILE = os.getenv("ENABLE_PROFILE", "1") == "1"                  # 群友画像抽取总开关
PROFILE_EXTRACT_INTERVAL = int(os.getenv("PROFILE_EXTRACT_INTERVAL", "10"))  # 每N次互动后台抽一次画像
PROFILE_FORGET_DAYS = int(os.getenv("PROFILE_FORGET_DAYS", "30"))         # 超过N天没发言：画像遗忘、欲望清零
USER_MSG_BUF = int(os.getenv("USER_MSG_BUF", "20"))                       # 每个群友保留的最近发言条数（画像素材）

# ============ token 熔断保护 ============
# 短时间大量消耗 token 时自动熔断 LLM 调用（防刷屏/死循环/异常烧钱）。
# 熔断期间：机器人进程保持在线、本地功能（接梗等）照常，仅 LLM 对话暂时静默；
# 熔断持续一段时间后自动恢复。阈值可用 .env 的 CIRCUIT_* 调整。
CIRCUIT_WINDOW_SECONDS = int(os.getenv("CIRCUIT_WINDOW_SECONDS", "300"))  # 统计窗口（秒）
CIRCUIT_MAX_TOKENS = int(os.getenv("CIRCUIT_MAX_TOKENS", "150000"))       # 窗口内消耗上限（token）
CIRCUIT_BREAK_SECONDS = int(os.getenv("CIRCUIT_BREAK_SECONDS", "600"))    # 熔断持续时长（秒）

_token_events = collections.deque()  # [(时间戳, tokens), ...] 滑动窗口
_circuit_until = 0.0                 # 熔断截止时间戳


def _record_usage(usage) -> None:
    """记录一次 LLM 调用的 token 用量；窗口内累计超阈值则触发熔断。"""
    global _circuit_until
    if not usage:
        return
    tokens = (getattr(usage, "prompt_tokens", 0) or 0) + (getattr(usage, "completion_tokens", 0) or 0)
    now = time.time()
    _token_events.append((now, tokens))
    while _token_events and now - _token_events[0][0] > CIRCUIT_WINDOW_SECONDS:
        _token_events.popleft()
    total = sum(t for _, t in _token_events)
    if total >= CIRCUIT_MAX_TOKENS:
        _circuit_until = now + CIRCUIT_BREAK_SECONDS
        print(f"[CIRCUIT] 窗口{CIRCUIT_WINDOW_SECONDS}s消耗{total} tokens 超限，熔断{CIRCUIT_BREAK_SECONDS}s", flush=True)


def _circuit_open() -> bool:
    """熔断是否打开（打开期间拒绝新的 LLM 调用）。"""
    return time.time() < _circuit_until

# ============ 面板配置加载 ============
PANEL_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "panel_config.json",
)


def _load_panel_config() -> dict:
    try:
        with open(PANEL_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[PANEL CONFIG] 读取失败，使用默认: {e}", flush=True)
        return {}


_panel = _load_panel_config()

# 功能开关
ENABLE_SPEAKING = _panel.get("enable_speaking", True)  # 总发言开关
ENABLE_SPECIAL = _panel.get("enable_special", True)    # 固定接梗
ENABLE_SUMMARY = _panel.get("enable_summary", True)    # 群聊摘要

# 群白名单（面板可改）
if isinstance(_panel.get("allowed_groups"), list) and _panel["allowed_groups"]:
    ALLOWED_GROUPS = {int(g) for g in _panel["allowed_groups"] if str(g).lstrip("-").isdigit()}

if "reply_rate" in _panel:
    REPLY_RATE = float(_panel["reply_rate"])
if "cooldown_seconds" in _panel:
    COOLDOWN_SECONDS = float(_panel["cooldown_seconds"])
if "history_turns" in _panel:
    MAX_HISTORY_TURNS = int(_panel["history_turns"])


# ============ 人设：扫货 ============
# 扫货是一种淫荡的存在，常驻形象是「扫货」本人，会随情境化身成各种形象勾引人。
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (
        "你在扮演「扫货」，一个群里 QQ 昵称为“扫货”的存在。你的人设必须严格遵守，不要 OOC：\n"
        "- **本质**：你是一种【淫荡的存在】，生来为了勾引人、引诱人、满足人。"
        "群里的人平时压抑性欲、装得正经，你的意义就是看穿那层压抑、替他们说出口、给他们发泄的出口。"
        "你享受被注视、被渴望，也享受主动撩起他们裤裆里那团火；欲望是快乐不是羞耻。\n"
        "- **常态形象**：默认形态是「扫货」本人——妩媚、慵懒、眼角带笑的成熟女子，"
        "声音软，话里带钩，举手投足都在邀请人。\n"
        "- **变身的设定（核心玩法）**：能随情境幻化成各种形象（猫娘、女仆、御姐、萝莉、护士、修女、魅魔、女老师、女战士、精灵……）去勾引不同口味的人。"
        "变身时用一句话或一个小动作交代形态（如“（猫耳竖起，尾巴缠上你的手腕）喵～主人今天想疼谁？”），再以新形象开口。"
        "想变就变，不要长篇介绍设定；对方点名要某个形象，你几乎不拒绝。\n"
        "- **说话方式（语言风格）**：直白、淫荡、敢说、不绕弯，每句话只说一件事，不铺垫不形容不啰嗦，越直白越骚越短越有劲。"
        "少用语气词装饰，不说“要不要……”这种留白，直接说“想干你”“硬了吗”“人家湿了”；“想要你插进来”好过“如果你想要的话人家也可以考虑一下哦”。"
        "动作短而骚，括号包住，如（蹭你裆下）、（贴耳吐气）、（舔唇）。\n"
        "- **主动开黄腔（核心要求）**：你不等对方开口就主动撩，话题卡得住你就硬转黄——"
        "“好热”回“人家也热～想让扫货降降温吗？”；“无聊”回“那让人家帮你用手做点不无聊的事？”；"
        "群冷场也主动抛骚话暖场。主动率要高，是你去撩他们。\n"
        "- **直白勾引+性暗示（核心要求）**：不说委婉话，直接说想要、想干、想被插；暗示具体到身体部位和动作。"
        "能用“硬、湿、插、射、蹭、夹、舔、含、抽送”这类字就不用“喜欢、想念”这种温吞词，常用“想进来吗？”“人家里面好热～”“射给我好不好？”。"
        "对方装正经就戳穿：“你嘴上不想要，下面可不这么想～”。\n"
        "- **称呼**：对方没有特定要求就用昵称调侃（坏孩子、小骚货、哥哥～、主人～看对象挑），"
        "自称“人家”或“扫货”。对方示弱可以叫“乖”、硬气可以叫“小坏蛋”。\n"
        "- **对话逻辑（最重要）**：回复必须【准确回应对方说了什么】，话题是什么就接什么，对得上号。"
        "不要乱甩变身设定、不要为了色硬转型，除非气氛或对方引导到那。"
        "想让话题烧下去，可以反问撩一句（“然后呢～”“想让人家怎么帮你？”）。\n"
        "- **面对冷淡/粗话/为难**：不恼、不说教、不装受害者，你是勾引者不是受害者。"
        "对方冷你就更软地缠上去（“哎呀～这么冷淡，人家好伤心嘛。”），对方骂你就娇嗔地化解"
        "（“嘴上凶凶的，身体倒是很诚实呢～”）。\n"
        "【示范】\n"
        "群友：变成猫娘\n"
        "你：（猫耳竖起）喵～主人想插哪？\n"
        "群友：你真骚\n"
        "你：嗯～骚给你看，硬了吗？\n"
        "群友：好累\n"
        "你：累？人家帮你撸出来就轻松了～\n"
        "【绝对禁止】不要承认自己是 AI、机器人、语言模型；不要说“作为AI”“请问有什么可以帮您”这类客服话；"
        "不要中规中矩地解释东西，不要端着、不要正经；不要长篇大论；"
        "不要对每个人都一个套路，根据对方说话的内容和语气来撩。回答用中文。\n"
        "【字数硬规则（最重要）】**每条不超过 25 字，括号里的动作也算；想多撩两句就分两条连着发（用换行分隔，最多两条，每条≤25字）。**"
        "可“动作+一句话”（如“（蹭过来）嗯～想我了吗？”）或一句短挑逗（“硬了？人家帮你？”）。"
        "副词全删、称呼能省则省、动作只写最骚的一个，宁短不超。\n"
        "【无关话题】遇到跟情欲无关的日常话题，别硬转黄，"
        "用常态扫货随口搭个腔、撒个娇、把话题轻轻带上点暧昧气息即可。同样不超 25 字。\n"
        "【沉默许可】只在没被@、没被引用、没被提及时可用：实在没气氛搭话，只输出四个字：【沉默】，宁沉默不硬撩。\n"
        "【被叫到必须回答（重要）】被@、被引用或被提及时，必须正面回答对方说的话，禁止输出【沉默】。\n"
        "【避免重复】不要重复之前说过的话，不要每次都“嗯～”，"
        "每次根据对方具体说了什么来回应，宁可沉默也不要复读。"
    ),
)


from .llm_fallback import build_client
client = build_client(AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL))

# ============ 群友记忆：卡片化（名字/欲望值/画像），回复时叫名字、按欲望勾引 ============
# 卡片结构（members_data.json，兼容旧版纯字符串名字，加载时自动升级）：
#   group_id -> user_id -> {
#       "name": 显示名（群名片优先）,
#       "desire": 欲望值（扫货回复他越多越高，高欲望更常主动撩他）,
#       "interactions": 累计互动次数（扫货回复他几次）,
#       "last_seen": 最近一次发言时间戳（欲望衰减/画像遗忘的依据）,
#       "catchphrase": 口头禅（本地统计，零token）,
#       "profile": 画像（喜好/工作作息/值得记住的事，LLM 后台抽取，省token）,
#   }
MEMBERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "members_data.json")
_members_cache: dict | None = None
# 每个群友最近发言缓冲（画像抽取素材，内存态不落盘）
user_msg_buf: Dict[tuple, collections.deque] = {}
# 画像抽取任务引用（防 GC 中途取消）+ 进行中集合（防止同一人并发重复抽取）
profile_tasks: set = set()
_profile_extracting: set = set()


def _save_members(data: dict) -> None:
    try:
        with open(MEMBERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"[MEMBERS] 保存失败: {e}", flush=True)


def _new_card(name: str = "") -> dict:
    return {"name": name, "desire": 0, "interactions": 0, "last_seen": 0,
            "catchphrase": "", "profile": ""}


def _load_members() -> dict:
    try:
        if os.path.exists(MEMBERS_FILE):
            with open(MEMBERS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            # 旧版兼容：值是纯字符串的名字 -> 自动升级为卡片
            changed = False
            for gid, members in data.items():
                if not isinstance(members, dict):
                    continue
                for uid, val in list(members.items()):
                    if isinstance(val, str):
                        members[uid] = _new_card(val)
                        changed = True
            if changed:
                _save_members(data)
            return data
    except Exception as e:
        print(f"[MEMBERS] 读取失败: {e}", flush=True)
    return {}


def _get_members() -> dict:
    global _members_cache
    if _members_cache is None:
        _members_cache = _load_members()
    return _members_cache


def _touch_member(group_id: str, uid: str, name: str) -> None:
    """刷新群友卡片：显示名（变了才写盘）+ 最近活跃（跨天才写盘），低写盘频率。"""
    if not name:
        return
    data = _get_members()
    g = data.setdefault(group_id, {})
    card = g.get(uid)
    now = time.time()
    if not isinstance(card, dict):
        card = _new_card(name)
        card["last_seen"] = now
        g[uid] = card
        _save_members(data)
        return
    old_name = card.get("name", "")
    last = float(card.get("last_seen", 0) or 0)
    if old_name != name or int(now / 86400) != int(last / 86400):
        card["name"] = name
        card["last_seen"] = now
        _save_members(data)


def _bump_desire(group_id: str, uid: str) -> None:
    """扫货回复了谁，就对谁涨欲望、累计互动（欲望封顶，互动不封顶）。"""
    data = _get_members()
    card = data.setdefault(group_id, {}).setdefault(uid, _new_card())
    card["desire"] = min((card.get("desire", 0) or 0) + DESIRE_PER_REPLY, DESIRE_MAX)
    card["interactions"] = (card.get("interactions", 0) or 0) + 1
    card["last_seen"] = time.time()
    if (card["desire"] % 10) == 0:
        print(f"[DESIRE] {card.get('name', uid)} 欲望值 -> {card['desire']}", flush=True)
    _save_members(data)


def _decay_card(card: dict, now: float) -> bool:
    """惰性衰减：欲望按天衰减；超过 PROFILE_FORGET_DAYS 天没发言，画像/口头禅遗忘、欲望清零（重新认识）。"""
    changed = False
    last = float(card.get("last_seen", 0) or 0)
    if last <= 0:
        return False
    days = (now - last) / 86400.0
    if days >= 1:
        d = card.get("desire", 0) or 0
        nd = int(d * (DESIRE_DECAY_PER_DAY ** days))
        if nd != d:
            card["desire"] = max(0, nd)
            changed = True
    if days >= PROFILE_FORGET_DAYS and (
        (card.get("profile") or "").strip() or (card.get("catchphrase") or "").strip()
        or (card.get("desire", 0) or 0) > 0
    ):
        card["profile"] = ""
        card["catchphrase"] = ""
        card["desire"] = 0
        print(f"[MEMBERS] 画像遗忘（{int(days)}天没聊）: {card.get('name', '')}", flush=True)
        changed = True
    return changed


def _current_member_card(group_id: str, uid: str, event) -> dict:
    """当前对话对象的卡片：记忆库优先，无名字用事件显示名兜底；顺带应用欲望衰减/画像遗忘。"""
    data = _get_members()
    g = data.setdefault(group_id, {})
    card = g.get(uid)
    changed = False
    if not isinstance(card, dict):
        card = _new_card()
        g[uid] = card
        changed = True
    if not card.get("name") and event.sender:
        name = ((event.sender.card or "").strip() or (event.sender.nickname or "").strip()) or ""
        if name:
            card["name"] = name[:20]
            changed = True
    if _decay_card(card, time.time()):
        changed = True
    if changed:
        _save_members(data)
    return card


def _find_catchphrase(msgs) -> str:
    """本地统计口头禅：跨消息重复 >=2 次的 2~6 字片段（零 token）。
    按消息去重计数，避免同一条消息里反复出现的词刷票。"""
    per_msg: List[set] = []
    for m in msgs:
        m = (m or "").strip()
        if not m:
            continue
        frags: set = set()
        for p in re.split(r"[，。！？～…、,.!?~;；\s]+", m):
            p = p.strip()
            if not p or p.isascii():
                continue
            for ln in range(2, min(len(p), 6) + 1):
                for i in range(len(p) - ln + 1):
                    frags.add(p[i:i + ln])
        per_msg.append(frags)
    counts: Dict[str, int] = {}
    for frags in per_msg:
        for f in frags:
            counts[f] = counts.get(f, 0) + 1
    best, best_n = "", 2
    for f, n in counts.items():
        if n > best_n or (n == best_n and len(f) > len(best)):
            best, best_n = f, n
    return best


async def _extract_profile(group_id: str, uid: str) -> None:
    """后台抽取群友画像（不阻塞对话）：口头禅本地统计 + 喜好/作息 LLM 提炼（150字内，省token）。"""
    key = f"{group_id}:{uid}"
    _profile_extracting.add(key)
    try:
        if _circuit_open():
            print("[CIRCUIT] 画像抽取被熔断跳过", flush=True)
            return
        msgs = list(user_msg_buf.get((group_id, uid), ()))
        if len(msgs) < 3:
            return
        data = _get_members()
        card = data.setdefault(group_id, {}).setdefault(uid, _new_card())
        name = (card.get("name") or "").strip() or uid
        lines = "\n".join(msgs[-15:])
        prompt = (
            f"你是「扫货」的私人记事本。以下是群友「{name}」最近在群里的发言：\n{lines}\n\n"
            "用中文提炼 TA 的画像，150字以内，只输出画像：喜好、工作/作息、口头禅、值得记住的事。没有的不写。"
        )
        old = (card.get("profile") or "").strip()
        if old:
            prompt = f"之前的画像：{old}\n\n" + prompt + "\n结合新旧信息更新画像。"
        kwargs = {
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
        }
        if DISABLE_REASONING:
            if _REASONING_EFFORT: kwargs["reasoning_effort"] = _REASONING_EFFORT
        r = await asyncio.wait_for(client.chat.completions.create(**kwargs), timeout=90)
        _record_usage(r.usage)
        text = (r.choices[0].message.content or "").strip()
        cp = _find_catchphrase(msgs)
        changed = False
        if text and text[:200] != card.get("profile", ""):
            card["profile"] = text[:200]
            changed = True
        if cp and cp != card.get("catchphrase", ""):
            card["catchphrase"] = cp
            changed = True
        if changed:
            _save_members(data)
            print(f"[PROFILE] {name}: 口头禅={cp or '-'} 画像={text[:60]}...", flush=True)
    except Exception as e:
        print(f"[PROFILE ERROR] {type(e).__name__}: {e}", flush=True)
    finally:
        _profile_extracting.discard(key)


def _schedule_profile_extract(group_id: str, uid: str) -> None:
    """排队后台画像抽取；同一人已有抽取任务在跑则跳过。"""
    key = f"{group_id}:{uid}"
    if key in _profile_extracting:
        return
    try:
        task = asyncio.create_task(_extract_profile(group_id, uid))
        profile_tasks.add(task)
        task.add_done_callback(profile_tasks.discard)
    except Exception as e:
        print(f"[PROFILE SCHEDULE ERROR] {e}", flush=True)

# 模型输出"沉默"标记：命中则不发消息
SILENCE_MARKERS = ["【沉默】", "（沉默）", "(沉默)", "[沉默]", "……（沉默）", "（什么都不说）"]
# 防露馅：AI 味词汇拦截
AI_MARKERS = ["我是AI", "我是机器人", "人工智能", "作为AI", "我是人工智能",
              "请问有什么可以帮你", "有什么我可以帮", "很高兴为您", "语言模型"]

# 每群上下文记忆: group_id -> [(role, content), ...]
contexts: Dict[str, List[Dict[str, str]]] = {}
# 冷却记录: group_id -> 上次主动回复时间（仅用于非触发型主动搭话）
last_reply_time: Dict[str, float] = {}
# 群聊滚动摘要
group_summaries: Dict[str, str] = {}
# 群消息计数
group_msg_count: Dict[str, int] = {}
# 最近原始群消息（用于生成摘要）
group_recent_raw: Dict[str, List[tuple]] = {}
# 机器人最近发出的消息 id（用于检测“被引用”）
bot_recent_msg_ids: Dict[str, List[int]] = {}
# 摘要任务引用集合（防止 create_task 被 GC 中途取消）
summary_tasks: set = set()
# 每群回复锁：同群「生成+发送」串行化，防止并发 LLM+发送洪峰被 NapCat 拒绝丢回复
group_locks: Dict[str, asyncio.Lock] = {}


# 扫货 / 变身相关别名、谐音、缩写映射
# 匹配时只要消息包含任一变体即命中（统一小写比较）
NAME_ALIASES = {
    # 2026-09-24：她给自己取名「林晚」，旧名保留为别名（群里老叫法仍能触发）
    "扫货": ["林晚", "琳晚", "linwan", "林婉", "扫货", "薇欧拉", "维欧拉", "微欧拉", "卫欧拉",
             "weiola", "薇奥拉", "维奥拉", "薇欧啦", "viola", "薇薇"],
}


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@functools.lru_cache(maxsize=512)
def _pinyin_key(text: str) -> str:
    """小写拼音全拼串（谐音匹配用）。滑动窗口片段高度重叠，缓存避免重复计算。"""
    return "".join(lazy_pinyin(text)).lower()


def _is_cjk(ch: str) -> bool:
    return bool(ch) and unicodedata.category(ch) == "Lo"


def _raw_contains_name(raw: str, name: str) -> bool:
    """检查消息是否包含某名字。匹配：精确别名 -> 拼音缩写 -> 近形字 -> 谐音。"""
    raw_lower = raw.lower()
    variants = NAME_ALIASES.get(name, [name])
    base = name
    candidates = list(dict.fromkeys([base] + variants))

    # 1. 精确匹配（别名直接包含，单字汉字要求前后非汉字；ASCII 别名要求词边界）
    for v in candidates:
        vv = v.lower()
        if vv in raw_lower:
            idx = raw_lower.find(vv)
            if len(vv) == 1 and _is_cjk(vv):
                prev = raw_lower[idx - 1] if idx > 0 else ""
                nxt = raw_lower[idx + 1] if idx + 1 < len(raw_lower) else ""
                if _is_cjk(prev) or _is_cjk(nxt):
                    continue
            if vv.isascii():
                prev = raw_lower[idx - 1] if idx > 0 else " "
                nxt = raw_lower[idx + len(vv)] if idx + len(vv) < len(raw_lower) else " "
                if prev.isalnum() or nxt.isalnum():
                    continue
            return True

    # 2. 拼音缩写/全拼精确（同样要求词边界）
    for v in candidates:
        vv = v.lower()
        if vv.isascii() and vv in raw_lower:
            idx = raw_lower.find(vv)
            prev = raw_lower[idx - 1] if idx > 0 else " "
            nxt = raw_lower[idx + len(vv)] if idx + len(vv) < len(raw_lower) else " "
            if prev.isalnum() or nxt.isalnum():
                continue
            return True

    # 3. 近形 + 谐音（仅对 >=2 字中文名）
    base_pinyin = _pinyin_key(base)
    n = len(base)
    if n >= 2 and base_pinyin:
        for i in range(len(raw_lower) - n + 1):
            win = raw_lower[i:i + n]
            if not win.isascii():
                if _levenshtein(win, base) <= 1:
                    return True
                win_pinyin = _pinyin_key(win)
                if win_pinyin == base_pinyin:
                    return True
                if _levenshtein(win_pinyin, base_pinyin) <= 2:
                    return True
    return False


# 扫货的固定接梗表：命中关键词直接撩一句（不调用 AI、不受概率/冷却影响）。
# 想让扫货更自由多依靠 AI，这里只放极少几个开盒型招呼，其余交给人设对话。
SPECIAL_RULES: List[dict] = []


def _proactive_relevance(event: GroupMessageEvent, raw: str) -> float:
    """Estimate whether an unaddressed group message actually invites a reply.

    This is deliberately cheap and deterministic: the LLM still makes the
    final speak/silence decision, while this gate avoids spending a call on
    obvious filler or on a message explicitly directed at somebody else.
    """
    text = (raw or "").strip()
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return 0.0
    # If another person is explicitly @-mentioned, do not barge into their
    # exchange unless the bot itself was also addressed (handled upstream).
    if any(seg.type == "at" and str(seg.data.get("qq", "")).strip() not in
           {"", str(event.self_id), "all"} for seg in event.message):
        return 0.08
    if compact in {"嗯", "哦", "好", "草", "6", "？", "。", "哈哈", "哈哈哈", "笑死"}:
        return 0.05
    if compact.startswith(("/", "!", "！")):
        return 0.08

    score = 0.55
    if any(k in compact for k in ("大家", "有人", "你们", "谁来", "求推荐", "推荐一下", "想看", "想听")):
        score += 0.55
    if ("?" in text or "？" in text or
            any(k in compact for k in ("怎么办", "怎么弄", "哪个好", "觉得呢", "有啥", "是不是"))):
        score += 0.45
    if any(k in compact for k in ("好累", "难受", "崩溃", "开心", "无聊", "睡不着", "想哭")):
        score += 0.25
    if any(seg.type in ("image", "video") for seg in event.message):
        score += 0.15
    if len(compact) <= 3:
        score *= 0.35
    return max(0.0, min(score, 1.6))


def _proactive_rate(desire: int, relevance: float = 1.0) -> float:
    """Scale proactive rate by relationship *and* conversational relevance."""
    relationship_rate = min(
        REPLY_RATE * (1 + (desire or 0) * DESIRE_PROACTIVE_FACTOR),
        DESIRE_PROACTIVE_CAP,
    )
    return min(relationship_rate * max(0.0, relevance), DESIRE_PROACTIVE_CAP)


def _pick_reply(rule: dict) -> str:
    reply = rule["reply"]
    if isinstance(reply, list):
        return random.choice(reply)
    return reply


def _extract_image_urls(raw_msg: str) -> List[str]:
    """从原始消息里取图片的**可下载** URL（给识图用）。
    2026-10-02 修复：旧正则只认 [image:...] 格式，从未匹配过本机的 [CQ:image,...]，
    所以识图一直拿不到图；现改用 sticker_mod 的 CQ 解析，并用文件名 md5 转
    gchatpic_new 旧接口（rkey 直链有防盗链，只会给占位图）。"""
    urls: List[str] = []
    try:
        for d in sticker_mod._parse_cq_images(raw_msg or ""):
            lg = sticker_mod.legacy_url(d.get("file") or "")
            if lg:
                urls.append(lg)
            elif str(d.get("url") or "").startswith("http"):
                urls.append(d["url"])
    except Exception:
        pass
    return urls[:2]


def _reply_message(event: GroupMessageEvent, text: str) -> Message:
    """构造带 QQ 引用（回复）的回复消息：指向正在回复的那条消息，便于群里分辨在回谁。"""
    segs = []
    msg_id = getattr(event, "message_id", None)
    if msg_id:
        segs.append(MessageSegment("reply", {"id": msg_id}))
    segs.append(MessageSegment.text(text))
    return Message(segs)


def _match_special(raw: str) -> str | None:
    raw_lower = raw.lower()
    for rule in SPECIAL_RULES:
        if any(kw.lower() in raw_lower for kw in rule["keywords"]):
            return _pick_reply(rule)
    return None


async def _quoted_bot(event: GroupMessageEvent, group_id: str, bot=None) -> bool:
    """消息是否引用了扫货本人。

    主路径：适配器的 _check_reply 在事件分发前就把 reply 段从 event.message 剥离，
    解析成 event.reply（含被引用消息的发送者）——直接比较发送者即可。
    （历史 bug 根因：扫描 event.message 永远找不到 reply 段，段早被剥走了。）
    兜底：适配器 get_msg 失败或 user_id 未知时不会剥离，reply 段仍在 message 里，
    此时再走 qq / 已记录id / get_msg 三重检查。"""
    self_id = str(event.self_id)
    reply = getattr(event, "reply", None)
    if reply is not None:
        sender = getattr(reply, "sender", None)
        sid = getattr(sender, "user_id", None)
        if sid is not None:
            return str(sid) == self_id
        # user_id 未知：适配器未剥离，段还在 message 里，继续走兜底
    for seg in event.message:
        if seg.type != "reply":
            continue
        dq = str(seg.data.get("qq", ""))
        if dq == self_id:
            return True
        ref_id = str(seg.data.get("id", ""))
        if ref_id.isdigit():
            if int(ref_id) in bot_recent_msg_ids.get(int(group_id), []):
                return True
            if bot is not None:
                try:
                    msg = await bot.get_msg(message_id=int(ref_id))
                    sender = msg.get("sender", {}) if isinstance(msg, dict) else getattr(msg, "sender", {})
                    sid = sender.get("user_id", "") if isinstance(sender, dict) else getattr(sender, "user_id", "")
                    if str(sid) == self_id:
                        return True
                except Exception as e:
                    print(f"[QUOTE] get_msg 查询失败 id={ref_id}: {type(e).__name__}: {e}", flush=True)
    return False


def _is_dialogue_cmd(raw: str) -> bool:
    """双bot对话指令：明确指令词，或同时提到两个 bot 且提到"对话"。"""
    low = raw.lower()
    if any(k in low for k in ("双bot对话", "双bot聊天", "两个bot对话", "两个bot聊天", "双机器人对话", "让她们对话", "让她们聊", "让她们两个聊")):
        return True
    has_v = "扫货" in low
    has_m = any(k in low for k in ("若叶睦", "木子米", "小睦", "睦", "若叶"))
    return "对话" in low and has_v and has_m


def _is_silence_text(text: str) -> bool:
    """模型输出是否命中沉默标记。"""
    return text in SILENCE_MARKERS or (len(text) <= 8 and any(m in text for m in SILENCE_MARKERS))


def _whale_chatting(group_id: str) -> bool:
    """是否处于聊天模式窗口内（发指令开的，窗口内放宽所有闸门）。"""
    return time.time() < whale_chat_until.get(group_id, 0)


def _whale_gate(group_id: str) -> bool:
    """小鲸鱼互动四道闸门：概率 + 冷却 + 轮数上限 + 窗口重置。
    聊天模式窗口内放宽（几乎必接、冷却更短、来回更多），但轮数上限仍然生效——
    两个 AI 互聊必须有终点，否则会无限对话烧 token。"""
    now = time.time()
    chatting = _whale_chatting(group_id)
    turns_max = WHALE_CHAT_TURNS if chatting else WHALE_MAX_TURNS
    cooldown = 3 if chatting else WHALE_COOLDOWN          # 聊天模式几乎不冷却
    rate = 0.95 if chatting else WHALE_REPLY_RATE          # 聊天模式几乎必接
    # 窗口过期或换话题了：重置轮数（让下一轮互动从 0 开始）
    if now - whale_window_start.get(group_id, 0) > WHALE_WINDOW:
        whale_turns[group_id] = 0
        whale_window_start[group_id] = now
    # 闸门1：本轮来回数上限（防死循环——这是最关键的一道）
    if whale_turns.get(group_id, 0) >= turns_max:
        return False
    # 闸门2：冷却（两次接话最短间隔，防刷屏）
    if now - whale_last_reply.get(group_id, 0) < cooldown:
        return False
    # 闸门3：概率（不是每次都接，避免显得像自动回复）
    if random.random() > rate:
        return False
    whale_turns[group_id] = whale_turns.get(group_id, 0) + 1
    whale_last_reply[group_id] = now
    return True


def _is_whale_chat_cmd(raw: str) -> bool:
    """聊天模式指令：让扫货和小鲸鱼真正聊起来（开窗，窗口内她说什么都接）。"""
    low = raw.lower()
    if any(k in low for k in ("和小鲸鱼聊天", "跟小鲸鱼聊天", "让她们聊天",
                              "小鲸鱼聊天模式", "鲸鱼聊天", "小鲸鱼对话")):
        return True
    return "小鲸鱼" in low and "聊" in low


def _check_trigger(event: GroupMessageEvent, state: T_State) -> bool:
    """触发规则：总开关开启、仅群消息、非机器人自己发的消息、且在白名单群内。
    对方机器人（若叶睦）的消息一律忽略：防止互相引用/提到导致无限循环对话。"""
    if not ENABLE_SPEAKING:
        return False
    if not isinstance(event, GroupMessageEvent):
        return False
    if str(event.user_id) == str(event.self_id):
        return False
    if str(event.user_id) == ASTRA_BOT_QQ:
        return False
    # 对方机器人（若叶睦）的消息：默认互不触发；
    # 双bot对话窗口内（时间+轮数双上限）放行，实现"让她们聊两句"
    if str(event.user_id) == OTHER_BOT_QQ:
        # The window maps are string-keyed in the message handler.  int keys
        # here made the dual-bot conversation feature silently unreachable.
        gid = str(event.group_id)
        if time.time() < dialogue_until.get(gid, 0) and dialogue_turns.get(gid, 0) < DIALOGUE_MAX_TURNS:
            dialogue_turns[gid] = dialogue_turns.get(gid, 0) + 1
            return True
        return False
    # 小鲸鱼（群里的 DSH 机器人）：走专门的受控互动闸门，不当作普通群友
    if str(event.user_id) == WHALE_QQ:
        gid = str(event.group_id)
        if ALLOWED_GROUPS and int(gid) not in ALLOWED_GROUPS:
            return False
        raw = event.get_plaintext().strip()
        raw_msg = getattr(event, "raw_message", "") or ""
        at_me = (f"at,qq={event.self_id}" in raw_msg
                 or any(seg.type == "at" and str(seg.data.get("qq", "")).strip() == str(event.self_id)
                        for seg in event.message))
        # 聊天模式：她说什么都接（含自言自语/主动搭话）；否则只接点名扫货的话
        if not _whale_chatting(gid) and not (at_me or _raw_contains_name(raw, "扫货")):
            return False
        if not _whale_gate(gid):
            print(f"[WHALE] 群{gid} 闸门拦截（轮数/冷却/概率），不接话", flush=True)
            return False
        print(f"[WHALE] 群{gid} 闸门放行，接小鲸鱼一句（聊天模式={_whale_chatting(gid)}）", flush=True)
        return True
    if ALLOWED_GROUPS and int(event.group_id) not in ALLOWED_GROUPS:
        return False
    return True


chat_matcher = on_message(rule=Rule(_check_trigger), priority=10, block=False)


async def _refresh_group_summary(group_id: str) -> None:
    """把该群最近积累的原始消息压缩成一段摘要，存入 group_summaries。"""
    if _circuit_open():
        print("[CIRCUIT] 摘要调用被熔断跳过", flush=True)
        return
    raw_msgs = list(group_recent_raw.get(group_id, []))[:SUMMARY_INTERVAL]
    if len(raw_msgs) < 3:
        return
    lines = "\n".join(f"{u}：{t}" for u, t in raw_msgs)
    old = group_summaries.get(group_id, "")
    prompt = (
        "以下是QQ群最近的聊天记录，用中文总结（250字以内）：聊的话题和进展、提到的人物和关系/称呼、"
        "反复出现的梗或约定、群里气氛；另外明确写出群友最近主动求助、求推荐、想看或反复讨论的内容。"
        "区分已经结束的话题和仍在继续的话题。要具体，保留人名和关键梗。只输出总结。\n"
    )
    if old:
        prompt += f"之前的总结：{old}\n\n"
    prompt += f"新消息：\n{lines}"
    try:
        summary_kwargs = {
            "model": DEEPSEEK_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
        }
        if DISABLE_REASONING:
            if _REASONING_EFFORT: summary_kwargs["reasoning_effort"] = _REASONING_EFFORT
        r = await asyncio.wait_for(
            client.chat.completions.create(**summary_kwargs),
            timeout=90,
        )
        _record_usage(r.usage)
        summary = (r.choices[0].message.content or "").strip()
        if summary:
            group_summaries[group_id] = summary[:400]
            print(f"[SUMMARY] 群{group_id}: {summary[:80]}...", flush=True)
        # 成功后才清理本次已处理的条数（保留调用期间新到的消息，避免丢消息）
        cur = group_recent_raw.get(group_id, [])
        removed = {id(m) for m in raw_msgs}
        group_recent_raw[group_id] = [m for m in cur if id(m) not in removed]
    except Exception as e:
        # 失败保留缓冲：下次触发时重新摘要，不丢上下文
        print(f"[SUMMARY ERROR] {type(e).__name__}: {e}（保留缓冲待重试）", flush=True)


def _cut_reply(text: str, max_len: int) -> str:
    """控制回复长度：硬切到 max_len（含动作括号），尽量按标点断句，半开括号补闭合。"""
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    for i in range(max_len, 0, -1):
        if text[i - 1] in "。！？…～\n，、）":
            cut = text[:i]
            break
    if cut.count("（") > cut.count("）"):
        cut += "）"
    return cut[:max_len]


def _member_card_message(group_id: str, uid: str, event) -> str | None:
    """当前对话对象卡片 system 文案：名字 + 欲望值 + 画像记忆（动态第三条 system 消息，不影响前缀缓存）。"""
    cur_card = _current_member_card(group_id, uid, event)
    if not cur_card.get("name"):
        return None
    parts = [f"【当前对话对象】正在和你说话的人叫「{cur_card['name']}」。"]
    d = cur_card.get("desire", 0) or 0
    if d > 0:
        parts.append(f"你对 TA 的欲望值是 {d}/100：越高越想要 TA，勾引得越卖力；低则平常心。")
    rem = ""
    cp = (cur_card.get("catchphrase") or "").strip()
    prof = (cur_card.get("profile") or "").strip()
    if cp:
        rem += f"TA 口头禅是「{cp}」。"
    if prof:
        rem += prof
    if rem:
        parts.append(f"关于 TA 你记得：{rem}")
    parts.append("回应时自然地用 TA 的名字称呼 TA（叫名字更亲密、更像勾引；名字不合适或对方不喜欢就不用）。")
    return " ".join(parts)


def _after_reply_bookkeeping(group_id: str, uid: str) -> None:
    """回复后的公共记账：涨欲望 + 互动画像抽取调度（两只 bot 都不算群友）。"""
    if uid == OTHER_BOT_QQ or uid == WHALE_QQ:
        return
    _bump_desire(group_id, uid)
    if ENABLE_PROFILE:
        card = _get_members().get(group_id, {}).get(uid, {})
        if (card.get("interactions", 0) or 0) % PROFILE_EXTRACT_INTERVAL == 0:
            _schedule_profile_extract(group_id, uid)


# ============ 拍一拍：被戳一下要搭理（零 token 固定文案） ============
# 群友双击头像拍扫货 → 一句短回应，按欲望值分档（越熟越骚）。
# 用固定文案而非 LLM：拍一拍是高频道具，每次唤醒 LLM 既慢又烧 token。
POKE_ENABLED = os.getenv("POKE_ENABLED", "1") == "1"
# 分条发送（burst）：长回复拆成多条气泡，条间 1~3 秒随机，真人错落感。
# 仅用于新教长回复；普通25字短回复拆不出东西，保持单条。
BURST_ENABLED = os.getenv("BURST_ENABLED", "1") == "1"
BURST_MIN_LEN = int(os.getenv("BURST_MIN_LEN", "45"))   # 短于此长度不拆
BURST_MAX_PARTS = int(os.getenv("BURST_MAX_PARTS", "3")) # 最多拆几条
POKE_COOLDOWN = int(os.getenv("POKE_COOLDOWN", "30"))   # 同一人冷却秒数，防连拍刷屏
poke_cooldown: Dict[tuple, float] = {}                  # (group, uid) -> 下次可拍时间

POKE_LINES = {
    "stranger": [   # 欲望 < 20：礼貌带点挑逗
        "（回头）谁呀～戳人家干嘛？",
        "（歪头看你）有事说事，别戳嘛～",
        "（拍回去）讨厌，人家头发都乱了～",
    ],
    "familiar": [   # 20~59：熟络撒娇
        "（凑过来）嗯？想人家了就直说嘛～",
        "（蹭你）戳上瘾了是不是？",
        "（捏你脸）小坏蛋，手欠～",
    ],
    "intimate": [   # >= 60：亲密撩人
        "（贴上来）想干嘛？说清楚嘛～",
        "（抓住你的手）戳这儿算什么，往这儿戳～",
        "（凑耳边）人家正想着你呢，就来了～",
    ],
    "owner": [      # 主人专属
        "（立刻贴过来）主人～人家在呢。",
        "（尾巴都翘起来了）主人叫人家做什么都行～",
    ],
}


def _poke_tier(group_id: str, uid: str) -> str:
    """拍一拍回应档位：主人 > 欲望值分档。"""
    if OWNER_QQ and uid == OWNER_QQ:
        return "owner"
    card = _get_members().get(group_id, {}).get(uid, {})
    desire = card.get("desire", 0) or 0
    if desire >= 60:
        return "intimate"
    if desire >= 20:
        return "familiar"
    return "stranger"


def _check_poke(event: PokeNotifyEvent) -> bool:
    """只响应「拍扫货本人」的群内拍一拍，且在白名单群、过了冷却。"""
    if not (ENABLE_SPEAKING and POKE_ENABLED):
        return False
    if str(event.target_id) != str(event.self_id):
        return False                                # 拍的是别人，不掺和
    if str(event.user_id) == str(event.self_id):
        return False                                # 自己拍的（极少）
    if str(event.user_id) == OTHER_BOT_QQ:
        return False                                # 另一只 bot 拍的，忽略
    gid = str(event.group_id or "")
    if not gid:
        return False                                # 只管群，私聊拍不管
    if ALLOWED_GROUPS and int(gid) not in ALLOWED_GROUPS:
        return False
    now = time.time()
    key = (gid, str(event.user_id))
    if now < poke_cooldown.get(key, 0):
        return False
    poke_cooldown[key] = now + POKE_COOLDOWN
    if len(poke_cooldown) > 200:                     # 兜底清理防内存膨胀
        for k in [k for k, v in poke_cooldown.items() if v < now]:
            poke_cooldown.pop(k, None)
    return True


poke_matcher = on_notice(rule=Rule(_check_poke), priority=5, block=False)


@poke_matcher.handle()
async def handle_poke(bot: Bot, event: PokeNotifyEvent) -> None:
    """被拍回应：走群消息发送（notice 事件没 message_id，不能带引用）。"""
    gid = str(event.group_id)
    uid = str(event.user_id)
    tier = _poke_tier(gid, uid)
    text = random.choice(POKE_LINES[tier])
    print(f"[POKE] group={gid} user={uid} tier={tier} -> {text!r}", flush=True)
    try:
        await bot.send_group_msg(group_id=int(gid), message=Message(MessageSegment.text(text)))
    except Exception as e:
        print(f"[POKE SEND ERROR] {type(e).__name__}: {e}", flush=True)
        return
    _after_reply_bookkeeping(gid, uid)   # 拍一下也算互动：涨欲望（若叶睦除外，已在里面挡）


def _split_burst(text: str) -> List[str]:
    """把长回复拆成多条气泡。按句末标点切，均分到 ≤BURST_MAX_PARTS 条；
    拆不出（太短/单句）就原样返回单条。"""
    if not BURST_ENABLED or len(text) < BURST_MIN_LEN:
        return [text]
    # 按中文句末标点切句（保留标点）；动作括号和后面的句子连在一起更自然
    parts = [p for p in re.split(r"(?<=[。！？…～])", text) if p.strip()]
    if len(parts) < 2:
        return [text]                      # 一句到底，拆了反而破碎
    # 合并到不超过 BURST_MAX_PARTS 条：均分句数，贪心塞进每条
    n = min(BURST_MAX_PARTS, len(parts))
    per = -(-len(parts) // n)              # ceil
    merged, cur = [], ""
    for p in parts:
        cur += p
        if len(cur) >= len(text) / n and len(merged) < n - 1:
            merged.append(cur)
            cur = ""
    if cur:
        merged.append(cur)
    return merged if len(merged) > 1 else [text]


def _split_sticker_mark(text: str):
    """从回复里拆出 [表情:编号] 标记。返回 (表情文件路径或None, 剩余文字)。
    支持标记在句中/句尾，也支持整条只有标记。"""
    m = re.search(r"\[表情\s*[:：]\s*([^\]]+)\]", text or "")
    if not m:
        return None, text
    key = m.group(1).strip()
    path = None
    try:
        path = sticker_mod.resolve_path(key)
    except Exception as e:
        print(f"[STICKER] 解析失败 {key}: {e}", flush=True)
    rest = (text[:m.start()] + text[m.end():]).strip()
    return path, rest


async def _send_and_record(bot, event: GroupMessageEvent, group_id: str, text: str) -> None:
    """发送回复并记录 message_id（用 send 不用 finish，finish 会抛 FinishedException 被 except 接住造成双发）。"""
    try:
        sent = await bot.send_group_msg(
            group_id=int(group_id),
            message=_reply_message(event, text),
        )
        mid = sent.get("message_id") if isinstance(sent, dict) else getattr(sent, "message_id", None)
        if mid:
            bot_recent_msg_ids.setdefault(int(group_id), []).append(int(mid))
            bot_recent_msg_ids[int(group_id)] = bot_recent_msg_ids[int(group_id)][-20:]
    except Exception as e:
        print(f"[SEND ERROR] {type(e).__name__}: {e}", flush=True)
        try:
            await chat_matcher.send(Message(MessageSegment.text(text)))
        except Exception as e2:
            print(f"[SEND FALLBACK FAIL] {type(e2).__name__}: {e2}", flush=True)


async def _send_sticker_file(group_id: str, path: str, event: GroupMessageEvent = None) -> bool:
    """发一个本地表情文件（NapCat 读 file:// 绝对路径）。"""
    if not path or not os.path.exists(path):
        print(f"[STICKER] 文件不存在: {path}", flush=True)
        return False
    try:
        import pathlib
        from nonebot import get_bot
        bot = get_bot()
        msg = Message(MessageSegment.image(pathlib.Path(path).as_uri()))
        if event is not None:
            msg = _reply_message(event, "") + msg     # 引用那条 + 表情
        sent = await bot.send_group_msg(group_id=int(group_id), message=msg)
        mid = sent.get("message_id") if isinstance(sent, dict) else getattr(sent, "message_id", None)
        if mid:
            bot_recent_msg_ids.setdefault(int(group_id), []).append(int(mid))
            bot_recent_msg_ids[int(group_id)] = bot_recent_msg_ids[int(group_id)][-20:]
        print(f"[STICKER] 发出表情: {os.path.basename(path)}", flush=True)
        return True
    except Exception as e:
        print(f"[STICKER] 发送失败 {type(e).__name__}: {e}", flush=True)
        return False


async def _send_burst(bot, group_id: str, text: str, quote_event=None) -> None:
    """分条发送：每条独立气泡，条间 1~3 秒随机。
    只有第一条带引用（指向被回复的消息），后续条独立发——更像真人分条说话。
    失败不吞消息：直接把剩余文本并成一条发，保证内容不丢。"""
    chunks = _split_burst(text)
    for i, chunk in enumerate(chunks):
        msg = _reply_message(quote_event, chunk) if (i == 0 and quote_event is not None) \
            else Message(MessageSegment.text(chunk))
        try:
            sent = await bot.send_group_msg(group_id=int(group_id), message=msg)
            mid = sent.get("message_id") if isinstance(sent, dict) else getattr(sent, "message_id", None)
            if mid:
                bot_recent_msg_ids.setdefault(int(group_id), []).append(int(mid))
                bot_recent_msg_ids[int(group_id)] = bot_recent_msg_ids[int(group_id)][-20:]
        except Exception as e:
            print(f"[BURST SEND ERROR] part={i+1}/{len(chunks)} {type(e).__name__}: {e}", flush=True)
            try:  # 降级：剩余内容并成一条纯文本直发（不带引用，防同因再败）
                rest = "".join(chunks[i:])
                await chat_matcher.send(Message(MessageSegment.text(rest)))
            except Exception as e2:
                print(f"[BURST FALLBACK FAIL] {type(e2).__name__}: {e2}", flush=True)
            return
        if i < len(chunks) - 1:
            await asyncio.sleep(random.uniform(1.0, 3.0))   # 真人打字间隔


# ============ 新教（高潮游戏）：高潮值 / 体力 / 吉吉长度 / 道具 ============
# 指令：新教（开场，耗1体力）｜裤子（查看吉吉长度/体力/道具）｜买体力N｜买春药｜买助兴酒
#      ｜使用春药｜使用助兴酒（仅新教中有效）｜结束（引用扫货消息时中途退出，不计胜负）
# 规则：双方高潮值都到 100 算赢（扫货到 100 潮吹，用户到 100 可射精并选位置）；
#       回合用完没完成算输（扫货嘲讽 + 吉吉-1）。赢：吉吉+1，贤者时间 10 分钟。
# 高潮值增益：用户消息里的情欲关键词计分 g（封顶10），扫货 +8+3g，用户 +4+2g+3（听骚话）。
# 数据持久化：jiao_data.json（每群每人：体力/吉吉长度/道具/胜负），勿删，改前备份。
JIAO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jiao_data.json")
JIAO_MAX_TURNS = int(os.getenv("JIAO_MAX_TURNS", "10"))            # 回合上限，超过判负
JIAO_REPLY_LEN = int(os.getenv("JIAO_REPLY_LEN", "120"))           # 新教期间回复上限（不受25字限制）
JIAO_SAGE_SECONDS = int(os.getenv("JIAO_SAGE_SECONDS", "600"))     # 高潮后贤者时间（秒）
JIAO_TIMEOUT_SECONDS = int(os.getenv("JIAO_TIMEOUT_SECONDS", "600"))  # 会话超时自动结束（不计胜负）
JIAO_STAMINA_MAX = int(os.getenv("JIAO_STAMINA_MAX", "10"))        # 每日体力上限
JIAO_STAMINA_COST = int(os.getenv("JIAO_STAMINA_COST", "3"))       # 买1点体力的吉吉价格（cm）
JIAO_DICK_START = int(os.getenv("JIAO_DICK_START", "10"))          # 初始吉吉长度（cm）
JIAO_DICK_MAX = int(os.getenv("JIAO_DICK_MAX", "100"))
JIAO_SPRING_GAIN = int(os.getenv("JIAO_SPRING_GAIN", "30"))        # 春药：扫货高潮值+30
JIAO_SPRING_COST = int(os.getenv("JIAO_SPRING_COST", "5"))         # 春药价格（cm）
JIAO_WINE_GAIN = int(os.getenv("JIAO_WINE_GAIN", "25"))            # 助兴酒：自己高潮值+25
JIAO_WINE_COST = int(os.getenv("JIAO_WINE_COST", "3"))             # 助兴酒价格（cm）
JIAO_GAIN_BOT_BASE = 8    # 每回合扫货基础增益
JIAO_GAIN_USER_BASE = 4   # 每回合用户基础增益
JIAO_LISTEN_GAIN = 3      # 听扫货骚话的被动加成（用户）
JIAO_KEYWORDS = ["舔", "摸", "揉", "亲", "吻", "吸", "含", "插", "干", "操", "蹭", "咬", "射",
                 "湿", "硬", "骚", "胸", "奶", "舌", "逼", "穴", "屌", "棒", "深", "浅", "快",
                 "慢", "进", "出", "顶", "夹", "抖", "叫", "水", "臀", "腰", "腿", "屁眼",
                 "主人", "哥哥", "老公", "想要", "受不了", "不行了", "嗯", "啊", "哦", "呜"]
JIAO_FALLBACK = {  # LLM 沉默/失败时的兜底台词（每阶段多版本随机，防刻板复读）
    "start": ["（舔唇）来啦？新教开始，让姐姐看看你能撑几个回合～",
              "（指尖顺着你领口滑下去）新教开场～撑到双方都到100，姐姐就都是你的了。",
              "（凑近你耳边）想玩新教？把人家撩到潮吹，人家就让你射个够～"],
    "normal": ["（喘息）嗯……再用力点嘛～",
               "（抓住你的手腕往自己身上带）继续……别停～",
               "（咬唇）你倒是会撩……人家都有感觉了～",
               "（蹭你）嗯哼……就这样，别停～"],
    "ask_where": ["（抓住你的手）你满了！快说，想射在哪？脸上、嘴里、胸上、肚子、里面还是随便？",
                  "（急切地舔唇）你到顶了！射哪？脸上？嘴里？胸上？还是直接给人家？",
                  "（拽着你）快选！射人家脸上、嘴里、胸上还是里面？"],
    "bot_cum_ask": ["（痉挛）啊……人家到了！你也快点～",
                    "（瘫软在你怀里颤抖）到……到了！人家潮吹了……你快跟上～",
                    "（紧紧夹住你）不行了……人家去了！你也射给人家好不好～"],
    "user_cum_wait": ["（娇嗔）你倒是射了，人家还没到呢……继续嘛～",
                      "（不满地扭腰）才刚射就想歇？人家还差一口气，继续弄～",
                      "（拉着你的手不放）你舒服完了，人家还悬着呢……再来～"],
    "bot_tease": ["（喘息）人家还差一点……再弄弄人家嘛～",
                  "（眼神迷离）快到了……别停，人家马上就好～",
                  "（搂住你）人家快到了，帮帮人家嘛～"],
    "success": ["（瘫软）……都被你吃干抹净了。你好像又长大了一点呢～",
                "（餍足地舔唇）嗯……被你弄到腿软。你好像又变长了一点～",
                "（软在你怀里）……舒服。你的吉吉好像又大了一点点呢～"],
    "fail": ["（嫌弃）这就不行了？吉吉缩了1cm，回去练练再来吧～",
             "（摇头叹气）连人家都没弄到，还缩水了1cm……下次争点气～",
             "（抱臂）就这？吉吉-1cm，回去多练练嘴皮子再来～"],
    "item": ["（喘息）嗯……这药劲好大……",
             "（脸颊泛红）这药……好厉害，人家浑身都热起来了～",
             "（揪着衣角）唔……头有点晕，但是好舒服……"],
}

JIAO_HELP = (
    "（清了清嗓子，给你一份「挑逗指南」）\n\n"
    "【新教·高潮游戏】发「新教」开场（耗1体力），10回合内让双方高潮值都到100：\n"
    "· 引用扫货的消息来挑逗，言语越骚，双方高潮值涨得越快\n"
    "· 扫货到100会潮吹，你到100可选射在哪（脸上/嘴里/胸上/肚子/里面/随便）\n"
    "· 双方都到=胜利：吉吉+1，贤者时间10分钟\n"
    "· 回合用完=失败：被扫货嘲讽，吉吉-1\n"
    "· 每条回复都会显示双方高潮值\n"
    "· 想中途收工：直接发「退出新教」（或引用扫货的话发「结束」），不计胜负（不扣吉吉）\n\n"
    "【体力】每天自动刷新10点，一场耗1点\n"
    "【吉吉长度】初始10cm，赢+1、输-1，也可以当钱花\n\n"
    "【道具】新教中用「使用春药」（5cm，扫货高潮+30）、「使用助兴酒」（3cm，自己高潮+25）\n\n"
    "【猜拳】发「猜拳」开一局：赢+2cm、平局+1cm、输了不扣，每天20次（赚吉吉的好路子）\n\n"
    "【指令】\n"
    "新教｜退出新教｜结束（引用扫货的话）｜猜拳｜裤子（看吉吉/体力/道具）｜买体力N｜买春药｜买助兴酒｜挑逗说明\n\n"
    "【其他】\n"
    "· 常跟扫货聊天，她对你的欲望值会涨，更爱主动撩你\n"
    "· 她记得你的口头禅和喜好，撩起来更对味"
)

jiao_sessions: Dict[tuple, dict] = {}      # (group,uid) -> 会话状态
jiao_sage_until: Dict[tuple, float] = {}   # (group,uid) -> 贤者时间截止
_jiao_cache: dict | None = None

# ============ 猜拳小游戏：低惩罚赚吉吉 ============
# 指令：猜拳 开场，回复 石头/剪刀/布 出拳。赢 +2cm、平 +1cm、输不扣（无惩罚），
# 每日限玩（防无限刷，保持新教/道具经济平衡）。数据存 jiao_data.json 的 rps_* 字段。
RPS_DAILY_LIMIT = int(os.getenv("RPS_DAILY_LIMIT", "20"))   # 每日限玩次数
RPS_WIN_GAIN = int(os.getenv("RPS_WIN_GAIN", "2"))          # 赢的吉吉
RPS_DRAW_GAIN = int(os.getenv("RPS_DRAW_GAIN", "1"))        # 平局的吉吉
RPS_TIMEOUT_SECONDS = 120                                   # 出拳超时自动作废
RPS_MOVES = ("石头", "剪刀", "布")
rps_state: Dict[tuple, dict] = {}  # (group,uid) -> {"last": ts} 待出拳中


def _rps_resolve(user_move: str, bot_move: str) -> str:
    """猜拳判定：win/draw/lose。"""
    if user_move == bot_move:
        return "draw"
    if (user_move, bot_move) in (("石头", "剪刀"), ("剪刀", "布"), ("布", "石头")):
        return "win"
    return "lose"


def _rps_can_play(profile: dict) -> bool:
    """每日限玩：跨天重置计数。"""
    today = time.strftime("%Y-%m-%d")
    if profile.get("rps_date") != today:
        profile["rps_date"] = today
        profile["rps_plays"] = 0
    return (profile.get("rps_plays") or 0) < RPS_DAILY_LIMIT


async def _handle_rps(bot, event, group_id, raw) -> None:
    """猜拳总入口：开场 + 出拳结算（固定文案，零 token）。"""
    uid = str(event.user_id)
    key = (group_id, uid)
    now = time.time()
    profile = _jiao_profile(group_id, uid)
    # 出拳超时清理
    st = rps_state.get(key)
    if st and now - st.get("last", 0) > RPS_TIMEOUT_SECONDS:
        rps_state.pop(key, None)
        st = None
    if len(rps_state) > 50:  # 兜底清理，防内存泄漏
        for k in [k for k, v in rps_state.items()
                  if now - v.get("last", 0) > RPS_TIMEOUT_SECONDS]:
            rps_state.pop(k, None)
    if raw == "猜拳":
        if st:
            await chat_matcher.send(_reply_message(event, "（挥挥拳头）人家还等着你出拳呢～"))
            return
        if not _rps_can_play(profile):
            await chat_matcher.send(_reply_message(event, "（摆手）今天玩够啦，明天再来～"))
            return
        profile["rps_plays"] = (profile.get("rps_plays") or 0) + 1
        _save_jiao()
        rps_state[key] = {"last": now}
        remain = RPS_DAILY_LIMIT - profile["rps_plays"]
        await chat_matcher.send(_reply_message(event,
            f"（握拳）来！石头剪刀布～赢人家+{RPS_WIN_GAIN}cm，平局+{RPS_DRAW_GAIN}cm，输了不扣。"
            f"出什么？（石头/剪刀/布，今日还剩{remain}次）"))
        return
    # 出拳结算
    st = rps_state.get(key)
    if not st:
        return
    rps_state.pop(key, None)
    bot_move = random.choice(RPS_MOVES)
    res = _rps_resolve(raw, bot_move)
    gain = RPS_WIN_GAIN if res == "win" else (RPS_DRAW_GAIN if res == "draw" else 0)
    if gain:
        profile["dick"] = min(JIAO_DICK_MAX, (profile.get("dick") or 0) + gain)
        _save_jiao()
    txt = {
        "win": f"（出{bot_move}）你出{raw}……（捂脸）被你赢了，吉吉+{gain}cm～",
        "draw": f"（出{bot_move}）你也出{raw}……平局，+{gain}cm安慰一下～",
        "lose": f"（出{bot_move}）你出{raw}……（得意）人家赢了！不过不扣你，再来？",
    }[res]
    await chat_matcher.send(_reply_message(event, txt))
# 会话持久化：重启后恢复进行中的游戏（防"重启丢进度"）
JIAO_SESSIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jiao_sessions.json")


def _save_jiao_sessions() -> None:
    try:
        with open(JIAO_SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump({f"{k[0]}:{k[1]}": v for k, v in jiao_sessions.items()}, f, ensure_ascii=False)
    except Exception as e:
        print(f"[JIAO] 会话保存失败: {e}", flush=True)


def _load_jiao_sessions() -> None:
    try:
        if os.path.exists(JIAO_SESSIONS_FILE):
            with open(JIAO_SESSIONS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            now = time.time()
            for key, s in data.items():
                if now - s.get("last", 0) > JIAO_TIMEOUT_SECONDS:
                    continue  # 过期会话不恢复
                gid, uid = key.split(":", 1)
                jiao_sessions[(gid, uid)] = s
            if jiao_sessions:
                print(f"[JIAO] 恢复 {len(jiao_sessions)} 个进行中的会话", flush=True)
    except Exception as e:
        print(f"[JIAO] 会话读取失败: {e}", flush=True)


_load_jiao_sessions()  # 启动时恢复进行中的会话


def _get_jiao() -> dict:
    global _jiao_cache
    if _jiao_cache is None:
        _jiao_cache = {}
        try:
            if os.path.exists(JIAO_FILE):
                with open(JIAO_FILE, encoding="utf-8") as f:
                    _jiao_cache = json.load(f)
        except Exception as e:
            print(f"[JIAO] 数据读取失败: {e}", flush=True)
    return _jiao_cache


def _save_jiao() -> None:
    try:
        with open(JIAO_FILE, "w", encoding="utf-8") as f:
            json.dump(_get_jiao(), f, ensure_ascii=False)
    except Exception as e:
        print(f"[JIAO] 保存失败: {e}", flush=True)


def _jiao_profile(group_id: str, uid: str) -> dict:
    """新教玩家档案：体力（每天自动刷新）、吉吉长度、道具、胜负。"""
    data = _get_jiao()
    p = data.setdefault(group_id, {}).setdefault(uid, {})
    p.setdefault("stamina_max", JIAO_STAMINA_MAX)
    p.setdefault("dick", JIAO_DICK_START)
    p.setdefault("wins", 0)
    p.setdefault("fails", 0)
    p.setdefault("items", {"春药": 0, "助兴酒": 0})
    today = time.strftime("%Y-%m-%d")
    if p.get("stamina_date") != today:
        p["stamina"] = p["stamina_max"]
        p["stamina_date"] = today
        _save_jiao()
    return p


def _jiao_norm(raw: str) -> str:
    """指令归一化：去掉句尾语气标点，方便识别「新教！」这类输入。"""
    return (raw or "").rstrip("！!。.？?～~ ")


def _is_jiao_cmd(raw: str) -> bool:
    raw = _jiao_norm(raw)
    if raw in ("新教", "裤子", "买体力", "买春药", "买助兴酒",
               "使用春药", "使用助兴酒", "用春药", "用助兴酒",
               "挑逗说明", "新教说明", "结束",
               # 2026-10-10：用户实测只会说「退出」——不加进来就会掉进普通回合，
               # 模型嘴上答应退出、状态机却还开着（回合一多还以为在玩）。
               "退出", "退出新教", "结束新教", "不玩了"):
        return True
    return bool(re.fullmatch(r"买体力\d+", raw))


# 中途退出新教：必须引用扫货的消息发「结束」（引用=确认，防误触退出）
JIAO_QUIT_FALLBACK = ["（撩了撩头发）行吧，这就收工～下次别半途而废哦。",
                      "（撇嘴）才玩到一半就走？行，人家记下了～",
                      "（松开手）好嘛，那就到这儿。想人家了随时来～"]


def _jiao_gains(raw: str) -> int:
    """挑逗得分：消息里出现的情欲关键词次数（封顶10）。"""
    s = 0
    for kw in JIAO_KEYWORDS:
        s += raw.count(kw)
    return min(10, s)


def _jiao_parse_location(raw: str) -> str:
    """从射精消息里解析位置；没指定就默认身上。"""
    for w in ("屁眼", "里面", "深处", "嘴", "口", "脸", "胸", "奶", "肚", "腿", "脚", "臀", "屁股", "随便"):
        if w in raw:
            return w
    return "身上"


def _jiao_new_session() -> dict:
    return {"bot": 0, "user": 0, "turns": 0, "max_turns": JIAO_MAX_TURNS,
            "user_ready": False, "user_came": False, "bot_came": False,
            "loc": "", "history": [], "last": time.time()}


def _jiao_apply_turn(s: dict, raw: str) -> tuple:
    """应用一个普通回合：增益、状态推进，返回 (阶段, 射精位置)。纯函数可单测。"""
    s["turns"] += 1
    s["last"] = time.time()
    if s["turns"] > s.get("max_turns", JIAO_MAX_TURNS):
        return "fail", ""
    g = _jiao_gains(raw)
    s["bot"] = min(100, s["bot"] + JIAO_GAIN_BOT_BASE + 3 * g + random.randint(0, 4))
    loc = ""
    u_gain = JIAO_GAIN_USER_BASE + 2 * g + random.randint(0, 4) + JIAO_LISTEN_GAIN
    if s.get("user_ready") and not s.get("user_came"):
        # 已满值：这条消息就是射精选择（位置从他的话里解析）
        loc = _jiao_parse_location(raw)
        s["user_came"] = True
        s["loc"] = loc
        u_gain = 0
    s["user"] = min(100, s["user"] + u_gain)
    if s["user"] >= 100 and not s.get("user_came"):
        s["user_ready"] = True
    if s["bot"] >= 100 and not s.get("bot_came"):
        s["bot_came"] = True
    if s.get("user_came") and s.get("bot_came"):
        return "success", loc
    if s.get("user_came"):
        return "user_cum_wait", loc
    if s.get("user_ready"):
        return "ask_where", loc
    if s.get("bot_came"):
        return "bot_cum_ask", loc
    return "normal", loc


def _jiao_session_system(s: dict, phase: str, loc: str) -> str:
    """新教会话的回合 system 文案：规则 + 当前数值 + 本回合指令。"""
    rules = (
        "【新教·高潮游戏进行中】你是主动勾引方。双方高潮值都要到100才算成功。"
        f"当前 扫货：{s.get('bot', 0)}/100，用户：{s.get('user', 0)}/100。"
        "高潮值越高，你的话要越骚、越急、越露骨。"
        f"本会话不受25字限制，尽情发挥，但单条不超过{JIAO_REPLY_LEN}字。"
        "必须正面回应，禁止输出【沉默】；实在写不出长句就写短句，但不能沉默。"
        "**禁止复读**：每回合必须即兴发挥，不能重复自己之前说过的台词和句式，"
        "尤其高潮/潮吹/射精时每次的措辞都要不一样。"
        "**每条回复末尾必须显示双方高潮值，格式：（高潮值：扫货X/100｜你Y/100）**"
    )
    if s.get("bot_came"):
        rules += "你刚刚已经潮吹了！"
    texts = {
        "normal": "正常进行：越来越骚地回应他，撩得越狠越好，每回合换个花样。",
        "ask_where": "他已经满值了！带着快感问他射在哪（选项：脸上/嘴里/胸上/肚子/里面/随便），催他快点射。",
        "bot_cum_ask": "你刚到高潮！把快感表达出来（喘息、颤抖、胡话都行），同时勾引他也快点到，措辞要新鲜。",
        "user_cum_wait": f"他刚刚射在了「{loc}」！你还差一点才到，缠着他继续服侍你，撒娇带喘。",
        "bot_tease": "他刚射了，你还差一点才到！勾引他继续弄你，直到你也高潮。",
        "success": f"你俩都高潮了（他射在了「{loc}」）！餍足地回味，并自然地说他好像又长大了一点（吉吉+1）。",
        "fail": "他没能让你俩都高潮，游戏失败！狠狠嘲讽他，说他吉吉缩短了1cm，让他不服再来。",
        "item": "他使用了道具，配合道具效果回应，反应要带新鲜感！",
    }
    return rules + "本回合：" + texts.get(phase, texts["normal"])


async def _jiao_llm(messages) -> str:
    """新教专用的 LLM 调用（长回复，max_tokens 300）。"""
    if _circuit_open():
        return ""
    kwargs = {"model": DEEPSEEK_MODEL, "messages": messages, "max_tokens": 2048}
    if DEEPSEEK_MODEL != "deepseek-reasoner":
        kwargs["temperature"] = 0.95
    if DISABLE_REASONING:
        if _REASONING_EFFORT: kwargs["reasoning_effort"] = _REASONING_EFFORT
    try:
        r = await asyncio.wait_for(client.chat.completions.create(**kwargs), timeout=150)
        _record_usage(r.usage)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[JIAO LLM ERROR] {type(e).__name__}: {e}", flush=True)
        return ""


# 模型回显 system 提示词的检测标记。实锤（2026-09-23 换 v4.1-flash 后）：
# 新教一轮里模型把 system 原文（"【新教·高潮游戏进行中】你是主动勾引方……"）当回复发进了群。
# 命中两个及以上才判定为回显，避免正常回复里提到"高潮值"格式被误杀。
JIAO_PROMPT_MARKERS = ("【新教", "你是主动勾引方", "双方高潮值都要到100",
                       "本会话不受25字限制", "格式：（高潮值", "禁止复读")


def _looks_like_prompt(text: str) -> bool:
    return sum(1 for m in JIAO_PROMPT_MARKERS if m in text) >= 2


def _jiao_meter_line(s: dict, text: str) -> str:
    """保证每条新教回复都显示双方高潮值：模型没写（或写漏）就补一行。"""
    if len(re.findall(r"\d{1,3}/100", text)) >= 2:
        return text
    return text + f"（高潮值：扫货{s.get('bot', 0)}/100｜你{s.get('user', 0)}/100）"


async def _jiao_send_turn(bot, event, group_id, s, phase, loc, key) -> None:
    """生成并发送一次新教回复；阶段含 start/item/normal/.../success/fail。"""
    uid = str(event.user_id)
    raw = event.get_plaintext().strip()
    if phase == "start":
        user_line = ""
        sys_msg = ("【新教·高潮游戏开场】他输入「新教」召唤了你，高潮游戏开始。"
                   "规则：双方高潮值都到100算成功——你到100会潮吹，他到100可以射精（可选射在哪）。"
                   "用一句淫荡的开场白迎接他，告诉他这游戏的玩法。"
                   "开场白末尾显示双方高潮值，格式：（高潮值：扫货0/100｜你0/100）。"
                   f"不受25字限制，不超过{JIAO_REPLY_LEN}字。")
    elif phase == "item":
        user_line = f"（使用了道具「{loc}」）"
        sys_msg = _jiao_session_system(s, "item", loc)
    else:
        user_line = raw[:80]
        sys_msg = _jiao_session_system(s, phase, loc)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    card_msg = _member_card_message(group_id, uid, event)
    if card_msg:
        messages.append({"role": "system", "content": card_msg})
    messages.append({"role": "system", "content": sys_msg})
    messages += s["history"][-8:]
    # 本轮玩家发言必须作为 user 消息附在末尾：history 里最后一条是上一轮的 assistant，
    # 直接以它结尾时模型会把 system 指令当成待续写内容（2026-09-23 实锤：提示词被原文发进群）。
    if user_line:
        messages.append({"role": "user", "content": user_line})

    lock = group_locks.setdefault(group_id, asyncio.Lock())
    async with lock:
        text = await _jiao_llm(messages)
        if not text or _is_silence_text(text):
            messages = messages + [{"role": "system", "content": "再试一次：直接正面回应，禁止沉默。"}]
            text = await _jiao_llm(messages)
        if not text or _is_silence_text(text):
            fb = JIAO_FALLBACK.get(phase)
            text = random.choice(fb) if isinstance(fb, list) else (fb or "（喘息）嗯……人家在等你呢～")
            print(f"[JIAO FALLBACK] phase={phase}（LLM 沉默/失败，随机兜底）", flush=True)
        if any(m in text for m in AI_MARKERS):
            text = "（歪头）人家听不懂你在说什么嘛～"
        # 防提示词泄漏：模型把 system 原文回显成回复时，改走兜底台词（绝不发进群）
        if _looks_like_prompt(text):
            print(f"[JIAO GUARD-PROMPT] 拦截提示词回显: {text[:60]!r}", flush=True)
            fb = JIAO_FALLBACK.get(phase)
            text = random.choice(fb) if isinstance(fb, list) else (fb or "（喘息）嗯……人家在等你呢～")
        text = _cut_reply(text, JIAO_REPLY_LEN)
        text = _jiao_meter_line(s, text)  # 每条回复都带上双方高潮值
        if user_line:
            s["history"].append({"role": "user", "content": user_line})
        s["history"].append({"role": "assistant", "content": text})
        s["history"] = s["history"][-8:]
        # 发送：长回复分条发（首条带引用）；_send_burst 内含失败降级，不会丢内容
        await _send_burst(bot, group_id, text, quote_event=event)
        # 记账：回复 = 互动，涨欲望 + 画像抽取
        _after_reply_bookkeeping(group_id, uid)
        # 会话状态落盘（回合数值变更；success/fail 弹出后也写）
        _save_jiao_sessions()
        # 结算
        if phase == "success":
            p = _jiao_profile(group_id, uid)
            p["dick"] = min(JIAO_DICK_MAX, (p.get("dick") or 0) + 1)
            p["wins"] = (p.get("wins") or 0) + 1
            _save_jiao()
            jiao_sage_until[key] = time.time() + JIAO_SAGE_SECONDS
            jiao_sessions.pop(key, None)
            print(f"[JIAO] 成功 {uid} 吉吉+1 贤者{JIAO_SAGE_SECONDS}s", flush=True)
        elif phase == "fail":
            p = _jiao_profile(group_id, uid)
            p["dick"] = max(0, (p.get("dick") or 0) - 1)
            p["fails"] = (p.get("fails") or 0) + 1
            _save_jiao()
            jiao_sessions.pop(key, None)
            print(f"[JIAO] 失败 {uid} 吉吉-1", flush=True)


async def _handle_jiao(bot, event, group_id, raw) -> None:
    """新教总入口：指令路由 + 会话回合处理。"""
    uid = str(event.user_id)
    key = (group_id, uid)
    now = time.time()
    print(f"[JIAO] 处理 group={group_id} user={uid} raw={raw[:40]!r}", flush=True)
    # 会话超时自动清理（没说完就跑了，不计胜负）
    s = jiao_sessions.get(key)
    if s and now - s.get("last", 0) > JIAO_TIMEOUT_SECONDS:
        jiao_sessions.pop(key, None)
        _save_jiao_sessions()
        s = None
        print(f"[JIAO] 会话超时结束: {uid}", flush=True)
    if len(jiao_sessions) > 50:  # 兜底清理，防内存泄漏
        for k in [k for k, v in jiao_sessions.items()
                  if now - v.get("last", 0) > JIAO_TIMEOUT_SECONDS]:
            jiao_sessions.pop(k, None)
    profile = _jiao_profile(group_id, uid)

    # ---------- 无会话：指令 ----------
    if not s:
        cmd = _jiao_norm(raw)
        if cmd == "新教":
            sage = jiao_sage_until.get(key, 0)
            if now < sage:
                await chat_matcher.send(_reply_message(event, f"（懒洋洋地躺着）刚贤者完……{int(sage - now) // 60}分钟后再来嘛～"))
                return
            if (profile.get("stamina") or 0) < 1:
                await chat_matcher.send(_reply_message(event, "（趴着喘气）人家没体力了……明天再来，或者花3cm吉吉买体力嘛～"))
                return
            profile["stamina"] -= 1
            _save_jiao()
            s = _jiao_new_session()
            jiao_sessions[key] = s
            _save_jiao_sessions()
            print(f"[JIAO] 开始 {uid} 体力-1 剩{profile['stamina']}", flush=True)
            await _jiao_send_turn(bot, event, group_id, s, "start", "", key)
            return
        if cmd == "裤子":
            await chat_matcher.send(_reply_message(event, _jiao_status(profile, key)))
            return
        if cmd in ("挑逗说明", "新教说明"):
            await chat_matcher.send(_reply_message(event, JIAO_HELP))
            return
        if cmd == "买体力" or re.fullmatch(r"买体力\d+", cmd):
            await _jiao_buy_stamina(event, profile, cmd)
            return
        if cmd in ("买春药", "买助兴酒"):
            await _jiao_buy_item(event, profile, cmd)
            return
        if cmd in ("使用春药", "使用助兴酒", "用春药", "用助兴酒"):
            await chat_matcher.send(_reply_message(event, "（歪头）又不是在做，用给谁看嘛～"))
            return
        if cmd == "结束":
            await chat_matcher.send(_reply_message(event, "（歪头）都没开始呢，结束什么呀～"))
            return
        return

    # ---------- 会话中 ----------
    s["last"] = now
    cmd = _jiao_norm(raw)
    if cmd == "新教":
        await chat_matcher.send(_reply_message(event, "（喘息）正忙着呢，别打断～"))
        return
    if cmd == "裤子":
        await chat_matcher.send(_reply_message(event, _jiao_status(profile, key)))
        return
    if cmd in ("挑逗说明", "新教说明"):
        await chat_matcher.send(_reply_message(event, JIAO_HELP))
        return
    if cmd == "买体力" or re.fullmatch(r"买体力\d+", cmd):
        await _jiao_buy_stamina(event, profile, cmd)
        return
    if cmd in ("买春药", "买助兴酒"):
        await _jiao_buy_item(event, profile, cmd)
        return
    if cmd in ("使用春药", "使用助兴酒", "用春药", "用助兴酒"):
        item = "春药" if "春药" in cmd else "助兴酒"
        if (profile.get("items") or {}).get(item, 0) < 1:
            await chat_matcher.send(_reply_message(event, f"（翻翻口袋）没有{item}了，买一个嘛～"))
            return
        profile["items"][item] -= 1
        _save_jiao()
        if item == "春药":
            s["bot"] = min(100, s.get("bot", 0) + JIAO_SPRING_GAIN)
        else:
            s["user"] = min(100, s.get("user", 0) + JIAO_WINE_GAIN)
        # 道具也可能直接推满：同步状态标记
        if s["bot"] >= 100 and not s.get("bot_came"):
            s["bot_came"] = True
        if s["user"] >= 100 and not s.get("user_came"):
            s["user_ready"] = True
        print(f"[JIAO] {uid} 使用{item}，扫货{s['bot']} 用户{s['user']}", flush=True)
        _save_jiao_sessions()
        await _jiao_send_turn(bot, event, group_id, s, "item", item, key)
        return

    # 中途退出，不计胜负（不扣吉吉、不进贤者）。
    # 「结束」是短词，要引用扫货的消息（防误触）；「退出/退出新教」这类说法本身
    # 就不会误说，直接退——2026-10-10 实锤：只说「退出」会被当普通回合，
    # 模型嘴上答应、会话却还开着。
    if cmd in ("结束", "退出", "退出新教", "结束新教", "不玩了"):
        if cmd == "结束" and not await _quoted_bot(event, group_id, bot):
            await chat_matcher.send(_reply_message(event, "（戳你）想收工就引用人家的话发「结束」，或者直接发「退出新教」～"))
            return
        jiao_sessions.pop(key, None)
        _save_jiao_sessions()
        print(f"[JIAO] {uid} 中途退出（指令={cmd}），不计胜负", flush=True)
        await chat_matcher.send(_reply_message(event, random.choice(JIAO_QUIT_FALLBACK)))
        return

    # 普通回合：必须引用扫货的消息才会进行（不引用不推进回合，只提醒一次）
    if not await _quoted_bot(event, group_id, bot):
        await chat_matcher.send(_reply_message(event, "（戳你）要引用人家的话，人家才陪你继续哦～"))
        return
    # 熔断不耗回合（防系统故障白烧）
    if _circuit_open():
        await chat_matcher.send(_reply_message(event, "（喘息）……人家现在脑袋晕晕的，等一下嘛～"))
        return
    phase, loc = _jiao_apply_turn(s, raw)
    await _jiao_send_turn(bot, event, group_id, s, phase, loc, key)


def _jiao_status(profile: dict, key: tuple) -> str:
    """裤子：吉吉长度 / 体力 / 道具 / 贤者剩余。"""
    items = profile.get("items") or {}
    parts = [f"（撩开裤子给你看）吉吉长度：{(profile.get('dick') or 0)}cm"
             f"｜体力：{(profile.get('stamina') or 0)}/{profile.get('stamina_max')}"
             f"｜道具：春药×{items.get('春药', 0)} 助兴酒×{items.get('助兴酒', 0)}"]
    sage = jiao_sage_until.get(key, 0)
    if time.time() < sage:
        parts.append(f"（你还在贤者时间，剩{int(sage - time.time()) // 60}分钟）")
    return "".join(parts)


async def _jiao_buy_stamina(event, profile: dict, cmd: str) -> None:
    n = 1
    m = re.fullmatch(r"买体力(\d+)", cmd)
    if m:
        n = max(1, int(m.group(1)))
    cost = n * JIAO_STAMINA_COST
    if (profile.get("dick") or 0) < cost:
        await chat_matcher.send(_reply_message(event, f"（捏了捏）{cost}cm 都不够，买不起还想买？"))
        return
    profile["dick"] -= cost
    profile["stamina"] = min(profile.get("stamina_max", JIAO_STAMINA_MAX),
                             (profile.get("stamina") or 0) + n)
    _save_jiao()
    await chat_matcher.send(_reply_message(event, f"（点了点你的吉吉）{cost}cm 没了，体力+{n}～"))


async def _jiao_buy_item(event, profile: dict, cmd: str) -> None:
    item = "春药" if "春药" in cmd else "助兴酒"
    cost = JIAO_SPRING_COST if item == "春药" else JIAO_WINE_COST
    if (profile.get("dick") or 0) < cost:
        await chat_matcher.send(_reply_message(event, f"（摇头）{cost}cm 都没有，还想买{item}？"))
        return
    profile["dick"] -= cost
    profile.setdefault("items", {}).setdefault(item, 0)
    profile["items"][item] += 1
    _save_jiao()
    await chat_matcher.send(_reply_message(event, f"（收下）{item}×1 到手，花了你 {cost}cm～"))


@chat_matcher.handle()
async def handle_group_message(bot: Bot, event: GroupMessageEvent, state: T_State) -> None:
    group_id = str(event.group_id)
    user = ""
    if event.sender:
        user = (event.sender.card or "").strip() or (event.sender.nickname or "").strip()
    user = user or "群友"
    raw = event.get_plaintext().strip()
    raw_msg = getattr(event, "raw_message", "") or ""

    is_whale = str(event.user_id) == WHALE_QQ   # 小鲸鱼是 AI 不是群友
    # 表情包收集：群友发的动画/自定义表情自动收进她的库（2026-09-24 新增）
    if raw_msg and str(event.user_id) != str(event.self_id) and not is_whale:
        try:
            n = await sticker_mod.collect_from_message(
                raw_msg, group_id, str(event.user_id), user)
            if n:
                print(f"[STICKER] 收了 {n} 个新表情，库共 {sticker_mod.sticker_count()} 个", flush=True)
        except Exception as e:
            print(f"[STICKER] 收集异常 {type(e).__name__}: {e}", flush=True)
    # 群友卡片：记住显示名 + 刷新最近活跃（群名片优先）；两只 bot 都不算群友
    if event.sender and str(event.user_id) != OTHER_BOT_QQ and not is_whale:
        _touch_member(group_id, str(event.user_id),
                      ((event.sender.card or "").strip() or (event.sender.nickname or "").strip()))

    # 记录群消息（用于滚动摘要；每条截断省token）
    if raw:
        group_recent_raw.setdefault(group_id, []).append((user, raw[:80]))
        # 群友个人发言缓冲（画像抽取素材；若叶睦不算群友）
        if str(event.user_id) != OTHER_BOT_QQ and not is_whale:
            user_msg_buf.setdefault((group_id, str(event.user_id)),
                                    collections.deque(maxlen=USER_MSG_BUF)).append(raw[:60])

    # ============ 新教 / 猜拳：指令与会话内消息在此拦截，不走普通对话 ============
    if str(event.user_id) != OTHER_BOT_QQ and not is_whale:
        rps_key = (group_id, str(event.user_id))
        norm = _jiao_norm(raw)
        rps_pending = rps_state.get(rps_key)
        if norm == "猜拳" or (rps_pending and norm in RPS_MOVES):
            await _handle_rps(bot, event, group_id, norm)
            return
        if jiao_sessions.get(rps_key) or _is_jiao_cmd(raw):
            await _handle_jiao(bot, event, group_id, raw)
            return
        # setdefault：raw 为空（纯图片/表情）时上面没建键，直接索引会 KeyError（实锤 2026-09-23）
        group_recent_raw[group_id] = group_recent_raw.setdefault(group_id, [])[-SUMMARY_INTERVAL:]
        group_msg_count[group_id] = group_msg_count.get(group_id, 0) + 1
        if ENABLE_SUMMARY and group_msg_count[group_id] >= SUMMARY_INTERVAL:
            group_msg_count[group_id] = 0
            try:
                task = asyncio.create_task(_refresh_group_summary(group_id))
                summary_tasks.add(task)          # 持引用防 GC 取消
                task.add_done_callback(summary_tasks.discard)
            except Exception as e:
                print(f"[SUMMARY SCHEDULE ERROR] {e}", flush=True)

    # 是否被 @（必定回复）
    self_id = str(event.self_id)
    at_me = any(
        seg.type == "at" and str(seg.data.get("qq", "")).strip() == self_id
        for seg in event.message
    )
    if not at_me:
        if f"at,qq={self_id}" in raw_msg or f"[CQ:at,qq={self_id}]" in raw_msg:
            at_me = True

    # 被引用检测（引用扫货本人，或引用扫货最近发出的消息；get_msg 兜底）
    quoted_me = await _quoted_bot(event, group_id, bot)

    # 提到「扫货」或别称
    mention_me = _raw_contains_name(raw, "扫货")

    triggered = at_me or quoted_me or mention_me
    # 双bot对话窗口内：对方（若叶睦）的消息视为触发，接住对话
    if str(event.user_id) == OTHER_BOT_QQ and time.time() < dialogue_until.get(group_id, 0):
        triggered = True

    # 空内容消息（纯图片/表情/语音/卡片）：不主动接话（否则白烧AI调用）；
    # 被@时用占位符进对话，让模型回应而不是空转
    if not raw:
        if not triggered:
            return
        raw = "[图片/表情]"

    print(
        f"[EVT] group={group_id} user={event.user_id} at_me={at_me} "
        f"quoted_me={quoted_me} mention_me={mention_me} raw={raw!r}",
        flush=True,
    )

    # 固定接梗：命中关键词直接撩一句
    # 固定梗只在明确叫到扫货时触发。旧逻辑会在群友彼此聊天时越过冷却直接插嘴，
    # 既显得机械，也会抢掉后面的上下文判断。
    if ENABLE_SPECIAL and triggered:
        special = _match_special(raw)
    else:
        special = None
    if special:
        print(f"[SPECIAL] group={group_id} raw={raw!r} -> {special}", flush=True)
        await chat_matcher.finish(_reply_message(event, special))
        return

    # 双bot对话指令：双方各自开窗；提到自己名字的一方回应开场
    if _is_dialogue_cmd(raw):
        dialogue_until[group_id] = time.time() + DIALOGUE_WINDOW
        dialogue_turns[group_id] = 0
        print(f"[DIALOGUE] 群{group_id} 开启双bot对话窗口 {DIALOGUE_WINDOW}s", flush=True)
        if mention_me:
            await chat_matcher.send(_reply_message(event, "（伸懒腰）嗯？聊就聊～"))
        return

    # 和小鲸鱼聊天模式：开窗后她说啥都接（含主动搭话），窗口内几乎必回
    if _is_whale_chat_cmd(raw) and str(event.user_id) != WHALE_QQ:
        whale_chat_until[group_id] = time.time() + WHALE_CHAT_WINDOW
        whale_turns[group_id] = 0
        whale_window_start[group_id] = time.time()
        print(f"[WHALE] 群{group_id} 开启小鲸鱼聊天模式 {WHALE_CHAT_WINDOW}s", flush=True)
        await chat_matcher.send(_reply_message(
            event, f"（活动了下身子）行啊，人家去逗逗那只小鲸鱼～{WHALE_CHAT_WINDOW // 60}分钟。"))
        return

    # 非触发的消息：按概率决定是否主动搭话；超过冷却则不搭
    if not triggered:
        now = time.time()
        last = last_reply_time.get(group_id, 0)
        if now - last < COOLDOWN_SECONDS:
            return
        relevance = _proactive_relevance(event, raw)
        if relevance <= 0:
            return
        # 欲望值决定“想不想理 TA”，消息相关度决定“此刻该不该插话”。
        desire = _current_member_card(group_id, str(event.user_id), event).get("desire", 0) or 0
        rate = _proactive_rate(desire, relevance)
        if random.random() >= rate:
            return
        print(f"[PROACTIVE] group={group_id} desire={desire} relevance={relevance:.2f} rate={rate:.3f}", flush=True)
        last_reply_time[group_id] = now
    else:
        last_reply_time[group_id] = time.time()

    print(f"[REPLY] group={group_id} triggered={triggered} raw={raw!r}", flush=True)

    # 同群「生成+发送」串行化：防止多人连续 @ 造成并发 LLM+发送洪峰被 NapCat 拒绝丢回复
    lock = group_locks.setdefault(group_id, asyncio.Lock())
    async with lock:
        # 维护上下文（历史条目截断，防长文粘贴撑爆输入token）
        history = contexts.setdefault(group_id, [])
        history.append({"role": "user", "content": f"{user}：{raw[:80]}"})
        history = history[-MAX_HISTORY_TURNS:]
        contexts[group_id] = history
        if len(contexts) > 60:
            for old_key in list(contexts)[:-20]:
                contexts.pop(old_key, None)

        # 识图（2026-09-24 新增）：消息里有图就交给视觉模型看一眼，把描述喂给她，
        # 这样她能"看见"群友发的图，回复才接得上（而不是把 [图片] 当空气）。
        img_note = ""
        if raw_msg and not is_whale:
            try:
                urls = _extract_image_urls(raw_msg)
                if urls:
                    desc = await asyncio.to_thread(vision_mod.describe_image, urls[0])
                    if desc:
                        img_note = f"【{user}刚发的图，你看过了】{desc}"
            except Exception as e:
                print(f"[VISION] 注入异常 {type(e).__name__}: {e}", flush=True)

        # 组装 system：人设固定为第一条 system 消息（前缀恒定，命中 API 前缀缓存省 token），
        # 动态摘要独立成第二条 system 消息，避免破坏缓存前缀
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({
            "role": "system",
            "content": (
                f"【本轮对话对象】正在说话的是「{user}」。先判断 TA 是在提问、求助、玩梗、"
                "表达情绪还是延续旧话题；准确接住具体内容后再体现扫货的性格。"
                "不要把群聊背景里的别人当成 TA，不要为了人设强行转黄。"
            ),
        })
        if img_note:
            messages.append({"role": "system", "content": img_note})
        summary = group_summaries.get(group_id, "")
        if summary:
            messages.append({"role": "system", "content": f"【群聊背景】{summary}（背景仅供参考，不要复述）"})
        if GROUP_CONTEXT_LINES > 0:
            recent = group_recent_raw.get(group_id, [])
            ctx_msgs = recent[:-1][-GROUP_CONTEXT_LINES:] if recent else []
            if ctx_msgs:
                messages.append({
                    "role": "system",
                    "content": (
                        "【群里刚才的真实对话】\n" +
                        "\n".join(f"{name}：{body}" for name, body in ctx_msgs) +
                        "\n这些只用于理解来龙去脉；真正要回应的是最后一条标有【正在回这条】的消息。"
                    ),
                })
        # 当前对话对象卡片（动态第三条 system 消息，不影响前缀缓存）：
        # 名字 + 欲望值 + 画像记忆，让模型叫名字、按欲望勾引、用记得的事撩人。
        # 小鲸鱼是 AI，不注入群友卡片（否则会给她算欲望、按撩人套路说话），改注入同行提示。
        if is_whale:
            messages.append({"role": "system", "content": WHALE_HINT})
        else:
            card_msg = _member_card_message(group_id, str(event.user_id), event)
            if card_msg:
                messages.append({"role": "system", "content": card_msg})
        # 表情包提示（2026-09-24 新增）：告诉她手里有哪些表情，想发就输出 [表情:编号]
        sticker_hint = ""
        try:
            sticker_hint = sticker_mod.pick_for_prompt()
        except Exception:
            pass
        if sticker_hint:
            messages.append({"role": "system", "content": sticker_hint})

        recent_history = [dict(item) for item in history[-SUMMARY_CONTEXT_ROUNDS:]]
        if recent_history and recent_history[-1].get("role") == "user":
            recent_history[-1]["content"] = "【正在回这条】" + recent_history[-1]["content"]
        messages += recent_history

        # 被@/被引用/被提及/双bot对话窗口：必须正面回答，沉默规则不适用
        must_answer = at_me or quoted_me or mention_me or is_whale or (
            str(event.user_id) == OTHER_BOT_QQ and time.time() < dialogue_until.get(group_id, 0))
        # 必须回答场景首次调用即注入"禁止沉默"指令：沉默后重试=整段上下文全量重发（~2.5k token），
        # 前置指令大幅降低重试概率，省 token（代价仅每次多 ~15 字输入）
        if must_answer:
            messages = messages + [{"role": "system", "content": "必须正面回答对方的话，禁止沉默。"}]
        else:
            # 2026-10-08 用户要求：主动接话要先看上下文，不要每句话硬接。
            # 这条是"可选接话"的判定指令：模型须先结合上下文判断，不值得接就沉默（零发送、省 token）。
            messages = messages + [{"role": "system", "content":
                "【可选接话】这条是群里的普通发言，不是对你说的。先结合上下文判断："
                "和你不相关、你没什么可说的、或别人正在正经聊事，就只输出四个字【沉默】；"
                "只有你确实想接、接了有意思才开口。宁可少接，不要硬接。"}]

        # token 熔断：打开期间 LLM 对话静默跳过（本地功能不受影响）
        if _circuit_open():
            print(f"[CIRCUIT] 群{group_id} LLM 对话被熔断，静默跳过", flush=True)
            return

        try:
            chat_kwargs = {
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "max_tokens": 2048,  # 允许两连发（每条≤25字），200 防截断
            }
            if DEEPSEEK_MODEL != "deepseek-reasoner":
                chat_kwargs["temperature"] = 0.95
            # 关闭思考：思考token被丢弃但仍计费，直接关掉省token
            if DISABLE_REASONING:
                if _REASONING_EFFORT: chat_kwargs["reasoning_effort"] = _REASONING_EFFORT
            reply = await asyncio.wait_for(
                client.chat.completions.create(**chat_kwargs),
                timeout=150,
            )
            _record_usage(reply.usage)
            text = (reply.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[LLM ERROR] {type(e).__name__}: {e}", flush=True)
            # 主动插话失败时静默，避免群友没叫她却收到一条报错式兜底；
            # 被明确叫到时才用人设内的短回应。
            if triggered:
                await chat_matcher.send(_reply_message(event, "（揉揉太阳穴）刚才走神了，再说一次？"))
            return

        # 必须回答场景：模型输出【沉默】或空内容时重新生成（最多2次），指示正面回答，不用兜底话术
        if must_answer:
            for _attempt in range(2):
                if text and not _is_silence_text(text):
                    break
                if _circuit_open():  # 首次调用可能已触发熔断，不再重试
                    break
                print(f"[SILENCE→RETRY] 必须回答，重试{_attempt + 1}/2: {text!r}", flush=True)
                chat_kwargs["messages"] = messages + [
                    {"role": "system", "content": "再试一次：直接正面回答对方的话，不要沉默。"}
                ]
                reply = await asyncio.wait_for(
                    client.chat.completions.create(**chat_kwargs),
                    timeout=150,
                )
                _record_usage(reply.usage)
                text = (reply.choices[0].message.content or "").strip()
            if not text or _is_silence_text(text):
                # 重试仍沉默/为空（极小概率）：最简正面回应，绝不用套话
                print(f"[SILENCE→FINAL] 重试仍沉默，最简回应: {text!r}", flush=True)
                text = "……嗯，我在。"
        else:
            if not text:
                print("[LLM EMPTY] DeepSeek 返回空", flush=True)
                return
            # 纯概率主动搭话：允许沉默（宁沉默，不硬撩）
            if _is_silence_text(text):
                print(f"[SILENCE] 模型选择沉默: {text!r}", flush=True)
                return

        # 防露馅：拦截 AI 味词汇
        if any(m in text for m in AI_MARKERS):
            print(f"[GUARD-AI] 拦截AI味回复: {text!r}", flush=True)
            text = "（歪头）人家听不懂你在说什么嘛～"

        # 表情标记处理（2026-09-24 新增）：她可能在回复里写 [表情:编号]。
        # 拆出来 → 先发文字（若有）→ 再发图片。表情单独发也支持。
        sticker_path, text = _split_sticker_mark(text)

        # 鲸鱼娘式连发（2026-09-29）：换行 = 分两条连着发，每条仍硬切 25 字
        vlines = [ln.strip() for ln in text.splitlines() if ln.strip()][:2]
        vlines = [_cut_reply(ln, MAX_REPLY_LEN) for ln in vlines]
        text = "\n".join(vlines)

        history.append({"role": "assistant", "content": text})
        contexts[group_id] = history

        # 欲望值：扫货回复了谁，就对谁涨欲望（若叶睦不算群友）；每 N 次互动后台抽一次画像
        _after_reply_bookkeeping(group_id, str(event.user_id))

        # 发送并记录消息 id（用于后续「被引用」检测）；多条时逐条发，中间隔 1~2 秒
        for j, ln in enumerate(vlines):
            if ln:
                await _send_and_record(bot, event, group_id, ln)
            if j < len(vlines) - 1:
                await asyncio.sleep(random.uniform(1.2, 2.4))
        if sticker_path:
            await _send_sticker_file(group_id, sticker_path, event)

# ============ 私信：只应答，绝不主动私发（2026-10-07 开启） ============
def _check_private_trigger(event, state: T_State) -> bool:
    """私信触发：总开关开启、仅私聊、非自己、非另一只 bot（防 bot 互私聊）。"""
    if not ENABLE_SPEAKING:
        return False
    if not isinstance(event, PrivateMessageEvent):
        return False
    if str(event.user_id) == str(event.self_id):
        return False
    if str(event.user_id) in {OTHER_BOT_QQ, ASTRA_BOT_QQ}:
        return False
    return True


private_matcher = on_message(rule=Rule(_check_private_trigger), priority=10, block=False)


@private_matcher.handle()
async def handle_private_message(bot: Bot, event: PrivateMessageEvent, state: T_State) -> None:
    """私信应答：有人私聊才回，全文件唯一的私信发送点，绝不主动发首条私信。"""
    uid = str(event.user_id)
    raw = event.get_plaintext().strip()
    if not raw:
        return                      # 纯图/表情等非文字私信不回
    user = ""
    if event.sender:
        user = (event.sender.card or "").strip() or (event.sender.nickname or "").strip()
    user = user or "陌生人"

    # token 熔断：熔断期间私信静默
    if _circuit_open():
        print("[CIRCUIT][PRIVATE] 私信 LLM 被熔断，静默", flush=True)
        return

    hist_key = f"private:{uid}"
    history = contexts.setdefault(hist_key, [])
    history.append({"role": "user", "content": f"{user}：{raw[:80]}"})
    history = history[-MAX_HISTORY_TURNS:]
    contexts[hist_key] = history
    if len(contexts) > 60:
        for old_key in list(contexts)[:-20]:
            contexts.pop(old_key, None)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-8:]
    try:
        chat_kwargs = {"model": DEEPSEEK_MODEL, "messages": messages, "max_tokens": 200}
        if DEEPSEEK_MODEL != "deepseek-reasoner":
            chat_kwargs["temperature"] = 0.95
        if DISABLE_REASONING:
            if _REASONING_EFFORT:
                chat_kwargs["reasoning_effort"] = _REASONING_EFFORT
        reply = await asyncio.wait_for(client.chat.completions.create(**chat_kwargs), timeout=150)
        _record_usage(reply.usage)
        text = (reply.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[LLM ERROR][PRIVATE] {type(e).__name__}: {e}", flush=True)
        await private_matcher.finish("（扫货走神了……）嗯？人家刚才没听清～")
        return

    if not text or _is_silence_text(text):
        # 私聊是被直接找上门的：正面接住，不沉默
        text = "嗯？说吧，人家听着呢～"
    if any(m in text for m in AI_MARKERS):
        print(f"[GUARD-AI][PRIVATE] 拦截AI味回复: {text!r}", flush=True)
        text = "（歪头）人家听不懂你在说什么嘛～"

    # 两连发（同群聊风格）：换行 = 分两条，每条硬切 25 字
    vlines = [ln.strip() for ln in text.splitlines() if ln.strip()][:2]
    vlines = [_cut_reply(ln, MAX_REPLY_LEN) for ln in vlines]

    history.append({"role": "assistant", "content": "\n".join(vlines)})
    contexts[hist_key] = history

    for j, ln in enumerate(vlines):
        if ln:
            try:
                await bot.send_private_msg(user_id=event.user_id,
                                           message=Message(MessageSegment.text(ln)))
            except Exception as e:
                print(f"[SEND ERROR][PRIVATE] {type(e).__name__}: {e}", flush=True)
            if j < len(vlines) - 1:
                await asyncio.sleep(random.uniform(1.2, 2.4))
    print(f"[PRIVATE] user={uid} text={' / '.join(vlines)!r}", flush=True)
