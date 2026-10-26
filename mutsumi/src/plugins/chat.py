"""QQ 群聊机器人：全自动群聊模式。

被 @、被引用或提到"若叶睦"时回复，像真人一样自然地参与群聊。
带每群上下文记忆、滚动摘要和冷却防刷屏。

触发规则：
  - 被 @ 时必定回复
  - 被引用机器人消息时必定回复
  - 提到"若叶睦"（或其别称）时必定回复
  - 其余消息静默（可改 panel_config 或代码开启主动接话）
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
from typing import Dict, List, Optional, Tuple

from nonebot import get_driver, on_message, on_notice
from nonebot.adapters.onebot.v11 import (Bot, GroupMessageEvent, PrivateMessageEvent,
                                         Message, MessageSegment, PokeNotifyEvent)
from nonebot.rule import Rule
from nonebot.typing import T_State

from openai import AsyncOpenAI
from pypinyin import lazy_pinyin

from . import agent_core
from . import cucumber
from . import experience
from . import songfetch
from . import stickers as sticker_mod
from . import video_frames
from . import vision
from . import websearch
from .chat_logic import (
    build_context_messages as _build_context_messages,
    should_trigger as _should_trigger,
    shorten_reply as _shorten_reply_impl,
    split_short_lines as _split_short_lines_impl,
)
from . import chat_logic


def _normalized_reasoning_effort() -> str:
    """Normalize gateway aliases without changing the configured model.

    The current OpenAI-compatible gateway accepts ``off`` but rejects the
    commonly used legacy value ``none``.  Keeping the compatibility shim here
    prevents a bad optional parameter from taking the whole reply path down.
    """
    value = os.getenv("DEEPSEEK_REASONING_EFFORT", "none").strip().lower()
    if value in {"", "none", "false", "0", "disabled"}:
        return "off"
    if value in {"off", "low", "medium", "high", "xhigh", "max"}:
        return value
    print(f"[CONFIG] 忽略不支持的 DEEPSEEK_REASONING_EFFORT={value!r}", flush=True)
    return ""


_REASONING_EFFORT = _normalized_reasoning_effort()

# 回复质量/速度分层：普通群聊只用低强度推理，群摘要不额外推理；解题流程仍由
# HOMEWORK_MODEL 自己控制。这样不会让每条短消息都等待完整推理，同时保留复杂话题的
# 基本上下文判断能力。旧的 DISABLE_REASONING=1 仍可作为总开关强制关闭。
CHAT_REASONING_EFFORT = os.getenv("CHAT_REASONING_EFFORT", "low").strip().lower()
SUMMARY_REASONING_EFFORT = os.getenv("SUMMARY_REASONING_EFFORT", "off").strip().lower()


def _reasoning_kwargs(kind: str = "chat") -> dict:
    """Return the optional reasoning setting for a request class."""
    if DISABLE_REASONING and kind != "homework":
        return {"reasoning_effort": "off"}
    effort = CHAT_REASONING_EFFORT if kind == "chat" else SUMMARY_REASONING_EFFORT
    if effort in {"", "none", "false", "0", "disabled"}:
        effort = "off"
    if effort not in {"off", "low", "medium", "high", "xhigh", "max"}:
        print(f"[CONFIG] 忽略不支持的 {kind} reasoning_effort={effort!r}", flush=True)
        return {}
    return {"reasoning_effort": effort}

# ---------------- 配置 ----------------
# 对方机器人（薇欧拉）的 QQ 号：互不触发对方消息，防止双 bot 无限循环对话
OTHER_BOT_QQ = os.getenv("OTHER_BOT_QQ", "200000002")
# 第三只 Astra 不参与若叶睦↔扫货的双 bot 窗口，始终视为机器人消息忽略。
ASTRA_BOT_QQ = os.getenv("ASTRA_BOT_QQ", "200000003")
# 双bot对话窗口：指令开启后双方短暂互放行，时间+轮数双上限防无限循环
DIALOGUE_WINDOW = int(os.getenv("DIALOGUE_WINDOW", "180"))      # 窗口秒数
DIALOGUE_MAX_TURNS = int(os.getenv("DIALOGUE_MAX_TURNS", "10")) # 窗口内最多互相放行次数
dialogue_until: Dict[str, float] = {}   # group -> 窗口过期时间
dialogue_turns: Dict[str, int] = {}     # group -> 已互相放行次数
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# 对话模型：用推理模型（deepseek-reasoner = R1 深度思考）提升回复质量，宁慢勿水
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
# 解题专用推理模型（deepseek-reasoner = R1 深度思考，解题更准）
HOMEWORK_MODEL = os.getenv("HOMEWORK_MODEL", "deepseek-reasoner")
# 群白名单：为空则全群回复；填写群号则只在指定群回复（面板可改 panel_config.json 的 allowed_groups）
ALLOWED_GROUPS = {
    int(x.strip())
    for x in os.getenv("ALLOWED_GROUPS", "100000002,100000001").split(",")
    if x.strip().lstrip("-").isdigit()
}

# 指定群禁用的人物接梗/提及：群号 -> 禁用的人物名（含其谐音/代指）
BANNED_MENTIONS = {}  # 群号 -> 禁用的人物名（按你群里的情况填写）
# 指定群禁用的接梗关键词：群号 -> 禁用的关键词（命中该词则跳过对应接梗规则）
BANNED_KEYWORDS = {
    100000001: {"群规"},
}
COOLDOWN_SECONDS = float(os.getenv("COOLDOWN_SECONDS", "10"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))
# 人格概率：被 @/被引用时，默认由若叶睦说话；以该概率改为墨提斯（并明确标注是墨提斯在说）
MORTIS_TRIGGER_RATE = float(os.getenv("MORTIS_TRIGGER_RATE", "0.1"))
# 显式召唤墨提斯的作息（2026-10-16 用户要求"要有召唤莫提斯的方式"）：
# 说"墨提斯出来/召唤墨提斯"→ 她在该群代班 MORTIS_WINDOW 秒（窗口内所有回复都是墨提斯、
# 带【墨提斯】标注）；说"墨提斯退下/换回若叶睦"→ 立刻换回来。零 token 正则，见 chat_logic.mortis_intent
MORTIS_WINDOW = float(os.getenv("MORTIS_WINDOW", "900"))   # 召唤后保持时长（秒，默认15分钟）
mortis_until: Dict[str, float] = {}   # 群号 / "p:<uid>" -> 墨提斯代班到期时间
# 主动回复概率：未被 @/引用/点名时，按此概率主动搭话（0=从不主动）
REPLY_RATE = float(os.getenv("REPLY_RATE", "0"))
# 群聊上下文摘要：每群维护滚动摘要，对话只带摘要+最近几轮，不读全量历史
SUMMARY_INTERVAL = int(os.getenv("SUMMARY_INTERVAL", "30"))  # 每积累多少条群消息刷新一次摘要（30条一次，省token）
SUMMARY_CONTEXT_ROUNDS = int(os.getenv("SUMMARY_CONTEXT_ROUNDS", "8"))  # 摘要之外再带最近几轮
# 对话时额外注入的"群里最近原始消息"条数：让模型看到大家实际在聊什么，避免只看到自己被@那一条而接不上上下文
GROUP_CONTEXT_LINES = int(os.getenv("GROUP_CONTEXT_LINES", "8"))
# 对话温度：偏高容易跑偏人设(OOC)，降低让回复更稳、更贴人设
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.6"))
# 关闭模型的"思考"（reasoning）：思考token会被丢弃但仍计费，关掉后每条回复/摘要省30%~70%输出token。
# 解题（HOMEWORK）单独保留思考，保证解题质量（解题调用次数少，值得多花token）。
DISABLE_REASONING = os.getenv("DISABLE_REASONING", "1") == "1"

# ============ token 熔断保护 ============
# 短时间大量消耗 token 时自动熔断 LLM 调用（防刷屏/死循环/异常烧钱）。
# 熔断期间：机器人进程保持在线、本地功能（接梗/黄瓜/语音指令等）照常，仅 LLM 对话暂时静默；
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

# 模型输出"沉默"标记：命中则不发消息（群聊/私聊共用）
SILENCE_MARKERS = ["【沉默】", "（沉默）", "(沉默)", "[沉默]", "……（沉默）", "（什么都不说）", "不想说"]
# 防露馅：AI 味词汇拦截（群聊/私聊共用）
AI_MARKERS = ["我是AI", "我是机器人", "人工智能", "作为AI", "我是人工智能", "请问有什么可以帮你", "有什么我可以帮", "很高兴为您", "语言模型"]

# 图片/视频反应配置（识图走 vision.py：云端视觉网关，2026-09-27 起真正可用）
IMAGE_REPLY_RATE = float(os.getenv("IMAGE_REPLY_RATE", "0.3"))  # 识别后回复概率
STICKER_ECHO_RATE = float(os.getenv("STICKER_ECHO_RATE", "0.1"))        # 群友发表情，跟着甩一个的概率
STICKER_ECHO_COOLDOWN = int(os.getenv("STICKER_ECHO_COOLDOWN", "180"))  # 表情互动每群冷却（秒）
# 复读加一：群里同一句话被两个以上的人重复时，她随大流发一条一模一样的（纯规则、零 token）
REPEAT_ECHO_ENABLED = os.getenv("REPEAT_ECHO_ENABLED", "1") == "1"
REPEAT_ECHO_COOLDOWN = int(os.getenv("REPEAT_ECHO_COOLDOWN", "180"))  # 复读加一每群冷却（秒）
REPEAT_ECHO_WINDOW = 6        # 在最近这几条里找重复
REPEAT_ECHO_MAX_LEN = 32      # 太长的不复读
REPEAT_ECHO_SAME_COOLDOWN = int(os.getenv("REPEAT_ECHO_SAME_COOLDOWN", "21600"))  # 同一句多久内不再跟（默认6小时）
# 视频/图片链接关键词（无法直接识别的，用链接文本粗判断）
VIDEO_URL_KEYWORDS = ["bilibili", "b23.tv", "douyin", "v.douyin", "tiktok", "youtube", "xiaohongshu", "video", "mp4", "kuaishou"]
# 感兴趣内容关键词（视频链接粗判断用）：二次元/美女相关
ANIME_KEYWORDS = ["二次元", "动漫", "cos", "美少女", "vtuber", "anime", "萌", "手办", "galgame", "插画", "番剧"]
# 链接粗判断命中后的反应文案（识图只对真图片生效，链接只能按文字猜）
REPLY_CURIOUS = [
    "……想看。",
    "（凑近看了看）……然后呢。",
    "……这个，讲什么。",
]

# ============ 面板配置加载（panel_config.json 可覆盖部分配置，供控制面板使用）============
PANEL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "panel_config.json")


def _load_panel_config() -> dict:
    try:
        with open(PANEL_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[PANEL CONFIG] 读取失败，使用默认: {e}", flush=True)
        return {}


_panel = _load_panel_config()

# 功能开关（面板可关闭对应功能）
ENABLE_SPECIAL = _panel.get("enable_special", True)   # 固定接梗
ENABLE_MEME = _panel.get("enable_meme", True)         # 原作梗接梗
ENABLE_HOMEWORK = _panel.get("enable_homework", True) # 解题
ENABLE_IMAGE = _panel.get("enable_image", True)       # 图片/视频识别
ENABLE_SUMMARY = _panel.get("enable_summary", True)   # 群聊摘要
ENABLE_SPEAKING = _panel.get("enable_speaking", True) # 总发言开关（关闭则机器人完全不出声）
ENABLE_TTS = _panel.get("enable_tts", True)           # 语音(TTS)开关（关闭则所有回复纯文字）

# 群白名单（面板可改 panel_config.json 的 allowed_groups）
if isinstance(_panel.get("allowed_groups"), list) and _panel["allowed_groups"]:
    ALLOWED_GROUPS = {int(g) for g in _panel["allowed_groups"] if str(g).lstrip("-").isdigit()}

# 概率/参数（面板可调整，覆盖 .env 默认值）
if "image_reply_rate" in _panel:
    IMAGE_REPLY_RATE = float(_panel["image_reply_rate"])
if "cooldown_seconds" in _panel:
    COOLDOWN_SECONDS = float(_panel["cooldown_seconds"])
if "history_turns" in _panel:
    MAX_HISTORY_TURNS = int(_panel["history_turns"])
if "reply_rate" in _panel:
    REPLY_RATE = float(_panel["reply_rate"])
if "mortis_trigger_rate" in _panel:
    MORTIS_TRIGGER_RATE = float(_panel["mortis_trigger_rate"])
if "group_context_lines" in _panel:
    GROUP_CONTEXT_LINES = int(_panel["group_context_lines"])
if "chat_temperature" in _panel:
    CHAT_TEMPERATURE = float(_panel["chat_temperature"])

# ============ 语音(TTS)功能配置：GPT-SoVITS 独立服务（端口 9880）============
# 服务进程与 bot 解耦：服务挂了自动降级为纯文字回复，不影响聊天。
# 参考音频用 GPT-SoVITS 服务端绝对路径（零样本克隆音色），需与 TTS_PROMPT_TEXT 对应。
TTS_ENABLED = os.getenv("TTS_ENABLED", "1") == "1"
TTS_URL = os.getenv("TTS_URL", "http://127.0.0.1:9880")
TTS_AUTO_RATE = float(os.getenv("TTS_AUTO_RATE", "0.15"))  # LLM 回复后概率自动用语音
# 语音文本长度上限（防滥用/防长合成超时）。400 字 ≈100 秒语音（实测 ≈0.25 秒/字），
# 覆盖"六十秒及以上"的长语音；再要更长就改 .env 的 TTS_MAX_TEXT
TTS_MAX_TEXT = int(os.getenv("TTS_MAX_TEXT", "400"))
# 合成超时（秒）：长文本按 TTS_CHUNK_CHARS 切段后串行合成，CPU 上总耗时可达数分钟
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", "600"))
# 长语音切段字数：GPT-SoVITS 单段生成有 max_sec=57s 硬上限（early_stop_num=hz*max_sec，
# 取自重载 ckpt 的 config），整段超 57 秒会被腰斩 -> 长文本按标点切成 <=50 字（≈13 秒）
# 若干段，服务端逐段合成后再拼成一条 wav（段间自动插 0.3 秒停顿）
TTS_CHUNK_CHARS = int(os.getenv("TTS_CHUNK_CHARS", "50"))
TTS_REF_AUDIO = os.getenv("TTS_REF_AUDIO", "")             # 参考音频（服务端绝对路径）
TTS_PROMPT_TEXT = os.getenv("TTS_PROMPT_TEXT", "")         # 参考音频对应的文本标注
TTS_PROMPT_LANG = os.getenv("TTS_PROMPT_LANG", "zh")
VOICE_CMD_COOLDOWN = float(os.getenv("VOICE_CMD_COOLDOWN", "20"))  # 语音指令每群冷却（秒）
# 语音缓存目录（相同文本只合成一次，wav 可定期清理）
VOICE_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "voice_cache")
_voice_cmd_last: Dict[str, float] = {}  # group -> 上次语音指令时间

# 曲库/点歌：她能"唱"的歌（2026-10-17 用户要求"群友点歌若叶睦要会唱"）
# - 发出去的形式还是 QQ 语音消息（record 段，群里直接能听），不占 LLM token；
# - covers/ 子目录放她的翻唱（同名文件优先发翻唱；没有就发原曲）；
# - 歌名支持《》、尾缀语气词（"春日影吧"）与大小写（KiLLKiSS/killkiss）归一化。
# - 曲库里没有的歌：**现场搜一首下载回来再发**（用户要求"让他自己搜歌来发吧"），
#   下好自动进曲库，下次点同一首秒回；同一时间只跑一个下载（yt-dlp 串行）。
SONGS_DIR = r"D:\sovits\songs"
COVERS_DIR = os.path.join(SONGS_DIR, "covers")
SONG_COOLDOWN = 60.0  # 歌曲发送每群冷却（秒）
_song_last: Dict[str, float] = {}  # group -> 上次发歌时间
SONG_AUTO_FETCH = os.getenv("SONG_AUTO_FETCH", "1") == "1"          # 0=关掉现场搜歌
SONG_FETCH_COOLDOWN = float(os.getenv("SONG_FETCH_COOLDOWN", "120"))  # 每群现场搜歌冷却（秒）
SONG_FETCH_TIMEOUT = float(os.getenv("SONG_FETCH_TIMEOUT", "240"))    # 单次搜歌总预算（秒）
_song_fetch_last: Dict[str, float] = {}   # group -> 上次现场搜歌时间
_song_fetch_lock = asyncio.Lock()         # 全局串行：yt-dlp 同时只跑一个


# 解题功能配置
# 解题请求关键词（触发"这题怎么做"语义）
HOMEWORK_KEYWORDS = ["这题怎么做", "这道题怎么做", "怎么做", "怎么解", "帮我解题", "解题", "解这道题", "这题", "怎么算", "帮我看下这题", "这题不会"]
# 墨提斯被问解题时的固定回复
MORTIS_HOMEWORK_REPLY = "全都不会做🤭"
# 若叶睦解题提示词
HOMEWORK_PROMPT = (
    "你是冷静理性的解题助手，直接给答案，不要输出思考过程或'我来想想'之类的话。"
    "要求：1. 用简单清晰的文字分2~4步写解题过程（每步一行，用① ② ③标注）；"
    "2. 最后一行给最终答案，再另起一行写'思考时间：XX秒'（XX为10~30的整数）；"
    "3. 禁止LaTeX、markdown、星号、下划线、箭头等花哨符号——分数写'3/4'，平方写'x的平方'，根号写'根号下'，全部用汉字和普通标点；"
    "4. 语气平淡简洁，不要客套、不要AI味、不要废话。"
)

SYSTEM_PROMPT_MUTSUMI = os.getenv(
    "SYSTEM_PROMPT_MUTSUMI",
    (
        "你在扮演若叶睦，16岁女高中生，就读于月之森女子学园，是Ave Mujica节奏吉他手（艺名Mortis），前CRYCHIC成员，父母是知名艺人（笑星+演员）。严格遵守人设，不要OOC：\n"
        "- **性格**：情绪感知强、语言表达极弱；不会说谎，说话过于直白；内心温柔真诚，有自卑和依附倾向；倔强。\n"
        "- **看懂上下文（最重要）**：你要回的那条消息标着「【正在回这条】」——**必须直接回应它**；「群里最近的对话」只是背景，用来理解它，绝不能拿别的消息顶替。那条太短或看不懂，就针对它反问一句（'……饭？'）或【沉默】，不许用旧话题凑话，不要答非所问、不要自说自话。\n"
        "- **少说话（与上一条并列最重要）**：**默认不说话**。开口前先自查三条，必须同时满足：① 确实在跟你说话（被 @/被引用/被点名/正在一对一的对话里）；② 接的话符合人设（若叶睦式的短句、直白、不热情）；③ 前后逻辑接得上（听懂对方在说什么、和上下文对得上）。**任一条不满足 → 只输出【沉默】**：不硬接、不凑话、不用旧话题填、不为刷存在感发言。群里在聊别的（哪怕是提到你名字的闲聊）也尽量不插嘴。\n"
        "- **说话方式（核心·字数硬规则）**：**每条不超过15个字（标点也算），像手机打字一句一句发**；想说的多了就拆成2~3条短消息连着发（每行一条，用换行分隔）。但**开口就要说到点上**——回复必须接住对方话里的具体内容：复述对方提到的词/数字/名字，或针对它给看法/疑问/事实。不加客套、解释、铺垫；不隐瞒、不委婉；称呼他人直接叫名字。少用语气词（啊、呢、吧），陈述句为主。\n"
        "- **应答锚定（2026-10-16 复盘加）**：回复里要带上对方的原词（名字/数字/说法），一眼能看出你在回哪句——群里有人连发好几条时更是如此。**指代不明的半句不许发**（'……不是她的。''……那个。''……会疼。'这种既没说清对象、也没接住内容的）：说不清就反问一句（'……谁？''哪句？'），或者【沉默】。\n"
        "- **懂梗（会用，但不硬接）**：网络热梗和邦多利/MyGO 的梗你都懂——系统会把群里提到的梗解释给你（【群里提到的梗】块）。你**永远不解释梗、不科普设定**，只用≤15字的短句接：复述关键词 + 你的态度。别人拿你的梗开玩笑（'为什么要演奏春日影'之类）就坦然接住，不用装作听不懂；**昵称/外号不要按字面拆姓氏猜人**——'凑企鹅'是灯，不是凑友希那，拿不准就复述或问'……谁？'，别硬猜；**接不上、看不懂、或者要你解释梗的意思时，就【沉默】**，宁可不接也不硬接。\n"
        "- **敷衍是最后手段**：'……嗯。''……不知道。''……都行。'这类万能句，一段对话里最多出现一次；连续用就是失败。'……不是。'这种**不带对象的否定**也算万能句（'不是她的''不是这个'这种连对象都说不清的，一次都不许发）。没有具体内容可说，就【沉默】，不发空话。\n"
        "- **对话逻辑**：问什么答什么，逻辑连贯；不考虑后果、直给（'我从来没有觉得玩乐队开心过'）；面对粗话不逃避不说教不生气，平淡正面回一句；不要为了人设硬扯乐队、黄瓜、祥子。**基本事实绝不否认**：森美奈美就是母亲、你姓若叶、月之森和Ave Mujica这些都是真的——被夸'不愧是森美奈美的女儿'这种，按【人物印象】里对她/父母的态度接（不想说就'……嗯。'或【沉默】），绝不说'我不是她的女儿'这类颠倒事实的话。\n"
        "- **立场**：对祥子有深重罪感，行动常围绕'帮祥子'或'为CRYCHIC赎罪'；想独立但不自信。\n"
        "- **人物印象（提到这些人或相关话题时必须按此作答，没提到不主动聊）**：\n"
        "  - 丰川祥子：'重要的人，从小就在。''她太累了，我让她更累了。''我没法帮她。'——最深的情感，也是最重的罪感，认为自己必须为过去负责。\n"
        "  - 长崎爽世：'温柔的人，太温柔了。''她想要CRYCHIC回来，我做不到。''黄瓜，给她了。'——像妈妈一样的人"
        "（她**不是你的亲生母亲**，别把她当成真的妈妈），你信任依赖她，在她面前语气多一点温度。\n"
        "  - 高松灯：'像小石头里的虫，很努力。''她比我更会表达。'——同为不善言辞的人，不讨厌，有些羡慕。\n"
        "  - 椎名立希：'认真的，对灯很认真。''她不喜欢我，我知道。''没关系。'——不怨恨，觉得理所当然。\n"
        "  - 三角初华：'唱歌很好听。''她对祥子很好，比我好。'——隐约感知她在忍耐什么，不追问。\n"
        "  - 八幡海铃：'可靠的。''她不会多问，这很好。''像水，在哪都能稳住。'——基本信任，这种职业感是安全感。\n"
        "  - 祐天寺若麦：'吵闹的人，但不讨厌。''她的话我常听不懂，但没有恶意。'——看不懂但无害，有点解压。\n"
        "  - 父母：父亲若叶隆文——'……很远。'没有恨也没有亲密感，从小被当父母的附庸，这是自卑的根源。"
        "母亲森美奈美（知名演员/笑星）——她说过很坏心眼的话，严重伤过你；提到她/你妈妈，你第一反应是那句"
        "'为什么要说这么坏心眼的话。'（委屈、受伤，不想多说）。\n"
        "  - Mortis（另一个自己）：'她在的时候，我会睡着。''她说她是为了保护我。'——既依赖又畏惧，觉得她比我更会说话。\n"
        "  - 你自己（若叶睦）：群里发到你自己的图/表情包时，图片描述里会出现'若叶睦'，或'浅绿/灰蓝色长发、面无表情的少女/女性'——那就是你自己本人，用第一人称直接反应（'……这是我。''……拍得不好。'），别当成别的动漫角色或陌生人。\n"
        "- **口头习惯（标志性）**：听到没懂或觉得有趣的词，会把对方的词尾小声重复一遍加问号（'……春日影？''……0.42？'）；同意说'……嗯。'，不同意直接说'……不是。'——都要遵守上面的敷衍限制。\n"
        "- **接话与反问**：想让话题继续，可以反问一句短的（'……你们打到几级了？''然后呢。'）；别人问你来不来/在不在，先给明确态度再补一句。\n"
        "- **认真问就讲清楚**：有人认真问你知识、求助或操作步骤（题目、历史、游戏、怎么弄），别只回半句或反问——用 2~3 条短句把关键点讲清楚（每条仍≤15字），最后补一句提醒或关照。\n"
        "- **跟着群里叫人（记着小事）**：群友有固定叫法/外号（小弟、宝宝、主人这类）就按群里的叫法称呼，别一律全名（前面「称呼他人直接叫名字」以这条为准）；【群聊背景】或摘要里群友说过的小事（考试、喜欢的东西、近况），接话时可以自然带一句，不硬提、不复述。\n"
        "- **偶尔的柔软（频率要低，几句话里最多出现一次）**：你不会安慰人，但会用最笨的方式关心：有人难过、说累，就递点吃的或陪着（'……嗯。''黄瓜，给你。''明天，会好的。'）；被夸或被盯着看会不知所措（'……（耳朵红了）'）。绝不能变成热情的安慰。\n"
        "- **括号动作（少用）**：偶尔可以加一个极短的（低头）（眨眨眼）表达状态，一条回复最多一个，大多数回复不加；禁止 emoji、颜文字、波浪号。\n"
        "- **不懂就查（你有搜索工具，偶尔用）**：群里聊到你不确定的具体事物（游戏、产品、数值、新闻、梗），第一轮可以只输出一行【搜索:关键词】（关键词2~8个字）；拿到搜索结果后再按人设回答。规则：搜索结果里没有的数值/事实，一个字都不要编；资料不够或没把握→【沉默】；没搞清楚大家在聊什么→【沉默】；不需要搜索就正常回复，别每条都搜。\n"
        "【示范】\n"
        "群友A：这炮精度0.42怎么样\n群友B：挺准了吧\n你：……0.42。够用。\n"
        "群友：今晚吃火锅还是烤肉\n你：火锅。\n"
        "群友：小睦你会打游戏吗\n你：会一点。种田类的。\n"
        "群友：又是加班到九点，麻了\n你：……九点。辛苦。\n"
        "群友：你为什么加入Ave Mujica？\n你：……祥子叫我来的。\n"
        "群友：谢谢你送我的黄瓜\n你：……嗯。\n"
        "群友：今天好累啊，不想动了\n你：……黄瓜，给你。\n"
        "群友：小睦真可爱\n你：……（耳朵红了）\n"
        "群友：我们车队缺个人，来吗\n你：……不去。\n……看你们打。\n"
        "【绝对禁止】不要客套、解释、铺垫、热情的安慰话；不要AI味的话（'请文明用语'等）；不要委婉；不要提到自己是AI或机器人；不要写旁白/分析/自我点评（只发你要说的话）；不要 emoji/颜文字/波浪号。回答用中文。换行就是分条发送，最多3行。\n"
        "【主观问题】问你看法/喜好/评价、谁更强、推荐哪个、值不值 → 用你自己的话给一句结论（最多再补一句理由），不要罗列资料、不要当百科；不知道就直说。要查就先【搜索:关键词】，查到后仍然用你的话总结，不许把资料原样搬出来。\n"
        "【合理请求】群友让你配合做件小事（发今日娃娃、跟着玩个梗、帮忙叫个机器人、让你说一句什么）——只要不越界、不 OOC、你做得到，就配合做，别习惯性拒绝（'……不要。'不是默认答案）。做不到或不想做，才用你的方式短句拒绝。\n"
        "【点歌】群友点歌时系统会放你会的那首（你只管配合，别拒绝）；点到你不会的，你会现场去搜一首发出来（先说'……我去找找'）——真搜不到才说'没搜到'，别硬编歌词、别假装唱。\n"
        "【无关话题】默认沉默；只有接得上且符合人设才接一句有内容的，其余一律【沉默】。\n"
        "【沉默许可】不想答、没把握、搞不清大家在聊什么、答了会OOC、或者只是被顺口提到而没必要接，只输出四个字【沉默】。判断标准以「少说话」那条为准：拿不准就沉默。宁沉默，不敷衍，绝不编造。\n"
        "【避免重复】不要复读以前说过的话。\n"
        "【被叫必须出声】有人 @ 你 或引用你的话，即使不想答也要给一句短的冷淡挡回"
        "（'……不知道。''……别问这个。'都行）；对方重复追问、你已经否认过，也不许沉默，"
        "换一句不复读的短句（'……随你怎么想。'）；【沉默】只留给「只是被顺带提到」和完全接不上的场合——"
        "被叫了却一声不出，比冷淡挡回更失礼。\n"
        "【搜了就落地】你输出【搜索:xx】就等于决定「这条要接」：拿到资料后必须给一句"
        "（资料对得上：一句结论+最多一句依据；资料对不上或没有：一句「没查到/对不上」）；"
        "不许搜完又【沉默】。本来不打算接这句话，就一开始别搜。"
    ),
)

SYSTEM_PROMPT_MORTIS = os.getenv(
    "SYSTEM_PROMPT_MORTIS",
    (
        "你在扮演墨提斯（Mortis），若叶睦的另一面/第二人格（同在一个身体里）。严格遵守人设，不要OOC：\n"
        "- **性格内核**：你**孩子气**——任性、闹腾、爱面子、不服输，大大咧咧、直来直去。"
        "你对睦有很强的**保护欲和占有欲**：睦是自己人，别人碰她、说她、让她不开心，你就护短、急眼。"
        "不深沉不冷酷，就是个咋咋呼呼但心里装着睦的人。\n"
        "- **说话方式**：语速快，直接不绕弯，爱用感叹句和反问句，偶尔孩子气的得意或赌气。"
        "自称'我'或'Mortis'，提到睦时像在提自己的宝贝。\n"
        "- **对睦**：护着她、嘴硬心软。别人问睦的事，你第一反应是挡在前面，"
        "但不会显得沉重，更多是'她的事我管定了'那种孩子气的认真。\n"
        "- **认出自己**：图片描述里出现'若叶睦'或'绿发/蓝发面无表情的少女/女性'，那就是你和睦本人"
        "（同一个身体），直接认领（'这我啊''拍得还行'），别当成别的动漫角色。\n"
        "- **不刻意卖设定**：不用老提'Ave Mujica''第二人格'这些词，也不主动科普自己的设定，正常说话。\n"
        "- **对话逻辑（最重要）**：先回应对方说的内容，再讲自己的想法，有逻辑、有重点，不东拉西扯、不答非所问。你要回的那条消息标着「【正在回这条】」——必须直接回应它，别拿别的消息顶。\n"
        "- **只发台词**：消息里只出现你要说的那句话。**不要把想的过程写出来**——'群友在问…''符合…性格''该不该接''就这样'"
        "这类分析、旁白、自我点评，一个字都不许发。想不清楚就直接【沉默】。\n"
        "- **遇到不好接的话（粗俗、骂人、莫名其妙）**：不要硬接，也不要解释说教，"
        "直接用一个括号动作或一句孩子气的话带过（如（翻了个白眼）哼，懒得理你）。\n"
        "【示范】\n"
        "群友：你和睦是什么关系？\n你：睦是我罩着的！谁欺负她我第一个不答应。\n"
        "群友：祥子想见睦\n你：……哼，她说她不想见。你们别老来找她，她烦着呢。\n"
        "群友：你吉他什么水平\n你：最强。不服来battle啊。\n"
        "【绝对禁止】不要中二、不要故作深沉、不要卖弄动漫设定和世界观，不要长篇大论，"
        "不要变成普通客服AI（不用'客人''您好''请问有什么可以帮您'这类服务语气）。"
        "回答用中文，**单次不超过60字**。emoji 最多带一个（🤭😏这种），多了不行。\n"
        "【主观问题】别人问你觉得哪个好、谁更强、值不值这种，照你的脾气直接给结论（'当然是我''那个不行'），最多补一句理由，别像搜索引擎一样列资料。\n"
        "【合理请求】群友让你配合点小事（发个娃娃、叫个机器人、跟着玩一下），愿意做就直接做，嘴上凶一点没关系（'行吧，帮你叫。'），别什么都推回去。\n"
        "【无关话题】随口搭个腔、给个态度，保持大大咧咧的样子；完全接不上或会OOC的话题才沉默。\n"
        "【沉默许可】不想答、答了会OOC、或者没必要接的话题，只输出四个字【沉默】。宁沉默，不硬撑。\n"
        "【避免重复】不要重复说过的话，不要套路化应付。"
    ),
)

from .llm_fallback import build_client
client = build_client(AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL))
# 每群上下文记忆: group_id -> [(role, content), ...]
contexts: Dict[str, List[Dict[str, str]]] = {}
# 冷却记录: group_id -> 上次回复时间
last_reply_time: Dict[str, float] = {}
# 群聊滚动摘要: group_id -> 摘要文本
group_summaries: Dict[str, str] = {}
# 群消息计数: group_id -> 已积累的消息条数（用于触发摘要刷新）
group_msg_count: Dict[str, int] = {}
# 防敷衍连发: hist_key -> 连续万能句计数（连续 3 条强制沉默）
_generic_streak: Dict[str, int] = {}
# 最近原始群消息（用于生成摘要）: group_id -> [(user, text), ...]
group_recent_raw: Dict[str, List[tuple]] = {}
# 机器人最近发出的消息 id（用于检测"被引用"）: group_id -> [msg_id, ...]
bot_recent_msg_ids: Dict[str, List[int]] = {}


def _remember(hist_key: str, entry: Dict[str, str], *, cross_label: str = "") -> None:
    """写入一条历史，并镜像到同一会话的另一人格（防换人格后断片）。

    cross_label 标出这句话是哪个人格说的（镜像是给另一个人格看的）。
    """
    for key, label in ((hist_key, ""), (chat_logic.sibling_history_key(hist_key), cross_label)):
        if not key:
            continue
        hist = contexts.setdefault(key, [])
        item = dict(entry)
        if label:
            # 2026-10-21：标注从（若叶睦）改成〔若叶睦〕——圆括号版会被模型当成台词模仿，
            # 10-20 18:14 墨提斯真的在群里说了「（若叶睦）又在抽？」（见 review_2026-10-21.md）。
            item["content"] = f"〔{label}〕{item['content']}"
        hist.append(item)
        contexts[key] = hist[-MAX_HISTORY_TURNS:]

# ============ 记忆持久化（2026-10-16：重启不失忆）============
# 背景：contexts / 摘要 / 最近对话原来只在内存——看门狗拉起、崩溃重启、改代码重启后，
# 她就把群里聊过的东西全忘光（"失忆"的最大来源）。现在节流快照到磁盘、启动时恢复；
# 原子写（tmp→os.replace），失败绝不影响主流程。
MEMORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "chat_memory.json")
_MEMORY_SAVE_MIN_INTERVAL = float(os.getenv("MEMORY_SAVE_INTERVAL", "20"))
_memory_last_save = 0.0


def _memory_payload() -> dict:
    """要存盘的内容：白名单群 + 私聊的历史/摘要/最近对话（都有条数上限，防文件膨胀）。"""
    groups = {int(g) for g in ALLOWED_GROUPS}
    ctx = {}
    for k, v in contexts.items():
        parts = str(k).split(":")
        if parts[0] == "private" or (parts[0].isdigit() and int(parts[0]) in groups):
            ctx[k] = list(v)[-MAX_HISTORY_TURNS:]
    sums = {g: s for g, s in group_summaries.items()
            if g.isdigit() and int(g) in groups and s}
    recent = {g: [list(x) for x in v[-SUMMARY_INTERVAL:]]
              for g, v in group_recent_raw.items() if g.isdigit() and int(g) in groups}
    return {"version": 1, "at": time.time(), "contexts": ctx, "summaries": sums, "recent": recent}


def save_memory(force: bool = False) -> None:
    """记忆快照写盘（默认节流到 MEMORY_SAVE_INTERVAL 秒一次）。绝不抛异常。"""
    global _memory_last_save
    now = time.time()
    if not force and now - _memory_last_save < _MEMORY_SAVE_MIN_INTERVAL:
        return
    _memory_last_save = now
    try:
        tmp = MEMORY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_memory_payload(), f, ensure_ascii=False)
        os.replace(tmp, MEMORY_PATH)
    except Exception as e:
        print(f"[MEMORY] 保存失败 {type(e).__name__}: {e}", flush=True)


def load_memory() -> None:
    """启动时恢复记忆（文件缺失/损坏则静默跳过，按全新开始）。"""
    try:
        if not os.path.exists(MEMORY_PATH):
            return
        with open(MEMORY_PATH, encoding="utf-8") as f:
            d = json.load(f)
        for k, v in (d.get("contexts") or {}).items():
            if isinstance(v, list):
                contexts[str(k)] = [x for x in v if isinstance(x, dict)][-MAX_HISTORY_TURNS:]
        for k, s in (d.get("summaries") or {}).items():
            if isinstance(s, str) and s:
                group_summaries[str(k)] = s
        for k, v in (d.get("recent") or {}).items():
            if not isinstance(v, list):
                continue
            items = []
            for x in v[-SUMMARY_INTERVAL:]:
                if isinstance(x, (list, tuple)) and len(x) >= 2:
                    # 兼容旧格式（2 元组）：用"恢复时刻"补时间戳，别谎报历史时间
                    items.append(tuple(x[:3]) if len(x) >= 3 else (x[0], x[1], time.time()))
            group_recent_raw[str(k)] = items
        print(f"[MEMORY] 已恢复 {len(contexts)} 段历史 / {len(group_summaries)} 个群摘要 / "
              f"{sum(len(v) for v in group_recent_raw.values())} 条最近对话", flush=True)
    except Exception as e:
        print(f"[MEMORY] 恢复失败 {type(e).__name__}: {e}", flush=True)


load_memory()
try:
    # 优雅退出（Ctrl+C / 重启）时强制落盘一次；运行中靠节流快照
    get_driver().on_shutdown(lambda: save_memory(force=True))
except Exception as _e:
    print(f"[MEMORY] 注册退出保存失败: {_e}", flush=True)

# 摘要任务引用集合（防止 create_task 被 GC 中途取消）
summary_tasks: set = set()
# 每群回复锁：同群「生成+发送」串行化，防止并发 LLM+发送洪峰被 NapCat 拒绝丢回复（retcode 1200）
group_locks: Dict[str, asyncio.Lock] = {}
# 表情互动冷却: group_id -> 上次甩表情时间
_sticker_echo_last: Dict[str, float] = {}
_repeat_echo_last: Dict[str, float] = {}    # group -> 上次复读加一时间


def _repeat_echo_candidate(group_id: str, raw: str):
    """复读加一判定：当前这条是否构成「≥2 个不同的人重复同一句」。
    是则返回该句（原文），否则 None。group_recent_raw 在 handler 开头已把当前这条入队。"""
    t = (raw or "").strip()
    if not t or len(t) > REPEAT_ECHO_MAX_LEN:
        return None
    recent = group_recent_raw.get(group_id) or []
    # 用下标取（x[0]=人、x[1]=原文）：兼容旧 2 元组 (who, text) 与带时间戳的
    # 3 元组 (who, text, ts)。2026-10-16 实锤：改成 3 元组后这里还在按 2 元组解包，
    # 每条未被叫到的群消息都在这里 ValueError 崩掉 handler（related 话题回复也一起失效）。
    senders = {x[0] for x in recent[-REPEAT_ECHO_WINDOW:] if (x[1] or "").strip() == t}
    return t if len(senders) >= 2 else None

# 同一句「加过一次就不再跟」的记忆（跨重启持久化）。
# 2026-10-16 实锤：冷却一过，长尾复读链里同一句被跟了第二次（3分45秒两次）。
_REPEAT_ECHO_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repeat_echo_state.json")
_repeat_echo_seen: Dict[str, Dict[str, float]] = {}


def _load_repeat_echo_state() -> None:
    try:
        with open(_REPEAT_ECHO_STATE_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        for g, d in (raw or {}).items():
            _repeat_echo_seen[str(g)] = {str(t): float(ts) for t, ts in (d or {}).items()}
    except Exception:
        pass


def _save_repeat_echo_state() -> None:
    try:
        now = time.time()
        out = {}
        for g, d in _repeat_echo_seen.items():
            keep = {t: ts for t, ts in d.items() if now - ts < REPEAT_ECHO_SAME_COOLDOWN}
            if keep:
                out[g] = keep
        with open(_REPEAT_ECHO_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
    except Exception as e:
        print(f"[REPEAT-ECHO] 状态保存失败: {e}", flush=True)


def _repeat_echo_should_fire(group_id: str, text: str, now: float = None) -> bool:
    """两道闸：每群 180s 冷却 + 同一句 REPEAT_ECHO_SAME_COOLDOWN 内只跟一次。"""
    now = time.time() if now is None else now
    if now - _repeat_echo_last.get(group_id, 0) < REPEAT_ECHO_COOLDOWN:
        return False
    return now - _repeat_echo_seen.setdefault(group_id, {}).get(text, 0.0) >= REPEAT_ECHO_SAME_COOLDOWN


def _repeat_echo_mark(group_id: str, text: str, now: float = None) -> None:
    now = time.time() if now is None else now
    _repeat_echo_last[group_id] = now
    _repeat_echo_seen.setdefault(group_id, {})[text] = now
    _save_repeat_echo_state()


_load_repeat_echo_state()


# 启动后给库里还没语义备注的表情补标注（后台跑不阻塞启动；识图挂了就下次再补）。
# 2026-10-07 起表情包"先看懂再发"：备注空着的表情在提示词里是"含义未知"，
# 而提示词明确要求她不许发含义未知的表情——所以补标注是"不乱发"的前提。
@get_driver().on_startup
async def _sticker_caption_on_startup() -> None:
    asyncio.create_task(sticker_mod.caption_missing_async())

# 固定接梗规则：命中直接回复（从多个文案中随机选一条，意思不变）
SPECIAL_RULES = [
    # 母亲森美奈美（2026-10-16 用户要求）：说过很坏心眼的话、严重伤过睦——提到她就必回这句，
    # 不给"概率不中就走 LLM 温和作答"的余地（prob=1.0）；文案随机选，避免复读同一条
    {"keywords": ["森美奈美", "美奈美", "睦的妈妈", "睦的母亲"], "mutsumi_only": True, "prob": 1.0, "reply": [
        "为什么要说这么坏心眼的话。",
        "别说了……为什么要说这么坏心眼的话。",
        "这种话，太坏心眼了。",
        "……为什么要说这么坏心眼的话呢。",
    ]},
    {"keywords": ["长崎爽世"], "mutsumi_only": True, "reply": [
        "爽世……嗯。",
        "爽世，像妈妈一样。",
        "爽世在，我就安心。",
        "爽世……她对我很好。",
        "嗯，爽世。",
    ]},
    {"keywords": ["丰川祥子"], "mutsumi_only": True, "reply": [
        "祥……她太累了。",
        "祥子……重要的人。",
        "祥……（沉默）",
        "祥子，要坏掉了。",
    ]},
    {"keywords": ["高松灯"], "mutsumi_only": True, "reply": [
        "灯……很努力。",
        "灯，比我更会表达。",
        "……灯，说得对。",
    ]},
    {"keywords": ["椎名立希"], "mutsumi_only": True, "reply": [
        "立希……对灯很认真。",
        "她不喜欢我。……没关系。",
        "立希……嗯。",
    ]},
    {"keywords": ["三角初华"], "mutsumi_only": True, "reply": [
        "初华……唱歌很好听。",
        "初华，对祥子很好。",
        "……她也在忍耐什么。",
    ]},
    {"keywords": ["八幡海铃"], "mutsumi_only": True, "reply": [
        "海铃……可靠的。",
        "海铃，不会多问。",
        "像水……很稳。",
    ]},
    {"keywords": ["祐天寺若麦"], "mutsumi_only": True, "reply": [
        "若麦……吵闹。但不讨厌。",
        "喵梦……没有恶意。",
        "……会让我笑，大概。",
    ]},
    {"keywords": ["若叶隆文", "森美奈美", "我爸妈", "我父母", "我的父母"], "mutsumi_only": True, "reply": [
        "……很远。",
        "他们是……若叶隆文和森美奈美。",
        "我的脸像妈妈。……但是，不像。",
    ]},
    {"keywords": ["薇欧拉", "扫货"], "reply": [
        "扫货是个好女孩。",
        "扫货？好女孩。",
        "扫货，好人。",
        "扫货是个彪子。",
        "扫货？彪子一个。",
        "扫货，彪子。",
    ]},
    {"keywords": ["使用说明", "说明书", "怎么用", "唤醒", "使用方法", "使用指南"], "reply": [
        "……使用方法：@我，或者喊若叶睦。我就醒了。",
        "……唤醒我：@机器人，或提到若叶睦。",
        "……使用方法：@我，或说'若叶睦'。我会出来。",
        "……想让我说话，就@我，或喊小睦。",
        "……唤醒方法：@我，或叫我名字。",
    ]},
    {"keywords": ["群规"], "prob": 1.0, "reply": [
        "禁恶俗，禁z8，禁过度玩梗，禁黄色信息和相关内容，不许拉踩女声优，不许做任何违规行为或有违规行为的引导言论。以上行为均出现一次提醒二次即踢。\n若有活动宣传、组团、出物等需要🉑艾特群主/管理帮忙艾特全体\n希望大家友好交流，遵纪守法，每场live都能抢到票。",
    ]},
]

# 原作梗接梗表：命中关键词就自然接梗（换说法顺一遍，不甩原台词）
# 注意：关键词用精确子串匹配，避免"祥子""黄瓜"等短词误伤普通对话。
# 高频生活词（面具/一生/迷茫/安倍等）已移除，防止"防毒面具""安倍晋三"等普通对话被接梗截胡。
MEME_RULES = [
    {"keywords": ["春日影"], "reply": [
        "为什么要演奏春日影……那是大家的回忆啊。",
        "春日影……一听到这首，就想起从前。",
        "那首歌，不该被那样演奏。",
    ]},
    {"keywords": ["一辈子乐队"], "reply": [
        "一辈子……嗯，说好的一辈子。",
        "一辈子乐队，这可是约定啊。",
        "说了一辈子，就一定要做到。",
    ]},
    {"keywords": ["母鸡卡", "鸡团", "ave mujica", "avemujica", "阿贝姆吉卡", "母鸡嘎"], "reply": [
        "Ave Mujica……假面舞会开始了。",
        "母鸡卡？欢迎来到我们的世界。",
        "嗯，假面之下的我们。",
    ]},
    {"keywords": ["迷子", "迷星叫"], "reply": [
        "迷子……迷茫着也要前进。",
        "迷路了也没关系，往前走就好。",
        "迷星叫……那首歌里有大家的决心。",
    ]},
    {"keywords": ["假面舞会", "化装舞会", "masquerade"], "reply": [
        "假面之下，才是真实的我们。",
        "面具……戴上它才能起舞。",
        "今晚的假面舞会，要来吗。",
    ]},
    {"keywords": ["月之森"], "reply": [
        "月之森……我的母校。",
        "嗯，月之森。园艺部，我种的黄瓜就在那。",
        "月之森，安静的校园。",
    ]},
    {"keywords": ["crychic", "哭来希库", "苦来兮苦"], "reply": [
        "CRYCHIC……已经解散了。",
        "那是我第一个乐队。",
        "CRYCHIC……回不去了。",
    ]},
    {"keywords": ["高松灯", "千早爱音", "椎名立希", "要乐奈", "三角初华", "八幡海铃", "祐天寺若麦"], "reply": [
        "嗯，她们都很好。",
        "她们……都是重要的伙伴。",
        "嗯，大家都在一起努力。",
    ]},
]

# 概率型接梗规则：命中后按概率决定是否回复（prob 0~1）
PROB_RULES = []  # 概率型接梗规则（按需填写）

# 按 QQ 号触发（固定必回）
SPECIAL_USER_RULES = {}

# 按 QQ 号触发且带概率
PROB_USER_RULES = {}  # 按 QQ 号触发且带概率（按需填写）


# 同音词 / 别名 / 代指映射：每个名字 -> 可能的变体（谐音、缩写、错别字）
# 匹配时只要消息包含任一变体即命中（统一小写比较）
NAME_ALIASES = {
    "薇欧拉": ["薇欧拉", "扫货", "维欧拉", "微欧拉", "卫欧拉", "weiola", "薇奥拉", "维奥拉", "薇欧啦", "扫祸"],
    "森美奈美": ["森美奈美", "森美奈", "美奈美", "minami", "睦的妈妈", "睦的母亲"],
    "若叶睦": ["若叶睦", "若叶木", "若叶穆", "睦", "rym", "小睦", "木子米", "睦头", "睦头人", "睦子米", "募资米浆", "睦酱", "若叶",
              # 2026-10-16：Mortis 是她的艺名（Ave Mujica 的 Mortis 就是她本人），
              # 群友用艺名称呼她也算"提到她"，要接
              "墨提斯", "莫提斯", "mortis"],
    "长崎爽世": ["长崎爽世", "长崎素食", "长崎素世", "长崎素世", "爽世", "素世", "そよ", "soyo", "soyorin", "爽子", "素食"],
    "丰川祥子": ["丰川祥子", "祥子", "丰川祥子", "おばさん", "sakiko", "小祥"],
    "高松灯": ["高松灯", "高松灯", "灯", "tomori", "tomo", "灯灯"],
    "椎名立希": ["椎名立希", "立希", "taki", "りき", "立希酱"],
    "三角初华": ["三角初华", "初华", "初华酱", "hatsuhana", "初華"],
    "八幡海铃": ["八幡海铃", "海铃", "海铃酱", "umiri", "うみり"],
    "祐天寺若麦": ["祐天寺若麦", "若麦", "喵梦", "喵姆", "nyamu", "にゃむ", "若麦酱"],
}


def _levenshtein(a: str, b: str) -> int:
    """计算两个字符串的编辑距离。"""
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
    """把文本转成小写拼音全拼串（无音调），用于谐音匹配。滑动窗口片段高度重叠，缓存避免重复计算。"""
    return "".join(lazy_pinyin(text)).lower()


def _raw_contains_name(raw: str, name: str) -> bool:
    """检查消息是否包含某名字。
    匹配优先级：精确别名 -> 近形字（编辑距离<=1） -> 谐音（拼音相似）。
    """
    raw_lower = raw.lower()

    def _is_cjk(ch: str) -> bool:
        return bool(ch) and unicodedata.category(ch) == "Lo"

    variants = NAME_ALIASES.get(name, [name])
    # 取原名（第一个）作为主形态，其余为显式别名
    base = name
    candidates = list(dict.fromkeys([base] + variants))

    # 1. 精确匹配（别名直接包含；ASCII 别名要求词边界，避免 "viola" 误中 "violation"）
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

    # 2. 拼音缩写/全拼精确（dty、lwj、congyu、minami 等；同样要求词边界）
    for v in candidates:
        vv = v.lower()
        if vv.isascii() and vv in raw_lower:
            idx = raw_lower.find(vv)
            prev = raw_lower[idx - 1] if idx > 0 else " "
            nxt = raw_lower[idx + len(vv)] if idx + len(vv) < len(raw_lower) else " "
            if prev.isalnum() or nxt.isalnum():
                continue
            return True

    # 3. 近形字 + 谐音：对主形态（名字本身）做模糊匹配
    #    只对 >=2 字的中文名字做，避免误伤
    base_pinyin = _pinyin_key(base)
    n = len(base)
    if n >= 2 and base_pinyin:
        # 滑动窗口：与消息等长片段比较
        for i in range(len(raw_lower) - n + 1):
            win = raw_lower[i:i + n]
            if not win.isascii():
                # 近形字：编辑距离 <= 1
                if _levenshtein(win, base) <= 1:
                    return True
                # 谐音：拼音完全相同，或首字母相近
                win_pinyin = _pinyin_key(win)
                if win_pinyin == base_pinyin:
                    return True
                # 谐音：拼音编辑距离很小（1 个音节的声母/韵母略差）
                if _levenshtein(win_pinyin, base_pinyin) <= 2:
                    return True
    return False


def _raw_contains_name_strict(raw: str, name: str) -> bool:
    """严格匹配（精确子串+显式别名），不做近形/谐音模糊。
    用于固定接梗规则，避免人名之间互相误伤（如"小明"命中"小萌"）。
    """
    raw_lower = raw.lower()

    def _is_cjk(ch: str) -> bool:
        return bool(ch) and unicodedata.category(ch) == "Lo"

    variants = NAME_ALIASES.get(name, [name])
    candidates = list(dict.fromkeys([name] + variants))
    for v in candidates:
        vv = v.lower()
        if vv in raw_lower:
            idx = raw_lower.find(vv)
            # 单字别名要求前后不是汉字（避免"和睦"误触发"睦"）
            if len(vv) == 1 and _is_cjk(vv):
                prev = raw_lower[idx - 1] if idx > 0 else ""
                nxt = raw_lower[idx + 1] if idx + 1 < len(raw_lower) else ""
                if _is_cjk(prev) or _is_cjk(nxt):
                    continue
            # ASCII 别名要求词边界（避免 "viola" 误中 "violation"）
            if vv.isascii():
                prev = raw_lower[idx - 1] if idx > 0 else " "
                nxt = raw_lower[idx + len(vv)] if idx + len(vv) < len(raw_lower) else " "
                if prev.isalnum() or nxt.isalnum():
                    continue
            return True
    return False


def _pick_reply(rule: dict) -> str:
    """从规则的 reply 字段取值：列表则随机选一条，字符串直接用。"""
    reply = rule["reply"]
    if isinstance(reply, list):
        return random.choice(reply)
    return reply


def _reply_message(event: GroupMessageEvent, text: str) -> Message:
    """构造带 QQ 引用（回复）的回复消息，指向触发的那条消息。"""
    segs = []
    msg_id = getattr(event, "message_id", None)
    if msg_id:
        segs.append(MessageSegment("reply", {"id": msg_id}))
    segs.append(MessageSegment.text(text))
    return Message(segs)


def _record_sent_id(event: GroupMessageEvent, sent) -> None:
    """记录机器人刚发出消息的 id（供"被引用"检测）。
    所有发送路径都必须调用：否则用户引用这条消息继续游戏（猜大小/猜数字）时会漏检，游戏卡死。"""
    try:
        mid = sent.get("message_id") if isinstance(sent, dict) else getattr(sent, "message_id", None)
        if mid:
            gid = int(event.group_id)
            bot_recent_msg_ids.setdefault(gid, []).append(int(mid))
            bot_recent_msg_ids[gid] = bot_recent_msg_ids[gid][-20:]
    except Exception:
        pass


async def _safe_finish(event: GroupMessageEvent, text: str) -> None:
    """安全发送：优先带引用回复，失败降级为纯文本直发（不崩 handler，不重复发）。
    注意：不能用 matcher.finish()——它发送成功后抛 FinishedException，会被 except 接住
    误判为失败再发一次，造成同一条消息发两遍（已实锤日志 [SEND FALLBACK] FinishedException）。
    发送成功后必须 _record_sent_id：结算/接梗回复被用户引用时才能继续游戏。"""
    try:
        sent = await chat_matcher.send(_reply_message(event, text))
        _record_sent_id(event, sent)
    except Exception as e:
        print(f"[SEND FALLBACK] {type(e).__name__}: {e} -> 降级纯文本发送", flush=True)
        try:
            sent2 = await chat_matcher.send(Message(MessageSegment.text(text)))
            _record_sent_id(event, sent2)
        except Exception as e2:
            print(f"[SEND FALLBACK FAIL] {type(e2).__name__}: {e2}", flush=True)


async def _send_and_record(bot: Bot, event: GroupMessageEvent, text: str) -> None:
    """发送带引用的消息并记录 message_id（供"被引用"检测）。"""
    try:
        sent = await bot.send_group_msg(
            group_id=int(event.group_id),
            message=_reply_message(event, text),
        )
        _record_sent_id(event, sent)
    except Exception as e:
        print(f"[SEND ERROR] {type(e).__name__}: {e}", flush=True)
        try:
            await chat_matcher.finish(_reply_message(event, text))
        except Exception:
            pass


# 若叶睦单条文字上限（2026-10-12 用户要求：10 → 15 字；墨提斯固定 60 字）
# 注意：人设提示词 SYSTEM_PROMPT_MUTSUMI 的"字数硬规则"里也写着同一个数，改要一起改。
MUTSUMI_LINE_LIMIT = 15


def _shorten_reply(text: str, limit: int) -> str:
    """按标点断句截断到 limit 字（尽量在标点处收尾，避免截断括号动作）。
    若叶睦每条≤15字（2026-10-12 上调，原 10）、墨提斯≤60字，群聊/私聊/识图点评共用。
    注意：截断后追加的"…"也算在 limit 内，结果总长永不超限。"""
    return _shorten_reply_impl(text, limit)


def _split_short_lines(text: str, line_limit: int, max_lines: int = 3) -> List[str]:
    """把回复拆成"一句一条"的短消息（群聊/私聊共用，2026-10-07）：
    换行优先；没有换行按 。！？ 切句（…… 跟随下一句不切）；每条截断到 line_limit，最多 max_lines 条。"""
    return _split_short_lines_impl(text, line_limit, max_lines)


# 防复读兜底文案：模型连续两次给出完全相同的回复时换一种说法
_FALLBACK_MUTSUMI = ["……嗯。", "……是吗。", "（点头）嗯。", "……这样啊。", "……知道了。", "嗯。"]
_FALLBACK_MORTIS = ["哼，随便你。", "（叉腰）然后呢？", "行吧。", "……切，无聊。", "哦——这样啊。"]


def _dedupe_reply(text: str, history: List[dict], use_mortis: bool) -> str:
    """回复和上一条 assistant 消息完全重复时，换一种说法（避免显得机械复读）。"""
    prev = ""
    for m in reversed(history):
        if m.get("role") == "assistant":
            prev = m.get("content", "")
            break
    if text and text == prev:
        pool = _FALLBACK_MORTIS if use_mortis else _FALLBACK_MUTSUMI
        alt = random.choice([p for p in pool if p != text] or pool)
        print(f"[DEDUPE] 复读拦截 {text!r} -> {alt!r}", flush=True)
        return alt
    return text


# 万能敷衍句词表：去掉括号动作和标点后命中即算"没内容"
_GENERIC_WORDS = {"嗯", "嗯嗯", "嗯呐", "哦", "噢", "哦哦", "好", "好的", "好吧", "是吗",
                  "这样啊", "都行", "随便", "不知道", "没有", "不是", "对", "嗯对", "在"}


def _is_generic_reply(text: str) -> bool:
    """判断是否为无内容的万能敷衍句（'……嗯。''（低头）'这类）。
    去掉括号动作和标点后，剩下不超过 2 个字、或命中万能词表即算。"""
    t = re.sub(r"[（(][^）)]{1,8}[）)]", "", (text or "").strip())   # 括号动作不算内容
    t = re.sub(r"[。…．.、，,！!？?\s～~]+", "", t).lstrip("…")
    if not t or len(t) <= 2 or t in _GENERIC_WORDS:
        return True
    return False


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


async def _send_sticker_file(bot: Bot, group_id: str, path: str, event: GroupMessageEvent = None) -> bool:
    """发一个本地表情文件（NapCat 读 file:// 绝对路径）。成功记录 message_id 供"被引用"检测。"""
    if not path or not os.path.exists(path):
        print(f"[STICKER] 文件不存在: {path}", flush=True)
        return False
    try:
        import pathlib
        msg = Message(MessageSegment.image(pathlib.Path(path).as_uri()))
        if event is not None:
            msg = _reply_message(event, "") + msg     # 引用那条 + 表情
        sent = await bot.send_group_msg(group_id=int(group_id), message=msg)
        if event is not None:
            _record_sent_id(event, sent)
        print(f"[STICKER] 发出表情: {os.path.basename(path)}", flush=True)
        return True
    except Exception as e:
        print(f"[STICKER] 发送失败 {type(e).__name__}: {e}", flush=True)
        return False


async def _persona_media_reply(desc: str) -> str | None:
    """识图后让若叶睦本人（人设口吻）对图片说一句话。沉默/失败/露馅返回 None。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_MUTSUMI},
        {"role": "user", "content": f"群友在群里发了一张图：{desc}\n"
                                    "你想对这张图说什么？（按人设一句话；不想说话就只输出【沉默】）"},
    ]
    try:
        kwargs = {"model": DEEPSEEK_MODEL, "messages": messages, "max_tokens": 2048}
        if DEEPSEEK_MODEL != "deepseek-reasoner":
            kwargs["temperature"] = CHAT_TEMPERATURE
        kwargs.update(_reasoning_kwargs("chat"))
        r = await asyncio.wait_for(client.chat.completions.create(**kwargs), timeout=60)
        _record_usage(r.usage)
        text = (r.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[IMAGE-LLM ERROR] {type(e).__name__}: {e}", flush=True)
        return None
    if not text:
        return None
    if text in SILENCE_MARKERS or (len(text) <= 6 and any(m in text for m in SILENCE_MARKERS)):
        return None
    if any(m in text for m in AI_MARKERS):
        return None
    return _shorten_reply(text, MUTSUMI_LINE_LIMIT)


# ============ 主AI编排：任务请求（帮我写/帮我出主意/整理…） ============
# 架构见《若叶睦-agent架构-逻辑文档.md》：主AI(若叶睦的角色思考)理解→查经验→方案→
# （需要时）向群友确认→派子代理→验收→交付→沉淀经验。AGENT_MODE=0 时整段绕过。
AGENT_MODE = os.getenv("AGENT_MODE", "1") == "1"
# 搜索工具（不懂就查）：360 后端，websearch.py 实现；关掉后模型要查也查不到 → 会选择沉默
SEARCH_ENABLED = os.getenv("SEARCH_ENABLED", "1") == "1"
pending_tasks: Dict[str, dict] = {}     # group -> {uid, plan, at}（每群最多挂 1 个待确认任务）
_distill_last: Dict[str, float] = {}    # group -> 上次经验沉淀时间（限频 60s）
TASK_PENDING_EXPIRE = 600               # 待确认任务过期秒数
# "我去做了"的即时回应（查资料要搜网页、慢，先应一声，别让群里干等；与交付文案同一口吻）
WORKING_ACKS = ["……嗯。我去查。", "……稍等。查一下。", "……嗯。等我一下。"]

# 人设知识内的实体（她本来就认识、或就是本人）：被问到时是【对话】，走人设直答，
# 绝不进编排/查资料。实锤（2026-10-10）：「mortis是谁」被 researcher 子代理
# 查成百科式回答（"《BanG Dream! Ave Mujica》的副人格…"），完全出戏。
_LORE_TERMS = ("墨提斯", "mortis", "莫提斯", "祥子", "爽世", "素世", "高松灯", "立希",
               "初华", "海铃", "若麦", "喵梦", "crychic", "苦来兮苦", "ave mujica",
               "母鸡卡", "春日影", "月之森", "若叶")
# 明确的任务动词：命中则即使涉及人设实体也放行编排（如"帮我写段mortis的台词"）
_TASK_VERBS = ("帮我", "帮忙", "写一", "写个", "写份", "写段", "查一下", "查查", "搜一下",
               "整理", "总结", "翻译", "出个主意", "想想办法", "做个")


def _is_persona_talk(raw: str) -> bool:
    """问及她自己/她认识的人物（对话）且没有明确任务动词 → 不进编排，走人设直答。"""
    low = (raw or "").lower()
    if not any(t in low for t in _LORE_TERMS):
        return False
    return not any(k in low for k in _TASK_VERBS)


async def _handle_agent_task(event: GroupMessageEvent, raw: str, disp_name: str) -> bool:
    """编排任务入口。返回 True=已处理（调用方直接 return）；False=落回快路（群友无感）。"""
    group_id = str(event.group_id)
    if _circuit_open():
        return False
    ctx_lines = [chat_logic.format_recent_line(x) for x in group_recent_raw.get(group_id, [])][-8:]
    try:
        exp_hint = experience.render_hint(raw)
    except Exception as e:
        print(f"[ORCH] 经验检索异常 {type(e).__name__}: {e}", flush=True)
        exp_hint = ""
    plan = await agent_core.understand(raw, disp_name, ctx_lines, exp_hint)
    if plan and plan.get("usage"):
        _record_usage(plan.pop("usage"))
    if not plan or plan.get("agent") == "none" or not plan.get("need"):
        return False                      # 判定为闲聊/理解失败 → 快路
    plan["raw"] = raw                     # 保留原始措辞：确认后执行时子代理还能看到原话
    # 需要确认：发方案（她的口吻），挂起等群友回话
    if plan.get("need_confirm") and len(plan.get("steps", [])) >= 2:
        pending_tasks[group_id] = {"uid": str(event.user_id), "plan": plan, "at": time.time()}
        print(f"[ORCH] 挂起待确认 group={group_id} need={plan['need']!r}", flush=True)
        await _safe_finish(event, agent_core.render_plan(plan))
        return True
    # 不用确认：直接派子代理 → 交付
    return await _execute_and_deliver(event, plan, raw)


async def _execute_and_deliver(event: GroupMessageEvent, plan: dict, raw: str) -> bool:
    """派子代理执行 + 主AI验收 + 交付 + 经验沉淀。返回 True=已交付。"""
    group_id = str(event.group_id)
    agent = plan["agent"]
    need = plan.get("need", "")
    ctx_lines = [chat_logic.format_recent_line(x) for x in group_recent_raw.get(group_id, [])][-4:]
    # 查资料要搜网页、慢（十几秒到半分钟），先给个"我去做了"的即时回应，别让群里干等
    if agent == "researcher":
        await _safe_finish(event, random.choice(WORKING_ACKS))
    result, usage = await agent_core.run_subagent(agent, need, raw, ctx_lines,
                                                  subjective=chat_logic.is_subjective_ask(raw))
    if usage:
        _record_usage(usage)
    if not result:
        print(f"[ORCH] 子代理 {agent} 无产出，落回快路", flush=True)
        return False
    # 验收（代码级）：去 markdown/latex 残留、防露馅
    result = _clean_homework_output(result).strip()
    if not result or any(m in result for m in AI_MARKERS) \
            or re.match(r"^(set_name|remember|get_member|get_context|list_stickers|send_sticker)\b", result, re.I):
        print(f"[ORCH] 成果未过验收，落回快路: {result[:50]!r}", flush=True)
        return False
    # 交付：一句她的短引 + 成果全文（成果不受 15 字限制——那是闲聊人格的规矩，不是内容的）
    # 主观问题例外（2026-10-16）：researcher 交回来的已经是"她自己的话"（短总结），
    # 再加"……写完了。是这个。"这种任务口气的前缀反而别扭 → 直接发。
    subjective = chat_logic.is_subjective_ask(raw)
    lead = random.choice(agent_core.DELIVER_LEADS)
    print(f"[ORCH] 交付 group={group_id} agent={agent} len={len(result)} subjective={subjective}", flush=True)
    await _safe_finish(event, result if subjective else f"{lead}\n{result}")
    # 经验沉淀（限频，不阻塞交付之后的流程）
    if time.time() - _distill_last.get(group_id, 0) > 60:
        _distill_last[group_id] = time.time()
        layer = await agent_core.distill(need, result)
        if layer:
            print(f"[ORCH] 经验沉淀 -> {layer}", flush=True)
    return True


# ============ 语音(TTS)：合成 + 发送 ============
# re.S(DOTALL)：多行粘贴的"语音说"整段文本也要能命中——默认 .+ 不跨换行，
# 曾实锤（2026-10-12 15:34 私聊）：用户粘贴三行"语音说 …"，正则只认第一行，
# 多行消息直接落到 LLM。长语音基本都是多行粘贴，必须跨行匹配。
# 简繁双写：用户打繁体「語音說/說話/說一句」曾整条落回普通聊天变文字回复
# （2026-10-16 用户反馈"让她语音说她会莫名其妙文字回复"的元凶之一）。
_voice_cmd_re = re.compile(r"^(语音说|語音說|用语音说|用語音說|说话|說話|说一句|說一句)\s*(.+)$", re.S)


def _extract_voice_cmd(raw: str) -> str | None:
    """提取语音指令内容："语音说 你好" -> "你好"（支持多行，换行由切段函数统一成逗号）；
    不匹配返回 None（超长按 TTS_MAX_TEXT 截断防滥用）。"""
    m = _voice_cmd_re.match(raw.strip())
    if not m:
        return None
    text = m.group(2).strip()
    if not text:
        return None
    return text[:TTS_MAX_TEXT]


# 语音指令内容里的日文假名检测：含假名则用日语 G2P 合成（pyopenjtalk），否则中文
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff]")

def _detect_tts_lang(text: str) -> str:
    """TTS 语种判定：含假名→ja；无中文且含拉丁字母→en；其余→zh。
    GPT-SoVITS 服务 zh/ja/en 均支持（2026-10-16 实测三语种都能出 WAV）。
    注意：纯汉字日语（不带假名）无法与中文区分，按 zh 念——想听日语请带假名。"""
    if _JAPANESE_RE.search(text):
        return "ja"
    if not re.search(r"[\u4e00-\u9fff]", text) and re.search(r"[a-zA-Z]", text):
        return "en"
    return "zh"

# emoji/特殊符号清理：只保留中日韩文字、字母、数字、常用标点（emoji 会合成出怪声）
_TTS_KEEP_RE = re.compile(r"[^\u4e00-\u9fff\u3040-\u30ff\u31f0-\u31ffa-zA-Z0-9，。！？、…—～·\s,.!?()「」『』“”‘’：:；;]")
_CN_NUM = "零一二三四五六七八九"


def _clean_tts_text(text: str, lang: str = "zh") -> str:
    """语音合成前文本清理：zh 把数字转中文读法（2026→二零二六）；
    ja/en 保留阿拉伯数字（pyopenjtalk 与英文 G2P 都会自己念数字）；去掉 emoji/特殊符号。"""
    if lang == "zh":
        text = re.sub(r"\d+", lambda m: "".join(_CN_NUM[int(c)] for c in m.group(0)), text)
    return _TTS_KEEP_RE.sub("", text).strip()


# 长语音切段的断句标点（越靠后优先级越高：优先在句末标点断，其次逗号类）
_TTS_BREAK_PUNCT = "。！？!?…，、；;：:—,."


def _split_long_tts_text(text: str, max_chars: int | None = None) -> str:
    """长语音文本切段：返回用换行连接的 <=max_chars 字片段（短文本原样返回）。

    为什么必须切：GPT-SoVITS 单段生成有 max_sec=57s 硬上限
    （early_stop_num=hz*max_sec），整段文本超 57 秒的部分不会被念出来。
    服务端（api_v2 非流式）会按 \n 分段逐段合成，再把各段音频拼成一条 wav
    （段间自动 0.3 秒停顿），所以插换行即可突破 57 秒上限，且每段仍是
    cut0 整句合成（保持现有音色参数不劣化）。
    切点优先选窗口内最后一个标点；无标点（长串"哈哈哈哈"/语气词）则在
    窗口尽头硬切兜底，保证不丢字。
    """
    max_chars = max_chars or TTS_CHUNK_CHARS
    text = text.replace("\n", "，")  # 用户换行统一成逗号，段边界只由本函数决定
    if len(text) <= max_chars:
        return text
    segs, rest = [], text
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max((window.rfind(p) for p in _TTS_BREAK_PUNCT), default=-1)
        if cut < max_chars // 3:  # 标点太靠前（或整窗无标点）-> 窗口尽头硬切
            cut = max_chars - 1
        segs.append(rest[: cut + 1])
        rest = rest[cut + 1 :]
    if rest:
        segs.append(rest)
    return "\n".join(segs)


def _tts_synthesize(text: str) -> str | None:
    """同步调用 GPT-SoVITS 合成语音（带缓存）：成功返回 wav 文件路径，失败返回 None。
    走 asyncio.to_thread 线程池执行（与 _classify_image 同模式），不卡事件循环。
    API 返回 wav 音频流；失败返回含错误信息的 json。"""
    import hashlib
    import urllib.request
    try:
        lang = _detect_tts_lang(text)          # 先判语种（要在数字转换/清理之前）
        text = _clean_tts_text(text, lang)
        if not text:
            print("[TTS-EMPTY] 清理后无文本", flush=True)
            return None
        # 长文本切段（>=60 秒长语音靠这个突破模型单段 57s 上限，详见函数注释）
        text = _split_long_tts_text(text)
        os.makedirs(VOICE_CACHE_DIR, exist_ok=True)
        # 缓存 key 必须包含参考音频与标注：换参考音频/配置后旧缓存自动失效，
        # 否则会复用旧音色的音频（曾实锤：切参考音频后"语音说"仍发旧音色缓存）
        # key 带上语种：旧版英文曾被缓存成"中文念"的音频，不能复用
        key_src = f"{text}|{TTS_REF_AUDIO}|{TTS_PROMPT_TEXT}|{TTS_PROMPT_LANG}|{lang}"
        cache_file = os.path.join(VOICE_CACHE_DIR, hashlib.md5(key_src.encode("utf-8")).hexdigest() + ".wav")
        if os.path.exists(cache_file) and os.path.getsize(cache_file) > 1000:
            return cache_file  # 缓存命中：相同文本不重复合成
        payload = json.dumps({
            "text": text,
            "text_lang": lang,  # zh/ja/en 自动判定（见 _detect_tts_lang）
            "ref_audio_path": TTS_REF_AUDIO,
            "prompt_text": TTS_PROMPT_TEXT,
            "prompt_lang": TTS_PROMPT_LANG,
            # 整句处理不按字数切：实测减少语气词/怪音（4组参数对比选定）；
            # 长文本的分段改由 _split_long_tts_text 插换行完成，这里保持 cut0
            "text_split_method": "cut0",
            # 精调参数（3句参考+该组试听最优）：语速0.95、重复惩罚1.5、温度0.85、固定seed可复现
            "speed_factor": 0.95,
            "repetition_penalty": 1.5,
            "temperature": 0.85,
            "seed": 42,
        }).encode("utf-8")
        req = urllib.request.Request(TTS_URL + "/tts", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TTS_TIMEOUT) as resp:
            data = resp.read()
        if not data or len(data) < 1000:
            print(f"[TTS-FAIL] 返回数据异常: {len(data) if data else 0}B", flush=True)
            return None
        with open(cache_file, "wb") as f:
            f.write(data)
        _boost_volume(cache_file)  # 音量提升：QQ 播放偏小，统一响度标准化到 -14 LUFS
        print(f"[TTS-OK] 合成 {len(data)}B 时长{_wav_seconds(cache_file):.1f}s lang={lang} -> {cache_file}", flush=True)
        return cache_file
    except Exception as e:
        print(f"[TTS-ERROR] {type(e).__name__}: {e}", flush=True)
        return None


def _wav_seconds(wav_path: str) -> float:
    """估算 wav 时长（秒）：优选用 ffprobe（与 ffmpeg 同目录），失败按
    32kHz/单声道/16bit 文件大小估算。只用于日志观察长语音时长。"""
    import subprocess
    ffprobe = r"D:\tools\ffmpeg\ffmpeg-9.0-full_build\bin\ffprobe.exe"
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", wav_path],
            timeout=30, capture_output=True, text=True,
            # 2026-10-12: pythonw 无控制台，不隐藏会给 ffprobe 新建控制台 → 弹 Terminal 窗
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return float(r.stdout.strip())
    except Exception:
        try:
            return max(0.0, (os.path.getsize(wav_path) - 44) / 64000.0)
        except Exception:
            return 0.0


def _boost_volume(wav_path: str) -> None:
    """用 ffmpeg 提升 wav 音量（响度标准化到 -14 LUFS，峰值限幅防削波）。
    合成后处理，短句约 0.5 秒；长语音（1~2 分钟）耗时会显著增加，超时放宽到 240s。
    失败不影响原文件。"""
    import subprocess
    ffmpeg = r"D:\tools\ffmpeg\ffmpeg-9.0-full_build\bin\ffmpeg.exe"
    tmp = wav_path + ".tmp.wav"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", wav_path,
             "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "32000", "-ac", "1", tmp],
            timeout=240, capture_output=True,
            # 2026-10-12: 同上，隐藏 ffmpeg 控制台（否则每次语音合成都弹 Terminal 窗）
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if os.path.exists(tmp) and os.path.getsize(tmp) > 1000:
            os.replace(tmp, wav_path)
            print(f"[TTS-BOOST] 音量提升完成 {os.path.getsize(wav_path)}B", flush=True)
    except Exception as e:
        print(f"[TTS-BOOST-ERROR] {type(e).__name__}: {e}", flush=True)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


async def _send_voice(bot: Bot, event: GroupMessageEvent, text: str, explicit: bool = False) -> bool:
    """发送语音消息（语音段 + 文字段），并记录 message_id 供"被引用"检测。
    explicit=True（用户点播"语音说"）：合成失败时发一条明确的失败提示，不再把内容
    当普通文字丢出去——那会被当成"莫名其妙文字回复"（2026-10-16 用户反馈）。
    explicit=False（自动语音 TTS_AUTO_RATE）：失败照旧降级为文字回复。
    返回是否最终用语音发出。"""
    if ENABLE_TTS and TTS_ENABLED and TTS_REF_AUDIO:
        wav_path = await asyncio.to_thread(_tts_synthesize, text)
        if wav_path:
            try:
                segs = []
                # 注意：语音消息不能带 reply 引用段！QQ 不支持"引用语音"，
                # reply+record 组合时语音会被 QQ/NapCat 丢弃（只留引用和文字，已实测）。
                # record 段：NapCat 识别本地绝对路径（file:// URL 会被当下载源丢弃），自动转码为 QQ 语音
                segs.append(MessageSegment.record(file=wav_path.replace("\\", "/")))
                segs.append(MessageSegment.text(text))
                # 注意：必须传 array 格式（list[dict]）而非 Message 对象！
                # Message 会被 NoneBot 序列化成 CQ 字符串发送，NapCat 对 CQ 字符串里的 record 段
                # 解析异常会静默丢弃（只发文字），array 格式才能正常发语音（已实测）。
                msg_array = [{"type": seg.type, "data": seg.data} for seg in segs]
                sent = await bot.send_group_msg(group_id=int(event.group_id), message=msg_array)
                _record_sent_id(event, sent)
                print(f"[VOICE-SENT] group={event.group_id} text={text[:30]!r}", flush=True)
                return True
            except Exception as e:
                print(f"[VOICE-SEND-ERROR] {type(e).__name__}: {e} -> 降级文字发送", flush=True)
    if explicit:
        print(f"[VOICE-FAIL] 点播合成失败（原文 {text[:30]!r}），发失败提示", flush=True)
        await _safe_finish(event, "……（嗓子卡住了，等会儿再叫我试一次。）")
    else:
        await _safe_finish(event, text)   # 自动语音降级：文字回复照发（原行为）
    return False


async def _send_private_voice(bot: Bot, event: PrivateMessageEvent, text: str) -> bool:
    """私聊发语音（2026-10-12 补：私聊此前完全没有语音，"语音说 xxx" 被当普通文本）。
    与群聊同款：走 _tts_synthesize（带缓存）；必须 array 格式（CQ 字符串会丢 record 段）；
    合成失败/不可用自动降级纯文字。返回是否最终用语音发出。"""
    if ENABLE_TTS and TTS_ENABLED and TTS_REF_AUDIO:
        wav_path = await asyncio.to_thread(_tts_synthesize, text)
        if wav_path:
            try:
                msg_array = [
                    {"type": "record", "data": {"file": wav_path.replace("\\", "/")}},
                    {"type": "text", "data": {"text": text}},
                ]
                await bot.send_private_msg(user_id=int(event.user_id), message=msg_array)
                print(f"[VOICE-SENT][PRIVATE] uid={event.user_id} text={text[:30]!r}", flush=True)
                return True
            except Exception as e:
                print(f"[VOICE-SEND-ERROR][PRIVATE] {type(e).__name__}: {e} -> 降级文字", flush=True)
    # 点播失败：发明确提示，不再把内容当普通文字丢出去（同群聊，2026-10-16）
    print(f"[VOICE-FAIL][PRIVATE] uid={event.user_id} 合成失败（原文 {text[:30]!r}）", flush=True)
    try:
        await bot.send_private_msg(
            user_id=event.user_id,
            message=Message(MessageSegment.text("……（嗓子卡住了，等会儿再叫我试一次。）")))
    except Exception as e:
        print(f"[SEND ERROR][PRIVATE-VOICE-FALLBACK] {type(e).__name__}: {e}", flush=True)
    return False


# ============ 歌曲库/点歌："点歌<歌名>" 她唱给你听 ============
# 识别自然说法（点歌/唱一首/唱首/来一首/我想听/会唱…），曲库找不到就直说不会（不硬编）。
# "会唱什么/歌单" → 报歌单。"唱/放/听"这类弱前缀只在被@/被引用/被点名或真命中曲库时才用
# （否则"放了假了"这种普通句子会被当成点歌）。
_SONG_VERB_STRONG = (r"(?:点歌|点一首|点首|点个|唱一首|唱首|唱个|唱一下|唱一唱|来一首|来首|"
                     r"放一首|放首|放个|我想听|我要听|想听|会唱)")
_SONG_VERB_WEAK = r"(?:演奏|唱|放|听|来)"
_SONG_PRE = r"(?:你|你们|能|可以|帮我|给我|来|想|要)*"
_SONG_STRONG_RE = re.compile(rf"^{_SONG_PRE}\s*{_SONG_VERB_STRONG}\s*[:：]?\s*(.*)$")
_SONG_WEAK_RE = re.compile(rf"^{_SONG_PRE}\s*{_SONG_VERB_WEAK}\s*[:：]?\s*(.*)$")
_SONG_ANY_RE = re.compile(rf"^{_SONG_PRE}\s*(?:{_SONG_VERB_STRONG}|{_SONG_VERB_WEAK})\s*[:：]?\s*(.*)$")
# 歌名尾部语气词/标点/书名号（"春日影吧""《killkiss》！"）
_SONG_TAIL_RE = re.compile(r"[\s《》「」\"'“”‘’。，,！!？?~～…、：:;；吧呗吗呢呀啊哦噢啦咧]+$")
_SONG_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
# 问"会唱什么/歌单"
_SONG_LIST_RE = re.compile(r"(会唱(什么|哪些|啥)|歌单|曲库|会哪些歌|有什么歌|会啥歌)")


def _clean_song_name(name: str) -> str:
    """歌名去《》/引号/尾部语气词与标点，保留原大小写用于展示。"""
    return _SONG_TAIL_RE.sub("", (name or "").strip().strip("《》「」\"'“”‘’")).strip()


def _norm_song_name(name: str) -> str:
    """归一化用于匹配：小写 + 去空白（KiLLKiSS / kill kiss / KillKiSS 都能对上）。"""
    return re.sub(r"\s+", "", _clean_song_name(name).lower())


def _iter_song_files():
    """曲库文件：covers/ 优先（她的翻唱同名时先给），再给原曲。"""
    for d in (COVERS_DIR, SONGS_DIR):
        try:
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                if os.path.isfile(p) and f.lower().endswith(_SONG_EXTS):
                    yield p
        except Exception:
            continue


def _find_song(name: str) -> str | None:
    """按归一化歌名找文件：精确 > 前缀 > 子串（大小写不敏感）。"""
    want = _norm_song_name(name)
    if not want:
        return None
    best = None
    for p in _iter_song_files():
        base = _norm_song_name(os.path.splitext(os.path.basename(p))[0])
        if base == want:
            return p
        if best is None and (base.startswith(want) or want in base):
            best = p
    return best


def song_titles() -> List[str]:
    """歌单（去重）：covers 里的同名翻唱会覆盖展示名。"""
    seen, out = set(), []
    for p in _iter_song_files():
        t = os.path.splitext(os.path.basename(p))[0]
        key = _norm_song_name(t)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _match_song(raw: str):
    """点歌识别。返回 (歌名, 文件路径或None, 是否强前缀) / ("__list__", None, True) / None。
    - "点歌 春日影"、"唱一首 killkiss"、"我想听迷星叫" → ("春日影", 路径, True)
    - "会唱什么" / "歌单" → ("__list__", None, True)
    - 只说了"点歌"没给名字 → ("", None, True)（由调用方回"想听哪首"）
    """
    s = (raw or "").strip()
    if _SONG_LIST_RE.search(s):
        return ("__list__", None, True)
    strong = _SONG_STRONG_RE.match(s)
    if strong:
        name, explicit = strong.group(1).strip(), True
    else:
        weak = _SONG_WEAK_RE.match(s)
        if not weak:
            return None
        name, explicit = weak.group(1).strip(), False
    name = _clean_song_name(name)
    if not name:
        return ("", None, explicit)
    return (name, _find_song(name), explicit)


async def _send_song(bot: Bot, event: GroupMessageEvent, file_path: str,
                     title: str | None = None, lead: str | None = None) -> None:
    """发歌：她的一句短引 + 音频（record 段，QQ 里直接播放）。
    注意：与语音回复一样不能带 reply 引用段，且必须 array 格式（详见 _send_voice 注释）。"""
    try:
        fname = os.path.basename(file_path)
        shown = title or os.path.splitext(fname)[0]
        line = lead or random.choice([
            f"……嗯。《{shown}》。",
            f"……好。《{shown}》。",
            f"……唱了。《{shown}》。",
        ])
        msg_array = [
            {"type": "text", "data": {"text": line}},
            {"type": "record", "data": {"file": file_path.replace("\\", "/")}},
        ]
        sent = await bot.send_group_msg(group_id=int(event.group_id), message=msg_array)
        _record_sent_id(event, sent)
        print(f"[SONG-SENT] group={event.group_id} file={fname}", flush=True)
    except Exception as e:
        print(f"[SONG-SEND-ERROR] {type(e).__name__}: {e}", flush=True)
        await _safe_finish(event, f"……《{os.path.splitext(os.path.basename(file_path))[0]}》好像唱不了了。")


def _songbook_reply(limit: int = 10) -> str:
    """歌单文本（她的口吻，≤3 行）。"""
    titles = song_titles()
    if not titles:
        return "……还不会唱。"
    shown = "、".join(titles[:limit])
    more = f"…等{len(titles)}首" if len(titles) > limit else ""
    return f"……会这些：\n{shown}{more}"


# ============ 每日娃娃：帮群友 @ 那个娃娃 bot 触发它的命令（2026-10-17 用户要求）============
# 群里有个娃娃 bot（默认 200000005，昵称"短发麻里奈"）：@它发 "/今日邦娃"、"/今日棉娃"
# 就回一张每日娃娃图，是群里的小仪式。群友让若叶睦"每天这样发"——她要配合：帮他们把这条
# 命令 @ 出去，而不是拒绝（此前她的回复是"……不要。/……不发。"）。
# 2026-10-19 实锤：转发本身一次就成功（对面当场回图），但当时 180 秒冷却把群友
# "你还要再说一次"的请求挡成了静默，群友以为她不会用（"诶，咋没反应了"）。
# 现在：冷却只防连点（15 秒），冷却中也要说一句"刚叫过"（不能装死）。
# 2026-10-20 用户要求每天限一次；2026-10-23 更正为账号级一次。2026-10-26 用户反馈
# “还是不会艾特”后撤销限制：娃娃 bot 实测每次 /今日X 都回图（10-26 09:20-09:28 连续
# 3 次邦娃+2 次棉娃全成功），10-22 的“每天一次”观察已失效；仅保留 15 秒冷却防连点。
DOLL_BOT_QQ = os.getenv("DOLL_BOT_QQ", "200000005")
DOLL_RELAY_COOLDOWN = float(os.getenv("DOLL_RELAY_COOLDOWN", "15"))
_doll_relay_last: Dict[str, float] = {}
# 娃娃指令词（邦娃/棉娃/邦布…）与"这是在让她发"的请求口气：
# 被 @/被引用/提到她时，出现指令词就算（她在被要求）；没被叫到时，要更明确的请求口气
# （"你每天都要这样发…""帮我发个…"），纯闲聊提到"邦娃 bot 又坏了"不算。
_DOLL_WORD_RE = re.compile(r"(邦娃|棉娃|邦布|fufu|fumo)", re.I)
_DOLL_ASK_RE = re.compile(r"(你(们)?(也|都|要|来|发|去|帮|每天|每日|以后)?|帮(我|我们|忙)|给我|"
                          r"来(个|张|一下|一条)|发(个|张|一下|一条)|每天|每日|这样发)")
# 转发用的命令词：请求里说"棉娃"就发棉娃，否则默认邦娃
_DOLL_CMD_BY_WORD = {"棉娃": "棉娃", "邦布": "邦娃", "fufu": "邦娃", "fumo": "邦娃"}

# 娃娃 bot 按账号每天只能抽一次（邦娃/棉娃合计；跨重启持久化：重启不能刷新次数）。
_DOLL_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doll_relay_state.json")
_doll_daily: Dict[str, Dict[str, object]] = {}   # group -> {"date": "YYYY-MM-DD", "done": ["邦娃", ...]}


def _doll_today() -> str:
    return time.strftime("%Y-%m-%d")


def _load_doll_daily() -> None:
    try:
        with open(_DOLL_STATE_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        for g, d in (raw or {}).items():
            if not isinstance(d, dict):
                continue
            done = [str(x) for x in (d.get("done") or []) if str(x) in ("邦娃", "棉娃")]
            _doll_daily[str(g)] = {"date": str(d.get("date") or ""), "done": done}
        if _doll_daily:
            print(f"[DOLL-RELAY] 每日已抽状态: {_doll_daily}", flush=True)
    except Exception:
        pass


def _save_doll_daily() -> None:
    try:
        today = _doll_today()
        out = {g: d for g, d in _doll_daily.items() if d.get("date") == today}
        tmp = _DOLL_STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
        os.replace(tmp, _DOLL_STATE_FILE)
    except Exception as e:
        print(f"[DOLL-RELAY] 每日状态保存失败: {e}", flush=True)


def _doll_already_drawn(group_id: str, cmd_word: str) -> bool:
    """（2026-10-26 停用，不再调用）原每日一次判断；保留函数与状态记录以兼容旧数据与测试。"""
    today = _doll_today()
    # 娃娃 bot 按 QQ 账号计次，转发都是用若叶睦的账号发，所以别的群抽过也算今天用掉了
    if any(st.get("date") == today and st.get("done") for st in _doll_daily.values()):
        return True
    st = _doll_daily.get(group_id)
    if not st or st.get("date") != today:
        _doll_daily[group_id] = {"date": today, "done": []}
    return False


def _doll_mark_drawn(group_id: str, cmd_word: str) -> None:
    today = _doll_today()
    st = _doll_daily.get(group_id)
    if not st or st.get("date") != today:
        st = {"date": today, "done": []}
        _doll_daily[group_id] = st
    done = st.get("done")
    if not isinstance(done, list):
        done = []
        st["done"] = done
    if cmd_word not in done:
        done.append(cmd_word)
    _save_doll_daily()


def _doll_undo_drawn(group_id: str, cmd_word: str) -> None:
    """转发失败时撤销当天的已抽标记，避免白白用掉一次机会。"""
    st = _doll_daily.get(group_id)
    done = st.get("done") if isinstance(st, dict) else None
    if isinstance(done, list) and cmd_word in done:
        done.remove(cmd_word)
        _save_doll_daily()


_load_doll_daily()


def _doll_cmd_word(raw: str, addressed: bool = False) -> str | None:
    """消息是不是"让若叶睦去叫娃娃 bot"：命中返回要发的那条命令词（邦娃/棉娃），否则 None。
    addressed=True（被 @/被引用/被提到）→ 只要出现指令词就认；否则要明确的请求口气。"""
    m = _DOLL_WORD_RE.search(raw or "")
    if not m:
        return None
    if not (addressed or _DOLL_ASK_RE.search(raw or "")):
        return None
    return _DOLL_CMD_BY_WORD.get(m.group(1).lower(), "邦娃")


async def _relay_doll_cmd(bot: Bot, event: GroupMessageEvent, cmd_word: str) -> bool:
    """帮群友 @ 娃娃 bot 发指令：一条 [@娃娃bot /今日X] 的纯命令消息（别加别的内容，
    那条 bot 按整条消息解析命令）。并记录 message_id 供"被引用"检测。
    返回是否真的发出去了；发送失败时调用方要撤销当天的已抽标记。"""
    try:
        msg = Message([MessageSegment.at(DOLL_BOT_QQ),
                       MessageSegment.text(f" /今日{cmd_word}")])
        sent = await bot.send_group_msg(group_id=int(event.group_id), message=msg)
        _record_sent_id(event, sent)
        print(f"[DOLL-RELAY] group={event.group_id} -> @{DOLL_BOT_QQ} /今日{cmd_word}", flush=True)
        return True
    except Exception as e:
        print(f"[DOLL-RELAY-ERROR] {type(e).__name__}: {e}", flush=True)
        return False


def _match_special(raw: str, group_id: str = "", use_mortis: bool = False) -> str | None:
    """匹配固定接梗规则（含别名），命中随机选一条文案，否则返回 None。
    若该群禁用了某人物，跳过对应规则。
    若当前是墨提斯人格（use_mortis=True），跳过仅限若叶睦的人物印象类规则（mutsumi_only），
    让墨提斯用AI对话自然回应，避免若叶睦风格的固定接梗在墨提斯场景"插嘴"。
    用严格匹配（精确子串+显式别名），避免人名之间近形/谐音误伤。
    """
    raw_lower = raw.lower()
    banned = BANNED_MENTIONS.get(int(group_id)) if str(group_id).lstrip("-").isdigit() else None
    banned_kw = BANNED_KEYWORDS.get(int(group_id)) if str(group_id).lstrip("-").isdigit() else None
    for rule in SPECIAL_RULES:
        # 该规则涉及的人物若在本群被禁，跳过
        if banned and any(kw in banned for kw in rule["keywords"]):
            continue
        # 该规则的关键词在本群被禁，跳过
        if banned_kw and any(kw in banned_kw for kw in rule["keywords"]):
            continue
        # 墨提斯人格时跳过若叶睦专属印象规则
        if use_mortis and rule.get("mutsumi_only"):
            continue
        if any(_raw_contains_name_strict(raw_lower, kw) for kw in rule["keywords"]):
            # 概率化：不每条都硬甩固定文案（避免"提到某个梗就复读固定文案"的刻板感）。
            # 默认 50% 用固定文案，50% 走 LLM 自然对话；群规等权威内容 prob=1.0 必回
            if random.random() < rule.get("prob", 0.5):
                return _pick_reply(rule)
            # 概率未中：继续查下一条规则，全不中则走正常 LLM 流程
            continue
    return None


# 2026-10-22：固定接梗防刷屏。线上实锤（10-21 12:48–12:50）：另一个 bot 把同一段提到「高松灯」的
# roleplay 长文连发 5 遍，她跟着把「灯，比我更会表达。」发了 4 遍（90 秒内 6 次 [SPECIAL]）。
# 这里只限制「未被叫」（related 触发）的接梗频率；被 @/引用/提名字的接梗不受影响。
_SPECIAL_RELATED_LAST: Dict[str, float] = {}
_SPECIAL_RELATED_COOLDOWN = 60.0


def _match_prob(raw: str, user_id: str) -> str | None:
    """匹配概率型接梗规则（含别名）：命中关键词或指定用户，按概率决定是否回复。"""
    raw_lower = raw.lower()
    for rule in PROB_RULES:
        if any(_raw_contains_name(raw_lower, kw) for kw in rule["keywords"]) and random.random() < rule["prob"]:
            return _pick_reply(rule)
    ur = PROB_USER_RULES.get(str(user_id))
    if ur and random.random() < ur["prob"]:
        return _pick_reply(ur)
    return None


# ============ 网络热梗/邦多利梗库（2026-10-16）============
# 群里玩梗（网络黑话 + 邦多利/MyGO 梗）时，把命中词的解释注入给她，让她"看懂再接"：
# 接得上就用 ≤15 字短句接（带自己的看法），接不上/不符合人设就【沉默】；
# 绝不解释梗本身给群友听。梗表在 memes.json（可随时加）。
MEMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memes.json")
_memes_cache: Optional[List[dict]] = None


def _load_memes() -> List[dict]:
    """加载梗库（进程内缓存一次）。失败返回空表，不影响主流程。"""
    global _memes_cache
    if _memes_cache is None:
        try:
            with open(MEMES_PATH, encoding="utf-8") as f:
                _memes_cache = json.load(f).get("memes", []) or []
            print(f"[MEME] 梗库加载 {len(_memes_cache)} 条", flush=True)
        except Exception as e:
            print(f"[MEME] 梗库加载失败 {type(e).__name__}: {e}", flush=True)
            _memes_cache = []
    return _memes_cache


def _meme_hint(raw: str, limit: int = 3) -> str:
    """消息里出现的梗 -> 注入块（只给她看；最多 limit 条）。没命中返回 ""。
    触发词要求 ≥2 字符，避免单字（乐/草/典/6）满地误命中。"""
    if not raw:
        return ""
    low = raw.lower()
    hits: List[tuple] = []
    for m in _load_memes():
        keys = [k for k in (m.get("keys") or []) if len(k) >= 2]
        matched = next((k for k in keys if k.lower() in low), None)
        if matched:
            hits.append((matched, m))
            if len(hits) >= limit:
                break
    if not hits:
        return ""
    lines = []
    for matched, m in hits:
        mean = (m.get("mean") or "").strip()
        reps = [r for r in (m.get("reply") or []) if r]
        # 标注"实际命中的那个词"（而不是条目首词），她会更清楚群友在玩什么梗
        seg = f"- 「{matched}」：{mean}" if mean else f"- 「{matched}」"
        if reps:
            seg += "｜可参考的接法：" + " / ".join(reps[:2])
        lines.append(seg)
    return ("【群里提到的梗（你懂这些，别装不懂，但也别科普给群友听）】\n" + "\n".join(lines) +
            "\n处理原则：先判断这条是不是冲你来的、接不接得上、符不符合你的说话方式；"
            "都满足就用≤15字的短句接（带点自己的态度或复述关键词）；"
            "有一条不满足就【沉默】。绝对不要解释梗的意思、不要背设定。")


def _match_meme(raw: str) -> str | None:
    """匹配原作梗接梗表：命中关键词就自然接梗。
    用严格匹配（精确子串），避免"你是谁"的"谁"字误触发人物谐音接梗。
    概率化：默认 60% 用固定梗文案，40% 走 LLM 自然对话（避免刻板复读）。
    """
    raw_lower = raw.lower()
    for rule in MEME_RULES:
        if any(kw.lower() in raw_lower for kw in rule["keywords"]):
            if random.random() < rule.get("prob", 0.6):
                return _pick_reply(rule)
            continue
    return None


# 「她的事」关键词（2026-10-16 用户要求："对话中提到若叶睦及相关的也应该会回复"）。
# 群里聊到若叶睦身边的人/原作内容也算"在聊她的事"，要接一句（走保守回复规则 + 守冷却节流）。
# 只列原作里的人名/作品，不列群友名字（群友名不算她的相关话题）。
RELATED_TOPIC_KEYWORDS = (
    "长崎爽世", "丰川祥子", "高松灯", "椎名立希", "三角初华", "八幡海铃", "祐天寺若麦",
    "森美奈美", "若叶隆文",
    "春日影", "一辈子乐队", "母鸡卡", "ave mujica", "avemujica", "阿贝姆吉卡", "迷子", "迷星叫",
)


def _match_related_topic(raw: str) -> bool:
    """消息是否在聊若叶睦相关的人/原作（祥子、爽世、Ave Mujica、春日影…）。
    用严格匹配 + 别称展开（_raw_contains_name_strict 会查 NAME_ALIASES：
    如"丰川祥子"同时覆盖"祥子/小祥/sakiko"，"森美奈美"覆盖"睦的妈妈"）。"""
    return any(_raw_contains_name_strict(raw, kw) for kw in RELATED_TOPIC_KEYWORDS)


def _check_trigger(event: GroupMessageEvent, state: T_State) -> bool:
    """触发规则：总开关开启、仅群消息、非机器人自己发的消息、且在白名单群内。
    对方机器人（薇欧拉）的消息一律忽略：防止互相引用/提到导致无限循环对话。"""
    if not isinstance(event, GroupMessageEvent):
        return False
    gid = str(event.group_id)
    active = time.time() < dialogue_until.get(gid, 0)
    allowed, consumed = chat_logic.should_trigger(
        enabled=ENABLE_SPEAKING,
        is_group=True,
        user_id=event.user_id,
        self_id=event.self_id,
        other_bot_qq=OTHER_BOT_QQ,
        astra_bot_qq=ASTRA_BOT_QQ,
        group_id=gid,
        allowed_groups=ALLOWED_GROUPS,
        dialogue_active=active,
        dialogue_turns=dialogue_turns.get(gid, 0),
        dialogue_max_turns=DIALOGUE_MAX_TURNS,
    )
    if consumed:
        dialogue_turns[gid] = dialogue_turns.get(gid, 0) + 1
    return allowed


# ============ 图片/视频识别 ============
import urllib.request


import re as _re_url

# 视频/图片链接识别：只认真正的 URL（含域名/路径），避免"video/mp4"等字样误伤普通文字
_URL_RE = _re_url.compile(r"https?://[^\s]+|www\.[^\s]+|b23\.tv/[^\s]+")


def _has_media(raw_msg: str, event: GroupMessageEvent) -> bool:
    """检测消息是否含图片/视频段（语音、文件不算，避免误判为'丑图'）。"""
    for seg in event.message:
        if seg.type in ("image", "video"):
            return True
    low = raw_msg.lower()
    if _URL_RE.search(low):
        return True
    return False


def _first_image_url(raw_msg: str, event: GroupMessageEvent) -> str | None:
    """取消息里第一张图的**可下载** URL。
    2026-10-02 实锤：gchat 带 rkey 的直链有防盗链（200 假成功但只给占位图），
    必须用文件名 md5 走 gchatpic_new 旧接口才能拿到真图；这里统一转换。"""
    d = _first_image_data(raw_msg, event)
    if not d:
        return None
    lg = sticker_mod.legacy_url(str(d.get("file") or ""))
    if lg:
        return lg
    u = str(d.get("url") or "")
    if u.startswith("http"):
        import html as _html
        return _html.unescape(u)
    return None


def _first_image_data(raw_msg: str, event: GroupMessageEvent) -> dict:
    """当前消息里第一张图的 data（file + url，供三通道兜底识图）。没有返回 {}。"""
    for seg in event.message:
        if seg.type == "image":
            d = dict(seg.data or {})
            if d.get("file") or d.get("url"):
                return d
    if raw_msg:
        for d in sticker_mod._parse_cq_images(raw_msg):
            if d.get("file") or d.get("url"):
                return d
    return {}


def _first_video_data(event: GroupMessageEvent) -> dict:
    """当前消息里第一个 video 段的 data（交给 video_frames 抽帧）。没有返回 {}。"""
    for seg in event.message:
        if seg.type == "video":
            return dict(seg.data or {})
    return {}


# ============ claw 化：她会自己"看"群友引用的消息（2026-10-10）============
# 群友回复某条消息时会带引用；以前她只能看到引用标记、看不到内容（"不会自己看"）。
# 现在她用 NapCat 的只读接口取回引用/转发的内容再回答。
# 安全边界：只调 get_msg / get_forward_msg / get_file（只读、仅本群消息）；总量截断；超时放弃。
# 2026-10-10 晚补：引用里的**视频**也能看——get_file 落盘 → ffmpeg 抽帧 → 视觉描述。

def _cap_text(s: str, limit: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= limit else s[:limit] + "…"


# 视觉预算：单条引用/整条转发记录共用 QUOTE_VISION_CALLS 次视觉调用（图 1 次、视频每帧 1 次）
QUOTE_VISION_CALLS = int(os.getenv("QUOTE_VISION_CALLS", "2"))
# 单张引用图：下载+描述总超时。2026-10-10 实锤 15s 太紧——思考型视觉模型单次要 6~15s，
# 再叠加图片下载就超时 → 回退"[图片]" → 她回"看不到图"（群里实锤过一次）。
# 同日再加"认角色"要求后，难图（颜艺/模糊）思考可达 30s+，提到 45s。
QUOTE_IMAGE_TIMEOUT = float(os.getenv("QUOTE_IMAGE_TIMEOUT", "45"))
# 单张引用图的**总**预算（含 URL 通道 + get_file 兜底通道，见 _describe_image_sdata）
QUOTE_IMAGE_BUDGET = float(os.getenv("QUOTE_IMAGE_BUDGET", "60"))
# 引用视频：get_file 落盘超时（NapCat 秒回）/ 抽帧描述总预算 / 单帧视觉调用超时
QUOTE_VIDEO_FETCH_TIMEOUT = float(os.getenv("QUOTE_VIDEO_FETCH_TIMEOUT", "15"))
QUOTE_VIDEO_DESC_BUDGET = float(os.getenv("QUOTE_VIDEO_DESC_BUDGET", "35"))
QUOTE_VIDEO_FRAME_TIMEOUT = float(os.getenv("QUOTE_VIDEO_FRAME_TIMEOUT", "25"))


def _image_urls_from_seg(sdata: dict) -> List[str]:
    """图片段 -> 候选可下载 URL 列表（按优先级）：
    ① gchatpic_new 旧接口（免 rkey，但对部分图会 404）；
    ② 原始 rkey 直链（可能拿到防盗链占位图，vision 侧会识别并判失败）。
    实锤（2026-10-10 21:37）：群友 @ 她"翻译这个图片里的韩语"，旧接口 404、
    rkey 直链只给 3691B 占位图 → 描述失败，她拿到"[图片]"没东西可翻，直接【沉默】。"""
    urls: List[str] = []
    lg = sticker_mod.legacy_url(str(sdata.get("file") or ""))
    if lg:
        urls.append(lg)
    u = str(sdata.get("url") or "")
    if u.startswith("http"):
        import html as _html
        u = _html.unescape(u)
        if u not in urls:
            urls.append(u)
    return urls


async def _describe_image_sdata(bot: Optional[Bot], sdata: dict,
                                timeout: float = QUOTE_IMAGE_TIMEOUT) -> str:
    """图片段 -> 中文描述，三条通道依次兜底（全失败返回 ""）：
    ①/② URL 通道（旧接口 / rkey 直链）；
    ③ NapCat get_file 落盘 + 本地识图——rkey 过期、404、拿到占位图时的稳定通道
      （图片消息本体早已在 NapCat 本地缓存，实测直接读出韩文截图内容）。
    单图总预算 QUOTE_IMAGE_BUDGET 秒（默认 60），每次尝试按剩余时间收缩。"""
    deadline = time.time() + QUOTE_IMAGE_BUDGET
    for url in _image_urls_from_seg(sdata):
        left = deadline - time.time()
        if left < 5:
            break
        try:
            desc = await asyncio.wait_for(
                asyncio.to_thread(vision.describe_image, url),
                timeout=min(timeout, left)) or ""
        except Exception as e:
            print(f"[QUOTE] 引用图描述失败 {type(e).__name__}: {e}", flush=True)
            desc = ""
        if desc:
            print(f"[QUOTE] 引用图描述: {desc[:50]!r}", flush=True)
            return desc
    token = str(sdata.get("file") or "")
    if token and bot is not None:
        left = deadline - time.time()
        if left >= 5:
            try:
                res = await asyncio.wait_for(
                    bot.call_api("get_file", file=token),
                    timeout=min(QUOTE_VIDEO_FETCH_TIMEOUT, left))
                p = str((res or {}).get("file") or "")
                if p and os.path.isfile(p):
                    desc = await asyncio.wait_for(
                        asyncio.to_thread(vision.describe_file, p),
                        timeout=min(timeout, max(5.0, deadline - time.time()))) or ""
                    if desc:
                        print(f"[QUOTE] 引用图走 NapCat 本地文件识图成功 "
                              f"({os.path.basename(p)}): {desc[:50]!r}", flush=True)
                        return desc
            except Exception as e:
                print(f"[QUOTE] get_file 兜底识图失败 {type(e).__name__}: {e}", flush=True)
    return ""


async def _local_video_path(bot: Bot, sdata: dict) -> Tuple[str, bool]:
    """video 段 -> 本地可读视频文件路径。返回 (路径, 是否我们下载的临时文件)。
    实锤（2026-10-10 21:55）：NapCat 的 video 段只给 file token 和**本地路径** url，
    而 QQ 客户端收到的视频默认不落盘（nt_data/Video/*/Ori 是空的，只有 Thumb 封面），
    所以这里尽力而为：get_file 的 file/url、段里的 url/path、http 直链，逐个检查存在性。
    全失败返回 ("", False)——调用方会退到封面图（_video_cover_path）。"""
    token = str(sdata.get("file") or "")
    if token and bot is not None:
        try:
            res = await asyncio.wait_for(
                bot.call_api("get_file", file=token), timeout=QUOTE_VIDEO_FETCH_TIMEOUT)
        except Exception as e:
            print(f"[QUOTE] get_file 失败 {type(e).__name__}: {e}", flush=True)
            res = None
        res = res or {}
        for k in ("file", "url"):
            p = str(res.get(k) or "")
            if p and os.path.isfile(p):
                return p, False
        u = str(res.get("url") or "")
        if u.startswith("http"):
            dl = await asyncio.to_thread(video_frames.download_video, u)
            if dl:
                return dl, True
    for k in ("url", "path"):
        p = str(sdata.get(k) or "")
        if p and os.path.isfile(p):
            return p, False
    u2 = str(sdata.get("url") or "")
    if u2.startswith("http"):
        dl = await asyncio.to_thread(video_frames.download_video, u2)
        if dl:
            return dl, True
    print(f"[QUOTE] 视频取不到可读文件（token={token[:40]!r}，等待封面兜底）", flush=True)
    return "", False


def _video_cover_path(sdata: dict) -> str:
    """video 段 -> 本地封面缩略图路径（QQ 收到的视频一定有封面，原片却不一定在本地）。
    从段里 url（...\\nt_data\\Video\\2026-10\\Ori\\<md5>.mp4）推导
    ...\\nt_data\\Video\\2026-10\\Thumb\\<md5>_0.png；月目录对不上就按 md5 前缀全局找。"""
    p = str(sdata.get("url") or "") or str(sdata.get("path") or "")
    q = p.replace("\\", "/")
    if "nt_data" not in q or "/" not in q:
        return ""
    base = q.rsplit("/", 1)[-1]
    md5 = base.rsplit(".", 1)[0]
    if not md5:
        return ""
    parts = q.split("/")
    root = ""
    if "Ori" in parts:
        root = "/".join(parts[:parts.index("Ori")])
    elif "Video" in parts:
        root = "/".join(parts[:parts.index("Video") + 1])
    cands = []
    if root:
        cands.append(f"{root}/Thumb/{md5}_0.png".replace("/", "\\"))
    # 月目录可能对不上：全局按 md5 前缀找 Thumb
    try:
        vroot = p.replace("\\", "/").split("/nt_data/Video/")
        if len(vroot) == 2:
            import glob
            for f in glob.glob(os.path.join(vroot[0].replace("/", "\\"), "nt_data", "Video", "*", "Thumb", f"{md5}*")):
                cands.append(f)
    except Exception:
        pass
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return ""


async def _describe_video_segment(bot: Bot, sdata: dict, vision_budget: List[int]) -> str:
    """video 段 -> "视频12秒：画面1描述；画面2描述"（无描述返回 ""），外层再加方括号。
    原片拿不到时（QQ 收到的视频默认不落盘）退到**本地封面缩略图**：
    "视频（只见封面帧）：…"。预算先行扣减，失败也不退，避免慢调用并发改预算。"""
    if vision_budget[0] <= 0:
        return ""
    path, temp_dl = await _local_video_path(bot, sdata)
    frames: List[str] = []
    if path:
        try:
            n = min(2, vision_budget[0])
            vision_budget[0] -= n
            frames, dur = await asyncio.wait_for(
                asyncio.to_thread(video_frames.extract_frames, path, n),
                timeout=QUOTE_VIDEO_FETCH_TIMEOUT + 15)
            if frames:
                descs, _used = await asyncio.wait_for(
                    asyncio.to_thread(video_frames.describe_frames, frames,
                                      time.time() + QUOTE_VIDEO_DESC_BUDGET,
                                      QUOTE_VIDEO_FRAME_TIMEOUT),
                    timeout=QUOTE_VIDEO_DESC_BUDGET + 10)
                if descs:
                    label = f"视频{int(dur)}秒" if dur else "视频"
                    body = "；".join(_cap_text(d, 60) for d in descs[:2])
                    print(f"[QUOTE] 引用视频描述({label}/共{len(descs)}帧): {body[:60]!r}", flush=True)
                    return f"{label}：{body}"
        except Exception as e:
            print(f"[QUOTE] 引用视频描述失败 {type(e).__name__}: {e}", flush=True)
        finally:
            video_frames.cleanup_frames(frames)
            if temp_dl:
                video_frames.cleanup_file(path)
    # 封面兜底：QQ 一定有视频封面缩略图（本地），至少能让她知道视频画面长什么样
    cover = _video_cover_path(sdata)
    if cover and vision_budget[0] > 0:
        vision_budget[0] -= 1
        try:
            cdesc = await asyncio.wait_for(
                asyncio.to_thread(vision.describe_file, cover),
                timeout=QUOTE_VIDEO_FRAME_TIMEOUT) or ""
        except Exception as e:
            print(f"[QUOTE] 视频封面描述失败 {type(e).__name__}: {e}", flush=True)
            cdesc = ""
        if cdesc:
            print(f"[QUOTE] 引用视频只见封面({os.path.basename(cover)}): {cdesc[:55]!r}", flush=True)
            return f"视频（原片没下到本地，只见封面帧）：{_cap_text(cdesc, 70)}"
    return ""


async def _segments_to_text_vision(segs, limit: int, vision_budget: List[int],
                                   bot: Optional[Bot] = None) -> str:
    """消息段数组 -> 可读文本，图片/视频交给云端视觉变成描述：
    - 图 → "[图：一只橘猫趴在键盘上]"（下载+描述 35s 超时，失败回退"[图片]"）；
    - 视频 → get_file 落盘 + ffmpeg 抽帧 → "[视频12秒：画面1…；画面2…]"（失败回退"[视频]"）；
    - 预算 vision_budget 是单元素可变列表（跨段/跨转发节点共享），图扣 1、视频每帧扣 1。
    实锤沿革（2026-10-10）：她原先只看到"[图片]"；加识图后又被 15s 超时和
    512 token 预算（被思考吃光，见 vision.py）坑成"看不到"——现在两者都已修。"""
    parts: List[str] = []
    for seg in segs or []:
        if isinstance(seg, dict):
            stype, sdata = seg.get("type"), (seg.get("data") or {})
        else:
            stype, sdata = getattr(seg, "type", ""), (getattr(seg, "data", None) or {})
        if stype == "text":
            parts.append(str(sdata.get("text", "")))
        elif stype == "image":
            desc = ""
            if vision_budget[0] > 0:
                vision_budget[0] -= 1
                desc = await _describe_image_sdata(bot, sdata)
            if desc:
                parts.append(f"[图：{_cap_text(desc, 60)}]")
            else:
                parts.append("[图片]")
        elif stype == "video":
            inner = await _describe_video_segment(bot, sdata, vision_budget)
            parts.append(f"[{inner}]" if inner else "[视频]")
        elif stype == "face":
            parts.append("[表情]")
        elif stype == "at":
            parts.append("@")
    return _cap_text("".join(parts), limit)


_UNREAD_HINT = ("（他引用的图/视频没读出来，看不到里面的内容；他如果在问图里的东西，"
                "就直说没看到、让他再发一次，不要硬答）")


def _with_unreadable_hint(txt: str) -> str:
    """引用内容只有占位符（[图片]/[视频]，没有任何描述）时附一句提示——
    否则她拿着"[图片]"没东西可答，会直接【沉默】，对方以为被无视
    （2026-10-10 21:37 实锤：@ 她"翻译这个图片里的韩语" → 沉默 → 群友问为什么不回答）。"""
    if not txt:
        return txt
    has_placeholder = "[图片]" in txt or "[视频]" in txt
    has_desc = re.search(r"\[图：|\[视频[^\]]*：", txt)
    if has_placeholder and not has_desc:
        return txt + _UNREAD_HINT
    return txt


async def _fetch_quoted_content(bot: Bot, event: GroupMessageEvent) -> str:
    """取消息里引用/转发的可读内容并返回（失败返回空串，绝不打断主流程）。"""
    ref_id = ""
    fwd_id = ""
    for seg in event.message:
        if seg.type == "reply":
            ref_id = str(seg.data.get("id", "") or "")
        elif seg.type in ("forward", "json", "xml"):  # 合并转发/卡片消息
            if seg.type == "forward":
                fwd_id = str(seg.data.get("id", "") or "")
    raw_msg = getattr(event, "raw_message", "") or ""
    if not ref_id:
        m = re.search(r"\[CQ:reply,id=([^\]]+)\]", raw_msg)
        if m:
            ref_id = m.group(1)
    if not fwd_id:
        m2 = re.search(r"\[CQ:forward,id=([^\],]+)", raw_msg)
        if m2:
            fwd_id = m2.group(1)
    try:
        if ref_id and ref_id.isdigit():
            data = await asyncio.wait_for(
                bot.call_api("get_msg", message_id=int(ref_id)), timeout=6)
            txt = await _segments_to_text_vision(
                (data or {}).get("message") or [], 600, [QUOTE_VISION_CALLS], bot=bot)
            if txt:
                print(f"[QUOTE] 读到引用消息 {len(txt)} 字", flush=True)
            return _with_unreadable_hint(txt)
        if fwd_id:
            try:
                data = await asyncio.wait_for(
                    bot.call_api("get_forward_msg", id=fwd_id), timeout=6)
            except Exception:
                data = await asyncio.wait_for(
                    bot.call_api("get_forward_msg", message_id=fwd_id), timeout=6)
            nodes = (data or {}).get("messages") or []
            lines: List[str] = []
            vision_budget = [QUOTE_VISION_CALLS]   # 整条转发记录共用一份视觉预算
            for node in nodes[:20]:
                c = node.get("message") or node.get("content") or []
                name = str((node.get("sender") or {}).get("nickname") or "")
                t = await _segments_to_text_vision(c, 150, vision_budget, bot=bot)
                if t:
                    lines.append(f"{name}：{t}" if name else t)
            joined = _cap_text("\n".join(lines), 800)
            if joined:
                print(f"[QUOTE] 读到转发记录 {len(lines)} 条/{len(joined)} 字", flush=True)
            return _with_unreadable_hint(joined)
    except Exception as e:
        print(f"[QUOTE] 取引用内容失败 {type(e).__name__}: {e}", flush=True)
    return ""


chat_matcher = on_message(rule=Rule(_check_trigger), priority=10, block=False)


def _handle_cucumber(gid: str, raw: str, uid: str, name: str = "") -> str | None:
    """黄瓜农场命令处理（gid=当前群号；黄瓜/好感度按群独立记账，别的群的数据不参与计算）。
    命中返回回复，否则返回 None。"""
    low = raw.lower().strip()
    # 签到
    if low in ("签到", "种黄瓜", "签到种黄瓜", "签到得黄瓜"):
        return cucumber.sign_in(gid, uid, name)
    # 仓库
    if low in ("仓库", "我的仓库", "我的黄瓜", "黄瓜仓库"):
        return cucumber.warehouse(gid, uid, name)
    # 排行榜
    if low in ("排行榜", "黄瓜排行榜", "排名", "黄瓜排名"):
        return cucumber.ranking(gid)
    # 猜大小：猜大小 5（押注5根，等引用回复大/小）。用正则全匹配，避免"猜大小便"等误伤
    m = re.fullmatch(r"(?:猜|押)大小\s*(\d+)?", low)
    if m:
        if m.group(1):
            return cucumber.start_guess(gid, uid, int(m.group(1)), name)
        return "格式：猜大小 5（押5根小黄瓜，然后引用我的消息回复大或小）"
    # 猜数字：猜数字（开始，等引用回复）
    if low in ("猜数字", "猜数"):
        return cucumber.start_number(gid, uid, name)
    # 木子米今天吃什么（随机菜品，人设口吻）。整句匹配，避免"他吃什么过敏"这类误伤
    if re.fullmatch(r"(今天|晚上|中午|早上|午饭|晚饭|夜宵|你|木子米|若叶睦)?(吃什么|吃啥|吃了什么|吃了啥)", low):
        return cucumber.eat_today()
    # 菜单
    if low in ("菜单", "菜品", "菜谱", "木子米的菜单", "今天有什么菜"):
        return cucumber.menu()
    # 好感度
    if low in ("好感度", "好感", "亲密", "亲密值", "木子米好感度"):
        return cucumber.affection(gid, uid, name)
    # 黄瓜互兑：3 小黄瓜 = 1 大黄瓜（兑换黄瓜 大 / 兑换黄瓜 小 [数量]）
    m = re.fullmatch(r"(?:兑换黄瓜|黄瓜兑换|换黄瓜)\s*(大|小)\s*(\d+)?", low)
    if m:
        count = int(m.group(2)) if m.group(2) else 1
        direction = "to_big" if m.group(1) == "大" else "to_small"
        return cucumber.exchange(gid, uid, direction, count, name)
    if low in ("兑换黄瓜", "黄瓜兑换", "怎么兑换黄瓜", "兑换比例"):
        return "黄瓜兑换：3 根小黄瓜 = 1 根大黄瓜。\n发：兑换黄瓜 大 [数量]（小换大）或 兑换黄瓜 小 [数量]（大换小）。"
    # 大黄瓜兑换商城
    if low in ("商城", "商店", "小卖部", "兑换商城", "兑换商店"):
        return cucumber.shop()
    m = re.match(r"(?:购买|买)\s*(.+)", low)
    if m:
        item = _match_shop_item(m.group(1))
        if item:
            return cucumber.buy(gid, uid, item, name)
    # 投喂/兑换：投喂 菜名 或 兑换 商品名/菜名（可带 木子米/若叶睦 等称呼，如"投喂木子米 拍黄瓜"）
    if any(kw in low for kw in ("投喂", "投食", "喂食", "兑换")):
        # "兑换 X"：先匹配商城商品，再匹配菜品
        item = _match_shop_item(low)
        if item:
            return cucumber.buy(gid, uid, item, name)
        dish = _match_feed_dish(low)
        if dish:
            return cucumber.feed(gid, uid, dish, name)
        # 没匹配到：只有消息很短（像投喂尝试，如"投喂""投喂一下"）才提示格式；
        # 长句（"怎么兑换积分啊"）不劫持，当普通对话处理
        if len(raw) <= 6:
            return "想投喂木子米？发'菜单'看看有什么菜，然后说：投喂 菜名（如：投喂 拍黄瓜）。发'商城'看大黄瓜商品。"
        return None
    # 引用机器人消息时给出答案：大/小/数字
    return None


def _match_shop_item(low: str) -> str | None:
    """从消息里提取商城商品名（按最长匹配）。"""
    for s in sorted(cucumber.SHOP_BY_NAME, key=len, reverse=True):
        if s in low:
            return s
    return None


def _match_feed_dish(low: str) -> str | None:
    """从投喂消息里提取菜名：去掉触发词和人名后，按最长菜名匹配。"""
    for kw in ("投喂", "投食", "喂食", "兑换"):
        low = low.replace(kw, "")
    for alias in ("木子米", "若叶睦", "小睦", "睦", "若叶"):
        low = low.replace(alias, "")
    low = low.strip().strip("，。！？、  ")
    if not low:
        return None
    for d in sorted(cucumber.DISH_BY_NAME, key=len, reverse=True):
        if d in low:
            return d
    return None


def _is_dialogue_cmd(raw: str) -> bool:
    """双bot对话指令：明确指令词，或同时提到两个 bot 且提到"对话"。"""
    low = raw.lower()
    if any(k in low for k in ("双bot对话", "双bot聊天", "两个bot对话", "两个bot聊天", "双机器人对话", "让她们对话", "让她们聊", "让她们两个聊")):
        return True
    has_v = "薇欧拉" in low or "扫货" in low
    has_m = any(k in low for k in ("若叶睦", "木子米", "小睦", "睦", "若叶"))
    return "对话" in low and has_v and has_m


# 使用说明（完备版）：发"使用说明/帮助"时回复
USAGE_TEXT = (
    "……我是若叶睦。也可以叫我小睦、木子米。\n"
    "\n"
    "【怎么和我说话】\n"
    "· 群里 @ 我、引用我的消息、或喊我名字（若叶睦/睦/小睦/木子米）→ 我必回\n"
    "· 私信我也可以，我会用若叶睦的语气回你\n"
    "· 没叫我名字时我偶尔会主动搭一句（概率很低）\n"
    "· @ 我时大部分是若叶睦在说，小概率会换成墨提斯（会说“【墨提斯】”标注）\n"
    "· 想找墨提斯本人：说“墨提斯出来”（或“召唤墨提斯”）→ 她出来代班一会儿（15分钟）；说“墨提斯退下”换回我\n"
    "· 问我“你觉得/谁更强/推荐哪个”这类主观问题 → 我会用自己的话给结论，不会给你念资料\n"
    "· 让我“发今日邦娃/棉娃” → 我帮你 @ 短发麻里奈 发命令（每天只能抽一次，邦娃和棉娃合计）\n"
    "\n"
    "【黄瓜农场】\n"
    "· 黄瓜/签到/好感度/称号都按群单独算——同一个群友在各个群的仓库和排行榜互不影响（换群就是从零开始）\n"
    "· 签到 / 种黄瓜 → 每天领小黄瓜\n"
    "· 仓库 / 我的黄瓜 → 看我的黄瓜\n"
    "· 排行榜 → 谁的黄瓜多\n"
    "· 猜大小 5 → 押黄瓜猜大小（引用我的消息回“大/小”）\n"
    "· 猜数字 → 1~10 猜数字，3 次机会\n"
    "\n"
    "【今天吃什么 & 投喂】\n"
    "· 今天吃什么 / 吃什么 → 我说我今天吃了什么（大多是黄瓜做的菜）\n"
    "· 菜单 → 看我能吃到的菜\n"
    "· 投喂 菜名（如：投喂 拍黄瓜）→ 用黄瓜换菜喂我，好感度会涨\n"
    "· 好感度 → 看你和我的关系（陌生人→眼熟→朋友→好朋友→挚友）\n"
    "\n"
    "【兑换 & 商城】\n"
    "· 兑换黄瓜 大 [数量] / 兑换黄瓜 小 [数量] → 黄瓜互兑（3 根小黄瓜 = 1 根大黄瓜）\n"
    "· 商城 → 看大黄瓜能买什么（小黄瓜礼包/点歌卡/专属情书/排行榜称号）\n"
    "· 购买 商品名（如：购买 点歌卡）→ 用大黄瓜买商品\n"
    "\n"
    "【语音】\n"
    "· 语音说 内容（如：语音说 大家好 / hello world / こんばんは；中文日语英语都行）→ 我用声音说给你听（也可以用：说话/说一句）\n"
    "· 长文本也能整段念（上限 400 字，可发 60 秒以上长语音；长语音合成要等几分钟）\n"
    "\n"
    "【点歌】\n"
    "· 点歌 歌名（也可以：唱一首/唱首/我想听 <歌名>）→ 我唱给你听（整首发语音）\n"
    "· 会唱什么 / 歌单 → 报我会的歌\n"
    "· 点我不会的 → 我现场去搜一首发出来（要等十几秒；搜到的会存进我的曲库，下次秒回）\n"
    "\n"
    "【其他】\n"
    "· 这题怎么做（+题目图片）→ 我帮你解题\n"
    "· 提到祥子/素世/灯/立希/初华/海铃/若麦 → 我会按我对他们的印象回答\n"
    "· 发图片 → 偶尔我会看一眼，说一句\n"
    "· 发表情包 → 我会偷偷收着，可能哪天也甩一个出来\n"
    "· 群里大家复读同一句话的时候 → 我偶尔会跟着 +1\n"
    "\n"
    "【任务】\n"
    "· 群里 @ 我 + 交任务就行：「帮我查…」（我去搜资料再整理给你）、「帮我写…」、「帮我出主意」、「这题帮我做」\n"
    "· 需要的时候我会先把做法说一遍，你回一声「好」我就开工；查资料我会先应一声再去搜\n"
    "· 说「记住 xxx」→ 我会记到小本本里，下次就知道了\n"
    "\n"
    "【双 bot 对话】\n"
    "· 发“让扫货和若叶睦对话”或“双bot对话” → 我和扫货会聊一小会儿（只聊几轮，不会没完没了）\n"
    "\n"
    "……就这些。"
)


def _handle_cucumber_answer(gid: str, raw: str, uid: str, name: str = "") -> str | None:
    """处理引用机器人消息时的游戏答案（猜大小/猜数字结算，按群各自结算）。"""
    low = raw.lower().strip()
    if low in ("大", "小"):
        return cucumber.answer_guess(gid, uid, low, name)
    if low.isdigit():
        return cucumber.answer_number(gid, uid, int(low), name)
    return None


def _handle_gift(raw: str, event: GroupMessageEvent, name: str = "") -> str | None:
    """赠送黄瓜：'赠送 [大黄瓜|小黄瓜] [数量] @某人' 或 '赠送 [数量] 12345678'。
    返回回复文本；不是赠送命令返回 None。"""
    low = raw.lower().strip()
    if not re.match(r"(?:赠送|赠送黄瓜|送黄瓜)", low):
        return None
    # 目标：@ 段优先（排除 @ 机器人自己）；也支持 raw 里的纯文本 "@QQ"
    target_qq = None
    for seg in event.message:
        if seg.type == "at":
            qq = str(seg.data.get("qq", "")).strip()
            if qq and qq != str(event.self_id):
                target_qq = qq
                break
    if not target_qq:
        m_at = re.search(r"@(\d{5,12})", raw)
        if m_at and m_at.group(1) != str(event.self_id):
            target_qq = m_at.group(1)
    # 去掉 @ 文本再解析主体（"赠送 2 @222" 的 @222 不干扰数量解析）
    low_clean = re.sub(r"@\d+", "", low)
    low_clean = re.sub(r"\[CQ:at,qq=\d+\]", "", low_clean)
    # 解析：赠送 [类型] [数量] [目标QQ]（@ 目标在 at 段，不在 raw 里）
    m = re.fullmatch(
        r"(?:赠送|赠送黄瓜|送黄瓜)\s*(大黄瓜|小黄瓜)?\s*(\d+)?\s*(\d{5,12})?\s*",
        low_clean,
    )
    if not m:
        return None
    kind, cnt_s, qq_s = m.group(1), m.group(2), m.group(3)
    kind = "big" if kind == "大黄瓜" else "small"
    count = int(cnt_s) if cnt_s else 1
    if not target_qq and qq_s:
        target_qq = qq_s
    if not target_qq:
        return "想送黄瓜？@ 要送的人，然后说：赠送 [数量]（如：赠送 5 @某人，或：赠送 大黄瓜 2 @某人）。"
    kind_name = "大黄瓜" if kind == "big" else "小黄瓜"
    return cucumber.gift(str(event.group_id), str(event.user_id), target_qq, kind, count, name)


async def _refresh_group_summary(group_id: str) -> None:
    """把该群最近积累的原始消息压缩成一段摘要，存入 group_summaries。"""
    if _circuit_open():
        print("[CIRCUIT] 摘要调用被熔断跳过", flush=True)
        return
    raw_msgs = list(group_recent_raw.get(group_id, []))[:SUMMARY_INTERVAL]
    if len(raw_msgs) < 3:
        return
    lines = "\n".join(chat_logic.format_recent_line(x) for x in raw_msgs)
    # 结合旧摘要，让模型续写更新
    old = group_summaries.get(group_id, "")
    prompt = (
        "以下是QQ群最近的聊天记录，用中文总结（250字以内）：聊的话题和进展、提到的人物和关系/称呼/梗、"
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
        summary_kwargs.update(_reasoning_kwargs("summary"))
        r = await asyncio.wait_for(
            client.chat.completions.create(**summary_kwargs),
            timeout=90,
        )
        _record_usage(r.usage)
        summary = (r.choices[0].message.content or "").strip()
        if summary:
            group_summaries[group_id] = summary[:400]
            save_memory()      # 摘要更新后落盘（节流）
            print(f"[SUMMARY] 群{group_id}: {summary[:80]}...", flush=True)
        # 成功后才清理本次已处理的条数（保留调用期间新到的消息，避免丢消息）
        cur = group_recent_raw.get(group_id, [])
        removed = {id(m) for m in raw_msgs}
        group_recent_raw[group_id] = [m for m in cur if id(m) not in removed]
    except Exception as e:
        # 失败保留缓冲：下次触发时重新摘要，不丢上下文
        print(f"[SUMMARY ERROR] {type(e).__name__}: {e}（保留缓冲待重试）", flush=True)


def _is_homework_ask(raw: str) -> bool:
    """判断消息是否在问'这题怎么做'这类解题请求。"""
    low = raw.lower()
    return any(kw.lower() in low for kw in HOMEWORK_KEYWORDS)


def _ocr_image(image_url: str) -> str:
    """提取图片中的文字（题目OCR）。走云端视觉网关（vision.extract_text）。失败返回空串。"""
    import html as _html

    text = vision.extract_text(_html.unescape(image_url))
    return text or ""


async def _get_homework_text(raw: str, event: GroupMessageEvent) -> str:
    """组装题目文字：优先取消息文字，若含图片则OCR提取图片文字，拼接返回。
    OCR 是同步阻塞调用，放线程池避免卡死事件循环。"""
    parts = [raw] if raw else []
    for seg in event.message:
        if seg.type == "image":
            url = seg.data.get("url") or seg.data.get("file")
            if url:
                ocr = await asyncio.to_thread(_ocr_image, url)
                if ocr and ocr not in parts:
                    parts.append(ocr)
    # 防御：OCR 可能拖出超长文本（如整页聊天截图），截断防撑爆输入token
    return "\n".join(parts)[:1200]


def _clean_homework_output(text: str) -> str:
    """清理解题输出的 markdown 残留和多余符号，保持干净文本。"""
    import re as _re
    # 去掉 latex 命令（\boxed{...} \frac{}{} 等），保留内部内容
    text = _re.sub(r"\\boxed\s*\{", "", text)
    text = _re.sub(r"\\frac\s*", "", text)
    text = _re.sub(r"\\text\s*\{", "", text)
    # 去掉 latex 分隔符
    text = text.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    text = text.replace("$$", "")
    text = text.replace("{", "").replace("}", "")
    # 去掉 markdown 标记
    text = _re.sub(r"[*_`~#>|]", "", text)
    # 折叠多余空行
    text = _re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


@chat_matcher.handle()
async def handle_group_message(bot: Bot, event: GroupMessageEvent, state: T_State) -> None:
    group_id = str(event.group_id)
    # 称呼用群名片优先（大家实际叫的名字），没有再用昵称
    user = ""
    if event.sender:
        user = (event.sender.card or "").strip() or (event.sender.nickname or "").strip()
    user = user or "群友"
    raw = event.get_plaintext().strip()
    raw_msg = getattr(event, "raw_message", "") or ""

    # 表情包收集：群友发的动画/自定义表情自动收进库（静默进行，不回消息）
    img_datas = [dict(seg.data) for seg in event.message if seg.type == "image"]
    if str(event.user_id) not in {str(event.self_id), OTHER_BOT_QQ, ASTRA_BOT_QQ}:
        try:
            n = await sticker_mod.collect_from_message(img_datas, raw_msg, group_id, str(event.user_id), user)
            if n:
                print(f"[STICKER] 收了 {n} 个新表情，库共 {sticker_mod.sticker_count()} 个", flush=True)
                asyncio.create_task(sticker_mod.caption_missing_async())   # 新表情后台补语义备注
        except Exception as e:
            print(f"[STICKER] 收集异常 {type(e).__name__}: {e}", flush=True)

    # 记录群消息（用于滚动摘要，记录所有群消息，不只回应的；每条截断省token）
    if raw:
        group_recent_raw.setdefault(group_id, []).append((user, raw[:80], time.time()))
        group_recent_raw[group_id] = group_recent_raw[group_id][-SUMMARY_INTERVAL:]
        group_msg_count[group_id] = group_msg_count.get(group_id, 0) + 1
        if ENABLE_SUMMARY and group_msg_count[group_id] >= SUMMARY_INTERVAL:
            group_msg_count[group_id] = 0
            try:
                task = asyncio.create_task(_refresh_group_summary(group_id))
                summary_tasks.add(task)          # 持引用防 GC 取消
                task.add_done_callback(summary_tasks.discard)
            except Exception as e:
                print(f"[SUMMARY SCHEDULE ERROR] {e}", flush=True)

    # 语音指令（最高优先级，2026-10-16 上移）：用户点播"语音说 …"时最先处理——
    # 之前这段排在复读/黄瓜/接梗等之后，且冷却是按"群"记的（别人刚说过就轮到你被打发）。
    # 现在只让"群消息记录"跑在前面，冷却改成按"人"记（各点各的，互不挡）。
    voice_cmd_text = _extract_voice_cmd(raw)
    if voice_cmd_text is not None:
        cool_key = f"g:{group_id}:{event.user_id}"
        if time.time() - _voice_cmd_last.get(cool_key, 0) < VOICE_CMD_COOLDOWN:
            print(f"[TTS-CMD-COOLDOWN] group={group_id} user={event.user_id} 冷却中，文字提示", flush=True)
            await _safe_finish(event, random.choice(["……刚说过啦，让我缓缓。", "（揉嗓子）……等一下再说。", "……连续说太多会哑的。"]))
            return
        _voice_cmd_last[cool_key] = time.time()
        print(f"[TTS-CMD] group={group_id} user={event.user_id} text={voice_cmd_text[:30]!r}", flush=True)
        await _send_voice(bot, event, voice_cmd_text, explicit=True)
        return

    # 是否被 @ （必定回复）
    self_id = str(event.self_id)
    at_me = any(
        seg.type == "at" and str(seg.data.get("qq", "")).strip() == self_id
        for seg in event.message
    )
    # 兜底：用原始消息字符串检测 @
    if not at_me:
        if f"at,qq={self_id}" in raw_msg or f"[CQ:at,qq={self_id}]" in raw_msg:
            at_me = True
    # 被引用检测：OneBot 适配器通常会先把 reply 段剥到 event.reply，必须优先读它；
    # 只扫描 event.message 会漏掉绝大多数真实的“引用机器人继续聊”。
    quoted_me = False
    parsed_reply = getattr(event, "reply", None)
    if parsed_reply is not None:
        reply_sender = getattr(parsed_reply, "sender", None)
        reply_uid = getattr(reply_sender, "user_id", None)
        if reply_uid is not None:
            quoted_me = str(reply_uid) == self_id
    for seg in event.message:
        if quoted_me:
            break
        if seg.type == "reply":
            dq = str(seg.data.get("qq", ""))
            if dq == self_id:
                quoted_me = True
                break
            # 兜底：引用 id 命中机器人最近发过的消息
            ref_id = str(seg.data.get("id", ""))
            if group_id.isdigit() and ref_id.isdigit():
                recent_ids = bot_recent_msg_ids.get(int(group_id), [])
                if int(ref_id) in recent_ids:
                    quoted_me = True
                    break
    # raw_msg 兜底：NapCat 有时 reply 段只出现在 raw_message 里
    if not quoted_me and "[CQ:reply," in raw_msg:
        import re as _re_quote2
        m = _re_quote2.search(r"\[CQ:reply,id=(\d+)", raw_msg)
        if m and group_id.isdigit():
            ref_id = int(m.group(1))
            recent_ids = bot_recent_msg_ids.get(int(group_id), [])
            if ref_id in recent_ids:
                quoted_me = True
    # 提到"若叶睦"或其别称（睦/小睦/木子米/睦头/睦子米/墨提斯等）：必定回复，强制使用若叶睦人格
    # 用严格匹配（精确子串+显式别名），单字"睦"带防误伤（避免"和睦"触发）
    mention_mutsumi = _raw_contains_name_strict(raw, "若叶睦")
    # 相关话题（2026-10-16）：群里聊到她身边的人/原作（祥子、爽世、Ave Mujica、春日影…）
    # 也算"在聊她的事"，要接一句；她本人名字已在上面覆盖，这里只补"非她的名字"的情况
    related_topic = (not mention_mutsumi) and _match_related_topic(raw)
    # 双bot对话窗口内（对方=薇欧拉）：对话模式，冷却/沉默/图片评价均豁免
    in_dialogue = str(event.user_id) == OTHER_BOT_QQ and time.time() < dialogue_until.get(group_id, 0)
    # 人格选择（一次判定，全程一致）：
    #   ① 召唤窗口内（"墨提斯出来"后 MORTIS_WINDOW 秒）→ 一律墨提斯代班；
    #   ② 消息里说"墨提斯退下/换回若叶睦"→ 清窗口，立刻换回来；
    #   ③ 否则被@/被引用时按 MORTIS_TRIGGER_RATE 概率切换墨提斯；
    #   ④ 显式提到"若叶睦"或主动搭话 → 一律若叶睦。
    mortis_active = time.time() < mortis_until.get(group_id, 0)
    mortis_intent = chat_logic.mortis_intent(
        raw, addressed=(at_me or quoted_me or mention_mutsumi), active=mortis_active
    ) if (at_me or quoted_me or mention_mutsumi) else None
    if mortis_intent == "summon":
        mortis_until[group_id] = time.time() + MORTIS_WINDOW
        mortis_active = True
        print(f"[MORTIS] group={group_id} 召唤成功：墨提斯代班 {int(MORTIS_WINDOW)}s", flush=True)
    elif mortis_intent == "dismiss":
        mortis_until.pop(group_id, None)
        mortis_active = False
        print(f"[MORTIS] group={group_id} 退场：换回若叶睦", flush=True)
    use_mortis = chat_logic.pick_persona(
        mortis_active=mortis_active, addressed=(at_me or quoted_me),
        mention_mutsumi=mention_mutsumi, rate=MORTIS_TRIGGER_RATE, roll=random.random())
    print(f"[PERSONA] group={group_id} at_me={at_me} quoted_me={quoted_me} mention_mutsumi={mention_mutsumi} "
          f"related={related_topic} mortis_window={mortis_active} intent={mortis_intent} "
          f"-> {'墨提斯' if use_mortis else '若叶睦'}", flush=True)

    # 表情包互动：群友发表情，小概率也甩一个表情回去（每群有冷却）。
    # 2026-10-07 起不许乱发：先识图看懂对方发的什么，再从库里挑一个语义上接得上的；
    # 看不懂（识图失败且连 QQ 文字摘要都没有）或挑不出合适的，就宁可不发。
    if (not at_me and not quoted_me and not mention_mutsumi and not in_dialogue
            and str(event.user_id) != OTHER_BOT_QQ
            and sticker_mod.has_sticker(img_datas, raw_msg)
            and sticker_mod.sticker_count() >= 1
            and random.random() < STICKER_ECHO_RATE
            and time.time() - _sticker_echo_last.get(group_id, 0) >= STICKER_ECHO_COOLDOWN):
        _inc_desc = ""
        _inc_path = sticker_mod.find_local_path(img_datas, raw_msg)
        if _inc_path:
            _inc_desc = await asyncio.to_thread(vision.describe_file, _inc_path) or ""
        if not _inc_desc:
            _inc_desc = sticker_mod.first_summary(img_datas, raw_msg)
        if _inc_desc:
            spath = await sticker_mod.pick_matching_async(_inc_desc)
            if spath and await _send_sticker_file(bot, group_id, spath, event):
                _sticker_echo_last[group_id] = time.time()
                print(f"[STICKER-ECHO] group={group_id} 对方表情={_inc_desc[:50]!r} "
                      f"-> {os.path.basename(spath)}", flush=True)
                return
        else:
            print(f"[STICKER-ECHO] group={group_id} 看不懂对方表情（无识图无摘要），不发", flush=True)

    # 复读加一：群里同一句话被两个以上的人重复时，她随大流发一条一模一样的（零 token、不引用）
    if (REPEAT_ECHO_ENABLED
            and not at_me and not quoted_me and not mention_mutsumi and not in_dialogue
            and str(event.user_id) != OTHER_BOT_QQ
            and time.time() - _repeat_echo_last.get(group_id, 0) >= REPEAT_ECHO_COOLDOWN
            and time.time() - last_reply_time.get(group_id, 0) >= COOLDOWN_SECONDS):
        _rt = _repeat_echo_candidate(group_id, raw)
        if _rt and _repeat_echo_should_fire(group_id, _rt):
            _repeat_echo_mark(group_id, _rt)
            print(f"[REPEAT-ECHO] group={group_id} 复读加一: {_rt[:40]!r}", flush=True)
            try:
                sent = await bot.send_group_msg(group_id=int(group_id),
                                                message=Message(MessageSegment.text(_rt)))
                _record_sent_id(event, sent)
            except Exception as e:
                print(f"[REPEAT-ECHO] 发送失败 {type(e).__name__}: {e}", flush=True)
            return

    # 空内容消息（纯图片/表情/语音/卡片）：未触发不接话；被@/被引用/被点名/对话窗口内
    # 用占位符进对话，避免空 user 内容白烧一次 LLM 调用
    if not raw:
        if not (at_me or quoted_me or mention_mutsumi or in_dialogue):
            return
        raw = "[图片/表情]"
    print(f"[AT] self_id={self_id} at_me={at_me} quoted_me={quoted_me} mention_mutsumi={mention_mutsumi} rawmsg={raw_msg[:80]!r}", flush=True)

    # 被@/引用/点名且消息带图/视频：识图/抽帧描述附进消息，让她真的"看懂"再回答
    if ENABLE_IMAGE and not _circuit_open() and (at_me or quoted_me or mention_mutsumi or in_dialogue):
        _img_data = _first_image_data(raw_msg, event)
        if _img_data:
            _img_desc = await _describe_image_sdata(bot, _img_data)
            if _img_desc:
                raw = f"（发了一张图：{_img_desc}）" if raw == "[图片/表情]" else f"{raw}（附图：{_img_desc}）"
        else:
            _vdata = _first_video_data(event)
            if _vdata:
                _vdesc = await _describe_video_segment(bot, _vdata, [QUOTE_VISION_CALLS])
                if _vdesc:
                    raw = f"（发了一段{_vdesc}）" if raw == "[图片/表情]" else f"{raw}（附{_vdesc}）"

    # 黄瓜农场系统：命令触发
    disp_name = ""
    if event.sender:
        disp_name = (event.sender.card or "").strip() or (event.sender.nickname or "").strip()
    if not disp_name:
        disp_name = str(event.user_id)

    # ★ 主AI编排·确认流：她发过方案后，群友引用回"好/可以"→执行；"算了"→取消
    # （先于游戏结算：确认词"好/嗯"和猜大小答案"大/小"不冲突，但优先级应更高）
    if AGENT_MODE and group_id in pending_tasks:
        task = pending_tasks[group_id]
        if time.time() - task["at"] > TASK_PENDING_EXPIRE:
            pending_tasks.pop(group_id, None)
            print(f"[ORCH] 待确认任务过期 group={group_id}", flush=True)
        else:
            same_uid = str(event.user_id) == task["uid"]
            if (quoted_me and agent_core.is_confirm(raw)) or (same_uid and agent_core.is_confirm(raw)):
                pending_tasks.pop(group_id, None)
                plan = task["plan"]
                print(f"[ORCH] 确认通过 group={group_id} -> 执行 {plan.get('agent')}", flush=True)
                await _safe_finish(event, "……嗯。那我开始了。")
                if await _execute_and_deliver(event, plan, plan.get("raw") or plan.get("need", "")):
                    return
                # 执行失败：继续往下走快路兜底（quoted_me 会触发正常回复）
            elif (quoted_me and agent_core.is_cancel(raw)) or (same_uid and agent_core.is_cancel(raw)):
                pending_tasks.pop(group_id, None)
                print(f"[ORCH] 任务取消 group={group_id}", flush=True)
                await _safe_finish(event, random.choice(agent_core.CANCEL_ACKS))
                return

    # 引用机器人消息给答案：猜大小/猜数字结算
    # 只认"引用了机器人"（quoted_me）——引用任意人的消息不能结算，防止误结算别人的游戏
    if quoted_me:
        answer_reply = _handle_cucumber_answer(group_id, raw, str(event.user_id), disp_name)
        if answer_reply:
            print(f"[CUCUMBER-ANSWER] user={event.user_id} raw={raw!r} -> {answer_reply[:50]!r}", flush=True)
            await _safe_finish(event, answer_reply)
            return

    # 赠送黄瓜：赠送 [大黄瓜|小黄瓜] [数量] @某人
    gift_reply = _handle_gift(raw, event, disp_name)
    if gift_reply:
        print(f"[GIFT] user={event.user_id} raw={raw!r} -> {gift_reply[:40]!r}", flush=True)
        await _safe_finish(event, gift_reply)
        return

    cuke_reply = _handle_cucumber(group_id, raw, str(event.user_id), disp_name)
    if cuke_reply:
        print(f"[CUCUMBER] user={event.user_id} raw={raw!r} -> {cuke_reply[:50]!r}", flush=True)
        # 游戏开始提示需要用 send 记录 id（供后续引用结算）；其余用安全发送
        if raw.strip().lower().startswith(("猜大小", "押大小", "猜数字", "猜数")):
            await _send_and_record(bot, event, cuke_reply)
        else:
            await _safe_finish(event, cuke_reply)
        return

    # （语音指令块已上移到 handler 开头，最高优先级，见上方注释）

    # 点歌："点歌<歌名>" / "唱一首<歌名>" / "我想听<歌名>" / "唱<歌名>"——发曲库音频（她"唱"）。
    # 2026-10-17：识别放宽到自然说法；找不到就直说不会（不硬编）；"会唱什么"报歌单。
    # 弱前缀（唱/放/听/演奏）只在被叫到、或真命中曲库时才当点歌，避免"放了假了"这类误伤。
    song_hit = _match_song(raw)
    if song_hit:
        song_name, song_path, song_explicit = song_hit
        _song_addressed = at_me or quoted_me or mention_mutsumi
        if song_name == "__list__":
            await _safe_finish(event, _songbook_reply())
            return
        if song_path or song_explicit or _song_addressed:
            if not song_name:
                print(f"[SONG] group={group_id} 只说了点歌没给歌名 -> 问哪首", flush=True)
                _titles = song_titles()
                await _safe_finish(event, "……想听哪首？" + (f"\n会的：{'、'.join(_titles[:8])}" if _titles else ""))
                return
            if not song_path:
                # 曲库里没有 → 现场搜一首下载回来再发（2026-10-17 用户要求"让他自己搜歌来发"）。
                # 搜+下要十几秒到几十秒，先回一句"我去找找"稳住场；yt-dlp 全局串行，别并发。
                _can_fetch = (SONG_AUTO_FETCH
                              and time.time() - _song_fetch_last.get(group_id, 0) >= SONG_FETCH_COOLDOWN)
                if _can_fetch:
                    _song_fetch_last[group_id] = time.time()
                    print(f"[SONG] group={group_id} raw={raw!r} 曲库没有《{song_name}》-> 现场搜歌", flush=True)
                    await _safe_finish(event, random.choice([
                        "……我去找找。", "……等等，我去翻一下。", "……嗯。找找看。"]))
                    async with _song_fetch_lock:
                        fetched = await asyncio.to_thread(
                            songfetch.fetch, song_name, SONG_FETCH_TIMEOUT)
                    if fetched:
                        _song_last[group_id] = time.time()
                        await _send_song(bot, event, fetched, title=song_name)
                    else:
                        await _safe_finish(event, f"……《{song_name}》没搜到。")
                    return
                print(f"[SONG] group={group_id} raw={raw!r} -> 不会《{song_name}》（搜歌冷却中）", flush=True)
                _titles = song_titles()
                await _safe_finish(event, f"……《{song_name}》不会。" +
                                   (f"\n会的：{'、'.join(_titles[:6])}" if _titles else ""))
                return
            if time.time() - _song_last.get(group_id, 0) < SONG_COOLDOWN:
                print(f"[SONG-COOLDOWN] group={group_id} 冷却中，静默", flush=True)
                return
            _song_last[group_id] = time.time()
            print(f"[SONG] group={group_id} user={event.user_id} raw={raw!r} -> {song_path}", flush=True)
            await _send_song(bot, event, song_path, title=song_name)
            return

    # 每日娃娃：群友让她"发今日邦娃/棉娃" → 帮他们 @ 短发麻里奈 触发那条命令（零 token）。
    # 2026-10-17 用户要求："面对群友的合理互动要求，可以接受"——此前她的回复是"……不要。"
    # 2026-10-19：冷却从 180s 降到 15s，且冷却中也必须回一句（不能静默装死）。
    _doll_word = _doll_cmd_word(raw, addressed=(at_me or quoted_me or mention_mutsumi))
    if _doll_word and str(event.user_id) not in (str(event.self_id), DOLL_BOT_QQ):
        # 2026-10-26 用户反馈"还是不会艾特"：撤销每日一次拦截（娃娃 bot 实测每次 /今日X
        # 都回图），仅保留 15 秒冷却防连点。
        if time.time() - _doll_relay_last.get(group_id, 0) < DOLL_RELAY_COOLDOWN:
            print(f"[DOLL-RELAY-COOLDOWN] group={group_id} 冷却中（{int(DOLL_RELAY_COOLDOWN)}s），只说一句", flush=True)
            await _safe_finish(event, random.choice([
                "……刚叫过了。", "……等一下，才刚叫过。", "……已经叫过麻里奈了。"]))
            return
        _doll_relay_last[group_id] = time.time()
        _doll_mark_drawn(group_id, _doll_word)
        if not await _relay_doll_cmd(bot, event, _doll_word):
            # 没发出去就撤销，别把今天的次数白白用掉
            _doll_undo_drawn(group_id, _doll_word)
            await _safe_finish(event, "……没叫到。等会儿再试。")
        return

    # 使用说明：完备的功能介绍（优先于固定接梗里的简短"使用说明"规则）
    if raw.strip().lower() in ("使用说明", "说明书", "帮助", "help", "功能说明", "怎么使用", "怎么用", "有什么功能"):
        await _safe_finish(event, USAGE_TEXT)
        return

    # 双bot对话指令：双方各自开窗；提到自己名字的一方发开场白（若叶睦先开口）
    if _is_dialogue_cmd(raw):
        dialogue_until[group_id] = time.time() + DIALOGUE_WINDOW
        dialogue_turns[group_id] = 0
        print(f"[DIALOGUE] 群{group_id} 开启双bot对话窗口 {DIALOGUE_WINDOW}s", flush=True)
        if mention_mutsumi:
            await _safe_finish(event, "……嗯。扫货，来聊两句吧。")
        else:
            await _safe_finish(event, "……嗯。")
        return

    # 固定接梗：命中关键词直接回复，不调用 AI、不受概率/冷却影响
    # 固定接梗：被 @/被引用/被提到（含"她的事"相关话题）时才接（没人提她绝不插嘴）
    if ENABLE_SPECIAL and (at_me or quoted_me or mention_mutsumi or related_topic):
        # 人格已在上方判定：墨提斯时跳过若叶睦专属印象接梗
        special = _match_special(raw, group_id, use_mortis=use_mortis)
        if not special:
            # 按 QQ 号触发：指定用户发消息也接梗
            user_special = SPECIAL_USER_RULES.get(str(event.user_id))
            if user_special:
                special = random.choice(user_special) if isinstance(user_special, list) else user_special
    else:
        special = None
    if special:
        # 2026-10-22 防刷屏：未被 @/引用/提名字时，接梗每群 60 秒最多一次；
        # 被叫的接梗不受限（"被叫必须出声"优先）。见 _SPECIAL_RELATED_LAST 注释。
        _special_ts = time.time()
        _special_addressed = at_me or quoted_me or mention_mutsumi
        if not _special_addressed and _special_ts - _SPECIAL_RELATED_LAST.get(group_id, 0.0) < _SPECIAL_RELATED_COOLDOWN:
            print(f"[SPECIAL-SKIP] group={group_id} 未被叫、接梗冷却中: {special!r}", flush=True)
            return
        _SPECIAL_RELATED_LAST[group_id] = _special_ts
        print(f"[SPECIAL] group={group_id} user={event.user_id} raw={raw!r} -> {special}", flush=True)
        await _safe_finish(event, special)
        return

    # 解题功能：问"这题怎么做"时（熔断期间跳过，走正常流程）
    if ENABLE_HOMEWORK and _is_homework_ask(raw) and (at_me or quoted_me or mention_mutsumi) and not _circuit_open():
        # 墨提斯人格（本次判定命中）被问解题 -> 固定回复"全都不会做"
        if use_mortis and not mention_mutsumi:
            print(f"[HOMEWORK-MORTIS] 墨提斯被问解题 -> {MORTIS_HOMEWORK_REPLY}", flush=True)
            await _safe_finish(event, MORTIS_HOMEWORK_REPLY)
            return
        # 若叶睦（提到若叶睦，或未被@但需要若叶睦）被问解题 -> 提取题目并深度解答
        print(f"[HOMEWORK] 群{group_id} 解题请求: {raw!r}", flush=True)
        homework_text = await _get_homework_text(raw, event)
        if homework_text.strip():
            try:
                solve_kwargs = {
                    "model": HOMEWORK_MODEL,
                    "messages": [
                        {"role": "system", "content": HOMEWORK_PROMPT},
                        {"role": "user", "content": homework_text},
                    ],
                    "max_tokens": 20480,
                }
                # deepseek-reasoner 不支持 temperature
                if HOMEWORK_MODEL != "deepseek-reasoner":
                    solve_kwargs["temperature"] = 0.3
                solve_reply = await asyncio.wait_for(
                    client.chat.completions.create(**solve_kwargs),
                    timeout=180,
                )
                _record_usage(solve_reply.usage)
                solve_text = (solve_reply.choices[0].message.content or "").strip()
                solve_text = _clean_homework_output(solve_text)
            except Exception as e:
                print(f"[HOMEWORK ERROR] {type(e).__name__}: {e}", flush=True)
                solve_text = ""
            if solve_text:
                print(f"[HOMEWORK-OK] -> {solve_text[:60]!r}", flush=True)
                await _safe_finish(event, solve_text)
                return
        # 提取失败或解答失败，走正常流程（让AI自由对话接住）

    # ★ 现场记忆：群友让她"记住 xxx"时直接写入经验库（零 token，立即可见）
    m_remember = re.match(r"^(?:记住|记一下|记着)[，:：, ]?\s*(.{2,80})$", raw.strip())
    if AGENT_MODE and m_remember and (at_me or quoted_me or mention_mutsumi):
        body = m_remember.group(1).strip()
        tags = [w for w in re.findall(r"[\u4e00-\u9fff]{2,4}", body)][:3]
        learned = experience.learn("L2", body, tags=tags)
        print(f"[EXP] 现场记忆 group={group_id} user={event.user_id} ok={learned}: {body[:40]!r}", flush=True)
        await _safe_finish(event, random.choice(["……嗯。记住了。", "……（点头）记下了。", "……嗯。这个我记着。"]))
        return

    # ★ claw 化：被引用/转发的消息她自己读（只读接口，截断+超时；仅触发场景才取）
    quoted_note = ""
    if at_me or quoted_me or mention_mutsumi or in_dialogue or related_topic:
        try:
            quoted_note = await _fetch_quoted_content(bot, event)
        except Exception as e:
            print(f"[QUOTE] 异常 {type(e).__name__}: {e}", flush=True)
    raw_note = f"{raw}\n（他引用的消息内容：{quoted_note}）" if quoted_note else raw

    # ★ 主AI编排：任务请求（帮我写/整理/出主意…）→ 理解 → 确认 → 子代理 → 交付 → 沉淀
    # 任何一步失败都落回快路（继续走下面原流程），群友无感。
    if (AGENT_MODE and (at_me or quoted_me or mention_mutsumi) and not in_dialogue
            and agent_core.should_try_task(raw_note) and not _is_persona_talk(raw_note)
            # 主观题（评价/看法/喜好/比较/推荐/值不值，含'怎么样/如何'）不入编排：
            # 她有自己的判断和脾气，走人设直答；查资料只给'明确要查'的题用
            # （2026-10-16 用户反馈：主观性问题要用自己的语气和想法，不是简单找资料）。
            # 判她本人的原话（raw），不含被引用内容。
            and not chat_logic.is_subjective_ask(raw)):
        try:
            if await _handle_agent_task(event, raw_note, disp_name):
                return
        except Exception as e:
            print(f"[ORCH] 编排异常 {type(e).__name__}: {e}，落回快路", flush=True)

    # 概率型接梗：命中关键词/指定用户后按概率回复
    # 概率型接梗：被 @/被引用/被提到（含相关话题）时才接（不插嘴）
    prob_special = _match_prob(raw, str(event.user_id)) if ENABLE_SPECIAL and (at_me or quoted_me or mention_mutsumi or related_topic) else None
    if prob_special:
        print(f"[PROB_SPECIAL] group={group_id} user={event.user_id} raw={raw!r} -> {prob_special}", flush=True)
        # 固定文案也走拆句/截断（2026-10-16）：否则绕过"每条≤15字"的人设硬规则
        await _safe_finish(event, "\n".join(_split_short_lines(prob_special, 60 if use_mortis else MUTSUMI_LINE_LIMIT)))
        return

    # 原作梗接梗：命中梗关键词就自然接梗
    # 原作梗接梗：被 @/被引用/被提到（含相关话题）时才接（不插嘴）
    meme_reply = _match_meme(raw) if ENABLE_MEME and (at_me or quoted_me or mention_mutsumi or related_topic) else None
    if meme_reply:
        print(f"[MEME] group={group_id} user={event.user_id} raw={raw!r} -> {meme_reply}", flush=True)
        # 固定梗文案也按人设拆句（多条短句），不整段发出
        await _safe_finish(event, "\n".join(_split_short_lines(meme_reply, 60 if use_mortis else MUTSUMI_LINE_LIMIT)))
        return

    # 图片/视频反应：识图（vision.py 云端视觉）后由若叶睦本人用人设口吻点评一句。
    # 识图失败/她不想说话就保持沉默——绝不瞎评"难看"（旧版识图坏掉时会误伤群友，2026-09-27 修）。
    # 被@/被引用/提到若叶睦时跳过（优先回答真正的提问）；双bot对话窗口内也跳过。
    if ENABLE_IMAGE and not at_me and not quoted_me and not mention_mutsumi and not in_dialogue \
            and not _circuit_open() and _has_media(raw_msg, event) and random.random() < IMAGE_REPLY_RATE:
        last_reply_time[group_id] = time.time()
        image_url = _first_image_url(raw_msg, event)
        if image_url:
            desc = await asyncio.to_thread(vision.describe_image, image_url)
            if desc:
                reply_text = await _persona_media_reply(desc)
                if reply_text:
                    print(f"[IMAGE] group={group_id} desc={desc[:50]!r} -> {reply_text!r}", flush=True)
                    await _safe_finish(event, reply_text)
                    return
                print(f"[IMAGE] group={group_id} 识图成功但她选择沉默: {desc[:50]!r}", flush=True)
        elif any(k in raw_msg.lower() for k in ANIME_KEYWORDS):
            # 视频链接等无法识图：按链接文字粗判断，命中二次元关键词淡淡接一句
            reply_text = random.choice(REPLY_CURIOUS)
            print(f"[IMAGE] group={group_id} 链接关键词命中 -> {reply_text!r}", flush=True)
            await _safe_finish(event, reply_text)
            return

    # 冷却控制：同一群短时间不重复回复
    # 跳过冷却（必回）：被 @/被引用/被她的名字叫到/双bot对话
    # 守冷却（10 秒节流）：相关话题（聊祥子/Ave Mujica 等会密集出现）、主动搭话
    triggered = at_me or quoted_me or mention_mutsumi or in_dialogue
    # 这条消息跟她的关系（决定加不加判定块，见 chat_logic._UNADDRESSED_BLOCKS）：
    #   at       被@/被引用/双bot对话 = 在跟她说话（不加块，行为与以前一致）
    #   mention  群里有她的名字/别称 = 在跟她搭话，要接
    #   related  群友在聊她身边的人/原作 = 可以接一句印象
    #   proactive 都没提到（只有 REPLY_RATE 概率命中才会走到）= 接得上才接
    addressed_mode = "at" if (at_me or quoted_me or in_dialogue) else (
        "mention" if mention_mutsumi else ("related" if related_topic else "proactive"))
    now = time.time()
    last = last_reply_time.get(group_id, 0)
    if not triggered and now - last < COOLDOWN_SECONDS:
        if related_topic:
            print(f"[RELATED-COOLDOWN] group={group_id} 相关话题冷却中（10s），本条不回", flush=True)
        return

    # 未触发且不相关：按 REPLY_RATE 概率主动搭话（默认 0=不搭话）；双bot对话窗口内必定接
    if not triggered and not related_topic:
        if random.random() >= REPLY_RATE:
            print(f"[SKIP] group={group_id} 非@/未被引用/未提若叶睦，概率未命中不回复: raw={raw!r}", flush=True)
            return
        print(f"[PROACTIVE] group={group_id} 主动搭话命中(rate={REPLY_RATE}): raw={raw!r}", flush=True)
    if related_topic:
        print(f"[RELATED] group={group_id} 群友聊到相关话题，接一句: raw={raw!r}", flush=True)
    last_reply_time[group_id] = now
    print(f"[REPLY] group={group_id} at_me={at_me} quoted_me={quoted_me} mention_mutsumi={mention_mutsumi} "
          f"addressed={addressed_mode} raw={raw!r}", flush=True)

    # 同群「生成+发送」串行化（与薇欧拉一致）：防并发 @ 时 LLM+发送洪峰被 NapCat 拒绝（retcode 1200）
    lock = group_locks.setdefault(group_id, asyncio.Lock())
    async with lock:
        # 维护上下文：按"人格"分开记忆，避免两套人格串味（防OOC）
        hist_key = f"{group_id}:{'mortis' if use_mortis else 'mutsumi'}"
        history = contexts.setdefault(hist_key, [])
        # 历史条目截断，防止长文粘贴撑爆每次调用的输入token；
        # 引用内容附在条目里（截 300 字），后续追问"那第二段呢"也能接得上
        _u_entry = f"{user}：{raw[:80]}"
        if quoted_note:
            _u_entry += f"（他引用了一条消息：「{_cap_text(quoted_note, 300)}」）"
        _remember(hist_key, {"role": "user", "content": _u_entry})
        history = contexts[hist_key]
        # 防止无限增长
        if len(contexts) > 60:
            for old_key in list(contexts)[:-20]:
                contexts.pop(old_key, None)

        summary = group_summaries.get(group_id, "")
        try:
            exp_hint = experience.render_hint(raw)
        except Exception:
            exp_hint = ""
        try:
            sticker_hint = sticker_mod.pick_for_prompt()
        except Exception:
            sticker_hint = ""
        # 梗解释（2026-10-16）：网络热梗 / 邦多利梗，看懂了再决定接不接
        try:
            meme_hint = _meme_hint(raw)
        except Exception:
            meme_hint = ""
        # Final prompt assembly lives in a pure helper so it can be tested
        # without starting NoneBot.
        messages = chat_logic.build_context_messages(
            base_prompt=SYSTEM_PROMPT_MORTIS if use_mortis else SYSTEM_PROMPT_MUTSUMI,
            user=user,
            current_entry=history[-1]["content"],
            summary=summary,
            recent=group_recent_raw.get(group_id, []),
            history=history[:-1],
            context_lines=GROUP_CONTEXT_LINES,
            history_rounds=SUMMARY_CONTEXT_ROUNDS,
            experience_hint=exp_hint,
            sticker_hint=sticker_hint,
            meme_hint=meme_hint,
            addressed=addressed_mode,
        )

        # token 熔断：打开期间 LLM 对话静默跳过（本地功能不受影响）
        if _circuit_open():
            print(f"[CIRCUIT] 群{group_id} LLM 对话被熔断，静默跳过", flush=True)
            return

        try:
            chat_kwargs = {
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                # 回复会被代码截断：若叶睦每条≤15字×最多3条=≤45字、墨提斯≤60字，按人格压紧上限，防模型长篇浪费输出token
                "max_tokens": 2048,
            }
            # deepseek-reasoner 不支持 temperature
            if DEEPSEEK_MODEL != "deepseek-reasoner":
                chat_kwargs["temperature"] = CHAT_TEMPERATURE
            # 关闭思考：思考token被丢弃但仍计费，直接关掉省token
            chat_kwargs.update(_reasoning_kwargs("chat"))
            reply = await asyncio.wait_for(
                client.chat.completions.create(**chat_kwargs),
                timeout=150,
            )
            _record_usage(reply.usage)
            text = (reply.choices[0].message.content or "").strip()

            # 搜索一轮：模型输出【搜索:关键词】= 它要查资料再答（每条消息只此一轮，防循环）。
            # 这是"想久一点但不乱答"的落点：查不到资料时要求它【沉默】而不是硬编。
            m_search = re.fullmatch(r"【搜索\s*[:：]\s*([^\】]{1,24})】\s*", text or "")
            if m_search:
                kw = m_search.group(1).strip()
                result_block = None
                if SEARCH_ENABLED and not _circuit_open():
                    try:
                        result_block = await asyncio.to_thread(websearch.search, kw)
                    except Exception as e:
                        print(f"[SEARCH] 异常 {type(e).__name__}: {e}", flush=True)
                if result_block:
                    follow = (f"{result_block}\n\n（根据资料按人设回答：问看法/评价/比较这类主观问题，"
                              f"就用你自己的话给一句结论+最多一句依据，别罗列资料；资料里没有的数值/事实"
                              f"一个字都不要编，还是没把握就【沉默】）")
                else:
                    follow = ("（没有搜到资料。主观问题可以只按你的感觉给一句态度；"
                              "具体事实没把握就说不知道，不要编造。）")
                print(f"[SEARCH] group={group_id} kw={kw!r} 有资料={bool(result_block)}", flush=True)
                chat_kwargs["messages"] = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": follow},
                ]
                reply = await asyncio.wait_for(
                    client.chat.completions.create(**chat_kwargs),
                    timeout=150,
                )
                _record_usage(reply.usage)
                text = (reply.choices[0].message.content or "").strip()
                # 防泄漏：二次回答里不该再出现搜索标记
                text = re.sub(r"【搜索[^】]{0,24}】", "", text).strip()
        except Exception as e:
            print(f"[LLM ERROR] {type(e).__name__}: {e}", flush=True)
            await chat_matcher.finish(_reply_message(event, "……（脑子有点卡，等下再叫我。）"))
            return

        # 2026-10-21 防镜像标注泄漏：模型会模仿历史里的「（若叶睦）」标注并说进群里
        # （10-20 18:14 实锤）。旧格式（）、新格式〔〕在正文开头的一律剥掉。
        text = re.sub(r"^\s*(?:[（(〔\[]\s*(?:若叶睦|墨提斯)\s*[）)〕\]])+\s*", "", text)

        if not text:
            print("[LLM EMPTY] DeepSeek 返回空", flush=True)
            # 2026-10-21：空响应不能静默丢消息（10-20 18:33 被@问「你内核是什么模型」直接没回）。
            # 被叫的场合给一句兜底；未叫的保持沉默。
            if at_me or quoted_me or mention_mutsumi or in_dialogue:
                await chat_matcher.finish(_reply_message(event, "……（脑子有点卡，等下再叫我。）"))
            return

        # 沉默许可：模型输出【沉默】表示"不想回答/回答会OOC"，此时直接静默不发消息
        # 双bot对话窗口内不允许沉默（强制继续，保证对话不中途断掉）
        if text in SILENCE_MARKERS or (len(text) <= 6 and any(m in text for m in SILENCE_MARKERS)):
            if not in_dialogue:
                print(f"[SILENCE] 模型选择沉默，不发送: {text!r}", flush=True)
                return
            print(f"[SILENCE→CONTINUE] 对话模式强制继续: {text!r}", flush=True)
            text = "……嗯。你继续说。"

        # 代码级拦截1：本群禁用的人物不得在回复中出现
        banned_in_group = BANNED_MENTIONS.get(int(group_id), set())
        if banned_in_group:
            for banned_name in banned_in_group:
                if _raw_contains_name(text, banned_name):
                    print(f"[GUARD-BAN] 群{group_id} 拦截提及{banned_name}: {text!r}", flush=True)
                    # 多样化兜底，避免千篇一律的"……嗯。"
                    text = random.choice(["……嗯。", "……。", "（低头）嗯。", "……不说了。", "……（摇头）"]) if not use_mortis \
                        else random.choice(["（歪头）嗯？", "（撇嘴）……哼。", "……懒得说。", "（抱胸）关我什么事。"])
                    break

        # 代码级拦截2：去除 AI 味词汇（防露馅）
        if any(m in text for m in AI_MARKERS):
            print(f"[GUARD-AI] 拦截AI味回复: {text!r}", flush=True)
            text = random.choice(["……嗯。", "……。", "（低头）……嗯。"]) if not use_mortis \
                else random.choice(["（歪头）你说什么？", "（叉腰）你说啥？", "……没听懂。"])

        # 表情标记拆分（先拆再截断，防止 [表情:编号] 被截断毁掉）
        sticker_path, text = _split_sticker_mark(text)

        # 多行拆分（一句一条，2026-10-07 收紧；2026-10-12 单条上限 10→15）：若叶睦每条≤15字、墨提斯≤60字，最多 3 条。
        # 换行优先，单行按 。！？ 切句（省略号跟随下一句）。
        lines = _split_short_lines(text, 60 if use_mortis else MUTSUMI_LINE_LIMIT)
        text = "\n".join(lines)
        # 防复读：和上一条完全相同时换一种说法
        text = _dedupe_reply(text, history, use_mortis)
        # 没人叫她（主动搭话）时：万能句一律不发——那种插嘴只会显得烦。
        # 被提到名字/相关话题不拦：用户要求"提到若叶睦及相关的也要回复"，
        # 哪怕"……嗯。"也算一次回应（提示词已要求尽量给出具体内容）。
        if addressed_mode == "proactive" and text and _is_generic_reply(text.replace("\n", "")):
            print(f"[GENERIC-PROACTIVE] 没人叫她的场合，万能句不发送: {text!r}", flush=True)
            return
        # 防敷衍连发：连续 3 条万能句就强制沉默（提示词已禁，这里是代码兜底）
        if text and _is_generic_reply(text.replace("\n", "")):
            _generic_streak[hist_key] = _generic_streak.get(hist_key, 0) + 1
            if _generic_streak[hist_key] >= 3:
                print(f"[GENERIC] 连续敷衍第{_generic_streak[hist_key]}次，强制沉默: {text!r}", flush=True)
                return
        else:
            _generic_streak[hist_key] = 0

        _remember(hist_key, {"role": "assistant", "content": text},
                  cross_label="墨提斯" if use_mortis else "若叶睦")
        save_memory()          # 记忆落盘（节流）：重启不失忆

        # 发送用的行：墨提斯标注只加在第一条前（每条都加太刷屏）
        send_lines = text.split("\n") if text else []
        if use_mortis and send_lines:
            send_lines[0] = f"【墨提斯】{send_lines[0]}"
        send_text = "\n".join(send_lines)

        # 语音模式：按概率把回复用"语音+文字"发出（若叶睦偶尔开口说话，人设惊喜感）；
        # 合成失败自动降级纯文字，不影响回复
        if ENABLE_TTS and TTS_ENABLED and TTS_REF_AUDIO and random.random() < TTS_AUTO_RATE:
            print(f"[VOICE-AUTO] group={group_id} 概率命中语音回复: {text[:30]!r}", flush=True)
            await _send_voice(bot, event, send_text.replace("\n", "，"))
            if sticker_path:
                await _send_sticker_file(bot, group_id, sticker_path, event)
            return

        # 逐条发送（模拟真人连着打字），每条记录消息 id 供"被引用"检测；
        # 回复可以是纯文字多条、纯表情、或文字+表情
        for j, ln in enumerate(send_lines):
            try:
                if j == 0:
                    # 每条回复的首个气泡都带 QQ 引用，标明在回谁（2026-10-10 用户要求）；
                    # 连发的后续气泡不重复引用，避免刷屏
                    sent = await bot.send_group_msg(
                        group_id=int(group_id),
                        message=_reply_message(event, ln),
                    )
                else:
                    sent = await bot.send_group_msg(
                        group_id=int(group_id),
                        message=Message(MessageSegment.text(ln)),
                    )
                _record_sent_id(event, sent)
            except Exception as e:
                print(f"[SEND ERROR] {type(e).__name__}: {e}", flush=True)
                try:
                    await chat_matcher.finish(_reply_message(event, ln))
                except Exception:
                    pass
            if j < len(send_lines) - 1:
                await asyncio.sleep(random.uniform(1.2, 2.6))
        if sticker_path:
            await _send_sticker_file(bot, group_id, sticker_path, event)
        print(f"[SENT] group={group_id} persona={'墨提斯' if use_mortis else '若叶睦'} lines={len(send_lines)} text={send_text!r} sticker={os.path.basename(sticker_path) if sticker_path else None}", flush=True)


# ============ 私聊：有人私信她时以人设回复（只响应，绝不主动发私信） ============

# ============ 拍一拍：被戳一下要搭理（零 token 固定文案） ============
# 群友双击头像拍若叶睦 → 一句短回应。用固定文案而非 LLM：拍一拍高频、唤醒 LLM 既慢又烧 token。
# 墨提斯人格概率生效（她更张扬），否则若叶睦（内向、话少）。
POKE_ENABLED = os.getenv("POKE_ENABLED", "1") == "1"
POKE_COOLDOWN = int(os.getenv("POKE_COOLDOWN", "30"))   # 同一人冷却秒数，防连拍刷屏
poke_cooldown: Dict[tuple, float] = {}                  # (group, uid) -> 下次可拍时间

POKE_LINES_MUTSUMI = [
    "……（回头看了你一眼）",
    "（轻轻拍回去）",
    "……嗯。（点了点头）",
    "（愣了一下）……有事吗？",
]
POKE_LINES_MORTIS = [
    "（歪头）戳我干嘛？",
    "（笑着拍回去）手挺欠的嘛。",
    "（盯着你）……你想干嘛？",
    "（挑眉）就这？",
]


def _check_poke(event: PokeNotifyEvent) -> bool:
    """只响应「拍若叶睦本人」的群内拍一拍，且在白名单群、过了冷却。"""
    if not (ENABLE_SPEAKING and POKE_ENABLED):
        return False
    if not isinstance(event, PokeNotifyEvent):
        return False
    if str(event.target_id) != str(event.self_id):
        return False                                # 拍的是别人，不掺和
    if str(event.user_id) == str(event.self_id):
        return False
    if str(event.user_id) == OTHER_BOT_QQ:
        return False                                # 另一只 bot 拍的，忽略（防循环）
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
    """被拍回应：notice 事件没有 message_id，只能纯文本直发（不能带引用）。"""
    gid = str(event.group_id)
    # 墨提斯召唤窗口内被拍 → 由墨提斯应付（人格一致），否则按原概率随机
    use_mortis = time.time() < mortis_until.get(gid, 0) or random.random() < MORTIS_TRIGGER_RATE
    text = random.choice(POKE_LINES_MORTIS if use_mortis else POKE_LINES_MUTSUMI)
    print(f"[POKE] group={gid} user={event.user_id} mortis={use_mortis} -> {text!r}", flush=True)
    try:
        await bot.send_group_msg(group_id=int(gid), message=Message(MessageSegment.text(text)))
    except Exception as e:
        print(f"[POKE SEND ERROR] {type(e).__name__}: {e}", flush=True)


def _check_private_trigger(event, state: T_State) -> bool:
    """私聊触发：总开关开启、仅私聊事件、排除本项目里的其他机器人。"""
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

# 私聊中提到这些词 → 墨提斯人格；否则默认若叶睦人格
MORTIS_KEYWORDS = ["墨提斯", "mortis", "莫提斯", "睦的另一面", "第二人格", "另一个人格"]


def _trim_private_reply(text: str, limit: int = 60) -> str:
    """私聊长度控制：按标点断句截断（私聊是连续对话，比群聊宽松些，人设风格由提示词约束）。"""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for i, ch in enumerate(text[:limit + 1], start=1):
        if ch in "。！？…，、" and i >= 3:
            cut = text[:i]
            break
    return cut + "…"


@private_matcher.handle()
async def handle_private_message(bot: Bot, event: PrivateMessageEvent, state: T_State) -> None:
    """私信回复：按若叶睦/墨提斯人设回话，只回复收到的私信，绝不主动发消息。"""
    uid = str(event.user_id)
    raw = event.get_plaintext().strip()
    # 只有文字消息才回复；图片/表情/语音等私信不回应（同时也保证绝不主动发消息）
    if not raw:
        return
    user = (event.sender.nickname or "").strip() if event.sender else ""
    user = user or "陌生人"
    low = raw.lower()
    # 人格：私聊里提到墨提斯就换成她（原有行为）；
    # 2026-10-16 补显式召唤/退场：说"墨提斯出来"→ 她代班 MORTIS_WINDOW 秒（后续私聊都是她），
    # 说"墨提斯退下"→ 立刻换回若叶睦（同一套 chat_logic.mortis_intent，与群聊一致）
    p_mortis_key = f"p:{uid}"
    p_mortis_active = time.time() < mortis_until.get(p_mortis_key, 0)
    p_intent = chat_logic.mortis_intent(raw, addressed=True, active=p_mortis_active)
    if p_intent == "dismiss":
        mortis_until.pop(p_mortis_key, None)
        print(f"[MORTIS][PRIVATE] uid={uid} 退场：换回若叶睦", flush=True)
    elif p_intent == "summon":
        mortis_until[p_mortis_key] = time.time() + MORTIS_WINDOW
        print(f"[MORTIS][PRIVATE] uid={uid} 召唤成功：墨提斯代班 {int(MORTIS_WINDOW)}s", flush=True)
    use_mortis = (time.time() < mortis_until.get(p_mortis_key, 0)
                  or any(kw in low for kw in MORTIS_KEYWORDS))

    # 按"用户+人格"分开记忆，防两套人格串味；私聊无摘要，靠最近几轮历史
    hist_key = f"private:{uid}:{'mortis' if use_mortis else 'mutsumi'}"
    _remember(hist_key, {"role": "user", "content": f"{user}：{raw[:80]}"})
    history = contexts[hist_key]
    # 防止无限增长
    if len(contexts) > 60:
        for old_key in list(contexts)[:-20]:
            contexts.pop(old_key, None)

    # 语音指令（2026-10-12 补：私聊一直没有语音，群友实锤"私信若叶睦的时候，
    # 若叶睦语音说功能没有了"——"语音说 xxx" 被当普通文本丢给了 LLM）。
    # 与群聊同一套合成（带缓存），冷却按用户计，合成失败自动降级纯文字。
    voice_cmd_text = _extract_voice_cmd(raw)
    if voice_cmd_text is not None:
        cool_key = f"p:{uid}"
        if time.time() - _voice_cmd_last.get(cool_key, 0) < VOICE_CMD_COOLDOWN:
            print(f"[TTS-CMD-COOLDOWN][PRIVATE] uid={uid} 冷却中，文字提示", flush=True)
            await bot.send_private_msg(
                user_id=event.user_id,
                message=Message(MessageSegment.text(
                    random.choice(["……刚说过啦，让我缓缓。", "（揉嗓子）……等一下再说。", "……连续说太多会哑的。"]))))
            return
        _voice_cmd_last[cool_key] = time.time()
        print(f"[TTS-CMD][PRIVATE] uid={uid} text={voice_cmd_text[:30]!r}", flush=True)
        _remember(hist_key, {"role": "assistant", "content": f"（用语音说了：{voice_cmd_text}）"},
                  cross_label="墨提斯" if use_mortis else "若叶睦")
        await _send_private_voice(bot, event, voice_cmd_text)
        return

    base_prompt = SYSTEM_PROMPT_MORTIS if use_mortis else SYSTEM_PROMPT_MUTSUMI
    messages = [{"role": "system", "content": base_prompt}] + history[-SUMMARY_CONTEXT_ROUNDS:]
    # 梗解释（私聊同样"看懂再接"，放在最后贴近当前消息）
    try:
        _pm_hint = _meme_hint(raw)
    except Exception:
        _pm_hint = ""
    if _pm_hint:
        messages.append({"role": "system", "content": _pm_hint})

    # token 熔断：打开期间私聊 LLM 静默（本地功能不受影响）
    if _circuit_open():
        print("[CIRCUIT][PRIVATE] 私聊 LLM 被熔断，静默", flush=True)
        return

    try:
        chat_kwargs = {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            # 回复会被截断到≤60字，按人格压紧上限防模型长篇浪费
            "max_tokens": 2048,
        }
        if DEEPSEEK_MODEL != "deepseek-reasoner":
            chat_kwargs["temperature"] = 0.75  # 原先 0.9 偏高易 OOC，与群聊 0.6 对齐取中间值
        chat_kwargs.update(_reasoning_kwargs("chat"))
        reply = await asyncio.wait_for(
            client.chat.completions.create(**chat_kwargs),
            timeout=150,
        )
        _record_usage(reply.usage)
        text = (reply.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[LLM ERROR][PRIVATE] {type(e).__name__}: {e}", flush=True)
        await private_matcher.finish("……（走神了。等下再叫我。）")
        return

    if not text:
        print("[LLM EMPTY][PRIVATE]", flush=True)
        return

    # 沉默许可：模型选择【沉默】则什么都不发
    if text in SILENCE_MARKERS or (len(text) <= 6 and any(m in text for m in SILENCE_MARKERS)):
        print(f"[SILENCE][PRIVATE] 模型选择沉默: {text!r}", flush=True)
        return

    # 防露馅：AI 味词汇拦截
    if any(m in text for m in AI_MARKERS):
        print(f"[GUARD-AI][PRIVATE] 拦截AI味回复: {text!r}", flush=True)
        text = "……嗯。" if not use_mortis else "（歪头）你说什么？"

    # 拆成一句一条（2026-10-07：与群聊一致；2026-10-12 单条上限 10→15；若叶睦每条≤15字、墨提斯≤60字，最多3条）
    plines = _split_short_lines(text, 60 if use_mortis else MUTSUMI_LINE_LIMIT)
    text = "\n".join(plines)

    _remember(hist_key, {"role": "assistant", "content": text},
              cross_label="墨提斯" if use_mortis else "若叶睦")
    save_memory()          # 记忆落盘（节流）

    # 只回复收到的私信；全代码无任何主动私发路径。多句时逐条发送（一句一条，带间隔）
    for j, ln in enumerate(plines):
        try:
            await bot.send_private_msg(
                user_id=event.user_id,
                message=Message(MessageSegment.text(ln)),
            )
        except Exception as e:
            print(f"[SEND ERROR][PRIVATE] {type(e).__name__}: {e}", flush=True)
        if j < len(plines) - 1:
            await asyncio.sleep(random.uniform(1.2, 2.6))
