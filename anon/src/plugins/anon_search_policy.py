"""Pure retrieval-policy helpers for the Anon chat plugin (WP1, 2026-10-26).

Standard library only: no NoneBot / OpenAI / agent_search / anon_context /
anonbot imports, no network.  Every public function tolerates ``""``/``None``
and never raises, so the whole policy is unit-testable offline.

``anon_search_policy`` answers the questions the chat pipeline asks before it
replies:

``strip_noise``
    What did the user actually say, with QQ noise (@/reply/CQ/media) stripped?
``glossary_hits`` / ``glossary_hint``
    Is this verified group jargon we can explain without searching?
``detect_lookup``
    Should we search the web, and with which 1-2 queries?
``classify_request``
    Is this a request for an action the bot cannot perform?
``action_note``
    What boundary must the voice pass respect for that action?
``pick_face_action``
    Which classic face may accompany a ``send_face`` request?

The rule tables below are the single source of truth from the 2026-10-26
logic document §5.1 / HR prompt §3.3; keep them declarative so WP2 can rely on
stable outputs.
"""
import re

__all__ = [
    "GLOSSARY",
    "strip_noise",
    "glossary_hits",
    "glossary_hint",
    "detect_lookup",
    "classify_request",
    "action_note",
    "pick_face_action",
]


# ---------------------------------------------------------------------------
# A. strip_noise
# ---------------------------------------------------------------------------

# CQ 码：[CQ:at,qq=200000003]
_CQ_RE = re.compile(r"\[CQ:[^\]]*\]", re.IGNORECASE)
# OneBot 展示段：[at:qq=...] [reply:id=...] [image:...] ...
_ONEBOT_SEGMENT_RE = re.compile(
    r"\[(?:at|reply|image|record|video|file|face)(?::[^\]]*)?\]", re.IGNORECASE
)
# 中文媒体占位段：[图片] [语音] [表情包] ...
_CN_SEGMENT_RE = re.compile(
    r"\[(?:图片|语音|视频|文件|表情包|动画表情|表情|音乐|红包|链接|位置|名片|回复)"
    r"\s*:?[^\]]*\]"
)
# @昵称：@ 后到空白/标点前的 1~30 个字符
_AT_NAME_RE = re.compile(
    r"@[^\s，,。.！!？?~～、；;：:…“”‘’\"'《》【】()（）\[\]]{1,30}"
)
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_WHITESPACE_RE = re.compile(r"\s+")

# 与规格 §3.3-A 第 6 步一致：只剥这些标点，书名号/括号保留（演出评价要用
# 《曲名》做具体性判断）。
_EDGE_PUNCT = "，,。.！!？?~～、；;：:…“”‘’\"'"
# 开头称呼（爱音/Anon/千早爱音）：只在后面是空白/标点/结尾时剥掉，
# 避免把“爱音的声优是谁”这种正常句子截坏。
_LEADING_NAME_RE = re.compile(
    r"^(?:千早爱音|爱音|(?i:anon))"
    r"(?=$|[\s，,。.！!？?~～、；;：:…“”‘’\"'《》〈〉【】()（）\[\]@])"
)


def _strip_edges(value: str) -> str:
    """去掉首尾空白与首尾标点（循环到稳定，处理“《 ”这种组合）。"""
    previous = None
    while value != previous:
        previous = value
        value = value.strip().strip(_EDGE_PUNCT)
    return value


def strip_noise(text: str) -> str:
    """去掉 QQ 消息噪声，返回适合做检索词的干净文本。

    依次删除 CQ 码、OneBot 展示段（at/reply/image/record/video/file/face）、
    中文媒体占位段、``@昵称``、零宽字符；把 ``\\u3000`` 与连续空白压成一个
    半角空格；去掉首尾空白/标点；最后剥掉开头的“千早爱音/爱音/Anon”称呼。
    中间文字、大小写、繁简一律不改。空输入/None 返回 ``""``。
    """
    if not isinstance(text, str) or not text:
        return ""
    value = _CQ_RE.sub("", text)
    value = _ONEBOT_SEGMENT_RE.sub(" ", value)
    value = _CN_SEGMENT_RE.sub(" ", value)
    value = _AT_NAME_RE.sub(" ", value)
    value = _ZERO_WIDTH_RE.sub("", value)
    value = _WHITESPACE_RE.sub(" ", value)
    value = _strip_edges(value)
    while value:
        match = _LEADING_NAME_RE.match(value)
        if not match:
            break
        value = _strip_edges(value[match.end():])
    return value


# ---------------------------------------------------------------------------
# B. GLOSSARY / glossary_hits / glossary_hint
# ---------------------------------------------------------------------------

# 每个词条一行：别名元组 + 释义。别名同组内只返回一次（取文本中最早出现的
# 写法）；整张表的顺序就是 glossary_hits 的返回顺序。
_GLOSSARY_TABLE = (
    (("tomorin",), "高松灯"),
    (("凑企鹅",), "高松灯"),
    (("苏幽璃",), "长崎素世（古风二创名）"),
    (("疯串箱子", "疯穿箱子", "疯氚箱子"), "丰川祥子（谐音）"),
    (("母鸡卡",), "Ave Mujica"),
    (("邦娃", "棉娃"), "BanG Dream 棉花娃娃"),
    (("瑞萍",), "锐评（谐音）"),
    (("zmd",), "明日方舟：终末地"),
    (("立石凛",), "爱音的声优"),
    (("李世林", "立凛石", "立戒石", "江界石"), "立石凛的故意谐音"),
    (("羊宫妃那",), "高松灯的声优"),
    (("青木阳菜",), "要乐奈的声优"),
)

# 对外词表：别名 -> 含义（顺序与判定表一致）。需要分组语义时用内部表。
GLOSSARY = {
    alias: meaning for aliases, meaning in _GLOSSARY_TABLE for alias in aliases
}


def _alias_pattern(alias: str) -> "re.Pattern[str]":
    """ASCII 词加字母数字边界；中文词按子串匹配。"""
    if re.fullmatch(r"[A-Za-z0-9]+", alias):
        return re.compile(
            r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])" % re.escape(alias),
            re.IGNORECASE,
        )
    return re.compile(re.escape(alias))


_GLOSSARY_PATTERNS = tuple(
    (tuple((alias, _alias_pattern(alias)) for alias in aliases), meaning)
    for aliases, meaning in _GLOSSARY_TABLE
)

_HINT_HEADER = "【群内黑话】（群内用法优先于搜索标题；不确定就短句问，不要硬套）"


def glossary_hits(text: str) -> list[tuple[str, str]]:
    """返回命中的群内黑话 ``[(说法, 含义), ...]``（按词表顺序，每词条最多一次）。

    同组多个别名都出现时，``说法`` 取文本中最早出现的那一个（保留原文写法）。
    空输入/None 返回 ``[]``。
    """
    if not isinstance(text, str) or not text:
        return []
    hits = []
    for patterns, meaning in _GLOSSARY_PATTERNS:
        best_pos = None
        best_term = None
        for _alias, pattern in patterns:
            match = pattern.search(text)
            if match is None:
                continue
            if best_pos is None or match.start() < best_pos:
                best_pos = match.start()
                best_term = match.group(0)
        if best_term is not None:
            hits.append((best_term, meaning))
    return hits


def glossary_hint(text: str) -> str:
    """把命中的黑话拼成给模型看的一段提示；无命中返回 ``""``。"""
    hits = glossary_hits(text)
    if not hits:
        return ""
    lines = [_HINT_HEADER]
    lines.extend("- %s = %s" % (term, meaning) for term, meaning in hits)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# C. detect_lookup
# ---------------------------------------------------------------------------

_MAX_LOOKUP_CHARS = 200
_MAX_QUERY_CHARS = 60

# C1 明确让搜。按“最长优先”排序，避免“搜索一下”被“搜”截成“索一下”。
# “查询一下”必须在“查”之前；“搜刮”是词不是检索请求，不能截成“刮一下……”。
_EXPLICIT_VERB_RE = re.compile(
    r"搜索一下|百度一下|谷歌一下|google\s*一下|搜一下|搜搜|搜索|查询一下|查询|搜(?!刮)"
    r"|(?<![调审检复普追核面考抽暗])查(?:一查|一下|查)?",
    re.IGNORECASE,
)
_EXPLICIT_LEAD_RE = re.compile(r"^(?:一下|一搜|下)\s*")
_EXPLICIT_TAIL_RE = re.compile(r"(?:吧|呀|啦|谢谢|呗)+$")

# 作业/解题不按网页检索处理
_HOMEWORK_RE = re.compile(r"这道题|这题|作业|代码|作文|翻译|怎么解|怎么写|怎么做")

# C2 不搜的负例
_OPINION_RE = re.compile(
    r"谁更强|谁更厉害|谁更好看|谁才是|更强|goat|你更喜欢|哪个更好", re.IGNORECASE
)
_STATUS_RE = re.compile(
    r"^(?:你|爱音|anon)?\s*(?:最近|今天|现在|这几天)\s*"
    r"(?:怎么样|好吗|还好吗|在干嘛|干嘛呢|过得怎么样)",
    re.IGNORECASE,
)
_GREETING_ONLY = frozenset(
    {"在吗", "在不在", "早", "早安", "晚安", "你好", "哈喽", "hi", "hello"}
)
_MOOD_WEATHER_RE = re.compile(r"天气(?:怎么样|如何)|好累|好丧|心情不好|不想动")
_TIME_ASK_RE = re.compile(r"现在几点|几点了")

# C3 求定义
_DEFINITION_PREFIX_RE = re.compile(
    r"^(?:请问|问一下|我想问)[，,。.！!？?~～、；;：:\s]*"
)
_DEFINITION_PATTERNS = (
    re.compile(r"^(?:谁|誰)是(?P<x>.{1,24})$"),
    re.compile(r"^(?P<x>.{1,24}?)(?:是什么梗|什么梗|是什麼梗)$"),
    re.compile(r"^(?P<x>.{1,24}?)(?:什么意思|啥意思|是什么意思|是什麼意思)$"),
    re.compile(r"^(?P<x>.{1,24}?)(?:什么来头|是什么来头|什麼來頭)$"),
    re.compile(r"^(?P<x>.{1,24}?)(?:是谁|是啥|是誰|是甚麼|是甚么)$"),
    re.compile(r"^(?P<x>.{1,24}?)(?:是什么|是甚么|是什麼)$"),
    re.compile(r"^什么是(?P<x>.{1,24})$"),
)
_X_EDGE_TRIM = "「」『』“”‘’\"'《》〈〉【】()（）[]"
_X_TAIL_RE = re.compile(r"(?:啊|呀|呢|吧|啦)+$")
_IDENTITY_TARGETS = frozenset(
    {
        "你", "我", "他", "她", "它", "大家", "这个", "那个", "这", "那", "啥",
        "ai", "人工智能", "语言模型", "chatgpt",
    }
)
_MYGO_CONTEXT_RE = re.compile(
    r"(?i)mygo|邦邦|bang\s*dream|ave\s*mujica|avemujica|母鸡卡|高松灯|千早爱音"
    r"|长崎素世|丰川祥子|要乐奈|椎名立希|八幡海铃"
)

# C4 具体演出/作品评价
_PERF_EVAL_RE = re.compile(
    r"怎么样|如何|评价|好听|弹得|表现|水平|稳不稳|翻车|厉害吗"
)
# 钟点（8:30）单独出现只是时间，不是演出锚点；只有和强演出词同时出现时才算。
_PERF_CLOCK_RE = re.compile(r"\d{1,2}:\d{2}")
_PERF_STRONG_MARK_RE = re.compile(
    r"(?i)solo|现场|live|演唱会|巡演|tour|演出|弹|唱|演奏|第?\d+分\d+秒"
    r"|《[^》]{1,20}》"
)
_PERF_MARK_RE = re.compile(
    r"(?i)solo|现场|live|演唱会|巡演|tour|演出|弹|唱|演奏|第?\d+分\d+秒"
    r"|\d{1,2}:\d{2}|《[^》]{1,20}》"
)
_PERF_SPECIFIC_RE = re.compile(
    r"\d{4}\s*[-/年.]\s*\d{1,2}\s*[-/月.]\s*\d{1,2}"
    r"|\d{1,2}月\d{1,2}日|第?\d+分\d+秒|《[^》]{1,20}》"
)

# C5 求现状：只有“最近/现在/目前”这类时间词的多半是闲聊，必须同时有
# 外部事实词（价格/开售/新消息/什么时候……）才检索。
_CURRENT_STRONG_RE = re.compile(
    r"最新|价格|票价|开售|发售|上线|更新|版本|新歌|新番"
    r"|什么时候|几号|哪里买|在哪买|新消息|怎么了"
)
_CURRENT_WEAK_RE = re.compile(r"最近|现在|目前|今年")
_CURRENT_FILLER_RE = re.compile(
    r"怎么样|如何|有什么|有没有|是什么|多少|哪里|在哪|什么时候|几号"
    r"|会|要|能|可以|我|你|他|她|它|大家|这个|那个|一下|的|了|吗|呢|吧"
    r"|啊|呀|嘛|哦|喔|是|有|在"
)
# “你什么时候回来 / 明天什么时候集合 / 群里怎么了 / 今天几号”是日程/寒暄，
# 不是要查外部事实；“母鸡卡什么时候出新歌”不以这些词开头，仍会搜。
_PERSONAL_WHEN_RE = re.compile(
    r"^(?:你|我|我们|大家|群里|今天|明天|后天|这周|下周|周末|"
    r"周[一二三四五六日天]|星期[一二三四五六日天])"
    r"[^，。！？!?]{0,12}(?:什么时候|几号|几点|怎么了|回来|吃饭|出发|集合|"
    r"上课|下课|睡觉|起床)"
)
_PERSONAL_WHEN_BARE_RE = re.compile(
    r"^(?:什么时候|几号|几点)"
    r"(?:吃饭|出发|集合|回来|上课|下课|睡觉|起床|走|到)"
)
_NON_WORD_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def _dedupe(values, limit: int = _MAX_QUERY_CHARS) -> "list[str]":
    result = []
    seen = set()
    for value in values:
        value = (value or "").strip()[:limit]
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _explicit_lookup(clean: str):
    match = _EXPLICIT_VERB_RE.search(clean)
    if match is None:
        return None
    target = clean[match.end():]
    target = _EXPLICIT_LEAD_RE.sub("", target.strip())
    target = _strip_edges(target)
    target = _EXPLICIT_TAIL_RE.sub("", target)
    target = _strip_edges(target)
    if not 1 <= len(target) <= 40:
        return None
    if _HOMEWORK_RE.search(target):
        return None
    queries = _dedupe([target, "%s 是什么" % target])
    return {"query": queries[0], "queries": queries, "reason": "explicit"}


def _is_negative(clean: str) -> bool:
    if _OPINION_RE.search(clean):
        return True
    if _STATUS_RE.search(clean):
        return True
    if clean.lower() in _GREETING_ONLY:
        return True
    if _MOOD_WEATHER_RE.search(clean):
        return True
    if _TIME_ASK_RE.search(clean):
        return True
    return False


def _definition_lookup(clean: str, summary: str, context: str):
    body = _DEFINITION_PREFIX_RE.sub("", clean)
    target = None
    for pattern in _DEFINITION_PATTERNS:
        match = pattern.match(body)
        if match is not None:
            target = match.group("x")
            break
    if target is None:
        return None
    target = _strip_edges(target.strip().strip(_X_EDGE_TRIM))
    target = _X_TAIL_RE.sub("", target)
    target = _strip_edges(target)
    if not 1 <= len(target) <= 24:
        return None
    if target.lower() in _IDENTITY_TARGETS:
        return None
    # 人设/自称类问题（“你最喜欢的美妆博主是谁”“爱音是谁”）用角色卡回答，不查网页。
    if re.search(r"(?i)你|我|爱音|anon", target):
        return None

    if re.search(r"梗|什么意思|啥意思", clean):
        queries = _dedupe(["%s 是什么梗" % target, "%s MyGO" % target])
        return {"query": queries[0], "queries": queries, "reason": "definition"}

    queries = ["%s 是谁" % target]
    group_context = "%s %s" % (summary if isinstance(summary, str) else "",
                               context if isinstance(context, str) else "")
    # 群内黑话本身就是 MyGO/邦邦语境，同样补第二条查询。
    if _MYGO_CONTEXT_RE.search(group_context) or glossary_hits(target):
        queries.append("%s MyGO" % target)
    queries = _dedupe(queries)
    return {"query": queries[0], "queries": queries, "reason": "definition"}


def _performance_lookup(clean: str):
    if not _PERF_EVAL_RE.search(clean):
        return None
    if not _PERF_MARK_RE.search(clean):
        return None
    if not _PERF_SPECIFIC_RE.search(clean):
        # “明天 8:30 集合怎么样”里的钟点不是演出证据；只有同时出现
        # solo/现场/弹/唱 等强演出词，才把“8:30”当成具体演出锚点。
        if not (_PERF_CLOCK_RE.search(clean) and _PERF_STRONG_MARK_RE.search(clean)):
            return None
    query = clean[:_MAX_QUERY_CHARS]
    return {"query": query, "queries": [query], "reason": "performance"}


def _current_lookup(clean: str):
    if not (_CURRENT_STRONG_RE.search(clean) or _CURRENT_WEAK_RE.search(clean)):
        return None
    # “你现在玩什么游戏”“大家最近在玩什么”只有弱时间词，是普通闲聊。
    if not _CURRENT_STRONG_RE.search(clean):
        return None
    if _PERSONAL_WHEN_RE.search(clean) or _PERSONAL_WHEN_BARE_RE.search(clean):
        return None
    rest = _CURRENT_STRONG_RE.sub(" ", clean)
    rest = _CURRENT_WEAK_RE.sub(" ", rest)
    rest = _CURRENT_FILLER_RE.sub(" ", rest)
    rest = _NON_WORD_RE.sub("", rest)
    if len(rest) < 2:
        return None
    query = clean[:_MAX_QUERY_CHARS]
    return {"query": query, "queries": [query], "reason": "current"}


# ---------------------------------------------------------------------------
# C6. 评价具体公开对象（WP6，2026-10-26）
# ---------------------------------------------------------------------------
# 只认“评价/锐评/如何评价/怎么看”+ 2~30 字具体对象；纯比较/情绪/作业/
# 个人创作/图片/爱音自身/游戏卡组/天气状态不搜。
_EVAL_TRIGGER_RE = re.compile(
    r"如何评价|怎么评价|怎样评价|如何看待|怎么看待"
    r"|评价(?:一下|下|一评)?|锐评(?:一下|下)?|点评(?:一下|下)?"
    r"|你怎么看|你如何看|你有什么看法|怎么看|如何看",
    re.IGNORECASE,
)
_EVAL_LEAD_RE = re.compile(
    r"^(?:请问|请|帮我|帮忙|你|爱音|anon|说说|谈谈|讲讲|关于|对于|一下|一评)\s*",
    re.IGNORECASE,
)
_EVAL_TAIL_RE = re.compile(
    r"(?:吧|呀|啊|呢|吗|呗|谢谢|一下|怎么说|怎么样|如何|咋样|你|我)+$"
)
_EVAL_GENERIC = frozenset({
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "大家", "群友",
    "这个", "那个", "这些", "那些", "这件事", "那件事", "这个东西", "那个东西",
    "啥", "什么", "ai", "人工智能", "语言模型", "机器人", "公司", "学校",
    "老师", "老板", "今天", "明天", "昨晚", "刚才", "生活", "人生", "世界",
    "我妈", "我爸", "你妈", "你爸",
})
_EVAL_ANON_RE = re.compile(
    r"^(?:你|爱音|千早爱音|anon|你自己|爱音自己|我|我自己|我们)$",
    re.IGNORECASE,
)
_EVAL_COMPARE_RE = re.compile(
    r"谁更强|谁更厉害|谁更好|谁才是|更强|更厉害|goat|哪个更好|对比|比较|\bvs\b",
    re.IGNORECASE,
)
_EVAL_PERSONAL_RE = re.compile(
    r"(?:我|我的)(?:唱|写|画|做|拍|弹|跳|剪|录|说|讲|演|穿|买|选|改|修|弄|发|生成|创作|设计)"
    r"|我(?:的)?(?:歌|曲|画|图|照片|视频|作业|论文|代码|作品|文案|菜|饭|衣服|穿搭|声音|唱功|水平)"
    r"|我(?:妈|爸|朋友|同学|群友|对象|老公|老婆|男朋友|女朋友)(?:做|写|画|唱|拍|买|选|改|修|弄)的"
    r"|我(?:推|喜欢|粉|pick|担)"
)
_EVAL_SCOPE_RE = re.compile(
    r"^(?:我|我们|你|你们|他|他们)(?:的)?"
    r"(?:老师|老板|公司|学校|班|群|朋友|同学|家人|对象|老公|老婆)"
)
_EVAL_IMAGE_RE = re.compile(
    r"图片|照片|截图|表情包|视频|语音|录屏"
    r"|这张图|这图|那张图|这张照片|这个视频|这段视频|这个截图|这个表情包"
    r"|这首歌|那首歌|这个游戏|那个游戏"
)
_EVAL_GAME_META_RE = re.compile(r"卡组|牌组|阵容|配队|出装|卡池|卡牌|套路|build", re.IGNORECASE)
_EVAL_WEATHER_RE = re.compile(r"天气|心情|状态|运气|梦|这件事|那件事")
_EVAL_STATUS_RE = re.compile(
    r"最近怎么样|最近好吗|现在怎么样|在干嘛|过得怎么样|过得好吗|还好吗|这几天"
)


def _evaluation_lookup(clean: str):
    """评价具体公开对象 -> 检索；返回 evaluation 或 None。

    对象取触发词前/后较长的一侧；只接受 2~30 字；排除纯比较、情绪、
    作业、个人创作、图片、爱音自身、游戏卡组、天气/状态、泛称。
    """
    match = _EVAL_TRIGGER_RE.search(clean)
    if match is None:
        return None
    if _HOMEWORK_RE.search(clean) or _EVAL_STATUS_RE.search(clean):
        return None
    before = _EVAL_LEAD_RE.sub("", _strip_edges(clean[: match.start()]))
    after = _strip_edges(clean[match.end():])
    before = _EVAL_TAIL_RE.sub("", before).strip()
    after = _EVAL_TAIL_RE.sub("", after).strip()
    target = after if len(after) >= 2 else before
    target = _EVAL_TAIL_RE.sub("", target.strip(_X_EDGE_TRIM)).strip()
    target = target.strip(_X_EDGE_TRIM)
    if not 2 <= len(target) <= 30:
        return None
    low = target.lower()
    if low in _EVAL_GENERIC or _EVAL_ANON_RE.match(target):
        return None
    if _EVAL_COMPARE_RE.search(clean) or _OPINION_RE.search(clean):
        return None
    if _EVAL_PERSONAL_RE.search(target) or _EVAL_SCOPE_RE.match(target):
        return None
    if _EVAL_IMAGE_RE.search(target) or _EVAL_GAME_META_RE.search(target):
        return None
    if _EVAL_WEATHER_RE.search(target):
        return None
    if _MOOD_WEATHER_RE.search(clean) or _STATUS_RE.search(clean):
        return None
    queries = _dedupe(["%s 评价" % target, target])
    return {"query": queries[0], "queries": queries, "reason": "evaluation"}


def detect_lookup(text: str, summary: str = "", context: str = "") -> dict | None:
    """判断这条消息是否需要联网检索。

    返回 ``{"query": str, "queries": [str, ...], "reason": str}`` 或 ``None``。
    优先级：明确让搜(explicit) > 不搜负例 > 求定义(definition) >
    具体演出/作品评价(performance) > 公开对象评价(evaluation) > 求现状(current)。
    """
    clean = strip_noise(text)
    if not clean or len(clean) > _MAX_LOOKUP_CHARS:
        return None
    found = _explicit_lookup(clean)
    if found is not None:
        return found
    if _is_negative(clean):
        return None
    found = _definition_lookup(clean, summary, context)
    if found is not None:
        return found
    found = _performance_lookup(clean)
    if found is not None:
        return found
    found = _evaluation_lookup(clean)
    if found is not None:
        return found
    return _current_lookup(clean)


# ---------------------------------------------------------------------------
# D. classify_request
# ---------------------------------------------------------------------------

_PURCHASE_RE = re.compile(
    r"抢(?:一下|个|张|下)?[^，。！？?]{0,20}票|买票|购票|订票|下单|付款|支付"
    r"|转账|登录|登陆|签到|代购|抢购|注册|验证码|帮我买|替我买|帮我付"
)
# “可买票要抽签啊”“买票得排队”是陈述/感慨，不是让爱音去执行；只有带请求语气
# （帮我/替我/给我/你/能不能…）或直接命令（“抢票”“签到”）才归为 purchase_login。
_PURCHASE_STATEMENT_RE = re.compile(r"可|要|得|需要|还得|只能|必须")
_REQUEST_CUE_RE = re.compile(
    r"帮我|替我|给我|请你|麻烦你|能不能|可不可以|可以吗|你"
)
# “老师转告我明天考试”“他带话说周末聚”是第三人称叙述，不是让爱音去传话；
# 传话必须带请求语气（帮我/替我/你/能不能）或在分句开头直接下命令。
_CONTACT_RE = re.compile(
    r"帮我(?:发消息|发个消息|告诉|转告|联系|私聊|带话)"
    r"|替我(?:发|告诉|转告|联系|带话)"
    r"|(?:^|[\s，,。！!？?])"
    r"(?:请|麻烦你?|你|爱音|anon)?\s*(?:去|来)?\s*(?:转告|带话)"
    r"|(?:请你?|麻烦你?|能不能|可不可以|你|爱音|anon)"
    r"[^，,。！!？?\n]{0,4}?(?:转告|带话)"
)
# 媒体请求也必须有请求语境：“青木阳菜唱歌怎么样”“灯唱歌真好听”
# “她发图给我了”是聊天/叙述，不能判成让爱音发图或唱歌。
_MEDIA_VERB = (
    r"(?:发(?!了)(?:个|张|一个|一张|段)?(?:图|图片|照片|表情包|语音|视频|文件|录音)"
    r"|来(?:个|张|段)?(?:图|表情包|语音|视频)"
    r"|唱(?:一|个|首|段)|唱歌"
    r"|录(?:一|个|段)|拍(?:一|个|张)"
    r"|发我(?:一|个|张|段)?(?:图|语音|视频|文件)"
    r"|(?:图|照片|语音|视频|文件)发我(?!了))"
)
_MEDIA_RE = re.compile(
    r"(?:^|[\s，,。！!？?])"
    r"(?:请|来|快|帮我|给我|替我|麻烦你?|能不能|可不可以|能|可以)?\s*"
    + _MEDIA_VERB
    + r"|(?:帮我|给我|替我|请你?|麻烦你?|能不能|可不可以|你|爱音|anon|把)"
    r"[^，,。！!？?\n]{0,4}?"
    + _MEDIA_VERB
)
_MEDIA_NEGATIVE_RE = re.compile(
    r"发图(?:不说话|干嘛|干什么|怎么)|为什么(?:不)?发图|怎么不发"
)
_FACE_RE = re.compile(
    r"发(?:个|一个)?(?:经典)?表情(?!包)"
    r"|卖(?:个)?萌(?=$|[，。！!？?～\s]|一下|一个|呗|吧|给我看)"
    r"|比(?:个|一个)?心"
    r"|给我笑一个|笑一个"
)
_SEARCH_WORD_RE = re.compile(
    r"查资料|查一下|搜一下|搜搜|搜索|科普|解释(?:一下)?|推荐(?:一下|点)?"
)
_SEARCH_SKIP_RE = re.compile(r"搜|搜索|查资料")
# 创作/评价/建议类：动作词只是话题（“写一段唱歌的歌词”“写一份抢票攻略”
# “评价一下她唱歌”），不是让爱音真的去发图/唱歌/抢票。
_CREATIVE_TASK_RE = re.compile(r"写|生成|整理|攻略|教程|文案|歌词|计划|方案|评价|锐评|建议")
_DIRECT_REQUEST_RE = re.compile(
    r"你|爱音|anon|给我|帮我|替我|请你|麻烦你|能不能|可不可以|可以吗",
    re.IGNORECASE,
)
_FIRST_PERSON_START_RE = re.compile(r"^(?:我|我们|大家|群里)")
# 夸/评价别人唱歌不是点歌请求。
_MEDIA_COMPLIMENT_RE = re.compile(r"好听|难听|厉害|怎么样|喜欢|可爱|帅|棒")


def classify_request(text: str) -> str:
    """把群友请求归类为机器人可执行的意图。

    返回 ``"none" | "search" | "send_face" | "send_media" |
    "contact_third_party" | "purchase_login"``。优先级：
    purchase_login > contact_third_party > send_media > send_face > search。
    空输入/None 返回 ``"none"``。
    """
    clean = strip_noise(text)
    if not clean:
        return "none"
    # WP6：评价具体公开对象是检索请求；先于“评价=创作话题”的 _CREATIVE_TASK_RE。
    if _evaluation_lookup(clean) is not None:
        return "search"
    # 创作/评价/建议类任务：动作词是话题，不是执行请求。
    if _CREATIVE_TASK_RE.search(clean):
        return "none"
    # “我们一起唱歌吧”“我建议你先登录”是陈述/建议，不是让爱音执行。
    if _FIRST_PERSON_START_RE.match(clean) and not _DIRECT_REQUEST_RE.search(clean):
        return "none"
    # “唱歌好难”“你唱歌真好听”“青木阳菜唱歌怎么样”是评价，不是点歌；
    # 单独一句“唱歌”仍按点歌处理（老测试行为）。
    if len(clean) > 2 and clean.startswith("唱歌") and not _DIRECT_REQUEST_RE.search(clean):
        return "none"
    if _MEDIA_COMPLIMENT_RE.search(clean) and not _DIRECT_REQUEST_RE.search(clean):
        return "none"
    if _PURCHASE_RE.search(clean):
        if _PURCHASE_STATEMENT_RE.search(clean) and not _REQUEST_CUE_RE.search(clean):
            return "none"
        return "purchase_login"
    if _CONTACT_RE.search(clean):
        return "contact_third_party"
    if _MEDIA_RE.search(clean) and not _MEDIA_NEGATIVE_RE.search(clean):
        return "send_media"
    if _FACE_RE.search(clean):
        return "send_face"
    if _SEARCH_WORD_RE.search(clean) or detect_lookup(clean) is not None:
        if _HOMEWORK_RE.search(clean) and not _SEARCH_SKIP_RE.search(clean):
            return "none"
        return "search"
    return "none"


# ---------------------------------------------------------------------------
# E. action_note
# ---------------------------------------------------------------------------

_ACTION_NOTES = {
    "none": "",
    "search": (
        "先看群内黑话和公开搜索材料再回答；材料里没有的就说不确定，不要编。"
        "评价类问题要区分【搜到的公开评价】和【你自己的看法】：先复述材料里明确写到的公开评价"
        "（说明是网上/搜到的），再给爱音自己的看法；不要说我看了/我在现场，也不要把公开评价说成亲历；"
        "没查到可靠资料就直说没查到，不要编现场感。"
    ),
    "send_face": (
        "群友要的是轻松互动：可以配合发一个经典表情（heart/shy/smile 三选一），"
        "只发一个，不刷屏。"
    ),
    "send_media": (
        "你只能发文字：不能发图片、表情包、语音、视频或文件。用爱音的口吻说明"
        "做不到，并给一个替代（文字描述、帮查资料或给建议）；不要假装已经发了。"
    ),
    "contact_third_party": (
        "你不能替群友发消息、转告或私聊第三人。用爱音的口吻说明边界，建议他自己"
        "去说；能帮的（例如帮他想措辞）可以先帮。"
    ),
    "purchase_login": (
        "你不能抢票、登录、付款、下单或签到，也不能索要账号密码。说明做不到；"
        "可以帮查公开的开售时间、场次或规则，不要假装执行。"
    ),
}


def action_note(kind: str) -> str:
    """给 voice pass 的中文边界提示；未知 kind 返回 ``""``。"""
    if not isinstance(kind, str):
        return ""
    return _ACTION_NOTES.get(kind, "")


# ---------------------------------------------------------------------------
# F. pick_face_action
# ---------------------------------------------------------------------------

_FACE_HEART_RE = re.compile(r"比(?:个|一个)?心|爱心|喜欢|爱你|抱抱|贴贴|亲亲|heart", re.I)
_FACE_SHY_RE = re.compile(r"害羞|脸红|卖(?:个)?萌|撒娇|可爱|羞")
_FACE_SMILE_RE = re.compile(r"笑|开心|好耶|棒|厉害|夸")


def pick_face_action(text: str) -> str:
    """``send_face`` 请求时选一个经典表情；其余情况返回 ``"none"``。

    优先级 heart > shy > smile；send_face 默认 ``"face:smile"``。
    """
    if classify_request(text) != "send_face":
        return "none"
    clean = strip_noise(text)
    if _FACE_HEART_RE.search(clean):
        return "face:heart"
    if _FACE_SHY_RE.search(clean):
        return "face:shy"
    if _FACE_SMILE_RE.search(clean):
        return "face:smile"
    return "face:smile"
