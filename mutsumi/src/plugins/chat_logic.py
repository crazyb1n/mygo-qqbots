"""Pure decision and prompt-building helpers for the Mutsumi chat plugin.

This module deliberately has no NoneBot/OpenAI imports.  It is safe to import in
offline checks and keeps the most regression-prone conversation rules testable.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence


# ============ 主观问题识别（2026-10-16 用户要求）============
# "涉及主观问题的询问，若叶睦要用符合人设的语气总结回复，不是单纯的检索问题"。
# 用途：这类问题即使走了查资料（researcher）链路，也要她用自己的话总结（先结论后依据、简短），
# 而不是交付一份 3~6 条的"资料汇编"；交付时也不再加"……写完了"这类任务口气的前缀。
# 判定是启发式：误判只影响"答案的语气/长短"，不影响事实来源（事实仍以搜索结果为准）。
_SUBJECTIVE_RE = re.compile(
    r"(你觉得|你认为|你看呢|你们觉得|大家觉得|你怎么看|你们怎么看|怎么看|看法|评价|感想|"
    r"谁更(强|厉害|好|好看|好听|好玩|牛)|哪个(更|好|强|值得)|哪家(好|强)|"
    r"值不值|值得(买|入|看|玩|去)|推不推荐|推荐哪|该不该|要不要|好不好|"
    r"(好吃|好玩|好看|好听|好喝|难看|难吃)吗|喜欢吗|帅吗|可爱吗|"
    r"怎么样|如何)"
)


def is_subjective_ask(raw: str) -> bool:
    """消息是不是"主观问题"（问看法/喜好/评价/比较/推荐/值不值）。"""
    return bool(_SUBJECTIVE_RE.search(raw or ""))


# ============ 墨提斯（Mortis）召唤 / 退场（2026-10-16 用户要求"要有召唤莫提斯的方式"）============
# 人格平时：被 @/被引用时按 MORTIS_TRIGGER_RATE 概率换墨提斯；
# 现在加显式开关：说"墨提斯出来/召唤墨提斯"→ 她出来代班一段时间；"墨提斯退下"→ 换回若叶睦。
_MORTIS_NAME_RE = re.compile(r"(墨提斯|莫提斯|mortis|モーティス)", re.I)
# 退场口气（只在消息里已出现墨提斯名字时才用，所以"回去/休息"这类常见词放这里也安全；
# 先判退场，避免"墨提斯退下吧"被"让/来"类召唤词误判成召唤）
_MORTIS_DISMISS_RE = re.compile(
    r"(退下|下去|回去|回来吧|走开|别闹|闭嘴|醒醒|先歇着|休息|下班|"
    r"让(若叶睦|小睦|睦)说|把(若叶睦|小睦|睦)换回来|我要(若叶睦|小睦|睦))"
)
# 没提墨提斯名字时也认的"换回来"指令（更严：只在墨提斯正代班时生效，避免"我先回去了"误伤）
_MORTIS_DISMISS_STRONG_RE = re.compile(
    r"(让(若叶睦|小睦|睦)说|换回(若叶睦|小睦|睦)|把(若叶睦|小睦|睦)换回来|"
    r"(若叶睦|小睦|睦)(出来|说话|来)|我要(若叶睦|小睦|睦)|墨提斯(退下|回去|走吧))"
)
# 召唤口气：动词必须紧邻名字（前后 ≤4 字），否则长句里随便一个"说说/来"都会误判召唤
_MORTIS_VERB = r"(?:出来|召唤|上线|上号|代班|接班|顶一下|换人|请|叫|让|找|来|说说|说话|你来说|醒醒)"
_MORTIS_SUMMON_NEAR_RE = re.compile(
    rf"(?:{_MORTIS_VERB}[^。！？!?\n]{{0,4}}(?:墨提斯|莫提斯|mortis|モーティス))|"
    rf"(?:(?:墨提斯|莫提斯|mortis|モーティス)[^。！？!?\n]{{0,4}}{_MORTIS_VERB})",
    re.I,
)
# 被叫到且几乎只说了名字（"@她 墨提斯"）也算召唤；清洗标点/@ 后的长度门槛
_MORTIS_CLEAN_RE = re.compile(r"[\s@，。！？!?~～…、,.：:；;\"'（）()【】\[\]]+")


def mortis_intent(raw: str, addressed: bool = False, active: bool = False) -> str | None:
    """识别"召唤墨提斯 / 让墨提斯退场"意图（零 token）。返回 'summon' / 'dismiss' / None。

    - 出现墨提斯的名字（含艺名 mortis）时：退场口气 → dismiss；召唤口气（动词紧邻名字）
      → summon；`addressed=True` 且清洗后几乎只有名字（≤8 字）→ 也算 summon
      （"@她 墨提斯"最自然的叫法）。
    - `active=True`（当前正由墨提斯代班）且没提名字但说了"让若叶睦说话/换回来"→ dismiss，
      否则不提名字一律 None（避免把群里的"我先回去了"当成换人指令）。
    """
    raw = raw or ""
    if not _MORTIS_NAME_RE.search(raw):
        return "dismiss" if (active and _MORTIS_DISMISS_STRONG_RE.search(raw)) else None
    if _MORTIS_DISMISS_RE.search(raw):
        return "dismiss"
    if _MORTIS_SUMMON_NEAR_RE.search(raw):
        return "summon"
    if addressed and len(_MORTIS_CLEAN_RE.sub("", raw)) <= 8:
        return "summon"
    return None


def pick_persona(*, mortis_active: bool, addressed: bool, mention_mutsumi: bool,
                 rate: float, roll: float) -> bool:
    """选人格：返回 True=墨提斯。

    - mortis_active（召唤窗口内）→ 必定墨提斯（召唤期间她代班，不受随机/提到名字影响）；
    - 否则：被 @/被引用时按 rate 概率换墨提斯；显式提到"若叶睦"的名字 → 一律若叶睦。
    """
    if mortis_active:
        return True
    return bool(addressed and not mention_mutsumi and roll < rate)


# 「没被叫到 / 被提到 / 聊到她相关的事」时的判定块（2026-10-15~16）：
# 用户要求①"尽量少主动接茬，除非符合人设、前后逻辑清晰才接"；②"对话中提到若叶睦及相关的
# 也应该会回复"。落地为三档（都插在当前消息前，离数据最近）：
#   mention   —— 群里出现她的名字/别称：要接（简短、贴人设、接住具体内容），真接不上才【沉默】
#   related   —— 群里在聊她身边的人/原作（祥子、爽世、Ave Mujica…）：可以接一句印象，接不上就沉默
#   proactive —— 没人叫她（只有 REPLY_RATE 命中才会走到）：接得上才接，别刷存在感
# 被 @/被引用/双bot对话（addressed="at"）不加任何块，行为与以前完全一致。
_UNADDRESSED_BLOCKS = {
    "mention": (
        "【有人提到你】群里的消息提到了你的名字/你的称呼——这是在跟你搭话，你要接。"
        "用你的方式回一句：短、直接，接住对方话里具体的东西（谁提到了你、说了什么、想让你做什么）。"
        "不要用 嗯/不知道/都行 这类万能句敷衍。"
        "只有当你完全接不上、或者开口就会 OOC（说到你不该知道的设定）时，才输出【沉默】。"
    ),
    "related": (
        "【聊到你身边的事】群友在聊你身边的人或与你有关的事（乐队、伙伴、原作剧情…）。"
        "你可以用一句符合你人设的话接住它：说出你的印象或态度，短、直白、别热情；"
        "接不上、没想法、或者开口会 OOC / 要编造时，只输出【沉默】。"
    ),
    "proactive": (
        "【没人叫你】这条消息没 @ 你、没引用你、也没在叫你。"
        "只有当你确实有一句符合你人设、和上下文逻辑接得上、有实质内容的话时，才接一句；"
        "接不上、没内容、只是想刷存在感 → 只输出【沉默】。宁可沉默，不要硬接。"
    ),
}


def shorten_reply(text: str, limit: int) -> str:
    """Keep a reply within ``limit`` characters, preferring punctuation breaks."""
    text = text or ""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    cut = text[: limit - 1]
    for i, ch in enumerate(cut, start=1):
        if ch in "。！？…，、" and i >= 3:
            cut = text[:i]
            break
    return cut + "…"


def split_short_lines(text: str, line_limit: int, max_lines: int = 3) -> list[str]:
    """Split model output into short messages and cap each line and count."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) <= 1 and lines:
        lines = [part.strip() for part in re.split(r"(?<=[。！？])", lines[0]) if part.strip()]
    return [shorten_reply(line, line_limit) for line in lines][:max_lines]


def should_trigger(
    *,
    enabled: bool,
    is_group: bool,
    user_id: str | int,
    self_id: str | int,
    other_bot_qq: str | int,
    astra_bot_qq: str | int,
    group_id: str | int,
    allowed_groups: set[int] | frozenset[int] | None,
    dialogue_active: bool = False,
    dialogue_turns: int = 0,
    dialogue_max_turns: int = 10,
) -> tuple[bool, bool]:
    """Return ``(allowed, consumed_dialogue_turn)`` for a group message.

    Keeping the dialogue-turn side effect explicit lets callers update their
    counter only after this function says a turn was consumed.
    """
    if not enabled or not is_group or str(user_id) == str(self_id):
        return False, False
    if str(user_id) == str(astra_bot_qq):
        return False, False
    if str(user_id) == str(other_bot_qq):
        ok = dialogue_active and dialogue_turns < dialogue_max_turns
        return ok, ok
    if allowed_groups and int(group_id) not in allowed_groups:
        return False, False
    return True, False


def sibling_history_key(hist_key: str) -> str | None:
    """同会话另一人格的历史键；用于把每一轮镜像过去，防随机换人格后断片。

    实锤（2026-10-20 00:19→00:21）：墨提斯回了「黄瓜」那句，历史写进 mortis 键；
    随机换回若叶睦后，她的历史里没有那条提问，只能猜着回「……不去。你说你的。」
    """
    for suffix, other in ((":mutsumi", ":mortis"), (":mortis", ":mutsumi")):
        if hist_key.endswith(suffix):
            return hist_key[: -len(suffix)] + other
    return None


def format_recent_line(item) -> str:
    """recent 行 -> 展示文本。兼容 2 元组 (who, text) 与 3 元组 (who, text, ts)：
    带时间戳时加 "[HH:MM] " 前缀，让她能分辨"刚刚"和"十分钟前"（2026-10-16）。
    函数内 import time，避免动模块顶部导入块。"""
    import time as _t
    try:
        if len(item) >= 3 and isinstance(item[2], (int, float)) and item[2] > 0:
            return f"[{_t.strftime('%H:%M', _t.localtime(float(item[2])))}] {item[0]}：{item[1]}"
        return f"{item[0]}：{item[1]}"
    except Exception:
        return str(item)


def build_context_messages(
    *,
    base_prompt: str,
    user: str,
    current_entry: str,
    summary: str = "",
    recent: Sequence[tuple[str, str]] = (),
    history: Sequence[Mapping[str, str]] = (),
    context_lines: int = 8,
    history_rounds: int = 8,
    experience_hint: str = "",
    sticker_hint: str = "",
    meme_hint: str = "",
    addressed: str = "at",
) -> list[dict[str, str]]:
    """Build ordered system/context/history messages for one reply.

    The current user entry is always the final message and is marked clearly;
    this prevents an older message from becoming the accidental answer target.

    ``addressed`` 说明这条消息跟她的关系（决定插不插嘴，2026-10-15 用户要求"少主动接茬"；
    2026-10-16 补充"提到若叶睦及相关内容要接"）：
      - ``at``：被 @ / 被引用 / 双bot对话 —— 在跟她说话，行为不变（不加块）；
      - ``mention``：群里出现她的名字 —— 是在跟她搭话，要接（接不上才沉默）；
      - ``related``：群里在聊她相关的人/原作 —— 可以接一句印象；
      - ``proactive``：没叫她任何形式（主动搭话概率命中才有）—— 接得上才接。
    判定块放在当前消息之前（离数据最近），实测能显著改变插嘴行为
    （见 E:\\bot\\_test_quiet_variants.py / _test_quiet_policy.py）。
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": base_prompt}]
    messages.append({
        "role": "system",
        "content": (
            f"【本轮对话对象】正在说话的是「{user}」。先判断 TA 这句话是在提问、求助、玩梗、"
            "表达情绪还是延续旧话题，再直接接住其中最具体的信息。不要把群聊背景里的别人当成 TA，"
            "不要为了展示人设硬扯旧梗。"
        ),
    })
    if summary:
        messages.append({"role": "system", "content": f"【群聊背景】{summary}（背景仅供参考，不要复述）"})
    if context_lines > 0 and recent:
        # The caller records current_entry in recent; exclude that final item.
        prior = list(recent[:-1])[-context_lines:]
        if prior:
            lines = [format_recent_line(x) for x in prior]
            messages.append({
                "role": "system",
                "content": "【群里最近的对话（只用来理解背景，不要复述）】\n" + "\n".join(lines),
            })
    if experience_hint:
        messages.append({"role": "system", "content": experience_hint})
    if sticker_hint:
        messages.append({"role": "system", "content": sticker_hint})
    messages.extend(dict(item) for item in list(history)[-history_rounds:])
    if addressed in _UNADDRESSED_BLOCKS:
        messages.append({"role": "system", "content": _UNADDRESSED_BLOCKS[addressed]})
    # 梗解释放在最贴近当前消息的位置（2026-10-16）：先看懂梗再决定接不接，
    # 接不上/不符合人设就沉默；这条只给她看，绝不能解释给群友听。
    if meme_hint:
        messages.append({"role": "system", "content": meme_hint})
    messages.append({"role": "user", "content": "【正在回这条】" + current_entry})
    return messages

