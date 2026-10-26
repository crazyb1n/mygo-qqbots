# -*- coding: utf-8 -*-
"""Per-user identity memory for the Anon chat plugin (pure stdlib, offline-testable).

2026-10-26 需求实锤：爱音“不认识人”——每用户对话历史只在内存、重启即清零，
anon_memory.json 只落群摘要，没有 QQ ↔ 群名片 ↔ 互动历史 的持久档案；
她还会把一次玩笑外号（如“冒牌喵梦”）固化成称呼反复用。

本模块只做纯逻辑，不 import NoneBot/OpenAI：

* 档案结构：``users[uid] = {names, first, last, count, last_msg}``
* 被动更新：每条白名单群/私聊消息只记名字、时间、条数和最后一句（零 LLM 成本）
* 渲染：给提示词用的【当前用户】块（QQ 尾号锚点 + 认识时间 + 上次说过什么）
* 边界：用户数/名字数/字段长度都有上限；群友可控文本压平控制符，防伪造提示词块
"""

from __future__ import annotations

import re
import time
from typing import Optional

MAX_USERS = 500
MAX_NAMES = 6
NAME_CHARS = 40
LAST_MSG_CHARS = 80
STAMP_CHARS = 16

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_SPACE_RE = re.compile(r"\s+")
# 零宽/双向控制符（常被用来隐藏伪标记）：写进提示词前一律删除。
_INVISIBLE_RE = re.compile(
    r"[\u00ad\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]"
)
# 纯占位文本（图片/语音/视频/表情及其组合）不写进“上次说过什么”。
_PLACEHOLDER_RE = re.compile(r"^(?:\[(?:图片|语音|视频|表情)\]\s*)+$")


def _clean(text, limit: int) -> str:
    value = _INVISIBLE_RE.sub("", str(text or ""))
    value = _CONTROL_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    return value[:limit]


def _as_int(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _stamp(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts if ts is not None else time.time()))


def uid_tail(uid) -> str:
    """QQ 尾号锚点：群友改群名片后，靠它把她眼里的“同一个人”串起来。"""
    value = str(uid or "").strip()
    return value[-4:] if len(value) >= 4 else value


def load_users(payload) -> dict:
    """从 MEMORY_FILE 的完整 JSON 里取出并规范化 ``users``；坏数据逐条丢弃。"""
    users = payload.get("users") if isinstance(payload, dict) else None
    if not isinstance(users, dict):
        return {}
    cleaned: dict = {}
    for uid, profile in users.items():
        uid = str(uid or "").strip()
        if not uid or not isinstance(profile, dict):
            continue
        names: dict = {}
        raw_names = profile.get("names")
        if isinstance(raw_names, dict):
            for name, info in raw_names.items():
                name = _clean(name, NAME_CHARS)
                if not name or not isinstance(info, dict):
                    continue
                names[name] = {
                    "first": _clean(info.get("first", ""), STAMP_CHARS),
                    "last": _clean(info.get("last", ""), STAMP_CHARS),
                    "count": _as_int(info.get("count")),
                }
        last_msg = _clean(profile.get("last_msg", ""), LAST_MSG_CHARS)
        if _PLACEHOLDER_RE.match(last_msg):
            last_msg = ""
        cleaned[uid] = {
            "names": names,
            "first": _clean(profile.get("first", ""), STAMP_CHARS),
            "last": _clean(profile.get("last", ""), STAMP_CHARS),
            "count": _as_int(profile.get("count")),
            "last_msg": last_msg,
        }
    # 落盘数据可能被手工改大：加载时也执行上限（每人名字数 + 总用户数）。
    for profile in cleaned.values():
        names = profile.get("names") or {}
        if len(names) > MAX_NAMES:
            latest = sorted(
                names.items(), key=lambda kv: str(kv[1].get("last", "")), reverse=True
            )[:MAX_NAMES]
            profile["names"] = dict(latest)
    prune(cleaned)
    return cleaned


def update(
    users: dict, uid: str, name: str, text: str, ts: Optional[float] = None
) -> str:
    """被动更新一条档案；返回 "new_user" / "new_name" / ""（调用方只用于日志）。"""
    uid = str(uid or "").strip()
    if not uid:
        return ""
    name = _clean(name or "", NAME_CHARS) or uid[-6:]
    stamp = _stamp(ts)
    status = ""
    profile = users.get(uid)
    if not isinstance(profile, dict):
        profile = {
            "names": {},
            "first": stamp,
            "last": stamp,
            "count": 0,
            "last_msg": "",
        }
        users[uid] = profile
        status = "new_user"
    names = profile.get("names")
    if not isinstance(names, dict):
        names = {}
        profile["names"] = names
    entry = names.get(name)
    if not isinstance(entry, dict):
        names[name] = {"first": stamp, "last": stamp, "count": 1}
        status = status or "new_name"
        _trim_names(names)
    else:
        entry["last"] = stamp
        entry["count"] = _as_int(entry.get("count")) + 1
    if not profile.get("first"):
        profile["first"] = stamp
    profile["last"] = stamp
    profile["count"] = _as_int(profile.get("count")) + 1
    cleaned_text = _clean(text, LAST_MSG_CHARS)
    if cleaned_text and not _PLACEHOLDER_RE.match(cleaned_text):
        profile["last_msg"] = cleaned_text
    prune(users, protect=uid)
    return status


def _trim_names(names: dict) -> None:
    if len(names) <= MAX_NAMES:
        return
    oldest = min(names, key=lambda key: str(names[key].get("last", "")))
    names.pop(oldest, None)


def prune(users: dict, max_users: int = MAX_USERS, protect: Optional[str] = None) -> None:
    """超过上限时淘汰最久没出现的人（同 last 时按 uid 稳定排序，便于测试）。

    ``protect`` 保护当前正在更新的 uid：满库瞬间的新用户不能被自己触发的淘汰
    立刻踢掉（否则调用方随后取 ``user_profiles[uid]`` 会 KeyError）。
    """
    if len(users) <= max_users:
        return
    ordered = [
        kv
        for kv in sorted(
            users.items(), key=lambda kv: (str(kv[1].get("last", "")), kv[0])
        )
        if kv[0] != protect
    ]
    for uid, _ in ordered[: max(0, len(users) - max_users)]:
        users.pop(uid, None)


def render(profile: Optional[dict], uid: str, current_name: str = "") -> str:
    """把档案渲染成提示词块；无档案返回 ""。群友可控文本全部压平截断。"""
    if not isinstance(profile, dict) or not profile:
        return ""
    name = _clean(current_name or "", NAME_CHARS)
    if not name:
        names = profile.get("names")
        if isinstance(names, dict) and names:
            name = max(
                names,
                key=lambda key: str(names[key].get("last", ""))
                if isinstance(names.get(key), dict)
                else "",
            )
    name = name or "群友"
    count = _as_int(profile.get("count"))
    first = _clean(profile.get("first", ""), STAMP_CHARS)
    last = _clean(profile.get("last", ""), STAMP_CHARS)
    last_msg = _clean(profile.get("last_msg", ""), LAST_MSG_CHARS)

    facts = []
    if count:
        facts.append(f"已见过 {count} 条消息")
    if first:
        facts.append(f"最早 {first}")
    if last:
        facts.append(f"上次互动 {last}")
    head = f"【当前用户】{name}（QQ …{uid_tail(uid)}）"
    if facts:
        head += "：" + "，".join(facts)
    body = head + "。"
    if last_msg:
        body += f"TA 上次说「{last_msg}」。"
    body += (
        "这是你认识的群友，别装作初次见面；称呼用 TA 当前的群名片名字。"
        "档案只用来认出对方，不要整段复述；被问起时可以自然地说出你记得的事。"
    )
    return body
