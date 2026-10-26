"""表情包系统：收集群友发的表情，让她在合适的时候发出去。

设计要点：
- **只在白名单群收集**，且只收真表情（sub_type=1 动画表情 / 7 自定义表情）——
  不收普通照片（sub_type=0），那是群友拍的图，不是表情包。
- 存本地：stickers/<hash>.<ext> + stickers.json（元数据：谁发的、什么时候、她写的备注）。
- 给她三个工具：
    list_stickers   看自己的表情库（带编号和备注，让她挑）
    send_sticker    发一个（按编号）
    sticker_note    用完后给这个表情写/改备注（她会自己学"这个表情适合什么场合"）
- 发送用 NapCat 的本地文件能力：enableLocalFile2Url=false 时用 file:// 绝对路径发本地文件。
"""
import hashlib
import html
import json
import os
import re
import time
from typing import Dict, List, Optional

import httpx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STICKER_DIR = os.path.join(BASE_DIR, "stickers")
STICKER_JSON = os.path.join(BASE_DIR, "stickers.json")

# 只收集这两类（真表情），不收普通照片
STICKER_SUBTYPES = {1, 7}

# QQ 防盗链占位图（"此图片来自QQ空间 未经允许不可引用"，3691B PNG）的 md5：
# 带 rkey 的直链经防盗链会返回它（HTTP 200 假成功），2026-10-02 实锤，必须识别并丢弃
PLACEHOLDER_MD5 = "5bb9bec07e71ef10a1a47521d396811d"


def legacy_url(file_name: str) -> Optional[str]:
    """图片文件名（32位hex+.ext）-> 免 rkey 的 gchat 原图直链。
    2026-10-02 实测：gchatpic_new 旧接口无需 rkey，直接返回真图且 md5 与文件名一致。"""
    m = re.fullmatch(r"([0-9A-Fa-f]{32})\.\w+", (file_name or "").strip())
    if not m:
        return None
    return f"https://gchat.qpic.cn/gchatpic_new/0/0-0-{m.group(1).upper()}/0"


def is_placeholder(content: bytes) -> bool:
    """判断下载内容是否为 QQ 防盗链占位图（假图）。"""
    if not content or len(content) < 100:
        return True
    return hashlib.md5(content).hexdigest() == PLACEHOLDER_MD5


_cache: Optional[dict] = None


def _load() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(STICKER_JSON, encoding="utf-8") as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    return _cache


def _save() -> None:
    try:
        os.makedirs(os.path.dirname(STICKER_JSON), exist_ok=True)
        with open(STICKER_JSON, "w", encoding="utf-8") as f:
            json.dump(_load(), f, ensure_ascii=False)
    except Exception as e:
        print(f"[STICKER] 保存失败: {e}", flush=True)


def _parse_cq_images(raw_msg: str) -> List[dict]:
    """解析 raw_message 里的所有 image 段，返回 [{file, url, sub_type, summary}, ...]。

    2026-09-29 修复：本机 NapCat raw_message 是 CQ 格式 [CQ:image,file=xx,sub_type=1,url=...]，
    旧解析器只认 nonebot 渲染的 [image:...] 格式，所以表情收集从未命中过（库一直为空的根因）。
    """
    if not raw_msg:
        return []
    out = []
    for m in re.finditer(r"\[(?:CQ:)?image[,:][^\]]*\]", raw_msg):
        seg = m.group(0)

        def _get(key: str) -> str:
            mm = re.search(r"[,:]" + re.escape(key) + r"=([^,\]]*)", seg)
            return mm.group(1) if mm else ""

        out.append({
            "file": _get("file"),
            "url": _get("url"),
            "sub_type": _get("sub_type") or _get("subType"),
            "summary": _get("summary"),
        })
    # CQ 串里 & 是 &amp; 转义的，下载前要还原
    for d in out:
        d["url"] = html.unescape(d["url"])
    return out


async def collect_from_message(raw_msg: str, group_id: str, uid: str, uname: str) -> int:
    """从一条群消息里收集表情。返回收集数量。"""
    imgs = _parse_cq_images(raw_msg)
    if not imgs:
        return 0
    added = 0
    for d in imgs:
        try:
            sub = int(d.get("sub_type", "0"))
        except Exception:
            sub = 0
        if sub not in STICKER_SUBTYPES:
            continue                              # 普通照片不收
        url = d.get("url") or ""
        url = html.unescape(d.get("url") or "")
        # 下载地址：优先用文件名 md5 走免 rkey 的 gchatpic_new 旧接口（实测唯一能给真图），
        # rkey 直链只做兜底。url 可能为空但 file 名有效——legacy 只依赖 file 名，别因 url 空跳过。
        dl_urls = []
        lg = legacy_url(d.get("file"))
        if lg:
            dl_urls.append(lg)
        if url.startswith("http") and url not in dl_urls:
            dl_urls.append(url)
        if not dl_urls:
            continue
        summary = (d.get("summary") or "").strip("[]").strip()
        key = d.get("file") or hashlib.md5(url.encode()).hexdigest()
        key = re.sub(r"[^0-9A-Za-z._-]", "", key)[:64]
        if not key:
            continue
        data = _load()
        if key in data:
            data[key]["uses_seen"] = (data[key].get("uses_seen") or 0) + 1
            _save()
            continue
        ext = os.path.splitext(key)[1] or ".png"
        if ext.lower() not in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"):
            ext = ".png"
        fname = f"{key}"
        path = os.path.join(STICKER_DIR, fname)
        try:
            os.makedirs(STICKER_DIR, exist_ok=True)
            content = None
            for dl in dl_urls:
                try:
                    with httpx.Client(timeout=15.0, follow_redirects=True, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
                        "Referer": "https://gchat.qpic.cn/",
                    }) as c:
                        r = c.get(dl)
                        r.raise_for_status()
                        cand = r.content
                    if is_placeholder(cand):
                        print(f"[STICKER] 占位图(防盗链)，跳过 {key} via {dl[:60]}", flush=True)
                        continue
                    content = cand
                    break
                except Exception as e:
                    print(f"[STICKER] 下载失败 {key}: {type(e).__name__}: {e}", flush=True)
            if content is None:
                continue
            if len(content) > 4 * 1024 * 1024:
                continue                          # 太大不收
            with open(path, "wb") as f:
                f.write(content)
        except Exception as e:
            print(f"[STICKER] 下载异常 {key}: {type(e).__name__}: {e}", flush=True)
            continue
        data[key] = {
            "file": os.path.basename(path),
            "path": path,
            "sub_type": sub,
            "text": summary[:20],                 # 表情上的文字（"？""nb"等）
            "from": uname,
            "from_uid": uid,
            "group": group_id,
            "at": time.time(),
            "uses_seen": 1,
            "note": "",                           # 她自己写的备注
            "sent": 0,                            # 她发过几次
        }
        added += 1
    if added:
        _save()
        print(f"[STICKER] 新收集 {added} 个表情（群{group_id}）", flush=True)
    return added


def resolve_path(ref: str) -> Optional[str]:
    """把编号/文件名/备注关键词解析成本地文件路径。找不到返回 None。"""
    ref = (ref or "").strip()
    if not ref:
        return None
    data = _load()
    if not data:
        return None
    items = list(data.items())
    items.sort(key=lambda kv: (bool(kv[1].get("note")), kv[1].get("uses_seen") or 0), reverse=True)
    # 1) 纯数字 = 列表里的序号
    if ref.isdigit():
        idx = int(ref) - 1
        if 0 <= idx < len(items):
            return items[idx][1].get("path")
    # 2) 直接就是 key
    if ref in data:
        return data[ref].get("path")
    # 3) 文件名匹配
    for k, v in data.items():
        if os.path.basename(v.get("file", "")) == ref or v.get("file") == ref:
            return v.get("path")
    # 4) 备注/表情上的文字模糊匹配
    for k, v in data.items():
        note = v.get("note") or ""
        if ref and (ref in note or ref == v.get("text")):
            return v.get("path")
    return None


def sticker_count() -> int:
    return len(_load())


def pick_for_prompt(note_hint: str = "") -> str:
    """挑几个表情塞进她的上下文，让她知道手里有什么牌，并给出发送格式。
    发送用简单标记 `[表情:编号]`——由 chat.py 检测后替换成真正的图片消息。"""
    data = _load()
    if not data:
        return ""
    items = list(data.items())
    # 有价值的在前：有备注的 > 见过的次数多的
    items.sort(key=lambda kv: (bool(kv[1].get("note")), kv[1].get("uses_seen") or 0), reverse=True)
    lines = []
    for key, it in items[:10]:
        desc = it.get("note") or ""
        if not desc and it.get("text"):
            desc = f"图上写着「{it['text']}」"
        if not desc:
            desc = "（没备注，看图猜的）"
        lines.append(f"- 编号 {key}：{desc}")
    return (
        "【你的表情包】群友发的表情你收着了，这些可以用：\n" + "\n".join(lines) +
        "\n\n想发的时候，在回复里写上 [表情:编号] —— 它会变成真的表情发出去。\n"
        "什么时候发：群友发了表情/搞笑图、气氛轻松、你想表达好笑/无语/赞同但懒得打字时。\n"
        "表情可以单独发（整个回复就只有 [表情:编号]），也可以跟在话后面。\n"
        "别人在聊正经事的时候别发。"
    )


# ============ 给 agent 的工具实现 ============
def tool_list_stickers(group_id: str, uid: str, ctx: dict) -> str:
    """列出她的表情库（带编号/备注）。"""
    data = _load()
    if not data:
        return "（你的表情库还是空的，群友发动画表情时会自动收进来）"
    items = list(data.items())
    items.sort(key=lambda kv: (bool(kv[1].get("note")), kv[1].get("uses_seen") or 0), reverse=True)
    lines = []
    for i, (k, v) in enumerate(items[:20], 1):
        desc = v.get("note") or (f"图上写着「{v['text']}」" if v.get("text") else "（没备注）")
        lines.append(f"{i}. {desc}（编号 {k}）")
    return "表情库：\n" + "\n".join(lines)


def tool_send_sticker(group_id: str, uid: str, ctx: dict, content: str = "") -> str:
    """她决定发某个表情。真正的发送由 chat.py 在拿到 pending_sticker 后执行。"""
    key = (content or "").strip().splitlines()[0].strip()
    data = _load()
    if not data:
        return "（表情库是空的）"
    # 支持按编号（序号）或直接给 key
    if key.isdigit():
        items = list(data.items())
        items.sort(key=lambda kv: (bool(kv[1].get("note")), kv[1].get("uses_seen") or 0), reverse=True)
        idx = int(key) - 1
        if 0 <= idx < len(items):
            key = items[idx][0]
    it = data.get(key)
    if not it:
        # 模糊匹配：备注/文字里包含
        for k2, v2 in data.items():
            if key and (key in (v2.get("note") or "") or key == v2.get("text")):
                it, key = v2, k2
                break
    if not it:
        return f"（表情库里没找到「{key}」，用 list_stickers 看看有哪些）"
    ctx["pending_sticker"] = it["path"]
    ctx["sticker_key"] = key
    ctx["sticker_dirty"] = True
    it["sent"] = (it.get("sent") or 0) + 1
    _save()
    return f"（发出表情：{it.get('note') or it.get('file')}）"


def tool_sticker_note(group_id: str, uid: str, ctx: dict, content: str = "") -> str:
    """她给最近发的表情写备注，学会"这个表情适合什么场合"。"""
    parts = (content or "").split("|", 1)
    if len(parts) == 2:
        key, note = parts[0].strip(), parts[1].strip()
    else:
        key, note = "", (content or "").strip()
    data = _load()
    if not data:
        return "（表情库是空的）"
    if not key:
        key = ctx.get("sticker_key") or ""
    if key not in data:
        return "（不知道你说的是哪个表情）"
    data[key]["note"] = note[:40]
    _save()
    return f"（记住了：这个表情 {note[:30]}）"


STICKER_TOOLS = {
    "list_stickers": tool_list_stickers,
    "send_sticker": tool_send_sticker,
    "sticker_note": tool_sticker_note,
}
STICKER_TOOLS_ARG = {"send_sticker", "sticker_note"}
