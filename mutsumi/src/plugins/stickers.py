"""表情包系统：收集群友发的表情，若叶睦在合适的时候也能甩一个出去。

从扫货（QQBot-Viola）移植，去掉 agent 工具部分，保留：
- 收集：只在白名单群收真表情（sub_type=1 动画表情 / 7 自定义表情），
  不收普通照片（sub_type=0，那是群友拍/存的图）。
- 存储：stickers/<key>.<ext> + stickers.json（谁发的、图上的字、次数）。
- 理解（2026-10-07 加）：每张表情用视觉模型写"语义备注"（画面/情绪/适合场景）
  存进 note —— 不认识的脸不发，看不懂的梗不发。
- 供给 LLM：pick_for_prompt() 把带含义的表情清单塞进上下文，她回复里写 [表情:编号]，
  由 chat.py 拆出来变成真正的图片消息；被动跟风（别人发表情她也回一个）走
  pick_matching()：先识图看懂对方发的什么，再从库里挑一个语义上接得上的，
  挑不出合适的就宁可不发。
"""
import asyncio
import hashlib
import html
import json
import os
import re
import time
import urllib.request
from typing import List, Optional

import httpx

try:
    from . import vision as vision_mod
except ImportError:          # 脱离包结构直接跑脚本（离线补标注等）时
    import vision as vision_mod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STICKER_DIR = os.path.join(BASE_DIR, "stickers")
STICKER_JSON = os.path.join(BASE_DIR, "stickers.json")

# 只收集这两类（真表情），不收普通照片
STICKER_SUBTYPES = {1, 7}

# QQ 防盗链占位图（"此图片来自QQ空间 未经允许不可引用"，3691B PNG）的 md5：
# 带 rkey 的直链经防盗链会返回它（HTTP 200 假成功），2026-10-02 实锤，必须识别并丢弃
PLACEHOLDER_MD5 = "5bb9bec07e71ef10a1a47521d396811d"


def legacy_url(file_name: str) -> Optional[str]:
    """图片文件名（32位hex+.ext，QQ 的原始 md5 命名）-> 免 rkey 的 gchat 原图直链。
    2026-10-02 实测：gchatpic_new 旧接口无需 rkey/登录态，直接返回真图且 md5 与文件名一致。"""
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
    """从 raw_message 里解析所有 image 段，返回 [{file, url, sub_type, summary}, ...]。

    兼容两种格式（2026-09-27 实锤：本机 NapCat raw_message 是 CQ 格式，
    扫货旧解析器只认 [image:...] 格式所以从没收集到过——这就是表情库为空的原因）：
    - [CQ:image,file=xx,sub_type=1,url=https://gchat.qpic.cn/...]   ← 本机实测
    - [image:file=xx,summary=xx,sub_type=1,...]                      ← 旧格式兜底
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
    return out


def _norm_images(seg_datas: List[dict], raw_msg: str = "") -> List[dict]:
    """把 nonebot image 段的 data 规范化成统一字典。
    优先用段数据（信息最全，无截断风险）；段里都没有 sub_type 时回退解析 raw CQ 串。"""
    out = []
    for d in seg_datas or []:
        out.append({
            "file": str(d.get("file") or ""),
            "url": str(d.get("url") or ""),
            "sub_type": str(d.get("sub_type") or d.get("subType") or ""),
            "summary": str(d.get("summary") or ""),
        })
    if out and all(not x["sub_type"] for x in out):
        out = _parse_cq_images(raw_msg)
    return out


def has_sticker(seg_datas: List[dict], raw_msg: str = "") -> bool:
    """这批 image 段里是否带真表情（动画/自定义表情，非普通照片）。"""
    for d in _norm_images(seg_datas, raw_msg):
        try:
            if int(d.get("sub_type", "0")) in STICKER_SUBTYPES:
                return True
        except Exception:
            continue
    return False


async def collect_from_message(seg_datas: List[dict], raw_msg: str, group_id: str,
                               uid: str, uname: str) -> int:
    """收集一批 image 段里的表情。返回收集数量。"""
    imgs = _norm_images(seg_datas, raw_msg)
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
        url = html.unescape(url)              # CQ/JSON 里 & 可能是 &amp; 转义的
        # 下载地址：优先用文件名 md5 走免 rkey 的 gchatpic_new 旧接口（实测唯一能给真图），
        # rkey 直链只做兜底（经防盗链只会拿到"未经允许不可引用"占位图）。
        # 注意：url 可能为空但 file 名有效——legacy 链接只依赖 file 名，不要因 url 空而跳过。
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
                    # gchatpic_new 旧接口不需要特殊头；rkey 链兜底时带 gchat Referer
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
            "note": "",                           # 她自己攒的印象
            "sent": 0,                            # 她发过几次
        }
        added += 1
    if added:
        _save()
        print(f"[STICKER] 新收集 {added} 个表情（群{group_id}）", flush=True)
    return added


def _sorted_items() -> list:
    """有价值的在前：有备注的 > 见过次数多的。"""
    items = list(_load().items())
    items.sort(key=lambda kv: (bool(kv[1].get("note")), kv[1].get("uses_seen") or 0), reverse=True)
    return items


def sticker_count() -> int:
    return len(_load())


def random_sticker_path() -> Optional[str]:
    """[已废弃 2026-10-07] 随机挑表情 = 乱发的根源，已被 pick_matching() 取代。
    保留签名只为兼容未知的旧调用点；永远返回 None。"""
    return None


def resolve_path(ref: str) -> Optional[str]:
    """把编号/文件名/备注关键词解析成本地文件路径。找不到返回 None。"""
    ref = (ref or "").strip()
    if not ref:
        return None
    data = _load()
    if not data:
        return None
    items = _sorted_items()
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


def pick_for_prompt() -> str:
    """挑几个表情塞进她的上下文，让她知道手里有什么牌（带语义备注），并给出发送格式。
    发送用简单标记 `[表情:编号]`——由 chat.py 检测后替换成真正的图片消息。
    编号用短数字（与 resolve_path 的序号逻辑一致），省 token 且小模型不易抄错。"""
    data = _load()
    if not data:
        return ""
    items = _sorted_items()
    lines = []
    for i, (key, it) in enumerate(items[:10], 1):
        desc = it.get("note") or ""
        if not desc and it.get("text"):
            desc = f"图上写着「{it['text']}」"
        if not desc:
            desc = "（含义未知）"
        lines.append(f"- 编号 {i}：{desc}")
    return (
        "【你的表情包】群友发的表情你收着了，这些可以用（都标了含义）：\n" + "\n".join(lines) +
        "\n\n想发的时候，在回复里写上 [表情:编号]（如 [表情:3]）—— 它会变成真的表情发出去。\n"
        "什么时候发：某个表情的含义**正好接得住当下这句话**（想表达好笑/无语/赞同/吐槽而懒得打字）。\n"
        "表情可以单独发（整个回复就只有 [表情:编号]），也可以跟在话后面。\n"
        "别人在聊正经事的时候别发。\n"
        "拿不准哪个合适就一个都不发，宁可只回文字——不许为了发而发。"
    )


# ---------------- 语义标注（2026-10-07 加：理解每个表情的意思） ----------------
CAPTION_PROMPT = (
    "这是一张QQ表情包。给它写一条档案备注，帮机器人以后在合适的时机使用它。"
    "一行内输出，格式：画面X；表达Y；适合Z。X≤18字描述画面/角色动作（一眼认得出角色就写角色名，"
    "如\"若叶睦\"；认不出就跳过，不要推敲），Y≤10字写情绪或梗，"
    "Z≤14字写什么聊天情景下发它最合适。只输出这一行，不要客套。"
)


def caption_one(key: str) -> Optional[str]:
    """给单个表情生成语义备注，写进 note 字段。成功返回备注文本。
    （note 本来就是给她"自己攒的印象"留的，一直空着——现在用视觉模型真正攒上。）
    失败记 caption_fails 计数，连续失败 3 次的不再重试（防启动时永远空转）。"""
    entry = _load().get(key)
    if not entry:
        return None
    path = entry.get("path")
    if not path or not os.path.exists(path):
        return None
    try:
        if os.path.getsize(path) > 3 * 1024 * 1024:
            _load()[key]["caption_fails"] = 99    # 超出识图尺寸上限，标记为永不重试
            _save()
            return None
    except OSError:
        return None
    prompt = CAPTION_PROMPT
    hint = (entry.get("text") or "").strip()
    if hint:
        prompt += f"（QQ 里这个表情的文字摘要是「{hint[:20]}」，可作参考，不一定对。）"
    desc = None
    for _ in range(2):                            # 网关偶发抽风，重试一次
        desc = vision_mod.describe_file(path, prompt)
        if desc:
            break
        time.sleep(0.5)
    if not desc:
        data = _load()
        data[key]["caption_fails"] = (data[key].get("caption_fails") or 0) + 1
        _save()
        return None
    _load()[key]["note"] = desc[:120]
    _load()[key]["captioned_at"] = time.time()
    _save()
    return desc


def caption_missing(max_n: int = 0) -> int:
    """给库里还没有语义备注的表情批量补标注（离线跑也行）。返回补了几张。
    max_n>0 时限制本次最多补几张。单张失败就跳过（连续 3 次失败的不再试）。"""
    todo = [k for k, v in _load().items()
            if not (v.get("note") or "").strip() and (v.get("caption_fails") or 0) < 3]
    if max_n:
        todo = todo[:max_n]
    n = 0
    for k in todo:
        if caption_one(k):
            n += 1
        time.sleep(0.2)                     # 温一点，别打爆网关
    if n:
        print(f"[STICKER] 给 {n} 个表情补了语义备注，库共 {sticker_count()} 个", flush=True)
    return n


async def caption_missing_async() -> int:
    """caption_missing 的异步包装（后台任务用，不卡事件循环）。"""
    return await asyncio.to_thread(caption_missing)


# ---------------- 按情景挑表情（不再随机） ----------------
def find_local_path(seg_datas: List[dict], raw_msg: str = "") -> Optional[str]:
    """这批 image 段里第一个真表情对应的本地文件路径（收进库的才返回）。"""
    for d in _norm_images(seg_datas, raw_msg):
        try:
            if int(d.get("sub_type", "0")) not in STICKER_SUBTYPES:
                continue
        except Exception:
            continue
        key = re.sub(r"[^0-9A-Za-z._-]", "", d.get("file") or "")[:64]
        entry = _load().get(key)
        if entry:
            p = entry.get("path")
            if p and os.path.exists(p):
                return p
    return None


def first_summary(seg_datas: List[dict], raw_msg: str = "") -> str:
    """这批 image 段里第一个真表情的 QQ 文字摘要（如"？"/"nb"），没有则空串。
    识图失败时的最低限度"意思"，连摘要都没有就说明我们对它一无所知 → 不发。"""
    for d in _norm_images(seg_datas, raw_msg):
        try:
            if int(d.get("sub_type", "0")) not in STICKER_SUBTYPES:
                continue
        except Exception:
            continue
        s = (d.get("summary") or "").strip("[]").strip()
        if s:
            return s
    return ""


def _llm_text(prompt: str, max_tokens: int = 2048) -> Optional[str]:
    """同步小 LLM 调用（表情挑选用）。失败/空响应重试一次，仍失败返回 None，绝不抛异常。
    max_tokens 给足：思考模型连思考带回答经常 500-1500 token，预算不够正文会被挤空。
    与 daily_news 同一网关姿势：DEEPSEEK_* 配置 + reasoning_effort。"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    effort = os.getenv("DEEPSEEK_REASONING_EFFORT", "none").strip()
    if not api_key:
        return None

    def _post(payload: dict) -> Optional[str]:
        req = urllib.request.Request(
            f"{base}/chat/completions", data=json.dumps(payload).encode(),
            headers={"content-type": "application/json",
                     "Authorization": f"Bearer {api_key}",
                     # UA 不能换：默认 Python UA 会被网关的 Cloudflare 1010 拦掉
                     "User-Agent": "mutsumi-bot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read().decode())
        return (d["choices"][0]["message"].get("content") or "").strip()

    try:
        payload = {"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens}
        if effort:
            payload["reasoning_effort"] = effort
        text = None
        for _ in range(2):
            try:
                text = _post(payload)
            except Exception:
                text = None
            if text:
                break
            time.sleep(0.4)
        return text or None
    except Exception as e:
        print(f"[STICKER] LLM 调用失败 {type(e).__name__}: {e}", flush=True)
        return None


def pick_matching(situation: str) -> Optional[str]:
    """根据情景（对方表情的识图描述 / 聊天内容）从库里挑一个语义上接得上的表情路径。
    挑不出合适的返回 None —— 宁可不发，也不乱发。"""
    captioned = [(k, v) for k, v in _sorted_items() if (v.get("note") or "").strip()]
    if not captioned:
        return None
    captioned = captioned[:15]
    lines = [f"- 编号{i}：{(v.get('note') or '')[:80]}" for i, (k, v) in enumerate(captioned, 1)]
    prompt = (
        "你是表情包管理员，原则是宁缺毋滥。对方刚发了一个表情，识图结果/当前情景是：\n"
        f"{(situation or '')[:200]}\n\n"
        "下面是库里可选的表情（编号：含义备注）：\n" + "\n".join(lines) +
        "\n\n从中挑一个发出去最能接上气氛的。硬标准（必须满足至少一条）：\n"
        "1. 情绪对得上：对方无语你回无语系、对方搞笑你回搞笑系、对方破防你回嘲讽/安慰系；\n"
        "2. 场景对得上：备注里写的「适合」情景正好就是现在的情景。\n"
        "只是「都挺可爱」「都是表情包」这种弱关联，一律回答：无\n"
        "只输出一个编号数字，或「无」。"
    )
    raw = _llm_text(prompt)
    if not raw:
        return None
    m = re.search(r"\d+", raw)
    if not m:
        return None                       # 模型说"无"或胡说 → 都不发
    idx = int(m.group(0)) - 1
    if not (0 <= idx < len(captioned)):
        return None
    p = captioned[idx][1].get("path")
    return p if p and os.path.exists(p) else None


async def pick_matching_async(situation: str) -> Optional[str]:
    """pick_matching 的异步包装（聊天 handler 里用，不卡事件循环）。"""
    return await asyncio.to_thread(pick_matching, situation)
