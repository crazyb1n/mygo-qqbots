"""多层经验库：按「抽象→一般→具体→执行」四层组织，主AI按需检索注入，并能自我沉淀。

对应架构文档《若叶睦-agent架构-逻辑文档.md》3.4 节。

- L0 抽象层：人格法则 = chat.py 的 SYSTEM_PROMPT_MUTSUMI（恒注入，不在这里）
- L1 一般层 scenarios.json：场景应对经验（被夸/被怼/求助/玩梗…），按 tags 检索
- L2 具体层 group.json：本群情报（群梗词典/成员称呼/约定/雷区），按关键词检索
- L3 执行层 tasks.json：任务经验（这类题怎么解、上次同类任务怎么做）

条目格式：{"text": ≤120字, "tags": [...], "about": "可选：相关人QQ号", "at": ts, "hits": n}
检索：关键词（tags/正文包含）+ 场景标签打分，每层 top3，总注入预算控制在 ~150 字。
沉淀：chat.py 的"记住"指令 / agent_core 任务后 distill；去重（新条目包含旧条目正文即视为重复）。
所有文件人工可编辑（改完不用重启？——不，缓存常驻，改完需要重启 bot）。
"""
import json
import os
import re
import time
from typing import Dict, List, Optional

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experience")
FILES = {
    "L1": os.path.join(BASE_DIR, "scenarios.json"),
    "L2": os.path.join(BASE_DIR, "group.json"),
    "L3": os.path.join(BASE_DIR, "tasks.json"),
}
# 各层容量上限（超过按 at+hits 淘汰最旧的）
CAPS = {"L1": 40, "L2": 80, "L3": 30}
# 检索返回条数
TOP_K = 3

_cache: Dict[str, Optional[list]] = {"L1": None, "L2": None, "L3": None}


def _load(layer: str) -> list:
    if _cache[layer] is None:
        try:
            with open(FILES[layer], encoding="utf-8") as f:
                _cache[layer] = json.load(f)
        except Exception:
            _cache[layer] = []
    return _cache[layer]


def _save(layer: str) -> None:
    try:
        os.makedirs(BASE_DIR, exist_ok=True)
        with open(FILES[layer], "w", encoding="utf-8") as f:
            json.dump(_load(layer), f, ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"[EXP] 保存{layer}失败: {e}", flush=True)


def _entry(text: str, tags=None, about: str = "") -> dict:
    return {"text": text.strip()[:120], "tags": list(tags or []), "about": about,
            "at": time.time(), "hits": 0}


SEED_L2 = [
    # —— 群梗词典（来源：chat.py 固定接梗规则沉淀，2026-09-28 播种）——
    {"text": "扫货/薇欧拉是另一只机器人（现名林晚，QQ200000002），群友调侃她'彪子'，也可以说是好女孩", "tags": ["扫货", "薇欧拉", "林晚", "机器人"], "about": ""},
    {"text": "dsh/小鲸鱼是群里另一只AI机器人（女仆鲸鱼娘，QQ200000004），不是真人，是群友的机器人同行", "tags": ["dsh", "小鲸鱼", "鲸鱼", "机器人"], "about": ""},
    {"text": "黄瓜是我的象征：我种黄瓜、送黄瓜、黄瓜农场游戏是群里的日常玩法（签到/投喂/好感度）", "tags": ["黄瓜", "农场", "签到", "投喂"], "about": ""},
]
SEED_L1 = [
    {"text": "有人连续问怪问题逗我时，不用每个都接，挑一个回，其余沉默，面无表情反而好笑", "tags": ["被逗", "怪问题", "玩梗"], "about": ""},
    {"text": "群友求助时如果帮不上，直说'不会'或'不知道'，但可以补一句谁来帮帮他之类的人话（很短）", "tags": ["求助"], "about": ""},
]
SEED_L3 = []


def _seed() -> None:
    """首启动播种：文件不存在才写入种子（不覆盖人工编辑）。"""
    os.makedirs(BASE_DIR, exist_ok=True)
    pairs = [("L1", SEED_L1), ("L2", SEED_L2), ("L3", SEED_L3)]
    for layer, seed in pairs:
        if not os.path.exists(FILES[layer]):
            try:
                with open(FILES[layer], "w", encoding="utf-8") as f:
                    json.dump(seed, f, ensure_ascii=False, indent=1)
                _cache[layer] = None
                print(f"[EXP] {layer} 播种 {len(seed)} 条", flush=True)
            except Exception as e:
                print(f"[EXP] 播种{layer}失败: {e}", flush=True)


_seed()


# ---------- 沉淀（写入） ----------

def _normalize(t: str) -> str:
    return re.sub(r"[\s，。！？、…,.!?~～]+", "", (t or "").strip())


def learn(layer: str, text: str, tags=None, about: str = "") -> bool:
    """记一条经验。去重：清洗后互相包含视为重复。超限淘汰最旧/最少用的。"""
    text = (text or "").strip()
    if not text or layer not in FILES:
        return False
    items = _load(layer)
    norm = _normalize(text)
    if not norm:
        return False
    for it in items:
        old = _normalize(it.get("text", ""))
        if not old:
            continue
        if norm in old or old in norm:
            it["hits"] = it.get("hits", 0) + 1   # 复述也算一次强化
            _save(layer)
            return False
    items.append(_entry(text, tags, about))
    cap = CAPS[layer]
    if len(items) > cap:
        items.sort(key=lambda it: it.get("at", 0) + it.get("hits", 0) * 86400, reverse=True)
        items[:] = items[:cap]
    _save(layer)
    print(f"[EXP] {layer} 新增经验: {text[:40]!r}", flush=True)
    return True


# ---------- 检索（注入） ----------

def _match_score(item: dict, text_norm: str, words: List[str]) -> int:
    score = 0
    hay = _normalize(item.get("text", ""))
    for t in item.get("tags", []):
        tn = _normalize(t)
        if tn and tn in text_norm:
            score += 3
    for w in words:
        if w and (w in hay or w in _normalize(" ".join(item.get("tags", [])))):
            score += 2
    if item.get("about") and item["about"] in text_norm:
        score += 4
    return score


def retrieve(text: str, layers=("L2", "L1", "L3")) -> List[str]:
    """按需检索：返回命中的经验条目文本（去重、按层顺序：具体→一般→执行）。
    没命中返回空列表——宁可不注入，也不塞无关记忆。"""
    text_norm = _normalize(text)
    if not text_norm:
        return []
    # 抽取关键词：2字以上的中文词片段 + 现有 tag/正文词的命中
    words = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
    out: List[str] = []
    for layer in layers:
        items = _load(layer)
        scored = []
        for it in items:
            s = _match_score(it, text_norm, words)
            if s > 0:
                scored.append((s, it))
        scored.sort(key=lambda x: (x[0], x[1].get("hits", 0)), reverse=True)
        for s, it in scored[:TOP_K]:
            line = it["text"]
            if line not in out:
                out.append(line)
                it["hits"] = it.get("hits", 0) + 1
        if scored:
            _save(layer)
        if len(out) >= TOP_K * 2:
            break
    return out[:TOP_K * 2]


def render_hint(text: str) -> str:
    """把检索结果渲染成可注入 system/user 的提示块。无命中返回空串。"""
    lines = retrieve(text)
    if not lines:
        return ""
    return "【你记得的关于这个群的事（贴合就自然用上，不合贴就忽略）】\n" + "\n".join(f"- {l}" for l in lines)


def note_for(uid: str) -> str:
    """某个人的专属记忆（about=QQ号）。没有返回空串。"""
    if not uid:
        return ""
    lines = [it["text"] for it in _load("L2") if it.get("about") == str(uid)]
    return "\n".join(lines)


def counts() -> dict:
    return {k: len(_load(k)) for k in FILES}
