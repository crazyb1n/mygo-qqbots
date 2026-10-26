"""若叶睦的黄瓜农场系统。

功能：
  - 每日签到得黄瓜
  - 玩小游戏赢黄瓜（猜大小、猜数字）
  - 黄瓜永久存仓库
  - 黄瓜排行榜

数据保存在 cucumber_data.json（永久持久化）——**按群独立记账**：
  黄瓜/签到/好感度/称号/游戏进度都以「群号:QQ号」为键，同一个群友在不同群
  各有一套数据，在一个群签到、投喂、押注不会影响他在别的群的仓库和排行榜。
  （2026-10-17 之前是不分群的全局数据，已在首次读取时自动迁移，见 _migrate_legacy。）
"""
import json
import os
import random
import time
from pathlib import Path

DATA_FILE = Path(__file__).parent / "cucumber_data.json"

DATA_VERSION = 2

# 老数据（2026-10-17 之前不分群）整体归属的群：这批数据都是在群 100000001 产生的，
# 需要改归属时在 .env 里设 CUCUMBER_LEGACY_GROUP=群号。
LEGACY_GROUP = (os.getenv("CUCUMBER_LEGACY_GROUP") or "100000001").strip()

# ---------- 数据存储 ----------


def _key(gid: str, uid: str) -> str:
    """按群记账的键。群号/QQ号都是纯数字，用冒号拼接无歧义。"""
    return f"{gid}:{uid}"


def _load() -> dict:
    try:
        if DATA_FILE.exists():
            with open(DATA_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and _needs_migration(data):
                _migrate_legacy(data)
            return data
    except Exception as e:
        print(f"[黄瓜] 读取失败: {e}", flush=True)
        # 文件损坏：先备份再返回空数据，避免下次保存直接覆盖导致数据永久丢失
        try:
            backup = DATA_FILE.with_suffix(".json.bak")
            import shutil
            shutil.copyfile(DATA_FILE, backup)
            print(f"[黄瓜] 已备份损坏文件到 {backup}", flush=True)
        except Exception as be:
            print(f"[黄瓜] 备份失败: {be}", flush=True)
    return {
        "version": DATA_VERSION,
        "users": {},      # "群号:QQ号" -> {"small": n, "big": n, "last_sign": "YYYY-MM-DD", "total_sign": n}
        "game": {},       # "群号:QQ号" -> {"last": 时间戳, "type": "..."} 冷却用
    }


def _needs_migration(data: dict) -> bool:
    """旧格式：users/game 的键是纯 QQ 号（没有 "群号:" 前缀）。"""
    if data.get("version") == DATA_VERSION:
        return False
    for section in ("users", "game"):
        for k in (data.get(section) or {}):
            if ":" not in str(k):
                return True
    return False


def _migrate_legacy(data: dict) -> None:
    """把不分群的老数据整体挂到 LEGACY_GROUP 名下：数据不丢，也不会串到别的群。
    迁移前留一份原始文件备份（一次性）。"""
    try:
        import shutil
        backup = DATA_FILE.with_name(DATA_FILE.name + ".pre_group_iso")
        if not backup.exists():
            shutil.copyfile(DATA_FILE, backup)
            print(f"[黄瓜] 迁移前备份老数据到 {backup.name}", flush=True)
    except Exception as e:
        print(f"[黄瓜] 迁移备份失败（继续迁移）: {e}", flush=True)
    moved = 0
    for section in ("users", "game"):
        old = data.get(section) or {}
        new = {(_key(LEGACY_GROUP, str(k)) if ":" not in str(k) else str(k)): v for k, v in old.items()}
        moved += sum(1 for k in old if ":" not in str(k))
        data[section] = new
    data["version"] = DATA_VERSION
    _save(data)
    print(f"[黄瓜] 数据已改为按群独立：{moved} 条老记录归入来源群 {LEGACY_GROUP}", flush=True)


def _save(data: dict) -> None:
    try:
        data["version"] = DATA_VERSION
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[黄瓜] 保存失败: {e}", flush=True)


def _get_user(data: dict, gid: str, uid: str) -> dict:
    u = data["users"].setdefault(_key(gid, uid),
                                 {"small": 0, "big": 0, "last_sign": "", "total_sign": 0, "name": ""})
    # 老用户可能缺新字段，补齐
    u.setdefault("small", 0)
    u.setdefault("big", 0)
    u.setdefault("last_sign", "")
    u.setdefault("total_sign", 0)
    u.setdefault("affection", 0)
    return u


def _set_name(data: dict, gid: str, uid: str, name: str) -> None:
    """记录用户显示名（群名片优先）。"""
    if name:
        data["users"][_key(gid, uid)]["name"] = name


# ---------- 签到 ----------

def sign_in(gid: str, uid: str, name: str = "") -> str:
    data = _load()
    u = _get_user(data, gid, uid)
    _set_name(data, gid, uid, name)
    today = time.strftime("%Y-%m-%d")
    if u["last_sign"] == today:
        return f"今天已经签到过啦，明天再来种黄瓜吧。"
    u["last_sign"] = today
    u["total_sign"] += 1
    # 签到奖励：随机 1~3 根小黄瓜，累计签到每 7 天额外得一根大黄瓜
    small = random.randint(1, 3)
    big = 0
    if u["total_sign"] % 7 == 0:
        big = 1
    u["small"] += small
    u["big"] += big
    _save(data)
    msg = f"签到成功！获得 {small} 根小黄瓜。"
    if big:
        msg += f" 连续签到 {u['total_sign']} 天，额外奖励 1 根大黄瓜！"
    msg += f"\n仓库：小黄瓜 x{u['small']}，大黄瓜 x{u['big']}"
    return msg


# ---------- 小游戏：猜大小 ----------

def start_guess(gid: str, uid: str, bet: int, name: str = "") -> str:
    """开始猜大小：押注，摇骰子保密，等用户回复答案。"""
    data = _load()
    u = _get_user(data, gid, uid)
    _set_name(data, gid, uid, name)
    g = data["game"].setdefault(_key(gid, uid), {"last": 0, "type": ""})

    if bet < 1:
        return "至少押 1 根小黄瓜哦。"
    if u["small"] < bet:
        return f"仓库里的小黄瓜不够啦，只有 {u['small']} 根。去签到或种黄瓜吧。"

    roll = random.randint(1, 6) + random.randint(1, 6)
    g["type"] = "guess"
    g["bet"] = bet
    g["target"] = roll
    g["result"] = "大" if roll >= 7 else "小"
    g["ts"] = time.time()
    _save(data)
    return f"骰子已摇好（两个骰子相加 2~12）。你押了 {bet} 根小黄瓜，猜'大'还是'小'？\n回复：引用这条消息，说 大 或 小。"


def answer_guess(gid: str, uid: str, choice: str, name: str = "") -> str:
    """结算猜大小。"""
    data = _load()
    u = _get_user(data, gid, uid)
    _set_name(data, gid, uid, name)
    g = data["game"].setdefault(_key(gid, uid), {"last": 0, "type": ""})

    if g.get("type") != "guess":
        return "你还没开始猜大小呢。先发'猜大小 押注数'开始吧。"
    if time.time() - g.get("ts", 0) > 300:
        g["type"] = ""
        _save(data)
        return "这局超时了，重新发'猜大小'再来吧。"
    if choice not in ("大", "小"):
        return "要说'大'或'小'哦。"

    roll = g.get("target", 7)
    result = g.get("result", "大")
    bet = g.get("bet", 1)
    g["type"] = ""
    if choice == result:
        win = bet
        u["small"] += win
        _save(data)
        return f"骰子：{roll}（{result}）！猜对了！赢得 {win} 根小黄瓜。现在仓库小黄瓜 x{u['small']}。"
    else:
        # 结算时余额可能已被其他操作（如投喂）消耗，扣到负数前保护
        if u["small"] < bet:
            deduct = u["small"]
            u["small"] = 0
            _save(data)
            return f"骰子：{roll}（{result}）……猜错了，但你黄瓜不够押注的 {bet} 根了，只扣了剩余的 {deduct} 根。现在仓库小黄瓜 x{u['small']}。"
        u["small"] -= bet
        _save(data)
        return f"骰子：{roll}（{result}）……猜错了，输掉 {bet} 根小黄瓜。现在仓库小黄瓜 x{u['small']}。"


# ---------- 小游戏：猜数字 ----------

def start_number(gid: str, uid: str, name: str = "") -> str:
    """开始猜数字：随机一个 1~10 保密，3 次机会。"""
    data = _load()
    u = _get_user(data, gid, uid)
    _set_name(data, gid, uid, name)
    g = data["game"].setdefault(_key(gid, uid), {"last": 0, "type": ""})

    target = random.randint(1, 10)
    g["type"] = "number"
    g["target"] = target
    g["tries"] = 0
    g["max_tries"] = 3
    g["ts"] = time.time()
    _save(data)
    return (
        f"我心里想了一个 1~10 的数字。\n"
        f"你有 3 次机会，猜中得 5 根大黄瓜。\n"
        f"回复：引用这条消息，说出你的数字（第 1 次）。"
    )


def answer_number(gid: str, uid: str, guess: int, name: str = "") -> str:
    """结算猜数字：3 次机会，每次给大了/小了提示。"""
    data = _load()
    u = _get_user(data, gid, uid)
    _set_name(data, gid, uid, name)
    g = data["game"].setdefault(_key(gid, uid), {"last": 0, "type": ""})

    if g.get("type") != "number":
        return "你还没开始猜数字呢。先发'猜数字'开始吧。"
    if time.time() - g.get("ts", 0) > 300:
        g["type"] = ""
        _save(data)
        return "这局超时了，重新发'猜数字'再来吧。"
    if guess < 1 or guess > 10:
        return "猜 1 到 10 之间的数字哦。"

    target = g.get("target", 5)
    g["tries"] = g.get("tries", 0) + 1
    remaining = g.get("max_tries", 3) - g["tries"]

    if guess == target:
        u["big"] += 5
        g["type"] = ""
        _save(data)
        return f"答案就是 {target}！第 {g['tries']} 次就猜中了！获得 5 根大黄瓜！仓库大黄瓜 x{u['big']}。"

    if remaining <= 0:
        # 三次用完
        u["small"] = max(0, u["small"] - 2)
        g["type"] = ""
        _save(data)
        return f"答案是 {target}，三次机会都用完了……扣 2 根小黄瓜。仓库小黄瓜 x{u['small']}。"

    hint = "大了" if guess > target else "小了"
    _save(data)
    return (
        f"猜{guess}……{hint}（第 {g['tries']} 次，还剩 {remaining} 次）\n"
        f"回复：引用这条消息，再猜一次。"
    )


# ---------- 仓库 ----------

def _display_name(u: dict) -> str:
    """显示名：有排行榜称号则带称号。"""
    disp = u.get("name") or ""
    title = u.get("title")
    if title:
        return f"「{title}」{disp}" if disp else f"「{title}」"
    return disp


def warehouse(gid: str, uid: str, name: str = "") -> str:
    data = _load()
    u = _get_user(data, gid, uid)
    _set_name(data, gid, uid, name)
    _save(data)
    disp = _display_name(u)
    head = f"黄瓜仓库（{disp}）：\n" if disp else "黄瓜仓库：\n"
    return (
        f"{head}"
        f"🥒 小黄瓜 x{u['small']}\n"
        f"🥒 大黄瓜 x{u['big']}\n"
        f"累计签到 {u['total_sign']} 天"
    )


# ---------- 排行榜 ----------

def ranking(gid: str) -> str:
    """本群排行榜：只统计当前群的记录，别的群的数据不参与。"""
    data = _load()
    prefix = f"{gid}:"
    users = {k: v for k, v in data["users"].items() if str(k).startswith(prefix)}
    if not users:
        return "还没有人种过黄瓜，快来签到吧！"
    # 按 大黄瓜*3 + 小黄瓜 综合排名（get 兜底，防历史/手工数据缺字段 KeyError）
    rows = sorted(
        users.items(),
        key=lambda kv: kv[1].get("big", 0) * 3 + kv[1].get("small", 0),
        reverse=True,
    )[:10]
    lines = ["🥒 黄瓜排行榜 🥒"]
    for i, (key, u) in enumerate(rows, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}.")
        disp = _display_name(u) or str(key).split(":", 1)[-1]
        lines.append(f"{medal} {disp}：大黄瓜x{u.get('big', 0)} 小黄瓜x{u.get('small', 0)}")
    return "\n".join(lines)


# ---------- 木子米的菜单与投喂（今天吃什么 + 黄瓜兑换菜品加好感度） ----------
# (菜名, 小黄瓜成本, 大黄瓜成本, 好感度+值)。黄瓜菜为主，混入少量其他菜换口味。
DISHES = [
    # 小黄瓜菜（成本1~3根小黄瓜）
    ("拍黄瓜", 1, 0, 1),
    ("凉拌黄瓜", 1, 0, 1),
    ("蒜泥黄瓜", 1, 0, 1),
    ("腌黄瓜", 1, 0, 1),
    ("酸黄瓜", 1, 0, 1),
    ("黄瓜泡菜", 1, 0, 1),
    ("黄瓜汁", 1, 0, 1),
    ("黄瓜拌木耳", 1, 0, 1),
    ("黄瓜拌花生", 1, 0, 1),
    ("蓑衣黄瓜", 2, 0, 1),
    ("黄瓜炒鸡蛋", 2, 0, 1),
    ("黄瓜蛋花汤", 2, 0, 1),
    ("黄瓜皮蛋汤", 2, 0, 1),
    ("黄瓜拌腐竹", 2, 0, 1),
    ("酸奶黄瓜沙拉", 2, 0, 1),
    ("黄瓜三明治", 2, 0, 1),
    ("黄瓜寿司卷", 2, 0, 1),
    ("黄瓜奶昔", 2, 0, 1),
    ("黄瓜炒肉丝", 3, 0, 2),
    ("黄瓜虾仁", 3, 0, 2),
    ("黄瓜鸡蛋饼", 3, 0, 2),
    ("黄瓜馅饺子", 3, 0, 2),
    # 其他菜（换换口味）
    ("蛋炒饭", 1, 0, 1),
    ("冰淇淋", 1, 0, 1),
    ("咖喱饭", 2, 0, 1),
    ("拉面", 2, 0, 1),
    ("煎饺", 2, 0, 1),
    ("麻婆豆腐", 2, 0, 1),
    ("草莓蛋糕", 2, 0, 1),
    ("红烧肉", 3, 0, 2),
    ("寿司拼盘", 3, 0, 2),
    # 大黄瓜豪华菜
    ("黄瓜刺身拼盘", 0, 1, 4),
    ("黄瓜天妇罗", 0, 1, 4),
    ("金枪鱼黄瓜沙拉", 0, 1, 4),
    ("黄瓜芝士焗饭", 0, 1, 4),
    ("黄瓜冰淇淋", 0, 1, 4),
    ("黄瓜蛋糕", 0, 1, 4),
]
DISH_BY_NAME = {d[0]: d for d in DISHES}

# 好感度等级（按累计好感度）
AFFECTION_TIERS = [(35, "挚友"), (20, "好朋友"), (10, "朋友"), (5, "眼熟"), (0, "陌生人")]

# 今天吃什么的回复模板（若叶睦人设口吻）
EAT_TEMPLATES = [
    "今天吃了{name}。……好吃。",
    "吃了{name}。嗯，很满足。",
    "……今天吃的是{name}。",
    "{name}。……我自己种的黄瓜做的。",
    "今天做了{name}，吃得很干净。",
    "（想了想）……{name}。",
    "嗯。{name}。",
]

# 投喂成功回复模板
FEED_TEMPLATES = [
    "（接过{name}，小口吃着）……好吃。",
    "（眼睛亮了）{name}……谢谢。",
    "（慢慢吃完了{name}）……嗯，谢谢你。",
    "（捧着{name}）……我会好好吃的。",
]


def random_dish_name() -> str:
    """随机一道菜（今天吃什么用）。"""
    return random.choice(DISHES)[0]


def eat_today() -> str:
    """木子米今天吃了什么：随机菜品 + 人设口吻。"""
    dish = random.choice(DISHES)
    name = dish[0]
    # 非黄瓜菜不用"自己种的黄瓜做的"这类模板，避免文案穿帮
    if "黄瓜" in name:
        pool = EAT_TEMPLATES
    else:
        pool = [t for t in EAT_TEMPLATES if "黄瓜" not in t]
    return random.choice(pool).format(name=name)


def _affection_tier(n: int) -> str:
    for threshold, tier in AFFECTION_TIERS:
        if n >= threshold:
            return tier
    return "陌生人"


def menu() -> str:
    """完整菜单：菜品+价格，投喂格式说明。"""
    small = [d for d in DISHES if d[2] == 0]
    big = [d for d in DISHES if d[2] > 0]
    lines = [
        "🥒 木子米的菜单 🥒",
        "（数字=所需小黄瓜根数，×N大=大黄瓜N根）",
        "【小黄瓜菜】" + " ".join(f"{d[0]}{d[1]}" for d in small),
        "【大黄瓜菜】" + " ".join(f"{d[0]}×{d[2]}大" for d in big),
        "投喂格式：投喂 菜名（如：投喂 拍黄瓜）。投喂后好感度+1~4，发'好感度'可查看。",
    ]
    return "\n".join(lines)


def feed(gid: str, uid: str, dish_name: str, name: str = "") -> str:
    """用黄瓜兑换菜品投喂木子米：扣黄瓜、加好感度。"""
    data = _load()
    u = _get_user(data, gid, uid)
    _set_name(data, gid, uid, name)
    dish = DISH_BY_NAME.get(dish_name)
    if not dish:
        return f"菜单里没有{dish_name}……看看'菜单'吧。"
    dname, s_cost, b_cost, gain = dish
    if u["small"] < s_cost or u["big"] < b_cost:
        need = f"{s_cost} 根小黄瓜" if b_cost == 0 else f"{b_cost} 根大黄瓜"
        return f"黄瓜不够啦……{dname}需要 {need}。先去签到或种黄瓜吧。"
    u["small"] -= s_cost
    u["big"] -= b_cost
    u["affection"] = u.get("affection", 0) + gain
    _save(data)
    msg = random.choice(FEED_TEMPLATES).format(name=dname)
    msg += f"\n好感度 +{gain}（{_affection_tier(u['affection'])} Lv.{u['affection']}）"
    return msg


def affection(gid: str, uid: str, name: str = "") -> str:
    """查看与木子米的好感度（按群独立：同一个群友在不同群的好感度互不影响）。"""
    data = _load()
    u = _get_user(data, gid, uid)
    _set_name(data, gid, uid, name)
    n = u.get("affection", 0)
    return f"你和木子米的好感度：{n}（{_affection_tier(n)}）\n投喂菜品可以增加好感度，发'菜单'看看有什么菜吧。"


# ---------- 黄瓜兑换（3 小黄瓜 = 1 大黄瓜） ----------
EXCHANGE_RATE = 3


def exchange(gid: str, uid: str, direction: str, count: int, name: str = "") -> str:
    """黄瓜互兑。direction: 'to_big'（小换大，3小→1大）/'to_small'（大换小，1大→3小）。"""
    data = _load()
    u = _get_user(data, gid, uid)
    _set_name(data, gid, uid, name)
    if count < 1:
        return "数量至少是 1 哦。"
    if direction == "to_big":
        need = count * EXCHANGE_RATE
        if u["small"] < need:
            return f"小黄瓜不够啦，换 {count} 根大黄瓜需要 {need} 根小黄瓜（当前 {u['small']} 根）。"
        u["small"] -= need
        u["big"] += count
        _save(data)
        return f"兑换成功：{need} 根小黄瓜 → {count} 根大黄瓜。仓库：小黄瓜x{u['small']} 大黄瓜x{u['big']}。"
    else:
        if u["big"] < count:
            return f"大黄瓜不够啦，换 {count} 根小黄瓜需要 {count} 根大黄瓜（当前 {u['big']} 根）。"
        u["big"] -= count
        u["small"] += count * EXCHANGE_RATE
        _save(data)
        return f"兑换成功：{count} 根大黄瓜 → {count * EXCHANGE_RATE} 根小黄瓜。仓库：小黄瓜x{u['small']} 大黄瓜x{u['big']}。"


# ---------- 赠送黄瓜（把自己的黄瓜送给别人） ----------

def gift(gid: str, uid_from: str, uid_to: str, kind: str, count: int, name: str = "") -> str:
    """赠送黄瓜给同群的别的玩家。kind: 'small'/'big'。目标用户不存在则自动建档。"""
    data = _load()
    u = _get_user(data, gid, uid_from)
    _set_name(data, gid, uid_from, name)
    if uid_from == uid_to:
        return "……不能送给自己啦。"
    if count < 1:
        return "数量至少是 1 哦。"
    if kind == "big":
        if u["big"] < count:
            return f"大黄瓜不够啦，要送 {count} 根大黄瓜（当前 {u['big']} 根）。去'兑换黄瓜 大'换一点吧。"
        u["big"] -= count
        ut = _get_user(data, gid, uid_to)
        ut["big"] = ut.get("big", 0) + count
        _save(data)
        return f"（把 {count} 根大黄瓜递过去）……给你的。\n对方收到 {count} 根大黄瓜！你还有大黄瓜x{u['big']}。"
    else:
        if u["small"] < count:
            return f"小黄瓜不够啦，要送 {count} 根小黄瓜（当前 {u['small']} 根）。去签到种黄瓜吧。"
        u["small"] -= count
        ut = _get_user(data, gid, uid_to)
        ut["small"] = ut.get("small", 0) + count
        _save(data)
        return f"（把 {count} 根小黄瓜递过去）……给你的。\n对方收到 {count} 根小黄瓜！你还有小黄瓜x{u['small']}。"


# ---------- 大黄瓜兑换商城 ----------
# (商品名, 大黄瓜价格, 类型, 附加参数)
SHOP_ITEMS = [
    ("小黄瓜礼包", 1, "small_pack", 4),        # 1大 → +4小
    ("点歌卡", 2, "song", None),               # 木子米唱一句
    ("专属情书", 3, "love_letter", None),      # 木子米写一句
    ("排行榜称号", 5, "title", None),          # 名字带永久称号
]
SHOP_BY_NAME = {s[0]: s for s in SHOP_ITEMS}

# 点歌卡：木子米"唱"的歌词（人设口吻）
SONG_POOL = [
    "（轻轻哼）♪ 黄瓜~黄瓜~种满园~ ♪",
    "（弹起吉他）♪ 迷子でも……进むよ。♪（小声）……就唱到这里。",
    "（哼）♪ 月之森的风，吹过黄瓜田~ ♪",
    "（清嗓）♪ 小小的黄瓜，长大啦~ ♪",
    "（抱着吉他点头）♪……嗯，这首歌是唱给你的。♪",
]
# 专属情书：木子米写的句子
LETTER_POOL = [
    "……给你的。'谢谢你陪我种黄瓜。'",
    "……'重要的话，要直说。你也很重要。'",
    "……'黄瓜分你一半。'（这是我能给出的全部了）",
    "……'如果哪天你也不开心了，就来我的黄瓜田。'",
    "……'我从来没觉得……和你说话不开心过。'",
]
# 排行榜称号（购买时随机分配一个未拥有的）
TITLE_POOL = ["黄瓜之王", "木子米的挚友", "种瓜大师", "黄瓜守护者", "睦的专属园丁"]


def shop() -> str:
    """兑换商城商品列表。"""
    lines = [
        "🥒 木子米的兑换商城 🥒",
        "（大黄瓜付款，发：购买 商品名）",
        "· 小黄瓜礼包 ×1大 → +4 根小黄瓜",
        "· 点歌卡 ×2大 → 木子米为你唱一句",
        "· 专属情书 ×3大 → 木子米给你写一句专属话",
        "· 排行榜称号 ×5大 → 你的名字带称号（仓库/排行榜可见，永久）",
        "黄瓜兑换：3 根小黄瓜 = 1 根大黄瓜（发：兑换黄瓜 大 或 兑换黄瓜 小）",
    ]
    return "\n".join(lines)


def buy(gid: str, uid: str, item_name: str, name: str = "") -> str:
    """用大黄瓜购买商城商品。"""
    data = _load()
    u = _get_user(data, gid, uid)
    _set_name(data, gid, uid, name)
    item = SHOP_BY_NAME.get(item_name)
    if not item:
        return f"商城没有{item_name}……发'商城'看看有什么吧。"
    iname, price, itype, extra = item
    if u["big"] < price:
        return f"大黄瓜不够啦，{iname}需要 {price} 根大黄瓜（当前 {u['big']} 根）。去'兑换黄瓜 大'换一点吧。"
    if itype == "title" and u.get("title"):
        return f"你已经有称号「{u['title']}」啦，一个就够啦。"
    u["big"] -= price
    if itype == "small_pack":
        u["small"] += extra
        _save(data)
        return f"打开小黄瓜礼包：获得 {extra} 根小黄瓜！仓库：小黄瓜x{u['small']} 大黄瓜x{u['big']}。"
    if itype == "song":
        _save(data)
        return f"（接过点歌卡）……好。\n{random.choice(SONG_POOL)}"
    if itype == "love_letter":
        _save(data)
        return f"（脸微微红，把信纸递给你）……给你的。\n{random.choice(LETTER_POOL)}"
    if itype == "title":
        owned = {u.get("title")} if u.get("title") else set()
        pool = [t for t in TITLE_POOL if t not in owned] or TITLE_POOL
        u["title"] = random.choice(pool)
        _save(data)
        return f"（郑重地点头）……以后你的名字就是「{u['title']}」了。"
    _save(data)
    return "（歪头）这个商品还没做好……"
