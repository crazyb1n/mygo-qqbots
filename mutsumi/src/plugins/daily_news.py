"""每日新闻推送：每天早报/午报/晚报三个时段，用"若叶睦"账号把知识类热榜新闻
（科技/财经/开发者/时政等，自动过滤娱乐八卦）整理成 Word 文档，发到指定 QQ 群和好友。

数据源：DailyHot API（https://github.com/imsyy/DailyHotApi），默认用公共实例
  https://dailyhotapi.vercel.app 。只取知识性榜单，娱乐新闻按关键词过滤掉。

架构说明（与既有 bot 完全兼容，不改 NapCat）：
  - 复用 bot 进程内已建立的 OneBot WS 连接：文本用 send_msg、文档用 upload_group_file/upload_private_file。
  - 定时用 nonebot_plugin_apscheduler 的全局 scheduler，避免新增常驻线程。
  - 时段从 .env 的 NEWS_SCHEDULE 读取（逗号分隔"名称=HH:MM"），默认 早报=08:00,午报=12:30,晚报=21:00。
  - 榜单从 .env 的 NEWS_BOARDS 读取（"分类=端点1|端点2,分类2=端点"）。
  - 总结正文：优先用 LLM 生成精炼中文总结；失败时降级为"按分类列标题"（不含链接），保证到点一定发出。
  - 触发即用 asyncio.create_task 包裹，防止网络阻塞拖慢整个事件循环。
"""
import asyncio
import contextlib
import html
import json
import os
_REASONING_EFFORT = os.getenv("DEEPSEEK_REASONING_EFFORT", "none").strip()
import re
import ssl
import urllib.request
from datetime import datetime

from nonebot import get_driver, logger
from nonebot_plugin_apscheduler import scheduler

from openai import AsyncOpenAI

driver = get_driver()

# ---------------- 配置 ----------------
NEWS_GROUPS = [int(x) for x in os.getenv("NEWS_GROUPS", "").split(",") if x.strip().isdigit()]
NEWS_FRIENDS = [int(x) for x in os.getenv("NEWS_FRIENDS", "").split(",") if x.strip().isdigit()]
# DailyHot API 实例地址
DAILYHOT_BASE = os.getenv("DAILYHOT_BASE", "https://dailyhotapi.vercel.app").rstrip("/")
# 每个榜单最多取几条
NEWS_PER_BOARD = int(os.getenv("NEWS_PER_BOARD", os.getenv("NEWS_PER_FEED", "6")))
# 生成的 Word 文档存放目录（绝对路径，NapCat 上传需要）
NEWS_DOC_DIR = os.getenv("NEWS_DOC_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "news_docs"))

# 娱乐类关键词：标题命中即过滤（保证只留知识性新闻）
NEWS_EXCLUDE = [k for k in os.getenv(
    "NEWS_EXCLUDE",
    "明星,综艺,恋情,八卦,离婚,结婚,官宣,塌房,网红,带货,选秀,爱豆,偶像,粉丝,演唱会,"
    "电视剧,追剧,剧透,番剧,动漫,游戏,球星,球员,球队,转会,电影票房,综艺节目,塌了,"
    "rapper,网红打卡,颜值,穿搭,美妆,恋情曝光,分手",
).split(",") if k.strip()]


def _parse_schedule(raw: str) -> list:
    """解析 NEWS_SCHEDULE：'早报=08:00,午报=12:30,晚报=21:00' -> [(名称, 时, 分), ...]。"""
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, _, hhmm = part.partition("=")
        m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", hhmm)
        if not m:
            logger.warning(f"NEWS_SCHEDULE 项格式错误，已跳过: {part!r}")
            continue
        h, mi = int(m.group(1)), int(m.group(2))
        if not (0 <= h <= 23 and 0 <= mi <= 59):
            logger.warning(f"NEWS_SCHEDULE 时间越界，已跳过: {part!r}")
            continue
        out.append((name.strip() or f"{h:02d}:{mi:02d}", h, mi))
    return out


def _parse_boards(raw: str) -> list:
    """解析 NEWS_BOARDS：'科技与商业=36kr|sspai,时政与社会=thepaper' -> [(分类, [端点]), ...]。"""
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        cat, _, eps = part.partition("=")
        eps = [e.strip() for e in eps.split("|") if e.strip()]
        if eps:
            out.append((cat.strip() or "热榜", eps))
    return out


NEWS_SCHEDULE = _parse_schedule(os.getenv("NEWS_SCHEDULE", "早报=08:00,午报=12:30,晚报=21:00"))
# 默认只取知识性榜单：科技财经 / 开发者开源 / 时政社会
NEWS_BOARDS = _parse_boards(os.getenv(
    "NEWS_BOARDS",
    "科技与商业=36kr|sspai,开发者与开源=github|juejin|v2ex,时政与社会=thepaper",
))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

# 抓取超时（秒）
FETCH_TIMEOUT = 20


# ---------------- 数据抓取（DailyHot） ----------------
def _clean(text: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", str(text or "")).strip())


def _http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (news-digest/1.0)"})
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT, context=ctx) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.URLError:
        # 本地缺 CA 证书包时降级为不校验（仅读公开榜单）
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT, context=ctx) as r:
            return json.loads(r.read().decode("utf-8", "replace"))


def _fetch_board(endpoint: str) -> list:
    """抓取单个榜单，返回 [{title, link, time, desc, source}, ...]。"""
    data = _http_json(f"{DAILYHOT_BASE}/{endpoint}")
    if data.get("code") != 200:
        raise RuntimeError(data.get("message") or f"code={data.get('code')}")
    src = data.get("title") or endpoint
    items = []
    for x in data.get("data", []):
        title = _clean(x.get("title"))
        if not title:
            continue
        hot = x.get("hot")
        items.append({
            "title": title,
            "link": (x.get("url") or x.get("mobileUrl") or "").strip(),
            "time": f"热度{hot}" if hot not in (None, "", 0) else "",
            "desc": _clean(x.get("desc"))[:200],
            "source": src,
        })
    return items


def _is_entertainment(title: str) -> bool:
    return any(k and k in title for k in NEWS_EXCLUDE)


async def _fetch_all():
    """抓取所有分类榜单并过滤娱乐、去重，返回 {分类: [条目, ...]}。"""
    result = {}
    for category, endpoints in NEWS_BOARDS:
        merged = []
        for ep in endpoints:
            try:
                merged.extend(await asyncio.to_thread(_fetch_board, ep))
            except Exception as e:
                logger.warning(f"榜单抓取失败 {ep}: {e}")
        # 过滤娱乐新闻
        merged = [it for it in merged if not _is_entertainment(it["title"])]
        # 去重（按标题）
        seen, uniq = set(), []
        for it in merged:
            if it["title"] in seen:
                continue
            seen.add(it["title"])
            uniq.append(it)
        result[category] = uniq[: NEWS_PER_BOARD * 2]
    return result


# ---------------- 总结生成 ----------------
class _SummaryBlocked(RuntimeError):
    """上游内容审核拦截了总结请求（finish_reason=content_filter）。"""


_REFUSAL_HINTS = ("抱歉", "无法回答", "无法识别", "未找到相关", "没有找到相关")


def _looks_like_refusal(text: str) -> bool:
    """识别模型把"审核拦截"包装成的短道歉话术，避免当成正文写进文档。"""
    compact = re.sub(r"\s+", "", text or "")
    return len(compact) < 80 and any(k in compact for k in _REFUSAL_HINTS)


def _build_prompt(data: dict) -> str:
    # 所有分类都为空时直接给哨兵，不能只输出分类标题——否则模型会回"抱歉没找到"
    if not any(data.values()):
        return "（今日未抓到可用新闻）"
    lines = []
    for category, items in data.items():
        lines.append(f"\n## {category}")
        for it in items:
            src = it.get("source", "")
            hot = f"（{it['time']}）" if it.get("time") else ""
            lines.append(f"- [{src}] {it['title']}{hot} {it.get('desc', '')[:100]}")
    return "\n".join(lines)


async def _summarize_with_llm(data: dict, edition: str) -> str:
    """用 LLM 生成精炼中文总结（不含链接）。edition 为"早报/午报/晚报"。"""
    from .llm_fallback import build_client
    client = build_client(AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL))
    prompt = (
        "你是一名科技/财经/时政新闻编辑，读者是一名大学生（理工科背景、对科技原理有兴趣）。"
        "下面是从中文热榜抓取的各分类条目，请整理成一份适合发给读者本人看的「{edition}」，用中文，"
        "语气自然，比普通简报更有信息量：既要快，也要让读者**每天真的学到新东西**。要求：\n"
        "1. 开头一句话总览今天/此刻最值得关注的主线；\n"
        "2. 分分类整理，每类挑 2-4 条**知识性、信息量高**的条目（科技、财经、产业、开源、时政、社会民生），"
        "每条先一两句话说清“发生了什么”，其中每类至少一条再补一句“为什么重要/影响是什么”；\n"
        "3. 明显是娱乐八卦/明星绯闻/影视综艺的内容直接略过，不要写；\n"
        "4. 分类之后单列一个【今日新知】板块：从今天的新闻里挑 1 个最有知识含量的新概念/新事物，"
        "用 150-250 字讲清楚“它是什么、为什么重要、和读者可能已知的什么知识相连”，"
        "让读者每天稳定学到一个新东西；\n"
        "5. 再单列一个【科技原理拆解】板块：从今天的科技/科学类新闻里挑 1-2 条，拆解它背后的"
        "科学技术原理——“它到底是怎么做到的？”每条 150-300 字、分 2-4 个要点讲清机制，"
        "面向大学生，可以打比方，但不要空话套话；如果当天挑不出，就拆解财经/时政新闻背后的一种机制"
        "（如汇率传导、关税、算力/能源约束）；\n"
        "6. 结尾一句收尾。全文控制在 1300 字以内。\n"
        "**重要：不要输出任何网址/链接，不要写原文出处，只讲新闻本身。**\n\n"
        + _build_prompt(data)
    )
    base_kwargs = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        # 额度给足，避免正文被截断
        "max_tokens": 20480,
    }
    # 关闭推理模型的思考链（思考token会被丢弃但仍占额度，长prompt下会把正文挤成空）；
    # 个别模型不支持该参数，失败则不带它重试一次
    try:
        resp = await client.chat.completions.create(
            **base_kwargs, extra_body=({"reasoning_effort": _REASONING_EFFORT} if _REASONING_EFFORT else {})
        )
    except Exception:
        resp = await client.chat.completions.create(**base_kwargs)
    choice = resp.choices[0]
    content = (choice.message.content or "").strip()
    # 内容审核返回的是 200 + 道歉话术，必须在落盘前拦住并降级
    if choice.finish_reason == "content_filter" or _looks_like_refusal(content):
        raise _SummaryBlocked(f"finish_reason={choice.finish_reason} content={content[:60]!r}")
    return content


def _summarize_fallback(data: dict, edition: str) -> str:
    """LLM 不可用时的降级：按分类列标题（不含链接）。"""
    today = datetime.now().strftime("%Y-%m-%d")
    out = [f"{edition} · {today}\n"]
    for category, items in data.items():
        if not items:
            continue
        out.append(f"\n【{category}】")
        for it in items[:NEWS_PER_BOARD]:
            out.append(f"· {it['title']}")
    return "\n".join(out)


def _strip_links(text: str) -> str:
    """兜底：LLM 偶尔仍会带链接，这里统一去掉 URL 与出处行。"""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)
    text = re.sub(r"[（(]\s*来源[:：]?\s*[^）)]*[）)]", "", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


# ---------------- Word 文档 ----------------
def _build_docx(content: str, title: str) -> str:
    """把总结文本渲染成 Word 文档，返回绝对路径。"""
    from docx import Document

    os.makedirs(NEWS_DOC_DIR, exist_ok=True)
    doc = Document()
    doc.add_heading(title, level=0)

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        # 板块标题：== xxx == / 【xxx】(允许同行带尾巴，如"【今日新知】光遗传学")
        #           / ## xxx / ### xxx / 整行加粗 **xxx**
        m = re.match(r"^\s*【(.+?)】\s*(.*)$", line)
        if m:
            doc.add_heading(m.group(1).strip(), level=1)
            if m.group(2).strip():
                doc.add_paragraph(m.group(2).strip())
            continue
        m = re.match(r"^\s*(?:={2,}\s*(.+?)\s*={2,}|#{2,4}\s*(.+?))\s*$", line)
        if not m:
            m = re.match(r"^\s*\*\*(.+?)\*\*\s*$", line)
        if m:
            doc.add_heading(next(g for g in m.groups() if g), level=1)
            continue
        # 列表项：· / - / • / * 后必须跟空格（避免把 **加粗** 误判为列表）
        if re.match(r"^\s*[·\-•]\s+", line) or re.match(r"^\s*\*\s+", line):
            doc.add_paragraph(re.sub(r"^\s*[·\-•*]\s+", "", line), style="List Bullet")
            continue
        # 行首粗体小节：**xxx**：内容
        m2 = re.match(r"^\*\*(.+?)\*\*[:：]?\s*(.*)$", line)
        if m2:
            p = doc.add_paragraph()
            p.add_run(m2.group(1)).bold = True
            if m2.group(2):
                p.add_run("：" + m2.group(2))
            continue
        # 普通行：内联 **粗体** 单独成 run
        if "**" in line:
            p = doc.add_paragraph()
            for idx, part in enumerate(re.split(r"\*\*(.+?)\*\*", line)):
                if not part:
                    continue
                (p.add_run(part).bold if idx % 2 else p.add_run(part))
            continue
        doc.add_paragraph(line)

    safe = re.sub(r"[\\/:*?\"<>|]", "_", title)
    path = os.path.abspath(os.path.join(NEWS_DOC_DIR, f"{safe}.docx"))
    doc.save(path)
    return path


# ---------------- 发送 ----------------
async def _send_news(bot, content: str, title: str, doc_path: str):
    """给每个目标群/好友发一条简短提示 + Word 文档。"""
    notice = f"{title}已整理好，详见文档。"
    for gid in NEWS_GROUPS:
        with contextlib.suppress(Exception):
            await bot.send_msg(message_type="group", group_id=gid, message=notice)
            await bot.upload_group_file(group_id=gid, file=doc_path, name=os.path.basename(doc_path))
            logger.info(f"新闻文档已发送到群 {gid}")
    for uid in NEWS_FRIENDS:
        with contextlib.suppress(Exception):
            await bot.send_msg(message_type="private", user_id=uid, message=notice)
            await bot.upload_private_file(user_id=uid, file=doc_path, name=os.path.basename(doc_path))
            logger.info(f"新闻文档已发送到好友 {uid}")


async def _run_digest(edition: str = "晚报"):
    """抓取并生成一次新闻总结（edition 为"早报/午报/晚报"），发到群和好友。"""
    if not (NEWS_GROUPS or NEWS_FRIENDS):
        logger.warning("未配置 NEWS_GROUPS/NEWS_FRIENDS，跳过新闻推送")
        return
    try:
        data = await _fetch_all()
    except Exception as e:
        logger.error(f"新闻抓取整体失败: {e}")
        return
    if not any(data.values()):
        logger.error(f"{edition} 所有榜单均无数据，跳过本次推送（避免发出空文档）")
        return

    content = ""
    try:
        content = await _summarize_with_llm(data, edition)
        logger.info(f"{edition} 总结由 LLM 生成")
    except _SummaryBlocked as e:
        # 上游审核常由单个分类（多为时政）触发：逐类剔除后重试，
        # 被剔除的分类改成标题列表附在文末，保证文档仍有信息量。
        logger.warning(f"{edition} 总结被内容审核拦截，尝试剔除单个分类重试: {e}")
        for cat in list(data):
            subset = {k: v for k, v in data.items() if k != cat}
            try:
                content = await _summarize_with_llm(subset, edition)
                titles = "\n".join(f"· {it['title']}" for it in data.get(cat, [])[:NEWS_PER_BOARD])
                if titles:
                    content += f"\n\n【{cat}】\n{titles}"
                logger.info(f"{edition} 总结由 LLM 生成（已剔除触发审核的分类: {cat}）")
                break
            except _SummaryBlocked:
                continue
            except Exception as retry_err:
                logger.warning(f"{edition} 剔除分类 {cat} 后重试失败: {retry_err}")
        if not content.strip():
            logger.warning(f"{edition} 剔除单类后仍被拦截，降级为标题模板")
            content = _summarize_fallback(data, edition)
    except Exception as e:
        logger.warning(f"LLM 总结失败，降级为模板: {e}")
        content = _summarize_fallback(data, edition)

    if not content.strip():
        content = _summarize_fallback(data, edition)

    content = _strip_links(content)
    today = datetime.now().strftime("%Y-%m-%d")
    title = f"{edition} · {today}"

    try:
        doc_path = await asyncio.to_thread(_build_docx, content, title)
        logger.info(f"{edition} 文档已生成: {doc_path}")
    except Exception as e:
        logger.error(f"Word 文档生成失败: {e}")
        # 文档失败则退回纯文本发送（仍不含链接）
        for bot in driver.bots.values():
            for gid in NEWS_GROUPS:
                with contextlib.suppress(Exception):
                    await bot.send_msg(message_type="group", group_id=gid, message=content)
            for uid in NEWS_FRIENDS:
                with contextlib.suppress(Exception):
                    await bot.send_msg(message_type="private", user_id=uid, message=content)
        return

    for bot in driver.bots.values():
        await _send_news(bot, content, title, doc_path)


# ---------------- 定时注册：每个时段一个 cron 任务 ----------------
def _register_schedules():
    """按 NEWS_SCHEDULE 注册定时任务；同名时段用唯一 id，避免冲突。"""
    for idx, (edition, hh, mm) in enumerate(NEWS_SCHEDULE):
        job_id = f"daily_news_{edition}_{hh:02d}{mm:02d}_{idx}"

        async def _job(edition: str = edition):
            asyncio.create_task(_run_digest(edition))

        scheduler.add_job(
            _job,
            "cron",
            hour=hh,
            minute=mm,
            id=job_id,
            misfire_grace_time=300,
            replace_existing=True,
        )
        logger.info(f"已注册新闻定时任务: {edition} {hh:02d}:{mm:02d} (id={job_id})")


if not NEWS_SCHEDULE:
    logger.warning("NEWS_SCHEDULE 为空或全部格式错误，未注册任何新闻定时任务")
else:
    _register_schedules()
