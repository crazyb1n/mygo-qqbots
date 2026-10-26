"""english_reading：英语晨读/晚读推送（与 daily_news 配套的个人学习插件）。

读者画像：大学生，教材 National Geographic Learning《Pathways》第 4 级
（RW4 = Reading, Writing & Critical Thinking 4；LS4 = Listening & Speaking 4），
课程 CEFR 参考级别 C1。目标：每天两篇短文，稳住阅读量、磨语感、攒词汇。
不求量大，只求每天能坚持看。

内容（Word 文档推送）：
  晨读：一篇约 350-900 词的英文文章（从本机可达的英文科技/科普 RSS 现抓现选），
        含导读、逐段译文+解读、生词表（只挑 B2 及以上：学术词汇表/新闻/专业词汇，
        IPA + 英文释义 + 中文 + 原文例句）、值得记住的表达、一句话收获。
        正文里的每个生词在第一次出现处就地中文括注（灰色显示），读的时候不用翻文末词表。
  晚读：另一篇约 200-500 词的短文（早上选材时一并定好、正文存 state 文件），
        含导读、参考译文、生词表、晨读词汇自测（含答案）。
        若当天早上没跑成（bot 停机等），晚读自己现选一篇，自测板块自动省略。

文章来源（2026-10-07 在本机实测可达且正文可抓）：SciTechDaily / ScienceDaily /
Quanta Magazine / Science News / Ars Technica / Nautilus / MIT Tech Review 的 RSS。
选材偏好贴合 Pathways RW4 十个单元的主题（工作与 AI、城市、艺术、交通、协作与群体智能、
语言、能源与资源、健康与长寿、真相与 AI 鉴伪、气候与地球），但任何高质量科技/科普文章
都可用；促销/比价/购物类标题在候选阶段就过滤掉。

工程约定（沿用 daily_news 的成熟姿势与项目红线）：
  - 数据源失败逐层降级；实在发不出就发一条简短文字说明，绝不静默、绝不崩 handler。
  - 阻塞操作全部 asyncio.to_thread / 线程池；每次抓取有超时。
  - LLM 调用姿势与 daily_news 完全一致（同一把 DEEPSEEK_* 配置，
    DEEPSEEK_REASONING_EFFORT 控制，带参失败自动重试一次）。
  - 已用文章 URL、当天晚读素材、晨读生词表记入 english_docs/english_state.json
    （跨天去重 + 晚读不依赖当次网络）。
"""
import asyncio
import contextlib
import json
import os
import re
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from nonebot import get_driver, logger
from nonebot_plugin_apscheduler import scheduler

driver = get_driver()

# ---------------- 配置 ----------------
# 推送目标：默认不推群、只私聊（ENGLISH_GROUPS 留空 = 不推群；ENGLISH_FRIENDS
# 留空 = 沿用 NEWS_FRIENDS）。想连群一起推，往 .env 里填群号即可。
def _ids(raw):
    return [int(x) for x in (raw or "").split(",") if x.strip().isdigit()]

ENGLISH_GROUPS = _ids(os.getenv("ENGLISH_GROUPS", ""))
ENGLISH_FRIENDS = _ids(os.getenv("ENGLISH_FRIENDS", "")) or _ids(os.getenv("NEWS_FRIENDS", ""))

def _parse_schedule(raw: str):
    out = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, _, hhmm = part.partition("=")
        m = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", hhmm)
        if not m:
            logger.warning(f"ENGLISH_SCHEDULE 项格式错误，已跳过: {part!r}")
            continue
        h, mi = int(m.group(1)), int(m.group(2))
        if 0 <= h <= 23 and 0 <= mi <= 59:
            out.append((name.strip() or f"{h:02d}:{mi:02d}", h, mi))
    return out

ENGLISH_SCHEDULE = _parse_schedule(os.getenv("ENGLISH_SCHEDULE", "晨读=07:50,晚读=20:45"))

DOC_DIR = os.getenv("ENGLISH_DOC_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "english_docs")
STATE_PATH = os.path.join(DOC_DIR, "english_state.json")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
_REASONING_EFFORT = os.getenv("DEEPSEEK_REASONING_EFFORT", "none").strip()

FETCH_TIMEOUT = 15
LLM_MAX_TOKENS = 20480          # 同 daily_news：给足额度，思考模型预算不足会把正文挤空
# 2026-10-15 实锤（晨读失败"课程文本异常"）：课程生成是"结构化写作"，思考纯属浪费——
# 配 DEEPSEEK_REASONING_EFFORT=high 时模型能把 20480 token 全花在思考上：
# 实测 same prompt：high → 87s / thinking 61916 字 / content 0 字（finish=length）；
#                    low → 76s / content 4833 字；off → 14s / content 4988 字 ✓。
# 所以课程/选题这类任务一律用 off，并留阶梯兜底（off → low → 配置值）。
LESSON_EFFORT = os.getenv("ENGLISH_LLM_EFFORT", "off")
MAX_CANDIDATE_FETCH = 8         # 最多抓几篇候选正文
MORNING_WORDS = (350, 900)      # 晨读正文词数窗口
EVENING_WORDS = (200, 650)      # 晚读正文词数窗口（更轻，但别太长）
POOL_WORDS = (280, 950)         # 候选池词数窗口（选材前的宽过滤）
RECENT_HOURS = 36               # 候选新鲜度
MAX_VOCAB = 12                  # 生词表行数上限（提示词要求 8-12）

UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

FEEDS = [  # (来源名, 优先级(小=优), RSS 地址)。The Verge/Smithsonian 已实测剔除：
           # 前者促销/消费类太多，后者文章页 403 抓不到正文。
    ("SciTechDaily", 1, "https://scitechdaily.com/feed/"),
    ("ScienceDaily", 1, "https://www.sciencedaily.com/rss/all.xml"),
    ("Quanta Magazine", 2, "https://api.quantamagazine.org/feed/"),
    ("Science News", 2, "https://www.sciencenews.org/feed"),
    ("Ars Technica", 2, "https://feeds.arstechnica.com/arstechnica/index"),
    ("Nautilus", 3, "https://nautil.us/feed/"),
    ("MIT Tech Review", 3, "https://www.technologyreview.com/feed/"),
]

# 选材加分主题：对应 Pathways RW4 十个单元（加一个"前沿科学与太空"兜底）
THEMES = [
    ("工作与AI", ["work", "job", "career", "employment", "labor", "workplace", "office",
                  "remote work", "automation", "robot", "hiring", "workforce"]),
    ("城市", ["city", "urban", "housing", "infrastructure", "traffic", "metro", "megacity"]),
    ("艺术", ["art", "museum", "artist", "design", "architecture", "music"]),
    ("交通", ["transport", "electric vehicle", " ev ", "battery", "train", "aviation",
              "self-driving", "autonomous"]),
    ("协作与群体智能", ["swarm", "collective", "cooperat", "collaborat", " ant ", " bee ",
                        "crowd", "flocking"]),
    ("语言", ["language", "linguistic", "bilingual", "translation", "speech"]),
    ("能源与资源", ["energy", "solar", "grid", "resource", "recycl", "mining", "fuel"]),
    ("健康与长寿", ["health", "aging", "longevity", "brain", "sleep", "diet", "cancer",
                    "disease", "medicine", "implant", "neural", "gene"]),
    ("真相与AI鉴伪", ["deepfake", "misinformation", "disinformation", "fake news", "decept",
                      "fraud", "scam", " bias", "detect", "authentic"]),
    ("气候与地球", ["climate", "carbon", "emission", "warming", "ocean", "biodiversity",
                    "glacier", "ecosystem"]),
    ("前沿科学与太空", ["space", "nasa", "quantum", "physics", "genome", "dna", "telescope",
                        "mars", "particle", "archaeolog"]),
]

_JUNK_TITLE = re.compile(
    r"don't miss|watch:|video:|quiz|slideshow|photos:|best of|sponsored|newsletter"
    r"|web exclusives|week in|gift guide|how to watch|live stream|signup|subscribe"
    # 促销/比价/购物类：对语言学习没营养（2026-10-07 实测 The Verge 源混入多被 LLM 选中的教训）
    r"|cheaper than ever|\bdeals?\b|\$\d|coupon|prime day|black friday|cyber monday"
    r"|\bgiveaway\b|hands-on|best .{0,30}(to buy|you can buy)",
    re.I)
_PROMO_PARA = re.compile(
    r"sign ?up|subscribe|newsletter|advertisement|all rights reserved|©|copyright"
    r"^\s*related:|read more:|cookie|privacy policy|terms of use", re.I)
_ATOM = "{http://www.w3.org/2005/Atom}"


# ---------------- 基础工具 ----------------
def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT, context=ctx) as r:
            return r.read()
    except urllib.error.URLError:
        # 本机缺 CA 证书包时降级为不校验（只读公开内容）
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT,
                                    context=ssl._create_unverified_context()) as r:
            return r.read()


def _clean_inline(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    for a, b in [("&nbsp;", " "), ("&#39;", "'"), ("&apos;", "'"), ("&amp;", "&"),
                 ("&quot;", '"'), ("&lt;", "<"), ("&gt;", ">"), ("&rsquo;", "'"),
                 ("&lsquo;", "'"), ("&rdquo;", '"'), ("&ldquo;", '"'), ("&mdash;", "—"),
                 ("&ndash;", "–"), ("&hellip;", "…"), ("&#8217;", "'"), ("&#8216;", "'"),
                 ("&#8220;", '"'), ("&#8221;", '"'), ("&#8212;", "—")]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def _extract_article(html: str):
    """从文章页提取正文段落列表。"""
    html = re.sub(r"<(script|style|noscript|form)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    paras = []
    for m in re.finditer(r"<p[^>]*>(.*?)</p>", html, flags=re.S | re.I):
        t = _clean_inline(m.group(1))
        if len(t) < 60:
            continue
        if _PROMO_PARA.search(t) and len(t) < 220:
            continue
        paras.append(t)
    return paras


def _parse_feed(xml_bytes: bytes):
    """解析 RSS2.0 / Atom，返回 [(title, link, summary, dt|None)]。"""
    root = ET.fromstring(xml_bytes)
    items = []
    for it in root.findall(".//item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        desc = _clean_inline(it.findtext("description") or "")
        dt = _parse_dt(it.findtext("pubDate") or "")
        if title and link:
            items.append((title, link, desc, dt))
    for it in root.findall(f".//{_ATOM}entry"):
        title = ((it.find(f"{_ATOM}title") is not None and it.find(f"{_ATOM}title").text) or "").strip()
        link = ""
        for l in it.findall(f"{_ATOM}link"):
            if l.get("rel") in (None, "alternate"):
                link = (l.get("href") or "").strip()
                break
        node = it.find(f"{_ATOM}summary") or it.find(f"{_ATOM}content")
        desc = _clean_inline(node.text or "") if node is not None and node.text else ""
        dt = _parse_dt((it.findtext(f"{_ATOM}published") or "") or (it.findtext(f"{_ATOM}updated") or ""))
        if title and link:
            items.append((title, link, desc, dt))
    return items


def _parse_dt(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw)
    except Exception:
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _theme_score(text: str) -> int:
    low = " " + text.lower() + " "
    return sum(1 for _, kws in THEMES for k in kws if k in low)


# ---------------- 状态（去重 + 晚读素材） ----------------
def _load_state() -> dict:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"used_urls": []}


def _save_state(state: dict):
    os.makedirs(DOC_DIR, exist_ok=True)
    state["used_urls"] = state.get("used_urls", [])[-40:]
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_PATH)


# ---------------- 选材 ----------------
def _fetch_one_feed(source: str, url: str):
    try:
        return _parse_feed(_http_get(url))
    except Exception as e:
        logger.warning(f"[english] 源抓取失败 {source}: {e}")
        return []


def _collect_candidates():
    """抓所有源 -> 过滤 -> 去重 -> 排序。返回 [(score, source, title, link, summary, dt)]。"""
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda f: _fetch_one_feed(f[0], f[2]), FEEDS))
    cutoff = datetime.now(timezone.utc) - timedelta(hours=RECENT_HOURS)
    seen, out = set(), []
    for (source, rank, _url), items in zip(FEEDS, results):
        for title, link, desc, dt in items:
            key = re.sub(r"\W+", "", title.lower())[:60]
            if not key or key in seen:
                continue
            if _JUNK_TITLE.search(title) or len(title) < 25:
                continue
            seen.add(key)
            out.append((source, rank, title, link, desc, dt))
    dated = [x for x in out if x[5] is not None]
    if dated:  # 有时间戳的源才做新鲜度过滤，没时间的条目保留
        out = [x for x in out if x[5] is None or x[5] >= cutoff]
    out.sort(key=lambda x: (-_theme_score(x[2] + " " + x[4]), x[1]))
    return out


def _fetch_articles(cands):
    """并发抓候选正文，返回 [(meta, text)]，text 为段落数组。"""
    def _one(c):
        source, _rank, title, link, desc, dt = c
        try:
            paras = _extract_article(_http_get(link).decode("utf-8", "replace"))
            return (c, paras)
        except Exception as e:
            logger.warning(f"[english] 正文抓取失败 {source}: {e}")
            return (c, [])
    with ThreadPoolExecutor(max_workers=4) as ex:
        got = list(ex.map(_one, cands[:MAX_CANDIDATE_FETCH]))
    return [(c, p) for c, p in got if p]


async def _pick_with_llm(cands):
    """让 LLM 从候选中挑晨读+晚读两篇；失败返回 None。"""
    lines = []
    for i, (c, paras) in enumerate(cands):
        source, _r, title, _link, desc, dt = c
        words = sum(len(p.split()) for p in paras)
        lines.append(f"{i}. [{source} | ~{words}词] {title}\n   摘要：{(desc or paras[0])[:200]}")
    prompt = (
        "你是英语精读课的选材编辑。学生是中国大学生（教材 Pathways 4 级，CEFR C1，"
        "理工背景，喜欢科技/科普）。请从下面候选文章里挑出今天的两篇阅读材料：\n"
        "- morning：信息量大的硬核科普或深度报道，知识密度高，约 400-800 词最佳；\n"
        "- evening：风格更轻、故事性/文化性强一点的短文（科研故事、考古、健康生活、"
        "趣味科学都好），约 200-650 词最佳；\n"
        "- **促销、比价、购物资讯、纯产品发布稿一律不要选**（对语言学习没营养）；\n"
        "- 两篇话题必须不同；尽量贴合这些主题（工作与AI/城市/艺术/交通/协作与群体智能/"
        "语言/能源与资源/健康与长寿/真相与AI鉴伪/气候与地球/前沿科学与太空）。\n"
        "只输出一个 JSON 对象，格式：{\"morning\": 序号, \"evening\": 序号}\n\n"
        + "\n".join(lines)
    )
    try:
        raw = await _call_llm(prompt, max_tokens=2048)
        m = re.search(r"\{[^{}]*\}", raw, re.S)
        if m:
            d = json.loads(m.group(0))
            mi, ei = int(d.get("morning", -1)), int(d.get("evening", -1))
            if 0 <= mi < len(cands) and 0 <= ei < len(cands) and mi != ei:
                return mi, ei
    except Exception as e:
        logger.warning(f"[english] LLM 选材失败，改用启发式: {e}")
    return None


def _heuristic_pick(cands, words_lo, words_hi, exclude_idx=()):
    """按窗口词数 + 主题分挑一篇。"""
    best, best_key = None, None
    for i, (c, paras) in enumerate(cands):
        if i in exclude_idx:
            continue
        words = sum(len(p.split()) for p in paras)
        if not (words_lo <= words <= words_hi):
            continue
        source, rank, title, _l, desc, dt = c
        key = (-_theme_score(title + " " + desc), rank, -(words - words_lo))
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


def _article_of(c, paras):
    source, _r, title, link, desc, dt = c
    text = "\n\n".join(paras)
    return {"title": title, "source": source, "link": link, "date": dt, "text": text,
            "words": len(text.split())}


# ---------------- LLM ----------------
async def _call_llm(prompt: str, max_tokens: int = LLM_MAX_TOKENS, want_lesson: bool = False) -> str:
    """调 LLM。课程生成（want_lesson=True）会校验正文（含【且 ≥400 字），
    不达标就按 off → low → 配置值 降档/升档重试（见 LESSON_EFFORT 处的实锤注释）；
    其他用途取第一个非空结果。全部失败返回已拿到的最长文本（可能为空）。"""
    from openai import AsyncOpenAI
    from .llm_fallback import build_client
    client = build_client(AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL))
    base_kwargs = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    # 选题也是机械任务：先 off（快且稳），课程生成再多一档 low 兜底，最后才是配置值
    ladder = [LESSON_EFFORT, "low", _REASONING_EFFORT or None] if want_lesson \
        else [LESSON_EFFORT, _REASONING_EFFORT or None]
    seen, best = set(), ""
    for eff in ladder:
        if eff in seen:
            continue
        seen.add(eff)
        try:
            if eff:
                resp = await client.chat.completions.create(
                    **base_kwargs, extra_body={"reasoning_effort": eff})
            else:
                resp = await client.chat.completions.create(**base_kwargs)
        except Exception as e:
            print(f"[english] LLM 调用失败(effort={eff}): {type(e).__name__}: {e}", flush=True)
            continue
        ch = resp.choices[0]
        text = (ch.message.content or "").strip()
        if len(text) > len(best):
            best = text
        if want_lesson:
            if "【" in text and len(text) >= 400:
                return text
            print(f"[english] 课程文本不达标(effort={eff}: {len(text)}字, finish={ch.finish_reason})，降档重试", flush=True)
        elif text:
            return text
    return best


# ---------------- 课程生成提示词 ----------------
def _lesson_prompt(article: dict, edition: str, morning_vocab) -> str:
    body = article["text"][:7000]
    head = (f"你是面向中国大学生的英语精读课教研老师。学生教材是 National Geographic Learning"
            f"《Pathways》第 4 级（RW4 阅读写作 / LS4 听说），CEFR 参考级别 C1。"
            f"请把下面这篇英文文章做成一份「英语{edition}」材料，帮学生提高阅读能力和语感。\n"
            f"文章标题：{article['title']}（来源：{article['source']}）\n\n"
            "输出严格遵守下面的分节格式（【】标题原样保留，便于排版），不要输出其他内容：\n\n"
            "【导读】\n"
            "用 2-3 句中文介绍这篇文章讲什么、为什么值得读，开头要有钩子。\n\n")
    if edition == "晨读":
        head += (
            "【正文】\n"
            "原样给出英文全文，保留段落，不要改写删节。\n"
            "生词就地标注（重要）：下面【生词表】里会列出的每个生词，在正文【第一次出现】的位置"
            "紧跟一个中文括注，格式如：coronagraph（日冕仪），让读者当场看懂、不用翻到后面查。"
            "括注只写最简短的中文意思（2~8 字），不写音标、词性、英文释义；同一个词只标第一次，"
            "后面再出现不重复标；标注不要改动原句、不要打断句子结构。\n\n"
            "【逐段解读】\n"
            "对正文每个段落依次输出（k 为段号）：\n"
            "▍Para k\n"
            "译文：（该段完整、自然的中文翻译，人名地名首次出现括注英文）\n"
            "解读：（一句话点出这一段在全文中的作用：论点/证据/转折/例证/让步等，帮学生抓篇章结构）\n\n"
            f"【生词表】\n"
            f"挑出学生可能不认识的词：只要 CEFR B2 及以上的词汇、学术词汇表(AWL)词条、"
            f"新闻/专业用语；基础词（a/the/make/take 这类）和专有名词一律不要；"
            f"按文中出现顺序挑 8-{MAX_VOCAB} 个，每行一个，用半角竖线分隔六列：\n"
            "| 单词 | 音标(IPA) | 词性 | 英文释义(简单英文) | 中文释义 | 原文例句(截一小句) |\n\n"
            "【值得记住的表达】\n"
            "从文中挑 4-6 个值得模仿的搭配或句式（地道 collocations / 好用的句型），每行格式：\n"
            "- 表达 —— 中文意思 —— 仿写：一句自创的、贴近大学生活的例句\n\n"
            "【一句话收获】\n"
            "一句中文总结今天最值得带走的东西。\n\n"
            "文章全文如下：\n\n")
    else:
        head += (
            "【正文】\n"
            "原样给出英文全文，保留段落，不要改写删节。\n"
            "生词就地标注（重要）：下面【生词表】里会列出的每个生词，在正文【第一次出现】的位置"
            "紧跟一个中文括注，格式如：coronagraph（日冕仪），让读者当场看懂、不用翻到后面查。"
            "括注只写最简短的中文意思（2~8 字），不写音标、词性、英文释义；同一个词只标第一次，"
            "后面再出现不重复标；标注不要改动原句、不要打断句子结构。\n\n"
            "【参考译文】\n"
            "整篇连贯的中文翻译（可分段，不用逐段标注）。\n\n"
            "【生词表】\n"
            "挑学生可能不认识的词：只要 CEFR B2 及以上、学术词汇表(AWL)、新闻/专业用语；"
            "基础词和专有名词不要；按出现顺序挑 6-10 个，每行一个，半角竖线分隔六列：\n"
            "| 单词 | 音标(IPA) | 词性 | 英文释义(简单英文) | 中文释义 | 原文例句(截一小句) |\n\n")
        if morning_vocab:
            vocab_txt = "；".join(f"{v['word']}（{v['cn']}）" for v in morning_vocab)
            head += (
                f"【晨读回顾】\n"
                f"今天晨读学过这些词：{vocab_txt}。\n"
                f"出 3-5 道自测题帮学生回忆，每行一道，格式：\n"
                f"1. 英文释义或中文提示（首字母提示，如 r_______）\n"
                f"题目要能不看原文回答。\n\n"
                f"【回顾答案】\n"
                f"每行一个：1. 答案单词 —— 一句中文提醒它在大意里的角色\n\n")
        head += "【一句话收获】\n一句中文总结今天最值得带走的东西。\n\n"
    return head + body


def _split_row(line: str):
    """把一行文本解析成表格行（生词表用）；不是表格行返回 None。
    容忍模型输出的格式漂移：首尾竖线可省、加粗星号、全角空格。"""
    t = line.strip().replace("｜", "|")
    if t.count("|") < 4:
        return None
    t = t.strip("|")
    cells = [c.strip().strip("*").strip() for c in t.split("|")]
    if len(cells) < 5:
        return None
    return cells


def _parse_vocab_rows(lesson_text: str):
    """从课程文本里解析生词表行，返回 [{word, cn, ...}]（供晚读回顾用）。"""
    rows = []
    for line in lesson_text.splitlines():
        cells = _split_row(line)
        if not cells or all(set(c) <= set("-: ") for c in cells):
            continue
        first = cells[0]
        if first in ("单词", "word", "Words", "Word"):
            continue
        if re.match(r"^[a-zA-Z' -]+$", first):
            rows.append({"word": first, "cn": cells[4] if len(cells) > 4 else ""})
    return rows[:MAX_VOCAB]


# ---------------- Word 文档 ----------------
_LEADINS = ("译文：", "解读：", "答案：", "仿写：", "参考：")

# 正文里的中文括注（生词就地释义）：渲染成灰色，和英文原文区分开，不打断阅读
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_GLOSS_SPLIT = re.compile(r"(（[^（）]*[\u4e00-\u9fff][^（）]*）)")
_GLOSS_GRAY = None


def _gray():
    """灰色（懒加载 RGBColor，避免模块导入时就依赖 docx.shared）。"""
    global _GLOSS_GRAY
    if _GLOSS_GRAY is None:
        from docx.shared import RGBColor
        _GLOSS_GRAY = RGBColor(0x8C, 0x8C, 0x8C)
    return _GLOSS_GRAY


def _add_body_line(doc, line: str):
    """正文行：含中文的（括注）用灰色渲染，其余保持正文样式。"""
    line = line.replace("**", "")
    p = doc.add_paragraph()
    for seg in _GLOSS_SPLIT.split(line):
        if not seg:
            continue
        run = p.add_run(seg)
        if seg.startswith("（") and _CJK_RE.search(seg):
            run.font.color.rgb = _gray()


def _add_rich_line(doc, line: str, bullet: bool = False):
    m = re.match(r"^([·\-•])\s+(.*)$", line)
    if m:
        line = m.group(2)
        bullet = True
    # 内联 **粗体** 与「前导词」加粗
    if "**" in line or any(line.startswith(p) for p in _LEADINS):
        p = doc.add_paragraph(style="List Bullet" if bullet else None)
        lead = next((x for x in _LEADINS if line.startswith(x)), None)
        idx = 0
        if lead:
            r = p.add_run(lead)
            r.bold = True
            idx = len(lead)
        rest = line[idx:]
        for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", rest)):
            if not part:
                continue
            run = p.add_run(part)
            run.bold = bool(i % 2)
        return
    doc.add_paragraph(line, style="List Bullet" if bullet else None)


def _build_docx(lesson_text: str, edition: str, article: dict) -> str:
    """把课程文本渲染成 Word；解析不了的行按普通段落兜底。"""
    from docx import Document
    os.makedirs(DOC_DIR, exist_ok=True)
    doc = Document()
    today = datetime.now().strftime("%Y-%m-%d")
    doc.add_heading(f"英语{edition} · {today}", 0)
    meta = (f"《{article['title']}》 — {article['source']} · "
            f"约 {article['words']} 词 · 难度参考 Pathways 4 / CEFR C1")
    p = doc.add_paragraph()
    r = p.add_run(meta)
    r.italic = True

    table_buf = []
    # 预扫正文块：有没有就地中文括注（决定正文标题下要不要加图例）
    body_has_gloss = False
    _in_body = False
    for raw in lesson_text.splitlines():
        h = re.match(r"^\s*【(.+?)】", raw)
        if h:
            _in_body = h.group(1).strip() == "正文"
            continue
        if _in_body and _GLOSS_SPLIT.search(raw):
            body_has_gloss = True
            break

    cur_section = ""
    for raw in lesson_text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        row = _split_row(line)                     # 生词表行（宽格式）
        if row:
            table_buf.append(row)
            continue
        if table_buf:
            _flush_table(doc, table_buf)
            table_buf = []
        m = re.match(r"^\s*【(.+?)】\s*(.*)$", line)  # 允许"【生词表】(6-10个)"式尾巴
        if m:
            cur_section = m.group(1).strip()
            doc.add_heading(cur_section, level=1)
            if m.group(2).strip():
                _add_rich_line(doc, m.group(2).strip())
            if cur_section == "正文" and body_has_gloss:
                lp = doc.add_paragraph()
                lr = lp.add_run("（灰色括注＝就地中文释义；生词完整信息见文末生词表）")
                lr.italic = True
                lr.font.color.rgb = _gray()
            continue
        m = re.match(r"^\s*#{1,4}\s*(.+?)\s*$", line)
        if m:
            cur_section = m.group(1).strip()
            doc.add_heading(cur_section, level=1)
            continue
        if line.startswith("▍"):
            doc.add_heading(line.lstrip("▍ ").strip(), level=2)
            continue
        if cur_section == "正文":
            _add_body_line(doc, line)
            continue
        _add_rich_line(doc, line)
    if table_buf:
        _flush_table(doc, table_buf)

    safe = re.sub(r"[\\/:*?\"<>|]", "_", f"英语{edition} · {today}")
    path = os.path.abspath(os.path.join(DOC_DIR, f"{safe}.docx"))
    doc.save(path)
    return path


def _flush_table(doc, rows):
    rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    try:
        table = doc.add_table(rows=len(rows), cols=ncols)
        try:
            table.style = "Table Grid"
        except Exception:
            pass
        for i, row in enumerate(rows):
            for j in range(ncols):
                cell = table.cell(i, j)
                cell.text = row[j] if j < len(row) else ""
                if i == 0:
                    for run in cell.paragraphs[0].runs:
                        run.bold = True
        doc.add_paragraph("")
    except Exception as e:
        logger.warning(f"[english] 生词表渲染失败，降级为文本行: {e}")
        for row in rows:
            doc.add_paragraph(" | ".join(row))


# ---------------- 发送 ----------------
def _targets():
    groups = ENGLISH_GROUPS
    friends = ENGLISH_FRIENDS
    return groups, friends


async def _send(content_notice: str, doc_path):
    groups, friends = _targets()
    if not (groups or friends):
        logger.warning("[english] 未配置 ENGLISH_GROUPS/ENGLISH_FRIENDS，跳过推送")
        return
    name = os.path.basename(doc_path) if doc_path else ""
    for bot in driver.bots.values():
        for gid in groups:
            with contextlib.suppress(Exception):
                await bot.send_msg(message_type="group", group_id=gid, message=content_notice)
                if doc_path:
                    await bot.upload_group_file(group_id=gid, file=doc_path, name=name)
        for uid in friends:
            with contextlib.suppress(Exception):
                await bot.send_msg(message_type="private", user_id=uid, message=content_notice)
                if doc_path:
                    await bot.upload_private_file(user_id=uid, file=doc_path, name=name)


async def _send_text_only(text: str):
    groups, friends = _targets()
    for bot in driver.bots.values():
        for gid in groups:
            with contextlib.suppress(Exception):
                await bot.send_msg(message_type="group", group_id=gid, message=text)
        for uid in friends:
            with contextlib.suppress(Exception):
                await bot.send_msg(message_type="private", user_id=uid, message=text)


# ---------------- 主流程 ----------------
async def _generate_and_send(edition: str, article: dict, morning_vocab=None):
    # want_lesson=True：内部会校验正文并按 off→low→配置值 重试（2026-10-15 晨读空正文实锤）
    lesson = await _call_llm(_lesson_prompt(article, edition, morning_vocab), want_lesson=True)
    if "【" not in lesson or len(lesson) < 400:
        raise RuntimeError("课程文本异常（过短或缺分节）")
    vocab = _parse_vocab_rows(lesson)
    doc_path = await asyncio.to_thread(_build_docx, lesson, edition, article)
    emoji = "☀️" if edition == "晨读" else "🌙"
    tail = "逐段精读+生词表" if edition == "晨读" else "短文+译文+词汇自测"
    notice = f"{emoji} 英语{edition}来啦：《{article['title']}》｜约{article['words']}词 · {tail}"
    await _send(notice, doc_path)
    return vocab, lesson


async def _run_edition(edition: str = "晨读"):
    groups, friends = _targets()
    if not (groups or friends):
        logger.warning("[english] 未配置推送目标，跳过")
        return
    state = await asyncio.to_thread(_load_state)
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        if edition == "晨读":
            state.setdefault("used_urls", [])
            cands = await asyncio.to_thread(_collect_candidates)
            arts = await asyncio.to_thread(_fetch_articles, cands)
            lo, hi = POOL_WORDS
            usable = [(c, p) for c, p in arts
                      if lo <= sum(len(x.split()) for x in p) <= hi
                      and c[3] not in state["used_urls"]]
            if not usable:  # 放宽窗口再试
                usable = [(c, p) for c, p in arts if c[3] not in state["used_urls"]]
            if not usable:
                raise RuntimeError("今天没抓到可用文章")

            def _w(i):
                return sum(len(x.split()) for x in usable[i][1])

            mi = ei = None
            pick = await _pick_with_llm(usable)
            if pick:
                mi, ei = pick
                # LLM 的选择也要过词数窗口，不达标就退回启发式
                if not (MORNING_WORDS[0] <= _w(mi) <= MORNING_WORDS[1]):
                    mi = None
                if ei == mi or not (EVENING_WORDS[0] <= _w(ei) <= EVENING_WORDS[1]):
                    ei = None
            if mi is None:
                mi = _heuristic_pick(usable, *MORNING_WORDS)
                if mi is None:
                    mi = 0
            if ei is None or ei == mi:
                ei = _heuristic_pick(usable, *EVENING_WORDS, exclude_idx=(mi,))
                if ei is None or ei == mi:
                    ei = next((i for i in range(len(usable)) if i != mi), mi)
            morning = _article_of(*usable[mi])
            vocab, _lesson = await _generate_and_send("晨读", morning)
            evening = _article_of(*usable[ei])
            state["date"] = today
            state["evening"] = {k: evening[k] for k in ("title", "source", "link", "words", "text")}
            state["morning_vocab"] = vocab
            state["used_urls"] = list({*state["used_urls"], morning["link"], evening["link"]})
            await asyncio.to_thread(_save_state, state)
        else:
            ev = state.get("evening") if state.get("date") == today else None
            if not ev:
                # 早上的环节没跑成：现选一篇轻量的
                cands = await asyncio.to_thread(_collect_candidates)
                arts = await asyncio.to_thread(_fetch_articles, cands)
                usable = [(c, p) for c, p in arts if c[3] not in state.get("used_urls", [])]
                idx = _heuristic_pick(usable, *EVENING_WORDS) or (0 if usable else None)
                if idx is None:
                    raise RuntimeError("今天没抓到可用文章")
                ev = _article_of(*usable[idx])
            mv = state.get("morning_vocab") if state.get("date") == today else None
            await _generate_and_send("晚读", ev, morning_vocab=mv or None)
            state["date"] = today
            state["used_urls"] = list({*state.get("used_urls", []), ev["link"]})
            await asyncio.to_thread(_save_state, state)
    except Exception as e:
        logger.error(f"[english] {edition}流程失败: {e}")
        emoji = "🌧" if edition == "晨读" else "🌙"
        with contextlib.suppress(Exception):
            await _send_text_only(
                f"{emoji} 今日英语{edition}生成失败了（{str(e)[:80]}）。"
                f"{'今晚晚读见。' if edition == '晨读' else '明早晨读见。'}")


def _register_schedules():
    for idx, (edition, hh, mm) in enumerate(ENGLISH_SCHEDULE):
        job_id = f"english_reading_{edition}_{hh:02d}{mm:02d}_{idx}"

        async def _job(edition: str = edition):
            asyncio.create_task(_run_edition(edition))

        scheduler.add_job(_job, "cron", hour=hh, minute=mm, id=job_id,
                          misfire_grace_time=300, replace_existing=True)
        logger.info(f"已注册英语{edition}定时任务: {hh:02d}:{mm:02d} (id={job_id})")


if not ENGLISH_SCHEDULE:
    logger.warning("ENGLISH_SCHEDULE 为空或全部格式错误，未注册英语晨晚读定时任务")
else:
    _register_schedules()
