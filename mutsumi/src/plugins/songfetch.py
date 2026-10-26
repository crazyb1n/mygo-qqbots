# -*- coding: utf-8 -*-
"""点歌自动搜歌：群友点了一首曲库没有的歌，现场搜一条合适的音源下载回来。

背景（2026-10-17 用户要求）："群友点歌让他自己搜歌来发吧"——不再只靠固定曲库，
曲库里没有的歌就现场去 B 站搜、下载到曲库，然后按正常点歌流程发出去（下次点同一首秒回）。

选源与下载沿用 E:\\bot\\_fetch_songs.py 的实测经验：
- `bilisearch` 搜索（结果里标题/时长常是 NA，要逐条 resolve）；
- 打分过滤：标题含歌名、时长 60~480s、排除 伴奏/翻唱/谱面/音游/反应视频/排行榜，
  优先 官方MV/完整版/试听/母带；
- **必须带浏览器 UA+Referer**（否则 B 站 412），**必须清代理**（Clash 挂了会连坐）；
- yt-dlp 与 ffmpeg 都是子进程，调用方用 `asyncio.to_thread` 包住，别堵事件循环。
"""
import os
import re
import subprocess
from typing import Optional

YTDLP = os.getenv("YTDLP_PATH", r"D:\sovits\ytdlp-venv\Scripts\yt-dlp.exe")
SONGS_DIR = os.getenv("SONGS_DIR", r"D:\sovits\songs")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
REFERER = "https://www.bilibili.com/"

# 排除（错源）：翻唱/伴奏/教学/谱面/音游/反应/排行榜…
BAD = ("伴奏", "翻唱", "cover", "教学", "教程", "钢琴", "吉他谱", "纯音乐", "片段", "剪辑",
       "混剪", "off vocal", "instrumental", "翻调", "谱面", "音游", "中二节奏", "maimai",
       "MASTER", "Phigros", "节奏大师", "反应", "reaction", "re：", "爆笑", "吐槽", "锐评",
       "解说", "排行", "构筑", "卡牌", "黑白双翼", "推荐", "总谱", "TAB", "原来", "切题")
# 加分（正源）
GOOD = ("完整版", "official", "mv", "试听", "原曲", "full", "高音质", "母带", "hires")


def _clean_env() -> dict:
    env = os.environ.copy()
    for k in list(env):
        if k.lower().endswith("_proxy") or k.upper() in ("ALL_PROXY", "WS_PROXY", "WSS_PROXY"):
            env.pop(k, None)
    return env


def _safe_name(name: str) -> str:
    """歌名 -> 安全文件名（去掉路径分隔符与非法字符）。"""
    s = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", name or "").strip()
    return (s or "未命名")[:60]


def _ytdlp(args, timeout: float):
    common = ["--user-agent", UA, "--referer", REFERER, "--no-warnings"]
    return subprocess.run([YTDLP, *common, *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=_clean_env(), timeout=timeout)


def _search(query: str, timeout: float = 60.0):
    """搜索候选 URL（前 8 条）。"""
    r = _ytdlp([f"bilisearch8:{query}", "--flat-playlist", "--print", "%(url)s"], timeout)
    return [u.strip() for u in (r.stdout or "").splitlines() if u.strip().startswith("http")]


def _resolve(url: str, timeout: float = 60.0):
    """单条解析 -> (标题, 时长秒)。失败 (None, 0)。"""
    r = _ytdlp([url, "--print", "%(title)s\t%(duration)s"], timeout)
    line = next((l for l in (r.stdout or "").splitlines() if "\t" in l), "")
    if not line:
        return None, 0.0
    title, _, dur = line.partition("\t")
    try:
        return title.strip(), float(dur)
    except Exception:
        return title.strip(), 0.0


def pick(query: str, want: str, extra: str = "", budget: float = 90.0):
    """打分选源：标题含歌名（或 extra 里的别名）、时长 60~480s；返回 (title, dur, url) 或 None。"""
    import time
    t0 = time.time()
    tokens = [t.lower() for t in ([want] + (extra.split() if extra else [])) if t]
    best = None
    for url in _search(query)[:8]:
        if time.time() - t0 > budget:
            print("[SONGFETCH] 选源超预算，用已找到的", flush=True)
            break
        title, dur = _resolve(url)
        if not title:
            continue
        low = title.lower()
        if not any(t in low for t in tokens):
            continue
        if not (60 <= dur <= 480):
            continue
        if any(b.lower() in low for b in BAD):
            continue
        score = sum(2 for g in GOOD if g in low) - (2 if dur > 300 else 0)
        if best is None or score > best[0]:
            best = (score, title, dur, url)
    return None if best is None else best[1:]


def fetch(name: str, timeout: float = 240.0, extra_tokens: str = "") -> Optional[str]:
    """现场搜一首歌并下载到曲库。成功返回 mp3 绝对路径；失败/超时返回 None。"""
    want = (name or "").strip()
    if not want:
        return None
    target = os.path.join(SONGS_DIR, f"{_safe_name(want)}.mp3")
    if os.path.exists(target) and os.path.getsize(target) > 50 * 1024:
        return target                       # 已有（并发时别人刚下好）
    os.makedirs(SONGS_DIR, exist_ok=True)
    try:
        hit = pick(f"{want} 完整版", want, extra_tokens, budget=timeout * 0.5)
        if not hit:
            hit = pick(want, want, extra_tokens, budget=timeout * 0.3)   # 退一步：原句搜
        if not hit:
            print(f"[SONGFETCH] {want!r} 没找到合适音源", flush=True)
            return None
        title, dur, url = hit
        print(f"[SONGFETCH] {want!r} <- {title} ({dur:.0f}s) {url}", flush=True)
        r = _ytdlp([url, "-x", "--audio-format", "mp3", "--audio-quality", "4",
                    "-o", os.path.join(SONGS_DIR, f"{_safe_name(want)}.%(ext)s"),
                    "--no-playlist"], timeout * 0.7)
        if os.path.exists(target) and os.path.getsize(target) > 50 * 1024:
            print(f"[SONGFETCH] 下载完成 {target} ({os.path.getsize(target)//1024}KB)", flush=True)
            return target
        print(f"[SONGFETCH] 下载失败: {(r.stderr or '')[-200:]}", flush=True)
        return None
    except Exception as e:
        print(f"[SONGFETCH] 异常 {type(e).__name__}: {e}", flush=True)
        return None


if __name__ == "__main__":       # 手测：python songfetch.py 学猫叫
    import sys
    print(fetch(sys.argv[1] if len(sys.argv) > 1 else "学猫叫"))
