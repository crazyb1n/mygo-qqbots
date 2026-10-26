# -*- coding: utf-8 -*-
"""视频关键帧理解：本地视频 → ffmpeg 抽帧 → 云端视觉描述（一句中文）。

背景（2026-10-10 实锤）：群友引用视频问"描述一下他们在干嘛"，她只看到 [视频] 占位，
回"看不到"。NapCat 的 video 段只给 file token / rkey url，没有可读内容。现在的链路：
  1) chat.py 用 NapCat 只读接口 get_file 把视频落到本地（可能是 NapCat 缓存路径，
     也可能是我们下载的临时文件）——本模块不碰 NapCat，只处理本地文件/URL；
  2) 本模块用 ffmpeg 按时长均匀抽 1~2 帧（640px jpg，避开片头片尾）；
  3) 每帧交给 vision.describe_file（与识图同一条云端视觉链路）得到中文描述。

失败一律降级返回空，绝不抛异常打断主流程；抽出的帧与下载的临时文件用后即删。
"""
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from typing import List, Optional, Tuple

from . import vision

# ffmpeg/ffprobe：优先环境变量，其次本项目语音响度同款（chat.py 已在用），最后 PATH
_FFMPEG_CANDIDATES = [
    os.getenv("FFMPEG_BIN", ""),
    r"D:\tools\ffmpeg\ffmpeg-9.0-full_build\bin\ffmpeg.exe",
]
_FFPROBE_CANDIDATES = [
    os.getenv("FFPROBE_BIN", ""),
    r"D:\tools\ffmpeg\ffmpeg-9.0-full_build\bin\ffprobe.exe",
]


def _find(candidates: List[str], name: str) -> Optional[str]:
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return shutil.which(name)


FFMPEG = _find(_FFMPEG_CANDIDATES, "ffmpeg")
FFPROBE = _find(_FFPROBE_CANDIDATES, "ffprobe")

# 帧描述提示词：多帧时说明是同一段视频的第几帧，方便模型串出"在干嘛"
FRAME_PROMPT = (
    "这是同一段短视频里的第 {i}/{n} 帧画面。用一句中文说清楚（≤60字）：画面里有谁/什么、"
    "正在做什么、什么场景或气氛。角色名一句话带过即可：一眼认得出就写"
    "（如\"若叶睦（BanG Dream!）\"），认不出就跳过，不要为此推敲。"
    "有文字就照抄。只输出这一句，不要客套，不要思考过程。"
)

DOWNLOAD_MAX_BYTES = int(os.getenv("VIDEO_MAX_BYTES", str(30 * 1024 * 1024)))  # 30MB 上限
DOWNLOAD_TIMEOUT = float(os.getenv("VIDEO_DOWNLOAD_TIMEOUT", "20"))
FFMPEG_TIMEOUT = float(os.getenv("VIDEO_FFMPEG_TIMEOUT", "25"))

# 2026-10-12: bot 是 pythonw（无控制台）启动，ffmpeg/ffprobe 不隐藏会弹 Terminal 窗
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def probe_duration(path: str) -> Optional[float]:
    """视频时长（秒）。失败返回 None。"""
    if not FFPROBE or not os.path.isfile(path):
        return None
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=15,
            creationflags=_NO_WINDOW).stdout.strip()
        d = float(out)
        return d if d > 0 else None
    except Exception:
        return None


def _timestamps(duration: Optional[float], n: int) -> List[float]:
    """均匀采样时间点：n=2 → 25%/75%；n=3 → 15%/50%/85%（避开片头片尾黑帧）。
    时长未知按 0/1/2 秒试；极短视频只取中点一帧（重复帧没有信息量）。"""
    if duration is None:
        return [float(i) for i in range(n)]
    if duration < 1.0:
        return [duration * 0.5]
    lo, hi = (0.25, 0.75) if n <= 2 else (0.15, 0.85)
    if n == 1:
        return [duration * 0.5]
    return [duration * (lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def extract_frames(path: str, n: int = 2) -> Tuple[List[str], Optional[float]]:
    """抽 n 帧到临时目录。返回 (帧 jpg 路径列表, 时长秒)。失败返回 ([], duration)。
    调用方负责用完删掉帧所在目录（用 cleanup_frames）。"""
    if not FFMPEG or not os.path.isfile(path):
        return [], None
    duration = probe_duration(path)
    workdir = tempfile.mkdtemp(prefix="mutsumi_vid_")
    frames: List[str] = []
    for t in _timestamps(duration, n):
        out = os.path.join(workdir, f"f_{len(frames)}_{int(t * 1000)}.jpg")
        try:
            r = subprocess.run(
                [FFMPEG, "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", path,
                 "-frames:v", "1", "-vf", "scale='min(640,iw)':-2", "-q:v", "4", out],
                capture_output=True, text=True, timeout=FFMPEG_TIMEOUT,
                creationflags=_NO_WINDOW)
            if os.path.isfile(out) and os.path.getsize(out) > 0:
                frames.append(out)
            else:
                # -ss 落在视频尾部之外（时长探测不准）时回退到中点再试一次
                try:
                    subprocess.run(
                        [FFMPEG, "-y", "-v", "error", "-ss", "0", "-i", path,
                         "-frames:v", "1", "-vf", "scale='min(640,iw)':-2", "-q:v", "4", out],
                        capture_output=True, text=True, timeout=FFMPEG_TIMEOUT,
                creationflags=_NO_WINDOW)
                except Exception:
                    pass
                if os.path.isfile(out) and os.path.getsize(out) > 0:
                    frames.append(out)
        except Exception as e:
            print(f"[VIDEO] 抽帧失败 t={t:.2f}s {type(e).__name__}: {e}", flush=True)
    if not frames and workdir:
        shutil.rmtree(workdir, ignore_errors=True)
        return [], duration
    return frames, duration


def describe_frames(frames: List[str], deadline_ts: float,
                    per_call_timeout: float = 20.0,
                    max_tokens: Optional[int] = None) -> Tuple[List[str], int]:
    """逐帧描述；总时间不越过 deadline_ts（每帧调用超时也会收缩到剩余时间）。
    返回 (描述列表, 实际发起的视觉调用次数)——调用次数给上层扣预算用。
    max_tokens 缺省跟随 vision.VISION_MAX_TOKENS（思考型模型预算，别写死小值）。"""
    descs: List[str] = []
    used = 0
    for i, p in enumerate(frames, 1):
        left = deadline_ts - time.time()
        if left < 3:
            break
        used += 1
        d = vision.describe_file(
            p, FRAME_PROMPT.format(i=i, n=len(frames)),
            max_tokens=max_tokens or vision.VISION_MAX_TOKENS, timeout=min(per_call_timeout, left))
        if d:
            descs.append(d)
    return descs, used


def cleanup_frames(frames: List[str]) -> None:
    """删掉抽帧产生的临时目录（整目录删，含 ffmpeg 可能留下的部分帧）。"""
    if not frames:
        return
    d = os.path.dirname(frames[0])
    if d and os.path.basename(d).startswith("mutsumi_vid_"):
        shutil.rmtree(d, ignore_errors=True)


def download_video(url: str) -> str:
    """get_file 没给本地路径时的兜底：从 rkey 直链下载到临时文件。失败返回 ""。
    调用方负责删除返回的文件（用 cleanup_file）。"""
    if not url.startswith("http"):
        return ""
    fd, out = tempfile.mkstemp(prefix="mutsumi_vdl_", suffix=".mp4")
    os.close(fd)
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://multimedia.nt.qq.com.cn/",
        })
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp, open(out, "wb") as f:
            got = 0
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                got += len(chunk)
                if got > DOWNLOAD_MAX_BYTES:
                    raise ValueError("视频超过大小上限")
                f.write(chunk)
        if os.path.getsize(out) > 0:
            return out
    except Exception as e:
        print(f"[VIDEO] 直链下载失败 {type(e).__name__}: {e}", flush=True)
    try:
        os.remove(out)
    except OSError:
        pass
    return ""


def cleanup_file(path: str) -> None:
    """删除我们下载的临时视频（NapCat 缓存路径绝不删，由调用方区分）。"""
    if path and os.path.basename(path).startswith("mutsumi_vdl_"):
        try:
            os.remove(path)
        except OSError:
            pass
