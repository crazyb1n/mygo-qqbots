"""薇欧拉 QQ 群聊机器人入口。

启动前请先配置 .env（含 DEEPSEEK_API_KEY）。
需搭配独立的 NapCat 实例（端口 3002）连接个人 QQ 号 200000002。
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

# 强制 UTF-8 输出：cmd/GBK 环境下 print emoji 会 UnicodeEncodeError 导致回复崩溃
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv(Path(__file__).parent / ".env")

import nonebot
from nonebot.adapters.onebot.v11 import Adapter


def main() -> None:
    nonebot.init()
    driver = nonebot.get_driver()
    driver.register_adapter(Adapter)
    # 绝对路径加载插件，不依赖工作目录（从任意目录启动都行）
    nonebot.load_plugins(str(Path(__file__).parent / "src" / "plugins"))
    nonebot.run()


if __name__ == "__main__":
    main()