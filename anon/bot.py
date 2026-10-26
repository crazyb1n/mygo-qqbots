"""QQBot Chihaya Anon entry point."""
import sys
from pathlib import Path

from dotenv import load_dotenv

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

import nonebot
from nonebot.adapters.onebot.v11 import Adapter


def main() -> None:
    nonebot.init()
    nonebot.get_driver().register_adapter(Adapter)
    nonebot.load_plugins(str(ROOT / "src" / "plugins"))
    nonebot.run()


if __name__ == "__main__":
    main()
