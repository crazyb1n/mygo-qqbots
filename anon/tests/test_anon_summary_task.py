# -*- coding: utf-8 -*-
"""Offline regression test for the WP4 summary-task reentrancy guard.

Imports anonbot (nonebot is installed in the venv) but never touches the
network or the real anon_memory.json: ``_ask_deepseek`` and ``_save_memory``
are patched, and module state is saved/restored around each test.
"""
import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plugins import anonbot


class RefreshGroupSummaryReentrancyTest(unittest.TestCase):
    GROUP = "test-reentrancy-group"

    def setUp(self):
        self._summaries = dict(anonbot.group_summaries)
        self._raw = {k: list(v) for k, v in anonbot.group_recent_raw.items()}
        self._counts = dict(anonbot.group_msg_count)
        anonbot.group_summaries.clear()
        anonbot.group_recent_raw.clear()
        anonbot.group_msg_count.clear()
        anonbot.summary_locks.clear()
        anonbot.group_summaries[self.GROUP] = "旧摘要"
        anonbot.group_recent_raw[self.GROUP] = [
            (f"群友{i}", f"消息{i}", 1000000 + i) for i in range(3)
        ]

    def tearDown(self):
        anonbot.group_summaries.clear()
        anonbot.group_recent_raw.clear()
        anonbot.group_msg_count.clear()
        anonbot.summary_locks.clear()
        anonbot.group_summaries.update(self._summaries)
        anonbot.group_recent_raw.update(self._raw)
        anonbot.group_msg_count.update(self._counts)

    def test_second_task_waits_and_builds_on_first_result(self):
        calls = []
        release = {}

        async def fake_ask(prompt, *args, **kwargs):
            tag = f"call{len(calls) + 1}"
            calls.append(prompt)
            event = asyncio.Event()
            release[tag] = event
            await event.wait()
            return f"总结{tag}"

        async def scenario():
            with mock.patch.object(anonbot, "_ask_deepseek", fake_ask), \
                    mock.patch.object(anonbot, "_save_memory", lambda: None):
                task1 = asyncio.create_task(anonbot._refresh_group_summary(self.GROUP))
                for _ in range(1000):
                    if calls:
                        break
                    await asyncio.sleep(0)
                self.assertEqual(len(calls), 1)

                # 第一个任务还挂在 LLM 调用里时启动第二个（复现原 bug 的
                # 时序：计数已清零、缓冲继续积累、第二个任务抢跑）。
                task2 = asyncio.create_task(anonbot._refresh_group_summary(self.GROUP))
                for _ in range(50):
                    await asyncio.sleep(0)
                # 每群摘要锁必须挡住第二个任务：它还不能进入 LLM 调用
                self.assertEqual(len(calls), 1)

                # 第一个任务运行期间，新一轮消息继续积累
                buffer = anonbot.group_recent_raw[self.GROUP]
                for i in range(3):
                    buffer.append((f"新人{i}", f"新消息{i}", 2000000 + i))

                release["call1"].set()
                await asyncio.wait_for(task1, 5)
                for _ in range(1000):
                    if len(calls) >= 2:
                        break
                    await asyncio.sleep(0)
                self.assertEqual(len(calls), 2)
                # 关键断言：第二个任务的 prompt 基于第一个任务刚写回的总结，
                # 而不是它启动时读到的旧摘要（旧代码会用旧摘要把新总结覆盖回去）。
                self.assertIn("总结call1", calls[1])
                self.assertIn("新消息0", calls[1])

                release["call2"].set()
                await asyncio.wait_for(task2, 5)
            self.assertEqual(anonbot.group_summaries[self.GROUP], "总结call2")
            # 两批消息分别被 task1/task2 处理完并清出缓冲，无残留也无丢失
            self.assertEqual(anonbot.group_recent_raw[self.GROUP], [])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
