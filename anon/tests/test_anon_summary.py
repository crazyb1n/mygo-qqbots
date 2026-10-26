"""Offline tests for the WP4 rolling group summary — no NoneBot, no network."""
import asyncio
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plugins import agent_core
from src.plugins import anon_context


class FormatRecentLineTest(unittest.TestCase):
    def test_with_timestamp(self):
        ts = time.mktime(time.strptime("2026-10-24 09:05", "%Y-%m-%d %H:%M"))
        self.assertEqual(
            anon_context.format_recent_line(("小明", "早呀", ts)),
            "[09:05] 小明：早呀",
        )

    def test_without_timestamp(self):
        self.assertEqual(anon_context.format_recent_line(("小明", "早呀")), "小明：早呀")

    def test_non_positive_or_bad_timestamp_falls_back(self):
        self.assertEqual(anon_context.format_recent_line(("小明", "早呀", 0)), "小明：早呀")
        self.assertEqual(anon_context.format_recent_line(("小明", "早呀", None)), "小明：早呀")
        self.assertEqual(anon_context.format_recent_line(("小明", "早呀", "早")), "小明：早呀")


class BuildSummaryPromptTest(unittest.TestCase):
    def test_instructions_and_new_messages(self):
        prompt = anon_context.build_summary_prompt("", ["小明：在吗", "小红：不在"])
        self.assertIn("【新消息】", prompt)
        self.assertIn("小明：在吗", prompt)
        self.assertIn("小红：不在", prompt)
        for keyword in (
            "QQ 群", "250", "话题", "人名", "外号", "梗", "只输出总结正文",
        ):
            with self.subTest(keyword=keyword):
                self.assertIn(keyword, prompt)
        self.assertNotIn("【旧摘要】", prompt)

    def test_new_messages_before_old_summary(self):
        prompt = anon_context.build_summary_prompt("之前在聊演唱会门票", ["小明：继续"])
        self.assertIn("【旧摘要】", prompt)
        self.assertIn("之前在聊演唱会门票", prompt)
        self.assertIn("【新消息】", prompt)
        self.assertIn("小明：继续", prompt)
        self.assertIn("不要照抄", prompt)
        self.assertLess(prompt.index("【新消息】"), prompt.index("【旧摘要】"))


class BuildChatPromptSummaryTest(unittest.TestCase):
    GREETING = "初次见面！我是千早爱音，请多指教～"

    def _build(self, summary=""):
        return anon_context.build_chat_prompt(
            [{"role": "user", "name": "小明", "content": "上次聊到吉他"}],
            [{"name": "小红", "text": "今天天气好"}],
            "小明",
            "在吗",
            "",
            False,
            self.GREETING,
            25,
            summary=summary,
        )

    def test_summary_line_injected_before_group_context(self):
        prompt = self._build(summary="群里在聊演唱会门票")
        self.assertIn("【群聊背景】群里在聊演唱会门票（背景仅供参考，不要复述）", prompt)
        self.assertIn("【群里刚才在聊什么】", prompt)
        self.assertLess(prompt.index("【群聊背景】"), prompt.index("【群里刚才在聊什么】"))

    def test_empty_summary_matches_no_summary_call(self):
        with_empty = self._build(summary="")
        positional = anon_context.build_chat_prompt(
            [{"role": "user", "name": "小明", "content": "上次聊到吉他"}],
            [{"name": "小红", "text": "今天天气好"}],
            "小明",
            "在吗",
            "",
            False,
            self.GREETING,
            25,
        )
        self.assertEqual(with_empty, positional)
        self.assertNotIn("【群聊背景】", with_empty)


class UnderstandSummaryInjectionTest(unittest.TestCase):
    def test_summary_line_in_planner_prompt(self):
        captured = {}

        async def fake_ask(prompt: str, max_chars: int) -> str:
            captured["prompt"] = prompt
            captured["max_chars"] = max_chars
            return '{"mode":"chat","reply":"嗯","agent":"none","steps":[],"complex":false}'

        plan = asyncio.run(
            agent_core.understand(
                "在吗",
                "小明",
                [],
                [],
                fake_ask,
                summary="测试摘要内容",
            )
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan["mode"], "chat")
        prompt = captured["prompt"]
        self.assertIn("【此前群聊摘要】测试摘要内容", prompt)
        self.assertIn("（背景仅供参考，不要复述）", prompt)
        self.assertLess(prompt.index("【此前群聊摘要】"), prompt.index("【群聊背景】"))

    def test_empty_kwargs_prompt_has_no_injection_blocks(self):
        # 零影响：summary/experience_hint 为空字符串时，planner prompt 与
        # 完全不传这两个参数一致，且不出现任何注入块字样。
        def make_fake(container):
            async def fake_ask(prompt: str, max_chars: int) -> str:
                container.append(prompt)
                return '{"mode":"chat","reply":"嗯","agent":"none","steps":[],"complex":false}'
            return fake_ask

        omitted = []
        plan_a = asyncio.run(agent_core.understand("在吗", "小明", [], [], make_fake(omitted)))
        empty = []
        plan_b = asyncio.run(
            agent_core.understand(
                "在吗", "小明", [], [], make_fake(empty), summary="", experience_hint=""
            )
        )
        self.assertIsNotNone(plan_a)
        self.assertIsNotNone(plan_b)
        self.assertEqual(omitted[0], empty[0])
        self.assertNotIn("【此前群聊摘要】", empty[0])
        self.assertNotIn("你记得的关于这个群的事", empty[0])


if __name__ == "__main__":
    unittest.main()
