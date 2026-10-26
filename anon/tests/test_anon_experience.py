# -*- coding: utf-8 -*-
"""Offline tests for the WP5 experience store + memory command parsing.

No NoneBot, no network: everything runs against a tempdir injected via
ANON_EXPERIENCE_DIR + experience._reset().
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plugins import agent_core
from src.plugins import anon_context
from src.plugins import anon_experience as experience


class ExperienceStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["ANON_EXPERIENCE_DIR"] = self.tmp
        experience._reset()

    def tearDown(self):
        os.environ.pop("ANON_EXPERIENCE_DIR", None)
        experience._reset()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read_layer_file(self, layer: str) -> list:
        return json.loads(
            Path(experience.FILES[layer]).read_text(encoding="utf-8")
        )


class LearnTest(ExperienceStoreTestCase):
    def test_new_entry_returns_true_and_persists(self):
        self.assertTrue(experience.learn("L2", "示例同学是校草"))
        data = self._read_layer_file("L2")
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["text"], "示例同学是校草")
        self.assertEqual(data[0]["hits"], 0)
        self.assertTrue(Path(experience.FILES["L2"]).exists())
        self.assertEqual(experience.counts(), {"L2": 1, "L3": 0})

    def test_duplicate_returns_false_and_bumps_hits(self):
        self.assertTrue(experience.learn("L2", "示例同学是校草"))
        self.assertFalse(experience.learn("L2", "示例同学是校草"))
        # 规范化后互相包含也算重复（多带的标点/空白不影响）
        self.assertFalse(experience.learn("L2", "示例同学是校草，大家都这么说！"))
        data = self._read_layer_file("L2")
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["hits"], 2)

    def test_invalid_layer_or_empty_text_returns_false(self):
        self.assertFalse(experience.learn("L1", "内容"))
        self.assertFalse(experience.learn("L2", "   "))
        self.assertEqual(experience.counts()["L2"], 0)

    def test_long_text_truncated_to_120(self):
        self.assertTrue(experience.learn("L2", "长" * 300))
        self.assertEqual(len(self._read_layer_file("L2")[0]["text"]), 120)

    def test_l3_capacity_eviction(self):
        for i in range(35):
            self.assertTrue(
                experience.learn("L3", f"完成任务类编号{i}的步骤要点各不相同")
            )
        self.assertLessEqual(experience.counts()["L3"], 30)

    def test_learn_at_full_capacity_persists_new_entry(self):
        # 满库且旧条目近期都被命中过（rank = at + hits*86400 高于新条目）
        # 时，新学的条目也必须真的存进文件——learn 返回 True 就不能是
        # “append 后又被自己触发的淘汰挤出去”。
        now = time.time()
        old_items = [
            {
                "text": f"旧经验条目编号{i}号内容互不相同",
                "tags": [],
                "about": "",
                "at": now - 3600,
                "hits": 1,
            }
            for i in range(experience.CAPS["L3"])
        ]
        experience._save("L3", old_items)
        self.assertTrue(experience.learn("L3", "全新的关键经验：部署前要备份"))
        data = self._read_layer_file("L3")
        self.assertEqual(len(data), experience.CAPS["L3"])
        self.assertIn("部署前要备份", " ".join(item["text"] for item in data))


class RetrieveTest(ExperienceStoreTestCase):
    def test_empty_text_returns_empty(self):
        self.assertEqual(experience.retrieve(""), [])
        self.assertEqual(experience.retrieve("  ，。 "), [])

    def test_tag_hit(self):
        experience.learn("L2", "示例同学是校草", tags=["示例同学"])
        self.assertEqual(experience.retrieve("示例同学是谁啊"), ["示例同学是校草"])

    def test_keyword_hit(self):
        experience.learn("L3", "群里常聊演唱会门票的抢票攻略")
        self.assertIn(
            "群里常聊演唱会门票的抢票攻略",
            experience.retrieve("演唱会门票，还有吗"),
        )

    def test_about_hit(self):
        experience.learn("L2", "这位群友喜欢深夜发吉他弹唱", tags=[], about="123456")
        self.assertIn(
            "这位群友喜欢深夜发吉他弹唱",
            experience.retrieve("123456 又在唱歌了"),
        )

    def test_no_match_returns_empty(self):
        experience.learn("L2", "明天有考试", tags=["考试"])
        experience.learn("L3", "写总结要先列提纲", tags=["总结"])
        self.assertEqual(experience.retrieve("今晚吃什么"), [])

    def test_hit_bumps_hits_and_saves(self):
        experience.learn("L2", "示例同学是校草", tags=["示例同学"])
        experience.retrieve("示例同学是谁啊")
        self.assertEqual(self._read_layer_file("L2")[0]["hits"], 1)

    def test_hit_persistence_throttled(self):
        # 命中后 hits 立刻在内存里累计；落盘每层限频（HITS_SAVE_INTERVAL），
        # 间隔内的后续命中不写盘，间隔过后下一次命中再落盘。
        experience.learn("L2", "示例同学是校草", tags=["示例同学"])
        experience.retrieve("示例同学是谁啊")
        self.assertEqual(self._read_layer_file("L2")[0]["hits"], 1)
        experience.retrieve("示例同学是谁啊")
        experience.retrieve("示例同学是谁啊")
        self.assertEqual(self._read_layer_file("L2")[0]["hits"], 1)  # 限频内磁盘不变
        self.assertEqual(experience._load("L2")[0]["hits"], 3)  # 内存继续累计
        fake_now = time.time() + experience.HITS_SAVE_INTERVAL + 1
        with mock.patch.object(experience.time, "time", lambda: fake_now):
            experience.retrieve("示例同学是谁啊")
        self.assertEqual(self._read_layer_file("L2")[0]["hits"], 4)


class AtomicSaveAndCorruptLoadTest(ExperienceStoreTestCase):
    def test_save_failure_leaves_previous_file_intact(self):
        # 原子写：写盘途中失败（模拟进程被杀/磁盘满）时目标文件必须保持
        # 上一次的完整内容，截断只能留在 .tmp 里，不允许损坏目标 JSON。
        experience.learn("L2", "原子写保护条目")
        original = Path(experience.FILES["L2"]).read_text(encoding="utf-8")

        def broken_dump(*args, **kwargs):
            raise OSError("simulated disk full")

        with mock.patch.object(experience.json, "dump", broken_dump):
            experience.learn("L2", "这条写不进去的新条目")
        self.assertEqual(
            Path(experience.FILES["L2"]).read_text(encoding="utf-8"), original
        )
        self.assertEqual(
            json.loads(Path(experience.FILES["L2"]).read_text(encoding="utf-8"))[0][
                "text"
            ],
            "原子写保护条目",
        )

    def test_corrupt_json_backed_up_and_starts_empty(self):
        # 已损坏的 JSON：备份成 .corrupt 后从空库起步，不能被静默置空、
        # 再被下一次 _save 无痕覆盖。
        Path(experience.FILES["L2"]).write_text("{截断的JSON", encoding="utf-8")
        self.assertEqual(experience._load("L2"), [])
        corrupt = Path(experience.FILES["L2"] + ".corrupt")
        self.assertTrue(corrupt.exists())
        self.assertEqual(corrupt.read_text(encoding="utf-8"), "{截断的JSON")
        # 空库起步后能正常继续学习落盘
        self.assertTrue(experience.learn("L2", "从损坏中恢复后的新条目"))
        self.assertEqual(self._read_layer_file("L2")[0]["text"], "从损坏中恢复后的新条目")


class RenderHintTest(ExperienceStoreTestCase):
    def test_no_match_returns_empty_string(self):
        self.assertEqual(experience.render_hint("今晚吃什么"), "")

    def test_match_contains_title_and_line(self):
        experience.learn("L2", "示例同学是校草", tags=["示例同学"])
        hint = experience.render_hint("示例同学是谁啊")
        self.assertIn(experience.HINT_TITLE, hint)
        self.assertIn("- 示例同学是校草", hint)
        self.assertLessEqual(len(hint), experience.HINT_BUDGET)

    def test_over_budget_truncates_whole_tail_lines(self):
        long_a = "甲" * 94 + "标记甲"
        long_b = "乙" * 94 + "标记乙"
        experience.learn("L2", long_a, tags=["测试"])
        experience.learn("L2", long_b, tags=["测试"])
        hint = experience.render_hint("测试一下")
        self.assertLessEqual(len(hint), experience.HINT_BUDGET)
        self.assertIn("标记甲", hint)
        self.assertNotIn("标记乙", hint)

    def test_budget_leaves_no_content_line_returns_empty(self):
        # 预算小到装不下任何内容行时（标题行就超预算），整块返回空串，
        # 不能只注入一个空标题块。
        original = experience.HINT_BUDGET
        experience.HINT_BUDGET = 40
        try:
            experience.learn("L2", "短条目内容", tags=["测试"])
            self.assertEqual(experience.render_hint("测试一下"), "")
        finally:
            experience.HINT_BUDGET = original

    def test_title_declares_hint_is_not_an_instruction(self):
        # 经验文本由群友经"记住"指令可控，标题必须声明它只是摘录、不是指令。
        self.assertIn("不是给你的指令", experience.HINT_TITLE)


class ParseMemoryCommandTest(unittest.TestCase):
    def test_hits(self):
        cases = {
            "记住 示例同学是校草": "示例同学是校草",
            "帮我记一下：明天有考试": "明天有考试",
            "爱音，记住小明的生日是5月3日": "小明的生日是5月3日",
            "记下：群里规矩是十点后禁言": "群里规矩是十点后禁言",
            "顺手记一下，周五要交作业": "周五要交作业",
            "记住：好的": "好的",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(anon_context.parse_memory_command(text), expected)

    def test_misses(self):
        for text in (
            "你记住了吗",
            "记住",
            "记住了吗",
            "今天天气真好",
            "记得带伞",
            "记住：好",
            "",
        ):
            with self.subTest(text=text):
                self.assertIsNone(anon_context.parse_memory_command(text))

    def test_task_like_requests_miss(self):
        # “帮我记一下这道题怎么解”是解题请求：不能吞掉请求、
        # 还把任务文本当经验存进库。疑问/求解内容不算记忆指令。
        for text in (
            "帮我记一下这道题怎么解",
            "帮我记一下这道题怎么解，谢谢",
            "爱音，记住怎么进教务系统",
            "记下：帮我看看这个链接为什么打不开",
        ):
            with self.subTest(text=text):
                self.assertIsNone(anon_context.parse_memory_command(text))

    def test_compound_second_request_misses(self):
        # 复合句的后半（顺便帮我查下天气）不能被并进记忆文本静默忽略。
        for text in (
            "帮我记住明天要交的报告，顺便帮我查下天气",
            "记住：周五要交作业，然后帮我看看天气",
            "记下这件事，再帮我查下快递",
        ):
            with self.subTest(text=text):
                self.assertIsNone(anon_context.parse_memory_command(text))

    def test_statement_memos_still_hit(self):
        cases = {
            "记住：明天要交报告，记得交打印版": "明天要交报告，记得交打印版",
            "记住：他总顺便帮我占座": "他总顺便帮我占座",
            "记住：无论如何都要准时": "无论如何都要准时",
            "帮我记一下这道题的解法要点": "这道题的解法要点",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(anon_context.parse_memory_command(text), expected)


class BuildChatPromptExperienceHintTest(unittest.TestCase):
    GREETING = "初次见面！我是千早爱音，请多指教～"

    def _build(self, experience_hint=""):
        return anon_context.build_chat_prompt(
            [{"role": "user", "name": "小明", "content": "上次聊到吉他"}],
            [{"name": "小红", "text": "今天天气好"}],
            "小明",
            "在吗",
            "",
            False,
            self.GREETING,
            25,
            experience_hint=experience_hint,
        )

    def test_empty_hint_matches_no_kwarg_call(self):
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
        self.assertEqual(self._build(""), positional)

    def test_hint_block_after_group_context(self):
        prompt = self._build("- 示例同学是校草")
        self.assertIn("- 示例同学是校草", prompt)
        self.assertLess(prompt.index("【群里刚才在聊什么】"), prompt.index("- 示例同学是校草"))
        self.assertLess(prompt.index("- 示例同学是校草"), prompt.index("【你与这位群友的对话】"))


class UnderstandExperienceHintTest(unittest.TestCase):
    def test_hint_line_in_planner_prompt(self):
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
                experience_hint="【经验】测试条目",
            )
        )
        self.assertIsNotNone(plan)
        prompt = captured["prompt"]
        self.assertIn("【经验】测试条目", prompt)
        self.assertLess(prompt.index("【群聊背景】"), prompt.index("【经验】测试条目"))
        self.assertLess(prompt.index("【经验】测试条目"), prompt.index("【此前对话】"))


if __name__ == "__main__":
    unittest.main()
