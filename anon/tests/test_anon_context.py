"""Offline tests for src/plugins/anon_context.py — no NoneBot, no network."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plugins import anon_context


class IsNameMentionTest(unittest.TestCase):
    def test_hits(self):
        for text in ("爱音在吗", "千早爱音", "anon 出来"):
            with self.subTest(text=text):
                self.assertTrue(anon_context.is_name_mention(text))

    def test_misses(self):
        for text in ("Astra 在吗", "今天天气不错"):
            with self.subTest(text=text):
                self.assertFalse(anon_context.is_name_mention(text))


class IsHelpRequestTest(unittest.TestCase):
    def test_exact_keyword_hits(self):
        for text in ("帮助", " 使用说明 ", "功能", "怎么用", "你会什么"):
            with self.subTest(text=text):
                self.assertTrue(anon_context.is_help_request(text))

    def test_longer_sentence_misses(self):
        self.assertFalse(anon_context.is_help_request("帮助我写作业"))


class CleanAnswerTest(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(anon_context.clean_answer("  好呀好呀  ", 25, 25), "好呀好呀")

    def test_task_length_branch_ends_with_ellipsis(self):
        # max_chars=30 > hard_limit=25 -> 任务长度分支：硬截 + …
        result = anon_context.clean_answer("一" * 60, 30, 25)
        self.assertEqual(result, "一" * 29 + "…")
        self.assertTrue(result.endswith("…"))

    def test_punctuation_boundary_branch(self):
        # max_chars == hard_limit 且 "。" 位于过半位置 -> 在标点边界收尾
        text = "一二三四五六七八九十甲乙丙。" + "丁" * 40
        result = anon_context.clean_answer(text, 25, 25)
        self.assertEqual(result, "一二三四五六七八九十甲乙丙。")

    def test_hard_truncation_branch_without_punctuation(self):
        # max_chars == hard_limit 且前半段无标点 -> 硬截 + …（闲聊长回复常见形态）
        result = anon_context.clean_answer("今" * 40, 25, 25)
        self.assertEqual(result, "今" * 24 + "…")
        self.assertEqual(len(result), 25)

    def test_leak_raises_value_error(self):
        for leaked in ("analysis: hidden", "Assistant：你好", "tool: call"):
            with self.subTest(leaked=leaked):
                with self.assertRaises(ValueError):
                    anon_context.clean_answer(leaked, 25, 25)

    def test_strips_assistant_prefix(self):
        self.assertEqual(anon_context.clean_answer("爱音：你好呀", 25, 25), "你好呀")
        self.assertEqual(anon_context.clean_answer("Anon: 早", 25, 25), "早")

    def test_strips_code_fence_and_markdown_emphasis(self):
        self.assertEqual(anon_context.clean_answer("```\n你好\n```", 25, 25), "你好")
        self.assertEqual(anon_context.clean_answer("**加粗**__下划线__", 25, 25), "加粗下划线")


class FormatGroupContextTest(unittest.TestCase):
    def test_limit_and_keep(self):
        items = [{"name": f"群友{i}", "text": str(i) * 300} for i in range(6)]
        lines = anon_context.format_group_context(items, limit=4, keep=10)
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], "群友2：" + "2" * 10)
        self.assertEqual(lines[-1], "群友5：" + "5" * 10)

    def test_empty(self):
        self.assertEqual(anon_context.format_group_context([], limit=8, keep=160), [])


class FormatHistoryTest(unittest.TestCase):
    def test_labels(self):
        history = [
            {"role": "user", "name": "小明", "content": "在吗"},
            {"role": "assistant", "content": "在的在的"},
            {"role": "user", "content": "嗯嗯"},  # 无 name -> 默认"群友"
        ]
        self.assertEqual(
            anon_context.format_history(history, rounds=6, keep=180),
            ["小明：在吗", "爱音：在的在的", "群友：嗯嗯"],
        )

    def test_rounds_truncation_takes_last(self):
        history = [
            {"role": "user", "name": f"群友{i}", "content": str(i)} for i in range(8)
        ]
        lines = anon_context.format_history(history, rounds=3, keep=180)
        self.assertEqual(lines, ["群友5：5", "群友6：6", "群友7：7"])

    def test_keep_truncates_content(self):
        history = [{"role": "assistant", "content": "一二三四五"}]
        self.assertEqual(
            anon_context.format_history(history, rounds=6, keep=3), ["爱音：一二三"]
        )

    def test_empty_history_returns_empty(self):
        self.assertEqual(anon_context.format_history([], rounds=6, keep=180), [])


class BuildChatPromptTest(unittest.TestCase):
    GREETING = "初次见面！我是千早爱音，请多指教～"

    def _build(self, first_message=False):
        return anon_context.build_chat_prompt(
            [{"role": "user", "name": "小明", "content": "上次聊到吉他"}],
            [{"name": "小红", "text": "今天天气好"}],
            "小明",
            "在吗",
            "",
            first_message,
            self.GREETING,
            25,
        )

    def test_no_role_card_block(self):
        # WP2 前缀缓存改造：角色卡挪到固定 system 消息，user 提示词不再包含角色卡块。
        prompt = self._build()
        self.assertNotIn("【千早爱音角色卡】", prompt)
        self.assertNotIn("你是爱音的角色卡内容。", prompt)

    def test_contains_current_target(self):
        self.assertIn("【正在回这条】小明：在吗", self._build())

    def test_no_fixed_greeting_even_first_message(self):
        # 2026-10-25 用户指令：取消首见固定开场白，任何情况下提示词都不得携带。
        first = self._build(first_message=True)
        self.assertNotIn(self.GREETING, first)
        self.assertNotIn("这是你第一次认出这位群友", first)
        self.assertNotIn(self.GREETING, self._build())

    def test_contains_group_context_line(self):
        prompt = self._build()
        self.assertIn("【群里刚才在聊什么】", prompt)
        self.assertIn("小红：今天天气好", prompt)

    def test_contains_history_line(self):
        self.assertIn("小明：上次聊到吉他", self._build())

    def test_empty_group_context_placeholder(self):
        prompt = anon_context.build_chat_prompt(
            [],
            [],
            "小明",
            "在吗",
            "",
            False,
            self.GREETING,
            25,
        )
        self.assertIn("（没有更多上下文）", prompt)


class BuildSummaryPromptTest(unittest.TestCase):
    """WP3（2026-10-26）：摘要提示词改为【新消息】在前、【旧摘要】只留话题标题与状态。"""

    OLD = "演唱会门票：已经买好，等开票；卡组讨论已结束。"
    LINES = ["小明：在吗", "小红：不在"]

    def test_new_messages_before_old_summary(self):
        prompt = anon_context.build_summary_prompt(self.OLD, self.LINES)
        self.assertIn("【新消息】", prompt)
        self.assertIn("【旧摘要】", prompt)
        self.assertLess(prompt.index("【新消息】"), prompt.index("【旧摘要】"))

    def test_old_summary_content_kept_after_new(self):
        prompt = anon_context.build_summary_prompt(self.OLD, self.LINES)
        self.assertIn(self.OLD, prompt)
        self.assertGreater(prompt.index(self.OLD), prompt.index("【新消息】"))
        # 指令段不得再用方括号标题，否则顺序断言会命中指令而不是真正的块头
        self.assertEqual(prompt.count("【旧摘要】"), 1)
        self.assertEqual(prompt.count("【新消息】"), 1)

    def test_empty_old_summary_has_no_old_block(self):
        prompt = anon_context.build_summary_prompt("", self.LINES)
        self.assertNotIn("【旧摘要】", prompt)
        self.assertNotIn("之前的总结", prompt)
        self.assertIn("【新消息】", prompt)
        for line in self.LINES:
            with self.subTest(line=line):
                self.assertIn(line, prompt)

    def test_short_single_paragraph_instruction(self):
        prompt = anon_context.build_summary_prompt("", self.LINES)
        for anchor in ("250", "一段", "不要分点", "只输出总结正文"):
            with self.subTest(anchor=anchor):
                self.assertIn(anchor, prompt)

    def test_old_summary_only_topics_not_copy(self):
        prompt = anon_context.build_summary_prompt(self.OLD, self.LINES)
        for anchor in (
            "话题标题",
            "状态",
            "只保留仍然有效的话题",
            "进行中/已结束",
            "不要照抄旧细节",
            "已经结束",
            "直接删掉",
        ):
            with self.subTest(anchor=anchor):
                self.assertIn(anchor, prompt)

    def test_new_message_lines_verbatim(self):
        prompt = anon_context.build_summary_prompt("", self.LINES)
        self.assertIn("小明：在吗\n小红：不在", prompt)

    def test_no_lines_still_well_formed(self):
        prompt = anon_context.build_summary_prompt("", [])
        self.assertIn("【新消息】", prompt)
        self.assertNotIn("【旧摘要】", prompt)
        self.assertIn("250", prompt)

    def test_old_summary_without_lines_keeps_order(self):
        prompt = anon_context.build_summary_prompt(self.OLD, [])
        self.assertLess(prompt.index("【新消息】"), prompt.index("【旧摘要】"))
        self.assertIn(self.OLD, prompt)


class IsStaleTest(unittest.TestCase):
    def test_boundaries(self):
        self.assertFalse(anon_context.is_stale(100, 100, 0))
        self.assertFalse(anon_context.is_stale(100, 100, -1))
        self.assertFalse(anon_context.is_stale(100, 160, 60))
        self.assertTrue(anon_context.is_stale(100, 160.01, 60))

    def test_bad_values_do_not_raise(self):
        self.assertFalse(anon_context.is_stale(None, 100, 60))
        self.assertFalse(anon_context.is_stale(100, None, 60))


if __name__ == "__main__":
    unittest.main()
