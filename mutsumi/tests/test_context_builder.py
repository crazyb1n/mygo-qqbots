import unittest

from src.plugins.chat_logic import (
    build_context_messages,
    shorten_reply,
    should_trigger,
    sibling_history_key,
)


class ContextBuilderTests(unittest.TestCase):
    def test_sibling_history_key_pairs_personas(self):
        """换人格后要能接上上一轮（2026-10-20：两套历史各记各的导致断片）。"""
        self.assertEqual(sibling_history_key("100000004:mutsumi"), "100000004:mortis")
        self.assertEqual(sibling_history_key("private:123:mortis"), "private:123:mutsumi")
        self.assertIsNone(sibling_history_key("100000004:dialog"))

    def test_trigger_helper_blocks_bots_and_allows_dialogue_turn(self):
        args = dict(enabled=True, is_group=True, user_id="bot", self_id="self",
                    other_bot_qq="bot", astra_bot_qq="astra", group_id="1",
                    allowed_groups={1}, dialogue_active=False, dialogue_turns=0,
                    dialogue_max_turns=2)
        self.assertEqual(should_trigger(**args), (False, False))
        args["dialogue_active"] = True
        self.assertEqual(should_trigger(**args), (True, True))
        args["dialogue_turns"] = 2
        self.assertEqual(should_trigger(**args), (False, False))

    def test_shortening_keeps_small_replies_cheap(self):
        self.assertEqual(shorten_reply("很短", 15), "很短")
        self.assertTrue(shorten_reply("这是一段很长很长很长的话。", 8).endswith("…"))

    def test_current_message_is_last_and_explicitly_anchored(self):
        messages = build_context_messages(
            base_prompt="PERSONA", user="小明", summary="正在聊游戏",
            recent=[("甲", "旧话题"), ("小明", "这把怎么配装")],
            history=[], current_entry="小明：这把怎么配装", context_lines=8, history_rounds=8,
        )
        self.assertEqual(messages[0], {"role": "system", "content": "PERSONA"})
        self.assertIn("甲：旧话题", messages[3]["content"])
        self.assertEqual(messages[-1]["role"], "user")
        self.assertTrue(messages[-1]["content"].startswith("【正在回这条】"))
        self.assertIn("这把怎么配装", messages[-1]["content"])

    def test_recent_context_does_not_repeat_current_message(self):
        messages = build_context_messages(
            base_prompt="PERSONA", user="小明", summary="",
            recent=[("甲", "上一句"), ("小明", "当前")], history=[],
            current_entry="小明：当前", context_lines=8, history_rounds=8,
        )
        background = "\n".join(m["content"] for m in messages if m["role"] == "system")
        self.assertIn("甲：上一句", background)
        self.assertNotIn("小明：当前", background)

    def test_hints_are_optional_and_history_is_copied(self):
        history = [{"role": "assistant", "content": "旧回复"}, {"role": "user", "content": "现在"}]
        messages = build_context_messages(
            base_prompt="P", user="u", summary="", recent=[], history=history[:-1],
            current_entry=history[-1]["content"], context_lines=0, history_rounds=1,
            experience_hint="记忆", sticker_hint="表情",
        )
        self.assertEqual(messages[-1]["content"], "【正在回这条】现在")
        self.assertTrue(any(m["content"] == "记忆" for m in messages))
        messages[-1]["content"] = "changed"
        self.assertEqual(history[-1]["content"], "现在")

    def test_unaddressed_modes_add_guard_block_right_before_message(self):
        """判定块（2026-10-15 少主动接茬 / 2026-10-16 提到她要接）：
        mention/related/proactive 三种关系的块要贴在当前消息前；
        被 @/被引用（默认 at）不加任何块，行为不变。"""
        common = dict(base_prompt="P", user="甲", summary="", recent=[], history=[],
                      current_entry="甲：在聊若叶睦", context_lines=0, history_rounds=8)
        at_msgs = build_context_messages(**common)
        mention_msgs = build_context_messages(**common, addressed="mention")
        related_msgs = build_context_messages(**common, addressed="related")
        proactive_msgs = build_context_messages(**common, addressed="proactive")
        # at：不加块（老行为，消息条数不变）
        self.assertEqual(len(at_msgs), 3)
        for block in ("有人提到你", "聊到你身边的事", "没人叫你"):
            self.assertFalse(any(block in m["content"] for m in at_msgs))
        # mention：块是最后一条 system，紧贴当前消息；要求"要接"而不是"沉默"
        self.assertEqual(mention_msgs[-2]["role"], "system")
        self.assertIn("有人提到你", mention_msgs[-2]["content"])
        self.assertIn("你要接", mention_msgs[-2]["content"])
        self.assertEqual(mention_msgs[-1]["role"], "user")
        # related：聊到她身边的人/原作，也可以接一句
        self.assertIn("聊到你身边的事", related_msgs[-2]["content"])
        # proactive：没人叫她，接得上才接
        self.assertIn("没人叫你", proactive_msgs[-2]["content"])
        # 未知取值：按不加块处理（防止调用方拼错变行为）
        self.assertEqual(len(build_context_messages(**common, addressed="oops")), 3)


if __name__ == "__main__":
    unittest.main()
