# -*- coding: utf-8 -*-
"""10:00 轮（2026-10-26）：图片诚实边界提示词 + QQ 表情 ID 修复 + 多气泡锚点。

覆盖 WP2（VOICE_RULES 第 12 条 / PLANNER_RULES / roles/Anon.md）与
WP3（FACE_ACTIONS smile=14、surprise=0）。全部离线纯断言，不联网、不改运行数据。
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plugins import agent_core, anon_search_policy

ROLE_CARD = Path(__file__).resolve().parents[1] / "roles" / "Anon.md"
AGENT_CORE_SRC = Path(__file__).resolve().parents[1] / "src" / "plugins" / "agent_core.py"


class FaceIdFixTest(unittest.TestCase):
    """WP3：Face ID 与 napcat qface 表一致。"""

    def test_verified_ids(self):
        self.assertEqual(agent_core.FACE_ACTIONS["smile"], 14)
        self.assertEqual(agent_core.FACE_ACTIONS["surprise"], 0)
        self.assertEqual(agent_core.FACE_ACTIONS["shy"], 6)
        self.assertEqual(agent_core.FACE_ACTIONS["heart"], 66)

    def test_other_ids_unchanged(self):
        expected = {
            "angry": 11, "cry": 9, "laugh": 13, "kiss": 66,
            "rose": 63, "hug": 49, "smug": 4, "awkward": 10,
        }
        for name, value in expected.items():
            self.assertEqual(agent_core.FACE_ACTIONS[name], value, msg=name)

    def test_default_send_face_maps_to_smile_14(self):
        act = anon_search_policy.pick_face_action("发个表情")
        self.assertEqual(act, "face:smile")
        self.assertEqual(agent_core.FACE_ACTIONS[act[5:]], 14)

    def test_face_source_comment_present(self):
        source = AGENT_CORE_SRC.read_text(encoding="utf-8")
        self.assertIn("qface-*.js", source)
        self.assertIn("0=/惊讶", source)
        self.assertIn("14=/微笑", source)


class PromptImageHonestyTest(unittest.TestCase):
    """WP2：提示词/角色卡的图片诚实边界锚点。"""

    def test_voice_rules_image_honesty(self):
        rules = agent_core.VOICE_RULES
        self.assertIn("不要假装看见", rules)
        self.assertIn("图这边看不到", rules)
        self.assertIn("再发近点", rules)
        self.assertIn("图太糊", rules)

    def test_voice_rule_12_unique_and_1_to_11_kept(self):
        rules = agent_core.VOICE_RULES
        self.assertEqual(rules.count("\n12. "), 1)
        for number in range(1, 12):
            self.assertIn(f"\n{number}. ", rules)

    def test_planner_rules_image_honesty(self):
        rules = agent_core.PLANNER_RULES
        self.assertIn("看不到图", rules)
        self.assertIn("带[图片]的消息先看[图片内容：...]", rules)
        self.assertIn("没有描述就不要猜图里是什么", rules)

    def test_role_card_speech_rule_added(self):
        text = ROLE_CARD.read_text(encoding="utf-8")
        self.assertIn("带图的问题先按消息里的[图片内容：...]回答；没有[图片内容：...]就说看不到，别猜。", text)

    def test_role_card_ability_rule_added(self):
        text = ROLE_CARD.read_text(encoding="utf-8")
        self.assertIn(
            "能看到群友发的图片并据此回答；读不出来就直说看不到。"
            "仍然不能发图片、表情包、语音、视频或文件。",
            text,
        )
        # 原有的“做不到”边界不能被覆盖
        self.assertIn("做不到：替人发消息", text)


class MultiBubbleAnchorTest(unittest.TestCase):
    """并发回归：另一会话落地的多气泡能力必须保留。"""

    def test_split_bubbles_text_kept(self):
        self.assertTrue(callable(getattr(agent_core, "split_bubbles_text", None)))
        self.assertEqual(agent_core.MAX_CHAT_BUBBLES, 3)
        text = agent_core.split_bubbles_text("一。\n二。\n三。\n四。")
        self.assertEqual(text.splitlines(), ["一。", "二。", "三。"])

    def test_voice_rule_5_multi_bubble_kept(self):
        self.assertIn("1~3 行", agent_core.VOICE_RULES)
        self.assertIn("每行一句一条", agent_core.VOICE_RULES)

    def test_parse_plan_multi_bubble_kept(self):
        plan = agent_core._parse_plan('{"mode":"chat","reply":"一。\\n二。\\n三。\\n四。"}')
        self.assertIsNotNone(plan)
        self.assertEqual(plan["reply"].splitlines(), ["一。", "二。", "三。"])


if __name__ == "__main__":
    unittest.main()
