import unittest

from src.plugins.chat_logic import is_subjective_ask, mortis_intent, pick_persona


class SubjectiveAskTests(unittest.TestCase):
    """主观问题识别（2026-10-16：主观题要用人设语气总结，不当检索题）。"""

    def test_opinion_and_comparison_questions_hit(self):
        for raw in ("若叶睦你觉得哪个吉他好",
                    "你们觉得这游戏值不值得买",
                    "Falcons 对 Spirit 谁更厉害",
                    "这两家店哪家好吃吗",
                    "推荐哪个",
                    "我该不该去",
                    "这歌怎么样",
                    "新版本如何"):
            self.assertTrue(is_subjective_ask(raw), raw)

    def test_factual_questions_do_not_hit(self):
        for raw in ("明天天气多少度",
                    "这个bug怎么修",
                    "几点开始",
                    "帮我总结一下链表知识点",
                    "谁在群里",
                    "打开一下这个文件"):
            self.assertFalse(is_subjective_ask(raw), raw)


class MortisIntentTests(unittest.TestCase):
    """召换/退场意图（2026-10-16 用户要求"要有召唤莫提斯的方式"）。"""

    def test_summon_phrases(self):
        for raw in ("墨提斯出来", "召唤墨提斯", "让墨提斯来", "mortis 上线", "叫莫提斯说两句"):
            self.assertEqual(mortis_intent(raw), "summon", raw)

    def test_long_chatter_with_verb_far_from_name_is_not_summon(self):
        # 长句闲聊里"说说/来"离名字很远，不该判成召唤（近邻规则）
        self.assertIsNone(mortis_intent("墨提斯和若叶睦是什么关系来着大家说说看", addressed=False))
        self.assertIsNone(mortis_intent("你们说墨提斯那次演出是不是很帅啊我觉得还行", addressed=False))

    def test_short_addressed_name_is_summon(self):
        # "被叫到 + 只说了名字" 也算召唤（最自然的叫法）
        self.assertEqual(mortis_intent("墨提斯", addressed=True), "summon")
        self.assertEqual(mortis_intent("@某个bot 墨提斯？", addressed=True), "summon")
        self.assertIsNone(mortis_intent("墨提斯", addressed=False))

    def test_dismiss_beats_summon_words(self):
        # 带她名字的退场指令：即使句子里有"让/来"这类召唤词，也按退场处理
        for raw in ("墨提斯退下", "墨提斯回去吧", "墨提斯别闹了", "让墨提斯先歇着"):
            self.assertEqual(mortis_intent(raw), "dismiss", raw)

    def test_nameless_dismiss_only_when_mortis_active(self):
        # 正由墨提斯代班时："让若叶睦说话/换回若叶睦"这类没提墨提斯名字的换人指令也算退场
        for raw in ("让若叶睦说话", "换回若叶睦", "若叶睦出来"):
            self.assertEqual(mortis_intent(raw, addressed=True, active=True), "dismiss", raw)
            self.assertIsNone(mortis_intent(raw, addressed=True, active=False), raw)
        # 群里的"我先回去了"不该触发换人
        self.assertIsNone(mortis_intent("我先回去了", addressed=True, active=True))

    def test_no_name_no_intent(self):
        self.assertIsNone(mortis_intent("若叶睦出来"))
        self.assertIsNone(mortis_intent("你好"))


class PickPersonaTests(unittest.TestCase):
    """人格选取：召唤窗口 > 随机概率；提到若叶睦的名字时强制若叶睦。"""

    def test_window_forces_mortis(self):
        self.assertTrue(pick_persona(mortis_active=True, addressed=False, mention_mutsumi=True,
                                     rate=0.0, roll=0.9))

    def test_random_rate_when_addressed(self):
        self.assertTrue(pick_persona(mortis_active=False, addressed=True, mention_mutsumi=False,
                                     rate=0.5, roll=0.1))
        self.assertFalse(pick_persona(mortis_active=False, addressed=True, mention_mutsumi=False,
                                      rate=0.5, roll=0.9))

    def test_mention_keeps_mutsumi_and_unaddressed_stays_mutsumi(self):
        self.assertFalse(pick_persona(mortis_active=False, addressed=True, mention_mutsumi=True,
                                      rate=1.0, roll=0.0))
        self.assertFalse(pick_persona(mortis_active=False, addressed=False, mention_mutsumi=False,
                                      rate=1.0, roll=0.0))


if __name__ == "__main__":
    unittest.main()
