"""Offline tests for the WP1 retrieval policy — no NoneBot, no network.

Run directly:
    python -X utf8 tests/test_anon_search_policy.py

The fixtures below are copied from ``bot_standalone.log`` (2026-10-25/26) with
the QQ noise kept verbatim; the test never reads the log file itself.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plugins import anon_search_policy as policy


class StripNoiseTest(unittest.TestCase):
    def test_at_segment(self):
        self.assertEqual(policy.strip_noise("[at:qq=200000003] 搜幽灵"), "搜幽灵")

    def test_reply_then_at_segment(self):
        self.assertEqual(
            policy.strip_noise(
                "[reply:id=487750944][at:qq=200000003] 你抽到Avemujica的票了嗎"
            ),
            "你抽到Avemujica的票了嗎",
        )

    def test_cq_code(self):
        self.assertEqual(policy.strip_noise("[CQ:at,qq=200000003] 搜幽灵"), "搜幽灵")

    def test_multiple_cq_codes(self):
        self.assertEqual(
            policy.strip_noise("[CQ:reply,id=123][CQ:at,qq=1] 搜幽灵"), "搜幽灵"
        )

    def test_whitespace_zero_width_and_trailing_punct(self):
        self.assertEqual(policy.strip_noise("  \u3000你好\u200b 在吗？ "), "你好 在吗")

    def test_leading_name_with_delimiter(self):
        self.assertEqual(policy.strip_noise("爱音，搜幽灵"), "搜幽灵")
        self.assertEqual(policy.strip_noise("爱音 搜幽灵"), "搜幽灵")
        self.assertEqual(policy.strip_noise("Anon: 搜幽灵"), "搜幽灵")
        self.assertEqual(policy.strip_noise("千早爱音 搜幽灵"), "搜幽灵")

    def test_leading_name_inside_a_word_is_kept(self):
        # “爱音的声优”里的“爱音”不是称呼，不能截坏。
        self.assertEqual(policy.strip_noise("爱音的声优是谁"), "爱音的声优是谁")

    def test_at_nickname(self):
        self.assertEqual(policy.strip_noise("@小明 搜幽灵"), "搜幽灵")

    def test_media_placeholders_only(self):
        self.assertEqual(policy.strip_noise("[图片] [语音]"), "")
        self.assertEqual(policy.strip_noise("[image:abc] [face:14]"), "")

    def test_internal_text_and_case_untouched(self):
        text = "你 抽到 Avemujica 的票了嗎"
        self.assertEqual(policy.strip_noise(text), text)

    def test_empty_and_none_tolerated(self):
        for value in ("", None, 123, [], False):
            with self.subTest(value=value):
                self.assertEqual(policy.strip_noise(value), "")

    def test_punctuation_only(self):
        self.assertEqual(policy.strip_noise("  ，。！ "), "")


class GlossaryTest(unittest.TestCase):
    def test_tomorin(self):
        self.assertEqual(policy.glossary_hits("誰是tomorin"), [("tomorin", "高松灯")])
        self.assertEqual(policy.glossary_hits("tomorin"), [("tomorin", "高松灯")])

    def test_tomorin_is_case_insensitive(self):
        self.assertEqual(policy.glossary_hits("Tomorin 在吗"), [("Tomorin", "高松灯")])

    def test_couqie(self):
        self.assertEqual(policy.glossary_hits("凑企鹅"), [("凑企鹅", "高松灯")])

    def test_suyouli(self):
        self.assertEqual(
            policy.glossary_hits("谁是苏幽璃"),
            [("苏幽璃", "长崎素世（古风二创名）")],
        )

    def test_feng_variants(self):
        for term in ("疯串箱子", "疯穿箱子", "疯氚箱子"):
            with self.subTest(term=term):
                self.assertEqual(
                    policy.glossary_hits(term), [(term, "丰川祥子（谐音）")]
                )

    def test_earliest_alias_wins(self):
        self.assertEqual(
            policy.glossary_hits("先疯穿箱子再疯串箱子"),
            [("疯穿箱子", "丰川祥子（谐音）")],
        )

    def test_mujica(self):
        self.assertEqual(policy.glossary_hits("母鸡卡"), [("母鸡卡", "Ave Mujica")])

    def test_one_hit_per_entry(self):
        self.assertEqual(
            policy.glossary_hits("邦娃和棉娃"), [("邦娃", "BanG Dream 棉花娃娃")]
        )

    def test_ruiping_and_zmd_order(self):
        self.assertEqual(
            policy.glossary_hits("瑞萍zmd"),
            [("瑞萍", "锐评（谐音）"), ("zmd", "明日方舟：终末地")],
        )

    def test_voice_actors(self):
        self.assertEqual(policy.glossary_hits("立石凛"), [("立石凛", "爱音的声优")])
        for term in ("李世林", "立凛石", "立戒石", "江界石"):
            with self.subTest(term=term):
                self.assertEqual(
                    policy.glossary_hits(term), [(term, "立石凛的故意谐音")]
                )
        self.assertEqual(
            policy.glossary_hits("羊宫妃那"), [("羊宫妃那", "高松灯的声优")]
        )
        self.assertEqual(
            policy.glossary_hits("青木阳菜"), [("青木阳菜", "要乐奈的声优")]
        )

    def test_no_ascii_false_positive(self):
        for text in ("tomorina 在吗", "zmdx"):
            with self.subTest(text=text):
                self.assertEqual(policy.glossary_hits(text), [])

    def test_no_chinese_false_positive(self):
        self.assertEqual(policy.glossary_hits("幽灵"), [])
        self.assertEqual(policy.glossary_hits("今天天气不错"), [])

    def test_empty_and_none_tolerated(self):
        for value in ("", None, 123):
            with self.subTest(value=value):
                self.assertEqual(policy.glossary_hits(value), [])
                self.assertEqual(policy.glossary_hint(value), "")

    def test_hint_exact(self):
        self.assertEqual(
            policy.glossary_hint("瑞萍zmd"),
            "【群内黑话】（群内用法优先于搜索标题；不确定就短句问，不要硬套）\n"
            "- 瑞萍 = 锐评（谐音）\n"
            "- zmd = 明日方舟：终末地",
        )

    def test_glossary_constant(self):
        for term, meaning in (
            ("tomorin", "高松灯"),
            ("凑企鹅", "高松灯"),
            ("苏幽璃", "长崎素世（古风二创名）"),
            ("疯串箱子", "丰川祥子（谐音）"),
            ("母鸡卡", "Ave Mujica"),
            ("邦娃", "BanG Dream 棉花娃娃"),
            ("瑞萍", "锐评（谐音）"),
            ("zmd", "明日方舟：终末地"),
            ("立石凛", "爱音的声优"),
            ("李世林", "立石凛的故意谐音"),
            ("羊宫妃那", "高松灯的声优"),
            ("青木阳菜", "要乐奈的声优"),
        ):
            with self.subTest(term=term):
                self.assertEqual(policy.GLOSSARY[term], meaning)


class DetectLookupTest(unittest.TestCase):
    def test_explicit_search(self):
        self.assertEqual(
            policy.detect_lookup("搜幽灵"),
            {"query": "幽灵", "queries": ["幽灵", "幽灵 是什么"], "reason": "explicit"},
        )

    def test_explicit_search_with_at_noise(self):
        self.assertEqual(
            policy.detect_lookup("[at:qq=200000003] 搜幽灵"),
            {"query": "幽灵", "queries": ["幽灵", "幽灵 是什么"], "reason": "explicit"},
        )

    def test_explicit_search_variants(self):
        for text in (
            "帮我搜幽灵",
            "帮我查一下幽灵",
            "查查幽灵",
            "百度一下幽灵",
            "谷歌一下幽灵",
            "搜索一下幽灵",
            "搜 幽灵 吧",
        ):
            with self.subTest(text=text):
                result = policy.detect_lookup(text)
                self.assertIsNotNone(result)
                self.assertEqual(result["reason"], "explicit")
                self.assertEqual(result["query"], "幽灵")

    def test_definition_tomorin(self):
        result = policy.detect_lookup("谁是tomorin")
        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "definition")
        self.assertEqual(result["query"], "tomorin 是谁")
        self.assertIn("tomorin", result["queries"][0])
        self.assertIn("tomorin MyGO", result["queries"])

    def test_definition_tomorin_with_mygo_summary(self):
        result = policy.detect_lookup("谁是tomorin", summary="群里在聊 MyGO")
        self.assertIsNotNone(result)
        self.assertEqual(result["queries"], ["tomorin 是谁", "tomorin MyGO"])

    def test_definition_traditional_who(self):
        result = policy.detect_lookup("誰是tomorin")
        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "definition")
        self.assertEqual(result["query"], "tomorin 是谁")

    def test_definition_feng(self):
        result = policy.detect_lookup("谁是疯串箱子")
        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "definition")
        self.assertEqual(result["query"], "疯串箱子 是谁")

    def test_definition_meme(self):
        self.assertEqual(
            policy.detect_lookup("幽灵是什么梗"),
            {
                "query": "幽灵 是什么梗",
                "queries": ["幽灵 是什么梗", "幽灵 MyGO"],
                "reason": "definition",
            },
        )

    def test_definition_what_is(self):
        result = policy.detect_lookup("什么是苏幽璃")
        self.assertIsNotNone(result)
        self.assertEqual(result["query"], "苏幽璃 是谁")

    def test_definition_suffix_who(self):
        result = policy.detect_lookup("苏幽璃是谁")
        self.assertIsNotNone(result)
        self.assertEqual(result["query"], "苏幽璃 是谁")

    def test_definition_meaning(self):
        result = policy.detect_lookup("zmd什么意思")
        self.assertIsNotNone(result)
        self.assertEqual(result["queries"], ["zmd 是什么梗", "zmd MyGO"])

    def test_definition_politeness_prefix(self):
        result = policy.detect_lookup("请问谁是tomorin")
        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "definition")

    def test_identity_targets_not_searched(self):
        for text in ("你是谁", "什么是AI", "我是谁", "这是啥"):
            with self.subTest(text=text):
                self.assertIsNone(policy.detect_lookup(text))

    def test_performance_real_log_line(self):
        text = (
            "青木阳菜在2025年7月1日 mygo zepp tour的猛独侵袭2分28秒"
            "吉他solo段弹的怎么样"
        )
        result = policy.detect_lookup("[at:qq=200000003] " + text)
        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "performance")
        self.assertEqual(result["query"], policy.strip_noise(text))
        self.assertEqual(len(result["queries"]), 1)

    def test_performance_without_date(self):
        result = policy.detect_lookup("猛独侵袭 2分28秒 solo 弹得怎么样")
        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "performance")

    def test_performance_song_title(self):
        result = policy.detect_lookup("《猛独侵袭》弹得怎么样")
        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "performance")

    def test_opinion_comparison_not_searched(self):
        for text in (
            "科比和詹姆斯谁是goat",
            "A和B谁更强",
            "谁是goat",
            "你更喜欢哪个",
            "哪个更好",
        ):
            with self.subTest(text=text):
                self.assertIsNone(policy.detect_lookup(text))

    def test_personal_state_not_searched(self):
        for text in (
            "你最近好吗",
            "你最近怎么样",
            "你今天怎么样",
            "在吗",
            "在不在",
            "早安",
            "晚安",
            "hello",
        ):
            with self.subTest(text=text):
                self.assertIsNone(policy.detect_lookup(text))

    def test_mood_and_weather_not_searched(self):
        for text in (
            "今天好丧啊",
            "今天天气怎么样",
            "天气如何",
            "好累",
            "心情不好",
            "不想动",
        ):
            with self.subTest(text=text):
                self.assertIsNone(policy.detect_lookup(text))

    def test_time_question_not_searched(self):
        self.assertIsNone(policy.detect_lookup("现在几点了"))

    def test_group_chat_opinions_not_searched(self):
        for text in (
            "这套卡组怎么样",
            "立希和海玲凑一对怎么说",
            "锐评一下满级觉醒超级骑士卡组怎么说",
            "今天天气不错",
            "视频通话",
        ):
            with self.subTest(text=text):
                self.assertIsNone(policy.detect_lookup(text))

    def test_homework_not_searched(self):
        self.assertIsNone(policy.detect_lookup("帮我查一下这道题怎么解"))
        self.assertIsNone(policy.detect_lookup("搜一下这道题怎么解"))

    def test_current_status(self):
        for text in ("MyGO 下次演出什么时候", "母鸡卡最近有什么新消息"):
            with self.subTest(text=text):
                result = policy.detect_lookup(text)
                self.assertIsNotNone(result)
                self.assertEqual(result["reason"], "current")
                self.assertEqual(result["queries"], [result["query"]])

    def test_personal_schedule_not_current(self):
        # 2026-10-26 复审 medium：日程/寒暄不查外部事实。
        for text in (
            "你什么时候回来",
            "什么时候吃饭",
            "我们什么时候出发",
            "明天什么时候集合",
            "群里怎么了",
            "今天几号",
        ):
            with self.subTest(text=text):
                self.assertIsNone(policy.detect_lookup(text))

    def test_empty_and_none_tolerated(self):
        for value in ("", None, 123):
            with self.subTest(value=value):
                self.assertIsNone(policy.detect_lookup(value))

    def test_overlong_text_ignored(self):
        self.assertIsNone(policy.detect_lookup("谁是" + "长" * 250))

    def test_contract_shape(self):
        positive = (
            "搜幽灵",
            "谁是tomorin",
            "幽灵是什么梗",
            "青木阳菜在2025年7月1日 mygo zepp tour的猛独侵袭2分28秒"
            "吉他solo段弹的怎么样",
            "MyGO 下次演出什么时候",
        )
        for text in positive:
            with self.subTest(text=text):
                result = policy.detect_lookup(text)
                self.assertIsNotNone(result)
                self.assertEqual(set(result), {"query", "queries", "reason"})
                self.assertIn(
                    result["reason"], {"explicit", "definition", "performance", "current"}
                )
                self.assertTrue(1 <= len(result["queries"]) <= 2)
                self.assertEqual(result["query"], result["queries"][0])
                self.assertEqual(len(result["queries"]), len(set(result["queries"])))
                for query in result["queries"]:
                    self.assertTrue(1 <= len(query) <= 60, query)


class ClassifyRequestTest(unittest.TestCase):
    def test_send_face(self):
        for text in ("发个表情", "卖个萌", "比个心", "发个经典表情", "给我笑一个"):
            with self.subTest(text=text):
                self.assertEqual(policy.classify_request(text), "send_face")

    def test_send_media(self):
        for text in (
            "发张图",
            "发个表情包",
            "发语音",
            "唱一段",
            "唱歌",
            "给我发照片",
            "发个视频",
            "录一段",
        ):
            with self.subTest(text=text):
                self.assertEqual(policy.classify_request(text), "send_media")

    def test_contact_third_party(self):
        for text in (
            "帮我转告小明明天开会",
            "帮我发消息给祥子",
            "替我联系一下老师",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    policy.classify_request(text), "contact_third_party"
                )

    def test_purchase_login(self):
        for text in (
            "可以帮我抢下周六日的avemujica应援场的票吗",
            "帮我抢票",
            "帮我登录一下账号",
            "签到",
            "帮我付款",
            "怎么下单",
        ):
            with self.subTest(text=text):
                self.assertEqual(policy.classify_request(text), "purchase_login")

    def test_search(self):
        for text in ("帮我查资料", "谁是tomorin", "搜幽灵", "解释一下这个梗"):
            with self.subTest(text=text):
                self.assertEqual(policy.classify_request(text), "search")

    def test_none_cases(self):
        for text in (
            "瑞萍zmd",
            "（凑近看你屏幕）发图不说话？想人家了直说嘛～",
            "视频通话",
            "我发了图",
            "今天天气不错",
            "桂花糕",
        ):
            with self.subTest(text=text):
                self.assertEqual(policy.classify_request(text), "none")

    def test_creative_and_statements_not_requests(self):
        # 2026-10-26 复审 high：动作词只是话题/陈述时，不能判成让爱音执行。
        for text in (
            "帮我写一段唱歌的歌词",
            "帮我写一份抢票攻略",
            "我们一起唱歌吧",
            "唱歌好难",
            "我建议你先登录，提前填好信息",
            "青木阳菜唱歌怎么样",
            "灯唱歌真好听",
        ):
            with self.subTest(text=text):
                self.assertEqual(policy.classify_request(text), "none")

    def test_empty_and_none_tolerated(self):
        for value in ("", None, 123):
            with self.subTest(value=value):
                self.assertEqual(policy.classify_request(value), "none")

    def test_homework_not_search(self):
        self.assertEqual(policy.classify_request("帮我查一下这道题怎么解"), "none")

    def test_only_documented_values(self):
        allowed = {
            "none",
            "search",
            "send_face",
            "send_media",
            "contact_third_party",
            "purchase_login",
        }
        for text in (
            "发个表情",
            "发张图",
            "帮我抢票",
            "帮我转告小明",
            "谁是tomorin",
            "随便聊聊",
        ):
            with self.subTest(text=text):
                self.assertIn(policy.classify_request(text), allowed)


class ActionNoteTest(unittest.TestCase):
    def test_unknown_and_empty(self):
        for kind in ("", None, "unknown", "send_image"):
            with self.subTest(kind=kind):
                self.assertEqual(policy.action_note(kind), "")

    def test_none_kind_is_empty(self):
        self.assertEqual(policy.action_note("none"), "")

    def test_search(self):
        self.assertIn("不确定", policy.action_note("search"))

    def test_send_face(self):
        note = policy.action_note("send_face")
        self.assertIn("一个", note)
        for token in ("heart", "shy", "smile"):
            self.assertIn(token, note)

    def test_send_media(self):
        note = policy.action_note("send_media")
        self.assertIn("不能发图片", note)
        self.assertIn("假装", note)

    def test_contact_third_party(self):
        self.assertIn("不能替", policy.action_note("contact_third_party"))

    def test_purchase_login(self):
        note = policy.action_note("purchase_login")
        self.assertIn("不能抢票", note)
        self.assertIn("账号密码", note)


class PickFaceActionTest(unittest.TestCase):
    def test_send_face_default(self):
        self.assertEqual(policy.pick_face_action("发个表情"), "face:smile")

    def test_shy(self):
        self.assertEqual(policy.pick_face_action("卖个萌"), "face:shy")

    def test_heart(self):
        self.assertEqual(policy.pick_face_action("比个心"), "face:heart")

    def test_media_is_none(self):
        self.assertEqual(policy.pick_face_action("发张图"), "none")

    def test_non_face_is_none(self):
        for text in ("今天天气不错", "", None, "视频通话"):
            with self.subTest(text=text):
                self.assertEqual(policy.pick_face_action(text), "none")

    def test_face_request_never_returns_none(self):
        for text in ("发个表情", "卖个萌", "比个心", "给我笑一个"):
            with self.subTest(text=text):
                self.assertNotEqual(policy.pick_face_action(text), "none")
                self.assertIn(
                    policy.pick_face_action(text),
                    {"face:heart", "face:shy", "face:smile"},
                )


class RealLogFalsePositiveTest(unittest.TestCase):
    """2026-10-26 主AI 补丁：真实日志里“能在当前群直接做”和“陈述句”不误判为外发。"""

    def test_ingroup_ask_is_allowed(self):
        # 19:47:56 原话：爱音直接在群里问群友A即可，不需要私聊第三人。
        self.assertEqual(
            policy.classify_request("你问一下群友A他爸是谁"), "none"
        )

    def test_purchase_statement_is_not_request(self):
        # 18:45:27 原话：“可买票要抽签啊”是感慨，不是让爱音去抢票。
        for text in ("可买票要抽签啊", "买票要抽签啊", "抢票得排队"):
            with self.subTest(text=text):
                self.assertEqual(policy.classify_request(text), "none")

    def test_narrative_send_is_not_request(self):
        # 01:49:33 原话：群友自己发视频邀请，不是让爱音发视频。
        for text in ("（发送视频通话邀请）", "我发了图", "视频通话"):
            with self.subTest(text=text):
                self.assertEqual(policy.classify_request(text), "none")

    def test_persona_definition_not_searched(self):
        # 人设/自称类问题用角色卡回答，不查网页。
        for text in ("你最喜欢的美妆博主是谁", "谁是爱音", "千早爱音是谁"):
            with self.subTest(text=text):
                self.assertIsNone(policy.detect_lookup(text))


class RequestContextTest(unittest.TestCase):
    """2026-10-26 评审修复：第三人称叙述不是让爱音执行动作。"""

    def test_third_person_media_narrative_is_none(self):
        for text in (
            "青木阳菜唱歌怎么样",
            "灯唱歌真好听",
            "她发图给我了",
            "她录了一段",
        ):
            with self.subTest(text=text):
                self.assertEqual(policy.classify_request(text), "none")

    def test_third_person_contact_narrative_is_none(self):
        for text in (
            "老师转告我明天考试",
            "他带话说周末聚",
            "老师让我转告你",
        ):
            with self.subTest(text=text):
                self.assertEqual(policy.classify_request(text), "none")

    def test_real_requests_still_classified(self):
        cases = (
            ("发张图", "send_media"),
            ("唱歌", "send_media"),
            ("想听你唱歌", "send_media"),
            ("帮我转告小明明天开会", "contact_third_party"),
            ("转告小明明天开会", "contact_third_party"),
            ("你转告他一声", "contact_third_party"),
        )
        for text, kind in cases:
            with self.subTest(text=text):
                self.assertEqual(policy.classify_request(text), kind)


class LookupFalsePositiveTest(unittest.TestCase):
    """2026-10-26 评审修复：钟点/弱时间词不单独触发搜索，查询词不被截坏。"""

    def test_clock_time_alone_is_not_performance(self):
        self.assertIsNone(policy.detect_lookup("明天 8:30 集合怎么样"))

    def test_clock_with_strong_performance_context(self):
        result = policy.detect_lookup("MyGO live 8:30 弹得怎么样")
        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "performance")

    def test_weak_current_words_alone_are_not_lookup(self):
        for text in ("你现在玩什么游戏", "大家最近在玩什么"):
            with self.subTest(text=text):
                self.assertIsNone(policy.detect_lookup(text))

    def test_query_verb_not_mangled(self):
        self.assertEqual(
            policy.detect_lookup("查询一下群公告"),
            {
                "query": "群公告",
                "queries": ["群公告", "群公告 是什么"],
                "reason": "explicit",
            },
        )

    def test_sougua_is_not_a_search_request(self):
        self.assertIsNone(policy.detect_lookup("搜刮一下附近的店"))


if __name__ == "__main__":
    unittest.main()
