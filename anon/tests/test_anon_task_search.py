# -*- coding: utf-8 -*-
"""WP6（2026-10-26 10:00）：researcher simple 任务必须搜索 + evaluation 判据。

全部离线：agent_search 必须 mock；不启动 NoneBot、不联网、不发 QQ。
"""
import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plugins import agent_core, anonbot, anon_search_policy as policy


class _FakePrivateEvent:
    def __init__(self, text, user_id="70001"):
        self.user_id = int(user_id)
        self.message_type = "private"
        self.raw_message = text
        self.message_id = 1
        self.self_id = "200000003"
        self.sender = None
        self.reply = None
        self._text = text

    def get_message(self):
        from nonebot.adapters.onebot.v11 import MessageSegment

        return [MessageSegment.text(self._text)]

    def is_tome(self):
        return True


class FakeBot:
    self_id = "200000003"

    def __init__(self):
        self.sent = []

    async def send(self, event, message):
        self.sent.append(message)


class TaskResearcherSearchRoutingTest(unittest.TestCase):
    """A 组：researcher 的 simple 任务也必须走 run_subagent（内含 agent_search）。"""

    def _run(self, plan, text="介绍一下蒋介石", user_id="70001"):
        calls = {"run": 0, "simple": 0}
        sent = []

        async def fake_understand(*args, **kwargs):
            return dict(plan)

        async def fake_run(*args, **kwargs):
            calls["run"] += 1
            return "研究结果"

        async def fake_simple(*args, **kwargs):
            calls["simple"] += 1
            return "简单结果"

        async def fake_send(bot, event, answer, face=None):
            sent.append({"text": answer, "face": face})

        async def fake_distill(*args, **kwargs):
            return None

        event = _FakePrivateEvent(text, user_id)
        key = ("private:" + user_id, user_id)
        anonbot.histories.pop(key, None)
        patches = [
            mock.patch.object(anonbot.agent_core, "understand", fake_understand),
            mock.patch.object(anonbot.agent_core, "run_subagent", fake_run),
            mock.patch.object(anonbot.agent_core, "answer_simple_task", fake_simple),
            mock.patch.object(anonbot, "_send_reply", fake_send),
            mock.patch.object(anonbot, "_log_sent", lambda *a, **k: None),
            mock.patch.object(anonbot, "_record_profile", lambda *a, **k: None),
            mock.patch.object(anonbot, "_distill", fake_distill),
            mock.patch.object(anonbot.experience, "render_hint", lambda *a, **k: ""),
        ]
        for patch in patches:
            patch.start()
        try:
            asyncio.run(anonbot.handle_message(FakeBot(), event))
        finally:
            for patch in reversed(patches):
                patch.stop()
            anonbot.histories.pop(key, None)
            anonbot.scope_locks.pop("private:" + user_id, None)
        return calls, sent

    def _plan(self, agent="researcher", complex_=False):
        return {
            "mode": "task", "reply": "", "agent": agent,
            "need": "介绍蒋介石", "steps": [], "complex": complex_,
            "fresh": False, "act": "none",
        }

    def test_researcher_simple_calls_run_subagent(self):
        calls, sent = self._run(self._plan())
        self.assertEqual(calls["run"], 1)
        self.assertEqual(calls["simple"], 0)
        self.assertEqual(sent[-1]["text"], "研究结果")

    def test_researcher_simple_does_not_call_answer_simple_task(self):
        calls, _sent = self._run(self._plan(), user_id="70002")
        self.assertEqual(calls["simple"], 0)

    def test_researcher_complex_calls_run_subagent(self):
        calls, _sent = self._run(self._plan(complex_=True), user_id="70003")
        self.assertEqual(calls["run"], 1)
        self.assertEqual(calls["simple"], 0)

    def test_writer_simple_still_calls_answer_simple_task(self):
        calls, sent = self._run(self._plan(agent="writer"), user_id="70004")
        self.assertEqual(calls["simple"], 1)
        self.assertEqual(calls["run"], 0)
        self.assertEqual(sent[-1]["text"], "简单结果")

    def test_task_route_source_contains_researcher_or(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "src" / "plugins" / "anonbot.py"
        ).read_text(encoding="utf-8")
        self.assertIn('plan.get("agent") == "researcher"', source)
        self.assertIn("WP6", source)
        self.assertNotIn('if plan.get("complex"):', source)


class EvaluationLookupTest(unittest.TestCase):
    """B 组：评价具体公开对象 -> reason=evaluation。"""

    def _assert_eval(self, text, target):
        result = policy.detect_lookup(text)
        self.assertIsNotNone(result, msg=text)
        self.assertEqual(result["reason"], "evaluation", msg=text)
        self.assertEqual(result["query"], result["queries"][0], msg=text)
        self.assertEqual(
            result["queries"][:2], ["%s 评价" % target, target], msg=text
        )

    def _assert_none(self, text):
        self.assertIsNone(policy.detect_lookup(text), msg=text)

    def test_positive_ras(self):
        self._assert_eval("评价一下ras", "ras")

    def test_positive_raychell(self):
        self._assert_eval("评价一下我妈raychell", "我妈raychell")

    def test_positive_yagumy(self):
        self._assert_eval("评价一下羊宫妃娜", "羊宫妃娜")

    def test_positive_tateishi(self):
        self._assert_eval("评价立石凛", "立石凛")

    def test_positive_new_game(self):
        self._assert_eval("邦多利新游你如何评价", "邦多利新游")

    def test_positive_mujica(self):
        self._assert_eval("锐评一下母鸡卡", "母鸡卡")

    def test_positive_mygo_album(self):
        self._assert_eval("你怎么看 MyGO 的新专辑", "MyGO 的新专辑")

    def test_positive_aoki_guitar(self):
        self._assert_eval("如何评价青木阳菜的吉他", "青木阳菜的吉他")

    def test_negative_my_singing(self):
        self._assert_none("评价一下我唱歌")

    def test_negative_image(self):
        self._assert_none("评价一下这张图")

    def test_negative_anon_self(self):
        self._assert_none("评价一下爱音")

    def test_negative_comparison(self):
        self._assert_none("评价一下我和祥子谁更强")

    def test_negative_mood(self):
        self._assert_none("评价一下今天好丧")

    def test_negative_homework(self):
        self._assert_none("评价一下这道题怎么解")

    def test_negative_personal_purchase(self):
        self._assert_none("评价一下我买的衣服")

    def test_negative_generic(self):
        self._assert_none("评价一下这个")

    def test_negative_game_deck_regression(self):
        self._assert_none("锐评一下满级觉醒超级骑士卡组怎么说")

    def test_negative_plain_performance_regression(self):
        self._assert_none("青木阳菜唱歌怎么样")

    def test_queries_shape(self):
        result = policy.detect_lookup("评价一下ras")
        self.assertEqual(result["queries"], ["ras 评价", "ras"])
        self.assertLessEqual(len(result["queries"]), 2)
        for query in result["queries"]:
            self.assertLessEqual(len(query), 60)
        self.assertEqual(len(result["queries"]), len(set(result["queries"])))

    def test_queries_keep_spaces_and_english(self):
        result = policy.detect_lookup("你怎么看 MyGO 的新专辑")
        self.assertEqual(
            result["queries"], ["MyGO 的新专辑 评价", "MyGO 的新专辑"]
        )

    def test_performance_precedes_evaluation(self):
        text = (
            "评价一下2025年7月1日 mygo zepp tour的猛独侵袭2分28秒"
            "吉他solo段弹得怎么样"
        )
        self.assertEqual(policy.detect_lookup(text)["reason"], "performance")

    def test_object_length_bounds(self):
        for text in (
            "评价一下",
            "评价一下AI",
            "评价一下公司",
            "评价一下" + "某" * 31,
        ):
            with self.subTest(text=text):
                self._assert_none(text)


class EvaluationActionNoteTest(unittest.TestCase):
    """C 组：评价类请求的检索分类与诚实边界。"""

    def test_search_action_note_has_evaluation_guard(self):
        note = policy.action_note("search")
        for token in (
            "搜到的公开评价",
            "你自己的看法",
            "我看了",
            "我在现场",
            "没查到可靠资料",
        ):
            self.assertIn(token, note)

    def test_evaluation_request_classified_as_search(self):
        self.assertEqual(policy.classify_request("评价一下ras"), "search")
        self.assertEqual(
            policy.classify_request("邦多利新游你如何评价"), "search"
        )
        for text in ("评价一下我唱歌", "评价一下爱音", "评价一下这张图"):
            with self.subTest(text=text):
                self.assertEqual(policy.classify_request(text), "none")

    def test_research_prompt_includes_evaluation_guard(self):
        captured = []

        async def fake_ask(prompt, limit):
            captured.append(prompt)
            if len(captured) == 1:
                return '{"queries":["ras 评价"]}'
            return "搜到的公开评价：RAS 现场很稳。"

        plan = {
            "mode": "task", "agent": "researcher", "need": "评价一下ras",
            "steps": [], "complex": False,
        }
        with mock.patch.object(
            agent_core.agent_search, "search", return_value="- RAS 评价摘要"
        ):
            result = asyncio.run(
                agent_core.run_subagent(
                    plan,
                    "评价一下ras",
                    "验收",
                    [],
                    fake_ask,
                    action_note=policy.action_note("search"),
                )
            )
        self.assertTrue(result)
        final_prompt = captured[-1]
        for token in (
            "搜到的公开评价", "我看了", "我在现场", "没查到可靠资料"
        ):
            self.assertIn(token, final_prompt)


if __name__ == "__main__":
    unittest.main()
