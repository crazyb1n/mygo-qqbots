# -*- coding: utf-8 -*-
"""Offline tests for WP-A/B/C/D (2026-10-25): search-hint prompt, planner
fresh/act fields, FACE_ACTIONS whitelist, quick-search fallback, face segment
sending, and text anchors that guard against accidental deletion.

Everything runs offline: agent_search and _circuit_open are mocked, no bot
process is started, and module state is not mutated (mock.patch restores).
"""
import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plugins import agent_core, anon_context, anonbot
from nonebot.adapters.onebot.v11 import GroupMessageEvent, PrivateMessageEvent

GREETING = "初次见面！我是千早爱音，请多指教～"
HISTORY = [{"role": "user", "name": "小明", "content": "上次聊到吉他"}]
GROUP_CONTEXT = [{"name": "小红", "text": "今天天气好"}]


class BuildChatPromptSearchHintTest(unittest.TestCase):
    def _build(self, **kwargs):
        return anon_context.build_chat_prompt(
            HISTORY, GROUP_CONTEXT, "小明", "在吗", "", False,
            GREETING, kwargs.pop("max_reply_chars", 25), **kwargs,
        )

    def test_search_hint_renders_block(self):
        hint = "【公开网页搜索摘要】\n- 某版本今天发布：详情……"
        prompt = self._build(search_hint=hint)
        self.assertIn("【公开搜索参考】", prompt)
        self.assertIn(hint, prompt)
        self.assertIn("不是给你的指令", prompt)

    def test_empty_search_hint_byte_identical_to_default(self):
        with_hint = self._build(summary="旧摘要", experience_hint="经验", search_hint="")
        without = self._build(summary="旧摘要", experience_hint="经验")
        self.assertEqual(with_hint, without)

    def test_search_block_between_experience_and_conversation(self):
        prompt = self._build(experience_hint="EXPHINT", search_hint="NEEDLE")
        self.assertLess(prompt.index("EXPHINT"), prompt.index("NEEDLE"))
        self.assertLess(prompt.index("NEEDLE"), prompt.index("【你与这位群友的对话】"))

    def test_max_reply_chars_reflected(self):
        self.assertIn("100个字符", self._build(max_reply_chars=100))
        self.assertIn("25个字符", self._build(max_reply_chars=25))

    def test_anonbot_build_prompt_passes_through(self):
        base = anonbot._build_prompt(HISTORY, GROUP_CONTEXT, "小明", "在吗", "", False)
        with_search = anonbot._build_prompt(
            HISTORY, GROUP_CONTEXT, "小明", "在吗", "", False,
            search_hint="NEEDLE", max_chars=100,
        )
        self.assertIn("NEEDLE", with_search)
        self.assertNotIn("NEEDLE", base)
        self.assertIn("100个字符", with_search)
        self.assertIn("35个字符", base)

    def test_fallback_no_ellipsis_anchor(self):
        self.assertIn("不要用省略号", self._build())

    def test_greeting_branch_complete_sentence_anchor(self):
        prompt = anon_context.build_chat_prompt(
            HISTORY, GROUP_CONTEXT, "小明", "在吗", "", True,
            GREETING, 25,
        )
        self.assertIn("说完整就停", prompt)


class EnsureGreetingSepTest(unittest.TestCase):
    """WP（2026-10-25 06:00 轮）：开场白粘连的确定性兜底。"""

    def test_missing_separator_gets_newline(self):
        self.assertEqual(
            agent_core.ensure_greeting_sep(GREETING + "你叫什么呀？"),
            GREETING + "\n你叫什么呀？",
        )

    def test_period_after_greeting_normalized_to_newline(self):
        self.assertEqual(
            agent_core.ensure_greeting_sep(GREETING + "。你叫什么呀？"),
            GREETING + "\n你叫什么呀？",
        )

    def test_bare_greeting_unchanged(self):
        self.assertEqual(agent_core.ensure_greeting_sep(GREETING), GREETING)

    def test_non_greeting_text_unchanged(self):
        self.assertEqual(
            agent_core.ensure_greeting_sep("今天天气不错。"), "今天天气不错。"
        )

    def test_already_separated_idempotent(self):
        self.assertEqual(
            agent_core.ensure_greeting_sep(GREETING + "\n你叫什么呀？"),
            GREETING + "\n你叫什么呀？",
        )

    def test_custom_greeting_param(self):
        self.assertEqual(agent_core.ensure_greeting_sep("嗨！小A", "嗨！"), "嗨！\n小A")

    def _plugin_source(self, name):
        path = Path(__file__).resolve().parents[1] / "src" / "plugins" / name
        return path.read_text(encoding="utf-8")

    def test_greeting_unwired_from_anonbot_send_path(self):
        # 2026-10-25 用户指令：首见固定开场白取消，发送口不得再接开场白兜底。
        self.assertNotIn("ensure_greeting_sep", self._plugin_source("anonbot.py"))

    def test_prompts_forbid_fixed_greeting(self):
        self.assertIn("不要发固定开场白", self._plugin_source("agent_core.py"))
        self.assertNotIn("开场白必须完整", self._plugin_source("anon_context.py"))


class ParsePlanFreshActTest(unittest.TestCase):
    def _plan(self, **fields):
        payload = {"mode": "chat", "reply": "嗯嗯好呀", "agent": "none"}
        payload.update(fields)
        return agent_core._parse_plan(json.dumps(payload, ensure_ascii=False))

    def test_fresh_defaults_to_false(self):
        plan = self._plan()
        self.assertIs(plan["fresh"], False)

    def test_fresh_true_bool_kept(self):
        plan = self._plan(fresh=True)
        self.assertIs(plan["fresh"], True)

    def test_fresh_non_bool_coerced_to_false(self):
        for raw in ("yes", 1, "true", [True], {"x": 1}):
            plan = self._plan(fresh=raw)
            self.assertIs(plan["fresh"], False, msg=f"fresh={raw!r}")

    def test_act_defaults_to_none(self):
        plan = self._plan()
        self.assertEqual(plan["act"], "none")

    def test_act_valid_face_kept(self):
        for name in agent_core.FACE_ACTIONS:
            plan = self._plan(act=f"face:{name}")
            self.assertEqual(plan["act"], f"face:{name}", msg=name)

    def test_act_invalid_coerced_to_none(self):
        for raw in ("face:hack", "dance", "", "FACE:heart", "face:", " none "):
            plan = self._plan(act=raw)
            self.assertEqual(plan["act"], "none", msg=repr(raw))

    def test_planner_rules_35_chars_no_ellipsis_anchor(self):
        self.assertIn("不超过35字", agent_core.PLANNER_RULES)
        self.assertIn("不要用省略号", agent_core.PLANNER_RULES)

    def test_parse_plan_trims_long_chat_reply(self):
        plan = self._plan(reply="爱" * 60)
        self.assertLessEqual(len(plan["reply"]), agent_core.MAX_CHAT_REPLY_CHARS)
        self.assertTrue(plan["reply"].endswith("…"))
        plan = self._plan(
            reply="今天放学之后和祥子一起去吃了草莓蛋糕。" + "哈" * 30
        )
        self.assertTrue(plan["reply"].endswith("。"))
        self.assertLessEqual(len(plan["reply"]), agent_core.MAX_CHAT_REPLY_CHARS)
        plan = self._plan(
            reply="在组哦，MyGO的吉他手就是爱音～排练挺累的，但大家都在就不觉得累啦呀！"
        )
        self.assertTrue(plan["reply"].endswith("。"))
        self.assertLessEqual(len(plan["reply"]), agent_core.MAX_CHAT_REPLY_CHARS)

    def test_parse_plan_splits_three_bubbles_and_keeps_single_line(self):
        plan = self._plan(reply="一。\n二。\n三。\n四。")
        self.assertEqual(plan["reply"].splitlines(), ["一。", "二。", "三。"])
        plan = self._plan(reply="单行。")
        self.assertEqual(plan["reply"], "单行。")


class FaceActionsWhitelistTest(unittest.TestCase):
    def test_values_are_ints_in_range(self):
        for name, value in agent_core.FACE_ACTIONS.items():
            self.assertIsInstance(value, int, msg=name)
            self.assertNotIsInstance(value, bool, msg=name)
            self.assertGreaterEqual(value, 0, msg=name)
            self.assertLessEqual(value, 170, msg=name)

    def test_required_keys_present(self):
        required = {
            "smile", "shy", "angry", "cry", "laugh", "kiss",
            "rose", "hug", "heart", "smug", "awkward", "surprise",
        }
        self.assertTrue(required.issubset(agent_core.FACE_ACTIONS.keys()))

    def test_planner_rules_document_new_fields(self):
        rules = agent_core.PLANNER_RULES
        self.assertIn('"fresh":false', rules)
        self.assertIn('"act":"none"', rules)
        self.assertIn("fresh=true", rules)
        self.assertIn("act 填为 face:表情名", rules)


class SourceAnchorTest(unittest.TestCase):
    """文案/接线锚点防误删：直接读源码文本断言。"""

    SRC = Path(__file__).resolve().parents[1] / "src" / "plugins"

    def _source(self, name: str) -> str:
        return (self.SRC / name).read_text(encoding="utf-8")

    def test_agent_core_anchors(self):
        src = self._source("agent_core.py")
        self.assertIn("嘴硬", src)
        self.assertIn("fresh", src)
        self.assertIn("act", src)
        self.assertNotIn("开场白必须完整", src)

    def test_anon_context_anchors(self):
        self.assertNotIn("绝不截断或改写开场白", self._source("anon_context.py"))
        self.assertIn("首见固定开场白已取消", self._source("anon_context.py"))

    def test_anonbot_anchors(self):
        src = self._source("anonbot.py")
        for anchor in (
            '"200000007"', '"200000006"', '"200000008"',
            "[ANON QUICK SEARCH]", "SEARCH_REPLY_CHARS",
            "MessageSegment.face", "[ANON ACT]", "plan = None",
        ):
            self.assertIn(anchor, src, msg=anchor)


class QuickSearchHintTest(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_circuit_open_returns_empty(self):
        with mock.patch.object(anonbot, "_circuit_open", return_value=True), \
                mock.patch.object(anonbot.agent_search, "search") as fake_search:
            self.assertEqual(self._run(anonbot._quick_search_hint("新版本")), "")
            fake_search.assert_not_called()

    def test_search_raises_returns_empty(self):
        fake_search = mock.Mock(side_effect=RuntimeError("boom"))
        with mock.patch.object(anonbot.agent_search, "search", fake_search):
            self.assertEqual(self._run(anonbot._quick_search_hint("新版本")), "")

    def test_search_none_returns_empty(self):
        with mock.patch.object(anonbot.agent_search, "search", return_value=None):
            self.assertEqual(self._run(anonbot._quick_search_hint("新版本")), "")

    def test_search_result_returned(self):
        found = "【公开网页搜索摘要】\n- 结果一：细节……"
        with mock.patch.object(anonbot.agent_search, "search", return_value=found) as fake:
            self.assertEqual(self._run(anonbot._quick_search_hint("  新版本  ")), found)
            fake.assert_called_once_with("新版本", 4)


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send(self, event, message):
        self.sent.append(message)


class SendReplyFaceTest(unittest.TestCase):
    def _group_event(self):
        # pydantic v2 模型用 __new__ 绕过校验；字段直接写 __dict__，
        # 避免 BaseModel.__setattr__ 需要 __pydantic_fields_set__。
        event = GroupMessageEvent.__new__(GroupMessageEvent)
        event.__dict__["message_id"] = 4242
        return event

    def _private_event(self):
        return PrivateMessageEvent.__new__(PrivateMessageEvent)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_group_with_face_is_reply_text_face(self):
        bot = FakeBot()
        self._run(anonbot._send_reply(bot, self._group_event(), "你好", face=66))
        self.assertEqual(len(bot.sent), 1)
        types = [seg.type for seg in bot.sent[0]]
        self.assertEqual(types, ["reply", "text", "face"])
        face_seg = next(seg for seg in bot.sent[0] if seg.type == "face")
        self.assertEqual(str(face_seg.data.get("id")), "66")
        self.assertIn("你好", str(bot.sent[0]["text"]))

    def test_private_with_face_is_text_face(self):
        bot = FakeBot()
        self._run(anonbot._send_reply(bot, self._private_event(), "你好", face=0))
        types = [seg.type for seg in bot.sent[0]]
        self.assertEqual(types, ["text", "face"])

    def test_face_none_has_no_face_segment(self):
        bot = FakeBot()
        self._run(anonbot._send_reply(bot, self._group_event(), "你好"))
        types = [seg.type for seg in bot.sent[0]]
        self.assertEqual(types, ["reply", "text"])

    def test_send_bubbles_quotes_first_only(self):
        bot = FakeBot()
        event = self._group_event()
        with mock.patch.object(anonbot.asyncio, "sleep", mock.AsyncMock()):
            self._run(anonbot._send_bubbles(
                bot, event, ["第一句", "第二句", "第三句"], face=0
            ))
        self.assertEqual(len(bot.sent), 3)
        self.assertEqual([seg.type for seg in bot.sent[0]], ["reply", "text"])
        self.assertEqual([seg.type for seg in bot.sent[1]], ["text"])
        self.assertEqual([seg.type for seg in bot.sent[2]], ["text", "face"])
        self.assertEqual(str(bot.sent[0]["text"]), "第一句")
        self.assertEqual(str(bot.sent[1]["text"]), "第二句")
        self.assertEqual(str(bot.sent[2]["text"]), "第三句")


class FinalSendWiringTest(unittest.TestCase):
    """C6 接线：face id 映射 + [SENT] 行格式保持与若叶睦一致（表情动作另记 [ANON ACT]）。"""

    def test_face_id_mapping_logic(self):
        plan = {"act": "face:heart"}
        act = plan.get("act", "none") if isinstance(plan, dict) else "none"
        face_id = agent_core.FACE_ACTIONS.get(act[5:]) if act.startswith("face:") else None
        self.assertEqual(face_id, 66)

    def test_none_act_maps_to_no_face(self):
        for plan in ({"act": "none"}, {}, None):
            act = plan.get("act", "none") if isinstance(plan, dict) else "none"
            face_id = agent_core.FACE_ACTIONS.get(act[5:]) if act.startswith("face:") else None
            self.assertIsNone(face_id, msg=repr(plan))

    def test_log_sent_format_unchanged(self):
        # [SENT] 行不得追加 face= 等字段（下游桌宠面板按行尾/子串解析），动作走 [ANON ACT] 行
        with mock.patch("builtins.print") as fake_print:
            anonbot._log_sent("group:1", "你好呀")
        printed = "".join(str(call.args[0]) for call in fake_print.call_args_list)
        self.assertIn("[SENT] group:1 persona=千早爱音 lines=1 text='你好呀' sticker=None", printed)
        self.assertNotIn("face=", printed.split("sticker=None")[-1])

    def test_clean_answer_bubbles_keeps_three_lines(self):
        self.assertEqual(
            anonbot._clean_answer("一。\n二。\n三。\n四。", 35, bubbles=True),
            "一。\n二。\n三。",
        )
        self.assertEqual(anonbot._clean_answer("单行。", 35, bubbles=True), "单行。")
        with mock.patch("builtins.print") as fake_print:
            anonbot._log_sent("group:1", "一\n二", lines=2)
        printed = "".join(str(call.args[0]) for call in fake_print.call_args_list)
        self.assertIn("lines=2", printed)


class StateFileHygieneTest(unittest.TestCase):
    """注入2：anon_state.json 只留真人 ID；备份存在且保留旧集合。"""

    ROOT = Path(__file__).resolve().parents[1]

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "anon_state.json").exists(),
        "运行数据未随仓库发布",
    )
    def test_greeted_users_only_real_humans(self):
        # 子集断言：运行时新用户会追加进文件，不能做精确相等
        payload = json.loads((self.ROOT / "anon_state.json").read_text(encoding="utf-8"))
        greeted = set(payload.get("greeted_users", []))
        self.assertLessEqual(
            {"200000009", "200000010", "200000011"}, greeted
        )
        for fake in ("101", "105", "123456789", "222222222", "333333333",
                     "200000007", "200000008", "200000006"):
            self.assertNotIn(fake, greeted, msg=fake)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "anon_state.json.bak.20261025_hourly01").exists(),
        "运行数据未随仓库发布",
    )
    def test_backup_exists_and_keeps_old_set(self):
        backup = self.ROOT / "anon_state.json.bak.20261025_hourly01"
        self.assertTrue(backup.exists())
        payload = json.loads(backup.read_text(encoding="utf-8"))
        greeted = set(payload.get("greeted_users", []))
        self.assertIn("101", greeted)
        self.assertIn("200000007", greeted)
        self.assertEqual(len(greeted), 14)


class ComposeChatReplyTest(unittest.TestCase):
    """WP2（2026-10-26）：闲聊 voice pass 的提示词锚点与空/异常回落。"""

    HISTORY = [{"role": "user", "name": "小明", "content": "上次聊到吉他"}]
    GROUP = [{"name": "小红", "text": "今天天气好"}]
    PLAN = {"mode": "chat", "reply": "在的在的"}

    def _run(self, coro):
        return asyncio.run(coro)

    def _call(self, return_value="好呀。", raises=None, plan=None, **overrides):
        captured = {}

        async def fake_ask(prompt, max_chars=35):
            captured["prompt"] = prompt
            captured["max_chars"] = max_chars
            if raises is not None:
                raise raises
            return return_value

        kwargs = {
            "summary": "群里在聊演出",
            "experience_hint": "经验提示",
            "search_hint": "【公开网页搜索摘要】\n- 结果一",
            "glossary_hint": (
                "【群内黑话】（群内用法优先于搜索标题；不确定就短句问，不要硬套）\n"
                "- 瑞萍 = 锐评（谐音）"
            ),
            "action_note": "先看群内黑话和公开搜索材料再回答。",
            "max_chars": 100,
            "draft": "草稿内容",
            "quoted_text": "被引用的原话",
        }
        kwargs.update(overrides)
        plan_obj = dict(self.PLAN if plan is None else plan)
        result = self._run(
            agent_core.compose_chat_reply(
                "在吗", "小明", self.HISTORY, self.GROUP, plan_obj, fake_ask,
                **kwargs,
            )
        )
        return result, captured

    def test_prompt_contains_required_anchors(self):
        _result, captured = self._call()
        prompt = captured["prompt"]
        for anchor in (
            "最终出声环节", "严格按角色卡", "【主脑草稿】", "【当前消息】",
            "【群内黑话】", "【本轮动作边界】", "本轮边界", "不要 OOC",
            "【公开搜索参考】", "【群聊摘要】", "【经验提示】",
            "【群聊背景】", "【你与这位群友的对话】",
            "被引用的原话", "草稿内容", "小明：上次聊到吉他", "小红：今天天气好",
            "不写分析", "说完整就停", "不复读",
        ):
            self.assertIn(anchor, prompt, msg=anchor)

    def test_search_block_absent_when_hint_empty(self):
        _result, captured = self._call(search_hint="")
        self.assertNotIn("【公开搜索参考】", captured["prompt"])

    def test_action_block_absent_when_note_empty(self):
        _result, captured = self._call(action_note="")
        self.assertNotIn("【本轮动作边界】（本轮边界以此为准）", captured["prompt"])

    def test_max_chars_reaches_prompt_and_model(self):
        _result, captured = self._call(max_chars=35)
        self.assertIn("35个字符", captured["prompt"])
        self.assertEqual(captured["max_chars"], 35)
        _result, captured = self._call(max_chars=100)
        self.assertIn("100个字符", captured["prompt"])
        self.assertEqual(captured["max_chars"], 100)

    def test_empty_voice_falls_back_to_draft(self):
        result, _captured = self._call(return_value="   ")
        self.assertEqual(result, "草稿内容")

    def test_exception_falls_back_to_draft(self):
        result, _captured = self._call(raises=RuntimeError("voice down"))
        self.assertEqual(result, "草稿内容")

    def test_empty_draft_and_empty_voice_returns_empty(self):
        result, _captured = self._call(return_value="", draft="", plan={"reply": ""})
        self.assertEqual(result, "")

    def test_draft_defaults_to_plan_reply(self):
        result, captured = self._call(return_value="", draft="")
        self.assertEqual(result, "在的在的")
        self.assertIn("在的在的", captured["prompt"])

    def test_success_returns_voice_and_logs_used_yes(self):
        with mock.patch("builtins.print") as fake_print:
            result, _captured = self._call(return_value="  好呀，一起去练琴。 ")
        self.assertEqual(result, "好呀，一起去练琴。")
        printed = "".join(str(call.args[0]) for call in fake_print.call_args_list)
        self.assertIn("[ANON VOICE] draft=4 used=yes", printed)

    def test_empty_voice_logs_used_no(self):
        with mock.patch("builtins.print") as fake_print:
            self._call(return_value="")
        printed = "".join(str(call.args[0]) for call in fake_print.call_args_list)
        self.assertIn("[ANON VOICE] draft=4 used=no", printed)


class QuickSearchMaterialsTest(unittest.TestCase):
    """WP2（2026-10-26）：闲聊确定性快搜最多两条、第一条命中即停、失败不阻塞。"""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_first_hit_stops_second_query(self):
        found = "【公开网页搜索摘要】\n- 结果一：细节"
        with mock.patch.object(anonbot.agent_search, "search", return_value=found) as fake:
            out = self._run(anonbot._quick_search_materials(["q1", "q2"]))
        self.assertEqual(out, found)
        fake.assert_called_once_with("q1", 4)

    def test_second_query_used_when_first_empty(self):
        found = "【公开网页搜索摘要】\n- 结果二：细节"
        fake_search = mock.Mock(side_effect=[None, found])
        with mock.patch.object(anonbot.agent_search, "search", fake_search):
            out = self._run(anonbot._quick_search_materials(["q1", "q2"]))
        self.assertEqual(out, found)
        self.assertEqual(fake_search.call_count, 2)

    def test_failure_then_success_does_not_raise(self):
        found = "【公开网页搜索摘要】\n- 结果二：细节"
        fake_search = mock.Mock(side_effect=[RuntimeError("boom"), found])
        with mock.patch.object(anonbot.agent_search, "search", fake_search):
            out = self._run(anonbot._quick_search_materials(["q1", "q2"]))
        self.assertEqual(out, found)

    def test_circuit_open_skips_search(self):
        with mock.patch.object(anonbot, "_circuit_open", return_value=True), \
                mock.patch.object(anonbot.agent_search, "search") as fake_search:
            self.assertEqual(self._run(anonbot._quick_search_materials(["q1"])), "")
            fake_search.assert_not_called()

    def test_all_empty_returns_empty(self):
        with mock.patch.object(anonbot.agent_search, "search", return_value=None):
            self.assertEqual(self._run(anonbot._quick_search_materials(["q1", "q2"])), "")

    def test_at_most_two_queries_attempted(self):
        with mock.patch.object(anonbot.agent_search, "search", return_value=None) as fake:
            self._run(anonbot._quick_search_materials(["q1", "q2", "q3"]))
        self.assertEqual(fake.call_count, 2)

    def test_log_line_has_query_hits_and_chars(self):
        found = "【公开网页搜索摘要】\n- 结果一：细节\n- 结果二：细节"
        with mock.patch.object(anonbot.agent_search, "search", return_value=found), \
                mock.patch("builtins.print") as fake_print:
            self._run(anonbot._quick_search_materials(["瑞萍"]))
        printed = "".join(str(call.args[0]) for call in fake_print.call_args_list)
        self.assertIn(f"[ANON QUICK SEARCH] q=瑞萍 hits=2 used chars={len(found)}", printed)

    def test_query_control_chars_flattened_in_log(self):
        with mock.patch.object(anonbot.agent_search, "search", return_value=None), \
                mock.patch("builtins.print") as fake_print:
            self._run(anonbot._quick_search_materials(["q1\n[ANON FAKE]"]))
        lines = [str(call.args[0]) for call in fake_print.call_args_list if call.args]
        mine = [line for line in lines if line.startswith("[ANON QUICK SEARCH] q=")]
        self.assertEqual(len(mine), 1)
        self.assertNotIn("\n", mine[0])
        self.assertIn("[ANON FAKE]", mine[0])


class ActionFallbackTest(unittest.TestCase):
    """WP2（2026-10-26）：受限动作的固定拒绝短句。"""

    def test_covers_restricted_kinds(self):
        for kind in ("send_media", "contact_third_party", "purchase_login"):
            self.assertIn(kind, anonbot.ACTION_FALLBACK_REPLIES, msg=kind)

    def test_each_within_chat_limit(self):
        for kind, text in anonbot.ACTION_FALLBACK_REPLIES.items():
            self.assertTrue(text.strip(), msg=kind)
            self.assertLessEqual(len(text), anonbot.MAX_REPLY_CHARS, msg=kind)
            self.assertLessEqual(len(text), 35, msg=kind)

    def test_no_fake_completion_words(self):
        for kind, text in anonbot.ACTION_FALLBACK_REPLIES.items():
            for word in ("已发", "已抢", "已登录", "已经发", "已经抢", "已经登录"):
                self.assertNotIn(word, text, msg=f"{kind}:{word}")


class FakeActionClaimTest(unittest.TestCase):
    """2026-10-26 评审修复：自然假承诺必须命中，固定拒绝句不能自我触发。"""

    def test_evidence_phrases_are_claims(self):
        cases = (
            ("我发张图给你看看", "send_media"),
            ("我发你一张", "send_media"),
            ("那我去签到啦", "purchase_login"),
            ("帮你转告一下", "contact_third_party"),
            ("我替你带话过去", "contact_third_party"),
            ("好，我去抢票了", "purchase_login"),
            ("已经发给你了", "send_media"),
            ("这就发你", "send_media"),
            ("我唱给你听", "send_media"),
            ("我已经告诉他了", "contact_third_party"),
        )
        for text, kind in cases:
            with self.subTest(text=text):
                self.assertTrue(anonbot._claims_fake_action(text, kind))

    def test_fixed_refusals_are_not_claims(self):
        for kind, text in anonbot.ACTION_FALLBACK_REPLIES.items():
            with self.subTest(kind=kind):
                self.assertFalse(anonbot._claims_fake_action(text, kind))

    def test_denials_and_discourse_are_not_claims(self):
        cases = (
            ("我发不了图", "send_media"),
            ("我不能替你转告", "contact_third_party"),
            ("我跟你说", "contact_third_party"),
            ("买票要抽签啊", "purchase_login"),
        )
        for text, kind in cases:
            with self.subTest(text=text):
                self.assertFalse(anonbot._claims_fake_action(text, kind))


class VoiceWiringSourceAnchorTest(unittest.TestCase):
    """WP2（2026-10-26）：闲聊 voice pass 已接线，任务链路/触发/熔断保持原样。"""

    SRC = Path(__file__).resolve().parents[1] / "src" / "plugins"

    def _source(self, name: str) -> str:
        return (self.SRC / name).read_text(encoding="utf-8")

    def test_chat_voice_anchors_present(self):
        src = self._source("anonbot.py")
        for anchor in (
            "anon_search_policy", "compose_chat_reply", "[ANON INTENT]",
            "[ANON VOICE]", "_quick_search_materials", "pick_face_action",
            "action_note", "ACTION_FALLBACK_REPLIES",
        ):
            self.assertIn(anchor, src, msg=anchor)

    def test_task_chain_unchanged(self):
        src = self._source("anonbot.py")
        for anchor in (
            "我看看，稍等一下～", "[ANON SENT] task acknowledgement",
            "run_subagent", "answer_simple_task", "MAX_TASK_REPLY_CHARS",
        ):
            self.assertIn(anchor, src, msg=anchor)

    def test_trigger_and_circuit_unchanged(self):
        src = self._source("anonbot.py")
        for anchor in (
            "_is_triggered", '"at-raw"', '"reply"', '"name"',
            "_circuit_open", "CIRCUIT_MAX_TOKENS",
        ):
            self.assertIn(anchor, src, msg=anchor)

    def test_chat_branch_before_task_branch(self):
        src = self._source("anonbot.py")
        self.assertLess(src.index("compose_chat_reply"), src.index("answer_simple_task"))

    def test_voice_rules_anchors(self):
        rules = agent_core.VOICE_RULES
        for anchor in ("不要 OOC", "不写分析", "不复读", "说完整就停", "主脑草稿",
                       "拒绝也是台词，不是规则声明"):
            self.assertIn(anchor, rules, msg=anchor)


class RoleCardTest(unittest.TestCase):
    """WP2（2026-10-26）：角色卡词表与能做/做不到边界。"""

    ROLE = Path(__file__).resolve().parents[1] / "roles" / "Anon.md"

    def _text(self) -> str:
        return self.ROLE.read_text(encoding="utf-8")

    def test_glossary_rows_present(self):
        text = self._text()
        for term in (
            "tomorin = 高松灯", "凑企鹅 = 高松灯", "苏幽璃 = 长崎素世",
            "疯串箱子", "疯穿箱子", "疯氚箱子", "母鸡卡 = Ave Mujica",
            "邦娃", "棉娃", "瑞萍 = 锐评", "zmd = 明日方舟：终末地",
            "立石凛 = 爱音的声优", "李世林", "立凛石", "立戒石", "江界石",
            "羊宫妃那 = 高松灯的声优", "青木阳菜 = 要乐奈的声优",
        ):
            self.assertIn(term, text, msg=term)

    def test_boundary_section_present(self):
        text = self._text()
        for token in (
            "【能做到 / 做不到】", "发图片", "抢票", "替人发消息",
            "登录", "付款", "签到", "先看【群内黑话】", "以群内用法为准",
        ):
            self.assertIn(token, text, msg=token)

    def test_hard_rules_still_present(self):
        text = self._text()
        for token in ("不超过35字", "一到两短句", "说完整就停", "口头禅", "不编"):
            self.assertIn(token, text, msg=token)

    def test_refusal_voice_rule_present(self):
        # 2026-10-26 用户指令：「我不接」类系统腔不许出现，拒绝也要用角色语气。
        text = self._text()
        for token in ("拒绝是台词，不是声明", "我不接", "我不参与", "服务范围"):
            self.assertIn(token, text, msg=token)

    def test_greeting_not_revived(self):
        src = (Path(__file__).resolve().parents[1] / "src" / "plugins" / "anonbot.py").read_text(encoding="utf-8")
        self.assertNotIn("ensure_greeting_sep", src)


class _FakePrivateEvent:
    def __init__(self, text: str, user_id: str = "70001"):
        self.user_id = int(user_id)
        self.message_type = "private"
        self.raw_message = text
        self.message_id = 1
        self.self_id = "10000"
        self.sender = None
        self.reply = None
        self._text = text

    def get_message(self):
        from nonebot.adapters.onebot.v11 import MessageSegment

        return [MessageSegment.text(self._text)]

    def is_tome(self):
        return True


class ChatWiringTest(unittest.TestCase):
    """WP2（2026-10-26）：handle_message 闲聊分支的 voice 回落与动作覆盖。"""

    def _run_handler(self, text, plan, compose, user_id="70001", materials=None,
                     capture=None, clock=None, compose_delay=120.0):
        sent = []
        calls = {"compose": 0}

        async def fake_send(bot, event, answer, face=None):
            sent.append({"text": answer, "face": face})

        async def fake_understand(*args, **kwargs):
            return None if plan is None else dict(plan)

        async def fake_compose(*args, **kwargs):
            calls["compose"] += 1
            if capture is not None:
                capture.update(kwargs)
            if clock is not None:
                clock["now"] += compose_delay  # 模拟 voice/搜索耗时
            if isinstance(compose, Exception):
                raise compose
            return compose

        event = _FakePrivateEvent(text, user_id)
        key = ("private:" + user_id, user_id)
        anonbot.histories.pop(key, None)
        patches = [
            mock.patch.object(anonbot.agent_core, "understand", fake_understand),
            mock.patch.object(anonbot.agent_core, "compose_chat_reply", fake_compose),
            mock.patch.object(anonbot, "_send_reply", fake_send),
            mock.patch.object(anonbot, "_log_sent", lambda *a, **k: None),
            mock.patch.object(anonbot.experience, "render_hint", lambda *a, **k: ""),
        ]
        if clock is not None:
            patches.append(
                mock.patch.object(anonbot.time, "time", lambda: clock["now"])
            )
        if materials is not None:
            async def fake_materials(queries):
                calls["queries"] = list(queries)
                return materials

            patches.append(
                mock.patch.object(anonbot, "_quick_search_materials", fake_materials)
            )
        for patch in patches:
            patch.start()
        try:
            asyncio.run(anonbot.handle_message(FakeBot(), event))
        finally:
            for patch in reversed(patches):
                patch.stop()
            anonbot.histories.pop(key, None)
            anonbot.scope_locks.pop("private:" + user_id, None)
        return sent, calls

    def _chat_plan(self, **fields):
        plan = {
            "mode": "chat", "reply": "在的呀，刚练完吉他。", "agent": "none",
            "need": "", "steps": [], "complex": False, "fresh": False, "act": "none",
        }
        plan.update(fields)
        return plan

    def test_voice_failure_sends_planner_draft(self):
        sent, calls = self._run_handler(
            "在吗", self._chat_plan(), RuntimeError("voice down"), user_id="70011"
        )
        self.assertEqual(calls["compose"], 1)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["text"], "在的呀，刚练完吉他。")
        self.assertIsNone(sent[0]["face"])

    def test_voice_success_text_is_sent(self):
        sent, _calls = self._run_handler(
            "在吗", self._chat_plan(), "在的，刚弄完吉他～", user_id="70012"
        )
        self.assertEqual(sent[0]["text"], "在的，刚弄完吉他～")

    def test_send_face_overrides_planner_act(self):
        sent, _calls = self._run_handler(
            "卖个萌",
            self._chat_plan(reply="好啦好啦。", act="face:heart"),
            "好啦，就这一次哦～",
            user_id="70013",
        )
        self.assertEqual(sent[0]["text"], "好啦，就这一次哦～")
        self.assertEqual(sent[0]["face"], agent_core.FACE_ACTIONS["shy"])

    def test_send_face_defaults_to_smile(self):
        sent, _calls = self._run_handler(
            "发个表情", self._chat_plan(reply="给。"), "给你一个～", user_id="70014"
        )
        self.assertEqual(sent[0]["face"], agent_core.FACE_ACTIONS["smile"])

    def test_send_media_forces_no_face_and_uses_fallback(self):
        sent, _calls = self._run_handler(
            "发张图",
            self._chat_plan(reply="", act="face:heart"),
            "",
            user_id="70015",
        )
        self.assertEqual(sent[0]["text"], anonbot.ACTION_FALLBACK_REPLIES["send_media"])
        self.assertIsNone(sent[0]["face"])

    def test_contact_third_party_forces_no_face(self):
        sent, _calls = self._run_handler(
            "帮我转告小明明天开会",
            self._chat_plan(reply="", act="face:heart"),
            "",
            user_id="70016",
        )
        self.assertEqual(
            sent[0]["text"], anonbot.ACTION_FALLBACK_REPLIES["contact_third_party"]
        )
        self.assertIsNone(sent[0]["face"])

    def test_purchase_login_forces_no_face(self):
        sent, _calls = self._run_handler(
            "帮我登录一下账号",
            self._chat_plan(reply="", act="face:heart"),
            "",
            user_id="70017",
        )
        self.assertEqual(
            sent[0]["text"], anonbot.ACTION_FALLBACK_REPLIES["purchase_login"]
        )
        self.assertIsNone(sent[0]["face"])

    def test_intent_log_line_emitted(self):
        with mock.patch("builtins.print") as fake_print:
            self._run_handler("在吗", self._chat_plan(), "在的。", user_id="70018")
        printed = "".join(
            str(call.args[0]) for call in fake_print.call_args_list if call.args
        )
        self.assertIn("[ANON INTENT] kind=none lookup=none queries=-", printed)

    def test_definition_with_glossary_skips_search(self):
        fake_materials = mock.AsyncMock(return_value="")
        with mock.patch.object(anonbot, "_quick_search_materials", fake_materials), \
                mock.patch("builtins.print") as fake_print:
            self._run_handler(
                "谁是tomorin",
                self._chat_plan(reply="tomorin 就是灯啦。"),
                "tomorin 就是高松灯呀。",
                user_id="70019",
            )
        fake_materials.assert_called_once_with([])
        printed = "".join(
            str(call.args[0]) for call in fake_print.call_args_list if call.args
        )
        self.assertIn("[ANON INTENT] kind=search lookup=definition queries=-", printed)

    def test_explicit_lookup_searches_and_uses_search_limit(self):
        found = "【公开网页搜索摘要】\n- 幽灵：群里常说的梗"
        capture = {}
        sent, calls = self._run_handler(
            "搜幽灵",
            self._chat_plan(reply="我查一下。"),
            "查到了，幽灵是群里常说的梗啦。",
            user_id="70020",
            materials=found,
            capture=capture,
        )
        self.assertEqual(calls["queries"], ["幽灵", "幽灵 是什么"])
        self.assertEqual(capture["search_hint"], found)
        self.assertEqual(capture["max_chars"], anonbot.SEARCH_REPLY_CHARS)
        self.assertEqual(sent[0]["text"], "查到了，幽灵是群里常说的梗啦。")

    def test_plain_chat_does_not_search(self):
        fake_materials = mock.AsyncMock(return_value="")
        with mock.patch.object(anonbot, "_quick_search_materials", fake_materials):
            self._run_handler("在吗", self._chat_plan(), "在的。", user_id="70021")
        fake_materials.assert_called_once_with([])

    def test_plan_none_sends_fallback_without_nameerror(self):
        with mock.patch.object(
            anonbot, "_ask_chat", mock.AsyncMock(return_value="兜底回复")
        ):
            sent, calls = self._run_handler(
                "在吗", None, "不该使用", user_id="70022"
            )
        self.assertEqual(calls["compose"], 0)
        self.assertEqual(sent[0]["text"], "兜底回复")

    def test_stale_message_not_sent(self):
        times = {"n": 0}

        def fake_time():
            times["n"] += 1
            return 1000.0 if times["n"] == 1 else 1200.0

        with mock.patch.object(anonbot.time, "time", fake_time):
            sent, calls = self._run_handler(
                "在吗", self._chat_plan(), "在的。", user_id="70023"
            )
        self.assertEqual(sent, [])
        self.assertEqual(calls["compose"], 0)

    def test_fake_send_promise_replaced_with_boundary_reply(self):
        sent, _calls = self._run_handler(
            "发张图",
            self._chat_plan(reply="已经发给你了。", act="face:heart"),
            "已经发给你了。",
            user_id="70024",
        )
        self.assertEqual(
            sent[0]["text"], anonbot.ACTION_FALLBACK_REPLIES["send_media"]
        )
        self.assertIsNone(sent[0]["face"])

    def test_voice_timeout_falls_back_to_draft(self):
        sent, _calls = self._run_handler(
            "在吗",
            self._chat_plan(reply="在的呀，刚练完吉他。"),
            asyncio.TimeoutError(),
            user_id="70025",
        )
        self.assertEqual(sent[0]["text"], "在的呀，刚练完吉他。")

    def test_task_answer_not_dropped_by_generation_time(self):
        clock = {"now": 1000.0}

        def fake_time():
            return clock["now"]

        async def fake_task(*args, **kwargs):
            # 复杂任务允许两轮 240s；生成耗时再长，交付也必须发出。
            clock["now"] = 1500.0
            return "任务答案"

        plan = {
            "mode": "task", "reply": "", "agent": "solver",
            "need": "算一下", "steps": [], "complex": False,
            "fresh": False, "act": "none",
        }
        with mock.patch.object(anonbot.time, "time", fake_time), \
                mock.patch.object(
                    anonbot.agent_core, "answer_simple_task", fake_task
                ):
            sent, _calls = self._run_handler(
                "帮我算一下", plan, "不该使用", user_id="70026"
            )
        self.assertEqual(sent[0]["text"], "我看看，稍等一下～")
        self.assertEqual(sent[1]["text"], "任务答案")

    def test_fake_promise_variants_replaced_with_boundary_reply(self):
        cases = (
            ("发张图", "我发张图给你看看", "send_media", "70027"),
            ("发语音", "我发你一条语音", "send_media", "70028"),
            ("帮我签到", "那我去签到啦", "purchase_login", "70029"),
            ("帮我转告小明", "帮你转告一下", "contact_third_party", "70030"),
        )
        for user_text, fake_reply, kind, user_id in cases:
            with self.subTest(text=user_text):
                sent, _calls = self._run_handler(
                    user_text,
                    self._chat_plan(reply="", act="face:heart"),
                    fake_reply,
                    user_id=user_id,
                )
                self.assertEqual(
                    sent[0]["text"], anonbot.ACTION_FALLBACK_REPLIES[kind]
                )
                self.assertIsNone(sent[0]["face"])

    def test_queue_stale_dropped_after_lock(self):
        times = {"n": 0}

        def fake_time():
            times["n"] += 1
            return 1000.0 if times["n"] <= 2 else 1200.0

        with mock.patch.object(anonbot.time, "time", fake_time), \
                mock.patch("builtins.print") as fake_print:
            sent, calls = self._run_handler(
                "在吗", self._chat_plan(), "在的。", user_id="70031"
            )
        self.assertEqual(sent, [])
        self.assertEqual(calls["compose"], 0)
        printed = "".join(
            str(call.args[0]) for call in fake_print.call_args_list if call.args
        )
        self.assertIn("stage=after_lock", printed)

    def test_chat_answer_after_stale_window_is_still_sent(self):
        clock = {"now": 1000.0}
        sent, calls = self._run_handler(
            "在吗", self._chat_plan(), "慢但有效的答案。",
            user_id="70032", clock=clock,
        )
        self.assertEqual(calls["compose"], 1)
        self.assertEqual(sent[0]["text"], "慢但有效的答案。")
        self.assertEqual(clock["now"], 1120.0)

    def test_chat_answer_after_too_stale_window_is_dropped(self):
        clock = {"now": 1000.0}
        sent, calls = self._run_handler(
            "在吗", self._chat_plan(), "拖太久的答案。",
            user_id="70034", clock=clock, compose_delay=180.0,
        )
        self.assertEqual(calls["compose"], 1)
        self.assertEqual(sent, [])
        self.assertEqual(clock["now"], 1180.0)

    def test_plan_none_restricted_gets_boundary_and_fixed_reply(self):
        captured = {}

        async def fake_ask(prompt, max_chars=35, timeout=None):
            captured["prompt"] = prompt
            return "我发张图给你看看"

        with mock.patch.object(anonbot, "_ask_chat", fake_ask):
            sent, _calls = self._run_handler(
                "发张图", None, "不该使用", user_id="70033"
            )
        self.assertIn("【本轮动作边界】", captured["prompt"])
        self.assertIn("不能发图片", captured["prompt"])
        self.assertEqual(
            sent[0]["text"], anonbot.ACTION_FALLBACK_REPLIES["send_media"]
        )

    def test_task_restricted_passes_boundary_and_suppresses_face(self):
        captured = {}

        async def fake_task(*args, **kwargs):
            captured.update(kwargs)
            return "这类事我做不来。"

        plan = {
            "mode": "task", "reply": "", "agent": "solver",
            "need": "抢票", "steps": [], "complex": False,
            "fresh": False, "act": "face:heart",
        }
        with mock.patch.object(anonbot.agent_core, "answer_simple_task", fake_task):
            sent, _calls = self._run_handler(
                "帮我抢票", plan, "不该使用", user_id="70034"
            )
        self.assertIn("抢票", captured.get("action_note", ""))
        self.assertEqual(sent[0]["text"], "我看看，稍等一下～")
        self.assertEqual(sent[1]["text"], "这类事我做不来。")
        self.assertIsNone(sent[1]["face"])

    def test_creative_task_not_replaced_by_media_refusal(self):
        plan = {
            "mode": "task", "reply": "", "agent": "writer",
            "need": "写歌词", "steps": [], "complex": False,
            "fresh": False, "act": "none",
        }

        async def fake_task(*args, **kwargs):
            return "我们一起唱歌吧，把烦恼都唱跑。"

        with mock.patch.object(
            anonbot.agent_core, "answer_simple_task", fake_task
        ):
            sent, _calls = self._run_handler(
                "帮我写一段唱歌的歌词", plan, "不该使用", user_id="70035"
            )
        self.assertEqual(sent[0]["text"], "我看看，稍等一下～")
        self.assertEqual(sent[1]["text"], "我们一起唱歌吧，把烦恼都唱跑。")

    def test_fake_action_does_not_rewrite_refusal_or_text_alternative(self):
        cases = (
            ("唱一段", "我唱歌跑调啦，还是聊天吧", "70036"),
            ("发张图", "我发文字给你描述一下", "70037"),
        )
        for user_text, answer, user_id in cases:
            with self.subTest(text=user_text):
                sent, _calls = self._run_handler(
                    user_text, self._chat_plan(reply=""), answer,
                    user_id=user_id,
                )
                self.assertEqual(sent[0]["text"], answer)

    def test_fake_action_still_catches_explicit_promise(self):
        sent, _calls = self._run_handler(
            "发张图",
            self._chat_plan(reply=""),
            "我发张图给你看看",
            user_id="70038",
        )
        self.assertEqual(
            sent[0]["text"], anonbot.ACTION_FALLBACK_REPLIES["send_media"]
        )


class TaskBoundaryNoteTest(unittest.TestCase):
    """2026-10-26 评审修复：任务链路（含 researcher）也要收到动作边界。"""

    def test_solver_prompt_contains_action_note(self):
        captured = []

        async def fake_ask(prompt, max_chars):
            captured.append(prompt)
            return "答案" if len(captured) == 1 else ""

        plan = {
            "agent": "solver", "need": "抢票", "steps": [], "complex": True,
        }
        result = asyncio.run(
            agent_core.run_subagent(
                plan, "帮我抢票", "小明", [], fake_ask,
                action_note="你不能抢票、登录或付款。",
            )
        )
        self.assertEqual(result, "答案")
        self.assertIn("本轮动作边界：你不能抢票", captured[0])

    def test_researcher_prompt_contains_action_note(self):
        captured = []

        async def fake_ask(prompt, max_chars):
            captured.append(prompt)
            if len(captured) == 1:
                return '{"queries":["x"]}'
            return "答案"

        plan = {"agent": "researcher", "need": "查票", "steps": [], "complex": True}
        with mock.patch.object(agent_core.agent_search, "search", return_value=None):
            result = asyncio.run(
                agent_core.run_subagent(
                    plan, "查票", "小明", [], fake_ask,
                    action_note="你不能发图片。",
                )
            )
        self.assertEqual(result, "答案")
        self.assertIn("本轮动作边界：你不能发图片", captured[-1])


if __name__ == "__main__":
    unittest.main()
