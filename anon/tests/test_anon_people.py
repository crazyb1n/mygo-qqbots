# -*- coding: utf-8 -*-
"""Offline tests for the persistent per-user profile module (anon_people).

No NoneBot, no network: pure stdlib logic + prompt-anchor assertions.
"""
import asyncio
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plugins import agent_core
from src.plugins import anon_context
from src.plugins import anon_people
from src.plugins import anonbot


class UidTailTest(unittest.TestCase):
    def test_tail(self):
        self.assertEqual(anon_people.uid_tail("200000011"), "0011")

    def test_short_or_empty(self):
        self.assertEqual(anon_people.uid_tail("123"), "123")
        self.assertEqual(anon_people.uid_tail(""), "")


class UpdateTest(unittest.TestCase):
    def test_new_user(self):
        users = {}
        status = anon_people.update(
            users, "200000011", "群友A", "你好呀", ts=1793000000
        )
        self.assertEqual(status, "new_user")
        profile = users["200000011"]
        self.assertEqual(profile["count"], 1)
        self.assertEqual(profile["last_msg"], "你好呀")
        self.assertIn("群友A", profile["names"])
        self.assertEqual(profile["first"], profile["last"])

    def test_repeat_message_increments(self):
        users = {}
        anon_people.update(users, "1", "A", "hi", ts=100)
        status = anon_people.update(users, "1", "A", "again", ts=200)
        self.assertEqual(status, "")
        self.assertEqual(users["1"]["count"], 2)
        self.assertEqual(users["1"]["names"]["A"]["count"], 2)
        self.assertEqual(users["1"]["last_msg"], "again")

    def test_name_change_keeps_both_and_reports(self):
        users = {}
        anon_people.update(users, "1", "旧名", "a", ts=100)
        status = anon_people.update(users, "1", "新名", "b", ts=200)
        self.assertEqual(status, "new_name")
        self.assertEqual(set(users["1"]["names"]), {"旧名", "新名"})

    def test_names_capped(self):
        users = {}
        for index in range(anon_people.MAX_NAMES + 2):
            anon_people.update(users, "1", f"n{index}", "x", ts=100 + index)
        self.assertEqual(len(users["1"]["names"]), anon_people.MAX_NAMES)
        self.assertNotIn("n0", users["1"]["names"])
        self.assertIn(f"n{anon_people.MAX_NAMES + 1}", users["1"]["names"])

    def test_control_chars_flattened(self):
        users = {}
        anon_people.update(users, "1", "A\n[ANON X]", "说\n话\x00", ts=100)
        self.assertNotIn("\n", "".join(users["1"]["names"]))
        self.assertNotIn("\n", users["1"]["last_msg"])
        self.assertNotIn("\x00", users["1"]["last_msg"])

    def test_last_msg_truncated(self):
        users = {}
        anon_people.update(users, "1", "A", "长" * 300, ts=100)
        self.assertEqual(len(users["1"]["last_msg"]), anon_people.LAST_MSG_CHARS)

    def test_empty_text_keeps_previous_last_msg(self):
        # 只 @ 不说话（图文消息被清空）时不能把上次的话抹掉。
        users = {}
        anon_people.update(users, "1", "A", "上次的话", ts=100)
        anon_people.update(users, "1", "A", "", ts=200)
        self.assertEqual(users["1"]["last_msg"], "上次的话")
        self.assertEqual(users["1"]["count"], 2)

    def test_empty_uid_ignored(self):
        users = {}
        self.assertEqual(anon_people.update(users, "", "A", "x"), "")
        self.assertEqual(users, {})


class PruneTest(unittest.TestCase):
    def test_over_cap_evicts_oldest_last(self):
        users = {}
        for index in range(6):
            anon_people.update(users, str(index), "A", "x", ts=100 + index)
        anon_people.prune(users, max_users=3)
        self.assertEqual(set(users), {"3", "4", "5"})


class LoadUsersTest(unittest.TestCase):
    def test_valid_payload_roundtrip(self):
        users = {}
        anon_people.update(users, "1", "A", "hi", ts=100)
        loaded = anon_people.load_users({"group_summaries": {}, "users": users})
        self.assertEqual(loaded["1"]["count"], 1)
        self.assertEqual(loaded["1"]["last_msg"], "hi")

    def test_missing_or_broken_payload(self):
        for payload in (None, {}, {"users": None}, {"users": []}, "x"):
            with self.subTest(payload=payload):
                self.assertEqual(anon_people.load_users(payload), {})

    def test_bad_profiles_dropped(self):
        payload = {"users": {"1": "x", "2": {"count": "7", "names": "bad"}, "": {}}}
        loaded = anon_people.load_users(payload)
        self.assertEqual(set(loaded), {"2"})
        self.assertEqual(loaded["2"]["count"], 7)
        self.assertEqual(loaded["2"]["names"], {})


class RenderTest(unittest.TestCase):
    def test_unknown_returns_empty(self):
        self.assertEqual(anon_people.render(None, "1", "A"), "")
        self.assertEqual(anon_people.render({}, "1", "A"), "")

    def test_known_contains_identity_facts(self):
        users = {}
        anon_people.update(users, "200000011", "群友A", "你好呀", ts=1793000000)
        block = anon_people.render(users["200000011"], "200000011", "群友A")
        self.assertIn("【当前用户】群友A", block)
        self.assertIn("…0011", block)
        self.assertIn("已见过 1 条消息", block)
        self.assertIn("你好呀", block)
        self.assertIn("别装作初次见面", block)

    def test_name_falls_back_to_latest_known(self):
        users = {}
        anon_people.update(users, "1", "旧名", "a", ts=100)
        anon_people.update(users, "1", "新名", "b", ts=200)
        block = anon_people.render(users["1"], "1", "")
        self.assertIn("新名", block)

    def test_name_truncated_and_no_newlines(self):
        users = {}
        long_name = "名" * 100
        anon_people.update(users, "1", long_name, "x", ts=100)
        block = anon_people.render(users["1"], "1", long_name)
        self.assertNotIn("\n", block)
        self.assertNotIn("名" * (anon_people.NAME_CHARS + 1), block)


class PromptInjectionTest(unittest.TestCase):
    BLOCK = "【当前用户】群友A（QQ …0011）已见过 1 条消息。这是你认识的群友，别装作初次见面。"

    def test_build_chat_prompt_has_profile_block(self):
        prompt = anon_context.build_chat_prompt(
            [], [], "群友A", "在吗", "", False, "", 35,
            user_profile=self.BLOCK,
        )
        self.assertIn("【当前用户】群友A", prompt)

    def test_build_chat_prompt_unchanged_without_profile(self):
        prompt = anon_context.build_chat_prompt(
            [], [], "群友A", "在吗", "", False, "", 35
        )
        self.assertNotIn("【当前用户】", prompt)

    def test_understand_injects_profile(self):
        captured = {}

        async def fake_ask(prompt, limit):
            captured["prompt"] = prompt
            return '{"mode":"chat","reply":"在呀","agent":"none","need":"","steps":[],"complex":false,"fresh":false,"act":"none"}'

        asyncio.run(
            agent_core.understand(
                "你还记得我吗", "群友A", [], [], fake_ask,
                user_profile=self.BLOCK,
            )
        )
        self.assertIn("【当前用户】群友A", captured["prompt"])
        self.assertIn("…0011", captured["prompt"])
        # P2-2 回归：带档案时仍只能有一行【当前用户】。
        self.assertEqual(captured["prompt"].count("【当前用户】"), 1)

    def test_compose_injects_profile(self):
        captured = {}

        async def fake_ask(prompt, limit):
            captured["prompt"] = prompt
            return "记得呀"

        answer = asyncio.run(
            agent_core.compose_chat_reply(
                "你还记得我吗", "群友A", [], [], {"reply": "记得"}, fake_ask,
                user_profile=self.BLOCK,
            )
        )
        self.assertEqual(answer, "记得呀")
        self.assertIn("【当前用户】群友A", captured["prompt"])

    def test_voice_rules_forbid_sticky_nickname(self):
        self.assertIn("外号", agent_core.VOICE_RULES)

    def test_role_card_has_address_rule(self):
        card = (
            Path(__file__).resolve().parents[1] / "roles" / "Anon.md"
        ).read_text(encoding="utf-8")
        self.assertIn("群名片", card)
        self.assertIn("外号", card)

    def test_zero_width_and_bidi_stripped(self):
        users = {}
        anon_people.update(users, "1", "A\u200bB", "x\u202ey\u2060", ts=100)
        anon_people.update(users, "2", "C\u2066D\u00ad", "z\u180e", ts=101)
        blob = json.dumps(users, ensure_ascii=False)
        for ch in ("\u200b", "\u202e", "\u2060", "\u2066", "\u00ad", "\u180e"):
            self.assertNotIn(ch, blob)

    def test_placeholder_text_keeps_previous_last_msg(self):
        users = {}
        anon_people.update(users, "1", "A", "真的话", ts=100)
        anon_people.update(users, "1", "A", "[图片]", ts=200)
        anon_people.update(users, "1", "A", "[图片] [表情]", ts=300)
        self.assertEqual(users["1"]["last_msg"], "真的话")

    def test_load_clears_legacy_placeholder_last_msg(self):
        loaded = anon_people.load_users(
            {"users": {"1": {"last_msg": "[图片]", "last": "2026-01-01"}}}
        )
        self.assertEqual(loaded["1"]["last_msg"], "")

    def test_prune_protects_current_uid(self):
        users = {}
        for index in range(4):
            anon_people.update(users, str(index), "A", "x", ts=100 + index)
        anon_people.prune(users, max_users=2, protect="0")
        self.assertIn("0", users)
        self.assertEqual(len(users), 2)

    def test_load_applies_name_cap(self):
        names = {
            f"n{index}": {"last": f"2026-01-{index + 1:02d}"}
            for index in range(anon_people.MAX_NAMES + 3)
        }
        loaded = anon_people.load_users(
            {"users": {"1": {"names": names, "last": "2026-02-01"}}}
        )
        self.assertEqual(len(loaded["1"]["names"]), anon_people.MAX_NAMES)


class _FakeBot:
    self_id = "10000"


class _FakePrivateEvent:
    def __init__(self, text: str, user_id: str = "71001"):
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


class HandlerWiringTest(unittest.TestCase):
    """P2-3：真跑 handle_message，验证档案注入的接线与“先渲染后更新”顺序。

    删掉 anonbot 里任一 ``user_profile=`` 传参、或把 render 挪到 update 之后，
    这组测试都会红。
    """

    USER_ID = "71001"
    TEST_UIDS = ("71001", "71002", "71003", "71004")

    def setUp(self):
        # 同一进程内重复跑测试也不能被上一次留下的档案影响（P3-F）。
        for uid in self.TEST_UIDS:
            anonbot.user_profiles.pop(uid, None)
            anonbot.histories.pop(("private:" + uid, uid), None)
            anonbot.scope_locks.pop("private:" + uid, None)

    def tearDown(self):
        self.setUp()

    def _run(self, text: str, plan=None, uid: str = None):
        captured: dict = {}
        uid = uid or self.USER_ID

        async def fake_understand(*args, **kwargs):
            captured["understand"] = kwargs
            return dict(plan) if plan else None

        async def fake_compose(*args, **kwargs):
            captured["compose"] = kwargs
            return "嗯，记得。"

        async def fake_ask(prompt, max_chars=35, timeout=None):
            captured["fallback_prompt"] = prompt
            return "嗯，记得，你上次提过。"

        async def fake_send(*args, **kwargs):
            captured["sent"] = True

        async def fake_materials(queries):
            return ""

        event = _FakePrivateEvent(text, uid)
        key = ("private:" + uid, uid)
        anonbot.histories.pop(key, None)
        anonbot.scope_locks.pop("private:" + uid, None)
        patches = [
            mock.patch.object(anonbot.agent_core, "understand", fake_understand),
            mock.patch.object(anonbot.agent_core, "compose_chat_reply", fake_compose),
            mock.patch.object(anonbot, "_ask_chat", fake_ask),
            mock.patch.object(anonbot, "_quick_search_materials", fake_materials),
            mock.patch.object(anonbot, "_send_reply", fake_send),
            mock.patch.object(anonbot, "_log_sent", lambda *a, **k: None),
            mock.patch.object(anonbot, "_save_user_profiles", lambda: None),
            mock.patch.object(anonbot.experience, "render_hint", lambda *a, **k: ""),
        ]
        for patch in patches:
            patch.start()
        try:
            asyncio.run(anonbot.handle_message(_FakeBot(), event))
        finally:
            for patch in reversed(patches):
                patch.stop()
            anonbot.histories.pop(key, None)
            anonbot.scope_locks.pop("private:" + uid, None)
        return captured

    def test_first_contact_has_no_profile_block(self):
        captured = self._run("我是小A", uid="71002")
        self.assertEqual(captured["understand"]["user_profile"], "")
        self.assertNotIn("【当前用户】", captured["fallback_prompt"])

    def test_profile_uses_previous_message_not_current(self):
        self._run("我是小A", uid="71003")
        second = self._run("你还记得我吗", uid="71003")
        block = second["understand"]["user_profile"]
        self.assertIn("【当前用户】", block)
        self.assertIn("QQ …1003", block)
        self.assertIn("我是小A", block)          # 上一句进入档案
        self.assertNotIn("你还记得我吗", block)   # 当前句还没写进档案
        # fallback 闲聊提示词也拿到同一块
        self.assertIn("我是小A", second["fallback_prompt"])

    def test_chat_plan_path_injects_profile(self):
        self._run("我是小A", uid="71004")
        self._run("你还记得我吗", uid="71004")
        plan = {
            "mode": "chat", "reply": "记得呀", "agent": "none",
            "need": "", "steps": [], "complex": False, "fresh": False, "act": "none",
        }
        third = self._run("在吗", plan=plan, uid="71004")
        block = third["compose"].get("user_profile", "")
        self.assertIn("【当前用户】", block)
        self.assertIn("你还记得我吗", block)   # 上次说 = 上一句，不是“在吗”
        self.assertNotIn("在吗", block)


class PersistenceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = Path(self.tmp) / "anon_memory.json"
        self._patches = [
            mock.patch.object(anonbot, "MEMORY_FILE", self.mem),
            mock.patch.object(anonbot, "group_summaries", {"100000003": "摘要"}),
            mock.patch.object(anonbot, "user_profiles", {}),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self._patches):
            patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_and_reload_both_keys(self):
        anonbot.user_profiles["200000011"] = {
            "names": {"群友A": {"first": "x", "last": "y", "count": 2}},
            "first": "x",
            "last": "y",
            "count": 3,
            "last_msg": "hi",
        }
        anonbot._save_memory()
        payload = json.loads(self.mem.read_text(encoding="utf-8"))
        self.assertEqual(payload["group_summaries"]["100000003"], "摘要")
        self.assertEqual(payload["users"]["200000011"]["count"], 3)
        loaded = anonbot._load_user_profiles()
        self.assertEqual(loaded["200000011"]["last_msg"], "hi")

    def test_throttle_allows_one_write_per_interval(self):
        writes = []
        with mock.patch.object(anonbot, "_save_memory", lambda: writes.append(1)), \
                mock.patch.object(anonbot, "_last_profile_save", 0.0), \
                mock.patch.object(anonbot.time, "monotonic", lambda: 100.0), \
                mock.patch("nonebot.get_driver", lambda: object()):
            anonbot._save_user_profiles()
            anonbot._save_user_profiles()
        self.assertEqual(len(writes), 1)

    def test_no_driver_skips_write(self):
        writes = []
        with mock.patch.object(anonbot, "_save_memory", lambda: writes.append(1)), \
                mock.patch.object(anonbot, "_last_profile_save", 0.0), \
                mock.patch.object(anonbot.time, "monotonic", lambda: 100.0), \
                mock.patch("nonebot.get_driver", side_effect=ValueError("not initialized")):
            anonbot._save_user_profiles()
        self.assertEqual(writes, [])


if __name__ == "__main__":
    unittest.main()
