# -*- coding: utf-8 -*-
"""WP1（2026-10-26 10:00）：视觉模块 + anonbot 图片接线，全部离线。

不联网：urlopen 一律 mock；不写 anon_memory.json / anon_state.json。
"""
import asyncio
import io
import sys
import time
import unittest
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plugins import anon_vision, anonbot


def _png() -> bytes:
    """最小 PNG 头 + 少量字节：足够走 mime 识别，不需要真解码。"""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class _FakeEvent:
    def __init__(self, message, reply=None):
        self._message = message
        self.reply = reply

    def get_message(self):
        return self._message


def _image(url):
    return SimpleNamespace(type="image", data={"url": url})


def _text(value):
    return SimpleNamespace(type="text", data={"text": value})


def _run(coro):
    return asyncio.run(coro)


class DescribeImageGuardTest(unittest.TestCase):
    def setUp(self):
        anon_vision._cache_clear()
        anon_vision._warned["disabled"] = False
        anon_vision._warned["no_key"] = False
        key_patch = mock.patch.object(anon_vision, "VISION_API_KEY", "test-key")
        key_patch.start()
        self.addCleanup(key_patch.stop)

    def test_empty_url_returns_empty(self):
        with mock.patch.object(anon_vision, "_download") as download:
            self.assertEqual(anon_vision.describe_image(""), "")
        download.assert_not_called()

    def test_file_scheme_rejected_without_download(self):
        with mock.patch.object(anon_vision, "_download") as download:
            self.assertEqual(anon_vision.describe_image("file:///x.png"), "")
        download.assert_not_called()

    def test_data_url_rejected(self):
        with mock.patch.object(anon_vision, "_download") as download:
            self.assertEqual(
                anon_vision.describe_image("data:image/png;base64,AAAA"), ""
            )
        download.assert_not_called()

    def test_disabled_returns_empty_without_network(self):
        with mock.patch.object(anon_vision, "VISION_ENABLED", False), \
                mock.patch.object(anon_vision, "_download") as download:
            self.assertEqual(anon_vision.describe_image("https://x/a.png"), "")
        download.assert_not_called()

    def test_missing_key_returns_empty_without_network(self):
        with mock.patch.object(anon_vision, "VISION_API_KEY", ""), \
                mock.patch.object(anon_vision, "_download") as download:
            self.assertEqual(anon_vision.describe_image("https://x/a.png"), "")
        download.assert_not_called()

    def test_download_error_returns_empty_and_logs_fail(self):
        buffer = io.StringIO()
        with mock.patch.object(
            anon_vision, "_download", side_effect=urllib.error.URLError("boom")
        ), redirect_stdout(buffer):
            self.assertEqual(anon_vision.describe_image("https://x/a.png"), "")
        output = buffer.getvalue()
        self.assertIn("[ANON VISION] fail sha=", output)
        self.assertIn("URLError", output)

    def test_oversized_download_returns_empty(self):
        oversized = b"x" * (anon_vision.MAX_DOWNLOAD_BYTES + 1)
        with mock.patch.object(anon_vision, "_download", return_value=oversized):
            self.assertEqual(anon_vision.describe_image("https://x/a.png"), "")

    def test_placeholder_image_returns_empty(self):
        with mock.patch.object(anon_vision, "_download", return_value=b"q" * 100), \
                mock.patch.object(
                    anon_vision, "_is_placeholder_raw", return_value=True
                ):
            self.assertEqual(anon_vision.describe_image("https://x/a.png"), "")

    def test_empty_vision_content_returns_empty(self):
        with mock.patch.object(anon_vision, "_download", return_value=_png()), \
                mock.patch.object(anon_vision, "_vision_call", return_value=""):
            self.assertEqual(anon_vision.describe_image("https://x/a.png"), "")

    def test_vision_exception_returns_empty(self):
        with mock.patch.object(anon_vision, "_download", return_value=_png()), \
                mock.patch.object(
                    anon_vision, "_vision_call", side_effect=RuntimeError("down")
                ):
            self.assertEqual(anon_vision.describe_image("https://x/a.png"), "")

    def test_success_returns_description(self):
        with mock.patch.object(anon_vision, "_download", return_value=_png()), \
                mock.patch.object(
                    anon_vision, "_vision_call", return_value="红色短袖，胸前印着ANON"
                ) as call:
            result = anon_vision.describe_image("https://x/a.png")
        self.assertEqual(result, "红色短袖，胸前印着ANON")
        self.assertEqual(call.call_count, 1)

    def test_cache_hit_skips_second_vision_call(self):
        with mock.patch.object(anon_vision, "_download", return_value=_png()), \
                mock.patch.object(
                    anon_vision, "_vision_call", return_value="红色短袖"
                ) as call:
            self.assertEqual(anon_vision.describe_image("https://x/a.png"), "红色短袖")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                second = anon_vision.describe_image("https://x/a.png")
        self.assertEqual(second, "红色短袖")
        self.assertEqual(call.call_count, 1)
        self.assertIn("cache hit", buffer.getvalue())

    def test_negative_cache_skips_second_download(self):
        with mock.patch.object(
            anon_vision, "_download", return_value=_png()
        ) as download, mock.patch.object(
            anon_vision, "_vision_call", side_effect=RuntimeError("down")
        ):
            self.assertEqual(anon_vision.describe_image("https://x/a.png"), "")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                second = anon_vision.describe_image("https://x/a.png")
        self.assertEqual(second, "")
        self.assertEqual(download.call_count, 1)
        self.assertIn("cache negative", buffer.getvalue())

    def test_cache_is_capped_at_64_entries(self):
        for index in range(70):
            anon_vision._cache_put("key-%d" % index, "v", anon_vision.CACHE_TTL_SECONDS)
        self.assertLessEqual(len(anon_vision._cache), anon_vision.CACHE_MAX_ENTRIES)
        self.assertEqual(len(anon_vision._cache), 64)

    def test_logs_never_contain_full_url(self):
        buffer = io.StringIO()
        with mock.patch.object(anon_vision, "_download", return_value=_png()), \
                mock.patch.object(
                    anon_vision, "_vision_call", return_value="红色短袖"
                ), redirect_stdout(buffer):
            anon_vision.describe_image("https://gchat.qpic.cn/secret/path.png?token=x")
        output = buffer.getvalue()
        self.assertIn("[ANON VISION] ok sha=", output)
        self.assertNotIn("http", output.lower())
        self.assertNotIn("gchat", output.lower())

    def test_clean_description_takes_first_sentence(self):
        self.assertEqual(
            anon_vision._clean_description("红色短袖。还有别的。"), "红色短袖"
        )

    def test_clean_description_truncates_to_limit(self):
        result = anon_vision._clean_description("啊" * 300)
        self.assertEqual(len(result), anon_vision.MAX_DESC_CHARS)

    def test_mime_detection(self):
        self.assertEqual(anon_vision._mime_of(b"\xff\xd8\xffxx"), "image/jpeg")
        self.assertEqual(anon_vision._mime_of(_png()), "image/png")
        self.assertEqual(anon_vision._mime_of(b"GIF89a...."), "image/gif")
        self.assertEqual(anon_vision._mime_of(b"RIFF1234WEBP"), "image/webp")

    def test_prepare_image_passes_small_png_through(self):
        mime, data, reason = anon_vision._prepare_image(_png())
        self.assertEqual(mime, "image/png")
        self.assertEqual(data, _png())
        self.assertEqual(reason, "")

    def test_placeholder_check_false_for_normal_bytes(self):
        self.assertFalse(anon_vision._is_placeholder_raw(_png()))


class ImageUrlExtractionTest(unittest.TestCase):
    def test_direct_image(self):
        event = _FakeEvent([_text("这衣服怎么样"), _image("https://x/a.png")])
        self.assertEqual(anonbot._image_urls(event), ["https://x/a.png"])

    def test_quoted_image_when_current_message_has_none(self):
        reply = SimpleNamespace(message=[_text("看图"), _image("https://x/q.png")])
        event = _FakeEvent([_text("这衣服怎么样")], reply=reply)
        self.assertEqual(anonbot._image_urls(event), ["https://x/q.png"])

    def test_direct_then_quote_dedupe_and_limit(self):
        reply = SimpleNamespace(message=[_image("https://x/c.png")])
        event = _FakeEvent(
            [
                _image("https://x/a.png"),
                _image("https://x/a.png"),
                _image("https://x/b.png"),
                _image("https://x/c.png"),
            ],
            reply=reply,
        )
        self.assertEqual(
            anonbot._image_urls(event), ["https://x/a.png", "https://x/b.png"]
        )

    def test_non_http_schemes_dropped(self):
        event = _FakeEvent(
            [
                _image("data:image/png;base64,AAAA"),
                _image("file:///tmp/a.png"),
                _image("base64://AAAA"),
                _image("https://x/ok.png"),
            ]
        )
        self.assertEqual(anonbot._image_urls(event), ["https://x/ok.png"])

    def test_bad_message_object_tolerated(self):
        class BadEvent:
            reply = None

            def get_message(self):
                raise TypeError("nope")

        self.assertEqual(anonbot._image_urls(BadEvent()), [])


class AugmentTextTest(unittest.TestCase):
    def test_no_urls_returns_text_without_log(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = _run(anonbot._augment_text_with_images("你好", []))
        self.assertEqual(result, "你好")
        self.assertNotIn("[ANON VISION]", buffer.getvalue())

    def test_all_success_injects_both_descriptions(self):
        def fake(url):
            return "红色短袖" if "/1.png" in url else "黑色长裤"

        buffer = io.StringIO()
        with mock.patch.object(anon_vision, "describe_image", side_effect=fake), \
                redirect_stdout(buffer):
            result = _run(
                anonbot._augment_text_with_images(
                    "这衣服怎么样", ["https://x/1.png", "https://x/2.png"]
                )
            )
        self.assertEqual(
            result, "这衣服怎么样 [图片内容：红色短袖；黑色长裤]"
        )
        self.assertIn(
            "[ANON VISION] images=2 described=2 chars=8", buffer.getvalue()
        )

    def test_partial_failure_mentions_failed_count(self):
        def fake(url):
            return "红色短袖" if "/1.png" in url else ""

        with mock.patch.object(anon_vision, "describe_image", side_effect=fake):
            result = _run(
                anonbot._augment_text_with_images(
                    "这衣服怎么样", ["https://x/1.png", "https://x/2.png"]
                )
            )
        self.assertIn("红色短袖", result)
        self.assertIn("另有1张读取失败", result)

    def test_all_failure_injects_exact_sentence(self):
        with mock.patch.object(anon_vision, "describe_image", return_value=""):
            result = _run(
                anonbot._augment_text_with_images("这衣服怎么样", ["https://x/a.png"])
            )
        self.assertTrue(
            result.endswith(
                " [图片内容：读取失败，你看不到这张图，不要假装看见]"
            )
        )

    def test_timeout_injects_failure_sentence(self):
        def slow(_url):
            time.sleep(0.2)
            return "红色短袖"

        buffer = io.StringIO()
        with mock.patch.object(anonbot, "VISION_BUDGET_SECONDS", 0.05), \
                mock.patch.object(anon_vision, "describe_image", side_effect=slow), \
                redirect_stdout(buffer):
            result = _run(
                anonbot._augment_text_with_images("这衣服怎么样", ["https://x/a.png"])
            )
        output = buffer.getvalue()
        self.assertIn("读取失败", result)
        self.assertIn("[ANON VISION] timeout", output)
        self.assertIn("images=1 described=0 chars=0", output)


if __name__ == "__main__":
    unittest.main()
