"""Content parts of a chat message: which types are read, which are refused, and how the refusal names the part. CPU only, no model.
Run: python -m unittest tests/test_hy4_content_parts.py"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "hy4"))
import serve_core as C  # noqa: E402


def parse(*msgs):
    return C.parse_chat({"messages": list(msgs)})


def user_text(parts):
    return parse({"role": "user", "content": parts})["messages"][-1]["content"]


class AcceptedParts(unittest.TestCase):
    def test_text_names(self):
        for t in ("text", "input_text", "output_text"):
            self.assertEqual(user_text([{"type": t, "text": "hi"}]), "hi", t)

    def test_refusal_is_text(self):
        r = parse({"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "refusal", "refusal": "I cannot"}]}, {"role": "user", "content": "ok"})
        self.assertEqual(r["messages"][1]["content"], "I cannot")

    def test_extra_keys_ignored(self):
        self.assertEqual(user_text([{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}, "annotations": [], "x": 1}]), "a")

    def test_bare_string_and_null_parts(self):
        self.assertEqual(user_text(["a", None, {"type": "text", "text": "b"}, "c"]), "abc")

    def test_empty_and_null_text(self):
        self.assertEqual(user_text([{"type": "text", "text": ""}, {"type": "text", "text": None}, {"type": "text"}, {"text": "z"}]), "z")

    def test_order_of_surviving_parts(self):
        parts = [{"type": "text", "text": "1"}, {"type": "thinking", "thinking": "skip"}, "2", {"type": "redacted_thinking", "data": "x"},
                 {"type": "output_text", "text": "3"}, {"type": "refusal", "refusal": "4"}, {"type": "input_text", "text": "5"}]
        self.assertEqual(user_text(parts), "12345")

    def test_reasoning_parts_become_reasoning_content_of_an_assistant(self):
        for kind, key in (("reasoning", "text"), ("thinking", "thinking"), ("reasoning_content", "text")):
            r = parse({"role": "user", "content": "q"},
                      {"role": "assistant", "content": [{"type": kind, key: "because"}, {"type": "text", "text": "answer"}]},
                      {"role": "user", "content": "next"})
            m = r["messages"][1]
            self.assertEqual((m["content"], m.get("reasoning_content")), ("answer", "because"), kind)

    def test_explicit_reasoning_field_wins(self):
        r = parse({"role": "user", "content": "q"},
                  {"role": "assistant", "content": [{"type": "thinking", "thinking": "part"}, {"type": "text", "text": "a"}], "reasoning_content": "field"},
                  {"role": "user", "content": "n"})
        self.assertEqual(r["messages"][1]["reasoning_content"], "field")

    def test_reasoning_parts_dropped_in_other_roles(self):
        r = parse({"role": "user", "content": [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "q"}]})
        self.assertEqual(r["messages"][0]["content"], "q")
        self.assertNotIn("reasoning_content", r["messages"][0])

    def test_tool_result_with_parts(self):
        r = parse({"role": "user", "content": "go"},
                  {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                  {"role": "tool", "tool_call_id": "c", "content": [{"type": "text", "text": "out", "cache_control": {}}, "put"]},
                  {"role": "user", "content": "k"})
        self.assertEqual(r["messages"][2]["content"], "output")

    def test_table_is_the_only_gate(self):
        """one line in PART_ACTIONS adds a type"""
        C.PART_ACTIONS["my_text"] = "text"
        try:
            self.assertEqual(user_text([{"type": "my_text", "text": "ok"}]), "ok")
        finally:
            del C.PART_ACTIONS["my_text"]


class MediaParts(unittest.TestCase):
    def setUp(self):
        self.old = os.environ.pop("KYOJIN_IMAGE_PARTS", None)

    def tearDown(self):
        os.environ.pop("KYOJIN_IMAGE_PARTS", None)
        if self.old is not None:
            os.environ["KYOJIN_IMAGE_PARTS"] = self.old

    def test_default_is_note_in_every_role_and_order(self):
        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
        want = {"image_url": "image", "input_image": "image", "image": "image", "input_audio": "audio", "audio": "audio", "file": "file", "input_file": "file"}
        for kind, what in want.items():
            self.assertEqual(user_text(["a", {"type": kind}, {"type": "text", "text": "b"}]), "a" + C.MEDIA_NOTES[what] + "b", kind)
        r = parse({"role": "system", "content": [img]}, {"role": "user", "content": "q"},
                  {"role": "assistant", "content": [{"type": "text", "text": "x"}, img]}, {"role": "user", "content": [img, "k"]})
        self.assertEqual([m["content"] for m in r["messages"]], [C.MEDIA_NOTES["image"], "q", "x" + C.MEDIA_NOTES["image"], C.MEDIA_NOTES["image"] + "k"])

    def test_file_note_text(self):
        self.assertEqual(user_text([{"type": "file", "file": {"file_id": "f"}}]), "[a file was attached; this model cannot read files]")

    def test_reject_is_the_strict_400(self):
        os.environ["KYOJIN_IMAGE_PARTS"] = "reject"
        for kind in C.MEDIA_PARTS:
            with self.assertRaises(C.RequestError) as cm:
                parse({"role": "user", "content": [{"type": "text", "text": "a"}, {"type": kind}]})
            self.assertIn(f"message 0, part 1: type '{kind}' is not supported", str(cm.exception))

    def test_typo_falls_back_to_note_with_one_warning(self):
        import contextlib
        import io
        os.environ["KYOJIN_IMAGE_PARTS"] = "rejct"
        C._WARNED_IMAGE_PARTS.clear()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(user_text([{"type": "image_url"}]), C.MEDIA_NOTES["image"])
            self.assertEqual(user_text([{"type": "image_url"}]), C.MEDIA_NOTES["image"])
        self.assertEqual(buf.getvalue().count("KYOJIN_IMAGE_PARTS"), 1)

    def test_unknown_types_still_400_in_note_mode(self):
        with self.assertRaises(C.RequestError):
            parse({"role": "user", "content": [{"type": "video_url"}]})


class RefusedParts(unittest.TestCase):
    def setUp(self):
        self.old = os.environ.get("KYOJIN_IMAGE_PARTS")
        os.environ["KYOJIN_IMAGE_PARTS"] = "reject"

    def tearDown(self):
        os.environ.pop("KYOJIN_IMAGE_PARTS", None)
        if self.old is not None:
            os.environ["KYOJIN_IMAGE_PARTS"] = self.old

    def refused(self, *msgs):
        with self.assertRaises(C.RequestError) as cm:
            parse(*msgs)
        self.assertEqual(cm.exception.status, 400)
        return str(cm.exception)

    def test_error_names_message_and_part(self):
        msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}] * 3 + [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "text", "text": "x"}, "y", {"type": "video", "url": "u"}]},
            {"role": "user", "content": "z"}]
        self.assertEqual(self.refused(*msgs), "message 7, part 2: type 'video' is not supported")

    def test_unknown_type(self):
        self.assertEqual(self.refused({"role": "user", "content": [{"type": "citation", "text": "x"}]}), "message 0, part 0: type 'citation' is not supported")

    def test_media_parts_are_refused_under_reject(self):
        for kind, what in (("image_url", "image"), ("input_image", "image"), ("image", "image"), ("input_audio", "audio"), ("audio", "audio"), ("file", "file"),
                           ("input_file", "file")):
            m = self.refused({"role": "user", "content": [{"type": "text", "text": "a"}, {"type": kind}]})
            self.assertIn(f"message 0, part 1: type '{kind}' is not supported", m)
            self.assertIn(what, m)

    def test_media_in_every_role_under_reject(self):
        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
        self.assertIn("message 0, part 0", self.refused({"role": "system", "content": [img]}))
        self.assertIn("message 2, part 1", self.refused(
            {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}, {"role": "tool", "content": ["x", img]}, {"role": "user", "content": "k"}))

    def test_no_type_and_no_text(self):
        self.assertIn("type None is not supported", self.refused({"role": "user", "content": [{"foo": 1}]}))

    def test_non_string_text(self):
        self.assertIn("message 0, part 0", self.refused({"role": "user", "content": [{"type": "text", "text": 5}]}))

    def test_bad_part_and_bad_content(self):
        self.assertIn("message 0, part 1: a part must be", self.refused({"role": "user", "content": ["a", 3]}))
        self.assertIn("message 0: content must be", self.refused({"role": "user", "content": 5}))


if __name__ == "__main__":
    unittest.main()
