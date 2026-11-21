"""CPU tests of KYOJIN_IMAGE_PARTS (tools/hy4/serve_core.py _content_text / parse_chat): `reject` (default) answers 400 for image and audio content
parts, `note` replaces each by a one-line text note. No GPU, no model, no server.
Run: python -m pytest tests/test_hy4_image_parts.py   (needs jinja2)"""
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "hy4"))
import serve_core as C  # noqa: E402

IMG = "[an image was attached; this model cannot see images]"
AUD = "[audio was attached; this model cannot hear audio]"
IMAGE_PARTS = ({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "input_image", "image_url": "x"}, {"type": "image", "source": {}})
AUDIO_PARTS = ({"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}, {"type": "audio", "audio": {}})


def env(mode):
    e = {k: v for k, v in os.environ.items() if k != "KYOJIN_IMAGE_PARTS"}
    if mode is not None:
        e["KYOJIN_IMAGE_PARTS"] = mode
    return mock.patch.dict(os.environ, e, clear=True)


def chat(*msgs):
    return C.parse_chat({"messages": list(msgs)})["messages"]


class ImageParts(unittest.TestCase):
    def test_default_and_reject_are_a_400(self):
        for mode in (None, "", "reject", "REJECT", "typo"):
            with env(mode):
                for part in IMAGE_PARTS + AUDIO_PARTS:
                    with self.assertRaises(C.RequestError, msg=f"{mode} {part}") as cm:
                        C._content_text([{"type": "text", "text": "a"}, part])
                    self.assertEqual(cm.exception.status, 400)
                    self.assertIn("only text content parts", str(cm.exception))

    def test_note_replaces_each_part_in_order(self):
        with env("note"):
            for part in IMAGE_PARTS:
                self.assertEqual(C._content_text([part]), IMG)
            for part in AUDIO_PARTS:
                self.assertEqual(C._content_text([part]), AUD)
            mixed = [{"type": "text", "text": "look: "}, IMAGE_PARTS[0], {"type": "text", "text": " and "}, AUDIO_PARTS[0], IMAGE_PARTS[1], None, "end"]
            self.assertEqual(C._content_text(mixed), f"look: {IMG} and {AUD}{IMG}end")
            self.assertEqual(C._content_text("plain"), "plain")
            self.assertEqual(C._content_text(None), "")

    def test_note_mode_is_read_per_call(self):
        part = [IMAGE_PARTS[0]]
        with env("note"):
            self.assertEqual(C._content_text(part), IMG)
        with env("reject"), self.assertRaises(C.RequestError):
            C._content_text(part)

    def test_unknown_parts_still_rejected_in_note_mode(self):
        with env("note"):
            for bad in ([{"type": "video_url", "video_url": {}}], [{"type": "file", "file": {}}], [{"type": "text", "text": 5}], [{}], [{"type": "image_url"}, {"type": "refusal"}], 5):
                with self.assertRaises(C.RequestError, msg=str(bad)):
                    C._content_text(bad)

    def test_every_role_in_a_chat(self):
        tool_call = {"id": "c1", "type": "function", "function": {"name": "shot", "arguments": "{}"}}
        msgs = [{"role": "user", "content": [{"type": "text", "text": "what is this?"}, IMAGE_PARTS[0]]},
                {"role": "assistant", "content": [{"type": "text", "text": "let me look"}, IMAGE_PARTS[1]], "tool_calls": [tool_call]},
                {"role": "tool", "tool_call_id": "c1", "content": [IMAGE_PARTS[0], {"type": "text", "text": "screenshot"}, AUDIO_PARTS[0]]},
                {"role": "user", "content": [{"type": "text", "text": "and now?"}]}]
        with env("note"):
            out = chat(*msgs)
        self.assertEqual(out[0]["content"], f"what is this?{IMG}")
        self.assertEqual(out[1]["content"], f"let me look{IMG}")
        self.assertEqual(out[1]["tool_calls"][0]["function"]["name"], "shot")
        self.assertEqual(out[2], {"role": "tool", "content": f"{IMG}screenshot{AUD}", "tool_call_id": "c1"})
        self.assertEqual(out[3]["content"], "and now?")
        with env(None):                        # default: the same chat is a 400, whichever role carries the part
            for m in msgs[:3]:
                with self.assertRaises(C.RequestError, msg=m["role"]):
                    chat({"role": "user", "content": "q"}, m, {"role": "user", "content": "x"})

    def test_system_message_with_image_part(self):
        with env("note"):
            out = chat({"role": "system", "content": [{"type": "text", "text": "rules "}, IMAGE_PARTS[0]]}, {"role": "user", "content": "hi"})
        self.assertEqual(out[0], {"role": "system", "content": f"rules {IMG}"})


if __name__ == "__main__":
    unittest.main()
