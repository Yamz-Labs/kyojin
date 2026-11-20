"""CPU tests of the OpenAI-compatibility of the Hy4 chat request parser (tools/hy4/serve_core.py parse_chat) and of the server routes that use it:
max_completion_tokens, stream_options, n, response_format, null / empty content parts, system and developer messages. No GPU, no model.
Run: python -m pytest tests/test_hy4_compat_cpu.py   (needs aiohttp + jinja2 + tokenizers)"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "hy4"))
sys.path.insert(0, HERE)
import serve as S  # noqa: E402
import serve_core as C  # noqa: E402
from test_hy4_tools_cpu import ProdServer, make_state, tool_meta  # noqa: E402
from aiohttp.test_utils import TestClient  # noqa: E402

USER = [{"role": "user", "content": "hi"}]


def parse(**body):
    return C.parse_chat({"messages": USER, **body}, 99)


class Parser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = C.Vocab(tool_meta())

    def render(self, r):
        return self.v.template.render(r["messages"], r["tools"], True, reasoning_effort=r["effort"])

    # -- max_completion_tokens
    def test_max_completion_tokens_is_the_alias_of_max_tokens(self):
        self.assertEqual(parse(max_completion_tokens=7)["max_tokens"], 7)
        self.assertEqual(parse(max_tokens=8)["max_tokens"], 8)
        self.assertEqual(parse(max_completion_tokens=7, max_tokens=8)["max_tokens"], 7)       # the newer name wins
        self.assertEqual(parse(max_completion_tokens=None, max_tokens=8)["max_tokens"], 8)    # an explicit null does not hide max_tokens
        self.assertEqual(parse(max_completion_tokens=None, max_tokens=None)["max_tokens"], 99)
        for bad in (0, -1, "7", 1.5, True):
            with self.assertRaises(C.RequestError, msg=str(bad)):
                parse(max_completion_tokens=bad)
        self.assertEqual(C.parse_completion({"prompt": "x", "max_completion_tokens": 5}, None)["max_tokens"], 5)

    # -- stream_options
    def test_stream_options(self):
        self.assertTrue(parse(stream=True, stream_options={"include_usage": True})["include_usage"])
        self.assertFalse(parse(stream=True, stream_options={"include_usage": False})["include_usage"])
        self.assertFalse(parse(stream=True, stream_options=None)["include_usage"])
        self.assertFalse(parse(stream=True, stream_options={})["include_usage"])
        for bad in ("yes", ["include_usage"], 1):
            with self.assertRaises(C.RequestError, msg=str(bad)):       # was an AttributeError, so a 500
                parse(stream=True, stream_options=bad)

    # -- n
    def test_n_above_one_is_a_clean_400(self):
        self.assertEqual(parse(n=1)["max_tokens"], 99)
        self.assertEqual(parse(n=None)["max_tokens"], 99)
        for bad in (2, 0, -1, "2", 1.5):
            with self.assertRaises(C.RequestError, msg=str(bad)) as cm:
                parse(n=bad)
            self.assertEqual((cm.exception.status, cm.exception.code), (400, "invalid_request_error"))
        with self.assertRaises(C.RequestError) as cm:
            parse(n=3)
        self.assertIn("one choice", str(cm.exception))

    # -- response_format
    def test_response_format(self):
        for rf in (None, {"type": "text"}):
            self.assertEqual(parse(response_format=rf)["messages"], USER)
        r = parse(response_format={"type": "json_object"})
        self.assertEqual(r["messages"][0], {"role": "system", "content": C.JSON_HINT})
        self.assertEqual(r["messages"][1:], USER)
        self.assertIn(C.JSON_HINT, self.render(r))
        schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        r = C.parse_chat({"messages": [{"role": "system", "content": "Be brief."}, *USER],
                          "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}})
        self.assertTrue(r["messages"][0]["content"].startswith("Be brief.\n\n" + C.JSON_SCHEMA_HINT))
        self.assertIn(json.dumps(schema), r["messages"][0]["content"])
        for bad in ("json", {"type": "xml"}, {}, {"type": "json_schema"}, {"type": "json_schema", "json_schema": {"name": "x"}}):
            with self.assertRaises(C.RequestError, msg=str(bad)):
                parse(response_format=bad)

    def test_response_format_and_tool_choice_share_one_instruction_block(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
        r = parse(tools=tools, tool_choice="required", response_format={"type": "json_object"})
        self.assertEqual(len(r["messages"]), 2)
        self.assertEqual(r["messages"][0]["content"], C.TOOL_HINTS["required"] + "\n\n" + C.JSON_HINT)

    # -- content parts
    def test_null_and_empty_content_parts(self):
        def text(content):
            return C.parse_chat({"messages": [{"role": "user", "content": content}]})["messages"][0]["content"]
        self.assertEqual(text(None), "")
        self.assertEqual(text(""), "")
        self.assertEqual(text([]), "")
        self.assertEqual(text([None, {"type": "text", "text": ""}, {"type": "text", "text": None}, {"type": "text"}]), "")
        self.assertEqual(text([{"type": "text", "text": "a"}, None, {"type": "text", "text": "b"}, "c"]), "abc")
        self.assertEqual(text([{"type": "input_text", "text": "a"}, {"type": "output_text", "text": "b"}, {"text": "c"}]), "abc")
        for bad in ([{"type": "image_url", "image_url": {"url": "x"}}], [{"type": "audio"}], [{"type": "text", "text": 5}], [{}], 5, {"type": "text", "text": "x"}):
            with self.assertRaises(C.RequestError, msg=str(bad)):
                text(bad)

    def test_empty_messages_still_render(self):
        r = C.parse_chat({"messages": [{"role": "system", "content": None}, {"role": "user", "content": [None]}, {"role": "assistant", "content": ""},
                                       {"role": "user", "content": ""}]})
        self.assertTrue(self.render(r).endswith("<think:6124c78e>"))

    # -- system and developer
    def test_system_and_developer_roles(self):
        r = C.parse_chat({"messages": [{"role": "developer", "content": "Be brief."}, *USER]})
        self.assertEqual(r["messages"][0], {"role": "system", "content": "Be brief."})
        r = C.parse_chat({"messages": [{"role": "system", "content": "Rule one."}, {"role": "developer", "content": "Rule two."}, *USER]})
        self.assertEqual([m["role"] for m in r["messages"]], ["system", "system", "user"])
        text = self.render(r)
        self.assertLess(text.index("Rule one."), text.index("Rule two."))
        self.assertLess(text.index("Rule two."), text.index("｜>hi"))
        r = C.parse_chat({"messages": [*USER, {"role": "assistant", "content": "hello"}, {"role": "developer", "content": "Switch to French."}, {"role": "user", "content": "again"}]})
        self.assertEqual([m["role"] for m in r["messages"]], ["user", "assistant", "system", "user"])        # a system message in the middle is rendered where it stands
        text = self.render(r)
        self.assertLess(text.index("hello"), text.index("Switch to French."))
        self.assertLess(text.index("Switch to French."), text.index("again"))

    def test_unknown_role_is_a_400(self):
        with self.assertRaises(C.RequestError):
            C.parse_chat({"messages": [{"role": "robot", "content": "x"}, *USER]})


class Http(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.st = make_state("Bonjour le monde")
        self.client = TestClient(ProdServer(S.build_app(self.st)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def chat(self, **body):
        return await self.client.post("/v1/chat/completions", json={"messages": USER, "reasoning_effort": "no_think", **body})

    async def test_max_completion_tokens_reaches_the_engine(self):
        r = await self.chat(max_completion_tokens=4)
        j = await r.json()
        self.assertEqual((r.status, j["choices"][0]["message"]["content"], j["usage"]["completion_tokens"]), (200, "Bonj", 4))
        self.assertEqual(self.st.engine.calls[-1][1], 4)

    async def test_n_and_stream_options_errors_are_400_bodies(self):
        for body in ({"n": 2}, {"stream": True, "stream_options": "yes"}, {"response_format": {"type": "xml"}}):
            r = await self.chat(**body)
            j = await r.json()
            self.assertEqual(r.status, 400, body)
            self.assertEqual(j["error"]["type"], "invalid_request_error")

    async def test_stream_usage_chunk(self):
        r = await self.chat(stream=True, stream_options={"include_usage": True})
        events = [e[6:] for e in (await r.read()).decode().split("\n\n") if e.startswith("data: ")]
        last = json.loads(events[-2])
        self.assertEqual((last["choices"], last["usage"]["completion_tokens"]), ([], len("Bonjour le monde")))
        r = await self.chat(stream=True)
        events = [e[6:] for e in (await r.read()).decode().split("\n\n") if e.startswith("data: ")]
        self.assertTrue(all("usage" not in json.loads(e) for e in events[:-1]))

    async def test_json_object_adds_the_instruction_to_the_prompt(self):
        r = await self.chat(response_format={"type": "json_object"}, messages=[{"role": "developer", "content": "Be brief."}, *USER])
        self.assertEqual(r.status, 200)
        prompt = self.st.vocab.decode(self.st.engine.calls[-1][0])
        self.assertIn("Be brief.\n\n" + C.JSON_HINT, prompt)

    async def test_developer_and_null_parts_are_served(self):
        r = await self.chat(messages=[{"role": "developer", "content": [{"type": "text", "text": "Be brief."}, None]}, {"role": "user", "content": None}])
        self.assertEqual(r.status, 200, await r.text())


if __name__ == "__main__":
    unittest.main()
