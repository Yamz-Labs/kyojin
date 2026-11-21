"""A request that fails (400 / 503 / client gone / engine error) must leave the server answering: the next normal request gets its reply.
CPU test of tools/hy4/serve.py with a fake engine (no GPU, no model). Every call has a timeout, so a stuck server fails the test fast.
Run: python -m unittest tests/test_hy4_request_error.py   (needs aiohttp + jinja2 + tokenizers)"""
import asyncio
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "hy4"))
sys.path.insert(0, HERE)
import serve as S  # noqa: E402
from test_hy4_serve_http import make_state  # noqa: E402

try:
    from aiohttp.test_utils import TestClient, TestServer
except ImportError:      # pragma: no cover
    TestClient = None

TMO = 10
IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
TXT = {"type": "text", "text": "what is this"}
USER = {"role": "user", "content": "salut"}
NOTHINK = {"reasoning_effort": "no_think", "max_tokens": 50}


def bad_cases():
    """(name, path, body): every one must be refused with a 400"""
    chat = "/v1/chat/completions"
    cases = [
        ("image_user", chat, {"messages": [{"role": "user", "content": [TXT, IMG]}]}),
        ("image_only", chat, {"messages": [{"role": "user", "content": [IMG]}]}),
        ("image_tool_result", chat, {"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": [TXT, IMG]},
            {"role": "user", "content": "and now"}]}),
        ("image_assistant", chat, {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": [TXT, IMG]}, {"role": "user", "content": "b"}]}),
        ("image_system", chat, {"messages": [{"role": "system", "content": [IMG]}, USER]}),
        ("audio_part", chat, {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AA", "format": "wav"}}]}]}),
        ("unknown_part", chat, {"messages": [USER, {"role": "assistant", "content": "a"}, {"role": "user", "content": [TXT, {"type": "video", "url": "u"}]}]}),
        ("content_number", chat, {"messages": [{"role": "user", "content": 5}]}),
        ("empty_messages", chat, {"messages": []}),
        ("no_messages", chat, {}),
        ("bad_tools", chat, {"messages": [USER], "tools": [1]}),
        ("tools_not_list", chat, {"messages": [USER], "tools": "x"}),
        ("tool_choice_needs_tools", chat, {"messages": [USER], "tool_choice": "required"}),
        ("bad_effort", chat, {"messages": [USER], "reasoning_effort": "ultra"}),
        ("bad_max_tokens", chat, {"messages": [USER], "max_tokens": "many"}),
        ("bad_stop", chat, {"messages": [USER], "stop": 5}),
        ("json_format", chat, {"messages": [USER], "response_format": {"type": "json_object"}}),
        ("oversize_prompt", chat, {"messages": [{"role": "user", "content": "x " * 20000}], "reasoning_effort": "no_think"}),
        ("completion_empty", "/v1/completions", {"prompt": ""}),
    ]
    out = []
    for name, path, body in cases:
        out.append((name, path, body))
        out.append((name + "_stream", path, dict(body, stream=True)))
    return out


@unittest.skipIf(TestClient is None, "aiohttp not installed")
class RequestError(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.old_mode = os.environ.get("KYOJIN_IMAGE_PARTS")
        os.environ["KYOJIN_IMAGE_PARTS"] = "reject"       # the strict 400; the default (note) has its own tests below
        self.st = make_state("Bonjour le monde")
        self.st.lock = asyncio.Lock()
        self.client = TestClient(TestServer(S.build_app(self.st)))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        os.environ.pop("KYOJIN_IMAGE_PARTS", None)
        if self.old_mode is not None:
            os.environ["KYOJIN_IMAGE_PARTS"] = self.old_mode

    async def post(self, path, body=None, **kw):
        async def go():
            r = await self.client.post(path, json=body, **kw) if body is not None else await self.client.post(path, **kw)
            return r.status, await r.read()
        return await asyncio.wait_for(go(), TMO)

    async def normal(self, stream=False):
        """a normal chat request answers 200 with the reply, and the lock is free"""
        body = dict(messages=[USER], stream=stream, **NOTHINK)
        status, raw = await self.post("/v1/chat/completions", body)
        self.assertEqual(status, 200, raw[:300])
        if stream:
            text = "".join(c["choices"][0]["delta"].get("content", "") for c in (json.loads(l[6:]) for l in raw.decode().split("\n\n") if l.startswith("data: {")) if c["choices"])
            self.assertTrue(raw.rstrip().endswith(b"[DONE]"))
        else:
            text = json.loads(raw)["choices"][0]["message"]["content"]
        self.assertEqual(text, "Bonjour le monde")
        self.assertFalse(self.st.lock.locked())

    async def test_refused_request_then_normal(self):
        for name, path, body in bad_cases():
            with self.subTest(name):
                status, raw = await self.post(path, body)
                self.assertEqual(status, 400, raw[:300])
                self.assertIn(b"error", raw)
                self.assertFalse(self.st.lock.locked())
                await self.normal(stream=False)
                await self.normal(stream=True)
        self.assertEqual(self.st.engine.calls.__len__(), 2 * len(bad_cases()))      # a refused request never reached the engine

    async def test_media_parts_answer_by_default(self):
        """KYOJIN_IMAGE_PARTS unset = note: the image / audio request is served, the model reads the note"""
        del os.environ["KYOJIN_IMAGE_PARTS"]
        n = 0
        for name, path, body in bad_cases():
            if not name.startswith(("image_", "audio_")):
                continue
            status, raw = await self.post(path, dict(body, max_tokens=50, reasoning_effort="no_think"))
            self.assertEqual(status, 200, (name, raw[:300]))
            n += 1
        self.assertEqual(n, 12)
        self.assertIn("an image was attached", self.st.vocab.decode(self.st.engine.calls[0][0]))
        await self.normal()

    async def test_omp_shaped_request_then_normal(self):
        """an agent client (omp): system + user text, an assistant tool call, a tool result that carries an image_url part, tools with JSON-schema parameters
        (nested objects, enums, anyOf, arrays). Served streamed and not, then a normal request answers; under reject the same request is a 400 and the next one answers"""
        del os.environ["KYOJIN_IMAGE_PARTS"]
        tools = [
            {"type": "function", "function": {"name": "read_file", "description": "read a file", "parameters": {
                "type": "object", "properties": {"path": {"type": "string"}, "range": {"type": "object", "properties": {
                    "start": {"type": "integer"}, "end": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "screenshot", "description": "take a screenshot", "parameters": {
                "type": "object", "properties": {"target": {"type": "string", "enum": ["screen", "window", "region"]},
                                                 "regions": {"type": "array", "items": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}}},
                "additionalProperties": False}}},
        ]
        msgs = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": [{"type": "text", "text": "look at the screen"}]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "screenshot", "arguments": "{\"target\": \"screen\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "ok"}, IMG]},
            {"role": "user", "content": [IMG, {"type": "text", "text": "and now?"}]},
        ]
        for stream in (False, True):
            status, raw = await self.post("/v1/chat/completions", dict(messages=msgs, tools=tools, tool_choice="auto", stream=stream, **NOTHINK))
            self.assertEqual(status, 200, raw[:300])
            await self.normal(stream)
        prompt = self.st.vocab.decode(self.st.engine.calls[0][0])
        self.assertIn("an image was attached", prompt)
        self.assertIn("screenshot", prompt)
        os.environ["KYOJIN_IMAGE_PARTS"] = "reject"
        for stream in (False, True):
            status, raw = await self.post("/v1/chat/completions", dict(messages=msgs, tools=tools, stream=stream, **NOTHINK))
            self.assertEqual(status, 400, raw[:300])
            self.assertIn(b"message 3, part 1: type 'image_url' is not supported", raw)
            await self.normal(stream)

    async def test_accepted_parts_answer_and_refused_part_is_named(self):
        parts = [{"type": "text", "text": "salut", "cache_control": {"type": "ephemeral"}}, "!", {"type": "thinking", "thinking": "hm"}]
        history = [USER, {"role": "assistant", "content": [{"type": "refusal", "refusal": "no"}, {"type": "reasoning", "text": "r"}]}, {"role": "user", "content": parts}]
        for stream in (False, True):
            status, raw = await self.post("/v1/chat/completions", dict(messages=history, stream=stream, **NOTHINK))
            self.assertEqual(status, 200, raw[:300])
        status, raw = await self.post("/v1/chat/completions", dict(messages=history[:2] + [{"role": "user", "content": [TXT, {"type": "video"}]}]))
        self.assertEqual(status, 400)
        self.assertIn(b"message 2, part 1: type 'video' is not supported", raw)

    async def test_bad_json_then_normal(self):
        for stream in (False, True):
            status, raw = await self.post("/v1/chat/completions", data=b"{not json", headers={"Content-Type": "application/json"})
            self.assertEqual(status, 400, raw)
            await self.normal(stream)

    async def test_image_flood_then_normal(self):
        """many refused requests in a row, some at the same time as a good one"""
        bad = {"messages": [{"role": "user", "content": [TXT, IMG]}]}
        res = await asyncio.wait_for(asyncio.gather(*[self.post("/v1/chat/completions", dict(bad, stream=i % 2 == 0)) for i in range(20)]), TMO)
        self.assertTrue(all(s == 400 for s, _ in res))
        await self.normal()

    async def test_not_ready_then_normal(self):
        self.st.ready = False
        status, _ = await self.post("/v1/chat/completions", dict(messages=[USER], **NOTHINK))
        self.assertEqual(status, 503)
        self.st.ready = True
        await self.normal()

    async def test_engine_error_releases(self):
        """the engine raises: 500 (not streamed) or an error event (streamed), then a normal request answers"""
        good = self.st.engine.generate

        def boom(*a, **k):
            raise RuntimeError("engine exploded")
        self.st.engine.generate = boom
        status, raw = await self.post("/v1/chat/completions", dict(messages=[USER], **NOTHINK))
        self.assertEqual(status, 500, raw)
        status, raw = await self.post("/v1/chat/completions", dict(messages=[USER], stream=True, **NOTHINK))
        self.assertEqual(status, 200)
        self.assertIn(b"engine exploded", raw)
        self.assertIn(b"[DONE]", raw)
        self.st.engine.generate = good
        await self.normal()
        await self.normal(stream=True)

    async def test_output_flush_error_releases(self):
        """the output stream raises while the worker flushes (and the engine raised first): the handler still gets its end event, the lock comes back"""
        import serve_core as C
        flush = C.OutputStream.finish

        def bad_finish(self):
            raise ValueError("flush exploded")
        good = self.st.engine.generate

        def boom(*a, **k):
            raise RuntimeError("engine exploded")
        C.OutputStream.finish = bad_finish
        try:
            for engine in (boom, good):
                self.st.engine.generate = engine
                status, raw = await self.post("/v1/chat/completions", dict(messages=[USER], **NOTHINK))
                self.assertEqual(status, 500, raw)
                status, raw = await self.post("/v1/chat/completions", dict(messages=[USER], stream=True, **NOTHINK))
                self.assertIn(b"exploded", raw)
                self.assertIn(b"[DONE]", raw)
                self.assertFalse(self.st.lock.locked())
        finally:
            C.OutputStream.finish = flush
            self.st.engine.generate = good
        await self.normal()

    async def test_parser_bug_is_a_500_and_the_server_lives(self):
        """an unexpected exception in a request parser (not a RequestError) is one 500 JSON error, then a normal request answers"""
        import serve_core as C
        parse = C.parse_chat

        def bug(*a, **k):
            raise KeyError("oops")
        C.parse_chat = bug
        try:
            for path in ("/v1/chat/completions", "/apply-template", "/tokenize"):
                status, raw = await self.post(path, dict(messages=[USER], **NOTHINK))
                self.assertEqual(status, 500, raw)
                self.assertIn(b"server_error", raw)
        finally:
            C.parse_chat = parse
        await self.normal()

    async def test_client_disconnect_releases(self):
        """the client drops a streamed request mid-generation: the lock comes back and the next request answers"""
        slow = self.st.engine.generate
        gate = asyncio.Event()

        def slow_gen(ids, max_new, on_tokens, speculative=True, cancel=lambda: False):
            import time
            on_tokens(self.st.engine.ids[:1])
            t0 = time.time()
            while not cancel() and time.time() - t0 < 5:
                time.sleep(0.02)
            return slow(ids, 1, on_tokens, speculative, cancel)
        self.st.engine.generate = slow_gen
        resp = await self.client.post("/v1/chat/completions", json=dict(messages=[USER], stream=True, **NOTHINK))
        await asyncio.wait_for(resp.content.readany(), TMO)
        resp.close()
        del gate
        for _ in range(100):
            if not self.st.lock.locked():
                break
            await asyncio.sleep(0.1)
        self.assertFalse(self.st.lock.locked())
        self.st.engine.generate = slow
        await self.normal()


if __name__ == "__main__":
    unittest.main()
