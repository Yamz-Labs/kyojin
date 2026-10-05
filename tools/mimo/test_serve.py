#!/usr/bin/env python3
"""msrv unit tests: model-free, fake engines only (no GPU, no exllamav3 import)."""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import serve


class FakeEngine:
    def __init__(self, output: str):
        self.output = output
        self.prompts = []
        self.resets = 0
        self.reset_gate_per_request = False

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def reset_gate(self) -> None:
        self.resets += 1

    def gate_stats(self):
        return {"n_spec": 3, "n_plain": 1}

    async def generate(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        if kwargs.get("reset_gate"):
            self.reset_gate()
        for i in range(0, len(self.output), 3):  # small chunks cut tags in half
            yield self.output[i:i + 3]


def run(coro):
    return asyncio.run(coro)


TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>"
            "{% endfor %}{% if tools %}TOOLS={{ tools|length }}{% endif %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


class ServeTests(unittest.TestCase):

    def test_template_rendering(self):
        out = serve.render_prompt(TEMPLATE, [{"role": "user", "content": "hi"}],
                                  [{"type": "function"}])
        self.assertEqual(out, "<|im_start|>user\nhi<|im_end|>TOOLS=1<|im_start|>assistant\n")

    def test_tool_call_arguments_string_is_normalized(self):
        # OpenAI sends arguments as a JSON string; the template must not double-encode it.
        out = serve.render_prompt(
            "{% for m in messages %}{{ m.tool_calls[0].function.arguments | tojson }}"
            "{% endfor %}",
            [{"role": "assistant", "content": "",
              "tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}]}], None)
        self.assertEqual(json.loads(out), {"a": 1})

    def test_parse_thinking_and_tool_calls(self):
        text = ('<think>reasoning with\nlines</think>Visible '
                '<tool_call><function=weather>{"name": "Paris",\n"x": 1}'
                '</function></tool_call>'
                '<tool_call><function=time>{"tz": "UTC"}</function></tool_call>')
        result = serve.parse_completion(text)
        self.assertEqual(result["reasoning_content"], "reasoning with\nlines")
        self.assertEqual(result["content"], "Visible ")
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "weather")
        self.assertEqual(json.loads(result["tool_calls"][0]["function"]["arguments"]),
                         {"name": "Paris", "x": 1})
        self.assertEqual(result["tool_calls"][1]["function"]["name"], "time")
        self.assertEqual(json.loads(result["tool_calls"][1]["function"]["arguments"]),
                         {"tz": "UTC"})

    def test_parse_plain_text_has_no_tool_calls(self):
        result = serve.parse_completion("just an answer mentioning <tool_call> inline")
        self.assertNotIn("tool_calls", result)
        self.assertEqual(result["content"], "just an answer mentioning <tool_call> inline")

    def test_stop_strings(self):
        self.assertEqual(serve.stop_text("hello STOP world", ["STOP"]), ("hello ", "STOP"))
        self.assertEqual(serve.stop_text("hello", ["STOP"]), ("hello", None))

    def test_sse_framing(self):
        self.assertEqual(serve.sse({"hello": "world"}), b'data: {"hello":"world"}\n\n')

    def test_config_max_position_handles_text_config(self):
        class C:
            config_dict: dict = {"text_config": {"max_position_embeddings": 65536}}
        self.assertEqual(serve.config_max_position(C), 65536)
        C.config_dict = {"max_position_embeddings": 4096}
        self.assertEqual(serve.config_max_position(C), 4096)
        C.config_dict = {}
        self.assertEqual(serve.config_max_position(C), 32768)

    def test_endpoints_models_health_and_usage_without_model(self):
        from aiohttp.test_utils import TestClient, TestServer
        engine = FakeEngine('<think>why</think>hello')
        app = serve.create_app(engine, "test-model", TEMPLATE)

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            models = await (await client.get("/v1/models")).json()
            self.assertEqual(models["data"][0]["id"], "test-model")
            health = await (await client.get("/health")).json()
            self.assertEqual(health["status"], "ok")
            self.assertEqual(health["spec_gate"]["n_spec"], 3)
            response = await client.post("/v1/chat/completions", json={
                "model": "test-model", "messages": [{"role": "user", "content": "hi"}]})
            body = await response.json()
            self.assertEqual(body["choices"][0]["message"]["content"], "hello")
            self.assertEqual(body["choices"][0]["message"]["reasoning_content"], "why")
            self.assertEqual(body["choices"][0]["finish_reason"], "stop")
            self.assertIn("usage", body)
            await client.close()

        run(check())

    def test_bad_requests(self):
        from aiohttp.test_utils import TestClient, TestServer

        async def post(body):
            app = serve.create_app(FakeEngine("x"), "m", TEMPLATE)
            client = TestClient(TestServer(app))
            await client.start_server()
            r = await client.post("/v1/chat/completions", json=body)
            await client.close()
            return r.status

        msgs = [{"role": "user", "content": "x"}]
        self.assertEqual(run(post({"model": "other", "messages": msgs})), 400)
        self.assertEqual(run(post({"model": "m", "messages": []})), 400)
        self.assertEqual(run(post({"model": "m", "messages": msgs, "stream": "yes"})), 400)

    def test_stream_thinking_and_tool_call(self):
        from aiohttp.test_utils import TestClient, TestServer
        engine = FakeEngine('<think>plan it</think>Sure.<tool_call><function=get_weather>'
                            '{"city": "Paris"}</function></tool_call>')
        app = serve.create_app(engine, "m", TEMPLATE)

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            raw = (await response.read()).decode()
            await client.close()
            return raw

        raw = run(check())
        self.assertTrue(raw.endswith("data: [DONE]\n\n"))
        chunks = [json.loads(line[6:]) for line in raw.split("\n") if line.startswith("data: {")]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        self.assertEqual("".join(d.get("reasoning_content", "") for d in deltas), "plan it")
        self.assertEqual("".join(d.get("content", "") for d in deltas), "Sure.")
        self.assertGreater(sum("reasoning_content" in d for d in deltas), 1)  # incremental
        calls = [d["tool_calls"] for d in deltas if "tool_calls" in d]
        self.assertEqual(calls[0][0]["function"]["name"], "get_weather")
        self.assertEqual(calls[0][0]["index"], 0)
        self.assertEqual([c["choices"][0]["finish_reason"] for c in chunks][-1], "tool_calls")
        self.assertTrue(all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1]))
        self.assertIn("usage", chunks[-1])

    def test_stream_and_parse_agree_on_bare_think_close(self):
        """No implicit think turn: a bare </think> is literal content in both paths."""
        from aiohttp.test_utils import TestClient, TestServer
        engine = FakeEngine("</think>answer")
        app = serve.create_app(engine, "m", TEMPLATE)

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            raw = (await r.read()).decode()
            await client.close()
            return raw

        self.assertEqual(serve.parse_completion("</think>answer")["content"], "</think>answer")
        chunks = [json.loads(l[6:]) for l in run(check()).split("\n") if l.startswith("data: {")]
        deltas = [c["choices"][0]["delta"] for c in chunks]
        self.assertEqual("".join(d.get("reasoning_content", "") for d in deltas), "")
        self.assertEqual("".join(d.get("content", "") for d in deltas), "</think>answer")

    def test_finish_reason_length_at_max_tokens(self):
        """A reply cut by max_tokens reports "length", streaming and not; a short reply stays "stop"."""
        from aiohttp.test_utils import TestClient, TestServer

        async def call(output, max_tokens, stream):
            app = serve.create_app(FakeEngine(output), "m", TEMPLATE)
            client = TestClient(TestServer(app))
            await client.start_server()
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": stream, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": "hi"}]})
            if stream:
                raw = (await r.read()).decode()
                out = [json.loads(l[6:]) for l in raw.split("\n") if l.startswith("data: {")][-1]
            else:
                out = await r.json()
            await client.close()
            return out["choices"][0]["finish_reason"]

        for stream in (False, True):
            self.assertEqual(run(call("one two three four", 4, stream)), "length")
            self.assertEqual(run(call("one two", 4, stream)), "stop")

    def test_stop_string_cuts_stream(self):
        from aiohttp.test_utils import TestClient, TestServer
        engine = FakeEngine("alpha STOP beta")
        app = serve.create_app(engine, "m", TEMPLATE)

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "stop": "STOP", "messages": [{"role": "user", "content": "hi"}]})
            body = await r.json()
            await client.close()
            return body

        self.assertEqual(run(check())["choices"][0]["message"]["content"], "alpha ")

    def test_gate_reset_flag_is_forwarded(self):
        from aiohttp.test_utils import TestClient, TestServer
        engine = FakeEngine("x")
        engine.reset_gate_per_request = True
        app = serve.create_app(engine, "m", TEMPLATE)

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            await client.post("/v1/chat/completions",
                              json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
            await client.close()

        run(check())
        self.assertEqual(engine.resets, 1)


if __name__ == "__main__":
    unittest.main()
