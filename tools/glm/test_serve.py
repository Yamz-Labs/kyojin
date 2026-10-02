#!/usr/bin/env python3
"""Model-free glm-serve tests. Fake engines emit deterministic GLM XML."""
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

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    async def generate(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        for i in range(0, len(self.output), 3):  # small chunks cut tags in half
            yield self.output[i:i + 3]


def run(coro):
    return asyncio.run(coro)


class ServeTests(unittest.TestCase):
    template = "{% for m in messages %}<|im_start|>{{ m.role }}: {{ m.content }}<|im_end|>{% endfor %}{% if tools %}TOOLS={{ tools|length }}{% endif %}"

    def test_template_rendering(self):
        out = serve.render_prompt(self.template, [{"role": "user", "content": "hi"}], [{"type": "function"}])
        self.assertEqual(out, "<|im_start|>user: hi<|im_end|>TOOLS=1")

    def test_parse_thinking_and_tool_calls(self):
        text = ('<think>reasoning with\nlines</think>Visible '
                '<tool_call>weather<arg_key>city</arg_key><arg_value>{"name": "Paris",\n"x": 1}</arg_value></tool_call>'
                '<tool_call>time<arg_key>tz</arg_key><arg_value>UTC</arg_value></tool_call>')
        result = serve.parse_completion(text)
        self.assertEqual(result["reasoning_content"], "reasoning with\nlines")
        self.assertEqual(result["content"], "Visible ")
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "weather")
        self.assertEqual(json.loads(result["tool_calls"][0]["function"]["arguments"]), {"city": {"name": "Paris", "x": 1}})
        self.assertEqual(result["tool_calls"][1]["function"]["name"], "time")

    def test_warmup_runs_one_long_prompt(self):
        engine = FakeEngine("ok")
        elapsed = serve.warmup(engine, self.template, prompt_tokens=300, decode_tokens=4)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(len(engine.prompts), 1)
        self.assertGreaterEqual(engine.count_tokens(engine.prompts[0]), 290)

    def test_dense_tune_path_follows_the_cpp_tuner(self):
        from unittest import mock
        with mock.patch.dict("os.environ", {"EXL3_DENSE_GEMM_TUNE_FILE": "/x/t.txt"}):
            self.assertEqual(serve.dense_tune_path(), Path("/x/t.txt"))
        with mock.patch.dict("os.environ", {"HOME": "/h"}, clear=True):
            self.assertEqual(serve.dense_tune_path(), Path("/h/.cache/exllamav3/dense_gemm_tune.txt"))
        with mock.patch.dict("os.environ", {"EXL3_DENSE_GEMM_TUNE_FILE": "/nonexistent/t.txt"}):
            self.assertEqual(serve.prime_dense_tune(10.0), (0, 0.0))

    def test_stop_strings(self):
        self.assertEqual(serve.stop_text("hello STOP world", ["STOP"]), ("hello ", "STOP"))
        self.assertEqual(serve.stop_text("hello", ["STOP"]), ("hello", None))

    def test_sse_framing(self):
        frame = serve.sse({"hello": "world"})
        self.assertEqual(frame, b'data: {"hello":"world"}\n\n')

    def test_endpoints_and_usage_without_model(self):
        from aiohttp.test_utils import TestClient, TestServer
        engine = FakeEngine('<think>why</think>hello')
        app = serve.create_app(engine, "test-model", self.template)

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            models = await (await client.get("/v1/models")).json()
            self.assertEqual(models["data"][0]["id"], "test-model")
            response = await client.post("/v1/chat/completions", json={
                "model": "test-model", "messages": [{"role": "user", "content": "hi"}]})
            body = await response.json()
            self.assertEqual(body["choices"][0]["message"]["content"], "hello")
            self.assertEqual(body["choices"][0]["message"]["reasoning_content"], "why")
            self.assertIn("usage", body)
            await client.close()

        run(check())

    def test_stream_open_think_and_tool_call(self):
        from aiohttp.test_utils import TestClient, TestServer
        engine = FakeEngine('plan it</think>Sure.<tool_call>get_weather<arg_key>city</arg_key>'
                            '<arg_value>Paris</arg_value></tool_call>')
        app = serve.create_app(engine, "m", self.template + "<think>")

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


if __name__ == "__main__":
    unittest.main()
