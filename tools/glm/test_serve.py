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



def _write_draft2s_pack(path, vocab, width):
    """A header-only draft2s pack (tools/qwen/draft2s_pack.py metadata layout) for the sidecar tests."""
    import json as _json, struct
    hdr = _json.dumps({"__metadata__": {"format": "draft2s-v1", "vocab": str(vocab), "in_features": str(width), "m": "512"}}).encode()
    path.write_bytes(struct.pack("<Q", len(hdr)) + hdr)


class FakeGenerator:
    num_draft_tokens = 2  # /metrics derives the round count from the engine's window


class FakeEngine:
    def __init__(self, output: str):
        self.output = output
        self.prompts = []
        self.greedy_generator = FakeGenerator()

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def prompt_ids(self, prompt: str):
        """The ids the routes count prompt tokens with (the real engines read them from an encode cache).

        Placeholders of count_tokens() length: the routes only ask for numel()."""
        import torch
        return torch.zeros(self.count_tokens(prompt), dtype=torch.long)

    async def generate(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        self.last_stats = {"cached_tokens": 8, "new_tokens": 4, "prompt_tokens": 20,
                           "time_prefill": 0.5, "time_generate": 1.0,
                           "accepted_draft_tokens": 6, "rejected_draft_tokens": 6}
        for i in range(0, len(self.output), 3):  # small chunks cut tags in half
            yield self.output[i:i + 3]


def run(coro):
    return asyncio.run(coro)


def metrics_samples(text: str) -> dict:
    return {line.split()[0]: float(line.split()[1]) for line in text.splitlines()
            if line and not line.startswith("#")}


class ServeTests(unittest.TestCase):
    template = "{% for m in messages %}<|im_start|>{{ m.role }}: {{ m.content }}<|im_end|>{% endfor %}{% if tools %}TOOLS={{ tools|length }}{% endif %}"



    def test_prompt_longer_than_the_context_is_a_clear_400(self):
        from aiohttp.test_utils import TestClient, TestServer

        class Small(FakeEngine):
            ctx = 256

        async def check():
            client = TestClient(TestServer(serve.create_app(Small("x"), "m", self.template)))
            await client.start_server()
            out = []
            for stream in (False, True):
                r = await client.post("/v1/chat/completions", json={
                    "model": "m", "stream": stream, "messages": [{"role": "user", "content": "word " * 400}]})
                out.append((r.status, await r.json()))
            await client.close()
            return out

        for status, body in run(check()):
            self.assertEqual(status, 400)
            self.assertEqual(body["error"]["type"], "invalid_request_error")
            self.assertIn("context length exceeded: prompt is", body["error"]["message"])
            self.assertIn("the server context is 256", body["error"]["message"])

    def test_unknown_path_and_wrong_method_answer_404_405_in_json(self):
        from aiohttp.test_utils import TestClient, TestServer

        async def check():
            client = TestClient(TestServer(serve.create_app(FakeEngine("x"), "m", self.template)))
            await client.start_server()
            out = []
            for method, path in (("GET", "/nope"), ("POST", "/nope"), ("POST", "/health"), ("GET", "/v1/chat/completions")):
                r = await client.request(method, path)
                out.append((r.status, r.headers["Content-Type"], await r.json()))
            await client.close()
            return out

        for (status, ctype, body), want in zip(run(check()), (404, 404, 405, 405)):
            self.assertEqual(status, want)
            self.assertTrue(ctype.startswith("application/json"))
            self.assertEqual((body["error"]["code"], body["error"]["type"]), (want, "invalid_request_error"))
            self.assertTrue(body["error"]["message"])

    def test_eh_sidecar_source_order(self):
        import os, tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as home:
            pack = Path(home) / "pack"
            pack.mkdir()
            with mock.patch.dict(os.environ, {"HOME": home}):
                os.environ.pop("EXL3_MTP_EH_FP16", None)
                # nothing anywhere: variable stays unset, no FileNotFoundError at load
                self.assertEqual(serve.apply_eh_sidecar_default(str(pack)), "absent")
                self.assertNotIn("EXL3_MTP_EH_FP16", os.environ)
                # old default path only
                old = Path(home) / "models" / "glm53-mtp-eh-proj-bf16.safetensors"
                old.parent.mkdir()
                old.write_bytes(b"x")
                self.assertEqual(serve.apply_eh_sidecar_default(str(pack)), "default")
                self.assertEqual(os.environ["EXL3_MTP_EH_FP16"], str(old))
                # file in the model folder wins over the old path
                os.environ.pop("EXL3_MTP_EH_FP16")
                (pack / serve.EH_SIDECAR_IN_PACK).write_bytes(b"x")
                self.assertEqual(serve.apply_eh_sidecar_default(str(pack)), "pack")
                self.assertEqual(os.environ["EXL3_MTP_EH_FP16"], str(pack / "mtp_eh_proj.st"))
                # explicit value (even a missing file, even "0") is never touched
                for value in ("/nope/x.safetensors", "0"):
                    os.environ["EXL3_MTP_EH_FP16"] = value
                    self.assertEqual(serve.apply_eh_sidecar_default(str(pack)), "explicit")
                    self.assertEqual(os.environ["EXL3_MTP_EH_FP16"], value)
        self.assertNotIn("EXL3_MTP_EH_FP16", serve.SPEED_ENV)
        self.assertFalse(serve.EH_SIDECAR_IN_PACK.endswith(".safetensors"))

    def test_draft2s_sidecar_source_order(self):
        import os, tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}):
            pack = Path(d)
            (pack / "config.json").write_text(json.dumps({"text_config": {"vocab_size": 154880, "hidden_size": 4096}}))
            os.environ.pop("EXL3_DRAFT2S", None)
            self.assertEqual(serve.apply_draft2s_default(str(pack)), "absent")
            self.assertNotIn("EXL3_DRAFT2S", os.environ)
            # a pack built for another model (Qwen's head) is not used
            _write_draft2s_pack(pack / serve.DRAFT2S_IN_PACK, 248320, 2560)
            self.assertEqual(serve.apply_draft2s_default(str(pack)), "mismatch")
            self.assertNotIn("EXL3_DRAFT2S", os.environ)
            _write_draft2s_pack(pack / serve.DRAFT2S_IN_PACK, 154880, 4096)
            self.assertEqual(serve.apply_draft2s_default(str(pack)), "pack")
            self.assertEqual(os.environ["EXL3_DRAFT2S"], str(pack / "draft2s.st"))
            # explicit path never touched; "0" / "off" / "" = off (empty, so the engine's truthiness check skips the head)
            os.environ["EXL3_DRAFT2S"] = "/nope/x.st"
            self.assertEqual(serve.apply_draft2s_default(str(pack)), "explicit")
            self.assertEqual(os.environ["EXL3_DRAFT2S"], "/nope/x.st")
            for value in ("0", "off", ""):
                os.environ["EXL3_DRAFT2S"] = value
                self.assertEqual(serve.apply_draft2s_default(str(pack)), "off")
                self.assertEqual(os.environ["EXL3_DRAFT2S"], "")
        self.assertNotIn("EXL3_DRAFT2S", serve.SPEED_ENV)
        self.assertFalse(serve.DRAFT2S_IN_PACK.endswith(".safetensors"))

    def test_stream_engine_error_still_ends_with_finish_reason(self):
        from aiohttp.test_utils import TestClient, TestServer

        class Boom(FakeEngine):
            async def generate(self, prompt: str, **kwargs):
                yield "partial "
                raise RuntimeError("kaput")

        async def check():
            client = TestClient(TestServer(serve.create_app(Boom("x"), "m", self.template)))
            await client.start_server()
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            raw = (await r.read()).decode()
            await client.close()
            return raw

        raw = run(check())
        self.assertTrue(raw.endswith("data: [DONE]\n\n"))
        chunks = [json.loads(l[6:]) for l in raw.split("\n") if l.startswith("data: {")]
        self.assertTrue(any("kaput" in c.get("error", {}).get("message", "") for c in chunks))
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "error")

    def test_max_completion_tokens_is_honored(self):
        self.assertEqual(serve.token_limit({"max_completion_tokens": 7, "max_tokens": 9}), 7)
        self.assertEqual(serve.token_limit({"max_tokens": 9}), 9)
        self.assertEqual(serve.token_limit({"max_tokens": None}), 32768)
        from aiohttp.test_utils import TestClient, TestServer

        class Spy(FakeEngine):
            async def generate(self, prompt: str, **kwargs):
                self.kwargs = kwargs
                async for piece in super().generate(prompt, **kwargs):
                    yield piece

        engine = Spy("one two three four")

        async def check():
            client = TestClient(TestServer(serve.create_app(engine, "m", self.template)))
            await client.start_server()
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "max_completion_tokens": 4, "messages": [{"role": "user", "content": "hi"}]})
            body = await r.json()
            await client.close()
            return body

        body = run(check())
        self.assertEqual(engine.kwargs["max_tokens"], 4)
        self.assertEqual(body["choices"][0]["finish_reason"], "length")

    def test_reply_budget_is_clamped_to_the_context_room(self):
        from aiohttp.test_utils import TestClient, TestServer

        class Spy(FakeEngine):
            ctx = 20
            num_draft = 1

            async def generate(self, prompt: str, **kwargs):
                self.kwargs = kwargs
                async for piece in super().generate(prompt, **kwargs):
                    yield piece

        engine = Spy("one two")

        async def post(content, **extra):
            client = TestClient(TestServer(serve.create_app(engine, "m", self.template)))
            await client.start_server()
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": content}], **extra})
            status = r.status
            await r.read()
            await client.close()
            return status

        # room = ctx - prompt - 1 slot - 1 draft token
        self.assertEqual(run(post("a b c d", max_completion_tokens=32768)), 200)
        used = engine.count_tokens(engine.prompts[-1])
        self.assertEqual(engine.kwargs["max_tokens"], 20 - used - 1 - 1)
        self.assertEqual(run(post("a b c d", max_tokens=3)), 200)
        self.assertEqual(engine.kwargs["max_tokens"], 3)
        engine.kwargs = None
        self.assertEqual(run(post(" ".join("w" * 30))), 400)
        self.assertIsNone(engine.kwargs)

    def test_template_rendering(self):
        out = serve.render_prompt(self.template, [{"role": "user", "content": "hi"}], [{"type": "function"}])
        self.assertEqual(out, "<|im_start|>user: hi<|im_end|>TOOLS=1")

    def test_developer_role_renders_as_system(self):
        dev = serve.render_prompt(self.template, [{"role": "developer", "content": "be brief"}, {"role": "user", "content": "hi"}], None)
        sys_ = serve.render_prompt(self.template, [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], None)
        self.assertEqual(dev, sys_)

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
            self.assertEqual(body["choices"][0]["finish_reason"], "stop")
            cut = await (await client.post("/v1/chat/completions", json={
                "model": "test-model", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]})).json()
            self.assertEqual(cut["choices"][0]["finish_reason"], "length")
            await client.close()

        run(check())

    def test_stream_parallel_tool_calls_one_message_each(self):
        from aiohttp.test_utils import TestClient, TestServer
        call = '<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>'
        app = serve.create_app(FakeEngine("plan</think>Sure." + call * 3), "m", self.template + "<think>")

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            raw = (await response.read()).decode()
            await client.close()
            return raw
        chunks = [json.loads(l[6:]) for l in run(check()).split("\n") if l.startswith("data: {")]
        msgs = [c["choices"][0]["delta"]["tool_calls"] for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
        self.assertEqual([len(m) for m in msgs], [1, 1, 1])
        self.assertEqual([m[0]["index"] for m in msgs], [0, 1, 2])

    def test_held_tail_is_only_a_possible_marker_prefix(self):
        marks = ["</think>", "<think>", serve.TOOL_OPEN, "STOP"]
        self.assertEqual(serve.held_tail("plain words", marks), 0)
        self.assertEqual(serve.held_tail("abc <tool_ca", marks), len("<tool_ca"))
        self.assertEqual(serve.held_tail("abc <", marks), 1)
        self.assertEqual(serve.held_tail("abc ST", marks), 2)
        self.assertEqual(serve.held_tail("abc </think", marks), len("</think"))
        self.assertEqual(serve.held_tail("<", marks), 1)
        self.assertEqual(serve.held_tail("", marks), 0)

    def test_stream_first_text_is_not_held_back_by_a_marker_length(self):
        from aiohttp.test_utils import TestClient, TestServer
        engine = FakeEngine('plan it</think>Sure.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>')
        app = serve.create_app(engine, "m", self.template + "<think>")

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            response = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            raw = (await response.read()).decode()
            await client.close()
            return raw

        chunks = [json.loads(l[6:]) for l in run(check()).split("\n") if l.startswith("data: {")]
        reasoning = [c["choices"][0]["delta"]["reasoning_content"] for c in chunks if "reasoning_content" in c["choices"][0]["delta"]]
        self.assertEqual(reasoning[0], "pla")   # the first 3-character engine chunk goes out at once

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


    def test_client_disconnect_cancels_the_job(self):
        """Stream and non-stream drops stop generation within a few steps, a job cancelled while queued
        never runs, and the request after them is served."""
        from aiohttp.test_utils import TestClient, TestServer

        class SlowEngine(FakeEngine):
            steps = 0
            stopped = None

            async def generate(self, prompt, **kwargs):
                self.prompts.append(prompt)
                cancel = kwargs.get("cancel")
                if "slow" not in prompt:
                    yield "ok"
                    return
                self.stopped = None
                for _ in range(1000):
                    if cancel is not None and cancel.is_set():
                        self.stopped = "cancelled"
                        return
                    self.steps += 1
                    await asyncio.sleep(0.01)
                    yield "t "
                self.stopped = "finished"

        def body(text, stream):
            return {"model": "m", "stream": stream, "messages": [{"role": "user", "content": text}]}

        async def settle(engine, want_steps_at_most):
            for _ in range(200):
                if engine.stopped is not None:
                    break
                await asyncio.sleep(0.02)
            self.assertEqual(engine.stopped, "cancelled")
            self.assertLess(engine.steps, want_steps_at_most)

        async def check():
            engine = SlowEngine("x")
            client = TestClient(TestServer(serve.create_app(engine, "m", self.template)))
            await client.start_server()
            # 1. streaming drop mid-stream
            response = await client.post("/v1/chat/completions", json=body("slow a", True))
            await response.content.readany()
            await asyncio.sleep(0.15)
            response.close()
            await settle(engine, 100)
            # 2. a following request is served at once
            ok = await asyncio.wait_for(client.post("/v1/chat/completions", json=body("hi", False)), 5)
            self.assertEqual(ok.status, 200)
            self.assertEqual((await ok.json())["choices"][0]["message"]["content"], "ok")
            # 3. non-streaming drop
            engine.steps = 0
            try:
                await asyncio.wait_for(client.post("/v1/chat/completions", json=body("slow b", False)), 0.3)
            except asyncio.TimeoutError:
                pass
            await settle(engine, 100)
            # 4. a job cancelled while queued never reaches the engine
            engine.steps, engine.prompts = 0, []
            first = await client.post("/v1/chat/completions", json=body("slow c", True))
            await first.content.readany()
            try:
                await asyncio.wait_for(client.post("/v1/chat/completions", json=body("slow queued", False)), 0.3)
            except asyncio.TimeoutError:
                pass
            first.close()
            ok = await asyncio.wait_for(client.post("/v1/chat/completions", json=body("hi again", False)), 5)
            self.assertEqual(ok.status, 200)
            self.assertFalse(any("slow queued" in p for p in engine.prompts))
            await client.close()

        run(check())

    def test_metrics_endpoint(self):
        from aiohttp.test_utils import TestClient, TestServer
        engine = FakeEngine('hello there')
        app = serve.create_app(engine, "m", self.template)

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            await client.post("/v1/chat/completions", json={
                "model": "m", "messages": [{"role": "user", "content": "one two three four"}]})
            resp = await client.get("/metrics")
            text = await resp.text()
            await client.close()
            return resp, text

        resp, text = run(check())
        self.assertEqual(resp.status, 200)
        self.assertTrue(resp.headers["Content-Type"].startswith("text/plain"))
        import serve_metrics
        for name, mtype, _ in serve_metrics.METRICS:
            self.assertIn(f"# HELP {name} ", text)
            self.assertIn(f"# TYPE {name} {mtype}", text)
        v = metrics_samples(text)
        st = engine.last_stats
        prompt_tokens = engine.count_tokens(engine.prompts[-1])
        serve_metrics.assert_reported(
            v, self, prompt_tokens=prompt_tokens, cached=st["cached_tokens"],
            predicted=st["new_tokens"], prefill_s=st["time_prefill"], generate_s=st["time_generate"],
            drafts=(st["accepted_draft_tokens"] + st["rejected_draft_tokens"]) // 2,
            draft_tokens=st["accepted_draft_tokens"] + st["rejected_draft_tokens"],
            accepted=st["accepted_draft_tokens"])
        self.assertEqual(v["llamacpp:requests_processing"], 0)
        self.assertEqual(v["llamacpp:requests_deferred"], 0)
        self.assertEqual(v["llamacpp:kv_cache_usage_ratio"], 0)  # fake engine: no get_cache_stats

    def test_stream_handler_finishes_after_stream(self):
        """After a streamed reply the handler returns; it must not fall into the non-stream path
        and wait forever on an empty queue."""
        from aiohttp.test_utils import TestClient, TestServer
        engine = FakeEngine("hello there")
        app = serve.create_app(engine, "m", self.template)

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            resp = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            await resp.read()
            await asyncio.sleep(0.2)
            pending = [t for t in asyncio.all_tasks()
                       if "_handle_request" in repr(t.get_coro()) and not t.done()]
            await client.close()
            return pending

        self.assertEqual(run(check()), [])

    def test_cancelled_request_does_not_recount_previous_stats(self):
        """An engine that returns on cancel without publishing stats leaves the previous request's
        last_stats behind; the worker clears it before each job so nothing is counted twice."""
        from aiohttp.test_utils import TestClient, TestServer

        class CancelEngine(FakeEngine):
            async def generate(self, prompt, **kwargs):
                if "cancelled" in prompt:
                    return
                    yield
                async for d in super().generate(prompt, **kwargs):
                    yield d

        engine = CancelEngine("hello there")
        app = serve.create_app(engine, "m", self.template)

        async def check():
            client = TestClient(TestServer(app))
            await client.start_server()
            for text in ("one two", "cancelled"):
                await client.post("/v1/chat/completions", json={
                    "model": "m", "messages": [{"role": "user", "content": text}]})
            text = await (await client.get("/metrics")).text()
            await client.close()
            return text

        v = metrics_samples(run(check()))
        self.assertEqual(v["llamacpp:tokens_predicted_total"], 4)  # one request, not two

    def test_two_queued_requests_are_both_counted(self):
        """Back-to-back requests each get their own timings handed over by the worker, so both are
        counted in full even though engine.last_stats is a single shared attr reset per job."""
        from aiohttp.test_utils import TestClient, TestServer

        class QueuedEngine(FakeEngine):
            def __init__(self):
                super().__init__("")
                self.calls = 0

            async def generate(self, prompt, **kwargs):
                self.prompts.append(prompt)
                self.calls += 1
                n = 3 * self.calls  # distinct per request: 3 then 6
                self.last_stats = {"cached_tokens": self.calls, "new_tokens": n,
                                   "prompt_tokens": 20, "time_prefill": 0.5, "time_generate": 1.0,
                                   "accepted_draft_tokens": 2, "rejected_draft_tokens": 2}
                await asyncio.sleep(0)  # let the second request sit queued behind this one
                yield f"reply {self.calls}"

        async def check():
            engine = QueuedEngine()
            client = TestClient(TestServer(serve.create_app(engine, "m", self.template)))
            await client.start_server()
            bodies = [
                {"model": "m", "messages": [{"role": "user", "content": "one two three four"}]},
                {"model": "m", "messages": [{"role": "user", "content": "five six seven eight nine"}]},
            ]
            responses = await asyncio.gather(*(client.post("/v1/chat/completions", json=b) for b in bodies))
            for r in responses:
                self.assertEqual(r.status, 200)
            # each reply reports its own generation, not the shared attr's last value (which would be 6 for both)
            payloads = await asyncio.gather(*(r.json() for r in responses))
            predicted = sorted(p["timings"]["predicted_n"] for p in payloads)
            self.assertEqual(predicted, [3, 6])
            resp = await client.get("/metrics")
            text = await resp.text()
            await client.close()
            return engine, text

        engine, text = run(check())
        v = metrics_samples(text)
        # both generations counted in full: predicted = 3 + 6, not a prompt-only first request
        self.assertEqual(v["llamacpp:tokens_predicted_total"], 9.0)
        self.assertEqual(v["llamacpp:prompt_tokens_total"], float(sum(engine.count_tokens(p) for p in engine.prompts)))

    def test_image_request_without_vision_tower_is_refused_not_dropped(self):
        from aiohttp.test_utils import TestClient, TestServer
        tpl = self.template.replace("{{ m.content }}", "{% if m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}{% if p.type == 'text' %}{{ p.text }}{% else %}" + serve.IMAGE_TRIPLE + "{% endif %}{% endfor %}{% endif %}")
        body = {"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}

        async def check():
            engine = FakeEngine("x")
            client = TestClient(TestServer(serve.create_app(engine, "m", tpl)))
            await client.start_server()
            r = await client.post("/v1/chat/completions", json=body)
            out = (r.status, (await r.json())["error"]["message"], engine.prompts)
            h = await (await client.get("/health")).json()
            await client.close()
            return out, h

        (status, message, prompts), health = run(check())
        self.assertEqual(status, 400)
        self.assertIn("vision tower", message)
        self.assertEqual(prompts, [])          # nothing was generated
        self.assertFalse(health["vision"])

    def test_video_and_audio_parts_are_refused(self):
        for kind in ("video_url", "input_audio"):
            with self.assertRaises(serve.web.HTTPBadRequest):
                serve.extract_images([{"role": "user", "content": [{"type": "text", "text": "x"}, {"type": kind, kind: {}}]}])

    def test_load_image_sources(self):
        import base64, io, tempfile
        from PIL import Image
        buf = io.BytesIO(); Image.new("RGB", (4, 3), (255, 0, 0)).save(buf, "PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        im, key = serve.load_image(uri)
        self.assertEqual(im.size, (4, 3))
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            f.write(buf.getvalue()); f.flush()
            self.assertEqual(serve.load_image(f.name)[1], key)
            self.assertEqual(serve.load_image("file://" + f.name)[1], key)
        for bad in ("data:image/png;base64,AAAA", "/nonexistent.png", "https://example.com/a.png"):
            with self.assertRaises(serve.web.HTTPBadRequest):
                serve.load_image(bad)



class ImageEmbeddingReuseTests(unittest.TestCase):
    """The same picture must come back as the same embedding object, so the prompt keeps its token ids."""

    def make_engine(self):
        import itertools
        import torch
        from image_cache import ImageEmbeddingCache
        ids, calls = itertools.count(1_000_000_000, 100), []

        class Emb:
            def __init__(self):
                self.first_index = next(ids)
                self.text_alias = f"<$EMB_{self.first_index}$>"
                self.mm_length = 3
                self.embeddings = torch.zeros(3, 4, dtype=torch.half)
                self.deepstack_embeddings = None

        class Tower:
            def get_image_embeddings(self, *args, **kw):
                calls.append(1)
                return Emb()

        eng = object.__new__(serve.ResidentEngine)
        eng.emb_cache, eng.vision, eng.tokenizer = ImageEmbeddingCache(), Tower(), None
        return eng, calls

    def test_embed_reuses_ids_and_encodes_once(self):
        from PIL import Image
        eng, calls = self.make_engine()
        img = Image.new("RGB", (64, 64))
        a = eng._embed([(img, "k1")])
        b = eng._embed([(img, "k1"), (img, "k2")])
        self.assertIs(a[0], b[0])
        self.assertNotEqual(b[0].first_index, b[1].first_index)
        self.assertEqual(len(calls), 2)


class SessionsTests(unittest.TestCase):
    """--sessions N: the app runs N requests at once; the engine driver routes one shared Generator's events."""
    template = ServeTests.template

    def test_requests_overlap_and_slot_action_waits_for_them(self):
        from aiohttp.test_utils import TestClient, TestServer

        class Slow(FakeEngine):
            sessions = 3
            live = 0
            peak = 0

            async def generate(self, prompt, stats=None, **kw):
                Slow.live += 1
                Slow.peak = max(Slow.peak, Slow.live)
                try:
                    await asyncio.sleep(0.3)
                    stats.update({"new_tokens": 1, "prompt_tokens": 2})
                    yield "ok"
                finally:
                    Slow.live -= 1

        class Store:
            busy = False
            seen_live = None

            def slots(self):
                return []

            def erase(self):
                Store.seen_live = Slow.live
                return {"id_slot": 0, "n_erased": 0}

        async def check():
            eng = Slow("x")
            eng.slot_store = Store()
            client = TestClient(TestServer(serve.create_app(eng, "m", self.template)))
            await client.start_server()
            body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            reqs = [asyncio.create_task(client.post("/v1/chat/completions", json=body)) for _ in range(4)]
            await asyncio.sleep(0.15)
            act = await client.post("/slots/0?action=erase")
            res = [await r for r in reqs]
            out = ([r.status for r in res], act.status)
            await client.close()
            return out

        statuses, act = run(check())
        self.assertEqual(statuses, [200] * 4)
        self.assertEqual(act, 200)
        self.assertEqual(Slow.peak, 3)          # 4 requests, 3 sessions
        self.assertEqual(Store.seen_live, 0)    # the slot action ran with no request inside generate()

    def test_driver_routes_events_by_serial_and_cancel_keeps_the_others(self):
        class Job:
            serial_number = None

            def __init__(self, n):
                self.n = n

        class Gen:
            def __init__(self):
                self.active_jobs, self.pending_jobs, self.k, self.cancelled = [], [], 0, []

            def enqueue(self, job):
                job.serial_number = len(self.active_jobs) + len(self.cancelled) + 100
                self.active_jobs.append(job)
                return job.serial_number

            def num_remaining_jobs(self):
                return len(self.active_jobs) + len(self.pending_jobs)

            def cancel(self, job):
                self.active_jobs.remove(job)
                self.cancelled.append(job)

            def clear_queue(self):
                self.active_jobs.clear()

            def iterate(self):
                self.k += 1
                out = []
                for j in list(self.active_jobs):
                    ev = {"serial": j.serial_number, "text": f"{j.n}.{self.k} "}
                    if self.k >= j.n:
                        ev["eos"] = True
                        ev["new_tokens"] = self.k
                        self.active_jobs.remove(j)
                    out.append(ev)
                return out

        async def check():
            eng = object.__new__(serve.ResidentEngine)
            eng.sessions, eng.greedy_generator = 2, Gen()

            async def one(n, cancel=None):
                st, txt = {}, []
                async for piece in eng._run_shared(Job(n), type("T", (), {"numel": lambda self: 1})(), cancel, st, 0.0, None):
                    txt.append(piece)
                return "".join(txt), st

            stop = asyncio.Event()
            a = asyncio.create_task(one(2))
            b = asyncio.create_task(one(50, stop))
            ra = await a
            stop.set()
            rb = await b
            return ra, rb, eng.greedy_generator

        (ta, sa), (tb, sb), gen = run(check())
        self.assertEqual(ta, "2.1 2.2 ")
        self.assertEqual(sa["new_tokens"], 2)
        self.assertTrue(tb.startswith("50.1 ") and "eos" not in sb)
        self.assertEqual(gen.num_remaining_jobs(), 0)   # the cancelled job left the Generator
        self.assertEqual(len(gen.cancelled), 1)

    def test_driver_follows_a_rebuilt_generator(self):
        """cache_salt change: _rebuild_generator swaps the Generator while the driver runs. Jobs enqueued afterwards must
        be stepped (the driver once kept the old Generator and the request waited forever)."""
        class Job:
            serial_number = None

        class Gen:
            made = 0

            def __init__(self):
                Gen.made += 1
                self.active_jobs, self.pending_jobs, self.k = [], [], 0

            def enqueue(self, job):
                job.serial_number = 100 + Gen.made
                self.active_jobs.append(job)
                return job.serial_number

            def num_remaining_jobs(self):
                return len(self.active_jobs)

            def cancel(self, job):
                self.active_jobs.remove(job)

            def clear_queue(self):
                self.active_jobs.clear()

            def iterate(self):
                out = []
                for j in list(self.active_jobs):
                    self.active_jobs.remove(j)
                    out.append({"serial": j.serial_number, "text": "ok", "eos": True, "new_tokens": 1})
                return out

        async def check():
            eng = object.__new__(serve.ResidentEngine)
            eng.sessions, eng.greedy_generator = 2, Gen()
            eng.torch = type("T", (), {"cuda": type("C", (), {"empty_cache": staticmethod(lambda: None)})})()
            eng._make_generator = Gen

            async def one():
                txt = []
                async for piece in eng._run_shared(Job(), type("T", (), {"numel": lambda self: 1})(), asyncio.Event(), {}, 0.0, None):
                    txt.append(piece)
                return "".join(txt)

            first = await one()
            await eng.reset_prefix_cache()
            second = await asyncio.wait_for(one(), 5)
            return first, second

        self.assertEqual(run(check()), ("ok", "ok"))


class VllmCompatTests(unittest.TestCase):
    """POST /tokenize, /v1/completions with token-id prompts, cache_salt (model-free: fake engine)."""
    template = ServeTests.template

    class Eng(FakeEngine):
        vocab_size = 1000
        ctx = 4096

        def __init__(self, output="abc", sessions=1, delay=0.0):
            super().__init__(output)
            self.sessions, self.delay, self.resets, self.active, self.calls = sessions, delay, [], 0, []

        def encode_ids(self, text, add_special_tokens=False):
            return ([0] if add_special_tokens else []) + [len(w) for w in text.split()]

        async def reset_prefix_cache(self):
            self.resets.append(self.active)
            await asyncio.sleep(self.delay)

        async def generate(self, prompt, stats=None, **kw):
            self.active += 1
            self.calls.append((prompt, kw))
            try:
                await asyncio.sleep(self.delay)
                st = {"cached_tokens": 99, "new_tokens": 3, "prompt_tokens": 3, "eos_reason": "max_new_tokens"}
                self.last_stats = st
                if stats is not None:
                    stats.update(st)
                for ch in self.output:
                    yield ch
            finally:
                self.active -= 1

    def call(self, engine, requests):
        from aiohttp.test_utils import TestClient, TestServer

        async def go():
            client = TestClient(TestServer(serve.create_app(engine, "m", self.template)))
            await client.start_server()
            try:
                out = []
                for path, payload in requests:
                    r = await client.post(path, json=payload)
                    out.append((r.status, await r.json() if r.content_type == "application/json" else await r.text()))
                return out
            finally:
                await client.close()
        return run(go())

    def one(self, engine, path, payload):
        return self.call(engine, [(path, payload)])[0]

    def test_tokenize_reply_shape_and_errors(self):
        eng = self.Eng()
        self.assertEqual(self.one(eng, "/tokenize", {"model": "m", "prompt": "ab cde f", "add_special_tokens": False}),
                         (200, {"count": 3, "max_model_len": 4096, "tokens": [2, 3, 1], "token_strs": None}))
        self.assertEqual(self.one(eng, "/tokenize", {"prompt": "ab"})[1]["tokens"], [0, 2])
        self.assertEqual(self.one(eng, "/tokenize", {"messages": [{"role": "user", "content": "hi"}]})[0], 200)
        for payload in ({"model": "other", "prompt": "x"}, {"prompt": 5}, {}, {"prompt": "x", "add_special_tokens": "no"}):
            self.assertEqual(self.one(eng, "/tokenize", payload)[0], 400, payload)
        self.assertEqual(self.one(FakeEngine("x"), "/tokenize", {"prompt": "x"})[0], 501)

    def test_token_id_prompt_reaches_the_engine_untouched(self):
        for prompt in ([5, 6, 7, 8], [[5, 6, 7, 8]]):
            eng = self.Eng()
            st, body = self.one(eng, "/v1/completions", {"model": "m", "prompt": prompt, "max_tokens": 8, "ignore_eos": True,
                                                         "temperature": 0.0, "add_special_tokens": False, "cache_salt": "s"})
            self.assertEqual(st, 200)
            kw = eng.calls[0][1]
            self.assertEqual((kw["prompt_ids"], kw["ignore_eos"], kw["max_tokens"]), ([5, 6, 7, 8], True, 8))
            self.assertEqual(body["usage"]["prompt_tokens"], 4)
            self.assertEqual(body["usage"]["prompt_tokens_details"], {"cached_tokens": 4})   # engine said 99: clamped
            self.assertEqual((body["choices"][0]["text"], body["choices"][0]["finish_reason"]), ("abc", "length"))

    def test_loop_guard_opt_out_reaches_the_engine(self):
        # per-request loop_guard=false (as on the Qwen server): passed to generate(); absent when the request says nothing
        eng = self.Eng()
        for path, payload in (("/v1/completions", {"model": "m", "prompt": [1, 2], "max_tokens": 4}),
                              ("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})):
            self.assertEqual(self.one(eng, path, payload)[0], 200, path)
            self.assertNotIn("loop_guard", eng.calls[-1][1])
            self.assertEqual(self.one(eng, path, payload | {"loop_guard": False})[0], 200, path)
            self.assertIs(eng.calls[-1][1]["loop_guard"], False)
            self.assertEqual(self.one(eng, path, payload | {"loop_guard": "no"})[0], 400, path)

    def test_stream_ends_with_usage_and_done(self):
        eng = self.Eng()
        st, raw = self.one(eng, "/v1/completions", {"model": "m", "prompt": [1, 2, 3], "max_tokens": 8, "stream": True,
                                                    "stream_options": {"include_usage": True}})
        events = [json.loads(x[6:]) for x in raw.split("\n\n") if x.startswith("data: ") and "[DONE]" not in x]
        self.assertEqual(st, 200)
        self.assertTrue(raw.rstrip().endswith("data: [DONE]"))
        self.assertEqual("".join(e["choices"][0]["text"] for e in events), "abc")
        self.assertGreater(len([e for e in events if e["choices"][0]["text"]]), 1)    # streamed, not buffered
        self.assertEqual(events[-1]["usage"]["prompt_tokens"], 3)
        self.assertIn("cached_tokens", events[-1]["usage"]["prompt_tokens_details"])

    def test_stop_string_cuts_text_and_stream(self):
        eng = self.Eng("one two STOP three")
        st, body = self.one(eng, "/v1/completions", {"model": "m", "prompt": "go", "stop": ["STOP"], "max_tokens": 50})
        self.assertEqual((st, body["choices"][0]["text"], body["choices"][0]["finish_reason"]), (200, "one two ", "stop"))
        st, raw = self.one(eng, "/v1/completions", {"model": "m", "prompt": "go", "stop": "STOP", "max_tokens": 50, "stream": True})
        events = [json.loads(x[6:]) for x in raw.split("\n\n") if x.startswith("data: ") and "[DONE]" not in x]
        self.assertEqual("".join(e["choices"][0]["text"] for e in events), "one two ")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")

    def test_validation(self):
        eng = self.Eng()
        for bad in ([-1], [1000], [1, True], [1, 2.5], [1, "a"], [[1], [2]], [], [[]]):
            self.assertEqual(self.one(eng, "/v1/completions", {"model": "m", "prompt": bad})[0], 400, bad)
        for extra in ({"ignore_eos": "yes"}, {"max_tokens": 0}, {"n": 2}, {"stream": "x"}, {"cache_salt": ""}, {"cache_salt": 3},
                      {"model": "other"}):
            self.assertEqual(self.one(eng, "/v1/completions", {"model": "m", "prompt": [1]} | extra)[0], 400, extra)
        self.assertEqual(self.one(eng, "/v1/completions", {"model": "m", "prompt": [1] * 5000})[0], 400)
        self.assertEqual(eng.calls, [])
        self.assertEqual(self.one(FakeEngine("x"), "/v1/completions", {"model": "m", "prompt": "hi", "cache_salt": "a"})[0], 400)

    def test_cache_salt_resets_only_when_the_namespace_changes(self):
        for sessions in (1, 2):
            eng = self.Eng(sessions=sessions)
            reqs = [("/v1/completions", {"model": "m", "prompt": [1, 2], "max_tokens": 2} | ({} if s is None else {"cache_salt": s}))
                    for s in (None, None, "a", "a", "a", "b", "a", None, None)]
            self.assertEqual([c for c, _ in self.call(eng, reqs)], [200] * 9)
            self.assertEqual(len(eng.resets), 4, sessions)          # None -> a, a -> b, b -> a, a -> None

    def test_requests_without_a_salt_never_reset_and_chat_takes_a_salt(self):
        eng = self.Eng()
        chat = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        self.call(eng, [("/v1/completions", {"model": "m", "prompt": [1]}), ("/v1/chat/completions", chat)])
        self.assertEqual(eng.resets, [])
        res = self.call(eng, [("/v1/chat/completions", chat | {"cache_salt": s}) for s in ("x", "x", "y")])
        self.assertEqual([c for c, _ in res], [200] * 3)
        self.assertEqual(len(eng.resets), 2)

    def test_chat_usage_reports_cached_tokens(self):
        eng = self.Eng("hello")
        st, body = self.one(eng, "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "one two three"}]})
        usage = body["usage"]
        self.assertEqual(usage["prompt_tokens_details"], {"cached_tokens": min(99, usage["prompt_tokens"])})

    def test_salt_change_waits_for_the_running_requests(self):
        eng = self.Eng(sessions=3, delay=0.05)
        from aiohttp.test_utils import TestClient, TestServer

        async def go():
            client = TestClient(TestServer(serve.create_app(eng, "m", self.template)))
            await client.start_server()
            try:
                def req(salt):
                    return client.post("/v1/completions", json={"model": "m", "prompt": [1, 2], "cache_salt": salt})
                return [r.status for r in await asyncio.gather(req("a"), req("a"), req("b"), req("b"), req("a"))]
            finally:
                await client.close()
        self.assertEqual(run(go()), [200] * 5)
        self.assertGreaterEqual(len(eng.resets), 2)
        self.assertEqual(eng.resets, [0] * len(eng.resets))

    def test_chat_template_kwargs_reach_the_template(self):
        kw = serve.template_kwargs({"chat_template_kwargs": {"reasoning_effort": "low"}, "reasoning_effort": "high"})
        self.assertEqual(kw["reasoning_effort"], "low")
        self.assertEqual(serve.template_kwargs({"reasoning_effort": "high"})["reasoning_effort"], "high")
        self.assertEqual(serve.template_kwargs({"chat_template_kwargs": {"clear_thinking": False}})["clear_thinking"], False)
        with self.assertRaises(Exception):
            serve.template_kwargs({"chat_template_kwargs": "x"})


if __name__ == "__main__":
    unittest.main()
