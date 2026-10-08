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


class FakeGenerator:
    num_draft_tokens = 2  # /metrics derives the round count from the engine's window


class FakeEngine:
    def __init__(self, output: str):
        self.output = output
        self.prompts = []
        self.greedy_generator = FakeGenerator()

    def count_tokens(self, text: str) -> int:
        return len(text.split())

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


if __name__ == "__main__":
    unittest.main()
