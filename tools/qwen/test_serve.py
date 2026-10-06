#!/usr/bin/env python3
"""qserve unit tests: model-free, fake engine (no GPU, no exllamav3 import)."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import serve

PACK = Path(os.environ.get("QWEN_PACK", os.path.expanduser("~/models/qwen38-yamz-v1")))
THINK_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                  "{% if tools %}TOOLS={{ tools|length }}{% endif %}"
                  "{% if add_generation_prompt %}<|im_start|>assistant\n"
                  "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n"
                  "{% else %}<think>\n{% endif %}{% endif %}"
                  "{% if reasoning_effort is defined %}EFFORT={{ reasoning_effort }}{% endif %}")
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                     "opts": {"type": "object"}}}}}]
CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>\n391\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
        "<parameter=opts>\n{\"a\": [1, 2]}\n</parameter>\n</function>\n</tool_call>")


class FakeEngine:
    ctx = 4096
    supports_vision = True
    spec_on = True

    def __init__(self, output: str, stats: dict | None = None):
        self.output, self.calls, self.last_stats = output, [], {}
        self.stats = stats or {"new_tokens": 7, "prompt_tokens": 11, "cached_tokens": 5, "eos_reason": "stop_token",
                               "time_prefill": 0.5, "time_generate": 1.0, "token_ids": [1, 2, 3]}

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def spec_stats(self):
        return {"speculative": True}

    async def generate(self, prompt, **kw):
        self.calls.append((prompt, kw))
        self.last_stats = dict(self.stats)
        for i in range(0, len(self.output), 3):  # small chunks cut tags in half
            yield self.output[i:i + 3]


def run(coro):
    return asyncio.run(coro)


def with_client(engine, fn, template=THINK_TEMPLATE, defaults=None):
    from aiohttp.test_utils import TestClient, TestServer

    async def go():
        client = TestClient(TestServer(serve.create_app(engine, "m", template, defaults or serve.load_defaults("/x"))))
        await client.start_server()
        try:
            return await fn(client)
        finally:
            await client.close()
    return run(go())


def sse_events(raw: str):
    return [json.loads(line[6:]) for line in raw.split("\n\n") if line.startswith("data: ") and "[DONE]" not in line]


class ParserTests(unittest.TestCase):
    def test_reasoning_first_split(self):
        m = serve.split_completion("let me think\nmore\n</think>\n\nThe answer.\n", True, None)
        self.assertEqual(m["reasoning_content"], "let me think\nmore")
        self.assertEqual(m["content"], "The answer.")

    def test_no_reasoning_when_thinking_off(self):
        m = serve.split_completion("Plain answer </think> stays", False, None)
        self.assertNotIn("reasoning_content", m)
        self.assertEqual(m["content"], "Plain answer </think> stays")

    def test_cut_during_thought_is_all_reasoning(self):
        m = serve.split_completion("still thinking", True, None)
        self.assertEqual((m["content"], m["reasoning_content"]), ("", "still thinking"))

    def test_stream_chunking_does_not_change_the_split(self):
        text = "ab\n</think>\n\nvisible text\n\n" + CALL
        whole = serve.split_completion(text, True, TOOLS)
        for size in (1, 2, 3, 5, 8):
            sp = serve.Splitter(True)
            pieces = []
            for i in range(0, len(text), size):
                pieces += sp.feed(text[i:i + size])
            pieces += sp.finish()
            self.assertEqual("".join(t for k, t in pieces if k == "content"), whole["content"], size)
            self.assertEqual("".join(t for k, t in pieces if k == "reasoning"), whole["reasoning_content"], size)
            self.assertEqual(serve.parse_tool_calls(sp.tool_text, TOOLS)[0]["function"]["arguments"],
                             whole["tool_calls"][0]["function"]["arguments"])

    def test_tool_call_typed_arguments(self):
        calls = serve.parse_tool_calls(CALL, TOOLS)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        # schema says string: "391" stays a string; integer and object are decoded
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "391", "days": 3, "opts": {"a": [1, 2]}})

    def test_tool_call_json_body_and_truncated_call(self):
        c = serve.parse_tool_calls('<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>', None)
        self.assertEqual((c[0]["function"]["name"], json.loads(c[0]["function"]["arguments"])), ("f", {"x": 1}))
        self.assertEqual(serve.parse_tool_calls("<tool_call>\n<function=f>\n<parameter=a>\n1", None), [])

    def test_two_tool_calls(self):
        m = serve.split_completion("</think>ok\n" + CALL + "\n" + CALL, True, TOOLS)
        self.assertEqual(len(m["tool_calls"]), 2)
        self.assertEqual(m["content"], "ok")


class RequestTests(unittest.TestCase):
    def test_template_kwargs_precedence(self):
        d = {"enable_thinking": None, "reasoning_effort": "low"}
        self.assertEqual(serve.resolve_template_kwargs({}, d), {"reasoning_effort": "low"})
        self.assertEqual(serve.resolve_template_kwargs({"reasoning_effort": "high"}, d), {"reasoning_effort": "xhigh"})
        self.assertEqual(serve.resolve_template_kwargs({"chat_template_kwargs": {"enable_thinking": False}}, d),
                         {"enable_thinking": False, "reasoning_effort": "low"})
        self.assertEqual(serve.resolve_template_kwargs({"reasoning_effort": "none"}, d), {"enable_thinking": False})
        with self.assertRaises(serve.BadRequest):
            serve.resolve_template_kwargs({"enable_thinking": "no"}, d)

    def test_sampling_defaults_come_from_generation_config(self):
        d = serve.load_defaults(str(PACK)) if PACK.is_dir() else serve.load_defaults("/x")
        self.assertEqual((d["temperature"], d["top_p"], d["top_k"]), (1.0, 0.95, 20))
        s = serve.sampling_from({"temperature": None, "top_k": 5}, d)
        self.assertEqual((s["temperature"], s["top_p"], s["top_k"]), (1.0, 0.95, 5))
        self.assertEqual(serve.sampling_from({"temperature": 0}, d)["temperature"], 0)
        with self.assertRaises(serve.BadRequest):
            serve.sampling_from({"top_p": 0}, d)

    def test_cli_flags_override_defaults(self):
        class A:
            default_temperature, default_top_p, default_top_k, default_min_p = 0.6, None, None, None
            default_reasoning_effort, no_thinking = "low", True
        d = serve.load_defaults(str(PACK) if PACK.is_dir() else "/x", A)
        self.assertEqual((d["temperature"], d["reasoning_effort"], d["enable_thinking"]), (0.6, "low", False))

    def test_extract_images_in_order(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "u1"}}]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u2"}}]}]
        self.assertEqual(serve.extract_images(msgs), ["u1", "u2"])

    def test_load_image_data_uri_and_rejects(self):
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (4, 4), (1, 2, 3)).save(buf, "PNG")
        img = serve.load_image("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
        self.assertEqual(img.size, (4, 4))
        for bad in ("file:///etc/passwd", "data:image/png;base64,AAAA", ""):
            with self.assertRaises(serve.BadRequest):
                serve.load_image(bad)

    @unittest.skipUnless((PACK / "chat_template.jinja").is_file(), "no pack")
    def test_template_matches_hf_apply_chat_template(self):
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(str(PACK))
        tpl = (PACK / "chat_template.jinja").read_text()
        history = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris?"},
                   {"role": "assistant", "content": "", "reasoning_content": "need the tool",
                    "tool_calls": [{"id": "c1", "type": "function",
                                    "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris", "days": 2})}}]},
                   {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                   {"role": "user", "content": "thanks, and a joke?"}]
        for msgs, tools, kw in (
                ([{"role": "user", "content": "hi"}], None, {}),
                ([{"role": "user", "content": "hi"}], None, {"enable_thinking": False}),
                ([{"role": "user", "content": "hi"}], None, {"reasoning_effort": "low"}),
                ([{"role": "user", "content": "hi"}], TOOLS, {"enable_thinking": False}),
                (history, TOOLS, {}),
                ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "what?"}]}],
                 None, {"enable_thinking": False})):
            ref_msgs = [dict(m) for m in msgs]
            for m in ref_msgs:  # HF wants dict arguments
                for c in m.get("tool_calls", []):
                    c["function"] = dict(c["function"], arguments=json.loads(c["function"]["arguments"]))
            expect = hf.apply_chat_template(ref_msgs, tools=tools, tokenize=False, add_generation_prompt=True, **kw)
            self.assertEqual(serve.render_prompt(tpl, serve.normalize_messages(msgs), tools, **kw), expect)


class HttpTests(unittest.TestCase):
    def chat(self, engine, payload, stream=False):
        async def fn(client):
            r = await client.post("/v1/chat/completions", json={"stream": stream, "model": "m"} | payload)
            return r.status, (await r.text() if stream else await r.json())
        return with_client(engine, fn)

    def test_models_and_health(self):
        async def fn(client):
            return await (await client.get("/v1/models")).json(), await (await client.get("/health")).json()
        models, health = with_client(FakeEngine(""), fn)
        self.assertEqual(models["data"][0]["id"], "m")
        self.assertEqual((health["status"], health["speculative"], health["vision"]), ("ok", True, True))

    def test_non_stream_reasoning_split_usage_timings(self):
        st, body = self.chat(FakeEngine("why\n</think>\n\nhello"), {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200)
        msg = body["choices"][0]["message"]
        self.assertEqual((msg["content"], msg["reasoning_content"]), ("hello", "why"))
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"], {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
                                         "prompt_tokens_details": {"cached_tokens": 5}})
        t = body["timings"]
        self.assertEqual((t["cache_n"], t["prompt_n"], t["prompt_ms"], t["predicted_per_second"]), (5, 6, 500.0, 7.0))

    def test_finish_reason_length_when_cut(self):
        eng = FakeEngine("half a thou", {"new_tokens": 4, "eos_reason": "max_new_tokens"})
        st, body = self.chat(eng, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(body["choices"][0]["message"]["content"], "")
        self.assertEqual(body["choices"][0]["message"]["reasoning_content"], "half a thou")
        self.assertEqual(eng.calls[0][1]["max_tokens"], 4)

    def test_thinking_off_has_no_reasoning(self):
        eng = FakeEngine("plain answer")
        st, body = self.chat(eng, {"messages": [{"role": "user", "content": "hi"}], "enable_thinking": False})
        self.assertEqual(body["choices"][0]["message"], {"role": "assistant", "content": "plain answer"})
        self.assertIn("<think>\n\n</think>", eng.calls[0][0])
        st, body = self.chat(eng, {"messages": [{"role": "user", "content": "hi"}],
                                   "chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low"})
        self.assertTrue(eng.calls[1][0].endswith("<think>\nEFFORT=low"))

    def test_sampling_defaults_and_overrides_reach_the_engine(self):
        eng = FakeEngine("x")
        self.chat(eng, {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(eng.calls[0][1]["sampling"], {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0})
        self.chat(eng, {"messages": [{"role": "user", "content": "hi"}], "temperature": 0, "top_p": None})
        self.assertEqual(eng.calls[1][1]["sampling"]["temperature"], 0)
        self.assertEqual(eng.calls[1][1]["sampling"]["top_p"], 0.95)

    def test_tool_call_round_trip(self):
        eng = FakeEngine("</think>\n\n" + CALL)
        st, body = self.chat(eng, {"messages": [{"role": "user", "content": "weather?"}], "tools": TOOLS})
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(json.loads(call["function"]["arguments"])["days"], 3)
        history = [{"role": "user", "content": "weather?"}, choice["message"],
                   {"role": "tool", "tool_call_id": call["id"], "content": "sunny"}]
        st, body = self.chat(eng, {"messages": history, "tools": TOOLS})
        self.assertEqual(st, 200)
        self.assertIn("<|im_start|>tool\nsunny", eng.calls[1][0])

    def test_stream_matches_non_stream(self):
        text = "reason\n</think>\n\nvisible words\n\n" + CALL
        eng = FakeEngine(text)
        payload = {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS}
        _, whole = self.chat(eng, payload)
        st, raw = self.chat(eng, payload, stream=True)
        self.assertTrue(raw.rstrip().endswith("data: [DONE]"))
        ev = sse_events(raw)
        self.assertEqual(ev[0]["choices"][0]["delta"]["role"], "assistant")
        reasoning = "".join(e["choices"][0]["delta"].get("reasoning_content", "") for e in ev)
        content = "".join(e["choices"][0]["delta"].get("content", "") for e in ev)
        msg = whole["choices"][0]["message"]
        self.assertEqual((reasoning, content), (msg["reasoning_content"], msg["content"]))
        calls = [c for e in ev for c in e["choices"][0]["delta"].get("tool_calls", [])]
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(ev[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(ev[-1]["usage"]["completion_tokens"], 7)
        self.assertIn("timings", ev[-1])

    def test_parallel_tool_calls_one_stream_message_each(self):
        eng = FakeEngine("</think>\n\n" + CALL + "\n" + CALL + "\n" + CALL)
        st, raw = self.chat(eng, {"messages": [{"role": "user", "content": "hi"}], "tools": TOOLS}, stream=True)
        msgs = [e["choices"][0]["delta"]["tool_calls"] for e in sse_events(raw) if e["choices"][0]["delta"].get("tool_calls")]
        self.assertEqual([len(m) for m in msgs], [1, 1, 1])
        self.assertEqual([m[0]["index"] for m in msgs], [0, 1, 2])
        for m in msgs:
            json.loads(m[0]["function"]["arguments"])

    def test_stream_length(self):
        eng = FakeEngine("abc", {"new_tokens": 2, "eos_reason": "max_new_tokens"})
        _, raw = self.chat(eng, {"messages": [{"role": "user", "content": "hi"}]}, stream=True)
        self.assertEqual(sse_events(raw)[-1]["choices"][0]["finish_reason"], "length")

    def test_return_token_ids_and_cache_flags(self):
        eng = FakeEngine("x")
        st, body = self.chat(eng, {"messages": [{"role": "user", "content": "hi"}], "return_token_ids": True,
                                   "speculative": False, "cache_prompt": False})
        self.assertEqual(body["token_ids"], [1, 2, 3])
        kw = eng.calls[0][1]
        self.assertEqual((kw["speculative"], kw["cache_prompt"]), (False, False))

    def test_bad_requests_are_400_json(self):
        eng = FakeEngine("x")
        good = [{"role": "user", "content": "hi"}]
        for payload in ({"messages": []}, {"messages": "x"}, {"messages": good, "stream": "yes"},
                        {"messages": good, "model": "other"}, {"messages": good, "max_tokens": 0},
                        {"messages": good, "temperature": "hot"}, {"messages": good, "stop": [1]},
                        {"messages": good, "tools": "x"}, {"messages": [{"content": "no role"}]},
                        {"messages": good, "chat_template_kwargs": 3}):
            st, body = self.chat(eng, payload)
            self.assertEqual(st, 400, payload)
            self.assertIn("message", body["error"])
        self.assertEqual(eng.calls, [])

    def test_prompt_longer_than_context_is_rejected(self):
        st, body = self.chat(FakeEngine("x"), {"messages": [{"role": "user", "content": "w " * 5000}]})
        self.assertEqual(st, 400)
        self.assertIn("context", body["error"]["message"])

    def test_max_tokens_clipped_to_context(self):
        eng = FakeEngine("x")
        self.chat(eng, {"messages": [{"role": "user", "content": "w " * 4000}], "max_tokens": 9999})
        self.assertLessEqual(eng.calls[0][1]["max_tokens"], 4096 - 4000)

    def test_image_without_vision_and_placeholder_mismatch(self):
        eng = FakeEngine("x")
        part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        msgs = [{"role": "user", "content": [part, {"type": "text", "text": "what"}]}]
        st, body = self.chat(eng, {"messages": msgs})  # fake template has no image placeholder
        self.assertEqual(st, 400)
        eng.supports_vision = False
        st, body = self.chat(eng, {"messages": msgs})
        self.assertIn("vision", body["error"]["message"])

    def test_images_reach_the_engine_in_order(self):
        tpl = THINK_TEMPLATE.replace("{{ m.content }}", "{{ m.content if m.content is string else '<|image_pad|>' }}")
        eng = FakeEngine("ok")

        async def fn(client):
            msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:u1"}},
                                                 {"type": "text", "text": "what"}]}]
            r = await client.post("/v1/chat/completions", json={"model": "m", "messages": msgs})
            return r.status
        self.assertEqual(with_client(eng, fn, template=tpl), 200)
        self.assertEqual(eng.calls[0][1]["images"], ["data:u1"])

    def test_apply_template(self):
        async def fn(client):
            r = await client.post("/apply-template", json={"messages": [{"role": "user", "content": "hi"}],
                                                          "enable_thinking": False})
            return await r.json()
        self.assertTrue(with_client(FakeEngine(""), fn)["prompt"].endswith("<think>\n\n</think>\n\n"))

    def test_invalid_json_is_400(self):
        async def fn(client):
            r = await client.post("/v1/chat/completions", data="{nope", headers={"Content-Type": "application/json"})
            return r.status
        self.assertEqual(with_client(FakeEngine(""), fn), 400)

    def test_requests_are_serialized(self):
        class Slow(FakeEngine):
            active = peak = 0

            async def generate(self, prompt, **kw):
                Slow.active += 1
                Slow.peak = max(Slow.peak, Slow.active)
                await asyncio.sleep(0.05)
                yield "x"
                Slow.active -= 1

        async def fn(client):
            body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            rs = await asyncio.gather(*[client.post("/v1/chat/completions", json=body) for _ in range(4)])
            return [r.status for r in rs]
        self.assertEqual(with_client(Slow("x"), fn), [200] * 4)
        self.assertEqual(Slow.peak, 1)

    def test_client_disconnect_cancels_the_job(self):
        class Endless(FakeEngine):
            stopped = False

            async def generate(self, prompt, cancel=None, **kw):
                for _ in range(2000):
                    if cancel is not None and cancel.is_set():
                        Endless.stopped = True
                        return
                    await asyncio.sleep(0.005)
                    yield "word "

        async def fn(client):
            body = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
            r = await client.post("/v1/chat/completions", json=body)
            await r.content.readline()
            r.close()
            for _ in range(100):
                await asyncio.sleep(0.05)
                if Endless.stopped:
                    break
            r2 = await client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
            return Endless.stopped
        self.assertTrue(with_client(Endless(""), fn))

    def test_completion_verbatim_prompt_and_slots(self):
        class Store:
            busy = False

            def slots(self):
                return [{"id": 0}]

            def erase(self):
                return {"id_slot": 0, "n_erased": 3}

            def save(self, name):
                return {"saved": name}

        eng = FakeEngine("raw text STOP tail", {"new_tokens": 3, "prompt_tokens": 9, "cached_tokens": 4, "eos_reason": "stop_token"})
        eng.slot_store = Store()

        async def fn(client):
            r = await (await client.post("/completion", json={"prompt": "<|im_start|>hi", "n_predict": 5, "stop": "STOP"})).json()
            e = await (await client.post("/slots/0?action=erase")).json()
            sv = await (await client.post("/slots/0?action=save", json={"filename": "a.bin"})).json()
            bad = await client.post("/slots/1?action=erase")
            bad2 = await client.post("/slots/0?action=nope")
            empty = await client.post("/completion", json={})
            return r, e, sv, bad.status, bad2.status, empty.status, await (await client.get("/slots")).json()
        r, e, sv, b1, b2, b3, sl = with_client(eng, fn)
        self.assertEqual((r["content"], r["tokens_cached"], r["stopped_word"], r["timings"]["prompt_n"]), ("raw text ", 4, "STOP", 5))
        self.assertEqual(eng.calls[0][0], "<|im_start|>hi")
        self.assertEqual((e["n_erased"], sv["saved"], b1, b2, b3, sl), (3, "a.bin", 400, 400, 400, [{"id": 0}]))

    def test_engine_error_is_500_json(self):
        class Boom(FakeEngine):
            async def generate(self, prompt, **kw):
                raise RuntimeError("kaput")
                yield ""
        st, body = self.chat(Boom(""), {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 500)
        self.assertIn("kaput", body["error"]["message"])


    def test_stream_engine_error_still_ends_with_finish_reason(self):
        class Boom(FakeEngine):
            async def generate(self, prompt, **kw):
                yield "abc "
                raise RuntimeError("kaput")
        _, raw = self.chat(Boom(""), {"messages": [{"role": "user", "content": "hi"}]}, stream=True)
        self.assertTrue(raw.endswith("data: [DONE]\n\n"))
        ev = sse_events(raw)
        self.assertIn("kaput", [e for e in ev if "error" in e][0]["error"]["message"])
        self.assertEqual(ev[-1]["choices"][0]["finish_reason"], "error")


class ReplyRoomTests(unittest.TestCase):
    """The reply budget must never make Job.prepare ask for more pages than the page table holds."""

    @staticmethod
    def pages_needed(x, new_tokens, draft):
        """Job.prepare for one sequence without an explicit requeue budget: max_rq = new + 1 + draft; pagetable.Sequence.prepare."""
        max_len = x + new_tokens + 1 + draft
        return (max_len + 255) // 256

    def test_room_is_exactly_the_largest_that_fits(self):
        for ctx in (4096, 4097, 4351, 131072, 272384, 98305):
            pages = ctx // 256                       # PageTable.max_pages = cache.max_num_tokens // PAGE_SIZE
            for draft in (0, 1, 3, 4):
                for x in (1, 255, 256, 257, 1000, ctx - 300, ctx - 5, ctx - 4):
                    room = serve.reply_room(ctx, x, draft)
                    if room < 1:
                        self.assertGreater(self.pages_needed(x, 1, draft), pages, (ctx, draft, x))
                        continue
                    self.assertLessEqual(self.pages_needed(x, room, draft), pages, (ctx, draft, x))      # fits
                    self.assertGreater(self.pages_needed(x, room + 1, draft), pages, (ctx, draft, x))     # one more does not

    def test_old_rule_overflowed_at_the_page_boundary(self):
        ctx, x, draft = 131072, 1000, 3
        old = ctx - x - 1
        self.assertGreater(self.pages_needed(x, old, draft), ctx // 256)
        self.assertEqual(serve.reply_room(ctx, x, draft), old - draft)
        self.assertEqual(serve.reply_room(ctx, x, 0), old)           # no speculation: nothing lost

    def test_draft_window_only_when_speculating(self):
        class E:
            ndt, draft_model = 3, object()
        self.assertEqual(serve.draft_window(E()), 3)
        self.assertEqual(serve.draft_window(E(), False), 0)
        self.assertEqual(serve.draft_window(FakeEngine("x")), 0)

    def test_http_clip_and_400(self):
        class E(FakeEngine):
            ctx, ndt, draft_model = 4096, 3, object()
        eng = E("x")
        words = lambda n: "w " * n

        async def fn(client):
            out = []
            for n, mt in ((100, 100000), (4096 - 4 - 3, 10), (4096, 10)):
                r = await client.post("/v1/completions", json={"model": "m", "prompt": words(n), "max_tokens": mt})
                out.append((r.status, await r.json()))
            return out
        res = with_client(eng, fn)
        self.assertEqual(res[0][0], 200)
        self.assertEqual(eng.calls[0][1]["max_tokens"], 4096 - eng.count_tokens(words(100)) - 1 - 3)
        self.assertEqual(res[2][0], 400)
        self.assertEqual(res[2][1]["error"]["type"], "invalid_request_error")
        self.assertIn("context length exceeded", res[2][1]["error"]["message"])

    def test_completions_honours_speculative_false(self):
        eng = FakeEngine("x")

        async def fn(client):
            await client.post("/v1/completions", json={"model": "m", "prompt": "hi", "speculative": False})
            await client.post("/v1/completions", json={"model": "m", "prompt": "hi"})
        with_client(eng, fn)
        self.assertEqual([c[1]["speculative"] for c in eng.calls], [False, True])


class MiscTests(unittest.TestCase):
    def test_sse_framing_and_stop_text(self):
        self.assertEqual(serve.sse({"a": 1}), b'data: {"a":1}\n\n')
        self.assertEqual(serve.stop_text("hello STOP x", ["STOP"]), ("hello ", "STOP"))

    def test_config_max_position(self):
        class C:
            config_dict = {"text_config": {"max_position_embeddings": 262144}}
        self.assertEqual(serve.config_max_position(C), 262144)

    def test_partial_tag_hold(self):
        self.assertEqual(serve._partial_tag("abc</thi", ["</think>"]), 5)
        self.assertEqual(serve._partial_tag("abc", ["</think>"]), 0)


class ServeEnvDefaultsTest(unittest.TestCase):
    def test_prefill_set_is_default_and_overridable(self):
        d = dict(serve.SERVE_ENV)
        for k in ("EXL3_PLE_HIP", "EXL3_DQ_HIP", "EXL3_GR_HIP", "EXL3_GDN_FUSE", "EXL3_PF_SKIP"):
            self.assertEqual(d[k], "1")
        self.assertEqual(d["EXL3_PF_DEFER"], "1")
        self.assertEqual(d["EXL3_PREFILL_CHUNK"], "4096")
        self.assertEqual(len(d), len(serve.SERVE_ENV))          # no key twice
        saved = os.environ.get("EXL3_GDN_FUSE"); os.environ["EXL3_GDN_FUSE"] = "0"
        try:
            for k, v in serve.SERVE_ENV:
                os.environ.setdefault(k, v)
            self.assertEqual(os.environ["EXL3_GDN_FUSE"], "0")   # the caller's value wins
        finally:
            if saved is None: os.environ.pop("EXL3_GDN_FUSE", None)
            else: os.environ["EXL3_GDN_FUSE"] = saved


if __name__ == "__main__":
    unittest.main()
