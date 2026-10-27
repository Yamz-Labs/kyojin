#!/usr/bin/env python3
"""qserve unit tests: model-free, fake engine (no GPU, no exllamav3 import)."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import types
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import serve



def _write_draft2s_pack(path, vocab, width):
    """A header-only draft2s pack (tools/qwen/draft2s_pack.py metadata layout) for the sidecar tests."""
    import json as _json, struct
    hdr = _json.dumps({"__metadata__": {"format": "draft2s-v1", "vocab": str(vocab), "in_features": str(width), "m": "512"}}).encode()
    path.write_bytes(struct.pack("<Q", len(hdr)) + hdr)


class SharedModuleTests(unittest.TestCase):
    def test_serve_metrics_resolves_next_to_the_server(self):
        # serve.py loads tools/mimo/serve.py, which imports serve_metrics from the sys.path of this folder
        import serve_metrics
        self.assertTrue(hasattr(serve_metrics, "Metrics"))

PACK = Path(os.environ.get("QWEN_PACK", os.path.expanduser("~/models/qwen38-yamz-v1")))
THINK_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                  "{% if tools %}TOOLS={{ tools|length }}{% endif %}"
                  "{% if add_generation_prompt %}<|im_start|>assistant\n"
                  "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n"
                  "{% else %}<think>\n{% endif %}{% endif %}"
                  "{% if reasoning_effort is defined %}EFFORT={{ reasoning_effort }}{% endif %}")
# THINK_TEMPLATE prints the effort after the generation prompt: the HTTP tests run without a default effort.
NO_EFFORT_DEFAULTS = dict(serve.load_defaults("/x"), reasoning_effort=None)
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

    def prompt_ids(self, prompt: str):
        """The ids the routes count prompt tokens with (the real engines read them from an encode cache).

        Placeholders of count_tokens() length: the routes only ask for numel()."""
        import torch
        return torch.zeros(self.count_tokens(prompt), dtype=torch.long)

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
        client = TestClient(TestServer(serve.create_app(engine, "m", template, defaults or NO_EFFORT_DEFAULTS)))
        await client.start_server()
        try:
            return await fn(client)
        finally:
            await client.close()
    return run(go())


def sse_events(raw: str):
    return [json.loads(line[6:]) for line in raw.split("\n\n") if line.startswith("data: ") and "[DONE]" not in line]


class ParserTests(unittest.TestCase):
    def test_draft2s_sidecar(self):
        import tempfile
        from unittest import mock
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {}):
            pack = Path(d)
            (pack / "config.json").write_text(json.dumps({"vocab_size": 248320, "hidden_size": 2560}))
            os.environ.pop("EXL3_DRAFT2S", None)
            self.assertIn("no draft2s.st", serve.configure_draft2s(str(pack)))
            self.assertNotIn("EXL3_DRAFT2S", os.environ)
            _write_draft2s_pack(pack / serve.DRAFT2S_IN_PACK, 154880, 4096)       # GLM's head: not this model
            self.assertIn("does not match", serve.configure_draft2s(str(pack)))
            self.assertNotIn("EXL3_DRAFT2S", os.environ)
            _write_draft2s_pack(pack / serve.DRAFT2S_IN_PACK, 248320, 2560)
            self.assertIn("two-stage", serve.configure_draft2s(str(pack)))
            self.assertEqual(os.environ["EXL3_DRAFT2S"], str(pack / "draft2s.st"))
            os.environ["EXL3_DRAFT2S"] = "/nope/x.st"
            self.assertIn("/nope/x.st", serve.configure_draft2s(str(pack)))
            self.assertEqual(os.environ["EXL3_DRAFT2S"], "/nope/x.st")
            for value in ("0", "off", ""):
                os.environ["EXL3_DRAFT2S"] = value
                self.assertIn("full head", serve.configure_draft2s(str(pack)))
                self.assertEqual(os.environ["EXL3_DRAFT2S"], "")
        self.assertNotIn("EXL3_DRAFT2S", dict(serve.SERVE_ENV))
        self.assertFalse(serve.DRAFT2S_IN_PACK.endswith(".safetensors"))

    def test_quoted_tool_call_tag_is_text(self):
        # reasoning and answers that mention the tag: no call, nothing swallowed (streamed in 3-character pieces too)
        text = "the Splitter (handles `<tool_call>` blocks) matters\n</think>\n\nUse <tool_call> then </tool_call> like this."
        for step in (len(text), 3, 1):
            sp = serve.Splitter(True)
            pieces = []
            for i in range(0, len(text), step):
                pieces += sp.feed(text[i:i + step])
            pieces += sp.finish()
            self.assertEqual("".join(t for k, t in pieces if k == "reasoning"), "the Splitter (handles `<tool_call>` blocks) matters")
            self.assertEqual("".join(t for k, t in pieces if k == "content"), "Use <tool_call> then </tool_call> like this.")
            self.assertEqual(sp.tool_text, "")

    def test_real_call_after_a_quotation(self):
        text = "see `<tool_call>` here\n</think>\n\n<tool_call>\n<function=f>\n<parameter=a>1</parameter>\n</function>\n</tool_call>"
        for step in (len(text), 2, 1):
            sp = serve.Splitter(True)
            for i in range(0, len(text), step):
                sp.feed(text[i:i + step])
            sp.finish()
            self.assertEqual([c["function"]["name"] for c in serve.parse_tool_calls(sp.tool_text, None)], ["f"])
        sp = serve.Splitter(False)
        sp.feed('<tool_call>\n{"name": "g", "arguments": {}}\n</tool_call>'); sp.finish()
        self.assertEqual([c["function"]["name"] for c in serve.parse_tool_calls(sp.tool_text, None)], ["g"])

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
        self.assertEqual(serve.resolve_template_kwargs({}, serve.load_defaults("/x")), {"reasoning_effort": "medium"})
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


class RoleTests(unittest.TestCase):
    def test_developer_role_is_the_system_role(self):
        msgs = serve.normalize_messages([{"role": "developer", "content": "be brief"}, {"role": "user", "content": "hi"}])
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])


class HttpTests(unittest.TestCase):
    def chat(self, engine, payload, stream=False, template=THINK_TEMPLATE):
        async def fn(client):
            r = await client.post("/v1/chat/completions", json={"stream": stream, "model": "m"} | payload)
            return r.status, (await r.text() if stream else await r.json())
        return with_client(engine, fn, template=template)

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

    def test_tool_text_matching_image_placeholder_is_not_an_image(self):
        eng = FakeEngine("x")
        marker = serve.IMAGE_PAD
        st, body = self.chat(eng, {"messages": [{"role": "tool", "content": "tool output: " + marker}]})
        self.assertEqual(st, 200, body)
        self.assertEqual(len(eng.calls), 1)
        self.assertEqual(eng.calls[0][1]["images"], [])

    def test_tool_call_argument_image_url_key_is_escaped(self):
        marker = serve.IMAGE_PAD
        tpl = THINK_TEMPLATE.replace("{{ m.content }}", "{{ m.tool_calls[0].function.arguments.image_url }}")
        eng = FakeEngine("x")
        messages = [{"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function", "function": {
                "name": "f", "arguments": {"image_url": marker}}}]}]
        st, body = self.chat(eng, {"messages": messages}, template=tpl)
        self.assertEqual(st, 200, body)
        self.assertEqual(eng.calls[0][1]["images"], [])
        self.assertNotIn(marker, eng.calls[0][0])

    def test_input_text_part_matching_image_placeholder_is_not_an_image(self):
        marker = serve.IMAGE_PAD
        eng = FakeEngine("x")
        st, body = self.chat(eng, {"messages": [{"role": "user", "content": [
            {"type": "input_text", "text": marker}]}]}, template=THINK_TEMPLATE.replace(
                "{{ m.content }}", "{{ m.content[0].text }}"))
        self.assertEqual(st, 200, body)
        self.assertEqual(eng.calls[0][1]["images"], [])

    def test_tool_schema_property_key_matching_image_placeholder_is_not_an_image(self):
        marker = serve.IMAGE_PAD
        tpl = THINK_TEMPLATE.replace("TOOLS={{ tools|length }}", "TOOLS={{ tools|tojson }}")
        eng = FakeEngine("x")
        tools = [{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {marker: {"type": "string"}}}}}]
        original_tools = json.loads(json.dumps(tools))
        st, body = self.chat(eng, {"messages": [{"role": "user", "content": "hi"}], "tools": tools},
                             template=tpl)
        self.assertEqual(st, 200, body)
        self.assertEqual(eng.calls[0][1]["images"], [])
        self.assertIn(marker, tools[0]["function"]["parameters"]["properties"])
        self.assertEqual(tools, original_tools)

    def test_tool_schema_description_matching_image_placeholder_is_not_an_image(self):
        marker = serve.IMAGE_PAD
        tpl = THINK_TEMPLATE.replace("TOOLS={{ tools|length }}", "TOOLS={{ tools[0].function.description }}")
        eng = FakeEngine("x")
        tools = [{"type": "function", "function": {"name": "f", "description": marker,
                  "parameters": {"type": "object", "properties": {}}}}]
        st, body = self.chat(eng, {"messages": [{"role": "user", "content": "hi"}], "tools": tools},
                             template=tpl)
        self.assertEqual(st, 200, body)
        self.assertEqual(eng.calls[0][1]["images"], [])

    def test_reasoning_matching_image_placeholder_is_not_an_image(self):
        marker = serve.IMAGE_PAD
        tpl = THINK_TEMPLATE.replace("{{ m.content }}", "{{ m.reasoning_content }}")
        eng = FakeEngine("x")
        messages = [{"role": "assistant", "content": "", "reasoning_content": marker}]
        st, body = self.chat(eng, {"messages": messages}, template=tpl)
        self.assertEqual(st, 200, body)
        self.assertEqual(eng.calls[0][1]["images"], [])

    def test_image_part_count_ignores_placeholder_in_tool_text(self):
        marker = serve.IMAGE_PAD
        tpl = THINK_TEMPLATE.replace("{{ m.content }}", "{{ m.content if m.content is string else '" + marker + "' }}")
        eng = FakeEngine("x")

        async def fn(client):
            msgs = [{"role": "tool", "content": marker},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image"}},
                                                       {"type": "input_text", "text": "note: " + marker}]}]
            r = await client.post("/v1/chat/completions", json={"model": "m", "messages": msgs})
            return r.status

        self.assertEqual(with_client(eng, fn, template=tpl), 200)
        self.assertEqual(eng.calls[0][1]["images"], ["data:image"])

    def test_image_field_only_parts_are_counted_and_passed_in_order(self):
        marker = serve.IMAGE_PAD
        tpl = THINK_TEMPLATE.replace("{{ m.content }}", "{{ m.content if m.content is string else '" + marker * 2 + "' }}")
        eng = FakeEngine("ok")
        image_urls = ["https://example.test/first.png", "data:second"]
        messages = [{"role": "user", "content": [
            {"image": image_urls[0]},
            {"type": "image_url", "image_url": {"url": image_urls[1]}},
        ]}]
        st, body = self.chat(eng, {"messages": messages}, template=tpl)
        self.assertEqual(st, 200, body)
        self.assertEqual(eng.calls[0][1]["images"], image_urls)

    def test_tool_property_sanitization_collision_is_rejected_without_mutation(self):
        marker = serve.IMAGE_PAD
        escaped_marker = marker[:1] + "\\" + marker[1:]
        tpl = THINK_TEMPLATE.replace("TOOLS={{ tools|length }}", "TOOLS={{ tools|tojson }}")
        eng = FakeEngine("x")
        tools = [{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {marker: {"type": "string"},
                                                  escaped_marker: {"type": "integer"}}}}}]
        original_tools = json.loads(json.dumps(tools))
        st, body = self.chat(eng, {"messages": [{"role": "user", "content": "hi"}], "tools": tools},
                             template=tpl)
        self.assertEqual(st, 400, body)
        self.assertIn("collision", body["error"]["message"].lower())
        self.assertEqual(eng.calls, [])
        self.assertEqual(tools, original_tools)

    def test_images_reach_the_engine_in_order(self):
        marker = serve.IMAGE_PAD
        tpl = THINK_TEMPLATE.replace("{{ m.content }}", "{{ m.content if m.content is string else '" + marker + "' }}")
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

    def test_sessions_run_together_and_reject_rebuilds(self):
        class Slow(FakeEngine):
            sessions = 2
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
            bad = [await client.post("/v1/chat/completions", json=body | flag) for flag in ({"speculative": False}, {"cache_prompt": False})]
            bad.append(await client.post("/completion", json={"prompt": "hi", "cache_prompt": False}))
            bad.append(await client.post("/v1/completions", json={"prompt": "hi", "cache_prompt": False}))
            return [r.status for r in rs], [r.status for r in bad]
        self.assertEqual(with_client(Slow("x"), fn), ([200] * 4, [400] * 4))
        self.assertEqual(Slow.peak, 2)

    def test_sessions_accept_speculative_false_without_a_drafter(self):
        class Plain(FakeEngine):
            sessions, spec_on = 2, False

        async def fn(client):
            body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            ok = await client.post("/v1/chat/completions", json=body | {"speculative": False})
            cold = await client.post("/v1/chat/completions", json=body | {"cache_prompt": False})
            return ok.status, cold.status
        self.assertEqual(with_client(Plain("x"), fn), (200, 400))

    def test_sessions_keep_their_own_stats(self):
        class PerPrompt(FakeEngine):
            sessions = 2

            async def generate(self, prompt, stats=None, **kw):
                await asyncio.sleep(0.05)
                stats.update(new_tokens=len(prompt), prompt_tokens=1, eos_reason="stop_token")
                yield "x"

        async def fn(client):
            bodies = [{"model": "m", "messages": [{"role": "user", "content": "a" * n}]} for n in (10, 200)]
            rs = await asyncio.gather(*[client.post("/v1/chat/completions", json=b) for b in bodies])
            return [(await r.json())["usage"]["completion_tokens"] for r in rs]
        a, b = with_client(PerPrompt("x"), fn)
        self.assertEqual(b - a, 190)

    def test_slot_action_waits_for_running_requests(self):
        order = []

        class Store:
            busy = False

            def save(self, name):
                order.append("save")
                return {"saved": name}

        class Slow(FakeEngine):
            sessions = 2

            async def generate(self, prompt, **kw):
                await asyncio.sleep(0.2)
                order.append("generated")
                yield "x"

        eng = Slow("x")
        eng.slot_store = Store()

        async def fn(client):
            body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            req = asyncio.ensure_future(client.post("/v1/chat/completions", json=body))
            await asyncio.sleep(0.05)
            save = await client.post("/slots/0?action=save", json={"filename": "a.bin"})
            return (await req).status, save.status
        self.assertEqual(with_client(eng, fn), (200, 200))
        self.assertEqual(order, ["generated", "save"])

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

    def test_failed_requests_are_logged_on_stderr(self):
        # the request log is off by default: an OOM on a long prompt left no server line at all (only the client saw the error event)
        import contextlib, io
        class Boom(FakeEngine):
            async def generate(self, prompt, **kw):
                raise RuntimeError("CUDA out of memory")
                yield ""
        for stream in (False, True):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.chat(Boom(""), {"messages": [{"role": "user", "content": "hi"}]}, stream=stream)
            self.assertIn("request failed", err.getvalue())
            self.assertIn("CUDA out of memory", err.getvalue())


class LogprobsTests(unittest.TestCase):
    ENTRIES = [(-0.5, [(1, -0.5), (9, -1.5)]), (-2.0, [(2, -0.25), (3, -2.0)]), None]

    def engine(self):
        eng = FakeEngine("abc", {"new_tokens": 3, "prompt_tokens": 2, "eos_reason": "stop_token", "token_ids": [1, 2, 3],
                                 "logprobs": self.ENTRIES})
        eng.token_str = lambda i: f"t{i}"
        return eng

    def post(self, path, payload):
        async def fn(client):
            r = await client.post(path, json={"model": "m"} | payload)
            return r.status, await r.json()
        return with_client(self.engine(), fn)

    def test_completions_shape_and_request(self):
        eng = self.engine()

        async def fn(client):
            r = await client.post("/v1/completions", json={"model": "m", "prompt": "hi", "max_tokens": 3, "logprobs": 2})
            return r.status, await r.json()
        st, body = with_client(eng, fn)
        self.assertEqual(st, 200)
        self.assertEqual(eng.calls[0][1]["logprobs"], 2)
        lp = body["choices"][0]["logprobs"]
        self.assertEqual(lp["tokens"], ["t1", "t2", "t3"])
        self.assertEqual(lp["token_ids"], [1, 2, 3])
        self.assertEqual(lp["token_logprobs"], [-0.5, -2.0, None])
        self.assertEqual(lp["top_logprobs"][0], {"t1": -0.5, "t9": -1.5})
        self.assertEqual(lp["top_logprobs_ids"][1], [[2, -0.25], [3, -2.0]])
        self.assertEqual(lp["text_offset"], [0, 2, 4])

    def test_chat_shape(self):
        st, body = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                      "logprobs": True, "top_logprobs": 2})
        self.assertEqual(st, 200)
        c = body["choices"][0]["logprobs"]["content"]
        self.assertEqual([x["token_id"] for x in c], [1, 2, 3])
        self.assertEqual(c[0]["logprob"], -0.5)
        self.assertEqual([t["token_id"] for t in c[0]["top_logprobs"]], [1, 9])
        self.assertIsNone(c[2]["logprob"])

    def test_off_by_default_and_bad_values(self):
        st, body = self.post("/v1/completions", {"prompt": "hi"})
        self.assertEqual(st, 200)
        self.assertIsNone(body["choices"][0]["logprobs"])
        for path, extra in (("/v1/completions", {"prompt": "hi", "logprobs": 21}),
                            ("/v1/completions", {"prompt": "hi", "logprobs": -1}),
                            ("/v1/completions", {"prompt": "hi", "logprobs": 2, "stream": True}),
                            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "logprobs": True, "top_logprobs": 25}),
                            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "top_logprobs": 3}),
                            ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "logprobs": 5})):
            self.assertEqual(self.post(path, extra)[0], 400, (path, extra))

    def stop_engine(self):
        eng = FakeEngine("", {"new_tokens": 0, "prompt_tokens": 2, "eos_reason": "stop_token", "token_ids": [],
                              "logprobs": [(-0.1, [(151645, -0.1), (9, -2.5)])], "logprob_token_ids": [151645]})
        eng.token_str = lambda i: f"t{i}"
        return eng

    def test_stop_token_position_is_scored(self):
        """max_tokens=1 and the model's argmax is the stop token: no content, finish_reason stop, logprobs present."""
        async def comp(client):
            r = await client.post("/v1/completions", json={"model": "m", "prompt": "hi", "max_tokens": 1, "logprobs": 2})
            return r.status, await r.json()
        st, body = with_client(self.stop_engine(), comp)
        ch = body["choices"][0]
        self.assertEqual((st, ch["text"], ch["finish_reason"]), (200, "", "stop"))
        self.assertEqual(ch["logprobs"]["token_ids"], [151645])
        self.assertEqual(ch["logprobs"]["top_logprobs_ids"], [[[151645, -0.1], [9, -2.5]]])

        async def chat(client):
            r = await client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                                                 "max_tokens": 1, "logprobs": True, "top_logprobs": 2})
            return r.status, await r.json()
        st, body = with_client(self.stop_engine(), chat)
        ch = body["choices"][0]
        self.assertEqual((st, ch["finish_reason"]), (200, "stop"))
        self.assertEqual([x["token_id"] for x in ch["logprobs"]["content"]], [151645])
        self.assertEqual([t["token_id"] for t in ch["logprobs"]["content"][0]["top_logprobs"]], [151645, 9])

    def test_stop_logprobs_from_event(self):
        import torch
        ev = {"eos_reason": "stop_token", "eos_triggering_token_id": 5, "stop_token_prob": torch.tensor([[0.5]]),
              "stop_token_top_k_tokens": torch.tensor([[[5, 8]]]), "stop_token_top_k_probs": torch.tensor([[[0.5, 0.25]]])}
        e = serve.QwenEngine._stop_logprobs(ev)
        self.assertAlmostEqual(e[0], -0.6931471805599453, places=6)
        self.assertEqual([a for a, _ in e[1]], [5, 8])
        self.assertIsNone(serve.QwenEngine._stop_logprobs({"eos_reason": "max_new_tokens"}))
        self.assertIsNone(serve.QwenEngine._stop_logprobs({"eos_reason": "stop_token"}))

    def test_event_logprobs(self):
        import torch
        ev = {"token_probs": torch.tensor([[0.5, 0.25]]), "top_k_tokens": torch.tensor([[[7, 8], [4, 5]]]),
              "top_k_probs": torch.tensor([[[0.5, 0.25], [0.5, 0.25]]])}
        out = serve.QwenEngine._event_logprobs(ev, 2)
        self.assertAlmostEqual(out[0][0], -0.6931471805599453, places=6)
        self.assertEqual([a for a, _ in out[1][1]], [4, 5])
        self.assertEqual(serve.QwenEngine._event_logprobs({}, 2), [None, None])


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

    def test_unknown_path_and_wrong_method_answer_404_405_in_json(self):
        from aiohttp.test_utils import TestClient, TestServer

        async def check():
            client = TestClient(TestServer(serve.create_app(FakeEngine("x"), "m", THINK_TEMPLATE, serve.load_defaults("/x"))))
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
        self.assertEqual(d["EXL3_PF_DEFER"], "0")
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

        eng = object.__new__(serve.QwenEngine)
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


class VllmCompatTests(unittest.TestCase):
    """POST /tokenize, token-id prompts on /v1/completions, cache_salt (model-free: fake engine)."""

    class Eng(FakeEngine):
        vocab_size = 1000

        def __init__(self, output="a b c", sessions=1, delay=0.0):
            super().__init__(output)
            self.sessions, self.delay, self.resets, self.active, self.log = sessions, delay, [], 0, []

        def encode_ids(self, text, add_special_tokens=False):
            return ([0] if add_special_tokens else []) + [len(w) for w in text.split()]

        async def reset_prefix_cache(self):
            self.resets.append(self.active)
            await asyncio.sleep(self.delay)

        async def generate(self, prompt, **kw):
            self.active += 1
            self.log.append(kw.get("prompt_ids"))
            try:
                await asyncio.sleep(self.delay)
                async for piece in super().generate(prompt, **kw):
                    yield piece
            finally:
                self.active -= 1

    def post(self, engine, path, payload):
        async def fn(client):
            r = await client.post(path, json=payload)
            return r.status, await r.json() if r.content_type == "application/json" else await r.text()
        return with_client(engine, fn)

    def test_tokenize_reply_shape(self):
        eng = self.Eng()
        st, body = self.post(eng, "/tokenize", {"model": "m", "prompt": "ab cde f", "add_special_tokens": False})
        self.assertEqual((st, body), (200, {"count": 3, "max_model_len": 4096, "tokens": [2, 3, 1], "token_strs": None}))
        self.assertEqual(self.post(eng, "/tokenize", {"prompt": "ab"})[1]["tokens"], [0, 2])   # vLLM default: special tokens on
        st, body = self.post(eng, "/tokenize", {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200)
        self.assertGreater(body["count"], 0)

    def test_tokenize_bad_requests(self):
        eng = self.Eng()
        for payload in ({"model": "other", "prompt": "x"}, {"prompt": 5}, {}, {"prompt": "x", "add_special_tokens": "no"},
                        {"messages": []}):
            self.assertEqual(self.post(eng, "/tokenize", payload)[0], 400, payload)
        self.assertEqual(self.post(FakeEngine(""), "/tokenize", {"prompt": "x"})[0], 501)

    def test_token_id_prompt_reaches_the_engine_untouched(self):
        for prompt in ([5, 6, 7, 8], [[5, 6, 7, 8]]):
            eng = self.Eng()
            st, body = self.post(eng, "/v1/completions", {"model": "m", "prompt": prompt, "max_tokens": 8, "ignore_eos": True,
                                                         "temperature": 0.0, "add_special_tokens": False, "cache_salt": "s1"})
            self.assertEqual(st, 200)
            self.assertEqual(eng.log, [[5, 6, 7, 8]])
            self.assertIs(eng.calls[0][1]["ignore_eos"], True)
            self.assertEqual(body["usage"]["prompt_tokens"], 4)                      # the ids sent, not the engine's count
            self.assertEqual(body["usage"]["prompt_tokens_details"], {"cached_tokens": 4})   # clamped to prompt_tokens

    def test_token_id_prompt_stream_ends_with_usage_and_done(self):
        eng = self.Eng("abcdef")
        st, raw = self.post(eng, "/v1/completions", {"model": "m", "prompt": [1, 2, 3], "max_tokens": 8, "stream": True,
                                                    "stream_options": {"include_usage": True}})
        events = sse_events(raw)
        self.assertEqual(st, 200)
        self.assertTrue(raw.rstrip().endswith("data: [DONE]"))
        self.assertEqual("".join(e["choices"][0]["text"] for e in events), "abcdef")
        self.assertEqual(events[-1]["usage"]["prompt_tokens"], 3)
        self.assertEqual(events[-1]["usage"]["prompt_tokens_details"]["cached_tokens"], 3)
        self.assertIsNone(eng.calls[0][1].get("ignore_eos"))

    def test_text_prompt_and_unknown_fields_still_accepted(self):
        eng = self.Eng()
        st, body = self.post(eng, "/v1/completions", {"model": "m", "prompt": "hello there", "max_tokens": 3, "seed": 7,
                                                     "user": "u", "add_special_tokens": False, "stream_options": {"include_usage": True}})
        self.assertEqual(st, 200)
        self.assertEqual(body["usage"]["prompt_tokens"], 11)
        self.assertEqual(eng.log, [None])

    def test_token_id_prompt_validation(self):
        eng = self.Eng()
        for bad in ([-1], [1000], [1, True], [1, 2.5], [1, "a"], [[1], [2]], [[1], 2], [], [[]]):
            self.assertEqual(self.post(eng, "/v1/completions", {"model": "m", "prompt": bad})[0], 400, bad)
        self.assertEqual(eng.log, [])        # nothing reached the engine
        self.assertEqual(self.post(eng, "/v1/completions", {"model": "m", "prompt": [1], "ignore_eos": "yes"})[0], 400)
        self.assertEqual(self.post(eng, "/v1/completions", {"model": "m", "prompt": [1] * 5000})[0], 400)   # longer than ctx

    def test_cache_salt_resets_only_when_the_namespace_changes(self):
        eng = self.Eng()

        async def fn(client):
            codes = []
            for salt in (None, None, "a", "a", "a", "b", "a", None, None):
                payload = {"model": "m", "prompt": [1, 2], "max_tokens": 2} | ({} if salt is None else {"cache_salt": salt})
                codes.append((await client.post("/v1/completions", json=payload)).status)
            return codes
        self.assertEqual(with_client(eng, fn), [200] * 9)
        self.assertEqual(len(eng.resets), 4)        # None -> a, a -> b, b -> a, a -> None

    def test_requests_without_a_salt_never_reset(self):
        eng = self.Eng()

        async def fn(client):
            await client.post("/v1/completions", json={"model": "m", "prompt": [1, 2]})
            await client.post("/v1/completions", json={"model": "m", "prompt": "x y"})
            await client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        with_client(eng, fn)
        self.assertEqual(eng.resets, [])

    def test_chat_accepts_cache_salt_too(self):
        eng = self.Eng()
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}

        async def fn(client):
            return [(await client.post("/v1/chat/completions", json=body | {"cache_salt": s})).status for s in ("x", "x", "y")]
        self.assertEqual(with_client(eng, fn), [200] * 3)
        self.assertEqual(len(eng.resets), 2)

    def test_salt_change_waits_for_the_running_requests(self):
        eng = self.Eng(sessions=3, delay=0.05)

        async def fn(client):
            def req(salt):
                return client.post("/v1/completions", json={"model": "m", "prompt": [1, 2], "cache_salt": salt})
            rs = await asyncio.gather(req("a"), req("a"), req("b"), req("b"), req("a"))
            return [r.status for r in rs]
        self.assertEqual(with_client(eng, fn), [200] * 5)
        self.assertGreaterEqual(len(eng.resets), 2)
        self.assertEqual(eng.resets, [0] * len(eng.resets))     # no request was running when the cache was dropped

    def test_failed_reset_is_retried_by_the_next_request(self):
        class Flaky(self.Eng):
            fail = True

            async def reset_prefix_cache(self):
                await super().reset_prefix_cache()
                if Flaky.fail:
                    Flaky.fail = False
                    raise RuntimeError("boom")
        eng = Flaky()

        async def fn(client):
            body = {"model": "m", "prompt": [1, 2], "cache_salt": "a"}
            return [(await client.post("/v1/completions", json=body)).status for _ in range(2)]
        st = with_client(eng, fn)
        self.assertEqual(st[0], 500)
        self.assertEqual((st[1], len(eng.resets)), (200, 2))

    def test_cache_salt_validation(self):
        eng = self.Eng()
        for bad in ("", 5, ["a"], True):
            self.assertEqual(self.post(eng, "/v1/completions", {"model": "m", "prompt": [1], "cache_salt": bad})[0], 400, bad)
        self.assertEqual(self.post(FakeEngine("x"), "/v1/completions", {"model": "m", "prompt": "hi", "cache_salt": "a"})[0], 400)


class SSEKeepAliveTests(unittest.TestCase):
    """KYOJIN_SSE_PING_S: a stream that has nothing to send yet writes a ':' comment (off by default)."""

    def setUp(self):
        os.environ.pop("KYOJIN_SSE_PING_S", None)
        self.addCleanup(os.environ.pop, "KYOJIN_SSE_PING_S", None)

    def test_a_ping_reaches_the_client_before_the_first_token(self):
        os.environ["KYOJIN_SSE_PING_S"] = "0.2"

        class Slow(FakeEngine):
            async def generate(self, prompt, **kw):
                # this server polls its queue once a second, so the wait has to outlast a poll
                await asyncio.sleep(2.3)          # a prefill the client has to sit through
                async for piece in FakeEngine.generate(self, prompt, **kw):
                    yield piece

        async def fn(client):
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            chunks = []
            while True:
                try:
                    data = await asyncio.wait_for(r.content.readany(), 5)
                except (asyncio.IncompleteReadError, asyncio.TimeoutError):
                    break
                if not data:
                    break
                chunks.append(data)
            return b"".join(chunks)

        body = with_client(Slow("hello"), fn)
        self.assertTrue([line for line in body.split(b"\n") if line.startswith(b":")], body[:400])
        # this template opens a <think> block, so the first visible token is reasoning content ("hel")
        self.assertLess(body.index(b": waiting"), body.index(b"hel"))   # before the first token
        self.assertTrue(body.rstrip().endswith(b"data: [DONE]"))          # the stream still ends the OpenAI way

    def test_a_streaming_reply_does_not_cost_the_client_its_connection(self):
        """The streaming route logs its summary while the reply is being closed: a name that is not
        bound there raises after the body is complete, and the client loses the connection it was
        reusing (the next request on it dies with ServerDisconnectedError, and the server logs a
        traceback per stream). Three streams on one client session must all succeed."""
        from aiohttp.test_utils import TestClient, TestServer

        class Slow(FakeEngine):
            async def generate(self, prompt, **kwargs):
                await asyncio.sleep(0.3)
                async for piece in FakeEngine.generate(self, prompt, **kwargs):
                    yield piece

        async def check():
            client = TestClient(TestServer(serve.create_app(Slow("hello"), "m", "{{ messages[0].content }}")))
            await client.start_server()
            dropped = 0
            for _ in range(3):
                try:
                    response = await client.post("/v1/chat/completions", json={
                        "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
                    await response.read()
                except Exception:                                     # noqa: BLE001  the bug drops the socket
                    dropped += 1
            await client.close()
            return dropped

        self.assertEqual(run(check()), 0)

    def test_nothing_is_written_when_the_switch_is_off(self):
        class Slow(FakeEngine):
            async def generate(self, prompt, **kw):
                await asyncio.sleep(1.3)
                async for piece in FakeEngine.generate(self, prompt, **kw):
                    yield piece

        async def fn(client):
            r = await client.post("/v1/chat/completions", json={
                "model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
            return await r.read()

        body = with_client(Slow("hello"), fn)
        self.assertEqual([line for line in body.split(b"\n") if line.startswith(b":")], [])
        self.assertIn(b"hel", body)


class PrefixDiskConcurrencyTests(unittest.TestCase):
    """The disk cache must work while other requests run (3 users): restore with a busy Generator, save of the request's own prompt."""

    def _engine(self):
        eng = serve.QwenEngine.__new__(serve.QwenEngine)
        calls = []

        class PD:
            def restore(self, ids):
                calls.append(("restore", ids))
                return {"resume_at": 256, "ram_at": 0, "pages": 1, "ms": 1.0}

            def save(self, ids):
                calls.append(("save", ids))
                return {"saved": 1}

        class Gen:
            def num_remaining_jobs(self):
                return 2    # other users are decoding

        async def post(fn, *args):
            return fn(*args)
        eng.prefix_disk, eng.generator, eng._post, eng._disk_ids = PD(), Gen(), post, {}
        return eng, calls

    def test_restore_runs_at_job_start_with_other_jobs(self):
        eng, calls = self._engine()
        job = types.SimpleNamespace(_restore_ids="ids-a")
        eng._pre_start(job)
        self.assertEqual(calls, [("restore", "ids-a")])
        eng._pre_start(job)                        # once per job
        eng._pre_start(types.SimpleNamespace())    # a job without ids (images, cache_prompt=false)
        self.assertEqual(calls, [("restore", "ids-a")])

    def test_save_uses_the_ids_of_its_own_request(self):
        eng, calls = self._engine()
        stats_a, stats_b = {}, {}
        eng._disk_ids[id(stats_a)] = "ids-a"
        eng._disk_ids[id(stats_b)] = "ids-b"     # B started after A and is still running when A ends
        asyncio.run(eng.disk_save(stats_a))
        self.assertEqual(calls, [("save", "ids-a")])
        asyncio.run(eng.disk_save(stats_a))       # nothing left for A
        asyncio.run(eng.disk_save(stats_b))
        self.assertEqual(calls, [("save", "ids-a"), ("save", "ids-b")])


class VramHeadroomTests(unittest.TestCase):
    def test_enough_is_none(self):
        self.assertIsNone(serve.vram_headroom_problem(2200, 262144, 0.0182, 1))

    def test_three_sessions_262144_names_a_context_that_fits(self):
        msg = serve.vram_headroom_problem(468, 262144, 0.0182, 3)   # measured on the 24 GB 3090: Qwen3.8-27B K4V4, sessions 3
        self.assertIn("-c ", msg)
        fits = int(msg.split("-c ")[1].split()[0])
        self.assertEqual(fits % 256, 0)
        self.assertLessEqual(fits, 262144 - (1024 - 468) / 0.0182)
        self.assertGreater(fits, 200000)

    def test_override(self):
        self.assertIsNone(serve.vram_headroom_problem(468, 262144, 0.0182, 3, min_free_mib=256))

    def test_free_vram_is_the_device_figure_after_the_cache_is_returned(self):
        calls = []
        cuda = types.SimpleNamespace(empty_cache=lambda: calls.append("empty"), mem_get_info=lambda: (1369 * 2**20, 24124 * 2**20))
        self.assertEqual(serve.free_vram_mib(types.SimpleNamespace(cuda=cuda)), 1369)
        self.assertEqual(calls, ["empty"])

    def test_1369_mib_left_at_262144_sessions_3_is_refused(self):
        # measured on the 24 GB 3090: this start was accepted (limit 1024), then the third distinct 50K prompt died with CUDA OOM
        msg = serve.vram_headroom_problem(1369, 262144, 0.0182, 3)
        self.assertIn("-c ", msg)
        self.assertLess(int(msg.split("-c ")[1].split()[0]), 262144)

    def test_196608_sessions_3_is_accepted(self):
        self.assertIsNone(serve.vram_headroom_problem(2600, 196608, 0.0182, 3))   # measured: ~2.6 GiB left, 50 calls clean in realA

    def test_lazy_draft_head_subset_counts_against_the_floor(self):
        # measured on the 24 GB 3090, Qwen3.8-27B k6v6 sessions 3 -c 183040: 1551 MiB left after load passed the 1536 floor, the 150 MiB
        # draft-head subset appeared in the first 32K request and the second distinct 32K prompt died with CUDA out of memory (48 MiB short)
        self.assertIsNone(serve.vram_headroom_problem(1551, 183040, 0.0182, 3))
        msg = serve.vram_headroom_problem(1551, 183040, 0.0182, 3, lazy_mib=150)
        self.assertIn("-c ", msg)
        fits = int(msg.split("-c ")[1].split()[0])
        self.assertLessEqual(fits, 183040 - (1536 - 1401) / 0.0182)
        self.assertIsNone(serve.vram_headroom_problem(1551 + 150, 183040, 0.0182, 3, lazy_mib=150))

    def test_lazy_bytes_of_the_draft_subset(self):
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from exllamav3.architecture.qwen3_5_mtp import Qwen3_5MTPModel as M
        inner = type("LinearEXL3", (), {})()
        inner.trellis, inner.svh, inner.K, inner.bias = torch.zeros(256, 1024, 16, dtype=torch.int16), torch.zeros(16384, dtype=torch.float16), 4, None
        lm = types.SimpleNamespace(inner=inner, out_features=16384, pre_scale=1.0, post_scale=1.0, softcap=0.0)
        model = types.SimpleNamespace(modules=[lm], logit_layer_idx=0)
        mtp = types.SimpleNamespace(attached_model=lambda: model, _head_plain=staticmethod(M._head_plain), __dict__={})
        env = {"EXL3_DRAFT_VOCAB": "4096"}
        with unittest.mock.patch.dict(os.environ, env):
            self.assertEqual(M.lazy_vram_bytes(mtp), 256 * 1024 * 16 * 2 // 4 + 4096 * 2)     # a quarter of the head rows
            with unittest.mock.patch.dict(os.environ, {"EXL3_DRAFT_ADAPT": "0"}):
                self.assertEqual(M.lazy_vram_bytes(mtp), 0)
            mtp.__dict__["_dad"] = {"sub": object()}
            self.assertEqual(M.lazy_vram_bytes(mtp), 0)       # already built: it is in the measured free figure
        with unittest.mock.patch.dict(os.environ, {"EXL3_DRAFT_VOCAB": "0"}):
            self.assertEqual(M.lazy_vram_bytes(types.SimpleNamespace(__dict__={})), 0)


class DflashFlagTests(unittest.TestCase):
    def test_dflash_policy_needs_a_drafter_directory(self):
        err = io.StringIO()
        with unittest.mock.patch.object(sys, "argv", ["serve.py", "--model", "m", "--draft-policy", "dflash"]), \
                unittest.mock.patch.object(sys, "stderr", err), self.assertRaises(SystemExit) as cm:
            serve.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--dflash-dir", err.getvalue())

    def _adapt_env(self, *extra, env=None):
        """EXL3_DFLASH_ADAPT as main() leaves it for the engine (load_all stubbed out)."""
        argv = ["serve.py", "--model", "m", "--draft-policy", "dflash", "--dflash-dir", "d", *extra]
        seen = {}

        def fake_load(args):
            seen["adapt"] = os.environ.get("EXL3_DFLASH_ADAPT")
            raise RuntimeError("stop")
        sh = serve.startup_health
        with unittest.mock.patch.dict(os.environ, env or {}), unittest.mock.patch.object(sys, "argv", argv), \
                unittest.mock.patch.object(serve, "load_all", fake_load), \
                unittest.mock.patch.object(sh, "check_model_dir"), unittest.mock.patch.object(sh, "start"), \
                unittest.mock.patch.object(sh, "report_compiler"), unittest.mock.patch.object(sh, "abort"), \
                self.assertRaises(RuntimeError):
            if not env:
                os.environ.pop("EXL3_DFLASH_ADAPT", None)
            serve.main()
        return seen["adapt"]

    def test_adaptive_depth_is_on_by_default(self):
        self.assertEqual(self._adapt_env(), "1")

    def test_adapt_off_flag_selects_the_fixed_block(self):
        self.assertEqual(self._adapt_env("--dflash-adapt", "off"), "0")

    def test_environment_wins_over_the_flag(self):
        self.assertEqual(self._adapt_env("--dflash-adapt", "on", env={"EXL3_DFLASH_ADAPT": "0"}), "0")

    def test_adapt_flag_rejects_other_values(self):
        err = io.StringIO()
        with unittest.mock.patch.object(sys, "argv", ["serve.py", "--model", "m", "--dflash-adapt", "maybe"]), \
                unittest.mock.patch.object(sys, "stderr", err), self.assertRaises(SystemExit) as cm:
            serve.main()
        self.assertEqual(cm.exception.code, 2)


class DflashVramGuardTests(unittest.TestCase):
    def test_min_free_is_the_measured_peak_plus_slack_and_context_scratch(self):
        self.assertAlmostEqual(serve.dflash_min_free_mib(1900, 45056), 1900 + 256 + 352)
        self.assertGreater(serve.dflash_min_free_mib(1900, 131072), serve.dflash_min_free_mib(1900, 45056))

    def test_nanbeige_45056_x2_with_2113_mib_left_is_refused_with_a_context_that_fits(self):
        # measured on the 24 GB 3090: 2113 MiB left after load passed the 1536 floor, then 28 CUDA OOM under two parallel streams
        need = serve.dflash_min_free_mib(1900, 45056)
        msg = serve.vram_headroom_problem(2113, 45056, 0.19, 2, need)
        self.assertIn("-c ", msg)
        fits = int(msg.split("-c ")[1].split()[0])
        self.assertEqual(fits % 256, 0)
        self.assertLess(fits, 45056)
        # the suggested context leaves the needed headroom (the context scratch term shrinks with it)
        self.assertIsNone(serve.vram_headroom_problem(2113 + (45056 - fits) * 0.19, fits, 0.19, 2, serve.dflash_min_free_mib(1900, fits)))

    def test_enough_headroom_passes(self):
        self.assertIsNone(serve.vram_headroom_problem(3000, 45056, 0.19, 2, serve.dflash_min_free_mib(1900, 45056)))

    def test_guard_floor_is_kept_when_the_measured_peak_is_small(self):
        self.assertEqual(max(serve.MIN_FREE_MIB, serve.dflash_min_free_mib(100, 4096)), serve.MIN_FREE_MIB)


class DraftDepthTests(unittest.TestCase):
    """exllamav3/generator/draft_depth.py (pure Python, loaded by path so no torch import)."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = Path(__file__).resolve().parents[2] / "exllamav3" / "generator" / "draft_depth.py"
        spec = importlib.util.spec_from_file_location("draft_depth_t", path)
        cls.M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.M)

    def ctl(self, **kw):
        return self.M.DraftDepth(7, **kw)

    def run_rounds(self, c, accepted, n):
        """n steps; `accepted` is a number or a repeating pattern of accepted counts per speculative round."""
        pat = accepted if isinstance(accepted, (list, tuple)) else (accepted,)
        for i in range(n):
            d = c.want()
            c.record(d, min(pat[i % len(pat)], d) if d else 0)

    def switch_off(self, c):
        while c.depth:
            self.run_rounds(c, 0, 1)

    def test_starts_at_full_depth(self):
        self.assertEqual(self.ctl().want(), 7)

    def test_warm_up_never_cuts(self):
        c = self.ctl()
        self.run_rounds(c, 0, 2)
        self.assertEqual(c.want(), 7)

    def test_high_acceptance_stays_full(self):
        c = self.ctl()
        self.run_rounds(c, 3, 50)
        self.assertEqual(c.want(), 7)

    def test_chat_like_acceptance_switches_off(self):
        c = self.ctl()
        self.run_rounds(c, 0, 8)
        self.assertEqual(c.depth, 0)
        self.assertEqual(c.want(), 0)

    def test_middle_acceptance_gets_the_short_block(self):
        c = self.ctl()
        self.run_rounds(c, (1, 2, 1, 0), 20)
        self.assertEqual(c.depth, 3)
        self.assertEqual(c.want(), 3)

    def test_short_block_returns_to_full_when_acceptance_climbs(self):
        c = self.ctl()
        self.run_rounds(c, (1, 2, 1, 0), 20)
        self.assertEqual(c.depth, 3)
        self.run_rounds(c, 3, 12)
        self.assertEqual(c.depth, 7)

    def test_off_probes_with_a_doubling_interval(self):
        c = self.ctl()
        self.switch_off(c)
        plain = 0
        while c.want() == 0:
            c.record(0, 0)
            plain += 1
        self.assertEqual(plain, 4)                      # first probe after probe_min plain steps
        c.record(c.want(), 0)                           # failed probe
        self.assertEqual(c.depth, 0)
        plain = 0
        while c.want() == 0:
            c.record(0, 0)
            plain += 1
        self.assertEqual(plain, 8)                      # doubled
        for _ in range(10):                             # capped at probe_max
            c.record(c.want() or 0, 0) if c.want() else None
            while c.want() == 0:
                c.record(0, 0)
            c.record(c.want(), 0)
        self.assertLessEqual(c.probe, 32)

    def test_good_probe_switches_speculation_back_on(self):
        c = self.ctl()
        self.switch_off(c)
        while c.want() == 0:
            c.record(0, 0)
        for _ in range(4):                              # a request that turned agentic: probes accept the whole block
            c.record(c.want() or 7, 7)
            while c.want() == 0:
                c.record(0, 0)
        self.assertGreater(c.depth, 0)

    def test_env_overrides_the_thresholds(self):
        with unittest.mock.patch.dict(os.environ, {"EXL3_DFLASH_ADAPT_OFF": "5", "EXL3_DFLASH_ADAPT_WARM": "1"}):
            c = self.ctl()
        self.run_rounds(c, 3, 3)
        self.assertEqual(c.depth, 0)


if __name__ == "__main__":
    unittest.main()
