"""DS4.1 server over the stub engine: HTTP behaviour, streaming, tool round trip, tokenize, defaults. CPU only."""
from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds41_chat as chat  # noqa: E402
import ds41_engine as eng  # noqa: E402
import ds41_testkit as kit  # noqa: E402
from ds41_engine import StubEngine, stub_reply  # noqa: E402

serve = kit.serve_module()
MODEL = "ds41-test"
TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a value", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}}}]
HELLO = [{"role": "user", "content": "Hello"}]


@asynccontextmanager
async def client_for(replies="", defaults=None, **engine_kw):
    engine = StubEngine(replies, **engine_kw)
    client = TestClient(TestServer(serve.create_app(engine, MODEL, defaults)))
    await client.start_server()
    try:
        yield client, engine
    finally:
        await client.close()


def run(coro):
    return asyncio.run(coro)


def sse_events(text: str) -> list:
    out = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                out.append("[DONE]" if line[6:] == "[DONE]" else json.loads(line[6:]))
    return out


def chat_body(messages=None, **kw):
    return {"model": MODEL, "messages": messages or HELLO, **kw}


async def post_chat(client, **kw):
    resp = await client.post("/v1/chat/completions", json=chat_body(**kw))
    return resp, (await resp.json()) if not kw.get("stream") else sse_events(await resp.text())


def assemble(events: list) -> dict:
    """What an OpenAI client builds from the stream."""
    msg = {"content": "", "reasoning_content": "", "tool_calls": {}}
    finish = usage = None
    for ev in events:
        if ev == "[DONE]" or "choices" not in ev:
            continue
        usage = ev.get("usage", usage)
        ch = ev["choices"][0]
        finish = ch["finish_reason"] or finish
        d = ch["delta"]
        msg["content"] += d.get("content") or ""
        msg["reasoning_content"] += d.get("reasoning_content") or ""
        for tc in d.get("tool_calls", []):
            slot = msg["tool_calls"].setdefault(tc["index"], {"id": None, "name": None, "arguments": "", "namespace": None})
            slot["id"] = tc.get("id", slot["id"])
            slot["namespace"] = tc.get("namespace", slot["namespace"])
            fn = tc.get("function", {})
            slot["name"] = fn.get("name", slot["name"])
            slot["arguments"] += fn.get("arguments", "")
    msg["tool_calls"] = [msg["tool_calls"][i] for i in sorted(msg["tool_calls"])]
    return {"message": msg, "finish_reason": finish, "usage": usage}


# ---- plain endpoints -----------------------------------------------------------------------------------


def test_health_models_metrics():
    async def go():
        async with client_for("x") as (c, _):
            h = await (await c.get("/health")).json()
            assert h["status"] == "ok" and h["model"] == MODEL
            m = await (await c.get("/v1/models")).json()
            assert [x["id"] for x in m["data"]] == [MODEL]
            met = await c.get("/metrics")
            assert met.status == 200 and "text/plain" in met.headers["Content-Type"]
    run(go())


def test_routes_table_matches_the_router():
    async def go():
        async with client_for("x") as (c, _):
            seen = {}
            for r in c.app.router.routes():
                if r.method != "HEAD" and r.resource is not None:
                    seen.setdefault(r.resource.canonical, set()).add(r.method)
            assert seen == {p: set(m) for p, m in serve.ROUTES.items()}
    run(go())


def test_score_route_teacher_forced():
    async def go():
        async with client_for("x") as (c, _):
            body = {"prompt_ids": [0, 5, 6], "continuation_ids": [7, 8, 9], "candidate_ids": [[7, 1], [8], [9, 2]]}
            r = await c.post("/v1/ds41/score", json=body)
            assert r.status == 200
            j = await r.json()
            assert [p["top1"] for p in j["positions"]] == [8, 8, 9] and j["prompt_tokens"] == 3
            assert len(j["positions"][0]["cand_logps"]) == 2
            r = await c.post("/v1/ds41/score", json={"messages": HELLO, "continuation_ids": [7], "candidate_ids": [[7]]})
            assert r.status == 200
            for bad in ({"prompt_ids": [1], "continuation_ids": [7], "candidate_ids": []},
                        {"continuation_ids": [7], "candidate_ids": [[7]]},
                        {"prompt_ids": [1], "continuation_ids": [10**9], "candidate_ids": [[7]]}):
                assert (await c.post("/v1/ds41/score", json=bad)).status == 400
    run(go())


# ---- chat, not streamed --------------------------------------------------------------------------------


def test_non_stream_reasoning_and_answer():
    async def go():
        async with client_for(stub_reply("It is 4.", reasoning="2+2")) as (c, e):
            resp, body = await post_chat(c, messages=[{"role": "user", "content": "2+2?"}])
            assert resp.status == 200
            msg = body["choices"][0]["message"]
            assert (msg["content"], msg["reasoning_content"], msg.get("tool_calls")) == ("It is 4.", "2+2", None)
            assert body["choices"][0]["finish_reason"] == "stop" and body["object"] == "chat.completion"
            u = body["usage"]
            assert u["prompt_tokens"] == len(e.calls[0]["ids"]) and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
            assert {"cache_n", "prompt_n", "predicted_n", "predicted_per_second"} <= set(body["timings"])
            assert e.prompts[0].endswith("<｜Assistant｜><think>")
    run(go())


def test_chat_mode_has_no_reasoning():
    async def go():
        async with client_for("Hi there.") as (c, e):
            _, body = await post_chat(c, thinking=False)
            msg = body["choices"][0]["message"]
            assert msg["content"] == "Hi there." and "reasoning_content" not in msg
            assert e.prompts[0].endswith("<｜Assistant｜></think>")
    run(go())


def test_reasoning_effort_and_chat_template_kwargs_reach_the_prompt():
    async def go():
        async with client_for("ok") as (c, e):
            await post_chat(c, reasoning_effort="low")
            await post_chat(c, reasoning_effort="none")
            await post_chat(c, chat_template_kwargs={"thinking": True, "reasoning_effort": 90})
            await post_chat(c, chat_template_kwargs={"enable_thinking": False})
            assert "Reasoning Effort: 50 " in e.prompts[0] and e.prompts[0].endswith("<think>")
            assert "Reasoning Effort" not in e.prompts[1] and e.prompts[1].endswith("</think>")
            assert "Reasoning Effort: 90 " in e.prompts[2]
            assert e.prompts[3].endswith("</think>")
    run(go())


def test_prompt_equals_the_official_rendering():
    async def go():
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello", "reasoning_content": "greet"}, {"role": "user", "content": "Bye"}]
        async with client_for("ok") as (c, e):
            await post_chat(c, messages=msgs, tools=TOOLS)
            assert e.prompts[0] == kit.reference_prompt(msgs, TOOLS, True)
            r = await c.post("/apply-template", json=chat_body(msgs, tools=TOOLS, reasoning_effort="high"))
            assert (await r.json())["prompt"] == kit.reference_prompt(msgs, TOOLS, True, "high")
    run(go())


def test_tool_call_non_stream():
    async def go():
        reply = stub_reply("Checking.", reasoning="need data", calls=[("lookup", {"query": "cats", "limit": 3})])
        async with client_for(reply) as (c, _):
            _, body = await post_chat(c, tools=TOOLS)
            ch = body["choices"][0]
            assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] == "Checking."
            (tc,) = ch["message"]["tool_calls"]
            assert tc["type"] == "function" and tc["id"].startswith("call_") and tc["function"]["name"] == "lookup"
            assert json.loads(tc["function"]["arguments"]) == {"query": "cats", "limit": 3}
    run(go())


def test_tool_round_trip():
    """Call, tool result, final answer: the second prompt carries the call and the result in the official form."""
    async def go():
        replies = [stub_reply("", reasoning="search", calls=[("lookup", {"query": "cats"})]),
                   stub_reply("Three cats.", reasoning="done")]
        async with client_for(replies) as (c, e):
            msgs = [{"role": "user", "content": "Find cats"}]
            _, first = await post_chat(c, messages=msgs, tools=TOOLS)
            assistant = first["choices"][0]["message"]
            assert assistant["tool_calls"] and first["choices"][0]["finish_reason"] == "tool_calls"
            msgs += [assistant, {"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"], "content": "c1, c2, c3"}]
            _, second = await post_chat(c, messages=msgs, tools=TOOLS)
            assert second["choices"][0]["message"]["content"] == "Three cats."
            assert e.prompts[1] == kit.reference_prompt(msgs, TOOLS, True)
            assert "<tool_result>c1, c2, c3</tool_result>" in e.prompts[1]
            assert second["usage"]["prompt_tokens"] == len(e.calls[1]["ids"])
    run(go())


def test_tool_choice_none_removes_tools_and_parsing():
    async def go():
        reply = stub_reply("text", reasoning="r", calls=[("lookup", {"query": "x"})])
        async with client_for(reply) as (c, e):
            _, body = await post_chat(c, tools=TOOLS, tool_choice="none")
            assert "## Tools" not in e.prompts[0]
            msg = body["choices"][0]["message"]
            assert "tool_calls" not in msg and "<｜DSML｜ calls>" in msg["content"]
    run(go())


def test_namespaced_call_keeps_the_namespace():
    async def go():
        reply = stub_reply("", calls=[("search::lookup", {"query": "x"})])
        async with client_for(reply) as (c, _):
            _, body = await post_chat(c, tools=TOOLS)
            tc = body["choices"][0]["message"]["tool_calls"][0]
            assert tc["function"]["name"] == "lookup" and tc.get("namespace") == "search"
    run(go())


# ---- chat, streamed ------------------------------------------------------------------------------------


def test_stream_deltas_and_final_chunk():
    async def go():
        async with client_for(stub_reply("The answer is four.", reasoning="two plus two"), chunk=2) as (c, _):
            resp, events = await post_chat(c, stream=True)
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            assert events[-1] == "[DONE]" and events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
            got = assemble(events)
            assert got["message"]["reasoning_content"] == "two plus two" and got["message"]["content"] == "The answer is four."
            assert got["finish_reason"] == "stop" and got["usage"]["completion_tokens"] > 0
            assert sum(1 for ev in events[:-1] if ev["choices"] and ev["choices"][0]["delta"].get("content")) > 3
            assert all(ev["id"] == events[0]["id"] for ev in events[:-1])
    run(go())


def test_stream_equals_non_stream():
    async def go():
        reply = stub_reply("Checking.", reasoning="need data", calls=[("lookup", {"query": "a \"b\"\nc", "limit": 3}),
                                                                      ("lookup", {"query": "z"})])
        async with client_for(reply, chunk=1) as (c, _):
            _, whole = await post_chat(c, tools=TOOLS)
            _, events = await post_chat(c, tools=TOOLS, stream=True)
            s = assemble(events)
            w = whole["choices"][0]["message"]
            assert s["message"]["content"] == w["content"] and s["message"]["reasoning_content"] == w["reasoning_content"]
            assert [(t["name"], t["arguments"]) for t in s["message"]["tool_calls"]] == \
                [(t["function"]["name"], t["function"]["arguments"]) for t in w["tool_calls"]]
            assert s["finish_reason"] == whole["choices"][0]["finish_reason"] == "tool_calls"
    run(go())


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 8, 64])
def test_stream_tool_call_deltas_any_chunk_size(chunk):
    async def go():
        reply = stub_reply("ok", reasoning="r", calls=[("a::f", {"x": 1, "s": "t"}), ("g", {})])
        async with client_for(reply, chunk=chunk) as (c, _):
            _, events = await post_chat(c, tools=TOOLS, stream=True)
            calls = [tc for ev in events if ev != "[DONE]" and ev["choices"] for tc in ev["choices"][0]["delta"].get("tool_calls", [])]
            starts = [tc for tc in calls if "id" in tc]
            assert [(t["index"], t["function"]["name"], t["function"]["arguments"]) for t in starts] == [(0, "f", ""), (1, "g", "")]
            assert starts[0]["namespace"] == "a" and starts[0]["type"] == "function" and starts[0]["id"] != starts[1]["id"]
            got = assemble(events)["message"]["tool_calls"]
            assert [json.loads(t["arguments"]) for t in got] == [{"x": 1, "s": "t"}, {}]
            if chunk <= 3:                                    # arguments arrive in several pieces, not one
                assert len([tc for tc in calls if "id" not in tc and tc["index"] == 0]) >= 3
    run(go())


def test_stream_finish_reason_length_for_a_cut_tool_call():
    async def go():
        reply = stub_reply("", reasoning="r", calls=[("lookup", {"query": "x" * 400})])
        async with client_for(reply) as (c, _):
            _, events = await post_chat(c, tools=TOOLS, stream=True, max_tokens=300)
            got = assemble(events)
            assert got["finish_reason"] == "length"
            assert isinstance(json.loads(got["message"]["tool_calls"][0]["arguments"]), dict)
            _, whole = await post_chat(c, tools=TOOLS, max_tokens=300)
            assert whole["choices"][0]["finish_reason"] == "length"
    run(go())


def test_finish_reason_length_when_the_budget_ends_the_answer():
    async def go():
        async with client_for("word " * 100) as (c, _):
            _, body = await post_chat(c, thinking=False, max_tokens=10)
            assert body["choices"][0]["finish_reason"] == "length" and body["usage"]["completion_tokens"] <= 10
            _, ok = await post_chat(c, thinking=False, max_tokens=5000)
            assert ok["choices"][0]["finish_reason"] == "stop"
    run(go())


def test_stop_strings_never_leak_into_the_stream():
    async def go():
        async with client_for("alpha beta STOP gamma", chunk=2) as (c, _):
            _, events = await post_chat(c, thinking=False, stream=True, stop=["STOP"])
            got = assemble(events)
            assert got["message"]["content"] == "alpha beta " and "ST" not in got["message"]["content"]
            assert got["finish_reason"] == "stop"
            _, whole = await post_chat(c, thinking=False, stop="STOP")
            assert whole["choices"][0]["message"]["content"] == "alpha beta "
    run(go())


# ---- tokenize ------------------------------------------------------------------------------------------


def test_tokenize_count_equals_the_rendered_prompt_length():
    async def go():
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "héllo 世界"}]
        async with client_for("ok") as (c, e):
            r = await (await c.post("/tokenize", json={"model": MODEL, "messages": msgs, "tools": TOOLS})).json()
            prompt = chat.render_chat(msgs, TOOLS)
            assert r["count"] == len(r["tokens"]) == len(e.prompt_ids(prompt))
            assert r["tokens"] == list(e.prompt_ids(prompt)) and r["tokens"][0] == e.tokenizer.bos_id
            assert r["tokens"].count(e.tokenizer.bos_id) == 1
            assert r["max_model_len"] == e.ctx
            _, body = await post_chat(c, messages=msgs, tools=TOOLS)             # the chat request reports the same length
            assert body["usage"]["prompt_tokens"] == r["count"]
            plain = await (await c.post("/tokenize", json={"prompt": "abc"})).json()
            assert plain["count"] == 4 and plain["tokens"][0] == e.tokenizer.bos_id
            plain = await (await c.post("/tokenize", json={"prompt": "abc", "add_special_tokens": False})).json()
            assert plain["count"] == 3
    run(go())


def test_tokenize_count_with_the_real_tokenizer():
    path = kit.tokenizer_json()
    if path is None:
        pytest.skip("tokenizer.json not available")
    async def go():
        msgs = [{"role": "user", "content": "Find cats"}, {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"query": "cats"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "c1, c2"}]
        async with client_for("ok", tokenizer=eng.HFTokenizer(path)) as (c, e):
            r = await (await c.post("/tokenize", json={"messages": msgs, "tools": TOOLS})).json()
            from tokenizers import Tokenizer
            want = Tokenizer.from_file(str(path)).encode(kit.reference_prompt(msgs, TOOLS, True), add_special_tokens=False).ids
            assert r["tokens"] == want and r["count"] == len(want)
            _, body = await post_chat(c, messages=msgs, tools=TOOLS)
            assert body["usage"]["prompt_tokens"] == len(want)
    run(go())


# ---- /v1/completions and /completion -------------------------------------------------------------------


def test_v1_completions_plain_and_streamed():
    async def go():
        async with client_for("Once upon a time.", chunk=3) as (c, e):
            r = await (await c.post("/v1/completions", json={"model": MODEL, "prompt": "Story:", "max_tokens": 50})).json()
            assert r["object"] == "text_completion" and r["choices"][0]["text"] == "Once upon a time."
            assert r["choices"][0]["finish_reason"] == "stop" and r["usage"]["prompt_tokens"] == len("Story:")
            assert e.prompts[0] == "Story:"                           # verbatim: no template, no BOS
            resp = await c.post("/v1/completions", json={"model": MODEL, "prompt": "Story:", "max_tokens": 50, "stream": True})
            events = sse_events(await resp.text())
            assert "".join(ev["choices"][0]["text"] for ev in events[:-1]) == "Once upon a time." and events[-1] == "[DONE]"
            assert events[-2]["usage"]["completion_tokens"] > 0
            r = await (await c.post("/v1/completions", json={"model": MODEL, "prompt": [5, 6, 7], "max_tokens": 4})).json()
            assert e.calls[-1]["ids"] == [5, 6, 7] and r["usage"]["prompt_tokens"] == 3
            r = await (await c.post("/v1/completions", json={"model": MODEL, "prompt": "x", "max_tokens": 3})).json()
            assert r["choices"][0]["finish_reason"] == "length"
    run(go())


def test_llama_cpp_completion_endpoint():
    async def go():
        async with client_for("raw text") as (c, e):
            r = await (await c.post("/completion", json={"prompt": "abc", "n_predict": 20})).json()
            assert r["content"] == "raw text" and e.prompts[0] == "abc"
    run(go())


# ---- caching -------------------------------------------------------------------------------------------


def test_cached_tokens_and_cache_prompt_false():
    async def go():
        async with client_for("ok") as (c, e):
            _, first = await post_chat(c)
            _, second = await post_chat(c)
            assert first["timings"]["cache_n"] == 0 and second["timings"]["cache_n"] > 0
            assert second["usage"]["prompt_tokens_details"]["cached_tokens"] == second["timings"]["cache_n"]
            _, third = await post_chat(c, cache_prompt=False)
            assert third["timings"]["cache_n"] == 0 and e.calls[-1]["cache_prompt"] is False
    run(go())


def test_cache_salt_change_resets_the_prefix_cache():
    async def go():
        async with client_for("ok") as (c, e):
            await post_chat(c, cache_salt="a")
            await post_chat(c, cache_salt="a")
            assert e.resets <= 1
            before = e.resets
            _, other = await post_chat(c, cache_salt="b")
            assert e.resets == before + 1 and other["timings"]["cache_n"] == 0
            _, again = await post_chat(c, cache_salt="b")
            assert e.resets == before + 1 and again["timings"]["cache_n"] > 0
    run(go())


# ---- sampling defaults ---------------------------------------------------------------------------------


def test_request_values_beat_server_defaults():
    async def go():
        async with client_for("ok", defaults=serve.Defaults(temperature=0.3, top_p=0.8)) as (c, e):
            await post_chat(c)
            await post_chat(c, temperature=0.0, top_p=0.5, max_tokens=7)
            assert (e.calls[0]["temperature"], e.calls[0]["top_p"]) == (0.3, 0.8)
            assert (e.calls[1]["temperature"], e.calls[1]["top_p"], e.calls[1]["max_tokens"]) == (0.0, 0.5, 7)
            await post_chat(c, temperature=None, top_p=None)
            assert (e.calls[2]["temperature"], e.calls[2]["top_p"]) == (0.3, 0.8)
    run(go())


def test_defaults_from_generation_config_then_model_card(tmp_path):
    ns = lambda **kw: serve.argparse.Namespace(**{**dict(model=str(tmp_path), default_temperature=None, default_top_p=None,  # noqa: E731
                                                          default_max_tokens=None, default_reasoning_effort=None,
                                                          default_thinking="on"), **kw})
    d = serve.build_defaults(ns())                                  # no generation_config.json: the model card values
    assert (d.temperature, d.top_p) == (serve.MODEL_CARD_SAMPLING["temperature"], serve.MODEL_CARD_SAMPLING["top_p"])
    (tmp_path / "generation_config.json").write_text(json.dumps({"temperature": 0.6, "top_p": 0.95, "do_sample": True}))
    d = serve.build_defaults(ns())
    assert (d.temperature, d.top_p) == (0.6, 0.95)
    d = serve.build_defaults(ns(default_temperature=0.1))          # a flag beats the file
    assert (d.temperature, d.top_p) == (0.1, 0.95)
    (tmp_path / "generation_config.json").write_text(json.dumps({"temperature": 0.7}))
    d = serve.build_defaults(ns())                                  # partial file: missing keys fall back
    assert (d.temperature, d.top_p) == (0.7, serve.MODEL_CARD_SAMPLING["top_p"])
    d = serve.build_defaults(ns(default_thinking="off", default_reasoning_effort="max"))
    assert d.thinking is False and d.reasoning_effort in ("max", 100)


def test_server_default_thinking_off_and_per_request_override():
    async def go():
        async with client_for("ok", defaults=serve.Defaults(thinking=False)) as (c, e):
            await post_chat(c)
            await post_chat(c, thinking=True)
            assert e.prompts[0].endswith("</think>") and e.prompts[1].endswith("<think>")
    run(go())


def test_server_default_max_tokens():
    async def go():
        async with client_for("word " * 100, defaults=serve.Defaults(max_tokens=6)) as (c, e):
            _, body = await post_chat(c, thinking=False)
            assert e.calls[0]["max_tokens"] == 6 and body["choices"][0]["finish_reason"] == "length"
    run(go())


# ---- errors --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [
    {"messages": []}, {"messages": "hi"}, {"model": "other", "messages": HELLO}, {"messages": HELLO, "stream": "yes"},
    {"messages": HELLO, "n": 2}, {"messages": HELLO, "stop": [1]}, {"messages": HELLO, "reasoning_effort": "turbo"},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
    {"messages": [{"role": "robot", "content": "x"}]}, {"messages": [{"role": "assistant", "content": "x"}]},
    {"messages": HELLO, "max_tokens": 0}, {"messages": HELLO, "temperature": "hot"}, [1, 2]],
    ids=lambda p: json.dumps(p)[:60])
def test_bad_chat_requests_are_400(payload):
    async def go():
        async with client_for("ok") as (c, e):
            r = await c.post("/v1/chat/completions", json=payload)
            assert r.status == 400, await r.text()
            assert e.prompts == []                               # nothing reached the engine
            r = await c.post("/v1/chat/completions", data=b"{not json", headers={"Content-Type": "application/json"})
            assert r.status == 400
    run(go())


@pytest.mark.parametrize("path,payload", [
    ("/v1/completions", {"prompt": ""}), ("/v1/completions", {"prompt": "x", "max_tokens": 0}),
    ("/v1/completions", {"prompt": [10 ** 9]}), ("/v1/completions", {"prompt": "x", "n": 2}),
    ("/tokenize", {}), ("/tokenize", {"messages": []}), ("/tokenize", {"prompt": 5}),
    ("/apply-template", {"messages": []}), ("/completion", {"prompt": ""})], ids=lambda x: x if isinstance(x, str) else "")
def test_bad_other_requests_are_400(path, payload):
    async def go():
        async with client_for("ok") as (c, _):
            r = await c.post(path, json=payload)
            assert r.status == 400, await r.text()
    run(go())


def test_prompt_longer_than_the_context_is_refused():
    async def go():
        async with client_for("ok", ctx=64) as (c, e):
            r = await c.post("/v1/chat/completions", json=chat_body([{"role": "user", "content": "x" * 500}]))
            assert r.status == 400 and e.prompts == []
    run(go())


def test_engine_error_mid_stream_ends_the_stream_cleanly():
    class Broken(StubEngine):
        async def generate(self, prompt, **kw):
            yield "partial"
            raise RuntimeError("boom")

    async def go():
        engine = Broken("")
        client = TestClient(TestServer(serve.create_app(engine, MODEL)))
        await client.start_server()
        try:
            resp = await client.post("/v1/chat/completions", json=chat_body(thinking=False, stream=True))
            events = sse_events(await resp.text())
            assert events[-1] == "[DONE]" and any("error" in ev for ev in events[:-1] if ev != "[DONE]")
            assert events[-2]["choices"][0]["finish_reason"] == "error"
        finally:
            await client.close()
    run(go())


def test_serial_requests_queue_in_order():
    async def go():
        async with client_for(lambda p: p[-40:], chunk=4) as (c, e):
            msgs = [[{"role": "user", "content": f"question {i}"}] for i in range(5)]
            rs = await asyncio.gather(*[post_chat(c, messages=m, thinking=False) for m in msgs])
            assert sorted(m[0]["content"] for m in msgs) == sorted(p.split("<｜User｜>")[1].split("<｜Assistant｜>")[0] for p in e.prompts)
            assert len(e.prompts) == 5 and all(r[0].status == 200 for r in rs)
    run(go())


# ---- command line --------------------------------------------------------------------------------------


def test_parse_engine_opts():
    assert serve.parse_engine_opts(["a=1", "b=x", "c=true", "d=1.5"]) == {"a": 1, "b": "x", "c": True, "d": 1.5}
    with pytest.raises(SystemExit):
        serve.parse_engine_opts(["novalue"])


def test_load_model_stub_and_missing_loader(tmp_path):
    e = eng.load_model(str(tmp_path), 4096, loader="stub", replies="hey")
    assert isinstance(e, StubEngine) and e.ctx == 4096 and e.count_tokens("ab") == 2
    with pytest.raises(eng.EngineUnavailable, match="--loader"):
        eng.load_model(str(tmp_path), 4096, loader="no_such_ds41_loader:load_model")
    assert isinstance(StubEngine(""), eng.Engine)
