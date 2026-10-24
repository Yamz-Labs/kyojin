"""The DS4.1 server over the synthetic pack (converter layout, framework loader, serving model), CPU only.

The model runs for real (prefill, decode, Engram rows, prefix cache). Where a test needs a well-formed reply (tool
calls, reasoning), a scripted sampler picks the tokens: the forward passes still run, only the choice is fixed."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("aiohttp")
import torch  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (str(HERE), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)
import ds41_engine as eng  # noqa: E402
import ds41_model as dm  # noqa: E402
import ds41_testkit as kit  # noqa: E402
from ds41_engine import SPECIAL_TOKENS, stub_reply  # noqa: E402

serve = kit.serve_module()
MODEL = "ds41-pack"
TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a value", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}}}}]
HELLO = [{"role": "user", "content": "Hello"}]
EOS = 1


def char_tokenizer(path: Path) -> None:
    """Byte-level char tokenizer with the DS4.1 special tokens: any text round-trips, ids stay below the pack vocab."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel
    vocab = {"<｜begin▁of▁sentence｜>": 0, "<｜end▁of▁sentence｜>": EOS, "<pad>": 2}
    for t in SPECIAL_TOKENS:
        vocab.setdefault(t, len(vocab))
    for ch in sorted(ByteLevel.alphabet()):                   # the Rust set iterates in a random order per process
        vocab[ch] = len(vocab)
    assert len(vocab) < 512
    while len(vocab) < 512:                                   # ids the model can sample but no text produces
        vocab[f"<p{len(vocab)}>"] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([ByteLevel(add_prefix_space=False, use_regex=False),
                                                pre_tokenizers.Split(Regex("[\\s\\S]"), "isolated")])
    tk.decoder = decoders.ByteLevel()
    tk.add_special_tokens(list(SPECIAL_TOKENS) + ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"])
    tk.save(str(path))


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    from exllamav3.dsv41.synth_pack import synth_config, write_pack
    d = tmp_path_factory.mktemp("ds41_pack")
    write_pack(str(d), synth_config(), seed=2)
    char_tokenizer(d / "tokenizer.json")
    # the pack checks its Engram token map against the tokenizer: record the compressed vocab of the new one
    from tokenizers import Tokenizer
    from exllamav3.dsv41.engram import build_compressed_token_map
    tk = Tokenizer.from_file(str(d / "tokenizer.json"))

    class Shim:
        backend_tokenizer = tk

        def __len__(self):
            return tk.get_vocab_size(with_added_tokens=True)
    _, nv = build_compressed_token_map(Shim())
    hf = json.loads((d / "config.json").read_text())
    hf["text_config"]["engram_compressed_vocab_size"] = nv
    (d / "config.json").write_text(json.dumps(hf))
    return str(d)


@pytest.fixture(scope="module")
def engine(pack):
    return dm.load_model(pack, 4096)


class Scripted:
    """Stands in for TopPSampler: returns the scripted token ids, then EOS."""
    queue: list[int] = []

    def __init__(self, *a, **k):
        pass

    def sample(self, logits_row):
        return Scripted.queue.pop(0) if Scripted.queue else EOS


def script(monkeypatch, engine, text: str) -> None:
    monkeypatch.setattr(dm, "TopPSampler", Scripted)
    Scripted.queue = list(engine.tk.encode(text))


def run(coro):
    return asyncio.run(coro)


def sse_events(text):
    out = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                out.append("[DONE]" if line[6:] == "[DONE]" else json.loads(line[6:]))
    return out


async def post(engine, body, path="/v1/chat/completions"):
    client = TestClient(TestServer(serve.create_app(engine, MODEL)))
    await client.start_server()
    try:
        resp = await client.post(path, json={"model": MODEL, **body})
        text = await resp.text()
        return resp.status, text
    finally:
        await client.close()


def chat(engine, **body):
    status, text = run(post(engine, body))
    assert status == 200, text
    return sse_events(text) if body.get("stream") else json.loads(text)


def test_pack_is_detected_and_loaded_as_the_serving_model(pack, engine):
    assert dm.is_pack(pack)
    assert type(engine.model).__name__ == "DSV41Serving" and engine.eos_id == EOS
    text = "héllo <think>x</think>"
    assert engine.tk._tk.decode(engine.tk.encode(text), skip_special_tokens=False) == text


def test_models_and_health(engine):
    async def go():
        client = TestClient(TestServer(serve.create_app(engine, MODEL)))
        await client.start_server()
        try:
            assert (await client.get("/health")).status == 200
            ids = [m["id"] for m in (await (await client.get("/v1/models")).json())["data"]]
            assert ids == [MODEL]
        finally:
            await client.close()
    run(go())


def test_plain_reply_usage_and_timings(engine):
    body = chat(engine, messages=HELLO, max_tokens=12, temperature=0)
    u, t = body["usage"], body["timings"]
    assert u["completion_tokens"] == 12 and u["prompt_tokens"] == u["total_tokens"] - 12
    assert body["choices"][0]["finish_reason"] == "length"
    for key in ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "prompt_per_second", "predicted_per_second"):
        assert key in t, key
    assert t["predicted_n"] == 12 and t["prompt_per_second"] > 0 and t["predicted_per_second"] > 0
    assert t["prompt_n"] + t["cache_n"] == u["prompt_tokens"]
    assert "draft_steps" not in t                              # plain decode: no draft fields


def test_stream_equals_non_stream_and_carries_usage_and_timings(engine):
    one = chat(engine, messages=HELLO, max_tokens=10, temperature=0)
    events = chat(engine, messages=HELLO, max_tokens=10, temperature=0, stream=True,
                  stream_options={"include_usage": True})
    assert events[-1] == "[DONE]"
    text = "".join((e["choices"][0]["delta"].get("content") or "") for e in events[:-1] if e.get("choices"))
    assert text == one["choices"][0]["message"]["content"]
    last = [e for e in events[:-1] if "usage" in e][-1]
    assert last["usage"]["completion_tokens"] == 10 and last["timings"]["predicted_per_second"] > 0


def test_thinking_toggle_changes_the_prompt(engine):
    on = chat(engine, messages=HELLO, max_tokens=2, temperature=0, chat_template_kwargs={"thinking": True})
    off = chat(engine, messages=HELLO, max_tokens=2, temperature=0, chat_template_kwargs={"thinking": False})
    for body, thinking in ((on, True), (off, False)):
        want = engine.count_tokens(kit.reference_prompt(HELLO, None, thinking))
        assert body["usage"]["prompt_tokens"] == want
    assert on["usage"]["prompt_tokens"] != off["usage"]["prompt_tokens"]


def test_reasoning_split(engine, monkeypatch):
    script(monkeypatch, engine, stub_reply("The answer.", reasoning="Let me think."))
    msg = chat(engine, messages=HELLO, max_tokens=64)["choices"][0]["message"]
    assert msg["reasoning_content"] == "Let me think." and msg["content"] == "The answer."


@pytest.mark.parametrize("stream", [False, True])
def test_parallel_tool_calls(engine, monkeypatch, stream):
    calls = [("lookup", {"query": "cats", "limit": 3}), ("lookup", {"query": "dogs"})]
    script(monkeypatch, engine, stub_reply("", reasoning="two lookups", calls=calls))
    out = chat(engine, messages=HELLO, tools=TOOLS, max_tokens=400, stream=stream)
    if stream:
        slots, finish = {}, None
        for ev in out[:-1]:
            if not ev.get("choices"):
                continue
            ch = ev["choices"][0]
            finish = ch["finish_reason"] or finish
            for tc in ch["delta"].get("tool_calls", []):
                s = slots.setdefault(tc["index"], {"name": None, "args": ""})
                s["name"] = tc.get("function", {}).get("name", s["name"])
                s["args"] += tc.get("function", {}).get("arguments", "")
        got = [(s["name"], json.loads(s["args"])) for _, s in sorted(slots.items())]
    else:
        finish = out["choices"][0]["finish_reason"]
        got = [(tc["function"]["name"], json.loads(tc["function"]["arguments"])) for tc in out["choices"][0]["message"]["tool_calls"]]
    assert finish == "tool_calls" and got == [(n, a) for n, a in calls]


def test_multi_turn_tool_round_trip_reaches_the_model(engine, monkeypatch):
    script(monkeypatch, engine, stub_reply("", reasoning="search", calls=[("lookup", {"query": "cats"})]))
    first = chat(engine, messages=[{"role": "user", "content": "Find cats"}], tools=TOOLS, max_tokens=400)
    assistant = first["choices"][0]["message"]
    msgs = [{"role": "user", "content": "Find cats"}, assistant,
            {"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"], "content": "c1, c2, c3"}]
    script(monkeypatch, engine, stub_reply("Three cats.", reasoning="done"))
    second = chat(engine, messages=msgs, tools=TOOLS, max_tokens=200)
    assert second["choices"][0]["message"]["content"] == "Three cats."
    assert second["usage"]["prompt_tokens"] == engine.count_tokens(kit.reference_prompt(msgs, TOOLS, True))


@pytest.mark.parametrize("reply", [
    "ok\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">cat",   # cut mid-call
    "<｜DSML｜ calls>\n<｜DSML｜ invoke>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",                                      # no name
    "</think></think><｜DSML｜ <｜DSML｜ calls>{not xml",                                                           # garbage
])
def test_malformed_tool_output_never_breaks_the_server(engine, monkeypatch, reply):
    script(monkeypatch, engine, reply)
    for stream in (False, True):
        Scripted.queue = list(engine.tk.encode(reply))
        out = chat(engine, messages=HELLO, tools=TOOLS, max_tokens=400, stream=stream)
        if stream:
            assert out[-1] == "[DONE]"
            args: dict[int, str] = {}
            for ev in out[:-1]:
                for tc in (ev.get("choices") or [{"delta": {}}])[0]["delta"].get("tool_calls", []):
                    args[tc["index"]] = args.get(tc["index"], "") + tc.get("function", {}).get("arguments", "")
            for a_ in args.values():
                json.loads(a_)                                # a streamed call is closed into valid JSON
        else:
            for tc in out["choices"][0]["message"].get("tool_calls") or []:
                json.loads(tc["function"]["arguments"])        # whatever is returned is valid JSON


def test_prefix_cache_reuses_the_prompt_and_matches_a_cold_run(pack, engine):
    ids = [(7 * i + 11) % 200 + 3 for i in range(40)]
    ext = ids + [(5 * i + 3) % 200 + 3 for i in range(9)]

    async def gen(e, prompt_ids):
        stats, parts = {}, []
        async for piece in e.generate("", max_tokens=8, temperature=0.0, top_p=1.0, stop=[], prompt_ids=prompt_ids,
                                      ignore_eos=True, stats=stats):
            parts.append(piece)
        return "".join(parts), stats

    def chunked_reference():
        """Greedy tokens of `ext` prefilled the way the cache does (prompt, then the tail at position len(ids)), with no
        cache involved. A one-shot prefill of the same tokens differs by float rounding (different matmul shapes) that the
        tiny random-weight net amplifies into other tokens, so the cache is held to the chunked computation, bit for bit."""
        m, tk = engine.model, engine.tk
        st = m.new_state(1, engine.max_seq)
        m.forward(torch.tensor([ids]), 0, st)
        lg, _ = m.forward(torch.tensor([ext[len(ids):]]), len(ids), st)
        toks, pos = [], len(ext)
        for _ in range(8):
            t = int(lg[-1].argmax())
            toks.append(t)
            lg, _ = m.forward(torch.tensor([[t]]), pos, st)
            pos += 1
        return toks

    async def go():
        engine.pfx.clear()
        _, s0 = await gen(engine, ids)
        warm, s1 = await gen(engine, ext)
        cold_engine = dm.load_model(pack, 4096, prefix_cache=False)
        _, s2 = await gen(cold_engine, ext)
        return s0, s1, s2, warm
    s0, s1, s2, warm = run(go())
    assert s0["cached_tokens"] == 0 and s1["cached_tokens"] == len(ids) and s2["cached_tokens"] == 0
    ref = chunked_reference()
    assert warm and warm == engine.tk._tk.decode(ref, skip_special_tokens=False)


def test_prefix_cache_miss_on_a_diverging_prompt(engine):
    async def gen(ids):
        stats = {}
        async for _ in engine.generate("", max_tokens=2, temperature=0.0, top_p=1.0, stop=[], prompt_ids=ids,
                                       ignore_eos=True, stats=stats):
            pass
        return stats
    a = [10 + i for i in range(30)]
    b = a[:20] + [99, 98, 97, 96, 95]
    run(gen(a))
    assert run(gen(b))["cached_tokens"] == 0


def test_warmup_leaves_no_cache(engine):
    assert engine.warmup(64, 2) > 0 and len(engine.pfx) == 0


def test_spec_hook_none_means_plain_decode(pack):
    e = dm.load_model(pack, 512, num_draft=3, spec_hook=lambda model, max_seq: None)
    assert e.spec is None and e.num_draft == 0


def test_spec_hook_is_called_with_the_model_and_a_seq_length(pack):
    seen = []

    def hook(model, max_seq):
        seen.append(max_seq)
        from exllamav3.dsv41.spec import PortSpec
        return PortSpec(model, max_seq)
    e = dm.load_model(pack, 512, num_draft=3, spec_hook=hook)
    assert seen and e.spec is not None
    status, text = run(post(e, {"messages": HELLO, "max_tokens": 10, "temperature": 0}))
    assert status == 200 and "draft_steps" in json.loads(text)["timings"]


@pytest.mark.parametrize("extra", [[], ["--num-draft", "3"]])
def test_command_line_start_warmup_and_request(pack, extra):
    """`serve.py --model <pack>`: the process starts, warms up, answers /health and a chat request with timings."""
    import os
    import socket
    import subprocess
    import time
    import urllib.request
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, EXL3_SERVE_WARMUP="1")
    proc = subprocess.Popen([sys.executable, str(HERE / "serve.py"), "--model", pack, "--port", str(port), "-c", "1024",
                             "--model-id", MODEL, *extra], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 240
        base = f"http://127.0.0.1:{port}"
        while time.time() < deadline:
            assert proc.poll() is None, proc.stdout.read()
            try:
                if json.load(urllib.request.urlopen(base + "/health", timeout=2)).get("status") == "ok":
                    break
            except Exception:                                   # noqa: BLE001  (still loading)
                time.sleep(1)
        else:
            raise AssertionError("server not ready")
        req = urllib.request.Request(base + "/v1/chat/completions", method="POST",
                                     headers={"Content-Type": "application/json"},
                                     data=json.dumps({"model": MODEL, "messages": HELLO, "max_tokens": 8, "temperature": 0}).encode())
        body = json.load(urllib.request.urlopen(req, timeout=120))
        assert body["timings"]["predicted_per_second"] > 0 and body["usage"]["completion_tokens"] > 0
        assert ("draft_steps" in body["timings"]) == bool(extra)
    finally:
        proc.terminate()
        out = proc.communicate(timeout=30)[0]
    assert "warm-up" in out
