#!/usr/bin/env python3
"""flint-serve -- OpenAI-compatible server for Flint EXL3 packs (single GPU, one resident model).

  python tools/flint/serve.py --model <pack> --port 8000 -c 131072 [--kv 8,8]

POST /v1/chat/completions (stream and non-stream), GET /v1/models, GET /health. Uses the model's own chat template
(<pack>/chat_template.jinja), the recommended sampling (temp 1.0 / top-p 0.95 / top-k 20, taken from
generation_config.json) and parses Flint's output: <think>..</think> -> reasoning_content, <tool_call>name<arg_key>k
</arg_key><arg_value>v</arg_value>..</tool_call> -> tool_calls. No grammar constraint on tool calls (the vendor SGLang
kit adds one); finish_reason "tool_calls" when a call was parsed.
"""
from __future__ import annotations

import argparse, asyncio, json, os, re, sys, threading, time, uuid, queue
from pathlib import Path

from aiohttp import web
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.S)
_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def render_prompt(template: str, messages: list[dict], tools: list[dict] | None) -> str:
    env = ImmutableSandboxedEnvironment()
    env.filters["tojson"] = lambda v, ensure_ascii=False, **kw: json.dumps(v, ensure_ascii=ensure_ascii)
    fixed = []
    for m in messages:
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function", c))
                if isinstance(fn.get("arguments"), str):
                    try: fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError: fn["arguments"] = {}
                calls.append({"function": fn})
            m = dict(m, tool_calls=calls)
        if isinstance(m.get("content"), list):    # OpenAI content parts -> text
            m = dict(m, content="".join(p.get("text", "") for p in m["content"] if isinstance(p, dict)))
        fixed.append(m)
    return env.from_string(template).render(messages=fixed, tools=tools or None, add_generation_prompt=True)


def _val(raw: str):
    try: return json.loads(raw)
    except json.JSONDecodeError: return raw


def parse_completion(text: str) -> dict:
    """text = everything the model wrote after <|assistant|>."""
    reasoning, content = "", text
    if THINK_OPEN in text:
        before, _, rest = text.partition(THINK_OPEN)
        reasoning, close, after = rest.partition(THINK_CLOSE)
        content = before + after if close else before
        if not close:        # cut off inside the reasoning
            content = before
    calls = []
    for m in _CALL_RE.finditer(content):
        body = m.group(1)
        name = body.split("<arg_key>", 1)[0].strip()
        if not name or not body.rstrip().endswith("</arg_value>") and "<arg_key>" in body and TOOL_CLOSE not in m.group(0):
            continue
        args = {k.strip(): _val(v) for k, v in _ARG_RE.findall(body)}
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    content = _CALL_RE.sub("", content).strip()
    msg = {"role": "assistant", "content": content}
    if reasoning.strip(): msg["reasoning_content"] = reasoning.strip()
    if calls: msg["tool_calls"] = calls
    return msg


class Engine:
    """One Generator on a worker thread; requests are Jobs, results are pushed to per-job queues."""

    def __init__(self, model_dir: str, ctx: int, kv: str, max_batch: int, ngram: int = 0):
        import torch
        if torch.version.cuda and not torch.version.hip:
            os.environ.setdefault("EXL3_CUDA_GRAPH", "1")   # NVIDIA preset: whole-forward decode graph (read at import)
        from exllamav3 import Config, Model, Cache, Tokenizer, Generator
        from exllamav3.cache import CacheLayer_quant
        self.config = Config.from_directory(model_dir)
        self.model = Model.from_config(self.config)
        if kv == "16":
            self.cache = Cache(self.model, max_num_tokens=ctx)
        else:
            kb, vb = (int(x) for x in kv.split(","))
            self.cache = Cache(self.model, max_num_tokens=ctx, layer_type=CacheLayer_quant, k_bits=kb, v_bits=vb)
        self.model.load()
        self.tok = Tokenizer.from_config(self.config)
        self.gen = Generator(model=self.model, cache=self.cache, tokenizer=self.tok, max_batch_size=max_batch, **({'ngram_match_min': ngram} if ngram else {}))
        self.ctx = ctx
        self.inbox: queue.Queue = queue.Queue()
        self.outq: dict[int, asyncio.Queue] = {}
        self.jobs: dict = {}
        self.loop = None
        self.eos = list(self.config.eos_token_id_list) if getattr(self.config, "eos_token_id_list", None) else [150001, 150003, 150005, 150018]
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            self._loop()
        except BaseException:           # a dead engine thread leaves a zombie server holding the GPU: exit loudly instead
            import traceback; traceback.print_exc(); sys.stderr.flush()
            os._exit(1)

    def _loop(self):
        while True:
            try:
                while True:
                    ident, job = self.inbox.get(block=not self.gen.num_remaining_jobs(), timeout=None if not self.gen.num_remaining_jobs() else 0)
                    if job is None:                      # cancel request
                        j = self.jobs.pop(ident, None)
                        if j is not None:
                            try: self.gen.cancel(j)
                            except Exception: pass
                        continue
                    self.jobs[ident] = job
                    self.gen.enqueue(job)
            except queue.Empty:
                pass
            if not self.gen.num_remaining_jobs():
                continue
            for r in self.gen.iterate():
                q = self.outq.get(r["identifier"])
                if q is not None:
                    self.loop.call_soon_threadsafe(q.put_nowait, r)
                if r.get("eos"):
                    self.jobs.pop(r["identifier"], None)

    def cancel(self, ident):
        self.outq.pop(ident, None)
        self.inbox.put((ident, None))

    async def submit(self, prompt: str, max_new: int, sampler, stop: list[str]):
        from exllamav3 import Job
        ids = self.tok.encode(prompt, add_bos=False, encode_special_tokens=True)
        ident = uuid.uuid4().int & 0x7FFFFFFF
        self.loop = asyncio.get_running_loop()
        q = asyncio.Queue()
        self.outq[ident] = q
        job = Job(input_ids=ids, max_new_tokens=max_new, sampler=sampler, stop_conditions=self.eos + stop,
                  identifier=ident, token_healing=False, decode_special_tokens=True)
        self.inbox.put((ident, job))
        return ident, ids.shape[-1], q


def make_app(engine: Engine, model_id: str, template: str, defaults: dict) -> web.Application:
    from exllamav3.generator.sampler import ComboSampler

    async def chat_inner(request: web.Request):
        body = await request.json()
        msgs, tools = body.get("messages") or [], body.get("tools")
        prompt = render_prompt(template, msgs, tools)
        temp = float(body.get("temperature", defaults["temperature"]))
        top_p = float(body.get("top_p", defaults["top_p"]))
        top_k = int(body.get("top_k", defaults["top_k"]))
        sampler = ComboSampler(temperature=temp, top_p=top_p, top_k=top_k)
        n_prompt = len(engine.tok.encode(prompt, add_bos=False, encode_special_tokens=True)[0])
        room = engine.ctx - n_prompt - 1
        if room <= 0:
            return web.json_response({"error": {"message": f"prompt is {n_prompt} tokens, context is {engine.ctx}"}}, status=400)
        max_new = min(int(body.get("max_tokens") or body.get("max_completion_tokens") or defaults["max_new"]), room)
        stop = body.get("stop") or []
        stop = [stop] if isinstance(stop, str) else stop
        ident, n_in, q = await engine.submit(prompt, max_new, sampler, stop)
        request["ident"] = ident
        created, cid = int(time.time()), f"chatcmpl-{uuid.uuid4().hex[:20]}"
        stream = bool(body.get("stream"))
        text, n_out, finish = "", 0, "stop"
        stats: dict = {}
        t_req = time.time()

        def log_req():
            print(f"req prompt={n_in} cached={stats.get('cached_tokens', 0)} new={n_in - stats.get('cached_tokens', 0)} "
                  f"out={n_out} prefill_s={stats.get('time_prefill', 0):.2f} generate_s={stats.get('time_generate', 0):.2f} "
                  f"queued_s={stats.get('time_enqueued', 0):.2f} wall_s={time.time() - t_req:.2f} finish={finish}", flush=True)

        def usage():
            c = int(stats.get("cached_tokens", 0))
            return {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out,
                    "prompt_tokens_details": {"cached_tokens": c}}

        async def collect(on_text=None):
            nonlocal text, n_out, finish
            while True:
                r = await q.get()
                if r.get("text"):
                    text += r["text"]
                    if on_text: await on_text(r["text"])
                if r.get("token_ids") is not None:
                    n_out += int(r["token_ids"].numel())
                if r.get("eos"):
                    stats.update({k: r[k] for k in ("cached_tokens", "time_prefill", "time_generate", "time_enqueued") if k in r})
                    finish = "length" if r.get("eos_reason") == "max_new_tokens" else "stop"
                    break
            engine.outq.pop(ident, None)

        if not stream:
            await collect()
            msg = parse_completion(text)
            if msg.get("tool_calls"): finish = "tool_calls"
            log_req()
            return web.json_response({"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                                      "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                      "usage": usage()})

        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        state = {"mode": "pre", "buf": ""}     # pre: before any marker; think; content; tool

        async def send(delta, fin=None):
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
            await resp.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())

        await send({"role": "assistant", "content": ""})

        async def on_text(t):
            # a tiny state machine: emit reasoning / content as they arrive, hold tool-call text back
            state["buf"] += t
            while True:
                b, m = state["buf"], state["mode"]
                if m in ("pre", "content"):
                    marks = [(b.find(x), x) for x in (THINK_OPEN, TOOL_OPEN) if b.find(x) >= 0]
                    if marks:
                        i, x = min(marks)
                        if i: await send({"content": b[:i]})
                        state["buf"] = b[i + len(x):]
                        state["mode"] = "think" if x == THINK_OPEN else "tool"
                        if x == TOOL_OPEN: state["buf"] = TOOL_OPEN + state["buf"]
                        continue
                    keep = max((k for x in (THINK_OPEN, TOOL_OPEN) for k in range(1, len(x)) if b.endswith(x[:k])), default=0)
                    if len(b) > keep:
                        await send({"content": b[:len(b) - keep]}); state["buf"] = b[len(b) - keep:]
                    return
                if m == "think":
                    i = b.find(THINK_CLOSE)
                    if i >= 0:
                        if i: await send({"reasoning_content": b[:i]})
                        state["buf"] = b[i + len(THINK_CLOSE):]; state["mode"] = "content"; continue
                    keep = max((k for k in range(1, len(THINK_CLOSE)) if b.endswith(THINK_CLOSE[:k])), default=0)
                    if len(b) > keep:
                        await send({"reasoning_content": b[:len(b) - keep]}); state["buf"] = b[len(b) - keep:]
                    return
                return   # tool: buffered until the end

        await collect(on_text)
        rest = state["buf"]
        if state["mode"] in ("pre", "content") and rest:
            await send({"content": rest})
        elif state["mode"] == "think" and rest:
            await send({"reasoning_content": rest})
        calls = []
        if state["mode"] == "tool" or TOOL_OPEN in text:
            msg = parse_completion(text)
            calls = msg.get("tool_calls", [])
            if calls:
                await send({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]})
                finish = "tool_calls"
        await send({}, finish)
        if body.get("stream_options", {}).get("include_usage", True):
            await resp.write(f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model_id, 'choices': [], 'usage': usage()})}\n\n".encode())
        log_req()
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def chat(request: web.Request):
        try:
            return await chat_inner(request)
        except BaseException:           # client went away (or error): free the job
            if request.get("ident") is not None:
                engine.cancel(request["ident"])
            raise

    async def models(_):
        return web.json_response({"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]})

    async def health(_):
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    return app


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", required=True); ap.add_argument("--model-id", default="flint")
    ap.add_argument("-c", "--ctx", type=int, default=131072)
    ap.add_argument("--kv", default=None, help="16 or K,V bits for the full-attention KV cache, e.g. 8,8 (default: 6,6 on NVIDIA, which fits 256K in 24 GB; 16 elsewhere)")
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--ngram", type=int, default=0, help="n-gram speculative drafting, min match length (0 = off; 5 gave +9 %% decode on code prompts, lossless)")
    ap.add_argument("--default-max-new", type=int, default=32768, help="reply budget when a request omits max_tokens (thinking included)")
    a = ap.parse_args()
    if a.kv is None:
        import torch
        a.kv = "6,6" if torch.version.cuda and not torch.version.hip else "16"
    gc = json.loads((Path(a.model) / "generation_config.json").read_text())
    defaults = {"max_new": a.default_max_new, "temperature": gc.get("temperature", 1.0), "top_p": gc.get("top_p", 0.95), "top_k": gc.get("top_k", 20)}
    template = (Path(a.model) / "chat_template.jinja").read_text()
    engine = Engine(a.model, a.ctx, a.kv, a.max_batch, a.ngram)
    print(f"flint-serve: {a.model} ctx={a.ctx} kv={a.kv} on {a.host}:{a.port}", flush=True)
    web.run_app(make_app(engine, a.model_id, template, defaults), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
