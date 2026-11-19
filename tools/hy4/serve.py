#!/usr/bin/env python3
"""OpenAI-compatible server for Hy4-flash (GGUF Q3_K) on one resident model: /v1/chat/completions, /v1/completions (both with SSE streaming),
/v1/models, /health, /tokenize, /apply-template.

  python tools/hy4/serve.py --model /path/Hy4-flash-Q3_K-00001-of-00002.gguf --port 8300 [-c 16384] [--num-draft 3]

Prefill runs the prefill fast path; decode runs the HIP-graph step with MTP speculative decoding (greedy; the stream equals plain greedy decoding of the
trunk). Sampling parameters are accepted and ignored. One request is served at a time (one resident model); others queue.
The shipped kernel flag set is SHIPPED_ENV below; every entry can be overridden from the environment. Run it inside the strix env wrapper
(tools/strix_halo/env.sh) and a GPU lock / memory cap of your choice: the server takes neither.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import serve_core as C  # noqa: E402
import serve_compat  # noqa: E402

# The flag set of the measured Hy4 decode (graph + MTP draft fast path + MoE expert-major order) and prefill (fused dequant GEMMs, banded fp16 MLA) numbers.
SHIPPED_ENV = {
    "HY4_HEAD_CPU": "0", "HY4_EXPERT_RESIDENT_GB": "72", "HY4_ENGRAM_GPU": "1", "HY4_ATTN_GROUPED": "1", "HY4_INDEXER_FUSED": "1",
    "HY4_NOSYNC": "1",                                                     # static-buffer step (needed by the graph)
    "HY4_HC_FUSED": "1", "HY4_HEAD_BF16": "1", "HY4_MLA_BG": "1", "HY4_ATTN_EAGER": "2", "HY4_KV_FUSED": "1",
    "HY4_IDX_POOLK": "1", "HY4_REF_FUSED": "1", "HY4_REF_ROUTER": "1", "HY4_REF_MLA": "1", "HY4_ROUTER_GEMV": "1", "HY4_MOE_SORT": "1",   # decode set
    "HY4_MTP_FAST": "1", "HY4_MTP_HEAD": "q5k",                            # draft head
    "HY4_NGRAM_EXT": "4", "HY4_NGRAM_MIN": "8", "HY4_NGRAM_GATE": "1", "HY4_NGRAM_LONG": "12",   # prompt-lookup rows on top of the MTP drafts: up to 4 extra verify rows when the last 8 tokens occurred earlier; paused while their acceptance is low
    "HY4_PREFILL_MOE": "1", "HY4_PREFILL_DENSE": "1", "HY4_PREFILL_ATTN": "1", "HY4_PREFILL_IX": "1", "HY4_ENGRAM_PREFETCH": "1",
    "HY4_PF_F16": "1", "HY4_MOE_NFIRST": "1", "KEEPCACHE": "1",           # prefill set
    "HY4_PF_WOT": "0", "HY4_MOE_HIP": "2", "HY4_PF_CK": "1", "HY4_PF_FUSE": "1",      # prefill set 2: shared gm_ext WMMA experts (HY4_GM_EXT), CK MLA core (EXL3_CKMLA_SO), one-pass hc / norm / indexer kernels
}


def apply_env(ctx: int) -> None:
    for k, v in SHIPPED_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("HY4_NOSYNC_CAP", str(((ctx + 64 + 511) // 512) * 512))   # K buffers / token history: fixed for the life of the process


class State:
    def __init__(self, args):
        self.args = args
        self.engine = None
        self.vocab = None
        self.ready = False
        self.error: str | None = None
        self.stage = "starting"
        self.lock: asyncio.Lock | None = None
        self.served = 0


def _err(status: int, message: str, code: str = "invalid_request_error"):
    from aiohttp import web
    return web.json_response(C.error_body(message, code), status=status)


def build_app(st: State):
    from aiohttp import web
    args = st.args

    async def health(_req):
        if st.error:
            return web.json_response({"status": "error", "error": st.error}, status=500)
        if not st.ready:
            return web.json_response({"status": "loading", "stage": st.stage}, status=503)
        return web.json_response({"status": "ok"})

    async def models(_req):
        return web.json_response({"object": "list", "data": [{"id": args.model_id, "object": "model", "created": int(time.time()), "owned_by": "kyojin",
                                                              "max_model_len": args.max_ctx}]})

    def need_ready():
        if not st.ready:
            raise C.RequestError("model is loading" if not st.error else st.error, 503, "unavailable")

    async def read_body(req):
        try:
            return await req.json()
        except Exception:
            raise C.RequestError("the body is not valid JSON")

    def prompt_for_chat(body):
        r = C.parse_chat(body, args.default_max_tokens)
        text = st.vocab.template.render(r["messages"], None, True, reasoning_effort=r["effort"])
        ids = st.vocab.encode(text)
        v = st.vocab
        r["in_think"] = r["effort"] != "no_think" and bool(v.think_begin) and text.rstrip().endswith(v.think_begin)
        return r, ids

    def check_len(n, max_new):
        if n + max_new + args.num_draft + 2 > args.max_ctx:
            raise C.RequestError(f"prompt ({n} tokens) + max_tokens ({max_new}) exceeds the context window ({args.max_ctx})", 400, "context_length_exceeded")

    async def run_job(ids, r, in_think, request):
        """async generator of ("delta", channel, text) events, then ("end", stats, finish_reason, n_generated)"""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        out = C.OutputStream(st.vocab, in_think=in_think, stops=r["stop"])
        cancel = threading.Event()
        res: dict = {}

        def on_tokens(toks):
            for ch, tx in out.push(toks):
                loop.call_soon_threadsafe(q.put_nowait, ("delta", ch, tx))
            return out.done

        def work():
            try:
                res["stats"] = st.engine.generate(ids, r["max_tokens"], on_tokens, speculative=r["speculative"], cancel=cancel.is_set)
            except Exception as e:      # noqa: BLE001
                traceback.print_exc()
                res["error"] = f"{type(e).__name__}: {e}"
            finally:
                for ch, tx in out.finish():
                    loop.call_soon_threadsafe(q.put_nowait, ("delta", ch, tx))
                loop.call_soon_threadsafe(q.put_nowait, ("end",))

        async with st.lock:
            fut = loop.run_in_executor(None, work)
            try:
                while True:
                    ev = await q.get()
                    if ev[0] == "end":
                        break
                    yield ev
            finally:
                cancel.set()
                await fut
        if "error" in res:
            raise RuntimeError(res["error"])
        stats = res["stats"]
        finish = out.finish_reason or "length"
        st.served += 1
        n_gen = out.n_tokens
        pf, dc = stats["prefill_s"], stats["decode_s"]
        print(f"req {st.served}: prompt {stats['prompt_tokens']} tok prefill {pf:.2f} s ({stats['prompt_tokens'] / pf:.0f} tok/s; trunk {stats.get('prefill_trunk_s', 0):.2f} s, "
              f"draft fill {stats.get('prime_s', 0):.2f} s {stats.get('prime_mode')}) | decode {stats['decode_tokens']} tok {dc:.2f} s = "
              f"{stats['decode_tokens'] / dc if dc else 0:.2f} tok/s spec={stats['speculative']} rounds {stats['rounds']} tau {stats['tau']} hist {stats['accept_hist']} | {finish}",
              flush=True)
        yield ("end", stats, finish, n_gen)

    def timings(stats):
        n, dec_n = stats["prompt_tokens"], stats["decode_tokens"]
        return serve_compat.timings(n, 0, stats["prefill_s"], stats["decode_s"], dec_n, speculative=stats["speculative"], rounds=stats["rounds"],
                                    tau=stats["tau"], accept_hist=stats["accept_hist"], ttft_ms=stats["ttft_s"] * 1000,
                                    prefill_trunk_ms=stats.get("prefill_trunk_s", 0) * 1000, draft_fill_ms=stats.get("prime_s", 0) * 1000)

    async def sse(request, gen_chunks):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(request)
        async for chunk in gen_chunks:
            await resp.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def chat(request):
        try:
            need_ready()
            r, ids = prompt_for_chat(await read_body(request))
            check_len(len(ids), r["max_tokens"])
            rid, created = C.new_id("chatcmpl"), int(time.time())
            m = args.model_id
            if not r["stream"]:
                content, reasoning, stats, finish, n_gen = [], [], None, "length", 0
                async for ev in run_job(ids, r, r["in_think"], request):
                    if ev[0] == "delta":
                        (reasoning if ev[1] == "reasoning" else content).append(ev[2])
                    else:
                        _, stats, finish, n_gen = ev
                return web.json_response(C.chat_response(rid, m, created, "".join(content), "".join(reasoning), finish,
                                                         serve_compat.usage(len(ids), n_gen), timings=timings(stats), system_fingerprint="hy4-strix-graph-mtp"))

            async def chunks():
                yield C.chat_chunk(rid, m, created, {"role": "assistant", "content": ""})
                async for ev in run_job(ids, r, r["in_think"], request):
                    if ev[0] == "delta":
                        yield C.chat_chunk(rid, m, created, {"reasoning_content" if ev[1] == "reasoning" else "content": ev[2]})
                    else:
                        _, stats, finish, n_gen = ev
                        yield C.chat_chunk(rid, m, created, {}, finish, timings=timings(stats))
                        if r["include_usage"]:
                            yield {"id": rid, "object": "chat.completion.chunk", "created": created, "model": m, "choices": [],
                                   "usage": serve_compat.usage(len(ids), n_gen)}
            return await sse(request, chunks())
        except C.RequestError as e:
            return _err(e.status, str(e), e.code)
        except RuntimeError as e:
            return _err(500, str(e), "server_error")

    async def completions(request):
        try:
            need_ready()
            r = C.parse_completion(await read_body(request), st.vocab.size, args.default_max_tokens)
            ids = r["prompt_ids"] if r["prompt_ids"] is not None else st.vocab.encode(r["prompt"])
            check_len(len(ids), r["max_tokens"])
            rid, created, m = C.new_id("cmpl"), int(time.time()), args.model_id
            if not r["stream"]:
                text, stats, finish, n_gen = [], None, "length", 0
                async for ev in run_job(ids, r, False, request):
                    if ev[0] == "delta":
                        text.append(ev[2])
                    else:
                        _, stats, finish, n_gen = ev
                body = C.completion_chunk(rid, m, created, "".join(text), finish)
                body.update(usage=serve_compat.usage(len(ids), n_gen), timings=timings(stats))
                return web.json_response(body)

            async def chunks():
                async for ev in run_job(ids, r, False, request):
                    if ev[0] == "delta":
                        yield C.completion_chunk(rid, m, created, ev[2])
                    else:
                        _, stats, finish, n_gen = ev
                        yield C.completion_chunk(rid, m, created, "", finish, timings=timings(stats))
                        if r["include_usage"]:
                            yield {"id": rid, "object": "text_completion", "created": created, "model": m, "choices": [], "usage": serve_compat.usage(len(ids), n_gen)}
            return await sse(request, chunks())
        except C.RequestError as e:
            return _err(e.status, str(e), e.code)
        except RuntimeError as e:
            return _err(500, str(e), "server_error")

    async def tokenize(request):
        try:
            need_ready()
            body = await read_body(request)
            if "messages" in body:
                _, ids = prompt_for_chat(body)
            else:
                ids = st.vocab.encode(str(body.get("content", body.get("prompt", ""))))
            return web.json_response(serve_compat.tokenize_reply(ids, args.max_ctx))
        except C.RequestError as e:
            return _err(e.status, str(e), e.code)

    async def apply_template(request):
        try:
            need_ready()
            r = C.parse_chat(await read_body(request), args.default_max_tokens)
            return web.json_response({"prompt": st.vocab.template.render(r["messages"], None, True, reasoning_effort=r["effort"])})
        except C.RequestError as e:
            return _err(e.status, str(e), e.code)

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/apply-template", apply_template)
    return app


def load_in_background(st: State):
    def work():
        try:
            from engine import Hy4Engine
            a = st.args
            eng = Hy4Engine(TREE, a.model, ctx=a.max_ctx, k=a.num_draft, chunk=a.prefill_chunk, prime_rows=a.draft_fill_rows)

            def progress(s):
                st.stage = s
                print("load:", s, flush=True)
            eng.load(progress)
            if os.environ.get("HY4_WARMUP_TOKENS", "0") != "0":
                st.stage = "warm-up"
                print("load: warm-up", flush=True)
                eng.warmup(int(os.environ["HY4_WARMUP_TOKENS"]))
            st.stage = "tokenizer"
            st.vocab = C.Vocab(eng.meta)
            st.engine = eng
            st.ready = True
            print(f"ready on {a.host}:{a.port} model-id {a.model_id} ctx {a.max_ctx} k={a.num_draft}", flush=True)
        except Exception as e:      # noqa: BLE001
            traceback.print_exc()
            st.error = f"{type(e).__name__}: {e}"
    threading.Thread(target=work, name="hy4-load", daemon=True).start()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="first shard of the Hy4-flash GGUF")
    ap.add_argument("--model-id", default="Hy4-flash")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("-c", "--max-ctx", type=int, default=16384, help="context window: static buffers are sized for it (K rows 0.1 MB per position)")
    ap.add_argument("--num-draft", type=int, default=3, help="MTP draft tokens per round (verify T = k + 1 rows)")
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    ap.add_argument("--draft-fill-rows", type=int, default=0, help="fill the draft head's cache for the last N prompt rows only (0 = the whole prompt)")
    ap.add_argument("--default-max-tokens", type=int, default=1024)
    args = ap.parse_args(argv)
    apply_env(args.max_ctx)
    st = State(args)
    app = build_app(st)

    async def on_start(_app):
        st.lock = asyncio.Lock()
        load_in_background(st)
    app.on_startup.append(on_start)
    from aiohttp import web
    web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
