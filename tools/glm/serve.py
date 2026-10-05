#!/usr/bin/env python3
"""OpenAI-compatible, single-flight server for GLM-5.3 EXL3."""
from __future__ import annotations

import traceback
import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
from jinja2.sandbox import ImmutableSandboxedEnvironment

DEFAULT_MODEL = "~/models/glm53-exl3-td205"
EH_SIDECAR_DEFAULT = "~/models/glm53-mtp-eh-proj-bf16.safetensors"
SPEED_ENV = {
    "EXL3_MOE_CFG": "2",
    "EXL3_HIP_PREFILL_MIN_ROWS": "2",
    "EXL3_BLOCK_GRAPH": "1",
    "EXL3_BLOCK_GRAPH_MLA": "2",
    "EXL3_HC_FN_HALF_PF": "1",
    "EXL3_HC_1PASS_PF": "1",
    # 26/09 WINs: decattn2 decode +4.6 % 4K; chunk4096/chunk4096b prefill -7.3 % pf4k.
    "EXL3_DEC_DSA_FAST": "1",
    "EXL3_MIDCHUNK_CKPT": "2",
    "EXL3_PREFILL_BIG_TAIL": "1",
    "EXL3_BIG_TAIL_DENSE": "1",
    "EXL3_PREFILL_BIG_CHUNK_MAXPOS": "32768",  # mem512k1: plain chunks beyond 32K depth, lower peak memory at 512K ctx
    # 27/09 acceptaudit2 WIN: unquantized MTP eh_proj (td205 stores it at 2 bpw, cos 0.79):
    # served accept +0.077 +- 0.008 (n=24), target ids 24/24 equal, serve_accept decode +4.8 %.
    # The published pack does not ship the sidecar: it is optional (see apply_eh_sidecar_default()).
    # 27/09 verifyfuse1 WIN: R-row verify linears in one launch (Hadamards in-kernel):
    # served R=2 round -3.04 +- 0.20 ms, +6.7 % t/s; R=1 bitwise, R=2 greedy 64/64.
    "EXL3_VERIFY_FUSE": "1",
    # 27/09 round2 WIN: one router launch per verify + wide MoE combine; with R=3 (--num-draft 2)
    # served +6.42 +- 1.34 % vs R=2 base (12 prompts/arm, greedy ids 128/128 x 3 equal).
    "EXL3_DEC_ROUTER_ROWS": "2",
    "EXL3_MOE_COMBINE_WIDE": "1",
    # 27/09 moedec1 WIN: R=3 union MoE with RM=3 instead of RM=4 (tighter registers/LDS
    # when every expert serves at most 3 rows); served R=3 round -1.08 ms, +1.46 +- 0.06 % t/s
    # (one box, one load, n=12/arm paired), greedy ids 12x128/128 identical, kernel bitwise.
    "EXL3_MOEDEC1_RM3": "1",
    # kbsgate 71655909: NLL n=64 mean +0.092 % (CI incl 0), median -0.003 %, dPPL -0.04 % -> within GOAL PPL <= +0.1 %; union MoE -3 %.
    "EXL3_MOEDEC1_KBS_A": "3",
    "EXL3_GEMV_R_DEC1": "1",
    # 27/09 dsaglue1 + stackbench3 WIN (one box, one load, n=27/arm paired): decode 29.53 -> 30.04 t/s,
    # +1.74 +- 0.43 %; greedy ids 3x64 and MTP accept identical. (round2's stack saw -0.08 % at n=12.)
    "EXL3_DSA_GLUE_FUSE": "1",
    # 27/09 pffuse1 + stackbench3 WIN: prefill mHC apply + next mix_norm fused; 16K 574.3 -> 579.8
    # (+0.95 +- 0.07 %), pffuse1 4K +1.46 +- 0.48 % (n=10). Decode neutral. Needs the rebuilt ext.
    "EXL3_PF_HC_FUSE": "1",
    # 27/09 moegemm WIN: mpw2x MoE grouped GEMM (no register spill, 224 VGPR): prefill 579.8 -> 598.2
    # t/s 4K (+3.24 +- 0.19 %, n=6), 16K +2.62 +- 0.24 % (n=4); ids identical, logits bitwise at 4K.
    "EXL3_MPW2X": "2",
    "EXL3_MLA_PF_FAST": "1",
    "EXL3_KDA_PF_SPLIT": "2",
    "EXL3_PF_NO_TAIL": "1",
    "EXL3_HOST_LEAN": "1",
    "EXL3_HOST_CUTS": "1",
    "EXL3_PF_GLUE": "1",
    "EXL3_PF_MSPLIT": "2048",
    "EXL3_PF_SKIP": "1",
}

def apply_eh_sidecar_default() -> str:
    """Use the optional unquantized MTP eh_proj sidecar only when it exists.

    An explicit EXL3_MTP_EH_FP16 is never touched (a wrong path still fails loudly at load).
    Otherwise the default path is used if the file is there, and the server starts without it
    (one log line) if not: the sidecar only raises draft acceptance, it is not needed to run.
    Returns the outcome: "explicit", "default" or "absent".
    """
    if "EXL3_MTP_EH_FP16" in os.environ:
        return "explicit"
    path = os.path.expanduser(EH_SIDECAR_DEFAULT)
    if os.path.isfile(path):
        os.environ["EXL3_MTP_EH_FP16"] = path
        return "default"
    print(f"serve: no MTP eh_proj sidecar at {EH_SIDECAR_DEFAULT}: running without it "
          "(draft acceptance is a little lower). To add it: python tools/glm/mtp_eh_sidecar.py "
          "<official GLM-5.3 checkpoint dir> <out.safetensors>, then set EXL3_MTP_EH_FP16=<out.safetensors>.",
          flush=True)
    return "absent"


# The template deliberately uses a zero-width space so normal prose does not trigger tools.
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


# Server-side defaults for fields a client leaves out (set from the CLI in main()).
# The engine's own default stays greedy; the lane passes the model card's sampling.
SERVE_DEFAULTS: dict[str, Any] = {"temperature": 0.0, "top_p": 1.0, "reasoning_effort": None}


def template_kwargs(body: dict[str, Any]) -> dict[str, Any]:
    kw = {"clear_thinking": body.get("clear_thinking")}
    effort = body.get("reasoning_effort", SERVE_DEFAULTS["reasoning_effort"])
    if effort is not None:
        kw["reasoning_effort"] = effort
    return kw


def render_prompt(template: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                  **kwargs: Any) -> str:
    """Render the model's own Jinja chat template."""
    # Same environment as transformers' apply_chat_template.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda value, ensure_ascii=False, indent=None, separators=None, sort_keys=False: \
        json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    def raise_exception(msg): raise ValueError(msg)
    env.globals["raise_exception"] = raise_exception
    # OpenAI sends tool-call arguments as a JSON string; the GLM template iterates them as a dict.
    fixed = []
    for m in messages:
        if m.get("tool_calls"):
            m = dict(m, tool_calls=[
                dict(c, function=dict(c["function"], arguments=_json_value(c["function"]["arguments"])))
                if isinstance(c.get("function", {}).get("arguments"), str) else c
                for c in m["tool_calls"]])
        fixed.append(m)
    return env.from_string(template).render(
        messages=fixed, tools=tools or [], add_generation_prompt=True, **kwargs)


def _json_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_completion(text: str) -> dict[str, Any]:
    """Split GLM think blocks and zero-width GLM tool calls into OpenAI fields."""
    calls: list[dict[str, Any]] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    cursor = 0
    pattern = re.compile(re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE), re.S)
    for match in pattern.finditer(text):
        visible = text[cursor:match.start()]
        if "<think>" in visible:
            before, after = visible.split("<think>", 1)
            content_parts.append(before)
            thinking, _, visible = after.partition("</think>")
            reasoning_parts.append(thinking)
        content_parts.append(visible)
        body = match.group(1)
        name_match = re.match(r"\s*([^\s<]+)", body)
        if not name_match:
            continue
        args: dict[str, Any] = {}
        for part in re.finditer(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", body, re.S):
            args[part.group(1).strip()] = _json_value(part.group(2).strip())
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": name_match.group(1), "arguments": json.dumps(args, ensure_ascii=False)}})
        cursor = match.end()
    tail = text[cursor:]
    if "<think>" in tail:
        before, after = tail.split("<think>", 1)
        content_parts.append(before)
        thinking, _, tail = after.partition("</think>")
        reasoning_parts.append(thinking)
    content_parts.append(tail)
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    reasoning = "".join(reasoning_parts)
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
        message["finish_reason"] = "tool_calls"
    return message


def opens_thinking(prompt: str) -> bool:
    """The GLM generation prompt ends in <think>, so the model output starts inside the block."""
    return prompt.rstrip().endswith("<think>")


def stop_text(text: str, stop: list[str]) -> tuple[str, str | None]:
    """Cut at the earliest stop string and return content and matched stop."""
    positions = [(text.find(s), s) for s in stop if s and text.find(s) >= 0]
    if not positions:
        return text, None
    pos, match = min(positions)
    return text[:pos], match


def sse(data: dict[str, Any], event: str | None = None) -> bytes:
    prefix = f"event: {event}\n" if event else ""
    return (prefix + f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n").encode()


# Decoding-side default penalties (ComboSampler kwargs), e.g. SERVE_SAMPLER_KW='{"dry_multiplier":0.8}'. Empty = none.
DEFAULT_SAMPLER_KW: dict = json.loads(os.environ.get("SERVE_SAMPLER_KW", "{}") or "{}")


class ResidentEngine:
    """One loaded target model, optional MTP model, and resident paged caches."""
    supports_prefix_reuse = True

    def __init__(self, model_path: str, max_history: int = 1, max_ctx: int = 131072, num_draft: int = 1):
        import torch
        from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
        from exllamav3.generator import generator as GM

        torch.set_grad_enabled(False)
        self.torch = torch
        self.Job = Job
        self.config = Config.from_directory(model_path)
        self.model = Model.from_config(self.config)
        self.tokenizer = Tokenizer.from_config(self.config)
        max_position = int(self.config.config_dict.get("max_position_embeddings") or self.config.config_dict.get("text_config", {}).get("max_position_embeddings") or 32768)
        ctx = min(max_position, max_ctx)
        self.cache = Cache(self.model, max_num_tokens=ctx + 4096,
                           max_history=max_history)
        self.model.load(device="cuda:0", progressbar=False)
        self.draft_model = Model.from_config(self.config, component="mtp")
        self.draft_cache = Cache(self.draft_model, max_num_tokens=ctx + 4096,
                                 max_history=max_history)
        self.draft_model.load(device="cuda:0", progressbar=False)
        self.greedy_generator = Generator(
            model=self.model, cache=self.cache, tokenizer=self.tokenizer,
            draft_model=self.draft_model, draft_cache=self.draft_cache, num_draft_tokens=num_draft,
            record_draft_stats=True)
        # n1f2: one drafted token, fused catch-up variant 2. One Generator only: a second one over the
        # same Cache takes ownership and breaks the first. Sampling also goes through MTP: a draft is
        # kept only when the independently sampled target token equals it, so outputs follow the target.
        GM.MTP_FUSE_CATCHUP = 2
        os.environ.setdefault("EXL3_MOE_UNION_V2", "1")  # the "+v2" of n1f2+v2

    def count_tokens(self, text: str) -> int:
        return int(self.tokenizer.encode(text, encode_special_tokens=True).numel())

    async def generate(self, prompt: str, *, max_tokens: int, temperature: float,
                       top_p: float, stop: list[str], sampler_kw: dict | None = None) -> AsyncIterator[str]:
        """Yield incremental decoded text. Caller serializes access to this method."""
        import time
        import torch
        from exllamav3.generator.sampler import ComboSampler, GreedySampler
        t0 = time.perf_counter()
        probe = getattr(self, "rss_probe", None)
        ids = self.tokenizer.encode(prompt, encode_special_tokens=True)
        if getattr(self, "slot_store", None) is not None:
            self.slot_store.note_prompt(prompt, ids)
        self.last_stats = {}
        kw = dict(sampler_kw if sampler_kw is not None else DEFAULT_SAMPLER_KW)  # rep_p / pres_p / dry_* (loop2)
        if temperature == 0:
            sampler = ComboSampler(temperature=0.0, **kw) if kw else GreedySampler()
        else:
            sampler = ComboSampler(temperature=max(temperature, 1e-6), top_p=top_p, min_p=0.0, **kw)
        generator = self.greedy_generator
        conditions = list(self.config.eos_token_id_list) + stop
        job = self.Job(input_ids=ids, max_new_tokens=max_tokens, sampler=sampler,
                       stop_conditions=conditions)
        generator.enqueue(job)
        loop = asyncio.get_running_loop()
        while generator.num_remaining_jobs():
            batch = await loop.run_in_executor(None, lambda: list(generator.iterate()))
            for event in batch:
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                text = event.get("text", "")
                if text:
                    yield text
                if event.get("eos"):
                    self.last_stats = event
                    if probe is not None:
                        probe.request(int(ids.numel()), int(event.get("new_tokens", 0)),
                                      time.perf_counter() - t0)
                    return
        torch.cuda.synchronize()


def warmup(engine: Any, template: str, prompt_tokens: int = 4608, decode_tokens: int = 16) -> float:
    """Run one ~4.6K-token prefill (a full 4096-row chunk plus a tail) and a short MTP decode before serving,
    so Triton JIT, allocator growth and decode graph capture do not land on the first user request."""
    base = " ".join(str(i) for i in range(1000))
    per = max(engine.count_tokens(base), 1) / 1000
    words = " ".join(str(i) for i in range(int(prompt_tokens / per)))
    prompt = render_prompt(template, [{"role": "user", "content": words + "\nCount on."}], None)

    async def run() -> None:
        async for _ in engine.generate(prompt, max_tokens=decode_tokens, temperature=0.0, top_p=1.0, stop=[]):
            pass

    t0 = time.perf_counter()
    asyncio.run(run())
    return time.perf_counter() - t0


def mem_line(stage: str) -> str:
    """GTT (whole unified-memory budget, all processes) next to this process's torch allocator."""
    import torch
    try:
        gtt = int(Path("/sys/class/drm/card1/device/mem_info_gtt_used").read_text()) / 2**30
    except OSError:
        gtt = float("nan")
    try:
        rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except OSError:
        rss = float("nan")
    return (f"serve: mem[{stage}] gtt {gtt:.1f} GB, torch alloc {torch.cuda.memory_allocated() / 2**30:.1f} "
            f"reserved {torch.cuda.memory_reserved() / 2**30:.1f} GB, rss {rss:.1f} GB")


def mem_diag(stage: str) -> None:
    """EXL3_SERVE_MEMDIAG=1: what the caching allocator holds (allocated by block size, inactive slack)."""
    import collections
    import torch
    act, ina = collections.Counter(), []
    for seg in torch.cuda.memory_snapshot():
        for blk in seg["blocks"]:
            if blk["state"] == "active_allocated":
                act[blk["size"]] += 1
            else:
                ina.append(blk["size"])
    big = sorted(((sz * n, sz, n) for sz, n in act.items()), reverse=True)[:14]
    small = sum(sz * n for sz, n in act.items() if sz < 2**26)
    print(f"serve: memdiag[{stage}] active >=64MB: " +
          ", ".join(f"{n}x{sz / 2**20:.0f}MB" for _, sz, n in big) +
          f"; active <64MB total {small / 2**30:.2f} GB; inactive total {sum(ina) / 2**30:.2f} GB, "
          f"largest {sorted(ina, reverse=True)[:5]}", flush=True)


def dense_tune_path() -> Path:
    """Same file the C++ dense GEMM tuner (hgemm.cu dtune) reads and appends to."""
    p = os.environ.get("EXL3_DENSE_GEMM_TUNE_FILE")
    return Path(p) if p else Path(os.environ.get("HOME", "/tmp")) / ".cache/exllamav3/dense_gemm_tune.txt"


def prime_dense_tune(budget_s: float = 600.0) -> tuple[int, float]:
    """Tune every dense prefill GEMM shape at every 256-row class before serving.

    The C++ tuner screens all rocBLAS solutions (~2.6 s per shape) the first time a prefill chunk
    lands in a new 256-row class, inside the request: a new tail length cost one ~40 s stall per
    class (15 shapes). Warm-up has just logged every shape it met, so cross those shapes with all
    classes 256..4096 and pay the tuning here. A cached key costs one GEMM. Tuned winners are
    bit-exact with the default path, so outputs do not change. Returns (keys, seconds)."""
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    path = dense_tune_path()
    if os.environ.get("EXL3_DENSE_GEMM_TUNE", "1") == "0" or not path.exists():
        return 0, 0.0
    # Only the row classes a prefill chunk can reach (<= 4096); the cache file may hold bigger classes
    # from other runs, and sizing the buffers for them cost ~6 GB of GTT on a 128 GB box.
    max_class = int(os.environ.get("EXL3_SERVE_DTUNE_MAX_CLASS", "4096"))
    shapes, classes, have, tag = set(), set(range(256, max_class + 1, 256)), set(), None
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) == 7:
            shapes.add((int(f[1]), int(f[2]), int(f[4]), int(f[5])))
            have.add((f[0], int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5])))
            tag = f[0]   # the last line carries the running library's tag (the C++ side appends)
    # Keys the C++ tuner already holds cost nothing to skip: no buffers, no GEMM.
    todo = [(n, k, ldn, f32, mc) for n, k, ldn, f32 in sorted(shapes) for mc in sorted(classes)
            if (tag, n, k, mc, ldn, f32) not in have]
    print(f"serve: dense GEMM prime: {len(todo)} of {len(shapes) * len(classes)} keys missing", flush=True)
    if not todo:
        return 0, 0.0
    t0, done = time.perf_counter(), 0
    gen = torch.Generator(device="cuda").manual_seed(0)
    # One set of buffers sized for the largest missing key, viewed per key (a/b/c per key left one caching
    # allocator block per distinct size, ~38 GB reserved on td205).
    def cbytes(n, ldn, f32, mc):
        return mc * (n if ldn else n + 64) * (4 if f32 else 2)
    a_buf = torch.randn(max(mc * k for n, k, _, _, mc in todo), device="cuda", dtype=torch.half, generator=gen)
    b_buf = torch.empty(max(k * n for n, k, _, _, _ in todo), device="cuda", dtype=torch.half)
    c_buf = torch.empty(max(cbytes(n, ldn, f32, mc) for n, _, ldn, f32, mc in todo), device="cuda", dtype=torch.uint8)
    print(mem_line("prime-bufs"), flush=True)
    last, shown = None, t0
    for n, k, ldn, f32, mc in todo:
        if time.perf_counter() - t0 > budget_s:
            break
        if time.perf_counter() - shown > 30.0:   # the port opens only after this pass: show it is alive
            shown = time.perf_counter()
            print(f"serve: dense GEMM prime: {done} of {len(todo)} keys, {shown - t0:.0f} s", flush=True)
        dt = torch.float32 if f32 else torch.half
        w = n if ldn else n + 64
        if last != (n, k):   # finite data: the tuner keeps only winners whose output equals the default's
            b_buf[:k * n].view(k, n).normal_(0, 0.02, generator=gen)
            last = (n, k)
        a = a_buf[:mc * k].view(mc, k)
        b = b_buf[:k * n].view(k, n)
        c = c_buf[:mc * w * dt.itemsize].view(dt).view(mc, w)
        ext.hgemm_recon(a, b, c if ldn else c[:, :n])
        done += 1
    torch.cuda.synchronize()
    del a_buf, b_buf, c_buf
    torch.cuda.empty_cache()
    return done, time.perf_counter() - t0


def create_app(engine: Any, model_id: str, template: str) -> web.Application:
    """Build the HTTP layer around a resident engine (also accepts a fake engine in tests)."""
    queue: asyncio.Queue[tuple[dict[str, Any], asyncio.Queue]] = asyncio.Queue()
    lock = asyncio.Lock()  # serializes generation with slot save/restore
    app = web.Application(client_max_size=16 * 1024**2)
    app.update(engine=engine, model_id=model_id, template=template, queue=queue)

    async def models(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "local"}]})

    async def worker() -> None:
        while True:
            body, out = await queue.get()
            store = getattr(engine, "slot_store", None)
            try:
                async with lock:
                    if store is not None:
                        store.busy = True
                    async for delta in engine.generate(
                        body["_prompt"],
                        max_tokens=body.get("max_tokens", 4096), temperature=body.get("temperature", 0.0),
                        top_p=body.get("top_p", 1.0), stop=body.get("_stop", [])):
                        out.put_nowait(delta)
                    out.put_nowait(None)
            except Exception as exc:
                out.put_nowait(exc)
            finally:
                if store is not None:
                    store.busy = False
                queue.task_done()

    async def start(_: web.Application) -> None:
        app["worker_task"] = asyncio.create_task(worker())

    async def close(app_: web.Application) -> None:
        app_["worker_task"].cancel()

    async def completions(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
            if body.get("model", model_id) != model_id:
                raise web.HTTPBadRequest(reason=f"unknown model; expected {model_id}")
            if not isinstance(body.get("messages"), list) or not body["messages"]:
                raise web.HTTPBadRequest(reason="messages must be a non-empty array")
            if "stream" in body and not isinstance(body["stream"], bool):
                raise web.HTTPBadRequest(reason="stream must be boolean")
            stops = body.get("stop", [])
            body["_stop"] = [stops] if isinstance(stops, str) else list(stops)
            prompt = render_prompt(template, body["messages"], body.get("tools"), **template_kwargs(body))
            prompt_tokens = engine.count_tokens(prompt)
            body["_prompt"] = prompt
            for key in ("temperature", "top_p"):
                if body.get(key) is None:
                    body[key] = SERVE_DEFAULTS[key]
        except (json.JSONDecodeError, web.HTTPException, TypeError) as exc:
            if isinstance(exc, web.HTTPException):
                raise
            raise web.HTTPBadRequest(reason=str(exc)) from exc

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        base = {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model_id}

        def event(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return base | {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        out: asyncio.Queue = asyncio.Queue()
        await queue.put((body, out))
        prefix = "<think>" if opens_thinking(prompt) else ""

        async def deltas() -> AsyncIterator[str]:
            while (item := await out.get()) is not None:
                if isinstance(item, Exception):
                    raise item
                yield item

        def usage(raw: str) -> dict[str, int]:
            completion_tokens = engine.count_tokens(raw)
            return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens}

        def finish_for(message: dict[str, Any], raw: str) -> str:
            # A reply cut by max_tokens is "length", not "stop" (clients use it to tell a truncated answer).
            finish = message.pop("finish_reason", "stop")
            if finish == "stop" and engine.count_tokens(raw) >= body.get("max_tokens", 4096):
                finish = "length"
            return finish

        def timings(raw: str) -> dict[str, Any]:
            # llama.cpp names: the swap orchestrator reads cache_n as the verdict of a KV restore
            st = getattr(engine, "last_stats", None) or {}
            cache_n = int(st.get("cached_tokens", 0))
            pre, gen = st.get("time_prefill") or 0.0, st.get("time_generate") or 0.0
            n = int(st.get("new_tokens", 0))
            return {"cache_n": cache_n, "prompt_n": max(prompt_tokens - cache_n, 0),
                    "prompt_ms": pre * 1000, "predicted_n": n, "predicted_ms": gen * 1000,
                    "prompt_per_second": (prompt_tokens - cache_n) / pre if pre else 0.0,
                    "predicted_per_second": n / gen if gen else 0.0}

        if body.get("stream", False):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await response.prepare(request)
            # Reasoning and prose stream as they decode. Tool calls are parsed only from complete
            # XML, so everything from <tool_call> on is held and sent once at the end.
            text, done, mode = prefix, len(prefix), "reasoning" if prefix else "content"
            hold = max([len("</think>"), len(TOOL_OPEN)] + [len(x) for x in body["_stop"]]) - 1

            async def flush(final: bool) -> None:
                nonlocal done, mode
                while mode != "tool":
                    tags = ["</think>"] if mode == "reasoning" else ["<think>", TOOL_OPEN]
                    hits = [(text.find(t, done), t) for t in tags if text.find(t, done) >= 0]
                    end = min(hits)[0] if hits else (len(text) if final else max(done, len(text) - hold))
                    piece = stop_text(text[done:end], body["_stop"])[0]
                    if piece:
                        key = "reasoning_content" if mode == "reasoning" else "content"
                        await response.write(sse(event({key: piece})))
                    done = end
                    if not hits:
                        return
                    pos, tag = min(hits)
                    done = pos + len(tag)
                    mode = {"</think>": "content", "<think>": "reasoning", TOOL_OPEN: "tool"}[tag]

            await response.write(sse(event({"role": "assistant", "content": ""})))
            async for delta in deltas():
                text += delta
                await flush(False)
            text, _ = stop_text(text, body["_stop"])
            await flush(True)
            message = parse_completion(text)
            if "tool_calls" in message:
                await response.write(sse(event({"tool_calls": [
                    dict(call, index=n) for n, call in enumerate(message["tool_calls"])]})))
            await response.write(sse(event({}, finish_for(message, text[len(prefix):])) |
                                     {"usage": usage(text[len(prefix):]), "timings": timings(text)}))
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        raw = "".join([d async for d in deltas()])
        text, _ = stop_text(prefix + raw, body["_stop"])
        message = parse_completion(text)
        finish = finish_for(message, text[len(prefix):])
        return web.json_response({
            "id": request_id, "object": "chat.completion", "created": created, "model": model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage(text[len(prefix):]), "timings": timings(text)})

    async def apply_template(request: web.Request) -> web.Response:
        """llama.cpp's POST /apply-template: render the chat template, generate nothing.

        Same render_prompt() call as /v1/chat/completions, so what the conformance tool reads here is
        byte-for-byte what a request would be sent.
        """
        try:
            body = await request.json()
            if not isinstance(body.get("messages"), list) or not body["messages"]:
                raise web.HTTPBadRequest(reason="messages must be a non-empty array")
            prompt = render_prompt(template, body["messages"], body.get("tools"), **template_kwargs(body))
        except (json.JSONDecodeError, web.HTTPException, TypeError) as exc:
            if isinstance(exc, web.HTTPException):
                raise
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        return web.json_response({"prompt": prompt})

    async def completion(request: web.Request) -> web.Response:
        """Minimal llama.cpp POST /completion over the same queue/generator as /v1/chat/completions.

        The prompt is used verbatim (no chat template, no implicit <think> prefix), and the engine's
        generate() calls slot_store.note_prompt(), so /slots reflects the probe and the slot can be
        saved. "content" is the raw continuation, reasoning block included, as llama.cpp does not
        split <think> for this endpoint.
        """
        try:
            body = await request.json()
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise web.HTTPBadRequest(reason="prompt must be a non-empty string")
            n_predict = int(body.get("n_predict", 128) or 0)
            if n_predict <= 0:
                n_predict = 128
            n_predict = min(n_predict, 8192)
            stops = body.get("stop", [])
            job = {"_prompt": prompt, "max_tokens": n_predict,
                   "temperature": float(body.get("temperature", 0.0)),
                   "top_p": float(body.get("top_p", 1.0)),
                   "_stop": [stops] if isinstance(stops, str) else list(stops)}
        except (json.JSONDecodeError, web.HTTPException, TypeError, ValueError) as exc:
            if isinstance(exc, web.HTTPException):
                raise
            raise web.HTTPBadRequest(reason=str(exc)) from exc

        out: asyncio.Queue = asyncio.Queue()
        await queue.put((job, out))
        text = ""
        while (item := await out.get()) is not None:
            if isinstance(item, Exception):
                raise item
            text += item
        text, stopped = stop_text(text, job["_stop"])
        st = getattr(engine, "last_stats", None) or {}
        return web.json_response({
            "id": f"cmpl-{uuid.uuid4().hex}", "object": "completion", "created": int(time.time()),
            "model": model_id, "content": text,
            "tokens_cached": int(st.get("cached_tokens", 0)),
            "tokens_predicted": engine.count_tokens(text),
            "cache_prompt": bool(body.get("cache_prompt", True)),
            "stop": True, "stopped_eos": bool(st.get("eos")), "stopped_word": stopped,
            "timings": {"prompt_n": int(st.get("cached_tokens", 0)),
                        "predicted_n": int(st.get("new_tokens", 0))}})

    async def slots(_: web.Request) -> web.Response:
        return web.json_response(engine.slot_store.slots())

    async def slot_action(request: web.Request) -> web.Response:
        store = engine.slot_store
        if request.match_info["id"] != "0":
            raise web.HTTPBadRequest(reason="only slot 0 exists")
        action = request.query.get("action")
        body = await request.json() if request.can_read_body else {}
        try:
            async with lock:
                if action == "erase":
                    return web.json_response(store.erase())
                if action not in ("save", "restore"):
                    raise web.HTTPBadRequest(reason="action must be save, restore or erase")
                fn = getattr(store, action)
                result = await asyncio.get_running_loop().run_in_executor(None, fn, body.get("filename", ""))
                return web.json_response(result)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            traceback.print_exc()
            return web.json_response({"error": {"code": 400, "message": f"{type(exc).__name__}: {exc}", "type": "invalid_request_error"}},
                                     status=400)

    if getattr(engine, "slot_store", None) is not None:
        app.router.add_get("/slots", slots)
        app.router.add_post("/slots/{id}", slot_action)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", completions)
    app.router.add_post("/apply-template", apply_template)
    app.router.add_post("/completion", completion)
    app.on_startup.append(start)
    app.on_cleanup.append(close)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-uncensor", action="store_true", help="ignore a bundled uncensor_spec.json in the model directory (same as EXL3_ABLIT_RUNTIME=off)")
    parser.add_argument("--model-id", default="glm-5.3-exl3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-draft", type=int, default=2, help="MTP draft tokens per round (2 = R=3 verify, round2 27/09)")
    parser.add_argument("--max-history", type=int, default=None, help="recurrent state history slots; default = --num-draft. Each extra slot costs ~2.15 GiB")
    parser.add_argument("-c", "--max-ctx", type=int, default=131072, help="context window (capped by max_position_embeddings)")
    parser.add_argument("--slot-save-path", default="~/cache/llama-slots",
                        help="directory for /slots save/restore files (llama.cpp compatible)")
    parser.add_argument("--chat-template", default=None, help="Jinja chat template file (default: the model's)")
    parser.add_argument("--default-temperature", type=float, default=0.0, help="used when a request omits temperature")
    parser.add_argument("--default-top-p", type=float, default=1.0, help="used when a request omits top_p")
    parser.add_argument("--default-reasoning-effort", default=None, help="template reasoning_effort when a request omits it")
    args = parser.parse_args()
    if args.no_uncensor:
        os.environ["EXL3_ABLIT_RUNTIME"] = "off"
    SERVE_DEFAULTS.update(temperature=args.default_temperature, top_p=args.default_top_p,
                          reasoning_effort=args.default_reasoning_effort)
    for key, value in SPEED_ENV.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("EXL3_MOE_UNION_V2", "1")
    apply_eh_sidecar_default()
    engine = ResidentEngine(args.model, max_history=args.max_history or args.num_draft, max_ctx=args.max_ctx, num_draft=args.num_draft)
    template = Path(args.chat_template or Path(args.model) / "chat_template.jinja").expanduser().read_text(encoding="utf-8")
    print(mem_line("loaded"), flush=True)
    # Warm-up before the slot store is attached, so the warm-up prompt never shows in /slots.
    # EXL3_SERVE_WARMUP=0 skips it.
    if os.environ.get("EXL3_SERVE_WARMUP", "1") != "0":
        print(f"serve: warm-up {warmup(engine, template):.1f} s", flush=True)
        print(mem_line("warm"), flush=True)
        if os.environ.get("EXL3_SERVE_EMPTY_CACHE", "1") == "1":   # mem512k1 lever (a): drop the allocator's inactive slack
            import torch
            torch.cuda.empty_cache()
            print(mem_line("emptied"), flush=True)
        if os.environ.get("EXL3_SERVE_MEMDIAG") == "1":
            mem_diag("warm")
        # EXL3_SERVE_DTUNE_PRIME_S: time budget for pre-tuning dense GEMM classes (0 = skip)
        budget = float(os.environ.get("EXL3_SERVE_DTUNE_PRIME_S", "600"))
        if budget > 0:
            keys, secs = prime_dense_tune(budget)
            print(f"serve: dense GEMM prime {keys} keys {secs:.1f} s", flush=True)
            print(mem_line("primed"), flush=True)
    from exllamav3.generator.slot_store import SlotStore
    engine.slot_store = SlotStore(engine.greedy_generator, [engine.cache, engine.draft_cache],
                                  args.slot_save_path, args.model_id)
    if os.environ.get("EXL3_SERVE_RSS_LOG"):  # rssleak1: per-request host-RSS accounting, off by default
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from rss_probe import install
        install(engine)
        print(f"serve: rss probe -> {os.environ['EXL3_SERVE_RSS_LOG']}", flush=True)
    web.run_app(create_app(engine, args.model_id, template), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
