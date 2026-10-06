#!/usr/bin/env python3
"""msrv -- OpenAI-compatible server for MiMo-V2.6 EXL3 with DFlash drafting + SpecGate.

The HTTP/streaming layer is adapted from tools/glm/serve.py (the GLM server); the engine wiring is
adapted from scripts/dflash-bench.py so the served decode path is the measured one (plain ~30 t/s,
DFlash with a 4 bpw EXL3 drafter and confidence-truncated draft length).

Speculation is on by default when a drafter is found: --drafter, else $MIMO_DRAFTER, else <model>/drafter, else
<model>-drafter next to the model directory. With no drafter the server logs one line and decodes plain.

Flags: --port --ctx --dflash/--no-dflash --dynamic-draft --draft-confidence --no-spec-gate --ndt.
"""
from __future__ import annotations

import traceback
import argparse
import asyncio
import json
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
from jinja2.sandbox import ImmutableSandboxedEnvironment

# serve_metrics.py is a symlink to the shared tools/strix_halo/serve_metrics.py
from serve_metrics import Metrics

DEFAULT_MODEL = os.path.expanduser("~/models/mimo26-exl3")
DEFAULT_MODEL_ID = "MiMo-2.6-EXL3"
DEFAULT_CTX = 4096
DEFAULT_NDT = 7
# Sampling used when a request omits temperature/top_p. Agent clients often send neither, so
# a plain-greedy default makes the lane loop; the lane launchers pass the model-card values (T1.0, top_p 0.95).
SERVE_DEFAULTS: dict[str, float] = {"temperature": 0.0, "top_p": 1.0}
# The template uses a zero-width space inside the tag so plain prose does not trigger tools.
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


# ---------------------------------------------------------------------------- template


def render_prompt(template: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                  **kwargs: Any) -> str:
    """Render the model's own Jinja chat template (MiMo chat_template.jinja)."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda value, ensure_ascii=False, indent=None, separators=None, \
        sort_keys=False: json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                                    separators=separators, sort_keys=sort_keys)
    def raise_exception(msg): raise ValueError(msg)
    env.globals["raise_exception"] = raise_exception
    # OpenAI sends tool-call arguments as a JSON string; the MiMo template accepts a dict or a
    # string, but normalizing keeps `tool_call.function.arguments` round-tripping exact.
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


# ---------------------------------------------------------------------------- completion parsing

_NAME_RE = re.compile(r"<function=([^\s>]+)\s*>")
_FUNC_RE = re.compile(r"<function=([^\s>]+)\s*>(.*?)(?:</function>|(?=<function=)|\Z)", re.S)
_PARAM_RE = re.compile(r"<parameter=([^\s>]+)\s*>(.*?)</parameter>", re.S)


def _schema_types(tools: list[dict[str, Any]] | None, name: str) -> dict[str, Any]:
    """Property schemas of the declared tool `name` ({} when unknown)."""
    for tool in tools or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if fn.get("name") == name:
            params = fn.get("parameters") or fn.get("input_schema") or {}
            props = params.get("properties") if isinstance(params, dict) else None
            return props if isinstance(props, dict) else {}
    return {}


def _coerce(value: str, schema: Any) -> Any:
    """XML parameter text -> typed value. Strings stay strings unless the schema asks for another type."""
    kind = schema.get("type") if isinstance(schema, dict) else None
    kinds = kind if isinstance(kind, list) else [kind]
    if "string" in kinds and len(kinds) == 1:
        return value
    parsed = _json_value(value)
    if parsed is value:  # not JSON
        return value
    if isinstance(parsed, (dict, list)):
        return parsed
    # scalars: only when the schema declares a non-string type, else keep the text (e.g. "007", "true" as a name)
    if kinds != [None] and any(k in kinds for k in ("integer", "number", "boolean", "null")):
        return parsed
    return value


def _xml_params(raw: str, props: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """MiMo's template teaches <parameter=key>value</parameter>: map it to a dict."""
    found = _PARAM_RE.findall(raw)
    if not found:
        return None
    props = props or {}
    return {k: _coerce(v.strip("\n") if props.get(k, {}).get("type") == "string" else v.strip(), props.get(k))
            for k, v in found}


def _parse_call(name: str, body: str, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    args_text = body.split("</function>")[0].strip()
    args: Any = {}
    if args_text:
        args = _json_value(args_text)
        if isinstance(args, str):  # not JSON: try the XML parameter form
            xml = _xml_params(args_text, _schema_types(tools, name))
            if xml is not None:
                args = xml
    # A malformed body is passed through raw (never silently turned into {} and executed):
    # the client fails json.loads and reports the error back to the model.
    arguments = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else args_text
    return {"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
            "function": {"name": name, "arguments": arguments}}


def parse_completion(text: str, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Split MiMo <think> blocks and <tool_call><function=NAME>...</function></tool_call>
    blocks into OpenAI message fields. A block body is JSON or <parameter=k>v</parameter> XML;
    one block may hold several <function=...> elements. An unterminated trailing <tool_call>
    (generation cut off) is dropped from the content and yields no call."""
    calls: list[dict[str, Any]] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []

    def prose(chunk: str) -> None:
        if "<think>" in chunk:
            before, after = chunk.split("<think>", 1)
            content_parts.append(before)
            thinking, _, chunk = after.partition("</think>")
            reasoning_parts.append(thinking)
        content_parts.append(chunk)

    # A bare "</think>" with no opener is left as literal content: MiMo's template never ends the
    # generation prompt inside a think block (it emits "<think></think>" for enable_thinking=false),
    # so the streamed and non-streamed paths classify the same text identically.
    cursor = 0
    pattern = re.compile(re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE), re.S)
    for match in pattern.finditer(text):
        prose(text[cursor:match.start()])
        cursor = match.end()  # always advance: a nameless block must not leak or duplicate text
        for fn in _FUNC_RE.finditer(match.group(1)):
            calls.append(_parse_call(fn.group(1), fn.group(2), tools))
    tail = text[cursor:]
    open_at = tail.find(TOOL_OPEN)
    if open_at >= 0 and tail[open_at + len(TOOL_OPEN):].lstrip().startswith("<function="):
        tail = tail[:open_at]  # cut-off call: do not leak raw XML; prose that merely mentions the tag stays
    prose(tail)
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
    reasoning = "".join(reasoning_parts)
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
        message["finish_reason"] = "tool_calls"
    return message


def stop_text(text: str, stop: list[str]) -> tuple[str, str | None]:
    positions = [(text.find(s), s) for s in stop if s and text.find(s) >= 0]
    if not positions:
        return text, None
    pos, match = min(positions)
    return text[:pos], match


def token_limit(body: dict[str, Any], default: int = 4096) -> int:
    """Reply token budget: max_completion_tokens (newer OpenAI field) wins over max_tokens."""
    for key in ("max_completion_tokens", "max_tokens"):
        value = body.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return min(value, body.get("_room") or value)
    return min(default, body.get("_room") or default)


def reply_room(engine: Any, prompt_tokens: int) -> int | None:
    """Tokens a reply may still use: server context minus prompt, one slot and the speculative window.
    None when the engine does not report a context (fake engines). A prompt with no room is refused,
    as the cache cannot hold it (the engine would otherwise fail the job with an assertion)."""
    ctx = getattr(engine, "ctx", None)
    if not ctx:
        return None
    room = ctx - prompt_tokens - 1 - getattr(engine, "num_draft", 0)
    if room < 1:
        raise web.HTTPBadRequest(reason=f"prompt is {prompt_tokens} tokens, the server context is {ctx}")
    return room


def sse(data: dict[str, Any], event: str | None = None) -> bytes:
    prefix = f"event: {event}\n" if event else ""
    return (prefix + f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n").encode()


# ---------------------------------------------------------------------------- engine


class ResidentEngine:
    """One loaded MiMo target model, optional DFlash drafter, one Generator, one Cache."""

    def __init__(self, model_path: str, drafter_path: str | None = None, ndt: int = DEFAULT_NDT,
                 spec_gate: bool = False, ctx: int | None = None,
                 reset_gate_per_request: bool = False,
                 dynamic_draft: bool = True, draft_confidence: float = 0.6):
        # Lossless speculation: every verify row uses the exact R=1 arithmetic, so
        # speculative output equals plain greedy output (20/20 prompts). Costs about 5 % chat and
        # 12 % code tok/s. EXL3_MIMO_LOSSLESS=0 restores the fast, not bit-identical verify.
        # Set before the exllamav3 import. Catch-up fusion stays automatic (on with the union MoE).
        if os.environ.get("EXL3_MIMO_LOSSLESS", "1") != "0":
            for k, v in (("EXL3_VERIFY_ATTN_LOOP", "1"), ("EXL3_VERIFY_GEMV_R", "1"),
                         ("EXL3_DEC_MOE_UNION", "1")):
                os.environ.setdefault(k, v)
        # the one-launch wide-N verify GEMV (lm_head) is a Qwen win but costs MiMo
        # about 7 % tok/s (35.6/31.0/42.3 -> 33.2/28.9/39.5). Off by default here; the caller's env wins.
        os.environ.setdefault("EXL3_VERIFY_WIDE_R", "0")
        import torch
        from exllamav3 import Config, Generator, Job, model_init

        torch.set_grad_enabled(False)
        self.torch = torch
        self.model_path = model_path
        self.reset_gate_per_request = reset_gate_per_request
        self.spec_gate_on = bool(spec_gate and drafter_path)
        # The Generator reads this at construction time (generator.py:197).
        os.environ["EXL3_SPEC_GATE"] = "1" if self.spec_gate_on else "0"
        # Device-side unique-expert table. Measured on gfx1151 through this server: without it a
        # speculative round costs 166 ms for 4 tokens (41.5 ms/token) against a 35.7 ms plain step,
        # so the gate switches speculation off and the server runs at plain speed. With it the gate
        # stays speculative (n_spec 145 / n_plain 1) and code decode reaches ~40-44 t/s. An
        # explicit value from the caller's environment always wins.
        os.environ.setdefault("EXL3_DEC_MOE_UNION_DEV", "1")

        argv = ["-m", model_path, "-cs", str(ctx or config_max_position(
            Config.from_directory(model_path)))]
        if drafter_path:
            argv += ["-dm", drafter_path, "-ndt", str(ndt)]
        parser = argparse.ArgumentParser(allow_abbrev=False)
        model_init.add_args(parser, cache=True, add_draft_model_args=True)
        args = parser.parse_args(argv)

        (self.model, self.config, self.cache, self.tokenizer, self.draft_model,
         self.draft_config, self.draft_cache) = model_init.init(args)
        self.ndt = ndt if drafter_path else 1
        self.drafter_path = drafter_path
        self.generator = Generator(
            model=self.model, cache=self.cache, tokenizer=self.tokenizer,
            draft_model=self.draft_model, draft_cache=self.draft_cache,
            num_draft_tokens=self.ndt, record_draft_stats=True,
            dynamic_draft_tokens=True, draft_confidence=draft_confidence)
        # The calibrator is always built by the generator; serving uses it only with --dynamic-draft
        if not dynamic_draft:
            self.generator.draft_calibrator = None
        elif self.generator.draft_calibrator is not None:
            self.load_prior(self.generator.draft_calibrator)
        self.Job = Job
        self.ctx = ctx or config_max_position(self.config)
        self.num_draft = self.ndt

    def count_tokens(self, text: str) -> int:
        return int(self.tokenizer.encode(text, encode_special_tokens=True).numel())

    def reset_gate(self) -> None:
        """Per-request SpecGate reset (open question: state across mixed requests).

        Default is KEEP: the gate's EMAs are pure timing measurements, they carry no prompt
        content, and the shadow-probe re-entry path (k36/k39) re-adapts within a few plain steps
        when the next request's acceptance is different. --spec-gate-reset-per-request forces a
        fresh SpecGate instead (costs the first few steps of every request).
        """
        from exllamav3.generator.generator import SpecGate
        if self.generator.spec_gate is not None:
            self.generator.spec_gate = SpecGate()

    async def generate(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float,
                       stop: list[str], reset_gate: bool = False,
                       cancel: asyncio.Event | None = None) -> AsyncIterator[str]:
        """Yield incremental decoded text. The caller serializes access to this method.

        When `cancel` is set (client gone) the job is cancelled between two generator steps. A step is one
        prefill chunk or one decode step, so a drop during prefill stops at the next chunk boundary."""
        import torch  # noqa: F401  (keeps the ROCm runtime loaded for the worker thread)
        from exllamav3.generator.sampler import ComboSampler, GreedySampler
        ids = self.tokenizer.encode(prompt, encode_special_tokens=True)
        if getattr(self, "slot_store", None) is not None:
            self.slot_store.note_prompt(prompt, ids)
        self.last_stats = {}
        if cancel is not None and cancel.is_set():
            return                          # the client left while the request was queued: no prefill
        sampler = (GreedySampler() if temperature == 0 else
                   ComboSampler(temperature=max(temperature, 1e-6), top_p=top_p, min_p=0.0))
        if reset_gate:
            self.reset_gate()
        job = self.Job(input_ids=ids, max_new_tokens=max_tokens, sampler=sampler,
                       stop_conditions=list(self.config.eos_token_id_list) + stop)
        self.generator.enqueue(job)
        loop = asyncio.get_running_loop()
        self.last_rounds = 0
        try:
            while self.generator.num_remaining_jobs():
                if cancel is not None and cancel.is_set():
                    self.generator.cancel(job)
                    return
                batch = await loop.run_in_executor(None, lambda: list(self.generator.iterate()))
                self.last_rounds += 1  # with a drafter resident, one step is one verification round
                for event in batch:
                    if event.get("error"):
                        raise RuntimeError(str(event["error"]))
                    text = event.get("text", "")
                    if text:
                        yield text
                    if event.get("eos"):
                        self.last_stats = event
                        return
            torch.cuda.synchronize()
        finally:
            # Whatever way the call ends (error, client gone, cancelled task) no job stays in the Generator:
            # a leftover job would run inside the next request.
            if self.generator.num_remaining_jobs():
                self.generator.clear_queue()

    def load_prior(self, cal) -> None:
        """Seed the confidence calibrator from draft_conf_prior.json in the drafter directory (EXL3_DRAFT_PRIOR=0 skips it)."""
        pf = os.path.join(self.drafter_path, "draft_conf_prior.json") if self.drafter_path else None
        if pf and os.path.isfile(pf) and os.environ.get("EXL3_DRAFT_PRIOR", "1") == "1":
            cal.load_prior(pf)

    def gate_stats(self) -> dict[str, Any] | None:
        gate = getattr(self.generator, "spec_gate", None)
        return gate.stats() if gate is not None else None


# ---------------------------------------------------------------------------- HTTP


def create_app(engine: Any, model_id: str, template: str) -> web.Application:
    """Build the HTTP layer around a resident engine (also accepts a fake engine in tests)."""
    queue: asyncio.Queue[tuple[dict[str, Any], asyncio.Queue, dict[str, Any]]] = asyncio.Queue()
    lock = asyncio.Lock()  # serializes generation with slot save/restore
    processing = 0  # admitted requests; closure, app config is immutable after startup
    app = web.Application(client_max_size=16 * 1024**2)
    app.update(engine=engine, model_id=model_id, template=template, queue=queue, metrics=Metrics())

    def observe(st: dict[str, Any], prompt_tokens: int) -> None:
        accepted = int(st.get("accepted_draft_tokens", 0))
        rejected = int(st.get("rejected_draft_tokens", 0))
        proposed = accepted + rejected
        # Dynamic (confidence-truncated) drafts make rounds non-derivable from token counts;
        # the engine counted its decode steps per request in generate(), one round each while
        # speculating. Gate fallbacks (plain steps) make this an upper bound when --spec-gate is on.
        app["metrics"].observe(
            prompt_tokens=prompt_tokens, cached=int(st.get("cached_tokens", 0)),
            predicted=int(st.get("new_tokens", 0)),
            prefill_s=float(st.get("time_prefill") or 0.0), generate_s=float(st.get("time_generate") or 0.0),
            drafts=int(getattr(engine, "last_rounds", 0)) if proposed else 0,
            draft_tokens=proposed, accepted=accepted)

    async def models(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "local"}]})

    async def health(_: web.Request) -> web.Response:
        gate = None
        try:
            gate = engine.gate_stats()
        except Exception:                                        # noqa: BLE001
            gate = None
        return web.json_response({"status": "ok", "model": model_id,
                                  "generator": getattr(engine, "generator", None) is not None,
                                  "spec_gate": gate})

    async def worker() -> None:
        nonlocal processing
        while True:
            body, out, request_stats = await queue.get()
            cancel = body.get("_cancel")
            if cancel is not None and cancel.is_set():   # the client left while the job was queued: no prefill, no decode
                out.put_nowait(None)
                queue.task_done()
                continue
            store = getattr(engine, "slot_store", None)
            processing += 1  # admitted: counted from the dequeue, incl. lock and slot restore
            try:
                async with lock:
                    if store is not None:
                        store.busy = True
                    async for delta in engine.generate(
                        body["_prompt"],
                        max_tokens=token_limit(body),
                        temperature=body.get("temperature", SERVE_DEFAULTS["temperature"]),
                        top_p=body.get("top_p", SERVE_DEFAULTS["top_p"]), stop=body.get("_stop", []),
                        reset_gate=body.get("_reset_gate", False),
                        cancel=body.get("_cancel")):
                        out.put_nowait(delta)
                    request_stats.update(getattr(engine, "last_stats", None) or {})
                    if request_stats:
                        observe(request_stats, body.get("_prompt_tokens") or int(request_stats.get("prompt_tokens", 0)))
                    out.put_nowait(None)
            except Exception as exc:                             # noqa: BLE001
                out.put_nowait(exc)
            finally:
                processing -= 1
                if store is not None:
                    store.busy = False
                queue.task_done()

    async def start(_: web.Application) -> None:
        app["worker_task"] = asyncio.create_task(worker())

    async def close(app_: web.Application) -> None:
        app_["worker_task"].cancel()

    async def drain(request: web.Request, cancel: asyncio.Event, out: asyncio.Queue) -> AsyncIterator[str]:
        """Items the worker produced for one request. Nothing is written to the client while a job is
        queued or prefilling (and never, for a non-streaming reply), so a client that hung up is noticed
        by polling the transport. Raises ConnectionResetError and sets `cancel` when it has left."""
        while True:
            tr = request.transport
            if tr is None or tr.is_closing():
                cancel.set()
                raise ConnectionResetError("client left")
            try:
                item = await asyncio.wait_for(out.get(), 0.5)
            except asyncio.TimeoutError:
                continue
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    def client_left(cancel: asyncio.Event, path: str) -> web.Response:
        cancel.set()
        print(f"serve: client left {path}, job cancelled", file=sys.stderr, flush=True)
        return web.Response(status=499)

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
            body["_reset_gate"] = bool(getattr(app["engine"], "reset_gate_per_request", False))
            prompt = render_prompt(template, body["messages"], body.get("tools"))
            prompt_tokens = engine.count_tokens(prompt)
            body["_prompt_tokens"] = prompt_tokens
            body["_room"] = reply_room(engine, prompt_tokens)
            body["_prompt"] = prompt
            for key in ("temperature", "top_p"):  # absent or explicit null -> lane default
                if body.get(key) is None:
                    body[key] = SERVE_DEFAULTS[key]
        except (json.JSONDecodeError, web.HTTPException, TypeError) as exc:
            if isinstance(exc, web.HTTPException):
                raise
            raise web.HTTPBadRequest(reason=str(exc)) from exc

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        base = {"id": request_id, "object": "chat.completion.chunk", "created": created,
                "model": model_id}

        def event(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return base | {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        out: asyncio.Queue = asyncio.Queue()
        request_stats: dict[str, Any] = {}
        cancel = body["_cancel"] = asyncio.Event()
        await queue.put((body, out, request_stats))
        # MiMo's generation prompt is "<|im_start|>assistant\n": no implicit <think> block.

        def deltas() -> AsyncIterator[str]:
            return drain(request, cancel, out)

        def usage(raw: str) -> dict[str, int]:
            completion_tokens = engine.count_tokens(raw)
            return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens}

        def finish_for(message: dict[str, Any], raw: str) -> str:
            # A reply cut by max_tokens is "length", not "stop" (clients use it to tell a truncated answer).
            finish = message.pop("finish_reason", "stop")
            if finish == "stop" and engine.count_tokens(raw) >= token_limit(body):
                finish = "length"
            return finish

        def timings() -> dict[str, Any]:
            # llama.cpp names: the swap orchestrator reads cache_n as the verdict of a KV restore
            st = request_stats
            cache_n = int(st.get("cached_tokens", 0))
            pre, gen = st.get("time_prefill") or 0.0, st.get("time_generate") or 0.0
            n = int(st.get("new_tokens", 0))
            return {"cache_n": cache_n, "prompt_n": max(prompt_tokens - cache_n, 0),
                    "prompt_ms": pre * 1000, "predicted_n": n, "predicted_ms": gen * 1000,
                    "prompt_per_second": (prompt_tokens - cache_n) / pre if pre else 0.0,
                    "predicted_per_second": n / gen if gen else 0.0}

        if body.get("stream", False):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream",
                                                  "Cache-Control": "no-cache"})
            await response.prepare(request)
            # Reasoning and prose stream as they decode. Tool calls are parsed only from complete
            # XML, so everything from <tool_call> on is held and sent once at the end.
            text, done, mode = "", 0, "content"
            hold = max([len("</think>"), len(TOOL_OPEN)] + [len(x) for x in body["_stop"]]) - 1
            # MiMo opens a turn with "<think>" only when the template asked for it, but a
            # <think>...</think> pair can appear in the output either way, so both tags are watched.
            tag_mode = {"<think>": "reasoning", "</think>": "content", TOOL_OPEN: "tool"}

            async def flush(final: bool) -> None:
                nonlocal done, mode
                while mode != "tool":
                    tags = [t for t, m in tag_mode.items() if t != "</think>" or mode == "reasoning"]
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
                    mode = tag_mode[tag]

            try:
                await response.write(sse(event({"role": "assistant", "content": ""})))
                async for delta in deltas():
                    text += delta
                    await flush(False)
                text, _ = stop_text(text, body["_stop"])
                await flush(True)
                message = parse_completion(text, body.get("tools"))
                if "tool_calls" in message:
                    await response.write(sse(event({"tool_calls": [
                        dict(call, index=n) for n, call in enumerate(message["tool_calls"])]})))
                await response.write(sse(event({}, finish_for(message, text)) |
                                         {"usage": usage(text), "timings": timings()}))
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
            except ConnectionResetError:
                client_left(cancel, "stream")                        # client left: nothing to send
                return response
            except asyncio.CancelledError:
                cancel.set()
                raise
            except Exception as exc:                                   # noqa: BLE001
                cancel.set()
                # Fail inside the stream, but still end it the OpenAI way: clients such as OpenCode report
                # "stream ended without finish_reason" and retry when the connection just drops.
                try:
                    await response.write(sse({"error": {"message": f"{type(exc).__name__}: {exc}"}}))
                    await response.write(sse(event({}, "error")))
                    await response.write(b"data: [DONE]\n\n")
                    await response.write_eof()
                except ConnectionError:
                    pass
            return response

        try:
            raw = "".join([d async for d in deltas()])
        except ConnectionResetError:
            return client_left(cancel, "request")
        except asyncio.CancelledError:
            cancel.set()
            raise
        except Exception:
            cancel.set()
            raise
        text, _ = stop_text(raw, body["_stop"])
        message = parse_completion(text, body.get("tools"))
        finish = finish_for(message, text)
        return web.json_response({
            "id": request_id, "object": "chat.completion", "created": created, "model": model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage(text), "timings": timings()})

    async def apply_template(request: web.Request) -> web.Response:
        """llama.cpp's POST /apply-template: render the chat template, generate nothing.

        Same render_prompt() call as /v1/chat/completions, so what the conformance tool reads here is
        byte-for-byte what a request would be sent.
        """
        try:
            body = await request.json()
            if not isinstance(body.get("messages"), list) or not body["messages"]:
                raise web.HTTPBadRequest(reason="messages must be a non-empty array")
            prompt = render_prompt(template, body["messages"], body.get("tools"))
        except (json.JSONDecodeError, web.HTTPException, TypeError) as exc:
            if isinstance(exc, web.HTTPException):
                raise
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        return web.json_response({"prompt": prompt})

    async def completion(request: web.Request) -> web.Response:
        """Minimal llama.cpp POST /completion over the same queue/generator as /v1/chat/completions.

        The prompt is used verbatim (no chat template), and the engine's generate() calls
        slot_store.note_prompt(), so /slots reflects the probe and the slot can be saved. "content" is
        the raw continuation, reasoning block included.
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
            room = reply_room(engine, engine.count_tokens(prompt))
            if room:
                n_predict = min(n_predict, room)
            stops = body.get("stop", [])
            job = {"_prompt": prompt, "max_tokens": n_predict,
                   "temperature": float(body.get("temperature", SERVE_DEFAULTS["temperature"])),
                   "top_p": float(body.get("top_p", SERVE_DEFAULTS["top_p"])),
                   "_stop": [stops] if isinstance(stops, str) else list(stops),
                   "_reset_gate": False}
        except (json.JSONDecodeError, web.HTTPException, TypeError, ValueError) as exc:
            if isinstance(exc, web.HTTPException):
                raise
            raise web.HTTPBadRequest(reason=str(exc)) from exc

        out: asyncio.Queue = asyncio.Queue()
        request_stats: dict[str, Any] = {}
        cancel = job["_cancel"] = asyncio.Event()
        await queue.put((job, out, request_stats))
        text = ""
        try:
            async for item in drain(request, cancel, out):
                text += item
        except ConnectionResetError:
            return client_left(cancel, "completion")
        except asyncio.CancelledError:
            cancel.set()
            raise
        except Exception:
            cancel.set()
            raise
        text, stopped = stop_text(text, job["_stop"])
        st = request_stats
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

    async def metrics_view(_: web.Request) -> web.Response:
        # used_tokens = pages referenced by active jobs: in-flight tokens, retained cache excluded.
        # get_cache_stats is memoized on the page tables; the except covers fake engines and a
        # page-table mutation in the worker thread mid-walk.
        gen = getattr(engine, "generator", None) or getattr(engine, "greedy_generator", None)
        try:
            cs = gen.get_cache_stats()
            kv = cs["used_tokens"] / cs["max_tokens"] if cs.get("max_tokens") else 0.0
        except Exception:                                        # noqa: BLE001
            kv = 0.0
        return web.Response(body=app["metrics"].render(
            processing=processing, deferred=queue.qsize(), kv_ratio=kv),
            headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"})

    if getattr(engine, "slot_store", None) is not None:
        app.router.add_get("/slots", slots)
        app.router.add_post("/slots/{id}", slot_action)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics_view)
    app.router.add_post("/v1/chat/completions", completions)
    app.router.add_post("/apply-template", apply_template)
    app.router.add_post("/completion", completion)
    app.on_startup.append(start)
    app.on_cleanup.append(close)
    return app


# ---------------------------------------------------------------------------- main


def config_max_position(config) -> int:
    """Max context from the EXL3 config, handling the nested text_config of MiMo/GLM."""
    cd = getattr(config, "config_dict", {}) or {}
    for value in (cd.get("max_position_embeddings"),
                  (cd.get("text_config") or {}).get("max_position_embeddings")):
        if value:
            return int(value)
    return 32768


def main() -> None:
    import torch
    from exllamav3 import Config

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-uncensor", action="store_true", help="ignore a bundled uncensor_spec.json in the model directory (same as EXL3_ABLIT_RUNTIME=off)")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--drafter", default=None,
                        help="drafter directory (default: $MIMO_DRAFTER, <model>/drafter, then <model>-drafter)")
    parser.add_argument("--ctx", "-c", type=int, default=0,
                        help="KV cache size in tokens; 0 = min(config max_position, 131072)")
    parser.add_argument("--ndt", type=int, default=DEFAULT_NDT)
    dflash = parser.add_mutually_exclusive_group()
    dflash.add_argument("--dflash", dest="dflash", action="store_true", default=True)
    dflash.add_argument("--no-dflash", dest="dflash", action="store_false")
    parser.add_argument("--spec-gate", dest="spec_gate", action="store_true", default=False,
                        help="cost-model gate that falls back to plain decode (off by default: with the 4 bpw drafter it costs more than it saves)")
    parser.add_argument("--no-spec-gate", dest="spec_gate", action="store_false")
    parser.add_argument("--dynamic-draft", action=argparse.BooleanOptionalAction, default=True,
                        help="confidence-truncated draft length: verify only the confident prefix (default on)")
    parser.add_argument("--draft-confidence", type=float, default=0.6)
    parser.add_argument("--spec-gate-reset-per-request", action="store_true",
                        help="rebuild the SpecGate for every request (default: keep it)")
    parser.add_argument("--slot-save-path", default="~/cache/llama-slots",
                        help="directory for /slots save/restore files (llama.cpp compatible)")
    parser.add_argument("--default-temperature", type=float, default=0.0, help="used when a request omits temperature")
    parser.add_argument("--default-top-p", type=float, default=1.0, help="used when a request omits top_p")
    args = parser.parse_args()
    if args.no_uncensor:
        os.environ["EXL3_ABLIT_RUNTIME"] = "off"
    SERVE_DEFAULTS.update(temperature=args.default_temperature, top_p=args.default_top_p)

    torch.set_grad_enabled(False)
    drafter = None
    if args.dflash:
        model_dir = Path(args.model).expanduser()
        cands = [c for c in (args.drafter, os.environ.get("MIMO_DRAFTER"), model_dir / "drafter",
                             model_dir.parent / (model_dir.name + "-drafter")) if c]
        # an explicit --drafter or $MIMO_DRAFTER wins and is not silently replaced by a default
        explicit = cands[0] if (args.drafter or os.environ.get("MIMO_DRAFTER")) else None
        for c in ([explicit] if explicit else cands):
            if (Path(c).expanduser() / "config.json").is_file():
                drafter = str(Path(c).expanduser())
                break
        if drafter is None:
            print(f"msrv: no drafter found ({explicit or 'looked in <model>/drafter and <model>-drafter'}); "
                  "decoding without speculation", flush=True)

    # --ctx 0 = the model's own max context, with the nested text_config handled like the GLM fix.
    probe = Config.from_directory(args.model)
    ctx = args.ctx or min(config_max_position(probe), 131072)
    engine = ResidentEngine(args.model, drafter, ndt=args.ndt, spec_gate=args.spec_gate, ctx=ctx,
                            reset_gate_per_request=args.spec_gate_reset_per_request,
                            dynamic_draft=args.dynamic_draft, draft_confidence=args.draft_confidence)
    from exllamav3.generator.slot_store import SlotStore
    engine.slot_store = SlotStore(engine.generator, [c for c in (engine.cache, engine.draft_cache) if c is not None],
                                  args.slot_save_path, args.model_id)
    template = (Path(args.model) / "chat_template.jinja").read_text(encoding="utf-8")
    print(f"msrv: model={args.model} drafter={drafter} ndt={args.ndt} "
          f"spec_gate={engine.spec_gate_on} ctx={ctx} "
          f"max_position={config_max_position(engine.config)}", flush=True)
    web.run_app(create_app(engine, args.model_id, template), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
