#!/usr/bin/env python3
"""OpenAI-compatible server for DeepSeek-V4.1-Flash-1017 (text in, text and tool calls out).

The HTTP layer is the one of tools/glm/serve.py; the model is behind the small contract of ds41_engine.py, the
prompt is rendered by ds41_chat.py (port of the official encoding) and the reply is split by ds41_parse.py.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import web

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.append(_p)
import startup_health  # noqa: E402  (GET /health with load progress while the model loads)
import serve_compat  # noqa: E402  (vLLM-style token-id prompts, cache_salt, /tokenize, usage details)
from serve_metrics import Metrics  # noqa: E402  (symlink to the shared tools/strix_halo/serve_metrics.py)
from serve_reqlog import Lifecycle, ReqLog, StreamWriter  # noqa: E402  (one line per request event; off by default)
from serve_ping import ServePing  # noqa: E402  (SSE keep-alive comments; off by default)
import ds41_chat  # noqa: E402
import ds41_engine  # noqa: E402
import ds41_parse  # noqa: E402

DEFAULT_MODEL = "~/models/deepseek-v4.1-flash-1017"
# What the model card recommends. Used when the pack has no generation_config.json (the Hugging Face repo of the
# model ships none); a generation_config.json in the pack folder wins over these, command-line flags win over both.
MODEL_CARD_SAMPLING = {"temperature": 1.0, "top_p": 1.0}
_UNKNOWN_SALT = object()   # cache namespace after a failed reset: the next request resets again


@dataclass
class Defaults:
    """Server-side values for fields a request leaves out."""
    temperature: float = MODEL_CARD_SAMPLING["temperature"]
    top_p: float = MODEL_CARD_SAMPLING["top_p"]
    max_tokens: int | None = None            # None: all the free context
    thinking: bool = True
    reasoning_effort: Any = ds41_chat.DEFAULT_EFFORT


def pack_sampling(model_dir: str) -> dict[str, float]:
    """temperature / top_p of the pack's generation_config.json, when it has them."""
    gc = Path(model_dir).expanduser() / "generation_config.json"
    if not gc.is_file():
        return {}
    cfg = json.loads(gc.read_text())
    return {k: float(cfg[k]) for k in ("temperature", "top_p") if cfg.get(k) is not None}


def effort_arg(text: str) -> Any:
    """--default-reasoning-effort: a level name or an integer 1..100."""
    try:
        value: Any = int(text)
    except ValueError:
        value = text
    try:
        ds41_chat.resolve_effort(value)
    except ds41_chat.ChatFormatError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def stop_text(text: str, stop: list[str]) -> tuple[str, str | None]:
    """Cut at the earliest stop string and return content and matched stop."""
    positions = [(text.find(s), s) for s in stop if s and text.find(s) >= 0]
    if not positions:
        return text, None
    pos, match = min(positions)
    return text[:pos], match


def token_limit(body: dict[str, Any], default: int | None = None) -> int:
    """Reply token budget: max_completion_tokens (newer OpenAI field) wins over max_tokens."""
    for key in ("max_completion_tokens", "max_tokens"):
        value = body.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return min(value, body.get("_room") or value)
    default = default or body.get("_room") or 32768   # unset: all the free context
    return min(default, body.get("_room") or default)


def reply_room(engine: Any, prompt_tokens: int) -> int | None:
    """Tokens a reply may still use: server context minus prompt and one slot. None when the engine reports no
    context (stub). A prompt with no room is refused."""
    ctx = getattr(engine, "ctx", None)
    if not ctx:
        return None
    room = ctx - prompt_tokens - 1 - getattr(engine, "num_draft", 0)
    if room < 1:
        raise web.HTTPBadRequest(reason=f"context length exceeded: prompt is {prompt_tokens} tokens, the server context is {ctx}")
    return room


def draft_timings(st: dict[str, Any]) -> dict[str, Any]:
    """Speculative-decoding fields for `timings`, only when drafting ran. Names are the engine stats keys of
    tools/glm/serve.py (`accepted_draft_tokens`, `rejected_draft_tokens`) plus the rounds and the per-round mean.
    Cost fields, when the engine reports them: `draft_len_mean` (drafts sent to verify per round), `verify_positions`
    (positions the target ran to verify), `conf_dropped_tokens` (drafts the confidence stop kept out of verify, each one
    a verify position saved), `conf_cut_rounds`, `tuner_k` (last length the auto tuner chose; 0 when not tuning)."""
    accepted, rejected = int(st.get("accepted_draft_tokens", 0)), int(st.get("rejected_draft_tokens", 0))
    dropped = int(st.get("conf_dropped_tokens") or 0)
    if accepted + rejected + dropped <= 0:
        return {}
    steps = int(st.get("draft_steps") or 0)
    per_step = float(st["accepted_per_step"]) if st.get("accepted_per_step") is not None else (accepted / steps if steps else 0.0)
    out = {"accepted_draft_tokens": accepted, "rejected_draft_tokens": rejected, "draft_steps": steps,
           "accepted_per_step": round(per_step, 4)}
    if st.get("draft_len_mean") is not None:
        out["draft_len_mean"] = round(float(st["draft_len_mean"]), 4)
    for key in ("verify_positions", "conf_dropped_tokens", "conf_cut_rounds"):
        if st.get(key) is not None:
            out[key] = int(st[key])
    if st.get("tuner_k"):
        out["draft_tuner_k"] = int(st["tuner_k"])
    return out


def sse(data: dict[str, Any], event: str | None = None) -> bytes:
    prefix = f"event: {event}\n" if event else ""
    return (prefix + f"data: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n").encode()


def _n(ids: Any) -> int:
    return int(ids.numel()) if hasattr(ids, "numel") else len(ids)


# Every route of create_app(), for the early /health listener (404 / 405 while loading); a test keeps it in step.
ROUTES = {"/health": ("GET",), "/v1/models": ("GET",), "/metrics": ("GET",), "/v1/chat/completions": ("POST",),
          "/apply-template": ("POST",), "/tokenize": ("POST",), "/completion": ("POST",), "/v1/completions": ("POST",),
          "/v1/ds41/score": ("POST",)}


def _memlog(prompt_tokens: int, new_tokens: int) -> None:
    """DS41_MEMLOG=<csv>: one row per finished request (host and torch memory), to chase memory growth."""
    path = os.environ.get("DS41_MEMLOG")
    if not path:
        return
    import glob
    import time
    import torch
    mi = {k: int(v.split()[0]) // 1024 for k, v in (ln.split(":") for ln in open("/proc/meminfo")) if k in ("MemAvailable", "SwapFree")}
    rss = int([ln for ln in open("/proc/self/status") if ln.startswith("VmRSS")][0].split()[1]) // 1024
    gtt = sum(int(open(f).read()) for f in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used")) // 2**20
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("t,prompt,new,mem_avail_mib,swap_free_mib,rss_mib,gtt_mib,torch_alloc_mib,torch_reserved_mib,torch_peak_reserved_mib\n")
        f.write(f"{time.time():.0f},{prompt_tokens},{new_tokens},{mi['MemAvailable']},{mi['SwapFree']},{rss},{gtt},"
                f"{torch.cuda.memory_allocated() >> 20},{torch.cuda.memory_reserved() >> 20},{torch.cuda.max_memory_reserved() >> 20}\n")
    torch.cuda.reset_peak_memory_stats()


def create_app(engine: Any, model_id: str, defaults: Defaults | None = None) -> web.Application:
    """Build the HTTP layer around an engine (the real one, or ds41_engine.StubEngine in tests)."""
    defaults = defaults or Defaults()
    queue: asyncio.Queue[tuple[dict[str, Any], asyncio.Queue, dict[str, Any]]] = asyncio.Queue()
    lock = asyncio.Lock()
    sessions = int(getattr(engine, "sessions", 1))
    gate = asyncio.Semaphore(sessions)   # sessions > 1: one permit per running request
    slot_lock = asyncio.Lock()
    salt_state: dict[str, Any] = {"current": None}   # cache_salt namespace of the prompt cache (None = requests without a salt)
    admit_lock = asyncio.Lock()
    processing = 0
    running = 0
    app = web.Application(client_max_size=64 * 1024**2, middlewares=[startup_health.json_errors_middleware()])
    app.update(engine=engine, model_id=model_id, defaults=defaults, queue=queue, metrics=Metrics())

    reqlog = ReqLog.from_env(server="ds41")
    app.update(reqlog=reqlog)
    background: dict[str, Any] = {}
    ping = ServePing.from_env(reqlog)
    app.update(ping=ping)
    lifecycle = Lifecycle(reqlog, ping=ping, counters=lambda: {"processing": processing, "running": running})

    def reqid() -> str | None:
        return lifecycle.reqid()

    def salt_of(body: dict[str, Any]) -> str | None:
        try:
            salt = serve_compat.cache_salt(body)
        except serve_compat.CompatError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        if salt is not None and not hasattr(engine, "reset_prefix_cache"):
            raise web.HTTPBadRequest(reason="cache_salt is not supported by this engine")
        return salt

    def sampling(body: dict[str, Any]) -> None:
        """Fill temperature / top_p from the server defaults when the request leaves them out (or sends null)."""
        for key in ("temperature", "top_p"):
            if body.get(key) is None:
                body[key] = getattr(defaults, key)
            elif isinstance(body[key], bool) or not isinstance(body[key], (int, float)):
                raise web.HTTPBadRequest(reason=f"{key} must be a number")

    def render_request(body: dict[str, Any]) -> tuple[str, bool]:
        """(prompt, parse_tools) of a chat request: the same call for /v1/chat/completions, /apply-template and /tokenize."""
        tools = body.get("tools")
        choice = body.get("tool_choice")
        parse_tools = choice != "none"
        if not parse_tools:
            tools = None                  # tool_choice "none": the model is not told about the tools
        try:
            opts = ds41_chat.chat_options(body, defaults.thinking, defaults.reasoning_effort)
            kw = dict(thinking=opts.thinking, reasoning_effort=opts.effort, drop_thinking=opts.drop_thinking)
            if getattr(engine, "supports_vision", False):
                # image parts stay in the prompt as one placeholder each; their records go to body["_image_records"]
                prompt, body["_image_records"] = ds41_chat.render_chat_vl(body["messages"], tools, **kw)
            else:
                prompt = ds41_chat.render_chat(body["messages"], tools, **kw)
        except ds41_chat.ChatFormatError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        return prompt, parse_tools

    async def prepare_images(body: dict[str, Any]) -> list[Any]:
        """Decode the images of a rendered request off the event loop (a URL fetch and a resize can take seconds)."""
        records = body.pop("_image_records", None) or []
        if not records:
            return []
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, engine.prepare_images, records)
        except ValueError as exc:                    # vision.ImageError is a ValueError
            raise web.HTTPBadRequest(reason=str(exc)) from exc

    def prompt_ids_of(prompt: str, images: list[Any]) -> list[int]:
        try:
            return list(engine.prompt_ids(prompt, images)) if images else list(engine.prompt_ids(prompt))
        except ValueError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc

    def observe(st: dict[str, Any], prompt_tokens: int) -> None:
        _memlog(prompt_tokens, int(st.get("new_tokens", 0)))
        num_draft = int(getattr(getattr(engine, "greedy_generator", None), "num_draft_tokens", 0) or 0)
        accepted = int(st.get("accepted_draft_tokens", 0))
        rejected = int(st.get("rejected_draft_tokens", 0))
        proposed = accepted + rejected
        app["metrics"].observe(
            prompt_tokens=prompt_tokens, cached=int(st.get("cached_tokens", 0)),
            predicted=int(st.get("new_tokens", 0)),
            prefill_s=float(st.get("time_prefill") or 0.0), generate_s=float(st.get("time_generate") or 0.0),
            drafts=int(st.get("draft_steps") or 0) or (proposed // num_draft if proposed and num_draft else 0),
            draft_tokens=proposed, accepted=accepted)

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "source": "kyojin", "model": model_id,
                                  "vision": bool(getattr(engine, "supports_vision", False)),
                                  **startup_health.compiler_health()})

    async def models(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "local"}]})

    @contextlib.asynccontextmanager
    async def admit(salt: str | None):
        """Permission to run one request. Every running request shares one cache_salt namespace: when the salt changes,
        the requests in flight finish, the prompt cache is dropped (engine.reset_prefix_cache) and the new request starts
        alone. The same salt (and traffic with no salt at all) never drops anything."""
        async def reset() -> None:
            salt_state["current"] = _UNKNOWN_SALT            # a failed reset must not leave the old namespace claimed
            await engine.reset_prefix_cache()
            salt_state["current"] = salt

        if sessions == 1:
            async with lock:
                if salt != salt_state["current"]:
                    await reset()
                yield
            return
        async with admit_lock:
            if salt == salt_state["current"]:
                await gate.acquire()
            else:
                async with slot_lock:
                    held = 0
                    try:
                        for _ in range(sessions):
                            await gate.acquire()
                            held += 1
                        await reset()
                    except BaseException:
                        for _ in range(held):
                            gate.release()
                        raise
                    for _ in range(held - 1):                # keep the permit of this request
                        gate.release()
        try:
            yield
        finally:
            gate.release()

    async def worker() -> None:
        nonlocal processing, running
        while True:
            body, out, request_stats = await queue.get()
            cancel = body.get("_cancel")
            rid = body.get("_reqid")
            if cancel is not None and cancel.is_set():   # the client left while the job was queued: no prefill, no decode
                lifecycle.job_skipped(rid, time.perf_counter() - (body.get("_queued_at") or time.perf_counter()))
                out.put_nowait(None)
                queue.task_done()
                continue
            processing += 1
            started = time.perf_counter()
            first_token = False
            try:
                async with admit(body.get("_cache_salt")):
                    running += 1
                    lifecycle.job_started(rid, started - (body.get("_queued_at") or started),
                                          prompt_tokens=body.get("_prompt_tokens"),
                                          images=len(body.get("_images") or ()), messages=body.get("_messages"), sessions=sessions,
                                          processing=processing, running=running)
                    if sessions == 1:
                        engine.last_stats = {}  # a cancelled or failed job must not re-count the previous one
                    try:
                        async for delta in engine.generate(
                            body["_prompt"],
                            max_tokens=token_limit(body, defaults.max_tokens), temperature=body.get("temperature", 0.0),
                            top_p=body.get("top_p", 1.0), stop=body.get("_stop", []),
                            cancel=body.get("_cancel"),
                            **({"prompt_ids": body["_prompt_ids"]} if body.get("_prompt_ids") is not None else {}),
                            **({"images": body["_images"]} if body.get("_images") else {}),
                            **({"ignore_eos": True} if body.get("_ignore_eos") else {}),
                            **({"cache_prompt": False} if body.get("cache_prompt") is False else {}),
                            **({"stats": request_stats} if sessions > 1 else {})):
                            if not first_token:
                                first_token = True
                                lifecycle.first_token(rid, started, prompt_tokens=body.get("_prompt_tokens"))
                            out.put_nowait(delta)
                    finally:
                        running -= 1
                    if sessions == 1:
                        request_stats.update(getattr(engine, "last_stats", None) or {})
                    if request_stats:
                        observe(request_stats, body.get("_prompt_tokens") or int(request_stats.get("prompt_tokens", 0)))
                    lifecycle.job_done(rid, request_stats, started,
                                       prompt_tokens=int(request_stats.get("prompt_tokens") or body.get("_prompt_tokens") or 0))
                    out.put_nowait(None)
            except Exception as exc:                          # noqa: BLE001
                lifecycle.job_error(rid, exc, started)
                out.put_nowait(exc)
            finally:
                processing -= 1
                queue.task_done()

    async def start(_: web.Application) -> None:
        app["worker_tasks"] = [asyncio.create_task(worker()) for _ in range(sessions)]
        app["worker_task"] = app["worker_tasks"][0]
        if reqlog.enabled and reqlog.heartbeat_s > 0:
            background["heartbeat"] = asyncio.create_task(lifecycle.heartbeat_forever())

    async def close(app_: web.Application) -> None:
        for t in app_["worker_tasks"]:
            t.cancel()
        beat = background.pop("heartbeat", None)
        if beat is not None:
            beat.cancel()
        reqlog.close()

    async def drain(request: web.Request, cancel: asyncio.Event, out: asyncio.Queue,
                    writer: StreamWriter | None = None, ping: ServePing | None = None,
                    ping_at: float | None = None) -> AsyncIterator[str]:
        """Items the worker produced for one request. Nothing is written to the client while a job is queued or
        prefilling, so a client that hung up is noticed by polling the transport. Raises ConnectionResetError and sets
        `cancel` when it has left."""
        while True:
            tr = request.transport
            if tr is None or tr.is_closing():
                cancel.set()
                if ping is not None:
                    ping.limit()
                raise ConnectionResetError("client left")
            if ping is not None:
                try:
                    await ping.comment(writer, ping_at or 0.0)
                except ConnectionResetError:
                    cancel.set()
                    raise
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
        lifecycle.client_left(path)
        print(f"serve: client left {path}, job cancelled", file=sys.stderr, flush=True)
        return web.Response(status=499)

    def tool_delta(ev: tuple) -> dict[str, Any]:
        if ev[0] == "tool_start":
            call: dict[str, Any] = {"index": ev[1], "id": ev[2], "type": "function",
                                    "function": {"name": ev[3], "arguments": ""}}
            if ev[4] is not None:
                call["namespace"] = ev[4]
            return {"tool_calls": [call]}
        return {"tool_calls": [{"index": ev[1], "function": {"arguments": ev[2]}}]}

    async def completions(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise web.HTTPBadRequest(reason="body must be a JSON object")
            if body.get("model", model_id) != model_id:
                raise web.HTTPBadRequest(reason=f"unknown model; expected {model_id}")
            if not isinstance(body.get("messages"), list) or not body["messages"]:
                raise web.HTTPBadRequest(reason="messages must be a non-empty array")
            if "stream" in body and not isinstance(body["stream"], bool):
                raise web.HTTPBadRequest(reason="stream must be boolean")
            if body.get("n") not in (None, 1):
                raise web.HTTPBadRequest(reason="n > 1 is not supported")
            for key in ("max_tokens", "max_completion_tokens"):
                v = body.get(key)
                if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
                    raise web.HTTPBadRequest(reason=f"{key} must be a positive integer")
            stops = body.get("stop") or []
            body["_stop"] = [stops] if isinstance(stops, str) else list(stops)
            if not all(isinstance(x, str) for x in body["_stop"]):
                raise web.HTTPBadRequest(reason="stop must be a string or a list of strings")
            prompt, parse_tools = render_request(body)
            images = await prepare_images(body)
            body["_images"] = images
            if images:
                body["_prompt_ids"] = prompt_ids_of(prompt, images)      # image placeholders expanded to their spans
                prompt_tokens = len(body["_prompt_ids"])
            else:
                prompt_tokens = _n(engine.prompt_ids(prompt))
            body["_prompt_tokens"] = prompt_tokens
            body["_room"] = reply_room(engine, prompt_tokens)
            body["_prompt"] = prompt
            body["_cache_salt"] = salt_of(body)
            sampling(body)
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
        request_stats: dict[str, Any] = {}
        cancel = body["_cancel"] = asyncio.Event()
        body["_reqid"] = reqid()
        body["_messages"] = len(body["messages"])
        body["_queued_at"] = time.perf_counter()
        ahead = queue.qsize()
        await queue.put((body, out, request_stats))
        lifecycle.queued(body["_reqid"], "/v1/chat/completions", queued_ahead=ahead,
                         prompt_tokens=prompt_tokens, messages=body["_messages"], images=len(images),
                         stream=bool(body.get("stream")))
        # The generation prompt ends in <think> (thinking) or </think> (chat): that is where the reply starts.
        parser = ds41_parse.StreamParser(thinking=prompt.endswith(ds41_chat.THINK_OPEN), parse_tools=parse_tools)
        stream_ping = app["ping"].per_request() if body.get("stream", False) else None
        ping_at = body["_queued_at"]
        hold = max([len(x) for x in body["_stop"]] + [1]) - 1       # characters kept back so a stop string is never sent

        def deltas(writer: StreamWriter | None = None) -> AsyncIterator[str]:
            return drain(request, cancel, out, writer, stream_ping, ping_at)

        def completion_tokens(raw: str) -> int:
            return int(request_stats.get("new_tokens") or engine.count_tokens(raw))

        def usage(raw: str) -> dict[str, int]:
            return serve_compat.usage(prompt_tokens, completion_tokens(raw), int(request_stats.get("cached_tokens") or 0))

        def finish_reason(raw: str, has_calls: bool, stopped: str | None) -> str:
            if parser.truncated:
                return "length"                          # a tool call cut by the token budget is not a finished call
            if has_calls:
                return "tool_calls"
            if stopped is None and (request_stats.get("eos_reason") == "max_new_tokens"
                                    or engine.count_tokens(raw) >= token_limit(body, defaults.max_tokens)):
                return "length"                          # clients use it to tell a truncated answer
            return "stop"

        def timings() -> dict[str, Any]:
            # llama.cpp names: the swap orchestrator reads cache_n as the verdict of a KV restore
            st = request_stats
            cache_n = int(st.get("cached_tokens", 0))
            pre, gen = st.get("time_prefill") or 0.0, st.get("time_generate") or 0.0
            n = int(st.get("new_tokens", 0))
            return {"cache_n": cache_n, "prompt_n": max(prompt_tokens - cache_n, 0),
                    "prompt_ms": pre * 1000, "predicted_n": n, "predicted_ms": gen * 1000,
                    "prompt_per_second": (prompt_tokens - cache_n) / pre if pre else 0.0,
                    "predicted_per_second": n / gen if gen else 0.0, **draft_timings(st)}

        if body.get("stream", False):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await response.prepare(request)
            writer = lifecycle.stream_started(body["_reqid"], "/v1/chat/completions", response, prompt_tokens=prompt_tokens)
            seen_output: list[float] = []
            calls_seen = 0

            async def send(events: list[tuple]) -> None:
                nonlocal calls_seen
                for ev in events:
                    if ev[0] in ("reasoning", "content"):
                        if not seen_output:
                            seen_output.append(time.perf_counter())
                            lifecycle.first_content(body["_reqid"], body["_queued_at"])
                        delta = {"reasoning_content" if ev[0] == "reasoning" else "content": ev[1]}
                    elif ev[0] == "tool_end":
                        continue
                    else:
                        calls_seen += ev[0] == "tool_start"
                        delta = tool_delta(ev)
                    await writer.write(sse(event(delta)))
                    stream_ping.note_output()

            try:
                await writer.write(sse(event({"role": "assistant", "content": ""})))
                stream_ping.note_output()
                raw, fed, stopped = "", 0, None
                async for delta in deltas(writer):
                    if stopped is not None:
                        continue                             # keep draining: the engine ends on its own stop condition
                    raw += delta
                    cut, stopped = stop_text(raw, body["_stop"])
                    if stopped is not None:
                        raw = cut
                    safe = len(raw) if stopped is not None else max(fed, len(raw) - hold)
                    if safe > fed:
                        await send(parser.feed(raw[fed:safe]))
                        fed = safe
                await send(parser.feed(raw[fed:]) + parser.finish())
                await writer.write(sse(event({}, finish_reason(raw, calls_seen > 0, stopped)) |
                                       {"usage": usage(raw), "timings": timings()}))
                await writer.write(b"data: [DONE]\n\n")
                await response.write_eof()
                lifecycle.stream_done(body["_reqid"], int(request_stats.get("new_tokens") or 0))
            except ConnectionResetError:
                client_left(cancel, "stream")
                return response
            except asyncio.CancelledError:
                cancel.set()
                raise
            except Exception as exc:                          # noqa: BLE001
                cancel.set()
                lifecycle.stream_error(body["_reqid"], exc)
                # Fail inside the stream, but still end it the OpenAI way: clients report "stream ended without
                # finish_reason" and retry when the connection just drops.
                try:
                    await writer.write(sse({"error": {"message": f"{type(exc).__name__}: {exc}"}}))
                    await writer.write(sse(event({}, "error")))
                    await writer.write(b"data: [DONE]\n\n")
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
        raw, stopped = stop_text(raw, body["_stop"])
        parsed = ds41_parse.split_events(parser.feed(raw) + parser.finish())
        message: dict[str, Any] = {"role": "assistant", "content": parsed["content"]}
        if parsed["reasoning_content"]:
            message["reasoning_content"] = parsed["reasoning_content"]
        if parsed["tool_calls"]:
            message["tool_calls"] = parsed["tool_calls"]
        finish = finish_reason(raw, bool(parsed["tool_calls"]), stopped)
        return web.json_response({
            "id": request_id, "object": "chat.completion", "created": created, "model": model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage(raw), "timings": timings()})

    async def apply_template(request: web.Request) -> web.Response:
        """llama.cpp's POST /apply-template: render the chat prompt, generate nothing. The same render_request() call
        as /v1/chat/completions, so what the conformance tool reads here is byte for byte what a request is sent."""
        try:
            body = await request.json()
            if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
                raise web.HTTPBadRequest(reason="messages must be a non-empty array")
            prompt, _ = render_request(body)
        except (json.JSONDecodeError, web.HTTPException, TypeError) as exc:
            if isinstance(exc, web.HTTPException):
                raise
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        return web.json_response({"prompt": prompt})

    async def completion(request: web.Request) -> web.Response:
        """Minimal llama.cpp POST /completion over the same queue as /v1/chat/completions. The prompt is used
        verbatim (no chat rendering); "content" is the raw continuation, reasoning block included."""
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
                   "temperature": float(defaults.temperature if body.get("temperature") is None else body["temperature"]),
                   "top_p": float(defaults.top_p if body.get("top_p") is None else body["top_p"]),
                   "_stop": [stops] if isinstance(stops, str) else list(stops),
                   "cache_prompt": body.get("cache_prompt", True), "_cache_salt": salt_of(body)}
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
            "timings": {"prompt_n": int(st.get("cached_tokens", 0)), "predicted_n": int(st.get("new_tokens", 0)),
                        **draft_timings(st)}})

    async def tokenize(request: web.Request) -> web.Response:
        """vLLM POST /tokenize: {prompt | messages, add_special_tokens?} -> {tokens, count, max_model_len}.

        With messages the chat prompt is rendered (BOS included) and tokenized: `count` is the prompt length a chat
        request with the same body would report."""
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise web.HTTPBadRequest(reason="body must be a JSON object")
            if body.get("model", model_id) != model_id:
                raise web.HTTPBadRequest(reason=f"unknown model; expected {model_id}")
            if not hasattr(engine, "encode_ids"):
                raise web.HTTPNotImplemented(reason="this engine cannot tokenize")
            add_special = body.get("add_special_tokens")
            if add_special is not None and not isinstance(add_special, bool):
                raise web.HTTPBadRequest(reason="add_special_tokens must be boolean")
            if body.get("messages") is not None:
                if not isinstance(body["messages"], list) or not body["messages"]:
                    raise web.HTTPBadRequest(reason="messages must be a non-empty array")
                text, _ = render_request(body)
                add_special = bool(add_special)          # a rendered chat prompt carries its own BOS
                images = await prepare_images(body)
                if images:
                    return web.json_response(serve_compat.tokenize_reply(
                        prompt_ids_of(text, images), getattr(engine, "ctx", None)))
            else:
                text = body.get("prompt")
                if not isinstance(text, str):
                    raise web.HTTPBadRequest(reason="prompt must be a string (or send messages)")
                add_special = True if add_special is None else add_special
            ids = engine.encode_ids(text, add_special)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        return web.json_response(serve_compat.tokenize_reply(ids, getattr(engine, "ctx", None)))

    async def v1_completions(request: web.Request) -> web.StreamResponse:
        """OpenAI POST /v1/completions over the same queue as chat: the prompt (text, or a list of token ids) is used
        verbatim (no chat rendering, no BOS added), one prompt, one choice. Streamed with a final `usage`."""
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise web.HTTPBadRequest(reason="body must be a JSON object")
            if body.get("model", model_id) != model_id:
                raise web.HTTPBadRequest(reason=f"unknown model; expected {model_id}")
            prompt = body.get("prompt")
            try:
                prompt_ids = serve_compat.token_prompt(prompt, getattr(engine, "vocab_size", None))
            except serve_compat.CompatError as exc:
                raise web.HTTPBadRequest(reason=str(exc)) from exc
            if prompt_ids is not None:
                prompt = ""
            else:
                if isinstance(prompt, list) and len(prompt) == 1:
                    prompt = prompt[0]
                if not isinstance(prompt, str) or not prompt:
                    raise web.HTTPBadRequest(reason="prompt must be a non-empty string or a list of token ids (one prompt per request)")
            if body.get("n") not in (None, 1) or body.get("echo") or body.get("best_of") not in (None, 1):
                raise web.HTTPBadRequest(reason="n, best_of and echo are not supported")
            for key in ("stream", "ignore_eos"):
                if body.get(key) is not None and not isinstance(body[key], bool):
                    raise web.HTTPBadRequest(reason=f"{key} must be boolean")
            max_tokens = 16 if body.get("max_tokens") is None else body["max_tokens"]
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
                raise web.HTTPBadRequest(reason="max_tokens must be a positive integer")
            stops = body.get("stop") or []
            stops = [stops] if isinstance(stops, str) else stops
            if not isinstance(stops, list) or not all(isinstance(x, str) for x in stops):
                raise web.HTTPBadRequest(reason="stop must be a string or a list of strings")
            prompt_tokens = len(prompt_ids) if prompt_ids is not None else engine.count_tokens(prompt)
            job = {"_prompt": prompt, "_prompt_ids": prompt_ids, "_ignore_eos": body.get("ignore_eos") is True,
                   "max_tokens": max_tokens, "_room": reply_room(engine, prompt_tokens), "_stop": list(stops),
                   "temperature": float(defaults.temperature if body.get("temperature") is None else body["temperature"]),
                   "top_p": float(defaults.top_p if body.get("top_p") is None else body["top_p"]),
                   "cache_prompt": body.get("cache_prompt", True), "_cache_salt": salt_of(body)}
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc

        limit = token_limit(job)
        request_id, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())
        out: asyncio.Queue = asyncio.Queue()
        request_stats: dict[str, Any] = {}
        cancel = job["_cancel"] = asyncio.Event()
        job["_reqid"] = reqid()
        job["_queued_at"] = time.perf_counter()
        ahead = queue.qsize()
        await queue.put((job, out, request_stats))
        lifecycle.queued(job["_reqid"], "/v1/completions", queued_ahead=ahead,
                         prompt_tokens=prompt_tokens, images=0, stream=bool(body.get("stream")))

        def chunk(text: str, finish: str | None) -> dict[str, Any]:
            return {"id": request_id, "object": "text_completion", "created": created, "model": model_id,
                    "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish}]}

        def finish_and_usage(text: str, stopped: str | None) -> tuple[str, dict[str, Any]]:
            n_new = int(request_stats.get("new_tokens") or engine.count_tokens(text))
            reason = "length" if stopped is None and (request_stats.get("eos_reason") == "max_new_tokens" or n_new >= limit) else "stop"
            return reason, serve_compat.usage(prompt_tokens, n_new, int(request_stats.get("cached_tokens") or 0))

        hold = max((len(x) for x in job["_stop"]), default=1) - 1

        if body.get("stream", False):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await response.prepare(request)
            writer = lifecycle.stream_started(job["_reqid"], "/v1/completions", response, prompt_tokens=prompt_tokens)
            ping = app["ping"].per_request()
            ping_at = job["_queued_at"]
            try:
                text, sent, stopped = "", 0, None
                async for delta in drain(request, cancel, out, writer, ping, ping_at):
                    text += delta
                    cut, stopped = stop_text(text, job["_stop"])
                    if stopped is not None:
                        cancel.set()            # a stop string ended the reply: the rest is not generated
                        text = cut
                        break
                    safe = max(sent, len(text) - hold)
                    if safe > sent:
                        if not sent:
                            lifecycle.first_content(job["_reqid"], job["_queued_at"])
                        await writer.write(sse(chunk(text[sent:safe], None)))
                        ping.note_output()
                        sent = safe
                if len(text) > sent:
                    await writer.write(sse(chunk(text[sent:], None)))
                reason, usage = finish_and_usage(text, stopped)
                await writer.write(sse(chunk("", reason) | {"usage": usage}))
                await writer.write(b"data: [DONE]\n\n")
                await response.write_eof()
                lifecycle.stream_done(job["_reqid"], int(request_stats.get("new_tokens") or 0))
            except ConnectionResetError:
                client_left(cancel, "stream")
            except asyncio.CancelledError:
                cancel.set()
                raise
            except Exception as exc:                          # noqa: BLE001
                cancel.set()
                lifecycle.stream_error(job["_reqid"], exc)
                try:
                    await writer.write(sse({"error": {"message": f"{type(exc).__name__}: {exc}"}}))
                    await response.write_eof()
                except ConnectionError:
                    pass
            return response

        try:
            raw = ""
            async for delta in drain(request, cancel, out):
                raw += delta
        except ConnectionResetError:
            return client_left(cancel, "request")
        except asyncio.CancelledError:
            cancel.set()
            raise
        except Exception:
            cancel.set()
            raise
        text, stopped = stop_text(raw, job["_stop"])
        reason, usage = finish_and_usage(text, stopped)
        return web.json_response(chunk(text, reason) | {"usage": usage})

    async def metrics_view(_: web.Request) -> web.Response:
        gen = getattr(engine, "generator", None) or getattr(engine, "greedy_generator", None)
        try:
            cs = gen.get_cache_stats()
            kv = cs["used_tokens"] / cs["max_tokens"] if cs.get("max_tokens") else 0.0
        except Exception:                                        # noqa: BLE001  (stub engines, page-table mutation mid-walk)
            kv = 0.0
        return web.Response(body=app["metrics"].render(processing=processing, deferred=queue.qsize(), kv_ratio=kv),
                            headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"})

    async def score(request: web.Request) -> web.Response:
        """POST /v1/ds41/score (quality gate): teacher-forced log-probs. Body: prompt_ids (token ids, used as is) or
        messages (rendered like /v1/chat/completions), continuation_ids, candidate_ids (one id list per continuation
        token). Reply: positions[i] = {top1, top1_logp, cand_logps} of the full-vocabulary distribution that predicts
        continuation_ids[i] given the prompt and continuation_ids[:i]."""
        if not hasattr(engine, "score"):
            raise web.HTTPNotImplemented(reason="this engine cannot score")
        def ids_ok(v):
            return isinstance(v, list) and all(isinstance(t, int) and not isinstance(t, bool) and 0 <= t < engine.vocab_size for t in v)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise web.HTTPBadRequest(reason="body must be a JSON object")
            cont, cands = body.get("continuation_ids"), body.get("candidate_ids")
            if (not ids_ok(cont) or not cont or not isinstance(cands, list) or len(cands) != len(cont)
                    or not all(ids_ok(c) for c in cands)):
                raise web.HTTPBadRequest(reason="continuation_ids and candidate_ids must be equal-length id lists")
            pids = body.get("prompt_ids")
            if pids is not None:
                if not ids_ok(pids) or not pids:
                    raise web.HTTPBadRequest(reason="prompt_ids must be a non-empty id list")
                prompt = ""
            elif isinstance(body.get("messages"), list) and body["messages"]:
                prompt, _ = render_request(body)
            else:
                raise web.HTTPBadRequest(reason="send prompt_ids or messages")
            t0 = time.time()
            positions = await engine.score(prompt, cont, cands, prompt_ids=pids)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        return web.json_response({"positions": positions, "prompt_tokens": len(pids) if pids else engine.count_tokens(prompt),
                                  "time_s": round(time.time() - t0, 3)})

    app.router.add_get("/health", health)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/metrics", metrics_view)
    app.router.add_post("/v1/chat/completions", lifecycle.wrapped(completions))
    app.router.add_post("/apply-template", apply_template)
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/v1/completions", lifecycle.wrapped(v1_completions))
    app.router.add_post("/completion", completion)
    app.router.add_post("/v1/ds41/score", score)
    app.on_startup.append(start)
    app.on_cleanup.append(close)
    return app


def build_defaults(args: argparse.Namespace) -> Defaults:
    """Model-card values, then the pack's generation_config.json, then the command line."""
    d = Defaults()
    for key, value in pack_sampling(args.model).items():
        setattr(d, key, value)
    if args.default_temperature is not None:
        d.temperature = args.default_temperature
    if args.default_top_p is not None:
        d.top_p = args.default_top_p
    if args.default_max_tokens is not None:
        d.max_tokens = args.default_max_tokens
    if args.default_reasoning_effort is not None:
        d.reasoning_effort = args.default_reasoning_effort
    d.thinking = args.default_thinking == "on"
    return d


def parse_engine_opts(pairs: list[str]) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"serve: --engine-opt wants KEY=VALUE, got {pair!r}")
        try:
            opts[key] = json.loads(raw)
        except ValueError:
            opts[key] = raw
    return opts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model folder (weights, config.json, tokenizer.json)")
    parser.add_argument("--model-id", default="deepseek-v4.1-flash", help="id reported by /v1/models and echoed in replies")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("-c", "--max-ctx", type=int, default=131072, help="context window in tokens (the model supports 1M)")
    parser.add_argument("--loader", default=None,
                        help=f"module:function that builds the engine (default {ds41_engine.DEFAULT_LOADER}); "
                             "'stub' = canned replies, no GPU")
    parser.add_argument("--engine-opt", action="append", default=[], metavar="KEY=VALUE",
                        help="extra keyword argument for the loader (JSON value or plain string); repeatable")
    parser.add_argument("--num-draft", type=int, default=0, metavar="N",
                        help="DSpark speculative decoding: up to N draft tokens per round (1..10; above the DSpark block size of 5 the drafter chains passes, exact output), "
                             "passed to the loader as num_draft; 0 = off (default)")
    parser.add_argument("--draft-auto", action="store_true",
                        help="choose the draft length of each request round from the running acceptance (1..4, at most "
                             "--num-draft; --num-draft defaults to 4 with this flag). Passed as draft_auto=true. Exact outputs")
    parser.add_argument("--draft-conf", type=float, default=None, metavar="P",
                        help="stop drafting at the first slot whose DSpark confidence is below P (0 < P < 1). Passed as "
                             "draft_conf_threshold. Exact outputs; needs --num-draft or --draft-auto")
    parser.add_argument("--default-temperature", type=float, default=None,
                        help="used when a request omits temperature (default: generation_config.json of the pack, else 1.0)")
    parser.add_argument("--default-top-p", type=float, default=None,
                        help="used when a request omits top_p (default: generation_config.json of the pack, else 1.0)")
    parser.add_argument("--default-max-tokens", type=int, default=None,
                        help="reply budget when a request omits max_tokens, thinking included (default: all the free context)")
    parser.add_argument("--default-reasoning-effort", type=effort_arg, default=None,
                        help="thinking effort when a request omits it: minimal, low, medium, high, xhigh, max or an integer 1..100 (default: high)")
    parser.add_argument("--default-thinking", choices=("on", "off"), default="on",
                        help="thinking mode when a request does not choose (default: on)")
    args = parser.parse_args()

    defaults = build_defaults(args)
    engine_opts = parse_engine_opts(args.engine_opt)
    if args.num_draft < 0:
        raise SystemExit("serve: --num-draft must be >= 0")
    if args.draft_conf is not None and not 0 < args.draft_conf < 1:
        raise SystemExit("serve: --draft-conf must be between 0 and 1 (exclusive)")
    if args.draft_auto and not args.num_draft:
        args.num_draft = 4
    if args.draft_conf is not None and not args.num_draft:
        raise SystemExit("serve: --draft-conf needs --num-draft or --draft-auto")
    if args.draft_auto:
        engine_opts["draft_auto"] = True
    if args.draft_conf is not None:
        engine_opts["draft_conf_threshold"] = args.draft_conf
    if args.num_draft:
        engine_opts["num_draft"] = args.num_draft              # only when on: a loader without drafting never sees it
        print(f"serve: DSpark speculative decoding on, up to {args.num_draft} draft tokens per round", flush=True)
    if args.loader != "stub":
        startup_health.check_model_dir("serve", args.model, ("config.json", "tokenizer.json"))
    startup_health.start("ds41", ["model", "warm-up"], args.host, args.port, routes=ROUTES)
    try:
        startup_health.stage("model")
        engine = ds41_engine.load_model(str(Path(args.model).expanduser()), args.max_ctx, loader=args.loader, **engine_opts)
    except BaseException as exc:                                  # noqa: BLE001
        startup_health.abort(f"{type(exc).__name__}: {exc}")
        raise
    # Warm-up before the first request (kernel JIT, allocator growth); EXL3_SERVE_WARMUP=0 skips it. The stub has none.
    if hasattr(engine, "warmup") and os.environ.get("EXL3_SERVE_WARMUP", "1") != "0":
        startup_health.stage("warm-up")
        print(f"serve: warm-up {engine.warmup():.1f} s", flush=True)
    app = create_app(engine, args.model_id, defaults)
    reqlog = app["reqlog"]
    print(f"serve: {reqlog.describe()}", flush=True)
    print(f"serve: {app['ping'].describe()}", flush=True)
    print(f"serve: defaults temperature {defaults.temperature}, top_p {defaults.top_p}, thinking "
          f"{'on' if defaults.thinking else 'off'}, effort {defaults.reasoning_effort}", flush=True)
    reqlog.log("serve_start", model=Path(args.model).name, model_id=args.model_id, pid=os.getpid(), ctx=args.max_ctx,
               sessions=int(getattr(engine, "sessions", 1)), heartbeat_s=reqlog.heartbeat_s,
               sse_ping_s=app["ping"].interval_s or None)
    sock = startup_health.finish()                                # the early listener's open socket: no refused connection
    if sock is not None:
        web.run_app(app, sock=sock)
    else:
        web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
