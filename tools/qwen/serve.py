#!/usr/bin/env python3
"""qserve -- OpenAI-compatible server for Qwen3.8-Flash-class EXL3 packs (Yamz).

One resident target model, its MTP drafter (lossless speculative decoding, the shipped "mix" draft rule on by
default), the vision tower (optional), one Generator. The paged KV cache and the GDN recurrent checkpoints live in
the Generator, so a follow-up turn only prefills what is new.

The HTTP helpers (template render, SSE framing, stop strings) come from tools/mimo/serve.py; the engine wiring is
the one measured by tools/qwen/spec_bench.py (same Generator arguments, same draft policy).

Start:  python tools/qwen/serve.py --model <pack> --port 8000 -c 65536      (see tools/qwen/SERVE.md)
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import functools
import gc
import importlib.util
import io
import json
import math
import os
import re
import sys
import threading
import time
import urllib.request
import uuid
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path
from typing import Any

import jinja2
from aiohttp import web

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent))
import startup_health  # noqa: E402  (GET /health with load progress while the model loads)
import image_cache  # noqa: E402  (shared bounded LRU of image embeddings; symlink to tools/strix_halo)
import serve_compat  # noqa: E402  (vLLM-style token-id prompts, cache_salt, /tokenize, usage details)
from l21_prompt_encode_cache import PromptEncodeCache  # noqa: E402  (incremental prompt encode)
# serve_ping.py is a symlink to the shared tools/strix_halo/serve_ping.py: an SSE comment line every
# KYOJIN_SSE_PING_S while a stream has nothing to send yet (off by default), so a long prefill does
# not look like a hung server to a client that only watches its socket.
from serve_ping import ServePing  # noqa: E402
# serve_reqlog.py is a symlink to the shared tools/strix_halo/serve_reqlog.py: off unless
# KYOJIN_REQLOG_DIR names a directory. Its output accounting also feeds the end-of-stream summary here.
from serve_reqlog import Lifecycle, ReqLog, StreamWriter, refusal_text, sanitize  # noqa: E402


def _load_mimo():
    spec = importlib.util.spec_from_file_location("mimo_serve", HERE.parent / "mimo" / "serve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mimo = _load_mimo()
render_prompt, stop_text, sse, config_max_position = (
    _mimo.render_prompt, _mimo.stop_text, _mimo.sse, _mimo.config_max_position)

DEFAULT_MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-EXL3-Yamz")
DEFAULT_MODEL_ID = "Qwen3.8-Flash-Yamz"
THINK_CLOSE = "</think>"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
IMAGE_PAD = "<|image_pad|>"
MAX_IMAGE_BYTES = 20 * 1024**2
# Used when generation_config.json is missing (values of the Qwen3.8 card).
FALLBACK_DEFAULTS: dict[str, Any] = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
# Thinking effort when a request names none. The model's own template defines xhigh and low only;
# chat_template.jinja next to this file is that template plus a medium level.
DEFAULT_EFFORT = "medium"
_UNKNOWN_SALT = object()   # cache namespace after a failed reset: the next request resets again
NO_CTX_REPLY = 32768  # reply budget when the engine reports no context length
DEFAULT_TEMPLATE = Path(__file__).with_name("chat_template.jinja")
# OpenAI effort names -> the Qwen template names (xhigh, medium, low).
EFFORT_ALIASES = {"high": "xhigh", "minimal": "low"}



class BadRequest(Exception):
    pass


PAGE_TOKENS = 256   # the K/V cache is allocated in pages of this many tokens (exllamav3.generator.pagetable.PAGE_SIZE)


def reply_room(ctx: int, prompt_tokens: int, draft: int = 0) -> int:
    """Largest max_new_tokens the generator accepts for a prompt, or less than 1 when none fits.
    Job.prepare reserves prompt + max_new_tokens + 1 + num_draft_tokens slots (the default requeue budget: the whole reply, one
    token and one speculative window) and asks for ceil(that / 256) pages, while the page table holds ctx // 256 pages, so
    the usable cache is ctx rounded DOWN to a page and the window is part of the reserve. Anything above this value ends in
    "Job requires N pages (only N-1 available)"."""
    return ctx // PAGE_TOKENS * PAGE_TOKENS - prompt_tokens - 1 - draft


def draft_window(engine: Any, speculative: bool = True) -> int:
    """Draft tokens the generator reserves per window for a request: the engine's ndt when it speculates, else 0."""
    return int(getattr(engine, "ndt", 0)) if speculative and getattr(engine, "draft_model", None) is not None else 0


# ---------------------------------------------------------------------------- parsing of the model output


def _partial_tag(buf: str, tags: list[str]) -> int:
    """Length of the longest suffix of buf that is a proper prefix of one of the tags."""
    best = 0
    for tag in tags:
        for n in range(min(len(tag) - 1, len(buf)), best, -1):
            if tag.startswith(buf[-n:]):
                best = n
                break
    return best


class Splitter:
    """Incremental split of the model text into reasoning, content and tool-call text.

    The same object serves the streamed and the non-streamed path, so both classify text identically.
    `reasoning_first`: the generation prompt ended inside a <think> block (template default), so the text
    starts as reasoning until </think>. From <tool_call> to </tool_call> everything is held for parse_tool_calls(); text after the close is content again.
    Whitespace at the edges of a piece (the newlines around </think> and <tool_call>) is dropped.
    """

    def __init__(self, reasoning_first: bool):
        self.mode = "reasoning" if reasoning_first else "content"
        self.buf = ""
        self.ws = ""
        self.lead = reasoning_first
        self.tool_text = ""
        self.tool_from = 0
        self.said = self.after_call = False

    def _emit(self, seg: str, out: list[tuple[str, str]]) -> None:
        seg = self.ws + seg
        self.ws = ""
        if self.lead:
            seg = seg.lstrip()
            self.lead = not seg
        body = seg.rstrip()
        self.ws = seg[len(body):]
        if body:
            if self.mode == "content":
                if self.after_call and self.said:
                    body = "\n" + body          # text that follows a tool call starts on its own line
                self.said, self.after_call = True, False
            out.append((self.mode, body))

    def feed(self, text: str, final: bool = False) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        self.buf += text
        while True:
            if self.mode == "tool":
                self.tool_text += self.buf
                self.buf = ""
                end = self.tool_text.find(TOOL_CLOSE, self.tool_from)
                if end < 0:
                    return out
                # call closed: text after it (or between two calls) is content again, a later <tool_call> reopens the mode
                end += len(TOOL_CLOSE)
                self.buf, self.tool_text = self.tool_text[end:], self.tool_text[:end]
                self.mode, self.lead, self.after_call = "content", True, True
                continue
            tags = [TOOL_OPEN] + ([THINK_CLOSE] if self.mode == "reasoning" else [])
            hits = [(self.buf.find(t), t) for t in tags if t in self.buf]
            if hits:
                pos, tag = min(hits)
                self._emit(self.buf[:pos], out)
                self.ws = ""
                self.buf = self.buf[pos + len(tag):]
                if tag == TOOL_OPEN:
                    self.mode, self.tool_from = "tool", len(self.tool_text)
                    self.tool_text += TOOL_OPEN
                else:
                    self.mode, self.lead = "content", True
                continue
            keep = 0 if final else _partial_tag(self.buf, tags)
            self._emit(self.buf[:len(self.buf) - keep], out)
            self.buf = self.buf[len(self.buf) - keep:]
            return out

    def finish(self) -> list[tuple[str, str]]:
        out = self.feed("", final=True)
        self.ws = ""
        return out


_FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)(?:</function>|$)", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>\n?(.*?)\n?</parameter>", re.S)


def _coerce(value: str, schema: dict[str, Any] | None) -> Any:
    """Typed value of one <parameter>. The schema type wins: "string" keeps the text, boolean accepts true/false in any case
    (the template shows Python-style True/False), integer and number accept their text form, anything else is read as JSON."""
    kind = (schema or {}).get("type")
    kinds = [k for k in (kind if isinstance(kind, list) else [kind]) if isinstance(k, str)]
    text = value.strip()
    if kinds and "string" in kinds and not (text == "null" and "null" in kinds):
        return value
    if "boolean" in kinds and text.lower() in ("true", "false"):
        return text.lower() == "true"
    if "null" in kinds and text in ("null", "None"):
        return None
    for want, conv in (("integer", int), ("number", float)):
        if want in kinds:
            try:
                got = conv(text)
            except ValueError:
                continue
            if isinstance(got, float) and not math.isfinite(got):
                continue                       # nan / inf are not JSON: keep the text
            return got
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if schema is None and not isinstance(parsed, (dict, list)):
        return value  # no schema: keep scalars as the model wrote them
    return parsed


def parse_tool_calls(tool_text: str, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Qwen XML tool calls (<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>) to OpenAI."""
    schemas = {}
    for t in tools or []:
        fn = t.get("function", t)
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties", {})
    calls = []
    for block in re.findall(re.escape(TOOL_OPEN) + r"(.*?)" + re.escape(TOOL_CLOSE), tool_text, re.S):
        fm = _FUNC_RE.search(block)
        if fm:
            name, props = fm.group(1), schemas.get(fm.group(1), {})
            args = {k: _coerce(v, props.get(k) if props else None) for k, v in _PARAM_RE.findall(fm.group(2))}
        else:  # JSON body: {"name": ..., "arguments": {...}}
            try:
                obj = json.loads(block.strip())
                name, args = obj["name"], obj.get("arguments", obj.get("parameters", {}))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": name,
                                   "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}})
    return calls


def split_completion(text: str, reasoning_first: bool, tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Whole-text version of Splitter: an OpenAI message dict."""
    sp = Splitter(reasoning_first)
    pieces = sp.feed(text) + sp.finish()
    msg: dict[str, Any] = {"role": "assistant",
                           "content": "".join(t for k, t in pieces if k == "content")}
    reasoning = "".join(t for k, t in pieces if k == "reasoning")
    if reasoning:
        msg["reasoning_content"] = reasoning
    calls = parse_tool_calls(sp.tool_text, tools)
    if calls:
        msg["tool_calls"] = calls
    return msg


# ---------------------------------------------------------------------------- request helpers


def resolve_template_kwargs(body: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    """enable_thinking / reasoning_effort for the chat template: chat_template_kwargs > top level > server defaults."""
    kw: dict[str, Any] = {}
    ctk = body.get("chat_template_kwargs") or {}
    if not isinstance(ctk, dict):
        raise BadRequest("chat_template_kwargs must be an object")
    for key in ("enable_thinking", "reasoning_effort"):
        for src in (ctk, body, defaults):
            if src.get(key) is not None:
                kw[key] = src[key]
                break
    effort = kw.get("reasoning_effort")
    if effort == "none":
        kw.pop("reasoning_effort")
        kw["enable_thinking"] = False
    elif isinstance(effort, str):
        kw["reasoning_effort"] = EFFORT_ALIASES.get(effort, effort)
    if "enable_thinking" in kw and not isinstance(kw["enable_thinking"], bool):
        raise BadRequest("enable_thinking must be boolean")
    return kw


def extract_images(messages: list[dict[str, Any]]) -> list[str]:
    """Image URLs of the messages in prompt order (OpenAI image_url parts)."""
    urls = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if _is_image_part(part):
                    iu = part.get("image_url", part.get("image"))
                    urls.append(iu.get("url") if isinstance(iu, dict) else iu)
    return urls


def _escape_image_pad(text: str) -> str:
    """Keep a textual mention from being interpreted as an image placeholder by validation."""
    return text.replace(IMAGE_PAD, IMAGE_PAD[:1] + "\\" + IMAGE_PAD[1:])


def _escape_render_text(value: Any) -> Any:
    """Escape marker-like text in arbitrary values supplied to the chat template."""
    if isinstance(value, str):
        return _escape_image_pad(value)
    if isinstance(value, list):
        return [_escape_render_text(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_escape_render_text(item) for item in value)
    if isinstance(value, dict):
        escaped = {}
        for key, item in value.items():
            escaped_key = _escape_render_text(key) if isinstance(key, str) else key
            if escaped_key in escaped:
                raise BadRequest("sanitization collision between dictionary keys")
            escaped[escaped_key] = _escape_render_text(item)
        return escaped
    return value


def _is_image_part(part: Any) -> bool:
    """Match the image-part shape accepted by extract_images."""
    return (
        isinstance(part, dict)
        and (part.get("type") in ("image_url", "image") or "image_url" in part or "image" in part)
    )


def _sanitize_message_content(content: Any) -> Any:
    """Escape text content, preserving only image parts in the content list."""
    if isinstance(content, str):
        return _escape_image_pad(content)
    if isinstance(content, list):
        return [part if _is_image_part(part) else _escape_render_text(part) for part in content]
    return _escape_render_text(content)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        if not isinstance(m, dict) or "role" not in m:
            raise BadRequest("each message needs a role")
        content = _sanitize_message_content(m.get("content"))
        m = {key: _escape_render_text(value) for key, value in m.items() if key != "content"}
        if content is None:
            content = ""
        m["content"] = content
        if m["role"] == "developer":                                   # OpenAI's newer name for the system role
            m["role"] = "system"
        out.append(m)
    return out


def load_image_keyed(url: str):
    """(PIL image, content hash of the encoded bytes) from a data: URI or an http(s) URL (size capped)."""
    from PIL import Image
    if not isinstance(url, str) or not url:
        raise BadRequest("image_url needs a url")
    if url.startswith("data:"):
        try:
            raw = base64.b64decode(url.split(",", 1)[1])
        except Exception as exc:                                      # noqa: BLE001
            raise BadRequest(f"bad data URI: {exc}") from exc
    elif url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=20) as r:            # noqa: S310
            raw = r.read(MAX_IMAGE_BYTES + 1)
    else:
        raise BadRequest("image url must be a data: URI or http(s) URL")
    if len(raw) > MAX_IMAGE_BYTES:
        raise BadRequest("image too large")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB"), image_cache.digest(raw)
    except Exception as exc:                                          # noqa: BLE001
        raise BadRequest(f"cannot decode image: {exc}") from exc


def load_image(url: str):
    """PIL image from a data: URI or an http(s) URL (size capped)."""
    return load_image_keyed(url)[0]


def sampling_from(body: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    s = {}
    for key in ("temperature", "top_p", "top_k", "min_p"):
        v = body.get(key)
        s[key] = defaults[key] if v is None else v
        if not isinstance(s[key], (int, float)) or isinstance(s[key], bool) or not math.isfinite(s[key]):
            raise BadRequest(f"{key} must be a finite number")
    if s["temperature"] < 0 or not 0 < s["top_p"] <= 1 or s["top_k"] < 0 or not 0 <= s["min_p"] < 1:
        raise BadRequest("sampling parameter out of range")
    s["top_k"] = int(s["top_k"])
    return s


def load_defaults(model_dir: str, args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Sampling defaults = the pack's generation_config.json; command-line flags override."""
    d = dict(FALLBACK_DEFAULTS)
    gc = Path(model_dir) / "generation_config.json"
    if gc.is_file():
        cfg = json.loads(gc.read_text())
        d.update({k: cfg[k] for k in ("temperature", "top_p", "top_k") if k in cfg})
    d.update({"enable_thinking": None, "reasoning_effort": DEFAULT_EFFORT})
    if args is not None:
        for key in ("temperature", "top_p", "top_k", "min_p"):
            v = getattr(args, "default_" + key, None)
            if v is not None:
                d[key] = v
        if args.default_reasoning_effort:
            d["reasoning_effort"] = args.default_reasoning_effort
        if args.no_thinking:
            d["enable_thinking"] = False
    return d


# ---------------------------------------------------------------------------- engine


class Heartbeat:
    """Prints one progress line every `every` seconds while a long start-up phase runs."""

    def __init__(self, label: str, every: float = 15.0):
        self.label, self.every, self.stop = label, every, threading.Event()

    def __enter__(self):
        self.t0 = time.time()
        print(f"qserve: {self.label} ...", flush=True)
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        while not self.stop.wait(self.every):
            print(f"qserve: {self.label} ... {time.time() - self.t0:.0f} s", flush=True)

    def __exit__(self, *exc):
        self.stop.set()
        print(f"qserve: {self.label} done in {time.time() - self.t0:.0f} s", flush=True)


# ---------------------------------------------------------------------------- loop guard

# The detectors live in tools/loopguard.py (shared with the GLM server).
from loopguard import (GUARD_ANSWER, loop_guard_default, GUARD_CHECK_EVERY, GUARD_MAX_N, GUARD_THINK, SPAN_LEN, SPAN_MIN, SPAN_TAIL,  # noqa: E402,F401
                       TEXT_MIN_CHARS, TEXT_RATIO, TEXT_WINDOW, LoopMonitor, answer_text_loop, counter_drift,
                       max_span_repeat, repeated_units, shingle_repeat, text_loop)

THINK_CLOSE_TEXT = "\n</think>\n\n"
THINK_CLOSE_TAG = "</think>"
FINAL_TRIES = 2


def gate_text(vis: str) -> str:
    """The text the release gate sees: reasoning + </think> + answer, each stripped (what Splitter returns to the client)."""
    think, tag, ans = vis.partition(THINK_CLOSE_TAG)
    return think.strip() + tag + ans.strip()



def final_cut(vis: str) -> int | None:
    """The gate rule holds on the visible text (thinking [+ </think> + answer]). Return the character index at which to cut
    the thinking: the longest prefix (line starts first, then 64-character steps) that (a) holds no 50-character span twice
    in its last 3000 characters, so that the answer, which repeats the last draft once more, cannot bring any span to the
    count of 3 whatever it says, and (b) with </think> and the answer already written does not trip the rule. None if no cut does (the answer trips the rule on its own)."""
    think, _, ans = vis.partition(THINK_CLOSE_TAG)
    think = think.rstrip()
    lines = sorted({m.end() for m in re.finditer("\n", think)}, reverse=True)
    steps = sorted(set(range(0, len(think), 64)) - set(lines), reverse=True)
    for c in lines + steps:
        pre = think[:c].rstrip()
        if max_span_repeat(pre) <= 1 and not text_loop(pre + THINK_CLOSE_TAG + ans.strip()):
            return c
    return None


class Retract(int):
    """Yielded by QwenEngine.generate after a loop-guard restart: drop this many trailing characters of the text
    streamed so far (the discarded part of the thinking). A streaming client has already received them."""


# Serving defaults applied at start-up (the caller's environment wins). The first six are the decode/verify set.
# The last five are the prefill set (served A/B on the test box, outputs identical): hand HIP kernels for the PLE chain,
# deinterleave_qg, the gated-residual GEMMs and the GDN chunk path (bitwise equal to the torch/triton paths, +21 % to +24 %
# prefill at 4K/16K), and EXL3_PF_SKIP (the MTP draft prefill fills K/V only and skips the 1-row lm_head, +0.5 % to +2 %).
# Set any of them to 0 to go back.
SERVE_ENV = (("EXL3_HOST_LEAN", "1"), ("EXL3_HOST_CUTS", "1"),   # batched verify readback + fewer host dispatches: +0.8 % speculative, ids identical
             ("EXL3_MOE_FUSED", "1"), ("EXL3_MOE_VALU", "1"), ("EXL3_VERIFY_ATTN_LOOP", "1"),
             ("EXL3_VERIFY_GEMV_R", "1"), ("EXL3_GEMV_R_DEC1", "1"), ("EXL3_MTP_FUSE_CATCHUP", "0"),
             ("EXL3_PLE_HIP", "1"), ("EXL3_DQ_HIP", "1"), ("EXL3_GR_HIP", "1"), ("EXL3_GDN_FUSE", "1"), ("EXL3_PF_SKIP", "1"),
             # the attention / GDN blocks keep o_proj's fp16 output (the residual apply widens in-kernel) and the ablit projection
             # reads the fp16 x directly, as on GLM: bit-identical, one-load A/B 6/6 + 6/6 + 6/6 logits equal.
             ("EXL3_NC2_ATTN_F16", "1"), ("EXL3_NC2_ABLIT_F16", "1"),
             # 4096-row prefill chunks (one MoE pass per 4096 rows, no 2048 + 2048 + tail split below 4097 tokens) and the
             # draft prefill run chunk by chunk with the target prefill (K/V and indexer planes only). EXL3_PF_DEFER=1 defers it behind the
             # first token instead; that lands the whole draft prefill (2.4 s at 256K) inside the first decode steps. 0 for either goes back.
             ("EXL3_PREFILL_CHUNK", "4096"), ("EXL3_PF_DEFER", "0"),
             # the last prefill chunk runs to the end of the prompt (a tail of up to 1024 rows beyond the 4096-row chunk is merged
             # into it) instead of a second forward pass that reads every expert again; the last-page recurrent state is written from
             # inside the chunk. 0 for EXL3_PF_NO_TAIL goes back. Tuned dense-GEMM solutions for the larger row classes ship in
             # exllamav3/model/dense_gemm_tune_seed.txt.
             ("EXL3_PF_NO_TAIL", "1"), ("EXL3_PF_TAIL_MERGE", "1024"),
             # the batched R-row sparse attention launch is row-invariant (the split plan no longer depends on R, see
             # qsa_split_plan; EXL3_QSA_ROWINV=0 goes back), so verify keeps ONE launch and speculative == plain bitwise above the
             # QSA sparse threshold (2051 tokens). The per-row fallback is EXL3_VERIFY_QSA_PROJ=1 EXL3_VERIFY_QSA_ATTN=all.
             ("EXL3_VERIFY_QSA_PROJ", "0"), ("EXL3_VERIFY_QSA_ATTN", "0"), ("EXL3_QSA_ROWINV", "1"),
             # checkpoint the decode state at the end of every answer, so the next chat turn only prefills
             # the new message (without it an answer shorter than the 2048-token grid is prefilled again). 0 = off.
             ("EXL3_ROLL_CKPT", "1"),
             # recurrent-state stashes (mid-prompt checkpoint, prompt-end tail) copy device -> pinned host without a host sync (a worker
             # thread settles them before any reader): the 72 per-layer .cpu() syncs of an 8K prompt drained the queue twice (idle 55-64 ms
             # of a 5.2 s prefill); bitwise (first-token + tail-resume logits equal, base vs on). 0 = the synchronous stash.
             ("EXL3_NC2_ASYNC_STASH", "1"),
             # a speculative rewind records the accepted recurrent-state history index in a per-slot base instead of copying
             # it back to index 0: bit-exact (ids equal off vs on, Qwen GDN ndt 7). 0 = copy.
             ("EXL3_STATE_REBASE", "1"),
             # a turn that resumes mid-page (tail checkpoint) and adds 1024..4095 tokens in one chunk no longer pays a separate
             # last-page forward: follow-up after a 12K conversation 1.47 -> 1.26 s (+1536 tokens), 2.37 -> 2.11 s (+3072), first-token
             # KLD vs a cold prefill 2.1e-3 / 1.9e-3 (shipped 4.8e-3 / 3.8e-3), top-1 equal; 768 and 6144 appends unchanged (bitwise).
             # Slot files keep the tail checkpoint, so a reload does not replay the gap. 1024 goes back.
             ("EXL3_TAIL_MAXGAP", "8192"),
             # a prompt that shares a cached prefix but not its recurrent checkpoint (a new conversation on the same long system
             # prompt) side-writes the state at the end of the shared pages, so the next one resumes there: first token after a
             # shared 9K / 11K / 12.1K / 14K prefix 1.04 -> 0.60, 2.25 -> 0.80, 2.92 -> 0.63, 1.62 -> 0.74 s; the writing prompt is
             # bitwise unchanged, KLD vs a cold prefill at the level of any resumed turn. 0 = off.
             ("EXL3_BRANCH_CKPT", "1"),
             # dense prompt-tail checkpoints: the recurrent state is also written at each of the 7 page boundaries below the last full
             # page, so a prompt that shares the cached prefix up to anywhere in its last 2K tokens (same document, another question)
             # resumes within one page of the divergence instead of at the previous 4096-row chunk start. 0 = off. A resume replays
             # a different number of rows than a cold prefill and the GEMM path depends on the row count, so the output is not yet
             # bit-identical to a cold prefill; the fix is in progress.
             ("EXL3_TAIL_DENSE", "7"),
             # sub-page prompt-tail checkpoints (EXL3_TAIL_SUB = number of 64-row steps below the prompt end): a prompt that diverges in
             # its final tokens resumes within 64 rows of the divergence. Same numerics caveat as the dense checkpoints. 0 = off.
             ("EXL3_TAIL_SUB", "0"),
             # bit-identical prefill levers (hyper-connection apply folded into the next norm, MoE glue v2), served
             # +2.4 % / +1.6 % / +1.8 % (fuse) and +0.9 / -0.1 / +0.3 % (glue) at 4K / 16K / 32K on bal. 0 = off.
             ("EXL3_PF_GR_FUSE", "1"), ("EXL3_MPW_GLUE", "1"),
             # GDN prefill in 16-row WY chunks (fla/hip/gdn_hf.hip, adapted from hipfire, fp32 forward-substitution inverse): one
             # kernel per layer instead of kkt_solve + the chunk kernel. Not bitwise; PPL equal (4.91972 -> 4.91967 on 24 x 2048
             # wiki rows) and closer to an exact fp32 GDN than the 64-row chain. 0 = off.
             ("EXL3_GDN_HF", "1"),
             # verify of 5..8 rows (deep draft tier of spec_policy.TierRule): row split, one-launch fused MoE, one-launch 6-row dense GEMV;
             # bit-identical per row, rows <= 4 untouched. 0 = off.
             ("EXL3_VERIFY_ROW_SPLIT", "2"), ("EXL3_MOE_R8", "1"), ("EXL3_GEMV_R_RM6", "1"),
             # Fidelity first: int8 mixer weights with group scales (fn 128 along H*D, up 64
             # along rank) and, when <model_dir>/hc_gs_sidecar.safetensors exists, its GPTQ-rounded codes (decoded-token KLD 0.0504 ->
             # 0.0455, plain decode +0.5-0.8 ms). A missing file means group scales with plain rounding. EXL3_GR_GS=0 = per-row scales.
             ("EXL3_GR_GS", "1"),
             # The verified serving command (SERVE.md, every served A/B log): k-split width 4 for the MoE matvec kernels and the
             # grouped prefill kernel from 2 rows up, so a speculative verify batch (2..4 rows) takes the deduplicating path.
             # Without them a bare `python serve.py` ran another kernel configuration than the one measured.
             ("EXL3_MOE_CFG", "2"), ("EXL3_HIP_PREFILL_MIN_ROWS", "2"),
             # Grouped MoE prefill GEMM: mpw2 (the engine default, +3-5 % prefill over the legacy kernel, identical ids). Its CPU bounds
             # proof (tools/qwen/check_mpw2_qwen.py --bits 4, bal pack: 4-bit routed experts, gate/up N 640, down N 2560, 63748 launches,
             # 0 failures, mutants fail) passed on this tree. The K6 draft experts never reach it (k2 12 aborts in a host check).
             ("MPW_KERN", "2"),
             # 16/10: tensor-core MoE GEMM from 9 rows (was 32). Short prompts (17-31 tokens) stop taking the per-expert GEMV
             # path: time to first token 212 vs 360 ms on short chat prompts, decode unchanged. Vs the true next tokens over
             # 1500 prompts it moves the model no more than prefilling the same prompt in two chunks (KLD ratio 0.98).
             ("EXL3_MOE_WMMA_MIN_ROWS", "9"),
             # Follow-up turns resume from the reply-end checkpoint (T2 prefill x0.90, quality gated).
             ("EXL3_TAIL_REPLY", "1"), ("EXL3_REPLY_FIRST", "1"),
             # 17/10: multi-block exact decode top-k (bitwise on 24+ cases, decode R<=1 T>=32768 only; saves 0.13/0.42/0.91 ms per token at 128K/256K/500K). 0 = one-block.
             ("EXL3_TOPK_PAR", "1"))

# NVIDIA only (RTX 3090; the AMD values above stay as they are). The verify of 2..8 rows costs
# 25-29 ms against 22 ms for one row, while a draft step costs ~1.5 ms, 0.93 ms of it the 248K-row K5 head: deeper drafts pay
# (tier thresholds 0.3/0.1 up to 3 rows, 0.6/0.4 beyond instead of 0.6/0.3 and 0.95/0.85), and the 1-row draft steps read only the
# first 98304 head rows (EXL3_DRAFT_VOCAB, self-disabling when the language needs ids above that, see qwen3_5_mtp.py). Verified
# tokens do not change. No depth cap on the draft rule: the many-row qdec kernel (cu/qdec_rows.cu) serves the hd 256 verify attention
# of 2..8 rows at 64K-246K (3090: mix decode at 241K tokens, capped rule -> uncapped, chat 73.3 -> 82.6, code 73.2 -> 80.7, prose 53.4 -> 52.9;
# 144K: 84.0/92.2/57.7 -> 82.4/99.5/58.5); with EXL3_QDEC=0 or EXL3_QDEC_ROWS=0 the Triton verify is back and QWSPEC_DEPTHRULE=
# "96000:0.45,0.2,4,0.8,0.6" (spec_policy.py) is the matching cap. EXL3_DRAFT_VOCAB=0 and QWSPEC_* set in the environment win.
# SERVE_ENV entries tuned on gfx1151 for conversation resume and HIP kernels, not validated on NVIDIA (the 3090 numbers were
# measured without them): CUDA starts without them, the environment still wins.
HIP_ONLY_SERVE_ENV = frozenset(("EXL3_BRANCH_CKPT", "EXL3_GDN_HF", "EXL3_MOE_WMMA_MIN_ROWS", "EXL3_NC2_ABLIT_F16", "EXL3_NC2_ASYNC_STASH",
                                "EXL3_NC2_ATTN_F16", "EXL3_REPLY_FIRST", "EXL3_STATE_REBASE", "EXL3_TAIL_DENSE", "EXL3_TAIL_MAXGAP",
                                "EXL3_TAIL_REPLY", "EXL3_TAIL_SUB", "EXL3_TOPK_PAR"))
CUDA_SERVE_ENV = (("EXL3_CUDA_GRAPH", "1"), ("EXL3_DRAFT_VOCAB", "98304"), ("QWSPEC_THF", "0.3"), ("QWSPEC_THV", "0.1"), ("QWSPEC_THFD", "0.6"), ("QWSPEC_THVD", "0.4"))

HC_GS_SIDECAR = "hc_gs_sidecar.safetensors"
# The sidecar of the published pack. A file with another hash (wrong pack, damaged download) is not used: plain group rounding, one log line.
HC_GS_SIDECAR_SHA256 = "473e7547adaad190788634556ce6b32614bdb280ccedadbed8789d39b59dcc8f"


def _sha256_file(path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def configure_hc_sidecar(model_path: str) -> str:
    """Point EXL3_GR_GS_SIDECAR at <model_path>/hc_gs_sidecar.safetensors when group scales are on, the variable is unset and
    the file exists. Call before importing exllamav3 (the engine reads it at import). Returns the log line."""
    if os.environ.get("EXL3_GR_GS", "0") == "0":
        return "hc mixer: per-row int8 scales (EXL3_GR_GS=0)"
    env = os.environ.get("EXL3_GR_GS_SIDECAR")
    if env:
        return f"hc mixer: group scales, codes from {env} (EXL3_GR_GS_SIDECAR)"
    f = Path(model_path) / HC_GS_SIDECAR
    if f.is_file():
        got = _sha256_file(f)
        if got != HC_GS_SIDECAR_SHA256:
            return f"hc mixer: group scales, plain rounding ({f} has sha256 {got[:12]}, expected {HC_GS_SIDECAR_SHA256[:12]}: sidecar not used)"
        os.environ["EXL3_GR_GS_SIDECAR"] = str(f)
        return f"hc mixer: group scales, codes from {f}"
    return f"hc mixer: group scales, plain rounding (no {HC_GS_SIDECAR} in the model directory)"


DRAFT2S_IN_PACK = "draft2s.st"  # safetensors layout, but not *.safetensors so the weight loader never indexes it


def configure_draft2s(model_path: str) -> str:
    """Point EXL3_DRAFT2S (two-stage draft head, exllamav3/modules/draft2s.py) at <model_path>/draft2s.st when the
    variable is unset, the file exists and its header was built for this model (vocab_size, hidden_size). "", "0" or "off" =
    off. The head only changes how the draft argmax is found, never the output (every draft is verified). Returns the log line."""
    env = os.environ.get("EXL3_DRAFT2S")
    if env is not None:
        if env.strip().lower() in ("", "0", "off"):
            os.environ["EXL3_DRAFT2S"] = ""
            return "draft head: full head (EXL3_DRAFT2S off)"
        return f"draft head: two-stage, pack {env} (EXL3_DRAFT2S)"
    f = Path(model_path).expanduser() / DRAFT2S_IN_PACK
    if not f.is_file():
        return f"draft head: full head (no {DRAFT2S_IN_PACK} in the model directory)"
    import struct
    try:
        with open(f, "rb") as fh:
            meta = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0])).get("__metadata__") or {}
        cfg = json.loads((Path(model_path).expanduser() / "config.json").read_text())
        cfg = cfg.get("text_config", cfg)
        ok = meta.get("format") == "draft2s-v1" and int(meta["vocab"]) == cfg["vocab_size"] \
            and int(meta["in_features"]) == cfg["hidden_size"]
    except (OSError, ValueError, KeyError, struct.error):
        ok = False
    if not ok:
        return f"draft head: full head ({f} does not match this model: not used)"
    os.environ["EXL3_DRAFT2S"] = str(f)
    return f"draft head: two-stage, pack {f}"

def _env_flag(name: str, default: str, off: tuple = ("", "0")) -> bool:
    return os.environ.get(name, default) not in off


# (module, latched value, what the environment says now). The engine reads these switches when its modules are imported.
LATCHED = (
    ("exllamav3.modules.hyperconnections", lambda m: m._GR_GS, lambda: os.environ.get("EXL3_GR_GS", "0")),
    ("exllamav3.modules.block_sparse_mlp_routing", lambda m: m._MOE_VALU, lambda: os.environ.get("EXL3_MOE_VALU", "0") != "0"),
    ("exllamav3.modules.block_sparse_mlp_routing", lambda m: m._HIP_PREFILL_MIN_ROWS,
     lambda: max(2, int(os.environ.get("EXL3_HIP_PREFILL_MIN_ROWS", "17")))),
    ("exllamav3.modules.block_sparse_mlp", lambda m: m._HIP_WMMA_MIN_ROWS, lambda: int(os.environ.get("EXL3_MOE_WMMA_MIN_ROWS", "32"))),
    ("exllamav3.modules.attention_fn.qsa_triton", lambda m: m.ROWINV["on"], lambda: _env_flag("EXL3_QSA_ROWINV", "1")),
    ("exllamav3.modules.qsa_indexer", lambda m: m.VERIFY_QSA_PROJ["on"], lambda: _env_flag("EXL3_VERIFY_QSA_PROJ", "0")),
    ("exllamav3.modules.qsa_indexer", lambda m: m.VERIFY_QSA_ATTN["mode"],
     lambda: {"": "", "0": "", "1": "all", "all": "all", "sel": "sel", "att": "att"}.get(os.environ.get("EXL3_VERIFY_QSA_ATTN", ""), "")),
    ("exllamav3.generator.generator", lambda m: m.MTP_FUSE_CATCHUP,
     lambda: None if os.environ.get("EXL3_MTP_FUSE_CATCHUP") is None else int(os.environ["EXL3_MTP_FUSE_CATCHUP"])),
)


def check_latched_env() -> None:
    """Import-order trap: if exllamav3 was imported before SERVE_ENV was applied (a harness, a wrapper), the modules latched the
    engine defaults (for example the per-row mixer path instead of group scales) and the server would run another code path
    without saying so. Raise instead. Modules not imported yet read the environment at their first import and are fine."""
    bad = []
    for name, got, want in LATCHED:
        mod = sys.modules.get(name)
        if mod is not None and got(mod) != want():
            bad.append(f"{name}: engine has {got(mod)!r}, environment says {want()!r}")
    if bad:
        raise RuntimeError("exllamav3 was imported before the serving environment was set; restart with the server's own entry "
                           "point (or set the EXL3_* switches in the shell before importing): " + "; ".join(bad))


class QwenEngine:
    """Target model + MTP drafter + (optional) vision tower + one Generator."""

    def __init__(self, model_path: str, ctx: int, ndt: int = 7, draft_policy: str = "mix",
                 vision: bool = True, max_chunk_size: int = 2048, cache_bits: int = 0, sessions: int = 1,
                 k_bits: int = 0, v_bits: int = 0):
        # The measured Qwen serving configuration. Read at import time by the engine:
        # set before importing exllamav3. The caller's environment wins.
        import torch
        on_cuda = bool(torch.version.cuda and not torch.version.hip)
        for k, v in SERVE_ENV:
            if on_cuda and k in HIP_ONLY_SERVE_ENV:
                continue
            os.environ.setdefault(k, v)
        check_latched_env()
        print("qserve:", configure_hc_sidecar(model_path), flush=True)
        print("qserve:", configure_draft2s(model_path), flush=True)
        if on_cuda:
            for k, v in CUDA_SERVE_ENV:
                os.environ.setdefault(k, v)
        from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
        torch.set_grad_enabled(False)
        self.torch, self.Generator, self.Job = torch, Generator, Job
        self.sessions = sessions
        self.row_budget = 0   # drafter-free lookup row cap: this tree has no --row-budget plumbing (rows = 0 = one active request only)
        import spec_policy  # tools/qwen/spec_policy.py
        pl = spec_policy.plain_lookup_from_env(self.row_budget)   # drafter-free lookup (plain rounds): its K needs the same history depth
        ndt = max(ndt, int(os.environ.get("QWSPEC_MAXD") or 0), int(os.environ.get("QWSPEC_FIXD") or 0), pl[1] if pl else 0)  # buffers fit the deepest draft
        self.model_path, self.ctx, self.ndt, self.draft_policy = model_path, ctx, ndt, draft_policy
        self.config = Config.from_directory(model_path)
        self.tokenizer = Tokenizer.from_config(self.config)
        if os.environ.get("KYO_TOKCACHE", "8") == "0":
            self.pec = None
        else:
            self.pec = PromptEncodeCache(self.tokenizer, n=int(os.environ.get("KYO_TOKCACHE", "8") or 0), cut="<|im_start|>", check=os.environ.get("KYO_TOKCACHE_CHECK") == "1", log=lambda m: print(f"qserve: {m}", flush=True))
        self.eos = list(self.config.eos_token_id_list)
        self.model = Model.from_config(self.config)
        # cache_bits 0 (default) = the fp16 K/V pages; 8 (--cache-bits 8) = packed int8 K/V pages (2.6 GiB less at 256K), indexer
        # planes stay fp16. Parallel sessions give equal output only with EXL3_ROW_INV_GROUPS=1 (see main()); the int8 pages stay opt-in.
        # k_bits / v_bits (--k-bits / --v-bits, 2..8) set the two sides separately and win over cache_bits; 0 = take cache_bits
        k_bits, v_bits = k_bits or cache_bits, v_bits or cache_bits
        if not (k_bits or v_bits) and torch.version.cuda and not torch.version.hip:
            k_bits = v_bits = 4   # NVIDIA preset: K4V4 pages (256K fits 24 GB, decode read 4x less); --cache-bits / --k-bits override
        self.cache_bits = cache_bits
        from exllamav3 import CacheLayer_quant
        qkw = dict(layer_type=CacheLayer_quant, k_bits=k_bits, v_bits=v_bits) if (k_bits and v_bits) else {}
        assert bool(k_bits) == bool(v_bits), "set both --k-bits and --v-bits (or --cache-bits)"
        # one recurrent (GDN) slot per session; the Generator caps its batch at the slot count
        self.cache = Cache(self.model, max_num_tokens=ctx, max_history=max(3, ndt), max_batch_size=sessions, **qkw)
        if os.environ.get("EXL3_WARM_DENSE", "1") != "0":
            from exllamav3.model.dense_warmup import seed_dense_tune
            seed_dense_tune()
        from exllamav3.util import jit_log, kernel_seed
        kernel_seed.install()   # shipped Triton kernels for this Triton version: nothing to compile on a first start
        jit_log.install()   # EXL3_LOG_JIT=1: one line per Triton compile
        startup_health.stage("target weights")
        with Heartbeat("loading target model"):
            self.model.load(max_chunk_size=max_chunk_size, progressbar=False, callback=startup_health.load_callback())
        from exllamav3.modules.hyperconnections import GS_STATS
        print("qserve: hc mixer sites by code source", dict(GS_STATS), flush=True)
        self.draft_model = self.draft_cache = None
        if draft_policy != "off":
            self.draft_model = Model.from_config(self.config, component="mtp")
            self.draft_cache = Cache(self.draft_model, max_num_tokens=ctx, max_history=max(3, ndt), **qkw)
            startup_health.stage("drafter weights")
            with Heartbeat("loading MTP drafter"):
                self.draft_model.load(progressbar=False, callback=startup_health.load_callback())
        self.vision = None
        self.emb_cache = image_cache.ImageEmbeddingCache()
        if vision:
            try:
                self.vision = Model.from_config(self.config, component="vision")
                startup_health.stage("vision tower")
                with Heartbeat("loading vision tower"):
                    self.vision.load(progressbar=False, callback=startup_health.load_callback())
            except Exception as exc:                                  # noqa: BLE001
                self.vision = None
                print(f"qserve: vision tower not loaded ({exc!r}); image input disabled", flush=True)
        self.supports_vision = self.vision is not None
        self.vocab_size = int(self.config.vocab_size or self.tokenizer.actual_vocab_size)   # rows of the embedding table
        startup_health.stage("engine setup")
        self._ngram_resident()
        self.generator = self.uninstall = None
        self.spec_on = self.draft_model is not None
        self.slot_store = None
        self.last_stats: dict[str, Any] = {}
        # The loaded model and tokenizer are about 1.9 M Python objects, and every later gc.collect() walked them all: the Generator
        # rebuild of a cache_salt change or cache_prompt=false spent 0.5 s there, inside time to first token (0.53 -> 0.03 s).
        # Keep them out of the collector, before the first Generator exists (a frozen Generator would never be freed). 0 = off.
        if os.environ.get("EXL3_GC_FREEZE", "1") != "0":
            gc.collect()
            gc.freeze()
        self._build(self.spec_on)

    def _ngram_resident(self) -> None:
        """t4: the kernel can swap part of the host-RAM n-gram table out while the weights load; every lookup on a swapped page then
        stalls on disk (transient 1-30 % slower first passes). Read it back in before READY. EXL3_NGRAM_POPULATE=0 = off."""
        if os.environ.get("EXL3_NGRAM_POPULATE", "1") == "0":
            return
        from exllamav3.modules.ngram_embedding import NGramEmbedding
        for mod in self.model.modules:
            for m in mod:
                if isinstance(m, NGramEmbedding):
                    info = m.ensure_resident()
                    sw = info["swap_before"]
                    if info["populated"]:
                        print(f"qserve: n-gram table was partly swapped out (process swap {sw / 2**20:.0f} MiB): read back "
                              f"{info['populated'] / 2**30:.1f} GiB in {info['seconds']} s, swap now {info.get('swap_after', 0) / 2**20:.0f} MiB", flush=True)
                    elif info.get("error"):
                        print(f"qserve: WARNING n-gram table is partly swapped out (process swap {sw / 2**20:.0f} MiB) and could not be read back: {info['error']}", flush=True)

    def _build(self, spec: bool) -> None:
        """(Re)create the Generator. Drops the prompt cache: used for mode switches and cache_prompt=false."""
        if self.uninstall:
            self.uninstall()
            self.uninstall = None
        self.generator = None
        gc.collect()          # the old Generator sits in reference cycles (jobs, hooks); free its buffers and n-gram index now
        self.torch.cuda.empty_cache()
        spec = spec and self.draft_model is not None
        kw = dict(model=self.model, cache=self.cache, tokenizer=self.tokenizer)
        import spec_policy  # tools/qwen/spec_policy.py: dynamic length + n-gram lookup, the shipped rule
        pl = None if spec else spec_policy.plain_lookup_from_env(self.row_budget)
        if spec:
            kw.update(draft_model=self.draft_model, draft_cache=self.draft_cache, num_draft_tokens=self.ndt)
        elif pl:
            kw.update(ngram_match_min=pl[0], num_draft_tokens=pl[1])
        # the drafter-free plain lookup answers the Generator's n-gram slot itself
        # (install_plain_lookup replaces iterate_ngram_gen), so the built-in n-gram path and its BC_SAM binding are never
        # used. On a build without BC_SAM (this ROCm ext returns None for ext.BC_SAM(); the built-in path raises at
        # construction) that check must not block the plain lookup. Restore the exact descriptor afterwards.
        _val = self.Generator.__dict__["_validate_ngram_drafting_support"]
        if pl:
            self.Generator._validate_ngram_drafting_support = staticmethod(lambda: None)
        try:
            self.generator = self.Generator(**kw)
        finally:
            self.Generator._validate_ngram_drafting_support = _val
        if pl:
            self.uninstall = spec_policy.install_plain_lookup(self.generator, pl)
        if spec and self.draft_policy == "mix":
            thf, thv, maxd = spec_policy.env_defaults(self.ndt)   # 0.6 / 0.3 / ndt unless QWSPEC_THF / THV / MAXD are set
            fx = os.environ.get("QWSPEC_FIXD")
            self.uninstall = spec_policy.install(self.generator, self.draft_model, self.model,
                                                 thf, thv, maxd, spec_policy.ngram_from_env(), {},
                                                 rule=spec_policy.rule_from_env(thf, thv, maxd), fixed=int(fx) if fx else None)
        self.spec_on = spec
        if self.slot_store is not None:
            self.slot_store.generator = self.generator

    def warmup(self) -> None:
        # dense GEMM warm-up at load (EXL3_WARM_DENSE=0 = off): the first request at a new row-count class
        # no longer pays the rocBLAS solution screening (0.15-4 s per shape).
        from exllamav3.model.dense_warmup import warm_dense_gemm
        warm_dense_gemm([self.model, self.draft_model],
                        max_rows=self.generator.max_chunk_size + (int(os.environ.get("EXL3_PF_TAIL_MERGE", "0"))
                                                                  if os.environ.get("EXL3_PF_NO_TAIL", "0") == "1" else 0))
        ids = self.tokenizer.encode("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                                    add_bos=False, encode_special_tokens=True)
        job = self.Job(input_ids=ids, max_new_tokens=24, sampler=self._sampler({"temperature": 0.0}),
                       stop_conditions=self.eos)
        self.generator.enqueue(job)
        while self.generator.num_remaining_jobs():
            list(self.generator.iterate())
        self.torch.cuda.synchronize()
        # The first full-size prefill chunk costs ~1.2 s more than every later one (one-time lazy init inside the
        # forward at chunk size). Pay it at load, not on the first request: one chunk plus a short tail of ordinary text
        # (EXL3_WARM_PREFILL=0 = off).
        if os.environ.get("EXL3_WARM_PREFILL", "1") != "0":
            chunk = int(self.generator.max_chunk_size)
            text = "<|im_start|>user\n" + "The quick brown fox jumps over the lazy dog near the river bank. " * (chunk // 6 + 64)
            ids = self.tokenizer.encode(text, add_bos=False, encode_special_tokens=True)
            ids = ids[:, :chunk + 64].contiguous()
            job = self.Job(input_ids=ids, max_new_tokens=2, sampler=self._sampler({"temperature": 0.0}),
                           stop_conditions=self.eos)
            self.generator.enqueue(job)
            while self.generator.num_remaining_jobs():
                list(self.generator.iterate())
            self.torch.cuda.synchronize()

    def count_tokens(self, text: str) -> int:
        return int(self.tokenizer.encode(text, add_bos=False, encode_special_tokens=True).numel())

    def prompt_ids(self, prompt: str):
        if self.pec and self.pec.n:
            return self.pec.encode(prompt)[0]
        ids = self.tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
        ids = ids[0] if isinstance(ids, tuple) else ids
        return ids.reshape(-1)

    def encode_ids(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """POST /tokenize: the ids of text (special-token text is encoded as the special token; add_special_tokens adds BOS)."""
        return [int(t) for t in self.tokenizer.encode(text, add_bos=add_special_tokens, encode_special_tokens=True).flatten().tolist()]

    # Persistent prompt cache (EXL3_PREFIX_DISK_DIR, exllamav3/generator/prefix_disk.py). Both run on the driver, generator idle.
    async def disk_restore(self, ids) -> None:
        """Before the request is enqueued: put a deeper resume point from the disk (an earlier run, an evicted prefix) back."""
        pd = self.prefix_disk
        try:
            r = await self._post(lambda: None if self.generator.num_remaining_jobs() else pd.restore(ids))
        except Exception as exc:                                        # noqa: BLE001 the disk never fails a request
            print(f"qserve: prefix disk restore failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return
        if r:
            print(f"qserve: prefix disk restore resume_at={r['resume_at']} (ram {r['ram_at']}) pages={r['pages']} {r['ms']:.0f} ms",
                  file=sys.stderr, flush=True)

    async def disk_save(self) -> None:
        """After a request: persist its prompt (new pages, checkpoints) with a background write."""
        pd = getattr(self, "prefix_disk", None)
        ids = getattr(self, "_disk_ids", None)
        if pd is None or ids is None:
            return
        try:
            r = await self._post(lambda: None if self.generator.num_remaining_jobs() else pd.save(ids))
        except Exception as exc:                                        # noqa: BLE001
            print(f"qserve: prefix disk save failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return
        if r is not None:
            self._disk_ids = None

    async def reset_prefix_cache(self) -> None:
        """Forget every cached prompt prefix (page table, recurrent checkpoints, n-gram index): a fresh Generator, the path
        cache_prompt=false takes. The caller has drained the requests (cache_salt namespace change)."""
        if getattr(self, "prefix_disk", None) is not None:
            self.prefix_disk.clear()
        await self._post(self._build, self.spec_on)
        if self.slot_store is not None:
            self.slot_store.erase()

    def token_str(self, token_id: int) -> str:
        return self.tokenizer.tokenizer.decode([int(token_id)], skip_special_tokens=False)

    def _sampler(self, s: dict[str, Any]):
        from exllamav3.generator.sampler import ArgmaxSampler, ComboSampler
        if s["temperature"] == 0 or s.get("top_k") == 1:
            return ArgmaxSampler()
        return ComboSampler(temperature=s["temperature"], top_k=s.get("top_k", 0), top_p=s.get("top_p", 1.0),
                            min_p=s.get("min_p", 0.0))

    def _embed(self, images: list[tuple[Any, str]]) -> list:
        """One MMEmbedding per image. The same bytes give the same embedding object (same pseudo token ids), so a
        history turn that repeats an image still matches the paged KV cache (see image_cache.py)."""
        def build(image):
            t0 = time.perf_counter()
            emb = self.vision.get_image_embeddings(tokenizer=self.tokenizer, image=image)
            print(f"qserve: image {image.size[0]}x{image.size[1]} -> {emb.mm_length} tokens, tower {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)
            return emb
        return self.emb_cache.get_many(images, build)

    def _encode(self, prompt: str, image_urls: list[str]):
        embs = self._embed([load_image_keyed(url) for url in image_urls]) if image_urls else []
        for emb in embs:
            prompt = prompt.replace(IMAGE_PAD, emb.text_alias, 1)
        kw = {"embeddings": embs} if embs else {}
        ids = self.prompt_ids(prompt).reshape(1, -1) if not embs else self.tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True, **kw)
        return ids, embs

    def _close_ids(self) -> tuple[int | None, list[int]]:
        """(id of the </think> token, ids of the text that closes a think block). Cached."""
        if not hasattr(self, "_think_close"):
            one = self.tokenizer.encode(THINK_CLOSE, add_bos=False, encode_special_tokens=True).flatten().tolist()
            seq = self.tokenizer.encode(THINK_CLOSE_TEXT, add_bos=False, encode_special_tokens=True).flatten().tolist()
            self._think_close = (one[0] if len(one) == 1 else None, seq)
        return self._think_close

    def _post(self, fn, *args) -> asyncio.Future:
        """Queue fn(*args) for the driver task, which runs it between two iterate() calls. The driver is the only code that
        touches the Generator, so concurrent requests never mutate it while a step runs in the executor."""
        self._ensure_driver()
        fut = asyncio.get_running_loop().create_future()
        self._cmds.append((fn, args, fut))
        self._wake.set()
        return fut

    def _ensure_driver(self) -> None:
        loop = asyncio.get_running_loop()
        if getattr(self, "_driver", None) is None or self._driver.get_loop() is not loop:
            self._cmds, self._routes, self._wake = [], {}, asyncio.Event()
            self._driver = loop.create_task(self._drive())
        elif self._driver.done():
            exc = None if self._driver.cancelled() else self._driver.exception()
            if exc is None:        # cancelled (shutdown): stays stopped
                raise RuntimeError("the engine driver has stopped")
            # crashed: _drive already failed everything that was waiting, so a fresh driver can take the next request
            print(f"qserve: WARNING the engine driver crashed ({exc!r}); starting a new one", file=sys.stderr, flush=True)
            self._driver = loop.create_task(self._drive())

    async def _drive(self) -> None:
        try:
            await self._drive_loop()
        finally:
            # cancelled (shutdown) or crashed: nothing will answer the waiting requests any more, so fail them now
            stopped = RuntimeError("the engine driver has stopped")
            for _, _, fut in self._cmds:
                if not fut.done():
                    fut.set_exception(stopped)
            self._cmds.clear()
            for _, q in self._routes.values():
                q.put_nowait(stopped)
            self._routes.clear()

    async def _drive_loop(self) -> None:
        loop = asyncio.get_running_loop()
        rf_wait = 0.0
        while True:
            while self._cmds:
                fn, args, fut = self._cmds.pop(0)
                try:
                    res = await loop.run_in_executor(None, fn, *args)
                except asyncio.CancelledError:
                    # shutdown while the command runs: it is no longer in self._cmds, so _drive cannot fail it
                    if not fut.done():
                        fut.set_exception(RuntimeError("the engine driver has stopped"))
                    raise
                except Exception as exc:                              # noqa: BLE001
                    if not fut.done():
                        fut.set_exception(exc)
                else:
                    if not fut.done():
                        fut.set_result(res)
            if not self.generator.num_remaining_jobs():
                # EXL3_REPLY_FIRST: release the finished jobs (reply-end stash) once idle AND their handlers have
                # dropped their routes (_drop is the last driver command a request waits for before its reply ends), at most
                # 50 ms after the step; a new request's iterate() releases them first anyway
                dj = getattr(self.generator, "deferred_jobs", None)
                if dj and not self._cmds:
                    if rf_wait < 0.05 and any(self._routes.get(getattr(j, "serial_number", None), (None,))[0] is j for j in dj):
                        self._wake.clear()
                        try:
                            await asyncio.wait_for(self._wake.wait(), 0.005)
                        except asyncio.TimeoutError:
                            pass
                        rf_wait += 0.005
                        continue
                    rf_wait = 0.0
                    await loop.run_in_executor(None, self.generator.flush_deferred)
                    continue
                self._wake.clear()
                if not self._cmds:
                    await self._wake.wait()
                continue
            try:
                events = await loop.run_in_executor(None, lambda: list(self.generator.iterate()))
            except Exception as exc:                                  # noqa: BLE001
                # a failed step leaves every job in an unknown state: fail them all, as the old serial loop did
                for _, q in self._routes.values():
                    q.put_nowait(exc)
                self._routes.clear()
                self.generator.clear_queue()
                continue
            by_serial: dict[int, list] = {}
            for ev in events:
                try:
                    by_serial.setdefault(ev["serial"], []).append(ev)
                except Exception as exc:                              # noqa: BLE001
                    self._fail_unroutable(ev, exc)
            for serial, evs in by_serial.items():
                route = self._routes.get(serial)
                if route is not None:      # None: cancelled after the step started
                    route[1].put_nowait(evs)

    def _fail_unroutable(self, ev, exc: Exception) -> None:
        """An event the driver cannot route must not stop the driver. Fail the request it belongs to (found through its job),
        or every request when that is unknown, and keep serving."""
        print(f"qserve: WARNING unroutable generator event ({exc!r}): {ev!r:.200}", file=sys.stderr, flush=True)
        job = ev.get("job") if isinstance(ev, dict) else None
        serial = getattr(job, "serial_number", None)
        route = self._routes.pop(serial, None) if serial is not None else None
        if route is not None:
            route[1].put_nowait(exc)
            self._cancel_serial(serial)
            return
        for _, q in self._routes.values():
            q.put_nowait(exc)
        self._routes.clear()
        self.generator.clear_queue()

    def _cancel_serial(self, serial: int) -> None:
        for job in self.generator.active_jobs + self.generator.pending_jobs:
            if job.serial_number == serial:
                self.generator.cancel(job)

    async def _submit(self, job) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()

        def enqueue():
            # on the driver, so no step can emit before the route exists. The job is kept with the route: a failed enqueue
            # does not use up its serial, so the next job gets the same one
            self._routes[self.generator.enqueue(job)] = (job, q)
        await self._post(enqueue)
        return q

    def _drop(self, job) -> asyncio.Future:
        """Stop routing the job and remove it from the Generator (a no-op when it already finished). Runs on the driver, after
        any enqueue posted before it, so a job cancelled while its enqueue is still queued does not stay behind."""
        def forget():
            serial = getattr(job, "serial_number", None)
            route = self._routes.get(serial)
            if route is not None and route[0] is job:
                del self._routes[serial]
                self._cancel_serial(serial)
        return self._post(forget)

    async def run_exclusive(self, fn, *args):
        """Run fn on the driver, between steps (slot save/restore/erase; the caller has drained the requests)."""
        return await self._post(fn, *args)

    async def generate(self, prompt: str, stats: dict[str, Any] | None = None, **kw: Any) -> AsyncIterator[str]:
        """See _generate. Whatever way the call ends (error in a step, client gone, cancelled task), no job of it stays in the
        Generator: a leftover job would run inside the next request and deliver its tokens there. Final stats land in
        `stats` (per request) and in self.last_stats (the last request to finish)."""
        stats = {} if stats is None else stats
        mine: list = []
        try:
            async for piece in self._generate(prompt, stats=stats, _mine=mine, **kw):
                yield piece
        finally:
            self.last_stats = stats
            if mine and not self._driver.done():
                # wait for the cancels: when this returns, no job of the request is left (slot actions rely on it)
                for res in await asyncio.shield(asyncio.gather(*[self._drop(job) for job in mine], return_exceptions=True)):
                    if isinstance(res, BaseException):
                        print(f"qserve: WARNING could not remove a finished request's job: {res!r}", file=sys.stderr, flush=True)

    async def _generate(self, prompt: str, *, max_tokens: int, sampling: dict[str, Any], stop: list[str],
                        images: list[str] | None = None, speculative: bool = True, cache_prompt: bool = True,
                        cancel: asyncio.Event | None = None, seed: int | None = None, reasoning_first: bool = False,
                        thinking_budget: int | None = None, loop_guard: bool | None = None,
                        loop_final: bool | None = None, logprobs: int | None = None, stats: dict[str, Any],
                        prompt_ids: list[int] | None = None, ignore_eos: bool = False, _mine: list) -> AsyncIterator[str]:
        """Yield decoded text pieces. Final stats land in `stats`. Every job enqueued is appended to `_mine`.

        Final check (on with the guard, loop_final=False turns it off): when the turn ends normally and the visible text
        (thinking + answer) trips the release-gate loop rule and some cut of the thinking cures it (a long valid code answer
        trips the rule on its own: no cut helps, nothing is redone), the thinking is cut back (final_cut), the close is forced and
        the answer is generated again, at most FINAL_TRIES times. Discarded tokens do not count against max_tokens.

        Loop guard (default on, EXL3_LOOP_GUARD=0 or loop_guard=False turns it off): when the think block loops (or
        thinking_budget tokens are used) the job is cancelled and restarted from prompt + the thinking kept so far +
        "\n</think>\n\n", so the answer proceeds. A loop inside the answer ends the turn: finish_reason "length" and choices[0].stop_reason "loop_guard"
        (plus timings.loop_guard with the detail), so a client can tell a guard stop from a natural end.
        Nothing is changed while no loop is detected.

        logprobs (None = off, k = 0..20): per generated token the log-probability of the token and of the k most likely tokens,
        taken from the model distribution before temperature and truncation. They land in stats["logprobs"], one entry
        per entry of token_ids ((token_logprob, [(id, logprob), ...]), None for a token the guard forced)."""
        if cancel is not None and cancel.is_set():
            return                          # the client left while the request was queued: no rebuild, no prefill
        want_spec = bool(speculative) and self.draft_model is not None
        if want_spec != self.spec_on or not cache_prompt:
            await self._post(self._build, want_spec)
        if prompt_ids is not None:   # token-id prompt: exactly these ids, no template, no BOS, no re-encoding
            ids, embs = self.torch.tensor([prompt_ids], dtype=self.torch.long), []
        else:
            ids, embs = await self._post(self._encode, prompt, images or [])
        ctx = getattr(self, "ctx", None)
        if ctx:   # the chat handler counted a picture as one token; its embeddings take many
            room = reply_room(ctx, int(ids.numel()), int(getattr(self.generator, "num_draft_tokens", 0)))
            if room < 1:
                raise BadRequest(f"context length exceeded: prompt is {int(ids.numel())} tokens, the server context is {ctx}")
            max_tokens = min(max_tokens, room)
        if self.slot_store is not None:
            self.slot_store.note_prompt(prompt, ids)
        if getattr(self, "prefix_disk", None) is not None and cache_prompt and not embs:
            await self.disk_restore(ids)
            self._disk_ids = ids
        guard_on = loop_guard_default() if loop_guard is None else bool(loop_guard)
        budget = thinking_budget if reasoning_first else None
        close_id, close_seq = self._close_ids()
        mon = None
        if guard_on or budget:
            mon = LoopMonitor("think" if reasoning_first else "answer", close_id, budget, detect=guard_on)
        token_ids: list[int] = []          # everything streamed to the client, all segments
        lps: list = []                     # parallel to token_ids when logprobs is asked
        fired: dict[str, Any] | None = None
        marks: list[tuple[int, int]] = [(0, 0)]   # (tokens, characters) streamed after each event of this segment
        chars = 0
        spent = 0.0                        # seconds of cancelled segments
        generated = 0                      # tokens the model produced, including the discarded thinking tail
        discount = 0                       # discarded tokens of final-check recoveries: not charged to max_tokens
        final_tries = 0
        final_on = guard_on and reasoning_first and close_id is not None and loop_final is not False
        vis = ""                          # the text the client holds (Retract applied)
        gmarks: list[tuple[int, int]] = [(0, 0)]   # (len(token_ids), chars) after each event, all segments
        job_ids = ids

        while True:
            max_new = max(max_tokens - generated + discount, 1)
            rq = serve_compat.rq_budget(ctx, self.sessions, int(job_ids.numel()), max_new, int(getattr(self.generator, "num_draft_tokens", 0)), PAGE_TOKENS)
            job = self.Job(input_ids=job_ids, max_new_tokens=max_new,
                           sampler=self._sampler(sampling), stop_conditions=([] if ignore_eos else self.eos) + list(stop),
                           embeddings=embs or None, decode_special_tokens=True, seed=seed,
                           **({} if rq is None else {"max_rq_tokens": rq}),
                           **({} if logprobs is None else {"return_probs": True, "return_top_tokens": logprobs}))
            _mine.append(job)
            q = await self._submit(job)
            seg: list[int] = []
            marks = [(0, 0)]
            base_chars = chars
            seg_t0 = time.perf_counter()
            hit = None
            redo = False
            while True:
                if cancel is not None and cancel.is_set():
                    return                 # generate() drops the job
                try:
                    batch = await asyncio.wait_for(q.get(), 0.5)
                except asyncio.TimeoutError:
                    continue               # a job waiting for cache pages sends nothing: still watch for the cancel
                if isinstance(batch, Exception):
                    raise batch
                for ev in batch:
                    if ev.get("error"):
                        raise RuntimeError(str(ev["error"]))
                    t = ev.get("token_ids")
                    toks = t.flatten().tolist() if t is not None and t.numel() else []
                    token_ids += toks
                    if logprobs is not None:
                        lps += self._event_logprobs(ev, len(toks))
                    seg += toks
                    generated += len(toks)
                    if ev.get("text"):
                        chars += len(ev["text"])
                        vis += ev["text"]
                        yield ev["text"]
                    marks.append((len(seg), chars - base_chars))
                    gmarks.append((len(token_ids), chars))
                    if mon is not None and toks and not ev.get("eos"):
                        hit = mon.feed(toks, ev.get("text") or "")
                        if hit:
                            break
                    if ev.get("eos") and final_on and final_tries < FINAL_TRIES and text_loop(gate_text(vis)) \
                            and (THINK_CLOSE_TAG in vis or ev.get("eos_reason") == "max_new_tokens"
                                 or generated - discount >= max_tokens) \
                            and (final_c := final_cut(vis)) is not None:   # no cut of the thinking cures it: the answer is the loop, or valid
                        redo = True
                        break
                    if ev.get("eos"):
                        self._routes.pop(job.serial_number, None)
                        stats.update({k: v for k, v in ev.items() if k not in ("job", "token_ids", "text", "held") and not k.startswith("stop_token_")})
                        stats["token_ids"] = token_ids
                        if logprobs is not None:
                            stats["logprobs"] = lps
                            stop_lp = self._stop_logprobs(ev)
                            if stop_lp is not None:   # the sampled stop token: no content, but its position is scored
                                stats["logprobs"] = lps + [stop_lp]
                                stats["logprob_token_ids"] = token_ids + [int(ev["eos_triggering_token_id"])]
                        if fired:
                            stats.update(prompt_tokens=int(ids.numel()), new_tokens=generated,
                                         cached_tokens=min(int(stats.get("cached_tokens", 0)), int(ids.numel())),
                                         loop_guard=fired)
                            if stats.get("time_generate") is not None:
                                stats["time_generate"] += spent
                        return
                if hit or redo:
                    break
            if redo:
                self._routes.pop(job.serial_number, None)   # it sent its eos
                cut_chars = final_c
                final_tries += 1
                keep_tokens, keep_chars = max(m for m in gmarks if m[1] <= cut_chars)
                fired = {"phase": "think", "kind": "final", "cut": keep_tokens, "at": len(token_ids), "period": 0,
                         "tokens_streamed": len(token_ids), "kept_tokens": keep_tokens, "tries": final_tries}
                print(f"qserve: loop guard final in think: gate rule holds on the answer, thinking cut to {keep_tokens} tokens "
                      f"({keep_chars} chars of {len(vis)}), try {final_tries}", file=sys.stderr, flush=True)
                spent += time.perf_counter() - seg_t0
                yield Retract(len(vis) - keep_chars)
                vis, chars = vis[:keep_chars], keep_chars
                discount += len(token_ids) - keep_tokens
                del token_ids[keep_tokens:]
                token_ids += close_seq
                del lps[keep_tokens:]
                lps += [None] * len(close_seq)
                generated += len(close_seq)
                chars += len(THINK_CLOSE_TEXT)
                vis += THINK_CLOSE_TEXT
                yield THINK_CLOSE_TEXT
                gmarks[:] = [m for m in gmarks if m[0] <= keep_tokens and m[1] <= keep_chars]
                gmarks.append((len(token_ids), chars))
                extra = self.torch.tensor([token_ids], dtype=ids.dtype)
                job_ids = self.torch.cat([ids, extra.to(ids.device)], dim=-1)
                mon.start_phase("answer")
                continue
            if not hit:
                return
            await self._drop(job)
            fired = dict(hit, tokens_streamed=len(token_ids))
            print(f"qserve: loop guard {hit['kind']} in {hit['phase']} at {hit['at']} tokens (period {hit['period']}, "
                  f"keep {hit['cut']})", file=sys.stderr, flush=True)
            n_total = int(ids.numel())
            if hit["phase"] == "answer" or close_id is None:
                stats.update({"prompt_tokens": n_total, "new_tokens": generated, "cached_tokens": 0,
                              "eos_reason": "loop_guard", "token_ids": token_ids, "loop_guard": fired})
                if logprobs is not None:
                    stats["logprobs"] = lps
                return
            # think block: keep the thinking up to the cut (on an event boundary), force the close, restart from there
            keep_tokens, keep_chars = max(m for m in marks if m[0] <= hit["cut"])
            spent += time.perf_counter() - seg_t0
            fired["kept_tokens"] = keep_tokens
            if chars - base_chars > keep_chars:
                yield Retract(chars - base_chars - keep_chars)
                chars = base_chars + keep_chars
                vis = vis[:chars]
            del token_ids[len(token_ids) - len(seg) + keep_tokens:]
            token_ids += close_seq
            del lps[len(token_ids) - len(close_seq):]
            lps += [None] * len(close_seq)
            generated += len(close_seq)
            chars += len(THINK_CLOSE_TEXT)
            vis += THINK_CLOSE_TEXT
            yield THINK_CLOSE_TEXT
            gmarks[:] = [m for m in gmarks if m[0] <= len(token_ids) - len(close_seq) and m[1] <= chars - len(THINK_CLOSE_TEXT)]
            gmarks.append((len(token_ids), chars))
            if generated - discount >= max_tokens:
                stats.update({"prompt_tokens": n_total, "new_tokens": generated, "cached_tokens": 0,
                              "eos_reason": "max_new_tokens", "token_ids": token_ids, "loop_guard": fired})
                if logprobs is not None:
                    stats["logprobs"] = lps
                return
            extra = self.torch.tensor([seg[:keep_tokens] + close_seq], dtype=ids.dtype)
            job_ids = self.torch.cat([ids, extra.to(ids.device)], dim=-1)
            mon.start_phase("answer")

    @staticmethod
    def _stop_logprobs(ev: dict[str, Any]):
        """Logprob entry of the stop token a stream event ended on (None when the event carries none)."""
        if ev.get("stop_token_prob") is None or ev.get("eos_reason") != "stop_token":
            return None
        e = QwenEngine._event_logprobs({"token_probs": ev["stop_token_prob"], "top_k_tokens": ev.get("stop_token_top_k_tokens"),
                                    "top_k_probs": ev.get("stop_token_top_k_probs")}, 1)[0]
        return e

    @staticmethod
    def _event_logprobs(ev: dict[str, Any], n: int) -> list:
        """(token_logprob, [(id, logprob), ...]) for the n tokens of a stream event."""
        probs, ids, tops = ev.get("token_probs"), ev.get("top_k_tokens"), ev.get("top_k_probs")
        if probs is None or probs.numel() != n:
            return [None] * n
        pl = probs.flatten().tolist()
        il = ids.reshape(n, -1).tolist() if ids is not None else [[]] * n
        tl = tops.reshape(n, -1).tolist() if tops is not None else [[]] * n
        return [(math.log(max(pl[i], 1e-45)), [(a, math.log(max(b, 1e-45))) for a, b in zip(il[i], tl[i])]) for i in range(n)]

    def spec_stats(self) -> dict[str, Any]:
        return {"speculative": self.spec_on, "draft_policy": self.draft_policy if self.spec_on else "off", "ndt": self.ndt}


# ---------------------------------------------------------------------------- HTTP


MAX_TOP_LOGPROBS = 20


def logprobs_arg(value: Any, top: Any = None, chat: bool = False) -> int | None:
    """The k of a request's logprobs fields, None when off. Completions: logprobs = k (0..20). Chat: logprobs = true plus
    top_logprobs = k (0..20, default 0)."""
    if chat:
        if value is None or value is False:
            if top not in (None, 0):
                raise BadRequest("top_logprobs needs logprobs = true")
            return None
        if value is not True:
            raise BadRequest("logprobs must be boolean")
        k = 0 if top is None else top
    else:
        if value is None:
            return None
        k = value
    if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k <= MAX_TOP_LOGPROBS:
        raise BadRequest(f"top_logprobs must be an integer from 0 to {MAX_TOP_LOGPROBS}")
    return k


def format_logprobs(entries: list, token_ids: list[int], token_str, chat: bool) -> dict[str, Any]:
    """OpenAI logprobs object for the generated tokens. Every token and candidate also carries its id. Values are natural
    logs of the model distribution before temperature, top-k, top-p and min-p. A token the loop guard forced has none (null)."""
    def tok(i: int) -> dict[str, Any]:
        t = token_str(i)
        return {"token": t, "bytes": list(t.encode("utf-8")), "token_id": int(i)}
    if chat:
        content = []
        for tid, e in zip(token_ids, entries):
            if e is None:
                content.append(tok(tid) | {"logprob": None, "top_logprobs": []})
            else:
                content.append(tok(tid) | {"logprob": e[0], "top_logprobs": [tok(a) | {"logprob": b} for a, b in e[1]]})
        return {"content": content}
    toks, offs, pos = [], [], 0
    for tid in token_ids:
        t = token_str(tid)
        toks.append(t)
        offs.append(pos)
        pos += len(t)
    return {"tokens": toks, "token_ids": [int(t) for t in token_ids], "text_offset": offs,
            "token_logprobs": [None if e is None else e[0] for e in entries],
            "top_logprobs": [None if e is None else {token_str(a): b for a, b in e[1]} for e in entries],
            "top_logprobs_ids": [None if e is None else [[int(a), b] for a, b in e[1]] for e in entries]}


# Every route of create_app(), for the early /health listener (404 / 405 while loading); a test keeps it in step.
ROUTES = {"/health": ("GET",), "/v1/models": ("GET",), "/slots": ("GET",), "/v1/chat/completions": ("POST",), "/apply-template": ("POST",), "/tokenize": ("POST",), "/completion": ("POST",), "/v1/completions": ("POST",), "/slots/{id}": ("POST",)}


def create_app(engine: Any, model_id: str, template: str, defaults: dict[str, Any] | None = None,
               default_max_tokens: int | None = None) -> web.Application:
    """HTTP layer around a resident engine (also accepts a fake engine in tests)."""
    defaults = defaults or load_defaults("/nonexistent")
    queue: asyncio.Queue = asyncio.Queue()
    sessions = getattr(engine, "sessions", 1)
    gate = asyncio.Semaphore(sessions)   # one permit per running request; slot actions take them all (drain)
    lock = asyncio.Lock()                # serializes the slot actions
    running = 0
    salt_state: dict[str, Any] = {"current": None}   # cache_salt namespace of the prompt cache (None = requests without a salt)
    admit_lock = asyncio.Lock()
    app = web.Application(client_max_size=64 * 1024**2, middlewares=[startup_health.json_errors_middleware()])
    app.update(engine=engine, model_id=model_id, template=template, defaults=defaults)
    # SSE keep-alive: off unless KYOJIN_SSE_PING_S names an interval (see tools/strix_halo/serve_ping.py).
    # A request copies this pinger, so its timers are its own; the start-up line reports the setting.
    ping = ServePing.from_env()
    app.update(ping=ping)
    # Request log: off unless KYOJIN_REQLOG_DIR names a directory (numbers and event names only). The name is bound
    # here, not only stored on the app: the streaming routes log from inside their own scope, and an
    # unbound name there raises after the reply is complete, which costs the client its connection.
    reqlog = ReqLog.from_env(server="qwen")
    app.update(reqlog=reqlog)
    background: dict[str, Any] = {}      # the heartbeat task, created at start-up (nothing when the log is off)

    # The request lifecycle is logged by one shared implementation (tools/strix_halo/serve_reqlog.py).
    # This server says when a phase begins and which counters it keeps; everything else — the request
    # table, the id on every line, the 'alive' heartbeat, the route wrapper and the summary line — is
    # the same code the other two servers run. `running` is this server's only live count.
    lifecycle = Lifecycle(reqlog, ping=ping, counters=lambda: {"running": running})

    def reqid() -> str | None:
        """The log id of the request being handled (None outside a logged route)."""
        return lifecycle.reqid()

    def track(rid: str | None, phase: str, **fields: Any) -> None:
        lifecycle.track(rid, phase, **fields)

    def err(status: int, message: str) -> web.Response:
        return web.json_response({"error": {"message": message, "type": "invalid_request_error", "code": status}},
                                 status=status)

    async def models(_: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "yamz",
             "max_context": getattr(engine, "ctx", None)}]})

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "source": "kyojin", "model": model_id, "vision": bool(engine.supports_vision), **startup_health.compiler_health(),
                                  **(engine.spec_stats() if hasattr(engine, "spec_stats") else {})})

    @contextlib.asynccontextmanager
    async def admit(salt: str | None):
        """One running-request permit. Every running request shares one cache_salt namespace: when the salt changes, the
        requests in flight finish, the prompt cache is dropped (engine.reset_prefix_cache) and the new request starts alone.
        The same salt (and traffic with no salt at all) never drops anything."""
        async with admit_lock:
            if salt == salt_state["current"]:
                await gate.acquire()
            else:
                async with lock:
                    held = 0
                    try:
                        for _ in range(sessions):
                            await gate.acquire()
                            held += 1
                        salt_state["current"] = _UNKNOWN_SALT        # a failed reset must not leave the old namespace claimed
                        await engine.reset_prefix_cache()
                        salt_state["current"] = salt
                    except BaseException:
                        for _ in range(held):
                            gate.release()
                        raise
                    for _ in range(held - 1):                        # keep the permit of this request
                        gate.release()
        try:
            yield
        finally:
            gate.release()

    async def worker() -> None:
        nonlocal running
        while True:
            job, out = await queue.get()
            store = getattr(engine, "slot_store", None)
            stats: dict[str, Any] = {}
            rid = job.get("_reqid")
            # `started` is when generation begins (what job_done reports as elapsed_s); the queue wait is
            # the difference, and it is logged on its own as wait_s.
            wait = time.perf_counter() - (job.get("_queued_at") or time.perf_counter())
            started = time.perf_counter() - wait
            if job.get("cancel") is not None and job["cancel"].is_set():
                # the client left while the job was queued: no prefill, no decode, one line
                lifecycle.job_skipped(rid, wait)
                out.put_nowait(("done", {}))
                queue.task_done()
                continue
            try:
                async with admit(job.get("cache_salt")):
                    running += 1
                    if store is not None:
                        store.busy = True
                    track(rid, "prefill", prompt_tokens=job.get("prompt_tokens"))
                    lifecycle.job_started(rid, wait, prompt_tokens=job.get("prompt_tokens"),
                                          images=len(job.get("images") or ()),
                                          messages=job.get("messages"), sessions=sessions, running=running)
                    first_token = False       # flipped once, so the log gets exactly one job_first_token line
                    try:
                        async for delta in engine.generate(
                                job["prompt"], stats=stats, max_tokens=job["max_tokens"], sampling=job["sampling"],
                                **({"prompt_ids": job["prompt_ids"]} if job.get("prompt_ids") is not None else {}),
                                **({"ignore_eos": True} if job.get("ignore_eos") else {}),
                                stop=job["stop"], images=job["images"], speculative=job["speculative"],
                                cache_prompt=job["cache_prompt"], cancel=job["cancel"],
                                seed=job.get("seed"), reasoning_first=job.get("reasoning_first", False),
                                thinking_budget=job.get("thinking_budget"), loop_guard=job.get("loop_guard"),
                                loop_final=job.get("loop_final"), logprobs=job.get("logprobs")):
                            if not first_token:
                                first_token = True
                                lifecycle.first_token(rid, started, prompt_tokens=job.get("prompt_tokens"))
                            out.put_nowait(delta)
                    finally:
                        running -= 1
                        if store is not None:
                            store.busy = running > 0
                    # a fake engine (tests) fills only last_stats
                    done = dict(stats or getattr(engine, "last_stats", None) or {})
                    print(f"qserve: request prompt={int(done.get('prompt_tokens') or 0)} cached={int(done.get('cached_tokens') or 0)} "
                          f"new={int(done.get('new_tokens') or 0)} prefill_s={float(done.get('time_prefill') or 0):.3f} "
                          f"generate_s={float(done.get('time_generate') or 0):.3f} stop={done.get('eos_reason')}",
                          file=sys.stderr, flush=True)
                    lifecycle.job_done(rid, done, started, prompt_tokens=job.get("prompt_tokens"))
                    out.put_nowait(("done", done))
                    if getattr(engine, "prefix_disk", None) is not None:
                        await engine.disk_save()    # the reply is out: the host copy of the new pages holds the lock, the write does not
            except Exception as exc:                                  # noqa: BLE001
                lifecycle.job_error(rid, exc, started)
                out.put_nowait(exc)
            finally:
                queue.task_done()

    async def start(_: web.Application) -> None:
        app["worker_tasks"] = [asyncio.create_task(worker()) for _ in range(sessions)]
        if reqlog.enabled and reqlog.heartbeat_s > 0:
            background["heartbeat"] = asyncio.create_task(lifecycle.heartbeat_forever())

    async def close(app_: web.Application) -> None:
        for t in app_["worker_tasks"]:
            t.cancel()
        beat = background.pop("heartbeat", None)
        if beat is not None:
            beat.cancel()
        reqlog.close()

    def check_sessions(job: dict[str, Any]) -> None:
        # both rebuild the Generator, which would drop the jobs of the other sessions (without a drafter there is
        # no speculative mode to leave, so speculative=false changes nothing)
        if sessions > 1 and (not job["cache_prompt"] or (not job["speculative"] and getattr(engine, "spec_on", False))):
            raise BadRequest("speculative=false or cache_prompt=false needs a server started with --sessions 1")

    def salt_of(body: dict[str, Any]) -> str | None:
        salt = serve_compat.cache_salt(body)
        if salt is not None and not hasattr(engine, "reset_prefix_cache"):
            raise BadRequest("cache_salt is not supported by this engine")
        return salt

    def prepare(body: Any) -> dict[str, Any]:
        """Validate a request and build the engine job. Raises BadRequest."""
        if not isinstance(body, dict):
            raise BadRequest("body must be a JSON object")
        if body.get("model", model_id) != model_id:
            raise BadRequest(f"unknown model; expected {model_id}")
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise BadRequest("messages must be a non-empty array")
        if "stream" in body and not isinstance(body["stream"], bool):
            raise BadRequest("stream must be boolean")
        if body.get("tools") is not None and not isinstance(body["tools"], list):
            raise BadRequest("tools must be an array")
        messages = normalize_messages(body["messages"])
        kwargs = resolve_template_kwargs(body, defaults)
        try:
            prompt = render_prompt(template, messages, _escape_render_text(body.get("tools")),
                                   **_escape_render_text(kwargs))
        except (ValueError, TypeError, KeyError, AttributeError, jinja2.TemplateError) as exc:
            raise BadRequest(f"chat template: {exc}") from exc
        images = extract_images(messages)
        if images and not engine.supports_vision:
            raise BadRequest("this server runs without the vision tower (--no-vision)")
        if len(images) != prompt.count(IMAGE_PAD):
            raise BadRequest("image parts and image placeholders do not match (images are allowed in user messages only)")
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        if not all(isinstance(s, str) for s in stops):
            raise BadRequest("stop must be a string or a list of strings")
        prompt_tokens = int(engine.prompt_ids(prompt).numel())
        ctx = getattr(engine, "ctx", None)
        max_tokens = body.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = body.get("max_tokens")
        if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1):
            raise BadRequest("max_tokens must be a positive integer")
        room = reply_room(ctx, prompt_tokens, draft_window(engine, body.get("speculative", True) is not False)) if ctx else (default_max_tokens or NO_CTX_REPLY)
        if room < 1:
            raise BadRequest(f"context length exceeded: prompt is {prompt_tokens} tokens, the server context is {ctx}")
        max_tokens = min(max_tokens or default_max_tokens or room, room)
        seed = body.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise BadRequest("seed must be an integer")
        budget = body.get("thinking_budget")
        if budget is not None and (not isinstance(budget, int) or isinstance(budget, bool) or budget < 1):
            raise BadRequest("thinking_budget must be a positive integer")
        if body.get("loop_guard") is not None and not isinstance(body["loop_guard"], bool):
            raise BadRequest("loop_guard must be boolean")
        if body.get("loop_final") is not None and not isinstance(body["loop_final"], bool):
            raise BadRequest("loop_final must be boolean")
        lp = logprobs_arg(body.get("logprobs"), body.get("top_logprobs"), chat=True)
        if lp is not None and body.get("stream"):
            raise BadRequest("logprobs are returned on non-streaming requests only")
        job = {"logprobs": lp, "seed": seed, "thinking_budget": budget, "loop_guard": body.get("loop_guard"), "loop_final": body.get("loop_final"), "prompt": prompt, "prompt_tokens": prompt_tokens, "max_tokens": max_tokens, "messages": len(messages),
               "sampling": sampling_from(body, defaults), "stop": stops, "images": images,
               "speculative": body.get("speculative", True) is not False,
               "cache_prompt": body.get("cache_prompt", True) is not False,
               "reasoning_first": prompt.rstrip().endswith("<think>"), "tools": body.get("tools"),
               "return_token_ids": bool(body.get("return_token_ids")), "cancel": asyncio.Event()}
        check_sessions(job)
        job["cache_salt"] = salt_of(body)
        job["_reqid"] = None            # set by the route, which knows the request's log id
        job["_queued_at"] = None
        return job

    async def completions(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
            job = prepare(body)
        except json.JSONDecodeError as exc:
            return err(400, f"invalid JSON: {exc}")
        except (BadRequest, serve_compat.CompatError) as exc:
            return err(400, str(exc))
        request_id, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        base = {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model_id}

        def event(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
            return base | {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        out: asyncio.Queue = asyncio.Queue()
        job["_reqid"] = reqid()
        job["_queued_at"] = queued_at = time.perf_counter()   # the wait a streaming client's comments count from
        ahead = queue.qsize()                     # requests already waiting when this one arrived
        await queue.put((job, out))
        lifecycle.queued(job["_reqid"], "/v1/chat/completions", queued_ahead=ahead,
                         prompt_tokens=job.get("prompt_tokens"), messages=len(body.get("messages") or ()),
                         images=len(job.get("images") or ()), stream=bool(body.get("stream")))

        def finish_and_stats(stats: dict[str, Any], text: str, has_calls: bool):
            n_new = int(stats.get("new_tokens") or engine.count_tokens(text))
            prompt_tokens = int(stats.get("prompt_tokens") or job["prompt_tokens"])
            cache_n = int(stats.get("cached_tokens", 0))
            pre, gen = stats.get("time_prefill") or 0.0, stats.get("time_generate") or 0.0
            if has_calls:
                reason = "tool_calls"
            elif stats.get("eos_reason") in ("max_new_tokens", "loop_guard"):
                reason = "length"       # a guard stop cut the answer short: clients must not read it as a natural end
            else:
                reason = "stop"
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": n_new,
                     "total_tokens": prompt_tokens + n_new, "prompt_tokens_details": {"cached_tokens": cache_n}}
            timings = {"cache_n": cache_n, "prompt_n": max(prompt_tokens - cache_n, 0), "prompt_ms": pre * 1000,
                       "predicted_n": n_new, "predicted_ms": gen * 1000,
                       "prompt_per_second": (prompt_tokens - cache_n) / pre if pre else 0.0,
                       "predicted_per_second": n_new / gen if gen else 0.0,
                       "speculative": bool(getattr(engine, "spec_on", False))}
            if stats.get("loop_guard"):
                timings["loop_guard"] = stats["loop_guard"]
            if "accepted_draft_tokens" in stats:
                timings["accepted_draft_tokens"] = int(stats["accepted_draft_tokens"])
                timings["rejected_draft_tokens"] = int(stats["rejected_draft_tokens"])
            return reason, usage, timings

        async def pieces(writer: StreamWriter | None = None, ping: ServePing | None = None,
                         ping_at: float = 0.0) -> AsyncIterator[str | dict]:
            while True:
                # a non-streaming client that hung up is only noticed here (nothing is written until the end)
                tr = request.transport
                if tr is None or tr.is_closing():
                    job["cancel"].set()
                    raise ConnectionResetError("client left")
                if ping is not None:
                    # A ':' comment while the engine works, checked before every poll: the only thing a
                    # streaming client can see until the prompt (or a held-back tool call) is done. It is
                    # None for a JSON reply.
                    try:
                        await ping.comment(writer, ping_at)
                    except ConnectionResetError:
                        job["cancel"].set()
                        raise
                try:
                    item = await asyncio.wait_for(out.get(), 1.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, tuple):
                    yield item[1]
                    return
                yield item

        sp = Splitter(job["reasoning_first"])
        if body.get("stream", False):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await response.prepare(request)
            ping, ping_at = app["ping"].per_request(), queued_at
            writer = lifecycle.stream_started(job["_reqid"], "/v1/chat/completions", response,
                                              prompt_tokens=job.get("prompt_tokens"))
            try:
                await writer.write(sse(event({"role": "assistant", "content": ""})))
                ping.note_output()
                stats, full = {}, ""
                visible = False   # flipped at the first frame with text: the client's first sign of life
                async for item in pieces(writer, ping, ping_at):
                    if isinstance(item, dict):
                        stats = item
                        continue
                    if isinstance(item, Retract):  # loop guard restart: this client already received the text
                        if sp.mode != "reasoning":
                            # final check after the close: the discarded answer is gone for the model, and the restart text
                            # begins with a new close tag. A fresh Splitter in reasoning mode swallows that tag and also drops
                            # whatever the discarded answer left behind (an open or finished tool call, a held partial tag).
                            sp = Splitter(True)
                        continue
                    full += item
                    for kind, piece in sp.feed(item):
                        key = "reasoning_content" if kind == "reasoning" else "content"
                        if not visible:
                            visible = True
                            lifecycle.first_content(job["_reqid"], queued_at)
                        await writer.write(sse(event({key: piece})))
                        ping.note_output()   # the client has text: comments wait for the next silence
                for kind, piece in sp.finish():
                    key = "reasoning_content" if kind == "reasoning" else "content"
                    await writer.write(sse(event({key: piece})))
                calls = parse_tool_calls(sp.tool_text, job["tools"])
                if calls:
                    for i, c in enumerate(calls):   # one stream message per call: some clients read only the first call of a message
                        await writer.write(sse(event({"tool_calls": [dict(c, index=i)]})))
                reason, usage, timings = finish_and_stats(stats, full, bool(calls))
                final = event({}, reason) | {"usage": usage, "timings": timings}
                if stats.get("eos_reason") == "loop_guard":
                    final["choices"][0]["stop_reason"] = "loop_guard"
                if job["return_token_ids"]:
                    final["token_ids"] = stats.get("token_ids", [])
                await writer.write(sse(final))
                await writer.write(b"data: [DONE]\n\n")
                await response.write_eof()
                lifecycle.stream_done(job["_reqid"], int(stats.get("new_tokens") or 0))
            except ConnectionResetError:  # client left: stop the job, no traceback
                job["cancel"].set()
                lifecycle.client_left("/v1/chat/completions", "stream", rid=job["_reqid"])
            except asyncio.CancelledError:
                job["cancel"].set()
                raise
            except Exception as exc:                                  # noqa: BLE001
                job["cancel"].set()
                lifecycle.stream_error(job["_reqid"], exc)
                try:
                    await writer.write(sse({"error": {"message": f"{type(exc).__name__}: {exc}"}}))
                    # End the stream the OpenAI way: a bare drop makes clients report "stream ended without finish_reason".
                    await writer.write(sse(event({}, "error")))
                    await writer.write(b"data: [DONE]\n\n")
                    await response.write_eof()
                except ConnectionError:
                    pass
            return response

        try:
            stats, full = {}, ""
            async for item in pieces():
                if isinstance(item, dict):
                    stats = item
                elif isinstance(item, Retract):
                    full = full[:len(full) - int(item)]
                else:
                    full += item
        except ConnectionResetError:
            return web.Response(status=499)                           # client left; the job is cancelled
        except BadRequest as exc:                                     # e.g. an image that does not decode (loaded by the worker)
            return err(400, str(exc))
        except Exception as exc:                                      # noqa: BLE001
            return web.json_response({"error": {"message": f"{type(exc).__name__}: {exc}", "type": "server_error"}},
                                     status=500)
        text, _ = stop_text(full, job["stop"])
        message = split_completion(text, job["reasoning_first"], job["tools"])
        reason, usage, timings = finish_and_stats(stats, text, "tool_calls" in message)
        payload = {"id": request_id, "object": "chat.completion", "created": created, "model": model_id,
                   "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                   "usage": usage, "timings": timings}
        if stats.get("eos_reason") == "loop_guard":
            payload["choices"][0]["stop_reason"] = "loop_guard"
        if job["return_token_ids"]:
            payload["token_ids"] = stats.get("token_ids", [])
        if job["logprobs"] is not None:
            payload["choices"][0]["logprobs"] = format_logprobs(
                stats.get("logprobs", []), stats.get("logprob_token_ids", stats.get("token_ids", [])), engine.token_str, chat=True)
        # The worker logs job_done for this job as well (a JSON reply takes the same path as a stream).
        return web.json_response(payload)

    async def apply_template(request: web.Request) -> web.Response:
        try:
            body = await request.json()
            if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
                raise BadRequest("messages must be a non-empty array")
            prompt = render_prompt(template, normalize_messages(body["messages"]),
                                   _escape_render_text(body.get("tools")),
                                   **_escape_render_text(resolve_template_kwargs(body, defaults)))
        except json.JSONDecodeError as exc:
            return err(400, f"invalid JSON: {exc}")
        except (BadRequest, ValueError, TypeError, KeyError, AttributeError, jinja2.TemplateError) as exc:
            return err(400, str(exc))
        return web.json_response({"prompt": prompt})

    async def tokenize(request: web.Request) -> web.Response:
        """vLLM POST /tokenize: {prompt | messages, add_special_tokens?} -> {tokens, count, max_model_len}."""
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise BadRequest("body must be a JSON object")
            if body.get("model", model_id) != model_id:
                raise BadRequest(f"unknown model; expected {model_id}")
            if not hasattr(engine, "encode_ids"):
                return err(501, "this engine cannot tokenize")
            add_special = body.get("add_special_tokens")
            if add_special is not None and not isinstance(add_special, bool):
                raise BadRequest("add_special_tokens must be boolean")
            if body.get("messages") is not None:
                if not isinstance(body["messages"], list) or not body["messages"]:
                    raise BadRequest("messages must be a non-empty array")
                if extract_images(body["messages"]):
                    raise BadRequest("images cannot be tokenized here")
                text = render_prompt(template, normalize_messages(body["messages"]),
                                     _escape_render_text(body.get("tools")),
                                     **_escape_render_text(resolve_template_kwargs(body, defaults)))
                add_special = bool(add_special)          # a rendered chat prompt carries its own special tokens
            else:
                text = body.get("prompt")
                if not isinstance(text, str):
                    raise BadRequest("prompt must be a string (or send messages)")
                add_special = True if add_special is None else add_special
            ids = engine.encode_ids(text, add_special)
        except json.JSONDecodeError as exc:
            return err(400, f"invalid JSON: {exc}")
        except (BadRequest, ValueError, TypeError, KeyError, AttributeError, jinja2.TemplateError) as exc:
            return err(400, str(exc))
        return web.json_response(serve_compat.tokenize_reply(ids, getattr(engine, "ctx", None)))

    async def completion(request: web.Request) -> web.Response:
        """Minimal llama.cpp POST /completion: prompt used verbatim (no template), same queue and generator."""
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise BadRequest("body must be a JSON object")
            prompt = body.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise BadRequest("prompt must be a non-empty string")
            n_predict = body.get("n_predict", 128)
            if n_predict is not None and (not isinstance(n_predict, int) or isinstance(n_predict, bool)):
                raise BadRequest("n_predict must be an integer")
            n_predict = n_predict if n_predict and n_predict > 0 else 128
            n_predict = min(n_predict, 8192)
            stops = body.get("stop") or []
            stops = [stops] if isinstance(stops, str) else stops
            if not isinstance(stops, list) or not all(isinstance(x, str) for x in stops):
                raise BadRequest("stop must be a string or a list of strings")
            ctx = getattr(engine, "ctx", None)
            if ctx:
                room = reply_room(ctx, engine.count_tokens(prompt), draft_window(engine))
                if room < 1:
                    raise BadRequest(f"context length exceeded: prompt is longer than the server context ({ctx})")
                n_predict = min(n_predict, room)
            job = {"_reqid": reqid(), "_queued_at": time.perf_counter(), "messages": 0,
                   "prompt": prompt, "max_tokens": n_predict,
                   "sampling": sampling_from({k: body.get(k) for k in ("temperature", "top_p", "top_k", "min_p")}, defaults),
                   "stop": stops, "images": [], "speculative": True,
                   "seed": body.get("seed") if isinstance(body.get("seed"), int) and not isinstance(body.get("seed"), bool) else None,
                   "cache_prompt": body.get("cache_prompt", True) is not False, "cancel": asyncio.Event()}
            check_sessions(job)
        except json.JSONDecodeError as exc:
            return err(400, f"invalid JSON: {exc}")
        except (BadRequest, TypeError, ValueError) as exc:
            return err(400, str(exc))
        out: asyncio.Queue = asyncio.Queue()
        ahead = queue.qsize()
        await queue.put((job, out))
        lifecycle.queued(job["_reqid"], "/completion", queued_ahead=ahead, messages=0, images=0,
                         stream=False)
        text, stats = "", {}
        while True:
            item = await out.get()
            if isinstance(item, Exception):
                return web.json_response({"error": {"message": f"{type(item).__name__}: {item}", "type": "server_error"}},
                                         status=500)
            if isinstance(item, tuple):
                stats = item[1]
                break
            text += item
        text, stopped = stop_text(text, job["stop"])
        return web.json_response({
            "id": f"cmpl-{uuid.uuid4().hex}", "object": "completion", "created": int(time.time()), "model": model_id,
            "content": text, "tokens_cached": int(stats.get("cached_tokens", 0)),
            "tokens_predicted": int(stats.get("new_tokens", 0)), "cache_prompt": job["cache_prompt"],
            "stop": True, "stopped_eos": stats.get("eos_reason") == "stop_token",
            "stopped_limit": stats.get("eos_reason") == "max_new_tokens", "stopped_word": stopped,
            "timings": {"prompt_n": int(stats.get("prompt_tokens", 0)) - int(stats.get("cached_tokens", 0)),
                        "predicted_n": int(stats.get("new_tokens", 0))}})

    async def v1_completions(request: web.Request) -> web.StreamResponse:
        """OpenAI POST /v1/completions: the prompt is used verbatim (no chat template), one prompt, one choice."""
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise BadRequest("body must be a JSON object")
            if body.get("model", model_id) != model_id:
                raise BadRequest(f"unknown model; expected {model_id}")
            prompt = body.get("prompt")
            prompt_ids = serve_compat.token_prompt(prompt, getattr(engine, "vocab_size", None))
            if prompt_ids is not None:
                prompt = ""
            else:
                if isinstance(prompt, list) and len(prompt) == 1:
                    prompt = prompt[0]
                if not isinstance(prompt, str) or not prompt:
                    raise BadRequest("prompt must be a non-empty string or a list of token ids (one prompt per request)")
            if body.get("ignore_eos") is not None and not isinstance(body["ignore_eos"], bool):
                raise BadRequest("ignore_eos must be boolean")
            if body.get("n") not in (None, 1) or body.get("echo") or body.get("best_of") not in (None, 1):
                raise BadRequest("n, best_of and echo are not supported")
            if "stream" in body and not isinstance(body["stream"], bool):
                raise BadRequest("stream must be boolean")
            max_tokens = body.get("max_tokens")
            if max_tokens is None:
                max_tokens = 16
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
                raise BadRequest("max_tokens must be a positive integer")
            stops = body.get("stop") or []
            stops = [stops] if isinstance(stops, str) else stops
            if not isinstance(stops, list) or not all(isinstance(x, str) for x in stops):
                raise BadRequest("stop must be a string or a list of strings")
            seed = body.get("seed")
            if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
                raise BadRequest("seed must be an integer")
            if body.get("loop_guard") is not None and not isinstance(body["loop_guard"], bool):
                raise BadRequest("loop_guard must be boolean")
            lp = logprobs_arg(body.get("logprobs"))
            if lp is not None and body.get("stream"):
                raise BadRequest("logprobs are returned on non-streaming requests only")
            prompt_tokens = len(prompt_ids) if prompt_ids is not None else engine.count_tokens(prompt)
            ctx = getattr(engine, "ctx", None)
            if ctx:
                room = reply_room(ctx, prompt_tokens, draft_window(engine, body.get("speculative", True) is not False))
                if room < 1:
                    raise BadRequest(f"context length exceeded: prompt is {prompt_tokens} tokens, the server context is {ctx}")
                max_tokens = min(max_tokens, room)
            job = {"_reqid": reqid(), "_queued_at": time.perf_counter(), "messages": 0,
                   "cache_salt": salt_of(body), "prompt_ids": prompt_ids, "ignore_eos": body.get("ignore_eos") is True,
                   "logprobs": lp, "prompt": prompt, "max_tokens": max_tokens, "stop": stops, "images": [], "speculative": body.get("speculative", True) is not False, "seed": seed,
                   "sampling": sampling_from(body, defaults), "cache_prompt": body.get("cache_prompt", True) is not False,
                   "loop_guard": body.get("loop_guard"), "cancel": asyncio.Event()}
            check_sessions(job)
        except json.JSONDecodeError as exc:
            return err(400, f"invalid JSON: {exc}")
        except (BadRequest, serve_compat.CompatError) as exc:
            return err(400, str(exc))
        request_id, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())

        def chunk(text: str, finish: str | None) -> dict[str, Any]:
            return {"id": request_id, "object": "text_completion", "created": created, "model": model_id,
                    "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish}]}

        def reason_usage(stats: dict[str, Any], text: str):
            n_new = int(stats.get("new_tokens") or engine.count_tokens(text))
            n_prompt = prompt_tokens if prompt_ids is not None else int(stats.get("prompt_tokens") or prompt_tokens)
            reason = "length" if stats.get("eos_reason") in ("max_new_tokens", "loop_guard") else "stop"
            return reason, serve_compat.usage(n_prompt, n_new, int(stats.get("cached_tokens") or 0))

        out: asyncio.Queue = asyncio.Queue()
        job["_reqid"] = reqid()
        job["_queued_at"] = queued_at = time.perf_counter()   # the wait a streaming client's comments count from
        ahead = queue.qsize()
        await queue.put((job, out))
        lifecycle.queued(job["_reqid"], "/v1/completions", queued_ahead=ahead,
                         prompt_tokens=job.get("prompt_tokens"), messages=0, images=0,
                         stream=bool(body.get("stream")))

        async def pieces(writer: StreamWriter | None = None, ping: ServePing | None = None,
                         ping_at: float = 0.0) -> AsyncIterator[str | dict]:
            while True:
                tr = request.transport
                if tr is None or tr.is_closing():
                    job["cancel"].set()
                    lifecycle.client_left("/v1/completions", "waiting", rid=job.get("_reqid"))
                    raise ConnectionResetError("client left")
                if ping is not None:
                    # A ':' comment while the engine works, checked before every poll: the only thing a
                    # streaming client can see until the prompt (or a held-back tool call) is done.
                    try:
                        await ping.comment(writer, ping_at)
                    except ConnectionResetError:
                        job["cancel"].set()
                        raise
                try:
                    item = await asyncio.wait_for(out.get(), 1.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, tuple):
                    yield item[1]
                    return
                yield item

        if body.get("stream", False):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await response.prepare(request)
            ping, ping_at = app["ping"].per_request(), queued_at
            writer = lifecycle.stream_started(job["_reqid"], "/v1/completions", response,
                                              prompt_tokens=job.get("prompt_tokens"))
            try:
                stats, full, visible = {}, "", False
                async for item in pieces(writer, ping, ping_at):
                    if isinstance(item, dict):
                        stats = item
                    elif not isinstance(item, Retract):
                        full += item
                        if not visible:
                            visible = True
                            lifecycle.first_content(job["_reqid"], queued_at)
                        await writer.write(sse(chunk(item, None)))
                reason, usage = reason_usage(stats, full)
                last = chunk("", reason) | {"usage": usage}
                if stats.get("eos_reason") == "loop_guard":
                    last["choices"][0]["stop_reason"] = "loop_guard"
                await writer.write(sse(last))
                await writer.write(b"data: [DONE]\n\n")
                await response.write_eof()
                lifecycle.stream_done(job["_reqid"], int(stats.get("new_tokens") or 0))
            except ConnectionResetError:
                job["cancel"].set()
                lifecycle.client_left("/v1/completions", "stream", rid=job["_reqid"])
            except asyncio.CancelledError:
                job["cancel"].set()
                raise
            except Exception as exc:                                  # noqa: BLE001
                job["cancel"].set()
                lifecycle.stream_error(job["_reqid"], exc)
                try:
                    await writer.write(sse({"error": {"message": f"{type(exc).__name__}: {exc}"}}))
                    await response.write_eof()
                except ConnectionError:
                    pass
            return response

        try:
            stats, full = {}, ""
            async for item in pieces():
                if isinstance(item, dict):
                    stats = item
                elif isinstance(item, Retract):
                    full = full[:len(full) - int(item)]
                else:
                    full += item
        except ConnectionResetError:
            return web.Response(status=499)
        except BadRequest as exc:
            return err(400, str(exc))
        except Exception as exc:                                      # noqa: BLE001
            return web.json_response({"error": {"message": f"{type(exc).__name__}: {exc}", "type": "server_error"}}, status=500)
        text, _ = stop_text(full, job["stop"])
        reason, usage = reason_usage(stats, text)
        out_ = chunk(text, reason) | {"object": "text_completion", "usage": usage}
        if stats.get("eos_reason") == "loop_guard":
            out_["choices"][0]["stop_reason"] = "loop_guard"
        if job["logprobs"] is not None:
            out_["choices"][0]["logprobs"] = format_logprobs(
                stats.get("logprobs", []), stats.get("logprob_token_ids", stats.get("token_ids", [])), engine.token_str, chat=False)
        return web.json_response(out_)

    async def slots(_: web.Request) -> web.Response:
        return web.json_response(engine.slot_store.slots())

    async def slot_action(request: web.Request) -> web.Response:
        """llama.cpp POST /slots/0?action=save|restore|erase (same SlotStore as the MiMo server)."""
        store = engine.slot_store
        if request.match_info["id"] != "0":
            return err(400, "only slot 0 exists")
        action = request.query.get("action")
        try:
            body = await request.json() if request.can_read_body else {}
            if action not in ("save", "restore", "erase"):
                return err(400, "action must be save, restore or erase")
            async with lock:
                held = 0
                try:
                    for _ in range(sessions):      # wait for the running requests to finish
                        await gate.acquire()
                        held += 1
                    fn = store.erase if action == "erase" else partial(getattr(store, action), body.get("filename", ""))
                    if hasattr(engine, "run_exclusive"):
                        return web.json_response(await engine.run_exclusive(fn))
                    return web.json_response(await asyncio.get_running_loop().run_in_executor(None, fn))
                finally:
                    for _ in range(held):
                        gate.release()
        except (ValueError, FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
            return err(400, f"{type(exc).__name__}: {exc}")

    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", lifecycle.wrapped(completions))
    app.router.add_post("/apply-template", apply_template)
    app.router.add_post("/tokenize", tokenize)
    app.router.add_post("/completion", lifecycle.wrapped(completion))
    app.router.add_post("/v1/completions", lifecycle.wrapped(v1_completions))
    if getattr(engine, "slot_store", None) is not None:
        app.router.add_get("/slots", slots)
        app.router.add_post("/slots/{id}", slot_action)
    app.on_startup.append(start)
    app.on_cleanup.append(close)
    return app


# ---------------------------------------------------------------------------- main


def load_all(args: argparse.Namespace):
    """Everything before the port opens: template, defaults, engine, slot store, warm-up."""
    model_dir = str(Path(args.model).expanduser())
    template = Path(args.chat_template or DEFAULT_TEMPLATE).expanduser().read_text(encoding="utf-8")
    defaults = load_defaults(model_dir, args)
    t0 = time.time()
    print(f"qserve: starting, model={model_dir} ctx={args.ctx} sessions={args.sessions} draft_policy={args.draft_policy} "
          f"defaults={ {k: v for k, v in defaults.items() if v is not None} }", flush=True)
    engine = QwenEngine(model_dir, args.ctx, ndt=args.ndt, draft_policy=args.draft_policy, vision=not args.no_vision,
                        cache_bits=args.cache_bits, sessions=args.sessions,
                        k_bits=args.k_bits, v_bits=args.v_bits)
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("EXL3_", "MPW"))}
    print(f"qserve: effective env {env}", flush=True)
    from exllamav3.generator.slot_store import SlotStore
    engine.slot_store = SlotStore(engine.generator, [c for c in (engine.cache, engine.draft_cache) if c is not None],
                                  args.slot_save_path, args.model_id)
    from exllamav3.util.disk_store import env_dir
    if env_dir():
        from exllamav3.generator.prefix_disk import PagedPrefixDisk, weights_identity
        engine.prefix_disk = PagedPrefixDisk(engine.slot_store, model_id=args.model_id, extra=weights_identity(args.model))
        print(f"qserve: prefix disk {engine.prefix_disk.disk.dir} ({len(engine.prefix_disk.pages)} pages, "
              f"{len(engine.prefix_disk.stashes)} checkpoints, cap {engine.prefix_disk.disk.cap >> 30} GiB)", flush=True)
    startup_health.stage("warm-up")
    with Heartbeat("warm-up (first decode steps)"):
        engine.warmup()
    if os.environ.get("EXL3_SWAP_IN", "1") != "0":
        from exllamav3.util.memory import swap_in_process
        print(f"qserve: process pages swapped out during the load, read back before READY: {swap_in_process()}", flush=True)
    return engine, template, defaults, model_dir, t0


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="pack directory")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="id shown by /v1/models and expected in requests")
    parser.add_argument("--ctx", "-c", type=int, default=65536, help="KV cache size in tokens (default 65536)")
    parser.add_argument("--cache-bits", type=int, default=0, choices=(0, 8),
                        help="bits per K/V element of the attention cache pages (0 = fp16, the default; 8 = packed int8, about 40 %% smaller pages, saves 2.6 GiB at 256K, greedy output differs from the 16-bit cache; with EXL3_ROW_INV_GROUPS=1 four parallel sessions matched solo runs, 16 of 16)")
    parser.add_argument("--k-bits", type=int, default=0, choices=(0, 2, 3, 4, 5, 6, 7, 8), help="K bits of the quantized attention cache (with --v-bits; overrides --cache-bits)")
    parser.add_argument("--v-bits", type=int, default=0, choices=(0, 2, 3, 4, 5, 6, 7, 8), help="V bits of the quantized attention cache")
    parser.add_argument("--ndt", type=int, default=7, help="max draft tokens per round (default 7; the rule drafts up to 3 for most rounds, deeper only on confident chains)")
    parser.add_argument("--draft-policy", choices=("mix", "mtp", "off"), default="mix",
                        help="mix = shipped rule (MTP + n-gram lookup, lossless), mtp = fixed MTP chain, off = plain decode")
    parser.add_argument("--no-vision", action="store_true", help="do not load the vision tower (saves memory)")
    parser.add_argument("--no-uncensor", action="store_true", help="ignore a bundled uncensor_spec.json in the model directory (same as EXL3_ABLIT_RUNTIME=off)")
    parser.add_argument("--default-temperature", type=float, default=None, help="when a request omits it (default: generation_config.json)")
    parser.add_argument("--default-top-p", type=float, default=None)
    parser.add_argument("--default-top-k", type=int, default=None)
    parser.add_argument("--default-min-p", type=float, default=None)
    parser.add_argument("--chat-template", default=None,
                        help="Jinja chat template file (default: chat_template.jinja next to this script; "
                             "<model>/chat_template.jinja is the model's unchanged one)")
    parser.add_argument("--default-reasoning-effort", choices=("low", "medium", "xhigh"), default=None,
                        help=f"thinking effort when a request omits it (default: {DEFAULT_EFFORT})")
    parser.add_argument("--no-thinking", action="store_true", help="thinking off unless a request turns it on")
    parser.add_argument("--default-max-tokens", type=int, default=None,
                        help="reply budget when a request omits max_tokens (default: all the free context)")
    parser.add_argument("--slot-save-path", default="~/cache/llama-slots")
    parser.add_argument("--sessions", type=int, default=1,
                        help="requests decoded together (default 1: one at a time, in order). Every session reserves the "
                             "pages for its prompt + reply budget when it starts, so a request that does not fit waits. A reply longer than "
                             "its share of the context (-c / sessions) is requeued, not reserved up front")
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("--sessions must be at least 1")
    # Measured with --ndt 3: greedy output equal to a solo run on two short prompts at 2 sessions with --cache-bits 0; not at
    # 4 sessions, and not at 2 sessions with the 8-bit cache. Equality at N > 1 is not guaranteed by default: the split layout of
    # the decode attention kernel depends on the batch. EXL3_ROW_INV_GROUPS=1 (opt-in, generator/row_groups.py) packs the decode
    # step into forwards of at most 8 token rows with one attention call per sequence: four cold 8K prompts, --ndt 7, --sessions 4
    # gave 16 of 16 answers equal to the solo run (16-bit and 8-bit cache), at about 3 % more wall time on that load and 8 to 10 %
    # less aggregate decode throughput on short prompts
    exact = all(os.environ.get(k, d) not in ("", "0") for k, d in (("EXL3_ROW_INV_GROUPS", "0"), ("EXL3_ROW_INV", "1")))
    if args.sessions > 1 and not exact:
        if args.draft_policy != "off" and args.sessions * (args.ndt + 1) > 8:
            print(f"qserve: WARNING --sessions {args.sessions} x (--ndt {args.ndt} + 1) = {args.sessions * (args.ndt + 1)} verify rows "
                  "is past 8: greedy output may differ from the same request run alone", flush=True)
        print(f"qserve: WARNING --sessions {args.sessions}: greedy output is not guaranteed equal to the same request run alone, "
              "with either cache width (the attention kernel splits its work by batch; EXL3_ROW_INV_GROUPS=1 makes it exact, a "
              "little slower)", flush=True)
    if os.environ.get("EXL3_SCHED_CHUNK_IL", "0") == "1":
        # chunked-prefill interleave (exllamav3/generator/chunk_il.py): needs another request running beside the big prefill
        print("qserve: chunk_il ON " + ("" if args.sessions > 1 else "(no effect: needs --sessions >= 2) ") +
              f"max_defer={os.environ.get('EXL3_CIL_MAX_DEFER_S', '8')}s frac={os.environ.get('EXL3_CIL_DEFER_FRAC', '0.18')} "
              f"slice={os.environ.get('EXL3_CIL_SLICE_S', '0.6')}s short_tok={os.environ.get('EXL3_CIL_SHORT_TOK', '4096')}", flush=True)

    if args.no_uncensor:
        os.environ["EXL3_ABLIT_RUNTIME"] = "off"
    # /health answers 503 with the load progress from here until READY (stdlib listener on the same port).
    stages = ["target weights"] + (["drafter weights"] if args.draft_policy != "off" else []) \
        + (["vision tower"] if not args.no_vision else []) + ["engine setup", "warm-up"]
    startup_health.check_model_dir("qserve", args.model)
    startup_health.start("qwen", stages, args.host, args.port, routes=ROUTES)
    startup_health.report_compiler()
    try:
        engine, template, defaults, model_dir, t0 = load_all(args)
    except BaseException as exc:                                  # noqa: BLE001
        startup_health.abort(f"{type(exc).__name__}: {exc}")
        raise
    app = create_app(engine, args.model_id, template, defaults, args.default_max_tokens)
    print(f"qserve: {app['ping'].describe()}", flush=True)
    print(f"qserve: {app['reqlog'].describe()}", flush=True)

    async def serve() -> None:
        runner = web.AppRunner(app)
        await runner.setup()
        sock = startup_health.finish()                            # the early listener's open socket: no refused connection
        site = web.SockSite(runner, sock) if sock is not None else web.TCPSite(runner, args.host, args.port)
        await site.start()
        print(f"qserve: READY on http://{args.host}:{args.port}  model={args.model_id} ctx={args.ctx} sessions={args.sessions} "
              f"speculative={engine.spec_on} vision={engine.supports_vision} (start-up {time.time() - t0:.0f} s)",
              flush=True)
        await asyncio.Event().wait()

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
