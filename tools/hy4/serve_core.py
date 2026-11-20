"""Pure-Python helpers of the Hy4 OpenAI-compatible server (no torch, no engine import: unit-testable on any machine).

  * request parsing for POST /v1/chat/completions and /v1/completions (RequestError = HTTP 400)
  * the chat template: the Jinja source stored in the GGUF (tokenizer.chat_template), rendered like the reference runtimes do
  * the tokenizer, rebuilt from the GGUF vocabulary + merges (special tokens spelled exactly as in the GGUF)
  * the output side: incremental detokenisation (UTF-8 safe), stop strings, EOS, reasoning / content split
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Iterable

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

PRETOK_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hy4_pretokenizer.json")

# OpenAI-style effort names -> the values the template accepts (max / high / low / no_think)
EFFORT_MAP = {"max": "max", "xhigh": "max", "high": "high", "medium": "high", "low": "low", "minimal": "low",
              "no_think": "no_think", "none": "no_think", "off": "no_think"}
DEFAULT_EFFORT = "high"


class RequestError(ValueError):
    """The request is malformed or cannot be served: the server answers `status` with an OpenAI error body."""

    def __init__(self, message: str, status: int = 400, code: str = "invalid_request_error"):
        super().__init__(message)
        self.status, self.code = status, code


# ------------------------------------------------------------------------------------------------ chat template
def _tojson(value: Any, ensure_ascii: bool = True, indent: int | None = None, sort_keys: bool = False, separators=None) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys, separators=separators)


def _raise_exception(message: str):
    raise jinja2.exceptions.TemplateError(message)


class ChatTemplate:
    """The GGUF chat template. Same environment as the reference runtimes: sandboxed, trim_blocks + lstrip_blocks, a tojson that takes
    ensure_ascii, raise_exception, loop controls."""

    def __init__(self, source: str):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise_exception
        env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
        self.source = source
        self.tpl = env.from_string(source)

    def render(self, messages: list[dict], tools: list | None = None, add_generation_prompt: bool = True, **kwargs: Any) -> str:
        try:
            return self.tpl.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kwargs)
        except jinja2.exceptions.TemplateError as e:
            raise RequestError(f"chat template: {e}") from e


# ------------------------------------------------------------------------------------------------ tokenizer
def build_tokenizer(tokens: list[str], merges: list[str], token_types: list[int], pretok_path: str = PRETOK_JSON):
    """tokenizers.Tokenizer for the byte-level BPE of the GGUF (tokenizer.ggml.model = gpt2, pre = hy_v4). Normal tokens (type 1) and merges come from the
    GGUF; every other token (control / user-defined) is registered as a special token under its GGUF spelling, so the template's
    `<｜hy_start:...｜>` markers encode to ONE id. The pre-tokenizer regexes are not stored in a GGUF: they come from hy4_pretokenizer.json."""
    from tokenizers import Tokenizer
    with open(pretok_path, encoding="utf-8") as f:
        extra = json.load(f)
    vocab = {t: i for i, t in enumerate(tokens) if token_types[i] == 1}
    spec = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
            for i, t in enumerate(tokens) if token_types[i] != 1]
    doc = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": spec, "normalizer": extra["normalizer"],
           "pre_tokenizer": extra["pre_tokenizer"], "post_processor": extra["post_processor"], "decoder": extra["decoder"],
           "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None, "end_of_word_suffix": None,
                     "fuse_unk": False, "byte_fallback": False, "ignore_merges": False, "vocab": vocab,
                     "merges": [m if isinstance(m, str) else " ".join(m) for m in merges]}}
    return Tokenizer.from_str(json.dumps(doc))


class Vocab:
    """tokenizer + the ids / markers the server needs, all derived from the GGUF metadata dict."""

    def __init__(self, meta: dict, pretok_path: str = PRETOK_JSON):
        toks = list(meta["tokenizer.ggml.tokens"])
        self.tokens = toks
        self.tok = build_tokenizer(toks, list(meta["tokenizer.ggml.merges"]), list(meta["tokenizer.ggml.token_type"]), pretok_path)
        self.size = len(toks)
        self.eos_id = int(meta["tokenizer.ggml.eos_token_id"])
        self.eos_ids = {self.eos_id}
        self.template = ChatTemplate(meta["tokenizer.chat_template"])
        self.think_begin = next((t for t in toks if t.startswith("<think")), None)
        self.think_end = next((t for t in toks if t.startswith("</think")), None)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)


# ------------------------------------------------------------------------------------------------ request parsing
def _as_int(body: dict, key: str, default: int | None, lo: int = 0) -> int | None:
    v = body.get(key, default)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or int(v) < lo:
        raise RequestError(f"{key} must be an integer >= {lo}")
    return int(v)


def _stops(body: dict) -> list[str]:
    s = body.get("stop")
    if s is None:
        return []
    if isinstance(s, str):
        s = [s]
    if not isinstance(s, list) or len(s) > 8 or not all(isinstance(x, str) and x for x in s):
        raise RequestError("stop must be a non-empty string or a list of at most 8 non-empty strings")
    return list(s)


def _common(body: dict, default_max: int) -> dict:
    if not isinstance(body, dict):
        raise RequestError("the body must be a JSON object")
    if _as_int(body, "n", 1, 1) != 1:
        raise RequestError("n must be 1")
    mt = body.get("max_completion_tokens", body.get("max_tokens"))
    max_tokens = default_max if mt is None else _as_int({"max_tokens": mt}, "max_tokens", default_max, 1)
    for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
        v = body.get(k)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
            raise RequestError(f"{k} must be a number")
    return {"max_tokens": max_tokens, "stream": bool(body.get("stream", False)), "stop": _stops(body),
            "include_usage": bool((body.get("stream_options") or {}).get("include_usage", False)),
            # the speculative decoder is greedy: sampling parameters are accepted and ignored; reported back in the reply
            "sampling_ignored": body.get("temperature") not in (None, 0, 0.0),
            "speculative": body.get("speculative", True) is not False}


def parse_effort(body: dict) -> str:
    kw = body.get("chat_template_kwargs") or {}
    if not isinstance(kw, dict):
        raise RequestError("chat_template_kwargs must be an object")
    v = kw.get("reasoning_effort", body.get("reasoning_effort"))
    if v is None:
        return DEFAULT_EFFORT
    if kw.get("enable_thinking") is False and "reasoning_effort" not in kw and body.get("reasoning_effort") is None:
        return "no_think"
    if not isinstance(v, str) or v not in EFFORT_MAP:
        raise RequestError(f"reasoning_effort must be one of {sorted(EFFORT_MAP)}")
    return EFFORT_MAP[v]


def _content_text(c: Any) -> str:
    """OpenAI content (string, or a list of {type: text} parts) -> text"""
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for p in c:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                out.append(p["text"])
            else:
                raise RequestError("only text content parts are supported (no images / audio)")
        return "".join(out)
    raise RequestError("message content must be a string or a list of text parts")


def parse_chat(body: dict, default_max: int = 1024) -> dict:
    """-> {messages, effort, max_tokens, stream, stop, include_usage, speculative, ...}. Raises RequestError."""
    req = _common(body, default_max)
    if body.get("tools") or body.get("functions") or body.get("tool_choice") not in (None, "none"):
        raise RequestError("tool calling is not supported by this backend yet")
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise RequestError("messages must be a non-empty list")
    out = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in ("system", "developer", "user", "assistant"):
            raise RequestError(f"messages[{i}]: role must be system, user or assistant")
        role = "system" if m["role"] == "developer" else m["role"]
        if role == "system" and i != 0:
            raise RequestError(f"messages[{i}]: a system message is only allowed first")
        item = {"role": role, "content": _content_text(m.get("content"))}
        rc = m.get("reasoning_content", m.get("reasoning"))
        if role == "assistant" and isinstance(rc, str) and rc:
            item["reasoning_content"] = rc
        out.append(item)
    if out[-1]["role"] == "assistant":
        raise RequestError("the last message must come from the user")
    req.update(messages=out, effort=parse_effort(body))
    return req


def parse_completion(body: dict, vocab_size: int | None, default_max: int = 256) -> dict:
    """-> {prompt: str | None, prompt_ids: list[int] | None, ...}"""
    req = _common(body, default_max)
    p = body.get("prompt")
    if isinstance(p, list) and len(p) == 1 and isinstance(p[0], (str, list)):
        p = p[0]
    if isinstance(p, str):
        req.update(prompt=p, prompt_ids=None)
    elif isinstance(p, list) and p and all(isinstance(x, int) and not isinstance(x, bool) for x in p):
        for x in p:
            if x < 0 or (vocab_size is not None and x >= vocab_size):
                raise RequestError(f"prompt token id {x} is outside the vocabulary")
        req.update(prompt=None, prompt_ids=list(p))
    else:
        raise RequestError("prompt must be a string or a list of token ids (one prompt per request)")
    if req["prompt"] == "":
        raise RequestError("prompt is empty")
    return req


# ------------------------------------------------------------------------------------------------ output side
class IncrementalDecoder:
    """token ids -> text deltas that never end in half a UTF-8 character (the standard prefix / read offset scheme)."""

    def __init__(self, vocab: Vocab):
        self.v, self.ids, self.prefix, self.read = vocab, [], 0, 0

    def push(self, ids: Iterable[int]) -> str:
        self.ids.extend(ids)
        old = self.v.decode(self.ids[self.prefix:self.read])
        new = self.v.decode(self.ids[self.prefix:])
        if len(new) > len(old) and not new.endswith("�"):
            self.prefix, self.read = self.read, len(self.ids)
            return new[len(old):]
        return ""

    def flush(self) -> str:
        old = self.v.decode(self.ids[self.prefix:self.read])
        new = self.v.decode(self.ids[self.prefix:])
        self.prefix = self.read = len(self.ids)
        return new[len(old):] if len(new) > len(old) else ""


def _hold(text: str, markers: list[str]) -> int:
    """length of the longest suffix of `text` that is a proper prefix of one of the markers (must wait for more text)"""
    best = 0
    for m in markers:
        for n in range(min(len(m) - 1, len(text)), best, -1):
            if text.endswith(m[:n]):
                best = n
                break
    return best


class OutputStream:
    """Per request: token ids in, (channel, text) deltas out, channel = "reasoning" | "content".
    in_think: the prompt ended inside an open think block, so the text starts as reasoning until `think_end`.
    Stops on an EOS id (dropped) or a stop string (cut; only searched in the content channel)."""

    def __init__(self, vocab: Vocab, in_think: bool = False, stops: list[str] | None = None):
        self.v = vocab
        self.dec = IncrementalDecoder(vocab)
        self.in_think = bool(in_think and vocab.think_end)
        self.stops = stops or []
        self.buf = ""
        self.done = False
        self.finish_reason: str | None = None
        self.n_tokens = 0          # generated tokens counted (EOS and anything after it excluded)

    def _drain(self, final: bool) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        while True:
            if self.in_think:
                i = self.buf.find(self.v.think_end)
                if i >= 0:
                    if i:
                        out.append(("reasoning", self.buf[:i]))
                    self.buf = self.buf[i + len(self.v.think_end):]
                    self.in_think = False
                    continue
                k = 0 if final else _hold(self.buf, [self.v.think_end])
                if len(self.buf) > k:
                    out.append(("reasoning", self.buf[:len(self.buf) - k]))
                    self.buf = self.buf[len(self.buf) - k:]
                break
            cut = min((j for j in (self.buf.find(s) for s in self.stops) if j >= 0), default=-1)
            if cut >= 0:
                if cut:
                    out.append(("content", self.buf[:cut]))
                self.buf, self.done, self.finish_reason = "", True, "stop"
                break
            k = 0 if final else _hold(self.buf, self.stops)
            if len(self.buf) > k:
                out.append(("content", self.buf[:len(self.buf) - k]))
                self.buf = self.buf[len(self.buf) - k:]
            break
        return out

    def push(self, ids: list[int]) -> list[tuple[str, str]]:
        """feed the tokens of one decode round; returns the deltas that are safe to send. `done` turns True on EOS or a stop string."""
        if self.done:
            return []
        take, eos = [], False
        for t in ids:
            if t in self.v.eos_ids:
                eos = True
                break
            take.append(t)
        self.n_tokens += len(take)
        self.buf += self.dec.push(take)
        if eos:
            self.buf += self.dec.flush()
        out = self._drain(final=eos)
        if eos:
            self.done, self.finish_reason = True, "stop"
        return out

    def finish(self) -> list[tuple[str, str]]:
        """flush at the end of the stream when it ended on the length limit (nothing is left after EOS or a stop string)"""
        if self.done:
            return []
        self.buf += self.dec.flush()
        out = self._drain(final=True)
        self.buf = ""
        return out


# ------------------------------------------------------------------------------------------------ response bodies
ABLIT_OFF = ("", "off", "0", "false", "no", "none")


def ablit_spec(environ, tree: str) -> str | None:
    """Decensor preset of the Hy4 server (on by default). HY4_ABLIT_SPEC unset -> the preset bundled in the repository (None when its files are absent);
    an explicit path wins; empty, 0, off, false, no or none -> None (no edit is installed, the forward is the unmodified one)."""
    v = environ.get("HY4_ABLIT_SPEC")
    if v is None:
        p = os.path.join(tree, "presets", "hy4-uncensor", "edit_spec.json")
        return p if os.path.isfile(p) and os.path.isfile(os.path.join(os.path.dirname(p), "edit_spec.safetensors")) else None
    v = v.strip()
    return None if v.lower() in ABLIT_OFF else v


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def chat_chunk(rid: str, model: str, created: int, delta: dict, finish: str | None = None, **extra: Any) -> dict:
    return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}


def chat_response(rid: str, model: str, created: int, content: str, reasoning: str, finish: str, usage: dict, **extra: Any) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {"id": rid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage, **extra}


def completion_chunk(rid: str, model: str, created: int, text: str, finish: str | None = None, **extra: Any) -> dict:
    return {"id": rid, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish, "logprobs": None}], **extra}


def error_body(message: str, code: str = "invalid_request_error") -> dict:
    return {"error": {"message": message, "type": code, "code": code}}
