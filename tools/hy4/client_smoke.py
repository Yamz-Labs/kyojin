#!/usr/bin/env python3
"""Client smoke gate of the Hy4 server: the release gate for every public release (issue #222 and its follow-ups).

It talks to a RUNNING server (tools/hy4/serve.py) over HTTP with the standard library only, and prints one PASS / FAIL table
(cases x modes), plus an optional JSON file.

  python tools/hy4/client_smoke.py --base http://127.0.0.1:8300 --ctx 16384 --json smoke.json

Start the server with `-c 16384` (the budget cases assume that window; pass --ctx when it differs).
Every case is run in the modes that apply (non-stream / stream) and is followed by a NORMAL request in the same mode:
a refused or odd request must never leave the server stuck. A cell reads `P/P` (case / normal request after it), `F/P`, ...

Groups
  A  media parts        image (user, tool result, assistant), audio and file parts: 200 with text
  T  tool calls         auto / required / named, single turn and multi-turn with results (one and two calls), tool_choice none
  K  agent clients      one replayed request shape each: omp, opencode (AI SDK), Hermes (system prompt size, tools array, stream_options, extra fields)
  B  refused requests   400 with the named reason (chat and completions)
  F  reply budget       no max_tokens on a long prompt, oversized max_tokens (clipped, not refused)
  C  dropped stream     a client that closes a stream after its first chunk (chat and completions)
  X  concurrency        4 requests at once (chat stream, chat non-stream, completions, tool call), twice; all 200, none hangs
  D  strict image parts only with --image-parts-reject (server started with KYOJIN_IMAGE_PARTS=reject): the 400 that names the part, then a normal request

Exit status: 0 = every check passed, 1 = a check failed, 7 = the server never became ready.
Each new client bug of the server adds its case here."""
from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

CHAT, COMP = "/v1/chat/completions", "/v1/completions"
IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="}}
AUD = {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}
FIL = {"type": "file", "file": {"file_id": "file-1"}}
TXT = {"type": "text", "text": "Describe the attachment in one short sentence."}
U = {"role": "user", "content": "Say hello in one short sentence."}
NOTHINK = {"reasoning_effort": "no_think", "max_tokens": 32}

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}
CLOCK = {"type": "function", "function": {
    "name": "get_time", "description": "Current local time of a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
SHOT = {"type": "function", "function": {"name": "screenshot", "description": "take a screenshot", "parameters": {
    "type": "object", "properties": {"target": {"type": "string", "enum": ["screen", "window"]},
                                     "box": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}},
    "required": ["target"]}}}
ASK_PARIS = {"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."}
HELLO = {"role": "user", "content": "Hello there."}
TOOL_EFFORT = {"reasoning_effort": "low", "max_tokens": 1024}


def tcall(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


CALL_W = {"role": "assistant", "content": None, "tool_calls": [tcall("call_1", "get_weather", {"city": "Paris"})]}
RES_W = {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"city": "Paris", "temp": 18, "unit": "c", "sky": "sunny"})}
CALL_2 = {"role": "assistant", "content": None, "tool_calls": [tcall("call_1", "get_weather", {"city": "Paris"}), tcall("call_2", "get_time", {"city": "Paris"})]}
RES_2 = [RES_W, {"role": "tool", "tool_call_id": "call_2", "content": "14:05"}]


# ---------------------------------------------------------------------------------------------------------------- HTTP
class Rep:
    """one reply: status, finish_reason, usage, text (content + reasoning), tool calls [{name, args}], whether the stream ended with [DONE], the raw error body"""

    def __init__(self):
        self.status, self.finish, self.usage, self.text, self.calls, self.done, self.err = 0, None, None, "", [], True, ""
        self.n_chunks = 0


def send(base: str, path: str, body, stream: bool, timeout: float = 300) -> Rep:
    r = Rep()
    if isinstance(body, dict):
        body = dict(body, stream=stream)
        if stream:
            body.setdefault("stream_options", {"include_usage": True})
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            r.status = resp.status
            if not stream:
                j = json.loads(resp.read().decode("utf-8", "replace"))
                c = j["choices"][0]
                m = c.get("message") or {}
                r.finish, r.usage = c.get("finish_reason"), j.get("usage")
                r.text = (m.get("content") or "") + (m.get("reasoning_content") or "") + (c.get("text") or "")
                r.calls = [{"name": t["function"]["name"], "args": t["function"]["arguments"]} for t in (m.get("tool_calls") or [])]
                return r
            slots: dict[int, dict] = {}
            r.done = False
            for raw in resp:
                if not raw.startswith(b"data: "):
                    continue
                d = raw[6:].decode("utf-8", "replace").strip()
                if d == "[DONE]":
                    r.done = True
                    break
                ch = json.loads(d)
                r.n_chunks += 1
                if ch.get("error"):
                    r.status, r.err = 500, d[:300]
                    return r
                r.usage = ch.get("usage") or r.usage
                for c in ch.get("choices") or []:
                    r.finish = c.get("finish_reason") or r.finish
                    dl = c.get("delta") or {}
                    r.text += (dl.get("content") or "") + (dl.get("reasoning_content") or "") + (c.get("text") or "")
                    for t in dl.get("tool_calls") or []:
                        s = slots.setdefault(t.get("index", 0), {"name": "", "args": ""})
                        s["name"] += (t.get("function") or {}).get("name") or ""
                        s["args"] += (t.get("function") or {}).get("arguments") or ""
            r.calls = [slots[k] for k in sorted(slots)]
            if not r.done:
                r.status, r.err = 500, "the stream ended without [DONE]"
            return r
    except urllib.error.HTTPError as e:
        r.status, r.err = e.code, e.read().decode("utf-8", "replace")
        return r


def valid_calls(calls, declared=("get_weather", "get_time", "screenshot")):
    """every call names a declared function and carries JSON-object arguments -> (ok, why)"""
    for c in calls:
        if c["name"] not in declared:
            return False, f"unknown function {c['name']!r}"
        try:
            if not isinstance(json.loads(c["args"]), dict):
                return False, f"arguments are not an object: {c['args'][:80]!r}"
        except ValueError:
            return False, f"arguments are not JSON: {c['args'][:80]!r}"
    return True, ""


def brief(r: Rep) -> str:
    if r.status != 200:
        return f"status {r.status} {r.err[:110]!r}"
    return f"status 200 finish {r.finish} calls {[c['name'] for c in r.calls]} text {r.text.strip()[:50]!r}"


# ---------------------------------------------------------------------------------------------------------------- checks
def chk_text(r: Rep, stream: bool):
    ok = r.status == 200 and (bool(r.text.strip()) or bool(r.calls))
    return ok, brief(r)


def chk_usage(r: Rep, stream: bool):
    """a client that asked for stream_options.include_usage gets the usage chunk"""
    ok, d = chk_text(r, stream)
    if ok and stream and not r.usage:
        return False, "no usage chunk although stream_options.include_usage was set; " + d
    return ok, d


def chk_call(r: Rep, stream: bool):
    """a tool call with valid JSON arguments, finish_reason tool_calls"""
    if r.status != 200:
        return False, brief(r)
    ok, why = valid_calls(r.calls)
    if not r.calls or not ok or r.finish != "tool_calls":
        return False, (why or "no tool call / finish_reason " + repr(r.finish)) + "; " + brief(r)
    return True, brief(r)


def chk_named(r: Rep, stream: bool):
    ok, d = chk_call(r, stream)
    if ok and r.calls[0]["name"] != "get_weather":
        return False, "named get_weather, got " + d
    return ok, d


def chk_final(r: Rep, stream: bool):
    """after the results came back: a text answer (finish stop) or one more valid call"""
    if r.status != 200:
        return False, brief(r)
    ok, why = valid_calls(r.calls)
    if not ok:
        return False, why
    if r.calls:
        return r.finish == "tool_calls", brief(r)
    return bool(r.text.strip()) and r.finish in ("stop", "length"), brief(r)


def chk_no_call(r: Rep, stream: bool):
    return r.status == 200 and not r.calls and bool(r.text.strip()), brief(r)


def chk_refused(want: str):
    def f(r: Rep, stream: bool):
        return r.status == 400 and want.lower() in r.err.lower(), f"status {r.status} {r.err[:110]!r}"
    return f


# ---------------------------------------------------------------------------------------------------------------- bodies
def words(n: int, salt: str = "") -> str:
    base = ("The build system caches every compiled object by content hash, a rebuild touches only the changed files, "
            "and the test runner reports each failure with the command that reproduces it. ")
    return (base * (n // 24 + 1)).strip()[: n * 6]


def omp_body():
    """omp: system + user, assistant tool_calls (JSON arguments), a tool result with a text part AND an image_url part, nested JSON-schema tools"""
    edit = {"type": "function", "function": {"name": "edit_files", "description": "apply edits to files", "parameters": {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["replace", "insert", "delete"]},
                       "options": {"type": "object", "properties": {"dry_run": {"type": "boolean"}, "ctx": {"type": "object", "properties": {"lines": {"type": "integer"}}}}},
                       "edits": {"type": "array", "items": {"type": "object", "properties": {"path": {"type": "string"}, "line": {"type": "integer"}, "text": {"type": "string"}},
                                                           "required": ["path", "text"]}}},
        "required": ["mode", "edits"]}}}
    call = {"role": "assistant", "content": None, "tool_calls": [tcall("call_7", "screenshot", {"target": "window"})]}
    return {"messages": [{"role": "system", "content": "You are a careful coding agent."}, {"role": "user", "content": "take a screenshot of the window and tell me what you see"}, call,
                         {"role": "tool", "tool_call_id": "call_7", "content": [{"type": "text", "text": "screenshot taken"}, IMG]},
                         {"role": "user", "content": "describe it in one short sentence"}], "tools": [edit, SHOT], **NOTHINK}


def opencode_body():
    """opencode (AI SDK openai-compatible provider): long system prompt, user content as text-part arrays, assistant content "" + tool_calls (two parallel calls),
    string tool results, strict JSON-schema tools with $schema / additionalProperties, tool_choice auto, sampling fields, a large max_tokens, include_usage"""
    sch = "http://json-schema.org/draft-07/schema#"
    tools = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p, "required": req, "additionalProperties": False, "$schema": sch}}}
             for n, d, p, req in (
                 ("bash", "Run a shell command in the project directory.", {"command": {"type": "string"}, "timeout": {"type": "number"}}, ["command"]),
                 ("read", "Read a file from the project.", {"filePath": {"type": "string"}, "offset": {"type": "number"}, "limit": {"type": "number"}}, ["filePath"]),
                 ("grep", "Search file contents with a regular expression.", {"pattern": {"type": "string"}, "include": {"type": "string"}}, ["pattern"]),
                 ("edit", "Replace text in a file.", {"filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"}, "replaceAll": {"type": "boolean"}},
                  ["filePath", "oldString", "newString"]))]
    calls = [tcall("call_a", "grep", {"pattern": "def main"}), tcall("call_b", "read", {"filePath": "src/app.py", "limit": 40})]
    return {"model": "hy4", "max_tokens": 32000, "temperature": 0.7, "top_p": 0.95, "tool_choice": "auto", "tools": tools,
            "messages": [{"role": "system", "content": "You are a coding agent working in a terminal.\n" + words(1800)},
                         {"role": "user", "content": [{"type": "text", "text": "Find where the program starts and say in one sentence what it does."}]},
                         {"role": "assistant", "content": "", "tool_calls": calls},
                         {"role": "tool", "tool_call_id": "call_a", "content": "src/app.py:12:def main():"},
                         {"role": "tool", "tool_call_id": "call_b", "content": "def main():\n    args = parse()\n    run(args)\n"},
                         {"role": "user", "content": [{"type": "text", "text": "Answer now in one short sentence."}]}], "reasoning_effort": "low"}


def hermes_body():
    """Hermes (OpenAI chat wire): a large system prompt, about a dozen tools with long descriptions, assistant turn with reasoning_content + tool_calls, string tool result,
    prompt_cache_key and other extra fields merged at the top level, no tool_choice, include_usage; stream"""
    names = ("terminal", "read_file", "write_file", "patch", "search_files", "todo", "memory", "web_search", "web_extract", "browser_navigate", "delegate_task", "session_search")
    tools = [{"type": "function", "function": {"name": n, "description": f"{n}: " + words(80), "parameters": {
        "type": "object", "properties": {"query": {"type": "string", "description": words(20)}, "path": {"type": "string"}, "limit": {"type": "integer"},
                                         "options": {"type": "object", "properties": {"recursive": {"type": "boolean"}, "mode": {"type": "string", "enum": ["fast", "full"]}}}},
        "required": ["query"]}}} for n in names]
    call = {"role": "assistant", "content": "", "reasoning_content": "I should look the file up first.", "tool_calls": [tcall("call_h1", "search_files", {"query": "README"})]}
    return {"model": "hy4", "temperature": 0.6, "max_tokens": 4096, "prompt_cache_key": "smoke-gate-1", "kyojin_flags": {}, "tools": tools,
            "messages": [{"role": "system", "content": "You are an agent with tools and a persistent memory.\n" + words(2200)},
                         {"role": "user", "content": "Is there a README in this project? Answer yes or no."}, call,
                         {"role": "tool", "tool_call_id": "call_h1", "content": json.dumps({"matches": ["README.md"], "count": 1})}],
            "reasoning_effort": "low"}


FILL = "The quick brown fox jumps over the lazy dog near the old river bank. " * 400   # about 6000 tokens
LONGQ = {"role": "user", "content": FILL + "\nIn one short sentence, what animal jumps in the text above?"}
LONGP = FILL + "\nThe animal that jumps is"


class Case:
    def __init__(self, group, name, path, body, check, modes=(False, True), kind=None):
        self.group, self.name, self.path, self.body, self.check, self.modes, self.kind = group, name, path, body, check, modes, kind


def cases() -> list[Case]:
    out = []
    for kind, part in (("image", IMG), ("audio", AUD), ("file", FIL)):
        out.append(Case("A", f"{kind}_user", CHAT, {"messages": [{"role": "user", "content": [TXT, part]}], **NOTHINK}, chk_text))
    out += [Case("A", "image_tool_result", CHAT, {"messages": [{"role": "user", "content": "take a screenshot"}, {"role": "assistant", "content": None, "tool_calls": [tcall("call_1", "screenshot", {"target": "screen"})]},
                                                                 {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "done"}, IMG]},
                                                                 {"role": "user", "content": "what do you see?"}], "tools": [SHOT], **NOTHINK}, chk_text),
            Case("A", "image_assistant", CHAT, {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "text", "text": "hello"}, IMG]},
                                                              {"role": "user", "content": "and now?"}], **NOTHINK}, chk_text)]
    # T: tool calls
    tw = [WEATHER, CLOCK]
    out += [Case("T", "single_auto", CHAT, {"messages": [ASK_PARIS], "tools": tw, "tool_choice": "auto", **TOOL_EFFORT}, chk_call),
            Case("T", "single_required", CHAT, {"messages": [HELLO], "tools": tw, "tool_choice": "required", **TOOL_EFFORT}, chk_call),
            Case("T", "single_named", CHAT, {"messages": [HELLO], "tools": tw, "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, **TOOL_EFFORT}, chk_named),
            Case("T", "single_none", CHAT, {"messages": [ASK_PARIS], "tools": tw, "tool_choice": "none", **NOTHINK}, chk_no_call),
            Case("T", "multi_auto", CHAT, {"messages": [ASK_PARIS, CALL_W, RES_W], "tools": tw, "tool_choice": "auto", **TOOL_EFFORT}, chk_final),
            Case("T", "multi_required", CHAT, {"messages": [ASK_PARIS, CALL_W, RES_W], "tools": tw, "tool_choice": "required", **TOOL_EFFORT}, chk_call),
            Case("T", "multi_named", CHAT, {"messages": [ASK_PARIS, CALL_W, RES_W], "tools": tw, "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, **TOOL_EFFORT}, chk_named),
            Case("T", "multi_two_calls_auto", CHAT, {"messages": [ASK_PARIS, CALL_2, *RES_2], "tools": tw, "tool_choice": "auto", **TOOL_EFFORT}, chk_final)]
    # K: agent clients
    out += [Case("K", "omp", CHAT, omp_body(), chk_usage), Case("K", "opencode", CHAT, opencode_body(), chk_usage), Case("K", "hermes", CHAT, hermes_body(), chk_usage)]
    # B: refused
    big = "word " * 24000
    for name, path, body, want in (
            ("unknown_part", CHAT, {"messages": [U, {"role": "assistant", "content": "a"}, {"role": "user", "content": [TXT, {"type": "video_url", "video_url": {"url": "u"}}]}]},
             "message 2, part 1: type 'video_url' is not supported"),
            ("bad_tools", CHAT, {"messages": [U], "tools": [1]}, "tools"),
            ("bad_effort", CHAT, {"messages": [U], "reasoning_effort": "ultra"}, "reasoning_effort"),
            ("bad_max_tokens", CHAT, {"messages": [U], "max_tokens": "many"}, "max_tokens"),
            ("empty_messages", CHAT, {"messages": []}, "messages"),
            ("bad_role", CHAT, {"messages": [{"role": "robot", "content": "x"}]}, "role"),
            ("oversize_prompt", CHAT, {"messages": [{"role": "user", "content": big}]}, "does not fit the context window"),
            ("bad_json", CHAT, b"{not json", "JSON"),
            ("bad_tool_choice", CHAT, {"messages": [U], "tools": [WEATHER], "tool_choice": "sometimes"}, "tool_choice"),
            ("tool_choice_without_tools", CHAT, {"messages": [U], "tool_choice": "required"}, "needs tools"),
            ("tool_choice_unknown_name", CHAT, {"messages": [U], "tools": [WEATHER], "tool_choice": {"type": "function", "function": {"name": "nope"}}}, "not in tools"),
            ("bad_stream_options", CHAT, {"messages": [U], "stream_options": 5}, "stream_options"),
            ("n_two", CHAT, {"messages": [U], "n": 2}, "n must be 1"),
            ("bad_tool_arguments", CHAT, {"messages": [U, {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "get_weather", "arguments": "not json"}}]},
                                                       {"role": "tool", "tool_call_id": "c", "content": "x"}]}, "arguments"),
            ("completions_empty_prompt", COMP, {"prompt": ""}, "prompt"),
            ("completions_oversize", COMP, {"prompt": big}, "does not fit the context window")):
        out.append(Case("B", name, path, body, chk_refused(want)))
    # F: reply budget
    out += [Case("F", "free_chat_think", CHAT, {"messages": [LONGQ]}, None, (False,), "free"),
            Case("F", "free_chat_nothink", CHAT, {"messages": [LONGQ], "reasoning_effort": "no_think"}, None, kind="free"),
            Case("F", "free_completions", COMP, {"prompt": LONGP, "stop": ["\n"]}, None, kind="free"),
            Case("F", "clip_chat", CHAT, {"messages": [LONGQ], "reasoning_effort": "no_think", "max_tokens": 100000}, None, kind="clip"),
            Case("F", "clip_chat_alias", CHAT, {"messages": [LONGQ], "reasoning_effort": "no_think", "max_completion_tokens": 100000}, None, kind="clip"),
            Case("F", "clip_completions", COMP, {"prompt": LONGP, "stop": ["\n"], "max_tokens": 100000}, None, kind="clip")]
    return out


# ---------------------------------------------------------------------------------------------------------------- runner
class Smoke:
    def __init__(self, base: str, ctx: int, timeout: float):
        self.base, self.ctx, self.timeout = base.rstrip("/"), ctx, timeout
        self.rows: list[dict] = []

    def normal(self, stream: bool):
        t0 = time.time()
        r = send(self.base, CHAT, {"messages": [U], **NOTHINK}, stream, timeout=120)
        return r.status == 200 and bool(r.text.strip()), (f"normal answered ({time.time() - t0:.1f} s)" if r.status == 200 else f"normal: {brief(r)}")

    def add(self, group, name, mode, ok, detail, after, after_detail, secs):
        self.rows.append({"group": group, "case": name, "mode": mode, "ok": bool(ok), "after_ok": after if after is None else bool(after), "detail": detail, "after_detail": after_detail,
                          "seconds": round(secs, 1)})
        print(f"{'PASS' if ok else 'FAIL'} {group} {name} {mode}: {detail}", flush=True)
        if after is not None:
            print(f"{'PASS' if after else 'FAIL'} {group} {name} {mode} then normal: {after_detail}", flush=True)

    def after(self, stream: bool):
        try:
            return self.normal(stream)
        except Exception as e:      # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"

    def budget_check(self, c: Case, r: Rep):
        n_p, n_c = (r.usage or {}).get("prompt_tokens"), (r.usage or {}).get("completion_tokens")
        d = f"status {r.status} finish {r.finish} prompt {n_p} completion {n_c} text {r.text.strip()[:40]!r}" if r.status == 200 else brief(r)
        ok = r.status == 200 and r.finish in ("stop", "length") and r.usage is not None
        if ok:
            ok = 4000 <= n_p <= 9000                 # the prompt really is about 6000 tokens
        if ok and r.finish == "length":              # ended by the budget: it must be the room left, not the old 1024 cap
            ok = n_c is not None and n_c > 1024 and n_p + n_c <= self.ctx
        return ok, d

    def run_case(self, c: Case):
        for stream in c.modes:
            mode = "stream" if stream else "plain"
            t0 = time.time()
            try:
                r = send(self.base, c.path, c.body, stream, timeout=900 if c.group == "F" else self.timeout)
                ok, d = self.budget_check(c, r) if c.group == "F" else c.check(r, stream)
            except Exception as e:      # noqa: BLE001
                ok, d = False, f"{type(e).__name__}: {e}"
            a_ok, a_d = self.after(stream)
            self.add(c.group, c.name, mode, ok, d, a_ok, a_d, time.time() - t0)

    def run_reject(self):
        """D: KYOJIN_IMAGE_PARTS=reject: media parts are refused with the named 400"""
        for c in cases():
            if c.group == "A" and c.name in ("image_user", "audio_user", "file_user"):
                for stream in (False, True):
                    t0 = time.time()
                    try:
                        r = send(self.base, c.path, c.body, stream)
                        ok = r.status == 400 and "is not supported" in r.err and "message" in r.err and "part" in r.err
                        d = f"status {r.status} {r.err[:110]!r}"
                    except Exception as e:      # noqa: BLE001
                        ok, d = False, f"{type(e).__name__}: {e}"
                    a_ok, a_d = self.after(stream)
                    self.add("D", "reject_" + c.name, "stream" if stream else "plain", ok, d, a_ok, a_d, time.time() - t0)

    def run_drop(self):
        """C: a client that closes the stream after its first chunk; the next normal request answers within 60 s"""
        host, port = urlsplit(self.base).hostname, urlsplit(self.base).port
        for name, path, body in (("drop_chat", CHAT, {"messages": [{"role": "user", "content": "Count from 1 to 300, one number per line."}], "stream": True, "max_tokens": 600, "reasoning_effort": "no_think"}),
                                 ("drop_completions", COMP, {"prompt": "1 2 3 4 5 6 7 8 9 10 11 12", "stream": True, "max_tokens": 600})):
            t0 = time.time()
            try:
                data = json.dumps(body).encode()
                s = socket.create_connection((host, port), timeout=60)
                s.sendall(f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n\r\n".encode() + data)
                got = b""
                while b"data: {" not in got.split(b"\r\n\r\n", 1)[-1]:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    got += chunk
                s.close()
                ok, d = b"data: {" in got, "closed after the first chunk"
            except Exception as e:      # noqa: BLE001
                ok, d = False, f"{type(e).__name__}: {e}"
            t1 = time.time()
            a_ok, a_d = self.after(False)
            a_ok = a_ok and time.time() - t1 < 60
            s_ok, s_d = self.after(True)
            self.add("C", name, "stream", ok, d, a_ok and s_ok, a_d if not a_ok else s_d, time.time() - t0)

    def run_concurrent(self):
        """X: four requests at once, twice; every one 200 with a normal output, none hangs, then a normal request"""
        jobs = [("chat_stream", CHAT, {"messages": [{"role": "user", "content": "Write two short sentences about rivers."}], "max_tokens": 64, "reasoning_effort": "no_think"}, True, chk_text),
                ("chat_plain", CHAT, {"messages": [{"role": "user", "content": "Write two short sentences about mountains."}], "max_tokens": 64, "reasoning_effort": "no_think"}, False, chk_text),
                ("completions", COMP, {"prompt": "The three primary colors are", "max_tokens": 32}, False, chk_text),
                ("tool_call", CHAT, {"messages": [ASK_PARIS], "tools": [WEATHER, CLOCK], "tool_choice": "required", **TOOL_EFFORT}, False, chk_call)]
        for rnd in (1, 2):
            res: dict[str, tuple] = {}

            def work(n, p, b, st, ck):
                t0 = time.time()
                try:
                    r = send(self.base, p, b, st, timeout=900)
                    ok, d = ck(r, st)
                except Exception as e:      # noqa: BLE001
                    ok, d = False, f"{type(e).__name__}: {e}"
                res[n] = (ok, f"{d} ({time.time() - t0:.0f} s)")
            t0 = time.time()
            ths = [threading.Thread(target=work, args=(n, p, b, st, ck)) for n, p, b, st, ck in jobs]
            for t in ths:
                t.start()
            for t in ths:
                t.join(1000)
            hung = [n for n, *_ in jobs if n not in res]
            ok = not hung and all(v[0] for v in res.values())
            d = f"hung: {hung}; " if hung else ""
            d += "; ".join(f"{n}: {'ok' if res[n][0] else res[n][1]}" for n, *_ in jobs if n in res)
            a_ok, a_d = self.after(False)
            self.add("X", f"four_at_once_round{rnd}", "mixed", ok, d, a_ok, a_d, time.time() - t0)

    def table(self) -> str:
        names = []
        for r in self.rows:
            if (r["group"], r["case"]) not in names:
                names.append((r["group"], r["case"]))
        w = max(len(f"{g} {n}") for g, n in names)
        lines = [f"{'case'.ljust(w)}  plain  stream  mixed"]
        for g, n in names:
            cells = []
            for mode in ("plain", "stream", "mixed"):
                row = next((x for x in self.rows if x["group"] == g and x["case"] == n and x["mode"] == mode), None)
                cells.append("-" if row is None else ("P" if row["ok"] else "F") + "/" + ("P" if row["after_ok"] else "F"))
            lines.append(f"{(g + ' ' + n).ljust(w)}  {cells[0]:5}  {cells[1]:6}  {cells[2]}")
        return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8300", help="server URL")
    ap.add_argument("--ctx", type=int, default=16384, help="context window the server was started with (-c)")
    ap.add_argument("--timeout", type=float, default=300, help="seconds per request (budget cases: 900)")
    ap.add_argument("--json", default=None, help="write the result rows here")
    ap.add_argument("--only", default=None, help="run only the groups listed (letters, e.g. ATK)")
    ap.add_argument("--image-parts-reject", action="store_true", help="the server runs with KYOJIN_IMAGE_PARTS=reject: run group D only")
    ap.add_argument("--ready-wait", type=float, default=600, help="seconds to wait for /health")
    a = ap.parse_args(argv)
    end = time.time() + a.ready_wait
    while True:
        try:
            with urllib.request.urlopen(a.base + "/health", timeout=5) as h:
                if h.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            pass
        if time.time() > end:
            print(f"server at {a.base} is not ready", flush=True)
            return 7
        time.sleep(2)
    s = Smoke(a.base, a.ctx, a.timeout)
    want = set((a.only or "ABCFKTX").upper()) if not a.image_parts_reject else {"D"}
    t0 = time.time()
    for c in cases():
        if c.group in want:
            s.run_case(c)
    if "C" in want:
        s.run_drop()
    if "X" in want:
        s.run_concurrent()
    if "D" in want:
        s.run_reject()
    alive = True
    try:
        urllib.request.urlopen(a.base + "/health", timeout=10).read()
    except Exception:      # noqa: BLE001
        alive = False
    s.rows.append({"group": "Z", "case": "server_alive_at_the_end", "mode": "plain", "ok": alive, "after_ok": alive, "detail": "", "after_detail": "", "seconds": 0})
    n_ok = sum(r["ok"] for r in s.rows) + sum(r["after_ok"] for r in s.rows if r["after_ok"] is not None and r["case"] != "server_alive_at_the_end")
    n_all = len(s.rows) + sum(1 for r in s.rows if r["after_ok"] is not None and r["case"] != "server_alive_at_the_end")
    ok = n_ok == n_all
    print("\n" + s.table(), flush=True)
    print(f"HY4_CLIENT_SMOKE {'PASS' if ok else 'FAIL'} {n_ok}/{n_all} checks in {time.time() - t0:.0f} s", flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"pass": ok, "checks": n_all, "passed": n_ok, "rows": s.rows}, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
