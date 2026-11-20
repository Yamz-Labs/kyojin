#!/usr/bin/env python3
"""Served smoke test of the Hy4 server's tool calling and client disconnect. It talks to a RUNNING server (tools/hy4/serve.py) over HTTP and prints PASS / FAIL
per case; the exit status is 1 when any case failed.

  python tools/hy4/tool_smoke.py --base http://127.0.0.1:8300

Cases (each needs the real model: the reply is whatever the model decides):
  1. tool call, not streamed    a weather question with one tool -> message.tool_calls, finish_reason "tool_calls"
  2. tool call, streamed        same, read from delta.tool_calls chunks (index, id + name, then the arguments)
  3. multi-turn round trip      the call and its result are sent back -> a final text answer, finish_reason "stop"
  4. disconnect mid-stream      close a long stream after a few chunks -> the engine is free again within a few seconds
  5. forced tool_choice         a greeting with tool_choice naming get_weather -> still a get_weather call, finish_reason "tool_calls"
Standard library only (http.client): it runs in the release venv as it is.
"""
from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
from urllib.parse import urlsplit

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                                                    "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]
QUESTION = [{"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."}]
RESULT = json.dumps({"city": "Paris", "temp": 18, "unit": "c", "sky": "sunny"})


class Reply:
    """one HTTP reply: status and the whole body"""

    def __init__(self, status: int, body: bytes):
        self.status_code, self.text = status, body.decode("utf-8", "replace")

    def json(self):
        return json.loads(self.text)


class Smoke:
    def __init__(self, base: str, model: str | None, effort: str, max_tokens: int, timeout: float):
        u = urlsplit(base if "//" in base else f"http://{base}")
        self.host, self.port, self.prefix = u.hostname or "127.0.0.1", u.port or (443 if u.scheme == "https" else 80), u.path.rstrip("/")
        self.https = u.scheme == "https"
        self.base, self.effort, self.max_tokens, self.timeout = base.rstrip("/"), effort, max_tokens, timeout
        self.model = model or "Hy4-flash"
        self.call: dict | None = None            # the tool call of case 1 (id, name, arguments), reused by case 3
        self.results: list[tuple[str, bool, str]] = []

    # -- plumbing
    def connect(self, timeout: float | None = None) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout or self.timeout)

    def send(self, method: str, path: str, body: dict | None = None, timeout: float | None = None):
        """-> (connection, response) with the headers read; the body is read by the caller"""
        conn = self.connect(timeout)
        data = json.dumps(body).encode() if body is not None else None
        conn.request(method, self.prefix + path, body=data, headers={"Content-Type": "application/json"} if data else {})
        return conn, conn.getresponse()

    def get(self, path: str, timeout: float | None = None) -> Reply:
        conn, resp = self.send("GET", path, timeout=timeout)
        try:
            return Reply(resp.status, resp.read())
        finally:
            conn.close()

    def body(self, messages, **kw) -> dict:
        return {"model": self.model, "messages": messages, "tools": TOOLS, "max_tokens": self.max_tokens, "reasoning_effort": self.effort, **kw}

    def post(self, body: dict) -> Reply:
        conn, resp = self.send("POST", "/v1/chat/completions", body)
        try:
            return Reply(resp.status, resp.read())
        finally:
            conn.close()

    def wait_ready(self, seconds: float = 600) -> None:
        end = time.time() + seconds
        while True:
            try:
                if self.get("/health", timeout=5).status_code == 200:
                    return
            except (OSError, http.client.HTTPException):
                pass
            if time.time() > end:
                sys.exit(f"server at {self.base} is not ready")
            time.sleep(2)

    def record(self, name: str, ok: bool, detail: str) -> None:
        self.results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    def run(self, name: str, fn) -> None:
        try:
            ok, detail = fn()
        except Exception as e:      # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        self.record(name, ok, detail)

    @staticmethod
    def check_call(name: str, args: str) -> tuple[bool, str]:
        if name != "get_weather":
            return False, f"unexpected function {name!r}"
        try:
            parsed = json.loads(args)
        except ValueError:
            return False, f"arguments are not JSON: {args[:120]!r}"
        if not isinstance(parsed, dict) or "paris" not in str(parsed.get("city", "")).lower():
            return False, f"arguments do not name Paris: {args[:120]!r}"
        return True, f"get_weather({args})"

    # -- cases
    def case_non_stream(self):
        r = self.post(self.body(QUESTION))
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        ch = r.json()["choices"][0]
        msg, calls = ch["message"], ch["message"].get("tool_calls") or []
        if ch["finish_reason"] != "tool_calls" or not calls:
            return False, f"finish_reason {ch['finish_reason']!r}, no tool_calls; content {str(msg.get('content'))[:160]!r}"
        c = calls[0]
        if not (c.get("id") and c.get("type") == "function" and isinstance(c["function"].get("arguments"), str)):
            return False, f"malformed tool call: {json.dumps(c)[:200]}"
        ok, detail = self.check_call(c["function"]["name"], c["function"]["arguments"])
        if ok:
            self.call = {"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
            detail += f" id {c['id']}, content {msg.get('content')!r}"
        return ok, detail

    def case_stream(self):
        conn, resp = self.send("POST", "/v1/chat/completions", self.body(QUESTION, stream=True))
        try:
            if resp.status != 200:
                return False, f"HTTP {resp.status}: {resp.read().decode('utf-8', 'replace')[:200]}"
            calls: dict[int, dict] = {}
            finish, done, n_chunks, text = None, False, 0, ""
            for raw in resp:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].decode().strip()
                if data == "[DONE]":
                    done = True
                    break
                ch = json.loads(data)
                n_chunks += 1
                if not ch.get("choices"):
                    continue
                c = ch["choices"][0]
                text += c["delta"].get("content") or ""
                for tc in c["delta"].get("tool_calls") or []:
                    slot = calls.setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
                    slot["id"] = tc.get("id") or slot["id"]
                    slot["name"] += (tc.get("function") or {}).get("name") or ""
                    slot["arguments"] += (tc.get("function") or {}).get("arguments") or ""
                finish = c.get("finish_reason") or finish
        finally:
            conn.close()
        if not done:
            return False, "the stream ended without [DONE]"
        if finish != "tool_calls" or not calls:
            return False, f"finish_reason {finish!r}, tool calls {len(calls)}; content {text[:160]!r}"
        first = calls[min(calls)]
        if not first["id"]:
            return False, "the first tool call chunk has no id"
        ok, detail = self.check_call(first["name"], first["arguments"])
        return ok, f"{detail} ({n_chunks} chunks)" if ok else detail

    def case_multi_turn(self):
        call = self.call or {"id": "call_smoke_1", "name": "get_weather", "arguments": json.dumps({"city": "Paris"})}
        messages = QUESTION + [
            {"role": "assistant", "content": None, "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]},
            {"role": "tool", "tool_call_id": call["id"], "content": RESULT}]
        r = self.post(self.body(messages))
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        ch = r.json()["choices"][0]
        content = ch["message"].get("content") or ""
        if ch["finish_reason"] != "stop" or not content.strip():
            return False, f"finish_reason {ch['finish_reason']!r}, content {content[:160]!r}, tool_calls {ch['message'].get('tool_calls')}"
        return True, f"answer {content.strip()[:100]!r} (mentions 18: {'18' in content})"

    def case_forced(self):
        choice = {"type": "function", "function": {"name": "get_weather"}}
        r = self.post(self.body([{"role": "user", "content": "Say hello to me."}], tool_choice=choice))
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}: {r.text[:200]}"
        ch = r.json()["choices"][0]
        calls = ch["message"].get("tool_calls") or []
        if ch["finish_reason"] != "tool_calls" or not calls:
            return False, f"finish_reason {ch['finish_reason']!r}, no tool_calls; content {str(ch['message'].get('content'))[:160]!r}"
        fn = calls[0]["function"]
        try:
            json.loads(fn["arguments"])
        except ValueError:
            return False, f"arguments are not JSON: {fn['arguments'][:120]!r}"
        return fn["name"] == "get_weather", f"{fn['name']}({fn['arguments']})"

    def case_disconnect(self):
        quick = self.body([{"role": "user", "content": "Say hello in one word."}], max_tokens=16, reasoning_effort="no_think")
        quick.pop("tools")
        t0 = time.time()
        r = self.post(quick)
        if r.status_code != 200:
            return False, f"baseline request failed: HTTP {r.status_code}"
        base = time.time() - t0
        long = self.body([{"role": "user", "content": "Count from 1 to 1000, one number per line, no other text."}], max_tokens=self.max_tokens * 2, reasoning_effort="no_think")
        long.pop("tools")
        long["stream"] = True
        conn, resp = self.send("POST", "/v1/chat/completions", long)
        n = 0
        for raw in resp:
            if raw.startswith(b"data: "):
                n += 1
                if n >= 5:
                    break
        conn.close()                                # the client leaves mid-stream
        t1 = time.time()
        r = self.post(quick)                        # queued behind the abandoned generation until the engine stops
        after = time.time() - t1
        if r.status_code != 200:
            return False, f"request after the disconnect failed: HTTP {r.status_code}: {r.text[:200]}"
        limit = base + 8.0
        return after <= limit, f"engine free again after {after:.1f} s (idle request {base:.1f} s, limit {limit:.1f} s); read {n} chunks before closing"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default="http://127.0.0.1:8300", help="server URL")
    ap.add_argument("--model", default=None, help="model id (default: the first of /v1/models)")
    ap.add_argument("--effort", default="low", help="reasoning_effort of the tool cases (no_think / low / high / max)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="token budget of one tool case")
    ap.add_argument("--timeout", type=float, default=300, help="seconds per request")
    args = ap.parse_args(argv)
    sm = Smoke(args.base, args.model, args.effort, args.max_tokens, args.timeout)
    sm.wait_ready()
    if args.model is None:
        try:
            sm.model = sm.get("/v1/models", timeout=10).json()["data"][0]["id"]
        except (OSError, http.client.HTTPException, KeyError, IndexError, ValueError):
            pass
    sm.run("1 tool call (non-stream)", sm.case_non_stream)
    sm.run("2 tool call (stream)", sm.case_stream)
    sm.run("3 multi-turn with a tool result", sm.case_multi_turn)
    sm.run("4 disconnect mid-stream", sm.case_disconnect)
    sm.run("5 forced tool_choice", sm.case_forced)
    failed = [n for n, ok, _ in sm.results if not ok]
    print(f"{len(sm.results) - len(failed)}/{len(sm.results)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
