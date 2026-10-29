#!/usr/bin/env python3
"""GPU gate of /admin/knobs: toggling a knob in a live server equals a fresh server started with that value, both ways.

  knob_md5.py --serve 'python -u tools/qwen/serve.py --model M --port {port} --model-id q ...' --knob EXL3_GEMV_TRI=0 [--knob ...]

S1 (started with --admin-knobs, knob at its default): md5 of greedy replies -> d0; POST the knob -> a1; POST it back -> d2.
S2 (fresh, knob in the environment, no admin): md5 -> f.  PASS needs d2 == d0 and f == a1 (and reports whether the knob changed the
bytes at all). The caller holds the GPU lock; one run is two model loads plus a few short requests.
"""
import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_inproc import http, load_prompts, request  # noqa: E402


def start(cmd, port, env, log):
    p = subprocess.Popen(shlex.split(cmd.format(port=port)), env={**os.environ, **env}, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                         start_new_session=True)
    for _ in range(900):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).status == 200:
                return p
        except Exception:
            pass
        if p.poll() is not None:
            sys.exit(f"server died, see {log}")
        time.sleep(2)
    stop(p)
    sys.exit("server did not become healthy")


def stop(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)                 # this server's own process group only
        p.wait(timeout=60)
    except Exception:
        os.killpg(p.pid, signal.SIGKILL)


def md5s(port, model, prompts, tokens):
    """md5 of each greedy reply: the text and the logprob of every token (a knob that moves the last bits shows there before it flips a token)"""
    out = []
    for pr in prompts:
        r = http(port, "/v1/chat/completions", {"model": model, "max_tokens": tokens, "temperature": 0, "enable_thinking": False,
                                                "logprobs": True, "top_logprobs": 1, **pr})["choices"][0]
        lp = [(e["token"], e["logprob"]) for e in (r.get("logprobs") or {}).get("content", [])]
        out.append(hashlib.md5((r["message"].get("content") or "").encode() + json.dumps(lp).encode()).hexdigest()[:12])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", required=True, help="server command, {port} is replaced; --admin-knobs is appended for S1")
    ap.add_argument("--knob", action="append", required=True, help="NAME=VALUE (the non-default value)")
    ap.add_argument("--port", type=int, default=8691)
    ap.add_argument("--synth", default="1500,3000")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--log-dir", default="/tmp")
    ap.add_argument("--require-change", action="store_true", help="also fail when the knob leaves the bytes unchanged")
    a = ap.parse_args()
    if a.corpus is None:
        from ab_inproc import CORPUS
        a.corpus = CORPUS
    knobs = dict(k.split("=", 1) for k in a.knob)
    prompts = load_prompts(a)
    p1 = start(a.serve + " --admin-knobs", a.port, {}, f"{a.log_dir}/knob_md5_s1.log")
    try:
        model = http(a.port, "/v1/models")["data"][0]["id"]
        md5s(a.port, model, prompts[:1], 8)                                      # warm
        d0 = md5s(a.port, model, prompts, a.tokens)
        eff = http(a.port, "/admin/knobs", knobs)["effective"]
        assert all(eff[k] == v for k, v in knobs.items()), eff
        a1 = md5s(a.port, model, prompts, a.tokens)
        eff = http(a.port, "/admin/knobs", {k: None for k in knobs})["effective"]
        d2 = md5s(a.port, model, prompts, a.tokens)
    finally:
        stop(p1)
    time.sleep(5)
    p2 = start(a.serve, a.port, knobs, f"{a.log_dir}/knob_md5_s2.log")
    try:
        f = md5s(a.port, model, prompts, a.tokens)                               # first requests after load: no warm-up needed for md5
    finally:
        stop(p2)
    print(json.dumps({"knobs": knobs, "default_first": d0, "toggled": a1, "toggled_back": d2, "fresh_with_knob": f}, indent=1))
    changed = d0 != a1
    ok = d2 == d0 and f == a1 and (changed or not a.require_change)
    print(f"bytes changed by the knob: {changed}; back == default: {d2 == d0}; toggled == fresh: {f == a1}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
