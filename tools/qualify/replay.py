"""Day-0 autotune by shape replay: start the pack's real server (same flags as the qualify serve stage) on a FRESH tune cache and drive a
fixed sweep of row counts through it, so the engine's own autotuner times every kernel the model reaches in the served path: fused
qkv / gate-up multi-GEMMs, MoE expert GEMMs, lm_head, CUDA-graph captures, whatever the architecture is. The winners it appends are
exactly the cache for this pack on this device; `finish()` copies them to <pack>/tuning/<signature>.bin (+ manifest).

The sweep covers the rows a request can put through a GEMM (the autotune key's row classes 2,4,8,16,24,32,48,64, then the dense path):
  - prompts of exact token counts (token-id prompts on /v1/completions): single forward of L rows, and L-row tails behind a full chunk
  - N concurrent decodes, N = 2..sessions (batched decode rows, speculative verify rows ride on the same classes)
Pure planning (sweep_requests, signature file name) is CPU-testable; the driver needs a server.
"""
from __future__ import annotations

import json
import random
import threading
import time
import urllib.request
from pathlib import Path

PROMPT_ROWS = (2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 200, 256)
CHUNK = 2048            # serve.py --max-chunk-size default: a prompt of CHUNK + r tokens ends in an r-row forward
TAILS = (3, 5, 9, 17, 25, 33, 49, 65, 100)


def sweep_requests(sessions: int, ctx: int, chunk: int = CHUNK) -> list[dict]:
    """Ordered sweep. Every entry: {kind, tokens, conc, max_tokens}. Lengths never exceed ctx - 64."""
    cap = ctx - 64
    out = [dict(kind="prompt", tokens=r, conc=1, max_tokens=8) for r in PROMPT_ROWS if r <= cap]
    out += [dict(kind="tail", tokens=chunk + r, conc=1, max_tokens=4) for r in TAILS if chunk + r <= cap]
    for n in range(2, sessions + 1):
        out.append(dict(kind="decode", tokens=48, conc=n, max_tokens=24))
    out.append(dict(kind="prompt", tokens=min(4096, cap), conc=1, max_tokens=16))    # full chunks: the dense/reconstruct path warms too
    return out


def token_prompt(n: int, seed: int, vocab: int = 30000) -> list[int]:
    """n distinct-ish ordinary token ids (random in [1000, vocab)): no shared prefix between requests, no special tokens."""
    rng = random.Random(seed)
    return [rng.randrange(1000, vocab) for _ in range(n)]


def complete(port: int, ids: list[int], max_tokens: int, model: str = "qual", timeout: int = 1800) -> dict:
    body = json.dumps({"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json", "Authorization": "Bearer x"})
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        return dict(ok=True, s=round(time.time() - t0, 3), usage=r.get("usage"))
    except Exception as e:   # noqa: BLE001
        return dict(ok=False, s=round(time.time() - t0, 3), err=repr(e)[:200])


def run_sweep(port: int, sessions: int, ctx: int, log=print, model: str = "qual") -> list[dict]:
    res = []
    for i, rq in enumerate(sweep_requests(sessions, ctx)):
        out = [None] * rq["conc"]

        def one(j):
            out[j] = complete(port, token_prompt(rq["tokens"], seed=i * 100 + j), rq["max_tokens"], model)
        th = [threading.Thread(target=one, args=(j,)) for j in range(rq["conc"])]
        t0 = time.time()
        [t.start() for t in th]; [t.join() for t in th]
        res.append(dict(**rq, wall_s=round(time.time() - t0, 2), ok=all(o["ok"] for o in out), err=next((o["err"] for o in out if not o["ok"]), None)))
        log(res[-1])
    return res
