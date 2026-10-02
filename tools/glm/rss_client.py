"""rssleak1 client: does the SERVED path (serve.py over HTTP) grow host RSS per request?

Phases, exactly the brief's: 30 short requests (256-token prompt + 16 tokens), then 10 with 16K
prompts. Unique prompts, greedy, so the outputs double as the bit-identity gate. Per request it
records the server process's own VmRSS/VmHWM (the authoritative number, read from
/proc/<pid>/status) next to the client-side view and the reported token counts; the server writes
its torch device/host and recurrent-cache numbers to EXL3_SERVE_RSS_LOG.

Usage: python tools/glm/rss_client.py --pid P --out OUT.json [--port 8299] [--n-short 30]
       [--n-long 10] [--short-tok 256] [--long-tok 16384] [--gen 16] [--seed 7]
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path

import aiohttp

STATUS_KEYS = ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem")


def proc_status(pid):
    out = {}
    try:
        for line in open(f"/proc/{pid}/status"):
            k = line.split(":")[0]
            if k in STATUS_KEYS:
                out[k] = int(line.split()[1]) // 1024
    except OSError:
        pass
    return out


async def one(session, url, model, prompt, gen, sem):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": gen, "temperature": 0.0, "top_p": 1.0, "stream": False}
    async with sem:
        t0 = time.perf_counter()
        async with session.post(url + "/v1/chat/completions", json=body) as r:
            j = await r.json()
            if r.status != 200:
                return {"error": str(j)[:200]}
        return {"wall": time.perf_counter() - t0, "j": j}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8299)
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="glm-5.3-exl3")
    ap.add_argument("--n-short", type=int, default=30)
    ap.add_argument("--n-long", type=int, default=10)
    ap.add_argument("--short-tok", type=int, default=256)
    ap.add_argument("--long-tok", type=int, default=16384)
    ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    words = (Path.home() / "bench/ppl/wiki.test.raw").read_text(encoding="utf-8").split()
    rng = random.Random(a.seed)
    url = f"http://127.0.0.1:{a.port}"
    sem = asyncio.Semaphore(1)          # single-flight, like the lane
    recs = []

    # words per token measured on this tokenizer, refined once from the first reply
    wpt = 0.75
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        for phase, n, target in (("short", a.n_short, a.short_tok),
                                  ("long", a.n_long, a.long_tok)):
            for i in range(n):
                k = max(int(target * wpt), 8)
                s0 = rng.randrange(0, len(words) - k)
                prompt = (f"[r{phase}{i} {rng.random():.9f}] " + " ".join(words[s0:s0 + k]) +
                          "\nAnswer in one short sentence.")
                r = await one(session, url, a.model, prompt, a.gen, sem)
                if "error" in r:
                    recs.append({"phase": phase, "i": i, **r})
                    print("ERROR", r["error"], flush=True)
                    continue
                j = r["j"]
                u = j.get("usage", {})
                ptok = u.get("prompt_tokens", 0)
                wpt = wpt * 0.7 + (k / max(ptok, 1)) * 0.3
                msg = j["choices"][0]["message"]
                text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
                rec = {"phase": phase, "i": i, "ptok": ptok, "gtok": u.get("completion_tokens"),
                       "wall": round(r["wall"], 2), "prompt_kw": k,
                       "hash": hashlib.sha1(text.encode()).hexdigest()[:12],
                       "srv": proc_status(a.pid)}
                recs.append(rec)
                if len(recs) % 5 == 0 or i == n - 1:
                    print(f"[{phase:5s} {i:3d}] ptok {ptok:6d} wall {r['wall']:7.2f} "
                          f"VmRSS {rec['srv'].get('VmRSS', -1):6d} MB  "
                          f"anon {rec['srv'].get('RssAnon', -1):6d}  "
                          f"file {rec['srv'].get('RssFile', -1):5d}  "
                          f"hwm {rec['srv'].get('VmHWM', -1):6d}  {rec['hash']}", flush=True)
                Path(a.out).write_text(json.dumps(recs, indent=1))
    Path(a.out).write_text(json.dumps(recs, indent=1))
    print("DONE", len(recs), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
