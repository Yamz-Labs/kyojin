"""mu_served_test.py: SERVED numbers of the multi-session server (serve.py --slots N) over HTTP, run inside a job wrapper.
Starts the server on a free port, waits for /health, then: 1 / 2 / 4 concurrent streaming chat requests (aggregate + per-session tok/s, ttft), and a long request
arriving while 3 sessions decode (its ttft, the running sessions' largest gap between stream chunks during its prefill, per sched chunk size = one server start each).
Env: MU_PORT (8341), MU_SLOTS (4), MU_CTX (8192), MU_NEW (96), MU_CHUNK (1024), MU_PFLEN (2048), MU_SPEC (1 = speculative requests)."""
import http.client
import json
import os
import subprocess
import sys
import threading
import time

tree = os.getcwd()
PORT = int(os.environ.get("MU_PORT", "8341"))
SLOTS = int(os.environ.get("MU_SLOTS", "4"))
CTX = int(os.environ.get("MU_CTX", "8192"))
NEW = int(os.environ.get("MU_NEW", "96"))
CHUNK = int(os.environ.get("MU_CHUNK", "1024"))
PFLEN = int(os.environ.get("MU_PFLEN", "2048"))
SPEC = os.environ.get("MU_SPEC", "1") == "1"
GGUF = "/path/to/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf"
PROMPTS = ["Explain in a few short paragraphs why the sky is blue and why sunsets are red.",
           "Ecris une courte histoire sur un chat qui apprend a naviguer sur un petit bateau.",
           "Write a Python function that merges two sorted lists, then explain its complexity.",
           "List five practical tips to keep a sourdough starter healthy, with one sentence of reason each."]


def post(path, body, res):
    t0 = time.perf_counter()
    res.update(t0=t0, chunks=[], usage=None)
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    res["status"] = r.status
    buf = b""
    while True:
        part = r.read1(65536)
        if not part:
            break
        buf += part
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            if not ev.startswith(b"data: ") or ev.endswith(b"[DONE]"):
                continue
            j = json.loads(ev[6:])
            if j.get("usage"):
                res["usage"] = j["usage"]
            ch = j.get("choices") or []
            if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content") or ch[0].get("text")):
                res["chunks"].append(time.perf_counter())
    res["t1"] = time.perf_counter()
    c.close()


def chat(i):
    return "/v1/chat/completions", {"messages": [{"role": "user", "content": PROMPTS[i % 4]}], "max_tokens": NEW, "stream": True, "stream_options": {"include_usage": True},
                                    "chat_template_kwargs": {"reasoning_effort": "no_think"}, "speculative": SPEC}


def launch(bodies, delay=0.0):
    rs = [dict() for _ in bodies]
    th = [threading.Thread(target=post, args=(p, b, r)) for (p, b), r in zip(bodies, rs)]
    for t in th:
        t.start()
        time.sleep(delay)
    return rs, th


def summarise(rs):
    t_lo = max(r["chunks"][0] for r in rs)
    t_hi = min(r["t1"] for r in rs)
    n = [r["usage"]["completion_tokens"] for r in rs]
    per = []
    for r in rs:                                     # tokens are streamed per round: count chunks in the window, scaled by tokens per chunk
        tot = r["usage"]["completion_tokens"]
        k = len(r["chunks"])
        inwin = sum(1 for t in r["chunks"] if t_lo < t <= t_hi)
        per.append(round(inwin * (tot / max(k, 1)) / max(t_hi - t_lo, 1e-9), 2))
    return dict(B=len(rs), agg_tok_s=round(sum(per), 2), per_session=per, window_s=round(t_hi - t_lo, 2), tokens=n,
                ttft=[round(r["chunks"][0] - r["t0"], 2) for r in rs])


def serve_up(chunk):
    env = dict(os.environ, HY4_WARMUP_TOKENS="64")
    cmd = [sys.executable, "-u", os.path.join(tree, "tools", "hy4", "serve.py"), "--model", GGUF, "--port", str(PORT), "-c", str(CTX), "--slots", str(SLOTS),
           "--sched-chunk", str(chunk), "--default-max-tokens", str(NEW)]
    p = subprocess.Popen(cmd, env=env, stdout=open(f"/tmp/mu_serve_{PORT}.log", "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 420:
        try:
            c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                return p
        except Exception:      # noqa: BLE001
            pass
        if p.poll() is not None:
            raise RuntimeError("server exited: " + open(f"/tmp/mu_serve_{PORT}.log").read()[-1500:])
        time.sleep(3)
    raise RuntimeError("server not ready")


def main():
    wiki = json.load(open("/path/to/wiki_ids.json"))
    out = []
    for ci, chunk in enumerate([int(x) for x in os.environ.get("MU_CHUNKS", str(CHUNK)).split(",")]):
        p = serve_up(chunk)
        try:
            print("served: ready, chunk", chunk, flush=True)
            if ci == 0:
                for B in (1, 2, 4):
                    post_all = launch([chat(i) for i in range(B)])
                    for t in post_all[1]:
                        t.join()
                    time.sleep(1)
                    rs, th = launch([chat(i) for i in range(B)])
                    for t in th:
                        t.join()
                    r = summarise(rs)
                    print("SERVED", json.dumps(r), flush=True)
                    out.append(r)
            rs, th = launch([chat(i) for i in range(3)])
            while not all(r.get("chunks") for r in rs):
                time.sleep(0.05)
            time.sleep(1.5)
            late = {}
            ts = time.perf_counter()
            tl = threading.Thread(target=post, args=("/v1/completions", {"prompt": wiki[:PFLEN], "max_tokens": 16, "stream": True, "stream_options": {"include_usage": True}, "speculative": SPEC}, late))
            tl.start()
            tl.join()
            te = time.perf_counter()
            for t in th:
                t.join()
            gaps = []
            for r in rs:
                tt = [t for t in r["chunks"] if ts - 0.5 <= t <= te + 0.5]
                gaps += [b - a for a, b in zip(tt, tt[1:])]
            base = []
            for r in rs:
                tt = [t for t in r["chunks"] if ts - 3 <= t <= ts - 0.2]
                base += [b - a for a, b in zip(tt, tt[1:])]
            res = dict(chunk=chunk, pf_tokens=PFLEN, late_ttft=round(late["chunks"][0] - late["t0"], 2) if late.get("chunks") else None,
                       max_gap=round(max(gaps), 3) if gaps else None, median_gap_before=round(sorted(base)[len(base) // 2], 3) if base else None)
            print("SERVED_PF", json.dumps(res), flush=True)
            out.append(res)
        finally:
            p.terminate()
            try:
                p.wait(30)
            except Exception:      # noqa: BLE001
                p.kill()
            time.sleep(3)
    print("MUS_DONE", json.dumps(out), flush=True)


main()
