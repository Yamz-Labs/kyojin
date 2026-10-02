"""Cross-restart check of the /slots disk cache of an EXL3 server (serve.py).

1. start the server, send a long-preamble request (greedy), save slot 0 to disk;
2. kill and restart the server (so no page survives in memory), restore the file;
3. send the same request: cache_n must cover the saved prefix, and the greedy text must be identical;
4. send a follow-up turn on top of it: cache_n must still cover the saved prefix.

usage: slot_check.py --port 8290 --model-id X --out res.json -- <server command...>
"""
import argparse, json, os, signal, socket, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

CORPUS = Path.home() / "bench/ppl/wiki.test.raw"


def call(base, path, body=None, timeout=3600):
    req = urllib.request.Request(base + path, json.dumps(body).encode() if body is not None else None,
                                 {"content-type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout))
    except urllib.error.HTTPError as e:
        sys.exit(f"{path}: HTTP {e.code} {e.reason}: {e.read().decode(errors='replace')}")


def start(cmd, log, port):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            sys.exit(f"port {port} is already in use: refusing to test against a stale server")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return p


def wait_up(base, p, limit=1800):
    t0 = time.time()
    while time.time() - t0 < limit:
        if p.poll() is not None:
            sys.exit(f"server exited rc={p.returncode}")
        try:
            json.load(urllib.request.urlopen(base + "/v1/models", timeout=5))
            return time.time() - t0
        except Exception:
            time.sleep(3)
    sys.exit("server did not come up")


def stop(p):
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8290)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--words", type=int, default=4000)
    ap.add_argument("--out", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    base = f"http://127.0.0.1:{a.port}"
    words = CORPUS.read_text(encoding="utf-8").split()[:a.words]
    system = "You are a careful assistant. Reference material follows.\n" + " ".join(words)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": "In two sentences, what is the first article about?"}]

    def chat(messages, n=48):
        t0 = time.perf_counter()
        d = call(base + "/v1", "/chat/completions", {"model": a.model_id, "messages": messages,
                                                     "max_tokens": n, "temperature": 0})
        m = d["choices"][0]["message"]
        return {"wall_s": round(time.perf_counter() - t0, 3), "timings": d.get("timings"),
                "prompt_tokens": d["usage"]["prompt_tokens"],
                "text": (m.get("reasoning_content") or "") + "|" + (m.get("content") or "")}

    res = {}
    fname = f"slotcheck-{a.model_id}.bin"
    with open(Path(a.out).with_suffix(".log"), "w") as log:
        p = start(cmd, log, a.port)
        try:
            res["load1_s"] = round(wait_up(base, p), 1)
            res["cold"] = chat(msgs)
            res["slots_before"] = [{k: v for k, v in s.items() if k != "prompt"} for s in call(base, "/slots")]
            res["save"] = call(base, "/slots/0?action=save", {"filename": fname})
        finally:
            stop(p)
        p = start(cmd, log, a.port)
        try:
            res["load2_s"] = round(wait_up(base, p), 1)
            res["erase"] = call(base, "/slots/0?action=erase", {})
            res["restore"] = call(base, "/slots/0?action=restore", {"filename": fname})
            res["warm"] = chat(msgs)
            turn2 = msgs + [{"role": "assistant", "content": res["warm"]["text"].split("|", 1)[1]},
                            {"role": "user", "content": "And the second one?"}]
            res["turn2"] = chat(turn2)
        finally:
            stop(p)
    n_saved = res["save"]["n_saved"]
    res["verdict"] = {
        "n_saved": n_saved,
        "warm_cache_n": (res["warm"]["timings"] or {}).get("cache_n"),
        "turn2_cache_n": (res["turn2"]["timings"] or {}).get("cache_n"),
        "same_text": res["cold"]["text"] == res["warm"]["text"],
        "cold_wall_s": res["cold"]["wall_s"], "warm_wall_s": res["warm"]["wall_s"],
    }
    v = res["verdict"]
    v["n_copied"] = res["restore"]["n_read"]
    v["pass"] = bool(n_saved > 0 and v["n_copied"] == n_saved and v["warm_cache_n"] >= n_saved and v["same_text"]
                     and v["turn2_cache_n"] >= n_saved)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("VERDICT", json.dumps(v), flush=True)


if __name__ == "__main__":
    main()
