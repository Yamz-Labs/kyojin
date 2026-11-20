"""spec2_arms.py: ONE staged runner job for the S7 draft-depth A/B (perf/hy4-strix-spec2). Runner driver: cwd = tree, env = lane env + XENV, GPU locks held by the runner.
Question: tau and ms per round on the 6 head_tau prompts (code / prose / chat x 2, 256 tok, greedy, spec) for
  A = deployed b2 arm C (k = 3)    B = A with --num-draft 2 (plain shallow chain: the free control)    C = A + HY4_SPEC2_GATE=1 (k = 2 base graph + confidence-gated tail step)
order SP2_ORDER (default ABCA: the second A is the drift control, reported as A2). Phase 0: graph-replay ms per verify round at T = 1..6 (4K, arm C flags) = the missing T3 / T5 / T6 of the cost model.
Metric per class: tok/s, tau, ms per round = 1000 * tau / tok/s, paired delta vs A (mean of both A runs). Gate rule (s7_model.py): C ships when it is >= B on code by 2 % and not below B on chat / prose by more than 1 %;
B ships alone when C does not beat it; neither when both lose to A on code by more than 2 %.
env: OD_PORT OD_NAME GGUF N1119 B2_H B2_DIR B2_GM B2_CO B2_PASSES SP2_ORDER SP2_REPS (default 2) SP2_LEVERS (default 1)
Output: table on stdout + <N1119>/out/<OD_NAME>.spec2.json. ~50 min GPU: TMAX 3300."""
import hashlib, json, os, re, signal, statistics, subprocess, sys, time, urllib.request
E = os.environ; tree = os.getcwd(); t0 = time.time()
H = E.get("B2_H", "/home/kamel"); N = E.get("N1119", f"{H}/scratch/n1119")
name, port = E["OD_NAME"], E["OD_PORT"]; tmax = int(E.get("OD_TMAX", "3300"))
GGUF = E.get("GGUF", f"{H}/models/hy4-flash-gguf/Hy4-flash-Q3_K-00001-of-00002.gguf")
B2 = E.get("B2_DIR", f"{H}/scratch/b2"); OUT = f"{N}/out"; os.makedirs(OUT, exist_ok=True)
GM = E.get("B2_GM", f"{B2}/gm_build"); CO = E.get("B2_CO", f"{B2}/moe_dec.co")
COMMON = {"EXL3_GM_BUILD_DIR": GM, "HY4_GM_EXT": f"{GM}/exl3_gm_ext.so", "MOE_DEC_CO": CO, "PCTXS": "4096", "PTS": "1,2,3,4,5,6", "REPS": "3", "LV_CTX": "16384"}
ARM_C = {"HY4_HEAD_Q8": "1", "HY4_GRAPH_FORK": "1", "HY4_RG_TUNE": "1", "HY4_BG_V2": "1", "HY4_DENSE_V2": "1", "HY4_MOE_V2": "1",
         "HY4_DRAFT_MOE_TILES": "1", "HY4_DRAFT_MLA_BG": "1", "HY4_DRAFT_IX_GV": "1",
         "HY4_DRAFT_VOCAB": "0", "HY4_DRAFT_VOCAB_REQ": "0", "HY4_DRAFT_TOPK": "0", "HY4_NGRAM_REPLACE": "0", "HY4_SPEC2_GATE": "0", "HY4_REQ_ROUNDLOG": "1"}
ARMS = {"A": ({}, "3"), "B": ({}, "2"), "C": ({"HY4_SPEC2_GATE": "1", "HY4_SPEC2_KB": "2"}, "3")}        # arm -> (extra env, --num-draft)
def log(*a): print(time.strftime("%T"), *a, flush=True)

def stop(srv):
    try: os.killpg(srv.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    time.sleep(8)
    try: os.killpg(srv.pid, signal.SIGKILL)
    except ProcessLookupError: pass

def chat(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", "X-Kyojin-Bench": "1"})
    r = json.load(urllib.request.urlopen(req, timeout=900)); tm = r.get("timings") or {}; txt = r["choices"][0]["message"].get("content") or ""
    return {"tps": tm.get("predicted_per_second"), "tau": tm.get("tau"), "n": tm.get("predicted_n"), "prompt_n": tm.get("prompt_n"), "md5": hashlib.md5(txt.encode()).hexdigest()[:12]}

def gate_lines(path):
    """per request (server log line 'req N: ...'): rounds, tau, ext rounds of the gate when present"""
    out = []
    for l in open(path, errors="replace"):
        m = re.search(r"^req (\d+): .*rounds (\d+) tau ([0-9.]+) hist (\[[^\]]*\])", l)
        if m:
            g = re.search(r"gate \{'rounds': (\d+), 'ext_rounds': (\d+)", l)
            out.append({"req": int(m.group(1)), "rounds": int(m.group(2)), "tau": float(m.group(3)), "hist": json.loads(m.group(4)), "gate_rounds": g and int(g.group(1)), "ext_rounds": g and int(g.group(2))})
    return out

def class_means(pts):
    r = {}
    for cls in ("code", "prose", "chat"):
        v = [p for p in pts.values() if p["cls"] == cls and p["tau"] and p["tps"]]
        if v:
            r[cls] = {"tau": statistics.mean(p["tau"] for p in v), "tps": statistics.mean(p["tps"] for p in v), "round_ms": statistics.mean(1000 * p["tau"] / p["tps"] for p in v)}
    return r

def main():
    res = {"levers": None, "arms": {}, "order": E.get("SP2_ORDER", "ABCA")}
    def save(): json.dump(res, open(f"{OUT}/{name}.spec2.json", "w"), indent=1)
    if E.get("SP2_LEVERS", "1") == "1":
        lg = f"{OUT}/{name}.levers.log"
        try:
            subprocess.run([sys.executable, "-u", f"{tree}/tools/hy4/step/b2_levers_arm.py"], cwd=tree, env={**E, **COMMON, **ARM_C}, stdout=open(lg, "w"), stderr=subprocess.STDOUT, timeout=1000)
        except subprocess.TimeoutExpired:
            log("SP2 levers TIMEOUT")
        line = next((l for l in open(lg, errors="replace").read().splitlines() if l.startswith("LEVERS_RESULT ")), None)
        if line:
            res["levers"] = json.loads(line.split(" ", 1)[1]); log("SP2 replay ms", {k: v["median"] for k, v in res["levers"]["round_ms"].items()})
        else:
            log("SP2 levers: NO RESULT")
        save()
    passes = [p.split("|", 3) for p in open(E.get("B2_PASSES", f"{B2}/head_tau.passes")).read().strip().split("\n")]
    seen = {}
    for letter in res["order"]:
        seen[letter] = seen.get(letter, 0) + 1
        key = letter if seen[letter] == 1 else f"{letter}{seen[letter]}"
        if time.time() - t0 > tmax - 600: log("SP2 skipped (deadline)", key); continue
        extra, nd = ARMS[letter]
        env = {**E, **COMMON, **ARM_C, **extra, "HY4_PRECAPTURE": "all"}
        sl = f"{OUT}/{name}.arm_{key}.server.log"
        srv = subprocess.Popen([sys.executable, "-u", f"{tree}/tools/hy4/serve.py", "--model", GGUF, "--port", port, "-c", "12288", "--model-id", "Hy4-Kyojin", "--num-draft", nd], env=env,
                               stdout=open(sl, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        ok, ts = False, time.time()
        while time.time() - ts < 900 and srv.poll() is None:
            try: urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3); ok = True; break
            except Exception: time.sleep(4)
        log("SP2 boot", key, {**extra, "num_draft": nd}, "ready", ok, f"load_s {time.time() - ts:.0f}")
        try:
            if ok:
                pts = {}
                for i, (label, mode, ex, rest) in enumerate(passes):
                    for rep in range(int(E.get("SP2_REPS", "2"))):
                        if time.time() - t0 > tmax - 60: log("SP2 skipped (deadline)", label); break
                        body = {"model": "Hy4-Kyojin", "max_tokens": 256, "temperature": 0, "stream": False, "cache_prompt": False, "speculative": True,
                                "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "no_think"}, **json.loads(ex)}
                        if i == 0 and rep == 0: chat(body)                  # warm-up (graphs, first-touch pages), discarded
                        r = chat(body); cls = label.split("headtau_")[1][:-3].rstrip("0123456789")
                        pts[f"{label}#{rep}"] = {"cls": cls, **r}
                        log(f"SP2 {key} {label}#{rep}: {r['tps'] and round(r['tps'], 2)} tok/s tau {r['tau']} round {r['tau'] and r['tps'] and round(1000 * r['tau'] / r['tps'], 1)} ms md5 {r['md5']}")
                res["arms"][key] = {"letter": letter, "num_draft": nd, "env": extra, "points": pts, "classes": class_means(pts)}
        finally:
            stop(srv)
        res["arms"].get(key, {})["reqs"] = gate_lines(sl)
        save()
    # ---- table
    arms = res["arms"]
    ref = [a for k, a in arms.items() if a["letter"] == "A"]
    print("\nSPEC2 TABLE: per class, mean of the head_tau prompts x reps: tok/s | tau | ms per round; delta tok/s vs A (mean of the A runs)")
    for cls in ("code", "prose", "chat"):
        base = statistics.mean(a["classes"][cls]["tps"] for a in ref if cls in a["classes"]) if ref else None
        for k, a in arms.items():
            c = a["classes"].get(cls)
            if c:
                ext = [q["ext_rounds"] / q["gate_rounds"] for q in a.get("reqs", []) if q.get("gate_rounds")]
                print(f"{cls:<6} arm {k:<3} (num-draft {a['num_draft']}{' gate' if a['env'] else ''}): {c['tps']:.2f} tok/s tau {c['tau']:.3f} round {c['round_ms']:.1f} ms"
                      + (f" | d vs A {100 * (c['tps'] / base - 1):+.1f} %" if base else "") + (f" | gate ext rate {statistics.mean(ext):.2f}" if ext else ""))
    if res["levers"]:
        print("\nSPEC2 replay ms per verify round (4K, arm C flags):", {k: v["median"] for k, v in res["levers"]["round_ms"].items()})
    def mean_cls(letter, cls):
        v = [a["classes"][cls]["tps"] for a in arms.values() if a["letter"] == letter and cls in a["classes"]]
        return statistics.mean(v) if v else None
    for cls in ("code", "prose", "chat"):
        t = {l: mean_cls(l, cls) for l in "ABC"}
        if all(t.values()):
            print(f"SPEC2_VERDICT {cls}: A {t['A']:.2f} B(k=2) {t['B']:.2f} C(gate) {t['C']:.2f} tok/s  B vs A {100 * (t['B'] / t['A'] - 1):+.1f} %  C vs A {100 * (t['C'] / t['A'] - 1):+.1f} %  C vs B {100 * (t['C'] / t['B'] - 1):+.1f} %")
    log("SPEC2_DONE", name, f"{time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
