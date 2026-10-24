# check_cache_hit_equality.py WT OUT.json [prompt_tokens...] : GPU check, one model load. Per prompt: cold run, then the same prompt resumed from each page checkpoint alone (tail checkpoints cleared); compares greedy tokens and the prompt-end recurrent state + MTP carry (max abs diff must be 0.0). NTOK=200 for longer greedy runs. Run under the GPU flock with the serving env.
import os, sys, json, time, pathlib
wt = sys.argv[1]; out = sys.argv[2]
LENS = [int(x) for x in sys.argv[3:]] or [5000, 7000]
sys.path.insert(0, wt); sys.path.insert(0, wt + "/tools/glm")
import torch
MODEL = os.environ.get('KYOJIN_GLM_MODEL', 'models/GLM-5.3-Flash-EXL3-Yamz')
import serve
for _k, _v in serve.SPEED_ENV.items(): os.environ.setdefault(_k, _v)   # what serve.py main() does
os.environ.setdefault('EXL3_MOE_UNION_V2', '1'); serve.apply_eh_sidecar_default(MODEL)
from exllamav3.generator.sampler import GreedySampler
eng = serve.ResidentEngine(MODEL, max_history=2, max_ctx=int(os.environ.get("CTX", "65536")), num_draft=int(os.environ.get("NDRAFT", "2")), vision=False)
g = eng.greedy_generator
files = ["exllamav3/generator/generator.py", "exllamav3/generator/job.py", "exllamav3/modules/gated_delta_net.py", "exllamav3/modules/mla_attn.py", "tools/glm/serve.py", "exllamav3/generator/pagetable.py"]
def mk(i, n):
    t = (pathlib.Path(wt) / files[i % len(files)]).read_text()
    ids = eng.tokenizer.encode(f"Source file {files[i%len(files)]} (variant {i}).\n\n" + t[1000:], encode_special_tokens=True)
    return ids[:, :n].clone()
def run(ids, n=int(os.environ.get("NTOK", "64"))):
    job = eng.Job(input_ids=ids, max_new_tokens=n, sampler=GreedySampler(), stop_conditions=[])
    g.enqueue(job); toks = []; info = {}
    t0 = time.perf_counter(); tf = None
    while g.num_remaining_jobs():
        for ev in g.iterate():
            if ev.get("token_ids") is not None and ev.get("stage") == "streaming":
                if tf is None: tf = time.perf_counter() - t0
                toks += ev["token_ids"].flatten().tolist()
            if ev.get("eos"): info = {k: ev.get(k) for k in ("cached_tokens", "new_tokens", "prompt_tokens", "time_prefill", "time_generate")}
    torch.cuda.synchronize()
    info["ttft"] = tf; info["wall"] = time.perf_counter() - t0
    return toks, info
res = []
rc = g.recurrent_cache
chunks = []
_f = eng.model.forward
def fw(*a, **k):
    ii = k.get("input_ids", a[0] if a else None)
    if ii is not None and ii.shape[-1] > 8: chunks.append(int(ii.shape[-1]))
    return _f(*a, **k)
eng.model.forward = fw
run(mk(99, 600), 4)   # warm
for i, n in enumerate(LENS):
    ids = mk(i, n); n = ids.shape[-1]
    chunks.clear()
    g.clear_prompt_ckpts()
    cold, ic = run(ids); cchunks = list(chunks)
    cold_tail = g.tail_ckpts[-1] if g.tail_ckpts else None
    snap = list(rc.items())
    pos = {k: v["position"] for k, v in snap}
    mine = sorted(set(pos.values()))
    r = {"i": i, "len": n, "cold": cold, "ic": ic, "cold_chunks": cchunks, "positions": mine, "arms": {}}
    tail, it = run(ids); r["tail_eq"] = tail == cold; r["it"] = it
    for P in sorted(mine, reverse=True):
        if P % 256 or P < 256: continue
        g.clear_prompt_ckpts()
        for k in list(rc.keys()): rc.pop(k)
        for k, v in snap:
            if v["position"] == P: rc[k] = v
        chunks.clear()
        t, inf = run(ids)
        fd = next((j for j, (a, b) in enumerate(zip(cold, t)) if a != b), None)
        dif = None
        if cold_tail is not None and g.tail_ckpts:
            nt = g.tail_ckpts[-1]; dif = {}
            for k, v in cold_tail["stashed"].items():
                if isinstance(v, tuple):
                    dif[str(k)[-30:]] = [float((a.float() - b.float()).abs().max()) for a, b in zip(v, nt["stashed"][k])]
            if cold_tail.get("carry") is not None and nt.get("carry") is not None:
                dif["carry"] = float((cold_tail["carry"].float() - nt["carry"].float()).abs().max())
        sd = None
        if dif:
            vals = [x for k, v in dif.items() for x in (v if isinstance(v, list) else [v])]
            sd = {"max": max(vals), "n_nonzero": sum(1 for x in vals if x > 0), "n": len(vals), "carry": dif.get("carry"), "first_layers": list(dif.items())[:3]}
        r["arms"][P] = {"eq": t == cold, "first_div": fd, "cached": inf["cached_tokens"], "chunks": list(chunks), "wall": inf["wall"], "toks": t, "statediff": sd}
        print(n, "arm", P, r["arms"][P]["eq"], fd, inf["cached_tokens"], chunks, json.dumps(sd), flush=True)
        rc.clear(); [rc.__setitem__(k, v) for k, v in snap]
    res.append(r); print(json.dumps({k: v for k, v in r.items() if k not in ("cold", "arms")}), flush=True)
    pathlib.Path(out).write_text(json.dumps(res))
