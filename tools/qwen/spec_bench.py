# measure speculation policies, lossless check vs plain (t0). Arms: plain, fix2 (R=3), dyn, mix (dyn + n-gram).
# usage: spec_bench.py OUT.json [--ntok 160] [--budget 540] [--arms plain,fix2,dyn,mix]
import argparse, json, os, sys, time, torch
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--model", default="~/models/qwen38-td205")
ap.add_argument("--ntok", type=int, default=160); ap.add_argument("--budget", type=float, default=540)
ap.add_argument("--arms", default="plain,fix2,dyn,mix"); ap.add_argument("--nper", type=int, default=6)
ap.add_argument("--thf", type=float, default=0.6); ap.add_argument("--thv", type=float, default=0.3)
a = ap.parse_args()
torch.set_grad_enabled(False)
os.environ.setdefault("EXL3_MTP_FUSE_CATCHUP", "0")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools", "glm")); sys.path.insert(0, os.path.join(ROOT, "tools", "qwen"))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler
from transformers import AutoTokenizer
from prompts10 import PROMPTS
from spec_prompts import EXTRA
import spec_policy
from exllamav3.modules import moe_fused
from exllamav3.modules.quant import exl3 as qexl3
# arm plain0 = plain greedy with the previous engine config (mf7 MoE kernel, lm_head loop, no GEMV_R_DEC1), every other arm = current config
CUR = (os.environ.get("EXL3_GEMV_R_DEC1", "0"), qexl3.WIDE_R["on"], moe_fused.MF9["variant"])
def set_cfg(old):
    os.environ["EXL3_GEMV_R_DEC1"], qexl3.WIDE_R["on"], moe_fused.MF9["variant"] = ("0", False, 0) if old else CUR
cfg = Config.from_directory(a.model); model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
cache = Cache(model, max_num_tokens=4096, max_history=3); model.load(max_chunk_size=2048, progressbar=False)
dm = Model.from_config(cfg, component="mtp"); dcache = Cache(dm, max_num_tokens=4096, max_history=3); dm.load(progressbar=False)
hf = AutoTokenizer.from_pretrained(a.model)
def ids(u):
    p = hf.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok.encode(p, add_bos=False, encode_special_tokens=True)
def make(arm):
    if arm in ("plain", "plain0"): return Generator(model=model, cache=cache, tokenizer=tok), None, {}
    n = 2 if arm == "fix2" else 3
    g = Generator(model=model, cache=cache, tokenizer=tok, draft_model=dm, draft_cache=dcache, num_draft_tokens=n)
    stats = {}
    if arm == "dyn": return g, spec_policy.install(g, dm, model, a.thf, a.thv, 3, None, stats), stats
    if arm == "mix": return g, spec_policy.install(g, dm, model, a.thf, a.thv, 3, (2, 3, 5), stats), stats
    return g, None, stats
GEN = {}
def run(arm, text, ntok):
    set_cfg(arm == "plain0")
    if arm not in GEN:
        for k in list(GEN):
            if GEN[k][1]: GEN[k][1]()
        GEN.clear()
        g, un, stats = make(arm); GEN[arm] = (g, un, stats)
    g, un, stats = GEN[arm]; stats.clear()
    job = Job(input_ids=ids(text), max_new_tokens=ntok, sampler=ArgmaxSampler(), stop_conditions=[])
    g.enqueue(job); toks = []; ttft = None; t0 = time.perf_counter(); nit = 0
    while g.num_remaining_jobs():
        for r in g.iterate():
            t = r.get("token_ids")
            if t is not None and t.numel():
                if ttft is None: torch.cuda.synchronize(); ttft = time.perf_counter() - t0
                toks += t.flatten().tolist()
        nit += 1
    torch.cuda.synchronize(); dec = time.perf_counter() - t0 - ttft
    return {"toks": toks, "tps": (len(toks) - 1) / dec, "iters": nit, "stats": dict(stats)}
arms = a.arms.split(",")
allp = [(k, i, p) for k in ("chat", "prose", "code") for i, p in enumerate(PROMPTS[k][:a.nper])] + [(k, i, p) for k, v in EXTRA.items() for i, p in enumerate(v[:4])]
for arm in arms: run(arm, "Hello", 24)
R = []; t0 = time.time()
for k, i, p in allp:
    if time.time() - t0 > a.budget: print("budget reached", flush=True); break
    row = {"kind": k, "idx": i}
    for arm in arms:
        row[arm] = run(arm, p, a.ntok)
    ref = row["plain"]["toks"] if "plain" in row else None
    for arm in arms:
        if ref is not None and arm != "plain":
            n = next((j for j, (u, v) in enumerate(zip(ref, row[arm]["toks"])) if u != v), None)
            row[arm]["first_div"] = n if n is not None else (None if len(ref) == len(row[arm]["toks"]) else min(len(ref), len(row[arm]["toks"])))
    R.append(row)
    print(k, i, " ".join(f"{arm} {row[arm]['tps']:.1f}" + ("" if arm == "plain" or row[arm].get("first_div") is None and True and ref is not None and row[arm].get("first_div") is None else f" DIV@{row[arm].get('first_div')}") for arm in arms), flush=True)
    json.dump(R, open(a.out, "w"))
import collections
print("SUMMARY (mean tok/s, identical/total)")
for k in ("chat", "prose", "code", "multi", "copy"):
    rs = [r for r in R if r["kind"] == k]
    if not rs: continue
    print("SUMMARY", k, " ".join(f"{arm} {sum(r[arm]['tps'] for r in rs)/len(rs):.1f}" + (f" ({sum(1 for r in rs if r[arm].get('first_div') is None)}/{len(rs)})" if arm != 'plain' else "") for arm in arms), flush=True)
for arm in ("dyn", "mix"):
    ss = [r[arm]["stats"] for r in R if arm in r]
    print("SUMMARY stats", arm, {k: sum(s.get(k, 0) for s in ss) for k in ("mtp", "ng", "rows")}, flush=True)
