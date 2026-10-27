"""Flint KV-format gate (copy of tools/qwen/kvgate.py): teacher-forced KL / top-1 of a quantized attention cache vs the fp16 cache, plus needle retrieval.
  kvgate.py kl     --kv 16|K,V [--depth 32768] [--win 512] [--text nat|copy] [--ref REF.pt | --save REF.pt]
  kvgate.py needle --kv 16|K,V --ctx 131072 [--pos 0.1,0.5,0.9] [--seed 0]
Prefix = real text (wiki/c4/technical, coherent run), window = the next --win tokens forced one by one through the decode path
('copy' = the window repeats a passage from near the start, so the model must read far K/V)."""
import argparse, ast, json, os, re, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler.sampler import Sampler

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("kl", "needle"))
ap.add_argument("--model", required=True)
ap.add_argument("--kv", default="16"); ap.add_argument("--depth", type=int, default=32768); ap.add_argument("--win", type=int, default=512)
ap.add_argument("--text", default="nat"); ap.add_argument("--ref"); ap.add_argument("--save")
ap.add_argument("--ctx", type=int, default=131072); ap.add_argument("--pos", default="0.1,0.5,0.9"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--gen", type=int, default=40)
a = ap.parse_args()

cfg = Config.from_directory(a.model); model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
need = a.depth + a.win if a.mode == "kl" else a.ctx
ctx = ((need + 1024 + 255) // 256) * 256
if a.kv == "16": cache = Cache(model, max_num_tokens=ctx, max_batch_size=1)
else:
    kb, vb = (int(x) for x in a.kv.split(","))
    cache = Cache(model, max_num_tokens=ctx, max_batch_size=1, layer_type=CacheLayer_quant, k_bits=kb, v_bits=vb)
t0 = time.time(); model.load(progressbar=False)
print(f"loaded {time.time()-t0:.0f}s alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB kv={a.kv} ctx={ctx}", flush=True)
gen = Generator(model=model, cache=cache, tokenizer=tok, **({"max_chunk_size": a.chunk} if a.chunk else {}))
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "exllamav3", "conversion", "standard_cal_data")

def long_text_ids(n, seed=0):
    """n tokens of coherent real text: wiki, then c4, then technical, then wiki again from another offset (no synthetic filler)."""
    out, tot, files, i = [], 0, ["wiki.utf8", "c4.utf8", "technical.utf8", "code.utf8"], 0
    cache_t = {}
    while tot < n:
        f = files[i % 4]; i += 1
        if f not in cache_t: cache_t[f] = tok.encode(open(f"{D}/{f}", encoding="utf8").read(), add_bos=False)[0].view(-1)
        t = cache_t[f]; take = min(len(t), n - tot + 16, 120000)
        o = (seed * 7919 + i * 31337) % max(1, len(t) - take + 1)
        out.append(t[o:o + take]); tot += take
    return torch.cat(out)[:n]

class TF(Sampler):
    """teacher-forced sampler: returns the next true token, scores the logits against the reference run."""
    def __init__(self, tgt, ref):
        super().__init__(); self.tgt, self.ref, self.i = tgt, ref, 0; self.logs = []; self.kl = []; self.t1 = []; self.nll = []
    def forward(self, logits, *args, **kw):
        lg = logits.reshape(-1).float(); t = self.tgt[self.i]
        lp = torch.log_softmax(lg, -1)
        self.nll.append(-lp[t].item())
        if self.ref is None: self.logs.append(lg.half().cpu())
        else:
            r = self.ref[self.i].to(lg.device).float(); rp = torch.log_softmax(r, -1)
            self.kl.append((rp.exp() * (rp - lp)).sum().item()); self.t1.append(float(r.argmax() == lg.argmax()))
        self.i += 1
        return torch.tensor([[t]], device=logits.device)

def run_kl():
    n = a.depth + a.win; ids = long_text_ids(n, a.seed)
    if a.text == "copy":
        src0 = 1000; ids[a.depth:] = ids[src0:src0 + a.win]
    prefix, tgt = ids[:a.depth].view(1, -1), ids[a.depth:].tolist()
    ref = torch.stack(torch.load(a.ref)) if a.ref else None
    if ref is not None: assert ref.shape[0] == a.win
    s = TF(tgt, ref)
    # warm-up on a short job (kernel compile / graph capture) before the timed one
    w = Job(input_ids=ids[:600].view(1, -1), max_new_tokens=4, sampler=TF(ids[600:700].tolist(), None)); gen.enqueue(w)
    while gen.num_remaining_jobs(): list(gen.iterate())
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    job = Job(input_ids=prefix, max_new_tokens=a.win, sampler=s, stop_conditions=[]); gen.enqueue(job)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): print("JOB ERROR", r["error"])
    dt = time.time() - t0
    assert s.i == a.win, s.i
    res = {"kv": a.kv, "depth": a.depth, "text": a.text, "win": a.win, "nll": sum(s.nll) / len(s.nll), "peak_gib": torch.cuda.max_memory_allocated() / 2**30, "sec": dt}
    if ref is None:
        if a.save: torch.save(s.logs, a.save)
    else:
        kl = torch.tensor(s.kl); res.update(kl=kl.mean().item(), kl_med=kl.median().item(), kl_p99=kl.quantile(0.99).item(), top1=sum(s.t1) / len(s.t1))
    print("RESULT", json.dumps(res), flush=True)

WORDS = ["amber", "falcon", "granite", "harbor", "indigo", "juniper", "kestrel", "lantern", "marble", "nectar", "orchid", "pewter"]
def run_needle():
    base = long_text_ids(a.ctx, a.seed + 5)
    fill = a.ctx - 1500
    ok = 0; res = []
    import random; rng = random.Random(a.seed)
    # warm-up
    w = Job(input_ids=base[:600].view(1, -1), max_new_tokens=4, sampler=TF(base[600:700].tolist(), None)); gen.enqueue(w)
    while gen.num_remaining_jobs(): list(gen.iterate())
    for p in [float(x) for x in a.pos.split(",")]:
        code = f"{rng.choice(WORDS)}-{rng.randint(1000,9999)}-{rng.choice(WORDS)}"
        needle = tok.encode(f"\n\nNote for the record: the vault passphrase of the Zeta project is {code}. Remember it.\n\n", add_bos=False)[0].view(-1)
        q = tok.encode("\n\n---\nQuestion: what is the vault passphrase of the Zeta project? Answer with the passphrase only.", add_bos=False)[0].view(-1)
        at = int(fill * p)
        # chat-formatted question, thinking off, so the answer is the first thing generated
        q = torch.cat([q, tok.encode("<|assistant|><think></think>", add_bos=False, encode_special_tokens=True)[0].view(-1)])
        pre = tok.encode("<|user|>", add_bos=False, encode_special_tokens=True)[0].view(-1)
        ids = torch.cat([pre, base[:at], needle, base[at:fill], q]).view(1, -1)
        t0 = time.time()
        job = Job(input_ids=ids, max_new_tokens=a.gen, sampler=__import__("exllamav3.generator.sampler.presets", fromlist=["x"]).GreedySampler()); gen.enqueue(job)
        txt = ""
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("error"): print("JOB ERROR", r["error"])
                txt += r.get("text", "")
        hit = code in txt; ok += hit
        res.append({"pos": p, "len": ids.shape[1], "code": code, "out": txt.strip()[:60], "hit": hit, "sec": round(time.time() - t0, 1)})
        print("NEEDLE", json.dumps(res[-1]), flush=True)
    print("RESULT", json.dumps({"kv": a.kv, "ctx": a.ctx, "hits": f"{ok}/{len(res)}", "peak_gib": torch.cuda.max_memory_allocated() / 2**30}), flush=True)

run_kl() if a.mode == "kl" else run_needle()
