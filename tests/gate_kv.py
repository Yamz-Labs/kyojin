"""gate_kv.py PACK OUT.pt [--kv 16|K,V] [--depth 4096] [--n 128] [--force REF.pt]
Greedy decode of 3 prompts (real text of ~depth tokens + a question) x n tokens through the production cached decode path, recording the
log-prob distribution at every generated step (fp16). With --force REF.pt the same cache path is then re-run teacher-forced on REF's
generated tokens, so two builds (qdec.hip on/off) can be compared step by step: ids, top-1 agreement, KL (gate_cmp.py).
Env EXL3_QDEC_HIP selects the kernel (the pack's attention reads it at import)."""
import argparse, os, sys, time
ROOT = os.environ.get("ENG") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import Sampler

ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("out"); ap.add_argument("--kv", default="16")
ap.add_argument("--depth", type=int, default=4096); ap.add_argument("--n", type=int, default=128); ap.add_argument("--force")
ap.add_argument("--np", type=int, default=3)
a = ap.parse_args()


class RecSampler(Sampler):
    """argmax (or the forced token), recording the log-softmax of every step"""
    def __init__(self, forced=None):
        super().__init__(); self.forced = forced; self.i = 0; self.lps = []
    def forward(self, logits, sequence_ids=None, rand_u32=None, tokenizer=None, logit_mask=None, return_state=False):
        lg = logits.float()
        if tokenizer is not None and tokenizer.actual_vocab_size < lg.shape[-1]:
            lg[..., tokenizer.actual_vocab_size:] = -float("inf")
        self.lps.append(torch.log_softmax(lg.reshape(-1, lg.shape[-1])[-1], -1).half().cpu())
        t = self.forced[self.i] if self.forced is not None else int(lg.reshape(-1, lg.shape[-1])[-1].argmax())
        self.i += 1
        return torch.full(logits.shape[:-1], t, dtype=torch.long, device=logits.device)

cfg = Config.from_directory(a.pack); model = Model.from_config(cfg); tok = Tokenizer.from_config(cfg)
ctx = ((a.depth + a.n + 512 + 255) // 256) * 256
if a.kv == "16": cache = Cache(model, max_num_tokens=ctx)
else:
    kb, vb = (int(x) for x in a.kv.split(",")); cache = Cache(model, max_num_tokens=ctx, layer_type=CacheLayer_quant, k_bits=kb, v_bits=vb)
t0 = time.time(); model.load(progressbar=False); print(f"loaded {time.time() - t0:.0f}s", flush=True)
gen = Generator(model=model, cache=cache, tokenizer=tok)
D = os.path.join(ROOT, "exllamav3/conversion/standard_cal_data")
srcs = ["wiki.utf8", "c4.utf8", "code.utf8"]
ref = torch.load(a.force) if a.force else None

def prompt_ids(i):
    if ref: return torch.tensor(ref["res"][i]["prompt"]).view(1, -1)
    s = open(f"{D}/{srcs[i % 3]}", encoding="utf8").read()
    off = 3000 * (i + 1)
    ids = tok.encode(s[off:off + max(a.depth * 8, 20000)], add_bos=False).flatten()[:max(a.depth - 40, 8)]
    q = tok.encode("\n\n---\nQuestion: summarize the text above in two sentences, in French.\nAnswer:", add_bos=False).flatten()
    return torch.cat([ids, q]).view(1, -1)

def run(ids, sampler):
    job = Job(input_ids=ids, max_new_tokens=a.n, sampler=sampler, stop_conditions=[])
    gen.enqueue(job); out = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("error"): print("JOB ERROR", r["error"])
            if r.get("token_ids") is not None and r["token_ids"].numel(): out += r["token_ids"].flatten().tolist()
    return out

res = []
for i in range(a.np):
    ids = prompt_ids(i)
    s = RecSampler(); t0 = time.time(); out = run(ids, s)
    r = {"prompt": ids.flatten().tolist(), "gen": out, "lp": torch.stack(s.lps)}
    if ref:
        s2 = RecSampler(forced=ref["res"][i]["gen"]); run(ids, s2)
        r["lp_forced"] = torch.stack(s2.lps); r["forced_gen"] = ref["res"][i]["gen"]
    res.append(r)
    print(f"prompt {i}: plen {ids.shape[-1]} gen {len(out)} steps-rec {len(s.lps)} {time.time() - t0:.0f}s", flush=True)
torch.save({"res": res, "kv": a.kv, "depth": a.depth, "hip": os.environ.get("EXL3_QDEC_HIP", "0")}, a.out)
print("saved", a.out)
