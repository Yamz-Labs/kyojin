"""Per-section GPU time of one draft pass (sync-bracketed, so the sum is an upper bound) and verify-vs-sequential logits check.
    python3 dspark_gpu_sections.py <pack> [--ctx 1024] [--reps 5]"""
import argparse, os, sys, time, collections
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]
ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--ctx", type=int, default=1024); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()
from ds41_serving import build_model
from exllamav3.dsv41.spec import PortSpec
from exllamav3.dsv41 import ops, dspark as D
m = build_model(a.pack, device="cuda:0", dtype="float16", layers=None, mtp=True, moe="auto", prefill_chunk=a.ctx)
spec = PortSpec(m, a.ctx + 64, max_depth=5)
g = torch.Generator().manual_seed(5)
prompt = torch.randint(0, m.cfg.vocab_size - 1, (a.ctx,), generator=g).tolist()
out = spec.target.prefill(prompt); pos = len(prompt); hid = out.hidden[-1:]
T = collections.defaultdict(float); N = collections.defaultdict(int); on = [False]

def wrap(obj, name, label):
    f = getattr(obj, name)
    def w(*x, **k):
        if not on[0]: return f(*x, **k)
        torch.cuda.synchronize(); t = time.perf_counter(); r = f(*x, **k); torch.cuda.synchronize()
        T[label] += time.perf_counter() - t; N[label] += 1; return r
    setattr(obj, name, w)
for n in ("rmsnorm", "hc_mixes", "hc_pre", "hc_post", "shared_key_attention", "rope_apply_"): wrap(ops, n, "ops." + n)
wrap(m, "_moe", "moe(total)"); wrap(m.router, "__call__", "  router") if False else None
H = m.mtp
wrap(H, "_attn", "attn(total)"); wrap(H, "_head", "head+markov+conf"); wrap(m, "head_logits", "  head_logits")
wrap(H, "main_proj", "main_proj") if False else None
for j in range(m.cfg.n_mtp_layers):
    w = m.lw[m.cfg.num_hidden_layers + j]
    for nm in ("wq_a", "wq_b", "wkv", "wo_b"):
        setattr(w, nm, (lambda f, lab: (lambda *x, **k: (torch.cuda.synchronize(), (t0 := time.perf_counter()), (r := f(*x, **k)), torch.cuda.synchronize(), T.__setitem__(lab, T[lab] + time.perf_counter() - t0) if on[0] else None, r)[-1]))(getattr(w, nm), "lin." + nm))
be0 = m.moe_backends[m.cfg.num_hidden_layers]
for l, be in m.moe_backends.items():
    if l >= m.cfg.num_hidden_layers: wrap(be, "run", "  experts(run)")
spec.drafter._next = pos
fn = lambda: spec.drafter.draft(1, hid, pos)
fn(); fn()
on[0] = True
t0 = time.perf_counter()
for _ in range(a.reps): fn()
torch.cuda.synchronize(); tot = (time.perf_counter() - t0) / a.reps * 1000
on[0] = False
print(f"draft pass (with sync brackets) {tot:.1f} ms")
for k, v in sorted(T.items(), key=lambda kv: -kv[1]): print(f"  {k:24s} {v / a.reps * 1000:7.2f} ms  calls/pass {N[k] // a.reps}")
# verify block vs sequential decode logits
import torch
spec.target.prefill(prompt)
toks = [int(x) for x in torch.randint(0, m.cfg.vocab_size - 1, (6,), generator=g)]
spec.target.verify(toks, pos); blk = spec.st
lb, hb = m.forward_block(torch.tensor([toks]), pos, spec.st) if False else (None, None)
spec.target.prefill(prompt); bo = spec.target.verify(toks, pos); lg_block = bo.logits.clone(); spec.target.truncate(pos + 1)
spec.target.prefill(prompt); seq = []
for i, t in enumerate(toks):
    lg, _ = m._forward(torch.tensor([[t]]), pos + i, spec.st, None, "last"); seq.append(lg[0].clone())
seq = torch.stack(seq)
d = (lg_block - seq).abs().amax(-1); rel = d / seq.abs().amax(-1)
print("block vs sequential logits max rel diff per row:", [f"{x:.2e}" for x in rel.tolist()], "argmax equal:", (lg_block.argmax(-1) == seq.argmax(-1)).tolist())

# teacher-forced replay: plain greedy tokens vs verify(6 rows)+truncate(1) at every step; row-0 logits compared to the plain step
steps = 40
spec.target.prefill(prompt[:256]); p0 = 256
plain_lg, plain_tok = [], []
lg, _ = m.forward(torch.tensor([prompt[:256]]), 0, spec.st) if False else (None, None)
spec.target.prefill(prompt[:256])
cur = int(spec.target.verify([prompt[256]], 256).logits[0].argmax()) if False else None
spec.reset(); ids = torch.tensor([prompt[:256]]); lg, _ = m.forward(ids, 0, spec.st)
cur = int(lg.argmax()); seq_tok = []
for i in range(steps):
    seq_tok.append(cur); plain_lg.append(lg[0].clone())
    lg, _ = m._forward(torch.tensor([[cur]]), 256 + i, spec.st, None, "last"); cur = int(lg.argmax())
# replay with blocks
spec.target.prefill(prompt[:256]); worst = []; flips = []
junk = [int(x) for x in torch.randint(0, m.cfg.vocab_size - 1, (5,), generator=g)]
for i in range(steps - 1):
    o = spec.target.verify([seq_tok[i]] + junk, 256 + i)
    spec.target.truncate(256 + i + 1)
    r = ((o.logits[0] - plain_lg[i + 1]).abs().max() / plain_lg[i + 1].abs().max()).item()
    worst.append(r); flips.append(int(o.logits[0].argmax()) != int(plain_lg[i + 1].argmax()))
print("teacher-forced block(6)+rollback vs plain, rel diff of row 0 per step:", " ".join(f"{x:.1e}" for x in worst))
print("steps where block argmax != plain argmax:", [i for i, f in enumerate(flips) if f])
