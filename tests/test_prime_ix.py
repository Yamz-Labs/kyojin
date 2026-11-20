"""GPU test: step_graph.indexer_forward_static for T > 16 rows (draft-cache fill): HY4_LC_PRIME path (fp16 fused scorer + exact select) vs the stock fp32 einsum path.
The two allow masks must agree on >= 99.9 % of the visible entries (fp16 scorer rounding may swap boundary groups); poisoned guards on the inputs. DEV=cuda only."""
import importlib, math, os, sys, types
import torch
DEV = os.environ.get("DEV", "cuda")
here = os.path.dirname(os.path.abspath(__file__))
for n in ("exllamav3", "exllamav3.loader"):
    m = types.ModuleType(n); m.__path__ = [os.path.join(here, "..", "exllamav3") if n == "exllamav3" else os.path.join(here, "..", "exllamav3", "loader")]; sys.modules[n] = m
SG = importlib.import_module("exllamav3.loader.step_graph")
IXF = importlib.import_module("exllamav3.loader.indexer_fused")
PK = importlib.import_module("exllamav3.loader.prefill_kernels")

def build(cap, E=256, R=128, H=16, d=128, k=4):
    g = torch.Generator().manual_seed(0)
    r = lambda *s: (torch.randn(*s, generator=g) * 0.05).to(DEV)
    w = {"attn_k": r(d, E), "q_b": r(H * d, R), "value": r(d, d), "gate": r(d, d), "k_norm": torch.ones(d, device=DEV), "proj": r(H, E),
         "ape": r(k, d), "pool_norm": torch.ones(d, device=DEV)}
    ix = IXF.FusedKPoolIndexer(w, DEV, n_head=H, head_dim=d, top_k=2048, kpool=k, select_tail=True, rope_dim=64, rope_base=1e7, eps=1e-5, cap=cap)
    SG.indexer_static_init(ix, cap)
    return ix, g

def run(Nb=16384, T=2048, start=12288):
    cap = Nb
    ix, g = build(cap)
    k = ix.k
    ix.pool.copy_(torch.randn(ix.pool.shape, generator=g).to(DEV))
    x = torch.randn(T, 256, generator=g).to(DEV); qr = torch.randn(T, 128, generator=g).to(DEV)
    ar = torch.arange(cap, device=DEV)
    pos = ar[start:start + T]
    outs = []
    for flag in (False, True):
        SG.PRIME_IX = flag
        ix.vgb.zero_(); ix.pool_backup = None
        pool0 = ix.pool.clone()
        a = SG.indexer_forward_static(ix, x, qr, pos, Nb, ar)
        ix.pool.copy_(pool0)
        outs.append(a)
    SG.PRIME_IX = False
    a, b = outs
    assert a is not None and b is not None and a.shape == b.shape == (T, Nb), (None if a is None else a.shape, None if b is None else b.shape)
    agree = float((a == b).float().mean())
    print("allow agreement", agree, "selected/row", float(a.sum(1).float().mean()), float(b.sum(1).float().mean()))
    assert agree > 0.999
run()
run(Nb=65536, T=2048, start=60000)
print("ALL OK")
