"""qdec.hip vs the Triton decode kernels on random packed caches (any bit pattern is a valid packed state), plus a microbench.
python tests/test_qdec_hip.py [--hd 128 --kb 8 --vb 8 --nqh 32 --nkv 4 --qlen 1 --ctx 4096 --bsz 1 --bench]"""
import argparse, math, os, sys, time
os.environ.setdefault("EXL3_QDEC_HIP", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from exllamav3.modules.attention_fn import triton_paged as tp

ap = argparse.ArgumentParser()
ap.add_argument("--hd", type=int, default=128); ap.add_argument("--kb", type=int, default=8); ap.add_argument("--vb", type=int, default=8)
ap.add_argument("--nqh", type=int, default=32); ap.add_argument("--nkv", type=int, default=4); ap.add_argument("--qlen", type=int, default=1)
ap.add_argument("--ctx", type=int, default=4096); ap.add_argument("--bsz", type=int, default=1); ap.add_argument("--bench", action="store_true")
ap.add_argument("--noncausal", action="store_true"); ap.add_argument("--sinks", action="store_true"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
dev = torch.device("cuda")
torch.manual_seed(a.seed)
PAGE = 256
gpt = a.nkv * a.hd // 32
npg = -(-(a.ctx + 64) // PAGE)
tot = a.bsz * npg + 3
def words(b): return torch.randint(-2**31, 2**31 - 1, (tot, PAGE, gpt * b), dtype=torch.int32, device=dev)
kc, vc = words(a.kb), words(a.vb)
ks = (torch.rand(tot, PAGE, gpt, device=dev) * 2 + 0.05).to(torch.float16) * torch.exp(torch.randn(tot, PAGE, gpt, device=dev) * 0.3).to(torch.float16)
vs = (torch.rand(tot, PAGE, gpt, device=dev) * 2 + 0.05).to(torch.float16)
perm = torch.randperm(tot, device=dev)[: a.bsz * npg].to(torch.int32).view(a.bsz, npg).contiguous()
seqs = torch.tensor([a.ctx - 7 * i for i in range(a.bsz)], dtype=torch.int32, device=dev)   # tokens already in the cache before this step's q_len
q = (torch.randn(a.bsz, a.qlen, a.nqh, a.hd, device=dev) * 1.0).to(torch.float16)
sinks = torch.randn(a.nqh, device=dev) if a.sinks else None
causal = not a.noncausal
qc = (ks, vs, a.kb, a.vb)
# the cache holds the q_len new tokens already: counted as pre_appended_len
def run(hip):
    tp._QDEC_HIP = hip
    return tp.paged_attn_triton_decode(q, None, None, kc, vc, perm, seqs, causal=causal, softmax_scale=1 / math.sqrt(a.hd), sinks=sinks,
                                       qc=qc, pre_appended_len=a.qlen, n_kv_heads_override=a.nkv, max_kv_len=a.ctx)
ref = run(False); out = run(True)
torch.cuda.synchronize()
d = (ref.float() - out.float()).abs()
rel = d.max().item() / max(ref.float().abs().max().item(), 1e-9)
print(f"hd{a.hd} k{a.kb}v{a.vb} {a.nqh}/{a.nkv} qlen{a.qlen} ctx{a.ctx} bsz{a.bsz}: max|d| {d.max().item():.3e} mean|d| {d.mean().item():.3e} ref max {ref.float().abs().max().item():.3e} "
      f"rel {rel:.3e} nan {int(torch.isnan(out).any())} identical-sum-check {int(not torch.equal(ref, out))}")
if a.bench:
    nbytes = a.bsz * a.nkv * (a.ctx) * ((a.kb + a.vb) * a.hd // 8 + 2 * (a.hd // 32) * 2)
    for name, hip in (("triton", False), ("hip", True)):
        for _ in range(3): run(hip)
        torch.cuda.synchronize(); t0 = time.perf_counter(); n = 50
        for _ in range(n): run(hip)
        torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / n
        print(f"  {name}: {dt * 1e6:8.1f} us/call  {nbytes / dt / 1e9:6.1f} GB/s (cache bytes touched {nbytes / 1e6:.1f} MB)")
