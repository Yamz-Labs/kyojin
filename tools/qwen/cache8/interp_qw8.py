"""CPU proof (Triton interpreter) of the native 8-bit paged cache writer: every masked-in pointer inside its tensor,
output words/scales bit-identical to the torch fallback. usage: TREE=<worktree> [MUT=name] python interp_qw8.py"""
import os, sys, re
os.environ["TRITON_INTERPRET"] = "1"; os.environ["HIP_VISIBLE_DEVICES"] = "-1"
TREE = os.environ["TREE"]; sys.path.insert(0, TREE)
import numpy as np, torch
import triton.runtime.interpreter as _I
torch.set_num_threads(2)
RANGES, VIOL = [], []
def _check(ptrs, mask, what):
    p = np.asarray(ptrs.data).astype(np.uint64); m = np.asarray(mask.data).astype(bool) if mask is not None else np.ones(p.shape, bool)
    sel = p[m]
    if sel.size == 0: return
    ok = np.zeros(sel.shape, bool)
    for lo, hi in RANGES: ok |= (sel >= lo) & (sel < hi)
    if not ok.all():
        VIOL.append(f"{what}: {int((~ok).sum())} accesses outside every tensor"); raise RuntimeError("bounds violation")
Bd = _I.InterpreterBuilder
_ml, _ms, _l, _s = Bd.create_masked_load, Bd.create_masked_store, Bd.create_load, Bd.create_store
Bd.create_masked_load = lambda self, p, m, o, *a: (_check(p, m, "load"), _ml(self, p, m, o, *a))[1]
Bd.create_masked_store = lambda self, p, v, m, *a: (_check(p, m, "store"), _ms(self, p, v, m, *a))[1]
Bd.create_load = lambda self, p, *a: (_check(p, None, "load1"), _l(self, p, *a))[1]
Bd.create_store = lambda self, p, v, *a: (_check(p, None, "store1"), _s(self, p, v, *a))[1]
import importlib.util
src = open(os.path.join(TREE, "exllamav3/modules/attention_fn/qwrite_triton.py")).read()
MUT = os.environ.get("MUT", "")
muts = {
  "src_stride": ("src * (G * 32)", "src * (G * 32 + 32)"),
  "dst_stride": ("dst * (G * 8) + og", "dst * (G * 8 + 8) + og"),
  "scale_stride": ("sc + dst * G +", "sc + dst * (G + 1) +"),
  "page": ("tok // 256)", "tok // 256 + 1)"),
  "midpoint": ("v * 128.0 + 128.0", "v * 128.0 + 127.0"),
  "shift": ("tl.arange(0, 4) * 8", "tl.arange(0, 4) * 4"),
}
if MUT:
    a, b = muts[MUT]; assert a in src; src = src.replace(a, b)
open(os.path.join(os.path.dirname(__file__), "_qw_mut.py"), "w").write(src)
spec = importlib.util.spec_from_file_location("qwm", os.path.join(os.path.dirname(__file__), "_qw_mut.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from exllamav3 import ext_fallbacks as fb
from exllamav3.ext_fallbacks import _quant_cache_paged_torch
def run(bsz, seq_len, seqlens, contig, npages=8, dim=512):
    G = dim // 32
    bt = torch.arange(npages, dtype=torch.int32).flip(0)[:bsz * 3].reshape(bsz, 3).contiguous() if bsz * 3 <= npages else None
    gen = torch.Generator().manual_seed(5)
    n = bsz * seq_len if contig else npages * 256
    k = (torch.randn(n, dim, generator=gen) * torch.exp(torch.randn(1, dim, generator=gen))).half().contiguous()
    v = torch.randn(n, dim, generator=gen).half().contiguous()
    shp = (bsz, seq_len, dim) if contig else (npages, 256, dim)
    k = k.view(shp); v = v.view(shp)
    cs = torch.tensor(seqlens, dtype=torch.int32)
    outs = []
    for impl in (0, 1):
        ko = torch.zeros(npages, 256, G * 8, dtype=torch.int32); vo = torch.zeros_like(ko)
        ks = torch.zeros(npages, 256, G, dtype=torch.half); vs = torch.zeros_like(ks)
        if impl == 0:
            _quant_cache_paged_torch(k, ko, ks, v, vo, vs, cs, bt, 256, seq_len, 0.0, contig)
        else:
            RANGES.clear()
            for t in (k, v, ko, vo, ks, vs, cs, bt): RANGES.append((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()))
            mod._quant_cache_paged8_kernel[(bsz * seq_len,)](k, v, ko, vo, ks, vs, cs, bt, seq_len, bt.shape[1], in_contig=contig, G=G)
        outs.append((ko, vo, ks, vs))
    return all(torch.equal(a, b) for a, b in zip(*outs))
res = []
try:
    res.append(run(1, 1, [255], True)); res.append(run(1, 4, [254], True)); res.append(run(2, 3, [10, 700], True))
    res.append(run(1, 5, [0], False)); res.append(run(2, 1, [255, 256], False))
    print("RESULT", "OK identical" if all(res) else f"MISMATCH {res}", "VIOL", VIOL)
except Exception as e:
    print("RESULT", "VIOLATION" if VIOL else f"ERROR {e!r}", VIOL)
