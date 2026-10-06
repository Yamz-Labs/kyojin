"""CPU proof (Triton interpreter) of dequant8_full: loads/stores inside their tensors (hostile block tables), output bit-identical to ext_fallbacks.dequant_cache_cont.
usage: TREE=<worktree> [MUT=name] python interp_dqf8.py"""
import os, sys
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
HERE = os.path.dirname(os.path.abspath(__file__))
d = os.path.join(TREE, "exllamav3/modules/attention_fn")
src = open(os.path.join(d, "qdeq_triton.py")).read()
MUT = os.environ.get("MUT", "")
muts = {
  "src_stride": ("src = ((page * 256 + tok) * G + grp) * 32 + lane", "src = ((page * 256 + tok) * G + grp) * 33 + lane"),
  "scale_idx": ("sidx = (page * 256 + tok) * G + grp", "sidx = (page * 256 + tok) * G + grp + 1"),
  "dst_page": ("dst = ((j.to(tl.int64) * 256 + tok) * G + grp) * 32 + lane", "dst = (((j.to(tl.int64) + 1) * 256 + tok) * G + grp) * 32 + lane"),
  "no_clamp": ("tl.minimum(tl.maximum(tl.load(btable + j), 0), P - 1).to(tl.int64)\n    tok = (tb * TBc + tl.arange(0, TBc))[:, None, None]", "tl.load(btable + j).to(tl.int64)\n    tok = (tb * TBc + tl.arange(0, TBc))[:, None, None]"),
  "midpoint": ("(c - 127.5)", "(c - 127.0)"),
  "stage_missing": ("        x = _h32_stage(x, TBc * G, 16)\n", ""),
  "v_from_k": ("c = tl.load(qv + src).to(tl.float32)", "c = tl.load(qk + src).to(tl.float32)"),
  "tok_overrun": ("tok = (tb * TBc + tl.arange(0, TBc))[:, None, None].to(tl.int64)", "tok = (tb * TBc + 1 + tl.arange(0, TBc))[:, None, None].to(tl.int64)"),
}
if MUT:
    a, b = muts[MUT]
    if a not in src: raise SystemExit(f"mutation {MUT} does not apply")
    src = src.replace(a, b)
src = src.replace("from .qwrite_triton import _h32_stage", "from qwrite_triton import _h32_stage")
sys.path.insert(0, d)
open(os.path.join(HERE, "_dqf_mut.py"), "w").write(src)
spec = importlib.util.spec_from_file_location("dqfm", os.path.join(HERE, "_dqf_mut.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from exllamav3.ext_fallbacks import dequant_cache_cont
G = 16; P = 5
gen = torch.Generator().manual_seed(7)
qk = torch.randint(-2**31, 2**31 - 1, (P, 256, G * 8), generator=gen, dtype=torch.int32); qv = torch.randint(-2**31, 2**31 - 1, (P, 256, G * 8), generator=gen, dtype=torch.int32)
sk = (torch.rand(P, 256, G, generator=gen) * 4 + 1e-3).half(); sv = (torch.rand(P, 256, G, generator=gen) * 4 + 1e-3).half()
def ref(q, s, pages):
    out = []
    for pg in pages.tolist():
        pg = min(max(pg, 0), P - 1); o = torch.empty(256, G * 32, dtype=torch.half)
        dequant_cache_cont(q[pg].contiguous(), s[pg].contiguous(), o, 0.0); out.append(o)
    return torch.cat(out, 0)
res = []
try:
    for pages in ([3, 0], [-2, 7, 1], [4]):
        pg = torch.tensor(pages, dtype=torch.int32); n = len(pages)
        ok_ = torch.full((n * 256 + 5, G * 32), 7.0, dtype=torch.half); ov = ok_.clone()
        RANGES.clear()
        for t in (qk, qv, sk, sv, ok_, ov, pg): RANGES.append((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()))
        mod._deqfull8_kernel[(n, 64)](qk.view(torch.uint8), sk, qv.view(torch.uint8), sv, ok_, ov, pg, P, TBc=4, G=G)
        rk, rv = ref(qk, sk, pg), ref(qv, sv, pg)
        res.append(torch.equal(ok_[: n * 256], rk) and torch.equal(ov[: n * 256], rv) and bool((ok_[n * 256:] == 7).all()) and bool((ov[n * 256:] == 7).all()))
    print("RESULT", "OK identical" if all(res) else f"MISMATCH {res}", "VIOL", VIOL)
except Exception as e:
    print("RESULT", "VIOLATION" if VIOL else f"ERROR {e!r}", VIOL)
