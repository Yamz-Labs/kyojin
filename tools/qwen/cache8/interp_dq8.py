"""CPU proof (Triton interpreter) of dequant8_pages: every load/store inside its tensor, output bit-identical to the torch reference, with hostile block tables (negative, past the end).
usage: TREE=<worktree> [MUT=name] python interp_dq8.py"""
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
src = open(os.path.join(TREE, "exllamav3/modules/attention_fn/qdeq_triton.py")).read()
MUT = os.environ.get("MUT", "")
muts = {
  "src_stride": ("src = (page * 256 + tok) * 512 + offs", "src = (page * 256 + tok) * 511 + offs"),
  "scale_idx": ("offs // 32", "offs // 16"),
  "scale_stride": ("sidx = (page * 256 + tok) * 16 +", "sidx = (page * 256 + tok) * 17 +"),
  "dst_page": ("dst = (j.to(tl.int64) * 256 + tok)", "dst = ((j.to(tl.int64) + 1) * 256 + tok)"),
  "no_clamp": ("tl.minimum(tl.maximum(tl.load(btable + j), 0), P - 1)", "tl.load(btable + j)"),
  "clamp_high": ("tl.minimum(tl.maximum(tl.load(btable + j), 0), P - 1)", "tl.minimum(tl.maximum(tl.load(btable + j), 0), P)"),
  "midpoint": ("(ck - 127.5)", "(ck - 127.0)"),
  "v_from_k": ("cv = tl.load(qv + src)", "cv = tl.load(qk + src)"),
  "tok_overrun": ("tok = (tb * TBc + tl.arange(0, TBc))[:, None].to(tl.int64)", "tok = (tb * TBc + 1 + tl.arange(0, TBc))[:, None].to(tl.int64)"),
}
if MUT:
    a, b = muts[MUT]; assert a in src; src = src.replace(a, b)
src = src.replace("from .qwrite_triton import _h32_stage", "from qwrite_triton import _h32_stage")
sys.path.insert(0, os.path.join(TREE, "exllamav3/modules/attention_fn"))
open(os.path.join(HERE, "_dq_mut.py"), "w").write(src)
spec = importlib.util.spec_from_file_location("dqm", os.path.join(HERE, "_dq_mut.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
def ref(q, s, pages):
    P = q.shape[0]; out = []
    for pg in pages.tolist():
        pg = min(max(pg, 0), P - 1)
        c = q[pg].view(torch.uint8).float(); sc = s[pg].float().repeat_interleave(32, dim=-1)   # (256, 512)
        out.append(((c.view(256, 512) - 127.5) * (sc / 128.0)).half().view(256, 2, 256))
    return torch.cat(out, 0)
gen = torch.Generator().manual_seed(3)
P = 6
qk = torch.randint(-2**31, 2**31 - 1, (P, 256, 128), generator=gen, dtype=torch.int32); qv = torch.randint(-2**31, 2**31 - 1, (P, 256, 128), generator=gen, dtype=torch.int32)
sk = (torch.rand(P, 256, 16, generator=gen) * 4 + 1e-3).half(); sv = (torch.rand(P, 256, 16, generator=gen) * 4 + 1e-3).half()
res = []
try:
    for pages in ([3, 0, 5], [-3, 9, 2, 4], [1]):
        pg = torch.tensor(pages, dtype=torch.int32); n = len(pages)
        ok_ = torch.full((n * 256 + 7, 2, 256), 7.0, dtype=torch.half); ov = ok_.clone()   # +7 spare rows must stay untouched
        RANGES.clear()
        for t in (qk, qv, sk, sv, ok_, ov, pg): RANGES.append((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()))
        mod._deq8_kernel[(n, 256 // mod.TB)](qk.view(torch.uint8), sk, qv.view(torch.uint8), sv, ok_, ov, pg, P, TBc=mod.TB, num_warps=4)
        rk, rv = ref(qk, sk, pg), ref(qv, sv, pg)
        res.append(torch.equal(ok_[: n * 256], rk) and torch.equal(ov[: n * 256], rv) and bool((ok_[n * 256:] == 7).all()) and bool((ov[n * 256:] == 7).all()))
    print("RESULT", "OK identical" if all(res) else f"MISMATCH {res}", "VIOL", VIOL)
except Exception as e:
    print("RESULT", "VIOLATION" if VIOL else f"ERROR {e!r}", VIOL)
