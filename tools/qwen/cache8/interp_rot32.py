"""CPU proof (Triton interpreter) of rot32: all accesses inside their tensors (tail rows masked), output bit-identical to the fp32 butterfly reference.
usage: TREE=<worktree> [MUT=name] python interp_rot32.py"""
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
  "stride": ("offs = row * 32 + tl.arange(0, 32)[None, :]", "offs = row * 33 + tl.arange(0, 32)[None, :]"),
  "no_mask_load": ("v = tl.load(x + offs, mask=m, other=0.0)", "v = tl.load(x + offs)"),
  "no_mask_store": ("tl.store(y + offs, (v * 0.17677669529663688).to(tl.float16), mask=m)", "tl.store(y + offs, (v * 0.17677669529663688).to(tl.float16))"),
  "mask_off_by_one": ("m = row < n", "m = row <= n"),
  "scale": ("v * 0.17677669529663688", "v * 0.1767767"),
  "stage_missing": ("    v = _h32_stage(v, BG, 16)\n", ""),
}
if MUT:
    a, b = muts[MUT]; assert a in src; src = src.replace(a, b)
src = src.replace("from .qwrite_triton import _h32_stage", "from qwrite_triton import _h32_stage")
sys.path.insert(0, d)
open(os.path.join(HERE, "_rot_mut.py"), "w").write(src)
spec = importlib.util.spec_from_file_location("rotm", os.path.join(HERE, "_rot_mut.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from exllamav3.ext_fallbacks import _hadamard32
res = []
try:
    for n in (1, 63, 64, 65, 200):
        gen = torch.Generator().manual_seed(n)
        x = (torch.randn(n, 32, generator=gen) * torch.exp(torch.randn(n, 1, generator=gen))).half().contiguous()
        y = torch.full((n + 70, 32), 9.0, dtype=torch.half)   # spare rows must stay untouched
        RANGES.clear()
        for t in (x, y): RANGES.append((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()))
        mod._rot32_kernel[((n + 63) // 64,)](x, y, n, BG=64)
        ref = (_hadamard32(x.float()) * torch.scalar_tensor(0.17677669529663688109, dtype=torch.float32)).half()
        res.append(torch.equal(y[:n], ref) and bool((y[n:] == 9).all()))
    print("RESULT", "OK identical" if all(res) else f"MISMATCH {res}", "VIOL", VIOL)
except Exception as e:
    print("RESULT", "VIOLATION" if VIOL else f"ERROR {e!r}", VIOL)
