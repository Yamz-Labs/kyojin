"""CPU proof (Triton interpreter = the kernel body executed on CPU) of the 8-bit packed-cache read path of the QSA gather kernel
(_qsa_sparse_split_kernel + _paged_attn_decode_combine_kernel, loaders _qc_load_kt/_qc_load_v) at the real Qwen geometry
(24 q heads, 2 kv heads, head_dim 256, page 256, K_pad 2080, pool of 1024 pages = 262144 tokens).
Checks: every masked-in pointer inside the exact tensor ranges (bounds), output == torch attention on the dequantised K/V.
usage: TREE=<worktree> [TRITON_PAGED=<mutated triton_paged.py>] [MUT=name] python interp_qsa8.py"""
import os, sys, importlib.util
os.environ["TRITON_INTERPRET"] = "1"; os.environ["HIP_VISIBLE_DEVICES"] = "-1"
TREE = os.environ["TREE"]; sys.path.insert(0, TREE); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, triton
import triton.runtime.interpreter as _I
_orig_patch = _I._patch_lang_tensor
def _patch(tensor, scope):
    _orig_patch(tensor, scope)
    scope.set_attr(tensor, "__index__", lambda self: int(np.asarray(self.handle.data).reshape(-1)[0]))
    scope.set_attr(tensor, "__bool__", lambda self: bool(np.asarray(self.handle.data).reshape(-1)[0]))
_I._patch_lang_tensor = _patch
torch.set_num_threads(2)
RANGES, VIOL = [], []
def _check(ptrs, mask, what):
    p = np.asarray(ptrs.data).astype(np.uint64); m = np.asarray(mask.data).astype(bool) if mask is not None else np.ones(p.shape, bool)
    sel = p[m]
    if sel.size == 0: return
    ok = np.zeros(sel.shape, bool)
    for lo, hi in RANGES: ok |= (sel >= lo) & (sel < hi)
    if not ok.all():
        VIOL.append(f"{what}: {int((~ok).sum())} masked-in accesses outside every tensor"); raise RuntimeError("bounds violation")
Bd = _I.InterpreterBuilder
_ml, _ms, _l, _s = Bd.create_masked_load, Bd.create_masked_store, Bd.create_load, Bd.create_store
def _ml2(self, ptrs, mask, other, *a): _check(ptrs, mask, "load"); return _ml(self, ptrs, mask, other, *a)
def _ms2(self, ptrs, value, mask, *a): _check(ptrs, mask, "store"); return _ms(self, ptrs, value, mask, *a)
def _l2(self, ptr, *a): _check(ptr, None, "load1"); return _l(self, ptr, *a)
def _s2(self, ptr, val, *a): _check(ptr, None, "store1"); return _s(self, ptr, val, *a)
Bd.create_masked_load, Bd.create_masked_store, Bd.create_load, Bd.create_store = _ml2, _ms2, _l2, _s2
MUT = os.environ.get("MUT", "")
if os.environ.get("TRITON_PAGED"):   # mutated copy of triton_paged.py replaces the module before qsa_triton imports its loaders
    import exllamav3.modules.attention_fn as pkg
    spec = importlib.util.spec_from_file_location("exllamav3.modules.attention_fn.triton_paged", os.environ["TRITON_PAGED"])
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
import exllamav3.modules.attention_fn.qsa_triton as Q
from exllamav3.modules.attention_fn.triton_paged import _paged_attn_decode_combine_kernel, _get_h32
from ref8 import pack8, dequant8
def jit_of(x):
    while 'Interpreted' not in type(x).__name__ and hasattr(x, 'fn'): x = x.fn
    return x
split = jit_of(Q._qsa_sparse_split_kernel); comb = jit_of(_paged_attn_decode_combine_kernel)
H, KVH, HD, PS, KPAD, NP = 24, 2, 256, 256, 2080, 1024
GPT = KVH * HD // 32; BITS = 8
bad = []
gen = torch.Generator().manual_seed(1)
R = 2
filled = [0, 1, 7, 255, 511, 1022, 1023]          # pages that carry data; the others stay zero words / zero scales
qk = torch.zeros(NP, PS, GPT * BITS, dtype=torch.int32); qv = torch.zeros_like(qk)
sk = torch.zeros(NP, PS, GPT, dtype=torch.half); sv = torch.zeros_like(sk)
Kt = torch.zeros(NP, PS, KVH * HD); Vt = torch.zeros_like(Kt)
for p in filled:
    kx = (torch.randn(PS, KVH * HD, generator=gen) * torch.exp(torch.randn(1, KVH * HD, generator=gen) * 0.5)).half()
    vx = torch.randn(PS, KVH * HD, generator=gen).half()
    w, s = pack8(kx); qk[p], sk[p] = w, s; w, s = pack8(vx); qv[p], sv[p] = w, s
# block table: logical page i -> physical page; 8 logical pages cover positions 0..2047
phys = [1023, 0, 255, 7, 511, 1, 1022, 255 if False else 1023]
bt = torch.tensor([phys, phys], dtype=torch.int32)
npg = bt.shape[1]
# index lists: R rows, each K_pad wide, positions < 8*256, -1 padded tail (like the real selector)
idx = torch.full((R, KPAD), -1, dtype=torch.int32)
for r in range(R):
    n = 1500 + 300 * r
    idx[r, :n] = torch.randperm(npg * PS, generator=gen)[:n].int()
q = torch.randn(R, H, HD, generator=gen).half()
scale = HD ** -0.5
h32 = _get_h32(torch.device("cpu"))
BN = Q._QSA_BN; BH = 16
splits, split_len = Q.qsa_split_plan(R, KVH * 1, KPAD, BN, 16)
programs = R * KVH * 1
po = torch.empty(programs * splits * BH * HD); pml = torch.empty(programs * splits * BH * 2)
o = torch.full((R, H, HD), float("nan"), dtype=torch.half)
if os.environ.get("BADBT"): bt[0, 0] = NP          # block table points one page past the pool
for t in (q, qk, qv, sk, sv, bt, idx, po, pml, o, h32):
    RANGES.append((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()))
if os.environ.get("SHORT"):                          # cache allocated one page short of what the table references
    RANGES[1] = (RANGES[1][0], RANGES[1][1] - PS * GPT * BITS * 4)
try:
    kbits = int(os.environ.get("KBITS", BITS))
    split[(programs, splits)](q, qk, qv, bt, idx, po, pml, KPAD, npg, splits, split_len, sk, sv, h32,
        n_q_heads=H, n_kv_heads=KVH, page_size=PS, head_dim=HD, K_pad=KPAD, scale=scale, BLOCK_H=BH, BLOCK_N=BN, PAGED=1, QCK=kbits, QCV=BITS)
    comb[(programs,)](po, pml, o, h32, splits, pml, QCV=BITS, HAS_SINKS=False, q_len=1, n_q_heads=H, n_kv_heads=KVH,
        head_dim=HD, HD_PAD=HD, BLOCK_M=1, BLOCK_H=BH, BLOCK_ROWS=BH)
except Exception as e:
    bad.append(f"kernel aborted: {str(e)[:100]}")
if os.environ.get('BADBT'):
    print('[BADBT]', 'VIOLATIONS' if bad or VIOL else 'CLEAN', bad, VIOL[:2]); sys.exit(1 if bad or VIOL else 0)
# reference: gather dequantised K/V of the selected positions, plain softmax attention
ref = torch.zeros(R, H, HD)
for r in range(R):
    sel = idx[r][idx[r] >= 0].long()
    rows = bt[r][sel // PS].long() * PS + sel % PS
    kf = dequant8(qk.view(-1, GPT * BITS)[rows], sk.view(-1, GPT)[rows]).view(-1, KVH, HD)
    vf = dequant8(qv.view(-1, GPT * BITS)[rows], sv.view(-1, GPT)[rows]).view(-1, KVH, HD)
    for h in range(H):
        kv = h // (H // KVH)
        sc = (kf[:, kv] @ q[r, h].float()) * scale
        ref[r, h] = torch.softmax(sc, 0) @ vf[:, kv]
err = (o.float() - ref).abs().max().item(); mag = ref.abs().max().item()
print(f"max |kernel - reference| = {err:.3e}  (reference max {mag:.3f}, rel {err / mag:.2e})  splits={splits} programs={programs}")
if not (err / mag < 3e-3): bad.append(f"kernel != reference (rel {err / mag:.2e})")
if torch.isnan(o.float()).any(): bad.append("NaN in output")
bad += sorted(set(VIOL))[:6]
print(f"[{MUT or 'baseline'}]", "VIOLATIONS" if bad else "CLEAN"); [print("  ", b) for b in bad[:12]]
sys.exit(1 if bad else 0)
