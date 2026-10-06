"""(decode split + combine kernels, q_len 1 and 4) CPU proof (Triton interpreter = the kernel body executed on CPU) of the 8-bit packed-cache read path of the QSA gather kernel
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
import exllamav3.modules.attention_fn.triton_paged
TP = sys.modules["exllamav3.modules.attention_fn.triton_paged"]
from exllamav3.modules.attention_fn.triton_paged import _get_h32
from ref8 import pack8, dequant8
def jit_of(x):
    while 'Interpreted' not in type(x).__name__ and hasattr(x, 'fn'): x = x.fn
    return x
split = jit_of(TP._paged_attn_decode_split_kernel); comb = jit_of(TP._paged_attn_decode_combine_kernel)
H, KVH, HD, PS, NP = 24, 2, 256, 256, 1024
GPT = KVH * HD // 32; BITS = 8
gen = torch.Generator().manual_seed(2)
filled = [0, 1, 7, 255, 511, 1022, 1023]
qk = torch.zeros(NP, PS, GPT * BITS, dtype=torch.int32); qv = torch.zeros_like(qk)
sk = torch.zeros(NP, PS, GPT, dtype=torch.half); sv = torch.zeros_like(sk)
for p in filled:
    kx = (torch.randn(PS, KVH * HD, generator=gen) * torch.exp(torch.randn(1, KVH * HD, generator=gen) * 0.5)).half()
    vx = torch.randn(PS, KVH * HD, generator=gen).half()
    w, s = pack8(kx); qk[p], sk[p] = w, s; w, s = pack8(vx); qv[p], sv[p] = w, s
phys = [1023, 0, 255, 7, 511, 1, 1022, 1023]
bt = torch.tensor([phys], dtype=torch.int32); npg = bt.shape[1]
if os.environ.get("BADBT"): bt[0, 3] = NP
h32 = _get_h32(torch.device("cpu"))
scale = HD ** -0.5; bad = []
BN = max(16, 8192 // HD)
for q_len, seqlen in ((1, 1500), (1, 7), (4, 1900), (4, 2044), (1, 255), (1, 256)):
    kv_append = q_len                                           # new tokens already in the cache (pre_appended_len)
    total = seqlen + kv_append
    block_m = triton.next_power_of_2(q_len); block_h = max(16 // block_m, 1); block_rows = block_m * block_h
    h_blocks = triton.cdiv(H // KVH, block_h); programs = KVH * h_blocks
    num_splits = 4; max_k = npg * PS + kv_append; split_len = triton.cdiv(triton.cdiv(max_k, num_splits), BN) * BN
    q = torch.randn(1, q_len, H, HD, generator=gen).half()
    po = torch.empty(programs * num_splits * block_rows * HD); pml = torch.empty(programs * num_splits * block_rows * 2)
    o = torch.full((1, q_len, H, HD), float("nan"), dtype=torch.half)
    cs = torch.tensor([seqlen], dtype=torch.int32); sinks = q
    RANGES.clear()
    for t in (q, qk, qv, sk, sv, bt, cs, po, pml, o, h32):
        RANGES.append((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()))
    if os.environ.get("SHORT"): RANGES[1] = (RANGES[1][0], RANGES[1][1] - PS * GPT * BITS * 4)
    try:
        split[(programs, num_splits)](q, qk, qv, bt, cs, o, po, pml, sk, sv, h32, split_len, npg, num_splits, sinks,
            int(os.environ.get("KBITS", BITS)), BITS, q_len, kv_append, H, KVH, PS, HD, HD, scale, True, -1, -1, 0.0, False, False, block_m, block_h, block_rows, BN)
        comb[(programs,)](po, pml, o, h32, num_splits, sinks, BITS, False, q_len, H, KVH, HD, HD, block_m, block_h, block_rows)
    except Exception as e:
        bad.append(f"q_len {q_len} len {seqlen}: kernel aborted: {str(e)[:90]}"); continue
    if os.environ.get('BADBT'): continue
    rowsel = torch.arange(total); rows = bt[0][rowsel // PS].long() * PS + rowsel % PS
    kf = dequant8(qk.view(-1, GPT * BITS)[rows], sk.view(-1, GPT)[rows]).view(-1, KVH, HD)
    vf = dequant8(qv.view(-1, GPT * BITS)[rows], sv.view(-1, GPT)[rows]).view(-1, KVH, HD)
    err = 0.0; mag = 1e-9
    for i in range(q_len):
        qa = total - q_len + i
        for h in range(H):
            kv = h // (H // KVH)
            sc = (kf[:qa + 1, kv] @ q[0, i, h].float()) * scale
            ref = torch.softmax(sc, 0) @ vf[:qa + 1, kv]
            err = max(err, (o[0, i, h].float() - ref).abs().max().item()); mag = max(mag, ref.abs().max().item())
    print(f"q_len {q_len} len {seqlen}: max err {err:.2e} (rel {err / mag:.2e})", flush=True)
    if torch.isnan(o.float()).any() or err / mag > 3e-3: bad.append(f"q_len {q_len} len {seqlen}: mismatch rel {err / mag:.2e}")
bad += sorted(set(VIOL))[:6]
print(f"[{MUT or 'baseline'}]", "VIOLATIONS" if bad else "CLEAN"); [print("  ", b) for b in bad[:12]]
sys.exit(1 if bad else 0)
