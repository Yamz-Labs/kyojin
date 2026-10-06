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
import contextlib
import exllamav3.modules.attention_fn.triton_paged
TP = sys.modules["exllamav3.modules.attention_fn.triton_paged"]
from exllamav3.modules.attention_fn.triton_paged import _get_h32
from ref8 import pack8, dequant8
TP._check_tensor = lambda *a, **k: None           # CPU harness: the wrappers insist on CUDA tensors
torch.cuda.device = lambda d: contextlib.nullcontext()
class _P: multi_processor_count = 16; major = 11; minor = 5
torch.cuda.get_device_properties = lambda d: _P()
torch.cuda.get_device_capability = lambda d: (11, 5)
H, KVH, HD, PS, NP = 24, 2, 256, 256, 1024
GPT = KVH * HD // 32; BITS = 8
gen = torch.Generator().manual_seed(3)
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
bad = []
MODE = os.environ.get("MODE", "both")
for fn, q_len, seqlen in (("prefill", 64, 0), ("prefill", 100, 1500), ("prefill", 64, 1984), ("prefill", 33, 300), ("decode", 1, 1500), ("decode", 4, 1900), ("decode", 1, 2047)):
    total = seqlen + q_len
    q = torch.randn(1, q_len, H, HD, generator=gen).half()
    out = torch.full((1, q_len, H, HD), float("nan"), dtype=torch.half)
    cs = torch.tensor([seqlen], dtype=torch.int32)
    RANGES.clear()
    for t in (q, qk, qv, sk, sv, bt, cs, out, h32):
        RANGES.append((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()))
    # partial buffers allocated inside the wrapper: widen the allowed range to the whole address space of fresh tensors via a hook on torch.empty
    _empty = torch.empty
    def _emp(*a, **k):
        t = _empty(*a, **k); RANGES.append((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())); return t
    torch.empty = _emp
    if os.environ.get("SHORT"): RANGES[1] = (RANGES[1][0], RANGES[1][1] - PS * GPT * BITS * 4)
    try:
        f = TP.paged_attn_triton_prefill if fn == "prefill" else TP.paged_attn_triton_decode
        f(q, None, None, qk, qv, bt, cs, causal=True, softmax_scale=HD ** -0.5, out=out, qc=(sk, sv, BITS, BITS), pre_appended_len=q_len, n_kv_heads_override=KVH)
    except Exception as e:
        import traceback; traceback.print_exc(); bad.append(f"{fn} q_len {q_len} len {seqlen}: aborted: {str(e)[:110]}"); torch.empty = _empty; continue
    torch.empty = _empty
    if os.environ.get('BADBT'): continue
    rowsel = torch.arange(total); rows = bt[0][rowsel // PS].long() * PS + rowsel % PS
    kf = dequant8(qk.view(-1, GPT * BITS)[rows], sk.view(-1, GPT)[rows]).view(-1, KVH, HD)
    vf = dequant8(qv.view(-1, GPT * BITS)[rows], sv.view(-1, GPT)[rows]).view(-1, KVH, HD)
    err = 0.0; mag = 1e-9
    for i in range(q_len):
        qa = total - q_len + i
        for h in range(H):
            kv = h // (H // KVH)
            sc = (kf[:qa + 1, kv] @ q[0, i, h].float()) * HD ** -0.5
            ref = torch.softmax(sc, 0) @ vf[:qa + 1, kv]
            err = max(err, (out[0, i, h].float() - ref).abs().max().item()); mag = max(mag, ref.abs().max().item())
    print(f"{fn} q_len {q_len} len {seqlen}: max err {err:.2e} (rel {err / mag:.2e})", flush=True)
    if torch.isnan(out.float()).any() or err / mag > 3e-3: bad.append(f"{fn} q_len {q_len} len {seqlen}: mismatch rel {err / mag:.2e}")
bad += sorted(set(VIOL))[:6]
print(f"[{MUT or 'baseline'}]", "VIOLATIONS" if bad else "CLEAN"); [print("  ", b) for b in bad[:12]]
sys.exit(1 if bad else 0)
