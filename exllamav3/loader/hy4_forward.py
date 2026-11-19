"""Hy4-flash (hy_v4_flash) WHOLE-MODEL forward, slow but correct: torch only, no extension kernel, runs on CPU or any torch device.

Composition of the registry blocks' reference maths (embedding, engram_hy4 + f8_rows, hyper_connection, mla + mla_window_sinks +
mla_output_gate, dsa_kpool_indexer, dense / latent_moe + router_score_bias, hyper_head, output_norm + output); the trunk is
block_count - nextn_predict_layers layers (blk 42, the MTP block, is never read). Only active when the GGUF plan selects those blocks
(`is_hy4_plan`); nothing else imports this file.

  weights   non-expert tensors are dequantised once per layer (Q8_0 / F32 / BF16) into `wdtype` on `device` (kept when cache_weights);
            routed experts stay raw in the GGUF (page cache): per call only the experts the router picked are read by offset,
            dequantised (Q4_K / Q3_K torch paths) and dropped. The two-limit SwiGLU clamp (gate min 10, up +-10) is the torch path.
  streams   [T, hc, n_embd] fp32 between layers; HC identity comb (no cross-stream mix); engram before hc_pre on ple.layers.
  cache     per layer the MLA K rows [N, kv_lora + rope] (V = first kv_lora columns), full length (window layers mask, sinks always);
            per full layer one KPoolIndexer (CPU) whose selected rows become the attention mask (all-causal selections skip the mask).
  decode    Hy4Forward.step(ids) appends to the caches; generate() = one prefill + greedy steps.
Activations are NOT Q8-quantised (llama.cpp CPU quantises them): small numeric differences vs the oracle are expected."""
from __future__ import annotations
import contextlib
import importlib.util
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from .engram_rows import F8Rows, Hy4Engram, load_dense, _dequant_q8_0, rms
from .mla_hy4 import Hy4MLA
from . import ref_fused as _rf
from . import step_nosync as _ns
from . import kv_fused as _KVF
from . import mla_bgemv as _BGM
from . import head_bf16 as _HB
from . import attn_eager as _AE
from . import router_gemv as _RG
from . import mla_hy4 as _mla_mod
from . import dense_req as _DR

_HCF = None
if os.environ.get("HY4_HC_FUSED", "0") == "1":      # fused hyper-connection sites (loader/hc_fused.py)
    from . import hc_fused as _HCF

_HERE = os.path.dirname(os.path.abspath(__file__))
_NEED = {"mla", "mla_window_sinks", "hyper_connection", "latent_moe", "engram_hy4"}


def is_hy4_plan(plan) -> bool:
    """registry predicate: the plan selected the Hy4 block set (a plain llama / qwen plan never does)"""
    names = set()
    for i in range(len(plan.layers)):
        names |= set(plan.blocks_at(i))
    return _NEED <= names and "hyper_head" in set(plan.glob.get("head_mod", []))


def _kpool_mod():
    sp = importlib.util.spec_from_file_location("exl_kpool_indexer", os.path.join(os.path.dirname(_HERE), "modules", "kpool_indexer.py"))
    m = importlib.util.module_from_spec(sp)
    sys.modules["exl_kpool_indexer"] = m
    sp.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ dequant (any device)
def _raw(raw, dev):
    return raw.to(dev) if torch.is_tensor(raw) else torch.from_numpy(np.ascontiguousarray(raw)).to(dev)


def dq_q4k(raw, N, K, dev="cpu"):
    """Q4_K (144 B / 256: fp16 d, fp16 dmin, 12 B 6-bit scales / mins, 128 B nibbles) -> fp32 [N, K] (ggml dequantize_row_q4_K)"""
    nb = K // 256
    r = _raw(raw, dev).reshape(N * nb, 144)
    d = r[:, 0:2].contiguous().view(torch.float16).float().reshape(-1, 1)
    dm = r[:, 2:4].contiguous().view(torch.float16).float().reshape(-1, 1)
    s = r[:, 4:16].int()
    sc = torch.cat([s[:, 0:4] & 63, (s[:, 8:12] & 15) | ((s[:, 0:4] >> 6) << 4)], 1).float()
    mn = torch.cat([s[:, 4:8] & 63, (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)], 1).float()
    qs = r[:, 16:].int().reshape(-1, 4, 32)
    out = torch.empty(N * nb, 8, 32, device=r.device)
    for j in range(4):
        out[:, 2 * j] = (qs[:, j] & 15).float() * (d * sc[:, 2 * j:2 * j + 1]) - dm * mn[:, 2 * j:2 * j + 1]
        out[:, 2 * j + 1] = (qs[:, j] >> 4).float() * (d * sc[:, 2 * j + 1:2 * j + 2]) - dm * mn[:, 2 * j + 1:2 * j + 2]
    return out.reshape(N, K)


def dq_q3k(raw, N, K, dev="cpu"):
    """Q3_K (110 B / 256: hmask 32, qs 64, 12 B packed 6-bit scales, fp16 d) -> fp32 [N, K] (ggml dequantize_row_q3_K)"""
    nb = K // 256
    b = _raw(raw, dev).reshape(N * nb, 110)
    d = b[:, 108:110].contiguous().view(torch.float16).float().reshape(-1, 1, 1, 1)
    s = b[:, 96:108].long()
    a0 = (s[:, 0:4] % 16) | (((s[:, 8:12] // 1) % 4) * 16)
    a1 = (s[:, 4:8] % 16) | (((s[:, 8:12] // 4) % 4) * 16)
    a2 = ((s[:, 0:4] // 16) % 16) | (((s[:, 8:12] // 16) % 4) * 16)
    a3 = ((s[:, 4:8] // 16) % 16) | (((s[:, 8:12] // 64) % 4) * 16)
    sc = (torch.cat([a0, a1, a2, a3], 1) - 32).float().reshape(-1, 2, 4, 2).repeat_interleave(16, dim=3)      # [nb, half, j, 32]
    sh = (2 * torch.arange(4, device=b.device)).reshape(1, 1, 4, 1)
    q = (b[:, 32:96].long().reshape(-1, 2, 1, 32) >> sh) % 4
    bit = (torch.arange(2, device=b.device).reshape(2, 1) * 4 + torch.arange(4, device=b.device).reshape(1, 4)).reshape(1, 2, 4, 1)
    hm = (b[:, 0:32].long().reshape(-1, 1, 1, 32) >> bit) % 2
    return (d * sc * (q - 4 * (1 - hm)).float()).reshape(N, K)


def dq_expert(raw, ty, N, K, dev):
    if ty == 12:
        return dq_q4k(raw, N, K, dev)
    if ty == 11:
        return dq_q3k(raw, N, K, dev)
    raise ValueError(f"routed expert tensor type {ty} not handled (Q4_K=12, Q3_K=11)")


# ggml type id -> (format name, block elements): the SAME names foreign_linear._MMVQ_FN uses; the kernel is getattr(ext, "fq_mmvq_" + name)
# (llama.cpp mmvq port, fq_mmvq.cu, bound in bindings.cpp). Generic: keyed on the stored ggml type + K only, nothing model specific.
GG_MMVQ = {2: ("q4_0", 32), 3: ("q4_1", 32), 6: ("q5_0", 32), 7: ("q5_1", 32), 11: ("q3_k", 256), 12: ("q4_k", 256), 13: ("q5_k", 256),
           14: ("q6_k", 256), 16: ("iq2_xxs", 256), 17: ("iq2_xs", 256), 18: ("iq3_xxs", 256), 20: ("iq4_nl", 32), 21: ("iq3_s", 256),
           22: ("iq2_s", 256), 23: ("iq4_xs", 256)}
MMVQ_MAXR = 16        # decode rows per expert served by the row kernels (foreign_linear: rows < FUSED_ROWS_MAX), launched <= 4 columns at a time
_EXT = []

# ggml type id -> (grouped-kernel fmt id, block elements, block bytes): the fq_moe_kq_grouped_mixed_q8 table (fq_moe_kq.cu kFmtBytes). Generic: keyed on the stored type only.
# gate/up kernels exist for ids 0-5 and 8; down for 0-8.
GG_GROUPED = {12: (0, 256, 144), 13: (1, 256, 176), 14: (2, 256, 210), 23: (3, 256, 136), 18: (4, 256, 98), 22: (5, 256, 82), 8: (6, 32, 34),
              7: (7, 32, 24), 11: (8, 256, 110)}
GROUPED_UG = (0, 1, 2, 3, 4, 5, 8)
EXPERT_PREAD = os.environ.get("HY4_EXPERT_PREAD", "0") == "1"                    # 1 = old path: experts pread from the GGUF every step ((1a) mmvq)
RESIDENT_GB = float(os.environ.get("HY4_EXPERT_RESIDENT_GB", "72"))               # VRAM byte budget for resident routed experts (stored bytes)
PF_MOE = os.environ.get("HY4_PREFILL_MOE", "0") in ("1", "i8", "ref")       # 1 = prefill rows (T > MMVQ_MAXR) of resident experts: sorted-token fused dequant GEMM (prefill_kernels.py)
PF_MOE_I8 = os.environ.get("HY4_PREFILL_MOE", "0") == "i8"                 # i8 = Q8_K activations on int8 WMMA (llama.cpp mmq style)
PF_MOE_REF = os.environ.get("HY4_PREFILL_MOE", "0") == "ref"         # ref = exact fp32 torch math on batched-dequantised resident experts (quality reference, same maths as the pread path)
PF_DENSE = os.environ.get("HY4_PREFILL_DENSE", "0") in ("1", "i8")    # 1 = prefill rows of stored Q8_0 dense matrices: fused dequant fp16 GEMM; i8 = Q8 activations on int8 WMMA (prefill_kernels.py)
PF_DENSE_I8 = os.environ.get("HY4_PREFILL_DENSE", "0") == "i8"
PF_DIAG = os.environ.get("HY4_PREFILL_DIAG", "0") == "1"           # log fast-path vs exact error of every prefill dense matmul (debug)
PF_IX = os.environ.get("HY4_PREFILL_IX", "0") == "1"                # 1 = k-pool indexer scores of prefill rows as an fp16 GEMM (indexer_fused scoring is fp32, 2 GB temporaries at T = 8192)
PF_ROUTER2 = os.environ.get("HY4_PREFILL_ROUTER", "0") == "2"       # 2 = router logits of prefill rows from the split-fp16 kernel (hi + lo parts, near-fp32 accuracy)
PF_ROUTER = os.environ.get("HY4_PREFILL_ROUTER", "0") == "1"       # 1 = router logits of prefill rows as an fp16 GEMM (fp32 accumulate): fp32 matmul runs at 2.7 TFLOPS here


def _pk():
    from . import prefill_kernels
    return prefill_kernels


def moe_ref(z, sel, w, rsp, lim, group=16):
    """exact fp32 reference of the prefill MoE on the resident stored bytes: experts dequantised `group` at a time (torch dq_q4k / dq_q3k, same values as
    the pread path), per-expert fp32 matmuls, weighted index_add. Slow (host sync for the counts); for quality gates only."""
    ty = {0: 12, 8: 11}
    T, S = sel.shape
    dev = z.device
    nff, nout, E, K = rsp.nff, rsp.nout, rsp.E, z.shape[1]
    lat = torch.zeros(T, nout, device=dev)
    flat = sel.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=E).tolist()
    starts = [0]
    for c in counts:
        starts.append(starts[-1] + c)
    wf = w.reshape(-1)
    for e0 in range(0, E, group):
        e1 = min(E, e0 + group)
        if starts[e1] == starts[e0]:
            continue
        n = e1 - e0
        Wg = dq_expert(rsp.bufs["gate"][e0 * rsp.ebytes[0]:e1 * rsp.ebytes[0]], ty[rsp.fu], n * nff, K, dev).view(n, nff, K)
        Wu = dq_expert(rsp.bufs["up"][e0 * rsp.ebytes[0]:e1 * rsp.ebytes[0]], ty[rsp.fu], n * nff, K, dev).view(n, nff, K)
        Wd = dq_expert(rsp.bufs["down"][e0 * rsp.ebytes[1]:e1 * rsp.ebytes[1]], ty[rsp.fd], n * nout, nff, dev).view(n, nout, nff)
        for e in range(e0, e1):
            if counts[e] == 0:
                continue
            pr = order[starts[e]:starts[e + 1]]
            rows = pr // S
            x = z[rows]
            hh = torch.nn.functional.silu((x @ Wg[e - e0].T).clamp(max=lim)) * (x @ Wu[e - e0].T).clamp(-lim, lim)
            lat.index_add_(0, rows, (hh @ Wd[e - e0].T) * wf[pr].unsqueeze(-1))
    return lat


def _ext():
    """the compiled extension: $HY4_EXT_DIR/exllamav3_ext*.so if set (a tree built for THIS torch, e.g. the HIP tree on the Strix), else the tree root"""
    if not _EXT:
        try:
            import glob
            d = os.environ.get("HY4_EXT_DIR") or os.path.dirname(os.path.dirname(_HERE))
            f = sorted(glob.glob(os.path.join(d, "exllamav3_ext*.so")))[0]
            sp = importlib.util.spec_from_file_location("exllamav3_ext", f)
            e = importlib.util.module_from_spec(sp)
            sys.modules["exllamav3_ext"] = e
            sp.loader.exec_module(e)
            _EXT.append(e)
        except Exception as ex:            # noqa: BLE001
            sys.modules.pop("exllamav3_ext", None)
            print("hy4_forward: extension unavailable:", repr(ex)[:160], flush=True)
            _EXT.append(None)
    return _EXT[0]


def mmvq_fn(ty, K):
    """kernel for stored ggml type `ty` with row length K, or None (-> torch dequant path)"""
    f = GG_MMVQ.get(int(ty))
    e = _ext()
    if f is None or e is None or K % f[1] != 0 or not hasattr(e, "fq_mmvq_" + f[0]):
        return None
    return getattr(e, "fq_mmvq_" + f[0])


def _dq_q8_0_t(packed):
    """stored Q8_0 blocks [N, nb, 34] uint8 (any device) -> fp32 [N, nb*32] (fp16 d, 32 int8): exact, same values as engram_rows._dequant_q8_0"""
    d = packed[:, :, 0:2].contiguous().view(torch.float16).float()
    q = packed[:, :, 2:].contiguous().view(torch.int8).float()
    return (q * d).reshape(packed.shape[0], -1)


# DENSE table (move 3): stored ggml type id -> (block elems, bytes per block, decode GEMV name in the extension, torch dequant [N, nb, bb] -> fp32 [N, K]).
# Generic: keyed on the stored ggml type + K only, nothing model specific. A dense matrix whose type is here stays as stored bytes on the GPU;
# rows <= DENSE_MAXR run the extension GEMV (activation quantised to q8_1 inside, like llama.cpp), more rows dequantise per call (no resident fp32 copy).
GG_DENSE = {8: (32, 34, "fq_gemv_q8_0", _dq_q8_0_t)}
DENSE_MAXR = 16
GM_FFN = os.environ.get("HY4_GM_FFN", "0") == "1"       # 1 = decode FFN (<= 8 rows) through the shared gm_ext ops: gm_route + one Q8_0 launch (rdown, shared gate, shared up) + gm_moe_kq_decode + rup / shared swiglu-down in one buffer
_GMX = []


def _gm():
    if not _GMX:
        try:
            from .. import gm_ext
            _GMX.append(gm_ext.get())
        except Exception as ex:            # noqa: BLE001
            print("hy4_forward: gm_ext unavailable:", repr(ex)[:160], flush=True)
            _GMX.append(None)
    return _GMX[0]


NO_ALIAS_LAT = os.environ.get("HY4_LAT_CLONE", "0") == "1"     # 1 = copy the grouped MoE output before the rup matmul (old behaviour)
DENSE_DQ = os.environ.get("HY4_DENSE_DQ", "0") == "1"       # 1 = old path: dense matrices dequantised once to fp32 and kept resident


def dense_fn(ty, K):
    """(spec, GEMV or None) for stored type `ty` with row length K, or None when the type is not in GG_DENSE / K is not a block multiple"""
    spec = GG_DENSE.get(int(ty))
    if spec is None or K % spec[0] != 0:
        return None
    e = _ext()
    return spec, (getattr(e, spec[2], None) if e is not None else None)


_TM_HOLD = [None]          # the live _Tm (stage timers inside prefill dense calls)


class QLinear:
    """a dense [N, K] matrix kept as STORED bytes on the device; `x @ q.T` / q(x) = x [.., K] -> [.., N]"""

    def __init__(self, packed, ty, spec, gemv, stats=None):
        self.packed, self.ty, self.spec, self.gemv, self.stats = packed, int(ty), spec, gemv, stats
        self.shape = (packed.shape[0], packed.shape[1] * spec[0])
        self.gname = None
        _DR.register(self)

    @property
    def T(self):
        return _QT(self)

    def dense(self):
        return self.spec[3](self.packed)

    def __call__(self, x):
        K, N = self.shape[1], self.shape[0]
        lead = x.shape[:-1]
        x2 = x.reshape(-1, K)
        if x2.shape[0] <= DENSE_MAXR and self.gemv is not None:
            xf = x2.float().contiguous()
            out = torch.empty(xf.shape[0], N, dtype=torch.float32, device=x.device)
            self.gemv(xf, self.packed, out)
            if self.stats is not None:
                self.stats["dense_gemv"] = self.stats.get("dense_gemv", 0) + 1
        elif PF_DENSE and self.ty in (13, 14) and x2.is_cuda:       # requantised dense (HY4_DENSE_REQ_ARMS): fp16 dequant + fp16 GEMM, like the Q8_0 prefill path
            out = _DR.qdense_kq(self, x2)
        elif PF_DENSE and self.ty == 8 and x2.is_cuda:
            k_ = _pk()
            out = k_.qdense_i8(x2, self.packed) if PF_DENSE_I8 else (k_.qdense_tm(x2, self.packed, _TM_HOLD[0]) if x2.shape[0] >= k_.DENSE_TM_MIN_M else k_.qdense(x2, self.packed, k_.FMT_Q8))
            if PF_DIAG:
                ex = x2.float() @ self.dense().T
                df = out.float() - ex
                rr = df.norm(dim=1) / ex.norm(dim=1).clamp_min(1e-30)
                w_ = int(rr.argmax())
                print(f"DIAG dense N={N} K={K} M={x2.shape[0]} rel={float(df.norm() / ex.norm()):.2e} worst_row={w_} {float(rr[w_]):.2e} row0={float(rr[0]):.2e} max|x|={float(x2.abs().max()):.3e} "
                      f"clamped_in={int((x2.abs() > 65504).sum())} max|out|={float(ex.abs().max()):.3e} out_over={int((ex.abs() > 65504).sum())}", flush=True)
            if self.stats is not None:
                self.stats["dense_pf"] = self.stats.get("dense_pf", 0) + 1
        else:
            out = x2.float() @ self.dense().T
            if self.stats is not None:
                self.stats["dense_dq"] = self.stats.get("dense_dq", 0) + 1
        return out.to(x.dtype).reshape(*lead, N)


class _QT:
    def __init__(self, q):
        self.q = q

    def __rmatmul__(self, x):
        return self.q(x)


def load_linear(g, name, dev, stats=None):
    """Q8_0 (any type in GG_DENSE) dense tensor as QLinear on a cuda device when its GEMV exists and HY4_DENSE_DQ != 1; else None (caller dequantises)"""
    ty, dims, off = g.infos[name]
    if DENSE_DQ or torch.device(dev).type != "cuda" or len(dims) != 2:
        return None
    f = dense_fn(ty, dims[0])
    if f is None or f[1] is None:
        return None
    spec, fn = f
    K, N = dims[0], dims[1]
    raw = np.ascontiguousarray(g._pread(off, N * (K // spec[0]) * spec[1]))
    q = QLinear(torch.from_numpy(raw).reshape(N, K // spec[0], spec[1]).to(dev), ty, spec, fn, stats)
    q.gname = name
    return q


HEAD_CHUNK = 8192


def head_gemm(h, W, chunk=HEAD_CHUNK):
    """output head, generic by dtype: h [R, D] fp32, W [V, D] bf16 / f16 / f32 (resident on h.device) -> logits [R, V] fp32.
    Per vocab chunk W is upcast to fp32 and multiplied (R == 1 = the same mat-vec the old CPU head used; R > 1 = one GEMM per chunk)."""
    if _HB.usable(h, W):        # HY4_HEAD_BF16=1: one Triton mat-vec on the stored weights (no fp32 copy of W per chunk)
        return _HB.gemv(h, W)
    V = W.shape[0]
    out = torch.empty(h.shape[0], V, dtype=torch.float32, device=h.device)
    for a in range(0, V, chunk):
        w = W[a:a + chunk].float()
        if h.shape[0] == 1:
            out[0, a:a + chunk] = w @ h[0]
        else:
            out[:, a:a + chunk] = h @ w.T
    return out


class _Tm:
    """per-class synchronised wall timers (off unless .on)"""
    CLASSES = ("attn proj", "MLA core + sinks", "indexer / kpool", "router", "routed experts", "shared expert", "Engram", "hc / norms", "head", "other")

    def __init__(self):
        self.on, self.acc = False, {}

    def reset(self):
        self.acc = {}

    @contextlib.contextmanager
    def t(self, name):
        if not self.on:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.time()
        yield
        torch.cuda.synchronize()
        self.acc[name] = self.acc.get(name, 0.0) + time.time() - t0


def _mv(o, dev, dtype=None):
    for k, v in list(vars(o).items()):
        if torch.is_tensor(v) and v.is_floating_point():
            setattr(o, k, v.to(dev, dtype) if (dtype is not None and v.dim() > 1) else v.to(dev))
    return o


# ------------------------------------------------------------------------------------------------ the model
class Hy4Forward:
    _gs = None            # step_graph._GS while a decode graph is warmed up / captured (static buffers), else None

    def __init__(self, g, device="cpu", n_layers=None, cache_weights=False, hook=None, wdtype=torch.float32, use_indexer=True, q8_act=False):
        m = g.meta
        p = m["general.architecture"] + "."
        self.g, self.dev, self.hook, self.cache_weights, self.wdtype = g, torch.device(device), hook, cache_weights, wdtype
        self.n_trunk = m[p + "block_count"] - m.get(p + "nextn_predict_layers", 0)
        self.n_layers = n_layers or self.n_trunk
        assert self.n_layers <= self.n_trunk
        self.D, self.hc = m[p + "embedding_length"], m[p + "hyper_connection.count"]
        self.eps, self.hc_eps = float(m[p + "attention.layer_norm_rms_epsilon"]), float(m[p + "hyper_connection.epsilon"])
        self.magnitude = float(m[p + "hyper_connection.magnitude"])
        self.n_dense = m[p + "leading_dense_block_count"]
        self.topk, self.w_scale = m[p + "expert_used_count"], float(m[p + "expert_weights_scale"])
        self.clamp = list(m[p + "swiglu_clamp_exp"])
        self.ple = set(m[p + "ple.layers"])
        self.idx_types = list(m[p + "attention.indexer.types"])
        self.swa = list(m[p + "attention.sliding_window_pattern"])
        self.ixc = dict(n_head=m[p + "attention.indexer.head_count"], head_dim=m[p + "attention.indexer.key_length"],
                        top_k=m[p + "attention.indexer.top_k"], kpool=m[p + "attention.indexer.kpool"],
                        select_tail=bool(m[p + "attention.indexer.kpool_select_tail"]), rope_dim=m[p + "rope.dimension_count"],
                        rope_base=float(m[p + "rope.freq_base"]), eps=self.eps)
        self.use_indexer, self.q8_act = use_indexer, q8_act
        self.tm = _Tm()
        _TM_HOLD[0] = self.tm
        self.expert_dq = os.environ.get("HY4_EXPERT_DQ", "0") == "1"       # 1 = old path (torch dequant) for decode rows too
        self.dense_dq = DENSE_DQ
        self.head_cpu = os.environ.get("HY4_HEAD_CPU", "0") == "1" or self.dev.type != "cuda"     # 1 = old path: head on the CPU, last position only
        self.table = F8Rows(g)
        self.kp = _kpool_mod()
        self.kc = [None] * self.n_layers
        self.ix, self.eng, self._lc = {}, {}, {}
        self.tokens: list[int] = []
        self.n = 0
        self.stats = {"layer_s": 0.0}
        ty, dims, off = g.infos["token_embd.weight"]
        assert ty == 8 and dims[0] % 32 == 0
        self._emb = (off, dims[0])
        self.out_norm = load_dense(g, "output_norm.weight").to(self.dev)
        self.hh = [load_dense(g, f"output_hc_{k}.weight").to(self.dev) for k in ("fn", "base", "scale")]
        self.out_w = g.get_tensor("output.weight", allow_bf16=True)           # [vocab, n_embd] bf16 (or f32)
        assert tuple(self.out_w.shape) == (m[p + "vocab_size"], self.D), self.out_w.shape
        if not self.head_cpu:
            self.out_w = self.out_w.to(self.dev)         # resident in its stored dtype (bf16: 0.74 GB)
        self._load_resident()

    def _load_resident(self):
        '''routed experts of every MoE layer (while the byte budget lasts) resident on the GPU in their STORED bytes; decode rows then go through
        fq_moe_kq_grouped_mixed_q8 (one launch set per layer). Layers whose formats are not in the grouped table, or past the budget, keep the pread path.'''
        self.res, self.res_bytes = {}, 0
        self.stats["resident_layers"], self.stats["resident_gib"] = 0, 0.0
        if EXPERT_PREAD or self.expert_dq or self.dev.type != "cuda":
            return
        e = _ext()
        if e is None or not hasattr(e, "fq_moe_kq_grouped_mixed_q8") or not hasattr(e, "fq_moe_kq_set_swiglu_limit"):
            print("hy4_forward: grouped kernel unavailable -> experts stay on the pread path", flush=True)
            return
        g, dev, budget, skipped = self.g, self.dev, int(RESIDENT_GB * 2**30), []
        for i in range(self.n_dense, self.n_layers):
            info = {n: g.infos[f"blk.{i}.ffn_{n}_exps.weight"] for n in ("gate", "up", "down")}
            f = {n: GG_GROUPED.get(int(info[n][0])) for n in info}
            if None in f.values() or f["gate"][0] != f["up"][0] or f["gate"][0] not in GROUPED_UG or info["gate"][1] != info["up"][1]:
                skipped.append((i, "format"))
                continue
            if any(info[n][1][0] % f[n][1] or (f[n][0] < 6 and info[n][1][0] % 256) for n in info):
                skipped.append((i, "shape"))
                continue
            rows = {n: (info[n][1][1], info[n][1][0] // f[n][1] * f[n][2], info[n][1][2]) for n in info}      # (rows, row bytes, experts)
            need = sum(r * rb * E for r, rb, E in rows.values())
            if self.res_bytes + need > budget:
                skipped.append((i, "budget"))
                break
            bufs, ptrs, strides = {}, {}, {}
            for n in ("gate", "up", "down"):
                r, rb, E = rows[n]
                tot = r * rb * E
                buf = torch.empty(tot, dtype=torch.uint8, device=dev)
                off = info[n][2]
                for o in range(0, tot, 1 << 27):
                    c = min(1 << 27, tot - o)
                    buf[o:o + c].copy_(torch.from_numpy(g._pread(off + o, c)))
                bufs[n], strides[n] = buf, rb
                ptrs[n] = torch.arange(E, dtype=torch.int64, device=dev) * (r * rb) + buf.data_ptr()
            self.res[i] = SimpleNamespace(bufs=bufs, ptrs=(ptrs["gate"], ptrs["up"], ptrs["down"]), strides=(strides["gate"], strides["up"], strides["down"]),
                                          fu=f["gate"][0], fd=f["down"][0], nff=rows["gate"][0], nout=rows["down"][0], E=rows["gate"][2],
                                          ebytes=(rows["gate"][0] * rows["gate"][1], rows["down"][0] * rows["down"][1]))
            self.res_bytes += need
        for fd in getattr(g, "fds", []):
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except Exception:             # noqa: BLE001
                pass
        self.stats["resident_layers"], self.stats["resident_gib"] = len(self.res), round(self.res_bytes / 2**30, 3)
        print(f"hy4_forward: resident experts {len(self.res)} layers {self.res_bytes / 2**30:.2f} GiB (budget {RESIDENT_GB} GiB) skipped={skipped[:3]}", flush=True)

    # -- weights
    def _w(self, name):
        q = load_linear(self.g, name, self.dev, self.stats)
        return q if q is not None else load_dense(self.g, name).to(self.dev, self.wdtype)

    def _lin(self, name):
        q = load_linear(self.g, name, self.dev, self.stats)
        return q if q is not None else load_dense(self.g, name)

    def _layer(self, i):
        if i in self._lc:
            return self._lc[i]
        g, b, dev = self.g, f"blk.{i}.", self.dev
        L = SimpleNamespace()
        L.mla = Hy4MLA(g, i, q8_act=self.q8_act, lin=self._lin)
        _mv(L.mla, dev, self.wdtype)
        if dev.type == "cuda":       # stored Q8_0 bytes of the two per-head MLA matrices (used when HY4_MLA_BG=1, T == 1)
            L.mla.bg_kb = _BGM.load(g, b + "attn_k_b.weight", L.mla.H, dev)
            L.mla.bg_vb = _BGM.load(g, b + "attn_v_b.weight", L.mla.H, dev)
        L.mla.tm = self.tm
        L.mla.kpos_arange = True        # layer(): kpos is torch.arange(N) (ref_fused window / causal block skip)
        if os.environ.get("HY4_ATTN_GROUPED", "0") == "1":          # flag off: nothing below runs, the old path is untouched
            from .attn_grouped import install as _attn_group
            _attn_group(L.mla)
        L.mla.post_norm, L.mla.sinks = L.mla.post_norm.to(dev), L.mla.sinks.to(dev)
        for tag in ("hca", "hcf"):
            s = "attn" if tag == "hca" else "ffn"
            setattr(L, tag, [load_dense(g, f"{b}hc_{s}_{k}.weight").to(dev) for k in ("fn", "base", "scale")])
        L.attn_norm, L.ffn_norm, L.post_ffw = (load_dense(g, b + n).to(dev) for n in ("attn_norm.weight", "ffn_norm.weight", "post_ffw_norm.weight"))
        if i < self.n_dense:
            L.dense = [self._w(f"{b}ffn_{n}.weight") for n in ("gate", "up", "down")]
            if os.environ.get("HY4_FFN_GROUPED", "0") == "1":
                from .attn_grouped import group_linears
                v = group_linears(L.dense[:2])
                if v is not None:
                    L.dense[:2] = v
        else:
            L.router = g.get_tensor(b + "ffn_gate_inp.weight").float().to(dev)
            L.rbias = g.get_tensor(b + "exp_probs_b.bias").float().to(dev)
            L.rdown, L.rup = self._w(b + "ffn_routed_down.weight"), self._w(b + "ffn_routed_up.weight")
            L.shexp = [self._w(f"{b}ffn_{n}_shexp.weight") for n in ("gate", "up", "down")]
            if os.environ.get("HY4_FFN_GROUPED", "0") == "1":
                from .attn_grouped import group_linears
                v = group_linears([L.rdown, L.shexp[0], L.shexp[1]])
                if v is not None:
                    L.rdown, L.shexp[0], L.shexp[1] = v
            L.ety = g.infos[b + "ffn_gate_exps.weight"][0]
            L.ety_down = g.infos[b + "ffn_down_exps.weight"][0]
        L.ix = bool(self.use_indexer and self.idx_types[i] and not self.swa[i] and (b + "indexer.attn_k.weight") in g.infos)
        if self.cache_weights:
            self._lc[i] = L
        return L

    def _indexer(self, i):
        if i not in self.ix:
            g, b = self.g, f"blk.{i}."
            names = {"attn_k": "indexer.attn_k.weight", "k_norm": "indexer.k_norm.weight", "q_b": "indexer.attn_q_b.weight",
                     "proj": "indexer.proj.weight", "value": "indexer_compressor_value.weight", "gate": "indexer_compressor_gate.weight",
                     "ape": "indexer_compressor_ape.weight", "pool_norm": "indexer_compressor_norm.weight"}
            w = {k: load_dense(g, b + s) for k, s in names.items()}
            q8 = tuple(k for k, s in names.items() if g.infos[b + s][0] == 8)
            self.ix[i] = self.kp.KPoolIndexer(w, q8_weights=q8, **self.ixc)
        return self.ix[i]

    def _indexer_fused(self, i):
        fx = self.__dict__.setdefault("ixf", {})
        if i not in fx:
            from .indexer_fused import FusedKPoolIndexer
            g, b = self.g, f"blk.{i}."
            big = {"attn_k": "indexer.attn_k.weight", "q_b": "indexer.attn_q_b.weight",
                   "value": "indexer_compressor_value.weight", "gate": "indexer_compressor_gate.weight"}
            small = {"k_norm": "indexer.k_norm.weight", "proj": "indexer.proj.weight",
                     "ape": "indexer_compressor_ape.weight", "pool_norm": "indexer_compressor_norm.weight"}
            w = {k: self._lin(b + s) for k, s in big.items()}       # QLinear (stored Q8 bytes) or fp32 tensor; `x @ w.T` works for both
            w.update({k: load_dense(g, b + s).to(self.dev) for k, s in small.items()})
            w = {k: (v.to(self.dev) if isinstance(v, torch.Tensor) else v) for k, v in w.items()}
            fx[i] = FusedKPoolIndexer(w, self.dev, **self.ixc)
            if PF_IX and self.dev.type == "cuda":
                fx[i] = _pk().pf_indexer(fx[i], self.tm)
        return fx[i]

    # -- pieces
    def _site(self, x, hw):
        fn, base, scale = hw
        T, H = x.shape[0], self.hc
        mix = rms(x.reshape(T, -1), self.eps) @ fn.T
        pre = torch.sigmoid(mix[:, :H] * scale[0] + base[:H]) + self.hc_eps
        post = self.magnitude * torch.sigmoid(mix[:, H:] * scale[1] + base[H:]) + self.hc_eps
        return pre, post

    def _mm(self, x, w):
        if isinstance(w, QLinear):
            return w(x).float()
        return (x.to(w.dtype) @ w.T).float()

    def _ffn(self, i, L, x):
        tm = self.tm
        if i < self.n_dense:
            gt, up, dn = L.dense
            with tm.t("other"):
                return self._mm(torch.nn.functional.silu(self._mm(x, gt)) * self._mm(x, up), dn), None
        T, dev = x.shape[0], self.dev
        if self._gm_ok(i, L, x):
            return self._ffn_gm(i, L, x), None
        rsp = self.res.get(i)
        pf = bool(PF_MOE and T > MMVQ_MAXR and dev.type == "cuda" and rsp is not None and rsp.fu in _pk().FROM_GROUPED and rsp.fd in _pk().FROM_GROUPED)
        with tm.t("router"):
            if _rf.ENABLED:       # HY4_REF_FUSED=1: one router kernel; the host sync (tolist) only where the pread loop needs it
                sel, w = _rf.router_select(_RG.logits(x, L.router) if _RG.usable(x, L.router) else x @ L.router.T, L.rbias, self.topk, self.w_scale)
                sel_l = [] if (pf or (self.res.get(i) is not None and T <= MMVQ_MAXR and dev.type == "cuda")) else sorted(set(sel.flatten().tolist()))
            else:
                if PF_ROUTER and T > MMVQ_MAXR and dev.type == "cuda":
                    if getattr(L, "router16", None) is None:
                        L.router16 = L.router.half().contiguous()
                    p = torch.sigmoid(_pk().mm_f32(_pk().to_f16(x), L.router16.T))
                else:
                    p = torch.sigmoid(_RG.logits(x, L.router) if _RG.usable(x, L.router) else x @ L.router.T)
                sel = torch.sort(p + L.rbias, dim=-1, descending=True, stable=True).indices[:, :self.topk]      # stable: tied experts resolve to the lowest id every call (torch.topk tie order varies between calls on gfx1151)
                w = torch.gather(p, 1, sel)
                w = w / w.sum(-1, keepdim=True).clamp(min=6.103515625e-5) * self.w_scale
                sel_l = [] if (pf or _ns.skip_host_ids(self, i, T, MMVQ_MAXR)) else sorted(set(sel.flatten().tolist()))
        with tm.t("other"):
            z = self._mm(x, L.rdown)
            lat = None          # allocated (zeros) only by the paths that accumulate into it: the grouped / prefill paths overwrite it
        lim, N = float(self.clamp[i]), z.shape[1]
        b = f"blk.{i}."
        if pf and PF_MOE_REF:
            with tm.t("routed experts"):
                lat = moe_ref(z, sel, w, rsp, lim)
            sel_l = []
        elif pf:
            with tm.t("routed experts"):
                k_ = _pk()
                lat = (k_.moe_prefill_i8 if PF_MOE_I8 else k_.moe_prefill)(z, sel, w, (rsp.bufs["gate"], rsp.bufs["up"], rsp.bufs["down"]), (k_.FROM_GROUPED[rsp.fu], k_.FROM_GROUPED[rsp.fd]), lim,
                                     rsp.nff, rsp.nout, (rsp.strides[0], rsp.strides[2]), rsp.ebytes, rsp.E, tm=tm)
            self.stats["prefill_moe_calls"] = self.stats.get("prefill_moe_calls", 0) + 1
            self.stats["moe_path_" + getattr(k_, "LAST_PATH", "?")] = self.stats.get("moe_path_" + getattr(k_, "LAST_PATH", "?"), 0) + 1
            sel_l = []
        rs = self.res.get(i) if (T <= MMVQ_MAXR and dev.type == "cuda") else None
        if rs is not None:
            with tm.t("routed experts"):
                ext, S = _ext(), sel.shape[1]
                if getattr(self, "_gpart", None) is None or self._gpart.shape[1] != N:
                    self._gpart = torch.empty(MMVQ_MAXR * S, N, dtype=torch.float32, device=dev)
                    self._gout = torch.empty(MMVQ_MAXR, N, dtype=torch.float32, device=dev)
                ext.fq_moe_kq_set_swiglu_limit(lim)
                ext.fq_moe_kq_grouped_mixed_q8(z.half().contiguous(), sel.contiguous(), w.half().contiguous(), *rs.ptrs, *rs.strides, rs.fu, rs.fd,
                                               self._gpart[:T * S], self._gout[:T])
                lat = self._gout[:T] if (self.hook is None and not NO_ALIAS_LAT) else self._gout[:T].clone()       # rup reads it before the next layer's grouped call overwrites _gout (one stream); hooks keep a copy
            self.stats["grouped_calls"] = self.stats.get("grouped_calls", 0) + 1
            sel_l = []          # the per-expert pread loop below runs zero times
        if lat is None:
            lat = torch.zeros(T, N, device=dev)
        kg, kd = (None, None) if (self.expert_dq or T > MMVQ_MAXR or dev.type != "cuda") else (mmvq_fn(L.ety, N), mmvq_fn(L.ety_down, N))
        use_k = kg is not None or kd is not None          # per tensor: a stored type without a kernel keeps the torch dequant for that tensor only
        with tm.t("routed experts"):
            if use_k:
                ext, di = _ext(), (dev.index if dev.index is not None else torch.cuda.current_device())
            for e in sel_l:
                rows, slot = (sel == e).nonzero(as_tuple=True)
                zi = z[rows]
                R = zi.shape[0]
                raw = {n: self.g.get_tensor(f"{b}ffn_{n}_exps.e{e}.weight") for n in ("gate", "up", "down")}
                ty = {"gate": L.ety, "up": L.ety, "down": L.ety_down}
                fk = {"gate": kg, "up": kg, "down": kd}

                def mv(xr, n):
                    """x [R,K] . W_n.T -> [R,N] fp32: mmvq on the stored bytes if the type has a kernel, else torch dequant (old path)"""
                    self.stats["mmvq_calls" if fk[n] is not None else "dq_calls"] = self.stats.get("mmvq_calls" if fk[n] is not None else "dq_calls", 0) + 1
                    if fk[n] is None:
                        return xr @ dq_expert(raw[n], ty[n], N, N, dev).T
                    Wb = _raw(raw[n], dev).reshape(N, -1).contiguous()
                    qb = ext.fq_q8_scratch(di, R, N)
                    ext.fq_quant_q8_1_rows(xr.half().contiguous(), qb)
                    o = torch.empty(R, N, dtype=torch.float32, device=dev)
                    for a0 in range(0, R, 4):
                        fk[n](qb[a0:a0 + 4], Wb, o[a0:a0 + 4])
                    return o
                ga, upv = mv(zi, "gate"), mv(zi, "up")
                h = torch.nn.functional.silu(ga.clamp(max=lim)) * upv.clamp(-lim, lim)
                dn_o = mv(h, "down")
                lat.index_add_(0, rows, dn_o * w[rows, slot].unsqueeze(-1))
        with tm.t("other"):
            y = self._mm(lat, L.rup)
        gt, up, dn = L.shexp
        with tm.t("shared expert"):
            sh = self._mm(torch.nn.functional.silu(self._mm(x, gt)) * self._mm(x, up), dn)
        return y + sh, lat

    def _gm_ok(self, i, L, x):
        """layer i FFN can run on the gm_ext ops: resident experts in a gm format, Q8_0 latent / shared matrices, 1..8 rows, no hooks"""
        if not GM_FFN or self.hook is not None or self.dev.type != "cuda" or i < self.n_dense or x.shape[0] > 8:
            return False
        rs = self.res.get(i)
        if rs is None or rs.fu not in (0, 8) or rs.fd not in (0, 2) or _gm() is None:
            return False
        if rs.fu == 8 and x.shape[0] >= 2:       # Q3_K gate/up: gm_moe_kq_decode matches stock only at R = 1 (GGUF agent 19/11); the stock grouped kernel is 1-2 % faster for verify rows
            return False
        return all(isinstance(q, QLinear) and q.ty == 8 for q in (L.rdown, L.rup, *L.shexp))

    def _ffn_gm(self, i, L, x):
        gm, rs, dev = _gm(), self.res[i], self.dev
        T, S = x.shape[0], self.topk
        w3 = getattr(L, "gm_w", None)
        if w3 is None:
            q8 = lambda q: q.packed.reshape(q.packed.shape[0], -1)
            w3 = L.gm_w = ([q8(L.rdown), q8(L.shexp[0]), q8(L.shexp[1])], [q8(L.rup)], q8(L.shexp[2]))
        N, D = L.rdown.shape[0], L.rup.shape[0]
        st = getattr(self, "_gms", None)
        if st is None:
            E = L.router.shape[0]
            st = self._gms = SimpleNamespace(
                sc=torch.empty(8, E, dtype=torch.float32, device=dev), sel=torch.empty(8, S, dtype=torch.int64, device=dev), w=torch.empty(8, S, dtype=torch.float16, device=dev),
                z3=torch.empty(8 * 3 * N, dtype=torch.float16, device=dev), qa=torch.empty(8 * max(x.shape[1], N) // 32 * 36, dtype=torch.uint8, device=dev),
                qm=torch.empty(8 * N // 32 * 36, dtype=torch.uint8, device=dev), act=torch.empty(8 * S, N, dtype=torch.float16, device=dev),
                part=torch.empty(8 * S, N, dtype=torch.float32, device=dev), lat=torch.empty(8, N, dtype=torch.float32, device=dev),
                meta=torch.zeros(8 * S * 9, dtype=torch.int32, device=dev), nometa=torch.empty(0, dtype=torch.int32, device=dev), nores=torch.empty(0, dtype=torch.float32, device=dev))
        sel, w = st.sel[:T], st.w[:T]
        xc = x.contiguous()
        gm.gm_route(xc, L.router, L.rbias, st.sc[:T], sel, w, 2, True, self.w_scale, 6.103515625e-5)
        z3 = st.z3[:3 * T * N]
        gm.gm_qkv_decode(xc, w3[0], [3, 3, 3], z3.view(1, -1), st.qa)
        z, g, u = z3[:T * N].view(T, N), z3[T * N:2 * T * N].view(T, N), z3[2 * T * N:].view(T, N)
        if T >= 2:
            st.meta.zero_()
        gm.gm_moe_kq_decode(z, sel.reshape(-1), w.reshape(-1), *rs.ptrs, *rs.strides[:3], rs.fd, st.act, st.qm, st.part, st.lat[:T], st.nores, rs.fu, float(self.clamp[i]),
                            st.meta if T >= 2 else st.nometa)
        y = torch.empty(T, D, dtype=torch.float32, device=dev)
        gm.gm_qkv_decode(st.lat[:T], w3[1], [3], y.view(1, -1), st.qm)
        gm.gm_q8_swiglu_down(g, u, w3[2], y, True)
        self.stats["gm_ffn_calls"] = self.stats.get("gm_ffn_calls", 0) + 1
        return y

    def _engram(self, i, x, ids, start):
        if os.environ.get("HY4_ENGRAM_GPU", "0") in ("1", "2") and self.dev.type == "cuda":      # flag off: the old CPU path below, untouched
            if i not in self.eng:
                from .engram_gpu import Hy4EngramGPU
                self.eng[i] = Hy4EngramGPU(self.g, i, self.dev, table=self.table, lin=None if os.environ.get("HY4_ENGRAM_GPU") == "2" else self._lin)    # 2 = dense fp32 kv (bit-exact vs CPU path)
            with self.tm.t("Engram"):
                fut = getattr(self, "_eng_pf", {}).pop((i, start, x.shape[0]), None)
                with self.tm.t("f engram: wait for prefetched rows"):
                    pre = fut.result() if fut is not None else None
                return self.eng[i](x, ids, start, pre=pre)
        if i not in self.eng:
            self.eng[i] = Hy4Engram(self.g, i, table=self.table)
        with self.tm.t("Engram"):
            return self.eng[i](x.cpu(), ids, start).to(self.dev)

    def prefetch_engram(self, ids_all, chunk):
        """HY4_ENGRAM_PREFETCH=1: the Engram row ids depend on token ids only, so hash + pread of every (chunk, Engram layer) of a known prompt run in a
        background thread, in consumption order, while the GPU works on earlier layers / chunks. Consumed by _engram (entries are dropped when used)."""
        if os.environ.get("HY4_ENGRAM_PREFETCH", "0") != "1" or os.environ.get("HY4_ENGRAM_GPU", "0") not in ("1", "2") or self.dev.type != "cuda":
            return
        from concurrent.futures import ThreadPoolExecutor
        from .engram_gpu import Hy4EngramGPU
        layers = sorted(i for i in self.ple if i < self.n_layers)
        for i in layers:
            if i not in self.eng:
                self.eng[i] = Hy4EngramGPU(self.g, i, self.dev, table=self.table, lin=None if os.environ.get("HY4_ENGRAM_GPU") == "2" else self._lin)
        toks = [int(t) for t in ids_all]
        self._eng_pf = {}
        ex = ThreadPoolExecutor(max_workers=2)
        for s in range(self.n, len(toks), chunk):
            n = min(chunk, len(toks) - s)
            for i in layers:
                self._eng_pf[(i, s, n)] = ex.submit(self.eng[i].host_rows, toks, s, n)
        ex.shutdown(wait=False)

    def _hk(self, name, i, t):
        if self.hook is not None:
            self.hook(name, i, t)

    def layer(self, i, x, pos, ids, start):
        L = self._layer(i)
        T = x.shape[0]
        if _AE.ENABLED:       # HY4_ATTN_EAGER: host bound for the rope tables (no sync)
            _AE.ensure(start + T)
        gs = self._gs
        if i in self.ple:
            x = gs.engram(self, i, x) if gs is not None else self._engram(i, x, ids, start)
            self._hk("engram_out", i, x)
        tm = self.tm
        with tm.t("hc / norms"):
            if _HCF is not None and _HCF.ok(x, L.hca, self.hc):
                post, cur = _HCF.site(x, L.hca, L.attn_norm, self.eps, self.hc_eps, self.magnitude)
            else:
                pre, post = self._site(x, L.hca)
                cur = rms((x * pre[:, :, None]).sum(1), self.eps) * L.attn_norm
        mla = L.mla
        if gs is not None and gs.T == 1 and _KVF.ENABLED and _AE.ENABLED:       # HY4_KV_FUSED: rms + rope + K-row write in ONE kernel
            kc = gs.kv_row_fused(i, mla, cur, pos)
        else:
            with tm.t("attn proj"):
                kn = mla.kv(cur, pos)
            kc = gs.append_kv(i, kn) if gs is not None else _ns.append_kv(self, i, kn, start) if _ns.active(self) else (torch.cat([self.kc[i], kn]) if self.kc[i] is not None else kn)
        N = kc.shape[0]
        allow = None
        if L.ix:
          with tm.t("indexer / kpool"):
            qr = mla._q(cur, "wq_a") @ mla.wq_a.T
            if _mla_mod.pf_f16():
                mla._qa_pre = qr
            qr = _HCF.rmsw(qr, mla.q_a_norm, mla.eps, emit16=_pk().PF_FUSE) if _HCF is not None else rms(qr, mla.eps) * mla.q_a_norm
            if _pk().PF_FUSE:
                mla._qr_n = qr          # attend_ck reuses the normalised q_a (and its fp16 copy) instead of a second norm
            if os.environ.get("HY4_INDEXER_FUSED", "0") == "1" and self.dev.type == "cuda":      # flag off: the old CPU path below, untouched
                allow = gs.indexer(self, i, cur, qr, N) if gs is not None else self._indexer_fused(i).forward(cur, qr, N)
            else:
                sel = self._indexer(i).forward(cur.cpu(), qr.cpu())
                if any(len(r) < int(pp) + 1 for r, pp in zip(sel.rows, pos.tolist())):
                    allow = torch.zeros(T, N, dtype=torch.bool)
                    for t, r in enumerate(sel.rows):
                        allow[t, r.long()] = True
                    allow = allow.to(self.dev)
        kqv, gated, out = mla.attend(cur, pos, kc, gs.ar[:N] if gs is not None else _ns.kpos(self, N) if _ns.active(self) else torch.arange(N, device=self.dev), allow)
        self.kc[i] = kc
        self._hk("attn_kqv", i, kqv)
        self._hk("attn_gated", i, gated)
        self._hk("attn_out", i, out)
        with tm.t("hc / norms"):
            if _HCF is not None and _HCF.ok(x, L.hcf, self.hc) and out.shape == (T, self.D):
                s1, post2, cur2 = _HCF.res_site(x, out, post, L.hcf, L.ffn_norm, self.eps, self.hc_eps, self.magnitude)
            else:
                s1 = x + post[:, :, None] * out[:, None, :]
                pre2, post2 = self._site(s1, L.hcf)
                cur2 = rms((s1 * pre2[:, :, None]).sum(1), self.eps) * L.ffn_norm
        f, lat = self._ffn(i, L, cur2)
        if lat is not None:
            self._hk("ffn_moe_out", i, lat)
        with tm.t("hc / norms"):
            if _HCF is not None and _HCF.ok(s1, L.hcf, self.hc) and f.shape == (T, self.D):
                s2 = _HCF.resid(s1, f, post2, L.post_ffw, self.eps)
            else:
                f = rms(f, self.eps) * L.post_ffw
                s2 = s1 + post2[:, :, None] * f[:, None, :]
        self._hk("l_out", i, s2)
        return s2

    def embed(self, ids):
        off, K = self._emb
        rb = K // 32 * 34
        rows = [_dequant_q8_0(self.g._pread(off + int(t) * rb, rb), K, 1) for t in ids]
        return torch.cat(rows).to(self.dev)

    def head(self, x, all_pos=False):
        """x [T, hc, n_embd] -> logits of the LAST position [vocab] (all_pos: [T, vocab])"""
        with self.tm.t("head"):
            return self._head(x, all_pos)

    def head_state(self, xl):
        """xl [R, hc, n_embd] -> [R, n_embd]: the trunk state the output head reads (hyper-head mix + output norm), no vocabulary GEMM"""
        fn, base, scale = self.hh
        R = xl.shape[0]
        if _HCF is not None and self.hook is None and _HCF.ok(xl, self.hh, self.hc):
            return _HCF.site(xl, self.hh, self.out_norm, self.eps, self.hc_eps, self.magnitude)[1]
        mix = rms(xl.reshape(R, -1), self.eps) @ fn.T
        pre = torch.sigmoid(mix * scale[0] + base) + self.hc_eps
        col = (xl * pre[:, :, None]).sum(1)
        self._hk("hc_head", -1, col)
        return rms(col, self.eps) * self.out_norm

    def _head(self, x, all_pos=False):
        xl = x if all_pos else x[-1:]
        h = self.head_state(xl)
        self._h = h           # the trunk state the output head reads (MTP draft input)
        if self.head_cpu:
            h = h.cpu()
        out = head_gemm(h.float(), self.out_w)
        return out if all_pos else (out[0] if getattr(self, "_ns_dev", False) else out[0].cpu())

    def step(self, ids, all_pos=False):
        """append ids (prefill = all prompt ids at once, decode = one id), return last-position logits (None on a truncated stack)"""
        if _ns.active(self):
            return _ns.step(self, ids, all_pos)
        if self.tm.on:
            torch.cuda.synchronize()
            base = sum(self.tm.acc.values())
        t0 = time.time()
        start = self.n
        self.tokens += [int(t) for t in ids]
        pos = torch.arange(start, start + len(ids), device=self.dev)
        e = self.embed(ids)
        x = e[:, None, :].expand(-1, self.hc, -1).contiguous()
        for i in range(self.n_layers):
            x = self.layer(i, x, pos, self.tokens, start)
        self.n += len(ids)
        self._x = x
        lg = self.head(x, all_pos) if self.n_layers == self.n_trunk else None
        if self.tm.on:
            torch.cuda.synchronize()
            tot = time.time() - t0
            self.tm.acc["other"] = self.tm.acc.get("other", 0.0) + max(0.0, tot - (sum(self.tm.acc.values()) - base))
            self.tm.last_total = tot
        self.stats["layer_s"] += time.time() - t0
        return lg

    def generate(self, ids, n_new):
        if _ns.active(self):
            return _ns.generate(self.step, ids, n_new)
        t0 = time.time()
        lg = self.step(ids)
        t1 = time.time()
        first, out = lg, []
        for _ in range(n_new):
            nxt = int(lg.argmax())
            out.append(nxt)
            if len(out) == n_new:
                break
            lg = self.step([nxt])
        t2 = time.time()
        return first, out, {"prefill_s": t1 - t0, "decode_s": t2 - t1, "decode_steps": max(len(out) - 1, 0)}
