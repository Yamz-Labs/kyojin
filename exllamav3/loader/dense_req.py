"""HY4_DENSE_REQ_ARMS=1: the dense Q8_0 matrices of the Hy4 trunk (attention + shared / latent projections) can be switched at run time to
Q6_K or Q5_K bytes (requantised offline by tools/hy4/dense_requant.py, same bytes as llama-quantize --allow-requantize --tensor-type) to measure
the quality cost and the decode gain of reading fewer weight bytes per token. Default off: nothing here is imported or registered.

  register(owner)      called by QLinear / attn_grouped._Group when the flag is set (weak references)
  prepare(kinds)       read OUTDIR/<kind>/<tensor>.bin for every registered owner whose tensors all have a file; owners without one stay Q8_0
  set_mode(mode, kern) mode "q8" | "q6_k" | "q5_k" for every prepared owner; kern "gemv" (fq_gemv_*_q8) | "mmvq" (fq_mmvq_*) picks the decode kernel.
                       Rows <= DENSE_MAXR run the K-quant GEMV on q8_1 activations; more rows (prefill) dequantise the weight to fp16 per call.
The caller drops the captured graphs (they bake the kernels and the weight pointers)."""
from __future__ import annotations

import os
import weakref

import numpy as np
import torch

ENABLED = os.environ.get("HY4_DENSE_REQ_ARMS", "0") == "1"
ALT_DIR = os.environ.get("HY4_DENSE_ALT", "/path/to/alt")
KQ = {"q6_k": dict(bb=210, blk=256), "q5_k": dict(bb=176, blk=256)}
_OWNERS: list = []
_EXT = []


def register(owner):
    if ENABLED:
        _OWNERS.append(weakref.ref(owner))


def _ext():
    from . import hy4_forward as hf
    return hf._ext()


def _dq_fn(kind):
    """torch dequant [N, K/256, bb] uint8 -> fp32 [N, K] (the QLinear.dense() spec slot): the extension's fp16 dequant kernel"""
    name = "fq_dequant_" + kind

    def f(packed):
        N, nb, bb = packed.shape
        out = torch.empty(N, nb * 256, dtype=torch.float16, device=packed.device)
        getattr(_ext(), name)(packed.reshape(N, -1).contiguous(), out)
        return out.float()
    return f


def dequant16(lin):
    """fp16 [K, N] (the layout torch.mm(x, w) wants) of a K-quant QLinear, one dequant launch"""
    N, nb, bb = lin.packed.shape
    out = torch.empty(N, nb * 256, dtype=torch.float16, device=lin.packed.device)
    getattr(_ext(), "fq_dequant_" + {210: "q6_k", 176: "q5_k"}[bb])(lin.packed.reshape(N, -1), out)
    return out.t()


def _pk():
    from . import prefill_kernels
    return prefill_kernels


def raw16(lin, x, tm=None):
    """prefill dense: x [M, K] -> raw fp16 GEMM output [M, N] (qdense_raw16 for stored Q8_0, fp16 dequant + GEMM for a requantised weight)"""
    pk = _pk()
    if lin.ty == 8:
        return pk.qdense_raw16(x, lin.packed, tm)
    x2 = pk.to_f16(x.reshape(-1, lin.shape[1])).contiguous()
    return torch.mm(x2, dequant16(lin))


def full(lin, x, tm=None):
    """prefill dense: x [.., K] -> [.., N] fp32 (qdense_tm for stored Q8_0)"""
    pk = _pk()
    if lin.ty == 8:
        return pk.qdense_tm(x, lin.packed, tm)
    return qdense_kq(lin, x)


def qdense_kq(lin, x):
    pk = _pk()
    lead = x.shape[:-1]
    x2 = pk.to_f16(x.reshape(-1, lin.shape[1])).contiguous()
    o = torch.mm(x2, dequant16(lin))
    out = torch.nan_to_num(o.float(), nan=0.0, posinf=pk.F16MAX, neginf=-pk.F16MAX)
    return out.reshape(*lead, lin.shape[0])


def _gemv(kind, kern):
    """decode GEMV with the QLinear signature (xf [M, K] fp32, packed [N, nb, bb], out [M, N] fp32): q8_1 quantisation of x, then the K-quant dot"""
    fg = "fq_gemv_%s_q8" % kind
    fm = "fq_mmvq_" + kind

    def f(x, packed, out):
        e = _ext()
        M, K = x.shape
        qb = e.fq_q8_scratch(x.get_device(), M, K)
        e.fq_quant_q8_1_rows(x, qb)
        p2 = packed.reshape(packed.shape[0], -1)
        # Q5_K has no multi-row GEMV kernel (rows 2..16 re-read the weights once per row): rows >= 2 use the mmvq kernel
        if kern == "mmvq" or (kind == "q5_k" and M >= 2):
            for a in range(0, M, 8):
                getattr(e, fm)(qb[a:a + 8], p2, out[a:a + 8])
        else:
            getattr(e, fg)(qb, p2, out)
    return f


def _owners():
    out = []
    for r in _OWNERS:
        o = r()
        if o is not None and type(o).__name__ != "GroupedView":
            out.append(o)
    return out


def prepare(kinds, alt_dir=None):
    """load the requantised bytes of every eligible owner to its device; returns (n_owners, n_prepared, {kind: bytes})"""
    alt_dir = alt_dir or ALT_DIR
    n_all = n_ok = 0
    tot = {k: 0 for k in kinds}
    seen = set()
    for o in _owners():
        if id(o) in seen:
            continue
        seen.add(id(o))
        n_all += 1
        names = list(getattr(o, "gnames", None) or ([o.gname] if getattr(o, "gname", None) else []))
        if not names or o.ty != 8 or "alt" in o.__dict__:
            continue
        alts = {}
        for k in kinds:
            parts = []
            for n in names:
                p = os.path.join(alt_dir, k, n + ".bin")
                if not os.path.exists(p):
                    parts = None
                    break
                parts.append(np.fromfile(p, dtype=np.uint8))
            if parts is None:
                alts = None
                break
            K = o.shape[1]
            arr = np.concatenate(parts).reshape(-1, K // 256, KQ[k]["bb"])
            if arr.shape[0] != o.packed.shape[0]:
                alts = None
                break
            alts[k] = torch.from_numpy(arr).to(o.packed.device)
            tot[k] += arr.nbytes
        if alts:
            o.alt = alts
            o.alt_q8 = (o.packed, o.ty, o.spec, o.gemv)
            n_ok += 1
    return n_all, n_ok, tot


def _apply(o, packed, ty, spec, gemv):
    o.packed, o.ty, o.spec, o.gemv = packed, ty, spec, gemv
    o.shape = (packed.shape[0], packed.shape[1] * spec[0])
    if hasattr(o, "members"):          # an attn_grouped._Group: re-point the member views and drop the cached outputs
        o.ref = None
        for m, off, n in zip(o.members, o.offs, o.sizes):
            m.packed, m.ty, m.spec, m.gemv = packed[off:off + n], ty, spec, gemv
            m.shape = (n, o.shape[1])


def set_mode(mode, kern="gemv"):
    """switch every prepared owner to `mode`; returns the number switched"""
    n = 0
    for o in _owners():
        if "alt" not in o.__dict__:
            continue
        if mode == "q8":
            _apply(o, *o.alt_q8)
        else:
            _apply(o, o.alt[mode], {"q6_k": 14, "q5_k": 13}[mode], (256, KQ[mode]["bb"], "fq_gemv_" + mode, _dq_fn(mode)), _gemv(mode, kern))
        n += 1
    return n


def resident_bytes(mode):
    """dense weight bytes the decode step reads for the current mode (prepared owners only)"""
    return sum(o.packed.numel() for o in _owners() if "alt" in o.__dict__)
