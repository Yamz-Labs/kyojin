"""CK fmha_fwd (hd 256, fp16, batch 1) behind ckmla_fwd, for un-absorbed MLA / any MHA prefill with strided operands.
fmha(q, k, v, scale, sink=None, bias=None, window=0): q [T, H, D], k [N, H, D], v [N, H, Dv] (any strides, last dim contiguous, row / head strides
multiples of 8), causal bottom-right (query t sees keys <= N - T + t), window > 0: only the last `window` keys; sink [H] fp32 (learnable softmax sink, in
scaled-logit space: it joins the denominator only); bias [T, N] fp16 elementwise (row stride multiple of 8) added to the scaled logits.
Returns o [T, H, Dv] fp16, or None when the library is missing / declines (caller keeps its fallback). Library: EXL3_CKMLA_SO or
~/.cache/kyojin/ck-fmha/libckmla.so (tools/strix_halo/ck_fmha/build_mla.sh)."""
import ctypes, os
from pathlib import Path
import torch

_fn = None
_tried = False


def _load():
    global _fn, _tried
    if _tried:
        return _fn
    _tried = True
    so = Path(os.environ.get("EXL3_CKMLA_SO") or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "kyojin" / "ck-fmha" / "libckmla.so")
    try:
        f = ctypes.CDLL(str(so)).ckmla_fwd
        f.restype = ctypes.c_float
        f.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 6 + [ctypes.c_float] + [ctypes.c_long] * 9 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        _fn = f
    except OSError:
        _fn = None
    return _fn


def available():
    return _load() is not None


def fmha(q, k, v, scale, sink=None, bias=None, window=0, out=None, time_it=False):
    f = _load()
    if f is None:
        return None
    T, H, D = q.shape
    N, Hk, Dk = k.shape
    Dv = v.shape[2]
    if D != 256 or Dv != 256 or Hk != H or q.dtype != torch.float16 or q.stride(2) != 1 or k.stride(2) != 1 or v.stride(2) != 1:
        return None
    if any(s % 8 for s in (q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1))):
        return None
    o = out if out is not None else torch.empty(T, H, Dv, dtype=torch.float16, device=q.device)
    if bias is not None and (bias.dtype != torch.float16 or bias.stride(1) != 1 or bias.stride(0) % 8):
        return None
    sk = sink.float().contiguous() if sink is not None else None
    stream = torch.cuda.current_stream(q.device).cuda_stream
    r = f(ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(o.data_ptr()),
          ctypes.c_void_p(bias.data_ptr() if bias is not None else 0), ctypes.c_void_p(sk.data_ptr() if sk is not None else 0),
          T, N, H, H, D, Dv, float(scale), q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), o.stride(0), o.stride(1),
          bias.stride(0) if bias is not None else 0, int(window - 1) if window else 0, 1, ctypes.c_void_p(stream), 1 if time_it else 0)
    if r < 0:
        return None
    return (o, r) if time_it else o
