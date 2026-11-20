"""ctypes wrapper of sp_mla.cu: sparse MLA prefill (one workgroup per query x 64 heads)."""
import ctypes, os, torch
_lib = None
def lib():
    global _lib
    if _lib is None:
        so = os.environ.get("SP_SO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "libspmla.so"))
        _lib = ctypes.CDLL(so); f = _lib.sp_launch; f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 5 + [ctypes.c_float] + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    return _lib

def mla_sparse_hip(R, kc16, qstart, scale, sinks, sel, qcat, fixed_tiles=0, out=None):
    """same contract as prefill_attn.mla_sparse but kc16 must already be fp16 contiguous [N, 576]: qcat fp16 [H, T, 576] -> o [H, T, R] fp16 (saturated)"""
    H, T, D = qcat.shape
    N = kc16.shape[0]
    assert R == 512 and D == 576 and kc16.shape[1] == 576 and kc16.dtype == torch.float16 and kc16.is_contiguous() and qcat.is_contiguous() and qcat.dtype == torch.float16
    assert N == qstart + T and sel.gidx.dtype == torch.int32 and sel.nsel.dtype == torch.int32 and sel.gidx.is_contiguous()
    o = out if out is not None else torch.empty(H, T, R, dtype=torch.float16, device=qcat.device)
    sk = sinks.float().contiguous() if sinks is not None else None
    r = lib().sp_launch(qcat.data_ptr(), kc16.data_ptr(), sk.data_ptr() if sk is not None else None, sel.gidx.data_ptr(), sel.nsel.data_ptr(), o.data_ptr(),
                        T, qstart, H, sel.gidx.shape[1], N, float(scale), int(bool(sel.tail)), int(sel.kpool), int(fixed_tiles), torch.cuda.current_stream().cuda_stream)
    assert r == 0, r
    return o
