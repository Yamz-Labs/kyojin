"""ctypes glue of ink2_router_cuda.cu: sigmoid router + top-8 + renorm for T <= 8 rows in ONE launch (fp32 router weights [E, D], optional selection bias [E])."""
import ctypes
import os

import torch


class Ink2RouterCuda:
    def __init__(self, so=None):
        so = so or os.environ.get("INK2_ROUTER_SO") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libink2_router_cuda.so")
        self.l = ctypes.CDLL(so)
        P, I, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
        self.l.ink2_router_launch.argtypes = [P, P, P, I, I, I, F, P, P, P]
        self.l.ink2_router_fires.argtypes = [P]
        self._bufs = {}

    def route(self, x, R, bias, wscale):
        """x [T, D] fp32 contiguous, R [E, D] fp32 contiguous, bias [E] fp32 or None -> (sel int32 [T, 8], wt fp32 [T, 8])"""
        T, D = x.shape
        E = R.shape[0]
        b = self._bufs.get(T)
        if b is None:
            b = self._bufs[T] = (torch.empty(T, 8, dtype=torch.int32, device=x.device), torch.empty(T, 8, dtype=torch.float32, device=x.device))
        sel, wt = b
        sel, wt = torch.empty_like(sel), torch.empty_like(wt)         # fresh outputs: the experts plan kernel reads them later (graph-safe, tiny)
        rc = self.l.ink2_router_launch(x.data_ptr(), R.data_ptr(), bias.data_ptr() if bias is not None else None, T, E, D, float(wscale), sel.data_ptr(), wt.data_ptr(),
                                       torch.cuda.current_stream().cuda_stream)
        assert rc == 0, f"router rc={rc}"
        return sel, wt

    def fires(self):
        a = (ctypes.c_uint * 2)()
        torch.cuda.synchronize()
        assert self.l.ink2_router_fires(ctypes.cast(a, ctypes.c_void_p)) == 0
        return int(a[0])
