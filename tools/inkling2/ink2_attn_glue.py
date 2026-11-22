"""ctypes glue of ink2_attn_cuda.cu (RTX 3090 sm_86): gkv global decode + lattn local ring attention for Inkling 2 Mini, T = 1..8 rows, wired into the eager step.

One AttnKV per application: K [8, cap, 128] fp16 row-major, V [8, cap/16, 128, 16] dim-major key tiles (header of the .cu); local slots are rings (cap = R = 1088).
step(): rel-bias table kernel + in-place append + attend with the fused combine (sink in the denominator, gate sigmoid, fp16 out) = 3 launches per application."""
import ctypes
import os

import torch

NKV, NH, HD, NREL, R_LOCAL = 8, 24, 128, 1024, 1088
SCALE = HD ** -0.5
GKV_SMAX, GKV_CMIN, LOC_SMAX, LOC_CMIN = (int(v) for v in os.environ.get("INK_ATTN_CFG", "16:64:8:16").split(":"))


class Ink2AttnCuda:
    def __init__(self, so=None):
        so = so or os.environ.get("INK2_ATTN_SO") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libink2_attn_cuda.so")
        self.l = ctypes.CDLL(so)
        P, I, L, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.c_float
        self.l.ink2attn_attend.argtypes = [P] * 6 + [I, I, I, L, F, I, I, I, P, P, P, P, I, I, P]
        self.l.ink2attn_rel.argtypes = [P, P, P, I, P]
        self.l.ink2attn_append.argtypes = [P] * 5 + [I, L, I, I, I, P]
        self.l.ink2attn_fires.argtypes = [P]
        self.l.ink2attn_ws_floats.restype = ctypes.c_size_t
        self.l.ink2attn_ws_floats.argtypes = [I]
        self.ws = torch.empty(self.l.ink2attn_ws_floats(256), dtype=torch.float32, device="cuda")
        self.cnt = torch.zeros(8, dtype=torch.int32, device="cuda")
        self.pos0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.rel_t = {}

    @staticmethod
    def _st():
        return torch.cuda.current_stream().cuda_stream

    @staticmethod
    def _chk(r, name):
        if r != 0:
            raise RuntimeError(f"{name} rc={r}")

    def new_cache(self, local, cap, dev):
        cap = R_LOCAL if local else (cap + 15) // 16 * 16
        return (torch.zeros(NKV, cap, HD, dtype=torch.float16, device=dev), torch.zeros(NKV, cap // 16, HD, 16, dtype=torch.float16, device=dev), cap)

    def set_pos0(self, p):
        self.pos0.fill_(int(p))

    def append(self, cache, kn, vn, T, local):
        """kn / vn [T, NKV, HD] fp32 contiguous; pos0 already set"""
        K, V, cap = cache
        self._chk(self.l.ink2attn_append(K.data_ptr(), V.data_ptr(), kn.data_ptr(), vn.data_ptr(), self.pos0.data_ptr(), T, cap, int(local), 2, 0, self._st()), "append")

    def step(self, cache, q, kn, vn, r, proj_t, sink, gate, T, local):
        """q [T, NH, HD] fp16 (clipped, tau-scaled), kn / vn [T, NKV, HD] fp32, r [T, NH, 16] fp32, proj_t [16, 1024] fp32 (rel_bias^T), sink [T, NH] fp32,
        gate [T, NH * HD] fp32 (raw logits; sigmoid applied in the kernel) -> out [T, NH * HD] fp32. pos0 = absolute position of the first new token (set_pos0)."""
        K, V, cap = cache
        rel = torch.empty(T, NH, NREL, dtype=torch.float32, device=q.device)
        self._chk(self.l.ink2attn_rel(r.data_ptr(), proj_t.data_ptr(), rel.data_ptr(), T, self._st()), "rel")
        self.append(cache, kn, vn, T, local)
        out = torch.empty(T, NH, HD, dtype=torch.float16, device=q.device)
        smax, cmin = (LOC_SMAX, LOC_CMIN) if local else (GKV_SMAX, GKV_CMIN)
        self._chk(self.l.ink2attn_attend(q.data_ptr(), K.data_ptr(), V.data_ptr(), self.pos0.data_ptr(), rel.data_ptr(), self.ws.data_ptr(), T, smax, cmin, cap, SCALE,
                                         int(local), 0, cap, self.cnt.data_ptr(), sink.data_ptr(), gate.data_ptr(), out.data_ptr(), 0, 2, self._st()), "attend")
        return out.float().reshape(T, NH * HD)

    def seed(self, cache, k, v, start, local):
        """k / v [NKV, n, HD] natural layout (synthetic KV for the decode bench) at absolute positions start .. start + n - 1"""
        kn = k.permute(1, 0, 2).float().contiguous()
        vn = v.permute(1, 0, 2).float().contiguous()
        self.set_pos0(start)
        self.append(cache, kn, vn, kn.shape[0], local)

    def fires(self):
        """device counters: [gkv attend blocks, lattn attend blocks, combine blocks, append blocks, rel blocks]"""
        a = (ctypes.c_uint * 8)()
        torch.cuda.synchronize()
        self._chk(self.l.ink2attn_fires(ctypes.cast(a, ctypes.c_void_p)), "fires")
        return list(a)[:5]
