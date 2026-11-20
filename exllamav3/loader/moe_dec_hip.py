"""Decode MoE stages (q8_1 activations) for GGUF k-quants on gfx1151: HIP code object moe_dec.co (built by hip_kq/build_moe.sh from moe_dec.cu).
Same arithmetic and accumulation order as fq_moe_kq.cu, so every arm is bit-equal to the shipped extension kernels. Arms (per stage):
  ug ('ref' = copy of the shipped kernel, 'v3' = unrolled loads + LDS activations / Q3_K one thread per 256-block), dn ('ref', 'v3').
Used by tools/hy4/step/opbench.py `experts_v2` and (flag HY4_MOE_V2, default off) by hy4_forward's routed-expert call."""
import ctypes, os
import torch
from . import kq_moe_hip as _K

CO = os.environ.get("MOE_DEC_CO", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exllamav3_ext", "hip_kq", "moe_dec.co"))


def _launch(name, grid, block, args, shmem=0):
    lib = _K._hip()
    fn = _K._fn(CO, name)
    cargs = []
    for a in args:
        if a is None:
            cargs.append(ctypes.c_void_p(0))
        elif isinstance(a, torch.Tensor):
            cargs.append(ctypes.c_void_p(a.data_ptr()))
        elif isinstance(a, float):
            cargs.append(ctypes.c_float(a))
        elif isinstance(a, tuple):
            cargs.append(a[1](a[0]))
        else:
            cargs.append(ctypes.c_int(a))
    ptrs = (ctypes.c_void_p * len(cargs))(*[ctypes.cast(ctypes.pointer(c), ctypes.c_void_p) for c in cargs])
    rc = lib.hipModuleLaunchKernel(fn, grid, 1, 1, block, 1, 1, shmem, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), ptrs, None)
    assert rc == 0, f"hipModuleLaunchKernel {name} rc={rc}"


def _i64(v):
    return (int(v), ctypes.c_long)


class MoeDec:
    """scratch + launches for one layer shape: R rows, topk, K (hidden, = down rows), I (ff), gate/up fmt 'q4_k' | 'q3_k', down Q4_K"""

    def __init__(self, dev, R, topk, K, I, fmt_ug, E, sort=True):
        assert K % 256 == 0 and I % 256 == 0 and K % 32 == 0 and I % 32 == 0 and fmt_ug in ("q4_k", "q3_k")
        self.R, self.topk, self.K, self.I, self.fmt, self.E = R, topk, K, I, fmt_ug, E
        self.S = R * topk
        self.sort = sort and R > 1 and self.S <= 128
        self.qbh = torch.empty(R, K // 32 * 36, dtype=torch.uint8, device=dev)
        self.act = torch.empty(self.S, I, dtype=torch.float16, device=dev)
        self.qba = torch.empty(self.S, I // 32 * 36, dtype=torch.uint8, device=dev)
        self.partial = torch.empty(self.S, K, dtype=torch.float32, device=dev)
        self.perm = torch.empty(self.S, dtype=torch.int32, device=dev) if self.sort else None
        self.out = torch.empty(R, K, dtype=torch.float32, device=dev)

    def prep(self, h, sel):
        if self.sort:
            _launch("moe_sort", 1, 128, [sel, self.perm, self.S, self.E])
        _launch("moe_quant_q8_1", self.R * (self.K // 32), 32, [h, self.qbh, self.R, self.K])

    def ug(self, arm, sel, ptrs, strides, lim):
        S, I, K, E, topk = self.S, self.I, self.K, self.E, self.topk
        gp, up_ = ptrs[0], ptrs[1]
        qs = self.qbh.shape[1]
        if arm == "ref":
            if self.fmt == "q4_k":
                nt = I // 32
                _launch("ug_q4_ref", S * nt, 256, [self.qbh, sel, gp, up_, _i64(strides[0]), _i64(strides[1]), _i64(qs), self.act, S, I, K, E, nt, topk, float(lim), self.perm, S])
            else:
                nt = I // 8
                _launch("ug_q3_ref", S * nt, 256, [self.qbh, sel, gp, up_, _i64(strides[0]), _i64(strides[1]), _i64(qs), self.act, S, I, K, E, nt, topk, float(lim), self.perm, S], 2048)
        else:
            nb, nt = K // 256, I // 32
            if self.fmt == "q4_k":
                _launch(f"ug_q4_v3_{nb}", S * nt, 256, [self.qbh, sel, gp, up_, _i64(strides[0]), _i64(strides[1]), _i64(qs), self.act, S, I, E, nt, topk, float(lim), self.perm, S])
            else:
                shm = (2 * 32 * (nb * 16 + 1) + 2 * 32 * nb + nb * 72) * 4
                _launch(f"ug_q3_v3_{nb}", S * nt, 32 * nb, [self.qbh, sel, gp, up_, _i64(strides[0]), _i64(strides[1]), _i64(qs), self.act, S, I, E, nt, topk, float(lim), self.perm, S], shm)

    def qact(self):
        _launch("moe_quant_q8_1", self.S * (self.I // 32), 32, [self.act, self.qba, self.S, self.I])

    def dn(self, arm, sel, ptrs, strides):
        S, I, K, E = self.S, self.I, self.K, self.E
        nt = K // 32
        qs = self.qba.shape[1]
        if arm == "ref":
            _launch("dn_q4_ref", S * nt, 256, [self.qba, _i64(qs), sel, ptrs[2], _i64(strides[2]), self.partial, S, I, K, E, nt, self.perm, S])
        else:
            _launch(f"dn_q4_v3_{I // 256}", S * nt, 256, [self.qba, _i64(qs), sel, ptrs[2], _i64(strides[2]), self.partial, S, K, E, nt, self.perm, S])

    def combine(self, wts):
        nb = (self.K + 255) // 256
        lib = _K._hip()
        fn = _K._fn(CO, "moe_combine")
        cargs = [ctypes.c_void_p(self.partial.data_ptr()), ctypes.c_void_p(wts.data_ptr()), ctypes.c_void_p(self.out.data_ptr()), ctypes.c_int(self.topk), ctypes.c_int(self.K)]
        pp = (ctypes.c_void_p * 5)(*[ctypes.cast(ctypes.pointer(c), ctypes.c_void_p) for c in cargs])
        rc = lib.hipModuleLaunchKernel(fn, nb, self.R, 1, 256, 1, 1, 0, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), pp, None)
        assert rc == 0

    def run(self, h, sel, wts, ptrs, strides, lim, ug="v3", dn="v3"):
        self.prep(h, sel)
        self.ug(ug, sel, ptrs, strides, lim)
        self.qact()
        self.dn(dn, sel, ptrs, strides)
        self.combine(wts)
        return self.out
