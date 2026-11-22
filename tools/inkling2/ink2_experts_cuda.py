"""CUDA (sm_86) variant of the grouped experts wrapper: same C ABI / Plan / h scratch as ink2_experts.py, plus the path counters of the kernels (fires-check)."""
import ctypes
import os

import numpy as np
import torch

from ink2_experts import Ink2Experts


class Ink2ExpertsCuda(Ink2Experts):
    def __init__(self, so=None):
        so = so or os.environ.get("INK2_SO") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libink2_experts_cuda.so")
        super().__init__(so)
        self.lib.ink2_fires_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        P, I, V = ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
        self.lib.ink2_plan_x_launch.argtypes = [P, P, I, P, P, I, V]
        self.lib.ink2_hprep_launch.argtypes = [I, I, P, P, V]
        self._fb = None

    def fires(self):
        """(gate_up entries launched, down blocks launched) counted on the device since load; call outside graph capture"""
        if self._fb is None:
            self._fb = torch.zeros(4, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        assert self.lib.ink2_fires_launch(self._fb.data_ptr(), torch.cuda.current_stream().cuda_stream) == 0
        torch.cuda.synchronize()
        a = self._fb.cpu().numpy().astype(np.uint32)
        return int(a[0]), int(a[1])

    def moe_nvfp4(self, x, sel, wt, gu, dn, s_gu, s_dn, beta, lb, out=None, **kw):
        assert gu.data_ptr() % 16 == 0 and dn.data_ptr() % 16 == 0 and x.data_ptr() % 16 == 0, "16-byte aligned weights / x required"
        return self._moe(x, sel, wt, gu, dn, s_gu, s_dn, beta, lb, out=out, **kw)

    def _moe(self, x, sel, wt, gu, dn, s_gu, s_dn, beta, lb, out=None, cfg_gu=0, cfg_dn=0, parts=3, q8=False):
        """cfg >= 16 = the v2 cp.async kernels: plan + x pre-conversion in ONE launch (ink2_plan_x), the gate_up epilogue also writes the fp16 B fragments of h, so the down kernel needs no prep launch;
        a down-only call (parts == 2) or a v1 gate_up in front of a v2 down converts the fp32 h scratch first (ink2_hprep)."""
        T = x.shape[0]
        plan, hs = self.alloc(T, x.device)
        if out is None:
            out = torch.empty(T, 2304, dtype=torch.float32, device=x.device)
        st = torch.cuda.current_stream().cuda_stream
        if parts & 1:
            if cfg_gu >= 16:
                assert self.lib.ink2_plan_x_launch(sel.data_ptr(), wt.data_ptr(), T, plan.data_ptr(), x.data_ptr(), 2 if q8 else 1, st) == 0
            else:
                self.plan(sel, wt, plan)
            assert (self.lib.ink2_gu_q8_launch if q8 else self.lib.ink2_gu_launch)(T, cfg_gu, x.data_ptr(), plan.data_ptr(), gu.data_ptr(), self._p(s_gu), hs.data_ptr(), beta, lb, st) == 0
        if parts & 2:
            if cfg_dn >= 16 and not ((parts & 1) and cfg_gu >= 16):
                assert self.lib.ink2_hprep_launch(T, 2 if q8 else 1, plan.data_ptr(), hs.data_ptr(), st) == 0
            assert (self.lib.ink2_dn_q8_launch if q8 else self.lib.ink2_dn_launch)(T, cfg_dn, plan.data_ptr(), hs.data_ptr(), dn.data_ptr(), self._p(s_dn), out.data_ptr(), st) == 0
        return out
