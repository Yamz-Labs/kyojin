"""ctypes wrapper of ink2_experts.hip: grouped routed experts (NVFP4 type 40 as stored), T = 1..8 rows, device-side expert union, no host sync.

  Ink2Experts(so).moe_nvfp4(x, sel, wt, gu, dn, s_gu, s_dn, ...) -> out [T, 2304] fp32
x [T, 2304] fp32, sel [T, 8] int32, wt [T, 8] fp32 (router weights, already renormalised x route_scale), gu [E, 2304, 1296] / dn [E, 2304, 648] uint8
(the GGUF bytes of ffn_gate_up_exps / ffn_down_exps of one layer, untouched), s_gu / s_dn [E] fp32 or None."""
import ctypes
import os

import torch

D, FF, TOPK = 2304, 1152, 8


class Ink2Experts:
    def __init__(self, so=None):
        so = so or os.environ.get("INK2_SO") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libink2_experts.so")
        self.lib = ctypes.CDLL(so)
        P, I, F, V = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_void_p
        self.lib.ink2_plan_launch.argtypes = [P, P, I, P, V]
        self.lib.ink2_gu_launch.argtypes = [I, I, P, P, P, P, P, F, F, V]
        self.lib.ink2_dn_launch.argtypes = [I, I, P, P, P, P, P, V]
        self.lib.ink2_gu_q8_launch.argtypes = [I, I, P, P, P, P, P, F, F, V]
        self.lib.ink2_dn_q8_launch.argtypes = [I, I, P, P, P, P, P, V]
        self.lib.ink2_readbw_launch.argtypes = [P, P, ctypes.c_long, ctypes.c_long, P, I, P, V]
        self.plan_bytes = self.lib.ink2_plan_bytes()
        self._plans = {}

    def alloc(self, T, dev):
        d = torch.device(dev)
        key = (T, d.type, d.index if d.index is not None else torch.cuda.current_device())
        if key not in self._plans:
            self._plans[key] = (torch.zeros(self.plan_bytes, dtype=torch.uint8, device=dev), torch.empty(8 * T * T * FF, dtype=torch.float32, device=dev))
        return self._plans[key]

    @staticmethod
    def _p(t):
        return None if t is None else t.data_ptr()

    def plan(self, sel, wt, plan):
        T = sel.shape[0]
        st = torch.cuda.current_stream().cuda_stream
        assert self.lib.ink2_plan_launch(sel.data_ptr(), wt.data_ptr(), T, plan.data_ptr(), st) == 0

    def moe_nvfp4(self, x, sel, wt, gu, dn, s_gu, s_dn, beta, lb, out=None, cfg_gu=0, cfg_dn=0, parts=3, q8=False):
        """parts bit 0 = plan + gate_up, bit 1 = down (ablation); returns out (and leaves h in the scratch: alloc(T)[1])"""
        T = x.shape[0]
        plan, hs = self.alloc(T, x.device)
        if out is None:
            out = torch.empty(T, D, dtype=torch.float32, device=x.device)
        st = torch.cuda.current_stream().cuda_stream
        if parts & 1:
            self.plan(sel, wt, plan)
            assert (self.lib.ink2_gu_q8_launch if q8 else self.lib.ink2_gu_launch)(T, cfg_gu, x.data_ptr(), plan.data_ptr(), gu.data_ptr(), self._p(s_gu), hs.data_ptr(), beta, lb, st) == 0
        if parts & 2:
            assert (self.lib.ink2_dn_q8_launch if q8 else self.lib.ink2_dn_launch)(T, cfg_dn, plan.data_ptr(), hs.data_ptr(), dn.data_ptr(), self._p(s_dn), out.data_ptr(), st) == 0
        return out

    def read_plan(self, plan):
        import numpy as np
        b = plan.cpu().numpy().tobytes()
        nu = int(np.frombuffer(b, np.int32, 1, 0)[0])
        eid = np.frombuffer(b, np.int32, 64, 8).copy()
        mask = np.frombuffer(b, np.uint32, 64, 8 + 256).copy()
        w = np.frombuffer(b, np.float32, 512, 8 + 512).reshape(64, 8).copy()
        return nu, eid, mask, w

    def readbw(self, gu, dn, ids, sink):
        """read the gate_up + down bytes of the experts `ids` (int32 device tensor) once: the measured read floor of the expert access pattern"""
        st = torch.cuda.current_stream().cuda_stream
        gb, db = gu[0].numel() * gu.element_size(), dn[0].numel() * dn.element_size()
        assert self.lib.ink2_readbw_launch(gu.data_ptr(), dn.data_ptr(), gb, db, ids.data_ptr(), ids.numel(), sink.data_ptr(), st) == 0
