"""ctypes wrapper of ink2_routerhead.hip: router (+ expert union plan) and LM head (Q8_0 / BF16) with softcap + argmax epilogue, T = 1..8 rows, no host sync, graph capturable.

  RH = Ink2RouterHead(so)
  RH.router(x [T, 2304] f32, R [128, 2304] f32, bias [128] f32, route_scale=8.0, cfg=0) -> sel [T, 8] int32, wt [T, 8] f32, plan (uint8 buffer = Plan of ink2_experts.hip)
  RH.head_q8(x, W [N_rows, 2448] uint8 (GGUF Q8_0 bytes untouched), N, softcap=30, logits=True, cand=None, cfg=0) -> (logits [T, N] f32 | None, idx [T] int32, val [T] f32)
  RH.head_bf16(x, W [N_rows, 2304] bf16 viewed as uint8 / bf16, ...) same
Router order inside a row = selection order (descending selection score), NOT torch.topk(sorted=False) order: compare picks as sets."""
import ctypes
import os

import torch

D, E, TOPK = 2304, 128, 8


class Ink2RouterHead:
    def __init__(self, so=None):
        so = so or os.environ.get("INK2_RH_SO") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libink2_rh.so")
        self.lib = L = ctypes.CDLL(so)
        P, I, F, V = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_void_p
        L.ink2_router_launch.argtypes = [I, I, P, P, P, P, P, P, P, F, V]
        L.ink2_head_q8_launch.argtypes = [I, I, P, P, I, P, F, P, P, P, P, V]
        L.ink2_head_bf16_launch.argtypes = [I, I, P, P, I, P, F, P, P, P, P, V]
        L.ink2_stream_read_launch.argtypes = [P, ctypes.c_long, I, P, V]
        self.plan_bytes, self.rs_bytes, self.hs_bytes = L.ink2_plan_bytes(), L.ink2_router_scratch_bytes(), L.ink2_head_scratch_bytes()
        self._bufs = {}
        self.fires = {"router": 0, "head_q8": 0, "head_bf16": 0, "stream": 0}

    def _scratch(self, dev):
        k = str(dev)
        if k not in self._bufs:
            self._bufs[k] = (torch.zeros(self.rs_bytes, dtype=torch.uint8, device=dev), torch.zeros(self.hs_bytes, dtype=torch.uint8, device=dev),
                             torch.zeros(self.plan_bytes, dtype=torch.uint8, device=dev))
        return self._bufs[k]

    @staticmethod
    def stream():
        return torch.cuda.current_stream().cuda_stream

    def router(self, x, R, bias, route_scale=8.0, cfg=0, sel=None, wt=None):
        T = x.shape[0]
        rs, _, plan = self._scratch(x.device)
        sel = torch.empty(T, TOPK, dtype=torch.int32, device=x.device) if sel is None else sel
        wt = torch.empty(T, TOPK, dtype=torch.float32, device=x.device) if wt is None else wt
        assert x.dtype == torch.float32 and x.is_contiguous() and R.is_contiguous() and R.dtype == torch.float32 and bias.dtype == torch.float32
        assert self.lib.ink2_router_launch(T, cfg, x.data_ptr(), R.data_ptr(), bias.data_ptr(), sel.data_ptr(), wt.data_ptr(), plan.data_ptr(), rs.data_ptr(), float(route_scale), self.stream()) == 0
        self.fires["router"] += 1
        return sel, wt, plan

    def _head(self, fn, key, x, W, N, softcap, logits, cand, cfg, out):
        T = x.shape[0]
        _, hs, _ = self._scratch(x.device)
        if logits is True:
            logits = torch.empty(T, N, dtype=torch.float32, device=x.device)
        idx = torch.empty(8, dtype=torch.int32, device=x.device) if out is None else out[0]
        val = torch.empty(8, dtype=torch.float32, device=x.device) if out is None else out[1]
        assert x.dtype == torch.float32 and x.is_contiguous()
        rc = fn(T, cfg, x.data_ptr(), W.data_ptr(), N, None if cand is None else cand.data_ptr(), float(softcap), None if not isinstance(logits, torch.Tensor) else logits.data_ptr(),
                hs.data_ptr(), idx.data_ptr(), val.data_ptr(), self.stream())
        if rc == 1:
            raise NotImplementedError(f"{key} cfg {cfg} T {T}: no such config (or LDS over 64 KB)")
        assert rc == 0
        self.fires[key] += 1
        return (logits if isinstance(logits, torch.Tensor) else None), idx[:T], val[:T]

    def head_q8(self, x, W, N, softcap=30.0, logits=True, cand=None, cfg=0, out=None):
        return self._head(self.lib.ink2_head_q8_launch, "head_q8", x, W, N, softcap, logits, cand, cfg, out)

    def head_bf16(self, x, W, N, softcap=30.0, logits=True, cand=None, cfg=0, out=None):
        return self._head(self.lib.ink2_head_bf16_launch, "head_bf16", x, W, N, softcap, logits, cand, cfg, out)

    def stream_read(self, buf, grid, sink):
        assert self.lib.ink2_stream_read_launch(buf.data_ptr(), buf.numel() * buf.element_size(), grid, sink.data_ptr(), self.stream()) == 0
        self.fires["stream"] += 1

    @staticmethod
    def read_plan(plan):
        import numpy as np
        b = plan.cpu().numpy().tobytes()
        nu = int(np.frombuffer(b, np.int32, 1, 0)[0])
        eid = np.frombuffer(b, np.int32, 64, 8).copy()
        mask = np.frombuffer(b, np.uint32, 64, 8 + 256).copy()
        w = np.frombuffer(b, np.float32, 512, 8 + 512).reshape(64, 8).copy()
        return nu, eid, mask, w
