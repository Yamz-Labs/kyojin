"""ctypes binding of libgkv.so (hipsp/gkv.cu): global-attention decode kernels (WMMA split-K attend, combine with sink and gate, rel table, in-place KV append)."""
import collections
import ctypes

HD = 128      # head dim


class Lib:
    def __init__(self, so):
        self.so = so
        self.l = ctypes.CDLL(so)
        P, I, L, F = ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.c_float
        self.l.gkv_attend.argtypes = [P] * 8 + [I, I, I, L, F, I, P]
        self.l.gkv_combine.argtypes = [P] * 5 + [I, I, I, I, I, P]
        self.l.gkv_rel.argtypes = [P, P, P, I, P]
        self.l.gkv_append.argtypes = [P] * 6 + [P, I, L, I, I, P]
        self.l.gkv_deq_test.argtypes = [P, P, I, I, P]
        self.l.gkv_ws_floats.restype = ctypes.c_size_t
        self.l.gkv_ws_floats.argtypes = [I]
        self.cnt = collections.Counter()
        self.dq = 1

    @staticmethod
    def stream():
        import torch
        return torch.cuda.current_stream().cuda_stream

    def chk(self, r, name):
        if r != 0:
            raise RuntimeError(f"{name} rc={r}")

    def attend(self, q, K, V, ks, vs, pos0, rel, ws, T, smax, cmin, cap, q8):
        self.cnt["attend_q8" if q8 else "attend_f16"] += 1
        qm = 0 if not q8 else self.dq
        self.chk(self.l.gkv_attend(q.data_ptr(), K, V, ks, vs, pos0.data_ptr(), rel.data_ptr() if rel is not None else None, ws.data_ptr(), T, smax, cmin, cap,
                                   HD ** -0.5, qm, self.stream()), "gkv_attend")

    def combine(self, ws, sink, gate, out, pos0, T, smax, cmin, out_bf16=0, gate_bf16=0):
        self.cnt["combine"] += 1
        self.chk(self.l.gkv_combine(ws.data_ptr(), sink.data_ptr() if sink is not None else None, gate.data_ptr() if gate is not None else None, out.data_ptr(), pos0.data_ptr(),
                                    T, smax, cmin, out_bf16, gate_bf16, self.stream()), "gkv_combine")

    def rel(self, r, proj, out, T):
        self.cnt["rel"] += 1
        self.chk(self.l.gkv_rel(r.data_ptr(), proj.data_ptr(), out.data_ptr(), T, self.stream()), "gkv_rel")

    def append(self, K, V, ks, vs, kn, vn, pos0, T, cap, q8, bf16=0):
        self.cnt["append_q8" if q8 else "append_f16"] += 1
        self.chk(self.l.gkv_append(K, V, ks, vs, kn.data_ptr(), vn.data_ptr(), pos0.data_ptr(), T, cap, 1 if q8 else 0, bf16, self.stream()), "gkv_append")
