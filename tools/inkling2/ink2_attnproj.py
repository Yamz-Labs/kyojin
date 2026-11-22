"""ctypes wrapper of ink2_attnproj.hip (fused attention projections, T = 1..8) + a numpy emulation of the kernel's index / decode math (CPU selftest).

  ap = Ink2AttnProj(so)                        # libink2_attnproj_w4.so / _w8.so
  segs = [Seg(w, sc, rows, K, xi, fmt, gs, out=None), ...]
  ap.run(segs, xs, T)                          # xs = [xq, xk, xv] fp32 [T, K] tensors (device); fills seg.out [T, rows] fp32
Formats: 0 = Q8 (w int8 [N,K], sc fp16 [N,K/32]), 1 = NV (w u8 [N,K/2], sc u8 [N,K/16], gs), 2 = BF (w bf16 [N,K], sc None)."""
import ctypes
import os

import numpy as np
import torch

F_Q8, F_NV, F_BF = 0, 1, 2
MAXSEG = 12
G_OF = {F_Q8: 16, F_NV: 8, F_BF: 32}
UE_OF = {F_Q8: 16, F_NV: 32, F_BF: 8}


class CSeg(ctypes.Structure):
    _fields_ = [("w", ctypes.c_void_p), ("sc", ctypes.c_void_p), ("out", ctypes.c_void_p), ("rows", ctypes.c_int), ("K", ctypes.c_int), ("ldo", ctypes.c_int),
                ("xi", ctypes.c_int), ("fmt", ctypes.c_int), ("blk0", ctypes.c_int), ("nblk", ctypes.c_int), ("gs", ctypes.c_float)]


class CParams(ctypes.Structure):
    _fields_ = [("s", CSeg * MAXSEG), ("x", ctypes.c_void_p * 3), ("nseg", ctypes.c_int), ("T", ctypes.c_int), ("total_blocks", ctypes.c_int), ("pad", ctypes.c_int)]


class Seg:
    def __init__(self, w, sc, rows, K, xi, fmt, gs=1.0, name=""):
        self.w, self.sc, self.rows, self.K, self.xi, self.fmt, self.gs, self.name = w, sc, rows, K, xi, fmt, gs, name
        self.out = None
        self.buf = None

    def nbytes(self):
        n = self.w.numel() * self.w.element_size()
        return n + (0 if self.sc is None else self.sc.numel() * self.sc.element_size())


class Ink2AttnProj:
    def __init__(self, so):
        self.lib = ctypes.CDLL(so)
        self.wpb = self.lib.ink2_ap_wpb()
        self.stagefirst = int(self.lib.ink2_ap_stagefirst())
        assert self.lib.ink2_ap_sizeof_params() == ctypes.sizeof(CParams), (self.lib.ink2_ap_sizeof_params(), ctypes.sizeof(CParams))
        self.lib.ink2_ap_launch.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.ink2_ap_read_launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.ink2_ap_readraw_launch.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.ink2_ap_fires.argtypes = [ctypes.c_int]
        self.lib.ink2_ap_fires.restype = ctypes.c_uint

    def rows_per_block(self, fmt):
        return self.wpb * (32 // G_OF[fmt])

    def params(self, segs, xs, T):
        assert 1 <= T <= 8 and len(segs) <= MAXSEG
        P = CParams()
        blk = 0
        for i, s in enumerate(segs):
            assert s.K in (2304, 3072), s.K
            assert s.w.data_ptr() % 16 == 0 and (s.sc is None or s.sc.data_ptr() % 2 == 0)
            if s.out is None or s.out.shape != (T, s.rows):
                if s.buf is None:
                    s.buf = torch.empty(8, s.rows, dtype=torch.float32, device=s.w.device)     # one [8, rows] buffer: out is a view of its first T rows (graphs of every T share its address)
                s.out = s.buf[:T]
            nb = -(-s.rows // self.rows_per_block(s.fmt))
            c = P.s[i]
            c.w, c.sc, c.out = s.w.data_ptr(), (s.sc.data_ptr() if s.sc is not None else 0), s.out.data_ptr()
            c.rows, c.K, c.ldo, c.xi, c.fmt, c.blk0, c.nblk, c.gs = s.rows, s.K, s.rows, s.xi, s.fmt, blk, nb, s.gs
            blk += nb
        for i, x in enumerate(xs):
            if x is not None:
                P.x[i] = x.data_ptr()
        P.nseg, P.T, P.total_blocks = len(segs), T, blk
        return P

    def smem(self, segs, T):
        return T * max(s.K for s in segs) * 2

    def run(self, segs, xs, T, P=None):
        P = P or self.params(segs, xs, T)
        st = torch.cuda.current_stream().cuda_stream
        r = self.lib.ink2_ap_launch(ctypes.byref(P), self.smem(segs, T), st)
        assert r == 0, r
        return P

    def read(self, segs, T, sink, P=None, blocks=1024):
        P = P or self.params(segs, [None] * 3, T)
        st = torch.cuda.current_stream().cuda_stream
        assert self.lib.ink2_ap_read_launch(ctypes.byref(P), sink.data_ptr(), blocks, st) == 0
        return P

    def readraw(self, buf, sink, blocks=1024):
        st = torch.cuda.current_stream().cuda_stream
        assert self.lib.ink2_ap_readraw_launch(buf.data_ptr(), buf.numel() * buf.element_size(), sink.data_ptr(), blocks, st) == 0

    def fires(self, reset=False):
        return int(self.lib.ink2_ap_fires(1 if reset else 0))


# ------------------------------------------------------------------------------------------------------------------------------ numpy emulation of the kernel
def _dec_pair(t):
    return ((t & 0x00070007) << 9) | ((t & 0x00080008) << 12)


def _h2(u):
    """uint32 -> two float16 (lo, hi)"""
    return np.array([u & 0xFFFF, (u >> 16) & 0xFFFF], dtype=np.uint16).view(np.float16)


def _sc_of(s):
    return float(np.array([(s & 0x7F) << 7], dtype=np.uint16).view(np.float16)[0])


def _dec_pair_lut(t):
    """NVLUT=1: perm byte LUT -> exact fp16 E2M1 pair"""
    hi = [0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46]
    a, b = t & 7, (t >> 16) & 7
    return ((hi[a] << 8) | (hi[b] << 24)) | ((t & 0x00080008) << 12)


def _sc_exact(s):
    s7 = s & 0x7F
    e, m = s7 >> 3, s7 & 7
    return m * 2.0 ** -9 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)


def emulate_rows(fmt, w, sc, gs, x, rows, K, lut=False):
    """follow ink2_attnproj.hip lane by lane for the listed rows; w / sc numpy views of the stored planes, x float32 [T, K]. Returns [T, len(rows)] float64."""
    G, UE = G_OF[fmt], UE_OF[fmt]
    NU = (K // UE) // G
    T = x.shape[0]
    xh = np.clip(x, -65504, 65504).astype(np.float16)
    out = np.zeros((T, len(rows)))
    for ri, r in enumerate(rows):
        acc = np.zeros((G, T))
        for j in range(G):
            for i in range(NU):
                u = j + G * i
                if fmt == F_Q8:
                    dw = np.frombuffer(np.ascontiguousarray(w[r, 16 * u:16 * u + 16]).tobytes(), dtype=np.uint32)
                    d = float(sc[r, (u >> 1)].astype(np.float32))
                    wv = []
                    for q in dw:
                        v = int(q) ^ 0x80808080
                        a = (v & 0xFF) | (0x64 << 8) | (((v >> 8) & 0xFF) << 16) | (0x64 << 24)
                        b = ((v >> 16) & 0xFF) | (0x64 << 8) | (((v >> 24) & 0xFF) << 16) | (0x64 << 24)
                        wv += list(_h2(a) - np.float16(1152)) + list(_h2(b) - np.float16(1152))
                    wv = np.array(wv, dtype=np.float32)
                    for t in range(T):
                        acc[j, t] += d * float(np.dot(wv, xh[t, 16 * u:16 * u + 16].astype(np.float32)))
                elif fmt == F_NV:
                    dw = np.frombuffer(np.ascontiguousarray(w[r, 16 * u:16 * u + 16]).tobytes(), dtype=np.uint32)
                    sb = int(sc[r, 2 * u]) | (int(sc[r, 2 * u + 1]) << 8)
                    s0, s1 = (_sc_exact(sb & 0xFF), _sc_exact(sb >> 8)) if lut else (_sc_of(sb & 0xFF), _sc_of(sb >> 8))
                    for t in range(T):
                        xv = xh[t, 32 * u:32 * u + 32].astype(np.float32)
                        pa = pb = 0.0
                        for d in range(4):
                            p = 0.0
                            for o in range(4):
                                wp = _h2((_dec_pair_lut if lut else _dec_pair)(int(dw[d]) >> (4 * o))).astype(np.float32)
                                e = 8 * d + 2 * o
                                p += float(wp[0] * xv[e] + wp[1] * xv[e + 1])
                            if d < 2:
                                pa += p
                            else:
                                pb += p
                        acc[j, t] += s0 * pa + s1 * pb
                else:
                    wb = np.ascontiguousarray(w[r, 8 * u:8 * u + 8]).view(np.uint16).astype(np.uint32) << 16
                    wf = wb.view(np.float32)
                    for t in range(T):
                        acc[j, t] += float(np.dot(wf, xh[t, 8 * u:8 * u + 8].astype(np.float32)))
        out[:, ri] = acc.sum(0) * gs * (4194304.0 if (fmt == F_NV and not lut) else 1.0)
    return out
