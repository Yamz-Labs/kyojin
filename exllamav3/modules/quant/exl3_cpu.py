"""
CPU EXL3 codec for the mul1 codebook: encoder (uncalibrated tail-biting Viterbi), trellis pack / unpack, weight
reconstruction and a CPU `LinearEXL3` stand-in. No GPU and no CUDA extension needed.

Why it exists: the pack format is only proven by loading a pack and running it. The GPU encoder
(`quantize_exl3`) and the GPU kernels cannot run in CPU-only CI; this module writes and reads the SAME container
(`<key>.{trellis,suh,svh,mul1}`) so the converter -> pack -> serving-loader chain is tested end to end on tiny models.

Layout (exllamav3_ext/quant/pack.cu, reconstruct.cu, quantize/quantize.py `fallback_quant`):
  * a linear is stored [in, out]; 16x16 tiles, tile (i, j) = rows 16i.., cols 16j..; trellis int16 [in/16, out/16, 16K]
  * inside a tile the 256 positions follow `tensor_core_perm` (position q <-> row-major element perm[q])
  * state_i (16 bit) = ((state_{i-1} << K) | v_i) & 0xFFFF, v_i = low K bits; the K-bit fields v_0..v_255 are written
    MSB first into a cyclic stream of 256 K bits (tail-biting: state_i = the 16 stream bits ending at field i, mod 256 K)
  * the stream is stored as big-endian 32-bit words held in little-endian int16 pairs (SWAP16 of the 16-bit words)
  * mul1 codebook: value(s) = fp16(1024 + bytesum(s * 0x83DCD12D mod 2^32)) * k_inv + k_bias, k_inv = fp16 0x1EEE,
    k_bias = fp16 0xC931 (one fp16 fma)
  * W[in, out] = diag(su) . Had128_in . Wr . Had128_out . diag(sv)   (Wr = decoded tiles; su carries -1/codebook_scale)

Encoder: the uncalibrated path of `quantize_exl3` (identity Hessian, `q_fallback`): output channel scales, input
channel scales, blockwise Hadamards, tail-biting Viterbi per tile. The global scale search is skipped (g_scale = 1) and
the random sign vectors come from the CPU generator, so the bytes differ from the GPU encoder (same format, same
quality class). The Viterbi is C++/OpenMP (built with g++ on first use, cached); a numpy reference is the fallback.
"""
from __future__ import annotations
import ctypes
import hashlib
import math
import os
import subprocess
import tempfile
import zlib
from functools import lru_cache

import numpy as np
import torch

HAD = 128
CODEBOOK_SCALE = 1.24371088
MUL1_MULT = 0x83DCD12D


# ------------------------------------------------------------------------------------------------ codebook, layout
@lru_cache(maxsize=1)
def mul1_lut() -> np.ndarray:
    """fp32 values of the 65536 mul1 states (exactly the fp16 the kernels produce)."""
    s = np.arange(65536, dtype=np.uint64)
    p = ((s * MUL1_MULT) & 0xFFFFFFFF).astype(np.uint32)
    bsum = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + (p >> 24)
    k_inv = np.array([0x1EEE], dtype=np.uint16).view(np.float16).astype(np.float64)[0]
    k_bias = np.array([0xC931], dtype=np.uint16).view(np.float16).astype(np.float64)[0]
    v = (1024.0 + bsum.astype(np.float64)) * k_inv + k_bias
    return v.astype(np.float16).astype(np.float32)


@lru_cache(maxsize=1)
def tile_perm() -> np.ndarray:
    """perm[q] = row-major index (r * 16 + c) of the tile element at trellis position q."""
    p = np.zeros(256, dtype=np.int64)
    for t in range(32):
        r0, c0 = (t % 4) * 2, t // 4
        r = (r0, r0 + 1, r0 + 8, r0 + 9)
        for h, c in enumerate((c0, c0 + 8)):
            for i in range(4):
                p[t * 8 + h * 4 + i] = r[i] * 16 + c
    return p


def pack_states(states: np.ndarray, K: int) -> np.ndarray:
    """states uint16 [T, 256] -> int16 [T, 16 K] (the trellis words)."""
    T = states.shape[0]
    v = (states.astype(np.uint32) & ((1 << K) - 1))
    bits = ((v[:, :, None] >> np.arange(K - 1, -1, -1, dtype=np.uint32)) & 1).astype(np.uint8).reshape(T, 256 * K)
    by = np.packbits(bits, axis=1)                                   # MSB first, [T, 32 K] bytes
    by = by.reshape(T, -1, 4)[:, :, ::-1]                            # big-endian 32-bit words -> little-endian
    return np.ascontiguousarray(by).reshape(T, -1).view(np.int16).reshape(T, 16 * K).copy()


def unpack_states(packed: np.ndarray, K: int) -> np.ndarray:
    """int16 [T, 16 K] -> uint16 [T, 256] states."""
    T = packed.shape[0]
    by = np.ascontiguousarray(packed).view(np.uint8).reshape(T, -1, 4)[:, :, ::-1].reshape(T, -1)
    bits = np.unpackbits(by, axis=1)                                 # [T, 256 K]
    n = 256 * K
    idx = ((np.arange(256)[:, None] + 1) * K - 16 + np.arange(16)[None, :]) % n      # window ending at field i
    w = bits[:, idx].astype(np.uint32)                               # [T, 256, 16], MSB first
    return (w << np.arange(15, -1, -1, dtype=np.uint32)).sum(-1).astype(np.uint16)


def decode_tiles(trellis: torch.Tensor) -> torch.Tensor:
    """trellis int16 [ti, tj, 16 K] -> regularized weight Wr float32 [ti * 16, tj * 16]."""
    ti, tj, w = trellis.shape
    K = w // 16
    st = unpack_states(trellis.reshape(ti * tj, w).cpu().numpy(), K)
    vals = mul1_lut()[st]                                            # [T, 256] in trellis order
    tiles = np.empty_like(vals)
    tiles[:, tile_perm()] = vals                                     # row-major tile elements
    out = tiles.reshape(ti, tj, 16, 16).transpose(0, 2, 1, 3).reshape(ti * 16, tj * 16)
    return torch.from_numpy(np.ascontiguousarray(out))


@lru_cache(maxsize=2)
def _had(dt):
    from ...util.hadamard import get_hadamard
    return get_hadamard(HAD).to(dt) / math.sqrt(HAD)


def reconstruct(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor) -> torch.Tensor:
    """Dense weight [in, out] float32 of an EXL3 linear (what `y = x @ W` multiplies with)."""
    wr = decode_tiles(trellis)
    k, n = wr.shape
    had = _had(torch.float32)
    w = (had @ wr.view(k // HAD, HAD, n)).view(k, n)                 # Had along in
    w = w * suh.float().view(k, 1)
    w = (w.view(k, n // HAD, HAD) @ had).view(k, n)                  # Had along out
    return w * svh.float().view(1, n)


# ------------------------------------------------------------------------------------------------ Viterbi
_CPP = r"""
#include <cstdint>
#include <cstring>
#include <vector>
#include <limits>
#include <algorithm>
extern "C" {
// tiles [T,256] float (trellis order), lut [65536], out [T,256] uint16 states
void exl3_cpu_viterbi(const float* tiles, const float* lut, int K, int T, uint16_t* out)
{
    const int L = 256, kr = 16 - K, E = 1 << kr, B = 1 << K;
    #pragma omp parallel
    {
        std::vector<uint16_t> hist((size_t)L * E);
        std::vector<float> cost(E), nc(E);
        auto forward = [&](const float* x, int first_pos, int start_edge, bool rolled) {
            for (int step = 0; step < L; ++step) {
                int pos = rolled ? (first_pos + step) % L : step;
                float xv = x[pos];
                uint16_t* h = &hist[(size_t)pos * E];
                for (int n = 0; n < E; ++n) {
                    float best = std::numeric_limits<float>::infinity();
                    uint16_t bp = 0;
                    for (int b = 0; b < B; ++b) {
                        uint32_t s = ((uint32_t)b << kr) | (uint32_t)n;
                        uint32_t p = s >> K;
                        float d = lut[s] - xv;
                        float m = d * d;
                        if (step > 0) m += cost[p];
                        else if (start_edge >= 0 && (int)p != start_edge) m = std::numeric_limits<float>::infinity();
                        if (m < best) { best = m; bp = (uint16_t)p; }
                    }
                    nc[n] = best; h[n] = bp;
                }
                cost.swap(nc);
            }
        };
        #pragma omp for schedule(dynamic)
        for (int t = 0; t < T; ++t) {
            const float* x = tiles + (size_t)t * L;
            forward(x, L / 2, -1, true);
            int edge = 0; float bc = cost[0];
            for (int n = 1; n < E; ++n) if (cost[n] < bc) { bc = cost[n]; edge = n; }
            for (int i = L - 1; i >= 0; --i) {
                int pos = (i + L / 2) % L;
                edge = hist[(size_t)pos * E + edge];
                if (pos == 0) break;
            }
            int start_edge = edge;
            forward(x, 0, start_edge, false);
            uint16_t* o = out + (size_t)t * L;
            edge = start_edge;
            for (int pos = L - 1; pos >= 0; --pos) {
                int prev = hist[(size_t)pos * E + edge];
                o[pos] = (uint16_t)(((prev << K) | edge) & 0xFFFF);
                edge = prev;
            }
        }
    }
}
}
"""

_lib = None


def _build_lib():
    global _lib
    if _lib is not None:
        return _lib or None
    tag = hashlib.sha1(_CPP.encode()).hexdigest()[:12]
    cache = os.environ.get("EXL3_CPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "exl3_cpu")
    os.makedirs(cache, exist_ok=True)
    so = os.path.join(cache, f"viterbi_{tag}.so")
    if not os.path.exists(so):
        src = os.path.join(cache, f"viterbi_{tag}.cpp")
        with open(src, "w") as f:
            f.write(_CPP)
        tmp = so + f".{os.getpid()}.tmp"
        try:
            subprocess.run(["g++", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", "-o", tmp, src],
                           check=True, capture_output=True)
            os.replace(tmp, so)
        except (OSError, subprocess.CalledProcessError):
            _lib = False
            return None
    lib = ctypes.CDLL(so)
    lib.exl3_cpu_viterbi.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.exl3_cpu_viterbi.restype = None
    _lib = lib
    return lib


def viterbi_py(tiles: np.ndarray, K: int) -> np.ndarray:
    """numpy reference of the C++ search (same algorithm, slow)."""
    lut = mul1_lut()
    kr, E, B, L = 16 - K, 1 << (16 - K), 1 << K, 256
    n = np.arange(E, dtype=np.uint32)[None, :]
    s = (np.arange(B, dtype=np.uint32)[:, None] << kr) | n                      # [B, E]
    p = s >> K
    dec = lut[s]
    out = np.zeros((tiles.shape[0], L), dtype=np.uint16)
    for ti, x in enumerate(tiles):
        hist = np.zeros((L, E), dtype=np.uint16)

        def forward(order, start_edge):
            cost = None
            for step, pos in enumerate(order):
                m = (dec - x[pos]) ** 2
                if cost is not None:
                    m = m + cost[p]
                elif start_edge is not None:
                    m = np.where(p == start_edge, m, np.inf)
                best = np.argmin(m, axis=0)
                cost = m[best, np.arange(E)]
                hist[pos] = p[best, np.arange(E)]
            return cost

        cost = forward(list(range(L // 2, L)) + list(range(L // 2)), None)
        edge = int(np.argmin(cost))
        for i in range(L - 1, -1, -1):
            pos = (i + L // 2) % L
            edge = int(hist[pos, edge])
            if pos == 0:
                break
        start = edge
        forward(range(L), start)
        edge = start
        for pos in range(L - 1, -1, -1):
            prev = int(hist[pos, edge])
            out[ti, pos] = ((prev << K) | edge) & 0xFFFF
            edge = prev
    return out


def viterbi(tiles: np.ndarray, K: int, force_python: bool = False) -> np.ndarray:
    """tiles float32 [T, 256] (trellis order) -> states uint16 [T, 256], tail-biting."""
    assert 1 <= K <= 8
    tiles = np.ascontiguousarray(tiles, dtype=np.float32)
    lib = None if force_python else _build_lib()
    if lib is None:
        return viterbi_py(tiles, K)
    out = np.empty(tiles.shape, dtype=np.uint16)
    lut = np.ascontiguousarray(mul1_lut())
    lib.exl3_cpu_viterbi(tiles.ctypes.data, lut.ctypes.data, K, tiles.shape[0], out.ctypes.data)
    return out


# ------------------------------------------------------------------------------------------------ encoder
def _signs(n: int, gen: torch.Generator) -> torch.Tensor:
    return (torch.randn(n, generator=gen).sign() + 1e-5).sign().float()


def quantize_linear(weight_oi: torch.Tensor, K: int, seed: int = 0, out_scales: bool = True) -> dict:
    """weight_oi [out, in] -> {"trellis", "suh", "svh", "mul1"}: the uncalibrated `quantize_exl3` path on the CPU."""
    out_f, in_f = weight_oi.shape
    assert in_f % HAD == 0 and out_f % HAD == 0, "EXL3 needs 128-aligned dims"
    gen = torch.Generator().manual_seed(seed)
    w = weight_oi.float().t().contiguous()                                      # [in, out]
    k, n = w.shape
    su = _signs(k, gen).unsqueeze(1)
    sv = _signs(n, gen).unsqueeze(0)
    ocs = w.square().mean(0, keepdim=True).sqrt()
    mean = ocs.mean().item()
    zero_w = mean <= 1e-30
    if not zero_w:
        ocs = ocs / mean
    zero_ocs = ocs.abs() < 1e-30
    if out_scales or zero_w:
        ocs = ocs.clone()
        ocs[zero_ocs] = 0.1
        sv = (sv * ocs + 1e-10).float()
    w = w / sv
    sv[zero_ocs] = 0.0
    had = _had(torch.float32)
    w = (w.view(k, n // HAD, HAD) @ had).view(k, n)                             # Had along out
    ics = w.square().mean(1, keepdim=True).sqrt()
    ics[ics.abs() < 1e-30] = 0.1
    su = (su * ics / (-CODEBOOK_SCALE) + 1e-10).float()
    w = w / su
    wr = (had @ w.view(k // HAD, HAD, n)).view(k, n)                            # Had along in
    ti, tj = k // 16, n // 16
    tiles = wr.view(ti, 16, tj, 16).permute(0, 2, 1, 3).reshape(ti * tj, 256).numpy()
    st = viterbi(tiles[:, tile_perm()], K)
    trellis = torch.from_numpy(pack_states(st, K)).view(ti, tj, 16 * K)
    return {"trellis": trellis, "suh": su.flatten().half().contiguous(), "svh": sv.flatten().half().contiguous(),
            "mul1": torch.tensor(MUL1_MULT - (1 << 32), dtype=torch.int32)}


# ------------------------------------------------------------------------------------------------ CPU linear
class CpuLinearEXL3:
    """`LinearEXL3` stand-in for tensors on the CPU: reconstructs the dense weight once, then a float matmul."""

    quant_type = "exl3"

    def __init__(self, config, in_features, out_features, scale=None, su=None, sv=None, suh=None, svh=None,
                 trellis=None, mcg=None, mul1=None, bias=None, out_dtype=None, transformers_fix=False, key=None):
        assert suh is not None and svh is not None and trellis is not None, "CPU EXL3 needs suh / svh / trellis"
        assert mcg is None, "CPU EXL3 supports the mul1 codebook only"
        self.config, self.key = config, key
        self.suh, self.svh, self.trellis = suh, svh, trellis
        self.K = trellis.shape[-1] / 16
        self.in_features, self.out_features = in_features, out_features
        self.bias = bias.half() if bias is not None and bias.dtype == torch.float else bias
        self.out_dtype = out_dtype
        self.default_out_dtype = out_dtype or torch.half
        self.mcg_tensor, self.mul1_tensor, self.mcg, self.mul1 = None, mul1, False, True
        self.swap_device = None
        self._w = None

    def weight(self) -> torch.Tensor:
        if self._w is None:
            self._w = reconstruct(self.trellis, self.suh, self.svh).half()
        return self._w

    get_weight_tensor = weight

    def unload(self):
        self._w = None

    def get_tensors(self, key: str):
        return {f"{key}.{n}": t.contiguous() for n, t in
                (("suh", self.suh), ("svh", self.svh), ("trellis", self.trellis), ("bias", self.bias),
                 ("mul1", self.mul1_tensor)) if t is not None}

    def forward(self, x: torch.Tensor, params: dict, out_dtype=None):
        y = (x.float() @ self.weight().float())
        if self.bias is not None:
            y = y + self.bias.float()
        return y.to(out_dtype or self.default_out_dtype)
