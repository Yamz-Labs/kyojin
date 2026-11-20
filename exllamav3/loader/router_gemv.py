"""HY4_ROUTER_GEMV=1: the router logits x [T, K] @ W[N, K].T for T <= MAXT rows as ONE Triton mat-vec (weights read once for all T rows, fp32 accumulate).

rocBLAS runs this shape (T = 1..8, N = 512, K = 3072, fp32 weights 6 MiB) as an SGEMM tile kernel at about 47 GB/s (134 us per layer, 5.5 ms per decode step over
41 MoE layers); the mat-vec below reads the same bytes at DRAM speed. Same products as the fp32 matmul, only the summation order differs. T > MAXT keeps the matmul."""
import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("HY4_ROUTER_GEMV", "0") == "1"
MAXT = 8
BN = int(os.environ.get("HY4_RG_BN", "4"))
TUNE = os.environ.get("HY4_RG_TUNE", "0") == "1"    # tuned K block (dec8_micro, real GGUF shape, fp16 weights): 1024 instead of 256, T=1 26.8 -> 16.2 us, T=4 30.7 -> 17.8 us per launch, max rel err 1.6e-6 (summation order only)
BK = int(os.environ.get("HY4_RG_BK", "1024" if TUNE else "256"))
NW = int(os.environ.get("HY4_RG_NW", "4"))
F16 = os.environ.get("HY4_RG_F16", "0") == "1"      # router weights stored fp16 (GGUF F32 -> fp16: rel 5e-4 per weight, logits ~1e-5): half the bytes of the 41 gemvs; fp32 accumulate


@triton.jit
def _k_rg(x_ptr, w_ptr, o_ptr, N, K, T, sxt, BT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    rm = rows < N
    tt = tl.arange(0, BT)
    tm = tt < T
    acc = tl.zeros([BT, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        cols = k0 + tl.arange(0, BK)
        cm = cols < K
        w = tl.load(w_ptr + rows[:, None] * K + cols[None, :], mask=rm[:, None] & cm[None, :], other=0.0).to(tl.float32)
        xv = tl.load(x_ptr + tt[:, None] * sxt + cols[None, :], mask=tm[:, None] & cm[None, :], other=0.0)
        acc += tl.sum(w[None, :, :] * xv[:, None, :], axis=2)
    tl.store(o_ptr + tt[:, None] * N + rows[None, :], acc, mask=tm[:, None] & rm[None, :])


def usable(x, w):
    return (ENABLED and x.dtype == torch.float32 and w.dtype in (torch.float32, torch.float16) and x.dim() == 2 and 1 <= x.shape[0] <= MAXT and w.is_contiguous()
            and x.stride(-1) == 1 and (x.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1"))


def store(w):
    """router weight as kept resident: fp16 when HY4_RG_F16=1 (and the gemv is on), else unchanged fp32"""
    return w.half().contiguous() if (ENABLED and F16) else w


def mm(x, w):
    """x [T, K] fp32 @ w[N, K].T -> [T, N] fp32: the Triton mat-vec when usable, else the matmul (fp16 weights upcast)"""
    return logits(x, w) if usable(x, w) else x @ w.float().T


def logits(x, w, bn=None, bk=None, nw=None):
    """x [T, K] fp32, w [N, K] fp32 (row-major) -> [T, N] fp32 = x @ w.T"""
    T, K = x.shape
    N = w.shape[0]
    out = torch.empty(T, N, dtype=torch.float32, device=x.device)
    bn, bk = bn or BN, bk or BK
    _k_rg[(triton.cdiv(N, bn),)](x, w, out, N, K, T, x.stride(0), BT=triton.next_power_of_2(T), BN=bn, BK=bk, num_warps=nw or NW)
    return out
