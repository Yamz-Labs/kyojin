"""HY4_HEAD_BF16=1: output head as ONE Triton mat-vec on the stored bf16 weights (fp32 accumulate, fp32 logits).
Old path (hy4_forward.head_gemm): per 8192-row chunk `W[a:a+c].float() @ h` = bf16 -> fp32 copy (write 4 B + read 4 B per weight) + GEMV:
~3.7 GB of traffic per token for a 0.74 GB matrix (18 ms). Here: 0.74 GB read once (~3.5 ms at 210 GB/s). Same products in fp32; only the
summation order differs (rel ~1e-6). Generic: any [V, D] bf16 / f16 / f32 weight, h [1, D] fp32."""
import os
import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("HY4_HEAD_BF16", "0") == "1"
BV = int(os.environ.get("HY4_HEAD_BV", "8"))
BD = int(os.environ.get("HY4_HEAD_BD", "512"))


@triton.jit
def _k_head(w_ptr, h_ptr, o_ptr, V, D, BV: tl.constexpr, BD: tl.constexpr):
    rows = tl.program_id(0) * BV + tl.arange(0, BV)
    rm = rows < V
    acc = tl.zeros([BV], dtype=tl.float32)
    for d0 in range(0, D, BD):
        cols = d0 + tl.arange(0, BD)
        cm = cols < D
        w = tl.load(w_ptr + rows[:, None] * D + cols[None, :], mask=rm[:, None] & cm[None, :], other=0.0).to(tl.float32)
        hv = tl.load(h_ptr + cols, mask=cm, other=0.0)
        acc += tl.sum(w * hv[None, :], axis=1)
    tl.store(o_ptr + rows, acc, mask=rm)


def usable(h, W):
    return ENABLED and h.shape[0] == 1 and W.dim() == 2 and W.is_contiguous() and h.dtype == torch.float32 and W.dtype in (torch.bfloat16, torch.float16, torch.float32) \
        and (W.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1")


def gemv(h, W):
    """h [1, D] fp32, W [V, D] -> logits [1, V] fp32"""
    V, D = W.shape
    out = torch.empty(1, V, dtype=torch.float32, device=W.device)
    _k_head[(triton.cdiv(V, BV),)](W, h.contiguous(), out, V, D, BV=BV, BD=BD, num_warps=4)
    return out
