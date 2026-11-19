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
ROW_TILES = os.environ.get("HY4_ROW_TILES", "0") == "1"     # 1 = verify-row tiles tuned on the Strix (T 2..4: BD 64 + 2 warps; T >= 5: BV 16, BD 32)


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


@triton.jit
def _k_headt(w_ptr, h_ptr, o_ptr, V, D, T, BT: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr):
    """T rows (verify): each weight tile is read once and applied to all rows; out [T, V]"""
    rows = tl.program_id(0) * BV + tl.arange(0, BV)
    rm = rows < V
    tt = tl.arange(0, BT)
    tm = tt < T
    acc = tl.zeros([BT, BV], dtype=tl.float32)
    for d0 in range(0, D, BD):
        cols = d0 + tl.arange(0, BD)
        cm = cols < D
        w = tl.load(w_ptr + rows[:, None] * D + cols[None, :], mask=rm[:, None] & cm[None, :], other=0.0).to(tl.float32)
        hv = tl.load(h_ptr + tt[:, None] * D + cols[None, :], mask=tm[:, None] & cm[None, :], other=0.0)
        acc += tl.sum(w[None, :, :] * hv[:, None, :], axis=2)
    tl.store(o_ptr + tt[:, None] * V + rows[None, :], acc, mask=tm[:, None] & rm[None, :])


def usable(h, W):
    return ENABLED and 1 <= h.shape[0] <= 16 and W.dim() == 2 and W.is_contiguous() and h.dtype == torch.float32 and W.dtype in (torch.bfloat16, torch.float16, torch.float32) \
        and (W.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1")


def gemv(h, W):
    """h [T, D] fp32 (T <= 16), W [V, D] -> logits [T, V] fp32"""
    V, D = W.shape
    T = h.shape[0]
    if T > 1:
        out = torch.empty(T, V, dtype=torch.float32, device=W.device)
        bv, bd, nw = BV, min(BD, 128), 4
        if ROW_TILES:
            bv, bd, nw = (BV, 64, 2) if T <= 4 else (16, 32, 4)
        _k_headt[(triton.cdiv(V, bv),)](W, h.contiguous(), out, V, D, T, BT=triton.next_power_of_2(T), BV=bv, BD=bd, num_warps=nw)
        return out
    out = torch.empty(1, V, dtype=torch.float32, device=W.device)
    _k_head[(triton.cdiv(V, BV),)](W, h.contiguous(), out, V, D, BV=BV, BD=BD, num_warps=4)
    return out
