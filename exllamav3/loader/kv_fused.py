"""HY4_KV_FUSED=1 (graph decode only): ONE Triton kernel writes the K-cache row of a token:
   K[pos] = [ rms(c[:R]) * kv_a_norm | rope(c[R:]) ]   (c = the kv_a projection output [1, R + P])
replacing slice + rms (pow, mean, add, rsqrt, mul) + mul + rope (index_select, 2 mul, add) + cat / copy + index_copy_ (~12 launches per layer).
rope reads the [n, 2, P/2, 2] (cos, sin | -sin, cos) table of attn_eager (same fp64-built values); the position comes from a device tensor."""
import os
import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("HY4_KV_FUSED", "0") == "1"


@triton.jit
def _k_kv(c_ptr, w_ptr, tab_ptr, pos_ptr, k_ptr, sc, sk, eps, R: tl.constexpr, P: tl.constexpr, BR: tl.constexpr, RING: tl.constexpr):
    pos = tl.load(pos_ptr)
    if RING > 0:
        slot = pos % RING          # window layer: ring buffer row (the rope table is still indexed by the absolute position)
    else:
        slot = pos
    offr = tl.arange(0, BR)
    mr = offr < R
    x = tl.load(c_ptr + offr, mask=mr, other=0.0)
    rc = tl.rsqrt(tl.sum(x * x, axis=0) / R + eps)
    w = tl.load(w_ptr + offr, mask=mr, other=0.0)
    tl.store(k_ptr + slot * sk + offr, x * rc * w, mask=mr)
    H: tl.constexpr = P // 2
    offh = tl.arange(0, H)
    x0 = tl.load(c_ptr + R + 2 * offh)
    x1 = tl.load(c_ptr + R + 2 * offh + 1)
    tb = tab_ptr + pos * (2 * H * 2)
    a0 = tl.load(tb + offh * 2)               # c
    a1 = tl.load(tb + offh * 2 + 1)           # s
    b0 = tl.load(tb + H * 2 + offh * 2)       # -s
    b1 = tl.load(tb + H * 2 + offh * 2 + 1)   # c
    tl.store(k_ptr + slot * sk + R + 2 * offh, x0 * a0 + x1 * b0)
    tl.store(k_ptr + slot * sk + R + 2 * offh + 1, x0 * a1 + x1 * b1)


def write_row(c, w, tab, pos, kbuf, R, P, eps, ring=0):
    """c [1, R+P] fp32 (unit column stride), w [R], tab [n, 2, P/2, 2], pos [1] long device, kbuf [cap, R+P] fp32"""
    _k_kv[(1,)](c, w, tab, pos, kbuf, c.stride(0), kbuf.stride(0), eps, R=R, P=P, BR=triton.next_power_of_2(R), RING=ring, num_warps=4)


@triton.jit
def _k_qrope(q_ptr, tab_ptr, pos_ptr, o_ptr, sqh, soh, NOPE: tl.constexpr, P: tl.constexpr):
    h = tl.program_id(0)
    pos = tl.load(pos_ptr)
    H: tl.constexpr = P // 2
    offh = tl.arange(0, H)
    x0 = tl.load(q_ptr + h * sqh + NOPE + 2 * offh)
    x1 = tl.load(q_ptr + h * sqh + NOPE + 2 * offh + 1)
    tb = tab_ptr + pos * (2 * H * 2)
    a0 = tl.load(tb + offh * 2)
    a1 = tl.load(tb + offh * 2 + 1)
    b0 = tl.load(tb + H * 2 + offh * 2)
    b1 = tl.load(tb + H * 2 + offh * 2 + 1)
    tl.store(o_ptr + h * soh + 2 * offh, x0 * a0 + x1 * b0)
    tl.store(o_ptr + h * soh + 2 * offh + 1, x0 * a1 + x1 * b1)


def rope_q(qq, tab, pos, nope, P):
    """qq [1, H, hd] fp32 (unit last stride; the pe part is qq[..., nope:nope+P]) -> rotated pe [1, H, P] (one launch)"""
    H = qq.shape[1]
    out = torch.empty(1, H, P, dtype=torch.float32, device=qq.device)
    _k_qrope[(H,)](qq, tab, pos, out, qq.stride(1), out.stride(1), NOPE=nope, P=P, num_warps=1)
    return out
