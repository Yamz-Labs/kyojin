"""One Triton kernel for the k-pool update of the static indexer (HY4_IDX_POOLK=1): gather the (partial) group of each verify row from the per-position
(value | gate) rows, softmax over the group with the ape bias, weighted sum, rms norm, rope of the trailing rope_dim, store into the pool row.
Replaces ~10 small torch kernels per indexer layer. Same maths as the torch path in step_graph.indexer_forward_static (fp32; exp / rsqrt rounding may differ in the last ulp)."""
import torch
import triton
import triton.language as tl

ENABLED = False


@triton.jit
def _slot(g, VR: tl.constexpr):
    if VR > 0:
        r = g % VR              # HY4_SWA_RING: the (value | gate) rows live in a ring of VR rows
    else:
        r = g
    return r


@triton.jit
def _pooled(vgb, ape, g0, offs, k: tl.constexpr, d: tl.constexpr, VR: tl.constexpr):
    m = tl.full(offs.shape, float("-inf"), tl.float32)
    for j in tl.static_range(k):
        gj = tl.load(vgb + _slot(g0 + j, VR) * 2 * d + d + offs) + tl.load(ape + j * d + offs)
        m = tl.maximum(m, gj)
    s = tl.zeros(offs.shape, tl.float32)
    for j in tl.static_range(k):
        gj = tl.load(vgb + _slot(g0 + j, VR) * 2 * d + d + offs) + tl.load(ape + j * d + offs)
        s += tl.exp(gj - m)
    acc = tl.zeros(offs.shape, tl.float32)
    for j in tl.static_range(k):
        gj = tl.load(vgb + _slot(g0 + j, VR) * 2 * d + d + offs) + tl.load(ape + j * d + offs)
        vj = tl.load(vgb + _slot(g0 + j, VR) * 2 * d + offs)
        acc += (tl.exp(gj - m) / s) * vj
    return acc


@triton.jit
def _pool_kernel(vgb, pos, ape, pnw, cosb, sinb, pool, eps, k: tl.constexpr, d: tl.constexpr, rd: tl.constexpr, VR: tl.constexpr):
    t = tl.program_id(0)
    p0 = tl.load(pos + t)
    gi = p0 // k
    g0 = gi * k
    allo = tl.arange(0, d)
    pall = _pooled(vgb, ape, g0, allo, k, d, VR)
    r = tl.rsqrt(tl.sum(pall * pall, 0) / d + eps)
    nr: tl.constexpr = d - rd
    offn = tl.arange(0, nr)
    pn = _pooled(vgb, ape, g0, offn, k, d, VR)
    tl.store(pool + gi * d + offn, pn * r * tl.load(pnw + offn))
    h: tl.constexpr = rd // 2
    ii = tl.arange(0, h)
    oa = nr + 2 * ii
    ob = oa + 1
    na = _pooled(vgb, ape, g0, oa, k, d, VR) * r * tl.load(pnw + oa)
    nb = _pooled(vgb, ape, g0, ob, k, d, VR) * r * tl.load(pnw + ob)
    c = tl.load(cosb + g0 * h + ii)
    s = tl.load(sinb + g0 * h + ii)
    tl.store(pool + gi * d + oa, na * c - nb * s)
    tl.store(pool + gi * d + ob, na * s + nb * c)


def pool_update(ix, pos):
    """pos [T] long (device): update the pool row of each position's group from ix.vgb (the new rows are already in place)"""
    cs, sn = ix._table(ix._max_pos)
    ape = ix.w["ape"]
    _pool_kernel[(pos.shape[0],)](ix.vgb, pos, ape, ix.w["pool_norm"], cs, sn, ix.pool, ix.eps, k=ix.k, d=ix.d, rd=ix.rope_dim, VR=getattr(ix, "vg_ring", 0))
