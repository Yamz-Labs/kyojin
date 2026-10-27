"""
DS4.1 HIP kernels (exllamav3/modules/dsv41_fn/dsv41_hip.hip) and the grouped-MoE swiglu limit against the CPU-exact torch
reference (exllamav3/dsv41/{ops,quant,engram}.py).

Opt-in (never collected by a plain pytest run): needs the box GPU, so run it under the GPU lock (<= 20 min window):
    flock -w 14400 ~/ds41_work/gpu.lock bash -c 'source env... ; DSV41_DEVICE=cuda python -m pytest --noconftest -q \
        tests/dsv41/test_dsv41_hipk.py'

Exactness contract (what "within the repo tolerance" means here):
  * the number formats are bit-identical to the reference on identical inputs (fq_rows, mxfp4_decode, engram ids): torch.equal;
  * kernels that reduce (rms, dot products, softmax) use a different summation tree than torch, so a result is allowed to differ
    in the last fp32 bits before an fp16 store; fp8/fp4 steps downstream are step functions, so a tiny fraction of elements may
    land on the neighbouring grid point. Each such test bounds that fraction (FRAC) and, for continuous outputs, rtol/atol.
"""
import os, sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exllamav3.dsv41 import ops, quant
from exllamav3.dsv41.engram import (EngramLayout, NgramHasher, MemoryEngramStore, dequant_rows, engram_inject)

DEV = os.environ.get("DSV41_DEVICE", "")
if not DEV:
    pytest.skip("GPU kernels are opt-in: run under the GPU lock with DSV41_DEVICE=cuda (DSV41_DEVICE=emu: CPU emulation "
                "of the kernel model, no GPU)", allow_module_level=True)
EMU = DEV == "emu"
if not EMU and not torch.cuda.is_available():
    pytest.skip("no HIP device", allow_module_level=True)

from exllamav3.modules.dsv41_fn import dsv41_hip as K
if EMU:
    import hipemu.build_emu as _emu
    _emu.install(K)
D = "cpu" if EMU else "cuda"

FRAC = 3e-3
EPS = 1e-20
g = torch.Generator().manual_seed(2024)


def cu(x):
    return x.contiguous() if EMU else x.cuda().contiguous()


def sync():
    if not EMU:
        torch.cuda.synchronize()


def frac_diff(a, b):
    return (a != b).float().mean().item()


def rows_wide(n, d, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=gen)
    return (x * torch.exp2(torch.randint(-6, 5, (n, d // 16), generator=gen).float().repeat_interleave(16, 1))).half()


def rope_fc(max_pos=512):
    return ops.rope_tables(64, max_pos, 65536, 160000.0, 16.0, 32, 1)


# --------------------------------------------------------------------------------------- fake-quant (bitwise)
@pytest.mark.parametrize("mode,fn", [(K.FQ_FP8, lambda x: quant.act_quant_fp8_(x, 32)),
                                     (K.FQ_FP4_UE8M0, lambda x: quant.fp4_act_quant_(x, 32)),
                                     (K.FQ_FP4_E4M3, lambda x: quant.fp4_act_quant_(x, 16, e4m3_scale=True))])
def test_fq_rows_bitwise(mode, fn):
    x = rows_wide(64, 4096, mode)
    ref = fn(x.clone())
    got = cu(x)
    K.fq_rows(got, mode)
    sync()
    assert torch.equal(got.cpu(), ref)


def test_mxfp4_decode_bitwise():
    raw = torch.randint(0, 256, (777, 136), dtype=torch.uint8, generator=g)
    raw[:, 128:] = torch.randint(100, 150, (777, 8), dtype=torch.uint8, generator=g)
    ref = dequant_rows(raw, torch.float16, "mxfp4")
    out = torch.empty(777, 256, dtype=torch.half, device=D)
    K.mxfp4_decode(cu(raw), out)
    inv = torch.randint(0, 777, (1000,), dtype=torch.int32, generator=g)
    out2 = torch.empty(1000, 256, dtype=torch.half, device=D)
    K.mxfp4_decode(cu(raw), out2, cu(inv))
    sync()
    assert torch.equal(out.cpu(), ref)
    assert torch.equal(out2.cpu(), ref[inv.long()])


def test_fp8_decode_bitwise():
    raw = torch.randint(0, 256, (777, 264), dtype=torch.uint8, generator=g)
    raw[:, :256][(raw[:, :256] & 127) == 127] = 0x38                       # no e4m3 NaN
    raw[:, 256:] = torch.randint(100, 150, (777, 8), dtype=torch.uint8, generator=g)
    ref = dequant_rows(raw, torch.float16, "fp8")
    out = torch.empty(777, 256, dtype=torch.half, device=D)
    K.fp8_decode(cu(raw), out)
    inv = torch.randint(0, 777, (1000,), dtype=torch.int32, generator=g)
    out2 = torch.empty(1000, 256, dtype=torch.half, device=D)
    K.fp8_decode(cu(raw), out2, cu(inv))
    sync()
    assert torch.equal(out.cpu(), ref)
    assert torch.equal(out2.cpu(), ref[inv.long()])


# --------------------------------------------------------------------------------------- attention prep
def test_kv_prep():
    R, win = 37, 64                                     # ring slots must be unique within one launch (blocks race otherwise)
    kv = rows_wide(R, 512, 1)
    w16 = (torch.randn(512, generator=g) * 0.5 + 1).half()
    fc = rope_fc()
    pos = torch.randint(0, 500, (R,), dtype=torch.int32, generator=g)
    slot = torch.arange(R, dtype=torch.int32) % win
    slot[::5] = -1
    assert len(set(slot[slot >= 0].tolist())) == int((slot >= 0).sum())
    x = ops.rmsnorm(kv.unsqueeze(0), w16, EPS)
    ops.rope_apply_(x[..., -64:], fc[pos.long()])
    quant.act_quant_fp8_(x, 32)
    ref = x[0]
    out = torch.empty(R, 512, dtype=torch.half, device=D)
    ring = torch.zeros(win, 512, dtype=torch.half, device=D)
    K.kv_prep(cu(kv), cu(w16.float()), cu(K.rope_table_f32(fc)), cu(pos), cu(slot), out, ring, EPS)
    sync()
    assert frac_diff(out.cpu(), ref) < FRAC
    ring_ref = torch.zeros(win, 512, dtype=torch.half)
    for r in range(R):
        if slot[r] >= 0:
            ring_ref[slot[r]] = out.cpu()[r]
    assert torch.equal(ring.cpu(), ring_ref)


def test_idx_q_prep():
    R = 9
    q = rows_wide(R, 32 * 128, 2)
    fc = rope_fc()
    pos = torch.randint(0, 500, (R,), dtype=torch.int32, generator=g)
    x = q.clone().view(1, R, 32, 128)
    ops.rope_apply_(x[..., -64:], fc[pos.long()])
    quant.fp4_act_quant_(x, 32)
    got = cu(q)
    K.idx_q_prep(got, cu(K.rope_table_f32(fc)), cu(pos))
    sync()
    assert frac_diff(got.cpu().view(1, R, 32, 128), x) < FRAC


def test_idx_k_store_and_pool_store():
    R, rows = 11, 40
    fc = rope_fc()
    ck = (torch.arange(R, dtype=torch.int32) * 2 + 3)
    dst = torch.randperm(rows, generator=g)[:R].int()
    dst[3] = -1
    # indexer key
    k = rows_wide(R, 128, 3)
    w16 = (torch.randn(128, generator=g) * 0.3 + 1).half()
    x = ops.rmsnorm(k.unsqueeze(0), w16, EPS)
    ops.rope_apply_(x[..., -64:], fc[ck.long()])
    quant.fp4_act_quant_(x, 32)
    pool = torch.zeros(rows, 128, dtype=torch.half, device=D)
    K.idx_k_store(cu(k), cu(w16.float()), cu(K.rope_table_f32(fc)), cu(ck), cu(dst), pool, EPS)
    # compressed pool
    lat = rows_wide(R, 512, 4)
    y = lat.clone().unsqueeze(0)
    ops.rope_apply_(y[..., -64:], fc[ck.long()])
    quant.fp4_act_quant_(y, 16, e4m3_scale=True)
    pool2 = torch.zeros(rows, 512, dtype=torch.half, device=D)
    K.pool_store(cu(lat), cu(K.rope_table_f32(fc)), cu(ck), cu(dst), pool2)
    sync()
    ok = dst >= 0
    assert frac_diff(pool.cpu()[dst[ok].long()], x[0][ok]) < FRAC
    assert frac_diff(pool2.cpu()[dst[ok].long()], y[0][ok]) < FRAC
    assert not pool.cpu()[torch.tensor(sorted(set(range(rows)) - set(dst[ok].tolist())))].any()


def ref_compress(kv, sc, kst, sst, start, ratio, w16):
    """model.py _compressor generalised to any start: groups completed by rows start .. start + R - 1."""
    R = kv.size(0)
    first = start // ratio
    last = (start + R) // ratio
    outs = []
    for gi in range(first, last):
        ks, ss = [], []
        for j in range(ratio):
            p = gi * ratio + j
            ks.append(kv[p - start] if p >= start else kst[j])
            ss.append(sc[p - start] if p >= start else sst[j])
        k, s = torch.stack(ks, 0), torch.stack(ss, 0)
        outs.append((k * s.softmax(dim=0)).sum(dim=0))
    if not outs:
        return None
    c = torch.stack(outs, 0)
    return ops.rmsnorm(c.half().unsqueeze(0), w16, EPS)[0]


@pytest.mark.parametrize("start,R,ratio", [(0, 11, 2), (0, 8, 2), (5, 6, 2), (6, 1, 2), (7, 1, 2), (3, 9, 4), (0, 1, 2)])
def test_compress(start, R, ratio):
    kv, sc = torch.randn(R, 512, generator=g), torch.randn(R, 512, generator=g) * 2
    kst, sst = torch.randn(ratio, 512, generator=g), torch.randn(ratio, 512, generator=g)
    w16 = (torch.randn(512, generator=g) * 0.4 + 1).half()
    ref = ref_compress(kv, sc, kst, sst, start, ratio, w16)
    ng = (start + R) // ratio - start // ratio
    out = torch.zeros(max(ng, 1), 512, dtype=torch.half, device=D)
    kst_d, sst_d = cu(kst), cu(sst)
    n = K.compress(cu(kv), cu(sc), kst_d, sst_d, cu(w16.float()), out, start, ratio, EPS)
    sync()
    assert n == ng
    if ref is not None:
        torch.testing.assert_close(out.cpu()[:ng].float(), ref.float(), rtol=3e-3, atol=2e-3)
    end = start + R                                     # state: rows of the trailing incomplete group
    for j in range(end % ratio):
        p = end - end % ratio + j
        if p >= start:
            assert torch.equal(kst_d.cpu()[j], kv[p - start]) and torch.equal(sst_d.cpu()[j], sc[p - start])
        else:
            assert torch.equal(kst_d.cpu()[j], kst[j])


# --------------------------------------------------------------------------------------- hyper-connections
def hc_params(seed=5):
    gen = torch.Generator().manual_seed(seed)
    fn = torch.randn(24, 4 * 5120, generator=gen) * 0.02
    scale = torch.tensor([0.7, 0.6, 0.5])
    base = torch.randn(24, generator=gen) * 0.3
    return fn, scale, base


@pytest.mark.parametrize("R,fn_half", [(1, False), (1, True), (6, False), (33, True)])
def test_hc_mix(R, fn_half):
    fn, scale, base = hc_params()
    if fn_half:
        fn = fn.half().float()
    h = torch.randn(R, 4, 5120, generator=g) * 3
    pre, post, comb = ops.hc_mixes(h.unsqueeze(0), fn, scale, base, 4, 20, 1e-6, EPS)
    d = lambda t: torch.empty(*t.shape[1:], dtype=torch.float, device=D).contiguous()
    pre_d, post_d, comb_d = d(pre), d(post), d(comb)
    part = torch.empty(R * K.HC_PART_CH * 25, dtype=torch.float, device=D)
    K.hc_mix(cu(h), cu(fn.half() if fn_half else fn), cu(scale), cu(base), pre_d, post_d, comb_d, part, 20, EPS, 1e-6)
    sync()
    for a, b in ((pre_d, pre), (post_d, post), (comb_d, comb)):
        torch.testing.assert_close(a.cpu(), b[0], rtol=3e-5, atol=1e-6)


@pytest.mark.parametrize("R,y_fp32", [(1, False), (1, True), (5, False)])
def test_hc_cn_apply_collapse_norm(R, y_fp32):
    h = torch.randn(R, 4, 5120, generator=g) * 2
    y = (torch.randn(R, 5120, generator=g) * 2).half()
    y = y.float() if y_fp32 else y
    post = torch.rand(R, 4, generator=g) * 2
    comb = torch.softmax(torch.randn(R, 4, 4, generator=g), -1)
    pre = torch.rand(R, 4, generator=g)
    w16 = (torch.randn(5120, generator=g) * 0.3 + 1).half()
    h_ref = ops.hc_post(y.float().unsqueeze(0), h.unsqueeze(0), post.unsqueeze(0), comb.unsqueeze(0))
    x_ref = ops.rmsnorm(ops.hc_pre(h_ref, pre.unsqueeze(0)).half(), w16, EPS)[0]
    hd, out = cu(h), torch.empty(R, 5120, dtype=torch.half, device=D)
    K.hc_cn(hd, cu(y), cu(post), cu(comb), cu(pre), cu(w16.float()), out, EPS, apply=True)
    sync()
    torch.testing.assert_close(hd.cpu(), h_ref[0], rtol=2e-6, atol=1e-6)
    torch.testing.assert_close(out.cpu().float(), x_ref.float(), rtol=3e-3, atol=2e-3)
    # collapse only (first layer / final norm): pre = (1, 0, 0, 0) gives stream 0 normed
    pre1 = torch.zeros(R, 4)
    pre1[:, 0] = 1
    x1 = ops.rmsnorm(h_ref[0, :, 0].half().unsqueeze(0), w16, EPS)[0]
    out1 = torch.empty(R, 5120, dtype=torch.half, device=D)
    K.hc_cn(hd, None, cu(post), cu(comb), cu(pre1), cu(w16.float()), out1, EPS, apply=False)
    sync()
    torch.testing.assert_close(out1.cpu().float(), x1.float(), rtol=3e-3, atol=2e-3)


@pytest.mark.parametrize("big", [68278.0, -300000.0])
def test_hc_cn_massive_activation_stays_finite(big):
    """A residual stream outlier beyond the fp16 range (seen at column 2455 of layer 30) used to turn the whole norm output
    into NaN (fp16(x) = inf, rstd = 0): the router then returned expert id E and the MoE kernels read out of range."""
    R = 3
    h = torch.randn(R, 4, 5120, generator=g) * 2
    h[1, 0, 2455] = big
    pre = torch.zeros(R, 4)
    pre[:, 0] = 1
    w16 = (torch.randn(5120, generator=g) * 0.3 + 1).half()
    hd, out = cu(h), torch.empty(R, 5120, dtype=torch.half, device=D)
    K.hc_cn(hd, None, cu(torch.zeros(R, 4)), cu(torch.zeros(R, 16)), cu(pre), cu(w16.float()), out, EPS, apply=False)
    sync()
    o = out.cpu().float()
    assert torch.isfinite(o).all()
    x = h[:, 0]
    x_ref = (w16.float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))).half().float()
    torch.testing.assert_close(o[1], x_ref[1], rtol=3e-3, atol=2e-3)
    torch.testing.assert_close(o[[0, 2]], x_ref[[0, 2]], rtol=3e-3, atol=2e-3)


# --------------------------------------------------------------------------------------- router
@pytest.mark.parametrize("R,wdt", [(1, torch.float), (3, torch.float), (9, torch.bfloat16), (5, torch.half)])
def test_router(R, wdt):
    E, KK = 384, 6
    gen = torch.Generator().manual_seed(R)
    x = (torch.randn(R, 5120, generator=gen) * 1.5).half()
    W = (torch.randn(E, 5120, generator=gen) * 0.03).to(wdt)
    bias = torch.randn(E, generator=gen) * 0.1
    wts_ref, idx_ref = ops.route(x, W.float(), bias, KK, "sqrtsoftplus", True, 1.5, 1.0)
    sel = torch.empty(R, KK, dtype=torch.int64, device=D)
    wts = torch.empty(R, KK, dtype=torch.half, device=D)
    wf = torch.empty(R, KK, dtype=torch.float, device=D)
    scores = torch.empty(R, E, dtype=torch.float, device=D)
    K.router(cu(x), cu(W), cu(bias), sel, wts, scores, KK, 1.5, 1.0, True, wf)
    sync()
    assert torch.equal(sel.cpu(), idx_ref)                       # random data: no near-ties, same order (descending key)
    torch.testing.assert_close(wf.cpu(), wts_ref, rtol=2e-4, atol=1e-6)
    torch.testing.assert_close(wts.cpu().float(), wts_ref.half().float(), rtol=2e-3, atol=1e-3)


# --------------------------------------------------------------------------------------- engram
@pytest.mark.parametrize("start,L,masked", [(0, 40, False), (5, 17, False), (0, 33, True)])
def test_engram_ids(start, L, masked):
    layers, ng, nh = [1, 14], 4, 8
    lay = EngramLayout.build(layers, None, ng, 16_000_000, nh, 256)
    V = 99092
    tmap = torch.randint(0, V, (129280,), generator=g)
    hasher = NgramHasher(lay, tmap.tolist(), 2, V, None, 1, 256, "cpu")
    ids = torch.randint(0, 129280, (1, start + L), generator=g)
    mask = (torch.rand(1, start + L, generator=g) > 0.2) if masked else None
    if start:
        hasher(ids[:, :start], 0, None if mask is None else mask[:, :start])
    ref = hasher(ids[:, start:], start, None if mask is None else mask[:, start:])[0]          # [L, nl, 24]
    cache = torch.full((256,), -1, dtype=torch.int64, device=D)
    if start:                                                    # history written by earlier chunks
        K.engram_comp(cu(ids[0, :start]), cu(tmap.int()), cache, None if mask is None else cu(mask[0, :start].to(torch.uint8)), 0)
    K.engram_comp(cu(ids[0, start:]), cu(tmap.int()), cache, None if mask is None else cu(mask[0, start:].to(torch.uint8)), start)
    rows = torch.empty(L, 2, 24, dtype=torch.int64, device=D)
    K.engram_hash(cache, cu(hasher.mult), cu(hasher.primes.flatten(1)), cu(hasher.offsets), rows, start, L, int(tmap[2]))
    sync()
    assert torch.equal(rows.cpu(), ref)


def test_engram_gate():
    R, hc, dim = 4, 4, 5120
    h = torch.randn(R, hc, dim, generator=g) * 2
    kv = (torch.randn(R, 5 * dim, generator=g) * 1.2).half()
    qk = torch.randn(hc, dim, generator=g)
    N = 8
    store = MemoryEngramStore([torch.zeros(N, 128, dtype=torch.uint8)], [torch.zeros(N, 8, dtype=torch.uint8)], "mxfp4")
    hid = torch.zeros(1, R, 24, dtype=torch.int64)
    ref = engram_inject(h.unsqueeze(0), hid, store, 0, lambda e: kv.unsqueeze(0).float(), qk, hc, dim, EPS)[0]
    hd = cu(h)
    K.engram_gate(hd, cu(kv), cu(qk), EPS)
    sync()
    torch.testing.assert_close(hd.cpu(), ref, rtol=2e-5, atol=2e-5)


# --------------------------------------------------------------------------------------- grouped MoE swiglu limit
def test_moe_silu_mul_limit():
    if EMU:
        pytest.skip("needs the device extension")
    from exllamav3.ext import exllamav3_ext as ext
    if not hasattr(ext, "exl3_moe_silu_mul_limit"):
        pytest.skip("extension built without exl3_moe_silu_mul_limit (rebuild)")
    n = 1 << 16
    gate = (torch.randn(n, generator=g) * 8).half()
    up = (torch.randn(n, generator=g) * 8).half()
    gate[:64] = torch.linspace(-40, 40, 64).half()
    up[:64] = torch.linspace(40, -40, 64).half()
    for L in (10.0, 0.0):
        gd, ud, od = cu(gate), cu(up), torch.empty(n, dtype=torch.half, device=D)
        ext.exl3_moe_silu_mul_limit(gd, ud, od, L)
        sync()
        if L > 0:                                              # reference of ops.swiglu, activation rounded to fp16 first
            gf, uf = gate.float().clamp(max=L), up.float().clamp(-L, L)
            ref = (torch.nn.functional.silu(gf).half() * uf.half())
        else:
            ref = torch.nn.functional.silu(gate.float()).half() * up
        d = (od.cpu().float() - ref.float()).abs()
        assert (d > 2 * 2 ** -10 * ref.float().abs().clamp_min(2 ** -14)).float().mean() < 1e-4, (L, d.max())


# --------------------------------------------------------------------------------------- engram, whole layer
class HashStore:
    """Row store whose row bytes are a pure function of the row id (a 384M-row table is never materialised)."""
    fmt = "mxfp4"

    def fetch(self, layer_pos, rows):
        r = rows.long().unsqueeze(1)
        b = (((r * 2654435761 + torch.arange(136).unsqueeze(0) * 40503 + layer_pos * 977) >> 5) & 255).to(torch.uint8)
        b[:, 128:] = (110 + (b[:, 128:].long() % 30)).to(torch.uint8)          # e8m0 scales around 2^-17 .. 2^12
        return b


@pytest.mark.parametrize("start,L", [(0, 12), (7, 3), (20, 1)])
def test_engram_layer_end_to_end(start, L):
    from exllamav3.modules.dsv41_fn.engram_dev import EngramDevice
    layers, ng, nh, hc, dim = [1, 14], 4, 8, 4, 5120
    lay = EngramLayout.build(layers, None, ng, 16_000_000, nh, 256)
    V = 99092
    gen = torch.Generator().manual_seed(21 + start)
    tmap = torch.randint(0, V, (129280,), generator=gen)
    from exllamav3.dsv41.engram import compute_multipliers
    mult = compute_multipliers(layers, ng, V)
    store = HashStore()
    ids = torch.randint(0, 129280, (1, start + L), generator=gen)
    ref_hasher = NgramHasher(lay, tmap.tolist(), 2, V, mult, 1, 128, "cpu")
    if start:
        ref_hasher(ids[:, :start], 0)
    ref_rows = ref_hasher(ids[:, start:], start)                                     # [1, L, nl, 24]
    dev = EngramDevice(lay, tmap.tolist(), 2, mult, 128, store, D)
    if start:
        dev.hash(cu(ids[0, :start]), 0)
    rows = dev.hash(cu(ids[0, start:]), start)
    sync()
    assert torch.equal(rows.cpu(), ref_rows[0])
    W = torch.randn(nh * 3 * 256, (hc + 1) * dim, generator=gen) * 0.01             # wkv as a dense [in, out] matrix
    qk = torch.randn(hc, dim, generator=gen)
    h = torch.randn(L, hc, dim, generator=gen)
    for lp in range(2):
        ref = engram_inject(h.unsqueeze(0), ref_rows[:, :, lp, :], store, lp, lambda e: (e.float() @ W).float(), qk, hc, dim, EPS)[0]
        hd = cu(h.clone())
        dev.inject(hd, rows[:, lp, :], lp, lambda e: (e.float() @ cu(W)).half(), cu(qk), EPS)
        sync()
        # wkv output is rounded to fp16 on the device (fp32 in the reference): one fp16 ulp of |v| in [4, 8) is 3.9e-3
        torch.testing.assert_close(hd.cpu(), ref, rtol=3e-3, atol=8e-3)


# --------------------------------------------------------------------------------------- delayed-pre mHC dataflow
def test_hc_chain_matches_block_semantics():
    """Three blocks of the DS4.1 hc dataflow (model.py _block_pre/_block_post: attention input collapses with the previous FFN
    site's pre, FFN input with the attention site's pre, final head with the last pre): reference ops vs the kernel chain
        cn(collapse only) ; per block: mix(attn), cn(apply attn y, pre = attn pre), mix(ffn), cn(apply ffn y, pre = ffn pre)
    with the sublayer outputs y = f(x) a fixed elementwise nonlinearity of the (fp16) sublayer input."""
    R, nlayer, hc, dim = 2, 3, 4, 5120
    gen = torch.Generator().manual_seed(77)
    P = []
    for l in range(nlayer):
        P.append(dict(a=hc_params(10 + l), f=hc_params(20 + l),
                      an=(torch.randn(dim, generator=gen) * 0.2 + 1).half(), fn_=(torch.randn(dim, generator=gen) * 0.2 + 1).half(),
                      sa=torch.rand(dim, generator=gen) + 0.5, sf=torch.rand(dim, generator=gen) + 0.5))
    nrm = (torch.randn(dim, generator=gen) * 0.2 + 1).half()
    h0 = torch.randn(R, hc, dim, generator=gen)

    def fy(x, s):                                           # sublayer stand-in on the fp16 input
        return (torch.tanh(x.float() * 0.7) * s).half()

    # ---- reference (ops.py, the way DSV41Graph chains blocks)
    h = h0.clone().unsqueeze(0)
    pre_mix = torch.zeros(1, R, hc)
    pre_mix[..., 0] = 1
    for l in range(nlayer):
        p = P[l]
        resid = h
        a_pre, a_post, a_comb = ops.hc_mixes(h, *p["a"], 4, 20, 1e-6, EPS)
        x = ops.rmsnorm(ops.hc_pre(h, pre_mix).half(), p["an"], EPS)
        h = ops.hc_post(fy(x, p["sa"]).float(), resid, a_post, a_comb)
        resid = h
        f_pre, f_post, f_comb = ops.hc_mixes(h, *p["f"], 4, 20, 1e-6, EPS)
        x = ops.rmsnorm(ops.hc_pre(h, a_pre).half(), p["fn_"], EPS)
        h = ops.hc_post(fy(x, p["sf"]).float(), resid, f_post, f_comb)
        pre_mix = f_pre
    ref = ops.rmsnorm(ops.hc_pre(h, pre_mix).half(), nrm, EPS)[0]

    # ---- kernels
    hd = cu(h0.clone())
    part = torch.empty(R * K.HC_PART_CH * 25, dtype=torch.float, device=D)
    mk = lambda n: torch.empty(R, n, dtype=torch.float, device=D)
    pre_mix = cu(torch.tensor([[1.0, 0, 0, 0]]).repeat(R, 1))
    x = torch.empty(R, dim, dtype=torch.half, device=D)
    K.hc_cn(hd, None, cu(torch.zeros(R, 4)), cu(torch.zeros(R, 16)), pre_mix, cu(P[0]["an"].float()), x, EPS, apply=False)
    for l in range(nlayer):
        p = P[l]
        a_pre, a_post, a_comb = mk(4), mk(4), mk(16)
        K.hc_mix(hd, cu(p["a"][0]), cu(p["a"][1]), cu(p["a"][2]), a_pre, a_post, a_comb, part, 20, EPS, 1e-6)
        y = fy(x.cpu(), p["sa"]) if not EMU else fy(x, p["sa"])
        xf = torch.empty(R, dim, dtype=torch.half, device=D)
        K.hc_cn(hd, cu(y), a_post, a_comb, a_pre, cu(p["fn_"].float()), xf, EPS, apply=True)
        f_pre, f_post, f_comb = mk(4), mk(4), mk(16)
        K.hc_mix(hd, cu(p["f"][0]), cu(p["f"][1]), cu(p["f"][2]), f_pre, f_post, f_comb, part, 20, EPS, 1e-6)
        y = fy(xf.cpu(), p["sf"]) if not EMU else fy(xf, p["sf"])
        nxt = P[l + 1]["an"] if l + 1 < nlayer else nrm
        x = torch.empty(R, dim, dtype=torch.half, device=D)
        K.hc_cn(hd, cu(y), f_post, f_comb, f_pre, cu(nxt.float()), x, EPS, apply=True)
    sync()
    torch.testing.assert_close(x.cpu().float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_prepared_launch_replays_and_updates_scalars():
    """K.prepare(): argument pack built once; replay == wrapper call, and set() changes a scalar in place (compress start)."""
    ratio, R = 2, 6
    kv, sc = torch.randn(R, 512, generator=g), torch.randn(R, 512, generator=g)
    kst, sst = torch.randn(ratio, 512, generator=g), torch.randn(ratio, 512, generator=g)
    w = cu((torch.randn(512, generator=g) * 0.4 + 1).half().float())
    kv_d, sc_d = cu(kv), cu(sc)
    out_a = torch.zeros(R // ratio + 1, 512, dtype=torch.half, device=D)
    K.compress(kv_d, sc_d, cu(kst), cu(sst), w, out_a, 4, ratio, EPS)
    out_b = torch.zeros(R // ratio + 1, 512, dtype=torch.half, device=D)
    kst_b, sst_b = cu(kst), cu(sst)
    pl = K.prepare("ds41_compress", (R // ratio,), 32, (kv_d, sc_d, kst_b, sst_b, w, out_b, 0, ratio, float(EPS)))
    pl.set(6, 4)                                   # start = 4 (arg index 6)
    pl()
    sync()
    assert torch.equal(out_a[:R // ratio].cpu(), out_b[:R // ratio].cpu())
