"""Per-op gates: every op of the Kyojin DS4.1 graph against the official reference (vendored model.py /
engram.py + CPU kernel shim) and, for the number formats, against ml_dtypes (independent oracle).
ml_dtypes / gguf live in ds41port/pylib (PYTHONPATH), tests needing them skip otherwise."""
import os, sys
import numpy as np
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
from harness import _fp8_block_quant, _fp4_row_quant
from exllamav3.dsv41 import ops, quant
from exllamav3.dsv41.engram import (NgramHasher, EngramLayout, compute_multipliers, build_compressed_token_map,
                                    is_prime)

R = load_reference()
K = R.kernel
ml = pytest.importorskip("ml_dtypes")


def adversarial(n_rows=512, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_rows, n, generator=g) * torch.exp2(torch.randint(-12, 12, (n_rows, 1), generator=g).float())
    # exact powers of two, ties of every grid, zeros, denormal-ish
    x[0] = 0
    x[1] = 448.0
    x[2] = torch.tensor([2.0 ** k for k in range(-20, 12)] * 2)[:n]
    x[3, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    x[4] = x[4].sign() * 1e-30
    x[5] = 1e-4
    x[6, 0] = 1e3     # amax/6 stays inside e4m3 range (overflow is out of domain: the model bounds it by RMSNorm)
    return x


# ----------------------------------------------------------------------------- number formats
def oracle_fp8(x: np.ndarray, block=32):
    x = x.astype(np.float32)
    xb = x.reshape(x.shape[0], -1, block)
    amax = np.maximum(np.abs(xb).max(-1, keepdims=True), np.float32(1e-4))
    t = (amax * np.float32(1.0 / 448.0)).astype(np.float32)
    s = (2.0 ** np.ceil(np.log2(t.astype(np.float64)))).astype(np.float32)
    q = np.clip(xb / s, -448, 448).astype(np.float32).astype(ml.float8_e4m3fn).astype(np.float32) * s
    return q.reshape(x.shape)


def test_act_quant_fp8_vs_ml_dtypes_and_shim():
    x = adversarial()
    mine = quant.act_quant_fp8_(x.clone(), 32)
    shim = K.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
    orc = torch.from_numpy(oracle_fp8(x.numpy()))
    assert torch.equal(mine, orc)
    assert torch.equal(mine, shim)
    # bf16 input path
    xb = x.to(torch.bfloat16)
    assert torch.equal(quant.act_quant_fp8_(xb.clone(), 32), K.act_quant(xb.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True))


def oracle_fp4(x: np.ndarray, block, e4m3):
    x = x.astype(np.float32)
    xb = x.reshape(x.shape[0], -1, block)
    amax = np.abs(xb).max(-1, keepdims=True)
    if e4m3:
        amax = np.maximum(amax, np.float32(6 * 2.0 ** -9))
        s = (amax / np.float32(6.0)).astype(np.float32).astype(ml.float8_e4m3fn).astype(np.float32)
    else:
        amax = np.maximum(amax, np.float32(6 * 2.0 ** -126))
        t = (amax * np.float32(1.0 / 6.0)).astype(np.float32)
        s = (2.0 ** np.ceil(np.log2(t.astype(np.float64)))).astype(np.float32)
    q = np.clip(xb / s, -6, 6).astype(np.float32).astype(ml.float4_e2m1fn).astype(np.float32) * s
    return q.reshape(x.shape)


@pytest.mark.parametrize("e4m3,block", [(False, 32), (True, 16)])
def test_fp4_act_quant_vs_ml_dtypes_and_shim(e4m3, block):
    x = adversarial(n=64)
    if e4m3:
        x = x.clamp(-2000, 2000)          # amax/6 inside e4m3 range; overflow is out of domain (RMSNorm bounds it)
    mine = quant.fp4_act_quant_(x.clone(), block, e4m3_scale=e4m3)
    shim = K.fp4_act_quant(x.clone(), block, True, torch.float8_e4m3fn if e4m3 else torch.float8_e8m0fnu)
    orc = torch.from_numpy(oracle_fp4(x.numpy(), block, e4m3))
    assert torch.equal(mine, orc)
    assert torch.equal(mine, shim)


def test_fp4_decode_matches_convert_py_table():
    # convert.py FP4_TABLE: low nibble first, sign = bit 3
    rows, cols = 7, 64
    g = torch.Generator().manual_seed(1)
    packed = torch.randint(0, 256, (rows, cols // 2), generator=g, dtype=torch.uint8)
    scale = torch.randint(110, 135, (rows, cols // 32), generator=g, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    tab = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    exp = torch.stack([tab[(packed & 15).long()], tab[(packed >> 4).long()]], -1).flatten(-2)
    exp = exp * scale.float().repeat_interleave(32, -1)
    assert torch.equal(quant.dequant_fp4_rows(packed.view(torch.int8), scale), exp)


def test_fp8_block_dequant():
    w = torch.randn(70, 100) * 3
    q, s = _fp8_block_quant(w)
    d = quant.dequant_fp8_block(q, s)
    # explicit per-block reference
    ref = torch.zeros(70, 100)
    for bi in range(0, 70, 32):
        for bj in range(0, 100, 32):
            ref[bi:bi + 32, bj:bj + 32] = q[bi:bi + 32, bj:bj + 32].float() * s[bi // 32, bj // 32].float()
    assert torch.equal(d, ref)
    assert (d - w).abs().max() <= (w.abs().max() * 0.07)


# ----------------------------------------------------------------------------- gemms
@pytest.mark.parametrize("kind", ["fp8", "fp4"])
def test_quantlin_vs_kernel_semantics(kind):
    """Dequant+matmul (what we run) vs the kernel's per-32-block accumulate-then-scale loop."""
    from exllamav3.dsv41.linear import QuantLin
    g = torch.Generator().manual_seed(2)
    N, Kd, M = 96, 128, 5
    w = torch.randn(N, Kd, generator=g) * 0.1
    x = torch.randn(M, Kd, generator=g)
    if kind == "fp8":
        wq, ws = _fp8_block_quant(w)
        w_deq = quant.dequant_fp8_block(wq, ws)
    else:
        wq, ws = _fp4_row_quant(w)
        w_deq = quant.dequant_fp4_rows(wq, ws)
    y = QuantLin(w_deq)(x)
    # kernel loop
    xq = x.clone(); quant.act_quant_fp8_(xq, 32)
    a8, a_s = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, False)
    out = torch.zeros(M, N)
    for kb in range(Kd // 32):
        a = a8[:, kb * 32:(kb + 1) * 32].float()
        if kind == "fp8":
            b = wq[:, kb * 32:(kb + 1) * 32].float()
            sb = ws.float().repeat_interleave(32, 0)[:N, kb]
        else:
            lo = wq.view(torch.uint8)[:, kb * 16:(kb + 1) * 16]
            tab = K.FP4_ALL
            b = torch.stack([tab[(lo & 15).long()], tab[(lo >> 4).long()]], -1).flatten(-2)
            sb = ws.float()[:, kb]
        out += (a @ b.T) * a_s.float()[:, kb:kb + 1] * sb[None, :]
    assert (y - out).abs().max() <= 1e-5 * out.abs().max()
    shim_y = (K.fp8_gemm(a8, a_s, wq, ws, torch.float8_e8m0fnu, 32) if kind == "fp8"
              else K.fp4_gemm(a8, a_s, wq.view(torch.float4_e2m1fn_x2) if False else wq.view(torch.uint8).view(torch.float4_e2m1fn_x2), ws, torch.float8_e8m0fnu, 32))
    assert torch.equal(y, shim_y)


# ----------------------------------------------------------------------------- hc / rope / attention helpers
def test_hc_split_sinkhorn_vs_shim_and_properties():
    g = torch.Generator().manual_seed(3)
    mixes = torch.randn(3, 11, 24, generator=g)
    scale = torch.tensor([0.8, 0.5, 0.9]); base = torch.randn(24, generator=g)
    a = ops.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    b = K.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    for u, v in zip(a, b):
        assert (u - v).abs().max() <= 1e-6
    comb = a[2]
    assert (comb.sum(-1) - 1).abs().max() < 1e-2 and (comb.sum(-2) - 1).abs().max() < 1e-5   # doubly stochastic


def test_rope_vs_official_bitwise():
    for orig, theta in [(0, 10000.0), (64, 40000.0), (4096, 160000.0)]:
        fc_ref = R.model.precompute_freqs_cis(64, 300, orig, theta, 16, 32, 1)
        fc = ops.rope_tables(64, 300, orig, theta, 16.0, 32, 1)
        assert torch.equal(fc, fc_ref)
        x = torch.randn(2, 17, 4, 64)
        a, b = x.clone(), x.clone()
        R.model.apply_rotary_emb(a, fc_ref[5:22]); ops.rope_apply_(b, fc[5:22])
        assert torch.equal(a, b)
        R.model.apply_rotary_emb(a, fc_ref[5:22], True); ops.rope_apply_(b, fc[5:22], True)
        assert torch.equal(a, b) and (a - x).abs().max() < 1e-5      # inverse undoes it


def test_window_idxs_vs_official():
    for (win, B, S, sp) in [(8, 2, 5, 0), (8, 1, 30, 0), (8, 2, 1, 3), (8, 2, 1, 8), (8, 1, 1, 37), (128, 1, 1, 130)]:
        R.model.get_window_topk_idxs.cache_clear()
        assert torch.equal(ops.window_idxs(win, B, S, sp), R.model.get_window_topk_idxs(win, B, S, sp))


def test_sparse_attention_vs_shim_and_dense():
    g = torch.Generator().manual_seed(4)
    B, S, H, D, N, Kk = 2, 6, 4, 64, 30, 12
    q, kv = torch.randn(B, S, H, D, generator=g), torch.randn(B, N, D, generator=g)
    sink = torch.randn(H, generator=g)
    idx = torch.randint(-1, N, (B, S, Kk), generator=g, dtype=torch.int32)
    idx[0, 0] = -1                                   # fully masked row -> zeros
    a = ops.sparse_attention(q, kv, sink, idx, 0.125)
    b = K.sparse_attn(q, kv, sink, idx, 0.125)
    assert torch.equal(a, b) and a[0, 0].abs().max() == 0
    # independent dense softmax (float64) incl. the sink as an extra zero-value key
    for bb in range(B):
        for s in range(1, S):
            ids = idx[bb, s][idx[bb, s] >= 0].long()
            for h in range(H):
                sc = (kv[bb, ids].double() @ q[bb, s, h].double()) * 0.125
                z = torch.cat([sc, sink[h].double().view(1)])
                p = torch.softmax(z, 0)[:-1]
                o = (p[:, None] * kv[bb, ids].double()).sum(0)
                assert (o - a[bb, s, h].double()).abs().max() < 1e-5


def test_sparse_attention_empty_compressed_pool():
    """Context shorter than one compression block: the compressed pool has no rows."""
    g = torch.Generator().manual_seed(5)
    B, S, H, D, N = 1, 3, 4, 64, 5
    q, kv = torch.randn(B, S, H, D, generator=g), torch.randn(B, N, D, generator=g)
    sink = torch.randn(H, generator=g)
    idx = torch.randint(-1, N, (B, S, 6), generator=g, dtype=torch.int32)
    empty = torch.empty(B, 0, D)
    assert torch.equal(ops.sparse_attention(q, kv, sink, idx, 0.125, empty), ops.sparse_attention(q, kv, sink, idx, 0.125))


def test_select_candidate_blocks_vs_official():
    g = torch.Generator().manual_seed(5)
    for S, ratio, tb, bs in [(23, 1, 3, 2), (43, 2, 3, 2), (30, 1, 100, 4), (9, 2, 2, 8)]:
        n = S // ratio
        logits = torch.randn(1, S, n, generator=g)
        clen = (torch.arange(1, S + 1) // ratio).unsqueeze(-1)
        logits.masked_fill_(torch.arange(n) >= clen, -torch.inf)
        assert torch.equal(ops.select_candidate_blocks(logits, clen, tb, bs),
                           R.model.select_candidate_blocks(logits, clen, tb, bs))
    lg = torch.randn(1, 1, 17)
    assert torch.equal(ops.select_candidate_blocks(lg, 17, 3, 4), R.model.select_candidate_blocks(lg, 17, 3, 4))


def test_router_vs_official_gate():
    cfg = tiny_config()
    g = torch.Generator().manual_seed(6)
    torch.set_default_dtype(torch.float32)
    gate = R.model.Gate(0, R.model.ModelArgs(**{"dim": 64, "n_routed_experts": 8, "n_activated_experts": 2,
                                                 "route_scale": 1.5, "score_func": "sqrtsoftplus"}))
    with torch.no_grad():
        gate.weight.copy_(torch.randn(8, 64, generator=g)); gate.bias.copy_(0.3 * torch.randn(8, generator=g))
    x = torch.randn(37, 64, generator=g)
    rw, ri = gate(x)
    mw, mi = ops.route(x, gate.weight, gate.bias, 2, "sqrtsoftplus", True, 1.5)
    assert torch.equal(ri, mi) and torch.equal(rw, mw)


def test_swiglu_clamp_vs_official_expert():
    torch.set_default_dtype(torch.float32)
    ex = R.model.Expert(64, 64, dtype=torch.float32, swiglu_limit=2.0)
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for w in (ex.w1, ex.w2, ex.w3):
            w.weight.copy_(torch.randn_like(w.weight) * 0.3)
    x = torch.randn(9, 64, generator=g) * 4
    wts = torch.rand(9, 1, generator=g)
    from exllamav3.dsv41.experts import ExpertHandle
    from exllamav3.dsv41.linear import DenseLin
    h = ExpertHandle(0, 0, DenseLin(ex.w1.weight.data), DenseLin(ex.w3.weight.data), DenseLin(ex.w2.weight.data))
    assert torch.equal(h.ffn(x, wts, 2.0), ex(x, wts))
    assert torch.equal(h.ffn(x, None, 2.0), ex(x))


# ----------------------------------------------------------------------------- engram
def test_engram_layout_multipliers_tokenmap_vs_official():
    cfg = tiny_config()
    tok = FakeTok(cfg.vocab_size)
    rmap, rn = R.engram.build_compressed_token_map(tok)
    mmap_, mn = build_compressed_token_map(tok)
    assert rmap == mmap_ and rn == mn and mn < cfg.vocab_size            # non-trivial compression
    args = ref_args(cfg)
    args.engram_compressed_vocab_size = mn
    rl = R.engram.EngramLayout.from_args(args)
    ml_ = EngramLayout.build(cfg.engram_layer_ids, None, cfg.engram_max_ngram_size, cfg.engram_vocab_size,
                             cfg.engram_n_heads, cfg.engram_head_dim)
    assert rl.primes == ml_.primes
    assert torch.equal(R.engram.compute_hash_multipliers(rl.layer_ids, 4, mn),
                       torch.from_numpy(compute_multipliers(cfg.engram_layer_ids, 4, mn)))
    for p in [2, 3, 97, 99991, 16000057, 2 ** 31 - 1, 561, 1105, 3215031751]:
        assert is_prime(p) == R.engram.isprime(p)


def test_hash_parity_100k_sequences():
    """design gate 4c: bit-identical hash ids over 100k random token sequences (prefill+decode paths, dead tokens)."""
    cfg = tiny_config()
    tok = FakeTok(cfg.vocab_size)
    tmap, nv = build_compressed_token_map(tok)
    cfg.engram_compressed_vocab_size = nv
    lay = EngramLayout.build(cfg.engram_layer_ids, None, cfg.engram_max_ngram_size, cfg.engram_vocab_size,
                             cfg.engram_n_heads, cfg.engram_head_dim)
    args = ref_args(cfg, max_seq=64, max_batch=100)
    rh = R.engram.NgramHashState(args, R.engram.EngramLayout.from_args(args), tok)
    mh = NgramHasher(lay, tmap, cfg.engram_pad_token_id, nv, None, 100, 64)
    g = torch.Generator().manual_seed(11)
    total = 0
    for rep in range(500):                    # 500 x 100 = 50k sequences x 25 positions (prefill 17 + 8 decode)
        rh.cache.fill_(-1); mh.reset()
        ids = torch.randint(0, cfg.vocab_size, (100, 25), generator=g)
        mask = torch.rand(100, 25, generator=g) > 0.1        # dead tokens (image spans)
        a = rh(ids[:, :17], 0, mask[:, :17]); b = mh(ids[:, :17], 0, mask[:, :17])
        assert torch.equal(a, b)
        for i in range(17, 25):
            a = rh(ids[:, i:i + 1], i, mask[:, i:i + 1]); b = mh(ids[:, i:i + 1], i, mask[:, i:i + 1])
            assert torch.equal(a, b)
        total += 100
    # second pass on different seeds, no mask, long sequences
    for rep in range(500):
        rh.cache.fill_(-1); mh.reset()
        ids = torch.randint(0, cfg.vocab_size, (100, 60), generator=g)
        assert torch.equal(rh(ids, 0), mh(ids, 0))
        total += 100
    assert total == 100_000
