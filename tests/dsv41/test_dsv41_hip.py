"""
Gate 3: the tiny DS4.1 model on the HIP device (gfx1151) against the CPU result.

Opt-in (never collected by a plain pytest run). Under the GPU lock, after unloading the lanes:
    flock -w 14400 ./ds41_work/gpu.lock bash -c 'curl -s 127.0.0.1:8271/unload; \
        source ds41port/pyenv.sh; DSV41_DEVICE=cuda python -m pytest -q tests/dsv41/test_dsv41_hip.py'
DSV41_DEVICE=cpu runs the same file CPU-vs-CPU (script check only).

What can be compared exactly: elementwise ops with identical inputs (quantisers: bitwise). What cannot:
anything downstream of a different GEMM accumulation order, because the fp8/fp4 fake-quant steps are
step functions (a last-bit difference flips ~1 element in 1e6 by one grid step). End to end therefore:
greedy ids equal, logits within LOGIT_TOL, and the count of tap elements off by > 1e-4 is reported.
"""
import os, sys
import numpy as np
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import *
from test_dsv41_e2e import setup, build_mine
from exllamav3.dsv41 import ops, quant

DEV = os.environ.get("DSV41_DEVICE", "")
if not DEV:
    pytest.skip("GPU gate is opt-in: run under the GPU lock with DSV41_DEVICE=cuda (or =cpu for a script check)",
                allow_module_level=True)
if DEV.startswith("cuda") and not torch.cuda.is_available():
    pytest.skip("no HIP device", allow_module_level=True)
LOGIT_TOL = 2e-2          # logits are O(5)
OP_TOL = 1e-5


def on(x):
    return x.to(DEV)


def test_device_is_gfx1151_or_cpu():
    if DEV == "cpu":
        return
    p = torch.cuda.get_device_properties(0)
    print("device:", p.name, getattr(p, "gcnArchName", "?"))
    assert "gfx1151" in getattr(p, "gcnArchName", "gfx1151")


# ------------------------------------------------------------------------ per-op, identical inputs
def adv(seed=0):
    from test_dsv41_ops import adversarial
    return adversarial(seed=seed)


def test_quantisers_bitwise():
    x = adv()
    a = quant.act_quant_fp8_(x.clone(), 32)
    b = quant.act_quant_fp8_(on(x.clone()), 32).cpu()
    assert torch.equal(a, b)
    for e4, blk in ((False, 32), (True, 16)):
        xx = x.clamp(-2000, 2000) if e4 else x
        a = quant.fp4_act_quant_(xx.clone(), blk, e4m3_scale=e4)
        b = quant.fp4_act_quant_(on(xx.clone()), blk, e4m3_scale=e4).cpu()
        assert torch.equal(a, b), e4


def test_ops_identical_inputs():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 9, 4, 64, generator=g)
    w = torch.randn(64, generator=g)
    assert (ops.rmsnorm(x, w, 1e-20) - ops.rmsnorm(on(x), on(w), 1e-20).cpu()).abs().max() <= OP_TOL
    fc = ops.rope_tables(16, 40, 64, 40000.0, 16.0, 32, 1)
    a, b = x.clone(), on(x.clone())
    ops.rope_apply_(a[..., -16:], fc[3:12]); ops.rope_apply_(b[..., -16:], on(fc[3:12]))
    assert (a - b.cpu()).abs().max() <= OP_TOL
    ops.rope_apply_(a[..., -16:], fc[3:12], True); ops.rope_apply_(b[..., -16:], on(fc[3:12]), True)
    assert (a - b.cpu()).abs().max() <= OP_TOL
    mixes = torch.randn(3, 5, 24, generator=g)
    sc, bs = torch.tensor([0.8, 0.5, 0.9]), torch.randn(24, generator=g)
    for u, v in zip(ops.hc_split_sinkhorn(mixes, sc, bs, 4, 20, 1e-6), ops.hc_split_sinkhorn(on(mixes), on(sc), on(bs), 4, 20, 1e-6)):
        assert (u - v.cpu()).abs().max() <= OP_TOL
    B, S, H, D, N, K = 2, 6, 4, 64, 30, 12
    q, kv = torch.randn(B, S, H, D, generator=g), torch.randn(B, N, D, generator=g)
    sink = torch.randn(H, generator=g)
    idx = torch.randint(-1, N, (B, S, K), generator=g, dtype=torch.int32)
    idx[0, 0] = -1
    a = ops.sparse_attention(q, kv, sink, idx, 0.125)
    b = ops.sparse_attention(on(q), on(kv), on(sink), on(idx), 0.125).cpu()
    assert (a - b).abs().max() <= 1e-4 * max(1.0, a.abs().max().item())
    assert b[0, 0].abs().max() == 0
    lg = torch.randn(1, 43, 43, generator=g)
    clen = (torch.arange(1, 44) // 1).unsqueeze(-1)
    lg.masked_fill_(torch.arange(43) >= clen, -torch.inf)
    assert torch.equal(ops.select_candidate_blocks(lg, clen, 3, 2),
                       ops.select_candidate_blocks(on(lg), on(clen), 3, 2).cpu())
    xx = torch.randn(37, 64, generator=g)
    gw, gb = torch.randn(8, 64, generator=g), 0.3 * torch.randn(8, generator=g)
    wa, ia = ops.route(xx, gw, gb, 2, "sqrtsoftplus", True, 1.5)
    wb, ib = ops.route(on(xx), on(gw), on(gb), 2, "sqrtsoftplus", True, 1.5)
    assert torch.equal(ia, ib.cpu()) and (wa - wb.cpu()).abs().max() <= OP_TOL


def test_ext_hc_mix_matches_our_hyperconnection():
    """The shipped fused hc_mix kernel (GLM/V4 HyperConnection, hc_mult 4, fp32 streams) has the same math as
    our hc mixing: post and comb must agree, and its `collapsed` = sum_c pre_own[c] * stream[c]. (V4.1 collapses
    with the PREVIOUS sublayer's pre, so only post/comb are reusable without a kernel change.)"""
    if DEV == "cpu":
        pytest.skip("HIP only")
    from exllamav3.ext import exllamav3_ext as ext
    if not (hasattr(ext, "hc_mix") and ext.hc_mix_supported(torch.device(DEV).index or 0)):
        pytest.skip("hc_mix not supported on this device")
    g = torch.Generator().manual_seed(9)
    R, H, D = 7, 4, 64
    streams = torch.randn(R, H, D, generator=g)
    fn = torch.randn(24, H * D, generator=g) * 0.02
    base, scale = torch.randn(24, generator=g) * 0.5, torch.tensor([0.8, 0.6, 0.7])
    pre, post, comb = ops.hc_mixes(streams.unsqueeze(0), fn, scale, base, 4, 20, 1e-6, 1e-20)
    s_, fn_, base_, scale_ = on(streams.contiguous()), on(fn), on(base), on(scale)
    chunks = ext.hc_mix_num_chunks(R, H * D)
    partials = torch.empty(R, chunks, 2 * H + H * H + 1, dtype=torch.float, device=DEV)
    kpost = torch.empty(R, H, dtype=torch.float, device=DEV)
    kcomb = torch.empty(R, H, H, dtype=torch.float, device=DEV)
    kcoll = torch.empty(R, D, dtype=torch.half, device=DEV)
    ext.hc_mix(s_, fn_, base_, scale_, 1e-20, 1e-6, 20, partials, kpost, kcomb, kcoll)
    assert (kpost.cpu() - post[0]).abs().max() <= 1e-4
    assert (kcomb.cpu() - comb[0]).abs().max() <= 1e-4
    want = ops.hc_pre(streams.unsqueeze(0), pre)[0]
    assert (kcoll.float().cpu() - want).abs().max() <= 5e-3 * max(1.0, want.abs().max().item())


# ------------------------------------------------------------------------ end to end (teacher forced)
# Free-running device vs CPU diverges chaotically: every fp8/fp4 fake-quant and every top-k is a step
# function, a last-bit GEMM difference flips ~1 element and the flip is amplified by the next layers (measured on the
# tiny model: 1e-6 in layers 0-1, a 6e-4 flip in layer 2, 1.3 in layer 7, one index of 172 differing in a top-k).
# So the gate is per layer with identical inputs: the CPU layer input, the CPU state before the step, and the CPU
# shared-attention hand-offs are fed to the device, the device layer output is compared to the CPU's.
import copy
from exllamav3.dsv41.model import SharedRt
from exllamav3.dsv41.engram import engram_inject
import torch.nn.functional as F

LAYER_TOL = 2e-2          # abs, on streams of scale ~5-10: allows an isolated fake-quant flip (~1e-3), not a wrong op
FLIP_FRAC = 2e-2          # at most 2% of all compared elements of a run may differ by > 1e-4


def state_to(src, dst):
    for name in ("win", "comp_kv", "comp_kv_state", "comp_score_state", "idx_k"):
        for k, v in getattr(src, name).items():
            getattr(dst, name)[k].copy_(v.to(DEV))
    if src.hasher is not None:
        dst.hasher.cache.copy_(src.hasher.cache.to(DEV))


def teacher_layers(dev, cpu_taps, ids, start_pos, dst, stats):
    """Run every trunk layer on the device from the CPU's layer inputs; compare outputs."""
    cfg = dev.cfg
    ids = ids.to(DEV)
    hashes = dst.hasher(ids, start_pos) if dst.hasher is not None else None
    rt = SharedRt()
    taps = {}
    for l in range(cfg.num_hidden_layers):
        w = dev.lw[l]
        if w.engram is not None:
            h0 = on(cpu_taps[f"L{l}.engram_in"])
            h1 = engram_inject(h0, hashes[:, :, w.engram.pos, :], dev.engram_store, w.engram.pos, w.engram.wkv,
                               w.engram.qk, cfg.hc_mult, cfg.hidden_size, cfg.rms_norm_eps)
            stats.append((f"L{l}.engram", cpu_taps[f"L{l}.in"], h1.cpu()))
        h = on(cpu_taps[f"L{l}.in"])
        pre = on(cpu_taps[f"L{l}.pre"])
        t = {}
        out, _ = dev._block(l, h, start_pos, pre, dst, rt, t)
        for k in (f"L{l}.attn", f"L{l}.ffn", f"L{l}.out", f"L{l}.comp"):
            if k in cpu_taps and k.split(".")[1] in ("attn", "ffn", "out") or k in t:
                if k in t and k in cpu_taps:
                    stats.append((k, cpu_taps[k], t[k].cpu()))
        for k in (f"L{l}.topk", f"L{l}.route.idx"):
            if k in cpu_taps and k in t:
                stats.append((k, cpu_taps[k], t[k].cpu()))
        taps[l] = out
    return taps


def check(stats, label, cfg):
    """Float tensors: max abs diff <= LAYER_TOL * scale and a bounded fraction of elements off by > 1e-4. Index
    tensors (top-k selections, routing) may differ at near-tie scores (the selection is a step function of
    last-bit differences; the reference has the same tie freedom): such a token row is excluded from the float
    comparison of the layers that consume that selection (up to the next index source), and counted."""
    bad_rows = {}                                   # layer -> set of token rows whose top-k selection differs
    idx_bad, idx_tot = 0, 0
    for name, a, b in stats:
        if a.dtype in (torch.int32, torch.int64):
            idx_bad += int((a != b).sum()); idx_tot += a.numel()
            if name.endswith(".topk"):
                l = int(name[1:].split(".")[0])
                m = (a != b).flatten(2).any(-1) if a.dim() >= 3 else (a != b).any(-1)
                bad_rows[l] = m                      # [B, S]
    def excluded(layer):
        keep = None
        for l, m in bad_rows.items():
            nxt = min([x for x in cfg.index_source_layer_ids if x > l] + [cfg.num_hidden_layers])
            if l <= layer < nxt:
                keep = m if keep is None else (keep | m)
        return keep
    worst, nflip, ntot = 0.0, 0, 0
    for name, a, b in stats:
        if a.dtype in (torch.int32, torch.int64):
            continue
        d = (a - b).abs()
        if name != "logits" and name[0] == "L":
            ex = excluded(int(name[1:].split(".")[0]))
            if ex is not None and d.dim() >= 2 and d.shape[:2] == ex.shape:
                d = d.masked_fill(ex.view(*ex.shape, *([1] * (d.dim() - 2))), 0)
        scale = max(1.0, a.abs().max().item())
        assert d.max().item() <= LAYER_TOL * scale, (label, name, d.max().item(), scale)
        worst = max(worst, d.max().item()); nflip += int((d > 1e-4).sum()); ntot += d.numel()
    mism = [(n, int((a != b).sum())) for n, a, b in stats if a.dtype in (torch.int32, torch.int64) and (a != b).any()]
    if mism:
        print(f"   [{label}] index mismatches:", mism)
    return worst, nflip, ntot, idx_bad, idx_tot


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_e2e_teacher_forced_device_vs_cpu(seed):
    cfg, tok, tmap, tables, sd = setup(True, seed)
    cpu = build_mine(cfg, sd, tables, tmap)
    dev = build_mine(cfg, sd, tables, tmap, device=DEV)
    sc = cpu.new_state(1, 256)
    g = torch.Generator().manual_seed(seed + 50)
    S, steps = 43, 16
    ids = torch.randint(0, cfg.vocab_size, (1, S + steps), generator=g)
    summary = []
    with torch.inference_mode():
        for start, chunk in [(0, ids[:, :S])] + [(i, ids[:, i:i + 1]) for i in range(S, S + steps)]:
            dst = dev.new_state(1, 256)
            state_to(sc, dst)                                   # CPU state BEFORE this step
            tc = {}
            lc, mc = cpu.forward(chunk, start, sc, tc)
            stats = []
            dev_h = teacher_layers(dev, tc, chunk, start, dst, stats)
            # head: final collapse + norm + logits from the CPU's last-layer input/pre
            fin = on(tc["final.in"]); fpre = on(tc["final.pre"])
            x = ops.rmsnorm(ops.hc_pre(fin, fpre), dev.norm_w, cfg.rms_norm_eps)
            ld = F.linear(x[:, -1].float(), dev.head_w)
            stats.append(("logits", lc, ld.cpu()))
            r = check(stats, f"step {start}", cfg)
            assert torch.equal(lc.argmax(-1), ld.cpu().argmax(-1)), start
            # dspark on identical inputs / state (win rings of the mtp stages were copied)
            dd = dev.mtp.forward_spec(on(lc.argmax(-1)), on(mc), start, dst)
            cc = cpu.mtp.forward_spec(lc.argmax(-1), mc, start, sc)
            if start > 0:
                assert (cc[1] - dd[1].cpu()).abs().max() <= 0.1, (start, (cc[1] - dd[1].cpu()).abs().max())
                assert (cc[2] - dd[2].cpu()).abs().max() <= 0.05
            summary.append((start,) + r)
    tot_flip, tot_el = sum(r[2] for r in summary), sum(r[3] for r in summary)
    assert tot_flip <= FLIP_FRAC * tot_el, (tot_flip, tot_el)
    assert sum(r[4] for r in summary) <= 0.05 * sum(r[5] for r in summary), (sum(r[4] for r in summary), sum(r[5] for r in summary))   # aggregated: one flip in a decode row spreads over that row
    print(f"seed {seed}: (step, worst, flips>1e-4, elements, idx mismatches, idx total)")
    for r in summary[:3] + summary[-2:]:
        print("   ", r)
    print("    overall worst %.2e, flips %d / %d, idx mismatches %d / %d" % (
        max(r[1] for r in summary), sum(r[2] for r in summary), sum(r[3] for r in summary),
        sum(r[4] for r in summary), sum(r[5] for r in summary)))
