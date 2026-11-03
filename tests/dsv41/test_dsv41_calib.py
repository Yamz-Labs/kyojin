"""
Calibration forward of the DS4.1 port (CPU, fp32, tiny random model, official checkpoint layout).

  * the block-by-block runner (CalibGraph.run_block, state carried per row in `ho`) reproduces the full-model
    forward: every block input / the final hidden, for both MoE modes (routed assignment / all experts)
  * the Hessians it captures equal Hessians computed independently from the taps of the full forward
  * expert assignment: counts, per-expert Hessian counts, w2 Hessian from the intermediate activation
  * MTP (dspark) stage calibration equals the model's forward_spec decode step at the same anchor
"""
import os, sys
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware import require
require("ext")
from harness import *
from test_dsv41_e2e import setup, build_mine
from exllamav3.dsv41 import ops, quant
from exllamav3.dsv41.calib import (CalibGraph, HessianSet, RecordingLin, wo_a_recorder, new_handoff, RowEmbed,
                                   moe_calibrate, expert_ffn, DSparkCalib, ExpertAssignment, moe_route)
from exllamav3.dsv41.engram import engram_inject
from exllamav3.dsv41.model import _LW


def recorded_layer(mine, l, hset):
    """Copy of the model's layer weights with every quantizable linear wrapped by a recorder."""
    cfg = mine.cfg
    w = mine.lw[l]
    c = _LW()
    c.__dict__.update(w.__dict__)
    p = f"layers.{l}" if l < cfg.num_hidden_layers else f"mtp.{l - cfg.num_hidden_layers}"
    for n in ("wq_a", "wq_b", "wkv", "wo_b"):
        setattr(c, n, RecordingLin(getattr(w, n), hset, f"{p}.attn.{n}"))
    c.wo_a_fn = wo_a_recorder(w.wo_a, cfg, hset, f"{p}.attn.wo_a")
    if w.idx is not None:
        ix = _LW()
        ix.__dict__.update(w.idx.__dict__)
        ix.wq_b = RecordingLin(w.idx.wq_b, hset, f"{p}.attn.indexer.wq_b")
        c.idx = ix
    if w.engram is not None:
        e = _LW()
        e.__dict__.update(w.engram.__dict__)
        e.wkv = RecordingLin(w.engram.wkv, hset, f"{p}.engram.wkv")
        c.engram = e
    return c, p


def make_graph(mine, mode, hsets, row_taps=None):
    cfg = mine.cfg
    g = CalibGraph(cfg, torch.float32)
    p = lambda l: f"layers.{l}" if l < cfg.num_hidden_layers else f"mtp.{l - cfg.num_hidden_layers}"
    for l in range(cfg.num_hidden_layers + cfg.n_mtp_layers):
        hs = hsets[l]
        g.lw[l], _ = recorded_layer(mine, l, hs)

    def moe_fn(l, x):
        w = mine.lw[l]
        hs = hsets[l]
        n_exp, topk = w.n_exp, w.topk
        pre = f"{p(l)}.ffn"

        def expert_fn(e, capture):
            h = mine.provider.lookup(l, e)
            if not capture:
                return h.w1, h.w3, h.w2, None, None
            return (h.w1, h.w3, h.w2,
                    lambda x_: hs.add(f"{pre}.experts.{e}.in", x_),
                    lambda h_: hs.add(f"{pre}.experts.{e}.mid", h_))

        def shared_fn(x_):
            sh = w.shared
            return expert_ffn(x_, sh.w1, sh.w3, sh.w2, cfg.swiglu_limit,
                              lambda t: hs.add(f"{pre}.shared.in", t), lambda t: hs.add(f"{pre}.shared.mid", t))
        y, asg = moe_calibrate(cfg, x, w.gate_w, w.gate_b, topk, n_exp, expert_fn, shared_fn, mode)
        hs.asg = asg
        return y
    g.moe_fn = moe_fn
    return g


def run_rows(mine, graph, ids):
    """Block-by-block over one row. Returns per-block inputs + final hidden + ho."""
    cfg = mine.cfg
    emb = RowEmbed(cfg, mine.embed_w)
    h = emb.embed(ids)
    hasher = None
    ho = new_handoff()
    ins = {}
    hashes = None
    st0 = mine.new_state(1, ids.size(1))
    hashes = st0.hasher(ids, 0) if st0.hasher is not None else None
    for l in range(cfg.num_hidden_layers):
        w = mine.lw[l]
        if w.engram is not None:
            e = graph.lw[l].engram
            h = engram_inject(h, hashes[:, :, w.engram.pos, :], mine.engram_store, w.engram.pos, e.wkv, e.qk,
                              cfg.hc_mult, cfg.hidden_size, cfg.rms_norm_eps)
        ins[l] = h.clone()
        h = graph.run_block(l, h, ho)
    return ins, graph.final_hidden(h, ho), ho


@pytest.mark.parametrize("mode", ["routed", "all"])
@pytest.mark.parametrize("quantized", [False, True])
def test_block_chain_equals_full_forward(mode, quantized):
    cfg, tok, tmap, tables, sd = setup(quantized, 3)
    mine = build_mine(cfg, sd, tables, tmap)
    L = 37
    ids = torch.randint(0, cfg.vocab_size, (1, L), generator=torch.Generator().manual_seed(7))
    st = mine.new_state(1, L)
    taps = {}
    mine.forward(ids, 0, st, taps)
    n = cfg.num_hidden_layers
    hsets = {l: HessianSet() for l in range(n + cfg.n_mtp_layers)}
    g = make_graph(mine, mode, hsets)
    ins, fin, ho = run_rows(mine, g, ids)
    worst = 0.0
    for l in range(n):
        d = (ins[l] - taps[f"L{l}.in"]).abs().max().item()
        scale = taps[f"L{l}.in"].abs().max().item()
        worst = max(worst, d / scale)
        assert d <= 2e-4 * scale, (l, d, scale)
    # the reference applies the routing weight before w2 and sums in a different order: equal to fp rounding
    ref_fin = ops.hc_pre(taps["final.in"], taps["final.pre"])
    assert (fin - ref_fin).abs().max().item() <= 2e-4 * ref_fin.abs().max().item()
    print("worst relative block-input diff:", worst)
    # hand-off bookkeeping: pools only for kv sources, pre_mix for every block
    assert sorted(ho["pool"]) == sorted(cfg.kv_source_layer_ids)
    assert sorted(ho["pre_mix"]) == list(range(n))
    assert set(ho["main"]) == set(cfg.dspark_target_layer_ids)


def test_block_hessians_match_independent_computation():
    cfg, tok, tmap, tables, sd = setup(True, 5)
    mine = build_mine(cfg, sd, tables, tmap)
    L = 41
    ids = torch.randint(0, cfg.vocab_size, (1, L), generator=torch.Generator().manual_seed(11))
    st = mine.new_state(1, L)
    taps = {}
    mine.forward(ids, 0, st, taps)
    n = cfg.num_hidden_layers
    hsets = {l: HessianSet() for l in range(n + cfg.n_mtp_layers)}
    g = make_graph(mine, "routed", hsets)
    ins, fin, ho = run_rows(mine, g, ids)
    ne = cfg.rms_norm_eps
    for l in range(n):
        w = mine.lw[l]
        hs = hsets[l]
        p = f"layers.{l}"
        # wq_a / wkv input = rmsnorm(hc_pre(block input, pre_mix of the previous block), attn_norm)
        h_in, pm = taps[f"L{l}.in"], taps[f"L{l}.pre"]
        x = ops.rmsnorm(ops.hc_pre(h_in, pm), w.attn_norm, ne).reshape(-1, cfg.hidden_size)
        Hx = x.T @ x
        for name in ("wq_a", "wkv"):
            H = hs.data[f"{p}.attn.{name}"]
            assert H["count"] == L
            assert (H["H"] - Hx).abs().max().item() <= 5e-3 * Hx.abs().max().item(), (l, name)
        # wo_b input: o_lora, count L; wo_a: one Hessian per group over the o heads
        assert hs.data[f"{p}.attn.wo_b"]["count"] == L
        for gi in range(cfg.o_groups):
            assert hs.data[f"{p}.attn.wo_a.{gi}"]["count"] == L
        if w.idx is not None:
            assert hs.data[f"{p}.attn.indexer.wq_b"]["count"] == L
        # routed experts: each token goes to exactly top-k experts
        asg = hs.asg
        total = sum(hs.data[f"{p}.ffn.experts.{e}.in"]["count"] for e in range(w.n_exp)
                    if f"{p}.ffn.experts.{e}.in" in hs.data)
        assert total == L * w.topk, (l, total)
        assert int(asg.counts.sum()) == L * w.topk
        assert hs.data[f"{p}.ffn.shared.in"]["count"] == L
        assert torch.equal(asg.idx, taps[f"L{l}.route.idx"])


def test_expert_hessians_routed_vs_all():
    """mode 'all': every expert sees every token (count L). mode 'routed': w1/w3 input H of an expert equals
    the sum over its routed tokens of x x^T; w2 input H equals the sum of swiglu activations."""
    cfg, tok, tmap, tables, sd = setup(False, 9)
    mine = build_mine(cfg, sd, tables, tmap)
    L = 53
    ids = torch.randint(0, cfg.vocab_size, (1, L), generator=torch.Generator().manual_seed(2))
    n = cfg.num_hidden_layers
    res = {}
    for mode in ("routed", "all"):
        hsets = {l: HessianSet() for l in range(n + cfg.n_mtp_layers)}
        g = make_graph(mine, mode, hsets)
        run_rows(mine, g, ids)
        res[mode] = hsets
    l = 3
    w = mine.lw[l]
    p = f"layers.{l}.ffn"
    hr, ha = res["routed"][l], res["all"][l]
    for e in range(w.n_exp):
        assert ha.data[f"{p}.experts.{e}.in"]["count"] == L
        c = int(hr.asg.counts[e])
        if c:
            assert hr.data[f"{p}.experts.{e}.in"]["count"] == c
        else:
            assert f"{p}.experts.{e}.in" not in hr.data
    # block outputs of both modes agree (routed mode only changes what the hooks see)
    # independent H_mid of the busiest expert from its routed rows
    e = int(hr.asg.counts.argmax())
    ti, tw = hr.asg.rows(e)
    # the x entering the MoE of layer l: rebuild it through the attention half
    hsets2 = {k: HessianSet(enabled=False) for k in range(n + cfg.n_mtp_layers)}
    g2 = make_graph(mine, "routed", hsets2)
    emb = RowEmbed(cfg, mine.embed_w)
    h = emb.embed(ids)
    ho = new_handoff()
    st0 = mine.new_state(1, L)
    hashes = st0.hasher(ids, 0)
    xin = None
    for k in range(l + 1):
        wk = mine.lw[k]
        if wk.engram is not None:
            h = engram_inject(h, hashes[:, :, wk.engram.pos, :], mine.engram_store, wk.engram.pos, wk.engram.wkv,
                              wk.engram.qk, cfg.hc_mult, cfg.hidden_size, cfg.rms_norm_eps)
        if k < l:
            h = g2.run_block(k, h, ho)
        else:
            xin, _ = g2.block_pre(k, h, ho)
    x = xin.reshape(-1, cfg.hidden_size)[ti]
    hd = mine.provider.lookup(l, e)
    mid = ops.swiglu(hd.w1(x), hd.w3(x), cfg.swiglu_limit)
    Hm = mid.T @ mid
    Hin = x.T @ x
    got_m, got_i = hr.data[f"{p}.experts.{e}.mid"]["H"], hr.data[f"{p}.experts.{e}.in"]["H"]
    assert (got_m - Hm).abs().max().item() <= 1e-4 * Hm.abs().max().item()
    assert (got_i - Hin).abs().max().item() <= 1e-4 * Hin.abs().max().item()


def test_dspark_stage_calibration_equals_forward_spec():
    cfg, tok, tmap, tables, sd = setup(False, 4)
    mine = build_mine(cfg, sd, tables, tmap)
    n = cfg.num_hidden_layers
    g_ = torch.Generator().manual_seed(21)
    L = 30
    ids = torch.randint(0, cfg.vocab_size, (1, L), generator=g_)
    p = 17                                     # anchor
    # reference path: prefill p tokens (seeds the rings), decode position p, then the draft step
    st = mine.new_state(1, 128)
    _, mm_pre = mine.forward(ids[:, :p], 0, st)
    mine.mtp.forward_spec(ids[:, p], mm_pre, 0, st)
    _, mm_dec = mine.forward(ids[:, p:p + 1], p, st)
    taps = {}
    mine.mtp.forward_spec(ids[:, p + 1], mm_dec, p, st, taps)
    # calibration path, fed the same main hidden states
    mh = torch.cat([mm_pre, mm_dec], dim=1)                   # [1, p + 1, 3d]
    d = cfg.hidden_size
    ho = new_handoff()
    for i, lid in enumerate(cfg.dspark_target_layer_ids):
        ho["main"][lid] = mh[..., i * d:(i + 1) * d]
    hsets = {l: HessianSet() for l in range(n + cfg.n_mtp_layers)}
    g = make_graph(mine, "routed", hsets)
    dc = DSparkCalib(g, mine.embed_w, mine.mtp.main_proj, mine.mtp.main_norm)
    mx = dc.main_x(ho)
    h, pm = dc.draft_input(ids, [p])
    ctaps = {}
    for j in range(cfg.n_mtp_layers):
        l = n + j
        mkv = dc.stage_mkv(l, mx)
        h, pm = dc.run_stage(l, h, pm, mkv, [p], ctaps)
        for k in ("attn", "ffn"):
            a, b = taps[f"S{l}.{k}"], ctaps[f"S{l}.{k}"]
            assert (a - b).abs().max().item() <= 2e-4 * max(a.abs().max().item(), 1e-3), (l, k)
    assert dc.anchors(30, 5) == sorted(set(dc.anchors(30, 5))) and all(1 <= a <= 28 for a in dc.anchors(30, 5))
    # wkv of an MTP stage sees both the draft block rows and the target rows (own Hessian)
    wk = hsets[n].data[f"mtp.0.attn.wkv"]
    assert wk["count"] == (p + 1) + cfg.dspark_block_size
