"""ArchRunner for DeepSeek-V4.1-Flash: thin wrapper over the reference graph in `exllamav3.dsv41` (nothing re-implemented).

ArchRunner interface used by capture_stream.StreamedCapture:
    hidden, n_experts, device, shared_groups
    embed_row(ids) -> h0 (state tensor of one row)            new_handoff() -> dict of per-row cross-layer state
    layer_begin(l, W, capture) -> ls (ls.hs: HessianSet-like with .add / .data / .enabled)      layer_end(ls)
    attn_half(ls, h, ho, ids) -> (x [1, L, d], ctx)           post(ls, ctx, y [1, L, d], ho) -> h_next
    dense_layout(ls) -> {group: [linears]}                    topk_of(ls)
    route(ls, x [n, d]) -> (wts, idx)                         expert_weights(ls, e) -> opaque
    expert_apply(ls, x, w, e, capture, weights) -> out        shared_apply(ls, x) -> out
    head_h_new() / head_h_add(hd, h, ho, norm_w)
Needs PYTHONPATH to contain a checkout with exllamav3/dsv41 (the DS4.1 engine); CPU or GPU.
"""
from __future__ import annotations

import types

import torch


class DS41Runner:
    shared_groups = ("ffn.shared_in", "ffn.shared_down")

    def __init__(self, cfg, device="cpu", embed_w=None, engram=None):
        from exllamav3.dsv41.stream_calib import StreamCalibrator
        # the calibrator object is used for its verified layer-weight bundle (_bundle) and graph, not for its row loop
        self.cal = StreamCalibrator(cfg, device, embed_w, engram, state_device="cpu")
        self.cfg, self.device, self.g = cfg, torch.device(device), self.cal.g
        self.hidden, self.n_experts = cfg.hidden_size, cfg.n_routed_experts

    def embed_row(self, ids):
        from exllamav3.dsv41.calib import RowEmbed
        return RowEmbed(self.cfg, self.cal.embed_w).embed(ids)

    def new_handoff(self):
        from exllamav3.dsv41.calib import new_handoff
        return new_handoff()

    def layer_begin(self, l, W, capture):
        from exllamav3.dsv41.calib import HessianSet, RecordingLin
        from exllamav3.dsv41.linear import DenseLin
        cfg, dev, g = self.cfg, self.device, self.g
        hs = HessianSet(dev, enabled=capture)
        w = self.cal._bundle(l, W, hs)
        g.lw = {l: w}
        ls = types.SimpleNamespace(l=l, W=W, hs=hs, w=w, capture=capture, eg=None, w_eg=None)
        if l in cfg.engram_layer_ids:
            assert self.cal.engram is not None, "engram layer without EngramSupport"
            ls.eg = (W.dense("engram.wkv").to(dev), W.dense("engram.q_weight"), W.dense("engram.k_weight"))
            ls.w_eg = RecordingLin(DenseLin(ls.eg[0]), hs, "engram.wkv")
        return ls

    def layer_end(self, ls):
        self.g.lw = {}

    def attn_half(self, ls, h, ho, ids):
        dev, cfg = self.device, self.cfg
        h = h.to(dev)
        if ls.eg is not None:
            h = self.cal._engram_with_hook(h, ids, ls.l, ls.eg, ls.w_eg)
        x, ctx = self.g.block_pre(ls.l, h, ho)
        ho["main"] = {k: v.to("cpu") for k, v in ho["main"].items()}
        return x.reshape(-1, cfg.hidden_size), ctx

    def post(self, ls, ctx, y, ho):
        return self.g.block_post(ls.l, ctx, y, ho)

    def dense_layout(self, ls):
        from exllamav3.dsv41.stream_calib import hessian_layout
        return hessian_layout(self.cfg, ls.l)

    def n_exp_of(self, ls):
        return ls.w.n_exp

    def topk_of(self, ls):
        return ls.w.topk

    def route(self, ls, x):
        from exllamav3.dsv41.calib import moe_route
        return moe_route(self.cfg, x.to(self.device), ls.w.gate_w, ls.w.gate_b, ls.w.topk)

    def expert_weights(self, ls, e):
        return tuple(t.to(self.device) for t in ls.W.expert(e))

    def expert_apply(self, ls, x, w, e, capture, weights):
        from exllamav3.dsv41.calib import expert_ffn
        w1, w3, w2 = w
        hi = hm = None
        if capture:
            hi = lambda t: ls.hs.add(f"ffn.experts.{e}.in", t)
            hm = lambda t: ls.hs.add(f"ffn.experts.{e}.mid", t)
        return expert_ffn(x, w1, w3, w2, self.cfg.swiglu_limit, hi, hm, weights)

    def shared_apply(self, ls, x):
        from exllamav3.dsv41.calib import expert_ffn
        s1, s3, s2 = (ls.W.dense(f"ffn.shared_experts.{k}").to(self.device) for k in ("w1", "w3", "w2"))
        if not ls.capture:
            return expert_ffn(x, s1, s3, s2, self.cfg.swiglu_limit)
        return expert_ffn(x, s1, s3, s2, self.cfg.swiglu_limit, lambda x_: ls.hs.add("ffn.shared_in", x_),
                          lambda x_: ls.hs.add("ffn.shared_down", x_))

    def head_h_new(self):
        from exllamav3.dsv41.calib import h_data_new
        return h_data_new(self.cfg.hidden_size, self.device, "head")

    def head_h_add(self, hd, h, ho, norm_w):
        from exllamav3.dsv41 import ops
        from exllamav3.dsv41.calib import h_data_add
        cfg = self.cfg
        x = ops.hc_pre(h.to(self.device), ho["pre_mix"][cfg.num_hidden_layers - 1].to(self.device))
        h_data_add(hd, ops.rmsnorm(x, norm_w.float().to(self.device), cfg.rms_norm_eps).reshape(-1, cfg.hidden_size))
