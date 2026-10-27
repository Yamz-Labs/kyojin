"""
Streaming calibration of DeepSeek-V4.1-Flash for a converter that fetches tensors layer by layer (no exllamav3
Config / Module tree): the Hessian of every linear of a block, expert by expert.

    cal = StreamCalibrator(cfg, device = "cuda:0", embed_w = embed_bf16, engram = EngramSupport(...))
    cal.start(rows)                                   # rows: list of [1, L] token tensors
    for l in range(cfg.num_hidden_layers):
        W = SourceLayerWeights(src, l)                # or any LayerWeights (dequantized bf16/fp32 dense tensors)
        cal.run_layer(l, W, sink, commit = False)     # capture pass: reference weights, Hessians -> sink
        ... quantize the linears using the Hessians the sink got ...
        cal.run_layer(l, Wq, None, commit = True)     # advance the rows through the layer (reference or the
                                                      # dequantized quantized weights: error propagation)
    for H in cal.head_hessian_chunks(norm_w): ...     # input of `head` (hc_pre + final norm), per row
    # MTP (dspark) stages, after the trunk has committed every layer:
    cal.mtp_begin(main_proj_w, main_norm_w, sink, commit = ...)
    for j in range(n_mtp): cal.run_mtp_stage(j, Wj, sink, commit = ...)

Sink protocol: `sink.emit(group, linears, H)` with
    group    Hessian name, e.g. "attn.in" or "ffn.experts.17.in"
    linears  names (relative to the layer prefix, official checkpoint names without `.weight`) that share it:
             ["attn.wq_a", "attn.wkv"], ["ffn.experts.17.w1", "ffn.experts.17.w3"], ["ffn.experts.17.w2"], ...
             (wo_a slices are "attn.wo_a.slice.G", the keys the EXL3 output uses)
    H        quantize_exl3 `H_data` dict (H = sum of x^T x over `H["count"]` rows, not yet normalized), or None
             for an expert the router never selected on these rows (quantize it uncalibrated)
Experts are processed one at a time: the Hessians of expert e are emitted as soon as it has been applied and freed
afterwards, so peak memory is one expert's H (hidden^2 + intermediate^2 floats), not 384 of them.

Expert Hessians use the "routed" assignment (an expert sees the tokens the router sent to it). The MoE output (and
therefore the next layer's inputs) is the exact reference forward.
"""
from __future__ import annotations
import torch
from . import ops, quant
from .config import DSV41Config
from .linear import DenseLin
from .model import _LW
from .engram import NgramHasher, engram_inject
from .calib import (CalibGraph, HessianSet, RecordingLin, wo_a_recorder, new_handoff, RowEmbed, moe_calibrate,
                    expert_ffn, DSparkCalib, h_data_add)


# ---------------------------------------------------------------------------------------------- weights
class LayerWeights:
    """Dense (dequantized) tensors of one block, names relative to the layer prefix (`layers.N.` / `mtp.J.`):
    dense("attn.wq_a") [out, in], dense("attn_norm"), dense("hc_attn_fn"), dense("attn.attn_sink"),
    dense("attn.wo_a") [G * r, in] (whole fused tensor), dense("ffn.gate"), dense("ffn.gate.bias"), optional
    compressor / indexer / engram tensors; expert(e) -> (w1, w3, w2) dense [out, in]."""

    def dense(self, name: str) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def has(self, name: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    def expert(self, e: int):  # pragma: no cover
        raise NotImplementedError


class SourceLayerWeights(LayerWeights):
    """LayerWeights over a WeightSource (official safetensors / dict / GGUF): fp8 / fp4 are decoded on read."""

    def __init__(self, src, layer: int, device="cpu"):
        self.src, self.p, self.device = src, src.layer_prefix(layer), device

    def dense(self, name):
        return self.src.tensor(f"{self.p}.{name}", torch.float32, self.device)

    def has(self, name):
        return self.src.has(f"{self.p}.{name}")

    def expert(self, e):
        return tuple(self.src.tensor(f"{self.p}.ffn.experts.{e}.{w}", torch.float32, self.device)
                     for w in ("w1", "w3", "w2"))


class EngramSupport:
    """Hash layout + token map + row store for the engram injection (see engram.py)."""

    def __init__(self, cfg: DSV41Config, token_map, store, multipliers=None, primes=None):
        from .engram import EngramLayout, compute_multipliers
        self.cfg, self.store = cfg, store
        self.token_map = token_map
        self.nv = cfg.engram_compressed_vocab_size or int(max(token_map)) + 1
        self.layout = EngramLayout.build(cfg.engram_layer_ids, cfg.engram_num_embeddings, cfg.engram_max_ngram_size,
                                         cfg.engram_vocab_size, cfg.engram_n_heads, cfg.engram_head_dim, primes)
        self.mult = multipliers if multipliers is not None else \
            compute_multipliers(self.layout.layer_ids, self.layout.max_ngram_size, self.nv)

    def inject(self, h, ids, l, wkv_dense, q_w, k_w):
        cfg = self.cfg
        pos = cfg.engram_layer_ids.index(l)
        hasher = NgramHasher(self.layout, self.token_map, cfg.engram_pad_token_id, self.nv, self.mult, 1,
                             ids.shape[1], h.device)
        hashes = hasher(ids.to(h.device), 0)
        qk = q_w.float().to(h.device) * k_w.float().to(h.device)
        return engram_inject(h, hashes[:, :, pos, :], self.store, pos, DenseLin(wkv_dense.to(h.device)), qk,
                             cfg.hc_mult, cfg.hidden_size, cfg.rms_norm_eps)


class CollectSink:
    """Sink that keeps everything (tests, small models)."""

    def __init__(self):
        self.items = []

    def emit(self, group, linears, H):
        self.items.append((group, list(linears), H))

    def by_group(self):
        return {g: (ls, H) for g, ls, H in self.items}


# ---------------------------------------------------------------------------------------------- calibrator
def hessian_layout(cfg: DSV41Config, l: int, mtp: bool = False) -> dict:
    """{hessian group: [linear names]} of one block (experts excluded, they are per expert)."""
    d = {}
    if mtp:
        d["attn.in"] = ["attn.wq_a"]
        d["attn.wkv"] = ["attn.wkv"]
    else:
        d["attn.in"] = ["attn.wq_a", "attn.wkv"]
    d["attn.q_b"] = ["attn.wq_b"] + (["attn.indexer.wq_b"] if (not mtp and cfg.is_index_source(l)) else [])
    for g in range(cfg.o_groups):
        d[f"attn.o.{g}"] = [f"attn.wo_a.slice.{g}"]
    d["attn.o_b"] = ["attn.wo_b"]
    d["ffn.shared_in"] = ["ffn.shared_experts.w1", "ffn.shared_experts.w3"]
    d["ffn.shared_down"] = ["ffn.shared_experts.w2"]
    if not mtp and l in cfg.engram_layer_ids:
        d["engram.wkv"] = ["engram.wkv"]
    return d


class StreamCalibrator:
    def __init__(self, cfg: DSV41Config, device="cpu", embed_w: torch.Tensor = None, engram: EngramSupport = None,
                 mode: str = "routed", state_device="cpu", min_expert_tokens: int = 0):
        """min_expert_tokens > 0: an expert that the router sent fewer tokens than that on the calibration rows gets
        the input Hessian of its own tokens blended with the all-token Hessian (the shared expert's input, same
        activation) scaled to reach `min_expert_tokens` rows; an expert with no tokens gets the scaled all-token one."""
        self.cfg, self.device, self.mode = cfg.validate(), torch.device(device), mode
        self.min_expert_tokens = min_expert_tokens
        self._deferred = []
        self.sdev = torch.device(state_device)
        self.embed_w = embed_w.float().to(self.device) if embed_w is not None else None
        self.engram = engram
        self.g = CalibGraph(cfg, torch.float32, self.device)
        self.rows, self.h, self.ho = [], [], []

    # ---- rows
    def start(self, rows: list):
        self.rows = [r.clone() for r in rows]
        emb = RowEmbed(self.cfg, self.embed_w)
        self.h = [emb.embed(r).to(self.sdev) for r in self.rows]
        self.ho = [new_handoff() for _ in self.rows]

    # ---- a trunk layer
    def _bundle(self, l, W: LayerWeights, hs: HessianSet, mtp: bool = False) -> _LW:
        cfg, dev = self.cfg, self.device
        D = lambda n: W.dense(n).to(dev)
        rec = lambda n, fn: RecordingLin(fn, hs, n)
        L_ = lambda n: DenseLin(D(n))
        w = _LW()
        w.l = l
        for t in ("attn", "ffn"):
            for p in ("fn", "base", "scale"):
                setattr(w, f"hc_{t}_{p}", D(f"hc_{t}_{p}").float())
        w.attn_norm, w.ffn_norm = D("attn_norm").float(), D("ffn_norm").float()
        w.sink = D("attn.attn_sink").float()
        w.q_norm, w.kv_norm = D("attn.q_norm").float(), D("attn.kv_norm").float()
        w.wq_a = rec("attn.in", L_("attn.wq_a"))
        w.wkv = rec("attn.wkv" if mtp else None, L_("attn.wkv"))
        w.wq_b = rec("attn.q_b", L_("attn.wq_b"))
        w.wo_b = rec("attn.o_b", L_("attn.wo_b"))
        w.wo_a = D("attn.wo_a")
        w.wo_a_fn = wo_a_recorder(w.wo_a, cfg, hs, "attn.o")
        w.comp = None
        if not mtp and cfg.is_kv_source(l):
            c = _LW()
            c.wkv = DenseLin(D("attn.compressor.wkv").float())
            c.wgate = DenseLin(D("attn.compressor.wgate").float()) if cfg.ratio(l) > 1 else None
            c.norm = D("attn.compressor.norm").float()
            w.comp = c
        w.idx = None
        if not mtp and cfg.is_index_source(l):
            ix = _LW()
            ix.wq_b = rec(None, L_("attn.indexer.wq_b"))
            ix.weights_proj = DenseLin(D("attn.indexer.weights_proj").float())
            ix.owns_k = cfg.is_kv_source(l)
            if ix.owns_k:
                ix.wk = DenseLin(D("attn.indexer.wk").float())
                ix.k_norm = D("attn.indexer.k_norm").float()
            w.idx = ix
        w.gate_w, w.gate_b = D("ffn.gate").float(), D("ffn.gate.bias").float()
        w.n_exp, w.topk = cfg.moe_counts(l)
        w.engram = None
        return w

    def _flush_deferred(self, sink, hs: HessianSet):
        """Blend the input Hessians of sparsely routed experts with the all-token Hessian (see __init__)."""
        if not self._deferred:
            return
        Hs = hs.data["ffn.shared_in"]
        for e, lins, H, c in self._deferred:
            blend = dict(Hs)
            blend["H"] = Hs["H"] * ((self.min_expert_tokens - c) / Hs["count"])
            blend["inf_nan"] = Hs["inf_nan"].clone()
            blend["first_key"] = f"ffn.experts.{e}.in"
            if H is not None:
                blend["H"] = blend["H"] + H["H"]
            blend["count"] = self.min_expert_tokens
            sink.emit(f"ffn.experts.{e}.in", lins, blend)
        self._deferred = []

    @staticmethod
    def _emit_groups(sink, hs: HessianSet, layout: dict, groups):
        for grp in groups:
            H = hs.data.pop(grp, None)
            if sink is not None:
                sink.emit(grp, layout[grp], H)

    def _moe(self, l, x, W, sink, hs, w):
        """Pooled MoE over [N, d] rows: expert by expert, Hessians emitted per expert."""
        cfg = self.cfg
        n_exp, topk = w.n_exp, w.topk
        cache = {}

        def expert_fn(e, capture):
            if e not in cache:
                cache.clear()
                cache[e] = tuple(t.to(self.device) for t in W.expert(e))
            w1, w3, w2 = cache[e]
            if not capture or sink is None:
                return w1, w3, w2, None, None
            return (w1, w3, w2, lambda t: hs.add(f"ffn.experts.{e}.in", t), lambda t: hs.add(f"ffn.experts.{e}.mid", t))

        def shared_fn(t):
            s1, s3, s2 = (W.dense(f"ffn.shared_experts.{k}").to(self.device) for k in ("w1", "w3", "w2"))
            if sink is None:
                return expert_ffn(t, s1, s3, s2, cfg.swiglu_limit)
            return expert_ffn(t, s1, s3, s2, cfg.swiglu_limit, lambda x_: hs.add("ffn.shared_in", x_),
                              lambda x_: hs.add("ffn.shared_down", x_))

        def on_expert(e, count):
            if sink is None:
                return
            ln = [f"ffn.experts.{e}.w1", f"ffn.experts.{e}.w3"], [f"ffn.experts.{e}.w2"]
            Hin = hs.data.pop(f"ffn.experts.{e}.in", None)
            if self.min_expert_tokens and count < self.min_expert_tokens:
                self._deferred.append((e, ln[0], Hin, count))          # blended once the all-token Hessian exists
            else:
                sink.emit(f"ffn.experts.{e}.in", ln[0], Hin)
            sink.emit(f"ffn.experts.{e}.mid", ln[1], hs.data.pop(f"ffn.experts.{e}.mid", None))

        y, asg = moe_calibrate(cfg, x, w.gate_w, w.gate_b, topk, n_exp, expert_fn, shared_fn, "routed", on_expert)
        self.last_assignment = asg
        return y

    def run_layer(self, l: int, W: LayerWeights, sink=None, commit: bool = True):
        """One trunk block over every row. Hessians go to `sink` (None: no capture); commit=True stores the
        outputs as the new row states."""
        cfg, dev, g = self.cfg, self.device, self.g
        hs = HessianSet(dev, enabled = sink is not None)
        w = self._bundle(l, W, hs)
        g.lw = {l: w}
        eg = None
        if l in cfg.engram_layer_ids:
            assert self.engram is not None, "engram layer without EngramSupport"
            eg = (W.dense("engram.wkv").to(dev), W.dense("engram.q_weight"), W.dense("engram.k_weight"))
            w_eg = RecordingLin(DenseLin(eg[0]), hs, "engram.wkv")
        xs, ctxs = [], []
        for i, h in enumerate(self.h):
            h = h.to(dev)
            if eg is not None:
                hh = self._engram_with_hook(h, self.rows[i], l, eg, w_eg)
                h = hh
            x, ctx = g.block_pre(l, h, self.ho[i])
            xs.append(x.reshape(-1, cfg.hidden_size))
            ctxs.append(ctx)
            self.ho[i]["main"] = {k: v.to(self.sdev) for k, v in self.ho[i]["main"].items()}
        lay = hessian_layout(cfg, l)
        shared = ("ffn.shared_in", "ffn.shared_down")
        if sink is not None:
            self._emit_groups(sink, hs, lay, [k for k in lay if k not in shared])
        lens = [x.shape[0] for x in xs]
        X = torch.cat(xs, dim=0)
        del xs
        self._deferred = []
        y = self._moe(l, X, W, sink, hs, w)
        if sink is not None:
            self._flush_deferred(sink, hs)
            self._emit_groups(sink, hs, lay, shared)
        outs = []
        for i, (yi, ctx) in enumerate(zip(y.split(lens), ctxs)):
            L = self.rows[i].shape[1]
            outs.append(g.block_post(l, ctx, yi.view(1, L, cfg.hidden_size), self.ho[i]))
        if commit:
            self.h = [o.to(self.sdev) for o in outs]
        g.lw = {}
        return None

    def _engram_with_hook(self, h, ids, l, eg, w_eg):
        cfg = self.cfg
        e = self.engram
        pos = cfg.engram_layer_ids.index(l)
        hasher = NgramHasher(e.layout, e.token_map, cfg.engram_pad_token_id, e.nv, e.mult, 1, ids.shape[1], h.device)
        hashes = hasher(ids.to(h.device), 0)
        qk = eg[1].float().to(h.device) * eg[2].float().to(h.device)
        return engram_inject(h, hashes[:, :, pos, :], e.store, pos, w_eg, qk, cfg.hc_mult, cfg.hidden_size,
                             cfg.rms_norm_eps)

    # ---- head
    def head_inputs(self, norm_w: torch.Tensor):
        """Yield the input rows of `head` per calibration row: [L, hidden] fp32 (hc_pre with the last block's pending
        pre-mix, then the final RMS norm)."""
        cfg = self.cfg
        nw = norm_w.float().to(self.device)
        for h, ho in zip(self.h, self.ho):
            x = ops.hc_pre(h.to(self.device), ho["pre_mix"][cfg.num_hidden_layers - 1].to(self.device))
            yield ops.rmsnorm(x, nw, cfg.rms_norm_eps).reshape(-1, cfg.hidden_size)

    def head_hessian(self, norm_w: torch.Tensor) -> dict:
        """Hessian of the output head over all rows (the framework's `block` qmap of the head Linear)."""
        from .calib import h_data_new
        hd = h_data_new(self.cfg.hidden_size, self.device, "head")
        for x in self.head_inputs(norm_w):
            h_data_add(hd, x)
        return hd

    # ---- MTP
    def mtp_begin(self, main_proj_w: torch.Tensor, main_norm_w: torch.Tensor, sink=None, anchors: int = 8,
                  commit_proj_w: torch.Tensor = None):
        """Prepare the dspark draft inputs from the target taps (ho["main"]) of every row; emits the Hessian of
        main_proj ("mtp.main_proj", all target positions). commit_proj_w: dense weight to use for the draft state
        (quantized main_proj); default the reference."""
        cfg, dev = self.cfg, self.device
        from .calib import h_data_new
        n = cfg.num_hidden_layers
        self.dc = DSparkCalib(self.g, self.embed_w, None, None)
        wp = (commit_proj_w if commit_proj_w is not None else main_proj_w).float().to(dev)
        hd = h_data_new(main_proj_w.shape[1], dev, "mtp.main_proj") if sink is not None else None
        self.mx, self.anc, self.dh = [], [], []
        for i, ho in enumerate(self.ho):
            mh = torch.cat([ho["main"][l].float() for l in cfg.dspark_target_layer_ids], dim=-1).to(dev)
            if hd is not None:
                h_data_add(hd, mh)
            self.mx.append(ops.rmsnorm(torch.nn.functional.linear(mh, wp), main_norm_w.float().to(dev),
                                       cfg.rms_norm_eps).cpu())
            self.anc.append(self.dc.anchors(self.rows[i].shape[1], anchors))
            h, pm = self.dc.draft_input(self.rows[i].to(dev), self.anc[i])
            self.dh.append((h.cpu(), pm.cpu()))
        if sink is not None:
            sink.emit("mtp.main_proj", ["mtp.main_proj"], hd)

    def run_mtp_stage(self, j: int, W: LayerWeights, sink=None, commit: bool = True):
        cfg, dev, g = self.cfg, self.device, self.g
        l = cfg.num_hidden_layers + j
        hs = HessianSet(dev, enabled = sink is not None)
        w = self._bundle(l, W, hs, mtp=True)
        g.lw = {l: w}
        xs, ctxs, lens = [], [], []
        for i in range(len(self.rows)):
            if not self.anc[i]:
                continue
            h, pm = self.dh[i][0].to(dev), self.dh[i][1].to(dev)
            mkv = self.dc.stage_mkv(l, self.mx[i].to(dev))
            x, ctx = self.dc.stage_pre(l, h, pm, mkv, self.anc[i])
            xs.append(x.reshape(-1, cfg.hidden_size))
            ctxs.append((i, ctx, x.shape))
        lay = hessian_layout(cfg, l, mtp=True)
        shared = ("ffn.shared_in", "ffn.shared_down")
        if sink is not None:
            self._emit_groups(sink, hs, lay, [k for k in lay if k not in shared])
        X = torch.cat(xs, dim=0)
        self._deferred = []
        y = self._moe(l, X, W, sink, hs, w)
        if sink is not None:
            self._flush_deferred(sink, hs)
            self._emit_groups(sink, hs, lay, shared)
        outs = {}
        for (i, ctx, shp), yi in zip(ctxs, y.split([x.shape[0] for x in xs])):
            outs[i] = self.dc.stage_post(l, ctx, yi.view(shp))
        if commit:
            for i, (h, pm) in outs.items():
                self.dh[i] = (h.cpu(), pm.cpu())
        g.lw = {}


# ---------------------------------------------------------------------------------------------- helpers for sinks
def linear_weight(cfg: DSV41Config, W: LayerWeights, name: str) -> torch.Tensor:
    """Dense [out, in] weight of a linear named like the sink does (official name without ".weight";
    "attn.wo_a.slice.G" is rows [G*r, (G+1)*r) of attn.wo_a; "ffn.experts.E.w1|w3|w2")."""
    import re
    m = re.match(r"attn\.wo_a\.slice\.(\d+)$", name)
    if m:
        g, r = int(m.group(1)), cfg.o_lora_rank
        return W.dense("attn.wo_a")[g * r:(g + 1) * r]
    m = re.match(r"ffn\.experts\.(\d+)\.(w[123])$", name)
    if m:
        return W.expert(int(m.group(1)))[("w1", "w3", "w2").index(m.group(2))]
    return W.dense(name)


def quantize_with_hessian(weight_out_in: torch.Tensor, H_data, K: int, seed: int, device, codebook: str = "mul1",
                          out_scales: bool = True):
    """The real EXL3 quantizer on one dense [out, in] weight with a streamed Hessian (None -> uncalibrated).
    Returns ({suffix: cpu tensor} EXL3 tensors of the linear, proxy_err). `device` is the int / device of the
    quantizer (cuda:0 -> 0)."""
    from ..modules.quant.exl3_lib.quantize import quantize_exl3
    dev = torch.device(device)
    if H_data is None:
        n = weight_out_in.shape[1]
        H_data = {"H": torch.empty((n, n), device="meta"), "first_key": f"w{seed}", "count": 0, "finalized": False,
                  "num_total": 0, "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev}
    qargs = {"seed": seed, "K": K, "devices": [dev.index if dev.index is not None else 0], "device_ratios": None,
             "apply_out_scales": out_scales}
    if codebook in ("mul1", "mcg"):
        qargs[codebook] = True
    w = weight_out_in.float().T.contiguous()
    _, proxy, out = quantize_exl3(w, H_data, qargs, False)
    return {k: v.cpu().contiguous() for k, v in out.items()}, proxy
