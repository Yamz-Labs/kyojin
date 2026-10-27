"""
DeepSeek-V4.1-Flash modules for the EXL3 conversion (framework registration).

The quantizer walks `model.modules`; each module loads its tensors, runs a calibration forward that fills
`params["capture"]` (Hessians per qmap, exactly like every other architecture), is quantized, reloaded and
advanced. The forward is the verified torch graph of `exllamav3.dsv41` (bitwise equal to the official reference on
the tiny model), driven block by block:

    DSV41Block      one trunk layer (attention, shared expert, 384 routed experts, hyper-connections, engram)
                    `mtp=True`: an MTP (dspark) stage, same tensors, calibrated separately (dsv41.calib.DSparkCalib)
    DSV41Raw        small tensors kept bit-exact (norms, hc params, router, compressor, bf16 projections)
    DSV41Final      hc_pre of the last block + final norm
    DSV41Head       output head (Linear, head_bits)

Cross-layer state of a calibration row (pending hc pre-mix, compressed pools, indexer keys, top-k, candidate blocks)
travels in params["quant_preserve"]["dsv41"] (see dsv41.calib). Blocks only run inside the conversion flow; the
serving graph of the port is `exllamav3.dsv41.model`.
"""
from __future__ import annotations
from typing_extensions import override
import os
import torch
import torch.nn.functional as F

from . import Module, Linear
from .quant.fp16 import LinearFP16
from ..model.config import Config
from ..dsv41 import ops
from ..dsv41.config import DSV41Config
from ..dsv41.linear import DenseLin
from ..dsv41.model import _LW
from ..dsv41.calib import (CalibGraph, new_handoff, moe_calibrate, expert_ffn, h_data_add, wo_a_recorder)
from ..dsv41.engram import NgramHasher, engram_inject


# ------------------------------------------------------------------------------------------------ helpers
def _pad_last(x: torch.Tensor, n: int) -> torch.Tensor:
    if x.shape[-1] < n:
        x = F.pad(x, (0, n - x.shape[-1]))
    return x


def torch_capture_H(lin: Linear, x: torch.Tensor, params: dict):
    """Linear.capture_H without the CUDA kernels: same Hessian record (shared per qmap, the first linear to
    ask owns the accumulation), finite rows only."""
    cap = params["capture"]
    if lin.qmap not in cap:
        cap[lin.qmap] = lin.init_H_data(True)
    hd = cap[lin.qmap]
    if hd["first_key"] == lin.key:
        h_data_add(hd, x)
    else:
        hd["num_total"] += x.numel()


class ModLin:
    """Callable view of a framework Linear for the torch graph: pads the input to the (128-aligned) weight width,
    captures the Hessian when `params` asks for it, applies the unquantized fp16 weight in fp32 on CPU / fp16 on
    GPU (LinearFP16) or the EXL3 kernel (LinearEXL3), trims padded outputs, returns fp32."""

    def __init__(self, lin: Linear, params: dict):
        self.lin, self.params = lin, params
        self.in_features, self.out_features = lin.in_features_unpadded, lin.out_features_unpadded

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        lin, params = self.lin, self.params
        xp = _pad_last(x, lin.in_features)
        if lin.qmap and "capture" in params:
            torch_capture_H(lin, xp, params)
        inner = lin.inner
        if isinstance(inner, LinearFP16):
            w = inner.weight
            if w.device.type == "cpu":
                y = F.linear(xp.float(), w.t().float().contiguous())     # same call as the dense reference
            else:
                y = torch.matmul(xp.half(), w).float()
            if inner.bias is not None:
                y = y + inner.bias.float()
        else:
            lead = xp.shape[:-1]
            y = inner.forward(xp.half().reshape(-1, xp.shape[-1]), {}, torch.half).float()
            y = y.reshape(*lead, -1)
        if lin.out_features != lin.out_features_unpadded:
            y = y[..., :lin.out_features_unpadded]
        return y.contiguous()


class DSV41Raw(Module):
    """Bit-exact pass-through tensors (norms, hyper-connection parameters, router, compressor/indexer projections
    that stay in bf16, engram q/k weights). dtype is kept as stored (bf16 / fp32)."""

    def __init__(self, config: Config | None, key: str, names: list, optional: list = ()):
        super().__init__(config, key, None)
        self.names = list(names)
        self.optional = list(optional)
        self.t = {}

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)          # child modules (MTP extras host a Linear)
        stc = self.config.stc
        for n in self.names:
            self.t[n] = stc.get_tensor(n, device, allow_bf16 = True, no_defer = True).contiguous()
        for n in self.optional:
            if stc.has_tensor(n):
                self.t[n] = stc.get_tensor(n, device, allow_bf16 = True, no_defer = True).contiguous()

    @override
    def unload(self):
        super().unload()
        self.t = {}

    @override
    def get_tensors(self):
        return {k: v for k, v in self.t.items()} if self.device is not None else {}

    @override
    def weights_numel(self):
        return 0

    def f32(self, name):
        return self.t[name].float()

    @override
    def forward(self, x, params, out_dtype = None):
        raise RuntimeError("DSV41Raw is a tensor holder")


# ------------------------------------------------------------------------------------------------ block
class _TierView:
    """What expert_store.ExpertStore.register needs from a layer of routed experts (the BlockSparseMLP surface)."""

    def __init__(self, blk):
        self.key = blk.key + ".ffn"
        self.config = blk.config
        self.gated = True
        self.num_experts = self.num_local_experts = blk.n_exp
        self.num_experts_per_tok = blk.topk         # read by the store's warm-seeded budget split (EXL3_EXPERT_WARM)
        self.routing_first = None
        self.tp_mode = None
        self.gates, self.ups, self.downs = blk.ex_w1, blk.ex_w3, blk.ex_w2
        self.routing_cfg = None


class DSV41Block(Module):

    def __init__(
        self,
        config,
        key: str,
        layer_idx: int,
        dcfg: DSV41Config,
        mtp: bool = False,
        moe_mode: str = "all",
        qbits_key: str = "bits",
        select_hq_bits: int = 2,
    ):
        super().__init__(config, key, None)
        self.layer_idx = None if mtp else layer_idx        # mtp stages are not in the trunk layer map
        self.l = layer_idx                                 # graph layer id (mtp: n_layers + j)
        self.dcfg = dcfg
        self.mtp = mtp
        self.moe_mode = moe_mode
        self.module_name = "DSV41Block"
        self.tier = None            # expert_store.TierSet when EXL3_EXPERT_BUDGET_GIB tiers this block's routed experts
        n = dcfg.num_hidden_layers
        d, hd, H = dcfg.hidden_size, dcfg.head_dim, dcfg.num_attention_heads
        G = dcfg.o_groups
        hq = select_hq_bits
        a = f"{key}.attn"
        self.is_kv = (not mtp) and dcfg.is_kv_source(layer_idx)
        self.is_idx = (not mtp) and dcfg.is_index_source(layer_idx)
        self.engram_pos = None
        if not mtp and layer_idx in dcfg.engram_layer_ids:
            self.engram_pos = dcfg.engram_layer_ids.index(layer_idx)
        self.n_exp, self.topk = dcfg.moe_counts(layer_idx)
        # experts the converter pruned (K = 0 in quantization_config.expert_bits): no tensors in the pack, no Linear here;
        # the pack's router bias masks them so they are never selected
        from ..dsv41.packspec import pruned_set
        self.pruned = pruned_set(config.config_dict.get("quantization_config"), f"{key}.ffn", self.n_exp)

        def lin(name, i, o, qmap, bits_key = qbits_key, hqb = hq, qgroup = None, **kw):
            m = Linear(config, f"{key}.{name}", i, o, qmap = qmap, qbits_key = bits_key, select_hq_bits = hqb,
                       out_dtype = torch.half, trim_padded_out = True, qgroup = qgroup, **kw)
            self.register_submodule(m)
            return m

        # attention
        self.wq_a = lin("attn.wq_a", d, dcfg.q_lora_rank, f"{a}.in", )
        self.wkv = lin("attn.wkv", d, hd, f"{a}.in") if not mtp else lin("attn.wkv", d, hd, f"{a}.wkv")
        self.wq_b = lin("attn.wq_b", dcfg.q_lora_rank, H * hd, f"{a}.q_b")
        gw = H * hd // G
        self.wo_a = [
            lin(f"attn.wo_a.slice.{g}", gw, dcfg.o_lora_rank, f"{a}.o.{g}", fkey = f"{a}.wo_a",
                frange = [g * dcfg.o_lora_rank, (g + 1) * dcfg.o_lora_rank], frange_dim = 0)
            for g in range(G)
        ]
        self.wo_b = lin("attn.wo_b", G * dcfg.o_lora_rank, d, f"{a}.o_b")
        self.idx_wq_b = None
        if self.is_idx:
            self.idx_wq_b = lin("attn.indexer.wq_b", dcfg.q_lora_rank, dcfg.index_n_heads * dcfg.index_head_dim,
                                f"{a}.q_b")
        # shared expert
        isz = dcfg.moe_intermediate_size * dcfg.n_shared_experts
        sg = f"{key}.shared"
        self.sh_w1 = lin("ffn.shared_experts.w1", d, isz, f"{key}.ffn.shared_in", qgroup = sg)
        self.sh_w3 = lin("ffn.shared_experts.w3", d, isz, f"{key}.ffn.shared_in", qgroup = sg)
        self.sh_w2 = lin("ffn.shared_experts.w2", isz, d, f"{key}.ffn.shared_down", qgroup = sg)
        # routed experts: gate (w1) / up (w3) share a qmap (one Hessian) in 'all' mode, one per expert in 'routed'
        eg = f"{key}.experts"
        self.ex_w1, self.ex_w3, self.ex_w2 = [], [], []
        ei = dcfg.moe_intermediate_size
        # K = 0 experts (pack config expert_bits == 0) have no tensors: never-loaded stand-ins, skipped by the grouped kernels
        from .moe_mixedk import PrunedLinear, pruned_experts
        qcfg = getattr(config, "config_dict", {}).get("quantization_config") if config is not None else None
        pruned = pruned_experts(qcfg, f"{key}.ffn", self.n_exp)
        for e in range(self.n_exp):
            if e in pruned:
                pk = f"{key}.ffn.experts.{e}"
                self.ex_w1.append(PrunedLinear(pk + ".w1", d, ei))
                self.ex_w3.append(PrunedLinear(pk + ".w3", d, ei))
                self.ex_w2.append(PrunedLinear(pk + ".w2", ei, d))
                continue
            qin = f"{key}.ffn.experts_in" if moe_mode == "all" else f"{key}.ffn.experts.{e}.in"
            self.ex_w1.append(lin(f"ffn.experts.{e}.w1", d, ei, qin, hqb = 0, qgroup = eg))
            self.ex_w3.append(lin(f"ffn.experts.{e}.w3", d, ei, qin, hqb = 0, qgroup = eg))
            self.ex_w2.append(lin(f"ffn.experts.{e}.w2", ei, d, f"{key}.ffn.experts.{e}.down", hqb = 0, qgroup = eg))
        # engram projection
        self.eg_wkv = None
        if self.engram_pos is not None:
            cols = (dcfg.engram_max_ngram_size - 1) * dcfg.engram_n_heads
            self.eg_wkv = lin("engram.wkv", cols * dcfg.engram_head_dim, d * (dcfg.hc_mult + 1), f"{key}.engram.wkv")

        # tensors kept as they are
        names = [f"{key}.hc_{t}_{p}" for t in ("attn", "ffn") for p in ("fn", "base", "scale")]
        names += [f"{key}.attn_norm.weight", f"{key}.ffn_norm.weight", f"{a}.attn_sink", f"{a}.q_norm.weight",
                  f"{a}.kv_norm.weight", f"{key}.ffn.gate.weight", f"{key}.ffn.gate.bias"]
        opt = [f"{key}.ffn.gate.bias_vl"]
        if self.is_kv:
            names += [f"{a}.compressor.wkv.weight", f"{a}.compressor.norm.weight", f"{a}.indexer.wk.weight",
                      f"{a}.indexer.k_norm.weight"]
            if dcfg.ratio(layer_idx) > 1:
                names += [f"{a}.compressor.wgate.weight"]
        if self.is_idx:
            names += [f"{a}.indexer.weights_proj.weight"]
        if self.engram_pos is not None:
            names += [f"{key}.engram.q_weight", f"{key}.engram.k_weight"]
        self.raw = DSV41Raw(config, f"{key}.raw", names, opt)
        self.register_submodule(self.raw)
        self._graph = None
        self._w = None

    def _tier_register(self, device):
        """TierSet when the SSD expert store is on and this trunk block qualifies (CUDA, no pruned experts), else None."""
        if (self.mtp or self.pruned or device is None or torch.device(device).type != "cuda" or
                os.environ.get("EXL3_EXPERT_BUDGET_GIB", "") == ""):
            return None
        from .expert_store import get_store
        return get_store(self.config).register(_TierView(self), device)

    @override
    def load(self, device: torch.device, **kwargs):
        ts = self._tier_register(device)
        if ts is None:
            return super().load(device, **kwargs)
        # tiered: the routed experts are registered (shell tensors), their bytes are read on demand into the store's slots
        self.device = device
        experts = {id(m) for m in self.ex_w1 + self.ex_w3 + self.ex_w2}
        for m in self.modules:
            if id(m) not in experts:
                m.load(device, **kwargs)
        self.tier = ts

    @override
    def unload(self):
        super().unload()
        self._graph = None
        self._w = None
        if self.tier is not None:
            self.tier.store.unregister(self.tier)
            self.tier = None

    @override
    def optimizer_targets(self):
        return [[m.key] for m in self.modules if isinstance(m, Linear)]

    # ------------------------------------------------------------------ layer weights for the graph
    def _bundle(self, params: dict) -> _LW:
        dc, key, raw = self.dcfg, self.key, self.raw
        a = f"{key}.attn"
        ML = lambda m: ModLin(m, params)
        f32 = lambda n: raw.t[n].float()
        w = _LW()
        w.l = self.l
        for t in ("attn", "ffn"):
            for p in ("fn", "base", "scale"):
                setattr(w, f"hc_{t}_{p}", f32(f"{key}.hc_{t}_{p}"))
        w.attn_norm, w.ffn_norm = f32(f"{key}.attn_norm.weight"), f32(f"{key}.ffn_norm.weight")
        w.sink = f32(f"{a}.attn_sink")
        w.q_norm, w.kv_norm = f32(f"{a}.q_norm.weight"), f32(f"{a}.kv_norm.weight")
        w.wq_a, w.wq_b, w.wkv, w.wo_b = ML(self.wq_a), ML(self.wq_b), ML(self.wkv), ML(self.wo_b)
        wo = [ML(m) for m in self.wo_a]
        w.wo_a = None
        w.wo_a_fn = lambda o: torch.stack([wo[g](o[:, :, g]) for g in range(dc.o_groups)], dim = 2)
        w.comp = None
        if self.is_kv:
            c = _LW()
            c.wkv = DenseLin(f32(f"{a}.compressor.wkv.weight"))
            c.wgate = DenseLin(f32(f"{a}.compressor.wgate.weight")) if dc.ratio(self.l) > 1 else None
            c.norm = f32(f"{a}.compressor.norm.weight")
            w.comp = c
        w.idx = None
        if self.is_idx:
            ix = _LW()
            ix.wq_b = ML(self.idx_wq_b)
            ix.weights_proj = DenseLin(f32(f"{a}.indexer.weights_proj.weight"))
            ix.owns_k = self.is_kv
            if self.is_kv:
                ix.wk = DenseLin(f32(f"{a}.indexer.wk.weight"))
                ix.k_norm = f32(f"{a}.indexer.k_norm.weight")
            w.idx = ix
        w.gate_w, w.gate_b = f32(f"{key}.ffn.gate.weight"), f32(f"{key}.ffn.gate.bias")
        w.n_exp, w.topk = self.n_exp, self.topk
        w.engram = None
        return w

    def _moe_fn(self, params: dict):
        dc, cap = self.dcfg, "capture" in params
        mode = self.moe_mode if cap else "routed"
        nocap = {k: v for k, v in params.items() if k != "capture"}
        w = self._w

        def expert_fn(e, capture):
            p = params if capture else nocap
            hook_mid = None
            if capture and cap:
                # w2 sees the routing-weighted intermediate in the reference forward, but the EXL3 kernels feed it
                # the unweighted activation (the weight is applied to the output): capture that one
                w2 = self.ex_w2[e]
                hook_mid = lambda h: torch_capture_H(w2, _pad_last(h, w2.in_features), params)
            return ModLin(self.ex_w1[e], p), ModLin(self.ex_w3[e], p), ModLin(self.ex_w2[e], nocap), None, hook_mid

        def shared_fn(x):
            return expert_ffn(x, ModLin(self.sh_w1, params), ModLin(self.sh_w3, params), ModLin(self.sh_w2, params),
                              dc.swiglu_limit)

        def moe(l, x):
            y, _ = moe_calibrate(dc, x, w.gate_w, w.gate_b, self.topk, self.n_exp, expert_fn, shared_fn, mode)
            return y
        return moe

    # ------------------------------------------------------------------ forward
    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype = None) -> torch.Tensor:
        if self.mtp:
            raise RuntimeError("MTP stages are calibrated by dsv41.calib.DSparkCalib / dsv41.convert_driver, "
                               "not by a state forward")
        if "quant_preserve" not in params:
            raise RuntimeError("DS4.1 blocks run inside the conversion flow only (they need params['quant_preserve'])")
        dc = self.dcfg
        ho = params["quant_preserve"].setdefault("dsv41", new_handoff())
        g = self.bind(params)
        h = x.float()
        if self.engram_pos is not None:
            h = self._engram(h, params)
        out = g.run_block(self.l, h, ho)
        self.unbind()
        return out

    def bind(self, params: dict) -> CalibGraph:
        """Graph with this block's weights (bound to `params`: capture dict, ...) and MoE installed."""
        if self._graph is None:
            self._graph = CalibGraph(self.dcfg, torch.float32, self.device)
        g = self._graph
        self._w = self._bundle(params)
        g.lw = {self.l: self._w}
        g.moe_fn = self._moe_fn(params)
        return g

    def unbind(self):
        if self._graph is not None:
            self._graph.lw = {}
        self._w = None

    def _engram(self, h: torch.Tensor, params: dict) -> torch.Tensor:
        dc, cfg = self.dcfg, self.config
        ids = params["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        eg = cfg.engram_runtime()
        hasher = NgramHasher(eg["layout"], eg["token_map"], dc.engram_pad_token_id, eg["compressed_vocab"],
                             eg["multipliers"], 1, ids.shape[1], h.device)
        hashes = hasher(ids.to(h.device), 0)
        qk = self.raw.f32(f"{self.key}.engram.q_weight") * self.raw.f32(f"{self.key}.engram.k_weight")
        return engram_inject(h, hashes[:, :, self.engram_pos, :], cfg.engram_store(), self.engram_pos,
                             ModLin(self.eg_wkv, params), qk, dc.hc_mult, dc.hidden_size, dc.rms_norm_eps)


# ------------------------------------------------------------------------------------------------ final + head
class DSV41Final(Module):
    """hc_pre of the last block's output with its pending pre-mix, then the final RMS norm (weight `norm`)."""

    def __init__(self, config, key: str, dcfg: DSV41Config):
        super().__init__(config, key, None)
        self.dcfg = dcfg
        self.w = None

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device, **kwargs):
        self.device = device
        self.w = self.config.stc.get_tensor(self.key + ".weight", device, allow_bf16 = True, no_defer = True)

    @override
    def unload(self):
        self.device = None
        self.w = None

    @override
    def get_tensors(self):
        return {self.key + ".weight": self.w} if self.device is not None else {}

    @override
    def weights_numel(self):
        return 0

    @override
    def forward(self, x, params, out_dtype = None):
        dc = self.dcfg
        ho = params["quant_preserve"]["dsv41"]
        h = ops.hc_pre(x.float(), ho["pre_mix"][dc.num_hidden_layers - 1])
        return ops.rmsnorm(h, self.w.float(), dc.rms_norm_eps)


class DSV41Head(Module):
    def __init__(self, config, key: str, dcfg: DSV41Config, alt_key = None):
        super().__init__(config, key, None)
        self.head = Linear(config, key, dcfg.hidden_size, dcfg.vocab_size, qmap = "block", qbits_key = "head_bits",
                           alt_key = alt_key, out_dtype = torch.half, caps = {"logits_output": True},
                           qgroup = key)
        self.register_submodule(self.head)

    @override
    def optimizer_targets(self):
        return [[self.head.key]]

    @override
    def forward(self, x, params, out_dtype = None):
        return ModLin(self.head, params)(x)
