"""Adapter #1: DeepSeek-V4.1-Flash (the reference pipeline, tools/ds41).

Plan side: tensor names, FP8/FP4 + `.scale` source pairs, Engram tables copied raw, MTP block `mtp.J`.
Capture side: wraps the existing `exllamav3.dsv41` graph (CalibGraph.block_pre / block_post, moe_route, expert_ffn);
nothing of the reference math is re-implemented.
"""
from __future__ import annotations

import re
from typing import Optional

from ..arch import ArchAdapter, Cls, Expert, register

RE_ATTN_ROLE = [
    ("q_a", re.compile(r"\.attn\.wq_a\.weight$")), ("kv", re.compile(r"\.attn\.wkv\.weight$")),
    ("q_b", re.compile(r"\.attn\.wq_b\.weight$")), ("wo_a", re.compile(r"\.attn\.wo_a\.weight$")),
    ("wo_b", re.compile(r"\.attn\.wo_b\.weight$")), ("idx_q_b", re.compile(r"\.attn\.indexer\.wq_b\.weight$")),
]
RE_EXPERT = re.compile(r"^(layers\.\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight$")
RE_MTP_EXPERT = re.compile(r"^mtp\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")
RE_LAYER = re.compile(r"^(layers|mtp)\.(\d+)\.")
RE_ENGRAM = re.compile(r"^layers\.(\d+)\.engram\.")


@register
class DS41Adapter(ArchAdapter):
    name = "ds41"
    attn_roles = ("q_a", "kv", "q_b", "wo_a", "wo_b", "idx_q_b")
    alloc_schema = "ds41-alloc/1"
    o_groups = 8
    n_layers = 40
    n_experts = 384
    topk = 6
    n_mtp = 3
    hidden = 5120
    expert_inter = 2304
    hc_mult = 4
    source_bytes_per_weight = 0.5

    def __init__(self, config=None):
        super().__init__(config)
        t = (config or {}).get("text_config", config or {})
        self.text = t
        if t:
            self.n_layers = t.get("num_hidden_layers", self.n_layers)
            self.n_experts = t.get("n_routed_experts", t.get("num_experts", self.n_experts))
            self.topk = t.get("num_experts_per_tok", self.topk)
            self.n_mtp = t.get("num_nextn_predict_layers", self.n_mtp)
            self.hidden = t.get("hidden_size", self.hidden)
            self.expert_inter = t.get("moe_intermediate_size", self.expert_inter)
            self.hc_mult = t.get("hc_mult", self.hc_mult)
            self.o_groups = t.get("o_groups", self.o_groups)

    # ------------------------------------------------------------------ plan side
    def unit_of(self, name):
        m = RE_ENGRAM.match(name)
        if m:
            return f"layers.{m.group(1)}.engram"
        m = RE_LAYER.match(name)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
        return "global"

    def expert_of(self, name) -> Optional[Expert]:
        m = RE_EXPERT.match(name)
        return Expert(m.group(1), int(m.group(1).split(".")[1]), int(m.group(2)), m.group(3)) if m else None

    def classify(self, name, T):
        t = T[name]
        if RE_ENGRAM.match(name):
            return Cls("raw", "engram")
        if name.endswith(".scale") and name[:-6] + ".weight" in T:
            return Cls("skip")
        if name.endswith(".weight") and name[:-7] + ".scale" in T:
            out_f = t["shape"][0]
            in_f = t["shape"][1] * (2 if t["dtype"] == "I8" else 1)
            c = dict(action="quant", weight=name, scale=name[:-7] + ".scale", out_f=out_f, in_f=in_f, base=name[:-7])
            m = RE_EXPERT.match(name)
            if m:
                return Cls(kind="routed", expert=Expert(m.group(1), int(m.group(1).split(".")[1]), int(m.group(2)),
                                                        m.group(3)), **c)
            if name.startswith("mtp."):
                return Cls(kind="mtp_routed" if RE_MTP_EXPERT.match(name) else "mtp", **c)
            if ".shared_experts." in name:
                return Cls(kind="shared", **c)
            if name.endswith("attn.wo_a.weight"):
                g = out_f // self.o_groups
                return Cls(kind="attn", role="wo_a", slices=[(f".slice.{i}", (i * g, (i + 1) * g))
                                                             for i in range(self.o_groups)], **c)
            role = next((r for r, rx in RE_ATTN_ROLE if rx.search(name)), None)
            return Cls(kind="attn", role=role, **c)
        if name == "head.weight":
            return Cls("quant", "head", weight=name, scale=None, out_f=t["shape"][0], in_f=t["shape"][1], base="head")
        return Cls("copy")

    def dense_hessian_dims(self):
        t = self.text if hasattr(self, "text") else {}
        d = self.hidden
        heads, hd = t.get("num_attention_heads", 64), t.get("head_dim", 512)
        g, r = self.o_groups, t.get("o_lora_rank", 1024)
        # attn.in (wq_a, wkv), q_b (+indexer wq_b), one o group per wo_a slice, o_b, shared in / down
        return [d, t.get("q_lora_rank", 1280)] + [heads * hd // g] * g + [g * r, d, self.expert_inter]

    def sheet_dims(self):
        return dict(hidden=self.hidden, expert_inter=self.expert_inter, n_experts=self.n_experts, topk=self.topk,
                    n_layers=self.n_layers, hc_mult=self.hc_mult)

    # ------------------------------------------------------------------ capture side
    def capture_runner(self, cfg, device="cpu", embed_w=None, engram=None, **kw):
        from ..runners.ds41 import DS41Runner
        return DS41Runner(cfg, device, embed_w, engram, **kw)
