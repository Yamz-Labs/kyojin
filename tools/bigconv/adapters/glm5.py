"""Adapter stub: GLM-5.x-Flash (zai-org/GLM-5.3-Flash, model_type glm5_next; real config + index in fixtures/).

Facts read from the release (config.json + model.safetensors.index.json, 45 trunk layers + 1 MTP layer):
  * source: fp8 e4m3, 128x128 blocks, `<name>.weight_scale_inv`; KDA projections, indexer wk/wq_b, router, hc_* are bf16.
  * 45 layers: 3 dense MLP (0-2), 42 sparse MoE (288 routed experts, top 8, 1 shared, sigmoid router + e_score bias).
  * attention: 34 KDA linear-attention layers (q/k/v/f_a/f_b/g_a/g_b/b_proj, q/k/v short conv, A_log, dt_bias, o_norm,
    o_proj) and 11 + 1 (MTP) deepseek-sparse MLA layers (q_a, q_b, kv_a_proj_with_mqa, kv_b, o_proj, lightning indexer
    wq_b / wk / weights_proj / k_norm / kpool compress ape+gate), qk_rope_head_dim = 0 (NoPE).
  * hyper-connections (mhc): hc_mult 4, hc_{attn,ffn}_{fn,base,scale} per layer, same family as DS4.1.
  * MTP layer (index 45): eh_proj, enorm, hnorm, shared_head.norm + a normal sparse block.
Plan side is complete. Capture forward is NOT implemented: TODO_OPS lists what the runner needs.
"""
from __future__ import annotations

import re

from ..arch import TableAdapter, register

P = r"^model\.language_model\.layers\.(\d+)\."


def _c(a, k, d=0):
    return a.text.get(k, d)


@register
class GLM5Adapter(TableAdapter):
    name = "glm5"
    layer_re = re.compile(P)
    expert_re = re.compile(r"^(model\.language_model\.layers\.(\d+))\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    expert_first_proj = "gate_proj"
    scale_suffix = ".weight_scale_inv"
    lang_prefixes = ("model.language_model.", "lm_head")
    attn_roles = ("q_a", "q_b", "kv_a", "kv_b", "o", "idx_wq_b", "idx_wk",
                  "kda_q", "kda_k", "kda_v", "kda_fa", "kda_fb", "kda_ga", "kda_gb", "dense_mlp")
    alloc_schema = "bigconv-alloc/1"
    source_bytes_per_weight = 1.0
    hc_mult = 4

    def __init__(self, config=None):
        super().__init__(config)
        self.text = (config or {}).get("text_config", config or {})
        t = self.text
        self.n_layers = t.get("num_hidden_layers", 45)
        self.n_experts = t.get("n_routed_experts", 288)
        self.topk = t.get("num_experts_per_tok", 8)
        self.n_mtp = t.get("num_nextn_predict_layers", 1)
        self.hidden = t.get("hidden_size", 4096)
        self.expert_inter = t.get("moe_intermediate_size", 2048)
        self.hc_mult = t.get("hc_mult", 4)
        self.mtp_from_layer = self.n_layers
        self.first_dense = t.get("first_k_dense_replace", 3)
        self.ROLES = self._roles()

    def _roles(self):
        R = re.compile
        kd = lambda a: a.text.get("linear_attn_config", {}).get("num_heads", 64) * a.text.get("linear_attn_config", {}).get("head_dim", 128)
        hd = lambda a: a.text.get("linear_attn_config", {}).get("head_dim", 128)
        H = lambda a: a.hidden
        qa = lambda a: a.text.get("q_lora_rank", 1536)
        nh = lambda a: a.text.get("num_attention_heads", 64)
        qk = lambda a: a.text.get("qk_nope_head_dim", 256) + a.text.get("qk_rope_head_dim", 0)
        vh = lambda a: a.text.get("v_head_dim", 256)
        kvl = lambda a: a.text.get("kv_lora_rank", 512)
        ihd = lambda a: a.text.get("index_head_dim", 128)
        inh = lambda a: a.text.get("index_n_heads", 32)
        mi = lambda a: a.text.get("moe_intermediate_size", 2048)
        di = lambda a: a.text.get("intermediate_size", 12288)
        return [
            (R(r"\.mlp\.experts\.\d+\.(gate|up)_proj\.weight$"), "routed", None, lambda a: (mi(a), H(a))),
            (R(r"\.mlp\.experts\.\d+\.down_proj\.weight$"), "routed", None, lambda a: (H(a), mi(a))),
            (R(r"\.mlp\.shared_experts\.(gate|up)_proj\.weight$"), "shared", None, lambda a: (mi(a) * a.text.get("n_shared_experts", 1), H(a))),
            (R(r"\.mlp\.shared_experts\.down_proj\.weight$"), "shared", None, lambda a: (H(a), mi(a) * a.text.get("n_shared_experts", 1))),
            (R(r"layers\.\d+\.mlp\.(gate|up)_proj\.weight$"), "dense", "dense_mlp", lambda a: (di(a), H(a))),
            (R(r"layers\.\d+\.mlp\.down_proj\.weight$"), "dense", "dense_mlp", lambda a: (H(a), di(a))),
            (R(r"\.mlp\.gate\.weight$"), "copy", None, lambda a: (a.n_experts, H(a))),
            (R(r"\.self_attn\.q_a_proj\.weight$"), "attn", "q_a", lambda a: (qa(a), H(a))),
            (R(r"\.self_attn\.q_b_proj\.weight$"), "attn", "q_b", lambda a: (nh(a) * qk(a), qa(a))),
            (R(r"\.self_attn\.kv_a_proj_with_mqa\.weight$"), "attn", "kv_a", lambda a: (kvl(a) + a.text.get("qk_rope_head_dim", 0), H(a))),
            (R(r"\.self_attn\.kv_b_proj\.weight$"), "attn", "kv_b", lambda a: (nh(a) * (a.text.get("qk_nope_head_dim", 256) + vh(a)), kvl(a))),
            (R(r"\.self_attn\.indexer\.wq_b\.weight$"), "attn", "idx_wq_b", lambda a: (inh(a) * ihd(a), qa(a))),
            (R(r"\.self_attn\.indexer\.wk\.weight$"), "attn", "idx_wk", lambda a: (ihd(a), H(a))),
            (R(r"\.self_attn\.indexer\.weights_proj\.weight$"), "copy", None, lambda a: (inh(a), H(a))),
            (R(r"\.self_attn\.(q|k|v)_proj\.weight$"), "attn", "kda_q", lambda a: (kd(a), H(a))),
            (R(r"\.self_attn\.f_a_proj\.weight$"), "attn", "kda_fa", lambda a: (hd(a), H(a))),
            (R(r"\.self_attn\.f_b_proj\.weight$"), "attn", "kda_fb", lambda a: (kd(a), hd(a))),
            (R(r"\.self_attn\.g_a_proj\.weight$"), "attn", "kda_ga", lambda a: (hd(a), H(a))),
            (R(r"\.self_attn\.g_b_proj\.weight$"), "attn", "kda_gb", lambda a: (kd(a), hd(a))),
            (R(r"\.self_attn\.b_proj\.weight$"), "copy", None, lambda a: (a.text.get("linear_attn_config", {}).get("num_heads", 64), H(a))),
            # o_proj: KDA layers [H, heads*head_dim]; MLA layers [H, heads*v_head_dim]; same number here (64*128 vs 64*256 differ)
            (R(r"\.self_attn\.o_proj\.weight$"), "attn", "o", None),
            (R(r"\.eh_proj\.weight$"), "attn", "mtp_eh", lambda a: (H(a), 2 * H(a))),
            (R(r"^lm_head\.weight$"), "head", None, lambda a: (a.text.get("vocab_size", 154880), H(a))),
        ]

    def shape_of(self, name, names=None):
        if name.endswith(".self_attn.o_proj.weight"):
            t, la = self.text, self.text.get("linear_attn_config", {})
            mla = names is not None and name[:-7] + self.scale_suffix in names
            return (self.hidden, t.get("num_attention_heads", 64) * t.get("v_head_dim", 256)) if mla else \
                (self.hidden, la.get("num_heads", 64) * la.get("head_dim", 128))
        return super().shape_of(name, names)

    def dense_hessian_dims(self):
        t = self.text
        mi = t.get("moe_intermediate_size", 2048)
        # MLA layer: attn.in (q_a, kv_a), q_b input, kv_b input, o input, indexer; KDA layer: attn.in, f_b/g_b input, o input;
        # sparse layers: shared in/down. Largest layer type = MLA + MoE.
        return [self.hidden, t.get("q_lora_rank", 1536), t.get("kv_lora_rank", 512),
                t.get("num_attention_heads", 64) * t.get("v_head_dim", 256), self.hidden, mi]

    def sheet_dims(self):
        return dict(hidden=self.hidden, expert_inter=self.expert_inter, n_experts=self.n_experts, topk=self.topk,
                    n_layers=self.n_layers, hc_mult=self.hc_mult)

    TODO_OPS = [
        "KDA (Kimi Delta Attention) chunked recurrence over a full row: q/k/v + short causal conv (kernel 4) + "
        "f_a/f_b decay gate (A_log, dt_bias, gate_lower_bound -5) + g_a/g_b output gate + b_proj beta + o_norm + o_proj "
        "(34 layers, the dominant new op; fp32 reference recurrence, no kernel needed on CPU/ROCm for capture)",
        "MLA NoPE attention (qk_rope_head_dim 0): q_a -> q_a_layernorm -> q_b ; kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b",
        "Lightning indexer with kpool compression (index_kpool 4, compress ape + gate, k_norm with bias) producing the "
        "top-k 2048 block set for the 12 DSA layers; indexer_types per layer (full/share)",
        "Hyper-connections (mhc, hc_mult 4, sinkhorn 20): same hc_pre/hc_post as DS4.1 -> reuse ops.hc_*",
        "Router: sigmoid + e_score_correction_bias (noaux_tc), n_group 1, norm_topk_prob, scale 2.5 (ds41 ops.route has the form)",
        "Expert FFN: silu gate * up with swiglu_limit 10 clamp (ds41 ops.swiglu applies the same clamp form)",
        "MTP layer 45: enorm/hnorm of (embedding, previous hidden) -> eh_proj [4096, 8192] -> one sparse block -> shared_head.norm",
        "Dense MLP layers 0-2 (intermediate 12288): plain gated MLP, no routing",
    ]

    def capture_runner(self, *a, **k):
        raise NotImplementedError("glm5 capture forward: TODO, ops needed:\n  - " + "\n  - ".join(self.TODO_OPS))
