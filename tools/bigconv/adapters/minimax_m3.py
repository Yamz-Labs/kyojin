"""Adapter stub: MiniMax M3 (MiniMaxAI/MiniMax-M3, MiniMaxM3SparseForConditionalGeneration; real config + index in fixtures/).

Facts read from the release (config.json + model.safetensors.index.json, bf16, 869 GB, 60 layers):
  * source: bf16 `.weight`, no scale tensors. Vision tower + projector + patch_merge_mlp are bf16 and copied.
  * layers 0-2 dense MLP (intermediate 12288); layers 3-59 MoE: 128 routed experts (w1 gate, w3 up, w2 down, 3072 x 6144),
    top 4, 1 shared expert (3072), sigmoid router with e_score_correction_bias (use_routing_bias).
  * attention: GQA 64 q heads / 4 kv heads, head_dim 128, per-head q/k RMS norm (gemma-style), rotary_dim 64 of 128,
    plus a sparse-attention index branch on layers 3-59 (index_q_proj/index_k_proj + norms, 4 index heads x 128, top 16 blocks of 128).
  * act: "swigluoai" (alpha 1.702, limit 7.0), routed_scaling_factor 2.0.
  * MTP: num_mtp_modules 7 in the config but NO mtp tensors in the release index -> n_mtp = 0 (nothing to convert).
Plan side is complete. Capture forward is NOT implemented: TODO_OPS lists what the runner needs.
"""
from __future__ import annotations

import re

from ..arch import TableAdapter, register

P = r"^language_model\.model\.layers\.(\d+)\."


@register
class MiniMaxM3Adapter(TableAdapter):
    name = "minimax_m3"
    layer_re = re.compile(P)
    expert_re = re.compile(r"^(language_model\.model\.layers\.(\d+))\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$")
    expert_first_proj = "w1"
    scale_suffix = None
    lang_prefixes = ("language_model.",)
    attn_roles = ("q", "k", "v", "o", "idx_q", "idx_k", "dense_mlp")
    source_bytes_per_weight = 2.0
    hc_mult = 1
    mtp_from_layer = None

    def __init__(self, config=None):
        super().__init__(config)
        self.text = (config or {}).get("text_config", config or {})
        t = self.text
        self.n_layers = t.get("num_hidden_layers", 60)
        self.n_experts = t.get("num_local_experts", 128)
        self.topk = t.get("num_experts_per_tok", 4)
        self.n_mtp = 0
        self.hidden = t.get("hidden_size", 6144)
        self.expert_inter = t.get("intermediate_size", 3072)
        self.ROLES = self._roles()

    def _roles(self):
        R = re.compile
        H = lambda a: a.hidden
        hd = lambda a: a.text.get("head_dim", 128)
        nq = lambda a: a.text.get("num_attention_heads", 64) * hd(a)
        nk = lambda a: a.text.get("num_key_value_heads", 4) * hd(a)
        sp = lambda a: a.text.get("sparse_attention_config", {})
        ix = lambda a: sp(a).get("sparse_num_index_heads", 4) * sp(a).get("sparse_index_dim", 128)
        ei = lambda a: a.text.get("intermediate_size", 3072)
        di = lambda a: a.text.get("dense_intermediate_size", 12288)
        si = lambda a: a.text.get("shared_intermediate_size", 3072)
        return [
            (R(r"\.block_sparse_moe\.experts\.\d+\.(w1|w3)\.weight$"), "routed", None, lambda a: (ei(a), H(a))),
            (R(r"\.block_sparse_moe\.experts\.\d+\.w2\.weight$"), "routed", None, lambda a: (H(a), ei(a))),
            (R(r"\.block_sparse_moe\.shared_experts\.(gate|up)_proj\.weight$"), "shared", None, lambda a: (si(a), H(a))),
            (R(r"\.block_sparse_moe\.shared_experts\.down_proj\.weight$"), "shared", None, lambda a: (H(a), si(a))),
            (R(r"\.block_sparse_moe\.gate\.weight$"), "copy", None, lambda a: (a.n_experts, H(a))),
            (R(r"layers\.\d+\.mlp\.(gate|up)_proj\.weight$"), "dense", "dense_mlp", lambda a: (di(a), H(a))),
            (R(r"layers\.\d+\.mlp\.down_proj\.weight$"), "dense", "dense_mlp", lambda a: (H(a), di(a))),
            (R(r"\.self_attn\.q_proj\.weight$"), "attn", "q", lambda a: (nq(a), H(a))),
            (R(r"\.self_attn\.k_proj\.weight$"), "attn", "k", lambda a: (nk(a), H(a))),
            (R(r"\.self_attn\.v_proj\.weight$"), "attn", "v", lambda a: (nk(a), H(a))),
            (R(r"\.self_attn\.o_proj\.weight$"), "attn", "o", lambda a: (H(a), nq(a))),
            (R(r"\.self_attn\.index_q_proj\.weight$"), "attn", "idx_q", lambda a: (ix(a), H(a))),
            (R(r"\.self_attn\.index_k_proj\.weight$"), "attn", "idx_k", lambda a: (sp(a).get("sparse_index_dim", 128), H(a))),
            (R(r"^language_model\.lm_head\.weight$"), "head", None, lambda a: (a.text.get("vocab_size", 200064), H(a))),
        ]

    def dense_hessian_dims(self):
        t = self.text
        # attn.in (q, k, v, index_q, index_k share the input), o input, shared in / down
        return [self.hidden, t.get("num_attention_heads", 64) * t.get("head_dim", 128), self.hidden,
                t.get("shared_intermediate_size", 3072)]

    def sheet_dims(self):
        return dict(hidden=self.hidden, expert_inter=self.expert_inter, n_experts=self.n_experts, topk=self.topk,
                    n_layers=self.n_layers, hc_mult=1)

    TODO_OPS = [
        "GQA attention (64 q / 4 kv heads, head_dim 128), partial rotary (rotary_dim 64, theta 5e6), per-head q/k RMS norm "
        "with gemma-style (1 + w) scale (use_gemma_norm, qk_norm_type per_head)",
        "Sparse-attention index branch (layers 3-59): index_q_proj/index_k_proj + index_q/k norms -> block scores, "
        "top 16 blocks of 128 keys; sparse_disable_index_value flags per layer; reference = dense scores + top-k",
        "Residual stream is plain (no hyper-connections): row state is [L, hidden], hc_mult 1",
        "Router: sigmoid + e_score_correction_bias, top 4 of 128, scale 2.0; shared expert 3072 added",
        "Expert FFN: swigluoai (alpha 1.702, limit 7.0) on w1/w3, w2 down; dense layers 0-2 use the same act at 12288",
        "No MTP tensors in the release (num_mtp_modules 7 in config): nothing to capture for the draft",
        "Vision tower / projector: copied bf16, not quantized, not captured",
    ]

    def capture_runner(self, *a, **k):
        raise NotImplementedError("minimax_m3 capture forward: TODO, ops needed:\n  - " + "\n  - ".join(self.TODO_OPS))
