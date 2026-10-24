from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import (
    RMSNorm, Embedding, TransformerBlock, Attention, SlidingAttention, SWAState,
    GatedMLP, BlockSparseMLP, Linear,
)
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence


class FlintConfig(Config):
    arch_string = "FlintForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": FlintModel},
            **kwargs
        )

        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        self.head_dim = self.read_cfg(int, "head_dim", None) or self.hidden_size // self.num_q_heads

        # layer_types: full_attention / sliding_attention. The config window counts the current token,
        # the kernels' left window keeps window + 1 keys including the query
        self.layer_types = self.read_cfg(list, "layer_types", no_default)
        assert len(self.layer_types) == self.num_hidden_layers
        self.sliding_window = self.read_cfg(int, "sliding_window", None)
        if "sliding_attention" in self.layer_types:
            assert self.sliding_window, "Flint: sliding_window must be set"
        # Layers without positional encoding (all full-attention layers)
        self.no_rope_layers = set(self.read_cfg(list, "no_rope_layer_indices", []))
        self.assert_cfg(str, "softmax_type", "learnable", True)

        # MLP / MoE: dense layers listed in mlp_only_layers, sigmoid top-k router renormalised over the picks,
        # plus an ungated shared expert
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", no_default)
        self.shared_expert_intermediate_size = self.read_cfg(int, "shared_expert_intermediate_size", 0)
        self.num_experts = self.read_cfg(int, "num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", no_default)
        self.mlp_only_layers = set(self.read_cfg(list, "mlp_only_layers", []))
        self.assert_cfg(bool, "norm_topk_prob", True, True)
        self.assert_cfg(str, "router_score_function", "sigmoid", True)

        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # fp16 overflow guard: the routed experts of layers 46 and 48 take act(gate) * up to ~1.6e5 / 2.3e5 on
        # some tokens (BF16 reference, 15K mixed tokens), past the fp16 max. Their intermediates are divided by
        # interm_div and the routing weights multiplied back (BlockSparseMLP.interm_div). Shared experts peak at 7e3.
        self.interm_div_layers = {int(k): float(v) for k, v in
                                  self.read_cfg(dict, "exl3_interm_div_layers", {"46": 16.0, "48": 16.0}).items()}

        # RoPE only on sliding layers (NEOX)
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)


class FlintModel(Model):
    config_class = FlintConfig

    def __init__(
        self,
        config: FlintConfig,
        key_prefix: str = "model",
        swa_full: bool = False,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        # swa_full = True keeps every layer on the paged full-length cache (for A/B); the default routes the
        # 35 sliding-window layers through SlidingAttention's per-slot window ring (513 slots, not full length)
        self.swa_full = swa_full

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            swa = config.layer_types[idx] == "sliding_attention"
            attn_kwargs = dict(
                config = config,
                key = f"{key_prefix}.layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.head_dim,
                num_q_heads = config.num_q_heads,
                num_kv_heads = config.num_kv_heads,
                rope_settings = None if idx in config.no_rope_layers else config.rope_settings,
                sm_scale = None,
                key_q = "q_proj",
                key_k = "k_proj",
                key_v = "v_proj",
                key_o = "o_proj",
                key_sinks = "softmax_offset",
                qmap = "block.attn",
                q_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.self_attn.q_norm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                k_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.self_attn.k_norm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                out_dtype = torch.float,
                select_hq_bits = 2,
            )
            if swa and not swa_full:
                attn = SlidingAttention(sliding_window = config.sliding_window - 1, **attn_kwargs)
            else:
                attn = Attention(sliding_window = config.sliding_window - 1 if swa else -1, **attn_kwargs)

            if idx in config.mlp_only_layers:
                mlp = GatedMLP(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    select_hq_bits = 1,
                )
            else:
                mlp = BlockSparseMLP(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.moe_intermediate_size,
                    num_experts = config.num_experts,
                    num_experts_per_tok = config.num_experts_per_tok,
                    key_up = "experts.{expert_idx}.up_proj",
                    key_gate = "experts.{expert_idx}.gate_proj",
                    key_down = "experts.{expert_idx}.down_proj",
                    key_routing_gate = "gate",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    interm_div = config.interm_div_layers.get(idx, 1.0),
                    router_type = "dots",
                    routed_scaling_factor = 1.0,
                    shared_experts = GatedMLP(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.mlp.shared_expert",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.shared_expert_intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        select_hq_bits = 2,
                    ) if config.shared_expert_intermediate_size else None,
                )

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = mlp,
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = f"{key_prefix}.embed_tokens"

        self.modules += [
            RMSNorm(
                config = config,
                key = f"{key_prefix}.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "lm_head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True

        self.caps.update({
            "supports_tp": False,
        })

        # SWA layers keep their KV in a per-slot window ring rather than the paged cache
        self.recurrent_state_cls = None
        if not self.swa_full:
            self.caps.update({
                "recurrent_states": True,
                "default_recurrent_checkpoint_interval": 2048,
            })
            self.recurrent_state_cls = SWAState


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        if not self.swa_full:
            prepare_for_recurrence(input_ids, params, self)
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<|system|>{system_prompt}"
        p += f"<|user|>{prompt}<|assistant|>"
        return p
