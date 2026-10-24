from __future__ import annotations
from typing_extensions import override
import os
import torch
import weakref

from ..cache import Cache
from ..ext import exllamav3_ext as ext
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear, BlockSparseMLP
from ..modules.arch_specific.qwen3_5_mtp import Qwen3_5MTPInputLayer
from ..modules.attn import prepare_for_attn
from ..util.tensor import get_for_device

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .qwen3_5 import Qwen3_5Config, Qwen3_5MoeConfig


class Qwen3_5MTPModel(Model):

    def __init__(
        self,
        config: Qwen3_5Config | Qwen3_5MoeConfig,
        use_moe: bool = False,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.use_moe = use_moe

        # Module list: optional embed, then pre_fc norms + fc, then num_mtp_layers * TransformerBlock, then norm
        self.input_layer = Qwen3_5MTPInputLayer(
            config = config,
            key = "mtp",
            key_pre_fc_norm_hidden = "mtp.pre_fc_norm_hidden",
            key_pre_fc_norm_embedding = "mtp.pre_fc_norm_embedding",
            key_fc = "mtp.fc",
            hidden_size = config.hidden_size,
            rms_norm_eps = config.rms_norm_eps,
            native_draft_len = 1,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
        )

        self.modules = [self.input_layer]

        self.first_block_idx = len(self.modules)
        self.attn_modules = []

        for idx in range(config.mtp_num_hidden_layers):
            attn = Attention(
                config = config,
                key = f"mtp.layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.head_dim,
                num_q_heads = config.num_q_heads,
                num_kv_heads = config.num_kv_heads,
                rope_settings = config.rope_settings,
                sm_scale = None,
                key_q = "q_proj",
                key_k = "k_proj",
                key_v = "v_proj",
                key_o = "o_proj",
                qmap = "block.attn",
                q_norm = RMSNorm(
                    config = config,
                    key = f"mtp.layers.{idx}.self_attn.q_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                k_norm = RMSNorm(
                    config = config,
                    key = f"mtp.layers.{idx}.self_attn.k_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                out_dtype = torch.float,
                interleaved_gate = True,
                qbits_key = "mtp_bits",
            )
            self.attn_modules.append(attn)

            self.modules.append(
                TransformerBlock(
                    config = config,
                    key = f"mtp.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"mtp.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"mtp.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    mlp = (
                        BlockSparseMLP(
                            config = config,
                            key = f"mtp.layers.{idx}.mlp",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.moe_intermediate_size,
                            num_experts = config.num_experts,
                            num_experts_per_tok = config.num_experts_per_tok,
                            key_up = "experts.{expert_idx}.up_proj",
                            key_gate = "experts.{expert_idx}.gate_proj",
                            key_down = "experts.{expert_idx}.down_proj",
                            key_gate_up_split = "experts.gate_up_proj",
                            key_down_split = "experts.down_proj",
                            key_routing_gate = "gate",
                            key_shared_gate = "shared_expert_gate",
                            transpose_fused_weights = False,
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            qbits_key = "mtp_bits",
                            shared_experts = GatedMLP(
                                config = config,
                                key = f"mtp.layers.{idx}.mlp.shared_expert",
                                hidden_size = config.hidden_size,
                                intermediate_size = config.shared_expert_intermediate_size,
                                key_up = "up_proj",
                                key_gate = "gate_proj",
                                key_down = "down_proj",
                                qmap = "block.mlp",
                                interm_dtype = torch.half,
                                out_dtype = torch.float,
                                qbits_key = "mtp_bits",
                                select_hq_bits = 2,
                            )
                        ) if use_moe else
                        GatedMLP(
                            config = config,
                            key = f"mtp.layers.{idx}.mlp",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.intermediate_size,
                            key_up = "up_proj",
                            key_gate = "gate_proj",
                            key_down = "down_proj",
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            qbits_key = "mtp_bits",
                        )
                    ),
                )
            )

        self.last_kv_module_idx = len(self.modules) - 1

        # Final norm
        self.final_norm = RMSNorm(
            config = config,
            key = "mtp.norm",
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.half,
            constant_bias = 1.0,
        )
        self.modules.append(self.final_norm)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 4,  # best measured performance
            "autosplit_load_fwd": False,
        })

        # Cross-references populated by attach_to()
        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # MTP doesn't take input_ids through Embedding here — embedding is handled in step()
        # But prepare_for_attn still wires up flash-attn params
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    def attach_to(self, target):
        """
        Bind to target model: borrow embed_tokens / lm_head and tell target to export hidden state.

        Qwen3.5/3.6 MTP consumes the trunk's post-final-norm hidden state. This differs from
        DeepSeek/GLM-style MTP heads, which consume a pre-norm residual stream.
        """
        self.input_layer.attached_model = weakref.ref(target)
        self.attached_model = weakref.ref(target)

        # Find the target's embedding (first module of class Embedding)
        target_embed = None
        for m in target.modules:
            if isinstance(m, Embedding):
                target_embed = m
                break
        assert target_embed is not None, "Could not locate target's Embedding module"
        self.target_embed = weakref.ref(target_embed)

        # lm_head is the last module
        assert isinstance(target.modules[-1], Linear), "Expected Linear lm_head as last target module"
        self.target_lm_head = weakref.ref(target.modules[-1])

        target_norm = target.modules[target.logit_layer_idx - 1]
        assert isinstance(target_norm, RMSNorm), "Expected target final RMSNorm immediately before lm_head"
        self.draft_verifier_params.update({
            "export_state_norm_keys": {target_norm.key},
        })


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    def _draft_subhead(self, lm):
        """The draft head restricted to the first EXL3_DRAFT_VOCAB vocabulary rows (a multiple of 128: the head's output
        Hadamard works on 128-row blocks, so a prefix of whole blocks is a self-contained EXL3 linear with the same
        trellis tiles). None when off or when the head is not a plain integer-K EXL3 linear. Every draft is verified by the
        target, so a draft that misses a rare id only costs an accept, never changes the text."""
        n = int(os.environ.get("EXL3_DRAFT_VOCAB", "0") or 0) // 128 * 128
        if n <= 0:
            return None
        cache = self.__dict__.setdefault("_dsub", {})
        if n in cache:
            return cache[n]
        sub = None
        inner = getattr(lm, "inner", None)
        if inner is not None and type(inner).__name__ == "LinearEXL3" and isinstance(inner.K, int) \
                and inner.bias is None and lm.pre_scale == 1.0 and lm.post_scale == 1.0 and lm.softcap == 0.0 \
                and n < lm.out_features:
            from ..modules.quant.exl3 import LinearEXL3
            sub = LinearEXL3(None, inner.in_features, n, suh = inner.suh, svh = inner.svh[:n].contiguous(),
                             trellis = inner.trellis[:, :n // 16].contiguous(), mcg = inner.mcg_tensor,
                             mul1 = inner.mul1_tensor, out_dtype = inner.out_dtype)
        cache[n] = sub
        return sub

    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        if not self.attached_model().loaded_tp:
            ll = self.attached_model().logit_layer_idx
            lm = self.attached_model().modules[ll]
            logits = lm.prepare_for_device(state, params)
            # EXL3_DRAFT_VOCAB=N: 1-row draft steps read only the first N vocabulary rows of the head (the common BPE ids).
            # Every 8th step runs both heads and compares the argmax: a language whose tokens sit above N (French, CJK ...)
            # switches the subset off until the probes agree again.
            sub = self._draft_subhead(lm) if state.shape[0] * state.shape[1] == 1 and not params.get("export_draft_logits") else None
            if sub is not None:
                st = self.__dict__.setdefault("_dsub_st", {"calls": 0, "ema": 0.0, "on": True})
                st["calls"] += 1
                probe = st["calls"] % 8 == 0
                if st["on"] or probe:
                    sl = sub.forward(logits, params)[..., :min(sub.out_features, self.attached_model().config.vocab_size)].float()
                    conf, ids = torch.max(sl, dim = -1)
                    pp = (1.0 / torch.exp(sl - conf.unsqueeze(-1)).sum(-1)).reshape(-1)    # max probability over the subset
                    if probe:
                        fl = lm.forward(logits, params)[..., :self.attached_model().config.vocab_size].float()
                        fconf, fids = torch.max(fl, dim = -1)
                        miss = float((fids != ids).any().item())
                        st["ema"] = 0.9 * st["ema"] + 0.1 * miss
                        st["on"] = st["ema"] < (0.12 if st["on"] else 0.04)     # hysteresis: ~1 % misses on English / code, ~20 % on French
                        conf, ids = fconf, fids
                        pp = (1.0 / torch.exp(fl - fconf.unsqueeze(-1)).sum(-1)).reshape(-1)
                    params["draft_p"] = pp
                    if params.get("export_draft_conf"):
                        params["draft_conf"] = conf
                    return ids
            logits = lm.forward(logits, params)
            lim = getattr(self, "vocab_limit", None)   # set by the Generator: the tokenizer's piece count when the head is padded
            if lim is not None:
                logits = logits[..., :lim]   # the padding columns have no piece: a draft there is rejected for sure
            if params.get("export_draft_conf"):
                # Per-position confidence for the generator's draft truncation: the argmax logit
                # value, over the unpadded vocabulary
                logits = logits[..., :self.attached_model().config.vocab_size]
                conf, ids = torch.max(logits, dim = -1)
                params["draft_conf"] = conf
                return ids
            return torch.argmax(logits, dim = -1)
        else:
            state = self.attached_model().tp_producer.send(state)
            argmax = self.attached_model().tp_dispatch_lm_head_argmax((state, {}))
            return argmax


class Qwen3_5MoeMTPModel(Qwen3_5MTPModel):

    def __init__(
        self,
        config: Qwen3_5Config | Qwen3_5MoeConfig,
        **kwargs
    ):
        super().__init__(config, use_moe = True, **kwargs)

