from __future__ import annotations
from typing_extensions import override
import os
import torch
import weakref

from ..model.config import Config
from ..model.model import Model
from ..modules import RMSNorm, Embedding, TransformerBlock, MLAttention, GatedMLP, Linear, BlockSparseMLP
from ..modules.arch_specific.qwen3_5_mtp import Qwen3_5MTPInputLayer
from ..modules.attn import prepare_for_attn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .glm5_next import Glm5NextConfig


class Glm5NextMTPModel(Model):
    """
    GLM-5.3-Flash MTP head: DeepSeek-V3 shape (enorm/hnorm/eh_proj concat into one trunk-style
    decoder layer, shared_head.norm before the shared lm_head), stored as last indexed layer
    Unlike the trunk layers it carries NO mHC tensors -- it is a plain residual block operating
    on the collapsed (post-mean, post-norm) trunk state.

    The config's index_share_for_mtp_iteration flag would let draft iterations reuse the
    trunk's top-k selection; as with GLM-5.2, this implementation lets the layer's own
    indexer score instead (self-consistent, and identical below index_topk context).
    """

    def __init__(
        self,
        config: Glm5NextConfig,
        key_prefix: str = "model.language_model",
        **kwargs
    ):
        super().__init__(config, **kwargs)

        first_mtp_layer = config.num_hidden_layers

        # Input layer: normed token embedding concatenated with normed target hidden state,
        # projected 2H -> H. Same mechanism as GLM-5.2/Qwen3.5/HyV3 MTP with DeepSeek-V3
        # tensor names and plain RMSNorms
        self.input_layer = Qwen3_5MTPInputLayer(
            config = config,
            key = f"{key_prefix}.layers.{first_mtp_layer}.input",
            key_pre_fc_norm_hidden = f"{key_prefix}.layers.{first_mtp_layer}.hnorm",
            key_pre_fc_norm_embedding = f"{key_prefix}.layers.{first_mtp_layer}.enorm",
            key_fc = f"{key_prefix}.layers.{first_mtp_layer}.eh_proj",
            hidden_size = config.hidden_size,
            rms_norm_eps = config.rms_norm_eps,
            native_draft_len = 1,
            out_dtype = torch.float,
            qbits_key = "mtp_bits",
            constant_bias = 0.0,
        )

        self.modules = [self.input_layer]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_mtp_layers):
            key = f"{key_prefix}.layers.{first_mtp_layer + idx}"
            self.modules.append(
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = MLAttention(
                        config = config,
                        key = f"{key}.self_attn",
                        layer_idx = idx,
                        hidden_size = config.hidden_size,
                        num_q_heads = config.num_q_heads,
                        kv_lora_rank = config.kv_lora_rank,
                        qk_nope_head_dim = config.qk_nope_head_dim,
                        qk_rope_head_dim = config.qk_rope_head_dim,
                        v_head_dim = config.v_head_dim,
                        rope_settings = None,
                        q_lora_rank = config.q_lora_rank,
                        sm_scale = config.sm_scale,
                        rms_norm_eps = config.rms_norm_eps,
                        qmap = "block.attn",
                        out_dtype = torch.float,
                        select_hq_bits = 2,
                        qbits_key = "mtp_bits",
                        indexer_mode = "full",
                        index_n_heads = config.index_n_heads,
                        index_head_dim = config.index_head_dim,
                        index_topk = config.index_topk,
                        index_kpool = config.index_kpool,
                        index_kpool_tail = config.index_kpool_tail,
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = BlockSparseMLP(
                        config = config,
                        key = f"{key}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.moe_intermediate_size,
                        num_experts = config.num_experts,
                        num_experts_per_tok = config.num_experts_per_tok,
                        key_up = "experts.{expert_idx}.up_proj",
                        key_gate = "experts.{expert_idx}.gate_proj",
                        key_down = "experts.{expert_idx}.down_proj",
                        key_routing_gate = "gate",
                        key_e_score_bias = "gate.e_score_correction_bias",
                        activation_fn = "silu",
                        act_limit = config.swiglu_limit,
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        router_type = "dots",
                        routed_scaling_factor = config.routed_scaling_factor,
                        n_group = config.n_group,
                        topk_group = config.topk_group,
                        qbits_key = "mtp_bits",
                        shared_experts = GatedMLP(
                            config = config,
                            key = f"{key}.mlp.shared_experts",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                            key_up = "up_proj",
                            key_gate = "gate_proj",
                            key_down = "down_proj",
                            activation_fn = "silu",
                            act_limit = config.swiglu_limit,
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            qbits_key = "mtp_bits",
                            select_hq_bits = 2,
                        ) if config.num_shared_experts else None,
                    ),
                )
            )

        self.last_kv_module_idx = len(self.modules) - 1

        # Final norm before the (shared) lm_head
        self.final_norm = RMSNorm(
            config = config,
            key = f"{key_prefix}.layers.{first_mtp_layer + config.num_mtp_layers - 1}.shared_head.norm",
            rms_norm_eps = config.rms_norm_eps,
            out_dtype = torch.half,
        )
        self.modules.append(self.final_norm)

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "default_draft_size": 3,
            "autosplit_load_fwd": False,
        })

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True

        # Cross-references populated by attach_to()
        self.target_embed = None
        self.target_lm_head = None
        self.attached_model = None


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # MTP doesn't take input_ids through Embedding here — embedding is handled by the
        # input layer. prepare_for_attn still wires up flash-attn params
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("MTP draft model does not have its own chat template")


    def attach_to(self, target):
        """
        Bind to target model: borrow embed_tokens / lm_head and tell the target to export its
        hidden state. hnorm consumes the trunk's mean-collapsed hc streams; which point of that
        path is tapped (before or after model.norm) follows EXL3_MTP_PRENORM_H, default OFF.
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
        assert isinstance(target_norm, RMSNorm), \
            "Expected target final RMSNorm immediately before lm_head"

        # EXL3_MTP_PRENORM_H (default OFF): the reference formula (llama.cpp
        # build_glm5next_mtp, glm5next.cpp:841-883) feeds hnorm the mean-collapsed hc streams
        # from BEFORE model.norm -- t_h_nextn is assigned pre-output_norm. exl3 by default taps
        # the post-norm state, i.e. hnorm sees an already-normalised input. When the flag is on,
        # tap the HyperHead (mean collapse) instead: its internal norm key
        # f"{hc_head}.norm" rides the same export_state_norm_keys plumbing.
        tap_key = target_norm.key
        if os.getenv("EXL3_MTP_PRENORM_H", "0") == "1":
            from exllamav3.modules.hyperconnections import HyperHead
            for m in target.modules[target.logit_layer_idx::-1]:
                if isinstance(m, HyperHead) and m.mean:
                    tap_key = m.norm.key
                    break
            else:
                raise RuntimeError("EXL3_MTP_PRENORM_H=1 but no mean HyperHead before lm_head")

        self.draft_verifier_params = {
            "export_state_norm_keys": {tap_key},
        }
        self.load_eh_proj_sidecar()


    def load_eh_proj_sidecar(self, path: str | None = None):
        """
        EXL3_MTP_EH_FP16=<safetensors> (default OFF): replace the quantized eh_proj with the
        unquantized weight from a sidecar file (key "<prefix>.layers.<n>.eh_proj.weight",
        [out, in] = [H, 2H], bf16/fp16, e.g. from the HF checkpoint via
        tools/glm/mtp_eh_sidecar.py). td205 stores eh_proj at mtp_bits = 2; on real MTP inputs
        its output has cos ~0.78 vs the bf16 weight, which costs ~0.1 of draft acceptance
        (acceptaudit2). The fp16 weight is 64 MiB, read once per draft forward.
        """
        path = path if path is not None else os.getenv("EXL3_MTP_EH_FP16", "")
        if not path or path == "0":
            return False
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"EXL3_MTP_EH_FP16={path}: sidecar missing (tools/glm/mtp_eh_sidecar.py)")
        from safetensors import safe_open
        from ..modules.quant.fp16 import LinearFP16
        fc = self.input_layer.fc
        assert fc.inner is not None, "EXL3_MTP_EH_FP16: load the MTP model before attach_to"
        with safe_open(path, "pt") as f:
            keys = [k for k in f.keys() if k.endswith("eh_proj.weight")]
            assert len(keys) == 1, f"EXL3_MTP_EH_FP16: expected one eh_proj.weight in {path}, got {keys}"
            w = f.get_tensor(keys[0])
        assert tuple(w.shape) == (fc.out_features, fc.in_features), \
            f"EXL3_MTP_EH_FP16: eh_proj shape {tuple(w.shape)} != {(fc.out_features, fc.in_features)}"
        w = w.to(fc.device).half().t().contiguous()
        fc.inner = LinearFP16(
            fc.in_features, fc.out_features, w, None,
            fc.in_features, fc.out_features, 0, 0,
            out_dtype = fc.out_dtype, key = fc.key,
        )
        fc.quant_type = "fp16"
        return True


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        ll = self.attached_model().logit_layer_idx
        lm = self.attached_model().modules[ll]
        logits = lm.prepare_for_device(state, params)
        logits = lm.forward(logits, params)
        if params.get("export_draft_conf"):
            # Per-position confidence for the generator's draft truncation: the argmax logit
            # value, over the unpadded vocabulary
            logits = logits[..., :self.attached_model().config.vocab_size]
            conf, ids = torch.max(logits, dim = -1)
            params["draft_conf"] = conf
            return ids
        return torch.argmax(logits, dim = -1)
