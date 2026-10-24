"""MiniCPM-V-4.7-35B-A3B text path on GGUF as shipped.

Same block family as Qwen3_5Moe (3x GDN linear + 1x full attention per 4,
256-expert SwiGLU MoE + shared expert): reuses GatedDeltaNet, Attention,
BlockSparseMLP and GatedMLP through ForeignLinear prebuilts over the raw
GGUF payloads (no requant). Text-only; vision tower is not covered.

GGUF tensor names (blk.{i}.*) are served by GGUFCollection; engine-style
module keys map through collection.name_map, expert slices through the
".e{idx}" protocol natively.
"""
from __future__ import annotations
from typing_extensions import override
import os

import numpy as np
import torch

from ..loader.gguf_collection import GGUFCollection
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle, RoPE
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    BlockSparseMLP,
    Linear,
    GatedDeltaNet,
    GatedMLP,
    GDNState,
)
from ..modules.foreign_linear import ForeignLinear
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence

MCPM_ARCH = "MiniCPMV4_7ForConditionalGeneration"
MCPM_GGUF_ENV = "MCPM_GGUF"
MCPM_FOREIGN = os.environ.get("MCPM_FOREIGN", "1") != "0"

_GGML_FMT = {12: "q4_k", 13: "q5_k", 14: "q6_k", 23: "iq4_xs", 0: "fp", 1: "fp"}

P = "model.language_model"


def _gguf_path() -> str:
    p = os.environ.get(MCPM_GGUF_ENV)
    assert p, f"{MCPM_GGUF_ENV} must point at the MiniCPM GGUF file"
    return p


def _map_layer(nm: dict, i: int):
    L = f"{P}.layers.{i}"
    G = f"blk.{i}"
    nm.update({
        f"{L}.input_layernorm.weight": f"{G}.attn_norm.weight",
        f"{L}.post_attention_layernorm.weight": f"{G}.post_attention_norm.weight",
        # GDN projs (ForeignLinear keys, no map needed for the module key
        # itself: the map covers the ".weight" tensor key)
        f"{L}.linear_attn.in_proj_qkv.weight": f"{G}.attn_qkv.weight",
        f"{L}.linear_attn.in_proj_z.weight": f"{G}.attn_gate.weight",
        f"{L}.linear_attn.in_proj_b.weight": f"{G}.ssm_beta.weight",
        f"{L}.linear_attn.in_proj_a.weight": f"{G}.ssm_alpha.weight",
        f"{L}.linear_attn.out_proj.weight": f"{G}.ssm_out.weight",
        f"{L}.linear_attn.conv1d.weight": f"{G}.ssm_conv1d.weight",
        f"{L}.linear_attn.A_log": f"{G}.ssm_a",
        f"{L}.linear_attn.dt_bias": f"{G}.ssm_dt.bias",
        f"{L}.linear_attn.norm.weight": f"{G}.ssm_norm.weight",
        # Full attention
        f"{L}.self_attn.q_proj.weight": f"{G}.attn_q.weight",
        f"{L}.self_attn.k_proj.weight": f"{G}.attn_k.weight",
        f"{L}.self_attn.v_proj.weight": f"{G}.attn_v.weight",
        f"{L}.self_attn.o_proj.weight": f"{G}.attn_output.weight",
        f"{L}.self_attn.q_norm.weight": f"{G}.attn_q_norm.weight",
        f"{L}.self_attn.k_norm.weight": f"{G}.attn_k_norm.weight",
        # MoE router + shared gate
        f"{L}.mlp.gate.weight": f"{G}.ffn_gate_inp.weight",
        f"{L}.mlp.shared_expert_gate.weight": f"{G}.ffn_gate_inp_shexp.weight",
    })


class MiniCPMV47Config(Config):
    arch_string = MCPM_ARCH

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, {"text": MiniCPMV47TextModel}, **kwargs)
        tc = "text_config"

        def pfx(key):
            return f"{tc}->{key}"

        self.head_dim = self.read_cfg(int, pfx("head_dim"), None)
        self.hidden_size = self.read_cfg(int, pfx("hidden_size"), no_default)
        self.num_q_heads = self.read_cfg(int, pfx("num_attention_heads"), no_default)
        self.num_kv_heads = self.read_cfg(int, pfx("num_key_value_heads"), self.num_q_heads)
        self.full_attention_interval = self.read_cfg(int, pfx("full_attention_interval"), 4)
        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads
        self.linear_conv_kernel_dim = self.read_cfg(int, pfx("linear_conv_kernel_dim"), 4)
        self.linear_num_key_heads = self.read_cfg(int, pfx("linear_num_key_heads"), 16)
        self.linear_num_value_heads = self.read_cfg(int, pfx("linear_num_value_heads"), 32)
        self.linear_key_head_dim = self.read_cfg(int, pfx("linear_key_head_dim"), 128)
        self.linear_value_head_dim = self.read_cfg(int, pfx("linear_value_head_dim"), 128)
        self.assert_cfg(str, pfx("hidden_act"), "silu", True)
        self.moe_intermediate_size = self.read_cfg(int, pfx("moe_intermediate_size"), no_default)
        self.num_experts = self.read_cfg(int, pfx("num_experts"), no_default)
        self.num_experts_per_tok = self.read_cfg(int, pfx("num_experts_per_tok"), no_default)
        self.shared_expert_intermediate_size = self.read_cfg(
            int, pfx("shared_expert_intermediate_size"), 512)
        self.rms_norm_eps = self.read_cfg(float, pfx("rms_norm_eps"), no_default)
        self.num_hidden_layers = self.read_cfg(int, pfx("num_hidden_layers"), no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        self.vocab_size = self.read_cfg(int, pfx("vocab_size"), no_default)
        self.max_position_embeddings = self.read_cfg(
            int, pfx("max_position_embeddings"), 262144)
        # layer_types from config when present (MiniCPM ships it)
        lt = self.read_cfg(list, pfx("layer_types"), None)
        if lt is not None:
            assert len(lt) == self.num_hidden_layers
            self.layer_types = [
                "full_attention" if t == "full_attention" else "linear_attention" for t in lt]
        else:
            self.layer_types = [
                "full_attention" if (idx + 1) % self.full_attention_interval == 0
                else "linear_attention" for idx in range(self.num_hidden_layers)]
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX, default_rope_theta=10000000,
            config_dict=self.read_cfg(dict, tc, no_default))
        # MTP head and vision tower are separate concerns (draft file)
        self.model_classes.pop("mtp", None)
        self.model_classes.pop("vision", None)

        # GGUF backing store + engine->gguf name map
        self.gguf = GGUFCollection(_gguf_path())
        nm: dict[str, str] = {}
        for i in range(self.num_hidden_layers):
            _map_layer(nm, i)
        nm[f"{P}.norm.weight"] = "output_norm.weight"
        nm["lm_head.weight"] = "output.weight"
        self.gguf.name_map = nm
        self.stc = self.gguf
        self.gguf_formats = {
            g: _GGML_FMT.get(ty, f"unk{ty}")
            for g, (ty, _, _) in self.gguf.infos.items()}
        # Embedding: dequant IQ4_XS token_embd to fp16 once (1 GB), served
        # as an in-memory override (vectorized numpy, no per-block Python).
        self._load_embedding_override()

    def _load_embedding_override(self):
        # token_embd dequantized to fp16 once (~1 GB) and served as an
        # in-memory override under the engine key (vectorized numpy, no
        # per-block Python). Format-generic (IQ4_XS / Q4_K).
        from ..foreign_quant.k_quants import dequant_iq4_xs_bulk, dequant_q4_K_bulk
        ty, dims, off = self.gguf.infos["token_embd.weight"]
        bbs = {23: 136, 12: 144}
        assert ty in bbs, f"token_embd type {ty}"
        bb = bbs[ty]
        K, N = dims[0], dims[1]
        assert K % 256 == 0
        raw = self.gguf._pread(off, N * (K // 256) * bb).reshape(N, (K // 256) * bb)
        nb = N * (K // 256)
        bulk = dequant_iq4_xs_bulk if ty == 23 else dequant_q4_K_bulk
        w = bulk(raw.reshape(nb, bb)).reshape(N, K)
        self.gguf.new_tensors = {
            f"{P}.embed_tokens.weight": torch.from_numpy(np.ascontiguousarray(w))}

    def get_tensor_name_fixes(self):
        return {}


def _fp_override(config, eng_weight_key: str, gguf_name: str):
    """Dequantize a small GGUF tensor on CPU and register it as an fp32
    in-memory override (Linear fp16 path picks it up with float2half).
    For the tiny GDN b/a projs the fused kernels require fp16."""
    from ..foreign_quant.k_quants import dequant_row
    ty, dims, off = config.gguf.infos[gguf_name]
    bbs = {12: 144, 13: 176, 14: 210, 23: 136}
    fmts = {12: "q4_k", 13: "q5_k", 14: "q6_k", 23: "iq4_xs"}
    assert ty in bbs, f"{gguf_name}: type {ty}"
    bb, fmt = bbs[ty], fmts[ty]
    K, N = dims[0], dims[1]
    raw = config.gguf._pread(off, N * (K // 256) * bb)
    blocks = [bytes(raw[i * bb:(i + 1) * bb]) for i in range(N * (K // 256))]
    # dequant_row concatenates in GGUF row order ([N, K]); engine fp Linears
    # keep (in, out) orientation, so transpose to [K, N] = [in, out].
    w = dequant_row(blocks, fmt).reshape(N, K).T
    if config.gguf.new_tensors is None:
        config.gguf.new_tensors = {}
    config.gguf.new_tensors[eng_weight_key] = torch.from_numpy(np.ascontiguousarray(w))


def _alog_override(config, eng_weight_key: str, gguf_name: str):
    """A_log override: the GGUF ssm_a tensor stores -exp(A_log) (llama.cpp
    qwen35moe: gate = softplus(alpha + dt) * ssm_a), while the engine's GDN
    (torch path and fused_op_2 alike) computes g = -exp(a_log) * softplus.
    Register log(-ssm_a) under the engine A_log key."""
    ty, dims, off = config.gguf.infos[gguf_name]
    assert ty == 0 and len(dims) == 1, f"{gguf_name}: type {ty} dims {dims}"
    raw = config.gguf._pread(off, dims[0] * 4)
    a = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    assert np.all(a < 0), f"{gguf_name}: expected all-negative -exp(A_log)"
    w = np.log(-a).astype(np.float32)
    if config.gguf.new_tensors is None:
        config.gguf.new_tensors = {}
    config.gguf.new_tensors[eng_weight_key] = torch.from_numpy(np.ascontiguousarray(w))


def _fl(config, eng_key: str, in_features: int, out_features: int,
        out_dtype=None):
    # Resolve engine key -> gguf name, then the base tensor's format
    # (expert-slice keys "....exps.e7" look up the base tensor).
    gkey = config.gguf.name_map.get(eng_key + ".weight", eng_key + ".weight")
    body = gkey[:-7] if gkey.endswith(".weight") else gkey
    if "exps.e" in body:
        body = body.rsplit(".e", 1)[0] + ".weight"
    elif not body.endswith(".weight"):
        body = body + ".weight"
    fmt = config.gguf_formats[body]
    assert fmt not in ("fp",) and not fmt.startswith("unk"), f"{eng_key}: {fmt}"
    return ForeignLinear(config, eng_key, in_features, out_features, fmt,
                         out_dtype=out_dtype)


class MiniCPMV47TextModel(Model):
    config_class = MiniCPMV47Config

    def __init__(self, config: MiniCPMV47Config, **kwargs):
        super().__init__(config, **kwargs)
        H = config.hidden_size
        self.modules += [
            Embedding(config=config, key=f"{P}.embed_tokens",
                      vocab_size=config.vocab_size, hidden_size=H)]
        self.first_block_idx = len(self.modules)
        for idx in range(config.num_hidden_layers):
            L = f"{P}.layers.{idx}"
            G = f"blk.{idx}"
            is_full = config.layer_types[idx] == "full_attention"
            if is_full:
                attn = Attention(
                    config=config, key=f"{L}.self_attn", layer_idx=idx,
                    hidden_size=H, head_dim=config.head_dim,
                    num_q_heads=config.num_q_heads, num_kv_heads=config.num_kv_heads,
                    rope_settings=config.rope_settings, sm_scale=None,
                    q_norm=RMSNorm(config=config, key=f"{L}.self_attn.q_norm",
                                   rms_norm_eps=config.rms_norm_eps),
                    k_norm=RMSNorm(config=config, key=f"{L}.self_attn.k_norm",
                                   rms_norm_eps=config.rms_norm_eps),
                    out_dtype=torch.float, interleaved_gate=True,
                    select_hq_bits=2,
                    **self._attn_projs(config, L, G, H))
            else:
                attn = GatedDeltaNet(
                    config=config, key=f"{L}.linear_attn", layer_idx=idx,
                    hidden_size=H, k_head_dim=config.linear_key_head_dim,
                    v_head_dim=config.linear_value_head_dim,
                    num_k_heads=config.linear_num_key_heads,
                    num_v_heads=config.linear_num_value_heads,
                    rms_norm_eps=config.rms_norm_eps,
                    conv_kernel_size=config.linear_conv_kernel_dim,
                    key_a_log="A_log", key_dt_bias="dt_bias", key_conv1d="conv1d",
                    key_qkv="in_proj_qkv", key_z="in_proj_z",
                    key_b="in_proj_b", key_a="in_proj_a",
                    key_norm="norm", key_o="out_proj", qmap="block.attn",
                    out_dtype=torch.float, select_hq_bits=2,
                    # GGUF v-heads are even/odd grouped (llama blocked style);
                    # repeat pairing mis-pairs 30/32 heads (see NOTEBOOK).
                    blocked_gqa=True,
                    **self._gdn_projs(config, L, G, H, idx))
            mlp = self._moe(config, L, G, H)
            self.modules += [
                TransformerBlock(
                    config=config, key=L, layer_idx=idx,
                    attn_norm=RMSNorm(config=config, key=f"{L}.input_layernorm",
                                      rms_norm_eps=config.rms_norm_eps),
                    attn=attn,
                    mlp_norm=RMSNorm(config=config, key=f"{L}.post_attention_layernorm",
                                     rms_norm_eps=config.rms_norm_eps),
                    mlp=mlp)]
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += [
            RMSNorm(config=config, key=f"{P}.norm",
                    rms_norm_eps=config.rms_norm_eps,
                    out_dtype=torch.half),
            _fl(config, "output", H, config.vocab_size)]
        self.modules[-1].caps["logits_output"] = True  # whole-forward CUDA graph tail marker
        self.logit_layer_idx = len(self.modules) - 1
        self.calibration_all_experts = True
        self.caps.update({
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
            "linear_attn": True})
        self.recurrent_state_cls = GDNState

    @staticmethod
    def _attn_projs(config, L, G, H):
        if not MCPM_FOREIGN:
            return {}
        hd, nq, nkv = config.head_dim, config.num_q_heads, config.num_kv_heads
        return dict(
            q_proj=_fl(config, f"{L}.self_attn.q_proj", H, 2 * nq * hd),
            k_proj=_fl(config, f"{L}.self_attn.k_proj", H, nkv * hd),
            v_proj=_fl(config, f"{L}.self_attn.v_proj", H, nkv * hd),
            o_proj=_fl(config, f"{L}.self_attn.o_proj", nq * hd, H,
                        out_dtype=torch.float))

    @staticmethod
    def _gdn_projs(config, L, G, H, idx):
        if not MCPM_FOREIGN:
            return {}
        kd = config.linear_key_head_dim * config.linear_num_key_heads
        vd = config.linear_value_head_dim * config.linear_num_value_heads
        nb = config.linear_num_value_heads
        # b/a are tiny ([2048, 32]) and the fused GDN kernels require fp16:
        # dequantize once on CPU into in-memory overrides, plain Linears.
        # b = beta (ssm_beta), a = alpha (ssm_alpha): swapped once, never again.
        # MCPM_SWAP_BA=1 restores the original (alpha/beta) order for A/B.
        if os.environ.get("MCPM_SWAP_BA", "0") == "1":
            _fp_override(config, f"{L}.linear_attn.in_proj_b.weight", f"blk.{idx}.ssm_alpha.weight")
            _fp_override(config, f"{L}.linear_attn.in_proj_a.weight", f"blk.{idx}.ssm_beta.weight")
        else:
            _fp_override(config, f"{L}.linear_attn.in_proj_b.weight", f"blk.{idx}.ssm_beta.weight")
            _fp_override(config, f"{L}.linear_attn.in_proj_a.weight", f"blk.{idx}.ssm_alpha.weight")
        # ssm_a stores -exp(A_log); the engine consumes A_log.
        _alog_override(config, f"{L}.linear_attn.A_log", f"blk.{idx}.ssm_a")
        return dict(
            qkv_proj=_fl(config, f"{L}.linear_attn.in_proj_qkv", H, 2 * kd + vd,
                          out_dtype=torch.float),
            z_proj=_fl(config, f"{L}.linear_attn.in_proj_z", H, vd,
                        out_dtype=torch.float),
            b_proj=Linear(config, f"{L}.linear_attn.in_proj_b", H, nb,
                          out_dtype=torch.float, pad_to=1),
            a_proj=Linear(config, f"{L}.linear_attn.in_proj_a", H, nb,
                          out_dtype=torch.float, pad_to=1),
            o_proj=_fl(config, f"{L}.linear_attn.out_proj", vd, H,
                        out_dtype=torch.float))

    @staticmethod
    def _moe(config, L, G, H):
        I = config.moe_intermediate_size
        E = config.num_experts
        if MCPM_FOREIGN:
            gates = [_fl(config, f"{G}.ffn_gate_exps.e{e}", H, I) for e in range(E)]
            ups = [_fl(config, f"{G}.ffn_up_exps.e{e}", H, I) for e in range(E)]
            downs = [_fl(config, f"{G}.ffn_down_exps.e{e}", I, H,
                         out_dtype=torch.float) for e in range(E)]
            SI = config.shared_expert_intermediate_size
            sh = GatedMLP(
                config=config, key=f"{L}.mlp.shared_expert",
                hidden_size=H, intermediate_size=SI,
                qmap="block.mlp", interm_dtype=torch.half, out_dtype=torch.float,
                select_hq_bits=2,
                gates=[_fl(config, f"{G}.ffn_gate_shexp", H, SI)],
                ups=[_fl(config, f"{G}.ffn_up_shexp", H, SI)],
                downs=[_fl(config, f"{G}.ffn_down_shexp", SI, H,
                           out_dtype=torch.float)])
            mlp = BlockSparseMLP(
                config=config, key=f"{L}.mlp", hidden_size=H,
                intermediate_size=I, num_experts=E,
                num_experts_per_tok=config.num_experts_per_tok,
                key_up=None, key_gate=None, key_down=None,
                key_routing_gate="gate", key_shared_gate="shared_expert_gate",
                qmap="block.mlp", interm_dtype=torch.half, out_dtype=torch.float,
                gates=gates, ups=ups, downs=downs, shared_experts=sh)
            # Prebuilt lists bypass the internal builders, which are the only
            # register_submodule call sites (Bolt precedent): register here so
            # Module.load() recurses into them.
            for m in gates + ups + downs + sh.gates + sh.ups + sh.downs:
                mlp.register_submodule(m)
        else:
            mlp = BlockSparseMLP(
                config=config, key=f"{L}.mlp", hidden_size=H,
                intermediate_size=I, num_experts=E,
                num_experts_per_tok=config.num_experts_per_tok,
                key_up="experts.{expert_idx}.up_proj",
                key_gate="experts.{expert_idx}.gate_proj",
                key_down="experts.{expert_idx}.down_proj",
                key_routing_gate="gate", key_shared_gate="shared_expert_gate",
                qmap="block.mlp", interm_dtype=torch.half, out_dtype=torch.float)
        return mlp

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += "<|im_start|>system\n" + f"{system_prompt}<|im_end|>\n"
        p += "<|im_start|>user\n" + f"{prompt}<|im_end|>\n" + "<|im_start|>assistant\n"
        return p
