"""Standard dense models (llama, qwen3) served straight from a plain GGUF.

No requant: every projection is a ForeignLinear over the raw GGUF payload (Q4_0 ... IQ4_XS, MXFP4, NVFP4).
Config, tensor names and tokenizer all come from the GGUF (loader/gguf_config.py). Build the config directory with
write_config_dir(gguf, dir), then Config.from_directory(dir) as for any other model.

llama.cpp stores llama-arch Q/K rows permuted for interleaved (adjacent-pair) rotary, so llama uses RopeStyle.GPTJ;
qwen3 is stored in HF order (NEOX).
"""
from __future__ import annotations
from typing_extensions import override
import os

import numpy as np
import torch

from ..loader.gguf_collection import GGUFCollection
from ..loader.gguf_config import GGML_FMT, name_map
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP
from ..modules.foreign_linear import ForeignLinear
from ..modules.attn import prepare_for_attn


def _fl(config, eng_key: str, in_features: int, out_features: int, out_dtype=None):
    gname = config.gguf.name_map[eng_key + ".weight"]
    fmt = config.gguf_formats[gname]
    assert fmt != "fp" and not fmt.startswith("unk"), f"{eng_key}: unsupported ggml type {fmt}"
    return ForeignLinear(config, eng_key, in_features, out_features, fmt, out_dtype=out_dtype)


def _mlp(config, L, H, I):
    mlp = GatedMLP(
        config=config, key=f"{L}.mlp", hidden_size=H, intermediate_size=I,
        qmap="block.mlp", interm_dtype=torch.half, out_dtype=torch.float,
        gates=[_fl(config, f"{L}.mlp.gate_proj", H, I)],
        ups=[_fl(config, f"{L}.mlp.up_proj", H, I)],
        downs=[_fl(config, f"{L}.mlp.down_proj", I, H, out_dtype=torch.float)])
    # the prebuilt-projection branch of GatedMLP does not register them for loading
    for m in mlp.gates + mlp.ups + mlp.downs:
        mlp.register_submodule(m)
    return mlp


class GGUFDenseConfig(Config):
    arch_string = "GGUFDenseForCausalLM"

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, {"text": GGUFDenseModel}, **kwargs)
        self.gguf_arch = self.read_cfg(str, "gguf_arch", no_default)
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        style = RopeStyle.GPTJ if self.gguf_arch == "llama" else RopeStyle.NEOX
        self.rope_settings = self.read_rope_settings_default(style)

        self.gguf = GGUFCollection(os.path.join(directory, "model.gguf"))
        self.gguf.name_map = name_map(self.config_dict)
        self.stc = self.gguf
        self.gguf_formats = {g: GGML_FMT.get(ty, f"unk{ty}") for g, (ty, _, _) in self.gguf.infos.items()}
        self.has_qk_norm = "blk.0.attn_q_norm.weight" in self.gguf.infos
        self._embedding_override()

    def _embedding_override(self):
        # token_embd dequantized to fp16 once and served as an in-memory tensor (rows are gathered, never matmul'd)
        gname = "token_embd.weight"
        ty, (K, N), _ = self.gguf.infos[gname]
        fmt = GGML_FMT.get(ty, f"unk{ty}")
        if fmt == "fp":
            return
        fl = ForeignLinear(self, "model.embed_tokens", K, N, fmt)
        # the K-quant / Q8_0 dequant kernels are device code: run them on the GPU, keep the table on the host
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fl.load(dev)
        w = torch.cat([fl._dequant_rows(i, min(4096, N - i)).cpu() for i in range(0, N, 4096)])
        fl.unload()
        self.gguf.new_tensors = {"model.embed_tokens.weight": w}

    def get_tensor_name_fixes(self):
        return {}


class GGUFDenseModel(Model):
    config_class = GGUFDenseConfig

    def __init__(self, config: GGUFDenseConfig, key_prefix: str = "model", **kwargs):
        super().__init__(config, **kwargs)
        H, hd = config.hidden_size, config.head_dim
        nq, nkv, I = config.num_q_heads, config.num_kv_heads, config.intermediate_size
        P = key_prefix
        self.modules += [Embedding(config=config, key=f"{P}.embed_tokens",
                                   vocab_size=config.vocab_size, hidden_size=H)]
        self.first_block_idx = len(self.modules)
        for idx in range(config.num_hidden_layers):
            L = f"{P}.layers.{idx}"
            norms = {}
            if config.has_qk_norm:
                norms = dict(
                    q_norm=RMSNorm(config=config, key=f"{L}.self_attn.q_norm", rms_norm_eps=config.rms_norm_eps),
                    k_norm=RMSNorm(config=config, key=f"{L}.self_attn.k_norm", rms_norm_eps=config.rms_norm_eps))
            self.modules += [TransformerBlock(
                config=config, key=L, layer_idx=idx,
                attn_norm=RMSNorm(config=config, key=f"{L}.input_layernorm", rms_norm_eps=config.rms_norm_eps),
                attn=Attention(
                    config=config, key=f"{L}.self_attn", layer_idx=idx, hidden_size=H, head_dim=hd,
                    num_q_heads=nq, num_kv_heads=nkv, rope_settings=config.rope_settings, sm_scale=None,
                    qmap="block.attn", out_dtype=torch.float,
                    q_proj=_fl(config, f"{L}.self_attn.q_proj", H, nq * hd),
                    k_proj=_fl(config, f"{L}.self_attn.k_proj", H, nkv * hd),
                    v_proj=_fl(config, f"{L}.self_attn.v_proj", H, nkv * hd),
                    o_proj=_fl(config, f"{L}.self_attn.o_proj", nq * hd, H, out_dtype=torch.float),
                    **norms),
                mlp_norm=RMSNorm(config=config, key=f"{L}.post_attention_layernorm", rms_norm_eps=config.rms_norm_eps),
                mlp=_mlp(config, L, H, I))]
        self.last_kv_module_idx = len(self.modules) - 1
        self.modules += [RMSNorm(config=config, key=f"{P}.norm", rms_norm_eps=config.rms_norm_eps,
                                 out_dtype=torch.half)]
        if config.tie_word_embeddings:
            # tied head: the quantized embedding payload doubles as the output projection
            config.gguf.name_map["lm_head.weight"] = "token_embd.weight"
        self.modules += [_fl(config, "lm_head", H, config.vocab_size)]
        self.modules[-1].caps["logits_output"] = True
        self.logit_layer_idx = len(self.modules) - 1

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        p += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        return p
