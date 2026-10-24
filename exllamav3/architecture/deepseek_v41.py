from __future__ import annotations
from typing_extensions import override
import glob
import os
import torch
from ..model.config import Config
from ..model.model import Model
from ..modules import Embedding, Linear
from ..modules.hyperconnections import ExpandStreams
from ..modules.dsv41 import DSV41Block, DSV41Raw, DSV41Final, DSV41Head
from ..dsv41.config import DSV41Config

# DeepSeek-V4.1-Flash: sliding / compressed (ratio 2 and 1) attention with shared compressed-KV pools and indexers,
# two-level (candidate block) selection, mHC residual streams with a delayed pre-mix, Engram n-gram memory at layers
# 1 and 14, 384 routed experts (top 6) + shared expert, dspark MTP stages. Tensor names are the official ones
# (layers.N.attn.wq_a, layers.N.ffn.experts.E.w1, mtp.J...., embed / head / norm).
#
# Registered for the EXL3 conversion: Config + Model so `convert.py` can walk the layers, and a calibration forward
# per block (modules/dsv41.py, exllamav3/dsv41/calib.py). The serving graph of the port is exllamav3.dsv41.model.
# Not handled: the vision tower (vision.*, aligner.*), copied by the converter front-end.


class DeepseekV41Config(Config):
    arch_string = "DeepseekV41ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": DeepseekV41Model, "mtp": DeepseekV41MTPModel},
            **kwargs
        )
        d = self.dsv41 = DSV41Config.from_hf_dict(self.config_dict)
        self.hidden_size = d.hidden_size
        self.num_hidden_layers = d.num_hidden_layers
        self.num_q_heads = d.num_attention_heads
        self.num_kv_heads = 1
        self.head_dim = d.head_dim
        self.tie_word_embeddings = self.read_cfg(bool, ["tie_word_embeddings", "text_config->tie_word_embeddings"],
                                                 False)
        # MoE calibration mode: "all" = every expert sees every token (exllamav3 default, one shared Hessian for
        # gate/up), "routed" = experts see only the tokens routed to them (a Hessian per expert; memory heavy)
        self.dsv41_moe_mode = kwargs.get("dsv41_moe_mode") or os.environ.get("EXL3_DSV41_MOE_MODE", "all")
        assert self.dsv41_moe_mode in ("all", "routed")
        self.dsv41_n_mtp = d.n_mtp_layers
        # the stages are present as `.weight` (source checkpoint, synthetic fp16 pack) or as EXL3 tensors (converted pack)
        if self.dsv41_n_mtp == 0 or not any(self.stc.has_tensor(f"mtp.0.attn.wkv.{s}") for s in ("weight", "trellis")):
            del self.model_classes["mtp"]
            self.dsv41_n_mtp = 0
        self._eg = None
        self._eg_store = kwargs.get("dsv41_engram_store")

    # ---- engram runtime (hash layout, token map, rows)
    def engram_runtime(self) -> dict:
        if self._eg is None:
            from tokenizers import Tokenizer as HFTokenizer
            from ..dsv41.engram import build_compressed_token_map, EngramLayout, compute_multipliers
            d = self.dsv41
            tk = HFTokenizer.from_file(os.path.join(self.directory, "tokenizer.json"))

            class _Tok:
                backend_tokenizer = tk

                def __len__(self):
                    return tk.get_vocab_size(with_added_tokens = True)
            token_map, nv = build_compressed_token_map(_Tok())
            if d.engram_compressed_vocab_size and nv != d.engram_compressed_vocab_size:
                raise ValueError(f"engram compressed vocab {nv} != config {d.engram_compressed_vocab_size}")
            layout = EngramLayout.build(d.engram_layer_ids, d.engram_num_embeddings, d.engram_max_ngram_size,
                                        d.engram_vocab_size, d.engram_n_heads, d.engram_head_dim)
            self._eg = {
                "layout": layout, "token_map": token_map, "compressed_vocab": nv,
                "multipliers": compute_multipliers(layout.layer_ids, layout.max_ngram_size, nv),
            }
        return self._eg

    def engram_store(self):
        """Row store of the engram tables. Order: a store passed as `dsv41_engram_store=` to Config.from_directory
        (in-memory tables, remote ranges); the Engram directory of the pack (`dsv41.packspec.resolve_engram_dir`:
        $EXL3_DSV41_ENGRAM_DIR, `quantization_config.engram.dir`, `<pack>/engram`) with plain files
        `layers.N.engram.embed.{weight,scale}.bin` (MXFP4 unless the config or $EXL3_DSV41_ENGRAM_FMT says fp8); else the
        tables inside the checkpoint shards of the model directory (e4m3 rows, or MXFP4 U8 [N, 128] tables).
        All reads are O_DIRECT row gathers (`dsv41.engram.make_reader`: native, parallel or serial), never mmap."""
        if self._eg_store is None:
            from ..dsv41 import packspec as PS
            qcfg = self.config_dict.get("quantization_config") or {}
            edir = PS.resolve_engram_dir(self.directory, qcfg)
            if edir and (os.path.exists(os.path.join(edir, PS.engram_files(self.dsv41.engram_layer_ids[0])[0]))
                         or glob.glob(os.path.join(edir, "*.safetensors"))):
                from ..dsv41.engram import RawBinEngramStore
                fmt = os.environ.get("EXL3_DSV41_ENGRAM_FMT") or PS.engram_cfg(qcfg).get("format") or "auto"
                self._eg_store = RawBinEngramStore.open(edir, self.dsv41.engram_layer_ids, self.dsv41.engram_head_dim,
                                                        fmt, allow_buffered = True)
        if self._eg_store is None:
            from ..dsv41.safetensors_io import SafetensorsWeightSource
            src = SafetensorsWeightSource(self.directory, self.dsv41.num_hidden_layers,
                                          self.dsv41.engram_layer_ids, allow_buffered_engram = True)
            self._eg_store = src.engram_store()
        return self._eg_store


class DeepseekV41Model(Model):
    config_class = DeepseekV41Config

    def __init__(
        self,
        config: DeepseekV41Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        d = config.dsv41

        self.modules += [
            Embedding(config = config, key = "embed", vocab_size = d.vocab_size, hidden_size = d.hidden_size),
            ExpandStreams(config = config, key = "hc_expand", hc_mult = d.hc_mult),
        ]
        self.first_block_idx = len(self.modules)
        for idx in range(d.num_hidden_layers):
            self.modules.append(DSV41Block(config, f"layers.{idx}", idx, d, moe_mode = config.dsv41_moe_mode))
        self.last_kv_module_idx = len(self.modules) - 1

        alt = "embed" if config.tie_word_embeddings and not config.stc.has_tensor("head.weight") else None
        self.modules += [
            DSV41Final(config, "norm", d),
            DSV41Head(config, "head", d, alt_key = alt),
        ]
        self.logit_layer_idx = len(self.modules) - 1
        # every expert sees the calibration tokens in "all" mode (see DeepseekV41Config.dsv41_moe_mode)
        self.calibration_all_experts = config.dsv41_moe_mode == "all"
        # per-row cross-layer state lives in quant_preserve, which is not checkpointed
        self.caps.update({"can_resume_quant": False, "supports_tp": False})

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        params["input_ids"] = input_ids
        return input_ids

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = f"{system_prompt}\n\n" if system_prompt else ""
        return p + f"<|User|>{prompt}<|Assistant|>"


class DeepseekV41MTPModel(Model):
    """MTP (dspark) stages mtp.0 .. mtp.{n-1}: full blocks with sliding attention and dspark_n_routed_experts
    experts, plus main_proj / main_norm (stage 0) and norm / markov head / confidence head (last stage)."""

    def __init__(
        self,
        config: DeepseekV41Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        d = config.dsv41
        n = d.num_hidden_layers
        self.first_block_idx = 0
        for j in range(config.dsv41_n_mtp):
            self.modules.append(
                DSV41Block(config, f"mtp.{j}", n + j, d, mtp = True, moe_mode = config.dsv41_moe_mode,
                           qbits_key = "mtp_bits")
            )
        last = f"mtp.{config.dsv41_n_mtp - 1}"
        extra = DSV41Raw(config, "mtp.extra",
                         ["mtp.0.main_norm.weight", f"{last}.norm.weight", f"{last}.markov_head.embed.weight",
                          f"{last}.markov_head.head.weight", f"{last}.confidence_head.proj.weight"])
        # main_proj: bf16-sized projection of the three target taps into the draft width
        self.main_proj = Linear(config, "mtp.0.main_proj", d.hidden_size * len(d.dspark_target_layer_ids),
                                d.hidden_size, qmap = "mtp.main_proj", qbits_key = "mtp_bits", select_hq_bits = 2,
                                out_dtype = torch.half, trim_padded_out = True)
        extra.register_submodule(self.main_proj)
        self.modules.append(extra)
        self.caps.update({"can_resume_quant": False, "supports_tp": False})
