"""
Tiny synthetic DS4.1 pack for tests and smoke runs: random weights, the converter's file layout and tensor names, written
without any quantizer (linears are plain fp16 `.weight` tensors, which the framework loads as `LinearFP16`; with
`exl3=True` the linears are quantized by `Linear.convert_exl3`-style tools elsewhere, see tools/ds41model/gpu_smoke.py).

Layout (same as tools/ds41/stream_convert.py): one shard per unit
    global.safetensors            embed, norm, head
    layers.N.safetensors          one trunk layer (attention, hc params, router, shared + routed experts, engram wkv)
    engram/layers.N.engram.embed.{weight,scale}.bin   Engram tables (MXFP4 rows: 128 + 8 bytes), read with O_DIRECT
Every tensor name, shape and unit comes from `packspec` (the module the converter and the validator share).
    mtp.J.safetensors             dspark stage J (+ main_proj / markov / confidence heads on the first / last stage)
plus config.json (HF `text_config` + `quantization_config`), tokenizer.json, model.safetensors.index.json.
wo_a is stored as the converter writes it: 8 slices `attn.wo_a.slice.G.weight` [o_lora_rank, H*hd/G].
"""
from __future__ import annotations
import json
import os
import re
import numpy as np
import torch
from safetensors.torch import save_file
from .config import DSV41Config
from .engram import build_compressed_token_map, EngramLayout
from . import packspec as PS


def make_tokenizer(vocab_size: int):
    """WordLevel tokenizer whose normalisation merges `w<i>` / `W<i>` (so the compressed token map is non trivial)."""
    from tokenizers import Tokenizer, models
    half = vocab_size // 2
    vocab = {}
    for i in range(half):
        vocab[f"w{i}"] = len(vocab)
        vocab[f"W{i}"] = len(vocab)
    while len(vocab) < vocab_size:
        vocab[f"<p{len(vocab)}>"] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab, unk_token="w0"))

    class _Tok:
        backend_tokenizer = tk

        def __len__(self):
            return vocab_size
    return tk, _Tok()


def _raw_value(name: str, shape, g, d: int, dtype_big):
    """Random value of one raw tensor of the spec (distribution by role; dtype as the release stores it)."""
    R = lambda *sh, sc=1.0: torch.randn(*sh, generator=g) * sc
    bf = torch.bfloat16
    if name in ("embed.weight",):
        return R(*shape).to(dtype_big)
    if re.search(r"\.hc_(attn|ffn)_fn$", name):
        return R(*shape, sc=0.3 * shape[1] ** -0.5)
    if re.search(r"\.hc_(attn|ffn)_base$", name):
        return R(*shape, sc=0.5)
    if re.search(r"\.hc_(attn|ffn)_scale$", name):
        return torch.tensor([0.8, 0.6, 0.7]) + 0.1 * R(3)
    if name.endswith("norm.weight"):
        return (1 + 0.1 * R(*shape)).to(torch.float32)
    if name.endswith(".attn_sink"):
        return R(*shape, sc=0.5)
    if name.endswith(".ffn.gate.bias") or name.endswith(".ffn.gate.bias_vl"):
        return 0.2 * R(*shape)
    if name.endswith("_weight"):                                   # engram q / k
        return 1 + 0.2 * R(*shape)
    if name.endswith("markov_head.embed.weight"):
        return R(*shape, sc=0.5).to(bf)
    if name.endswith("markov_head.head.weight"):
        return R(*shape, sc=0.3).to(bf)
    if name.endswith("confidence_head.proj.weight"):
        return R(*shape, sc=0.1).to(bf)
    return R(*shape, sc=shape[-1] ** -0.5).to(bf)                  # router gate, compressor, indexer projections


def random_state_dict(cfg: DSV41Config, seed: int = 0, dtype=torch.float16) -> dict:
    """Tensors of the tensor-name spec (`packspec`): fp16 plain linears, fp32 norms / hc / router bias, bf16 small
    projections. Sized by `cfg`; the Engram tables are made by `random_engram_tables`."""
    g = torch.Generator().manual_seed(seed)
    sd = {}
    d = cfg.hidden_size
    for lin in PS.linears(cfg, mtp=True):
        sc = d ** -0.5 * 2 if lin.key == "head" else lin.in_f ** -0.5
        sd[lin.key + ".weight"] = (torch.randn(lin.out_f, lin.in_f, generator=g) * sc).to(dtype)
    for raw in PS.raws(cfg, mtp=True):
        if raw.optional:
            continue
        sd[raw.name] = _raw_value(raw.name, raw.shape, g, d, dtype)
    return sd


def random_engram_tables(cfg: DSV41Config, num_rows, seed: int = 1):
    """MXFP4 release layout per layer: U8 [N, head_dim/2] (two e2m1 per byte) + e8m0 [N, head_dim/32]."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for n in num_rows:
        v = torch.randint(0, 256, (n, cfg.engram_head_dim // 2), generator=g, dtype=torch.uint8)
        s = torch.randint(121, 126, (n, cfg.engram_head_dim // 32), generator=g, dtype=torch.uint8)
        out.append((v, s.view(torch.float8_e8m0fnu)))
    return out


def write_engram_dir(path: str, tables, layer_ids) -> dict:
    """Engram tables as plain files in `path` (`layers.N.engram.embed.{weight,scale}.bin`, MXFP4 rows of 128 + 8 bytes):
    what the O_DIRECT row reader opens. Returns the `quantization_config.engram` entry."""
    os.makedirs(path, exist_ok=True)
    rows = []
    for (v, sc), lid in zip(tables, layer_ids):
        wf, sf = PS.engram_files(lid)
        v.contiguous().view(torch.uint8).numpy().tofile(os.path.join(path, wf))
        sc.contiguous().view(torch.uint8).numpy().tofile(os.path.join(path, sf))
        rows.append(int(v.shape[0]))
    return rows


def write_pack(path: str, cfg: DSV41Config, seed: int = 0, expert_bits: dict = None, extra_qcfg: dict = None,
               engram: str = "dir"):
    """Write a synthetic fp16 pack for `cfg` into `path`; returns (cfg with engram sizes filled in, state dict, tables).
    engram="dir": the tables are files in `<path>/engram` (pack layout); "shards": inside `layers.N.engram.safetensors`
    (the older layout, still loadable)."""
    os.makedirs(path, exist_ok=True)
    tk, shim = make_tokenizer(cfg.vocab_size)
    tmap, nv = build_compressed_token_map(shim)
    cfg.engram_compressed_vocab_size = nv
    lay = EngramLayout.build(cfg.engram_layer_ids, None, cfg.engram_max_ngram_size, cfg.engram_vocab_size,
                             cfg.engram_n_heads, cfg.engram_head_dim)
    cfg.engram_num_embeddings = list(lay.num_embeddings)
    tables = random_engram_tables(cfg, lay.num_embeddings, seed + 1)
    sd = random_state_dict(cfg, seed)
    units = {}
    for k, v in sd.items():
        units.setdefault(PS.unit_of(k), {})[k] = v.contiguous()
    eg_cfg = None
    if engram == "shards":
        for i, lid in enumerate(cfg.engram_layer_ids):
            units[f"layers.{lid}.engram"] = {f"layers.{lid}.engram.embed.weight": tables[i][0].contiguous(),
                                             f"layers.{lid}.engram.embed.scale": tables[i][1].contiguous()}
    elif cfg.engram_layer_ids:
        rows = write_engram_dir(os.path.join(path, PS.ENGRAM_DIR), tables, cfg.engram_layer_ids)
        eg_cfg = {"dir": PS.ENGRAM_DIR, "format": PS.ENGRAM_FMT, "layers": list(cfg.engram_layer_ids), "rows": rows,
                  "head_dim": cfg.engram_head_dim}
    weight_map, total = {}, 0
    for u, t in units.items():
        f = PS.shard_file(u)
        save_file(t, os.path.join(path, f), metadata={"format": "pt"})
        total += os.path.getsize(os.path.join(path, f))
        for k in t:
            weight_map[k] = f
    PS.write_index(os.path.join(path, "model.safetensors.index.json"), weight_map, total)
    qcfg = PS.build_qcfg(bits=16, head_bits=16, mtp_bits=16, expert_bits=expert_bits, engram=eg_cfg,
                         extra={"synthetic": True, **(extra_qcfg or {})})
    hf = {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "dtype": "bfloat16",
          "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2, "text_config": cfg.to_hf_text_dict(),
          "quantization_config": qcfg}
    hf["text_config"]["vocab_size"] = cfg.vocab_size
    json.dump(hf, open(os.path.join(path, "config.json"), "w"), indent=2)
    tk.save(os.path.join(path, "tokenizer.json"))
    return cfg, sd, tables, tmap


def synth_config(**over) -> DSV41Config:
    """Tiny structure-faithful config with 128-aligned dims (the framework Linear needs no padding):
    8 trunk layers + 2 dspark stages, kv sources (2, 4), index sources (2, 4, 5), candidate source 4, engram 1 / 5."""
    from .config import tiny_config
    d = dict(hidden_size=128, moe_intermediate_size=128, q_lora_rank=128, o_lora_rank=128, head_dim=128,
             qk_rope_head_dim=16, o_groups=2, num_attention_heads=4, index_n_heads=16, index_head_dim=128,
             engram_head_dim=64, engram_n_heads=2, vocab_size=512)
    d.update(over)
    return tiny_config(**d)
