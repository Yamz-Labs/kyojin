"""Tiny random DS4.1 checkpoint on disk in the OFFICIAL layout (config.json with text_config, one safetensors shard,
engram tables as `layers.N.engram.embed.{weight,scale}`, tokenizer.json), 128-aligned dims so the framework's
Linear needs no padding."""
from __future__ import annotations
import json, os, sys
import torch
from safetensors.torch import save_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import FakeTok, make_state_dict, make_engram_tables, tiny_config, _fp4_row_quant
from exllamav3.dsv41.engram import build_compressed_token_map, EngramLayout


def fw_config(**over):
    d = dict(hidden_size=128, moe_intermediate_size=128, q_lora_rank=128, o_lora_rank=128, head_dim=128,
             qk_rope_head_dim=16, o_groups=2, num_attention_heads=4, index_n_heads=4, index_head_dim=32,
             engram_head_dim=64, engram_n_heads=2, vocab_size=512)
    d.update(over)
    return tiny_config(**d)


def write_tiny(path: str, seed: int = 0, cfg=None, engram_fmt: str = "fp8"):
    os.makedirs(path, exist_ok=True)
    cfg = cfg or fw_config()
    tok = FakeTok(cfg.vocab_size)
    tmap, nv = build_compressed_token_map(tok)
    cfg.engram_compressed_vocab_size = nv
    lay = EngramLayout.build(cfg.engram_layer_ids, None, cfg.engram_max_ngram_size, cfg.engram_vocab_size,
                             cfg.engram_n_heads, cfg.engram_head_dim)
    cfg.engram_num_embeddings = list(lay.num_embeddings)
    tables = make_engram_tables(cfg, lay.num_embeddings, seed + 1)
    if engram_fmt == "mxfp4":
        # transcode the e4m3 rows like the MXFP4 release: e2m1 pairs (low nibble first) + e8m0 per 32
        tt = []
        for v, s in tables:
            x = (v.float().unflatten(-1, (-1, 32)) * s.view(torch.float8_e8m0fnu).float().unsqueeze(-1)).flatten(-2)
            packed, sc = _fp4_row_quant(x)
            tt.append((packed.view(torch.uint8), sc))
        tables = tt
    sd = make_state_dict(cfg, seed, fp8=True, fp4=True)
    for i, lid in enumerate(cfg.engram_layer_ids):
        sd[f"layers.{lid}.engram.embed.weight"] = tables[i][0].contiguous()
        sd[f"layers.{lid}.engram.embed.scale"] = tables[i][1].contiguous()
    sd = {k: v.contiguous() for k, v in sd.items()}
    save_file(sd, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    hf = {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "dtype": "bfloat16",
          "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
          "text_config": cfg.to_hf_text_dict()}
    hf["text_config"]["vocab_size"] = cfg.vocab_size
    json.dump(hf, open(os.path.join(path, "config.json"), "w"))
    tok.backend_tokenizer.save(os.path.join(path, "tokenizer.json"))
    return cfg, sd, tables, tmap
