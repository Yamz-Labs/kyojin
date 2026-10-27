"""Synthetic EXL3 DS4.1 pack at REAL dimensions with the dspark stages: first N trunk layers (symlinked from a trunk-only pack
made by make_synth_pack-style tooling or built here) + 3 dspark stages (128 routed experts top-3, shared expert, K=4 linears).

usage: dspark_synth_pack.py <trunk_pack_dir> <out_dir> <expert_bits: 1|2|4|1,2|mix> [trunk_layers=3] [seed=7]
  expert_bits  K of the draft routed experts ("1,2" = half K1, half K2). Trunk shards are symlinked from <trunk_pack_dir>.
"""
import copy
import json
import os
import sys

import torch
from safetensors.torch import save_file

from exllamav3.dsv41.config import DSV41Config
from exllamav3.dsv41 import synth_pack as SP

LIN = ("wq_a", "wq_b", ".wkv", "wo_a.slice", "wo_b", "shared_experts", "main_proj")


def exl3(prefix, in_f, out_f, k, g):
    return {prefix + ".trellis": torch.randint(-32768, 32768, (in_f // 16, out_f // 16, 16 * k), generator=g, dtype=torch.int16),
            prefix + ".suh": (torch.randint(0, 2, (in_f,), generator=g) * 2 - 1).half(),
            prefix + ".svh": ((torch.randint(0, 2, (out_f,), generator=g) * 2 - 1) / in_f ** 0.5).half(),
            prefix + ".mul1": torch.tensor([0x83DCD12D - (1 << 32)], dtype=torch.int32)}


def pack_config(cfg: DSV41Config, n: int) -> DSV41Config:
    """Real dims, first n trunk layers, all dspark stages kept, taps = the n trunk layers' inputs (ids 0..n-1)."""
    c = copy.deepcopy(cfg)
    c.num_hidden_layers = n
    c.dspark_target_layer_ids = list(range(n))
    c.kv_source_layer_ids = [i for i in c.kv_source_layer_ids if i < n]
    c.index_source_layer_ids = [i for i in c.index_source_layer_ids if i < n]
    if c.candidate_source_layer_id >= n:
        c.candidate_source_layer_id = -1
    c.engram_layer_ids, c.engram_num_embeddings = [], []
    return c


def build(trunk_pack, out, kspec, n=3, seed=7):
    os.makedirs(out, exist_ok=True)
    full = DSV41Config()
    cfg = pack_config(full, n)
    d, I = cfg.hidden_size, cfg.moe_intermediate_size
    ne, topk = cfg.dspark_n_routed_experts, cfg.dspark_num_experts_per_tok
    small = copy.deepcopy(cfg)
    small.n_routed_experts = 1
    small.dspark_n_routed_experts = 1
    sd = SP.random_state_dict(small, seed)
    g = torch.Generator().manual_seed(seed)
    new = {}
    for k, v in sd.items():
        if not k.startswith("mtp."):
            continue
        if ".ffn.experts." in k:
            continue
        if k.endswith(".weight") and v.dtype == torch.float16 and v.dim() == 2 and any(t in k for t in LIN):
            o, i = v.shape
            new.update(exl3(k[:-7], i, o, 4, g))
        else:
            new[k] = v
    ks_all = {"1": [1], "2": [2], "4": [4], "1,2": [1, 2], "mix": [1, 1, 2, 4]}[kspec]
    expert_bits = {}
    for j in range(cfg.n_mtp_layers):
        p = f"mtp.{j}.ffn"
        kb = [ks_all[e % len(ks_all)] for e in range(ne)]
        for e, kk in enumerate(kb):
            for w, (i, o) in (("w1", (d, I)), ("w3", (d, I)), ("w2", (I, d))):
                new.update(exl3(f"mtp.{j}.ffn.experts.{e}.{w}", i, o, kk, g))
        new[p + ".gate.weight"] = (torch.randn(ne, d, generator=g) * d ** -0.5).to(torch.bfloat16)
        new[p + ".gate.bias"] = 0.2 * torch.randn(ne, generator=g)
        expert_bits[p] = kb
    # trunk expert_bits from the trunk pack config
    tcfg = json.load(open(os.path.join(trunk_pack, "config.json")))
    expert_bits.update(tcfg["quantization_config"]["expert_bits"])
    units = {}
    for k, v in new.items():
        units.setdefault(SP._unit_of(k), {})[k] = v.contiguous()
    tidx = json.load(open(os.path.join(trunk_pack, "model.safetensors.index.json")))
    wm = {k: f for k, f in tidx["weight_map"].items()}
    total = tidx["metadata"]["total_size"]
    for f in set(wm.values()):
        dst = os.path.join(out, f)
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(os.path.join(trunk_pack, f)), dst)
    for u, t in units.items():
        f = f"{u}.safetensors"
        save_file(t, os.path.join(out, f), metadata={"format": "pt"})
        total += os.path.getsize(os.path.join(out, f))
        for k in t:
            wm[k] = f
    json.dump({"metadata": {"total_size": total}, "weight_map": wm}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
    hf = copy.deepcopy(tcfg)
    hf["text_config"] = cfg.to_hf_text_dict()
    hf["text_config"]["vocab_size"] = cfg.vocab_size
    hf["quantization_config"]["expert_bits"] = expert_bits
    hf["quantization_config"]["mtp_bits"] = 4
    json.dump(hf, open(os.path.join(out, "config.json"), "w"), indent=2)
    for f in ("tokenizer.json",):
        dst = os.path.join(out, f)
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(os.path.join(trunk_pack, f)), dst)
    print("pack", out, f"mtp experts K={kspec}", f"{total / 2**30:.2f} GiB")


if __name__ == "__main__":
    a = sys.argv
    build(a[1], a[2], a[3], int(a[4]) if len(a) > 4 else 3, int(a[5]) if len(a) > 5 else 7)
