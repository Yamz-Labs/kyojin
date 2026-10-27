"""Synthetic EXL3 DS4.1 DEPTH pack at REAL dimensions: 4 trunk layers that mirror the attention roles of the real stack
  layer 0 = layer 2/8/14 (ratio 2, kv + index source, full pool scoring)
  layer 1 = layer 20     (ratio 1, kv + index + candidate source)
  layer 2 = layer 24..36 (ratio 1, index source reading the candidate blocks)
  layer 3 = layer 21..   (ratio 1, consumer of the shared top-k)
usage: ds41_make_depth_pack.py <out_dir> [seed=0]   (engine tree on PYTHONPATH)"""
import sys, os, json, copy, torch
from safetensors.torch import save_file
from exllamav3.dsv41.config import DSV41Config
from exllamav3.dsv41.engram import build_compressed_token_map
from exllamav3.dsv41 import synth_pack as SP

out, N = sys.argv[1], 4
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
os.makedirs(out, exist_ok=True)
cfg = DSV41Config()                       # real V4.1-Flash dims
cfg.num_hidden_layers = 4; cfg.compress_ratios = [2, 1, 1, 1]; cfg.kv_source_layer_ids = [0, 1]
cfg.index_source_layer_ids = [0, 1, 2]; cfg.candidate_source_layer_id = 1
cfg.num_nextn_predict_layers = 0; cfg.dspark_target_layer_ids = []; cfg.engram_layer_ids = []; cfg.engram_num_embeddings = []
E, d, I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
g = torch.Generator().manual_seed(seed)
small = copy.deepcopy(cfg); small.num_hidden_layers = N; small.n_routed_experts = 1
sd = SP.random_state_dict(small, seed)
LIN = ("wq_a", "wq_b", ".wkv", "wo_a.slice", "wo_b", "indexer.wq_b", "shared_experts")

def exl3(prefix, in_f, out_f, k):
    return {prefix + ".trellis": torch.randint(-32768, 32768, (in_f // 16, out_f // 16, 16 * k), generator=g, dtype=torch.int16),
            prefix + ".suh": (torch.randint(0, 2, (in_f,), generator=g) * 2 - 1).half(),
            prefix + ".svh": ((torch.randint(0, 2, (out_f,), generator=g) * 2 - 1) / in_f ** 0.5).half(),
            prefix + ".mul1": torch.tensor([0x83DCD12D - (1 << 32)], dtype=torch.int32)}

new = {}
for k, v in sd.items():
    if k.endswith(".weight") and v.dtype == torch.float16 and v.dim() == 2 and any(t in k for t in LIN) and "compressor" not in k:
        o, i = v.shape
        new.update(exl3(k[:-7], i, o, 4))
    else:
        new[k] = v
sd = new
sd = {k: v for k, v in sd.items() if ".ffn.experts." not in k}
expert_bits = {}
for l in range(N):
    p = f"layers.{l}.ffn"
    sd[p + ".gate.weight"] = (torch.randn(E, d, generator=g) * d ** -0.5).to(torch.bfloat16)
    bias = 0.2 * torch.randn(E, generator=g)
    perm = torch.randperm(E, generator=g).tolist()
    ks = [0] * 32 + [3] * 32 + [1] * 120 + [2] * (E - 184)             # 32 pruned, 32 hot K3, 120 K1, rest K2
    kbits = [0] * E
    for pos, e in enumerate(perm): kbits[e] = ks[pos]
    for e, kk in enumerate(kbits):
        if kk == 0:
            bias[e] = -1e4; continue
        for w, (i, o) in (("w1", (d, I)), ("w3", (d, I)), ("w2", (I, d))):
            sd.update(exl3(f"layers.{l}.ffn.experts.{e}.{w}", i, o, kk))
    sd[p + ".gate.bias"] = bias.float()
    expert_bits[p] = kbits
    print("layer", l, "K counts", {k: kbits.count(k) for k in (0, 1, 2, 3)}, flush=True)

tk, shim = SP.make_tokenizer(cfg.vocab_size)
tmap, nv = build_compressed_token_map(shim); cfg.engram_compressed_vocab_size = nv
units = {}
for k, v in sd.items(): units.setdefault(SP._unit_of(k), {})[k] = v.contiguous()
wm, total = {}, 0
for u, t in units.items():
    f = f"{u}.safetensors"; save_file(t, os.path.join(out, f), metadata={"format": "pt"})
    total += os.path.getsize(os.path.join(out, f))
    for k in t: wm[k] = f
json.dump({"metadata": {"total_size": total}, "weight_map": wm}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
qcfg = {"quant_method": "exl3", "bits": 4, "head_bits": 16, "mtp_bits": 4, "codebook": "mul1", "synthetic": True, "expert_bits": expert_bits}
hf = {"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "dtype": "bfloat16", "bos_token_id": 0, "eos_token_id": 1,
      "pad_token_id": 2, "text_config": cfg.to_hf_text_dict(), "quantization_config": qcfg}
hf["text_config"]["vocab_size"] = cfg.vocab_size
json.dump(hf, open(os.path.join(out, "config.json"), "w"), indent=2)
tk.save(os.path.join(out, "tokenizer.json"))
print("pack", out, f"{total / 2**30:.2f} GiB")
