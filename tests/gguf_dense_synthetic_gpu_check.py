"""GPU: a tiny qwen3-arch GGUF (random weights, Q4_0/Q8_0 via gguf-py) served by GGUFDenseModel must match a torch
reference forward over the dequantized weights. Needs the built extension and the gguf package.
Usage: python tests/gguf_dense_synthetic_gpu_check.py <scratch_dir> [fmt]"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, gguf
from gguf import GGMLQuantizationType as T

out = sys.argv[1]; FMT = sys.argv[2] if len(sys.argv) > 2 else "Q4_0"
os.makedirs(out, exist_ok=True)
H, nq, nkv, hd, I, V, NL = 256, 4, 2, 64, 512, 512, 2
rng = np.random.default_rng(0)
path = os.path.join(out, "tiny.gguf")
w = gguf.GGUFWriter(path, "qwen3")
w.add_block_count(NL); w.add_context_length(512); w.add_embedding_length(H); w.add_feed_forward_length(I)
w.add_head_count(nq); w.add_head_count_kv(nkv); w.add_key_length(hd); w.add_value_length(hd)
w.add_layer_norm_rms_eps(1e-6); w.add_rope_freq_base(10000.0)
ref = {}
qt = getattr(T, FMT)
def mat(name, n, k, quant=True):
    a = (rng.standard_normal((n, k)) * 0.05).astype(np.float32)
    if quant:
        q = gguf.quants.quantize(a, qt); ref[name] = torch.from_numpy(gguf.quants.dequantize(q, qt))
        w.add_tensor(name, q, raw_dtype=qt)
    else:
        ref[name] = torch.from_numpy(a); w.add_tensor(name, a)
def vec(name, n):
    a = (1 + 0.1 * rng.standard_normal(n)).astype(np.float32); ref[name] = torch.from_numpy(a); w.add_tensor(name, a)
mat("token_embd.weight", V, H)
vec("output_norm.weight", H)
for i in range(NL):
    g = f"blk.{i}"
    vec(f"{g}.attn_norm.weight", H); vec(f"{g}.ffn_norm.weight", H)
    mat(f"{g}.attn_q.weight", nq * hd, H); mat(f"{g}.attn_k.weight", nkv * hd, H); mat(f"{g}.attn_v.weight", nkv * hd, H)
    mat(f"{g}.attn_output.weight", H, nq * hd)
    vec(f"{g}.attn_q_norm.weight", hd); vec(f"{g}.attn_k_norm.weight", hd)
    mat(f"{g}.ffn_gate.weight", I, H); mat(f"{g}.ffn_up.weight", I, H); mat(f"{g}.ffn_down.weight", H, I)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

def rms(x, wt): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * wt
def rope(x, pos):  # NEOX, x [S, h, hd]
    half = hd // 2
    f = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32) / half))
    ang = pos[:, None] * f[None]; c, s = ang.cos()[:, None], ang.sin()[:, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)
ids = torch.tensor([3, 17, 99, 250, 5, 71, 400, 12])
S = len(ids); pos = torch.arange(S, dtype=torch.float32)
x = ref["token_embd.weight"][ids]
for i in range(NL):
    g = f"blk.{i}"
    h = rms(x, ref[f"{g}.attn_norm.weight"])
    q = (h @ ref[f"{g}.attn_q.weight"].T).view(S, nq, hd); k = (h @ ref[f"{g}.attn_k.weight"].T).view(S, nkv, hd)
    v = (h @ ref[f"{g}.attn_v.weight"].T).view(S, nkv, hd)
    q = rope(rms(q, ref[f"{g}.attn_q_norm.weight"]), pos); k = rope(rms(k, ref[f"{g}.attn_k_norm.weight"]), pos)
    k = k.repeat_interleave(nq // nkv, 1); v = v.repeat_interleave(nq // nkv, 1)
    att = torch.einsum("qhd,khd->hqk", q, k) / hd ** 0.5
    att = att.masked_fill(torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf")).softmax(-1)
    x = x + torch.einsum("hqk,khd->qhd", att, v).reshape(S, -1) @ ref[f"{g}.attn_output.weight"].T
    h = rms(x, ref[f"{g}.ffn_norm.weight"])
    x = x + (torch.nn.functional.silu(h @ ref[f"{g}.ffn_gate.weight"].T) * (h @ ref[f"{g}.ffn_up.weight"].T)) @ ref[f"{g}.ffn_down.weight"].T
ref_logits = rms(x, ref["output_norm.weight"]) @ ref["token_embd.weight"].T

from exllamav3.loader.gguf_config import write_config_dir
from exllamav3.model.config import Config
from exllamav3.model.model import Model
d = write_config_dir(path, os.path.join(out, "cfg"))
config = Config.from_directory(d); model = Model.from_config(config); model.load()
got = model.forward(ids[None].cuda(), params={"attn_mode": "flash_attn_nc"})[0].float().cpu()
err = ((got - ref_logits).norm() / ref_logits.norm()).item()
print(FMT, "rel err", err, "argmax match", (got.argmax(-1) == ref_logits.argmax(-1)).float().mean().item())
assert err < 2e-2
