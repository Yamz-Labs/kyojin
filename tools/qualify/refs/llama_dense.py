"""Streamed BF16 reference for dense Llama-arch packs (GQA, full attention, RoPE half-split): layer by layer through the GPU, so a 66 GB model
runs on a 24 GB card. COPIED from fleet/scratch_nanb/tools/ref_stream.py (drill 0, nanb) with attribution; unchanged maths.
usage: llama_dense.py MODEL_DIR DOCS.pt OUTDIR   -> OUTDIR/logits_<name>.pt (fp16 [T,V] or [TAIL,V]; position i predicts i+1); env TAIL=n keeps the last n rows.
Per-arch: a new family adds refs/<family>.py with the same CLI; pack.FAMILIES[...]['ref'] names it."""
import sys, os, json, glob, torch, time
from safetensors import safe_open
M, DOCS, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
cfg = json.load(open(f"{M}/config.json")); dev = "cuda"
wm = json.load(open(f"{M}/model.safetensors.index.json"))["weight_map"]
H, NH, NKV, HD, EPS, TH = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"], cfg["rms_norm_eps"], cfg["rope_theta"]
handles = {}
def get(n):
    f = wm[n]
    if f not in handles: handles[f] = safe_open(f"{M}/{f}", "pt", device="cpu")
    return handles[f].get_tensor(n).to(dev, non_blocking=True)
def rms(x, w):   # fp32 math, bf16 out (matmul input)
    xf = x.float(); return ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)) * w.float()).to(torch.bfloat16)
def rope(x, cos, sin):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
d = torch.load(DOCS); names, ids = d["names"], d["ids"]
inv = 1.0 / (TH ** (torch.arange(0, HD, 2, device=dev, dtype=torch.float64) / HD))
cs = {}
with torch.inference_mode():
    emb = get("model.embed_tokens.weight")
    hs = [emb[torch.tensor(i, device=dev)].unsqueeze(0).float().contiguous() for i in ids]; del emb
    for n_i, i in enumerate(ids):
        fr = torch.outer(torch.arange(len(i), device=dev, dtype=torch.float64), inv)
        cs[n_i] = (fr.cos().float()[None, None], fr.sin().float()[None, None])
    t0 = time.time()
    for L in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{L}."
        w = {k: get(p + k + ".weight") for k in ["input_layernorm", "post_attention_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]}
        for j, h in enumerate(hs):
            T = h.shape[1]; c, s = cs[j]
            x = rms(h, w["input_layernorm"])
            q = (x @ w["self_attn.q_proj"].T).view(1, T, NH, HD).transpose(1, 2)
            k = (x @ w["self_attn.k_proj"].T).view(1, T, NKV, HD).transpose(1, 2)
            v = (x @ w["self_attn.v_proj"].T).view(1, T, NKV, HD).transpose(1, 2)
            q, k = rope(q.float(), c, s).to(torch.bfloat16), rope(k.float(), c, s).to(torch.bfloat16)
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            h = h + (a.transpose(1, 2).reshape(1, T, NH * HD) @ w["self_attn.o_proj"].T).float()
            x = rms(h, w["post_attention_layernorm"])
            m = torch.nn.functional.silu(x @ w["mlp.gate_proj"].T) * (x @ w["mlp.up_proj"].T)
            hs[j] = h + (m @ w["mlp.down_proj"].T).float()
        del w
        if L % 8 == 0: print(f"layer {L} {time.time()-t0:.0f}s", flush=True)
    nw = get("model.norm.weight"); hd = get("lm_head.weight")
    for j, n in enumerate(names):
        TAIL = int(os.environ.get("TAIL", "0")); hj = hs[j][:, -TAIL:] if TAIL else hs[j]
        lg = (rms(hj, nw)[0] @ hd.T).float()
        torch.save(lg.half().cpu(), f"{OUT}/logits_{n}.pt"); print(n, tuple(lg.shape), flush=True)
