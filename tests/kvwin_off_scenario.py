"""Deterministic quantized-cache attention scenario WITHOUT the KVWIN ring (prefill chunk, decode, verify block) through
attn_dispatch. Usage: python kvwin_off_scenario.py <exllamav3 tree> <out.pt>; run it on a tree before and after the KVWIN
change: the saved outputs must be bit-identical (EXL3_KVWIN unset = today)."""
import sys, types, torch
tree, out = sys.argv[1], sys.argv[2]
sys.path[:0] = sys.argv[3:] + [tree]     # optional directory holding only the built exllamav3_ext (wins over a .so inside the tree)
from exllamav3.cache.quant import CacheLayer_quant
from exllamav3.modules.attention_fn.dispatch import attn_dispatch

dev = "cuda:0"
torch.manual_seed(0)
kvh, qh, hd = 2, 8, 128
res = {}
for bits in ((4, 4), (6, 6), (8, 8)):
    attn = types.SimpleNamespace(num_kv_heads = kvh, head_dim = hd)
    cl = CacheLayer_quant(None, attn, 0, 8 * 256, *bits)
    cl.alloc(torch.device(dev))
    bt = torch.tensor([[5, 2, 7, 0]], dtype = torch.int32, device = dev)
    L = 0
    outs = []
    for q_len in (300, 1, 5, 16, 1, 40, 1):
        q = torch.randn(1, q_len, qh, hd, device = dev).half()
        k = torch.randn(1, q_len, kvh, hd, device = dev).half()
        v = torch.randn(1, q_len, kvh, hd, device = dev).half()
        sl = torch.tensor([L], dtype = torch.int32, device = dev)
        o = attn_dispatch(q = q, k = k, v = v, cache = cl, cache_idx = 0, block_table = bt, cache_seqlens = sl,
                          causal = True, sm_scale = hd ** -0.5, max_kv_len = L)
        outs.append(o.cpu())
        L += q_len
    res[bits] = outs
torch.save(res, out)
print("saved", out)
