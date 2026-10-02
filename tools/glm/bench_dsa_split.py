"""Isolated microbench for the GLM decode DSA attention split kernel
(_dsa_attn_split_kernel in exllamav3/modules/attention_fn/dsa_triton.py).

No kernel change: this only builds the exact GLM decode
shapes, times the same Python entry point the model uses (dsa_attn -> split +
combine) and prints a reference (plain torch gather + softmax) plus rel-L2 so a
future kernel variant can be gated numerically.

GLM-5.3 (glm53-exl3-td205) decode shapes, from config.json text_config:
  num_attention_heads  H  = 64
  kv_lora_rank        D_c = 512
  qk_rope_head_dim    D_r = 0    (absorbed queries: q_pe is (R, H, 0), OUT_LATENT)
  index_topk          K   = 2048
  index_kpool         compress = 4 -> pool entries = context // 4
  11 deepseek_sparse_attention layers (layer_types) -> 11 calls/token

Decode call path: mla_attn.py:1206 _attend_sparse -> dsa_attn(q_pe=..., out_latent=True)
-> n_splits = 16 (R<=8, est > 256) -> block_h = min(32,16) = 16 -> hb = 4 ->
grid (R*hb, n_splits) = (4R, 16) -> _dsa_attn_split_kernel + _dsa_attn_combine_kernel.

usage: python tools/glm/bench_dsa_split.py [--ctx 4096,32768] [--rows 1,2]
                                          [--splits 16] [--tune 32,4;64,4] [--dt 0]
"""
import argparse
import os
import statistics
import sys
from itertools import product

import torch
import triton

sys.path.insert(0, os.getcwd())
torch.set_grad_enabled(False)

import exllamav3.modules.attention_fn.dsa_triton as D

H = 64
D_C = 512
D_R = 0
TOPK = 2048
COMPRESS = 4
PAGE_SIZE = 64          # ckv_cache.shape[1] (block-table page, paged pool)
SCALE = D_C ** -0.5
N_CU = torch.cuda.get_device_properties(0).multi_processor_count


def build(ctx, R, device="cuda:0"):
    """Exact decode inputs -> (dsa_attn args, kwargs, valid entries).

    The model always passes the full (R, index_topk) selection, -1 padded when the
    context/kpool offers fewer entries, and k_len = indices.shape[1]
    (mla_attn.py:1208) -- so k_len is TOPK at every context, not min(ctx/4, TOPK).
    """
    pool_len = max(1, ctx // COMPRESS)
    valid = min(TOPK, pool_len)
    pages = -(-pool_len // PAGE_SIZE)
    g = torch.Generator(device="cpu").manual_seed(1234)

    # pool_c (pages, page_size, D_c) fp16; pool_r empty (D_r == 0)
    pool_c = torch.randn(pages, PAGE_SIZE, D_C, generator=g, dtype=torch.float32).half().to(device)
    pool_r = torch.empty(pages, PAGE_SIZE, 0, dtype=torch.float16, device=device)
    # identity block table (contiguous per-slot pool), one row shared by every q row
    block_table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
    # top-k selection: a random sorted subset of pool entries, -1 padded up to TOPK
    sel = torch.stack([torch.randperm(pool_len, generator=g)[:valid].sort().values
                       for _ in range(R)]).to(torch.int32).to(device)
    indices = torch.full((R, TOPK), -1, dtype=torch.int32, device=device)
    indices[:, :valid] = sel
    q_lat = torch.randn(H, R, D_C, generator=g, dtype=torch.float32).half().to(device)
    q_pe = torch.empty(R, H, D_R, dtype=torch.float16, device=device)
    # positional order of dsa_attn(): q, pool_c, pool_r, block_table
    return ((q_lat, pool_c, pool_r, block_table),
            dict(indices=indices, k_len=TOPK, pool_len=pool_len, scale=SCALE,
                 page_size=PAGE_SIZE, q_pe=q_pe, out_latent=True),
            valid)


def timed(fn, reps, warmup=20):
    """(median of `reps` synced calls, amortized back-to-back mean). CUDA events.

    The amortized mean keeps host launch cost but drops the fixed per-call event
    overhead, so median - amort isolates that overhead."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(reps)]
    for e0, e1 in ev:
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return statistics.median(e0.elapsed_time(e1) for e0, e1 in ev), e0.elapsed_time(e1) / reps


def kernel_ms(t, reps=200, warmup=20):
    return timed(lambda: D.dsa_attn(*t[0], **t[1]), reps, warmup)


def ref_attn(t):
    """Plain torch: gather top-k latents, fp32 dot + softmax + weighted sum. (H, R, D_c)

    Masks the -1 PADDED entries -- mask the valid ones and a context with no padding
    (32K) softmaxes an all -inf row into NaN.
    """
    q_lat, pool_c = t[0][0], t[0][1]
    idx = t[1]["indices"].long()
    page, tok = idx.clamp(min=0) // PAGE_SIZE, idx.clamp(min=0) % PAGE_SIZE
    kc = pool_c.reshape(-1, D_C)[t[0][3][0][page].long() * PAGE_SIZE + tok]  # (R, K, D_c)
    s = torch.einsum("hrc,rkc->hrk", q_lat.float(), kc.float()) * SCALE
    p = torch.softmax(s.masked_fill(~(idx >= 0)[None], float("-inf")), dim=-1)
    return torch.einsum("hrk,rkc->hrc", p, kc.float())


def rel_l2(a, b):
    return ((a - b).norm() / b.norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="4096,32768")
    ap.add_argument("--rows", default="1,2")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--dt", default="4",
                    help="EXL3_DSA_DEC_DT_R: rows <= this use the dt kernel (served default 4; 0 = plain split kernel)")
    ap.add_argument("--splits", type=int, default=0, help="0 = launcher default (16)")
    ap.add_argument("--tune", default="", help="BN,NW sweep, e.g. 32,4;64,4;64,8 (block_n,num_warps)")
    a = ap.parse_args()
    os.environ["EXL3_DSA_DEC_DT_R"] = a.dt

    print(f"dev={torch.cuda.get_device_name(0)} CUs={N_CU} H={H} D_c={D_C} D_r={D_R} "
          f"topk={TOPK} compress={COMPRESS} page={PAGE_SIZE} dt_cfg_R={a.dt}")
    print(f"{'ctx':>6} {'R':>2} {'k_len':>6} {'n_splits':>8} {'grid':>10} {'progs':>6} "
          f"{'ms':>8} {'amort':>8} {'MB':>7} {'GB/s':>7} {'relL2':>9}")
    tune = [s for s in a.tune.split(";") if s] or [""]
    for ctx, R, spec in product([int(x) for x in a.ctx.split(",")],
                                [int(x) for x in a.rows.split(",")], tune):
        t = build(ctx, R)
        if a.splits:
            t[1]["n_splits"] = a.splits
        if spec:
            t[1]["block_n"], t[1]["num_warps"] = (int(x) for x in spec.split(","))
        out = D.dsa_attn(*t[0], **t[1]).float()
        rel = rel_l2(out, ref_attn(t))
        ms, am = kernel_ms(t, reps=a.reps)
        # grid the launcher used (dsa_triton.py:1345-1358, 1401)
        n_splits = a.splits or (16 if min(t[1]["k_len"], t[1]["pool_len"]) > 256 else 8)
        hb = triton.cdiv(H, 16)                       # block_h = min(32, 16)
        by = t[2] * D_C * 2                           # valid entries gathered, fp16
        print(f"{ctx:>6} {R:>2} {t[1]['k_len']:>6} {n_splits:>8} "
              f"{str((R * hb, n_splits)):>10} {R * hb * n_splits:>6} {ms:>8.3f} {am:>8.3f} "
              f"{by / 1e6:>7.2f} {by / ms / 1e6:>7.2f} {rel:>9.2e}"
              f"  [bn,nw={spec or '32,4'}]", flush=True)
        del t, out
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
