"""MiMo global-attention prefill cost: one 2048-row chunk appended at a given past length,
through paged_attn_triton_prefill (the kernel the model's global layers call). Head geometry:
64 q heads, 4 kv heads, qk dim 192, v padded to 192. Optional tile overrides (bm bn warps stages)."""
import argparse, os, statistics, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
p = argparse.ArgumentParser()
p.add_argument("--past", type = int, nargs = "+", default = [0, 2048, 6144, 14336])
p.add_argument("--chunk", type = int, default = 2048)
p.add_argument("--kvh", type = int, default = 4)
p.add_argument("--window", type = int, default = -1)
p.add_argument("--cfg", type = str, nargs = "*", default = ["default"])
p.add_argument("--vdim", type = int, nargs = "+", default = [192, 128])
p.add_argument("--mla", action = "store_true", help = "GLM MLA MHA-form prefill (mla_attn_triton_prefill_mha) instead")
p.add_argument("--mla-q", type = int, nargs = "+", default = [2048], help = "--mla: q_len per case (zipped with --past)")
p.add_argument("--mla-env", type = str, nargs = "*", default = ["0", "1"], help = "--mla: EXL3_MLA_MHA values, paired in one process")
p.add_argument("--reps", type = int, default = 5)
p.add_argument("--inner", type = int, default = 4, help = "--mla: calls per timed event pair")
p.add_argument("--tile", type = int, default = 4096, help = "--mla: tile_size (small = multi-tile state carry)")
p.add_argument("--meta", action = "store_true", help = "--mla: print regs/spills of the compiled kernels")
args = p.parse_args()


def mla_bench():
    """GLM MLA dense prefill: H 64, D_nope 256, D_r 0, D_v 256, D_c 512, fp16 cache, page 256.
    Times the whole mla_attn_triton_prefill_mha call and the gather+2 GEMMs alone (kernel ms =
    difference), and checks every variant against an fp32 torch reference on the same fp16
    up-projection. Variants alternate a,b,a,b per rep."""
    import exllamav3.modules.attention_fn.mla_triton as M
    DEV = "cuda:0"; H = 64; DN = 256; DV = 256; DC = 512; PAGE = 256
    torch.manual_seed(0)
    w_uk = (torch.randn((DC, H * DN), device = DEV) * DC ** -0.5).half()
    w_uv = (torch.randn((DC, H * DV), device = DEV) * DC ** -0.5).half()
    scale = DN ** -0.5
    pasts = args.past if len(args.past) == len(args.mla_q) else [0] * len(args.mla_q)
    for q_len, past in zip(args.mla_q, pasts):
        total = past + q_len
        pages = (total + PAGE - 1) // PAGE
        ckv = torch.randn((pages, PAGE, 1, DC), device = DEV).half()
        kpe = torch.empty((pages, PAGE, 1, 0), device = DEV, dtype = torch.half)
        bt = torch.randperm(pages, device = DEV).int()[None]
        # peaky scores like a trained model: scale q up so softmax is not flat
        q = (torch.randn((q_len, H, DN), device = DEV) * 2.0).half()
        call = lambda: M.mla_attn_triton_prefill_mha(q, w_uk, w_uv, ckv, kpe, bt, [past], 1, q_len, DV, DN, scale,
                                                     pre_appended_len = q_len, tile_size = args.tile)
        # reference: same gather order + fp16 up-projection, fp32 attention
        lat = ckv.view(-1, DC)[(bt[0].long()[:, None] * PAGE + torch.arange(PAGE, device = DEV)[None]).view(-1)[:total]]
        kn = torch.mm(lat, w_uk).view(total, H, DN); vv = torch.mm(lat, w_uv).view(total, H, DV)
        ref = torch.empty((q_len, H, DV), device = DEV)
        causal = torch.arange(total, device = DEV)[None, :] <= (past + torch.arange(q_len, device = DEV))[:, None]
        for h0 in range(0, H, 8):
            s = torch.einsum("qhd,khd->hqk", q[:, h0:h0 + 8].float(), kn[:, h0:h0 + 8].float()) * scale
            s = s.masked_fill(~causal[None], -float("inf")).softmax(-1)
            ref[:, h0:h0 + 8] = torch.einsum("hqk,khd->qhd", s, vv[:, h0:h0 + 8].float())
        def upproj():
            torch.mm(lat, w_uk); torch.mm(lat, w_uv)
        def timed(f):
            e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
            e0.record()
            for _ in range(args.inner): f()
            e1.record(); e1.synchronize(); return e0.elapsed_time(e1) / args.inner
        outs, ts = {}, {e: [] for e in args.mla_env}
        for e in args.mla_env:
            os.environ["EXL3_MLA_MHA"] = e
            outs[e] = call().float()
            for _ in range(10): call()
        t_up = statistics.median([timed(upproj) for _ in range(args.reps)])
        for _ in range(args.reps):
            for e in args.mla_env:
                os.environ["EXL3_MLA_MHA"] = e
                ts[e].append(timed(call))
        flops = 2 * H * q_len * (past + (q_len + 1) / 2) * (DN + DV)
        base = statistics.median(ts[args.mla_env[0]])
        for e in args.mla_env:
            o = outs[e]
            rel = ((o - ref).norm() / ref.norm()).item(); mx = (o - ref).abs().max().item()
            vs0 = (o - outs[args.mla_env[0]]).abs().max().item()
            t = statistics.median(ts[e]); tk = t - t_up
            print(f"mla q {q_len} past {past} tile {args.tile} env {e:>18s}: x{base / t:5.2f} call {t:7.2f} ms  kernel~{tk:7.2f} ms "
                  f"({flops / tk / 1e9:5.1f} TFLOP/s)  relL2 {rel:.2e} maxabs {mx:.2e}  vs[{args.mla_env[0]}] {vs0:.2e}",
                  flush = True)
    if args.meta:
        for kern in (M._mla_prefill_mha_kernel, M._mla_prefill_mha2_kernel):
            for dev, c in kern.device_caches.items():
                for key, ck in c[0].items():
                    md = ck.metadata
                    print("META", getattr(kern, "__name__", "?"), "regs", ck.n_regs, "spills", ck.n_spills, "lds", md.shared,
                          "warps", md.num_warps, flush = True)


if args.mla:
    mla_bench()
    sys.exit(0)
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill
DEV = "cuda:0"; H = 64; D = 192; PAGE = 256
torch.manual_seed(0)
for past in args.past:
    total = past + args.chunk
    pages = (total + PAGE - 1) // PAGE
    kc = torch.randn((pages, PAGE, args.kvh, D), device = DEV).half() * 0.5
    vc = torch.randn((pages, PAGE, args.kvh, D), device = DEV).half()
    bt = torch.arange(pages, dtype = torch.int32, device = DEV)[None]
    q = torch.randn((1, args.chunk, H, D), device = DEV).half()
    k = torch.randn((1, args.chunk, args.kvh, D), device = DEV).half()
    v = torch.randn((1, args.chunk, args.kvh, D), device = DEV).half()
    sl0 = torch.tensor([past], dtype = torch.int32, device = DEV)
    w_ = args.window if args.window >= 0 else None
    o_full = paged_attn_triton_prefill(q, k, v, kc, vc, bt, sl0, causal = True, softmax_scale = D ** -0.5, window_size = w_, v_dim = 192)
    o_v128 = paged_attn_triton_prefill(q, k, v, kc, vc, bt, sl0, causal = True, softmax_scale = D ** -0.5, window_size = w_, v_dim = 128)
    d = (o_full[..., :128].float() - o_v128[..., :128].float()).abs().max().item()
    print(f"past {past}: v_dim 128 vs 192 on lanes 0..127: max abs diff {d:.3e} (bit-exact {d == 0.0})", flush = True)
    for cfg_vd in [(c, vd) for vd in args.vdim for c in args.cfg]:
        cfg, vd = cfg_vd
        kw = {}
        if cfg != "default":
            bm, bn, w, s = map(int, cfg.split(","))
            kw = dict(block_m = bm, block_n = bn, num_warps = w, num_stages = s)
        def f():
            sl = torch.tensor([past], dtype = torch.int32, device = DEV)
            return paged_attn_triton_prefill(q, k, v, kc, vc, bt, sl, causal = True,
                                             softmax_scale = D ** -0.5,
                                             window_size = args.window if args.window >= 0 else None,
                                             v_dim = vd, **kw)
        try:
            for _ in range(2): f()
            ts = []
            for _ in range(5):
                e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
                e0.record(); f(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1))
            t = statistics.median(ts)
            flops = 2 * H * args.chunk * (past + args.chunk / 2) * (D + 128) if args.window < 0 else 0
            print(f"past {past:6d} chunk {args.chunk} vdim {vd} cfg {cfg:12s}: {t:8.2f} ms  {flops / t / 1e9:6.1f} TFLOP/s (useful, v=128)", flush = True)
        except Exception as e:
            print(f"past {past} cfg {cfg}: FAIL {type(e).__name__}: {str(e)[:100]}", flush = True)
