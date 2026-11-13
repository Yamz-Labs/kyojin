"""Shared expert fused into the grouped MoE decode launches (ext.fq_moe_kq_grouped_mixed_sh_q8) vs an fp32 reference on the
dequantized weights: 8 routed experts (gate / up Q4_K, down Q8_0 id 6 = v1 down kernel, or Q4_K id 0 = v2 down kernel at I 768),
plus a Q8_0 shared expert of the same intermediate size, with and without a sigmoid gate row; R 1 / 2. Also checks that the
routed part equals the plain grouped call (the shared slots must not disturb it). BOLT_FQ_MOE_V2=0 covers the v1 upgate kernel.
Run in the tree: [BOLT_FQ_MOE_V2=0] python test_moe_sh_fused.py"""
import numpy as np
import torch
import torch.nn.functional as F
from exllamav3.foreign_quant import gap_formats
from exllamav3.foreign_quant.k_quants import dequant_q4_K_bulk

E, K, topk = 8, 2560, 2


def f16bytes(a):
    return np.frombuffer(np.asarray(a, np.float16).tobytes(), np.uint8)


def rand_q4k(rng, N, Kc):
    nb = Kc // 256
    b = rng.integers(0, 256, (N, nb, 144), dtype=np.uint8)
    b[:, :, 0:2] = f16bytes(np.full(N * nb, 0.0004)).reshape(N, nb, 2)
    b[:, :, 2:4] = f16bytes(np.full(N * nb, 0.0002)).reshape(N, nb, 2)
    deq = dequant_q4_K_bulk(b.reshape(N * nb, 144)).reshape(N, Kc).astype(np.float32)
    return b.reshape(N, nb * 144), deq


def rand_q8(rng, N, Kc, d = 0.003):
    nb = Kc // 32
    b = rng.integers(0, 256, (N, nb, 34), dtype=np.uint8)
    b[:, :, 0:2] = f16bytes(np.full(N * nb, d)).reshape(N, nb, 2)
    t = torch.from_numpy(b.reshape(N, nb * 34)).cuda()
    return b.reshape(N, nb * 34), gap_formats.dequant("q8_0", t).float().cpu().numpy()


def act(hf, wg, wu):
    return (F.silu(torch.tensor(hf @ wg.T)) * torch.tensor(hf @ wu.T)).half().float().numpy()


def main():
    global ext
    from exllamav3.ext import exllamav3_ext as ext
    bad = 0
    for fmt_d, fid, I in (("q8_0", 6, 640), ("q4_k", 0, 768)):
        rng = np.random.default_rng(fid + 11)
        gates, ups, downs, Wg, Wu, Wd = [], [], [], [], [], []
        for e in range(E):
            g, gd = rand_q4k(rng, I, K); u, ud = rand_q4k(rng, I, K)
            d, dd = rand_q4k(rng, K, I) if fmt_d == "q4_k" else rand_q8(rng, K, I)
            gates.append(torch.from_numpy(g).cuda()); ups.append(torch.from_numpy(u).cuda()); downs.append(torch.from_numpy(d).cuda())
            Wg.append(gd); Wu.append(ud); Wd.append(dd)
        sg, sgd = rand_q8(rng, I, K, 0.0015); su, sud = rand_q8(rng, I, K, 0.0015); sd, sdd = rand_q8(rng, K, I)
        shg, shu, shd = (torch.from_numpy(a).cuda().reshape(-1).contiguous() for a in (sg, su, sd))
        gp = torch.tensor([t.data_ptr() for t in gates], dtype=torch.int64, device="cuda")
        up = torch.tensor([t.data_ptr() for t in ups], dtype=torch.int64, device="cuda")
        dp = torch.tensor([t.data_ptr() for t in downs], dtype=torch.int64, device="cuda")
        gs = (K // 256) * 144
        ds = downs[0].shape[1]
        for R in (1, 2):
            for gated in (True, False):
                h = (torch.randn(R, K) * 0.5).half().cuda()
                sel = torch.from_numpy(np.stack([rng.choice(E, topk, replace=False) for _ in range(R)])).long().cuda().reshape(-1)
                wts = torch.rand(R * topk).half().cuda()
                wrow = torch.from_numpy(rng.standard_normal(K).astype(np.float32) * 0.02).cuda() if gated else \
                    torch.empty(0, dtype=torch.float, device="cuda")
                plain = torch.zeros(R, K, dtype=torch.float, device="cuda")
                ext.fq_moe_kq_grouped_mixed_q8(h, sel, wts, gp, up, dp, gs, gs, ds, 0, fid,
                                               torch.zeros(R * topk, K, device="cuda"), plain)
                fused = torch.zeros(R, K, dtype=torch.float, device="cuda")
                ext.fq_moe_kq_grouped_mixed_sh_q8(h, sel, wts, gp, up, dp, gs, gs, ds, 0, fid,
                                                  torch.zeros(R * (topk + 1), K, device="cuda"), fused, shg, shu, shd, wrow)
                torch.cuda.synchronize()
                hf = h.float().cpu().numpy()
                ref_sh = np.zeros((R, K), np.float32)
                for r in range(R):
                    gsc = 1.0 / (1.0 + np.exp(-float(hf[r] @ wrow.cpu().numpy()))) if gated else 1.0
                    ref_sh[r] = gsc * (act(hf[r], sgd, sud) @ sdd.T)
                got_sh = (fused - plain).cpu().numpy()        # the shared part the fused call added
                rel = float(np.abs(got_sh - ref_sh).max() / (np.abs(ref_sh).max() + 1e-9))
                l2 = float(np.linalg.norm(got_sh - ref_sh) / (np.linalg.norm(ref_sh) + 1e-9))
                ok = l2 < 3e-2 and rel < 6e-2 and np.isfinite(fused.cpu().numpy()).all() and np.abs(ref_sh).max() > 0
                bad += not ok
                print(f"down {fmt_d} I {I} R {R} gate {gated}: shared part max rel {rel:.2e} L2 rel {l2:.2e} "
                      f"(|shared| max {np.abs(ref_sh).max():.3f}, |routed| max {plain.abs().max().item():.3f}) {'ok' if ok else 'BAD'}")
    print("PASS" if bad == 0 else f"FAIL ({bad})")
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
