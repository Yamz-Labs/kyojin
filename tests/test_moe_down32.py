"""Grouped MoE decode kernel with a 32-block down (ids 6 = q8_0, 7 = q5_1) vs an fp32 reference on the dequantized weights:
8 experts, K 2560, I 640, top-2, R 1 / 2, gate / up Q4_K. Tolerance covers the q8_1 activation quantization (same as the other
formats on this path). Also: q4_k down (id 0) at I 768 still works (regression guard). Run in the tree: python test_moe_down32.py"""
import numpy as np
import torch
import torch.nn.functional as F
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.foreign_quant import gap_formats
from exllamav3.foreign_quant.k_quants import dequant_q4_K_bulk

rng = np.random.default_rng(0)
E, K, topk = 8, 2560, 2


def f16bytes(a):
    return np.frombuffer(np.asarray(a, np.float16).tobytes(), np.uint8)


def rand_q4k(N, Kc):
    nb = Kc // 256
    b = rng.integers(0, 256, (N, nb, 144), dtype=np.uint8)
    b[:, :, 0:2] = f16bytes(np.full(N * nb, 0.0004)).reshape(N, nb, 2)
    b[:, :, 2:4] = f16bytes(np.full(N * nb, 0.0002)).reshape(N, nb, 2)
    deq = dequant_q4_K_bulk(b.reshape(N * nb, 144)).reshape(N, Kc).astype(np.float32)
    return b.reshape(N, nb * 144), deq


def rand_32(fmt, N, Kc):
    bb = {"q8_0": 34, "q5_1": 24}[fmt]
    nb = Kc // 32
    b = rng.integers(0, 256, (N, nb, bb), dtype=np.uint8)
    b[:, :, 0:2] = f16bytes(np.full(N * nb, 0.003)).reshape(N, nb, 2)
    if fmt == "q5_1":
        b[:, :, 2:4] = f16bytes(rng.standard_normal(N * nb) * 0.02).reshape(N, nb, 2)
    t = torch.from_numpy(b.reshape(N, nb * bb)).cuda()
    deq = gap_formats.dequant(fmt, t).float().cpu().numpy()
    return b.reshape(N, nb * bb), deq


bad = 0
for fmt_d, fid, I in (("q8_0", 6, 640), ("q5_1", 7, 640), ("q4_k", 0, 768)):
    rng = np.random.default_rng(fid + 1)      # fixed data per format (deterministic verdict)
    gates, ups, downs, Wg, Wu, Wd = [], [], [], [], [], []
    for e in range(E):
        g, gd = rand_q4k(I, K); u, ud = rand_q4k(I, K)
        d, dd = (rand_q4k(K, I) if fmt_d == "q4_k" else rand_32(fmt_d, K, I))
        gates.append(torch.from_numpy(g).cuda()); ups.append(torch.from_numpy(u).cuda()); downs.append(torch.from_numpy(d).cuda())
        Wg.append(gd); Wu.append(ud); Wd.append(dd)
    gp = torch.tensor([t.data_ptr() for t in gates], dtype=torch.int64, device="cuda")
    up = torch.tensor([t.data_ptr() for t in ups], dtype=torch.int64, device="cuda")
    dp = torch.tensor([t.data_ptr() for t in downs], dtype=torch.int64, device="cuda")
    gs = (K // 256) * 144
    ds = downs[0].shape[1]
    for R in (1, 2):
        h = (torch.randn(R, K) * 0.5).half().cuda()
        sel = torch.from_numpy(rng.choice(E, (R, topk), replace=False) if R == 1 else
                               np.stack([rng.choice(E, topk, replace=False) for _ in range(R)])).long().cuda().reshape(-1)
        wts = torch.rand(R * topk).half().cuda()
        partial = torch.zeros(R * topk, K, dtype=torch.float, device="cuda")
        routed = torch.zeros(R, K, dtype=torch.float, device="cuda")
        ext.fq_moe_kq_grouped_mixed_q8(h, sel, wts, gp, up, dp, gs, gs, ds, 0, fid, partial, routed)
        torch.cuda.synchronize()
        hf = h.float().cpu().numpy()
        ref = np.zeros((R, K), np.float32)
        for r in range(R):
            for j in range(topk):
                e = int(sel[r * topk + j]); w = float(wts[r * topk + j])
                gg = hf[r] @ Wg[e].T; uu = hf[r] @ Wu[e].T
                a = (F.silu(torch.tensor(gg)) * torch.tensor(uu)).half().float().numpy()
                ref[r] += w * (a @ Wd[e].T)
        got = routed.cpu().numpy()
        amax = max(float(np.abs((F.silu(torch.tensor(hf[r] @ Wg[int(sel[r * topk + j])].T)) * torch.tensor(hf[r] @ Wu[int(sel[r * topk + j])].T)).numpy()).max()) for r in range(R) for j in range(topk))
        rel = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9))
        l2 = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-9))
        ok = l2 < 3e-2 and rel < 5e-2 and np.isfinite(got).all()
        bad += not ok
        print(f"down {fmt_d} (id {fid}) I {I} R {R}: max rel err {rel:.2e} L2 rel {l2:.2e} (act max {amax:.1f}) {'ok' if ok else 'BAD'}")
print("PASS" if bad == 0 else f"FAIL ({bad})")
