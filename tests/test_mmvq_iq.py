"""llama.cpp mmvq-style IQ decode kernels (fq_mmvq.cu: fq_mmvq_<fmt> dense GEMV, fq_moe_mmvq_grouped MoE) on synthetic blocks of all 7 IQ formats.
Reference = dequantized weights (fp32) times the q8_1-DEQUANTIZED activations (what the integer dots compute), so the only remaining difference
is the float summation order: a correct port agrees to ~1e-5 relative, a wrong grid / sign / scale / index bit gives errors of order 1.
Random bytes are valid blocks for every IQ format (only the half scale d is set to a sane value).
Usage: python tests/test_mmvq_iq.py"""
import sys

import torch

import exllamav3_ext as ext
from exllamav3.foreign_quant import gap_formats

FMTS = {  # name: (block bytes, elements per block, mmvq id)
    "iq2_xxs": (66, 256, 0), "iq2_xs": (74, 256, 1), "iq2_s": (82, 256, 2), "iq3_xxs": (98, 256, 3),
    "iq3_s": (110, 256, 4), "iq4_nl": (18, 32, 5), "iq4_xs": (136, 256, 6),
}


def make_weights(fmt, n, k, seed, dscale=1.0):
    bb, qk, _ = FMTS[fmt]
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (n, k // qk, bb), generator=g, dtype=torch.uint8)
    d = ((torch.rand(n, k // qk, generator=g) * 0.02 + 0.001) * dscale).half().view(torch.uint8).reshape(n, k // qk, 2)
    w[:, :, 0:2] = d
    return w.reshape(n, -1).cuda()


def dequant(fmt, w, n, k):
    bb, qk, _ = FMTS[fmt]
    if fmt == "iq2_xxs":
        out = torch.empty((n, k), dtype=torch.half, device="cuda")
        ext.fq_dequant_iq2_xxs(w.reshape(n, k // 256, 66), out)
        return out.float()
    if fmt == "iq4_xs":
        out = torch.empty((n, k), dtype=torch.half, device="cuda")
        ext.fq_dequant_iq4_xs(w, out)
        return out.float()
    return gap_formats.dequant(fmt, w).float().reshape(n, k)


def q8_dequant(qb, m, k):
    b = qb.reshape(m, k // 32, 36)
    d = b[:, :, 0:2].contiguous().view(torch.half).float().reshape(m, k // 32, 1)
    q = b[:, :, 4:].contiguous().view(torch.int8).float().reshape(m, k // 32, 32)
    return (d * q).reshape(m, k)


OLD = {"iq2_xxs": "fq_gemv_iq2_xxs_q8", "iq3_xxs": "fq_gemv_iq3_xxs_q8", "iq2_s": "fq_gemv_iq2_s_q8", "iq3_s": "fq_gemv_iq3_s_q8",
       "iq4_xs": "fq_gemv_iq4_xs_q8"}
worst_old = [0.0]


def test_dense():
    worst = 0.0
    for fmt, (bb, qk, _) in FMTS.items():
        for n, k in ((1024, 4096), (777, 2048), (512, 512)):
            w = make_weights(fmt, n, k, 7)
            wd = dequant(fmt, w, n, k)
            for m in (1, 2, 3, 4, 5, 7, 8, 9, 16):
                x = (torch.randn(m, k, generator=torch.Generator().manual_seed(m)) * 0.5).half().cuda()
                qb = ext.fq_q8_scratch(torch.cuda.current_device(), m, k)
                ext.fq_quant_q8_1_rows(x, qb)
                ref = q8_dequant(qb, m, k) @ wd.t()
                for dt in (torch.half, torch.float):
                    out = torch.empty((m, n), dtype=dt, device="cuda")
                    getattr(ext, f"fq_mmvq_{fmt}")(qb, w, out)
                    torch.cuda.synchronize()
                    rel = float((out.float() - ref).norm() / ref.norm())
                    tol = 3e-3 if dt == torch.half else 1e-3      # the fp16 dequant of the reference alone is ~3e-4
                    assert rel < tol, f"{fmt} [{n}x{k}] m={m} {dt}: rel err {rel:.6f}"
                    worst = max(worst, rel if dt == torch.float else 0.0)
                    old = OLD.get(fmt)
                    if old is not None and dt == torch.float:
                        # same integer dots as the previous kernels: only the float summation order differs
                        o = torch.empty((m, n), dtype=torch.float, device="cuda")
                        getattr(ext, old)(qb, w.reshape(n, k // 256, 66) if fmt == "iq2_xxs" else w, o)
                        torch.cuda.synchronize()
                        d = float((out - o).norm() / o.norm())
                        assert d < 2e-5, f"{fmt} [{n}x{k}] m={m}: differs from {old} by {d:.2e}"
                        worst_old[0] = max(worst_old[0], d)
        print(f"dense {fmt}: ok")
    print(f"dense worst fp32-out rel err vs dequant reference {worst:.2e}, vs previous q8 kernels {worst_old[0]:.2e}")


def moe_case(fug, fd, K, I, E, R, topk, seed):
    bbu, qku, idu = FMTS[fug]
    bbd, qkd, idd = FMTS[fd]
    wg = [make_weights(fug, I, K, seed + e, 0.1) for e in range(E)]
    wu = [make_weights(fug, I, K, seed + 100 + e, 0.1) for e in range(E)]
    wd = [make_weights(fd, K, I, seed + 200 + e, 0.1) for e in range(E)]
    Gd = [dequant(fug, w, I, K) for w in wg]; Ud = [dequant(fug, w, I, K) for w in wu]; Dd = [dequant(fd, w, K, I) for w in wd]
    ptrs = lambda ws: torch.tensor([w.data_ptr() for w in ws], dtype=torch.int64, device="cuda")
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(R, K, generator=g) * 0.5).half().cuda()
    sel = torch.stack([torch.randperm(E, generator=g)[:topk] for _ in range(R)]).cuda()
    wts = torch.rand(R, topk, generator=g).half().cuda()
    qb = ext.fq_q8_scratch(torch.cuda.current_device(), R, K).clone()
    ext.fq_quant_q8_1_rows(x, qb)
    xd = q8_dequant(qb, R, K)
    ref = torch.zeros(R, K, device="cuda")
    for r in range(R):
        for j in range(topk):
            e = int(sel[r, j])
            gg = (xd[r] @ Gd[e].t()).half().float(); uu = (xd[r] @ Ud[e].t()).half().float()
            act = (torch.sigmoid(gg) * gg * uu).half()
            # the down dots see the q8_1 quantization of the half activations
            aq = ext.fq_q8_scratch(torch.cuda.current_device(), 1, I).clone()
            ext.fq_quant_q8_1_rows(act.reshape(1, I).contiguous(), aq)
            ref[r] += float(wts[r, j]) * (q8_dequant(aq, 1, I)[0] @ Dd[e].t())
    partial = torch.empty(R * topk, K, dtype=torch.float, device="cuda")
    out = torch.empty(R, K, dtype=torch.float, device="cuda")
    ext.fq_moe_mmvq_grouped(x, sel.reshape(-1).contiguous(), wts.reshape(-1).contiguous(), ptrs(wg), ptrs(wu), ptrs(wd),
                            wg[0].shape[1], wu[0].shape[1], wd[0].shape[1], idu, idd, partial, out)
    torch.cuda.synchronize()
    OLDID = {"iq4_xs": 3, "iq3_xxs": 4, "iq2_s": 5}
    if fug in OLDID and fd in OLDID:
        part2 = torch.empty_like(partial); out2 = torch.empty_like(out)
        ext.fq_moe_kq_grouped_mixed_q8(x, sel.reshape(-1).contiguous(), wts.reshape(-1).contiguous(), ptrs(wg), ptrs(wu), ptrs(wd),
                                       wg[0].shape[1], wu[0].shape[1], wd[0].shape[1], OLDID[fug], OLDID[fd], part2, out2)
        torch.cuda.synchronize()
        d = float((out - out2).norm() / out2.norm())
        print(f"   vs fq_moe_kq_grouped_mixed_q8: {d:.2e}")
        assert d < 5e-3, f"differs from the previous MoE kernel by {d}"
    return float((out - ref).norm() / ref.norm())


def test_moe():
    cases = [("iq2_s", "iq3_xxs"), ("iq2_s", "iq4_xs"), ("iq3_xxs", "iq3_xxs"), ("iq2_xs", "iq3_s"), ("iq4_nl", "iq4_nl"), ("iq2_xxs", "iq2_xxs"),
             ("iq3_s", "iq2_xs"), ("iq4_xs", "iq4_xs")]
    for fug, fd in cases:
        for R in (1, 4, 8):
            rel = moe_case(fug, fd, 1024, 512, 8, R, 4, 3)
            print(f"moe ug={fug} down={fd} R={R}: rel err {rel:.2e}")
            assert rel < 5e-3, f"{fug}/{fd} R={R}: rel err {rel}"


if __name__ == "__main__":
    test_dense()
    test_moe()
    print("PASS")
