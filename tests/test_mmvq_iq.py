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
    "iq3_s": (110, 256, 4), "iq4_nl": (18, 32, 5), "iq4_xs": (136, 256, 6), "q5_k": (176, 256, 7),
}
# dense-only K-quant entries (llama.cpp vec_dot_q4_K / q6_K_q8_1 in fq_mmvq.cu; no grouped MoE id)
KQ_DENSE = {"q4_k": (144, 256, 8), "q6_k": (210, 256, 9), "q3_k": (110, 256, 10)}
# few-row MMQ formats (fq_mmq_few.cu, llama.cpp PR #31533 port): name -> (block bytes, elements per block, fq_mmq_few fmt id)
MMQ_FEW = {"q8_0": (34, 32, 0), "q4_k": (144, 256, 1), "q5_k": (176, 256, 2), "q6_k": (210, 256, 3), "q3_k": (110, 256, 4), "iq4_xs": (136, 256, 5)}


def make_weights(fmt, n, k, seed, dscale=1.0):
    bb, qk, _ = {**FMTS, **KQ_DENSE, **MMQ_FEW}[fmt]
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (n, k // qk, bb), generator=g, dtype=torch.uint8)
    d = ((torch.rand(n, k // qk, generator=g) * 0.02 + 0.001) * dscale).half().view(torch.uint8).reshape(n, k // qk, 2)
    if fmt in ("q6_k", "q3_k"):           # d is the last field of block_q6_K / block_q3_K
        w[:, :, bb - 2:bb] = d
    else:
        w[:, :, 0:2] = d
    if fmt in ("q5_k", "q4_k"):           # half2 (d, dmin)
        w[:, :, 2:4] = d
    return w.reshape(n, -1).cuda()


def dequant(fmt, w, n, k):
    bb, qk, _ = {**FMTS, **KQ_DENSE, **MMQ_FEW}[fmt]
    if fmt == "q8_0":
        out = torch.empty((n, k), dtype=torch.half, device="cuda")
        ext.fq_dequant_q8_0(w.reshape(n, k // 32, 34), out)
        return out.float()
    if fmt == "iq2_xxs":
        out = torch.empty((n, k), dtype=torch.half, device="cuda")
        ext.fq_dequant_iq2_xxs(w.reshape(n, k // 256, 66), out)
        return out.float()
    if fmt in ("iq4_xs", "q5_k", "q4_k", "q6_k"):
        out = torch.empty((n, k), dtype=torch.half, device="cuda")
        getattr(ext, f"fq_dequant_{fmt}")(w, out)
        return out.float()
    return gap_formats.dequant(fmt, w).float().reshape(n, k)   # q3_k: the torch gap reference


def q8_dequant(qb, m, k):
    b = qb.reshape(m, k // 32, 36)
    d = b[:, :, 0:2].contiguous().view(torch.half).float().reshape(m, k // 32, 1)
    q = b[:, :, 4:].contiguous().view(torch.int8).float().reshape(m, k // 32, 32)
    return (d * q).reshape(m, k)


OLD = {"iq2_xxs": "fq_gemv_iq2_xxs_q8", "iq3_xxs": "fq_gemv_iq3_xxs_q8", "iq2_s": "fq_gemv_iq2_s_q8", "iq3_s": "fq_gemv_iq3_s_q8",
       "iq4_xs": "fq_gemv_iq4_xs_q8", "q5_k": "fq_gemv_q5_k_q8", "q4_k": "fq_gemv_q4_k_q8", "q6_k": "fq_gemv_q6_k_q8"}
worst_old = [0.0]


def test_dense():
    worst = 0.0
    for fmt, (bb, qk, _) in {**FMTS, **KQ_DENSE}.items():
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
        if fmt == "q3_k":                 # vs the Triton gap kernel (fp16 activations, so the q8_1 quantization error shows: ~1e-2 relative)
            from exllamav3.foreign_quant import triton_iq
            n, k, m = 1024, 4096, 4
            w = make_weights(fmt, n, k, 7)
            x = (torch.randn(m, k, generator=torch.Generator().manual_seed(m)) * 0.5).half().cuda()
            qb = ext.fq_q8_scratch(torch.cuda.current_device(), m, k)
            ext.fq_quant_q8_1_rows(x, qb)
            o = torch.empty((m, n), dtype=torch.float, device="cuda")
            ext.fq_mmvq_q3_k(qb, w, o)
            t = triton_iq.gemv("q3_k", x, w.reshape(n, -1)).float()
            d = float((o - t).norm() / t.norm())
            print(f"dense q3_k vs the Triton gap kernel: {d:.2e}")
            assert d < 1e-2, f"q3_k differs from the Triton gap kernel by {d}"
        print(f"dense {fmt}: ok")
    print(f"dense worst fp32-out rel err vs dequant reference {worst:.2e}, vs previous q8 kernels {worst_old[0]:.2e}")


def test_dequant():
    for fmt, (bb, qk, i) in FMTS.items():
        if fmt == "q5_k":
            continue
        n, k = 96, 1024
        w = make_weights(fmt, n, k, 11)
        ref = dequant(fmt, w, n, k).half()
        out = torch.empty((n, k), dtype=torch.half, device="cuda")
        ext.fq_dequant_iq(w, out, i)
        torch.cuda.synchronize()
        bad = int((out.view(torch.int16) != ref.view(torch.int16)).sum())
        print(f"dequant {fmt}: {bad} of {n * k} fp16 values differ from the reference")
        assert bad == 0, f"{fmt}: fq_dequant_iq differs from the reference"
    n, k = 96, 1024
    w = make_weights("q3_k", n, k, 11)
    ref = gap_formats.dequant("q3_k", w).half().reshape(n, k)
    out = torch.empty((n, k), dtype=torch.half, device="cuda")
    ext.fq_dequant_iq(w, out, 10)
    torch.cuda.synchronize()
    bad = int((out.view(torch.int16) != ref.view(torch.int16)).sum())
    print(f"dequant q3_k: {bad} of {n * k} fp16 values differ from the torch gap reference")
    assert bad == 0, "q3_k: fq_dequant_iq differs from the reference"


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


def few_q8(x, ds4):
    """llama.cpp quantize_mmq_few_q8_1 in torch: q = roundf(x * 127/amax) per 32 values, d = 1/(127/amax), stored as half for DS4 (Q4_K / Q5_K)."""
    m, k = x.shape
    xb = x.float().reshape(m, k // 32, 32)
    d_inv = 127.0 / xb.abs().amax(-1, keepdim=True)
    v = xb * d_inv
    q = torch.nan_to_num(torch.sign(v) * torch.floor(v.abs() + 0.5))
    d = 1.0 / d_inv
    if ds4:
        d = d.half().float()
    return (q * d).reshape(m, k)


def test_mmq_few():
    """fq_mmq_few vs (1) fp32 dequantized weights x the same q8_1 activations emulated in torch: only the float summation order and, for
    Q4_K / Q5_K, the half (d * sum) of the min term differ -> rel < 2e-3; (2) the existing decode kernels (fq_mmvq_<fmt>, fq_gemv_q8_0 for
    Q8_0), which quantize the activations with d = amax/127 stored as half -> a few q values off by one, rel < 5e-3."""
    if not hasattr(ext, "fq_mmq_few") or torch.cuda.get_device_capability()[0] < 8:
        print("mmq_few: skipped (needs a CUDA sm_80+ build)")
        return
    worst_ref = worst_old = 0.0
    for fmt, (bb, qk, fid) in MMQ_FEW.items():
        for n, k in ((1024, 4096), (777, 2048), (3000, 4096)):   # 777: partial row tile; 3000 >= 82 SMs * 32: the 32-row tiles (9..16 rows)
            w = make_weights(fmt, n, k, 5)
            wd = dequant(fmt, w, n, k)
            for m in range(2, 25):
                x = (torch.randn(m, k, generator=torch.Generator().manual_seed(m)) * 0.5).half().cuda()
                ref = few_q8(x, fmt in ("q4_k", "q5_k")) @ wd.t()
                out = torch.empty((m, n), dtype=torch.float, device="cuda")
                scratch = ext.fq_q8_scratch(torch.cuda.current_device(), 8 * ((m + 7) // 8), k)
                assert ext.fq_mmq_few(x, scratch, w, out, fid), f"{fmt} [{n}x{k}] m={m}: not served"
                torch.cuda.synchronize()
                rel = float((out - ref).norm() / ref.norm())
                assert rel < 2e-3, f"{fmt} [{n}x{k}] m={m}: rel err {rel:.2e} vs dequant reference"
                worst_ref = max(worst_ref, rel)
                o = torch.empty((m, n), dtype=torch.float, device="cuda")
                if fmt == "q8_0":
                    ext.fq_gemv_q8_0(x, w.reshape(n, -1, 34), o)
                else:
                    qb = ext.fq_q8_scratch(torch.cuda.current_device(), m, k)
                    ext.fq_quant_q8_1_rows(x, qb)
                    getattr(ext, f"fq_mmvq_{fmt}")(qb, w, o)
                torch.cuda.synchronize()
                d = float((out - o).norm() / o.norm())
                assert d < 5e-3, f"{fmt} [{n}x{k}] m={m}: differs from the existing kernel by {d:.2e}"
                worst_old = max(worst_old, d)
                oh = torch.empty((m, n), dtype=torch.half, device="cuda")
                ext.fq_mmq_few(x, scratch, w, oh, fid)
                torch.cuda.synchronize()
                assert torch.equal(oh, out.half()), f"{fmt} m={m}: half output != rounded float output"
        print(f"mmq_few {fmt}: ok")
    print(f"mmq_few worst rel err vs dequant reference {worst_ref:.2e}, vs existing kernels {worst_old:.2e}")


def test_moe():
    cases = [("iq2_s", "iq3_xxs"), ("iq2_s", "iq4_xs"), ("iq3_xxs", "iq3_xxs"), ("iq2_xs", "iq3_s"), ("iq4_nl", "iq4_nl"), ("iq2_xxs", "iq2_xxs"),
             ("iq3_s", "iq2_xs"), ("iq4_xs", "iq4_xs")]
    for fug, fd in cases:
        for R in (1, 4, 8, 40):
            rel = moe_case(fug, fd, 1024, 512, 8, R, 4, 3)
            print(f"moe ug={fug} down={fd} R={R}: rel err {rel:.2e}")
            assert rel < 5e-3, f"{fug}/{fd} R={R}: rel err {rel}"


if __name__ == "__main__":
    test_dequant()
    test_dense()
    test_mmq_few()
    test_moe()
    print("PASS")
