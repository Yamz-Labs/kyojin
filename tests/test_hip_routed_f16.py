"""Routed (MoE) HIP F16-WMMA prefill op (ext.fq_routed_f16_moe, rocm/routed_f16: Gufo's RoutedPairF16 / RoutedDownF16 kernels).

Random raw GGUF blocks at the Qwen3.8-Flash-Next shapes (gate/up Q4_K / Q5_K K = 2560, I = 640; down Q5_1 / Q8_0 K = 640 -> 2560),
one allocation per expert, random top-k ids, ragged buckets (T = 7, 128, 600, 4096). Reference: Kyojin's dequant then an fp32
silu(x Wg^T) * (x Wu^T) -> Wd^T chain; the gmmq q8_1 path (ext.fq_mmq_moe) is compared to the same reference and to this op.
Gate: max|err| / max|ref| < 1e-2 (the op rounds weights, activations and the SwiGLU product to F16).
Runs under pytest or as a plain script (ROCm gfx1151 only; skips elsewhere)."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_hip_gmmq import FMTS, PAD, dequant, make_raw  # noqa: E402

TOL = 1e-2
IDS = {"q4_k": 12, "q5_k": 13, "q5_1": 7, "q8_0": 8}


def _ext():
    if not torch.cuda.is_available() or not torch.version.hip or \
            "gfx1151" not in torch.cuda.get_device_properties(0).gcnArchName:
        return None
    try:
        import exllamav3_ext as ext
    except ImportError:
        return None
    return ext if hasattr(ext, "fq_routed_f16_moe") else None


def make_experts(fmt, E, g, dev):
    _, qk, bb, K, M = FMTS[fmt]
    raw = make_raw(fmt, E, M, K, g)
    bufs, views = [], []
    for e in range(E):
        n = raw[e].numel()
        b = torch.zeros(n + PAD, dtype=torch.uint8, device=dev)
        b[:n] = raw[e].reshape(-1).to(dev)
        bufs.append(b)
        views.append(b[:n].view(M, -1))
    ptrs = torch.tensor([v.data_ptr() for v in views], dtype=torch.int64, device=dev)
    return bufs, views, ptrs


def tiles_for(ids, E):
    dev = ids.device
    counts = torch.bincount(ids.view(-1), minlength=E)
    nt = (counts + 127) // 128
    total = int(nt.sum())
    first = torch.cumsum(nt, 0) - nt
    eidx = torch.repeat_interleave(torch.arange(E, device=dev), nt, output_size=total)
    return counts.to(torch.int32), (eidx | ((torch.arange(total, device=dev) - first[eidx]) << 16)).to(torch.int32)


def run_case(ext, gfmt, dfmt, E, T, n_used, seed, timing=False):
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(seed)
    _, _, _, K, I = FMTS[gfmt]
    gb, gv, gp = make_experts(gfmt, E, g, dev)
    ub, uv, up = make_experts(gfmt, E, g, dev)
    db, dv, dp = make_experts(dfmt, E, g, dev)
    x = (torch.randn(T, K, generator=g) * 0.03).half().to(dev)
    ids = torch.stack([torch.randperm(E, generator=g)[:n_used] for _ in range(T)]).to(torch.int32).to(dev).contiguous()
    counts, tiles = tiles_for(ids, E)
    out = torch.full((T * n_used, K), float("nan"), dtype=torch.half, device=dev)
    run = lambda: ext.fq_routed_f16_moe(x, ids, counts, tiles, gp, up, dp, IDS[gfmt], IDS[dfmt], I, out)
    run()
    torch.cuda.synchronize()
    # fp32 reference from dequantized weights
    xr = x.float()
    ref = torch.zeros(T * n_used, K, dtype=torch.float, device=dev)
    flat = ids.reshape(-1)
    for e in flat.unique().tolist():
        rows = (flat == e).nonzero().reshape(-1)
        xt = xr[rows // n_used]
        a = torch.nn.functional.silu(xt @ dequant(ext, gfmt, gv[e]).T) * (xt @ dequant(ext, gfmt, uv[e]).T)
        ref[rows] = a @ dequant(ext, dfmt, dv[e]).T
    scale = ref.abs().max().item()
    o = out.float()
    finite = bool(torch.isfinite(o).all().item())
    rel = (o - ref).abs().max().item() / scale
    # the existing gmmq path (q8_1 activations)
    xf = x.float()
    gg = torch.empty(T * n_used, I, dtype=torch.float, device=dev)
    uu = torch.empty_like(gg)
    ext.fq_mmq_moe(xf, ids, gp, IDS[gfmt], I, gg)
    ext.fq_mmq_moe(xf, ids, up, IDS[gfmt], I, uu)
    a = (torch.nn.functional.silu(gg) * uu).to(torch.half)
    dn = torch.empty(T * n_used, K, dtype=torch.float, device=dev)
    ext.fq_mmq_moe(a, ids.view(-1, 1), dp, IDS[dfmt], K, dn)
    rel_gm = (dn - ref).abs().max().item() / scale
    rel_vs_gm = (o - dn).abs().max().item() / scale
    ok = finite and rel < TOL
    msg = (f"{gfmt}/{dfmt} E={E:3d} T={T:4d} top={n_used:2d}: f16_vs_ref={rel:.3e} gmmq_vs_ref={rel_gm:.3e} "
           f"f16_vs_gmmq={rel_vs_gm:.3e} finite={finite} -> {'PASS' if ok else 'FAIL'}")
    if timing:
        def bench(fn):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / 5 * 1e3
        def gm():
            ext.fq_mmq_moe(xf, ids, gp, IDS[gfmt], I, gg)
            ext.fq_mmq_moe(xf, ids, up, IDS[gfmt], I, uu)
            ext.fq_mmq_moe(a, ids.view(-1, 1), dp, IDS[dfmt], K, dn)
        msg += f"  [f16 {bench(run):.2f} ms, gmmq gate+up+down {bench(gm):.2f} ms]"
    print(msg, flush=True)
    return ok


CASES = [  # (gate fmt, down fmt, E, T, top_k, timing)
    *[("q4_k", "q5_1", 64, T, 10, False) for T in (7, 128, 600, 4096)],
    *[("q5_k", "q8_0", 64, T, 10, False) for T in (7, 600, 4096)],
    ("q4_k", "q8_0", 64, 129, 8, False),
    ("q4_k", "q5_1", 512, 4096, 10, True),
    ("q5_k", "q8_0", 512, 4096, 10, True),
]


def run_all():
    ext = _ext()
    if ext is None:
        print("SKIP: needs a ROCm gfx1151 torch and an exllamav3_ext with fq_routed_f16_moe")
        return None
    fails = 0
    for i, (gf, df, E, T, k, timing) in enumerate(CASES):
        fails += not run_case(ext, gf, df, E, T, k, seed=200 + i, timing=timing)
    print(f"{'PASS' if fails == 0 else 'FAIL'}: {len(CASES) - fails}/{len(CASES)} cases")
    return fails == 0


def test_hip_routed_f16():
    import pytest
    if ext0 is not None and hasattr(ext0, "fq_routed_f16_combine_shared"):
        assert run_combine_shared(ext0)
    res = run_all()
    if res is None:
        pytest.skip("ROCm gfx1151 op (fq_routed_f16_moe) not available")
    assert res


def run_combine(ext):
    """ext.fq_routed_f16_combine against an fp64 reference sum_s w[t, s] * rows[t, s, :] and against torch's F32 multiply + sum (the path it replaces)."""
    dev = torch.device("cuda")
    ok = True
    for T, top_k, K in ((1, 10, 2560), (7, 10, 2560), (600, 10, 2560), (4096, 10, 2560), (33, 8, 640)):
        g = torch.Generator(device="cpu").manual_seed(T * 31 + K)
        rows = (torch.randn(T * top_k, K, generator=g) * 3).half().to(dev)
        w = torch.rand(T, top_k, generator=g).float().to(dev)
        out = torch.empty(T, K, dtype=torch.float, device=dev)
        ext.fq_routed_f16_combine(rows, w, out)
        torch.cuda.synchronize()
        ref = (rows.view(T, top_k, K).double() * w.double().unsqueeze(-1)).sum(1)
        old = (rows.view(T, top_k, K) * w.reshape(T, top_k, 1)).sum(1)
        e_ref = ((out.double() - ref).abs().max() / ref.abs().max()).item()
        e_old = ((out - old).abs().max() / ref.abs().max()).item()
        print(f"combine T={T:5d} top_k={top_k} K={K}: rel err vs fp64 {e_ref:.2e}, vs torch path {e_old:.2e}")
        ok &= e_ref < 1e-6 and e_old < 1e-6
    return ok


def test_hip_routed_f16_combine():
    ext = _ext()
    if ext is None or not hasattr(ext, "fq_routed_f16_combine"):
        return
    assert run_combine(ext)


def run_combine_shared(ext):
    """ext.fq_routed_f16_combine_shared against an fp64 reference sum_s w[t, s] * rows[t, s, :] + shared[t, :] * sigmoid(f16(x[t, :] . gate_w)) and
    against the separate torch path it replaces (combine, F32 widen, gate matmul in half, sigmoid, multiply, add). x is a row-padded view in one case."""
    dev = torch.device("cuda")
    ok = True
    for T, top_k, K, pad in ((1, 10, 2560, 0), (7, 10, 2560, 0), (600, 10, 2560, 64), (4096, 10, 2560, 0), (33, 8, 640, 0)):
        g = torch.Generator(device="cpu").manual_seed(T * 17 + K)
        rows = (torch.randn(T * top_k, K, generator=g) * 3).half().to(dev)
        w = torch.rand(T, top_k, generator=g).float().to(dev)
        sh = torch.randn(T, K, generator=g).half().to(dev)
        xb = torch.randn(T, K + pad, generator=g).half().to(dev)
        x = xb[:, :K]
        gw = (torch.randn(K, 1, generator=g) * 0.05).half().to(dev)
        out = torch.empty(T, K, dtype=torch.float, device=dev)
        ext.fq_routed_f16_combine_shared(rows, w, sh, x, gw.view(-1), out)
        torch.cuda.synchronize()
        gate = (x.double() @ gw.double()).half().double()
        ref = (rows.view(T, top_k, K).double() * w.double().unsqueeze(-1)).sum(1) + sh.double() * torch.sigmoid(gate)
        old = (rows.view(T, top_k, K) * w.reshape(T, top_k, 1)).sum(1)
        old.add_(sh.float() * torch.sigmoid((x @ gw).float()))
        e_ref = ((out.double() - ref).abs().max() / ref.abs().max()).item()
        e_old = ((out - old).abs().max() / ref.abs().max()).item()
        print(f"combine_shared T={T:5d} top_k={top_k} K={K} pad={pad}: rel err vs fp64 {e_ref:.2e}, vs torch path {e_old:.2e}")
        ok &= e_ref < 1e-4 and e_old < 1e-4   # 1e-4: the gate dot is rounded to F16, a one-ulp flip vs the GEMM's summation order moves a few rows by <= ~1e-4 of the max
    return ok


def test_hip_routed_f16_combine_shared():
    ext = _ext()
    if ext is None or not hasattr(ext, "fq_routed_f16_combine_shared"):
        return
    assert run_combine_shared(ext)


if __name__ == "__main__":
    ext0 = _ext()
    if ext0 is not None and hasattr(ext0, "fq_routed_f16_combine"):
        assert run_combine(ext0)
    if ext0 is not None and hasattr(ext0, "fq_routed_f16_combine_shared"):
        assert run_combine_shared(ext0)
    res = run_all()
    sys.exit(0 if res in (True, None) else 1)
