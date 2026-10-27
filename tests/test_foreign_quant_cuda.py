"""GPU tests for the foreign-quant kernels (Q8_0 / IQ2_XXS / NVFP4 -> fp16).

Bit-exactness: ext dequant output (fp16) must equal the numpy reference
(exllamav3/foreign_quant/*.py, float32) cast to fp16, compared as uint16 bits.
Fused path: fq_gemm_* (M = 1..16) vs dequant + torch.matmul within fp16
tolerance. Runs on synthetic payloads always, and on REAL Bolt tensors from
BOLT_MODEL_DIR when it is set.

Standalone script (the CUDA venv has no pytest):
    python tests/test_foreign_quant_cuda.py
Exit nonzero on the first failure.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import torch

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FQ_DIR = os.path.join(ENGINE, "exllamav3", "foreign_quant")
MODEL_DIR = os.environ.get("BOLT_MODEL_DIR", "")
sys.path.insert(0, ENGINE)

if "pytest" in sys.modules and (not torch.cuda.is_available() or torch.version.hip is not None):
    import pytest  # the standalone run (CUDA venv) has no pytest
    pytest.skip("NVIDIA CUDA required", allow_module_level = True)


def _load(name):
    path = os.path.join(FQ_DIR, name + ".py")
    spec = importlib.util.spec_from_file_location(f"fq_{name}_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


q8_0 = _load("q8_0")
iq2_xxs = _load("iq2_xxs")
nvfp4 = _load("nvfp4")

from exllamav3.ext import exllamav3_ext as ext  # noqa: E402

PASS = []
FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    (PASS if cond else FAIL).append(name)


def bits16(a: torch.Tensor) -> np.ndarray:
    return a.detach().to(torch.float16).cpu().contiguous().view(torch.uint16).numpy()


def ref16(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).astype(np.float16).view(np.uint16)


def finite_scales(rng, shape):
    """Random fp16 scales with exponent != 31 (finite: no Inf/NaN)."""
    u = rng.integers(0, 65536, size=shape).astype(np.uint16)
    exp_full = (u & 0x7C00) == 0x7C00
    u[exp_full] &= 0xFBFF  # clear top exponent bit -> exp <= 30
    return u


# --------------------------------------------------------------------------
# Synthetic bit-exact dequant
# --------------------------------------------------------------------------

def test_q8_0_synth():
    rng = np.random.default_rng(0)
    n, kb = 8, 5
    payload = np.empty((n, kb, 34), np.uint8)
    payload[:, :, :2] = finite_scales(rng, (n, kb)).view(np.uint8).reshape(n, kb, 2)
    payload[:, :, 2:] = rng.integers(0, 256, size=(n, kb, 32)).astype(np.uint8)
    ref = ref16(q8_0.dequant_q8_0(payload))
    out = torch.empty((n, kb * 32), dtype=torch.float16, device="cuda")
    ext.fq_dequant_q8_0(torch.from_numpy(payload).cuda(), out)
    torch.cuda.synchronize()
    check("q8_0 synth bit-exact", np.array_equal(bits16(out).ravel(), ref.ravel()),
          f"shape={(n, kb * 32)}")


def test_iq2_xxs_synth():
    rng = np.random.default_rng(2)
    n, kb = 4, 3
    payload = rng.integers(0, 256, size=(n, kb, 66)).astype(np.uint8)
    ref = ref16(iq2_xxs.dequant_iq2_xxs(payload))
    out = torch.empty((n, kb * 256), dtype=torch.float16, device="cuda")
    ext.fq_dequant_iq2_xxs(torch.from_numpy(payload).cuda(), out)
    torch.cuda.synchronize()
    check("iq2_xxs synth bit-exact", np.array_equal(bits16(out).ravel(), ref.ravel()),
          f"shape={(n, kb * 256)}")


def test_nvfp4_synth():
    rng = np.random.default_rng(3)
    n, k = 8, 64
    packed = rng.integers(0, 256, size=(n, k // 2)).astype(np.uint8)
    scales = rng.integers(0, 256, size=(n, k // 16)).astype(np.uint8)
    scales[(scales == 0x7F) | (scales == 0xFF)] = 0x38  # no NaN scales here
    g = np.float32(0.37)
    ref = ref16(nvfp4.dequant_nvfp4(packed, scales, g))
    out = torch.empty((n, k), dtype=torch.float16, device="cuda")
    ext.fq_dequant_nvfp4(torch.from_numpy(packed).cuda(), torch.from_numpy(scales).cuda(),
                         float(g), out)
    torch.cuda.synchronize()
    check("nvfp4 synth bit-exact", np.array_equal(bits16(out).ravel(), ref.ravel()), f"shape={(n, k)}")


def test_nvfp4_e4m3_finite_sweep():
    # Every finite E4M3 code as a scale over a fixed nibble pattern.
    codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
    codes[(codes == 0x7F) | (codes == 0xFF)] = 0x00
    packed = np.full((16, 128), 0x11, np.uint8)  # all nibbles = 1 -> 0.5
    ref = ref16(nvfp4.dequant_nvfp4(packed, codes, 1.0))
    out = torch.empty((16, 256), dtype=torch.float16, device="cuda")
    ext.fq_dequant_nvfp4(torch.from_numpy(packed).cuda(), torch.from_numpy(codes).cuda(), 1.0, out)
    torch.cuda.synchronize()
    check("nvfp4 e4m3 finite-code sweep bit-exact", np.array_equal(bits16(out).ravel(), ref.ravel()))


# --------------------------------------------------------------------------
# Real Bolt tensors (skip gracefully while the download is incomplete)
# --------------------------------------------------------------------------

def _open_shard(fname):
    from safetensors import safe_open
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.isfile(path):
        return None
    return safe_open(path, framework="pt")


def _load_real():
    """Returns dict of (tensor, meta) or None if any needed shard is missing."""
    index_p = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    if not os.path.isfile(index_p):
        return None
    with open(index_p) as f:
        wmap = json.load(f)["weight_map"]
    need = {
        "lm_head": "language_model.lm_head.weight",
        "fc1": "language_model.backbone.layers.1.mixer.fc1_latent_proj.weight",
        "exp_up": "language_model.backbone.layers.1.mixer.experts.0.up_proj.weight",
        "exp_down": "language_model.backbone.layers.1.mixer.experts.0.down_proj.weight",
        "in_w": "language_model.backbone.layers.0.mixer.in_proj.weight",
        "in_s": "language_model.backbone.layers.0.mixer.in_proj.weight_scale",
        "in_g": "language_model.backbone.layers.0.mixer.in_proj.weight_scale_2",
    }
    handles = {}
    out = {}
    try:
        for k, name in need.items():
            fn = wmap[name]
            if fn not in handles:
                h = _open_shard(fn)
                if h is None:
                    return None
                handles[fn] = h
            t = handles[fn].get_tensor(name)
            if "scale_2" in name:
                out[k] = (t, None)
            else:
                out[k] = (t[:4].contiguous(), t.shape)
    except Exception as e:
        print(f"  [SKIP] real tensors: {e}")
        return None
    return out


def test_real():
    real = _load_real()
    if real is None:
        print("  [SKIP] real tensors (shards not downloaded yet)")
        return
    dev = "cuda"
    # Q8_0 lm_head rows: [4, 128, 34] -> [4, 4096]
    t, _ = real["lm_head"]
    raw = t.contiguous().numpy().reshape(-1, 128, 34)[:, :4, :]
    ref = ref16(q8_0.dequant_q8_0(raw))
    out = torch.empty((raw.shape[0], 128), dtype=torch.float16, device=dev)
    ext.fq_dequant_q8_0(torch.from_numpy(raw).reshape(-1, 4, 34).cuda(), out)
    torch.cuda.synchronize()
    check("q8_0 real lm_head bit-exact", np.array_equal(bits16(out).ravel(), ref.ravel()))
    # Q8_0 latent proj
    t, _ = real["fc1"]
    raw = t.contiguous().numpy().reshape(-1, 128, 34)[:, :4, :]
    ref = ref16(q8_0.dequant_q8_0(raw))
    out = torch.empty((raw.shape[0], 128), dtype=torch.float16, device=dev)
    ext.fq_dequant_q8_0(torch.from_numpy(raw).reshape(-1, 4, 34).cuda(), out)
    torch.cuda.synchronize()
    check("q8_0 real fc1_latent_proj bit-exact", np.array_equal(bits16(out).ravel(), ref.ravel()))
    # IQ2_XXS expert up/down
    for k, per in (("exp_up", 1024), ("exp_down", 1024)):
        t, _ = real[k]
        raw = t.contiguous().numpy().reshape(-1, 4 if k == "exp_up" else 8, 66)[:, :2, :]
        ref = ref16(iq2_xxs.dequant_iq2_xxs(raw))
        out = torch.empty((raw.shape[0], 512), dtype=torch.float16, device=dev)
        ext.fq_dequant_iq2_xxs(torch.from_numpy(raw).reshape(-1, 2, 66).cuda(), out)
        torch.cuda.synchronize()
        check(f"iq2_xxs real {k} bit-exact", np.array_equal(bits16(out).ravel(), ref.ravel()))
    # NVFP4 in_proj
    w, _ = real["in_w"]
    s, _ = real["in_s"]
    (g, _) = real["in_g"]
    w_np = w.contiguous().view(torch.uint8).numpy().reshape(-1, 2048)[:, :64]
    s_np = s.contiguous().view(torch.uint8).numpy().reshape(-1, 256)[:, :8]
    gv = float(g.float().item())
    ref = ref16(nvfp4.dequant_nvfp4(w_np, s_np, np.float32(gv)))
    out = torch.empty((w_np.shape[0], 128), dtype=torch.float16, device=dev)
    ext.fq_dequant_nvfp4(torch.from_numpy(w_np).cuda(), torch.from_numpy(s_np).cuda(), gv, out)
    torch.cuda.synchronize()
    check("nvfp4 real in_proj bit-exact", np.array_equal(bits16(out).ravel(), ref.ravel()))


# --------------------------------------------------------------------------
# Fused dequant-GEMM vs dequant + torch.matmul (fp16 tolerance)
# --------------------------------------------------------------------------

def _gemm_check(name, fmt, packed_np, scales_np, g, N, K, Mlist=(1, 2, 5, 16)):
    dev = "cuda"
    rng = np.random.default_rng(7)
    if fmt == "q8_0":
        w_ref = q8_0.dequant_q8_0(packed_np).reshape(N, K)
        W = torch.from_numpy(packed_np).cuda()
        def run(x, o):
            ext.fq_gemm_q8_0(x, W, o)
    elif fmt == "iq2_xxs":
        w_ref = iq2_xxs.dequant_iq2_xxs(packed_np).reshape(N, K)
        W = torch.from_numpy(packed_np).cuda()
        def run(x, o):
            ext.fq_gemm_iq2_xxs(x, W, o)
    else:
        w_ref = nvfp4.dequant_nvfp4(packed_np, scales_np, np.float32(g)).reshape(N, K)
        W = torch.from_numpy(packed_np).cuda()
        S = torch.from_numpy(scales_np).cuda()
        def run(x, o):
            ext.fq_gemm_nvfp4(x, W, S, float(g), o)
    Wt = torch.from_numpy(w_ref).half().cuda()
    worst_abs, worst_rel = 0.0, 0.0
    for M in Mlist:
        x = torch.from_numpy(rng.standard_normal((M, K)).astype(np.float32)).half().cuda() * 0.5
        ref = torch.matmul(x.float(), Wt.t().float()).half()
        out = torch.empty((M, N), dtype=torch.float16, device=dev)
        run(x, out)
        torch.cuda.synchronize()
        d = (out.float() - ref.float()).abs()
        worst_abs = max(worst_abs, d.max().item())
        denom = ref.float().abs().clamp_min(1.0)
        worst_rel = max(worst_rel, (d / denom).max().item())
    # fp16 tolerance: one-ulp-level agreement (at 2048, 1 ulp = 2.0 absolute).
    ok = worst_abs <= 1e-2 or worst_rel <= 5e-3
    check(f"fq_gemm_{fmt} M={Mlist} within fp16 tol", ok,
          f"max_abs_diff={worst_abs:.2e} worst_rel={worst_rel:.2e}")


def test_gemm_synth():
    rng = np.random.default_rng(11)
    N, K = 256, 512
    p = np.empty((N, K // 32, 34), np.uint8)
    p[:, :, :2] = finite_scales(rng, (N, K // 32)).view(np.uint8).reshape(N, K // 32, 2)
    p[:, :, 2:] = rng.integers(-64, 65, size=(N, K // 32, 32)).astype(np.int8).view(np.uint8)
    _gemm_check("q8", "q8_0", p, None, 0.0, N, K)
    p2 = rng.integers(0, 256, size=(N, K // 256, 66)).astype(np.uint8)
    _gemm_check("iq2", "iq2_xxs", p2, None, 0.0, N, K)
    pk = rng.integers(0, 256, size=(N, K // 2)).astype(np.uint8)
    sc = rng.integers(0, 256, size=(N, K // 16)).astype(np.uint8)
    sc[(sc == 0x7F) | (sc == 0xFF)] = 0x38
    _gemm_check("nvfp4", "nvfp4", pk, sc, 0.5, N, K)


def test_gemm_real_shapes():
    # Bolt-native shapes: latent proj (Q8_0 4096->1024), expert up (IQ2 1024->2048).
    rng = np.random.default_rng(13)
    p = np.empty((1024, 128, 34), np.uint8)
    p[:, :, :2] = finite_scales(rng, (1024, 128)).view(np.uint8).reshape(1024, 128, 2)
    p[:, :, 2:] = rng.integers(-64, 65, size=(1024, 128, 32)).astype(np.int8).view(np.uint8)
    _gemm_check("q8-latent", "q8_0", p, None, 0.0, 1024, 4096, Mlist=(1, 9, 16))
    p2 = rng.integers(0, 256, size=(2048, 4, 66)).astype(np.uint8)
    _gemm_check("iq2-expert", "iq2_xxs", p2, None, 0.0, 2048, 1024, Mlist=(1, 9, 16))


def _gemv_check(name, fmt, packed_np, scales_np, g, N, K, Mlist=(1, 3)):
    # Same contract as _gemm_check, but through the cooperative GEMV kernels
    # (fq_gemv.cu): half and float in/out, out written directly in the
    # requested dtype (no extra .to() copy).
    dev = "cuda"
    rng = np.random.default_rng(7)
    if fmt == "q8_0":
        w_ref = q8_0.dequant_q8_0(packed_np).reshape(N, K)
        W = torch.from_numpy(packed_np).cuda()
        def run(x, o):
            ext.fq_gemv_q8_0(x, W, o)
    elif fmt == "iq2_xxs":
        w_ref = iq2_xxs.dequant_iq2_xxs(packed_np).reshape(N, K)
        W = torch.from_numpy(packed_np).cuda()
        def run(x, o):
            ext.fq_gemv_iq2_xxs(x, W, o)
    else:
        w_ref = nvfp4.dequant_nvfp4(packed_np, scales_np, np.float32(g)).reshape(N, K)
        W = torch.from_numpy(packed_np).cuda()
        S = torch.from_numpy(scales_np).cuda()
        def run(x, o):
            # fp-tolerance contract is the fp32 dot's: pin the q8 path off
            # (the q8 path has its own rms-normalized gate in test_q8nv).
            os.environ["BOLT_FQ_Q8NV"] = "0"
            try:
                ext.fq_gemv_nvfp4(x, W, S, float(g), o)
            finally:
                os.environ.pop("BOLT_FQ_Q8NV", None)
    Wt = torch.from_numpy(w_ref).half().cuda()
    worst_abs, worst_rel = 0.0, 0.0
    for M in Mlist:
        x = torch.from_numpy(rng.standard_normal((M, K)).astype(np.float32)).half().cuda() * 0.5
        ref = torch.matmul(x.float(), Wt.t().float())
        for dt, xf in ((torch.float16, x), (torch.float32, x.float())):
            out = torch.empty((M, N), dtype=dt, device=dev)
            run(xf, out)
            torch.cuda.synchronize()
            d = (out.float() - ref).abs()
            worst_abs = max(worst_abs, d.max().item())
            denom = ref.abs().clamp_min(1.0)
            worst_rel = max(worst_rel, (d / denom).max().item())
    ok = worst_abs <= 1e-2 or worst_rel <= 5e-3
    check(f"fq_gemv_{fmt} M={Mlist} h/f in+out within fp16 tol", ok,
          f"max_abs_diff={worst_abs:.2e} worst_rel={worst_rel:.2e}")


def test_gemv_synth():
    rng = np.random.default_rng(11)
    N, K = 256, 512
    p = np.empty((N, K // 32, 34), np.uint8)
    p[:, :, :2] = finite_scales(rng, (N, K // 32)).view(np.uint8).reshape(N, K // 32, 2)
    p[:, :, 2:] = rng.integers(-64, 65, size=(N, K // 32, 32)).astype(np.int8).view(np.uint8)
    _gemv_check("q8", "q8_0", p, None, 0.0, N, K)
    p2 = rng.integers(0, 256, size=(N, K // 256, 66)).astype(np.uint8)
    _gemv_check("iq2", "iq2_xxs", p2, None, 0.0, N, K)
    pk = rng.integers(0, 256, size=(N, K // 2)).astype(np.uint8)
    sc = rng.integers(0, 256, size=(N, K // 16)).astype(np.uint8)
    sc[(sc == 0x7F) | (sc == 0xFF)] = 0x38
    _gemv_check("nvfp4", "nvfp4", pk, sc, 0.5, N, K)


def test_gemv_real_shapes():
    # Bolt decode shapes incl. the wide ends (lm_head Q8_0, mamba NVFP4).
    rng = np.random.default_rng(13)
    p = np.empty((1024, 128, 34), np.uint8)
    p[:, :, :2] = finite_scales(rng, (1024, 128)).view(np.uint8).reshape(1024, 128, 2)
    p[:, :, 2:] = rng.integers(-64, 65, size=(1024, 128, 32)).astype(np.int8).view(np.uint8)
    _gemv_check("q8-latent", "q8_0", p, None, 0.0, 1024, 4096)
    p2 = rng.integers(0, 256, size=(2560, 4, 66)).astype(np.uint8)
    _gemv_check("iq2-expert-up", "iq2_xxs", p2, None, 0.0, 2560, 1024)
    pk = rng.integers(0, 256, size=(18560, 2048)).astype(np.uint8)
    sc = rng.integers(0, 256, size=(18560, 256)).astype(np.uint8)
    sc[(sc == 0x7F) | (sc == 0xFF)] = 0x38
    _gemv_check("nvfp4-mamba-in", "nvfp4", pk, sc, 1.0, 18560, 4096, Mlist=(1,))


def test_gemv_e4m3_sweep():
    # Every E4M3 scale code through the GEMV LUT vs the dequant path.
    dev = "cuda"
    rng = np.random.default_rng(17)
    N, K = 64, 256
    pk = rng.integers(0, 256, size=(N, K // 2)).astype(np.uint8)
    sc = np.arange(256, dtype=np.uint8)[None, :].repeat(N, axis=0)[:, :K // 16]
    w_ref = nvfp4.dequant_nvfp4(pk, sc, np.float32(1.0)).reshape(N, K)
    W = torch.from_numpy(pk).cuda()
    S = torch.from_numpy(sc).cuda()
    Wt = torch.from_numpy(w_ref).half().cuda()
    x = torch.from_numpy(rng.standard_normal((1, K)).astype(np.float32)).half().cuda() * 0.3
    ref = torch.matmul(x.float(), Wt.t().float())
    out = torch.empty((1, N), dtype=torch.float16, device=dev)
    ext.fq_gemv_nvfp4(x, W, S, 1.0, out)
    torch.cuda.synchronize()
    finite = torch.isfinite(ref)
    d = (out.float() - ref).abs() / ref.abs().clamp_min(1.0)
    d = torch.where(finite, d, torch.zeros_like(d))
    check("fq_gemv_nvfp4 all-256 E4M3 codes", d.max().item() <= 5e-3,
          f"worst_rel={d.max().item():.2e}")


def test_moe_grouped():
    # Grouped IQ2_XXS MoE (fq_moe.cu: up + down + fold launches) vs the
    # manual per-expert fq_gemv + torch relu2-square + weighted fold.
    # ones payload: codes zero -> w=1.0 exactly, keeps full Bolt sizes
    # finite; random small checks summation-order agreement.
    dev = torch.device("cuda", 0)
    torch.manual_seed(2)
    for tag, E, S, K, I, random in (
        ("ones", 16, 9, 1024, 2048, False),
        ("ones-bigE", 240, 9, 1024, 2560, False),
        ("random-small", 8, 4, 256, 256, True),
    ):
        ups, downs = [], []
        for is_up, (N, KK) in (True, (I, K)), (False, (K, I)):
            for _ in range(E):
                if random:
                    p = torch.randint(0, 256, (N, KK // 256, 64),
                                      dtype=torch.uint8, device=dev)
                    o = torch.empty((N, KK // 256, 66), dtype=torch.uint8, device=dev)
                    o[..., 0:2] = torch.tensor([0x00, 0x3C], dtype=torch.uint8, device=dev)
                    o[..., 2:].copy_(p)
                else:
                    o = torch.zeros((N, KK // 256, 66), dtype=torch.uint8, device=dev)
                    o[..., 0:2] = torch.tensor([0x00, 0x3C], dtype=torch.uint8, device=dev)
                (ups if is_up else downs).append(o)
        up_ptrs = torch.tensor([u.data_ptr() for u in ups], dtype=torch.int64, device=dev)
        down_ptrs = torch.tensor([d.data_ptr() for d in downs], dtype=torch.int64, device=dev)
        h = (torch.randn(1, K, device=dev) * (0.01 if random else 0.1)).half()
        sel = torch.tensor(sorted(torch.randperm(E)[:S].tolist()), dtype=torch.long, device=dev)
        wts = (torch.rand(S, device=dev).half() + 0.1)
        ref = torch.zeros(1, K, dtype=torch.float32, device=dev)
        for s in range(S):
            e = int(sel[s])
            u = torch.empty(1, I, dtype=torch.float16, device=dev)
            ext.fq_gemv_iq2_xxs(h, ups[e], u)
            a = torch.square(torch.relu(u))
            p = torch.empty(1, K, dtype=torch.float32, device=dev)
            ext.fq_gemv_iq2_xxs(a, downs[e], p)
            ref += p * float(wts[s])
        assert torch.isfinite(ref).all(), f"moe grouped {tag}: reference overflowed"
        partial = torch.empty(S, K, dtype=torch.float32, device=dev)
        out = torch.empty(1, K, dtype=torch.float32, device=dev)
        ext.fq_moe_iq2xxs_grouped(h, sel, wts, up_ptrs, down_ptrs,
                                  (K // 256) * 66, (I // 256) * 66, partial, out)
        torch.cuda.synchronize()
        d = (out - ref).abs()
        rel = (d / (ref.abs() + 1.0)).max().item()
        check(f"fq_moe_grouped {tag} E={E} S={S} K={K} I={I}",
              rel <= 5e-3 and rel == rel, f"maxrel1={rel:.2e}")


def test_mlp_shared():
    # Fused shared-expert up+relu2 (fq_mlp.cu) vs the torch path it
    # replaces (fq_gemv up + square(relu)): bit-identical expected (same
    # dot grid, same epilogue arithmetic), full MLP checked within tol.
    # ones payload keeps full Bolt sizes finite (K=4096, I=5376).
    dev = torch.device("cuda", 0)
    torch.manual_seed(3)
    for tag, K, I, random in (("ones", 4096, 5376, False),
                              ("random-small", 256, 256, True)):
        if random:
            p = torch.randint(0, 256, (I, K // 256, 64), dtype=torch.uint8, device=dev)
            up = torch.empty((I, K // 256, 66), dtype=torch.uint8, device=dev)
            up[..., 0:2] = torch.tensor([0x00, 0x3C], dtype=torch.uint8, device=dev)
            up[..., 2:].copy_(p)
            pd = torch.randint(0, 256, (K, I // 256, 64), dtype=torch.uint8, device=dev)
            down = torch.empty((K, I // 256, 66), dtype=torch.uint8, device=dev)
            down[..., 0:2] = torch.tensor([0x00, 0x3C], dtype=torch.uint8, device=dev)
            down[..., 2:].copy_(pd)
        else:
            up = torch.zeros((I, K // 256, 66), dtype=torch.uint8, device=dev)
            up[..., 0:2] = torch.tensor([0x00, 0x3C], dtype=torch.uint8, device=dev)
            down = torch.zeros((K, I // 256, 66), dtype=torch.uint8, device=dev)
            down[..., 0:2] = torch.tensor([0x00, 0x3C], dtype=torch.uint8, device=dev)
        x = (torch.randn(1, K, device=dev) * (0.01 if random else 0.1)).half()
        u = torch.empty(1, I, dtype=torch.float16, device=dev)
        ext.fq_gemv_iq2_xxs(x, up, u)
        a = torch.square(torch.relu(u))
        h = torch.empty(1, I, dtype=torch.float16, device=dev)
        ext.fq_mlp_shared_iq2xxs(x, up, h)
        torch.cuda.synchronize()
        dh = (h.float() - a.float()).abs().max().item()
        check(f"fq_mlp_shared {tag} up+relu2 K={K} I={I}", dh == 0.0, f"maxabs={dh:.2e}")
        assert torch.isfinite(a).all(), f"mlp shared {tag}: reference overflowed"
        po = torch.empty(1, K, dtype=torch.float32, device=dev)
        ext.fq_gemv_iq2_xxs(a, down, po)
        pf = torch.empty(1, K, dtype=torch.float32, device=dev)
        ext.fq_gemv_iq2_xxs(h, down, pf)
        torch.cuda.synchronize()
        dd = (pf - po).abs().max().item()
        check(f"fq_mlp_shared {tag} full-mlp K={K} I={I}", dd == 0.0, f"maxabs={dd:.2e}")


def test_q8act():
    # q8_1-activation IQ2_XXS path (Bolt phase 2, BOLT_FQ_Q8ACT=1): integer
    # dp4a dots over once-per-row q8_1 activations, same arithmetic as
    # llama.cpp's mmvq vec dots. Gate: quantize matches the ggml
    # quantize_q8_1 arithmetic (d=amax/127, round-half-away) bit-exactly;
    # q8 dots agree with the fp dots (top-1 + small rel diff) at Bolt
    # shared shapes with small scales (real Bolt expert scales are ~1e-2;
    # d=1.0 random payloads overflow fp16, same as the MoE tests note).
    import struct
    dev = torch.device("cuda", 0)
    torch.manual_seed(11)
    for K in (256, 1024, 4096):
        x = (torch.randn(1, K, device=dev) * 0.3).half()
        qb = torch.empty(1, (K // 32) * 36, dtype=torch.uint8, device=dev)
        ext.fq_quant_q8_1_rows(x, qb)
        torch.cuda.synchronize()
        qbn = qb.cpu().numpy()
        xf = x.cpu().float().numpy()[0]
        worst_q, worst_d = 0, 0.0
        for b in range(K // 32):
            blk = qbn[0, b * 36:(b + 1) * 36]
            d = struct.unpack("<e", bytes(blk[0:2]))[0]
            ref = xf[b * 32:(b + 1) * 32].astype(np.float64)
            amax = np.abs(ref).max()
            dref = amax / 127.0
            # round-half-away (ggml), not numpy banker's: allow off-by-one
            q = blk[4:].astype(np.int8).astype(np.int32)
            qe = np.zeros(32, dtype=np.int32) if amax == 0 else np.clip(
                np.floor(ref / dref + 0.5), -127, 127).astype(np.int32)
            worst_q = max(worst_q, int(np.abs(q - qe).max()))
            worst_d = max(worst_d, abs(d - dref))
        check(f"fq_q8act quant K={K}", worst_q <= 1 and worst_d <= 2e-3,
              f"maxqdiff={worst_q} maxdd={worst_d:.2e}")
    for tag, N, K in (("shared_up", 5376, 4096), ("shared_down", 4096, 5376),
                      ("expert_up", 2560, 1024), ("expert_down", 1024, 2560)):
        p = torch.randint(0, 256, (N, K // 256, 64), dtype=torch.uint8, device=dev)
        sc = (torch.rand(N, K // 256, device=dev) * 0.02).half()
        scb = sc.cpu().numpy().view(np.uint8).reshape(N, K // 256, 2)
        pk = torch.empty((N, K // 256, 66), dtype=torch.uint8, device=dev)
        pk[..., 0:2].copy_(torch.from_numpy(scb).to(dev))
        pk[..., 2:].copy_(p)
        x = (torch.randn(2, K, device=dev) * 0.1).half()
        oo = torch.empty(2, N, dtype=torch.float, device=dev)
        os.environ["BOLT_FQ_Q8ACT"] = "1"
        ext.fq_gemv_iq2_xxs(x, pk, oo)
        torch.cuda.synchronize()
        new = oo.clone().float()
        os.environ.pop("BOLT_FQ_Q8ACT", None)
        ext.fq_gemv_iq2_xxs(x, pk, oo)
        torch.cuda.synchronize()
        old = oo.clone().float()
        assert torch.isfinite(old).all() and torch.isfinite(new).all(), \
            f"q8act {tag}: overflow (scales too big?)"
        d = (new - old).abs()
        rel = (d / (old.abs() + 1.0))
        agree = (new.argmax(1) == old.argmax(1)).sum().item()
        check(f"fq_q8act gemv {tag}", agree == 2 and rel.max().item() <= 0.3
              and rel.mean().item() <= 3e-2,
              f"top1={agree}/2 maxrel={rel.max().item():.2e} meanrel={rel.mean().item():.2e}")


def test_q8nv():
    # q8_1-activation NVFP4 + Q8_0 paths (Bolt phase 2, BOLT_FQ_Q8NV=1 /
    # BOLT_FQ_Q8Q8=1): integer dp4a dots over once-per-row q8_1 activations,
    # same factorization as llama.cpp's vec_dot_nvfp4_q8_1 / vec_dot_q8_0_q8_1
    # (doubled-e2m1 int8 table + halved ue4m3 scale; direct int8 for Q8).
    # Gate: exact-arithmetic payloads (lossless activation quant) match the
    # closed form to fp16-scale noise; random payloads agree with the fp
    # dots at the activation-quant noise floor (rms-normalized).
    dev = torch.device("cuda", 0)
    torch.manual_seed(0)
    os.environ["BOLT_FQ_Q8NV"] = "1"
    os.environ["BOLT_FQ_Q8Q8"] = "1"
    try:
        for N, K in ((256, 1024), (4096, 8192)):
            p = torch.full((N, K // 2), 0x55, dtype=torch.uint8, device=dev)
            s = torch.full((N, K // 16), 0x38, dtype=torch.uint8, device=dev)
            x = torch.full((1, K), 0.5, dtype=torch.half, device=dev)
            oo = torch.empty(1, N, dtype=torch.float, device=dev)
            ext.fq_gemv_nvfp4(x, p, s, 0.5, oo, True)
            torch.cuda.synchronize()
            rel = ((oo.float() - 0.75 * K).abs() / (0.75 * K)).max().item()
            check(f"fq_q8nv exact {N}x{K}", rel < 1e-3, f"maxrel={rel:.2e}")
        for tag, N, K in (("mamba_in", 18560, 4096), ("mamba_out", 4096, 8192)):
            p = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
            se = torch.randint(5, 9, (N, K // 16), device=dev, dtype=torch.uint8)
            sm = torch.randint(0, 8, (N, K // 16), device=dev, dtype=torch.uint8)
            s = (se << 3) | sm
            x = (torch.randn(2, K, device=dev) * 0.1).half()
            oo = torch.empty(2, N, dtype=torch.float, device=dev)
            ext.fq_gemv_nvfp4(x, p, s, 1.0, oo, True)
            torch.cuda.synchronize()
            new = oo.clone()
            os.environ["BOLT_FQ_Q8NV"] = "0"
            ext.fq_gemv_nvfp4(x, p, s, 1.0, oo, True)
            torch.cuda.synchronize()
            old = oo.clone()
            os.environ["BOLT_FQ_Q8NV"] = "1"
            d = (new - old).abs()
            rms = old.pow(2).mean().sqrt().item()
            agree = (new.argmax(1) == old.argmax(1)).sum().item()
            check(f"fq_q8nv {tag}", agree == 2 and (d.max().item() / rms) <= 0.1,
                  f"top1={agree}/2 max/rms={(d.max().item() / rms):.2e}")
        for tag, N, K in (("latent", 1024, 4096), ("lm_head", 4096, 4096)):
            o = torch.empty((N, K // 32, 34), dtype=torch.uint8, device=dev)
            o[..., 0:2] = torch.tensor([0x00, 0x3C], dtype=torch.uint8, device=dev)
            o[..., 2:].copy_(torch.randint(0, 256, (N, K // 32, 32),
                                           dtype=torch.uint8, device=dev))
            x = (torch.randn(2, K, device=dev) * 0.3).half()
            oo = torch.empty(2, N, dtype=torch.float, device=dev)
            ext.fq_gemv_q8_0(x, o, oo)
            torch.cuda.synchronize()
            new = oo.clone()
            os.environ["BOLT_FQ_Q8Q8"] = "0"
            ext.fq_gemv_q8_0(x, o, oo)
            torch.cuda.synchronize()
            old = oo.clone()
            os.environ["BOLT_FQ_Q8Q8"] = "1"
            d = (new - old).abs()
            rms = old.pow(2).mean().sqrt().item()
            agree = (new.argmax(1) == old.argmax(1)).sum().item()
            check(f"fq_q8q8 {tag}", agree == 2 and (d.max().item() / rms) <= 0.1,
                  f"top1={agree}/2 max/rms={(d.max().item() / rms):.2e}")
    finally:
        os.environ.pop("BOLT_FQ_Q8NV", None)
        os.environ.pop("BOLT_FQ_Q8Q8", None)


def test_q8moe():
    # q8_1-activation grouped MoE (Bolt phase 2, BOLT_FQ_Q8MOE=1): same
    # (slot, tile) grid as fq_moe.cu, integer-dp4a lane dots over
    # once-per-step q8_1 activations. Gate: ones payload (all experts
    # identical, full-scale, no cancellation) + small-scale random
    # payload, vs the fp grouped path (rms-normalized: random payloads
    # cancel in the fold, so absolute-tolerance metrics mislead).
    dev = torch.device("cuda", 0)
    torch.manual_seed(2)
    E, S, K, I = 8, 3, 1024, 2560
    for tag, ones in (("ones", True), ("small", False)):
        ups, downs = [], []
        for _ in range(E):
            for lst, N, KK in ((ups, I, K), (downs, K, I)):
                o = torch.empty((N, KK // 256, 66), dtype=torch.uint8, device=dev)
                if ones:
                    o[..., 0:2] = torch.tensor([0x00, 0x3C], dtype=torch.uint8, device=dev)
                    o[..., 2:].fill_(1)
                else:
                    sc = (torch.rand(N, KK // 256, device=dev) * 0.02).half()
                    scb = sc.cpu().numpy().view(np.uint8).reshape(N, KK // 256, 2)
                    o[..., 0:2].copy_(torch.from_numpy(scb).to(dev))
                    o[..., 2:].copy_(torch.randint(0, 256, (N, KK // 256, 64),
                                                   dtype=torch.uint8, device=dev))
                lst.append(o)
        up_ptrs = torch.tensor([u.data_ptr() for u in ups], dtype=torch.int64, device=dev)
        down_ptrs = torch.tensor([d.data_ptr() for d in downs], dtype=torch.int64, device=dev)
        h = (torch.randn(1, K, device=dev) * 0.1).half()
        sel = torch.tensor([0, 2, 5], dtype=torch.long, device=dev)
        wts = (torch.rand(S, device=dev).half() + 0.1)
        partial = torch.empty(S, K, dtype=torch.float32, device=dev)
        out = torch.empty(1, K, dtype=torch.float32, device=dev)
        up_s, down_s = (K // 256) * 66, (I // 256) * 66
        os.environ["BOLT_FQ_Q8MOE"] = "1"
        ext.fq_moe_iq2xxs_grouped(h, sel, wts, up_ptrs, down_ptrs, up_s, down_s, partial, out)
        torch.cuda.synchronize()
        new = out.clone()
        os.environ.pop("BOLT_FQ_Q8MOE", None)
        ext.fq_moe_iq2xxs_grouped(h, sel, wts, up_ptrs, down_ptrs, up_s, down_s, partial, out)
        torch.cuda.synchronize()
        old = out.clone()
        okf = bool(torch.isfinite(old).all() and torch.isfinite(new).all())
        d = (new - old).abs()
        rms = old.pow(2).mean().sqrt().item()
        check(f"fq_q8moe {tag}", okf and (d.max().item() / rms) <= 0.15,
              f"finite={okf} max/rms={(d.max().item() / rms):.2e}")


def test_q8v2():
    # v2 IQ2_XXS q8 dot (Bolt phase 2 step 13, BOLT_FQ_Q8V2=1): same
    # integer arithmetic as the v1 dot (const-grid like llama's
    # iq2xxs_grid, inline ksigns unpack, direct u16 header loads; no
    # header staging, no signs32 table, one barrier). Gate: BITWISE vs
    # v1 on shared/expert shapes (ones payload: exact, no fp16
    # rounding in the reference path) + grouped-MoE bitwise vs v1.
    dev = torch.device("cuda", 0)
    torch.manual_seed(3)
    os.environ["BOLT_FQ_Q8ACT"] = "1"
    try:
        for tag, N, K in (("shared_up", 5376, 4096), ("shared_down", 4096, 5376),
                          ("expert_up", 2560, 1024), ("expert_down", 1024, 2560)):
            p = torch.empty((N, K // 256, 66), dtype=torch.uint8, device=dev)
            p[..., 0:2] = torch.tensor([0x00, 0x3C], dtype=torch.uint8, device=dev)
            p[..., 2:].fill_(1)
            x = (torch.randn(1, K, device=dev) * 0.3).half()
            oo = torch.empty(1, N, dtype=torch.float, device=dev)
            os.environ["BOLT_FQ_Q8V2"] = "1"
            ext.fq_gemv_iq2_xxs(x, p, oo)
            torch.cuda.synchronize()
            new = oo.clone()
            os.environ["BOLT_FQ_Q8V2"] = "0"
            ext.fq_gemv_iq2_xxs(x, p, oo)
            torch.cuda.synchronize()
            old = oo.clone()
            nbad = int((new != old).sum())
            check(f"fq_q8v2 gemv {tag}", nbad == 0, f"nbad={nbad}/{N}")
        # grouped MoE + fused shared MLP take the same v2 dot: bitwise.
        E, S, K, I = 8, 3, 1024, 2560
        ups, downs = [], []
        for _ in range(E):
            for lst, N, KK in ((ups, I, K), (downs, K, I)):
                o = torch.empty((N, KK // 256, 66), dtype=torch.uint8, device=dev)
                o[..., 0:2] = torch.tensor([0x00, 0x3C], dtype=torch.uint8, device=dev)
                o[..., 2:].fill_(1)
                lst.append(o)
        up_ptrs = torch.tensor([u.data_ptr() for u in ups], dtype=torch.int64, device=dev)
        down_ptrs = torch.tensor([d.data_ptr() for d in downs], dtype=torch.int64, device=dev)
        h = (torch.randn(1, K, device=dev) * 0.1).half()
        sel = torch.tensor([0, 2, 5], dtype=torch.long, device=dev)
        wts = (torch.rand(S, device=dev).half() + 0.1)
        partial = torch.empty(S, K, dtype=torch.float32, device=dev)
        out = torch.empty(1, K, dtype=torch.float32, device=dev)
        up_s, down_s = (K // 256) * 66, (I // 256) * 66
        os.environ["BOLT_FQ_Q8MOE"] = "1"
        os.environ["BOLT_FQ_Q8V2"] = "1"
        ext.fq_moe_iq2xxs_grouped(h, sel, wts, up_ptrs, down_ptrs, up_s, down_s, partial, out)
        torch.cuda.synchronize()
        new = out.clone()
        os.environ["BOLT_FQ_Q8V2"] = "0"
        ext.fq_moe_iq2xxs_grouped(h, sel, wts, up_ptrs, down_ptrs, up_s, down_s, partial, out)
        torch.cuda.synchronize()
        old = out.clone()
        nbad = int((new != old).sum())
        check("fq_q8v2 grouped", nbad == 0, f"nbad={nbad}/{K}")
        os.environ.pop("BOLT_FQ_Q8MOE", None)
    finally:
        os.environ.pop("BOLT_FQ_Q8ACT", None)
        os.environ.pop("BOLT_FQ_Q8V2", None)


def main():
    assert torch.cuda.is_available(), "CUDA required"
    for f in ("fq_dequant_q8_0", "fq_dequant_iq2_xxs", "fq_dequant_nvfp4",
              "fq_gemm_q8_0", "fq_gemm_iq2_xxs", "fq_gemm_nvfp4",
              "fq_gemv_q8_0", "fq_gemv_iq2_xxs", "fq_gemv_nvfp4",
              "fq_moe_iq2xxs_grouped", "fq_mlp_shared_iq2xxs",
              "fq_quant_q8_1_rows", "fq_gemv_iq2_xxs_q8",
              "fq_gemv_nvfp4_q8", "fq_gemv_q8_0_q8"):
        assert hasattr(ext, f), f"ext missing {f} (rebuild?)"
    print(f"ext: {ext.__file__}  cuda={torch.cuda.get_device_name(0)}")
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    test_q8_0_synth()
    test_iq2_xxs_synth()
    test_nvfp4_synth()
    test_nvfp4_e4m3_finite_sweep()
    test_real()
    test_gemm_synth()
    test_gemm_real_shapes()
    test_gemv_synth()
    test_gemv_real_shapes()
    test_gemv_e4m3_sweep()
    test_moe_grouped()
    test_mlp_shared()
    test_q8act()
    test_q8nv()
    test_q8moe()
    test_q8v2()
    t1.record()
    torch.cuda.synchronize()
    print(f"done in {t0.elapsed_time(t1) / 1000:.1f}s: {len(PASS)} pass, {len(FAIL)} fail")
    if FAIL:
        print("FAILURES:", FAIL)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
