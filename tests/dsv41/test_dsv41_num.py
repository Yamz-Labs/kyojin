"""Bit-exactness of the number formats the DS4.1 HIP kernels use (dsv41_num.h), CPU only.

The header is compiled with g++ (tests/dsv41/num_cpu.cpp) and compared bit for bit against the torch reference
(exllamav3/dsv41/quant.py, engram.py). Run with `pytest --noconftest` so nothing opens /dev/kfd.
"""
import ctypes, os, subprocess, pathlib
import numpy as np
import pytest
import torch

from hardware import require
require("ext")
from exllamav3.dsv41 import quant
from exllamav3.dsv41.engram import (EngramLayout, NgramHasher, compute_multipliers, dequant_rows)

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def lib():
    out = ROOT / "scratch" / "build"
    out.mkdir(parents=True, exist_ok=True)
    so = out / "num_cpu.so"
    src = ROOT / "tests" / "dsv41" / "num_cpu.cpp"
    hdr = ROOT / "exllamav3" / "modules" / "dsv41_fn" / "dsv41_num.h"
    if not so.exists() or so.stat().st_mtime < max(src.stat().st_mtime, hdr.stat().st_mtime):
        subprocess.run(["g++", "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(so), str(src)], check=True)
    return ctypes.CDLL(str(so))


def _rows(n_rows, n, seed, kind):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_rows, n, generator=g)
    if kind == "wide":        # per-group magnitude spread over 20 binades, exact zeros, tiny and huge values
        x = x * torch.exp2(torch.randint(-12, 9, (n_rows, n // 16), generator=g).float().repeat_interleave(16, 1))
    elif kind == "edge":      # values at / around the rounding thresholds of the grids
        base = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 448.0, 464.0, 480.0, 0.015625, 0.0078125,
                             0.001953125, 1e-4, 1e-30])
        x = base[torch.randint(0, len(base), (n_rows, n), generator=g)] * torch.exp2(
            torch.randint(-3, 4, (n_rows, 1), generator=g).float())
        x = x * (torch.randint(0, 2, (n_rows, n), generator=g).float() * 2 - 1)
        x = torch.nextafter(x, x + (torch.randint(-1, 2, (n_rows, n), generator=g).float()))
    x[0, : min(n, 64)] = 0.0
    return x.contiguous()


def _fq(lib, x, mode):
    y = x.clone().float().contiguous()
    lib.ds_fq_rows(ctypes.c_void_p(y.data_ptr()), ctypes.c_long(y.size(0)), ctypes.c_long(y.size(1)), mode)
    return y


@pytest.mark.parametrize("kind", ["gauss", "wide", "edge"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fp8_fq_bitwise(lib, kind, dtype):
    x = _rows(512, 512, 1, kind).to(dtype)
    ref = quant.act_quant_fp8_(x.clone(), 32)
    got = _fq(lib, x, 0).to(dtype)
    assert torch.equal(ref.float().view(torch.int32), got.float().view(torch.int32))


@pytest.mark.parametrize("kind", ["gauss", "wide", "edge"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fp4_ue8m0_fq_bitwise(lib, kind, dtype):
    x = _rows(512, 512, 2, kind).to(dtype)
    ref = quant.fp4_act_quant_(x.clone(), 32)
    got = _fq(lib, x, 1).to(dtype)
    assert torch.equal(ref.float().view(torch.int32), got.float().view(torch.int32))


@pytest.mark.parametrize("kind", ["gauss", "wide", "edge"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fp4_e4m3_scale_fq_bitwise(lib, kind, dtype):
    # the reference turns an e4m3 scale > 448 (|x| > 2688, never reached by a normed latent) into NaN; the kernel saturates
    x = _rows(512, 512, 3, kind).clamp(-2688, 2688).to(dtype)
    ref = quant.fp4_act_quant_(x.clone(), 16, e4m3_scale=True)
    got = _fq(lib, x, 2).to(dtype)
    assert torch.equal(ref.float().view(torch.int32), got.float().view(torch.int32))


def test_e4m3_round_all_codes_and_midpoints(lib):
    codes = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    codes = codes[torch.isfinite(codes)]
    pts = [codes]
    s = torch.sort(codes.abs().unique()).values
    mids = (s[1:] + s[:-1]) / 2
    for m in (mids, torch.nextafter(mids, torch.zeros(1)), torch.nextafter(mids, torch.full((1,), 1e9))):
        pts += [m, -m]
    pts.append(torch.randn(1 << 18) * 100)
    x = torch.cat(pts).clamp(-448, 448).contiguous()
    ref = x.to(torch.float8_e4m3fn).float()
    y = torch.empty_like(x)
    lib.ds_e4m3_round_arr(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), ctypes.c_long(x.numel()))
    assert torch.equal(ref.view(torch.int32), y.view(torch.int32))


def test_mxfp4_decode_bitwise(lib):
    g = torch.Generator().manual_seed(7)
    raw = torch.randint(0, 256, (4096, 136), dtype=torch.uint8, generator=g)
    raw[:, 128:] = torch.randint(100, 150, (4096, 8), dtype=torch.uint8, generator=g)   # sane e8m0 range
    raw[:16, 128:] = torch.randint(0, 255, (16, 8), dtype=torch.uint8, generator=g)      # incl. 0 (2^-127)
    ref = dequant_rows(raw, torch.float32, "mxfp4")
    out = torch.empty(4096, 256)
    lib.ds_mxfp4_decode(ctypes.c_void_p(raw.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_long(4096))
    assert torch.equal(ref.view(torch.int32), out.view(torch.int32))


@pytest.mark.parametrize("start,L,masked", [(0, 40, False), (5, 17, False), (0, 33, True)])
def test_engram_hash_bitwise(lib, start, L, masked):
    layers, ng, nh = [1, 14], 4, 8
    lay = EngramLayout.build(layers, None, ng, 16_000_000, nh, 256)       # real primes (derived like the reference)
    V = 99092
    g = torch.Generator().manual_seed(11)
    tmap = torch.randint(0, V, (129280,), generator=g).tolist()
    hasher = NgramHasher(lay, tmap, 2, V, None, 1, 128, "cpu")
    ids = torch.randint(0, 129280, (1, start + L), generator=g)
    mask = None
    if masked:
        mask = torch.rand(1, start + L, generator=g) > 0.2
    if start:
        hasher(ids[:, :start], 0, None if mask is None else mask[:, :start])
    ref = hasher(ids[:, start:], start, None if mask is None else mask[:, start:])        # [1, L, nl, 24]
    cache = np.ascontiguousarray(hasher.cache[0].numpy().astype(np.int64))
    mult = np.ascontiguousarray(hasher.mult.numpy())
    primes = np.ascontiguousarray(hasher.primes.numpy())
    offs = np.ascontiguousarray(hasher.offsets.numpy())
    pad = int(torch.as_tensor(tmap)[2])
    out = np.zeros((L, len(layers), (ng - 1) * nh), dtype=np.int64)
    P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    lib.ds_engram_pos.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int64,
                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    for r in range(L):
        for l in range(len(layers)):
            lib.ds_engram_pos(P(cache), start + r, ng, nh, pad, P(mult[l]), P(primes[l]), P(offs[l]), P(out[r, l]))
    assert np.array_equal(out, ref[0].numpy())
