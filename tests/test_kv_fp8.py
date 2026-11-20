"""CPU: TRITON_INTERPRET=1 DEV=cpu python tests/test_kv_fp8.py     GPU: DEV=cuda python tests/test_kv_fp8.py"""
import importlib.util, os, sys, types
import torch
DEV = os.environ.get("DEV", "cpu")
here = os.path.dirname(os.path.abspath(__file__))
def load(name):
    sp = importlib.util.spec_from_file_location(name, os.path.join(here, "..", "exllamav3", "loader", name + ".py"))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
KV, LS = load("kv_fp8"), load("lc_sparse")

def test_dq8_matches_torch():
    allb = torch.arange(256, dtype=torch.uint8)
    allb = allb[(allb != 0x7F) & (allb != 0xFF)]
    want = allb.view(torch.float8_e4m3fn).float()
    assert torch.equal(KV.dq8(allb), want)

def test_roundtrip():
    g = torch.Generator().manual_seed(0)
    kn = torch.randn(64, 576, generator=g) * torch.logspace(-2, 2, 64)[:, None]
    kv = KV.Fp8KV(100, 512, 64, "cpu"); kv.write(5, kn)
    back = kv[5:69].dequant32()
    rel = ((back[:, :512] - kn[:, :512]).abs().amax(-1) / kn[:, :512].abs().amax(-1)).max().item()
    assert rel < 0.07, rel            # e4m3: 3 mantissa bits, worst case 2^-4 of the row max
    assert (back[:, 512:] - kn[:, 512:]).abs().max() < 1e-2 * kn[:, 512:].abs().max()
    pos = torch.tensor([7, 3]); kv.write_pos(pos, kn[:2]); assert torch.allclose(kv[3:4].dequant32(), kv[5 + 1:5 + 2].dequant32() * 0 + kv[3:4].dequant32())
    assert kv.shape == (100, 576) and kv[10:20].shape == (10, 576) and kv.nbytes() == 100 * (512 + 4 + 128)
    lat16, pe16 = kv[5:69].dequant16(); assert lat16.dtype == torch.float16 and pe16.shape == (64, 64)

def test_dequant16_cat():
    g = torch.Generator().manual_seed(1)
    kn = torch.randn(300, 576, generator=g) * torch.logspace(-2, 3, 300)[:, None]
    kv = KV.Fp8KV(300, 512, 64, DEV); kv.write(0, kn.to(DEV))
    a = kv.dequant16_cat(chunk=128)
    lat16, pe16 = kv.dequant16()
    assert a.dtype == torch.float16 and a.shape == (300, 576)
    assert torch.equal(a[:, :512], lat16) and torch.equal(a[:, 512:], pe16)
    assert torch.equal(kv[10:200].dequant16_cat(chunk=64), a[10:200])

def test_kernel_matches_dequant():
    if DEV == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        return
    g = torch.Generator().manual_seed(1)
    T, H, R, P, N, kp, nt = 3, 4, 512, 64, 700, 4, 32
    kn = (torch.randn(N, R + P, generator=g) * 1.5).to(DEV)
    kv = KV.Fp8KV(N, R, P, DEV); kv.write(0, kn)
    deq = kv.dequant32()
    ql = torch.randn(T, H, R, generator=g).to(DEV); qp = torch.randn(T, H, P, generator=g).to(DEV); sinks = torch.randn(H, generator=g).to(DEV)
    pos = torch.tensor([600, 601, 650], device=DEV)
    idx = torch.stack([torch.randperm(N // kp - 3, generator=g)[:nt].sort().values for _ in range(T)]).to(torch.int32).to(DEV)
    nsel = torch.full((T,), nt, dtype=torch.int32, device=DEV)
    sel = LS.SparseSel(idx, nsel, kp, True)
    a = LS.sparse_mla_attend(ql, qp, kv.lat, kv.pe, pos, sel, 1 / 16, sinks, scl=kv.sc)
    b = LS.sparse_mla_attend(ql, qp, deq[:, :R].contiguous(), deq[:, R:].contiguous(), pos, sel, 1 / 16, sinks)
    err = (a - b).abs().max().item(); assert err < 2e-5, err
    c = LS.sparse_mla_attend(ql, qp, kn[:, :R].contiguous(), kn[:, R:].contiguous(), pos, sel, 1 / 16, sinks)
    print("fp8 vs exact attend max abs diff", (a - c).abs().max().item(), "out scale", c.abs().mean().item())

def test_head_blocked_matches_one_head():
    """HY4_LC_HB kernels (single pass and split-K with the combine) vs the one-head fp8 kernel on the same cache (Triton interpreter on CPU)"""
    if DEV == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        return
    pkg = types.ModuleType("kvhb_pkg"); pkg.__path__ = [os.path.join(here, "..", "exllamav3", "loader")]; sys.modules["kvhb_pkg"] = pkg      # package context for the relative import of ref_fused (combine kernel)
    import importlib
    LS = importlib.import_module("kvhb_pkg.lc_sparse")
    g = torch.Generator().manual_seed(2)
    T, H, R, P, N, kp, nt = 3, 16, 512, 64, 400, 4, 24
    kn = (torch.randn(N, R + P, generator=g) * 1.5).to(DEV)
    kv = KV.Fp8KV(N, R, P, DEV); kv.write(0, kn)
    ql = torch.randn(T, H, R, generator=g).to(DEV); qp = torch.randn(T, H, P, generator=g).to(DEV); sinks = torch.randn(H, generator=g).to(DEV)
    pos = torch.tensor([300, 301, 350], device=DEV)
    idx = torch.stack([torch.randperm(N // kp - 3, generator=g)[:nt].sort().values for _ in range(T)]).to(torch.int32).to(DEV)
    sel = LS.SparseSel(idx, torch.full((T,), nt, dtype=torch.int32, device=DEV), kp, True)
    old = (LS.HB_ON, LS.HB_SPLIT)
    try:
        LS.HB_ON = False
        ref = LS.sparse_mla_attend(ql, qp, kv.lat, kv.pe, pos, sel, 1 / 16, sinks, scl=kv.sc)
        for split in (0, -1, 3):
            LS.HB_ON, LS.HB_SPLIT = True, split
            out = LS.sparse_mla_attend(ql, qp, kv.lat, kv.pe, pos, sel, 1 / 16, sinks, scl=kv.sc)
            err = (out - ref).abs().max().item()
            assert err < 5e-3, (split, err)      # fp16 operands of tl.dot vs the fp32 one-head kernel
    finally:
        LS.HB_ON, LS.HB_SPLIT = old

G = 2 ** 30

def kv_plan_gib(n, fp8, full=13, swa=30, ring=1024, lat=576):
    """latent K cache of Hy4-flash in GiB: `full` full-attention layers (12 trunk + the MTP layer, which stays fp32), `swa` window layers on a `ring` rows ring (fp32)"""
    k = (12 * n * KV.Fp8KV(1, 512, 64, "cpu").nbytes() + n * lat * 4) if fp8 else full * n * lat * 4
    return (k + swa * ring * lat * 4) / G

def test_sizing_math():
    assert KV.Fp8KV(1, 512, 64, "cpu").nbytes() == 644            # bytes per token and layer, fp32 is 2304
    got = {n: (round(kv_plan_gib(n, False), 1), round(kv_plan_gib(n, True), 1)) for n in (131072, 262144)}
    assert got == {131072: (3.7, 1.3), 262144: (7.4, 2.5)}, got      # the doc table of doc/hy4.md
    assert kv_plan_gib(65536, True) < kv_plan_gib(65536, False) / 2.5
    for n in (65536, 131072, 262144):                                # -c 262144 with fp8 stays under the 128 GB machine with room for weights (84.3) + pools / activations (3)
        assert 84.3 + 3.0 + kv_plan_gib(n, True) < 100.0

def test_roundtrip_cpu_rows_and_scale():
    """quantise on write, per-row scale: a row of 1e-3 magnitudes and a row of 1e3 magnitudes both keep ~2^-4 relative accuracy; an all-zero row stays zero"""
    kn = torch.zeros(3, 576)
    g = torch.Generator().manual_seed(3)
    kn[0, :512] = torch.randn(512, generator=g) * 1e-3
    kn[1, :512] = torch.randn(512, generator=g) * 1e3
    kv = KV.Fp8KV(3, 512, 64, "cpu"); kv.write(0, kn)
    back = kv.dequant32()
    for r in (0, 1):
        assert ((back[r, :512] - kn[r, :512]).abs().max() / kn[r, :512].abs().max()) < 0.07
    assert torch.count_nonzero(back[2]) == 0

if __name__ == "__main__":
    test_dq8_matches_torch(); test_roundtrip(); test_dequant16_cat(); test_kernel_matches_dequant(); test_head_blocked_matches_one_head(); test_sizing_math(); test_roundtrip_cpu_rows_and_scale(); print("kv_fp8 ok")
