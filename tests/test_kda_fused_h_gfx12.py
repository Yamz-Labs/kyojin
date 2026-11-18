"""gfx12 (RDNA4) parity, determinism and checkpoint tests for the fused KDA HIP state kernel.

The HIP kernel port of kda_fused_h.hip must reproduce the Triton fused path
(chunk_kda_fwd_h_fused, fuse_o=True) bitwise on gfx1200/gfx1201, including the final
state and the CK4/CK8 checkpoint side-writes. Skipped on every other device.

STATUS: the gfx12 path compiles, loads, runs and is bitwise equal to the Triton fused
path on gfx1201 for the shape/ragged/h0/ht matrix and CK4/CK8 checkpoints. The missing
piece was the bf16 x bf16 product rounding: under Triton 3.9 it rounds to nearest even
on both gfx11.5 and gfx12. `kda_fused_h.hip` does the same in `f2bf_mul`, and the runtime
gate now includes gfx1200/gfx1201.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.version.hip, reason="requires ROCm")

T_ALL = (513, 777, 1500, 3001, 4095)


def _gfx12_device():
    from exllamav3.util.arch_list import device_arch
    if not torch.cuda.is_available():
        return None
    for i in range(torch.cuda.device_count()):
        try:
            if device_arch(i) in ("gfx1200", "gfx1201"):
                return i
        except Exception:                                   # noqa: BLE001
            pass
    return None


@pytest.fixture
def dev():
    i = _gfx12_device()
    if i is None:
        pytest.skip("requires a gfx1200/gfx1201 device")
    torch.cuda.set_device(i)
    return torch.device("cuda", i)


@pytest.fixture(scope="module")
def kda():
    from exllamav3.vendor.fla import RCP_LN2
    from exllamav3.vendor.fla.cumsum import chunk_local_cumsum
    from exllamav3.vendor.fla.kda_intra_fused import chunk_kda_fwd_intra_pinned
    from exllamav3.vendor.fla import kda_fused_h as KH
    from exllamav3.vendor.fla.hip import kda_fused_h_hip as HH
    return RCP_LN2, chunk_local_cumsum, chunk_kda_fwd_intra_pinned, KH, HH


def test_runtime_gate_accepts(dev, kda):
    assert kda[4].supported(dev) is True


def _synth(dev, B, T, seed=0):
    H = HV = 64
    K = V = 128
    g = torch.Generator(device = dev).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device = dev, generator = g)
    q = torch.nn.functional.normalize(r(B, T, H, K), dim = -1).bfloat16()
    k = torch.nn.functional.normalize(r(B, T, H, K), dim = -1).bfloat16()
    v = r(B, T, HV, V).bfloat16()
    g_raw = (-5.0 * torch.sigmoid(r(B, T, HV, K) - 2.0)).float()
    beta = torch.sigmoid(r(B, T, HV)).bfloat16()
    h0 = 0.1 * r(B, HV, K, V).float()
    return q, k, v, g_raw, beta, h0


def _inputs(kda, q, k, v, g_raw, beta):
    RCP_LN2, chunk_local_cumsum, chunk_kda_fwd_intra_pinned = kda[:3]
    g = chunk_local_cumsum(g_raw, chunk_size = 64, scale = RCP_LN2)
    scale = k.shape[-1] ** -0.5
    Aqk, Akk = chunk_kda_fwd_intra_pinned(q = q, k = k, gk = g, beta = beta, scale = scale, chunk_size = 64)
    return g, scale, Aqk, Akk


def _run_hip(kda, q, k, v, g, beta, scale, Aqk, Akk, h0, ht):
    HH = kda[4]
    o = torch.empty_like(v)
    s = k.new_empty(q.shape[0], v.shape[2], q.shape[-1], v.shape[-1], dtype = torch.float32) if ht else None
    HH.chunk_kda_fwd_h_fused_hip(q, k, v, g, beta, Akk, Aqk, scale, initial_state = h0,
                                 output_final_state = ht, chunk_size = 64, o = o, final_state = s)
    return o, s


@pytest.mark.parametrize("T", T_ALL)
@pytest.mark.parametrize("B", (1, 2))
@pytest.mark.parametrize("use_h0", (True, False))
@pytest.mark.parametrize("store_ht", (True, False))
def test_bitwise_vs_triton(dev, kda, T, B, use_h0, store_ht):
    KH = kda[3]
    q, k, v, g_raw, beta, h0 = _synth(dev, B, T, seed = T + B)
    h0 = h0 if use_h0 else None
    g, scale, Aqk, Akk = _inputs(kda, q, k, v, g_raw, beta)

    _, _, ref_o, ref_s = KH.chunk_kda_fwd_h_fused(
        q, k, v, g, beta, Akk, Aqk, scale, initial_state = h0,
        output_final_state = store_ht, chunk_size = 64, fuse_o = True)

    o, s = _run_hip(kda, q, k, v, g, beta, scale, Aqk, Akk, h0, store_ht)
    assert torch.isfinite(o).all()
    assert torch.equal(o, ref_o)
    if store_ht:
        assert s is not None and torch.equal(s, ref_s)
    else:
        assert s is None

    o2, s2 = _run_hip(kda, q, k, v, g, beta, scale, Aqk, Akk, h0, store_ht)
    assert torch.equal(o, o2)
    if store_ht:
        assert torch.equal(s, s2)


@pytest.mark.parametrize("chunks", [(2, 5, 8, 11), tuple(range(1, 9))], ids = ("CK4", "CK8"))
def test_checkpoint_states(dev, kda, chunks):
    HH, KH = kda[4], kda[3]
    T = 777
    q, k, v, g_raw, beta, h0 = _synth(dev, 1, T, seed = 17)
    g, scale, Aqk, Akk = _inputs(kda, q, k, v, g_raw, beta)
    # Independent oracle: the Triton fused loop's per-chunk entry state. h[:, n] (0-indexed
    # chunk n) is the state entering chunk n, exactly the state after n chunks that CK chunk n
    # side-writes. h is bf16 here, so compare with a tolerance; a wrong state-row or checkpoint
    # mapping is still O(1) off.
    href, _, _, _ = KH.chunk_kda_fwd_h_fused(
        q, k, v, g, beta, Akk, Aqk, scale, initial_state = h0,
        output_final_state = False, chunk_size = 64, fuse_o = False)
    cks = [{"s": torch.zeros(1, v.shape[2], q.shape[-1], v.shape[-1], dtype = torch.float32, device = dev),
            "chunk": n} for n in chunks]
    o = torch.empty_like(v)
    final = k.new_empty(1, v.shape[2], q.shape[-1], v.shape[-1], dtype = torch.float32)
    HH.chunk_kda_fwd_h_fused_hip(q, k, v, g, beta, Akk, Aqk, scale, initial_state = h0,
                                 output_final_state = True, chunk_size = 64, o = o, final_state = final, ckpt = cks)
    assert torch.isfinite(o).all() and torch.isfinite(final).all()
    for c in cks:
        assert c.get("ok") is True
        ref = href[:, c["chunk"]].float()
        torch.testing.assert_close(c["s"], ref, rtol = 5e-2, atol = 1e-2,
                                   msg = f"checkpoint chunk {c['chunk']} vs Triton entry state")
