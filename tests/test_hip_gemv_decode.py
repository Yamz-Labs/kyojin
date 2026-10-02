import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant import exl3 as exl3_quant

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

# -----------------------------------------------------------------------------------
# ROCm HIP EXL3 decode (GEMV) correctness oracle.
#
# Rationale: on a ROCm-only box (no CUDA GPU available) the HIP decode GEMV cannot be
# diffed against a CUDA reference. Instead we use the engine's own known-correct path as
# the oracle: the `reconstruct=True` path (PyTorch reconstruct of EXL3 weights + rocBLAS
# fp16 GEMM) is the reference, and the `reconstruct=False` path (which the HIP GEMV serves)
# must reproduce it to fp16 tolerance. This is the same comparison pattern test_qgemm.py
# applies to its two internal paths, applied at decode batch sizes (m <= 16, the MMODE-0/1/2
# regimes) where the GEMV path is the entire story.
#
# ROUTE PROOF (hard requirement): comparing reconstruct=False against reconstruct=True is
# only meaningful if reconstruct=False actually ran the HIP GEMV. On a build where the GEMV
# is not wired, both sides silently take the same reconstruct path and the comparison
# passes vacuously. Therefore every comparison case spies on the extension bindings and
# asserts:
#   * ext.exl3_gemv exists (FAIL, never skip, if the binding is absent), and
#   * it was called for reconstruct=False, and
#   * ext.reconstruct / ext.reconstruct_slice were NOT called (no silent fallback).
#
# Skips cleanly when there is no HIP build / ROCm device / test model present.
#
# Model: set EXL3_TEST_MODEL to a compatible Qwen3.8-27B EXL3 model directory.
# -----------------------------------------------------------------------------------

# Decode regime: the m==1 fast path plus both small-m and 16-row HIP GEMV modes.
# m==1 is autoregressive decode.
BATCH_SIZES = [1, 2, 8, 9, 12, 16]

# Per-arch proj keys exercising decode-relevant shapes. Keys must match the loaded
# model's module names (Qwen3.8 hybrid: "model.language_model" prefix; full-attention
# layers are 3, 7, ...; layer 0 is linear_attention).
TEST_KEYS = [
    ("model.language_model.layers.3.self_attn.q_proj", "model.language_model.layers.3.input_layernorm"),
    ("model.language_model.layers.3.self_attn.k_proj", "model.language_model.layers.3.input_layernorm"),
    ("model.language_model.layers.3.self_attn.v_proj", "model.language_model.layers.3.input_layernorm"),
    ("model.language_model.layers.3.self_attn.o_proj", None),   # in = heads*head_dim, not the hidden norm output
    ("model.language_model.layers.0.mlp.up_proj", "model.language_model.layers.0.post_attention_layernorm"),
    ("model.language_model.layers.0.mlp.gate_proj", "model.language_model.layers.0.post_attention_layernorm"),
    ("model.language_model.layers.0.mlp.down_proj", None),
    ("model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.input_layernorm"),
]

# The default Qwen3.8 fixture uses the mul1 codebook. Set EXL3_TEST_MODEL to the
# Qwen3.5-9B fixture to exercise the MCG codebook with the same oracle matrix.
DEFAULT_MODEL = os.path.expanduser("~/Models/Qwen3.8-27B-exl3")
MODEL = os.environ.get("EXL3_TEST_MODEL", DEFAULT_MODEL)


def _roc_available():
    # ROCm build active and at least one visible device
    if not (torch.version.hip and torch.cuda.is_available()):
        return False
    return True


def _require_gfx12(device=0):
    if not _roc_available():
        pytest.skip("ROCm build / device not available")
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    if arch.split(":", 1)[0] not in ("gfx1200", "gfx1201"):
        pytest.skip(f"HIP GEMV oracle requires gfx1200/gfx1201, got {arch or 'unknown'}")
    assert hasattr(ext, "exl3_gemv"), \
        "gfx12 target build is missing the required ext.exl3_gemv binding"
    assert hasattr(ext, "exl3_gemv_supported"), \
        "gfx12 target build is missing the required ext.exl3_gemv_supported binding"
    assert ext.exl3_gemv_supported(device), \
        f"ext.exl3_gemv_supported rejected gfx12 device {device} ({arch})"


@pytest.fixture(scope="module")
def model():
    _require_gfx12()
    if not os.path.isdir(MODEL):
        pytest.skip(f"Test model not found: {MODEL} (set EXL3_TEST_MODEL)")
    config = Config.from_directory(MODEL)
    return Model.from_config(config)


def _spy_ext_route():
    """Temporarily wrap ext.exl3_gemv / ext.reconstruct / ext.reconstruct_slice /
    ext.reconstruct_had_slice / ext.hgemm / ext.hgemm_recon with call-counting spies so a
    silent fallback into ANY reconstruction path cannot slip through. hgemm and
    hgemm_recon share the same counter (reconstruct_hgemm may call either). Returns
    (calls, restore)."""
    calls = {"gemv": 0, "reconstruct": 0, "reconstruct_slice": 0, "reconstruct_had_slice": 0, "hgemm": 0}
    real_gemv = ext.exl3_gemv
    real_reconstruct = ext.reconstruct
    real_reconstruct_slice = getattr(ext, "reconstruct_slice", None)
    real_reconstruct_had_slice = getattr(ext, "reconstruct_had_slice", None)
    real_hgemm = ext.hgemm
    real_hgemm_recon = getattr(ext, "hgemm_recon", None)

    def gemv_spy(*args, **kwargs):
        calls["gemv"] += 1
        return real_gemv(*args, **kwargs)

    def reconstruct_spy(*args, **kwargs):
        calls["reconstruct"] += 1
        return real_reconstruct(*args, **kwargs)

    def reconstruct_slice_spy(*args, **kwargs):
        calls["reconstruct_slice"] += 1
        return real_reconstruct_slice(*args, **kwargs)

    def reconstruct_had_slice_spy(*args, **kwargs):
        calls["reconstruct_had_slice"] += 1
        return real_reconstruct_had_slice(*args, **kwargs)

    def hgemm_spy(*args, **kwargs):
        calls["hgemm"] += 1
        return real_hgemm(*args, **kwargs)

    def hgemm_recon_spy(*args, **kwargs):
        calls["hgemm"] += 1
        return real_hgemm_recon(*args, **kwargs)

    ext.exl3_gemv = gemv_spy
    ext.reconstruct = reconstruct_spy
    if real_reconstruct_slice is not None:
        ext.reconstruct_slice = reconstruct_slice_spy
    if real_reconstruct_had_slice is not None:
        ext.reconstruct_had_slice = reconstruct_had_slice_spy
    ext.hgemm = hgemm_spy
    if real_hgemm_recon is not None:
        ext.hgemm_recon = hgemm_recon_spy

    def restore():
        ext.exl3_gemv = real_gemv
        ext.reconstruct = real_reconstruct
        if real_reconstruct_slice is not None:
            ext.reconstruct_slice = real_reconstruct_slice
        if real_reconstruct_had_slice is not None:
            ext.reconstruct_had_slice = real_reconstruct_had_slice
        ext.hgemm = real_hgemm
        if real_hgemm_recon is not None:
            ext.hgemm_recon = real_hgemm_recon

    return calls, restore


# The livescope model fixture used by the comparison test loads individual modules and MUST
# unload them even when an assertion fails, or aggregate modules stay resident for later tests.
def _load_then_compare(model, linear_key, norm_key, batch_size, out_dtype=None, expected_K=None):
    # Route proof: a missing binding is a FAILURE, not a skip -- the whole point of this
    # suite is to prove the HIP GEMV route executes.
    assert hasattr(ext, "exl3_gemv"), \
        "ext.exl3_gemv is missing: the HIP decode GEMV is not wired into the extension"

    norm = model.find_module(norm_key) if norm_key else None
    linear = model.find_module(linear_key)
    try:
        if norm:
            norm.load(device="cuda")
        linear.load(device="cuda")
        if expected_K is not None:
            assert linear.inner.K == expected_K
            assert linear.inner.mcg != linear.inner.mul1

        torch.manual_seed(0)
        x = torch.randn((1, batch_size, linear.in_features), dtype=torch.float16, device="cuda")
        if norm:
            x = norm.forward(x, {})

        # Spied GEMV forward: proves the route and forbids a silent reconstruct fallback
        calls, restore = _spy_ext_route()
        try:
            x_gemv = linear.forward(x, {"reconstruct": False}, out_dtype)
        finally:
            restore()
        assert calls["gemv"] > 0, \
            f"{linear_key} (m={batch_size}): reconstruct=False did not call ext.exl3_gemv"
        assert calls["reconstruct"] == 0, \
            f"{linear_key} (m={batch_size}): reconstruct=False silently fell back to ext.reconstruct"
        assert calls["reconstruct_slice"] == 0, \
            f"{linear_key} (m={batch_size}): reconstruct=False silently fell back to ext.reconstruct_slice"
        assert calls["reconstruct_had_slice"] == 0, \
            f"{linear_key} (m={batch_size}): reconstruct=False silently fell back to ext.reconstruct_had_slice"
        assert calls["hgemm"] == 0, \
            f"{linear_key} (m={batch_size}): reconstruct=False silently reached ext.hgemm (reconstruct fallback)"

        # Reference forward, unspied
        x_ref = linear.forward(x, {"reconstruct": True}, out_dtype)

        # The complete gfx1201 matrix, including K6 vocabulary heads, observed a
        # 0.0078125 maximum absolute deviation. Keep the bound tight enough to catch
        # extraction errors without inheriting the upstream suite's 0.05 tolerance.
        tol = 0.01
        torch.testing.assert_close(x_gemv, x_ref, rtol=tol, atol=tol)

        # Report-grade deviation metrics for reproducing the measured tolerance bound.
        d = (x_gemv.float() - x_ref.float()).abs()
        rel = d / x_ref.float().abs().clamp_min(1e-3)
        return d.max().item(), rel.max().item()
    finally:
        linear.unload()
        if norm:
            norm.unload()


@torch.inference_mode()
def test_hip_gemv_direct_binding_accepts_rank3_inputs():
    """The direct HIP binding must accept arbitrary-rank contiguous leading dims."""
    _require_gfx12()
    size_k = size_n = 128
    A = torch.randn((1, 1, size_k), dtype=torch.float16, device="cuda") * 1e-3
    B = torch.full((size_k // 16, size_n // 16, 4 * 16), 0x1111, dtype=torch.int16, device="cuda")
    suh = (torch.randint(0, 2, (size_k,), device="cuda") * 2 - 1).to(torch.float16)
    svh = (torch.randint(0, 2, (size_n,), device="cuda") * 2 - 1).to(torch.float16)

    actual = torch.empty((1, 1, size_n), dtype=torch.float16, device="cuda")
    A_had = torch.empty_like(A)
    ext.exl3_gemv(A, B, actual, suh, A_had, svh, False, False)

    A_ref = torch.empty_like(A)
    weight = torch.empty((size_k, size_n), dtype=torch.float16, device="cuda")
    expected = torch.empty_like(actual)
    ext.had_r_128(A.view(1, size_k), A_ref.view(1, size_k), suh, None, 1.0)
    ext.reconstruct(weight, B, 4, False, False)
    ext.hgemm(A_ref.view(1, size_k), weight, expected.view(1, size_n))
    ext.had_r_128(expected.view(1, size_n), expected.view(1, size_n), None, svh, 1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.03)


@torch.inference_mode()
def test_hip_gemv_k5_plain_codebook_is_rejected():
    """K5 requires the MCG or mul1 codebook; cb0 remains unsupported."""
    _require_gfx12()
    size = 128
    A = torch.randn((1, size), dtype=torch.float16, device="cuda") * 1e-3
    B = torch.full((size // 16, size // 16, 5 * 16), 0x1111,
                   dtype=torch.int16, device="cuda")
    signs = torch.ones((size,), dtype=torch.float16, device="cuda")
    actual = torch.empty((1, size), dtype=torch.float16, device="cuda")
    A_had = torch.empty_like(A)

    with pytest.raises(RuntimeError, match="not eligible"):
        ext.exl3_gemv(A, B, actual, signs, A_had, signs, False, False)


@pytest.mark.parametrize(
    "mcg, multiplier, addend",
    [
        pytest.param(False, 89226354, 64248484, id="plain"),
        pytest.param(True, 0xCBAC1FED, 0, id="mcg"),
    ],
)
@torch.inference_mode()
def test_procedural_codebook_matches_lop3_zero_state(mcg, multiplier, addend):
    """The portable codebook expression must match the original PTX lop3 LUT 0x6a.

    A zero packed trellis decodes every weight from state zero, making the expected
    fp16 codebook value independent of the trellis layout and reconstruction path.
    """
    _require_gfx12()
    K = 4
    packed = torch.zeros((8, 8, K * 16), dtype=torch.int16, device="cuda")
    weight = torch.empty((128, 128), dtype=torch.float16, device="cuda")
    ext.reconstruct(weight, packed, K, mcg, False)

    product = (0 * multiplier + addend) & 0xFFFFFFFF
    lop3 = 0x3B603B60 ^ (product & 0x8FFF8FFF)
    halves = torch.frombuffer(bytearray(lop3.to_bytes(4, "little")), dtype=torch.float16)
    expected = halves.sum(dtype=torch.float16)

    torch.testing.assert_close(weight, torch.full_like(weight, expected), rtol=0, atol=0)


@pytest.mark.parametrize(
    "K, mcg, mul1",
    [
        pytest.param(4, False, False, id="k4-plain"),
        pytest.param(4, True, False, id="k4-mcg"),
        pytest.param(4, False, True, id="k4-mul1"),
        pytest.param(2, True, False, id="k2-mcg"),
        pytest.param(2, False, True, id="k2-mul1"),
        pytest.param(3, True, False, id="k3-mcg"),
        pytest.param(3, False, True, id="k3-mul1"),
    ],
)
@torch.inference_mode()
def test_hip_gemv_synthetic_oracle(K, mcg, mul1):
    """Every HIP-instantiated bitrate/codebook family matches the reconstruct oracle."""
    _require_gfx12()
    torch.manual_seed(1234 + K + 10 * mcg + 100 * mul1)
    size_k = size_n = 128
    # These nonzero repeated cycles are valid finite trellises for their codebooks. Random
    # packed words are not valid test data because several intentionally decode to fp16 NaNs.
    packed_cycle = {
        (4, False, False): 0x1111,
        (4, True, False): 0x2222,
        (4, False, True): 0x1111,
        (2, True, False): 0x5555,
        (2, False, True): 0x5555,
        (3, True, False): 0x2492,
        (3, False, True): 0x2492,
    }[K, mcg, mul1]
    A = torch.randn((1, size_k), dtype=torch.float16, device="cuda") * 1e-3
    B = torch.full((size_k // 16, size_n // 16, K * 16), packed_cycle,
                   dtype=torch.int16, device="cuda")
    suh = (torch.randint(0, 2, (size_k,), device="cuda") * 2 - 1).to(torch.float16)
    svh = (torch.randint(0, 2, (size_n,), device="cuda") * 2 - 1).to(torch.float16)

    actual = torch.empty((1, size_n), dtype=torch.float16, device="cuda")
    A_had = torch.empty_like(A)
    ext.exl3_gemv(A, B, actual, suh, A_had, svh, mcg, mul1)

    A_ref = torch.empty_like(A)
    weight = torch.empty((size_k, size_n), dtype=torch.float16, device="cuda")
    expected = torch.empty_like(actual)
    ext.had_r_128(A, A_ref, suh, None, 1.0)
    ext.reconstruct(weight, B, K, mcg, mul1)
    ext.hgemm(A_ref, weight, expected)
    ext.had_r_128(expected, expected, None, svh, 1.0)

    # All seven cases were bit-identical on gfx1201. A 0.03 absolute allowance retains
    # modest fp16 rounding headroom without hiding an extraction/codebook mismatch.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.03)


@pytest.mark.parametrize(
    "K, mcg, mul1",
    [
        pytest.param(3, False, True, id="k3-mul1"),
        pytest.param(4, True, False, id="k4-mcg"),
        pytest.param(4, False, True, id="k4-mul1"),
        pytest.param(5, False, True, id="k5-mul1"),
        pytest.param(6, True, False, id="k6-mcg"),
        pytest.param(6, False, True, id="k6-mul1"),
    ],
)
@pytest.mark.parametrize("rows", [9, 12, 16], ids=["rows-9", "rows-12", "rows-16"])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@torch.inference_mode()
def test_hip_gemv_rows_9_to_16_synthetic_route_to_gemv(K, mcg, mul1, rows, out_dtype):
    _require_gfx12()
    actual, expected, calls, x, size_n = _run_rows_synthetic_oracle(K, mcg, mul1, rows, out_dtype)
    assert calls["gemv"] == 1
    assert calls["reconstruct"] == 0
    assert calls["reconstruct_slice"] == 0
    assert calls["reconstruct_had_slice"] == 0
    assert calls["hgemm"] == 0
    assert actual.shape == x.shape[:-1] + (size_n,)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.03)


@torch.inference_mode()
def test_hip_gemv_k2_rows_8_routes_to_gemv():
    _require_gfx12()
    actual, expected, calls, x, size_n = _run_rows_synthetic_oracle(2, True, False, 8, torch.float16)
    assert calls["gemv"] == 1
    assert calls["reconstruct"] == 0
    assert calls["reconstruct_slice"] == 0
    assert calls["reconstruct_had_slice"] == 0
    assert calls["hgemm"] == 0
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.03)


@torch.inference_mode()
def test_hip_gemv_k2_rows_9_falls_back():
    _require_gfx12()
    actual, expected, calls, x, size_n = _run_rows_synthetic_oracle(2, True, False, 9, torch.float16)
    assert calls["gemv"] == 0
    assert calls["reconstruct"] > 0
    assert calls["hgemm"] > 0
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.03)


def _make_distinct_finite_trellis(K, size_k, size_n, device="cuda"):
    """Build a unique, finite packed stream for every (K-slice, N-tile)."""
    kslices = size_k // 16
    ntiles = size_n // 16
    tile_id = torch.arange(kslices * ntiles, dtype=torch.int64).view(kslices, ntiles, 1)
    word = torch.arange(K * 16, dtype=torch.int64).view(1, 1, K * 16)
    mixed = tile_id * 1103515245 + word * 12345 + tile_id * word * 2654435761
    choices = ((mixed ^ (mixed >> 16)) >> 8) & 3
    # The first eight base-4 digits encode tile_id, guaranteeing distinct streams for
    # every shape in this suite while retaining only known-finite packed words.
    choices = torch.where(word < 8, (tile_id >> (2 * word)) & 3, choices)
    palette = torch.tensor([0x1111, 0x2222, 0x5555, 0x2492], dtype=torch.int16)
    return palette[choices].to(device)


def _assert_reconstructed_tiles_differ(weight):
    tiles = [weight[:16, :16], weight[16:32, :16], weight[:16, 16:32]]
    assert all(not torch.equal(a, b) for i, a in enumerate(tiles) for b in tiles[i + 1:]), \
        "representative reconstructed 16x16 tiles unexpectedly alias"


def _run_k5_synthetic_oracle(mcg, mul1, size_k, size_n, batch_size, out_dtype):
    from exllamav3.modules.quant.exl3 import LinearEXL3

    torch.manual_seed(5000 + size_k + size_n + 10 * mcg + 100 * mul1 + batch_size)
    K = 5
    trellis = _make_distinct_finite_trellis(K, size_k, size_n)
    suh = (torch.randint(0, 2, (size_k,), device="cuda") * 2 - 1).to(torch.float16)
    svh = (torch.randint(0, 2, (size_n,), device="cuda") * 2 - 1).to(torch.float16)
    codebook = torch.ones((), dtype=torch.int32, device="cuda")
    linear = LinearEXL3(None, size_k, size_n, suh=suh, svh=svh, trellis=trellis,
                        mcg=codebook if mcg else None, mul1=codebook if mul1 else None)
    x = torch.randn((1, batch_size, size_k), dtype=torch.float16, device="cuda") * 1e-3

    weight = torch.empty((size_k, size_n), dtype=torch.float16, device="cuda")
    ext.reconstruct(weight, trellis, K, mcg, mul1)
    assert torch.isfinite(weight).all()
    _assert_reconstructed_tiles_differ(weight)
    expected = linear.forward(x, {"reconstruct": True}, out_dtype)

    calls, restore = _spy_ext_route()
    try:
        actual = linear.forward(x, {"reconstruct": False}, out_dtype)
    finally:
        restore()
    assert calls["gemv"] == 1
    assert calls["reconstruct"] == 0
    assert calls["reconstruct_slice"] == 0
    assert calls["reconstruct_had_slice"] == 0
    assert calls["hgemm"] == 0
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.03)


def _run_rows_synthetic_oracle(K, mcg, mul1, rows, out_dtype=torch.float16):
    from exllamav3.modules.quant.exl3 import LinearEXL3

    _require_gfx12()
    torch.manual_seed(7000 + 10 * K + 100 * mcg + 1000 * mul1 + rows)
    size_k = size_n = 128
    trellis = _make_distinct_finite_trellis(K, size_k, size_n)
    suh = (torch.randint(0, 2, (size_k,), device="cuda") * 2 - 1).to(torch.float16)
    svh = (torch.randint(0, 2, (size_n,), device="cuda") * 2 - 1).to(torch.float16)
    codebook = torch.ones((), dtype=torch.int32, device="cuda")
    linear = None
    actual = expected = calls = x = None
    try:
        linear = LinearEXL3(None, size_k, size_n, suh=suh, svh=svh, trellis=trellis,
                            mcg=codebook if mcg else None, mul1=codebook if mul1 else None)
        x = torch.randn((1, rows, size_k), dtype=torch.float16, device="cuda") * 1e-3

        expected = linear.forward(x, {"reconstruct": True}, out_dtype)
        calls, restore = _spy_ext_route()
        try:
            actual = linear.forward(x, {"reconstruct": False}, out_dtype)
        finally:
            restore()
    finally:
        if linear is not None:
            linear.unload()
    return actual, expected, calls, x, size_n


@pytest.mark.parametrize("size_n", [128, 8320], ids=["cfg0-exact-tail", "cfg1"])
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@torch.inference_mode()
def test_hip_gemv_k5_synthetic_oracle(size_n, batch_size, out_dtype):
    """The K5 mul1/cb2 m1..16 fp16/fp32 matrix matches reconstruct+hgemm."""
    _require_gfx12()
    _run_k5_synthetic_oracle(False, True, 128, size_n, batch_size, out_dtype)


@pytest.mark.parametrize("size_n", [128, 8320], ids=["cfg0-exact-tail", "cfg1"])
@torch.inference_mode()
def test_hip_gemv_k5_prefetch_refill(size_n):
    """K5 mul1/cb2 remains correct after both launch configurations refill prefetch."""
    _require_gfx12()
    _run_k5_synthetic_oracle(False, True, 1152, size_n, 1, torch.float16)


@pytest.mark.parametrize("test_key", TEST_KEYS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@torch.inference_mode()
def test_hip_gemv_matches_reference(model, test_key, batch_size):
    """HIP decode GEMV (reconstruct=False) must match the reconstruct=True reference,
    and must actually route through the GEMV kernel to do it."""
    _load_then_compare(model, test_key[0], test_key[1], batch_size)


@torch.inference_mode()
def test_hip_gemv_route_only(model):
    """Focused m==1 route proof on one representative projection, independent of the
    numeric matrix: the GEMV binding fires and the reconstruct fallback does not."""
    assert hasattr(ext, "exl3_gemv"), "ext.exl3_gemv is missing"
    norm = model.find_module("model.language_model.layers.3.input_layernorm")
    linear = model.find_module("model.language_model.layers.3.self_attn.q_proj")
    try:
        norm.load(device="cuda")
        linear.load(device="cuda")
        torch.manual_seed(0)
        x = torch.randn((1, 1, linear.in_features), dtype=torch.float16, device="cuda")
        x = norm.forward(x, {})
        calls, restore = _spy_ext_route()
        try:
            y = linear.forward(x, {"reconstruct": False})
        finally:
            restore()
        assert calls["gemv"] == 1, f"expected exactly one exl3_gemv call, got {calls['gemv']}"
        assert calls["reconstruct"] == 0
        assert calls["reconstruct_slice"] == 0
        assert calls["reconstruct_had_slice"] == 0
        assert calls["hgemm"] == 0, "reconstruct=False must not reach ext.hgemm"
        assert y.shape[-1] == linear.out_features
        assert torch.isfinite(y.float()).all()
    finally:
        linear.unload()
        norm.unload()


@pytest.mark.parametrize("rows", [9, 12, 16])
@torch.inference_mode()
def test_hip_gemv_rows_9_to_16_route_to_gemv(model, rows):
    """The MMODE-2 rows must use GEMV rather than reconstruct+hgemm."""
    assert exl3_quant.EXL3_GEMV_HIP_MAX_M == 16
    linear = model.find_module("model.language_model.layers.3.self_attn.q_proj")
    try:
        linear.load(device="cuda")
        assert linear.inner.K == 4
        assert linear.inner.mcg != linear.inner.mul1
        x = torch.randn((1, rows, linear.in_features), dtype=torch.float16, device="cuda")
        calls, restore = _spy_ext_route()
        try:
            y = linear.forward(x, {"reconstruct": False})
        finally:
            restore()
        assert calls["gemv"] == 1, f"m={rows} must make exactly one GEMV call"
        assert calls["reconstruct"] == 0
        assert calls["reconstruct_slice"] == 0
        assert calls["reconstruct_had_slice"] == 0
        assert calls["hgemm"] == 0
        assert torch.isfinite(y.float()).all()
    finally:
        linear.unload()


@torch.inference_mode()
def test_hip_gemv_import_time_max_m_parser(monkeypatch):
    import importlib

    original = exl3_quant.EXL3_GEMV_HIP_MAX_M
    try:
        monkeypatch.delenv("EXL3_GEMV_HIP_MAX_M", raising=False)
        importlib.reload(exl3_quant)
        assert exl3_quant.EXL3_GEMV_HIP_MAX_M == 16

        monkeypatch.setenv("EXL3_GEMV_HIP_MAX_M", "8")
        importlib.reload(exl3_quant)
        assert exl3_quant.EXL3_GEMV_HIP_MAX_M == 8

        monkeypatch.setenv("EXL3_GEMV_HIP_MAX_M", "bogus")
        importlib.reload(exl3_quant)
        assert exl3_quant.EXL3_GEMV_HIP_MAX_M == 16

        monkeypatch.setenv("EXL3_GEMV_HIP_MAX_M", "0")
        importlib.reload(exl3_quant)
        assert exl3_quant.EXL3_GEMV_HIP_MAX_M == 1

        monkeypatch.setenv("EXL3_GEMV_HIP_MAX_M", "17")
        importlib.reload(exl3_quant)
        assert exl3_quant.EXL3_GEMV_HIP_MAX_M == 16
    finally:
        monkeypatch.delenv("EXL3_GEMV_HIP_MAX_M", raising=False)
        if original != 16:
            monkeypatch.setenv("EXL3_GEMV_HIP_MAX_M", str(original))
        importlib.reload(exl3_quant)


@torch.inference_mode()
def test_hip_gemv_rollback_cap_falls_back(model, monkeypatch):
    """The EXL3_GEMV_HIP_MAX_M rollback cap returns m=9 to reconstruct+hgemm."""
    import importlib

    assert exl3_quant.EXL3_GEMV_HIP_MAX_M == 16
    try:
        with monkeypatch.context() as rollback:
            rollback.setenv("EXL3_GEMV_HIP_MAX_M", "8")
            importlib.reload(exl3_quant)
            assert exl3_quant.EXL3_GEMV_HIP_MAX_M == 8
            linear = model.find_module("model.language_model.layers.3.self_attn.q_proj")
            try:
                linear.load(device="cuda")
                x = torch.randn((1, 9, linear.in_features), dtype=torch.float16, device="cuda")
                calls, restore = _spy_ext_route()
                try:
                    y = linear.forward(x, {"reconstruct": False})
                finally:
                    restore()
                assert calls["gemv"] == 0
                assert calls["reconstruct"] > 0 and calls["hgemm"] > 0
                assert torch.isfinite(y.float()).all()
            finally:
                linear.unload()
    finally:
        importlib.reload(exl3_quant)


@torch.inference_mode()
def test_hip_gemv_rows_over_limit_falls_back(model):
    """A K=4 decode batch with exactly seventeen rows must bypass HIP GEMV."""
    assert exl3_quant.EXL3_GEMV_HIP_MAX_M == 16
    linear = model.find_module("model.language_model.layers.3.self_attn.q_proj")
    try:
        linear.load(device="cuda")
        assert linear.inner.K == 4
        assert linear.inner.mcg != linear.inner.mul1
        x = torch.randn((1, 17, linear.in_features), dtype=torch.float16, device="cuda")
        calls, restore = _spy_ext_route()
        try:
            y = linear.forward(x, {"reconstruct": False})
        finally:
            restore()
        assert calls["gemv"] == 0, "m=17 must bypass the m <= 16 GEMV kernel"
        assert calls["reconstruct"] > 0 and calls["hgemm"] > 0
        assert torch.isfinite(y.float()).all()
    finally:
        linear.unload()


@torch.inference_mode()
def test_hip_gemv_env_opt_out_falls_back(model, monkeypatch):
    """EXL3_GEMV=0 forces the eligible K6 vocabulary head back to reconstruct+hgemm."""
    linear = model.find_module("lm_head")
    try:
        linear.load(device="cuda")
        assert linear.inner.K == 6
        x = torch.randn((1, 1, linear.in_features), dtype=torch.float16, device="cuda")
        monkeypatch.setenv("EXL3_GEMV", "0")
        calls, restore = _spy_ext_route()
        try:
            y = linear.forward(x, {"reconstruct": False})
        finally:
            restore()
        assert calls["gemv"] == 0
        reconstruct_calls = calls["reconstruct"] + calls["reconstruct_slice"] + calls["reconstruct_had_slice"]
        assert reconstruct_calls > 0 and calls["hgemm"] > 0
        assert torch.isfinite(y.float()).all()
    finally:
        linear.unload()


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32], ids=["fp16", "fp32"])
@torch.inference_mode()
def test_hip_gemv_k6_lm_head_matches_reference(model, batch_size, out_dtype):
    """Each model fixture's K6 vocabulary head routes directly and matches its oracle."""
    _load_then_compare(model, "lm_head", None, batch_size, out_dtype, expected_K=6)


@torch.inference_mode()
def test_hip_gemv_k7_falls_back():
    """K7 remains outside the HIP kernel family and uses reconstruct+hgemm."""
    from exllamav3.modules.quant.exl3 import LinearEXL3

    _require_gfx12()
    K = 7
    size = 128
    trellis = torch.full((size // 16, size // 16, K * 16), 0x1111,
                         dtype=torch.int16, device="cuda")
    signs = torch.ones((size,), dtype=torch.float16, device="cuda")
    mul1 = torch.tensor(0x83DCD12D, dtype=torch.uint32, device="cuda").view(torch.int32)
    linear = LinearEXL3(None, size, size, suh=signs, svh=signs,
                        trellis=trellis, mul1=mul1)
    x = torch.randn((1, 1, size), dtype=torch.float16, device="cuda")

    calls, restore = _spy_ext_route()
    try:
        y = linear.forward(x, {"reconstruct": False})
    finally:
        restore()

    assert calls["gemv"] == 0
    assert calls["reconstruct"] > 0 and calls["hgemm"] > 0
    assert torch.isfinite(y).all()


@torch.inference_mode()
def test_hip_decode_smoke(model):
    """One-token autoregressive decode sanity: greedy output runs the m==1 (MMODE0) decode path."""
    from exllamav3 import Cache, Tokenizer, Generator

    # Use a FRESH model instance so a full model.load() here cannot interleave with the
    # partially-loaded state left by the comparison test's shared fixture.
    if not os.path.isfile(os.path.join(MODEL, "tokenizer.json")):
        pytest.skip(f"No tokenizer.json in {MODEL} (set EXL3_TEST_MODEL)")
    fresh = Model.from_config(model.config)
    cache = Cache(fresh, max_num_tokens=2048, max_batch_size=1)  # must exist before model.load()
    fresh.load(device="cuda")
    tokenizer = Tokenizer.from_config(fresh.config)
    gen = Generator(model=fresh, cache=cache, tokenizer=tokenizer)
    calls, restore = _spy_ext_route()
    try:
        response = gen.generate(
            prompt="Q: The capital of France is\nA:",
            stop_conditions=[],
            max_new_tokens=8,
            completion_only=True,
            add_bos=True,
        )
    finally:
        restore()
        fresh.unload()
    # The point is that generation RAN through the HIP decode path without erroring.
    # A model may emit <eos> at the first token, so require only that a completion was
    # produced, not that it be non-trivial text. Ineligible layers such as the K=6
    # lm_head may reconstruct during sampling, but at least one decode projection must GEMV.
    assert response is not None
    assert calls["gemv"] > 0, "decode smoke test never called ext.exl3_gemv"
