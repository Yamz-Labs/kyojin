"""Native ROCm parity and routing coverage for fused hyperconnection kernels."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import hyperconnections as hc

IS_GFX12 = bool(
    torch.version.hip
    and torch.cuda.is_available()
    and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":", 1)[0]
    in ("gfx1200", "gfx1201")
)
pytestmark = pytest.mark.skipif(not IS_GFX12, reason="gfx12 ROCm native kernel tests")
DEVICE = "cuda:0"
H = 4


def _require_native_symbols():
    for name in ("hc_mix_supported", "hc_mix_num_chunks", "hc_mix", "hc_head", "hc_apply", "gr_mix"):
        assert hasattr(ext, name), f"ROCm extension is missing native {name} binding"


def _gated_residual(d: int, rank: int, use_combine: bool):
    module = object.__new__(hc.GatedResidual)
    module.hc_mult = H
    module.hidden_size = d
    module.rms_eps = 1e-5
    module.use_combine = use_combine
    module.out_dtype = None
    module.norm_w_raw = (torch.randn((H, d), device=DEVICE) * 0.15).half()
    module.norm_w = module.w_h = None
    module.down_h = module.up_h = module.upx_h = None
    module.inject_h = module.proj_h = module.fn_h = None
    module.rank = 0
    down = (torch.randn((rank, H * d), device=DEVICE) * 0.08).half()
    up = (torch.randn((H * d, rank), device=DEVICE) * 0.08).half()
    inject = (torch.randn((H, H * d), device=DEVICE) * 0.08).half() if use_combine else None
    module._prepare(down, up, inject)
    return module


@pytest.mark.parametrize("rows", [1, 7, 32])
@pytest.mark.parametrize("use_combine", [True, False], ids=["site", "final"])
def test_gr_mix_matches_torch_reference_fp16(rows, use_combine):
    _require_native_symbols()
    torch.manual_seed(1201 + rows + use_combine)
    d, rank = 2560, 320
    module = _gated_residual(d, rank, use_combine)
    streams = torch.randn((1, rows, H, d), device=DEVICE) * 0.25
    expected_post, expected_mixed = module._mix_ref(streams)

    dots = torch.empty((rows, module.fn_h.shape[0] + 1, H), dtype=torch.float, device=DEVICE)
    post = torch.empty((rows, H), dtype=torch.float, device=DEVICE) if use_combine else None
    mixed = torch.empty((rows, d), dtype=torch.half, device=DEVICE)
    ext.gr_mix(streams.view(rows, H, d), module.fn_h, module.upx_h, module.w_h,
               module.rms_eps, dots, post, mixed)

    assert mixed.dtype == torch.float16
    torch.testing.assert_close(mixed, expected_mixed.view(rows, d).half(), rtol=2e-3, atol=2e-3)
    if use_combine:
        torch.testing.assert_close(post, expected_post.view(rows, H), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("rows", [1, 7])
@pytest.mark.parametrize("y_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("with_comb", [False, True])
def test_hc_apply_matches_reference_and_updates_in_place(rows, y_dtype, with_comb):
    _require_native_symbols()
    torch.manual_seed(2201 + rows + with_comb)
    d = 68
    x = torch.randn((rows, H, d), device=DEVICE)
    original = x.clone()
    y = torch.randn((rows, d), device=DEVICE, dtype=y_dtype)
    post = torch.randn((rows, H), device=DEVICE)
    comb = torch.randn((rows, H, H), device=DEVICE) if with_comb else None
    if comb is None:
        expected = original + post.unsqueeze(-1) * y.float().unsqueeze(1)
    else:
        expected = post.unsqueeze(-1) * y.float().unsqueeze(1) + torch.matmul(
            comb.transpose(-1, -2), original
        )
    ptr = x.data_ptr()

    result = ext.hc_apply(x, y, post, comb)

    assert result is None
    assert x.data_ptr() == ptr
    torch.testing.assert_close(x, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("rows", [1, 7])
def test_hc_mix_matches_torch_reference_fp16(rows):
    _require_native_symbols()
    torch.manual_seed(3201 + rows)
    d, m = 64, 2 * H + H * H
    streams = torch.randn((rows, H, d), device=DEVICE) * 0.25
    fn = (torch.randn((m, H * d), device=DEVICE) * 0.05).half()
    base = torch.randn((m,), device=DEVICE) * 0.1
    scale = torch.randn((3,), device=DEVICE) * 0.2
    rms_eps, hc_eps, iters = 1e-5, 1e-5, 4
    flat = streams.flatten(1)
    normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    mix = F.linear(normed, fn.float())
    pre = torch.sigmoid(mix[:, :H] * scale[0] + base[:H]) + hc_eps
    expected_post = 2 * torch.sigmoid(mix[:, H:2 * H] * scale[1] + base[H:2 * H])
    expected_comb = torch.softmax(
        (mix[:, 2 * H:] * scale[2] + base[2 * H:]).view(rows, H, H), dim=-1
    ) + hc_eps
    expected_comb = expected_comb / (expected_comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        expected_comb = expected_comb / (expected_comb.sum(dim=-1, keepdim=True) + hc_eps)
        expected_comb = expected_comb / (expected_comb.sum(dim=-2, keepdim=True) + hc_eps)
    expected_collapsed = (pre.unsqueeze(-1) * streams).sum(1).half()
    chunks = ext.hc_mix_num_chunks(rows, H * d)
    partials = torch.empty((rows, chunks, m + 1), device=DEVICE)
    post = torch.empty((rows, H), device=DEVICE)
    comb = torch.empty((rows, H, H), device=DEVICE)
    collapsed = torch.empty((rows, d), dtype=torch.half, device=DEVICE)

    ext.hc_mix(streams, fn, base, scale, rms_eps, hc_eps, iters,
               partials, post, comb, collapsed)

    torch.testing.assert_close(post, expected_post, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(comb, expected_comb, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(collapsed, expected_collapsed, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("rows", [1, 7])
def test_hc_head_matches_torch_reference_fp16(rows):
    _require_native_symbols()
    torch.manual_seed(4201 + rows)
    d = 64
    streams = torch.randn((rows, H, d), device=DEVICE) * 0.25
    fn = (torch.randn((H, H * d), device=DEVICE) * 0.05).half()
    base = torch.randn((H,), device=DEVICE) * 0.1
    scale = torch.randn((1,), device=DEVICE) * 0.2
    rms_eps, hc_eps = 1e-5, 1e-5
    flat = streams.flatten(1)
    normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(F.linear(normed, fn.float()) * scale + base) + hc_eps
    expected = (pre.unsqueeze(-1) * streams).sum(1).half()
    chunks = ext.hc_mix_num_chunks(rows, H * d)
    partials = torch.empty((rows, chunks, H + 1), device=DEVICE)
    collapsed = torch.empty((rows, d), dtype=torch.half, device=DEVICE)

    ext.hc_head(streams, fn, base, scale, rms_eps, hc_eps, partials, collapsed)

    torch.testing.assert_close(collapsed, expected, rtol=2e-3, atol=2e-3)


def test_gated_residual_module_routes_through_native_mix_and_apply(monkeypatch):
    _require_native_symbols()
    torch.manual_seed(5201)
    d, rank = 64, 17
    module = _gated_residual(d, rank, True)
    streams = torch.randn((1, 3, H, d), device=DEVICE) * 0.25
    calls = {"gr_mix": 0, "hc_apply": 0}
    real_gr_mix, real_hc_apply = ext.gr_mix, ext.hc_apply

    def gr_mix_spy(*args, **kwargs):
        calls["gr_mix"] += 1
        return real_gr_mix(*args, **kwargs)

    def hc_apply_spy(*args, **kwargs):
        calls["hc_apply"] += 1
        return real_hc_apply(*args, **kwargs)

    monkeypatch.setattr(
        hc,
        "ext",
        SimpleNamespace(
            hc_mix_supported=lambda _device: True,
            gr_mix=gr_mix_spy,
            hc_apply=hc_apply_spy,
        ),
    )
    expected_post, expected_mixed = module._mix_ref(streams)
    post, _, mixed = module.mix(streams, {})
    y = torch.randn((1, 3, d), device=DEVICE, dtype=torch.half)
    original = streams.clone()
    result = module.apply_(streams, y, post, None, {})

    assert calls == {"gr_mix": 1, "hc_apply": 1}
    assert result.data_ptr() == streams.data_ptr()
    torch.testing.assert_close(post, expected_post, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(mixed, expected_mixed.half(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(result, original + post.unsqueeze(-1) * y.float().unsqueeze(-2),
                               rtol=2e-5, atol=2e-5)


def _mix_reference(streams, fn, base, scale, rms_eps, hc_eps, iters=None):
    rows, streams_h, _ = streams.shape
    flat = streams.flatten(1)
    normed = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    mix = F.linear(normed, fn.float())
    pre = torch.sigmoid(mix[:, :streams_h] * scale[0] + base[:streams_h]) + hc_eps
    collapsed = (pre.unsqueeze(-1) * streams).sum(1)
    if iters is None:
        return collapsed
    post = 2 * torch.sigmoid(
        mix[:, streams_h:2 * streams_h] * scale[1] + base[streams_h:2 * streams_h]
    )
    comb = torch.softmax(
        (mix[:, 2 * streams_h:] * scale[2] + base[2 * streams_h:]).view(
            rows, streams_h, streams_h
        ),
        dim=-1,
    ) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    return post, comb, collapsed


def _native_mix_tensors(rows=2, d=64, *, head=False, fn_dtype=torch.float16,
                        out_dtype=torch.float16):
    m = H if head else 2 * H + H * H
    streams = torch.randn((rows, H, d), device=DEVICE)
    fn = torch.randn((m, H * d), device=DEVICE, dtype=fn_dtype)
    base = torch.randn((m,), device=DEVICE)
    scale = torch.randn((1 if head else 3,), device=DEVICE)
    chunks = ext.hc_mix_num_chunks(rows, H * d)
    partials = torch.empty((rows, chunks, m + 1), device=DEVICE)
    collapsed = torch.empty((rows, d), dtype=out_dtype, device=DEVICE)
    if head:
        return streams, fn, base, scale, partials, collapsed
    post = torch.empty((rows, H), device=DEVICE)
    comb = torch.empty((rows, H, H), device=DEVICE)
    return streams, fn, base, scale, partials, post, comb, collapsed


def test_gfx12_reports_hc_mix_supported():
    _require_native_symbols()
    assert ext.hc_mix_supported(torch.cuda.current_device()) is True


@pytest.mark.parametrize("head", [False, True], ids=["mix", "head"])
def test_real_dimension_multi_partial_float_fn_and_output(head):
    _require_native_symbols()
    torch.manual_seed(6201 + head)
    rows, d = 3, 2560
    args = _native_mix_tensors(rows, d, head=head, fn_dtype=torch.float32,
                               out_dtype=torch.float32)
    streams, fn, base, scale, partials = args[:5]
    assert partials.size(1) > 1
    if head:
        collapsed = args[5]
        expected = _mix_reference(streams, fn, base, scale, 1e-5, 1e-5)
        ext.hc_head(streams, fn, base, scale, 1e-5, 1e-5, partials, collapsed)
        torch.testing.assert_close(collapsed, expected, rtol=3e-4, atol=3e-4)
    else:
        post, comb, collapsed = args[5:]
        expected_post, expected_comb, expected_collapsed = _mix_reference(
            streams, fn, base, scale, 1e-5, 1e-5, 4
        )
        ext.hc_mix(streams, fn, base, scale, 1e-5, 1e-5, 4,
                   partials, post, comb, collapsed)
        torch.testing.assert_close(post, expected_post, rtol=3e-4, atol=3e-4)
        torch.testing.assert_close(comb, expected_comb, rtol=3e-4, atol=3e-4)
        torch.testing.assert_close(collapsed, expected_collapsed, rtol=3e-4, atol=3e-4)


def test_native_launch_uses_current_custom_stream_after_producer():
    _require_native_symbols()
    d = 64
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.empty((5, H, d), device=DEVICE)
        y = torch.empty((5, d), dtype=torch.half, device=DEVICE)
        post = torch.empty((5, H), device=DEVICE)
        x.normal_()
        y.normal_()
        post.normal_()
        original = x.clone()
        expected = original + post.unsqueeze(-1) * y.float().unsqueeze(1)
        ext.hc_apply(x, y, post, None)
    stream.synchronize()
    torch.testing.assert_close(x, expected, rtol=2e-5, atol=2e-5)


def test_wrappers_reject_invalid_dtype_shape_device_and_alignment():
    _require_native_symbols()
    mix_args = list(_native_mix_tensors())
    with pytest.raises(RuntimeError, match="collapsed must have dtype float32 or float16"):
        ext.hc_mix(*mix_args[:4], 1e-5, 1e-5, 4, mix_args[4], mix_args[5], mix_args[6],
                   torch.empty_like(mix_args[7], dtype=torch.bfloat16))
    with pytest.raises(RuntimeError, match="fn must have shape"):
        ext.hc_mix(mix_args[0], mix_args[1][:H], *mix_args[2:4], 1e-5, 1e-5, 4,
                   *mix_args[4:])
    with pytest.raises(RuntimeError, match="partials must have shape"):
        ext.hc_mix(*mix_args[:4], 1e-5, 1e-5, 4, mix_args[4][:, :0], *mix_args[5:])

    head_args = list(_native_mix_tensors(head=True))
    with pytest.raises(RuntimeError, match="fn must be a CUDA tensor"):
        ext.hc_head(head_args[0], head_args[1].cpu(), *head_args[2:4], 1e-5, 1e-5,
                    *head_args[4:])
    noncontiguous_base = torch.empty((H, 2), device=DEVICE)[:, 0]
    with pytest.raises(RuntimeError, match="base must be contiguous"):
        ext.hc_head(head_args[0], head_args[1], noncontiguous_base, head_args[3],
                    1e-5, 1e-5, *head_args[4:])
    offset_streams = torch.empty((head_args[0].numel() + 1,), device=DEVICE)[1:].view_as(
        head_args[0]
    )
    with pytest.raises(RuntimeError, match="streams must be 16-byte aligned"):
        ext.hc_head(offset_streams, *head_args[1:4], 1e-5, 1e-5, *head_args[4:])

    x = torch.randn((2, H, 64), device=DEVICE)
    y = torch.randn((2, 64), device=DEVICE)
    post = torch.randn((2, H), device=DEVICE)
    with pytest.raises(RuntimeError, match="y must have dtype float32 or float16"):
        ext.hc_apply(x, y.bfloat16(), post, None)
    with pytest.raises(RuntimeError, match="post must have shape"):
        ext.hc_apply(x, y, torch.randn((2, 3), device=DEVICE), None)
    offset_y = torch.empty((y.numel() + 1,), device=DEVICE)[1:].view_as(y)
    with pytest.raises(RuntimeError, match="y must be 16-byte aligned"):
        ext.hc_apply(x, offset_y, post, None)

    module = _gated_residual(64, 17, True)
    streams = torch.randn((2, H, 64), device=DEVICE)
    dots = torch.empty((2, module.fn_h.shape[0] + 1, H), device=DEVICE)
    gr_post = torch.empty((2, H), device=DEVICE)
    mixed = torch.empty((2, 64), dtype=torch.half, device=DEVICE)
    with pytest.raises(RuntimeError, match="mixed must have dtype float32 or float16"):
        ext.gr_mix(streams, module.fn_h, module.upx_h, module.w_h, 1e-5, dots, gr_post,
                   mixed.bfloat16())
    with pytest.raises(RuntimeError, match="upt must have shape"):
        ext.gr_mix(streams, module.fn_h, module.upx_h[:, :-1].contiguous(), module.w_h, 1e-5, dots,
                   gr_post, mixed)
    offset_fn = torch.empty((module.fn_h.numel() + 1,), dtype=torch.half,
                            device=DEVICE)[1:].view_as(module.fn_h)
    with pytest.raises(RuntimeError, match="fn must be 16-byte aligned"):
        ext.gr_mix(streams, offset_fn, module.upx_h, module.w_h, 1e-5, dots, gr_post, mixed)


def test_zero_rows_still_validate_dtype_before_return():
    _require_native_symbols()
    args = _native_mix_tensors(0, 64)
    with pytest.raises(RuntimeError, match="collapsed must have dtype float32 or float16"):
        ext.hc_mix(*args[:4], 1e-5, 1e-5, 4, *args[4:7],
                   torch.empty((0, 64), dtype=torch.bfloat16, device=DEVICE))


@pytest.mark.parametrize(
    "wrapper, d",
    [(wrapper, d) for wrapper in ("mix", "head", "apply", "gr") for d in (4, 64)
     if wrapper != "gr" or d % 8 == 0],
)
def test_zero_rows_validate_then_return_without_launch(wrapper, d):
    _require_native_symbols()
    if wrapper == "mix":
        args = _native_mix_tensors(0, d)
        ext.hc_mix(*args[:4], 1e-5, 1e-5, 4, *args[4:])
    elif wrapper == "head":
        args = _native_mix_tensors(0, d, head=True, out_dtype=torch.float32)
        ext.hc_head(*args[:4], 1e-5, 1e-5, *args[4:])
    elif wrapper == "apply":
        ext.hc_apply(torch.empty((0, H, d), device=DEVICE),
                     torch.empty((0, d), dtype=torch.half, device=DEVICE),
                     torch.empty((0, H), device=DEVICE), None)
    else:
        module = _gated_residual(d, 17, True)
        ext.gr_mix(
            torch.empty((0, H, d), device=DEVICE), module.fn_h, module.upx_h, module.w_h,
            1e-5, torch.empty((0, module.fn_h.shape[0] + 1, H), device=DEVICE),
            torch.empty((0, H), device=DEVICE),
            torch.empty((0, d), dtype=torch.half, device=DEVICE),
        )


def test_small_dimensions_have_nonzero_launch_chunks():
    _require_native_symbols()
    for head in (False, True):
        args = _native_mix_tensors(1, 4, head=head)
        assert args[4].size(1) == 1
        if head:
            ext.hc_head(*args[:4], 1e-5, 1e-5, *args[4:])
        else:
            ext.hc_mix(*args[:4], 1e-5, 1e-5, 4, *args[4:])
        assert torch.isfinite(args[-1]).all()

    x = torch.randn((1, H, 4), device=DEVICE)
    y = torch.randn((1, 4), dtype=torch.half, device=DEVICE)
    post = torch.randn((1, H), device=DEVICE)
    ext.hc_apply(x, y, post, None)
    assert torch.isfinite(x).all()


def test_gated_residual_row_33_uses_gemm_on_unsupported_wave(monkeypatch):
    _require_native_symbols()
    module = _gated_residual(64, 17, True)
    streams = torch.randn((1, 33, H, 64), device=DEVICE)

    def unexpected_fused_path(*_args, **_kwargs):
        raise AssertionError("R=33 must use the GEMM path")

    rms_norm_calls = 0

    def rms_norm_spy(*args, **kwargs):
        nonlocal rms_norm_calls
        rms_norm_calls += 1
        return ext.rms_norm(*args, **kwargs)

    monkeypatch.setattr(module, "_mix_ref", unexpected_fused_path)
    monkeypatch.setattr(
        hc,
        "ext",
        SimpleNamespace(
            hc_mix_supported=lambda _device: False,
            gr_mix=unexpected_fused_path,
            rms_norm=rms_norm_spy,
        ),
    )
    post, mixed = module._mix(streams)
    assert rms_norm_calls == 1
    assert post.shape == (33, H)
    assert mixed.shape == (33, 64)
    assert torch.isfinite(post).all() and torch.isfinite(mixed).all()
