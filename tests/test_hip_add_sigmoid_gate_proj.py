"""ROCm coverage for the fused shared-expert sigmoid projection gate."""
from __future__ import annotations

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


def _gfx12_devices():
    if not (torch.version.hip and torch.cuda.is_available()):
        return []
    return [
        index for index in range(torch.cuda.device_count())
        if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").split(":", 1)[0]
        in ("gfx1200", "gfx1201")
    ]


DEVICES = _gfx12_devices() or [None]


def _require_native(device_index):
    if device_index is None:
        pytest.skip("ROCm gfx12 device unavailable")
    assert getattr(ext.add_sigmoid_gate_proj, "__module__", None) == "exllamav3_ext"


def _case(shape, device_index, *, offset=False):
    device = torch.device("cuda", device_index)

    def make(shape, dtype):
        if not offset:
            return torch.randn(shape, dtype=dtype, device=device)
        numel = torch.Size(shape).numel()
        storage = torch.empty(numel + 1, dtype=dtype, device=device)
        view = storage[1:].view(shape)
        view.normal_()
        assert view.is_contiguous()
        return view

    x = make(shape, torch.float32)
    y = make(shape, torch.float16)
    z = make(shape, torch.float32)
    w = make((shape[-1], 1), torch.float16)
    return x, y, z, w


@pytest.mark.parametrize("device_index", DEVICES)
@pytest.mark.parametrize("shape", [(0, 32), (1, 17), (3, 2, 2560)])
@pytest.mark.parametrize("offset", [False, True], ids=["base", "offset"])
@torch.inference_mode()
def test_native_add_sigmoid_gate_proj_matches_fp32_reference_and_updates_z_in_place(
    device_index, shape, offset
):
    _require_native(device_index)
    x, y, z, w = _case(shape, device_index, offset=offset)
    identity = id(z)
    ptr = z.data_ptr()
    expected = z.clone() + x * torch.sigmoid(y.float().matmul(w.float()))

    result = ext.add_sigmoid_gate_proj(x, y, z, w)

    assert result is None
    assert id(z) == identity and z.data_ptr() == ptr
    torch.testing.assert_close(z, expected, rtol=5e-5, atol=3e-5)


@pytest.mark.parametrize("device_index", DEVICES)
@torch.inference_mode()
def test_native_add_sigmoid_gate_proj_uses_current_stream(device_index):
    _require_native(device_index)
    device = torch.device("cuda", device_index)
    x = torch.ones((2, 2560), dtype=torch.float32, device=device)
    z = torch.zeros_like(x)
    y = torch.zeros((2, 2560), dtype=torch.float16, device=device)
    w = torch.zeros((2560, 1), dtype=torch.float16, device=device)
    source_y = torch.full_like(y, 0.25)
    source_w = torch.full_like(w, 1 / 2560)
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        y.copy_(source_y)
        w.copy_(source_w)
        ext.add_sigmoid_gate_proj(x, y, z, w)
    stream.synchronize()

    expected_gate = torch.sigmoid(torch.tensor(0.25, device=device))
    torch.testing.assert_close(z, torch.full_like(z, expected_gate), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("device_index", DEVICES)
@pytest.mark.parametrize(
    "bad_arg,dtype",
    [("x", torch.float16), ("z", torch.float16), ("y", torch.float32), ("w", torch.float32)],
)
def test_native_add_sigmoid_gate_proj_rejects_invalid_dtypes(device_index, bad_arg, dtype):
    _require_native(device_index)
    tensors = list(_case((2, 32), device_index))
    tensors["xyzw".index(bad_arg)] = tensors["xyzw".index(bad_arg)].to(dtype)
    with pytest.raises(RuntimeError, match="datatype|dtype"):
        ext.add_sigmoid_gate_proj(*tensors)


@pytest.mark.parametrize("device_index", DEVICES)
@pytest.mark.parametrize("bad_arg", ["x", "y", "z", "w"])
def test_native_add_sigmoid_gate_proj_rejects_noncontiguous_tensors(device_index, bad_arg):
    _require_native(device_index)
    tensors = list(_case((2, 32), device_index))
    index = "xyzw".index(bad_arg)
    tensor = tensors[index]
    tensors[index] = torch.empty((*tensor.shape[:-1], tensor.shape[-1] * 2), dtype=tensor.dtype,
                                 device=tensor.device)[..., ::2]
    assert not tensors[index].is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        ext.add_sigmoid_gate_proj(*tensors)


@pytest.mark.parametrize("device_index", DEVICES)
@pytest.mark.parametrize(
    "shapes",
    [
        ((), (), (), (1, 1)),
        ((2, 0), (2, 0), (2, 0), (0, 1)),
        ((2, 32), (1, 32), (2, 32), (32, 1)),
        ((2, 32), (2, 32), (1, 32), (32, 1)),
        ((2, 32), (2, 32), (2, 32), (32,)),
        ((2, 32), (2, 32), (2, 32), (32, 2)),
    ],
)
def test_native_add_sigmoid_gate_proj_rejects_invalid_shapes(device_index, shapes):
    _require_native(device_index)
    device = torch.device("cuda", device_index)
    dtypes = (torch.float32, torch.float16, torch.float32, torch.float16)
    tensors = [torch.empty(shape, dtype=dtype, device=device) for shape, dtype in zip(shapes, dtypes)]
    with pytest.raises(RuntimeError, match="shape|size|dimension|D must"):
        ext.add_sigmoid_gate_proj(*tensors)


def test_native_add_sigmoid_gate_proj_rejects_cpu_tensors():
    if not torch.version.hip:
        pytest.skip("ROCm build unavailable")
    tensors = (
        torch.empty((1, 32), dtype=torch.float32),
        torch.empty((1, 32), dtype=torch.float16),
        torch.empty((1, 32), dtype=torch.float32),
        torch.empty((32, 1), dtype=torch.float16),
    )
    with pytest.raises(RuntimeError, match="CUDA|device"):
        ext.add_sigmoid_gate_proj(*tensors)


@pytest.mark.skipif(len(_gfx12_devices()) < 2, reason="requires two gfx12 devices")
def test_native_add_sigmoid_gate_proj_rejects_mixed_devices():
    x, y, z, w = _case((1, 32), _gfx12_devices()[0])
    w = w.to(torch.device("cuda", _gfx12_devices()[1]))
    with pytest.raises(RuntimeError, match="same.*device"):
        ext.add_sigmoid_gate_proj(x, y, z, w)
