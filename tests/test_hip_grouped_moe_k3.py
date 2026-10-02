"""gfx12 grouped K3/mul1 decode path for the Qwen3.8 Flash routed experts."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from hip_flash_quant import exl3_expert_quant
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.lora import LoRA
from exllamav3.modules.block_sparse_mlp import _HIP_PREFILL_MAX_EXPERT_ROWS
from exllamav3.modules import block_sparse_mlp as block_sparse_mlp_module
from exllamav3.modules import block_sparse_mlp_routing as routing_module

HIDDEN = 2560
INTERMEDIATES = (640, 768)
TOP_K = 10
NUM_EXPERTS = 16
DEVICE_INDICES = list(range(torch.cuda.device_count())) or [0]
MODEL = Path(os.environ.get(
    "EXL3_FLASH_TEST_MODEL", "~/Models/Qwen3.8-Flash-Next-exl3"
)).expanduser()


def _require_gfx12(device_index=0):
    if not (torch.version.hip and torch.cuda.is_available()):
        pytest.skip("ROCm build / device not available")
    arch = getattr(torch.cuda.get_device_properties(device_index), "gcnArchName", "")
    if arch.split(":", 1)[0] not in ("gfx1200", "gfx1201"):
        pytest.skip(f"grouped MoE requires gfx1200/gfx1201, got {arch or 'unknown'}")
    assert hasattr(ext, "exl3_moe_gfx12_k3"), \
        "gfx12 target build is missing ext.exl3_moe_gfx12_k3"
    assert ext.exl3_gemv_supported(device_index)


def _require_prefill():
    if not hasattr(ext, "exl3_moe_gfx12_k3_prefill"):
        pytest.skip("prefill binding not compiled into this build")


def _require_hip_router():
    if not hasattr(ext, "routing_std_gfx12_bsz1"):
        pytest.skip("HIP gfx12 router binding not compiled into this build")


@pytest.mark.parametrize(
    "value, expected",
    [
        pytest.param(None, 16, id="default"),
        pytest.param("8", 8, id="valid"),
        pytest.param("invalid", 16, id="malformed"),
        pytest.param("0", 1, id="clamped-low"),
        pytest.param("99", 16, id="clamped-high"),
    ],
)
def test_grouped_row_cap_reloads_and_preserves_the_prefill_boundary(monkeypatch, value, expected):
    """The rollback cap is import-time configuration and never opens invalid prefill rows."""
    try:
        with monkeypatch.context() as env:
            if value is None:
                env.delenv("EXL3_HIP_GROUPED_MAX_ROWS", raising=False)
            else:
                env.setenv("EXL3_HIP_GROUPED_MAX_ROWS", value)
            importlib.reload(routing_module)
            module = importlib.reload(block_sparse_mlp_module)
            assert module._HIP_GROUPED_MAX_ROWS == expected
            assert module._HIP_PREFILL_MIN_ROWS == expected + 1
            assert not module._hip_grouped_rows_eligible(expected + 1)
            if expected < 16:
                assert module._hip_prefill_rows_eligible(expected + 1)
    finally:
        importlib.reload(routing_module)
        importlib.reload(block_sparse_mlp_module)


def _ptrs(tensors):
    return torch.tensor([t.data_ptr() for t in tensors], dtype=torch.long, device="cuda")


def _projection(k, n, seed):
    words = (k // 16) * (n // 16) * 48
    # 0x2492 is a finite K3/mul1 trellis cycle. Per-expert input/output signs make
    # matrices distinct without introducing invalid procedural-codebook states.
    tensors, suhs, svhs = [], [], []
    for expert in range(NUM_EXPERTS):
        stream = torch.full((words,), 0x2492, dtype=torch.int16, device="cuda")
        tensors.append(stream.view(k // 16, n // 16, 48))
        gen = torch.Generator(device="cuda").manual_seed(seed * 100 + expert)
        suhs.append((torch.randint(0, 2, (k,), generator=gen, device="cuda") * 2 - 1).half())
        svhs.append((torch.randint(0, 2, (n,), generator=gen, device="cuda") * 2 - 1).half())
    return tensors, suhs, svhs


@pytest.fixture(scope="module", params=INTERMEDIATES, ids=lambda width: f"intermediate-{width}")
def synthetic_grouped_case(request):
    _require_gfx12()
    intermediate = request.param
    torch.manual_seed(1201 + intermediate)
    gate = _projection(HIDDEN, intermediate, 11)
    up = _projection(HIDDEN, intermediate, 23)
    down = _projection(intermediate, HIDDEN, 37)
    return intermediate, gate, up, down


def _linear_ref(x, projection, expert):
    trellis, suhs, svhs = projection
    k, n = x.shape[-1], svhs[expert].numel()
    xh = torch.empty_like(x)
    y = torch.empty((1, n), dtype=torch.float16, device=x.device)
    w = torch.empty((k, n), dtype=torch.float16, device=x.device)
    ext.had_r_128(x, xh, suhs[expert], None, 1.0)
    ext.reconstruct(w, trellis[expert], 3, False, True)
    ext.hgemm(xh, w, y)
    ext.had_r_128(y, y, None, svhs[expert], 1.0)
    return y


def _oracle(x, selected, weights, gate, up, down):
    results = []
    for token in range(x.shape[0]):
        assignments = []
        token_x = x[token:token + 1]
        for slot in range(TOP_K):
            expert = int(selected[token, slot])
            g = _linear_ref(token_x, gate, expert)
            u = _linear_ref(token_x, up, expert)
            a = torch.nn.functional.silu(g.float()).half() * u
            d = _linear_ref(a, down, expert).float()
            assignments.append((expert, slot, d * weights[token, slot].float()))
        # Independent per-token oracle matching the established expert-sorted fp32
        # accumulation. Duplicate-expert ties retain their routing-slot order.
        assignments.sort(key=lambda item: (item[0], item[1]))
        result = torch.zeros_like(assignments[0][2])
        for _, _, assignment in assignments:
            result = result + assignment
        results.append(result)
    return torch.cat(results, dim=0)


def _buffers(intermediate, rows=1, fill=None):
    assignments = rows * TOP_K
    specs = {
        "output": ((rows, HIDDEN), torch.float32),
        "gu_had": ((2 * assignments, HIDDEN), torch.float16),
        "gu_out": ((2 * assignments, intermediate), torch.float16),
        "down_had": ((assignments, intermediate), torch.float16),
        "down_out": ((assignments, HIDDEN), torch.float32),
    }
    result = {}
    for name, (shape, dtype) in specs.items():
        result[name] = torch.empty(shape, dtype=dtype, device="cuda")
        if fill is not None:
            result[name].fill_(fill)
    return result


def _grouped_args(gate, up, down):
    args = []
    for projection in (gate, up, down):
        args.extend(_ptrs(part) for part in projection)
    return args


def _run_grouped(x, selected, weights, gate, up, down, buffers=None, pointer_args=None):
    intermediate = down[1][0].numel()
    buffers = buffers or _buffers(intermediate, x.shape[0])
    args = pointer_args or _grouped_args(gate, up, down)
    ext.exl3_moe_gfx12_k3(
        x, buffers["output"], selected, weights, *args,
        buffers["gu_had"], buffers["gu_out"], buffers["down_had"], buffers["down_out"],
    )
    return buffers["output"]


def _run_prefill(x, selected, weights, gate, up, down):
    rows = x.shape[0]
    assignments = rows * TOP_K
    intermediate = down[1][0].numel()
    order = torch.argsort(selected.reshape(-1), stable=True)
    expert_count = torch.bincount(selected.reshape(-1), minlength=NUM_EXPERTS + 1)
    output = torch.empty((rows, HIDDEN), dtype=torch.float32, device=x.device)
    gu_had = torch.empty((2 * assignments, HIDDEN), dtype=torch.float16, device=x.device)
    gu_out = torch.empty((2 * assignments, intermediate), dtype=torch.float16, device=x.device)
    down_out = torch.empty((assignments, HIDDEN), dtype=torch.float32, device=x.device)
    offsets = torch.empty((NUM_EXPERTS + 1,), dtype=torch.long, device=x.device)
    inverse = torch.empty((assignments,), dtype=torch.long, device=x.device)
    # workspace must cover experts * CHUNKS_PER_EXPERT; CHUNKS_PER_EXPERT tracks
    # the binding's MOE_PREFILL_MAX_ROWS (2048 since c2f73d1), not the old 512-era 320
    chunks = torch.empty(
        (NUM_EXPERTS * (_HIP_PREFILL_MAX_EXPERT_ROWS // 16),),
        dtype=torch.int, device=x.device)
    chunk_count = torch.empty((1,), dtype=torch.int, device=x.device)
    ext.exl3_moe_gfx12_k3_prefill(
        x, output, selected, weights, order, expert_count,
        *_grouped_args(gate, up, down),
        gu_had, gu_out, down_out, offsets, inverse, chunks, chunk_count,
    )
    return output


@pytest.mark.parametrize("rows", [2, 6, 17, 64], ids=["boundary-2", "boundary-6", "boundary-17", "rows-64"])
@torch.inference_mode()
def test_prefill_k3_matches_reconstruction_oracle(synthetic_grouped_case, rows):
    _require_prefill()
    _, gate, up, down = synthetic_grouped_case
    x = torch.randn((rows, HIDDEN), dtype=torch.float16, device="cuda") * 1e-3
    selected = torch.tensor(
        [[(row * 3 + slot * 5) % NUM_EXPERTS for slot in range(TOP_K)] for row in range(rows)],
        dtype=torch.long, device="cuda",
    )
    weights = torch.rand((rows, TOP_K), dtype=torch.float16, device="cuda")
    weights /= weights.sum(dim=-1, keepdim=True)
    expected = _oracle(x, selected, weights, gate, up, down)
    actual = _run_prefill(x, selected, weights, gate, up, down)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.04)


@torch.inference_mode()
def test_prefill_k3_preserves_concentrated_duplicate_slots(synthetic_grouped_case):
    _require_prefill()
    _, gate, up, down = synthetic_grouped_case
    rows = 52  # 520 assignments to one expert, above the per-token row count
    x_gen = torch.Generator(device="cuda").manual_seed(5201 + rows)
    w_gen = torch.Generator(device="cuda").manual_seed(5202 + rows)
    x = torch.randn((rows, HIDDEN), dtype=torch.float16, device="cuda", generator=x_gen) * 1e-2
    selected = torch.zeros((rows, TOP_K), dtype=torch.long, device="cuda")
    weights = torch.rand((rows, TOP_K), dtype=torch.float16, device="cuda", generator=w_gen)
    weights /= weights.sum(dim=-1, keepdim=True)
    expected_rows = []
    for row in range(rows):
        g = _linear_ref(x[row:row + 1], gate, 0)
        u = _linear_ref(x[row:row + 1], up, 0)
        activated = torch.nn.functional.silu(g.float()).half() * u
        down_row = _linear_ref(activated, down, 0).float()
        total = torch.zeros_like(down_row)
        for slot in range(TOP_K):
            total = total + down_row * weights[row, slot].float()
        expected_rows.append(total)
    expected = torch.cat(expected_rows)
    actual = _run_prefill(x, selected, weights, gate, up, down)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=0.04)


@torch.inference_mode()
def test_prefill_k3_is_deterministic(synthetic_grouped_case):
    _require_prefill()
    _, gate, up, down = synthetic_grouped_case
    rows = 17
    x = torch.randn((rows, HIDDEN), dtype=torch.float16, device="cuda") * 1e-3
    selected = torch.tensor(
        [[(row + slot * 5) % NUM_EXPERTS for slot in range(TOP_K)] for row in range(rows)],
        dtype=torch.long, device="cuda",
    )
    weights = torch.rand((rows, TOP_K), dtype=torch.float16, device="cuda")
    weights /= weights.sum(dim=-1, keepdim=True)
    first = _run_prefill(x, selected, weights, gate, up, down).clone()
    second = _run_prefill(x, selected, weights, gate, up, down).clone()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param([9, 2, 15, 1, 12, 7, 4, 0, 11, 5], id="shuffled"),
        pytest.param([7, 2, 7, 1, 2, 7, 4, 1, 9, 2], id="duplicates"),
    ],
)
@pytest.mark.parametrize("rows", [1, 3, 5, 8, 12, 16])
@torch.inference_mode()
def test_grouped_k3_matches_reconstruction_oracle(synthetic_grouped_case, ids, rows):
    _, gate, up, down = synthetic_grouped_case
    x = torch.randn((rows, HIDDEN), dtype=torch.float16, device="cuda") * 1e-3
    selected = torch.stack([
        torch.tensor(ids[row:] + ids[:row], dtype=torch.long, device="cuda")
        for row in range(rows)
    ])
    weights = torch.rand((rows, TOP_K), dtype=torch.float16, device="cuda")
    weights /= weights.sum(dim=-1, keepdim=True)
    expected = _oracle(x, selected, weights, gate, up, down)
    actual = _run_grouped(x, selected, weights, gate, up, down)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.04)


@pytest.mark.parametrize("rows", [1, 3, 5, 8, 12, 16])
@torch.inference_mode()
def test_grouped_k3_is_deterministic(synthetic_grouped_case, rows):
    _, gate, up, down = synthetic_grouped_case
    x = torch.randn((rows, HIDDEN), dtype=torch.float16, device="cuda") * 1e-3
    ids = [7, 2, 7, 1, 2, 7, 4, 1, 9, 2]
    selected = torch.stack([
        torch.tensor(ids[row:] + ids[:row], dtype=torch.long, device="cuda")
        for row in range(rows)
    ])
    weights = torch.rand((rows, TOP_K), dtype=torch.float16, device="cuda")
    weights /= weights.sum(dim=-1, keepdim=True)
    first = _run_grouped(x, selected, weights, gate, up, down).clone()
    second = _run_grouped(x, selected, weights, gate, up, down).clone()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 3, 5, 8, 12, 16])
@torch.inference_mode()
def test_grouped_k3_output_rows_are_isolated(synthetic_grouped_case, rows):
    _, gate, up, down = synthetic_grouped_case
    x = torch.randn((rows, HIDDEN), dtype=torch.float16, device="cuda") * 1e-3
    selected = torch.tensor(
        [[(row * 3 + slot * 5) % NUM_EXPERTS for slot in range(TOP_K)] for row in range(rows)],
        dtype=torch.long, device="cuda",
    )
    weights = torch.rand((rows, TOP_K), dtype=torch.float16, device="cuda")
    weights /= weights.sum(dim=-1, keepdim=True)
    baseline = _run_grouped(x, selected, weights, gate, up, down).clone()

    changed_x = x.clone()
    changed_selected = selected.clone()
    changed_weights = weights.clone()
    changed_row = rows - 1
    changed_x[changed_row].mul_(-3)
    changed_selected[changed_row] = changed_selected[changed_row].roll(3)
    changed_weights[changed_row] = changed_weights[changed_row].roll(1)
    changed = _run_grouped(
        changed_x, changed_selected, changed_weights, gate, up, down).clone()

    assert not torch.equal(changed[changed_row], baseline[changed_row])
    keep = torch.tensor([row for row in range(rows) if row != changed_row], dtype=torch.long, device="cuda")
    torch.testing.assert_close(
        changed.index_select(0, keep), baseline.index_select(0, keep), rtol=0, atol=0)


@torch.inference_mode()
def test_grouped_k3_preserves_16_row_all_duplicate_slots(synthetic_grouped_case):
    """All 160 assignments may target one expert without cross-row loss or aliasing."""
    _, gate, up, down = synthetic_grouped_case
    rows = 16
    x_gen = torch.Generator(device="cuda").manual_seed(3201)
    w_gen = torch.Generator(device="cuda").manual_seed(3202)
    x = torch.randn((rows, HIDDEN), dtype=torch.float16, device="cuda", generator=x_gen) * 1e-2
    selected = torch.zeros((rows, TOP_K), dtype=torch.long, device="cuda")
    weights = torch.rand((rows, TOP_K), dtype=torch.float16, device="cuda", generator=w_gen)
    weights /= weights.sum(dim=-1, keepdim=True)
    expected = _oracle(x, selected, weights, gate, up, down)
    actual = _run_grouped(x, selected, weights, gate, up, down)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=0.04)


@torch.inference_mode()
def test_invalid_device_ids_and_null_entries_zero_all_work(synthetic_grouped_case):
    intermediate, gate, up, down = synthetic_grouped_case
    x = torch.randn((1, HIDDEN), dtype=torch.float16, device="cuda") * 1e-3
    weights = torch.full((1, TOP_K), 1 / TOP_K, dtype=torch.float16, device="cuda")
    invalid = torch.tensor(
        [[-1, NUM_EXPERTS, -99, NUM_EXPERTS + 1, -2, 999, -8, 77, -4, 1024]],
        dtype=torch.long, device="cuda",
    )
    buffers = _buffers(intermediate, fill=float("nan"))
    actual = _run_grouped(x, invalid, weights, gate, up, down, buffers=buffers)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    for workspace in buffers.values():
        assert torch.isfinite(workspace).all()



@pytest.mark.parametrize(
    "null_table",
    range(9),
    ids=[
        "gate-trellis", "gate-suh", "gate-svh",
        "up-trellis", "up-suh", "up-svh",
        "down-trellis", "down-suh", "down-svh",
    ],
)
@torch.inference_mode()
def test_null_pointer_entries_zero_initialized_output(synthetic_grouped_case, null_table):
    intermediate, gate, up, down = synthetic_grouped_case
    x = torch.randn((1, HIDDEN), dtype=torch.float16, device="cuda") * 1e-3
    weights = torch.full((1, TOP_K), 1 / TOP_K, dtype=torch.float16, device="cuda")
    selected = torch.tensor([[9, 2, 15, 1, 12, 7, 4, 0, 11, 5]], dtype=torch.long, device="cuda")
    pointer_args = _grouped_args(gate, up, down)
    pointer_args[null_table] = torch.zeros_like(pointer_args[null_table])
    buffers = _buffers(intermediate, fill=float("nan"))
    actual = _run_grouped(
        x, selected, weights, gate, up, down, buffers=buffers, pointer_args=pointer_args)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
    assert torch.isfinite(buffers["down_out"]).all()


def _offset_view(shape, dtype):
    numel = torch.Size(shape).numel()
    storage = torch.empty(numel + 8, dtype=dtype, device="cuda")
    view = storage[1:1 + numel].view(shape)
    assert view.is_contiguous() and view.data_ptr() % 16 != 0
    return view


@pytest.mark.parametrize("misaligned", ["A", "output", "gu_had", "gu_out", "down_had", "down_out"])
@torch.inference_mode()
def test_grouped_rejects_misaligned_offset_views(synthetic_grouped_case, misaligned):
    intermediate, gate, up, down = synthetic_grouped_case
    x = torch.randn((1, HIDDEN), dtype=torch.float16, device="cuda")
    selected = torch.arange(TOP_K, dtype=torch.long, device="cuda").view(1, -1)
    weights = torch.full((1, TOP_K), 1 / TOP_K, dtype=torch.float16, device="cuda")
    buffers = _buffers(intermediate)
    if misaligned == "A":
        x = _offset_view(x.shape, x.dtype)
    else:
        tensor = buffers[misaligned]
        buffers[misaligned] = _offset_view(tensor.shape, tensor.dtype)
    with pytest.raises(RuntimeError, match=f"{misaligned} must be 16-byte aligned"):
        _run_grouped(x, selected, weights, gate, up, down, buffers=buffers)


@pytest.fixture(scope="module")
def flash_model():
    _require_gfx12()
    if not MODEL.is_dir():
        pytest.skip(f"Flash model not found: {MODEL}")
    codebook, bits = exl3_expert_quant(MODEL)
    if bits["routed"] != 3 or codebook != "mul1":
        pytest.skip(
            f"grouped route needs K3/mul1 routed experts; {MODEL} has "
            f"K{bits['routed']}/{codebook} (set EXL3_FLASH_TEST_MODEL)"
        )
    return Model.from_config(Config.from_directory(str(MODEL)))


def _route_spies():
    bindings = {
        "grouped": "exl3_moe_gfx12_k3",
        "gemv": "exl3_gemv",
        "shared_gate": "add_sigmoid_gate_proj",
        "reconstruct": "reconstruct",
        "reconstruct_had_slice": "reconstruct_had_slice",
        "hgemm": "hgemm",
    }
    calls = {name: 0 for name in bindings}
    calls["routed_k3_gemv"] = 0
    originals = {name: getattr(ext, binding) for name, binding in bindings.items()}
    real_hgemm_recon = getattr(ext, "hgemm_recon", None)

    def hgemm_recon_spy(*args, **kwargs):
        calls["hgemm"] += 1  # reconstruct_hgemm may call either GEMM entry point
        return real_hgemm_recon(*args, **kwargs)

    for name, binding in bindings.items():
        def spy(*args, _name=name, **kwargs):
            calls[_name] += 1
            if _name == "gemv" and args[1].shape[-1] // 16 == 3:
                calls["routed_k3_gemv"] += 1
            return originals[_name](*args, **kwargs)
        setattr(ext, binding, spy)
    if real_hgemm_recon is not None:
        ext.hgemm_recon = hgemm_recon_spy

    def restore():
        for name, binding in bindings.items():
            setattr(ext, binding, originals[name])
        if real_hgemm_recon is not None:
            ext.hgemm_recon = real_hgemm_recon

    return calls, restore


@pytest.mark.parametrize("device_index", DEVICE_INDICES)
@pytest.mark.parametrize("rows", [1, 3, 5, 8, 16])
@torch.inference_mode()
def test_flash_multirow_route_and_decline_guards(flash_model, device_index, rows, monkeypatch):
    _require_gfx12(device_index)
    _require_hip_router()
    device = torch.device("cuda", device_index)
    mlp = flash_model.find_module("model.language_model.layers.0.mlp")
    try:
        mlp.load(device=device)
        assert mlp.support_hip_grouped
        assert mlp.tp_mode is None
        assert mlp.bc is None
        assert mlp.multi_gate.linears == mlp.gates
        assert mlp.multi_up.linears == mlp.ups
        assert mlp.multi_down.linears == mlp.downs
        x = torch.randn((1, rows, HIDDEN), dtype=torch.float16, device=device)

        assert getattr(ext.add_sigmoid_gate_proj, "__module__", None) == "exllamav3_ext"
        calls, restore = _route_spies()
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as profile:
                actual = mlp.forward(x, {}).clone()
        finally:
            restore()
        assert calls == {
            "grouped": 1, "gemv": 3, "routed_k3_gemv": 0, "shared_gate": 1,
            "reconstruct": 0, "reconstruct_had_slice": 0, "hgemm": 0,
        }
        shared_gate_matmuls = [
            event for event in profile.events()
            if event.name == "aten::matmul"
            and len(event.input_shapes) >= 2
            and event.input_shapes[1] == [HIDDEN, 1]
        ]
        assert not shared_gate_matmuls
        assert torch.isfinite(actual).all()

        # The dedicated route uses the same K3 arithmetic and deterministic expert-sorted
        # reduction as the per-expert fallback, including independently for verification rows.
        monkeypatch.setenv("EXL3_HIP_GROUPED_MOE", "0")
        fallback = mlp.forward(x, {}).clone()
        monkeypatch.setenv("EXL3_HIP_GROUPED_MOE", "1")
        torch.testing.assert_close(actual, fallback, rtol=0, atol=1e-6)

        guard_cases = [
            ("grouped-disabled", {}, {"EXL3_HIP_GROUPED_MOE": "0"}, None, True),
            *(([("multirow-disabled", {}, {"EXL3_HIP_GROUPED_MOE_MULTIROW": "0"}, None, True)])
              if rows > 1 else []),
            ("gemv-disabled", {}, {"EXL3_GEMV": "0"}, None, True),
            ("activate-all", {"activate_all_experts": True}, {}, None, False),
            ("reconstruct", {"reconstruct": True}, {}, None, False),
            ("autosplit", {"autosplit_measure": True}, {}, None, False),
            ("act-limit", {}, {}, 1.0, False),
            ("tensor-tp", {}, {}, "channels", False),
            ("expert-tp", {}, {}, "experts", False),
        ]
        for name, params, env, state, compare_fallback in guard_cases:
            monkeypatch.setenv("EXL3_HIP_GROUPED_MOE", env.get("EXL3_HIP_GROUPED_MOE", "1"))
            monkeypatch.setenv(
                "EXL3_HIP_GROUPED_MOE_MULTIROW",
                env.get("EXL3_HIP_GROUPED_MOE_MULTIROW", "1"),
            )
            monkeypatch.setenv("EXL3_GEMV", env.get("EXL3_GEMV", "1"))
            old_limit, old_tp = mlp.act_limit, mlp.tp_mode
            if isinstance(state, float):
                mlp.act_limit = state
            elif isinstance(state, str):
                mlp.tp_mode = state
            calls, restore = _route_spies()
            try:
                guarded = mlp.forward(x, params)
            finally:
                restore()
                mlp.act_limit, mlp.tp_mode = old_limit, old_tp
            assert calls["grouped"] == 0, f"{name} entered grouped route"
            assert torch.isfinite(guarded).all()
            if compare_fallback:
                torch.testing.assert_close(guarded, fallback, rtol=0, atol=0.04)

        if rows == 16:
            monkeypatch.setenv("EXL3_HIP_GROUPED_MOE", "1")
            oversized = torch.randn((1, 17, HIDDEN), dtype=torch.float16, device=device)
            calls, restore = _route_spies()
            try:
                guarded = mlp.forward(oversized, {})
            finally:
                restore()
            assert calls["grouped"] == 0, "m=17 must decline the grouped route"
            assert torch.isfinite(guarded).all()
    finally:
        mlp.unload()


@pytest.mark.parametrize("device_index", DEVICE_INDICES[:2])
@pytest.mark.parametrize("rows", [6, 8, 12, 16])
@torch.inference_mode()
def test_flash_grouped_row_cap_rolls_back_to_prefill_at_6(flash_model, device_index, rows, monkeypatch):
    _require_prefill()
    _require_gfx12(device_index)
    device = torch.device("cuda", device_index)
    mlp = flash_model.find_module("model.language_model.layers.0.mlp")
    old_max_rows = os.environ.get("EXL3_HIP_GROUPED_MAX_ROWS")
    old_prefill = os.environ.get("EXL3_HIP_GROUPED_MOE_PREFILL")
    original_module = block_sparse_mlp_module
    original_grouped = ext.exl3_moe_gfx12_k3
    original_prefill = ext.exl3_moe_gfx12_k3_prefill
    grouped_calls = 0
    prefill_calls = 0

    def grouped_spy(*args, **kwargs):
        nonlocal grouped_calls
        grouped_calls += 1
        return original_grouped(*args, **kwargs)

    def prefill_spy(*args, **kwargs):
        nonlocal prefill_calls
        prefill_calls += 1
        return original_prefill(*args, **kwargs)

    try:
        monkeypatch.setenv("EXL3_HIP_GROUPED_MAX_ROWS", "5")
        monkeypatch.setenv("EXL3_HIP_GROUPED_MOE_PREFILL", "1")
        importlib.reload(routing_module)
        importlib.reload(block_sparse_mlp_module)
        monkeypatch.setattr(ext, "exl3_moe_gfx12_k3", grouped_spy)
        monkeypatch.setattr(ext, "exl3_moe_gfx12_k3_prefill", prefill_spy)
        mlp.load(device=device)
        assert mlp.support_hip_grouped

        x_gen = torch.Generator(device=device).manual_seed(7101 + device_index * 100 + rows)
        x = torch.randn((1, rows, HIDDEN), dtype=torch.float16, device=device, generator=x_gen)
        result = mlp.forward(x, {}).clone()
        assert grouped_calls == 0
        assert prefill_calls == 1
        assert torch.isfinite(result).all()

        grouped_calls = 0
        prefill_calls_before = prefill_calls
        monkeypatch.setenv("EXL3_HIP_GROUPED_MOE_PREFILL", "0")
        fallback = mlp.forward(x, {}).clone()
        assert grouped_calls == 0
        assert prefill_calls == prefill_calls_before
        assert torch.isfinite(fallback).all()
        torch.testing.assert_close(result, fallback, rtol=0, atol=0.04)

        grouped_calls = 0
        prefill_calls = 0
        monkeypatch.setenv("EXL3_HIP_GROUPED_MAX_ROWS", "16")
        monkeypatch.setenv("EXL3_HIP_GROUPED_MOE_PREFILL", "1")
        importlib.reload(routing_module)
        importlib.reload(block_sparse_mlp_module)
        mlp.unload()
        mlp.load(device=device)
        rerun = mlp.forward(x, {}).clone()
        assert grouped_calls == 1
        assert prefill_calls == 0
        assert torch.isfinite(rerun).all()
    finally:
        if old_max_rows is None:
            os.environ.pop("EXL3_HIP_GROUPED_MAX_ROWS", None)
        else:
            os.environ["EXL3_HIP_GROUPED_MAX_ROWS"] = old_max_rows
        if old_prefill is None:
            os.environ.pop("EXL3_HIP_GROUPED_MOE_PREFILL", None)
        else:
            os.environ["EXL3_HIP_GROUPED_MOE_PREFILL"] = old_prefill
        importlib.reload(routing_module)
        importlib.reload(original_module)
        mlp.unload()


@pytest.mark.parametrize("device_index", DEVICE_INDICES[:2])
@pytest.mark.parametrize("rows", [17, 64], ids=["rows-17", "rows-64"])
@torch.inference_mode()
def test_flash_prefill_route_matches_layer_fallback(flash_model, device_index, rows, monkeypatch):
    _require_prefill()
    _require_gfx12(device_index)
    device = torch.device("cuda", device_index)
    mlp = flash_model.find_module("model.language_model.layers.0.mlp")
    try:
        mlp.load(device=device)
        assert mlp.support_hip_prefill
        x_gen = torch.Generator(device=device).manual_seed(6101 + device_index * 100 + rows)
        x = torch.randn((1, rows, HIDDEN), dtype=torch.float16, device=device, generator=x_gen)
        original = ext.exl3_moe_gfx12_k3_prefill
        calls = 0

        def spy(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ext, "exl3_moe_gfx12_k3_prefill", spy)
        monkeypatch.setenv("EXL3_HIP_GROUPED_MOE_PREFILL", "1")
        actual = mlp.forward(x, {}).clone()
        assert calls == 1
        monkeypatch.setenv("EXL3_HIP_GROUPED_MOE_PREFILL", "0")
        expected = mlp.forward(x, {}).clone()
        assert calls == 1
        difference = (actual - expected).abs()
        assert torch.isfinite(actual).all()
        # Observed deterministic device-1 max delta is ~7.35e-4; keep a narrow envelope.
        assert difference.max().item() <= 1e-3
        assert difference.mean().item() <= 2e-5
    finally:
        mlp.unload()


@pytest.mark.parametrize("device_index", DEVICE_INDICES[:2])
@torch.inference_mode()
def test_expert_lora_permanently_invalidates_grouped_route(flash_model, device_index, tmp_path):
    _require_gfx12(device_index)
    device = torch.device("cuda", device_index)
    mlp = flash_model.find_module("model.language_model.layers.0.mlp")
    adapter = None
    try:
        mlp.load(device=device)
        assert mlp.support_hip_grouped
        target = mlp.gates[0]
        rank = 1
        config = {
            "r": rank,
            "lora_alpha": rank,
            "fan_in_fan_out": False,
        }
        (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf8")
        save_file({
            f"base_model.{target.key}.lora_A.default.weight":
                torch.ones((rank, target.in_features_unpadded), dtype=torch.float16),
            f"base_model.{target.key}.lora_B.default.weight":
                torch.ones((target.out_features_unpadded, rank), dtype=torch.float16),
        }, tmp_path / "adapter_model.safetensors")
        adapter = LoRA.from_directory(flash_model, str(tmp_path))
        assert target.key in adapter.target_modules
        assert target.lora_a_tensors and target.lora_b_tensors
        assert mlp.hip_grouped_lora_blocked
        assert not mlp.support_hip_grouped
        assert not mlp.support_hip_prefill

        x = torch.randn((1, 1, HIDDEN), dtype=torch.float16, device=device) * 1e-3
        calls, restore = _route_spies()
        try:
            result = mlp.forward(x, {})
        finally:
            restore()
        assert calls["grouped"] == 0
        assert torch.isfinite(result).all()

        adapter.unload()
        adapter = None
        assert not target.lora_a_tensors and not target.lora_b_tensors
        assert mlp.hip_grouped_lora_blocked
        assert not mlp.support_hip_grouped
        assert not mlp.support_hip_prefill
    finally:
        if adapter is not None:
            adapter.unload()
        mlp.unload()


def test_grouped_source_is_hip_only_and_does_not_enable_generic_mgemm():
    from exllamav3.exllamav3_ext.build_config import ROCM_EXCLUDE_FILES
    assert "quant/exl3_gemm.cu" in ROCM_EXCLUDE_FILES
    assert "libtorch/blocksparse_mlp.cpp" in ROCM_EXCLUDE_FILES
    if torch.version.hip:
        assert not hasattr(ext, "exl3_mgemm")
