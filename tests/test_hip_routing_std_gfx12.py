"""gfx12 wave32 batch-one standard softmax MoE router."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import os

import pytest
import torch

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm router tests", allow_module_level = True)

from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import block_sparse_mlp as bsm
from exllamav3.modules import block_sparse_mlp_routing as routing_module
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

HIDDEN = 2560
EXPERTS = 512
TOP_K = 10
GFX12_DEVICES = [
    index for index in range(torch.cuda.device_count())
    if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").split(":", 1)[0]
    in ("gfx1200", "gfx1201")
]
MODEL = Path(os.environ.get(
    "EXL3_FLASH_TEST_MODEL", "~/Models/Qwen3.8-Flash-Next-exl3"
)).expanduser()


def _require_gfx12(device_index):
    if device_index not in GFX12_DEVICES:
        pytest.skip(f"device {device_index} is not gfx12")
    assert getattr(torch.cuda.get_device_properties(device_index), "warp_size", 0) == 32
    assert hasattr(ext, "routing_std_gfx12_bsz1")


def _case(device_index=0, rows=1, seed=1201):
    _require_gfx12(device_index)
    device = torch.device("cuda", device_index)
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn((rows, HIDDEN), generator=generator, device=device, dtype=torch.half) * 0.02
    gate = torch.randn((HIDDEN, EXPERTS), generator=generator, device=device, dtype=torch.half) * 0.02
    return hidden.contiguous(), gate.contiguous()


def _buffers(device, rows=1):
    return (
        torch.empty((rows, EXPERTS), device=device, dtype=torch.half),
        torch.empty((rows, TOP_K), device=device, dtype=torch.long),
        torch.empty((rows, TOP_K), device=device, dtype=torch.half),
    )


def _native(hidden, gate_t, buffers=None):
    scores, selected, weights = buffers or _buffers(hidden.device, hidden.shape[0])
    ext.routing_std_gfx12_bsz1(hidden, gate_t, scores, selected, weights)
    return scores, selected, weights


def _routing_cfg(gate):
    scores, selected, weights = _buffers(gate.device)
    return bsm.RoutingCFG(
        gate_tensor=gate, gate_tensor_t=None,
        num_experts=EXPERTS, num_experts_per_tok=TOP_K,
        router_logits_bsz1=scores, routing_weights_bsz1=weights,
        selected_experts_bsz1=selected,
        e_score_correction_bias=None, e_score_bias_h=None,
        routed_scaling_factor=None, n_group=None, topk_group=None,
        per_expert_scale=None,
    )


def _oracle(hidden, gate):
    # The native GEMV accumulates half products in fp32, then stores rounded fp16 scores.
    # Rank those stored scores, not the unrounded accumulation, with the router's explicit
    # lower-ID tie break.
    scores = torch.matmul(hidden.float(), gate.float()).half()
    selected = torch.tensor(
        [
            sorted(range(EXPERTS), key=lambda index: (-float(row[index]), index))[:TOP_K]
            for row in scores.cpu()
        ],
        dtype=torch.long,
        device=hidden.device,
    )
    selected_scores = scores.float().gather(1, selected)
    maximum = selected_scores.max(dim=-1, keepdim=True).values
    # This is the extended-real softmax contract for scores rounded to either infinity:
    # every maximum receives one unit of mass, avoiding inf - inf / -inf - -inf NaNs.
    exponentials = torch.where(
        selected_scores == maximum,
        torch.ones_like(selected_scores),
        torch.exp(selected_scores - maximum),
    )
    return scores, selected, (exponentials / exponentials.sum(dim=-1, keepdim=True)).half()


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 12, 16])
@torch.inference_mode()
def test_native_router_matches_independent_torch_route(device_index, rows):
    hidden, gate = _case(device_index, rows)
    actual_scores, actual_selected, actual_weights = _native(hidden, gate.T.contiguous())
    expected_scores, expected_selected, expected_weights = _oracle(hidden, gate)
    torch.cuda.synchronize(device_index)

    torch.testing.assert_close(actual_scores, expected_scores, rtol=2e-3, atol=2e-3)
    assert torch.equal(actual_selected, expected_selected)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 12, 16])
@torch.inference_mode()
def test_native_router_matches_exact_tie_oracle_deterministically(device_index, rows):
    _require_gfx12(device_index)
    device = torch.device("cuda", device_index)
    hidden = torch.zeros((rows, HIDDEN), device=device, dtype=torch.half)
    hidden[:, 0] = 1
    logits = torch.full((EXPERTS,), -4, device=device, dtype=torch.half)
    logits[torch.tensor([31, 17, 2], device=device)] = 2
    logits[torch.arange(3, 10, device=device)] = 1
    gate = torch.zeros((HIDDEN, EXPERTS), device=device, dtype=torch.half)
    gate[0] = logits
    expected_scores, expected_selected, expected_weights = _oracle(hidden, gate)
    expected_ids = torch.tensor(
        [2, 17, 31, 3, 4, 5, 6, 7, 8, 9], device=device).expand(rows, -1)
    assert torch.equal(expected_selected, expected_ids)

    outputs = []
    for _ in range(8):
        scores, selected, weights = _native(hidden, gate.T.contiguous())
        outputs.append((scores.clone(), selected.clone(), weights.clone()))
    torch.cuda.synchronize(device_index)

    for scores, selected, weights in outputs:
        assert torch.equal(scores, expected_scores)
        assert torch.equal(selected, expected_selected)
        assert torch.equal(weights, expected_weights)


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@pytest.mark.parametrize("rows", [1, 4, 8, 12, 16])
@torch.inference_mode()
def test_native_router_selects_distinct_ids_for_all_negative_infinity_scores(device_index, rows):
    _require_gfx12(device_index)
    device = torch.device("cuda", device_index)
    # fp32 GEMV accumulation overflows only when the result is rounded to the score's fp16
    # storage type. It therefore exercises genuine -inf logits, not a host-side sentinel.
    hidden = torch.ones((rows, HIDDEN), device=device, dtype=torch.half)
    gate = torch.full((HIDDEN, EXPERTS), -32, device=device, dtype=torch.half)
    expected_scores, expected_selected, expected_weights = _oracle(hidden, gate)
    assert torch.isneginf(expected_scores).all()
    assert torch.equal(
        expected_selected,
        torch.arange(TOP_K, device=device, dtype=torch.long).expand(rows, -1),
    )
    assert torch.equal(
        expected_weights,
        torch.full((rows, TOP_K), 0.1, device=device, dtype=torch.half),
    )

    outputs = [_native(hidden, gate.T.contiguous()) for _ in range(4)]
    torch.cuda.synchronize(device_index)
    for scores, selected, weights in outputs:
        assert torch.equal(scores, expected_scores)
        assert torch.equal(selected, expected_selected)
        assert torch.equal(weights, expected_weights)


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 12, 16])
@torch.inference_mode()
def test_native_router_uses_current_custom_stream(device_index, rows):
    hidden, gate = _case(device_index, rows, seed=77)
    gate_t = gate.T.contiguous()
    stream = torch.cuda.Stream(device=device_index)
    with torch.cuda.stream(stream):
        hidden.fill_(0.03125)
        scores, selected, weights = _native(hidden, gate_t)
        done = torch.cuda.Event()
        done.record(stream)
    done.synchronize()
    expected_scores, expected_selected, expected_weights = _oracle(hidden, gate)
    torch.testing.assert_close(scores, expected_scores, rtol=2e-3, atol=2e-3)
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-3, atol=2e-3)


def _offset_view(shape, dtype, device):
    count = torch.Size(shape).numel()
    storage = torch.empty(count + 16, dtype=dtype, device=device)
    offset = 1 if dtype != torch.long else 1
    return storage[offset:offset + count].view(shape)


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@pytest.mark.parametrize("bad", [
    "hidden-shape", "hidden-dtype", "hidden-noncontiguous", "hidden-alignment",
    "gate-shape", "gate-dtype", "gate-noncontiguous", "gate-alignment",
    "scores-shape", "selected-dtype", "weights-shape",
])
@torch.inference_mode()
def test_native_router_rejects_invalid_inputs(device_index, bad):
    hidden, gate = _case(device_index)
    gate_t = gate.T.contiguous()
    scores, selected, weights = _buffers(hidden.device)
    if bad == "hidden-shape": hidden = hidden[:, :-1]
    elif bad == "hidden-dtype": hidden = hidden.float()
    elif bad == "hidden-noncontiguous": hidden = torch.empty((HIDDEN, 2), device=hidden.device, dtype=torch.half)[:, 0].view(1, -1)
    elif bad == "hidden-alignment": hidden = _offset_view((1, HIDDEN), torch.half, hidden.device)
    elif bad == "gate-shape": gate_t = gate_t[:, :-1]
    elif bad == "gate-dtype": gate_t = gate_t.float()
    elif bad == "gate-noncontiguous": gate_t = gate
    elif bad == "gate-alignment": gate_t = _offset_view((EXPERTS, HIDDEN), torch.half, hidden.device)
    elif bad == "scores-shape": scores = scores[:, :-1]
    elif bad == "selected-dtype": selected = selected.int()
    elif bad == "weights-shape": weights = weights[:, :-1]
    with pytest.raises(RuntimeError, match="routing_std_gfx12_bsz1"):
        ext.routing_std_gfx12_bsz1(hidden, gate_t, scores, selected, weights)


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 12, 16])
@torch.inference_mode()
def test_python_route_executes_native_and_profiles_without_aten_mm_or_topk(device_index, rows, monkeypatch):
    hidden, gate = _case(device_index, rows)
    scores, selected, weights = _buffers(hidden.device)
    cfg = SimpleNamespace(
        gate_tensor=gate, gate_tensor_t=gate.T.contiguous(), router_bias=None,
        num_experts=EXPERTS, num_experts_per_tok=TOP_K,
        router_logits_bsz1=scores, selected_experts_bsz1=selected,
        routing_weights_bsz1=weights, per_expert_scale=None,
    )
    calls = {"native": 0}
    real_native = ext.routing_std_gfx12_bsz1

    def spy(*args, **kwargs):
        calls["native"] += 1
        return real_native(*args, **kwargs)

    monkeypatch.setattr(ext, "routing_std_gfx12_bsz1", spy)
    with torch.cuda.device(device_index):
        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]) as profile:
            actual_selected, actual_weights = bsm.routing_std(rows, cfg, hidden, {})
    torch.cuda.synchronize(device_index)
    _, expected_selected, expected_weights = _oracle(hidden, gate)
    assert calls["native"] == 1
    assert torch.equal(actual_selected, expected_selected)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=2e-3, atol=2e-3)
    names = [event.name for event in profile.events()]
    assert "aten::mm" not in names
    assert "aten::topk" not in names
    assert any("routing_gemv_gfx12_bsz1_kernel" in name for name in names)
    assert any("routing_std_topk_gfx12_bsz1_kernel" in name for name in names)


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@pytest.mark.parametrize("mode", ["disabled", "rows-over-bound", "activate-all", "scale", "bias"])
@torch.inference_mode()
def test_python_route_falls_back_for_unsupported_modes(device_index, mode, monkeypatch):
    hidden, gate = _case(device_index)
    bsz, params = 1, {}
    if mode == "rows-over-bound":
        bsz = 17
        hidden = hidden.expand(bsz, -1).contiguous()
    elif mode == "activate-all": params["activate_all_experts"] = True
    elif mode == "disabled": monkeypatch.setenv("EXL3_HIP_ROUTER", "0")
    scale = torch.ones(EXPERTS, device=hidden.device, dtype=torch.bfloat16) if mode == "scale" else None
    bias = torch.zeros(EXPERTS, device=hidden.device, dtype=torch.half) if mode == "bias" else None
    cfg = SimpleNamespace(
        gate_tensor=gate, gate_tensor_t=None, router_bias=bias,
        num_experts=EXPERTS, num_experts_per_tok=TOP_K,
        router_logits_bsz1=torch.empty((1, EXPERTS), device=hidden.device, dtype=torch.half),
        selected_experts_bsz1=torch.empty((1, TOP_K), device=hidden.device, dtype=torch.long),
        routing_weights_bsz1=torch.empty((1, TOP_K), device=hidden.device, dtype=torch.half),
        per_expert_scale=scale,
    )
    calls = {"native": 0}
    real_native = ext.routing_std_gfx12_bsz1

    def spy(*args, **kwargs):
        calls["native"] += 1
        return real_native(*args, **kwargs)

    monkeypatch.setattr(ext, "routing_std_gfx12_bsz1", spy)
    selected, weights = bsm.routing_std(bsz, cfg, hidden, params)
    assert calls["native"] == 0
    if mode == "disabled":
        assert cfg.gate_tensor_t is None
    expected_logits = (hidden @ gate).float()
    if params.get("activate_all_experts"):
        expected_selected = torch.arange(EXPERTS, device=hidden.device).expand(bsz, -1)
        expected_weights = torch.softmax(expected_logits, dim=-1)
    else:
        values, expected_selected = torch.topk(expected_logits, TOP_K, dim=-1)
        expected_weights = torch.softmax(values, dim=-1)
    if scale is not None:
        expected_weights *= scale.float()[expected_selected]
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights.half(), rtol=0, atol=0)


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@torch.inference_mode()
def test_python_router_reuses_one_cfg_workspace_across_multirow_calls(device_index):
    _require_gfx12(device_index)
    device = torch.device("cuda", device_index)
    hidden = torch.zeros((8, HIDDEN), dtype=torch.half, device=device)
    hidden[:, 0] = 1
    hidden[:, 1] = torch.arange(8, dtype=torch.half, device=device)
    gate = torch.zeros((HIDDEN, EXPERTS), dtype=torch.half, device=device)
    gate[0] = -torch.arange(EXPERTS, dtype=torch.half, device=device)
    gate[1] = 1
    cfg = _routing_cfg(gate)
    one_ptrs = tuple(
        tensor.data_ptr() for tensor in (
            cfg.router_logits_bsz1, cfg.selected_experts_bsz1, cfg.routing_weights_bsz1
        )
    )
    multi_ptrs = None
    outputs = {}

    for rows in (8, 1, 4, 8, 1, 4):
        y = hidden[:rows]
        expected_scores, expected_selected, expected_weights = _oracle(y, gate)
        selected, weights = bsm.routing_std(rows, cfg, y, {})
        scores = cfg.router_logits_bsz1 if rows == 1 else cfg.router_logits_gfx12[:rows]
        torch.cuda.synchronize(device_index)

        assert torch.equal(scores, expected_scores)
        assert torch.equal(selected, expected_selected)
        assert torch.equal(weights, expected_weights)
        if rows == 1:
            assert tuple(tensor.data_ptr() for tensor in (scores, selected, weights)) == one_ptrs
        else:
            current_ptrs = tuple(tensor.data_ptr() for tensor in (
                cfg.router_logits_gfx12, cfg.selected_experts_gfx12, cfg.routing_weights_gfx12
            ))
            if multi_ptrs is None:
                multi_ptrs = current_ptrs
            assert current_ptrs == multi_ptrs
            assert tuple(tensor.data_ptr() for tensor in (scores, selected, weights)) == multi_ptrs
            assert tuple(scores.shape) == (rows, EXPERTS)
            assert tuple(selected.shape) == tuple(weights.shape) == (rows, TOP_K)
        current = (scores.clone(), selected.clone(), weights.clone())
        if rows in outputs:
            assert all(torch.equal(actual, expected) for actual, expected in zip(current, outputs[rows]))
        outputs[rows] = current


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@torch.inference_mode()
def test_unsupported_row_count_is_rejected_by_native_binding(device_index):
    hidden, gate = _case(device_index, rows=17)
    with pytest.raises(RuntimeError, match="supports 1 through 16 rows"):
        _native(hidden, gate.T.contiguous())


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@torch.inference_mode()
def test_bias_route_and_missing_binding_do_not_enter_native(device_index, monkeypatch):
    hidden, gate = _case(device_index)
    bias = torch.linspace(-0.1, 0.1, EXPERTS, device=hidden.device, dtype=torch.half)
    cfg = SimpleNamespace(
        gate_tensor=gate, gate_tensor_t=None, router_bias=bias,
        num_experts=EXPERTS, num_experts_per_tok=TOP_K,
        router_logits_bsz1=torch.empty((1, EXPERTS), device=hidden.device, dtype=torch.half),
        selected_experts_bsz1=torch.empty((1, TOP_K), device=hidden.device, dtype=torch.long),
        routing_weights_bsz1=torch.empty((1, TOP_K), device=hidden.device, dtype=torch.half),
        per_expert_scale=None,
    )
    monkeypatch.setattr(bsm, "ext", SimpleNamespace())
    selected, weights = bsm.routing_std_bias(1, cfg, hidden, {})
    values, expected_selected = torch.topk((hidden @ gate).float() + bias.float(), TOP_K, dim=-1)
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, torch.softmax(values, dim=-1).half(), rtol=0, atol=0)

    multi_hidden = hidden.expand(2, -1).contiguous()
    cfg.router_bias = None
    selected, weights = bsm.routing_std(2, cfg, multi_hidden, {})
    values, expected_selected = torch.topk((multi_hidden @ gate).float(), TOP_K, dim=-1)
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, torch.softmax(values, dim=-1).half(), rtol=0, atol=0)


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@torch.inference_mode()
def test_every_flash_router_real_weights_match_torch_on_random_states(device_index):
    _require_gfx12(device_index)
    if not MODEL.is_dir():
        pytest.skip(f"Flash model not found: {MODEL}")
    model = Model.from_config(Config.from_directory(str(MODEL)))
    routers = [module for module in model if isinstance(module, BlockSparseMLP)]
    assert routers
    device = torch.device("cuda", device_index)
    checked = 0
    try:
        for layer_index, mlp in enumerate(routers):
            mlp.load(device=device)
            try:
                cfg = mlp.routing_cfg
                if not routing_module._hip_router_config_supported(cfg):
                    continue
                assert cfg.gate_tensor_t is not None
                assert cfg.gate_tensor_t.shape == (EXPERTS, HIDDEN)
                assert cfg.gate_tensor_t.untyped_storage().nbytes() == EXPERTS * HIDDEN * 2
                transpose_ptr = cfg.gate_tensor_t.data_ptr()
                bsm._prepare_hip_router_gate_t(cfg)
                assert cfg.gate_tensor_t.data_ptr() == transpose_ptr
                generator = torch.Generator(device=device).manual_seed(1201 + layer_index)
                for rows in (1, 2, 4, 8):
                    for scale in (0.02, 0.25, 1.0):
                        hidden = torch.randn(
                            (rows, HIDDEN), generator=generator, device=device, dtype=torch.half
                        ) * scale
                        expected_scores, expected_selected, expected_weights = _oracle(
                            hidden, cfg.gate_tensor)
                        actual_selected, actual_weights = bsm.routing_std(rows, cfg, hidden, {})
                        actual_scores = (
                            cfg.router_logits_bsz1 if rows == 1
                            else cfg.router_logits_gfx12[:rows]
                        )
                        torch.testing.assert_close(
                            actual_scores, expected_scores, rtol=2e-3, atol=2e-3,
                            msg=lambda message: f"{mlp.key}: {message}",
                        )
                        assert torch.equal(actual_selected, expected_selected), mlp.key
                        torch.testing.assert_close(
                            actual_weights, expected_weights, rtol=2e-3, atol=2e-3,
                            msg=lambda message: f"{mlp.key}: {message}",
                        )
                checked += 1
            finally:
                mlp.unload()
        assert checked == len(routers)
    finally:
        model.unload()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@torch.inference_mode()
def test_flash_mlp_forward_has_native_router_route_and_output_parity(device_index, monkeypatch):
    _require_gfx12(device_index)
    if not MODEL.is_dir():
        pytest.skip(f"Flash model not found: {MODEL}")
    model = Model.from_config(Config.from_directory(str(MODEL)))
    mlp = next(module for module in model if isinstance(module, BlockSparseMLP))
    device = torch.device("cuda", device_index)
    original_routing = mlp.routing_fn
    real_native = ext.routing_std_gfx12_bsz1
    mode = {"name": None}
    routes = {}
    native_calls = 0

    def capture_route(bsz, cfg, y, params):
        selected, weights = original_routing(bsz, cfg, y, params)
        if cfg is mlp.routing_cfg and mode["name"] is not None:
            routes[mode["name"]] = (selected.clone(), weights.clone())
        return selected, weights

    def native_spy(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return real_native(*args, **kwargs)

    try:
        mlp.load(device=device)
        if not routing_module._hip_router_config_supported(mlp.routing_cfg):
            pytest.skip(f"{mlp.key} is not a supported Flash router")
        generator = torch.Generator(device=device).manual_seed(1201)
        x = torch.randn((1, 4, HIDDEN), generator=generator, device=device, dtype=torch.half) * 0.02
        mlp.routing_fn = capture_route
        monkeypatch.setattr(ext, "routing_std_gfx12_bsz1", native_spy)
        # The grouped expert route is orthogonal; keep the established expert execution fixed
        # so this compares only the native router with its standard fallback.
        monkeypatch.setenv("EXL3_HIP_GROUPED_MOE", "0")

        mode["name"] = "native"
        monkeypatch.setenv("EXL3_HIP_ROUTER", "1")
        native_output = mlp.forward(x, {}).clone()
        torch.cuda.synchronize(device_index)

        mode["name"] = "fallback"
        monkeypatch.setenv("EXL3_HIP_ROUTER", "0")
        fallback_output = mlp.forward(x, {}).clone()
        torch.cuda.synchronize(device_index)

        assert native_calls == 1
        native_selected, native_weights = routes["native"]
        fallback_selected, fallback_weights = routes["fallback"]
        # Compare top-k membership row-wise: exactly tied scores may be emitted in
        # different orders by the native kernel and the fallback sort, which is not a
        # routing divergence; tied experts carry equal weights, so the weight and output
        # comparisons below are order-insensitive for them.
        assert torch.equal(native_selected.sort(dim=1).values,
                           fallback_selected.sort(dim=1).values)
        torch.testing.assert_close(native_weights, fallback_weights, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(native_output, fallback_output, rtol=2e-3, atol=2e-5)
    finally:
        mlp.routing_fn = original_routing
        mlp.unload()
        model.unload()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("device_index", GFX12_DEVICES[:2] or [0])
@torch.inference_mode()
def test_non_gfx12_architecture_uses_python_fallback(device_index, monkeypatch):
    hidden, gate = _case(device_index)
    buffers = _buffers(hidden.device)
    cfg = SimpleNamespace(
        gate_tensor=gate, gate_tensor_t=None, router_bias=None,
        num_experts=EXPERTS, num_experts_per_tok=TOP_K,
        router_logits_bsz1=buffers[0], selected_experts_bsz1=buffers[1],
        routing_weights_bsz1=buffers[2], per_expert_scale=None,
    )
    monkeypatch.setattr(
        torch.cuda, "get_device_properties",
        lambda _index: SimpleNamespace(gcnArchName="gfx1100", warp_size=32),
    )
    calls = {"native": 0}
    monkeypatch.setattr(
        ext, "routing_std_gfx12_bsz1",
        lambda *_args, **_kwargs: calls.__setitem__("native", calls["native"] + 1),
    )
    selected, weights = bsm.routing_std(1, cfg, hidden, {})
    _, expected_selected, expected_weights = _oracle(hidden, gate)
    assert calls["native"] == 0
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
