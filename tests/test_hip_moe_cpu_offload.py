"""ROCm CPU expert-offload contracts: registered-SHM flag kernels, the CPU expert forward
against the gfx12 grouped kernel, and the HIP gating of unported tiers.

The flag tests exercise ext.exl3_moe_flag_write/exl3_moe_flag_wait over a registered shared
memory region: kernel waits are bounded (~30 s, calibrated at startup) and a timed-out wait
sets the abort flag rather than hanging the stream. The abort-path and watchdog tests that
need a full timeout budget run only with EXL3_HIP_SLOW_TESTS=1.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest
import torch

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm build / device not available", allow_module_level = True)

from hip_flash_quant import exl3_expert_quant
from exllamav3.ext import exllamav3_ext as ext

FLAG_REQUIRED_BINDINGS = (
    "cuda_host_register", "cuda_host_get_device_pointer", "cuda_host_unregister",
    "exl3_moe_flag_write", "exl3_moe_flag_wait",
)
for _name in FLAG_REQUIRED_BINDINGS:
    if not hasattr(ext, _name):
        pytest.skip(f"{_name} binding is unavailable", allow_module_level = True)

_slow_ok = os.environ.get("EXL3_HIP_SLOW_TESTS", "0") != "0"

# Test-local layout inside the registered region: abort word, four 64-byte flag slots,
# one payload block. The flag kernels take arbitrary device-visible u32 addresses, so the
# layout only has to stay consistent within this file.
ABORT_OFFSET = 0
FLAGS_OFFSET = 4096
FLAG_STRIDE = 64
PAYLOAD_OFFSET = 8192
PAYLOAD_WORDS = 64 * 2560 * 2 // 4
REGION_BYTES = PAYLOAD_OFFSET + PAYLOAD_WORDS * 4 + 4096

HIP_HOST_REGISTER_PORTABLE = 0x01
HIP_HOST_REGISTER_MAPPED = 0x02


def _gfx12_devices() -> list[int]:
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        arch = getattr(props, "gcnArchName", "").split(":", 1)[0]
        if arch in ("gfx1200", "gfx1201") and getattr(props, "warp_size", 0) == 32:
            devices.append(index)
    return devices


GFX12_DEVICES = _gfx12_devices()
if not GFX12_DEVICES:
    pytest.skip("requires a gfx1200/gfx1201 wave32 device", allow_module_level = True)


@pytest.fixture(scope = "module")
def registered_region():
    """One registered shared-memory region with host and device-visible bases."""
    torch.zeros(1, device = f"cuda:{GFX12_DEVICES[0]}")
    shm = shared_memory.SharedMemory(create = True, size = REGION_BYTES)
    u8 = np.frombuffer(shm.buf, dtype = np.uint8)
    u8[:] = 0
    base = u8.ctypes.data
    rc = ext.cuda_host_register(base, REGION_BYTES,
                                HIP_HOST_REGISTER_PORTABLE | HIP_HOST_REGISTER_MAPPED)
    assert rc is None, f"cuda_host_register failed: {rc}"
    gpu_base = ext.cuda_host_get_device_pointer(base)
    u32 = np.frombuffer(shm.buf, dtype = np.uint32)
    region = {"host": base, "device": gpu_base, "u32": u32}
    try:
        yield region
    finally:
        ext.cuda_host_unregister(base)
        # shm.close() fails while any export of shm.buf exists, so drop every numpy view
        # (and the dict holding them) before closing
        region.clear()
        del u32, u8
        shm.close()
        shm.unlink()


def _flag_host(region, index):
    return region["host"] + FLAGS_OFFSET + index * FLAG_STRIDE


def _flag_device(region, index):
    return region["device"] + FLAGS_OFFSET + index * FLAG_STRIDE


def _abort_device(region):
    return region["device"] + ABORT_OFFSET


def _pattern_u32(seq: int, words: int) -> np.ndarray:
    i = np.arange(words, dtype = np.uint64)
    seq_v = np.full(words, seq & 0xFFFFFFFFFFFFFFFF, dtype = np.uint64)
    v = seq_v * np.uint64(0x9E3779B97F4A7C15) + i * np.uint64(0xBF58476D1CE4E5B9)
    v ^= v >> np.uint64(30)
    v *= np.uint64(0x94D049BB133111EB)
    v ^= v >> np.uint64(31)
    return (v & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def _reduce(words: np.ndarray) -> tuple[int, int]:
    return int(words.astype(np.uint64).sum()) & 0xFFFFFFFF, int(np.bitwise_xor.reduce(words))


def _payload_host_view(region):
    return np.ctypeslib.as_array(
        ctypes.cast(region["host"] + PAYLOAD_OFFSET, ctypes.POINTER(ctypes.c_uint32)),
        shape = (PAYLOAD_WORDS,),
    )


@pytest.mark.parametrize("device_index", GFX12_DEVICES, ids = lambda i: f"cuda:{i}")
@torch.inference_mode()
def test_flag_write_wait_round_trip_advances_sequence(device_index, registered_region):
    torch.cuda.set_device(device_index)
    flag = _flag_device(registered_region, 0)
    abort = _abort_device(registered_region)
    # The first wait runs the one-time poll-rate calibration for the bounded timeout
    ext.exl3_moe_flag_write(flag, 5000)
    torch.cuda.synchronize(device_index)
    ext.exl3_moe_flag_wait(flag, 5000, abort)
    torch.cuda.synchronize(device_index)
    for seq in range(5001, 5011):
        ext.exl3_moe_flag_write(flag, seq)
        ext.exl3_moe_flag_wait(flag, seq, abort)
        torch.cuda.synchronize(device_index)
    assert int(registered_region["u32"][FLAGS_OFFSET // 4]) == 5010
    assert int(registered_region["u32"][ABORT_OFFSET // 4]) == 0


@pytest.mark.parametrize("device_index", GFX12_DEVICES, ids = lambda i: f"cuda:{i}")
@torch.inference_mode()
def test_cpu_publish_gpu_wait_acquires_payload(device_index, registered_region):
    torch.cuda.set_device(device_index)
    flag_word = (FLAGS_OFFSET + 1 * FLAG_STRIDE) // 4
    payload = _payload_host_view(registered_region)
    for k in range(3):
        seq = 6000 + k
        payload[:] = _pattern_u32(seq, PAYLOAD_WORDS)
        registered_region["u32"][flag_word] = 0
        registered_region["u32"][flag_word] = seq   # release-store on the publishing side
        ext.exl3_moe_flag_wait(_flag_device(registered_region, 1), seq,
                               _abort_device(registered_region))
        torch.cuda.synchronize(device_index)
        assert int(registered_region["u32"][ABORT_OFFSET // 4]) == 0
        assert int(registered_region["u32"][flag_word]) == seq


@pytest.mark.parametrize("device_index", GFX12_DEVICES, ids = lambda i: f"cuda:{i}")
@torch.inference_mode()
def test_gpu_publish_cpu_consume_after_d2h_copy(device_index, registered_region):
    torch.cuda.set_device(device_index)
    payload_dev = torch.empty(PAYLOAD_WORDS, dtype = torch.int32, device = f"cuda:{device_index}")
    payload_host = torch.frombuffer(registered_region["u32"].base, dtype = torch.int32,
                                    count = PAYLOAD_WORDS, offset = PAYLOAD_OFFSET)
    flag_word = (FLAGS_OFFSET + 2 * FLAG_STRIDE) // 4
    for k in range(3):
        seq = 7000 + k
        pattern = _pattern_u32(seq, PAYLOAD_WORDS)
        expected = _reduce(pattern)
        registered_region["u32"][flag_word] = 0
        payload_dev.copy_(torch.from_numpy(pattern.astype(np.int32)))
        payload_host.copy_(payload_dev)      # stream-ordered D2H into the mapped payload
        ext.exl3_moe_flag_write(_flag_device(registered_region, 2), seq)
        torch.cuda.synchronize(device_index)
        deadline = time.time() + 5.0
        while np.int32(int(registered_region["u32"][flag_word]) - seq) < 0 and time.time() < deadline:
            time.sleep(0.0005)
        assert np.int32(int(registered_region["u32"][flag_word]) - seq) >= 0, \
            "CPU never observed the published flag"
        assert _reduce(_payload_host_view(registered_region).copy()) == expected


@pytest.mark.parametrize("device_index", GFX12_DEVICES, ids = lambda i: f"cuda:{i}")
@torch.inference_mode()
def test_host_write_unblocks_pending_kernel_wait(device_index, registered_region):
    torch.cuda.set_device(device_index)
    flag_word = (FLAGS_OFFSET + 3 * FLAG_STRIDE) // 4
    registered_region["u32"][flag_word] = 0
    ext.exl3_moe_flag_wait(_flag_device(registered_region, 3), 987654,
                           _abort_device(registered_region))
    time.sleep(0.05)               # the wait kernel is resident on the stream by now
    registered_region["u32"][flag_word] = 987654   # watchdog path: host satisfies the value
    done = torch.cuda.Event()
    done.record()
    deadline = time.time() + 10.0
    while not done.query() and time.time() < deadline:
        time.sleep(0.0005)
    assert done.query(), "host write did not unblock the pending wait"
    assert int(registered_region["u32"][ABORT_OFFSET // 4]) == 0


@pytest.mark.skipif(not _slow_ok, reason = "needs the full ~30 s wait budget (EXL3_HIP_SLOW_TESTS=1)")
@torch.inference_mode()
def test_unsatisfiable_wait_is_bounded_and_sets_abort(registered_region):
    device_index = GFX12_DEVICES[0]
    torch.cuda.set_device(device_index)
    registered_region["u32"][ABORT_OFFSET // 4] = 0
    start = time.perf_counter()
    ext.exl3_moe_flag_wait(_flag_device(registered_region, 0), 0x7ABCDEF0,
                           _abort_device(registered_region))
    torch.cuda.synchronize(device_index)
    elapsed = time.perf_counter() - start
    assert 25.0 <= elapsed <= 45.0, f"wait returned after {elapsed:.1f}s"
    assert int(registered_region["u32"][ABORT_OFFSET // 4]) == 1
    # A satisfied wait on the same stream proceeds after the abort path completed
    registered_region["u32"][ABORT_OFFSET // 4] = 0
    ext.exl3_moe_flag_write(_flag_device(registered_region, 0), 0x7ABCDEF0)
    start = time.perf_counter()
    ext.exl3_moe_flag_wait(_flag_device(registered_region, 0), 0x7ABCDEF0,
                           _abort_device(registered_region))
    torch.cuda.synchronize(device_index)
    assert time.perf_counter() - start < 1.0


MODEL = Path(os.environ.get(
    "EXL3_FLASH_TEST_MODEL", "~/Models/Qwen3.8-Flash-Next-exl3"
)).expanduser()
LAYER = 0
N_EXPERTS = 4
HIDDEN, INTERM, TOP_K = 2560, 640, 10


def _expert_tensors():
    """gate/up/down trellis+suh+svh for experts [0, N_EXPERTS) of layer LAYER, CPU-resident."""
    keys = {
        f"model.language_model.layers.{LAYER}.mlp.experts.{e}.{proj}.{suffix}"
        for e in range(N_EXPERTS) for proj in ("gate_proj", "up_proj", "down_proj")
        for suffix in ("trellis", "suh", "svh")
    }
    tensors = dict.fromkeys(keys)
    import safetensors.torch as st
    for path in sorted(MODEL.glob("model-*.safetensors")):
        with st.safe_open(str(path), framework = "pt", device = "cpu") as fh:
            for key in list(tensors):
                if key in fh.keys():
                    tensors[key] = fh.get_tensor(key)
    missing = [key for key, value in tensors.items() if value is None]
    assert not missing, f"missing expert tensors: {missing[:3]}"
    return tensors


@pytest.fixture(scope = "module")
def k3_expert_layer():
    """Real K3/mul1 expert tensors from the checkpoint, plus the CPU worker registration."""
    if not MODEL.is_dir():
        pytest.skip(f"Flash model not found: {MODEL} (set EXL3_FLASH_TEST_MODEL)")
    codebook, bits = exl3_expert_quant(MODEL)
    if bits["routed"] != 3 or codebook != "mul1":
        pytest.skip(
            f"CPU expert forward needs K3/mul1 routed experts; {MODEL} has "
            f"K{bits['routed']}/{codebook} (set EXL3_FLASH_TEST_MODEL)"
        )
    if not (hasattr(ext, "exl3_moe_cpu_make_layer") and hasattr(ext, "exl3_moe_cpu_forward")
            and hasattr(ext, "exl3_moe_gfx12_k3")):
        pytest.skip("CPU offload or grouped kernel bindings are unavailable")
    tensors = _expert_tensors()

    def part(proj, suffix, half = False):
        out = [
            tensors[f"model.language_model.layers.{LAYER}.mlp.experts.{e}.{proj}.{suffix}"]
            for e in range(N_EXPERTS)
        ]
        return [t.half() for t in out] if half else out

    groups = [part(p, s, s != "trellis") for p in ("gate_proj", "up_proj", "down_proj")
              for s in ("trellis", "suh", "svh")]
    handle = ext.exl3_moe_cpu_make_layer(*groups, [], [], [], 0, 0.0, 0)
    # Pointer tables alias the GPU-resident tensors, so those must outlive every kernel
    # call that uses the tables; building them here keeps them alive for the fixture's scope
    device = torch.device("cuda", GFX12_DEVICES[0])
    alive, tables = [], []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for suffix in ("trellis", "suh", "svh"):
            moved = [
                tensors[f"model.language_model.layers.{LAYER}.mlp.experts.{e}.{proj}.{suffix}"]
                    .to(device)
                for e in range(N_EXPERTS)
            ]
            if suffix != "trellis":
                moved = [t.half() for t in moved]
            alive.extend(moved)
            tables.append(_ptr_table(moved, device))
    torch.cuda.synchronize(device)
    return tensors, handle, alive, tables


def _ptr_table(tensors, device):
    return torch.tensor([t.data_ptr() for t in tensors], dtype = torch.long, device = device)


@torch.inference_mode()
def test_cpu_expert_forward_matches_grouped_kernel(k3_expert_layer):
    """The CPU mul1 kernel quantizes activations to int8, so CPU-vs-GPU tolerance is the
    int8-activation approximation envelope; the GPU path must match the fp16 reconstruct
    reference tightly, and both paths must be deterministic."""
    device_index = GFX12_DEVICES[0]
    torch.cuda.set_device(device_index)
    tensors, handle, _alive, tables = k3_expert_layer
    generator = torch.Generator().manual_seed(20260907)
    for rows in (1, 4, 16):
        y = torch.randn(rows, HIDDEN, generator = generator, dtype = torch.half) * 0.5
        selected = torch.randint(0, N_EXPERTS, (rows, TOP_K), generator = generator,
                                 dtype = torch.long)
        if rows >= 4:
            selected[1, 0] = -1            # GPU-resident-masked sentinel (split semantics)
            selected[2, 3] = selected[0, 0]   # duplicate id
        weights = torch.rand(rows, TOP_K, generator = generator, dtype = torch.half) * 0.4 + 0.05

        out_cpu = torch.empty(rows, HIDDEN, dtype = torch.float)
        ext.exl3_moe_cpu_forward(handle, y.clone(), selected.clone(), weights, out_cpu, 8)
        out_cpu2 = torch.empty(rows, HIDDEN, dtype = torch.float)
        ext.exl3_moe_cpu_forward(handle, y.clone(), selected.clone(), weights, out_cpu2, 8)
        assert torch.equal(out_cpu, out_cpu2), "CPU forward is not deterministic"

        assignments = rows * TOP_K
        gu_had = torch.empty(2 * assignments, HIDDEN, dtype = torch.half, device = device_index)
        gu_out = torch.empty(2 * assignments, INTERM, dtype = torch.half, device = device_index)
        down_had = torch.empty(assignments, INTERM, dtype = torch.half, device = device_index)
        down_out = torch.empty(assignments, HIDDEN, dtype = torch.float, device = device_index)
        output = torch.empty(rows, HIDDEN, dtype = torch.float, device = device_index)
        ext.exl3_moe_gfx12_k3(
            y.to(device_index), output, selected.to(device_index), weights.to(device_index),
            *tables, gu_had, gu_out, down_had, down_out,
        )
        torch.cuda.synchronize(device_index)

        rel_rms = ((out_cpu - output.cpu()).pow(2).mean().sqrt()
                   / (output.cpu().pow(2).mean().sqrt() + 1e-9)).item()
        assert rel_rms <= 0.02, f"rows={rows}: CPU/GPU rel RMS {rel_rms:.4f} exceeds the int8 envelope"

        reference = torch.zeros(rows, HIDDEN, dtype = torch.float, device = device_index)
        expert_ids = sorted({int(v) for row in selected.tolist() for v in row if 0 <= v < N_EXPERTS})
        for expert in expert_ids:
            def expert_suffix(proj, suffix):
                return tensors[f"model.language_model.layers.{LAYER}.mlp.experts.{expert}.{proj}.{suffix}"] \
                    .to(device_index)
            slots = (selected == expert).nonzero()[:, 0].to(device_index)
            xg = y.to(device_index).index_select(0, slots)
            wseg = weights.to(device_index)[selected == expert].float().unsqueeze(1)

            def projection(x_in, trellis, suh, svh):
                k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
                xh = torch.empty_like(x_in)
                ext.had_r_128(x_in, xh, suh, None, 1.0)
                w = torch.empty(k * n, dtype = torch.half, device = device_index).view(k, n)
                ext.reconstruct(w, trellis, 3, False, True)
                o = torch.empty(x_in.shape[0], n, dtype = torch.half, device = device_index)
                ext.hgemm_recon(xh, w, o)
                ext.had_r_128(o, o, None, svh, 1.0)
                return o

            gate = projection(xg, expert_suffix("gate_proj", "trellis"),
                              expert_suffix("gate_proj", "suh").half(),
                              expert_suffix("gate_proj", "svh").half())
            up = projection(xg, expert_suffix("up_proj", "trellis"),
                            expert_suffix("up_proj", "suh").half(),
                            expert_suffix("up_proj", "svh").half())
            act = (torch.nn.functional.silu(gate.float()) * up.float()).half()
            ah = torch.empty_like(act)
            down_suh = expert_suffix("down_proj", "suh").half()
            down_svh = expert_suffix("down_proj", "svh").half()
            ext.had_r_128(act, ah, down_suh, None, 1.0)
            down_trellis = expert_suffix("down_proj", "trellis")
            kd, nd = down_trellis.shape[0] * 16, down_trellis.shape[1] * 16
            wd = torch.empty(kd * nd, dtype = torch.half, device = device_index).view(kd, nd)
            ext.reconstruct(wd, down_trellis, 3, False, True)
            od = torch.empty(act.shape[0], nd, dtype = torch.half, device = device_index)
            ext.hgemm_recon(ah, wd, od)
            ext.had_r_128(od, od, None, down_svh, 1.0)
            reference.index_add_(0, slots, od.float()[:, :HIDDEN] * wseg)
        assert (output - reference).abs().max().item() <= 1e-3, \
            "grouped kernel diverged from the reconstruct reference"


def test_split_fused_issue_disabled_on_hip():
    """The fused split issue/collect tier needs the CUDA memops path and stays off on HIP."""
    from exllamav3.modules import block_sparse_mlp_cpu
    assert torch.version.hip
    assert block_sparse_mlp_cpu._split_fused is False
