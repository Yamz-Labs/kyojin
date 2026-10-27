from __future__ import annotations
from typing_extensions import override
import os
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.tensor import to2
from . import Module, Linear
from .multilinear import MultiLinear
from .moe_mixedk import MixedKStore, MixedKUnsupported, expert_ks, layer_is_mixed, check_expert_bits, \
    PrunedLinear, pruned_experts, parse_expert_bits
from ..ext import exllamav3_ext as ext
from dataclasses import dataclass
from .mlp import MLP, GatedMLP
from .rmsnorm import RMSNorm
from .layernorm import LayerNorm
from .block_sparse_mlp_cpu import BlockSparseMLP_CPU
from .moe_batch_recon import PAD_MAX
from ..model.model_tp_alloc import TPAllocation
from ..util import profile_opt
from ..util.tensor import g_tensor_cache, buffered_interleaved_arange
from .quant.exl3 import dec_workspace
from ..util.row_inv import ROW_INV_MAX_ROWS, job_slices
from .block_sparse_mlp_routing import (
    RoutingCFG, ROUTING_ACT_SIGMOID, ROUTING_ACT_SQRTSP,
    routing_std, routing_std_bias, routing_ds3, routing_dots, routing_sqrtsp, routing_sqrtsp_hash,
    _HIP_ROUTER_HIDDEN, _HIP_ROUTER_EXPERTS, _HIP_ROUTER_TOP_K,
    _HIP_GROUPED_MAX_ROWS, _HIP_PREFILL_MIN_ROWS, _HIP_PREFILL_MAX_ROWS, _HIP_PREFILL_MAX_EXPERT_ROWS,
    _hip_grouped_rows_eligible, _hip_prefill_rows_eligible, _prepare_hip_router_gate_t, _MOE_VALU,
)

# Row capacity of the fused MoE kernel's per-group temp buffers (experts with more assigned
# rows take the reconstruct paths); EXL3_MOE_FUSED_ROWS overrides for tuning sweeps
TEMP_ROWS_FUSED = int(os.environ.get("EXL3_MOE_FUSED_ROWS", 128))
TEMP_ROWS_GRAPH = 32
# Batched reconstruct tier for the experts above the fused kernel's row capacity at prefill
# (moe_batch_recon.py); EXL3_MOE_BATCH_RECON=0 restores the per-expert reconstruct loop
BATCH_RECON = os.environ.get("EXL3_MOE_BATCH_RECON", "1") != "0"
# Row tiles for the fused kernel: experts with more than MTILE_T1 rows run through a 32-row tile
# instance, more than MTILE_T2 through a 64-row one (32 for the N = 256 shape), each its own
# launch over its expert range (mul1 codebook only). EXL3_MOE_MTILE=0 keeps the single 16-row
# launch
MTILE = os.environ.get("EXL3_MOE_MTILE", "1") != "0"
MTILE_T1, MTILE_T2 = 16, 32
# Fused-kernel row capacity per expert when the wide tiles apply: with them the fused kernel
# beats the batched reconstruct tier up to 256 rows (Qwen3.8 4k chunk: 128 -> 256 rows +3%)
FUSED_ROWS_WIDE = int(os.environ.get("EXL3_MOE_FUSED_ROWS_WIDE", 256))
# Grouped prefill tier (exl3_moe_grouped: token-parallel hadamard kernels + two grouped one-CTA-per-tile GEMMs) for the
# experts above MOEG_LO rows, instead of the persistent expert-group kernel's 64-row launch. Needs the deterministic slot
# path, the mul1 codebook and 2..6 bit tensors; EXL3_MOE_GROUPED=0 restores the persistent kernel
MOEG = os.environ.get("EXL3_MOE_GROUPED", "1") != "0"
MOEG_LO = int(os.environ.get("EXL3_MOE_GROUPED_LO", 33))
# Row capacity of the grouped tier per expert (the persistent kernel is capped by its temp buffers, this tier is not); experts above
# it take the batched reconstruct tier as before
MOEG_ROWS = int(os.environ.get("EXL3_MOE_GROUPED_ROWS", 1 << 20))
# Deterministic (slot + gather) accumulation for the fused kernel's outputs; EXL3_MOE_FUSED_DET=0
# restores the atomic adds
FUSED_DET = os.environ.get("EXL3_MOE_FUSED_DET", "1") != "0"
MAX_BSZN = 8  # must match MAX_BSZN in exllamav3_ext/libtorch/blocksparse_mlp.h
# R-row union MoE (exl3_dec_moe_union): one kernel call carries at most MAX_BSZN (= MOE_R_MAX in exl3_dec.cu) rows; a verify step of 9..UNION_ROWS
# rows runs as job-aligned slices of at most MAX_BSZN rows (each slice is the exact 2..8 row call, rows are independent in an MoE block).
# ROCm only (the CUDA fused kernels have their own COOP_ROWS); EXL3_ROW_INV_MAX_ROWS=8 restores the single-call cap.
UNION_ROWS = max(MAX_BSZN, ROW_INV_MAX_ROWS) if torch.version.hip else MAX_BSZN
# Row capacity of the fused decode kernels (exl3_moe_coop): up to this many token rows (bounded by 256 slots = rows * top_k) take
# them instead of the prefill-shaped exl3_moe kernel or row chunks; rows <= MAX_BSZN run exactly as before. The kernels group the
# slots by expert, so each picked expert is decoded once for all of its rows. CUDA only; EXL3_MOE_COOP_ROWS=8 restores the old cap
COOP_ROWS = max(MAX_BSZN, int(os.environ.get("EXL3_MOE_COOP_ROWS", "32"))) if torch.version.hip is None else MAX_BSZN
# Row chunking: 9..EXL3_MOE_ROW_CHUNK_MAX rows of a pack without a fused MoE kernel run as <= MAX_BSZN-row slices through
# the bszN decode kernels instead of the per-expert tier (host readback of the expert counts + one launch group per expert,
# per layer); rows are independent in an MoE block. 0 disables. Measured on an RTX 3090 (Qwen3-30B-A3B 3INST)
_ROW_CHUNK_MAX = int(os.environ.get("EXL3_MOE_ROW_CHUNK_MAX", "64"))
# Fused exl3_moe kernel for 3INST packs (no mcg / mul1 codebook, e.g. upstream-quantized public packs), CUDA. Without it their
# > MAX_BSZN-row blocks take the row chunks above or the per-expert tier. 0 restores that.
_FUSED_3INST = os.environ.get("EXL3_MOE_FUSED_3INST", "1") != "0"
# ... but up to this many rows the row chunks stay faster (2 bszN slices beat one exl3_moe launch at 9-16 rows on a 3090)
_FUSED_3INST_MIN = int(os.environ.get("EXL3_MOE_FUSED_3INST_MIN", "16"))
# Sync-free fused tier bound: top-k picks distinct experts, so no expert holds more rows than there are tokens and
# num_tokens <= fused_rows already keeps every expert inside the fused tier (the num_tokens * top_k bound skipped the
# expert-count readback only up to fused_rows / top_k tokens). EXL3_MOE_SYNC_FREE_TOKENS=0 restores the top-k bound.
# CUDA default only (measured there); ROCm keeps the top-k bound unless set to 1
SYNC_FREE_TOKENS = os.environ.get("EXL3_MOE_SYNC_FREE_TOKENS", "1" if torch.version.hip is None else "0") != "0"


def _union_slices(x: torch.Tensor, rows: int) -> list:
    """Row ranges (lo, hi) of the union MoE kernel calls for `rows` token rows: one call up to MAX_BSZN rows, else job-aligned slices
    (util.row_inv.job_slices). x is the MoE input (jobs, rows per job, H) of a verify forward."""
    return job_slices(rows, int(x.shape[-2]) if x.dim() >= 3 else 1, MAX_BSZN)


# Expert-deduplicated grouped MoE (verify batches decode each picked expert once for all of its
# rows); EXL3_MOE_DEDUP=0 restores the per-assignment kernel
_MOE_DEDUP = os.environ.get("EXL3_MOE_DEDUP", "1") != "0"

def _nc2_shared_fuse() -> bool:
    return os.environ.get("EXL3_NC2_SHARED_FUSE", "0") == "1"

def _moe_sync_free_count() -> bool:
    # Default on: torch.bincount on a device tensor synchronizes the host once per MoE
    # layer per chunk (48 layers), locking the host to the device. The scatter_add_
    # histogram below is device-only. EXL3_MOE_SYNC_FREE_COUNT=0 restores bincount.
    return os.environ.get("EXL3_MOE_SYNC_FREE_COUNT", "1") != "0"


# Persistent all-ones source for the sync-free expert histogram, one per device. Sized
# to the max assignments of the gfx12 prefill route (2048 rows x top-k 10).
_moe_sync_free_ones = {}


# Grouped WMMA prefill: minimum chunk rows (below it the per-expert GEMV path wins) and one
# shared, grow-only workspace per device (layers run one at a time)
_HIP_WMMA_MIN_ROWS = int(os.environ.get("EXL3_MOE_WMMA_MIN_ROWS", "32"))
_hip_wmma_ws = {}
# Debug: EXL3_MOE_WMMA_STATS=1 records each call's per-expert counts (host sync per layer)
_HIP_WMMA_STATS = [] if os.environ.get("EXL3_MOE_WMMA_STATS") else None


def _hip_wmma_workspace(device, assignments: int, hidden: int, interm: int, experts: int) -> dict:
    ws = _hip_wmma_ws.get(device)
    need = (assignments, hidden, interm, experts)
    if ws is None or any(a < b for a, b in zip(ws["cap"], need)):
        cap = tuple(max(a, b) for a, b in zip(ws["cap"], need)) if ws is not None else need
        a, h, i, e = cap
        # The grouped GEMM's last tile of an expert reads/writes up to a tile (<= 128 rows) past the
        # last assignment row: a tight tail faults when the allocator places the buffer at a segment end
        # (seen on Qwen3.8-Flash-Next once the surrounding temporaries stopped padding the pool)
        slack = 128 * max(h, i)
        _hip_wmma_ws[device] = None
        ws = {
            "cap": cap,
            "gu_had": torch.empty((2 * a * h + slack,), dtype = torch.half, device = device),
            "gu_out": torch.empty((2 * a * i + slack,), dtype = torch.half, device = device),
            "offsets": torch.empty((e + 1,), dtype = torch.long, device = device),
            "inverse": torch.empty((a,), dtype = torch.long, device = device),
            "tiles": torch.empty((a // 64 + e + 1,), dtype = torch.int, device = device),
            "tile_count": torch.empty((1,), dtype = torch.int, device = device),
        }
        _hip_wmma_ws[device] = ws
    return ws


def _scatter_expert_count(flat_expert_local: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Sync-free int64 histogram identical to torch.bincount(flat_expert_local, minlength = num_bins)."""
    n = flat_expert_local.numel()
    dev = flat_expert_local.device
    if n <= _HIP_PREFILL_MAX_EXPERT_ROWS:
        ones = _moe_sync_free_ones.get(dev)
        if ones is None:
            ones = torch.ones(_HIP_PREFILL_MAX_EXPERT_ROWS, dtype = torch.long, device = dev)
            _moe_sync_free_ones[dev] = ones
        src = ones[:n]
    else:
        # Rare: paths beyond the gfx12 prefill envelope (e.g. chunk > 512 rows). A fresh
        # ones tensor here is still device-side (no host sync).
        src = torch.ones(n, dtype = torch.long, device = dev)
    expert_count = torch.zeros(num_bins, dtype = torch.long, device = dev)
    expert_count.scatter_add_(0, flat_expert_local, src)
    return expert_count


# EXL3_MOE_ALIGN_CHUNK (default 4096, 0 = off): a prefill chunk of more than this many rows runs the grouped
# expert GEMM as ONE launch whose per-expert row lists are laid out exactly like the chunk-by-chunk path:
# every (expert, sub-chunk) segment starts on an even row (the GEMM result of a row depends on its row parity
# inside the expert list), so the output is bit-identical to running the sub-chunks one after the other.
_MOE_ALIGN_CHUNK = int(os.environ.get("EXL3_MOE_ALIGN_CHUNK", "4096"))


def _aligned_expert_layout(flat: torch.Tensor, num_tokens: int, top_k: int, num_experts: int, chunk: int):
    """flat: (num_tokens * top_k) int64 expert ids, slot = token * top_k + k. Returns (order, sel_flat, dummy_tokens):
    order lists the assignment slots grouped by expert, each (expert, chunk of `chunk` tokens) segment in slot order
    and padded to an even length with dummy slots; sel_flat is the expert id of every slot including the dummy ones
    (slots >= num_tokens * top_k belong to dummy tokens num_tokens .. num_tokens + dummy_tokens - 1). No host sync
    (EXL3_MOE_ALIGN_SYNC_FREE, default on)."""
    dev = flat.device
    a0 = num_tokens * top_k
    nc = (num_tokens + chunk - 1) // chunk
    nseg = num_experts * nc
    dummy_tokens = (nseg + top_k - 1) // top_k
    total = a0 + dummy_tokens * top_k
    key = flat * nc + torch.arange(a0, device = dev) // (top_k * chunk)
    srt = key.argsort(stable = True)
    # torch.bincount on a device tensor syncs the host (min/max reduction + D2H) once per MoE layer: with big chunks (> 4096 rows)
    # that drained the queue 36-39 times per forward. Same int64 histogram, device only (EXL3_MOE_ALIGN_SYNC_FREE=0 = bincount, read per call).
    kc = _scatter_expert_count(key, nseg) if os.environ.get("EXL3_MOE_ALIGN_SYNC_FREE", "1") != "0" else torch.bincount(key, minlength = nseg)
    odd = kc & 1
    padbefore = torch.cumsum(odd, 0) - odd
    pos = torch.arange(a0, device = dev) + padbefore[key[srt]]
    pad_pos = torch.cumsum(kc, 0) + padbefore
    pad_pos = torch.where(odd == 1, pad_pos, torch.full_like(pad_pos, total))
    pad_slot = a0 + padbefore
    order = torch.arange(total + 1, device = dev)
    order[pos] = srt
    order[pad_pos] = pad_slot
    sel_flat = torch.full((total + 1,), num_experts - 1, dtype = flat.dtype, device = dev)
    sel_flat[:a0] = flat
    sel_flat[torch.where(odd == 1, pad_slot, torch.full_like(pad_slot, total))] = \
        torch.arange(nseg, device = dev) // nc
    return order[:total], sel_flat[:total], dummy_tokens


@dataclass
class FusedBuffers:
    temp_state_g: torch.Tensor
    temp_state_u: torch.Tensor
    temp_intermediate_g: torch.Tensor
    temp_intermediate_u: torch.Tensor


_SHARED_SIDE_HIP = bool(torch.version.hip)   # EXL3_SHARED_SIDE=1 is read per call (A/B inside one process)
_side_state: dict = {}


def _side(device):
    st = _side_state.get(device)
    if st is None:
        st = _side_state[device] = (torch.cuda.Stream(device = device), torch.cuda.Event(), torch.cuda.Event())
    return st


@dataclass
class HIPGroupedBuffers:
    gu_had: torch.Tensor
    gu_out: torch.Tensor
    down_had: torch.Tensor
    down_out: torch.Tensor
    output: torch.Tensor


@dataclass
class HIPPrefillBuffers:
    gu_had: torch.Tensor
    gu_out: torch.Tensor
    down_out: torch.Tensor
    output: torch.Tensor
    expert_offsets: torch.Tensor
    inverse_order: torch.Tensor
    expert_chunks: torch.Tensor
    chunk_count: torch.Tensor


@dataclass
class ExpertsCFG:
    yh: torch.Tensor
    interm_g: torch.Tensor
    interm_u: torch.Tensor
    interm_a: torch.Tensor
    out_d: torch.Tensor
    out_d2: torch.Tensor
    min_expert: int
    max_expert: int
    out_bszn: torch.Tensor | None = None   # (MAX_BSZN, H) fp32 routed sum of the bsz<=MAX_BSZN path


# Routed-weighted per-expert Hessians for the quantizer (see Linear.cap_rkey, convert_model.build_rwh_blends)
_RWH = os.environ.get("EXL3_RWH", "0") == "1"

class _RouteTrace:
    """Diagnostic (EXL3_ROUTE_TRACE=path): append every MoE layer's router top-k, one record per forward
    (layer id, rows, k, segment, int16 experts). Read-only hook, outputs unchanged; SIGUSR1 starts a new segment
    (= a new conversation). Syncs once per layer, run with graphs off."""
    def __init__(self, path):
        import signal, threading
        self.f = open(path, "ab", buffering = 1 << 20)
        self.keys = {}
        self.seg = 0
        self.kf = open(path + ".keys", "a")
        self.lock = threading.Lock()
        signal.signal(signal.SIGUSR1, self._seg)

    def _seg(self, *a):
        self.seg += 1
        self.f.flush()

    def record(self, key, bsz, sel):
        import numpy as np
        if torch.cuda.is_current_stream_capturing():
            return
        kid = self.keys.get(key)
        if kid is None:
            kid = self.keys[key] = len(self.keys)
            self.kf.write(f"{kid} {key}\n"); self.kf.flush()
        a = sel.detach().to(torch.int16).cpu().numpy().reshape(bsz, -1)
        with self.lock:
            np.array([kid, bsz, a.shape[1], self.seg], dtype = np.int32).tofile(self.f)
            a.tofile(self.f)


_ROUTE_TRACE = _RouteTrace(os.environ["EXL3_ROUTE_TRACE"]) if os.environ.get("EXL3_ROUTE_TRACE") else None


class BlockSparseMLP(BlockSparseMLP_CPU, Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        num_local_experts: int | None = None,
        latent_size: int | None = None,
        key_latent_in: str | None = None,
        key_latent_out: str | None = None,
        latent_in: Linear | None = None,
        latent_out: Linear | None = None,
        latent_hq_bits: int = 0,
        key_up: str | None = None,
        key_gate: str | None = None,
        key_down: str | None = None,
        key_gate_split: str | None = None,
        key_up_split: str | None = None,
        key_gate_up_split: str | None = None,
        key_down_split: str | None = None,
        key_routing_gate: str | None = None,
        key_shared_gate: str | None = None,
        key_e_score_bias: str | None = "gate.e_score_correction_bias",
        key_e_score_bias_vl: str | None = None,
        key_tid2eid: str | None = None,
        key_per_expert_scale: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype = None,
        activation_fn: str = "silu",
        act_limit: float = 0.0,
        interm_dtype: torch.dtype = None,
        fused_fp16_interm: bool = True,
        interm_div: float = 1.0,
        router_type: str = "std",
        routing_gate: Linear | None = None,
        shared_gate: Linear | None = None,
        routed_scaling_factor: float | None = None,
        n_group: int | None = None,
        topk_group: int | None = None,
        shared_experts: MLP | GatedMLP | None = None,
        shared_experts_post_norm: RMSNorm | LayerNorm | None = None,
        router_pre_norm: RMSNorm | LayerNorm | None = None,
        routed_pre_norm: RMSNorm | LayerNorm | None = None,
        routed_post_norm: RMSNorm | LayerNorm | None = None,
        gates: list[Linear | Module] = None,
        ups: list[Linear | Module] = None,
        downs: list[Linear | Module] = None,
        routing_first: int | None = None,
        routing_last: int | None = None,
        routing_device: int | None = None,
        transposed_load: bool = True,
        transpose_fused_weights: bool = True,
        ftranspose_after_load: bool = True,
        frange_dim: int = 0,
        gate_up_interleaved: bool = False,
        alt_residual_channel: bool = False,
        qbits_key: str = "bits"
    ):
        super().__init__(config, key, None)

        self.interm_dtype = interm_dtype
        # The fused prefill tier (exl3_moe) keeps its gate/up intermediates in fp16 no matter
        # what interm_dtype says (exl3_moe.cu: TORCH_CHECK_DTYPE(temp_intermediate_*, kHalf)),
        # so on a model configured for fp32 intermediates *because* fp16 overflows, that tier
        # silently puts the fp16 range back for every expert under its row cap -- which at
        # prefill is nearly all of them. fused_fp16_interm = False drops the tier when
        # interm_dtype is not fp16, leaving the graph / batched-reconstruct / dequant paths,
        # all of which honour it. Default True = upstream behaviour for every other model
        self.fused_fp16_interm = fused_fp16_interm
        self.interm_div = interm_div
        self.router_type = router_type
        if interm_div != 1.0:
            assert router_type in ("dots", "ds3", "std"), \
                "interm_div requires a router type that can fold the compensation into the routing weights"
            if router_type != "std":
                routed_scaling_factor = (routed_scaling_factor if routed_scaling_factor is not None else 1.0) * interm_div
        self.activation_fn = activation_fn
        self.intermediate_size = intermediate_size
        self.intermediate_size_padded = (intermediate_size + 127) // 128 * 128
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.f_threshold = min(self.num_experts // self.num_experts_per_tok, 4)
        self.num_local_experts = num_local_experts if num_local_experts is not None else num_experts
        self.hidden_size = hidden_size
        # Latent MoE (Nemotron-3 Super): the routed experts run at latent_size, between a
        # projection of the block input down to that width and a projection of the routed sum
        # back up. The router and the shared experts see the full-width block input
        self.latent_size = latent_size
        self.expert_size = latent_size or hidden_size
        self.router_type = router_type
        self.act_limit = act_limit
        self.alt_residual_channel = alt_residual_channel

        self.routing_first = routing_first
        self.routing_last = routing_last
        self.routing_device = routing_device

        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group

        assert out_dtype in (torch.float, None), \
            f"BlockSparseMLP output dtype must be float"

        assert shared_experts is None or shared_experts.out_dtype in (torch.float, None), \
            f"Shared experts output dtype must be float"

        assert num_experts_per_tok <= TEMP_ROWS_GRAPH, \
            f"Too many experts per token, max supported is {TEMP_ROWS_GRAPH}"

        if routing_gate is None and key_routing_gate is None:
            self.routing_gate = None
        elif routing_gate is None:
            self.routing_gate = Linear(
                config = config,
                key = f"{key}.{key_routing_gate}",
                in_features = hidden_size,
                out_features = num_experts,
                qmap = None,
                out_dtype = torch.half,
                pad_to = 1,
                # The gfx12 router transpose is derived during this module's load. Its fp16
                # source must therefore be filled before deferred loading completes.
                no_defer_load = bool(
                    torch.version.hip and router_type == "std" and
                    hidden_size == _HIP_ROUTER_HIDDEN and
                    num_experts == _HIP_ROUTER_EXPERTS and
                    num_experts_per_tok == _HIP_ROUTER_TOP_K and
                    key_per_expert_scale is None and interm_div == 1.0
                ),
            )
            self.register_submodule(self.routing_gate)
        else:
            self.routing_gate = routing_gate
            self.register_submodule(self.routing_gate)

        if shared_gate is None and key_shared_gate is None:
            self.shared_gate = None
        elif shared_gate is None:
            self.shared_gate = Linear(
                config = config,
                key = f"{key}.{key_shared_gate}",
                in_features = hidden_size,
                out_features = 1,
                qmap = None,
                out_dtype = torch.float,
                pad_to = 1,
            )
            self.register_submodule(self.shared_gate)
        else:
            self.shared_gate = shared_gate
            self.register_submodule(self.shared_gate)

        if latent_in is None and key_latent_in is None:
            assert latent_size is None, "latent_size requires the latent projection keys"
            self.latent_in = None
            self.latent_out = None
        else:
            assert latent_size, "latent projections require latent_size"
            # The down projection shares the block input (and its Hessian) with the shared
            # experts; the up projection's input is the routed sum
            self.latent_in = latent_in if latent_in is not None else Linear(
                config = config,
                key = f"{key}.{key_latent_in}",
                in_features = hidden_size,
                out_features = latent_size,
                qmap = qmap + ".input" if qmap else None,
                out_dtype = torch.half,
                select_hq_bits = latent_hq_bits,
                qgroup = key + ".latent",
                qbits_key = qbits_key,
            )
            self.latent_out = latent_out if latent_out is not None else Linear(
                config = config,
                key = f"{key}.{key_latent_out}",
                in_features = latent_size,
                out_features = hidden_size,
                qmap = qmap + ".latent_out" if qmap else None,
                out_dtype = torch.float,
                select_hq_bits = latent_hq_bits,
                qgroup = key + ".latent",
                qbits_key = qbits_key,
            )
            self.register_submodule(self.latent_in)
            self.register_submodule(self.latent_out)

        # Non-gated experts (NemotronH: up/down with relu2). Gateless relu2 rides every quantized
        # fast path (relu(u) * u = relu2(u): the BC graphs and mgemm loops skip the gate GEMM, the
        # fused kernel synthesizes the gate lane, the batched reconstruct tier and the CPU worker
        # take up/down only); other gateless activations run the dense per-expert path
        self.gated = (
            key_gate is not None or key_gate_split is not None or key_gate_up_split is not None or
            (gates is not None and len(gates) > 0)
        )

        if gates is not None:
            assert ups is not None and (not self.gated or len(ups) == len(gates))
            assert downs is not None and len(downs) == len(ups)
            self.num_slices = len(ups)
            self.gates = gates
            self.ups = ups
            self.downs = downs

        else:
            self.gates = []
            self.ups = []
            self.downs = []

            # The experts' input is the block input, or the latent projection of it (a
            # different activation, so a different Hessian) in a latent MoE
            expert_qmap = (qmap + (".latent" if latent_size else ".input")) if qmap else None
            # K = 0 experts (pack config expert_bits == 0): no tensors, stand-ins that are never loaded
            qcfg = getattr(config, "config_dict", {}).get("quantization_config") if config is not None else None
            pruned = pruned_experts(qcfg, key, self.num_local_experts) if self.num_local_experts == num_experts else set()
            for idx in range(self.num_local_experts):

                if idx in pruned:
                    pk = f"{key}.experts.{idx}"
                    self.ups.append(PrunedLinear(pk + ".w3", self.expert_size, intermediate_size))
                    if self.gated:
                        self.gates.append(PrunedLinear(pk + ".w1", self.expert_size, intermediate_size))
                    self.downs.append(PrunedLinear(pk + ".w2", intermediate_size, self.expert_size))
                    continue

                fkey_gate, fkey_up, fkey_down = (
                    f"{key}.{key_gate_up_split}" if key_gate_up_split else
                    f"{key}.{key_gate_split}" if key_gate_split else
                    None,
                    f"{key}.{key_gate_up_split}" if key_gate_up_split else
                    f"{key}.{key_up_split}" if key_up_split else
                    None,
                    f"{key}.{key_down_split}" if key_down_split else
                    None
                )

                gate = None if not self.gated else Linear(
                    config = config,
                    key = f"{key}.{key_gate}".replace("{expert_idx}", str(idx)),
                    fkey = fkey_gate,
                    fidx = idx,
                    frange = (0, intermediate_size) if key_gate_up_split else None,
                    finterleaved = gate_up_interleaved,
                    in_features = self.expert_size,
                    out_features = intermediate_size,
                    qmap = expert_qmap,
                    out_dtype = self.interm_dtype,
                    transposed_load = transposed_load,
                    transpose_fused_weights = transpose_fused_weights,
                    ftranspose_after_load = ftranspose_after_load,
                    frange_dim = frange_dim,
                    qgroup = key + ".block_gud",
                    qbits_key = qbits_key,
                )
                up = Linear(
                    config = config,
                    key = f"{key}.{key_up}".replace("{expert_idx}", str(idx)),
                    fkey = fkey_up,
                    fidx = idx,
                    frange = (intermediate_size, intermediate_size * 2) if key_gate_up_split else None,
                    finterleaved = gate_up_interleaved,
                    in_features = self.expert_size,
                    out_features = intermediate_size,
                    qmap = expert_qmap,
                    out_dtype = self.interm_dtype,
                    transposed_load = transposed_load,
                    transpose_fused_weights = transpose_fused_weights,
                    ftranspose_after_load = ftranspose_after_load,
                    frange_dim = frange_dim,
                    qgroup = key + ".block_gud",
                    qbits_key = qbits_key,
                    weight_scale = 1.0 / interm_div,
                )
                down = Linear(
                    config = config,
                    key = f"{key}.{key_down}".replace("{expert_idx}", str(idx)),
                    fkey = fkey_down,
                    fidx = idx,
                    in_features = intermediate_size,
                    out_features = self.expert_size,
                    qmap = qmap + f".{idx}.down" if qmap else None,
                    out_dtype = torch.float,
                    allow_input_padding = True,
                    transposed_load = transposed_load,
                    transpose_fused_weights = transpose_fused_weights,
                    ftranspose_after_load = ftranspose_after_load,
                    # The input dim pads to match the gate/up padded output width; padded output
                    # columns (zeros, or quantization noise over zero weights) are trimmed
                    trim_padded_out = True,
                    qgroup = key + ".block_gud",
                    qbits_key = qbits_key,
                )

                if _RWH and qmap:
                    for lin, nm in ((gate, "input"), (up, "input"), (down, "down")):
                        if lin is not None:
                            lin.cap_rkey = f"{qmap}.r{idx}.{nm}"
                            lin.qmap_q = f"{qmap}.q{idx}.{nm}"

                self.ups.append(up)
                if gate is not None:
                    self.gates.append(gate)
                self.downs.append(down)

                self.register_submodule(up)
                self.register_submodule(gate)
                self.register_submodule(down)

        if self.gated:
            self.gateless_act = None
            match activation_fn:
                case "silu":
                    self.activation_fn_call = ext.silu_mul
                    self.activation_fn_idx = 0
                case "gelu":
                    self.activation_fn_call = ext.gelu_mul
                    self.activation_fn_idx = 1
                case "swiglu_oai":
                    self.activation_fn_call = ext.silu_oai_mul
                    self.activation_fn_idx = 3
                case "relu2":
                    self.activation_fn_call = ext.relu2_mul
                    self.activation_fn_idx = 2
                case _:
                    raise ValueError(f"Unknown activation function {activation_fn}")
        else:
            # Gateless relu2 rides the gated fast paths via relu(u) * u = relu2(u): the act call
            # sites pass (u, u, a) and the fused MoE kernel's MOE_ACT_RELU2_NOGATE synthesizes
            # the gate lane from u
            self.activation_fn_call = ext.relu_mul if activation_fn == "relu2" else None
            self.activation_fn_idx = 2 if activation_fn == "relu2" else -1
            match activation_fn:
                case "silu": self.gateless_act = F.silu
                case "gelu": self.gateless_act = lambda x: F.gelu(x, approximate = "tanh")
                case "relu2": self.gateless_act = lambda x: torch.square(F.relu(x))
                case _:
                    raise ValueError(f"Unknown gateless activation function {activation_fn}")

        self.is_quantized = False
        self.support_fused = False
        self.fused_3inst = False
        self.support_quant_paths = False
        self.multi_gate = None
        self.multi_up = None
        self.multi_down = None
        # Per-expert K inside the layer (hot-3 / cold-2): one MultiLinear triple per K, see moe_mixedk.py
        self.mixed_k = None
        self.mixed_outputs = None
        self.expert_bits = None     # optional pack config row (quantization_config["expert_bits"]), validated at load
        self.pruned = set(i for i, u in enumerate(self.ups) if getattr(u, "pruned", False))   # K0 experts
        if self.ups and config is not None and gates is None and self.num_local_experts == num_experts:
            _q = getattr(config, "config_dict", {}).get("quantization_config")
            _b = parse_expert_bits(_q, key, len(self.ups)) if _q else None
            self.expert_bits = _b

        self.routing_cfg = None
        self.experts_cfg = None
        self.coop_rows = MAX_BSZN   # raised to the fused kernels' row capacity when the buffers are built

        # Persistent broadcast targets for TP ranks without the router (see forward)
        self.bcast_sel_bsz1 = None
        self.bcast_weights_bsz1 = None

        self.e_score_correction_bias = None
        self.e_score_correction_bias_key = key_e_score_bias
        self.e_score_bias_vl = None
        self.e_score_bias_vl_key = key_e_score_bias_vl
        self.tid2eid = None
        self.tid2eid_key = key_tid2eid
        self.per_expert_scale = None
        self.per_expert_scale_key = key_per_expert_scale

        self.shared_experts = shared_experts
        if shared_experts is not None:
            self.register_submodule(shared_experts)

        match router_type:
            case "std": self.routing_fn = routing_std
            case "std_bias": self.routing_fn = routing_std_bias
            case "ds3": self.routing_fn = routing_ds3
            case "dots": self.routing_fn = routing_dots
            case "sqrtsp": self.routing_fn = routing_sqrtsp
            case "sqrtsp_hash": self.routing_fn = routing_sqrtsp_hash
            case _: raise ValueError(f"Unknown router type {router_type}")

        self.tp_reduce = False
        self.tp_mode = None

        self.shared_experts_post_norm = shared_experts_post_norm
        self.router_pre_norm = router_pre_norm
        self.routed_pre_norm = routed_pre_norm
        self.routed_post_norm = routed_post_norm
        self.register_submodule(self.shared_experts_post_norm)
        self.register_submodule(self.router_pre_norm)
        self.register_submodule(self.routed_pre_norm)
        self.register_submodule(self.routed_post_norm)

        self.bc = None
        self.bc_sh_exp = False
        self.fused_mode_buffers = None
        self.mtile_ok = False
        self.fused_rows = TEMP_ROWS_FUSED
        self.batch_recon = None
        self.support_hip_grouped = False
        self.support_dec_moe = False
        self.dec_moe_mcg = None      # uniform codebook flag for exl3_dec_moe (None = no fused path)
        self.dec_shf = None          # shared-expert fold args for exl3_dec_moe_shared (decB8), None = ineligible
        self.hip_grouped_buffers = None
        self.support_hip_prefill = False
        self.hip_prefill_buffers = None
        self.support_hip_wmma = False
        self.wmma_mcg = None         # uniform codebook flag for exl3_moe_prefill_wmma (None = off)
        self._pf_assignments = 0
        self.hip_grouped_lora_blocked = False
        self.tier = None             # expert_store.LayerTier when EXL3_EXPERT_BUDGET_GIB tiers this layer's experts
        self._cpu_init_state()

    @override
    def optimizer_targets(self):
        g, u, d = [], [], []
        for m in self.gates: g += m.optimizer_targets()
        for m in self.ups: u += m.optimizer_targets()
        for m in self.downs: d += m.optimizer_targets()
        if self.latent_in is not None:
            u = self.latent_in.optimizer_targets() + u
            d = d + self.latent_out.optimizer_targets()
        if self.shared_experts:
            s = self.shared_experts.optimizer_targets()
            return [s, [g + u, d]]
        else:
            return [[g + u, d]]


    def _hip_k3_prefill_selected(self, rows: int) -> bool:
        """Dispatch switch between the legacy K=3 prefill kernel (exl3_moe_gfx12_k3_prefill, 17..2048 rows) and the
        grouped WMMA GEMM (mpw2) for layers whose experts are all K=3. Read per call (one process can flip it).
        EXL3_HIP_GROUPED_MOE_PREFILL: 1 = always legacy (the earlier default), 0 = never legacy,
        unset / auto = mpw2 wherever it is eligible, legacy only below the WMMA row threshold (verify rows with
        EXL3_HIP_PREFILL_MIN_ROWS=2, rows 17..31) or when MPW_KERN=1 pins the old grouped GEMM.
        Both kernels are equally accurate against a float64 reference; mpw2 is 2.6x faster per layer."""
        v = os.environ.get("EXL3_HIP_GROUPED_MOE_PREFILL", "auto")
        if v == "0": return False
        if v == "1": return True
        return not (
            self.support_hip_wmma and rows >= _HIP_WMMA_MIN_ROWS and
            os.environ.get("EXL3_MOE_WMMA", "1") != "0" and os.environ.get("MPW_KERN", "2") != "1"
        )

    def _ensure_hip_prefill_buffers(self, num_tokens):
        """Allocate the gfx12 prefill workspace sized to the actual chunk so chunk512
        serving never reserves the full 2048-row envelope. Grows only if a larger
        chunk routes later in the same process."""
        top_k = self.num_experts_per_tok
        assignments = num_tokens * top_k
        if self.hip_prefill_buffers is not None and self._pf_assignments >= assignments:
            return self.hip_prefill_buffers
        device = self.device
        H = self.hidden_size
        I = self.intermediate_size_padded
        rows = assignments // top_k
        E_local = self.num_local_experts or self.num_experts
        self.hip_prefill_buffers = HIPPrefillBuffers(
            gu_had = g_tensor_cache.get(
                device, (2 * assignments, H), torch.half, f"moe_gfx12_pf_gu_had_a{assignments}"),
            gu_out = g_tensor_cache.get(
                device, (2 * assignments, I), torch.half, f"moe_gfx12_pf_gu_out_a{assignments}"),
            down_out = g_tensor_cache.get(
                device, (assignments, H), torch.float, f"moe_gfx12_pf_down_out_a{assignments}"),
            output = g_tensor_cache.get(
                device, (rows, H), torch.float, f"moe_gfx12_pf_output_a{rows}"),
            # Sized to the resident expert count: an expert-range shard (CPU split) holds
            # E_local < num_experts and the kernel asserts int64[E_local + 1]. Keyed on E so a
            # split and an unsplit layer on the same device do not share the cache slot
            expert_offsets = g_tensor_cache.get(
                device, (E_local + 1,), torch.long, f"moe_gfx12_pf_offsets_e{E_local}"),
            inverse_order = g_tensor_cache.get(
                device, (assignments,), torch.long, f"moe_gfx12_pf_inverse_a{assignments}"),
            expert_chunks = g_tensor_cache.get(
                device,
                (E_local * (_HIP_PREFILL_MAX_EXPERT_ROWS // 16),),
                torch.int,
                f"moe_gfx12_pf_chunks_e{E_local}",
            ),
            chunk_count = g_tensor_cache.get(
                device, (1,), torch.int, "moe_gfx12_pf_chunk_count"),
        )
        self._pf_assignments = assignments
        return self.hip_prefill_buffers


    def invalidate_hip_grouped_for_lora(self, target: Linear):
        if target in self.gates or target in self.ups or target in self.downs:
            self.hip_grouped_lora_blocked = True
            self.support_hip_grouped = False
            self.support_hip_prefill = False
            self.support_dec_moe = False
            self.support_hip_wmma = False


    def load_local(self, **kwargs):

        # Test if experts can be fused
        num_exl3_tensors = 0
        num_nonexl3_tensors = 0
        for l in self.gates + self.ups + self.downs:
            if l.quant_type == "exl3":
                num_exl3_tensors += 1
            else:
                num_nonexl3_tensors += 1
        if num_exl3_tensors and num_nonexl3_tensors:
            print(f" !! Warning, partially quantized block-sparse MLP layer: {self.key}")
        self.is_quantized = (num_exl3_tensors > 0 and num_nonexl3_tensors == 0)

        # Per-expert K: a layer whose experts differ in trellis bitrate cannot use the single-K tables
        # (MultiLinear, BC_BlockSparseMLP, exl3_dec_moe, exl3_moe). It runs the per-K grouped HIP kernels
        # (moe_mixedk.MixedKStore) and the dense per-expert path (K-agnostic, each Linear carries its own K).
        # Decided here, before the single-K gates below.
        self.mixed_k = None
        mixed_ks = None
        if self.is_quantized and self.gated and len(self.ups) > 1:
            try:
                mixed_ks = expert_ks(self.gates, self.ups, self.downs)
            except MixedKUnsupported:
                mixed_ks = None
            if mixed_ks is not None and self.expert_bits is not None:
                check_expert_bits(self.expert_bits, mixed_ks)
        is_mixed = mixed_ks is not None and layer_is_mixed(mixed_ks)

        # The quantized fast paths (mgemm/BC/fused kernels) don't yet support per-expert biases,
        # activations other than silu/gelu (or gateless relu2), or trimmed (padded) down
        # projections; configurations with any of those run every batch size through the dense
        # per-expert path, which handles all of them (gpt-oss)
        has_mgemm = hasattr(ext, "exl3_mgemm")
        self.support_quant_paths = (
            has_mgemm and self.is_quantized and not is_mixed and
            (self.activation_fn in ("silu", "gelu") if self.gated else self.activation_fn == "relu2") and
            all(l.inner.bias is None for l in self.gates + self.ups + self.downs) and
            all(not l.trim_padded_out or l.out_features == l.out_features_unpadded for l in self.downs)
        )

        # The fused bsz<=MAX_BSZN kernels (BC_BlockSparseMLP.run_bszN) additionally support
        # the gpt-oss activation, per-expert biases (all-or-nothing per projection) and padded
        # dims (input zero-padded and output trimmed in the kernel); the dense per-expert path
        # covers those configurations for every other batch shape
        def _uniform_bias(ls):
            has = [l.inner.bias is not None for l in ls]
            return all(has) or not any(has)
        self.support_bc_bszn = (
            has_mgemm and self.is_quantized and not is_mixed and
            (self.activation_fn in ("silu", "gelu", "swiglu_oai") if self.gated else self.activation_fn == "relu2") and
            _uniform_bias(self.gates) and _uniform_bias(self.ups) and _uniform_bias(self.downs) and
            not self.config.infer_params.no_reconstruct
        )

        # Dedicated ROCm decode route. Keep this exact until additional shapes and split
        # semantics have their own oracle coverage; in particular, expert-parallel and
        # intermediate-split modules must continue through the established fallback.
        full_expert_layer = (
            len(self.ups) == self.num_experts and
            (self.num_local_experts is None or self.num_local_experts == self.num_experts) and
            (self.routing_first is None or (
                self.routing_first == 0 and self.routing_last == self.num_experts
            )) and
            self.cpu_split_first is None
        )
        # CPU expert split (--moe_cpu_split): the module keeps the HEAD [0, first) experts and
        # the pointer tables are built from that slice, so the grouped HIP kernels see a
        # contiguous local range starting at 0 and mask picks >= experts to exact zeros (their
        # rows go to the CPU partial that cpu_split_combine folds back in). Same partial-sum
        # contract the CUDA fused path documents for expert-range shards. Without this the
        # split layers fell to the reconstruct fallback (5x slower GPU side than the grouped
        # kernel), which made offloading a net loss. EXL3_HIP_GROUPED_SPLIT=0 restores the gate.
        cpu_head_split_layer = (
            self.cpu_split_first is not None and
            self.routing_first == 0 and self.routing_last == self.cpu_split_first and
            len(self.ups) == self.cpu_split_first and
            self.num_local_experts == self.cpu_split_first and
            os.environ.get("EXL3_HIP_GROUPED_SPLIT", "1") != "0"
        )
        full_expert_layer = full_expert_layer or cpu_head_split_layer
        hip_grouped_device = False
        if (
            bool(torch.version.hip) and hasattr(ext, "exl3_moe_gfx12_k3") and
            hasattr(ext, "exl3_gemv_supported")
        ):
            device_index = torch.device(self.device).index
            if device_index is None:
                device_index = torch.cuda.current_device()
            # The grouped-MoE kernels launch exl3_gemv_kernel_body (ported to both
            # WMMA families) plus elementwise helpers with no arch intrinsics, so
            # any WMMA GEMV device qualifies. The shape checks below do the real
            # filtering. NOTE: other gfx12-only kernels (routing, hyperconnection
            # fusion) still need an explicit family-1 / arch-string test.
            hip_grouped_device = ext.exl3_gemv_supported(device_index)
        self.support_hip_grouped = (
            hip_grouped_device and self.is_quantized and self.gated and self.activation_fn == "silu" and
            # Shapes the grouped kernel handles: hidden and intermediate are 128-aligned (its grids
            # are width/128 and width/32), top-k 6 (DeepSeek-V4.1), 8 (MiMo-V2) or 10 (Qwen3.8).
            # The old 2560 / 640-768 / top-10 gate was the Qwen3.8-Flash-Next pack only.
            self.hidden_size % 128 == 0 and self.intermediate_size_padded % 128 == 0 and
            self.num_experts_per_tok in (6, 8, 10) and self.interm_dtype == torch.half and
            # A swiglu limit (DeepSeek-V4.1: 10.0) is applied in moe_silu_mul_kernel. Only taken where the
            # fused exl3_dec_moe epilogue cannot serve the shape (I % 512 != 0, e.g. DS4.1's 2304), so models
            # that already run the fused path (GLM-5.3-Flash) keep their routing; EXL3_MOE_GROUPED_LIMIT=0 off.
            (self.act_limit == 0 or (
                0.0 < self.act_limit <= 65504.0 and os.environ.get("EXL3_MOE_GROUPED_LIMIT", "1") != "0" and
                (self.hidden_size % 512 != 0 or self.intermediate_size_padded % 512 != 0))) and
            self.tp_mode is None and not self.hip_grouped_lora_blocked and
            full_expert_layer and
            all(l.inner.bias is None for l in self.gates + self.ups + self.downs) and
            all(not l.trim_padded_out or l.out_features == l.out_features_unpadded for l in self.downs) and
            not self.config.infer_params.no_reconstruct
        )

        # Batch-1 fused decode (exl3_dec_moe, gfx11.5): gate/up + activation + down + weighted sum
        # over the routed experts in two launches, any hidden/intermediate width that is a multiple
        # of 512 and any mul1 K the decode kernels support (checked once the tables exist)
        # act_limit (GLM-5.3-Flash swiglu_limit = 10.0) is applied inside the gu kernel's epilogue;
        # the bound keeps the fp16 clamp limit finite (0 = unclamped, the MiMo/Qwen default).
        self.support_dec_moe = (
            bool(torch.version.hip) and hasattr(ext, "exl3_dec_moe") and
            os.environ.get("EXL3_DEC", "1") != "0" and os.environ.get("EXL3_DEC_MOE", "1") != "0" and
            self.is_quantized and self.gated and self.activation_fn == "silu" and not is_mixed and
            self.interm_dtype == torch.half and 0.0 <= self.act_limit <= 65504.0 and
            self.tp_mode is None and
            full_expert_layer and not self.hip_grouped_lora_blocked and
            self.hidden_size % 512 == 0 and self.intermediate_size_padded % 512 == 0 and
            all(l.inner.bias is None for l in self.gates + self.ups + self.downs) and
            all(not l.trim_padded_out or l.out_features == l.out_features_unpadded for l in self.downs) and
            all(getattr(l.inner, "dec_ok_moe", False) for l in self.gates + self.ups + self.downs) and
            not self.config.infer_params.no_reconstruct
        )

        # Grouped WMMA prefill (gfx11.5, exl3_moe_prefill.cu): any expert shape with 128-aligned
        # widths, any top-k, K 2 / 2.5 / 3 / 4 / 5 / 6, either codebook, and a clamped SwiGLU limit
        # (checked once the pointer tables exist below). Full expert layers only;
        # EXL3_MOE_WMMA=0 disables it per forward.
        hip_wmma_device = False
        if bool(torch.version.hip) and hasattr(ext, "exl3_moe_prefill_wmma") and hasattr(ext, "exl3_gemv_wmma_family"):
            device_index = torch.device(self.device).index
            if device_index is None:
                device_index = torch.cuda.current_device()
            hip_wmma_device = ext.exl3_gemv_wmma_family(device_index) == 2
        self.support_hip_wmma = (
            hip_wmma_device and self.is_quantized and self.gated and self.activation_fn == "silu" and
            # act_limit (GLM-5.3-Flash swiglu_limit = 10.0) is applied in mpw_act_kernel; the
            # bound is the fp16 clamp limit (0 = unclamped, the MiMo/Qwen default)
            0.0 <= self.act_limit <= 65504.0 and self.tp_mode is None and self.latent_in is None and
            self.hidden_size % 128 == 0 and self.intermediate_size_padded % 128 == 0 and
            self.ups[0].in_features == self.hidden_size and self.downs[0].out_features == self.hidden_size and
            len(self.ups) == self.num_experts and
            (self.num_local_experts is None or self.num_local_experts == self.num_experts) and
            (self.routing_first is None or (self.routing_first == 0 and self.routing_last == self.num_experts)) and
            self.cpu_split_first is None and not self.cpu_offload and
            all(l.inner.bias is None for l in self.gates + self.ups + self.downs) and
            all(not l.trim_padded_out or l.out_features == l.out_features_unpadded for l in self.downs) and
            not self.config.infer_params.no_reconstruct
        )

        # Make fused modules (only used by the quantized fast paths). Gateless experts have no
        # gate MultiLinear; the up module doubles as a placeholder wherever the fast paths want
        # gate pointer tables (never dereferenced, the gate GEMMs are skipped)
        if is_mixed and (self.support_hip_grouped or self.support_hip_wmma) and \
                not self.config.infer_params.no_reconstruct:
            # One MultiLinear triple per K group; the single-K tables, fused kernels and BC graph stay off
            # (support_quant_paths / support_bc_bszn / support_dec_moe were cleared above)
            try:
                self.mixed_k = MixedKStore(self.device, self.gates, self.ups, self.downs)
            except MixedKUnsupported:
                self.mixed_k = None
            self.support_hip_grouped = self.support_hip_grouped and self.mixed_k is not None and self.mixed_k.decode_ok
            self.support_hip_wmma = self.support_hip_wmma and self.mixed_k is not None and self.mixed_k.wmma_ok
            if self.mixed_k is not None and not (self.support_hip_grouped or self.support_hip_wmma):
                self.mixed_k.unload()
                self.mixed_k = None
            self.support_hip_prefill = False
            if self.tier is not None and self.mixed_k is not None:
                # tiered per-expert-K layer: each K group's tables become views of the store's group buffer
                self.tier.bind_mixed(self.mixed_k)
        elif (self.support_quant_paths or self.support_bc_bszn or self.support_hip_grouped or
            self.support_dec_moe or self.support_hip_wmma) and not self.config.infer_params.no_reconstruct:
            self.multi_gate = MultiLinear(self.device, self.gates, allow_bias = True) if self.gated else None
            self.multi_up = MultiLinear(self.device, self.ups, allow_bias = True)
            self.multi_down = MultiLinear(self.device, self.downs, allow_bias = True)
            if self.tier is not None:
                # the kernels address experts through these tables: they become views of the store's
                # per-layer table buffer, patched in place as experts enter and leave the slots
                lt = self.tier
                lt.build_tables(self.device)
                for i, m in enumerate((self.multi_gate, self.multi_up, self.multi_down)):
                    m.ptrs_trellis, m.ptrs_suh, m.ptrs_svh = lt.view(i * 3), lt.view(i * 3 + 1), lt.view(i * 3 + 2)

            # Enable fully fused kernel if possible (uniform mcg or mul1 codebook across gate/up/down,
            # and an activation the fused kernel implements)
            cbs = (
                self.multi_gate.q_cb() if self.gated else self.multi_up.q_cb(),
                self.multi_up.q_cb(),
                self.multi_down.q_cb(),
            )
            self.support_fused = (
                hasattr(ext, "exl3_moe") and hasattr(ext, "exl3_moe_max_concurrency") and
                cbs[0] == cbs[1] == cbs[2] and
                cbs[0] in (((True, False), (False, True), (False, False)) if _FUSED_3INST else ((True, False), (False, True))) and
                self.support_quant_paths and
                (self.fused_fp16_interm or self.interm_dtype in (None, torch.half))
            )
            self.fused_3inst = self.support_fused and cbs[0] == (False, False)
            # One K for the whole layer: the grouped GEMV is templated on the experts' trellis
            # bitrate and gate/up/down share it (the allocator assigns per layer).
            ks = {multi.K for multi in (self.multi_gate, self.multi_up, self.multi_down)}
            self.support_hip_grouped = self.support_hip_grouped and len(ks) == 1 and all(
                multi.K in (1, 2, 2.5, 3, 4, 5, 6) and multi.mul1 and not multi.mcg
                for multi in (self.multi_gate, self.multi_up, self.multi_down)
            )
            # The prefill kernel (exl3_moe_gfx12_k3_prefill) is still Qwen3.8-only: K=3, top-10 and
            # a 2560/640-768 hidden/intermediate are hardcoded in it, so it must not follow the
            # generalized decode gate above.
            self.support_hip_prefill = (
                self.support_hip_grouped and hasattr(ext, "exl3_moe_gfx12_k3_prefill") and
                self.hidden_size == 2560 and self.intermediate_size_padded in (640, 768) and
                self.num_experts_per_tok == 10 and ks == {3}
            )
            self.support_dec_moe = self.support_dec_moe and (
                self.multi_gate.K == self.multi_up.K and
                self.ups[0].in_features == self.hidden_size and self.downs[0].out_features == self.hidden_size
            )
            # Codebook: exl3_dec_moe takes one flag for all three projections (the kernel is
            # templated on the codebook, not per projection), so the layer needs a uniform one.
            # mul1 and mcg both decode; anything else leaves the fused path off.
            if cbs[0] == cbs[1] == cbs[2] and cbs[0] in ((True, False), (False, True)):
                self.dec_moe_mcg = bool(cbs[0][0])
            self.support_dec_moe = self.support_dec_moe and self.dec_moe_mcg is not None
            if self.support_dec_moe:
                self.dec_moe_out = g_tensor_cache.get(self.device, (1, self.hidden_size), torch.float, "dec_moe_out")
                self.dec_moe_act = g_tensor_cache.get(
                    self.device, (self.num_experts_per_tok, self.intermediate_size_padded), torch.half, "dec_moe_act")
                self.dec_shf_setup()
                # R-row union MoE (exl3_dec_moe_union): shared, per-device
                # workspace (like dec_workspace) sized for the worst case (MAX_BSZN rows, this
                # layer's topk/H/I) and reused across layers -- forward calls are stream-ordered
                # so a later layer never races an earlier one's still-in-flight kernel.
                numex = self.num_experts_per_tok
                Ipad = self.intermediate_size_padded
                Ru = MAX_BSZN
                self.dec_moe_union_out = g_tensor_cache.get(
                    self.device, (UNION_ROWS, self.hidden_size), torch.float, "dec_moe_union_out")
                self.dec_moe_union_act = g_tensor_cache.get(
                    self.device, (Ru * numex, Ipad), torch.half, "dec_moe_union_act")
                self.dec_moe_union_down_part = g_tensor_cache.get(
                    self.device, (Ru * numex, self.hidden_size), torch.float, "dec_moe_union_down_part")
                # scratch/counters: CTR_RA in exl3_dec_moe (exl3_dec.cu) starts at offset 0, the
                # same as batch-1's CTR_A -- deliberate, since the two never run in the same
                # decode step (a step is all-bsz1 or all-R, single CUDA stream, so whichever ran
                # last has already reset its range to 0 via arrive()) -- so the union kernel
                # reuses the same global dec_workspace() the batch-1 path already keeps zeroed,
                # no separate allocation. DEC_SCRATCH_FLOATS (2<<20) is exactly Ru*numex*max(2*8*
                # Ipad, 8*H) for this model's shapes; a TORCH_CHECK in the kernel catches it if a
                # future shape needs more.
            self.support_hip_wmma = self.support_hip_wmma and all(
                multi.K in (1, 2, 2.5, 3, 4, 5, 6) and (multi.mul1 != multi.mcg)
                for multi in (self.multi_gate, self.multi_up, self.multi_down)
            ) and self.multi_gate.K == self.multi_up.K
            # Codebook: mpw_gemm is templated on it and takes one flag for all three projections
            #, so the layer needs a uniform codebook. Unknown (neither mul1 nor mcg,
            # or mixed) leaves the path off.
            wmma_cbs = {(m.mul1, m.mcg) for m in (self.multi_gate, self.multi_up, self.multi_down)}
            if len(wmma_cbs) == 1:
                wmma_mul1, wmma_mcg = wmma_cbs.pop()
                if wmma_mul1 != wmma_mcg:
                    self.wmma_mcg = wmma_mcg
            self.support_hip_wmma = self.support_hip_wmma and self.wmma_mcg is not None

        if self.tier is not None and self.multi_up is None and self.mixed_k is None:
            raise RuntimeError(f"{self.key}: tiered experts need the quantized MoE fast paths (mgemm tables)")

        # Temp buffers for graph, dq and fused-bsz1 paths
        numex = self.num_experts_per_tok
        H = self.expert_size
        # The gate/up input width and the down output width are the (possibly 128-padded)
        # quantized dims; both equal H for aligned models
        Hi = self.ups[0].in_features
        Ho = self.downs[0].out_features
        I = self.intermediate_size_padded
        device = self.device

        # bszn_rows bounds the fused bsz 1..MAX_BSZN path (BC_BlockSparseMLP.run_bszN, one
        # slot per (token, expert) pick); buffers grow to whichever of that or the single-expert
        # graph loop's TEMP_ROWS_GRAPH requirement is larger, sharing the one cache entry per name
        # (g_tensor_cache is exact-shape-keyed and never evicts -- growing a differently-shaped
        # second entry under the same name would silently double memory forever)
        self.coop_rows = max(MAX_BSZN, min(COOP_ROWS, 256 // numex))
        bszn_rows = self.coop_rows * numex

        temp_hidden = g_tensor_cache.get(device, (max(TEMP_ROWS_GRAPH * 2, bszn_rows), Hi), torch.half, "moe1_temp_hidden")
        temp_interm = g_tensor_cache.get(device, (max(TEMP_ROWS_GRAPH * 2, 2 * bszn_rows), I), self.interm_dtype, "moe1_temp_interm")
        temp_activa = g_tensor_cache.get(device, (max(TEMP_ROWS_GRAPH, bszn_rows), I), torch.half, "moe1_temp_activa")
        temp_output = g_tensor_cache.get(device, (max(TEMP_ROWS_GRAPH, bszn_rows), Ho), torch.float, "moe1_temp_output")

        yh = temp_hidden[:bszn_rows].view(bszn_rows, 1, Hi)
        interm_g = temp_interm[:bszn_rows].view(bszn_rows, 1, I)
        interm_u = temp_interm[bszn_rows:bszn_rows*2].view(bszn_rows, 1, I)
        interm_a = temp_activa[:bszn_rows].view(bszn_rows, 1, I)
        yh2 = temp_hidden
        interm_gu = temp_interm
        interm_a2 = temp_activa
        out_d = temp_output[:bszn_rows].view(bszn_rows, 1, Ho)
        out_d2 = temp_output

        # Completion counters + expert-run table of the fused decode kernels
        # (BC_BlockSparseMLP.run_bszN): the kernels reset each other's counters for the next
        # call, so they only need zeroing once
        coop_ctr = g_tensor_cache.get(device, (bszn_rows * (I // 128) + self.coop_rows * (Ho // 128) + 2 * bszn_rows + 3,), torch.int, "moe1_coop_ctr")
        coop_ctr.zero_()
        # Rotated up-projection input per slot for the fused kernels at bsz > 1 (yh holds the gate's)
        had_u = g_tensor_cache.get(device, (bszn_rows, Hi), torch.half, "moe1_temp_hidden_u")

        # Expert interval for split module (-1, -1) indicate no split
        mine, maxe = self.routing_first, self.routing_last
        if mine is None or maxe - mine == self.num_experts:
            mine, maxe = -1, -1

        # Routed sum of the fused decode path, one exact-width row per token (the kernel trims the
        # padded down width on the way out)
        out_bszn = g_tensor_cache.get(device, (self.coop_rows, H), torch.float, "moe1_out_bszn")

        cfg = ExpertsCFG(
            yh = yh,
            interm_g = interm_g,
            interm_u = interm_u,
            interm_a = interm_a,
            out_d = out_d,
            out_d2 = out_d2,
            min_expert = mine,
            max_expert = maxe,
            out_bszn = out_bszn,
        )
        self.experts_cfg = cfg

        if self.support_hip_grouped:
            max_assignments = _HIP_GROUPED_MAX_ROWS * numex
            # Projection-major flat storage keeps every prefix slice contiguous up to the
            # runtime grouped-row cap. All layers on a device share these max-sized cache
            # entries; calls only take views.
            self.hip_grouped_buffers = HIPGroupedBuffers(
                gu_had = g_tensor_cache.get(
                    device, (2 * max_assignments, H), torch.half, "moe_gfx12_gu_had"),
                gu_out = g_tensor_cache.get(
                    device, (2 * max_assignments, I), torch.half, "moe_gfx12_gu_out"),
                down_had = g_tensor_cache.get(
                    device, (max_assignments, I), torch.half, "moe_gfx12_down_had"),
                down_out = g_tensor_cache.get(
                    device, (max_assignments, H), torch.float, "moe_gfx12_down_out"),
                output = g_tensor_cache.get(
                    device, (_HIP_GROUPED_MAX_ROWS, H), torch.float, "moe_gfx12_output"),
            )
            if self.mixed_k is not None:
                # One fp32 output per K group (group 0 uses the usual buffer); the partial sums are added
                self.mixed_outputs = [self.hip_grouped_buffers.output] + [
                    g_tensor_cache.get(device, (_HIP_GROUPED_MAX_ROWS, H), torch.float, f"moe_gfx12_output_k{i}")
                    for i in range(1, self.mixed_k.num_groups)
                ]

        if self.support_hip_prefill:
            # Workspace is allocated lazily at dispatch sized to the actual chunk
            # (see _ensure_hip_prefill_buffers), not reserved at the 2048-row cap, so
            # chunk512 serving does not overallocate VRAM.
            self.hip_prefill_buffers = None
            self._pf_assignments = 0

        if (self.support_quant_paths or self.support_bc_bszn) \
                and not self.config.infer_params.no_reconstruct:

            # Embed bound classes for shared experts and shared gate
            sh_exp_bc = None
            sh_exp_t = None
            sh_gate_bc = None
            self.bc_sh_exp = False
            if (
                self.shared_experts
                and isinstance(self.shared_experts, GatedMLP)
                and self.shared_experts.bc is not None
                and self.shared_experts_post_norm is None   # TODO: embed post_norm in BC
                and not self.alt_residual_channel  # TODO: allow residual channel switching in BC (Gemma4)
                and self.latent_in is None  # the graph would merge the shared output at the latent width
            ):
                self.bc_sh_exp = True
                sh_exp_bc = self.shared_experts.bc
                sh_exp_t = torch.empty((1, self.coop_rows, self.hidden_size), dtype = torch.float, device = self.device)
                if self.shared_gate:
                    assert self.shared_gate.quant_type == "fp16"
                    sh_gate_bc = self.shared_gate.inner.bc

            # Pointer lists for fused modes. Gateless experts reuse the up tables as gate
            # placeholders: valid memory for the (uniform) table loads, never dereferenced
            u_trellis_ptr = torch.tensor([l.inner.trellis.data_ptr() for l in self.ups])
            u_suh_ptr = torch.tensor([l.inner.suh.data_ptr() for l in self.ups])
            u_svh_ptr = torch.tensor([l.inner.svh.data_ptr() for l in self.ups])
            if self.gated:
                g_trellis_ptr = torch.tensor([l.inner.trellis.data_ptr() for l in self.gates])
                g_suh_ptr = torch.tensor([l.inner.suh.data_ptr() for l in self.gates])
                g_svh_ptr = torch.tensor([l.inner.svh.data_ptr() for l in self.gates])
            else:
                g_trellis_ptr, g_suh_ptr, g_svh_ptr = u_trellis_ptr, u_suh_ptr, u_svh_ptr
            gu_trellis_ptr = torch.stack((g_trellis_ptr, u_trellis_ptr), dim = 0).T.contiguous().to(self.device)
            gu_suh_ptr = torch.stack((g_suh_ptr, u_suh_ptr), dim = 0).T.contiguous().to(self.device)
            gu_svh_ptr = torch.stack((g_svh_ptr, u_svh_ptr), dim = 0).T.contiguous().to(self.device)
            if self.tier is not None:
                gu_trellis_ptr, gu_suh_ptr, gu_svh_ptr = (self.tier.gu_view(i) for i in range(3))

            dq_temp_up = g_tensor_cache.get(device, (Hi, I), torch.half, "dq_temp")
            dq_temp_down = dq_temp_up.view(I, Ho)

            # Per-expert bias pointer tables for the fused decode path
            def _bias_ptrs(ls):
                if ls[0].inner.bias is None:
                    return None
                return torch.tensor([l.inner.bias.data_ptr() for l in ls],
                                    dtype = torch.long, device = device)
            gate_bias_ptrs = _bias_ptrs(self.gates) if self.gated else None
            up_bias_ptrs = _bias_ptrs(self.ups)
            down_bias_ptrs = _bias_ptrs(self.downs)

            # Bound class for the fused-decode, graph and dq paths (gateless: the up module stands in
            # for the unused gate pointer args, and the gates list is empty)
            multi_gate = self.multi_gate if self.gated else self.multi_up
            self.bc = ext.BC_BlockSparseMLP(
                yh2,
                cfg.yh,
                interm_gu,
                cfg.interm_g,
                cfg.interm_u,
                cfg.interm_a,
                interm_a2,
                cfg.out_d,
                cfg.out_d2,
                sh_exp_t,
                coop_ctr,
                had_u,
                dq_temp_up,
                dq_temp_down,
                cfg.min_expert,
                cfg.max_expert,
                multi_gate.ptrs_trellis,
                multi_gate.ptrs_suh,
                multi_gate.ptrs_svh,
                multi_gate.K,
                multi_gate.mcg,
                multi_gate.mul1,
                self.multi_up.ptrs_trellis,
                self.multi_up.ptrs_suh,
                self.multi_up.ptrs_svh,
                self.multi_up.K,
                self.multi_up.mcg,
                self.multi_up.mul1,
                self.multi_down.ptrs_trellis,
                self.multi_down.ptrs_suh,
                self.multi_down.ptrs_svh,
                self.multi_down.K,
                self.multi_down.mcg,
                self.multi_down.mul1,
                self.activation_fn == "silu",
                self.activation_fn == "gelu",
                self.activation_fn == "swiglu_oai",
                sh_exp_bc,
                sh_gate_bc,
                self.act_limit,
                [x.inner.bc for x in self.gates],
                [x.inner.bc for x in self.ups],
                [x.inner.bc for x in self.downs],
                gu_trellis_ptr,
                gu_suh_ptr,
                gu_svh_ptr,
                cfg.out_bszn,
                gate_bias_ptrs,
                up_bias_ptrs,
                down_bias_ptrs,
                act_relu2 = self.activation_fn == "relu2",
            )

            # Larger buffers for fused path, if supported. Wide row tiles (32 / 64 rows, separate
            # N = 128 kernel instances) exist for the mul1 codebook; with them the fused tier's
            # row capacity rises to FUSED_ROWS_WIDE. Dims that are multiples of 256 keep the
            # N = 256 instance for the <= 16-row launch only (it is the faster 16-row tiling)
            if self.support_fused:
                self.mtile_ok = MTILE and bool(self.multi_up.mul1)
                self.fused_rows = FUSED_ROWS_WIDE if self.mtile_ok else TEMP_ROWS_FUSED
                R = self.fused_rows
                C = ext.exl3_moe_max_concurrency(torch.device(device).index)
                self.fused_mode_buffers = FusedBuffers(
                    temp_state_g = g_tensor_cache.get(device, (C, R, H), torch.half, "moe2_temp_state_g"),
                    temp_state_u = g_tensor_cache.get(device, (C, R, H), torch.half, "moe2_temp_state_u"),
                    temp_intermediate_g = g_tensor_cache.get(device, (C, R, I), torch.half, "moe2_temp_intermediate_g"),
                    temp_intermediate_u = g_tensor_cache.get(device, (C, R, I), torch.half, "moe2_temp_intermediate_u"),
                )
                self.f_threshold = min(self.num_experts // self.num_experts_per_tok, 4)


    def _tier_register(self, device):
        """LayerTier when the expert store is on and this layer qualifies, else None (resident load)"""
        if (device is None or torch.device(device).type != "cuda" or
                self.cpu_split_first is not None or self.cpu_offload or
                os.environ.get("EXL3_EXPERT_BUDGET_GIB", "") == ""):
            return None
        from .expert_store import get_store
        return get_store(self.config).register(self, device)
    def _moeg_ok(self):
        # Grouped prefill tier preconditions (see MOEG), evaluated once
        ok = self.__dict__.get("_moeg_ok_cache")
        if ok is None:
            ok = self._moeg_ok_eval()
            self._moeg_ok_cache = ok
        return ok

    def _moeg_ok_eval(self):
        if not MOEG or self.fused_mode_buffers is None or not self.gated or not self.mtile_ok:
            return False
        if self.activation_fn_idx not in (0, 1):
            return False
        I = self.fused_mode_buffers.temp_intermediate_g.shape[2]
        H = self.hidden_size
        if H % 128 or I % 128 or H % 32 or I % 32:
            return False
        if not (self.multi_gate.mul1 and self.multi_up.mul1 and self.multi_down.mul1):
            return False
        if not hasattr(ext, "exl3_moe_grouped"):
            return False
        return bool(ext.exl3_moe_grouped_supported(self.multi_gate.K, self.multi_up.K, self.multi_down.K))

    def load_routing(self, **kwargs):

        if self.interm_div != 1.0 and self.router_type == "std":
            # std routing has no scaling factor; fold the interm_div compensation into the
            # per-expert scale, which routing_std applies after top-k normalization. Both the
            # GPU and CPU-offload load paths come through here, and unload clears the tensor
            if self.per_expert_scale is None:
                self.per_expert_scale = torch.full(
                    (self.num_experts,), self.interm_div, dtype = torch.bfloat16, device = self.device)
            else:
                self.per_expert_scale = (self.per_expert_scale.float() * self.interm_div).to(torch.bfloat16)

        router_logits_bsz1 = torch.empty((1, self.num_experts), dtype = torch.half, device = self.device)
        routing_weights_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.half, device = self.device)
        selected_experts_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.long, device = self.device)

        self.routing_cfg = RoutingCFG(
            gate_tensor = self.routing_gate.inner.weight,
            router_bias = getattr(self.routing_gate.inner, "bias", None),
            gate_tensor_t = None,  # created lazily on first bsz-1 call (weights may be deferred here)
            num_experts = self.num_experts,
            num_experts_per_tok = self.num_experts_per_tok,
            router_logits_bsz1 = router_logits_bsz1,
            routing_weights_bsz1 = routing_weights_bsz1,
            selected_experts_bsz1 = selected_experts_bsz1,
            e_score_correction_bias = self.e_score_correction_bias,
            e_score_bias_h = None,
            tid2eid = self.tid2eid,
            e_score_bias_vl = self.e_score_bias_vl,
            routed_scaling_factor = self.routed_scaling_factor,
            n_group = self.n_group,
            topk_group = self.topk_group,
            per_expert_scale = self.per_expert_scale,
        )
        _prepare_hip_router_gate_t(self.routing_cfg)


    @override
    def load(self, device: torch.Device, **kwargs):
        # CPU expert offload (see block_sparse_mlp_cpu.py): a whole-layer claim replaces the
        # GPU load entirely; a split registration shrinks the module to its GPU slice first
        if self.cpu_maybe_offload_load(device, **kwargs):
            return
        self.cpu_maybe_split_load(device, **kwargs)
        # Tiered experts (expert_store.py, EXL3_EXPERT_BUDGET_GIB): the routed experts are not loaded here, only
        # registered; their bytes are read on demand into the store's slots
        tier = self._tier_register(device)
        if tier is None:
            super().load(device, **kwargs)
        else:
            self.device = device
            expert_set = set(self.gates + self.ups + self.downs)
            for module in self.modules:
                if module not in expert_set:
                    module.load(device, **kwargs)
            self.tier = tier

        if self.e_score_correction_bias_key:
            for k in [self.e_score_correction_bias_key, "gate.e_score_correction_bias"]:
                esb = self.config.stc.get_tensor(
                    f"{self.key}.{k}",
                    self.device,
                    optional = True,
                    allow_bf16 = True,
                    no_defer = True,
                )
                if esb is not None:
                    self.e_score_correction_bias = esb if esb.dtype == torch.half else esb.float()
                    break
        if self.e_score_bias_vl_key:
            esb_vl = self.config.stc.get_tensor(
                f"{self.key}.{self.e_score_bias_vl_key}",
                self.device,
                optional = True,
                allow_bf16 = True,
                no_defer = True,
            )
            self.e_score_bias_vl = esb_vl.float() if esb_vl is not None else None
        if self.tid2eid_key:
            self.tid2eid = self.config.stc.get_tensor(
                f"{self.key}.{self.tid2eid_key}",
                self.device,
                no_defer = True,
            )
        if self.per_expert_scale_key:
            self.per_expert_scale = self.config.stc.get_tensor(
                f"{self.key}.{self.per_expert_scale_key}",
                self.device,
                optional = True,
                allow_bf16 = True,
            )
        if device is not None and torch.device(device).type == "cuda":
            self.cpu_post_load()
            self.load_local(**kwargs)
            self.load_routing(**kwargs)


    def _batch_recon_layer(self, y):
        """Per-module batched reconstruct state (moe_batch_recon.BatchReconLayer), built on
        first use; None when the module's configuration isn't covered (same conditions as the
        DQ path plus unpadded hidden dims; fp32 intermediates as in the DQ path are supported)"""
        if self.batch_recon is False:
            return None
        if self.batch_recon is None:
            ok = (
                self.tier is None and  # reads the per-expert scale tensors, which a tiered layer does not keep
                BATCH_RECON and self.bc is not None and self.support_quant_paths and
                hasattr(ext, "reconstruct_batch") and
                self.interm_dtype in (torch.half, None, torch.float) and self.multi_up is not None and
                self.multi_up.in_features == y.shape[1] and
                self.multi_down.out_features == y.shape[1] and
                self.multi_down.in_features == self.multi_up.out_features
            )
            if not ok:
                self.batch_recon = False
                return None
            from .moe_batch_recon import BatchReconLayer
            mg, mu, md = self.multi_gate, self.multi_up, self.multi_down
            scales = {p: ([l.inner.suh for l in m.linears], [l.inner.svh for l in m.linears])
                      for p, m in (("g", mg), ("u", mu), ("d", md)) if m is not None}
            self.batch_recon = BatchReconLayer(
                (mg.in_features, mg.out_features, mg.K) if mg is not None else None,
                (mu.in_features, mu.out_features, mu.K),
                (md.in_features, md.out_features, md.K),
                mg.q_cb() if mg is not None else None, mu.q_cb(), md.q_cb(),
                self.activation_fn, self.act_limit, self.device, scales,
                interm_fp32 = self.interm_dtype != torch.half)
            self.batch_recon.set_static_pointers(
                mg.ptrs_trellis if mg is not None else None, mu.ptrs_trellis, md.ptrs_trellis)
        return self.batch_recon

    def prefill_worst_case_parts(self, rows: int, assignments: int) -> tuple[int, int]:
        """Upper bound on the GPU-expert prefill transients of a `rows`-token chunk with
        `assignments` routed rows landing on GPU-resident experts, as (fixed, variable) bytes:
        the fp32 accumulator, the output state and the deterministic slot scratch (every
        assignment slotted, padded) live for the whole call; the larger of one
        batched-reconstruct group and the per-expert dequant path with every row on one expert
        is the working set on top. The autosplit measuring forward routes a dummy state, so
        what it sees of these depends on that routing; the loader takes this bound instead
        (autosplit_extra_measure)"""
        if self.cpu_offload or (self.multi_up is None and self.mixed_k is None):
            return 0, 0
        h = self.expert_size
        if self.mixed_k is not None:
            # compacted picks + per-pick fp32 results + workspace of the largest group (moe_mixedk.prefill)
            return (rows + 1) * h * 4, assignments * h * 4 * 2
        fixed = (rows + 1) * h * 4 + rows * h * 2
        if FUSED_DET:
            fixed += (int(assignments * PAD_MAX) + 1) * h * 4
        r = min(rows, assignments)
        isz = (self.interm_dtype or torch.float).itemsize
        per_expert = r * (2 * self.intermediate_size_padded * isz + 2 * h * 2 + h * 4)
        if self.interm_dtype != torch.half:
            per_expert += r * self.intermediate_size_padded * 2
        recon = self._batch_recon_layer(torch.empty((1, h), dtype = torch.half, device = self.device))
        batched = recon.worst_case_bytes(assignments, slot_mode = FUSED_DET) if recon is not None else 0
        return fixed, max(per_expert, batched)

    def autosplit_extra_measure(self, params):
        """Autosplit loader hook: allocate (and drop) the worst-case prefill transient so the
        device keeps headroom for it. The CPU-offload host's per-device stream state and this
        layer's tier statics are allocated for real here: the measuring forward skips the CPU
        path, and they would otherwise appear unaccounted on the first real prefill"""
        if os.environ.get("EXL3_AUTOSPLIT_WORSTCASE", "1") == "0":
            return
        rows = getattr(self, "_measure_rows", 0)
        if not rows or self.device is None or self.device.type != "cuda":
            return
        A = rows * self.num_experts_per_tok
        host = getattr(self, "cpu_host", None)
        if host is not None and getattr(self, "cpu_layer_idx", None) is not None:
            # Split layers: the assignments divide between GPU-resident and CPU-resident
            # experts. Both sides' whole-call buffers live at once; the CPU side's group
            # working set is freed before the GPU side allocates its own. Take the worst split
            total = 0
            for a in (A * i // 4 for i in range(5)):
                gf, gv = self.prefill_worst_case_parts(rows, a)
                cf, cv = host.prefill_worst_case_parts(self.cpu_layer_idx, rows, self.device, A - a)
                total = max(total, gf + cf + max(gv, cv))
        else:
            total = sum(self.prefill_worst_case_parts(rows, A))
        if total > 0:
            t = torch.empty((total,), dtype = torch.uint8, device = self.device)
            del t

    def _run_batch_recon(self, recon, y, fhs_ext, token_sorted, weight_sorted, expert_count_list, groups,
                         scratch = None, tables = None):
        """Run the planned groups through the batched reconstruct tier. With a slot scratch each
        group's down projection lands in its slots (summed later by exl3_moe_gather); otherwise
        each group accumulates into fhs_ext (rows + 1, the last row a padding sink). Returns the
        set of experts handled."""
        rows = y.shape[0]
        y_ext = torch.cat([y, torch.zeros((1, y.shape[1]), dtype = y.dtype, device = y.device)])
        tok_ext = torch.cat([token_sorted, torch.full((1,), rows, dtype = token_sorted.dtype, device = y.device)])
        w_ext = torch.cat([weight_sorted.half(), torch.zeros((1,), dtype = torch.half, device = y.device)])
        starts = [0]
        for c in expert_count_list:
            starts.append(starts[-1] + c)
        handled = set()
        slot = None
        if scratch is not None:
            # groups were laid out after the fused slots, in plan order, len(grp) * cmax each
            slot = self._fused_slots if self.fused_mode_buffers is not None else 0
        for grp in groups:
            out_slab = None
            if scratch is not None:
                cmax = max(expert_count_list[e] for e in grp)
                out_slab = scratch[slot : slot + len(grp) * cmax]
                slot += len(grp) * cmax
            recon.run_group(
                y_ext, fhs_ext, tok_ext, w_ext,
                grp, [starts[e] for e in grp], [expert_count_list[e] for e in grp],
                out_slab = out_slab)
            handled.update(grp)
        return handled

    @override
    def unload(self):
        self.cpu_unload()
        if self.tier is not None:
            self.tier.store.unregister(self.tier)
            self.tier = None
        self.bc = None
        self.fused_mode_buffers = None
        self.dec_moe_mcg = None
        self.dec_shf = None
        self.batch_recon = None
        self.support_hip_grouped = False
        self.support_dec_moe = False
        self.hip_grouped_buffers = None
        self.support_hip_prefill = False
        self.hip_prefill_buffers = None
        self.support_hip_wmma = False
        self._pf_assignments = 0
        self.hip_grouped_lora_blocked = False
        if self.mixed_k is not None:
            self.mixed_k.unload()
            self.mixed_k = None
        self.mixed_outputs = None
        if self.multi_gate is not None:
            self.multi_gate.unload()
            self.multi_gate = None
        if self.multi_up is not None:
            self.multi_up.unload()
            self.multi_up = None
        if self.multi_down is not None:
            self.multi_down.unload()
            self.multi_down = None
        self.routing_cfg = None
        self.experts_cfg = None
        self.e_score_correction_bias = None
        self.tid2eid = None
        self.per_expert_scale = None
        self.bcast_sel_bsz1 = None
        self.bcast_weights_bsz1 = None
        super().unload()


    @override
    def dec_shf_setup(self):
        """decB8: shared expert folded into the batch-1 routed decode (exl3_dec_moe_shared) as slot
        topk with weight 1. Eligible when the shared expert is a plain one-slice quantized GatedMLP
        of the routed experts' shape (H, padded I), same activation / limit / codebook kind, and a
        K pair the kernel instantiates (routed K 2 / 2.5, shared K 4). Used per forward only under
        EXL3_DEC_SHARED_FOLD (default 1; 0 = separate shared launch)."""
        self.dec_shf = None
        se = self.shared_experts
        if not (se is not None and isinstance(se, GatedMLP) and hasattr(ext, "exl3_dec_moe_shared")): return
        if (self.shared_gate is not None or self.shared_experts_post_norm is not None or
            self.routed_post_norm is not None or self.latent_in is not None or self.latent_out is not None or
            self.alt_residual_channel or getattr(se, "tp_reduce", False) or se.num_slices != 1): return
        if not (se.activation_fn == "silu" and se.interm_dtype == torch.half and
                float(se.act_limit or 0.0) == float(self.act_limit)): return
        lins = (se.gates[0], se.ups[0], se.downs[0])
        if not all(getattr(l, "inner", None) is not None and hasattr(l.inner, "trellis") for l in lins): return
        g, u, d = (l.inner for l in lins)
        if any(l.bias is not None for l in (g, u, d)): return
        if any(getattr(l, "trim_padded_out", False) and l.out_features != l.out_features_unpadded for l in lins): return
        H, I = self.hidden_size, self.intermediate_size_padded
        if tuple(g.trellis.shape[:2]) != (H // 16, I // 16) or tuple(u.trellis.shape[:2]) != (H // 16, I // 16) or \
                tuple(d.trellis.shape[:2]) != (I // 16, H // 16): return
        if not (g.mcg == u.mcg == d.mcg == bool(self.dec_moe_mcg)): return
        if not self.dec_moe_mcg and not (g.mul1 and u.mul1 and d.mul1): return
        if not (g.K == u.K == d.K == 4 and self.multi_up.K in (2, 2.5) and self.multi_down.K in (2, 2.5)): return
        self.dec_shf = dict(
            t = (g.trellis, g.suh, g.svh, u.trellis, u.suh, u.svh, d.trellis, d.suh, d.svh),
            K = (float(g.K), float(d.K)),
            act = g_tensor_cache.get(self.device, (self.num_experts_per_tok + 1, I), torch.half, "dec_shf_act"),
        )

    def _fq_grouped_ok(self, y: torch.Tensor) -> bool:
        """Eligibility for the foreign-quant grouped decode path (Bolt latent
        MoE: gateless relu2, IQ2_XXS routed experts, Q8_0 latent projs). Any
        configuration this does not cover returns False and the caller falls
        through to the established paths. BOLT_FQ_MOE_GROUPED=0 disables."""
        if os.environ.get("BOLT_FQ_MOE_GROUPED", "1") == "0":
            return False
        if not hasattr(ext, "fq_moe_iq2xxs_grouped"):
            return False
        if self.gated:
            return False
        if y.dtype != torch.half or not y.is_contiguous():
            return False
        from .foreign_linear import ForeignLinear
        if self.latent_in is None or self.latent_out is None:
            return False
        if not (isinstance(self.latent_in, ForeignLinear) and self.latent_in.format == "q8_0"):
            return False
        if not (isinstance(self.latent_out, ForeignLinear) and self.latent_out.format == "q8_0"):
            return False
        if len(self.ups) != self.num_experts or len(self.downs) != self.num_experts:
            return False
        for m in self.ups + self.downs:
            if not (isinstance(m, ForeignLinear) and m.format == "iq2_xxs"):
                return False
            if m.packed is None or not m.packed.is_contiguous():
                return False
        if self.shared_experts is not None:
            se = self.shared_experts
            if len(getattr(se, "ups", [None])) != 1 or len(getattr(se, "downs", [None])) != 1:
                return False
            for m in se.ups + se.downs:
                if not (isinstance(m, ForeignLinear) and m.format == "iq2_xxs"):
                    return False
                if m.packed is None or not m.packed.is_contiguous():
                    return False
        if self.router_pre_norm is not None or self.routed_pre_norm is not None:
            return False
        if self.routed_post_norm is not None or self.shared_gate is not None:
            return False
        if self.shared_experts_post_norm is not None:
            return False
        if self.alt_residual_channel or self.tp_reduce or self.tp_mode is not None:
            return False
        if self.routing_device is not None or self.cpu_split_first is not None:
            return False
        if self.cpu_offload:
            return False
        K = self.expert_size
        I = self.intermediate_size
        if K % 256 != 0 or I % 256 != 0 or K > 4096 or I > 4096:
            return False
        if self.ups[0].in_features != K or self.ups[0].out_features != I:
            return False
        if self.downs[0].in_features != I or self.downs[0].out_features != K:
            return False
        return True

    def _forward_foreign_grouped(self, x, y, selected_experts, routing_weights,
                                 params, out_dtype, eshape):
        """Grouped foreign MoE for batch-1 latent decode. Returns the routed
        sum projected back to the residual width (same value as the dense
        per-expert torch path within fp16 tolerance), or None when ineligible.
        Skips the dense path's argsort / bincount+tolist host sync /
        index_select / index_add_ as well as its 2*topk launches: experts run
        in one fq_moe_iq2xxs_grouped call (slot kernel + weighted fold)."""
        if not self._fq_grouped_ok(y):
            return None
        S = self.num_experts_per_tok
        if tuple(selected_experts.shape) != (1, S):
            return None
        sel = selected_experts.reshape(-1)
        wts = routing_weights.reshape(-1)
        if wts.dtype != torch.half:
            return None
        if not sel.is_contiguous():
            sel = sel.contiguous()
        if not wts.is_contiguous():
            wts = wts.contiguous()
        h = y.reshape(1, -1)
        dev = h.device
        K = h.shape[1]
        # Expert pointer tables (weights are static: rebuild only when the
        # allocations change).
        tbl = getattr(self, "_fq_ptrs", None)
        if tbl is None or tbl[0] != self.ups[0].packed.data_ptr() or \
                tbl[1] != self.downs[0].packed.data_ptr() or tbl[4] != dev:
            up_ptrs = torch.tensor([u.packed.data_ptr() for u in self.ups],
                                   dtype=torch.int64, device=dev)
            down_ptrs = torch.tensor([d.packed.data_ptr() for d in self.downs],
                                     dtype=torch.int64, device=dev)
            up_stride = (self.ups[0].in_features // 256) * 66
            down_stride = (self.downs[0].in_features // 256) * 66
            tbl = (self.ups[0].packed.data_ptr(), self.downs[0].packed.data_ptr(),
                   up_ptrs, down_ptrs, dev, up_stride, down_stride)
            self._fq_ptrs = tbl
        _, _, up_ptrs, down_ptrs, _, up_stride, down_stride = tbl
        if getattr(self, "_fq_partial", None) is None or \
                self._fq_partial.shape != (S, K) or self._fq_partial.device != dev:
            self._fq_partial = torch.empty((S, K), dtype=torch.float, device=dev)
            self._fq_routed = torch.empty((1, K), dtype=torch.float, device=dev)
        ext.fq_moe_iq2xxs_grouped(h, sel, wts, up_ptrs, down_ptrs,
                                   up_stride, down_stride,
                                   self._fq_partial, self._fq_routed)
        final_hidden_states = self.latent_out.forward(self._fq_routed, params)
        if self.shared_experts is not None:
            se_params = {**params, "moe_valu": True} if _MOE_VALU else params
            se = self._forward_foreign_shared(x, se_params)
            if se is None:
                se = self.shared_experts.forward(x, se_params)
            final_hidden_states = final_hidden_states + se.to(torch.float)
        if out_dtype is not None:
            final_hidden_states = final_hidden_states.to(out_dtype)
        return final_hidden_states

    def _fq_grouped_kq_ok(self, y: torch.Tensor) -> str | None:
        """Eligibility for the KQ grouped decode path (MiniCPM SwiGLU MoE:
        gated, ForeignLinear KQ/IQ4_XS routed experts, no latent). Returns the
        format string or None. MCPM_FQ_MOE_GROUPED=0 disables."""
        if os.environ.get("MCPM_FQ_MOE_GROUPED", "1") == "0":
            return None
        if not self.gated:
            return None
        if y.dtype != torch.half or not y.is_contiguous():
            return None
        from .foreign_linear import ForeignLinear
        if self.latent_in is not None or self.latent_out is not None:
            return None
        if len(self.gates) != self.num_experts or len(self.ups) != self.num_experts \
                or len(self.downs) != self.num_experts:
            return None
        fmt = None
        for m in self.gates + self.ups + self.downs:
            if not isinstance(m, ForeignLinear):
                return None
            if m.format not in ("q4_k", "q5_k", "q6_k", "iq4_xs"):
                return None
            if fmt is None:
                fmt = m.format
            elif m.format != fmt:
                return None  # mixed formats (Q4_K_M gate/up vs down) need separate dots
            if m.packed is None or not m.packed.is_contiguous():
                return None
        if self.router_pre_norm is not None or self.routed_pre_norm is not None:
            return None
        if self.routed_post_norm is not None:
            return None
        if self.shared_experts_post_norm is not None:
            return None
        if self.alt_residual_channel or self.tp_reduce or self.tp_mode is not None:
            return None
        if self.routing_device is not None or self.cpu_split_first is not None:
            return None
        if self.cpu_offload:
            return None
        K = self.expert_size
        I = self.intermediate_size
        if K % 256 != 0 or I % 256 != 0:
            return None
        if self.gates[0].in_features != K or self.gates[0].out_features != I:
            return None
        if self.downs[0].in_features != I or self.downs[0].out_features != K:
            return None
        if self.activation_fn not in ("silu", None):
            return None
        return fmt

    def _forward_foreign_grouped_kq(self, x, y, selected_experts, routing_weights,
                                    params, out_dtype, eshape):
        """Grouped KQ MoE for batch-1 SwiGLU decode. One fq_moe kernel runs
        all selected experts (gate+up dots, torch-half SwiGLU, down dots,
        weighted fold), replacing the per-expert torch loop (top-k launches
        + python dispatch per layer). Returns routed+shared sum or None."""
        fmt = self._fq_grouped_kq_ok(y)
        if fmt is None:
            return None
        S = self.num_experts_per_tok
        if tuple(selected_experts.shape) != (1, S):
            return None
        sel = selected_experts.reshape(-1)
        wts = routing_weights.reshape(-1)
        if wts.dtype != torch.half:
            return None
        if not sel.is_contiguous():
            sel = sel.contiguous()
        if not wts.is_contiguous():
            wts = wts.contiguous()
        h = y.reshape(1, -1)
        dev = h.device
        K = h.shape[1]
        _BB = {"q4_k": 144, "q5_k": 176, "q6_k": 210, "iq4_xs": 136}
        bb = _BB[fmt]
        tbl = getattr(self, "_fq_kq_ptrs", None)
        if tbl is None or tbl[0] != self.gates[0].packed.data_ptr() or tbl[4] != dev:
            gate_ptrs = torch.tensor([m.packed.data_ptr() for m in self.gates],
                                     dtype=torch.int64, device=dev)
            up_ptrs = torch.tensor([m.packed.data_ptr() for m in self.ups],
                                   dtype=torch.int64, device=dev)
            down_ptrs = torch.tensor([m.packed.data_ptr() for m in self.downs],
                                     dtype=torch.int64, device=dev)
            tbl = (self.gates[0].packed.data_ptr(), gate_ptrs, up_ptrs, down_ptrs, dev)
            self._fq_kq_ptrs = tbl
        _, gate_ptrs, up_ptrs, down_ptrs, _ = tbl
        I = self.intermediate_size
        gate_stride = (K // 256) * bb
        up_stride = gate_stride
        down_stride = (I // 256) * bb
        if getattr(self, "_fq_kq_partial", None) is None or \
                self._fq_kq_partial.shape != (S, K) or self._fq_kq_partial.device != dev:
            self._fq_kq_partial = torch.empty((S, K), dtype=torch.float, device=dev)
            self._fq_kq_routed = torch.empty((1, K), dtype=torch.float, device=dev)
        _FN = {"q4_k": "fq_moe_q4k_grouped_q8", "q5_k": "fq_moe_q5k_grouped_q8",
               "q6_k": "fq_moe_q6k_grouped_q8", "iq4_xs": "fq_moe_iq4xs_grouped_q8"}
        getattr(ext, _FN[fmt])(h, sel, wts, gate_ptrs, up_ptrs, down_ptrs,
                               gate_stride, up_stride, down_stride,
                               self._fq_kq_partial, self._fq_kq_routed)
        final_hidden_states = self._fq_kq_routed
        if self.shared_experts is not None:
            se_params = {**params, "moe_valu": True} if _MOE_VALU else params
            se = self.shared_experts.forward(x, se_params)
            if self.shared_gate is not None:
                ext.add_sigmoid_gate_proj(se, x, final_hidden_states,
                                          self.shared_gate.inner.weight)
            else:
                final_hidden_states = final_hidden_states + se.to(torch.float)
        if out_dtype is not None:
            final_hidden_states = final_hidden_states.to(out_dtype)
        return final_hidden_states

    def _forward_foreign_shared(self, x, params):
        """Fused dense shared-expert up+relu2 (fq_mlp.cu) + ForeignLinear
        down for batch-1 decode. Bit-identical to the shared MLP torch path
        (same up dot grid, torch-exact relu2 epilogue, same down dot), so it
        only kills the torch relu/square launches, the intermediate u
        allocation, and u's global round-trip. Any configuration this does
        not cover returns None and the caller falls through to the torch
        path. BOLT_FQ_MLP_SHARED=0 disables."""
        if os.environ.get("BOLT_FQ_MLP_SHARED", "1") == "0":
            return None
        if not hasattr(ext, "fq_mlp_shared_iq2xxs"):
            return None
        se = self.shared_experts
        if se is None or se.num_slices != 1:
            return None
        if se.activation_fn != "relu2" or se.interm_dtype == torch.float:
            return None
        from .foreign_linear import ForeignLinear
        up, down = se.ups[0], se.downs[0]
        if not (isinstance(up, ForeignLinear) and up.format == "iq2_xxs"):
            return None
        if not (isinstance(down, ForeignLinear) and down.format == "iq2_xxs"):
            return None
        if up.packed is None or down.packed is None:
            return None
        K, I = up.in_features, up.out_features
        if down.in_features != I or I != se.intermediate_size:
            return None
        if x.shape[-1] != K or x.numel() != K:
            return None
        xr = x.reshape(1, K)
        if xr.dtype != torch.half or not xr.is_contiguous():
            return None
        dev = xr.device
        h = getattr(self, "_fq_shared_h", None)
        if h is None or tuple(h.shape) != (1, I) or h.device != dev:
            h = torch.empty((1, I), dtype=torch.half, device=dev)
            self._fq_shared_h = h
        ext.fq_mlp_shared_iq2xxs(xr, up.packed.reshape(I, -1, 66), h)
        return down.forward(h, params)

    def _forward_foreign_prefill_grouped(self, x, y, selected_experts, routing_weights,
                                         params, out_dtype, eshape):
        """MiniCPM IQ4_XS prefill MoE (rows >= MCPM_FQ_PF_GROUPED_MIN): two fused
        Triton grouped GEMMs over expert-sorted rows (gate+up+SwiGLU, then
        down x routing weight) that decode the IQ4_XS weights in the K loop
        (fq_moe_prefill.py), tile map built on device (no host sync, ~15
        launches per layer instead of thousands). Same fp16-rounded weights
        and fp32 accumulation as the per-expert path. Returns the block output
        or None. MCPM_FQ_PF_GROUPED=0 disables."""
        if os.environ.get("MCPM_FQ_PF_GROUPED", "1") == "0":
            return None
        R = y.shape[0]
        if R < int(os.environ.get("MCPM_FQ_PF_GROUPED_MIN", "256")):
            return None
        if self._fq_grouped_kq_ok(y) != "iq4_xs" or self.act_limit or self.activation_fn not in ("silu", None):
            return None
        if (self.num_local_experts or self.num_experts) != self.num_experts:
            return None
        top_k = self.num_experts_per_tok
        if tuple(selected_experts.shape) != (R, top_k) or selected_experts.dtype != torch.long:
            return None
        from .fq_moe_prefill import build_tiles, gmm_iq4, perm_cols
        E = self.num_experts
        dev = y.device
        K = y.shape[1]
        I = self.intermediate_size
        wts = routing_weights
        if wts.dtype != torch.half:
            wts = wts.to(torch.half)
        flat_expert = selected_experts.reshape(-1)
        order = flat_expert.argsort(stable=True)
        token_sorted = order // top_k
        weight_sorted = wts.reshape(-1)[order]
        counts = _scatter_expert_count(flat_expert, E + 1)[:E]
        M = R * top_k
        tile_e, tile_r0, offs = build_tiles(counts, M, 64)
        tbl = getattr(self, "_fq_pf_ptrs", None)
        if tbl is None or tbl[0] != self.gates[0].packed.data_ptr() or tbl[3] != dev:
            gu = torch.tensor([m.packed.data_ptr() for m in self.gates + self.ups], dtype=torch.int64, device=dev)
            dn_p = torch.tensor([m.packed.data_ptr() for m in self.downs], dtype=torch.int64, device=dev)
            tbl = (self.gates[0].packed.data_ptr(), gu, dn_p, dev)
            self._fq_pf_ptrs = tbl
        _, gu_ptrs, dn_ptrs, _ = tbl
        xs = perm_cols(y.index_select(0, token_sorted))
        a = torch.empty((M, I), dtype=torch.half, device=dev)
        gmm_iq4(xs, gu_ptrs, a, tile_e, tile_r0, offs, I, K, E, swiglu=True)
        dn = torch.empty((M, K), dtype=torch.half, device=dev)
        gmm_iq4(a, dn_ptrs, dn, tile_e, tile_r0, offs, K, I, E, wrow=weight_sorted)
        inv = torch.empty_like(order)
        inv[order] = torch.arange(order.numel(), device=dev)    # row of each (token, k) pick in the sorted layout
        final_hidden_states = dn.index_select(0, inv).view(R, top_k, K).float().sum(1)
        final_hidden_states = final_hidden_states.reshape(eshape)
        if self.shared_experts:
            y_se = self.shared_experts.forward(x, {**params, "moe_valu": True} if _MOE_VALU else params)
            if self.shared_gate:
                ext.add_sigmoid_gate(y_se, self.shared_gate.forward(x, params), final_hidden_states)
            else:
                final_hidden_states += y_se
        if out_dtype is not None:
            final_hidden_states = final_hidden_states.to(out_dtype)
        return final_hidden_states

    def _forward_foreign_prefill(self, x, y, selected_experts, routing_weights,
                                 params, out_dtype, eshape):
        """Lean foreign prefill MoE (rows > 1): the dense per-expert torch
        path (full-fp16 dequant + cublas mm per expert) WITHOUT the EXL3
        tier machinery around it (fused/recon tier planning, numpy slot
        tables, prefill-shaped scratch + gather, extra syncs). Same op
        sequence per expert (sort, index_select, up, act, down, weighted
        index_add_), hence bitwise-identical to the stock dense path for
        foreign configs; only the scaffolding is skipped. Returns the
        full block output (routed + shared + latent_out, like the tail
        below), or None when ineligible. BOLT_FQ_PF=0 disables."""
        if os.environ.get("BOLT_FQ_PF", "1") == "0":
            return None
        if not self._fq_grouped_ok(y):
            return None
        if (self.num_local_experts or self.num_experts) != self.num_experts:
            return None
        R = y.shape[0]
        if R <= 1:
            return None
        if tuple(selected_experts.shape)[0] != R:
            return None
        num_tokens, top_k = selected_experts.shape
        if top_k != self.num_experts_per_tok:
            return None
        if selected_experts.dtype != torch.long:
            return None
        if not selected_experts.is_contiguous():
            selected_experts = selected_experts.contiguous()
        wts = routing_weights
        if wts.dtype != torch.half:
            wts = wts.to(torch.half)
        if not wts.is_contiguous():
            wts = wts.contiguous()
        if not x.is_contiguous() or x.dtype != torch.half:
            return None
        num_ex = self.num_experts
        E = num_ex
        dev = y.device
        K = y.shape[1]
        flat_expert = selected_experts.reshape(-1)
        flat_weight = wts.reshape(-1)
        flat_token = buffered_interleaved_arange(num_tokens, top_k, device=dev)
        order = flat_expert.argsort(stable=True)
        token_sorted = flat_token[order]
        weight_sorted = flat_weight[order]
        expert_count = _scatter_expert_count(flat_expert, E + 1)
        expert_count_list = expert_count.tolist()
        final_hidden_states = torch.zeros((R, K), dtype=torch.float, device=dev)
        # Single sorted gather; per-expert narrow views (no per-expert copy).
        # Same values through the same kernels -> bitwise-identical
        # (gated A/B: pf_1g.sh). BOLT_FQ_PF1G=0 restores per-expert
        # index_select.
        one_gather = os.environ.get("BOLT_FQ_PF1G", "1") != "0"
        xs_all = y.index_select(0, token_sorted) if one_gather else None
        start = 0
        for expert_idx in range(num_ex):
            count = expert_count_list[expert_idx]
            end = start + count
            if count > 0:
                top_x = token_sorted[start:end]
                w = weight_sorted[start:end].unsqueeze(1)
                if one_gather:
                    current_state = xs_all.narrow(0, start, count)
                else:
                    current_state = y.index_select(0, top_x)
                u = self.ups[expert_idx].forward(current_state, params)
                if self.gated:
                    g = self.gates[expert_idx].forward(current_state, params)
                    a = u if self.interm_dtype == torch.half else torch.empty_like(u, dtype=torch.half)
                    self.activation_fn_call(g, u, a, self.act_limit)
                else:
                    a = self.gateless_act(u)
                    if a.dtype != torch.half:
                        a = a.half()
                current_state = self.downs[expert_idx].forward(a, params)
                current_state.mul_(w)
                final_hidden_states.index_add_(0, top_x, current_state)
            start = end
        final_hidden_states = final_hidden_states.reshape(eshape)
        if self.latent_out is not None:
            final_hidden_states = self.latent_out.forward(final_hidden_states.to(torch.half), params)
        if self.shared_experts:
            y_se = self.shared_experts.forward(x, {**params, "moe_valu": True} if _MOE_VALU else params)
            final_hidden_states += y_se
        if out_dtype is not None:
            final_hidden_states = final_hidden_states.to(out_dtype)
        return final_hidden_states

    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if _ROW_CHUNK_MAX and self.coop_rows < x.numel() // x.shape[-1] <= \
                (min(_ROW_CHUNK_MAX, _FUSED_3INST_MIN) if self.support_fused else _ROW_CHUNK_MAX) and self._row_chunk_ok() and \
                not params.get("autosplit_measure"):
            return self._forward_row_chunks(x, params, out_dtype)

        if self.alt_residual_channel:
            y = params["residual"].view(-1, self.hidden_size)
        else:
            y = x.view(-1, self.hidden_size)
        bsz = y.shape[0]
        if os.environ.get("BSZ_DEBUG"):
            print(f"[bsz_debug] layer={self.key} bsz={bsz} union={os.environ.get('EXL3_DEC_MOE_UNION')}")
        bc_sh_exp = False
        side_y = None
        #  R-row union MoE with the unique-expert table built on device (no host sync),
        # fed by the batched router -- independent of EXL3_DEC_MOE_UNION's bit-exact per-row loops
        union_dev = os.environ.get("EXL3_DEC_MOE_UNION_DEV", "0") != "0"
        # V2 selects the device-built unique-expert table without enabling the legacy
        # EXL3_DEC_MOE_UNION_DEV path globally and avoids the selected.to(CPU) host
        # synchronization in exl3_dec_moe_union. Default on since base0925 (test box):
        # +9 % MTP n1f2 tok/s, identical tokens on 4 prompts; EXL3_MOE_UNION_V2=0 disables.
        union_v2 = os.environ.get("EXL3_MOE_UNION_V2", "1") != "0"
        if params.get("autosplit_measure"):
            self._measure_rows = bsz
        # Shape of the routed sum: the experts' width is the latent width in a latent MoE
        eshape = x.shape[:-1] + (self.expert_size,)

        # Eligibility for the fused decode kernels (bsz 1..MAX_BSZN): computed up front so it
        # can override the f_threshold-based routing below (bsz>=f_threshold would otherwise
        # always fall through to the exl3_moe/dense path first, capping this tier's reach at
        # f_threshold-1 instead of MAX_BSZN). Expert-range shards (CPU split, TP) are masked
        # inside the kernel (out-of-range picks contribute exact zeros). Shared experts run
        # through BC_GatedMLP's own multi-row graph ahead of the kernel (see mlp.py)
        bszn_eligible = self.bc is not None and bsz <= self.coop_rows

        # Routing
        if self.router_pre_norm:
            z = self.router_pre_norm.forward(y, params, out_dtype = torch.half)
        else:
            z = y

        pre_routed = params.pop("dec_routed", None)
        if pre_routed is not None:
            selected_experts, routing_weights = pre_routed
        elif self.routing_gate is not None:
            selected_experts, routing_weights = self.routing_fn(bsz, self.routing_cfg, z, params)
        elif bsz == 1:
            # Stable broadcast targets rather than per-call allocations
            if self.bcast_sel_bsz1 is None:
                self.bcast_sel_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.long, device = self.device)
                self.bcast_weights_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.half, device = self.device)
            selected_experts = self.bcast_sel_bsz1
            routing_weights = self.bcast_weights_bsz1
        else:
            selected_experts = torch.empty((bsz, self.num_experts_per_tok), dtype = torch.long, device = self.device)
            routing_weights = torch.empty((bsz, self.num_experts_per_tok), dtype = torch.half, device = self.device)

        if _ROUTE_TRACE is not None:
            _ROUTE_TRACE.record(self.key, bsz, selected_experts)

        # Extra norm (Gemma4)
        if self.routed_pre_norm:
            y = self.routed_pre_norm.forward(y, params, out_dtype = torch.half)

        # Latent MoE: the experts (GPU and CPU-resident alike) see the projected input
        if self.latent_in is not None:
            y = self.latent_in.forward(y, params)

        # Foreign-quant grouped decode (Bolt latent MoE): one kernel runs all
        # selected experts (see _forward_foreign_grouped). Strict eligibility
        # inside; falls through to the paths below otherwise.
        # KQ grouped decode (MiniCPM SwiGLU MoE): same idea via the KQ dots
        # (see _forward_foreign_grouped_kq). Tried first (Bolt returns None
        # without latent projs anyway).
        if bsz == 1 and not params.get("activate_all_experts") and \
                not params.get("reconstruct") and not params.get("autosplit_measure"):
            grouped = self._forward_foreign_grouped_kq(
                x, y, selected_experts, routing_weights, params, out_dtype, eshape)
            if grouped is not None:
                return grouped
            grouped = self._forward_foreign_grouped(
                x, y, selected_experts, routing_weights, params, out_dtype, eshape)
            if grouped is not None:
                return grouped

        # Lean foreign prefill (Bolt latent MoE, rows > 1): same values as
        # the dense per-expert torch path, minus the EXL3 tier scaffolding
        # (see _forward_foreign_prefill). Strict eligibility inside; falls
        # through otherwise. Skipped for DFlash verify batches (bit-exact
        # union path owns those).
        if bsz > 1 and not params.get("activate_all_experts") and \
                not params.get("reconstruct") and not params.get("autosplit_measure") and \
                not params.get("dflash_verify"):
            pf = self._forward_foreign_prefill_grouped(
                x, y, selected_experts, routing_weights, params, out_dtype, eshape)
            if pf is None:
                pf = self._forward_foreign_prefill(
                    x, y, selected_experts, routing_weights, params, out_dtype, eshape)
            if pf is not None:
                return pf

        # Broadcast routing indices and weights
        if self.routing_device is not None:
            params["backend"].broadcast(selected_experts, src_device = self.routing_device)
            params["backend"].broadcast(routing_weights, src_device = self.routing_device)

        # Tiered experts: make the routed union resident and the pointer tables current before any expert kernel
        if self.tier is not None:
            self.tier.store.prepare(self.tier, selected_experts, params, z)

        # CPU expert offload (block_sparse_mlp_cpu.py): split layers hand the tail experts'
        # share to the worker now so it computes concurrently with the GPU expert paths below
        # (folded back in by cpu_split_combine); whole-layer offload replaces the routed sum
        cpu_partial = None
        cpu_pending = None
        if self.cpu_split_first is not None and not params.get("autosplit_measure"):
            cpu_partial, cpu_pending = self.cpu_split_submit(y, bsz, selected_experts, routing_weights)

        if self.cpu_offload:
            final_hidden_states = self.cpu_offload_forward(eshape, y, selected_experts, routing_weights, params)

        # Empty slice
        elif self.intermediate_size == 0 or self.num_local_experts == 0:
            final_hidden_states = torch.zeros(eshape, dtype = torch.float, device = y.device)

        # gfx11.5 batch-1 fused decode (exl3_dec_moe): two launches for the whole routed sum
        elif (
            bsz == 1 and self.support_dec_moe and self.tp_mode is None and not (_MOE_VALU and self.support_hip_grouped) and
            y.dtype == torch.half and y.is_contiguous() and
            selected_experts.dtype == torch.long and routing_weights.dtype == torch.half and
            selected_experts.numel() == self.num_experts_per_tok and
            not params.get("activate_all_experts") and
            not params.get("reconstruct") and not params.get("autosplit_measure")
        ):
            scratch, counters = dec_workspace(y.device)
            # decB8: shared expert folded into the same two launches (slot topk, weight 1)
            shf = (
                self.dec_shf is not None and cpu_partial is None and cpu_pending is None and
                os.environ.get("EXL3_DEC_SHARED_FOLD", "1") != "0"
            )
            # Fused residual add: the down kernel adds the routed sum straight into the fp32
            # residual when nothing else post-processes the sum
            res = params.get("mlp_residual_out")
            fuse_res = (
                res is not None and res.dtype == torch.float and res.is_contiguous() and
                res.numel() == self.hidden_size and self.expert_size == self.hidden_size and
                self.latent_out is None and self.routed_post_norm is None and (shf or not self.shared_experts) and
                cpu_partial is None and cpu_pending is None and not self.tp_reduce and out_dtype is None
            )
            if (
                _SHARED_SIDE_HIP and os.environ.get("EXL3_SHARED_SIDE", "0") != "0" and not shf and self.shared_experts is not None and isinstance(self.shared_experts, GatedMLP) and
                self.shared_gate is None and self.shared_experts_post_norm is None and cpu_partial is None and cpu_pending is None and
                not self.tp_reduce and not self.routed_post_norm
            ):
                # batch-1 decode: the shared expert's three small GEMVs do not depend on the routed experts, run them on a side
                # stream (own GEMV scratch/counters) while the routed kernels occupy the main one; joined at the shared add below
                from .quant.exl3 import dec_side_workspace
                side, ev_in, ev_out = _side(y.device)
                ev_in.record(torch.cuda.current_stream(y.device))
                with torch.cuda.stream(side):
                    side.wait_event(ev_in)
                    with dec_side_workspace():
                        side_y = self.shared_experts.forward(x, {**params, "moe_valu": True} if _MOE_VALU else params)
                    ev_out.record(side)
            if shf:
                ext.exl3_dec_moe_shared(
                    y,
                    res.view(1, -1) if fuse_res else self.dec_moe_out,
                    selected_experts.contiguous(),
                    routing_weights.contiguous(),
                    self.multi_gate.ptrs_trellis,
                    self.multi_gate.ptrs_suh,
                    self.multi_gate.ptrs_svh,
                    self.multi_up.ptrs_trellis,
                    self.multi_up.ptrs_suh,
                    self.multi_up.ptrs_svh,
                    self.multi_down.ptrs_trellis,
                    self.multi_down.ptrs_suh,
                    self.multi_down.ptrs_svh,
                    *self.dec_shf["t"],
                    self.dec_shf["act"],
                    scratch,
                    counters,
                    self.intermediate_size_padded,
                    self.multi_up.K,
                    self.multi_down.K,
                    *self.dec_shf["K"],
                    fuse_res,
                    self.act_limit,
                    self.dec_moe_mcg,
                )
                bc_sh_exp = True    # shared expert already summed in: skip the separate forward
            else:
                ext.exl3_dec_moe(
                    y,
                    res.view(1, -1) if fuse_res else self.dec_moe_out,
                    selected_experts.contiguous(),
                    routing_weights.contiguous(),
                    self.multi_gate.ptrs_trellis,
                    self.multi_gate.ptrs_suh,
                    self.multi_gate.ptrs_svh,
                    self.multi_up.ptrs_trellis,
                    self.multi_up.ptrs_suh,
                    self.multi_up.ptrs_svh,
                    self.multi_down.ptrs_trellis,
                    self.multi_down.ptrs_suh,
                    self.multi_down.ptrs_svh,
                    self.dec_moe_act,
                    scratch,
                    counters,
                    self.intermediate_size_padded,
                    self.multi_up.K,
                    self.multi_down.K,
                    fuse_res,
                    self.act_limit,
                    self.dec_moe_mcg,
                )
            if fuse_res:
                params["mlp_residual_done"] = True
                return res
            final_hidden_states = self.dec_moe_out.view(eshape)

        # R-row union decode (exl3_dec_moe_union), DFlash verify: bit-exact vs
        # bsz independent batch-1 exl3_dec_moe calls under shared routing --
        # each unique expert across the bsz rows' picks is decoded once and applied to every row
        # that picked it in the exact batch-1 per-row arithmetic order. Opt-in (default off):
        # Measured: no clear speed win yet at R=1 or R=6-8 (host-side unique/assign
        # build dominates), only R=2..5; EXL3_DEC_MOE_UNION=1 to use it anyway (e.g. to get a
        # bit-exact verify path even before that's fixed).
        elif (
            1 < bsz <= UNION_ROWS and self.support_dec_moe and self.tp_mode is None and
            not (_MOE_VALU and self.support_hip_grouped and bsz <= 8) and
            hasattr(ext, "exl3_dec_moe_union") and
            (union_dev or union_v2 or os.environ.get("EXL3_DEC_MOE_UNION", "0") != "0") and params.get("dflash_verify") and
            y.dtype == torch.half and y.is_contiguous() and
            selected_experts.dtype == torch.long and routing_weights.dtype == torch.half and
            tuple(selected_experts.shape) == (bsz, self.num_experts_per_tok) and
            selected_experts.is_contiguous() and routing_weights.is_contiguous() and
            not params.get("activate_all_experts") and
            not params.get("reconstruct") and not params.get("autosplit_measure")
        ):
            numex = self.num_experts_per_tok
            res = params.get("mlp_residual_out")
            fuse_res = (
                res is not None and res.dtype == torch.float and res.is_contiguous() and
                res.numel() == bsz * self.hidden_size and self.expert_size == self.hidden_size and
                self.latent_out is None and self.routed_post_norm is None and not self.shared_experts and
                cpu_partial is None and cpu_pending is None and not self.tp_reduce and out_dtype is None
            )
            out_buf = res.view(bsz, -1) if fuse_res else self.dec_moe_union_out[:bsz]
            scratch, counters = dec_workspace(y.device)
            sel_c, wts_c = selected_experts.contiguous(), routing_weights.contiguous()
            for lo, hi in _union_slices(x, bsz):
                # one slice = the exact 2..8 row call (_union_slices never leaves a lone row: that would be the batch-1 kernel)
                ext.exl3_dec_moe_union(
                    y[lo:hi], out_buf[lo:hi], sel_c[lo:hi], wts_c[lo:hi],
                    self.multi_gate.ptrs_trellis, self.multi_gate.ptrs_suh, self.multi_gate.ptrs_svh,
                    self.multi_up.ptrs_trellis, self.multi_up.ptrs_suh, self.multi_up.ptrs_svh,
                    self.multi_down.ptrs_trellis, self.multi_down.ptrs_suh, self.multi_down.ptrs_svh,
                    self.dec_moe_union_act[:(hi - lo) * numex],
                    self.dec_moe_union_down_part[:(hi - lo) * numex],
                    scratch, counters,
                    self.intermediate_size_padded,
                    self.multi_up.K,
                    self.multi_down.K,
                    fuse_res,
                    self.act_limit,
                    self.dec_moe_mcg,
                    union_dev or union_v2,
                )
            if fuse_res:
                params["mlp_residual_done"] = True
                return res
            final_hidden_states = self.dec_moe_union_out[:bsz].view(eshape)

        # Per-expert K layer (hot-3 / cold-2): one grouped launch set per K group on lut-remapped picks, the
        # group partial sums added (moe_mixedk.MixedKStore.decode). Rows above the grouped cap and below the
        # WMMA threshold run the dense per-expert path, which is K-agnostic.
        elif (
            self.mixed_k is not None and self.support_hip_grouped and self.tp_mode is None and
            _hip_grouped_rows_eligible(bsz, False) and
            tuple(y.shape) == (bsz, self.expert_size) and y.dtype == torch.half and
            tuple(selected_experts.shape) == (bsz, self.num_experts_per_tok) and
            selected_experts.dtype == torch.long and
            tuple(routing_weights.shape) == (bsz, self.num_experts_per_tok) and
            routing_weights.dtype == torch.half and
            y.is_contiguous() and selected_experts.is_contiguous() and routing_weights.is_contiguous() and
            os.environ.get("EXL3_HIP_GROUPED_MOE", "1") != "0" and
            (bsz == 1 or os.environ.get("EXL3_HIP_GROUPED_MOE_MULTIROW", "1") != "0") and
            os.environ.get("EXL3_GEMV", "1") != "0" and
            not params.get("activate_all_experts") and
            not params.get("reconstruct") and not params.get("autosplit_measure")
        ):
            final_hidden_states = self.mixed_k.decode(
                ext, y, selected_experts, routing_weights, self.hip_grouped_buffers, self.mixed_outputs,
                self.act_limit, _MOE_DEDUP,
            ).view(x.shape)

        # gfx12 grouped decode/verification path: selected IDs and weights stay on-device,
        # and every row-major assignment slot remains distinct.
        elif (
            self.mixed_k is None and
            _hip_grouped_rows_eligible(bsz, self.support_hip_prefill) and self.support_hip_grouped and
            self.tp_mode is None and
            tuple(y.shape) == (bsz, self.expert_size) and y.dtype == torch.half and
            tuple(selected_experts.shape) == (bsz, self.num_experts_per_tok) and
            selected_experts.dtype == torch.long and
            tuple(routing_weights.shape) == (bsz, self.num_experts_per_tok) and
            routing_weights.dtype == torch.half and
            y.is_contiguous() and selected_experts.is_contiguous() and routing_weights.is_contiguous() and
            os.environ.get("EXL3_HIP_GROUPED_MOE", "1") != "0" and
            (bsz == 1 or os.environ.get("EXL3_HIP_GROUPED_MOE_MULTIROW", "1") != "0") and
            os.environ.get("EXL3_GEMV", "1") != "0" and
            not params.get("activate_all_experts") and
            not params.get("reconstruct") and not params.get("autosplit_measure")
        ):
            buffers = self.hip_grouped_buffers
            assignments = bsz * self.num_experts_per_tok
            output = buffers.output[:bsz]
            ext.exl3_moe_gfx12_k3(
                y,
                output,
                selected_experts,
                routing_weights,
                self.multi_gate.ptrs_trellis,
                self.multi_gate.ptrs_suh,
                self.multi_gate.ptrs_svh,
                self.multi_up.ptrs_trellis,
                self.multi_up.ptrs_suh,
                self.multi_up.ptrs_svh,
                self.multi_down.ptrs_trellis,
                self.multi_down.ptrs_suh,
                self.multi_down.ptrs_svh,
                buffers.gu_had[:2 * assignments],
                buffers.gu_out[:2 * assignments],
                buffers.down_had[:assignments],
                buffers.down_out[:assignments],
                float(self.multi_up.K),
                _MOE_DEDUP,
                float(self.act_limit),
            )
            final_hidden_states = output.view(x.shape)

        # gfx12 native prefill path: the kernel consumes the expert-major grouping directly,
        # so the fused kernel's per-expert count readback and tier planning are skipped for
        # eligible shapes.
        elif (
            self.support_hip_prefill and self.tp_mode is None and
            (self.num_local_experts == self.num_experts or
             (self.cpu_split_first is not None and self.routing_first == 0 and
              self.num_local_experts == self.cpu_split_first and
              os.environ.get("EXL3_HIP_GROUPED_SPLIT", "1") != "0")) and
            _hip_prefill_rows_eligible(bsz) and
            y.dtype == torch.half and y.is_contiguous() and
            selected_experts.is_contiguous() and routing_weights.is_contiguous() and
            self._hip_k3_prefill_selected(bsz) and
            os.environ.get("EXL3_GEMV", "1") != "0" and
            not params.get("activate_all_experts") and
            not params.get("reconstruct") and not params.get("autosplit_measure")
        ):
            num_tokens, top_k = selected_experts.shape
            flat_expert_local = selected_experts.reshape(-1)
            # Expert-range shard (CPU split keeps the head slice): picks outside [0, E_local)
            # go to the sentinel bucket E_local so the kernel's per-expert counts and the
            # metadata pass only cover the resident experts (the kernel zero-masks the rest)
            E_local = self.num_local_experts or self.num_experts
            if E_local != self.num_experts:
                flat_expert_local = torch.where(
                    flat_expert_local < E_local, flat_expert_local,
                    torch.full_like(flat_expert_local, E_local))
            order = flat_expert_local.argsort(stable = True)
            if _moe_sync_free_count():
                expert_count = _scatter_expert_count(flat_expert_local, E_local + 1)
            else:
                expert_count = torch.bincount(flat_expert_local, minlength = E_local + 1)
            buffers = self._ensure_hip_prefill_buffers(num_tokens)
            assignments = num_tokens * top_k
            output = buffers.output[:num_tokens]
            ext.exl3_moe_gfx12_k3_prefill(
                y,
                output,
                selected_experts,
                routing_weights,
                order,
                expert_count,
                self.multi_gate.ptrs_trellis,
                self.multi_gate.ptrs_suh,
                self.multi_gate.ptrs_svh,
                self.multi_up.ptrs_trellis,
                self.multi_up.ptrs_suh,
                self.multi_up.ptrs_svh,
                self.multi_down.ptrs_trellis,
                self.multi_down.ptrs_suh,
                self.multi_down.ptrs_svh,
                buffers.gu_had[:2 * assignments],
                buffers.gu_out[:2 * assignments],
                buffers.down_out[:assignments],
                buffers.expert_offsets,
                buffers.inverse_order[:assignments],
                buffers.expert_chunks,
                buffers.chunk_count,
            )
            final_hidden_states = output.view(x.shape)

        # Per-expert K layer, prefill: picks grouped by K, one WMMA launch set per group (compacted picks,
        # the kernel has no masked picks), per-token sum in routing-slot order (moe_mixedk.MixedKStore.prefill)
        elif (
            self.mixed_k is not None and self.support_hip_wmma and bsz >= _HIP_WMMA_MIN_ROWS and
            y.dtype == torch.half and (y.is_contiguous() or
                                       (y.dim() == 2 and y.stride(1) == 1 and y.stride(0) >= y.size(1) and y.stride(0) % 4 == 0)) and
            os.environ.get("EXL3_MOE_WMMA", "1") != "0" and
            not params.get("activate_all_experts") and
            not params.get("reconstruct") and not params.get("autosplit_measure")
        ):
            sh_add = None
            if (_nc2_shared_fuse() and self.shared_experts and not self.shared_gate and
                    not self.shared_experts_post_norm and not self.tp_reduce and
                    self.routed_post_norm is None and self.latent_out is None):
                sh_params = {**params, "moe_valu": True} if _MOE_VALU else dict(params)
                sh_params["nc2_down_half"] = True
                sh_add = self.shared_experts.forward(x, sh_params)
                if sh_add.dtype == torch.half and sh_add.is_contiguous():
                    sh_add = sh_add.view(bsz, self.hidden_size)
                    bc_sh_exp = True    # shared expert summed in below: skip the tail add
                else:
                    raise RuntimeError(f"EXL3_NC2_SHARED_FUSE: shared expert output {sh_add.dtype}, contiguous "
                                       f"{sh_add.is_contiguous()}: fp16 contiguous needed")
            final_hidden_states = self.mixed_k.prefill(
                ext, y, selected_experts.contiguous(), routing_weights.float().contiguous(),
                lambda a, e: _hip_wmma_workspace(self.device, a, self.hidden_size, self.intermediate_size_padded, e),
                _scatter_expert_count if _moe_sync_free_count() else
                (lambda flat, bins: torch.bincount(flat, minlength = bins)),
                self.act_limit, self.hidden_size, self.intermediate_size_padded,
                sh_add = sh_add, chunk = _MOE_ALIGN_CHUNK,
            ).view(eshape)

        # Grouped WMMA prefill (gfx11.5): one launch per projection over all experts, trellis
        # decoded inside the GEMM; replaces the per-expert reconstruct + hgemm loop at prefill
        elif (
            self.mixed_k is None and self.support_hip_wmma and bsz >= _HIP_WMMA_MIN_ROWS and
            y.dtype == torch.half and (y.is_contiguous() or                 # or a row-padded producer view
                                       (y.dim() == 2 and y.stride(1) == 1 and y.stride(0) >= y.size(1) and y.stride(0) % 4 == 0)) and
            os.environ.get("EXL3_MOE_WMMA", "1") != "0" and
            not params.get("activate_all_experts") and
            not params.get("reconstruct") and not params.get("autosplit_measure")
        ):
            num_tokens, top_k = selected_experts.shape
            sel = selected_experts.contiguous()
            # fp32 routing weights: the kernel's weighted combine runs in fp32 end to end
            # (fp16 weights rounded here were half the KLD gap vs the per-expert path)
            rw = routing_weights.float().contiguous()
            flat = sel.view(-1)
            dummy = 0
            y_k, rw_k, sel_k = y, rw, sel
            if _MOE_ALIGN_CHUNK and num_tokens > _MOE_ALIGN_CHUNK:
                order, flat, dummy = _aligned_expert_layout(flat, num_tokens, top_k, self.num_experts, _MOE_ALIGN_CHUNK)
                # EXL3_MPW_SHORT_A (default 1, read per call, needs the ext flag): the gather reads the dummy rows as zeros,
                # so y is passed as is instead of copied whole to append them; same gathered bytes
                if not (hasattr(ext, "EXL3_MPW_SHORT_A") and os.environ.get("EXL3_MPW_SHORT_A", "1") != "0"):
                    y_k = torch.cat((y, torch.zeros((dummy, y.shape[1]), dtype = y.dtype, device = y.device)))
                rw_k = torch.cat((rw, torch.zeros((dummy, top_k), dtype = rw.dtype, device = rw.device)))
                sel_k = flat.view(num_tokens + dummy, top_k)
            else:
                order = flat.argsort(stable = True)
            if _moe_sync_free_count():
                expert_count = _scatter_expert_count(flat, self.num_experts + 1)
            else:
                expert_count = torch.bincount(flat, minlength = self.num_experts + 1)
            if _HIP_WMMA_STATS is not None:
                _HIP_WMMA_STATS.append((self.key, expert_count[:self.num_experts].cpu()))
            assignments = (num_tokens + dummy) * top_k
            ws = _hip_wmma_workspace(self.device, assignments, self.hidden_size,
                                     self.intermediate_size_padded, self.num_experts)
            output = torch.empty((num_tokens + dummy, self.hidden_size), dtype = torch.float, device = y.device)
            gu_had = ws["gu_had"][:2 * assignments * self.hidden_size].view(2 * assignments, self.hidden_size)
            # down_out aliases gu_had: the gate/up input is dead once the gate/up GEMM has run
            down_out = ws["gu_had"][:2 * assignments * self.hidden_size].view(torch.float).view(assignments, self.hidden_size)
            # EXL3_NC2_SHARED_FUSE=1 (default off, read per call): the shared expert runs first (it only
            # reads x) and returns its down GEMM's fp16 output; the reduce widens it (exact) and adds it in
            # fp32 in its epilogue = the arithmetic of the widen copy + `final_hidden_states += y` it
            # replaces; both kernels disappear. Bit-exact.
            sh_add = None
            if (_nc2_shared_fuse() and self.shared_experts and not self.shared_gate and
                    not self.shared_experts_post_norm and not self.tp_reduce and
                    self.routed_post_norm is None and self.latent_out is None):
                sh_params = {**params, "moe_valu": True} if _MOE_VALU else dict(params)
                sh_params["nc2_down_half"] = True
                sh_add = self.shared_experts.forward(x, sh_params)
                if sh_add.dtype == torch.half and sh_add.is_contiguous():
                    sh_add = sh_add.view(num_tokens, self.hidden_size)
                    bc_sh_exp = True    # shared expert summed in by the reduce: skip the tail add
                else:
                    raise RuntimeError(f"EXL3_NC2_SHARED_FUSE: shared expert output {sh_add.dtype}, contiguous "
                                       f"{sh_add.is_contiguous()}: fp16 contiguous needed")
            ext.exl3_moe_prefill_wmma(
                y_k, output, sel_k, rw_k, order, expert_count,
                self.multi_gate.ptrs_trellis, self.multi_gate.ptrs_suh, self.multi_gate.ptrs_svh,
                self.multi_up.ptrs_trellis, self.multi_up.ptrs_suh, self.multi_up.ptrs_svh,
                self.multi_down.ptrs_trellis, self.multi_down.ptrs_suh, self.multi_down.ptrs_svh,
                float(self.multi_gate.K), float(self.multi_down.K),
                gu_had,
                ws["gu_out"][:2 * assignments * self.intermediate_size_padded].view(2 * assignments, self.intermediate_size_padded),
                down_out,
                ws["offsets"], ws["inverse"][:assignments], ws["tiles"], ws["tile_count"],
                float(self.act_limit), bool(self.wmma_mcg), sh_add,
            )
            final_hidden_states = output[:num_tokens].view(eshape)

        # Torch/C++/fused path
        elif (
            (bsz >= self.f_threshold and not bszn_eligible) or not self.is_quantized or
            self.config.infer_params.no_reconstruct or
            not (self.support_quant_paths or bszn_eligible)
        ):
            # One spare row: the batched reconstruct tier's padding sink (never read back)
            fhs_ext = torch.zeros((y.shape[0] + 1, y.shape[1]), dtype = torch.float, device = y.device)
            final_hidden_states = fhs_ext[:y.shape[0]]

            # if self.routing_device is None or self.num_local_experts == self.num_experts:
            #     expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = self.num_local_experts)
            # else:
            #     selected_experts -= self.routing_first
            #     invalid = (selected_experts < 0) | (selected_experts >= self.num_local_experts)
            #     shifted = torch.where(invalid, torch.zeros_like(selected_experts), selected_experts + 1)
            #     expert_mask = F.one_hot(shifted, num_classes = self.num_local_experts + 1)[..., 1:]

            if self.num_local_experts is None or self.num_local_experts > 0:

                num_ex = self.num_local_experts or self.num_experts

                num_tokens, top_k = selected_experts.shape
                E = self.num_local_experts

                # Flatten assignments
                flat_expert_global = selected_experts.reshape(-1)               # [num_tokens * top_k]
                flat_weight = routing_weights.reshape(-1)                       # [num_tokens * top_k]

                # Token indices corresponding to each flattened assignment
                flat_token = buffered_interleaved_arange(num_tokens, top_k, device = y.device)

                # Map to local expert ids whenever this module holds a slice (TP shard or
                # CPU expert split), not only in the multi-device routing case
                if self.num_local_experts == self.num_experts:
                    flat_expert_local = flat_expert_global
                else:
                    flat_expert_local = flat_expert_global - self.routing_first
                    valid = (flat_expert_local >= 0) & (flat_expert_local < E)
                    flat_expert_local = torch.where(valid, flat_expert_local, torch.full_like(flat_expert_local, E))

                # Group once by local expert id (including sentinel for expert-P mode)
                order = flat_expert_local.argsort(stable = True)
                token_sorted = flat_token[order]
                weight_sorted = flat_weight[order]

                # Count how many assignments per expert. With few enough tokens no expert can exceed
                # the fused kernel's row capacity (an expert gets at most one row per token), so the
                # readback (a CPU sync per layer, ~33% idle at MTP verify shapes) is skipped and
                # everything is fused
                if _moe_sync_free_count():
                    expert_count = _scatter_expert_count(flat_expert_local, E + 1)
                else:
                    expert_count = torch.bincount(flat_expert_local, minlength = E + 1)
                if self.fused_mode_buffers is not None and \
                        (num_tokens if SYNC_FREE_TOKENS else num_tokens * top_k) <= self.fused_rows:
                    expert_count_list = None
                else:
                    expert_count_list = expert_count.tolist()

                moeg = expert_count_list is not None and FUSED_DET and self._moeg_ok()
                fused_cap = max(self.fused_rows, MOEG_ROWS) if moeg else self.fused_rows
                self._fused_cap = fused_cap

                # Tier plan: fused kernel for experts up to self.fused_rows rows, batched
                # reconstruct groups above that up to the tile cap, per-expert reconstruct beyond
                recon = None
                groups = []
                min_rows = 0
                if expert_count_list is None:
                    fused_total = num_tokens * top_k
                else:
                    fused_total = 0
                    if self.fused_mode_buffers is not None:
                        min_rows = fused_cap
                        fused_total = sum(c for c in expert_count_list[:num_ex] if 0 < c <= fused_cap)
                    recon = self._batch_recon_layer(y)
                    if recon is not None:
                        lim = max(min_rows, TEMP_ROWS_GRAPH)
                        heavy = [e for e in range(num_ex) if lim < expert_count_list[e] <= recon.max_rows]
                        if heavy:
                            from .moe_batch_recon import plan_groups
                            groups = plan_groups(heavy, lambda e: expert_count_list[e], recon.cap)

                # Deterministic accumulation (FUSED_DET): every assignment of the fused and
                # batched tiers gets a slot in one fp32 scratch (fused experts: count rows each,
                # weighted by the kernel; batched experts: their group's padded row count each,
                # unweighted) and one exl3_moe_gather per layer sums each token's slots in k
                # order, so identical inputs give identical outputs. The atomic alternative adds
                # contributions in arrival order. Slot tables come from the host-side counts, the
                # all-fused fast path builds them on the device. Scratch is prefill-shaped and per
                # call. Its traffic (written once by the GEMMs, read once by the gather) matches
                # what the atomic index_add_ path moved
                scratch = tables = inv_order = None
                n_grouped = n_slots = 0
                if FUSED_DET and (fused_total or groups):
                    A = flat_expert_local.shape[0]
                    inv_order = torch.empty_like(order).scatter_(
                        0, order, torch.arange(A, device = order.device))
                    if expert_count_list is None:
                        # Row num_ex of expert_count is the sentinel bucket for picks outside this
                        # module's expert slice (TP shard, CPU split). Its slots are never written,
                        # so the gather below is restricted to the first num_ex table rows
                        expert_start = torch.cumsum(expert_count, 0) - expert_count
                        tables = torch.stack([expert_start, expert_start, (expert_count > 0).long()])
                        n_slots = fused_total
                    else:
                        import numpy as np
                        E1 = len(expert_count_list)
                        base = np.zeros(E1, dtype = np.int64)
                        kind = np.zeros(E1, dtype = np.int64)
                        starts_np = np.cumsum([0] + expert_count_list[:-1]).astype(np.int64)
                        gbase = np.zeros(E1, dtype = np.int64)
                        n_slots = 0
                        n_grouped = 0
                        if self.fused_mode_buffers is not None:
                            for e in range(num_ex):
                                c = expert_count_list[e]
                                if 0 < c <= fused_cap:
                                    if moeg and c >= MOEG_LO:
                                        # Grouped tier: compact work rows, combined per token by the tier itself (kind 0 for the gather)
                                        gbase[e] = n_grouped; n_grouped += c
                                    else:
                                        base[e] = n_slots; kind[e] = 1; n_slots += c
                        self._fused_slots = n_slots
                        for grp in groups:
                            cmax = max(expert_count_list[e] for e in grp)
                            for b, e in enumerate(grp):
                                base[e] = n_slots + b * cmax; kind[e] = 2
                            n_slots += len(grp) * cmax
                        tables = torch.from_numpy(np.stack([base, starts_np, kind, gbase])).to(y.device, non_blocking = True)
                    scratch = torch.empty((max(n_slots, 1), y.shape[1]), dtype = torch.float, device = y.device)

                def run_fused(num_active, count_lo = 1, count_hi = self.fused_rows, m_tile = 16):
                    # Gateless: the up module stands in for the gate pointer tables (the kernel
                    # skips the gate GEMM when activation_fn_idx is MOE_ACT_RELU2_NOGATE)
                    multi_gate = self.multi_gate if self.gated else self.multi_up
                    ext.exl3_moe(
                        y,
                        final_hidden_states,
                        expert_count,
                        token_sorted,
                        weight_sorted,
                        self.fused_mode_buffers.temp_state_g,
                        self.fused_mode_buffers.temp_state_u,
                        self.fused_mode_buffers.temp_intermediate_g,
                        self.fused_mode_buffers.temp_intermediate_u,
                        self.activation_fn_idx,
                        multi_gate.K,
                        self.multi_up.K,
                        self.multi_down.K,
                        multi_gate.ptrs_trellis,
                        multi_gate.ptrs_suh,
                        multi_gate.ptrs_svh,
                        self.multi_up.ptrs_trellis,
                        self.multi_up.ptrs_suh,
                        self.multi_up.ptrs_svh,
                        self.multi_down.ptrs_trellis,
                        self.multi_down.ptrs_suh,
                        self.multi_down.ptrs_svh,
                        multi_gate.mcg,
                        multi_gate.mul1,
                        self.multi_up.mcg,
                        self.multi_up.mul1,
                        self.multi_down.mcg,
                        self.multi_down.mul1,
                        self.act_limit,
                        num_active,
                        scratch, tables[0] if tables is not None else None,
                        count_lo, count_hi, m_tile
                    )

                def run_grouped(count_lo, count_hi):
                    ext.exl3_moe_grouped(
                        y,
                        expert_count,
                        tables[1],
                        token_sorted,
                        weight_sorted,
                        self.activation_fn_idx,
                        self.multi_gate.K,
                        self.multi_up.K,
                        self.multi_down.K,
                        self.multi_gate.ptrs_trellis,
                        self.multi_gate.ptrs_suh,
                        self.multi_gate.ptrs_svh,
                        self.multi_up.ptrs_trellis,
                        self.multi_up.ptrs_suh,
                        self.multi_up.ptrs_svh,
                        self.multi_down.ptrs_trellis,
                        self.multi_down.ptrs_suh,
                        self.multi_down.ptrs_svh,
                        self.act_limit,
                        final_hidden_states,
                        flat_expert_local,
                        inv_order,
                        tables[3],
                        n_grouped,
                        count_lo, count_hi, 64,
                        self.fused_mode_buffers.temp_intermediate_g.shape[2]
                    )

                # num_active -1 = unknown (all fused), kernel launches at max concurrency
                if self.fused_mode_buffers is not None:
                    if expert_count_list is None:
                        run_fused(-1)
                    else:
                        counts = [c for c in expert_count_list[:num_ex] if 0 < c <= fused_cap]
                        t1 = sum(1 for c in counts if MTILE_T1 < c <= MTILE_T2)
                        t2 = sum(1 for c in counts if c > MTILE_T2)
                        if self.mtile_ok and (t1 or t2):
                            # One launch per row tile over its expert range, largest first
                            t0 = len(counts) - t1 - t2
                            if t2 and moeg and scratch is not None:
                                # Grouped GEMM tier for the experts above MOEG_LO rows, the persistent kernel keeps the rest
                                tg = sum(1 for c in counts if c >= MOEG_LO)
                                if tg:
                                    run_grouped(MOEG_LO, max(counts))
                                tm = t2 - tg
                                if tm:
                                    run_fused(tm, MTILE_T2 + 1, MOEG_LO - 1, 64)
                            elif t2:
                                run_fused(t2, MTILE_T2 + 1, fused_cap, 64)
                            if t1:
                                run_fused(t1, MTILE_T1 + 1, MTILE_T2, 32)
                            if t0:
                                run_fused(t0, 1, MTILE_T1, 16)
                        else:
                            run_fused(len(counts))

                # Batched reconstruct tier (into slots when deterministic, else accumulating)
                batched = ()
                if groups:
                    batched = self._run_batch_recon(
                        recon, y, fhs_ext, token_sorted, weight_sorted, expert_count_list, groups,
                        scratch, tables)

                # One fixed-order gather over every slot
                if scratch is not None and n_slots:
                    ext.exl3_moe_gather(final_hidden_states, scratch, flat_expert_local, inv_order,
                                        tables[1, :num_ex], tables[0, :num_ex], tables[2, :num_ex], weight_sorted)

                out_state = None
                interm = None
                interm_a = None
                max_count = 0
                start = 0

                # expert_count_list None: everything already handled by the fused kernel above
                rwh_w = self._rwh_true_weights(z) if (_RWH and "capture" in params and params.get("activate_all_experts")) else None
                for expert_idx in range(num_ex if expert_count_list is not None else 0):
                    count = expert_count_list[expert_idx]
                    end = start + count
                    if count <= min_rows or expert_idx in batched or expert_idx in self.pruned:
                        start = end
                        continue

                    if self.tier is not None and not params.get("autosplit_measure"):
                        raise RuntimeError(f"{self.key}: tiered experts do not support the per-expert graph/DQ/torch path "
                                           f"({count} rows on expert {expert_idx}); use the fused/grouped/WMMA paths")
                    top_x = token_sorted[start:end]
                    w = weight_sorted[start:end].unsqueeze(1)
                    if rwh_w is not None:
                        w_e = rwh_w[top_x, expert_idx]
                        sel = (w_e > 0).nonzero().squeeze(1)
                        params["_cap_rw"] = (sel, w_e[sel])

                    current_state = y.index_select(0, top_x)

                    if self.bc is not None and self.support_quant_paths:
                        # Graph path
                        if count <= TEMP_ROWS_GRAPH:
                            self.bc.run_single_expert(current_state, expert_idx)
                            current_state = self.experts_cfg.out_d2[:count]

                        # DQ path
                        else:
                            if count > max_count:
                                out_state = torch.empty((count, self.expert_size), dtype = torch.float, device = self.device)
                                interm = torch.empty((count * 2, self.intermediate_size_padded), dtype = self.interm_dtype, device = self.device)
                                interm_a = interm[:count] if self.interm_dtype == torch.half else \
                                    torch.empty_like(interm[:count], dtype = torch.half)
                                out_state_ = out_state
                                interm_ = interm
                                interm_a_ = interm_a
                                max_count = count
                            elif count == max_count:
                                out_state_ = out_state
                                interm_ = interm
                                interm_a_ = interm_a
                            else:
                                out_state_ = out_state[:count]
                                interm_ = interm[:count * 2]
                                interm_a_ = interm_a[:count]

                            yh = torch.empty((count * 2, self.expert_size), dtype = torch.half, device = self.device)
                            self.bc.run_single_expert_dq(current_state, expert_idx, yh, interm_, interm_a_, out_state)
                            current_state = out_state_
                    else:

                        # Torch path
                        def mlp(exp_i, xc):
                            u = self.ups[exp_i].forward(xc, params)
                            if self.gated:
                                g = self.gates[exp_i].forward(xc, params)
                                a = u if self.interm_dtype == torch.half else torch.empty_like(u, dtype = torch.half)
                                self.activation_fn_call(g, u, a, self.act_limit)
                            else:
                                a = self.gateless_act(u)
                                if a.dtype != torch.half:
                                    a = a.half()
                            return self.downs[exp_i].forward(a, params)

                        current_state = mlp(expert_idx, current_state)

                    params.pop("_cap_rw", None)
                    current_state.mul_(w)
                    final_hidden_states.index_add_(0, top_x, current_state)
                    start = end

            final_hidden_states = final_hidden_states.reshape(eshape)

        # Fused decode kernels (bsz 1..MAX_BSZN): two launches run every (token, expert) slot
        # through the expert MLP (no sort/dedup -- overlap between tokens this small is rare and
        # not worth the argsort/bincount host-sync cost that the fused/exl3_moe path pays). Shared
        # experts (if present) run through their own multi-row BC_GatedMLP graph and are merged
        # inside the kernel. Expert-range shards (CPU split, TP) produce a partial sum here:
        # out-of-range picks are masked inside the kernel and contribute exact zeros. Every
        # quantized configuration that reaches this point has self.bc (it is built whenever the
        # quantized paths apply), so this is the last tier
        else:
            assert bszn_eligible
            # Shared experts: merged in the kernel from their 8-row graph, or above MAX_BSZN rows by the regular tail below
            # (their R-row kernels; measured 1.2-2 ms/step faster at 10-32 rows than 8-row slices of the graph)
            sh_in_bc = self.bc_sh_exp and bsz <= MAX_BSZN
            self.bc.run_bszN(y, selected_experts, routing_weights, sh_in_bc)
            final_hidden_states = self.experts_cfg.out_bszn[:bsz].view(eshape)
            bc_sh_exp = sh_in_bc

        # CPU tail partial folds in before the post norms (nonlinear: they must see the
        # complete routed sum)
        final_hidden_states = self.cpu_split_combine(final_hidden_states, cpu_partial, cpu_pending, eshape)

        # Latent MoE: project the routed sum back to the residual width (linear, so under TP
        # each rank's partial sum projects on its own ahead of the reduction)
        if self.latent_out is not None:
            final_hidden_states = self.latent_out.forward(final_hidden_states.to(torch.half), params)

        # The post norms are nonlinear, so under TP their inputs must be complete sums, not
        # per-rank partials: reduce the routed and shared contributions separately before the
        # norms (Gemma4 MoE), after which every rank holds identical complete tensors and the
        # final reduction is skipped
        pre_norm_reduce = self.tp_reduce and (
            self.routed_post_norm is not None or
            (self.shared_experts is not None and self.shared_experts_post_norm is not None and not bc_sh_exp)
        )
        if pre_norm_reduce:
            params["backend"].all_reduce(
                final_hidden_states,
                self.intermediate_size > 0 and self.num_local_experts > 0
            )

        # Extra norm (Gemma4)
        if self.routed_post_norm:
            final_hidden_states = self.routed_post_norm.forward(final_hidden_states, params)

        # Shared experts
        # EXL3_SHGATE_FUSE (default 1, read per call): prefill rows with a shared gate take the shared expert's fp16 down output
        # and add float(y) * sigmoid(gate) into the routed sum in one pass (gr_pf.shgate_add), replacing the fp32 widen copy, the
        # torch multiply and the torch add of add_sigmoid_gate's fallback. Same arithmetic, bit-identical. 0 goes back.
        if (self.shared_experts and not bc_sh_exp and self.shared_gate and bsz > 32 and not self.tp_reduce
                and not self.shared_experts_post_norm and final_hidden_states.dtype == torch.float
                and final_hidden_states.is_contiguous() and os.environ.get("EXL3_SHGATE_FUSE", "1") != "0"):
            sh_params = {**params, "moe_valu": True} if _MOE_VALU else dict(params)
            sh_params["nc2_down_half"] = True
            yh = self.shared_experts.forward(x, sh_params)
            z = self.shared_gate.forward(x, params)
            if yh.dtype == torch.half and yh.is_contiguous() and z.dtype == torch.float and z.is_contiguous():
                from .gated_delta_net_fn.gr_pf import shgate_add
                shgate_add(final_hidden_states.view(-1, self.hidden_size), yh.view(-1, self.hidden_size), torch.sigmoid(z))
            else:
                ext.add_sigmoid_gate(yh.float(), z.float(), final_hidden_states)
        elif self.shared_experts and not bc_sh_exp:
            if side_y is not None:
                cur = torch.cuda.current_stream(final_hidden_states.device)
                cur.wait_event(_side(final_hidden_states.device)[2])
                side_y.record_stream(cur)
                y = side_y
            else:
                y = self.shared_experts.forward(x, {**params, "moe_valu": True} if _MOE_VALU else params)
            if pre_norm_reduce:
                params["backend"].all_reduce(y, True)
            if self.shared_experts_post_norm:
                y = self.shared_experts_post_norm.forward(y, params)
            if self.shared_gate:
                if bsz > 32:
                    z = self.shared_gate.forward(x, params)
                    ext.add_sigmoid_gate(y, z, final_hidden_states)
                else:
                    ext.add_sigmoid_gate_proj(y, x, final_hidden_states, self.shared_gate.inner.weight)
            else:
                final_hidden_states += y

        # Output reduction
        if self.tp_reduce and not pre_norm_reduce:
            params["backend"].all_reduce(
                final_hidden_states,
                (self.intermediate_size > 0 and self.num_local_experts > 0) or bool(self.shared_experts)
            )

        if out_dtype is not None:
            final_hidden_states = final_hidden_states.to(out_dtype)
        return final_hidden_states


    def _rwh_true_weights(self, z: torch.Tensor) -> torch.Tensor:
        """[rows, E] float: the model's actual top-k routing weights (sigmoid scores renormalised over the picks, no
        scaling), zero for unselected experts. Used to weight the per-expert calibration Hessians (EXL3_RWH)."""
        cfg = self.routing_cfg
        zz = z.view(-1, z.shape[-1])
        logits = torch.matmul(zz, cfg.gate_tensor)[:, :self.num_experts].float()
        scores = torch.sigmoid(logits)
        top = torch.topk(scores, self.num_experts_per_tok, dim = -1)
        wv = top.values / (top.values.sum(dim = -1, keepdim = True) + 1e-20)
        out = torch.zeros_like(scores)
        out.scatter_(1, top.indices, wv)
        return out


    def _row_chunk_ok(self) -> bool:
        # bszN decode kernels exist (BC, CUDA) and no fused MoE kernel would take the rows (or only the 3INST one, which
        # loses below _FUSED_3INST_MIN rows); ROCm has no BC object
        return (self.bc is not None and (not self.support_fused or self.fused_3inst) and self.support_bc_bszn and not self.alt_residual_channel and
                self.cpu_split_first is None and not self.cpu_offload and self.routing_device is None and self.tp_mode is None)


    def _forward_row_chunks(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None) -> torch.Tensor:
        # Each slice gets its own params: no fused residual add (the caller adds y) and its rows of a pre-computed routing
        y = x.reshape(-1, x.shape[-1])
        routed = params.pop("dec_routed", None)
        outs = []
        for a in range(0, y.shape[0], self.coop_rows):
            sub = {k: v for k, v in params.items() if k != "mlp_residual_out"}
            if routed is not None:
                sub["dec_routed"] = (routed[0][a:a + self.coop_rows], routed[1][a:a + self.coop_rows])
            o = self.forward(y[a:a + self.coop_rows].unsqueeze(0), sub, out_dtype)
            outs.append(o.reshape(-1, o.shape[-1]).clone())   # the bszN output buffer is reused by the next slice
        return torch.cat(outs).view(x.shape[:-1] + (outs[0].shape[-1],))


    def dec_norm_route(self, norm, xa: torch.Tensor, r: torch.Tensor, params: dict, rows: int = 1):
        """Batch-1 decode: MLP pre-norm (r += xa, y = norm(r)) fused with the router in one
        exl3_dec_router_norm launch. Returns y (fp16), routing left in params["dec_routed"]; None
        when the fused path does not apply (the caller then runs the norm itself)."""
        from .block_sparse_mlp_routing import _dec_router_ok
        if not (
            self.support_dec_moe and hasattr(ext, "exl3_dec_router_norm") and
            os.environ.get("EXL3_DEC_ROUTER_NORM", "1") != "0" and os.environ.get("EXL3_DEC_ROUTER", "1") != "0" and
            self.routing_fn is routing_dots and self.routing_gate is not None and
            self.router_pre_norm is None and self.routed_pre_norm is None and self.latent_in is None and
            self.routing_device is None and not self.alt_residual_channel and self.cpu_split_first is None and
            xa.numel() == rows * self.hidden_size and r.numel() == rows * self.hidden_size and
            xa.dtype in (torch.half, torch.float) and r.dtype == torch.float and
            xa.is_contiguous() and r.is_contiguous() and
            not norm.span_heads and norm.groups == 1 and
            (norm.weight is None or norm.weight.dtype in (torch.half, torch.bfloat16)) and
            norm.key not in params.get("export_state_norm_keys", ())
        ):
            return None
        buf = "dec_norm_y" if rows == 1 else "dec_norm_y_rows"
        y = getattr(self, buf, None)
        if y is None or y.device != xa.device or y.shape[0] != rows:
            y = torch.empty((rows, self.hidden_size), dtype = torch.half, device = xa.device)
            setattr(self, buf, y)
        cfg = self.routing_cfg
        if not _dec_router_ok(1, cfg, y[:1], params):
            return None
        from .quant.exl3 import dec_workspace
        bias = getattr(cfg, "dec_bias", False)
        if bias is False:
            esb = cfg.e_score_correction_bias
            bias = esb.float().contiguous() if esb is not None else None
            cfg.dec_bias = bias
        scratch, counters = dec_workspace(xa.device)
        scale = cfg.routed_scaling_factor if cfg.routed_scaling_factor is not None else 1.0
        if rows == 1:
            sel, wts = cfg.selected_experts_bsz1, cfg.routing_weights_bsz1
        else:
            # One launch for all rows (grid.y = row): each row is bitwise the single-row launch
            sel = getattr(self, "dec_sel_rows", None)
            if sel is None or sel.shape[0] != rows or sel.device != xa.device:
                sel = self.dec_sel_rows = torch.empty((rows, self.num_experts_per_tok), dtype = torch.long, device = xa.device)
                self.dec_wts_rows = torch.empty((rows, self.num_experts_per_tok), dtype = torch.half, device = xa.device)
            wts = self.dec_wts_rows
        ext.exl3_dec_router_norm(xa.view(rows, -1), r.view(rows, -1), norm.weight, norm.rms_norm_eps, norm.constant_bias,
                                 norm.constant_scale, y, cfg.gate_tensor, bias, sel,
                                 wts, scratch, counters, float(scale))
        params["dec_routed"] = (sel, wts)
        return y.view(xa.shape)


    def dec_norm_route_r(self, norm, xa: torch.Tensor, r: torch.Tensor, params: dict):
        """DFlash verify (R>1): R sequential dec_norm_route calls, one per row. Each call is the
        literal batch-1 exl3_dec_router_norm launch (same kernel, same args, just row j's data),
        so row j of the returned y and of dec_routed's (selected, weights) is bit-identical to
        what plain batch-1 decode would produce stepping through row j alone -- this is what
        an earlier row-invariance probe missed: it exercised
        exl3_dec_router (unfused), not the exl3_dec_router_norm/dec_norm_route path real decode
        actually takes, which is why it saw a k-order mismatch and (for one cell) a 1.22e-4
        weight mismatch -- neither is possible here, since this never runs a second, independent
        router: it replays the one real kernel R times. Falls back to None (generic norm+router)
        if any row can't take the fused path, or if xa/r aren't [R, H] contiguous views sharing
        storage with the real residual (the fused kernel updates r in place; a copy would silently
        break the residual chain for the rows after it, so this refuses instead of copying)."""
        R = xa.shape[0]
        if R == 1:
            # Identical to calling dec_norm_route directly -- no clone, no extra allocation, so
            # R==1 (all of batch-1 decode, flag or no flag) pays nothing for this wrapper existing.
            # dec_norm_route's raw-buffer return is consumed immediately by the caller either way.
            return self.dec_norm_route(norm, xa[0], r[0], params)
        # R>1 (DFlash verify): gated by the same flag as the union MoE kernel it feeds, so
        # EXL3_DEC_MOE_UNION=0 reproduces the exact earlier behavior (generic norm+router,
        # dedup MoE) for a clean A/B against it -- R sequential kernel launches here is a real
        # per-layer cost even though it's bit-exact, not a free win to force on unconditionally.
        # Also refuse above MAX_BSZN: prefill calls this with R = chunk size (hundreds+), where
        # R sequential single-row launches would be catastrophically slower than the batched
        # prefill path this falls back to -- this is not what the flag or the kernel is for.
        if os.environ.get("EXL3_DEC_MOE_UNION", "0") == "0" or R > UNION_ROWS or not params.get("dflash_verify"):
            return None
        try:
            xa.view(R, -1)
            r.view(R, -1)
        except RuntimeError:
            return None
        if os.environ.get("EXL3_DEC_NORM_ROUTE_ROWS", "1") != "0":
            # One launch (grid.y = row), bitwise the R single-row launches below. None when the fused
            # path does not apply: the caller then runs the generic norm + router.
            return self.dec_norm_route(norm, xa.view(R, -1), r.view(R, -1), params, R)
        ys, sels, wtss = [], [], []
        for i in range(R):
            sub = dict(params)
            sub.pop("dec_routed", None)
            y_i = self.dec_norm_route(norm, xa[i], r[i], sub)
            if y_i is None:
                return None
            sel_i, wts_i = sub["dec_routed"]
            # y_i is a view into self.dec_norm_y, a buffer dec_norm_route reuses every call
            # (like cfg.selected_experts_bsz1/routing_weights_bsz1 below) -- clone now, before
            # the next row's call overwrites it.
            ys.append(y_i.reshape(1, -1).clone())
            sels.append(sel_i.clone())
            wtss.append(wts_i.clone())
        params["dec_routed"] = (torch.cat(sels, dim = 0), torch.cat(wtss, dim = 0))
        return torch.cat(ys, dim = 0)


    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.e_score_correction_bias is not None:
            t[f"{self.key}.{self.e_score_correction_bias_key}"] = self.e_score_correction_bias.contiguous()
        if self.e_score_bias_vl is not None:
            t[f"{self.key}.{self.e_score_bias_vl_key}"] = self.e_score_bias_vl.contiguous()
        if self.tid2eid is not None:
            t[f"{self.key}.{self.tid2eid_key}"] = self.tid2eid.contiguous()
        if self.per_expert_scale is not None:
            t[f"{self.key}.{self.per_expert_scale_key}"] = self.per_expert_scale.contiguous()
        return t


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        storage = 0
        storage += self.routing_gate.storage_size()
        if self.shared_gate:
            storage += self.shared_gate.storage_size()
        for g in self.gates: storage += g.storage_size()
        for u in self.ups: storage += u.storage_size()
        for d in self.downs: storage += d.storage_size()
        # The latent projections are replicated on every rank
        storage_d = 0
        if self.latent_in is not None:
            storage_d += self.latent_in.storage_size() + self.latent_out.storage_size()
        # TODO: More precise overhead estimate accounting for gate etc.
        overhead_d = self.hidden_size * torch.float.itemsize
        overhead_s = 4 * self.intermediate_size * (self.interm_dtype or torch.half).itemsize
        if self.interm_dtype != torch.half:
            overhead_s += self.intermediate_size * torch.half.itemsize
        recons = max(
            self.gates[0].recons_size() if self.gated else 0,
            self.ups[0].recons_size(),
            self.downs[0].recons_size()
        )
        use_tp_split = options.get("moe_tensor_split", False)
        tpa = TPAllocation(
            key = self.key,
            channel_width = 128 if use_tp_split else 1,
            channel_unit = "channels" if use_tp_split else "experts",
            storage_per_device = storage_d,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = self.ups[0].out_features // 128 if use_tp_split else self.num_experts,
            limit_key = "moe"
        )
        tpa_list = [tpa]
        if self.shared_experts:
            tpa_list += self.shared_experts.make_tp_allocation(options)
        return tpa_list


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": BlockSparseMLP,
            "kwargs": {
                "key": self.key,
                "hidden_size": self.hidden_size,
                "latent_size": self.latent_size,
                "intermediate_size": self.intermediate_size,
                "activation_fn": self.activation_fn,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "interm_dtype": self.interm_dtype,
                "fused_fp16_interm": self.fused_fp16_interm,
                "router_type": self.router_type,
                "routed_scaling_factor": self.routed_scaling_factor,
                "n_group": self.n_group,
                "topk_group": self.topk_group,
                "act_limit": self.act_limit,
                "alt_residual_channel": self.alt_residual_channel,
                "key_tid2eid": self.tid2eid_key,
                "key_e_score_bias_vl": self.e_score_bias_vl_key,
            },
            # Hash-MoE bootstrap layers (DeepSeek-V4): frozen token->experts table, needed
            # wherever routing runs (the output device, like the routing gate)
            "tid2eid": producer.send(self.tid2eid) if self.tid2eid is not None else None,
            "routing_gate": _export(self.routing_gate),
            "shared_gate": _export(self.shared_gate),
            "latent_in": _export(self.latent_in),
            "latent_out": _export(self.latent_out),
            "e_score_correction_bias": producer.send(self.e_score_correction_bias),
            "e_score_bias_vl": producer.send(self.e_score_bias_vl) if self.e_score_bias_vl is not None else None,
            "per_expert_scale": producer.send(self.per_expert_scale),
            "gates": [_export(self.gates[i]) for i in range(self.num_experts)] if self.gated else None,
            "ups": [_export(self.ups[i]) for i in range(self.num_experts)],
            "downs": [_export(self.downs[i]) for i in range(self.num_experts)],
            "shared_experts": self.shared_experts.tp_export(plan, producer) \
                if self.shared_experts is not None else None,
            "shared_experts_post_norm": _export(self.shared_experts_post_norm),
            "router_pre_norm": _export(self.router_pre_norm),
            "routed_pre_norm": _export(self.routed_pre_norm),
            "routed_post_norm": _export(self.routed_post_norm),
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        consumer = local_context["consumer"]
        key = exported["kwargs"]["key"]
        device = local_context["device"]
        output_device = local_context["output_device"]
        first, last, unit = plan[key]

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        def _import_no_reduce(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan, skip_reduction = True) \
                if exported.get(name) else None

        def _import_i(name, i):
            nonlocal exported, plan
            return exported[name][i]["cls"].tp_import(local_context, exported[name][i], plan) \
                if exported.get(name) else None

        def _import_i_split(name, i, split):
            nonlocal exported, plan
            return exported[name][i]["cls"].tp_import_split(local_context, exported[name][i], plan, split) \
                if exported.get(name) else None

        # Gateless experts (NemotronH) export gates as None; the local module gets an empty
        # gates list so the ctor derives gated = False
        gated = exported.get("gates") is not None

        # Tensor parallel
        if unit == "channels":
            num_local_experts = exported["kwargs"]["num_experts"]
            gu_split = (True, first, last)
            d_split = (False, first, last)
            exported["kwargs"]["intermediate_size"] = last - first
            gates = [_import_i_split("gates", i, gu_split) for i in range(num_local_experts)] if gated else []
            ups = [_import_i_split("ups", i, gu_split) for i in range(num_local_experts)]
            downs = [_import_i_split("downs", i, d_split) for i in range(num_local_experts)]
            routing_first = 0
            routing_last = num_local_experts

        # Expert parallel
        elif unit == "experts":
            num_local_experts = last - first
            gates = [_import_i("gates", i) for i in range(first, last)] if gated else []
            ups = [_import_i("ups", i) for i in range(first, last)]
            downs = [_import_i("downs", i) for i in range(first, last)]
            routing_first = first
            routing_last = last

        else:
            assert False

        module = BlockSparseMLP(
            config = None,
            **exported["kwargs"],
            num_local_experts = num_local_experts,
            gates = gates,
            ups = ups,
            downs = downs,
            shared_experts = _import_no_reduce("shared_experts"),
            shared_gate = _import("shared_gate"),
            latent_in = _import("latent_in"),
            latent_out = _import("latent_out"),
            routing_gate = _import("routing_gate") if device == output_device else None,
            routing_first = routing_first,
            routing_last = routing_last,
            routing_device = output_device,
            shared_experts_post_norm = _import("shared_experts_post_norm"),
            router_pre_norm = _import("router_pre_norm"),
            routed_pre_norm = _import("routed_pre_norm"),
            routed_post_norm = _import("routed_post_norm"),
        )

        module.device = device
        module.tp_mode = unit
        module.e_score_correction_bias = consumer.recv(exported["e_score_correction_bias"], cuda = True)
        if exported.get("e_score_bias_vl") is not None:
            module.e_score_bias_vl = consumer.recv(exported["e_score_bias_vl"], cuda = True)
        module.per_expert_scale = consumer.recv(exported["per_expert_scale"], cuda = True)
        if exported.get("tid2eid") is not None and device == output_device:
            module.tid2eid = consumer.recv(exported["tid2eid"], cuda = True)
        if unit == "channels" or num_local_experts > 0:
            module.load_local()
        if module.routing_gate is not None:
            module.load_routing()
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True
        return module
