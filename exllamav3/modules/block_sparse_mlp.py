from __future__ import annotations
from typing_extensions import override
import os
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.tensor import to2
from . import Module, Linear
from .multilinear import MultiLinear
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
# Deterministic (slot + gather) accumulation for the fused kernel's outputs; EXL3_MOE_FUSED_DET=0
# restores the atomic adds
FUSED_DET = os.environ.get("EXL3_MOE_FUSED_DET", "1") != "0"
MAX_BSZN = 8  # must match MAX_BSZN in exllamav3_ext/libtorch/blocksparse_mlp.h


# Expert-deduplicated grouped MoE (verify batches decode each picked expert once for all of its
# rows); EXL3_MOE_DEDUP=0 restores the per-assignment kernel
_MOE_DEDUP = os.environ.get("EXL3_MOE_DEDUP", "1") != "0"

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


@dataclass
class FusedBuffers:
    temp_state_g: torch.Tensor
    temp_state_u: torch.Tensor
    temp_intermediate_g: torch.Tensor
    temp_intermediate_u: torch.Tensor


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
            for idx in range(self.num_local_experts):

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
        self.support_quant_paths = False
        self.multi_gate = None
        self.multi_up = None
        self.multi_down = None

        self.routing_cfg = None
        self.experts_cfg = None

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

        # The quantized fast paths (mgemm/BC/fused kernels) don't yet support per-expert biases,
        # activations other than silu/gelu (or gateless relu2), or trimmed (padded) down
        # projections; configurations with any of those run every batch size through the dense
        # per-expert path, which handles all of them (gpt-oss)
        has_mgemm = hasattr(ext, "exl3_mgemm")
        self.support_quant_paths = (
            has_mgemm and self.is_quantized and
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
            has_mgemm and self.is_quantized and
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
            # are width/128 and width/32), top-k 8 (MiMo-V2) or 10 (Qwen3.8). The old
            # 2560 / 640-768 / top-10 gate was the Qwen3.8-Flash-Next pack only.
            self.hidden_size % 128 == 0 and self.intermediate_size_padded % 128 == 0 and
            self.num_experts_per_tok in (8, 10) and self.interm_dtype == torch.half and
            self.act_limit == 0 and self.tp_mode is None and not self.hip_grouped_lora_blocked and
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
            self.is_quantized and self.gated and self.activation_fn == "silu" and
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
        # widths, any top-k, K 2 / 2.5 / 3 / 4, either codebook, and a clamped SwiGLU limit
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
        if (self.support_quant_paths or self.support_bc_bszn or self.support_hip_grouped or
            self.support_dec_moe or self.support_hip_wmma) and not self.config.infer_params.no_reconstruct:
            self.multi_gate = MultiLinear(self.device, self.gates, allow_bias = True) if self.gated else None
            self.multi_up = MultiLinear(self.device, self.ups, allow_bias = True)
            self.multi_down = MultiLinear(self.device, self.downs, allow_bias = True)

            # Enable fully fused kernel if possible (uniform mcg or mul1 codebook across gate/up/down,
            # and an activation the fused kernel implements)
            cbs = (
                self.multi_gate.q_cb() if self.gated else self.multi_up.q_cb(),
                self.multi_up.q_cb(),
                self.multi_down.q_cb(),
            )
            self.support_fused = (
                hasattr(ext, "exl3_moe") and hasattr(ext, "exl3_moe_max_concurrency") and
                cbs[0] == cbs[1] == cbs[2] and cbs[0] in ((True, False), (False, True)) and
                self.support_quant_paths and
                (self.fused_fp16_interm or self.interm_dtype in (None, torch.half))
            )
            # One K for the whole layer: the grouped GEMV is templated on the experts' trellis
            # bitrate and gate/up/down share it (the allocator assigns per layer).
            ks = {multi.K for multi in (self.multi_gate, self.multi_up, self.multi_down)}
            self.support_hip_grouped = self.support_hip_grouped and len(ks) == 1 and all(
                multi.K in (2, 2.5, 3, 4) and multi.mul1 and not multi.mcg
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
            # mul1 and mcg both decode (REPORT-21); anything else leaves the fused path off.
            if cbs[0] == cbs[1] == cbs[2] and cbs[0] in ((True, False), (False, True)):
                self.dec_moe_mcg = bool(cbs[0][0])
            self.support_dec_moe = self.support_dec_moe and self.dec_moe_mcg is not None
            if self.support_dec_moe:
                self.dec_moe_out = g_tensor_cache.get(self.device, (1, self.hidden_size), torch.float, "dec_moe_out")
                self.dec_moe_act = g_tensor_cache.get(
                    self.device, (self.num_experts_per_tok, self.intermediate_size_padded), torch.half, "dec_moe_act")
                self.dec_shf_setup()
                # R-row union MoE (REPORT-16/17, exl3_dec_moe_union): shared, per-device
                # workspace (like dec_workspace) sized for the worst case (MAX_BSZN rows, this
                # layer's topk/H/I) and reused across layers -- forward calls are stream-ordered
                # so a later layer never races an earlier one's still-in-flight kernel.
                numex = self.num_experts_per_tok
                Ipad = self.intermediate_size_padded
                Ru = MAX_BSZN
                self.dec_moe_union_out = g_tensor_cache.get(
                    self.device, (Ru, self.hidden_size), torch.float, "dec_moe_union_out")
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
                multi.K in (2, 2.5, 3, 4) and (multi.mul1 != multi.mcg)
                for multi in (self.multi_gate, self.multi_up, self.multi_down)
            ) and self.multi_gate.K == self.multi_up.K
            # Codebook: mpw_gemm is templated on it and takes one flag for all three projections
            # (REPORT-22), so the layer needs a uniform codebook. Unknown (neither mul1 nor mcg,
            # or mixed) leaves the path off.
            wmma_cbs = {(m.mul1, m.mcg) for m in (self.multi_gate, self.multi_up, self.multi_down)}
            if len(wmma_cbs) == 1:
                wmma_mul1, wmma_mcg = wmma_cbs.pop()
                if wmma_mul1 != wmma_mcg:
                    self.wmma_mcg = wmma_mcg
            self.support_hip_wmma = self.support_hip_wmma and self.wmma_mcg is not None

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
        bszn_rows = MAX_BSZN * numex

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
        coop_ctr = g_tensor_cache.get(device, (bszn_rows * (I // 128) + MAX_BSZN * (Ho // 128) + 2 * bszn_rows + 3,), torch.int, "moe1_coop_ctr")
        coop_ctr.zero_()
        # Rotated up-projection input per slot for the fused kernels at bsz > 1 (yh holds the gate's)
        had_u = g_tensor_cache.get(device, (bszn_rows, Hi), torch.half, "moe1_temp_hidden_u")

        # Expert interval for split module (-1, -1) indicate no split
        mine, maxe = self.routing_first, self.routing_last
        if mine is None or maxe - mine == self.num_experts:
            mine, maxe = -1, -1

        # Routed sum of the fused decode path, one exact-width row per token (the kernel trims the
        # padded down width on the way out)
        out_bszn = g_tensor_cache.get(device, (MAX_BSZN, H), torch.float, "moe1_out_bszn")

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
                sh_exp_t = torch.empty((1, MAX_BSZN, self.hidden_size), dtype = torch.float, device = self.device)
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
        super().load(device, **kwargs)

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
        if self.cpu_offload or self.multi_up is None:
            return 0, 0
        h = self.expert_size
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
            slot = sum(c for c in expert_count_list[:self.num_local_experts or self.num_experts]
                       if 0 < c <= self.fused_rows) if self.fused_mode_buffers is not None else 0
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

    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.alt_residual_channel:
            y = params["residual"].view(-1, self.hidden_size)
        else:
            y = x.view(-1, self.hidden_size)
        bsz = y.shape[0]
        if os.environ.get("BSZ_DEBUG"):
            print(f"[bsz_debug] layer={self.key} bsz={bsz} union={os.environ.get('EXL3_DEC_MOE_UNION')}")
        bc_sh_exp = False
        # REPORT-28: R-row union MoE with the unique-expert table built on device (no host sync),
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
        bszn_eligible = self.bc is not None and bsz <= MAX_BSZN

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

        # Extra norm (Gemma4)
        if self.routed_pre_norm:
            y = self.routed_pre_norm.forward(y, params, out_dtype = torch.half)

        # Latent MoE: the experts (GPU and CPU-resident alike) see the projected input
        if self.latent_in is not None:
            y = self.latent_in.forward(y, params)

        # Broadcast routing indices and weights
        if self.routing_device is not None:
            params["backend"].broadcast(selected_experts, src_device = self.routing_device)
            params["backend"].broadcast(routing_weights, src_device = self.routing_device)

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

        # R-row union decode (REPORT-16/17, exl3_dec_moe_union), DFlash verify: bit-exact vs
        # bsz independent batch-1 exl3_dec_moe calls under shared routing (REPORT-16 Sec 1) --
        # each unique expert across the bsz rows' picks is decoded once and applied to every row
        # that picked it in the exact batch-1 per-row arithmetic order. Opt-in (default off):
        # REPORT-16 measured no clear speed win yet at R=1 or R=6-8 (host-side unique/assign
        # build dominates), only R=2..5; EXL3_DEC_MOE_UNION=1 to use it anyway (e.g. to get a
        # bit-exact verify path even before that's fixed).
        elif (
            1 < bsz <= MAX_BSZN and self.support_dec_moe and self.tp_mode is None and
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
            ext.exl3_dec_moe_union(
                y, out_buf, selected_experts.contiguous(), routing_weights.contiguous(),
                self.multi_gate.ptrs_trellis, self.multi_gate.ptrs_suh, self.multi_gate.ptrs_svh,
                self.multi_up.ptrs_trellis, self.multi_up.ptrs_suh, self.multi_up.ptrs_svh,
                self.multi_down.ptrs_trellis, self.multi_down.ptrs_suh, self.multi_down.ptrs_svh,
                self.dec_moe_union_act[:bsz * numex],
                self.dec_moe_union_down_part[:bsz * numex],
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

        # gfx12 grouped decode/verification path: selected IDs and weights stay on-device,
        # and every row-major assignment slot remains distinct.
        elif (
            _hip_grouped_rows_eligible(bsz, self.support_hip_prefill) and self.support_hip_grouped and
            self.tp_mode is None and self.act_limit == 0 and
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

        # Grouped WMMA prefill (gfx11.5): one launch per projection over all experts, trellis
        # decoded inside the GEMM; replaces the per-expert reconstruct + hgemm loop at prefill
        elif (
            self.support_hip_wmma and bsz >= _HIP_WMMA_MIN_ROWS and
            y.dtype == torch.half and y.is_contiguous() and
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
            order = flat.argsort(stable = True)
            if _moe_sync_free_count():
                expert_count = _scatter_expert_count(flat, self.num_experts + 1)
            else:
                expert_count = torch.bincount(flat, minlength = self.num_experts + 1)
            if _HIP_WMMA_STATS is not None:
                _HIP_WMMA_STATS.append((self.key, expert_count[:self.num_experts].cpu()))
            ws = _hip_wmma_workspace(self.device, num_tokens * top_k, self.hidden_size,
                                     self.intermediate_size_padded, self.num_experts)
            assignments = num_tokens * top_k
            output = torch.empty((num_tokens, self.hidden_size), dtype = torch.float, device = y.device)
            gu_had = ws["gu_had"][:2 * assignments * self.hidden_size].view(2 * assignments, self.hidden_size)
            # down_out aliases gu_had: the gate/up input is dead once the gate/up GEMM has run
            down_out = ws["gu_had"][:2 * assignments * self.hidden_size].view(torch.float).view(assignments, self.hidden_size)
            ext.exl3_moe_prefill_wmma(
                y, output, sel, rw, order, expert_count,
                self.multi_gate.ptrs_trellis, self.multi_gate.ptrs_suh, self.multi_gate.ptrs_svh,
                self.multi_up.ptrs_trellis, self.multi_up.ptrs_suh, self.multi_up.ptrs_svh,
                self.multi_down.ptrs_trellis, self.multi_down.ptrs_suh, self.multi_down.ptrs_svh,
                float(self.multi_gate.K), float(self.multi_down.K),
                gu_had,
                ws["gu_out"][:2 * assignments * self.intermediate_size_padded].view(2 * assignments, self.intermediate_size_padded),
                down_out,
                ws["offsets"], ws["inverse"][:assignments], ws["tiles"], ws["tile_count"],
                float(self.act_limit), bool(self.wmma_mcg),
            )
            final_hidden_states = output.view(eshape)

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

                # Count how many assignments per expert. With few enough total assignments no
                # expert can exceed the fused kernel's row capacity, so the readback (a CPU sync
                # per layer, ~33% idle at MTP verify shapes) is skipped and everything is fused
                if _moe_sync_free_count():
                    expert_count = _scatter_expert_count(flat_expert_local, E + 1)
                else:
                    expert_count = torch.bincount(flat_expert_local, minlength = E + 1)
                if self.fused_mode_buffers is not None and num_tokens * top_k <= self.fused_rows:
                    expert_count_list = None
                else:
                    expert_count_list = expert_count.tolist()

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
                        min_rows = self.fused_rows
                        fused_total = sum(c for c in expert_count_list[:num_ex] if 0 < c <= self.fused_rows)
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
                        n_slots = 0
                        if self.fused_mode_buffers is not None:
                            for e in range(num_ex):
                                c = expert_count_list[e]
                                if 0 < c <= self.fused_rows:
                                    base[e] = n_slots; kind[e] = 1; n_slots += c
                        for grp in groups:
                            cmax = max(expert_count_list[e] for e in grp)
                            for b, e in enumerate(grp):
                                base[e] = n_slots + b * cmax; kind[e] = 2
                            n_slots += len(grp) * cmax
                        tables = torch.from_numpy(np.stack([base, starts_np, kind])).to(y.device, non_blocking = True)
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

                # num_active -1 = unknown (all fused), kernel launches at max concurrency
                if self.fused_mode_buffers is not None:
                    if expert_count_list is None:
                        run_fused(-1)
                    else:
                        counts = [c for c in expert_count_list[:num_ex] if 0 < c <= self.fused_rows]
                        t1 = sum(1 for c in counts if MTILE_T1 < c <= MTILE_T2)
                        t2 = sum(1 for c in counts if c > MTILE_T2)
                        if self.mtile_ok and (t1 or t2):
                            # One launch per row tile over its expert range, largest first
                            t0 = len(counts) - t1 - t2
                            if t2:
                                run_fused(t2, MTILE_T2 + 1, self.fused_rows, 64)
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
                if scratch is not None:
                    ext.exl3_moe_gather(final_hidden_states, scratch, flat_expert_local, inv_order,
                                        tables[1, :num_ex], tables[0, :num_ex], tables[2, :num_ex], weight_sorted)

                out_state = None
                interm = None
                interm_a = None
                max_count = 0
                start = 0

                # expert_count_list None: everything already handled by the fused kernel above
                for expert_idx in range(num_ex if expert_count_list is not None else 0):
                    count = expert_count_list[expert_idx]
                    end = start + count
                    if count <= min_rows or expert_idx in batched:
                        start = end
                        continue

                    top_x = token_sorted[start:end]
                    w = weight_sorted[start:end].unsqueeze(1)

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
            self.bc.run_bszN(y, selected_experts, routing_weights)
            final_hidden_states = self.experts_cfg.out_bszn[:bsz].view(eshape)
            bc_sh_exp = self.bc_sh_exp

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
        if self.shared_experts and not bc_sh_exp:
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
        REPORT-16 Sec 6 found missing: that report's row-invariance probe exercised
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
        # EXL3_DEC_MOE_UNION=0 reproduces the exact pre-REPORT-17 behavior (generic norm+router,
        # dedup MoE) for a clean A/B against it -- R sequential kernel launches here is a real
        # per-layer cost even though it's bit-exact, not a free win to force on unconditionally.
        # Also refuse above MAX_BSZN: prefill calls this with R = chunk size (hundreds+), where
        # R sequential single-row launches would be catastrophically slower than the batched
        # prefill path this falls back to -- this is not what the flag or the kernel is for.
        if os.environ.get("EXL3_DEC_MOE_UNION", "0") == "0" or R > MAX_BSZN or not params.get("dflash_verify"):
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
