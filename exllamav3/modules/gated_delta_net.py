from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.tensor import get_for_device, to2
from . import Module, Linear
from ..ext import exllamav3_ext as ext
from ..model.model_tp_alloc import TPAllocation
from .gated_rmsnorm import GatedRMSNorm

_MIDCKPT_STREAM = None  # EXL3_MIDCHUNK_CKPT=2 side stream for the checkpoint D2H
from ..cache import Cache
from ..util.tensor import g_tensor_cache
from .multilinear import SlicedMultiLinear
import os

# Sliced qkv+z projection bundle at decode for the split-projection GDN (Qwen3.5 / Qwen3.8 style):
# one mgemm over equal-width column slices, see attn.py. EXL3_QKV_SLICE=0 disables it
_qkv_slice_enable = os.environ.get("EXL3_QKV_SLICE", "1") != "0"
from ..model.model_tp_shared import TPTensorWrapper
from .gated_delta_net_fn import causal_conv1d_update, gated_delta_rule_fn
from ..cache.recurrent import (
    mp_cache_recurrent_stash,
    mp_cache_recurrent_unstash,
    mp_cache_recurrent_clear,
    new_checkpoint_handle,
)
from ..util import profile_opt
from .attention_fn.bc_attn import MAX_BSZ as _BC_MAX_BSZ, MAX_QLEN as _BC_MAX_QLEN

# Fused KDA beta/g gate kernel (ROCm fused_elt_rocm.cu); read at import
_FUSE_KDA_GATE = hasattr(ext, "kda_gate") and os.environ.get("EXL3_FUSE_KDA_GATE", "1") != "0"
# Decode-graph KDA gate: keep dt_bias in fp32 (as prefill and the HF ForgetGate do) instead of a bf16
# copy. GLM dt_bias is stored fp32 and is not bf16-exact (max rounding error 0.03)
_KDA_DT_F32 = os.environ.get("EXL3_KDA_DT_F32", "0") != "0"
KDA_KNOBS = {
    "half_a": os.environ.get("EXL3_KDA_HALF_A", "1") != "0",
    "dec_cat": os.environ.get("EXL3_KDA_DEC_CAT", "1") != "0",
    "lr_batched": os.environ.get("EXL3_KDA_DEC_LR_BATCHED", "1") != "0",
    # kda-dec step 2: f_b fused into the gate kernel, g_b fused into the gated norm (bit-exact)
    "lr_fused": os.environ.get("EXL3_KDA_DEC_LR_FUSED", "1") != "0",
}


def _collect_rewind_jobs(layers, slot: int, last_history: int, num_tokens: int):
    """Split a batch of recurrent-layer states into per-device (conv_jobs, state_jobs) for the
    batched rewind kernels. With layer-split loading a single cache's GDN layers span multiple
    devices, so jobs must be grouped by the device each layer's state actually lives on --
    launching them all under one device index dereferences foreign pointers (illegal memory
    access on any multi-GPU split). Only GDNLayerState instances (GDN and Mamba2 alike) are
    batched; any other recurrent-state type sharing the same cache (e.g. SWA, short-conv)
    falls back to its own .rewind() call, unchanged."""
    jobs_by_device = {}
    for l in layers:
        if isinstance(l, GDNLayerState):
            # l.device may be a plain string ("cuda:0") in some TP contexts rather than a
            # torch.device, so normalize rather than assume a .index attribute
            device_index = torch.device(l.device).index
            conv_jobs, state_jobs = jobs_by_device.setdefault(device_index, ([], []))
            cj = l.rewind_conv_job(slot, last_history, num_tokens)
            if cj is not None:
                conv_jobs.append(cj)
            sj = l.rewind_state_job(slot, last_history, num_tokens)
            if sj is not None:
                state_jobs.append(sj)
        else:
            l.rewind(slot, last_history, num_tokens)
    return jobs_by_device


def _dispatch_rewind_jobs(jobs_by_device):
    for device_index, (conv_jobs, state_jobs) in jobs_by_device.items():
        if conv_jobs:
            ext.batched_conv_rewind(conv_jobs, device_index)
        if state_jobs:
            ext.batched_state_rewind(state_jobs, device_index)


def mp_cache_recurrent_rewind(local_context: dict, cache_id: int, slot: int, last_history, num_tokens):
    recurrent_modules = local_context["recurrent_modules"]
    layers = [module.tp_recurrent_lookup[cache_id] for module in recurrent_modules]
    _dispatch_rewind_jobs(_collect_rewind_jobs(layers, slot, last_history, num_tokens))


class GDNState:

    def __init__(
        self,
        cache: Cache,
        slot: int,
        position: int,
        clear: bool = True,
        stashed: dict = None,
        test_state: bool = False,
        exported: bool = False,
    ):
        self.slot = slot
        self.position = position
        self.cache = cache
        self.last_history = 0
        self.exported = exported

        if not exported:
            assert test_state or position == 0 or stashed is not None, \
                "State must be new, restored from checkpoint or marked as a test state."

            if clear and stashed is None:
                if not self.cache.model.loaded_tp:
                    for l in self.cache.get_all_recurrent_layers().values():
                        l.clear(slot)
                else:
                    self.cache.model.tp_dispatch_all(mp_cache_recurrent_clear, (id(self.cache), self.slot))

            if stashed is not None:
                self.unstash(stashed)

            self.checkpoint_size = sum(
                l.get_checkpoint_size()
                for l in self.cache.get_all_recurrent_layers().values()
            )


    def free(self):
        self.cache.release_state(self)


    def rewind(self, num_tokens: int):
        if not self.cache.model.loaded_tp:
            _dispatch_rewind_jobs(_collect_rewind_jobs(
                self.cache.get_all_recurrent_layers().values(), self.slot, self.last_history, num_tokens
            ))
        else:
            self.cache.model.tp_dispatch_all(mp_cache_recurrent_rewind, (id(self.cache), self.slot, self.last_history, num_tokens))
        self.position -= num_tokens
        self.last_history = 0


    def rollback_capacity(self):
        # The state advances destructively; rewinding is only possible immediately after a forward pass that
        # recorded per-token history (speculative decoding), never at an arbitrary later point
        return 0


    def stash(self):
        stashed = {
            "position": self.position,
            "checkpoint_size": self.checkpoint_size
        }
        if not self.cache.model.loaded_tp:
            for k, l in self.cache.get_all_recurrent_layers().items():
                stashed[k] = l.stash(self.slot)
        else:
            cp_handle = new_checkpoint_handle()
            self.cache.model.tp_dispatch_all(mp_cache_recurrent_stash, (id(self.cache), cp_handle, self.slot))
            stashed["tp_handle"] = cp_handle
        return stashed


    def unstash(self, stashed: dict):
        assert self.position == stashed["position"]
        if not self.cache.model.loaded_tp:
            for k, l in self.cache.get_all_recurrent_layers().items():
                l.unstash(self.slot, stashed[k])
        else:
            cp_handle = stashed["tp_handle"]
            self.cache.model.tp_dispatch_all(mp_cache_recurrent_unstash, (id(self.cache), cp_handle, self.slot))


    def post_advance(self):
        pass


    def tp_export(self):
        return GDNState(
            cache = id(self.cache),
            slot = self.slot,
            position = self.position,
            exported = True,
        )


    def reset(self):
        self.position = 0


class GDNLayerState:

    def __init__(
        self,
        module: GatedDeltaNet,
        max_batch_size: int,
        max_history: int,
        cache_id: int,
    ):
        self.module = module
        self.conv_state = torch.empty(
            (max_batch_size, module.fdim_qkv, module.conv_kernel_size + max_history),
            dtype = torch.bfloat16,
            device = "meta"
        )
        self.recurrent_state = torch.empty(
            (max_batch_size, max_history + 1, module.num_v_heads, module.k_head_dim, module.v_head_dim),
            dtype = torch.float,
            device = "meta"
        )
        self.device = None
        self.max_history = max_history
        self.max_batch_size = max_batch_size
        self.cache_id = cache_id


    def get_checkpoint_size(self):
        return (
            self.module.fdim_qkv * self.module.conv_kernel_size * 2 +
            self.module.num_v_heads * self.module.k_head_dim * self.module.v_head_dim * 4
        )


    def storage_size(self):
        return sum(t.numel() * t.element_size() for t in [self.conv_state, self.recurrent_state])


    def alloc(self, device):
        self.conv_state = torch.empty_like(self.conv_state, device = device)
        self.recurrent_state = torch.empty_like(self.recurrent_state, device = device)
        self.conv_state.zero_()
        self.recurrent_state.zero_()
        self.device = device


    def free(self):
        self.conv_state = torch.empty_like(self.conv_state, device = "meta")
        self.recurrent_state = torch.empty_like(self.recurrent_state, device = "meta")
        self.device = None


    def clear(self, idx: int):
        if self.device is not None:
            self.conv_state[idx].zero_()
            self.recurrent_state[idx].zero_()


    def get_state_tensors(self):
        return (
            self.conv_state,
            self.recurrent_state,
        )


    def rewind(self, slot: int, last_history: int, num_tokens: int):
        assert num_tokens <= last_history
        if num_tokens > 0:
            r_state = self.recurrent_state[slot, 0]
            r_state_rewind = self.recurrent_state[slot, last_history + 1 - num_tokens]
            r_state.copy_(r_state_rewind)
        cdim = self.module.conv_kernel_size
        if last_history > 0:
            c_state = self.conv_state[slot, :, :cdim]
            p = self.conv_state.shape[-1] - num_tokens
            c_state_rewind = self.conv_state[slot, :, p - cdim : p]
            temp = c_state_rewind.clone()
            c_state.copy_(temp)


    def rewind_conv_job(self, slot: int, last_history: int, num_tokens: int):
        """Job descriptor for the batched conv-state rewind kernel (ext.batched_conv_rewind),
        computed without performing any copy. Same gating condition as rewind()'s conv branch."""
        if last_history == 0:
            return None
        cdim = self.module.conv_kernel_size
        p = self.conv_state.shape[-1] - num_tokens
        return ext.ConvRewindJob(
            self.conv_state[slot, 0, p - cdim].data_ptr(),
            self.conv_state[slot, 0, 0].data_ptr(),
            self.conv_state.shape[1],
            cdim,
            self.conv_state.stride(1),
        )


    def rewind_state_job(self, slot: int, last_history: int, num_tokens: int):
        """Job descriptor for the batched recurrent-state rewind kernel (ext.batched_state_rewind),
        computed without performing any copy. Same gating condition as rewind()'s state branch."""
        if num_tokens == 0:
            return None
        return ext.StateRewindJob(
            self.recurrent_state[slot, last_history + 1 - num_tokens].data_ptr(),
            self.recurrent_state[slot, 0].data_ptr(),
            self.recurrent_state[slot, 0].numel(),
        )


    def stash(self, slot, position: int = 0):
        cdim = self.module.conv_kernel_size
        return (
            self.recurrent_state[slot, :1].cpu(),
            self.conv_state[slot, :, :cdim].cpu()
        )


    def unstash(self, slot, stashed, position: int = 0):
        cdim = self.module.conv_kernel_size
        s, c = stashed
        self.recurrent_state[slot, :1].copy_(s)
        self.conv_state[slot, :, :cdim].copy_(c)


    def tp_export(self, plan):
        return {
            "cls": GDNLayerState,
            "args": {
                "cache_id": self.cache_id,
                "max_history": self.max_history,
                "max_batch_size": self.max_batch_size,
            }
        }


class GatedDeltaNet(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        k_head_dim: int,
        v_head_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        rms_norm_eps: float,
        conv_kernel_size: int,
        beta_scale: float = 1.0,
        key_a_log: str | None = None,
        key_dt_bias: str | None = None,
        key_conv1d: str | None = None,
        key_conv1d_q: str | None = None,
        key_conv1d_k: str | None = None,
        key_conv1d_v: str | None = None,
        key_fused_ba: str | None = None,
        key_fused_qkvz: str | None = None,
        key_qkv: str | None = None,
        key_qkv_alt: list | None = None,
        key_z: str | None = None,
        key_b: str | None = None,
        key_a: str | None = None,
        key_f_a: str | None = None,
        key_f_b: str | None = None,
        key_g_a: str | None = None,
        key_g_b: str | None = None,
        gate_lower_bound: float | None = None,
        key_norm: str | None = None,
        key_o: str | None = None,
        a_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        conv1d_weight: torch.Tensor | None = None,
        conv1d_bias: torch.Tensor | None = None,
        qkv_proj: Linear | None = None,
        z_proj: Linear | None = None,
        b_proj: Linear | None = None,
        a_proj: Linear | None = None,
        norm: GatedRMSNorm | None = None,
        o_proj: Linear | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        select_hq_bits: int = 0,
    ):
        super().__init__(config, key, None)
        self.module_name = "GatedDeltaNet"

        self.q_priority = 1 + select_hq_bits
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.k_head_dim = k_head_dim
        self.v_head_dim = v_head_dim
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.num_v_groups = num_v_heads // num_k_heads if num_k_heads else 0
        self.rms_norm_eps = rms_norm_eps
        self.conv_kernel_size = conv_kernel_size
        self.k_dim = self.k_head_dim * self.num_k_heads
        self.v_dim = self.v_head_dim * self.num_v_heads
        self.beta_scale = beta_scale

        self.out_dtype = out_dtype

        self.fdim_qkvz = 2 * self.num_k_heads * self.k_head_dim + 2 * self.num_v_heads * self.v_head_dim
        self.fdim_ba = 2 * self.num_v_heads
        self.fdim_qkv = 2 * self.num_k_heads * self.k_head_dim + self.num_v_heads * self.v_head_dim

        if self.num_k_heads == 0:
            return

        # KDA mode (GLM5.3/Kimi linear attention): per-k-channel decay from a low-rank forget
        # gate (f_a/f_b + per-channel dt_bias + per-head A_log, "safe gate" when
        # gate_lower_bound is set), a low-rank sigmoid output gate (g_a/g_b) in place of z,
        # and in-kernel q/k l2norm. Shares the conv, projections, cache and rewind machinery
        self.kda = key_f_a is not None
        self.gate_lower_bound = gate_lower_bound
        if self.kda:
            assert key_f_b and key_g_a and key_g_b, \
                "KDA mode requires key_f_a, key_f_b, key_g_a and key_g_b"
            assert key_b and not key_a, \
                "KDA mode takes key_b for beta; decay comes from the f projections"
            assert num_k_heads == num_v_heads and k_head_dim == v_head_dim, \
                "KDA mode requires uniform head geometry"

        if key_qkv or key_z:
            assert key_qkv and (key_z or self.kda), \
                "GatedDeltaNet split qkv/z projections require both key_qkv and key_z"
        if (key_b or key_a) and not self.kda:
            assert key_b and key_a, \
                "GatedDeltaNet split b/a projections require both key_b and key_a"

        if key_fused_qkvz:
            self.qkvz_proj = Linear(
                config,
                f"{key}.{key_fused_qkvz}",
                hidden_size,
                self.fdim_qkvz,
                qmap = qmap + ".input",
                out_dtype = torch.float,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".qkvz",
            )
            self.qkv_proj = None
            self.z_proj = None
            self.register_submodule(self.qkvz_proj)
        elif qkv_proj:
            self.qkv_proj = qkv_proj
            self.z_proj = z_proj
            self.qkvz_proj = None
            self.register_submodule(self.qkv_proj)
            self.register_submodule(self.z_proj)
        elif key_qkv:
            self.qkv_proj = Linear(
                config,
                f"{key}.{key_qkv}",
                hidden_size,
                self.fdim_qkv,
                qmap = qmap + ".input",
                out_dtype = torch.float,
                alt_key = None if not key_qkv_alt else [f"{key}.{x}" for x in key_qkv_alt],
                select_hq_bits = select_hq_bits,
                qgroup = key + ".qkvz",
            )
            self.qkvz_proj = None
            self.register_submodule(self.qkv_proj)
            if key_z:
                self.z_proj = Linear(
                    config,
                    f"{key}.{key_z}",
                    hidden_size,
                    self.v_dim,
                    qmap = qmap + ".input",
                    out_dtype = torch.float,
                    select_hq_bits = select_hq_bits,
                    qgroup = key + ".qkvz",
                )
                self.register_submodule(self.z_proj)
            else:
                self.z_proj = None
        else:
            self.qkvz_proj = None
            self.qkv_proj = None
            self.z_proj = None

        if key_fused_ba:
            self.ba_proj = Linear(config, f"{key}.{key_fused_ba}", hidden_size, self.fdim_ba, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.b_proj = None
            self.a_proj = None
            self.register_submodule(self.ba_proj)
        elif b_proj:
            self.b_proj = b_proj
            self.a_proj = a_proj
            self.ba_proj = None
            self.register_submodule(self.b_proj)
            self.register_submodule(self.a_proj)
        elif key_b and self.kda:
            self.b_proj = Linear(config, f"{key}.{key_b}", hidden_size, self.num_v_heads, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.a_proj = None
            self.ba_proj = None
            self.register_submodule(self.b_proj)
        elif key_b:
            self.b_proj = Linear(config, f"{key}.{key_b}", hidden_size, self.num_v_heads, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.a_proj = Linear(config, f"{key}.{key_a}", hidden_size, self.num_v_heads, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.ba_proj = None
            self.register_submodule(self.b_proj)
            self.register_submodule(self.a_proj)
        else:
            self.b_proj = None
            self.a_proj = None
            self.ba_proj = None

        # KDA low-rank forget-gate and output-gate projections. Kept in fp16 (qmap = None):
        # the reference fp8 checkpoints exclude them from quantization, which is a strong
        # sensitivity signal
        if self.kda:
            self.f_a_proj = Linear(config, f"{key}.{key_f_a}", hidden_size, self.k_head_dim, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.f_b_proj = Linear(config, f"{key}.{key_f_b}", self.k_head_dim, self.k_dim, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.g_a_proj = Linear(config, f"{key}.{key_g_a}", hidden_size, self.v_head_dim, qmap = None, out_dtype = torch.float, pad_to = 1)
            self.g_b_proj = Linear(config, f"{key}.{key_g_b}", self.v_head_dim, self.v_dim, qmap = None, out_dtype = torch.float, pad_to = 1)
            for m in (self.f_a_proj, self.f_b_proj, self.g_a_proj, self.g_b_proj):
                self.register_submodule(m)
        else:
            self.f_a_proj = self.f_b_proj = self.g_a_proj = self.g_b_proj = None

        if o_proj:
            self.o_proj = o_proj
            self.register_submodule(self.o_proj)
        else:
            self.o_proj = Linear(
                config,
                f"{key}.{key_o}",
                self.v_head_dim * self.num_v_heads,
                hidden_size,
                qmap = qmap + ".output",
                out_dtype = self.out_dtype,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".o",
            )
            self.register_submodule(self.o_proj)

        if norm is not None:
            self.norm = norm
            self.register_submodule(self.norm)
        else:
            self.norm = GatedRMSNorm(
                config, f"{key}.{key_norm}", self.rms_norm_eps, out_dtype = torch.half,
                gate_activation = "sigmoid" if self.kda else "silu")
            self.register_submodule(self.norm)

        self.a_log = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_weight_flat = None
        self.conv1d_bias = None
        self.conv1d_q_weight = None
        self.conv1d_k_weight = None
        self.conv1d_v_weight = None

        if dt_bias is not None:
            self.a_log = a_log
            self.dt_bias = dt_bias
            self.key_a_log = None
            self.key_dt_bias = None
        else:
            self.key_a_log = f"{key}.{key_a_log}"
            self.key_dt_bias = f"{key}.{key_dt_bias}"
        if conv1d_weight is not None:
            self.conv1d_weight = conv1d_weight
            self.conv1d_bias = conv1d_bias
            self.key_conv1d_weight = None,
            self.key_conv1d_bias = None,
            self.key_conv1d_q_weight = None,
            self.key_conv1d_k_weight = None,
            self.key_conv1d_v_weight = None,
        else:
            self.key_conv1d_weight = f"{key}.{key_conv1d}.weight"
            self.key_conv1d_bias = f"{key}.{key_conv1d}.bias"
            self.key_conv1d_q_weight = f"{key}.{key_conv1d_q}.weight" if key_conv1d_q else None
            self.key_conv1d_k_weight = f"{key}.{key_conv1d_k}.weight" if key_conv1d_k else None
            self.key_conv1d_v_weight = f"{key}.{key_conv1d_v}.weight" if key_conv1d_v else None

        self.conv_dim = self.k_head_dim * self.num_k_heads

        self.caps.update({
            "recurrent_cache": True
        })
        self.layer_state_cls = GDNLayerState

        self.bc = None
        self.bc_split = False
        self.bsz1_pa_args = []
        self.ba_weight_t = None
        self.ba_bias = None
        self.ba_weight_filled = False

        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}
        self.tp_reduce = False
        self.has_split_cache = False


    @override
    def optimizer_targets(self):
        if self.qkvz_proj is not None:
            return [[
                self.qkvz_proj.optimizer_targets(),
                self.o_proj.optimizer_targets(),
            ]]

        targets = []
        if self.qkv_proj is not None:
            targets.append(self.qkv_proj.optimizer_targets())
        if self.z_proj is not None:
            targets.append(self.z_proj.optimizer_targets())
        targets.append(self.o_proj.optimizer_targets())
        return [targets]


    def load_local(self, device, **kwargs):

        if self.num_k_heads == 0:
            return

        # Recurrent states
        for rl in self.recurrent_layers:
            rl.alloc(device)

        is_quantized = (
            self.qkvz_proj is not None and self.qkvz_proj.quant_format_id() == "exl3" and
            self.ba_proj is not None and self.ba_proj.quant_format_id() is None and
            self.o_proj is not None and self.o_proj.quant_format_id() == "exl3"
        )

        if is_quantized:
            self.bsz1_pa_args = [
                (device, (1, self.fdim_qkv, 1), torch.bfloat16),
                (device, (1, 1, self.num_v_heads, self.v_head_dim), torch.bfloat16, "a"),
                (device, (1, 1, self.num_v_heads), torch.bfloat16),
                (device, (1, 1, self.num_v_heads), torch.float),
                (device, (1, 1, self.fdim_qkvz), torch.float),
                (device, (1, 1, self.fdim_ba), torch.float),
                (device, (1, self.fdim_qkv, self.conv_kernel_size + 1), torch.bfloat16, "a"),
                (device, (1, self.fdim_qkv, 2), torch.bfloat16, "b"),
                (device, (1, 1, self.num_v_heads, self.v_head_dim), torch.bfloat16, "b"),
                (device, (1, 1, self.num_v_heads * self.v_head_dim), torch.half),
            ]

            self.bc = ext.BC_GatedDeltaNet(
                *(g_tensor_cache.get(*arg) for arg in self.bsz1_pa_args),
                self.qkvz_proj.inner.bc,
                self.ba_proj.inner.bc,
                self.dt_bias,
                self.a_log,
                self.num_k_heads,
                self.num_v_heads,
                self.k_head_dim,
                self.v_head_dim,
                self.conv1d_weight,
                self.conv1d_bias,
                self.norm.bc,
                self.o_proj.inner.bc,
                self.beta_scale
            )

        # Fuse conv1d weights and cache the flattened weight (normally done lazily in forward,
        # needed here for the split-projection batched call)
        if self.conv1d_weight is None and self.conv1d_q_weight is not None:
            self.conv1d_weight = torch.cat([
                self.conv1d_q_weight,
                self.conv1d_k_weight,
                self.conv1d_v_weight,
            ], dim = 0)
            self.conv1d_q_weight = None
            self.conv1d_k_weight = None
            self.conv1d_v_weight = None
        if self.conv1d_weight_flat is None and self.conv1d_weight is not None:
            self.conv1d_weight_flat = self.conv1d_weight.squeeze(1).contiguous()

        is_quantized_split = (
            device != torch.device("cpu") and
            self.qkvz_proj is None and self.ba_proj is None and
            self.qkv_proj is not None and self.qkv_proj.quant_type == "exl3" and
            self.z_proj is not None and self.z_proj.quant_type == "exl3" and
            self.b_proj is not None and self.b_proj.quant_type == "fp16" and
            self.a_proj is not None and self.a_proj.quant_type == "fp16" and
            self.o_proj is not None and self.o_proj.quant_type == "exl3" and
            self.conv1d_weight_flat is not None and
            self.conv1d_weight_flat.dtype == torch.bfloat16 and
            (self.conv1d_bias is None or self.conv1d_bias.dtype == torch.bfloat16) and
            self.dt_bias is not None and self.dt_bias.dtype == torch.bfloat16
        )

        if is_quantized_split:
            # Merge the small unquantized b/a projections into a single fp16 GEMV. The weights may
            # not be materialized yet (deferred load), so only allocate here — the BC keeps a
            # reference — and copy the actual values in on the first forward pass
            nv, hv = self.num_v_heads, self.v_head_dim
            self.ba_weight_t = torch.empty((2 * nv, self.hidden_size), dtype = torch.half, device = device)
            has_bias = (
                self.b_proj.inner.get_bias_tensor() is not None or
                self.a_proj.inner.get_bias_tensor() is not None
            )
            self.ba_bias = torch.empty((2 * nv,), dtype = torch.half, device = device) if has_bias else None
            self.ba_weight_filled = False

            self.bc = ext.BC_GatedDeltaNetSplit(
                self.qkv_proj.inner.bc,
                self.z_proj.inner.bc,
                self.o_proj.inner.bc,
                self.ba_weight_t,
                self.ba_bias,
                self.dt_bias,
                self.a_log,
                self.num_k_heads,
                nv,
                self.k_head_dim,
                hv,
                self.conv1d_weight_flat,
                self.conv1d_bias,
                self.norm.bc,
                self.beta_scale
            )
            self.bc_split = self.bc is not None

            # Sliced qkv+z bundle: both projections read x and are cut into equal-width column
            # slices run as one launch (SlicedMultiLinear); the graph object gets the tables, the
            # eager path (m <= 32) uses them directly
            self.multi_qkvz = None
            if (
                _qkv_slice_enable and
                self.qkv_proj.inner.bias is None and self.z_proj.inner.bias is None and
                self.qkv_proj.inner.K == self.z_proj.inner.K and
                self.qkv_proj.in_features == self.z_proj.in_features
            ):
                try:
                    self.multi_qkvz = SlicedMultiLinear(self.device, [self.qkv_proj, self.z_proj])
                except (ValueError, AssertionError):
                    self.multi_qkvz = None
                # Unfusing policy judged on the slice width (see attn.py)
                if self.multi_qkvz is not None and not self.config.infer_params.use_mgemm(
                    self.multi_qkvz.K, self.multi_qkvz.width, self.multi_qkvz.mul1, device,
                ):
                    self.multi_qkvz = None
            if self.multi_qkvz is not None:
                mq = self.multi_qkvz
                self.bc.set_qkvz_bundle(mq.ptrs_trellis, mq.ptrs_suh, mq.ptrs_svh, mq.meta, mq.K, bool(mq.mcg), bool(mq.mul1))
                # fp32 outputs: the mgemm C argument carries the dtype and slice width only
                self.prealloc_qkvz_carrier = g_tensor_cache.get(device, (mq.num_slices, 1, mq.width), torch.float, "qkvzc_1")

        is_quantized_kda = (
            device != torch.device("cpu") and self.kda and
            self.qkv_proj is not None and self.qkv_proj.quant_type == "exl3" and
            self.o_proj is not None and self.o_proj.quant_type == "exl3" and
            all(p is not None and p.quant_type == "fp16" for p in
                (self.b_proj, self.f_a_proj, self.f_b_proj, self.g_a_proj, self.g_b_proj)) and
            self.conv1d_weight_flat is not None and
            self.conv1d_weight_flat.dtype == torch.bfloat16 and
            (self.conv1d_bias is None or self.conv1d_bias.dtype == torch.bfloat16) and
            self.dt_bias is not None
        )

        if is_quantized_kda:
            # The gate op reads dt_bias as bf16 (default) or fp32 (EXL3_KDA_DT_F32=1; GLM stores it
            # fp32 and it is not bf16-exact). Allocated here, FILLED in the deferred-fill
            # block on the first forward: at this point dt_bias may still be an unfilled
            # deferred tensor, and a .to() would copy garbage
            self.dt_bias_bc = torch.empty_like(self.dt_bias, dtype = torch.float if _KDA_DT_F32 else torch.bfloat16)
            # Transposed fp16 copies of the small projections for the graph GEMVs; weights may
            # not be materialized yet (deferred load) so only allocate here and fill on the
            # first forward (the BC keeps references)
            nv, hk, hv = self.num_v_heads, self.k_head_dim, self.v_head_dim
            hs = self.hidden_size
            self.kda_b_t = torch.empty((nv, hs), dtype = torch.half, device = device)
            self.kda_fa_t = torch.empty((hk, hs), dtype = torch.half, device = device)
            self.kda_fb_t = torch.empty((nv * hk, hk), dtype = torch.half, device = device)
            self.kda_ga_t = torch.empty((hv, hs), dtype = torch.half, device = device)
            self.kda_gb_t = torch.empty((nv * hv, hv), dtype = torch.half, device = device)
            self.ba_weight_filled = False

            self.bc = ext.BC_GatedDeltaNetSplit(
                self.qkv_proj.inner.bc,
                self.o_proj.inner.bc,
                self.kda_b_t,
                self.kda_fa_t,
                self.kda_fb_t,
                self.kda_ga_t,
                self.kda_gb_t,
                self.dt_bias_bc,
                self.a_log,
                float(self.gate_lower_bound or 0.0),
                self.num_k_heads,
                nv,
                self.k_head_dim,
                hv,
                self.conv1d_weight_flat,
                self.conv1d_bias,
                self.norm.bc,
                self.beta_scale
            )
            self.bc_split = self.bc is not None


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)
        if self.key_a_log is not None:
            self.a_log = self.config.stc.get_tensor(self.key_a_log, self.device, optional = False, allow_bf16 = True)
            # Kimi Linear stores A_log as (1, 1, H, 1); the kernels take it as (H,)
            self.a_log = self.a_log.reshape(-1)
            self.dt_bias = self.config.stc.get_tensor(self.key_dt_bias, self.device, optional = False, allow_bf16 = True)
        if self.key_conv1d_weight is not None:
            # no_defer: load_local concatenates/flattens (copies) these immediately, which a
            # deferred (unfilled) tensor would corrupt
            self.conv1d_weight = self.config.stc.get_tensor(self.key_conv1d_weight, self.device, optional = True, allow_bf16 = True, no_defer = True)
            self.conv1d_bias = self.config.stc.get_tensor(self.key_conv1d_bias, self.device, optional = True, allow_bf16 = True, no_defer = True)
            if self.conv1d_weight is None:
                self.conv1d_q_weight = self.config.stc.get_tensor(self.key_conv1d_q_weight, self.device, optional = False, allow_bf16 = True, no_defer = True)
                self.conv1d_k_weight = self.config.stc.get_tensor(self.key_conv1d_k_weight, self.device, optional = False, allow_bf16 = True, no_defer = True)
                self.conv1d_v_weight = self.config.stc.get_tensor(self.key_conv1d_v_weight, self.device, optional = False, allow_bf16 = True, no_defer = True)
        self.norm.load(device, **kwargs)
        self.load_local(device, **kwargs)


    @override
    def unload(self):
        if self.bc is not None:
            # for arg in self.bsz1_pa_args:
            #     g_tensor_cache.drop(*arg)
            self.bc = None
            self.bc_split = False
            self.bsz1_pa_args = []
        self.ba_weight_t = None
        self.ba_bias = None
        self.ba_weight_filled = False
        self.multi_qkvz = None
        self.prealloc_qkvz_carrier = None
        self.a_log = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_weight_flat = None
        self.conv1d_bias = None
        self.conv1d_q_weight = None
        self.conv1d_k_weight = None
        self.conv1d_v_weight = None
        self.norm.unload()
        for cl in self.recurrent_layers:
            cl.free()
        super().unload()


    def split_fused_inputs(self, mixed_qkvz, mixed_ba):
        # mixed_qkvz and mixed_ba have same (bsz, seqlen)
        # both are contiguous
        bsz, seqlen, _ = mixed_qkvz.shape

        mixed_qkvz = mixed_qkvz.view(
            bsz,
            seqlen,
            self.num_k_heads,
            2 * self.k_head_dim + 2 * self.v_head_dim * self.num_v_heads // self.num_k_heads,
        )
        mixed_ba = mixed_ba.view(
            bsz,
            seqlen,
            self.num_k_heads,
            2 * self.num_v_heads // self.num_k_heads
        )

        split_arg_list_qkvz = [
            self.k_head_dim,
            self.k_head_dim,
            (self.num_v_groups * self.v_head_dim),
            (self.num_v_groups * self.v_head_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads
        ]
        q, k, v, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim = 3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim = 3)

        q = q.reshape(bsz, seqlen, -1)
        k = k.reshape(bsz, seqlen, -1)
        v = v.reshape(bsz, seqlen, -1)
        z = z.reshape(bsz, seqlen, -1, self.v_head_dim)
        b = b.reshape(bsz, seqlen, self.num_v_heads)
        a = a.reshape(bsz, seqlen, self.num_v_heads)
        mixed_qkv = torch.cat((q, k, v), dim = -1)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        return mixed_qkv, z, b, a


    def _bc_configure_slot_kda(self, bsz: int, seqlen: int, history: bool):
        device = self.device
        f = self.fdim_qkv
        nv, hk, hv = self.num_v_heads, self.k_head_dim, self.v_head_dim
        qkv             = g_tensor_cache.get(device, (bsz, seqlen, f), torch.float, "s_qkv")
        z               = g_tensor_cache.get(device, (bsz, seqlen, nv, hv), torch.float, "s_z")
        b_out           = g_tensor_cache.get(device, (bsz, seqlen, nv), torch.float, "s_kb")
        fa_out          = g_tensor_cache.get(device, (bsz, seqlen, hk), torch.float, "s_kfa")
        fb_out          = g_tensor_cache.get(device, (bsz, seqlen, nv * hk), torch.float, "s_kfb")
        ga_out          = g_tensor_cache.get(device, (bsz, seqlen, hv), torch.float, "s_kga")
        beta            = g_tensor_cache.get(device, (bsz, seqlen, nv), torch.bfloat16, "s_beta")
        g               = g_tensor_cache.get(device, (bsz, seqlen, nv, hk), torch.float, "s_kg4")
        mixed_qkv       = g_tensor_cache.get(device, (bsz, f, seqlen), torch.bfloat16, "s_mqkv")
        conv_out        = g_tensor_cache.get(device, (bsz, seqlen, f), torch.bfloat16, "s_conv")
        core_attn_out   = g_tensor_cache.get(device, (bsz, seqlen, nv, hv), torch.bfloat16, "s_cao")
        core_attn_out_f = g_tensor_cache.get(device, (bsz, seqlen, nv * hv), torch.half, "s_caof")
        qkv_xh = g_tensor_cache.get(device, (bsz, seqlen, self.hidden_size), torch.half, "s_qkv_xh")
        o_xh   = g_tensor_cache.get(device, (bsz, seqlen, nv * hv), torch.half, "s_o_xh")
        self.bc.configure_slot_kda(
            bsz, seqlen, history,
            qkv, z, b_out, fa_out, fb_out, ga_out, beta, g, mixed_qkv, conv_out,
            core_attn_out, core_attn_out_f, qkv_xh, o_xh,
        )

    def project_qkvz_sliced(self, x: torch.Tensor, bsz: int, seqlen: int) -> tuple:
        """qkv and z projections as one sliced mgemm (fp32 outputs, like the Linears); m <= 32"""
        mq = self.multi_qkvz
        m = bsz * seqlen
        hidden = self.qkv_proj.in_features
        x = x.half()
        if x.shape[-1] < hidden:
            x = torch.nn.functional.pad(x, (0, hidden - x.shape[-1]))
        x = x.contiguous().view(1, m, hidden)
        xh = torch.empty((mq.num_src, m, hidden), dtype = torch.half, device = x.device)
        qkv = torch.empty((bsz, seqlen, self.qkv_proj.out_features), dtype = torch.float, device = x.device)
        z = torch.empty((bsz, seqlen, self.z_proj.out_features), dtype = torch.float, device = x.device)
        c_ptrs = mq.c_ptrs([qkv.view(m, -1), z.view(m, -1)])
        ext.exl3_mgemm(
            x,
            mq.ptrs_trellis,
            self.prealloc_qkvz_carrier.expand(mq.num_slices, m, mq.width),
            mq.ptrs_suh,
            xh,
            mq.ptrs_svh,
            None,
            None,
            mq.K,
            -1,
            mq.mcg,
            mq.mul1,
            -1,
            -1,
            0,
            1,
            mq.size_n_list,
            c_ptrs,
            mq.n_stride_list,
            mq.had_src_list,
            mq.num_src,
        )
        return qkv, z


    def _bc_configure_slot(self, bsz: int, seqlen: int, history: bool):
        """Allocate (or fetch, if already cached at this exact shape) the per-(bsz, seqlen)
        statics for the BC_GatedDeltaNetSplit graph slot and hand them to C++. Called at most
        once per (bsz, seqlen, history) combination per layer instance"""
        device = self.device
        f = self.fdim_qkv
        nv, hv = self.num_v_heads, self.v_head_dim
        qkv             = g_tensor_cache.get(device, (bsz, seqlen, f), torch.float, "s_qkv")
        z               = g_tensor_cache.get(device, (bsz, seqlen, nv, hv), torch.float, "s_z")
        ba              = g_tensor_cache.get(device, (bsz, seqlen, 2 * nv), torch.float, "s_ba")
        beta            = g_tensor_cache.get(device, (bsz, seqlen, nv), torch.bfloat16, "s_beta")
        g               = g_tensor_cache.get(device, (bsz, seqlen, nv), torch.float, "s_g")
        mixed_qkv       = g_tensor_cache.get(device, (bsz, f, seqlen), torch.bfloat16, "s_mqkv")
        conv_out        = g_tensor_cache.get(device, (bsz, seqlen, f), torch.bfloat16, "s_conv")
        core_attn_out   = g_tensor_cache.get(device, (bsz, seqlen, nv, hv), torch.bfloat16, "s_cao")
        core_attn_out_f = g_tensor_cache.get(device, (bsz, seqlen, nv * hv), torch.half, "s_caof")
        qkv_xh = g_tensor_cache.get(device, (bsz, seqlen, self.hidden_size), torch.half, "s_qkv_xh")
        z_xh   = g_tensor_cache.get(device, (bsz, seqlen, self.hidden_size), torch.half, "s_z_xh")
        o_xh   = g_tensor_cache.get(device, (bsz, seqlen, nv * hv), torch.half, "s_o_xh")
        self.bc.configure_slot(
            bsz, seqlen, history,
            qkv, z, ba, beta, g, mixed_qkv, conv_out, core_attn_out, core_attn_out_f,
            qkv_xh, z_xh, o_xh,
        )


    def _kda_qkv_f16(self, x: torch.Tensor) -> bool:
        # EXL3_KDA_QKV_F16=1 (default): take qkv_proj's output in fp16 instead of fp32. On HIP the fp32 output
        # is an fp16 GEMM plus a widen (_f32_via_f16), so the fp16 buffer holds the same values; the
        # conv casts to bf16 in-kernel and accumulates in fp32. Drops a ~200 MB fp32 write per KDA
        # layer per 2K chunk (and the MALL flush it causes). Bit-exact only on the reconstruct path
        if os.environ.get("EXL3_KDA_QKV_F16", "1") != "1":
            return False
        from .quant import exl3 as _exl3
        m = self.qkv_proj
        rows = x.numel() // x.shape[-1]
        if not (_exl3._f32_via_f16 and rows >= _exl3._f32_via_f16_min_rows):
            return False
        if type(m.inner).__name__ != "LinearEXL3" or m.inner.bias is not None:
            return False
        if rows <= _exl3.AUTO_RECONSTRUCT_THRESHOLD or self.config.infer_params.no_reconstruct:
            return False
        return (
            not m.lora_a_tensors and m.pre_scale == 1.0 and m.post_scale == 1.0 and m.softcap == 0.0 and
            m.out_features == m.out_features_unpadded
        )

    def _kda_f16_gates(self, x: torch.Tensor) -> bool:
        # EXL3_KDA_F16_GATES=1: keep the KDA gate projections in fp16 (no fp32 widen of the
        # 2048x8192 g_b/f_b outputs, ~0.5 ms each). Bit-exact only where LinearFP16 already
        # computes fp32 outputs as an fp16 GEMM plus a widen (HIP, rows >= the via-f16 threshold)
        # z is then fp16: the fused gated norm must take an fp16 gate (ext.py ROCM_KNOBS["gnorm_f16g"]),
        # else it drops to the torch fallback (5.9 vs 0.5 ms per call, -3 % 4K prefill; glm-next r3)
        if os.environ.get("EXL3_KDA_F16_GATES", "1") != "1":
            return False
        from .quant import fp16 as _fp16
        if not (_fp16._f32_via_f16 and x.numel() // x.shape[-1] >= _fp16._f32_via_f16_min_rows):
            return False
        mods = (self.g_a_proj, self.g_b_proj, self.b_proj, self.f_a_proj, self.f_b_proj)
        return all(
            m.quant_type == "fp16" and type(m.inner).__name__ == "LinearFP16" and m.inner.bias is None and
            m.inner._pinned_store is None and not m.lora_a_tensors and
            m.pre_scale == 1.0 and m.post_scale == 1.0 and m.softcap == 0.0 and
            m.out_features == m.out_features_unpadded
            for m in mods
        )

    def _kda_dec_cat_ok(self) -> bool:
        ok = getattr(self, "_kda_dec_cat_ok_v", None)
        if ok is None:
            mods = (self.g_a_proj, self.g_b_proj, self.b_proj, self.f_a_proj, self.f_b_proj)
            ok = hasattr(ext, "skinny_cat") and all(
                m.quant_type == "fp16" and type(m.inner).__name__ == "LinearFP16" and m.inner.bias is None and
                m.inner._pinned_store is None and not m.lora_a_tensors and
                m.pre_scale == 1.0 and m.post_scale == 1.0 and m.softcap == 0.0 and
                m.out_features == m.out_features_unpadded and m.inner.weight.is_contiguous()
                for m in mods
            ) and self.g_a_proj.out_features == self.f_a_proj.out_features \
              and self.g_a_proj.out_features * 2 + self.b_proj.out_features <= 512
            self._kda_dec_cat_ok_v = ok
        return ok

    def _kda_lr_fused_ok(self) -> bool:
        # kda_fb_gate / kda_gb_norm preconditions that do not depend on the input (cached)
        ok = getattr(self, "_kda_lr_fused_ok_v", None)
        if ok is None:
            wg, wf = self.g_b_proj.inner.weight, self.f_b_proj.inner.weight
            hd = self.num_v_heads * self.k_head_dim
            ok = hasattr(ext, "kda_fb_gate") and hasattr(ext, "kda_gb_norm") and _FUSE_KDA_GATE and \
                os.environ.get("EXL3_FUSE_GNORM", "1") != "0" and \
                self.g_a_proj.out_features == 128 and self.f_a_proj.out_features == 128 and \
                wg.shape == wf.shape == (128, hd) and self.v_head_dim == 128 and self.k_head_dim * self.num_v_heads == hd and \
                hd % 64 == 0 and wg.data_ptr() % 16 == 0 and wf.data_ptr() % 16 == 0 and \
                self.b_proj.out_features == self.num_v_heads and \
                self.dt_bias.dtype == torch.float and self.a_log.dtype == torch.float and \
                self.dt_bias.is_contiguous() and self.a_log.is_contiguous() and self.dt_bias.numel() == hd and \
                self.norm.gate_activation == "sigmoid" and \
                self.norm.out_dtype in (torch.half, torch.float) and \
                self.norm.weight.dtype in (torch.bfloat16, torch.float) and self.norm.weight.is_contiguous() and \
                self.norm.weight.numel() == self.norm.groups * self.v_head_dim
            self._kda_lr_fused_ok_v = ok
        return ok

    def _kda_dec_cat(self, x: torch.Tensor, params: dict):
        # EXL3_KDA_DEC_CAT=1 (decode, rows <= 8, fp16 x): g_a | f_a | b in ONE skinny launch
        # (ext.skinny_cat: same per-column reduction order as the separate skinny launches, so
        # bit-identical; g_a/f_a rounded to fp16 RNE in-kernel exactly like ha). skinny_cat_w reads
        # the three weights in place (no concat copy). With EXL3_KDA_DEC_LR_FUSED=1 the g_b/f_b second
        # stages move into kda_fb_gate / kda_gb_norm: returns (None, b, None, gaf) and the caller
        # runs them. Else EXL3_KDA_DEC_LR_BATCHED=1 runs g_b/f_b as one strided-batched GEMM
        lead = x.shape[:-1]
        rows = x.numel() // x.shape[-1]
        na = self.g_a_proj.out_features
        gaf = torch.empty((2, rows, na), dtype = torch.half, device = x.device)
        b = torch.empty((*lead, self.b_proj.out_features), dtype = torch.float, device = x.device)
        if hasattr(ext, "skinny_cat_w"):
            ext.skinny_cat_w(x.view(rows, x.shape[-1]),
                             [m.inner.weight for m in (self.g_a_proj, self.f_a_proj, self.b_proj)],
                             [gaf[0], gaf[1], b])
        else:
            w = getattr(self, "kda_dec_cat_w", None)
            if w is None:
                w = torch.cat([m.inner.weight for m in (self.g_a_proj, self.f_a_proj, self.b_proj)], 1).contiguous()
                self.kda_dec_cat_w = w
            ext.skinny_cat(x.view(rows, x.shape[-1]), w, [gaf[0], gaf[1], b])
        if KDA_KNOBS["lr_fused"] and self._kda_lr_fused_ok():
            return None, b, None, gaf
        if KDA_KNOBS["lr_batched"] and self.g_b_proj.inner.weight.shape == self.f_b_proj.inner.weight.shape:
            lw = getattr(self, "kda_dec_lr_w", None)
            if lw is None:
                lw = torch.stack([self.g_b_proj.inner.weight, self.f_b_proj.inner.weight]).contiguous()
                self.kda_dec_lr_w = lw
            zf = torch.empty((2, rows, lw.shape[-1]), dtype = torch.float, device = x.device)
            ext.hgemm_batched(gaf, lw, zf)
            z = zf[0].view(*lead, lw.shape[-1])
            f = zf[1].view(*lead, lw.shape[-1])
        else:
            z = self.g_b_proj.forward(gaf[0].view(*lead, na), params)
            f = self.f_b_proj.forward(gaf[1].view(*lead, na), params)
        return z, b, f, None

    def _kda_gb_norm(self, core_attn_out: torch.Tensor, gaf: torch.Tensor, params: dict) -> torch.Tensor:
        # g_b folded into the gated norm (z only feeds the norm). Same conditions as the fused norm
        # path GatedRMSNorm.forward would take; otherwise materialize z and use the norm as before
        n = self.norm
        if core_attn_out.dtype == torch.bfloat16 and core_attn_out.is_contiguous():
            y = torch.empty_like(core_attn_out, dtype = n.out_dtype)
            ext.kda_gb_norm(core_attn_out, gaf[0], self.g_b_proj.inner.weight, n.weight, y,
                            n.rms_norm_eps, n.constant_bias, n.groups, n.gate_first, 1)
            return y
        z = self.g_b_proj.forward(gaf[0].view(*core_attn_out.shape[:-2], gaf.shape[-1]), params)
        return n.forward(core_attn_out, params, gate = z.view(core_attn_out.shape))

    def _kda_gates_f16(self, x: torch.Tensor, params: dict, small_first: bool):
        h = torch.half
        if os.environ.get("EXL3_KDA_SMALL_CAT", "0") == "1":
            # g_a | b | f_a share x: one N=320 GEMM instead of three N<=128 ones (48 vs 3x16 WGs)
            w = getattr(self, "kda_small_cat_w", None)
            if w is None:
                w = torch.cat([m.inner.weight for m in (self.g_a_proj, self.b_proj, self.f_a_proj)], 1).contiguous()
                self.kda_small_cat_w = w
            x2 = x.view(-1, x.shape[-1])
            y = torch.matmul(x2, w).view(*x.shape[:-1], w.shape[1])
            n_ga, n_b = self.g_a_proj.out_features, self.b_proj.out_features
            ga, b, fa = y[..., :n_ga], y[..., n_ga:n_ga + n_b], y[..., n_ga + n_b:]
        elif small_first:
            # x-consumers back to back while x is still in the MALL
            ga = self.g_a_proj.forward(x, params, out_dtype = h)
            b = self.b_proj.forward(x, params, out_dtype = h)
            fa = self.f_a_proj.forward(x, params, out_dtype = h)
        else:
            ga = self.g_a_proj.forward(x, params, out_dtype = h)
            b = fa = None
        z = self.g_b_proj.forward(ga, params, out_dtype = h)
        if b is None:
            b = self.b_proj.forward(x, params, out_dtype = h)
        beta = torch.sigmoid(b.float() * self.beta_scale).to(torch.bfloat16)
        if fa is None:
            fa = self.f_a_proj.forward(x, params, out_dtype = h)
        f = self.f_b_proj.forward(fa, params, out_dtype = h)
        return z, beta, f

    @override
    def _midchunk_segments(self, params: dict, bsz: int, seqlen: int, save_state: bool, save_history: bool):
        """Row bounds [0, r1, .., T] for a mid-chunk checkpoint split, or None. Only the KDA chunk path, bsz 1,
        rows % 64 == 0 (fla chunk BT) and every segment > 256 rows (same conv kernel as the unsplit call)."""
        rows = params.get("checkpoint_rows")
        if not rows or not self.kda or bsz != 1 or not save_state or save_history or params.get("midchunk_mode", 1) != 1:
            return None
        rows = sorted(set(int(r) for r in rows))
        segs = [0] + rows + [seqlen]
        if any(r % 64 for r in rows) or any(b - a <= 256 for a, b in zip(segs[:-1], segs[1:])):
            return None
        return segs


    def _midchunk_kda_ckpt(self, params: dict, bsz: int, seqlen: int, save_state: bool, save_history: bool,
                           mixed_qkv: torch.Tensor):
        """EXL3_MIDCHUNK_CKPT=2: one unsplit call; the HIP KDA kernel side-writes S at the checkpoint row, conv state
        there = the last cdim pre-conv inputs (bf16). Single row, rows % 64 == 0 only. Returns the ckpt dict or None."""
        rows = params.get("checkpoint_rows")
        if not rows or params.get("midchunk_mode") != 2 or not self.kda or bsz != 1 or not save_state or save_history:
            return None
        rows = sorted(set(int(r) for r in rows))
        cdim = self.conv_kernel_size
        # EXL3_PF_NO_TAIL=1: the job also asks for the prompt's last full page, so up to 2 rows (kernel side-writes 2)
        max_rows = 2 if os.environ.get("EXL3_PF_NO_TAIL", "0") == "1" else 1
        if len(rows) > max_rows or any(r % 64 or not cdim <= r < seqlen for r in rows):
            return None
        cks = [{
            "row": r,
            "chunk": r // 64,
            "s": torch.empty((1, self.num_v_heads, self.k_head_dim, self.v_head_dim), dtype = torch.float, device = self.device),
            "conv": mixed_qkv[0, :, r - cdim:r].to(torch.bfloat16).contiguous(),
        } for r in rows]
        return cks[0] if len(cks) == 1 else cks


    def _midchunk_kda_d2h(self, params: dict, ck: dict):
        """Async D2H of a written mode-2 checkpoint into pinned buffers on a side stream; the event is synced at
        stash time (Job.stash_midchunk)."""
        if not ck.get("ok"):
            return
        global _MIDCKPT_STREAM
        if _MIDCKPT_STREAM is None:
            _MIDCKPT_STREAM = torch.cuda.Stream(device = self.device)
        side = _MIDCKPT_STREAM
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            hs = torch.empty(ck["s"].shape, dtype = ck["s"].dtype, pin_memory = True)
            hc = torch.empty(ck["conv"].shape, dtype = ck["conv"].dtype, pin_memory = True)
            hs.copy_(ck["s"], non_blocking = True)
            hc.copy_(ck["conv"], non_blocking = True)
            ev = torch.cuda.Event()
            ev.record(side)
        ck["s"].record_stream(side)
        ck["conv"].record_stream(side)
        layer_instance = (self.layer_idx, params.get("layer_instance", 0))
        params.setdefault("midchunk_states", {}).setdefault(ck["row"], {})[layer_instance] = (hs, hc, ev)


    def _midchunk_conv_core(self, mixed_qkv, beta, g, conv_state, recurrent_state, recurrent_slots,
                            params, segs, g_cumsum):
        """Conv + KDA core as one call per segment, state chained in place through the slot. After each
        segment but the last, save GPU clones of the slices GDNLayerState.stash() copies (S, conv) in
        params["midchunk_states"][row][layer_instance]."""
        slot = params["recurrent_states"][0].slot
        layer_instance = (self.layer_idx, params.get("layer_instance", 0))
        cdim = self.conv_kernel_size
        saved = params.setdefault("midchunk_states", {})
        outs = []
        for i, (a, b) in enumerate(zip(segs[:-1], segs[1:])):
            last = i == len(segs) - 2
            qkv_s = causal_conv1d_update(
                mixed_qkv = mixed_qkv[:, :, a:b],
                conv_state = conv_state,
                recurrent_slots = recurrent_slots,
                conv1d_weight = self.conv1d_weight_flat,
                conv1d_bias = self.conv1d_bias,
                history = False,
                params = params,
            )
            if not last:
                conv_snap = conv_state[slot, :, :cdim].clone()
            outs.append(gated_delta_rule_fn(
                mixed_qkv = qkv_s,
                beta = beta[:, a:b],
                g = g[:, a:b],
                recurrent_state = recurrent_state,
                recurrent_slots = recurrent_slots,
                history = False,
                save_state = True,
                num_k_heads = self.num_k_heads,
                num_v_heads = self.num_v_heads,
                k_dim = self.k_dim,
                v_dim = self.v_dim,
                k_head_dim = self.k_head_dim,
                v_head_dim = self.v_head_dim,
                params = params,
                channelwise_g = True,
                g_cumsum = g_cumsum,
            ))
            if not last:
                saved.setdefault(b, {})[layer_instance] = (recurrent_state[slot, :1].clone(), conv_snap)
        return torch.cat(outs, dim = 1)


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.num_k_heads == 0:
            x = torch.zeros_like(x, dtype = self.out_dtype)
            if self.tp_reduce:
                params["backend"].all_reduce(x, False)
            return to2(x, out_dtype, self.out_dtype)

        bsz, seqlen, _ = x.shape
        save_history = params.get("recurrent_history", False)

        # Post load, fuse conv1d weights if needed
        if self.conv1d_weight is None:
            self.conv1d_weight = torch.cat([
                self.conv1d_q_weight,
                self.conv1d_k_weight,
                self.conv1d_v_weight,
            ], dim = 0)
            self.conv1d_q_weight = None
            self.conv1d_k_weight = None
            self.conv1d_v_weight = None
        if self.conv1d_weight_flat is None:
            self.conv1d_weight_flat = self.conv1d_weight.squeeze(1).contiguous()

        # Previous state
        rsg = params.get("recurrent_states")
        if rsg:
            recurrent_slots = get_for_device(params, "recurrent_slots", self.device)
            layer_instance = (self.layer_idx, params.get("layer_instance", 0))
            if rsg[0].exported:
                rsl = self.tp_recurrent_lookup[rsg[0].cache]
            else:
                rsl = rsg[0].cache.get_recurrent_layer(layer_instance)
            conv_state, recurrent_state = rsl.get_state_tensors()
            save_state = True
        else:
            recurrent_slots = None
            conv_state, recurrent_state = None, None
            save_state = False
            save_history = False  # no SD without prior state, for simplicity

        # Deferred fill of the merged b/a projection (weights are materialized by now)
        if self.bc_split and not self.ba_weight_filled and self.kda:
            self.dt_bias_bc.copy_(self.dt_bias)
            self.kda_b_t.copy_(self.b_proj.inner.get_weight_tensor().T)
            self.kda_fa_t.copy_(self.f_a_proj.inner.get_weight_tensor().T)
            self.kda_fb_t.copy_(self.f_b_proj.inner.get_weight_tensor().T)
            self.kda_ga_t.copy_(self.g_a_proj.inner.get_weight_tensor().T)
            self.kda_gb_t.copy_(self.g_b_proj.inner.get_weight_tensor().T)
            self.ba_weight_filled = True
        elif self.bc_split and not self.ba_weight_filled:
            self.ba_weight_t.copy_(torch.cat([
                self.b_proj.inner.get_weight_tensor(),
                self.a_proj.inner.get_weight_tensor(),
            ], dim = -1).T)
            if self.ba_bias is not None:
                nv = self.num_v_heads
                b_bias = self.b_proj.inner.get_bias_tensor()
                a_bias = self.a_proj.inner.get_bias_tensor()
                if b_bias is None: b_bias = torch.zeros(nv, dtype = torch.half, device = self.device)
                if a_bias is None: a_bias = torch.zeros(nv, dtype = torch.half, device = self.device)
                self.ba_bias.copy_(torch.cat([b_bias, a_bias]))
            self.ba_weight_filled = True

        # Fused C++ path for decode with split projections, generalized over (bsz, seqlen) up to
        # (_BC_MAX_BSZ, _BC_MAX_QLEN) and over save_history (needed for MTP draft/verify). Runs
        # the entire layer in one call, replayed through an internal CUDA graph per (bsz, seqlen,
        # history) shape from the third invocation of that shape on
        if (
            self.bc_split and save_state and
            recurrent_slots is not None and
            1 <= bsz <= _BC_MAX_BSZ and 1 <= seqlen <= _BC_MAX_QLEN
        ):
            if self.bc.needs_configure(bsz, seqlen, save_history):
                if self.kda:
                    self._bc_configure_slot_kda(bsz, seqlen, save_history)
                else:
                    self._bc_configure_slot(bsz, seqlen, save_history)
            y = torch.empty_like(x, dtype = self.out_dtype or torch.half)
            self.bc.run_bszN(x, y, conv_state, recurrent_state, recurrent_slots, save_history)
            if self.tp_reduce:
                params["backend"].all_reduce(y)
            return to2(y, out_dtype, self.out_dtype)

        # Torch path
        # Qwen3.5 uses split projections (in_proj_qkv/in_proj_z/in_proj_b/in_proj_a),
        # while Qwen3-Next uses fused projections. The fused C++ helper expects the
        # packed layout used by fused projections; applying it to split qkv tensors
        # causes incorrect head ordering and broken generations.
        g_cumsum = False
        lr_gaf = None
        if self.qkvz_proj is not None and self.ba_proj is not None:
            qkvz = self.qkvz_proj.forward(x, params)
            ba = self.ba_proj.forward(x, params)

            mixed_qkv = torch.empty((bsz, self.fdim_qkv, seqlen), dtype = torch.bfloat16, device = self.device)
            z = torch.empty((bsz, seqlen, self.num_v_heads, self.v_head_dim), dtype = torch.bfloat16, device = self.device)
            beta = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = self.device)
            g = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.float, device = self.device)

            ext.gated_delta_net_fused_op(
                qkvz, ba,
                self.dt_bias,
                self.a_log,
                mixed_qkv, z, beta, g,
                self.num_k_heads,
                self.num_v_heads,
                self.k_head_dim,
                self.v_head_dim,
                self.beta_scale
            )
        elif self.kda:
            # EXL3_KDA_SMALL_FIRST=1: run the small x-consumers (g_a, b, f_a) before qkv_proj.
            # qkv_proj writes a ~200 MB output that evicts x from the 32 MB MALL; the small-N
            # GEMMs (16 workgroups, DRAM-latency bound) then take 1.0-1.5 ms instead of 0.2 ms.
            # Same kernels, same inputs: bit-exact. End-to-end gain alone was noise (+0.1%, step 3 A/B)
            small_first = os.environ.get("EXL3_KDA_SMALL_FIRST", "1") == "1"
            qkv_dtype = torch.half if self._kda_qkv_f16(x) else None
            if not small_first:
                qkv = self.qkv_proj.forward(x, params, qkv_dtype)

            if self._kda_f16_gates(x):
                z, beta, f = self._kda_gates_f16(x, params, small_first)
                b = None
            else:
                # Low-rank sigmoid output gate stands in for z (applied by the gated norm)
                # r35: at decode (rows <= 8, half x) the skinny hgemm rounds its fp32 sum straight to
                # fp16 (RNE), identical to fp32 out + .to(half), minus one copy launch per projection
                ha = torch.half if (KDA_KNOBS["half_a"] and x.dtype == torch.half and
                                    x.numel() // x.shape[-1] <= 8) else None
                beta = None
                if ha is not None and KDA_KNOBS["dec_cat"] and self._kda_dec_cat_ok():
                    z, b, f, lr_gaf = self._kda_dec_cat(x, params)
                else:
                    z = self.g_b_proj.forward(self.g_a_proj.forward(x, params, ha).to(torch.half), params)
                    b = self.b_proj.forward(x, params)
                    f = self.f_b_proj.forward(self.f_a_proj.forward(x, params, ha).to(torch.half), params)
            if z is not None:
                z = z.view(bsz, seqlen, self.num_v_heads, self.v_head_dim)

            if small_first:
                qkv = self.qkv_proj.forward(x, params, qkv_dtype)
            # EXL3_KDA_FUSED_CAST=1 (default): hand the conv the fp16 (bsz, f, seq) view; it casts
            # to bf16 in-kernel (bit-equal), saving a full transpose-cast pass over qkv
            if os.environ.get("EXL3_KDA_FUSED_CAST", "1") != "0":
                mixed_qkv = qkv.transpose(1, 2)
            else:
                mixed_qkv = qkv.transpose(1, 2).to(torch.bfloat16).contiguous()

            # Per-k-channel log decay from the low-rank forget gate: "safe gate" form when a
            # lower bound is configured, else softplus (per the Kimi/GLM5.3 reference).
            # f may be fp16 (EXL3_KDA_F16_GATES): fp16 + fp32 promotes in one fp32 kernel
            # EXL3_KDA_G_FUSED=1 (default, +2.0% 4K prefill, bit-exact): when the fla chunk path will run
            # (prefill, no history), one Triton pass computes the gate AND fla's chunk-local cumsum
            # (bit-equal to the torch chain + chunk_local_cumsum; kda_gate.py uses BS=32 tiles for that).
            # Saves ~2.9 ms per KDA layer per 2048-row chunk (3.5 -> 0.6 ms, scratch/dg/s5_mb.py)
            g_cumsum = (
                os.environ.get("EXL3_KDA_G_FUSED", "1") == "1"
                and seqlen >= self.num_v_heads and not save_history
            )
            if lr_gaf is not None:
                # EXL3_KDA_DEC_LR_FUSED: f = f_a @ f_b and the gate in one launch (kda_gate math)
                g_cumsum = False
                beta = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = x.device)
                g = torch.empty((bsz, seqlen, self.num_v_heads, self.k_head_dim), dtype = torch.float, device = x.device)
                lb = self.gate_lower_bound
                ext.kda_fb_gate(lr_gaf[1], self.f_b_proj.inner.weight, b, self.dt_bias, self.a_log,
                                float(self.beta_scale), float(lb or 0.0), lb is not None, beta, g)
            elif g_cumsum:
                if beta is None:
                    beta = torch.sigmoid(b.float() * self.beta_scale).to(torch.bfloat16)
                from .gated_delta_net_fn.kda_gate import kda_gate
                from ..vendor.fla import RCP_LN2
                g = kda_gate(
                    f.contiguous(), self.dt_bias.float(), torch.exp(self.a_log.float()),
                    self.num_v_heads, self.k_head_dim, self.gate_lower_bound,
                    cumsum_chunk = 64, cumsum_scale = RCP_LN2,
                )
            elif beta is None and _FUSE_KDA_GATE and b.dtype == torch.float and f.dtype == torch.float \
                    and self.dt_bias.dtype == torch.float and self.a_log.dtype == torch.float \
                    and b.is_contiguous() and f.is_contiguous():
                # One launch for beta and g (fused_elt_rocm.cu kda_gate), same per-element math
                beta = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = f.device)
                g = torch.empty((bsz, seqlen, self.num_v_heads, self.k_head_dim), dtype = torch.float, device = f.device)
                lb = self.gate_lower_bound
                ext.kda_gate(b, f, self.dt_bias, self.a_log, float(self.beta_scale),
                             float(lb or 0.0), lb is not None, beta, g)
            else:
                if beta is None:
                    beta = torch.sigmoid(b.float() * self.beta_scale).to(torch.bfloat16)
                gf = (f + self.dt_bias.float().view(1, 1, -1)) \
                    .view(bsz, seqlen, self.num_v_heads, self.k_head_dim)
                decay = torch.exp(self.a_log.float()).view(1, 1, self.num_v_heads, 1)
                if self.gate_lower_bound is not None:
                    g = self.gate_lower_bound * torch.sigmoid(decay * gf)
                else:
                    g = -decay * torch.where(gf > 20.0, gf, torch.log1p(torch.exp(gf)))
        else:
            if getattr(self, "multi_qkvz", None) is not None and bsz * seqlen <= 32:
                qkv, z = self.project_qkvz_sliced(x, bsz, seqlen)
            else:
                qkv = self.qkv_proj.forward(x, params)
                z = self.z_proj.forward(x, params)
            z = z.view(bsz, seqlen, self.num_v_heads, self.v_head_dim)
            b = self.b_proj.forward(x, params)
            a = self.a_proj.forward(x, params)

            mixed_qkv = qkv.transpose(1, 2).to(torch.bfloat16).contiguous()

            beta = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.bfloat16, device = self.device)
            g = torch.empty((bsz, seqlen, self.num_v_heads), dtype = torch.float, device = self.device)

            ext.gated_delta_net_fused_op_2(
                b, a,
                self.dt_bias,
                self.a_log,
                beta, g,
                self.beta_scale
            )

        # Mid-chunk recurrent checkpoints (EXL3_MIDCHUNK_CKPT, set by the job via params): split conv + KDA
        # core at the given rows, chaining conv state and S, and save the state at each split row
        segs = self._midchunk_segments(params, bsz, seqlen, save_state, save_history)
        if segs is not None:
            core_attn_out = self._midchunk_conv_core(
                mixed_qkv, beta, g, conv_state, recurrent_state, recurrent_slots, params, segs, g_cumsum
            )
        else:
            ck = self._midchunk_kda_ckpt(params, bsz, seqlen, save_state, save_history, mixed_qkv)
            if ck is not None:
                params["kda_ckpt"] = ck
            # Convolution. EXL3_PF_GLUE=1 (default off, bit-exact): KDA chunk path writes q | k | v planes
            plane = (self.k_dim if os.environ.get("EXL3_PF_GLUE", "0") == "1" and self.kda
                     and self.k_dim == self.v_dim and not save_history and seqlen >= self.num_v_heads else 0)
            mixed_qkv = causal_conv1d_update(
                mixed_qkv = mixed_qkv,
                conv_state = conv_state,
                recurrent_slots = recurrent_slots,
                conv1d_weight = self.conv1d_weight_flat,
                conv1d_bias = self.conv1d_bias,
                history = save_history,
                params = params,
                plane = plane,
            )

            # Delta rule
            core_attn_out = gated_delta_rule_fn(
                mixed_qkv = mixed_qkv,
                beta = beta,
                g = g,
                recurrent_state = recurrent_state,
                recurrent_slots = recurrent_slots,
                history = save_history,
                save_state = save_state,
                num_k_heads = self.num_k_heads,
                num_v_heads = self.num_v_heads,
                k_dim = self.k_dim,
                v_dim = self.v_dim,
                k_head_dim = self.k_head_dim,
                v_head_dim = self.v_head_dim,
                params = params,
                channelwise_g = self.kda,
                g_cumsum = g_cumsum,
            )
            if ck is not None:
                params.pop("kda_ckpt")
                for c in (ck if isinstance(ck, list) else [ck]):
                    self._midchunk_kda_d2h(params, c)

        # Norm
        if lr_gaf is not None:
            core_attn_out = self._kda_gb_norm(core_attn_out, lr_gaf, params)
        else:
            core_attn_out = self.norm.forward(core_attn_out, params, gate = z)
        core_attn_out = core_attn_out.view(bsz, seqlen, self.num_v_heads * self.v_head_dim)

        # Output projection
        x = self.o_proj.forward(core_attn_out, params)

        # TP reduction
        if self.tp_reduce:
            params["backend"].all_reduce(x)

        return to2(x, out_dtype, self.out_dtype)


    @override
    def get_tensors(self):
        t = super().get_tensors()
        for x, k in [
            (self.a_log, self.key_a_log),
            (self.dt_bias, self.key_dt_bias),
            (self.conv1d_weight, self.key_conv1d_weight),
            (self.conv1d_bias, self.key_conv1d_bias),
        ]:
            if x is not None:
                t[k] = x
        return t


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        assert self.qkv_proj is not None
        assert self.z_proj is not None
        assert self.b_proj is not None
        assert self.a_proj is not None
        storage = 0
        storage += self.qkv_proj.storage_size()
        storage += self.z_proj.storage_size()
        storage += self.b_proj.storage_size()
        storage += self.a_proj.storage_size()
        for cl in self.recurrent_layers:
            storage += cl.storage_size()
        overhead_d = 0
        overhead_d += self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = 0
        overhead_s += 2 * self.num_k_heads * self.k_head_dim * torch.half.itemsize
        overhead_s += 2 * self.num_v_heads * self.v_head_dim * torch.half.itemsize
        recons = max(
            self.qkv_proj.recons_size(),
            self.z_proj.recons_size(),
        )
        channel_width = 1
        channels_to_split = self.num_k_heads
        assert self.num_v_heads % self.num_k_heads == 0, \
            "num_k_heads doesn't divide num_v_heads"
        while channel_width * self.k_head_dim < 128:
            assert channels_to_split % 2 == 0, \
                "Model's K/V heads cannot divide into 128-channel tensors"
            channel_width *= 2
            channels_to_split //= 2
        assert (channel_width * self.k_head_dim) % 128 == 0 and (channel_width * self.v_head_dim) % 128 == 0, \
            "Model's K/V heads cannot divide into 128-channel tensors"
        tpa = TPAllocation(
            key = self.key,
            channel_width = channel_width,
            channel_unit = "K-heads",
            storage_per_device = 0,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = channels_to_split,
            limit_key = "linear_attn"
        )
        return [tpa]


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            if child is None:
                return None
            if isinstance(child, torch.Tensor):
                return TPTensorWrapper.tp_export(child, plan, producer)
            else:
                return child.tp_export(plan, producer)

        return {
            "cls": GatedDeltaNet,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "hidden_size": self.hidden_size,
                "k_head_dim": self.k_head_dim,
                "v_head_dim": self.v_head_dim,
                "rms_norm_eps": self.rms_norm_eps,
                "conv_kernel_size": self.conv_kernel_size,
                "beta_scale": self.beta_scale,
                "out_dtype": self.out_dtype,
            },
            "num_k_heads": self.num_k_heads,
            "num_v_heads": self.num_v_heads,
            "num_kv_group": self.num_v_heads // self.num_k_heads,
            **{name: _export(getattr(self, name, None)) for name in (
                "qkv_proj",
                "z_proj",
                "b_proj",
                "a_proj",
                "o_proj",
                "norm",
                "conv1d_weight",
                "conv1d_bias",
                "a_log",
                "dt_bias",
            )},
            "device": self.device,
            "recurrent_layers": [
                rl.tp_export(plan) for rl in self.recurrent_layers
            ]
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        key = exported["kwargs"]["key"]
        k_head_dim = exported["kwargs"]["k_head_dim"]
        v_head_dim = exported["kwargs"]["v_head_dim"]
        G = exported["num_kv_group"]
        global_num_k_heads = exported["num_k_heads"]
        global_num_v_heads = exported["num_v_heads"]
        device = local_context["device"]
        first, last, unit = plan[key]
        assert unit == "K-heads"
        num_k_heads = last - first
        num_v_heads = (last - first) * G

        q_split = (True, first * k_head_dim, last * k_head_dim) \
            if num_k_heads else None
        k_split = (True, (global_num_k_heads + first) * k_head_dim, (global_num_k_heads + last) * k_head_dim) \
            if num_k_heads else None
        v_split = (True, (global_num_k_heads * 2 + first * G) * v_head_dim, (global_num_k_heads * 2 + last * G) * v_head_dim) \
            if num_k_heads else None
        z_split = (True, first * v_head_dim * G, last * v_head_dim * G) \
            if num_k_heads else None
        o_split = (False, first * v_head_dim * G, last * v_head_dim * G) \
            if num_k_heads else None
        a_split = (True, first * G, last * G) \
            if num_k_heads else None
        b_split = (True, first * G, last * G) \
            if num_k_heads else None

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        def _import_split(name, split):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split(local_context, exported[name], plan, split) \
                if split and exported.get(name) else None

        def _import_split_3(name, split_0, split_1, split_2):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split_3(local_context, exported[name], plan, split_0, split_1, split_2) \
                if split_0 and exported.get(name) else None

        module = GatedDeltaNet(
            config = None,
            **exported["kwargs"],
            num_k_heads = num_k_heads,
            num_v_heads = num_v_heads,
            conv1d_weight = _import_split_3("conv1d_weight", q_split, k_split, v_split),
            conv1d_bias = _import_split_3("conv1d_bias", q_split, k_split, v_split),
            qkv_proj = _import_split_3("qkv_proj", q_split, k_split, v_split),
            z_proj = _import_split("z_proj", z_split),
            o_proj = _import_split("o_proj", o_split),
            b_proj = _import_split("b_proj", b_split),
            a_proj = _import_split("a_proj", a_split),
            norm = _import("norm"),
            a_log = _import_split("a_log", a_split),
            dt_bias = _import_split("dt_bias", a_split),
        )

        if num_k_heads:
            recurrent_layers = exported["recurrent_layers"]
            if len(recurrent_layers):
                module.has_split_cache = True
                for rl in exported["recurrent_layers"]:
                    rli = rl["cls"](module, **rl["args"])
                    module.recurrent_layers.append(rli)
                    module.tp_recurrent_lookup[rl["args"]["cache_id"]] = rli

        module.device = device
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True

        module.load_local(device)
        torch.cuda.synchronize()
        return module
