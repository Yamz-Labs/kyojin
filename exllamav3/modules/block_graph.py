"""Bounded per-block HIP graph capture for ROCm decode.

Opt-in via EXL3_BLOCK_GRAPH=1 (default off; deployment defaults unchanged). Scope: one graph
per (device, layer instance, (B, Q), recurrent-history mode,
cache generation) slot for non-QSA GDN+MoE TransformerBlocks. QSA layers, the PLE layer, the
epilogue (mixer + lm_head), TP modes and prefill decline and stay eager. MLA blocks (hc or the
plain-residual MTP draft head, EXL3_BLOCK_GRAPH_DRAFT) run piecewise, see _mla_forward.

Contract:
- The block's forward mutates the fp32 stream stack in place, so the graph uses a static
  input buffer (copy-in before replay, clone out after). Downstream semantics are unchanged.
- Recurrent slot ids change as data, not addresses: a static device copy of the recurrent
  slots tensor is refreshed before every replay; conv/recurrent state tensors are
  cache-owned and stable for the cache lifetime.
- Warmups are completed logical calls whose outputs are used; capture records without
  executing; the capture call then replays once, committing exactly one state transition.
- Graphs on a device share one private pool. Replays are sequential on the decode stream in
  invariant block order, which is the documented safe sharing pattern.
- A failed capture declines the slot; by default the error is raised, since a capture error
  may mean captured-region work already ran. A failed
  replay disables the slot and raises: replay failure is not automatically safe to retry
  eagerly.
"""

from __future__ import annotations

import os
from collections import Counter

import torch

from . import ablit_runtime

from torch.utils._pytree import tree_flatten, tree_unflatten

from .block_sparse_mlp import BlockSparseMLP
from .gated_delta_net import GatedDeltaNet
from .mla_attn import MLAttention, MAX_DECODE_QLEN as _MLA_MAX_QLEN

_env_int = lambda name, default: int(os.environ.get(name, default) or default)

BLOCK_GRAPH_ENABLED = os.environ.get("EXL3_BLOCK_GRAPH", "0") == "1"
BLOCK_GRAPH_WARMUPS = max(1, _env_int("EXL3_BLOCK_GRAPH_WARMUPS", 3))
BLOCK_GRAPH_MAX_SLOTS = max(1, _env_int("EXL3_BLOCK_GRAPH_MAX_SLOTS", 4))

# Rows above the dense GEMV cap reconstruct+hgemm; still captureable, but the first bounded
# implementation keeps the served decode envelope. Revisit after (B,Q) family measurements.
BLOCK_GRAPH_MAX_ROWS = _env_int("EXL3_BLOCK_GRAPH_MAX_ROWS", 16)
BLOCK_GRAPH_DEC_MOE = os.environ.get("EXL3_BLOCK_GRAPH_DEC_MOE", "1") == "1"
# R-row verify blocks (MTP/DFlash, union MoE on the device table): EXL3_BLOCK_GRAPH_VERIFY=1 graphs
# them. Off by default: at R=2 on the test box it replays but runs ~0.5 ms slower than eager (glm-mtp step 2b).
# Module global: a harness may flip it between arms.
BLOCK_GRAPH_VERIFY = os.environ.get("EXL3_BLOCK_GRAPH_VERIFY", "0") == "1"
BLOCK_GRAPH_VERIFY_MAX_ROWS = 8  # MOE_R_MAX
BLOCK_GRAPH_DEBUG = os.environ.get("EXL3_BLOCK_GRAPH_DEBUG", "0") == "1"
# Piecewise capture of MLA/DSA blocks (GLM-5.3 DSA layers), see _mla_forward. The attention core
# (cache append, indexer keys/planes, top-k, sparse/dense attention) sizes its launches from host
# context lengths every step, so it stays eager between two captured segments:
#   1 = post segment only (W_UV unfold + o_proj + attn hc apply + the whole MLP half) and the
#       attn hc mix + norm; 2 = also the q/kv projections, norms and W_UK absorb (needs no rope,
#       no l4 scale). 0 = MLA blocks stay fully eager (previous behaviour).
BLOCK_GRAPH_MLA = _env_int("EXL3_BLOCK_GRAPH_MLA", 0)
# glm-mtp step 10: the plain-residual MLA+MoE block (GLM-5.3 MTP draft head, no hc) through the same
# piecewise path when BLOCK_GRAPH_MLA is on. 0 = eager (previous behaviour), 1 = batch-1 draft rows,
# 2 = also the R-row fused catch-up forward (dflash_verify, union MoE on the device table). Replay
# is bitwise equal to eager but gains nothing end to end (step 10: dfwd_R2 2.72 -> 2.68 ms, t/s
# within noise): the draft layer is kernel-bound, not host-bound, so the default stays eager.
# Module global: a harness may flip it between arms (call purge()).
BLOCK_GRAPH_DRAFT = _env_int("EXL3_BLOCK_GRAPH_DRAFT", 0)
# r35: one shared stream buffer per (device, shape, dtype) serves as every graphed block's
# static input. A block's graph updates it in place, so the next graphed block finds its
# input already in place: no copy-in, no output clone (2 copyBuffer launches per block).
# Runtime-mutable; call purge() after toggling (captured graphs bake in the buffer address).
from .quant.exl3 import VERIFY_FUSE as _VF   # verifyfuse1: fused and unfused verify never share a graph
BG_KNOBS = {"shared_x": os.environ.get("EXL3_BG_SHARED_X", "1") != "0"}
_shared_x: dict = {}


def _shared_buf(x: torch.Tensor) -> torch.Tensor:
    k = (x.device.index, tuple(x.shape), x.dtype)
    b = _shared_x.get(k)
    if b is None:
        b = _shared_x[k] = torch.empty_like(x)
    return b


_devices_env = os.environ.get("EXL3_BLOCK_GRAPH_DEVICES")
BLOCK_GRAPH_DEVICES = ({int(d) for d in _devices_env.split(",") if d.strip()}
                       if _devices_env else None)
BLOCK_GRAPH_MAX_TOTAL = _env_int("EXL3_BLOCK_GRAPH_MAX_TOTAL", 0)  # 0 = unlimited

# One shared private pool per device across every captured block graph. Captures and replays
# are strictly sequential on the decode stream in invariant fwd-module order, which is the
# sharing order torch's graph pools are documented for (same pattern as batch capture).
# EXL3_BLOCK_GRAPH_SHARED_POOL=0 gives every graph its own pool (isolation/debugging).
BLOCK_GRAPH_SHARED_POOL = os.environ.get("EXL3_BLOCK_GRAPH_SHARED_POOL", "1") == "1"
# torch.cuda.graph lazily creates ONE class-global default capture stream on the first device
# used (torch/cuda/graphs.py default_capture_stream) and reuses it for captures on every other
# device, so multi-device capture must pass an explicit per-device stream. Set
# EXL3_BLOCK_GRAPH_EXPLICIT_STREAM=0 to reproduce the cross-device stream trap for diagnosis.
BLOCK_GRAPH_EXPLICIT_STREAM = os.environ.get("EXL3_BLOCK_GRAPH_EXPLICIT_STREAM", "1") == "1"
_capture_streams: dict[int, object] = {}
# Capture-time failures can mean escaped execution mutated state (Python side effects run
# during capture; wrongly-streamed kernels may run for real). Fail closed by default; the old
# decline-to-eager behavior stays available for diagnosis via =eager.
BLOCK_GRAPH_CAPTURE_FALLBACK = os.environ.get("EXL3_BLOCK_GRAPH_CAPTURE_FALLBACK", "closed")
_shared_pools: dict[int, object] = {}
_registry: list["BlockGraphRunner"] = []
_total_captures = [0]


def _capture_stream_for(device_index: int):
    """One retained capture stream per device, passed explicitly to torch.cuda.graph."""
    stream = _capture_streams.get(device_index)
    if stream is None:
        stream = _capture_streams[device_index] = torch.cuda.Stream(device=device_index)
    return stream


def _pool_for(device_index: int):
    if not BLOCK_GRAPH_SHARED_POOL:
        return None
    pool = _shared_pools.get(device_index)
    if pool is None:
        pool = _shared_pools[device_index] = torch.cuda.graph_pool_handle()
    return pool


class _BlockGraphSlot:
    __slots__ = ("graph", "x_in", "slots_dev", "last_used", "slots_src")

    def __init__(self, graph, x_in, slots_dev, stamp):
        self.graph = graph
        self.x_in = x_in
        self.slots_dev = slots_dev
        self.last_used = stamp
        self.slots_src = None


def _verify_union_dev_ok(x: torch.Tensor, params: dict, draft: bool = False) -> bool:
    """R-row DFlash/MTP verify through exl3_dec_moe_union with the device-built unique table
    (EXL3_DEC_MOE_UNION_DEV=1, REPORT-28): no host sync, so the block is capturable. The host
    table path (UNION_DEV=0) copies the picks to the CPU and must stay eager."""
    rows = x.shape[0] * x.shape[1]
    return ((BLOCK_GRAPH_DRAFT >= 2 if draft else BLOCK_GRAPH_VERIFY) and
            1 < rows <= BLOCK_GRAPH_VERIFY_MAX_ROWS and
            bool(params.get("dflash_verify")) and
            os.environ.get("EXL3_DEC_MOE_UNION_DEV", "0") != "0")


class BlockGraphRunner:
    """Owns the graph slots of one TransformerBlock. Attached lazily on first eligible decode
    call; all decline paths return None and the caller falls through to the eager forward."""

    def __init__(self, block):
        self.block = block
        self.slots: dict[tuple, _BlockGraphSlot] = {}
        self.warmups_left: dict[tuple, int] = {}
        self.disabled: set[tuple] = set()
        self.stamp = 0
        self.stats = {"captures": 0, "replays": 0, "warmups": 0, "declines": Counter(),
                      "capture_errors": Counter()}
        self.segs: dict[tuple, tuple] = {}   # piecewise segments (MLA blocks)
        self._static_ids: set[int] = set()

    # -- eligibility -----------------------------------------------------------

    def slot_key(self, x: torch.Tensor, params: dict):
        """Specialization key, or None when this call must not be graphed."""
        block = self.block
        attn = block.attn
        mlp = block.mlp
        if not torch.version.hip:
            return None
        if params.get("prefill") or params.get("reconstruct") or \
                params.get("activate_all_experts") or params.get("autosplit_measure") or \
                params.get("quant_preserve") is not None or "capture" in params:
            self.stats["declines"]["call_mode"] += 1
            return None
        if isinstance(attn, MLAttention) and BLOCK_GRAPH_MLA:
            return self._mla_key(x, params)
        if not isinstance(attn, GatedDeltaNet) or attn.bc is not None:
            self.stats["declines"]["attn_type"] += 1
            return None
        if getattr(attn, "qsa_indexer", None) is not None:
            self.stats["declines"]["qsa"] += 1
            return None
        if not self._mlp_io_ok(x, params):
            return None
        bsz, seqlen = x.shape[0], x.shape[1]
        num_v_heads = getattr(attn, "num_v_heads", None)
        if num_v_heads is not None and seqlen >= num_v_heads:
            # The chunk-rule path consumes slot ids host-side at capture time, so the
            # replay-time device-side slots refresh cannot retarget it. Capture stays on
            # the recurrent-kernel path, which reads slot ids from device memory.
            self.stats["declines"]["chunk_path"] += 1
            return None
        rsg = params.get("recurrent_states")
        if not rsg or rsg[0].exported:
            self.stats["declines"]["recurrent_state"] += 1
            return None
        recurrent_slots = params.get("recurrent_slots")
        if recurrent_slots is None or recurrent_slots.shape[0] != bsz:
            self.stats["declines"]["recurrent_slots"] += 1
            return None
        history = bool(params.get("recurrent_history", False))
        return (x.device.index, bsz, seqlen, history, id(rsg[0].cache),
                params.get("layer_instance", 0),
                tuple(x.shape), str(x.dtype), bool(params.get("dflash_verify")), _VF["on"])

    def _mlp_io_ok(self, x: torch.Tensor, params: dict, draft: bool = False) -> bool:
        """Checks shared by every captured block type: MLP route, TP, device, io, rows. `draft`:
        plain-residual MLA block (MTP head), residual (bsz, seq, hidden) instead of an hc stack."""
        attn = self.block.attn
        mlp = self.block.mlp
        # Batch-1 decode MoE (exl3_dec_router + exl3_dec_moe, GLM-5.3: E = 288, K 2): both launch
        # sync-free on device tables, so a rows == 1 block is capturable (REPORT-31)
        dec_moe_ok = BLOCK_GRAPH_DEC_MOE and getattr(mlp, "support_dec_moe", False) and \
            x.dim() >= 2 and (x.shape[0] * x.shape[1] == 1 or _verify_union_dev_ok(x, params, draft))
        if not isinstance(mlp, BlockSparseMLP) or \
                not (getattr(mlp, "support_hip_grouped", False) or
                     getattr(mlp, "support_hip_prefill", False) or dec_moe_ok):
            self.stats["declines"]["mlp_route"] += 1
            return False
        if getattr(mlp, "tp_mode", None) is not None or getattr(attn, "tp_reduce", False) or \
                getattr(mlp, "tp_reduce", False):
            self.stats["declines"]["tp"] += 1
            return False
        if x.device != torch.device(self.block.device) or x.device.type != "cuda":
            self.stats["declines"]["device"] += 1
            return False
        if x.dtype != torch.float32 or x.dim() != (3 if draft else 4) or not x.is_contiguous():
            self.stats["declines"]["io"] += 1
            return False
        bsz, seqlen = x.shape[0], x.shape[1]
        rows = bsz * seqlen
        if not (1 <= rows <= BLOCK_GRAPH_MAX_ROWS):
            self.stats["declines"]["rows"] += 1
            return False
        if BLOCK_GRAPH_DEVICES is not None and x.device.index not in BLOCK_GRAPH_DEVICES:
            self.stats["declines"]["device_filter"] += 1
            return False
        if os.environ.get("EXL3_MOE_SYNC_FREE_COUNT", "1") == "0":
            # torch.bincount in the generic MoE branch issues a capture-rejected H2D even when
            # the gfx12 prefill route is eligible; capture requires the sync-free histogram.
            self.stats["declines"]["sync_free_count_off"] += 1
            return False
        return True

    def _mla_key(self, x: torch.Tensor, params: dict):
        """Eligibility of an MLA/DSA block for the piecewise path (key prefix, or None)."""
        block = self.block
        attn = block.attn
        draft = block.attn_hc is None and block.mlp_hc is None
        if draft:
            # plain residual: the graph return skips forward()'s tail (layer scalar, to2), so
            # those must be no-ops; post-norms / residual scalars are not handled in seg_b
            ok = BLOCK_GRAPH_DRAFT >= 1 and block.attn_post_norm is None and \
                block.mlp_post_norm is None and block.attn_resid_scalar is None and \
                block.mlp_resid_scalar is None and block.layer_scalar_f is None and \
                block.out_dtype in (None, torch.float32) and block.mlp is not None
        else:
            ok = block.attn_hc is not None and block.mlp_hc is not None and \
                block.attn_post_norm is None and block.mlp_post_norm is None
        if not ok:
            self.stats["declines"]["mla_layout"] += 1
            return None
        if params.get("attn_mode") != "flash_attn" or params.get("inv_freq") is not None or \
                not params.get("causal", True) or params.get("non_causal_spans") is not None or \
                attn.has_split_cache:
            self.stats["declines"]["mla_mode"] += 1
            return None
        export = params.get("export_state_layers")
        if export and block.layer_idx in export:
            self.stats["declines"]["export_state"] += 1
            return None
        if not self._mlp_io_ok(x, params, draft):
            return None
        if x.shape[1] > _MLA_MAX_QLEN:
            self.stats["declines"]["rows"] += 1
            return None
        return (x.device.index, x.shape[0], x.shape[1], params.get("layer_instance", 0),
                tuple(x.shape), str(x.dtype), bool(params.get("dflash_verify")), _VF["on"])

    # -- capture / replay ------------------------------------------------------

    def maybe_forward(self, x: torch.Tensor, params: dict):
        key = self.slot_key(x, params)
        if key is None:
            return None
        if key in self.disabled:
            self.stats["declines"]["disabled_slot"] += 1
            return None
        if isinstance(self.block.attn, MLAttention):
            return self._mla_forward(key, x, params)

        warmups = self.warmups_left.get(key)
        if warmups is None:
            self.warmups_left[key] = BLOCK_GRAPH_WARMUPS
            self.stats["warmups"] += 1
            return None  # completed logical call, output used downstream
        if warmups > 1:
            self.warmups_left[key] = warmups - 1
            self.stats["warmups"] += 1
            return None

        slot = self.slots.get(key)
        if slot is None:
            if BLOCK_GRAPH_MAX_TOTAL and _total_captures[0] >= BLOCK_GRAPH_MAX_TOTAL:
                self.stats["declines"]["total_cap"] += 1
                return None
            slot = self._capture(key, x, params)
            if slot is None:
                return None
        return self._replay(slot, x, params)

    def _capture(self, key, x: torch.Tensor, params: dict):
        dev = x.device
        # Capture must not start while other devices have work in flight: the capture begins
        # on the block's device while the same logical step's earlier-device kernels may still
        # be queued, and hipGraph capture on ROCm has no cross-device ordering with them.
        for d in range(torch.cuda.device_count()):
            if torch.cuda.memory_allocated(d) > 0:
                torch.cuda.synchronize(d)
        torch.cuda.synchronize(dev)
        x_in = _shared_buf(x) if BG_KNOBS["shared_x"] else torch.empty_like(x)
        live_slots = params["recurrent_slots"]
        slots_cpu = live_slots.clone()  # stable CPU source; its device copy is slot-static
        slots_dev = get_static_device_copy(slots_cpu, dev)
        params_cap = dict(params)
        params_cap["recurrent_slots"] = slots_cpu
        graph = torch.cuda.CUDAGraph()
        try:
            self._capturing = True
            pool = _pool_for(dev.index)
            # Capture and replay must run under the block's device context: torch captures on
            # the current device's side stream, and kernels the extension launches into other
            # devices' streams during capture are not recorded.
            #
            # torch.cuda.graph keeps ONE class-global default capture stream (created on the
            # first device used) and would otherwise reuse it for captures on every other
            # device, recording this device's kernels onto the wrong device's stream. Always
            # pass an explicit per-device stream.
            with torch.cuda.device(dev):
                if BLOCK_GRAPH_EXPLICIT_STREAM:
                    capture_stream = _capture_stream_for(dev.index)
                else:
                    capture_stream = None  # reproduces the cross-device stream trap
                mode = os.environ.get("EXL3_BLOCK_GRAPH_CAPTURE_MODE", "thread_local")
                graph_kwargs = {"capture_error_mode": mode}
                if pool is not None:
                    graph_kwargs["pool"] = pool
                if capture_stream is not None:
                    graph_kwargs["stream"] = capture_stream
                with torch.cuda.graph(graph, **graph_kwargs):
                    if capture_stream is not None:
                        assert capture_stream.device.index == dev.index, \
                            f"capture stream on device {capture_stream.device.index}, " \
                            f"expected {dev.index}"
                        assert torch.cuda.current_stream(dev) == capture_stream
                        assert torch.cuda.is_current_stream_capturing()
                    self.block.forward(x_in, params_cap)
        except Exception as e:
            self.disabled.add(key)
            self.stats["captures"] += 1
            self.stats["capture_failed"] = self.stats.get("capture_failed", 0) + 1
            self.stats["declines"][f"capture_error:{type(e).__name__}"] += 1
            self.stats["capture_error_msg"] = repr(e)[:500]
            if BLOCK_GRAPH_CAPTURE_FALLBACK == "eager":
                # DIAGNOSTIC ONLY: a capture error may mean captured-region work already ran
                # for real (wrong-stream escapes); re-running eagerly can double-advance state.
                return None
            raise
        finally:
            self._capturing = False

        # LRU slot replacement: evict only after the device is quiescent.
        self.stamp += 1
        self._evict_to_limit(dev)
        self.slots[key] = _BlockGraphSlot(graph, x_in, slots_dev, self.stamp)
        self.stats["captures"] += 1
        _total_captures[0] += 1
        if BLOCK_GRAPH_DEBUG:
            print(f"[blockgraph] captured {self.block.key} key={key[1:4]}", flush=True)
        return self.slots[key]

    def _evict_to_limit(self, dev):
        while len(self.slots) >= BLOCK_GRAPH_MAX_SLOTS:
            oldest = min(self.slots.values(), key = lambda s: s.last_used)
            evict_key = next(k for k, v in self.slots.items() if v is oldest)
            if dev is not None:
                torch.cuda.synchronize(dev)
            del self.slots[evict_key]
            self.stats["evictions"] = self.stats.get("evictions", 0) + 1

    def _replay(self, slot: _BlockGraphSlot, x: torch.Tensor, params: dict):
        self.stamp += 1
        slot.last_used = self.stamp
        try:
            if slot.x_in.data_ptr() != x.data_ptr():
                slot.x_in.copy_(x)
            # Slot ids are immutable per slot-tuple: the live CPU tensor is _static_dev_cache
            # with a persistent per-device copy. Refresh the graph's fixed-address slots buffer
            # with a stream-ordered D2D copy only when the source object changes (job slot
            # reassignment); same-tuple continuations upload nothing (a blocking pageable H2D
            # here would drain the queue once per captured block). The held reference makes
            # object identity exact: a recycled allocation address cannot alias the source.
            live = params["recurrent_slots"]
            if live is not slot.slots_src:
                from ..util.tensor import get_for_device
                live_dev = get_for_device(params, "recurrent_slots", x.device)
                slot.slots_dev.copy_(live_dev)
                slot.slots_src = live
            with torch.cuda.device(x.device):
                slot.graph.replay()
        except Exception as e:
            # Launch-time failure: disable the slot. The step is not silently retried eagerly;
            # the exception propagates to the caller's normal failure handling.
            self.disabled.add(self._key_of(slot))
            self.stats["declines"][f"replay_error:{type(e).__name__}"] += 1
            raise
        self.stats["replays"] += 1
        if BLOCK_GRAPH_DEBUG:
            print(f"[blockgraph] replay {self.block.key}", flush=True)
        if BG_KNOBS["shared_x"] and slot.x_in is _shared_x.get(
                (x.device.index, tuple(x.shape), x.dtype)):
            return slot.x_in
        return slot.x_in.clone()

    # -- piecewise capture (MLA/DSA blocks) --------------------------------------

    def _run_seg(self, key, fn, args: tuple):
        """Run fn(*args) as a captured segment: BLOCK_GRAPH_WARMUPS eager calls, then capture
        (records without executing) and replay. Tensor args are copied into the segment's
        static inputs unless they already are those buffers (an earlier segment's outputs);
        returns the segment's static outputs, valid until its next replay."""
        seg = self.segs.get(key)
        if seg is None:
            if key in self.disabled:
                return fn(*args)
            w = self.warmups_left.get(key, BLOCK_GRAPH_WARMUPS)
            if w > 0:
                self.warmups_left[key] = w - 1
                self.stats["warmups"] += 1
                return fn(*args)
            if BLOCK_GRAPH_MAX_TOTAL and _total_captures[0] >= BLOCK_GRAPH_MAX_TOTAL:
                self.stats["declines"]["total_cap"] += 1
                return fn(*args)
            seg = self._capture_seg(key, fn, args)
            if seg is None:
                return fn(*args)
        ins, graph, outs, dev = seg
        try:
            for s_, a in zip(ins, args):
                if a is not None and s_.data_ptr() != a.data_ptr():
                    s_.copy_(a)
            with torch.cuda.device(dev):
                graph.replay()
        except Exception as e:
            self.disabled.add(key)
            self.segs.pop(key, None)
            self.stats["declines"][f"replay_error:{type(e).__name__}"] += 1
            raise
        self.stats["replays"] += 1
        return outs

    def _capture_seg(self, key, fn, args: tuple):
        dev = next(a.device for a in args if a is not None)
        torch.cuda.synchronize(dev)
        # Inputs that are already static (earlier segment outputs) are captured by address
        ins = tuple(a if a is None or id(a) in self._static_ids else a.clone() for a in args)
        graph = torch.cuda.CUDAGraph()
        try:
            self._capturing = True
            pool = _pool_for(dev.index)
            with torch.cuda.device(dev):
                kw = {"capture_error_mode": os.environ.get("EXL3_BLOCK_GRAPH_CAPTURE_MODE", "thread_local")}
                if pool is not None:
                    kw["pool"] = pool
                if BLOCK_GRAPH_EXPLICIT_STREAM:
                    kw["stream"] = _capture_stream_for(dev.index)
                with torch.cuda.graph(graph, **kw):
                    outs = fn(*ins)
        except Exception as e:
            self.disabled.add(key)
            self.stats["captures"] += 1
            self.stats["capture_failed"] = self.stats.get("capture_failed", 0) + 1
            self.stats["declines"][f"capture_error:{type(e).__name__}"] += 1
            self.stats["capture_error_msg"] = repr(e)[:500]
            if BLOCK_GRAPH_CAPTURE_FALLBACK == "eager":
                return None  # DIAGNOSTIC ONLY (see _capture)
            raise
        finally:
            self._capturing = False
        flat, _ = tree_flatten(outs)
        for t in list(ins) + flat:
            if isinstance(t, torch.Tensor):
                self._static_ids.add(id(t))
        seg = self.segs[key] = (ins, graph, outs, dev)
        self.stats["captures"] += 1
        _total_captures[0] += 1
        if BLOCK_GRAPH_DEBUG:
            print(f"[blockgraph] captured segment {self.block.key} {key[0]} {key[2:4]}", flush=True)
        return seg

    def _mla_forward(self, key, x: torch.Tensor, params: dict) -> torch.Tensor:
        """TransformerBlock.forward for an hc MLA/DSA block as captured segment A (attn hc mix +
        norm [+ projections/absorb]), the eager attention core, and captured segment B (unfold
        + o_proj + attn hc apply + MLP half). Same kernels in the same order as the eager
        forward, so outputs are bit-identical."""
        block = self.block
        attn = block.attn
        bsz, seqlen = x.shape[0], x.shape[1]
        pre_ok = BLOCK_GRAPH_MLA >= 2 and \
            (attn.rope is None or attn.qk_rope_head_dim == 0) and not attn.l4_beta

        shared = None
        if BG_KNOBS["shared_x"]:
            shared = _shared_buf(x)
            if shared.data_ptr() != x.data_ptr():
                shared.copy_(x)
            x = shared
            self._static_ids.add(id(shared))

        if block.attn_hc is None:
            return self._mla_plain_forward(key, x, params, pre_ok, shared)

        def seg_a(x_):
            fused = block.attn_hc.mix_norm(x_, params, block.attn_norm) if block.attn_norm else None
            if fused is not None:
                hc_post, hc_comb, y = fused
            else:
                hc_post, hc_comb, y = block.attn_hc.mix(x_, params)
                y = y.half()
                if block.attn_norm:
                    y = block.attn_norm.forward(y, params, out_dtype = torch.half)
            pre = attn._attend_pre(y, bsz, seqlen, params, 0, None, None, None) if pre_ok else None
            return x_, hc_post, hc_comb, y, pre

        x_s, hc_post, hc_comb, y, pre = self._run_seg(("a",) + key, seg_a, (x,))

        params["_mla_defer_post"] = True
        if pre is not None:
            params["_mla_pre"] = pre
        try:
            o_lat = attn.decode_flash_attn(y, bsz, seqlen, params)
        finally:
            params.pop("_mla_defer_post", None)
            params.pop("_mla_pre", None)

        def seg_b(x_, o_lat_, hc_post_, hc_comb_):
            yb = attn.attend_post(o_lat_, bsz, seqlen, params)
            if attn.out_dtype is not None:
                yb = yb.to(attn.out_dtype)
            if block.ablit:
                ablit_runtime.project(yb, block.ablit[0], *block.ablit[2:])
            if block.attn_resid_scalar is not None:
                yb *= block.attn_resid_scalar
            x_ = block.attn_hc.apply_(x_, yb, hc_post_, hc_comb_, params)
            return block._forward_mlp(x_, None, params)

        bkey = ("b",) + key + (tuple(o_lat.shape), str(o_lat.dtype))
        graphed = bkey in self.segs
        out = self._run_seg(bkey, seg_b, (x_s, o_lat, hc_post, hc_comb))
        graphed = graphed or bkey in self.segs
        # a replayed segment returns its static buffer, overwritten by the next replay
        if graphed and shared is not None and out.data_ptr() == shared.data_ptr() \
                and out.shape == shared.shape:
            return out  # updated in place in the shared stream buffer
        return out.clone() if graphed else out

    def _mla_plain_forward(self, key, x, params, pre_ok, shared):
        """_mla_forward for a plain-residual MLA block (MTP draft head): segment A = attn norm
        [+ projections/absorb], eager attention core, segment B = unfold + o_proj + residual
        (fused into the MLP input norm when the eager forward fuses it) + MLP half. Mirrors
        TransformerBlock.forward's no-hc branch kernel for kernel."""
        block = self.block
        attn = block.attn
        bsz, seqlen = x.shape[0], x.shape[1]

        def seg_a(x_):
            y = block.attn_norm.forward(x_, params, out_dtype = torch.half) if block.attn_norm \
                else x_.half()
            pre = attn._attend_pre(y, bsz, seqlen, params, 0, None, None, None) if pre_ok else None
            return x_, y, pre

        x_s, y, pre = self._run_seg(("a",) + key, seg_a, (x,))

        params["_mla_defer_post"] = True
        if pre is not None:
            params["_mla_pre"] = pre
        try:
            o_lat = attn.decode_flash_attn(y, bsz, seqlen, params)
        finally:
            params.pop("_mla_defer_post", None)
            params.pop("_mla_pre", None)

        def seg_b(x_, o_lat_):
            yb = attn.attend_post(o_lat_, bsz, seqlen, params)
            if attn.out_dtype is not None:
                yb = yb.to(attn.out_dtype)
            y_resid = None
            if block.mlp_norm is not None and block.mlp_norm.can_fuse_residual(x_, yb):
                y_resid = yb
            else:
                x_ += yb
            return block._forward_mlp(x_, y_resid, params)

        bkey = ("b",) + key + (tuple(o_lat.shape), str(o_lat.dtype))
        graphed = bkey in self.segs
        out = self._run_seg(bkey, seg_b, (x_s, o_lat))
        graphed = graphed or bkey in self.segs
        if graphed and shared is not None and out.data_ptr() == shared.data_ptr() \
                and out.shape == shared.shape:
            return out
        return out.clone() if graphed else out

    def _key_of(self, slot):
        return next(k for k, v in self.slots.items() if v is slot)

    _capturing = False


def get_static_device_copy(cpu_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Persistent per-device copy of a small CPU control tensor (same convention as
    _static_dev_cache in util/tensor.get_for_device): the address is stable, contents are
    updated by the caller before each replay. The source tensor is flagged _static_dev_cache
    so get_for_device resolves to this copy during capture instead of issuing an unpinned H2D
    copy node (rejected during stream capture)."""
    cpu_tensor._static_dev_cache = True
    cache = cpu_tensor.__dict__.get("_static_dev_copies")
    if cache is None:
        cache = cpu_tensor._static_dev_copies = {}
    dv = cache.get(device)
    if dv is None:
        dv = cpu_tensor.to(device)
        cache[device] = dv
    # modules may look the copy up by their device string ("cuda:0", GLM-5.3 blocks) rather than
    # the torch.device: register both keys or get_for_device issues an H2D copy inside capture
    cache.setdefault(str(torch.device(device)), dv)
    return dv


def maybe_graph_forward(block, x: torch.Tensor, params: dict):
    """Entry hook for TransformerBlock.forward. Returns a tensor on the graphed path, or None
    to decline (caller then runs the eager forward)."""
    if not BLOCK_GRAPH_ENABLED:
        return None  # dynamic check: the flag may be toggled mid-process for qualification
    runner = getattr(block, "block_graph_runner", None)
    if runner is None:
        # the verdict depends on BLOCK_GRAPH_MLA, which may be toggled mid-process: key it
        if getattr(block, "_block_graph_checked", None) == ("checked", BLOCK_GRAPH_MLA):
            return None
        block._block_graph_checked = ("checked", BLOCK_GRAPH_MLA)
        if not isinstance(block.mlp, BlockSparseMLP) or not (
            isinstance(block.attn, GatedDeltaNet) or
            (BLOCK_GRAPH_MLA and isinstance(block.attn, MLAttention))
        ):
            return None  # cheap structural pre-filter; full eligibility checked per call
        runner = block.block_graph_runner = BlockGraphRunner(block)
        _registry.append(runner)
    if runner._capturing:
        return None
    return runner.maybe_forward(x, params)


def global_stats():
    out = {"runners": len(_registry), "captures": 0, "replays": 0, "warmups": 0,
           "evictions": 0, "declines": Counter(), "capture_failed": 0}
    for r in _registry:
        out["captures"] += r.stats["captures"]
        out["replays"] += r.stats["replays"]
        out["warmups"] += r.stats["warmups"]
        out["evictions"] += r.stats.get("evictions", 0)
        out["capture_failed"] += r.stats.get("capture_failed", 0)
        out["declines"].update(r.stats["declines"])
    return out

def purge():
    """Release every live block graph (qualification harness teardown: models are reloaded
    between runs and stale graphs would otherwise hold their pools alive)."""
    for r in _registry:
        if r.slots or r.segs:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            r.slots.clear()
            r.segs.clear()
            r._static_ids.clear()
        b = r.block
        if b is not None:
            for attr in ("block_graph_runner", "_block_graph_checked"):
                if hasattr(b, attr):
                    try:
                        delattr(b, attr)
                    except Exception:
                        pass
    _registry.clear()
    _shared_pools.clear()
    _shared_x.clear()
    _total_captures[0] = 0
