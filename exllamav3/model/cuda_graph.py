"""Whole-forward CUDA graph for decode on CUDA (opt-in: EXL3_CUDA_GRAPH=1, default off).

Scope: Qwen3-MoE style models and (stage C) GatedDeltaNet hybrids
such as Qwen3_5, and LFM2 ShortConv hybrids (Embedding -> [TransformerBlock(Attention|GatedDeltaNet|ShortConv +
BlockSparseMLP|GatedMLP)]* ->
RMSNorm -> lm_head Linear), B=1, Q in 1..8, single CUDA device, no TP. Everything else declines to the
eager path with no behaviour change (this module returns None and Model.forward carries on as before).

Stage C (recurrent hybrids), per step values made static or keyed:
- recurrent slot ids: static int32 device tensor e.slots, refreshed by a stream-ordered D2D copy only when the
  live slot tensor object changes (block_graph.py logic); params["recurrent_slots"] in the graph params is e.slots.
- conv/recurrent state tensors: cache-owned, allocated once (GDNLayerState.alloc), rewind/stash copy in place, so
  their pointers are stable; the key holds id(cache) and a fingerprint of the first KV and recurrent state tensor.
- history flag (recurrent_history) and dflash_verify are in the key (they select the BC slot / state stride).
- positions (mrope): separate static buffer e.pos, refreshed per step (already handled for non-aliased positions).
- export_state_layers / export_state_norm_keys (DFlash2 and MTP verify: target hidden taps) are in the key; the
  tap tensors produced inside the capture are kept (e.exports) and cloned into params["export_states"] per replay.
- prepare_for_recurrence (slot tensor) runs in Model.prepare_inputs before this runner; advance_recurrent_states
  runs after it in Model.forward, exactly as in eager. Rewind (generator reject_remainder) is host side, after.

What is graphed: all modules after the Embedding (blocks + final norm + lm_head), from the embedding
OUTPUT to logits. The CPU embedding lookup stays eager (as today) and its result is copied into a
static device buffer. The C++ BC objects run their eager launch path while the stream is capturing
(stage A), so the outer torch.cuda.graph records plain kernel launches on static buffers.

Per capture key (see _key): static device buffers x_in, block_table, cache_seqlens (+ positions when it
is a separate tensor, e.g. mrope). The params dict used for warmup and capture holds those STATIC CUDA
tensors, so util.tensor.get_for_device returns them as they are (to_device is a no-op for same-device
tensors) and no host-to-device copy is captured. Before every replay the step's values are copy_'d into
the static buffers (stream-ordered, non_blocking when the source is pinned: the generator's staging
buffers are; it synchronizes once per step before refilling them).

Lifecycle per key: EXL3_CUDA_GRAPH_WARMUPS (2) real eager calls through the same static buffers (BC slot
configure, autotune, Triton compile happen there; their logits are returned normally), then the next
call captures (capture records, it does not execute) and replays once to commit the step. Every later
call = copy-in + replay. Host-only bookkeeping (advance_recurrent_states) stays in Model.forward.

Output lifetime: the graph's logits tensor lives in the private pool shared by all graphs of a device and
is overwritten by the next replay of ANY graph on that device. By default (EXL3_CUDA_GRAPH_CLONE=1) a
clone is returned (one small copy: Q*vocab*2 bytes); set it to 0 only when every caller consumes the
logits before the next forward on the device.

Failure policy: a capture error disables the key and raises (a capture error can mean captured work ran
for real); EXL3_CUDA_GRAPH_FALLBACK=eager declines to the eager path instead (diagnostic). A replay error
disables the key and raises. A capture is recreated when the block table width changes (every 4096
tokens at B=1, exactly the width the generator passes, so split numerics match eager); LRU cap
EXL3_CUDA_GRAPH_MAX (6). purge() releases everything (call between harness arms / model reloads).
"""

from __future__ import annotations

import os
import weakref
from collections import Counter, OrderedDict

import torch

_env_int = lambda name, default: int(os.environ.get(name, default) or default)

# Module global: a harness may flip it between arms (then call purge()).
CUDA_GRAPH_ENABLED = os.environ.get("EXL3_CUDA_GRAPH", "0") == "1"
CUDA_GRAPH_WARMUPS = max(1, _env_int("EXL3_CUDA_GRAPH_WARMUPS", 2))
# EXL3_CUDA_GRAPH_MULTI=1 (default off): several jobs per graph (B in 1..4, Q <= 16, B*Q <= 32), bigger LRU. Off =
# the B=1, Q <= 8 behaviour above, unchanged. set_multi() flips the globals for harness arms (then call purge()).
CUDA_GRAPH_MULTI = False
CUDA_GRAPH_MAX = 6
CUDA_GRAPH_MAX_ROWS = 8  # B*Q, MAX_BSZN of the BC objects
CUDA_GRAPH_MAX_B = 1
CUDA_GRAPH_MAX_Q = 8


def set_multi(on: bool):
    global CUDA_GRAPH_MULTI, CUDA_GRAPH_MAX, CUDA_GRAPH_MAX_ROWS, CUDA_GRAPH_MAX_B, CUDA_GRAPH_MAX_Q
    CUDA_GRAPH_MULTI = bool(on)
    CUDA_GRAPH_MAX = max(1, _env_int("EXL3_CUDA_GRAPH_MAX", 16 if on else 6))
    CUDA_GRAPH_MAX_ROWS = 32 if on else 8
    CUDA_GRAPH_MAX_B = 4 if on else 1
    CUDA_GRAPH_MAX_Q = 16 if on else 8


set_multi(os.environ.get("EXL3_CUDA_GRAPH_MULTI", "0") == "1")
CUDA_GRAPH_CLONE = os.environ.get("EXL3_CUDA_GRAPH_CLONE", "1") != "0"
CUDA_GRAPH_FALLBACK = os.environ.get("EXL3_CUDA_GRAPH_FALLBACK", "closed")
CUDA_GRAPH_DEBUG = os.environ.get("EXL3_CUDA_GRAPH_DEBUG", "0") == "1"

# Only these params keys may be present; anything else (export_state_layers, activate_all_experts,
# autosplit_measure, reconstruct, capture, ovr, sim_kvq, non_causal_spans, last_tokens_only, skip_logits,
# draft verifier extras, ...) means a path the graph was not designed for -> eager.
_ALLOWED = frozenset({
    "attn_mode", "block_table", "cache", "cache_seqlens", "recurrent_states", "indexed_embeddings",
    "positions", "position_ids", "position", "recurrent_history", "pinned_staging", "dflash_verify",
    "dflash_table_pages", "layer_instance", "input_ids", "dev_cache", "recurrent_slots",
    "export_state_layers", "export_state_norm_keys",
})
# NB: "swa_cs" is set by the runner on its own params copy only, never by callers

_pools: dict[int, object] = {}
_streams: dict[int, object] = {}
_registry: list = []


def _live_graphs(dev_index: int) -> int:
    n = 0
    for ref in _registry:
        r = ref()
        if r is not None and r.dev is not None and r.dev.index == dev_index:
            n += sum(1 for e in r.entries.values() if e.graph is not None)
    return n


def _pool_for(dev_index: int):
    p = _pools.get(dev_index)
    # Once every graph of a pool is released (LRU evictions while new keys are still warming up, e.g. varying
    # draft widths) torch keeps the pool registered with use_count 0 as long as a returned output still holds one
    # of its blocks, and capture_begin on that handle hits "it->second->use_count > 0" (CUDACachingAllocator).
    # A fresh handle then: same graphs, only the pool identity changes
    if p is None or _live_graphs(dev_index) == 0:
        p = _pools[dev_index] = torch.cuda.graph_pool_handle()
    return p


def _stream_for(dev_index: int):
    s = _streams.get(dev_index)
    if s is None:
        s = _streams[dev_index] = torch.cuda.Stream(device = dev_index)
    return s


class _Entry:
    __slots__ = ("graph", "out", "x_in", "bt", "cs", "pos", "params", "warm", "last_used", "cache_ref", "fp",
                 "slots", "slots_src", "exports", "swa_cs", "swa_host")


def _cache_fp(cache):
    """Address fingerprint of the cache's first layer: a reloaded/reallocated cache invalidates graphs."""
    try:
        layer = next(iter(cache.layers.values()), None)
        k = getattr(layer, "k", None)
        fp = [k.data_ptr() if torch.is_tensor(k) else 0]
        rl = getattr(cache, "recurrent_layers", None)
        if rl:
            r = next(iter(rl.values()))
            # 0 while freed (meta); ShortConv layers have conv_state only
            fp += [t.data_ptr() for t in (getattr(r, "conv_state", None), getattr(r, "recurrent_state", None))
                   if torch.is_tensor(t)]
        return tuple(fp)
    except Exception:
        return ()


def _foreign_gdn_ok(a) -> bool:
    """MiniCPM-style GDN: split foreign/fp16 projections, no KDA/TP/heads-less.
    Capture compatibility of the torch path is verified empirically (a capture
    error declines the key); this only admits the attempt."""
    from ..modules.foreign_linear import ForeignLinear
    from ..modules import Linear
    if a.kda or a.tp_reduce or a.num_k_heads == 0:
        return False
    if a.qkvz_proj is not None or a.ba_proj is not None:
        return False
    for p in (a.qkv_proj, a.z_proj, a.b_proj, a.a_proj, a.o_proj):
        if p is None:
            return False
        if type(p) is ForeignLinear:
            if p.quant_type != "foreign":
                return False
        elif type(p) is Linear:
            if p.quant_type != "fp16":
                return False
        else:
            return False
    return True


class CudaGraphRunner:

    def __init__(self, model):
        self.model = model
        self.entries: OrderedDict = OrderedDict()
        self.disabled: set = set()
        self.stats = {"captures": 0, "replays": 0, "warmups": 0, "declines": Counter()}
        self.tick = 0
        self._model_ok = None
        self.dev = None
        self.has_gdn = False
        self.has_conv = False
        self.has_mamba = False
        self.has_swa = False
        self.swa_mods = []
        _registry.append(weakref.ref(self))

    # -- eligibility ---------------------------------------------------------------------------

    def _check_model(self) -> bool:
        from ..modules import (Embedding, TransformerBlock, Attention, BlockSparseMLP, GatedMLP, RMSNorm, Linear)
        from ..modules.gated_delta_net import GatedDeltaNet
        from ..modules.short_conv import ShortConv
        from ..modules.mamba2 import Mamba2
        from ..modules.sliding_attn import SlidingAttention
        from ..modules.foreign_linear import ForeignLinear
        m = self.model
        fm = m.fwd_modules
        if m.loaded_tp or len(fm) < 3 or type(fm[0][0]) is not Embedding:
            return False
        if m._get_prefetch_layers or getattr(m.config, "moe_cpu_hosts", None):
            return False
        devs = set()
        self.has_gdn = False
        self.has_conv = False
        self.has_mamba = False
        self.has_swa = False
        self.swa_mods = []
        for mod, inst, _ in fm[1:]:
            t = type(mod)
            if t is TransformerBlock:
                if mod.attn is None:
                    pass  # MoE/norm-only block (Bolt "E"): the mlp check below applies
                elif type(mod.attn) is Mamba2:
                    # Bolt: torch-path Mamba2 over as-shipped foreign projs (the exl3
                    # BC path sets .bc and stays on its own lane). No TP, no heads-less.
                    a = mod.attn
                    if a.bc is not None or a.tp_reduce or a.num_k_heads == 0 or \
                            type(a.in_proj) is not ForeignLinear or \
                            type(a.o_proj) is not ForeignLinear:
                        return False
                    self.has_mamba = True
                    if mod.mlp is not None:
                        return False
                elif type(mod.attn) is GatedDeltaNet:
                    # only the fused BC_GatedDeltaNetSplit path (static buffers, no shape-keyed Python caches
                    # such as _qd3_ba/_qd3_bg); KDA, TP-reduced and head-less variants stay eager
                    a = mod.attn
                    if not a.bc_split or a.bc is None or a.kda or a.tp_reduce or a.num_k_heads == 0:
                        # MCPM_FQ_GRAPH=1 (MiniCPM foreign-GDN experiment): allow the
                        # torch-path GDN when every projection is as-shipped
                        # (ForeignLinear) or plain fp16, so capture can be
                        # attempted; anything else in the layer stays eager.
                        if not (os.environ.get("MCPM_FQ_GRAPH", "1") == "1" and
                                _foreign_gdn_ok(a)):
                            return False
                    self.has_gdn = True
                elif type(mod.attn) is ShortConv:
                    # LFM2: the conv slot is resolved on the host (CPU slot tensor) and baked into the capture,
                    # so the slot id is part of the key; TP-reduced variants stay eager
                    if mod.attn.tp_reduce:
                        return False
                    self.has_conv = True
                elif type(mod.attn) is SlidingAttention:
                    # sliding-window ring (Flint): host-side window shift + in-state seqlens are prepared per step
                    # by the runner (sliding_attn.swa_graph_prep) into a static buffer, block table is per slot
                    if mod.attn.tp_reduce or mod.attn.num_kv_heads == 0:
                        return False
                    self.has_swa = True
                    self.swa_mods.append((mod.attn, inst))
                elif type(mod.attn) is not Attention or getattr(mod.attn, "qsa_indexer", None) is not None:
                    return False
                if mod.mlp is None:
                    # Attn/mamba-only blocks: Bolt foreign Attention (all projs
                    # as-shipped) or Bolt Mamba2 above; anything else stays eager.
                    if mod.attn is not None and type(mod.attn) is Mamba2:
                        pass
                    elif type(mod.attn) is Attention and all(
                            type(getattr(mod.attn, k, None)) in (ForeignLinear, type(None))
                            for k in ("q_proj", "k_proj", "v_proj", "o_proj")) and \
                            any(getattr(mod.attn, k, None) is not None
                                for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
                        pass
                    else:
                        return False
                elif type(mod.mlp) not in (BlockSparseMLP, GatedMLP):
                    return False
                if mod.attn_hc is not None or mod.mlp_hc is not None or mod.ablit:
                    return False
            elif t is Linear:
                if mod.lora_a_tensors:
                    return False
            elif t is ForeignLinear:
                pass  # Bolt lm_head over as-shipped weights (no LoRA/adapters)
            elif t is not RMSNorm:
                return False
            if mod.device is None:
                return False
            devs.add(torch.device(mod.device))
        if len(devs) != 1 or (self.has_gdn and self.has_conv):
            return False   # GDN graphs pass device slot ids, ShortConv reads them on the host: no mixed model
        dev = devs.pop()
        if dev.type != "cuda" or dev.index is None:
            return False
        if not fm[-1][0].caps.get("logits_output"):
            return False
        self.dev = dev
        return True

    def _decline(self, ids, params):
        """None when this call may be graphed, else a short reason."""
        if (ids.dim() != 2 or not 1 <= ids.shape[0] <= CUDA_GRAPH_MAX_B or ids.shape[1] > CUDA_GRAPH_MAX_Q
                or ids.shape[0] * ids.shape[1] > CUDA_GRAPH_MAX_ROWS):
            return "shape"
        for k in params:
            if k not in _ALLOWED:
                return "param:" + k
        if self.has_swa and ids.shape[1] > 1:
            return "swa_multi"   # m>1 (ngram / MTP verify) sliding steps use paths that are not capture-safe (per-call tensors): eager
        bt = params.get("block_table")
        cs = params.get("cache_seqlens")
        if not torch.is_tensor(bt) or not torch.is_tensor(cs) or params.get("cache") is None or bt.dim() != 2:
            return "no_block_table"
        if params.get("indexed_embeddings"):
            return "state"
        rs = params.get("recurrent_states")
        if self.has_gdn or self.has_conv or self.has_mamba or self.has_swa:
            slots = params.get("recurrent_slots")
            nb = ids.shape[0] if CUDA_GRAPH_MULTI else 1   # one recurrent state / slot per job
            if (not rs or len(rs) != nb or any(getattr(r, "exported", False) for r in rs[:nb])
                    or not torch.is_tensor(slots) or slots.numel() != nb):
                return "state"
            if self.has_conv and slots.device.type != "cpu":
                return "state"
        elif rs:
            return "state"
        if params.get("position") or params.get("position_ids") is not None:
            return "position"
        if params.get("attn_mode", "flash_attn") != "flash_attn":
            return "attn_mode"
        if torch.cuda.is_current_stream_capturing():
            return "capturing"
        return None

    # -- entry points --------------------------------------------------------------------------

    def forward(self, ids: torch.Tensor, params: dict):
        if self._model_ok is None:
            self._model_ok = self._check_model()
        if not self._model_ok:
            self.stats["declines"]["model"] += 1
            return None
        why = self._decline(ids, params)
        if why is not None:
            self.stats["declines"][why] += 1
            return None

        fm = self.model.fwd_modules
        # Embedding stays eager, exactly as forward_ls would run it
        emb, inst0, _ = fm[0]
        params["layer_instance"] = inst0
        x = emb.prepare_for_device(ids, params)
        x = emb.forward(x, params)
        tail = fm[1:]

        key = self._key(x, params)
        if key in self.disabled:
            self.stats["declines"]["disabled_key"] += 1
            return self._run_tail(tail, x, params)

        cache = params["cache"]
        e = self.entries.get(key)
        if e is not None and (e.cache_ref() is not cache or e.fp != _cache_fp(cache)):
            self._drop(key)  # cache object recycled / reallocated: static pointers are stale
            e = None
        if e is None:
            e = self._new_entry(key, x, params)
        self.tick += 1
        e.last_used = self.tick
        self.entries.move_to_end(key)
        dev = self.dev

        self._load(e, x, params)
        if e.graph is None:
            if e.warm < CUDA_GRAPH_WARMUPS:
                e.warm += 1
                self.stats["warmups"] += 1
                # real eager call through the same static buffers; output is the step's result
                p = dict(e.params)
                y = self._run_tail(tail, e.x_in, p)
                if "export_states" in p:
                    params["export_states"] = p["export_states"]
                return y
            if not self._capture(key, e, tail, x, params):
                return self._run_tail(tail, x, params)   # FALLBACK=eager decline
        try:
            with torch.cuda.device(dev):
                e.graph.replay()
        except Exception:
            self._drop(key)
            self.disabled.add(key)
            self.stats["declines"]["replay_error"] += 1
            raise
        self.stats["replays"] += 1
        if CUDA_GRAPH_DEBUG:
            print(f"[cudagraph] replay key={key[1:5]}", flush = True)
        if e.exports is not None:
            # target hidden taps (DFlash2 / MTP verify): the generator keeps them past the next forward
            params["export_states"] = [t.clone() for t in e.exports] if CUDA_GRAPH_CLONE else list(e.exports)
        return e.out.clone() if CUDA_GRAPH_CLONE else e.out

    # -- internals -----------------------------------------------------------------------------

    @staticmethod
    def _run_tail(tail, x, p):
        # Same loop as Model.forward_ls minus the embedding (declines guarantee no last_tokens_only /
        # skip_logits / moe_cpu_hosts)
        for module, instance, idx in tail:
            p["layer_instance"] = instance
            x = module.prepare_for_device(x, p)
            x = module.forward(x, p)
        return x

    def _key(self, x, params):
        bt = params["block_table"]
        cs = params["cache_seqlens"]
        pos = params.get("positions")
        pages = params.get("dflash_table_pages")
        exl = params.get("export_state_layers")
        exn = params.get("export_state_norm_keys")
        return (
            self.dev.index, x.shape[0], x.shape[1], bt.shape[1], id(params["cache"]),
            tuple(x.shape), x.dtype, bt.dtype, cs.dtype, tuple(cs.shape),
            None if (pos is None or pos is cs) else (tuple(pos.shape), pos.dtype),
            bool(params.get("dflash_verify")), bool(params.get("recurrent_history")),
            # per-job page counts feed only the MLA decode kernel's KV split (mla_attn.py); MLAttention models
            # never get here (_check_model), and with B jobs they would recapture whenever one job crosses 16 pages
            None if CUDA_GRAPH_MULTI else (tuple(pages) if pages is not None else None),
            (self.has_gdn, self.has_mamba), tuple(sorted(exl)) if exl else None, tuple(sorted(exn)) if exn else None,
            tuple(int(s) for s in params["recurrent_slots"].tolist()) if self.has_conv else None,   # every job's conv slot is baked in
            tuple(r.slot for r in params["recurrent_states"]) if self.has_swa else None,   # per-slot ring block table is baked in
        )

    def _new_entry(self, key, x, params):
        while len(self.entries) >= CUDA_GRAPH_MAX:
            old = next(iter(self.entries))
            self._drop(old, sync = True)
            self.stats["evictions"] = self.stats.get("evictions", 0) + 1
        dev = self.dev
        bt, cs, pos = params["block_table"], params["cache_seqlens"], params.get("positions")
        e = _Entry()
        e.graph = None
        e.out = None
        e.warm = 0
        e.last_used = 0
        e.cache_ref = weakref.ref(params["cache"])
        e.fp = _cache_fp(params["cache"])
        e.slots = e.slots_src = e.exports = e.swa_cs = e.swa_host = None
        with torch.cuda.device(dev):
            e.x_in = torch.empty(x.shape, dtype = x.dtype, device = dev)
            e.bt = torch.zeros(bt.shape, dtype = bt.dtype, device = dev)
            e.cs = torch.zeros(cs.shape, dtype = cs.dtype, device = dev)
            e.pos = None if (pos is None or pos is cs) else torch.zeros(pos.shape, dtype = pos.dtype, device = dev)
            if self.has_swa:
                e.swa_cs = torch.zeros(x.shape[0], dtype = torch.int32, device = dev)
                e.swa_host = torch.zeros(x.shape[0], dtype = torch.int32).pin_memory()
            if self.has_gdn or self.has_mamba:
                # Fixed-address slot buffer refreshed only when the live slot
                # tensor object changes (stable across steps for one job, so
                # the capture never bakes a per-step upload address).
                e.slots = torch.zeros(params["recurrent_slots"].shape, dtype = torch.int32, device = dev)
        # Params dict for warmup and capture: static CUDA tensors, nothing host-resident. dev_cache,
        # input_ids and layer_instance are per-call and are rebuilt by the modules
        p = {k: v for k, v in params.items() if k not in ("dev_cache", "input_ids", "layer_instance")}
        p["block_table"] = e.bt
        p["cache_seqlens"] = e.cs
        p["positions"] = e.cs if e.pos is None else e.pos
        if e.slots is not None:
            p["recurrent_slots"] = e.slots
        if e.swa_cs is not None:
            p["swa_cs"] = e.swa_cs
        e.params = p
        self.entries[key] = e
        return e

    def _load(self, e, x, params):
        """Per-step values into the static buffers (stream-ordered on the current stream)."""
        # Multi-job (EXL3_CUDA_GRAPH_MULTI) per-job state, all either keyed or copied here, nothing else is read
        # inside the capture: job order / join / leave -> row order of x, block table, cache_seqlens, positions
        # and the slot tuple change, all copied below (B, Q, table width are in the key, so a count or width
        # change selects/creates another entry); per-job page counts (dflash_table_pages) only feed MLA (see
        # _key); recurrent_states (per-job host position / rewind) are touched only by Model.forward after the
        # replay and by the generator; state tensors are cache-owned with fixed pointers (_cache_fp), the slot
        # ids select the row inside them; BC split config derives from the (keyed) table width, not from the
        # cache lengths; per-job sampling state never enters the forward.
        e.x_in.copy_(x, non_blocking = True)
        e.bt.copy_(params["block_table"], non_blocking = True)
        e.cs.copy_(params["cache_seqlens"], non_blocking = True)
        if e.pos is not None:
            e.pos.copy_(params["positions"], non_blocking = True)
        if e.swa_cs is not None:
            # window shift (rare, eager copies) + in-state seqlens of this step, then into the static buffer
            from ..modules.sliding_attn import swa_graph_prep
            cs = swa_graph_prep(params["cache"], self.swa_mods, params["recurrent_states"], x.shape[1])
            e.swa_host.copy_(torch.tensor(cs, dtype = torch.int32))
            e.swa_cs.copy_(e.swa_host, non_blocking = True)
        if e.slots is not None:
            # Slot ids as data: the live slot tensor is immutable per slot tuple (cached, with a persistent
            # device copy); refresh the graph's fixed-address buffer only when the source object changes.
            # Holding the reference makes the identity test exact (a recycled address cannot alias it)
            live = params["recurrent_slots"]
            if live is not e.slots_src:
                from ..util.tensor import get_for_device
                e.slots.copy_(get_for_device(params, "recurrent_slots", self.dev))
                e.slots_src = live

    def _capture(self, key, e, tail, x, params) -> bool:
        dev = self.dev
        for d in range(torch.cuda.device_count()):
            if torch.cuda.memory_allocated(d) > 0:
                torch.cuda.synchronize(d)
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.device(dev):
                stream = _stream_for(dev.index)
                with torch.cuda.graph(g, pool = _pool_for(dev.index), stream = stream,
                                      capture_error_mode = "thread_local"):
                    assert torch.cuda.is_current_stream_capturing()
                    pc = dict(e.params)
                    out = self._run_tail(tail, e.x_in, pc)
        except Exception as ex:
            self._drop(key)
            self.disabled.add(key)
            self.stats["declines"][f"capture_error:{type(ex).__name__}"] += 1
            self.stats["capture_error_msg"] = repr(ex)[:500]
            if CUDA_GRAPH_FALLBACK == "eager":
                return False   # diagnostic only: captured-region work may have run for real
            raise
        e.graph = g
        e.out = out
        e.exports = list(pc["export_states"]) if pc.get("export_states") else None
        self.stats["captures"] += 1
        if CUDA_GRAPH_DEBUG:
            print(f"[cudagraph] captured key={key[1:5]}", flush = True)
        return True

    def _drop(self, key, sync: bool = False):
        e = self.entries.pop(key, None)
        if e is not None and (sync or e.graph is not None):
            try:
                torch.cuda.synchronize(self.dev)
            except Exception:
                pass
        del e

    def clear(self):
        if self.entries:
            try:
                torch.cuda.synchronize(self.dev)
            except Exception:
                pass
        self.entries.clear()
        self.disabled.clear()
        self._model_ok = None


def maybe_forward(model, ids: torch.Tensor, params: dict):
    """Hook for Model.forward (non-TP branch). Returns the logits, or None to run the eager path."""
    if not CUDA_GRAPH_ENABLED or torch.version.hip or not torch.cuda.is_available():
        return None
    r = model.__dict__.get("_cuda_graph_runner")
    if r is None:
        r = model.__dict__["_cuda_graph_runner"] = CudaGraphRunner(model)
    return r.forward(ids, params)


def purge(model = None):
    """Release graphs (every runner, or one model's). Harness arms / model unload."""
    for ref in list(_registry):
        r = ref()
        if r is None:
            _registry.remove(ref)
        elif model is None or r.model is model:
            r.clear()
    if model is None:
        _pools.clear()
        _streams.clear()
