"""
Tiered MoE expert store: keep a hot set of routed experts resident in GPU-visible memory and read
the cold ones from the checkpoint on demand.

Off by default. EXL3_EXPERT_BUDGET_GIB=<float> turns it on; unset leaves BlockSparseMLP on today's
fully resident path (this module is then never imported by the forward).

Design
------
* The kernels address an expert through per-layer pointer tables (MultiLinear.ptrs_*, and the stacked
  gate/up tables of BC_BlockSparseMLP), indexed by the router's expert id. The store never changes
  what a kernel sees: it only decides which slot an expert's bytes live in and patches the table
  entries. Expert ids, slot contents and the launch sequence are identical to the resident run, so
  the output is bitwise identical (the store is a cache, never a numeric change).
* One pool of fixed-size SLOTS per size class (experts of the same byte layout), shared by all layers
  of the class. (layer, expert) -> slot. A layer's whole expert union for one forward (decode row,
  MTP verify rows, prefill chunk) is made resident before its kernels launch, so a miss costs one
  read and is never split across kernel launches.
* Policy: frequency x recency. Each use adds 1 to an exponentially decayed counter (half-life
  EXL3_EXPERT_HALFLIFE steps, a step = one forward of the first registered layer); the victim is the
  slot with the smallest decayed counter, never one of the layer's own current experts. Counters
  survive eviction, and can be seeded from routing counts (EXL3_EXPERT_WARM=<json>, written by
  EXL3_EXPERT_STATS_OUT=<json>).
* Misses are read with O_DIRECT (never mmap, never the page cache) into a fixed pinned staging ring
  (4 KiB aligned reads of the aligned superset of each expert's bytes, EXL3_EXPERT_READERS threads in
  parallel), then copied tensor by tensor into the slot. Byte offsets of every expert tensor are
  recorded at registration from the safetensors headers.
* Prefetch (EXL3_EXPERT_AHEAD=1): the router input of layer L is pushed through the routers of L+1..L+DEPTH; the
  guessed experts are read into staging entries while layer L computes. Reads go through a two-queue pool:
  demand reads always run first, at most EXL3_EXPERT_PF_READERS threads run prefetch reads, a prefetch the forward
  turns out to need is promoted to demand, and one the forward does not need is cancelled if it has not started.
  The staging ring (EXL3_EXPERT_RING entries, pinned) is the whole prefetch RAM budget and is subtracted from the
  auto budget. Prefetch only moves WHEN bytes are read, never which bytes a kernel sees: outputs are identical.
* Experts are registered at BlockSparseMLP.load instead of being loaded: only the small per-expert
  tensor shells (views of one shared dummy slot per class) exist until the pools are allocated
  (end of Model.load_gen, or the first forward).
"""

from __future__ import annotations

import atexit
import collections
import json
import os
import re
import threading
import time
from concurrent.futures import CancelledError

import numpy as np
import torch

ALIGN = 4096           # O_DIRECT logical alignment (offset, length, memory address)
SLOT_ALIGN = 256       # every tensor in a slot starts 256 B aligned, like the loader's arena
MERGE_GAP = 128 << 10  # tensors of one expert closer than this share a read

# order of the nine per-expert tensors, also the row order of the layer table buffer
PROJS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("trellis", "suh", "svh")
NT = len(PROJS) * len(PARTS)
_PART_DTYPE = {"trellis": ("I16", torch.int16), "suh": ("F16", torch.half), "svh": ("F16", torch.half)}


def budget_gib() -> float | None:
    """EXL3_EXPERT_BUDGET_GIB: GiB of slot pools, or "auto" (= whatever device/host memory is free once the rest of
    the model, the KV cache and the staging ring are in place, minus EXL3_EXPERT_RESERVE_GIB; returns inf here and
    ExpertStore.finalize resolves it). None: store off."""
    v = os.environ.get("EXL3_EXPERT_BUDGET_GIB")
    if v is None or v.strip() == "":
        return None
    if v.strip().lower() == "auto":
        return float("inf")
    return float(v)


def enabled() -> bool:
    return budget_gib() is not None


def _ceil(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def get_store(config):
    """The store of this Config (one per checkpoint, shared by the main model and the MTP head)."""
    st = getattr(config, "_expert_store", None)
    if st is None:
        st = ExpertStore(config)
        config._expert_store = st
    return st


class SizeClass:
    """Experts with the same byte layout: one pool of slots."""

    def __init__(self, cid: int, sizes: tuple, shapes: tuple):
        self.cid = cid
        self.sizes = sizes          # NT byte sizes in table order
        self.shapes = shapes        # NT tensor shapes
        self.offs = []              # slot-relative offset of each tensor
        o = 0
        for s in sizes:
            self.offs.append(o)
            o += _ceil(s, SLOT_ALIGN)
        self.slot_bytes = _ceil(o, SLOT_ALIGN)
        self.dummy = None           # (slot_bytes,) uint8 device tensor: the "null slot" every shell points at
        self.layers = []            # LayerTier members
        self.n_experts_max = 0
        self.total_bytes = 0
        self.entry_bytes = 0        # staging bytes one expert needs (max over members)
        # pool (allocated by finalize)
        self.nslots = 0
        self.chunks = []            # uint8 device tensors
        self.spc = 0                # slots per chunk
        self.slot_addr = None       # int64 (nslots,)
        self.owner_layer = None     # int32 (nslots,) -1 free
        self.owner_expert = None    # int32
        self.sval = None            # float64 decayed use counter at time st
        self.pinned = None          # bool (nslots,): never an eviction victim (EXL3_EXPERT_PIN_FRAC)
        self.st = None              # float64 step of last update
        self.dval = None            # float64 (nslots,): decayed decode-use counter at time dst (EXL3_EXPERT_PROTECT)
        self.dst = None
        self.scr_t = None           # float64 (nslots,): load number of a prefill scratch slot, inf = stable slot
        self.free = []              # free slot stack


class LayerTier:
    """One expert GROUP of a BlockSparseMLP: the experts sharing a byte layout (one K). A layer whose experts all
    have the same K is one group; a per-expert-K layer (moe_mixedk) is one group per (K_gate_up, K_down).
    Holds expert byte locations, group-local expert -> slot map and the pointer-table buffer the kernels read."""

    def __init__(self, store, module, cls: SizeClass, lidx: int, files: list, foff: np.ndarray, fid: np.ndarray,
                 ids: np.ndarray | None = None):
        self.store = store
        self.module = module
        self.cls = cls
        self.lidx = lidx
        self.key = module.key
        self.E = len(foff)
        # global (layer-level) expert ids of the group, ascending: local index i is expert ids[i]
        self.ids = np.arange(self.E, dtype = np.int64) if ids is None else np.asarray(ids, dtype = np.int64)
        self.set = None             # TierSet
        self.gidx = 0
        self.files = files                  # file ids index store.files
        self.foff = foff                    # (E, NT) absolute byte offsets
        self.fid = fid                      # (E, NT) file ids
        self.smap = np.full(self.E, -1, dtype = np.int32)
        self.hval = np.zeros(self.E, dtype = np.float64)   # decayed counter of non-resident experts
        self.ht = np.zeros(self.E, dtype = np.float64)
        self.hdv = np.zeros(self.E, dtype = np.float64)    # decayed DECODE-use counter of non-resident experts (EXL3_EXPERT_PROTECT)
        self.hdt = np.zeros(self.E, dtype = np.float64)
        self.counts = np.zeros(self.E, dtype = np.int64)   # cumulative presence counts (stats / warm file)
        self.host = np.zeros(NT * self.E + 6 * self.E, dtype = np.int64)
        self.tab = None                     # device int64 buffer, same layout as host
        self.dirty = False
        self.version = 0                    # bumped on every table upload (consumers holding copies refresh on change)
        self.live = True
        self.v_ptrs = None

    # -- tables ---------------------------------------------------------------------------------------
    def build_tables(self, device):
        """Device buffer holding the nine per-projection tables and the three stacked gate/up tables.
        Everything points at the class's dummy slot until an expert is made resident."""
        E = self.E
        base = self.cls.dummy.data_ptr()
        for t in range(NT):
            self.host[t * E:(t + 1) * E] = base + self.cls.offs[t]
        self._fill_gu(np.arange(E))
        self.tab = torch.from_numpy(self.host.copy()).to(device)
        self.dirty = False

    def view(self, t: int) -> torch.Tensor:
        E = self.E
        return self.tab[t * E:(t + 1) * E]

    def gu_view(self, part: int) -> torch.Tensor:
        """(E, 2) table, column 0 = gate, 1 = up. part: 0 trellis, 1 suh, 2 svh"""
        E = self.E
        o = NT * E + 2 * E * part
        return self.tab[o:o + 2 * E].view(E, 2)

    def _fill_gu(self, es: np.ndarray):
        E = self.E
        for part in range(3):
            o = NT * E + 2 * E * part
            g = self.host[(0 * 3 + part) * E:(0 * 3 + part + 1) * E]
            u = self.host[(1 * 3 + part) * E:(1 * 3 + part + 1) * E]
            self.host[o + 2 * es] = g[es]
            self.host[o + 2 * es + 1] = u[es]

    def set_expert_ptrs(self, e: int, addr: int | None):
        """Point expert e at slot memory at `addr` (None: the dummy slot)."""
        E = self.E
        base = self.cls.dummy.data_ptr() if addr is None else addr
        for t in range(NT):
            self.host[t * E + e] = base + self.cls.offs[t]
        for part in range(3):
            o = NT * E + 2 * E * part
            self.host[o + 2 * e] = self.host[(0 * 3 + part) * E + e]
            self.host[o + 2 * e + 1] = self.host[(1 * 3 + part) * E + e]
        self.dirty = True

    def flush(self):
        if self.dirty:
            self.tab.copy_(torch.from_numpy(self.host.copy()))
            self.dirty = False
            self.version += 1


class TierSet:
    """Per BlockSparseMLP (what `module.tier` holds): the layer's expert groups. `kmap[e]` is the group of expert e,
    `lslot[e]` its index inside the group (the group's pointer-table position, the same numbering moe_mixedk uses:
    groups ascending by (K_gate_up, K_down), experts ascending inside a group). A single-group set forwards the
    LayerTier attributes the uniform-K code uses (view, gu_view, smap, tab, ...)."""

    def __init__(self, store, module, sidx: int, groups: list, kmap: np.ndarray, lslot: np.ndarray):
        self.store = store
        self.module = module
        self.key = module.key
        self.sidx = sidx
        self.groups = groups
        self.kmap = kmap
        self.lslot = lslot
        self.E = module.num_experts
        self.next = None
        self.live = True
        self.counts = np.zeros(self.E, dtype = np.int64)    # cumulative presence counts of the layer (stats / warm file)
        for gi, g in enumerate(groups):
            g.set = self
            g.gidx = gi

    @property
    def single(self) -> bool:
        return len(self.groups) == 1

    def __getattr__(self, name):
        # only reached for names the set does not define: single-group sets behave as their one LayerTier
        if name in ("cls", "smap", "tab", "dirty", "lidx", "view", "gu_view", "flush", "hval", "ht", "foff", "fid"):
            g = self.__dict__.get("groups")
            if g is not None and len(g) == 1:
                return getattr(g[0], name)
            raise AttributeError(f"{name}: a {len(g or [])}-group tier set has no single {name}")
        raise AttributeError(name)

    def build_tables(self, device):
        for g in self.groups:
            g.build_tables(device)

    def flush(self):
        for g in self.groups:
            g.flush()

    @property
    def mixed(self) -> bool:
        return len(self.groups) > 1

    def bind_mixed(self, mixed_k):
        """Point the nine tables of every moe_mixedk KGroup at the matching group's table buffer."""
        mixed_k.volatile_tables = True      # slot pointers move: the unified prefill must not cache a concatenated copy
        if len(mixed_k.groups) != len(self.groups):
            raise RuntimeError(f"{self.key}: {len(mixed_k.groups)} K groups, tier has {len(self.groups)}")
        for kg, g in zip(mixed_k.groups, self.groups):
            if list(kg.experts) != g.ids.tolist():
                raise RuntimeError(f"{self.key}: K group {kg.index} experts differ from the tier group")
            g.build_tables(self.store.device)
            for i, m in enumerate((kg.gate, kg.up, kg.down)):
                m.ptrs_trellis, m.ptrs_suh, m.ptrs_svh = g.view(i * 3), g.view(i * 3 + 1), g.view(i * 3 + 2)


class _Fut:
    """Result handle of a task of the IO pool. state: 0 queued, 1 running, 2 done, 3 cancelled."""
    __slots__ = ("fn", "args", "prio", "state", "res", "exc", "ev", "pool")

    def __init__(self, pool, fn, args, prio):
        self.pool, self.fn, self.args, self.prio = pool, fn, args, prio
        self.state, self.res, self.exc = 0, None, None
        self.ev = threading.Event()

    def done(self) -> bool:
        return self.state >= 2

    def result(self):
        self.ev.wait()
        if self.state == 3:
            raise CancelledError()
        if self.exc is not None:
            raise self.exc
        return self.res

    def cancel(self) -> bool:
        """Cancel a task that has not started. True when the task will never run (also when already cancelled)."""
        with self.pool.cv:
            if self.state == 0:
                self.state = 3
                self.ev.set()
            return self.state == 3

    def promote(self):
        """A queued prefetch task becomes a demand task (a forward now waits for it)."""
        with self.pool.cv:
            if self.state == 0 and self.prio:
                self.prio = 0
                self.pool.dq.append(self)
                self.pool.cv.notify()


class PrioPool:
    """Reader threads with two queues. Demand tasks (prio 0) always run before queued prefetch tasks (prio 1), and at
    most `pf_max` threads run prefetch tasks at once, so a demand read never waits behind more than one read.
    Queued prefetch tasks can be cancelled; a queued prefetch task can be promoted to demand."""

    def __init__(self, nthreads: int, pf_max: int, name: str = "expert-read"):
        self.cv = threading.Condition()
        self.dq = collections.deque()
        self.pq = collections.deque()
        self.pf_max = max(1, min(pf_max, nthreads))
        self.pf_running = 0
        self.closed = False
        self.threads = [threading.Thread(target = self._run, name = f"{name}-{i}", daemon = True) for i in range(nthreads)]
        for t in self.threads:
            t.start()

    def submit(self, fn, *args, prio: int = 0) -> _Fut:
        f = _Fut(self, fn, args, prio)
        with self.cv:
            (self.pq if prio else self.dq).append(f)
            self.cv.notify()
        return f

    def _next(self):
        while self.dq:
            f = self.dq.popleft()
            if f.state == 0:
                return f
        if self.pf_running < self.pf_max:
            while self.pq:
                f = self.pq.popleft()
                if f.state == 0:
                    return f
        return None

    def _run(self):
        while True:
            with self.cv:
                f = self._next()
                while f is None:
                    if self.closed:
                        return
                    self.cv.wait()
                    f = self._next()
                f.state = 1
                pf = bool(f.prio)
                if pf:
                    self.pf_running += 1
            try:
                f.res = f.fn(*f.args)
            except BaseException as e:      # noqa: BLE001
                f.exc = e
            with self.cv:
                f.state = 2
                if pf:
                    self.pf_running -= 1
                    self.cv.notify()        # a prefetch slot is free again
            f.ev.set()

    def shutdown(self, wait: bool = True):
        with self.cv:
            self.closed = True
            for q in (self.dq, self.pq):
                for f in q:
                    if f.state == 0:
                        f.state = 3
                        f.ev.set()
                q.clear()
            self.cv.notify_all()
        if wait:
            for t in self.threads:
                t.join()


class _Item:
    """One expert read in flight (or done) in a staging entry"""
    __slots__ = ("entry", "eb", "futs", "nbytes", "copies", "depth")

    def __init__(self, entry, eb, futs, nbytes, copies, depth = 0):
        self.entry, self.eb, self.futs, self.nbytes, self.copies, self.depth = entry, eb, futs, nbytes, copies, depth


class ExpertStore:

    def __init__(self, config):
        self.config = config
        bg = budget_gib()
        self.auto_budget = bg == float("inf")
        self.budget = 0 if self.auto_budget else int(bg * (1 << 30))
        self.reserve_gib = float(os.environ.get("EXL3_EXPERT_RESERVE_GIB", "6"))
        self.classes: dict = {}
        self.layers: list[LayerTier] = []       # every expert group of every layer (index = LayerTier.lidx)
        self.sets: list[TierSet] = []           # one per tiered BlockSparseMLP
        self.files: list[str] = []
        self.file_ids: dict = {}
        self.fds: dict = {}
        self.step = 0
        self.halflife = float(os.environ.get("EXL3_EXPERT_HALFLIFE", "256"))
        self.nreaders = int(os.environ.get("EXL3_EXPERT_READERS", "8"))
        self.ring_n = int(os.environ.get("EXL3_EXPERT_RING", "320"))
        self.chunk = max(ALIGN, int(os.environ.get("EXL3_EXPERT_CHUNK_KIB", "512")) * 1024 // ALIGN * ALIGN)
        self.pin_frac = float(os.environ.get("EXL3_EXPERT_PIN_FRAC", "0"))
        self.reserve = int(os.environ.get("EXL3_EXPERT_RESERVE", "16"))   # staging entries prefetch must leave to demand reads
        self.ahead = os.environ.get("EXL3_EXPERT_AHEAD", "0") != "0"      # router-ahead prefetch of the next layer
        self.ahead_rows = int(os.environ.get("EXL3_EXPERT_AHEAD_ROWS", "16"))
        # decode lookahead depth: layers L+1..L+depth are guessed from layer L's router input
        self.ahead_depth = max(1, int(os.environ.get("EXL3_EXPERT_AHEAD_DEPTH", "1")))
        # experts guessed per row for the target layer at depth d (1-based; the last value covers deeper targets),
        # comma list, each >= the router's top-k (default: exactly top-k). Wider guesses trade wasted reads for recall;
        # the extra experts are read after every top-k guess of every depth.
        # issue the guesses before this layer's own misses are served (1, default) or after (0: the first version)
        self.ahead_early = os.environ.get("EXL3_EXPERT_AHEAD_EARLY", "1") != "0"
        self.ahead_k = [max(0, int(x)) for x in os.environ.get("EXL3_EXPERT_AHEAD_K", "0").split(",") if x.strip()] or [0]
        # reader threads that may run prefetch reads at once (the others only take demand reads)
        self.pf_readers = max(1, int(os.environ.get("EXL3_EXPERT_PF_READERS", str(max(1, self.nreaders - 2)))))
        # use weight of a prefill-sized call (>= ahead_all_rows rows) in the decayed counter; < 1 = scan resistant
        self.pf_weight = float(os.environ.get("EXL3_EXPERT_PF_WEIGHT", "1.0"))
        # prefill chunks (>= this many rows) touch nearly every expert of the next layer: prefetch the
        # non-resident ones (up to ahead_all_max reads, the staging ring is the limit) while this layer computes
        self.ahead_all_rows = int(os.environ.get("EXL3_EXPERT_AHEAD_ALL_ROWS", "64"))
        self.ahead_all_max = int(os.environ.get("EXL3_EXPERT_AHEAD_ALL_MAX", "1000000"))
        # prefill streaming (calls of >= ahead_all_rows rows): read the non-resident experts of the layer about to run and of
        # the next `pf_depth` layers from the start of the layer in front (no router needed), and send the experts a prefill
        # call has to load into a scratch ring of slots so that the resident (decode / warm) set is not flushed by the sweep
        self.pf_ahead = os.environ.get("EXL3_EXPERT_PF_AHEAD", "1") != "0"
        self.pf_depth = max(0, int(os.environ.get("EXL3_EXPERT_PF_DEPTH", "1")))
        self.pf_scratch = os.environ.get("EXL3_EXPERT_PF_SCRATCH", "1") != "0"
        # top-up (default on): after a prefill layer's own misses are served the staging entries are free again; issue the
        # reads of the next layer's experts that did not fit in the ring at the layer start, so they overlap this layer's compute
        self.pf_topup = os.environ.get("EXL3_EXPERT_PF_TOPUP", "1") != "0"
        # chunked prefill: the model sets this while another chunk follows; the layer after the last one is then layer 0 again
        self.pf_wrap = False
        self.pf_wrap_on = os.environ.get("EXL3_EXPERT_PF_WRAP", "1") != "0"
        # prefill: issue the first layers' reads at the start of the forward call (before engram / embedding), deeper than pf_depth
        self.pf_upfront = os.environ.get("EXL3_EXPERT_PF_UPFRONT", "1") != "0"
        self.pf_upfront_depth = max(0, int(os.environ.get("EXL3_EXPERT_PF_UPFRONT_DEPTH", "2")))
        # scan-resistant admission (EXL3_EXPERT_PROTECT=1; off = the policy above unchanged). A prefill-sized call (>= ahead_all_rows rows)
        # admits its misses into a bounded STREAM region (EXL3_EXPERT_STREAM_FRAC of the class slots, never below one layer of the
        # class) and evicts only stream entries first; a decode call that hits a stream entry promotes it. Decode victims are ranked by the
        # decayed DECODE use (half life EXL3_EXPERT_HOT_HALFLIFE steps, prefill calls never touch it; seeded from the warm counts).
        self.protect = os.environ.get("EXL3_EXPERT_PROTECT", "0") != "0"
        self.stream_frac = float(os.environ.get("EXL3_EXPERT_STREAM_FRAC", "0.04"))
        self.hot_halflife = float(os.environ.get("EXL3_EXPERT_HOT_HALFLIFE", "128"))
        self.hot_tie = float(os.environ.get("EXL3_EXPERT_HOT_TIE", "0.01"))     # weight of the overall counter in the victim rank
        # prompt-tail promotion (needs PROTECT): the experts picked by the last N rows of a prefill-sized call predict the decode that
        # follows the prompt; they are made stable (not stream) with decode rank `boost`, so the next layers' sweep does not recycle them
        self.tail_rows = int(os.environ.get("EXL3_EXPERT_TAIL_ROWS", "0"))
        self.tail_boost = float(os.environ.get("EXL3_EXPERT_TAIL_BOOST", "1.0"))
        self.trace_tail = {}
        self.scr_clock = 0
        self.free_entries = collections.deque()
        self.staged = {}
        self.draining = []
        self._ahead_pending = {}     # target layer sidx -> (depth, predicted ids) for the current step
        self.verbose = os.environ.get("EXL3_EXPERT_VERBOSE", "0") != "0"
        self.pool = None
        self.stage = None
        self.stage_np = None
        self.stage_base = 0
        self.entry_bytes = 0
        self.events = []
        self.device = None
        self.is_cuda = False
        self.finalized_classes = set()
        self.warm = None
        wp = os.environ.get("EXL3_EXPERT_WARM")
        if wp:
            with open(wp) as f:
                self.warm = self._norm_warm(json.load(f))
        self.stats_out = os.environ.get("EXL3_EXPERT_STATS_OUT")
        # EXL3_EXPERT_TRACE_OUT=<npz>: every layer call's unique expert set (+ rows), for offline policy simulation
        self.trace_out = os.environ.get("EXL3_EXPERT_TRACE_OUT")
        self.trace = []
        self.trace_marks = []      # len(trace) at each reset_stats(): phase boundaries (prefill / decode)
        self.reset_stats()
        if self.stats_out or self.trace_out or os.environ.get("EXL3_EXPERT_STATS", "0") != "0":
            atexit.register(self._atexit)

    # -- routing counts (warm start / hot set) ----------------------------------------------------------
    _LAYER_PFX = re.compile(r"^(.*?\blayers\.)(\d+)\.")
    _LAYER_RX = re.compile(r"(?:^|\.)(layers|mtp)\.(\d+)(?:\.|$)")

    @classmethod
    def _norm_warm(cls, raw: dict) -> dict:
        """Accepts {module key: [count per expert]} (EXL3_EXPERT_STATS_OUT) and the calibration frequency file of
        tools/ds41/plan.py ({"3" | "layers.3": {expert: count}}). Returns {key: list of counts}; lookups go through
        _warm_counts (exact module key first, then the layer number)."""
        out = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                n = max(int(e) for e in v) + 1 if v else 0
                arr = [0.0] * n
                for e, c in v.items():
                    arr[int(e)] = float(c)
                v = arr
            out[str(k)] = list(v)
        return out

    def _warm_counts(self, lt) -> list | None:
        """Counts of a group's layer (padded with zeros to the layer's expert count), or None."""
        if not self.warm:
            return None
        cnt = self.warm.get(lt.key)
        if cnt is None:
            m = self._LAYER_RX.search(lt.key)
            if m:
                n = m.group(2)
                for k in (f"{m.group(1)}.{n}", f"{m.group(1)}.{n}.ffn", f"layers.{n}" if m.group(1) == "layers" else None, n if m.group(1) == "layers" else None):
                    if k is not None and k in self.warm:
                        cnt = self.warm[k]
                        break
        if cnt is None:
            return None
        E = lt.set.E
        if len(cnt) > E:
            return None
        return list(cnt) + [0.0] * (E - len(cnt))

    # -- stats ----------------------------------------------------------------------------------------
    def reset_stats(self):
        self.trace_marks.append(len(self.trace))
        self.st_t0 = time.perf_counter()
        self.st_uses = 0          # expert uses (unique per forward and layer)
        self.st_hits = 0
        self.st_misses = 0
        self.st_bytes = 0         # bytes read from storage (aligned)
        self.st_wait_by_layer = {}
        self.st_wait_s = 0.0      # of which: blocked on reads that were not finished yet
        self.st_read_s = 0.0      # wall time inside miss service
        self.st_ev_s = 0.0        # host blocked on staging-entry copy events
        self.st_prep_s = 0.0      # host time in prepare() after the routing readback (store work: serve, issue, abandon)
        self.st_copy_s = 0.0      # host time spent enqueueing the staging -> slot copies
        self.st_sync_s = 0.0      # wall time of the routing readback (includes the GPU work before it)
        self.st_calls = 0
        self.st_miss_calls = 0
        self.st_ahead_issued = 0
        self.st_ahead_used = 0
        self.st_ahead_wasted = 0
        self.st_ahead_pred = 0       # predicted experts of the next layer
        self.st_ahead_right = 0      # of those, picked by the real router
        self.st_ahead_pred_d = {}    # per lookahead depth
        self.st_ahead_right_d = {}
        self.st_ahead_issued_d = {}  # reads issued / used per lookahead depth (0: prefill-ahead)
        self.st_ahead_used_d = {}
        self.st_ahead_cancelled = 0  # wasted reads cancelled before the NVMe saw them
        self.st_ahead_peak = 0       # most staging entries held by prefetches at once

    def stats(self) -> dict:
        uses = max(self.st_uses, 1)
        return dict(
            uses = self.st_uses, hits = self.st_hits, misses = self.st_misses,
            hit_rate = self.st_hits / uses, bytes = self.st_bytes, read_s = self.st_read_s, wait_s = self.st_wait_s, wait_by_layer = dict(self.st_wait_by_layer),
            sync_s = self.st_sync_s, copy_s = self.st_copy_s, ev_s = self.st_ev_s, prep_s = self.st_prep_s, calls = self.st_calls, miss_calls = self.st_miss_calls,
            gbps = (self.st_bytes / self.st_read_s / 1e9) if self.st_read_s > 0 else 0.0,
            wall_s = time.perf_counter() - self.st_t0,
            gbps_wall = self.st_bytes / max(time.perf_counter() - self.st_t0, 1e-9) / 1e9,
            ahead_issued = self.st_ahead_issued, ahead_used = self.st_ahead_used, ahead_wasted = self.st_ahead_wasted,
            ahead_pred = self.st_ahead_pred, ahead_right = self.st_ahead_right,
            ahead_pred_d = dict(self.st_ahead_pred_d), ahead_right_d = dict(self.st_ahead_right_d),
            ahead_issued_d = dict(self.st_ahead_issued_d), ahead_used_d = dict(self.st_ahead_used_d),
            ahead_cancelled = self.st_ahead_cancelled, ahead_peak = self.st_ahead_peak,
            stage_bytes = self.ring_n * self.entry_bytes,
        )

    def _atexit(self):
        try:
            if os.environ.get("EXL3_EXPERT_STATS", "0") != "0":
                print(" -- expert store stats:", self.stats(), flush = True)
            if self.stats_out:
                self.dump_counts(self.stats_out)
            if self.trace_out:
                self.dump_trace(self.trace_out)
        except Exception as e:                                      # noqa: BLE001
            print(f" !! expert store atexit: {e!r}")

    def dump_trace(self, path: str):
        """npz: step, layer (sidx), rows per call, and the unique expert ids concatenated with offsets."""
        tr = self.trace
        lens = np.array([len(t[3]) for t in tr], dtype = np.int64)
        np.savez(path, step = np.array([t[0] for t in tr], dtype = np.int32),
                 layer = np.array([t[1] for t in tr], dtype = np.int16), rows = np.array([t[2] for t in tr], dtype = np.int32),
                 off = np.concatenate(([0], np.cumsum(lens))),
                 ids = np.concatenate([t[3] for t in tr]).astype(np.int16) if tr else np.zeros(0, np.int16),
                 mark = np.array(self.trace_marks, dtype = np.int64), tail_idx = np.array(sorted(self.trace_tail), dtype = np.int64),
                 tail_rows = np.stack([self.trace_tail[k] for k in sorted(self.trace_tail)]) if self.trace_tail and len({v.shape for v in self.trace_tail.values()}) == 1 else np.zeros(0, np.int16), layout = np.array(json.dumps(self.policy_layout())))

    def policy_layout(self) -> dict:
        """What an offline policy simulation needs: per tier set its expert count, group class ids and (mixed-K) the
        global-id -> (group, local id) maps; per class the slot count and slot bytes."""
        sets = []
        for ts in self.sets:
            d = {"sidx": ts.sidx, "key": ts.key, "E": int(ts.E), "k": int(ts.module.num_experts_per_tok), "cls": [g.cls.cid for g in ts.groups], "lidx": [g.lidx for g in ts.groups]}
            if not ts.single:
                d["kmap"] = [int(x) for x in ts.kmap]
                d["lslot"] = [int(x) for x in ts.lslot]
            else:
                d["ids"] = [int(x) for x in ts.groups[0].ids]
            sets.append(d)
        return {"sets": sets, "classes": {str(c.cid): {"nslots": int(c.nslots), "slot_bytes": int(c.slot_bytes)} for c in self.classes.values()}}

    def dump_counts(self, path: str):
        out = {ts.key: ts.counts.tolist() for ts in self.sets}
        with open(path, "w") as f:
            json.dump(out, f)

    # -- registration (at BlockSparseMLP.load) --------------------------------------------------------
    def _file_id(self, name: str) -> int:
        i = self.file_ids.get(name)
        if i is None:
            i = len(self.files)
            self.files.append(name)
            self.file_ids[name] = i
        return i

    @staticmethod
    def _tensor_loc(stc, key: str):
        """(filename, absolute offset, nbytes, dtype str, shape) of a tensor from the safetensors headers"""
        s = stc.find_stc(key) if hasattr(stc, "find_stc") else stc
        fn = s.tensor_file_map.get(key)
        if fn is None:
            return None
        hdr = s.file_headers[fn]
        h = hdr[key]
        beg, end = h["data_offsets"]
        return fn, hdr["_header_offset"] + beg, end - beg, h["dtype"], h["shape"]

    def probe(self, module, device):
        """Eligibility + byte plan of a BlockSparseMLP, from header metadata only. Returns None
        (with a reason printed) when the module has to load resident as before, else a list of groups
        (dict: ids, foff, fid, sizes, shapes, flags, kkey), ascending by kkey = (K_gate_up, K_down)."""
        def no(why):
            print(f" !! expert store: {module.key} stays resident ({why})")
            return None

        if (not module.gated or module.num_local_experts != module.num_experts or
                module.routing_first is not None or module.tp_mode is not None):
            return no("not a full gated expert layer")
        if len(module.ups) != module.num_experts or len(module.gates) != module.num_experts:
            return no("expert list mismatch")
        stc = module.config.stc
        E = module.num_experts
        foff = np.zeros((E, NT), dtype = np.int64)
        fid = np.zeros((E, NT), dtype = np.int32)
        per = []        # per expert: (sizes, shapes, flags)
        for e in range(E):
            lins = (module.gates[e], module.ups[e], module.downs[e])
            sz, sh = [], []
            flags = []
            for pi, lin in enumerate(lins):
                for k2 in (".su", ".sv", ".bias", ".scale"):
                    if stc.find_stc(lin.key + k2).tensor_file_map.get(lin.key + k2) is not None:
                        return no(f"unsupported tensor {k2}")
                s_ = stc.find_stc(lin.key + ".mcg").tensor_file_map
                s2 = stc.find_stc(lin.key + ".mul1").tensor_file_map
                flags.append((lin.key + ".mcg" in s_, lin.key + ".mul1" in s2))
                for pj, part in enumerate(PARTS):
                    loc = self._tensor_loc(stc, f"{lin.key}.{part}")
                    if loc is None:
                        return no(f"missing {lin.key}.{part}")
                    fn, off, nb, dt, shape = loc
                    if dt != _PART_DTYPE[part][0]:
                        return no(f"{part} dtype {dt}")
                    t = pi * 3 + pj
                    foff[e, t] = off
                    fid[e, t] = self._file_id(fn)
                    sz.append(nb)
                    sh.append(tuple(shape))
            # table order is (gate, up, down) x (trellis, suh, svh) = PROJS x PARTS
            per.append((tuple(sz), tuple(sh), flags))
        # K of a projection = trellis last dim / 16 (the loader's rule); gate and up of an expert share one launch
        def kof(shape):
            return shape[-1] / 16.0
        kk = []
        for sz, sh, flags in per:
            kg, ku, kd = kof(sh[0]), kof(sh[3]), kof(sh[6])
            if kg != ku:
                return no(f"gate K {kg} != up K {ku}")
            kk.append((kg, kd))
        groups = []
        for key in sorted(set(kk)):
            ids = np.array([e for e in range(E) if kk[e] == key], dtype = np.int64)
            sz, sh, fl = per[int(ids[0])]
            for e in ids:
                if per[int(e)] != (sz, sh, fl):
                    return no("experts of one K differ in size or codebook")
            if any(f[0] and f[1] for f in fl):
                return no("both mcg and mul1 present")
            groups.append(dict(ids = ids, foff = foff[ids], fid = fid[ids], sizes = sz, shapes = sh, flags = fl,
                               kkey = key))
        # shape sanity against the Linear's declared dims
        for g in groups:
            e0 = int(g["ids"][0])
            sh = g["shapes"]
            for lin, tsh in ((module.gates[e0], sh[0]), (module.ups[e0], sh[3]), (module.downs[e0], sh[6])):
                if tsh[0] * 16 != lin.in_features or tsh[1] * 16 != lin.out_features:
                    return no("trellis shape does not match the linear")
        return groups

    def register(self, module, device, shells: bool = True):
        """Replace the experts' load: returns the TierSet (shell tensors installed) or None."""
        plan = self.probe(module, device)
        if plan is None:
            return None
        dev = torch.device(device)
        if self.device is None:
            self.device = dev
            self.is_cuda = dev.type == "cuda"
        E = module.num_experts
        kmap = np.zeros(E, dtype = np.int32)
        lslot = np.zeros(E, dtype = np.int32)
        gts = []
        for gi, g in enumerate(plan):
            key = (g["sizes"], g["shapes"])
            cls = self.classes.get(key)
            if cls is None:
                cls = SizeClass(len(self.classes), g["sizes"], g["shapes"])
                self.classes[key] = cls
            if cls.dummy is None:
                cls.dummy = torch.zeros(cls.slot_bytes, dtype = torch.uint8, device = dev)
            lt = LayerTier(self, module, cls, len(self.layers), self.files, g["foff"], g["fid"], g["ids"])
            lt.flags = g["flags"]
            kmap[g["ids"]] = gi
            lslot[g["ids"]] = np.arange(len(g["ids"]), dtype = np.int32)
            cls.layers.append(lt)
            cls.n_experts_max = max(cls.n_experts_max, lt.E)
            cls.total_bytes += lt.E * cls.slot_bytes
            cls.entry_bytes = max(cls.entry_bytes, max(self._plan_reads(lt, e)[2] for e in range(lt.E)))
            self.layers.append(lt)
            gts.append(lt)
        ts = TierSet(self, module, len(self.sets), gts, kmap, lslot)
        m = re.match(r"^(.*?\blayers\.)(\d+)\.", module.key + ".")
        if m and self.sets:
            prev = self.sets[-1]
            pm = re.match(r"^(.*?\blayers\.)(\d+)\.", prev.module.key + ".")
            if pm and pm.group(1) == m.group(1) and int(pm.group(2)) + 1 == int(m.group(2)):
                prev.next = ts
        self.sets.append(ts)
        if not shells:
            return ts
        # shells: every expert Linear gets the shape-correct tensors, all views of its class's dummy slot
        from .quant.exl3 import LinearEXL3
        for lt in gts:
            cls, flag0 = lt.cls, lt.flags
            for e in lt.ids:
                e = int(e)
                for pi, lin in enumerate((module.gates[e], module.ups[e], module.downs[e])):
                    t = [self.view_of(cls, pi * 3 + pj) for pj in range(3)]
                    mcg_t = torch.zeros((), dtype = torch.int32) if flag0[pi][0] else None
                    mul1_t = torch.zeros((), dtype = torch.int32) if flag0[pi][1] else None
                    lin.device = device
                    lin.used_alt_key = False
                    lin.inner = LinearEXL3(
                        module.config, lin.in_features, lin.out_features, None, None, None,
                        t[1], t[2], t[0], mcg_t, mul1_t, None, lin.out_dtype, key = lin.key)
                    lin.quant_type = "exl3"
        return ts

    @staticmethod
    def view_of(cls: SizeClass, t: int) -> torch.Tensor:
        dt = (_PART_DTYPE[PARTS[t % 3]][1])
        o, n = cls.offs[t], cls.sizes[t]
        return cls.dummy[o:o + n].view(dt).view(cls.shapes[t])

    def unregister(self, ts):
        if not ts.live:
            return
        ts.live = False
        for lt in ts.groups:
            lt.live = False
            cls = lt.cls
            if cls.owner_layer is not None:
                mask = cls.owner_layer == lt.lidx
                for sl in np.nonzero(mask)[0]:
                    cls.owner_layer[sl] = -1
                    cls.owner_expert[sl] = -1
                    cls.pinned[sl] = False
                    cls.free.append(int(sl))
            if lt in cls.layers:
                cls.layers.remove(lt)
            lt.smap[:] = -1
        if not any(t.live for t in self.sets):
            self.release()

    def release(self):
        for cls in self.classes.values():
            cls.chunks = []
            cls.dummy = None
        self.classes = {}
        self.layers = []
        self.sets = []
        self.stage = None
        self.stage_np = None
        for fd in self.fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds = {}
        if self.pool is not None:
            self.pool.shutdown(wait = True)
            self.pool = None
        self.finalized_classes = set()
        self.config._expert_store = None

    # -- read plans -----------------------------------------------------------------------------------
    def _plan_reads(self, lt: LayerTier, e: int):
        """Aligned O_DIRECT reads covering the nine tensors of expert e.
        Returns (reads, copies, entry_bytes): reads = [(file_id, aligned_off, aligned_len, stage_rel)],
        copies = [(tensor_idx, stage_rel_of_tensor_bytes)]"""
        cls = lt.cls
        offs = lt.foff[e]
        fids = lt.fid[e]
        order = sorted(range(NT), key = lambda t: (int(fids[t]), int(offs[t])))
        groups = []   # [file, start, end, [tensor idx]]
        for t in order:
            f, o, n = int(fids[t]), int(offs[t]), cls.sizes[t]
            if groups and groups[-1][0] == f and o - groups[-1][2] <= MERGE_GAP:
                groups[-1][2] = max(groups[-1][2], o + n)
                groups[-1][3].append(t)
            else:
                groups.append([f, o, o + n, [t]])
        reads, copies = [], []
        rel = 0
        for f, s, en, ts in groups:
            a0 = s & ~(ALIGN - 1)
            a1 = _ceil(en, ALIGN)
            reads.append((f, a0, a1 - a0, rel))
            for t in ts:
                copies.append((t, rel + int(offs[t]) - a0))
            rel += a1 - a0
        return reads, copies, rel

    # -- pools ----------------------------------------------------------------------------------------
    def finalize(self):
        """Allocate slot pools for every class registered since the last call, plus the staging ring.
        Budget: the remaining bytes split over the new classes in proportion to their expert bytes,
        at least one layer's worth of slots per class (a forward must be able to hold a whole union)."""
        new = [c for c in self.classes.values() if c.cid not in self.finalized_classes and c.dummy is not None]
        if not new:
            return
        used = sum(c.nslots * c.slot_bytes for c in self.classes.values() if c.cid in self.finalized_classes)
        if self.auto_budget:
            self.budget = used + self.auto_budget_bytes()
            print(f" -- expert store: auto budget {self.budget / 2**30:.2f} GiB", flush = True)
        remaining = max(self.budget - used, 0)
        tot = sum(c.total_bytes for c in new)
        dev = self.device
        shares = self._split_budget(new, remaining)
        for c in new:
            share = shares[c.cid]
            cap = sum(l.E for l in c.layers)
            c.nslots = int(min(cap, max(c.n_experts_max, share // c.slot_bytes)))
            c.spc = max(1, (1 << 30) // c.slot_bytes)
            nch = (c.nslots + c.spc - 1) // c.spc
            c.chunks = []
            for i in range(nch):
                n = min(c.spc, c.nslots - i * c.spc)
                c.chunks.append(torch.empty(n * c.slot_bytes, dtype = torch.uint8, device = dev))
            c.slot_addr = np.array(
                [c.chunks[s // c.spc].data_ptr() + (s % c.spc) * c.slot_bytes for s in range(c.nslots)], dtype = np.int64)
            c.owner_layer = np.full(c.nslots, -1, dtype = np.int32)
            c.owner_expert = np.full(c.nslots, -1, dtype = np.int32)
            c.sval = np.zeros(c.nslots, dtype = np.float64)
            c.pinned = np.zeros(c.nslots, dtype = bool)
            c.st = np.zeros(c.nslots, dtype = np.float64)
            c.dval = np.zeros(c.nslots, dtype = np.float64)
            c.dst = np.zeros(c.nslots, dtype = np.float64)
            c.free = list(range(c.nslots - 1, -1, -1))
            self.finalized_classes.add(c.cid)
            print(f" -- expert store: class {c.cid}: {len(c.layers)} layers x {c.n_experts_max} experts, "
                  f"slot {c.slot_bytes / 2**20:.2f} MiB, pool {c.nslots} slots = {c.nslots * c.slot_bytes / 2**30:.2f} GiB "
                  f"of {c.total_bytes / 2**30:.2f} GiB", flush = True)
        need = max(c.entry_bytes for c in self.classes.values())
        if self.stage is None or need > self.entry_bytes:
            if self.stage is not None and self.is_cuda:
                torch.cuda.synchronize()
            self.entry_bytes = _ceil(need, ALIGN)
            self.stage = torch.empty(self.ring_n * self.entry_bytes + ALIGN, dtype = torch.uint8, pin_memory = self.is_cuda)
            addr = self.stage.data_ptr()
            self.stage_base = (-addr) % ALIGN
            self.stage_np = self.stage.numpy()
            self.events = [None] * self.ring_n
            self.free_entries = collections.deque(range(self.ring_n))
            self.staged = {}
            self.draining = []
        if self.pool is None:
            self.pool = PrioPool(self.nreaders, self.pf_readers)
        if self.warm:
            self._warm_fill(new)

    def _split_budget(self, new, remaining: int) -> dict:
        """Bytes of the pool of each new size class. Without routing counts: in proportion to the class's expert
        bytes. With a warm file: greedy by pick probability per byte over every (layer, expert) of every class (the
        hottest bytes win, whatever their K), the leftover spread in proportion to the class's expert bytes."""
        tot = max(sum(c.total_bytes for c in new), 1)
        shares = {c.cid: remaining * c.total_bytes / tot for c in new}
        if not self.warm:
            return shares
        cand = []
        for c in new:
            for lt in c.layers:
                cnt = self._warm_counts(lt)
                if cnt is None or sum(cnt) <= 0:
                    continue
                k, t = lt.module.num_experts_per_tok, float(sum(cnt))
                for e in lt.ids:
                    if cnt[int(e)] > 0:
                        cand.append((min(1.0, cnt[int(e)] * k / t) / c.slot_bytes, c))
        if not cand:
            return shares
        cand.sort(key = lambda x: -x[0])
        take = {c.cid: 0 for c in new}
        left = remaining
        for _, c in cand:
            if c.slot_bytes <= left and take[c.cid] < sum(l.E for l in c.layers):
                take[c.cid] += 1
                left -= c.slot_bytes
        out = {}
        for c in new:
            out[c.cid] = take[c.cid] * c.slot_bytes + left * c.total_bytes / tot
        return out

    def auto_budget_bytes(self) -> int:
        """Bytes the pools may take now: free device memory (and host MemAvailable: on a unified-memory APU both are
        the same RAM) minus the runtime reserve and the pinned staging ring. Everything else (weights, KV cache,
        activations) is already allocated when finalize runs at the end of the load."""
        free = None
        if self.is_cuda:
            try:
                free = torch.cuda.mem_get_info(self.device)[0]
            except Exception:                                       # noqa: BLE001
                free = None
        try:
            with open("/proc/meminfo") as f:
                for ln in f:
                    if ln.startswith("MemAvailable:"):
                        ma = int(ln.split()[1]) * 1024
                        free = ma if free is None else min(free, ma)
                        break
        except OSError:
            pass
        if free is None:
            raise RuntimeError("expert store: EXL3_EXPERT_BUDGET_GIB=auto needs torch.cuda.mem_get_info or /proc/meminfo")
        need = max(c.entry_bytes for c in self.classes.values())
        ring = self.ring_n * _ceil(need, ALIGN)
        cap = float(os.environ.get("EXL3_EXPERT_BUDGET_MAX_GIB", "inf")) * (1 << 30)
        return int(max(min(free - self.reserve_gib * (1 << 30) - ring, cap), 0))

    def _fd(self, fid: int) -> int:
        fd = self.fds.get(fid)
        if fd is None:
            fd = os.open(self.files[fid], os.O_RDONLY | os.O_DIRECT)
            self.fds[fid] = fd
        return fd

    # -- policy ---------------------------------------------------------------------------------------
    def _decay(self, val, t_old, now):
        return val * np.exp2(-(now - t_old) / self.halflife)

    def _alloc_slots(self, cls: SizeClass, n: int, pinned: np.ndarray, scratch: bool = False) -> np.ndarray:
        """n slots of cls: free ones first, then the lowest decayed counters not in `pinned`.
        Evicted owners get their counter written back and their table entry reset to the dummy.
        scratch (prefill sweep): after the free slots, the oldest scratch slots (loaded by an earlier prefill call) go first,
        and the slots taken here become scratch themselves: a sweep over all layers recycles one small ring of slots
        and leaves the rest of the pool (the stable resident set) alone."""
        if self.protect:
            return self._alloc_prot(cls, n, pinned, scratch)
        out = []
        n0 = n
        while n and cls.free:
            out.append(cls.free.pop())
            n -= 1
        nfree = len(out)
        if n and scratch:
            if cls.scr_t is None:
                cls.scr_t = np.full(cls.nslots, np.inf)
            t = cls.scr_t.copy()
            t[cls.owner_layer < 0] = np.inf
            if pinned is not None and len(pinned):
                t[pinned] = np.inf
            m = int(np.isfinite(t).sum())
            k = min(n, m)
            if k:
                vic = np.argpartition(t, k - 1)[:k] if k < len(t) else np.arange(len(t))
                vic = vic[np.argsort(t[vic], kind = "stable")]
                for s in vic:
                    s = int(s)
                    ol, oe = int(cls.owner_layer[s]), int(cls.owner_expert[s])
                    ltv = self.layers[ol]
                    ltv.hval[oe] = cls.sval[s]
                    ltv.ht[oe] = cls.st[s]
                    ltv.smap[oe] = -1
                    ltv.set_expert_ptrs(oe, None)
                    cls.owner_layer[s] = -1
                    cls.owner_expert[s] = -1
                    out.append(s)
                n -= k
            if n and pinned is not None and len(pinned):
                pinned = np.concatenate((pinned, np.array(out, dtype = np.int64)))
            elif n:
                pinned = np.array(out, dtype = np.int64)
        if n:
            now = float(self.step)
            score = self._decay(cls.sval, cls.st, now)
            score[cls.owner_layer < 0] = np.inf
            score[cls.pinned] = np.inf
            if pinned is not None and len(pinned):
                score[pinned] = np.inf
            avail = int(np.isfinite(score).sum())
            if n > avail:
                raise RuntimeError(
                    f"expert store: class {cls.cid} pool of {cls.nslots} slots cannot hold {n} more experts "
                    f"(pinned {0 if pinned is None else len(pinned)}); raise EXL3_EXPERT_BUDGET_GIB")
            vic = np.argpartition(score, n - 1)[:n] if n < len(score) else np.arange(len(score))
            for s in vic:
                s = int(s)
                ol, oe = int(cls.owner_layer[s]), int(cls.owner_expert[s])
                ltv = self.layers[ol]
                ltv.hval[oe] = cls.sval[s]
                ltv.ht[oe] = cls.st[s]
                ltv.smap[oe] = -1
                ltv.set_expert_ptrs(oe, None)
                cls.owner_layer[s] = -1
                cls.owner_expert[s] = -1
                out.append(s)
        res = np.array(out, dtype = np.int64)
        if cls.scr_t is not None:
            cls.scr_t[res] = np.inf         # a slot the normal policy takes is stable again
        if scratch and len(res) > nfree:
            if cls.scr_t is None:
                cls.scr_t = np.full(cls.nslots, np.inf)
            self.scr_clock += 1
            cls.scr_t[res[nfree:]] = self.scr_clock
        return res

    # -- scan-resistant admission (EXL3_EXPERT_PROTECT) -------------------------------------------------
    def _decay_h(self, val, t_old, now):
        return val * np.exp2(-(now - t_old) / self.hot_halflife)

    def _hot_rank(self, cls: SizeClass, now: float) -> np.ndarray:
        """Eviction rank of every slot of cls (lowest goes first): decayed decode use, the overall counter as tie-break."""
        return self._decay_h(cls.dval, cls.dst, now) + self.hot_tie * self._decay(cls.sval, cls.st, now)

    def _evict_slot(self, cls: SizeClass, s: int):
        ol, oe = int(cls.owner_layer[s]), int(cls.owner_expert[s])
        ltv = self.layers[ol]
        ltv.hval[oe] = cls.sval[s]
        ltv.ht[oe] = cls.st[s]
        ltv.hdv[oe] = cls.dval[s]
        ltv.hdt[oe] = cls.dst[s]
        ltv.smap[oe] = -1
        ltv.set_expert_ptrs(oe, None)
        cls.owner_layer[s] = -1
        cls.owner_expert[s] = -1

    def _alloc_prot(self, cls: SizeClass, n: int, hits: np.ndarray, scan: bool) -> np.ndarray:
        """n slots of cls for the misses of one call. Free slots first, then:
        scan (prefill-sized call): the oldest STREAM slots (loaded by an earlier scan call, never reused by a decode call), then
          the lowest-ranked stable slots; every slot taken becomes a stream slot (a sweep over all layers recycles one small ring);
        decode call: stream slots first (probation), then the lowest decode rank.
        Never a slot of `hits` (the experts this call uses) or a pinned one."""
        out = []
        while n and cls.free:
            out.append(cls.free.pop())
            n -= 1
        nfree = len(out)
        if cls.scr_t is None:
            cls.scr_t = np.full(cls.nslots, np.inf)
        if n:
            now = float(self.step)
            valid = cls.owner_layer >= 0
            valid[cls.pinned] = False
            if hits is not None and len(hits):
                valid[hits] = False
            if n > int(valid.sum()):
                raise RuntimeError(
                    f"expert store: class {cls.cid} pool of {cls.nslots} slots cannot hold {n} more experts "
                    f"(pinned {0 if hits is None else len(hits)}); raise EXL3_EXPERT_BUDGET_GIB")
            strm = np.flatnonzero(valid & np.isfinite(cls.scr_t))
            k = min(n, len(strm))
            vic = []
            if k:
                vic.append(strm[np.argsort(cls.scr_t[strm], kind = "stable")[:k]])
            if n > k:
                rest = np.flatnonzero(valid & ~np.isfinite(cls.scr_t))
                r = self._hot_rank(cls, now)[rest]
                m = n - k
                vic.append(rest[np.argpartition(r, m - 1)[:m]] if m < len(rest) else rest)
            for s in np.concatenate(vic):
                self._evict_slot(cls, int(s))
                out.append(int(s))
        res = np.array(out, dtype = np.int64)
        cls.scr_t[res] = np.inf
        if scan and len(res):
            # taken for a scan: stream until a decode call hits it. Free slots join the stream only up to its cap (a cold pool
            # filled by the first prefill must not be all probation); slots taken from the stable set always do (<= one layer union)
            self.scr_clock += 1
            room = max(cls.n_experts_max, int(self.stream_frac * cls.nslots)) - int((np.isfinite(cls.scr_t) & (cls.owner_layer >= 0)).sum())
            cls.scr_t[res[:nfree][:max(room, 0)]] = self.scr_clock
            cls.scr_t[res[nfree:]] = self.scr_clock
        return res

    def _touch_hot(self, cls: SizeClass, hs: np.ndarray, now: float, scan: bool):
        """Hits of one call. Decode: decode use +1 and a stream slot is promoted to stable. Scan: nothing (a prefill sweep neither
        adds decode use nor promotes)."""
        if scan:
            return
        cls.dval[hs] = self._decay_h(cls.dval[hs], cls.dst[hs], now) + 1.0
        cls.dst[hs] = now
        if cls.scr_t is not None:
            cls.scr_t[hs] = np.inf

    def _admit_hot(self, lt: LayerTier, miss: np.ndarray, ns: np.ndarray, now: float, scan: bool):
        cls = lt.cls
        cls.dval[ns] = self._decay_h(lt.hdv[miss], lt.hdt[miss], now) + (0.0 if scan else 1.0)
        cls.dst[ns] = now

    # -- staging ring + reads -------------------------------------------------------------------------
    def _reclaim_done(self):
        keep = []
        for it in self.draining:
            if all(f.done() for f in it.futs):
                self.free_entries.append(it.entry)
            else:
                keep.append(it)
        self.draining = keep

    def _acquire(self, wait: bool):
        """A free staging entry (its previous H2D copy finished), or None."""
        if not self.free_entries:
            self._reclaim_done()
            if not self.free_entries:
                if not wait or not self.draining:
                    return None
                it = self.draining.pop(0)
                tw = time.perf_counter()
                for f in it.futs:
                    f.result()
                self.st_wait_s += time.perf_counter() - tw      # blocked on a read nobody wanted any more (holds a ring entry)
                self.free_entries.append(it.entry)
        ei = self.free_entries.popleft()
        ev = self.events[ei]
        if ev is not None:
            te = time.perf_counter()
            ev.synchronize()
            self.st_ev_s += time.perf_counter() - te            # host waits for the GPU to reach the entry's H2D copies
            self.events[ei] = None
        return ei

    def _submit(self, lt: LayerTier, e: int, ei: int, prio: int = 0, depth: int = 0):
        reads, copies, _ = self._plan_reads(lt, e)
        eb = self.stage_base + ei * self.entry_bytes
        futs = []
        ch = self.chunk
        for fid, a0, alen, rel in reads:
            o = 0
            while o < alen:
                n = min(ch, alen - o)
                futs.append(self.pool.submit(self._read1, fid, a0 + o, n, eb + rel + o, prio = prio))
                o += n
        nb = sum(r[2] for r in reads)
        self.st_bytes += nb     # every issued read, used or not
        return _Item(ei, eb, futs, nb, copies, depth)

    def _read1(self, fid: int, off: int, n: int, dst_abs: int):
        """Worker thread: one aligned O_DIRECT read straight into the pinned staging entry."""
        fd = self._fd(fid)
        dst = memoryview(self.stage_np)[dst_abs: dst_abs + n]
        got = 0
        while got < n:
            r = os.preadv(fd, [dst[got:]], off + got)
            if r <= 0:
                break          # EOF inside the last block: the copies only touch real tensor bytes
            got += r
        return got

    def _publish(self, lt: LayerTier, item, e: int, s: int):
        """Reads of `item` done: copy the expert's tensors into slot s, patch the tables, free the entry."""
        cls = lt.cls
        tw = time.perf_counter()
        for f in item.futs:
            f.result()
        dw = time.perf_counter() - tw
        self.st_wait_s += dw                            # blocked on the NVMe
        self.st_wait_by_layer[lt.set.sidx] = self.st_wait_by_layer.get(lt.set.sidx, 0.0) + dw
        tc = time.perf_counter()
        slot_chunk = cls.chunks[s // cls.spc]
        so = (s % cls.spc) * cls.slot_bytes
        stage = self.stage
        eb = item.eb
        for t, rel in item.copies:
            nbytes = cls.sizes[t]
            d = slot_chunk[so + cls.offs[t]: so + cls.offs[t] + nbytes]
            d.copy_(stage[eb + rel: eb + rel + nbytes], non_blocking = True)
        if self.is_cuda:
            ev = torch.cuda.Event()
            ev.record()
            self.events[item.entry] = ev
        self.st_copy_s += time.perf_counter() - tc
        self.free_entries.append(item.entry)
        lt.set_expert_ptrs(e, int(cls.slot_addr[s]))
        lt.smap[e] = s
        cls.owner_layer[s] = lt.lidx
        cls.owner_expert[s] = e

    def _load(self, lt: LayerTier, experts: np.ndarray, slots: np.ndarray):
        """Make `experts` resident in `slots` (file order): take prefetched reads where there are some, read the
        rest, enqueue the H2D copies on the current stream."""
        order = np.lexsort((lt.foff[experts, 0], lt.fid[experts, 0]))
        experts = experts[order]
        slots = slots[order]
        pending = collections.deque()
        lidx = lt.lidx
        for i in range(len(experts)):
            e = int(experts[i])
            item = self.staged.pop((lidx, e), None)
            if item is not None:
                self.st_ahead_used += 1
                self.st_ahead_used_d[item.depth] = self.st_ahead_used_d.get(item.depth, 0) + 1
                for f in item.futs:
                    f.promote()         # a forward waits for this read now: it leaves the prefetch queue
            else:
                while True:
                    ei = self._acquire(wait = True)
                    if ei is not None:
                        break
                    if pending:
                        it, e0, s0 = pending.popleft()
                        self._publish(lt, it, e0, s0)
                    elif self.staged:
                        k = next(iter(self.staged))
                        self._drop_staged(self.staged.pop(k))
                    else:
                        raise RuntimeError("expert store: no staging entry available")
                item = self._submit(lt, e, ei)
            pending.append((item, e, int(slots[i])))
        while pending:
            it, e0, s0 = pending.popleft()
            self._publish(lt, it, e0, s0)

    def _drop_staged(self, item):
        """A prefetched read nobody used. Reads that have not started are cancelled (the entry is free at once and
        the NVMe never sees them); started ones finish, then the entry is reclaimed."""
        self.st_ahead_wasted += 1
        if all(f.cancel() for f in item.futs):
            self.st_ahead_cancelled += 1
            self.st_bytes -= item.nbytes
            self.free_entries.append(item.entry)
        else:
            self.draining.append(item)

    def _abandon_staged(self, ts: TierSet):
        """Prefetched reads of this layer's groups that the forward did not use: let them finish, then reuse the entries."""
        lids = {g.lidx for g in ts.groups}
        dead = [k for k in self.staged if k[0] in lids]
        for k in dead:
            self._drop_staged(self.staged.pop(k))

    def _prefetch(self, lt: LayerTier, experts: np.ndarray, limit: int | None = None, depth: int = 0):
        """Start O_DIRECT reads (no slot yet) of the predicted-but-not-resident experts of `lt`."""
        experts = np.asarray(experts, dtype = np.int64)
        if not len(experts):
            return
        experts = experts[lt.smap[experts] < 0]
        if not len(experts):
            return
        experts = experts[np.lexsort((lt.foff[experts, 0], lt.fid[experts, 0]))]
        for e in experts:
            e = int(e)
            if (lt.lidx, e) in self.staged:
                continue
            if len(self.free_entries) <= self.reserve:
                self._reclaim_done()
                if len(self.free_entries) <= self.reserve:
                    break
            ei = self._acquire(wait = False)
            if ei is None:
                break
            self.staged[(lt.lidx, e)] = self._submit(lt, e, ei, prio = 1, depth = depth)
            self.st_ahead_issued += 1
            self.st_ahead_issued_d[depth] = self.st_ahead_issued_d.get(depth, 0) + 1
            self.st_ahead_peak = max(self.st_ahead_peak, len(self.staged))
            if limit is not None:
                limit -= 1
                if limit <= 0:
                    break

    # -- warm start -----------------------------------------------------------------------------------
    def _warm_fill(self, classes):
        """Seed the pools from routing counts (EXL3_EXPERT_WARM json: {layer key: counts per expert}). The hottest
        experts of the class, by per-call pick probability, are read in; the top EXL3_EXPERT_PIN_FRAC of the pool
        (never below one layer's worth of free slots) is pinned: never evicted."""
        for cls in classes:
            cand = []
            for lt in cls.layers:
                cnt = self._warm_counts(lt)
                if cnt is None:
                    continue
                tot = float(sum(cnt))
                if tot <= 0:
                    continue
                k = lt.module.num_experts_per_tok
                for i, e in enumerate(lt.ids):
                    c = cnt[int(e)]
                    if c > 0:
                        # per-call pick probability, as the steady-state value of the decayed counter
                        p = min(1.0, c * k / tot)
                        cand.append((p * self.halflife / np.log(2.0), lt.lidx, i))
            cand.sort(reverse = True)
            cand = cand[:cls.nslots]
            npin = 0
            if self.pin_frac > 0:
                npin = int(min(self.pin_frac * cls.nslots, max(cls.nslots - cls.n_experts_max, 0), len(cand)))
            by_layer: dict = {}
            for n, (c, li, e) in enumerate(cand):
                by_layer.setdefault(li, []).append((e, c, n < npin))
            t0 = time.time()
            nb0 = self.st_bytes
            for li, items in by_layer.items():
                lt = self.layers[li]
                es = np.array([e for e, _, _ in items], dtype = np.int64)
                sl = self._alloc_slots(cls, len(es), None)
                self._load(lt, es, sl)
                for e, c, pin in items:
                    sx = int(lt.smap[e])
                    cls.sval[sx] = float(c)
                    cls.st[sx] = 0.0
                    cls.dval[sx] = float(c) * self.hot_halflife / self.halflife      # same steady-state use rate, slower decay
                    cls.dst[sx] = 0.0
                    if pin:
                        cls.pinned[sx] = True
                lt.flush()
            if self.is_cuda:
                torch.cuda.synchronize()
            dt = time.time() - t0
            print(f" -- expert store: warm class {cls.cid}: {len(cand)} experts ({npin} pinned), "
                  f"{(self.st_bytes - nb0) / 2**30:.2f} GiB in {dt:.1f}s", flush = True)
            self.st_bytes = 0

    # -- the per-forward entry ------------------------------------------------------------------------
    def _classes_ready(self, ts) -> bool:
        return all(g.cls.cid in self.finalized_classes for g in ts.groups)

    def _predict(self, ts: TierSet, z: torch.Tensor, nxt = None, k: int = 0):
        """Device tensor (rows, k) of the experts the router of layer `nxt` (default: the next one) would pick for
        this layer's router input (a cheap guess: adjacent layers' inputs are close). None when not applicable."""
        if nxt is None:
            nxt = ts.next
        if nxt is None or not nxt.live or not self._classes_ready(nxt):
            return None
        cfg = getattr(nxt.module, "routing_cfg", None)
        if cfg is None or cfg.gate_tensor is None or z is None:
            return None
        w = cfg.gate_tensor
        if w.dim() != 2 or z.dim() != 2 or w.shape[0] != z.shape[1] or w.shape[1] != nxt.E or z.dtype != w.dtype:
            return None
        logits = (z @ w).float()
        if cfg.router_bias is not None:
            logits = logits + cfg.router_bias.float()
        if cfg.e_score_correction_bias is not None:
            logits = logits + cfg.e_score_correction_bias.float()
        return torch.topk(logits, min(max(k, cfg.num_experts_per_tok), nxt.E), dim = -1).indices

    def _serve_group(self, lt: LayerTier, u: np.ndarray, now: float, w: float = 1.0, scratch: bool = False):
        """Make the group-local experts `u` (unique) resident; returns the slots of the hits (the group's pinned set)."""
        cls = lt.cls
        scan = scratch and self.pf_scratch
        slots = lt.smap[u]
        hit = slots >= 0
        nh = int(hit.sum())
        self.st_uses += len(u)
        self.st_hits += nh
        hs = slots[hit].astype(np.int64)
        if nh:
            cls.sval[hs] = self._decay(cls.sval[hs], cls.st[hs], now) + w
            cls.st[hs] = now
            if self.protect:
                self._touch_hot(cls, hs, now, scan)
        if nh < len(u):
            miss = u[~hit].astype(np.int64)
            self.st_misses += len(miss)
            self.st_miss_calls += 1
            tm = time.perf_counter()
            ns = self._alloc_slots(cls, len(miss), hs, scan)
            self._load(lt, miss, ns)
            for e, sx in zip(miss, ns):
                cls.sval[int(sx)] = self._decay(lt.hval[e], lt.ht[e], now) + w
                cls.st[int(sx)] = now
            if self.protect:
                self._admit_hot(lt, miss, ns, now, scan)
            self.st_read_s += time.perf_counter() - tm
        lt.flush()

    def _split(self, ts: TierSet, ids: np.ndarray):
        """Global expert ids -> [(group, group-local ids)] (ids already unique and in range)."""
        if ts.single:
            return [(ts.groups[0], ids)]
        km = ts.kmap[ids]
        out = []
        for gi, g in enumerate(ts.groups):
            m = km == gi
            if m.any():
                out.append((g, ts.lslot[ids[m]].astype(np.int64)))
        return out

    def prepare(self, ts: TierSet, selected_experts: torch.Tensor, params: dict, z: torch.Tensor | None = None):
        """Make every expert the router picked resident and the layer's tables current. Runs after
        routing and before the layer's expert kernels; host-syncs on the routing result (and, with
        EXL3_EXPERT_AHEAD, reads back the guess for the next layer in the same sync)."""
        if self.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("expert store: CUDA graph capture is not supported on a tiered layer")
        if params.get("autosplit_measure"):
            ts.flush()      # tables still hold the dummy slot: valid memory for a measuring forward
            return
        if not self._classes_ready(ts):
            self.finalize()
        t0 = time.perf_counter()
        flat = selected_experts.reshape(-1)
        rows = selected_experts.shape[0]
        preds = []          # [(depth, target TierSet, device ids)]
        ahead_all = False
        if self.pf_ahead and not self.ahead and self._after(ts) is not None and rows >= self.ahead_all_rows:
            ahead_all = True            # prefill call: streaming of the next layers (prefill_issue starts it earlier)
        if self.ahead and ts.next is not None:
            if rows <= self.ahead_rows:
                tgt = ts.next
                for d in range(1, self.ahead_depth + 1):
                    if tgt is None:
                        break
                    pd = self._predict(ts, z, tgt, self.ahead_k[min(d, len(self.ahead_k)) - 1])
                    if pd is not None:
                        preds.append((d, tgt, pd))
                    tgt = tgt.next
            elif rows >= self.ahead_all_rows and ts.next.live and self._classes_ready(ts.next):
                ahead_all = True        # prefill chunk: next layer will want (nearly) every expert
        if preds:
            both = torch.cat((flat,) + tuple(p[2].reshape(-1).to(flat.dtype) for p in preds)).cpu().numpy()
            sel = both[:flat.numel()]
            psel, o = [], flat.numel()
            for d, tgt, pd in preds:
                psel.append((d, tgt, both[o:o + pd.numel()].reshape(pd.shape)))
                o += pd.numel()
        else:
            sel, psel = flat.cpu().numpy(), None
        t1 = time.perf_counter()
        self.st_sync_s += t1 - t0
        try:
            self._prepare_tail(ts, sel, psel, rows, flat, ahead_all)
        finally:
            self.st_prep_s += time.perf_counter() - t1

    def _prepare_tail(self, ts, sel, psel, rows, flat, ahead_all):
        if ts.sidx == 0:
            self.step += 1
            if self.staged and rows < self.ahead_all_rows:
                self._abandon_staged_all()
            self._ahead_pending = {}
        self.st_calls += 1
        now = float(self.step)
        u = np.unique(sel)
        ok = (u >= 0) & (u < ts.E)
        if not ok.all():
            u = u[ok]
        ts.counts[u] += 1
        tail = None
        if rows >= self.ahead_all_rows and (self.tail_rows or self.trace_out is not None):
            m = sel.reshape(rows, -1)[-max(self.tail_rows, 16):]
            if self.tail_rows:
                tail = np.unique(m[-self.tail_rows:])
                tail = tail[(tail >= 0) & (tail < ts.E)]
            if self.trace_out is not None:
                self.trace_tail[len(self.trace)] = m.astype(np.int16)
        if self.trace_out is not None:
            self.trace.append((self.step, ts.sidx, rows, u.astype(np.int16)))
        for ap in self._ahead_pending.pop(ts.sidx, ()):
            r = int(np.isin(ap[1], u).sum())
            self.st_ahead_right += r
            self.st_ahead_right_d[ap[0]] = self.st_ahead_right_d.get(ap[0], 0) + r
        if self.ahead_early:
            # guesses for the next layers go to the (lower priority) prefetch queue BEFORE this layer's misses are served:
            # demand reads still run first, and the guessed reads overlap the wait for them and the whole layer compute
            self._issue_ahead(ts, psel, ahead_all)
        self._serve_set(ts, u, rows, now, tail)
        if self.staged:
            self._abandon_staged(ts)
        if not self.ahead_early:
            self._issue_ahead(ts, psel, ahead_all)
        elif ahead_all and psel is None and self.pf_topup:
            self._issue_ahead(ts, None, True)

    def _serve_set(self, ts: TierSet, u: np.ndarray, rows: int, now: float, tail: np.ndarray | None = None):
        """Make the unique global experts `u` of one layer call resident (the policy entry: replayed by the offline simulator)."""
        scan = rows >= self.ahead_all_rows
        w = self.pf_weight if scan else 1.0
        for lt, loc in self._split(ts, u):
            lt.counts[loc] += 1
            self._serve_group(lt, loc, now, w, scan)
        if tail is not None and self.protect and scan and len(tail):
            for lt, loc in self._split(ts, tail):
                sl = lt.smap[loc]
                sl = sl[sl >= 0].astype(np.int64)
                cls = lt.cls
                if cls.scr_t is not None:
                    cls.scr_t[sl] = np.inf
                cls.dval[sl] = np.maximum(self._decay_h(cls.dval[sl], cls.dst[sl], now), self.tail_boost)
                cls.dst[sl] = now

    def _after(self, ts: TierSet):
        """The tier set the forward reaches after `ts`: the next layer, or (chunked prefill, pf_wrap) the first one again."""
        if ts.next is not None or not (self.pf_wrap and self.pf_wrap_on) or not self.sets:
            return ts.next
        f = self.sets[0]
        pf, pt = self._LAYER_PFX.match(f.key + "."), self._LAYER_PFX.match(ts.key + ".")
        if f is ts or not pf or not pt or pf.group(1) != pt.group(1):
            return None
        return f

    def _issue_ahead(self, ts: TierSet, psel, ahead_all: bool):
        """Start the guessed reads of the layers after `ts` (see prepare)."""
        if psel is not None:
            extra = []
            for d, tgt, ids in psel:
                pu = np.unique(ids)
                pu = pu[(pu >= 0) & (pu < tgt.E)]
                self.st_ahead_pred += len(pu)
                self.st_ahead_pred_d[d] = self.st_ahead_pred_d.get(d, 0) + len(pu)
                self._ahead_pending.setdefault(tgt.sidx, []).append((d, pu))
                k0 = tgt.module.routing_cfg.num_experts_per_tok
                if ids.shape[1] > k0:
                    # core = in some row's top-k0 guess; read first, nearest depth first. The wider ranks wait.
                    core = np.unique(ids[:, :k0])
                    core = core[(core >= 0) & (core < tgt.E)]
                    rest = np.setdiff1d(pu, core)
                    extra.append((d, tgt, rest, ids[:, k0:]))
                    pu = core
                for lt, loc in self._split(tgt, pu):
                    self._prefetch(lt, loc, depth = d)
            for d, tgt, rest, wide in extra:
                if not len(rest):
                    continue
                # rank-major: the experts of the lowest extra rank (over rows) first
                rk = np.full(tgt.E, 1 << 30, dtype = np.int64)
                for j in range(wide.shape[1] - 1, -1, -1):
                    col = wide[:, j]
                    col = col[(col >= 0) & (col < tgt.E)]
                    rk[col] = j
                order = rest[np.argsort(rk[rest], kind = "stable")]
                for j in np.unique(rk[order]):
                    for lt, loc in self._split(tgt, order[rk[order] == j]):
                        self._prefetch(lt, loc, depth = d)
        elif ahead_all:
            tgt = self._after(ts)
            for _ in range(max(1, self.pf_depth)):
                if tgt is None or not tgt.live or not self._classes_ready(tgt):
                    break
                for lt in tgt.groups:
                    self._prefetch(lt, np.arange(lt.E, dtype = np.int64), limit = self.ahead_all_max)
                tgt = self._after(tgt)

    def prefill_issue(self, ts: TierSet, rows: int, upfront: bool = False):
        """Called by the model before the attention of the layer owning `ts` (prefill call of `rows` rows): start the
        reads of that layer's non-resident experts and of the next `pf_depth` layers, so the NVMe works while the GPU
        runs the attention and the layers in front. No router needed: a prefill call picks nearly every expert."""
        if not self.pf_ahead or rows < self.ahead_all_rows or ts is None:
            return
        if not self._classes_ready(ts):
            self.finalize()
        if upfront and not self.pf_upfront:
            return
        tgt = ts
        for _ in range((self.pf_upfront_depth if upfront else self.pf_depth) + 1):
            if tgt is None or not tgt.live or not self._classes_ready(tgt):
                break
            for lt in tgt.groups:
                self._prefetch(lt, np.arange(lt.E, dtype = np.int64), limit = self.ahead_all_max)
            tgt = self._after(tgt)

    def _abandon_staged_all(self):
        for k in list(self.staged):
            self._drop_staged(self.staged.pop(k))
