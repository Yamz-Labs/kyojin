import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from ..constants import PAGE_SIZE
from ..util.memory import malloc_trim, set_host_mmap_threshold

# Checkpoint stashes are MB-scale host allocations with LRU (i.e. interleaved) lifetimes —
# exactly the churn glibc retains after free (issue #277). Return memory to the OS once
# enough has been released; per-event cost at this threshold is a few ms. The accumulator
# is per-process, which also gives each tensor-parallel rank its own (their stashes live
# in the child processes)
_TRIM_THRESHOLD = 256 * 1024**2
_freed_bytes = 0

def note_freed(nbytes: int):
    global _freed_bytes
    _freed_bytes += nbytes
    if _freed_bytes >= _TRIM_THRESHOLD:
        _freed_bytes = 0
        malloc_trim()


class RecurrentCache(OrderedDict):
    def __init__(
        self,
        model,
        max_size: int = 4 * 1024**3,
    ):
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.model = model
        # This cache is what churns MB-scale host blocks (see util.memory.set_host_mmap_threshold
        # for why glibc strands them), so fix the allocator policy before the first stash.
        self.mmap_threshold = set_host_mmap_threshold()

        # Optionally set by the Generator; enables stranded-first eviction and staleness metrics
        self.pagetable = None
        # EXL3_NC2_ASYNC_STASH: stashes whose D2H is still in flight (see settle)
        self._pending = []
        self._pin_pool = {}  # (shape, dtype) -> pinned host tensors free for reuse
        # page hash -> exactness tag of the checkpoint stashed under it (Job._stash_exact); outside the stash dict, whose
        # asynchronous finisher thread is still writing to it
        self.exact_meta = OrderedDict()
        self._worker = None
        self.metrics = {
            "stash_evictions": 0,           # checkpoints dropped by LRU pressure
            "stash_evictions_stranded": 0,  # of those, checkpoints that were already unrestorable
            "stash_evictions_live_kv": 0,   # of those, checkpoints whose anchor KV page was still cached
            "stash_pruned": 0,              # stranded checkpoints dropped by prune_stranded()
        }


    def _finish(self, st: dict):
        """Worker thread: wait for the stash's D2H event, replace the pinned copies by pageable clones (the cache
        never retains pinned memory), return the pinned buffers to the pool."""
        st.pop("_ev").synchronize()
        for k, v in st.items():
            if isinstance(v, tuple) and v and all(hasattr(t, "is_pinned") and t.is_pinned() for t in v):
                st[k] = tuple(t.clone() for t in v)
                for t in v:
                    self._pin_pool.setdefault((tuple(t.shape), t.dtype), []).append(t)

    def settle(self):
        """
        EXL3_NC2_ASYNC_STASH: wait until every pending asynchronous stash is finished (_finish). Called before any
        reader sees an entry; normally a no-op since the worker ran during the next chunk.
        """
        pend = self._pending
        if not pend:
            return
        self._pending = []
        for f in pend:
            f.result()

    def async_stash_ok(self, state) -> bool:
        return os.environ.get("EXL3_NC2_ASYNC_STASH", "0") == "1" and not self.model.loaded_tp and \
            getattr(state, "supports_async_stash", False)

    def stash_async(self, state) -> dict:
        """Stash state without a host sync; the copies are pending until settle()."""
        self.settle()
        stashed_state = state.stash(async_ = True, pin_pool = self._pin_pool)
        if "_ev" in stashed_state:
            if self._worker is None:
                self._worker = ThreadPoolExecutor(1, thread_name_prefix = "rc-stash")
            self._pending.append(self._worker.submit(self._finish, stashed_state))
        return stashed_state

    def get(self, key, default = None):
        self.settle()
        return super().get(key, default)

    def __getitem__(self, key):
        self.settle()
        return super().__getitem__(key)

    def items(self):
        self.settle()
        return super().items()

    def values(self):
        self.settle()
        return super().values()

    def get_stashed(self, key, default = None):
        """
        Fetch state from cache and move it to the end of the queue
        """
        self.settle()
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default


    def put(self, key, state, stashed = None):
        """
        Add state to cache. stashed: an already stashed state (same format as state.stash()), e.g. a mid-chunk
        checkpoint; state is ignored then
        """
        if key in self:
            self.move_to_end(key)
        else:
            if stashed is None and self.async_stash_ok(state):
                stashed_state = self.stash_async(state)
            else:
                stashed_state = state.stash() if stashed is None else stashed
            state_size = stashed_state["checkpoint_size"]
            while self.update_total_size() + state_size > self.max_size:
                assert self.current_size >= 0, "Not enough space in cache for single state"
                pt = self.pagetable

                # A checkpoint whose anchor page chain has been broken by KV eviction can never be restored by
                # an allocation, so drop stranded checkpoints (oldest first) before restorable ones. This is a
                # pure win: if the conversation returns, the replay prefill recreates the same checkpoint at no
                # extra cost, since the missing pages force a replay past this position either way.
                popped_key = None
                if pt is not None:
                    for k in self:
                        if not pt.is_resumable(k):
                            popped_key = k
                            break
                if popped_key is not None:
                    popped = self.pop(popped_key)
                    self.metrics["stash_evictions_stranded"] += 1
                else:
                    popped_key, popped = self.popitem(last = False)
                    if pt is not None:
                        page = pt.referenced_pages.get(popped_key) or pt.unreferenced_pages.get(popped_key)
                        if page is not None and page.kv_position == PAGE_SIZE:
                            self.metrics["stash_evictions_live_kv"] += 1

                self.metrics["stash_evictions"] += 1
                self.exact_meta.pop(popped_key, None)
                note_freed(popped["checkpoint_size"])
                if self.model.loaded_tp:
                    self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))

            self[key] = stashed_state
            self.exact_meta.pop(key, None)    # a tag belongs to the stash it was made for (Job._stash_exact sets it after put)
            self.update_total_size()


    def drop(self, key) -> None:
        """Remove one checkpoint (rolling-checkpoint cleanup)."""
        popped = self.pop(key, None)
        self.exact_meta.pop(key, None)
        if popped is None:
            return
        note_freed(popped["checkpoint_size"])
        if self.model.loaded_tp:
            self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))
        self.update_total_size()


    def prune_stranded(self) -> int:
        """
        Drop all checkpoints whose anchor page chain has been broken by KV eviction. A stranded checkpoint can
        never be restored by an allocation, and if its conversation returns, the replay prefill recreates it at
        no extra cost, so this only frees system RAM that would otherwise sit dead until LRU pressure reaches it.
        Intended to be called when the generator goes idle.
        """
        if self.pagetable is None:
            return 0
        stranded = [k for k in self if not self.pagetable.is_resumable(k)]
        for k in stranded:
            popped = self.pop(k)
            self.exact_meta.pop(k, None)
            self.metrics["stash_pruned"] += 1
            note_freed(popped["checkpoint_size"])
            if self.model.loaded_tp:
                self.model.tp_dispatch_all(mp_cache_recurrent_del, (id(self), popped["tp_handle"]))
        if stranded:
            self.update_total_size()
        return len(stranded)


    def update_total_size(self):
        seen = set()
        total = 0
        for v in OrderedDict.values(self):  # sizes only: must not settle in-flight stashes
            if id(v) in seen:
                continue
            seen.add(id(v))
            total += v["checkpoint_size"]
        self.current_size = total
        return total


# Checkpoint handles key the per-rank recurrent_cache dicts and must be unique across all
# recurrent module types (GDN, short-conv, SWA states all stash through the same dict)
_next_checkpoint_handle = 0

def new_checkpoint_handle() -> int:
    global _next_checkpoint_handle
    h = _next_checkpoint_handle
    _next_checkpoint_handle += 1
    return h


# Per-rank functions for tensor-parallel mode

def mp_cache_recurrent_clear(local_context: dict, cache_id: int, slot: int):
    recurrent_modules = local_context["recurrent_modules"]
    for module in recurrent_modules:
        recurrent_layer = module.tp_recurrent_lookup[cache_id]
        recurrent_layer.clear(slot)


def mp_cache_recurrent_stash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int = 0):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = []
    for module in recurrent_modules:
        l = module.tp_recurrent_lookup[cache_id]
        stashed.append(l.stash(slot, position))
    recurrent_cache[cp_handle] = stashed


def mp_cache_recurrent_unstash(local_context: dict, cache_id: int, cp_handle: int, slot: int, position: int = 0):
    recurrent_modules = local_context["recurrent_modules"]
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache[cp_handle]
    for module, s in zip(recurrent_modules, stashed):
        l = module.tp_recurrent_lookup[cache_id]
        l.unstash(slot, s, position)


def _stashed_bytes(obj) -> int:
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_stashed_bytes(o) for o in obj)
    return 0


def mp_cache_recurrent_del(local_context: dict, cache_id: int, cp_handle: int):
    recurrent_cache = local_context["recurrent_cache"]
    stashed = recurrent_cache.pop(cp_handle)
    note_freed(_stashed_bytes(stashed))
