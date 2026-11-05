"""
Persistent prompt cache on NVMe for the paged EXL3 generators (GLM, Qwen): a conversation resumed after a lane unload, a model
swap or a reboot pays only its new tokens.

What is stored, content-addressed (a page hash is blake2b(page ids, previous hash), so it names the whole prefix up to that page and
is stable across processes):
  segment  the KV pages of one save that were not on the disk yet, with their hashes (a growing conversation writes only its new pages)
  stash    one recurrent-state checkpoint (anchored on a page hash). The deepest checkpoint of the saved chain, plus the ones at 2K, 4K,
           8K ... tokens (a new conversation on the same system prompt resumes there); a deeper stash of the same chain supersedes
           the older non-power-of-two ones
  tail     the prompt-end checkpoint (partial page + recurrent state), so the next turn resumes at the prompt end instead of replaying
           up to EXL3_TAIL_MAXGAP tokens

save(ids) runs with the generator idle: the GPU -> host copy of the new pages is the only synchronous part, the file writes happen on
the DiskStore thread (rate limited, free-space floor, LRU cap). restore(ids) runs with the generator idle, before the request is
enqueued, and only when the disk holds a deeper resume point than the RAM caches; it puts pages / stash / tail back exactly like
SlotStore.restore, so the request is a normal prompt-cache hit.

Exactness: pages, stashes and the tail are the bytes the generator held (torch.save round trip). A disk resume therefore equals the
RAM resume of the same checkpoint bit for bit; versus a fresh prefill it carries the same chunking difference a RAM resume has.
"""
from __future__ import annotations

import hashlib
import os
import time

import torch

from ..util.disk_store import DiskStore, env_dir, namespace
from ..util.attention_mode import ATTENTION_CACHE_MODE
from .pagetable import PAGE_SIZE, _randomhash, tensor_hash_checksum
from .slot_store import SlotStore

FORMAT = "paged1"
_BULK_OFF_FILE = os.environ.get("EXL3_PD_BULK_OFF_FILE", "")      # bench switch: while this file exists the per-page copies run


def bulk_on() -> bool:
    """pinned bulk restore (default on CUDA: 390 -> 295 ms; on ROCm unified memory it costs 329 -> 617 ms); EXL3_PD_BULK=0|1 forces it"""
    return os.environ.get("EXL3_PD_BULK", "0" if torch.version.hip else "1") != "0" and not (_BULK_OFF_FILE and os.path.exists(_BULK_OFF_FILE))


def weights_identity(model_dir: str) -> str:
    """The weights a saved state was computed with: resolved folder + size/mtime of its config and index + EXL3_PREFIX_DISK_TAG."""
    parts = [os.path.realpath(os.path.expanduser(model_dir)), os.environ.get("EXL3_PREFIX_DISK_TAG", "")]
    for n in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        try:
            st = os.stat(os.path.join(os.path.expanduser(model_dir), n))
            parts.append(f"{n}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            pass
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def _hx(h) -> str:
    return "none" if h is None else h.hex()


PIN_PAGES = 32                      # every checkpoint of the first 8K tokens is kept: the system prompt + tools of any session resumes near its end


def tail_page_usable(e: dict, ph, L: int) -> bool:
    """Whether the partial page of an in-memory prompt-end checkpoint still holds its K/V: it is an unreferenced page, another job
    may have taken it. The job refuses such an entry (Job._tail_restore); counting it as resumable here kept the deeper copy on
    the disk from being restored (a follow-up turn re-prefilled 2-4K tokens while the disk held its prompt end)."""
    src = e.get("page")
    if src is None:
        return True
    return src.prev_hash == ph and src.kv_position >= L and torch.equal(src.sequence[:, :L], e["ids"].view(1, -1).to(src.sequence.device))


def _pow2(d: int) -> bool:
    return d >= 8 and d & (d - 1) == 0          # 8 pages = 2048 tokens


class PagedPrefixDisk:

    def __init__(self, slots: SlotStore, root: str | None = None, model_id: str = "", extra: str = ""):
        self.slots = slots
        self.model_id = model_id
        tensors = slots._tensors()
        layout = [(tuple(t.shape[1:]), str(t.dtype)) for t in tensors]
        parts = (FORMAT, model_id, layout, extra)
        # Normal mode retains the legacy namespace, including its argument count.
        if ATTENTION_CACHE_MODE is not None:
            parts += (ATTENTION_CACHE_MODE,)
        self.ns = namespace(*parts)
        self.disk = DiskStore(root or env_dir(), self.ns)
        self.pages: dict[bytes, tuple[str, int]] = {}                 # page hash -> (segment, position in segment)
        self.stashes: dict[bytes, tuple[str, int]] = {}               # anchor hash -> (entry, depth in pages)
        self.tails: dict[str, dict] = {}                              # entry -> idx
        self.pending: list[torch.Tensor] = []
        self._staging = ([None, None], [None, None])               # pinned host buffers + their last device-copy events
        self.stats = {"saves": 0, "save_ms": 0.0, "restores": 0, "restore_ms": 0.0, "restored_tokens": 0}
        self.load_index()

    # ---- index

    def load_index(self) -> int:
        n = 0
        self.pages, self.stashes, self.tails = {}, {}, {}
        for name in self.disk.entries():
            ix = self.disk.read_idx(name)
            if not ix or ix.get("ns") != self.ns:
                self.disk.remove(name)
                continue
            k = ix["k"]
            if k == "seg":
                for pos, h in enumerate(ix["hashes"]):
                    self.pages[h] = (name, pos)
            elif k == "stash":
                self.stashes[ix["anchor"]] = (name, ix["depth"])
            elif k == "tail":
                self.tails[name] = ix
            n += 1
        return n

    def _forget(self, name: str) -> None:
        self.disk.remove(name)
        for h in [h for h, (nm, _) in self.pages.items() if nm == name]:
            del self.pages[h]
        for a in [a for a, (nm, _) in self.stashes.items() if nm == name]:
            del self.stashes[a]
        self.tails.pop(name, None)

    def _have(self, name: str) -> bool:
        return os.path.exists(self.disk._p(name, ".idx"))

    def clear(self) -> None:
        """Forget everything (cache_salt change): the entries were computed under another namespace."""
        self.disk.flush()
        for name in self.disk.entries():
            self.disk.remove(name)
        self.pages, self.stashes, self.tails = {}, {}, {}

    # ---- hashes

    @staticmethod
    def chain(ids: torch.Tensor, max_pages: int | None = None) -> list[bytes]:
        ids = ids.view(1, -1).to("cpu", torch.long)
        n = ids.shape[1] // PAGE_SIZE if max_pages is None else max_pages
        out, prev = [], None
        for i in range(n):
            prev = tensor_hash_checksum(ids[:, i * PAGE_SIZE:(i + 1) * PAGE_SIZE], prev)
            out.append(prev)
        return out

    # ---- save

    @torch.inference_mode()
    def save(self, ids: torch.Tensor) -> dict:
        """Persist the resumable state of `ids` (a prompt that just ran). The generator must be idle."""
        t0 = time.perf_counter()
        sl = self.slots
        g = sl.generator
        rc = g.recurrent_cache
        if rc is not None:
            rc.on_evict = self.spill
        ids = ids.view(1, -1).to("cpu", torch.long)
        hashes, pages = sl._resident_chain(ids)
        if rc is None or not hashes:
            return {"saved": 0}
        rc.settle()
        stash_at = {}                                                  # depth (pages) -> (anchor hash, stash)
        for pi in range(len(hashes)):
            s = rc.get(hashes[pi])
            if s is not None and s["position"] == (pi + 1) * PAGE_SIZE and "tp_handle" not in s:
                stash_at[pi + 1] = (hashes[pi], s)
        deepest = max(stash_at) if stash_at else 0
        tail = sl._find_tail(hashes, ids)
        if tail is not None and (tail[0] < deepest or "tp_handle" in tail[1]["stashed"]):
            tail = None
        keep_depth = max(deepest, 0 if tail is None else tail[0])
        if os.environ.get("EXL3_PD_DEBUG"): print(f"PDDBG save N={ids.shape[1]} pages={len(hashes)} stash_depths={sorted(stash_at)} tail={None if tail is None else tail[0]} keep={keep_depth}", flush=True)
        if keep_depth == 0:
            return {"saved": 0}
        want = {d for d in stash_at if d == deepest or _pow2(d) or d <= PIN_PAGES}
        # 1. pages not on the disk yet
        ok = {}
        def on_disk(h):                                               # indexed AND still there (the LRU may have evicted the file)
            ent = self.pages.get(h)
            if ent is None:
                return False
            if ent[0] not in ok:
                ok[ent[0]] = self._have(ent[0])
            return ok[ent[0]]
        new = [i for i in range(keep_depth) if not on_disk(hashes[i])]
        wrote = 0
        if new:
            src = [pages[i] for i in new]
            torch.cuda.synchronize()
            tens = []
            for t in sl._tensors():
                idx = torch.tensor([p.page_index for p in src], dtype=torch.long, device=t.device)
                tens.append(t.index_select(0, idx).cpu())
            hs = [hashes[i] for i in new]
            name = f"s_{_hx(hs[0])[:20]}_{len(hs)}"
            self.disk.put(name, {"hashes": hs, "sequences": [p.sequence.cpu().clone() for p in src], "tensors": tens},
                          {"ns": self.ns, "k": "seg", "hashes": hs})
            for pos, h in enumerate(hs):
                self.pages[h] = (name, pos)
            wrote += len(hs)
        # 2. stashes (deepest + 2K, 4K, 8K ...), the newest deepest supersedes the older non-pinned ones of this chain
        for d in sorted(want):
            a, s = stash_at[d]
            if a in self.stashes and self._have(self.stashes[a][0]):
                self.disk.touch(self.stashes[a][0])
                continue
            name = f"c_{_hx(a)[:24]}"
            self.disk.put(name, {"stash": s}, {"ns": self.ns, "k": "stash", "anchor": a, "depth": d})
            self.stashes[a] = (name, d)
        chain_set = set(hashes)
        for a, (name, d) in list(self.stashes.items()):
            if a in chain_set and d < deepest and not _pow2(d) and d > PIN_PAGES:
                del self.stashes[a]
                self.disk.remove(name)
        # 3. the prompt-end checkpoint
        if tail is not None:
            k, e = tail
            prev = hashes[k - 1] if k else None
            tid = e["ids"].cpu().clone()
            name = f"t_{_hx(prev)[:16]}_{hashlib.sha1(tid.numpy().tobytes()).hexdigest()[:16]}"
            if name not in self.tails or not self._have(name):
                src = e["page"]
                torch.cuda.synchronize()
                tens = []
                for t in sl._tensors():
                    idx = torch.tensor([src.page_index], dtype=torch.long, device=t.device)
                    tens.append(t.index_select(0, idx).cpu())
                ix = {"ns": self.ns, "k": "tail", "prev": prev, "pages": k, "ids": tid}
                self.disk.put(name, {"ids": tid, "stashed": e["stashed"], "tensors": tens, "draft_ok": e.get("draft_ok", True),
                                     "carry": None if e["carry"] is None else e["carry"].cpu()}, ix)
                self.tails[name] = ix
            for nm, ix in list(self.tails.items()):                      # older prompt-ends of this conversation
                if nm != name and ix["pages"] <= k and (ix["prev"] is None or ix["prev"] in chain_set):
                    kk = ix["pages"]
                    L = ix["ids"].shape[-1]
                    if kk * PAGE_SIZE + L <= k * PAGE_SIZE + tid.shape[-1] and \
                            torch.equal(ids[0, kk * PAGE_SIZE:kk * PAGE_SIZE + L], ix["ids"].view(-1)):
                        self._forget(nm)
        ms = (time.perf_counter() - t0) * 1000
        self.stats["saves"] += 1
        self.stats["save_ms"] += ms
        return {"saved": wrote, "stashes": sorted(want), "tail": tail is not None, "ms": ms}

    # ---- spill (disk-first)

    def spill(self, anchor: bytes, stash: dict) -> None:
        """RecurrentCache eviction hook: a checkpoint leaving host RAM goes to the disk tier (async write) when save() would keep it
        (page aligned, 2K/4K/8K... or inside the pinned first 8K tokens), so a small RAM cap loses no resume point. Deepest and
        prompt-end checkpoints are written by save() at the end of the request."""
        pos = stash.get("position", 0)
        if "tp_handle" in stash or pos <= 0 or pos % PAGE_SIZE:
            return
        d = pos // PAGE_SIZE
        branch = anchor in getattr(self.slots.generator.recurrent_cache, "branch_keys", ())
        if not (_pow2(d) or d <= PIN_PAGES or branch):
            return
        if anchor in self.stashes and self._have(self.stashes[anchor][0]):
            self.disk.touch(self.stashes[anchor][0])
            return
        name = f"c_{_hx(anchor)[:24]}"
        self.disk.put(name, {"stash": stash}, {"ns": self.ns, "k": "stash", "anchor": anchor, "depth": d})
        self.stashes[anchor] = (name, d)

    # ---- restore

    @torch.inference_mode()
    def restore(self, ids: torch.Tensor) -> dict | None:
        """Put the deepest resumable state of `ids` found on the disk back into the caches, when it beats what RAM holds."""
        t0 = time.perf_counter()
        sl = self.slots
        g = sl.generator
        rc = g.recurrent_cache
        pt = g.pagetable
        if rc is not None:
            rc.on_evict = self.spill
        if rc is None or not self.pages:
            return None
        ids = ids.view(1, -1).to("cpu", torch.long)
        N = ids.shape[1]
        hashes = self.chain(ids, (N - 1) // PAGE_SIZE)               # a whole-prompt hit leaves nothing to prefill
        if not hashes:
            return None
        # what RAM can resume from
        ram_depth = 0
        ram_res = 0
        for i, h in enumerate(hashes):
            pg = pt.referenced_pages.get(h) or pt.unreferenced_pages.get(h)
            if pg is None or pg.kv_position != PAGE_SIZE:
                break
            ram_res = i + 1
            s = rc.get(h)
            if s is not None and s["position"] == (i + 1) * PAGE_SIZE:
                ram_depth = i + 1
        ram_pos = ram_depth * PAGE_SIZE
        for e in g.tail_ckpts:
            ph = e["prev_hash"]
            k = 0 if ph is None else (hashes.index(ph) + 1 if ph in hashes else -1)
            if k < 0 or k > ram_res:
                continue    # tail without its resident page chain is unreachable: prefill matches pages from page 0
            L = e["ids"].shape[-1]
            if torch.equal(e["ids"].cpu().view(1, -1), ids[:, k * PAGE_SIZE:k * PAGE_SIZE + L]) and tail_page_usable(e, ph, L):
                ram_pos = max(ram_pos, k * PAGE_SIZE + L)
        # what the disk has: pages are needed from 0, so the usable chain stops at the first page it lacks
        avail = 0
        for h in hashes:
            ent = self.pages.get(h)
            if ent is None or not self._have(ent[0]):
                break
            avail += 1
        disk_depth, disk_anchor = 0, None
        for d in range(avail, 0, -1):
            ent = self.stashes.get(hashes[d - 1])
            if ent is not None and self._have(ent[0]):
                disk_depth, disk_anchor = d, hashes[d - 1]
                break
        tail_ix, tail_name, tail_pos = None, None, 0
        for nm, ix in self.tails.items():
            k = ix["pages"]
            L = ix["ids"].shape[-1]
            if k > avail or disk_depth == 0 or disk_depth > k or not self._have(nm):
                continue
            if ix["prev"] != (hashes[k - 1] if k else None):
                continue
            if k * PAGE_SIZE + L >= N or not torch.equal(ix["ids"].view(1, -1), ids[:, k * PAGE_SIZE:k * PAGE_SIZE + L]):
                continue
            if k * PAGE_SIZE + L > tail_pos:
                tail_ix, tail_name, tail_pos = ix, nm, k * PAGE_SIZE + L
        disk_pos = max(disk_depth * PAGE_SIZE, tail_pos)
        if os.environ.get("EXL3_PD_DEBUG"): print(f"PDDBG restore N={N} ram_pos={ram_pos} ram_depth={ram_depth} avail={avail} disk_depth={disk_depth} tail_pos={tail_pos} disk_pos={disk_pos} stashes={len(self.stashes)} tails={len(self.tails)} rc={len(rc)}", flush=True)
        if disk_pos <= ram_pos:
            return None
        need_pages = max(disk_depth, 0 if tail_ix is None else tail_ix["pages"])
        # load the segments that hold the pages RAM does not have
        want = {}
        for i in range(need_pages):
            h = hashes[i]
            pg = pt.referenced_pages.get(h) or pt.unreferenced_pages.get(h)
            if pg is not None and pg.kv_position == PAGE_SIZE:
                continue
            want.setdefault(self.pages[h][0], []).append((i, h))
        # every page to put back takes an unreferenced page: while a long prompt holds the pool there is none to take (the restore
        # raised IndexError on the empty eviction order and the request prefilled from zero)
        take = sum(len(v) for v in want.values()) + (0 if tail_ix is None else 1)
        if take > pt.num_unreferenced_pages():
            return None
        loaded = {}
        BULK = bulk_on()
        try:
            for nm in want:
                loaded[nm] = self.disk.read(nm, mmap=BULK)
            stash = self.disk.read(self.stashes[disk_anchor][0])["stash"]
            tail_data = self.disk.read(tail_name) if tail_name else None
        except Exception:                                                # noqa: BLE001 torn / evicted under us: drop, no restore
            for nm in list(want) + [self.stashes[disk_anchor][0]] + ([tail_name] if tail_name else []):
                if not self._have(nm) or nm not in loaded and nm in want:
                    self._forget(nm)
            return None
        tensors = sl._tensors()
        for d in loaded.values():
            if len(d["tensors"]) != len(tensors) or any(s.shape[1:] != t.shape[1:] or s.dtype != t.dtype
                                                        for s, t in zip(d["tensors"], tensors)):
                return None
        protected = set(hashes[:need_pages])
        order = None
        copied = 0
        prev = None
        plan = {}
        for i in range(need_pages):
            h = hashes[i]
            live = pt.referenced_pages.get(h) or pt.unreferenced_pages.get(h)
            if live is not None and live.kv_position == PAGE_SIZE:
                prev = h
                continue
            if live is not None:
                print(f"prefix_disk: restore aborted at page {i} kv={live.kv_position} idx={len(self.pages)}", flush=True)
                return None
            nm, pos = self.pages[h]
            d = loaded[nm]
            if order is None:
                order = pt.build_eviction_order(protected)
            while order[0].ref_count:
                order.popleft()
            page = order.popleft()
            pt.evict(page, protected)
            pt.access_serial += 1
            page.add_ref_clear(pt.access_serial, h)
            page.prev_hash = prev
            page.sequence.copy_(d["sequences"][pos])
            page.kv_position = PAGE_SIZE
            if BULK:
                plan.setdefault(nm, ([], []))
                plan[nm][0].append(pos)
                plan[nm][1].append(page.page_index)
            else:
                for s, t in zip(d["tensors"], tensors):
                    t[page.page_index].copy_(s[pos], non_blocking=False)
            page.sub_ref()
            copied += 1
            prev = h
        t_copy = time.perf_counter()
        if plan:
            self._bulk_copy(plan, loaded, tensors)
        copy_ms = (time.perf_counter() - t_copy) * 1000
        if disk_anchor not in rc:
            rc[disk_anchor] = stash
        rc.move_to_end(disk_anchor)
        rc.update_total_size()
        if tail_data is not None and getattr(g, "tail_ckpt_max", 0) > 0:
            sl._restore_tail({"ids": tail_data["ids"], "stashed": tail_data["stashed"], "carry": tail_data["carry"],
                              "draft_ok": tail_data["draft_ok"]},
                             hashes[tail_ix["pages"] - 1] if tail_ix["pages"] else None,
                             {"tensors": tail_data["tensors"]}, [t for t in tensors], order, protected)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        self.stats["restores"] += 1
        self.stats["restore_ms"] += ms
        self.stats["restored_tokens"] += disk_pos
        if os.environ.get("EXL3_PD_DEBUG"): print(f"PDDBG restored pages={copied} bulk={BULK} total_ms={ms:.0f} read_ms={(t_copy - t0) * 1000:.0f} copy_ms={copy_ms:.0f}", flush=True)
        return {"resume_at": disk_pos, "ram_at": ram_pos, "pages": copied, "ms": ms}

    def _bulk_copy(self, plan: dict, loaded: dict, tensors: list) -> None:
        """
        All restored pages in one pass per cache tensor: gather the rows of every segment into a pinned staging buffer (CPU, straight
        from the mmapped file), one host-to-device copy, one index_copy_ into the page pool. Two staging buffers: the gather of
        tensor j+1 overlaps the device copy of tensor j. Same bytes as the per-page copies, so the restored pages are identical.
        """
        dev = tensors[0].device
        order = list(plan)
        src = {nm: torch.tensor(plan[nm][0], dtype=torch.long) for nm in order}
        dst = torch.tensor([i for nm in order for i in plan[nm][1]], dtype=torch.long, device=dev)
        n = int(dst.numel())
        stream = torch.cuda.current_stream(dev)
        bufs, events = self._staging
        for j, t in enumerate(tensors):
            row = t[0].numel() * t.element_size()
            need = n * row
            b = j & 1
            if bufs[b] is None or bufs[b].numel() < need:
                if events[b] is not None:
                    events[b].synchronize()
                bufs[b] = torch.empty(max(need, 1 << 20), dtype=torch.uint8, pin_memory=True)
            if events[b] is not None:
                events[b].synchronize()                              # the device copy that read this buffer last has finished
            stage = bufs[b][:need].view(t.dtype).view((n,) + tuple(t.shape[1:]))
            o = 0
            for nm in order:
                k = src[nm].numel()
                torch.index_select(loaded[nm]["tensors"][j], 0, src[nm], out=stage[o:o + k])
                o += k
            g = stage.to(dev, non_blocking=True)
            t.index_copy_(0, dst, g)
            ev = events[b] = torch.cuda.Event()
            ev.record(stream)
            del g
        if os.environ.get("EXL3_PD_VERIFY"):                             # test aid: every restored page equals its file bytes
            torch.cuda.synchronize()
            for j, t in enumerate(tensors):
                got = t.index_select(0, dst).cpu()
                want = torch.cat([loaded[nm]["tensors"][j].index_select(0, src[nm]) for nm in order])
                assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), f"bulk restore mismatch in tensor {j}"
            print(f"PDVERIFY ok pages={n} tensors={len(tensors)}", flush=True)
